//! Coarse carrier-offset estimation from the signal spectrum.
//!
//! WHY THIS EXISTS: the symbol-rate carrier loop (`carrier_recovery`) can only
//! resolve frequency modulo symbol_rate/8 (2 kHz): an offset of exactly 2 kHz
//! rotates 8PSK by one whole constellation step per symbol, which lands on the
//! grid and is indistinguishable, at the constellation level, from the same
//! signal carrying a different (running-sum) bit sequence. Its decision-
//! directed detector and 8th-power frequency-lock aid therefore pull in only
//! +-1 kHz, and a larger offset locks "falsely" to the nearest multiple of
//! 2 kHz and then decodes garbage. Stepping the LO and trusting the loop's
//! lock indicator cannot tell those apart.
//!
//! The spectrum can: a root-raised-cosine signal occupies a known band
//! (`symbol_rate * (1 + rolloff)` wide) centred on its carrier, regardless of
//! the data. Correlating the measured power spectral density against the known
//! (raised-cosine) spectral shape and taking the best-scoring centre gives the
//! carrier offset to well under 1 kHz, which is all the carrier loop then needs
//! to finish the job.
//!
//! The template has constant total weight, so flat noise adds the same amount
//! to the score at every candidate centre - the estimate is not biased by the
//! noise floor, only made less certain by it. `contrast` reports how far the
//! best score stands above the median score (~0 for noise only).

use num_complex::Complex32;

use crate::psd::welch_psd;

pub struct CoarseEstimate {
    /// Estimated signal centre relative to the input's 0 Hz, in Hz.
    pub offset_hz: f64,
    /// (best score - median score) / median score. Noise-only input gives a
    /// small value (a few percent); a real signal gives much more.
    pub contrast: f64,
}

/// Magnitude-squared of the root-raised-cosine response, i.e. the raised
/// cosine: the transmitted power spectral density shape.
fn raised_cosine(f_abs: f64, symbol_rate_hz: f64, rolloff: f64) -> f64 {
    let f1 = (1.0 - rolloff) * symbol_rate_hz / 2.0;
    let f2 = (1.0 + rolloff) * symbol_rate_hz / 2.0;
    if f_abs <= f1 {
        1.0
    } else if f_abs >= f2 {
        0.0
    } else {
        0.5 * (1.0 + (std::f64::consts::PI * (f_abs - f1) / (f2 - f1)).cos())
    }
}

/// Estimate the signal's centre frequency within `+-max_offset_hz` of 0 Hz.
/// `iq` is complex baseband at `sample_rate_hz` (needs at least `fft_size`
/// samples; many more averages better). Returns `None` if the input is too
/// short.
pub fn estimate_offset(
    iq: &[Complex32],
    sample_rate_hz: f64,
    symbol_rate_hz: f64,
    rolloff: f64,
    max_offset_hz: f64,
    fft_size: usize,
) -> Option<CoarseEstimate> {
    if iq.len() < fft_size {
        return None;
    }
    let psd = welch_psd(iq, sample_rate_hz, fft_size, 0.5);
    let bin_hz = sample_rate_hz / fft_size as f64;
    let power: Vec<(f64, f64)> = psd
        .iter()
        .map(|p| (p.freq_hz, 10f64.powf(p.power_db / 10.0)))
        .collect();

    let half_width = (1.0 + rolloff) * symbol_rate_hz / 2.0;
    let n_candidates = (max_offset_hz / bin_hz).floor() as i64;
    let mut scores = Vec::with_capacity(2 * n_candidates as usize + 1);
    for k in -n_candidates..=n_candidates {
        let centre = k as f64 * bin_hz;
        let score: f64 = power
            .iter()
            .filter(|(f, _)| (f - centre).abs() < half_width)
            .map(|&(f, p)| p * raised_cosine((f - centre).abs(), symbol_rate_hz, rolloff))
            .sum();
        scores.push(score);
    }

    let (best_idx, &best) = scores
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())?;
    let mut sorted = scores.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = sorted[sorted.len() / 2].max(1e-30);

    // Parabolic interpolation through the peak and its neighbours.
    let mut frac = 0.0;
    if best_idx > 0 && best_idx + 1 < scores.len() {
        let (l, c, r) = (scores[best_idx - 1], scores[best_idx], scores[best_idx + 1]);
        let denom = l - 2.0 * c + r;
        if denom.abs() > 1e-30 {
            frac = (0.5 * (l - r) / denom).clamp(-1.0, 1.0);
        }
    }
    let offset_hz = (best_idx as i64 - n_candidates) as f64 * bin_hz + frac * bin_hz;
    Some(CoarseEstimate {
        offset_hz,
        contrast: (best - median) / median,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rrc::{PulseShaper, ShapeMode};
    use crate::symbol_map::psk8_point;

    const RS: f64 = 16_000.0;
    const SPS: usize = 8;
    const FS: f64 = RS * SPS as f64;
    const BETA: f64 = 0.35;

    struct Rng(u32);
    impl Rng {
        fn next(&mut self) -> u32 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 17;
            self.0 ^= self.0 << 5;
            self.0
        }
        fn gauss(&mut self) -> f64 {
            let u1 = (f64::from(self.next()) + 1.0) / (f64::from(u32::MAX) + 2.0);
            let u2 = (f64::from(self.next()) + 1.0) / (f64::from(u32::MAX) + 2.0);
            (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
        }
    }

    /// 8PSK/RRC at `offset_hz` plus complex AWGN at the given Es/N0 (None = noise only).
    fn make_signal(
        offset_hz: f64,
        es_n0_db: Option<f64>,
        n_symbols: usize,
        seed: u32,
    ) -> Vec<Complex32> {
        let mut rng = Rng(seed);
        let mut shaper = PulseShaper::new(SPS, BETA, 8, ShapeMode::Rrc);
        let mut shaped = Vec::new();
        for _ in 0..n_symbols {
            shaper.process_symbol(psk8_point((rng.next() >> 5) as u8 & 7), &mut shaped);
        }
        let rms =
            (shaped.iter().map(|c| c.norm_sqr()).sum::<f32>() / shaped.len() as f32).sqrt() as f64;
        // Noise variance per complex sample such that Es/N0 holds in the symbol-rate bandwidth.
        let sigma = match es_n0_db {
            Some(db) => (rms * rms * SPS as f64 / 10f64.powf(db / 10.0) / 2.0).sqrt(),
            None => rms / 2f64.sqrt(),
        };
        shaped
            .iter()
            .enumerate()
            .map(|(k, &s)| {
                let ph = std::f64::consts::TAU * offset_hz * k as f64 / FS;
                let rot = Complex32::new(ph.cos() as f32, ph.sin() as f32);
                let sig = if es_n0_db.is_some() {
                    s * rot
                } else {
                    Complex32::new(0.0, 0.0)
                };
                sig + Complex32::new((sigma * rng.gauss()) as f32, (sigma * rng.gauss()) as f32)
            })
            .collect()
    }

    #[test]
    fn finds_the_offset_across_the_search_range() {
        for (i, offset) in [
            -6_000.0f64,
            -4_100.0,
            -2_000.0,
            -300.0,
            0.0,
            250.0,
            1_999.0,
            3_300.0,
            5_500.0,
            6_000.0,
        ]
        .iter()
        .enumerate()
        {
            for snr in [12.0f64, 20.0] {
                let sig = make_signal(*offset, Some(snr), 4_096, 0x1000 + i as u32);
                let est = estimate_offset(&sig, FS, RS, BETA, 6_500.0, 1024).unwrap();
                assert!(
                    (est.offset_hz - offset).abs() < 150.0,
                    "offset {offset} Hz at {snr} dB: estimated {:.0} Hz",
                    est.offset_hz
                );
            }
        }
    }

    #[test]
    fn contrast_separates_signal_from_noise() {
        let mut noise_max = 0.0f64;
        for seed in 1..=8u32 {
            let noise = make_signal(0.0, None, 4_096, seed * 77);
            noise_max = noise_max.max(
                estimate_offset(&noise, FS, RS, BETA, 6_500.0, 1024)
                    .unwrap()
                    .contrast,
            );
        }
        let mut sig_min = f64::MAX;
        for seed in 1..=8u32 {
            let sig = make_signal(2_500.0, Some(10.0), 4_096, seed * 31);
            sig_min = sig_min.min(
                estimate_offset(&sig, FS, RS, BETA, 6_500.0, 1024)
                    .unwrap()
                    .contrast,
            );
        }
        println!("noise-only max contrast {noise_max:.3}, 10 dB signal min contrast {sig_min:.3}");
        assert!(
            sig_min > 2.0 * noise_max,
            "signal ({sig_min}) not clearly above noise ({noise_max})"
        );
    }
}
