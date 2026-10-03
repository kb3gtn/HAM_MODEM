//! Power spectral density estimation (Welch's method: averaged, windowed,
//! overlapping FFT segments) for offline analysis of captured IQ - a
//! software stand-in for a spectrum analyzer.

use num_complex::Complex32;
use rustfft::FftPlanner;

pub struct PsdPoint {
    pub freq_hz: f64,
    pub power_db: f64,
}

/// Convert bladeRF SC16Q11 samples back to normalized complex floats.
pub fn complex_i16_to_f32(samples: &[bladerf::ComplexI16]) -> Vec<Complex32> {
    samples
        .iter()
        .map(|c| Complex32::new(c.re as f32 / 2048.0, c.im as f32 / 2048.0))
        .collect()
}

/// The complex mean (time-domain average) of a batch of IQ samples - a
/// direct estimate of the DC/carrier-leakage component. A zero-mean
/// modulated signal (e.g. scrambled or near-balanced PRBS data) averages out
/// to ~0 over enough samples, so this isolates the DC term without needing
/// an FFT, and works whether or not real data is being transmitted at the
/// same time.
pub fn complex_mean(samples: &[bladerf::ComplexI16]) -> Complex32 {
    assert!(!samples.is_empty(), "cannot take the mean of zero samples");
    // Accumulate in f64: a running sum over hundreds of thousands of i16
    // samples can reach magnitudes where f32's ~7 significant digits start
    // losing precision in the per-sample contribution.
    let (mut sum_i, mut sum_q) = (0.0f64, 0.0f64);
    for c in samples {
        sum_i += c.re as f64;
        sum_q += c.im as f64;
    }
    let n = samples.len() as f64;
    Complex32::new((sum_i / n / 2048.0) as f32, (sum_q / n / 2048.0) as f32)
}

/// Welch's method PSD estimate.
///
/// `iq`: complex baseband samples. `sample_rate_hz`: the rate they were
/// captured at. `fft_size`: segment length (a power of 2 is recommended for
/// rustfft's fast path, though not required). `overlap`: fraction in [0, 1).
///
/// Returns points ordered by increasing frequency from -sample_rate_hz/2 to
/// +sample_rate_hz/2 (a "centered"/fftshifted layout, matching how a
/// spectrum analyzer displays a baseband capture).
pub fn welch_psd(
    iq: &[Complex32],
    sample_rate_hz: f64,
    fft_size: usize,
    overlap: f64,
) -> Vec<PsdPoint> {
    assert!(fft_size > 0, "fft_size must be positive");
    assert!((0.0..1.0).contains(&overlap), "overlap must be in [0, 1)");
    assert!(
        iq.len() >= fft_size,
        "capture ({} samples) shorter than one FFT segment ({fft_size})",
        iq.len()
    );

    let step = (((fft_size as f64) * (1.0 - overlap)).round() as usize).max(1);

    // Hann window - reduces spectral leakage from segment edges.
    let window: Vec<f32> = (0..fft_size)
        .map(|n| {
            let w =
                0.5 - 0.5 * (2.0 * std::f64::consts::PI * n as f64 / (fft_size as f64 - 1.0)).cos();
            w as f32
        })
        .collect();
    let window_power: f64 = window.iter().map(|w| (*w as f64) * (*w as f64)).sum();

    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(fft_size);

    let mut accum = vec![0.0f64; fft_size];
    let mut n_segments = 0usize;
    let mut start = 0;
    while start + fft_size <= iq.len() {
        let mut buf: Vec<Complex32> = iq[start..start + fft_size]
            .iter()
            .zip(window.iter())
            .map(|(s, w)| s * w)
            .collect();
        fft.process(&mut buf);
        for (i, c) in buf.iter().enumerate() {
            accum[i] += c.norm_sqr() as f64;
        }
        n_segments += 1;
        start += step;
    }

    let scale = 1.0 / (n_segments as f64 * window_power * sample_rate_hz);
    let bin_hz = sample_rate_hz / fft_size as f64;

    (0..fft_size)
        .map(|i| {
            let shifted_idx = (i + fft_size / 2) % fft_size;
            let power_linear = accum[shifted_idx] * scale;
            let power_db = 10.0 * power_linear.max(1e-20).log10();
            let freq_hz = (i as f64 - fft_size as f64 / 2.0) * bin_hz;
            PsdPoint { freq_hz, power_db }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finds_the_peak_of_a_pure_tone_at_the_right_frequency() {
        let sample_rate_hz = 800_000.0;
        let tone_hz = 100_000.0;
        let n = 4096;
        let iq: Vec<Complex32> = (0..n)
            .map(|k| {
                let phase = 2.0 * std::f64::consts::PI * tone_hz * (k as f64) / sample_rate_hz;
                Complex32::new(phase.cos() as f32, phase.sin() as f32)
            })
            .collect();

        let psd = welch_psd(&iq, sample_rate_hz, 1024, 0.5);
        let peak = psd
            .iter()
            .max_by(|a, b| a.power_db.partial_cmp(&b.power_db).unwrap())
            .unwrap();

        let bin_hz = sample_rate_hz / 1024.0;
        assert!(
            (peak.freq_hz - tone_hz).abs() < bin_hz,
            "peak at {} Hz, expected near {tone_hz} Hz",
            peak.freq_hz
        );
    }

    #[test]
    fn complex_mean_isolates_a_constant_offset_from_zero_mean_noise() {
        // A constant DC term plus a zero-mean alternating +-full-scale signal:
        // the mean should recover just the DC term, independent of the data.
        let dc = bladerf::ComplexI16::new(200, -100);
        let samples: Vec<bladerf::ComplexI16> = (0..10_000)
            .map(|i| {
                let wobble = if i % 2 == 0 { 2000 } else { -2000 };
                bladerf::ComplexI16::new(dc.re + wobble, dc.im - wobble)
            })
            .collect();
        let mean = complex_mean(&samples);
        let expected = Complex32::new(dc.re as f32 / 2048.0, dc.im as f32 / 2048.0);
        assert!(
            (mean - expected).norm() < 1e-4,
            "mean {mean:?}, expected {expected:?}"
        );
    }

    #[test]
    fn output_is_ordered_low_to_high_frequency() {
        let iq = vec![Complex32::new(1.0, 0.0); 2048];
        let psd = welch_psd(&iq, 800_000.0, 1024, 0.5);
        for w in psd.windows(2) {
            assert!(w[0].freq_hz < w[1].freq_hz);
        }
    }
}
