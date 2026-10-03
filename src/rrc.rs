//! Root-raised-cosine pulse shaping (TX) and matched filtering (RX): shared
//! tap generation, plus a streaming interpolating FIR filter (zero-stuff by
//! `sps`, then filter) for TX and a plain (non-interpolating) FIR for RX.

use std::collections::VecDeque;
use std::f64::consts::PI;

use num_complex::Complex32;

/// The raw root-raised-cosine impulse response, before any use-specific
/// scaling. `sps`: samples per symbol. `rolloff`: excess bandwidth factor in
/// (0, 1]. `span_symbols`: filter length in symbol periods (total tap count
/// is `span_symbols * sps + 1`).
fn raw_rrc_shape(sps: usize, rolloff: f64, span_symbols: usize) -> Vec<f64> {
    assert!(sps >= 2, "sps must be at least 2");
    assert!(rolloff > 0.0 && rolloff <= 1.0, "rolloff must be in (0, 1]");
    let n_taps = span_symbols * sps + 1;
    let center = (n_taps - 1) as f64 / 2.0;
    let beta = rolloff;

    let mut taps = Vec::with_capacity(n_taps);
    for i in 0..n_taps {
        let t = (i as f64 - center) / sps as f64; // in symbol periods (Ts = 1)
        let h = if t.abs() < 1e-8 {
            1.0 - beta + 4.0 * beta / PI
        } else if (t.abs() - 1.0 / (4.0 * beta)).abs() < 1e-8 {
            (beta / std::f64::consts::SQRT_2)
                * ((1.0 + 2.0 / PI) * (PI / (4.0 * beta)).sin()
                    + (1.0 - 2.0 / PI) * (PI / (4.0 * beta)).cos())
        } else {
            let num =
                (PI * t * (1.0 - beta)).sin() + 4.0 * beta * t * (PI * t * (1.0 + beta)).cos();
            let den = PI * t * (1.0 - (4.0 * beta * t).powi(2));
            num / den
        };
        taps.push(h);
    }
    taps
}

/// TX pulse-shaping taps: scaled by `sps` to compensate for the amplitude
/// loss inherent in zero-stuffing before filtering (interpolation gain) -
/// this is the standard, exact compensation for that specific effect. It
/// does NOT account for ISI overshoot from real (non-DC, non-impulse) data,
/// which depends on the actual bit sequence and can't be derived
/// analytically with a simple multiplier - see `PulseShaper::new`'s
/// empirical calibration for that part. (An earlier version of this
/// function tried to fold a fixed headroom factor in here too; that
/// produced full-scale clipping on real data because the assumption behind
/// the factor was wrong - see project memory for how that was found and fixed.)
pub fn rrc_taps(sps: usize, rolloff: f64, span_symbols: usize) -> Vec<f32> {
    let interpolation_gain = sps as f64;
    raw_rrc_shape(sps, rolloff, span_symbols)
        .into_iter()
        .map(|h| (h * interpolation_gain) as f32)
        .collect()
}

/// RX matched-filter taps: unit-energy normalized (sum of squares = 1), NOT
/// scaled by `sps` - there's no upsampling/zero-stuffing on the RX side to
/// compensate for, so the TX interpolation-gain scaling would be wrong here.
/// Unit energy is the standard matched-filter normalization: it doesn't
/// arbitrarily change the signal's gain or noise power, just correlates
/// against the known pulse shape.
pub fn matched_filter_taps(sps: usize, rolloff: f64, span_symbols: usize) -> Vec<f32> {
    let raw = raw_rrc_shape(sps, rolloff, span_symbols);
    let energy: f64 = raw.iter().map(|h| h * h).sum();
    let norm = energy.sqrt();
    raw.into_iter().map(|h| (h / norm) as f32).collect()
}

/// Direct-form FIR filter with real-valued taps applied to a complex signal
/// (filters I and Q independently with the same coefficients - correct for
/// any real-coefficient filter applied to a complex baseband signal, e.g.
/// the RX matched filter, which must handle nonzero Q once the signal has
/// been frequency-shifted or has any real quadrature content).
struct ComplexFir {
    taps: Vec<f32>,
    delay_line: VecDeque<Complex32>,
}

impl ComplexFir {
    fn new(taps: Vec<f32>) -> Self {
        let n = taps.len();
        ComplexFir {
            taps,
            delay_line: VecDeque::from(vec![Complex32::new(0.0, 0.0); n]),
        }
    }

    fn process(&mut self, x: Complex32) -> Complex32 {
        self.delay_line.push_front(x);
        self.delay_line.pop_back();
        self.delay_line
            .iter()
            .zip(self.taps.iter())
            .fold(Complex32::new(0.0, 0.0), |acc, (x, h)| acc + x * h)
    }
}

/// RX matched filter: correlates the incoming (complex) symbol-domain signal
/// against the known pulse shape. Paired with the TX `PulseShaper` in `Rrc`
/// mode, this forms the matched-filter pair required for Nyquist ISI-free
/// detection at the correct symbol-spaced sampling instants - see the
/// `tx_rx_matched_filter_pair_is_isi_free_at_symbol_spacing` test.
pub struct MatchedFilter {
    fir: ComplexFir,
}

impl MatchedFilter {
    pub fn new(sps: usize, rolloff: f64, span_symbols: usize) -> Self {
        MatchedFilter {
            fir: ComplexFir::new(matched_filter_taps(sps, rolloff, span_symbols)),
        }
    }

    pub fn process(&mut self, x: Complex32) -> Complex32 {
        self.fir.process(x)
    }

    pub fn process_block(&mut self, xs: &[Complex32]) -> Vec<Complex32> {
        xs.iter().map(|&x| self.process(x)).collect()
    }
}

/// Which pulse shape `PulseShaper` produces.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShapeMode {
    /// Root-raised-cosine pulse shaping (the normal mode) - band-limits the
    /// signal, suppressing the sidelobes a rectangular pulse would have.
    Rrc,
    /// Rectangular (zero-order-hold) pulses: hold each symbol flat for `sps`
    /// samples, no filtering at all. Produces the classic sinc-shaped
    /// (sin(x)/x) spectrum of unshaped NRZ data - useful specifically as a
    /// reference/contrast case, since RRC shaping exists to suppress exactly
    /// that spectrum's sidelobes.
    Rectangular,
}

/// Upsamples by `sps` (zero-stuffing) and RRC-filters, one complex symbol in
/// at a time, `sps` shaped complex samples out (the same real taps filter I
/// and Q). Can also produce unshaped rectangular
/// pulses instead - see `ShapeMode`.
pub struct PulseShaper {
    mode: ShapeMode,
    fir: Option<ComplexFir>, // None when mode is Rectangular
    sps: usize,
    rect_scale: f32,
}

/// Target peak amplitude (fraction of full scale) after calibration.
/// 0.8 leaves 20% headroom below the i16 clipping ceiling for whatever the
/// calibration run didn't happen to exercise, without giving up too much
/// dynamic range. Also used directly as the rectangular-mode amplitude,
/// since a held symbol's peak is trivially exactly 1.0 with no ISI overshoot
/// to leave margin for - same headroom convention either way.
const CALIBRATION_TARGET_PEAK: f32 = 0.8;
/// Number of symbols to run through the filter when measuring its real
/// worst-case peak. Needs to be well beyond the filter span so multiple
/// overlapping symbol tails (the actual source of ISI overshoot) get
/// exercised, and long enough that a pseudorandom sequence covers a
/// representative mix of bit patterns.
const CALIBRATION_SYMBOLS: usize = 4000;

impl PulseShaper {
    pub fn new(sps: usize, rolloff: f64, span_symbols: usize, mode: ShapeMode) -> Self {
        match mode {
            ShapeMode::Rrc => {
                let raw_taps = rrc_taps(sps, rolloff, span_symbols);
                let scale = Self::calibrate_scale(&raw_taps, sps);
                let taps: Vec<f32> = raw_taps.into_iter().map(|h| h * scale).collect();
                PulseShaper {
                    mode,
                    fir: Some(ComplexFir::new(taps)),
                    sps,
                    rect_scale: 1.0,
                }
            }
            ShapeMode::Rectangular => PulseShaper {
                mode,
                fir: None,
                sps,
                rect_scale: CALIBRATION_TARGET_PEAK,
            },
        }
    }

    /// Empirically measures the actual peak output magnitude (|I+jQ|, which
    /// is what matters once the TX frequency shift rotates the signal
    /// between I and Q) these taps produce for a long pseudorandom 8PSK
    /// symbol sequence, then returns the scale factor that brings that peak
    /// to `CALIBRATION_TARGET_PEAK`.
    ///
    /// This exists because the real peak depends on how several overlapping
    /// symbol tails (ISI, not eliminated until a matched filter at the RX
    /// end) happen to add up for actual data - a single-impulse or
    /// constant-DC analysis both under-estimate it. Measuring against real
    /// data sidesteps needing that derivation to be right.
    fn calibrate_scale(raw_taps: &[f32], sps: usize) -> f32 {
        let mut fir = ComplexFir::new(raw_taps.to_vec());
        let mut lfsr_state: u32 = 0xACE1_2345; // any nonzero seed
        let mut peak = 0.0f32;
        for _ in 0..CALIBRATION_SYMBOLS {
            // xorshift32 - not cryptographic, just needs to exercise a
            // realistic mix of symbol patterns for this measurement.
            lfsr_state ^= lfsr_state << 13;
            lfsr_state ^= lfsr_state >> 17;
            lfsr_state ^= lfsr_state << 5;
            let symbol =
                Complex32::from_polar(1.0, std::f32::consts::FRAC_PI_4 * (lfsr_state & 7) as f32);
            for k in 0..sps {
                let x = if k == 0 {
                    symbol
                } else {
                    Complex32::new(0.0, 0.0)
                };
                peak = peak.max(fir.process(x).norm());
            }
        }
        if peak > 0.0 {
            CALIBRATION_TARGET_PEAK / peak
        } else {
            1.0
        }
    }

    /// Feed one complex symbol, get back `sps` shaped output samples.
    pub fn process_symbol(&mut self, symbol: Complex32, out: &mut Vec<Complex32>) {
        match self.mode {
            ShapeMode::Rrc => {
                let fir = self.fir.as_mut().expect("Fir present in Rrc mode");
                out.push(fir.process(symbol));
                for _ in 1..self.sps {
                    out.push(fir.process(Complex32::new(0.0, 0.0)));
                }
            }
            ShapeMode::Rectangular => {
                for _ in 0..self.sps {
                    out.push(symbol * self.rect_scale);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn taps_have_expected_length_and_symmetry() {
        let taps = rrc_taps(8, 0.35, 8);
        assert_eq!(taps.len(), 8 * 8 + 1);
        for i in 0..taps.len() {
            assert!(
                (taps[i] - taps[taps.len() - 1 - i]).abs() < 1e-4,
                "tap {i} not symmetric with its mirror"
            );
        }
    }

    #[test]
    fn matched_filter_taps_are_unit_energy() {
        let taps = matched_filter_taps(8, 0.35, 8);
        let energy: f32 = taps.iter().map(|h| h * h).sum();
        assert!(
            (energy - 1.0).abs() < 1e-4,
            "energy {energy}, expected ~1.0"
        );
    }

    /// The critical correctness check for a matched-filter pair: TX RRC
    /// pulse shaping followed by RX RRC matched filtering should satisfy the
    /// Nyquist ISI-free criterion - an isolated symbol impulse, after both
    /// filters, has zero crossings at every symbol-spaced instant except its
    /// own peak. Without this, timing recovery downstream has no clean
    /// sampling instant to lock onto.
    #[test]
    fn tx_rx_matched_filter_pair_is_isi_free_at_symbol_spacing() {
        let sps = 8;
        let rolloff = 0.35;
        let span_symbols = 8;

        let mut shaper = PulseShaper::new(sps, rolloff, span_symbols, ShapeMode::Rrc);
        let mut matched = MatchedFilter::new(sps, rolloff, span_symbols);

        // A long run of zero symbols with a single isolated +1 impulse in
        // the middle, with enough margin on both sides for the combined
        // (TX + RX) filter transient to fully settle.
        let n_symbols = 41;
        let impulse_index = 20;
        let mut shaped = Vec::new();
        for k in 0..n_symbols {
            let symbol = if k == impulse_index {
                Complex32::new(1.0, 0.0)
            } else {
                Complex32::new(0.0, 0.0)
            };
            shaper.process_symbol(symbol, &mut shaped);
        }

        let out = matched.process_block(&shaped);

        // Find the combined filter pair's peak response - the correct
        // symbol-sampling instant - rather than hand-deriving the exact
        // group delay analytically.
        let (peak_idx, peak_val) = out
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.norm().partial_cmp(&b.norm()).unwrap())
            .map(|(i, c)| (i, c.norm()))
            .unwrap();
        assert!(
            peak_val > 0.0,
            "matched filter output should have a nonzero peak"
        );

        // Neighboring symbol-spaced instants (+-1, +-2, +-3 symbols away)
        // should be near zero relative to the peak - the Nyquist ISI-free
        // property.
        for k in [-3i64, -2, -1, 1, 2, 3] {
            let idx = peak_idx as i64 + k * sps as i64;
            if idx < 0 || idx as usize >= out.len() {
                continue; // near the edge, outside this test's margin
            }
            let neighbor = out[idx as usize].norm();
            assert!(
                neighbor < peak_val * 0.05,
                "symbol-spaced neighbor at offset {k} has magnitude {neighbor}, expected < 5% of peak {peak_val} (ISI should be ~0 there)"
            );
        }
    }

    #[test]
    fn peak_is_at_center() {
        let taps = rrc_taps(8, 0.35, 8);
        let center = taps.len() / 2;
        let peak = taps.iter().cloned().fold(f32::MIN, f32::max);
        assert!(
            (taps[center] - peak).abs() < 1e-6,
            "center tap should be the peak"
        );
    }

    #[test]
    fn pulse_shaper_produces_sps_samples_per_symbol() {
        let mut shaper = PulseShaper::new(8, 0.35, 8, ShapeMode::Rrc);
        let mut out = Vec::new();
        shaper.process_symbol(Complex32::new(1.0, 0.0), &mut out);
        assert_eq!(out.len(), 8);
    }

    #[test]
    fn rectangular_mode_holds_the_symbol_flat_for_sps_samples() {
        let mut shaper = PulseShaper::new(8, 0.35, 8, ShapeMode::Rectangular);
        let mut out = Vec::new();
        shaper.process_symbol(Complex32::new(1.0, 0.0), &mut out);
        assert_eq!(out.len(), 8);
        assert!(
            out.iter().all(|&s| (s - out[0]).norm() < 1e-6),
            "rectangular pulse should be flat across the symbol period"
        );
        assert!(
            out[0].re > 0.0 && out[0].norm() <= 1.0,
            "amplitude should be positive and within full scale"
        );

        out.clear();
        shaper.process_symbol(Complex32::new(-1.0, 0.0), &mut out);
        assert!(
            out.iter().all(|&s| s.re < 0.0),
            "opposite symbol should flip sign"
        );
    }

    /// Regression test for a real bug (found in the original BPSK version): an earlier version scaled taps using
    /// `sps * 0.5` as a fixed headroom factor, reasoning from a single
    /// isolated symbol / constant-DC case. On real (pseudorandom) data,
    /// overlapping symbol tails pushed the actual peak past a normalized
    /// amplitude of 4 - well past the i16 clipping ceiling - which only
    /// showed up once real data was pushed through real hardware. This
    /// checks the calibrated shaper stays within its target headroom for a
    /// long, different-seed pseudorandom sequence (not the exact one used
    /// internally to calibrate, so this isn't just checking its own math
    /// against itself).
    #[test]
    fn calibrated_shaper_does_not_exceed_target_headroom_on_independent_random_data() {
        let sps = 8;
        let mut shaper = PulseShaper::new(sps, 0.35, 8, ShapeMode::Rrc);
        let mut lfsr_state: u32 = 0x1357_9BDF; // different seed than the internal calibration
        let mut peak = 0.0f32;
        let mut out = Vec::new();
        for _ in 0..6000 {
            lfsr_state ^= lfsr_state << 13;
            lfsr_state ^= lfsr_state >> 17;
            lfsr_state ^= lfsr_state << 5;
            let symbol = Complex32::from_polar(
                1.0,
                std::f32::consts::FRAC_PI_4 * ((lfsr_state >> 3) & 7) as f32,
            );
            out.clear();
            shaper.process_symbol(symbol, &mut out);
            for &s in &out {
                peak = peak.max(s.norm());
            }
        }
        // Small margin above the calibration target: different seed/length
        // than the internal calibration run, so this isn't exactly bounded
        // by CALIBRATION_TARGET_PEAK, but should be close if calibration
        // generalizes rather than overfitting to its own sequence.
        assert!(
            peak < 0.95,
            "peak amplitude {peak} exceeds safe margin below full scale (1.0) - would clip on real hardware"
        );
    }
}
