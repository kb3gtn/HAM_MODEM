//! DC blocking (high-pass) filter for RX IQ: the classic single-pole
//! "DC blocker", y[n] = x[n] - x[n-1] + alpha*y[n-1], applied identically to
//! the complex sample (equivalent to applying it independently to I and Q,
//! since it's a linear filter with real coefficients).
//!
//! IMPORTANT: this is best used on a signal that isn't centered at 0 Hz. The
//! 8PSK signal has no carrier line (flat-topped spectrum, unlike the old BPSK
//! design's DC peak), so a narrow blocker costs little even on an unshifted
//! signal, but the LO leakage it targets sits at the same frequency as the
//! middle of an unshifted signal's band. This is meant to pair with `TxSignalGenerator`'s
//! `freq_shift_hz`: shift the signal off DC on TX, then this filter can
//! remove just the leakage sitting at DC on RX without touching the signal.

use num_complex::Complex32;

pub struct DcBlocker {
    alpha: f32,
    prev_x: Complex32,
    prev_y: Complex32,
}

impl DcBlocker {
    pub fn new(alpha: f32) -> Self {
        assert!((0.0..1.0).contains(&alpha), "alpha must be in [0, 1)");
        DcBlocker {
            alpha,
            prev_x: Complex32::new(0.0, 0.0),
            prev_y: Complex32::new(0.0, 0.0),
        }
    }

    /// Convenience constructor: derive alpha from a target -3dB cutoff
    /// frequency. Uses the standard small-angle approximation
    /// `alpha ~= 1 - 2*pi*cutoff/sample_rate`, valid when cutoff << sample_rate
    /// (alpha close to 1) - true for the intended use (a cutoff of a few
    /// hundred Hz against an 800kHz hardware rate).
    pub fn from_cutoff_hz(cutoff_hz: f64, sample_rate_hz: f64) -> Self {
        let alpha = 1.0 - 2.0 * std::f64::consts::PI * cutoff_hz / sample_rate_hz;
        Self::new(alpha.clamp(0.0, 0.999_999) as f32)
    }

    pub fn process(&mut self, x: Complex32) -> Complex32 {
        let y = x - self.prev_x + self.alpha * self.prev_y;
        self.prev_x = x;
        self.prev_y = y;
        y
    }

    pub fn process_block(&mut self, xs: &[Complex32]) -> Vec<Complex32> {
        xs.iter().map(|&x| self.process(x)).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn removes_a_pure_dc_offset() {
        let mut blocker = DcBlocker::from_cutoff_hz(500.0, 800_000.0);
        let dc = Complex32::new(0.3, -0.2);
        let mut last = Complex32::new(0.0, 0.0);
        for _ in 0..20_000 {
            last = blocker.process(dc);
        }
        assert!(
            last.norm() < 1e-3,
            "residual DC {last:?} should have settled near zero"
        );
    }

    #[test]
    fn passes_a_tone_well_above_cutoff_with_near_unity_gain() {
        let sample_rate = 800_000.0;
        let cutoff = 300.0;
        let tone_hz = 50_000.0; // ~170x the cutoff
        let mut blocker = DcBlocker::from_cutoff_hz(cutoff, sample_rate);

        let n = 4000;
        let tone: Vec<Complex32> = (0..n)
            .map(|k| {
                let phase = 2.0 * std::f64::consts::PI * tone_hz * (k as f64) / sample_rate;
                Complex32::new(phase.cos() as f32, phase.sin() as f32)
            })
            .collect();

        let out = blocker.process_block(&tone);
        // Compare RMS amplitude over the back half (after the filter's own
        // transient has settled) to the input's RMS amplitude (1.0 for a
        // unit-magnitude tone).
        let settled = &out[n / 2..];
        let rms: f32 =
            (settled.iter().map(|c| c.norm_sqr()).sum::<f32>() / settled.len() as f32).sqrt();
        assert!(
            (rms - 1.0).abs() < 0.05,
            "tone RMS {rms} should be close to 1.0 (near-unity gain)"
        );
    }

    #[test]
    fn attenuates_a_tone_at_dc_more_than_one_well_above_cutoff() {
        // Sanity check on the "don't use this on an unshifted signal" warning:
        // confirms the filter really does treat 0 Hz very differently from a
        // frequency safely above the cutoff.
        let sample_rate = 800_000.0;
        let mut blocker_dc = DcBlocker::from_cutoff_hz(300.0, sample_rate);
        let mut blocker_tone = DcBlocker::from_cutoff_hz(300.0, sample_rate);

        let n = 4000;
        let dc = Complex32::new(1.0, 0.0);
        let dc_out = blocker_dc.process_block(&vec![dc; n]);
        let dc_rms: f32 =
            (dc_out[n / 2..].iter().map(|c| c.norm_sqr()).sum::<f32>() / (n / 2) as f32).sqrt();

        let tone: Vec<Complex32> = (0..n)
            .map(|k| {
                let phase = 2.0 * std::f64::consts::PI * 50_000.0 * (k as f64) / sample_rate;
                Complex32::new(phase.cos() as f32, phase.sin() as f32)
            })
            .collect();
        let tone_out = blocker_tone.process_block(&tone);
        let tone_rms: f32 =
            (tone_out[n / 2..].iter().map(|c| c.norm_sqr()).sum::<f32>() / (n / 2) as f32).sqrt();

        assert!(
            dc_rms < tone_rms / 10.0,
            "DC ({dc_rms}) should be attenuated far more than the off-DC tone ({tone_rms})"
        );
    }
}
