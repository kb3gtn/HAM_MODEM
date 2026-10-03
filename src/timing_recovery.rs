//! Symbol timing recovery for 8PSK: consumes the matched filter's oversampled
//! (nominally `sps` samples/symbol) complex output and produces one complex
//! sample per symbol.
//!
//! Chain ordering (changed from the BPSK design): timing recovery now runs
//! BEFORE carrier recovery. The BPSK chain ran a decision-directed M&M
//! detector after an oversampled Costas loop, which relied on the signal
//! already being phase-aligned. 8PSK decisions are only reliable within
//! +-22.5 degrees of a constellation point, so a decision-directed timing
//! detector can't be trusted until the carrier is locked - and the carrier
//! loop (now at symbol rate) can't run until timing is resolved. The Gardner
//! detector breaks the deadlock: `Re{ conj(y_mid) * (y_n - y_prev) }` is
//! invariant to a common phase rotation of all three samples, needs no
//! decisions, and tolerates the small per-symbol rotation of a residual
//! frequency offset.
//!
//! Structure: a 4-point cubic (Lagrange) fractional interpolator supplies
//! samples at arbitrary fractional positions; at every strobe it is
//! evaluated twice (the strobe itself and the half-symbol point before it).
//! The Gardner error drives a 2nd-order PI loop that adjusts both the
//! estimated samples-per-symbol (`omega`) and the next strobe instant.
//!
//! The input is expected to be amplitude-normalized (see `agc::Agc`), since
//! the Gardner error scales with signal power.

use num_complex::Complex32;
use std::collections::VecDeque;

/// Samples of history kept. Cubic interpolation at the half-symbol point
/// reaches back ~omega/2 + 3 samples; this leaves headroom up to sps ~ 20.
const HISTORY_LEN: usize = 32;

pub struct TimingRecovery {
    omega: f64,         // current estimate of samples per symbol
    omega_nominal: f64, // the configured/expected samples per symbol
    omega_limit: f64,   // max fractional deviation of omega from nominal
    damping: f64,
    alpha: f64, // proportional gain (nudges the next strobe instant directly)
    beta: f64,  // integral gain (nudges omega, the rate estimate)
    t: f64,     // absolute input-sample index of the newest sample
    next_strobe_time: f64,
    history: VecDeque<Complex32>, // newest at front, oldest at back
    prev_y: Complex32,
    have_prev: bool,
}

impl TimingRecovery {
    /// `samples_per_symbol`: the nominal, fixed oversampling ratio of the
    /// incoming stream (e.g. 8.0 - matches the RX resampler's target rate).
    /// `loop_bandwidth`: normalized loop bandwidth in cycles/symbol (the
    /// loop runs once per strobe; 0.02 converges a 0.6% symbol-clock offset
    /// within a few hundred symbols, 0.01 is too sluggish for that).
    /// `damping`: standard PLL damping factor (0.707 is the usual default).
    pub fn new(samples_per_symbol: f64, loop_bandwidth: f64, damping: f64) -> Self {
        let mut recovery = TimingRecovery {
            omega: samples_per_symbol,
            omega_nominal: samples_per_symbol,
            omega_limit: 0.02, // allow omega to drift up to 2% from nominal
            damping,
            alpha: 0.0,
            beta: 0.0,
            t: -1.0,
            next_strobe_time: samples_per_symbol,
            history: VecDeque::with_capacity(HISTORY_LEN + 1),
            prev_y: Complex32::new(0.0, 0.0),
            have_prev: false,
        };
        recovery.set_loop_bandwidth(loop_bandwidth);
        recovery
    }

    /// Retune the loop bandwidth at runtime (standard bilinear-transform PI
    /// gain design, normalized in cycles/symbol).
    pub fn set_loop_bandwidth(&mut self, normalized_bw: f64) {
        let denom = 1.0 + 2.0 * self.damping * normalized_bw + normalized_bw * normalized_bw;
        self.alpha = (4.0 * self.damping * normalized_bw) / denom;
        self.beta = (4.0 * normalized_bw * normalized_bw) / denom;
    }

    /// Forget the learned symbol-clock offset and strobe phase: omega returns
    /// to nominal and the next strobe is one symbol after the newest sample.
    /// The sample history is kept (it is just recent input), so the loop
    /// resumes immediately rather than going through the startup skip.
    pub fn reset(&mut self) {
        self.omega = self.omega_nominal;
        self.next_strobe_time = self.t + self.omega_nominal;
        self.prev_y = Complex32::new(0.0, 0.0);
        self.have_prev = false;
    }

    /// Interpolate at absolute input-sample time `tau`. Caller guarantees
    /// enough history/lookahead (see `process`).
    fn interp(&self, tau: f64) -> Complex32 {
        let n = tau.floor();
        let mu = (tau - n) as f32;
        let offset0 = (self.t - n) as usize; // history index of x[n]
        cubic_lagrange(
            self.history[offset0 + 1],
            self.history[offset0],
            self.history[offset0 - 1],
            self.history[offset0 - 2],
            mu,
        )
    }

    /// Feed one oversampled input sample. Returns `Some(sample)` on symbols
    /// where a strobe fired (roughly every `omega` input samples), `None`
    /// otherwise.
    pub fn process(&mut self, x: Complex32) -> Option<Complex32> {
        self.t += 1.0;
        self.history.push_front(x);
        if self.history.len() > HISTORY_LEN {
            self.history.pop_back();
        }

        let n = self.next_strobe_time.floor();
        // Need x[n+2] to have arrived (cubic lookahead), and the half-symbol
        // point's x[n_mid-1] to still be in history.
        if self.t < n + 2.0 {
            return None;
        }
        let mid_time = self.next_strobe_time - self.omega / 2.0;
        let oldest_needed = mid_time.floor() - 1.0;
        if self.t - oldest_needed >= self.history.len() as f64 {
            // Startup: not enough history yet; skip this strobe slot.
            self.next_strobe_time += self.omega;
            return None;
        }

        let y = self.interp(self.next_strobe_time);
        let y_mid = self.interp(mid_time);

        // Gardner timing error: zero at the correct sampling instant. Positive
        // when sampling late.
        let timing_error = if self.have_prev {
            f64::from((y_mid.conj() * (y - self.prev_y)).re)
        } else {
            0.0
        };
        // Bound the per-symbol correction: a single wild sample (e.g. at
        // startup, before AGC settles) must not fling the loop.
        let timing_error = timing_error.clamp(-2.0, 2.0);

        self.omega -= self.beta * timing_error;
        let lo = self.omega_nominal * (1.0 - self.omega_limit);
        let hi = self.omega_nominal * (1.0 + self.omega_limit);
        self.omega = self.omega.clamp(lo, hi);

        self.next_strobe_time += self.omega - self.alpha * timing_error;

        self.prev_y = y;
        self.have_prev = true;

        Some(y)
    }

    pub fn process_block(&mut self, xs: &[Complex32]) -> Vec<Complex32> {
        xs.iter().filter_map(|&x| self.process(x)).collect()
    }

    /// Current samples-per-symbol estimate (tracks real symbol-clock offset
    /// away from the nominal rate, within `omega_limit`).
    pub fn omega(&self) -> f64 {
        self.omega
    }
}

/// Exact cubic (Lagrange) interpolation through 4 consecutive samples at
/// integer positions -1, 0, 1, 2, evaluated at fractional position `mu` in
/// [0, 1) (0 = `x0`, approaching 1 = `x1`).
fn cubic_lagrange(
    x_m1: Complex32,
    x0: Complex32,
    x1: Complex32,
    x2: Complex32,
    mu: f32,
) -> Complex32 {
    let l_m1 = mu * (mu - 1.0) * (mu - 2.0) / -6.0;
    let l_0 = (mu + 1.0) * (mu - 1.0) * (mu - 2.0) / 2.0;
    let l_1 = (mu + 1.0) * mu * (mu - 2.0) / -2.0;
    let l_2 = (mu + 1.0) * mu * (mu - 1.0) / 6.0;
    x_m1 * l_m1 + x0 * l_0 + x1 * l_1 + x2 * l_2
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::rrc::{MatchedFilter, PulseShaper, ShapeMode};
    use crate::symbol_map::{psk8_point, psk8_position};

    /// Pseudorandom 8PSK symbol positions (xorshift32).
    pub(crate) fn random_positions(n: usize, seed: u32) -> Vec<u8> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 17;
                s ^= s << 5;
                ((s >> 4) & 7) as u8
            })
            .collect()
    }

    /// TX RRC shape + RX matched filter at a high internal oversampling
    /// factor, then decimated to the target sps with a chosen starting
    /// phase - injects a known sub-sample fractional timing offset
    /// (`phase_offset / oversample_factor` of one target-rate sample).
    pub(crate) fn oversampled_matched_filtered(
        positions: &[u8],
        target_sps: usize,
        oversample_factor: usize,
        phase_offset: usize,
        channel_gain: Complex32,
    ) -> Vec<Complex32> {
        let hi_sps = target_sps * oversample_factor;
        let rolloff = 0.35;
        let span_symbols = 8;
        let mut shaper = PulseShaper::new(hi_sps, rolloff, span_symbols, ShapeMode::Rrc);
        let mut matched = MatchedFilter::new(hi_sps, rolloff, span_symbols);
        let mut shaped = Vec::new();
        for &p in positions {
            shaper.process_symbol(psk8_point(p), &mut shaped);
        }
        let filtered = matched.process_block(&shaped);
        let scale = 1.0 / rms(&filtered);
        filtered
            .into_iter()
            .skip(phase_offset)
            .step_by(oversample_factor)
            .map(|s| s * scale * channel_gain)
            .collect()
    }

    /// RMS amplitude, used to hand the timing loop a unit-power input like
    /// the AGC does in the real chain.
    fn rms(xs: &[Complex32]) -> f32 {
        (xs.iter().map(|c| c.norm_sqr()).sum::<f32>() / xs.len() as f32)
            .sqrt()
            .max(1e-6)
    }

    /// Fraction of recovered symbols whose position matches the transmitted
    /// one, allowing for an unknown constant rotation (0..8 steps) and an
    /// unknown alignment lag - carrier recovery hasn't run here, so the
    /// per-symbol phase also needs a static (no frequency) channel.
    fn best_position_agreement(out: &[Complex32], positions: &[u8]) -> f64 {
        let mut best = 0.0f64;
        for rot in 0..8u8 {
            for lag in -80i64..80 {
                let mut correct = 0usize;
                let mut total = 0usize;
                for (i, y) in out.iter().enumerate() {
                    let j = i as i64 + lag;
                    if j < 0 || j as usize >= positions.len() {
                        continue;
                    }
                    total += 1;
                    if (psk8_position(*y).wrapping_sub(rot) & 7) == positions[j as usize] {
                        correct += 1;
                    }
                }
                if total > 20 {
                    best = best.max(correct as f64 / total as f64);
                }
            }
        }
        best
    }

    #[test]
    fn locks_to_a_fixed_fractional_timing_offset() {
        let sps = 8;
        let positions = random_positions(1500, 0xDEAD_BEEF);
        // phase_offset=2 of oversample_factor=4 -> 0.5-sample offset.
        let input =
            oversampled_matched_filtered(&positions, sps, 4, 2, Complex32::from_polar(1.0, 0.3));

        let mut recovery = TimingRecovery::new(sps as f64, 0.02, 0.707);
        let output = recovery.process_block(&input);
        assert!(
            output.len() > positions.len() / 2,
            "expected roughly one strobe per symbol, got {}",
            output.len()
        );

        let tail = &output[output.len() - 400..];
        let settled = &positions[positions.len() - 450..];
        let agreement = best_position_agreement(tail, settled);
        assert!(
            agreement > 0.95,
            "settled position agreement only {agreement}, expected >0.95"
        );
    }

    #[test]
    fn tracks_a_symbol_rate_frequency_offset() {
        let sps = 8;
        let positions = random_positions(4000, 0x1234_5678);
        let baseline =
            oversampled_matched_filtered(&positions, sps, 4, 0, Complex32::new(1.0, 0.0));

        // Stretch so the true samples/symbol becomes ~8.05 (linear
        // interpolation for ground truth, independent of the cubic under test).
        let true_sps = 8.05;
        let step = sps as f64 / true_sps;
        let mut stretched = Vec::new();
        let mut pos = 0.0f64;
        while (pos.floor() as usize) + 1 < baseline.len() {
            let i = pos.floor() as usize;
            let frac = (pos - pos.floor()) as f32;
            stretched.push(baseline[i] * (1.0 - frac) + baseline[i + 1] * frac);
            pos += step;
        }

        let mut recovery = TimingRecovery::new(sps as f64, 0.02, 0.707);
        let output = recovery.process_block(&stretched);

        let omega_err = (recovery.omega() - true_sps).abs();
        assert!(
            omega_err < 0.02,
            "omega estimate {} off from true {} by {omega_err}, expected <0.02",
            recovery.omega(),
            true_sps
        );

        let tail = &output[output.len() - 400..];
        let settled = &positions[positions.len() - 450..];
        let agreement = best_position_agreement(tail, settled);
        assert!(
            agreement > 0.9,
            "settled position agreement only {agreement}, expected >0.9"
        );
    }

    #[test]
    fn timing_lock_is_insensitive_to_carrier_phase() {
        // Gardner is phase-invariant: any static carrier phase must lock.
        let sps = 8;
        let positions = random_positions(1500, 0xC0FF_EE11);
        // Phases chosen away from the 22.5-degree slicer boundaries (this test
        // slices without carrier recovery).
        for phase in [0.0f32, 0.1, 0.9, 2.4, -1.55] {
            let input = oversampled_matched_filtered(
                &positions,
                sps,
                4,
                1,
                Complex32::from_polar(0.6, phase),
            );
            // Amplitude-normalized input is assumed (AGC): rescale here.
            let input: Vec<Complex32> = input.iter().map(|&c| c / 0.6).collect();
            let mut recovery = TimingRecovery::new(sps as f64, 0.02, 0.707);
            let output = recovery.process_block(&input);
            let agreement = best_position_agreement(
                &output[output.len() - 400..],
                &positions[positions.len() - 450..],
            );
            assert!(agreement > 0.95, "phase {phase}: agreement {agreement}");
        }
    }
}
