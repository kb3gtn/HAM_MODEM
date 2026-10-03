//! Carrier recovery for BPSK/QPSK/8PSK (M-PSK, M = 2/4/8), running at SYMBOL
//! rate on the timing-recovered samples (one per symbol). The doc below is
//! written for 8PSK; for other M read "8" as M and "45 degrees" as 360/M
//! degrees, and note that every threshold that was tuned for 8PSK is rescaled
//! (see `phase_error_scale`, `max_frequency_hz`).
//!
//! An NCO-driven derotator closed by two detectors through a 2nd-order PI
//! loop filter:
//!  * a decision-directed phase detector (angle to the nearest of the 8
//!    constellation points, +-22.5 degrees linear range), and
//!  * a frequency-lock aid: the angle between consecutive 8th-power
//!    samples, `arg(y[n]^8 * conj(y[n-1]^8)) / 8`. Raising to the 8th power
//!    strips the modulation, so this measures residual frequency (in
//!    rad/symbol) without decisions, with an unambiguous range of
//!    +-1/16 cycle/symbol (+-1 kHz at 16 ksym/s). The decision-directed
//!    detector alone has a small pull-in range at 8PSK - its linear range is
//!    narrow, so an initial offset of even ~10 degrees/symbol would stall it.
//!
//! KNOWN, EXPECTED LIMITATION: the loop is blind to rotations by multiples of
//! 45 degrees - it may settle on any of 8 phases, and decision-directed
//! tracking can cycle-slip by +-45 degrees under noise. This is resolved
//! downstream, not here: see `hdlc::RotationResolvingDeframer` (and the PRBS
//! BERT bank in `rx_signal`), which test all 8 rotations of the demapped
//! stream and pick whichever produces valid frames.

use num_complex::Complex32;

use crate::symbol_map::Modulation;

/// Fraction of the measured residual-frequency error applied per symbol by
/// the frequency-lock aid.
const DEFAULT_FLL_GAIN: f64 = 0.005;
/// IIR coefficient for the smoothed |phase error| lock metric.
const LOCK_METRIC_RATE: f64 = 0.01;
/// The frequency-lock aid is unambiguous only within +-1/(2M) cycle/symbol
/// (+-1 kHz at 16 ksym/s 8PSK), so the NCO is never allowed outside that
/// much rad/symbol: a state beyond it could never pull back in to a real
/// signal.
/// Metric above which the loop is considered unlocked (noise-only gives
/// ~pi/16 = 0.196; a marginal-SNR but locked link was seen at ~0.10-0.13).
const UNLOCKED_METRIC: f64 = 0.17;
/// While unlocked, the frequency estimate decays toward 0 by this fraction
/// per symbol, so after a signal dropout the NCO re-centres and the real
/// carrier (within +-1 kHz of 0) is again inside the pull-in range. Small
/// enough that the frequency-lock aid wins while a real signal is acquiring.
const UNLOCKED_FREQ_LEAK: f64 = 1e-3;
/// IIR coefficient for the transition-rate ("do the symbols actually vary?")
/// metric and for the SNR statistics (~500-symbol averaging).
const TRANSITION_RATE: f64 = 0.01;
const SNR_RATE: f64 = 0.002;
/// Random 8PSK data changes position 7/8 of the time; a loop that has latched
/// onto a CW spur sees (nearly) none. Below this it is not a real lock.
const MIN_TRANSITION_FRACTION: f64 = 0.5;

pub struct CarrierLoop {
    modulation: Modulation,
    symbol_rate_hz: f64,
    damping: f64,
    alpha: f64, // proportional gain
    beta: f64,  // integral gain
    fll_gain: f64,
    freq: f64,  // current frequency estimate, radians/symbol
    phase: f64, // current phase estimate, radians
    prev_y8: Option<Complex32>,
    mean_abs_phase_error: f64,
    prev_position: Option<u8>,
    transition_fraction: f64,
    /// Running least-squares estimate of the received symbol amplitude (the
    /// real part of y * conj(nearest ideal point)) and of the squared error
    /// vector |y - gain*ideal|^2, for the SNR estimate.
    gain: f64,
    error_power: f64,
}

impl CarrierLoop {
    pub fn new(
        modulation: Modulation,
        symbol_rate_hz: f64,
        loop_bandwidth_hz: f64,
        damping: f64,
    ) -> Self {
        let m = f64::from(modulation.num_points());
        let mut loop_ = CarrierLoop {
            modulation,
            symbol_rate_hz,
            damping,
            alpha: 0.0,
            beta: 0.0,
            fll_gain: DEFAULT_FLL_GAIN,
            freq: 0.0,
            phase: 0.0,
            prev_y8: None,
            mean_abs_phase_error: std::f64::consts::PI / m,
            prev_position: None,
            transition_fraction: (m - 1.0) / m,
            gain: 1.0,
            error_power: 1.0,
        };
        loop_.set_loop_bandwidth_hz(loop_bandwidth_hz);
        loop_
    }

    /// Retune the loop bandwidth at runtime. Standard second-order PLL gain
    /// design (bilinear-transform based): given normalized loop bandwidth
    /// `loop_bw` (radians/symbol) and damping factor `zeta`,
    ///   alpha = 4*zeta*loop_bw / (1 + 2*zeta*loop_bw + loop_bw^2)
    ///   beta  = 4*loop_bw^2    / (1 + 2*zeta*loop_bw + loop_bw^2)
    pub fn set_loop_bandwidth_hz(&mut self, bandwidth_hz: f64) {
        let loop_bw = 2.0 * std::f64::consts::PI * bandwidth_hz / self.symbol_rate_hz;
        let denom = 1.0 + 2.0 * self.damping * loop_bw + loop_bw * loop_bw;
        self.alpha = (4.0 * self.damping * loop_bw) / denom;
        self.beta = (4.0 * loop_bw * loop_bw) / denom;
    }

    /// Set the frequency-lock-aid gain (0 disables it).
    pub fn set_fll_gain(&mut self, gain: f64) {
        self.fll_gain = gain;
    }

    /// Derotate one symbol-rate sample and update the loop from it.
    pub fn process(&mut self, x: Complex32) -> Complex32 {
        let (sin_p, cos_p) = self.phase.sin_cos();
        let y = x * Complex32::new(cos_p as f32, -sin_p as f32);

        let m = self.modulation;
        let error = f64::from(m.phase_error(y));
        self.mean_abs_phase_error += LOCK_METRIC_RATE * (error.abs() - self.mean_abs_phase_error);

        // Symbol-error statistics against the nearest ideal point.
        let position = m.position(y);
        let ideal = m.point(position);
        let changed = if self.prev_position.is_some_and(|p| p != position) {
            1.0
        } else {
            0.0
        };
        if self.prev_position.is_some() {
            self.transition_fraction += TRANSITION_RATE * (changed - self.transition_fraction);
        }
        self.prev_position = Some(position);
        let amplitude = f64::from((y * ideal.conj()).re);
        self.gain += SNR_RATE * (amplitude - self.gain);
        let err = y - ideal * self.gain as f32;
        self.error_power += SNR_RATE * (f64::from(err.norm_sqr()) - self.error_power);

        // Frequency-lock aid on the modulation-stripped 8th power.
        let mag = y.norm();
        let mut fll_error = 0.0;
        if mag > 1e-9 {
            // Raise to the M-th power (repeated squaring): strips the modulation.
            let mut ym = y / mag;
            for _ in 0..m.bits_per_symbol() {
                ym = ym * ym;
            }
            if let Some(prev) = self.prev_y8 {
                fll_error = f64::from((ym * prev.conj()).arg()) / f64::from(m.num_points());
            }
            self.prev_y8 = Some(ym);
        }

        self.freq += self.beta * error + self.fll_gain * fll_error;
        if self.mean_abs_phase_error > UNLOCKED_METRIC * self.phase_error_scale() {
            self.freq *= 1.0 - UNLOCKED_FREQ_LEAK;
        }
        let max_freq = 2.0 * std::f64::consts::PI * self.max_frequency_hz() / self.symbol_rate_hz;
        self.freq = self.freq.clamp(-max_freq, max_freq);
        self.phase += self.freq + self.alpha * error;
        self.phase = wrap_phase(self.phase);

        y
    }

    pub fn process_block(&mut self, xs: &[Complex32]) -> Vec<Complex32> {
        xs.iter().map(|&x| self.process(x)).collect()
    }

    /// Factor (8/M) that converts a mean-|phase error| threshold tuned for
    /// 8PSK into the equivalent for this constellation: the decision region
    /// is +-pi/M wide, so noise-only input measures pi/(2M) (0.196 rad for
    /// 8PSK, 0.39 QPSK, 0.79 BPSK).
    pub fn phase_error_scale(&self) -> f64 {
        8.0 / f64::from(self.modulation.num_points())
    }

    /// Largest NCO frequency (Hz) the loop is allowed: the frequency-lock
    /// aid's unambiguous range, symbol_rate / (2M).
    pub fn max_frequency_hz(&self) -> f64 {
        self.symbol_rate_hz / (2.0 * f64::from(self.modulation.num_points()))
    }

    /// Spacing (Hz) of the frequency offsets at which this constellation
    /// looks identical from one symbol to the next (symbol_rate / M): a
    /// decision-directed lock can land on any of them (see `coarse_freq`).
    pub fn alias_spacing_hz(&self) -> f64 {
        self.symbol_rate_hz / f64::from(self.modulation.num_points())
    }

    /// Current frequency estimate in Hz.
    pub fn frequency_hz(&self) -> f64 {
        self.freq * self.symbol_rate_hz / (2.0 * std::f64::consts::PI)
    }

    pub fn phase_radians(&self) -> f64 {
        self.phase
    }

    /// Zero the frequency estimate (used when the receiver re-tunes its
    /// front-end LO during a search, so stale frequency doesn't carry over).
    pub fn reset_frequency(&mut self) {
        self.freq = 0.0;
        self.prev_y8 = None;
    }

    /// Return the loop to its just-constructed state (frequency, phase and all
    /// lock/SNR statistics); the loop bandwidth and FLL gain are kept.
    pub fn reset(&mut self) {
        self.freq = 0.0;
        self.phase = 0.0;
        self.prev_y8 = None;
        let m = f64::from(self.modulation.num_points());
        self.mean_abs_phase_error = std::f64::consts::PI / m;
        self.prev_position = None;
        self.transition_fraction = (m - 1.0) / m;
        self.gain = 1.0;
        self.error_power = 1.0;
    }

    /// True when the loop looks genuinely locked to modulated data: low
    /// decision phase error AND symbols that actually vary (a CW spur gives
    /// low phase error too, but never changes constellation position).
    pub fn is_locked(&self, max_phase_error: f64) -> bool {
        let m = f64::from(self.modulation.num_points());
        // Random data changes position (M-1)/M of the time; the threshold is
        // MIN_TRANSITION_FRACTION at 8PSK's 7/8, scaled for the others.
        let min_transitions = MIN_TRANSITION_FRACTION * ((m - 1.0) / m) / (7.0 / 8.0);
        self.mean_abs_phase_error < max_phase_error && self.transition_fraction > min_transitions
    }

    /// Standard deviation of the symbol error vector, relative to the
    /// received symbol amplitude (so 0 = perfect, 1 = error as large as the
    /// signal). This is the RMS length of the complex error `y/gain -
    /// ideal`; each of I and Q has `1/sqrt(2)` of it. Only meaningful while
    /// locked.
    pub fn symbol_error_std(&self) -> f64 {
        if self.gain <= 1e-6 {
            return f64::INFINITY;
        }
        self.error_power.max(0.0).sqrt() / self.gain
    }

    /// Estimated symbol SNR (Es/N0 at the matched-filter output) in dB, from
    /// the symbol-error standard deviation: SNR = 1 / std^2. Decision-directed,
    /// so it is accurate above roughly 12 dB and optimistically biased below
    /// that (wrong decisions look like small errors); it is not meaningful
    /// unless the loop is locked.
    pub fn snr_db(&self) -> f64 {
        let std = self.symbol_error_std();
        -20.0 * std.max(1e-6).log10()
    }

    /// Smoothed mean |decision phase error| in radians: ~0 when locked on a
    /// clean signal, approaching pi/16 (uniform over +-pi/8) when there is no
    /// lock. A cheap lock indicator for status reporting.
    pub fn mean_abs_phase_error(&self) -> f64 {
        self.mean_abs_phase_error
    }
}

fn wrap_phase(phase: f64) -> f64 {
    let tau = std::f64::consts::TAU;
    let wrapped = phase % tau;
    if wrapped > std::f64::consts::PI {
        wrapped - tau
    } else if wrapped < -std::f64::consts::PI {
        wrapped + tau
    } else {
        wrapped
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::symbol_map::{psk8_phase_error, psk8_point};

    const FS: f64 = 16_000.0;

    /// Pseudorandom 8PSK symbols, rotated by a (possibly time-varying) phase.
    fn rotated_stream(n: usize, phase_at: impl Fn(usize) -> f64) -> Vec<Complex32> {
        let mut s: u32 = 0xDEAD_BEEF;
        (0..n)
            .map(|k| {
                s ^= s << 13;
                s ^= s >> 17;
                s ^= s << 5;
                let p = ((s >> 4) & 7) as u8;
                let ph = phase_at(k);
                psk8_point(p) * Complex32::new(ph.cos() as f32, ph.sin() as f32)
            })
            .collect()
    }

    /// Residual phase error of the settled output, reduced mod 45 degrees.
    fn settled_error(out: &[Complex32]) -> f32 {
        let tail = &out[out.len() - 500..];
        tail.iter().map(|&y| psk8_phase_error(y).abs()).sum::<f32>() / tail.len() as f32
    }

    #[test]
    fn acquires_a_static_phase_offset_modulo_45_degrees() {
        for true_phase in [0.0f64, 0.2, 0.6, 1.5, -2.8] {
            let mut loop_ = CarrierLoop::new(Modulation::Psk8, FS, 100.0, 0.707);
            let out = loop_.process_block(&rotated_stream(4000, |_| true_phase));
            let err = settled_error(&out);
            assert!(err < 0.02, "phase {true_phase}: mean |err| {err} rad");
        }
    }

    #[test]
    fn tracks_a_frequency_offset() {
        for true_freq_hz in [50.0f64, -200.0, 500.0, -700.0] {
            let mut loop_ = CarrierLoop::new(Modulation::Psk8, FS, 100.0, 0.707);
            let w = 2.0 * std::f64::consts::PI * true_freq_hz / FS;
            let out = loop_.process_block(&rotated_stream(8000, |k| w * k as f64));
            let freq_err = (loop_.frequency_hz() - true_freq_hz).abs();
            assert!(
                freq_err < 5.0,
                "{true_freq_hz} Hz: estimate off by {freq_err} Hz"
            );
            let err = settled_error(&out);
            assert!(err < 0.03, "{true_freq_hz} Hz: mean |err| {err} rad");
        }
    }

    #[test]
    fn wider_bandwidth_acquires_a_phase_step_faster() {
        let true_phase = 0.3;
        let settle_time = |bandwidth_hz: f64| -> usize {
            let mut loop_ = CarrierLoop::new(Modulation::Psk8, FS, bandwidth_hz, 0.707);
            let input = rotated_stream(8000, |_| true_phase);
            let mut run = 0;
            for (i, &x) in input.iter().enumerate() {
                let y = loop_.process(x);
                if psk8_phase_error(y).abs() < 0.03 {
                    run += 1;
                    if run >= 20 {
                        return i;
                    }
                } else {
                    run = 0;
                }
            }
            usize::MAX
        };
        let narrow = settle_time(20.0);
        let wide = settle_time(300.0);
        assert!(
            wide < narrow,
            "wider loop ({wide} symbols) should settle faster than narrow ({narrow})"
        );
    }

    #[test]
    fn recovers_after_a_signal_dropout() {
        // Lock at +400 Hz, lose the signal for a long time (noise only: the
        // frequency estimate random-walks), then bring the signal back. The
        // loop must re-centre during the dropout and reacquire afterwards.
        let w = 2.0 * std::f64::consts::PI * 400.0 / FS;
        let mut loop_ = CarrierLoop::new(Modulation::Psk8, FS, 100.0, 0.707);
        loop_.process_block(&rotated_stream(4000, |k| w * k as f64));
        assert!(
            (loop_.frequency_hz() - 400.0).abs() < 10.0,
            "initial lock failed"
        );

        let mut s: u32 = 0xBADC_0FFE;
        let mut rnd = || {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            f64::from(s) / f64::from(u32::MAX) - 0.5
        };
        for _ in 0..20_000 {
            loop_.process(Complex32::new(rnd() as f32, rnd() as f32));
            assert!(
                loop_.frequency_hz().abs() <= 1000.0 + 1e-6,
                "frequency left the clamp: {}",
                loop_.frequency_hz()
            );
        }
        assert!(
            loop_.frequency_hz().abs() < 150.0,
            "NCO did not re-centre during dropout: {} Hz",
            loop_.frequency_hz()
        );

        let out = loop_.process_block(&rotated_stream(8000, |k| w * k as f64));
        assert!(
            (loop_.frequency_hz() - 400.0).abs() < 10.0,
            "failed to reacquire: {} Hz",
            loop_.frequency_hz()
        );
        assert!(settled_error(&out) < 0.03);
    }

    fn gauss(state: &mut u32) -> f64 {
        let mut u = || {
            *state ^= *state << 13;
            *state ^= *state >> 17;
            *state ^= *state << 5;
            (f64::from(*state) + 1.0) / (f64::from(u32::MAX) + 2.0)
        };
        (-2.0 * u().ln()).sqrt() * (std::f64::consts::TAU * u()).cos()
    }

    #[test]
    fn snr_estimate_tracks_the_true_symbol_snr() {
        // 8PSK plus complex AWGN at a known Es/N0 (unit-power symbols, so
        // noise variance per complex sample = 10^(-snr/10)).
        let mut state = 0x5EED_1234u32;
        for snr_db in [16.0f64, 20.0, 24.0, 28.0] {
            let sigma = (10f64.powf(-snr_db / 10.0) / 2.0).sqrt();
            let clean = rotated_stream(30_000, |_| 0.0);
            let noisy: Vec<Complex32> = clean
                .iter()
                .map(|&c| {
                    c + Complex32::new(
                        (sigma * gauss(&mut state)) as f32,
                        (sigma * gauss(&mut state)) as f32,
                    )
                })
                .collect();
            let mut loop_ = CarrierLoop::new(Modulation::Psk8, FS, 50.0, 0.707);
            loop_.process_block(&noisy);
            assert!(
                loop_.is_locked(0.16),
                "snr {snr_db}: not locked (err {})",
                loop_.mean_abs_phase_error()
            );
            let est = loop_.snr_db();
            assert!(
                (est - snr_db).abs() < 0.8,
                "true {snr_db} dB, estimated {est:.2} dB (error std {:.4})",
                loop_.symbol_error_std()
            );
        }
    }

    #[test]
    fn a_cw_spur_is_not_reported_as_lock() {
        // A pure tone: the decision phase error goes to ~0 (the loop happily
        // locks onto it) but the symbols never change position.
        let w = 2.0 * std::f64::consts::PI * 300.0 / FS;
        let tone: Vec<Complex32> = (0..20_000)
            .map(|k| Complex32::from_polar(1.0, (w * k as f64) as f32))
            .collect();
        let mut loop_ = CarrierLoop::new(Modulation::Psk8, FS, 100.0, 0.707);
        loop_.process_block(&tone);
        assert!(
            loop_.mean_abs_phase_error() < 0.05,
            "expected the PLL to latch onto the tone"
        );
        assert!(
            !loop_.is_locked(0.16),
            "a CW spur must not count as a data lock"
        );
    }
}
