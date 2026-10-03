//! The full BPSK/QPSK/8PSK RX DSP chain: hardware IQ -> DC block (fixed cutoff, at
//! hardware rate) -> RX resample (hardware rate -> `sps` x symbol rate,
//! fixed ratio) -> matched RRC filter -> AGC -> Gardner timing recovery
//! (decimates to one complex sample per symbol) -> symbol-rate carrier
//! recovery (decision-directed PLL + frequency-lock aid). Mirrors
//! `TxSignalGenerator`'s role on the TX side.
//!
//! Output is one derotated complex symbol per received symbol (pre-slicing).
//!
//! KNOWN, EXPECTED LIMITATION: carrier recovery cannot resolve the 8-fold
//! (45 degree) phase ambiguity, so symbols may come out rotated by any
//! multiple of 45 degrees, and the rotation can change after a cycle slip.
//! Consumers handle that by testing all 8 rotation hypotheses: the embedded
//! BERT bank below does it for PRBS, and `hdlc::RotationResolvingDeframer`
//! does it for framed data.

use bladerf::ComplexI16;
use num_complex::Complex32;
use std::collections::VecDeque;

use crate::agc::Agc;
use crate::carrier_recovery::CarrierLoop;
use crate::coarse_freq;
use crate::dc_block::DcBlocker;
use crate::fec_bank::{branch_pair_phase, branch_rotation, num_branches, SoftDecoderBank};
use crate::prbs::PrbsPattern;
use crate::psd::complex_i16_to_f32;
use crate::resampler::RxResampler;
use crate::rrc::MatchedFilter;
use crate::rx_bert::{LockState, RxBert, RxBertControl, RxBertStatus};
use crate::symbol_map::Modulation;
use crate::timing_recovery::TimingRecovery;
use crate::Block;

/// How many of the most recent derotated symbols are kept for telemetry
/// (the constellation display).
pub const RECENT_SYMBOLS: usize = 256;

/// Per-sample AGC adaptation rate (symbol-domain rate): ~40 ms time constant.
const AGC_RATE: f32 = 2e-4;

/// Frequency acquisition. The carrier loop alone pulls in only about +-1 kHz
/// (and mistakes offsets near multiples of symbol_rate/8 = 2 kHz for lock -
/// see `coarse_freq`). So whenever it is not locked, the receiver measures the
/// signal's spectral centre (`coarse_freq`) and retunes its front-end LO
/// (the shift NCO) by that much, leaving the carrier loop only a small
/// residual. Search range +-6 kHz.
///
/// The time/frequency constants below were tuned at 16 ksym/s 8PSK. For other
/// symbol rates and constellations they are rescaled in `RxSignalProcessor::new`:
/// those quoted in Hz in proportion to the symbol rate (or, where they relate
/// to the carrier loop's capture range or alias lattice, to those), and the
/// phase-error thresholds by `CarrierLoop::phase_error_scale`.
const MAX_SEARCH_OFFSET_HZ: f64 = 6_000.0;
/// ...but never less than this: a +-2.5 ppm crystal at 440 MHz is already
/// +-1.1 kHz, however slow the symbol rate (the symbol-domain Nyquist rate
/// even at the lowest rate, 8 x 2 kHz / 2 = 8 kHz, leaves room for it).
const MIN_SEARCH_RANGE_HZ: f64 = 3_000.0;
const TUNED_SYMBOL_RATE_HZ: f64 = 16_000.0;
const ACQ_FFT_SIZE: usize = 1024;
/// Symbol-domain samples collected per estimate (~256 ms at 128 kHz).
const ACQ_SAMPLES: usize = 32_768;
/// Ignore estimates whose spectral contrast is below this: noise-only input
/// measures ~0.015, a 10 dB signal >= 0.13.
const MIN_ACQ_CONTRAST: f64 = 0.05;
/// Only retune if the estimate differs from the current LO by more than this
/// (the carrier loop easily absorbs less).
const MIN_RETUNE_HZ: f64 = 200.0;
/// Symbols to wait after a retune/unlock before judging lock (~190 ms).
const SETTLE_SYMBOLS: u32 = 3_000;
/// Smoothed |phase error| below which (together with varying symbols) the
/// carrier loop counts as locked. Noise-only is ~0.196; a marginal-SNR but
/// working link was measured at 0.10-0.13.
const LOCKED_PHASE_ERROR: f64 = 0.16;
// A "lock" with the carrier loop's own frequency near its clamp is suspect
// (it is probably hanging on an alias), so it doesn't count: the limit is 0.8
// of the loop's capture range (800 Hz at 16 ksym/s 8PSK).
//
// An apparent carrier lock is accepted only if the loop's frequency is within
// 0.45 of the alias lattice spacing symbol_rate/M of the spectral estimate
// (900 Hz at 16 ksym/s 8PSK: alias locks are off by multiples of 2 kHz; the
// estimate itself is good to ~150 Hz).
/// While locked, the lock is re-checked against the spectrum this often
/// (~2 s): a frequency step after lock (e.g. the TX retuned by 1 kHz) can drag
/// the carrier loop onto a 2 kHz alias without it ever looking unlocked.
const REVERIFY_SYMBOLS: u32 = 32_000;
/// Once locked, the lock is declared lost after this many consecutive
/// symbols of looking unlocked (~125 ms), restarting acquisition.
const UNLOCK_SYMBOLS: u32 = 2_000;
const UNLOCK_PHASE_ERROR: f64 = 0.18;

pub struct RxSignalProcessor {
    dc_block: DcBlocker,
    /// Mixes the signal back to baseband after the DC blocker. The TX
    /// offset-tunes its signal away from the LO (so LO leakage at DC falls
    /// outside the 21.6 kHz occupied band); the DC blocker removes that
    /// leakage at DC, then this NCO removes the intentional offset so the
    /// carrier loop only has to track the true residual (crystal/Doppler)
    /// error, well inside its +-1 kHz capture range.
    shift_phase: f64,
    shift_phase_increment: f64,
    /// Nominal TX offset-tuning shift plus the search's current offset.
    base_shift_hz: f64,
    hardware_sample_rate: f64,
    modulation: Modulation,
    /// Rescaled acquisition thresholds (see the constants' docs).
    max_search_offset_hz: f64,
    min_retune_hz: f64,
    locked_phase_error: f64,
    unlock_phase_error: f64,
    max_locked_residual_hz: f64,
    max_alias_disagreement_hz: f64,
    /// Frequency acquisition state (see `MAX_SEARCH_OFFSET_HZ`).
    search_enabled: bool,
    search_offset_hz: f64,
    symbol_rate_hz: f64,
    rolloff: f64,
    symbol_domain_rate: f64,
    /// Symbol-domain samples being collected for a coarse estimate (only
    /// while unlocked and settled).
    acquiring: bool,
    /// The collection in progress is verifying an apparent carrier lock
    /// against the spectrum, rather than searching for a signal.
    verifying: bool,
    acq_buf: Vec<Complex32>,
    carrier_locked: bool,
    symbols_since_verify: u32,
    symbols_in_state: u32,
    unlocked_run: u32,
    resampler: RxResampler,
    matched_filter: MatchedFilter,
    agc: Agc,
    timing: TimingRecovery,
    carrier: CarrierLoop,
    /// Soft Viterbi decoders, one per (rotation, pair-phase) hypothesis, and one
    /// PRBS checker per decoder output. Only the correct hypothesis decodes to
    /// the PRBS, so exactly that one reaches `Synced` - this is the live
    /// ambiguity resolution for bench BER testing (the data path does the
    /// equivalent in `hdlc::RotationResolvingDeframer`). The checked bits are
    /// the DECODED data bits, so the BERT reports post-FEC error rate.
    fec: SoftDecoderBank,
    /// The PRBS bank is stopped while something else (the HDLC deframer) has
    /// the link; see `set_bert_enabled`.
    bert_enabled: bool,
    decoded: Vec<Vec<u8>>,
    berts: Vec<RxBert>,
    /// Hardware-rate IQ pending enough samples for one full resampler input
    /// chunk - the RX side is push-paced, so `process` accepts hardware
    /// reads of any length and buffers here.
    input_fifo: VecDeque<Complex32>,
    /// The last `RECENT_SYMBOLS` carrier-derotated, amplitude-normalized
    /// symbols (oldest first), for the constellation display.
    recent: VecDeque<Complex32>,
}

impl RxSignalProcessor {
    /// `hardware_sample_rate`/`symbol_rate_hz`/`sps` together fix the RX
    /// resampler's ratio. `dc_cutoff_hz` removes LO leakage at DC; `freq_shift_hz`
    /// must equal the TX's offset-tuning shift (see `shift_phase`). `carrier_loop_bandwidth_hz` configures the
    /// symbol-rate carrier loop; `timing_loop_bandwidth` (cycles/symbol) the
    /// Gardner loop. `resampler_input_chunk_frames` sizes the RX resampler's
    /// internal processing chunk (independent of the caller's read size).
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        pattern: PrbsPattern,
        modulation: Modulation,
        symbol_rate_hz: f64,
        hardware_sample_rate: f64,
        sps: usize,
        rolloff: f64,
        span_symbols: usize,
        dc_cutoff_hz: f64,
        freq_shift_hz: f64,
        carrier_loop_bandwidth_hz: f64,
        carrier_damping: f64,
        timing_loop_bandwidth: f64,
        timing_damping: f64,
        resampler_input_chunk_frames: usize,
        max_resample_ratio_relative: f64,
    ) -> Self {
        let symbol_domain_rate = symbol_rate_hz * sps as f64;
        let resample_ratio = symbol_domain_rate / hardware_sample_rate;
        let carrier = CarrierLoop::new(
            modulation,
            symbol_rate_hz,
            carrier_loop_bandwidth_hz,
            carrier_damping,
        );
        let rate_scale = symbol_rate_hz / TUNED_SYMBOL_RATE_HZ;
        let phase_scale = carrier.phase_error_scale();
        RxSignalProcessor {
            modulation,
            max_search_offset_hz: (MAX_SEARCH_OFFSET_HZ * rate_scale).max(MIN_SEARCH_RANGE_HZ),
            min_retune_hz: MIN_RETUNE_HZ * rate_scale,
            locked_phase_error: LOCKED_PHASE_ERROR * phase_scale,
            unlock_phase_error: UNLOCK_PHASE_ERROR * phase_scale,
            // 0.8 of the carrier loop's capture range; 0.45 of the alias lattice spacing.
            max_locked_residual_hz: 0.8 * carrier.max_frequency_hz(),
            max_alias_disagreement_hz: 0.45 * carrier.alias_spacing_hz(),
            dc_block: DcBlocker::from_cutoff_hz(dc_cutoff_hz, hardware_sample_rate),
            shift_phase: 0.0,
            shift_phase_increment: std::f64::consts::TAU * freq_shift_hz / hardware_sample_rate,
            base_shift_hz: freq_shift_hz,
            hardware_sample_rate,
            search_enabled: true,
            search_offset_hz: 0.0,
            symbol_rate_hz,
            rolloff,
            symbol_domain_rate,
            acquiring: false,
            verifying: false,
            acq_buf: Vec::new(),
            carrier_locked: false,
            symbols_since_verify: 0,
            symbols_in_state: 0,
            unlocked_run: 0,
            resampler: RxResampler::new(
                resample_ratio,
                resampler_input_chunk_frames,
                max_resample_ratio_relative,
            ),
            matched_filter: MatchedFilter::new(sps, rolloff, span_symbols),
            agc: Agc::new(AGC_RATE),
            timing: TimingRecovery::new(sps as f64, timing_loop_bandwidth, timing_damping),
            carrier,
            fec: SoftDecoderBank::new(modulation),
            bert_enabled: true,
            decoded: vec![Vec::new(); num_branches(modulation)],
            berts: (0..num_branches(modulation))
                .map(|_| RxBert::new(pattern))
                .collect(),
            input_fifo: VecDeque::new(),
            recent: VecDeque::with_capacity(RECENT_SYMBOLS),
        }
    }

    /// Feed hardware-rate IQ samples (any length). Returns the derotated
    /// complex symbols produced by this call (zero or more), each of which
    /// has also been demapped under all 8 rotation hypotheses and clocked
    /// into the embedded BERT bank.
    pub fn process(&mut self, hw_iq: &[ComplexI16]) -> Vec<Complex32> {
        self.input_fifo.extend(complex_i16_to_f32(hw_iq));

        let chunk_frames = self.resampler.input_chunk_frames();
        let mut symbols = Vec::new();
        while self.input_fifo.len() >= chunk_frames {
            let chunk: Vec<Complex32> = self.input_fifo.drain(..chunk_frames).collect();

            let blocked: Vec<Complex32> = chunk
                .iter()
                .map(|&x| {
                    let x = self.dc_block.process(x);
                    let (sin_p, cos_p) = self.shift_phase.sin_cos();
                    self.shift_phase =
                        (self.shift_phase + self.shift_phase_increment) % std::f64::consts::TAU;
                    x * Complex32::new(cos_p as f32, -sin_p as f32)
                })
                .collect();
            let interleaved: Vec<f32> = blocked.iter().flat_map(|c| [c.re, c.im]).collect();
            let resampled = self.resampler.process(&interleaved);

            for pair in resampled.chunks_exact(2) {
                let symbol_domain = Complex32::new(pair[0], pair[1]);
                if self.acquiring {
                    self.acq_buf.push(symbol_domain);
                    if self.acq_buf.len() >= ACQ_SAMPLES {
                        self.coarse_retune();
                    }
                }
                let matched = self.matched_filter.process(symbol_domain);
                let normalized = self.agc.process(matched);
                if let Some(y) = self.timing.process(normalized) {
                    let derotated = self.carrier.process(y);
                    self.update_search();
                    for d in &mut self.decoded {
                        d.clear();
                    }
                    if self.bert_enabled {
                        self.fec.process_symbol(derotated, &mut self.decoded);
                        for (bert, bits) in self.berts.iter_mut().zip(&self.decoded) {
                            for &bit in bits {
                                bert.process_bit(bit);
                            }
                        }
                        self.narrow_or_widen_bert_bank();
                    }
                    if self.recent.len() == RECENT_SYMBOLS {
                        self.recent.pop_front();
                    }
                    self.recent.push_back(derotated);
                    symbols.push(derotated);
                }
            }
        }
        symbols
    }

    /// Once a PRBS checker has synced, only its branch needs decoding; if it
    /// loses sync, every branch is searched again (restarted fresh).
    fn narrow_or_widen_bert_bank(&mut self) {
        if self.fec.is_restricted() {
            let still_synced =
                (0..self.berts.len()).any(|b| self.fec.is_active(b) && self.berts[b].is_synced());
            if !still_synced {
                for b in self.fec.activate_all() {
                    self.berts[b].restart_search();
                }
            }
        } else if let Some(b) = self.berts.iter().position(RxBert::is_synced) {
            self.fec.restrict_to(b);
        }
    }

    /// Stop or restart the PRBS checkers (they are only useful when the link
    /// carries a PRBS; running 2-16 Viterbi decoders for nothing costs CPU).
    /// Restarting searches all branches afresh.
    pub fn set_bert_enabled(&mut self, enabled: bool) {
        if enabled == self.bert_enabled {
            return;
        }
        self.bert_enabled = enabled;
        if enabled {
            for b in self.fec.activate_all() {
                self.berts[b].restart_search();
            }
        } else {
            self.fec.deactivate_all();
        }
    }

    pub fn bert_enabled(&self) -> bool {
        self.bert_enabled
    }

    /// Lock / acquisition state machine, advanced once per symbol.
    fn update_search(&mut self) {
        self.symbols_in_state = self.symbols_in_state.saturating_add(1);
        if self.carrier_locked {
            let looks_unlocked = self.carrier.mean_abs_phase_error() > self.unlock_phase_error
                || !self.carrier.is_locked(f64::MAX);
            self.unlocked_run = if looks_unlocked {
                self.unlocked_run + 1
            } else {
                0
            };
            if self.unlocked_run > UNLOCK_SYMBOLS {
                self.carrier_locked = false;
                self.symbols_in_state = 0;
                self.acquiring = false;
                self.verifying = false;
                self.acq_buf.clear();
            } else if !self.acquiring {
                self.symbols_since_verify = self.symbols_since_verify.saturating_add(1);
                if self.search_enabled && self.symbols_since_verify >= REVERIFY_SYMBOLS {
                    self.verifying = true;
                    self.acquiring = true;
                    self.acq_buf.clear();
                }
            }
        } else if self.symbols_in_state >= SETTLE_SYMBOLS && !self.acquiring {
            let apparent_lock = self.carrier.is_locked(self.locked_phase_error)
                && self.carrier.frequency_hz().abs() < self.max_locked_residual_hz;
            if apparent_lock && !self.search_enabled {
                self.carrier_locked = true;
                self.unlocked_run = 0;
            } else if apparent_lock || self.search_enabled {
                // Either an apparent lock that must be verified against the
                // spectrum (it may be a 2 kHz-lattice alias), or no lock and
                // a signal to look for. Both need a spectrum measurement.
                self.verifying = apparent_lock;
                self.acquiring = true;
                self.acq_buf.clear();
            }
        }
    }

    /// Called when `ACQ_SAMPLES` have been collected: measure the spectral
    /// centre. If searching, retune the LO onto a clearly present signal.
    /// If verifying an apparent carrier lock, accept it only when the carrier
    /// loop's own frequency agrees with the spectrum (an alias would be off by
    /// a multiple of 2 kHz); otherwise it is a false lock - retune instead.
    fn coarse_retune(&mut self) {
        self.acquiring = false;
        let verifying = std::mem::take(&mut self.verifying);
        let estimate = coarse_freq::estimate_offset(
            &self.acq_buf,
            self.symbol_domain_rate,
            self.symbol_rate_hz,
            self.rolloff,
            self.max_search_offset_hz,
            ACQ_FFT_SIZE,
        );
        self.acq_buf.clear();
        self.symbols_in_state = 0;
        // `est.offset_hz` is relative to the CURRENT LO (the samples were
        // taken after it), so it is also the correction to apply.
        let confident = estimate.as_ref().filter(|e| e.contrast >= MIN_ACQ_CONTRAST);
        if verifying {
            let consistent = confident.is_none_or(|e| {
                (e.offset_hz - self.carrier.frequency_hz()).abs() < self.max_alias_disagreement_hz
            });
            if consistent && self.carrier.is_locked(self.locked_phase_error) {
                self.carrier_locked = true;
                self.unlocked_run = 0;
                self.symbols_since_verify = 0;
                return;
            }
            // Either never verified or no longer consistent: not a trustworthy
            // lock (this is how a mid-session frequency step onto a 2 kHz
            // alias is caught).
            self.carrier_locked = false;
        }
        if let Some(e) = confident {
            if e.offset_hz.abs() >= self.min_retune_hz || verifying {
                let new_offset = (self.search_offset_hz + e.offset_hz)
                    .clamp(-self.max_search_offset_hz, self.max_search_offset_hz);
                self.set_search_offset(new_offset);
            }
        }
    }

    fn set_search_offset(&mut self, offset_hz: f64) {
        self.search_offset_hz = offset_hz;
        self.shift_phase_increment =
            std::f64::consts::TAU * (self.base_shift_hz + offset_hz) / self.hardware_sample_rate;
        // Stale frequency from before the retune would offset the new one.
        self.carrier.reset_frequency();
    }

    /// Change the nominal TX offset-tuning shift the receiver removes (must
    /// match the transmitter's). Resets the acquisition offset to 0 and drops
    /// the current lock so it is re-established on the new frequency.
    pub fn set_freq_shift_hz(&mut self, freq_shift_hz: f64) {
        self.base_shift_hz = freq_shift_hz;
        self.set_search_offset(0.0);
        self.carrier_locked = false;
        self.symbols_since_verify = 0;
        self.symbols_in_state = 0;
        self.acquiring = false;
        self.verifying = false;
        self.acq_buf.clear();
    }

    /// Throw away all tracking state and start acquiring again: timing and
    /// carrier loops back to their initial state, search offset back to the
    /// nominal shift, lock dropped. Loop bandwidths and the search-enabled
    /// setting are kept.
    pub fn reacquire(&mut self) {
        self.timing.reset();
        self.carrier.reset();
        self.set_search_offset(0.0);
        self.carrier_locked = false;
        self.symbols_since_verify = 0;
        self.symbols_in_state = 0;
        self.unlocked_run = 0;
        self.acquiring = false;
        self.verifying = false;
        self.acq_buf.clear();
    }

    /// Enable/disable the frequency acquisition. Disabling also returns the LO to
    /// the nominal shift (offset 0).
    pub fn set_search_enabled(&mut self, enabled: bool) {
        self.search_enabled = enabled;
        if !enabled {
            self.acquiring = false;
            self.acq_buf.clear();
            self.set_search_offset(0.0);
        }
    }

    pub fn search_enabled(&self) -> bool {
        self.search_enabled
    }

    /// Current acquisition LO offset (Hz) relative to the nominal shift.
    pub fn search_offset_hz(&self) -> f64 {
        self.search_offset_hz
    }

    /// True once the carrier loop has locked to modulated data (and not since
    /// lost it); false while searching.
    pub fn carrier_locked(&self) -> bool {
        self.carrier_locked
    }

    /// Best estimate of the total transmitter frequency offset from the
    /// nominal: search LO offset plus the carrier loop's residual.
    pub fn total_offset_hz(&self) -> f64 {
        self.search_offset_hz + self.carrier.frequency_hz()
    }

    /// Symbol-error standard deviation (relative to symbol amplitude) and the
    /// SNR derived from it - `None` unless locked, since both are
    /// meaningless on noise.
    pub fn symbol_error_std(&self) -> Option<f64> {
        self.carrier_locked.then(|| self.carrier.symbol_error_std())
    }

    pub fn snr_db(&self) -> Option<f64> {
        self.carrier_locked.then(|| self.carrier.snr_db())
    }

    pub fn bert_control(&mut self, cmd: RxBertControl) {
        for bert in &mut self.berts {
            bert.handle_control(cmd);
        }
    }

    fn synced_branch(&self) -> Option<usize> {
        self.berts
            .iter()
            .position(|b| b.status().locked_state == LockState::Synced)
    }

    /// The rotation hypothesis (0..M, in 360/M degree steps) whose BERT is
    /// currently `Synced`, if any - i.e. the carrier loop's present phase
    /// ambiguity as resolved by the PRBS pattern.
    pub fn bert_rotation(&self) -> Option<u8> {
        self.synced_branch()
            .map(|b| branch_rotation(self.modulation, b))
    }

    /// Which coded bit (0 or 1 of a pair) the synced decoder treats as the
    /// start of a code pair.
    pub fn bert_pair_phase(&self) -> Option<u8> {
        self.synced_branch()
            .map(|b| branch_pair_phase(self.modulation, b))
    }

    /// Status of the BERT that matters: the synced hypothesis if there is
    /// one, otherwise branch 0's (which will be in `Search`).
    pub fn bert_status(&self) -> RxBertStatus {
        self.berts[self.synced_branch().unwrap_or(0)].status()
    }

    /// Raw (pre-FEC) channel bit error rate measured by the synced decoder
    /// re-encoding its output and comparing with what was received; `None`
    /// while the BERT isn't synced.
    pub fn pre_fec_ber(&self) -> Option<f64> {
        self.synced_branch().map(|b| self.fec.recent_channel_ber(b))
    }

    /// The most recent derotated symbols (up to `RECENT_SYMBOLS`, oldest
    /// first) - for plotting the received constellation.
    pub fn recent_symbols(&self) -> Vec<Complex32> {
        self.recent.iter().copied().collect()
    }

    pub fn modulation(&self) -> Modulation {
        self.modulation
    }

    pub fn carrier_frequency_hz(&self) -> f64 {
        self.carrier.frequency_hz()
    }

    pub fn carrier_phase_radians(&self) -> f64 {
        self.carrier.phase_radians()
    }

    /// Smoothed mean |phase error| of the carrier loop (radians) - ~0 when
    /// locked, ~pi/16 when not.
    pub fn carrier_phase_error(&self) -> f64 {
        self.carrier.mean_abs_phase_error()
    }

    /// Retune the carrier loop bandwidth at runtime (capture range vs noise
    /// trade-off) - exposed for live MQTT control.
    pub fn set_carrier_loop_bandwidth_hz(&mut self, bandwidth_hz: f64) {
        self.carrier.set_loop_bandwidth_hz(bandwidth_hz);
    }

    pub fn timing_omega(&self) -> f64 {
        self.timing.omega()
    }

    /// Retune the timing recovery loop bandwidth at runtime.
    pub fn set_timing_loop_bandwidth(&mut self, normalized_bw: f64) {
        self.timing.set_loop_bandwidth(normalized_bw);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::params::*;
    use crate::rrc::ShapeMode;
    use crate::tx_signal::TxSignalGenerator;

    fn make_tx(freq_shift_hz: f64) -> TxSignalGenerator {
        TxSignalGenerator::new(
            PrbsPattern::Pn15,
            Modulation::Psk8,
            SYMBOL_RATE_HZ,
            HARDWARE_SAMPLE_RATE,
            SPS,
            RRC_ROLLOFF,
            RRC_SPAN_SYMBOLS,
            ShapeMode::Rrc,
            4096,
            10.0,
            1.0,
            freq_shift_hz,
        )
    }

    fn make_rx(rx_shift_hz: f64) -> RxSignalProcessor {
        RxSignalProcessor::new(
            PrbsPattern::Pn15,
            Modulation::Psk8,
            SYMBOL_RATE_HZ,
            HARDWARE_SAMPLE_RATE as f64,
            SPS,
            RRC_ROLLOFF,
            RRC_SPAN_SYMBOLS,
            50.0, // dc_cutoff_hz
            rx_shift_hz,
            100.0, // carrier_loop_bandwidth_hz
            0.707, // carrier_damping
            0.02,  // timing_loop_bandwidth
            0.707, // timing_damping
            4096,  // resampler_input_chunk_frames
            10.0,  // max_resample_ratio_relative
        )
    }

    /// Rotate hardware IQ by a constant carrier phase (and add nothing else):
    /// models an arbitrary unknown channel phase.
    fn rotate(chunk: &[ComplexI16], phase: f64) -> Vec<ComplexI16> {
        let (s, c) = phase.sin_cos();
        chunk
            .iter()
            .map(|x| {
                let (i, q) = (f64::from(x.re), f64::from(x.im));
                ComplexI16::new(
                    (i * c - q * s).round() as i16,
                    (i * s + q * c).round() as i16,
                )
            })
            .collect()
    }

    /// TX shifts by `tx_shift_hz`; RX un-shifts by `rx_shift_hz` - any
    /// difference is a residual carrier offset for the carrier loop to track.
    fn run_link(
        tx_shift_hz: f64,
        rx_shift_hz: f64,
        carrier_phase: f64,
        chunks: usize,
    ) -> RxSignalProcessor {
        let mut tx = make_tx(tx_shift_hz);
        let mut rx = make_rx(rx_shift_hz);
        for _ in 0..chunks {
            let hw = rotate(&tx.next_chunk(), carrier_phase);
            rx.process(&hw);
        }
        rx
    }

    // ---- modulation / symbol-rate coverage ----

    fn make_tx_for(m: Modulation, rate: f64, freq_shift_hz: f64) -> TxSignalGenerator {
        TxSignalGenerator::new(
            PrbsPattern::Pn15,
            m,
            rate,
            HARDWARE_SAMPLE_RATE,
            SPS,
            RRC_ROLLOFF,
            RRC_SPAN_SYMBOLS,
            ShapeMode::Rrc,
            4096,
            10.0,
            1.0,
            freq_shift_hz,
        )
    }

    fn make_rx_for(m: Modulation, rate: f64, rx_shift_hz: f64) -> RxSignalProcessor {
        RxSignalProcessor::new(
            PrbsPattern::Pn15,
            m,
            rate,
            HARDWARE_SAMPLE_RATE as f64,
            SPS,
            RRC_ROLLOFF,
            RRC_SPAN_SYMBOLS,
            50.0,
            rx_shift_hz,
            100.0 * rate / SYMBOL_RATE_HZ,
            0.707,
            0.02,
            0.707,
            4096,
            10.0,
        )
    }

    /// Shift that clears the occupied band at `rate` (the default 15 kHz
    /// suffices at the default rate).
    fn shift_for(rate: f64) -> f64 {
        (occupied_bandwidth_hz(rate) / 2.0 + 4_000.0).max(15_000.0)
    }

    /// Run `seconds` of PRBS link; TX is `offset_hz` away from the RX's expectation.
    fn run_mod_link(
        m: Modulation,
        rate: f64,
        offset_hz: f64,
        carrier_phase: f64,
        seconds: f64,
    ) -> RxSignalProcessor {
        let shift = shift_for(rate);
        let mut tx = make_tx_for(m, rate, shift + offset_hz);
        let mut rx = make_rx_for(m, rate, shift);
        let chunks = (seconds * f64::from(HARDWARE_SAMPLE_RATE) / 4096.0).ceil() as usize;
        for _ in 0..chunks {
            let hw = rotate(&tx.next_chunk(), carrier_phase);
            rx.process(&hw);
        }
        rx
    }

    #[test]
    fn every_modulation_and_symbol_rate_links_with_zero_errors() {
        for m in Modulation::ALL {
            for rate in [4_000.0, 16_000.0, 64_000.0] {
                for phase in [0.3, 2.2] {
                    // Acquisition takes a fixed number of symbols; give slow links longer.
                    let seconds = 1.0 + 60_000.0 / rate;
                    let rx = run_mod_link(m, rate, 0.0, phase, seconds);
                    let st = rx.bert_status();
                    assert!(
                        rx.bert_rotation().is_some(),
                        "{m:?} @ {rate}: no BERT lock (locked {}, carrier err {:.3} rad, freq {:.1} Hz)",
                        rx.carrier_locked(),
                        rx.carrier_phase_error(),
                        rx.carrier_frequency_hz()
                    );
                    assert_eq!(
                        st.bit_errors_received, 0,
                        "{m:?} @ {rate} phase {phase}: errors on a clean link"
                    );
                    assert!(
                        st.bits_received > 1_000,
                        "{m:?} @ {rate}: only {} bits checked",
                        st.bits_received
                    );
                }
            }
        }
    }

    /// As `run_mod_link`, at the sample rate and SPS the modem would pick for `rate`.
    fn run_profile_link(
        m: Modulation,
        rate: f64,
        offset_hz: f64,
        symbols: f64,
    ) -> (RxSignalProcessor, u32) {
        let hw = hardware_sample_rate_for(rate);
        let sps = sps_for(rate, hw);
        let shift = shift_for(rate);
        let mut tx = TxSignalGenerator::new(
            PrbsPattern::Pn15,
            m,
            rate,
            hw,
            sps,
            RRC_ROLLOFF,
            RRC_SPAN_SYMBOLS,
            ShapeMode::Rrc,
            chunk_frames_for(hw),
            10.0,
            1.0,
            shift + offset_hz,
        );
        let mut rx = RxSignalProcessor::new(
            PrbsPattern::Pn15,
            m,
            rate,
            f64::from(hw),
            sps,
            RRC_ROLLOFF,
            RRC_SPAN_SYMBOLS,
            50.0,
            shift,
            100.0 * rate / SYMBOL_RATE_HZ,
            0.707,
            0.02,
            0.707,
            chunk_frames_for(hw),
            10.0,
        );
        let chunks = (symbols * f64::from(hw) / rate / chunk_frames_for(hw) as f64).ceil() as usize;
        for _ in 0..chunks {
            rx.process(&tx.next_chunk());
        }
        (rx, hw)
    }

    #[test]
    fn the_high_sample_rate_profile_links_from_150_ksym_to_1_msym() {
        for (m, rate) in [
            (Modulation::Bpsk, 150_000.0),
            (Modulation::Qpsk, 150_000.0),
            (Modulation::Psk8, 150_000.0),
            (Modulation::Bpsk, 500_000.0),
            (Modulation::Qpsk, 500_000.0),
            (Modulation::Bpsk, 1_000_000.0),
            (Modulation::Qpsk, 1_000_000.0),
            // 4 MSPS / (8 sps x rate) = 2 and 4: the polyphase decimator; 1: pass-through.
            (Modulation::Qpsk, 250_000.0),
            (Modulation::Psk8, 125_000.0),
            (Modulation::Psk8, 500_000.0),
        ] {
            let (rx, hw) = run_profile_link(m, rate, 0.0, 150_000.0);
            assert_eq!(hw, HIGH_HARDWARE_SAMPLE_RATE);
            assert!(
                rx.bert_rotation().is_some(),
                "{m:?} @ {rate} ({} sps): no BERT lock (locked {}, carrier err {:.3} rad, freq {:.1} Hz)",
                sps_for(rate, hw),
                rx.carrier_locked(),
                rx.carrier_phase_error(),
                rx.carrier_frequency_hz()
            );
            assert_eq!(
                rx.bert_status().bit_errors_received,
                0,
                "{m:?} @ {rate}: errors on a clean link"
            );
            assert!(
                rx.bert_status().bits_received > 1_000,
                "{m:?} @ {rate}: only {} bits checked",
                rx.bert_status().bits_received
            );
        }
    }

    #[test]
    fn integer_ratio_fast_paths_link_in_the_standard_profile_too() {
        // 800 kSPS / (8 sps x rate) = 1, 2, 4, 5.
        for (m, rate) in [
            (Modulation::Qpsk, 100_000.0),
            (Modulation::Bpsk, 50_000.0),
            (Modulation::Psk8, 25_000.0),
            (Modulation::Qpsk, 20_000.0),
        ] {
            let (rx, hw) = run_profile_link(m, rate, 0.0, 80_000.0);
            assert_eq!(hw, HARDWARE_SAMPLE_RATE);
            assert!(
                rx.bert_rotation().is_some(),
                "{m:?} @ {rate}: no BERT lock (locked {})",
                rx.carrier_locked()
            );
            assert_eq!(rx.bert_status().bit_errors_received, 0, "{m:?} @ {rate}");
        }
    }

    #[test]
    fn the_slowest_rates_still_acquire_a_crystal_sized_offset() {
        // 1.5 kHz is a ~3.4 ppm error at 440 MHz - many times the symbol rate here.
        for (m, rate, offset) in [
            (Modulation::Qpsk, 2_000.0, 1_500.0),
            (Modulation::Bpsk, 2_000.0, -1_500.0),
            (Modulation::Psk8, 4_000.0, 1_200.0),
        ] {
            let (rx, hw) = run_profile_link(m, rate, offset, 120_000.0);
            assert_eq!(hw, HARDWARE_SAMPLE_RATE);
            assert!(
                rx.carrier_locked(),
                "{m:?} @ {rate} offset {offset}: never locked"
            );
            assert!(rx.bert_rotation().is_some(), "{m:?} @ {rate}: no BERT lock");
            assert!(
                (rx.total_offset_hz() - offset).abs() < 60.0,
                "{m:?} @ {rate}: estimated {} for {offset}",
                rx.total_offset_hz()
            );
        }
    }

    #[test]
    fn the_high_profile_acquires_a_frequency_offset() {
        // 0.1 x the symbol rate off - within the +-0.375 x rate search range.
        for (m, rate) in [(Modulation::Qpsk, 250_000.0), (Modulation::Bpsk, 500_000.0)] {
            let offset = 0.1 * rate;
            let (rx, _) = run_profile_link(m, rate, offset, 300_000.0);
            assert!(rx.carrier_locked(), "{m:?} @ {rate}: never locked");
            assert!(rx.bert_rotation().is_some(), "{m:?} @ {rate}: no BERT lock");
            assert!(
                (rx.total_offset_hz() - offset).abs() < 0.01 * rate,
                "{m:?} @ {rate}: estimated {} for {offset}",
                rx.total_offset_hz()
            );
        }
    }

    /// Throughput benchmark (run: cargo test --release --lib bench_rx -- --ignored --nocapture):
    /// how many times faster than real time the receiver runs, per modulation/rate,
    /// for the DSP+PRBS bank and for the HDLC deframer separately.
    #[test]
    #[ignore]
    fn bench_rx_throughput() {
        use crate::ax25_signal::Ax25RxPipeline;
        use std::time::Instant;
        for (m, rate) in [
            (Modulation::Psk8, 16_000.0),
            (Modulation::Psk8, 64_000.0),
            (Modulation::Psk8, 150_000.0),
            (Modulation::Qpsk, 500_000.0),
            (Modulation::Bpsk, 1_000_000.0),
            (Modulation::Psk8, 500_000.0),
        ]
        .into_iter()
        .enumerate()
        .filter(|(i, _)| {
            std::env::var("BENCH_CASE")
                .ok()
                .and_then(|v| v.parse::<usize>().ok())
                .is_none_or(|c| c == *i)
        })
        .map(|(_, c)| c)
        {
            let hw = hardware_sample_rate_for(rate);
            let sps = sps_for(rate, hw);
            let shift = shift_for(rate);
            let mut tx = TxSignalGenerator::new(
                PrbsPattern::Pn15,
                m,
                rate,
                hw,
                sps,
                RRC_ROLLOFF,
                RRC_SPAN_SYMBOLS,
                ShapeMode::Rrc,
                chunk_frames_for(hw),
                10.0,
                1.0,
                shift,
            );
            let mut rx = RxSignalProcessor::new(
                PrbsPattern::Pn15,
                m,
                rate,
                f64::from(hw),
                sps,
                RRC_ROLLOFF,
                RRC_SPAN_SYMBOLS,
                50.0,
                shift,
                100.0 * rate / SYMBOL_RATE_HZ,
                0.707,
                0.02,
                0.707,
                chunk_frames_for(hw),
                10.0,
            );
            let seconds = (200_000.0 / rate).max(0.25);
            let chunks: Vec<_> = (0
                ..(seconds * f64::from(hw) / chunk_frames_for(hw) as f64).ceil() as usize)
                .map(|_| tx.next_chunk())
                .collect();
            let signal_s = chunks.len() as f64 * chunk_frames_for(hw) as f64 / f64::from(hw);
            let t = Instant::now();
            let mut symbols = Vec::new();
            for c in &chunks {
                symbols.extend(rx.process(c));
            }
            let rx_s = t.elapsed().as_secs_f64();
            let mut pipe = Ax25RxPipeline::new(m);
            let t = Instant::now();
            pipe.process(&symbols);
            let hdlc_s = t.elapsed().as_secs_f64();
            println!(
                "{m:?} @ {rate:>9} sym/s ({hw} SPS, sps {sps}): DSP+BERT bank {:6.2}x real time | HDLC bank {:6.2}x | together {:6.2}x",
                signal_s / rx_s,
                signal_s / hdlc_s,
                signal_s / (rx_s + hdlc_s)
            );
        }
    }

    /// As `bench_rx_throughput`, for a data link (idle flags + a few frames): the
    /// HDLC bank once it has locked and narrowed to its one branch.
    #[test]
    #[ignore]
    fn bench_rx_data_mode() {
        use crate::ax25_signal::{Ax25RxPipeline, Ax25TxSignalGenerator};
        use crate::kiss::KissServer;
        use std::io::Write;
        use std::sync::Arc;
        use std::time::Instant;
        for (m, rate) in [
            (Modulation::Psk8, 16_000.0),
            (Modulation::Psk8, 150_000.0),
            (Modulation::Qpsk, 500_000.0),
            (Modulation::Psk8, 500_000.0),
            (Modulation::Bpsk, 1_000_000.0),
        ] {
            let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
            let addr = listener.local_addr().unwrap();
            drop(listener);
            let kiss = Arc::new(KissServer::start(&addr.to_string()).unwrap());
            std::thread::sleep(std::time::Duration::from_millis(50));
            let mut client = std::net::TcpStream::connect(addr).unwrap();
            std::thread::sleep(std::time::Duration::from_millis(50));
            for k in 0..4 {
                client
                    .write_all(&crate::kiss::encode(
                        format!("bench frame number {k}").as_bytes(),
                    ))
                    .unwrap();
            }
            std::thread::sleep(std::time::Duration::from_millis(100));

            let hw = hardware_sample_rate_for(rate);
            let sps = sps_for(rate, hw);
            let shift = shift_for(rate);
            let mut tx = Ax25TxSignalGenerator::new(
                kiss,
                m,
                rate,
                hw,
                sps,
                RRC_ROLLOFF,
                RRC_SPAN_SYMBOLS,
                chunk_frames_for(hw),
                10.0,
                1.0,
                shift,
            );
            let mut rx = RxSignalProcessor::new(
                PrbsPattern::Pn15,
                m,
                rate,
                f64::from(hw),
                sps,
                RRC_ROLLOFF,
                RRC_SPAN_SYMBOLS,
                50.0,
                shift,
                100.0 * rate / SYMBOL_RATE_HZ,
                0.707,
                0.02,
                0.707,
                chunk_frames_for(hw),
                10.0,
            );
            rx.set_bert_enabled(false); // the engine stops it once the HDLC bank has the link
            let chunk_s = chunk_frames_for(hw) as f64 / f64::from(hw);
            let n_chunks = ((600_000.0 / rate / chunk_s).ceil() as usize).max(60);
            let mut pipe = Ax25RxPipeline::new(m);
            let mut frames = 0;
            let (mut timed_symbols, mut timed) = (0usize, 0.0f64);
            let mut dsp = 0.0f64;
            for k in 0..n_chunks {
                let chunk = tx.next_chunk();
                let t = Instant::now();
                let symbols = rx.process(&chunk);
                dsp += t.elapsed().as_secs_f64();
                let t = Instant::now();
                frames += pipe.process(&symbols).len();
                if pipe.locked_rotation().is_some() && k > n_chunks / 2 {
                    timed += t.elapsed().as_secs_f64();
                    timed_symbols += symbols.len();
                }
            }
            let hdlc_rt = timed_symbols as f64 / rate / timed;
            println!(
                "{m:?} @ {rate:>9}: {frames} frames, locked rot {:?}; DSP {:5.2}x real time | HDLC bank (locked) {:7.1}x",
                pipe.locked_rotation(),
                n_chunks as f64 * chunk_s / dsp,
                hdlc_rt
            );
        }
    }

    #[test]
    fn rotation_hypotheses_cover_every_phase_for_bpsk_and_qpsk() {
        for m in [Modulation::Bpsk, Modulation::Qpsk] {
            let mut seen = std::collections::BTreeSet::new();
            for phase in [0.0f64, 0.8, 1.7, 2.4, 3.3, -1.2, -2.0] {
                let rx = run_mod_link(m, 16_000.0, 0.0, phase, 4.0);
                let rot = rx
                    .bert_rotation()
                    .unwrap_or_else(|| panic!("{m:?} phase {phase}: no lock"));
                assert!(u16::from(rot) < u16::from(m.num_points()));
                assert_eq!(
                    rx.bert_status().bit_errors_received,
                    0,
                    "{m:?} phase {phase}"
                );
                seen.insert(rot);
            }
            assert!(seen.len() >= 2, "{m:?}: only rotation {seen:?} ever seen");
        }
    }

    #[test]
    fn acquisition_works_for_bpsk_and_qpsk_off_frequency() {
        for m in [Modulation::Bpsk, Modulation::Qpsk] {
            for offset in [2_500.0f64, -3_300.0, 5_000.0, 700.0] {
                let rx = run_mod_link(m, 16_000.0, offset, 0.4, 6.0);
                assert!(rx.carrier_locked(), "{m:?} offset {offset}: never locked");
                assert!(
                    rx.bert_rotation().is_some(),
                    "{m:?} offset {offset}: no BERT lock"
                );
                assert!(
                    rx.bert_status().bit_errors_received < 50,
                    "{m:?} offset {offset}: {} errors",
                    rx.bert_status().bit_errors_received
                );
                assert!(
                    (rx.total_offset_hz() - offset).abs() < 40.0,
                    "{m:?} offset {offset}: estimated {}",
                    rx.total_offset_hz()
                );
            }
        }
    }

    #[test]
    fn full_chain_recovers_the_prbs_for_any_carrier_phase() {
        // The carrier loop may settle on any of 8 rotations; the BERT bank
        // must find the right hypothesis whichever it is. Phases are chosen
        // to cover different 45-degree sectors.
        let mut rotations_seen = std::collections::BTreeSet::new();
        for phase in [0.0f64, 0.5, 1.2, 2.0, 2.9, -0.8, -1.9, -2.7] {
            let rx = run_link(15_000.0, 15_000.0, phase, 250);
            let rotation = rx.bert_rotation().unwrap_or_else(|| {
                panic!("phase {phase}: no BERT hypothesis locked (carrier err {:.3} rad, freq {:.1} Hz)", rx.carrier_phase_error(), rx.carrier_frequency_hz())
            });
            rotations_seen.insert(rotation);
            let status = rx.bert_status();
            assert_eq!(status.locked_state, LockState::Synced);
            assert_eq!(
                status.bit_errors_received, 0,
                "phase {phase}: errors on a clean link"
            );
        }
        assert!(rotations_seen.len() >= 2, "expected the carrier loop to land on several different rotations across phases, saw {rotations_seen:?}");
    }

    #[test]
    fn full_chain_pulls_in_a_residual_carrier_offset() {
        // Two independent oscillators: model +-600 Hz of residual offset.
        for residual_hz in [600.0f64, -600.0, 150.0] {
            let rx = run_link(15_000.0 + residual_hz, 15_000.0, 0.7, 400);
            assert!(
                rx.bert_rotation().is_some(),
                "residual {residual_hz} Hz: no BERT hypothesis locked"
            );
            assert_eq!(
                rx.bert_status().bit_errors_received,
                0,
                "residual {residual_hz} Hz: errors after lock"
            );
            let est = rx.carrier_frequency_hz();
            // Estimated NCO frequency is the residual offset (the receiver
            // derotates by +est to cancel a signal at +residual).
            assert!(
                (est - residual_hz).abs() < 20.0,
                "residual {residual_hz} Hz: estimate {est} Hz"
            );
        }
    }

    /// Gaussian noise via Box-Muller on xorshift32.
    struct Noise(u32);
    impl Noise {
        fn uniform(&mut self) -> f64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 17;
            self.0 ^= self.0 << 5;
            (f64::from(self.0) + 1.0) / (f64::from(u32::MAX) + 2.0)
        }
        fn gauss(&mut self) -> f64 {
            (-2.0 * self.uniform().ln()).sqrt() * (std::f64::consts::TAU * self.uniform()).cos()
        }
    }

    #[test]
    fn full_chain_holds_lock_in_noise() {
        // Es/N0 = 24 dB: well above 8PSK's ~17 dB for 1e-3 BER, so the loops
        // must stay locked and the bit error rate must be tiny. (Noise bandwidth
        // is the full hardware rate; Es/N0 = P_signal * (Fs/Rs) / sigma^2.)
        let es_n0_db = 24.0;
        let mut tx = make_tx(15_000.0);
        let mut rx = make_rx(15_000.0);
        let mut noise = Noise(0x1234_ABCD);
        let chunks: Vec<Vec<ComplexI16>> =
            (0..400).map(|_| rotate(&tx.next_chunk(), 1.1)).collect();
        let power: f64 = chunks
            .iter()
            .flatten()
            .map(|c| (f64::from(c.re).powi(2) + f64::from(c.im).powi(2)))
            .sum::<f64>()
            / chunks.iter().map(Vec::len).sum::<usize>() as f64;
        let sigma2 =
            power * (HARDWARE_SAMPLE_RATE as f64 / SYMBOL_RATE_HZ) / 10f64.powf(es_n0_db / 10.0);
        let per_axis = (sigma2 / 2.0).sqrt();
        for chunk in &chunks {
            let noisy: Vec<ComplexI16> = chunk
                .iter()
                .map(|c| {
                    let i = f64::from(c.re) + per_axis * noise.gauss();
                    let q = f64::from(c.im) + per_axis * noise.gauss();
                    ComplexI16::new(
                        i.round().clamp(-2048.0, 2047.0) as i16,
                        q.round().clamp(-2048.0, 2047.0) as i16,
                    )
                })
                .collect();
            rx.process(&noisy);
        }
        assert!(
            rx.bert_rotation().is_some(),
            "no BERT lock at Es/N0 {es_n0_db} dB"
        );
        let st = rx.bert_status();
        let ber = st.bit_errors_received as f64 / st.bits_received.max(1) as f64;
        assert!(
            ber < 1e-3,
            "BER {ber:.2e} at Es/N0 {es_n0_db} dB ({} errors / {} bits)",
            st.bit_errors_received,
            st.bits_received
        );
        assert!(!st.sync_loss, "BERT lost sync at Es/N0 {es_n0_db} dB");
    }

    fn run_with_offset(offset_hz: f64, search: bool, chunks: usize) -> RxSignalProcessor {
        let mut tx = make_tx(15_000.0 + offset_hz);
        let mut rx = make_rx(15_000.0);
        rx.set_search_enabled(search);
        for _ in 0..chunks {
            rx.process(&tx.next_chunk());
        }
        rx
    }

    #[test]
    fn acquisition_finds_signals_several_khz_off_frequency() {
        for offset in [
            2_500.0f64, -3_300.0, 4_000.0, -1_800.0, 2_000.0, -5_200.0, 5_900.0,
        ] {
            let rx = run_with_offset(offset, true, 700);
            assert!(
                rx.carrier_locked(),
                "offset {offset} Hz: never locked (LO offset {} Hz)",
                rx.search_offset_hz()
            );
            assert!(
                rx.bert_rotation().is_some(),
                "offset {offset} Hz: no BERT hypothesis locked"
            );
            assert!(
                rx.bert_status().bit_errors_received < 50,
                "offset {offset} Hz: {} errors",
                rx.bert_status().bit_errors_received
            );
            let est = rx.total_offset_hz();
            assert!(
                (est - offset).abs() < 30.0,
                "offset {offset} Hz: total offset estimate {est} Hz"
            );
        }
    }

    #[test]
    fn without_the_sweep_a_large_offset_does_not_lock() {
        // Proves the test above is discriminating: same signal, acquisition off.
        let rx = run_with_offset(3_000.0, false, 700);
        assert!(!rx.carrier_locked());
        assert!(rx.bert_rotation().is_none());
    }

    #[test]
    fn acquisition_relocks_after_the_signal_frequency_jumps() {
        let mut rx = make_rx(15_000.0);
        let mut tx = make_tx(15_000.0 + 3_000.0);
        for _ in 0..700 {
            rx.process(&tx.next_chunk());
        }
        assert!(rx.carrier_locked());
        // Transmitter retunes by -6 kHz: the receiver must notice and re-acquire.
        let mut tx = make_tx(15_000.0 - 3_000.0);
        for _ in 0..900 {
            rx.process(&tx.next_chunk());
        }
        assert!(rx.carrier_locked(), "did not re-lock after the jump");
        assert!(
            (rx.total_offset_hz() + 3_000.0).abs() < 30.0,
            "offset estimate {}",
            rx.total_offset_hz()
        );
    }

    #[test]
    fn snr_telemetry_matches_the_injected_noise() {
        let es_n0_db = 22.0;
        let mut tx = make_tx(15_000.0);
        let mut rx = make_rx(15_000.0);
        let mut noise = Noise(0x0BAD_F00D);
        let chunks: Vec<Vec<ComplexI16>> = (0..300).map(|_| tx.next_chunk()).collect();
        let power: f64 = chunks
            .iter()
            .flatten()
            .map(|c| f64::from(c.re).powi(2) + f64::from(c.im).powi(2))
            .sum::<f64>()
            / chunks.iter().map(Vec::len).sum::<usize>() as f64;
        let per_axis = (power * (HARDWARE_SAMPLE_RATE as f64 / SYMBOL_RATE_HZ)
            / 10f64.powf(es_n0_db / 10.0)
            / 2.0)
            .sqrt();
        for chunk in &chunks {
            let noisy: Vec<ComplexI16> = chunk
                .iter()
                .map(|c| {
                    ComplexI16::new(
                        (f64::from(c.re) + per_axis * noise.gauss()).round() as i16,
                        (f64::from(c.im) + per_axis * noise.gauss()).round() as i16,
                    )
                })
                .collect();
            rx.process(&noisy);
        }
        let snr = rx.snr_db().expect("locked, so SNR should be reported");
        assert!(
            (snr - es_n0_db).abs() < 1.5,
            "injected Es/N0 {es_n0_db} dB, telemetry says {snr:.2} dB"
        );
        let std = rx.symbol_error_std().unwrap();
        assert!(
            (std - 10f64.powf(-es_n0_db / 20.0)).abs() < 0.02,
            "error std {std}"
        );
    }

    /// Run the PRBS link through AWGN at `es_n0_db`; returns the receiver.
    fn run_noisy(es_n0_db: f64, chunks_n: usize, seed: u32) -> RxSignalProcessor {
        let mut tx = make_tx(15_000.0);
        let mut rx = make_rx(15_000.0);
        let mut noise = Noise(seed);
        let chunks: Vec<Vec<ComplexI16>> = (0..chunks_n).map(|_| tx.next_chunk()).collect();
        let power: f64 = chunks
            .iter()
            .flatten()
            .map(|c| f64::from(c.re).powi(2) + f64::from(c.im).powi(2))
            .sum::<f64>()
            / chunks.iter().map(Vec::len).sum::<usize>() as f64;
        let per_axis = (power * (HARDWARE_SAMPLE_RATE as f64 / SYMBOL_RATE_HZ)
            / 10f64.powf(es_n0_db / 10.0)
            / 2.0)
            .sqrt();
        for chunk in &chunks {
            let noisy: Vec<ComplexI16> = chunk
                .iter()
                .map(|c| {
                    ComplexI16::new(
                        (f64::from(c.re) + per_axis * noise.gauss()).round() as i16,
                        (f64::from(c.im) + per_axis * noise.gauss()).round() as i16,
                    )
                })
                .collect();
            rx.process(&noisy);
        }
        rx
    }

    #[test]
    fn coding_gain_through_the_full_chain() {
        // At Es/N0 where uncoded 8PSK has a raw BER of ~1e-2..1e-3 the coded
        // link must be essentially clean, with the decoder's own measurement of
        // the raw channel errors showing what it fixed.
        for es_n0_db in [11.0f64, 13.0, 15.0] {
            let rx = run_noisy(es_n0_db, 1_500, 0xFEC0_0001);
            let st = rx.bert_status();
            println!(
                "Es/N0 {es_n0_db:4.1} dB: BERT {:?} rot {:?} | decoded errors {} / {} bits | pre-FEC BER {:?} | SNR est {:?}",
                st.locked_state,
                rx.bert_rotation(),
                st.bit_errors_received,
                st.bits_received,
                rx.pre_fec_ber(),
                rx.snr_db()
            );
        }
        let rx = run_noisy(15.0, 1_500, 0xFEC0_0002);
        let st = rx.bert_status();
        assert!(rx.bert_rotation().is_some(), "no lock at 15 dB");
        assert_eq!(
            st.bit_errors_received, 0,
            "{} errors after FEC at Es/N0 15 dB",
            st.bit_errors_received
        );
        let pre = rx.pre_fec_ber().unwrap();
        assert!(
            pre > 1e-4,
            "at 15 dB the raw channel should show some errors for FEC to fix (measured {pre})"
        );
    }

    #[test]
    fn recent_symbols_holds_the_latest_clean_constellation() {
        let rx = run_link(15_000.0, 15_000.0, 0.4, 300);
        let recent = rx.recent_symbols();
        assert_eq!(recent.len(), RECENT_SYMBOLS);
        // Locked on a clean link: every symbol sits near a unit-magnitude 8PSK point.
        let worst = recent
            .iter()
            .map(|&y| {
                (y - crate::symbol_map::psk8_point(crate::symbol_map::psk8_position(y))).norm()
            })
            .fold(0.0f32, f32::max);
        assert!(worst < 0.35, "worst symbol error {worst}");
    }

    /// Feed `chunks` of TX output (shift `tx_shift_hz`) through AWGN at `es_n0_db`.
    fn feed_noisy(
        rx: &mut RxSignalProcessor,
        tx_shift_hz: f64,
        chunks: usize,
        es_n0_db: f64,
        noise: &mut Noise,
    ) {
        let mut tx = make_tx(tx_shift_hz);
        for _ in 0..chunks {
            let chunk = tx.next_chunk();
            let power: f64 = chunk
                .iter()
                .map(|c| f64::from(c.re).powi(2) + f64::from(c.im).powi(2))
                .sum::<f64>()
                / chunk.len() as f64;
            let per_axis = (power * (HARDWARE_SAMPLE_RATE as f64 / SYMBOL_RATE_HZ)
                / 10f64.powf(es_n0_db / 10.0)
                / 2.0)
                .sqrt();
            let noisy: Vec<ComplexI16> = chunk
                .iter()
                .map(|c| {
                    ComplexI16::new(
                        (f64::from(c.re) + per_axis * noise.gauss()).round() as i16,
                        (f64::from(c.im) + per_axis * noise.gauss()).round() as i16,
                    )
                })
                .collect();
            rx.process(&noisy);
        }
    }

    #[test]
    fn acquisition_catches_an_alias_lock_after_a_small_frequency_step() {
        // Locked at +300 Hz, then the TX steps by +1 kHz. On real hardware the
        // carrier loop followed onto the 2 kHz alias (residual -700 Hz) without
        // ever looking unlocked, and stayed "locked" while decoding garbage.
        // Periodic re-verification against the spectrum must notice and
        // retune onto the true +1300 Hz.
        for seed in 1..=6u32 {
            let mut rx = make_rx(15_000.0);
            let mut noise = Noise(0xA11A_5000 + seed);
            feed_noisy(&mut rx, 15_300.0, 400, 21.0, &mut noise);
            assert!(rx.carrier_locked(), "seed {seed}: initial lock failed");
            feed_noisy(&mut rx, 16_300.0, 1_550, 21.0, &mut noise);
            assert!(
                rx.carrier_locked(),
                "seed {seed}: not locked after the step"
            );
            assert!(
                (rx.total_offset_hz() - 1_300.0).abs() < 40.0,
                "seed {seed}: total offset {} Hz, expected ~1300 (an alias lock reads -700)",
                rx.total_offset_hz()
            );
            assert!(
                rx.bert_rotation().is_some(),
                "seed {seed}: BERT never re-synced after the step"
            );
        }
    }
}
