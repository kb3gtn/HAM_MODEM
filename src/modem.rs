//! The combined BPSK/QPSK/8PSK modem: a transmit engine and a receive engine, each a
//! hardware-paced loop, written against small traits so the same code runs on a
//! real bladeRF (`src/bin/modem.rs`) or on an in-memory mock in tests.
//!
//! Neither engine knows about MQTT or USB:
//!  * the radio is a `TxRadio` / `RxRadio` (write/read IQ, set gain, set LO);
//!  * telemetry goes out through a `Telemetry` sink (the binary maps it onto
//!    MQTT topics `<base>/<suffix>`; tests collect it in a Vec);
//!  * commands arrive on a crossbeam channel of `TxControlMsg` / `RxControlMsg`
//!    (the binary fills it from MQTT), and `shutdown` ends the loop.
//!
//! Both engines share ONE `KissServer`: a KISS host talks to a single TCP
//! socket, sending frames to transmit and receiving the frames the receiver
//! hears. The TX engine pulls outgoing frames from it, the RX engine pushes
//! decoded frames to it.
//!
//! JSON shapes (the MQTT contract with the GUI) are defined by the serde types
//! here; see the field docs.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use bladerf::ComplexI16;
use crossbeam_channel::Receiver;
use serde::{Deserialize, Serialize};

use crate::ax25_signal::{Ax25RxPipeline, Ax25TxSignalGenerator};
use crate::kiss::KissServer;
use crate::params::*;
use crate::prbs::PrbsPattern;
use crate::psd::complex_i16_to_f32;
use crate::rrc::ShapeMode;
use crate::rx_bert::{RxBertControl, RxBertStatus};
use crate::rx_signal::{RxSignalProcessor, RECENT_SYMBOLS};
use crate::symbol_map::Modulation;
use crate::tx_bert::{TxBertControl, TxBertStatus};
use crate::tx_signal::TxSignalGenerator;

// IQ frames per radio write (TX) / read (RX), and per RX resampler chunk, are
// `params::chunk_frames_for(hardware sample rate)`: ~10 ms of samples.
const MAX_RESAMPLE_RATIO_RELATIVE: f64 = 10.0;
const CARRIER_DAMPING: f64 = 0.707;
pub const DEFAULT_CARRIER_BANDWIDTH_HZ: f64 = 100.0;
pub const TIMING_LOOP_BANDWIDTH: f64 = 0.02;
const TIMING_DAMPING: f64 = 0.707;
pub const DEFAULT_DC_CUTOFF_HZ: f64 = 50.0;

/// Do two symbol rates need different radio settings (sample rate or analog
/// bandwidth)? If so, changing between them restarts the radio.
fn radio_setup_differs(a: f64, b: f64) -> bool {
    hardware_sample_rate_for(a) != hardware_sample_rate_for(b)
        || analog_bandwidth_hz(a) != analog_bandwidth_hz(b)
}

/// Is `hz` a symbol rate the modem accepts?
pub fn symbol_rate_in_range(hz: f64) -> bool {
    (MIN_SYMBOL_RATE_HZ..=MAX_SYMBOL_RATE_HZ).contains(&hz)
}

/// The TX offset-tuning shift exists to push LO leakage (at the LO frequency)
/// outside the signal's occupied band; flag a shift too small for the rate.
fn warn_if_shift_too_small(who: &str, symbol_rate_hz: f64, freq_shift_hz: f64) {
    let half_band = occupied_bandwidth_hz(symbol_rate_hz) / 2.0;
    if freq_shift_hz.abs() < half_band {
        eprintln!("[{who}] warning: freq shift {freq_shift_hz} Hz is inside the signal's +-{half_band:.0} Hz occupied band at {symbol_rate_hz} sym/s - LO leakage will land on the signal; raise the shift (on both TX and RX)");
    }
}

// ------------------------------------------------------------------------
// Interfaces
// ------------------------------------------------------------------------

/// Where status/telemetry goes. `topic_suffix` is e.g. `"tx/status"`; the
/// implementation prepends the base topic. Must never block the real-time loop.
pub trait Telemetry: Send + Sync {
    fn publish(&self, topic_suffix: &str, payload: &str, retain: bool);
}

pub trait TxRadio {
    /// Hand a chunk of hardware-rate IQ to the radio. Blocking here (until the
    /// radio can take more) is what paces the whole TX loop.
    fn write(&mut self, iq: &[ComplexI16]) -> Result<(), String>;
    fn set_gain(&mut self, db: i32) -> Result<(), String>;
    fn set_frequency(&mut self, hz: u64) -> Result<(), String>;
    /// Stop the stream, change the IQ sample rate and analog bandwidth, and
    /// start streaming again. Called only between engine runs (see
    /// `run_engines`), never while `write` is in progress.
    fn set_sample_rate(&mut self, hz: u32, analog_bandwidth_hz: u32) -> Result<(), String>;
}

pub trait RxRadio {
    /// Fill `buf` completely with hardware-rate IQ (blocking).
    fn read(&mut self, buf: &mut [ComplexI16]) -> Result<(), String>;
    fn set_gain(&mut self, db: i32) -> Result<(), String>;
    fn set_frequency(&mut self, hz: u64) -> Result<(), String>;
    /// As `TxRadio::set_sample_rate`.
    fn set_sample_rate(&mut self, hz: u32, analog_bandwidth_hz: u32) -> Result<(), String>;
}

/// Why an engine's `run` returned.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RunOutcome {
    /// The stop flag was set.
    Stopped,
    /// A symbol-rate change moved this engine to the other sample-rate
    /// profile: stop everything, let `run_engines` reconfigure the radio, and
    /// run again. The engine keeps all its settings.
    SampleRateChange,
}

// ------------------------------------------------------------------------
// MQTT message shapes
// ------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TxSource {
    Bert,
    Data,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum TxControlMsg {
    SetSource(TxSource),
    Bert(TxBertControl),
    /// Live TX gain change (dB).
    SetGainDb(i32),
    /// Retune the LO (Hz).
    SetFrequencyHz(u64),
    /// Change the DSP offset-tuning shift (Hz); the RX must be given the same.
    SetFreqShiftHz(f64),
    /// Change the modulation. The transmit chain is rebuilt (BERT/idle
    /// state restarts); the far end's receiver must be switched to match and
    /// will drop lock and re-acquire.
    SetModulation(Modulation),
    /// Change the symbol rate (Hz, `MIN_SYMBOL_RATE_HZ..=MAX_SYMBOL_RATE_HZ`).
    /// Same consequences as `SetModulation`. The occupied bandwidth grows
    /// with the rate: raise the freq shift so LO leakage stays outside it.
    SetSymbolRateHz(f64),
}

#[derive(Debug, Clone, Serialize)]
pub struct TxStatusMsg {
    pub sample_rate_hz: u32,
    pub modulation: Modulation,
    pub source: TxSource,
    pub bert: TxBertStatus,
    pub gain_db: i32,
    pub frequency_hz: u64,
    pub freq_shift_hz: f64,
    pub pattern: String,
    pub bit_rate_bps: f64,
    pub info_bit_rate_bps: f64,
    pub symbol_rate_hz: f64,
    pub kiss_client_connected: bool,
    pub frames_sent: u64,
    pub uptime_s: f64,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum RxControlMsg {
    Bert(RxBertControl),
    /// Live RX gain change (dB).
    SetGainDb(i32),
    /// Retune the carrier-recovery loop bandwidth (Hz).
    SetCarrierBandwidthHz(f64),
    /// Retune the Gardner timing-recovery loop bandwidth (cycles/symbol).
    SetTimingBandwidth(f64),
    /// Enable/disable the spectral frequency acquisition (default on).
    SetSearchEnabled(bool),
    /// Retune the LO (Hz).
    SetFrequencyHz(u64),
    /// Change the nominal TX offset shift the RX removes (Hz); must match the
    /// TX. Drops lock and re-acquires.
    SetFreqShiftHz(f64),
    /// Reset the timing and carrier loops and re-acquire from scratch.
    Reacquire,
    /// Change the demodulator's modulation. The receive chain is rebuilt
    /// (lock is dropped and re-acquired; BERT/frame decoder state restarts).
    SetModulation(Modulation),
    /// Change the symbol rate (Hz, `MIN_SYMBOL_RATE_HZ..=MAX_SYMBOL_RATE_HZ`).
    /// The carrier loop bandwidth is rescaled in proportion.
    SetSymbolRateHz(f64),
}

#[derive(Debug, Clone, Serialize)]
pub struct RxStatusMsg {
    pub sample_rate_hz: u32,
    pub modulation: Modulation,
    pub level_dbfs: f32,
    pub bert: RxBertStatus,
    pub bert_rotation: Option<u8>,
    /// Raw channel BER before the Viterbi decoder, measured by re-encoding its
    /// output (the BERT-synced branch); null unless synced.
    pub bert_pre_fec_ber: Option<f64>,
    pub frames_received: u64,
    pub bad_frames: u64,
    pub frame_rotation: Option<u8>,
    /// Same, for the HDLC-locked branch; null unless locked.
    pub frame_pre_fec_ber: Option<f64>,
    pub carrier_frequency_hz: f64,
    pub carrier_phase_radians: f64,
    pub carrier_phase_error: f64,
    /// True once the carrier loop has locked to data AND been verified against
    /// the spectrum (see `coarse_freq`).
    pub carrier_locked: bool,
    /// LO retune applied by frequency acquisition, relative to freq_shift_hz.
    pub lo_search_offset_hz: f64,
    /// lo_search_offset_hz + carrier_frequency_hz: the total TX/RX frequency error.
    pub total_offset_hz: f64,
    /// Estimated symbol SNR (Es/N0, dB) from the symbol-error standard
    /// deviation; null unless locked. Accurate above ~12 dB.
    pub snr_db: Option<f64>,
    /// RMS symbol error vector relative to symbol amplitude; null unless locked.
    pub symbol_error_std: Option<f64>,
    pub timing_sps: f64,
    pub gain_db: i32,
    pub search_enabled: bool,
    pub carrier_bandwidth_hz: f64,
    pub timing_bandwidth: f64,
    pub frequency_hz: u64,
    pub freq_shift_hz: f64,
    pub pattern: String,
    pub bit_rate_bps: f64,
    pub info_bit_rate_bps: f64,
    pub symbol_rate_hz: f64,
    pub kiss_client_connected: bool,
    pub uptime_s: f64,
}

/// Published once a second on `rx/symbols`: the latest 256 derotated symbols as
/// parallel I/Q arrays (rounded to 3 decimals; ~4 KB of JSON).
#[derive(Debug, Clone, Serialize)]
pub struct RxSymbolsMsg {
    pub i: Vec<f32>,
    pub q: Vec<f32>,
}

/// Commands for the whole process (topic `modem/control`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ModemControlMsg {
    /// Stop both engines and exit cleanly.
    Shutdown,
}

/// Process status (topic `modem/status`, retained). The broker also publishes
/// `{"state":"offline"}` here, as the MQTT last-will, if the process dies
/// without saying goodbye.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModemStatusMsg {
    /// "online" or "offline".
    pub state: String,
    pub pid: u32,
    pub uptime_s: f64,
    pub tx_enabled: bool,
    pub rx_enabled: bool,
    /// "none" or "rfic-bist".
    pub loopback: String,
    pub kiss_port: u16,
}

// ------------------------------------------------------------------------
// Transmit engine
// ------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct TxConfig {
    pub modulation: Modulation,
    pub symbol_rate_hz: f64,
    pub frequency_hz: u64,
    pub gain_db: i32,
    pub freq_shift_hz: f64,
    pub pattern: PrbsPattern,
    pub source: TxSource,
}

pub struct TxEngine {
    bert_generator: TxSignalGenerator,
    data_generator: Ax25TxSignalGenerator,
    kiss: Arc<KissServer>,
    modulation: Modulation,
    symbol_rate_hz: f64,
    hardware_rate: u32,
    /// The DSP no longer matches `symbol_rate_hz`/`hardware_rate`; rebuilt by
    /// `set_hardware_sample_rate`.
    dirty: bool,
    /// A symbol-rate change needs the radio reconfigured: `run` returns.
    review: bool,
    /// Frames sent by generators that have since been replaced.
    frames_sent_before: u64,
    use_bert: bool,
    gain_db: i32,
    frequency_hz: u64,
    freq_shift_hz: f64,
    pattern: PrbsPattern,
    start: Instant,
}

impl TxEngine {
    fn make_generators(
        pattern: PrbsPattern,
        modulation: Modulation,
        symbol_rate_hz: f64,
        hardware_rate: u32,
        freq_shift_hz: f64,
        kiss: Arc<KissServer>,
    ) -> (TxSignalGenerator, Ax25TxSignalGenerator) {
        let sps = sps_for(symbol_rate_hz, hardware_rate);
        let chunk_frames = chunk_frames_for(hardware_rate);
        let bert_generator = TxSignalGenerator::new(
            pattern,
            modulation,
            symbol_rate_hz,
            hardware_rate,
            sps,
            RRC_ROLLOFF,
            RRC_SPAN_SYMBOLS,
            ShapeMode::Rrc,
            chunk_frames,
            MAX_RESAMPLE_RATIO_RELATIVE,
            1.0, // full calibrated amplitude; level is set by the radio's TX gain
            freq_shift_hz,
        );
        let data_generator = Ax25TxSignalGenerator::new(
            kiss,
            modulation,
            symbol_rate_hz,
            hardware_rate,
            sps,
            RRC_ROLLOFF,
            RRC_SPAN_SYMBOLS,
            chunk_frames,
            MAX_RESAMPLE_RATIO_RELATIVE,
            1.0,
            freq_shift_hz,
        );
        (bert_generator, data_generator)
    }

    pub fn new(cfg: TxConfig, kiss: Arc<KissServer>) -> Self {
        let hardware_rate = hardware_sample_rate_for(cfg.symbol_rate_hz);
        let (bert_generator, data_generator) = Self::make_generators(
            cfg.pattern,
            cfg.modulation,
            cfg.symbol_rate_hz,
            hardware_rate,
            cfg.freq_shift_hz,
            kiss.clone(),
        );
        TxEngine {
            bert_generator,
            data_generator,
            kiss,
            modulation: cfg.modulation,
            symbol_rate_hz: cfg.symbol_rate_hz,
            hardware_rate,
            dirty: false,
            review: false,
            frames_sent_before: 0,
            use_bert: cfg.source == TxSource::Bert,
            gain_db: cfg.gain_db,
            frequency_hz: cfg.frequency_hz,
            freq_shift_hz: cfg.freq_shift_hz,
            pattern: cfg.pattern,
            start: Instant::now(),
        }
    }

    fn apply(&mut self, cmd: TxControlMsg, radio: &mut dyn TxRadio) {
        match cmd {
            TxControlMsg::SetSource(TxSource::Bert) => {
                self.use_bert = true;
                println!("[tx] source -> BERT");
            }
            TxControlMsg::SetSource(TxSource::Data) => {
                self.use_bert = false;
                println!("[tx] source -> DATA");
            }
            TxControlMsg::Bert(c) => self.bert_generator.bert_control(c),
            TxControlMsg::SetGainDb(db) => match radio.set_gain(db) {
                Ok(()) => {
                    self.gain_db = db;
                    println!("[tx] gain -> {db} dB");
                }
                Err(e) => eprintln!("[tx] failed to set gain to {db} dB: {e}"),
            },
            TxControlMsg::SetFrequencyHz(hz) => match radio.set_frequency(hz) {
                Ok(()) => {
                    self.frequency_hz = hz;
                    println!("[tx] frequency -> {hz} Hz");
                }
                Err(e) => eprintln!("[tx] failed to set frequency to {hz} Hz: {e}"),
            },
            TxControlMsg::SetFreqShiftHz(hz) => {
                self.bert_generator.set_freq_shift_hz(hz);
                self.data_generator.set_freq_shift_hz(hz);
                self.freq_shift_hz = hz;
                println!("[tx] freq shift -> {hz} Hz - the RX must be set to the same");
            }
            TxControlMsg::SetModulation(m) => {
                if m == self.modulation {
                    println!("[tx] modulation already {}", m.name());
                } else {
                    self.modulation = m;
                    self.rebuild();
                    println!(
                        "[tx] modulation -> {} - the RX must be set to the same",
                        m.name()
                    );
                }
            }
            TxControlMsg::SetSymbolRateHz(hz) => {
                if !symbol_rate_in_range(hz) {
                    eprintln!("[tx] symbol rate {hz} Hz rejected: must be {MIN_SYMBOL_RATE_HZ}..={MAX_SYMBOL_RATE_HZ}");
                } else {
                    let profile_change = radio_setup_differs(hz, self.symbol_rate_hz);
                    self.symbol_rate_hz = hz;
                    if profile_change {
                        self.dirty = true;
                        self.review = true;
                    } else {
                        self.rebuild();
                    }
                    println!(
                        "[tx] symbol rate -> {hz} sym/s{} - the RX must be set to the same",
                        if profile_change {
                            " (radio sample rate/bandwidth change: restarting the radio)"
                        } else {
                            ""
                        }
                    );
                    warn_if_shift_too_small("tx", hz, self.freq_shift_hz);
                }
            }
        }
    }

    /// Replace both generators with ones for the current modulation/symbol
    /// rate (keeping shift, pattern and the KISS connection). The output is
    /// not phase-continuous across this - the far end is expected to drop lock.
    fn rebuild(&mut self) {
        self.frames_sent_before += self.data_generator.frames_sent();
        let (bert, data) = Self::make_generators(
            self.pattern,
            self.modulation,
            self.symbol_rate_hz,
            self.hardware_rate,
            self.freq_shift_hz,
            self.kiss.clone(),
        );
        self.bert_generator = bert;
        self.data_generator = data;
        self.dirty = false;
    }

    /// The sample rate this engine's symbol rate wants.
    pub fn required_hardware_rate(&self) -> u32 {
        hardware_sample_rate_for(self.symbol_rate_hz)
    }

    pub fn symbol_rate_hz(&self) -> f64 {
        self.symbol_rate_hz
    }

    pub fn hardware_rate(&self) -> u32 {
        self.hardware_rate
    }

    /// Adopt the radio's (new) sample rate, rebuilding the DSP if it changed
    /// or a symbol-rate change is still pending.
    pub fn set_hardware_sample_rate(&mut self, hz: u32) {
        if hz != self.hardware_rate || self.dirty {
            self.hardware_rate = hz;
            self.rebuild();
        }
    }

    fn status(&self) -> TxStatusMsg {
        TxStatusMsg {
            sample_rate_hz: self.hardware_rate,
            modulation: self.modulation,
            source: if self.use_bert {
                TxSource::Bert
            } else {
                TxSource::Data
            },
            bert: self.bert_generator.bert_status(),
            gain_db: self.gain_db,
            frequency_hz: self.frequency_hz,
            freq_shift_hz: self.freq_shift_hz,
            pattern: format!("{:?}", self.pattern),
            bit_rate_bps: bit_rate_bps(self.modulation, self.symbol_rate_hz),
            info_bit_rate_bps: info_bit_rate_bps(self.modulation, self.symbol_rate_hz),
            symbol_rate_hz: self.symbol_rate_hz,
            kiss_client_connected: self.data_generator.has_kiss_client(),
            frames_sent: self.frames_sent_before + self.data_generator.frames_sent(),
            uptime_s: self.start.elapsed().as_secs_f64(),
        }
    }

    /// Run until `shutdown` is set (or the radio fails). The radio write blocks
    /// until the hardware is ready for more samples - that block IS the pacing
    /// authority for this loop. Switching sources doesn't reset the one being
    /// switched away from; it simply stops being polled.
    pub fn run(
        &mut self,
        radio: &mut dyn TxRadio,
        control: &Receiver<TxControlMsg>,
        telemetry: &dyn Telemetry,
        stop: &AtomicBool,
        status_interval: Duration,
    ) -> Result<RunOutcome, String> {
        let mut last_status = Instant::now() - status_interval;
        while !stop.load(Ordering::Relaxed) {
            while let Ok(cmd) = control.try_recv() {
                self.apply(cmd, radio);
            }
            if std::mem::take(&mut self.review) {
                return Ok(RunOutcome::SampleRateChange);
            }
            let iq = if self.use_bert {
                self.bert_generator.next_chunk()
            } else {
                self.data_generator.next_chunk()
            };
            radio
                .write(&iq)
                .map_err(|e| format!("tx radio write failed: {e}"))?;

            if last_status.elapsed() >= status_interval {
                if let Ok(json) = serde_json::to_string(&self.status()) {
                    telemetry.publish("tx/status", &json, true);
                }
                last_status = Instant::now();
            }
        }
        Ok(RunOutcome::Stopped)
    }
}

// ------------------------------------------------------------------------
// Receive engine
// ------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct RxConfig {
    pub modulation: Modulation,
    pub symbol_rate_hz: f64,
    pub frequency_hz: u64,
    pub gain_db: i32,
    pub freq_shift_hz: f64,
    pub pattern: PrbsPattern,
    pub carrier_bandwidth_hz: f64,
    pub dc_cutoff_hz: f64,
}

pub struct RxEngine {
    modulation: Modulation,
    symbol_rate_hz: f64,
    hardware_rate: u32,
    dirty: bool,
    review: bool,
    dc_cutoff_hz: f64,
    rx: RxSignalProcessor,
    ax25: Ax25RxPipeline,
    kiss: Arc<KissServer>,
    gain_db: i32,
    frequency_hz: u64,
    freq_shift_hz: f64,
    carrier_bandwidth_hz: f64,
    timing_bandwidth: f64,
    pattern: PrbsPattern,
    frames_received: u64,
    last_bits_received: u64,
    start: Instant,
}

impl RxEngine {
    #[allow(clippy::too_many_arguments)]
    fn make_rx(
        pattern: PrbsPattern,
        modulation: Modulation,
        symbol_rate_hz: f64,
        hardware_rate: u32,
        dc_cutoff_hz: f64,
        freq_shift_hz: f64,
        carrier_bandwidth_hz: f64,
        timing_bandwidth: f64,
    ) -> RxSignalProcessor {
        RxSignalProcessor::new(
            pattern,
            modulation,
            symbol_rate_hz,
            f64::from(hardware_rate),
            sps_for(symbol_rate_hz, hardware_rate),
            RRC_ROLLOFF,
            RRC_SPAN_SYMBOLS,
            dc_cutoff_hz,
            freq_shift_hz,
            carrier_bandwidth_hz,
            CARRIER_DAMPING,
            timing_bandwidth,
            TIMING_DAMPING,
            chunk_frames_for(hardware_rate), // one resampler chunk per read
            MAX_RESAMPLE_RATIO_RELATIVE,
        )
    }

    pub fn new(cfg: RxConfig, kiss: Arc<KissServer>) -> Self {
        let hardware_rate = hardware_sample_rate_for(cfg.symbol_rate_hz);
        let rx = Self::make_rx(
            cfg.pattern,
            cfg.modulation,
            cfg.symbol_rate_hz,
            hardware_rate,
            cfg.dc_cutoff_hz,
            cfg.freq_shift_hz,
            cfg.carrier_bandwidth_hz,
            TIMING_LOOP_BANDWIDTH,
        );
        RxEngine {
            modulation: cfg.modulation,
            symbol_rate_hz: cfg.symbol_rate_hz,
            hardware_rate,
            dirty: false,
            review: false,
            dc_cutoff_hz: cfg.dc_cutoff_hz,
            rx,
            ax25: Ax25RxPipeline::new(cfg.modulation),
            kiss,
            gain_db: cfg.gain_db,
            frequency_hz: cfg.frequency_hz,
            freq_shift_hz: cfg.freq_shift_hz,
            carrier_bandwidth_hz: cfg.carrier_bandwidth_hz,
            timing_bandwidth: TIMING_LOOP_BANDWIDTH,
            pattern: cfg.pattern,
            frames_received: 0,
            last_bits_received: 0,
            start: Instant::now(),
        }
    }

    fn apply(&mut self, cmd: RxControlMsg, radio: &mut dyn RxRadio) {
        match cmd {
            RxControlMsg::Bert(c) => self.rx.bert_control(c),
            RxControlMsg::SetGainDb(db) => match radio.set_gain(db) {
                Ok(()) => {
                    self.gain_db = db;
                    println!("[rx] gain -> {db} dB");
                }
                Err(e) => eprintln!("[rx] failed to set gain to {db} dB: {e}"),
            },
            RxControlMsg::SetFrequencyHz(hz) => match radio.set_frequency(hz) {
                Ok(()) => {
                    self.frequency_hz = hz;
                    println!("[rx] frequency -> {hz} Hz");
                }
                Err(e) => eprintln!("[rx] failed to set frequency to {hz} Hz: {e}"),
            },
            RxControlMsg::SetFreqShiftHz(hz) => {
                self.rx.set_freq_shift_hz(hz);
                self.freq_shift_hz = hz;
                println!("[rx] freq shift -> {hz} Hz - dropping lock to re-acquire");
            }
            RxControlMsg::SetCarrierBandwidthHz(hz) => {
                self.rx.set_carrier_loop_bandwidth_hz(hz);
                self.carrier_bandwidth_hz = hz;
                println!("[rx] carrier loop bandwidth -> {hz} Hz");
            }
            RxControlMsg::Reacquire => {
                self.rx.reacquire();
                println!("[rx] reacquire - timing/carrier loops reset");
            }
            RxControlMsg::SetModulation(m) => {
                if m == self.modulation {
                    println!("[rx] modulation already {}", m.name());
                } else {
                    self.modulation = m;
                    self.rebuild();
                    println!(
                        "[rx] modulation -> {} - link dropped, re-acquiring",
                        m.name()
                    );
                }
            }
            RxControlMsg::SetSymbolRateHz(hz) => {
                if !symbol_rate_in_range(hz) {
                    eprintln!("[rx] symbol rate {hz} Hz rejected: must be {MIN_SYMBOL_RATE_HZ}..={MAX_SYMBOL_RATE_HZ}");
                } else {
                    // Keep the carrier loop's bandwidth relative to the symbol rate.
                    self.carrier_bandwidth_hz *= hz / self.symbol_rate_hz;
                    let profile_change = radio_setup_differs(hz, self.symbol_rate_hz);
                    self.symbol_rate_hz = hz;
                    if profile_change {
                        self.dirty = true;
                        self.review = true;
                    } else {
                        self.rebuild();
                    }
                    println!(
                        "[rx] symbol rate -> {hz} sym/s (carrier loop BW -> {:.1} Hz){} - link dropped, re-acquiring",
                        self.carrier_bandwidth_hz,
                        if profile_change { ", radio sample rate/bandwidth change: restarting the radio" } else { "" }
                    );
                    warn_if_shift_too_small("rx", hz, self.freq_shift_hz);
                }
            }
            RxControlMsg::SetSearchEnabled(on) => {
                self.rx.set_search_enabled(on);
                println!("[rx] frequency acquisition -> {on}");
            }
            RxControlMsg::SetTimingBandwidth(bw) => {
                self.rx.set_timing_loop_bandwidth(bw);
                self.timing_bandwidth = bw;
                println!("[rx] timing loop bandwidth -> {bw}");
            }
        }
    }

    /// The PRBS checkers and the HDLC deframer each run a bank of Viterbi
    /// decoders; a link carries a PRBS or frames, never both, so once one
    /// side has the link the other bank is stopped (and restarted, searching
    /// afresh, when it lets go). This is what keeps the CPU load at one bank.
    fn balance_decoder_banks(&mut self) {
        let bert_synced = self.rx.bert_rotation().is_some();
        let hdlc_locked = self.ax25.locked_rotation().is_some();
        self.rx.set_bert_enabled(!hdlc_locked || bert_synced);
        self.ax25.set_enabled(!bert_synced);
    }

    /// Replace the DSP chain and frame decoder with ones for the current
    /// modulation/symbol rate, keeping gain/LO/shift/loop settings. All
    /// tracking state and decoder history restart, so the link re-acquires.
    fn rebuild(&mut self) {
        let search = self.rx.search_enabled();
        self.rx = Self::make_rx(
            self.pattern,
            self.modulation,
            self.symbol_rate_hz,
            self.hardware_rate,
            self.dc_cutoff_hz,
            self.freq_shift_hz,
            self.carrier_bandwidth_hz,
            self.timing_bandwidth,
        );
        self.rx.set_search_enabled(search);
        self.ax25 = Ax25RxPipeline::new(self.modulation);
        self.last_bits_received = 0;
        self.dirty = false;
    }

    /// The sample rate this engine's symbol rate wants.
    pub fn required_hardware_rate(&self) -> u32 {
        hardware_sample_rate_for(self.symbol_rate_hz)
    }

    pub fn symbol_rate_hz(&self) -> f64 {
        self.symbol_rate_hz
    }

    pub fn hardware_rate(&self) -> u32 {
        self.hardware_rate
    }

    /// Adopt the radio's (new) sample rate, rebuilding the DSP if it changed
    /// or a symbol-rate change is still pending.
    pub fn set_hardware_sample_rate(&mut self, hz: u32) {
        if hz != self.hardware_rate || self.dirty {
            self.hardware_rate = hz;
            self.rebuild();
        }
    }

    fn status(&self, level_dbfs: f32) -> RxStatusMsg {
        RxStatusMsg {
            sample_rate_hz: self.hardware_rate,
            modulation: self.modulation,
            level_dbfs,
            bert: self.rx.bert_status(),
            bert_rotation: self.rx.bert_rotation(),
            bert_pre_fec_ber: self.rx.pre_fec_ber(),
            frames_received: self.frames_received,
            bad_frames: self.ax25.bad_frame_count(),
            frame_rotation: self.ax25.locked_rotation(),
            frame_pre_fec_ber: self.ax25.pre_fec_ber(),
            carrier_frequency_hz: self.rx.carrier_frequency_hz(),
            carrier_phase_radians: self.rx.carrier_phase_radians(),
            carrier_phase_error: self.rx.carrier_phase_error(),
            carrier_locked: self.rx.carrier_locked(),
            lo_search_offset_hz: self.rx.search_offset_hz(),
            total_offset_hz: self.rx.total_offset_hz(),
            snr_db: self.rx.snr_db(),
            symbol_error_std: self.rx.symbol_error_std(),
            timing_sps: self.rx.timing_omega(),
            gain_db: self.gain_db,
            search_enabled: self.rx.search_enabled(),
            carrier_bandwidth_hz: self.carrier_bandwidth_hz,
            timing_bandwidth: self.timing_bandwidth,
            frequency_hz: self.frequency_hz,
            freq_shift_hz: self.freq_shift_hz,
            pattern: format!("{:?}", self.pattern),
            bit_rate_bps: bit_rate_bps(self.modulation, self.symbol_rate_hz),
            info_bit_rate_bps: info_bit_rate_bps(self.modulation, self.symbol_rate_hz),
            symbol_rate_hz: self.symbol_rate_hz,
            kiss_client_connected: self.kiss.has_client(),
            uptime_s: self.start.elapsed().as_secs_f64(),
        }
    }

    fn print_status(&self, s: &RxStatusMsg) {
        let fmt_rot = |r: Option<u8>| {
            r.map_or("-".to_string(), |r| {
                format!(
                    "{:.0}deg",
                    f64::from(r) * 360.0 / f64::from(self.modulation.num_points())
                )
            })
        };
        let ber = if s.bert.bits_received > self.last_bits_received {
            s.bert.bit_errors_received as f64 / s.bert.bits_received as f64
        } else {
            0.0
        };
        println!(
            "[{:6.1}s] level {:6.1} dBFS | BERT {:?}@{} sync_loss={} bits {} errs {} BER {ber:.2e} (pre-FEC {}) | frames {} (HDLC rot {}) | carrier {:7.1} Hz err {:.3} rad {} | offset {:+7.1} Hz | SNR {} | timing sps {:.4}",
            s.uptime_s,
            s.level_dbfs,
            s.bert.locked_state,
            fmt_rot(s.bert_rotation),
            s.bert.sync_loss,
            s.bert.bits_received,
            s.bert.bit_errors_received,
            s.bert_pre_fec_ber.map_or("-".to_string(), |b| format!("{b:.1e}")),
            s.frames_received,
            fmt_rot(s.frame_rotation),
            s.carrier_frequency_hz,
            s.carrier_phase_error,
            if s.carrier_locked { "LOCKED" } else { "searching" },
            s.total_offset_hz,
            s.snr_db.map_or("-".to_string(), |v| format!("{v:.1} dB (err std {:.3})", s.symbol_error_std.unwrap_or(0.0))),
            s.timing_sps,
        );
    }

    /// Run until `shutdown` is set (or the radio fails). The BERT bank (inside
    /// `RxSignalProcessor`) and the AX.25 pipeline always both see the same
    /// symbol stream - neither is gated by a mode switch, so the receiver
    /// follows whatever the transmitter is sending.
    pub fn run(
        &mut self,
        radio: &mut dyn RxRadio,
        control: &Receiver<RxControlMsg>,
        telemetry: &dyn Telemetry,
        stop: &AtomicBool,
        status_interval: Duration,
    ) -> Result<RunOutcome, String> {
        let mut buf = vec![ComplexI16::new(0, 0); chunk_frames_for(self.hardware_rate)];
        let mut last_status = Instant::now();
        while !stop.load(Ordering::Relaxed) {
            while let Ok(cmd) = control.try_recv() {
                self.apply(cmd, radio);
            }
            if std::mem::take(&mut self.review) {
                return Ok(RunOutcome::SampleRateChange);
            }
            radio
                .read(&mut buf)
                .map_err(|e| format!("rx radio read failed: {e}"))?;
            let symbols = self.rx.process(&buf);

            for frame in self.ax25.process(&symbols) {
                self.frames_received += 1;
                self.kiss.send_frame(&frame);
                println!(
                    "[{:6.1}s] RX FRAME #{} ({} bytes): {:02X?}",
                    self.start.elapsed().as_secs_f64(),
                    self.frames_received,
                    frame.len(),
                    frame
                );
            }
            self.balance_decoder_banks();

            if last_status.elapsed() >= status_interval {
                let iq = complex_i16_to_f32(&buf);
                let rms = (iq.iter().map(|c| c.norm_sqr()).sum::<f32>() / iq.len() as f32).sqrt();
                let dbfs = 20.0 * rms.max(1e-9).log10();
                let status = self.rx.bert_status();
                let msg = self.status(dbfs);
                self.print_status(&msg);
                self.last_bits_received = status.bits_received;
                if let Ok(json) = serde_json::to_string(&msg) {
                    telemetry.publish("rx/status", &json, true);
                }

                // Constellation snapshot: not retained (a stale picture is
                // worse than none). Skipped until 256 symbols exist.
                let recent = self.rx.recent_symbols();
                if recent.len() == RECENT_SYMBOLS {
                    let round = |x: f32| (x * 1000.0).round() / 1000.0;
                    let sym = RxSymbolsMsg {
                        i: recent.iter().map(|c| round(c.re)).collect(),
                        q: recent.iter().map(|c| round(c.im)).collect(),
                    };
                    if let Ok(json) = serde_json::to_string(&sym) {
                        telemetry.publish("rx/symbols", &json, false);
                    }
                }
                last_status = Instant::now();
            }
        }
        Ok(RunOutcome::Stopped)
    }
}

// ------------------------------------------------------------------------
// Running both engines, across sample-rate profile changes
// ------------------------------------------------------------------------

/// Run the TX and/or RX engine, each on its own thread, until `shutdown` is
/// set or an engine fails. When a symbol-rate change moves an engine to the
/// other sample-rate profile, both engines are stopped, the radios are told
/// the new sample rate and analog bandwidth, the engines adopt it (rebuilding
/// their DSP), and both are started again: the link drops and re-acquires.
/// The sample rate is shared (one converter clock), so it is the higher of
/// what the running engines want.
#[allow(clippy::type_complexity)]
pub fn run_engines<T: TxRadio + Send, R: RxRadio + Send>(
    mut tx: Option<(&mut TxEngine, &mut T, &Receiver<TxControlMsg>)>,
    mut rx: Option<(&mut RxEngine, &mut R, &Receiver<RxControlMsg>)>,
    telemetry: &dyn Telemetry,
    shutdown: &AtomicBool,
    status_interval: Duration,
) -> Result<(), String> {
    // The analog bandwidth the radios were last set to (by the caller at start-up).
    let initial_rate = tx
        .iter()
        .map(|(e, _, _)| e.symbol_rate_hz())
        .chain(rx.iter().map(|(e, _, _)| e.symbol_rate_hz()))
        .fold(0.0, f64::max);
    let mut applied_bandwidth = analog_bandwidth_hz(initial_rate);
    loop {
        let stop = AtomicBool::new(false);
        let (tx_result, rx_result) = std::thread::scope(|scope| {
            let stop = &stop;
            let tx_handle = tx.as_mut().map(|(engine, radio, control)| {
                scope.spawn(move || {
                    let r = engine.run(*radio, control, telemetry, stop, status_interval);
                    if !matches!(r, Ok(RunOutcome::Stopped)) {
                        stop.store(true, Ordering::Relaxed);
                    }
                    r
                })
            });
            let rx_handle = rx.as_mut().map(|(engine, radio, control)| {
                scope.spawn(move || {
                    let r = engine.run(*radio, control, telemetry, stop, status_interval);
                    if !matches!(r, Ok(RunOutcome::Stopped)) {
                        stop.store(true, Ordering::Relaxed);
                    }
                    r
                })
            });
            while !(tx_handle.as_ref().is_none_or(|h| h.is_finished())
                && rx_handle.as_ref().is_none_or(|h| h.is_finished()))
            {
                if shutdown.load(Ordering::Relaxed) {
                    stop.store(true, Ordering::Relaxed);
                }
                std::thread::sleep(Duration::from_millis(10));
            }
            (
                tx_handle.map(|h| {
                    h.join()
                        .unwrap_or_else(|_| Err("tx engine panicked".into()))
                }),
                rx_handle.map(|h| {
                    h.join()
                        .unwrap_or_else(|_| Err("rx engine panicked".into()))
                }),
            )
        });

        let mut errors = Vec::new();
        if let Some(Err(e)) = &tx_result {
            errors.push(format!("tx: {e}"));
        }
        if let Some(Err(e)) = &rx_result {
            errors.push(format!("rx: {e}"));
        }
        if !errors.is_empty() {
            return Err(errors.join("; "));
        }
        if shutdown.load(Ordering::Relaxed) {
            return Ok(());
        }

        // A sample-rate review: pick the rate, reconfigure the radios, restart.
        let want = tx
            .iter()
            .map(|(e, _, _)| e.required_hardware_rate())
            .chain(rx.iter().map(|(e, _, _)| e.required_hardware_rate()))
            .max()
            .expect("at least one engine");
        let max_symbol_rate = tx
            .iter()
            .map(|(e, _, _)| e.symbol_rate_hz())
            .chain(rx.iter().map(|(e, _, _)| e.symbol_rate_hz()))
            .fold(0.0, f64::max);
        let bandwidth = analog_bandwidth_hz(max_symbol_rate);
        let current = tx
            .iter()
            .map(|(e, _, _)| e.hardware_rate())
            .chain(rx.iter().map(|(e, _, _)| e.hardware_rate()))
            .next();
        if current != Some(want) || bandwidth != applied_bandwidth {
            applied_bandwidth = bandwidth;
            println!("[modem] sample rate -> {want} Hz, analog bandwidth {bandwidth} Hz");
            if let Some((_, radio, _)) = tx.as_mut() {
                radio
                    .set_sample_rate(want, bandwidth)
                    .map_err(|e| format!("tx set sample rate: {e}"))?;
            }
            if let Some((_, radio, _)) = rx.as_mut() {
                radio
                    .set_sample_rate(want, bandwidth)
                    .map_err(|e| format!("rx set sample rate: {e}"))?;
            }
        }
        if let Some((engine, _, _)) = tx.as_mut() {
            engine.set_hardware_sample_rate(want);
        }
        if let Some((engine, _, _)) = rx.as_mut() {
            engine.set_hardware_sample_rate(want);
        }
    }
}

// ------------------------------------------------------------------------
// Tests: the whole modem in software (mock radios, no hardware, no broker)
// ------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kiss::KissDecoder;
    use crossbeam_channel::{bounded, unbounded, Sender};
    use std::collections::VecDeque;
    use std::io::{Read, Write};
    use std::net::TcpStream;
    use std::sync::Mutex;

    #[derive(Default)]
    struct Collect(Mutex<Vec<(String, String)>>);
    impl Telemetry for Collect {
        fn publish(&self, topic_suffix: &str, payload: &str, _retain: bool) {
            self.0
                .lock()
                .unwrap()
                .push((topic_suffix.to_string(), payload.to_string()));
        }
    }
    impl Collect {
        fn last(&self, topic: &str) -> Option<serde_json::Value> {
            let g = self.0.lock().unwrap();
            g.iter()
                .rev()
                .find(|(t, _)| t == topic)
                .and_then(|(_, p)| serde_json::from_str(p).ok())
        }
        fn count(&self, topic: &str) -> usize {
            self.0
                .lock()
                .unwrap()
                .iter()
                .filter(|(t, _)| t == topic)
                .count()
        }
    }

    type RateLog = Arc<Mutex<Vec<(u32, u32)>>>;

    struct MockTx {
        link: Sender<Vec<ComplexI16>>,
        shutdown: Arc<AtomicBool>,
        gain: Arc<Mutex<i32>>,
        freq: Arc<Mutex<u64>>,
        rates: RateLog,
    }
    impl TxRadio for MockTx {
        fn write(&mut self, iq: &[ComplexI16]) -> Result<(), String> {
            // Bounded link = the pacing; never hang forever once the RX is gone.
            let mut chunk = Some(iq.to_vec());
            let mut waited = 0;
            while let Some(c) = chunk.take() {
                match self.link.send_timeout(c, Duration::from_millis(100)) {
                    Ok(()) => {}
                    Err(crossbeam_channel::SendTimeoutError::Timeout(c)) => {
                        // A real radio's write times out too; the engine may be
                        // stopping for a sample-rate change with the RX already stopped.
                        waited += 1;
                        if self.shutdown.load(Ordering::Relaxed) || waited >= 5 {
                            return Ok(());
                        }
                        chunk = Some(c);
                    }
                    // The RX side finished first during shutdown: not an error then.
                    Err(_) if self.shutdown.load(Ordering::Relaxed) => return Ok(()),
                    Err(_) => return Err("link closed".into()),
                }
            }
            Ok(())
        }
        fn set_gain(&mut self, db: i32) -> Result<(), String> {
            *self.gain.lock().unwrap() = db;
            Ok(())
        }
        fn set_frequency(&mut self, hz: u64) -> Result<(), String> {
            *self.freq.lock().unwrap() = hz;
            Ok(())
        }
        fn set_sample_rate(&mut self, hz: u32, analog_bandwidth_hz: u32) -> Result<(), String> {
            self.rates.lock().unwrap().push((hz, analog_bandwidth_hz));
            Ok(())
        }
    }

    struct MockRx {
        link: Receiver<Vec<ComplexI16>>,
        pending: VecDeque<ComplexI16>,
        shutdown: Arc<AtomicBool>,
        gain: Arc<Mutex<i32>>,
        freq: Arc<Mutex<u64>>,
        rates: RateLog,
    }
    impl RxRadio for MockRx {
        fn read(&mut self, buf: &mut [ComplexI16]) -> Result<(), String> {
            let mut waited = 0;
            while self.pending.len() < buf.len() {
                match self.link.recv_timeout(Duration::from_millis(100)) {
                    Ok(chunk) => self.pending.extend(chunk),
                    Err(_) => {
                        waited += 1;
                        // Shutting down, or the TX stopped for a sample-rate
                        // change: return silence rather than block forever.
                        if self.shutdown.load(Ordering::Relaxed) || waited >= 5 {
                            buf.fill(ComplexI16::new(0, 0));
                            return Ok(());
                        }
                    }
                }
            }
            for slot in buf.iter_mut() {
                *slot = self.pending.pop_front().unwrap();
            }
            Ok(())
        }
        fn set_gain(&mut self, db: i32) -> Result<(), String> {
            *self.gain.lock().unwrap() = db;
            Ok(())
        }
        fn set_frequency(&mut self, hz: u64) -> Result<(), String> {
            *self.freq.lock().unwrap() = hz;
            Ok(())
        }
        fn set_sample_rate(&mut self, hz: u32, analog_bandwidth_hz: u32) -> Result<(), String> {
            self.rates.lock().unwrap().push((hz, analog_bandwidth_hz));
            self.pending.clear();
            while self.link.try_recv().is_ok() {} // samples from the old rate are meaningless
            Ok(())
        }
    }

    /// Sets the shutdown flag when dropped - including while unwinding from a
    /// failed assertion - so a test failure can't leave engine threads running
    /// (a `thread::scope` would otherwise wait for them forever).
    struct ShutdownGuard(Arc<AtomicBool>);
    impl Drop for ShutdownGuard {
        fn drop(&mut self) {
            self.0.store(true, Ordering::Relaxed);
        }
    }

    fn wait_for(what: &str, timeout_s: f64, mut cond: impl FnMut() -> bool) {
        let start = Instant::now();
        while start.elapsed().as_secs_f64() < timeout_s {
            if cond() {
                return;
            }
            std::thread::sleep(Duration::from_millis(50));
        }
        panic!("timed out after {timeout_s} s waiting for: {what}");
    }

    /// The full modem, both engines, joined by an in-memory link: PRBS lock with
    /// a residual frequency offset, live commands (gain/frequency/source), a KISS
    /// frame in one end and out the other over the SAME socket, clean shutdown.
    #[test]
    fn engines_run_a_full_link_with_commands_and_a_shared_kiss_socket() {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        drop(listener);
        let kiss = Arc::new(KissServer::start(&addr.to_string()).unwrap());
        std::thread::sleep(Duration::from_millis(50));

        let shutdown = Arc::new(AtomicBool::new(false));
        let telemetry = Arc::new(Collect::default());
        let (link_tx, link_rx) = bounded::<Vec<ComplexI16>>(4);
        let (tx_ctl, tx_ctl_rx) = unbounded::<TxControlMsg>();
        let (rx_ctl, rx_ctl_rx) = unbounded::<RxControlMsg>();
        let (tx_gain, tx_freq) = (Arc::new(Mutex::new(0)), Arc::new(Mutex::new(0)));
        let (rx_gain, rx_freq) = (Arc::new(Mutex::new(0)), Arc::new(Mutex::new(0)));

        // 15 kHz nominal shift on both ends, plus a 350 Hz oscillator error.
        let mut tx_engine = TxEngine::new(
            TxConfig {
                modulation: Modulation::Psk8,
                symbol_rate_hz: SYMBOL_RATE_HZ,
                frequency_hz: 439_500_000,
                gain_db: -10,
                freq_shift_hz: 15_350.0,
                pattern: PrbsPattern::Pn15,
                source: TxSource::Bert,
            },
            kiss.clone(),
        );
        let mut rx_engine = RxEngine::new(
            RxConfig {
                modulation: Modulation::Psk8,
                symbol_rate_hz: SYMBOL_RATE_HZ,
                frequency_hz: 439_500_000,
                gain_db: 30,
                freq_shift_hz: 15_000.0,
                pattern: PrbsPattern::Pn15,
                carrier_bandwidth_hz: DEFAULT_CARRIER_BANDWIDTH_HZ,
                dc_cutoff_hz: DEFAULT_DC_CUTOFF_HZ,
            },
            kiss.clone(),
        );

        let mut client = TcpStream::connect(addr).unwrap();
        client
            .set_read_timeout(Some(Duration::from_millis(100)))
            .unwrap();
        std::thread::sleep(Duration::from_millis(50));

        std::thread::scope(|scope| {
            let _guard = ShutdownGuard(shutdown.clone());
            let (sd, tel) = (shutdown.clone(), telemetry.clone());
            let mut mock_tx = MockTx {
                link: link_tx,
                shutdown: shutdown.clone(),
                gain: tx_gain.clone(),
                freq: tx_freq.clone(),
                rates: RateLog::default(),
            };
            let tx_handle = scope.spawn(move || {
                tx_engine.run(
                    &mut mock_tx,
                    &tx_ctl_rx,
                    &*tel,
                    &sd,
                    Duration::from_millis(100),
                )
            });
            let (sd, tel) = (shutdown.clone(), telemetry.clone());
            let mut mock_rx = MockRx {
                link: link_rx,
                pending: VecDeque::new(),
                shutdown: shutdown.clone(),
                gain: rx_gain.clone(),
                freq: rx_freq.clone(),
                rates: RateLog::default(),
            };
            let rx_handle = scope.spawn(move || {
                rx_engine.run(
                    &mut mock_rx,
                    &rx_ctl_rx,
                    &*tel,
                    &sd,
                    Duration::from_millis(100),
                )
            });

            // 1) PRBS: carrier locks (verified), BERT syncs, nothing wrong with the offset.
            wait_for("carrier lock + BERT sync", 40.0, || {
                telemetry.last("rx/status").is_some_and(|s| {
                    s["carrier_locked"] == true && s["bert"]["locked_state"] == "Synced"
                })
            });
            let s = telemetry.last("rx/status").unwrap();
            let off = s["total_offset_hz"].as_f64().unwrap();
            assert!(
                (off - 350.0).abs() < 40.0,
                "total offset {off} Hz, expected ~350"
            );
            assert!(
                s["snr_db"].as_f64().unwrap() > 30.0,
                "clean link should read a high SNR: {}",
                s["snr_db"]
            );
            let t = telemetry.last("tx/status").unwrap();
            assert_eq!(t["source"], "Bert");
            assert_eq!(t["gain_db"], -10);
            assert!(
                telemetry.count("rx/symbols") >= 1,
                "constellation snapshots should be flowing"
            );

            // 2) Commands reach the radios and are echoed in the status.
            tx_ctl.send(TxControlMsg::SetGainDb(-5)).unwrap();
            tx_ctl
                .send(TxControlMsg::SetFrequencyHz(439_600_000))
                .unwrap();
            rx_ctl.send(RxControlMsg::SetGainDb(42)).unwrap();
            rx_ctl
                .send(RxControlMsg::SetFrequencyHz(439_700_000))
                .unwrap();
            wait_for("commands applied", 10.0, || {
                *tx_gain.lock().unwrap() == -5
                    && *tx_freq.lock().unwrap() == 439_600_000
                    && *rx_gain.lock().unwrap() == 42
                    && *rx_freq.lock().unwrap() == 439_700_000
            });
            wait_for("status echoes the commands", 10.0, || {
                let t = telemetry.last("tx/status").unwrap();
                let r = telemetry.last("rx/status").unwrap();
                t["gain_db"] == -5
                    && t["frequency_hz"] == 439_600_000u64
                    && r["gain_db"] == 42
                    && r["frequency_hz"] == 439_700_000u64
            });

            // 3) DATA mode: a KISS frame in on the one socket comes back out on it,
            //    having crossed TX engine -> link -> RX engine.
            tx_ctl
                .send(TxControlMsg::SetSource(TxSource::Data))
                .unwrap();
            wait_for("source switched", 10.0, || {
                telemetry.last("tx/status").unwrap()["source"] == "Data"
            });
            // Let the receiver's HDLC deframer see idle flags first.
            std::thread::sleep(Duration::from_millis(1500));
            let payload = b"frame across the combined modem".to_vec();
            client.write_all(&crate::kiss::encode(&payload)).unwrap();

            let mut decoder = KissDecoder::new();
            let mut got = Vec::new();
            let start = Instant::now();
            let mut buf = [0u8; 2048];
            while got.is_empty() && start.elapsed().as_secs() < 30 {
                if let Ok(n) = client.read(&mut buf) {
                    got.extend(decoder.feed(&buf[..n]));
                }
            }
            assert_eq!(
                got,
                vec![payload],
                "frame did not come back through TX -> RX over the shared KISS socket"
            );
            wait_for("frames_received to show up in the status", 10.0, || {
                telemetry.last("rx/status").is_some_and(|r| {
                    r["frames_received"].as_u64().unwrap_or(0) >= 1
                        && r["kiss_client_connected"] == true
                })
            });

            // 4) Clean shutdown.
            shutdown.store(true, Ordering::Relaxed);
            tx_handle
                .join()
                .unwrap()
                .expect("tx engine ended with an error");
            rx_handle
                .join()
                .unwrap()
                .expect("rx engine ended with an error");
        });
    }

    /// Live reconfiguration: Reacquire, then modulation and symbol-rate changes
    /// on both engines. The link must drop and re-acquire each time, a TX/RX
    /// mismatch must NOT link, and a KISS frame must still cross at the end.
    #[test]
    fn engines_switch_modulation_and_symbol_rate_and_reacquire() {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        drop(listener);
        let kiss = Arc::new(KissServer::start(&addr.to_string()).unwrap());
        std::thread::sleep(Duration::from_millis(50));

        let shutdown = Arc::new(AtomicBool::new(false));
        let telemetry = Arc::new(Collect::default());
        let (link_tx, link_rx) = bounded::<Vec<ComplexI16>>(4);
        let (tx_ctl, tx_ctl_rx) = unbounded::<TxControlMsg>();
        let (rx_ctl, rx_ctl_rx) = unbounded::<RxControlMsg>();
        let zero = || (Arc::new(Mutex::new(0)), Arc::new(Mutex::new(0u64)));
        let ((tx_gain, tx_freq), (rx_gain, rx_freq)) = (zero(), zero());

        // 25 kHz shift: clear of the occupied band up to 18 ksym/s.
        let mut tx_engine = TxEngine::new(
            TxConfig {
                modulation: Modulation::Psk8,
                symbol_rate_hz: SYMBOL_RATE_HZ,
                frequency_hz: 1,
                gain_db: -10,
                freq_shift_hz: 25_000.0,
                pattern: PrbsPattern::Pn15,
                source: TxSource::Bert,
            },
            kiss.clone(),
        );
        let mut rx_engine = RxEngine::new(
            RxConfig {
                modulation: Modulation::Psk8,
                symbol_rate_hz: SYMBOL_RATE_HZ,
                frequency_hz: 1,
                gain_db: 30,
                freq_shift_hz: 25_000.0,
                pattern: PrbsPattern::Pn15,
                carrier_bandwidth_hz: DEFAULT_CARRIER_BANDWIDTH_HZ,
                dc_cutoff_hz: DEFAULT_DC_CUTOFF_HZ,
            },
            kiss.clone(),
        );
        let mut client = TcpStream::connect(addr).unwrap();
        client
            .set_read_timeout(Some(Duration::from_millis(100)))
            .unwrap();
        std::thread::sleep(Duration::from_millis(50));

        std::thread::scope(|scope| {
            let _guard = ShutdownGuard(shutdown.clone());
            let (sd, tel) = (shutdown.clone(), telemetry.clone());
            let mut mock_tx = MockTx {
                link: link_tx,
                shutdown: shutdown.clone(),
                gain: tx_gain.clone(),
                freq: tx_freq.clone(),
                rates: RateLog::default(),
            };
            let tx_handle = scope.spawn(move || {
                tx_engine.run(
                    &mut mock_tx,
                    &tx_ctl_rx,
                    &*tel,
                    &sd,
                    Duration::from_millis(100),
                )
            });
            let (sd, tel) = (shutdown.clone(), telemetry.clone());
            let mut mock_rx = MockRx {
                link: link_rx,
                pending: VecDeque::new(),
                shutdown: shutdown.clone(),
                gain: rx_gain.clone(),
                freq: rx_freq.clone(),
                rates: RateLog::default(),
            };
            let rx_handle = scope.spawn(move || {
                rx_engine.run(
                    &mut mock_rx,
                    &rx_ctl_rx,
                    &*tel,
                    &sd,
                    Duration::from_millis(100),
                )
            });

            let rx_is = |modulation: &str, rate: f64| {
                telemetry.last("rx/status").is_some_and(|s| {
                    s["modulation"] == modulation && s["symbol_rate_hz"].as_f64() == Some(rate)
                })
            };
            let linked = |modulation: &str, rate: f64| {
                telemetry.last("rx/status").is_some_and(|s| {
                    s["modulation"] == modulation
                        && s["symbol_rate_hz"].as_f64() == Some(rate)
                        && s["carrier_locked"] == true
                        && s["bert"]["locked_state"] == "Synced"
                })
            };

            wait_for("initial 8PSK link", 60.0, || linked("Psk8", 16_000.0));

            // Reacquire: the loops reset, so the lock drops and then comes back.
            rx_ctl.send(RxControlMsg::Reacquire).unwrap();
            wait_for("lock re-established after Reacquire", 60.0, || {
                linked("Psk8", 16_000.0)
            });

            // RX alone switched: TX still sends 8PSK, so there must be no link.
            rx_ctl
                .send(RxControlMsg::SetModulation(Modulation::Qpsk))
                .unwrap();
            wait_for("rx status shows QPSK", 10.0, || rx_is("Qpsk", 16_000.0));
            std::thread::sleep(Duration::from_millis(2000));
            let s = telemetry.last("rx/status").unwrap();
            assert_ne!(
                s["bert"]["locked_state"], "Synced",
                "mismatched modulation must not decode: {s}"
            );

            // TX follows: the link comes up in QPSK, with the bit rates of QPSK.
            tx_ctl
                .send(TxControlMsg::SetModulation(Modulation::Qpsk))
                .unwrap();
            wait_for("QPSK link", 60.0, || linked("Qpsk", 16_000.0));
            let (t, r) = (
                telemetry.last("tx/status").unwrap(),
                telemetry.last("rx/status").unwrap(),
            );
            assert_eq!(t["modulation"], "Qpsk");
            assert_eq!(t["bit_rate_bps"].as_f64(), Some(32_000.0));
            assert_eq!(r["info_bit_rate_bps"].as_f64(), Some(16_000.0));

            // Symbol rate change on both (BPSK at the same time): 8 ksym/s BPSK.
            for m in [
                TxControlMsg::SetModulation(Modulation::Bpsk),
                TxControlMsg::SetSymbolRateHz(8_000.0),
            ] {
                tx_ctl.send(m).unwrap();
            }
            for m in [
                RxControlMsg::SetModulation(Modulation::Bpsk),
                RxControlMsg::SetSymbolRateHz(8_000.0),
            ] {
                rx_ctl.send(m).unwrap();
            }
            wait_for("8 ksym/s BPSK link", 90.0, || linked("Bpsk", 8_000.0));
            let r = telemetry.last("rx/status").unwrap();
            assert_eq!(r["bit_rate_bps"].as_f64(), Some(8_000.0));
            assert_eq!(r["info_bit_rate_bps"].as_f64(), Some(4_000.0));
            assert!(
                (r["carrier_bandwidth_hz"].as_f64().unwrap() - 50.0).abs() < 1e-6,
                "carrier BW should scale with the rate: {}",
                r["carrier_bandwidth_hz"]
            );

            // Out-of-range rates are rejected and change nothing.
            tx_ctl
                .send(TxControlMsg::SetSymbolRateHz(5_000_000.0))
                .unwrap();
            rx_ctl.send(RxControlMsg::SetSymbolRateHz(10.0)).unwrap();
            std::thread::sleep(Duration::from_millis(500));
            assert_eq!(
                telemetry.last("tx/status").unwrap()["symbol_rate_hz"].as_f64(),
                Some(8_000.0)
            );
            assert!(rx_is("Bpsk", 8_000.0));

            // And to the top of the range, 64 ksym/s 8PSK (occupies ~86 kHz;
            // the mock link has no LO leakage, so the small shift is harmless).
            for m in [
                TxControlMsg::SetModulation(Modulation::Psk8),
                TxControlMsg::SetSymbolRateHz(64_000.0),
                TxControlMsg::SetFreqShiftHz(60_000.0),
            ] {
                tx_ctl.send(m).unwrap();
            }
            for m in [
                RxControlMsg::SetModulation(Modulation::Psk8),
                RxControlMsg::SetSymbolRateHz(64_000.0),
                RxControlMsg::SetFreqShiftHz(60_000.0),
            ] {
                rx_ctl.send(m).unwrap();
            }
            wait_for("64 ksym/s 8PSK link", 90.0, || linked("Psk8", 64_000.0));

            // Back to QPSK 16k for a framed-data check across the rebuilt chains.
            for m in [
                TxControlMsg::SetModulation(Modulation::Qpsk),
                TxControlMsg::SetSymbolRateHz(16_000.0),
                TxControlMsg::SetFreqShiftHz(25_000.0),
                TxControlMsg::SetSource(TxSource::Data),
            ] {
                tx_ctl.send(m).unwrap();
            }
            for m in [
                RxControlMsg::SetModulation(Modulation::Qpsk),
                RxControlMsg::SetSymbolRateHz(16_000.0),
                RxControlMsg::SetFreqShiftHz(25_000.0),
            ] {
                rx_ctl.send(m).unwrap();
            }
            wait_for("QPSK data mode", 30.0, || {
                rx_is("Qpsk", 16_000.0) && telemetry.last("tx/status").unwrap()["source"] == "Data"
            });
            std::thread::sleep(Duration::from_millis(3000));
            let payload = b"frame after live reconfiguration".to_vec();
            client.write_all(&crate::kiss::encode(&payload)).unwrap();
            let mut decoder = KissDecoder::new();
            let mut got = Vec::new();
            let start = Instant::now();
            let mut buf = [0u8; 2048];
            while got.is_empty() && start.elapsed().as_secs() < 40 {
                if let Ok(n) = client.read(&mut buf) {
                    got.extend(decoder.feed(&buf[..n]));
                }
            }
            assert_eq!(
                got,
                vec![payload],
                "frame did not cross the link after reconfiguration"
            );

            shutdown.store(true, Ordering::Relaxed);
            tx_handle
                .join()
                .unwrap()
                .expect("tx engine ended with an error");
            rx_handle
                .join()
                .unwrap()
                .expect("rx engine ended with an error");
        });
    }

    /// `run_engines` across the sample-rate profile boundary: 800 kSPS -> 4 MSPS
    /// -> 800 kSPS as the symbol rate goes 16k -> 250k -> 16k. Each crossing
    /// stops both engines, reconfigures both radios, and the link re-acquires.
    #[test]
    fn run_engines_reconfigures_the_radio_across_the_sample_rate_profiles() {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        drop(listener);
        let kiss = Arc::new(KissServer::start(&addr.to_string()).unwrap());
        std::thread::sleep(Duration::from_millis(50));

        let shutdown = Arc::new(AtomicBool::new(false));
        let telemetry = Arc::new(Collect::default());
        let (link_tx, link_rx) = bounded::<Vec<ComplexI16>>(4);
        let (tx_ctl, tx_ctl_rx) = unbounded::<TxControlMsg>();
        let (rx_ctl, rx_ctl_rx) = unbounded::<RxControlMsg>();
        let zero = || (Arc::new(Mutex::new(0)), Arc::new(Mutex::new(0u64)));
        let ((tx_gain, tx_freq), (rx_gain, rx_freq)) = (zero(), zero());
        let (tx_rates, rx_rates) = (RateLog::default(), RateLog::default());

        let mut tx_engine = TxEngine::new(
            TxConfig {
                modulation: Modulation::Psk8,
                symbol_rate_hz: SYMBOL_RATE_HZ,
                frequency_hz: 1,
                gain_db: -10,
                freq_shift_hz: 25_000.0,
                pattern: PrbsPattern::Pn15,
                source: TxSource::Bert,
            },
            kiss.clone(),
        );
        let mut rx_engine = RxEngine::new(
            RxConfig {
                modulation: Modulation::Psk8,
                symbol_rate_hz: SYMBOL_RATE_HZ,
                frequency_hz: 1,
                gain_db: 30,
                freq_shift_hz: 25_000.0,
                pattern: PrbsPattern::Pn15,
                carrier_bandwidth_hz: DEFAULT_CARRIER_BANDWIDTH_HZ,
                dc_cutoff_hz: DEFAULT_DC_CUTOFF_HZ,
            },
            kiss.clone(),
        );
        let mut mock_tx = MockTx {
            link: link_tx,
            shutdown: shutdown.clone(),
            gain: tx_gain,
            freq: tx_freq,
            rates: tx_rates.clone(),
        };
        let mut mock_rx = MockRx {
            link: link_rx,
            pending: VecDeque::new(),
            shutdown: shutdown.clone(),
            gain: rx_gain,
            freq: rx_freq,
            rates: rx_rates.clone(),
        };

        std::thread::scope(|scope| {
            let _guard = ShutdownGuard(shutdown.clone());
            let supervisor = scope.spawn(|| {
                run_engines(
                    Some((&mut tx_engine, &mut mock_tx, &tx_ctl_rx)),
                    Some((&mut rx_engine, &mut mock_rx, &rx_ctl_rx)),
                    &*telemetry,
                    &shutdown,
                    Duration::from_millis(100),
                )
            });

            let linked = |modulation: &str, rate: f64, sample_rate: u32| {
                telemetry.last("rx/status").is_some_and(|s| {
                    s["modulation"] == modulation
                        && s["symbol_rate_hz"].as_f64() == Some(rate)
                        && s["sample_rate_hz"] == sample_rate
                        && s["carrier_locked"] == true
                        && s["bert"]["locked_state"] == "Synced"
                })
            };
            wait_for("initial link at 800 kSPS", 60.0, || {
                linked("Psk8", 16_000.0, 800_000)
            });
            assert!(
                tx_rates.lock().unwrap().is_empty(),
                "no radio reconfiguration expected yet"
            );

            // Up: BPSK at 250 ksym/s needs 4 MSPS and a ~405 kHz analog filter. The
            // shift must clear the 337 kHz occupied band (half = 169 kHz).
            for m in [
                TxControlMsg::SetModulation(Modulation::Bpsk),
                TxControlMsg::SetFreqShiftHz(200_000.0),
                TxControlMsg::SetSymbolRateHz(250_000.0),
            ] {
                tx_ctl.send(m).unwrap();
            }
            for m in [
                RxControlMsg::SetModulation(Modulation::Bpsk),
                RxControlMsg::SetFreqShiftHz(200_000.0),
                RxControlMsg::SetSymbolRateHz(250_000.0),
            ] {
                rx_ctl.send(m).unwrap();
            }
            wait_for("link at 4 MSPS", 90.0, || {
                linked("Bpsk", 250_000.0, 4_000_000)
            });
            let expected = (4_000_000, analog_bandwidth_hz(250_000.0));
            assert_eq!(tx_rates.lock().unwrap().last(), Some(&expected));
            assert_eq!(rx_rates.lock().unwrap().last(), Some(&expected));
            assert_eq!(
                telemetry.last("tx/status").unwrap()["sample_rate_hz"],
                4_000_000
            );

            // Down again: QPSK at 16 ksym/s is back on the 800 kSPS profile.
            for m in [
                TxControlMsg::SetModulation(Modulation::Qpsk),
                TxControlMsg::SetFreqShiftHz(25_000.0),
                TxControlMsg::SetSymbolRateHz(16_000.0),
            ] {
                tx_ctl.send(m).unwrap();
            }
            for m in [
                RxControlMsg::SetModulation(Modulation::Qpsk),
                RxControlMsg::SetFreqShiftHz(25_000.0),
                RxControlMsg::SetSymbolRateHz(16_000.0),
            ] {
                rx_ctl.send(m).unwrap();
            }
            wait_for("link back at 800 kSPS", 90.0, || {
                linked("Qpsk", 16_000.0, 800_000)
            });
            let expected = (800_000, 200_000);
            assert_eq!(tx_rates.lock().unwrap().last(), Some(&expected));
            assert_eq!(rx_rates.lock().unwrap().last(), Some(&expected));

            shutdown.store(true, Ordering::Relaxed);
            supervisor.join().unwrap().expect("run_engines failed");
        });
    }

    #[test]
    fn a_failing_radio_stops_the_engine_with_an_error() {
        struct DeadTx;
        impl TxRadio for DeadTx {
            fn write(&mut self, _: &[ComplexI16]) -> Result<(), String> {
                Err("usb unplugged".into())
            }
            fn set_gain(&mut self, _: i32) -> Result<(), String> {
                Ok(())
            }
            fn set_frequency(&mut self, _: u64) -> Result<(), String> {
                Ok(())
            }
            fn set_sample_rate(&mut self, _: u32, _: u32) -> Result<(), String> {
                Ok(())
            }
        }
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        drop(listener);
        let kiss = Arc::new(KissServer::start(&addr.to_string()).unwrap());
        let mut engine = TxEngine::new(
            TxConfig {
                modulation: Modulation::Psk8,
                symbol_rate_hz: SYMBOL_RATE_HZ,
                frequency_hz: 1,
                gain_db: 0,
                freq_shift_hz: 15_000.0,
                pattern: PrbsPattern::Pn15,
                source: TxSource::Bert,
            },
            kiss,
        );
        let (_c, rx) = unbounded();
        let err = engine
            .run(
                &mut DeadTx,
                &rx,
                &Collect::default(),
                &AtomicBool::new(false),
                Duration::from_secs(1),
            )
            .unwrap_err();
        assert!(err.contains("usb unplugged"), "{err}");
    }

    #[test]
    fn control_and_status_json_shapes_match_the_documented_contract() {
        // The GUI and any external tooling depend on these exact shapes.
        assert_eq!(
            serde_json::to_string(&TxControlMsg::SetGainDb(-12)).unwrap(),
            r#"{"SetGainDb":-12}"#
        );
        assert_eq!(
            serde_json::to_string(&TxControlMsg::SetSource(TxSource::Data)).unwrap(),
            r#"{"SetSource":"Data"}"#
        );
        assert_eq!(
            serde_json::to_string(&RxControlMsg::SetSearchEnabled(true)).unwrap(),
            r#"{"SetSearchEnabled":true}"#
        );
        assert_eq!(
            serde_json::to_string(&RxControlMsg::Reacquire).unwrap(),
            r#""Reacquire""#
        );
        assert_eq!(
            serde_json::to_string(&RxControlMsg::SetModulation(Modulation::Bpsk)).unwrap(),
            r#"{"SetModulation":"Bpsk"}"#
        );
        assert_eq!(
            serde_json::to_string(&TxControlMsg::SetSymbolRateHz(32000.0)).unwrap(),
            r#"{"SetSymbolRateHz":32000.0}"#
        );
        let parsed: TxControlMsg = serde_json::from_str(r#"{"SetModulation":"Qpsk"}"#).unwrap();
        assert!(matches!(
            parsed,
            TxControlMsg::SetModulation(Modulation::Qpsk)
        ));
        assert_eq!(
            serde_json::to_string(&ModemControlMsg::Shutdown).unwrap(),
            r#""Shutdown""#
        );
        let parsed: RxControlMsg = serde_json::from_str(r#"{"SetFrequencyHz":439500000}"#).unwrap();
        assert!(matches!(parsed, RxControlMsg::SetFrequencyHz(439_500_000)));
        let parsed: RxControlMsg = serde_json::from_str(r#"{"Bert":"ResetStats"}"#).unwrap();
        assert!(matches!(
            parsed,
            RxControlMsg::Bert(RxBertControl::ResetStats)
        ));
    }
}
