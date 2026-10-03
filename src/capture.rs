//! Shared TX+RX loopback capture procedure: configure the device, stream TX
//! (via `TxSignalGenerator`) and RX concurrently, capture N samples, and
//! save both raw IQ and a Welch PSD estimate to disk. Used by both
//! `loopback_capture` (internal RFIC_BIST loopback) and `rf_loopback_capture`
//! (real TX1 -> RX2 via cable), which differ only in device configuration,
//! not in the capture/analysis procedure itself.
//!
//! Device open/configure is split from capturing (`OpenCapture::open` vs
//! `OpenCapture::capture`) so a caller can keep one device session open
//! across multiple captures - needed for anything that adjusts live device
//! state (e.g. `set_correction`) between captures, since state like that
//! appears to reset when the device is reopened (matching what was
//! observed with gain/AGC settings resetting between separate bladeRF-cli
//! invocations). `run_capture` is a convenience wrapper for the common
//! single-capture case.

use std::fs::File;
use std::io::{BufWriter, Write};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::Duration;

use bladerf::{
    BladeRF, BladeRfAny, Channel, ChannelLayoutRx, ChannelLayoutTx, ComplexI16, GainMode, Loopback,
    RxChannel, RxSyncStream, StreamConfig, TxChannel, TxSyncStream,
};

use crate::prbs::PrbsPattern;
use crate::psd::{complex_i16_to_f32, welch_psd};
use crate::rrc::ShapeMode;
use crate::tx_signal::TxSignalGenerator;

pub use crate::params::{HARDWARE_SAMPLE_RATE, RRC_ROLLOFF, RRC_SPAN_SYMBOLS, SPS};
pub const BANDWIDTH: u32 = 200_000;
pub const CHUNK_FRAMES: usize = 8192;
pub const MAX_RESAMPLE_RATIO_RELATIVE: f64 = 10.0;
pub const PSD_FFT_SIZE: usize = 4096;

pub fn parse_pattern(s: &str) -> PrbsPattern {
    match s.to_lowercase().as_str() {
        "pn11" => PrbsPattern::Pn11,
        "pn23" => PrbsPattern::Pn23,
        "pn15" => PrbsPattern::Pn15,
        other => {
            eprintln!("Unknown pattern '{other}', defaulting to pn15");
            PrbsPattern::Pn15
        }
    }
}

#[derive(Clone, Copy)]
pub struct CaptureConfig {
    pub duration_s: f64,
    pub symbol_rate_hz: f64,
    pub pattern: PrbsPattern,
    pub shape_mode: ShapeMode,
    /// Extra linear backoff on top of the RRC's own calibrated headroom -
    /// see TxSignalGenerator. 1.0 for a real RF path where the TX/RX gain
    /// controls do the level-setting; lower for loopback modes that bypass
    /// those controls.
    pub digital_scale: f32,
    pub tx_channel: TxChannel,
    pub rx_channel: RxChannel,
    /// None skips setting frequency at all (irrelevant in some loopback modes).
    pub frequency_hz: Option<u64>,
    pub tx_gain_db: i32,
    pub rx_gain_db: i32,
    pub loopback: Loopback,
    /// Constant (I, Q) pre-compensation bias added to every TX sample -
    /// see `TxSignalGenerator::set_tx_dc_bias`. (0.0, 0.0) for no compensation.
    pub tx_dc_bias: (f32, f32),
    /// Constant frequency shift (Hz) applied to the TX signal in DSP before
    /// quantization - offset-tunes the transmitted signal away from the LO
    /// frequency so carrier/LO leakage (which stays at the LO frequency)
    /// lands outside the signal's occupied bandwidth. 0.0 for no shift.
    pub tx_freq_shift_hz: f64,
}

pub struct CaptureResult {
    pub samples: Vec<ComplexI16>,
    pub peak_magnitude: i16,
}

/// An opened, configured device with live TX/RX streams, ready to capture
/// repeatedly. Kept open across captures so device-state changes made
/// between captures (e.g. `dev().set_correction(...)`) actually take effect
/// on the next capture rather than being reset by a fresh device open.
pub struct OpenCapture {
    dev: Arc<BladeRfAny>,
    tx_stream: TxSyncStream<Arc<BladeRfAny>, ComplexI16, BladeRfAny>,
    rx_stream: RxSyncStream<Arc<BladeRfAny>, ComplexI16, BladeRfAny>,
    tx_channel: TxChannel,
    rx_channel: Channel,
}

impl OpenCapture {
    pub fn open(cfg: &CaptureConfig) -> Self {
        let dev = Arc::new(BladeRfAny::open_first().expect("open bladeRF (is it plugged in?)"));
        if !dev.is_fpga_configured().unwrap_or(false) {
            eprintln!("FPGA is not loaded. Run, e.g.:");
            eprintln!("  bladeRF-cli -e \"load fpga /usr/share/bladerf/fpga/hostedxA4.rbf\"");
            std::process::exit(1);
        }

        // SAFETY: a valid Loopback variant is always passed by this module's callers.
        unsafe { dev.set_loopback(cfg.loopback) }.expect("set loopback mode");

        let tx_ch: Channel = cfg.tx_channel.into();
        let rx_ch: Channel = cfg.rx_channel.into();

        if let Some(freq) = cfg.frequency_hz {
            dev.set_frequency(tx_ch, freq).expect("set tx frequency");
            dev.set_frequency(rx_ch, freq).expect("set rx frequency");
        }
        dev.set_sample_rate(tx_ch, HARDWARE_SAMPLE_RATE)
            .expect("set tx sample rate");
        dev.set_sample_rate(rx_ch, HARDWARE_SAMPLE_RATE)
            .expect("set rx sample rate");
        dev.set_bandwidth(tx_ch, BANDWIDTH)
            .expect("set tx bandwidth");
        dev.set_bandwidth(rx_ch, BANDWIDTH)
            .expect("set rx bandwidth");
        dev.set_gain(tx_ch, cfg.tx_gain_db).expect("set tx gain");
        // Manual, not AGC (the device's default): a fixed known gain gives a
        // repeatable level across runs instead of AGC settling wherever it settles.
        dev.set_gain_mode(rx_ch, GainMode::Manual)
            .expect("set rx gain mode");
        dev.set_gain(rx_ch, cfg.rx_gain_db).expect("set rx gain");

        let stream_config = StreamConfig::default();
        let tx_stream = BladeRfAny::tx_streamer_arc::<ComplexI16>(
            dev.clone(),
            stream_config,
            ChannelLayoutTx::SISO(cfg.tx_channel),
        )
        .expect("create tx streamer");
        let rx_stream = BladeRfAny::rx_streamer_arc::<ComplexI16>(
            dev.clone(),
            stream_config,
            ChannelLayoutRx::SISO(cfg.rx_channel),
        )
        .expect("create rx streamer");

        tx_stream.enable().expect("enable tx");
        rx_stream.enable().expect("enable rx");

        OpenCapture {
            dev,
            tx_stream,
            rx_stream,
            tx_channel: cfg.tx_channel,
            rx_channel: rx_ch,
        }
    }

    /// The underlying device handle - use for e.g. `set_correction`/`get_correction`
    /// between captures.
    pub fn dev(&self) -> &Arc<BladeRfAny> {
        &self.dev
    }

    pub fn rx_channel(&self) -> Channel {
        self.rx_channel
    }

    pub fn tx_channel(&self) -> TxChannel {
        self.tx_channel
    }

    /// Run one capture using this already-open device/streams. Only the
    /// signal-generation parameters are needed here - device config (gain,
    /// frequency, loopback, channels) was already applied in `open`.
    #[allow(clippy::too_many_arguments)]
    pub fn capture(
        &self,
        duration_s: f64,
        symbol_rate_hz: f64,
        pattern: PrbsPattern,
        shape_mode: ShapeMode,
        digital_scale: f32,
        tx_dc_bias: (f32, f32),
        tx_freq_shift_hz: f64,
    ) -> CaptureResult {
        let stop = Arc::new(AtomicBool::new(false));
        let captured = {
            let tx_stream = &self.tx_stream;
            thread::scope(|scope| {
                let tx_stop = stop.clone();
                scope.spawn(move || {
                    let stop = tx_stop;
                    let mut generator = TxSignalGenerator::new(
                        pattern,
                        crate::symbol_map::Modulation::Psk8,
                        symbol_rate_hz,
                        HARDWARE_SAMPLE_RATE,
                        SPS,
                        RRC_ROLLOFF,
                        RRC_SPAN_SYMBOLS,
                        shape_mode,
                        CHUNK_FRAMES,
                        MAX_RESAMPLE_RATIO_RELATIVE,
                        digital_scale,
                        tx_freq_shift_hz,
                    );
                    generator.set_tx_dc_bias(tx_dc_bias.0, tx_dc_bias.1);
                    while !stop.load(Ordering::Relaxed) {
                        let iq = generator.next_chunk();
                        if tx_stream.write(&iq, Duration::from_secs(2)).is_err() {
                            break;
                        }
                    }
                });

                let total_samples = (duration_s * HARDWARE_SAMPLE_RATE as f64).round() as usize;
                let mut captured: Vec<ComplexI16> = Vec::with_capacity(total_samples);
                let mut buf = vec![ComplexI16::new(0, 0); CHUNK_FRAMES];

                while captured.len() < total_samples {
                    self.rx_stream
                        .read(&mut buf, Duration::from_secs(2))
                        .expect("rx stream read");
                    captured.extend_from_slice(&buf);
                }
                captured.truncate(total_samples);
                stop.store(true, Ordering::Relaxed);
                captured
            })
        };

        let peak = captured
            .iter()
            .map(|c| c.re.unsigned_abs().max(c.im.unsigned_abs()))
            .max()
            .unwrap_or(0) as i16;

        CaptureResult {
            samples: captured,
            peak_magnitude: peak,
        }
    }
}

pub fn run_capture(cfg: &CaptureConfig) -> CaptureResult {
    let open = OpenCapture::open(cfg);
    open.capture(
        cfg.duration_s,
        cfg.symbol_rate_hz,
        cfg.pattern,
        cfg.shape_mode,
        cfg.digital_scale,
        cfg.tx_dc_bias,
        cfg.tx_freq_shift_hz,
    )
}

pub fn report_level(peak: i16) {
    println!(
        "Capture complete. Peak sample magnitude: {peak} / 2048 full scale ({:.1}%)",
        100.0 * peak as f64 / 2048.0
    );
    if peak >= 2047 {
        println!("WARNING: capture appears clipped - consider lowering RX/TX gain.");
    } else if peak < 100 {
        println!("WARNING: capture looks very weak - consider raising RX and/or TX gain.");
    }
}

pub fn save_iq_and_psd(samples: &[ComplexI16], iq_out_path: &str, psd_out_path: &str) {
    {
        let mut f = BufWriter::new(File::create(iq_out_path).expect("create IQ output file"));
        for c in samples {
            f.write_all(&c.re.to_le_bytes()).unwrap();
            f.write_all(&c.im.to_le_bytes()).unwrap();
        }
        println!("Wrote raw IQ (i16 interleaved, little-endian) to {iq_out_path}");
    }

    let iq_f32 = complex_i16_to_f32(samples);
    let fft_size = PSD_FFT_SIZE.min(iq_f32.len());
    let psd = welch_psd(&iq_f32, HARDWARE_SAMPLE_RATE as f64, fft_size, 0.5);
    {
        let mut f = BufWriter::new(File::create(psd_out_path).expect("create PSD output file"));
        writeln!(f, "freq_hz,power_db").unwrap();
        for p in &psd {
            writeln!(f, "{},{}", p.freq_hz, p.power_db).unwrap();
        }
        println!("Wrote PSD ({} points) to {psd_out_path}", psd.len());
    }
}
