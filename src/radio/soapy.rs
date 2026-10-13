//! SoapySDR backend: any radio with a SoapySDR driver module (bladeRF, USRP,
//! LimeSDR, PlutoSDR, ...). The device is chosen with `RadioConfig::device_args`
//! (a SoapySDR args string such as `driver=bladerf` or `driver=uhd,type=b200`)
//! and the ports with the 0-indexed `tx_channel` / `rx_channel`.
//!
//! Streams are CS16 (interleaved int16 I/Q), the modem's native `Iq16`.

use std::time::Duration;

use soapysdr::{Device, Direction, ErrorCode, RxStream, TxStream};

use super::{RadioConfig, Radios};
use crate::modem::{RxRadio, TxRadio};
use crate::Iq16;

const IO_TIMEOUT: Duration = Duration::from_secs(2);
/// USB transfer sizing hint passed as stream args (drivers ignore unknown keys).
const STREAM_ARGS: &str = "buffers=32,buflen=16384";

fn timeout_us() -> i64 {
    IO_TIMEOUT.as_micros() as i64
}

/// Apply the "tx|rx" common tuning. Gain is clamped to the device's range.
fn tune(
    dev: &Device,
    dir: Direction,
    ch: usize,
    name: &str,
    freq_hz: u64,
    gain_db: i32,
    rate: u32,
    bw: u32,
) -> Result<(), String> {
    dev.set_sample_rate(dir, ch, rate as f64)
        .map_err(|e| format!("set {name} sample rate: {e}"))?;
    report_rate(dev, dir, ch, name, rate);
    dev.set_bandwidth(dir, ch, bw as f64)
        .map_err(|e| format!("set {name} bandwidth: {e}"))?;
    dev.set_frequency(dir, ch, freq_hz as f64, ())
        .map_err(|e| format!("set {name} frequency: {e}"))?;
    set_gain(dev, dir, ch, name, gain_db)
}

/// Warn when the device quantized the sample rate (e.g. a USRP master-clock divider):
/// the DSP assumes the requested rate exactly.
fn report_rate(dev: &Device, dir: Direction, ch: usize, name: &str, requested: u32) {
    match dev.sample_rate(dir, ch) {
        Ok(actual) if (actual - requested as f64).abs() > 0.5 => eprintln!(
            "[soapy] warning: {name} sample rate is {actual} Hz, not the requested {requested} Hz; timing and frequencies will be off"
        ),
        Ok(actual) => println!("[soapy] {name} sample rate {actual} Hz"),
        Err(_) => {}
    }
}

fn set_gain(
    dev: &Device,
    dir: Direction,
    ch: usize,
    name: &str,
    gain_db: i32,
) -> Result<(), String> {
    let mut gain = gain_db as f64;
    if let Ok(range) = dev.gain_range(dir, ch) {
        let clamped = gain.clamp(range.minimum, range.maximum);
        if clamped != gain {
            eprintln!(
                "[soapy] {name} gain {gain} dB outside the device range [{}, {}]; using {clamped}",
                range.minimum, range.maximum
            );
            gain = clamped;
        }
    }
    dev.set_gain(dir, ch, gain)
        .map_err(|e| format!("set {name} gain: {e}"))
}

/// Open a stream on one port. Some drivers (the bladeRF 2.0 module) only open
/// a single-channel stream on port 0; for a higher port that fails, so retry
/// over `0..=ch` and let the caller discard (RX) or zero (TX) the other
/// channels. Returns the stream and the wanted port's position in its channel list.
fn open_stream<S>(
    ch: usize,
    open: impl Fn(&[usize]) -> Result<S, soapysdr::Error>,
) -> Result<(S, usize), soapysdr::Error> {
    match open(&[ch]) {
        Ok(stream) => Ok((stream, 0)),
        Err(e) if ch == 0 => Err(e),
        Err(_) => {
            let chans: Vec<usize> = (0..=ch).collect();
            open(&chans).map(|stream| (stream, ch))
        }
    }
}

/// The modem's `Iq16` is full scale +-2048 (SC16Q11, the bladeRF's native
/// range). Other devices stream CS16 at a different full scale (a USRP uses
/// 32768): this is the factor from the modem's range to the device's.
fn scale_to_device(dev: &Device, dir: Direction, ch: usize) -> f32 {
    match dev.native_stream_format(dir, ch) {
        Ok((_, full_scale)) if full_scale > 0.0 => (full_scale / 2048.0) as f32,
        _ => 1.0,
    }
}

/// `dst = src * scale`, saturating to the i16 range.
fn scale_into(dst: &mut Vec<Iq16>, src: &[Iq16], scale: f32) {
    dst.clear();
    dst.extend(src.iter().map(|c| {
        let f = |v: i16| (v as f32 * scale).round().clamp(-32768.0, 32767.0) as i16;
        Iq16::new(f(c.re), f(c.im))
    }));
}

struct SoapyTx {
    dev: Device,
    channel: usize,
    /// `None` only transiently, while `set_sample_rate` recreates it.
    stream: Option<TxStream<Iq16>>,
    /// Position of `channel` in the stream's channel list, and zero filler for the rest.
    wanted: usize,
    zeros: Vec<Iq16>,
    /// Modem range -> device range (1.0 when they match), and its output buffer.
    scale: f32,
    scaled: Vec<Iq16>,
    underflows: u64,
}

impl SoapyTx {
    fn n_streams(&self) -> usize {
        self.wanted + 1
    }
}

impl TxRadio for SoapyTx {
    fn write(&mut self, iq: &[Iq16]) -> Result<(), String> {
        if self.zeros.len() < iq.len() {
            self.zeros.resize(iq.len(), Iq16::new(0, 0));
        }
        let iq = if self.scale == 1.0 {
            iq
        } else {
            scale_into(&mut self.scaled, iq, self.scale);
            &self.scaled[..]
        };
        let zeros = &self.zeros[..iq.len()];
        let bufs: Vec<&[Iq16]> = (0..self.n_streams())
            .map(|i| if i == self.wanted { iq } else { zeros })
            .collect();
        let stream = self.stream.as_mut().ok_or("tx stream is not available")?;
        match stream.write_all(&bufs, None, false, timeout_us()) {
            Ok(()) => Ok(()),
            // Underflow is not fatal: the device ran dry and the stream carries on.
            Err(e) if e.code == ErrorCode::Underflow => {
                self.underflows += 1;
                if self.underflows.is_power_of_two() {
                    eprintln!("[soapy] TX underflow #{}", self.underflows);
                }
                Ok(())
            }
            Err(e) => Err(e.to_string()),
        }
    }
    fn set_gain(&mut self, db: i32) -> Result<(), String> {
        set_gain(&self.dev, Direction::Tx, self.channel, "TX", db)
    }
    fn set_frequency(&mut self, hz: u64) -> Result<(), String> {
        self.dev
            .set_frequency(Direction::Tx, self.channel, hz as f64, ())
            .map_err(|e| e.to_string())
    }
    fn set_sample_rate(&mut self, hz: u32, analog_bandwidth_hz: u32) -> Result<(), String> {
        // The driver sizes its stream for the sample rate when the stream is
        // created, so drop the old stream, retune, then open a fresh one.
        if let Some(mut old) = self.stream.take() {
            old.deactivate(None)
                .map_err(|e| format!("deactivate TX stream: {e}"))?;
        }
        self.dev
            .set_sample_rate(Direction::Tx, self.channel, hz as f64)
            .map_err(|e| format!("set TX sample rate: {e}"))?;
        report_rate(&self.dev, Direction::Tx, self.channel, "TX", hz);
        self.dev
            .set_bandwidth(Direction::Tx, self.channel, analog_bandwidth_hz as f64)
            .map_err(|e| format!("set TX bandwidth: {e}"))?;
        let (mut stream, wanted) = open_stream(self.channel, |c| {
            self.dev.tx_stream_args::<Iq16, _>(c, STREAM_ARGS)
        })
        .map_err(|e| format!("create TX stream: {e}"))?;
        stream
            .activate(None)
            .map_err(|e| format!("activate TX stream: {e}"))?;
        self.wanted = wanted;
        self.stream = Some(stream);
        Ok(())
    }
    fn stop(&mut self) {
        if let Some(stream) = self.stream.as_mut() {
            let _ = stream.deactivate(None);
        }
    }
}

struct SoapyRx {
    dev: Device,
    channel: usize,
    stream: Option<RxStream<Iq16>>,
    /// Position of `channel` in the stream's channel list, and a sink for the rest.
    wanted: usize,
    scratch: Vec<Vec<Iq16>>,
    /// Device range -> modem range divisor (1.0 when they match).
    scale: f32,
    overflows: u64,
}

impl RxRadio for SoapyRx {
    fn read(&mut self, buf: &mut [Iq16]) -> Result<(), String> {
        let mut filled = 0;
        while filled < buf.len() {
            let want = buf.len() - filled;
            for sink in &mut self.scratch {
                if sink.len() < want {
                    sink.resize(want, Iq16::new(0, 0));
                }
            }
            let mut wanted = Some(&mut buf[filled..]);
            let mut sinks = self.scratch.iter_mut();
            let mut bufs: Vec<&mut [Iq16]> = Vec::with_capacity(self.wanted + 1);
            for i in 0..=self.wanted {
                if i == self.wanted {
                    bufs.extend(wanted.take());
                } else if let Some(sink) = sinks.next() {
                    bufs.push(&mut sink[..want]);
                }
            }
            let stream = self.stream.as_mut().ok_or("rx stream is not available")?;
            match stream.read(&mut bufs, timeout_us()) {
                Ok(n) => filled += n,
                // Samples were dropped; carry on with what the stream delivers next.
                Err(e) if e.code == ErrorCode::Overflow => {
                    self.overflows += 1;
                    if self.overflows.is_power_of_two() {
                        eprintln!("[soapy] RX overflow #{}", self.overflows);
                    }
                }
                Err(e) => return Err(e.to_string()),
            }
        }
        if self.scale != 1.0 {
            let inv = 1.0 / self.scale;
            for c in buf.iter_mut() {
                *c = Iq16::new(
                    (c.re as f32 * inv).round() as i16,
                    (c.im as f32 * inv).round() as i16,
                );
            }
        }
        Ok(())
    }
    fn set_gain(&mut self, db: i32) -> Result<(), String> {
        set_gain(&self.dev, Direction::Rx, self.channel, "RX", db)
    }
    fn set_frequency(&mut self, hz: u64) -> Result<(), String> {
        self.dev
            .set_frequency(Direction::Rx, self.channel, hz as f64, ())
            .map_err(|e| e.to_string())
    }
    fn set_sample_rate(&mut self, hz: u32, analog_bandwidth_hz: u32) -> Result<(), String> {
        // The driver sizes its stream for the sample rate when the stream is
        // created, so drop the old stream, retune, then open a fresh one.
        if let Some(mut old) = self.stream.take() {
            old.deactivate(None)
                .map_err(|e| format!("deactivate RX stream: {e}"))?;
        }
        self.dev
            .set_sample_rate(Direction::Rx, self.channel, hz as f64)
            .map_err(|e| format!("set RX sample rate: {e}"))?;
        report_rate(&self.dev, Direction::Rx, self.channel, "RX", hz);
        self.dev
            .set_bandwidth(Direction::Rx, self.channel, analog_bandwidth_hz as f64)
            .map_err(|e| format!("set RX bandwidth: {e}"))?;
        let (mut stream, wanted) = open_stream(self.channel, |c| {
            self.dev.rx_stream_args::<Iq16, _>(c, STREAM_ARGS)
        })
        .map_err(|e| format!("create RX stream: {e}"))?;
        stream
            .activate(None)
            .map_err(|e| format!("activate RX stream: {e}"))?;
        self.wanted = wanted;
        self.stream = Some(stream);
        Ok(())
    }
    fn stop(&mut self) {
        if let Some(stream) = self.stream.as_mut() {
            let _ = stream.deactivate(None);
        }
    }
}

/// Open the SoapySDR device named by `cfg.device_args` and bring up the enabled halves.
pub fn open(cfg: &RadioConfig) -> Result<Radios, String> {
    let dev = Device::new(cfg.device_args.as_str())
        .map_err(|e| format!("cannot open SoapySDR device '{}': {e}", cfg.device_args))?;
    println!(
        "[soapy] {} / {}",
        dev.driver_key().unwrap_or_default(),
        dev.hardware_key().unwrap_or_default()
    );

    // Internal loopback is device specific; map the CLI's names onto the
    // "loopback" setting the bladeRF module exposes, and pass others through.
    if cfg.loopback != "none" {
        let value = match cfg.loopback.as_str() {
            "rfic-bist" => "rf_bist",
            other => other,
        };
        dev.write_setting("loopback", value)
            .map_err(|e| format!("set loopback '{value}': {e}"))?;
    }

    let (rate, bw) = (cfg.sample_rate_hz, cfg.analog_bandwidth_hz);
    let mut radios = Radios { tx: None, rx: None };
    if cfg.tx_enabled {
        let ch = cfg.tx_channel;
        tune(
            &dev,
            Direction::Tx,
            ch,
            "TX",
            cfg.tx_freq_hz,
            cfg.tx_gain_db,
            rate,
            bw,
        )?;
        let (mut stream, wanted) =
            open_stream(ch, |c| dev.tx_stream_args::<Iq16, _>(c, STREAM_ARGS))
                .map_err(|e| format!("create TX stream: {e}"))?;
        stream
            .activate(None)
            .map_err(|e| format!("activate TX stream: {e}"))?;
        radios.tx = Some(Box::new(SoapyTx {
            dev: dev.clone(),
            channel: ch,
            stream: Some(stream),
            wanted,
            zeros: Vec::new(),
            scale: scale_to_device(&dev, Direction::Tx, ch),
            scaled: Vec::new(),
            underflows: 0,
        }));
    }
    if cfg.rx_enabled {
        let ch = cfg.rx_channel;
        tune(
            &dev,
            Direction::Rx,
            ch,
            "RX",
            cfg.rx_freq_hz,
            cfg.rx_gain_db,
            rate,
            bw,
        )?;
        // Manual gain, not AGC: a fixed known gain gives a repeatable level.
        if dev.has_gain_mode(Direction::Rx, ch).unwrap_or(false) {
            dev.set_gain_mode(Direction::Rx, ch, false)
                .map_err(|e| format!("set RX gain mode: {e}"))?;
            set_gain(&dev, Direction::Rx, ch, "RX", cfg.rx_gain_db)?;
        }
        let (mut stream, wanted) =
            open_stream(ch, |c| dev.rx_stream_args::<Iq16, _>(c, STREAM_ARGS))
                .map_err(|e| format!("create RX stream: {e}"))?;
        stream
            .activate(None)
            .map_err(|e| format!("activate RX stream: {e}"))?;
        radios.rx = Some(Box::new(SoapyRx {
            dev: dev.clone(),
            channel: ch,
            stream: Some(stream),
            wanted,
            scratch: (0..wanted).map(|_| Vec::new()).collect(),
            scale: scale_to_device(&dev, Direction::Rx, ch),
            overflows: 0,
        }));
    }
    Ok(radios)
}
