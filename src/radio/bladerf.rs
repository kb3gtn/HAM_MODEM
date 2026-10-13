//! bladeRF backend (libbladeRF via the `bladerf` crate): TX on the physical
//! TX1 port, RX on RX2 (or RX1 in internal-loopback mode).

use std::sync::Arc;
use std::time::Duration;

use bladerf::{
    BladeRF, BladeRfAny, Channel, ChannelLayoutRx, ChannelLayoutTx, GainMode, Loopback, RxChannel,
    RxSyncStream, StreamConfig, TxChannel, TxSyncStream,
};

use super::{RadioConfig, Radios};
use crate::modem::{RxRadio, TxRadio};
use crate::params::chunk_frames_for;
use crate::Iq16;

type TxStream = TxSyncStream<Arc<BladeRfAny>, Iq16, BladeRfAny>;
type RxStream = RxSyncStream<Arc<BladeRfAny>, Iq16, BladeRfAny>;

/// Stream buffers sized to the sample rate (~160 ms of samples in 16 buffers).
fn stream_config_for(hardware_sample_rate: u32) -> StreamConfig {
    StreamConfig::new(
        16,
        chunk_frames_for(hardware_sample_rate),
        8,
        Duration::from_millis(3500),
    )
    .expect("valid stream config")
}

struct BladeTx {
    dev: Arc<BladeRfAny>,
    channel: Channel,
    layout: TxChannel,
    /// `None` only transiently, while `set_sample_rate` reconfigures it.
    stream: Option<TxStream>,
}

impl TxRadio for BladeTx {
    fn write(&mut self, iq: &[Iq16]) -> Result<(), String> {
        self.stream
            .as_ref()
            .ok_or("tx stream is not available")?
            .write(iq, Duration::from_secs(2))
            .map_err(|e| e.to_string())
    }
    fn set_gain(&mut self, db: i32) -> Result<(), String> {
        self.dev
            .set_gain(self.channel, db)
            .map_err(|e| e.to_string())
    }
    fn set_frequency(&mut self, hz: u64) -> Result<(), String> {
        self.dev
            .set_frequency(self.channel, hz)
            .map_err(|e| e.to_string())
    }
    fn set_sample_rate(&mut self, hz: u32, analog_bandwidth_hz: u32) -> Result<(), String> {
        let stream = self.stream.take().ok_or("tx stream is not available")?;
        stream
            .disable()
            .map_err(|e| format!("disable TX stream: {e}"))?;
        self.dev
            .set_sample_rate(self.channel, hz)
            .map_err(|e| format!("set TX sample rate: {e}"))?;
        self.dev
            .set_bandwidth(self.channel, analog_bandwidth_hz)
            .map_err(|e| format!("set TX bandwidth: {e}"))?;
        let stream = stream
            .reconfigure::<Iq16>(stream_config_for(hz), ChannelLayoutTx::SISO(self.layout))
            .map_err(|e| format!("reconfigure TX stream: {e}"))?;
        stream
            .enable()
            .map_err(|e| format!("enable TX stream: {e}"))?;
        self.stream = Some(stream);
        Ok(())
    }
    fn stop(&mut self) {
        if let Some(stream) = self.stream.as_ref() {
            let _ = stream.disable();
        }
    }
}

struct BladeRx {
    dev: Arc<BladeRfAny>,
    channel: Channel,
    layout: RxChannel,
    stream: Option<RxStream>,
}

impl RxRadio for BladeRx {
    fn read(&mut self, buf: &mut [Iq16]) -> Result<(), String> {
        self.stream
            .as_ref()
            .ok_or("rx stream is not available")?
            .read(buf, Duration::from_secs(2))
            .map_err(|e| e.to_string())
    }
    fn set_gain(&mut self, db: i32) -> Result<(), String> {
        self.dev
            .set_gain(self.channel, db)
            .map_err(|e| e.to_string())
    }
    fn set_frequency(&mut self, hz: u64) -> Result<(), String> {
        self.dev
            .set_frequency(self.channel, hz)
            .map_err(|e| e.to_string())
    }
    fn set_sample_rate(&mut self, hz: u32, analog_bandwidth_hz: u32) -> Result<(), String> {
        let stream = self.stream.take().ok_or("rx stream is not available")?;
        stream
            .disable()
            .map_err(|e| format!("disable RX stream: {e}"))?;
        self.dev
            .set_sample_rate(self.channel, hz)
            .map_err(|e| format!("set RX sample rate: {e}"))?;
        self.dev
            .set_bandwidth(self.channel, analog_bandwidth_hz)
            .map_err(|e| format!("set RX bandwidth: {e}"))?;
        let stream = stream
            .reconfigure::<Iq16>(stream_config_for(hz), ChannelLayoutRx::SISO(self.layout))
            .map_err(|e| format!("reconfigure RX stream: {e}"))?;
        stream
            .enable()
            .map_err(|e| format!("enable RX stream: {e}"))?;
        self.stream = Some(stream);
        Ok(())
    }
    fn stop(&mut self) {
        if let Some(stream) = self.stream.as_ref() {
            let _ = stream.disable();
        }
    }
}

/// Open the first bladeRF and bring up the enabled halves.
pub fn open(cfg: &RadioConfig) -> Result<Radios, String> {
    let dev = Arc::new(BladeRfAny::open_first().map_err(|e| {
        format!("cannot open the bladeRF (is it plugged in? is another program using it?): {e}")
    })?);
    if !dev.is_fpga_configured().unwrap_or(false) {
        return Err("FPGA is not loaded. Run, e.g.:\n  bladeRF-cli -e \"load fpga /usr/share/bladerf/fpga/hostedxA4.rbf\"".into());
    }
    let loopback_on = cfg.loopback != "none";
    let loopback_mode = match cfg.loopback.as_str() {
        "none" => Loopback::None,
        "rfic-bist" => Loopback::RficBist,
        other => return Err(format!("unknown bladeRF loopback mode '{other}'")),
    };
    // SAFETY: only valid Loopback variants are constructed above.
    unsafe { dev.set_loopback(loopback_mode) }.map_err(|e| format!("set loopback mode: {e}"))?;

    // IMPORTANT: this crate's channel enums are 0-indexed (Tx0, Rx0, Rx1),
    // matching libbladeRF's internal macros, while Nuand's silkscreen and
    // bladeRF-cli call the ports "TX1"/"RX1"/"RX2" (1-indexed). So physical
    // TX1 = Tx0 and physical RX2 = Rx1. The internal loopbacks pair TX1 with
    // RX1 (= Rx0).
    let tx_channel = TxChannel::Tx0;
    let rx_channel = if loopback_on {
        RxChannel::Rx0
    } else {
        RxChannel::Rx1
    };
    let tx_ch: Channel = tx_channel.into();
    let rx_ch: Channel = rx_channel.into();
    let (rate, bw) = (cfg.sample_rate_hz, cfg.analog_bandwidth_hz);

    if cfg.tx_enabled {
        dev.set_frequency(tx_ch, cfg.tx_freq_hz)
            .map_err(|e| format!("set TX frequency: {e}"))?;
        dev.set_sample_rate(tx_ch, rate)
            .map_err(|e| format!("set TX sample rate: {e}"))?;
        dev.set_bandwidth(tx_ch, bw)
            .map_err(|e| format!("set TX bandwidth: {e}"))?;
        dev.set_gain(tx_ch, cfg.tx_gain_db)
            .map_err(|e| format!("set TX gain: {e}"))?;
    }
    if cfg.rx_enabled {
        dev.set_frequency(rx_ch, cfg.rx_freq_hz)
            .map_err(|e| format!("set RX frequency: {e}"))?;
        dev.set_sample_rate(rx_ch, rate)
            .map_err(|e| format!("set RX sample rate: {e}"))?;
        dev.set_bandwidth(rx_ch, bw)
            .map_err(|e| format!("set RX bandwidth: {e}"))?;
        // Manual, not AGC: a fixed known gain gives a repeatable level.
        dev.set_gain_mode(rx_ch, GainMode::Manual)
            .map_err(|e| format!("set RX gain mode: {e}"))?;
        dev.set_gain(rx_ch, cfg.rx_gain_db)
            .map_err(|e| format!("set RX gain: {e}"))?;
    }

    let stream_config = stream_config_for(rate);
    let mut radios = Radios { tx: None, rx: None };
    if cfg.tx_enabled {
        let stream = BladeRfAny::tx_streamer_arc::<Iq16>(
            dev.clone(),
            stream_config,
            ChannelLayoutTx::SISO(tx_channel),
        )
        .map_err(|e| format!("create TX streamer: {e}"))?;
        stream.enable().map_err(|e| format!("enable TX1: {e}"))?;
        radios.tx = Some(Box::new(BladeTx {
            dev: dev.clone(),
            channel: tx_ch,
            layout: tx_channel,
            stream: Some(stream),
        }));
    }
    if cfg.rx_enabled {
        let stream = BladeRfAny::rx_streamer_arc::<Iq16>(
            dev.clone(),
            stream_config,
            ChannelLayoutRx::SISO(rx_channel),
        )
        .map_err(|e| format!("create RX streamer: {e}"))?;
        stream.enable().map_err(|e| format!("enable RX: {e}"))?;
        radios.rx = Some(Box::new(BladeRx {
            dev: dev.clone(),
            channel: rx_ch,
            layout: rx_channel,
            stream: Some(stream),
        }));
    }
    Ok(radios)
}
