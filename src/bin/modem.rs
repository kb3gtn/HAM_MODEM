//! The BPSK/QPSK/8PSK modem: transmitter and receiver in one process, on one bladeRF
//! (TX1 and RX2 running together, each at its own frequency), with one KISS
//! socket and one MQTT connection. Replaces the old separate `tx_path` /
//! `rx_path` binaries (`--no-rx` / `--no-tx` give either half on its own).
//!
//! MODULATION: BPSK / QPSK / 8PSK (1 / 2 / 3 bits per symbol, Gray mapped),
//! selectable at start-up (--modulation) and live over MQTT, at a selectable
//! symbol rate of 2 ksym/s to 1 Msym/s (default 16 ksym/s, i.e. 48 kbps 8PSK on the
//! air), RRC rolloff 0.35 -> occupied bandwidth 1.35 x the symbol rate (21.6 kHz
//! at the default; there is no bandwidth cap - see `params`). A K=7 rate-1/2
//! convolutional code with a soft-decision Viterbi decoder halves the data rate
//! before HDLC overhead. The receiver acquires frequency offsets (up to 0.375
//! x the symbol rate, +-6 kHz at the default) from the signal spectrum and
//! resolves the constellation's phase ambiguity in its decoders. Changing the
//! modulation or symbol rate on the TX and RX drops the link, which then
//! re-acquires. The SDR runs at 800 kSPS up to 100 ksym/s and at 4 MSPS above it
//! (the radio is stopped, reconfigured and restarted on a change that crosses
//! that boundary, with the analog filter following the signal width).
//! See `modem` (engines, `run_engines`), `rx_signal`, `coarse_freq`.
//!
//! USAGE
//!   modem [options]
//!     --tx-freq HZ       TX LO frequency                    (439500000)
//!     --rx-freq HZ       RX LO frequency                    (439500000)
//!     --tx-gain DB       TX1 gain, device range [-23.75,66] (-23)
//!     --rx-gain DB       RX gain                            (50)
//!     --tx-shift HZ      TX DSP offset tune (keeps LO leakage out of band) (15000)
//!     --rx-shift HZ      RX DSP shift; must equal the far end's --tx-shift (15000)
//!     --pattern P        BERT pattern: pn11 | pn15 | pn23    (pn15)
//!     --source S         initial TX source: bert | data      (bert)
//!     --modulation M     bpsk | qpsk | 8psk (TX and RX)       (8psk)
//!     --symbol-rate HZ   symbol rate, 2000..1000000 (TX and RX) (16000)
//!     --carrier-bw HZ    receiver carrier loop bandwidth     (100, scaled with the symbol rate)
//!     --dc-cutoff HZ     receiver DC blocker cutoff          (50)
//!     --kiss-port N      KISS TCP port (TX frames in, RX frames out) (8001)
//!     --mqtt-host H --mqtt-port N --mqtt-topic T             (127.0.0.1 1883 psk8)
//!     --no-tx / --no-rx  run only one half (TX1 untouched / no receive chain)
//!     --loopback MODE    none | rfic-bist: loop TX back into the RX inside the
//!                        radio chip, for a single-radio self-test (uses RX1; no
//!                        antenna/cable needed). Use --tx-gain -23 --rx-gain 0:
//!                        the loopback adds its own gain.                    (none)
//!
//! MQTT (base topic <T>; the GUI speaks all of these):
//!   <T>/tx/status, <T>/rx/status (retained, 1 Hz), <T>/rx/symbols (1 Hz, the
//!   latest 256 symbols), <T>/tx/control, <T>/rx/control, and
//!   <T>/modem/status (retained; "online" with pid/uptime, or "offline" - also
//!   published by the broker as the last will if the process dies) and
//!   <T>/modem/control (`"Shutdown"` stops both engines and exits cleanly).
//!   JSON shapes are the serde types in `modem.rs`.
//!
//! Typing `bert` or `data` + Enter on stdin switches the TX source.
//!
//! THREADS: one TX thread (paced by the radio write) and one RX thread (paced
//! by the radio read), one MQTT thread routing commands, and this main thread
//! publishing the process status and watching for shutdown or failure.
//!
//! This transmits real RF. Confirm you're authorized to transmit at the chosen
//! frequency and that your RF path (attenuator/antenna) is set up safely.

use std::io::BufRead;
use std::process::exit;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use bladerf::{
    BladeRF, BladeRfAny, Channel, ChannelLayoutRx, ChannelLayoutTx, ComplexI16, GainMode, Loopback,
    RxChannel, RxSyncStream, StreamConfig, TxChannel, TxSyncStream,
};
use crossbeam_channel::{bounded, Sender};
use rumqttc::{Client, Event, LastWill, MqttOptions, Packet, QoS};

use ham_modem::capture::parse_pattern;
use ham_modem::kiss::KissServer;
use ham_modem::modem::*;
use ham_modem::params::*;
use ham_modem::prbs::PrbsPattern;
use ham_modem::symbol_map::Modulation;

const STATUS_INTERVAL: Duration = Duration::from_secs(1);

// ------------------------------------------------------------------------
// Command line
// ------------------------------------------------------------------------

const USAGE: &str = "8PSK modem (TX1 + RX2 on one bladeRF)

usage: modem [options]
  --tx-freq HZ       TX LO frequency                              (439500000)
  --rx-freq HZ       RX LO frequency                              (439500000)
  --tx-gain DB       TX1 gain, device range [-23.75, 66]          (-23)
  --rx-gain DB       RX gain                                      (50)
  --tx-shift HZ      TX DSP offset tune, keeps LO leakage out of band (15000)
  --rx-shift HZ      RX DSP shift; must equal the far end's --tx-shift (15000)
  --pattern P        BERT pattern: pn11 | pn15 | pn23             (pn15)
  --source S         initial TX source: bert | data               (bert)
  --modulation M     bpsk | qpsk | 8psk, for TX and RX            (8psk)
  --symbol-rate HZ   symbol rate 2000..1000000, for TX and RX       (16000)
                     (occupied bandwidth is 1.35 x this; raise --tx-shift /
                     --rx-shift above half of it)
  --carrier-bw HZ    receiver carrier loop bandwidth              (100 at 16000 sym/s, scaled with the rate)
  --dc-cutoff HZ     receiver DC blocker cutoff                   (50)
  --kiss-port N      KISS TCP port, TX frames in / RX frames out  (8001)
  --mqtt-host H      MQTT broker host                             (127.0.0.1)
  --mqtt-port N      MQTT broker port                             (1883)
  --mqtt-topic T     MQTT base topic                              (psk8)
  --no-tx            receive only (TX1 untouched)
  --no-rx            transmit only
  --loopback MODE    none | rfic-bist: loop TX back into RX1 inside the radio
                     chip for a single-radio self-test (none). Use
                     --tx-gain -23 --rx-gain 0; the loopback adds its own gain.
  -h, --help         this text";

struct Args {
    tx_freq: u64,
    rx_freq: u64,
    tx_gain: i32,
    rx_gain: i32,
    tx_shift: f64,
    rx_shift: f64,
    pattern: PrbsPattern,
    source: TxSource,
    carrier_bw: f64,
    carrier_bw_set: bool,
    modulation: Modulation,
    symbol_rate: f64,
    dc_cutoff: f64,
    kiss_port: u16,
    mqtt_host: String,
    mqtt_port: u16,
    mqtt_topic: String,
    tx_enabled: bool,
    rx_enabled: bool,
    loopback: String,
}

impl Args {
    fn defaults() -> Self {
        Args {
            tx_freq: 439_500_000,
            rx_freq: 439_500_000,
            tx_gain: -23, // conservative: near the bottom of the device's TX1 range
            rx_gain: 50,
            tx_shift: 15_000.0,
            rx_shift: 15_000.0,
            pattern: PrbsPattern::Pn15,
            source: TxSource::Bert,
            carrier_bw: DEFAULT_CARRIER_BANDWIDTH_HZ,
            carrier_bw_set: false,
            modulation: DEFAULT_MODULATION,
            symbol_rate: SYMBOL_RATE_HZ,
            dc_cutoff: DEFAULT_DC_CUTOFF_HZ,
            kiss_port: 8001,
            mqtt_host: "127.0.0.1".into(),
            mqtt_port: 1883,
            mqtt_topic: "psk8".into(),
            tx_enabled: true,
            rx_enabled: true,
            loopback: "none".into(),
        }
    }

    /// `Ok(None)` means help was requested and printed.
    fn parse(argv: &[String]) -> Result<Option<Args>, String> {
        let mut a = Args::defaults();
        let mut i = 0;
        while i < argv.len() {
            let flag = argv[i].as_str();
            let mut value = |name: &str| -> Result<String, String> {
                i += 1;
                argv.get(i)
                    .cloned()
                    .ok_or_else(|| format!("{name} needs a value"))
            };
            fn num<T: std::str::FromStr>(name: &str, s: String) -> Result<T, String> {
                s.parse()
                    .map_err(|_| format!("{name}: '{s}' is not a valid number"))
            }
            match flag {
                "-h" | "--help" => {
                    println!("{USAGE}");
                    return Ok(None);
                }
                "--tx-freq" => a.tx_freq = num(flag, value(flag)?)?,
                "--rx-freq" => a.rx_freq = num(flag, value(flag)?)?,
                "--tx-gain" => a.tx_gain = num(flag, value(flag)?)?,
                "--rx-gain" => a.rx_gain = num(flag, value(flag)?)?,
                "--tx-shift" => a.tx_shift = num(flag, value(flag)?)?,
                "--rx-shift" => a.rx_shift = num(flag, value(flag)?)?,
                "--pattern" => a.pattern = parse_pattern(&value(flag)?),
                "--source" => {
                    a.source = match value(flag)?.to_lowercase().as_str() {
                        "bert" => TxSource::Bert,
                        "data" => TxSource::Data,
                        other => {
                            return Err(format!("--source must be bert or data, not '{other}'"))
                        }
                    }
                }
                "--carrier-bw" => {
                    a.carrier_bw = num(flag, value(flag)?)?;
                    a.carrier_bw_set = true;
                }
                "--modulation" => {
                    let v = value(flag)?;
                    a.modulation = Modulation::parse(&v).ok_or_else(|| {
                        format!("--modulation must be bpsk, qpsk or 8psk, not '{v}'")
                    })?;
                }
                "--symbol-rate" => {
                    a.symbol_rate = num(flag, value(flag)?)?;
                    if !symbol_rate_in_range(a.symbol_rate) {
                        return Err(format!("--symbol-rate must be {MIN_SYMBOL_RATE_HZ}..={MAX_SYMBOL_RATE_HZ}, not {}", a.symbol_rate));
                    }
                }
                "--dc-cutoff" => a.dc_cutoff = num(flag, value(flag)?)?,
                "--kiss-port" => a.kiss_port = num(flag, value(flag)?)?,
                "--mqtt-host" => a.mqtt_host = value(flag)?,
                "--mqtt-port" => a.mqtt_port = num(flag, value(flag)?)?,
                "--mqtt-topic" => a.mqtt_topic = value(flag)?,
                "--no-tx" => a.tx_enabled = false,
                "--no-rx" => a.rx_enabled = false,
                "--loopback" => {
                    a.loopback = value(flag)?.to_lowercase();
                    // (The FPGA "firmware" loopback is deliberately not offered: it
                    // isn't paced by the sample clock, so TX free-runs, the RX
                    // can't keep up and the radio drops samples.)
                    if !["none", "rfic-bist"].contains(&a.loopback.as_str()) {
                        return Err(format!(
                            "--loopback must be none or rfic-bist, not '{}'",
                            a.loopback
                        ));
                    }
                }
                other => return Err(format!("unknown option '{other}' (try --help)")),
            }
            i += 1;
        }
        if !a.carrier_bw_set {
            a.carrier_bw = DEFAULT_CARRIER_BANDWIDTH_HZ * a.symbol_rate / SYMBOL_RATE_HZ;
        }
        if !a.tx_enabled && !a.rx_enabled {
            return Err("--no-tx and --no-rx together leave nothing to run".into());
        }
        Ok(Some(a))
    }
}

// ------------------------------------------------------------------------
// bladeRF radios
// ------------------------------------------------------------------------

type TxStream = TxSyncStream<Arc<BladeRfAny>, ComplexI16, BladeRfAny>;
type RxStream = RxSyncStream<Arc<BladeRfAny>, ComplexI16, BladeRfAny>;

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
    fn write(&mut self, iq: &[ComplexI16]) -> Result<(), String> {
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
            .reconfigure::<ComplexI16>(stream_config_for(hz), ChannelLayoutTx::SISO(self.layout))
            .map_err(|e| format!("reconfigure TX stream: {e}"))?;
        stream
            .enable()
            .map_err(|e| format!("enable TX stream: {e}"))?;
        self.stream = Some(stream);
        Ok(())
    }
}

struct BladeRx {
    dev: Arc<BladeRfAny>,
    channel: Channel,
    layout: RxChannel,
    stream: Option<RxStream>,
}

impl RxRadio for BladeRx {
    fn read(&mut self, buf: &mut [ComplexI16]) -> Result<(), String> {
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
            .reconfigure::<ComplexI16>(stream_config_for(hz), ChannelLayoutRx::SISO(self.layout))
            .map_err(|e| format!("reconfigure RX stream: {e}"))?;
        stream
            .enable()
            .map_err(|e| format!("enable RX stream: {e}"))?;
        self.stream = Some(stream);
        Ok(())
    }
}

fn fail(msg: impl std::fmt::Display) -> ! {
    eprintln!("modem: {msg}");
    exit(1);
}

// ------------------------------------------------------------------------
// MQTT
// ------------------------------------------------------------------------

struct MqttTelemetry {
    client: Client,
    base: String,
}

impl Telemetry for MqttTelemetry {
    fn publish(&self, topic_suffix: &str, payload: &str, retain: bool) {
        // try_publish: never block a real-time loop. A blocking publish() stalls
        // forever once the 64-slot request queue fills (no broker => ~64 s).
        let _ = self.client.try_publish(
            format!("{}/{topic_suffix}", self.base),
            QoS::AtMostOnce,
            retain,
            payload.to_owned(),
        );
    }
}

fn main() {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let args = match Args::parse(&argv) {
        Ok(Some(a)) => a,
        Ok(None) => return,
        Err(e) => {
            eprintln!("modem: {e}");
            exit(2);
        }
    };
    let base = args.mqtt_topic.clone();
    let loopback_on = args.loopback != "none";

    println!(
        "modem: {} at {} sym/s, {} bps on the air / {} bps data (K=7 r=1/2 FEC), occupied BW {:.0} Hz",
        args.modulation.name(),
        args.symbol_rate,
        bit_rate_bps(args.modulation, args.symbol_rate),
        info_bit_rate_bps(args.modulation, args.symbol_rate),
        occupied_bandwidth_hz(args.symbol_rate)
    );
    if args.tx_shift.abs().min(args.rx_shift.abs()) < occupied_bandwidth_hz(args.symbol_rate) / 2.0
    {
        eprintln!("warning: a freq shift is inside the signal's occupied band (+-{:.0} Hz) - LO leakage will land on the signal; raise --tx-shift/--rx-shift", occupied_bandwidth_hz(args.symbol_rate) / 2.0);
    }
    if args.tx_enabled {
        println!("  TX: {} Hz, gain {} dB (device TX1 range [-23.75, 66] dB), shift {} Hz, source {:?}, BERT {:?}", args.tx_freq, args.tx_gain, args.tx_shift, args.source, args.pattern);
    } else {
        println!("  TX: disabled");
    }
    if args.rx_enabled {
        println!(
            "  RX: {} Hz, gain {} dB, shift {} Hz, carrier loop {} Hz, DC cutoff {} Hz",
            args.rx_freq, args.rx_gain, args.rx_shift, args.carrier_bw, args.dc_cutoff
        );
    } else {
        println!("  RX: disabled");
    }
    println!(
        "  KISS port {}; MQTT {}:{} topic '{base}'; loopback {}",
        args.kiss_port, args.mqtt_host, args.mqtt_port, args.loopback
    );
    if args.tx_enabled && !loopback_on {
        println!("This transmits real RF. Confirm you're authorized to transmit at this frequency");
        println!("and that your RF path (attenuator/antenna) is set up safely.");
    }
    println!();

    // ---- radio ----
    let dev = Arc::new(BladeRfAny::open_first().unwrap_or_else(|e| {
        fail(format!(
            "cannot open the bladeRF (is it plugged in? is another program using it?): {e}"
        ))
    }));
    if !dev.is_fpga_configured().unwrap_or(false) {
        eprintln!("FPGA is not loaded. Run, e.g.:");
        eprintln!("  bladeRF-cli -e \"load fpga /usr/share/bladerf/fpga/hostedxA4.rbf\"");
        exit(1);
    }
    let loopback_mode = match args.loopback.as_str() {
        "rfic-bist" => Loopback::RficBist,
        _ => Loopback::None,
    };
    // SAFETY: only valid Loopback variants are constructed above.
    unsafe { dev.set_loopback(loopback_mode) }
        .unwrap_or_else(|e| fail(format!("set loopback mode: {e}")));

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

    let hardware_rate = hardware_sample_rate_for(args.symbol_rate);
    let analog_bandwidth = analog_bandwidth_hz(args.symbol_rate);
    println!("SDR sample rate {hardware_rate} Hz, analog bandwidth {analog_bandwidth} Hz");
    let mut tx_radio = None;
    let mut rx_radio = None;
    if args.tx_enabled {
        dev.set_frequency(tx_ch, args.tx_freq)
            .unwrap_or_else(|e| fail(format!("set TX frequency: {e}")));
        dev.set_sample_rate(tx_ch, hardware_rate)
            .unwrap_or_else(|e| fail(format!("set TX sample rate: {e}")));
        dev.set_bandwidth(tx_ch, analog_bandwidth)
            .unwrap_or_else(|e| fail(format!("set TX bandwidth: {e}")));
        dev.set_gain(tx_ch, args.tx_gain)
            .unwrap_or_else(|e| fail(format!("set TX gain: {e}")));
    }
    if args.rx_enabled {
        dev.set_frequency(rx_ch, args.rx_freq)
            .unwrap_or_else(|e| fail(format!("set RX frequency: {e}")));
        dev.set_sample_rate(rx_ch, hardware_rate)
            .unwrap_or_else(|e| fail(format!("set RX sample rate: {e}")));
        dev.set_bandwidth(rx_ch, analog_bandwidth)
            .unwrap_or_else(|e| fail(format!("set RX bandwidth: {e}")));
        // Manual, not AGC: a fixed known gain gives a repeatable level.
        dev.set_gain_mode(rx_ch, GainMode::Manual)
            .unwrap_or_else(|e| fail(format!("set RX gain mode: {e}")));
        dev.set_gain(rx_ch, args.rx_gain)
            .unwrap_or_else(|e| fail(format!("set RX gain: {e}")));
    }
    let stream_config = stream_config_for(hardware_rate);
    if args.tx_enabled {
        let stream = BladeRfAny::tx_streamer_arc::<ComplexI16>(
            dev.clone(),
            stream_config,
            ChannelLayoutTx::SISO(tx_channel),
        )
        .unwrap_or_else(|e| fail(format!("create TX streamer: {e}")));
        stream
            .enable()
            .unwrap_or_else(|e| fail(format!("enable TX1: {e}")));
        tx_radio = Some(BladeTx {
            dev: dev.clone(),
            channel: tx_ch,
            layout: tx_channel,
            stream: Some(stream),
        });
    }
    if args.rx_enabled {
        let stream = BladeRfAny::rx_streamer_arc::<ComplexI16>(
            dev.clone(),
            stream_config,
            ChannelLayoutRx::SISO(rx_channel),
        )
        .unwrap_or_else(|e| fail(format!("create RX streamer: {e}")));
        stream
            .enable()
            .unwrap_or_else(|e| fail(format!("enable RX: {e}")));
        rx_radio = Some(BladeRx {
            dev: dev.clone(),
            channel: rx_ch,
            layout: rx_channel,
            stream: Some(stream),
        });
    }

    // ---- KISS: one socket, both directions ----
    let kiss = Arc::new(
        KissServer::start(&format!("0.0.0.0:{}", args.kiss_port))
            .unwrap_or_else(|e| fail(format!("bind KISS port {}: {e}", args.kiss_port))),
    );
    println!("KISS TCP server listening on port {}.", args.kiss_port);

    // ---- MQTT ----
    let mut mqtt_options = MqttOptions::new(
        format!("{base}_modem"),
        args.mqtt_host.clone(),
        args.mqtt_port,
    );
    mqtt_options.set_keep_alive(Duration::from_secs(5));
    mqtt_options.set_last_will(LastWill::new(
        format!("{base}/modem/status"),
        r#"{"state":"offline"}"#,
        QoS::AtLeastOnce,
        true,
    ));
    let (mqtt_client, mut mqtt_connection) = Client::new(mqtt_options, 64);
    let tx_control_topic = format!("{base}/tx/control");
    let rx_control_topic = format!("{base}/rx/control");
    let modem_control_topic = format!("{base}/modem/control");
    for t in [&tx_control_topic, &rx_control_topic, &modem_control_topic] {
        mqtt_client
            .subscribe(t, QoS::AtLeastOnce)
            .unwrap_or_else(|e| fail(format!("subscribe {t}: {e}")));
    }
    let telemetry = Arc::new(MqttTelemetry {
        client: mqtt_client.clone(),
        base: base.clone(),
    });

    let shutdown = Arc::new(AtomicBool::new(false));
    let (tx_ctl, tx_ctl_rx) = bounded::<TxControlMsg>(16);
    let (rx_ctl, rx_ctl_rx) = bounded::<RxControlMsg>(16);
    {
        let (tx_ctl, rx_ctl, shutdown) = (tx_ctl.clone(), rx_ctl.clone(), shutdown.clone());
        thread::spawn(move || {
            for notification in mqtt_connection.iter() {
                match notification {
                    Ok(Event::Incoming(Packet::Publish(p))) => {
                        let payload = String::from_utf8_lossy(&p.payload).to_string();
                        if p.topic == tx_control_topic {
                            match serde_json::from_str::<TxControlMsg>(&payload) {
                                Ok(cmd) => {
                                    println!("[mqtt] {payload} -> {cmd:?}");
                                    let _ = tx_ctl.try_send(cmd);
                                }
                                Err(e) => eprintln!("[mqtt] bad TX command '{payload}': {e}"),
                            }
                        } else if p.topic == rx_control_topic {
                            match serde_json::from_str::<RxControlMsg>(&payload) {
                                Ok(cmd) => {
                                    println!("[mqtt] {payload} -> {cmd:?}");
                                    let _ = rx_ctl.try_send(cmd);
                                }
                                Err(e) => eprintln!("[mqtt] bad RX command '{payload}': {e}"),
                            }
                        } else if p.topic == modem_control_topic {
                            match serde_json::from_str::<ModemControlMsg>(&payload) {
                                Ok(ModemControlMsg::Shutdown) => {
                                    println!("[mqtt] shutdown requested");
                                    shutdown.store(true, Ordering::Relaxed);
                                }
                                Err(e) => eprintln!("[mqtt] bad modem command '{payload}': {e}"),
                            }
                        }
                    }
                    Ok(_) => {}
                    Err(e) => {
                        eprintln!("[mqtt] connection error: {e}");
                        // rumqttc retries immediately; without a pause a missing
                        // broker turns this into a busy loop flooding stderr.
                        thread::sleep(Duration::from_secs(2));
                    }
                }
            }
        });
    }

    // Typing `bert` / `data` on stdin switches the TX source.
    {
        let tx_ctl: Sender<TxControlMsg> = tx_ctl.clone();
        thread::spawn(move || {
            for line in std::io::stdin().lock().lines() {
                let Ok(line) = line else { break };
                match line.trim().to_lowercase().as_str() {
                    "bert" => {
                        let _ = tx_ctl.try_send(TxControlMsg::SetSource(TxSource::Bert));
                    }
                    "data" | "ax25" => {
                        let _ = tx_ctl.try_send(TxControlMsg::SetSource(TxSource::Data));
                    }
                    "" => {}
                    other => println!("[stdin] unknown command '{other}' (try: bert, data)"),
                }
            }
        });
    }

    // ---- engines: one supervisor thread runs TX and RX and restarts both when a
    // symbol-rate change needs a different SDR sample rate ----
    let mut tx_engine = tx_radio.as_ref().map(|_| {
        TxEngine::new(
            TxConfig {
                modulation: args.modulation,
                symbol_rate_hz: args.symbol_rate,
                frequency_hz: args.tx_freq,
                gain_db: args.tx_gain,
                freq_shift_hz: args.tx_shift,
                pattern: args.pattern,
                source: args.source,
            },
            kiss.clone(),
        )
    });
    let mut rx_engine = rx_radio.as_ref().map(|_| {
        RxEngine::new(
            RxConfig {
                modulation: args.modulation,
                symbol_rate_hz: args.symbol_rate,
                frequency_hz: args.rx_freq,
                gain_db: args.rx_gain,
                freq_shift_hz: args.rx_shift,
                pattern: args.pattern,
                carrier_bandwidth_hz: args.carrier_bw,
                dc_cutoff_hz: args.dc_cutoff,
            },
            kiss.clone(),
        )
    });
    let supervisor = {
        let (telemetry, shutdown) = (telemetry.clone(), shutdown.clone());
        thread::spawn(move || {
            let result = run_engines(
                tx_engine
                    .as_mut()
                    .zip(tx_radio.as_mut())
                    .map(|(e, r)| (e, r, &tx_ctl_rx)),
                rx_engine
                    .as_mut()
                    .zip(rx_radio.as_mut())
                    .map(|(e, r)| (e, r, &rx_ctl_rx)),
                &*telemetry,
                &shutdown,
                STATUS_INTERVAL,
            );
            println!("[modem] engines stopped; disabling streams");
            if let Some(stream) = tx_radio.as_ref().and_then(|r| r.stream.as_ref()) {
                let _ = stream.disable();
            }
            if let Some(stream) = rx_radio.as_ref().and_then(|r| r.stream.as_ref()) {
                let _ = stream.disable();
            }
            println!("[modem] streams disabled");
            result
        })
    };

    println!("Running. Ctrl+C, or MQTT {base}/modem/control \"Shutdown\", to stop.");
    println!();

    // ---- main thread: process status + watch for shutdown / failure ----
    let start = Instant::now();
    let mut last_status = Instant::now() - STATUS_INTERVAL;
    let mut exit_code = 0;
    while !shutdown.load(Ordering::Relaxed) {
        if last_status.elapsed() >= STATUS_INTERVAL {
            let msg = ModemStatusMsg {
                state: "online".into(),
                pid: std::process::id(),
                uptime_s: start.elapsed().as_secs_f64(),
                tx_enabled: args.tx_enabled,
                rx_enabled: args.rx_enabled,
                loopback: args.loopback.clone(),
                kiss_port: args.kiss_port,
            };
            if let Ok(json) = serde_json::to_string(&msg) {
                telemetry.publish("modem/status", &json, true);
            }
            last_status = Instant::now();
        }
        if supervisor.is_finished() {
            // The engines ended on their own, which only happens on a radio failure.
            shutdown.store(true, Ordering::Relaxed);
            break;
        }
        thread::sleep(Duration::from_millis(100));
    }
    println!("Shutting down ...");
    // Wait for the engines, but not forever: a radio call stuck in the driver
    // (e.g. a stream that won't drain) must not keep the process alive after
    // it was told to stop. Process exit releases the device either way.
    let deadline = Instant::now() + Duration::from_secs(6);
    while Instant::now() < deadline && !supervisor.is_finished() {
        thread::sleep(Duration::from_millis(50));
    }
    if !supervisor.is_finished() {
        eprintln!(
            "modem: the engines did not stop within 6 s (stuck in a radio call); exiting anyway"
        );
        exit_code = 1;
    } else {
        match supervisor.join() {
            Ok(Ok(())) => {}
            Ok(Err(e)) => {
                eprintln!("modem: engine stopped: {e}");
                exit_code = 1;
            }
            Err(_) => {
                eprintln!("modem: engine supervisor panicked");
                exit_code = 1;
            }
        }
    }
    telemetry.publish("modem/status", r#"{"state":"offline"}"#, true);
    thread::sleep(Duration::from_millis(400)); // let the MQTT thread flush the goodbye
    exit(exit_code);
}
