//! Captures TX output via the REAL RF path: physical TX1, through an
//! external cable and attenuator, into physical RX2. Unlike
//! loopback_capture (internal digital self-test), this exercises the actual
//! analog TX and RX chains, real DACs, mixers, PAs, LNAs, and ADCs, not just
//! the software DSP chain.
//!
//! REQUIRES: TX1 physically connected to RX2 through an attenuator with
//! enough pad to not overdrive RX2's input. This transmits real RF power
//! onto that cable (not into free space, but still real analog TX output) -
//! confirm your cabling/attenuator before running this.
//!
//! Usage: rf_loopback_capture [frequency_hz] [duration_s] [symbol_rate_hz] [tx_gain_db] [rx_gain_db] [pattern] [iq_out] [psd_out] [shape] [tx_dc_bias_i] [tx_dc_bias_q] [freq_shift_hz]
//! Defaults: 145800000 0.5 16000 0 20 pn15 capture.iq32 capture_psd.csv rrc 0.0 0.0 0.0
//! shape: rrc | rect (rectangular/zero-order-hold, no filtering - produces
//! the classic sinc-shaped spectrum RRC shaping exists to suppress)
//! tx_dc_bias_i/q: TX-side DC pre-compensation, see dc_calibrate
//! freq_shift_hz: offset-tune the TX signal away from the LO frequency in
//! DSP, so carrier/LO leakage (which stays at the LO) lands outside the
//! signal's occupied bandwidth instead of on top of it

use bladerf::{Loopback, RxChannel, TxChannel};

use ham_modem::capture::{self, CaptureConfig};
use ham_modem::rrc::ShapeMode;

fn parse_shape_mode(s: &str) -> ShapeMode {
    match s.to_lowercase().as_str() {
        "rect" | "rectangular" | "none" => ShapeMode::Rectangular,
        "rrc" => ShapeMode::Rrc,
        other => {
            eprintln!("Unknown shape mode '{other}', defaulting to rrc");
            ShapeMode::Rrc
        }
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let frequency_hz: u64 = args
        .get(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(145_800_000);
    let duration_s: f64 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(0.5);
    let symbol_rate_hz: f64 = args
        .get(3)
        .and_then(|s| s.parse().ok())
        .unwrap_or(ham_modem::params::SYMBOL_RATE_HZ);
    let tx_gain_db: i32 = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(0);
    let rx_gain_db: i32 = args.get(5).and_then(|s| s.parse().ok()).unwrap_or(20);
    let pattern = args
        .get(6)
        .map(|s| capture::parse_pattern(s))
        .unwrap_or(ham_modem::prbs::PrbsPattern::Pn15);
    let iq_out_path = args
        .get(7)
        .cloned()
        .unwrap_or_else(|| "capture.iq32".to_string());
    let psd_out_path = args
        .get(8)
        .cloned()
        .unwrap_or_else(|| "capture_psd.csv".to_string());
    let shape_mode = args
        .get(9)
        .map(|s| parse_shape_mode(s))
        .unwrap_or(ShapeMode::Rrc);
    let tx_dc_bias_i: f32 = args.get(10).and_then(|s| s.parse().ok()).unwrap_or(0.0);
    let tx_dc_bias_q: f32 = args.get(11).and_then(|s| s.parse().ok()).unwrap_or(0.0);
    let freq_shift_hz: f64 = args.get(12).and_then(|s| s.parse().ok()).unwrap_or(0.0);

    println!(
        "bladeRF REAL RF PATH capture: TX1 -> attenuator -> RX2 (cable, not internal loopback)"
    );
    println!("  Frequency:  {frequency_hz} Hz");
    println!("  Duration:   {duration_s} s");
    println!("  Symbol rate: {symbol_rate_hz} sym/s");
    println!("  TX gain:    {tx_gain_db} dB (TX1 range [-23.75, 66] dB)");
    println!("  RX gain:    {rx_gain_db} dB (RX2 range likely similar to RX1's [-15, 60] dB - not independently verified)");
    println!("  Pattern:    {pattern:?}");
    println!("  Shape mode: {shape_mode:?}");
    println!("  Freq shift: {freq_shift_hz} Hz");
    println!();
    println!("This transmits real RF power onto the TX1 cable. Confirm TX1 is");
    println!("connected to RX2 through a sufficient attenuator before proceeding.");
    println!();

    // Gotchas confirmed against this hardware (see loopback_capture.rs for
    // the loopback-mode one, not relevant here since loopback is None):
    // this crate's Channel/TxChannel/RxChannel enums are 0-indexed (Tx0,
    // Rx0, Rx1, ...) matching libbladeRF's internal macros, while Nuand's
    // silkscreen/bladeRF-cli call the ports "TX1"/"RX1"/"RX2" (1-indexed).
    // Physical TX1 = TxChannel::Tx0, physical RX2 = RxChannel::Rx1.
    let cfg = CaptureConfig {
        duration_s,
        symbol_rate_hz,
        pattern,
        shape_mode,
        digital_scale: 1.0, // real analog gain controls do the level-setting here, unlike RFIC_BIST
        tx_channel: TxChannel::Tx0, // physical TX1
        rx_channel: RxChannel::Rx1, // physical RX2
        frequency_hz: Some(frequency_hz),
        tx_gain_db,
        rx_gain_db,
        loopback: Loopback::None,
        tx_dc_bias: (tx_dc_bias_i, tx_dc_bias_q),
        tx_freq_shift_hz: freq_shift_hz,
    };

    let result = capture::run_capture(&cfg);
    capture::report_level(result.peak_magnitude);
    capture::save_iq_and_psd(&result.samples, &iq_out_path, &psd_out_path);
}
