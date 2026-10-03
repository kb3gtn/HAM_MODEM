//! Captures TX output via bladeRF's internal RFIC BIST loopback (a digital
//! self-test path entirely inside the AD9361 - no RF mixer/PA/LNA/antenna
//! path, no external cabling) and writes raw IQ plus a PSD estimate to disk.
//! A software stand-in for a spectrum analyzer, to get some confidence in
//! the TX DSP chain when a real analyzer isn't available.
//!
//! Usage: loopback_capture [duration_s] [symbol_rate_hz] [pattern] [iq_out] [psd_out] [digital_scale]
//! Defaults: 0.5 16000 pn15 capture.iq32 capture_psd.csv 0.05

use bladerf::{Loopback, RxChannel, TxChannel};

use ham_modem::capture::{self, CaptureConfig};
use ham_modem::rrc::ShapeMode;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let duration_s: f64 = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(0.5);
    let symbol_rate_hz: f64 = args
        .get(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(ham_modem::params::SYMBOL_RATE_HZ);
    let pattern = args
        .get(3)
        .map(|s| capture::parse_pattern(s))
        .unwrap_or(ham_modem::prbs::PrbsPattern::Pn15);
    let iq_out_path = args
        .get(4)
        .cloned()
        .unwrap_or_else(|| "capture.iq32".to_string());
    let psd_out_path = args
        .get(5)
        .cloned()
        .unwrap_or_else(|| "capture_psd.csv".to_string());
    // Extra digital backoff on top of the RRC's own calibrated headroom.
    // RFIC_BIST appears to add its own gain ahead of the RX ADC that isn't
    // reachable through the normal TX/RX VGA gain controls (changing those
    // had no effect on the observed clipping) - back off in software instead.
    let digital_scale: f32 = args.get(6).and_then(|s| s.parse().ok()).unwrap_or(0.05);

    println!("bladeRF RFIC BIST loopback capture (digital self-test loopback - no RF/antenna path involved)");
    println!("  Duration:      {duration_s} s");
    println!("  Symbol rate:   {symbol_rate_hz} sym/s (default = the modem's fixed symbol rate)");
    println!("  Pattern:       {pattern:?}");
    println!("  Digital scale: {digital_scale} (extra backoff - RFIC_BIST's own gain isn't controlled by TX/RX VGA settings)");
    println!();

    // IMPORTANT gotchas confirmed against this hardware:
    // - This crate's Channel/TxChannel/RxChannel enums are 0-indexed (Tx0,
    //   Rx0, ...), matching libbladeRF's internal macros, while Nuand's
    //   silkscreen/bladeRF-cli call the ports "TX1"/"RX1" etc (1-indexed).
    //   Physical TX1 = TxChannel::Tx0, physical RX1 = RxChannel::Rx0.
    // - Loopback::BbTxlpfRxlpf and the other baseband/RF-LNA modes are
    //   LMS6002D (bladeRF1) names, rejected by this AD9361-based bladeRF 2.0.
    //   Only None/Firmware/RficBist are valid here - confirmed empirically.
    let cfg = CaptureConfig {
        duration_s,
        symbol_rate_hz,
        pattern,
        shape_mode: ShapeMode::Rrc,
        digital_scale,
        tx_dc_bias: (0.0, 0.0),
        tx_freq_shift_hz: 0.0,
        tx_channel: TxChannel::Tx0, // physical TX1
        rx_channel: RxChannel::Rx0, // physical RX1 - paired with Tx0 by RFIC_BIST
        frequency_hz: None,         // irrelevant: RFIC_BIST bypasses the mixer entirely
        tx_gain_db: 0,
        rx_gain_db: -15,
        loopback: Loopback::RficBist,
    };

    let result = capture::run_capture(&cfg);
    capture::report_level(result.peak_magnitude);
    capture::save_iq_and_psd(&result.samples, &iq_out_path, &psd_out_path);
}
