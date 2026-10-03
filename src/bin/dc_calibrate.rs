//! Estimates a TX-side (I, Q) pre-compensation bias to cancel carrier/LO
//! leakage, using a simple two-point complex-linear calibration:
//!
//!   1. Capture with no TX bias -> measure the baseline DC offset dc0.
//!   2. Capture with a known trial bias -> measure dc1.
//!   3. The difference reveals the complex gain G from injected bias to
//!      observed DC (dc1 - dc0 = G * trial_bias), so the bias that should
//!      null the baseline is -dc0 / G.
//!   4. Capture again with that computed bias to verify.
//!
//! This can only cancel the TX-attributable portion of the observed DC
//! offset - RX-side ADC/mixer DC offset is independent of what's
//! transmitted and needs correcting separately (see project memory).
//! Valid because TX and RX share the same on-board reference clock in this
//! single-device test, so the DC term is genuine DC, not a slowly rotating
//! beat from independent LOs.
//!
//! Usage: dc_calibrate [frequency_hz] [duration_s] [symbol_rate_hz] [tx_gain_db] [rx_gain_db] [pattern] [trial_bias]
//! Defaults: 145800000 0.2 16000 0 50 pn15 0.1

use bladerf::{Loopback, RxChannel, TxChannel};
use num_complex::Complex32;

use ham_modem::capture::{self, CaptureConfig};
use ham_modem::psd::complex_mean;
use ham_modem::rrc::ShapeMode;

fn run(cfg: &CaptureConfig) -> Complex32 {
    let result = capture::run_capture(cfg);
    capture::report_level(result.peak_magnitude);
    complex_mean(&result.samples)
}

fn db_full_scale(c: Complex32) -> f32 {
    20.0 * c.norm().max(1e-9).log10()
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let frequency_hz: u64 = args
        .get(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(145_800_000);
    let duration_s: f64 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(0.2);
    let symbol_rate_hz: f64 = args
        .get(3)
        .and_then(|s| s.parse().ok())
        .unwrap_or(ham_modem::params::SYMBOL_RATE_HZ);
    let tx_gain_db: i32 = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(0);
    let rx_gain_db: i32 = args.get(5).and_then(|s| s.parse().ok()).unwrap_or(50);
    let pattern = args
        .get(6)
        .map(|s| capture::parse_pattern(s))
        .unwrap_or(ham_modem::prbs::PrbsPattern::Pn15);
    let trial_bias: f32 = args.get(7).and_then(|s| s.parse().ok()).unwrap_or(0.1);

    println!("TX DC pre-compensation calibration: TX1 -> attenuator -> RX2");
    println!("  Frequency:  {frequency_hz} Hz");
    println!("  TX gain:    {tx_gain_db} dB, RX gain: {rx_gain_db} dB");
    println!("  Trial bias: {trial_bias}");
    println!();

    let base_cfg = |tx_dc_bias: (f32, f32)| CaptureConfig {
        duration_s,
        symbol_rate_hz,
        pattern,
        shape_mode: ShapeMode::Rrc,
        digital_scale: 1.0,
        tx_channel: TxChannel::Tx0,
        rx_channel: RxChannel::Rx1,
        frequency_hz: Some(frequency_hz),
        tx_gain_db,
        rx_gain_db,
        loopback: Loopback::None,
        tx_dc_bias,
        tx_freq_shift_hz: 0.0,
    };

    println!("--- Baseline capture (no compensation) ---");
    let dc0 = run(&base_cfg((0.0, 0.0)));
    println!(
        "  DC offset: I={:.4} Q={:.4} ({:.1} dBFS)",
        dc0.re,
        dc0.im,
        db_full_scale(dc0)
    );
    println!();

    println!("--- Trial capture (bias = {trial_bias} + 0j) ---");
    let dc1 = run(&base_cfg((trial_bias, 0.0)));
    println!(
        "  DC offset: I={:.4} Q={:.4} ({:.1} dBFS)",
        dc1.re,
        dc1.im,
        db_full_scale(dc1)
    );
    println!();

    let gain = (dc1 - dc0) / Complex32::new(trial_bias, 0.0);
    if gain.norm() < 1e-6 {
        eprintln!("Trial bias produced no measurable change ({gain:?}) - can't solve for a correcting bias.");
        eprintln!("Try a larger --trial-bias, or check the TX/RX path is actually connected.");
        std::process::exit(1);
    }
    let correcting_bias = -dc0 / gain;
    println!(
        "Estimated coupling gain: {:.4}{:+.4}j  ->  correcting bias: I={:.4} Q={:.4}",
        gain.re, gain.im, correcting_bias.re, correcting_bias.im
    );
    println!();

    println!("--- Verification capture (computed correcting bias applied) ---");
    let dc_final = run(&base_cfg((correcting_bias.re, correcting_bias.im)));
    println!(
        "  DC offset: I={:.4} Q={:.4} ({:.1} dBFS)",
        dc_final.re,
        dc_final.im,
        db_full_scale(dc_final)
    );
    println!();

    let improvement_db = db_full_scale(dc0) - db_full_scale(dc_final);
    println!(
        "Result: baseline {:.1} dBFS -> corrected {:.1} dBFS ({:+.1} dB change)",
        db_full_scale(dc0),
        db_full_scale(dc_final),
        improvement_db
    );
    println!("Remember: this can only cancel the TX-attributable share of the offset.");
    println!("If a substantial DC term remains, that's likely the RX side (see project memory: RX1 already had large factory correction values).");
}
