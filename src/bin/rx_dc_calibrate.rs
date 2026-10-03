//! Estimates RX-side DC offset correction register values (RX1's own
//! `CorrectionDcOffsetI`/`CorrectionDcOffsetQ`) to cancel the residual
//! carrier/LO leakage that TX-side sample pre-compensation (dc_calibrate)
//! couldn't reach - see project memory for why that pointed at the RX side.
//!
//! Unlike dc_calibrate (which only needed a linear complex gain, since it
//! adjusts our own already-orthogonal digital I/Q samples), this uses the
//! full 2x2 real Jacobian: hardware correction DACs for I and Q may not be
//! perfectly orthogonal in their effect on the observed complex mean, so
//! each is perturbed independently to measure real cross-coupling.
//!
//! IMPORTANT: this keeps ONE device session open across all captures
//! (`OpenCapture`), because device state like this appears to reset on a
//! fresh device open (matching what was observed with gain/AGC settings
//! resetting between separate bladeRF-cli invocations) - `set_correction`
//! calls between captures would otherwise have no lasting effect.
//!
//! Usage: rx_dc_calibrate [frequency_hz] [duration_s] [symbol_rate_hz] [tx_gain_db] [rx_gain_db] [pattern] [trial_delta]
//! Defaults: 145800000 0.2 16000 0 50 pn15 200

use bladerf::{
    BladeRF, Channel, CorrectionDcOffsetI, CorrectionDcOffsetQ, CorrectionValue, Loopback,
    RxChannel, TxChannel,
};
use num_complex::Complex32;

use ham_modem::capture::{CaptureConfig, OpenCapture};
use ham_modem::psd::complex_mean;
use ham_modem::rrc::ShapeMode;

fn get_rx_correction(dev: &impl BladeRF, ch: Channel) -> (i16, i16) {
    let i: CorrectionDcOffsetI = dev.get_correction(ch).expect("get I correction");
    let q: CorrectionDcOffsetQ = dev.get_correction(ch).expect("get Q correction");
    (i.value(), q.value())
}

fn set_rx_correction(dev: &impl BladeRF, ch: Channel, i: i16, q: i16) {
    dev.set_correction(ch, CorrectionDcOffsetI::new_saturating(i))
        .expect("set I correction");
    dev.set_correction(ch, CorrectionDcOffsetQ::new_saturating(q))
        .expect("set Q correction");
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
        .map(|s| ham_modem::capture::parse_pattern(s))
        .unwrap_or(ham_modem::prbs::PrbsPattern::Pn15);
    let trial_delta: i16 = args.get(7).and_then(|s| s.parse().ok()).unwrap_or(200);

    println!("RX DC offset correction calibration: TX1 -> attenuator -> RX2");
    println!("  Frequency:   {frequency_hz} Hz");
    println!("  TX gain:     {tx_gain_db} dB, RX gain: {rx_gain_db} dB");
    println!("  Trial delta: {trial_delta} correction units (range is [-2048, 2048])");
    println!();

    let cfg = CaptureConfig {
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
        tx_dc_bias: (0.0, 0.0),
        tx_freq_shift_hz: 0.0,
    };

    let open = OpenCapture::open(&cfg);
    let rx_ch = open.rx_channel();

    let (base_i, base_q) = get_rx_correction(open.dev().as_ref(), rx_ch);
    println!("Current RX1 correction (from prior auto-cal): I={base_i} Q={base_q}");
    println!();

    let capture_mean = |label: &str| -> Complex32 {
        println!("--- {label} ---");
        let result = open.capture(
            duration_s,
            symbol_rate_hz,
            pattern,
            ShapeMode::Rrc,
            1.0,
            (0.0, 0.0),
            0.0,
        );
        ham_modem::capture::report_level(result.peak_magnitude);
        let mean = complex_mean(&result.samples);
        println!(
            "  DC offset: I={:.4} Q={:.4} ({:.1} dBFS)",
            mean.re,
            mean.im,
            db_full_scale(mean)
        );
        println!();
        mean
    };

    set_rx_correction(open.dev().as_ref(), rx_ch, base_i, base_q);
    let dc0 = capture_mean("Baseline (current RX1 correction)");

    set_rx_correction(open.dev().as_ref(), rx_ch, base_i + trial_delta, base_q);
    let dc_i = capture_mean(&format!(
        "Trial: I correction {} (+{trial_delta})",
        base_i + trial_delta
    ));

    set_rx_correction(open.dev().as_ref(), rx_ch, base_i, base_q + trial_delta);
    let dc_q = capture_mean(&format!(
        "Trial: Q correction {} (+{trial_delta})",
        base_q + trial_delta
    ));

    // Jacobian columns: how the observed complex mean moves per unit change
    // in each correction register, measured independently since hardware
    // correction DACs aren't guaranteed perfectly orthogonal.
    let d_per_i = (dc_i - dc0) / trial_delta as f32;
    let d_per_q = (dc_q - dc0) / trial_delta as f32;
    println!(
        "Sensitivity: d(mean)/d(I_corr) = {:.6}{:+.6}j, d(mean)/d(Q_corr) = {:.6}{:+.6}j",
        d_per_i.re, d_per_i.im, d_per_q.re, d_per_q.im
    );

    let (a, b, c, d) = (d_per_i.re, d_per_i.im, d_per_q.re, d_per_q.im);
    let det = a * d - b * c;
    if det.abs() < 1e-9 {
        eprintln!(
            "Sensitivity matrix is singular (det={det:.2e}) - can't solve for a correcting delta."
        );
        eprintln!("Try a larger --trial-delta, or check the TX/RX path is actually connected.");
        set_rx_correction(open.dev().as_ref(), rx_ch, base_i, base_q); // restore baseline
        std::process::exit(1);
    }
    let (e, f) = (dc0.re, dc0.im);
    let delta_i = (-e * d + f * c) / det;
    let delta_q = (b * e - a * f) / det;

    let final_i = (base_i as f32 + delta_i).round().clamp(-2048.0, 2048.0) as i16;
    let final_q = (base_q as f32 + delta_q).round().clamp(-2048.0, 2048.0) as i16;
    println!("Computed correcting delta: I={delta_i:.1} Q={delta_q:.1} -> final correction I={final_i} Q={final_q}");
    println!();

    set_rx_correction(open.dev().as_ref(), rx_ch, final_i, final_q);
    let dc_final = capture_mean("Verification (computed RX1 correction applied)");

    println!(
        "Result: baseline {:.1} dBFS -> corrected {:.1} dBFS ({:+.1} dB change)",
        db_full_scale(dc0),
        db_full_scale(dc_final),
        db_full_scale(dc0) - db_full_scale(dc_final)
    );
    println!("Final RX1 correction values (I={final_i}, Q={final_q}) are only applied for this");
    println!("device session - they are not known to persist across a power cycle/reopen.");
}
