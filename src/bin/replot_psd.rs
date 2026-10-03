//! Recomputes a PSD estimate from a previously saved raw IQ capture, at
//! whatever FFT size/overlap you want - no need to re-run hardware capture
//! just to look at a capture with different resolution. Optionally applies
//! the DC-blocking filter first, to validate it against real captures
//! without a new hardware run.
//!
//! Usage: replot_psd <iq_file> [sample_rate_hz] [fft_size] [overlap] [psd_out] [dc_block_cutoff_hz]
//! Defaults: - 800000 8192 0.5 replot_psd.csv (no DC blocking)
//!
//! The IQ file format is raw i16 I,Q pairs, interleaved, little-endian -
//! the same format `capture::save_iq_and_psd` writes.

use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};

use num_complex::Complex32;

use ham_modem::dc_block::DcBlocker;
use ham_modem::psd::welch_psd;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let iq_path = args.get(1).expect("usage: replot_psd <iq_file> [sample_rate_hz] [fft_size] [overlap] [psd_out] [dc_block_cutoff_hz]");
    let sample_rate_hz: f64 = args
        .get(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(800_000.0);
    let fft_size: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(8192);
    let overlap: f64 = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(0.5);
    let psd_out_path = args
        .get(5)
        .cloned()
        .unwrap_or_else(|| "replot_psd.csv".to_string());
    let dc_block_cutoff_hz: Option<f64> = args.get(6).and_then(|s| s.parse().ok());

    let mut bytes = Vec::new();
    BufReader::new(File::open(iq_path).expect("open IQ file"))
        .read_to_end(&mut bytes)
        .expect("read IQ file");
    assert!(
        bytes.len() % 4 == 0,
        "IQ file length isn't a multiple of 4 bytes (i16 I + i16 Q)"
    );

    let mut iq: Vec<Complex32> = bytes
        .chunks_exact(4)
        .map(|c| {
            let i = i16::from_le_bytes([c[0], c[1]]) as f32 / 2048.0;
            let q = i16::from_le_bytes([c[2], c[3]]) as f32 / 2048.0;
            Complex32::new(i, q)
        })
        .collect();

    let bin_hz = sample_rate_hz / fft_size as f64;
    println!("Loaded {} samples from {iq_path}", iq.len());
    println!("FFT size {fft_size} -> bin spacing {bin_hz:.2} Hz, overlap {overlap}");

    if let Some(cutoff) = dc_block_cutoff_hz {
        println!("Applying DC blocker, cutoff {cutoff} Hz");
        let mut blocker = DcBlocker::from_cutoff_hz(cutoff, sample_rate_hz);
        iq = blocker.process_block(&iq);
    }

    let psd = welch_psd(&iq, sample_rate_hz, fft_size, overlap);

    let mut f = BufWriter::new(File::create(&psd_out_path).expect("create PSD output file"));
    writeln!(f, "freq_hz,power_db").unwrap();
    for p in &psd {
        writeln!(f, "{},{}", p.freq_hz, p.power_db).unwrap();
    }
    println!("Wrote PSD ({} points) to {psd_out_path}", psd.len());
}
