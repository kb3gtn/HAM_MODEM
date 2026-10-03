//! Shared TX signal generation: BERT -> K=7 r=1/2 convolutional encoder -> BPSK/QPSK/8PSK symbol map (1/2/3 coded bits/symbol) -> RRC pulse shape ->
//! resample -> hardware IQ chunks. Used by both the `modem` binary (real antenna
//! output) and `loopback_capture` (baseband loopback) so the two share
//! exactly the same signal-generation code, not two copies of it.

use std::collections::VecDeque;

use bladerf::ComplexI16;

use crate::conv::ConvEncoder;
use crate::prbs::PrbsPattern;
use crate::resampler::TxResampler;
use crate::rrc::{PulseShaper, ShapeMode};
use crate::symbol_map::Modulation;
use crate::tx_bert::{TxBert, TxBertControl, TxBertStatus};
use crate::Block;
use num_complex::Complex32;

pub struct TxSignalGenerator {
    modulation: Modulation,
    bert: TxBert,
    encoder: ConvEncoder,
    /// Coded bits waiting to be packed three to a symbol (each PRBS bit
    /// yields two coded bits, so the 3-per-symbol packing never lines up).
    coded_bits: VecDeque<u8>,
    pulse_shaper: PulseShaper,
    resampler: TxResampler,
    fifo: VecDeque<f32>,
    shaped_buf: Vec<Complex32>,
    /// Extra linear scale applied just before i16 quantization, on top of
    /// the RRC's own calibrated headroom. 1.0 for normal use (the modem);
    /// loopback_capture uses this to back off further, since some loopback
    /// paths (e.g. RFIC_BIST) appear to add their own gain ahead of the RX
    /// ADC that isn't reachable through the normal TX/RX VGA gain controls.
    output_scale: f32,
    /// Constant (I, Q) added to every output sample, before quantization -
    /// software-side DC pre-compensation for carrier/LO leakage. This can
    /// only null out the TX-attributable portion of any observed DC offset;
    /// RX-side ADC/mixer DC offset is independent of what's transmitted and
    /// needs correcting on the RX side instead (see project memory).
    tx_dc_bias: (f32, f32),
    /// Per-sample NCO phase increment (radians) implementing a constant
    /// frequency shift applied to the hardware-rate output - offset-tunes
    /// the transmitted signal away from the LO frequency itself, so LO/DC
    /// leakage (which stays exactly at the LO frequency, unaffected by this)
    /// lands outside the signal's own occupied bandwidth instead of on top
    /// of it. See project memory for why this is the DSP-side alternative
    /// to the TX/RX hardware correction registers, which showed no further
    /// gains available.
    hardware_sample_rate: f64,
    freq_shift_phase_increment: f64,
    /// Running NCO phase (radians), persists across `next_chunk` calls so
    /// the shift is phase-continuous across chunk boundaries.
    freq_shift_phase: f64,
}

impl TxSignalGenerator {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        pattern: PrbsPattern,
        modulation: Modulation,
        symbol_rate_hz: f64,
        hardware_sample_rate: u32,
        sps: usize,
        rolloff: f64,
        span_symbols: usize,
        shape_mode: ShapeMode,
        output_chunk_frames: usize,
        max_resample_ratio_relative: f64,
        output_scale: f32,
        freq_shift_hz: f64,
    ) -> Self {
        let symbol_domain_rate = symbol_rate_hz * sps as f64;
        let resample_ratio = hardware_sample_rate as f64 / symbol_domain_rate;
        let freq_shift_phase_increment =
            2.0 * std::f64::consts::PI * freq_shift_hz / hardware_sample_rate as f64;
        TxSignalGenerator {
            modulation,
            bert: TxBert::new(pattern),
            encoder: ConvEncoder::new(),
            coded_bits: VecDeque::new(),
            pulse_shaper: PulseShaper::new(sps, rolloff, span_symbols, shape_mode),
            resampler: TxResampler::new(
                resample_ratio,
                output_chunk_frames,
                max_resample_ratio_relative,
            ),
            fifo: VecDeque::new(),
            shaped_buf: Vec::with_capacity(sps),
            output_scale,
            tx_dc_bias: (0.0, 0.0),
            hardware_sample_rate: hardware_sample_rate as f64,
            freq_shift_phase_increment,
            freq_shift_phase: 0.0,
        }
    }

    /// Set the constant (I, Q) pre-compensation bias added to every output
    /// sample before quantization. Units match the normalized (pre-2048x)
    /// sample scale, so e.g. 0.05 is 5% of full scale.
    pub fn set_tx_dc_bias(&mut self, i: f32, q: f32) {
        self.tx_dc_bias = (i, q);
    }

    /// Change the TX offset-tuning frequency shift at runtime. The NCO phase is
    /// kept, so the output stays phase-continuous across the change.
    pub fn set_freq_shift_hz(&mut self, freq_shift_hz: f64) {
        self.freq_shift_phase_increment =
            std::f64::consts::TAU * freq_shift_hz / self.hardware_sample_rate;
    }

    /// Produce the next fixed-size chunk of hardware-rate IQ samples.
    pub fn next_chunk(&mut self) -> Vec<ComplexI16> {
        let needed = self.resampler.input_frames_needed() * 2; // interleaved I,Q
        while self.fifo.len() < needed {
            // PRBS bit -> K=7 r=1/2 encoder -> two coded bits; `bps` coded
            // bits (MSB first) form one symbol.
            let bps = self.modulation.bits_per_symbol();
            while self.coded_bits.len() < bps {
                let pair = self.encoder.encode_bit(self.bert.next_bit());
                self.coded_bits.extend(pair);
            }
            let mut bits = [0u8; 3];
            for b in bits.iter_mut().take(bps) {
                *b = self.coded_bits.pop_front().unwrap();
            }
            let symbol = self.modulation.map(&bits[..bps]);
            self.shaped_buf.clear();
            self.pulse_shaper
                .process_symbol(symbol, &mut self.shaped_buf);
            for &s in &self.shaped_buf {
                self.fifo.push_back(s.re); // I
                self.fifo.push_back(s.im); // Q
            }
        }
        let input: Vec<f32> = self.fifo.drain(..needed).collect();
        let output = self.resampler.process(&input);

        output
            .chunks_exact(2)
            .map(|c| {
                let (i_raw, q_raw) = (c[0], c[1]);
                // Frequency shift: multiply by e^(j*phase), phase-continuous
                // across chunk boundaries. Applied before the DC bias/output
                // scale so those still act on the final transmitted signal.
                let (sin_p, cos_p) = self.freq_shift_phase.sin_cos();
                let i_shifted = i_raw * cos_p as f32 - q_raw * sin_p as f32;
                let q_shifted = i_raw * sin_p as f32 + q_raw * cos_p as f32;
                self.freq_shift_phase += self.freq_shift_phase_increment;
                if self.freq_shift_phase.abs() > std::f64::consts::TAU {
                    self.freq_shift_phase %= std::f64::consts::TAU;
                }

                let i_val = i_shifted * self.output_scale + self.tx_dc_bias.0;
                let q_val = q_shifted * self.output_scale + self.tx_dc_bias.1;
                let i = (i_val * 2048.0).round().clamp(-2048.0, 2047.0) as i16;
                let q = (q_val * 2048.0).round().clamp(-2048.0, 2047.0) as i16;
                ComplexI16::new(i, q)
            })
            .collect()
    }

    pub fn bert_control(&mut self, cmd: TxBertControl) {
        self.bert.handle_control(cmd);
    }

    pub fn bert_status(&self) -> TxBertStatus {
        self.bert.status()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::params::*;
    use crate::psd::{complex_i16_to_f32, welch_psd, PsdPoint};

    fn generate_psd(freq_shift_hz: f64, chunks: usize) -> Vec<PsdPoint> {
        let mut gen = TxSignalGenerator::new(
            PrbsPattern::Pn15,
            Modulation::Psk8,
            SYMBOL_RATE_HZ,
            HARDWARE_SAMPLE_RATE,
            SPS,
            RRC_ROLLOFF,
            RRC_SPAN_SYMBOLS,
            ShapeMode::Rrc,
            4096,
            10.0,
            1.0,
            freq_shift_hz,
        );
        let mut samples = Vec::new();
        for _ in 0..chunks {
            samples.extend(gen.next_chunk());
        }
        let iq = complex_i16_to_f32(&samples);
        welch_psd(&iq, HARDWARE_SAMPLE_RATE as f64, 4096, 0.5)
    }

    fn linear(p: &PsdPoint) -> f64 {
        10f64.powf(p.power_db / 10.0)
    }

    /// Power-weighted mean frequency, and the width of the narrowest
    /// contiguous-about-the-centroid band holding `fraction` of the power.
    fn centroid_and_occupied_bw(psd: &[PsdPoint], fraction: f64) -> (f64, f64) {
        let total: f64 = psd.iter().map(linear).sum();
        let centroid = psd.iter().map(|p| p.freq_hz * linear(p)).sum::<f64>() / total;
        let mut best = f64::MAX;
        // Symmetric-about-centroid growth: the occupied bandwidth that
        // matters for a channel mask is the power-containing span around the
        // carrier.
        let mut half = 0.0;
        while half < 400_000.0 {
            let inside: f64 = psd
                .iter()
                .filter(|p| (p.freq_hz - centroid).abs() <= half)
                .map(linear)
                .sum();
            if inside / total >= fraction {
                best = 2.0 * half;
                break;
            }
            half += 100.0;
        }
        (centroid, best)
    }

    /// With 8PSK there is no carrier line to find a peak at: the spectrum is
    /// a flat-topped RRC shape, so check the power centroid instead - it
    /// should sit at the requested frequency shift, and stay put (no
    /// discontinuity artifacts) across chunk boundaries.
    #[test]
    fn freq_shift_moves_the_spectrum_by_the_requested_amount() {
        let (centroid, _) = centroid_and_occupied_bw(&generate_psd(50_000.0, 60), 0.99);
        assert!(
            (centroid - 50_000.0).abs() < 300.0,
            "centroid at {centroid} Hz, expected near 50 kHz"
        );
    }

    #[test]
    fn zero_freq_shift_leaves_the_signal_centered_at_baseband() {
        let (centroid, _) = centroid_and_occupied_bw(&generate_psd(0.0, 60), 0.99);
        assert!(
            centroid.abs() < 300.0,
            "centroid at {centroid} Hz, expected near 0 Hz"
        );
    }

    /// The headline channel requirement: 99% of the transmitted power must
    /// fit inside 25 kHz, and the strict spectral-mask view - everything
    /// outside +-12.5 kHz of centre at least 30 dB below the in-band
    /// density - must hold too.
    /// cargo test --release --lib bench_tx -- --ignored --nocapture
    #[test]
    #[ignore]
    fn bench_tx_throughput() {
        use std::time::Instant;
        for (m, rate) in [
            (Modulation::Psk8, 16_000.0),
            (Modulation::Psk8, 64_000.0),
            (Modulation::Psk8, 150_000.0),
            (Modulation::Psk8, 500_000.0),
            (Modulation::Qpsk, 500_000.0),
            (Modulation::Bpsk, 1_000_000.0),
        ] {
            let hw = hardware_sample_rate_for(rate);
            let sps = sps_for(rate, hw);
            let mut tx = TxSignalGenerator::new(
                PrbsPattern::Pn15,
                m,
                rate,
                hw,
                sps,
                RRC_ROLLOFF,
                RRC_SPAN_SYMBOLS,
                ShapeMode::Rrc,
                chunk_frames_for(hw),
                10.0,
                1.0,
                15_000.0,
            );
            let chunks = (f64::from(hw) * 0.5 / chunk_frames_for(hw) as f64).ceil() as usize;
            let t = Instant::now();
            for _ in 0..chunks {
                tx.next_chunk();
            }
            let signal_s = chunks as f64 * chunk_frames_for(hw) as f64 / f64::from(hw);
            println!(
                "TX {m:?} @ {rate:>9} sym/s ({hw} SPS, sps {sps}): {:6.2}x real time",
                signal_s / t.elapsed().as_secs_f64()
            );
        }
    }

    #[test]
    fn occupied_bandwidth_fits_within_25_khz() {
        let psd = generate_psd(0.0, 120);
        let (centroid, obw99) = centroid_and_occupied_bw(&psd, 0.99);
        println!("measured 99% occupied bandwidth: {obw99} Hz (centroid {centroid:.0} Hz)");
        assert!(
            obw99 <= MAX_OCCUPIED_BANDWIDTH_HZ,
            "99% occupied bandwidth {obw99} Hz exceeds 25 kHz"
        );
        // Sanity: the theoretical RRC bandwidth is symbol_rate*(1+rolloff).
        assert!(
            obw99 > 0.8 * SYMBOL_RATE_HZ,
            "occupied bandwidth {obw99} Hz implausibly narrow"
        );

        let in_band: Vec<f64> = psd
            .iter()
            .filter(|p| (p.freq_hz - centroid).abs() < 0.4 * SYMBOL_RATE_HZ)
            .map(|p| p.power_db)
            .collect();
        let in_band_db = 10.0
            * (in_band.iter().map(|d| 10f64.powf(d / 10.0)).sum::<f64>() / in_band.len() as f64)
                .log10();
        let worst_out_of_band = psd
            .iter()
            .filter(|p| (p.freq_hz - centroid).abs() > MAX_OCCUPIED_BANDWIDTH_HZ / 2.0)
            .map(|p| p.power_db)
            .fold(f64::MIN, f64::max);
        assert!(
            worst_out_of_band < in_band_db - 30.0,
            "out-of-band peak {worst_out_of_band:.1} dB is within 30 dB of the in-band level {in_band_db:.1} dB"
        );
    }
}
