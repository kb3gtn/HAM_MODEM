//! The AX.25/KISS TX and RX signal-chain wrappers - the counterparts to
//! `tx_signal::TxSignalGenerator`/`rx_signal::RxSignalProcessor`'s PRBS/BERT
//! mode, sharing the same underlying pulse-shaping/resample/DSP machinery
//! but sourcing/sinking real framed packet data via KISS instead of a test
//! pattern. Kept as separate types (some duplication of the TX tail logic
//! with `TxSignalGenerator`) rather than refactoring that already-tested
//! struct to be source-agnostic - deliberately the lower-risk choice for
//! adding this feature, not a statement that the duplication is permanent.
//!
//! TX bit order (per project decision): KISS frame from host -> raw AX.25
//! bytes -> `hdlc::frame_bits` (bit-stuff + FCS + flags) -> `Scrambler` ->
//! `conv::ConvEncoder` (K=7 r=1/2: every data bit becomes two coded bits) ->
//! 8PSK symbol map (3 coded bits/symbol, continuous across frame boundaries -
//! HDLC is bit-synchronous, so frames need not be symbol-aligned)/pulse shape/
//! resample/freq-shift/quantize (same tail as `TxSignalGenerator`). When no frame is queued, continuously scrambled
//! idle flags are sent instead - satisfying the established "TX bit source
//! must never block" pacing rule with a real, standard HDLC idle-line
//! convention, not an arbitrary filler pattern.
//!
//! RX bit order (mirror image): derotated symbols sliced to constellation
//! positions -> `hdlc::RotationResolvingDeframer` (8 parallel demap ->
//! `Descrambler` -> HDLC branches that also resolve the 8PSK 45-degree phase
//! ambiguity - see `hdlc` module docs) -> valid frames wrapped in KISS and
//! sent to the connected host.

use std::collections::VecDeque;
use std::sync::Arc;

use bladerf::ComplexI16;

use crate::conv::ConvEncoder;
use crate::hdlc;
use crate::kiss::KissServer;
use crate::resampler::TxResampler;
use crate::rrc::{PulseShaper, ShapeMode};
use crate::scrambler::Scrambler;
use crate::symbol_map::Modulation;
use num_complex::Complex32;

pub struct Ax25TxSignalGenerator {
    modulation: Modulation,
    /// Shared with the RX side: a KISS host talks to ONE socket, sending
    /// frames to transmit and receiving the frames the receiver hears.
    kiss: Arc<KissServer>,
    scrambler: Scrambler,
    encoder: ConvEncoder,
    /// Coded bits waiting to be packed three to a symbol.
    coded_bits: VecDeque<u8>,
    /// Scrambled bits from the current frame (or idle-flag filler), waiting
    /// to be pulse-shaped one at a time.
    pending_bits: VecDeque<u8>,
    pulse_shaper: PulseShaper,
    resampler: TxResampler,
    fifo: VecDeque<f32>,
    shaped_buf: Vec<Complex32>,
    output_scale: f32,
    hardware_sample_rate: f64,
    freq_shift_phase_increment: f64,
    freq_shift_phase: f64,
    frames_sent: u64,
}

impl Ax25TxSignalGenerator {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        kiss: Arc<KissServer>,
        modulation: Modulation,
        symbol_rate_hz: f64,
        hardware_sample_rate: u32,
        sps: usize,
        rolloff: f64,
        span_symbols: usize,
        output_chunk_frames: usize,
        max_resample_ratio_relative: f64,
        output_scale: f32,
        freq_shift_hz: f64,
    ) -> Self {
        let symbol_domain_rate = symbol_rate_hz * sps as f64;
        let resample_ratio = hardware_sample_rate as f64 / symbol_domain_rate;
        let freq_shift_phase_increment =
            2.0 * std::f64::consts::PI * freq_shift_hz / hardware_sample_rate as f64;
        Ax25TxSignalGenerator {
            modulation,
            kiss,
            scrambler: Scrambler::new(),
            encoder: ConvEncoder::new(),
            coded_bits: VecDeque::new(),
            pending_bits: VecDeque::new(),
            pulse_shaper: PulseShaper::new(sps, rolloff, span_symbols, ShapeMode::Rrc),
            resampler: TxResampler::new(
                resample_ratio,
                output_chunk_frames,
                max_resample_ratio_relative,
            ),
            fifo: VecDeque::new(),
            shaped_buf: Vec::with_capacity(sps),
            output_scale,
            hardware_sample_rate: hardware_sample_rate as f64,
            freq_shift_phase_increment,
            freq_shift_phase: 0.0,
            frames_sent: 0,
        }
    }

    fn next_bit(&mut self) -> u8 {
        if self.pending_bits.is_empty() {
            let raw_bits: Vec<u8> = match self.kiss.try_recv_frame() {
                Some(frame) if !frame.is_empty() => {
                    self.frames_sent += 1;
                    hdlc::frame_bits(&frame)
                }
                _ => hdlc::FLAG.to_vec(),
            };
            self.pending_bits
                .extend(raw_bits.iter().map(|&b| self.scrambler.scramble_bit(b)));
        }
        self.pending_bits.pop_front().unwrap_or(0)
    }

    pub fn frames_sent(&self) -> u64 {
        self.frames_sent
    }

    pub fn has_kiss_client(&self) -> bool {
        self.kiss.has_client()
    }

    /// Change the TX offset-tuning frequency shift at runtime. The NCO phase is
    /// kept, so the output stays phase-continuous across the change.
    pub fn set_freq_shift_hz(&mut self, freq_shift_hz: f64) {
        self.freq_shift_phase_increment =
            std::f64::consts::TAU * freq_shift_hz / self.hardware_sample_rate;
    }

    /// Produce the next fixed-size chunk of hardware-rate IQ samples -
    /// identical tail shape to `TxSignalGenerator::next_chunk`, just pulling
    /// bits from the KISS/AX.25 source above instead of a `TxBert`.
    pub fn next_chunk(&mut self) -> Vec<ComplexI16> {
        let needed = self.resampler.input_frames_needed() * 2;
        while self.fifo.len() < needed {
            let bps = self.modulation.bits_per_symbol();
            while self.coded_bits.len() < bps {
                let data_bit = self.next_bit();
                let pair = self.encoder.encode_bit(data_bit);
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
                self.fifo.push_back(s.re);
                self.fifo.push_back(s.im);
            }
        }
        let input: Vec<f32> = self.fifo.drain(..needed).collect();
        let output = self.resampler.process(&input);

        output
            .chunks_exact(2)
            .map(|c| {
                let (i_raw, q_raw) = (c[0], c[1]);
                let (sin_p, cos_p) = self.freq_shift_phase.sin_cos();
                let i_shifted = i_raw * cos_p as f32 - q_raw * sin_p as f32;
                let q_shifted = i_raw * sin_p as f32 + q_raw * cos_p as f32;
                self.freq_shift_phase += self.freq_shift_phase_increment;
                if self.freq_shift_phase.abs() > std::f64::consts::TAU {
                    self.freq_shift_phase %= std::f64::consts::TAU;
                }

                let i_val = i_shifted * self.output_scale;
                let q_val = q_shifted * self.output_scale;
                let i = (i_val * 2048.0).round().clamp(-2048.0, 2047.0) as i16;
                let q = (q_val * 2048.0).round().clamp(-2048.0, 2047.0) as i16;
                ComplexI16::new(i, q)
            })
            .collect()
    }
}

/// RX-side AX.25 post-processing: takes the DSP chain's derotated complex
/// symbols, slices them to constellation positions, and runs the
/// rotation-resolving descramble/deframe bank - reporting complete,
/// FCS-valid frames as they're found. Deliberately separate from
/// `RxSignalProcessor` (which stays general-purpose: DSP chain + optional
/// PRBS BERT for bench testing) rather than baked into it.
pub struct Ax25RxPipeline {
    deframer: hdlc::RotationResolvingDeframer,
}

impl Ax25RxPipeline {
    pub fn new(modulation: Modulation) -> Self {
        Ax25RxPipeline {
            deframer: hdlc::RotationResolvingDeframer::new(modulation),
        }
    }

    /// Feed the chain's symbols (as returned by `RxSignalProcessor::process`).
    /// Returns any complete frames found.
    pub fn process(&mut self, symbols: &[Complex32]) -> Vec<Vec<u8>> {
        symbols
            .iter()
            .flat_map(|&y| self.deframer.process_symbol(y))
            .collect()
    }

    pub fn bad_frame_count(&self) -> u64 {
        self.deframer.bad_frame_count()
    }

    /// Raw (pre-FEC) channel BER seen by the locked decoder, if locked.
    pub fn pre_fec_ber(&self) -> Option<f64> {
        self.deframer.pre_fec_ber()
    }

    /// Stop or restart the frame decoder (see `RotationResolvingDeframer::set_enabled`).
    pub fn set_enabled(&mut self, enabled: bool) {
        self.deframer.set_enabled(enabled);
    }

    pub fn is_enabled(&self) -> bool {
        self.deframer.is_enabled()
    }

    /// The (360/M)-degree rotation step the deframer is currently locked to.
    pub fn locked_rotation(&self) -> Option<u8> {
        self.deframer.locked_rotation()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prbs::PrbsPattern;
    use crate::rx_signal::RxSignalProcessor;

    /// Full software-only integration test, mirroring `rx_signal.rs`'s own
    /// integration test but exercising the AX.25/KISS path end to end: a
    /// KISS client sends a frame in on the TX side, the frame is scrambled
    /// and pulse-shaped, run through the DSP chain (matched filter, timing and
    /// carrier recovery - a real channel, not a shortcut), then descrambled
    /// and deframed on the RX side, and the exact original bytes must come
    /// back out.
    #[test]
    fn kiss_frame_survives_the_full_tx_rx_chain() {
        use std::io::Write;
        use std::net::TcpStream;
        use std::time::Duration;

        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        drop(listener);
        let kiss = Arc::new(KissServer::start(&addr.to_string()).expect("bind kiss server"));
        std::thread::sleep(Duration::from_millis(50));

        let mut client = TcpStream::connect(addr).expect("connect to kiss server");
        std::thread::sleep(Duration::from_millis(50));

        use crate::params::*;
        let hardware_rate = HARDWARE_SAMPLE_RATE as f64;
        let symbol_rate_hz = SYMBOL_RATE_HZ;
        let sps = SPS;
        let rolloff = RRC_ROLLOFF;
        let span_symbols = RRC_SPAN_SYMBOLS;

        // Offset-tune the TX away from the LO (as in real use), with a small
        // deliberate mismatch at the RX for the carrier loop to pull in.
        let freq_shift_hz = 15_000.0;
        let residual_hz = 300.0;
        let mut tx = Ax25TxSignalGenerator::new(
            kiss,
            Modulation::Psk8,
            symbol_rate_hz,
            hardware_rate as u32,
            sps,
            rolloff,
            span_symbols,
            4096,
            10.0,
            1.0,
            freq_shift_hz + residual_hz,
        );

        let mut rx = RxSignalProcessor::new(
            PrbsPattern::Pn15, // irrelevant in ax25 mode - RxSignalProcessor's embedded BERT is simply unused here
            Modulation::Psk8,
            symbol_rate_hz,
            hardware_rate,
            sps,
            rolloff,
            span_symbols,
            50.0,
            freq_shift_hz,
            100.0,
            0.707,
            0.02,
            0.707,
            4096,
            10.0,
        );
        let mut ax25_rx = Ax25RxPipeline::new(Modulation::Psk8);

        // Warm up on idle-flag filler first, so carrier/timing recovery have
        // settled BEFORE the real frame is injected - otherwise the frame would be the very
        // first thing transmitted, arriving during the unlocked transient
        // and getting lost for reasons having nothing to do with framing
        // correctness.
        for _ in 0..300 {
            let hw_chunk = tx.next_chunk();
            let decisions = rx.process(&hw_chunk);
            ax25_rx.process(&decisions);
        }

        let payload = b"a real AX.25-ish test frame".to_vec();
        client.write_all(&crate::kiss::encode(&payload)).unwrap();
        std::thread::sleep(Duration::from_millis(100));

        let mut found_frames = Vec::new();
        for _ in 0..100 {
            let hw_chunk = tx.next_chunk();
            let decisions = rx.process(&hw_chunk);
            found_frames.extend(ax25_rx.process(&decisions));
        }

        assert!(
            found_frames.contains(&payload),
            "expected to find {payload:?} among received frames {found_frames:?}"
        );
    }
}
