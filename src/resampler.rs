//! Wraps rubato's arbitrary-ratio sinc resampler for both directions:
//! `TxResampler` has a fixed OUTPUT chunk size (matches what the bladeRF
//! write call wants), variable input consumption - the pull-shaped API
//! matching TX's hardware pull pacing (see tx_chain.rs's pacing-model doc
//! comment). `RxResampler` has a fixed INPUT chunk size (matches the
//! fixed-size buffers bladeRF hands us per RX read), variable output -
//! push-shaped, matching RX's hardware push pacing. I and Q are carried as
//! interleaved samples in one buffer (2 channels) in both.

use audioadapter_buffers::direct::InterleavedSlice;
use rubato::{
    Async, FixedAsync, Resampler, SincInterpolationParameters, SincInterpolationType,
    WindowFunction,
};

const CHANNELS: usize = 2; // I, Q

struct TxSinc {
    resampler: Async<f32>,
    output_chunk_frames: usize,
}

impl TxSinc {
    /// `resample_ratio` = output_rate / input_rate (e.g. hardware_rate /
    /// symbol_domain_rate). `output_chunk_frames` = fixed number of IQ
    /// frames produced per `process` call, sized to match the bladeRF TX
    /// write buffer. `max_resample_ratio_relative` allows the ratio to be
    /// adjusted later (e.g. for a data-rate change) without rebuilding the
    /// resampler, up to that multiple in either direction.
    pub fn new(
        resample_ratio: f64,
        output_chunk_frames: usize,
        max_resample_ratio_relative: f64,
    ) -> Self {
        let params = SincInterpolationParameters::new(128, WindowFunction::Blackman2)
            .oversampling_factor(512)
            .interpolation(SincInterpolationType::Cubic);
        let resampler = Async::<f32>::new_sinc(
            resample_ratio,
            max_resample_ratio_relative,
            &params,
            output_chunk_frames,
            CHANNELS,
            FixedAsync::Output,
        )
        .expect("valid resampler construction parameters");
        TxSinc {
            resampler,
            output_chunk_frames,
        }
    }

    /// How many interleaved input frames (I,Q pairs) `process` needs next.
    pub fn input_frames_needed(&self) -> usize {
        self.resampler.input_frames_next()
    }

    /// `input_interleaved` must have exactly `input_frames_needed() * 2`
    /// samples (I0,Q0,I1,Q1,...). Returns `output_chunk_frames * 2`
    /// interleaved output samples.
    pub fn process(&mut self, input_interleaved: &[f32]) -> Vec<f32> {
        let n_in = self.input_frames_needed();
        assert_eq!(
            input_interleaved.len(),
            n_in * CHANNELS,
            "input buffer size must match input_frames_needed()"
        );
        let input_adapter = InterleavedSlice::new(input_interleaved, CHANNELS, n_in)
            .expect("valid input adapter dimensions");
        let mut output = vec![0.0f32; self.output_chunk_frames * CHANNELS];
        let mut output_adapter =
            InterleavedSlice::new_mut(&mut output, CHANNELS, self.output_chunk_frames)
                .expect("valid output adapter dimensions");
        self.resampler
            .process_into_buffer(&input_adapter, &mut output_adapter, None)
            .expect("resampling failed");
        output
    }
}

struct RxSinc {
    resampler: Async<f32>,
    input_chunk_frames: usize,
}

impl RxSinc {
    /// `resample_ratio` = output_rate / input_rate (e.g. symbol_domain_rate
    /// / hardware_rate - a fraction less than 1, since this decimates).
    /// `input_chunk_frames` = fixed number of IQ frames consumed per
    /// `process` call, sized to match the buffers the bladeRF RX read
    /// hands us. `max_resample_ratio_relative` allows the ratio to be
    /// adjusted later (e.g. for a data-rate change) without rebuilding the
    /// resampler, up to that multiple in either direction.
    pub fn new(
        resample_ratio: f64,
        input_chunk_frames: usize,
        max_resample_ratio_relative: f64,
    ) -> Self {
        let params = SincInterpolationParameters::new(128, WindowFunction::Blackman2)
            .oversampling_factor(512)
            .interpolation(SincInterpolationType::Cubic);
        let resampler = Async::<f32>::new_sinc(
            resample_ratio,
            max_resample_ratio_relative,
            &params,
            input_chunk_frames,
            CHANNELS,
            FixedAsync::Input,
        )
        .expect("valid resampler construction parameters");
        RxSinc {
            resampler,
            input_chunk_frames,
        }
    }

    /// How many interleaved output frames (I,Q pairs) the next `process`
    /// call will produce - varies call to call, since input is fixed-size
    /// here and output is not.
    #[allow(dead_code)]
    fn output_frames_next(&self) -> usize {
        self.resampler.output_frames_next()
    }

    /// `input_interleaved` must have exactly `input_chunk_frames() * 2`
    /// samples (I0,Q0,I1,Q1,...). Returns a variable number of interleaved
    /// output samples - check `output_frames_next()` beforehand, or just
    /// use the returned `Vec`'s length.
    pub fn process(&mut self, input_interleaved: &[f32]) -> Vec<f32> {
        assert_eq!(
            input_interleaved.len(),
            self.input_chunk_frames * CHANNELS,
            "input buffer size must match input_chunk_frames"
        );
        let input_adapter =
            InterleavedSlice::new(input_interleaved, CHANNELS, self.input_chunk_frames)
                .expect("valid input adapter dimensions");
        let out_frames_max = self.resampler.output_frames_max();
        let mut output = vec![0.0f32; out_frames_max * CHANNELS];
        let mut output_adapter = InterleavedSlice::new_mut(&mut output, CHANNELS, out_frames_max)
            .expect("valid output adapter dimensions");
        let (_frames_read, frames_written) = self
            .resampler
            .process_into_buffer(&input_adapter, &mut output_adapter, None)
            .expect("resampling failed");
        output.truncate(frames_written * CHANNELS);
        output
    }

    #[allow(dead_code)]
    pub fn input_chunk_frames(&self) -> usize {
        self.input_chunk_frames
    }
}

// ------------------------------------------------------------------------
// Public resamplers: pass-through / integer-ratio fast paths, else sinc
// ------------------------------------------------------------------------

/// Is `x` (to within a tiny tolerance) an integer, and which?
fn as_integer(x: f64) -> Option<usize> {
    let r = x.round();
    ((x - r).abs() < 1e-9 && r >= 1.0).then_some(r as usize)
}

/// I0, the modified Bessel function of order 0 (series; for the Kaiser window).
fn bessel_i0(x: f64) -> f64 {
    let (mut sum, mut term) = (1.0, 1.0);
    for k in 1..60 {
        term *= (x / (2.0 * f64::from(k))).powi(2);
        sum += term;
        if term < 1e-14 * sum {
            break;
        }
    }
    sum
}

/// Low-pass FIR for an integer rate change by `factor`: a Kaiser-windowed sinc
/// cut off at half the LOWER rate (so -6 dB there), `12 * factor + 1` taps,
/// normalised to DC gain 1. The signal of interest (the RRC band, at most
/// ~0.17 of the lower rate even at 4 samples/symbol) sits well inside the flat
/// part, and the first alias band is above ~0.67 of it.
fn integer_lowpass(factor: usize) -> Vec<f32> {
    let n = 12 * factor + 1;
    let beta = 7.0;
    let mid = (n - 1) as f64 / 2.0;
    let mut taps: Vec<f64> = (0..n)
        .map(|k| {
            let x = k as f64 - mid;
            let arg = std::f64::consts::PI * x / factor as f64;
            let sinc = if x == 0.0 { 1.0 } else { arg.sin() / arg };
            let r = x / mid;
            sinc * bessel_i0(beta * (1.0 - r * r).max(0.0).sqrt()) / bessel_i0(beta)
        })
        .collect();
    let sum: f64 = taps.iter().sum();
    for t in &mut taps {
        *t /= sum;
    }
    taps.into_iter().map(|t| t as f32).collect()
}

/// Polyphase decimate-by-D of interleaved I/Q, carrying state across calls.
struct IntegerDecimator {
    factor: usize,
    taps: Vec<f32>,
    /// The newest `taps.len()` input frames, newest last (I,Q interleaved).
    history: Vec<f32>,
    /// Input frames until the next output is due.
    until_output: usize,
}

impl IntegerDecimator {
    fn new(factor: usize) -> Self {
        let taps = integer_lowpass(factor);
        IntegerDecimator {
            factor,
            history: vec![0.0; taps.len() * CHANNELS],
            taps,
            until_output: factor,
        }
    }

    fn process(&mut self, input: &[f32]) -> Vec<f32> {
        let n = self.taps.len();
        let mut out = Vec::with_capacity(input.len() / self.factor + CHANNELS);
        // Work on history ++ input so each output reads a contiguous window.
        let mut buf = std::mem::take(&mut self.history);
        buf.extend_from_slice(input);
        let frames_in = input.len() / CHANNELS;
        for k in 0..frames_in {
            self.until_output -= 1;
            if self.until_output == 0 {
                self.until_output = self.factor;
                // Window of the n frames ending at input frame k (inclusive).
                let window = &buf[(k + 1) * CHANNELS..(k + 1 + n) * CHANNELS];
                let (mut i, mut q) = (0.0f32, 0.0f32);
                for (t, frame) in self.taps.iter().rev().zip(window.chunks_exact(CHANNELS)) {
                    i += t * frame[0];
                    q += t * frame[1];
                }
                out.push(i);
                out.push(q);
            }
        }
        self.history = buf.split_off(frames_in * CHANNELS);
        out
    }
}

/// Polyphase interpolate-by-U of interleaved I/Q (zero-stuff + low-pass),
/// carrying state across calls. Gain is U so amplitudes are preserved.
struct IntegerInterpolator {
    factor: usize,
    /// `taps` scaled by U.
    taps: Vec<f32>,
    /// Newest `ceil(taps/U)` input frames, newest last.
    history: Vec<f32>,
}

impl IntegerInterpolator {
    fn new(factor: usize) -> Self {
        let taps: Vec<f32> = integer_lowpass(factor)
            .into_iter()
            .map(|t| t * factor as f32)
            .collect();
        let hist_frames = taps.len().div_ceil(factor);
        IntegerInterpolator {
            factor,
            taps,
            history: vec![0.0; hist_frames * CHANNELS],
        }
    }

    fn process(&mut self, input: &[f32]) -> Vec<f32> {
        let u = self.factor;
        let hist_frames = self.history.len() / CHANNELS;
        let mut buf = std::mem::take(&mut self.history);
        buf.extend_from_slice(input);
        let frames_in = input.len() / CHANNELS;
        let mut out = Vec::with_capacity(frames_in * u * CHANNELS);
        for k in 0..frames_in {
            // Input frame k is buf frame hist_frames + k; taps[p + u*m] weights frame (k - m).
            for phase in 0..u {
                let (mut i, mut q) = (0.0f32, 0.0f32);
                let mut m = 0;
                while phase + u * m < self.taps.len() {
                    let frame = (hist_frames + k - m) * CHANNELS;
                    let t = self.taps[phase + u * m];
                    i += t * buf[frame];
                    q += t * buf[frame + 1];
                    m += 1;
                }
                out.push(i);
                out.push(q);
            }
        }
        self.history = buf.split_off(frames_in * CHANNELS);
        out
    }
}

enum TxInner {
    Pass,
    Interp(IntegerInterpolator),
    Sinc(TxSinc),
}

pub struct TxResampler {
    inner: TxInner,
    output_chunk_frames: usize,
    /// Interpolated output frames not yet handed out (Interp only).
    leftover: std::collections::VecDeque<f32>,
}

impl TxResampler {
    /// `resample_ratio` = output_rate / input_rate. A ratio of exactly 1 passes
    /// samples through, an integer ratio uses a polyphase interpolator, and
    /// anything else the arbitrary-ratio sinc resampler. `output_chunk_frames`
    /// = fixed number of IQ frames produced per `process` call.
    /// `max_resample_ratio_relative` only applies to the sinc path.
    pub fn new(
        resample_ratio: f64,
        output_chunk_frames: usize,
        max_resample_ratio_relative: f64,
    ) -> Self {
        let inner = match as_integer(resample_ratio) {
            Some(1) => TxInner::Pass,
            Some(u) => TxInner::Interp(IntegerInterpolator::new(u)),
            None => TxInner::Sinc(TxSinc::new(
                resample_ratio,
                output_chunk_frames,
                max_resample_ratio_relative,
            )),
        };
        TxResampler {
            inner,
            output_chunk_frames,
            leftover: std::collections::VecDeque::new(),
        }
    }

    /// How many interleaved input frames (I,Q pairs) `process` needs next.
    pub fn input_frames_needed(&self) -> usize {
        match &self.inner {
            TxInner::Pass => self.output_chunk_frames,
            TxInner::Interp(i) => {
                (self.output_chunk_frames - self.leftover.len() / CHANNELS).div_ceil(i.factor)
            }
            TxInner::Sinc(s) => s.input_frames_needed(),
        }
    }

    /// `input_interleaved` must have exactly `input_frames_needed() * 2`
    /// samples (I0,Q0,I1,Q1,...). Returns `output_chunk_frames * 2`
    /// interleaved output samples.
    pub fn process(&mut self, input_interleaved: &[f32]) -> Vec<f32> {
        assert_eq!(
            input_interleaved.len(),
            self.input_frames_needed() * CHANNELS,
            "input buffer size must match input_frames_needed()"
        );
        match &mut self.inner {
            TxInner::Pass => input_interleaved.to_vec(),
            TxInner::Interp(i) => {
                self.leftover.extend(i.process(input_interleaved));
                self.leftover
                    .drain(..self.output_chunk_frames * CHANNELS)
                    .collect()
            }
            TxInner::Sinc(s) => s.process(input_interleaved),
        }
    }
}

enum RxInner {
    Pass,
    Decim(IntegerDecimator),
    Sinc(RxSinc),
}

pub struct RxResampler {
    inner: RxInner,
    input_chunk_frames: usize,
}

impl RxResampler {
    /// `resample_ratio` = output_rate / input_rate (a fraction at most 1, since
    /// this decimates). A ratio of exactly 1 passes samples through, 1/D for
    /// an integer D uses a polyphase decimator, anything else the
    /// arbitrary-ratio sinc resampler. `input_chunk_frames` = fixed number of
    /// IQ frames consumed per `process` call.
    pub fn new(
        resample_ratio: f64,
        input_chunk_frames: usize,
        max_resample_ratio_relative: f64,
    ) -> Self {
        let inner = match as_integer(1.0 / resample_ratio) {
            Some(1) => RxInner::Pass,
            Some(d) => RxInner::Decim(IntegerDecimator::new(d)),
            None => RxInner::Sinc(RxSinc::new(
                resample_ratio,
                input_chunk_frames,
                max_resample_ratio_relative,
            )),
        };
        RxResampler {
            inner,
            input_chunk_frames,
        }
    }

    /// `input_interleaved` must have exactly `input_chunk_frames() * 2`
    /// samples (I0,Q0,I1,Q1,...). Returns a variable number of interleaved
    /// output samples.
    pub fn process(&mut self, input_interleaved: &[f32]) -> Vec<f32> {
        assert_eq!(
            input_interleaved.len(),
            self.input_chunk_frames * CHANNELS,
            "input buffer size must match input_chunk_frames"
        );
        match &mut self.inner {
            RxInner::Pass => input_interleaved.to_vec(),
            RxInner::Decim(d) => d.process(input_interleaved),
            RxInner::Sinc(s) => s.process(input_interleaved),
        }
    }

    pub fn input_chunk_frames(&self) -> usize {
        self.input_chunk_frames
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tone(n: usize, freq_cycles_per_sample: f64) -> Vec<f32> {
        (0..n)
            .flat_map(|k| {
                let ph = std::f64::consts::TAU * freq_cycles_per_sample * k as f64;
                [ph.cos() as f32, ph.sin() as f32]
            })
            .collect()
    }

    fn rms(x: &[f32]) -> f32 {
        (x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32).sqrt()
    }

    #[test]
    fn ratio_one_is_a_pass_through() {
        let mut rx = RxResampler::new(1.0, 1024, 10.0);
        let input = tone(1024, 0.01);
        assert_eq!(rx.process(&input), input);
        let mut tx = TxResampler::new(1.0, 1024, 10.0);
        assert_eq!(tx.input_frames_needed(), 1024);
        assert_eq!(tx.process(&input), input);
    }

    #[test]
    fn integer_decimator_keeps_in_band_tones_and_rejects_alias_bands() {
        for d in [2usize, 4, 5] {
            let mut rx = RxResampler::new(1.0 / d as f64, 8192, 10.0);
            // In band: 0.1 of the OUTPUT rate; out of band: 0.9 of it (first alias).
            for (f_out, expect_pass) in [(0.05, true), (0.15, true), (0.9, false)] {
                let mut r = RxResampler::new(1.0 / d as f64, 8192, 10.0);
                let mut out = Vec::new();
                for blk in 0..4 {
                    let x: Vec<f32> =
                        tone(8192 * (blk + 1), f_out / d as f64)[8192 * blk * 2..].to_vec();
                    out.extend(r.process(&x));
                }
                let settled = &out[out.len() / 2..];
                let level = rms(settled);
                if expect_pass {
                    assert!(
                        (level - std::f32::consts::FRAC_1_SQRT_2).abs() < 0.02,
                        "D={d} f={f_out}: level {level}"
                    );
                } else {
                    assert!(level < 0.01, "D={d} f={f_out}: alias leaked at {level}");
                }
            }
            // Output length: input/D per call, carried across calls.
            let out_len: usize = (0..5)
                .map(|_| rx.process(&vec![0.0; 8192 * 2]).len() / 2)
                .sum();
            assert_eq!(out_len, 5 * 8192 / d, "D={d}");
        }
    }

    #[test]
    fn integer_interpolator_preserves_amplitude_and_rejects_images() {
        for u in [2usize, 4, 5] {
            let mut tx = TxResampler::new(u as f64, 4096, 10.0);
            let mut out = Vec::new();
            let mut k0 = 0usize;
            for _ in 0..6 {
                let n = tx.input_frames_needed();
                let ph = 0.05; // a tone at 0.05 of the INPUT rate
                let x: Vec<f32> = (k0..k0 + n)
                    .flat_map(|k| {
                        let a = std::f64::consts::TAU * ph * k as f64;
                        [a.cos() as f32, a.sin() as f32]
                    })
                    .collect();
                k0 += n;
                let y = tx.process(&x);
                assert_eq!(y.len(), 4096 * 2, "U={u}: fixed output chunks");
                out.extend(y);
            }
            let settled = &out[out.len() / 2..];
            assert!(
                (rms(settled) - std::f32::consts::FRAC_1_SQRT_2).abs() < 0.02,
                "U={u}: level {}",
                rms(settled)
            );
            // The image at 1 - 0.05 of the input rate (in output-rate terms: (1-0.05)/U) must be gone:
            // the output is a clean tone, so its power is entirely at the wanted frequency.
            let w = std::f64::consts::TAU * 0.05 / u as f64;
            let (mut ci, mut cq) = (0.0f64, 0.0f64);
            for (k, f) in settled.chunks_exact(2).enumerate() {
                let (c, s) = ((w * k as f64).cos(), (w * k as f64).sin());
                ci += f64::from(f[0]) * c + f64::from(f[1]) * s;
                cq += f64::from(f[1]) * c - f64::from(f[0]) * s;
            }
            let tone_amp = (ci * ci + cq * cq).sqrt() / (settled.len() / 2) as f64;
            let total = f64::from(rms(settled)) * std::f64::consts::SQRT_2; // amplitude of the whole output
            assert!(
                (tone_amp - total).abs() < 0.02,
                "U={u}: tone {tone_amp} vs total {total}: images present"
            );
        }
    }

    #[test]
    fn produces_fixed_size_output_chunks() {
        // 20x upsample, matching the 5kbps default (40kHz symbol-domain -> 800kHz hardware).
        let mut resampler = TxResampler::new(20.0, 1024, 10.0);
        let n_in = resampler.input_frames_needed();
        assert!(n_in > 0);
        let input = vec![0.0f32; n_in * 2];
        let output = resampler.process(&input);
        assert_eq!(output.len(), 1024 * 2);
    }

    #[test]
    fn rx_resampler_decimates_a_fixed_input_chunk() {
        // 1/20 decimation, the inverse of the TX test above: 800kHz hardware
        // -> 40kHz symbol-domain.
        let mut resampler = RxResampler::new(1.0 / 20.0, 8192, 10.0);
        let input = vec![0.0f32; 8192 * 2];
        let output = resampler.process(&input);
        assert_eq!(
            output.len() % CHANNELS,
            0,
            "output must be whole interleaved frames"
        );
        // Output should be roughly input_len/20 frames (exact chunk sizing
        // is the resampler's own internal bookkeeping, not something to
        // hardcode here - just check it's in the right ballpark).
        let out_frames = output.len() / 2;
        assert!(
            out_frames > 8192 / 20 / 2 && out_frames < 8192 / 20 * 2,
            "output {out_frames} frames, expected roughly {} (8192/20)",
            8192 / 20
        );
    }

    #[test]
    fn tx_then_rx_resampler_round_trip_preserves_a_tone() {
        // Sanity check the two resamplers are consistent inverses of each
        // other: upsample a tone 20x on "TX", decimate it back 20x on "RX",
        // and confirm the tone survives recognizably (same frequency,
        // reasonable amplitude) - not a precise bit-exact check, just that
        // nothing is drastically broken end to end.
        let mut tx = TxResampler::new(20.0, 8192, 10.0);
        let mut rx = RxResampler::new(1.0 / 20.0, 8192, 10.0);

        let tone_hz = 2000.0;
        let low_rate = 40_000.0;
        let n_in = tx.input_frames_needed();
        let low_rate_tone: Vec<f32> = (0..n_in)
            .flat_map(|k| {
                let phase = 2.0 * std::f64::consts::PI * tone_hz * (k as f64) / low_rate;
                [phase.cos() as f32, phase.sin() as f32]
            })
            .collect();

        let hi_rate = tx.process(&low_rate_tone);
        assert_eq!(hi_rate.len(), 8192 * 2);

        let back = rx.process(&hi_rate);
        assert!(!back.is_empty());
        // Just confirm it didn't collapse to silence/garbage.
        let rms: f32 = (back.iter().map(|s| s * s).sum::<f32>() / back.len() as f32).sqrt();
        assert!(rms > 0.1, "round-tripped signal RMS {rms} looks too small - resamplers may not be consistent inverses");
    }

    #[test]
    #[ignore]
    fn bench_front_end_stages() {
        use crate::dc_block::DcBlocker;
        use num_complex::Complex32;
        use std::time::Instant;
        let n = 8_000_000usize;
        let x: Vec<Complex32> = (0..n)
            .map(|k| Complex32::new((k as f32 * 0.01).sin(), (k as f32 * 0.013).cos()))
            .collect();
        let t = Instant::now();
        let mut dc = DcBlocker::from_cutoff_hz(50.0, 4e6);
        let y: Vec<Complex32> = x.iter().map(|&v| dc.process(v)).collect();
        println!(
            "dc block        : {:6.1} Msamples/s",
            n as f64 / t.elapsed().as_secs_f64() / 1e6
        );
        let t = Instant::now();
        let (mut ph, inc) = (0.0f64, 0.05f64);
        let z: Vec<Complex32> = y
            .iter()
            .map(|&v| {
                let (s, c) = ph.sin_cos();
                ph = (ph + inc) % std::f64::consts::TAU;
                v * Complex32::new(c as f32, -s as f32)
            })
            .collect();
        println!(
            "sin_cos NCO     : {:6.1} Msamples/s",
            n as f64 / t.elapsed().as_secs_f64() / 1e6
        );
        let inter: Vec<f32> = z.iter().flat_map(|c| [c.re, c.im]).collect();
        for (ratio, name) in [
            (0.25f64, "rx sinc 4M->1M  "),
            (0.5, "rx sinc 4M->2M  "),
            (0.8, "rx sinc 5/4 ratio"),
        ] {
            let mut r = RxResampler::new(ratio, 40960, 10.0);
            let t = Instant::now();
            let mut done = 0;
            for chunk in inter.chunks_exact(40960 * 2) {
                r.process(chunk);
                done += 40960;
            }
            println!(
                "{name}: {:6.1} Msamples/s (input)",
                done as f64 / t.elapsed().as_secs_f64() / 1e6
            );
        }
        let mut r = TxResampler::new(4.0, 40960, 10.0);
        let t = Instant::now();
        let mut out = 0;
        for _ in 0..40 {
            let need = r.input_frames_needed();
            r.process(&vec![0.1f32; need * 2]);
            out += 40960;
        }
        println!(
            "tx sinc 1M->4M  : {:6.1} Msamples/s (output)",
            out as f64 / t.elapsed().as_secs_f64() / 1e6
        );
    }
}
