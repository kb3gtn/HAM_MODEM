//! K=7, rate-1/2 convolutional code (the NASA/CCSDS standard, generators
//! 171 and 133 octal) with a soft-decision Viterbi decoder.
//!
//! Encoder: each input bit produces two coded bits, `(parity(reg & G1),
//! parity(reg & G2))` in that order, where `reg` is the 7-bit window of the
//! current input (bit 6) and the six previous inputs (bit 0 oldest). The
//! stream is continuous - no termination/flush - which is what a
//! free-running HDLC link needs.
//!
//! Decoder: input is one soft value per coded bit ("LLR", positive = bit 0
//! more likely, negative = bit 1 more likely, magnitude = confidence; scale
//! is irrelevant). It pairs consecutive values into code symbols, runs the
//! 64-state add-compare-select, and emits decoded bits in blocks via
//! traceback with a fixed decision depth, so latency is bounded
//! (`DECISION_DEPTH`..`DECISION_DEPTH + TRACEBACK_BLOCK` input bits) and the
//! decoder can run forever on a stream whose start state it doesn't know.
//!
//! PAIR PHASE: the receiver does not know which coded bit is the first of a
//! pair (3 coded bits ride on each 8PSK symbol, so the pairing alternates
//! relative to symbol boundaries). `ViterbiDecoder::new(true)` drops the first
//! value it is given, which realigns the pairing; running one decoder of each
//! phase covers both.
//!
//! CHANNEL-ERROR TELEMETRY: when bits are emitted, the decoder re-encodes the
//! winning path and compares it to the hard decisions of what was actually
//! received. The mismatches are the raw channel errors the code corrected -
//! a pre-FEC bit error rate measured on live data without needing to know
//! what was sent. (Only meaningful once the decoder is on the right
//! alignment/rotation; elsewhere it just reports ~50%.)

const NUM_STATES: usize = 64;
const G1: u8 = 0b111_1001; // 171 octal
const G2: u8 = 0b101_1011; // 133 octal

/// Bits of traceback history required before a decoded bit is released.
pub const DECISION_DEPTH: usize = 64;
/// Input bits decoded per traceback.
pub const TRACEBACK_BLOCK: usize = 64;
const HISTORY: usize = DECISION_DEPTH + TRACEBACK_BLOCK;

/// Smoothing for the recent channel-BER estimate: per traceback block.
const RECENT_BER_RATE: f64 = 0.02;

fn parity(x: u8) -> u8 {
    (x.count_ones() & 1) as u8
}

/// `OUT[(input << 6) | state]` = coded pair as `(c1 << 1) | c2`.
const fn build_output_table() -> [u8; 128] {
    let mut t = [0u8; 128];
    let mut i = 0;
    while i < 128 {
        let reg = i as u8; // bit 6 = current input, bits 5..0 = state
        let c1 = ((reg & G1).count_ones() & 1) as u8;
        let c2 = ((reg & G2).count_ones() & 1) as u8;
        t[i] = (c1 << 1) | c2;
        i += 1;
    }
    t
}
static OUT: [u8; 128] = build_output_table();

const HALF: usize = NUM_STATES / 2;

/// Per butterfly `j` (predecessors 2j and 2j+1, successors j and 32+j): the
/// coded pair of the (state 2j, input 0) branch. Both generators have their
/// first and last taps set, which makes the other three branches of the
/// butterfly either the same pair or its complement (`^ 3`):
///   2j   -> j     : X[j]        2j+1 -> j     : X[j] ^ 3
///   2j   -> 32+j  : X[j] ^ 3    2j+1 -> 32+j  : X[j]
/// so only two branch metrics are needed per butterfly (see the unit test
/// that checks this against `OUT`).
const fn build_butterfly_table() -> [u8; HALF] {
    let mut t = [0u8; HALF];
    let mut j = 0;
    while j < HALF {
        t[j] = OUT[2 * j];
        j += 1;
    }
    t
}
static X: [u8; HALF] = build_butterfly_table();

pub struct ConvEncoder {
    state: u8, // the six previous input bits, most recent at bit 5
}

impl ConvEncoder {
    pub fn new() -> Self {
        ConvEncoder { state: 0 }
    }

    /// Encode one bit into its two coded bits `[c1, c2]`.
    pub fn encode_bit(&mut self, bit: u8) -> [u8; 2] {
        let reg = ((bit & 1) << 6) | self.state;
        self.state = reg >> 1;
        [parity(reg & G1), parity(reg & G2)]
    }
}

impl Default for ConvEncoder {
    fn default() -> Self {
        Self::new()
    }
}

/// One add-compare-select step over all 64 states: returns the new path
/// metrics (offset so state 0 is 0) and, per new state, whether the survivor
/// came from the odd predecessor.
fn acs(path_metric: &[f32; NUM_STATES], l1: f32, l2: f32) -> ([f32; NUM_STATES], u64) {
    // Cost of hypothesising each coded pair (c1,c2): a coded 1 costs its
    // LLR, a coded 0 costs nothing (LLR > 0 favours 0).
    let bm = [0.0, l2, l1, l1 + l2];
    // Branch metrics per butterfly, laid out for the vectorizable loop below.
    let mut bm_same = [0.0f32; HALF];
    let mut bm_comp = [0.0f32; HALF];
    for j in 0..HALF {
        bm_same[j] = bm[X[j] as usize];
        bm_comp[j] = bm[(X[j] ^ 3) as usize];
    }
    let mut even = [0.0f32; HALF];
    let mut odd = [0.0f32; HALF];
    for j in 0..HALF {
        even[j] = path_metric[2 * j];
        odd[j] = path_metric[2 * j + 1];
    }
    let mut new_pm = [0.0f32; NUM_STATES];
    let (mut dec_lo, mut dec_hi) = (0u32, 0u32);
    for j in 0..HALF {
        // successor j (input 0): from 2j with X, from 2j+1 with X^3
        let (a, b) = (even[j] + bm_same[j], odd[j] + bm_comp[j]);
        let take_odd = b < a;
        new_pm[j] = if take_odd { b } else { a };
        dec_lo |= u32::from(take_odd) << j;
        // successor 32+j (input 1): from 2j with X^3, from 2j+1 with X
        let (a, b) = (even[j] + bm_comp[j], odd[j] + bm_same[j]);
        let take_odd = b < a;
        new_pm[HALF + j] = if take_odd { b } else { a };
        dec_hi |= u32::from(take_odd) << j;
    }
    let dec = u64::from(dec_lo) | (u64::from(dec_hi) << 32);
    // Any constant keeps the metric differences; state 0 is cheaper than the minimum.
    let base = new_pm[0];
    for m in &mut new_pm {
        *m -= base;
    }
    (new_pm, dec)
}

pub struct ViterbiDecoder {
    path_metric: [f32; NUM_STATES],
    /// Per step, bit `s` set = the survivor into state `s` came from the odd
    /// predecessor.
    decisions: Vec<u64>,
    /// Per step, the hard-decided received pair `(c1 << 1) | c2`.
    hard: Vec<u8>,
    steps: usize,
    pending: Option<f32>,
    skip_next: bool,
    coded_bits: u64,
    channel_errors: u64,
    recent_ber: f64,
}

impl ViterbiDecoder {
    /// `skip_first_llr`: drop the first soft value received, shifting the
    /// pairing by one coded bit (see module docs).
    pub fn new(skip_first_llr: bool) -> Self {
        ViterbiDecoder {
            path_metric: [0.0; NUM_STATES],
            decisions: vec![0; HISTORY],
            hard: vec![0; HISTORY],
            steps: 0,
            pending: None,
            skip_next: skip_first_llr,
            coded_bits: 0,
            channel_errors: 0,
            recent_ber: 0.5,
        }
    }

    /// Feed one coded bit's soft value; any newly decoded bits are appended
    /// to `out`.
    pub fn push_llr(&mut self, llr: f32, out: &mut Vec<u8>) {
        if self.skip_next {
            self.skip_next = false;
            return;
        }
        match self.pending.take() {
            None => self.pending = Some(llr),
            Some(l1) => self.step(l1, llr, out),
        }
    }

    fn step(&mut self, l1: f32, l2: f32, out: &mut Vec<u8>) {
        let (new_pm, dec) = acs(&self.path_metric, l1, l2);
        self.path_metric = new_pm;

        let idx = self.steps % HISTORY;
        self.decisions[idx] = dec;
        self.hard[idx] = (u8::from(l1 < 0.0) << 1) | u8::from(l2 < 0.0);
        self.steps += 1;

        if self.steps >= HISTORY && self.steps % TRACEBACK_BLOCK == 0 {
            self.traceback(out);
        }
    }

    fn traceback(&mut self, out: &mut Vec<u8>) {
        let mut state = self
            .path_metric
            .iter()
            .enumerate()
            .min_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(s, _)| s)
            .unwrap_or(0);
        let first_step = self.steps - HISTORY;
        let mut bits = [0u8; HISTORY];
        let mut errors = 0u32;
        for k in (0..HISTORY).rev() {
            let idx = (first_step + k) % HISTORY;
            let x = ((self.decisions[idx] >> state) & 1) as usize;
            let u = state >> 5;
            let prev = ((state & 31) << 1) | x;
            bits[k] = u as u8;
            if k < TRACEBACK_BLOCK {
                errors += (OUT[(u << 6) | prev] ^ self.hard[idx]).count_ones();
            }
            state = prev;
        }
        out.extend_from_slice(&bits[..TRACEBACK_BLOCK]);
        self.coded_bits += 2 * TRACEBACK_BLOCK as u64;
        self.channel_errors += u64::from(errors);
        let block_ber = f64::from(errors) / (2.0 * TRACEBACK_BLOCK as f64);
        if self.coded_bits == 2 * TRACEBACK_BLOCK as u64 {
            self.recent_ber = block_ber; // first block: replace the "unknown" prior outright
        } else {
            self.recent_ber += RECENT_BER_RATE * (block_ber - self.recent_ber);
        }
    }

    /// (coded bits compared, of which disagreed with the decoded path).
    pub fn channel_error_counts(&self) -> (u64, u64) {
        (self.coded_bits, self.channel_errors)
    }

    /// Smoothed raw (pre-FEC) channel bit error rate over roughly the last
    /// 12 kbit. Starts at 0.5 (nothing known).
    pub fn recent_channel_ber(&self) -> f64 {
        self.recent_ber
    }
}

#[cfg(test)]
mod tests {
    /// The original, straightforward 64-state ACS, as the reference for `acs`.
    fn reference_acs(
        path_metric: &[f32; NUM_STATES],
        l1: f32,
        l2: f32,
    ) -> ([f32; NUM_STATES], u64) {
        let bm = [0.0, l2, l1, l1 + l2];
        let mut new_pm = [0.0f32; NUM_STATES];
        let mut dec: u64 = 0;
        for j in 0..NUM_STATES / 2 {
            let (s0, s1) = (2 * j, 2 * j + 1);
            let (p0, p1) = (path_metric[s0], path_metric[s1]);
            for u in 0..2usize {
                let s_new = (u << 5) | j;
                let m0 = p0 + bm[OUT[(u << 6) | s0] as usize];
                let m1 = p1 + bm[OUT[(u << 6) | s1] as usize];
                if m1 < m0 {
                    new_pm[s_new] = m1;
                    dec |= 1u64 << s_new;
                } else {
                    new_pm[s_new] = m0;
                }
            }
        }
        (new_pm, dec)
    }

    #[test]
    fn butterfly_branches_are_the_same_pair_or_its_complement() {
        for j in 0..HALF {
            let x = OUT[2 * j];
            assert_eq!(OUT[2 * j + 1], x ^ 3, "j={j}: (2j+1, u=0)");
            assert_eq!(OUT[(1 << 6) | (2 * j)], x ^ 3, "j={j}: (2j, u=1)");
            assert_eq!(OUT[(1 << 6) | (2 * j + 1)], x, "j={j}: (2j+1, u=1)");
        }
    }

    #[test]
    fn fast_acs_matches_the_reference_on_random_metrics() {
        let mut rng = Rng(0xACE5_1234);
        let mut pm = [0.0f32; NUM_STATES];
        for step in 0..20_000 {
            let (l1, l2) = ((rng.gauss() * 3.0) as f32, (rng.gauss() * 3.0) as f32);
            let (fast_pm, fast_dec) = acs(&pm, l1, l2);
            let (ref_pm, ref_dec) = reference_acs(&pm, l1, l2);
            assert_eq!(fast_dec, ref_dec, "step {step}: decisions differ");
            for s in 0..NUM_STATES {
                assert!(
                    ((fast_pm[s] - fast_pm[0]) - (ref_pm[s] - ref_pm[0])).abs() < 1e-3,
                    "step {step} state {s}"
                );
            }
            pm = fast_pm;
        }
    }

    use super::*;

    struct Rng(u32);
    impl Rng {
        fn next(&mut self) -> u32 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 17;
            self.0 ^= self.0 << 5;
            self.0
        }
        fn bit(&mut self) -> u8 {
            ((self.next() >> 7) & 1) as u8
        }
        fn gauss(&mut self) -> f64 {
            let u1 = (f64::from(self.next()) + 1.0) / (f64::from(u32::MAX) + 2.0);
            let u2 = (f64::from(self.next()) + 1.0) / (f64::from(u32::MAX) + 2.0);
            (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
        }
    }

    fn encode(bits: &[u8]) -> Vec<u8> {
        let mut enc = ConvEncoder::new();
        bits.iter().flat_map(|&b| enc.encode_bit(b)).collect()
    }

    fn decode(llrs: &[f32], skip: bool) -> (Vec<u8>, ViterbiDecoder) {
        let mut dec = ViterbiDecoder::new(skip);
        let mut out = Vec::new();
        for &l in llrs {
            dec.push_llr(l, &mut out);
        }
        (out, dec)
    }

    fn hard_llr(coded: &[u8]) -> Vec<f32> {
        coded
            .iter()
            .map(|&c| if c == 0 { 4.0 } else { -4.0 })
            .collect()
    }

    #[test]
    fn encoder_impulse_response_matches_the_171_133_generators() {
        // A single 1 then zeros reads out the generator taps, newest tap first:
        // G1 = 1111001, G2 = 1011011.
        let coded = encode(&[1, 0, 0, 0, 0, 0, 0]);
        let g1: Vec<u8> = coded.iter().step_by(2).copied().collect();
        let g2: Vec<u8> = coded.iter().skip(1).step_by(2).copied().collect();
        assert_eq!(g1, vec![1, 1, 1, 1, 0, 0, 1]);
        assert_eq!(g2, vec![1, 0, 1, 1, 0, 1, 1]);
    }

    #[test]
    fn decodes_an_error_free_stream_exactly() {
        let mut rng = Rng(0xC0DE_1234);
        let bits: Vec<u8> = (0..4000).map(|_| rng.bit()).collect();
        let (out, dec) = decode(&hard_llr(&encode(&bits)), false);
        assert!(
            out.len() >= 3_800 - TRACEBACK_BLOCK,
            "only {} bits out",
            out.len()
        );
        assert_eq!(&out[..], &bits[..out.len()]);
        assert_eq!(dec.channel_error_counts().1, 0);
        assert!(dec.recent_channel_ber() < 1e-6);
    }

    #[test]
    fn corrects_scattered_hard_errors_and_counts_them() {
        let mut rng = Rng(0xFEED_BEEF);
        let bits: Vec<u8> = (0..6000).map(|_| rng.bit()).collect();
        let mut coded = encode(&bits);
        let mut flipped = 0u64;
        for i in (100..coded.len()).step_by(61) {
            coded[i] ^= 1; // one error every ~61 coded bits: well within the code's power
            flipped += 1;
        }
        let (out, dec) = decode(&hard_llr(&coded), false);
        assert_eq!(
            &out[..],
            &bits[..out.len()],
            "decoded bits differ despite sparse errors"
        );
        let (compared, errors) = dec.channel_error_counts();
        assert!(compared > 10_000);
        // Every injected error inside the compared span must have been seen.
        assert!(
            errors as f64 > 0.95 * (flipped as f64 * compared as f64 / coded.len() as f64),
            "counted {errors} of ~{flipped} injected"
        );
    }

    #[test]
    fn pair_phase_skip_realigns_a_stream_that_starts_mid_pair() {
        let mut rng = Rng(0x1357_9BDF);
        let bits: Vec<u8> = (0..3000).map(|_| rng.bit()).collect();
        let llrs = hard_llr(&encode(&bits));
        // Receiver starts one coded bit late.
        let late = &llrs[1..];
        let (wrong, _) = decode(late, false);
        let (right, dec) = decode(late, true);
        // With the right phase the decoder tracks the stream (delayed by one
        // input bit versus a start at 0, since a half pair was dropped).
        let agree = |out: &[u8]| {
            out.iter()
                .zip(bits.iter().skip(1))
                .filter(|(a, b)| a == b)
                .count() as f64
                / out.len() as f64
        };
        assert!(
            agree(&right[200..]) > 0.999
                || right[200..]
                    .iter()
                    .zip(bits.iter().skip(1 + 200))
                    .all(|(a, b)| a == b),
            "right phase did not decode"
        );
        assert!(dec.recent_channel_ber() < 0.01);
        let wrong_agree = wrong
            .iter()
            .zip(bits.iter())
            .filter(|(a, b)| a == b)
            .count() as f64
            / wrong.len() as f64;
        assert!(
            wrong_agree < 0.7,
            "wrong phase should not decode ({wrong_agree})"
        );
    }

    /// Soft decisions at a noise level where the raw channel is bad: the
    /// decoder must beat the raw (uncoded) bit error rate by a wide margin,
    /// and soft input must beat hard input.
    #[test]
    fn soft_decoding_beats_hard_decoding_and_the_raw_channel() {
        let mut rng = Rng(0xABCD_0042);
        let bits: Vec<u8> = (0..60_000).map(|_| rng.bit()).collect();
        let coded = encode(&bits);
        // BPSK-like per coded bit: +-1 plus Gaussian noise; Es/N0 of the coded bit = 0.5 dB
        // (Eb/N0 = 3.5 dB at rate 1/2): raw channel ~6% errors.
        let sigma = (1.0 / (2.0 * 10f64.powf(0.05))).sqrt();
        let rx: Vec<f64> = coded
            .iter()
            .map(|&c| (if c == 0 { 1.0 } else { -1.0 }) + sigma * rng.gauss())
            .collect();
        let raw_errors = rx
            .iter()
            .zip(&coded)
            .filter(|(&r, &c)| (r < 0.0) != (c == 1))
            .count();
        let raw_ber = raw_errors as f64 / coded.len() as f64;

        let soft: Vec<f32> = rx
            .iter()
            .map(|&r| (2.0 * r / (sigma * sigma)) as f32)
            .collect();
        let hard: Vec<f32> = rx
            .iter()
            .map(|&r| if r >= 0.0 { 1.0 } else { -1.0 })
            .collect();
        let ber = |llrs: &[f32]| {
            let (out, _) = decode(llrs, false);
            out.iter().zip(&bits).filter(|(a, b)| a != b).count() as f64 / out.len() as f64
        };
        let (soft_ber, hard_ber) = (ber(&soft), ber(&hard));
        println!("raw {raw_ber:.4}, hard-decoded {hard_ber:.5}, soft-decoded {soft_ber:.5}");
        assert!(
            raw_ber > 0.05,
            "test channel too clean to mean anything: {raw_ber}"
        );
        assert!(
            soft_ber < raw_ber / 20.0,
            "soft {soft_ber} vs raw {raw_ber}"
        );
        assert!(
            soft_ber < hard_ber,
            "soft ({soft_ber}) should beat hard ({hard_ber})"
        );
    }
}
