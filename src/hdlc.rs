//! Bit-level HDLC framing (TX) and deframing (RX) - the physical framing
//! AX.25 rides on: flag delimiters (`0x7E`), bit-stuffing (insert a 0 after
//! every 5 consecutive 1s in the data, to keep 6-in-a-row unique to flags),
//! and the FCS (`crc::ax25_fcs`). This module is deliberately blind to
//! AX.25's own address/control/PID semantics - per project design, this
//! program is a KISS-style "dumb pipe" TNC: frame CONTENT is opaque bytes
//! handed through unexamined between the KISS host interface and the air.
//! Only framing (finding/inserting frame boundaries and validating/
//! computing the FCS) is this module's job.
//!
//! Bytes are serialized/reconstructed LSB-first per bit, matching the
//! standard AX.25/HDLC bit-transmission-order convention.
//!
//! PHASE AMBIGUITY (M-PSK; written for 8PSK, BPSK/QPSK are the same with M =
//! 2/4): carrier recovery can settle on any of M phases (multiples of 360/M
//! degrees), so the received symbol stream may be rotated by
//! 0..M-1 constellation steps, and the rotation can change after a cycle slip.
//! This is resolved here rather than via differential encoding (which would
//! roughly double the symbol error rate for no benefit). Unlike the BPSK
//! case, a rotation is NOT a simple bit inversion - Gray-labelled symbol
//! rotation scrambles the bits non-trivially - and the G3RUH descrambler
//! sits between the demapper and the deframer, so each hypothesis needs its
//! own full demap -> descramble -> `HdlcDeframer` branch.
//! `RotationResolvingDeframer` runs all 16 branches (8 rotations x 2 code-pair
//! phases, each with its own soft Viterbi decoder) in parallel on the same
//! symbol stream: only the correct rotation ever produces FCS-valid frames
//! (the 16-bit FCS is the discriminator), the others see noise-like data.
//! It locks onto the branch that delivers frames, and re-locks if a later
//! cycle slip moves the signal to a different branch.

use crate::crc::ax25_fcs;
use crate::fec_bank::{branch_pair_phase, branch_rotation, num_branches, SoftDecoderBank};
use crate::scrambler::Descrambler;
use crate::symbol_map::Modulation;
use num_complex::Complex32;

/// The HDLC flag pattern (`0x7E`) - also usable directly as continuous
/// idle-line fill between real frames (see project memory's established
/// "TX bit source must never block" pacing rule).
pub const FLAG: [u8; 8] = [0, 1, 1, 1, 1, 1, 1, 0];

/// Frames a raw AX.25 payload (address+control+PID+info - NOT including the
/// FCS) into a bit-stuffed bit stream wrapped in opening/closing flags,
/// ready to hand to a scrambler. `payload` must be non-empty.
pub fn frame_bits(payload: &[u8]) -> Vec<u8> {
    assert!(!payload.is_empty(), "cannot frame an empty payload");
    let fcs = ax25_fcs(payload);

    let mut data_bits = Vec::with_capacity((payload.len() + 2) * 8);
    for &byte in payload
        .iter()
        .chain([(fcs & 0xFF) as u8, (fcs >> 8) as u8].iter())
    {
        for i in 0..8 {
            data_bits.push((byte >> i) & 1);
        }
    }

    let mut out = Vec::with_capacity(data_bits.len() + data_bits.len() / 5 + 16);
    out.extend_from_slice(&FLAG);
    let mut ones_run = 0u32;
    for &bit in &data_bits {
        out.push(bit);
        if bit == 1 {
            ones_run += 1;
            if ones_run == 5 {
                out.push(0); // stuff bit
                ones_run = 0;
            }
        } else {
            ones_run = 0;
        }
    }
    out.extend_from_slice(&FLAG);
    out
}

/// A single-hypothesis bit-level HDLC deframer: finds flags, undoes bit
/// stuffing, reassembles bytes, and validates the FCS - assuming the bit
/// stream it's fed is already correctly demapped/descrambled. See
/// `RotationResolvingDeframer` for the phase-ambiguity-blind wrapper
/// actually meant for RX use.
pub struct HdlcDeframer {
    ones_run: u32,
    /// A single terminating 0-bit not yet committed as data. It can't be
    /// committed the moment it arrives: if six more 1s immediately follow
    /// it, this bit turns out to have been a flag's own leading edge, not
    /// real data, and must be discarded instead - a real, unavoidable
    /// one-bit-of-lookahead ambiguity in bit-synchronous HDLC (a genuine bug
    /// found via testing: an earlier version committed this bit immediately
    /// and silently corrupted byte alignment whenever real frame data ended
    /// in fewer than 5 trailing 1-bits right before the closing flag).
    pending_zero: bool,
    bit_buf: u8,
    bit_count: u8,
    frame_bytes: Vec<u8>,
    have_synced: bool,
    /// Flags that arrived with nothing between them and the previous flag:
    /// the idle line (continuous flags) as opposed to frames. On noise this
    /// is rare (about once per 2^14 bits), on an idle link once per 8.
    empty_flag_count: u64,
    /// Abort sequences (7+ ones) seen: impossible in a healthy stream (bit
    /// stuffing), once per ~256 bits in garbage.
    abort_count: u64,
    /// Count of byte-aligned, non-empty candidate frames that failed FCS -
    /// a useful RX diagnostic (garbage/noise vs. genuine silence look very
    /// different: noise tends to produce a trickle of bad frames, not zero).
    bad_frame_count: u64,
}

impl HdlcDeframer {
    pub fn new() -> Self {
        HdlcDeframer {
            ones_run: 0,
            pending_zero: false,
            bit_buf: 0,
            bit_count: 0,
            frame_bytes: Vec::new(),
            have_synced: false,
            empty_flag_count: 0,
            abort_count: 0,
            bad_frame_count: 0,
        }
    }

    /// Bad frames plus aborts: events that a healthy link almost never
    /// produces and noise produces constantly.
    pub fn garbage_event_count(&self) -> u64 {
        self.bad_frame_count + self.abort_count
    }

    pub fn bad_frame_count(&self) -> u64 {
        self.bad_frame_count
    }

    /// Back-to-back flags seen so far (see the field docs).
    pub fn empty_flag_count(&self) -> u64 {
        self.empty_flag_count
    }

    /// Feed one raw bit. Returns `Some(payload)` (FCS already stripped and
    /// verified) when a flag just terminated a complete, valid frame.
    pub fn process_bit(&mut self, bit: u8) -> Option<Vec<u8>> {
        let bit = bit & 1;
        if bit == 1 {
            self.ones_run = (self.ones_run + 1).min(7);
            return None;
        }

        // bit == 0: resolve whatever run of 1s (0..=7, saturated) preceded
        // it, using THIS bit as the terminator - but only commit that
        // terminator itself as data once we know it isn't a flag's leading
        // edge (see `pending_zero`'s doc comment).
        let run = self.ones_run;
        self.ones_run = 0;
        match run {
            0..=4 => {
                if self.pending_zero {
                    self.push_bit(0);
                }
                for _ in 0..run {
                    self.push_bit(1);
                }
                self.pending_zero = true;
                None
            }
            5 => {
                // The 5 ones were real data; this run length also confirms
                // any prior pending zero was real data too (a flag's ones
                // run is 6, not 5). This terminating 0 itself is a stuff
                // bit, discarded outright (not even held pending).
                if self.pending_zero {
                    self.push_bit(0);
                }
                for _ in 0..5 {
                    self.push_bit(1);
                }
                self.pending_zero = false;
                None
            }
            6 => {
                // Flag: the six ones, any pending zero before them (which
                // retroactively turns out to have been the flag's own
                // leading edge, not data), AND this terminating zero are
                // ALL discarded - never treated as data. Flag detection
                // itself never depended on `pending_zero` (six-ones-then-0
                // is sufficient on its own), so a following back-to-back
                // flag's own "six ones then 0" detects fine on its own
                // merits without needing this bit held pending - it doesn't
                // matter whether real encoders share this boundary bit
                // between consecutive flags or not.
                self.pending_zero = false;
                self.on_flag()
            }
            _ => {
                // 7+ ones then a 0: an abort/error condition, not a flag -
                // discard the in-progress frame and resync on the next flag.
                self.abort_count += 1;
                self.reset_frame();
                self.pending_zero = true;
                None
            }
        }
    }

    fn push_bit(&mut self, bit: u8) {
        if !self.have_synced {
            return;
        }
        self.bit_buf |= bit << self.bit_count;
        self.bit_count += 1;
        if self.bit_count == 8 {
            self.frame_bytes.push(self.bit_buf);
            self.bit_buf = 0;
            self.bit_count = 0;
        }
    }

    fn on_flag(&mut self) -> Option<Vec<u8>> {
        if self.have_synced && self.bit_count == 0 && self.frame_bytes.is_empty() {
            self.empty_flag_count += 1;
        }
        let result = if self.have_synced && self.bit_count == 0 && self.frame_bytes.len() >= 3 {
            let bytes = std::mem::take(&mut self.frame_bytes);
            let data_len = bytes.len() - 2;
            let received_fcs = u16::from(bytes[data_len]) | (u16::from(bytes[data_len + 1]) << 8);
            if ax25_fcs(&bytes[..data_len]) == received_fcs {
                Some(bytes[..data_len].to_vec())
            } else {
                self.bad_frame_count += 1;
                None
            }
        } else {
            None
        };
        self.reset_frame();
        self.have_synced = true;
        result
    }

    fn reset_frame(&mut self) {
        self.bit_buf = 0;
        self.bit_count = 0;
        self.frame_bytes.clear();
    }
}

impl Default for HdlcDeframer {
    fn default() -> Self {
        Self::new()
    }
}

/// One hypothesis: its Viterbi output is descrambled, then deframed.
struct RotationBranch {
    descrambler: Descrambler,
    deframer: HdlcDeframer,
}

/// The ambiguity-blind RX deframer: a bank of 16 parallel soft-Viterbi ->
/// descramble -> `HdlcDeframer` branches (8 rotations x 2 code-pair phases,
/// see `fec_bank`), fed the same symbol stream. See module docs for why.
///
/// Lock logic: frames from the currently locked branch are delivered
/// immediately. With no lock yet, the first valid frame from any branch is
/// delivered and locks that branch. After that, a valid frame from a
/// *different* branch is held as a "challenger"; if the very next valid frame
/// also comes from that same branch (with none from the locked branch in
/// between), the lock moves there and both are delivered - the signature of a
/// real carrier cycle slip. A lone frame from another branch is far more
/// likely a ~2^-16 FCS false positive on noise than a real frame, so it is
/// dropped when it is never confirmed. (Cost: the first frame after a real
/// re-rotation is lost if it is the only one.)
pub struct RotationResolvingDeframer {
    modulation: Modulation,
    fec: SoftDecoderBank,
    decoded: Vec<Vec<u8>>,
    branches: Vec<RotationBranch>,
    locked: Option<usize>,
    challenger: Option<(usize, Vec<u8>)>,
    /// False while the bank is stopped (see `set_enabled`).
    enabled: bool,
    /// Link-health tracking while locked: evidence (valid frames + idle
    /// flags) is counted per window of `HEALTH_WINDOW_CODED_BITS` coded bits;
    /// a window without enough evidence means the locked branch has stopped
    /// making sense (a carrier cycle slip changed the rotation, or the signal
    /// is gone), so every branch is searched again.
    health_symbols: u32,
    health_evidence: u64,
    evidence_baseline: u64,
    bad_baseline: u64,
    /// The short (garbage) window's counters.
    garbage_symbols: u32,
    garbage_evidence: u64,
    garbage_flag_baseline: u64,
    garbage_baseline: u64,
}

/// Long window: holds at least a few frames even on a saturated link of
/// maximum-length (~330 byte, ~2.7 kbit) frames: 32768 coded bits = 16 kbit.
const HEALTH_WINDOW_CODED_BITS: u32 = 32_768;
/// Evidence (valid frames + idle flags) needed per long window for the lock
/// to be kept. An idle link gives ~2000 flags, a saturated one ~6 frames;
/// garbage (the locked branch has stopped making sense) about one empty flag.
const HEALTH_MIN_EVIDENCE: u64 = 4;
/// Short window, for a quick verdict on garbage: a locked branch that is
/// decoding noise (e.g. after a carrier cycle slip) emits aborts and bad
/// frames every ~256 bits, a healthy one almost none.
const GARBAGE_WINDOW_CODED_BITS: u32 = 4_096;
/// More than `2 x evidence + this` garbage events in a short window loses the lock.
const GARBAGE_SLACK: u64 = 3;

impl RotationResolvingDeframer {
    pub fn new(modulation: Modulation) -> Self {
        RotationResolvingDeframer {
            modulation,
            fec: SoftDecoderBank::new(modulation),
            decoded: vec![Vec::new(); num_branches(modulation)],
            branches: (0..num_branches(modulation))
                .map(|_| RotationBranch {
                    descrambler: Descrambler::new(),
                    deframer: HdlcDeframer::new(),
                })
                .collect(),
            locked: None,
            challenger: None,
            enabled: true,
            health_symbols: 0,
            health_evidence: 0,
            evidence_baseline: 0,
            bad_baseline: 0,
            garbage_symbols: 0,
            garbage_evidence: 0,
            garbage_flag_baseline: 0,
            garbage_baseline: 0,
        }
    }

    /// Stop (or restart) the whole bank: used when another consumer (the PRBS
    /// checker) has the link, so 2-16 Viterbi decoders need not run. Stopping
    /// forgets the lock; restarting searches every branch afresh.
    pub fn set_enabled(&mut self, enabled: bool) {
        if enabled == self.enabled {
            return;
        }
        self.enabled = enabled;
        self.locked = None;
        self.challenger = None;
        if enabled {
            self.search_all_branches();
        } else {
            self.fec.deactivate_all();
        }
    }

    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Run every branch again from a clean state.
    fn search_all_branches(&mut self) {
        for b in self.fec.activate_all() {
            self.branches[b] = RotationBranch {
                descrambler: Descrambler::new(),
                deframer: HdlcDeframer::new(),
            };
        }
        self.health_symbols = 0;
        self.health_evidence = 0;
        self.garbage_symbols = 0;
        self.garbage_evidence = 0;
    }

    /// Feed one received (carrier-derotated) complex symbol. Returns any
    /// frames delivered by this symbol (normally zero or one).
    pub fn process_symbol(&mut self, y: Complex32) -> Vec<Vec<u8>> {
        if !self.enabled {
            return Vec::new();
        }
        for d in &mut self.decoded {
            d.clear();
        }
        self.fec.process_symbol(y, &mut self.decoded);

        let mut found: Vec<(usize, Vec<u8>)> = Vec::new();
        for (branch, (state, bits)) in self.branches.iter_mut().zip(&self.decoded).enumerate() {
            for &bit in bits {
                let descrambled = state.descrambler.descramble_bit(bit);
                if let Some(frame) = state.deframer.process_bit(descrambled) {
                    found.push((branch, frame));
                }
            }
        }

        let mut delivered = Vec::new();
        for (branch, frame) in found {
            match self.locked {
                None => {
                    self.locked = Some(branch);
                    delivered.push(frame);
                }
                Some(l) if l == branch => {
                    self.challenger = None;
                    delivered.push(frame);
                }
                Some(_) => match self.challenger.take() {
                    Some((cb, held)) if cb == branch => {
                        self.locked = Some(branch);
                        delivered.push(held);
                        delivered.push(frame);
                    }
                    _ => self.challenger = Some((branch, frame)),
                },
            }
        }
        self.track_lock(delivered.len() as u64);
        delivered
    }

    /// Narrow the bank to the locked branch, and widen it again if that
    /// branch stops delivering evidence of a healthy link.
    fn track_lock(&mut self, frames_delivered: u64) {
        let Some(locked) = self.locked else {
            return;
        };
        if !self.fec.is_restricted() {
            self.fec.restrict_to(locked);
            self.health_symbols = 0;
            self.health_evidence = 0;
            self.evidence_baseline = self.branches[locked].deframer.empty_flag_count();
            self.bad_baseline = self.branches[locked].deframer.bad_frame_count();
            self.garbage_symbols = 0;
            self.garbage_evidence = 0;
            self.garbage_flag_baseline = self.evidence_baseline;
            self.garbage_baseline = self.branches[locked].deframer.garbage_event_count();
        }
        self.health_evidence += frames_delivered;
        self.health_symbols += 1;
        self.garbage_evidence += frames_delivered;
        self.garbage_symbols += 1;
        let bits_per_symbol = self.modulation.bits_per_symbol() as u32;
        if self.garbage_symbols * bits_per_symbol >= GARBAGE_WINDOW_CODED_BITS {
            let deframer = &self.branches[locked].deframer;
            let evidence =
                self.garbage_evidence + (deframer.empty_flag_count() - self.garbage_flag_baseline);
            let garbage = deframer.garbage_event_count() - self.garbage_baseline;
            if garbage > 2 * evidence + GARBAGE_SLACK {
                self.locked = None;
                self.challenger = None;
                self.search_all_branches();
                return;
            }
            self.garbage_symbols = 0;
            self.garbage_evidence = 0;
            self.garbage_flag_baseline = deframer.empty_flag_count();
            self.garbage_baseline = deframer.garbage_event_count();
        }
        if self.health_symbols * bits_per_symbol >= HEALTH_WINDOW_CODED_BITS {
            let flags = self.branches[locked].deframer.empty_flag_count();
            let bad = self.branches[locked].deframer.bad_frame_count();
            let evidence = self.health_evidence + (flags - self.evidence_baseline);
            if evidence < HEALTH_MIN_EVIDENCE || bad - self.bad_baseline > evidence {
                self.locked = None;
                self.challenger = None;
                self.search_all_branches();
            } else {
                self.health_symbols = 0;
                self.health_evidence = 0;
                self.evidence_baseline = flags;
                self.bad_baseline = bad;
            }
        }
    }

    /// The rotation (0..M, 360/M-degree steps) currently locked onto, if any.
    pub fn locked_rotation(&self) -> Option<u8> {
        self.locked.map(|b| branch_rotation(self.modulation, b))
    }

    /// The code-pair phase (0/1) currently locked onto, if any.
    pub fn locked_pair_phase(&self) -> Option<u8> {
        self.locked.map(|b| branch_pair_phase(self.modulation, b))
    }

    /// Raw (pre-FEC) channel BER seen by the locked branch's decoder.
    pub fn pre_fec_ber(&self) -> Option<f64> {
        self.locked.map(|b| self.fec.recent_channel_ber(b))
    }

    /// FCS-failure count of the locked branch (0 before lock). Per-branch,
    /// not summed: the fifteen wrong branches see noise and would swamp a
    /// meaningful count from the real one.
    pub fn bad_frame_count(&self) -> u64 {
        self.locked
            .map_or(0, |b| self.branches[b].deframer.bad_frame_count())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn feed(deframer: &mut HdlcDeframer, bits: &[u8]) -> Vec<Vec<u8>> {
        bits.iter()
            .filter_map(|&b| deframer.process_bit(b))
            .collect()
    }

    #[test]
    fn frames_and_deframes_a_payload_round_trip() {
        let payload = b"hello AX.25-ish world".to_vec();
        let bits = frame_bits(&payload);

        let mut deframer = HdlcDeframer::new();
        let frames = feed(&mut deframer, &bits);

        assert_eq!(frames, vec![payload]);
    }

    #[test]
    fn deframes_back_to_back_frames_each_fully_flagged() {
        let payload_a = b"first frame".to_vec();
        let payload_b = b"second frame, different length".to_vec();

        // Two independently, fully-flagged frames concatenated directly -
        // flag detection (six 1s then a 0) doesn't depend on a shared
        // boundary bit between adjacent flags, so this is exactly as valid
        // as an encoder that optimizes away the redundant middle flag.
        let mut bits = frame_bits(&payload_a);
        bits.extend_from_slice(&frame_bits(&payload_b));

        let mut deframer = HdlcDeframer::new();
        let frames = feed(&mut deframer, &bits);

        assert_eq!(frames, vec![payload_a, payload_b]);
    }

    #[test]
    fn a_corrupted_frame_fails_fcs_and_is_dropped() {
        let payload = b"this frame will get corrupted".to_vec();
        let mut bits = frame_bits(&payload);
        // Flip a bit safely inside the frame body (well past the opening
        // flag, well before the closing one).
        let flip_idx = 20;
        bits[flip_idx] ^= 1;

        let mut deframer = HdlcDeframer::new();
        let frames = feed(&mut deframer, &bits);

        assert!(
            frames.is_empty(),
            "a corrupted frame should fail FCS and be dropped, got {frames:?}"
        );
        assert_eq!(
            deframer.bad_frame_count(),
            1,
            "the FCS failure should have been counted"
        );
    }

    #[test]
    fn idle_flags_between_frames_produce_no_spurious_frames() {
        let payload = b"real frame".to_vec();
        let mut bits = Vec::new();
        for _ in 0..10 {
            bits.extend_from_slice(&FLAG); // idle-line fill
        }
        bits.extend_from_slice(&frame_bits(&payload));
        for _ in 0..10 {
            bits.extend_from_slice(&FLAG);
        }

        let mut deframer = HdlcDeframer::new();
        let frames = feed(&mut deframer, &bits);

        assert_eq!(frames, vec![payload]);
    }

    #[test]
    fn deframes_frames_separated_by_a_boundary_sharing_flag_sequence() {
        // The more bandwidth-efficient real-world idle-fill convention:
        // consecutive flags share their boundary 0 (01111110 repeated at
        // period 7, not 8: ...0111111001111110...). Flag detection here
        // never actually depends on a distinct leading zero being present
        // (six-1s-then-0 is sufficient on its own), so this should decode
        // identically to fully-separated flags.
        let payload_a = b"one".to_vec();
        let payload_b = b"two".to_vec();

        let mut bits = frame_bits(&payload_a);
        bits.truncate(bits.len() - 8); // drop a's whole trailing flag
        bits.extend_from_slice(&FLAG); // exactly one shared flag between them
        bits.extend_from_slice(&frame_bits(&payload_b)[8..]); // skip b's leading flag

        let mut deframer = HdlcDeframer::new();
        let frames = feed(&mut deframer, &bits);

        assert_eq!(frames, vec![payload_a, payload_b]);
    }

    use crate::conv::ConvEncoder;
    use crate::scrambler::Scrambler;
    use crate::symbol_map::{psk8_map, psk8_point};

    /// What the TX does: scramble, convolutionally encode (2 coded bits per
    /// data bit), pack 3 coded bits per 8PSK symbol. Data is zero-padded so
    /// the coded stream fills whole symbols.
    fn to_symbols(bits: &[u8]) -> Vec<Complex32> {
        let mut data = bits.to_vec();
        while data.len() % 3 != 0 {
            data.push(0);
        }
        let mut scrambler = Scrambler::new();
        let mut encoder = ConvEncoder::new();
        let coded: Vec<u8> = data
            .iter()
            .flat_map(|&b| encoder.encode_bit(scrambler.scramble_bit(b)))
            .collect();
        coded
            .chunks(3)
            .map(|c| psk8_map([c[0], c[1], c[2]]))
            .collect()
    }

    fn idle(n_flags: usize) -> Vec<u8> {
        FLAG.iter().cycle().take(n_flags * 8).copied().collect()
    }

    /// Frames separated by idle flags, with a long idle tail: the Viterbi
    /// decoder releases bits ~64-128 bits late, so the last frame needs
    /// following bits to be flushed out.
    fn stream(frames: &[&[u8]]) -> Vec<u8> {
        let mut bits = idle(8);
        for f in frames {
            bits.extend(frame_bits(f));
            bits.extend(idle(2));
        }
        bits.extend(idle(24));
        bits
    }

    fn feed_symbols(
        d: &mut RotationResolvingDeframer,
        symbols: &[Complex32],
        rotation: u8,
    ) -> Vec<Vec<u8>> {
        let rot = psk8_point(rotation);
        symbols
            .iter()
            .flat_map(|&y| d.process_symbol(y * rot))
            .collect()
    }

    #[test]
    fn rotation_resolver_recovers_frames_under_every_rotation() {
        for rotation in 0..8u8 {
            let frames: Vec<&[u8]> = vec![
                b"first frame under rotation",
                b"second one, a little longer than the first",
            ];
            let symbols = to_symbols(&stream(&frames));
            let mut d = RotationResolvingDeframer::new(Modulation::Psk8);
            let got = feed_symbols(&mut d, &symbols, rotation);
            assert_eq!(
                got,
                frames.iter().map(|f| f.to_vec()).collect::<Vec<_>>(),
                "rotation {rotation}"
            );
            assert_eq!(d.locked_rotation(), Some(rotation));
        }
    }

    #[test]
    fn rotation_resolver_handles_a_receiver_that_joins_mid_stream() {
        // Dropping symbols from the front changes where the code pairs fall
        // relative to the receiver's first symbol: exercises the pair-phase
        // hypotheses, not just the rotations.
        let frames: Vec<&[u8]> = vec![b"a frame that arrives after the receiver joined"];
        let mut bits = stream(&frames);
        bits.extend(stream(&frames));
        let symbols = to_symbols(&bits);
        let mut phases = std::collections::BTreeSet::new();
        for skip in 0..4 {
            let mut d = RotationResolvingDeframer::new(Modulation::Psk8);
            let got = feed_symbols(&mut d, &symbols[skip..], 3);
            assert!(!got.is_empty(), "skip {skip}: nothing decoded");
            assert!(got.iter().all(|f| f == frames[0]), "skip {skip}: {got:?}");
            phases.insert(d.locked_pair_phase().unwrap());
        }
        assert_eq!(
            phases.len(),
            2,
            "expected both pair phases to be needed across different join points"
        );
    }

    #[test]
    fn rotation_resolver_relocks_after_a_mid_stream_rotation_change() {
        // Simulates a carrier-loop cycle slip: rotation 2 for a while, then 5.
        let frames_a: Vec<&[u8]> = vec![b"before the slip, one", b"before the slip, two"];
        let frames_b: Vec<&[u8]> = vec![
            b"after the slip, one",
            b"after the slip, two",
            b"after the slip, three",
        ];

        // One continuous scrambled/coded stream, so encoder and scrambler
        // state carry over exactly as on the air; only the rotation changes.
        // Once locked the bank decodes only the locked branch, so after the
        // slip it must notice (the locked branch now decodes garbage: aborts
        // and bad frames) and search every branch again. That takes a short
        // window of a few hundred bits, during which frames are lost.
        let mut bits = stream(&frames_a);
        let split_bits = bits.len();
        bits.extend(idle(600)); // ~4.8 kbit of idle line after the slip
        bits.extend(stream(&frames_b));
        let symbols = to_symbols(&bits);
        let split = split_bits * 2 / 3; // symbol index of the change (2 coded bits per data bit, 3 per symbol)

        let mut d = RotationResolvingDeframer::new(Modulation::Psk8);
        let mut got = feed_symbols(&mut d, &symbols[..split], 2);
        assert_eq!(d.locked_rotation(), Some(2));
        got.extend(feed_symbols(&mut d, &symbols[split..], 5));

        let expect_a: Vec<Vec<u8>> = frames_a.iter().map(|f| f.to_vec()).collect();
        assert_eq!(&got[..2], &expect_a[..], "pre-slip frames");
        assert!(
            got.len() >= 4,
            "expected re-lock and post-slip frames, got {got:?}"
        );
        assert_eq!(got.last().unwrap(), &frames_b.last().unwrap().to_vec());
        assert_eq!(d.locked_rotation(), Some(5));
    }

    #[test]
    fn noise_does_not_produce_frames_in_practice() {
        // Random symbols: sixteen branches of FCS-checked garbage. Expect
        // essentially no deliveries (each is a ~2^-16 event per candidate).
        let mut s: u32 = 0x9E37_79B9;
        let symbols: Vec<Complex32> = (0..60_000)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 17;
                s ^= s << 5;
                psk8_point(((s >> 8) & 7) as u8)
            })
            .collect();
        let mut d = RotationResolvingDeframer::new(Modulation::Psk8);
        let got = feed_symbols(&mut d, &symbols, 0);
        assert!(
            got.len() <= 1,
            "unexpected number of false frames from noise: {}",
            got.len()
        );
    }
}
