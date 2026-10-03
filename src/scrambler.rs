//! G3RUH self-synchronizing scrambler/descrambler: the standard whitening
//! polynomial used by the K9NG/G3RUH 9600-baud AX.25 packet radio stack,
//! `G(x) = x^17 + x^12 + 1`. Chosen (per project decision) over NRZI/
//! differential encoding: the 8PSK carrier-phase ambiguity (8 rotations) is
//! instead handled entirely at the receiver in `hdlc.rs`'s
//! `RotationResolvingDeframer` (one descrambler + deframer per rotation
//! hypothesis), at zero error-rate cost. This scrambler's only job is
//! whitening/guaranteeing symbol transition density for timing recovery.
//!
//! "Self-synchronizing" (a.k.a. multiplicative) means the descrambler needs
//! no shared initial state with the scrambler: its own shift register is
//! fed directly from the received (still-scrambled) bit stream, so it
//! naturally produces correct output once 17 real bits have passed through
//! it, with no explicit synchronization handshake. The trade-off is limited
//! error multiplication: one bit error in the channel causes exactly 3
//! errors after descrambling (the direct hit plus the two tap positions it
//! later passes through) - a well-known, accepted property of this class of
//! scrambler, not a bug.

const TAP_A: u32 = 17; // x^17
const TAP_B: u32 = 12; // x^12
const REG_MASK: u32 = (1 << TAP_A) - 1; // 17-bit register

pub struct Scrambler {
    reg: u32,
}

impl Scrambler {
    pub fn new() -> Self {
        Scrambler { reg: 0 }
    }

    /// Scramble one bit. The shift register is fed by the scrambler's OWN
    /// output (not the input) - the defining property of this
    /// self-synchronizing construction.
    pub fn scramble_bit(&mut self, bit: u8) -> u8 {
        let fb = (((self.reg >> (TAP_A - 1)) ^ (self.reg >> (TAP_B - 1))) & 1) as u8;
        let out = (bit & 1) ^ fb;
        self.reg = ((self.reg << 1) | u32::from(out)) & REG_MASK;
        out
    }

    pub fn scramble(&mut self, bits: &[u8]) -> Vec<u8> {
        bits.iter().map(|&b| self.scramble_bit(b)).collect()
    }
}

impl Default for Scrambler {
    fn default() -> Self {
        Self::new()
    }
}

pub struct Descrambler {
    reg: u32,
}

impl Descrambler {
    pub fn new() -> Self {
        Descrambler { reg: 0 }
    }

    /// Start with an arbitrary register state rather than 0 - exists mainly
    /// to demonstrate/test the self-synchronizing property (no shared
    /// initial state with the scrambler is required; any starting state
    /// converges once 17 real received bits have passed through).
    pub fn new_with_state(reg: u32) -> Self {
        Descrambler {
            reg: reg & REG_MASK,
        }
    }

    /// Descramble one bit. The shift register is fed by the RECEIVED
    /// (scrambled) bit - the mirror image of `Scrambler::scramble_bit` - so
    /// this is stateless with respect to the scrambler's own state, only
    /// needing 17 bits of the incoming stream to produce correct output.
    pub fn descramble_bit(&mut self, bit: u8) -> u8 {
        let bit = bit & 1;
        let fb = (((self.reg >> (TAP_A - 1)) ^ (self.reg >> (TAP_B - 1))) & 1) as u8;
        let out = bit ^ fb;
        self.reg = ((self.reg << 1) | u32::from(bit)) & REG_MASK;
        out
    }

    pub fn descramble(&mut self, bits: &[u8]) -> Vec<u8> {
        bits.iter().map(|&b| self.descramble_bit(b)).collect()
    }
}

impl Default for Descrambler {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn random_bits(n: usize, seed: u32) -> Vec<u8> {
        let mut lfsr_state = seed;
        (0..n)
            .map(|_| {
                lfsr_state ^= lfsr_state << 13;
                lfsr_state ^= lfsr_state >> 17;
                lfsr_state ^= lfsr_state << 5;
                (lfsr_state & 1) as u8
            })
            .collect()
    }

    #[test]
    fn scramble_then_descramble_recovers_original_bits() {
        let bits = random_bits(5000, 0xDEAD_BEEF);
        let mut scrambler = Scrambler::new();
        let mut descrambler = Descrambler::new();

        let scrambled = scrambler.scramble(&bits);
        let recovered = descrambler.descramble(&scrambled);

        assert_eq!(recovered, bits);
    }

    #[test]
    fn scrambling_a_constant_stream_still_produces_transitions() {
        // The whole point of the scrambler: a long run of a single value
        // (the worst case for timing recovery/DC balance) should come out
        // looking like a pseudorandom, transition-rich sequence.
        let bits = vec![1u8; 2000];
        let mut scrambler = Scrambler::new();
        let scrambled = scrambler.scramble(&bits);

        let transitions = scrambled.windows(2).filter(|w| w[0] != w[1]).count();
        // A real random sequence transitions ~50% of the time; just check
        // it's not degenerate (e.g. still constant or trivially periodic
        // with a very short period) - well above a tiny lower bound is
        // enough to confirm whitening is happening at all.
        assert!(transitions > 500, "only {transitions} transitions in 2000 bits - scrambler isn't whitening a constant stream");
    }

    #[test]
    fn single_bit_error_causes_exactly_three_output_errors() {
        // Documented, expected error-multiplication property of this class
        // of self-synchronizing scrambler - a regression check that nothing
        // about the tap positions/construction has quietly changed that.
        let bits = random_bits(200, 0x1234_5678);
        let mut scrambler = Scrambler::new();
        let scrambled = scrambler.scramble(&bits);

        let mut corrupted = scrambled.clone();
        let error_pos = 100;
        corrupted[error_pos] ^= 1;

        let mut descrambler = Descrambler::new();
        let recovered = descrambler.descramble(&corrupted);

        let error_count = recovered
            .iter()
            .zip(bits.iter())
            .filter(|(a, b)| a != b)
            .count();
        assert_eq!(
            error_count, 3,
            "expected exactly 3 output bit errors from 1 input error, got {error_count}"
        );
    }

    #[test]
    fn descrambler_syncs_without_shared_initial_state() {
        // The self-synchronizing property: a descrambler started completely
        // independently (no shared seed/state with the scrambler) should
        // still recover the correct data after the register has filled with
        // real received bits (17 bits).
        let bits = random_bits(1000, 0xC0FF_EE11);
        let mut scrambler = Scrambler::new();
        let scrambled = scrambler.scramble(&bits);

        // Deliberately mismatched initial state vs. the scrambler's (which
        // started at 0) - the self-sync property means this shouldn't matter.
        let mut descrambler = Descrambler::new_with_state(0x1FFFF);
        let recovered = descrambler.descramble(&scrambled);

        // Skip the first 17 bits (register fill time) - everything after
        // must match exactly regardless of the mismatched starting state.
        assert_eq!(recovered[17..], bits[17..]);
    }
}
