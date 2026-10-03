//! PRBS patterns and the shared Fibonacci LFSR used by both the TX and RX BERT.
//!
//! Taps are the user's ITU-convention polynomials:
//!   PN11: 1 + x^9  + x^11
//!   PN15: 1 + x^14 + x^15
//!   PN23: 1 + x^18 + x^23
//! all with an all-ones initial register state.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrbsPattern {
    Pn11,
    Pn15,
    Pn23,
}

impl PrbsPattern {
    /// (register length in bits, tap position) for the recurrence
    /// s[k] = s[k-tap] ^ s[k-length].
    pub const fn params(self) -> (u32, u32) {
        match self {
            PrbsPattern::Pn11 => (11, 9),
            PrbsPattern::Pn15 => (15, 14),
            PrbsPattern::Pn23 => (23, 18),
        }
    }
}

/// A Fibonacci LFSR implementing s[k] = s[k-tap] ^ s[k-length].
///
/// Bit 0 (LSB) of `register` holds the most recently shifted-in bit (delay 1);
/// bit `length-1` holds the oldest bit in the register (delay `length`).
#[derive(Debug, Clone)]
pub struct Lfsr {
    length: u32,
    tap: u32,
    mask: u32,
    register: u32,
}

impl Lfsr {
    pub fn new(pattern: PrbsPattern) -> Self {
        let (length, tap) = pattern.params();
        let mask = (1u32 << length) - 1;
        Lfsr {
            length,
            tap,
            mask,
            register: mask, // all-ones init
        }
    }

    /// Reset the register to its initial (all-ones) state.
    pub fn reset(&mut self) {
        self.register = self.mask;
    }

    /// The feedback bit predicted from the CURRENT register contents.
    /// Does not modify the register.
    pub fn feedback(&self) -> u8 {
        let a = (self.register >> (self.tap - 1)) & 1;
        let b = (self.register >> (self.length - 1)) & 1;
        (a ^ b) as u8
    }

    /// Shift `bit` into the register as the new delay-1 bit.
    pub fn shift_in(&mut self, bit: u8) {
        self.register = ((self.register << 1) | (bit as u32 & 1)) & self.mask;
    }

    /// True if the register is in the all-zero state. This is the one
    /// degenerate fixed point of any XOR-tap Fibonacci LFSR (feedback of an
    /// all-zero register is always 0 ^ 0 = 0), and it never occurs in a
    /// genuine maximal-length PRBS sequence - only the 2^n - 1 nonzero
    /// states do. Seeing it while tracking a receive stream means the input
    /// isn't a real instance of the sequence (e.g. a constant/degenerate
    /// pattern), not that alignment was found.
    pub fn is_zero(&self) -> bool {
        self.register == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pn11_period_is_2047() {
        // A correctly-built maximal-length LFSR must return to its initial
        // state after exactly 2^n - 1 generated bits.
        let mut lfsr = Lfsr::new(PrbsPattern::Pn11);
        let initial = lfsr.register;
        for _ in 0..(2047 - 1) {
            let b = lfsr.feedback();
            lfsr.shift_in(b);
            assert_ne!(
                lfsr.register, initial,
                "returned to initial state too early"
            );
        }
        let b = lfsr.feedback();
        lfsr.shift_in(b);
        assert_eq!(
            lfsr.register, initial,
            "did not return to initial state at period 2047"
        );
    }

    #[test]
    fn pn15_period_is_32767() {
        let mut lfsr = Lfsr::new(PrbsPattern::Pn15);
        let initial = lfsr.register;
        for _ in 0..(32767 - 1) {
            let b = lfsr.feedback();
            lfsr.shift_in(b);
        }
        let b = lfsr.feedback();
        lfsr.shift_in(b);
        assert_eq!(lfsr.register, initial);
    }
}
