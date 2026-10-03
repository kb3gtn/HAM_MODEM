//! TX BERT: PRBS generator with single-shot error injection.

use crate::prbs::{Lfsr, PrbsPattern};
use crate::Block;

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum TxBertControl {
    /// Flip exactly one bit of the next transmitted output, then clear itself.
    /// Does not corrupt the generator's internal sequence - later bits are
    /// unaffected.
    InjectError,
    /// Full reset: reinitialize the LFSR to its initial all-ones state,
    /// zero the telemetry counters (bits_sent, bit_errors_sent), and cancel
    /// any pending single-shot error injection.
    Reset,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct TxBertStatus {
    pub bits_sent: u64,
    pub bit_errors_sent: u64,
}

pub struct TxBert {
    lfsr: Lfsr,
    bits_sent: u64,
    bit_errors_sent: u64,
    inject_pending: bool,
}

impl TxBert {
    pub fn new(pattern: PrbsPattern) -> Self {
        TxBert {
            lfsr: Lfsr::new(pattern),
            bits_sent: 0,
            bit_errors_sent: 0,
            inject_pending: false,
        }
    }

    /// Produce the next bit of the PRBS output stream.
    pub fn next_bit(&mut self) -> u8 {
        let true_bit = self.lfsr.feedback();
        self.lfsr.shift_in(true_bit);
        self.bits_sent += 1;

        if self.inject_pending {
            self.inject_pending = false;
            self.bit_errors_sent += 1;
            true_bit ^ 1
        } else {
            true_bit
        }
    }
}

impl Block for TxBert {
    type Control = TxBertControl;
    type Status = TxBertStatus;

    fn handle_control(&mut self, cmd: TxBertControl) {
        match cmd {
            TxBertControl::InjectError => self.inject_pending = true,
            TxBertControl::Reset => {
                self.lfsr.reset();
                self.bits_sent = 0;
                self.bit_errors_sent = 0;
                self.inject_pending = false;
            }
        }
    }

    fn status(&self) -> TxBertStatus {
        TxBertStatus {
            bits_sent: self.bits_sent,
            bit_errors_sent: self.bit_errors_sent,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inject_error_flips_exactly_one_bit() {
        let mut clean = TxBert::new(PrbsPattern::Pn11);
        let mut dut = TxBert::new(PrbsPattern::Pn11);

        // advance both identically for a bit
        for _ in 0..20 {
            assert_eq!(clean.next_bit(), dut.next_bit());
        }

        dut.handle_control(TxBertControl::InjectError);
        let clean_bit = clean.next_bit();
        let dut_bit = dut.next_bit();
        assert_ne!(clean_bit, dut_bit, "injected bit should differ");

        // sequence resumes identical afterwards - internal state wasn't corrupted
        for _ in 0..50 {
            assert_eq!(clean.next_bit(), dut.next_bit());
        }

        assert_eq!(dut.status().bit_errors_sent, 1);
        assert_eq!(dut.status().bits_sent, 71);
    }

    #[test]
    fn reset_returns_to_initial_sequence() {
        let mut a = TxBert::new(PrbsPattern::Pn15);
        let mut b = TxBert::new(PrbsPattern::Pn15);

        for _ in 0..1000 {
            a.next_bit();
        }
        a.handle_control(TxBertControl::Reset);
        assert_eq!(
            a.status(),
            TxBertStatus {
                bits_sent: 0,
                bit_errors_sent: 0
            }
        );

        for _ in 0..2000 {
            assert_eq!(a.next_bit(), b.next_bit());
        }
    }
}
