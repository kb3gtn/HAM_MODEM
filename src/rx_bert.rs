//! RX BERT: self-synchronizing PRBS checker with a Search/Synced lock state
//! machine and a latching sync-loss alarm.

use std::collections::VecDeque;

use crate::prbs::{Lfsr, PrbsPattern};
use crate::Block;

/// Consecutive correct feedback-vs-input matches required to declare lock
/// while in Search state.
const SEARCH_LOCK_STREAK: u32 = 100;
/// Sliding window size (bits) used to evaluate loss of sync while Synced.
const ERROR_WINDOW: usize = 1000;
/// If the error count within the last ERROR_WINDOW bits exceeds this while
/// Synced, drop back to Search.
const ERROR_WINDOW_THRESHOLD: u32 = 300;

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum RxBertControl {
    /// Full reset: reinitialize the LFSR to its initial state, force the
    /// checker back into Search state, zero the telemetry counters
    /// (bits_received, bit_errors_received), and clear the latched
    /// sync-loss flag.
    Reset,
    /// Zero the telemetry counters (bits_received, bit_errors_received) and
    /// clear the latched sync-loss flag if set. Does NOT touch the LFSR or
    /// current lock state - if currently Synced, stays Synced and keeps
    /// tracking without re-acquiring.
    ResetStats,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum LockState {
    Search,
    Synced,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct RxBertStatus {
    pub bits_received: u64,
    pub bit_errors_received: u64,
    pub locked_state: LockState,
    pub sync_loss: bool,
}

pub struct RxBert {
    lfsr: Lfsr,
    state: LockState,
    search_streak: u32,
    error_window: VecDeque<bool>,
    errors_in_window: u32,
    bits_received: u64,
    bit_errors_received: u64,
    sync_loss_latched: bool,
}

impl RxBert {
    pub fn new(pattern: PrbsPattern) -> Self {
        RxBert {
            lfsr: Lfsr::new(pattern),
            state: LockState::Search,
            search_streak: 0,
            error_window: VecDeque::with_capacity(ERROR_WINDOW),
            errors_in_window: 0,
            bits_received: 0,
            bit_errors_received: 0,
            sync_loss_latched: false,
        }
    }

    /// Back to Search from scratch (register and lock state), keeping the
    /// telemetry counters and the sync-loss latch: used when the decoder
    /// feeding this checker was restarted.
    pub fn restart_search(&mut self) {
        self.lfsr.reset();
        self.state = LockState::Search;
        self.search_streak = 0;
        self.error_window.clear();
        self.errors_in_window = 0;
    }

    pub fn is_synced(&self) -> bool {
        self.state == LockState::Synced
    }

    /// Clock in one received bit.
    pub fn process_bit(&mut self, actual: u8) {
        let actual = actual & 1;
        self.bits_received += 1;
        // Predicted bit from the register's CURRENT contents, before update.
        let predicted = self.lfsr.feedback();
        let is_error = predicted != actual;

        match self.state {
            LockState::Search => {
                // Track the raw input directly while searching for alignment.
                self.lfsr.shift_in(actual);
                if is_error {
                    self.search_streak = 0;
                } else {
                    self.search_streak += 1;
                    // Never lock onto the all-zero register: it's the one
                    // degenerate fixed point of the LFSR and never occurs in
                    // a genuine PRBS sequence, so seeing it means we're
                    // tracking a constant/degenerate input, not real alignment.
                    if self.search_streak > SEARCH_LOCK_STREAK && !self.lfsr.is_zero() {
                        self.enter_synced();
                    }
                }
            }
            LockState::Synced => {
                // Free-run on our own feedback; compare against actual input
                // for error counting only.
                self.lfsr.shift_in(predicted);
                self.push_window_result(is_error);
                if is_error {
                    self.bit_errors_received += 1;
                }
                if self.errors_in_window > ERROR_WINDOW_THRESHOLD {
                    self.enter_search();
                }
            }
        }
    }

    fn enter_synced(&mut self) {
        self.state = LockState::Synced;
        self.search_streak = 0;
        self.error_window.clear();
        self.errors_in_window = 0;
    }

    fn enter_search(&mut self) {
        self.state = LockState::Search;
        self.search_streak = 0;
        self.error_window.clear();
        self.errors_in_window = 0;
        self.sync_loss_latched = true;
    }

    fn push_window_result(&mut self, is_error: bool) {
        self.error_window.push_back(is_error);
        if is_error {
            self.errors_in_window += 1;
        }
        if self.error_window.len() > ERROR_WINDOW {
            if let Some(oldest) = self.error_window.pop_front() {
                if oldest {
                    self.errors_in_window -= 1;
                }
            }
        }
    }
}

impl Block for RxBert {
    type Control = RxBertControl;
    type Status = RxBertStatus;

    fn handle_control(&mut self, cmd: RxBertControl) {
        match cmd {
            RxBertControl::Reset => {
                self.lfsr.reset();
                self.state = LockState::Search;
                self.search_streak = 0;
                self.error_window.clear();
                self.errors_in_window = 0;
                self.sync_loss_latched = false;
                self.bits_received = 0;
                self.bit_errors_received = 0;
            }
            RxBertControl::ResetStats => {
                self.bits_received = 0;
                self.bit_errors_received = 0;
                self.sync_loss_latched = false;
            }
        }
    }

    fn status(&self) -> RxBertStatus {
        RxBertStatus {
            bits_received: self.bits_received,
            bit_errors_received: self.bit_errors_received,
            locked_state: self.state,
            sync_loss: self.sync_loss_latched,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tx_bert::{TxBert, TxBertControl};

    #[test]
    fn locks_within_expected_bits_and_counts_zero_errors_on_clean_link() {
        let mut tx = TxBert::new(PrbsPattern::Pn15);
        let mut rx = RxBert::new(PrbsPattern::Pn15);

        for _ in 0..5000 {
            rx.process_bit(tx.next_bit());
        }

        let s = rx.status();
        assert_eq!(s.locked_state, LockState::Synced);
        assert_eq!(s.bit_errors_received, 0);
        assert!(!s.sync_loss);
        assert_eq!(s.bits_received, 5000);
    }

    #[test]
    fn injected_error_is_counted_once_locked() {
        let mut tx = TxBert::new(PrbsPattern::Pn11);
        let mut rx = RxBert::new(PrbsPattern::Pn11);

        for _ in 0..500 {
            rx.process_bit(tx.next_bit());
        }
        assert_eq!(rx.status().locked_state, LockState::Synced);

        tx.handle_control(TxBertControl::InjectError);
        rx.process_bit(tx.next_bit());

        assert_eq!(rx.status().bit_errors_received, 1);
        assert_eq!(tx.status().bit_errors_sent, 1);

        // link stays locked and clean afterwards
        for _ in 0..500 {
            rx.process_bit(tx.next_bit());
        }
        assert_eq!(rx.status().bit_errors_received, 1);
        assert_eq!(rx.status().locked_state, LockState::Synced);
        assert!(!rx.status().sync_loss);
    }

    #[test]
    fn loses_sync_on_sustained_errors_and_latches_until_reset() {
        let mut tx = TxBert::new(PrbsPattern::Pn11);
        let mut rx = RxBert::new(PrbsPattern::Pn11);

        for _ in 0..500 {
            rx.process_bit(tx.next_bit());
        }
        assert_eq!(rx.status().locked_state, LockState::Synced);

        // Feed uncorrelated bits to blow the 300-in-1000 error budget.
        for _ in 0..1000 {
            rx.process_bit(0);
        }

        let s = rx.status();
        assert_eq!(s.locked_state, LockState::Search);
        assert!(s.sync_loss);

        // Latch persists even if it re-acquires lock on the (still-running) tx sequence.
        for _ in 0..500 {
            rx.process_bit(tx.next_bit());
        }
        assert!(rx.status().sync_loss);

        rx.handle_control(RxBertControl::Reset);
        let s = rx.status();
        assert!(!s.sync_loss);
        assert_eq!(s.locked_state, LockState::Search);
        assert_eq!(s.bits_received, 0);
        assert_eq!(s.bit_errors_received, 0);
    }

    #[test]
    fn reset_stats_clears_counters_and_latch_without_dropping_lock() {
        let mut tx = TxBert::new(PrbsPattern::Pn11);
        let mut rx = RxBert::new(PrbsPattern::Pn11);

        for _ in 0..500 {
            rx.process_bit(tx.next_bit());
        }
        assert_eq!(rx.status().locked_state, LockState::Synced);

        tx.handle_control(TxBertControl::InjectError);
        rx.process_bit(tx.next_bit());
        assert_eq!(rx.status().bit_errors_received, 1);

        rx.handle_control(RxBertControl::ResetStats);
        let s = rx.status();
        assert_eq!(s.bits_received, 0);
        assert_eq!(s.bit_errors_received, 0);
        assert!(!s.sync_loss);
        // Lock state is untouched - still Synced, no re-acquisition needed.
        assert_eq!(s.locked_state, LockState::Synced);

        // Keeps tracking correctly afterwards.
        for _ in 0..500 {
            rx.process_bit(tx.next_bit());
        }
        assert_eq!(rx.status().bit_errors_received, 0);
        assert_eq!(rx.status().locked_state, LockState::Synced);
    }

    #[test]
    fn reset_stats_clears_latch_set_by_a_prior_sync_loss() {
        let mut tx = TxBert::new(PrbsPattern::Pn11);
        let mut rx = RxBert::new(PrbsPattern::Pn11);

        for _ in 0..500 {
            rx.process_bit(tx.next_bit());
        }
        for _ in 0..1000 {
            rx.process_bit(0);
        }
        assert!(rx.status().sync_loss);

        rx.handle_control(RxBertControl::ResetStats);
        assert!(!rx.status().sync_loss);
    }

    /// Regression test: feeding a constant 0 bit stream must never produce a
    /// false lock. Continuously shifting 0s into the register during Search
    /// drives it to the all-zero state after `length` bits, at which point
    /// feedback() trivially predicts 0 forever, matching the constant input.
    /// Without the is_zero() guard, this was observed to falsely enter
    /// Synced and then report zero errors indefinitely.
    #[test]
    fn constant_zero_input_never_falsely_locks() {
        let mut rx = RxBert::new(PrbsPattern::Pn11);
        for _ in 0..10_000 {
            rx.process_bit(0);
        }
        assert_eq!(rx.status().locked_state, LockState::Search);
    }
}
