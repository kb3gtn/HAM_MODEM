//! Black-box TX BERT <-> RX BERT testbench: connects the two through a
//! trivial ideal channel (bit out of TX fed directly into RX) and exercises
//! locking, error injection/counting, and sync-loss/latch/reset behavior
//! end-to-end through the public API only.

use ham_modem::prbs::PrbsPattern;
use ham_modem::rx_bert::{LockState, RxBert, RxBertControl};
use ham_modem::tx_bert::{TxBert, TxBertControl};
use ham_modem::Block;

fn run_clean_bits(tx: &mut TxBert, rx: &mut RxBert, n: u64) {
    for _ in 0..n {
        rx.process_bit(tx.next_bit());
    }
}

#[test]
fn full_bert_session_pn23() {
    let mut tx = TxBert::new(PrbsPattern::Pn23);
    let mut rx = RxBert::new(PrbsPattern::Pn23);

    // Acquire lock.
    run_clean_bits(&mut tx, &mut rx, 200);
    assert_eq!(rx.status().locked_state, LockState::Synced);
    assert_eq!(rx.status().bit_errors_received, 0);

    // Run a clean stretch and inject a few isolated errors.
    run_clean_bits(&mut tx, &mut rx, 10_000);
    for _ in 0..5 {
        tx.handle_control(TxBertControl::InjectError);
        run_clean_bits(&mut tx, &mut rx, 1);
        run_clean_bits(&mut tx, &mut rx, 1_000);
    }

    let s = rx.status();
    assert_eq!(s.locked_state, LockState::Synced);
    assert_eq!(s.bit_errors_received, 5);
    assert_eq!(tx.status().bit_errors_sent, 5);
    assert!(!s.sync_loss);

    // Bring the channel down hard enough to lose sync.
    for _ in 0..1000 {
        rx.process_bit(0);
    }
    let s = rx.status();
    assert_eq!(s.locked_state, LockState::Search);
    assert!(s.sync_loss);

    // Sync-loss stays latched until explicitly reset.
    run_clean_bits(&mut tx, &mut rx, 500);
    assert!(rx.status().sync_loss);
    rx.handle_control(RxBertControl::Reset);
    assert!(!rx.status().sync_loss);
}
