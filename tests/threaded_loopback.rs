//! Real-thread version of the TX BERT <-> RX BERT testbench: two actual OS
//! threads (spawn_tx_thread/spawn_rx_thread) connected by a crossbeam data
//! channel, each independently controllable and observable through its own
//! control-in/status-out channels - exercising the same crossbeam plumbing
//! an external MQTT bridge would eventually sit behind.

use std::time::{Duration, Instant};

use crossbeam_channel::{bounded, unbounded, Receiver};

use ham_modem::prbs::PrbsPattern;
use ham_modem::rx_bert::{LockState, RxBertControl, RxBertStatus};
use ham_modem::rx_chain::{spawn_rx_thread, RxControl, RxStatus};
use ham_modem::throttle::ThrottleControl;
use ham_modem::tx_bert::TxBertControl;
use ham_modem::tx_chain::{spawn_tx_thread, TxControl};

const STATUS_INTERVAL: Duration = Duration::from_millis(5);
const WAIT_TIMEOUT: Duration = Duration::from_secs(2);
// Fast rate to keep the test quick - this is a software pacing knob only at
// this stage, not yet tied to any real symbol timing.
const TEST_BIT_RATE: f64 = 200_000.0;

/// Drain rx_status until `pred` holds on some received status, or panic
/// after WAIT_TIMEOUT. Necessary because control commands and their effects
/// are only visible on the next periodic status tick, not synchronously.
fn wait_for_rx_status(
    status_rx: &Receiver<RxStatus>,
    pred: impl Fn(&RxBertStatus) -> bool,
) -> RxBertStatus {
    let deadline = Instant::now() + WAIT_TIMEOUT;
    loop {
        let remaining = deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() {
            panic!("timed out waiting for expected RX status");
        }
        if let Ok(RxStatus::Prbs(s)) = status_rx.recv_timeout(remaining) {
            if pred(&s) {
                return s;
            }
        }
    }
}

#[test]
fn threaded_tx_rx_bert_loopback() {
    let (bit_tx, bit_rx) = bounded::<u8>(64);
    let (tx_control_tx, tx_control_rx) = unbounded::<TxControl>();
    let (tx_status_tx, _tx_status_rx) = unbounded(); // TX status not exercised by this test
    let (rx_control_tx, rx_control_rx) = unbounded::<RxControl>();
    let (rx_status_tx, rx_status_rx) = unbounded::<RxStatus>();

    let tx_handle = spawn_tx_thread(
        PrbsPattern::Pn15,
        TEST_BIT_RATE,
        bit_tx,
        tx_control_rx,
        tx_status_tx,
        STATUS_INTERVAL,
    );
    let rx_handle = spawn_rx_thread(
        PrbsPattern::Pn15,
        bit_rx,
        rx_control_rx,
        rx_status_tx,
        STATUS_INTERVAL,
    );

    // Lock should be acquired well within the timeout at this bit rate.
    let s = wait_for_rx_status(&rx_status_rx, |s| s.locked_state == LockState::Synced);
    assert_eq!(s.bit_errors_received, 0);

    // Inject one error via the control channel and confirm it shows up in status.
    tx_control_tx
        .send(TxControl::Prbs(TxBertControl::InjectError))
        .unwrap();
    let s = wait_for_rx_status(&rx_status_rx, |s| s.bit_errors_received >= 1);
    assert_eq!(s.bit_errors_received, 1);
    assert_eq!(s.locked_state, LockState::Synced);

    // ResetStats clears counters/latch without dropping lock.
    rx_control_tx
        .send(RxControl::Prbs(RxBertControl::ResetStats))
        .unwrap();
    let s = wait_for_rx_status(&rx_status_rx, |s| s.bit_errors_received == 0);
    assert_eq!(s.locked_state, LockState::Synced);

    // Clean shutdown: dropping each control sender disconnects that
    // channel, which the thread's select! loop observes and exits on.
    drop(tx_control_tx);
    drop(rx_control_tx);
    tx_handle.join().unwrap();
    rx_handle.join().unwrap();
}

#[test]
fn bit_rate_control_paces_transmission_and_can_change_at_runtime() {
    let (bit_tx, bit_rx) = bounded::<u8>(4);
    let (tx_control_tx, tx_control_rx) = unbounded::<TxControl>();
    let (tx_status_tx, _tx_status_rx) = unbounded();

    let slow_rate = 1_000.0; // 1 kbps
    let fast_rate = 20_000.0; // 20 kbps
    let tx_handle = spawn_tx_thread(
        PrbsPattern::Pn11,
        slow_rate,
        bit_tx,
        tx_control_rx,
        tx_status_tx,
        Duration::from_secs(10), // status not exercised by this test
    );

    let n = 200u32;
    let start = Instant::now();
    for _ in 0..n {
        bit_rx
            .recv_timeout(WAIT_TIMEOUT)
            .expect("bit timeout at slow rate");
    }
    let elapsed_slow = start.elapsed();
    let expected_slow = Duration::from_secs_f64(f64::from(n) / slow_rate);
    assert!(
        elapsed_slow >= expected_slow.mul_f64(0.8) && elapsed_slow <= expected_slow.mul_f64(1.5),
        "elapsed {elapsed_slow:?} not consistent with expected {expected_slow:?} at {slow_rate} bps"
    );

    // Bump the rate at runtime and confirm subsequent bits arrive faster.
    tx_control_tx
        .send(TxControl::Throttle(ThrottleControl::SetRate {
            items_per_second: fast_rate,
        }))
        .unwrap();

    let start = Instant::now();
    for _ in 0..n {
        bit_rx
            .recv_timeout(WAIT_TIMEOUT)
            .expect("bit timeout at fast rate");
    }
    let elapsed_fast = start.elapsed();
    let expected_fast = Duration::from_secs_f64(f64::from(n) / fast_rate);
    assert!(
        elapsed_fast >= expected_fast.mul_f64(0.8) && elapsed_fast <= expected_fast.mul_f64(1.5),
        "elapsed {elapsed_fast:?} not consistent with expected {expected_fast:?} at {fast_rate} bps"
    );
    assert!(
        elapsed_fast < elapsed_slow,
        "rate change had no effect: fast run ({elapsed_fast:?}) not faster than slow run ({elapsed_slow:?})"
    );

    drop(tx_control_tx);
    tx_handle.join().unwrap();
}
