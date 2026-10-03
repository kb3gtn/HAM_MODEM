//! The TX direction thread: owns every TX-side block (currently just the
//! BERT PRBS generator) and multiplexes its data output against control
//! input and periodic status output on one real-time thread.
//!
//! More blocks (scrambler, symbol map, RRC pulse shape, resampler) get added
//! into `TxChain`'s processing later - they stay in THIS thread, not new
//! threads, per the block-architecture decision: modular blocks as structs
//! called in sequence within one thread per direction, with crossbeam
//! channels used only for genuine cross-thread traffic (data in/out of the
//! thread, control in, status out).
//!
//! `TxChain` itself has ZERO timing awareness - it's pull-shaped, only ever
//! answering "give me the next output." Pacing is handled entirely outside
//! it, by the `Throttle` block owned by `spawn_tx_thread`'s driver loop. See
//! that function's doc comment for why this split matters.

use std::thread::{self, JoinHandle};
use std::time::Duration;

use crossbeam_channel::{select, Receiver, Sender, TrySendError};

use crate::prbs::PrbsPattern;
use crate::throttle::{Throttle, ThrottleControl, ThrottleStatus};
use crate::tx_bert::{TxBert, TxBertControl, TxBertStatus};
use crate::Block;

#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum TxControl {
    Prbs(TxBertControl),
    Throttle(ThrottleControl),
}

#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum TxStatus {
    Prbs(TxBertStatus),
    Throttle(ThrottleStatus),
}

pub struct TxChain {
    bert: TxBert,
}

impl TxChain {
    pub fn new(pattern: PrbsPattern) -> Self {
        TxChain {
            bert: TxBert::new(pattern),
        }
    }

    pub fn next_bit(&mut self) -> u8 {
        self.bert.next_bit()
    }
}

impl Block for TxChain {
    type Control = TxBertControl;
    type Status = TxBertStatus;

    fn handle_control(&mut self, cmd: TxBertControl) {
        self.bert.handle_control(cmd);
    }

    fn status(&self) -> TxBertStatus {
        self.bert.status()
    }
}

/// Spawn the TX direction thread. `status_interval` is exposed (rather than
/// hardcoded to 1s) so tests can use a much shorter interval and stay fast.
/// `initial_bits_per_second` sets the `Throttle`'s starting rate; change it
/// later via `TxControl::Throttle(ThrottleControl::SetRate { .. })`.
///
/// PACING MODEL, IMPORTANT: the `Throttle` block used below is a TEST-MODE
/// STAND-IN for real hardware demand, not the production pacing mechanism.
/// `TxChain::next_bit()` (and everything added to the chain later, e.g.
/// scrambler, symbol map, RRC, resampler) is deliberately pull-shaped and
/// timing-agnostic: it only answers "give me the next output," never assumes
/// who's asking or how often. Once a bladeRF TX stream exists, its blocking
/// write call becomes the real pacing authority (pull-based, propagating
/// backward through the chain exactly like `Throttle` does today; see
/// `SincFixedOut`'s "give me N outputs" shape, chosen for this reason). At
/// that point `Throttle` gets swapped out for a hardware-paced driver loop;
/// TxChain itself does not change. Never let `Throttle` become load-bearing
/// for anything beyond standalone testing.
pub fn spawn_tx_thread(
    pattern: PrbsPattern,
    initial_bits_per_second: f64,
    bit_out: Sender<u8>,
    control_in: Receiver<TxControl>,
    status_out: Sender<TxStatus>,
    status_interval: Duration,
) -> JoinHandle<()> {
    thread::spawn(move || {
        let mut chain = TxChain::new(pattern);
        let mut throttle = Throttle::new(initial_bits_per_second);
        let ticker = crossbeam_channel::tick(status_interval);
        // A bit is generated once, then retried across loop iterations until
        // it's actually sent - so a control/tick event winning the select
        // never silently drops an already-generated bit.
        let mut pending: Option<u8> = None;

        loop {
            let bit = *pending.get_or_insert_with(|| chain.next_bit());
            select! {
                recv(throttle.deadline()) -> _ => match bit_out.try_send(bit) {
                    Ok(()) => {
                        pending = None;
                        throttle.advance();
                    }
                    Err(TrySendError::Full(_)) => throttle.resync(),
                    Err(TrySendError::Disconnected(_)) => break,
                },
                recv(control_in) -> cmd => match cmd {
                    Ok(TxControl::Prbs(c)) => chain.handle_control(c),
                    Ok(TxControl::Throttle(c)) => throttle.handle_control(c),
                    Err(_) => break, // controller disconnected
                },
                recv(ticker) -> _ => {
                    let _ = status_out.try_send(TxStatus::Prbs(chain.status()));
                    let _ = status_out.try_send(TxStatus::Throttle(throttle.status()));
                },
            }
        }
    })
}
