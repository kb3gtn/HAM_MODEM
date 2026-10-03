//! The RX direction thread: owns every RX-side block (currently just the
//! BERT PRBS checker) and multiplexes its data input against control input
//! and periodic status output on one real-time thread. See tx_chain.rs for
//! why this is one thread hosting multiple blocks, not one thread per block.

use std::thread::{self, JoinHandle};
use std::time::Duration;

use crossbeam_channel::{select, Receiver, Sender};

use crate::prbs::PrbsPattern;
use crate::rx_bert::{RxBert, RxBertControl, RxBertStatus};
use crate::Block;

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum RxControl {
    Prbs(RxBertControl),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum RxStatus {
    Prbs(RxBertStatus),
}

pub struct RxChain {
    bert: RxBert,
}

impl RxChain {
    pub fn new(pattern: PrbsPattern) -> Self {
        RxChain {
            bert: RxBert::new(pattern),
        }
    }

    pub fn process_bit(&mut self, bit: u8) {
        self.bert.process_bit(bit);
    }

    pub fn dispatch_control(&mut self, cmd: RxControl) {
        match cmd {
            RxControl::Prbs(c) => self.bert.handle_control(c),
        }
    }

    pub fn collect_status(&self) -> RxStatus {
        RxStatus::Prbs(self.bert.status())
    }
}

/// Spawn the RX direction thread. `status_interval` is exposed (rather than
/// hardcoded to 1s) so tests can use a much shorter interval and stay fast.
pub fn spawn_rx_thread(
    pattern: PrbsPattern,
    bit_in: Receiver<u8>,
    control_in: Receiver<RxControl>,
    status_out: Sender<RxStatus>,
    status_interval: Duration,
) -> JoinHandle<()> {
    thread::spawn(move || {
        let mut chain = RxChain::new(pattern);
        let ticker = crossbeam_channel::tick(status_interval);

        loop {
            select! {
                recv(bit_in) -> bit => match bit {
                    Ok(b) => chain.process_bit(b),
                    Err(_) => break, // upstream disconnected
                },
                recv(control_in) -> cmd => match cmd {
                    Ok(c) => chain.dispatch_control(c),
                    Err(_) => break, // controller disconnected
                },
                recv(ticker) -> _ => {
                    let _ = status_out.try_send(chain.collect_status());
                },
            }
        }
    })
}
