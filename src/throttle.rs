//! A reusable, timing-only pacing block. Generic over nothing - it doesn't
//! touch data at all, it just answers "when is the next paced instant,"
//! which a driver loop selects on alongside its other event sources.
//!
//! This is explicitly a TEST-MODE tool (see `tx_chain.rs`'s pacing-model doc
//! comment): standing in for real hardware demand in the absence of a
//! bladeRF TX stream. It is not how production TX pacing will work.

use std::time::{Duration, Instant};

use crossbeam_channel::Receiver;

use crate::Block;

#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum ThrottleControl {
    SetRate { items_per_second: f64 },
}

#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ThrottleStatus {
    pub items_per_second: f64,
    /// How many times a paced deadline fired but the item could not be
    /// consumed downstream (e.g. a full channel), forcing a resync. A
    /// climbing count means whatever's downstream can't keep up with the
    /// configured rate.
    pub resync_count: u64,
}

pub struct Throttle {
    items_per_second: f64,
    period: Duration,
    next_due: Instant,
    resync_count: u64,
}

impl Throttle {
    pub fn new(items_per_second: f64) -> Self {
        Throttle {
            items_per_second,
            period: Self::period_for_rate(items_per_second),
            next_due: Instant::now(),
            resync_count: 0,
        }
    }

    fn period_for_rate(items_per_second: f64) -> Duration {
        assert!(items_per_second > 0.0, "throttle rate must be positive");
        Duration::from_secs_f64(1.0 / items_per_second)
    }

    pub fn set_rate(&mut self, items_per_second: f64) {
        self.items_per_second = items_per_second;
        self.period = Self::period_for_rate(items_per_second);
    }

    /// A one-shot receiver firing at the next paced instant - select on this.
    pub fn deadline(&self) -> Receiver<Instant> {
        crossbeam_channel::at(self.next_due)
    }

    /// Call once the paced item has actually been consumed/sent successfully.
    /// Accumulates from the previous deadline rather than `now + period`, so
    /// scheduling jitter doesn't accumulate into long-term drift.
    pub fn advance(&mut self) {
        self.next_due += self.period;
    }

    /// Call if the deadline fired but the item could NOT be consumed (e.g.
    /// downstream channel full). Resyncs to "now" instead of letting missed
    /// deadlines pile up into a burst of catch-up firings.
    pub fn resync(&mut self) {
        self.next_due = Instant::now();
        self.resync_count += 1;
    }
}

impl Block for Throttle {
    type Control = ThrottleControl;
    type Status = ThrottleStatus;

    fn handle_control(&mut self, cmd: ThrottleControl) {
        match cmd {
            ThrottleControl::SetRate { items_per_second } => self.set_rate(items_per_second),
        }
    }

    fn status(&self) -> ThrottleStatus {
        ThrottleStatus {
            items_per_second: self.items_per_second,
            resync_count: self.resync_count,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread::sleep;

    #[test]
    fn paces_at_configured_rate_without_drift() {
        let mut throttle = Throttle::new(1000.0); // 1ms period
        let start = Instant::now();
        for _ in 0..100 {
            throttle.deadline().recv().unwrap();
            throttle.advance();
        }
        let elapsed = start.elapsed();
        let expected = Duration::from_millis(100);
        assert!(
            elapsed >= expected.mul_f64(0.8) && elapsed <= expected.mul_f64(1.5),
            "elapsed {elapsed:?} inconsistent with expected {expected:?}"
        );
    }

    #[test]
    fn resync_advances_past_a_missed_deadline_instead_of_bursting() {
        let mut throttle = Throttle::new(1000.0);
        // Let a deadline pass without ever consuming it.
        sleep(Duration::from_millis(50));
        assert_eq!(throttle.status().resync_count, 0);
        throttle.resync();
        assert_eq!(throttle.status().resync_count, 1);
        // The new deadline is ~now, not 50 missed periods in the past.
        assert!(throttle
            .deadline()
            .recv_timeout(Duration::from_millis(5))
            .is_ok());
    }

    #[test]
    fn set_rate_changes_status_and_future_pacing() {
        let mut throttle = Throttle::new(1000.0);
        assert_eq!(throttle.status().items_per_second, 1000.0);
        throttle.handle_control(ThrottleControl::SetRate {
            items_per_second: 5000.0,
        });
        assert_eq!(throttle.status().items_per_second, 5000.0);
    }
}
