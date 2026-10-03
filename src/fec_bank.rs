//! The receiver's bank of soft Viterbi decoders, one per phase-ambiguity
//! hypothesis. The carrier loop leaves the constellation rotated by an
//! unknown multiple of 360/M degrees (M hypotheses: 2 for BPSK, 4 for QPSK, 8
//! for 8PSK), and when the symbols carry an odd number of coded bits the
//! receiver also does not know which coded bit starts a code pair (2
//! hypotheses; with an even number every symbol holds whole pairs and there is
//! only 1). So 2-16 decoders run in parallel on the same symbol stream; only
//! the correct one produces meaningful bits. Consumers (the PRBS BERT bank,
//! the HDLC deframer) pick the right branch by whether its output is valid.
//!
//! Branch index = `rotation * pair_phases + pair_phase`.

use num_complex::Complex32;

use crate::conv::ViterbiDecoder;
use crate::symbol_map::Modulation;

/// Code-pair alignment hypotheses: 2 when a symbol carries an odd number of
/// coded bits (pairs straddle symbol boundaries), else 1.
pub fn pair_phases(m: Modulation) -> usize {
    if m.bits_per_symbol() % 2 == 1 {
        2
    } else {
        1
    }
}

pub fn num_branches(m: Modulation) -> usize {
    m.num_points() as usize * pair_phases(m)
}

pub fn branch_rotation(m: Modulation, branch: usize) -> u8 {
    (branch / pair_phases(m)) as u8
}

pub fn branch_pair_phase(m: Modulation, branch: usize) -> u8 {
    (branch % pair_phases(m)) as u8
}

pub struct SoftDecoderBank {
    modulation: Modulation,
    decoders: Vec<ViterbiDecoder>,
    /// Branches currently being fed. Normally all of them; once a consumer
    /// has found the right branch it can `restrict_to` it (the others cost
    /// CPU and have nothing to say), and go back with `activate_all`.
    active: Vec<bool>,
}

impl SoftDecoderBank {
    pub fn new(modulation: Modulation) -> Self {
        SoftDecoderBank {
            modulation,
            decoders: (0..num_branches(modulation))
                .map(|b| ViterbiDecoder::new(branch_pair_phase(modulation, b) == 1))
                .collect(),
            active: vec![true; num_branches(modulation)],
        }
    }

    /// Feed only `branch` from now on. Its decoder keeps running; the others
    /// stop (their state goes stale and is reset by `activate_all`).
    pub fn restrict_to(&mut self, branch: usize) {
        for (b, a) in self.active.iter_mut().enumerate() {
            *a = b == branch;
        }
    }

    /// Stop feeding every branch (the bank is idle until `activate_all`).
    pub fn deactivate_all(&mut self) {
        self.active.fill(false);
    }

    /// Feed every branch again. Branches that were stopped restart with a
    /// fresh decoder (a stopped one holds stale path metrics); those that
    /// kept running are left alone. Returns the branches that were restarted.
    pub fn activate_all(&mut self) -> Vec<usize> {
        let mut restarted = Vec::new();
        for b in 0..self.decoders.len() {
            if !self.active[b] {
                self.decoders[b] = ViterbiDecoder::new(branch_pair_phase(self.modulation, b) == 1);
                self.active[b] = true;
                restarted.push(b);
            }
        }
        restarted
    }

    pub fn is_active(&self, branch: usize) -> bool {
        self.active[branch]
    }

    /// True while some branch is not being fed.
    pub fn is_restricted(&self) -> bool {
        self.active.iter().any(|a| !a)
    }

    pub fn any_active(&self) -> bool {
        self.active.iter().any(|a| *a)
    }

    /// Feed one received (carrier-derotated) symbol. Newly decoded bits for
    /// branch `b` are appended to `outs[b]` (the caller clears them as it
    /// likes). `outs` must have `num_branches(modulation)` entries.
    pub fn process_symbol(&mut self, y: Complex32, outs: &mut [Vec<u8>]) {
        let m = self.modulation;
        debug_assert_eq!(outs.len(), self.decoders.len());
        let distances = m.distances(y);
        let phases = pair_phases(m);
        let bps = m.bits_per_symbol();
        for rotation in 0..m.num_points() {
            let first = rotation as usize * phases;
            if !self.active[first..first + phases].iter().any(|a| *a) {
                continue;
            }
            let soft = m.soft_bits(&distances, rotation);
            for phase in 0..phases {
                let branch = first + phase;
                if !self.active[branch] {
                    continue;
                }
                for &llr in &soft[..bps] {
                    self.decoders[branch].push_llr(llr, &mut outs[branch]);
                }
            }
        }
    }

    /// Smoothed raw (pre-FEC) channel BER seen by a branch's decoder.
    pub fn recent_channel_ber(&self, branch: usize) -> f64 {
        self.decoders[branch].recent_channel_ber()
    }
}
