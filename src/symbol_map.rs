//! Gray-coded 8PSK constellation mapping. No scrambling or encoding here.
//!
//! Three bits (MSB first) form a Gray label; the label selects one of eight
//! equally spaced unit-circle points at `position * 45` degrees. Gray
//! labelling means a hard-decision error to an adjacent point costs exactly
//! one bit error.
//!
//! PHASE AMBIGUITY: carrier recovery for 8PSK can lock with any of 8 phase
//! offsets (multiples of 45 degrees). A rotation of `r` steps shifts every
//! received constellation *position* by `r` (mod 8) - so the receiver can
//! undo hypothesis `r` with `psk8_demap(sym, r)`. Operating on positions
//! (not on labels/bits) keeps this exact regardless of the bit labelling.

use num_complex::Complex32;
use std::f32::consts::{FRAC_PI_4, TAU};

pub const BITS_PER_SYMBOL: usize = 3;
pub const NUM_POINTS: u8 = 8;

/// Gray label (3 bits) -> constellation position 0..8.
pub fn label_to_position(label: u8) -> u8 {
    let l = label & 7;
    l ^ (l >> 1) ^ (l >> 2)
}

/// Constellation position 0..8 -> Gray label.
pub fn position_to_label(position: u8) -> u8 {
    let p = position & 7;
    p ^ (p >> 1)
}

/// Unit-magnitude constellation point at `position * 45` degrees.
pub fn psk8_point(position: u8) -> Complex32 {
    Complex32::from_polar(1.0, FRAC_PI_4 * f32::from(position & 7))
}

/// Pack three bits (MSB first, each 0/1) into a label.
pub fn bits_to_label(bits: [u8; 3]) -> u8 {
    ((bits[0] & 1) << 2) | ((bits[1] & 1) << 1) | (bits[2] & 1)
}

/// Unpack a label into three bits, MSB first.
pub fn label_to_bits(label: u8) -> [u8; 3] {
    [(label >> 2) & 1, (label >> 1) & 1, label & 1]
}

/// Map three bits (MSB first) to a unit-magnitude 8PSK symbol.
pub fn psk8_map(bits: [u8; 3]) -> Complex32 {
    psk8_point(label_to_position(bits_to_label(bits)))
}

/// Nearest constellation position (0..8) to a received sample.
pub fn psk8_position(sym: Complex32) -> u8 {
    let steps = (sym.arg().rem_euclid(TAU) / FRAC_PI_4).round() as i32;
    (steps & 7) as u8
}

/// Hard-decision demap under rotation hypothesis `rotation` (0..8): the
/// received position is rotated back by `rotation` steps before looking up
/// the Gray label. Returns the three bits, MSB first.
pub fn psk8_demap(sym: Complex32, rotation: u8) -> [u8; 3] {
    psk8_demap_position(psk8_position(sym), rotation)
}

/// As `psk8_demap`, starting from an already-sliced position.
pub fn psk8_demap_position(position: u8, rotation: u8) -> [u8; 3] {
    label_to_bits(position_to_label(position.wrapping_sub(rotation) & 7))
}

/// Signed angular distance (radians, within +-pi/8) from the nearest
/// constellation point - the decision-directed phase error.
pub fn psk8_phase_error(sym: Complex32) -> f32 {
    let nearest = psk8_point(psk8_position(sym));
    (sym * nearest.conj()).arg()
}

/// Squared distance from a received sample to each of the 8 ideal points.
pub fn psk8_distances(sym: Complex32) -> [f32; 8] {
    let mut d = [0.0f32; 8];
    for (p, slot) in d.iter_mut().enumerate() {
        *slot = (sym - psk8_point(p as u8)).norm_sqr();
    }
    d
}

/// Max-log soft values for the three bits of a symbol (MSB first) under
/// rotation hypothesis `rotation`, from `psk8_distances`. Convention matches
/// `conv::ViterbiDecoder`: positive = bit 0 more likely, negative = bit 1 more
/// likely, magnitude = confidence (distance-squared units; the decoder is
/// insensitive to a common scale).
pub fn psk8_soft_bits(distances: &[f32; 8], rotation: u8) -> [f32; 3] {
    let mut min0 = [f32::INFINITY; 3];
    let mut min1 = [f32::INFINITY; 3];
    for (p, &d) in distances.iter().enumerate() {
        let bits = label_to_bits(position_to_label((p as u8).wrapping_sub(rotation) & 7));
        for i in 0..3 {
            if bits[i] == 0 {
                min0[i] = min0[i].min(d);
            } else {
                min1[i] = min1[i].min(d);
            }
        }
    }
    [min1[0] - min0[0], min1[1] - min0[1], min1[2] - min0[2]]
}

/// The supported PSK constellations. All are Gray-labelled, unit-magnitude,
/// equally spaced on the circle with position 0 at angle 0, and share the
/// same `position`-based rotation-ambiguity handling as 8PSK above: carrier
/// recovery can settle on any of `num_points()` phases (multiples of
/// 360/M degrees), and a rotation of `r` steps shifts every received
/// *position* by `r` (mod M).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
pub enum Modulation {
    Bpsk,
    Qpsk,
    #[default]
    Psk8,
}

/// Largest constellation (8PSK); sizes the fixed distance/soft-value arrays.
pub const MAX_POINTS: usize = 8;

impl Modulation {
    pub const ALL: [Modulation; 3] = [Modulation::Bpsk, Modulation::Qpsk, Modulation::Psk8];

    pub fn bits_per_symbol(self) -> usize {
        match self {
            Modulation::Bpsk => 1,
            Modulation::Qpsk => 2,
            Modulation::Psk8 => 3,
        }
    }

    /// Constellation size M (2, 4 or 8).
    pub fn num_points(self) -> u8 {
        1 << self.bits_per_symbol()
    }

    /// Angle between adjacent constellation points, radians.
    pub fn step_radians(self) -> f32 {
        TAU / f32::from(self.num_points())
    }

    /// Gray label (`bits_per_symbol` bits) -> constellation position.
    pub fn label_to_position(self, label: u8) -> u8 {
        let l = label & (self.num_points() - 1);
        let mut p = l;
        let mut s = l >> 1;
        while s != 0 {
            p ^= s;
            s >>= 1;
        }
        p
    }

    /// Constellation position -> Gray label.
    pub fn position_to_label(self, position: u8) -> u8 {
        let p = position & (self.num_points() - 1);
        p ^ (p >> 1)
    }

    pub fn point(self, position: u8) -> Complex32 {
        Complex32::from_polar(
            1.0,
            self.step_radians() * f32::from(position & (self.num_points() - 1)),
        )
    }

    /// Map `bits_per_symbol` bits (MSB first, each 0/1) to a symbol.
    pub fn map(self, bits: &[u8]) -> Complex32 {
        debug_assert_eq!(bits.len(), self.bits_per_symbol());
        let label = bits.iter().fold(0u8, |acc, &b| (acc << 1) | (b & 1));
        self.point(self.label_to_position(label))
    }

    /// Nearest constellation position to a received sample.
    pub fn position(self, sym: Complex32) -> u8 {
        let steps = (sym.arg().rem_euclid(TAU) / self.step_radians()).round() as i32;
        (steps & (i32::from(self.num_points()) - 1)) as u8
    }

    /// Signed angular distance (within +-pi/M) from the nearest point.
    pub fn phase_error(self, sym: Complex32) -> f32 {
        (sym * self.point(self.position(sym)).conj()).arg()
    }

    /// Squared distance to each ideal point; only the first `num_points()`
    /// entries are meaningful.
    pub fn distances(self, sym: Complex32) -> [f32; MAX_POINTS] {
        let mut d = [0.0f32; MAX_POINTS];
        for (p, slot) in d.iter_mut().enumerate().take(self.num_points() as usize) {
            *slot = (sym - self.point(p as u8)).norm_sqr();
        }
        d
    }

    /// Max-log soft values for the symbol's bits (MSB first; only the first
    /// `bits_per_symbol()` entries are meaningful) under rotation hypothesis
    /// `rotation`. Same sign convention as `psk8_soft_bits`.
    pub fn soft_bits(self, distances: &[f32; MAX_POINTS], rotation: u8) -> [f32; 3] {
        let bps = self.bits_per_symbol();
        let m = self.num_points();
        let mut min0 = [f32::INFINITY; 3];
        let mut min1 = [f32::INFINITY; 3];
        for p in 0..m {
            let d = distances[p as usize];
            let label = self.position_to_label(p.wrapping_sub(rotation) & (m - 1));
            for i in 0..bps {
                if (label >> (bps - 1 - i)) & 1 == 0 {
                    min0[i] = min0[i].min(d);
                } else {
                    min1[i] = min1[i].min(d);
                }
            }
        }
        let mut out = [0.0f32; 3];
        for i in 0..bps {
            out[i] = min1[i] - min0[i];
        }
        out
    }

    /// Hard-decision bits (MSB first, `bits_per_symbol` of them) of an
    /// already-sliced position under rotation hypothesis `rotation`.
    pub fn demap_position(self, position: u8, rotation: u8) -> [u8; 3] {
        let bps = self.bits_per_symbol();
        let label =
            self.position_to_label(position.wrapping_sub(rotation) & (self.num_points() - 1));
        let mut out = [0u8; 3];
        for (i, slot) in out.iter_mut().enumerate().take(bps) {
            *slot = (label >> (bps - 1 - i)) & 1;
        }
        out
    }

    /// Short lowercase name for CLIs and logs.
    pub fn name(self) -> &'static str {
        match self {
            Modulation::Bpsk => "bpsk",
            Modulation::Qpsk => "qpsk",
            Modulation::Psk8 => "8psk",
        }
    }

    pub fn parse(s: &str) -> Option<Modulation> {
        match s.to_lowercase().as_str() {
            "bpsk" => Some(Modulation::Bpsk),
            "qpsk" => Some(Modulation::Qpsk),
            "8psk" | "psk8" => Some(Modulation::Psk8),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gray_labelling_round_trips_and_neighbours_differ_by_one_bit() {
        for p in 0..8u8 {
            assert_eq!(label_to_position(position_to_label(p)), p);
        }
        for p in 0..8u8 {
            let a = position_to_label(p);
            let b = position_to_label((p + 1) & 7);
            assert_eq!(
                (a ^ b).count_ones(),
                1,
                "positions {p} and {} not Gray-adjacent",
                (p + 1) & 7
            );
        }
    }

    #[test]
    fn demap_inverts_map_for_all_labels() {
        for label in 0..8u8 {
            let bits = label_to_bits(label);
            assert_eq!(psk8_demap(psk8_map(bits), 0), bits);
        }
    }

    #[test]
    fn symbols_are_unit_magnitude_and_45_degrees_apart() {
        for p in 0..8u8 {
            let s = psk8_point(p);
            assert!((s.norm() - 1.0).abs() < 1e-6);
            assert_eq!(psk8_position(s), p);
        }
    }

    #[test]
    fn rotation_hypothesis_undoes_a_constellation_rotation() {
        for rot in 0..8u8 {
            let rotator = psk8_point(rot);
            for label in 0..8u8 {
                let bits = label_to_bits(label);
                let received = psk8_map(bits) * rotator;
                assert_eq!(
                    psk8_demap(received, rot),
                    bits,
                    "rotation {rot} label {label}"
                );
            }
        }
    }

    #[test]
    fn position_slicer_tolerates_phase_noise_up_to_22_degrees() {
        for p in 0..8u8 {
            for deg in [-21.0f32, -10.0, 0.0, 10.0, 21.0] {
                let s = psk8_point(p) * Complex32::from_polar(0.7, deg.to_radians());
                assert_eq!(psk8_position(s), p);
                assert!((psk8_phase_error(s) - deg.to_radians()).abs() < 1e-4);
            }
        }
    }

    #[test]
    fn soft_bits_have_the_right_sign_and_follow_the_rotation_hypothesis() {
        for rot in 0..8u8 {
            for label in 0..8u8 {
                let bits = label_to_bits(label);
                let rx = psk8_map(bits) * psk8_point(rot); // channel rotates by `rot` steps
                let soft = psk8_soft_bits(&psk8_distances(rx), rot);
                for i in 0..3 {
                    assert_eq!(
                        soft[i] > 0.0,
                        bits[i] == 0,
                        "rot {rot} label {label} bit {i}: {soft:?}"
                    );
                    assert!(
                        soft[i].abs() > 0.5,
                        "ideal symbols should give confident values: {soft:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn soft_bits_lose_confidence_between_points() {
        // Halfway between two adjacent points (22.5 degrees off): the bit that
        // differs between them must be ~undecided.
        let mid = Complex32::from_polar(1.0, std::f32::consts::FRAC_PI_8);
        let soft = psk8_soft_bits(&psk8_distances(mid), 0);
        let a = position_to_label(0);
        let b = position_to_label(1);
        let differing = (a ^ b).trailing_zeros() as usize; // bit index counted from LSB
        assert!(soft[2 - differing].abs() < 1e-3, "{soft:?}");
    }

    #[test]
    fn generic_psk8_matches_the_dedicated_8psk_functions() {
        let m = Modulation::Psk8;
        for label in 0..8u8 {
            assert_eq!(m.label_to_position(label), label_to_position(label));
            assert_eq!(m.position_to_label(label), position_to_label(label));
            let bits = label_to_bits(label);
            assert!((m.map(&bits) - psk8_map(bits)).norm() < 1e-6);
        }
        for deg in (0..360).step_by(7) {
            let s = Complex32::from_polar(0.8, (deg as f32).to_radians());
            assert_eq!(m.position(s), psk8_position(s));
            assert!((m.phase_error(s) - psk8_phase_error(s)).abs() < 1e-6);
            for rot in 0..8u8 {
                assert_eq!(
                    m.soft_bits(&m.distances(s), rot),
                    psk8_soft_bits(&psk8_distances(s), rot)
                );
                assert_eq!(
                    m.demap_position(m.position(s), rot),
                    psk8_demap_position(psk8_position(s), rot)
                );
            }
        }
    }

    #[test]
    fn every_modulation_is_gray_round_trips_and_survives_rotation() {
        for m in Modulation::ALL {
            let n = m.num_points();
            let bps = m.bits_per_symbol();
            for p in 0..n {
                assert_eq!(m.label_to_position(m.position_to_label(p)), p);
                let a = m.position_to_label(p);
                let b = m.position_to_label((p + 1) % n);
                if n > 2 {
                    assert_eq!(
                        (a ^ b).count_ones(),
                        1,
                        "{m:?}: positions {p},{} not Gray-adjacent",
                        (p + 1) % n
                    );
                }
                assert!((m.point(p).norm() - 1.0).abs() < 1e-6);
                assert_eq!(m.position(m.point(p)), p);
            }
            for rot in 0..n {
                for label in 0..n {
                    let bits: Vec<u8> = (0..bps).map(|i| (label >> (bps - 1 - i)) & 1).collect();
                    let rx = m.map(&bits) * m.point(rot);
                    let hard = m.demap_position(m.position(rx), rot);
                    assert_eq!(&hard[..bps], &bits[..], "{m:?} rot {rot} label {label}");
                    let soft = m.soft_bits(&m.distances(rx), rot);
                    for i in 0..bps {
                        assert_eq!(
                            soft[i] > 0.0,
                            bits[i] == 0,
                            "{m:?} rot {rot} label {label} bit {i}: {soft:?}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn modulation_names_parse_back() {
        for m in Modulation::ALL {
            assert_eq!(Modulation::parse(m.name()), Some(m));
        }
        assert_eq!(Modulation::parse("nope"), None);
        assert_eq!(
            serde_json::to_string(&Modulation::Qpsk).unwrap(),
            r#""Qpsk""#
        );
    }
}
