//! Slow amplitude normalizer placed ahead of timing recovery. The Gardner
//! timing-error magnitude scales with signal power, and the analog front end
//! gain is arbitrary, so the loop needs a roughly unit-power input to keep its
//! gains meaningful. 8PSK carries no information in amplitude, so a plain
//! power-tracking AGC costs nothing.

use num_complex::Complex32;

pub struct Agc {
    avg_power: f32,
    rate: f32,
    count: u32,
}

impl Agc {
    /// `rate`: steady-state IIR coefficient per sample (e.g. 2e-4). During the
    /// first ~1/rate samples a running mean is used instead, so the gain
    /// converges immediately rather than at the steady-state time constant.
    pub fn new(rate: f32) -> Self {
        Agc {
            avg_power: 0.0,
            rate,
            count: 0,
        }
    }

    pub fn process(&mut self, x: Complex32) -> Complex32 {
        self.count = self.count.saturating_add(1);
        let k = (1.0 / self.count as f32).max(self.rate);
        self.avg_power += k * (x.norm_sqr() - self.avg_power);
        if self.avg_power > 1e-12 {
            x / self.avg_power.sqrt()
        } else {
            x
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalizes_to_unit_power_regardless_of_input_level() {
        for level in [0.001f32, 0.3, 40.0] {
            let mut agc = Agc::new(2e-4);
            let mut last = Vec::new();
            for k in 0..20_000 {
                let y = agc.process(Complex32::from_polar(level, 0.1 * k as f32));
                if k >= 19_000 {
                    last.push(y.norm_sqr());
                }
            }
            let mean: f32 = last.iter().sum::<f32>() / last.len() as f32;
            assert!(
                (mean - 1.0).abs() < 0.02,
                "level {level}: mean power {mean}"
            );
        }
    }
}
