#[derive(Clone, Debug)]
pub struct PoissonEncoder {
    pub num_steps: usize,
}

impl PoissonEncoder {
    pub fn new(steps: usize) -> Self {
        Self { num_steps: steps }
    }

    pub fn encode(&self, input: f32) -> Vec<u8> {
        let probability = input.clamp(0.0, 1.0);
        (0..self.num_steps)
            .map(|_| {
                if crate::rng::gen_unit_f32() < probability {
                    1
                } else {
                    0
                }
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn length_matches_num_steps() {
        let enc = PoissonEncoder::new(50);
        let spikes = enc.encode(0.5);
        assert_eq!(spikes.len(), 50);
    }

    #[test]
    fn zero_input_produces_no_spikes() {
        let enc = PoissonEncoder::new(100);
        let spikes = enc.encode(0.0);
        assert!(spikes.iter().all(|&s| s == 0));
    }

    #[test]
    fn full_input_produces_all_spikes() {
        let enc = PoissonEncoder::new(100);
        let spikes = enc.encode(1.0);
        assert!(spikes.iter().all(|&s| s == 1));
    }

    #[test]
    fn values_are_binary() {
        let enc = PoissonEncoder::new(200);
        let spikes = enc.encode(0.4);
        assert!(spikes.iter().all(|&s| s == 0 || s == 1));
    }

    #[test]
    fn empty_steps_produces_empty() {
        let enc = PoissonEncoder::new(0);
        let spikes = enc.encode(0.5);
        assert_eq!(spikes.len(), 0);
    }

    #[test]
    fn negative_input_clamped_to_zero() {
        let enc = PoissonEncoder::new(50);
        let spikes = enc.encode(-0.5);
        assert!(spikes.iter().all(|&s| s == 0));
    }

    #[test]
    fn above_one_input_clamped_to_one() {
        let enc = PoissonEncoder::new(100);
        let spikes = enc.encode(1.5);
        assert!(spikes.iter().all(|&s| s == 1));
    }

    #[test]
    fn spike_count_bounded_by_steps() {
        let enc = PoissonEncoder::new(100);
        let spikes = enc.encode(0.5);
        let count = spikes.iter().filter(|&&s| s == 1).count();
        assert!(count <= 100);
    }

    #[test]
    fn never_panics() {
        let enc = PoissonEncoder::new(50);
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| enc.encode(0.5)));
        assert!(result.is_ok());
        let result =
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| enc.encode(f32::NAN)));
        assert!(result.is_ok());
        let result =
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| enc.encode(f32::INFINITY)));
        assert!(result.is_ok());
    }
}
