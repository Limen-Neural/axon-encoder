/// Poisson spike train encoder.
///
/// Generates spike trains with Poisson-distributed timing based on input probability.
/// Each step generates an independent binary spike (0 or 1) based on the input probability.
///
/// # Mathematical Model
///
/// ```text
/// probability = clamp(input, 0.0, 1.0)
/// spike[i] = 1 if random() < probability else 0
/// ```
///
/// # When to Use
///
/// - Generating baseline spike trains with controllable average rates
/// - Poisson-like random spike generation for stochastic encoders
/// - Creating temporal patterns with controllable firing rates
///
/// # Note
///
/// This encoder is NOT part of the `Encoder` trait because its output type (`Vec<u8>`)
/// differs from other encoders (`EncodedOutput`). It operates in a different mode:
/// the input is a single probability (0.0 to 1.0) and the output is a spike train
/// over multiple time steps.
#[derive(Clone, Debug)]
pub struct PoissonEncoder {
    pub num_steps: usize,
}

impl PoissonEncoder {
    pub fn new(steps: usize) -> Self {
        Self { num_steps: steps }
    }

    /// Encodes a single probability value into a spike train.
    ///
    /// Each of the `num_steps` represents an independent time step where
    /// a spike occurs with the given probability.
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

    /// Encodes a single step - returns 1 or 0 based on input probability.
    ///
    /// Useful for streaming mode where you want one spike decision at a time.
    pub fn encode_step(&self, input: f32) -> u8 {
        let probability = input.clamp(0.0, 1.0);
        if crate::rng::gen_unit_f32() < probability {
            1
        } else {
            0
        }
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
    fn spike_count_produces_mixed_output() {
        let enc = PoissonEncoder::new(100);
        let spikes = enc.encode(0.5);
        let count = spikes.iter().filter(|&&s| s == 1).count();
        assert!(
            count > 0 && count < 100,
            "p=0.5 should produce mixed output, got {} spikes",
            count
        );
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
