/// Poisson spike train encoder.
///
/// Generates spike trains with Poisson-distributed timing based on either a per-step
/// probability or an explicit firing rate and time step.
///
/// # Mathematical Model
///
/// ```text
/// // Dimensionless probability input:
/// probability = clamp(input, 0.0, 1.0)
/// spike[i] = 1 if random() < probability else 0
///
/// // Physical rate input:
/// probability = 1 - exp(-rate_hz * dt_seconds)
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
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct PoissonEncoder {
    pub num_steps: usize,
}

/// Converts a firing rate in hertz and a time-bin width in seconds into the
/// per-bin spike probability for a homogeneous Poisson process.
///
/// The returned value is `1 - exp(-rate_hz * dt_seconds)`. Non-finite or
/// non-positive rates produce `0.0`; invalid `dt_seconds` must be rejected by
/// callers before constructing encoders and is treated as silent here to avoid
/// producing NaN probabilities in stochastic paths.
pub fn probability_from_rate_hz(rate_hz: f32, dt_seconds: f32) -> f32 {
    if !rate_hz.is_finite() || rate_hz <= 0.0 || !dt_seconds.is_finite() || dt_seconds <= 0.0 {
        0.0
    } else {
        (1.0 - (-rate_hz * dt_seconds).exp()).clamp(0.0, 1.0)
    }
}

impl PoissonEncoder {
    pub fn new(steps: usize) -> Self {
        Self { num_steps: steps }
    }

    /// Encodes a firing rate in hertz into a spike train using an explicit time
    /// step in seconds for each bin.
    pub fn encode_rate_hz(&self, rate_hz: f32, dt_seconds: f32) -> Vec<u8> {
        self.encode(probability_from_rate_hz(rate_hz, dt_seconds))
    }

    /// Encodes a single rate-based step using an explicit time step in seconds.
    pub fn encode_rate_hz_step(&self, rate_hz: f32, dt_seconds: f32) -> u8 {
        self.encode_step(probability_from_rate_hz(rate_hz, dt_seconds))
    }

    /// Encodes a single probability value into a spike train.
    ///
    /// Each of the `num_steps` represents an independent time step where
    /// a spike occurs with the given probability.
    pub fn encode(&self, input: f32) -> Vec<u8> {
        let probability = input.clamp(0.0, 1.0);
        let mut rng = rand::rng();
        (0..self.num_steps)
            .map(|_| {
                if crate::rng::gen_unit_f32_with_rng(&mut rng) < probability {
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
        let mut rng = rand::rng();
        if crate::rng::gen_unit_f32_with_rng(&mut rng) < probability {
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
    fn test_poisson_encode_step() {
        let enc = PoissonEncoder::new(1);
        let mut ones = 0;
        let mut zeros = 0;
        for _ in 0..100 {
            let s = enc.encode_step(0.5);
            if s == 1 {
                ones += 1;
            } else {
                zeros += 1;
            }
        }
        assert!(ones > 0 && zeros > 0);
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

    #[test]
    fn rate_probability_uses_explicit_dt_seconds() {
        let probability = probability_from_rate_hz(10.0, 0.01);
        let expected = 1.0 - (-0.1_f32).exp();
        assert!((probability - expected).abs() < f32::EPSILON);
    }

    #[test]
    fn rate_probability_invalid_inputs_are_silent() {
        for (rate_hz, dt_seconds) in [
            (0.0, 0.01),
            (-1.0, 0.01),
            (f32::NAN, 0.01),
            (10.0, 0.0),
            (10.0, f32::NAN),
        ] {
            assert_eq!(probability_from_rate_hz(rate_hz, dt_seconds), 0.0);
        }
    }
}
