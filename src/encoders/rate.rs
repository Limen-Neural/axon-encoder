use crate::prelude::*;

/// Encodes analog values as spike rates based on input intensity.
///
/// Each input channel is mapped to a firing rate between `base_rate` and `max_rate`.
/// In batch mode (`encode`), each call generates independent probabilistic spikes.
/// In streaming mode (`encode_step`), accumulates probability and fires deterministically
/// when the accumulated value exceeds a threshold per channel.
///
/// # Mathematical Model
///
/// For batch encoding:
/// ```text
/// rate = base_rate + normalized * (max_rate - base_rate)
/// probability = rate / 10.0
/// spike if random() < probability
/// ```
///
/// For streaming (`encode_step`):
/// ```text
/// accumulator[i] += normalized[i] * (max_rate - base_rate) / 10.0
/// spike if accumulator[i] >= 1.0 (then accumulator -= 1.0)
/// ```
///
/// # When to Use
///
/// - Converting continuous sensor values to spike rates
/// - Poisson-like spike generation with controllable average rates
/// - Real-time encoding where spike timing follows input intensity
///
/// # Parameters
///
/// - `base_rate`: Minimum firing rate (Hz equivalent) when input is at range minimum
/// - `max_rate`: Maximum firing rate (Hz equivalent) when input is at range maximum
/// - `range`: Tuple of (min, max) input values
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct RateEncoder {
    base_rate: f32,
    max_rate: f32,
    range: (f32, f32),
    accumulators: Vec<f32>,
}

impl RateEncoder {
    pub fn new(base_rate: f32, max_rate: f32, range: (f32, f32)) -> Self {
        Self {
            base_rate,
            max_rate,
            range,
            accumulators: Vec::new(),
        }
    }

    fn normalize(&self, value: f32) -> f32 {
        ((value - self.range.0) / (self.range.1 - self.range.0)).clamp(0.0, 1.0)
    }

    fn ensure_accumulators(&mut self, num_channels: usize) {
        if self.accumulators.len() < num_channels {
            self.accumulators.resize(num_channels, 0.0);
        }
    }
}

impl Encoder for RateEncoder {
    fn encode(&mut self, input: &[f32]) -> EncodedOutput {
        let mut output = EncodedOutput::new();
        if input.is_empty() {
            return output;
        }

        for (i, &value) in input.iter().enumerate() {
            let normalized = self.normalize(value);
            let rate = self.base_rate + normalized * (self.max_rate - self.base_rate);
            let probability = (rate / 10.0).clamp(0.0, 1.0);

            if crate::rng::gen_unit_f32() < probability {
                output.spikes.push(SpikeEvent {
                    channel: i as u16,
                    timestamp: 0,
                    polarity: true,
                });
            }
        }

        output
    }

    fn encode_step(&mut self, input: &[f32]) -> EncodedOutput {
        let mut output = EncodedOutput::new();
        if input.is_empty() {
            return output;
        }

        self.ensure_accumulators(input.len());

        for (i, &value) in input.iter().enumerate() {
            let normalized = self.normalize(value);
            let rate_increment =
                (self.base_rate + normalized * (self.max_rate - self.base_rate)) / 10.0;
            self.accumulators[i] += rate_increment;

            while self.accumulators[i] >= 1.0 {
                output.spikes.push(SpikeEvent {
                    channel: i as u16,
                    timestamp: 0,
                    polarity: true,
                });
                self.accumulators[i] -= 1.0;
            }
        }

        output
    }

    fn reset(&mut self) {
        for acc in self.accumulators.iter_mut() {
            *acc = 0.0;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rate_encoder_basic() {
        let mut encoder = RateEncoder::new(0.0, 10.0, (0.0, 100.0));
        let input = [0.0, 50.0, 100.0];
        let output = encoder.encode(&input);
        assert!(output.spikes.len() <= 3);
    }

    #[test]
    fn test_rate_encoder_empty_input() {
        let mut encoder = RateEncoder::new(0.0, 10.0, (0.0, 100.0));
        let input: [f32; 0] = [];
        let output = encoder.encode(&input);
        assert_eq!(output.spikes.len(), 0);
    }

    #[test]
    fn test_rate_encoder_single_channel() {
        let mut encoder = RateEncoder::new(5.0, 10.0, (0.0, 1.0));
        let input = [0.5];
        let output = encoder.encode(&input);
        assert!(output.spikes.len() <= 1);
    }

    #[test]
    fn test_rate_encoder_below_min() {
        let mut encoder = RateEncoder::new(0.0, 10.0, (0.0, 100.0));
        let input = [-50.0, -100.0, -1.0];
        let output = encoder.encode(&input);
        assert!(
            output.spikes.is_empty(),
            "Below-min inputs should produce no spikes"
        );
    }

    #[test]
    fn test_rate_encoder_above_max() {
        let mut encoder = RateEncoder::new(0.0, 10.0, (0.0, 100.0));
        let input = [150.0, 200.0, 101.0];
        let output = encoder.encode(&input);
        assert!(output.spikes.len() <= 3);
        for spike in &output.spikes {
            assert!(spike.channel < 3);
        }
    }

    #[test]
    fn test_rate_encoder_reset_does_not_panic() {
        let mut encoder = RateEncoder::new(5.0, 10.0, (0.0, 1.0));
        let input = [0.5; 10];
        encoder.encode(&input);
        encoder.reset();
        encoder.encode(&input);
    }

    #[test]
    fn test_rate_encoder_never_panics() {
        let mut encoder = RateEncoder::new(0.0, 10.0, (0.0, 100.0));
        let inputs: [&[f32]; 4] = [&[], &[0.0], &[50.0, 100.0], &[f32::MIN, f32::MAX]];
        for input in inputs {
            let result =
                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| encoder.encode(input)));
            assert!(result.is_ok());
        }
    }
}
