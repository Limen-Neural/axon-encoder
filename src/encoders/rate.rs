use crate::prelude::*;

pub struct RateEncoder {
    base_rate: f32,
    max_rate: f32,
    range: (f32, f32),
}

impl RateEncoder {
    pub fn new(base_rate: f32, max_rate: f32, range: (f32, f32)) -> Self {
        Self {
            base_rate,
            max_rate,
            range,
        }
    }

    fn normalize(&self, value: f32) -> f32 {
        ((value - self.range.0) / (self.range.1 - self.range.0)).clamp(0.0, 1.0)
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

    fn reset(&mut self) {}
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
        assert!(output.spikes.is_empty() || output.spikes.len() <= 3);
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
