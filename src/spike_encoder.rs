use crate::types::{EncodedOutput, SpikeEvent};

pub struct DerivativeEncoder {
    last_values: Vec<f32>,
    thresholds: Vec<f32>,
}

impl DerivativeEncoder {
    /// Creates a new derivative encoder with specific thresholds for each channel
    pub fn new(thresholds: Vec<f32>) -> Self {
        Self {
            last_values: vec![0.0; thresholds.len()],
            thresholds,
        }
    }

    pub fn encode_step(&mut self, current_values: &[f32]) -> EncodedOutput {
        let mut output = EncodedOutput::new();

        for (i, &current_val) in current_values.iter().enumerate() {
            if i >= self.thresholds.len() {
                break;
            }

            let delta = current_val - self.last_values[i];

            // Excitatory spike on positive jump exceeding threshold
            if delta > self.thresholds[i] {
                output.spikes.push(SpikeEvent {
                    channel: i as u16,
                    timestamp: 0,
                    polarity: true,
                });
            }
            // Inhibitory/Negative spike on sudden drop
            else if delta < -self.thresholds[i] {
                output.spikes.push(SpikeEvent {
                    channel: i as u16,
                    timestamp: 0,
                    polarity: false,
                });
            }

            self.last_values[i] = current_val;
        }
        output
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_derivative_encoder_new() {
        let encoder = DerivativeEncoder::new(vec![1.0, 2.0, 3.0]);
        assert_eq!(encoder.thresholds, vec![1.0, 2.0, 3.0]);
        assert_eq!(encoder.last_values, vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_derivative_encoder_encode_step_happy_path() {
        let mut encoder = DerivativeEncoder::new(vec![1.0, 2.0]);

        // First step: jumps exceeding thresholds
        // Channel 0: 0.0 -> 1.5 (delta = 1.5 > 1.0) -> Excitatory spike
        // Channel 1: 0.0 -> -2.5 (delta = -2.5 < -2.0) -> Inhibitory spike
        let output = encoder.encode_step(&[1.5, -2.5]);
        assert_eq!(output.spikes.len(), 2);

        assert_eq!(output.spikes[0].channel, 0);
        assert_eq!(output.spikes[0].timestamp, 0);
        assert!(output.spikes[0].polarity);

        assert_eq!(output.spikes[1].channel, 1);
        assert_eq!(output.spikes[1].timestamp, 0);
        assert!(!output.spikes[1].polarity);

        assert_eq!(encoder.last_values, vec![1.5, -2.5]);

        // Second step: jumps below thresholds
        // Channel 0: 1.5 -> 2.0 (delta = 0.5 <= 1.0) -> No spike
        // Channel 1: -2.5 -> -1.0 (delta = 1.5 <= 2.0) -> No spike
        let output2 = encoder.encode_step(&[2.0, -1.0]);
        assert!(output2.spikes.is_empty());
        assert_eq!(encoder.last_values, vec![2.0, -1.0]);

        // Third step: sudden drops/jumps exceeding thresholds again
        // Channel 0: 2.0 -> 0.0 (delta = -2.0 < -1.0) -> Inhibitory spike
        // Channel 1: -1.0 -> 2.0 (delta = 3.0 > 2.0) -> Excitatory spike
        let output3 = encoder.encode_step(&[0.0, 2.0]);
        assert_eq!(output3.spikes.len(), 2);

        assert_eq!(output3.spikes[0].channel, 0);
        assert_eq!(output3.spikes[0].timestamp, 0);
        assert!(!output3.spikes[0].polarity);

        assert_eq!(output3.spikes[1].channel, 1);
        assert_eq!(output3.spikes[1].timestamp, 0);
        assert!(output3.spikes[1].polarity);

        assert_eq!(encoder.last_values, vec![0.0, 2.0]);
    }

    #[test]
    fn test_derivative_encoder_below_threshold() {
        let mut encoder = DerivativeEncoder::new(vec![1.0, 2.0]);

        // Delta is exactly equal to threshold (delta = 1.0, which is not > 1.0)
        // No spikes should be generated, but last_values should be updated.
        let output = encoder.encode_step(&[1.0, -2.0]);
        assert!(output.spikes.is_empty());
        assert_eq!(encoder.last_values, vec![1.0, -2.0]);
    }

    #[test]
    fn test_derivative_encoder_channel_mismatch() {
        let mut encoder = DerivativeEncoder::new(vec![1.0, 2.0]);

        // Scenario 1: Input has more elements than thresholds.
        // It should break early and only process the first 2 channels.
        let output = encoder.encode_step(&[2.0, -3.0, 5.0]);
        assert_eq!(output.spikes.len(), 2);
        assert_eq!(output.spikes[0].channel, 0);
        assert_eq!(output.spikes[1].channel, 1);
        assert_eq!(encoder.last_values, vec![2.0, -3.0]);

        // Scenario 2: Input has fewer elements than thresholds.
        // It should only process the provided elements.
        let mut encoder2 = DerivativeEncoder::new(vec![1.0, 2.0]);
        let output2 = encoder2.encode_step(&[2.0]);
        assert_eq!(output2.spikes.len(), 1);
        assert_eq!(output2.spikes[0].channel, 0);
        assert_eq!(encoder2.last_values, vec![2.0, 0.0]);
    }
}
