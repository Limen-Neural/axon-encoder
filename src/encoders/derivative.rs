use crate::prelude::*;

/// Encodes based on the rate of change (derivative) of the input.
///
/// Fires an excitatory spike when the positive change exceeds a threshold,
/// and an inhibitory spike when the negative change exceeds the threshold.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(try_from = "DerivativeEncoderRepr"))]
pub struct DerivativeEncoder {
    last_values: Vec<f32>,
    thresholds: Vec<f32>,
}

#[cfg(feature = "serde")]
#[derive(serde::Deserialize)]
struct DerivativeEncoderRepr {
    last_values: Vec<f32>,
    thresholds: Vec<f32>,
}

#[cfg(feature = "serde")]
impl TryFrom<DerivativeEncoderRepr> for DerivativeEncoder {
    type Error = String;

    fn try_from(r: DerivativeEncoderRepr) -> Result<Self, String> {
        if r.last_values.len() != r.thresholds.len() {
            return Err(format!(
                "mismatched last_values length ({}) and thresholds length ({})",
                r.last_values.len(),
                r.thresholds.len()
            ));
        }
        if r.thresholds.len() > u16::MAX as usize + 1 {
            return Err("too many channels (max 65536)".into());
        }
        if r.thresholds.iter().any(|v| !v.is_finite()) {
            return Err("thresholds must be finite".into());
        }
        if r.last_values.iter().any(|v| !v.is_finite()) {
            return Err("last_values must be finite".into());
        }
        Ok(Self {
            last_values: r.last_values,
            thresholds: r.thresholds,
        })
    }
}

impl DerivativeEncoder {
    /// Creates a new `DerivativeEncoder` with specific thresholds for each channel.
    pub fn new(thresholds: Vec<f32>) -> Self {
        assert!(
            thresholds.len() <= u16::MAX as usize + 1,
            "too many channels (max 65536)"
        );
        assert!(
            thresholds.iter().all(|v| v.is_finite()),
            "thresholds must be finite"
        );
        let num_channels = thresholds.len();
        Self {
            last_values: vec![0.0; num_channels],
            thresholds,
        }
    }
}

impl Encoder for DerivativeEncoder {
    fn encode(&mut self, input: &[f32]) -> EncodedOutput {
        self.encode_step(input)
    }

    fn encode_step(&mut self, current_values: &[f32]) -> EncodedOutput {
        let mut output = EncodedOutput::new();

        for (i, &current_val) in current_values.iter().enumerate() {
            if i >= self.thresholds.len() {
                break;
            }

            let delta = current_val - self.last_values[i];

            // Excitatory spike on positive jump exceeding threshold
            if delta > self.thresholds[i] {
                output.spikes.push(SpikeEvent {
                    channel: u16::try_from(i).expect("channel index exceeds u16::MAX"),
                    timestamp: 0,
                    polarity: true,
                });
            }
            // Inhibitory/Negative spike on sudden drop
            else if delta < -self.thresholds[i] {
                output.spikes.push(SpikeEvent {
                    channel: u16::try_from(i).expect("channel index exceeds u16::MAX"),
                    timestamp: 0,
                    polarity: false,
                });
            }

            self.last_values[i] = current_val;
        }
        output
    }

    fn reset(&mut self) {
        for val in self.last_values.iter_mut() {
            *val = 0.0;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_derivative_encoder_basic() {
        let mut encoder = DerivativeEncoder::new(vec![1.0, 2.0]);

        // Initial jump
        let output = encoder.encode(&[1.5, 1.5]);
        assert_eq!(output.spikes.len(), 1);
        assert_eq!(output.spikes[0].channel, 0);
        assert!(output.spikes[0].polarity);

        // Stay same
        let output = encoder.encode(&[1.5, 1.5]);
        assert!(output.spikes.is_empty());

        // Jump down
        let output = encoder.encode(&[0.0, 1.5]);
        assert_eq!(output.spikes.len(), 1);
        assert_eq!(output.spikes[0].channel, 0);
        assert!(!output.spikes[0].polarity);

        // Jump up on channel 1
        let output = encoder.encode(&[0.0, 4.0]);
        assert_eq!(output.spikes.len(), 1);
        assert_eq!(output.spikes[0].channel, 1);
        assert!(output.spikes[0].polarity);
    }

    #[test]
    fn test_derivative_encoder_reset() {
        let mut encoder = DerivativeEncoder::new(vec![1.0]);
        encoder.encode(&[5.0]);
        encoder.reset();
        assert_eq!(encoder.last_values[0], 0.0);
    }

    #[test]
    fn test_derivative_encoder_empty_and_mismatched() {
        let mut encoder = DerivativeEncoder::new(vec![1.0]);
        let output = encoder.encode(&[]);
        assert!(output.spikes.is_empty());

        let output = encoder.encode(&[2.0, 3.0]);
        assert_eq!(output.spikes.len(), 1); // Only channel 0 should be processed
    }

    #[cfg(feature = "serde")]
    #[test]
    fn test_derivative_serde_rejects_too_many_channels() {
        let values: Vec<f32> = vec![0.0; (u16::MAX as usize) + 2];
        let value = serde_json::json!({
            "last_values": values.clone(),
            "thresholds": values,
        });
        let res: Result<DerivativeEncoder, _> = serde_json::from_value(value);
        assert!(res.is_err());
    }
}

#[cfg(test)]
mod branch_coverage_tests {
    use super::*;

    #[test]
    fn derivative_encoder_initializes_channel_state() {
        let encoder = DerivativeEncoder::new(vec![1.0, 2.0, 3.0]);
        assert_eq!(encoder.thresholds, vec![1.0, 2.0, 3.0]);
        assert_eq!(encoder.last_values, vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn derivative_encoder_tracks_positive_and_negative_steps() {
        let mut encoder = DerivativeEncoder::new(vec![1.0, 2.0]);

        let output = encoder.encode_step(&[1.5, -2.5]);
        assert_eq!(output.spikes.len(), 2);
        assert_eq!(output.spikes[0].channel, 0);
        assert!(output.spikes[0].polarity);
        assert_eq!(output.spikes[1].channel, 1);
        assert!(!output.spikes[1].polarity);
        assert_eq!(encoder.last_values, vec![1.5, -2.5]);

        let output = encoder.encode_step(&[2.0, -1.0]);
        assert!(output.spikes.is_empty());
        assert_eq!(encoder.last_values, vec![2.0, -1.0]);

        let output = encoder.encode_step(&[0.0, 2.0]);
        assert_eq!(output.spikes.len(), 2);
        assert!(!output.spikes[0].polarity);
        assert!(output.spikes[1].polarity);
        assert_eq!(encoder.last_values, vec![0.0, 2.0]);
    }

    #[test]
    fn derivative_encoder_does_not_fire_at_threshold() {
        let mut encoder = DerivativeEncoder::new(vec![1.0, 2.0]);
        let output = encoder.encode_step(&[1.0, -2.0]);
        assert!(output.spikes.is_empty());
        assert_eq!(encoder.last_values, vec![1.0, -2.0]);
    }

    #[test]
    fn derivative_encoder_handles_channel_count_mismatches() {
        let mut encoder = DerivativeEncoder::new(vec![1.0, 2.0]);
        let output = encoder.encode_step(&[2.0, -3.0, 5.0]);
        assert_eq!(output.spikes.len(), 2);
        assert_eq!(encoder.last_values, vec![2.0, -3.0]);

        let mut encoder = DerivativeEncoder::new(vec![1.0, 2.0]);
        let output = encoder.encode_step(&[2.0]);
        assert_eq!(output.spikes.len(), 1);
        assert_eq!(output.spikes[0].channel, 0);
        assert_eq!(encoder.last_values, vec![2.0, 0.0]);
    }
}
