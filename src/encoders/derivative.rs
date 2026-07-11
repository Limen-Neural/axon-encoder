use crate::prelude::*;

/// Encodes based on the rate of change (derivative) of the input.
///
/// Fires an excitatory spike when the positive change exceeds a threshold,
/// and an inhibitory spike when the negative change exceeds the threshold.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
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
            return Err("last_values and thresholds must have the same length".into());
        }
        if r.thresholds.len() > u16::MAX as usize + 1 {
            return Err("too many channels (max 65536)".into());
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
}
