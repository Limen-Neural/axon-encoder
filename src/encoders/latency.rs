use crate::prelude::*;

/// Encodes analog values into latency-coded spike times.
///
/// Each input channel produces exactly one positive spike whose timestamp is
/// determined by the input strength within the configured range. Stronger
/// inputs fire earlier. Values below the range minimum map to the latest
/// possible spike at `max_latency`, and values above the range maximum map to
/// timestamp `0`.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct LatencyEncoder {
    max_latency: u64,
    range: (f32, f32),
}

impl LatencyEncoder {
    /// Creates a new `LatencyEncoder`.
    ///
    /// # Panics
    ///
    /// Panics if `range.0 >= range.1`.
    pub fn new(max_latency: u64, range: (f32, f32)) -> Self {
        assert!(range.0 < range.1, "range min must be less than range max");

        Self { max_latency, range }
    }

    fn clamp_and_normalize(&self, value: f32) -> f32 {
        let clamped = value.clamp(self.range.0, self.range.1);
        (clamped - self.range.0) / (self.range.1 - self.range.0)
    }

    fn timestamp_for(&self, value: f32) -> u64 {
        if self.max_latency == 0 {
            return 0;
        }
        if value.is_nan() {
            return self.max_latency;
        }

        let normalized = self.clamp_and_normalize(value) as f64;
        ((1.0 - normalized) * self.max_latency as f64).round() as u64
    }
}

impl Encoder for LatencyEncoder {
    fn encode(&mut self, input: &[f32]) -> EncodedOutput {
        let mut output = EncodedOutput::new();
        output.spikes.reserve(input.len());

        for (channel, &value) in input.iter().enumerate() {
            output.spikes.push(SpikeEvent {
                channel: channel as u16,
                timestamp: self.timestamp_for(value),
                polarity: true,
            });
        }

        output
    }

    fn encode_step(&mut self, input: &[f32]) -> EncodedOutput {
        self.encode(input)
    }

    fn reset(&mut self) {
        // Stateless encoder.
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn latency_encoder_emits_one_positive_spike_per_channel() {
        let mut encoder = LatencyEncoder::new(10, (0.0, 1.0));

        let output = encoder.encode(&[0.0, 0.5, 1.0]);

        assert_eq!(output.spikes.len(), 3);
        assert_eq!(
            output.spikes,
            vec![
                SpikeEvent {
                    channel: 0,
                    timestamp: 10,
                    polarity: true,
                },
                SpikeEvent {
                    channel: 1,
                    timestamp: 5,
                    polarity: true,
                },
                SpikeEvent {
                    channel: 2,
                    timestamp: 0,
                    polarity: true,
                },
            ]
        );
    }

    #[test]
    fn test_latency_encoder_nan() {
        let mut encoder = LatencyEncoder::new(10, (0.0, 1.0));
        let output = encoder.encode(&[f32::NAN]);
        assert_eq!(output.spikes[0].timestamp, 10);
    }

    #[test]
    fn test_latency_encoder_reset() {
        let mut encoder = LatencyEncoder::new(10, (0.0, 1.0));
        encoder.reset(); // Should do nothing
    }

    #[test]
    fn latency_encoder_stronger_inputs_fire_earlier() {
        let mut encoder = LatencyEncoder::new(12, (0.0, 3.0));

        let output = encoder.encode(&[0.5, 1.5, 2.5]);

        assert_eq!(output.spikes.len(), 3);
        assert!(output.spikes[0].timestamp > output.spikes[1].timestamp);
        assert!(output.spikes[1].timestamp > output.spikes[2].timestamp);
    }

    #[test]
    fn latency_encoder_clamps_inputs_to_range() {
        let mut encoder = LatencyEncoder::new(8, (2.0, 6.0));

        let output = encoder.encode(&[0.0, 2.0, 4.0, 6.0, 9.0]);

        assert_eq!(
            output
                .spikes
                .iter()
                .map(|spike| spike.timestamp)
                .collect::<Vec<_>>(),
            vec![8, 8, 4, 0, 0]
        );
    }

    #[test]
    fn latency_encoder_encode_step_matches_encode() {
        let mut encoder = LatencyEncoder::new(20, (-1.0, 1.0));
        let input = [-1.0, -0.25, 0.75, 1.5];

        let batch = encoder.encode(&input);
        let step = encoder.encode_step(&input);

        assert_eq!(batch, step);
    }

    #[test]
    fn latency_encoder_handles_empty_input() {
        let mut encoder = LatencyEncoder::new(5, (0.0, 1.0));

        let output = encoder.encode(&[]);

        assert!(output.spikes.is_empty());
    }

    #[test]
    fn latency_encoder_supports_zero_max_latency() {
        let mut encoder = LatencyEncoder::new(0, (0.0, 1.0));

        let output = encoder.encode(&[-1.0, 0.5, 2.0]);

        assert_eq!(output.spikes.len(), 3);
        assert!(output.spikes.iter().all(|spike| spike.timestamp == 0));
    }

    #[test]
    #[should_panic(expected = "range min must be less than range max")]
    fn latency_encoder_rejects_invalid_range() {
        let _ = LatencyEncoder::new(5, (1.0, 1.0));
    }
}
