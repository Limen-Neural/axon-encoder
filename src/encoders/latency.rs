use crate::prelude::*;

/// Encodes analog values into latency-coded spike times
///
/// Each input channel produces exactly one positive spike whose timestamp is
/// determined by the input strength within the configured range. Stronger
/// inputs fire earlier. Values below the range minimum map to the latest
/// possible spike at `max_latency`, and values above the range maximum map to
/// timestamp `0`
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
pub struct LatencyEncoder {
    max_latency: u64,
    range: (f32, f32),
}

impl LatencyEncoder {
    /// Creates a new `LatencyEncoder`
    ///
    /// # Panics
    ///
    /// Panics if `range.0 >= range.1` or if either bound is non-finite
    pub fn new(max_latency: u64, range: (f32, f32)) -> Self {
        assert!(
            range.0.is_finite() && range.1.is_finite() && range.0 < range.1,
            "range must be finite and min must be less than max"
        );

        Self { max_latency, range }
    }

    fn normalize(&self, value: f32) -> f64 {
        // Use f64 to prevent overflow for valid f32 ranges (e.g., f32::MIN..f32::MAX).
        let clamped = value.clamp(self.range.0, self.range.1) as f64;
        let lo = self.range.0 as f64;
        let hi = self.range.1 as f64;
        (clamped - lo) / (hi - lo)
    }

    fn timestamp_for(&self, value: f32) -> u64 {
        if self.max_latency == 0 {
            return 0;
        }
        if value.is_nan() {
            return self.max_latency;
        }

        let normalized = self.normalize(value);
        ((1.0 - normalized) * self.max_latency as f64).round() as u64
    }

    fn timestamp_for_with_latency_scale(&self, value: f32, latency_scale: f32) -> u64 {
        let scaled_latency = ((self.max_latency as f64) * (latency_scale as f64)).round() as u64;
        if scaled_latency == 0 {
            return 0;
        }
        if value.is_nan() {
            return scaled_latency;
        }

        let normalized = self.normalize(value);
        ((1.0 - normalized) * scaled_latency as f64).round() as u64
    }

    fn encode_with_latency_scale(&mut self, input: &[f32], latency_scale: f32) -> EncodedOutput {
        let mut output = EncodedOutput::new();
        output.spikes.reserve(input.len());

        for (channel, &value) in input.iter().enumerate() {
            let Ok(channel) = u16::try_from(channel) else {
                // Remaining channels exceed u16::MAX; stop rather than wrap.
                break;
            };
            output.spikes.push(SpikeEvent {
                channel,
                timestamp: self.timestamp_for_with_latency_scale(value, latency_scale),
                polarity: true,
            });
        }

        output
    }
}

impl Encoder for LatencyEncoder {
    fn encode(&mut self, input: &[f32]) -> EncodedOutput {
        let mut output = EncodedOutput::new();
        output.spikes.reserve(input.len());

        for (channel, &value) in input.iter().enumerate() {
            let Ok(channel) = u16::try_from(channel) else {
                // Remaining channels exceed u16::MAX; stop rather than wrap.
                break;
            };
            output.spikes.push(SpikeEvent {
                channel,
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

impl ModulatedEncoder for LatencyEncoder {
    fn encode_with_gains(&mut self, input: &[f32], gains: EncodingGains) -> EncodedOutput {
        self.encode_with_latency_scale(input, gains.sanitize().latency_scale)
    }
}

#[cfg(feature = "serde")]
impl<'de> serde::Deserialize<'de> for LatencyEncoder {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(serde::Deserialize)]
        struct Helper {
            max_latency: u64,
            range: (f32, f32),
        }

        let helper = Helper::deserialize(deserializer)?;

        if !helper.range.0.is_finite()
            || !helper.range.1.is_finite()
            || !matches!(
                helper.range.0.partial_cmp(&helper.range.1),
                Some(std::cmp::Ordering::Less)
            )
        {
            return Err(serde::de::Error::custom(
                "range must be finite and min must be less than max",
            ));
        }

        Ok(Self {
            max_latency: helper.max_latency,
            range: helper.range,
        })
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
    fn latency_encoder_nan_maps_to_max_latency() {
        let mut encoder = LatencyEncoder::new(7, (0.0, 1.0));

        let output = encoder.encode(&[f32::NAN, 1.0]);

        assert_eq!(output.spikes[0].timestamp, 7);
        assert_eq!(output.spikes[1].timestamp, 0);
    }

    #[test]
    #[should_panic(expected = "range must be finite and min must be less than max")]
    fn latency_encoder_rejects_invalid_range() {
        let _ = LatencyEncoder::new(5, (1.0, 1.0));
    }

    #[test]
    #[should_panic(expected = "range must be finite and min must be less than max")]
    fn latency_encoder_rejects_infinite_range() {
        let _ = LatencyEncoder::new(10, (f32::NEG_INFINITY, f32::INFINITY));
    }

    #[test]
    fn latency_encoder_truncates_channel_overflow() {
        let mut encoder = LatencyEncoder::new(1, (0.0, 1.0));
        let input = vec![0.0f32; (u16::MAX as usize) + 2];
        let output = encoder.encode(&input);
        assert_eq!(output.spikes.len(), u16::MAX as usize + 1);
    }

    #[test]
    fn latency_encoder_encode_with_modulators_identity() {
        let mut encoder = LatencyEncoder::new(10, (0.0, 1.0));
        let curves = NeuromodulatorGainCurves::default();
        let mods = NeuroModulators::default();

        let plain = encoder.encode(&[0.5]);
        let modulated = encoder.encode_with_modulators(&[0.5], &mods, &curves);

        assert_eq!(plain.spikes[0].timestamp, modulated.spikes[0].timestamp);
    }

    #[test]
    fn latency_encoder_encode_with_modulators_latency_scale() {
        let mut encoder = LatencyEncoder::new(10, (0.0, 1.0));
        let curves = NeuromodulatorGainCurves {
            dopamine: ModulatorGainCurves {
                latency: Some(GainCurve::new((0.0, 1.0), (0.5, 0.5))),
                ..Default::default()
            },
            ..Default::default()
        };
        let mods = NeuroModulators {
            dopamine: 1.0,
            ..Default::default()
        };

        let output = encoder.encode_with_modulators(&[0.5], &mods, &curves);
        // latency_scale = 0.5, so max_latency = 10 * 0.5 = 5
        // normalized(0.5) = 0.5, timestamp = (1.0 - 0.5) * 5 = 2.5 → 3
        assert_eq!(output.spikes[0].timestamp, 3);
    }

    #[test]
    fn latency_encoder_encode_step_with_modulators_matches_encode() {
        let mut encoder = LatencyEncoder::new(10, (0.0, 1.0));
        let curves = NeuromodulatorGainCurves::default();
        let mods = NeuroModulators::default();

        let batch = encoder.encode_with_modulators(&[0.5], &mods, &curves);
        let step = encoder.encode_step_with_modulators(&[0.5], &mods, &curves);

        assert_eq!(batch, step);
    }

    #[test]
    fn latency_encoder_modulators_zero_scale_maps_to_zero() {
        let mut encoder = LatencyEncoder::new(10, (0.0, 1.0));
        let curves = NeuromodulatorGainCurves {
            dopamine: ModulatorGainCurves {
                latency: Some(GainCurve::new((0.0, 1.0), (1.0, 0.0))),
                ..Default::default()
            },
            ..Default::default()
        };
        let mods = NeuroModulators {
            dopamine: 1.0,
            ..Default::default()
        };

        let output = encoder.encode_with_modulators(&[0.5, f32::NAN], &mods, &curves);
        assert_eq!(output.spikes.len(), 2);
        assert!(output.spikes.iter().all(|s| s.timestamp == 0));
    }
}
