use crate::prelude::*;
use std::collections::VecDeque;

/// Encodes temporal patterns by tracking history of values per channel.
///
/// Fires a spike when the rate of change exceeds configurable thresholds.
/// Useful for detecting sudden changes or motion in sensor signals.
///
/// # Mathematical Model
///
/// Computes the difference between recent average (last 3 values) and older average
/// (previous 3 values before that). A spike is generated when this change exceeds
/// the threshold:
///
/// ```text
/// change = |mean(history[-3:]) - mean(history[-6:-3])|
/// spike if change > threshold
/// ```
///
/// # When to Use
///
/// - Detecting sudden changes in signal (edge detection)
/// - Motion detection in video or sensor streams
/// - Event-based encoding where changes are more important than absolute values
///
/// # Parameters
///
/// - `history_depth`: How many past values to track per channel
/// - `change_thresholds`: Vec of (threshold, spike_value) pairs - fires when change exceeds threshold
/// - `num_channels`: Number of input channels
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
pub struct TemporalEncoder {
    history: Vec<VecDeque<f32>>,
    history_depth: usize,
    change_thresholds: Vec<(f32, u16)>,
}

impl TemporalEncoder {
    /// Creates a new `TemporalEncoder`
    ///
    /// # Panics
    ///
    /// Panics if `history_depth < 6`
    pub fn new(
        history_depth: usize,
        change_thresholds: Vec<(f32, u16)>,
        num_channels: usize,
    ) -> Self {
        assert!(history_depth >= 6, "history_depth must be at least 6");
        Self {
            history: vec![VecDeque::with_capacity(history_depth); num_channels],
            history_depth,
            change_thresholds,
        }
    }

    fn encode_with_threshold_scale(
        &mut self,
        input: &[f32],
        threshold_scale: f32,
    ) -> EncodedOutput {
        let mut output = EncodedOutput::new();
        for (i, &value) in input.iter().enumerate() {
            if i >= self.history.len() {
                break;
            }
            let Ok(channel) = u16::try_from(i) else {
                // Remaining channels exceed u16::MAX; stop rather than wrap.
                break;
            };
            let channel_history = &mut self.history[i];
            if channel_history.len() == self.history_depth {
                channel_history.pop_front();
            }
            channel_history.push_back(value);

            if channel_history.len() < 6 {
                continue;
            }

            let recent_avg = channel_history.iter().rev().take(3).sum::<f32>() / 3.0;
            let older_avg = channel_history.iter().rev().skip(3).take(3).sum::<f32>() / 3.0;
            let change = (recent_avg - older_avg).abs();

            for &(threshold, _spike_val) in self.change_thresholds.iter().rev() {
                if change > (threshold * threshold_scale).max(0.0) {
                    output.spikes.push(SpikeEvent {
                        channel,
                        timestamp: 0,   // Simplified
                        polarity: true, // Or use spike_val to determine polarity/strength
                    });
                    break; // Only fire one spike per channel per step
                }
            }
        }
        output
    }

    /// Encodes input using neuromodulator-driven gain curves.
    ///
    /// Inherent wrapper so callers need not import [`ModulatedEncoder`].
    pub fn encode_with_modulators(
        &mut self,
        input: &[f32],
        modulators: &NeuroModulators,
        gain_curves: &NeuromodulatorGainCurves,
    ) -> EncodedOutput {
        <Self as ModulatedEncoder>::encode_with_modulators(self, input, modulators, gain_curves)
    }

    /// Step-wise variant of [`encode_with_modulators`](Self::encode_with_modulators).
    pub fn encode_step_with_modulators(
        &mut self,
        input: &[f32],
        modulators: &NeuroModulators,
        gain_curves: &NeuromodulatorGainCurves,
    ) -> EncodedOutput {
        <Self as ModulatedEncoder>::encode_step_with_modulators(
            self,
            input,
            modulators,
            gain_curves,
        )
    }
}

impl Encoder for TemporalEncoder {
    fn encode(&mut self, input: &[f32]) -> EncodedOutput {
        self.encode_with_threshold_scale(input, 1.0)
    }

    fn encode_step(&mut self, input: &[f32]) -> EncodedOutput {
        let safe_input = if input.len() > self.history.len() {
            &input[..self.history.len()]
        } else {
            input
        };
        self.encode_with_threshold_scale(safe_input, 1.0)
    }

    fn reset(&mut self) {
        for history in self.history.iter_mut() {
            history.clear();
        }
    }
}

impl ModulatedEncoder for TemporalEncoder {
    fn encode_with_gains(&mut self, input: &[f32], gains: EncodingGains) -> EncodedOutput {
        let safe_input = if input.len() > self.history.len() {
            &input[..self.history.len()]
        } else {
            input
        };
        self.encode_with_threshold_scale(safe_input, gains.sanitize().threshold_scale)
    }
}

#[cfg(feature = "serde")]
impl<'de> serde::Deserialize<'de> for TemporalEncoder {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        use std::collections::VecDeque;

        #[derive(serde::Deserialize)]
        struct Helper {
            history: Vec<VecDeque<f32>>,
            history_depth: usize,
            change_thresholds: Vec<(f32, u16)>,
        }

        let helper = Helper::deserialize(deserializer)?;

        if helper.history_depth < 6 {
            return Err(serde::de::Error::custom("history_depth must be at least 6"));
        }

        for (i, deque) in helper.history.iter().enumerate() {
            if deque.len() > helper.history_depth {
                return Err(serde::de::Error::custom(format!(
                    "history channel {} length ({}) exceeds history_depth ({})",
                    i,
                    deque.len(),
                    helper.history_depth
                )));
            }
        }

        Ok(Self {
            history: helper.history,
            history_depth: helper.history_depth,
            change_thresholds: helper.change_thresholds,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_temporal_encoder() {
        let mut encoder = TemporalEncoder::new(6, vec![(2.0, 1), (5.0, 2)], 1);
        let _output = encoder.encode(&[1.0]);
        let _output = encoder.encode(&[1.0]);
        let _output = encoder.encode(&[1.0]);
        let _output = encoder.encode(&[8.0]);
        let _output = encoder.encode(&[8.0]);
        let output = encoder.encode(&[8.0]);
        assert!(!output.spikes.is_empty());
    }

    #[test]
    fn test_temporal_encoder_modulators_reduce_threshold() {
        let mut encoder = TemporalEncoder::new(6, vec![(4.5, 1)], 1);
        let modulators = NeuroModulators {
            tempo: 1.0,
            ..Default::default()
        };
        let gain_curves = NeuromodulatorGainCurves {
            tempo: ModulatorGainCurves {
                threshold: Some(GainCurve::new((0.0, 1.0), (1.0, 0.5))),
                ..Default::default()
            },
            ..Default::default()
        };

        for _ in 0..3 {
            encoder.encode(&[1.0]);
        }
        for _ in 0..2 {
            encoder.encode(&[5.0]);
        }
        assert!(encoder.encode(&[5.0]).spikes.is_empty());

        encoder.reset();

        for _ in 0..3 {
            encoder.encode_step_with_modulators(&[1.0], &modulators, &gain_curves);
        }
        for _ in 0..2 {
            encoder.encode_step_with_modulators(&[5.0], &modulators, &gain_curves);
        }
        let output = encoder.encode_step_with_modulators(&[5.0], &modulators, &gain_curves);
        assert_eq!(output.spikes.len(), 1);
    }

    #[test]
    fn test_temporal_encoder_encode_with_modulators() {
        let mut encoder = TemporalEncoder::new(6, vec![(4.5, 1)], 1);
        let modulators = NeuroModulators {
            tempo: 1.0,
            ..Default::default()
        };
        let gain_curves = NeuromodulatorGainCurves {
            tempo: ModulatorGainCurves {
                threshold: Some(GainCurve::new((0.0, 1.0), (1.0, 0.5))),
                ..Default::default()
            },
            ..Default::default()
        };

        for _ in 0..3 {
            encoder.encode_with_modulators(&[1.0], &modulators, &gain_curves);
        }
        for _ in 0..2 {
            encoder.encode_with_modulators(&[5.0], &modulators, &gain_curves);
        }
        let output = encoder.encode_with_modulators(&[5.0], &modulators, &gain_curves);
        assert_eq!(output.spikes.len(), 1);
    }

    #[test]
    fn test_temporal_encoder_step_longer_input() {
        let mut encoder = TemporalEncoder::new(6, vec![(4.5, 1)], 2);
        let output = encoder.encode_step(&[1.0, 2.0, 3.0]);
        assert!(output.spikes.len() <= 2);
    }

    #[cfg(feature = "serde")]
    #[test]
    fn test_temporal_serde_history_channel_too_long() {
        let json = r#"{
            "history": [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
            "history_depth": 6,
            "change_thresholds": []
        }"#;
        let res: Result<TemporalEncoder, _> = serde_json::from_str(json);
        assert!(res.is_err());
    }
}
