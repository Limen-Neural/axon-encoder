use crate::prelude::*;
use std::collections::VecDeque;
use std::fmt;

/// Errors that can occur when initializing a [`PredictiveEncoder`].
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PredictiveEncoderError {
    /// `history_depth` was less than 5 (the minimum window used by the predictor).
    HistoryDepthTooSmall,
}

impl fmt::Display for PredictiveEncoderError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::HistoryDepthTooSmall => write!(f, "history_depth must be at least 5"),
        }
    }
}

impl std::error::Error for PredictiveEncoderError {}

/// Encodes based on predictive deviation from expected values.
///
/// Maintains a running average (threshold) and fires a spike when the input
/// deviates significantly from this prediction. Useful for detecting anomalies
/// or unexpected changes in sensor data.
///
/// # Mathematical Model
///
/// Tracks an exponentially weighted moving average of recent values per channel.
/// A spike fires when the absolute deviation from this predicted value exceeds
/// the threshold:
///
/// ```text
/// threshold[i] = 0.9 * threshold[i] + 0.1 * mean(history[-5:])
/// deviation = |value - threshold[i]|
/// spike if deviation > threshold
/// ```
///
/// # When to Use
///
/// - Anomaly detection in sensor streams
/// - Learning patterns and detecting deviations
/// - Adaptive encoding that adjusts to baseline activity
///
/// # Parameters
///
/// - `history_depth`: Number of past values to track per channel
/// - `deviation_thresholds`: Vec of (threshold, spike_value) pairs
/// - `num_channels`: Number of input channels
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
pub struct PredictiveEncoder {
    history: Vec<VecDeque<f32>>,
    thresholds: Vec<f32>,
    history_depth: usize,
    deviation_thresholds: Vec<(f32, u16)>,
}

impl PredictiveEncoder {
    /// Creates a new `PredictiveEncoder`.
    ///
    /// # Errors
    ///
    /// Returns [`PredictiveEncoderError::HistoryDepthTooSmall`] if `history_depth < 5`.
    pub fn new(
        history_depth: usize,
        deviation_thresholds: Vec<(f32, u16)>,
        num_channels: usize,
    ) -> Result<Self, PredictiveEncoderError> {
        if history_depth < 5 {
            return Err(PredictiveEncoderError::HistoryDepthTooSmall);
        }
        Ok(Self {
            history: vec![VecDeque::with_capacity(history_depth); num_channels],
            thresholds: vec![0.0; num_channels],
            history_depth,
            deviation_thresholds,
        })
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
            let channel_history = &mut self.history[i];
            if channel_history.len() == self.history_depth {
                channel_history.pop_front();
            }
            channel_history.push_back(value);

            if channel_history.len() < 5 {
                continue;
            }

            let recent_avg = channel_history.iter().rev().take(5).sum::<f32>() / 5.0;
            self.thresholds[i] = 0.9 * self.thresholds[i] + 0.1 * recent_avg;

            let deviation = (value - self.thresholds[i]).abs();

            for &(threshold, _spike_val) in self.deviation_thresholds.iter().rev() {
                if deviation > (threshold * threshold_scale).max(0.0) {
                    let Ok(channel) = u16::try_from(i) else {
                        break;
                    };
                    output.spikes.push(SpikeEvent {
                        channel,
                        timestamp: 0,   // Simplified
                        polarity: true, // Indicates a deviation spike
                    });
                    break;
                }
            }
        }
        output
    }

    /// Encode input using neuromodulator-driven gain curves.
    ///
    /// Evaluates `gain_curves` against the current `modulators` to produce
    /// an [`EncodingGains`], then uses the `threshold_scale` component to
    /// modulate the deviation detection threshold. Values > 1.0 raise the
    /// effective threshold (less sensitive — larger deviations required to
    /// spike); values in (0, 1) lower it (more sensitive).
    ///
    /// Input is truncated to the number of tracked channels. Expected
    /// modulator range: any finite f32. Expected gain range after
    /// sanitization: `[0.0, 10,000.0]`.
    pub fn encode_with_modulators(
        &mut self,
        input: &[f32],
        modulators: &NeuroModulators,
        gain_curves: &NeuromodulatorGainCurves,
    ) -> EncodedOutput {
        // Match encode_step_with_modulators: only process channels we track.
        let safe_input = if input.len() > self.history.len() {
            &input[..self.history.len()]
        } else {
            input
        };
        let gains = gain_curves.evaluate(modulators);
        self.encode_with_threshold_scale(safe_input, gains.threshold_scale)
    }

    /// Step-wise variant of [`encode_with_modulators`](Self::encode_with_modulators).
    /// Identical behavior, provided for API symmetry with the [`Encoder`] trait.
    pub fn encode_step_with_modulators(
        &mut self,
        input: &[f32],
        modulators: &NeuroModulators,
        gain_curves: &NeuromodulatorGainCurves,
    ) -> EncodedOutput {
        let safe_input = if input.len() > self.history.len() {
            &input[..self.history.len()]
        } else {
            input
        };
        let gains = gain_curves.evaluate(modulators);
        self.encode_with_threshold_scale(safe_input, gains.threshold_scale)
    }
}

impl Encoder for PredictiveEncoder {
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
        for threshold in self.thresholds.iter_mut() {
            *threshold = 0.0;
        }
    }
}

#[cfg(feature = "serde")]
impl<'de> serde::Deserialize<'de> for PredictiveEncoder {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        use std::collections::VecDeque;

        #[derive(serde::Deserialize)]
        struct Helper {
            history: Vec<VecDeque<f32>>,
            thresholds: Vec<f32>,
            history_depth: usize,
            deviation_thresholds: Vec<(f32, u16)>,
        }

        let helper = Helper::deserialize(deserializer)?;

        if helper.history.len() != helper.thresholds.len() {
            return Err(serde::de::Error::custom(format!(
                "mismatched history length ({}) and thresholds length ({})",
                helper.history.len(),
                helper.thresholds.len()
            )));
        }

        if helper.history_depth < 5 {
            return Err(serde::de::Error::custom("history_depth must be at least 5"));
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
            thresholds: helper.thresholds,
            history_depth: helper.history_depth,
            deviation_thresholds: helper.deviation_thresholds,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_predictive_encoder_rejects_small_history_depth() {
        let err = PredictiveEncoder::new(4, vec![(2.0, 1)], 1).err();
        assert_eq!(err, Some(PredictiveEncoderError::HistoryDepthTooSmall));
        assert_eq!(
            PredictiveEncoderError::HistoryDepthTooSmall.to_string(),
            "history_depth must be at least 5"
        );
        assert!(PredictiveEncoder::new(5, vec![(2.0, 1)], 1).is_ok());
        assert!(PredictiveEncoder::new(0, vec![(2.0, 1)], 1).is_err());
    }

    #[test]
    fn test_predictive_encoder() {
        let mut encoder =
            PredictiveEncoder::new(5, vec![(2.0, 1)], 1).expect("valid PredictiveEncoder");
        let _output = encoder.encode(&[1.0]);
        let _output = encoder.encode(&[1.0]);
        let _output = encoder.encode(&[1.0]);
        let _output = encoder.encode(&[1.0]);
        let _output = encoder.encode(&[1.0]);
        let output = encoder.encode(&[10.0]);
        assert!(!output.spikes.is_empty());
    }

    #[test]
    fn test_predictive_encoder_reset() {
        let mut encoder =
            PredictiveEncoder::new(5, vec![(2.0, 1)], 2).expect("valid PredictiveEncoder");
        for _ in 0..6 {
            encoder.encode(&[1.0, 2.0]);
        }
        encoder.reset();
        assert!(encoder.history.iter().all(|h| h.is_empty()));
        assert!(encoder.thresholds.iter().all(|&t| t == 0.0));
    }

    #[test]
    fn test_predictive_encoder_multi_channel() {
        let mut encoder =
            PredictiveEncoder::new(5, vec![(2.0, 1)], 3).expect("valid PredictiveEncoder");
        for _ in 0..6 {
            encoder.encode(&[1.0, 2.0, 3.0]);
        }
        let output = encoder.encode(&[10.0, 20.0, 30.0]);
        // All channels should spike on large deviation
        assert!(!output.spikes.is_empty());
    }

    #[test]
    fn test_predictive_encoder_input_truncation() {
        let mut encoder =
            PredictiveEncoder::new(5, vec![(2.0, 1)], 2).expect("valid PredictiveEncoder");
        for _ in 0..6 {
            // 4 values but only 2 channels tracked — should truncate
            encoder.encode(&[1.0, 2.0, 3.0, 4.0]);
        }
        // Should not panic; only first 2 channels processed
        let output = encoder.encode(&[10.0, 20.0, 30.0, 40.0]);
        assert!(output.spikes.len() <= 2);
    }

    #[test]
    fn test_predictive_encoder_step_input_truncation() {
        let mut encoder =
            PredictiveEncoder::new(5, vec![(2.0, 1)], 2).expect("valid PredictiveEncoder");
        for _ in 0..6 {
            encoder.encode_step(&[1.0, 2.0, 3.0]);
        }
        let output = encoder.encode_step(&[10.0, 20.0, 30.0]);
        assert!(output.spikes.len() <= 2);
    }

    #[test]
    fn test_predictive_encoder_encode_with_modulators() {
        let mut encoder =
            PredictiveEncoder::new(5, vec![(5.0, 1)], 1).expect("valid PredictiveEncoder");
        let mods = NeuroModulators {
            acetylcholine: 1.0,
            ..Default::default()
        };
        let curves = NeuromodulatorGainCurves {
            acetylcholine: ModulatorGainCurves {
                threshold: Some(GainCurve::new((0.0, 1.0), (1.0, 0.5))),
                ..Default::default()
            },
            ..Default::default()
        };
        for _ in 0..5 {
            encoder.encode_with_modulators(&[1.0], &mods, &curves);
        }
        let output = encoder.encode_with_modulators(&[5.0], &mods, &curves);
        assert_eq!(output.spikes.len(), 1);
    }

    #[test]
    fn test_predictive_encoder_modulators_reduce_threshold() {
        let mut encoder =
            PredictiveEncoder::new(5, vec![(5.0, 1)], 1).expect("valid PredictiveEncoder");
        let modulators = NeuroModulators {
            acetylcholine: 1.0,
            ..Default::default()
        };
        let gain_curves = NeuromodulatorGainCurves {
            acetylcholine: ModulatorGainCurves {
                threshold: Some(GainCurve::new((0.0, 1.0), (1.0, 0.5))),
                ..Default::default()
            },
            ..Default::default()
        };

        for _ in 0..5 {
            encoder.encode(&[1.0]);
        }
        assert!(encoder.encode(&[5.0]).spikes.is_empty());

        encoder.reset();

        for _ in 0..5 {
            encoder.encode_step_with_modulators(&[1.0], &modulators, &gain_curves);
        }
        let output = encoder.encode_step_with_modulators(&[5.0], &modulators, &gain_curves);
        assert_eq!(output.spikes.len(), 1);
    }

    #[test]
    fn test_predictive_encoder_encode_with_modulators_truncate() {
        let mut encoder =
            PredictiveEncoder::new(5, vec![(5.0, 1)], 1).expect("valid PredictiveEncoder");
        let mods = NeuroModulators {
            acetylcholine: 1.0,
            ..Default::default()
        };
        let curves = NeuromodulatorGainCurves {
            acetylcholine: ModulatorGainCurves {
                threshold: Some(GainCurve::new((0.0, 1.0), (1.0, 0.5))),
                ..Default::default()
            },
            ..Default::default()
        };
        for _ in 0..5 {
            encoder.encode_with_modulators(&[1.0, 2.0], &mods, &curves);
        }
        let output = encoder.encode_with_modulators(&[5.0, 6.0], &mods, &curves);
        assert_eq!(output.spikes.len(), 1);
    }

    #[test]
    fn test_predictive_encoder_step_shorter_input() {
        let mut encoder =
            PredictiveEncoder::new(5, vec![(2.0, 1)], 2).expect("valid PredictiveEncoder");
        for _ in 0..6 {
            encoder.encode_step(&[1.0]);
        }
        let output = encoder.encode_step(&[10.0]);
        assert!(!output.spikes.is_empty());
    }

    #[cfg(feature = "serde")]
    #[test]
    fn test_predictive_serde_history_channel_too_long() {
        let json = r#"{
            "history": [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
            "thresholds": [0.0],
            "history_depth": 5,
            "deviation_thresholds": []
        }"#;
        let res: Result<PredictiveEncoder, _> = serde_json::from_str(json);
        assert!(res.is_err());
    }
}
