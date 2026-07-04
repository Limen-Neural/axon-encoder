use crate::prelude::*;
use std::collections::VecDeque;

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
    /// # Panics
    ///
    /// Panics if `history_depth < 5`.
    pub fn new(
        history_depth: usize,
        deviation_thresholds: Vec<(f32, u16)>,
        num_channels: usize,
    ) -> Self {
        assert!(history_depth >= 5, "history_depth must be at least 5");
        Self {
            history: vec![VecDeque::with_capacity(history_depth); num_channels],
            thresholds: vec![0.0; num_channels],
            history_depth,
            deviation_thresholds,
        }
    }
}

impl Encoder for PredictiveEncoder {
    fn encode(&mut self, input: &[f32]) -> EncodedOutput {
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
                if deviation > threshold {
                    output.spikes.push(SpikeEvent {
                        channel: i as u16,
                        timestamp: 0,   // Simplified
                        polarity: true, // Indicates a deviation spike
                    });
                    break;
                }
            }
        }
        output
    }

    fn encode_step(&mut self, input: &[f32]) -> EncodedOutput {
        let safe_input = if input.len() > self.history.len() {
            &input[..self.history.len()]
        } else {
            input
        };
        self.encode(safe_input)
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
    fn test_predictive_encoder() {
        let mut encoder = PredictiveEncoder::new(5, vec![(2.0, 1)], 1);
        encoder.encode(&[1.0]);
        encoder.encode(&[1.0]);
        encoder.encode(&[1.0]);
        encoder.encode(&[1.0]);
        encoder.encode(&[1.0]);
        let output = encoder.encode(&[10.0]);
        assert!(!output.spikes.is_empty());
    }
}
