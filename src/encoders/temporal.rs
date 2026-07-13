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
    /// Creates a new `TemporalEncoder`.
    ///
    /// # Panics
    ///
    /// Panics if `history_depth < 6`.
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
}

impl Encoder for TemporalEncoder {
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

            if channel_history.len() < 6 {
                continue;
            }

            let recent_avg = channel_history.iter().rev().take(3).sum::<f32>() / 3.0;
            let older_avg = channel_history.iter().rev().skip(3).take(3).sum::<f32>() / 3.0;
            let change = (recent_avg - older_avg).abs();

            for &(threshold, _spike_val) in self.change_thresholds.iter().rev() {
                if change > threshold {
                    output.spikes.push(SpikeEvent {
                        channel: i as u16,
                        timestamp: 0,   // Simplified
                        polarity: true, // Or use spike_val to determine polarity/strength
                    });
                    break; // Only fire one spike per channel per step
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
        encoder.encode(&[1.0]);
        encoder.encode(&[1.0]);
        encoder.encode(&[1.0]);
        encoder.encode(&[8.0]);
        encoder.encode(&[8.0]);
        let output = encoder.encode(&[8.0]);
        assert!(!output.spikes.is_empty());
    }

    #[test]
    #[should_panic(expected = "history_depth must be at least 6")]
    fn test_temporal_encoder_new_panics_on_invalid_depth() {
        let _ = TemporalEncoder::new(5, vec![(2.0, 1)], 1);
    }
}
