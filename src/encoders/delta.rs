use crate::prelude::*;

/// A simple delta-based encoder.
///
/// Fires a spike when the absolute difference between the current input and the last
/// encoded value exceeds a threshold. This is useful for event-based encoding where
/// only changes in the input signal are relevant.
///
/// # Mathematical Model
///
/// ```text
/// delta = |current_value - last_value|
/// spike if delta > threshold
/// ```
///
/// # When to Use
///
/// - Event-based encoding where changes are more important than absolute values
/// - Sensor data where baseline can drift but changes are meaningful
/// - Reducing power consumption by only encoding when changes occur
///
/// # Parameters
///
/// - `threshold`: Minimum change required to trigger a spike
/// - `num_channels`: Number of input channels to track
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct DeltaEncoder {
    last_values: Vec<f32>,
    threshold: f32,
}

impl DeltaEncoder {
    pub fn new(threshold: f32, num_channels: usize) -> Self {
        Self {
            last_values: vec![0.0; num_channels],
            threshold,
        }
    }

    fn encode_with_threshold_scale(
        &mut self,
        input: &[f32],
        threshold_scale: f32,
    ) -> EncodedOutput {
        let mut output = EncodedOutput::new();
        let effective_threshold = (self.threshold * threshold_scale).max(0.0);

        for (i, &value) in input.iter().enumerate() {
            if i >= self.last_values.len() {
                break;
            }
            let delta = (value - self.last_values[i]).abs();
            if delta > effective_threshold {
                output.spikes.push(SpikeEvent {
                    channel: u16::try_from(i).expect("channel index exceeds u16::MAX"),
                    timestamp: 0,
                    polarity: value > self.last_values[i],
                });
                self.last_values[i] = value;
            }
        }
        output
    }

    /// Encode input using neuromodulator-driven gain curves.
    ///
    /// Evaluates `gain_curves` against the current `modulators` to produce
    /// an [`EncodingGains`], then uses the `threshold_scale` component to
    /// modulate the delta-change detection threshold. Values > 1.0 raise the
    /// effective threshold (less sensitive — larger changes required to spike);
    /// values in (0, 1) lower it (more sensitive).
    ///
    /// `NeuromodulatorGainCurves` composes per-modulator curves
    /// (dopamine, cortisol, acetylcholine, tempo) multiplicatively via
    /// [`EncodingGains::sanitize`]. Expected modulator range: any finite f32.
    /// Expected gain output range: `[MIN_GAIN_SCALE, MAX_GAIN_SCALE]`
    /// (0.0 to 10,000.0 after sanitization).
    pub fn encode_with_modulators(
        &mut self,
        input: &[f32],
        modulators: &NeuroModulators,
        gain_curves: &NeuromodulatorGainCurves,
    ) -> EncodedOutput {
        let gains = gain_curves.evaluate(modulators);
        self.encode_with_threshold_scale(input, gains.threshold_scale)
    }

    /// Step-wise variant of [`encode_with_modulators`](Self::encode_with_modulators).
    ///
    /// Identical behavior — provided for API symmetry with the [`Encoder`] trait's
    /// `encode` / `encode_step` pair.
    pub fn encode_step_with_modulators(
        &mut self,
        input: &[f32],
        modulators: &NeuroModulators,
        gain_curves: &NeuromodulatorGainCurves,
    ) -> EncodedOutput {
        let gains = gain_curves.evaluate(modulators);
        self.encode_with_threshold_scale(input, gains.threshold_scale)
    }
}

impl Encoder for DeltaEncoder {
    fn encode(&mut self, input: &[f32]) -> EncodedOutput {
        self.encode_with_threshold_scale(input, 1.0)
    }

    fn encode_step(&mut self, input: &[f32]) -> EncodedOutput {
        let safe_input = if input.len() > self.last_values.len() {
            &input[..self.last_values.len()]
        } else {
            input
        };
        self.encode(safe_input)
    }

    fn reset(&mut self) {
        for val in self.last_values.iter_mut() {
            *val = 0.0;
        }
    }
}

/// Simplified: delta-based spike generation (per feature)
///
/// This is a utility function that takes a slice of deltas and returns a boolean spike train.
/// It can be used to feed the resulting binary/event sequences into LIF/RSNN layers.
pub fn encode_deltas_to_spikes(deltas: &[f32], threshold: f32) -> Vec<bool> {
    deltas.iter().map(|&d| d.abs() > threshold).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_delta_encoder() {
        let mut encoder = DeltaEncoder::new(2.0, 1);
        let output = encoder.encode(&[1.0]); // 1.0 - 0.0 = 1.0 < 2.0 -> no spike
        assert!(output.spikes.is_empty());

        let output = encoder.encode(&[3.5]); // 3.5 - 0.0 = 3.5 > 2.0 -> spike
        assert!(!output.spikes.is_empty());
        assert!(output.spikes[0].polarity);

        let output = encoder.encode(&[4.0]); // 4.0 - 3.5 = 0.5 < 2.0 -> no spike
        assert!(output.spikes.is_empty());

        let output = encoder.encode(&[1.0]); // 1.0 - 3.5 = -2.5.abs() = 2.5 > 2.0 -> spike
        assert!(!output.spikes.is_empty());
        assert!(!output.spikes[0].polarity);
    }

    #[test]
    fn test_delta_encoder_encode_step() {
        let mut encoder = DeltaEncoder::new(2.0, 2);
        let output = encoder.encode_step(&[3.0, 3.0, 3.0]); // 3rd channel ignored
        assert_eq!(output.spikes.len(), 2);
    }

    #[test]
    fn test_delta_encoder_multi_channel_reset() {
        let mut encoder = DeltaEncoder::new(1.0, 2);
        encoder.encode(&[2.0, 2.0]);
        assert_eq!(encoder.last_values, vec![2.0, 2.0]);
        encoder.reset();
        assert_eq!(encoder.last_values, vec![0.0, 0.0]);
    }

    #[test]
    fn test_delta_encoder_empty_input() {
        let mut encoder = DeltaEncoder::new(1.0, 5);
        let output = encoder.encode(&[]);
        assert!(output.spikes.is_empty());
    }

    #[test]
    fn test_encode_deltas_to_spikes() {
        let deltas = [0.1, 0.5, -0.8, 1.2];
        let threshold = 0.7;
        let spikes = encode_deltas_to_spikes(&deltas, threshold);
        assert_eq!(spikes, vec![false, false, true, true]);
    }

    #[test]
    fn test_delta_encoder_modulators_reduce_threshold() {
        let mut encoder = DeltaEncoder::new(1.0, 1);
        let modulators = NeuroModulators {
            dopamine: 1.0,
            ..Default::default()
        };
        let gain_curves = NeuromodulatorGainCurves {
            dopamine: ModulatorGainCurves {
                threshold: Some(GainCurve::new((0.0, 1.0), (1.0, 0.5))),
                ..Default::default()
            },
            ..Default::default()
        };

        assert!(encoder.encode(&[0.75]).spikes.is_empty());
        encoder.reset();

        let modulated = encoder.encode_with_modulators(&[0.75], &modulators, &gain_curves);
        assert_eq!(modulated.spikes.len(), 1);
    }
}
