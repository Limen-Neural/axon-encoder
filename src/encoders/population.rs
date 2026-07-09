use crate::prelude::*;

/// Encodes a single analog value across a population of neurons.
///
/// Each neuron in the population is "tuned" to a specific preferred value within
/// the input range. The neuron fires based on a Gaussian-like tuning curve centered
/// on its preferred value. This creates a distributed representation where multiple
/// neurons contribute to encoding a single input value.
///
/// # Mathematical Model
///
/// Uses a Gaussian tuning curve to determine each neuron's firing rate:
///
/// ```text
/// preferred_value[i] = range_min + (i / num_neurons) * (range_max - range_min)
/// distance = |input - preferred_value[i]|
/// rate = exp(-distance² / (2 * tuning_width²))
/// spike if random() < rate
/// ```
///
/// # When to Use
///
/// - Encoding position or continuous values with distributed representation
/// - When multiple neurons should contribute to representing a single value
/// - Creating more robust encoding that doesn't rely on a single neuron
///
/// # Parameters
///
/// - `num_neurons`: Number of neurons in the population per input channel
/// - `input_range`: Tuple of (min, max) input values
/// - `tuning_width`: Controls how broadly neurons respond (larger = wider spread)
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct PopulationEncoder {
    num_neurons: usize,
    input_range: (f32, f32),
    tuning_width: f32, // Controls how broadly a neuron responds to stimuli
}

impl PopulationEncoder {
    pub fn new(num_neurons: usize, input_range: (f32, f32), tuning_width: f32) -> Self {
        Self {
            num_neurons,
            input_range,
            tuning_width,
        }
    }

    /// Returns the number of neurons in the population.
    pub fn num_neurons(&self) -> usize {
        self.num_neurons
    }

    fn get_rate_with_tuning_width(
        &self,
        input: f32,
        neuron_index: usize,
        tuning_width: f32,
    ) -> f32 {
        let range_span = self.input_range.1 - self.input_range.0;
        let preferred_value =
            self.input_range.0 + (neuron_index as f32 / self.num_neurons as f32) * range_span;

        let distance = (input - preferred_value).abs();
        // Gaussian-like response curve
        (-(distance * distance) / (2.0 * tuning_width * tuning_width)).exp()
    }

    fn effective_tuning_width(&self, sensitivity_scale: f32) -> f32 {
        let safe_scale = sensitivity_scale.max(f32::EPSILON);
        (self.tuning_width / safe_scale).max(f32::EPSILON)
    }

    fn encode_with_sensitivity_scale(
        &mut self,
        input: &[f32],
        sensitivity_scale: f32,
    ) -> EncodedOutput {
        let mut output = EncodedOutput::new();
        // Zero (or negative) sensitivity fully suppresses population responses
        // instead of widening the Gaussian toward a flat, always-on curve.
        if !sensitivity_scale.is_finite() || sensitivity_scale <= 0.0 {
            return output;
        }
        let tuning_width = self.effective_tuning_width(sensitivity_scale);

        // This encoder expects a single value in the input slice
        if let Some(&value) = input.first() {
            for i in 0..self.num_neurons {
                let rate = self.get_rate_with_tuning_width(value, i, tuning_width);
                if crate::rng::gen_unit_f32() < rate {
                    output.spikes.push(SpikeEvent {
                        channel: i as u16,
                        timestamp: 0, // Simplified
                        polarity: true,
                    });
                }
            }
        }
        output
    }

    pub fn encode_with_modulators(
        &mut self,
        input: &[f32],
        modulators: &NeuroModulators,
        gain_curves: &NeuromodulatorGainCurves,
    ) -> EncodedOutput {
        let gains = gain_curves.evaluate(modulators);
        self.encode_with_sensitivity_scale(input, gains.sensitivity_scale)
    }

    pub fn encode_step_with_modulators(
        &mut self,
        input: &[f32],
        modulators: &NeuroModulators,
        gain_curves: &NeuromodulatorGainCurves,
    ) -> EncodedOutput {
        let gains = gain_curves.evaluate(modulators);
        self.encode_with_sensitivity_scale(input, gains.sensitivity_scale)
    }
}

impl Encoder for PopulationEncoder {
    fn encode(&mut self, input: &[f32]) -> EncodedOutput {
        self.encode_with_sensitivity_scale(input, 1.0)
    }

    fn encode_step(&mut self, input: &[f32]) -> EncodedOutput {
        self.encode(input)
    }

    fn reset(&mut self) {
        // No state to reset
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_population_encoder() {
        let mut encoder = PopulationEncoder::new(10, (0.0, 100.0), 10.0);
        // Encode a value in the middle of the range.
        let input = [50.0];
        let output = encoder.encode(&input);

        // The neuron whose preferred value is closest to 50.0 should have the highest chance of firing.
        // We can't guarantee a spike due to the probabilistic nature, but we can check the rates.
        let rates: Vec<f32> = (0..10)
            .map(|i| encoder.get_rate_with_tuning_width(50.0, i, encoder.tuning_width))
            .collect();
        let max_rate_index = rates
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;

        // For a 10-neuron setup over a 0-100 range, the 5th neuron (index 4 or 5) should be near the max.
        assert!(
            max_rate_index == 4 || max_rate_index == 5,
            "Peak activity should be near the middle neuron for an input of 50."
        );
        assert!(output.spikes.len() <= 10);
    }

    #[test]
    fn test_population_encoder_modulators_adjust_sensitivity() {
        let encoder = PopulationEncoder::new(10, (0.0, 100.0), 10.0);
        let modulators = NeuroModulators {
            tempo: 1.0,
            ..Default::default()
        };
        let gain_curves = NeuromodulatorGainCurves {
            tempo: ModulatorGainCurves {
                sensitivity: Some(GainCurve::new((0.0, 1.0), (1.0, 2.0))),
                ..Default::default()
            },
            ..Default::default()
        };

        let baseline_width = encoder.effective_tuning_width(1.0);
        let modulated_width =
            encoder.effective_tuning_width(gain_curves.evaluate(&modulators).sensitivity_scale);
        let baseline_rate = encoder.get_rate_with_tuning_width(50.0, 0, baseline_width);
        let modulated_rate = encoder.get_rate_with_tuning_width(50.0, 0, modulated_width);

        assert!(modulated_width < baseline_width);
        assert!(modulated_rate < baseline_rate);
    }
}
