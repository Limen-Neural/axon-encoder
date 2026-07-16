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

    /// Effective tuning width under a sensitivity gain.
    ///
    /// Scales **≥ 1** narrow the Gaussian (`width / scale`) so high sensitivity is
    /// more selective. Scales in **(0, 1)** keep the base width and rely on rate
    /// scaling in `encode_with_sensitivity_scale` so low (but nonzero) gain
    /// *suppresses* activity instead of widening toward universal firing.
    fn effective_tuning_width(&self, sensitivity_scale: f32) -> f32 {
        if !sensitivity_scale.is_finite() || sensitivity_scale <= 0.0 {
            return self.tuning_width.max(f32::EPSILON);
        }
        if sensitivity_scale >= 1.0 {
            return (self.tuning_width / sensitivity_scale).max(f32::EPSILON);
        }
        // Sub-unity: do not widen; rate scaling handles suppression.
        self.tuning_width.max(f32::EPSILON)
    }

    fn encode_with_sensitivity_scale(
        &mut self,
        input: &[f32],
        sensitivity_scale: f32,
    ) -> EncodedOutput {
        let mut output = EncodedOutput::new();
        // Zero/negative/non-finite sensitivity fully suppresses population responses.
        if !sensitivity_scale.is_finite() || sensitivity_scale <= 0.0 {
            return output;
        }
        let tuning_width = self.effective_tuning_width(sensitivity_scale);
        // Rate gain: scales > 1 also narrow width; scales in (0, 1) only reduce rate
        // so small positive gains never produce near-universal firing.
        let rate_gain = sensitivity_scale.min(1.0);

        // This encoder expects a single value in the input slice
        if let Some(&value) = input.first() {
            let mut rng = rand::rng();
            for i in 0..self.num_neurons {
                let Ok(channel) = u16::try_from(i) else {
                    // Remaining neurons exceed u16::MAX; stop rather than wrap.
                    break;
                };
                let rate = self.get_rate_with_tuning_width(value, i, tuning_width) * rate_gain;
                if crate::rng::gen_unit_f32_with_rng(&mut rng) < rate {
                    output.spikes.push(SpikeEvent {
                        channel,
                        timestamp: 0, // Simplified
                        polarity: true,
                    });
                }
            }
        }
        output
    }

    /// Encode input using neuromodulator-driven gain curves.
    ///
    /// Evaluates `gain_curves` against the current `modulators` to produce
    /// an [`EncodingGains`], then uses the `sensitivity_scale` component to
    /// modulate the population tuning width. Values >= 1.0 narrow the
    /// Gaussian (more selective); values in (0, 1) suppress firing rate
    /// without widening.
    ///
    /// Expected modulator range: any finite f32. Expected gain range after
    /// sanitization: `[0.0, 10,000.0]`.
    pub fn encode_with_modulators(
        &mut self,
        input: &[f32],
        modulators: &NeuroModulators,
        gain_curves: &NeuromodulatorGainCurves,
    ) -> EncodedOutput {
        let gains = gain_curves.evaluate(modulators);
        self.encode_with_sensitivity_scale(input, gains.sensitivity_scale)
    }

    /// Step-wise variant of [`encode_with_modulators`](Self::encode_with_modulators).
    /// Identical behavior, provided for API symmetry with the [`Encoder`] trait.
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
    fn test_population_encoder_empty_input() {
        let mut encoder = PopulationEncoder::new(10, (0.0, 100.0), 10.0);
        let empty: [f32; 0] = [];
        let via_encode = encoder.encode(&empty);
        assert!(
            via_encode.spikes.is_empty(),
            "empty input must yield no spikes through encode"
        );
        let via_scale = encoder.encode_with_sensitivity_scale(&empty, 1.0);
        assert!(
            via_scale.spikes.is_empty(),
            "empty input must yield no spikes through encode_with_sensitivity_scale"
        );
        let via_step = encoder.encode_step(&empty);
        assert!(via_step.spikes.is_empty());
    }

    #[test]
    fn test_effective_tuning_width_sub_unity() {
        let encoder = PopulationEncoder::new(10, (0.0, 100.0), 10.0);
        // Sub-unity sensitivity should NOT widen the tuning width
        let width = encoder.effective_tuning_width(0.5);
        assert_eq!(width, encoder.tuning_width.max(f32::EPSILON));
    }

    #[test]
    fn test_effective_tuning_width_zero_and_negative() {
        let encoder = PopulationEncoder::new(10, (0.0, 100.0), 10.0);
        assert_eq!(
            encoder.effective_tuning_width(0.0),
            encoder.tuning_width.max(f32::EPSILON)
        );
        assert_eq!(
            encoder.effective_tuning_width(-1.0),
            encoder.tuning_width.max(f32::EPSILON)
        );
        assert_eq!(
            encoder.effective_tuning_width(f32::NAN),
            encoder.tuning_width.max(f32::EPSILON)
        );
    }

    #[test]
    fn test_encode_with_zero_sensitivity_returns_empty() {
        let mut encoder = PopulationEncoder::new(10, (0.0, 100.0), 10.0);
        let output = encoder.encode_with_sensitivity_scale(&[50.0], 0.0);
        assert!(output.spikes.is_empty());
    }

    #[test]
    fn test_encode_with_negative_sensitivity_returns_empty() {
        let mut encoder = PopulationEncoder::new(10, (0.0, 100.0), 10.0);
        let output = encoder.encode_with_sensitivity_scale(&[50.0], -1.0);
        assert!(output.spikes.is_empty());
    }

    #[test]
    fn test_encode_with_nan_sensitivity_returns_empty() {
        let mut encoder = PopulationEncoder::new(10, (0.0, 100.0), 10.0);
        let output = encoder.encode_with_sensitivity_scale(&[50.0], f32::NAN);
        assert!(output.spikes.is_empty());
    }

    #[test]
    fn test_sub_unity_sensitivity_suppresses_firing() {
        let encoder = PopulationEncoder::new(10, (0.0, 100.0), 10.0);
        // Sub-unity scale should NOT widen tuning width (that's handled by effective_tuning_width)
        // but the rate_gain = scale.min(1.0) should suppress firing probability.
        let baseline_width = encoder.effective_tuning_width(1.0);
        let suppressed_width = encoder.effective_tuning_width(0.1);
        // Widths should be equal (sub-unity doesn't widen)
        assert_eq!(baseline_width, suppressed_width);

        // Rate gain at 0.1 should be 0.1x the baseline rate
        let baseline_rate = encoder.get_rate_with_tuning_width(50.0, 5, baseline_width);
        let suppressed_rate = encoder.get_rate_with_tuning_width(50.0, 5, suppressed_width) * 0.1;
        // Suppressed rate should be substantially lower
        assert!(
            suppressed_rate < baseline_rate * 0.15,
            "suppressed_rate {} should be < 15% of baseline_rate {}",
            suppressed_rate,
            baseline_rate
        );
    }

    #[test]
    fn test_encode_with_modulators_uses_gain_curves() {
        let mut encoder = PopulationEncoder::new(10, (0.0, 100.0), 10.0);
        let mods = NeuroModulators::default();
        let curves = NeuromodulatorGainCurves::default();
        // With identity gains, should produce similar output to plain encode
        let output = encoder.encode_with_modulators(&[50.0], &mods, &curves);
        assert!(output.spikes.len() <= 10);
    }

    #[test]
    fn test_encode_step_with_modulators() {
        let mut encoder = PopulationEncoder::new(10, (0.0, 100.0), 10.0);
        let mods = NeuroModulators::default();
        let curves = NeuromodulatorGainCurves::default();
        let output = encoder.encode_step_with_modulators(&[50.0], &mods, &curves);
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

    #[test]
    fn test_population_encoder_step_and_accessors() {
        let mut encoder = PopulationEncoder::new(10, (0.0, 100.0), 10.0);
        assert_eq!(encoder.num_neurons(), 10);

        let step_output = encoder.encode_step(&[50.0]);
        assert!(step_output.spikes.len() <= 10);

        encoder.reset();
        assert_eq!(encoder.num_neurons(), 10);
    }
}
