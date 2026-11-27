use crate::encoders::population_core::{PopulationCore, impl_population_encoder_api};
use crate::prelude::*;

/// Encodes angular input across a population of direction-tuned neurons.
///
/// Each input channel is an angle in **radians**. Angles use the standard
/// mathematical convention: zero points east and positive angles rotate
/// counterclockwise. The population's preferred directions are evenly spaced
/// over the half-open circle `[0, 2π)`, so neuron `i` prefers
/// `i * 2π / num_neurons`. For input channel `j`, neuron `i` emits on spike
/// channel `j * num_neurons + i`.
///
/// # Mathematical model
///
/// The encoder uses a von Mises tuning curve over circular distance:
///
/// ```text
/// preferred_angle[i] = i * 2π / num_neurons
/// delta = signed circular difference input - preferred_angle[i], in [-π, π)
/// concentration = 1 / tuning_width²
/// probability = exp(concentration * (cos(delta) - 1))
/// spike if random() < probability
/// ```
///
/// The tuning-curve result is a per-call firing probability, not a rate in
/// hertz. Every emitted timestamp is zero, so it contains no timing information
/// within a call. Inputs wrap naturally: `0`, `2π`, and `-2π` are the same
/// direction. Non-finite inputs produce no spike because their circular
/// distance is not finite.
///
/// # Panics
///
/// Encoding panics if `input.len() * num_neurons` exceeds 65,536, because
/// spike channel IDs are `u16`. No input channel is silently truncated.
///
/// # Examples
///
/// ```rust
/// use axon_encoder::prelude::*;
/// # fn main() -> Result<(), EncoderError> {
/// let mut enc = DirectionalPopulationEncoder::try_new(16, 0.35)?;
/// let angles = enc.preferred_angles_radians();
/// assert_eq!(angles.len(), 16);
/// let out = enc.encode_step(&[std::f32::consts::FRAC_PI_2]);
/// assert!(out.spikes.iter().all(|spike| spike.channel < 16));
/// # Ok(())
/// # }
/// ```
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
pub struct DirectionalPopulationEncoder {
    num_neurons: usize,
    tuning_width: f32,
}

impl DirectionalPopulationEncoder {
    /// Creates a new `DirectionalPopulationEncoder`, panicking if configuration is invalid.
    ///
    /// Prefer [`try_new`](Self::try_new) for typed validation errors.
    /// `tuning_width` is the von Mises width in radians (its small-angle
    /// Gaussian-equivalent width).
    pub fn new(num_neurons: usize, tuning_width: f32) -> Self {
        Self::try_new(num_neurons, tuning_width)
            .expect("invalid DirectionalPopulationEncoder configuration")
    }

    /// Creates a new `DirectionalPopulationEncoder`, returning an [`EncoderError`] for invalid configuration.
    ///
    /// `tuning_width` is the von Mises width in radians (its small-angle
    /// Gaussian-equivalent width).
    pub fn try_new(num_neurons: usize, tuning_width: f32) -> Result<Self, EncoderError> {
        if num_neurons == 0 {
            return Err(EncoderError::CountMustBePositive {
                parameter: "num_neurons",
            });
        }
        crate::error::validate_channel_count(num_neurons)?;
        if !tuning_width.is_finite() || tuning_width <= 0.0 {
            return Err(EncoderError::NonPositiveOrNonFinite {
                parameter: "tuning_width",
            });
        }
        Ok(Self {
            num_neurons,
            tuning_width,
        })
    }

    /// Returns the number of direction-tuned neurons per input channel.
    pub fn num_neurons(&self) -> usize {
        self.num_neurons
    }

    /// Preferred direction of each neuron in one population, in radians.
    ///
    /// The returned vector has length [`num_neurons`](Self::num_neurons) and is
    /// indexed by the neuron index within one input channel's population.
    /// Angles are evenly spaced over `[0, 2π)`; zero points east and positive
    /// angles rotate counterclockwise. When several input channels are encoded,
    /// this same sequence repeats for each channel: spike channel
    /// `j * num_neurons + i` has preferred angle index `i`.
    pub fn preferred_angles_radians(&self) -> Vec<f32> {
        (0..self.num_neurons)
            .map(|index| self.preferred_angle(index))
            .collect()
    }

    fn preferred_angle(&self, neuron_index: usize) -> f32 {
        neuron_index as f32 * std::f32::consts::TAU / self.num_neurons as f32
    }
}

impl PopulationCore for DirectionalPopulationEncoder {
    fn num_neurons(&self) -> usize {
        self.num_neurons
    }

    fn tuning_width(&self) -> f32 {
        self.tuning_width
    }

    fn encoder_kind(&self) -> &'static str {
        "directional population"
    }

    fn get_rate_with_tuning_width(
        &self,
        input: f32,
        neuron_index: usize,
        tuning_width: f32,
    ) -> f32 {
        let preferred_angle = self.preferred_angle(neuron_index);
        let wrapped = (input - preferred_angle).rem_euclid(std::f32::consts::TAU);
        let circular_delta = if wrapped <= std::f32::consts::PI {
            wrapped
        } else {
            wrapped - std::f32::consts::TAU
        };
        // cos(delta) - 1 catastrophically rounds to zero for small offsets.
        // The half-angle identity keeps narrow tuning curves selective there.
        let sin_half_delta = (circular_delta * 0.5).sin();
        (-2.0 * sin_half_delta * sin_half_delta / (tuning_width * tuning_width)).exp()
    }
}

impl_population_encoder_api!(DirectionalPopulationEncoder);

#[cfg(feature = "serde")]
impl<'de> serde::Deserialize<'de> for DirectionalPopulationEncoder {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(serde::Deserialize)]
        struct Helper {
            num_neurons: usize,
            tuning_width: f32,
        }
        let helper = Helper::deserialize(deserializer)?;
        Self::try_new(helper.num_neurons, helper.tuning_width).map_err(serde::de::Error::custom)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::encoders::population_core::PopulationCore;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    #[test]
    fn preferred_angles_cover_the_circle_in_neuron_order() {
        let encoder = DirectionalPopulationEncoder::new(8, 0.25);
        let angles = encoder.preferred_angles_radians();

        assert_eq!(angles.len(), 8);
        assert_eq!(angles[0], 0.0);
        for (index, &angle) in angles.iter().enumerate() {
            let expected = index as f32 * std::f32::consts::TAU / 8.0;
            assert_eq!(angle, expected);
            assert!((0.0..std::f32::consts::TAU).contains(&angle));
        }
    }

    #[test]
    fn tuning_response_peaks_at_the_preferred_angle() {
        let encoder = DirectionalPopulationEncoder::new(8, 0.25);
        let angles = encoder.preferred_angles_radians();

        for (index, &angle) in angles.iter().enumerate() {
            assert_eq!(
                encoder.get_rate_with_tuning_width(angle, index, encoder.tuning_width),
                1.0,
                "neuron {index} must peak exactly at its preferred angle"
            );
            for other in 0..encoder.num_neurons() {
                if other != index {
                    assert!(
                        encoder.get_rate_with_tuning_width(angle, other, encoder.tuning_width)
                            < 1.0
                    );
                }
            }
        }
    }

    #[test]
    fn narrow_tuning_stays_selective_at_small_offsets() {
        let encoder = DirectionalPopulationEncoder::new(16, 0.01);

        let response = encoder.get_rate_with_tuning_width(0.0001, 0, 0.01);

        assert!(
            response < 1.0,
            "small offset must not round to peak: {response}"
        );
        assert!(
            response > 0.9999,
            "half-angle response changed unexpectedly: {response}"
        );
    }

    #[test]
    fn tuning_response_wraps_across_zero() {
        let encoder = DirectionalPopulationEncoder::new(16, 0.25);
        let width = encoder.tuning_width;
        let epsilon = 0.01;

        let positive = encoder.get_rate_with_tuning_width(epsilon, 0, width);
        let wrapped_positive =
            encoder.get_rate_with_tuning_width(std::f32::consts::TAU + epsilon, 0, width);
        let negative = encoder.get_rate_with_tuning_width(-epsilon, 0, width);
        let wrapped_negative =
            encoder.get_rate_with_tuning_width(std::f32::consts::TAU - epsilon, 0, width);

        assert!((positive - wrapped_positive).abs() < 1e-6);
        assert!((negative - wrapped_negative).abs() < 1e-6);
        assert!(positive > 0.99);
        assert!(negative > 0.99);
    }

    #[test]
    fn preferred_neuron_fires_on_every_call_at_its_exact_angle() {
        let mut encoder = DirectionalPopulationEncoder::new(8, 0.25);
        let angles = encoder.preferred_angles_radians();
        let mut rng = StdRng::seed_from_u64(0x0D1E_5C70);

        for (index, &angle) in angles.iter().enumerate() {
            let out = encoder.encode_step_with_rng(&[angle], &mut rng);
            assert!(
                out.spikes.iter().any(|spike| spike.channel == index as u16),
                "preferred neuron {index} must fire at its exact preferred angle"
            );
        }
    }

    #[test]
    fn seeded_encode_with_rng_is_reproducible() {
        let inputs: &[&[f32]] = &[
            &[0.1, 2.0],
            &[std::f32::consts::TAU - 0.1, -0.2],
            &[f32::NAN, 4.0],
        ];
        let mut a = DirectionalPopulationEncoder::new(8, 0.3);
        let mut b = a.clone();
        let mut rng_a = StdRng::seed_from_u64(0xA61E5);
        let mut rng_b = StdRng::seed_from_u64(0xA61E5);

        for input in inputs {
            assert_eq!(
                a.encode_step_with_rng(input, &mut rng_a),
                b.encode_step_with_rng(input, &mut rng_b)
            );
        }
    }

    #[test]
    fn input_channels_fan_out_in_channel_major_order() {
        let mut encoder = DirectionalPopulationEncoder::new(4, 0.2);
        let mut rng = StdRng::seed_from_u64(7);
        let out = encoder.encode_step_with_rng(&[0.0, std::f32::consts::PI], &mut rng);

        assert!(out.spikes.iter().all(|spike| spike.channel < 8));
        assert!(out.spikes.iter().any(|spike| spike.channel == 0));
        assert!(out.spikes.iter().any(|spike| spike.channel == 6));
    }

    #[test]
    fn non_finite_input_and_inactive_gain_are_silent() {
        let mut encoder = DirectionalPopulationEncoder::new(8, 0.25);
        assert!(
            encoder
                .encode_step_with_rng(&[f32::NAN], &mut StdRng::seed_from_u64(1))
                .spikes
                .is_empty()
        );
        assert!(
            encoder
                .encode_step_with_rng(&[f32::INFINITY], &mut StdRng::seed_from_u64(1))
                .spikes
                .is_empty()
        );
        assert!(
            encoder
                .encode_with_sensitivity_scale(&[0.0], 0.0)
                .spikes
                .is_empty()
        );
        assert!(
            encoder
                .encode_with_sensitivity_scale(&[0.0], f32::NAN)
                .spikes
                .is_empty()
        );
    }

    #[test]
    fn sink_and_modulated_paths_write_valid_instant_spikes() {
        let mut encoder = DirectionalPopulationEncoder::new(8, 0.25);
        let mut sink = Vec::new();
        encoder.encode_step_into(&[0.0], &mut sink);
        assert!(sink.iter().all(|spike| spike.timestamp == 0));
        assert!(sink.iter().all(|spike| spike.channel < 8));

        sink.clear();
        let modulators = NeuroModulators::default();
        let curves = NeuromodulatorGainCurves::default();
        encoder.encode_step_with_modulators_into(&[0.0], &modulators, &curves, &mut sink);
        assert!(sink.iter().all(|spike| spike.timestamp == 0));
    }

    #[test]
    fn try_new_validation() {
        assert_eq!(
            DirectionalPopulationEncoder::try_new(0, 0.2).err(),
            Some(EncoderError::CountMustBePositive {
                parameter: "num_neurons"
            })
        );
        assert_eq!(
            DirectionalPopulationEncoder::try_new(u16::MAX as usize + 2, 0.2).err(),
            Some(EncoderError::NumChannelsTooLarge)
        );
        assert_eq!(
            DirectionalPopulationEncoder::try_new(8, 0.0).err(),
            Some(EncoderError::NonPositiveOrNonFinite {
                parameter: "tuning_width"
            })
        );
    }
}
