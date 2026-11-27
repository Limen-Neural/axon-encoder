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
/// delta = (input - preferred_angle[i]) mod 2π
/// circular_distance = min(delta, 2π - delta)
/// concentration = 1 / tuning_width²
/// probability = exp(concentration * (cos(input - preferred_angle[i]) - 1))
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

    fn response_with_tuning_width(
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
        ((circular_delta.cos() - 1.0) / (tuning_width * tuning_width)).exp()
    }

    /// Effective tuning width under a sensitivity gain.
    ///
    /// Matches [`PopulationEncoder`]: scales at or above one narrow the tuning
    /// curve, while positive sub-unity scales suppress firing probability
    /// instead of widening toward universal firing.
    fn effective_tuning_width(&self, sensitivity_scale: f32) -> f32 {
        if !sensitivity_scale.is_finite() || sensitivity_scale <= 0.0 {
            return self.tuning_width.max(f32::EPSILON);
        }
        if sensitivity_scale >= 1.0 {
            return (self.tuning_width / sensitivity_scale).max(f32::EPSILON);
        }
        self.tuning_width.max(f32::EPSILON)
    }

    /// Spike-emitting core shared by every returning and sink-based path.
    fn encode_with_sensitivity_scale_into<S: SpikeSink + ?Sized, R: rand::Rng + ?Sized>(
        &mut self,
        input: &[f32],
        sensitivity_scale: f32,
        rng: &mut R,
        sink: &mut S,
    ) {
        let spike_channels = input
            .len()
            .checked_mul(self.num_neurons)
            .expect("directional population channel count overflow");
        assert!(
            spike_channels <= crate::error::MAX_SPIKE_CHANNELS,
            "directional population channel count exceeds u16 channel-ID space"
        );

        if !sensitivity_scale.is_finite() || sensitivity_scale <= 0.0 {
            return;
        }
        let tuning_width = self.effective_tuning_width(sensitivity_scale);
        let rate_gain = sensitivity_scale.min(1.0);

        for (input_channel, &value) in input.iter().enumerate() {
            let channel_base = input_channel * self.num_neurons;
            for i in 0..self.num_neurons {
                let channel = u16::try_from(channel_base + i)
                    .expect("directional population channel count was validated above");
                let rate = self.response_with_tuning_width(value, i, tuning_width) * rate_gain;
                if crate::rng::gen_unit_f32_with_rng(rng) < rate {
                    sink.push(SpikeEvent::at_step_start(channel, true));
                }
            }
        }
    }

    /// Encodes with an explicit sensitivity scale.
    fn encode_with_sensitivity_scale(
        &mut self,
        input: &[f32],
        sensitivity_scale: f32,
    ) -> EncodedOutput {
        let mut output = EncodedOutput::new();
        let mut rng = rand::rng();
        self.encode_with_sensitivity_scale_into(
            input,
            sensitivity_scale,
            &mut rng,
            &mut output.spikes,
        );
        output
    }

    /// [`Encoder::encode`] with a caller-supplied RNG.
    ///
    /// Pass a seeded generator (for example `StdRng::seed_from_u64`) when the
    /// spike selection must be reproducible; the draw order is one sample per
    /// direction-tuned neuron, input channel by input channel.
    ///
    /// [`Encoder::encode`]: crate::Encoder::encode
    pub fn encode_with_rng<R: rand::Rng + ?Sized>(
        &mut self,
        input: &[f32],
        rng: &mut R,
    ) -> EncodedOutput {
        let mut output = EncodedOutput::new();
        self.encode_with_sensitivity_scale_into(input, 1.0, rng, &mut output.spikes);
        output
    }

    /// [`Encoder::encode_step`] with a caller-supplied RNG. Identical to
    /// [`encode_with_rng`](Self::encode_with_rng): batch and streaming share
    /// one path for this encoder.
    ///
    /// [`Encoder::encode_step`]: crate::Encoder::encode_step
    pub fn encode_step_with_rng<R: rand::Rng + ?Sized>(
        &mut self,
        input: &[f32],
        rng: &mut R,
    ) -> EncodedOutput {
        self.encode_with_rng(input, rng)
    }
}

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

impl Encoder for DirectionalPopulationEncoder {
    fn encode(&mut self, input: &[f32]) -> EncodedOutput {
        self.encode_with_sensitivity_scale(input, 1.0)
    }

    fn encode_step(&mut self, input: &[f32]) -> EncodedOutput {
        self.encode(input)
    }

    fn encode_into(&mut self, input: &[f32], sink: &mut dyn SpikeSink) {
        crate::sink::through_chunks(sink, |sink| {
            let mut rng = rand::rng();
            self.encode_with_sensitivity_scale_into(input, 1.0, &mut rng, sink)
        });
    }

    fn encode_step_into(&mut self, input: &[f32], sink: &mut dyn SpikeSink) {
        self.encode_into(input, sink);
    }

    /// One call is one tick; batch and streaming are identical here.
    ///
    /// Each tuned neuron independently emits at most one spike per call, all at
    /// [`TickOffset::ZERO`](crate::time::TickOffset::ZERO), so the population
    /// code is carried by *which* channels fire rather than by their offsets.
    /// Ticks are dimensionless: tuning curves yield a per-call probability, not
    /// a rate in hertz.
    fn time_model(&self) -> TimeModel {
        TimeModel::INSTANT
    }

    fn reset(&mut self) {
        // No state to reset
    }
}

impl ModulatedEncoder for DirectionalPopulationEncoder {
    fn encode_with_gains(&mut self, input: &[f32], gains: EncodingGains) -> EncodedOutput {
        self.encode_with_sensitivity_scale(input, gains.sanitize().sensitivity_scale)
    }

    fn encode_with_gains_into(
        &mut self,
        input: &[f32],
        gains: EncodingGains,
        sink: &mut dyn SpikeSink,
    ) {
        let sensitivity_scale = gains.sanitize().sensitivity_scale;
        crate::sink::through_chunks(sink, |sink| {
            let mut rng = rand::rng();
            self.encode_with_sensitivity_scale_into(input, sensitivity_scale, &mut rng, sink)
        });
    }

    fn encode_step_with_gains_into(
        &mut self,
        input: &[f32],
        gains: EncodingGains,
        sink: &mut dyn SpikeSink,
    ) {
        self.encode_with_gains_into(input, gains, sink);
    }

    fn encode_with_modulators_into(
        &mut self,
        input: &[f32],
        modulators: &NeuroModulators,
        gain_curves: &NeuromodulatorGainCurves,
        sink: &mut dyn SpikeSink,
    ) {
        self.encode_with_gains_into(input, gain_curves.evaluate(modulators), sink);
    }

    fn encode_step_with_modulators_into(
        &mut self,
        input: &[f32],
        modulators: &NeuroModulators,
        gain_curves: &NeuromodulatorGainCurves,
        sink: &mut dyn SpikeSink,
    ) {
        self.encode_step_with_gains_into(input, gain_curves.evaluate(modulators), sink);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
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
                encoder.response_with_tuning_width(angle, index, encoder.tuning_width),
                1.0,
                "neuron {index} must peak exactly at its preferred angle"
            );
            for other in 0..encoder.num_neurons() {
                if other != index {
                    assert!(
                        encoder.response_with_tuning_width(angle, other, encoder.tuning_width)
                            < 1.0
                    );
                }
            }
        }
    }

    #[test]
    fn tuning_response_wraps_across_zero() {
        let encoder = DirectionalPopulationEncoder::new(16, 0.25);
        let width = encoder.tuning_width;
        let epsilon = 0.01;

        let positive = encoder.response_with_tuning_width(epsilon, 0, width);
        let wrapped_positive =
            encoder.response_with_tuning_width(std::f32::consts::TAU + epsilon, 0, width);
        let negative = encoder.response_with_tuning_width(-epsilon, 0, width);
        let wrapped_negative =
            encoder.response_with_tuning_width(std::f32::consts::TAU - epsilon, 0, width);

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
