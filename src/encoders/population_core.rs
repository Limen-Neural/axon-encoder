//! Shared implementation for stochastic tuning-curve population encoders.
//!
//! The scalar and directional population encoders differ only in how a tuned
//! neuron responds to an input. Their fan-out, sensitivity scaling, seeded RNG
//! draw order, instant time model, and allocation-free sink/modulated paths are
//! identical and live here once so the public encoders cannot drift apart.

use crate::prelude::*;

/// Crate-internal tuning model shared by population-style encoders.
pub(crate) trait PopulationCore {
    /// Number of tuned neurons emitted for each input channel.
    fn num_neurons(&self) -> usize;

    /// Base tuning width before sensitivity modulation.
    fn tuning_width(&self) -> f32;

    /// Human-readable encoder kind used in panic messages.
    fn encoder_kind(&self) -> &'static str;

    /// Unmodulated response of one neuron at an explicit tuning width.
    fn get_rate_with_tuning_width(&self, input: f32, neuron_index: usize, tuning_width: f32)
    -> f32;

    /// Effective tuning width under a sensitivity gain.
    ///
    /// Scales at or above one narrow the tuning curve. Positive sub-unity
    /// scales keep the base width and suppress firing probability instead of
    /// widening toward universal firing.
    fn effective_tuning_width(&self, sensitivity_scale: f32) -> f32 {
        let base_width = self.tuning_width().max(f32::EPSILON);
        if !sensitivity_scale.is_finite() || sensitivity_scale <= 0.0 {
            return base_width;
        }
        if sensitivity_scale >= 1.0 {
            return (self.tuning_width() / sensitivity_scale).max(f32::EPSILON);
        }
        base_width
    }

    /// Spike-emitting core shared by every returning and sink-based path.
    fn encode_with_sensitivity_scale_into<S: SpikeSink + ?Sized, R: rand::Rng + ?Sized>(
        &mut self,
        input: &[f32],
        sensitivity_scale: f32,
        rng: &mut R,
        sink: &mut S,
    ) {
        let num_neurons = self.num_neurons();
        let spike_channels = input
            .len()
            .checked_mul(num_neurons)
            .unwrap_or_else(|| panic!("{} channel count overflow", self.encoder_kind()));
        assert!(
            spike_channels <= crate::error::MAX_SPIKE_CHANNELS,
            "{} channel count exceeds u16 channel-ID space",
            self.encoder_kind()
        );

        // Zero/negative/non-finite sensitivity fully suppresses population responses.
        if !sensitivity_scale.is_finite() || sensitivity_scale <= 0.0 {
            return;
        }
        let tuning_width = self.effective_tuning_width(sensitivity_scale);
        // Scales > 1 narrow the width above; scales in (0, 1) only reduce rate
        // so small positive gains never produce near-universal firing.
        let rate_gain = sensitivity_scale.min(1.0);

        for (input_channel, &value) in input.iter().enumerate() {
            let channel_base = input_channel * num_neurons;
            for i in 0..num_neurons {
                let channel = u16::try_from(channel_base + i)
                    .expect("population channel count was validated above");
                let rate = self.get_rate_with_tuning_width(value, i, tuning_width) * rate_gain;
                if crate::rng::gen_unit_f32_with_rng(rng) < rate {
                    sink.push(SpikeEvent::at_step_start(channel, true));
                }
            }
        }
    }

    /// Encodes through the thread-local RNG with an explicit sensitivity scale.
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

    /// Encodes with a caller-supplied RNG and no sensitivity modulation.
    fn encode_with_rng<R: rand::Rng + ?Sized>(
        &mut self,
        input: &[f32],
        rng: &mut R,
    ) -> EncodedOutput {
        let mut output = EncodedOutput::new();
        self.encode_with_sensitivity_scale_into(input, 1.0, rng, &mut output.spikes);
        output
    }
}

macro_rules! impl_population_encoder_api {
    ($ty:ty) => {
        impl $ty {
            /// [`Encoder::encode`] with a caller-supplied RNG.
            ///
            /// Pass a seeded generator (for example `StdRng::seed_from_u64`) when the
            /// spike selection must be reproducible; the draw order is one sample per
            /// tuned neuron, input channel by input channel.
            ///
            /// [`Encoder::encode`]: crate::Encoder::encode
            pub fn encode_with_rng<R: rand::Rng + ?Sized>(
                &mut self,
                input: &[f32],
                rng: &mut R,
            ) -> $crate::types::EncodedOutput {
                <Self as $crate::encoders::population_core::PopulationCore>::encode_with_rng(
                    self, input, rng,
                )
            }

            /// [`Encoder::encode_step`] with a caller-supplied RNG. Identical to
            /// `encode_with_rng`: batch and streaming share one path for this encoder.
            ///
            /// [`Encoder::encode_step`]: crate::Encoder::encode_step
            pub fn encode_step_with_rng<R: rand::Rng + ?Sized>(
                &mut self,
                input: &[f32],
                rng: &mut R,
            ) -> $crate::types::EncodedOutput {
                self.encode_with_rng(input, rng)
            }
        }

        impl $crate::Encoder for $ty {
            fn encode(&mut self, input: &[f32]) -> $crate::types::EncodedOutput {
                <Self as $crate::encoders::population_core::PopulationCore>::encode_with_sensitivity_scale(
                    self, input, 1.0,
                )
            }

            fn encode_step(&mut self, input: &[f32]) -> $crate::types::EncodedOutput {
                self.encode(input)
            }

            fn encode_into(&mut self, input: &[f32], sink: &mut dyn $crate::sink::SpikeSink) {
                $crate::sink::through_chunks(sink, |sink| {
                    let mut rng = rand::rng();
                    <Self as $crate::encoders::population_core::PopulationCore>::encode_with_sensitivity_scale_into(
                        self, input, 1.0, &mut rng, sink,
                    )
                });
            }

            fn encode_step_into(&mut self, input: &[f32], sink: &mut dyn $crate::sink::SpikeSink) {
                self.encode_into(input, sink);
            }

            /// One call is one tick; batch and streaming are identical here.
            ///
            /// Each tuned neuron independently emits at most one spike per call, all at
            /// [`TickOffset::ZERO`](crate::time::TickOffset::ZERO), so the population
            /// code is carried by *which* channels fire rather than by their offsets.
            /// Ticks are dimensionless: tuning curves yield a per-call probability, not
            /// a rate in hertz.
            fn time_model(&self) -> $crate::time::TimeModel {
                $crate::time::TimeModel::INSTANT
            }

            fn reset(&mut self) {
                // No state to reset
            }
        }

        impl $crate::ModulatedEncoder for $ty {
            fn encode_with_gains(
                &mut self,
                input: &[f32],
                gains: $crate::modulators::EncodingGains,
            ) -> $crate::types::EncodedOutput {
                <Self as $crate::encoders::population_core::PopulationCore>::encode_with_sensitivity_scale(
                    self,
                    input,
                    gains.sanitize().sensitivity_scale,
                )
            }

            fn encode_with_gains_into(
                &mut self,
                input: &[f32],
                gains: $crate::modulators::EncodingGains,
                sink: &mut dyn $crate::sink::SpikeSink,
            ) {
                let sensitivity_scale = gains.sanitize().sensitivity_scale;
                $crate::sink::through_chunks(sink, |sink| {
                    let mut rng = rand::rng();
                    <Self as $crate::encoders::population_core::PopulationCore>::encode_with_sensitivity_scale_into(
                        self,
                        input,
                        sensitivity_scale,
                        &mut rng,
                        sink,
                    )
                });
            }

            fn encode_step_with_gains_into(
                &mut self,
                input: &[f32],
                gains: $crate::modulators::EncodingGains,
                sink: &mut dyn $crate::sink::SpikeSink,
            ) {
                self.encode_with_gains_into(input, gains, sink);
            }

            fn encode_with_modulators_into(
                &mut self,
                input: &[f32],
                modulators: &$crate::modulators::NeuroModulators,
                gain_curves: &$crate::modulators::NeuromodulatorGainCurves,
                sink: &mut dyn $crate::sink::SpikeSink,
            ) {
                self.encode_with_gains_into(input, gain_curves.evaluate(modulators), sink);
            }

            fn encode_step_with_modulators_into(
                &mut self,
                input: &[f32],
                modulators: &$crate::modulators::NeuroModulators,
                gain_curves: &$crate::modulators::NeuromodulatorGainCurves,
                sink: &mut dyn $crate::sink::SpikeSink,
            ) {
                self.encode_step_with_gains_into(input, gain_curves.evaluate(modulators), sink);
            }
        }
    };
}

pub(crate) use impl_population_encoder_api;
