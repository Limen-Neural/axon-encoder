pub mod baseline_hold;
pub mod delta;
pub mod derivative;
pub mod latency;
pub mod phase;
pub mod population;
pub mod predictive;
pub mod rate;
pub mod temporal;

#[cfg(test)]
pub(crate) mod property_support;

pub use baseline_hold::BaselineHoldTernaryEncoder;
pub use delta::{DeltaEncoder, encode_deltas_to_spikes};
pub use derivative::DerivativeEncoder;
pub use latency::LatencyEncoder;
pub use phase::PhaseEncoder;
pub use population::PopulationEncoder;
pub use predictive::{PredictiveEncoder, PredictiveEncoderError};
pub use rate::RateEncoder;
pub use temporal::TemporalEncoder;

// Share sink forwarding for encoders whose streaming gain path is identical
// to their batch gain path. Their implementations still supply the core
// encode_with_gains_into method; these adapters avoid intermediate outputs.
macro_rules! forward_modulator_sink {
    ($method:ident, $destination:ident) => {
        fn $method(
            &mut self,
            input: &[f32],
            modulators: &$crate::modulators::NeuroModulators,
            gain_curves: &$crate::modulators::NeuromodulatorGainCurves,
            sink: &mut dyn $crate::sink::SpikeSink,
        ) {
            self.$destination(input, gain_curves.evaluate(modulators), sink);
        }
    };
}

macro_rules! impl_same_step_modulated_sinks {
    () => {
        fn encode_step_with_gains_into(
            &mut self,
            input: &[f32],
            gains: $crate::modulators::EncodingGains,
            sink: &mut dyn $crate::sink::SpikeSink,
        ) {
            self.encode_with_gains_into(input, gains, sink);
        }

        $crate::encoders::forward_modulator_sink!(
            encode_with_modulators_into,
            encode_with_gains_into
        );
        $crate::encoders::forward_modulator_sink!(
            encode_step_with_modulators_into,
            encode_step_with_gains_into
        );
    };
}
pub(crate) use {forward_modulator_sink, impl_same_step_modulated_sinks};
