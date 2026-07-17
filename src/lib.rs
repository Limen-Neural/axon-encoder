//! # axon-encoder
//!
//! Flexible sensory encoding for spiking neural networks.

pub mod encoder;
pub mod encoders;
pub mod modulators;
#[cfg(feature = "ndarray")]
pub mod ndarray_ext;
pub mod poisson;
pub mod rng;
pub mod types;

#[cfg(feature = "ndarray")]
pub use ndarray_ext::NdarrayEncoderExt;

pub mod prelude {
    pub use crate::Encoder;
    pub use crate::ModulatedEncoder;
    pub use crate::encoder::*;
    pub use crate::encoders::*;
    pub use crate::modulators::*;
    #[cfg(feature = "ndarray")]
    pub use crate::ndarray_ext::NdarrayEncoderExt;
    pub use crate::poisson::*;
    pub use crate::types::*;
}

use modulators::{EncodingGains, NeuroModulators, NeuromodulatorGainCurves};
use types::EncodedOutput;

/// Encoders that can apply neuromodulator-driven gain curves.
///
/// Object-safe so callers can use `&mut dyn ModulatedEncoder` when the concrete
/// encoder type is not known at compile time. Implementations map the relevant
/// component of [`EncodingGains`] to encoder-specific scaling; public modulator
/// helpers are provided once here.
///
/// Concrete encoders also keep inherent `encode_with_modulators` /
/// `encode_step_with_modulators` wrappers so existing call sites need not import
/// this trait.
pub trait ModulatedEncoder: Encoder {
    /// Encodes input using already evaluated encoding gains.
    ///
    /// Implementations must sanitize `gains` (or the component they use) before
    /// applying them.
    fn encode_with_gains(&mut self, input: &[f32], gains: EncodingGains) -> EncodedOutput;

    /// Encodes one streaming step using already evaluated encoding gains.
    ///
    /// Stateful encoders should override this when streaming requires distinct
    /// state handling from the batch path.
    fn encode_step_with_gains(&mut self, input: &[f32], gains: EncodingGains) -> EncodedOutput {
        self.encode_with_gains(input, gains)
    }

    /// Encodes input using neuromodulator-driven gain curves.
    fn encode_with_modulators(
        &mut self,
        input: &[f32],
        modulators: &NeuroModulators,
        gain_curves: &NeuromodulatorGainCurves,
    ) -> EncodedOutput {
        self.encode_with_gains(input, gain_curves.evaluate(modulators))
    }

    /// Encodes one streaming step using neuromodulator-driven gain curves.
    fn encode_step_with_modulators(
        &mut self,
        input: &[f32],
        modulators: &NeuroModulators,
        gain_curves: &NeuromodulatorGainCurves,
    ) -> EncodedOutput {
        self.encode_step_with_gains(input, gain_curves.evaluate(modulators))
    }
}

/// The core trait for all encoders in this crate.
///
/// Encoders convert continuous analog values into discrete spike events for
/// spiking neural networks (SNNs). Two modes are supported:
///
/// - **Batch mode** (`encode`): Process a complete input vector at once.
/// - **Streaming mode** (`encode_step`): Process incrementally, one step at a time.
///
/// # Example
///
/// ```rust
/// use axon_encoder::prelude::*;
///
/// let mut encoder = RateEncoder::new(5.0, 50.0, (0.0, 1.0));
/// let input = [0.25, 0.75, 0.5];
///
/// // Batch encoding
/// let output = encoder.encode(&input);
///
/// // Reset for streaming (if using stateful encoder)
/// encoder.reset();
/// ```
pub trait Encoder {
    /// Encodes a slice of analog values into spike events (batch mode).
    fn encode(&mut self, input: &[f32]) -> EncodedOutput;

    /// Encodes a single step incrementally (streaming mode).
    ///
    /// By default, this delegates to `encode()` for stateless encoders.
    /// Stateful encoders should override this to maintain state between calls.
    ///
    /// # Arguments
    ///
    /// * `input` - A slice of analog values to encode
    ///
    /// # Returns
    ///
    /// An `EncodedOutput` containing any spike events generated in this step
    fn encode_step(&mut self, input: &[f32]) -> EncodedOutput {
        self.encode(input)
    }

    /// Resets the encoder to its initial state
    fn reset(&mut self);
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_lib_prelude_imports() {
        use crate::prelude::*;
        let _ = EncoderConfig::default();
    }

    #[test]
    fn test_encoder_default_encode_step_delegates_to_encode() {
        use crate::prelude::*;

        struct PassThrough;
        impl Encoder for PassThrough {
            fn encode(&mut self, input: &[f32]) -> EncodedOutput {
                let mut out = EncodedOutput::new();
                for (i, &v) in input.iter().enumerate() {
                    out.spikes.push(SpikeEvent {
                        channel: i as u16,
                        timestamp: v as u64,
                        polarity: true,
                    });
                }
                out
            }
            fn reset(&mut self) {}
        }

        let mut enc = PassThrough;
        let out = enc.encode_step(&[1.0, 2.0]);
        assert_eq!(out.spikes.len(), 2);
    }
}
