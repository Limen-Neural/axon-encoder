//! # axon-encoder
//!
//! Flexible sensory encoding for spiking neural networks.

pub mod encoder;
pub mod encoders;
pub mod modulators;
pub mod poisson;
pub mod rng;
pub mod types;

pub mod prelude {
    pub use crate::encoder::*;
    pub use crate::encoders::*;
    pub use crate::modulators::*;
    pub use crate::poisson::*;
    pub use crate::types::*;
    pub use crate::Encoder;
}

use types::EncodedOutput;

/// The core trait for all encoders in this crate.
///
/// Encoders convert continuous analog values into discrete spike events for
/// spiking neural networks (SNNs). Two modes are supported:
///
/// - **Batch mode** (`encode`): Process a complete input vector at once
/// - **Streaming mode** (`encode_step`): Process incrementally, one step at a time
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

    /// Resets the encoder to its initial state.
    fn reset(&mut self);
}
