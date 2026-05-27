//! # axon-encoder
//!
//! Flexible sensory encoding for spiking neural networks.

pub mod encoder;
pub mod encoders;
pub mod modulators;
pub mod poisson;
pub mod rng;
pub mod spike_encoder;
pub mod types;

pub mod prelude {
    pub use crate::encoder::*;
    pub use crate::encoders::*;
    pub use crate::modulators::*;
    pub use crate::poisson::*;
    pub use crate::spike_encoder::*;
    pub use crate::types::*;
    pub use crate::Encoder;
}

use types::EncodedOutput;

/// The core trait for all encoders in this crate.
pub trait Encoder {
    /// Encodes a slice of analog values into spike events.
    fn encode(&mut self, input: &[f32]) -> EncodedOutput;
    fn reset(&mut self);
}
