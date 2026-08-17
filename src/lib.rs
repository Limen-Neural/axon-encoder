//! # axon-encoder
//!
//! Flexible sensory encoding for spiking neural networks: continuous signals
//! in, spike events out. Optional [`EncodingGains`] scale rate / threshold /
//! latency / sensitivity without requiring an external neuromodulator runtime.
//!
//! ## Spike time contract
//!
//! Every encoder here shares one time model: a [`SpikeEvent::timestamp`] is a
//! [`TickOffset`] — encoder ticks counted **from the start of the call that
//! emitted it**. Timestamps are call-relative, so this crate never owns a clock
//! or a scheduler; the caller keeps absolute time with a [`TimeCursor`] and
//! advances it by [`TimeModel::step_ticks`] per call. Encoders configured in
//! physical units also report a [`Timebase`], the duration of one tick.
//!
//! ```rust
//! use axon_encoder::prelude::*;
//! # fn main() -> Result<(), EncoderError> {
//! let mut encoder = LatencyEncoder::try_new(9, (0.0, 1.0))?;
//! let mut cursor = TimeCursor::new(encoder.time_model());
//!
//! let first = encoder.encode(&[1.0, 0.0]);
//! assert_eq!(first.spikes[0].timestamp, 0); // strong input → early
//! assert_eq!(first.spikes[1].timestamp, 9); // weak input → late
//! cursor.advance();
//!
//! let second = encoder.encode(&[1.0, 0.0]);
//! // Offsets repeat per call; absolute time does not.
//! assert_eq!(second.spikes[0].timestamp, 0);
//! assert_eq!(cursor.absolute(second.spikes[0].timestamp), 10);
//! # Ok(())
//! # }
//! ```
//!
//! See the [`time`] module for the full contract: batch versus streaming, spike
//! ordering, and conversion guidance for simulators and hardware adapters.
//!
//! [`SpikeEvent::timestamp`]: types::SpikeEvent::timestamp
//! [`TickOffset`]: time::TickOffset
//! [`TimeCursor`]: time::TimeCursor
//! [`TimeModel::step_ticks`]: time::TimeModel::step_ticks
//! [`Timebase`]: time::Timebase

pub mod encoder;
pub mod encoders;
pub mod error;
pub mod modulators;
#[cfg(feature = "ndarray")]
pub mod ndarray_ext;
pub mod poisson;
pub mod rng;
pub mod time;
pub mod types;

pub use error::EncoderError;
#[cfg(feature = "ndarray")]
pub use ndarray_ext::NdarrayEncoderExt;

pub mod prelude {
    pub use crate::Encoder;
    pub use crate::ModulatedEncoder;
    pub use crate::encoder::*;
    pub use crate::encoders::*;
    pub use crate::error::*;
    pub use crate::modulators::*;
    #[cfg(feature = "ndarray")]
    pub use crate::ndarray_ext::NdarrayEncoderExt;
    pub use crate::poisson::*;
    pub use crate::time::*;
    pub use crate::types::*;
}

use modulators::{EncodingGains, NeuroModulators, NeuromodulatorGainCurves};
use time::TimeModel;
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
///
/// # Examples
///
/// Prefer the **streaming** path for doctests: batch `encode_with_modulators` is
/// stochastic, while `encode_step_with_modulators` on rate encoders is deterministic.
///
/// ```rust
/// use axon_encoder::prelude::*;
/// # fn main() -> Result<(), EncoderError> {
/// let mut enc = RateEncoder::try_new(0.0, 100.0, (0.0, 1.0), 0.01)?;
/// let mods = NeuroModulators {
///     dopamine: 1.0,
///     ..Default::default()
/// };
/// let curves = NeuromodulatorGainCurves {
///     dopamine: ModulatorGainCurves {
///         firing_rate: Some(GainCurve::new((0.0, 1.0), (1.0, 2.0))),
///         ..Default::default()
///     },
///     ..Default::default()
/// };
/// // Accumulates rate_hz * dt; at unit input with elevated gain, a spike fires soon.
/// let mut saw_spike = false;
/// for _ in 0..20 {
///     if !enc
///         .encode_step_with_modulators(&[1.0], &mods, &curves)
///         .spikes
///         .is_empty()
///     {
///         saw_spike = true;
///         break;
///     }
/// }
/// assert!(saw_spike);
/// # Ok(())
/// # }
/// ```
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
/// # Time semantics
///
/// Both modes emit call-relative timestamps: a [`SpikeEvent::timestamp`] counts
/// encoder ticks from the start of *that* call, never from an absolute origin.
/// [`time_model`](Encoder::time_model) reports how a call maps onto the caller's
/// timeline — how far the origin advances per call, how far a single call can
/// reach, and how long a tick is in physical units when the encoder knows.
///
/// [`SpikeEvent::timestamp`]: types::SpikeEvent::timestamp
///
/// # Example
///
/// ```rust
/// use axon_encoder::prelude::*;
/// # fn main() -> Result<(), EncoderError> {
///
/// let mut encoder = RateEncoder::try_new(5.0, 50.0, (0.0, 1.0), 0.010)?;
/// let input = [0.25, 0.75, 0.5];
///
/// // Batch encoding
/// let output = encoder.encode(&input);
///
/// // Reset for streaming (if using stateful encoder)
/// encoder.reset();
/// # Ok(())
/// # }
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

    /// Describes how this encoder places spikes in time.
    ///
    /// Callers use it to drive any encoder — including a `&mut dyn Encoder` —
    /// on one timeline: advance a [`TimeCursor`](time::TimeCursor) by
    /// [`TimeModel::step_ticks`](time::TimeModel::step_ticks) after each call,
    /// and convert offsets with [`TimeModel::timebase`](time::TimeModel::timebase).
    ///
    /// The default is [`TimeModel::INSTANT`](time::TimeModel::INSTANT): one call
    /// is one tick and every spike lands at
    /// [`TickOffset::ZERO`](time::TickOffset::ZERO). Encoders that place spikes
    /// *within* a step (latency, phase) override it.
    fn time_model(&self) -> TimeModel {
        TimeModel::INSTANT
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

    /// Guard: `axon-encoder` must not depend on the neuromod crate (#21).
    ///
    /// Uses `cargo metadata` so table syntax, workspace inheritance, rename/
    /// package aliases, and normal/dev/build/target scopes are all covered
    /// without matching description prose.
    #[test]
    fn cargo_toml_has_no_neuromod_crate_dependency() {
        // `CARGO` is always set when this crate is built by cargo (no fallback branch).
        let output = std::process::Command::new(env!("CARGO"))
            .args(["metadata", "--no-deps", "--locked", "--format-version", "1"])
            .current_dir(env!("CARGO_MANIFEST_DIR"))
            .output()
            .expect("spawn cargo metadata");
        // Always materialize stderr so a --locked/offline failure is actionable
        // and codecov does not see a cold format arm.
        let metadata_detail = format!(
            "cargo metadata failed (status={:?}): {}",
            output.status.code(),
            String::from_utf8_lossy(&output.stderr)
        );
        assert!(output.status.success(), "{metadata_detail}");

        let meta: serde_json::Value =
            serde_json::from_slice(&output.stdout).expect("parse cargo metadata json");
        let packages = meta["packages"].as_array().expect("packages array");
        let deps = packages
            .iter()
            .find(|p| p["name"] == "axon-encoder")
            .expect("axon-encoder package in metadata")["dependencies"]
            .as_array()
            .expect("dependencies array");

        // Collect offenders so a failure names them; build the message on the
        // success path too so codecov patch does not see cold format arms.
        let forbidden: Vec<&serde_json::Value> =
            deps.iter().filter(|d| d["name"] == "neuromod").collect();
        let detail = format!(
            "forbidden neuromod deps (name/kind): {:?}",
            forbidden
                .iter()
                .map(|d| (&d["name"], &d["kind"]))
                .collect::<Vec<_>>()
        );
        assert!(forbidden.is_empty(), "{detail}");
    }

    #[test]
    fn test_encoder_default_encode_step_delegates_to_encode() {
        use crate::prelude::*;

        struct PassThrough;
        impl Encoder for PassThrough {
            fn encode(&mut self, input: &[f32]) -> EncodedOutput {
                let mut out = EncodedOutput::new();
                for (i, &v) in input.iter().enumerate() {
                    out.spikes.push(SpikeEvent::new(i as u16, v as u64, true));
                }
                out
            }
            fn reset(&mut self) {}
        }

        let mut enc = PassThrough;
        let out = enc.encode_step(&[1.0, 2.0]);
        assert_eq!(out.spikes.len(), 2);
        // Out-of-crate impls inherit the instant time model without opting in.
        assert_eq!(enc.time_model(), TimeModel::INSTANT);
    }
}
