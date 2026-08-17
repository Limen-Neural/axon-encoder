//! Standardized types for encoder inputs and outputs.

use crate::time::TickOffset;

/// A single spike event.
///
/// # Time semantics
///
/// `timestamp` is a [`TickOffset`]: a count of encoder ticks measured **from the
/// start of the `encode` / `encode_step` call that emitted this spike**. It is
/// never absolute and never wall-clock — this crate owns no clock. Reconstruct
/// absolute time with a [`TimeCursor`], and get the physical duration of a tick
/// (when the encoder has one) from [`Encoder::time_model`].
///
/// See the [`time`](crate::time) module for the full contract, including the
/// ordering rule for multiple spikes on one channel within a step.
///
/// [`TimeCursor`]: crate::time::TimeCursor
/// [`Encoder::time_model`]: crate::Encoder::time_model
///
/// ```rust
/// use axon_encoder::prelude::*;
///
/// let spike = SpikeEvent::new(3, TickOffset::new(7), true);
/// assert_eq!(spike.timestamp, 7); // TickOffset compares against u64
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SpikeEvent {
    /// Emitting channel ID.
    pub channel: u16,
    /// Ticks from the start of the emitting call. See [`TickOffset`].
    pub timestamp: TickOffset,
    /// `true` for excitatory / positive-going events, `false` for inhibitory.
    pub polarity: bool,
}

impl SpikeEvent {
    /// Builds a spike event.
    ///
    /// `timestamp` accepts a [`TickOffset`] or anything convertible into one,
    /// notably a bare `u64`.
    ///
    /// A pre-0.5 struct literal does **not** compile against the new field type
    /// — Rust applies no `Into` conversion in field initializers — so migrate
    /// `SpikeEvent { channel, timestamp: 5, polarity }` to this constructor, or
    /// wrap the value as `TickOffset::new(5)` to keep the literal.
    ///
    /// ```rust
    /// use axon_encoder::prelude::*;
    ///
    /// assert_eq!(
    ///     SpikeEvent::new(0, 5u64, true),
    ///     SpikeEvent::new(0, TickOffset::new(5), true),
    /// );
    /// ```
    #[inline]
    pub fn new(channel: u16, timestamp: impl Into<TickOffset>, polarity: bool) -> Self {
        Self {
            channel,
            timestamp: timestamp.into(),
            polarity,
        }
    }

    /// Builds a spike at the first tick of the emitting call.
    ///
    /// The common case: encoders that report *whether* a channel fired in this
    /// step rather than *when* within it.
    #[inline]
    pub const fn at_step_start(channel: u16, polarity: bool) -> Self {
        Self {
            channel,
            timestamp: TickOffset::ZERO,
            polarity,
        }
    }
}

/// Optional metadata about the encoding process.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct EncodingMetadata {
    // Add any relevant metadata fields here, e.g.:
    // pub source_sample_index: u64,
}

/// The standardized output of an encoder.
#[derive(Clone, Debug, Default, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct EncodedOutput {
    pub spikes: Vec<SpikeEvent>,
    pub embeddings: Option<Vec<f32>>,
    pub metadata: Option<EncodingMetadata>,
}

impl EncodedOutput {
    pub fn new() -> Self {
        Self::default()
    }
}

/// General-purpose configuration for encoders.
#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct EncoderConfig {
    pub input_channels: usize,
    pub output_channels: usize,
}

impl Default for EncoderConfig {
    fn default() -> Self {
        Self {
            input_channels: 256,
            output_channels: 256,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encoded_output_new() {
        let output = EncodedOutput::new();
        assert!(output.spikes.is_empty());
        assert!(output.embeddings.is_none());
        assert!(output.metadata.is_none());
    }

    #[test]
    fn test_encoder_config_default() {
        let config = EncoderConfig::default();
        assert_eq!(config.input_channels, 256);
        assert_eq!(config.output_channels, 256);
    }

    #[test]
    fn spike_event_constructors_agree() {
        let literal = SpikeEvent {
            channel: 4,
            timestamp: TickOffset::new(9),
            polarity: false,
        };
        assert_eq!(SpikeEvent::new(4, 9u64, false), literal);
        assert_eq!(SpikeEvent::new(4, TickOffset::new(9), false), literal);
        assert_eq!(
            SpikeEvent::at_step_start(4, false),
            SpikeEvent::new(4, TickOffset::ZERO, false)
        );
    }
}
