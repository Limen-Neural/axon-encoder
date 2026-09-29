use std::fmt;

/// Stable error type returned by fallible encoder constructors.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub enum EncoderError {
    /// A rate parameter must be finite.
    NonFiniteRate { parameter: &'static str },
    /// `base_rate` must not exceed `max_rate`.
    RateOrder,
    /// Numeric range bounds must be finite and strictly increasing.
    InvalidRange { parameter: &'static str },
    /// Count parameters must be positive.
    CountMustBePositive { parameter: &'static str },
    /// A threshold/tuning-width style parameter must be finite and positive.
    NonPositiveOrNonFinite { parameter: &'static str },
    /// A parameter must be finite and non-negative (`>= 0`).
    NonNegativeFinite { parameter: &'static str },
    /// Channel/neuron count exceeds the `u16` channel-ID range used when emitting spikes.
    NumChannelsTooLarge,
    /// Temporal history depth is too small for the encoder window.
    HistoryDepthTooSmall { minimum: usize },
    /// A deserialized state vector has inconsistent lengths.
    StateLengthMismatch {
        left: &'static str,
        right: &'static str,
    },
    /// A deserialized history channel contains more samples than allowed by history_depth.
    HistoryLengthExceedsDepth { channel: usize },
    /// Cycle/window parameters must be non-zero.
    WindowMustBePositive { parameter: &'static str },
    /// A window parameter leaves no room for its exclusive upper bound.
    ///
    /// Emitted when a `parameter + 1` tick span would overflow `u64`, which
    /// would make the encoder's declared span unable to contain its own spikes.
    WindowTooLarge { parameter: &'static str },
}

impl fmt::Display for EncoderError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NonFiniteRate { parameter } => write!(f, "{parameter} must be finite"),
            Self::RateOrder => write!(f, "base_rate must be less than or equal to max_rate"),
            Self::InvalidRange { parameter } => write!(
                f,
                "{parameter} must be finite and min must be less than max"
            ),
            Self::CountMustBePositive { parameter } => {
                write!(f, "{parameter} must be greater than 0")
            }
            Self::NonPositiveOrNonFinite { parameter } => {
                write!(f, "{parameter} must be finite and greater than 0")
            }
            Self::NonNegativeFinite { parameter } => {
                write!(f, "{parameter} must be finite and non-negative")
            }
            Self::NumChannelsTooLarge => write!(
                f,
                "num_channels exceeds u16::MAX as usize + 1 (max addressable spike channels)"
            ),
            Self::HistoryDepthTooSmall { minimum } => {
                write!(f, "history_depth must be at least {minimum}")
            }
            Self::StateLengthMismatch { left, right } => {
                write!(f, "mismatched {left} and {right} lengths")
            }
            Self::HistoryLengthExceedsDepth { channel } => {
                write!(f, "history channel {channel} length exceeds history_depth")
            }
            Self::WindowMustBePositive { parameter } => {
                write!(f, "{parameter} must be greater than 0")
            }
            Self::WindowTooLarge { parameter } => write!(
                f,
                "{parameter} must be less than u64::MAX (its window spans {parameter} + 1 ticks)"
            ),
        }
    }
}

impl std::error::Error for EncoderError {}

/// Error returned by a runtime streaming operation on a
/// [`StreamingEncoder`](crate::streaming::StreamingEncoder).
///
/// Unlike [`EncoderError`], which reports invalid *construction* parameters,
/// this reports a failed *operation* on an otherwise valid wrapper. It is kept
/// a dedicated enum so a caller can match the streaming backpressure case
/// without pulling in constructor-validation variants.
///
/// # Backpressure
///
/// [`StreamingEncoder::encode_step`](crate::streaming::StreamingEncoder::encode_step)
/// returns [`StreamingError::Backpressure`] when the wrapper is blocked: a
/// prior call under [`FlushPolicy::Manual`](crate::streaming::FlushPolicy::Manual)
/// produced output that did not fit the remaining capacity and is being held.
/// The caller must
/// [`flush_into`](crate::streaming::StreamingEncoder::flush_into) (or
/// [`reset`](crate::streaming::StreamingEncoder::reset)) before encoding again.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub enum StreamingError {
    /// The wrapper is blocked and cannot accept a new call until it is flushed.
    ///
    /// `buffered_spikes` counts both queued and held spikes, and `capacity` is
    /// the queue's configured bound. A held call can make the total exceed it.
    Backpressure {
        /// Spikes currently queued or held in the wrapper.
        buffered_spikes: usize,
        /// The configured spike capacity.
        capacity: usize,
    },
}

impl fmt::Display for StreamingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Backpressure {
                buffered_spikes,
                capacity,
            } => write!(
                f,
                "cannot encode while blocked: {buffered_spikes} spike(s) buffered (queue capacity {capacity}); flush first"
            ),
        }
    }
}

impl std::error::Error for StreamingError {}

pub(crate) const MAX_SPIKE_CHANNELS: usize = u16::MAX as usize + 1;

pub(crate) fn validate_range(
    parameter: &'static str,
    range: (f32, f32),
) -> Result<(), EncoderError> {
    if range.0.is_finite() && range.1.is_finite() && range.0 < range.1 {
        Ok(())
    } else {
        Err(EncoderError::InvalidRange { parameter })
    }
}

/// Like [`validate_range`], but also rejects spans that overflow f32 arithmetic
/// (e.g. `(f32::MIN, f32::MAX)` → span is `+inf`).
///
/// Use for encoders that normalize with f32 division (`RateEncoder`,
/// `PopulationEncoder`). Latency/Phase normalize in f64 and may accept wider
/// finite bounds.
pub(crate) fn validate_range_f32_span(
    parameter: &'static str,
    range: (f32, f32),
) -> Result<(), EncoderError> {
    validate_range(parameter, range)?;
    if (range.1 - range.0).is_finite() {
        Ok(())
    } else {
        Err(EncoderError::InvalidRange { parameter })
    }
}

pub(crate) fn validate_channel_count(num_channels: usize) -> Result<(), EncoderError> {
    if num_channels <= MAX_SPIKE_CHANNELS {
        Ok(())
    } else {
        Err(EncoderError::NumChannelsTooLarge)
    }
}

/// Validates that `value` is finite and `>= 0`.
pub(crate) fn validate_non_negative_finite(
    parameter: &'static str,
    value: f32,
) -> Result<(), EncoderError> {
    if value.is_finite() && value >= 0.0 {
        Ok(())
    } else {
        Err(EncoderError::NonNegativeFinite { parameter })
    }
}
