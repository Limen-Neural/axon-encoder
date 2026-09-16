//! The spike time contract: what [`SpikeEvent::timestamp`] means, in what unit,
//! and relative to what reference frame.
//!
//! [`SpikeEvent::timestamp`]: crate::types::SpikeEvent::timestamp
//!
//! # The one time model
//!
//! Every encoder in this crate follows the same rule:
//!
//! > **A spike timestamp is a [`TickOffset`]: an unsigned count of encoder
//! > *ticks* measured from the start of the call that emitted it.**
//!
//! Timestamps are **call-relative**, never absolute and never wall-clock. This
//! crate owns no clock and no scheduler: the caller owns absolute time and
//! reconstructs it with a [`TimeCursor`].
//!
//! - **Batch** (`Encoder::encode`) — the call is one window whose origin is its
//!   own tick 0. Offsets fall in `0..span_ticks`.
//! - **Streaming** (`Encoder::encode_step`) — identical rule, once per call. The
//!   caller advances its own origin by [`TimeModel::step_ticks`] after each call.
//!
//! A *tick* is dimensionless by default. When an encoder is configured in
//! physical units (for example [`RateEncoder::try_new`]'s `dt_seconds`), it
//! reports the physical duration of one tick as a [`Timebase`], so downstream
//! simulators and hardware adapters can convert without guessing.
//!
//! [`RateEncoder::try_new`]: crate::encoders::RateEncoder::try_new
//!
//! # Ordering contract
//!
//! Within one [`EncodedOutput::spikes`] slice:
//!
//! 1. Spikes are **channel-major**: channel IDs are non-decreasing.
//! 2. Within one channel, offsets are **non-decreasing**.
//! 3. Repeated spikes from one channel at the same offset (a burst emitted in a
//!    single step, as [`RateEncoder`] streaming can produce) are **contiguous**
//!    and mutually unordered — they are indistinguishable events, so the run
//!    length is a spike *count*, not a sequence.
//!
//! [`EncodedOutput::spikes`]: crate::types::EncodedOutput::spikes
//! [`RateEncoder`]: crate::encoders::RateEncoder
//!
//! # Converting to absolute time
//!
//! ```rust
//! use axon_encoder::prelude::*;
//! # fn main() -> Result<(), EncoderError> {
//! let mut encoder = LatencyEncoder::try_new(10, (0.0, 1.0))?;
//! let mut cursor = TimeCursor::new(encoder.time_model());
//!
//! for _ in 0..3 {
//!     let out = encoder.encode_step(&[1.0, 0.0]);
//!     for spike in &out.spikes {
//!         // Absolute tick on the caller's timeline.
//!         let _absolute = cursor.absolute(spike.timestamp);
//!     }
//!     cursor.advance();
//! }
//! // Three presentations of an 11-tick window.
//! assert_eq!(cursor.origin(), 33);
//! # Ok(())
//! # }
//! ```

use core::fmt;
use core::num::NonZeroU64;

use crate::error::EncoderError;
use crate::types::SpikeEvent;

/// A spike time, in encoder ticks, relative to the start of the emitting call.
///
/// This is the type of [`SpikeEvent::timestamp`]. It is deliberately a distinct
/// type from a bare `u64` so a call-relative tick offset cannot be mistaken for
/// an absolute timestamp, a wall-clock value, or a nanosecond count.
///
/// [`SpikeEvent::timestamp`]: crate::types::SpikeEvent::timestamp
///
/// # Migration from `u64`
///
/// `TickOffset` converts to and from `u64` and compares directly against `u64`,
/// so pre-0.5 call sites migrate with minimal churn:
///
/// ```rust
/// use axon_encoder::prelude::*;
///
/// let offset = TickOffset::new(7);
/// assert_eq!(offset, 7); // comparisons against u64 still work
/// assert_eq!(offset.ticks(), 7u64); // explicit unwrap
/// let from_u64: TickOffset = 7u64.into(); // struct-literal call sites
/// assert_eq!(from_u64, offset);
/// ```
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(transparent))]
pub struct TickOffset(u64);

impl TickOffset {
    /// The start of the emitting call.
    pub const ZERO: Self = Self(0);

    /// Wraps a raw tick count.
    #[inline]
    pub const fn new(ticks: u64) -> Self {
        Self(ticks)
    }

    /// The raw tick count.
    #[inline]
    pub const fn ticks(self) -> u64 {
        self.0
    }

    /// Whether the spike lands on the first tick of its call.
    #[inline]
    pub const fn is_zero(self) -> bool {
        self.0 == 0
    }

    /// Shifts the offset later, saturating at `u64::MAX`.
    #[inline]
    pub const fn saturating_add(self, ticks: u64) -> Self {
        Self(self.0.saturating_add(ticks))
    }

    /// Shifts the offset later, returning `None` on overflow.
    #[inline]
    pub const fn checked_add(self, ticks: u64) -> Option<Self> {
        match self.0.checked_add(ticks) {
            Some(sum) => Some(Self(sum)),
            None => None,
        }
    }
}

impl From<u64> for TickOffset {
    #[inline]
    fn from(ticks: u64) -> Self {
        Self(ticks)
    }
}

impl From<TickOffset> for u64 {
    #[inline]
    fn from(offset: TickOffset) -> Self {
        offset.0
    }
}

impl PartialEq<u64> for TickOffset {
    #[inline]
    fn eq(&self, other: &u64) -> bool {
        self.0 == *other
    }
}

impl PartialEq<TickOffset> for u64 {
    #[inline]
    fn eq(&self, other: &TickOffset) -> bool {
        *self == other.0
    }
}

impl PartialOrd<u64> for TickOffset {
    #[inline]
    fn partial_cmp(&self, other: &u64) -> Option<core::cmp::Ordering> {
        self.0.partial_cmp(other)
    }
}

impl PartialOrd<TickOffset> for u64 {
    #[inline]
    fn partial_cmp(&self, other: &TickOffset) -> Option<core::cmp::Ordering> {
        self.partial_cmp(&other.0)
    }
}

impl fmt::Display for TickOffset {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}t", self.0)
    }
}

/// The physical duration of one encoder tick.
///
/// A `Timebase` is what turns dimensionless [`TickOffset`] values into physical
/// time. Encoders configured in physical units report one; purely combinatorial
/// encoders report `None` and leave the tick duration to the caller.
///
/// Stored as whole nanoseconds so tick arithmetic stays exact — accumulating
/// `f32` seconds across a long run drifts, integer nanoseconds do not.
///
/// ```rust
/// use axon_encoder::prelude::*;
/// # fn main() -> Result<(), EncoderError> {
/// let tb = Timebase::try_from_seconds(0.001)?;
/// assert_eq!(tb.tick_nanos(), 1_000_000);
/// assert_eq!(tb.offset_nanos(TickOffset::new(5)), 5_000_000);
/// # Ok(())
/// # }
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(try_from = "u64", into = "u64"))]
pub struct Timebase {
    tick_nanos: NonZeroU64,
}

impl Timebase {
    /// One nanosecond per tick.
    pub const NANOSECOND: Self = Self::from_nanos(NonZeroU64::new(1).expect("1 is non-zero"));
    /// One microsecond per tick.
    pub const MICROSECOND: Self =
        Self::from_nanos(NonZeroU64::new(1_000).expect("1e3 is non-zero"));
    /// One millisecond per tick.
    pub const MILLISECOND: Self =
        Self::from_nanos(NonZeroU64::new(1_000_000).expect("1e6 is non-zero"));
    /// One second per tick.
    pub const SECOND: Self =
        Self::from_nanos(NonZeroU64::new(1_000_000_000).expect("1e9 non-zero"));

    /// Builds a timebase from a non-zero nanosecond tick duration.
    #[inline]
    pub const fn from_nanos(tick_nanos: NonZeroU64) -> Self {
        Self { tick_nanos }
    }

    /// Builds a timebase from a nanosecond tick duration.
    ///
    /// # Errors
    ///
    /// Returns [`EncoderError::WindowMustBePositive`] when `tick_nanos` is zero.
    pub fn try_from_nanos(tick_nanos: u64) -> Result<Self, EncoderError> {
        NonZeroU64::new(tick_nanos).map(Self::from_nanos).ok_or(
            EncoderError::WindowMustBePositive {
                parameter: "tick_nanos",
            },
        )
    }

    /// Builds a timebase from a tick duration in seconds.
    ///
    /// # Errors
    ///
    /// Returns [`EncoderError::NonPositiveOrNonFinite`] when `tick_seconds` is
    /// not finite and positive, rounds to less than one nanosecond, or exceeds
    /// the `u64` nanosecond range.
    pub fn try_from_seconds(tick_seconds: f64) -> Result<Self, EncoderError> {
        let invalid = EncoderError::NonPositiveOrNonFinite {
            parameter: "tick_seconds",
        };
        if !tick_seconds.is_finite() || tick_seconds <= 0.0 {
            return Err(invalid);
        }
        let nanos = (tick_seconds * 1e9).round();
        // Exclusive upper bound: `u64::MAX as f64` rounds *up* to 2^64, so an
        // inclusive bound would admit 2^64 nanoseconds and let the `as u64` cast
        // silently saturate it back to `u64::MAX`. Below 2^64 the cast is exact.
        if !(1.0..(u64::MAX as f64)).contains(&nanos) {
            return Err(invalid);
        }
        Self::try_from_nanos(nanos as u64).map_err(|_| invalid)
    }

    /// Builds a timebase from a tick *rate* in hertz (ticks per second).
    ///
    /// # Errors
    ///
    /// Same conditions as [`try_from_seconds`](Self::try_from_seconds), applied
    /// to `1.0 / hz`, but reported against `hz`.
    pub fn try_from_hz(hz: f64) -> Result<Self, EncoderError> {
        // Report the parameter the caller supplied, not the one this delegates to.
        let invalid = EncoderError::NonPositiveOrNonFinite { parameter: "hz" };
        if !hz.is_finite() || hz <= 0.0 {
            return Err(invalid);
        }
        Self::try_from_seconds(1.0 / hz).map_err(|_| invalid)
    }

    /// Tick duration in whole nanoseconds.
    #[inline]
    pub const fn tick_nanos(self) -> u64 {
        self.tick_nanos.get()
    }

    /// Tick duration in seconds.
    #[inline]
    pub fn tick_seconds(self) -> f64 {
        self.tick_nanos.get() as f64 / 1e9
    }

    /// Tick rate in hertz.
    #[inline]
    pub fn hz(self) -> f64 {
        1e9 / self.tick_nanos.get() as f64
    }

    /// Converts a spike offset to nanoseconds, saturating at `u64::MAX`.
    #[inline]
    pub const fn offset_nanos(self, offset: TickOffset) -> u64 {
        offset.ticks().saturating_mul(self.tick_nanos.get())
    }

    /// Converts a spike offset to seconds.
    #[inline]
    pub fn offset_seconds(self, offset: TickOffset) -> f64 {
        offset.ticks() as f64 * self.tick_seconds()
    }

    /// Whole ticks contained in a nanosecond duration (rounded down).
    #[inline]
    pub const fn ticks_from_nanos(self, nanos: u64) -> u64 {
        nanos / self.tick_nanos.get()
    }
}

impl TryFrom<u64> for Timebase {
    type Error = EncoderError;

    fn try_from(tick_nanos: u64) -> Result<Self, Self::Error> {
        Self::try_from_nanos(tick_nanos)
    }
}

impl From<Timebase> for u64 {
    fn from(timebase: Timebase) -> Self {
        timebase.tick_nanos.get()
    }
}

impl fmt::Display for Timebase {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}ns/tick", self.tick_nanos.get())
    }
}

/// How one encoder lays its spikes out in time.
///
/// Reported by [`Encoder::time_model`] so a caller can drive any encoder — or a
/// `&mut dyn Encoder` whose concrete type it does not know — on a single
/// timeline without special-casing.
///
/// [`Encoder::time_model`]: crate::Encoder::time_model
///
/// Two independent quantities:
///
/// - [`step_ticks`](Self::step_ticks) — how far the caller's origin advances per
///   call. This is the encoder's *rate of time*.
/// - [`span_ticks`](Self::span_ticks) — the exclusive upper bound on offsets a
///   single call may emit. This is the encoder's *reach*.
///
/// They differ for encoders whose windows overlap: [`PhaseEncoder`] advances one
/// tick per call but places spikes anywhere in the ongoing cycle, so
/// `span_ticks > step_ticks` and [`is_overlapping`](Self::is_overlapping) is
/// `true`.
///
/// [`PhaseEncoder`]: crate::encoders::PhaseEncoder
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TimeModel {
    step_ticks: NonZeroU64,
    span_ticks: NonZeroU64,
    timebase: Option<Timebase>,
}

impl TimeModel {
    const ONE: NonZeroU64 = NonZeroU64::new(1).expect("1 is non-zero");

    /// One call, one tick: every spike lands at [`TickOffset::ZERO`].
    ///
    /// The model for encoders that report *whether* a channel fired in this
    /// step, not *when* within it (rate, delta, derivative, predictive,
    /// temporal, population).
    pub const INSTANT: Self = Self {
        step_ticks: Self::ONE,
        span_ticks: Self::ONE,
        timebase: None,
    };

    /// A non-overlapping window of `span_ticks`: each call is one presentation
    /// and the caller's origin advances by the full window.
    ///
    /// `span_ticks` is clamped to at least 1.
    #[inline]
    pub const fn window(span_ticks: u64) -> Self {
        let span = match NonZeroU64::new(span_ticks) {
            Some(span) => span,
            None => Self::ONE,
        };
        Self {
            step_ticks: span,
            span_ticks: span,
            timebase: None,
        }
    }

    /// A window of `span_ticks` that advances only `step_ticks` per call, so
    /// consecutive calls overlap in time.
    ///
    /// Both arguments are clamped to at least 1, and `step_ticks` is clamped to
    /// at most `span_ticks`. A `TimeModel` therefore never describes a *gap*: a
    /// caller advancing by `step_ticks` can never skip a tick that a call was
    /// entitled to emit into. Pass `step_ticks == span_ticks` (or use
    /// [`window`](Self::window)) for a non-overlapping model.
    #[inline]
    pub const fn overlapping(step_ticks: u64, span_ticks: u64) -> Self {
        let span = match NonZeroU64::new(span_ticks) {
            Some(span) => span,
            None => Self::ONE,
        };
        let step = match NonZeroU64::new(step_ticks) {
            Some(step) if step.get() <= span.get() => step,
            // Zero, or a stride wider than the window the encoder can reach.
            Some(_) => span,
            None => Self::ONE,
        };
        Self {
            step_ticks: step,
            span_ticks: span,
            timebase: None,
        }
    }

    /// Attaches a physical tick duration.
    #[inline]
    pub const fn with_timebase(self, timebase: Timebase) -> Self {
        self.with_timebase_opt(Some(timebase))
    }

    /// Attaches an optional physical tick duration.
    ///
    /// Useful when the duration comes from a fallible conversion: a tick that
    /// rounds below one nanosecond stays dimensionless rather than wrong.
    #[inline]
    pub const fn with_timebase_opt(self, timebase: Option<Timebase>) -> Self {
        Self {
            step_ticks: self.step_ticks,
            span_ticks: self.span_ticks,
            timebase,
        }
    }

    /// Ticks the caller's origin advances after one `encode` / `encode_step` call.
    #[inline]
    pub const fn step_ticks(self) -> u64 {
        self.step_ticks.get()
    }

    /// Exclusive upper bound on offsets emitted by a single call.
    #[inline]
    pub const fn span_ticks(self) -> u64 {
        self.span_ticks.get()
    }

    /// Physical duration of one tick, when the encoder is configured in
    /// physical units.
    #[inline]
    pub const fn timebase(self) -> Option<Timebase> {
        self.timebase
    }

    /// Whether consecutive calls cover overlapping tick ranges.
    #[inline]
    pub const fn is_overlapping(self) -> bool {
        self.span_ticks.get() > self.step_ticks.get()
    }

    /// Whether `offset` is within the span a single call may emit.
    #[inline]
    pub const fn contains(self, offset: TickOffset) -> bool {
        offset.ticks() < self.span_ticks.get()
    }
}

impl Default for TimeModel {
    fn default() -> Self {
        Self::INSTANT
    }
}

/// Caller-owned absolute clock for a stream of encoder calls.
///
/// This crate never holds absolute time. A `TimeCursor` is the small piece of
/// bookkeeping that turns per-call [`TickOffset`] values into a monotonically
/// advancing timeline — the conversion every simulator and hardware adapter
/// needs, written once:
///
/// ```rust
/// use axon_encoder::prelude::*;
/// # fn main() -> Result<(), EncoderError> {
/// // 2 kHz over 1 ms steps: exactly two whole spikes per step.
/// let mut encoder = RateEncoder::try_new(2000.0, 2000.0, (0.0, 1.0), 0.001)?;
/// let mut cursor = TimeCursor::new(encoder.time_model());
///
/// let mut absolute_nanos = Vec::new();
/// for _ in 0..4 {
///     let out = encoder.encode_step(&[1.0]);
///     absolute_nanos.extend(
///         out.spikes
///             .iter()
///             .filter_map(|spike| cursor.absolute_nanos(spike.timestamp)),
///     );
///     cursor.advance();
/// }
/// // dt = 1 ms, so four steps span 4 ms of encoder time.
/// assert_eq!(cursor.origin(), 4);
/// assert_eq!(absolute_nanos, vec![0, 0, 1_000_000, 1_000_000, 2_000_000, 2_000_000, 3_000_000, 3_000_000]);
/// # Ok(())
/// # }
/// ```
///
/// # Saturating convenience versus checked accounting
///
/// [`absolute`](Self::absolute), [`absolute_nanos`](Self::absolute_nanos),
/// [`advance`](Self::advance), and [`advance_by`](Self::advance_by) saturate at
/// `u64::MAX`. That is the right default for examples and short runs: they never
/// panic. They can, however, collapse distinct later events onto the same
/// timestamp once the timeline is exhausted.
///
/// Long-running runtimes that must detect horizon exhaustion should use
/// [`checked_absolute`](Self::checked_absolute),
/// [`checked_absolute_nanos`](Self::checked_absolute_nanos),
/// [`checked_advance`](Self::checked_advance), and
/// [`checked_advance_by`](Self::checked_advance_by), and fail closed when those
/// return `None`. Mutating checked advances leave the origin unchanged on
/// overflow.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TimeCursor {
    model: TimeModel,
    origin: u64,
}

impl TimeCursor {
    /// Starts a cursor at absolute tick 0.
    #[inline]
    pub const fn new(model: TimeModel) -> Self {
        Self { model, origin: 0 }
    }

    /// Starts a cursor at an arbitrary absolute tick.
    #[inline]
    pub const fn starting_at(model: TimeModel, origin: u64) -> Self {
        Self { model, origin }
    }

    /// The time model this cursor advances by.
    #[inline]
    pub const fn model(self) -> TimeModel {
        self.model
    }

    /// Absolute tick of the current call's start.
    #[inline]
    pub const fn origin(self) -> u64 {
        self.origin
    }

    /// Absolute tick of a spike emitted by the current call, saturating at
    /// `u64::MAX`.
    ///
    /// This is the convenience path: it never panics, but once the sum overflows
    /// every later offset maps to the same timestamp. Use
    /// [`checked_absolute`](Self::checked_absolute) when a runtime must detect
    /// that collapse.
    #[inline]
    pub const fn absolute(self, offset: TickOffset) -> u64 {
        self.origin.saturating_add(offset.ticks())
    }

    /// Absolute tick of a spike emitted by the current call, or `None` when
    /// `origin + offset` exceeds `u64::MAX`.
    #[inline]
    pub const fn checked_absolute(self, offset: TickOffset) -> Option<u64> {
        self.origin.checked_add(offset.ticks())
    }

    /// Absolute nanoseconds of a spike emitted by the current call, or `None`
    /// when the encoder reports no [`Timebase`].
    ///
    /// The tick count saturates at `u64::MAX` before multiplying by the tick
    /// duration, and the product itself saturates, so a horizon-exhausted cursor
    /// reports `u64::MAX` nanoseconds rather than wrapping. Use
    /// [`checked_absolute_nanos`](Self::checked_absolute_nanos) when that
    /// collapse must be detected.
    #[inline]
    pub const fn absolute_nanos(self, offset: TickOffset) -> Option<u64> {
        match self.model.timebase() {
            Some(timebase) => Some(self.absolute(offset).saturating_mul(timebase.tick_nanos())),
            None => None,
        }
    }

    /// Absolute nanoseconds of a spike emitted by the current call, or `None`
    /// when the encoder reports no [`Timebase`] or when the conversion is not
    /// representable in `u64`.
    ///
    /// Representable means both `origin + offset` and that sum times the tick
    /// duration fit in `u64`. Either overflow, or a missing timebase, yields
    /// `None`.
    #[inline]
    pub const fn checked_absolute_nanos(self, offset: TickOffset) -> Option<u64> {
        match self.model.timebase() {
            Some(timebase) => match self.checked_absolute(offset) {
                Some(ticks) => ticks.checked_mul(timebase.tick_nanos()),
                None => None,
            },
            None => None,
        }
    }

    /// Absolute ticks for a whole call's spikes, in emission order.
    #[inline]
    pub fn absolute_times(self, spikes: &[SpikeEvent]) -> impl Iterator<Item = u64> + '_ {
        spikes
            .iter()
            .map(move |spike| self.absolute(spike.timestamp))
    }

    /// Advances past one encoder call, returning the new origin.
    ///
    /// Saturates at `u64::MAX`. A runtime that must stop when the timeline can
    /// no longer advance should use [`checked_advance`](Self::checked_advance).
    #[inline]
    pub const fn advance(&mut self) -> u64 {
        self.origin = self.origin.saturating_add(self.model.step_ticks());
        self.origin
    }

    /// Advances past one encoder call, returning the new origin, or `None` when
    /// `origin + step_ticks` would overflow `u64`.
    ///
    /// On overflow the cursor is left unchanged, so a runtime can fail closed
    /// without losing its last valid origin:
    ///
    /// ```rust
    /// use axon_encoder::prelude::*;
    ///
    /// let mut cursor = TimeCursor::starting_at(TimeModel::window(11), u64::MAX - 5);
    /// assert_eq!(cursor.checked_advance(), None);
    /// assert_eq!(cursor.origin(), u64::MAX - 5);
    ///
    /// // Saturating convenience still clamps if the caller wants that:
    /// assert_eq!(cursor.advance(), u64::MAX);
    /// ```
    #[inline]
    pub const fn checked_advance(&mut self) -> Option<u64> {
        self.checked_advance_by(1)
    }

    /// Advances past `calls` encoder calls, returning the new origin.
    ///
    /// Both the `calls * step_ticks` product and the subsequent addition
    /// saturate at `u64::MAX`. Use
    /// [`checked_advance_by`](Self::checked_advance_by) when that collapse must
    /// be detected.
    #[inline]
    pub const fn advance_by(&mut self, calls: u64) -> u64 {
        self.origin = self
            .origin
            .saturating_add(calls.saturating_mul(self.model.step_ticks()));
        self.origin
    }

    /// Advances past `calls` encoder calls, returning the new origin, or `None`
    /// when `calls * step_ticks` or `origin + that product` would overflow `u64`.
    ///
    /// On overflow the cursor is left unchanged. Zero calls always succeed and
    /// leave the origin as-is, including when the origin is already `u64::MAX`.
    #[inline]
    pub const fn checked_advance_by(&mut self, calls: u64) -> Option<u64> {
        let delta = match calls.checked_mul(self.model.step_ticks()) {
            Some(delta) => delta,
            None => return None,
        };
        let origin = match self.origin.checked_add(delta) {
            Some(origin) => origin,
            None => return None,
        };
        self.origin = origin;
        Some(origin)
    }

    /// Returns the origin to 0, keeping the model. Pair with `Encoder::reset`.
    #[inline]
    pub const fn reset(&mut self) {
        self.origin = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Origins, offsets, and call counts that sit on the `u64` horizon.
    const HORIZON: [u64; 8] = [
        0,
        1,
        2,
        u64::MAX / 2,
        u64::MAX / 2 + 1,
        u64::MAX - 2,
        u64::MAX - 1,
        u64::MAX,
    ];

    fn physical(tick_nanos: u64) -> TimeModel {
        TimeModel::INSTANT.with_timebase(Timebase::from_nanos(
            NonZeroU64::new(tick_nanos).expect("tick duration is non-zero"),
        ))
    }

    #[test]
    fn checked_absolute_returns_exact_sum_or_none() {
        let cursor = TimeCursor::starting_at(TimeModel::INSTANT, 10);
        assert_eq!(cursor.checked_absolute(TickOffset::ZERO), Some(10));
        assert_eq!(cursor.checked_absolute(TickOffset::new(5)), Some(15));

        let at_max = TimeCursor::starting_at(TimeModel::INSTANT, u64::MAX);
        assert_eq!(at_max.checked_absolute(TickOffset::ZERO), Some(u64::MAX));
        assert_eq!(at_max.checked_absolute(TickOffset::new(1)), None);

        let near_max = TimeCursor::starting_at(TimeModel::INSTANT, u64::MAX - 5);
        assert_eq!(
            near_max.checked_absolute(TickOffset::new(5)),
            Some(u64::MAX)
        );
        assert_eq!(near_max.checked_absolute(TickOffset::new(6)), None);
    }

    #[test]
    fn saturating_absolute_is_unchanged_at_the_horizon() {
        let cursor = TimeCursor::starting_at(TimeModel::INSTANT, u64::MAX - 1);
        assert_eq!(cursor.absolute(TickOffset::new(5)), u64::MAX);
        assert_eq!(cursor.checked_absolute(TickOffset::new(5)), None);
        assert_eq!(cursor.origin(), u64::MAX - 1);
    }

    #[test]
    fn prop_checked_absolute_agrees_with_saturating_addition() {
        for origin in HORIZON {
            for offset in HORIZON {
                let cursor = TimeCursor::starting_at(TimeModel::INSTANT, origin);
                let saturating = cursor.absolute(TickOffset::new(offset));
                let checked = cursor.checked_absolute(TickOffset::new(offset));
                match origin.checked_add(offset) {
                    Some(sum) => {
                        assert_eq!(checked, Some(sum), "{origin} + {offset}");
                        assert_eq!(saturating, sum, "{origin} + {offset}");
                    }
                    None => {
                        assert_eq!(checked, None, "{origin} + {offset} overflow");
                        assert_eq!(saturating, u64::MAX, "{origin} + {offset} saturate");
                    }
                }
            }
        }
    }

    #[test]
    fn checked_absolute_nanos_requires_a_timebase() {
        let dimensionless = TimeCursor::new(TimeModel::INSTANT);
        assert_eq!(dimensionless.checked_absolute_nanos(TickOffset::ZERO), None);
        assert_eq!(dimensionless.absolute_nanos(TickOffset::ZERO), None);
    }

    #[test]
    fn checked_absolute_nanos_returns_exact_product_or_none() {
        let cursor = TimeCursor::starting_at(physical(1_000_000), 3);
        assert_eq!(
            cursor.checked_absolute_nanos(TickOffset::new(1)),
            Some(4_000_000)
        );

        let unit = TimeCursor::starting_at(physical(1), u64::MAX);
        assert_eq!(
            unit.checked_absolute_nanos(TickOffset::ZERO),
            Some(u64::MAX)
        );
        assert_eq!(unit.checked_absolute_nanos(TickOffset::new(1)), None);

        // Addition fits; multiplication does not.
        let doubled = TimeCursor::starting_at(physical(2), u64::MAX / 2 + 1);
        assert_eq!(
            doubled.checked_absolute(TickOffset::ZERO),
            Some(u64::MAX / 2 + 1)
        );
        assert_eq!(doubled.checked_absolute_nanos(TickOffset::ZERO), None);
        assert_eq!(doubled.absolute_nanos(TickOffset::ZERO), Some(u64::MAX));

        let max_tick = TimeCursor::starting_at(physical(u64::MAX), 0);
        assert_eq!(
            max_tick.checked_absolute_nanos(TickOffset::new(1)),
            Some(u64::MAX)
        );
        assert_eq!(max_tick.checked_absolute_nanos(TickOffset::new(2)), None);
    }

    #[test]
    fn prop_checked_absolute_nanos_agrees_with_saturating_multiplication() {
        let tick_nanos = [1u64, 2, 3, 1_000_000, u64::MAX / 2, u64::MAX];
        for nanos in tick_nanos {
            for origin in HORIZON {
                for offset in HORIZON {
                    let cursor = TimeCursor::starting_at(physical(nanos), origin);
                    let saturating = cursor.absolute_nanos(TickOffset::new(offset));
                    let checked = cursor.checked_absolute_nanos(TickOffset::new(offset));
                    match origin
                        .checked_add(offset)
                        .and_then(|ticks| ticks.checked_mul(nanos))
                    {
                        Some(product) => {
                            assert_eq!(checked, Some(product), "{origin}+{offset}×{nanos}");
                            assert_eq!(saturating, Some(product), "{origin}+{offset}×{nanos}");
                        }
                        None => {
                            assert_eq!(checked, None, "{origin}+{offset}×{nanos} overflow");
                            assert_eq!(
                                saturating,
                                Some(u64::MAX),
                                "{origin}+{offset}×{nanos} saturate"
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn checked_advance_is_atomic_on_overflow() {
        let mut cursor = TimeCursor::starting_at(TimeModel::window(11), u64::MAX - 5);
        assert_eq!(cursor.checked_advance(), None);
        assert_eq!(cursor.origin(), u64::MAX - 5);

        let mut at_max = TimeCursor::starting_at(TimeModel::INSTANT, u64::MAX);
        assert_eq!(at_max.checked_advance(), None);
        assert_eq!(at_max.origin(), u64::MAX);
        assert_eq!(at_max.advance(), u64::MAX);
    }

    #[test]
    fn checked_advance_returns_the_new_origin_when_representable() {
        let mut cursor = TimeCursor::new(TimeModel::window(11));
        assert_eq!(cursor.checked_advance(), Some(11));
        assert_eq!(cursor.origin(), 11);

        let mut near_max = TimeCursor::starting_at(TimeModel::INSTANT, u64::MAX - 1);
        assert_eq!(near_max.checked_advance(), Some(u64::MAX));
        assert_eq!(near_max.origin(), u64::MAX);
    }

    #[test]
    fn checked_advance_by_zero_calls_is_a_no_op() {
        for origin in HORIZON {
            let mut cursor = TimeCursor::starting_at(TimeModel::window(11), origin);
            assert_eq!(cursor.checked_advance_by(0), Some(origin));
            assert_eq!(cursor.origin(), origin);
            assert_eq!(cursor.advance_by(0), origin);
        }
    }

    #[test]
    fn checked_advance_by_reports_multiplication_overflow() {
        let mut cursor = TimeCursor::new(TimeModel::window(2));
        assert_eq!(cursor.checked_advance_by(u64::MAX), None);
        assert_eq!(cursor.origin(), 0);

        let mut wide = TimeCursor::new(TimeModel::window(u64::MAX));
        assert_eq!(wide.checked_advance_by(2), None);
        assert_eq!(wide.origin(), 0);
        assert_eq!(wide.checked_advance_by(1), Some(u64::MAX));
        assert_eq!(wide.origin(), u64::MAX);
    }

    #[test]
    fn checked_advance_by_reports_addition_overflow() {
        let mut cursor = TimeCursor::starting_at(TimeModel::INSTANT, 1);
        assert_eq!(cursor.checked_advance_by(u64::MAX), None);
        assert_eq!(cursor.origin(), 1);

        let mut from_zero = TimeCursor::new(TimeModel::INSTANT);
        assert_eq!(from_zero.checked_advance_by(u64::MAX), Some(u64::MAX));
        assert_eq!(from_zero.origin(), u64::MAX);
    }

    #[test]
    fn saturating_advance_by_is_unchanged() {
        let mut cursor = TimeCursor::starting_at(TimeModel::INSTANT, u64::MAX - 1);
        assert_eq!(cursor.advance_by(u64::MAX), u64::MAX);
        assert_eq!(cursor.origin(), u64::MAX);
    }

    #[test]
    fn prop_checked_advance_by_agrees_with_saturating_mul_then_add() {
        for step in HORIZON {
            if step == 0 {
                continue;
            }
            let model = TimeModel::window(step);
            for origin in HORIZON {
                for calls in HORIZON {
                    let mut checked_cursor = TimeCursor::starting_at(model, origin);
                    let mut saturating_cursor = TimeCursor::starting_at(model, origin);
                    let checked = checked_cursor.checked_advance_by(calls);
                    let saturating = saturating_cursor.advance_by(calls);
                    match calls
                        .checked_mul(step)
                        .and_then(|delta| origin.checked_add(delta))
                    {
                        Some(sum) => {
                            assert_eq!(checked, Some(sum), "{origin} + {calls}×{step}");
                            assert_eq!(checked_cursor.origin(), sum);
                            assert_eq!(saturating, sum);
                        }
                        None => {
                            assert_eq!(checked, None, "{origin} + {calls}×{step} overflow");
                            assert_eq!(
                                checked_cursor.origin(),
                                origin,
                                "checked_advance_by must not mutate on overflow"
                            );
                            assert_eq!(saturating, u64::MAX, "{origin} + {calls}×{step} saturate");
                        }
                    }
                }
            }
        }
    }
}
