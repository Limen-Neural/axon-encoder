//! Bounded buffering and explicit flush policy for a borrowed [`Encoder`].
//!
//! An [`Encoder`] emits spikes per call and hands them straight to the caller.
//! Some runtimes want the opposite: encode many steps eagerly, then deliver the
//! accumulated spikes to a downstream consumer in coarser, back-pressured
//! batches. [`StreamingEncoder`] is that seam. It borrows an encoder, drives its
//! [`encode_step_into`](Encoder::encode_step_into) path into an internal bounded
//! queue, and delivers whole batches to a [`BatchSink`] under a
//! [`FlushPolicy`] the caller chooses.
//!
//! # Ownership: everything is borrowed
//!
//! [`StreamingEncoder`] owns neither the encoder it drives nor the target its
//! batches land in. It holds the encoder as `&'a mut E` where
//! `E: `[`Encoder`]` + ?Sized`, so it wraps a concrete encoder *or* a
//! `&mut dyn `[`Encoder`] trait object without a generic monomorphization per
//! type. The caller keeps ownership of the encoder for the wrapper's lifetime
//! and gets it back when the borrow ends; nothing is moved in. Delivery targets
//! are borrowed the same way: [`encode_step`](StreamingEncoder::encode_step)
//! and [`flush_into`](StreamingEncoder::flush_into) take a
//! `&mut dyn `[`BatchSink`], and the caller owns whatever storage that sink
//! writes into (a `Vec<`[`SpikeEvent`]`>`, a ring queue, a hardware adapter, or
//! just a closure). This mirrors how [`Encoder`] hands spikes to a caller-owned
//! [`SpikeSink`](crate::sink::SpikeSink); the wrapper allocates only its own
//! bounded queue.
//!
//! # What a batch is
//!
//! Each accepted [`encode_step`](StreamingEncoder::encode_step) call becomes at
//! most one [`SpikeBatch`]: the spikes that call produced, tagged with the
//! *call sequence number* and the *origin*, the absolute tick at which that
//! call started, as reported by [`TimeCursor::origin`] before the cursor
//! advanced past the call. Spike timestamps inside a batch stay **call-relative
//! [`TickOffset`](crate::time::TickOffset)s**; they are never rebased. Absolute time for a spike is
//! `batch.origin() + spike.timestamp.ticks()`.
//!
//! # Memory bound
//!
//! The buffering is hard-bounded, not merely advisory. At any instant the
//! wrapper holds at most `capacity` queued spikes, plus the output staged or
//! held for the one call in flight. The staging buffer for ordinary calls is
//! reused. A call whose output is larger than `capacity` is delivered straight
//! through as a single batch; its oversized allocation is released after
//! delivery. Oversized held storage is released after flush or reset. There is
//! no per-batch overhead that scales with call count once batches are delivered:
//! the queue reclaims its contiguous storage on every full drain.
//!
//! # Capacity and blocking
//!
//! The queue holds at most `capacity` spikes. What happens when a call's output
//! does not fit depends on the [`FlushPolicy`]:
//!
//! - [`FlushPolicy::Manual`] never delivers on its own. When a call does not
//!   fit, its output is *held* and the wrapper becomes
//!   [`blocked`](StreamingEncoder::is_blocked); the next
//!   [`encode_step`](StreamingEncoder::encode_step) returns
//!   [`StreamingError::Backpressure`] until the caller
//!   [`flush_into`](StreamingEncoder::flush_into)s.
//! - [`FlushPolicy::OnCapacity`] delivers every queued batch (reason
//!   [`FlushReason::Capacity`]) when a new call would overflow the queue.
//! - [`FlushPolicy::OnCapacityOrAge`] additionally delivers (reason
//!   [`FlushReason::Age`]) once the oldest queued batch has aged
//!   `max_age_ticks` encoder ticks. The age deadline is checked on every
//!   accepted [`encode_step`](StreamingEncoder::encode_step) call, including
//!   calls whose encoder output is empty: an accepted call always advances the
//!   cursor, so a queued batch can age out and be delivered on an empty call
//!   even though that call queues nothing of its own.
//!
//! The encoder is only ever invoked by
//! [`encode_step`](StreamingEncoder::encode_step); flushing never touches it.
//! When capacity and age would both apply to an older queued batch on one
//! call, capacity is checked first and [`StepReport::reason`] is
//! [`FlushReason::Capacity`]. Age is checked after admitting the new batch.
//!
//! # Age is measured in ticks, never wall-clock
//!
//! This crate owns no clock, and neither does the wrapper. The "age" that
//! [`FlushPolicy::OnCapacityOrAge`] acts on is a count of encoder **ticks**: the
//! difference between the current [`TimeCursor`] origin and the oldest queued
//! batch's origin, both advanced by
//! [`TimeModel::step_ticks`](crate::time::TimeModel::step_ticks) once per
//! accepted call. The wrapper never reads a system clock, so its behavior is
//! fully determined by the calls the caller makes.
//!
//! A caller that wants a *real-time* deadline builds it themselves without the
//! wrapper touching a clock: read [`pending_age_ticks`] (or your own timer)
//! and, when your policy says so, call [`flush_into`] explicitly. That keeps
//! the wall-clock decision in caller code where the clock actually lives.
//!
//! [`pending_age_ticks`]: StreamingEncoder::pending_age_ticks
//! [`flush_into`]: StreamingEncoder::flush_into
//!
//! # Reset
//!
//! [`reset`](StreamingEncoder::reset) forwards [`Encoder::reset`] to the
//! borrowed encoder and **discards** every queued and held spike; it is not a
//! flush. Call [`flush_into`] first if the buffered output still matters. The
//! call sequence counter and the [`TimeCursor`] are not rewound, so the caller
//! timeline stays monotonic across a reset.
//! Dropping the wrapper also discards pending output: it has no sink to flush
//! into during destruction. Call [`flush_into`] before dropping it when those
//! batches must be delivered.
//!
//! # Sequence exhaustion
//!
//! After assigning sequence `u64::MAX`, the wrapper returns
//! [`StreamingError::SequenceExhausted`] before invoking the encoder. Reset
//! preserves this exhausted state because it cannot rewind sequence numbers.
//!
//! # Sink panics
//!
//! Delivery follows a take-before-call discipline: a batch is removed from the
//! queue before it is handed to the sink. So if a [`BatchSink`] panics, the
//! batches already delivered in that call are **not** replayed on a later
//! flush, and the in-flight batch is lost. A wrapper whose sink panicked should
//! be [`reset`](StreamingEncoder::reset) before reuse. This mirrors the
//! [`SpikeSink`](crate::sink::SpikeSink) panic contract, where a panic ends the
//! call and keeps the accepted prefix.
//!
//! # Why this is not an [`Encoder`]
//!
//! [`StreamingEncoder`] deliberately does not implement [`Encoder`]. The trait's
//! contract is per-call: the spikes an [`encode_step`](Encoder::encode_step)
//! produces belong to that call and are returned or written to the sink before
//! it returns. The whole point of this wrapper is the opposite: delivery is
//! deferred and policy-driven, so a single [`encode_step`](StreamingEncoder::encode_step)
//! may deliver nothing, deliver several earlier calls' batches, or be rejected
//! outright with [`StreamingError::Backpressure`]. That cannot honor the trait's
//! "output belongs to this call" guarantee, so the buffering API is expressed as
//! inherent methods instead, and [`reset`](StreamingEncoder::reset) is an
//! inherent method rather than [`Encoder::reset`].

use std::collections::VecDeque;

use crate::Encoder;
use crate::error::{EncoderError, StreamingError};
use crate::time::{TimeCursor, TimeModel};
use crate::types::SpikeEvent;

/// When a [`StreamingEncoder`] delivers its queued batches automatically.
///
/// The policy is fixed at construction and read back with
/// [`StreamingEncoder::policy`]. See the [module docs](self) for the full
/// behavior of each variant.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FlushPolicy {
    /// Never deliver automatically; the caller flushes explicitly.
    ///
    /// A call whose output does not fit the remaining capacity is held and the
    /// wrapper becomes blocked until [`StreamingEncoder::flush_into`] runs.
    Manual,
    /// Deliver every queued batch when a new call's output would overflow the
    /// capacity.
    OnCapacity,
    /// Like [`OnCapacity`](Self::OnCapacity), and additionally deliver once the
    /// oldest queued batch has aged `max_age_ticks` encoder ticks.
    ///
    /// Age is measured in the tick unit of
    /// [`TimeModel::step_ticks`](crate::time::TimeModel::step_ticks): it is the
    /// difference between the current cursor origin and the oldest queued
    /// batch's origin. `max_age_ticks` must be non-zero.
    OnCapacityOrAge {
        /// Age, in encoder ticks, at which the oldest queued batch triggers a
        /// flush.
        max_age_ticks: u64,
    },
}

/// Why a [`StreamingEncoder`] delivered its batches.
///
/// Reported through [`StepReport::reason`] and [`FlushReport::reason`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FlushReason {
    /// An explicit [`StreamingEncoder::flush_into`] call.
    Manual,
    /// A new call would have overflowed the capacity.
    Capacity,
    /// The oldest queued batch reached the configured age.
    Age,
}

/// A borrowed view of one delivered batch of spikes.
///
/// A batch corresponds to exactly one accepted
/// [`encode_step`](StreamingEncoder::encode_step) call. It carries the call's
/// monotonically increasing [`sequence`](Self::sequence) number, its
/// [`origin`](Self::origin) (the absolute tick the call started at, from
/// [`TimeCursor::origin`]), and a slice of the [`spikes`](Self::spikes) that
/// call produced.
///
/// # Time
///
/// Spike timestamps in [`spikes`](Self::spikes) are the unchanged
/// call-relative [`TickOffset`](crate::time::TickOffset)s the encoder emitted:
/// they are never rebased onto the absolute timeline. The absolute tick of a spike is
/// `batch.origin() + spike.timestamp.ticks()`; convert with a
/// [`TimeCursor`] if a physical
/// [`Timebase`](crate::time::Timebase) is needed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SpikeBatch<'a> {
    sequence: u64,
    origin: u64,
    spikes: &'a [SpikeEvent],
}

impl<'a> SpikeBatch<'a> {
    /// The call sequence number of this batch.
    ///
    /// Sequence numbers are assigned once per accepted call, increase
    /// monotonically, and are not rewound by [`StreamingEncoder::reset`].
    #[inline]
    pub fn sequence(&self) -> u64 {
        self.sequence
    }

    /// The absolute tick at which the emitting call started.
    ///
    /// Add a spike's [`TickOffset::ticks`](crate::time::TickOffset::ticks) to this to get its absolute tick.
    #[inline]
    pub fn origin(&self) -> u64 {
        self.origin
    }

    /// The spikes produced by the emitting call, with call-relative timestamps.
    #[inline]
    pub fn spikes(&self) -> &[SpikeEvent] {
        self.spikes
    }
}

/// A destination for whole [`SpikeBatch`]es delivered by a [`StreamingEncoder`].
///
/// This is to batches what [`SpikeSink`](crate::sink::SpikeSink) is to
/// individual spikes: a single-method, object-safe trait for "somewhere a batch
/// can go". [`StreamingEncoder::encode_step`] and
/// [`StreamingEncoder::flush_into`] take `&mut dyn BatchSink`.
///
/// # Contract
///
/// - Batches arrive **oldest first**. Their [`sequence`](SpikeBatch::sequence)
///   numbers are strictly increasing across a run of deliveries.
/// - A batch borrows the wrapper's storage only for the duration of the
///   [`deliver`](Self::deliver) call. A sink that needs to retain spikes copies
///   them out.
/// - A sink that **panics** ends the delivering call. Batches already delivered
///   in that call are not redelivered on a later flush, the in-flight batch is
///   lost, and the wrapper must be [`reset`](StreamingEncoder::reset) before
///   reuse, mirroring the [`SpikeSink`](crate::sink::SpikeSink) panic contract.
///
/// A closure of type `FnMut(SpikeBatch<'_>)` implements `BatchSink` through a
/// blanket impl, so the common case needs no named type.
///
/// # Examples
///
/// ```rust
/// use axon_encoder::prelude::*;
/// # fn main() -> Result<(), EncoderError> {
/// let mut encoder = RateEncoder::try_new(0.0, 100.0, (0.0, 1.0), 0.1)?;
/// let mut streaming = StreamingEncoder::try_new(&mut encoder, 8, FlushPolicy::OnCapacity)?;
///
/// let mut delivered_batches = 0usize;
/// let mut sink = |batch: SpikeBatch<'_>| {
///     delivered_batches += 1;
///     let _ = (batch.sequence(), batch.origin(), batch.spikes());
/// };
///
/// streaming
///     .encode_step(&[1.0], &mut sink)
///     .expect("first step is never blocked");
/// streaming.flush_into(&mut sink);
/// # Ok(())
/// # }
/// ```
pub trait BatchSink {
    /// Accepts one batch of spikes.
    fn deliver(&mut self, batch: SpikeBatch<'_>);
}

/// Any `FnMut(SpikeBatch<'_>)` is a batch sink.
impl<F: FnMut(SpikeBatch<'_>)> BatchSink for F {
    #[inline]
    fn deliver(&mut self, batch: SpikeBatch<'_>) {
        self(batch);
    }
}

/// The outcome of one [`StreamingEncoder::encode_step`] call.
///
/// Read the fields through the accessor methods. A step may deliver zero or
/// more batches (when the policy triggers a flush), leave some spikes queued,
/// and/or leave the wrapper blocked.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StepReport {
    reason: Option<FlushReason>,
    delivered_batches: usize,
    queued_spikes: usize,
    blocked: bool,
}

impl StepReport {
    /// Why batches were delivered, or `None` if none were. Capacity is checked
    /// before age; an older batch removed for capacity is reported as
    /// [`FlushReason::Capacity`] even if it reached the age deadline too.
    #[inline]
    pub fn reason(&self) -> Option<FlushReason> {
        self.reason
    }

    /// How many batches were delivered to the sink during the step.
    #[inline]
    pub fn delivered_batches(&self) -> usize {
        self.delivered_batches
    }

    /// How many spikes remain queued after the step.
    #[inline]
    pub fn queued_spikes(&self) -> usize {
        self.queued_spikes
    }

    /// Whether the wrapper is blocked (holding output under
    /// [`FlushPolicy::Manual`]) after the step.
    #[inline]
    pub fn blocked(&self) -> bool {
        self.blocked
    }
}

/// The outcome of one [`StreamingEncoder::flush_into`] call.
///
/// Read the fields through the accessor methods.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FlushReport {
    reason: Option<FlushReason>,
    delivered_batches: usize,
}

impl FlushReport {
    /// [`FlushReason::Manual`] if anything was delivered, `None` otherwise.
    #[inline]
    pub fn reason(&self) -> Option<FlushReason> {
        self.reason
    }

    /// How many batches were delivered to the sink during the flush.
    #[inline]
    pub fn delivered_batches(&self) -> usize {
        self.delivered_batches
    }
}

/// A queued batch record: where its spikes live in the contiguous storage,
/// plus the call metadata delivered alongside them.
#[derive(Clone, Copy, Debug)]
struct BatchRecord {
    sequence: u64,
    origin: u64,
    start: usize,
    len: usize,
}

/// A staged-but-held batch, kept when [`FlushPolicy::Manual`] output does not
/// fit. Its spikes live in `held_spikes`, not the main queue storage.
#[derive(Clone, Copy, Debug)]
struct HeldBatch {
    sequence: u64,
    origin: u64,
    len: usize,
}

/// A bounded, explicitly flushed buffer wrapped around a borrowed [`Encoder`].
///
/// See the [module docs](self) for the batching model, capacity semantics, and
/// flush policies. `StreamingEncoder` deliberately does **not** implement
/// [`Encoder`]: its deferred, policy-driven delivery is incompatible with the
/// per-call output contract of that trait.
///
/// # Examples
///
/// ```rust
/// use axon_encoder::prelude::*;
/// # fn main() -> Result<(), EncoderError> {
/// let mut encoder = RateEncoder::try_new(0.0, 100.0, (0.0, 1.0), 0.1)?;
/// let mut streaming = StreamingEncoder::try_new(&mut encoder, 16, FlushPolicy::OnCapacity)?;
/// assert_eq!(streaming.capacity(), 16);
/// assert!(streaming.is_empty());
///
/// let mut batches = 0usize;
/// let report = streaming
///     .encode_step(&[1.0], &mut |_b: SpikeBatch<'_>| batches += 1)
///     .expect("first step is never blocked");
/// assert!(!report.blocked());
///
/// let flushed = streaming.flush_into(&mut |_b: SpikeBatch<'_>| batches += 1);
/// assert!(streaming.is_empty());
/// let _ = flushed.delivered_batches();
/// # Ok(())
/// # }
/// ```
pub struct StreamingEncoder<'a, E: Encoder + ?Sized> {
    encoder: &'a mut E,
    capacity: usize,
    policy: FlushPolicy,
    /// Contiguous storage for all queued (non-held) batch spikes.
    storage: Vec<SpikeEvent>,
    /// Batch records indexing into `storage`, oldest first.
    queue: VecDeque<BatchRecord>,
    /// Reusable staging buffer for the current call's encoder output.
    staging: Vec<SpikeEvent>,
    /// Spikes of a held batch (Manual policy, output did not fit).
    held_spikes: Vec<SpikeEvent>,
    /// Metadata for the held batch, when one is held.
    held: Option<HeldBatch>,
    /// Caller-timeline cursor, advanced once per accepted call.
    cursor: TimeCursor,
    /// Next call sequence, or `None` after assigning `u64::MAX`.
    sequence: Option<u64>,
}

impl<'a, E: Encoder + ?Sized> StreamingEncoder<'a, E> {
    /// Wraps `encoder` in a bounded buffer with the given `capacity` and
    /// `policy`.
    ///
    /// The [`TimeCursor`] is initialized from
    /// `encoder.time_model()`.
    ///
    /// # Errors
    ///
    /// - [`EncoderError::CountMustBePositive`] with `parameter: "capacity"`
    ///   when `capacity` is zero.
    /// - [`EncoderError::CountMustBePositive`] with `parameter: "max_age_ticks"`
    ///   when `policy` is [`FlushPolicy::OnCapacityOrAge`] with a zero
    ///   `max_age_ticks`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use axon_encoder::prelude::*;
    /// # fn main() -> Result<(), EncoderError> {
    /// let mut encoder = RateEncoder::try_new(0.0, 100.0, (0.0, 1.0), 0.1)?;
    /// let streaming = StreamingEncoder::try_new(&mut encoder, 4, FlushPolicy::Manual)?;
    /// assert_eq!(streaming.capacity(), 4);
    /// # Ok(())
    /// # }
    /// ```
    pub fn try_new(
        encoder: &'a mut E,
        capacity: usize,
        policy: FlushPolicy,
    ) -> Result<Self, EncoderError> {
        if capacity == 0 {
            return Err(EncoderError::CountMustBePositive {
                parameter: "capacity",
            });
        }
        if let FlushPolicy::OnCapacityOrAge { max_age_ticks: 0 } = policy {
            return Err(EncoderError::CountMustBePositive {
                parameter: "max_age_ticks",
            });
        }
        let cursor = TimeCursor::new(encoder.time_model());
        Ok(Self {
            encoder,
            capacity,
            policy,
            storage: Vec::new(),
            queue: VecDeque::new(),
            staging: Vec::new(),
            held_spikes: Vec::new(),
            held: None,
            cursor,
            sequence: Some(0),
        })
    }

    /// The configured spike capacity.
    #[inline]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// The configured flush policy.
    #[inline]
    pub fn policy(&self) -> FlushPolicy {
        self.policy
    }

    /// The total number of spikes currently queued (excludes a held batch,
    /// which is not part of the bounded queue).
    #[inline]
    pub fn buffered_spikes(&self) -> usize {
        self.storage.len()
    }

    /// The number of queued batches (excludes a held batch).
    #[inline]
    pub fn pending_batches(&self) -> usize {
        self.queue.len()
    }

    /// Whether nothing is queued and no batch is held.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.queue.is_empty() && self.held.is_none()
    }

    /// Whether the wrapper is blocked, holding output under
    /// [`FlushPolicy::Manual`] that did not fit the capacity.
    #[inline]
    pub fn is_blocked(&self) -> bool {
        self.held.is_some()
    }

    /// Ticks from the oldest queued batch's origin to the current cursor
    /// origin, or `0` when the queue is empty.
    #[inline]
    pub fn pending_age_ticks(&self) -> u64 {
        match self.queue.front() {
            Some(record) => self.cursor.origin().saturating_sub(record.origin),
            None => 0,
        }
    }

    /// A copy of the current [`TimeCursor`].
    #[inline]
    pub fn cursor(&self) -> TimeCursor {
        self.cursor
    }

    /// The wrapped encoder's [`TimeModel`].
    #[inline]
    pub fn time_model(&self) -> TimeModel {
        self.encoder.time_model()
    }

    /// Delivers every queued batch to `sink`, oldest first.
    ///
    /// Advances the queue read position *before* each `deliver` call so a
    /// panicking sink never causes redelivery of an already-delivered batch
    /// (mirrors the take-before-call discipline of the internal chunked sink).
    /// Returns the number of batches delivered.
    fn drain_queue(&mut self, sink: &mut dyn BatchSink) -> usize {
        let mut delivered = 0usize;
        while let Some(record) = self.queue.pop_front() {
            delivered += 1;
            let spikes = &self.storage[record.start..record.start + record.len];
            sink.deliver(SpikeBatch {
                sequence: record.sequence,
                origin: record.origin,
                spikes,
            });
        }
        // Fully drained: reclaim the contiguous storage in one shot.
        Self::clear_reusable(&mut self.storage, self.capacity);
        delivered
    }

    /// Retain reusable storage only while its allocation fits the configured
    /// queue bound. Oversized calls can otherwise leave a large idle buffer.
    fn clear_reusable(buffer: &mut Vec<SpikeEvent>, capacity: usize) {
        if buffer.capacity() > capacity {
            *buffer = Vec::new();
        } else {
            buffer.clear();
        }
    }

    /// Encodes one streaming step, buffering its output and delivering batches
    /// per the [`FlushPolicy`].
    ///
    /// # Errors
    ///
    /// Returns [`StreamingError::Backpressure`] when the wrapper is already
    /// blocked (a held [`FlushPolicy::Manual`] batch). In that case the encoder
    /// is **not** invoked and neither the cursor nor the sequence advances;
    /// the caller must [`flush_into`](Self::flush_into) first.
    /// Returns [`StreamingError::SequenceExhausted`] after the call numbered
    /// `u64::MAX`; this also rejects input before invoking the encoder and is
    /// not cleared by [`reset`](Self::reset).
    ///
    /// Otherwise the encoder is invoked exactly once, the cursor and sequence
    /// advance exactly once, and a [`StepReport`] describes the result. Under
    /// [`FlushPolicy::Manual`] no batch is ever delivered during this call; an
    /// oversized-or-overflowing call is held instead. Empty encoder output
    /// queues no batch of its own and makes no sink call for that (nonexistent)
    /// output.
    ///
    /// The age trigger is evaluated on every accepted call, empty output
    /// included: an accepted call always advances the cursor, so under
    /// [`FlushPolicy::OnCapacityOrAge`] a pre-existing queued batch that ages
    /// past `max_age_ticks` on this call is delivered here (reason
    /// [`FlushReason::Age`]), whether or not this call produced any spikes.
    pub fn encode_step(
        &mut self,
        input: &[f32],
        sink: &mut dyn BatchSink,
    ) -> Result<StepReport, StreamingError> {
        // (1) Pre-encode backpressure: reject before touching the encoder or
        // advancing the timeline.
        if let Some(held) = self.held.as_ref() {
            return Err(StreamingError::Backpressure {
                buffered_spikes: self.buffered_spikes().saturating_add(held.len),
                capacity: self.capacity,
            });
        }
        let sequence = self.sequence.ok_or(StreamingError::SequenceExhausted)?;

        // (2) Stage this call's output through the borrowed encoder.
        Self::clear_reusable(&mut self.staging, self.capacity);
        self.encoder.encode_step_into(input, &mut self.staging);

        // (3) Record this call's metadata, then advance exactly once.
        let origin = self.cursor.origin();
        self.cursor.advance();
        self.sequence = sequence.checked_add(1);

        let (mut reason, mut delivered_batches) = self.admit_staged(sequence, origin, sink);

        // Age trigger after admission (OnCapacityOrAge only). Evaluated on
        // every accepted call, empty output included: the cursor advanced for
        // this call, so a pre-existing queued batch can legitimately age past
        // `max_age_ticks` on an empty call and must be delivered here.
        if let FlushPolicy::OnCapacityOrAge { max_age_ticks } = self.policy
            && !self.queue.is_empty()
            && self.pending_age_ticks() >= max_age_ticks
        {
            delivered_batches += self.drain_queue(sink);
            reason = Some(FlushReason::Age);
        }

        let report = self.finish_step(reason, delivered_batches);
        Self::clear_reusable(&mut self.staging, self.capacity);
        Ok(report)
    }

    /// Admit one non-empty staged call under the configured capacity policy.
    /// Empty calls leave the queue alone; `encode_step` still checks its age.
    fn admit_staged(
        &mut self,
        sequence: u64,
        origin: u64,
        sink: &mut dyn BatchSink,
    ) -> (Option<FlushReason>, usize) {
        let staged_len = self.staging.len();
        if staged_len == 0 {
            return (None, 0);
        }
        if staged_len <= self.capacity - self.buffered_spikes() {
            self.push_staged_batch(sequence, origin);
            return (None, 0);
        }

        match self.policy {
            FlushPolicy::Manual => {
                // Move the call's spikes into the held slot; keeping a second
                // copy in staging would exceed the documented memory bound.
                self.held_spikes.clear();
                std::mem::swap(&mut self.staging, &mut self.held_spikes);
                self.held = Some(HeldBatch {
                    sequence,
                    origin,
                    len: staged_len,
                });
                (None, 0)
            }
            FlushPolicy::OnCapacity | FlushPolicy::OnCapacityOrAge { .. } => {
                let mut delivered = self.drain_queue(sink);
                if staged_len <= self.capacity {
                    self.push_staged_batch(sequence, origin);
                } else {
                    // A single oversized call bypasses the bounded queue.
                    delivered += 1;
                    sink.deliver(SpikeBatch {
                        sequence,
                        origin,
                        spikes: &self.staging,
                    });
                    Self::clear_reusable(&mut self.staging, self.capacity);
                }
                (Some(FlushReason::Capacity), delivered)
            }
        }
    }

    /// Appends the current staging buffer to contiguous storage and records a
    /// queued batch. The caller guarantees it fits within `capacity`.
    fn push_staged_batch(&mut self, sequence: u64, origin: u64) {
        let start = self.storage.len();
        let len = self.staging.len();
        self.storage.extend_from_slice(&self.staging);
        self.queue.push_back(BatchRecord {
            sequence,
            origin,
            start,
            len,
        });
    }

    /// Builds a [`StepReport`] from the current state.
    #[inline]
    fn finish_step(&self, reason: Option<FlushReason>, delivered_batches: usize) -> StepReport {
        StepReport {
            reason,
            delivered_batches,
            queued_spikes: self.buffered_spikes(),
            blocked: self.is_blocked(),
        }
    }

    /// Delivers all buffered output to `sink`: queued batches oldest first,
    /// then the held batch (if any). Clears the blocked state.
    ///
    /// Makes no sink call when there is nothing to deliver. The encoder is
    /// never invoked. Batches already delivered are not redelivered if the sink
    /// panics mid-flush; see the [`BatchSink`] panic contract.
    ///
    /// Returns a [`FlushReport`] with [`FlushReason::Manual`] when anything was
    /// delivered (reason `None` otherwise) and the delivered batch count.
    pub fn flush_into(&mut self, sink: &mut dyn BatchSink) -> FlushReport {
        let mut delivered_batches = self.drain_queue(sink);

        // Deliver the held batch after the queue. Take it before delivering so
        // a panic cannot cause redelivery, and clear the blocked state.
        if let Some(held) = self.held.take() {
            delivered_batches += 1;
            let spikes = &self.held_spikes[..held.len];
            sink.deliver(SpikeBatch {
                sequence: held.sequence,
                origin: held.origin,
                spikes,
            });
            Self::clear_reusable(&mut self.held_spikes, self.capacity);
        }

        let reason = if delivered_batches > 0 {
            Some(FlushReason::Manual)
        } else {
            None
        };
        FlushReport {
            reason,
            delivered_batches,
        }
    }

    /// Resets the wrapped encoder and discards all buffered output.
    ///
    /// Forwards [`Encoder::reset`] to the borrowed encoder, drops every queued
    /// and held spike, and clears the blocked state. The sequence counter and
    /// the [`TimeCursor`] are **not** rewound, so the
    /// caller timeline stays monotonic across a reset. Flush first if queued
    /// output still matters.
    ///
    /// This is an inherent method, not [`Encoder::reset`]: `StreamingEncoder`
    /// does not implement [`Encoder`].
    pub fn reset(&mut self) {
        self.encoder.reset();
        Self::clear_reusable(&mut self.storage, self.capacity);
        self.queue.clear();
        Self::clear_reusable(&mut self.staging, self.capacity);
        Self::clear_reusable(&mut self.held_spikes, self.capacity);
        self.held = None;
    }
}

#[cfg(test)]
#[path = "streaming/tests.rs"]
mod tests;
