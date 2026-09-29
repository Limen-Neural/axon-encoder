//! Bounded, caller-driven delivery of streaming encoder output.
//!
//! [`StreamingEncoder`] borrows an [`Encoder`], including a
//! `&mut dyn Encoder`; the caller owns both the encoder and the [`BatchSink`].
//! Each non-empty call is delivered as one [`SpikeBatch`], preserving spike
//! order and call-relative offsets. Add the batch origin to an offset to get
//! absolute ticks. Only spikes pass through this API, as with [`SpikeSink`](crate::SpikeSink).
//!
//! The queue holds at most `capacity` spikes. [`FlushPolicy::Manual`] can also
//! retain one complete call that did not fit, then rejects further input with
//! [`StreamingError::Backpressure`](crate::StreamingError) *before* invoking the
//! encoder. Thus retained spike count is bounded by capacity plus one call's
//! output; a single call's output size is controlled by the borrowed encoder.
//! Other policies flush before admitting output that would exceed capacity and
//! deliver an oversized call directly. The sink must keep up with delivery;
//! this wrapper owns no transport or background worker.
//!
//! [`FlushPolicy::OnCapacityOrAge`] measures age in encoder ticks, advanced by
//! accepted calls. No wall clock is read. Callers with a real-time deadline can
//! inspect [`StreamingEncoder::pending_age_ticks`] and call
//! [`StreamingEncoder::flush_into`] on their own schedule. `reset` discards all
//! pending output; flush first when that output matters. A sink panic removes
//! the in-flight batch before delivery, so it is never replayed; remaining
//! batches stay valid, but call [`StreamingEncoder::reset`] before reuse, as
//! required by the sink panic contract.
//!
//! The wrapper does not implement `Encoder`: its deferred output belongs to
//! earlier calls and requires batch metadata that `Encoder` cannot return.

use std::collections::VecDeque;

use crate::Encoder;
use crate::error::{EncoderError, StreamingError};
use crate::time::{TimeCursor, TimeModel};
use crate::types::SpikeEvent;

/// When queued batches are delivered automatically during `encode_step`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FlushPolicy {
    /// Only [`StreamingEncoder::flush_into`] delivers batches.
    Manual,
    /// Flush queued batches when a new call does not fit. An oversized call is
    /// delivered directly after the queue is drained.
    OnCapacity,
    /// Also flush when the oldest pending call has aged by `max_age_ticks`.
    /// Age advances by `TimeModel::step_ticks` on each accepted call.
    OnCapacityOrAge { max_age_ticks: u64 },
}

/// Why a non-empty delivery occurred.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FlushReason {
    /// Requested by the caller.
    Manual,
    /// New output would exceed capacity.
    Capacity,
    /// Oldest output reached the configured age.
    Age,
}

/// A borrowed view of one encoder call's non-empty spike output.
#[derive(Clone, Copy, Debug)]
pub struct SpikeBatch<'a> {
    sequence: u64,
    origin: u64,
    spikes: &'a [SpikeEvent],
}

impl<'a> SpikeBatch<'a> {
    /// Zero-based accepted call number, including calls with no spikes.
    pub fn sequence(self) -> u64 {
        self.sequence
    }
    /// Absolute tick at the start of the emitting call.
    pub fn origin(self) -> u64 {
        self.origin
    }
    /// Spikes in original order, with unchanged call-relative offsets.
    pub fn spikes(self) -> &'a [SpikeEvent] {
        self.spikes
    }
}

/// A destination for complete batches, called in accepted-call order.
///
/// A sink may append a batch's spikes to an existing [`SpikeSink`](crate::SpikeSink).
/// It is never called for an empty output or empty flush. If it panics, its
/// accepted prefix remains and the in-flight batch is discarded.
pub trait BatchSink {
    /// Accept one non-empty batch. Borrowed data is valid for this call only.
    fn push_batch(&mut self, batch: SpikeBatch<'_>);
}

impl<F> BatchSink for F
where
    F: for<'a> FnMut(SpikeBatch<'a>),
{
    fn push_batch(&mut self, batch: SpikeBatch<'_>) {
        self(batch);
    }
}

/// Delivery and queue state after one accepted encoder call.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StepReport {
    /// Reason for automatic delivery, if any.
    pub flush_reason: Option<FlushReason>,
    /// Number of batches delivered during this call.
    pub delivered_batches: usize,
    /// Number of spikes retained, including a held call.
    pub buffered_spikes: usize,
    /// Whether the next encode call will be rejected until flush or reset.
    pub blocked: bool,
}

/// Delivery outcome of an explicit or automatic flush.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FlushReport {
    /// Why delivery occurred, or `None` if there was nothing to deliver.
    pub reason: Option<FlushReason>,
    /// Number of delivered batches.
    pub delivered_batches: usize,
    /// Number of delivered spikes.
    pub delivered_spikes: usize,
}

struct OwnedBatch {
    sequence: u64,
    origin: u64,
    spikes: Vec<SpikeEvent>,
}

/// Borrows an encoder and buffers its non-empty streaming calls.
pub struct StreamingEncoder<'a, E: Encoder + ?Sized> {
    encoder: &'a mut E,
    capacity: usize,
    policy: FlushPolicy,
    queue: VecDeque<OwnedBatch>,
    held: Option<OwnedBatch>,
    staging: Vec<SpikeEvent>,
    queued_spikes: usize,
    cursor: TimeCursor,
    next_sequence: u64,
}

impl<'a, E: Encoder + ?Sized> StreamingEncoder<'a, E> {
    /// Construct a wrapper with a positive spike capacity and, when used, age.
    pub fn new(
        encoder: &'a mut E,
        capacity: usize,
        policy: FlushPolicy,
    ) -> Result<Self, EncoderError> {
        if capacity == 0 {
            return Err(EncoderError::CountMustBePositive {
                parameter: "capacity",
            });
        }
        if matches!(policy, FlushPolicy::OnCapacityOrAge { max_age_ticks: 0 }) {
            return Err(EncoderError::CountMustBePositive {
                parameter: "max_age_ticks",
            });
        }
        let cursor = TimeCursor::new(encoder.time_model());
        Ok(Self {
            encoder,
            capacity,
            policy,
            queue: VecDeque::new(),
            held: None,
            staging: Vec::new(),
            queued_spikes: 0,
            cursor,
            next_sequence: 0,
        })
    }

    /// Configured maximum queued spikes, excluding a held call.
    pub fn capacity(&self) -> usize {
        self.capacity
    }
    /// Configured automatic delivery policy.
    pub fn policy(&self) -> FlushPolicy {
        self.policy
    }
    /// Retained spikes, including a held call after manual backpressure.
    pub fn buffered_spikes(&self) -> usize {
        self.queued_spikes + self.held.as_ref().map_or(0, |b| b.spikes.len())
    }
    /// Retained non-empty calls, including a held call.
    pub fn pending_batches(&self) -> usize {
        self.queue.len() + usize::from(self.held.is_some())
    }
    /// Whether no spikes await delivery.
    pub fn is_empty(&self) -> bool {
        self.pending_batches() == 0
    }
    /// Whether manual backpressure prevents another encoder call.
    pub fn is_blocked(&self) -> bool {
        self.held.is_some()
    }
    /// Age of the oldest pending call in encoder ticks.
    pub fn pending_age_ticks(&self) -> Option<u64> {
        self.queue
            .front()
            .or(self.held.as_ref())
            .map(|batch| self.cursor.origin().saturating_sub(batch.origin))
    }
    /// Timeline after all accepted calls.
    pub fn cursor(&self) -> TimeCursor {
        self.cursor
    }
    /// Time model reported by the borrowed encoder.
    pub fn time_model(&self) -> TimeModel {
        self.encoder.time_model()
    }

    /// Encode one call and deliver according to the configured policy.
    ///
    /// A blocked call returns an error without touching the encoder or timeline.
    /// Empty input is forwarded to the encoder and still advances the timeline.
    pub fn encode_step(
        &mut self,
        input: &[f32],
        sink: &mut dyn BatchSink,
    ) -> Result<StepReport, StreamingError> {
        if self.is_blocked() {
            return Err(StreamingError::Backpressure {
                buffered_spikes: self.buffered_spikes(),
                capacity: self.capacity,
            });
        }
        self.staging.clear();
        self.encoder.encode_step_into(input, &mut self.staging);
        let origin = self.cursor.origin();
        let sequence = self.next_sequence;
        self.cursor.advance();
        self.next_sequence = self.next_sequence.saturating_add(1);

        let mut delivered_batches = 0;
        let mut flush_reason = None;
        if !self.staging.is_empty() {
            let batch = OwnedBatch {
                sequence,
                origin,
                spikes: std::mem::take(&mut self.staging),
            };
            if batch.spikes.len() > self.capacity - self.queued_spikes {
                match self.policy {
                    FlushPolicy::Manual => self.held = Some(batch),
                    FlushPolicy::OnCapacity | FlushPolicy::OnCapacityOrAge { .. } => {
                        let report = self.deliver_queued(sink, FlushReason::Capacity);
                        delivered_batches += report.delivered_batches;
                        if batch.spikes.len() > self.capacity {
                            Self::deliver_batch(sink, batch);
                            delivered_batches += 1;
                        } else {
                            self.push(batch);
                        }
                        flush_reason = Some(FlushReason::Capacity);
                    }
                }
            } else {
                self.push(batch);
            }
        }
        if let FlushPolicy::OnCapacityOrAge { max_age_ticks } = self.policy
            && self
                .pending_age_ticks()
                .is_some_and(|age| age >= max_age_ticks)
        {
            let report = self.deliver_queued(sink, FlushReason::Age);
            delivered_batches += report.delivered_batches;
            flush_reason = Some(FlushReason::Age);
        }
        Ok(StepReport {
            flush_reason,
            delivered_batches,
            buffered_spikes: self.buffered_spikes(),
            blocked: self.is_blocked(),
        })
    }

    /// Deliver every pending batch in order. Does not invoke the encoder.
    pub fn flush_into(&mut self, sink: &mut dyn BatchSink) -> FlushReport {
        let mut report = self.deliver_queued(sink, FlushReason::Manual);
        if let Some(batch) = self.held.take() {
            report.delivered_batches += 1;
            report.delivered_spikes += batch.spikes.len();
            report.reason = Some(FlushReason::Manual);
            Self::deliver_batch(sink, batch);
        }
        report
    }

    /// Reset encoder state and discard pending output without rewinding metadata.
    pub fn reset(&mut self) {
        self.encoder.reset();
        self.queue.clear();
        self.held = None;
        self.staging.clear();
        self.queued_spikes = 0;
    }

    fn push(&mut self, batch: OwnedBatch) {
        self.queued_spikes += batch.spikes.len();
        self.queue.push_back(batch);
    }

    fn deliver_queued(&mut self, sink: &mut dyn BatchSink, reason: FlushReason) -> FlushReport {
        let mut report = FlushReport {
            reason: None,
            delivered_batches: 0,
            delivered_spikes: 0,
        };
        while let Some(batch) = self.queue.pop_front() {
            self.queued_spikes -= batch.spikes.len();
            report.delivered_batches += 1;
            report.delivered_spikes += batch.spikes.len();
            report.reason = Some(reason);
            Self::deliver_batch(sink, batch);
        }
        report
    }

    fn deliver_batch(sink: &mut dyn BatchSink, batch: OwnedBatch) {
        sink.push_batch(SpikeBatch {
            sequence: batch.sequence,
            origin: batch.origin,
            spikes: &batch.spikes,
        });
    }
}
