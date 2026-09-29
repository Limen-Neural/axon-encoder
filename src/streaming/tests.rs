use super::*;

/// A minimal deterministic encoder: emits one spike per non-zero input
/// channel at tick zero. `TimeModel::INSTANT` (step_ticks 1).
struct CountingEncoder {
    resets: usize,
}

impl CountingEncoder {
    fn new() -> Self {
        Self { resets: 0 }
    }
}

impl Encoder for CountingEncoder {
    fn encode(&mut self, input: &[f32]) -> crate::types::EncodedOutput {
        let mut out = crate::types::EncodedOutput::new();
        for (i, &v) in input.iter().enumerate() {
            if v != 0.0 {
                out.spikes.push(SpikeEvent::at_step_start(i as u16, true));
            }
        }
        out
    }

    fn reset(&mut self) {
        self.resets += 1;
    }
}

/// An encoder with a wider step so age arithmetic is not trivially 1/call.
struct WideStepEncoder;

impl Encoder for WideStepEncoder {
    fn encode(&mut self, input: &[f32]) -> crate::types::EncodedOutput {
        let mut out = crate::types::EncodedOutput::new();
        for (i, &v) in input.iter().enumerate() {
            if v != 0.0 {
                out.spikes.push(SpikeEvent::at_step_start(i as u16, true));
            }
        }
        out
    }

    fn time_model(&self) -> TimeModel {
        TimeModel::window(4)
    }

    fn reset(&mut self) {}
}

/// Collects delivered batches as (sequence, origin, spike count).
fn collector(sink: &mut Vec<(u64, u64, usize)>) -> impl BatchSink + '_ {
    move |batch: SpikeBatch<'_>| {
        sink.push((batch.sequence(), batch.origin(), batch.spikes().len()));
    }
}

#[test]
fn zero_capacity_is_rejected() {
    let mut enc = CountingEncoder::new();
    assert!(matches!(
        StreamingEncoder::try_new(&mut enc, 0, FlushPolicy::Manual),
        Err(EncoderError::CountMustBePositive {
            parameter: "capacity"
        })
    ));
}

#[test]
fn zero_max_age_is_rejected() {
    let mut enc = CountingEncoder::new();
    assert!(matches!(
        StreamingEncoder::try_new(
            &mut enc,
            4,
            FlushPolicy::OnCapacityOrAge { max_age_ticks: 0 },
        ),
        Err(EncoderError::CountMustBePositive {
            parameter: "max_age_ticks"
        })
    ));
}

#[test]
fn accessors_reflect_construction() {
    let mut enc = CountingEncoder::new();
    let s = StreamingEncoder::try_new(&mut enc, 7, FlushPolicy::OnCapacity).unwrap();
    assert_eq!(s.capacity(), 7);
    assert_eq!(s.policy(), FlushPolicy::OnCapacity);
    assert_eq!(s.buffered_spikes(), 0);
    assert_eq!(s.pending_batches(), 0);
    assert!(s.is_empty());
    assert!(!s.is_blocked());
    assert_eq!(s.pending_age_ticks(), 0);
    assert_eq!(s.time_model(), TimeModel::INSTANT);
    assert_eq!(s.cursor().origin(), 0);
}

#[test]
fn fitting_calls_queue_without_delivery() {
    let mut enc = CountingEncoder::new();
    let mut s = StreamingEncoder::try_new(&mut enc, 8, FlushPolicy::OnCapacity).unwrap();
    let mut delivered = Vec::new();

    let report = s
        .encode_step(&[1.0, 1.0], &mut collector(&mut delivered))
        .unwrap();
    assert_eq!(report.reason(), None);
    assert_eq!(report.delivered_batches(), 0);
    assert_eq!(report.queued_spikes(), 2);
    assert!(!report.blocked());
    assert_eq!(s.buffered_spikes(), 2);
    assert_eq!(s.pending_batches(), 1);
    assert!(delivered.is_empty());
}

#[test]
fn empty_output_queues_nothing_and_calls_no_sink() {
    let mut enc = CountingEncoder::new();
    let mut s = StreamingEncoder::try_new(&mut enc, 8, FlushPolicy::OnCapacity).unwrap();
    let mut delivered = Vec::new();

    let report = s
        .encode_step(&[0.0, 0.0], &mut collector(&mut delivered))
        .unwrap();
    assert_eq!(report.delivered_batches(), 0);
    assert_eq!(report.queued_spikes(), 0);
    assert!(s.is_empty());
    assert!(delivered.is_empty());
    // Cursor and sequence still advanced for the accepted call.
    assert_eq!(s.cursor().origin(), 1);
}

#[test]
fn cursor_and_sequence_advance_once_per_accepted_call() {
    let mut enc = CountingEncoder::new();
    let mut s = StreamingEncoder::try_new(&mut enc, 8, FlushPolicy::OnCapacity).unwrap();
    let mut delivered = Vec::new();
    for _ in 0..3 {
        s.encode_step(&[1.0], &mut collector(&mut delivered))
            .unwrap();
    }
    assert_eq!(s.cursor().origin(), 3);
}

#[test]
fn on_capacity_flushes_queue_then_queues_new_batch() {
    let mut enc = CountingEncoder::new();
    let mut s = StreamingEncoder::try_new(&mut enc, 2, FlushPolicy::OnCapacity).unwrap();
    let mut delivered = Vec::new();

    // Fill capacity with two 1-spike batches.
    s.encode_step(&[1.0], &mut collector(&mut delivered))
        .unwrap();
    s.encode_step(&[1.0], &mut collector(&mut delivered))
        .unwrap();
    assert_eq!(s.buffered_spikes(), 2);

    // Third call does not fit -> deliver both, then queue the new one.
    let report = s
        .encode_step(&[1.0], &mut collector(&mut delivered))
        .unwrap();
    assert_eq!(report.reason(), Some(FlushReason::Capacity));
    assert_eq!(report.delivered_batches(), 2);
    assert_eq!(report.queued_spikes(), 1);
    assert_eq!(delivered, vec![(0, 0, 1), (1, 1, 1)]);
    assert_eq!(s.buffered_spikes(), 1);
}

#[test]
fn oversized_call_is_delivered_directly() {
    let mut enc = CountingEncoder::new();
    let mut s = StreamingEncoder::try_new(&mut enc, 2, FlushPolicy::OnCapacity).unwrap();
    let mut delivered = Vec::new();

    // Queue one spike first.
    s.encode_step(&[1.0], &mut collector(&mut delivered))
        .unwrap();
    // A 3-spike call exceeds capacity 2: flush queue, deliver oversized directly.
    let report = s
        .encode_step(&[1.0, 1.0, 1.0], &mut collector(&mut delivered))
        .unwrap();
    assert_eq!(report.reason(), Some(FlushReason::Capacity));
    assert_eq!(report.delivered_batches(), 2);
    assert_eq!(report.queued_spikes(), 0);
    // Second delivered batch is the oversized one, seq 1, 3 spikes.
    assert_eq!(delivered, vec![(0, 0, 1), (1, 1, 3)]);
    assert!(s.is_empty());
    // Queued spikes never exceed capacity.
    assert!(s.buffered_spikes() <= s.capacity());
}

#[test]
fn manual_holds_and_blocks_without_sink_call() {
    let mut enc = CountingEncoder::new();
    let mut s = StreamingEncoder::try_new(&mut enc, 1, FlushPolicy::Manual).unwrap();
    let mut delivered = Vec::new();

    s.encode_step(&[1.0], &mut collector(&mut delivered))
        .unwrap();
    assert_eq!(s.buffered_spikes(), 1);

    // Second call does not fit under Manual: held, blocked, no delivery.
    let report = s
        .encode_step(&[1.0], &mut collector(&mut delivered))
        .unwrap();
    assert!(report.blocked());
    assert_eq!(report.delivered_batches(), 0);
    assert!(s.is_blocked());
    assert!(
        s.staging.is_empty(),
        "held output must be moved, not copied"
    );
    assert_eq!(s.held_spikes.len(), 1);
    assert!(delivered.is_empty());

    // Next call is rejected with backpressure, without advancing.
    let origin_before = s.cursor().origin();
    let err = s
        .encode_step(&[1.0], &mut collector(&mut delivered))
        .unwrap_err();
    assert_eq!(
        err,
        StreamingError::Backpressure {
            buffered_spikes: 1,
            capacity: 1
        }
    );
    assert_eq!(s.cursor().origin(), origin_before);
    assert_eq!(
        err.to_string(),
        "cannot encode while blocked: 1 spike(s) buffered at capacity 1; flush first"
    );
}

#[test]
fn flush_delivers_queue_then_held_and_unblocks() {
    let mut enc = CountingEncoder::new();
    let mut s = StreamingEncoder::try_new(&mut enc, 1, FlushPolicy::Manual).unwrap();
    let mut delivered = Vec::new();

    s.encode_step(&[1.0], &mut collector(&mut delivered))
        .unwrap(); // seq 0 queued
    s.encode_step(&[1.0], &mut collector(&mut delivered))
        .unwrap(); // seq 1 held

    let report = s.flush_into(&mut collector(&mut delivered));
    assert_eq!(report.reason(), Some(FlushReason::Manual));
    assert_eq!(report.delivered_batches(), 2);
    // Queue (seq 0) first, then held (seq 1).
    assert_eq!(delivered, vec![(0, 0, 1), (1, 1, 1)]);
    assert!(!s.is_blocked());
    assert!(s.is_empty());
}

#[test]
fn flush_on_empty_makes_no_sink_call() {
    let mut enc = CountingEncoder::new();
    let mut s = StreamingEncoder::try_new(&mut enc, 4, FlushPolicy::Manual).unwrap();
    let mut delivered = Vec::new();
    let report = s.flush_into(&mut collector(&mut delivered));
    assert_eq!(report.reason(), None);
    assert_eq!(report.delivered_batches(), 0);
    assert!(delivered.is_empty());
}

#[test]
fn age_trigger_delivers_when_oldest_batch_ages_out() {
    let mut enc = WideStepEncoder; // step_ticks = 4
    let mut s = StreamingEncoder::try_new(
        &mut enc,
        16,
        FlushPolicy::OnCapacityOrAge { max_age_ticks: 4 },
    )
    .unwrap();
    let mut delivered = Vec::new();

    // First call queues (origin 0). After admission cursor origin is 4,
    // oldest age = 4 >= 4 -> flush by age.
    let report = s
        .encode_step(&[1.0], &mut collector(&mut delivered))
        .unwrap();
    assert_eq!(report.reason(), Some(FlushReason::Age));
    assert_eq!(report.delivered_batches(), 1);
    assert_eq!(delivered, vec![(0, 0, 1)]);
    assert!(s.is_empty());
    s.reset();
}

#[test]
fn pending_age_ticks_tracks_oldest_origin() {
    let mut enc = CountingEncoder::new(); // step_ticks = 1
    let mut s = StreamingEncoder::try_new(&mut enc, 8, FlushPolicy::OnCapacity).unwrap();
    let mut delivered = Vec::new();
    // Queue at origin 0; cursor advances to 1 then 2.
    s.encode_step(&[1.0], &mut collector(&mut delivered))
        .unwrap();
    s.encode_step(&[1.0], &mut collector(&mut delivered))
        .unwrap();
    // Oldest origin 0, cursor origin 2 -> age 2.
    assert_eq!(s.pending_age_ticks(), 2);
}

#[test]
fn reset_discards_output_but_keeps_timeline() {
    let mut enc = CountingEncoder::new();
    let mut s = StreamingEncoder::try_new(&mut enc, 1, FlushPolicy::Manual).unwrap();
    let mut delivered = Vec::new();
    s.encode_step(&[1.0], &mut collector(&mut delivered))
        .unwrap(); // queued
    s.encode_step(&[1.0], &mut collector(&mut delivered))
        .unwrap(); // held/blocked
    assert!(s.is_blocked());
    let origin_before = s.cursor().origin();

    s.reset();
    assert!(s.is_empty());
    assert!(!s.is_blocked());
    assert_eq!(s.buffered_spikes(), 0);
    // Timeline is monotonic: cursor not rewound.
    assert_eq!(s.cursor().origin(), origin_before);
    assert!(delivered.is_empty());
}

#[test]
fn panic_mid_flush_does_not_redeliver() {
    let mut enc = CountingEncoder::new();
    let mut s = StreamingEncoder::try_new(&mut enc, 4, FlushPolicy::OnCapacity).unwrap();
    let mut ok = Vec::new();
    s.encode_step(&[1.0], &mut collector(&mut ok)).unwrap(); // seq 0
    s.encode_step(&[1.0], &mut collector(&mut ok)).unwrap(); // seq 1

    let mut seen: Vec<u64> = Vec::new();
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        s.flush_into(&mut |batch: SpikeBatch<'_>| {
            seen.push(batch.sequence());
            panic!("sink refused batch");
        });
    }));
    assert!(result.is_err());
    assert_eq!(seen, vec![0]); // panicked on the first, second not reached

    // The first batch was already popped; a later flush must not redeliver it.
    let mut after: Vec<u64> = Vec::new();
    s.flush_into(&mut |batch: SpikeBatch<'_>| after.push(batch.sequence()));
    assert!(!after.contains(&0));
}
