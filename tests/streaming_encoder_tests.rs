//! Behavioral tests for the buffered [`StreamingEncoder`] wrapper.
//!
//! The wrapper is only correct if the batches it delivers are exactly the
//! spikes the borrowed encoder would have produced through its
//! [`encode_step_into`](Encoder::encode_step_into) path — same spikes, same
//! channel order, same call-relative [`TickOffset`]s, never rebased — merely
//! deferred and grouped by the [`FlushPolicy`]. These tests pin that down
//! against a *separate* reference encoder driven directly, mirroring the
//! `assert_paths_agree` style in `tests/encode_into_tests.rs`, and cover the
//! capacity/age/backpressure/reset/panic behavior of the buffer itself.
//!
//! Every reference encoder here is deterministic on the streaming path:
//! [`RateEncoder::encode_step`] is accumulator-driven and [`PhaseEncoder`] is
//! combinatorial, so equality (not just invariants) is available.

use std::panic::{AssertUnwindSafe, catch_unwind};

use axon_encoder::prelude::*;

/// One delivered batch, copied out of the borrowed [`SpikeBatch`] view so it can
/// outlive the `deliver` call and be compared after the run.
#[derive(Clone, Debug, PartialEq, Eq)]
struct CapturedBatch {
    sequence: u64,
    origin: u64,
    spikes: Vec<SpikeEvent>,
}

/// A closure [`BatchSink`] that copies every delivered batch into `out`.
fn capture(out: &mut Vec<CapturedBatch>) -> impl BatchSink + '_ {
    move |batch: SpikeBatch<'_>| {
        out.push(CapturedBatch {
            sequence: batch.sequence(),
            origin: batch.origin(),
            spikes: batch.spikes().to_vec(),
        });
    }
}

/// The reference spikes a freshly built `encoder` emits for `input` via the
/// direct [`encode_step_into`](Encoder::encode_step_into) path.
fn reference_step<E: Encoder>(encoder: &mut E, input: &[f32]) -> Vec<SpikeEvent> {
    let mut buffer = Vec::new();
    encoder.encode_step_into(input, &mut buffer);
    buffer
}

/// A deterministic rate encoder emitting ~1 spike per step at input `1.0`
/// (10 Hz * 0.1 s = 1.0 accumulator increment per step).
fn rate_one_per_step() -> RateEncoder {
    RateEncoder::try_new(0.0, 10.0, (0.0, 1.0), 0.1).expect("valid RateEncoder")
}

// --- Borrows: concrete and trait-object -------------------------------------

#[test]
fn concrete_and_trait_object_borrows_behave_identically() {
    let inputs: &[&[f32]] = &[&[1.0], &[1.0], &[1.0], &[1.0]];

    // Concrete borrow.
    let mut concrete_encoder = rate_one_per_step();
    let mut concrete_out = Vec::new();
    {
        let mut streaming =
            StreamingEncoder::try_new(&mut concrete_encoder, 64, FlushPolicy::Manual)
                .expect("valid wrapper");
        for input in inputs {
            streaming
                .encode_step(input, &mut capture(&mut concrete_out))
                .expect("not blocked");
        }
        streaming.flush_into(&mut capture(&mut concrete_out));
    }

    // Trait-object borrow of an identically configured encoder.
    let mut dyn_encoder = rate_one_per_step();
    let mut dyn_out = Vec::new();
    {
        let e: &mut dyn Encoder = &mut dyn_encoder;
        let mut streaming =
            StreamingEncoder::try_new(e, 64, FlushPolicy::Manual).expect("valid wrapper");
        for input in inputs {
            streaming
                .encode_step(input, &mut capture(&mut dyn_out))
                .expect("not blocked");
        }
        streaming.flush_into(&mut capture(&mut dyn_out));
    }

    assert_eq!(
        concrete_out, dyn_out,
        "concrete and trait-object borrows must deliver identical batches"
    );
    assert!(
        !concrete_out.is_empty(),
        "the run must actually deliver spikes for the comparison to mean something"
    );
}

// --- Manual flush: spikes/offsets/order match the direct reference path ------

#[test]
fn manual_flush_batches_match_direct_encode_step_into() {
    let inputs: &[&[f32]] = &[&[1.0], &[1.0], &[1.0], &[1.0], &[1.0], &[1.0]];

    let mut wrapped_encoder = rate_one_per_step();
    let mut delivered = Vec::new();
    {
        let mut streaming =
            StreamingEncoder::try_new(&mut wrapped_encoder, 1024, FlushPolicy::Manual)
                .expect("valid wrapper");
        for input in inputs {
            streaming
                .encode_step(input, &mut capture(&mut delivered))
                .expect("capacity is generous; never blocked");
        }
        // Manual policy delivers nothing until an explicit flush.
        assert!(
            delivered.is_empty(),
            "Manual policy must not deliver during encode_step"
        );
        streaming.flush_into(&mut capture(&mut delivered));
    }

    // Reference encoder, built identically, driven on the same inputs.
    let mut reference_encoder = rate_one_per_step();
    for (i, input) in inputs.iter().enumerate() {
        let expected = reference_step(&mut reference_encoder, input);
        // Empty reference output produces no batch, so batches skip empty calls.
        // With input 1.0 every call is non-empty here.
        let batch = &delivered[i];
        assert_eq!(
            batch.spikes, expected,
            "batch {i}: spikes/channel-order/offsets must match the direct path exactly"
        );
        assert_eq!(
            batch.sequence, i as u64,
            "batch {i}: sequence is the call index"
        );
        assert_eq!(
            batch.origin, i as u64,
            "batch {i}: origin advances by step_ticks (1) per call"
        );
    }
    assert_eq!(delivered.len(), inputs.len());
}

#[test]
fn batch_offsets_are_call_relative_never_rebased() {
    // PhaseEncoder places spikes across the cycle by value, so offsets are
    // non-zero and would visibly change if the wrapper rebased them onto the
    // absolute timeline. Drive several calls, then confirm every delivered
    // offset still equals the direct per-call reference offset.
    let inputs: &[&[f32]] = &[&[1.0, 0.0, 0.5], &[0.25, 0.75, 1.0], &[0.5, 0.5, 0.5]];

    let mut wrapped = PhaseEncoder::try_new(8, (0.0, 1.0)).expect("valid PhaseEncoder");
    let mut delivered = Vec::new();
    {
        let mut streaming =
            StreamingEncoder::try_new(&mut wrapped, 256, FlushPolicy::Manual).expect("valid");
        for input in inputs {
            streaming
                .encode_step(input, &mut capture(&mut delivered))
                .expect("not blocked");
        }
        streaming.flush_into(&mut capture(&mut delivered));
    }

    let mut reference = PhaseEncoder::try_new(8, (0.0, 1.0)).expect("valid PhaseEncoder");
    for (i, input) in inputs.iter().enumerate() {
        let expected = reference_step(&mut reference, input);
        assert_eq!(
            delivered[i].spikes, expected,
            "batch {i}: offsets must stay the unchanged call-relative TickOffsets"
        );
        assert_eq!(delivered[i].origin, i as u64);
    }
}

// --- Sequence + origin monotonicity -----------------------------------------

#[test]
fn sequence_increases_by_one_and_origin_by_step_ticks_per_call() {
    let inputs: &[&[f32]] = &[&[1.0], &[1.0], &[1.0], &[1.0], &[1.0]];

    let mut encoder = rate_one_per_step();
    let mut delivered = Vec::new();
    {
        let mut streaming =
            StreamingEncoder::try_new(&mut encoder, 1024, FlushPolicy::Manual).expect("valid");
        for input in inputs {
            streaming
                .encode_step(input, &mut capture(&mut delivered))
                .expect("not blocked");
        }
        // RateEncoder step_ticks == 1, so origin equals the number of calls.
        assert_eq!(streaming.time_model().step_ticks(), 1);
        assert_eq!(streaming.cursor().origin(), inputs.len() as u64);
        streaming.flush_into(&mut capture(&mut delivered));
    }

    for (i, batch) in delivered.iter().enumerate() {
        assert_eq!(batch.sequence, i as u64, "sequence must be +1 per call");
        assert_eq!(
            batch.origin, i as u64,
            "origin must be +step_ticks per call"
        );
    }
}

// --- Empty flush -------------------------------------------------------------

#[test]
fn flush_on_never_fed_wrapper_makes_no_sink_call() {
    let mut encoder = rate_one_per_step();
    let mut streaming =
        StreamingEncoder::try_new(&mut encoder, 8, FlushPolicy::Manual).expect("valid");

    let mut calls = 0usize;
    let report = streaming.flush_into(&mut |_batch: SpikeBatch<'_>| calls += 1);

    assert_eq!(calls, 0, "the sink closure must never be invoked");
    assert_eq!(report.delivered_batches(), 0);
    assert_eq!(report.reason(), None);
}

// --- OnCapacity --------------------------------------------------------------

#[test]
fn on_capacity_never_exceeds_bound_and_flushes_with_capacity_reason() {
    // 1 spike/step, capacity 3: the 4th call cannot fit, triggering a Capacity
    // flush of the 3 queued batches before the 4th is queued.
    let mut encoder = rate_one_per_step();
    let mut delivered = Vec::new();
    let mut capacity_flush_seen = false;

    {
        let mut streaming =
            StreamingEncoder::try_new(&mut encoder, 3, FlushPolicy::OnCapacity).expect("valid");
        for _ in 0..9 {
            let report = streaming
                .encode_step(&[1.0], &mut capture(&mut delivered))
                .expect("OnCapacity never blocks");
            assert!(
                streaming.buffered_spikes() <= streaming.capacity(),
                "queued spikes must never exceed capacity"
            );
            if report.reason() == Some(FlushReason::Capacity) {
                capacity_flush_seen = true;
                assert!(report.delivered_batches() >= 1);
            }
        }
        streaming.flush_into(&mut capture(&mut delivered));
    }

    assert!(
        capacity_flush_seen,
        "a Capacity flush must occur once a call no longer fits"
    );

    // Every delivered batch still matches the direct reference, in order.
    let mut reference = rate_one_per_step();
    for (i, batch) in delivered.iter().enumerate() {
        let expected = reference_step(&mut reference, &[1.0]);
        assert_eq!(batch.spikes, expected, "batch {i} diverged from reference");
        assert_eq!(batch.sequence, i as u64);
    }
    assert_eq!(delivered.len(), 9, "all nine calls must be delivered once");
}

#[test]
fn oversized_call_is_delivered_directly_as_one_batch() {
    // A single call produces more spikes than the whole capacity, so it cannot
    // be queued: after flushing whatever was queued, it is delivered directly.
    // 30 Hz * 0.1 s = 3 spikes/step at input 1.0; capacity 2 is smaller.
    let mut encoder = RateEncoder::try_new(0.0, 30.0, (0.0, 1.0), 0.1).expect("valid");
    let mut delivered = Vec::new();

    {
        let mut streaming =
            StreamingEncoder::try_new(&mut encoder, 2, FlushPolicy::OnCapacity).expect("valid");
        // Confirm the config really produces an oversized call (3 > 2).
        let mut probe = RateEncoder::try_new(0.0, 30.0, (0.0, 1.0), 0.1).expect("valid");
        assert_eq!(
            reference_step(&mut probe, &[1.0]).len(),
            3,
            "config must emit 3 spikes/step for the oversized-delivery test"
        );

        let report = streaming
            .encode_step(&[1.0], &mut capture(&mut delivered))
            .expect("not blocked");
        // First call: 3 spikes > capacity 2, queue empty, delivered directly.
        assert_eq!(report.delivered_batches(), 1);
        assert_eq!(report.reason(), Some(FlushReason::Capacity));
        assert_eq!(report.queued_spikes(), 0);
        assert!(streaming.is_empty());
    }

    assert_eq!(delivered.len(), 1);
    assert_eq!(
        delivered[0].spikes.len(),
        3,
        "the oversized call is delivered whole as a single batch"
    );
    assert_eq!(delivered[0].sequence, 0);
    assert!(
        delivered[0].spikes.len() > 2,
        "oversized batch legitimately exceeds capacity"
    );
}

// --- OnCapacityOrAge ---------------------------------------------------------

#[test]
fn on_capacity_or_age_flushes_by_age_after_configured_ticks() {
    // step_ticks == 1 for RateEncoder, so max_age_ticks == 3 means the oldest
    // queued batch ages out three calls after it was queued.
    let mut encoder = rate_one_per_step();
    let mut delivered = Vec::new();
    let mut age_flush_seen = false;

    {
        let mut streaming = StreamingEncoder::try_new(
            &mut encoder,
            64,
            FlushPolicy::OnCapacityOrAge { max_age_ticks: 3 },
        )
        .expect("valid");

        for _ in 0..3 {
            let report = streaming
                .encode_step(&[1.0], &mut capture(&mut delivered))
                .expect("not blocked");
            if report.reason() == Some(FlushReason::Age) {
                age_flush_seen = true;
                assert!(report.delivered_batches() >= 1);
                // Age flush drains the queue.
                assert!(streaming.is_empty());
            }
        }
        streaming.flush_into(&mut capture(&mut delivered));
    }

    assert!(
        age_flush_seen,
        "a batch must age out and flush with FlushReason::Age"
    );
    assert_eq!(delivered.len(), 3, "all three calls delivered exactly once");
}

// --- Manual: held output, backpressure, flush order -------------------------

#[test]
fn manual_holds_blocks_and_rejects_without_advancing_the_encoder() {
    // Capacity 2 with 1 spike/step: two calls fill the queue, the third does
    // not fit and is held under Manual, blocking the wrapper.
    let mut encoder = rate_one_per_step();
    let mut delivered = Vec::new();

    // Reference sequence: what the encoder SHOULD emit across the accepted
    // calls, if the rejected call never reaches it.
    let mut reference = rate_one_per_step();

    {
        let mut streaming =
            StreamingEncoder::try_new(&mut encoder, 2, FlushPolicy::Manual).expect("valid");

        streaming
            .encode_step(&[1.0], &mut capture(&mut delivered))
            .expect("call 0 queues");
        streaming
            .encode_step(&[1.0], &mut capture(&mut delivered))
            .expect("call 1 queues");
        assert_eq!(streaming.buffered_spikes(), 2);
        assert!(!streaming.is_blocked());

        // Call 2 does not fit under Manual: held, blocked, nothing delivered.
        let report = streaming
            .encode_step(&[1.0], &mut capture(&mut delivered))
            .expect("held, not an error");
        assert!(report.blocked());
        assert_eq!(report.delivered_batches(), 0);
        assert!(streaming.is_blocked());
        assert!(delivered.is_empty(), "Manual never delivers during a step");

        let origin_before = streaming.cursor().origin();
        let sequence_before = origin_before; // step_ticks == 1

        // Call 3 is rejected with Backpressure BEFORE touching the encoder.
        let err = streaming
            .encode_step(&[1.0], &mut capture(&mut delivered))
            .expect_err("blocked wrapper must reject");
        assert_eq!(
            err,
            StreamingError::Backpressure {
                buffered_spikes: 2,
                capacity: 2,
            }
        );
        // Neither cursor nor sequence advanced on the rejected call.
        assert_eq!(streaming.cursor().origin(), origin_before);

        // Flush: queued batches (seq 0,1) BEFORE the held batch (seq 2).
        let flush = streaming.flush_into(&mut capture(&mut delivered));
        assert_eq!(flush.delivered_batches(), 3);
        assert_eq!(flush.reason(), Some(FlushReason::Manual));
        assert!(!streaming.is_blocked(), "flush clears the blocked state");

        // Sanity: the rejected call did not advance the timeline, so after the
        // three accepted calls the origin is exactly 3.
        assert_eq!(streaming.cursor().origin(), 3);
        let _ = sequence_before;
    }

    // Ordering: queued-before-held, sequence strictly increasing.
    assert_eq!(delivered.len(), 3);
    assert_eq!(delivered[0].sequence, 0);
    assert_eq!(delivered[1].sequence, 1);
    assert_eq!(delivered[2].sequence, 2);

    // The rejected input was never consumed by the wrapper's encoder: the three
    // delivered batches match a reference driven with exactly three calls.
    for batch in &delivered {
        let expected = reference_step(&mut reference, &[1.0]);
        assert_eq!(
            batch.spikes, expected,
            "delivered spikes must match a reference that saw only the accepted calls"
        );
    }
    // A fourth reference call is what the wrapper's encoder would emit NEXT;
    // it must still line up, proving the rejected call did not desync state.
    let mut post_flush_delivered = Vec::new();
    {
        let mut streaming =
            StreamingEncoder::try_new(&mut encoder, 8, FlushPolicy::Manual).expect("valid");
        streaming
            .encode_step(&[1.0], &mut capture(&mut post_flush_delivered))
            .expect("unblocked after flush");
        streaming.flush_into(&mut capture(&mut post_flush_delivered));
    }
    let expected_next = reference_step(&mut reference, &[1.0]);
    assert_eq!(
        post_flush_delivered[0].spikes, expected_next,
        "the encoder's state advanced by exactly the accepted calls"
    );
}

// --- Constructor validation --------------------------------------------------

#[test]
fn zero_capacity_is_rejected() {
    let mut encoder = rate_one_per_step();
    // StreamingEncoder is intentionally not Debug, so match rather than unwrap_err.
    match StreamingEncoder::try_new(&mut encoder, 0, FlushPolicy::OnCapacity) {
        Err(err) => assert_eq!(
            err,
            EncoderError::CountMustBePositive {
                parameter: "capacity"
            }
        ),
        Ok(_) => panic!("zero capacity must be rejected"),
    }
}

#[test]
fn zero_max_age_ticks_is_rejected() {
    let mut encoder = rate_one_per_step();
    match StreamingEncoder::try_new(
        &mut encoder,
        8,
        FlushPolicy::OnCapacityOrAge { max_age_ticks: 0 },
    ) {
        Err(err) => assert_eq!(
            err,
            EncoderError::CountMustBePositive {
                parameter: "max_age_ticks"
            }
        ),
        Ok(_) => panic!("zero max_age_ticks must be rejected"),
    }
}

// --- Reset + monotonic metadata ---------------------------------------------

#[test]
fn reset_clears_buffer_keeps_timeline_and_matches_fresh_encoder() {
    let mut encoder = rate_one_per_step();
    let mut delivered = Vec::new();

    let mut streaming =
        StreamingEncoder::try_new(&mut encoder, 2, FlushPolicy::Manual).expect("valid");

    // Feed until blocked so reset has queued AND held output to discard.
    streaming
        .encode_step(&[1.0], &mut capture(&mut delivered))
        .expect("queued");
    streaming
        .encode_step(&[1.0], &mut capture(&mut delivered))
        .expect("queued");
    streaming
        .encode_step(&[1.0], &mut capture(&mut delivered))
        .expect("held/blocked");
    assert!(streaming.is_blocked());
    let origin_before = streaming.cursor().origin();
    assert_eq!(origin_before, 3);

    streaming.reset();
    assert!(
        streaming.is_empty(),
        "reset discards queued and held output"
    );
    assert!(!streaming.is_blocked(), "reset clears the blocked state");
    assert_eq!(streaming.buffered_spikes(), 0);
    // Timeline is monotonic: cursor is NOT rewound.
    assert_eq!(
        streaming.cursor().origin(),
        origin_before,
        "reset must not rewind the cursor"
    );
    assert!(delivered.is_empty(), "reset delivers nothing");

    // After reset the wrapped encoder behaves like a FRESH one: reset forwarded
    // to the encoder wipes its accumulator, so its next output matches a
    // brand-new reference encoder's first call.
    let mut post_reset = Vec::new();
    let seq_before_next = streaming.cursor().origin();
    streaming
        .encode_step(&[1.0], &mut capture(&mut post_reset))
        .expect("not blocked after reset");
    streaming.flush_into(&mut capture(&mut post_reset));

    let mut fresh_reference = rate_one_per_step();
    let expected = reference_step(&mut fresh_reference, &[1.0]);
    assert_eq!(
        post_reset[0].spikes, expected,
        "after reset the encoder starts fresh"
    );
    // Sequence and cursor keep climbing (monotonic), never rewound by reset.
    assert!(
        post_reset[0].origin >= seq_before_next,
        "origin stays monotonic across reset"
    );
    assert_eq!(post_reset[0].sequence, 3, "sequence continues, not rewound");
    assert_eq!(post_reset[0].origin, 3);
}

// --- Empty input reaches the encoder and advances the timeline ---------------

#[test]
fn empty_input_advances_phase_encoder_through_the_wrapper() {
    // PhaseEncoder advances current_phase() once per call, even on empty input.
    // Routing empty input through the wrapper must still reach the encoder.
    let mut encoder = PhaseEncoder::try_new(8, (0.0, 1.0)).expect("valid");
    let mut delivered = Vec::new();

    {
        let mut streaming =
            StreamingEncoder::try_new(&mut encoder, 8, FlushPolicy::OnCapacity).expect("valid");
        assert_eq!(streaming.cursor().origin(), 0);

        // Three empty calls: each reaches the encoder, queues nothing, calls no
        // sink, yet advances cursor and sequence.
        for expected_origin in 1..=3u64 {
            let report = streaming
                .encode_step(&[], &mut capture(&mut delivered))
                .expect("not blocked");
            assert_eq!(report.delivered_batches(), 0, "empty output queues nothing");
            assert_eq!(report.queued_spikes(), 0);
            assert!(streaming.is_empty());
            assert_eq!(
                streaming.cursor().origin(),
                expected_origin,
                "cursor advances even on empty input"
            );
        }
    }

    assert!(delivered.is_empty(), "empty calls make no sink call");
    // The encoder's background phase advanced once per empty call.
    assert_eq!(
        encoder.current_phase(),
        3,
        "empty input must reach the encoder and advance its phase"
    );
}

#[test]
fn empty_input_preserves_rate_encoder_backlog_through_the_wrapper() {
    // A huge rate * dt queues a per-channel backlog capped per step; subsequent
    // empty/zero calls drain that backlog. Driving those calls through the
    // wrapper must preserve and advance the encoder's internal state exactly as
    // the direct path does.
    let make = || RateEncoder::try_new(0.0, 1.0e6, (0.0, 1.0), 1.0).expect("valid");
    let inputs: &[&[f32]] = &[&[1.0], &[], &[0.0], &[], &[0.0]];

    let mut wrapped = make();
    let mut delivered = Vec::new();
    {
        // Capacity larger than the per-step cap so nothing is dropped or split
        // by capacity; we are testing state preservation, not the flush policy.
        let mut streaming =
            StreamingEncoder::try_new(&mut wrapped, 8192, FlushPolicy::Manual).expect("valid");
        for input in inputs {
            streaming
                .encode_step(input, &mut capture(&mut delivered))
                .expect("not blocked");
        }
        streaming.flush_into(&mut capture(&mut delivered));
    }

    // Reference: identical encoder driven directly on the same inputs.
    let mut reference = make();
    let mut expected_batches: Vec<Vec<SpikeEvent>> = Vec::new();
    for input in inputs {
        let out = reference_step(&mut reference, input);
        // Non-empty outputs become batches; empty ones do not.
        if !out.is_empty() {
            expected_batches.push(out);
        }
    }

    let delivered_spikes: Vec<Vec<SpikeEvent>> =
        delivered.iter().map(|b| b.spikes.clone()).collect();
    assert_eq!(
        delivered_spikes, expected_batches,
        "the wrapper must drain the backlog identically to the direct path"
    );
    assert!(
        expected_batches.len() >= 2,
        "the backlog must span more than one call for this test to bite"
    );
}

// --- Sink panic + no replay --------------------------------------------------

#[test]
fn sink_panic_on_second_batch_does_not_replay_the_first() {
    let mut encoder = rate_one_per_step();
    let mut streaming =
        StreamingEncoder::try_new(&mut encoder, 8, FlushPolicy::Manual).expect("valid");

    // Queue three distinct batches (seq 0, 1, 2).
    let mut sink_calls = Vec::new();
    for _ in 0..3 {
        streaming
            .encode_step(&[1.0], &mut capture(&mut sink_calls))
            .expect("queued");
    }
    assert!(sink_calls.is_empty(), "Manual queues without delivering");

    // Flush with a sink that panics on the SECOND delivered batch.
    let mut seen: Vec<u64> = Vec::new();
    let result = catch_unwind(AssertUnwindSafe(|| {
        streaming.flush_into(&mut |batch: SpikeBatch<'_>| {
            seen.push(batch.sequence());
            if seen.len() == 2 {
                panic!("sink refuses the second batch");
            }
        });
    }));
    assert!(
        result.is_err(),
        "the sink panic must propagate out of flush"
    );
    assert_eq!(
        seen,
        vec![0, 1],
        "the first two batches were delivered before the panic"
    );

    // Flush again with a non-panicking sink: neither already-delivered batch
    // (seq 0 or 1) may be replayed.
    let mut after: Vec<u64> = Vec::new();
    streaming.flush_into(&mut |batch: SpikeBatch<'_>| after.push(batch.sequence()));
    assert!(
        !after.contains(&0),
        "batch 0 was delivered; it must never be replayed"
    );
    assert!(
        !after.contains(&1),
        "batch 1 was delivered; it must never be replayed"
    );
}
