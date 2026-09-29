use axon_encoder::prelude::*;

#[derive(Debug, PartialEq, Eq)]
struct Delivered(u64, u64, Vec<SpikeEvent>);

fn collect(batches: &mut Vec<Delivered>, batch: SpikeBatch<'_>) {
    batches.push(Delivered(
        batch.sequence(),
        batch.origin(),
        batch.spikes().to_vec(),
    ));
}

#[test]
fn manual_flush_preserves_calls_through_trait_object() {
    let mut encoder = LatencyEncoder::new(9, (0.0, 1.0));
    let mut reference = LatencyEncoder::new(9, (0.0, 1.0));
    let dynamic: &mut dyn Encoder = &mut encoder;
    let mut stream = StreamingEncoder::new(dynamic, 4, FlushPolicy::Manual).unwrap();
    let inputs: &[&[f32]] = &[&[1.0, 0.0], &[], &[0.5], &[0.0]];
    let mut expected = Vec::new();
    let mut received = Vec::new();
    for (i, input) in inputs.iter().enumerate() {
        let mut spikes = Vec::new();
        reference.encode_step_into(input, &mut spikes);
        if !spikes.is_empty() {
            expected.push(Delivered(i as u64, i as u64 * 10, spikes));
        }
        let report = stream
            .encode_step(input, &mut |b: SpikeBatch<'_>| collect(&mut received, b))
            .unwrap();
        assert_eq!(report.flush_reason, None);
        assert_eq!(report.delivered_batches, 0);
    }
    assert_eq!(stream.cursor().origin(), 40);
    assert_eq!(stream.pending_batches(), 3);
    let report = stream.flush_into(&mut |b: SpikeBatch<'_>| collect(&mut received, b));
    assert_eq!(report.reason, Some(FlushReason::Manual));
    assert_eq!(report.delivered_batches, 3);
    assert_eq!(received, expected);
    assert!(stream.is_empty());
    assert_eq!(
        stream
            .flush_into(&mut |b: SpikeBatch<'_>| collect(&mut received, b))
            .delivered_batches,
        0
    );
}

#[test]
fn capacity_and_oversized_batches_deliver_in_order() {
    let mut encoder = LatencyEncoder::new(2, (0.0, 1.0));
    let mut stream = StreamingEncoder::new(&mut encoder, 2, FlushPolicy::OnCapacity).unwrap();
    let mut received = Vec::new();
    stream
        .encode_step(&[0.0, 1.0], &mut |b: SpikeBatch<'_>| {
            collect(&mut received, b)
        })
        .unwrap();
    assert_eq!(stream.buffered_spikes(), 2);
    let report = stream
        .encode_step(&[0.5], &mut |b: SpikeBatch<'_>| collect(&mut received, b))
        .unwrap();
    assert_eq!(report.flush_reason, Some(FlushReason::Capacity));
    assert_eq!(received.iter().map(|b| b.0).collect::<Vec<_>>(), [0]);
    assert_eq!(stream.buffered_spikes(), 1);
    let report = stream
        .encode_step(&[0.0; 4], &mut |b: SpikeBatch<'_>| {
            collect(&mut received, b)
        })
        .unwrap();
    assert_eq!(report.delivered_batches, 2);
    assert_eq!(report.buffered_spikes, 0);
    assert_eq!(received.iter().map(|b| b.0).collect::<Vec<_>>(), [0, 1, 2]);
    assert_eq!(received[2].1, 6);
}

#[test]
fn age_is_measured_in_encoder_ticks_even_on_empty_output() {
    let mut encoder = LatencyEncoder::new(3, (0.0, 1.0));
    let mut stream = StreamingEncoder::new(
        &mut encoder,
        8,
        FlushPolicy::OnCapacityOrAge { max_age_ticks: 8 },
    )
    .unwrap();
    let mut received = Vec::new();
    stream
        .encode_step(&[1.0], &mut |b: SpikeBatch<'_>| collect(&mut received, b))
        .unwrap();
    assert_eq!(stream.pending_age_ticks(), Some(4));
    let report = stream
        .encode_step(&[], &mut |b: SpikeBatch<'_>| collect(&mut received, b))
        .unwrap();
    assert_eq!(report.flush_reason, Some(FlushReason::Age));
    assert_eq!(received.len(), 1);
    assert_eq!(stream.pending_age_ticks(), None);
}

#[test]
fn empty_input_advances_phase_state_without_creating_a_batch() {
    let mut encoder = PhaseEncoder::new(8, (0.0, 1.0));
    let mut reference = PhaseEncoder::new(8, (0.0, 1.0));
    let mut stream = StreamingEncoder::new(&mut encoder, 2, FlushPolicy::Manual).unwrap();
    let mut received = Vec::new();
    reference.encode_step(&[]);
    let report = stream
        .encode_step(&[], &mut |b: SpikeBatch<'_>| collect(&mut received, b))
        .unwrap();
    assert_eq!(report.buffered_spikes, 0);
    assert_eq!(stream.cursor().origin(), 1);
    assert_eq!(
        stream
            .flush_into(&mut |b: SpikeBatch<'_>| collect(&mut received, b))
            .delivered_batches,
        0
    );
    let expected = reference.encode_step(&[0.5]).spikes;
    stream
        .encode_step(&[0.5], &mut |b: SpikeBatch<'_>| collect(&mut received, b))
        .unwrap();
    stream.flush_into(&mut |b: SpikeBatch<'_>| collect(&mut received, b));
    assert_eq!(received, [Delivered(1, 1, expected)]);
}

#[test]
fn empty_input_preserves_rate_backlog_for_the_next_call() {
    let mut encoder = RateEncoder::try_new(0.0, 2000.0, (0.0, 1.0), 1.0).unwrap();
    let mut reference = encoder.clone();
    let mut stream = StreamingEncoder::new(&mut encoder, 2048, FlushPolicy::Manual).unwrap();
    let mut received = Vec::new();
    let mut expected_batches = Vec::new();
    for (sequence, input) in [&[1.0][..], &[][..], &[0.0][..]].into_iter().enumerate() {
        let expected = reference.encode_step(input).spikes;
        stream
            .encode_step(input, &mut |b: SpikeBatch<'_>| collect(&mut received, b))
            .unwrap();
        if sequence == 1 {
            assert_eq!(stream.pending_batches(), 1);
        }
        if !expected.is_empty() {
            assert_eq!(expected.len(), if sequence == 0 { 1024 } else { 976 });
            expected_batches.push(Delivered(sequence as u64, sequence as u64, expected));
        }
    }
    stream.flush_into(&mut |b: SpikeBatch<'_>| collect(&mut received, b));
    assert_eq!(received, expected_batches);
}

#[test]
fn manual_backpressure_rejects_before_encoder_runs_then_flushes() {
    let mut encoder = DeltaEncoder::new(0.0, 3);
    let mut reference = DeltaEncoder::new(0.0, 3);
    let mut stream = StreamingEncoder::new(&mut encoder, 1, FlushPolicy::Manual).unwrap();
    let mut received = Vec::new();
    for input in [&[1.0, 0.0, 0.0][..], &[1.0, 1.0, 0.0][..]] {
        stream
            .encode_step(input, &mut |b: SpikeBatch<'_>| collect(&mut received, b))
            .unwrap();
        reference.encode_step(input);
    }
    assert!(stream.is_blocked());
    assert_eq!(stream.buffered_spikes(), 2);
    assert_eq!(stream.pending_batches(), 2);
    let before = stream.cursor().origin();
    assert_eq!(
        stream.encode_step(&[1.0, 1.0, 1.0], &mut |b: SpikeBatch<'_>| collect(
            &mut received,
            b
        )),
        Err(StreamingError::Backpressure {
            buffered_spikes: 2,
            capacity: 1
        })
    );
    assert_eq!(stream.cursor().origin(), before);
    assert!(received.is_empty());
    assert_eq!(
        stream
            .flush_into(&mut |b: SpikeBatch<'_>| collect(&mut received, b))
            .delivered_batches,
        2
    );
    assert_eq!(received.iter().map(|b| b.0).collect::<Vec<_>>(), [0, 1]);
    assert!(!stream.is_blocked());
    let expected = reference.encode_step(&[1.0, 1.0, 1.0]).spikes;
    stream
        .encode_step(&[1.0, 1.0, 1.0], &mut |b: SpikeBatch<'_>| {
            collect(&mut received, b)
        })
        .unwrap();
    stream.flush_into(&mut |b: SpikeBatch<'_>| collect(&mut received, b));
    assert_eq!(received[2].2, expected);
    assert_eq!(received[2].0, 2);
}

#[test]
fn reset_discards_output_but_preserves_timeline() {
    let mut encoder = DeltaEncoder::new(0.0, 1);
    let mut stream = StreamingEncoder::new(&mut encoder, 1, FlushPolicy::Manual).unwrap();
    let mut received = Vec::new();
    stream
        .encode_step(&[1.0], &mut |b: SpikeBatch<'_>| collect(&mut received, b))
        .unwrap();
    stream
        .encode_step(&[0.0], &mut |b: SpikeBatch<'_>| collect(&mut received, b))
        .unwrap();
    assert!(stream.is_blocked());
    stream.reset();
    assert!(stream.is_empty());
    assert!(!stream.is_blocked());
    assert_eq!(stream.cursor().origin(), 2);
    stream
        .encode_step(&[1.0], &mut |b: SpikeBatch<'_>| collect(&mut received, b))
        .unwrap();
    stream.flush_into(&mut |b: SpikeBatch<'_>| collect(&mut received, b));
    assert_eq!(
        received,
        [Delivered(
            2,
            2,
            DeltaEncoder::new(0.0, 1).encode_step(&[1.0]).spikes
        )]
    );
}

#[test]
fn sink_panic_does_not_replay_delivered_or_in_flight_batch() {
    let mut encoder = LatencyEncoder::new(1, (0.0, 1.0));
    let mut stream = StreamingEncoder::new(&mut encoder, 4, FlushPolicy::Manual).unwrap();
    for _ in 0..3 {
        stream
            .encode_step(&[1.0], &mut |_: SpikeBatch<'_>| {
                panic!("manual mode called sink")
            })
            .unwrap();
    }
    let mut seen = Vec::new();
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        stream.flush_into(&mut |batch: SpikeBatch<'_>| {
            if batch.sequence() == 1 {
                panic!("sink failed");
            }
            seen.push(batch.sequence());
        });
    }));
    assert!(result.is_err());
    assert_eq!(seen, [0]);
    assert_eq!(stream.pending_batches(), 1);
    stream.flush_into(&mut |batch: SpikeBatch<'_>| seen.push(batch.sequence()));
    assert_eq!(seen, [0, 2]);
    stream.reset();
}

#[test]
fn constructor_rejects_zero_bounds() {
    let mut encoder = DeltaEncoder::new(0.0, 1);
    assert!(matches!(
        StreamingEncoder::new(&mut encoder, 0, FlushPolicy::Manual),
        Err(EncoderError::CountMustBePositive {
            parameter: "capacity"
        })
    ));
    assert!(matches!(
        StreamingEncoder::new(
            &mut encoder,
            1,
            FlushPolicy::OnCapacityOrAge { max_age_ticks: 0 }
        ),
        Err(EncoderError::CountMustBePositive {
            parameter: "max_age_ticks"
        })
    ));
}
