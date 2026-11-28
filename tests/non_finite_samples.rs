//! Regression matrix for the public non-finite sample contract.
use axon_encoder::prelude::*;

const NON_FINITE: [f32; 3] = [f32::NAN, f32::INFINITY, f32::NEG_INFINITY];

fn history_encoders() -> Vec<Box<dyn Encoder>> {
    vec![
        Box::new(BaselineHoldTernaryEncoder::new(vec![0.5; 3])),
        Box::new(DeltaEncoder::new(0.5, 3)),
        Box::new(DerivativeEncoder::new(vec![0.5; 3])),
        Box::new(EmbeddingRateEncoder::new(
            3,
            EmbeddingEncoderConfig { v_th: 0.5 },
        )),
        Box::new(PredictiveEncoder::try_new(6, vec![(0.5, 1)], 3).unwrap()),
        Box::new(TemporalEncoder::new(6, vec![0.5], 3)),
    ]
}

fn assert_all_entrypoints(encoder: &mut dyn Encoder, expected: &[SpikeEvent]) {
    assert_eq!(encoder.encode(&NON_FINITE).spikes, expected);
    encoder.reset();
    assert_eq!(encoder.encode_step(&NON_FINITE).spikes, expected);
    encoder.reset();
    let mut sink = Vec::new();
    encoder.encode_into(&NON_FINITE, &mut sink);
    assert_eq!(sink, expected);
    encoder.reset();
    sink.clear();
    encoder.encode_step_into(&NON_FINITE, &mut sink);
    assert_eq!(sink, expected);
    encoder.reset();
}

#[test]
fn non_finite_sample_matrix_for_the_encoder_trait() {
    for mut encoder in history_encoders() {
        assert_all_entrypoints(encoder.as_mut(), &[]);
    }

    assert_all_entrypoints(&mut PhaseEncoder::new(8, (0.0, 1.0)), &[]);
    assert_all_entrypoints(&mut PopulationEncoder::new(3, (0.0, 1.0), 0.2), &[]);
    assert_all_entrypoints(&mut RateEncoder::new(10.0, 100.0, (0.0, 1.0)), &[]);

    let mut latency = LatencyEncoder::new(8, (0.0, 1.0));
    assert_all_entrypoints(
        &mut latency,
        &[
            SpikeEvent::new(0, 8_u64, true),
            SpikeEvent::new(1, 0_u64, true),
            SpikeEvent::new(2, 8_u64, true),
        ],
    );
}

#[test]
fn non_finite_samples_do_not_seed_or_advance_finite_history() {
    for (mut control, mut observed) in history_encoders().into_iter().zip(history_encoders()) {
        // Interleave rejected samples during warm-up, then once again after it.
        for _ in 0..6 {
            assert!(observed.encode_step(&NON_FINITE).spikes.is_empty());
            assert_eq!(
                observed.encode_step(&[0.0; 3]),
                control.encode_step(&[0.0; 3])
            );
        }
        assert!(observed.encode_step(&NON_FINITE).spikes.is_empty());
        let expected = control.encode_step(&[2.0; 3]);
        assert_eq!(expected.spikes.len(), 3, "probe must exercise all channels");
        assert_eq!(observed.encode_step(&[2.0; 3]), expected);
    }
}

#[test]
fn rejected_samples_do_not_stop_the_phase_clock_or_block_rate_backlog() {
    let mut phase = PhaseEncoder::new(8, (0.0, 1.0));
    assert!(phase.encode_step(&NON_FINITE).spikes.is_empty());
    assert_eq!(phase.current_phase(), 1);

    let mut rate = RateEncoder::try_new(0.0, 1e6, (0.0, 1.0), 1.0).unwrap();
    assert!(!rate.encode_step(&[1.0; 3]).spikes.is_empty());
    let mut zero_drive = rate.clone();
    let owed = zero_drive.encode_step(&[0.0; 3]);
    assert!(
        !owed.spikes.is_empty(),
        "fixture must have outstanding spikes"
    );
    assert_eq!(rate.encode_step(&NON_FINITE), owed);
    assert_eq!(rate, zero_drive);
}

#[test]
fn poisson_probability_endpoints_and_physical_inputs_are_distinct() {
    let encoder = PoissonEncoder::new(4);
    for (bad, expected) in NON_FINITE.into_iter().zip([0, 1, 0]) {
        assert_eq!(encoder.encode(bad), vec![expected; 4]);
        assert_eq!(encoder.encode_step(bad), expected);
        assert_eq!(encoder.encode_rate_hz(bad, 1.0), vec![0; 4]);
        assert_eq!(encoder.encode_rate_hz_step(1.0, bad), 0);
    }
}
