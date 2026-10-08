use axon_encoder::prelude::{
    EmbeddingEncoderConfig, EmbeddingRateEncoder, EncodedOutput, Encoder, PoissonEncoder,
    PopulationEncoder, RateEncoder,
};
use rand::SeedableRng;
use rand::rngs::StdRng;

#[test]
fn encoded_output_defaults_are_stable() {
    let output: EncodedOutput = EncodedOutput::new();

    assert!(output.spikes.is_empty());
    assert!(output.embeddings.is_none());
}

#[test]
fn encoded_output_supports_optional_dense_embedding() {
    let mut output = EncodedOutput::new();
    output.embeddings = Some(vec![0.1, 0.2, 0.3]);

    assert_eq!(
        output.embeddings.as_deref(),
        Some([0.1, 0.2, 0.3].as_slice())
    );
}

#[test]
fn embedding_rate_encoder_produces_standardized_output() {
    let mut encoder = EmbeddingRateEncoder::new(3, EmbeddingEncoderConfig { v_th: 0.4 });
    let output = encoder.encode(&[0.5, 0.1, 0.5]);

    assert!(!output.spikes.is_empty());
    assert!(output.embeddings.is_none());
}

#[test]
fn seeded_public_apis_replay_all_stochastic_encoders() {
    let inputs = [[0.2, 0.8, 0.5], [1.0, 0.0, 0.4], [0.6, 0.3, 0.9]];

    let mut rate_a = RateEncoder::try_new(2.0, 80.0, (0.0, 1.0), 0.01).unwrap();
    let mut rate_b = rate_a.clone();
    let mut rate_rng_a = StdRng::seed_from_u64(0x12_1390);
    let mut rate_rng_b = StdRng::seed_from_u64(0x12_1390);
    for input in inputs {
        assert_eq!(
            rate_a.encode_with_rng(&input, &mut rate_rng_a),
            rate_b.encode_with_rng(&input, &mut rate_rng_b)
        );
    }

    let mut population_a = PopulationEncoder::new(5, (0.0, 1.0), 0.2);
    let mut population_b = population_a.clone();
    let mut population_rng_a = StdRng::seed_from_u64(0x12_1390);
    let mut population_rng_b = StdRng::seed_from_u64(0x12_1390);
    for input in inputs {
        assert_eq!(
            population_a.encode_step_with_rng(&input, &mut population_rng_a),
            population_b.encode_step_with_rng(&input, &mut population_rng_b)
        );
    }

    let poisson = PoissonEncoder::new(32);
    let mut poisson_rng_a = StdRng::seed_from_u64(0x12_1390);
    let mut poisson_rng_b = StdRng::seed_from_u64(0x12_1390);
    for probability in [0.2, 0.8, 0.5] {
        assert_eq!(
            poisson.encode_with_rng(probability, &mut poisson_rng_a),
            poisson.encode_with_rng(probability, &mut poisson_rng_b)
        );
    }
}

#[test]
fn prelude_exports_the_streaming_types() {
    // A glob import of the prelude must bring every public streaming type into
    // scope. Referencing each one here proves the re-exports resolve; the body
    // exercises them lightly so the imports are genuinely used, not just named.
    use axon_encoder::prelude::{
        BatchSink, EncoderError, FlushPolicy, FlushReason, RateEncoder, SpikeBatch, StepReport,
        StreamingEncoder, StreamingError,
    };

    // FlushPolicy and FlushReason values.
    let policy: FlushPolicy = FlushPolicy::OnCapacityOrAge { max_age_ticks: 4 };
    assert_eq!(policy, FlushPolicy::OnCapacityOrAge { max_age_ticks: 4 });
    let _reason: FlushReason = FlushReason::Capacity;

    // StreamingError variant is reachable through the prelude.
    let _err: StreamingError = StreamingError::Backpressure {
        buffered_spikes: 1,
        capacity: 1,
    };

    // Construct a StreamingEncoder over a RateEncoder to prove the wrapper type
    // and its constructor resolve from the prelude.
    let mut encoder = RateEncoder::try_new(0.0, 10.0, (0.0, 1.0), 0.1).expect("valid RateEncoder");
    let mut streaming =
        StreamingEncoder::try_new(&mut encoder, 8, policy).expect("valid StreamingEncoder");

    // BatchSink is usable via a closure; StepReport is the encode_step result.
    let mut sink = |_batch: SpikeBatch<'_>| {};
    let report: StepReport = streaming
        .encode_step(&[1.0], &mut sink)
        .expect("first step never blocks");
    let _ = report.delivered_batches();

    // FlushReport comes back from flush_into.
    let flush = streaming.flush_into(&mut sink);
    let _ = flush.delivered_batches();

    // BatchSink is object-safe: coerce the closure to a trait object.
    let _dyn_sink: &mut dyn BatchSink = &mut sink;

    // EncoderError is the constructor error type, also from the prelude.
    let _construct_err: Result<StreamingEncoder<'_, RateEncoder>, EncoderError> =
        StreamingEncoder::try_new(&mut encoder, 0, FlushPolicy::Manual);
}
