#[cfg(feature = "serde")]
#[test]
fn test_serde_serialization_deserialization() {
    use axon_encoder::prelude::*;

    // 1. Test SpikeEvent
    let spike = SpikeEvent {
        channel: 12,
        timestamp: 42,
        polarity: true,
    };
    let serialized_spike = serde_json::to_string(&spike).unwrap();
    let deserialized_spike: SpikeEvent = serde_json::from_str(&serialized_spike).unwrap();
    assert_eq!(spike, deserialized_spike);

    // 2. Test EncoderConfig
    let config = EncoderConfig {
        input_channels: 10,
        output_channels: 20,
    };
    let serialized_config = serde_json::to_string(&config).unwrap();
    let deserialized_config: EncoderConfig = serde_json::from_str(&serialized_config).unwrap();
    assert_eq!(config, deserialized_config);

    // 3. Test EncodedOutput
    let mut output = EncodedOutput::new();
    output.spikes.push(spike);
    output.embeddings = Some(vec![1.0, 2.0, 3.0]);
    output.metadata = Some(EncodingMetadata::default());

    let serialized_output = serde_json::to_string(&output).unwrap();
    let deserialized_output: EncodedOutput = serde_json::from_str(&serialized_output).unwrap();
    assert_eq!(output, deserialized_output);

    // 4. Test EmbeddingEncoderConfig
    let embed_config = EmbeddingEncoderConfig { v_th: 1.5 };
    let serialized_embed_config = serde_json::to_string(&embed_config).unwrap();
    let deserialized_embed_config: EmbeddingEncoderConfig =
        serde_json::from_str(&serialized_embed_config).unwrap();
    assert_eq!(embed_config, deserialized_embed_config);

    // 5. Test EncoderState
    let state = EncoderState::new_zeros(5);
    let serialized_state = serde_json::to_string(&state).unwrap();
    let deserialized_state: EncoderState = serde_json::from_str(&serialized_state).unwrap();
    assert_eq!(state, deserialized_state);

    // 6. Test RateEncoder
    let rate_encoder = RateEncoder::new(2.0, 10.0, (0.0, 1.0));
    let serialized_rate = serde_json::to_string(&rate_encoder).unwrap();
    let deserialized_rate: RateEncoder = serde_json::from_str(&serialized_rate).unwrap();
    assert_eq!(rate_encoder, deserialized_rate);

    // 7. Test DeltaEncoder
    let delta_encoder = DeltaEncoder::new(0.5, 3);
    let serialized_delta = serde_json::to_string(&delta_encoder).unwrap();
    let deserialized_delta: DeltaEncoder = serde_json::from_str(&serialized_delta).unwrap();
    assert_eq!(delta_encoder, deserialized_delta);

    // 8. Test PopulationEncoder
    let pop_encoder = PopulationEncoder::new(5, (0.0, 1.0), 0.2);
    let serialized_pop = serde_json::to_string(&pop_encoder).unwrap();
    let deserialized_pop: PopulationEncoder = serde_json::from_str(&serialized_pop).unwrap();
    assert_eq!(pop_encoder, deserialized_pop);

    // 9. Test PredictiveEncoder
    let pred_encoder = PredictiveEncoder::new(10, vec![(1.0, 1), (2.0, 2)], 2);
    let serialized_pred = serde_json::to_string(&pred_encoder).unwrap();
    let deserialized_pred: PredictiveEncoder = serde_json::from_str(&serialized_pred).unwrap();
    assert_eq!(pred_encoder, deserialized_pred);

    // 10. Test TemporalEncoder
    let temp_encoder = TemporalEncoder::new(5, vec![(0.5, 1)], 2);
    let serialized_temp = serde_json::to_string(&temp_encoder).unwrap();
    let deserialized_temp: TemporalEncoder = serde_json::from_str(&serialized_temp).unwrap();
    assert_eq!(temp_encoder, deserialized_temp);
}
