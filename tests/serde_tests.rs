#![cfg(feature = "serde")]

use axon_encoder::prelude::*;

#[test]
fn test_serde_core_io() {
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
}

#[test]
fn test_serde_encoders_and_state() {
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
    let temp_encoder = TemporalEncoder::new(6, vec![(0.5, 1)], 2);
    let serialized_temp = serde_json::to_string(&temp_encoder).unwrap();
    let deserialized_temp: TemporalEncoder = serde_json::from_str(&serialized_temp).unwrap();
    assert_eq!(temp_encoder, deserialized_temp);

    // 11. Test GainCurve and modulation config types
    let gain_curve = GainCurve::new((0.0, 1.0), (0.5, 2.0));
    let serialized_curve = serde_json::to_string(&gain_curve).unwrap();
    let deserialized_curve: GainCurve = serde_json::from_str(&serialized_curve).unwrap();
    assert_eq!(gain_curve, deserialized_curve);

    let modulator_curves = ModulatorGainCurves {
        threshold: Some(gain_curve),
        sensitivity: Some(GainCurve::new((0.0, 1.0), (1.0, 1.5))),
        firing_rate: Some(GainCurve::new((0.0, 1.0), (1.0, 2.5))),
    };
    let serialized_modulator_curves = serde_json::to_string(&modulator_curves).unwrap();
    let deserialized_modulator_curves: ModulatorGainCurves =
        serde_json::from_str(&serialized_modulator_curves).unwrap();
    assert_eq!(modulator_curves, deserialized_modulator_curves);

    let encoding_gains = EncodingGains {
        threshold_scale: 0.75,
        sensitivity_scale: 1.25,
        firing_rate_scale: 1.5,
    };
    let serialized_gains = serde_json::to_string(&encoding_gains).unwrap();
    let deserialized_gains: EncodingGains = serde_json::from_str(&serialized_gains).unwrap();
    assert_eq!(encoding_gains, deserialized_gains);

    let neuromodulator_gain_curves = NeuromodulatorGainCurves {
        dopamine: modulator_curves,
        cortisol: ModulatorGainCurves {
            threshold: Some(GainCurve::new((0.0, 1.0), (1.0, 0.5))),
            ..Default::default()
        },
        acetylcholine: ModulatorGainCurves {
            firing_rate: Some(GainCurve::new((0.0, 1.0), (1.0, 1.2))),
            ..Default::default()
        },
        tempo: ModulatorGainCurves {
            sensitivity: Some(GainCurve::new((0.0, 1.0), (1.0, 1.1))),
            ..Default::default()
        },
    };
    let serialized_neuromodulator_gain_curves =
        serde_json::to_string(&neuromodulator_gain_curves).unwrap();
    let deserialized_neuromodulator_gain_curves: NeuromodulatorGainCurves =
        serde_json::from_str(&serialized_neuromodulator_gain_curves).unwrap();
    assert_eq!(
        neuromodulator_gain_curves,
        deserialized_neuromodulator_gain_curves
    );
    // 11. Test LatencyEncoder
    let latency_encoder = LatencyEncoder::new(12, (0.0, 1.0));
    let serialized_latency = serde_json::to_string(&latency_encoder).unwrap();
    let deserialized_latency: LatencyEncoder = serde_json::from_str(&serialized_latency).unwrap();
    assert_eq!(latency_encoder, deserialized_latency);

    // 12. Test PhaseEncoder
    let phase_encoder = PhaseEncoder::new(8, (0.0, 1.0));
    let serialized_phase = serde_json::to_string(&phase_encoder).unwrap();
    let deserialized_phase: PhaseEncoder = serde_json::from_str(&serialized_phase).unwrap();
    assert_eq!(phase_encoder, deserialized_phase);
}

#[test]
fn test_serde_validation_failures() {
    // 1. Mismatched history and thresholds length in PredictiveEncoder
    let invalid_pred_json = r#"{
        "history": [[0.0]],
        "thresholds": [],
        "history_depth": 10,
        "deviation_thresholds": []
    }"#;
    let res: Result<PredictiveEncoder, _> = serde_json::from_str(invalid_pred_json);
    assert!(res.is_err());

    // 2. PredictiveEncoder history_depth too small
    let invalid_pred_depth_json = r#"{
        "history": [[0.0]],
        "thresholds": [0.0],
        "history_depth": 2,
        "deviation_thresholds": []
    }"#;
    let res: Result<PredictiveEncoder, _> = serde_json::from_str(invalid_pred_depth_json);
    assert!(res.is_err());

    // 3. TemporalEncoder history_depth too small
    let invalid_temp_depth_json = r#"{
        "history": [[0.0]],
        "history_depth": 5,
        "change_thresholds": []
    }"#;
    let res: Result<TemporalEncoder, _> = serde_json::from_str(invalid_temp_depth_json);
    assert!(res.is_err());

    // 4. LatencyEncoder invalid range (min >= max) must be rejected
    let invalid_latency_json = r#"{
        "max_latency": 5,
        "range": [1.0, 0.5]
    }"#;
    let res: Result<LatencyEncoder, _> = serde_json::from_str(invalid_latency_json);
    assert!(res.is_err());

    let equal_range_latency_json = r#"{
        "max_latency": 5,
        "range": [1.0, 1.0]
    }"#;
    let res: Result<LatencyEncoder, _> = serde_json::from_str(equal_range_latency_json);
    assert!(res.is_err());
}
