use axon_encoder::prelude::*;

/// These helpers call the `ModulatedEncoder` trait methods, not inherent
/// wrappers on the concrete encoder types.
fn encode_via_dyn(
    encoder: &mut dyn ModulatedEncoder,
    input: &[f32],
    modulators: &NeuroModulators,
    curves: &NeuromodulatorGainCurves,
) -> EncodedOutput {
    encoder.encode_with_modulators(input, modulators, curves)
}

fn encode_step_via_dyn(
    encoder: &mut dyn ModulatedEncoder,
    input: &[f32],
    modulators: &NeuroModulators,
    curves: &NeuromodulatorGainCurves,
) -> EncodedOutput {
    encoder.encode_step_with_modulators(input, modulators, curves)
}

#[test]
fn modulated_encoder_supports_trait_object_dispatch() {
    let mut encoder = LatencyEncoder::new(10, (0.0, 1.0));
    let output = encode_via_dyn(
        &mut encoder,
        &[0.5],
        &NeuroModulators::default(),
        &NeuromodulatorGainCurves::default(),
    );

    assert_eq!(output.spikes.len(), 1);
    assert_eq!(output.spikes[0].timestamp, 5);
}

#[test]
fn modulated_encoder_preserves_rate_step_accumulation() {
    let mut encoder = RateEncoder::new(0.0, 5.0, (0.0, 1.0));
    let modulators = NeuroModulators::default();
    let curves = NeuromodulatorGainCurves::default();

    let first = encode_step_via_dyn(&mut encoder, &[1.0], &modulators, &curves);
    let second = encode_step_via_dyn(&mut encoder, &[1.0], &modulators, &curves);

    assert!(first.spikes.is_empty());
    assert_eq!(second.spikes.len(), 1);
}

#[test]
fn direct_gain_dispatch_sanitizes_non_finite_values() {
    let mut encoder = LatencyEncoder::new(10, (0.0, 1.0));
    let output = encoder.encode_with_gains(
        &[0.5],
        EncodingGains {
            latency_scale: f32::NAN,
            ..EncodingGains::identity()
        },
    );

    assert_eq!(output.spikes[0].timestamp, 5);
}

#[test]
fn embedding_rate_encoder_supports_trait_object_dispatch() {
    let mut encoder =
        EmbeddingRateEncoder::try_new(2, EmbeddingEncoderConfig { v_th: 0.5 }).unwrap();

    // `&mut dyn Encoder`: streaming state accumulates across calls.
    {
        let dyn_encoder: &mut dyn Encoder = &mut encoder;
        assert!(dyn_encoder.encode_step(&[0.3, 0.0]).spikes.is_empty());
        let second = dyn_encoder.encode_step(&[0.3, 0.0]);
        assert_eq!(second.spikes, vec![SpikeEvent::at_step_start(0, true)]);
    }

    encoder.reset();

    // `&mut dyn ModulatedEncoder`: gains still reach the shared threshold path.
    let output = encode_via_dyn(
        &mut encoder,
        &[0.6, 0.0],
        &NeuroModulators::default(),
        &NeuromodulatorGainCurves::default(),
    );
    assert_eq!(output.spikes, vec![SpikeEvent::at_step_start(0, true)]);
}
