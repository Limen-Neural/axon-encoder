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

#[test]
fn derivative_gain_scales_signed_crossings_and_keeps_sample_history() {
    let mut derivative = DerivativeEncoder::new(vec![1.0, 2.0]);
    let encoder: &mut dyn ModulatedEncoder = &mut derivative;
    let gains = |threshold_scale| EncodingGains {
        threshold_scale,
        // The other scales do not affect derivative encoding.
        sensitivity_scale: 0.0,
        firing_rate_scale: 0.0,
        latency_scale: 0.0,
    };
    let step = |channel, polarity| SpikeEvent::at_step_start(channel, polarity);

    let first = encoder.encode_with_gains(&[0.75, -1.75], gains(0.5));
    assert_eq!(first.spikes, [step(0, true), step(1, false)]);
    assert!(
        encoder
            .encode_step_with_gains(&[3.0, -3.0], gains(4.0))
            .spikes
            .is_empty()
    );
    assert!(
        encoder
            .encode_step_with_gains(&[f32::NAN, f32::INFINITY], gains(0.0))
            .spikes
            .is_empty()
    );
    // NaN gain becomes identity; the prior suppressed finite step still
    // updated the history, and the non-finite step did not.
    assert!(
        encoder
            .encode_step_with_gains(&[3.75, -2.0], gains(f32::NAN))
            .spikes
            .is_empty()
    );
    let crossing = encoder.encode_step_with_gains(&[2.5, 0.25], gains(1.0));
    assert_eq!(crossing.spikes, [step(0, false), step(1, true)]);
    assert!(
        encoder
            .encode_step_with_gains(&[2.5, 0.25], gains(0.0))
            .spikes
            .is_empty()
    );
    let zero_threshold = encoder.encode_step_with_gains(&[2.75, 0.0], gains(0.0));
    assert_eq!(zero_threshold.spikes, [step(0, true), step(1, false)]);
}

#[test]
fn derivative_identity_gain_matches_ordinary_encoding_at_channel_limit() {
    let channels = usize::from(u16::MAX) + 1;
    let mut ordinary = DerivativeEncoder::new(vec![0.5; channels]);
    let mut modulated = ordinary.clone();
    let mut input = vec![0.0; channels + 1];
    input[channels - 1] = -1.0;
    input[channels] = 2.0;

    let expected = ordinary.encode_step(&input);
    assert_eq!(
        expected.spikes,
        [SpikeEvent::at_step_start(u16::MAX, false)]
    );
    assert_eq!(
        modulated.encode_step_with_gains(&input, EncodingGains::identity()),
        expected
    );
    assert_eq!(modulated, ordinary);
}
