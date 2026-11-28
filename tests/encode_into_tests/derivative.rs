use axon_encoder::prelude::*;

#[test]
fn derivative_modulator_curve_changes_batch_and_streaming_sink_output() {
    let mut returning = DerivativeEncoder::new(vec![1.0; 2]);
    let mut sink_based = returning.clone();
    let mut ordinary = returning.clone();
    let modulators = NeuroModulators {
        dopamine: 1.0,
        ..Default::default()
    };
    let curves = NeuromodulatorGainCurves {
        dopamine: ModulatorGainCurves {
            threshold: Some(GainCurve::new((0.0, 1.0), (1.0, 0.5))),
            ..Default::default()
        },
        ..Default::default()
    };
    let inputs = [[0.75, -0.75], [0.0, 0.0]];
    let mut buffer = Vec::new();

    for (step, input) in inputs.iter().enumerate() {
        buffer.clear();
        let spikes = if step == 0 {
            sink_based.encode_with_modulators_into(input, &modulators, &curves, &mut buffer);
            returning
                .encode_with_modulators(input, &modulators, &curves)
                .spikes
        } else {
            sink_based.encode_step_with_modulators_into(input, &modulators, &curves, &mut buffer);
            returning
                .encode_step_with_modulators(input, &modulators, &curves)
                .spikes
        };
        let expected = [
            SpikeEvent::at_step_start(0, step == 0),
            SpikeEvent::at_step_start(1, step != 0),
        ];
        assert!(ordinary.encode_step(input).spikes.is_empty());
        assert_eq!(spikes, expected);
        assert_eq!(buffer, expected);
    }
}

#[test]
fn derivative_direct_gain_sinks_match_returning_paths_and_append() {
    let channels = 70; // More than one sink chunk.
    let mut returning = DerivativeEncoder::new(vec![1.0; channels]);
    let mut sink_based = returning.clone();
    let gains = EncodingGains {
        threshold_scale: 0.5,
        ..EncodingGains::identity()
    };
    let prefix = SpikeEvent::at_step_start(99, true);
    let mut buffer = vec![prefix];

    let initial = vec![0.75; channels + 1];
    let batch = returning.encode_with_gains(&initial, gains);
    assert_eq!(batch.spikes.len(), channels);
    sink_based.encode_with_gains_into(&initial, gains, &mut buffer);
    assert_eq!(&buffer[1..], &batch.spikes);

    let mut next = vec![0.0; channels - 1];
    next[0] = f32::NAN; // Skip one channel without losing its history.
    let step = returning.encode_step_with_gains(&next, gains);
    assert_eq!(step.spikes.len(), channels - 2);
    let appended_at = buffer.len();
    sink_based.encode_step_with_gains_into(&next, gains, &mut buffer);
    assert_eq!(&buffer[appended_at..], &step.spikes);
    assert_eq!(buffer[0], prefix);
    assert_eq!(&buffer[1..appended_at], &batch.spikes);
    assert_eq!(sink_based, returning);
}
