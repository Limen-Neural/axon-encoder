//! Conformance suite for the crate-wide spike time contract (#62 / RM-368).
//!
//! Every public encoder is driven through the same assertions, so a new encoder
//! joins the contract by being added to [`encoders`] rather than by hand-writing
//! its own time tests. The rules under test are the ones documented in
//! `axon_encoder::time`:
//!
//! 1. `SpikeEvent::timestamp` is a call-relative [`TickOffset`], bounded by
//!    `time_model().span_ticks()`.
//! 2. Spikes are channel-major, with non-decreasing offsets inside a channel.
//! 3. Repeated spikes from one channel at one offset are contiguous.
//! 4. The time model is stable across calls, so a caller can build a
//!    [`TimeCursor`] once and advance it per call.

use axon_encoder::prelude::*;

/// Inputs chosen to exercise clamping, non-finite handling, and both ends of
/// every encoder's configured range.
const INPUTS: [&[f32]; 6] = [
    &[],
    &[0.0, 0.5, 1.0],
    &[1.0, 0.0, 1.0, 0.0],
    &[-5.0, 5.0, 0.25],
    &[f32::NAN, f32::INFINITY, f32::NEG_INFINITY, 0.75],
    &[0.5; 8],
];

/// Builds one encoder instance for the conformance tables.
type ModulatedFactory = fn() -> Box<dyn ModulatedEncoder>;

/// Every public encoder with a gain-aware path, configured for a 0.0..1.0 input
/// range. Single source of truth for both encoder tables below.
const MODULATED_FACTORIES: &[(&str, ModulatedFactory)] = &[
    ("RateEncoder", || {
        Box::new(RateEncoder::try_new(5.0, 400.0, (0.0, 1.0), 0.002).expect("valid rate"))
    }),
    ("LatencyEncoder", || {
        Box::new(LatencyEncoder::try_new(10, (0.0, 1.0)).expect("valid latency"))
    }),
    ("PhaseEncoder", || {
        Box::new(PhaseEncoder::try_new(16, (0.0, 1.0)).expect("valid phase"))
    }),
    ("PopulationEncoder", || {
        Box::new(PopulationEncoder::try_new(8, (0.0, 1.0), 0.2).expect("valid population"))
    }),
    ("DeltaEncoder", || {
        Box::new(DeltaEncoder::try_new(0.1, 8).expect("valid delta"))
    }),
    ("TemporalEncoder", || {
        Box::new(TemporalEncoder::try_new(8, vec![(0.2, 1)], 8).expect("valid temporal"))
    }),
    ("PredictiveEncoder", || {
        Box::new(PredictiveEncoder::try_new(8, vec![(0.2, 1)], 8).expect("valid predictive"))
    }),
];

/// Every public `Encoder` implementation: the gain-aware ones upcast to
/// `dyn Encoder`, plus `DerivativeEncoder`, which has no modulated path.
fn encoders() -> Vec<(&'static str, Box<dyn Encoder>)> {
    let mut all: Vec<(&'static str, Box<dyn Encoder>)> = MODULATED_FACTORIES
        .iter()
        .map(|(label, build)| (*label, build() as Box<dyn Encoder>))
        .collect();
    all.push((
        "DerivativeEncoder",
        Box::new(DerivativeEncoder::try_new(vec![0.1; 8]).expect("valid derivative")),
    ));
    all
}

/// The gain-aware subset, as `dyn ModulatedEncoder` trait objects.
fn modulated_encoders() -> Vec<(&'static str, Box<dyn ModulatedEncoder>)> {
    MODULATED_FACTORIES
        .iter()
        .map(|(label, build)| (*label, build()))
        .collect()
}

/// Gains spanning identity, silencing, stretching, and non-finite values.
fn gain_cases() -> Vec<EncodingGains> {
    vec![
        EncodingGains::identity(),
        EncodingGains {
            latency_scale: 4.0,
            sensitivity_scale: 4.0,
            threshold_scale: 0.0,
            firing_rate_scale: 8.0,
        },
        EncodingGains {
            latency_scale: 0.0,
            sensitivity_scale: 0.5,
            threshold_scale: 2.0,
            firing_rate_scale: 0.25,
        },
        EncodingGains {
            latency_scale: f32::NAN,
            sensitivity_scale: f32::INFINITY,
            threshold_scale: -1.0,
            firing_rate_scale: f32::NAN,
        },
    ]
}

/// Asserts the full ordering and bounding contract for one call's output.
fn assert_call_conforms(label: &str, model: TimeModel, spikes: &[SpikeEvent]) {
    let mut previous: Option<SpikeEvent> = None;
    // (channel, offset) pairs whose run has ended. Rules 1 and 2 already make a
    // split run impossible, so this is a redundant check kept as a direct
    // statement of rule 3 — it is the rule consumers rely on when they count a
    // burst in one pass.
    let mut closed: Vec<(u16, u64)> = Vec::new();

    for spike in spikes {
        assert!(
            model.contains(spike.timestamp),
            "{label}: offset {} outside span {}",
            spike.timestamp,
            model.span_ticks()
        );

        if let Some(previous) = previous {
            assert!(
                previous.channel <= spike.channel,
                "{label}: channels not in ascending order ({} after {})",
                spike.channel,
                previous.channel
            );
            if previous.channel == spike.channel {
                assert!(
                    previous.timestamp <= spike.timestamp,
                    "{label}: offsets decrease within channel {} ({} after {})",
                    spike.channel,
                    spike.timestamp,
                    previous.timestamp
                );
            }
            if (previous.channel, previous.timestamp) != (spike.channel, spike.timestamp) {
                closed.push((previous.channel, previous.timestamp.ticks()));
            }
        }

        assert!(
            !closed.contains(&(spike.channel, spike.timestamp.ticks())),
            "{label}: coincident spikes on channel {} at offset {} are not contiguous",
            spike.channel,
            spike.timestamp
        );

        previous = Some(*spike);
    }
}

/// Drives one encoder through batch and streaming calls, asserting the contract
/// on every call and checking that the reported model never changes.
fn assert_encoder_conforms(label: &str, encoder: &mut dyn Encoder) {
    let model = encoder.time_model();
    assert!(model.span_ticks() >= 1, "{label}: span must be positive");
    assert!(model.step_ticks() >= 1, "{label}: step must be positive");

    for input in INPUTS {
        let batch = encoder.encode(input);
        assert_call_conforms(&format!("{label} (batch)"), model, &batch.spikes);
        assert_eq!(
            encoder.time_model(),
            model,
            "{label}: time model changed mid-stream"
        );
    }

    encoder.reset();

    for input in INPUTS {
        let streamed = encoder.encode_step(input);
        assert_call_conforms(&format!("{label} (streaming)"), model, &streamed.spikes);
        assert_eq!(
            encoder.time_model(),
            model,
            "{label}: time model changed mid-stream"
        );
    }
}

#[test]
fn every_encoder_conforms_to_the_time_contract() {
    for (label, mut encoder) in encoders() {
        assert_encoder_conforms(label, encoder.as_mut());
    }
}

#[test]
fn modulated_paths_conform_to_the_time_contract() {
    for (label, mut encoder) in modulated_encoders() {
        let model = encoder.time_model();
        for gains in gain_cases() {
            for input in INPUTS {
                let batch = encoder.encode_with_gains(input, gains);
                assert_call_conforms(&format!("{label} (gains, batch)"), model, &batch.spikes);

                let streamed = encoder.encode_step_with_gains(input, gains);
                assert_call_conforms(
                    &format!("{label} (gains, streaming)"),
                    model,
                    &streamed.spikes,
                );
            }
        }
        assert_eq!(
            encoder.time_model(),
            model,
            "{label}: time model changed under modulation"
        );
    }
}

#[test]
fn offsets_are_call_relative_not_absolute() {
    // Whatever an encoder emits on its first call, it must emit again on a later
    // call given the same input: nothing accumulates into the offset itself.
    for (label, mut encoder) in encoders() {
        let input = [1.0, 0.0, 1.0, 0.0];

        encoder.reset();
        let first: Vec<u64> = encoder
            .encode_step(&input)
            .spikes
            .iter()
            .map(|spike| spike.timestamp.ticks())
            .collect();

        for _ in 0..16 {
            let _ = encoder.encode_step(&input);
        }

        let later = encoder.encode_step(&input);
        let span = encoder.time_model().span_ticks();
        assert!(
            later.spikes.iter().all(|spike| spike.timestamp < span),
            "{label}: offsets drifted out of the declared span after 17 calls"
        );

        // Deterministic encoders must reproduce the first call exactly; the
        // stochastic ones (rate, population) only have to stay in the span.
        if matches!(label, "LatencyEncoder" | "PhaseEncoder") {
            let later_offsets: Vec<u64> = later
                .spikes
                .iter()
                .map(|spike| spike.timestamp.ticks())
                .collect();
            assert_eq!(
                first, later_offsets,
                "{label}: offsets are not call-relative"
            );
        }
    }
}

/// Asserts that consecutive calls never place a spike before the end of the
/// previous call's window.
fn assert_calls_never_spill_backwards(label: &str, encoder: &mut dyn Encoder) {
    let mut cursor = TimeCursor::new(encoder.time_model());
    let mut previous_end = 0u64;

    for _ in 0..8 {
        let out = encoder.encode_step(&[1.0, 0.25, 0.75, 0.0]);
        let earliest = cursor
            .absolute_times(&out.spikes)
            .min()
            .unwrap_or(previous_end);
        assert!(
            earliest >= previous_end,
            "{label}: call spilled backwards past the previous window"
        );
        previous_end = cursor.advance();
    }
}

#[test]
fn cursor_keeps_non_overlapping_encoders_monotonic() {
    let non_overlapping = encoders()
        .into_iter()
        .filter(|(_, encoder)| !encoder.time_model().is_overlapping());

    for (label, mut encoder) in non_overlapping {
        assert_calls_never_spill_backwards(label, encoder.as_mut());
    }
}

#[test]
fn only_physically_calibrated_encoders_report_a_timebase() {
    // RateEncoder is configured in seconds, so its tick has a duration.
    let rate = RateEncoder::try_new(1.0, 10.0, (0.0, 1.0), 0.004).expect("valid rate");
    let timebase = rate
        .time_model()
        .timebase()
        .expect("rate reports a timebase");
    assert_eq!(timebase.tick_nanos(), 4_000_000);
    assert_eq!(timebase.offset_nanos(TickOffset::ZERO), 0);

    // A tick below one nanosecond has no representable duration; the model stays
    // dimensionless rather than reporting a wrong one.
    let ultrafast = RateEncoder::try_new(1.0, 10.0, (0.0, 1.0), 1e-12).expect("valid rate");
    assert!(ultrafast.time_model().timebase().is_none());

    // The rest count ticks without claiming a duration.
    for (label, encoder) in encoders() {
        if label == "RateEncoder" {
            continue;
        }
        assert!(
            encoder.time_model().timebase().is_none(),
            "{label}: reported a timebase it cannot know"
        );
    }
}

#[test]
fn rate_bursts_are_contiguous_coincident_repeats() {
    // 5 kHz over a 10 ms step: several spikes per channel land in one step.
    let mut encoder =
        RateEncoder::try_new(5_000.0, 5_000.0, (0.0, 1.0), 0.010).expect("valid rate");
    let model = encoder.time_model();

    let out = encoder.encode_step(&[1.0, 1.0]);
    assert_call_conforms("RateEncoder burst", model, &out.spikes);

    let channel_zero = out.spikes.iter().filter(|spike| spike.channel == 0).count();
    assert!(
        channel_zero > 1,
        "expected a burst on channel 0, got {channel_zero} spike(s)"
    );

    // Every spike in the burst is coincident: the run length is the count.
    assert!(
        out.spikes
            .iter()
            .all(|spike| spike.timestamp == TickOffset::ZERO),
        "burst spikes must share the step's single tick"
    );

    // Channel 0's run is contiguous, so counting is a single pass.
    let first_one = out
        .spikes
        .iter()
        .position(|spike| spike.channel == 1)
        .expect("channel 1 also fires");
    assert!(
        out.spikes[..first_one]
            .iter()
            .all(|spike| spike.channel == 0),
        "channel 0's burst is not contiguous"
    );
}

#[test]
fn embedding_rate_encoder_conforms() {
    // `EmbeddingRateEncoder` keeps its own `forward` API rather than the
    // `Encoder` trait, but it reports and honours the same contract.
    let encoder = EmbeddingRateEncoder::new(&[0.2, 0.9, 0.5], EmbeddingEncoderConfig { v_th: 0.4 });
    let model = encoder.time_model();
    assert_eq!(model, TimeModel::INSTANT);

    let mut state = EncoderState::new_zeros(3);
    for _ in 0..4 {
        let (out, next) = encoder.forward(&state);
        assert_call_conforms("EmbeddingRateEncoder", model, &out.spikes);
        state = next;
    }
}

#[test]
fn conformance_harness_rejects_contract_violations() {
    // Guard the guard: the shared assertions must actually fail on bad output.
    let model = TimeModel::window(4);

    let out_of_span = [SpikeEvent::new(0, 4u64, true)];
    assert!(violates(model, &out_of_span), "span bound not enforced");

    let unsorted_channels = [
        SpikeEvent::new(1, 0u64, true),
        SpikeEvent::new(0, 0u64, true),
    ];
    assert!(
        violates(model, &unsorted_channels),
        "channel-major order not enforced"
    );

    let backwards_offsets = [
        SpikeEvent::new(0, 2u64, true),
        SpikeEvent::new(0, 1u64, true),
    ];
    assert!(
        violates(model, &backwards_offsets),
        "within-channel offset order not enforced"
    );

    let split_repeat = [
        SpikeEvent::new(0, 1u64, true),
        SpikeEvent::new(0, 2u64, true),
        SpikeEvent::new(0, 1u64, true),
    ];
    assert!(
        violates(model, &split_repeat),
        "split coincident run not rejected"
    );

    let legal = [
        SpikeEvent::new(0, 1u64, true),
        SpikeEvent::new(0, 1u64, true),
        SpikeEvent::new(0, 3u64, true),
        SpikeEvent::new(2, 0u64, true),
    ];
    assert!(!violates(model, &legal), "legal output rejected");
}

/// Whether [`assert_call_conforms`] rejects `spikes`.
///
/// Silences the panic hook so expected failures do not print backtraces.
fn violates(model: TimeModel, spikes: &[SpikeEvent]) -> bool {
    let previous_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        assert_call_conforms("harness self-test", model, spikes)
    }));
    std::panic::set_hook(previous_hook);
    result.is_err()
}
