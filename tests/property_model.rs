//! Seeded adversarial property sweep over every public `Encoder` (LIM-1334).
//!
//! `tests/time_semantics.rs` checks the time contract on a fixed input table;
//! this suite drives the same contract — plus dimension, reset, sink-parity,
//! and malformed-input rules — through a seeded [`StdRng`] so each trial is
//! reproducible from its index. There is no `proptest` dependency: the sampler
//! is the same table-plus-uniform approach as `src/encoders/property_support.rs`
//! (which is crate-internal, `#![cfg(test)]`, so integration tests cannot
//! import it). Every failure message carries the trial index, the encoder
//! label, and a summary of the input that produced it.
//!
//! Properties:
//!
//! 1. Every call's spikes satisfy `time_model().contains`, are channel-major,
//!    and keep coincident runs contiguous ([`assert_call_conforms`]).
//! 2. `RateEncoder`: batch emits at most one spike per channel; streaming at
//!    most 1024 per channel per step; at `rate·dt ≤ 0.2` the streaming mean
//!    count per step equals `rate·dt` to within one spike over the run.
//! 3. Fixed-width encoders never emit `channel >= cap` for over-long input,
//!    and shorter slices never panic.
//! 4. `LatencyEncoder` / `PhaseEncoder` offsets stay inside the span under
//!    `EncodingGains`, including `latency_scale > 1`.
//! 5. `encode_into` / `encode_step_into` match the returning APIs on
//!    deterministic encoders for lengths around 0/1/63/64/65/128/2048.
//! 6. After `reset()` a deterministic encoder matches a fresh instance.
//! 7. Sequences mixing empty / shorter / longer / NaN / ±inf never panic, and
//!    (LIM-1332) a deterministic encoder that saw a non-finite step matches
//!    one that never did.
//!
//! LIM-1234's panicking-sink coverage lives in `tests/streaming_encoder_tests.rs`
//! and is not repeated here.

use axon_encoder::prelude::*;
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

mod common;
use common::assert_call_conforms;

/// Trials per property per encoder. Kept small enough that the whole suite
/// runs in the same order of time as `time_semantics`.
const TRIALS: usize = 48;
const SEED: u64 = 0x1334;
const CHANNELS: usize = 8;
const RANGE: (f32, f32) = (0.0, 1.0);
const STREAM_CAP: usize = 1024;

/// Lengths that straddle allocation and chunking boundaries.
const LENGTHS: [usize; 9] = [0, 1, 7, 63, 64, 65, 128, 2047, 2048];

// --- encoder table -----------------------------------------------------------

type Factory = fn() -> Box<dyn Encoder>;
type ModulatedFactory = fn() -> Box<dyn ModulatedEncoder>;

/// One public encoder under test.
struct Subject {
    label: &'static str,
    build: Factory,
    /// Gain-aware twin, where the encoder has one.
    modulated: Option<ModulatedFactory>,
    /// Hard upper bound on `channel` for an input of `len` values.
    channel_cap: fn(usize) -> usize,
    /// Output is a pure function of (config, input history).
    deterministic: bool,
    /// A non-finite sample is a no-op on state (LIM-1332 skip rule). `Phase`
    /// advances its oscillation on every call regardless, by design.
    skips_non_finite: bool,
}

fn fixed_width(len: usize) -> usize {
    len.min(CHANNELS)
}
fn input_width(len: usize) -> usize {
    len
}
fn population_width(len: usize) -> usize {
    len * CHANNELS
}

fn subjects() -> Vec<Subject> {
    vec![
        Subject {
            label: "RateEncoder",
            build: || Box::new(RateEncoder::try_new(5.0, 400.0, RANGE, 0.002).unwrap()),
            modulated: Some(|| Box::new(RateEncoder::try_new(5.0, 400.0, RANGE, 0.002).unwrap())),
            channel_cap: input_width,
            deterministic: false,
            skips_non_finite: true,
        },
        Subject {
            label: "LatencyEncoder",
            build: || Box::new(LatencyEncoder::try_new(10, RANGE).unwrap()),
            modulated: Some(|| Box::new(LatencyEncoder::try_new(10, RANGE).unwrap())),
            channel_cap: input_width,
            deterministic: true,
            skips_non_finite: true,
        },
        Subject {
            label: "PhaseEncoder",
            build: || Box::new(PhaseEncoder::try_new(16, RANGE).unwrap()),
            modulated: Some(|| Box::new(PhaseEncoder::try_new(16, RANGE).unwrap())),
            channel_cap: input_width,
            deterministic: true,
            skips_non_finite: false,
        },
        Subject {
            label: "PopulationEncoder",
            build: || Box::new(PopulationEncoder::try_new(CHANNELS, RANGE, 0.2).unwrap()),
            modulated: Some(|| Box::new(PopulationEncoder::try_new(CHANNELS, RANGE, 0.2).unwrap())),
            channel_cap: population_width,
            deterministic: false,
            skips_non_finite: true,
        },
        Subject {
            label: "DeltaEncoder",
            build: || Box::new(DeltaEncoder::try_new(0.1, CHANNELS).unwrap()),
            modulated: Some(|| Box::new(DeltaEncoder::try_new(0.1, CHANNELS).unwrap())),
            channel_cap: fixed_width,
            deterministic: true,
            skips_non_finite: true,
        },
        Subject {
            label: "DerivativeEncoder",
            build: || Box::new(DerivativeEncoder::try_new(vec![0.1; CHANNELS]).unwrap()),
            modulated: None,
            channel_cap: fixed_width,
            deterministic: true,
            skips_non_finite: true,
        },
        Subject {
            label: "BaselineHoldTernaryEncoder",
            build: || Box::new(BaselineHoldTernaryEncoder::try_new(vec![0.1; CHANNELS]).unwrap()),
            modulated: None,
            channel_cap: fixed_width,
            deterministic: true,
            skips_non_finite: true,
        },
        Subject {
            label: "TemporalEncoder",
            build: || Box::new(TemporalEncoder::try_new(8, vec![(0.2, 1)], CHANNELS).unwrap()),
            modulated: Some(|| {
                Box::new(TemporalEncoder::try_new(8, vec![(0.2, 1)], CHANNELS).unwrap())
            }),
            channel_cap: fixed_width,
            deterministic: true,
            skips_non_finite: true,
        },
        Subject {
            label: "PredictiveEncoder",
            build: || Box::new(PredictiveEncoder::try_new(8, vec![(0.2, 1)], CHANNELS).unwrap()),
            modulated: Some(|| {
                Box::new(PredictiveEncoder::try_new(8, vec![(0.2, 1)], CHANNELS).unwrap())
            }),
            channel_cap: fixed_width,
            deterministic: true,
            skips_non_finite: true,
        },
        Subject {
            label: "EmbeddingRateEncoder",
            build: || {
                Box::new(
                    EmbeddingRateEncoder::try_new(CHANNELS, EmbeddingEncoderConfig { v_th: 0.3 })
                        .unwrap(),
                )
            },
            modulated: Some(|| {
                Box::new(
                    EmbeddingRateEncoder::try_new(CHANNELS, EmbeddingEncoderConfig { v_th: 0.3 })
                        .unwrap(),
                )
            }),
            channel_cap: fixed_width,
            deterministic: true,
            skips_non_finite: true,
        },
    ]
}

// --- seeded samplers -----------------------------------------------------------

fn rng_for(property: &str, label: &str) -> StdRng {
    // One stream per (property, encoder) so a trial index names one input.
    let salt = property
        .bytes()
        .chain(label.bytes())
        .fold(0u64, |h, b| h.wrapping_mul(31).wrapping_add(u64::from(b)));
    StdRng::seed_from_u64(SEED ^ salt)
}

/// One sample: in-range uniform most of the time, with the edges and
/// non-finite values over-represented.
fn sample_value(rng: &mut StdRng) -> f32 {
    match rng.random_range(0u8..12) {
        0 => f32::NAN,
        1 => f32::INFINITY,
        2 => f32::NEG_INFINITY,
        3 => RANGE.0 - 10.0,
        4 => RANGE.1 + 10.0,
        5 => RANGE.0,
        6 => RANGE.1,
        7 => f32::MAX,
        _ => RANGE.0 + rng.random::<f32>() * (RANGE.1 - RANGE.0),
    }
}

fn sample_finite(rng: &mut StdRng) -> f32 {
    RANGE.0 + rng.random::<f32>() * (RANGE.1 - RANGE.0)
}

/// A slice whose length is empty, shorter, equal, or longer than the width.
fn sample_len(rng: &mut StdRng) -> usize {
    match rng.random_range(0u8..6) {
        0 => 0,
        1 => rng.random_range(1..CHANNELS),
        2 => CHANNELS,
        3 => CHANNELS + rng.random_range(1..4),
        _ => rng.random_range(0..=2 * CHANNELS),
    }
}

fn sample_input(rng: &mut StdRng, len: usize, finite_only: bool) -> Vec<f32> {
    (0..len)
        .map(|_| {
            if finite_only {
                sample_finite(rng)
            } else {
                sample_value(rng)
            }
        })
        .collect()
}

fn summarize(input: &[f32]) -> String {
    let non_finite = input.iter().filter(|v| !v.is_finite()).count();
    let head: Vec<String> = input.iter().take(6).map(|v| format!("{v}")).collect();
    format!(
        "len={} non_finite={} head=[{}{}]",
        input.len(),
        non_finite,
        head.join(", "),
        if input.len() > 6 { ", …" } else { "" }
    )
}

fn ctx(property: &str, label: &str, trial: usize, input: &[f32]) -> String {
    format!("[{property}] trial {trial} {label}: {}", summarize(input))
}

fn gains_case(rng: &mut StdRng) -> EncodingGains {
    let pick = |rng: &mut StdRng| match rng.random_range(0u8..8) {
        0 => 0.0,
        1 => -1.0,
        2 => f32::NAN,
        3 => f32::INFINITY,
        4 => 0.5,
        5 => 2.0,
        6 => 4.0,
        _ => 1.0,
    };
    EncodingGains {
        latency_scale: pick(rng),
        sensitivity_scale: pick(rng),
        threshold_scale: pick(rng),
        firing_rate_scale: pick(rng),
    }
}

fn collect_into(encoder: &mut dyn Encoder, input: &[f32], step: bool) -> Vec<SpikeEvent> {
    let mut sink = Vec::new();
    if step {
        encoder.encode_step_into(input, &mut sink);
    } else {
        encoder.encode_into(input, &mut sink);
    }
    sink
}

// --- 1. time contract ------------------------------------------------------------

#[test]
fn every_call_conforms_to_the_time_contract_under_adversarial_input() {
    for subject in subjects() {
        let mut rng = rng_for("time", subject.label);
        let mut encoder = (subject.build)();
        let model = encoder.time_model();
        for trial in 0..TRIALS {
            let input = {
                let len = sample_len(&mut rng);
                sample_input(&mut rng, len, false)
            };
            let label = ctx("time", subject.label, trial, &input);
            let out = if trial % 2 == 0 {
                encoder.encode(&input)
            } else {
                encoder.encode_step(&input)
            };
            assert_call_conforms(&label, model, &out.spikes);
            assert_eq!(encoder.time_model(), model, "{label}: model drifted");
        }
    }
}

// --- 2. rate bounds and mean ------------------------------------------------------

#[test]
fn rate_batch_and_streaming_respect_their_caps() {
    let mut rng = rng_for("rate-caps", "RateEncoder");
    for trial in 0..TRIALS {
        // Enormous rates so the caps, not the input, bind.
        let max_hz = [400.0f32, 1e5, 1e7, 1e9][trial % 4];
        let mut encoder = RateEncoder::try_new(0.0, max_hz, RANGE, 0.1).unwrap();
        let input = {
            let len = sample_len(&mut rng);
            sample_input(&mut rng, len, false)
        };
        let label = ctx(
            "rate-caps",
            &format!("RateEncoder@{max_hz}Hz"),
            trial,
            &input,
        );

        let batch = encoder.encode(&input).spikes;
        assert!(
            batch.len() <= input.len(),
            "{label}: batch emitted {} > {} channels",
            batch.len(),
            input.len()
        );
        let streamed = encoder.encode_step(&input).spikes;
        assert!(
            streamed.len() <= STREAM_CAP * input.len(),
            "{label}: streaming emitted {} > 1024 × {}",
            streamed.len(),
            input.len()
        );
    }
}

#[test]
fn rate_streaming_mean_count_matches_rate_times_dt_when_small() {
    let mut rng = rng_for("rate-mean", "RateEncoder");
    let steps = 2_000usize;
    for trial in 0..TRIALS {
        let dt = [0.001f32, 0.002, 0.005, 0.01][trial % 4];
        let max_hz = 200.0f32;
        let x = sample_finite(&mut rng);
        let expected = f64::from(max_hz) * f64::from(x) * f64::from(dt); // ≤ 0.2 × 10
        if expected > 0.2 {
            continue;
        }
        let mut encoder = RateEncoder::try_new(0.0, max_hz, RANGE, dt).unwrap();
        let count: usize = (0..steps)
            .map(|_| encoder.encode_step(&[x]).spikes.len())
            .sum();
        let mean = count as f64 / steps as f64;
        assert!(
            (mean - expected).abs() <= 1.0 / steps as f64 + 1e-9,
            "[rate-mean] trial {trial} RateEncoder x={x} dt={dt}: mean {mean} vs {expected}"
        );
    }
}

// --- 3. dimensions -----------------------------------------------------------------

#[test]
fn excess_channels_never_emit_past_the_cap_and_short_slices_do_not_panic() {
    for subject in subjects() {
        let mut rng = rng_for("dims", subject.label);
        let mut encoder = (subject.build)();
        for trial in 0..TRIALS {
            let len = sample_len(&mut rng);
            let input = sample_input(&mut rng, len, false);
            let label = ctx("dims", subject.label, trial, &input);
            let cap = (subject.channel_cap)(len);
            for spikes in [
                encoder.encode(&input).spikes,
                encoder.encode_step(&input).spikes,
                collect_into(encoder.as_mut(), &input, false),
                collect_into(encoder.as_mut(), &input, true),
            ] {
                for spike in &spikes {
                    assert!(
                        usize::from(spike.channel) < cap,
                        "{label}: channel {} >= cap {cap}",
                        spike.channel
                    );
                }
            }
        }
    }
}

// --- 4. gains keep offsets inside the span ------------------------------------------

#[test]
fn latency_and_phase_offsets_stay_inside_the_span_under_gains() {
    for subject in subjects() {
        let Some(build) = subject.modulated else {
            continue;
        };
        let mut rng = rng_for("gains", subject.label);
        let mut encoder = build();
        let model = encoder.time_model();
        for trial in 0..TRIALS {
            let gains = if trial % 3 == 0 {
                EncodingGains {
                    latency_scale: 1.0 + rng.random::<f32>() * 8.0, // > 1 on purpose
                    ..EncodingGains::identity()
                }
            } else {
                gains_case(&mut rng)
            };
            let input = {
                let len = sample_len(&mut rng);
                sample_input(&mut rng, len, false)
            };
            let label = format!(
                "{} gains={gains:?}",
                ctx("gains", subject.label, trial, &input)
            );
            let out = encoder.encode_with_gains(&input, gains);
            assert_call_conforms(&label, model, &out.spikes);
            let mut sink = Vec::new();
            encoder.encode_step_with_gains_into(&input, gains, &mut sink);
            assert_call_conforms(&format!("{label} (step sink)"), model, &sink);
        }
    }
}

// --- 5. sink parity ---------------------------------------------------------------

#[test]
fn sink_paths_match_returning_paths_on_deterministic_encoders() {
    for subject in subjects().into_iter().filter(|s| s.deterministic) {
        let mut rng = rng_for("sink", subject.label);
        for (trial, &len) in LENGTHS.iter().enumerate() {
            let mut returning = (subject.build)();
            let mut sinking = (subject.build)();
            // A short warm-up sequence, then the probe length, so stateful
            // encoders are compared mid-stream rather than cold.
            for round in 0..3 {
                let input = sample_input(&mut rng, if round == 2 { len } else { CHANNELS }, false);
                let label = ctx("sink", subject.label, trial, &input);
                let step = round % 2 == 1;
                let expected = if step {
                    returning.encode_step(&input).spikes
                } else {
                    returning.encode(&input).spikes
                };
                let actual = collect_into(sinking.as_mut(), &input, step);
                assert_eq!(
                    actual, expected,
                    "{label}: sink path diverged (step={step})"
                );
            }
        }
    }
}

// --- 6. reset ------------------------------------------------------------------------

#[test]
fn reset_restores_a_deterministic_encoder_to_fresh() {
    for subject in subjects().into_iter().filter(|s| s.deterministic) {
        let mut rng = rng_for("reset", subject.label);
        for trial in 0..TRIALS {
            let mut used = (subject.build)();
            let fresh = (subject.build)();
            let mut fresh = fresh;
            for _ in 0..rng.random_range(1..12) {
                let input = {
                    let len = sample_len(&mut rng);
                    sample_input(&mut rng, len, false)
                };
                let _ = used.encode_step(&input);
            }
            used.reset();
            for round in 0..6 {
                let input = {
                    let len = sample_len(&mut rng);
                    sample_input(&mut rng, len, false)
                };
                let label = ctx("reset", subject.label, trial, &input);
                assert_eq!(
                    used.encode_step(&input).spikes,
                    fresh.encode_step(&input).spikes,
                    "{label}: round {round} after reset differs from fresh"
                );
            }
        }
    }
}

// --- 7. malformed sequences and finite recovery ---------------------------------------

#[test]
fn malformed_sequences_never_panic() {
    for subject in subjects() {
        let mut rng = rng_for("malformed", subject.label);
        let mut encoder = (subject.build)();
        for trial in 0..TRIALS {
            let input = {
                let len = sample_len(&mut rng);
                sample_input(&mut rng, len, false)
            };
            let _label = ctx("malformed", subject.label, trial, &input);
            let _ = encoder.encode(&input);
            let _ = encoder.encode_step(&input);
            let _ = collect_into(encoder.as_mut(), &input, false);
            let _ = collect_into(encoder.as_mut(), &input, true);
        }
    }
}

/// LIM-1332: a non-finite sample is a no-op, so an encoder that saw one
/// behaves exactly like one that never did.
#[test]
fn finite_recovery_after_non_finite_steps() {
    for subject in subjects()
        .into_iter()
        .filter(|s| s.deterministic && s.skips_non_finite)
    {
        let mut rng = rng_for("recovery", subject.label);
        for trial in 0..TRIALS {
            let mut poisoned = (subject.build)();
            let mut clean = (subject.build)();
            // Shared finite prefix.
            for _ in 0..rng.random_range(0..10) {
                let input = sample_input(&mut rng, CHANNELS, true);
                assert_eq!(
                    poisoned.encode_step(&input).spikes,
                    clean.encode_step(&input).spikes
                );
            }
            // Only the poisoned encoder sees an all-non-finite step.
            let bad: Vec<f32> = (0..sample_len(&mut rng))
                .map(|_| [f32::NAN, f32::INFINITY, f32::NEG_INFINITY][rng.random_range(0..3)])
                .collect();
            // Stateless Latency still emits (at its collision endpoint) for a
            // non-finite sample; the property under test is that *state* is
            // untouched, which the shared suffix below proves.
            let _ = poisoned.encode_step(&bad);
            let label = ctx("recovery", subject.label, trial, &bad);
            // Shared finite suffix must agree.
            for round in 0..8 {
                let input = {
                    let len = sample_len(&mut rng);
                    sample_input(&mut rng, len, true)
                };
                assert_eq!(
                    poisoned.encode_step(&input).spikes,
                    clean.encode_step(&input).spikes,
                    "{label}: round {round} after the non-finite step diverged"
                );
            }
        }
    }
}
