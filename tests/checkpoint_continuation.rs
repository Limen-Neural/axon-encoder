#![cfg(feature = "serde")]

//! Operational serde checkpoint continuation for deterministic streaming.
//!
//! Existing serde tests round-trip structs and reject invalid payloads. This
//! suite proves the live-encoder contract: serialize mid-stream, deserialize a
//! second copy, feed the same suffix, and get identical spikes plus identical
//! final serialized state.
//!
//! Stochastic batch paths (`RateEncoder::encode`, `PopulationEncoder`,
//! `PoissonEncoder`) are out of scope unless the caller owns RNG state.

use axon_encoder::prelude::*;
use serde::Serialize;
use serde::de::DeserializeOwned;

/// One streaming action on a live encoder.
#[derive(Clone, Copy, Debug)]
enum Op {
    Step(&'static [f32]),
    Reset,
}

/// One row of the continuation matrix.
struct ContinuationCase {
    /// Distinguishes rows in failure messages (`encoder` is the type name).
    label: &'static str,
    /// Checkpoint after this many prefix ops (`0` = serialize before any step).
    split: usize,
    /// When true, the checkpoint JSON must differ from a freshly built encoder
    /// so the row cannot pass on construction defaults alone.
    expect_mutated: bool,
    /// Substring that must appear in the checkpoint JSON (Rate backlog).
    checkpoint_must_contain: Option<&'static str>,
    ops: &'static [Op],
}

fn apply_op<E: Encoder>(encoder: &mut E, op: Op) -> Option<EncodedOutput> {
    match op {
        Op::Step(input) => Some(encoder.encode_step(input)),
        Op::Reset => {
            encoder.reset();
            None
        }
    }
}

fn assert_checkpoint_continuation<E>(
    encoder: &str,
    factory: impl Fn() -> E,
    case: &ContinuationCase,
) where
    E: Encoder + Serialize + DeserializeOwned,
{
    let ContinuationCase {
        label,
        split,
        expect_mutated,
        checkpoint_must_contain,
        ops,
    } = case;
    let split = *split;

    assert!(
        split <= ops.len(),
        "{encoder} ({label}): split {split} exceeds op count {}",
        ops.len()
    );

    let mut live = factory();
    for op in &ops[..split] {
        let _ = apply_op(&mut live, *op);
    }

    let checkpoint = serde_json::to_string(&live).unwrap_or_else(|err| {
        panic!("{encoder} ({label}) checkpoint at split={split}: serialize failed: {err}")
    });

    if let Some(needle) = checkpoint_must_contain {
        assert!(
            checkpoint.contains(needle),
            "{encoder} ({label}) checkpoint at split={split}: expected {needle:?} in {checkpoint}"
        );
    }

    if *expect_mutated {
        let fresh = serde_json::to_string(&factory())
            .unwrap_or_else(|err| panic!("{encoder} ({label}): fresh serialize failed: {err}"));
        assert_ne!(
            checkpoint, fresh,
            "{encoder} ({label}) checkpoint at split={split}: serialized state matched construction defaults; config plus mutable state must survive as a distinct checkpoint\ncheckpoint={checkpoint}"
        );
    }

    let mut restored: E = serde_json::from_str(&checkpoint).unwrap_or_else(|err| {
        panic!("{encoder} ({label}) checkpoint at split={split}: deserialize failed: {err}")
    });

    for (offset, op) in ops[split..].iter().enumerate() {
        let step = split + offset;
        match *op {
            Op::Step(input) => {
                let live_out = live.encode_step(input);
                let restored_out = restored.encode_step(input);
                assert_eq!(
                    live_out, restored_out,
                    "{encoder} ({label}) checkpoint at split={split} step={step}: outputs differ\nlive={live_out:?}\nrestored={restored_out:?}"
                );
            }
            Op::Reset => {
                live.reset();
                restored.reset();
            }
        }
    }

    let live_final = serde_json::to_string(&live).unwrap_or_else(|err| {
        panic!(
            "{encoder} ({label}) checkpoint at split={split}: final live serialize failed: {err}"
        )
    });
    let restored_final = serde_json::to_string(&restored).unwrap_or_else(|err| {
        panic!(
            "{encoder} ({label}) checkpoint at split={split}: final restored serialize failed: {err}"
        )
    });
    assert_eq!(
        live_final, restored_final,
        "{encoder} ({label}) checkpoint at split={split}: final serialized state differs\nlive={live_final}\nrestored={restored_final}"
    );

    eprintln!("ok {encoder} / {label} split={split}");
}

fn run_cases<E>(encoder: &str, factory: impl Fn() -> E + Copy, cases: &[ContinuationCase])
where
    E: Encoder + Serialize + DeserializeOwned,
{
    for case in cases {
        assert_checkpoint_continuation(encoder, factory, case);
    }
}

const NAN: f32 = f32::NAN;
const INF: f32 = f32::INFINITY;

/// Delta: non-default threshold, mutated `last_values`, empty/short/NaN, reset.
const DELTA_OPS: &[Op] = &[
    Op::Step(&[0.0, 0.0]),
    Op::Step(&[0.5, 0.0]),
    Op::Step(&[0.5, 0.9]),
    Op::Step(&[]),
    Op::Step(&[0.95]),
    Op::Step(&[NAN, 2.0]),
    Op::Reset,
    Op::Step(&[0.6, 0.6]),
];

/// Derivative always writes `last_values`; a finite step after Inf keeps the
/// checkpoint JSON finite.
const DERIVATIVE_OPS: &[Op] = &[
    Op::Step(&[0.0, 0.0]),
    Op::Step(&[0.5, -0.9]),
    Op::Step(&[]),
    Op::Step(&[0.6]),
    Op::Step(&[INF, 0.0]),
    Op::Step(&[1.0, 1.0]),
    Op::Reset,
    Op::Step(&[0.8, -0.8]),
];

/// Temporal needs 6 samples before the dual-window detector can fire.
/// Split 5 is immediately before that boundary; split 6 is immediately after
/// the first burst (`1,1,1,8,8,8`).
const TEMPORAL_OPS: &[Op] = &[
    Op::Step(&[1.0, 1.0]),
    Op::Step(&[1.0, 1.0]),
    Op::Step(&[1.0, 1.0]),
    Op::Step(&[8.0, 1.0]),
    Op::Step(&[8.0, 1.0]),
    Op::Step(&[8.0, 1.0]),
    Op::Step(&[]),
    Op::Step(&[8.0]),
    Op::Reset,
    Op::Step(&[1.0, 1.0]),
    Op::Step(&[1.0, 1.0]),
    Op::Step(&[1.0, 1.0]),
    Op::Step(&[1.0, 1.0]),
    Op::Step(&[1.0, 1.0]),
    Op::Step(&[1.0, 1.0]),
];

/// Predictive warm-up is 5 samples. Split 4 is immediately before it completes;
/// split 5 is immediately after (baseline seeded, no spike); split 6 is the
/// first prediction-error burst.
const PREDICTIVE_OPS: &[Op] = &[
    Op::Step(&[1.0, 1.0]),
    Op::Step(&[1.0, 1.0]),
    Op::Step(&[1.0, 1.0]),
    Op::Step(&[1.0, 1.0]),
    Op::Step(&[1.0, 1.0]),
    Op::Step(&[4.0, 1.0]),
    Op::Step(&[]),
    Op::Step(&[4.0]),
    Op::Reset,
    Op::Step(&[1.0, 1.0]),
    Op::Step(&[1.0, 1.0]),
    Op::Step(&[1.0, 1.0]),
    Op::Step(&[1.0, 1.0]),
    Op::Step(&[1.0, 1.0]),
    Op::Step(&[4.0, 4.0]),
];

/// Embedding IAF: `v_th = 1.0`, drive `0.6` — split 1 is immediately before the
/// crossing, split 2 immediately after the soft-reset burst.
const EMBEDDING_OPS: &[Op] = &[
    Op::Step(&[0.6, 0.4]),
    Op::Step(&[0.6, 0.4]),
    Op::Step(&[0.9, 0.9]),
    Op::Step(&[]),
    Op::Step(&[0.5]),
    Op::Step(&[NAN, 0.5]),
    Op::Reset,
    Op::Step(&[1.5, 1.5]),
];

/// Phase advances one tick even on empty input; `current_phase` is the state.
const PHASE_OPS: &[Op] = &[
    Op::Step(&[0.0, 1.0]),
    Op::Step(&[0.5]),
    Op::Step(&[]),
    Op::Step(&[0.25, 0.75]),
    Op::Step(&[NAN, 0.5]),
    Op::Reset,
    Op::Step(&[1.0]),
];

/// Deterministic rate with a non-default `dt_seconds` (not `new()`'s 0.1).
const RATE_OPS: &[Op] = &[
    Op::Step(&[1.0]),
    Op::Step(&[0.5]),
    Op::Step(&[]),
    Op::Step(&[0.5]),
    Op::Step(&[0.1, 0.8]),
    Op::Step(&[0.3]),
    Op::Step(&[NAN]),
    Op::Reset,
    Op::Step(&[1.0, 1.0]),
];

/// Huge `rate × dt` fills `pending_spikes` past the per-step cap. Split 0 is
/// immediately before that burst; split 1 is immediately after, with a
/// non-empty backlog still queued.
const RATE_PENDING_OPS: &[Op] = &[
    Op::Step(&[1.0]),
    Op::Step(&[0.0]),
    Op::Step(&[0.0]),
    Op::Step(&[]),
    Op::Step(&[0.0]),
];

#[test]
fn checkpoint_continuation_matrix() {
    run_cases(
        "DeltaEncoder",
        || DeltaEncoder::try_new(0.25, 2).expect("valid DeltaEncoder"),
        &[
            ContinuationCase {
                label: "before_any_step",
                split: 0,
                expect_mutated: false,
                checkpoint_must_contain: None,
                ops: DELTA_OPS,
            },
            ContinuationCase {
                label: "mutated_last_values",
                split: 2,
                expect_mutated: true,
                checkpoint_must_contain: None,
                ops: DELTA_OPS,
            },
            ContinuationCase {
                label: "after_empty_and_short",
                split: 5,
                expect_mutated: true,
                checkpoint_must_contain: None,
                ops: DELTA_OPS,
            },
            ContinuationCase {
                label: "after_non_finite_then_reset_suffix",
                split: 6,
                expect_mutated: true,
                checkpoint_must_contain: None,
                ops: DELTA_OPS,
            },
        ],
    );

    run_cases(
        "DerivativeEncoder",
        || DerivativeEncoder::try_new(vec![0.4, 0.8]).expect("valid DerivativeEncoder"),
        &[
            ContinuationCase {
                label: "before_any_step",
                split: 0,
                expect_mutated: false,
                checkpoint_must_contain: None,
                ops: DERIVATIVE_OPS,
            },
            ContinuationCase {
                label: "after_signed_burst",
                split: 2,
                expect_mutated: true,
                checkpoint_must_contain: None,
                ops: DERIVATIVE_OPS,
            },
            ContinuationCase {
                label: "after_short_input_non_finite_in_suffix",
                split: 4,
                expect_mutated: true,
                checkpoint_must_contain: None,
                ops: DERIVATIVE_OPS,
            },
        ],
    );

    run_cases(
        "TemporalEncoder",
        || TemporalEncoder::try_new(8, vec![(0.5, 1)], 2).expect("valid TemporalEncoder"),
        &[
            ContinuationCase {
                label: "immediately_before_warmup",
                split: 5,
                expect_mutated: true,
                checkpoint_must_contain: None,
                ops: TEMPORAL_OPS,
            },
            ContinuationCase {
                label: "immediately_after_warmup_burst",
                split: 6,
                expect_mutated: true,
                checkpoint_must_contain: None,
                ops: TEMPORAL_OPS,
            },
            ContinuationCase {
                label: "after_empty_short_then_reset_suffix",
                split: 8,
                expect_mutated: true,
                checkpoint_must_contain: None,
                ops: TEMPORAL_OPS,
            },
        ],
    );

    run_cases(
        "PredictiveEncoder",
        || PredictiveEncoder::try_new(8, vec![(0.5, 1)], 2).expect("valid PredictiveEncoder"),
        &[
            ContinuationCase {
                label: "immediately_before_warmup",
                split: 4,
                expect_mutated: true,
                checkpoint_must_contain: None,
                ops: PREDICTIVE_OPS,
            },
            ContinuationCase {
                label: "immediately_after_warmup",
                split: 5,
                expect_mutated: true,
                checkpoint_must_contain: None,
                ops: PREDICTIVE_OPS,
            },
            ContinuationCase {
                label: "immediately_after_error_burst",
                split: 6,
                expect_mutated: true,
                checkpoint_must_contain: None,
                ops: PREDICTIVE_OPS,
            },
            ContinuationCase {
                label: "reset_in_suffix",
                split: 8,
                expect_mutated: true,
                checkpoint_must_contain: None,
                ops: PREDICTIVE_OPS,
            },
        ],
    );

    run_cases(
        "EmbeddingRateEncoder",
        || {
            EmbeddingRateEncoder::try_new(2, EmbeddingEncoderConfig { v_th: 1.0 })
                .expect("valid EmbeddingRateEncoder")
        },
        &[
            ContinuationCase {
                label: "immediately_before_threshold_crossing",
                split: 1,
                expect_mutated: true,
                checkpoint_must_contain: None,
                ops: EMBEDDING_OPS,
            },
            ContinuationCase {
                label: "immediately_after_threshold_burst",
                split: 2,
                expect_mutated: true,
                checkpoint_must_contain: None,
                ops: EMBEDDING_OPS,
            },
            ContinuationCase {
                label: "after_empty_short_non_finite",
                split: 6,
                expect_mutated: true,
                checkpoint_must_contain: None,
                ops: EMBEDDING_OPS,
            },
        ],
    );

    run_cases(
        "PhaseEncoder",
        || PhaseEncoder::try_new(8, (0.0, 1.0)).expect("valid PhaseEncoder"),
        &[
            ContinuationCase {
                label: "before_any_step",
                split: 0,
                expect_mutated: false,
                checkpoint_must_contain: None,
                ops: PHASE_OPS,
            },
            ContinuationCase {
                label: "mutated_current_phase",
                split: 2,
                expect_mutated: true,
                checkpoint_must_contain: None,
                ops: PHASE_OPS,
            },
            ContinuationCase {
                label: "after_empty_input_tick",
                split: 3,
                expect_mutated: true,
                checkpoint_must_contain: None,
                ops: PHASE_OPS,
            },
            ContinuationCase {
                label: "reset_in_suffix",
                split: 5,
                expect_mutated: true,
                checkpoint_must_contain: None,
                ops: PHASE_OPS,
            },
        ],
    );

    run_cases(
        "RateEncoder",
        || RateEncoder::try_new(0.0, 20.0, (0.0, 1.0), 0.05).expect("valid RateEncoder"),
        &[
            ContinuationCase {
                label: "custom_dt_mutated_phase",
                split: 2,
                expect_mutated: true,
                checkpoint_must_contain: None,
                ops: RATE_OPS,
            },
            ContinuationCase {
                label: "after_empty_short_non_finite",
                split: 7,
                expect_mutated: true,
                checkpoint_must_contain: None,
                ops: RATE_OPS,
            },
            ContinuationCase {
                label: "reset_in_suffix",
                split: 4,
                expect_mutated: true,
                checkpoint_must_contain: None,
                ops: RATE_OPS,
            },
        ],
    );

    run_cases(
        "RateEncoder",
        || RateEncoder::try_new(0.0, 1.0e6, (0.0, 1.0), 1.0).expect("valid RateEncoder"),
        &[
            ContinuationCase {
                label: "immediately_before_pending_burst",
                split: 0,
                expect_mutated: false,
                checkpoint_must_contain: None,
                ops: RATE_PENDING_OPS,
            },
            ContinuationCase {
                label: "non_empty_pending_spikes",
                split: 1,
                expect_mutated: true,
                checkpoint_must_contain: Some("pending_spikes"),
                ops: RATE_PENDING_OPS,
            },
        ],
    );
}

/// Construction-default `RateEncoder::new` must not be what a custom-dt live
/// checkpoint serializes to — config and accumulators both have to survive.
#[test]
fn checkpoint_continuation_rate_config_not_construction_defaults() {
    let mut live = RateEncoder::try_new(1.0, 40.0, (-0.5, 1.5), 0.02).expect("valid RateEncoder");
    let _ = live.encode_step(&[1.0]);
    let _ = live.encode_step(&[0.25]);

    let checkpoint = serde_json::to_string(&live).expect("serialize live RateEncoder");
    let defaults = serde_json::to_string(&RateEncoder::new(1.0, 40.0, (-0.5, 1.5)))
        .expect("serialize default-dt RateEncoder");
    assert_ne!(
        checkpoint, defaults,
        "RateEncoder checkpoint at split=2: custom dt_seconds plus mutated phase must not match RateEncoder::new defaults\ncheckpoint={checkpoint}\ndefaults={defaults}"
    );

    let mut restored: RateEncoder =
        serde_json::from_str(&checkpoint).expect("deserialize RateEncoder");
    for (step, input) in [&[0.8][..], &[0.0, 1.0][..], &[][..]].iter().enumerate() {
        let live_out = live.encode_step(input);
        let restored_out = restored.encode_step(input);
        assert_eq!(
            live_out,
            restored_out,
            "RateEncoder (config_plus_state) checkpoint at split=2 step={}: outputs differ\nlive={live_out:?}\nrestored={restored_out:?}",
            step + 2
        );
    }

    let live_final = serde_json::to_string(&live).expect("final live");
    let restored_final = serde_json::to_string(&restored).expect("final restored");
    assert_eq!(
        live_final, restored_final,
        "RateEncoder (config_plus_state) checkpoint at split=2: final serialized state differs"
    );
}
