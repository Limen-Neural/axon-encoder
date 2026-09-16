#![cfg(feature = "serde")]

//! Operational serde checkpoint continuation for deterministic streaming.
//!
//! Existing serde tests round-trip structs and reject invalid payloads. This
//! suite proves the live-encoder contract: serialize mid-stream, deserialize a
//! second copy, feed the same suffix, and get identical spikes plus identical
//! final serialized state.
//!
//! Stochastic paths (`RateEncoder::encode`, `PopulationEncoder` batch and
//! streaming, `PoissonEncoder`) are out of scope: they draw a thread-local
//! generator serde does not capture, and they do not take caller-owned RNG.

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
    /// When true, the restored checkpoint must differ from a freshly built encoder
    /// so the row cannot pass on construction defaults alone.
    expect_mutated: bool,
    /// Minimum spikes the live suffix must emit (0 = state/reset/silence rows).
    min_suffix_spikes: usize,
    /// Substring that must appear in the checkpoint JSON (Rate backlog).
    checkpoint_must_contain: Option<&'static str>,
    ops: &'static [Op],
}

impl ContinuationCase {
    const fn new(
        label: &'static str,
        split: usize,
        expect_mutated: bool,
        min_suffix_spikes: usize,
        ops: &'static [Op],
    ) -> Self {
        Self {
            label,
            split,
            expect_mutated,
            min_suffix_spikes,
            checkpoint_must_contain: None,
            ops,
        }
    }

    const fn with_needle(mut self, needle: &'static str) -> Self {
        self.checkpoint_must_contain = Some(needle);
        self
    }
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
    E: Encoder + Serialize + DeserializeOwned + PartialEq + std::fmt::Debug,
{
    let ContinuationCase {
        label,
        split,
        expect_mutated,
        min_suffix_spikes,
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

    let mut restored: E = serde_json::from_str(&checkpoint).unwrap_or_else(|err| {
        panic!("{encoder} ({label}) checkpoint at split={split}: deserialize failed: {err}")
    });

    if *expect_mutated {
        let fresh = factory();
        assert_ne!(
            restored, fresh,
            "{encoder} ({label}) checkpoint at split={split}: restored state matched a freshly constructed encoder; config plus mutable state must survive as a distinct checkpoint\nrestored={restored:?}"
        );
    }

    let mut suffix_spikes = 0usize;
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
                suffix_spikes += live_out.spikes.len();
            }
            Op::Reset => {
                live.reset();
                restored.reset();
            }
        }
    }

    assert!(
        suffix_spikes >= *min_suffix_spikes,
        "{encoder} ({label}) checkpoint at split={split}: suffix produced {suffix_spikes} spikes, expected at least {min_suffix_spikes}"
    );

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
    E: Encoder + Serialize + DeserializeOwned + PartialEq + std::fmt::Debug,
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

/// Derivative always writes `last_values`. Inf is only in the suffix of a
/// *finite* checkpoint: JSON cannot round-trip NaN/Inf (see the dedicated test).
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
            ContinuationCase::new("before_any_step", 0, false, 1, DELTA_OPS),
            ContinuationCase::new("mutated_last_values", 2, true, 1, DELTA_OPS),
            ContinuationCase::new("after_empty_and_short", 5, true, 1, DELTA_OPS),
            ContinuationCase::new("after_non_finite_then_reset_suffix", 6, true, 1, DELTA_OPS),
        ],
    );

    run_cases(
        "DerivativeEncoder",
        || DerivativeEncoder::try_new(vec![0.4, 0.8]).expect("valid DerivativeEncoder"),
        &[
            ContinuationCase::new("before_any_step", 0, false, 1, DERIVATIVE_OPS),
            ContinuationCase::new("after_signed_burst", 2, true, 1, DERIVATIVE_OPS),
            // split=4 is the last finite prefix; Inf is only in the suffix.
            ContinuationCase::new(
                "finite_checkpoint_then_non_finite_suffix",
                4,
                true,
                1,
                DERIVATIVE_OPS,
            ),
        ],
    );

    run_cases(
        "TemporalEncoder",
        || TemporalEncoder::try_new(8, vec![(0.5, 1)], 2).expect("valid TemporalEncoder"),
        &[
            ContinuationCase::new("immediately_before_warmup", 5, true, 1, TEMPORAL_OPS),
            ContinuationCase::new("immediately_after_warmup_burst", 6, true, 0, TEMPORAL_OPS),
            ContinuationCase::new(
                "after_empty_short_then_reset_suffix",
                8,
                true,
                0,
                TEMPORAL_OPS,
            ),
        ],
    );

    run_cases(
        "PredictiveEncoder",
        || PredictiveEncoder::try_new(8, vec![(0.5, 1)], 2).expect("valid PredictiveEncoder"),
        &[
            ContinuationCase::new("immediately_before_warmup", 4, true, 1, PREDICTIVE_OPS),
            ContinuationCase::new("immediately_after_warmup", 5, true, 1, PREDICTIVE_OPS),
            ContinuationCase::new("immediately_after_error_burst", 6, true, 1, PREDICTIVE_OPS),
            ContinuationCase::new("reset_in_suffix", 8, true, 1, PREDICTIVE_OPS),
        ],
    );

    run_cases(
        "EmbeddingRateEncoder",
        || {
            EmbeddingRateEncoder::try_new(2, EmbeddingEncoderConfig { v_th: 1.0 })
                .expect("valid EmbeddingRateEncoder")
        },
        &[
            ContinuationCase::new(
                "immediately_before_threshold_crossing",
                1,
                true,
                1,
                EMBEDDING_OPS,
            ),
            ContinuationCase::new(
                "immediately_after_threshold_burst",
                2,
                true,
                1,
                EMBEDDING_OPS,
            ),
            ContinuationCase::new("after_empty_short_non_finite", 6, true, 1, EMBEDDING_OPS),
        ],
    );

    run_cases(
        "PhaseEncoder",
        || PhaseEncoder::try_new(8, (0.0, 1.0)).expect("valid PhaseEncoder"),
        &[
            ContinuationCase::new("before_any_step", 0, false, 1, PHASE_OPS),
            ContinuationCase::new("mutated_current_phase", 2, true, 1, PHASE_OPS),
            ContinuationCase::new("after_empty_input_tick", 3, true, 1, PHASE_OPS),
            ContinuationCase::new("reset_in_suffix", 5, true, 1, PHASE_OPS),
        ],
    );

    run_cases(
        "RateEncoder",
        || RateEncoder::try_new(0.0, 20.0, (0.0, 1.0), 0.05).expect("valid RateEncoder"),
        &[
            ContinuationCase::new("custom_dt_mutated_phase", 2, true, 1, RATE_OPS),
            ContinuationCase::new("after_empty_short_non_finite", 7, true, 1, RATE_OPS),
            ContinuationCase::new("reset_in_suffix", 4, true, 1, RATE_OPS),
        ],
    );

    run_cases(
        "RateEncoder",
        || RateEncoder::try_new(0.0, 1.0e6, (0.0, 1.0), 1.0).expect("valid RateEncoder"),
        &[
            ContinuationCase::new(
                "immediately_before_pending_burst",
                0,
                false,
                1,
                RATE_PENDING_OPS,
            ),
            ContinuationCase::new("non_empty_pending_spikes", 1, true, 1, RATE_PENDING_OPS)
                .with_needle("pending_spikes"),
        ],
    );
}

/// Custom `dt_seconds` and mutated phase must both survive — not only one of them.
#[test]
fn checkpoint_continuation_rate_config_not_construction_defaults() {
    let mut live = RateEncoder::try_new(1.0, 40.0, (-0.5, 1.5), 0.02).expect("valid RateEncoder");
    let _ = live.encode_step(&[1.0]);
    let _ = live.encode_step(&[0.25]);

    let checkpoint = serde_json::to_string(&live).expect("serialize live RateEncoder");
    let restored: RateEncoder = serde_json::from_str(&checkpoint).expect("deserialize RateEncoder");

    assert_eq!(
        restored.dt_seconds(),
        0.02,
        "RateEncoder checkpoint at split=2: custom dt_seconds must survive"
    );
    assert_ne!(
        restored.dt_seconds(),
        RateEncoder::DEFAULT_DT_SECONDS,
        "RateEncoder checkpoint at split=2: dt_seconds must not collapse to RateEncoder::new"
    );
    let fresh_same_config =
        RateEncoder::try_new(1.0, 40.0, (-0.5, 1.5), 0.02).expect("valid RateEncoder");
    assert_ne!(
        restored, fresh_same_config,
        "RateEncoder checkpoint at split=2: mutated phase must differ from a fresh encoder with the same config"
    );

    let mut restored = restored;
    let mut suffix_spikes = 0usize;
    for (step, input) in [&[0.8][..], &[0.0, 1.0][..], &[][..]].iter().enumerate() {
        let live_out = live.encode_step(input);
        let restored_out = restored.encode_step(input);
        assert_eq!(
            live_out,
            restored_out,
            "RateEncoder (config_plus_state) checkpoint at split=2 step={}: outputs differ\nlive={live_out:?}\nrestored={restored_out:?}",
            step + 2
        );
        suffix_spikes += live_out.spikes.len();
    }
    assert!(
        suffix_spikes >= 1,
        "RateEncoder (config_plus_state) checkpoint at split=2: suffix produced no spikes"
    );

    let live_final = serde_json::to_string(&live).expect("final live");
    let restored_final = serde_json::to_string(&restored).expect("final restored");
    assert_eq!(
        live_final, restored_final,
        "RateEncoder (config_plus_state) checkpoint at split=2: final serialized state differs"
    );
}

/// Documented contract boundary: JSON checkpoints require finite floats.
#[test]
fn checkpoint_continuation_derivative_non_finite_state_is_not_json_round_trippable() {
    let mut encoder = DerivativeEncoder::try_new(vec![0.4]).expect("valid DerivativeEncoder");
    let _ = encoder.encode_step(&[f32::INFINITY]);
    // serde_json maps Inf to JSON null; the encoder's f32 state cannot read that back.
    let json = serde_json::to_string(&encoder).expect("serialize Inf last_values");
    assert!(
        json.contains("null"),
        "expected Inf last_values to become JSON null, got {json}"
    );
    let err = serde_json::from_str::<DerivativeEncoder>(&json)
        .expect_err("null last_values must not deserialize into a restorable DerivativeEncoder");
    assert!(
        err.to_string().contains("invalid type"),
        "deserialize error should explain the invalid last_values, got: {err}"
    );
}
