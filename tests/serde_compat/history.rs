//! Historical reader/writer pairs, migration, rejection, and tolerated fields.
//! These are schema tests, not mid-stream checkpoint continuation tests.

use axon_encoder::prelude::*;
use serde::{Deserialize, Serialize};
use serde_json::json;

use super::{assert_json, assert_rejected, fixture_text, read_fixture};

// Exact persisted field layout from v0.3.0:src/encoders/rate.rs. Serde's
// derived reader ignores unknown fields, including dt_seconds/pending_spikes.
#[derive(Debug, Deserialize, Serialize, PartialEq)]
struct RateEncoderV03 {
    base_rate: f32,
    max_rate: f32,
    range: (f32, f32),
    accumulators: Vec<f32>,
}

#[test]
fn serde_compat_v03_rate_migrates_combined_accumulators() {
    let file = "v0.3/rate_combined.json";
    let pair = "v0.3 writer -> HEAD reader";
    let restored: RateEncoder = read_fixture(file, pair);
    assert_eq!(restored.dt_seconds(), 0.1, "{file} ({pair})");
    assert_eq!(
        serde_json::to_value(restored).unwrap(),
        json!({"base_rate": 0.0, "max_rate": 10.0, "range": [0.0, 1.0],
            "dt_seconds": 0.1_f32, "accumulators": [0.25, 0.5, 0.0], "pending_spikes": [2, 0, 4]}),
        "{file} ({pair}): whole backlog and fractional phase must split"
    );
}

#[test]
fn serde_compat_old_rate_reader_loses_new_backlog() {
    for (file, writer) in [
        ("v0.4/rate_pending.json", "v0.4"),
        ("head/rate_pending.json", "HEAD"),
    ] {
        let pair = format!("{writer} writer -> simulated v0.3 reader");
        let old: RateEncoderV03 = read_fixture(file, &pair);
        assert_eq!(
            old,
            RateEncoderV03 {
                base_rate: 0.0,
                max_rate: 10.0,
                range: (0.0, 1.0),
                accumulators: vec![0.25, 0.5],
            },
            "{file} ({pair}): unknown fields must be ignored"
        );
        // Reloading what the old reader can persist proves that whole-spike
        // debt is lost, not merely that the new field is absent from its struct.
        let lost: RateEncoder =
            serde_json::from_value(serde_json::to_value(&old).unwrap()).unwrap();
        assert_eq!(
            lost.dt_seconds(),
            0.1,
            "{file} ({pair}): custom dt is also lost"
        );
        assert_eq!(
            serde_json::to_value(lost).unwrap()["pending_spikes"],
            json!([0, 0]),
            "{file} ({pair}): backlog loss"
        );
        let current: RateEncoder = read_fixture(file, "new writer -> HEAD reader");
        assert_json(
            &current,
            file,
            "HEAD writer -> historical new-format golden",
        );
        assert_eq!(
            serde_json::to_value(current).unwrap()["pending_spikes"],
            json!([3, 7]),
            "{file} ({writer} writer -> HEAD reader): backlog retained"
        );
    }
}

#[test]
fn serde_compat_v04_embedding_state_is_not_migrated() {
    assert_rejected::<EmbeddingRateEncoder>(
        "v0.4/embedding_rate.json",
        "v0.4 writer -> HEAD reader",
        "missing field `membrane_potentials`",
    );
}

#[test]
fn serde_compat_v04_spike_json_is_identical() {
    let file = "v0.4/spike_event.json";
    let pair = "v0.4 writer -> HEAD reader/writer";
    let spike: SpikeEvent = read_fixture(file, pair);
    assert_eq!(spike, SpikeEvent::new(12, 42u64, false), "{file} ({pair})");
    assert_eq!(
        serde_json::to_string(&spike).unwrap(),
        fixture_text(file, pair).trim(),
        "{file} ({pair}): TickOffset must be transparent"
    );
}

#[test]
fn serde_compat_v04_latency_maximum_is_rejected() {
    assert_rejected::<LatencyEncoder>(
        "v0.4/latency_max.json",
        "v0.4 writer -> HEAD reader",
        &EncoderError::WindowTooLarge {
            parameter: "max_latency",
        }
        .to_string(),
    );
}

#[test]
fn serde_compat_head_timebase_zero_is_rejected() {
    assert_rejected::<Timebase>(
        "head/timebase_zero.json",
        "HEAD payload -> HEAD reader",
        &EncoderError::WindowMustBePositive {
            parameter: "tick_nanos",
        }
        .to_string(),
    );
}

#[test]
fn serde_compat_unknown_spike_fields_are_accepted() {
    let file = "head/spike_event_extra.json";
    let pair = "HEAD extended writer -> HEAD reader";
    let spike: SpikeEvent = read_fixture(file, pair);
    assert_eq!(spike, SpikeEvent::new(12, 42u64, false), "{file} ({pair})");
    assert_json(
        &spike,
        "v0.5/spike_event.json",
        "HEAD extended writer -> HEAD reader/writer (unknown field discarded)",
    );
}

#[test]
fn serde_compat_unknown_poisson_fields_are_accepted() {
    let file = "head/poisson_extra.json";
    let pair = "HEAD extended writer -> HEAD reader";
    let encoder: PoissonEncoder = read_fixture(file, pair);
    assert_eq!(encoder.num_steps, 17, "{file} ({pair})");
    assert_json(
        &encoder,
        "v0.5/poisson.json",
        "HEAD extended writer -> HEAD reader/writer (unknown field discarded)",
    );
}

#[test]
fn serde_compat_encoding_gains_default_and_sanitize() {
    for (file, expected) in [
        (
            "head/encoding_gains_empty.json",
            EncodingGains {
                threshold_scale: 1.0,
                sensitivity_scale: 1.0,
                firing_rate_scale: 1.0,
                latency_scale: 1.0,
            },
        ),
        (
            "head/encoding_gains_unsanitized.json",
            EncodingGains {
                threshold_scale: 0.0,
                sensitivity_scale: 10_000.0,
                firing_rate_scale: 1.0,
                latency_scale: 0.0,
            },
        ),
    ] {
        let pair = "HEAD partial payload -> HEAD reader";
        let gains: EncodingGains = read_fixture(file, pair);
        assert_eq!(gains, expected, "{file} ({pair})");
    }
}

#[test]
fn serde_compat_neuromodulators_missing_fields_are_rejected() {
    assert_rejected::<NeuroModulators>(
        "head/neuromodulators_empty.json",
        "HEAD empty payload -> HEAD reader",
        "missing field `dopamine`",
    );
}

#[test]
fn serde_compat_rate_first_step_keeps_zero_filled_pending_vector() {
    let file = "head/rate_first_step.json";
    let pair = "HEAD writer after first encode_step -> HEAD golden/reader";
    let mut encoder = RateEncoder::try_new(0.0, 10.0, (0.0, 1.0), 0.125).unwrap();
    encoder.encode_step(&[0.25]);
    assert_json(&encoder, file, pair);
    let restored: RateEncoder = read_fixture(file, pair);
    assert_eq!(restored, encoder, "{file} ({pair})");
}

#[test]
fn serde_compat_v04_output_discards_removed_metadata() {
    let file = "v0.4/encoded_output.json";
    let pair = "v0.4 writer -> HEAD reader";
    let output: EncodedOutput = read_fixture(file, pair);
    assert_eq!(
        output,
        EncodedOutput {
            spikes: vec![
                SpikeEvent::new(12, 42u64, false),
                SpikeEvent::at_step_start(3, true)
            ],
            embeddings: Some(vec![0.25, -0.5])
        },
        "{file} ({pair})"
    );
    assert_json(
        &output,
        "v0.5/encoded_output.json",
        "v0.4 writer -> HEAD reader/writer (metadata discarded)",
    );
}
