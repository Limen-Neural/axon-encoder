#![cfg(feature = "serde")]

use axon_encoder::CHECKPOINT_SCHEMA_VERSION;
use axon_encoder::prelude::*;
use serde_json::json;

#[test]
fn spike_event_accepts_unknown_json_fields_without_changing_the_wire_shape() {
    let ticks = (1_u64 << 53) + 17;
    let legacy = json!({"channel": 3, "timestamp": ticks, "polarity": true});
    let mut future = legacy.clone();
    future["application_extension"] = json!({"data": 42});
    let expected = SpikeEvent::new(3, ticks, true);

    for payload in [legacy.clone(), future] {
        let decoded: SpikeEvent = serde_json::from_str(&payload.to_string()).unwrap();
        assert_eq!(decoded, expected);
        assert_eq!(serde_json::to_value(decoded).unwrap(), legacy);
    }
}

#[test]
fn optional_json_fields_may_be_omitted_without_making_required_fields_optional() {
    let output: EncodedOutput = serde_json::from_str(r#"{"spikes":[]}"#).unwrap();
    assert!(output.spikes.is_empty());
    assert!(output.embeddings.is_none());
    assert!(serde_json::from_str::<EncodedOutput>("{}").is_err());

    let curves: ModulatorGainCurves = serde_json::from_str("{}").unwrap();
    assert_eq!(curves, ModulatorGainCurves::default());
}

#[derive(serde::Deserialize)]
struct AppCheckpoint {
    schema_version: u32,
    encoder: DerivativeEncoder,
    cursor: TimeCursor,
}

#[test]
fn checkpoint_restores_encoder_and_absolute_clock_after_non_finite_input() {
    const INVALID: [f32; 3] = [f32::NAN, f32::INFINITY, f32::NEG_INFINITY];

    let mut live = DerivativeEncoder::new(vec![0.5; 3]);
    let origin = (1_u64 << 53) + 17;
    let mut clock = TimeCursor::starting_at(live.time_model(), origin);
    assert_eq!(live.encode_step(&[1.0, -1.0, 0.25]).spikes.len(), 2);
    clock.advance();
    assert!(live.encode_step(&INVALID).spikes.is_empty());
    clock.advance();

    let bare_encoder = serde_json::to_value(&live).unwrap();
    let bare_cursor = serde_json::to_value(clock).unwrap();
    let decoded_bare: DerivativeEncoder = serde_json::from_value(bare_encoder.clone()).unwrap();
    assert_eq!(decoded_bare, live);

    // The application owns the envelope and version. Neither bare payload
    // writes the marker, and extra fields within each payload are discarded.
    assert_eq!(CHECKPOINT_SCHEMA_VERSION, 1);
    let mut payload = json!({
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "encoder": bare_encoder,
        "cursor": bare_cursor,
    });
    payload["encoder"]["future_extension"] = json!({"ignored": true});
    payload["cursor"]["future_extension"] = json!(55);
    let mut restored: AppCheckpoint = serde_json::from_str(&payload.to_string()).unwrap();
    assert_eq!(restored.schema_version, CHECKPOINT_SCHEMA_VERSION);
    assert_eq!(
        serde_json::to_value(&restored.encoder).unwrap(),
        bare_encoder
    );
    assert_eq!(serde_json::to_value(restored.cursor).unwrap(), bare_cursor);
    assert_eq!(restored.cursor.origin(), origin + 2);

    let mut buffer = Vec::new();
    for input in [[0.0, 0.0, 1.0], INVALID, [0.25, -1.0, 0.0]] {
        let expected = live.encode_step(&input);
        if clock.origin() == origin + 2 {
            assert_eq!(
                expected.spikes,
                [
                    SpikeEvent::at_step_start(0, false),
                    SpikeEvent::at_step_start(1, true),
                    SpikeEvent::at_step_start(2, true),
                ],
            );
        }
        buffer.clear();
        restored.encoder.encode_step_into(&input, &mut buffer);
        assert_eq!(buffer, expected.spikes);
        for (restored_spike, live_spike) in buffer.iter().zip(expected.spikes) {
            assert_eq!(
                restored.cursor.absolute(restored_spike.timestamp),
                clock.absolute(live_spike.timestamp)
            );
        }
        restored.cursor.advance();
        clock.advance();
    }
    assert_eq!(restored.encoder, live);
    assert_eq!(restored.cursor, clock);
    assert_eq!(clock.origin(), origin + 5);
}
