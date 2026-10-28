//! Phase: bin-center inverse plus non-finite channel drop.

use axon_encoder::prelude::*;

/// Reference inverse of [`PhaseEncoder`]; defined once in `tests/common/oracles.rs`.
use super::oracles::phase_decode;

#[test]
fn phase_bin_center_inverse_respects_the_half_bin_bound() {
    // AC: phase inverse error bound <= 0.5 / cycle_steps on the quantization
    // grid (times the range span; here span = 1.0).
    let cycle_steps = 16u64;
    let range = (0.0f32, 1.0f32);
    let span = (range.1 - range.0) as f64;
    let half_bin = 0.5 / cycle_steps as f64 * span;
    assert_eq!(half_bin, 0.03125);

    let mut encoder = PhaseEncoder::try_new(cycle_steps, range).expect("valid phase");

    // Sweep the range densely; every probe must decode within half a bin.
    let mut max_abs_err = 0.0f64;
    for step in 0..=200 {
        let value = range.0 + (range.1 - range.0) * (step as f32 / 200.0);
        let out = encoder.encode(&[value]);
        assert_eq!(out.spikes.len(), 1, "finite input emits one spike");
        let bin = out.spikes[0].timestamp.ticks();
        let decoded = phase_decode(bin, cycle_steps, range);
        max_abs_err = max_abs_err.max((decoded - value as f64).abs());
    }
    assert!(
        max_abs_err <= half_bin + 1e-9,
        "phase reconstruction error {max_abs_err} exceeds half-bin bound {half_bin}"
    );
}

#[test]
fn phase_non_finite_inputs_drop_the_channel() {
    // Non-finite inputs are dropped, not emitted, so a decoder must read
    // `spike.channel` to recover the input index and MUST NOT assume
    // `spikes[i].channel == i`. For input [0.0, NaN, 1.0] the output carries
    // channels [0, 2] only.
    let cycle_steps = 16u64;
    let range = (0.0f32, 1.0f32);
    let mut encoder = PhaseEncoder::try_new(cycle_steps, range).expect("valid phase");

    let input = [0.0f32, f32::NAN, 1.0f32];
    let out = encoder.encode(&input);
    assert_eq!(out.spikes.len(), 2, "the NaN channel is dropped");

    let channels: Vec<u16> = out.spikes.iter().map(|spike| spike.channel).collect();
    assert_eq!(channels, vec![0, 2], "surviving channels are 0 and 2");

    // Decode each surviving spike back to its *own* input index via
    // spike.channel, and confirm the reconstruction lands in the right bin.
    for spike in &out.spikes {
        let original = input[spike.channel as usize];
        let decoded = phase_decode(spike.timestamp.ticks(), cycle_steps, range);
        let span = (range.1 - range.0) as f64;
        let half_bin = 0.5 / cycle_steps as f64 * span;
        assert!(
            (decoded - original as f64).abs() <= half_bin + 1e-9,
            "channel {} decoded {decoded} vs input {original}",
            spike.channel
        );
    }

    // All three kinds of non-finite input drop their channel.
    let out = encoder.encode(&[f32::NAN, f32::INFINITY, f32::NEG_INFINITY, 0.5]);
    assert_eq!(out.spikes.len(), 1, "only the finite channel survives");
    assert_eq!(out.spikes[0].channel, 3);
}
