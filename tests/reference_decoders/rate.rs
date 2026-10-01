//! Rate: streaming count/time recovers rate_hz; batch is capped by the
//! Bernoulli per-call ceiling and can never recover Hz when rate*dt >= 1.

use axon_encoder::prelude::*;

/// Count-rate decode: spikes counted over `steps` calls, divided by elapsed
/// time `steps * dt_seconds`.
fn count_rate_hz(spike_count: u64, steps: u64, dt_seconds: f64) -> f64 {
    spike_count as f64 / (steps as f64 * dt_seconds)
}

#[test]
fn rate_streaming_count_over_time_recovers_the_rate() {
    // 50 Hz at dt = 1 ms: the deterministic streaming accumulator advances by
    // rate*dt = 0.05 spikes/step, so it takes 20 steps to emit one spike. Over
    // many steps the count/time decode converges to 50 Hz exactly.
    let dt = 0.001f32;
    // base_rate == max_rate pins the rate independent of the input value, so the
    // oracle is exact rather than input-dependent.
    let mut encoder = RateEncoder::try_new(50.0, 50.0, (0.0, 1.0), dt).expect("valid rate");

    // Per-step expected increment is exactly 0.05.
    // Drive 20 steps: the accumulator reaches 1.0 and emits precisely one spike.
    let mut count = 0u64;
    for _ in 0..20 {
        count += encoder.encode_step(&[1.0]).spikes.len() as u64;
    }
    assert_eq!(count, 1, "0.05 spikes/step reaches 1 spike after 20 steps");

    // Over a long run the count/time decode is 50 Hz within tight tolerance.
    encoder.reset();
    let steps = 20_000u64;
    let mut total = 0u64;
    for _ in 0..steps {
        total += encoder.encode_step(&[1.0]).spikes.len() as u64;
    }
    let decoded = count_rate_hz(total, steps, dt as f64);
    assert!(
        (decoded - 50.0).abs() <= 0.5,
        "streaming count-rate {decoded} Hz should recover ~50 Hz"
    );
}

#[test]
fn rate_batch_is_bernoulli_capped_while_streaming_recovers_hz() {
    // AC: a high rate*dt assertion (5 kHz x 10 ms => rate*dt = 50) proving a
    // count-rate decoder CANNOT silently pass on the batch `encode` path.
    //
    // Batch `encode` is Bernoulli: at most one spike per channel per call with
    // probability 1 - exp(-rate*dt). At rate*dt = 50 that probability is
    // indistinguishable from 1.0, so the batch count-rate decode saturates at
    // probability/dt ~= 1/dt = 100 Hz and can never reach the true 5 kHz.
    //
    // Streaming `encode_step` accumulates rate*dt = 50 whole spikes per step and
    // recovers ~5000 Hz.
    let dt = 0.010f32;
    let dt_f64 = dt as f64;
    let true_rate = 5_000.0f32;
    // Pin base == max so the emitted rate is exactly 5 kHz for any input.
    let mut encoder = RateEncoder::try_new(true_rate, true_rate, (0.0, 1.0), dt).expect("valid");

    // The theoretical batch ceiling: probability / dt.
    let batch_prob = probability_from_rate_hz(true_rate, dt);
    let batch_ceiling_hz = batch_prob as f64 / dt_f64;
    assert!(
        (batch_ceiling_hz - 100.0).abs() < 1e-3,
        "batch ceiling {batch_ceiling_hz} Hz should be ~100 Hz (1 - exp(-50) ~= 1)"
    );

    // Empirical batch decode over many independent calls: each call yields at
    // most one spike per channel, so the decode cannot exceed ~1/dt = 100 Hz.
    let calls = 2_000u64;
    let mut batch_count = 0u64;
    for _ in 0..calls {
        let out = encoder.encode(&[1.0]);
        assert!(
            out.spikes.len() <= 1,
            "batch emits at most one spike/channel"
        );
        batch_count += out.spikes.len() as u64;
    }
    let batch_hz = count_rate_hz(batch_count, calls, dt_f64);
    assert!(
        batch_hz <= 110.0,
        "batch count-rate {batch_hz} Hz must stay near the ~100 Hz Bernoulli ceiling"
    );
    // Lower bound: with probability = 1 - exp(-rate*dt) and rate*dt = 50 the
    // per-call spike probability is indistinguishable from 1.0, so essentially
    // every call must emit its one spike. Without this bound `batch_hz <= 110`
    // is trivially satisfied even by a batch path that emits nothing, so a
    // batch-silencing regression would pass silently. The chance of even a
    // single miss across 2000 calls is ~2000 * exp(-50) ~= 4e-19, so requiring
    // that every call spiked is safe from flaking.
    assert_eq!(
        batch_count, calls,
        "at rate*dt = 50 every batch call is near-certain to emit exactly one spike"
    );
    assert!(
        batch_hz >= 90.0,
        "batch count-rate {batch_hz} Hz must sit near the ~100 Hz ceiling from below, \
         not collapse toward zero"
    );

    // Empirical streaming decode over the same elapsed time recovers ~5 kHz.
    encoder.reset();
    let steps = 2_000u64;
    let mut stream_count = 0u64;
    for _ in 0..steps {
        stream_count += encoder.encode_step(&[1.0]).spikes.len() as u64;
    }
    let stream_hz = count_rate_hz(stream_count, steps, dt_f64);
    assert!(
        (stream_hz - 5_000.0).abs() <= 5.0,
        "streaming count-rate {stream_hz} Hz should recover ~5000 Hz"
    );

    // The whole point: batch is dramatically less than streaming, so a
    // count-rate decoder that only saw `encode` output would be silently wrong.
    assert!(
        stream_hz > batch_hz * 40.0,
        "streaming {stream_hz} Hz must dwarf batch {batch_hz} Hz at rate*dt = 50"
    );
}
