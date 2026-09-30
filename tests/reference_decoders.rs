//! In-tree inverse oracles for the latency, phase, rate, and population
//! encoders (LIM-1340, parent LIM-1265).
//!
//! The crate ships **no production decoder**. Downstream count-rate, TTFS
//! (time-to-first-spike), and center-of-mass consumers therefore have no shared
//! reference for "what value produced this spike train". This file provides the
//! inverses used *only as test oracles*: they let LIM-1260's decoder-consistency
//! audit cite these numbers instead of re-deriving them, and they pin the
//! properties a real decoder must respect (collision points, quantization
//! bounds, the batch-versus-stream rate cap, and the population endpoint bias).
//!
//! Every oracle here is derived from the encoder source at `main` `fca28fa`,
//! not from the issue's `f3e375f` measurement table. Two things in that table
//! are stale and are corrected below:
//!
//!   * Population preferred values now cover both range endpoints via
//!     `preferred[i] = min + (i / (num_neurons - 1)) * span` (PR #101), not the
//!     old `i / num_neurons` grid. So for `N = 8` over `(0, 1)`, neuron 7 sits
//!     at `1.0`, not `0.875`.
//!   * The durable population invariant is therefore *not* a hardcoded
//!     `0.951 / 0.875` figure but the endpoint bias: the center-of-mass decode
//!     at `x = max` is strictly less than `max` for `N > 1`, because the
//!     Gaussian neighbors all sit below the maximum and pull the CoM inward.
//!
//! There is intentionally **no public decoder API**: these inverses live in the
//! test tree until a later 0.5 design explicitly asks for one.

use axon_encoder::prelude::*;

// ---------------------------------------------------------------------------
// 1. Latency: linear TTFS inverse plus non-finite collision points.
// ---------------------------------------------------------------------------

/// Reference inverse of [`LatencyEncoder`].
///
/// Forward (from `src/encoders/latency.rs`):
/// `offset = round((1 - normalized) * max_latency).min(max_latency)` where
/// `normalized = (clamp(value, min, max) - min) / (max - min)`.
///
/// Inverting the linear map (ignoring the integer rounding) gives
/// `x = min + (1 - ts / max_latency) * (max - min)`.
fn latency_decode(ts: u64, max_latency: u64, range: (f32, f32)) -> f64 {
    let (min, max) = (range.0 as f64, range.1 as f64);
    let fraction = 1.0 - (ts as f64 / max_latency as f64);
    min + fraction * (max - min)
}

#[test]
fn latency_linear_inverse_is_exact_on_the_grid() {
    // With max_latency = 20 over (0, 1), every probe of the form i/20 lands on a
    // tick with zero rounding error, so the inverse reconstructs it exactly.
    let max_latency = 20u64;
    let range = (0.0f32, 1.0f32);
    let mut encoder = LatencyEncoder::try_new(max_latency, range).expect("valid latency");

    // Compare against the exact f64 grid point i/max_latency: the forward map
    // rounds cleanly on this grid, so the inverse reconstructs it with zero
    // error. (The f32 probe value itself carries representation error, so the
    // oracle is checked against the ideal grid point, which is what the issue's
    // "max abs err = 0" row measures.)
    let mut max_abs_err = 0.0f64;
    for i in 0..=max_latency {
        let grid = i as f64 / max_latency as f64;
        let value = grid as f32;
        let out = encoder.encode(&[value]);
        assert_eq!(out.spikes.len(), 1, "one spike per channel");
        let ts = out.spikes[0].timestamp.ticks();
        // Each grid probe should land on tick max_latency - i.
        assert_eq!(
            ts,
            max_latency - i,
            "grid probe {i} lands on the expected tick"
        );
        let decoded = latency_decode(ts, max_latency, range);
        max_abs_err = max_abs_err.max((decoded - grid).abs());
    }
    // The forward map lands each grid probe on its exact tick (asserted above),
    // so the only residual is f64 arithmetic rounding in the inverse itself,
    // bounded by machine epsilon. This is the "max abs err = 0" property from
    // the issue table for a linear TTFS inverse.
    assert!(
        max_abs_err <= f64::EPSILON,
        "grid probes reconstruct exactly (residual {max_abs_err} exceeds f64 epsilon)"
    );
}

#[test]
fn latency_golden_vectors_endpoints_and_interior() {
    // Golden (value -> timestamp) vectors for both endpoints and one interior
    // point. Strong (max) fires first at ts 0; weak (min) fires last at
    // max_latency.
    let max_latency = 20u64;
    let range = (0.0f32, 1.0f32);
    let mut encoder = LatencyEncoder::try_new(max_latency, range).expect("valid latency");

    // (input, expected timestamp)
    let golden = [
        (0.0f32, 20u64), // range min  -> latest spike
        (1.0f32, 0u64),  // range max  -> earliest spike
        (0.25f32, 15u64),
        (0.5f32, 10u64),
        (0.75f32, 5u64),
    ];
    for (value, expected_ts) in golden {
        let out = encoder.encode(&[value]);
        assert_eq!(
            out.spikes[0].timestamp.ticks(),
            expected_ts,
            "latency forward golden vector for {value}"
        );
        // And the inverse round-trips exactly on these grid points.
        let decoded = latency_decode(expected_ts, max_latency, range);
        assert!(
            (decoded - value as f64).abs() < 1e-12,
            "latency inverse for {value} decoded to {decoded}"
        );
    }

    // Spikes are emitted in channel order regardless of timing.
    let out = encoder.encode(&[0.0, 1.0]);
    assert_eq!(out.spikes[0].channel, 0);
    assert_eq!(out.spikes[1].channel, 1);
    assert_eq!(out.spikes[0].timestamp.ticks(), 20);
    assert_eq!(out.spikes[1].timestamp.ticks(), 0);

    // The presentation window is max_latency + 1 ticks wide.
    assert_eq!(encoder.time_model().span_ticks(), max_latency + 1);
}

#[test]
fn latency_non_finite_inputs_collide_with_the_endpoints() {
    // NaN and -inf both land on the same tick as the range minimum, and +inf on
    // the same tick as the range maximum. A TTFS decoder therefore *cannot*
    // distinguish these from a genuine endpoint reading: the collisions are the
    // point of this oracle.
    let max_latency = 20u64;
    let range = (0.0f32, 1.0f32);
    let mut encoder = LatencyEncoder::try_new(max_latency, range).expect("valid latency");

    let min_ts = encoder.encode(&[range.0]).spikes[0].timestamp.ticks();
    let max_ts = encoder.encode(&[range.1]).spikes[0].timestamp.ticks();
    assert_eq!(min_ts, max_latency);
    assert_eq!(max_ts, 0);

    // NaN and -inf collide with the minimum (latest tick).
    for value in [f32::NAN, f32::NEG_INFINITY] {
        let ts = encoder.encode(&[value]).spikes[0].timestamp.ticks();
        assert_eq!(ts, min_ts, "input {value} must collide with range min");
        // Decoding the colliding tick yields the range minimum.
        assert!((latency_decode(ts, max_latency, range) - range.0 as f64).abs() < 1e-12);
    }

    // +inf clamps to the maximum (earliest tick).
    let plus_inf_ts = encoder.encode(&[f32::INFINITY]).spikes[0].timestamp.ticks();
    assert_eq!(plus_inf_ts, max_ts, "+inf must collide with range max");
    assert!((latency_decode(plus_inf_ts, max_latency, range) - range.1 as f64).abs() < 1e-12);
}

// ---------------------------------------------------------------------------
// 2. Phase: bin-center inverse plus non-finite channel drop.
// ---------------------------------------------------------------------------

/// Reference inverse of [`PhaseEncoder`].
///
/// Forward (from `src/encoders/phase.rs`):
/// `bin = floor(normalized * cycle_steps).min(cycle_steps - 1)`.
///
/// The best point estimate for a quantized bin is its center, so decode bin `b`
/// to `x = min + ((b + 0.5) / cycle_steps) * (max - min)`. The reconstruction
/// error is then bounded by half a bin width.
fn phase_decode(bin: u64, cycle_steps: u64, range: (f32, f32)) -> f64 {
    let (min, max) = (range.0 as f64, range.1 as f64);
    min + ((bin as f64 + 0.5) / cycle_steps as f64) * (max - min)
}

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

// ---------------------------------------------------------------------------
// 3. Rate: streaming count/time recovers rate_hz; batch is capped by the
//    Bernoulli per-call ceiling and can never recover Hz when rate*dt >= 1.
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// 4. Population: preferred-value grid and center-of-mass endpoint bias.
// ---------------------------------------------------------------------------

/// The preferred value of neuron `i` in an `n`-neuron population over `range`.
///
/// This mirrors the CURRENT source (`src/encoders/population.rs`): neuron 0 is
/// pinned to `min`, neuron `n - 1` to `max`, and the interior neurons sit on the
/// endpoint-covering grid `min + (i / (n - 1)) * span`. This replaces the stale
/// `i / n` grid in the LIM-1340 issue table (which put neuron 7 of 8 at 0.875);
/// on the current grid neuron 7 of 8 sits at `1.0`.
fn population_preferred(i: usize, n: usize, range: (f32, f32)) -> f64 {
    let (min, max) = (range.0 as f64, range.1 as f64);
    if n == 1 || i == 0 {
        min
    } else if i == n - 1 {
        max
    } else {
        min + (i as f64 / (n - 1) as f64) * (max - min)
    }
}

/// Gaussian tuning response of neuron `i` to `input`, matching the encoder's
/// `get_rate_with_tuning_width`.
fn population_response(input: f64, i: usize, n: usize, range: (f32, f32), width: f64) -> f64 {
    let preferred = population_preferred(i, n, range);
    let distance = (input - preferred).abs();
    (-(distance * distance) / (2.0 * width * width)).exp()
}

/// Pooled center-of-mass decode: the tuning-weighted mean of preferred values.
fn population_center_of_mass(input: f64, n: usize, range: (f32, f32), width: f64) -> f64 {
    let mut weighted = 0.0f64;
    let mut total = 0.0f64;
    for i in 0..n {
        let response = population_response(input, i, n, range, width);
        weighted += population_preferred(i, n, range) * response;
        total += response;
    }
    weighted / total
}

#[test]
fn population_preferred_grid_covers_both_endpoints() {
    // Document and pin the ACTUAL preferred-value formula: i / (n - 1), not the
    // stale i / n from the issue table. For N = 8 over (0, 1), neuron 7 sits at
    // 1.0 (not 0.875), and neuron 0 sits at 0.0.
    let n = 8usize;
    let range = (0.0f32, 1.0f32);

    assert_eq!(population_preferred(0, n, range), 0.0);
    assert_eq!(population_preferred(n - 1, n, range), 1.0);
    // Interior neuron on the i/(n-1) grid.
    assert!((population_preferred(4, n, range) - 4.0 / 7.0).abs() < 1e-12);

    // The stale i/N grid would have put neuron 7 at 0.875; the current grid does
    // not. This guards against regressing to the old formula.
    assert!((population_preferred(7, n, range) - 0.875).abs() > 0.1);
}

#[test]
fn population_peak_firer_pins_the_encoder_to_the_endpoint_grid() {
    // Regression guard that OBSERVES THE PRODUCTION ENCODER, not just the copied
    // helper: drive `PopulationEncoder::encode` thousands of times and assert
    // that the empirically most-active neuron matches the neuron the current
    // `i / (n - 1)` grid predicts. This is what ties the oracle to real output
    // so a grid change breaks the test.
    //
    // Setup: N = 10 over (0, 100), width 10, probing at x = 800 / 9, which is
    // exactly neuron 8's preferred value on the CURRENT grid. Two things follow:
    //
    //   * On the current `i / (n - 1)` grid, preferred[8] == x, so neuron 8
    //     fires with probability exp(0) = 1.0 on EVERY call (a deterministic
    //     anchor), while its nearest competitors (neurons 7 and 9) fire at only
    //     ~0.54. Neuron 8 is therefore the unambiguous highest-frequency firer.
    //   * On the OLD `i / n` grid the same neurons would sit at 80 and 90, so
    //     the closest neuron to x = 88.9 would be neuron 9 (pref 90), not neuron
    //     8. The predicted peak differs between the two grids, so reverting the
    //     encoder to `i / n` moves the observed peak and fails this test.
    let n = 10usize;
    let range = (0.0f32, 100.0f32);
    let width = 10.0f32;
    let probe = 800.0f32 / 9.0; // == population_preferred(8, ...) on the current grid.
    let mut encoder = PopulationEncoder::try_new(n, range, width).expect("valid population");

    // The neuron whose preferred value is closest to the probe under the
    // current grid. Computed from the oracle helper, not hardcoded.
    let predicted_peak = (0..n)
        .min_by(|&a, &b| {
            let da = (population_preferred(a, n, range) - probe as f64).abs();
            let db = (population_preferred(b, n, range) - probe as f64).abs();
            da.partial_cmp(&db).unwrap()
        })
        .unwrap();
    assert_eq!(
        predicted_peak, 8,
        "current grid predicts neuron 8 as the peak"
    );

    // Empirical per-neuron firing counts over many independent encode calls.
    let iterations = 8_000u64;
    let mut counts = vec![0u64; n];
    for _ in 0..iterations {
        for spike in encoder.encode(&[probe]).spikes {
            counts[spike.channel as usize] += 1;
        }
    }

    // Neuron 8 fires with probability 1.0 on the current grid, so it must be
    // present on every call.
    assert_eq!(
        counts[predicted_peak], iterations,
        "neuron {predicted_peak} sits exactly on the probe and must fire every call"
    );

    // The empirically most-active neuron must be the one the current grid
    // predicts. On the old i/n grid the peak would land elsewhere (neuron 9),
    // so this assertion actually distinguishes the two grids.
    let observed_peak = (0..n).max_by_key(|&i| counts[i]).unwrap();
    assert_eq!(
        observed_peak, predicted_peak,
        "observed peak neuron {observed_peak} must match the i/(n-1) grid prediction {predicted_peak}; \
         counts = {counts:?}"
    );

    // The peak must be a strict, comfortable margin above its nearest rival so
    // sampling noise cannot flip the winner: neuron 8 fires ~2x as often as the
    // ~0.54-probability neurons 7 and 9.
    let runner_up = (0..n)
        .filter(|&i| i != observed_peak)
        .map(|i| counts[i])
        .max()
        .unwrap();
    assert!(
        counts[observed_peak] > runner_up + iterations / 4,
        "peak neuron {observed_peak} ({}) must dominate its runner-up ({runner_up}) by a wide margin",
        counts[observed_peak]
    );
}

#[test]
fn population_center_of_mass_is_biased_inward_at_the_maximum() {
    // Durable invariant (replaces the stale 0.951/0.875 numbers): for N > 1 the
    // center-of-mass decode at x = range max is STRICTLY LESS THAN max, because
    // every neuron below the top pulls the pooled mean down. This is the
    // endpoint-bias property a real decoder must account for.
    let range = (0.0f32, 1.0f32);
    let width = 0.2f64;

    for n in [2usize, 4, 8, 16] {
        let com_at_max = population_center_of_mass(range.1 as f64, n, range, width);
        assert!(
            com_at_max < range.1 as f64,
            "N={n}: CoM at max {com_at_max} must be strictly below {}",
            range.1
        );
        // Symmetrically, the CoM at the minimum is biased upward (strictly
        // above min), confirming the bias is an endpoint effect, not a global
        // offset.
        let com_at_min = population_center_of_mass(range.0 as f64, n, range, width);
        assert!(
            com_at_min > range.0 as f64,
            "N={n}: CoM at min {com_at_min} must be strictly above {}",
            range.0
        );
    }

    // Interior points are reconstructed with far smaller error than the
    // endpoints: the bias is specific to the range edges.
    let n = 8usize;
    let interior = 0.5f64;
    let com_interior = population_center_of_mass(interior, n, range, width);
    assert!(
        (com_interior - interior).abs() < 0.05,
        "interior CoM {com_interior} should closely track {interior}"
    );
}

#[test]
fn population_fires_at_tick_zero_and_channels_map_to_neurons() {
    // Population spikes all land at TickOffset::ZERO, and for input channel j
    // neuron i emits on spike channel j * num_neurons + i. An empirical CoM over
    // firing neurons therefore maps channel -> neuron index -> preferred value.
    let n = 8usize;
    let range = (0.0f32, 1.0f32);
    let width = 0.2f32;
    let mut encoder = PopulationEncoder::try_new(n, range, width).expect("valid population");

    // Single input channel: all spikes are at tick zero and channels are in
    // 0..n mapping directly to neuron indices. Probe exactly at neuron 4's
    // preferred value (4 / 7 on the current grid) so that neuron fires with
    // probability exp(0) = 1.0 and the output is guaranteed non-empty. A bare
    // `encode(&[0.5])` could emit zero spikes (~1 in 2400 runs) because every
    // neuron independently misses on one unseeded-RNG call; anchoring on a
    // deterministic neuron removes that flake without weakening the tick-zero
    // or channel-mapping checks below.
    let deterministic_neuron = 4usize;
    let probe = (deterministic_neuron as f32) / ((n - 1) as f32); // == preferred[4].
    let out = encoder.encode(&[probe]);
    assert!(
        out.spikes
            .iter()
            .any(|s| s.channel as usize == deterministic_neuron),
        "the neuron sitting on the probe (prob 1.0) must always fire"
    );
    assert!(!out.spikes.is_empty());
    assert!(
        out.spikes.iter().all(|s| s.timestamp == TickOffset::ZERO),
        "all population spikes land at tick zero"
    );
    assert!(out.spikes.iter().all(|s| (s.channel as usize) < n));

    // Empirical pooled CoM at x = max, averaged over many calls, is strictly
    // below max (the same inward bias as the analytic oracle).
    let width_f64 = width as f64;
    let mut weighted = 0.0f64;
    let mut spikes = 0u64;
    for _ in 0..5_000 {
        for spike in encoder.encode(&[range.1]).spikes {
            let neuron = spike.channel as usize;
            weighted += population_preferred(neuron, n, range);
            spikes += 1;
        }
    }
    assert!(spikes > 0);
    let empirical_com = weighted / spikes as f64;
    assert!(
        empirical_com < range.1 as f64,
        "empirical CoM at max {empirical_com} must be below {}",
        range.1
    );

    // Two input channels: the second population lives on channels [n, 2n).
    let out = encoder.encode(&[0.0, 1.0]);
    assert!(
        out.spikes
            .iter()
            .any(|s| (s.channel as usize) >= n && (s.channel as usize) < 2 * n),
        "input channel 1 must emit on the second population's channel block"
    );

    // The analytic oracle stays consistent with the width the encoder uses.
    let analytic_com = population_center_of_mass(range.1 as f64, n, range, width_f64);
    assert!(analytic_com < range.1 as f64);
}
