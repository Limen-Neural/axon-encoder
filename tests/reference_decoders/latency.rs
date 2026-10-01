//! Latency: linear TTFS inverse plus non-finite collision points.

use axon_encoder::prelude::*;

/// Reference inverse of [`LatencyEncoder`].
///
/// Forward (from `src/encoders/latency.rs`):
/// `offset = round((1 - normalized) * max_latency).min(max_latency)` where
/// `normalized = (clamp(value, min, max) - min) / (max - min)`.
///
/// Inverting the linear map (ignoring the integer rounding) gives
/// `x = min + (1 - ts / max_latency) * (max - min)`.
///
/// # Degenerate `max_latency == 0`
///
/// `LatencyEncoder::try_new(0, range)` is a documented-valid configuration: the
/// presentation window is `max_latency + 1 == 1` tick wide and *every* input
/// collapses onto tick `0` (see `latency_zero_max_latency_is_non_invertible`).
/// That map is non-invertible — the whole range maps to a single tick, so no
/// spike timestamp can recover which value produced it. Rather than silently
/// computing `1 - 0/0 = NaN`, the oracle documents this collision explicitly by
/// returning the range midpoint, the minimum-squared-error point estimate when
/// every value in `[min, max]` collapses to the same tick.
fn latency_decode(ts: u64, max_latency: u64, range: (f32, f32)) -> f64 {
    let (min, max) = (range.0 as f64, range.1 as f64);
    if max_latency == 0 {
        // Non-invertible collision: the entire range maps to tick 0. Report the
        // range midpoint as the defined (documented) degenerate answer instead
        // of a silent NaN from the 0/0 division below.
        return min + 0.5 * (max - min);
    }
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

#[test]
fn latency_zero_max_latency_is_non_invertible() {
    // `max_latency == 0` is a documented-valid config: the presentation window
    // is one tick wide (max_latency + 1) and every input, regardless of
    // magnitude or sign, collapses onto tick 0. The forward map is therefore
    // non-invertible — all values share a single tick, so a TTFS decoder cannot
    // recover the input from the timestamp.
    let range = (0.0f32, 1.0f32);
    let mut encoder = LatencyEncoder::try_new(0, range).expect("max_latency 0 is valid");

    // The declared window is a single tick.
    assert_eq!(encoder.time_model().span_ticks(), 1);

    // Every input — endpoints, interior, out-of-range, and non-finite — emits
    // exactly one spike at tick 0.
    for value in [
        range.0,
        range.1,
        0.5,
        -5.0,
        5.0,
        f32::NAN,
        f32::INFINITY,
        f32::NEG_INFINITY,
    ] {
        let out = encoder.encode(&[value]);
        assert_eq!(out.spikes.len(), 1, "input {value} emits one spike");
        assert_eq!(
            out.spikes[0].timestamp.ticks(),
            0,
            "input {value} collapses onto tick 0"
        );
    }

    // The oracle represents this collision explicitly rather than returning a
    // silent NaN from `1 - 0/0`: it reports the range midpoint (0.5 here), the
    // minimum-squared-error estimate when the whole range maps to one tick.
    let decoded = latency_decode(0, 0, range);
    assert!(
        decoded.is_finite(),
        "the degenerate decode must be finite, not NaN"
    );
    let midpoint = (range.0 as f64 + range.1 as f64) / 2.0;
    assert!(
        (decoded - midpoint).abs() < 1e-12,
        "degenerate decode {decoded} must be the range midpoint {midpoint}"
    );
}
