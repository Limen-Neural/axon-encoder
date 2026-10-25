//! Reference inverse formulas shared by `tests/reference_decoders/` (which
//! pins each formula's collision points) and `tests/decode_consistency.rs`
//! (which round-trips a probe grid through them). One definition each.
//!
//! These are **test oracles**, not a public decoder: the crate deliberately
//! ships no `Decoder` trait.

#![allow(dead_code)]

/// Latency inverse: `x̂ = min + (1 - t / L) · span` with `L = max_latency`.
///
/// Forward (`src/encoders/latency.rs`):
/// `offset = round((1 - normalized) * max_latency).min(max_latency)`.
///
/// `max_latency == 0` is a valid configuration whose window is one tick wide,
/// so every input collapses onto tick `0` and the map is non-invertible.
/// Rather than return `NaN` from `1 - 0/0`, the oracle reports the range
/// midpoint, the minimum-squared-error estimate when the whole range maps to
/// one tick.
pub fn latency_decode(ts: u64, max_latency: u64, range: (f32, f32)) -> f64 {
    let (min, max) = (f64::from(range.0), f64::from(range.1));
    if max_latency == 0 {
        return min + 0.5 * (max - min);
    }
    let fraction = 1.0 - (ts as f64 / max_latency as f64);
    min + fraction * (max - min)
}

/// Phase inverse: bin center, `x̂ = min + ((b + 0.5) / C) · span`.
///
/// Forward (`src/encoders/phase.rs`):
/// `bin = floor(normalized * cycle_steps).min(cycle_steps - 1)`.
/// The reconstruction error is bounded by half a bin while `1 / cycle_steps`
/// is resolvable in f64.
pub fn phase_decode(bin: u64, cycle_steps: u64, range: (f32, f32)) -> f64 {
    let (min, max) = (f64::from(range.0), f64::from(range.1));
    min + ((bin as f64 + 0.5) / cycle_steps as f64) * (max - min)
}

/// Population preferred value of neuron `i` of `n` on the endpoint-covering
/// grid, computed in f32 with the encoder's operation order
/// (`range.0 + (i / (n - 1)) * span`) and then widened.
pub fn population_preferred(i: usize, n: usize, range: (f32, f32)) -> f64 {
    let (min, max) = range;
    if n == 1 || i == 0 {
        f64::from(min)
    } else if i == n - 1 {
        f64::from(max)
    } else {
        let span = max - min;
        f64::from(min + (i as f32 / (n - 1) as f32) * span)
    }
}
