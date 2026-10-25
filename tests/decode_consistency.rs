//! Decoder contract: analog values are recoverable from the temporal and rate
//! codes, and the rules a downstream decoder must follow to do so (LIM-1294).
//!
//! `tests/time_semantics.rs` proves offsets stay inside `span_ticks` and are
//! channel-major; `tests/reference_decoders/` pins each encoder's inverse
//! formula and its collision points. This suite closes the gap between them:
//! it round-trips a documented probe grid through every value-carrying encoder
//! and locks the three things a decoder gets wrong in practice.
//!
//! 1. **Polarity.** Latency puts *strong* input at *early* ticks (`x̂ = 1 -
//!    t/L`); Phase puts *strong* input at *late* bins (`x̂ ≈ (b + 0.5)/C`). The
//!    two offsets correlate with the input with opposite sign (`≈ -1` vs `≈
//!    +1`), so a Phase decoder applied to Latency offsets is wrong.
//! 2. **Emission order ≠ time order.** Latency spikes come out channel-major,
//!    not by timestamp. A decoder that treats emission order as temporal order
//!    (e.g. "first spike wins") must sort by `timestamp` first.
//! 3. **Overlapping windows.** Phase offsets are call-relative; converting them
//!    to absolute time through a [`TimeCursor`] can reorder spikes across
//!    calls. The absolute chronology is not the emission order.
//!
//! The crate ships no `Decoder` trait. The inverse formulas come from
//! `tests/common/oracles.rs`, the single definition also used by
//! `tests/reference_decoders/`; the count-weighted center of mass below is
//! this suite's own, since the reference suite weights by analytic tuning
//! response rather than observed spike counts. README "Decoder contract"
//! points here.

use axon_encoder::prelude::*;

/// Probe grid shared by every round-trip test: both range endpoints, the
/// midpoint, interior tenths, and one off-grid value (`0.37`) that no
/// configuration below quantizes exactly, so a rounding regression cannot hide
/// behind a grid that happens to divide evenly.
const PROBES: [f32; 8] = [0.0, 0.1, 0.3, 0.37, 0.5, 0.7, 0.9, 1.0];
const RANGE: (f32, f32) = (0.0, 1.0);

// --- oracles -----------------------------------------------------------------

mod common;
use common::oracles::population_preferred;
use common::oracles::{latency_decode as latency_decode_in, phase_decode as phase_decode_in};

fn latency_decode(offset: u64, max_latency: u64) -> f64 {
    latency_decode_in(offset, max_latency, RANGE)
}

fn phase_decode(bin: u64, cycle_steps: u64) -> f64 {
    phase_decode_in(bin, cycle_steps, RANGE)
}

fn preferred(i: usize, n: usize) -> f64 {
    population_preferred(i, n, RANGE)
}

/// Population center of mass over pooled spike counts.
fn population_com(counts: &[u64]) -> f64 {
    let n = counts.len();
    let (mut weighted, mut total) = (0.0f64, 0.0f64);
    for (i, &count) in counts.iter().enumerate() {
        weighted += preferred(i, n) * count as f64;
        total += count as f64;
    }
    assert!(total > 0.0, "population_com: no spikes to decode");
    weighted / total
}

/// Pearson correlation of two equally long series.
fn correlation(xs: &[f64], ys: &[f64]) -> f64 {
    let n = xs.len() as f64;
    let (mx, my) = (xs.iter().sum::<f64>() / n, ys.iter().sum::<f64>() / n);
    let cov: f64 = xs.iter().zip(ys).map(|(x, y)| (x - mx) * (y - my)).sum();
    let vx: f64 = xs.iter().map(|x| (x - mx).powi(2)).sum();
    let vy: f64 = ys.iter().map(|y| (y - my).powi(2)).sum();
    let r = cov / (vx * vy).sqrt();
    assert!(
        r.is_finite(),
        "correlation: degenerate series (vx={vx}, vy={vy})"
    );
    r
}

/// One round-trip check over [`PROBES`]: encode the grid once with `encoder`,
/// decode every offset with `decode`, and require each probe back within
/// `tolerance`.
fn assert_round_trip(
    label: &str,
    encoder: &mut dyn Encoder,
    decode: &dyn Fn(u64) -> f64,
    tolerance: f64,
) {
    let spikes = encoder.encode(&PROBES).spikes;
    assert_eq!(spikes.len(), PROBES.len(), "{label}: one spike per probe");
    for (spike, &x) in spikes.iter().zip(PROBES.iter()) {
        let decoded = decode(spike.timestamp.ticks());
        assert!(
            (decoded - f64::from(x)).abs() <= tolerance + 1e-9,
            "{label}: x={x}: offset {} decodes to {decoded}, more than {tolerance} off",
            spike.timestamp
        );
    }
}

// --- Latency -----------------------------------------------------------------

#[test]
fn latency_round_trips_the_probe_grid_within_half_a_tick() {
    let mut encoder = LatencyEncoder::try_new(10, RANGE).expect("valid latency");
    assert_round_trip("latency", &mut encoder, &|t| latency_decode(t, 10), 0.05);
}

/// Values that quantize to a tick exactly decode back to themselves.
#[test]
fn latency_is_exact_on_its_own_tick_grid() {
    let max_latency = 10u64;
    let mut encoder = LatencyEncoder::try_new(max_latency, RANGE).expect("valid latency");
    let grid: Vec<f32> = (0..=max_latency).map(|t| t as f32 / 10.0).collect();
    let out = encoder.encode(&grid);
    assert_eq!(out.spikes.len(), grid.len(), "one spike per grid point");
    for (spike, &x) in out.spikes.iter().zip(grid.iter()) {
        let decoded = latency_decode(spike.timestamp.ticks(), max_latency) as f32;
        assert!(
            (decoded - x).abs() <= 1e-6,
            "x={x}: {} -> {decoded}",
            spike.timestamp
        );
    }
}

/// Emission order is channel-major, not time order. A decoder that reads
/// "earliest spike" must sort by timestamp first: for input where channel
/// order and value order disagree, the first *emitted* spike is not the first
/// *in time*.
#[test]
fn latency_decoder_must_sort_by_timestamp_not_emission_order() {
    let max_latency = 10u64;
    let mut encoder = LatencyEncoder::try_new(max_latency, RANGE).expect("valid latency");
    // Channel 0 is weak (late), channel 2 is strong (early).
    let input = [0.1f32, 0.5, 0.9];
    let spikes = encoder.encode(&input).spikes;

    let emitted_channels: Vec<u16> = spikes.iter().map(|s| s.channel).collect();
    assert_eq!(emitted_channels, vec![0, 1, 2], "emission is channel-major");

    let mut by_time = spikes.clone();
    by_time.sort_by_key(|s| (s.timestamp.ticks(), s.channel));
    let time_order: Vec<u16> = by_time.iter().map(|s| s.channel).collect();
    assert_eq!(time_order, vec![2, 1, 0], "time order is strongest first");
    assert_ne!(
        emitted_channels, time_order,
        "emission order and time order differ, so sorting is mandatory"
    );

    // Decoding in time order recovers the values in descending order.
    let decoded: Vec<f64> = by_time
        .iter()
        .map(|s| latency_decode(s.timestamp.ticks(), max_latency))
        .collect();
    assert!(decoded.windows(2).all(|w| w[0] >= w[1]), "{decoded:?}");
}

// --- Phase ---------------------------------------------------------------------

#[test]
fn phase_round_trips_the_probe_grid_within_half_a_bin() {
    let mut encoder = PhaseEncoder::try_new(16, RANGE).expect("valid phase");
    assert_round_trip("phase", &mut encoder, &|b| phase_decode(b, 16), 0.5 / 16.0);
}

/// Phase and Latency have opposite polarity: the offset grows with the input
/// for Phase and shrinks with it for Latency. Applying the Phase inverse to
/// Latency offsets (or vice versa) without flipping produces a decode that is
/// anti-correlated with the truth.
#[test]
fn phase_decoder_applied_to_latency_offsets_is_wrong_without_a_polarity_flip() {
    let (max_latency, cycle_steps) = (10u64, 10u64);
    let mut latency = LatencyEncoder::try_new(max_latency, RANGE).expect("valid latency");
    let mut phase = PhaseEncoder::try_new(cycle_steps, RANGE).expect("valid phase");
    let truth: Vec<f64> = PROBES.iter().map(|&x| f64::from(x)).collect();

    let latency_offsets: Vec<f64> = latency
        .encode(&PROBES)
        .spikes
        .iter()
        .map(|s| s.timestamp.ticks() as f64)
        .collect();
    let phase_offsets: Vec<f64> = phase
        .encode(&PROBES)
        .spikes
        .iter()
        .map(|s| s.timestamp.ticks() as f64)
        .collect();

    let latency_corr = correlation(&truth, &latency_offsets);
    let phase_corr = correlation(&truth, &phase_offsets);
    assert!(
        latency_corr < -0.99,
        "latency offsets vs input: {latency_corr}"
    );
    assert!(phase_corr > 0.99, "phase offsets vs input: {phase_corr}");

    // The wrong decoder: Phase inverse on Latency offsets.
    let misdecoded: Vec<f64> = latency_offsets
        .iter()
        .map(|&t| phase_decode(t as u64, cycle_steps))
        .collect();
    assert!(
        correlation(&truth, &misdecoded) < -0.99,
        "a Phase decoder on Latency offsets is anti-correlated with the truth"
    );
    // Flipping the polarity (t -> L - t) repairs it.
    let repaired: Vec<f64> = latency_offsets
        .iter()
        .map(|&t| phase_decode(max_latency - t as u64, cycle_steps))
        .collect();
    assert!(correlation(&truth, &repaired) > 0.99, "{repaired:?}");
}

/// Phase offsets are call-relative inside an overlapping window. Converting
/// them to absolute ticks with a `TimeCursor` yields a chronology that differs
/// from the emission order, so a decoder must not read emission order as time.
#[test]
fn phase_overlapping_windows_absolute_chronology_differs_from_emission_order() {
    let cycle_steps = 8u64;
    let mut encoder = PhaseEncoder::try_new(cycle_steps, RANGE).expect("valid phase");
    let mut cursor = TimeCursor::new(encoder.time_model());

    let mut emission_offsets = Vec::new();
    let mut absolute = Vec::new();
    for &x in &[0.9f32, 0.1, 0.9, 0.1] {
        let out = encoder.encode_step(&[x]);
        assert_eq!(out.spikes.len(), 1);
        let spike = out.spikes[0];
        emission_offsets.push(spike.timestamp.ticks());
        absolute.push(cursor.absolute(spike.timestamp));
        cursor.advance();
    }
    assert_eq!(emission_offsets, vec![7, 0, 7, 0], "call-relative offsets");
    assert_eq!(absolute, vec![7, 1, 9, 3], "absolute = origin + offset");

    // Emission order (call index) and absolute chronology disagree.
    let mut chronological: Vec<usize> = (0..absolute.len()).collect();
    chronological.sort_by_key(|&i| absolute[i]);
    assert_eq!(chronological, vec![1, 3, 0, 2]);
    assert_ne!(chronological, vec![0, 1, 2, 3]);
    assert!(
        absolute.windows(2).any(|w| w[0] > w[1]),
        "absolute times are not monotone in emission order: {absolute:?}"
    );
}

// --- Rate (streaming) ----------------------------------------------------------

/// Streaming rate: `count / elapsed` recovers the configured rate to within one
/// spike over the elapsed window, on the whole probe grid.
#[test]
fn streaming_rate_count_over_time_recovers_the_probe_grid() {
    let (base_hz, max_hz, dt) = (0.0f32, 100.0f32, 0.01f32);
    let steps = 500u64;
    let mut encoder = RateEncoder::try_new(base_hz, max_hz, RANGE, dt).expect("valid rate");
    let mut counts = vec![0u64; PROBES.len()];
    for _ in 0..steps {
        for spike in encoder.encode_step(&PROBES).spikes {
            counts[usize::from(spike.channel)] += 1;
        }
    }
    let elapsed = f64::from(dt) * steps as f64;
    for (i, &x) in PROBES.iter().enumerate() {
        let expected_hz = f64::from(base_hz) + f64::from(x) * f64::from(max_hz - base_hz);
        let decoded_hz = counts[i] as f64 / elapsed;
        assert!(
            (decoded_hz - expected_hz).abs() <= 1.0 / elapsed + 1e-9,
            "x={x}: {decoded_hz} Hz vs {expected_hz} Hz"
        );
        // Back to the analog value.
        let decoded_x = (decoded_hz - f64::from(base_hz)) / f64::from(max_hz - base_hz);
        assert!(
            (decoded_x - f64::from(x)).abs() <= 1.0 / (elapsed * f64::from(max_hz)) + 1e-9,
            "x={x}: decoded {decoded_x}"
        );
    }
}

// --- Population ------------------------------------------------------------------

/// Population center of mass over pooled stochastic spikes recovers interior
/// probe values to within a quarter of the neuron spacing; endpoints are
/// biased inward (documented in `tests/reference_decoders/population.rs`), so
/// only the sign of their error is asserted.
#[test]
fn population_center_of_mass_recovers_the_probe_grid() {
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    let (n, width) = (11usize, 0.1f32);
    let mut encoder = PopulationEncoder::try_new(n, RANGE, width).expect("valid population");
    let mut rng = StdRng::seed_from_u64(0x1294);
    let spacing = 1.0 / (n - 1) as f64;
    let trials = 2_000;

    for &x in &PROBES {
        let mut counts = vec![0u64; n];
        for _ in 0..trials {
            for spike in encoder.encode_with_rng(&[x], &mut rng).spikes {
                counts[usize::from(spike.channel)] += 1;
            }
        }
        let decoded = population_com(&counts);
        let truth = f64::from(x);
        if x == RANGE.0 {
            assert!(
                decoded > truth,
                "x={x}: CoM {decoded} must be biased inward"
            );
        } else if x == RANGE.1 {
            assert!(
                decoded < truth,
                "x={x}: CoM {decoded} must be biased inward"
            );
        } else {
            assert!(
                (decoded - truth).abs() <= 0.25 * spacing,
                "x={x}: CoM {decoded}"
            );
        }
    }
}
