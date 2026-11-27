//! Directional population: preferred-angle metadata and population-vector decoding.

use axon_encoder::prelude::*;
use rand::SeedableRng;
use rand::rngs::StdRng;

fn circular_difference(a: f64, b: f64) -> f64 {
    let wrapped = (a - b).rem_euclid(std::f64::consts::TAU);
    wrapped.min(std::f64::consts::TAU - wrapped)
}

/// Population-vector inverse for one direction population.
///
/// Each spike count weights the unit vector at that neuron's preferred angle.
/// The resultant direction is `atan2(y, x)`, wrapped back into `[0, 2π)`. This
/// is deliberately circular: linearly averaging angles near zero would decode
/// directions on opposite sides of the wrap as pointing near π.
fn population_vector_decode(counts: &[u64], preferred_angles: &[f32]) -> f64 {
    assert_eq!(counts.len(), preferred_angles.len());
    let mut x = 0.0f64;
    let mut y = 0.0f64;
    for (&count, &angle) in counts.iter().zip(preferred_angles) {
        x += count as f64 * f64::from(angle).cos();
        y += count as f64 * f64::from(angle).sin();
    }
    assert!(x != 0.0 || y != 0.0, "no spikes to decode");
    y.atan2(x).rem_euclid(std::f64::consts::TAU)
}

fn empirical_counts(
    encoder: &mut DirectionalPopulationEncoder,
    input: f32,
    iterations: usize,
    seed: u64,
) -> Vec<u64> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut counts = vec![0u64; encoder.num_neurons()];
    for _ in 0..iterations {
        for spike in encoder.encode_step_with_rng(&[input], &mut rng).spikes {
            counts[spike.channel as usize] += 1;
        }
    }
    counts
}

#[test]
fn directional_population_vector_reconstructs_probe_directions() {
    let mut encoder = DirectionalPopulationEncoder::new(16, 0.35);
    let angles = encoder.preferred_angles_radians();

    for (probe, seed) in [
        (0.2f32, 0xD11E_0001),
        (1.3, 0xD11E_0002),
        (2.9, 0xD11E_0003),
        (std::f32::consts::TAU - 0.2, 0xD11E_0004),
    ] {
        let counts = empirical_counts(&mut encoder, probe, 8_000, seed);
        let decoded = population_vector_decode(&counts, &angles);
        let error = circular_difference(decoded, f64::from(probe));
        assert!(
            error <= 0.08,
            "probe {probe}: population-vector decode {decoded} has circular error {error}; counts={counts:?}"
        );
    }
}

#[test]
fn directional_population_vector_decode_wraps_across_zero() {
    let mut encoder = DirectionalPopulationEncoder::new(16, 0.35);
    let angles = encoder.preferred_angles_radians();
    // Two representations of the same probe just east of zero. If either side
    // of the wrap were averaged linearly, the result would point backwards near π.
    for probe in [
        0.03f32,
        std::f32::consts::TAU + 0.03,
        -std::f32::consts::TAU + 0.03,
    ] {
        let counts = empirical_counts(&mut encoder, probe, 8_000, 0xD11E_DEAD);
        let decoded = population_vector_decode(&counts, &angles);
        let error = circular_difference(decoded, 0.03);
        assert!(
            error <= 0.08,
            "wrapped probe {probe}: decode {decoded} has circular error {error}"
        );
    }
}

#[test]
fn directional_population_preferred_angle_export_matches_neuron_ids() {
    let encoder = DirectionalPopulationEncoder::new(16, 0.35);
    let angles = encoder.preferred_angles_radians();

    assert_eq!(angles.len(), encoder.num_neurons());
    assert!(angles.iter().all(|angle| angle.is_finite()));
    assert!(
        angles
            .iter()
            .all(|&angle| (0.0..std::f32::consts::TAU).contains(&angle))
    );
    for (neuron_id, &angle) in angles.iter().enumerate() {
        let expected = neuron_id as f32 * std::f32::consts::TAU / encoder.num_neurons() as f32;
        assert_eq!(angle, expected, "neuron id {neuron_id} angle mismatch");
    }
}
