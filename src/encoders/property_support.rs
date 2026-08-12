//! Shared scaffolding for encoder property-style tests (#69 / LIM-1016).
//!
//! Seeded samplers keep failures reproducible without a proptest dependency.

#![cfg(test)]

use crate::types::SpikeEvent;
use rand::RngExt;
use rand::rngs::StdRng;

/// Default number of trials per property suite.
pub const TRIALS: usize = 256;

/// Positive finite magnitudes safe for encoder constructors and encode paths.
pub fn sample_positive_finite(rng: &mut StdRng) -> f32 {
    let table = [
        1e-6_f32, 1e-3, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 25.0, 50.0, 100.0,
    ];
    table[rng.random_range(0..table.len())]
}

/// Gain / sensitivity scales biased toward inactive and edge cases.
pub fn sample_gain_scale(rng: &mut StdRng) -> f32 {
    match rng.random_range(0u8..10) {
        0 => 0.0,
        1 => -1.0,
        2 => f32::NAN,
        3 => f32::INFINITY,
        4 => f32::NEG_INFINITY,
        5 => 1e-6,
        6 => 0.5,
        7 => 2.0,
        _ => 1.0,
    }
}

/// Whether a gain/sensitivity scale fully silences stochastic encoders.
#[inline]
pub fn scale_is_inactive(scale: f32) -> bool {
    !scale.is_finite() || scale <= 0.0
}

/// Sample an input value relative to `range`, including non-finite extremes.
pub fn sample_input_value(rng: &mut StdRng, range: (f32, f32)) -> f32 {
    match rng.random_range(0u8..12) {
        0 => f32::NAN,
        1 => f32::INFINITY,
        2 => f32::NEG_INFINITY,
        3 => range.0 - 10.0,
        4 => range.1 + 10.0,
        5 => range.0,
        6 => range.1,
        7 => (range.0 + range.1) * 0.5,
        _ => range.0 + rng.random::<f32>() * (range.1 - range.0),
    }
}

/// Assert batch spikes are unique, in-range, and use the standard polarity/timestamp.
pub fn assert_unique_channel_spikes(spikes: &[SpikeEvent], max_channel_exclusive: usize) {
    let mut seen = std::collections::BTreeSet::new();
    for spike in spikes {
        assert!(
            (spike.channel as usize) < max_channel_exclusive,
            "channel {} out of range for {max_channel_exclusive}",
            spike.channel
        );
        assert!(
            seen.insert(spike.channel),
            "duplicate channel {}",
            spike.channel
        );
        assert_eq!(spike.timestamp, 0);
        assert!(spike.polarity, "stochastic encoders emit positive polarity");
    }
}
