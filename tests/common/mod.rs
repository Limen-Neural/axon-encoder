//! Assertions shared between the integration suites.
//!
//! `tests/time_semantics.rs` owns the time-contract conformance tables and
//! `tests/property_model.rs` the seeded adversarial sweep (LIM-1334); both
//! judge a call's output with the same [`assert_call_conforms`], so the
//! contract has one definition.

#![allow(dead_code)]

pub mod oracles;

use axon_encoder::prelude::*;

/// Asserts the full ordering and bounding contract for one call's output.
pub fn assert_call_conforms(label: &str, model: TimeModel, spikes: &[SpikeEvent]) {
    let mut previous: Option<SpikeEvent> = None;
    // (channel, offset) pairs whose run has ended. Rules 1 and 2 already make a
    // split run impossible, so this is a redundant check kept as a direct
    // statement of rule 3 — it is the rule consumers rely on when they count a
    // burst in one pass.
    let mut closed: Vec<(u16, u64)> = Vec::new();

    for spike in spikes {
        assert!(
            model.contains(spike.timestamp),
            "{label}: offset {} outside span {}",
            spike.timestamp,
            model.span_ticks()
        );

        if let Some(previous) = previous {
            assert!(
                previous.channel <= spike.channel,
                "{label}: channels not in ascending order ({} after {})",
                spike.channel,
                previous.channel
            );
            if previous.channel == spike.channel {
                assert!(
                    previous.timestamp <= spike.timestamp,
                    "{label}: offsets decrease within channel {} ({} after {})",
                    spike.channel,
                    spike.timestamp,
                    previous.timestamp
                );
            }
            if (previous.channel, previous.timestamp) != (spike.channel, spike.timestamp) {
                closed.push((previous.channel, previous.timestamp.ticks()));
            }
        }

        assert!(
            !closed.contains(&(spike.channel, spike.timestamp.ticks())),
            "{label}: coincident spikes on channel {} at offset {} are not contiguous",
            spike.channel,
            spike.timestamp
        );

        previous = Some(*spike);
    }
}
