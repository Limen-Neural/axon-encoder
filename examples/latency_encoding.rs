//! Latency Encoding Example
//!
//! Demonstrates deterministic latency coding where stronger inputs produce
//! earlier spikes and weaker inputs produce later spikes.
//!
//! ```
//! cargo run --example latency_encoding
//! ```

use axon_encoder::prelude::*;

fn main() {
    let mut encoder = LatencyEncoder::try_new(12, (0.0, 1.0)).expect("valid LatencyEncoder");
    let input = [-0.2, 0.1, 0.5, 0.9, 1.3];
    let output = encoder.encode(&input);

    println!("=== Latency Encoding ===");
    println!("max_latency: 12");
    println!("range: (0.0, 1.0)");
    println!("input: {:?}\n", input);

    for spike in output.spikes {
        println!(
            "channel {} -> offset {} ticks, polarity {}",
            spike.channel,
            spike.timestamp.ticks(),
            spike.polarity
        );
    }

    // Offsets are relative to the start of each call; see examples/spike_timebase.rs
    // for turning them into an absolute timeline.
    let model = encoder.time_model();
    println!(
        "\npresentation window: {} ticks (origin advances {} per call)",
        model.span_ticks(),
        model.step_ticks()
    );
}
