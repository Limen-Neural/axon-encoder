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
    let mut encoder = LatencyEncoder::new(12, (0.0, 1.0));
    let input = [-0.2, 0.1, 0.5, 0.9, 1.3];
    let output = encoder.encode(&input);

    println!("=== Latency Encoding ===");
    println!("max_latency: 12");
    println!("range: (0.0, 1.0)");
    println!("input: {:?}\n", input);

    for spike in output.spikes {
        println!(
            "channel {} -> timestamp {} polarity {}",
            spike.channel, spike.timestamp, spike.polarity
        );
    }
}
