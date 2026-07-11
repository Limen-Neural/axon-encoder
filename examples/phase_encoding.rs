//! Phase Encoding Example
//!
//! Demonstrates phase-based encoding, where each channel emits one spike per
//! step and the spike timestamp shifts within a repeating cycle according to
//! the normalized input value.
//!
//! ```
//! cargo run --example phase_encoding
//! ```

use axon_encoder::prelude::*;

fn main() {
    let mut encoder = PhaseEncoder::new(8, (0.0, 1.0));
    let readings = [[0.1, 0.5, 0.9], [0.1, 0.5, 0.9], [0.9, 0.5, 0.1]];

    println!("=== Phase Encoding ===");
    println!("Cycle length: 8 steps\n");

    for (step, input) in readings.iter().enumerate() {
        let output = encoder.encode(input);
        let spikes: Vec<String> = output
            .spikes
            .iter()
            .map(|spike| format!("ch{}@t{}", spike.channel, spike.timestamp))
            .collect();

        println!("Step {}: input {:?} -> {:?}", step, input, spikes);
    }
}
