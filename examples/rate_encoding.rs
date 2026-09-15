//! Rate Encoding Example

use axon_encoder::prelude::*;

fn main() {
    const INPUT_CHANNELS: usize = 256;
    println!("=== Rate Encoding Example ===");
    println!("Architecture: {INPUT_CHANNELS} independently encoded channels");

    let mut encoder =
        RateEncoder::try_new(5.0, 100.0, (0.0, 1.0), 0.010).expect("valid RateEncoder");

    let inputs: Vec<f32> = (0..INPUT_CHANNELS)
        .map(|i| i as f32 / (INPUT_CHANNELS - 1) as f32)
        .collect();

    println!("Input channels: {}\n", inputs.len());

    for step in 0..5 {
        let output = encoder.encode(&inputs);
        println!(
            "Step {}: {}/{} channels fired",
            step,
            output.spikes.len(),
            INPUT_CHANNELS
        );
    }
}
