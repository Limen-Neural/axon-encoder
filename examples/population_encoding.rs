//! Population Encoding Example
//!
//! Demonstrates population coding where a single analog value is distributed
//! across a population of neurons, each tuned to a preferred stimulus value
//! via Gaussian tuning curves. Neurons closest to the input fire most often.
//!
//! ```
//! cargo run --example population_encoding
//! ```

use axon_encoder::prelude::*;

fn main() {
    // Create a population encoder:
    //   num_neurons  = 20    (size of the neural population)
    //   input_range  = (0.0, 100.0)
    //   tuning_width = 10.0  (standard deviation of Gaussian tuning curves)
    let mut encoder =
        PopulationEncoder::try_new(20, (0.0, 100.0), 10.0).expect("valid PopulationEncoder");
    let num_neurons = encoder.num_neurons();

    let test_values = [10.0, 50.0, 90.0];

    println!("=== Population Encoding ===");
    println!(
        "{} neurons covering range [0, 100], tuning width = 10.0\n",
        num_neurons
    );

    for &value in &test_values {
        println!("--- Input value: {} ---", value);

        // Run multiple trials to show probabilistic firing.
        let mut spike_counts = vec![0u32; num_neurons];
        let trials = 100;
        for _ in 0..trials {
            let output = encoder.encode(&[value]);
            for spike in &output.spikes {
                spike_counts[spike.channel as usize] += 1;
            }
        }

        // Print a simple histogram of spike counts per neuron.
        for (neuron, &count) in spike_counts.iter().enumerate() {
            let bar = "#".repeat(count as usize / 2);
            if count > 0 {
                println!("  Neuron {:2}: {:3} spikes  {}", neuron, count, bar);
            }
        }
        println!();
    }
}
