//! Embedding-Driven Encoding Example
//!
//! `EmbeddingRateEncoder` is a general-purpose integrate-and-fire `Encoder`:
//! it accumulates each call's input into a persistent per-channel membrane
//! potential and fires whenever a channel crosses `config.v_th`. It fits any
//! fixed-width numeric vector, not just embeddings.
//!
//! ```
//! cargo run --example embedding_encoding
//! ```

use axon_encoder::prelude::*;

fn main() {
    // A small fixed-width vector, e.g. one row of an embedding matrix.
    let drive = [0.2_f32, 0.9, 0.5, 0.1];

    let mut encoder =
        EmbeddingRateEncoder::try_new(drive.len(), EmbeddingEncoderConfig { v_th: 0.4 })
            .expect("valid EmbeddingRateEncoder configuration");

    println!("=== Embedding-Driven Encoding ===");
    println!("Drive: {drive:?}, v_th: 0.4\n");

    // Membrane potentials persist across calls, so repeating the same drive
    // can cause additional threshold crossings as potentials keep accumulating.
    for step in 0..4 {
        let output = encoder.encode(&drive);
        println!("Step {step}: {} channels fired", output.spikes.len());
    }

    encoder.reset();
    println!("\nAfter reset:");
    let output = encoder.encode(&drive);
    println!("Step 0: {} channels fired", output.spikes.len());
}
