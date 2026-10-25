//! Exports encoder parameters as a NIR (Neuromorphic Intermediate
//! Representation) graph.
//!
//! Run with:
//!
//! ```text
//! cargo run --example nir_export --features nir
//! cargo run --example nir_export --features nir -- /tmp/encoder.nir
//! ```
//!
//! Building the graph needs only the `nir` feature. Writing the `.nir` file
//! needs `nir-rs`'s `hdf5` feature as well — this example reports that boundary
//! instead of failing, so it is runnable on a machine without libhdf5.

use axon_encoder::prelude::*;
use std::env;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let channels = 4;
    let encoder = EmbeddingRateEncoder::try_new(
        channels,
        EmbeddingEncoderConfig {
            // The exported threshold: the encoder fires when a channel's
            // membrane potential reaches this value.
            v_th: 0.4,
        },
    )?;

    let graph = encoder.to_nir(channels)?;

    println!("Exporting {} channel(s) as NIR", encoder.num_channels());
    for (name, node) in &graph.nodes {
        println!("  node {name:<8} type {}", node.type_name());
    }
    for (from, to) in &graph.edges {
        println!("  edge {from} -> {to}");
    }

    match env::args().nth(1) {
        Some(path) => match encoder.write_nir(&path, channels) {
            Ok(()) => println!("Wrote {path}"),
            Err(error) => {
                // Expected when the linked `nir-rs` build has no `hdf5`
                // feature: the graph above is still the export result.
                println!("Could not write {path}: {error}");
                println!("Enable `nir-rs`'s `hdf5` feature in your own dependency to serialize.");
            }
        },
        None => println!("Pass a path to write a .nir file."),
    }

    Ok(())
}
