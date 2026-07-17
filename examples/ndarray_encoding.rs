//! Encode `ndarray` rows directly with the `ndarray` feature enabled.
//!
//! ```bash
//! cargo run --example ndarray_encoding --features ndarray
//! ```

use axon_encoder::prelude::*;
use ndarray::arr2;

fn main() {
    println!("Rate encoding with ndarray");
    let input = arr2(&[[0.2_f32, 0.8], [0.7, 0.1], [0.9, 0.9]]);
    let mut encoder = RateEncoder::new(0.0, 10.0, (0.0, 1.0));

    for (row_idx, output) in encoder
        .encode_step_array2(input.view())
        .into_iter()
        .enumerate()
    {
        println!("row {row_idx}: {} spike(s)", output.spikes.len());
    }
}
