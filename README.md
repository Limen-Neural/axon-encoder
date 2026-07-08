# Axon Encoder

[![CI](https://github.com/Limen-Neural/axon-encoder/actions/workflows/ci.yml/badge.svg)](https://github.com/Limen-Neural/axon-encoder/actions/workflows/ci.yml)

**A flexible and easy-to-use sensory encoding library for Spiking Neural Networks (SNNs).**

`axon-encoder` provides a collection of algorithms to convert real-world, continuous data (like sensor readings, telemetry, or control signals) into spikes—the event-based signals that SNNs understand. This process, known as sensory encoding, is the first step in building powerful and efficient neuromorphic systems.

## What is Sensory Encoding?

Traditional neural networks process dense, continuous values. Spiking Neural Networks, on the other hand, are event-driven: they process sparse, discrete "spikes" that occur at specific points in time.

**Sensory encoding is the bridge between the analog world and the spiking world.** This library gives you the tools to translate your data into meaningful spike trains using various strategies.

## Features

- **A Suite of Encoders**: Choose the right encoding strategy for your data.
    - **`RateEncoder`**: Encodes a value based on the *rate* of firing. Higher input values result in a higher spike frequency.
    - **`DerivativeEncoder`**: Fires spikes based on the *rate of change* of the input. It's great for detecting sudden jumps or drops in a signal.
    - **`TemporalEncoder`**: Detects *temporal patterns* in your data, firing when specific sequences or changes over time are observed.
    - **`PopulationEncoder`**: Encodes a value across a *population* of neurons, where each neuron is tuned to a specific input range.
    - **`DeltaEncoder`**: A simple and efficient encoder that fires a spike when the input value changes by a certain amount.
- **Extensible**: The `Encoder` trait makes it easy to create your own custom encoders.
- **Feature-gated `ndarray` helpers**: With the `ndarray` feature enabled, any encoder can process `ArrayView1` and row-major `ArrayView2` inputs directly through `NdarrayEncoderExt`.
- **Lightweight**: Built with minimal dependencies to be fast and easy to integrate into any project.

## Installation

To use `axon-encoder` in your project, add the following to your `Cargo.toml`:

```toml
[dependencies]
axon-encoder = { git = "https://github.com/Limen-Neural/axon-encoder.git" }
```
*(Note: Once published to crates.io, this will be `axon-encoder = "0.3.0"`)*

For local development, you can use a path dependency:
```toml
[dependencies]
axon-encoder = { path = "../axon-encoder" }
```

To enable direct `ndarray` view helpers:

```toml
[dependencies]
axon-encoder = { git = "https://github.com/Limen-Neural/axon-encoder.git", features = ["ndarray"] }
```

## Quick Start

Here's how to get started with a simple `RateEncoder`.

```rust
use axon_encoder::prelude::*;

fn main() {
    // 1. Load the default configuration, which defines the number of channels.
    //    You can customize this to match your input data.
    let config = EncoderConfig::default(); // Defaults to 256 channels.

    // 2. Initialize an encoder. Let's use a RateEncoder.
    //    It will map input values from a range of (0.0, 1.0) to a firing
    //    rate between 5 Hz and 100 Hz.
    let mut encoder = RateEncoder::new(5.0, 100.0, (0.0, 1.0));

    // 3. Create a sample input stimulus.
    //    Here, we create a simple ramp from 0.0 to 1.0.
    let input: Vec<f32> = (0..config.input_channels)
        .map(|i| i as f32 / (config.input_channels - 1) as f32)
        .collect();

    // 4. Encode the input into spikes!
    let output = encoder.encode(&input);

    // The `output.spikes` vector now contains the generated SpikeEvents.
    println!(
        "Input stimulus of {} values generated {} spikes.",
        input.len(),
        output.spikes.len()
    );
}
```

## Examples

For more detailed examples of each encoder, check out the files in the `/examples` directory. You can run any example with:

```bash
cargo run --example <example_name>
```

For instance, to run the delta encoding example:
```bash
cargo run --example delta_encoding
```

To run the ndarray example:
```bash
cargo run --example ndarray_encoding --features ndarray
```

## A Note for Rust Newcomers

Welcome to Rust! If you're new to the language, some of the syntax in the Quick Start example might seem unfamiliar. Here are a few tips:

- **The Prelude Pattern**: The line `use axon_encoder::prelude::*;` is a common pattern in Rust libraries. The `prelude` is a module that conveniently exports all the most commonly used types and traits, so you can get started with a single `use` statement.

- **Structs and `impl`**: Rust is not a traditional object-oriented language, but it supports similar concepts using `structs` to hold data and `impl` (implementation) blocks to define methods on those structs. In the example, `RateEncoder` is a struct, and its `new` and `encode` methods are defined in an `impl` block.

## Design Philosophy

- **Simplicity and Focus**: The library is designed to do one thing well: sensory encoding. It is unopinionated about your SNN architecture or simulation environment.
- **Performance**: The core encoding loops are designed to be efficient with minimal memory allocation.
- **Accessibility**: We aim to make SNNs more accessible to newcomers by providing clear documentation and easy-to-use tools.

## Purpose and Scope

### Owns

- **Sensory Encoding Algorithms**: Implementation of core mathematical SNN encoding mechanisms (e.g., Rate, Derivative, Temporal, Population, and Delta encoding).
- **Signal-to-Spike Translation**: Converting continuous real-world streams/vectors into discrete biological/event-driven spike events.
- **Deterministic and Stochastic Pipelines**: Algorithms for both deterministic value-to-spike mappings and stochastic Poisson-process spike generators.

### Does Not Own

- **SNN Simulation Engine**: `axon-encoder` does not simulate spiking neural networks, calculate synaptic plasticity (STDP), or manage network topologies. (See [synaptic-mesh](https://github.com/Limen-Neural/synaptic-mesh) and [plasticity-lab](https://github.com/Limen-Neural/plasticity-lab) instead).
- **Domain-Specific Experiments**: Contains no domain-specific code, financial/trading logic, or mining telemetry.
- **Hardware Bindings**: Focuses strictly on software implementations, leaving specific FPGA/ASIC/GPU compilation and execution to downstream crates like [silicon-bridge](https://github.com/Limen-Neural/silicon-bridge).

## Contributing

Contributions are welcome! Whether it's a new encoder, a bug fix, or improved documentation, please feel free to open an issue or submit a pull request.

## License

This project is dual-licensed under either:

- Apache License, Version 2.0 ([LICENSE-APACHE-2.0](LICENSE-APACHE-2.0) or <http://www.apache.org/licenses/LICENSE-2.0>)
- MIT License ([LICENSE-MIT](LICENSE-MIT) or <http://opensource.org/licenses/MIT>)

at your option.
