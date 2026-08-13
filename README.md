# Axon Encoder

[![CI](https://github.com/Limen-Neural/axon-encoder/actions/workflows/ci.yml/badge.svg)](https://github.com/Limen-Neural/axon-encoder/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/Limen-Neural/axon-encoder/branch/main/graph/badge.svg)](https://codecov.io/gh/Limen-Neural/axon-encoder)
[![Qodana](https://github.com/Limen-Neural/axon-encoder/actions/workflows/qodana_code_quality.yml/badge.svg)](https://github.com/Limen-Neural/axon-encoder/actions/workflows/qodana_code_quality.yml)

**A flexible and easy-to-use sensory encoding library for Spiking Neural Networks (SNNs).**

`axon-encoder` provides a collection of algorithms to convert real-world, continuous data (like sensor readings, telemetry, or control signals) into spikes—the event-based signals that SNNs understand. This process, known as sensory encoding, is the first step in building powerful and efficient neuromorphic systems.

## Requirements

| | |
| --- | --- |
| **MSRV** | **Rust 1.97.1** (`rust-version` in `Cargo.toml`) |
| **Edition** | 2024 |
| **Pin** | [`rust-toolchain.toml`](rust-toolchain.toml) (channel `1.97.1`) |

CI installs the same toolchain on **Linux, macOS, and Windows**. Keep
`Cargo.toml` `rust-version`, `rust-toolchain.toml`, and the version string in
[`.github/workflows/ci.yml`](.github/workflows/ci.yml) identical (the CI job
fails if they drift). See [REVIEW.md](REVIEW.md) for the bump procedure.

## Development environments

Three supported paths (all use Rust **1.97.1**):

| Path | When to use |
| --- | --- |
| **Native toolchain** | Local `rustup` + `cargo` (see [REVIEW.md](REVIEW.md)) |
| **Dev Container / Codespaces** | VS Code or GitHub Codespaces via [`.devcontainer/`](.devcontainer/) |
| **Docker** | No local Rust install — build and run tests in a container |

### Dev Container

Open the repo in VS Code (“Reopen in Container”) or GitHub Codespaces. The
container installs rustfmt/clippy and runs `cargo fetch` on create.

### Docker (test without local Rust)

```bash
docker build -t axon-encoder:dev .
docker run --rm axon-encoder:dev
```

That image runs `cargo test --all-features --locked` by default (see
[`Dockerfile`](Dockerfile)). CI also builds it on pushes to `main` and pull
requests targeting `main`
([`.github/workflows/docker.yml`](.github/workflows/docker.yml)).

Multi-OS native CI already covers Linux / macOS / Windows
([`.github/workflows/ci.yml`](.github/workflows/ci.yml)).

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
  - **`LatencyEncoder`**: Encodes stronger inputs as earlier spike times within a fixed temporal window.
- **Extensible**: The `Encoder` trait makes it easy to create your own custom encoders.
- **Feature-gated `ndarray` helpers**: With the `ndarray` feature enabled, encoders can process `ArrayView1` / `ArrayView2` inputs via `NdarrayEncoderExt` (standard row-major layout is most efficient). Independent row encoding via `encode_array2` requires the encoder type to implement `Clone`.
- **Lightweight**: Built with minimal dependencies to be fast and easy to integrate into any project.

## Randomness (stochastic encoders)

Encoders that sample spikes stochastically (`RateEncoder`,
`PopulationEncoder`, `PoissonEncoder`) draw unit floats in `[0, 1)` via
`axon_encoder::rng`:

- **Default:** `gen_unit_f32()` uses a thread-local generator from `rand`.
  Sequences are **not** reproducible across runs.
- **Reproducible experiments:** pass a seeded RNG into
  `gen_unit_f32_with_rng(&mut rng)` (for example `rand::rngs::StdRng` with
  `SeedableRng`). See the `rng` module docs for details.
- These helpers are for **sensory / spike sampling**, not for
  cryptographic secrets or key material.

## WebAssembly

When targeting `wasm32-unknown-unknown`, enable a working
[getrandom](https://docs.rs/getrandom) backend for that target (often the
JS/browser feature set required by your toolchain). Stochastic encoders
depend on OS/entropy-backed RNGs through `rand`.

## Installation

The crate is currently **not yet published** to [crates.io](https://crates.io).
The in-tree version is **0.4.0** (experimental pre-1.0). First crates.io publish
is tracked in [issue #60](https://github.com/Limen-Neural/axon-encoder/issues/60).

**Git (bleeding edge / until #60 lands):**

```toml
[dependencies]
axon-encoder = { git = "https://github.com/Limen-Neural/axon-encoder.git" }
```

**Path (local development):**

```toml
[dependencies]
axon-encoder = { path = "../axon-encoder" }
```

**Target crates.io form (after #60):**

```toml
[dependencies]
axon-encoder = "0.4"
```

To enable direct `ndarray` view helpers (declare `ndarray` yourself so you can construct and name `ArrayView` values):
```toml
[dependencies]
axon-encoder = { git = "https://github.com/Limen-Neural/axon-encoder.git", features = ["ndarray"] }
ndarray = "0.16"
```

## Quick Start

Here's how to get started with a simple `RateEncoder`.

```rust
use axon_encoder::prelude::*;

fn main() {
    // 1. Load the default configuration, which defines the number of channels.
    //    You can customize this to match your input data.
    let config = EncoderConfig::default(); // Defaults to 256 channels.

    // 2. Initialize an encoder. Prefer try_new for typed validation errors;
    //    RateEncoder::new(...) panics on invalid rates/ranges (dt defaults to 0.1).
    //    Maps (0.0, 1.0) inputs to 5–100 Hz at a 10 ms sampling interval.
    let mut encoder = RateEncoder::try_new(5.0, 100.0, (0.0, 1.0), 0.010)
        .expect("valid RateEncoder configuration");

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

### Rate encoder time semantics

`RateEncoder` treats `base_rate` and `max_rate` as physical firing rates in hertz. New code should prefer `RateEncoder::try_new(base_rate_hz, max_rate_hz, range, dt_seconds)` so the sampling interval is explicit and validated as finite and strictly positive. Stochastic batch encoding converts rates to per-step probabilities with `p = 1 - exp(-rate_hz * dt_seconds)`, while streaming encoding accumulates expected spikes with `phase += rate_hz * dt_seconds`.

For migration compatibility, `RateEncoder::new(base_rate, max_rate, range)` remains available and uses `dt_seconds = 0.1`, preserving the old deterministic `/ 10.0` increment for unit rates.

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
- **Generic encoding controls**: `EncodingGains`, `ModulatedEncoder`, and optional gain-curve helpers used to scale rate/threshold/latency/sensitivity **without** depending on the [`neuromod`](https://github.com/Limen-Neural/neuromod) crate.

### Does Not Own

- **SNN Simulation Engine**: `axon-encoder` does not simulate spiking neural networks, calculate synaptic plasticity (STDP), or manage network topologies. (See [synaptic-mesh](https://github.com/Limen-Neural/synaptic-mesh) and [plasticity-lab](https://github.com/Limen-Neural/plasticity-lab) instead).
- **Biological modulator dynamics**: Long-horizon dopamine/cortisol/ACh dynamics, STDP, and network reward loops live in [`neuromod`](https://github.com/Limen-Neural/neuromod) — not here.
- **Domain-Specific Experiments**: Contains no domain-specific code, financial/trading logic, or mining telemetry.
- **Hardware Bindings**: Focuses strictly on software implementations, leaving specific FPGA/ASIC/GPU compilation and execution to downstream crates like [silicon-bridge](https://github.com/Limen-Neural/silicon-bridge).

### Sibling crates (no direct dependency)

`axon-encoder` and `neuromod` are **independent siblings**. Neither crate may depend on the other:

```text
              application / adapter
               ↙             ↘
         neuromod          axon-encoder
```

Downstream apps map biological state → generic `EncodingGains` (see
[`examples/sibling_gains_adapter.rs`](examples/sibling_gains_adapter.rs) and
[`docs/architecture-siblings.md`](docs/architecture-siblings.md)).

## Contributing

Contributions are welcome! Whether it's a new encoder, a bug fix, or improved documentation, please feel free to open an issue or submit a pull request.

## License

This project is dual-licensed under either:

- Apache License, Version 2.0 ([LICENSE-APACHE-2.0](LICENSE-APACHE-2.0) or <http://www.apache.org/licenses/LICENSE-2.0>)
- MIT License ([LICENSE-MIT](LICENSE-MIT) or <http://opensource.org/licenses/MIT>)

at your option.

## Encoder construction errors

Public encoder types provide `try_new(...)` constructors for runtime configuration
validation. Prefer these fallible constructors in library/application code so
invalid rates, ranges, window sizes, thresholds, and channel counts are reported
as typed [`EncoderError`](https://docs.rs/axon-encoder/latest/axon_encoder/enum.EncoderError.html)
values instead of panics.

Constructor conventions:

- Most encoders: `try_new(...) -> Result<Self, EncoderError>`; legacy
  `new(...)` panics on invalid configuration (e.g. `RateEncoder::new`,
  `LatencyEncoder::new`, `DeltaEncoder::new`, `DerivativeEncoder::new`).
- `PredictiveEncoder` is the exception: `new(...)` already returns
  `Result<Self, PredictiveEncoderError>` for source compatibility, while
  `try_new(...)` returns the unified `EncoderError`.
