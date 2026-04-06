# spikenaut-encoder

**Flexible sensory encoding for spiking neural networks**

Converts continuous telemetry, sensor, or time-series data into biologically plausible spike trains. Designed for cyber-physical systems, crypto mining telemetry, HFT streams, and general neuromorphic research.

## Features

- Rate, Temporal, Predictive (anomaly), Population, and Neuromodulator-driven (16-channel) encoding
- Homeostatic adaptation and rolling Z-score normalization
- Standardized `EncodedOutput` for easy integration with any SNN engine
- Optional embedding output for hybrid SNN-LLM pipelines

## Installation

```toml
[dependencies]
spikenaut-encoder = "0.2"
```

## Quick Start (Pure SNN)

```rust
use spikenaut_encoder::prelude::*;

// Example: 2-channel rate encoder
let mut encoder = RateEncoder::new(5.0, 100.0, (0.0, 100.0));
let telemetry = [75.0, 25.0];
let output = encoder.encode(&telemetry);

for spike in output.spikes {
    println!("Spike on channel {}!", spike.channel);
}
```

## Hybrid Usage (with spikenaut-synapse)

```rust
use spikenaut_encoder::prelude::*;

let mut encoder = NeuromodSensoryEncoder::new(8, 16);
let telemetry = [75.0, 1.05, 95.0, 180.0, 2100.0, 900.0, 45.0, 1.0];
let output = encoder.encode(&telemetry);

if let Some(embedding) = output.embeddings {
    // Feed to spike-to-expert projector -> OLMoE
    println!("Generated embedding for hybrid model: {:?}", embedding);
}
```

## Design Philosophy

- **Independent & Pluggable**: Designed to be a self-contained, reusable library with a simple, flexible API.
- **No LLM Dependency**: The core library is pure SNN, with optional outputs for hybrid models, ensuring it remains lightweight.
- **Extensible**: The `Encoder` trait allows researchers to easily implement their own custom encoding strategies.

## Integration with Spikenaut Ecosystem

`spikenaut-encoder` is a core component of the [Spikenaut ecosystem](https://github.com/Spikenaut), designed to provide clean, spike-based data for other Spikenaut libraries like `spikenaut-synapse` and `spikenaut-reward`.

## Contributing

Contributions are welcome! Please feel free to open an issue or submit a pull request.
