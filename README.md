<p align="center">
  <img src="docs/logo.png" width="220" alt="Spikenaut">
</p>

<h1 align="center">spikenaut-encoder</h1>
<p align="center">Hardware telemetry → spike train conversion for neuromorphic systems</p>

<p align="center">
  <a href="https://crates.io/crates/spikenaut-encoder"><img src="https://img.shields.io/crates/v/spikenaut-encoder" alt="crates.io"></a>
  <a href="https://docs.rs/spikenaut-encoder"><img src="https://docs.rs/spikenaut-encoder/badge.svg" alt="docs.rs"></a>
  <img src="https://img.shields.io/badge/license-GPL--3.0-orange" alt="GPL-3.0">
</p>

---

Converts continuous sensor data (temperature, voltage, power, hashrate, or any numeric
stream) into biologically realistic spike trains using multiple encoding strategies with
adaptive thresholds and homeostatic normalization.

## Features

- `RateEncoder` — Poisson spike generation proportional to signal magnitude
- `TemporalEncoder` — burst coding triggered by rapid signal changes
- `PredictiveEncoder` — adaptive EMA threshold; spikes on deviation from expectation (anomaly detection)
- `PopulationEncoder` — multiple neurons encoding sub-ranges of a single channel
- `NeuromodSensoryEncoder` — 16-channel Poisson encoder with per-channel gains, biases, and homeostatic adaptation
- Z-score normalization with rolling mean/variance

## Installation

```toml
spikenaut-encoder = "0.1"
```

## Quick Start

```rust
use spikenaut_encoder::{NeuromodSensoryEncoder, EncoderConfig};

let mut encoder = NeuromodSensoryEncoder::new(EncoderConfig::default());

// Convert 8 telemetry channels to 16 spike channels
let telemetry = [75.0f32, 1.05, 95.0, 180.0, 2100.0, 900.0, 45.0, 1.0];
let spikes = encoder.encode(&telemetry);  // [bool; 16]
```

### Predictive Encoding (Anomaly Detection)

```rust
use spikenaut_encoder::PredictiveEncoder;

let mut enc = PredictiveEncoder::new(0.1); // EMA alpha
for sample in stream {
    let spike = enc.update(sample);  // fires when |x - ema| > threshold
}
```

## Encoding Strategies

| Encoder | Biological Analogue | Use Case |
|---------|--------------------|-|
| `RateEncoder` | Rate coding | Slow signals, graded responses |
| `TemporalEncoder` | Burst coding | Rapid transients, events |
| `PredictiveEncoder` | Predictive coding | Anomaly detection, novelty |
| `PopulationEncoder` | Population coding | High-precision single channels |

## Extracted from Production

Extracted from [Eagle-Lander](https://github.com/rmems/Eagle-Lander), a private
neuromorphic crypto-mining supervisor. The encoding pipeline was decoupled from
GPU-specific telemetry paths so it works with any numeric data source.

## Part of the Spikenaut Ecosystem

| Library | Purpose |
|---------|---------|
| [spikenaut-reward](https://github.com/rmems/spikenaut-reward) | Homeostatic reward (efferent arm) |
| [spikenaut-backend](https://github.com/rmems/spikenaut-backend) | SNN backend abstraction |
| [neuromod](https://crates.io/crates/neuromod) | Neuromodulator dynamics |

## License

GPL-3.0-or-later
