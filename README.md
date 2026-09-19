# Axon Encoder

[![Crates.io](https://img.shields.io/crates/v/axon-encoder.svg)](https://crates.io/crates/axon-encoder)
[![Documentation](https://docs.rs/axon-encoder/badge.svg)](https://docs.rs/axon-encoder)
[![License](https://img.shields.io/crates/l/axon-encoder.svg)](https://github.com/Limen-Neural/axon-encoder#license)
[![CI](https://github.com/Limen-Neural/axon-encoder/actions/workflows/ci.yml/badge.svg)](https://github.com/Limen-Neural/axon-encoder/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/Limen-Neural/axon-encoder/branch/main/graph/badge.svg)](https://codecov.io/gh/Limen-Neural/axon-encoder)
[![Codacy code quality](https://img.shields.io/badge/code%20quality-Codacy-222f29?logo=codacy)](https://app.codacy.com/gh/Limen-Neural/axon-encoder/dashboard)
[![Maintainability](https://qlty.sh/gh/Limen-Neural/projects/axon-encoder/maintainability.svg)](https://qlty.sh/gh/Limen-Neural/projects/axon-encoder)

**A flexible sensory encoding library for spiking neural networks (SNNs).**

`axon-encoder` turns continuous data—sensor readings, telemetry, control
signals—into **spikes**, the event-based signals SNNs process. Use it as the
front-end of a neuromorphic pipeline without pulling in a full SNN simulator.

## Installation

**0.4.x is experimental (pre-1.0).** Cargo treats `axon-encoder = "0.4"` as
`^0.4` (that is `>= 0.4.0, < 0.5.0`): compatible **patch** updates only.
A `0.5` release is a new breaking line; pin `"=0.4.0"` if you need an exact
crate version.

```toml
[dependencies]
axon-encoder = "0.4"
```

Optional features:

| Feature | Purpose |
| --- | --- |
| `serde` | Serialize configs, gain types, and live encoder state. Deterministic `encode_step` paths resume exactly from a JSON-serializable checkpoint; stochastic paths that draw a thread-local RNG are not replay-stable. |
| `ndarray` | Encode from `ndarray` views (`ArrayView1` / `ArrayView2`) |
| `wasm-js` | Enable JavaScript-host entropy for `wasm32-unknown-unknown` through `getrandom`'s `wasm_js` backend. Intended for browsers, Web Workers, and supported Node.js hosts. |

```toml
[dependencies]
axon-encoder = { version = "0.4", features = ["ndarray"] }
ndarray = "0.16" # declare yourself so you can build ArrayView values
```

Requires **Rust 1.98.1+** (edition 2024). See `rust-version` in `Cargo.toml`.

### Serde checkpoints

With `features = ["serde"]`, encoder structs serialize **configuration plus
live mutable state** (history, membrane, phase, rate accumulators, pending
spike backlog). Restore a **JSON-serializable** checkpoint and keep calling
`encode_step` with the remaining inputs: deterministic encoders emit the same
spikes and finish in the same state as the original.

That contract covers Delta, Derivative, Temporal, Predictive, Phase,
`EmbeddingRateEncoder`, and `RateEncoder::encode_step`, provided the live
floats are finite. JSON cannot represent NaN/Inf (`serde_json` writes
`null`), and deserialize rejects that payload, so a snapshot taken after
`DerivativeEncoder` stores a non-finite `last_values` entry is not
restorable via `serde_json`.

It does **not** cover stochastic paths: `RateEncoder::encode` (batch), and
both batch and streaming `PopulationEncoder` / `PoissonEncoder`. Those methods
build a thread-local generator internally; serde does not capture it, and they
do not take caller-owned RNG state.

## Quick start

```rust
use axon_encoder::prelude::*;

fn main() {
    // Prefer try_new: typed validation instead of panics on bad config.
    // Range is (min, max); values are clamped to that span. Endpoints map to
    // base_rate / max_rate (here 5–100 Hz at a 10 ms sampling interval).
    let mut encoder = RateEncoder::try_new(5.0, 100.0, (0.0, 1.0), 0.010)
        .expect("valid RateEncoder configuration");

    // Inclusive endpoints 0.0 ..= 1.0 (matches the range above).
    let input: Vec<f32> = (0..64).map(|i| i as f32 / 63.0).collect();
    let output = encoder.encode(&input);

    println!(
        "Input of {} values produced {} spikes.",
        input.len(),
        output.spikes.len()
    );
}
```

Full API docs: [docs.rs/axon-encoder](https://docs.rs/axon-encoder).

## Reusing storage: `encode_into` and `SpikeSink`

`encode` / `encode_step` allocate a fresh `Vec<SpikeEvent>` per call. That is
the right default for exploration, but a runtime stepping thousands of channels
wants one buffer, allocated once. `encode_into` / `encode_step_into` write into
any `SpikeSink` you own instead:

```rust
use axon_encoder::prelude::*;

fn main() {
    let mut encoder = DeltaEncoder::try_new(0.1, 3).expect("valid DeltaEncoder");
    let mut buffer: Vec<SpikeEvent> = Vec::new();

    for step in [[0.5, 0.0, 0.0], [0.5, 0.9, 0.0]] {
        buffer.clear(); // keeps the capacity, drops last step's spikes
        encoder.encode_step_into(&step, &mut buffer);
        println!("{} spikes", buffer.len());
    }
}
```

Same spikes, same order, same state advancement as the returning APIs — and
zero allocations per step once the buffer is warm. `Vec<SpikeEvent>` and
`EncodedOutput` implement `SpikeSink` out of the box; a downstream event
buffer, ring queue, or hardware adapter implements the one-method trait itself,
so no `Vec<SpikeEvent>` is ever built — it keeps spikes in whatever form it
already wants:

```rust
use axon_encoder::prelude::*;

struct EventQueue {
    events: Vec<(u16, u64)>,
}

impl SpikeSink for EventQueue {
    fn push(&mut self, event: SpikeEvent) {
        self.events.push((event.channel, event.timestamp.ticks()));
    }

    fn reserve(&mut self, additional: usize) {
        self.events.reserve(additional);
    }
}
```

`SpikeSink` has a third, optional method — `extend_from_slice` — which defaults
to `push` in a loop. Override it when your sink can take a slice more cheaply
than repeated pushes; encoders deliver spikes through it in fixed-size runs, so
writing through a trait object costs one virtual call per run rather than per
spike.

Encoders **append** to a sink and never clear it, so the caller decides where
step boundaries are. The trait is object-safe, so `&mut dyn Encoder` and
`&mut dyn ModulatedEncoder` still work; `ModulatedEncoder` has the matching
`encode_with_gains_into` / `encode_with_modulators_into`. `PoissonEncoder` is
a documented exception — it does not implement `Encoder`. See
`cargo run --example encode_into_sink`.

## Output and configuration ownership

`EncodedOutput` is deliberately small and framework-agnostic: it carries the
emitted `spikes` and, for embedding-producing encoders, an optional dense
`embeddings` vector. Source identifiers, biological state, tracing data, and
other domain-specific telemetry belong in downstream adapters rather than the
core encoding result.

Each concrete encoder owns its configuration through its constructor
parameters (and, where applicable, a focused encoder-specific config such as
`EmbeddingEncoderConfig`). There is no shared `EncoderConfig`; this avoids
forcing unrelated algorithms into one oversized configuration object.

## Spike time semantics

Every encoder here shares **one time model**, so a consumer can integrate any of
them — or a `&mut dyn Encoder` — without special cases:

> A `SpikeEvent::timestamp` is a `TickOffset`: a count of encoder **ticks**
> measured from the start of the `encode` / `encode_step` call that emitted it.

Timestamps are **call-relative**. They are never absolute and never wall-clock,
because this crate owns no clock and no scheduler. The caller keeps absolute
time in a `TimeCursor` and advances it once per call:

```rust
use axon_encoder::prelude::*;

fn main() -> Result<(), EncoderError> {
    let mut encoder = LatencyEncoder::try_new(9, (0.0, 1.0))?;
    let mut cursor = TimeCursor::new(encoder.time_model());

    for _ in 0..3 {
        let output = encoder.encode_step(&[0.9, 0.1]);
        for spike in &output.spikes {
            // Absolute tick on your timeline; absolute_nanos(..) when a
            // Timebase is available.
            let _tick = cursor.absolute(spike.timestamp);
        }
        cursor.advance(); // by time_model().step_ticks()
    }
    Ok(())
}
```

`Encoder::time_model()` reports the three things a consumer needs:

| | Meaning |
| --- | --- |
| `step_ticks()` | How far your origin advances per call |
| `span_ticks()` | Exclusive bound on offsets a single call can emit |
| `timebase()` | Physical duration of one tick, when the encoder knows it |

Per encoder:

| Encoder | `step_ticks` | `span_ticks` | `timebase` |
| --- | --- | --- | --- |
| `RateEncoder` | 1 | 1 | `dt_seconds` |
| `LatencyEncoder` | `max_latency + 1` | `max_latency + 1` | none |
| `PhaseEncoder` | 1 | `cycle_steps` | none |
| `PopulationEncoder`, `DeltaEncoder`, `DerivativeEncoder`, `TemporalEncoder`, `PredictiveEncoder`, `EmbeddingRateEncoder` | 1 | 1 | none |

**Batch versus streaming.** Both modes follow the same rule, once per call —
`encode` is not a longer window than `encode_step`. `PhaseEncoder` advances its
oscillation by one tick in either mode; the stateful encoders update history in
either mode; `LatencyEncoder` is stateless, so the two are identical.

**Ordering.** Within one `spikes` slice: channel IDs are non-decreasing, offsets
are non-decreasing within a channel, and repeated spikes from one channel at one
offset (a `RateEncoder` burst) are contiguous and mutually unordered — the run
length is a spike *count*, not a sequence. Note this is channel-major, not
globally time-sorted: sort by `timestamp` if you need a chronological stream.

`PhaseEncoder` is the one encoder whose calls overlap (`span_ticks >
step_ticks`), since a call can place a spike anywhere in the ongoing cycle.

Run `cargo run --example spike_timebase` for a worked integration: two encoders,
two cursors, one merged nanosecond-timed stream.

## Migrating from 0.4

The 0.5 line removes public placeholders that had no authoritative consumer:

- **`EncoderConfig` was removed.** Configure each encoder through its own
  constructor parameters or focused config type. The former 256-channel
  defaults did not control the concrete encoders.
- **`EncodingMetadata` and `EncodedOutput::metadata` were removed.** The type
  was empty, so it provided no stable semantics. Keep application or
  framework-specific telemetry in a downstream adapter. Common timebase
  semantics are handled by the explicit time types introduced in
  [issue #62](https://github.com/Limen-Neural/axon-encoder/issues/62), rather
  than a catch-all metadata bag.
- **`EncodedOutput::embeddings` remains.** It is the optional normalized dense
  vector produced alongside spikes by `EmbeddingRateEncoder::forward`.

Additional breaking changes in the 0.5 line:

1. **`SpikeEvent::timestamp` is now `TickOffset`, not `u64`.** The type converts
   both ways and compares against `u64`, so reads like
   `assert_eq!(spike.timestamp, 5)` and `spike.timestamp <= other` still work.
   Construction sites need `SpikeEvent::new(channel, 5u64, true)`,
   `SpikeEvent::at_step_start(channel, true)`, or `TickOffset::new(5)` in the
   struct literal; use `spike.timestamp.ticks()` where a raw `u64` is required.
   The serde representation is unchanged — `TickOffset` is `#[serde(transparent)]`,
   so 0.4 payloads still deserialize.
2. **`PhaseEncoder` emits call-relative offsets.** It previously emitted
   `current_phase + phase_offset`, an absolute value that no other encoder used.
   The old number is `cursor.absolute(spike.timestamp)`, or
   `phase_before_the_call + spike.timestamp.ticks()` if you track the
   oscillation yourself; cycle position stays `absolute % cycle_steps`.
   Capture `current_phase()` **before** the emitting call — every encode call
   advances it afterward, so a read taken after the call is one tick ahead.

Two smaller behavior changes, both in service of making `span_ticks()` a hard
bound rather than an advisory one:

- A neuromodulated `latency_scale` above `1.0` no longer stretches spikes past
  `max_latency`. Latency gains still shorten the window.
- `LatencyEncoder::try_new` rejects `max_latency == u64::MAX` with
  `EncoderError::WindowTooLarge`, since the presentation window is
  `max_latency + 1` ticks and that value has no representable window. The
  `serde` `Deserialize` impl routes through `try_new`, so a 0.4 payload
  persisted with that `max_latency` now fails to load rather than round-tripping.

`Encoder::time_model()` has a default implementation, so out-of-crate `Encoder`
impls keep compiling and inherit `TimeModel::INSTANT`.

### Rate encoder time semantics

`RateEncoder` treats `base_rate` and `max_rate` as firing rates in **hertz**.
Prefer `RateEncoder::try_new(base_rate_hz, max_rate_hz, range, dt_seconds)` so the
sampling interval is explicit (finite and strictly positive). Stochastic batch
encoding uses `p = 1 - exp(-rate_hz * dt_seconds)`; streaming accumulates
`phase += rate_hz * dt_seconds`.

`RateEncoder::new(base_rate, max_rate, range)` remains for compatibility and
uses `dt_seconds = 0.1`.

### Constructor errors

Most encoders expose `try_new(...) -> Result<Self, EncoderError>` for invalid
rates, ranges, windows, thresholds, or channel counts. Prefer those over
panicking `new(...)` in libraries and applications. `PredictiveEncoder` is the
exception: its `new(...)` already returns a `Result`.

### Embedding-driven encoding

`EmbeddingRateEncoder` is a general-purpose integrate-and-fire `Encoder`: it
accumulates each call's input into a persistent per-channel membrane
potential and fires whenever a channel crosses `config.v_th`, so it fits any
fixed-width numeric vector — not just embeddings.

```rust
use axon_encoder::prelude::*;

fn main() -> Result<(), EncoderError> {
    let drive = [0.2_f32, 0.9, 0.5];
    let mut encoder = EmbeddingRateEncoder::try_new(drive.len(), EmbeddingEncoderConfig {
        v_th: 0.4,
    })?;

    for _ in 0..3 {
        let output = encoder.encode(&drive);
        println!("{} channels fired", output.spikes.len());
    }

    encoder.reset(); // zero the membrane potentials before reusing the encoder
    Ok(())
}
```

**Migrating from `forward` / `EncoderState` (0.4, removed in 0.5).** 0.4
threaded state explicitly and normalized a fixed embedding vector once at
construction:

```text
let enc = EmbeddingRateEncoder::new(&embeddings, config);
let (out, next) = enc.forward(&EncoderState::new_zeros(embeddings.len()));
```

0.5 owns its membrane state internally and takes the drive vector as
`encode`'s input, matching every other `Encoder` in the crate:

```text
let mut enc = EmbeddingRateEncoder::try_new(embeddings.len(), config)?;
let out = enc.encode(&embeddings);
```

The built-in min-max normalization is also removed, since it was a hidden
construction-time transform whose result depended on the full embedding
distribution rather than on any one call's input. Callers that relied on it
should normalize before calling `encode`, using the former formula
`(x - min) / (max - min + 1e-5)`.

## Features

- **Encoders** for different signal structures:
  - **`RateEncoder`** — spike *rate* tracks input magnitude
  - **`DerivativeEncoder`** — fires on *change* (jumps / drops)
  - **`TemporalEncoder`** — *patterns* over time
  - **`PopulationEncoder`** — value distributed across a *population* of units
  - **`DeltaEncoder`** — spike when the signal moves by a threshold
  - **`LatencyEncoder`** — stronger input → earlier spike in a window
  - **`PoissonEncoder`** — Poisson-process style sampling
  - **`EmbeddingRateEncoder`** — general-purpose integrate-and-fire over a
    fixed-width numeric vector
- **`Encoder` / `ModulatedEncoder` traits** — plug in custom encoders or apply
  gain scales (`EncodingGains`) without owning a full neuromodulator runtime
- **`SpikeSink` + `encode_into`** — write spikes into caller-owned storage and
  reuse one buffer across steps, or translate straight into your own event type
- **Optional `ndarray` helpers** — `NdarrayEncoderExt` for view-based batch input
- **Small dependency surface** — easy to embed in larger systems

## Randomness (stochastic encoders)

`RateEncoder`, `PopulationEncoder`, and `PoissonEncoder` sample unit floats in
`[0, 1)` via `axon_encoder::rng`:

- **Default:** `gen_unit_f32()` uses a thread-local `rand` generator (not
  reproducible across runs).
- **Reproducible runs:** `gen_unit_f32_with_rng(&mut rng)` with a seeded RNG
  (for example `rand::rngs::StdRng`).
- For **encoding only** — not cryptographic use.

## WebAssembly

Consumers targeting browsers, Web Workers, or supported Node.js hosts (Node.js
19+) on `wasm32-unknown-unknown` must enable the `wasm-js` feature to select
`getrandom`'s supported `wasm_js` backend:

```toml
[dependencies]
axon-encoder = { version = "0.4", features = ["wasm-js"] }
```

The feature is opt-in because `wasm32-unknown-unknown` also supports non-JS and
non-Web runtimes where a JavaScript backend is unavailable. Leave `wasm-js`
disabled for those targets and select a randomness backend appropriate to the
runtime at the final binary or application layer.

## Examples

Clone the repository and run:

```bash
cargo run --example rate_encoding
cargo run --example delta_encoding
cargo run --example embedding_encoding
cargo run --example spike_timebase
cargo run --example encode_into_sink
cargo run --example ndarray_encoding --features ndarray
```

Other examples live under `examples/` (latency, population, temporal,
predictive, gain-adapter patterns, and more).

## What this crate is (and is not)

### In scope

- Sensory / signal → spike encoding algorithms
- Deterministic and stochastic encoding pipelines
- Generic gain controls (`EncodingGains`, gain curves) used only for scaling
  rate, threshold, latency, or sensitivity at encode time

### Out of scope

- Full SNN simulation, network topology, or synaptic plasticity (STDP)
- Long-horizon biological neuromodulator *dynamics* or reward loops (this crate
  only provides encoding-local gain helpers)
- FPGA / ASIC / GPU device bindings

The library is intentionally unopinionated about which simulator or hardware
stack you plug the spikes into.

## Contributing

Issues and pull requests are welcome—new encoders, fixes, and docs improvements
alike. Development notes and CI conventions live in the repository
(`REVIEW.md`, `.github/`).

The `.devcontainer/` configuration is available for VS Code Dev Containers
and Codespaces contributor workflows. It is an editor development environment,
not a published or supported distribution artifact; consumers should use the
crate from Cargo as described in [Installation](#installation).

## License

Dual-licensed under either of:

- Apache License, Version 2.0 ([LICENSE-APACHE-2.0](LICENSE-APACHE-2.0) or
  <http://www.apache.org/licenses/LICENSE-2.0>)
- MIT License ([LICENSE-MIT](LICENSE-MIT) or
  <http://opensource.org/licenses/MIT>)

at your option.
