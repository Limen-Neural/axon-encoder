# Changelog

## [Unreleased]

## [0.7.0] - 2026-11-14

### Added

- `nir` feature (off by default): NIR (Neuromorphic Intermediate Representation)
  parameter-graph export built on the maintained `nir-rs` crate. `ToNir::to_nir`
  projects an encoder's **static parameters** onto a `nir_rs::NirGraph`;
  `ToNir::write_nir` / `nir::write_nir` delegate serialization to
  `nir_rs::io::write`, so this crate gains no HDF5 dependency and never
  hand-rolls the wire format. Exported graphs carry `axon_encoder.*` provenance
  metadata and are validated against `nir-rs` before they are returned.
  `EmbeddingRateEncoder` is the one mapped encoder (an integrate-and-fire
  neuron, exported as a NIR `IF` node with `config.v_th` as `v_threshold`);
  encoders whose semantics NIR has no node for are deliberately unmapped and
  documented as such in the `nir` module. Verified end to end: a graph written
  through `nir-rs` 0.4.5 loads in the Python reference implementation `nir`
  1.0.8. See [issue #43](https://github.com/Limen-Neural/axon-encoder/issues/43).
- `EmbeddingRateEncoder::num_channels`, returning the tracked channel count.
- `examples/nir_export.rs`: builds a NIR graph from an encoder and reports the
  `nir-rs`/HDF5 boundary instead of failing on a machine without libhdf5.

- `BaselineHoldTernaryEncoder`: per-channel baseline-hold ternary encoding with
  inclusive signed threshold crossings and baseline updates only on events
  (Corinth `TelemetryEncoder` semantics from
  [corinth-canal#158](https://github.com/rmems/corinth-canal/issues/158)).
  Each channel seeds independently on its first observed sample. Exposes
  `encode_ternary` for dense `{-1, 0, +1}` output plus an `Encoder` adapter
  that maps sign to spike polarity at tick zero. A Corinth
  `TelemetryFunnel::reset` must not call this encoder's `reset` (Corinth keeps
  the telemetry baseline across GIF reset). Joint Corinth adoption of
  axon-encoder plus neuromod is still blocked by the `NeuroModulators` /
  `PoissonEncoder` name collision
  ([corinth-canal#143](https://github.com/rmems/corinth-canal/issues/143)).
- `serde` feature: `Serialize`/`Deserialize` for `TimeModel` and `TimeCursor`,
  so an encoder checkpoint can now persist the caller's clock (absolute
  `origin`, `step_ticks`/`span_ticks` tick geometry, optional `Timebase`) as
  named-field JSON and resume the same timeline after restore.

### Fixed

- `BaselineHoldTernaryEncoder` no longer uses one `initialized` flag. A short
  first frame seeds only the channels it contains; unobserved channels stay
  unseeded instead of comparing later samples against `0.0`. `reset` clears
  every channel. Serde round-trips a per-channel `seeded` flag.

### Changed

- `rust-version` lowered from `1.99.0` to the measured MSRV **1.88.0**
  (let-chains in `src/streaming.rs`; `1.85` fails). The optional `nir`
  feature remains floored at Rust 1.98.1 by `nir-rs` 0.4.5's own
  `rust-version`. The dev toolchain stays
  pinned at 1.99.0 (`rust-toolchain.toml`, CI, devcontainer) and a dedicated
  `msrv` CI job now builds and tests on the `rust-version` toolchain, per the
  MSRV-sweep policy ([issue #123](https://github.com/Limen-Neural/axon-encoder/issues/123),
  parent [neuromod#214](https://github.com/Limen-Neural/neuromod/issues/214)).
- **Breaking:** `TemporalEncoder::new` / `try_new` take `change_thresholds: Vec<f32>`
  instead of `Vec<(f32, u16)>`. The `u16` "spike value" was never read (it set
  neither polarity nor channel), so it is removed; pass `vec![0.5]` where you
  passed `vec![(0.5, 1)]`. Checkpoints written with the pair form still
  deserialize (labels dropped); new checkpoints write plain floats. The encoder
  is now documented as a dual-window change detector rather than a temporal
  code, `history_depth` beyond `TemporalEncoder::WINDOW` (6) is documented and
  tested as buffer-only, and `TemporalEncoder::WINDOW` is public. An empty
  `change_thresholds` is now rejected (`CountMustBePositive`) in `try_new` and
  on deserialize instead of producing an encoder that never fires.
  `PredictiveEncoder` keeps its `(f32, u16)` pairs for now.
- `serde` feature: `Timebase` now serializes as `{"tick_nanos": N}` instead of
  a bare `u64`, disambiguating it from a `TickOffset` tick count. The v0.5
  bare-integer payload still deserializes; zero and other invalid values are
  rejected in both forms.

## [0.6.0] - 2026-10-01

### Added

- Seeded-RNG encoding surfaces for reproducible spike generation:
  `PoissonEncoder::encode_with_rng`, `encode_rate_hz_with_rng`,
  `encode_step_with_rng`, `encode_rate_hz_step_with_rng`,
  `PopulationEncoder::{encode_with_rng, encode_step_with_rng}`, and
  `RateEncoder::encode_with_rng` — same semantics as the stochastic unseeded
  variants but driven by a caller-supplied `R: rand::Rng + ?Sized`.
  `RateEncoder::encode_step` remains deterministic and does not require an RNG.
  Exact seeded streams are not guaranteed across crate or RNG-version upgrades.
- `examples/export_spike_viz.rs`: staged spike-viz export writer driving
  each encoder over a shared stimulus through `TimeCursor`, emitting
  `t.npy` / `neuron_id.npy` / `amp.npy` / `stimulus.npy` / `meta.json`
  with `axon_encoder_git_sha` provenance.

### Fixed

- `export_spike_viz`: `stimulus.npy` rows are now tick-aligned for
  window-model encoders whose `TimeCursor` advances multiple ticks per
  call (latency stimulus is 704×8, not 64×8).
- Spike polarity is preserved in exports as signed amplitude (±1.0).
