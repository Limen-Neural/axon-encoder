# Changelog

## [0.6.0] - 2026-10-01

### Added

- Seeded-RNG encoding surfaces for reproducible spike generation:
  `PoissonEncoder::encode_with_rng`, `encode_rate_hz_with_rng`,
  `encode_step_with_rng`, `encode_rate_hz_step_with_rng`, and
  `PopulationEncoder::{encode_with_rng, encode_step_with_rng}`, and
  `RateEncoder::encode_with_rng` — same semantics as the stochastic unseeded
  variants but driven by a caller-supplied `R: rand::Rng + ?Sized`.
  `RateEncoder::encode_step` remains deterministic and does not require an RNG.
- `examples/export_spike_viz.rs`: staged spike-viz export writer driving
  each encoder over a shared stimulus through `TimeCursor`, emitting
  `t.npy` / `neuron_id.npy` / `amp.npy` / `stimulus.npy` / `meta.json`
  with `axon_encoder_git_sha` provenance.

### Fixed

- `export_spike_viz`: `stimulus.npy` rows are now tick-aligned for
  window-model encoders whose `TimeCursor` advances multiple ticks per
  call (latency stimulus is 704×8, not 64×8).
- Spike polarity is preserved in exports as signed amplitude (±1.0).
