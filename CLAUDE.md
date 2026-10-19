# CLAUDE.md

Claude-specific notes only. There is no `AGENTS.md` in this repo; the quality gate is `REVIEW.md`, and the feature list,
encoder guide, and sibling-crate decision are in `README.md`, `docs/ENCODER_GUIDE.md`, and `docs/architecture-siblings.md`,
so they are not repeated here.

## Commands

Run from the repo root. "Verified" means it ran successfully in a fresh checkout on Rust 1.99.0 (the `rust-toolchain.toml` pin;
Linux, CPU only). `Cargo.lock` is tracked, so `--locked` is correct. Set `CARGO_TARGET_DIR` outside the checkout on a small disk.

| Command | Status |
|---|---|
| `cargo fmt --check` | verified (silent on success) |
| `cargo test --locked` | verified |
| `cargo test --features serde --locked` | verified |
| `cargo test --all-features --locked` | verified |
| `cargo clippy --all-targets --all-features -- -D warnings` | verified |
| `RUSTDOCFLAGS='-D warnings' cargo doc --no-deps --all-features` | verified |
| `cargo run --example export_spike_viz -- rate <out_dir>` | verified (writes `t.npy`, `neuron_id.npy`, `amp.npy`, `stimulus.npy`, `meta.json`) |
| `cargo bench --no-run`, `cargo bench --bench allocations` | not run (large Criterion build; no disk headroom here) |
| `cargo +1.99.0 check --locked --target wasm32-unknown-unknown --no-default-features --features serde,wasm-js` | not run (wasm32 target not installed here) |
| `cargo audit`, `cargo llvm-cov --locked --all-features --workspace --lcov --output-path lcov.info` | not run (`cargo-audit` not installed; coverage is a CI job) |

CI (`ci.yml`) runs wasm, audit, and a Linux/macOS/Windows matrix with an MSRV-pin check; `coverage.yml` uploads to Codecov.
Run the fmt and clippy gates before committing; CI fails on either. The full local bar and regression guards are in `REVIEW.md`.

## Architecture map

- `src/lib.rs`: `Encoder` and `ModulatedEncoder` traits plus the `prelude`. `encoder.rs` holds `EmbeddingRateEncoder`.
- `src/encoders/`: `rate`, `population`, `delta`, `derivative`, `temporal`, `predictive`, `latency`, `phase`.
- `src/modulators.rs`: `EncodingGains`, `GainCurve`, `NeuromodulatorGainCurves` (encoding-side helpers only).
- `src/time.rs`: spike time contract (`TickOffset`, `Timebase`, `TimeModel`). `types.rs`: `SpikeEvent`, `EncodedOutput`.
- `src/sink.rs`, `streaming.rs`: allocation-free `SpikeSink` path and the buffered `StreamingEncoder`. `rng.rs`, `poisson.rs`: sampling.
- `examples/export_spike_viz.rs`: writes the staging layout that spike-viz packs into `spikes.npz` + `meta.json`.
- `tests/`: `time_semantics`, `public_api`, `serde_compat` goldens, `reference_decoders`. `benches/`: `encoders` (Criterion), `allocations` (CSV).

## Invariants and non-goals

- Ground truth for spike-viz (the `spike-viz` sibling repository). Do not change the export file names, `meta.json` fields, or `schema_version`
  without a matching change on the spike-viz side (`docs/axon-encoder-export.md` there).
- Seeded paths must stay bit-for-bit reproducible: stochastic encoders go through `*_with_rng`, and the export uses one seeded `StdRng`.
  Thread-local RNG paths are not replay-stable; `encode_step` on deterministic encoders resumes exactly from a finite JSON serde checkpoint (`DerivativeEncoder` rejects non-finite `last_values` on restore).
- `axon-encoder` and `neuromod` must not depend on each other; adapters sit downstream (`docs/architecture-siblings.md`).
- Keep the `ModulatedEncoder` impls and `encode_*_with_modulators_into` sink paths, `PhaseEncoder`, and `src/modulators.rs` intact (`REVIEW.md` "Do not merge if").
- `EncodedOutput` carries `SpikeEvent`s and an optional embedding only; no `EncoderConfig` or `EncodingMetadata` placeholders.
- No GPU or CUDA code; features are `serde`, `ndarray`, `wasm-js`, all off by default. Enable `wasm-js` only for `wasm32-unknown-unknown`.
- Rust source files have no SPDX headers; do not add them.

## Gotchas

- `Cargo.toml` `rust-version`, `rust-toolchain.toml`, the `toolchain:` pins in `ci.yml` and `coverage.yml`, and the Rust version in the `.devcontainer/Dockerfile` `FROM rust:<version>-slim-bookworm` tag must stay identical.
- `cargo test --test serde_tests` without `--features serde` runs zero tests.
- `Cargo.toml` `exclude` lists `REVIEW.md` and `.agents/` but not `CLAUDE.md`, so a published package would include it; left unchanged here.
- `.claude/` is not gitignored; do not commit local agent state there.
- Never run `cargo publish` or create release tags.
- Edit only what the task needs.
