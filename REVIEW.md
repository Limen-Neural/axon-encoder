# Local review quality gate

These commands are the **human quality bar** beyond GitHub Actions.
Run them before claiming a PR is ready when the change touches `src/`,
`Cargo.toml`, public APIs, or randomness.

They match the suite used to sign off PR #37 (rmems, 2026-07-15) and are
**required** for security-oriented PRs such as PR #50 so merge thrash
cannot land “green CI” while deleting product surface.

## MSRV pin rule

`Cargo.toml` `rust-version`, `rust-toolchain.toml` `channel`, and the
`toolchain:` string in `.github/workflows/ci.yml` must stay **identical**
(currently **1.99.0**). The contributor-only `.devcontainer/Dockerfile`
`FROM rust:<ver>` tag must match too. CI fails if these pins drift
(issue #67 / LIM-1014, #61 / LIM-972).

The coverage workflow uses the same pin. `.agents/setup` and `.agents/resume`
read `rust-toolchain.toml` directly, so fresh and resumed orbs use Rust 1.99.0
without a separate version constant.

To bump MSRV:

1. Set the new version in Cargo.toml, rust-toolchain.toml, ci.yml, and the
   `FROM rust:` tag in `.devcontainer/Dockerfile`; update coverage.yml and the
   devcontainer display name and README.md's Rust requirement too.
2. Run the mandatory commands below on that toolchain
   (`rustup run <ver> cargo test --locked`, etc.).
3. Confirm GitHub Actions matrix (Linux / macOS / Windows) is green.
4. Rebuild the devcontainer and confirm the editor tooling starts correctly.

Do not bump only one pin.

## When to run

- Before every push that changes encoder / modulator / RNG code
- After resolving merges with `main`
- Before requesting review or merge on PR #50 and similar PRs

## Mandatory commands

### Format + core locked test matrix

```bash
# Success is silent: exit 0 and no stdout means formatting is clean.
cargo fmt --check

cargo test --locked
cargo test --features serde --locked
cargo clippy --all-features -- -D warnings
RUSTDOCFLAGS='-D warnings' cargo doc --no-deps --all-features
cargo test --package axon-encoder --lib -- rng::tests
cargo test --message-format=json-diagnostic-rendered-ansi \
  --color=always --no-run --package axon-encoder --lib \
  --profile test
```

### This PR’s edge-case filters

```bash
cargo test --locked --package axon-encoder --lib -- \
  test_population_encoder_empty_input \
  test_rate_encoder_non_finite_rate_scale_silences \
  rng::tests
```

### Benchmarks (smoke)

```bash
# From a terminal (not RustRover's default "test" runner flags):
cargo bench
# Or a single Criterion bench:
cargo bench --bench encoders
# Allocation CSV smoke (custom harness, not Criterion):
cargo bench --bench allocations
```

### Measured encoder coverage

`benches/encoders.rs` has **27 Criterion groups**, each with three scales.
**T+A** below means both elapsed time and allocations are measured. The
returning path is `encode`, the step path is `encode_step`, and the reusable
path writes into a caller-owned sink (`encode_into` or `encode_step_into`).

| Encoder | Returning path | Step path | Reusable path |
| --- | --- | --- | --- |
| Rate | `encode`: T+A | `encode_step`: T+A | `encode_into` and `encode_step_into`: T+A |
| Population | `encode`: T+A | Delegates to `encode`; represented by returning path | `encode_into`: T+A; equivalent `encode_step_into` represented by it |
| Delta | `encode`: T+A | `encode_step`: T+A | `encode_step_into`: T+A; `encode_into` equivalent at these configured widths |
| Derivative | `encode`: T+A | `encode_step`: T+A (same core) | `encode_step_into`: T+A; equivalent `encode_into` represented by it |
| Temporal | `encode`: T+A | `encode_step`: T+A | `encode_step_into`: T+A; `encode_into` equivalent at these configured widths |
| Predictive | `encode`: T+A | `encode_step`: T+A | `encode_step_into`: T+A; `encode_into` equivalent at these configured widths |
| Latency | `encode`: T+A | Allocation only; stateless and equivalent to returning path | `encode_into`: T+A; equivalent `encode_step_into`: allocation only |
| Phase | `encode`: T+A | `encode_step`: T+A (same phase advancement) | `encode_into`: T+A; equivalent `encode_step_into` represented by it |
| EmbeddingRate | `encode`: T+A | `encode_step`: T+A (same core) | `encode_step_into`: T+A; equivalent `encode_into` represented by it |
| Poisson | Scalar `encode`: T+A | Scalar `encode_step`: not measured; no `Encoder` implementation | Not applicable: no sink API |
| `ModulatedEncoder` | Not timed or allocation-profiled | Not timed or allocation-profiled | Allocation-only Delta `encode_step_with_modulators_into` smoke; no per-encoder modulated timing coverage |

Scale and fixture details:

- Ordinary `BenchmarkId`s are **256, 1,000, 10,000 input channels**. Population
  uses those values as **neuron counts for one scalar input** (`[75.0]`), not
  input-channel counts. Poisson uses **10, 100, 1,000 simulation steps** for
  scalar probability `0.5`; it returns a bit train, whose set bits are counted
  as spikes in the allocation report.
- Derivative uses thresholds `vec![0.1; size]`, seeds zeros, and alternates
  `0.25` and zero per channel. Returning path and step path are equivalent;
  the reusable path starts with a capacity-`size` sink seeded with zeros.
- Phase uses 16 cycle steps, range `(0.0, 1.0)`, and normalized input. Each call
  emits `size` spikes and advances the same encoder's phase by one tick.
- **EmbeddingRate** means `EmbeddingRateEncoder`, not a separate `Embedding`
  encoder. `EmbeddingEncoderConfig { v_th: 0.5 }` and drive `vec![0.5; size]`
  keep every membrane bounded: each call adds and subtracts exactly `0.5`.
  Warm calls verify full output before measurement; there are no timed resets.
- Rate's returning path and `encode_into` reusable path sample stochastic
  spikes from normalized input. Its step path and `encode_step_into` reusable
  path accumulate deterministic phase using the same normalized input. These
  are distinct workloads: compare like operations. The allocation-only
  `encode_step(backlog)` row retains eight channels at 100,000 Hz and `dt=0.1`.
- Delta's returning path holds shifted input constant and settles silent;
  its step path and reusable path alternate normalized and shifted inputs.
  Temporal's returning path holds normalized input constant; its step path
  and reusable path prime and cycle three low then three high inputs.
  Predictive primes five low calls and cycles three low then three high inputs
  on all measured paths. The allocation report samples its first high call
  after advancing through the cycle's three low calls. These workload
  differences matter when interpreting returning path and reusable path comparisons.

Criterion measures **elapsed time over repeated calls**, including returning
output destruction or reusable sink clear/refill. Construction, fixtures,
explicit priming, and buffer allocation stay outside `b.iter`; state persists.
Criterion defaults are unchanged. `benches/allocations.rs` instead counts
**one warmed call**, with the matching fixture and priming. Constant-input
returning paths are warmed to steady state (six calls for Temporal); stochastic
paths warm their thread-local RNG. New Derivative, Phase, and EmbeddingRate
rows assert `spikes == size`. Counts are observations, not timing estimates.
The CSV schema remains:

```text
encoder,operation,scale_type,scale,allocations,bytes,spikes
```

`allocations` counts successful allocation/reallocation requests; `bytes`
counts allocated bytes plus net reallocation growth, not peak resident memory.
Only rows reporting zero establish zero allocations for that operation,
fixture, scale, and warm-up. A silent (`spikes=0`) call does not establish
allocation behavior under active output. Buffer reuse alone proves nothing;
reusable path allocations are reported without failing the benchmark. Document
any unexpected allocations here and in README.md and open a separate issue;
keep encoder implementation changes out of coverage work.

The local Rust 1.98.1 run recorded **91 allocation rows**, including **36
reusable rows** with zero allocations and zero bytes at every measured scale.
All 27 Derivative, Phase, and EmbeddingRate rows emitted `size` spikes. No
unexpected reusable path allocation was observed, so no allocation follow-up
issue was needed. Delta and Temporal constant-input returning paths reported
zero spikes; their counts do not describe active output.

### Repeatable local regression comparison

Use **Rust 1.99.0**, the **same host**, the same lockfile/dependencies where
possible, and Criterion defaults on both commits. Keep load and power settings
stable. Record any dependency differences. Use one checkout and retain its
`target/criterion` directory across switches; do not run `cargo clean` between
runs. With separate worktrees, set the same absolute `CARGO_TARGET_DIR` on both
so the branch reuses the comparison's `target/criterion` baseline directory.
Keep logs and CSVs outside the checkout. Start with a clean tree or commit your
branch changes before switching:

```bash
# On the chosen comparison commit (record its full SHA; "main" is a label):
git switch --detach <comparison-commit>
cargo +1.99.0 bench --bench encoders -- --save-baseline main
cargo +1.99.0 bench --bench allocations > /tmp/allocations-main.csv

# On the branch commit, on the same host and using the saved baseline directory:
git switch <branch>
cargo +1.99.0 bench --bench encoders -- --baseline main
cargo +1.99.0 bench --bench allocations > /tmp/allocations-branch.csv
diff -u /tmp/allocations-main.csv /tmp/allocations-branch.csv
```

Both baseline commands select **`--bench encoders`** because the flags belong to
Criterion; `allocations` is a separate custom CSV harness, and the library test
harness does not accept those flags. Allocation counts require their own run
on **each** commit; Criterion cannot infer them. Compare CSVs by
`(encoder, operation, scale_type, scale)`, including spikes and bytes, allowing
for stochastic output differences and explicitly identifying new rows.

If the comparison predates a group, that group has no baseline: report it as
new coverage, not a regression. Run common groups with the same positional
Criterion filter on both baseline commands, and run new groups separately
without `--baseline`. Do not copy branch results into a comparison baseline.
Use `cargo bench` as the final local completion check for both targets. CI
runs `cargo bench --no-run` on the pinned toolchain; timing stays local.

A benchmark PR report must include comparison and branch **full SHAs** (and
whether the tree was dirty), Rust version, host/OS/CPU and noise/load conditions,
commands and baseline label/directory, encoder **operation and scale**, elapsed
time and Criterion **change % / confidence interval**, and allocation
**allocations / bytes / spikes** on both commits. Mark missing baselines and
allocation-only rows, summarize the verdict, and link any allocation follow-up
issue. Include the complete CSVs as artifacts rather than full timing logs.

### How to read results

- Prefer Criterion **change %** over absolute ns
  (machine-dependent).
- “Change within noise threshold” / low-single-digit % is
  usually fine.
- Real regressions are multi-x slowdowns or consistent multi-percent
  hits across scales.
- Do **not** paste full Criterion logs into PR comments by default
  (see [How to post results on a PR](#how-to-post-results-on-a-pr)).

### How to post results on a PR

Reviewers and authors should post **human-readable** results, not raw
IDE/terminal dumps. Prefer **verdict first**, then small tables.

**Do:**

- Lead with **branch tip SHA** and a one-line pass/fail verdict
- Use markdown tables (Check → Result; Encoder → allocs; Area → Read)
- Name the **command** once (e.g. `cargo bench --bench allocations`)
- State host noise (laptop vs dedicated) and that Criterion `change %`
  is vs a **local baseline**, not necessarily `main`, unless you used
  `--save-baseline` / `--baseline`
- For feature PRs, add any extra matrix row (e.g. `--features ndarray`)
- Put optional raw logs in a `<details>` block only if someone needs them

**Don’t:**

- Paste “Testing started at…”, full sample collection chatter, or
  hundreds of Criterion lines
- Use RustRover’s default bench runner flags (`--format=json`,
  `-Z unstable-options`, `--show-output`) — Criterion uses
  `harness = false` and rejects those args
- Claim “regressed” on paths the PR did not touch without a re-run

**Suggested comment titles (one comment each):**

1. `## Local verification (REVIEW.md)` — mandatory + edges + guards + examples  
2. `## Allocations smoke` — summary table only  
3. `## Criterion benches` (optional) — highlights table + verdict  
4. `## Follow-up` (optional) — only if you re-ran a suspicious filter  

**Template (copy/adapt):**

```markdown
## Local verification (REVIEW.md)

**Branch tip:** `<sha>`  
**Host:** local Linux (noisy).  
**Verdict:** All mandatory checks passed.

| Check | Result |
|-------|--------|
| `cargo fmt --check` | pass |
| `cargo test --locked` | pass |
| `cargo test --features serde --locked` | pass (8 serde tests) |
| `cargo clippy --all-features -- -D warnings` | pass |
| `RUSTDOCFLAGS='-D warnings' cargo doc --no-deps --all-features` | pass |
| `rng::tests` | pass |
| Edge filters | pass |
| Regression guards | pass |
| Examples | pass, no panics |
| Extra (if any) | e.g. `cargo test --features ndarray` pass |

## Allocations smoke

**Command:** `cargo bench --bench allocations`  
**Verdict:** Healthy / or call out issues.

| Encoder | @256 | @1k | @10k | Notes |
|---------|------|-----|------|--------|
| … | … | … | … | … |
```

### Examples (behavioral smoke)

```bash
cargo run --color=always --package axon-encoder \
  --example delta_encoding --profile dev
cargo run --color=always --package axon-encoder \
  --example embedding_encoding --profile dev
cargo run --color=always --package axon-encoder \
  --example latency_encoding --profile dev
cargo run --color=always --package axon-encoder \
  --example population_encoding --profile dev
cargo run --color=always --package axon-encoder \
  --example predictive_encoding --profile dev
cargo run --color=always --package axon-encoder \
  --example rate_encoding --profile dev
cargo run --color=always --package axon-encoder \
  --example spike_timebase --profile dev
cargo run --color=always --package axon-encoder \
  --example temporal_encoding --profile dev
```

When the PR adds or restores the `ndarray` feature:

```bash
cargo test --features ndarray --locked
cargo run --color=always --package axon-encoder \
  --example ndarray_encoding --features ndarray --profile dev
```

Each should print its encoder banner without panic.

## Regression guards (security / dependency PRs)

After any “security” or dependency PR, confirm product APIs from
PR #37 still exist:

```bash
# Gain-curve / neuromod stack
test "$(wc -l < src/modulators.rs)" -gt 400
rg -n 'pub struct GainCurve|NeuromodulatorGainCurves|EncodingGains' \
  src/modulators.rs

# encode_*_with_modulators lives on ModulatedEncoder, and every encoder
# implements it and overrides the allocation-free sink path
rg -n 'fn encode_with_modulators\(' src/lib.rs
for spec in \
  'delta.rs:DeltaEncoder' 'latency.rs:LatencyEncoder' 'phase.rs:PhaseEncoder' \
  'population.rs:PopulationEncoder' 'predictive.rs:PredictiveEncoder' \
  'rate.rs:RateEncoder' 'temporal.rs:TemporalEncoder'; do
  file="src/encoders/${spec%%:*}"
  encoder="${spec#*:}"
  rg -q "impl ModulatedEncoder for ${encoder}\b" "$file" || {
    echo "missing ModulatedEncoder impl: $encoder"; exit 1
  }
  rg -q '^\s*fn encode_with_modulators_into\(' "$file" || {
    echo "missing allocation-free modulator sink path: $encoder"; exit 1
  }
  rg -q '^\s*fn encode_step_with_modulators_into\(' "$file" || {
    echo "missing allocation-free modulator sink step: $encoder"; exit 1
  }
done
# The returning encode_*_with_modulators stay trait methods: no inherent
# wrappers back on the encoder types
test -z "$(rg -n 'fn encode_with_modulators\(|fn encode_step_with_modulators\(' \
  src/encoders/*.rs || true)"

# PhaseEncoder published
rg -n 'pub mod phase|pub use phase::PhaseEncoder' \
  src/encoders/mod.rs

# Serde coverage for gain / phase types
rg -n 'GainCurve|PhaseEncoder|NeuromodulatorGainCurves' \
  tests/serde_tests.rs

# Spike time contract published and enforced (#62 / RM-368)
rg -n 'pub struct TickOffset|pub struct Timebase|pub struct TimeModel' src/time.rs
rg -n 'fn time_model' src/lib.rs src/encoders/*.rs
cargo test --locked --test time_semantics

# EncodedOutput API retained; placeholders remain removed (#65 / LIM-976)
rg -n 'pub spikes: Vec<SpikeEvent>|pub embeddings: Option<Vec<f32>>' src/types.rs
! rg -n 'pub struct (EncoderConfig|EncodingMetadata)' src/types.rs
test -f tests/public_api.rs
cargo test --locked --test public_api
```

## Serde integration tests

`tests/serde_tests.rs` is gated with `#![cfg(feature = "serde")]`.

Without the feature you get **0 tests** (looks like "no tests found"):

```bash
# Wrong — compiles the harness but runs zero tests
cargo test --test serde_tests

# Right — 8 tests
cargo test --features serde --test serde_tests
cargo test --features serde --locked
```

## Diff hygiene

```bash
git fetch origin main
git diff --stat origin/main...HEAD
# Expect only intentional files; no mass deletions under modulators
test "$(wc -l < src/modulators.rs)" -gt 400
```

## Origin hygiene (never push local tooling)

These paths must stay untracked and ignored (aligned with `.gitignore`):

- `.worktrees/`
- `.swarm/`
- `.beads/`
- `.idea/`

```bash
git ls-files .worktrees .swarm .beads .idea   # must print nothing
git check-ignore -v .worktrees .swarm .beads .idea
```

## Do not merge if

- `src/modulators.rs` collapsed to a decay-only stub (~tens of lines)
- `PhaseEncoder` missing from `src/encoders/mod.rs`
- Net deletion of `ModulatedEncoder::encode_with_modulators` (or its impls on the main encoders)
- A bot “sync / resolve feedback” commit rewrites half the tree
  (thousands of lines deleted)
- `git diff origin/main` shows unexpected public-API removals

## Pass criteria

- All mandatory commands exit 0
- `cargo fmt --check` is silent (no output) with exit 0
- Examples print their encoder banners without panic
- Clippy reports zero warnings under `-D warnings`
- Serde feature tests pass (`--features serde`)
- Regression guards pass (no silent deletion of neuromod APIs)
- Diff hygiene: only intentional files for the PR
- Local tooling dirs are not in the commit
