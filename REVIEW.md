# Local review quality gate

These commands are the **human quality bar** beyond GitHub Actions.
Run them before claiming a PR is ready when the change touches `src/`,
`Cargo.toml`, public APIs, or randomness.

They match the suite used to sign off PR #37 (rmems, 2026-07-15) and are
**required** for security-oriented PRs such as PR #50 so merge thrash
cannot land “green CI” while deleting product surface.

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
  --example latency_encoding --profile dev
cargo run --color=always --package axon-encoder \
  --example population_encoding --profile dev
cargo run --color=always --package axon-encoder \
  --example predictive_encoding --profile dev
cargo run --color=always --package axon-encoder \
  --example rate_encoding --profile dev
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

# encode_*_with_modulators (including PhaseEncoder)
rg -n 'fn encode_with_modulators' src/encoders/*.rs

# PhaseEncoder published
rg -n 'pub mod phase|pub use phase::PhaseEncoder' \
  src/encoders/mod.rs

# Serde coverage for gain / phase types
rg -n 'GainCurve|PhaseEncoder|NeuromodulatorGainCurves' \
  tests/serde_tests.rs
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
- Net deletion of `encode_with_modulators` on the main encoders
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
