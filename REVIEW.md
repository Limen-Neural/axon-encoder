# Local review quality gate

These commands are the **human quality bar** beyond GitHub Actions. Run them
before claiming a PR is ready when the change touches `src/`, `Cargo.toml`,
public APIs, or randomness.

They match the suite used to sign off PR #37 (rmems, 2026-07-15) and are
**required** for security-oriented PRs such as PR #50 so merge thrash cannot
land “green CI” while deleting product surface.

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
cargo test --message-format=json-diagnostic-rendered-ansi --color=always --no-run --package axon-encoder --lib --profile test
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
cargo bench
```

**How to read results**

- Prefer Criterion **change %** over absolute ns (machine-dependent).
- “Change within noise threshold” / low-single-digit % is usually fine.
- Real regressions are multi‑× slowdowns or consistent multi‑percent hits across scales.
- Optional for the PR comment: paste the allocation CSV header + a few key Criterion lines
  (Rate / Population / Poisson).

### Examples (behavioral smoke)

```bash
cargo run --color=always --package axon-encoder --example delta_encoding --profile dev
cargo run --color=always --package axon-encoder --example latency_encoding --profile dev
cargo run --color=always --package axon-encoder --example population_encoding --profile dev
cargo run --color=always --package axon-encoder --example predictive_encoding --profile dev
cargo run --color=always --package axon-encoder --example rate_encoding --profile dev
cargo run --color=always --package axon-encoder --example temporal_encoding --profile dev
```

Each should print its encoder banner without panic.

## Regression guards (security / dependency PRs)

After any “security” or dependency PR, confirm product APIs from PR #37 still exist:

```bash
# Gain-curve / neuromod stack
test "$(wc -l < src/modulators.rs)" -gt 400
rg -n 'pub struct GainCurve|NeuromodulatorGainCurves|EncodingGains' src/modulators.rs

# encode_*_with_modulators on encoders (including PhaseEncoder)
rg -n 'fn encode_with_modulators' src/encoders/*.rs

# PhaseEncoder published
rg -n 'pub mod phase|pub use phase::PhaseEncoder' src/encoders/mod.rs

# Serde coverage for gain / phase types
rg -n 'GainCurve|PhaseEncoder|NeuromodulatorGainCurves' tests/serde_tests.rs
```

## Diff hygiene

```bash
git fetch origin main
git diff --stat origin/main...HEAD
# Expect only intentional files; no mass deletions under src/modulators.rs
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
- A bot “sync / resolve feedback” commit rewrites half the tree (−thousands of lines)
- `git diff origin/main` shows unexpected public-API removals

## Pass criteria

- All mandatory commands exit 0
- `cargo fmt --check` is silent (no output) with exit 0
- Examples print their encoder banners without panic
- Clippy reports zero warnings under `-D warnings`
- Serde feature tests pass
- Regression guards pass (no silent deletion of neuromod / PhaseEncoder APIs)
- Diff hygiene: only intentional files for the PR
- Local tooling dirs are not in the commit
