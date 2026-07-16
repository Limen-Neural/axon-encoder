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

### Core locked test matrix

```bash
cargo test --locked
cargo test --features serde --locked
cargo clippy --all-features -- -D warnings
cargo test --package axon-encoder --lib -- rng::tests
cargo test --message-format=json-diagnostic-rendered-ansi --color=always --no-run --package axon-encoder --lib --profile test
```

### Benchmarks (smoke)

```bash
cargo bench
```

### Examples (behavioral smoke)

```bash
cargo run --color=always --package axon-encoder --example delta_encoding --profile dev
cargo run --color=always --package axon-encoder --example latency_encoding --profile dev
cargo run --color=always --package axon-encoder --example population_encoding --profile dev
cargo run --color=always --package axon-encoder --example predictive_encoding --profile dev
cargo run --color=always --package axon-encoder --example rate_encoding --profile dev
cargo run --color=always --package axon-encoder --example temporal_encoding --profile dev
```

## Regression guards (security PRs)

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

## Origin hygiene (never push local tooling)

These paths must stay untracked and ignored:

- `.worktrees/`
- `.swarm/`
- `.junie/`
- `.idea/`

```bash
git ls-files .worktrees .swarm .junie .idea   # must print nothing
git check-ignore -v .worktrees .swarm .junie .idea
```

## Pass criteria

- All mandatory commands exit 0
- Examples print their encoder banners without panic
- Clippy reports zero warnings under `-D warnings`
- Serde feature tests pass
- Regression guards pass (no silent deletion of neuromod / PhaseEncoder APIs)
- `git diff origin/main` has no unexpected public-API removals
- Local tooling dirs are not in the commit
