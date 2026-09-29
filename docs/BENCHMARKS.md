# Benchmark guide

The repository has two complementary benchmarks:

```text
cargo bench --bench encoders
cargo bench --bench allocations
```

`encoders` is a Criterion benchmark for throughput. It reports separate groups
for encoder, operation, and channel/population scale. It also includes
caller-owned `SpikeSink` variants, so compare a returning operation only with
its like-for-like `_into` counterpart when investigating allocation reuse.

`allocations` prints CSV-like rows with `encoder, operation, scale type, scale,
allocations, bytes, spikes`. Returning-path rows measure the operation as
configured, including any first-use initialization; they are not universally
warmed steady-state measurements. The reusable-sink rows deliberately pre-size
and warm their buffers; a zero there means the measured operation did not grow
its buffer, not that an end-to-end application has no allocations. Always
inspect the `spikes` column: a row that emits no spikes is not evidence of
efficient spike delivery.

## Make comparisons meaningful

Do not rank encoders against each other from their headline numbers. They do
different work: `PopulationEncoder` scales by neurons for one value,
`LatencyEncoder` maps a channel vector into a window, and history-based
encoders process changing input sequences. The benchmark inputs and warm-up
rules are part of each workload.

For a useful comparison:

1. Compare the same encoder, operation, input scale, and feature set before
   and after a change.
2. Keep the compiler, CPU power policy, and machine load stable; run several
   times and look for a persistent trend rather than a single estimate.
3. Compare `encode_step` with `encode_step_into` only when the same emitted
   work is being performed; report spike counts alongside allocation results.
4. For a new production workload, add a workload-specific benchmark instead
   of treating these representative inputs as cross-encoder parity.

The stochastic batch paths (`RateEncoder::encode`, `PopulationEncoder`, and
`PoissonEncoder`) can vary in spike count and timing between runs. Measure
distributions or repeated runs where that variability matters; do not infer
replayability or deterministic equivalence from a benchmark result.
