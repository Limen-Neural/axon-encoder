# Architecture: `axon-encoder` ↔ `neuromod` siblings

Decision for [issue #21](https://github.com/Limen-Neural/axon-encoder/issues/21).

## Dependency rule

| Direction | Allowed? |
| --- | --- |
| `axon-encoder` → `neuromod` | **Forbidden** |
| `neuromod` → `axon-encoder` | **Forbidden** |
| App / adapter → both | **Required** for integration |

```text
              application / adapter
               ↙             ↘
         neuromod          axon-encoder
```

## Ownership

### `axon-encoder`

- `Encoder`, `ModulatedEncoder`
- Signal-to-spike algorithms (rate, latency, population, …)
- Generic `EncodingGains` (threshold / sensitivity / firing-rate / latency scales)
- Spike event output (`EncodedOutput`, `SpikeEvent`)

### `neuromod`

- Biological modulator state and dynamics (e.g. dopamine, cortisol, ACh)
- Neuron models, STDP-style plasticity building blocks
- Simulation-oriented neuromodulation APIs

### Downstream adapter

- Maps neuromodulator (or other policy) state into `EncodingGains`
- Chooses when / how encoding scales change
- Never forces either library crate to import the other

## Audit: biologically named types in `axon-encoder`

These names are **kept** in `axon-encoder` as **encoding-side** helpers, not as a
dependency on the `neuromod` crate:

| Type | Decision | Rationale |
| --- | --- | --- |
| `EncodingGains` | **Retain** (core) | Generic encoder scales; framework-neutral |
| `GainCurve` / `ModulatorGainCurves` | **Retain** | Piecewise maps level → scale for encoding |
| `NeuroModulators` | **Retain** (encoding adapter) | Lightweight 4-scalar bag + decay for gain evaluation **inside this crate only**; not the neuromod runtime |
| `NeuromodulatorGainCurves` | **Retain** (encoding adapter) | Composes per-channel curves into `EncodingGains` |

They do **not** re-export traits through `neuromod`, and `neuromod` must not
re-export `Encoder` / `ModulatedEncoder` as its own API surface.

## Integration example

See [`examples/sibling_gains_adapter.rs`](../examples/sibling_gains_adapter.rs):
an app-local `AppModState` is converted to `EncodingGains` without importing
`neuromod`.
