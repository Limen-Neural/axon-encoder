# Serde JSON compatibility goldens

These hand-checked payloads anchor field layouts from the `v0.3.0`, `v0.4.0`,
and `v0.5.0` Git tags. They were reconstructed from those tags' serde derives,
helper structs, and validation, not automatically regenerated from HEAD.
`v0.5/` covers **all 20 current public serde types**; the current reader loads
each and the current writer is checked against the same JSON. `head/` adds
partial, extended, invalid, and rate-state edge cases. Tests compare JSON values
(ignoring whitespace/key order), except the historical `SpikeEvent` test also
checks the identical compact JSON bytes. Integers stay integers, including a
`TickOffset` greater than 2^53; fixtures must not be passed through a JavaScript
number formatter.

## Version provenance

At the 2026-09-15 audit commit `f3e375fed5d10ce4e97506138a9d2a32599b51fc`,
**Cargo.toml still said `0.4.0` while main already carried 0.5-line wire breaks**
(owned embedding membrane state, transparent tick types, output placeholder
removal, and stricter validation). A manifest version alone was not a schema
version. At implementation time (2026-10-02), main's manifest says `0.6.0` and
`v0.5.0` is tagged; the 20 v0.5 schemas below still describe HEAD. This work
does not bump the crate version or introduce a schema marker.

Source layouts: `src/encoders/rate.rs`, `src/encoder.rs`, `src/types.rs`,
`src/time.rs`, `src/modulators.rs`, `src/poisson.rs`, and each concrete encoder
under `src/encoders/`. The simulated v0.3 reader uses the exact five rate fields
from that tag (`accumulators: Vec<f32>`); it is not an old crate dependency.

## Reader/writer matrix

All paths below are relative to this directory. HEAD means the checked-out
reader/writer, not a promise that future incompatible versions must load.

| Payload / writer | Reader | Locked outcome |
| --- | --- | --- |
| `v0.3/rate_combined.json` | HEAD | Missing `dt_seconds` becomes 0.1; `[2.25,0.5,4.0]` splits into phases `[0.25,0.5,0.0]` and debt `[2,0,4]`. |
| `v0.4/rate_pending.json`, `head/rate_pending.json` | simulated v0.3 | Loads while ignoring new fields; re-saving **loses debt `[3,7]`** and custom dt 0.125. Loading with HEAD retains debt. |
| `v0.4/embedding_rate.json` | HEAD | Reject: `membrane_potentials` required; `normalized_embeddings` is not migrated. |
| `v0.4/spike_event.json` | HEAD | Identical compact JSON; the former `u64` is now transparent `TickOffset`. |
| `v0.4/latency_max.json` | HEAD | Reject `u64::MAX` with `WindowTooLarge` (accepted by v0.4 validation). |
| `v0.4/encoded_output.json` | HEAD | Loads; removed empty `metadata` placeholder is discarded. |
| `head/timebase_zero.json` | HEAD | Reject zero tick duration. |
| `head/spike_event_extra.json`, `head/poisson_extra.json` | HEAD | Unknown fields accepted and discarded, not rejected. |
| `head/encoding_gains_empty.json` | HEAD | `{}` defaults to identity. |
| `head/encoding_gains_unsanitized.json` | HEAD | Negative threshold becomes 0, sensitivity caps at 10,000, missing rate becomes 1, explicit zero latency stays zero. |
| `head/neuromodulators_empty.json` | HEAD | Reject missing required fields; Rust `Default` does not imply serde defaults. |
| `head/rate_first_step.json` | HEAD | First `encode_step(&[0.25])` at dt 0.125 writes phase `[0.3125]` and **`pending_spikes: [0]`**, not an omitted field. |
| Every `v0.5/*.json` | HEAD | Load into independently constructed values; HEAD serialization matches golden JSON. |

The v0.5 inventory is `SpikeEvent`, `EncodedOutput`, `TickOffset`, `Timebase`,
`EmbeddingEncoderConfig`, `EmbeddingRateEncoder`, `RateEncoder`, `DeltaEncoder`,
`DerivativeEncoder`, `LatencyEncoder`, `PhaseEncoder`, `PopulationEncoder`,
`PoissonEncoder`, `PredictiveEncoder`, `TemporalEncoder`, `NeuroModulators`,
`GainCurve`, `EncodingGains`, `ModulatorGainCurves`, and `NeuromodulatorGainCurves`.
There is no current serde API for removed `EncoderState`, `EncoderConfig`, or
`EncodingMetadata`; the old output fixture captures metadata's removal.

## Scope and maintenance

**Stochastic encoders are config-only here.** Population batch encoding,
Poisson trains, and Rate batch encoding do not persist a random stream in these
payloads. These tests never generate stochastic trains or use RNG injection
and do not claim spike-train replay. Live deterministic checkpoint continuation
belongs to LIM-1233's `tests/checkpoint_continuation.rs`, not this matrix.

Run `cargo test --features serde --locked serde_compat`. Failures include the
fixture file and writer/reader pair. Full verification:

```bash
cargo test --all-features --locked
cargo clippy --all-targets --all-features -- -D warnings
```

Do not overwrite historical goldens to make a schema change pass. Add a new
version-labelled payload and explicitly update the matrix's load, reject, or
loss decision under the schema-versioning work. JSON only is intentional;
bincode compatibility is not tested or claimed.
