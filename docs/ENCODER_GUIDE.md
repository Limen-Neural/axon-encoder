# Encoder guide

This guide chooses and operates the public `axon-encoder` 0.5 API for a
real-time pipeline. Start with `try_new`, keep the encoder alive for the life
of a stream, and call `encode_step` once per incoming frame.

## Choose an encoder

| Need | Encoder | Streaming consideration |
| --- | --- | --- |
| Intensity represented as rate | `RateEncoder` | `encode_step` integrates rate deterministically. Batch `encode` is stochastic. |
| Change from a baseline | `DeltaEncoder` | Keeps prior values; retain it across frames. |
| Change velocity | `DerivativeEncoder` | Keeps prior values; use finite input when checkpointing to JSON. |
| Earlier event for stronger input | `LatencyEncoder` | Stateless; one call can span `max_latency + 1` ticks. |
| Periodic phase code | `PhaseEncoder` | Calls overlap its cycle window; use its reported time model. |
| Tuning-curve population code | `PopulationEncoder` | Batch and streaming paths are stochastic. |
| Temporal pattern or prediction | `TemporalEncoder`, `PredictiveEncoder` | Keep state across frames. |
| Thresholded embedding drive | `EmbeddingRateEncoder` | Keeps per-channel membrane potentials. |

`PoissonEncoder` is a standalone spike-train generator rather than an
`Encoder` implementation. It provides its own `encode_step` and
`encode_rate_hz_step` streaming methods, but has no `SpikeSink` path.

## Batch and streaming are separate call styles

`encode(&input)` consumes one call-sized window and returns a fresh
`EncodedOutput`. `encode_step(&frame)` consumes exactly one streaming frame;
it is not a partial call to `encode`. Stateful encoders generally advance their
history, phase, or membrane in both modes. `RateEncoder` is deliberately
different: batch `encode` makes an independent stochastic draw and does not
advance the deterministic streaming accumulator used by `encode_step`.

For a hot sensor loop, reuse storage with `encode_step_into`:

```rust
use axon_encoder::prelude::*;

fn main() -> Result<(), EncoderError> {
    let mut encoder = DeltaEncoder::try_new(0.1, 3)?;
    let mut spikes = Vec::<SpikeEvent>::new();
    for frame in [[0.0, 0.0, 0.0], [0.0, 0.3, 0.0]] {
        spikes.clear();
        encoder.encode_step_into(&frame, &mut spikes);
        // Send `spikes` downstream before the next frame.
    }
    Ok(())
}
```

Encoders append to a `SpikeSink`; clearing it is the caller's step-boundary
policy. A custom queue or hardware adapter can implement `SpikeSink` to avoid
building a `Vec<SpikeEvent>` altogether. See
[`examples/streaming_sensor.rs`](../examples/streaming_sensor.rs) for a full
loop.

## Time is call-relative

Each `SpikeEvent::timestamp` is a `TickOffset` measured from the start of the
call that produced it. It is not wall-clock time and it is not a globally
increasing counter. Keep the absolute origin in a `TimeCursor`:

```rust
use axon_encoder::prelude::*;

fn main() -> Result<(), EncoderError> {
    let mut encoder = LatencyEncoder::try_new(9, (0.0, 1.0))?;
    let mut cursor = TimeCursor::new(encoder.time_model());
    let output = encoder.encode_step(&[0.9]);
    for spike in output.spikes {
        let absolute_tick = cursor.absolute(spike.timestamp);
        println!("{absolute_tick}");
    }
    cursor.advance(); // advances by time_model().step_ticks()
    Ok(())
}
```

Use `encoder.time_model()` rather than assuming a one-tick call. `span_ticks`
is the exclusive offset bound for a call, while `step_ticks` is how far the
cursor advances after that call. An encoder with a `Timebase` can also use
`cursor.absolute_nanos(offset)`; dimensionless encoders require the caller to
declare a `Timebase` before physical-time conversion.

## Modulation

Use `EncodingGains` when your application already computes generic scales.
Use `NeuroModulators` plus `NeuromodulatorGainCurves` when a named modulator
bag is convenient. The curves are a mapping policy owned by the application,
not an external neuromodulator runtime or an asserted biological model.

```rust
use axon_encoder::prelude::*;

fn main() -> Result<(), EncoderError> {
    let mut encoder = RateEncoder::try_new(0.0, 100.0, (0.0, 1.0), 0.01)?;
    let modulators = NeuroModulators { dopamine: 0.5, ..Default::default() };
    let curves = NeuromodulatorGainCurves {
        dopamine: ModulatorGainCurves {
            firing_rate: Some(GainCurve::new((0.0, 1.0), (1.0, 1.5))),
            ..Default::default()
        },
        ..Default::default()
    };
    let output = encoder.encode_step_with_modulators(&[0.8], &modulators, &curves);
    println!("{} spikes", output.spikes.len());
    Ok(())
}
```

See [`examples/neuromodulated_encoding.rs`](../examples/neuromodulated_encoding.rs).

## Reset and checkpoints

Call `reset()` before an independent trial when prior stream state must not
affect it. Do not reset between frames of one continuous stream.

With the `serde` feature, encoder configuration and live state are serializable.
Restoring a finite JSON checkpoint continues deterministic `encode_step` paths
exactly. It does **not** make a stochastic path replayable: batch
`RateEncoder::encode` and both `PopulationEncoder` and `PoissonEncoder` modes
create thread-local RNG state internally, and the public API does not accept a
caller-owned RNG. Treat their outputs as fresh draws after restore.

Run the examples with the default feature set:

```text
cargo run --example streaming_sensor
cargo run --example neuromodulated_encoding
```
