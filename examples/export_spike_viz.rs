//! spike-viz export writer: drives a real encoder over a shared stimulus and
//! writes the contract staging layout that `scripts/pack_axon_export.py` in
//! rmems/spike-viz assembles into `spikes.npz` + `meta.json`.
//!
//! ```text
//! cargo run --example export_spike_viz -- <encoder> <out_dir>
//! ```
//!
//! `<encoder>` is one of `rate`, `poisson`, `latency`, `population`,
//! `directional_population`, `temporal`, `predictive`. `<out_dir>` receives
//! `t.npy`, `neuron_id.npy`, `amp.npy`, `stimulus.npy`, and `meta.json`.
//!
//! Provenance: `AXON_ENCODER_GIT_SHA` (set by the generation script) is recorded
//! verbatim in `meta.json`. Stochastic encoders draw from a seeded `StdRng`
//! through their `*_with_rng` public surfaces, so the output is bit-for-bit
//! reproducible for a fixed seed.
//!
//! Time semantics: `SpikeEvent::timestamp` is a call-relative `TickOffset`;
//! every event is exported at `TimeCursor::absolute(offset)`, i.e. an absolute
//! encoder tick. `dt_seconds` is the exporter's sampling convention: encoders
//! with a `Timebase` carry a physical tick, dimensionless encoders do not —
//! the declared `dt` is what a consumer should render, not an encoder claim.

use std::env;
use std::fs;
use std::path::Path;

use axon_encoder::prelude::*;
use rand::SeedableRng;
use rand::rngs::StdRng;

/// Number of `encode_step` calls per case (presentation count for windowed
/// encoders; the flat `n_steps` covers `calls * span_ticks`).
const CALLS: usize = 64;
/// Input channels per step.
const CHANNELS: usize = 8;
/// Export sampling convention in seconds (also RateEncoder's real `dt`).
const DT_SECONDS: f64 = 0.001;
/// Single seed for all stochastic paths.
const SEED: u64 = 0x5EED_0001;
/// Environment variable carrying the exporting commit's SHA (set by the
/// generation script).
const GIT_SHA_ENV: &str = "AXON_ENCODER_GIT_SHA";

/// Shared normalized stimulus: `stimulus[call][channel]`, a deterministic
/// mixed sine/ramp. Directional encoding scales these samples by 2π before use.
fn stimulus() -> Vec<Vec<f32>> {
    (0..CALLS)
        .map(|t| {
            (0..CHANNELS)
                .map(|ch| {
                    let phase = (t as f32 / CALLS as f32) * std::f32::consts::TAU;
                    let offset = ch as f32 / CHANNELS as f32;
                    (0.5 + 0.4 * (phase * (1.0 + offset)).sin() + 0.1 * offset).clamp(0.0, 1.0)
                })
                .collect()
        })
        .collect()
}

struct Events {
    t: Vec<i64>,
    neuron_id: Vec<i64>,
    amp: Vec<f32>,
}

impl Events {
    fn new() -> Self {
        Self {
            t: Vec::new(),
            neuron_id: Vec::new(),
            amp: Vec::new(),
        }
    }

    /// Appends one call's spikes at their absolute ticks.
    fn extend(&mut self, out: &EncodedOutput, cursor: TimeCursor) {
        for spike in &out.spikes {
            self.t.push(cursor.absolute(spike.timestamp) as i64);
            self.neuron_id.push(i64::from(spike.channel));
            self.amp.push(if spike.polarity { 1.0 } else { -1.0 });
        }
    }
}

/// Drives any `Encoder` for `stimulus.len()` calls through a `TimeCursor`.
fn drive(encoder: &mut dyn Encoder, stimulus: &[Vec<f32>]) -> (Events, u64) {
    let model = encoder.time_model();
    let mut cursor = TimeCursor::new(model);
    let mut events = Events::new();
    for step in stimulus {
        events.extend(&encoder.encode_step(step), cursor);
        cursor.advance();
    }
    // Total exported tick span: origin after all calls, plus one window's
    // slack so a windowed/overlapping encoder's last-call offsets stay in
    // bounds.
    let n_steps = cursor.origin().max(model.span_ticks());
    (events, n_steps)
}

struct Case {
    events: Events,
    n_neurons: usize,
    n_steps: u64,
    config: serde_json::Value,
    stimulus: Vec<Vec<f32>>,
    preferred_angles_radians: Option<Vec<f32>>,
}

impl Case {
    fn new(events: Events, n_neurons: usize, n_steps: u64, config: serde_json::Value) -> Self {
        Self {
            events,
            n_neurons,
            n_steps,
            config,
            stimulus: Vec::new(),
            preferred_angles_radians: None,
        }
    }

    fn with_stimulus(mut self, stimulus: &[Vec<f32>]) -> Self {
        self.stimulus = stimulus.to_vec();
        self
    }

    fn with_preferred_angles(mut self, preferred_angles_radians: Vec<f32>) -> Self {
        self.preferred_angles_radians = Some(preferred_angles_radians);
        self
    }
}

/// Encodes `name` over the shared stimulus and returns the case to export.
fn generate(name: &str, stim: &[Vec<f32>], rng: &mut StdRng) -> Case {
    match name {
        // Deterministic: the streaming path accumulates expected spikes in f64
        // phase state — no RNG involved.
        "rate" => {
            let mut enc = RateEncoder::try_new(50.0, 400.0, (0.0, 1.0), DT_SECONDS as f32)
                .expect("rate config");
            let (events, n_steps) = drive(&mut enc, stim);
            Case::new(
                events,
                CHANNELS,
                n_steps,
                serde_json::json!({"base_rate_hz": 50.0, "max_rate_hz": 400.0, "range": [0.0, 1.0]}),
            )
            .with_stimulus(stim)
        }
        "poisson" => poisson_case(stim, rng),
        // Deterministic; window model — absolute ticks advance by span per call.
        "latency" => {
            let mut enc = LatencyEncoder::new(10, (0.0, 1.0));
            let (events, n_steps) = drive(&mut enc, stim);
            Case::new(
                events,
                CHANNELS,
                n_steps,
                serde_json::json!({"max_latency": 10, "range": [0.0, 1.0]}),
            )
            .with_stimulus(stim)
        }
        "population" => population_case(stim, rng),
        "directional_population" => directional_population_case(stim, rng),
        // Deterministic threshold crossings over a rolling window.
        "temporal" => {
            let mut enc = TemporalEncoder::new(8, vec![0.05, 0.15, 0.30], CHANNELS);
            let (events, n_steps) = drive(&mut enc, stim);
            Case::new(
                events,
                CHANNELS,
                n_steps,
                serde_json::json!({"history_depth": 8, "change_thresholds": [0.05, 0.15, 0.30]}),
            )
            .with_stimulus(stim)
        }
        // Deterministic deviation predictions over a rolling window.
        "predictive" => {
            let mut enc =
                PredictiveEncoder::new(8, vec![(0.05, 1), (0.15, 2), (0.30, 3)], CHANNELS)
                    .expect("predictive config");
            let (events, n_steps) = drive(&mut enc, stim);
            Case::new(
                events,
                CHANNELS,
                n_steps,
                serde_json::json!({"history_depth": 8, "deviation_thresholds": [[0.05, 1], [0.15, 2], [0.30, 3]]}),
            )
            .with_stimulus(stim)
        }
        other => {
            eprintln!("unknown encoder {other:?}");
            std::process::exit(2);
        }
    }
}

/// One seeded Bernoulli draw per (channel, step): the channel's rate at
/// step `t` follows the same time-varying stimulus every encoder consumes.
/// Draw order is channel-major, front to back (documented for reproducibility).
fn poisson_case(stim: &[Vec<f32>], rng: &mut StdRng) -> Case {
    let enc = PoissonEncoder::new(CALLS);
    let mut events = Events::new();
    for ch in 0..CHANNELS {
        for (step, row) in stim.iter().enumerate() {
            let rate_hz = 50.0 + 300.0 * row[ch];
            if enc.encode_rate_hz_step_with_rng(rate_hz, DT_SECONDS as f32, rng) != 0 {
                events.t.push(step as i64);
                events.neuron_id.push(ch as i64);
                events.amp.push(1.0);
            }
        }
    }
    Case::new(
        events,
        CHANNELS,
        CALLS as u64,
        serde_json::json!({"channel_rates_hz": "50 + 300 * stimulus[t][ch]"}),
    )
    .with_stimulus(stim)
}

/// Seeded RNG through the public surface; output channels fan out to
/// inputs × tuned neurons.
fn population_case(stim: &[Vec<f32>], rng: &mut StdRng) -> Case {
    const NEURONS_PER_INPUT: usize = 4;
    let mut enc = PopulationEncoder::new(NEURONS_PER_INPUT, (0.0, 1.0), 0.15);
    let model = enc.time_model();
    let mut cursor = TimeCursor::new(model);
    let mut events = Events::new();
    for step in stim {
        events.extend(&enc.encode_step_with_rng(step, rng), cursor);
        cursor.advance();
    }
    Case::new(
        events,
        CHANNELS * NEURONS_PER_INPUT,
        cursor.origin().max(model.span_ticks()),
        serde_json::json!({"num_neurons": NEURONS_PER_INPUT, "input_range": [0.0, 1.0], "tuning_width": 0.15}),
    )
    .with_stimulus(stim)
}

/// Directional angles are the shared normalized stimulus scaled by 2π. The
/// exported angles repeat per input channel because neuron IDs fan out as
/// `input_channel * NEURONS_PER_INPUT + preferred_angle_index`.
fn directional_population_case(stim: &[Vec<f32>], rng: &mut StdRng) -> Case {
    const NEURONS_PER_INPUT: usize = 8;
    let angular_stim: Vec<Vec<f32>> = stim
        .iter()
        .map(|row| {
            row.iter()
                .map(|value| value * std::f32::consts::TAU)
                .collect()
        })
        .collect();

    let mut enc = DirectionalPopulationEncoder::new(NEURONS_PER_INPUT, 0.35);
    let model = enc.time_model();
    let mut cursor = TimeCursor::new(model);
    let mut events = Events::new();
    for step in &angular_stim {
        events.extend(&enc.encode_step_with_rng(step, rng), cursor);
        cursor.advance();
    }

    let one_population = enc.preferred_angles_radians();
    let preferred_angles: Vec<f32> = (0..CHANNELS)
        .flat_map(|_| one_population.iter().copied())
        .collect();
    Case::new(
        events,
        CHANNELS * NEURONS_PER_INPUT,
        cursor.origin().max(model.span_ticks()),
        serde_json::json!({
            "num_neurons": NEURONS_PER_INPUT,
            "tuning_width_radians": 0.35,
            "input_units": "radians",
            "angle_convention": "zero east, counterclockwise positive; input is the shared normalized stimulus scaled by 2π"
        }),
    )
    .with_stimulus(&angular_stim)
    .with_preferred_angles(preferred_angles)
}

fn write_outputs(out_dir: &Path, case: &Case) {
    let stim = case.stimulus.as_slice();
    write_npy(&out_dir.join("t.npy"), &NpyArray::I64(&case.events.t));
    write_npy(
        &out_dir.join("neuron_id.npy"),
        &NpyArray::I64(&case.events.neuron_id),
    );
    write_npy(&out_dir.join("amp.npy"), &NpyArray::F32(&case.events.amp));
    // Window models advance several ticks per call, so repeat each call's row
    // to keep stimulus rows tick-aligned with spike timestamps.
    let ticks_per_call = (case.n_steps as usize / stim.len().max(1)).max(1);
    let stim_flat: Vec<f32> = stim
        .iter()
        .flat_map(|row| row.iter().copied().cycle().take(row.len() * ticks_per_call))
        .collect();
    write_npy2d(
        &out_dir.join("stimulus.npy"),
        &stim_flat,
        CALLS * ticks_per_call,
        CHANNELS,
    );
}

fn write_meta(out_dir: &Path, name: &str, case: &Case) {
    let mut meta = serde_json::json!({
        "schema_version": "1.0",
        "encoder": name,
        "dt_seconds": DT_SECONDS,
        "seed": SEED,
        "n_neurons": case.n_neurons,
        "n_steps": case.n_steps,
        "axon_encoder_git_sha": env::var(GIT_SHA_ENV).ok(),
        "axon_encoder_version": env!("CARGO_PKG_VERSION"),
        "synthetic": false,
        "encoder_config": case.config,
        "stimulus_notes": "Shared deterministic sine/ramp stimulus, 8 input channels; for directional_population the exported stimulus is scaled by 2π into radians.",
        "notes": "Generated by axon-encoder `export_spike_viz` example; dt_seconds is the export sampling convention when the encoder reports no Timebase.",
    });
    if let Some(angles) = &case.preferred_angles_radians {
        meta["preferred_angles_radians"] = serde_json::json!(angles);
    }
    fs::write(
        out_dir.join("meta.json"),
        serde_json::to_string_pretty(&meta).expect("meta serialization"),
    )
    .expect("write meta.json");
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() != 3 {
        eprintln!("usage: export_spike_viz <encoder> <out_dir>");
        eprintln!(
            "encoders: rate poisson latency population directional_population (directional) temporal predictive"
        );
        std::process::exit(2);
    }
    let name = match args[1].as_str() {
        "directional" => "directional_population",
        other => other,
    };
    let out_dir = Path::new(&args[2]);
    fs::create_dir_all(out_dir).expect("create output directory"); // skipcq: RS-E1015

    let stim = stimulus();
    let mut rng = StdRng::seed_from_u64(SEED);
    let case = generate(name, &stim, &mut rng);

    write_outputs(out_dir, &case); // skipcq: RS-E1015
    write_meta(out_dir, name, &case); // skipcq: RS-E1015

    println!(
        "{name}: {} events, N={}, T={} -> {}",
        case.events.t.len(),
        case.n_neurons,
        case.n_steps,
        out_dir.display()
    );
}

enum NpyArray<'a> {
    I64(&'a [i64]),
    F32(&'a [f32]),
}

/// Minimal NumPy `.npy` v1.0 writer (little-endian, C-order, 1-D).
fn write_npy(path: &Path, array: &NpyArray<'_>) {
    let (descr, len, data): (&str, usize, Vec<u8>) = match array {
        NpyArray::I64(v) => (
            "<i8",
            v.len(),
            v.iter().flat_map(|x| x.to_le_bytes()).collect(),
        ),
        NpyArray::F32(v) => (
            "<f4",
            v.len(),
            v.iter().flat_map(|x| x.to_le_bytes()).collect(),
        ),
    };
    write_npy_impl(path, descr, &[len], &data);
}

/// 2-D float32 variant for `stimulus.npy` (`[rows, cols]`).
fn write_npy2d(path: &Path, data: &[f32], rows: usize, cols: usize) {
    let bytes: Vec<u8> = data.iter().flat_map(|x| x.to_le_bytes()).collect();
    write_npy_impl(path, "<f4", &[rows, cols], &bytes);
}

fn write_npy_impl(path: &Path, descr: &str, shape: &[usize], data: &[u8]) {
    let shape_str = if shape.len() == 1 {
        format!("({},)", shape[0])
    } else {
        format!(
            "({})",
            shape
                .iter()
                .map(|d| d.to_string())
                .collect::<Vec<_>>()
                .join(", ")
        )
    };
    let dict = format!("{{'descr': '{descr}', 'fortran_order': False, 'shape': {shape_str}, }}");
    // v1.0 header: magic(6) + ver(2) + hlen(2) + dict, padded to 64-byte align
    // and terminated with '\n'.
    let pad = (64 - ((10 + dict.len() + 1) % 64)) % 64;
    let header = format!("{dict}{}\n", " ".repeat(pad));
    let mut bytes = Vec::with_capacity(10 + header.len() + data.len());
    bytes.extend_from_slice(b"\x93NUMPY");
    bytes.extend_from_slice(&[1, 0]);
    bytes.extend_from_slice(&(header.len() as u16).to_le_bytes());
    bytes.extend_from_slice(header.as_bytes());
    bytes.extend_from_slice(data);
    fs::write(path, bytes).expect("write npy");
}
