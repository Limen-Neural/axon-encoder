//! Downstream adapter: map application modulator-like state into
//! [`EncodingGains`] without depending on the `neuromod` crate.
//!
//! Architecture (#21):
//!
//! ```text
//!              application / adapter  (this example)
//!               ↙             ↘
//!         neuromod          axon-encoder
//! ```
//!
//! Run: `cargo run --example sibling_gains_adapter`

use axon_encoder::prelude::*;

/// Stand-in for state that might live in an app or in the neuromod crate.
/// Kept local so this example has zero cross-crate deps.
#[derive(Clone, Copy, Debug, Default)]
struct AppModState {
    dopamine: f32,
    cortisol: f32,
}

/// Pure adapter: biological (or other) scalars → generic encoding gains.
fn gains_from_app_state(state: AppModState) -> EncodingGains {
    // Example policy only — real apps own this mapping.
    let firing = (1.0 + state.dopamine).clamp(0.0, 4.0);
    let threshold = (1.0 / (1.0 + state.cortisol)).clamp(0.1, 2.0);
    EncodingGains {
        firing_rate_scale: firing,
        threshold_scale: threshold,
        sensitivity_scale: 1.0,
        latency_scale: 1.0,
    }
    .sanitize()
}

fn main() -> Result<(), EncoderError> {
    let mut encoder = RateEncoder::try_new(5.0, 50.0, (0.0, 1.0), 0.01)?;
    let state = AppModState {
        dopamine: 0.5,
        cortisol: 0.2,
    };
    let gains = gains_from_app_state(state);

    // Apply gains without NeuroModulators / NeuromodulatorGainCurves if desired.
    let out = encoder.encode_with_gains(&[0.8], gains);
    println!(
        "adapter gains: firing_rate_scale={}, threshold_scale={}; spikes={}",
        gains.firing_rate_scale,
        gains.threshold_scale,
        out.spikes.len()
    );

    // Optional: same crate's gain-curve path (still no neuromod dependency).
    let curves = NeuromodulatorGainCurves {
        dopamine: ModulatorGainCurves {
            firing_rate: Some(GainCurve::new((0.0, 1.0), (1.0, 2.0))),
            ..Default::default()
        },
        ..Default::default()
    };
    let bag = NeuroModulators {
        dopamine: state.dopamine,
        cortisol: state.cortisol,
        ..Default::default()
    };
    let via_curves = curves.evaluate(&bag);
    println!(
        "curve-evaluated firing_rate_scale={}",
        via_curves.firing_rate_scale
    );

    Ok(())
}
