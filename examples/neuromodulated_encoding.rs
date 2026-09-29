//! Neuromodulated Streaming Example
//!
//! Applies public `NeuroModulators` and `NeuromodulatorGainCurves` to a
//! streaming `RateEncoder`. The streaming rate path is deterministic; this
//! example deliberately does not promise replayability for stochastic batch
//! paths.
//!
//! ```
//! cargo run --example neuromodulated_encoding
//! ```

use axon_encoder::prelude::*;

fn main() -> Result<(), EncoderError> {
    let mut encoder = RateEncoder::try_new(0.0, 100.0, (0.0, 1.0), 0.010)?;
    let mut cursor = TimeCursor::new(encoder.time_model());
    let curves = NeuromodulatorGainCurves {
        dopamine: ModulatorGainCurves {
            // Dopamine maps from its 0..1 application-level range to a
            // 1x..2x firing-rate scale. This is application policy, not a
            // built-in biological model.
            firing_rate: Some(GainCurve::new((0.0, 1.0), (1.0, 2.0))),
            ..Default::default()
        },
        ..Default::default()
    };
    let mut modulators = NeuroModulators {
        dopamine: 0.8,
        ..Default::default()
    };

    for sample in [0.2_f32, 0.6, 1.0, 0.6, 0.2] {
        let output = encoder.encode_step_with_modulators(&[sample], &modulators, &curves);
        let gains = curves.evaluate(&modulators);
        for spike in &output.spikes {
            println!(
                "input {sample:.1}, rate scale {:.2}: channel {} at absolute tick {}",
                gains.firing_rate_scale,
                spike.channel,
                cursor.absolute(spike.timestamp)
            );
        }
        cursor.advance();
        modulators.decay();
    }

    Ok(())
}
