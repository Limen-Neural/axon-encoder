use crate::prelude::*;

/// Encodes analog values as phase-locked spikes within a repeating oscillation cycle.
///
/// Each input channel produces at most one positive spike per call, with the spike
/// timestamp positioned relative to the current background phase according to the
/// normalized input value. Higher values map to later phase bins.
///
/// Timestamps are computed as `current_phase + phase_offset`, which keeps ordering
/// stable *within* a single encode call (higher-value channels get later timestamps).
/// Ordering *across* calls is not globally guaranteed, since `phase_offset` can exceed
/// the per-call phase advance. Cycle-relative phase is recoverable as
/// `timestamp % cycle_steps`.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
pub struct PhaseEncoder {
    cycle_steps: u64,
    range: (f32, f32),
    current_phase: u64,
}

/// Validates `cycle_steps` and `range`, returning an error message if invalid.
///
/// Shared by both `PhaseEncoder::new` (which panics on failure) and the
/// `Deserialize` impl (which surfaces the message as a deserialization error).
fn validate_params(cycle_steps: u64, range: (f32, f32)) -> Result<(), &'static str> {
    if cycle_steps == 0 {
        return Err("cycle_steps must be greater than 0");
    }
    if !range.0.is_finite() || !range.1.is_finite() || range.0 >= range.1 {
        return Err("range min must be less than range max and both must be finite");
    }
    Ok(())
}

impl PhaseEncoder {
    /// Creates a new `PhaseEncoder`.
    ///
    /// # Panics
    ///
    /// Panics if `cycle_steps == 0` or if range bounds are non-finite or `range.0 >= range.1`.
    pub fn new(cycle_steps: u64, range: (f32, f32)) -> Self {
        if let Err(message) = validate_params(cycle_steps, range) {
            panic!("{message}");
        }

        Self {
            cycle_steps,
            range,
            current_phase: 0,
        }
    }

    fn normalize(&self, value: f32) -> f32 {
        ((value - self.range.0) / (self.range.1 - self.range.0)).clamp(0.0, 1.0)
    }

    fn phase_offset(&self, normalized: f32) -> u64 {
        ((normalized * self.cycle_steps as f32).floor() as u64).min(self.cycle_steps - 1)
    }

    fn encode_current_cycle(&self, input: &[f32]) -> EncodedOutput {
        let mut output = EncodedOutput::new();

        for (channel, &value) in input.iter().enumerate() {
            // Non-finite inputs are invalid readings — skip rather than emit a
            // misleading phase-0 spike (NaN as u64 saturates to 0).
            if !value.is_finite() {
                continue;
            }

            let Ok(channel_u16) = u16::try_from(channel) else {
                // Remaining channels exceed u16::MAX; stop rather than wrap.
                break;
            };

            let phase_offset = self.phase_offset(self.normalize(value));
            // Monotonic timestamps preserve higher-value → later-phase ordering
            // even when phase_offset would wrap a modular cycle counter.
            output.spikes.push(SpikeEvent {
                channel: channel_u16,
                timestamp: self.current_phase.saturating_add(phase_offset),
                polarity: true,
            });
        }

        output
    }

    fn advance_phase(&mut self) {
        self.current_phase = self.current_phase.saturating_add(1);
    }

    fn encode_current_cycle_with_sensitivity_scale(
        &self,
        input: &[f32],
        sensitivity_scale: f32,
    ) -> EncodedOutput {
        let mut output = EncodedOutput::new();

        // Guard: zero or non-finite sensitivity collapses the range, suppressing all output.
        if !sensitivity_scale.is_finite() || sensitivity_scale <= 0.0 {
            return output;
        }

        let scaled_range = (
            self.range.0,
            self.range.0 + (self.range.1 - self.range.0) * sensitivity_scale,
        );

        for (channel, &value) in input.iter().enumerate() {
            if !value.is_finite() {
                continue;
            }

            let Ok(channel_u16) = u16::try_from(channel) else {
                break;
            };

            let normalized =
                ((value - scaled_range.0) / (scaled_range.1 - scaled_range.0)).clamp(0.0, 1.0);
            let phase_offset = self.phase_offset(normalized);
            output.spikes.push(SpikeEvent {
                channel: channel_u16,
                timestamp: self.current_phase.saturating_add(phase_offset),
                polarity: true,
            });
        }

        output
    }

    /// Encode input using neuromodulator-driven gain curves.
    ///
    /// Evaluates `gain_curves` against the current `modulators` to produce
    /// an [`EncodingGains`], then uses the `sensitivity_scale` component to
    /// modulate the input-to-phase mapping range. Values > 1.0 widen the range
    /// (less sensitive); values in (0, 1) narrow it (more sensitive).
    pub fn encode_with_modulators(
        &mut self,
        input: &[f32],
        modulators: &NeuroModulators,
        gain_curves: &NeuromodulatorGainCurves,
    ) -> EncodedOutput {
        let gains = gain_curves.evaluate(modulators);
        let output =
            self.encode_current_cycle_with_sensitivity_scale(input, gains.sensitivity_scale);
        self.advance_phase();
        output
    }

    /// Step-wise variant of [`encode_with_modulators`](Self::encode_with_modulators).
    ///
    /// Identical behavior — provided for API symmetry with the [`Encoder`] trait's
    /// `encode` / `encode_step` pair.
    pub fn encode_step_with_modulators(
        &mut self,
        input: &[f32],
        modulators: &NeuroModulators,
        gain_curves: &NeuromodulatorGainCurves,
    ) -> EncodedOutput {
        let gains = gain_curves.evaluate(modulators);
        let output =
            self.encode_current_cycle_with_sensitivity_scale(input, gains.sensitivity_scale);
        self.advance_phase();
        output
    }
}

impl Encoder for PhaseEncoder {
    fn encode(&mut self, input: &[f32]) -> EncodedOutput {
        let output = self.encode_current_cycle(input);
        self.advance_phase();
        output
    }

    fn encode_step(&mut self, input: &[f32]) -> EncodedOutput {
        // Streaming and batch modes share the same phase-step semantics for
        // this encoder: each call advances the background oscillation by one.
        let output = self.encode_current_cycle(input);
        self.advance_phase();
        output
    }

    fn reset(&mut self) {
        self.current_phase = 0;
    }
}

#[cfg(feature = "serde")]
impl<'de> serde::Deserialize<'de> for PhaseEncoder {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(serde::Deserialize)]
        struct Helper {
            cycle_steps: u64,
            range: (f32, f32),
            #[serde(default)]
            current_phase: u64,
        }

        let helper = Helper::deserialize(deserializer)?;

        validate_params(helper.cycle_steps, helper.range).map_err(serde::de::Error::custom)?;

        Ok(Self {
            cycle_steps: helper.cycle_steps,
            range: helper.range,
            current_phase: helper.current_phase,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_phase_mapping_clamps_and_quantizes() {
        let mut encoder = PhaseEncoder::new(8, (0.0, 10.0));

        let output = encoder.encode(&[-5.0, 0.0, 5.0, 10.0, 15.0]);
        let timestamps: Vec<u64> = output.spikes.iter().map(|spike| spike.timestamp).collect();
        let polarities: Vec<bool> = output.spikes.iter().map(|spike| spike.polarity).collect();

        assert_eq!(timestamps, vec![0, 0, 4, 7, 7]);
        assert_eq!(polarities, vec![true; 5]);
    }

    #[test]
    fn test_phase_advances_after_each_call() {
        let mut encoder = PhaseEncoder::new(4, (0.0, 1.0));

        assert_eq!(encoder.encode(&[0.0]).spikes[0].timestamp, 0);
        assert_eq!(encoder.encode(&[0.0]).spikes[0].timestamp, 1);
        assert_eq!(encoder.encode_step(&[0.0]).spikes[0].timestamp, 2);
        assert_eq!(encoder.encode_step(&[0.0]).spikes[0].timestamp, 3);
        // Monotonic absolute phase time (cycle phase is timestamp % cycle_steps).
        assert_eq!(encoder.encode(&[0.0]).spikes[0].timestamp, 4);
        assert_eq!(encoder.encode(&[0.0]).spikes[0].timestamp % 4, 1);
    }

    #[test]
    fn test_within_call_ordering_preserved_after_phase_advance() {
        let mut encoder = PhaseEncoder::new(8, (0.0, 1.0));
        // Advance near the end of a modular cycle so a wrap would reorder.
        for _ in 0..6 {
            encoder.encode(&[0.0]);
        }
        let output = encoder.encode(&[0.125, 0.375]); // offsets 1 and 3
        let timestamps: Vec<u64> = output.spikes.iter().map(|s| s.timestamp).collect();
        // 6+1=7, 6+3=9 — strictly ordered (no modular wrap inversion).
        assert_eq!(timestamps, vec![7, 9]);
        assert!(timestamps[0] < timestamps[1]);
    }

    #[test]
    fn test_reset_restores_initial_phase() {
        let mut encoder = PhaseEncoder::new(8, (0.0, 1.0));

        encoder.encode(&[0.0]);
        encoder.encode(&[0.0]);
        encoder.reset();

        let output = encoder.encode(&[1.0]);
        assert_eq!(output.spikes[0].timestamp, 7);
    }

    #[test]
    fn test_empty_input_returns_no_spikes() {
        let mut encoder = PhaseEncoder::new(4, (0.0, 1.0));

        let output = encoder.encode(&[]);
        assert!(output.spikes.is_empty());

        let next_output = encoder.encode(&[0.0]);
        assert_eq!(next_output.spikes[0].timestamp, 1);
    }

    #[test]
    fn test_nan_input_skips_channel() {
        let mut encoder = PhaseEncoder::new(8, (0.0, 1.0));
        let output = encoder.encode(&[0.0, f32::NAN, 1.0]);
        assert_eq!(output.spikes.len(), 2);
        assert_eq!(output.spikes[0].channel, 0);
        assert_eq!(output.spikes[1].channel, 2);
    }

    #[test]
    #[should_panic(expected = "cycle_steps must be greater than 0")]
    fn test_zero_cycle_steps_rejected() {
        let _ = PhaseEncoder::new(0, (0.0, 1.0));
    }

    #[test]
    #[should_panic(expected = "range min must be less than range max")]
    fn test_invalid_range_rejected() {
        let _ = PhaseEncoder::new(8, (1.0, 1.0));
    }

    #[test]
    fn test_encode_step_matches_encode() {
        let input = [2.5, 7.5];
        let mut encode_encoder = PhaseEncoder::new(8, (0.0, 10.0));
        let mut step_encoder = PhaseEncoder::new(8, (0.0, 10.0));

        assert_eq!(
            encode_encoder.encode(&input),
            step_encoder.encode_step(&input)
        );
        assert_eq!(
            encode_encoder.encode(&input),
            step_encoder.encode_step(&input)
        );
    }

    #[cfg(feature = "serde")]
    #[test]
    fn test_deserialize_rejects_zero_cycle_steps() {
        let json = r#"{"cycle_steps":0,"range":[0.0,1.0],"current_phase":0}"#;
        let err = serde_json::from_str::<PhaseEncoder>(json).unwrap_err();
        assert!(err.to_string().contains("cycle_steps"));
    }

    #[test]
    fn test_encode_with_modulators_identity() {
        let mut encoder = PhaseEncoder::new(8, (0.0, 1.0));
        let curves = NeuromodulatorGainCurves::default();
        let mods = NeuroModulators::default();

        let plain = encoder.encode(&[0.5]);
        let mut encoder2 = PhaseEncoder::new(8, (0.0, 1.0));
        let modulated = encoder2.encode_with_modulators(&[0.5], &mods, &curves);

        assert_eq!(plain.spikes[0].timestamp, modulated.spikes[0].timestamp);
    }

    #[test]
    fn test_encode_with_modulators_sensitivity_scale() {
        let mut encoder = PhaseEncoder::new(8, (0.0, 1.0));
        let curves = NeuromodulatorGainCurves {
            dopamine: ModulatorGainCurves {
                sensitivity: Some(GainCurve::new((0.0, 1.0), (0.5, 0.5))),
                ..Default::default()
            },
            ..Default::default()
        };
        let mods = NeuroModulators {
            dopamine: 1.0,
            ..Default::default()
        };

        let output = encoder.encode_with_modulators(&[0.5], &mods, &curves);
        // sensitivity_scale = 0.5, range = (0.0, 0.5)
        // value 0.5 maps to normalized 1.0, phase_offset = 7
        assert_eq!(output.spikes[0].timestamp, 7);
    }

    #[test]
    fn test_encode_step_with_modulators_matches_encode() {
        let input = [0.5];
        let curves = NeuromodulatorGainCurves::default();
        let mods = NeuroModulators::default();

        let mut encoder1 = PhaseEncoder::new(8, (0.0, 1.0));
        let mut encoder2 = PhaseEncoder::new(8, (0.0, 1.0));

        let batch = encoder1.encode_with_modulators(&input, &mods, &curves);
        let step = encoder2.encode_step_with_modulators(&input, &mods, &curves);

        assert_eq!(batch, step);
    }

    #[test]
    fn test_encode_with_modulators_zero_sensitivity_suppresses() {
        let mut encoder = PhaseEncoder::new(8, (0.0, 1.0));
        let curves = NeuromodulatorGainCurves {
            dopamine: ModulatorGainCurves {
                sensitivity: Some(GainCurve::new((0.0, 1.0), (0.0, 0.0))),
                ..Default::default()
            },
            ..Default::default()
        };
        let mods = NeuroModulators {
            dopamine: 1.0,
            ..Default::default()
        };

        let output = encoder.encode_with_modulators(&[0.5], &mods, &curves);
        assert!(output.spikes.is_empty());
    }
}
