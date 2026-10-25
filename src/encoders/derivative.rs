use crate::prelude::*;

/// Encodes based on the rate of change (derivative) of the input.
///
/// Fires an excitatory spike when the positive change exceeds a threshold,
/// and an inhibitory spike when the negative change exceeds the threshold.
///
/// # Non-finite input
///
/// A `NaN` or `±inf` sample is **skipped**: it is not stored in `last_values`,
/// it emits no spike, and the channel's state is left exactly as it was. The
/// next finite sample is compared against the last *finite* sample, so the
/// encoder behaves as if the non-finite step never happened. This is the
/// crate-wide rule for stateful encoders (see also [`TemporalEncoder`] and
/// [`PredictiveEncoder`]); constructor and serde validation of non-finite
/// configuration is unchanged. Without the skip, `+inf` followed by a finite
/// value would compute `finite - inf = -inf` and emit a spurious inhibitory
/// spike, and `NaN` would silence the channel for one step.
///
/// [`TemporalEncoder`]: crate::encoders::TemporalEncoder
/// [`PredictiveEncoder`]: crate::encoders::PredictiveEncoder
///
/// # Examples
///
/// ```rust
/// use axon_encoder::prelude::*;
/// # fn main() -> Result<(), EncoderError> {
/// let mut enc = DerivativeEncoder::try_new(vec![0.2])?;
/// let _ = enc.encode_step(&[0.0]); // seed previous sample
/// let out = enc.encode_step(&[0.5]); // +0.5 change exceeds threshold
/// assert_eq!(out.spikes.len(), 1);
/// assert!(out.spikes[0].polarity);
/// # Ok(())
/// # }
/// ```
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(try_from = "DerivativeEncoderRepr"))]
pub struct DerivativeEncoder {
    last_values: Vec<f32>,
    thresholds: Vec<f32>,
}

#[cfg(feature = "serde")]
#[derive(serde::Deserialize)]
struct DerivativeEncoderRepr {
    last_values: Vec<f32>,
    thresholds: Vec<f32>,
}

#[cfg(feature = "serde")]
impl TryFrom<DerivativeEncoderRepr> for DerivativeEncoder {
    type Error = String;

    fn try_from(r: DerivativeEncoderRepr) -> Result<Self, String> {
        if r.last_values.len() != r.thresholds.len() {
            return Err(format!(
                "mismatched last_values length ({}) and thresholds length ({})",
                r.last_values.len(),
                r.thresholds.len()
            ));
        }
        if r.last_values.iter().any(|v| !v.is_finite()) {
            return Err("last_values must be finite".into());
        }
        let mut encoder = Self::try_new(r.thresholds).map_err(|error| error.to_string())?;
        encoder.last_values = r.last_values;
        Ok(encoder)
    }
}

impl DerivativeEncoder {
    /// Creates a new `DerivativeEncoder`, panicking if configuration is invalid.
    ///
    /// Prefer [`try_new`](Self::try_new) for typed validation errors.
    ///
    /// # Panics
    ///
    /// Panics if any threshold is non-finite/negative or the channel count is too large.
    pub fn new(thresholds: Vec<f32>) -> Self {
        Self::try_new(thresholds).expect("invalid DerivativeEncoder configuration")
    }

    /// Creates a new `DerivativeEncoder`, returning an [`EncoderError`] for invalid configuration.
    ///
    /// Each threshold must be finite and non-negative; channel count must fit `u16` IDs.
    pub fn try_new(thresholds: Vec<f32>) -> Result<Self, EncoderError> {
        crate::error::validate_channel_count(thresholds.len())?;
        for &threshold in &thresholds {
            crate::error::validate_non_negative_finite("threshold", threshold)?;
        }
        let num_channels = thresholds.len();
        Ok(Self {
            last_values: vec![0.0; num_channels],
            thresholds,
        })
    }

    /// Spike-emitting core: writes straight into `sink`, allocating nothing.
    ///
    /// Every public encoding path on this encoder routes through here, so the
    /// returning and sink-based APIs cannot drift apart.
    fn encode_step_into_sink<S: SpikeSink + ?Sized>(
        &mut self,
        current_values: &[f32],
        sink: &mut S,
    ) {
        sink.reserve(current_values.len().min(self.thresholds.len()));

        for (i, &current_val) in current_values.iter().enumerate() {
            if i >= self.thresholds.len() {
                break;
            }
            // Skip rule: a non-finite sample is neither stored nor compared.
            if !current_val.is_finite() {
                continue;
            }

            let delta = current_val - self.last_values[i];

            // Excitatory spike on positive jump exceeding threshold
            if delta > self.thresholds[i] {
                sink.push(SpikeEvent::at_step_start(
                    u16::try_from(i).expect("channel index exceeds u16::MAX"),
                    true,
                ));
            }
            // Inhibitory/Negative spike on sudden drop
            else if delta < -self.thresholds[i] {
                sink.push(SpikeEvent::at_step_start(
                    u16::try_from(i).expect("channel index exceeds u16::MAX"),
                    false,
                ));
            }

            self.last_values[i] = current_val;
        }
    }
}

impl Encoder for DerivativeEncoder {
    fn encode(&mut self, input: &[f32]) -> EncodedOutput {
        self.encode_step(input)
    }

    fn encode_step(&mut self, current_values: &[f32]) -> EncodedOutput {
        let mut output = EncodedOutput::new();
        self.encode_step_into_sink(current_values, &mut output.spikes);
        output
    }

    fn encode_into(&mut self, input: &[f32], sink: &mut dyn SpikeSink) {
        crate::sink::through_chunks(sink, |sink| self.encode_step_into_sink(input, sink));
    }

    fn encode_step_into(&mut self, input: &[f32], sink: &mut dyn SpikeSink) {
        self.encode_into(input, sink);
    }

    /// One call is one tick; `encode` and `encode_step` are the same path.
    ///
    /// A channel emits at most one spike per call — excitatory on a jump,
    /// inhibitory on a drop — always at
    /// [`TickOffset::ZERO`](crate::time::TickOffset::ZERO). Ticks are
    /// dimensionless: the derivative is per *step*, so the caller owns the
    /// physical step duration.
    fn time_model(&self) -> TimeModel {
        TimeModel::INSTANT
    }

    fn reset(&mut self) {
        for val in self.last_values.iter_mut() {
            *val = 0.0;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_derivative_encoder_basic() {
        let mut encoder = DerivativeEncoder::new(vec![1.0, 2.0]);

        // Initial jump
        let output = encoder.encode(&[1.5, 1.5]);
        assert_eq!(output.spikes.len(), 1);
        assert_eq!(output.spikes[0].channel, 0);
        assert!(output.spikes[0].polarity);

        // Stay same
        let output = encoder.encode(&[1.5, 1.5]);
        assert!(output.spikes.is_empty());

        // Jump down
        let output = encoder.encode(&[0.0, 1.5]);
        assert_eq!(output.spikes.len(), 1);
        assert_eq!(output.spikes[0].channel, 0);
        assert!(!output.spikes[0].polarity);

        // Jump up on channel 1
        let output = encoder.encode(&[0.0, 4.0]);
        assert_eq!(output.spikes.len(), 1);
        assert_eq!(output.spikes[0].channel, 1);
        assert!(output.spikes[0].polarity);
    }

    #[test]
    fn test_derivative_encoder_reset() {
        let mut encoder = DerivativeEncoder::new(vec![1.0]);
        encoder.encode(&[5.0]);
        encoder.reset();
        assert_eq!(encoder.last_values[0], 0.0);
    }

    #[test]
    fn test_derivative_encoder_empty_and_mismatched() {
        let mut encoder = DerivativeEncoder::new(vec![1.0]);
        let output = encoder.encode(&[]);
        assert!(output.spikes.is_empty());

        let output = encoder.encode(&[2.0, 3.0]);
        assert_eq!(output.spikes.len(), 1); // Only channel 0 should be processed
    }

    #[cfg(feature = "serde")]
    #[test]
    fn test_derivative_serde_rejects_too_many_channels() {
        let values: Vec<f32> = vec![0.0; (u16::MAX as usize) + 2];
        let value = serde_json::json!({
            "last_values": values.clone(),
            "thresholds": values,
        });
        let res: Result<DerivativeEncoder, _> = serde_json::from_value(value);
        assert!(res.is_err());
    }

    #[test]
    fn test_derivative_encoder_try_new_validation() {
        assert!(DerivativeEncoder::try_new(vec![0.0, 1.0]).is_ok());
        assert_eq!(
            DerivativeEncoder::try_new(vec![f32::NAN]).err(),
            Some(EncoderError::NonNegativeFinite {
                parameter: "threshold"
            })
        );
        assert_eq!(
            DerivativeEncoder::try_new(vec![-1.0]).err(),
            Some(EncoderError::NonNegativeFinite {
                parameter: "threshold"
            })
        );
        assert_eq!(
            DerivativeEncoder::try_new(vec![1.0; u16::MAX as usize + 2]).err(),
            Some(EncoderError::NumChannelsTooLarge)
        );
    }
}

#[cfg(test)]
mod branch_coverage_tests {
    use super::*;

    #[test]
    fn derivative_encoder_initializes_channel_state() {
        let encoder = DerivativeEncoder::new(vec![1.0, 2.0, 3.0]);
        assert_eq!(encoder.thresholds, vec![1.0, 2.0, 3.0]);
        assert_eq!(encoder.last_values, vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn derivative_encoder_tracks_positive_and_negative_steps() {
        let mut encoder = DerivativeEncoder::new(vec![1.0, 2.0]);

        let output = encoder.encode_step(&[1.5, -2.5]);
        assert_eq!(output.spikes.len(), 2);
        assert_eq!(output.spikes[0].channel, 0);
        assert!(output.spikes[0].polarity);
        assert_eq!(output.spikes[1].channel, 1);
        assert!(!output.spikes[1].polarity);
        assert_eq!(encoder.last_values, vec![1.5, -2.5]);

        let output = encoder.encode_step(&[2.0, -1.0]);
        assert!(output.spikes.is_empty());
        assert_eq!(encoder.last_values, vec![2.0, -1.0]);

        let output = encoder.encode_step(&[0.0, 2.0]);
        assert_eq!(output.spikes.len(), 2);
        assert!(!output.spikes[0].polarity);
        assert!(output.spikes[1].polarity);
        assert_eq!(encoder.last_values, vec![0.0, 2.0]);
    }

    #[test]
    fn derivative_encoder_does_not_fire_at_threshold() {
        let mut encoder = DerivativeEncoder::new(vec![1.0, 2.0]);
        let output = encoder.encode_step(&[1.0, -2.0]);
        assert!(output.spikes.is_empty());
        assert_eq!(encoder.last_values, vec![1.0, -2.0]);
    }

    const NON_FINITE: [f32; 3] = [f32::NAN, f32::INFINITY, f32::NEG_INFINITY];

    /// Collects one step's spikes through the named public path.
    fn step_via(path: &str, enc: &mut DerivativeEncoder, input: &[f32]) -> Vec<SpikeEvent> {
        let mut sink = Vec::new();
        match path {
            "encode" => return enc.encode(input).spikes,
            "encode_step" => return enc.encode_step(input).spikes,
            "encode_into" => enc.encode_into(input, &mut sink),
            "encode_step_into" => enc.encode_step_into(input, &mut sink),
            other => panic!("unknown path {other}"),
        }
        sink
    }

    const PATHS: [&str; 4] = ["encode", "encode_step", "encode_into", "encode_step_into"];

    /// Skip rule: a non-finite sample emits nothing and is never stored, while
    /// a finite sample in the same slice is handled normally.
    #[test]
    fn derivative_encoder_non_finite_sample_is_not_stored() {
        for bad in NON_FINITE {
            let mut enc = DerivativeEncoder::new(vec![0.5, 0.5]);
            let out = enc.encode_step(&[bad, 0.1]);
            assert!(
                out.spikes.is_empty(),
                "{bad}: no fire on the non-finite step"
            );
            assert_eq!(enc.last_values, vec![0.0, 0.1], "{bad}: must not be stored");
        }
    }

    /// After a non-finite step, a finite jump fires exactly as on a fresh
    /// encoder: Inf -> finite yields an excitatory spike, not a fake
    /// inhibitory one.
    #[test]
    fn derivative_encoder_recovers_after_non_finite_sample() {
        for bad in NON_FINITE {
            let mut live = DerivativeEncoder::new(vec![0.5, 0.5]);
            let mut fresh = DerivativeEncoder::new(vec![0.5, 0.5]);
            let _ = live.encode_step(&[bad, 0.1]);
            let _ = fresh.encode_step(&[0.0, 0.1]);

            let live_out = live.encode_step(&[1.0, 0.1]);
            let fresh_out = fresh.encode_step(&[1.0, 0.1]);
            assert_eq!(live_out.spikes, fresh_out.spikes, "{bad}: finite recovery");
            assert_eq!(
                live_out.spikes,
                vec![SpikeEvent::at_step_start(0, true)],
                "{bad}: must be a single excitatory spike"
            );
        }
    }

    #[test]
    fn derivative_encoder_skip_rule_holds_on_every_public_path() {
        for path in PATHS {
            let mut enc = DerivativeEncoder::new(vec![0.5]);
            let silent: Vec<_> = NON_FINITE
                .iter()
                .flat_map(|&bad| step_via(path, &mut enc, &[bad]))
                .collect();
            assert!(silent.is_empty(), "{path}: non-finite steps must be silent");
            assert_eq!(enc.last_values, vec![0.0], "{path}: must not be stored");
            assert_eq!(
                step_via(path, &mut enc, &[1.0]),
                vec![SpikeEvent::at_step_start(0, true)],
                "{path}: finite recovery"
            );
        }
    }

    #[test]
    fn derivative_encoder_handles_channel_count_mismatches() {
        let mut encoder = DerivativeEncoder::new(vec![1.0, 2.0]);
        let output = encoder.encode_step(&[2.0, -3.0, 5.0]);
        assert_eq!(output.spikes.len(), 2);
        assert_eq!(encoder.last_values, vec![2.0, -3.0]);

        let mut encoder = DerivativeEncoder::new(vec![1.0, 2.0]);
        let output = encoder.encode_step(&[2.0]);
        assert_eq!(output.spikes.len(), 1);
        assert_eq!(output.spikes[0].channel, 0);
        assert_eq!(encoder.last_values, vec![2.0, 0.0]);
    }
}
