//! `EmbeddingRateEncoder`: an integrate-and-fire rate encoder that owns its
//! membrane state, kept in its own module rather than [`crate::encoders`].
//!
//! This module is the crate's Tier 2 encoding boundary: encoders admitted
//! here follow the same public trait model as the `src/encoders/*.rs`
//! family — [`Encoder`] plus [`ModulatedEncoder`], fallible `try_new`
//! construction, and in-place `reset` — but are domain-agnostic
//! general-purpose encoders rather than members of that family's
//! signal-specific catalogue (rate / delta / derivative / latency / phase /
//! population / temporal / predictive). The module split marks that
//! distinction; the trait surface does not.

use crate::prelude::*;

fn validate_v_th(v_th: f32) -> Result<(), EncoderError> {
    if v_th.is_finite() && v_th > 0.0 {
        Ok(())
    } else {
        Err(EncoderError::NonPositiveOrNonFinite { parameter: "v_th" })
    }
}

#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(try_from = "EmbeddingEncoderConfigRepr"))]
pub struct EmbeddingEncoderConfig {
    pub v_th: f32,
}

#[cfg(feature = "serde")]
#[derive(serde::Deserialize)]
struct EmbeddingEncoderConfigRepr {
    v_th: f32,
}

#[cfg(feature = "serde")]
impl TryFrom<EmbeddingEncoderConfigRepr> for EmbeddingEncoderConfig {
    type Error = String;

    fn try_from(r: EmbeddingEncoderConfigRepr) -> Result<Self, String> {
        validate_v_th(r.v_th).map_err(|error| error.to_string())?;
        Ok(Self { v_th: r.v_th })
    }
}

/// Rate-style integrate-and-fire encoder over a fixed channel count.
///
/// Accumulates each channel's input into a persistent membrane potential and
/// emits a spike when it crosses `config.v_th`, soft-resetting by subtracting
/// the effective threshold. State persists across calls to
/// [`encode`](Encoder::encode) and [`encode_step`](Encoder::encode_step)
/// alike — both route through the same accumulation core — so call
/// [`reset`](Encoder::reset) to zero it between independent runs.
///
/// # Examples
///
/// ```rust
/// use axon_encoder::prelude::*;
/// # fn main() -> Result<(), EncoderError> {
/// let mut enc = EmbeddingRateEncoder::try_new(2, EmbeddingEncoderConfig { v_th: 0.4 })?;
/// let out = enc.encode(&[0.5, 1.0]);
/// assert!(!out.spikes.is_empty());
/// # Ok(())
/// # }
/// ```
///
/// # Migrating from `forward` / `EncoderState` (removed)
///
/// 0.4 threaded state explicitly through `forward` and normalized a fixed
/// embedding vector once at construction:
///
/// ```text
/// let enc = EmbeddingRateEncoder::new(&embeddings, config);
/// let (out, next) = enc.forward(&EncoderState::new_zeros(embeddings.len()));
/// ```
///
/// 0.5 owns its state and takes the drive vector as `encode`'s input instead:
///
/// ```rust
/// use axon_encoder::prelude::*;
/// # fn main() -> Result<(), EncoderError> {
/// # let embeddings = vec![0.5_f32, 1.0];
/// # let config = EmbeddingEncoderConfig { v_th: 0.4 };
/// let mut enc = EmbeddingRateEncoder::try_new(embeddings.len(), config)?;
/// let out = enc.encode(&embeddings);
/// # let _ = out;
/// # Ok(())
/// # }
/// ```
///
/// The built-in min-max normalization is also removed, since it applied a
/// hidden per-call transform whose result depended on the input distribution.
/// Callers that relied on it should normalize before calling `encode`, using
/// the former formula `(x - min) / (max - min + 1e-5)`.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(try_from = "EmbeddingRateEncoderRepr"))]
pub struct EmbeddingRateEncoder {
    pub config: EmbeddingEncoderConfig,
    membrane_potentials: Vec<f32>,
}

#[cfg(feature = "serde")]
#[derive(serde::Deserialize)]
struct EmbeddingRateEncoderRepr {
    config: EmbeddingEncoderConfig,
    membrane_potentials: Vec<f32>,
}

#[cfg(feature = "serde")]
impl TryFrom<EmbeddingRateEncoderRepr> for EmbeddingRateEncoder {
    type Error = String;

    fn try_from(r: EmbeddingRateEncoderRepr) -> Result<Self, String> {
        if r.membrane_potentials.iter().any(|v| !v.is_finite()) {
            return Err("membrane_potentials must be finite".into());
        }
        let mut encoder = EmbeddingRateEncoder::try_new(r.membrane_potentials.len(), r.config)
            .map_err(|error| error.to_string())?;
        encoder.membrane_potentials = r.membrane_potentials;
        Ok(encoder)
    }
}

impl EmbeddingRateEncoder {
    /// Creates an encoder for `num_channels` channels, panicking on invalid `config`.
    ///
    /// Prefer [`try_new`](Self::try_new) for typed validation errors.
    pub fn new(num_channels: usize, config: EmbeddingEncoderConfig) -> Self {
        Self::try_new(num_channels, config).expect("invalid EmbeddingRateEncoder configuration")
    }

    /// Creates an encoder for `num_channels` channels, returning an
    /// [`EncoderError`] for invalid configuration.
    ///
    /// `config.v_th` must be finite and strictly positive; `num_channels`
    /// must fit the `u16` channel-ID range used when emitting spikes.
    pub fn try_new(
        num_channels: usize,
        config: EmbeddingEncoderConfig,
    ) -> Result<Self, EncoderError> {
        validate_v_th(config.v_th)?;
        crate::error::validate_channel_count(num_channels)?;
        Ok(Self {
            config,
            membrane_potentials: vec![0.0; num_channels],
        })
    }

    /// Spike-emitting core: writes straight into `sink`, allocating nothing.
    ///
    /// Every public encoding path on this encoder routes through here, so the
    /// returning and sink-based APIs cannot drift apart. `input` is aligned
    /// to the tracked channel count: excess values are ignored, and a
    /// shorter slice leaves the remaining channels' potentials untouched.
    fn accumulate_and_fire<S: SpikeSink + ?Sized>(
        &mut self,
        input: &[f32],
        threshold_scale: f32,
        sink: &mut S,
    ) {
        let effective_threshold = self.config.v_th * threshold_scale;
        let aligned_len = input.len().min(self.membrane_potentials.len());

        for (i, &value) in input[..aligned_len].iter().enumerate() {
            let potential = &mut self.membrane_potentials[i];
            *potential += value;

            if *potential >= effective_threshold {
                sink.push(SpikeEvent::at_step_start(
                    u16::try_from(i).expect("channel index exceeds u16::MAX"),
                    true,
                ));
                *potential -= effective_threshold; // Soft reset
            }
        }
    }
}

impl Encoder for EmbeddingRateEncoder {
    /// Accumulates `input` into persistent per-channel membrane potentials
    /// that carry across every call — batch or streaming alike.
    fn encode(&mut self, input: &[f32]) -> EncodedOutput {
        let mut output = EncodedOutput::new();
        self.accumulate_and_fire(input, 1.0, &mut output.spikes);
        output
    }

    /// Identical to [`encode`](Self::encode): state already persists across
    /// calls, so batch and streaming share one path.
    fn encode_step(&mut self, input: &[f32]) -> EncodedOutput {
        self.encode(input)
    }

    fn encode_into(&mut self, input: &[f32], sink: &mut dyn SpikeSink) {
        crate::sink::through_chunks(sink, |sink| self.accumulate_and_fire(input, 1.0, sink));
    }

    fn encode_step_into(&mut self, input: &[f32], sink: &mut dyn SpikeSink) {
        self.encode_into(input, sink);
    }

    /// One call is one tick, and every spike lands at
    /// [`TickOffset::ZERO`](crate::time::TickOffset::ZERO): this encoder
    /// reports *whether* a channel crossed threshold in the step, not *when*
    /// within it. Ticks are dimensionless — `EmbeddingEncoderConfig` carries
    /// no `dt`, so the caller owns the physical step duration.
    fn time_model(&self) -> TimeModel {
        TimeModel::INSTANT
    }

    fn reset(&mut self) {
        self.membrane_potentials.fill(0.0);
    }
}

impl ModulatedEncoder for EmbeddingRateEncoder {
    /// Maps the sanitized `threshold_scale` onto the effective firing
    /// threshold: a scale above `1.0` raises `v_th` and reduces spiking,
    /// while `0.0` means every non-negative accumulation spikes immediately.
    fn encode_with_gains(&mut self, input: &[f32], gains: EncodingGains) -> EncodedOutput {
        let threshold_scale = gains.sanitize().threshold_scale;
        let mut output = EncodedOutput::new();
        self.accumulate_and_fire(input, threshold_scale, &mut output.spikes);
        output
    }

    fn encode_step_with_gains(&mut self, input: &[f32], gains: EncodingGains) -> EncodedOutput {
        self.encode_with_gains(input, gains)
    }

    fn encode_with_gains_into(
        &mut self,
        input: &[f32],
        gains: EncodingGains,
        sink: &mut dyn SpikeSink,
    ) {
        let threshold_scale = gains.sanitize().threshold_scale;
        crate::sink::through_chunks(sink, |sink| {
            self.accumulate_and_fire(input, threshold_scale, sink)
        });
    }

    fn encode_step_with_gains_into(
        &mut self,
        input: &[f32],
        gains: EncodingGains,
        sink: &mut dyn SpikeSink,
    ) {
        self.encode_with_gains_into(input, gains, sink);
    }

    /// Skips the intermediate [`EncodedOutput`] the trait default builds; see
    /// [`RateEncoder`](crate::encoders::RateEncoder)'s override of the same
    /// method for why every encoder in this crate does this.
    fn encode_with_modulators_into(
        &mut self,
        input: &[f32],
        modulators: &NeuroModulators,
        gain_curves: &NeuromodulatorGainCurves,
        sink: &mut dyn SpikeSink,
    ) {
        self.encode_with_gains_into(input, gain_curves.evaluate(modulators), sink);
    }

    /// Streaming counterpart of
    /// [`encode_with_modulators_into`](Self::encode_with_modulators_into).
    fn encode_step_with_modulators_into(
        &mut self,
        input: &[f32],
        modulators: &NeuroModulators,
        gain_curves: &NeuromodulatorGainCurves,
        sink: &mut dyn SpikeSink,
    ) {
        self.encode_step_with_gains_into(input, gain_curves.evaluate(modulators), sink);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_embedding_rate_encoder_basic() {
        let config = EmbeddingEncoderConfig { v_th: 0.9 };
        let mut encoder = EmbeddingRateEncoder::try_new(3, config).unwrap();

        let output = encoder.encode(&[0.5, 1.0, 0.0]);
        assert_eq!(output.spikes.len(), 1);
        assert_eq!(output.spikes[0].channel, 1);

        // Channel 0: 0.5 + 0.5 = 1.0 > 0.9 -> spike
        // Channel 1: (1.0-0.9) + 1.0 = 1.1 > 0.9 -> spike
        let output2 = encoder.encode(&[0.5, 1.0, 0.0]);
        assert_eq!(output2.spikes.len(), 2);
    }

    #[test]
    fn try_new_rejects_non_positive_or_non_finite_v_th() {
        for v_th in [0.0, -1.0, f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            assert_eq!(
                EmbeddingRateEncoder::try_new(1, EmbeddingEncoderConfig { v_th }).err(),
                Some(EncoderError::NonPositiveOrNonFinite { parameter: "v_th" })
            );
        }
    }

    #[test]
    fn try_new_rejects_too_many_channels() {
        assert_eq!(
            EmbeddingRateEncoder::try_new(
                u16::MAX as usize + 2,
                EmbeddingEncoderConfig { v_th: 1.0 }
            )
            .err(),
            Some(EncoderError::NumChannelsTooLarge)
        );
    }

    #[test]
    #[should_panic(expected = "invalid EmbeddingRateEncoder configuration")]
    fn new_panics_on_invalid_config() {
        let _ = EmbeddingRateEncoder::new(1, EmbeddingEncoderConfig { v_th: 0.0 });
    }

    #[test]
    fn reset_zeroes_membrane_potentials_in_place() {
        let mut encoder =
            EmbeddingRateEncoder::try_new(3, EmbeddingEncoderConfig { v_th: 0.4 }).unwrap();
        encoder.encode(&[0.1, 0.2, 0.3]);
        encoder.reset();
        assert_eq!(encoder.membrane_potentials, vec![0.0, 0.0, 0.0]);
    }

    /// Equivalent to the pre-0.5 `forward` behavior on a zero state, driven
    /// with an already-normalized vector as a literal (the removed
    /// min-max-normalizing constructor is not reconstructed here).
    #[test]
    fn embedding_rate_encoder_matches_pre_0_5_forward_output() {
        let mut encoder =
            EmbeddingRateEncoder::try_new(3, EmbeddingEncoderConfig { v_th: 0.9 }).unwrap();
        let drive = [0.499_995, 0.999_990, 0.0];
        let out = encoder.encode(&drive);

        assert_eq!(out.spikes, vec![SpikeEvent::at_step_start(1, true)]);
    }

    #[cfg(feature = "serde")]
    #[test]
    fn test_embedding_rate_encoder_deserialize_rejects_too_many_channels() {
        let membrane_potentials: Vec<f32> = vec![0.0; (u16::MAX as usize) + 2];
        let value = serde_json::json!({
            "config": {"v_th": 1.0},
            "membrane_potentials": membrane_potentials,
        });
        let res: Result<EmbeddingRateEncoder, _> = serde_json::from_value(value);
        assert!(res.is_err());
    }

    #[cfg(feature = "serde")]
    #[test]
    fn test_embedding_rate_encoder_deserialize_rejects_non_finite_state() {
        // JSON has no NaN literal; `null` in its place is a type error, which
        // is still the rejection this test wants.
        let json = r#"{"config":{"v_th":1.0},"membrane_potentials":[0.0,null]}"#;
        let res: Result<EmbeddingRateEncoder, _> = serde_json::from_str(json);
        assert!(res.is_err());
    }
}

#[cfg(test)]
mod encode_coverage_tests {
    use super::*;

    #[test]
    fn state_persists_across_encode_calls() {
        let mut encoder =
            EmbeddingRateEncoder::try_new(2, EmbeddingEncoderConfig { v_th: 0.6 }).unwrap();

        for _ in 0..3 {
            let out = encoder.encode(&[0.1, 0.7]);
            assert_eq!(out.spikes, vec![SpikeEvent::at_step_start(1, true)]);
        }

        assert!((encoder.membrane_potentials[0] - 0.3).abs() < 1e-5);
        assert!((encoder.membrane_potentials[1] - 0.3).abs() < 1e-5);
    }

    #[test]
    fn encode_step_matches_encode() {
        let mut via_encode =
            EmbeddingRateEncoder::try_new(2, EmbeddingEncoderConfig { v_th: 0.5 }).unwrap();
        let mut via_step = via_encode.clone();

        for input in [[0.2, 0.6], [0.3, 0.1], [0.5, 0.9]] {
            assert_eq!(
                via_encode.encode(&input),
                via_step.encode_step(&input),
                "encode and encode_step must produce identical output"
            );
        }
    }

    #[test]
    fn encode_into_matches_encode() {
        let mut via_encode =
            EmbeddingRateEncoder::try_new(2, EmbeddingEncoderConfig { v_th: 0.5 }).unwrap();
        let mut via_sink = via_encode.clone();

        for input in [[0.2, 0.6], [0.3, 0.1]] {
            let returned = via_encode.encode(&input);
            let mut buffer = Vec::new();
            via_sink.encode_into(&input, &mut buffer);
            assert_eq!(returned.spikes, buffer);
        }
    }

    #[test]
    fn encode_truncates_excess_channels() {
        let mut encoder =
            EmbeddingRateEncoder::try_new(1, EmbeddingEncoderConfig { v_th: 0.5 }).unwrap();
        let out = encoder.encode(&[0.6, 0.9]); // second value ignored: only 1 channel tracked
        assert_eq!(out.spikes, vec![SpikeEvent::at_step_start(0, true)]);
    }

    #[test]
    fn encode_handles_shorter_input_as_fewer_active_channels() {
        let mut encoder =
            EmbeddingRateEncoder::try_new(2, EmbeddingEncoderConfig { v_th: 0.5 }).unwrap();
        let out = encoder.encode(&[0.6]); // channel 1 untouched
        assert_eq!(out.spikes, vec![SpikeEvent::at_step_start(0, true)]);
        assert_eq!(encoder.membrane_potentials[1], 0.0);
    }

    #[test]
    fn modulated_zero_threshold_scale_spikes_on_any_nonnegative_input() {
        let mut encoder =
            EmbeddingRateEncoder::try_new(1, EmbeddingEncoderConfig { v_th: 10.0 }).unwrap();
        let out = encoder.encode_with_gains(
            &[0.001],
            EncodingGains {
                threshold_scale: 0.0,
                ..EncodingGains::identity()
            },
        );
        assert_eq!(out.spikes.len(), 1);
    }

    #[test]
    fn modulated_threshold_scale_raises_effective_threshold() {
        let mut encoder =
            EmbeddingRateEncoder::try_new(1, EmbeddingEncoderConfig { v_th: 0.5 }).unwrap();
        let out = encoder.encode_with_gains(
            &[0.6],
            EncodingGains {
                threshold_scale: 2.0,
                ..EncodingGains::identity()
            },
        );
        assert!(out.spikes.is_empty(), "effective threshold 1.0 > 0.6");
    }

    #[test]
    fn modulated_paths_ignore_non_threshold_gain_components() {
        let mut baseline =
            EmbeddingRateEncoder::try_new(1, EmbeddingEncoderConfig { v_th: 0.5 }).unwrap();
        let mut modulated = baseline.clone();

        let out = baseline.encode(&[0.6]);
        let gained = modulated.encode_with_gains(
            &[0.6],
            EncodingGains {
                threshold_scale: 1.0,
                sensitivity_scale: 4.0,
                firing_rate_scale: 4.0,
                latency_scale: 4.0,
            },
        );
        assert_eq!(out, gained);
    }
}
