use crate::prelude::*;

/// Encodes analog values as spike rates based on input intensity.
///
/// Each input channel is mapped to a firing rate between `base_rate` and `max_rate`.
/// In batch mode (`encode`), each call generates independent probabilistic spikes.
/// In streaming mode (`encode_step`), accumulates expected spikes and fires deterministically
/// when the accumulated value exceeds a threshold per channel.
///
/// # Mathematical Model
///
/// For batch encoding:
/// ```text
/// rate_hz = base_rate + normalized * (max_rate - base_rate)
/// probability = 1 - exp(-rate_hz * dt_seconds)
/// spike if random() < probability
/// ```
///
/// For streaming (`encode_step`):
/// ```text
/// rate_hz = base_rate + normalized[i] * (max_rate - base_rate)
/// accumulator[i] += rate_hz * dt_seconds
/// spike if accumulator[i] >= 1.0 (then accumulator -= 1.0)
/// ```
///
/// # When to Use
///
/// - Converting continuous sensor values to spike rates
/// - Poisson-like spike generation with controllable average rates
/// - Real-time encoding where spike timing follows input intensity
///
/// # Parameters
///
/// - `base_rate`: Minimum firing rate in hertz (Hz) when input is at range minimum
/// - `max_rate`: Maximum firing rate in hertz (Hz) when input is at range maximum
/// - `range`: Tuple of (min, max) input values
/// - `dt_seconds`: Duration, in seconds, represented by each encode step
///
/// # Migration
///
/// [`RateEncoder::new`] keeps the previous constructor shape and uses
/// `dt_seconds = 0.1`, which preserves the old deterministic `/ 10.0`
/// increment for unit rates. Prefer [`RateEncoder::try_new`] for new code that
/// wants explicit time-step configuration and validation.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
pub struct RateEncoder {
    base_rate: f32,
    max_rate: f32,
    range: (f32, f32),
    dt_seconds: f32,
    /// Fractional phase per channel, kept in `[0, 1)`.
    ///
    /// Serialized as `accumulators` for backward compatibility with earlier
    /// checkpoints that stored a single combined float per channel.
    #[cfg_attr(feature = "serde", serde(rename = "accumulators"))]
    phases: Vec<f64>,
    /// Exact whole-spike backlog per channel (drainable past f64's `2^53` cliff).
    #[cfg_attr(feature = "serde", serde(default, skip_serializing_if = "Vec::is_empty"))]
    pending_spikes: Vec<u64>,
}

impl RateEncoder {
    /// Compatibility time step used by [`RateEncoder::new`].
    ///
    /// A 100 ms step makes `rate_hz * dt_seconds` equal to the previous
    /// deterministic `/ 10.0` increment for the same rate value.
    pub const DEFAULT_DT_SECONDS: f32 = 0.1;

    /// Creates a rate encoder with the compatibility `dt_seconds = 0.1`.
    ///
    /// Prefer [`RateEncoder::try_new`] when selecting an explicit sampling
    /// interval for new code.
    ///
    /// # Panics
    ///
    /// Panics if rates or range are invalid (`dt_seconds` is always the valid
    /// default `0.1`, so callers cannot panic via the time step).
    pub fn new(base_rate: f32, max_rate: f32, range: (f32, f32)) -> Self {
        Self::try_new(base_rate, max_rate, range, Self::DEFAULT_DT_SECONDS)
            .expect("invalid RateEncoder configuration")
    }

    /// Creates a rate encoder with an explicit time step in seconds.
    ///
    /// Rates must be finite and non-negative with `base_rate <= max_rate`.
    /// `dt_seconds` must be finite and strictly positive. Range must be a
    /// non-degenerate finite f32 span (bounds finite, ordered, and
    /// `max - min` finite in f32).
    pub fn try_new(
        base_rate: f32,
        max_rate: f32,
        range: (f32, f32),
        dt_seconds: f32,
    ) -> Result<Self, EncoderError> {
        crate::error::validate_non_negative_finite("base_rate", base_rate)?;
        crate::error::validate_non_negative_finite("max_rate", max_rate)?;
        if base_rate > max_rate {
            return Err(EncoderError::RateOrder);
        }
        crate::error::validate_range_f32_span("range", range)?;
        Self::validate_dt_seconds(dt_seconds)?;
        Ok(Self {
            base_rate,
            max_rate,
            range,
            dt_seconds,
            phases: Vec::new(),
            pending_spikes: Vec::new(),
        })
    }

    /// Returns the configured time step in seconds.
    pub fn dt_seconds(&self) -> f32 {
        self.dt_seconds
    }

    pub fn default_dt_seconds() -> f32 {
        Self::DEFAULT_DT_SECONDS
    }

    fn validate_dt_seconds(dt_seconds: f32) -> Result<(), EncoderError> {
        if dt_seconds.is_finite() && dt_seconds > 0.0 {
            Ok(())
        } else {
            Err(EncoderError::NonPositiveOrNonFinite {
                parameter: "dt_seconds",
            })
        }
    }

    fn normalize(&self, value: f32) -> f32 {
        ((value - self.range.0) / (self.range.1 - self.range.0)).clamp(0.0, 1.0)
    }

    fn ensure_accumulators(&mut self, num_channels: usize) {
        if self.phases.len() < num_channels {
            self.phases.resize(num_channels, 0.0);
            self.pending_spikes.resize(num_channels, 0);
        }
    }

    /// Split a non-negative finite value into exact whole spikes + fractional phase.
    ///
    /// Values at or above `u64::MAX` saturate the whole-spike count (pathological
    /// rates); the fractional part is then zero.
    fn split_whole_and_frac(value: f64) -> (u64, f64) {
        if !value.is_finite() || value <= 0.0 {
            return (0, 0.0);
        }
        if value < 1.0 {
            return (0, value);
        }
        // u64::MAX as f64 rounds to 2^64; every finite f32 increment is well
        // below that once cast through f64, but keep the guard for serde loads.
        if value >= u64::MAX as f64 {
            return (u64::MAX, 0.0);
        }
        let whole = value.trunc() as u64;
        let frac = value - whole as f64;
        // Keep phase in [0, 1). At the edge of exact-integer range the subtraction
        // can land slightly outside due to rounding; fold that into whole spikes.
        if frac >= 1.0 {
            (whole.saturating_add(1), 0.0)
        } else if frac < 0.0 {
            (whole, 0.0)
        } else {
            (whole, frac)
        }
    }

    fn apply_streaming_increment(&mut self, channel_idx: usize, increment: f32) {
        let sum = self.phases[channel_idx] + f64::from(increment);
        if !sum.is_finite() {
            // Pathological overflow of phase + increment; drop this step's add
            // rather than poison state (phase stays valid in [0, 1)).
            return;
        }
        let (whole, frac) = Self::split_whole_and_frac(sum);
        self.pending_spikes[channel_idx] =
            self.pending_spikes[channel_idx].saturating_add(whole);
        self.phases[channel_idx] = frac;
    }

    fn encode_with_rate_scale(&mut self, input: &[f32], rate_scale: f32) -> EncodedOutput {
        let mut output = EncodedOutput::new();
        if input.is_empty() {
            return output;
        }
        // Match PopulationEncoder: non-finite or non-positive scales fully silence.
        // Avoids NaN probabilities that would silently never spike.
        if !rate_scale.is_finite() || rate_scale <= 0.0 {
            return output;
        }

        let mut rng = rand::rng();
        for (i, &value) in input.iter().enumerate() {
            let Ok(channel) = u16::try_from(i) else {
                // Remaining channels exceed u16::MAX; stop rather than wrap.
                break;
            };
            let normalized = self.normalize(value);
            let rate =
                (self.base_rate + normalized * (self.max_rate - self.base_rate)) * rate_scale;
            let probability =
                crate::poisson::probability_from_rate_hz(rate.max(0.0), self.dt_seconds);

            if crate::rng::gen_unit_f32_with_rng(&mut rng) < probability {
                output.spikes.push(SpikeEvent {
                    channel,
                    timestamp: 0,
                    polarity: true,
                });
            }
        }

        output
    }

    /// Cap on spikes emitted per channel per streaming step.
    ///
    /// Bounds allocation when `rate_hz * dt_seconds` is huge. Remaining whole
    /// spikes stay in the exact `u64` pending queue and drain on later
    /// `encode_step` calls (no permanent loss of expected spike count, and no
    /// stall past f64's `2^53` integer cliff where `acc -= 1.0` would no-op).
    /// Non-finite increments are skipped so the emission loop always terminates.
    const MAX_SPIKES_PER_CHANNEL_PER_STEP: usize = 1024;

    fn streaming_increment(&self, value: f32, rate_scale: f32) -> Option<f32> {
        let normalized = self.normalize(value);
        let rate_hz = ((self.base_rate + normalized * (self.max_rate - self.base_rate))
            * rate_scale)
            .max(0.0);
        let increment = rate_hz * self.dt_seconds;
        // Non-finite increments (e.g. rate * f32::MAX) must not poison state.
        increment.is_finite().then_some(increment)
    }

    fn emit_capped_channel_spikes(
        &mut self,
        channel: u16,
        channel_idx: usize,
        output: &mut EncodedOutput,
    ) {
        let pending = self.pending_spikes[channel_idx];
        if pending == 0 {
            return;
        }
        let emit = pending.min(Self::MAX_SPIKES_PER_CHANNEL_PER_STEP as u64) as usize;
        for _ in 0..emit {
            output.spikes.push(SpikeEvent {
                channel,
                timestamp: 0,
                polarity: true,
            });
        }
        self.pending_spikes[channel_idx] = pending - emit as u64;
        // Any remaining whole spikes stay queued for subsequent steps.
    }

    fn rate_scale_is_active(rate_scale: f32) -> bool {
        rate_scale.is_finite() && rate_scale > 0.0
    }

    fn encode_step_with_rate_scale(&mut self, input: &[f32], rate_scale: f32) -> EncodedOutput {
        let mut output = EncodedOutput::new();
        if input.is_empty() {
            return output;
        }
        if !Self::rate_scale_is_active(rate_scale) {
            return output;
        }

        self.ensure_accumulators(input.len());

        for (i, &value) in input.iter().enumerate() {
            let Ok(channel) = u16::try_from(i) else {
                break;
            };
            let Some(increment) = self.streaming_increment(value, rate_scale) else {
                continue;
            };
            self.apply_streaming_increment(i, increment);
            self.emit_capped_channel_spikes(channel, i, &mut output);
        }

        output
    }

    /// Encodes input using neuromodulator-driven gain curves.
    ///
    /// Inherent wrapper so callers need not import [`ModulatedEncoder`].
    pub fn encode_with_modulators(
        &mut self,
        input: &[f32],
        modulators: &NeuroModulators,
        gain_curves: &NeuromodulatorGainCurves,
    ) -> EncodedOutput {
        <Self as ModulatedEncoder>::encode_with_modulators(self, input, modulators, gain_curves)
    }

    /// Step-wise variant of [`encode_with_modulators`](Self::encode_with_modulators).
    ///
    /// Uses the internal accumulator-based rate scale path for streaming.
    pub fn encode_step_with_modulators(
        &mut self,
        input: &[f32],
        modulators: &NeuroModulators,
        gain_curves: &NeuromodulatorGainCurves,
    ) -> EncodedOutput {
        <Self as ModulatedEncoder>::encode_step_with_modulators(
            self,
            input,
            modulators,
            gain_curves,
        )
    }
}

#[cfg(feature = "serde")]
impl<'de> serde::Deserialize<'de> for RateEncoder {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(serde::Deserialize)]
        struct Helper {
            base_rate: f32,
            max_rate: f32,
            range: (f32, f32),
            #[serde(default = "RateEncoder::default_dt_seconds")]
            dt_seconds: f32,
            /// Legacy combined float (phase + whole spikes) and/or fractional phase.
            #[serde(default)]
            accumulators: Vec<f64>,
            /// Exact whole-spike backlog (new format). Folded with any whole part
            /// still present in `accumulators` for forward/backward compatibility.
            #[serde(default)]
            pending_spikes: Vec<u64>,
        }

        let helper = Helper::deserialize(deserializer)?;
        let mut encoder = Self::try_new(
            helper.base_rate,
            helper.max_rate,
            helper.range,
            helper.dt_seconds,
        )
        .map_err(serde::de::Error::custom)?;
        // Allow values >= 1.0 in `accumulators` (legacy combined representation).
        // Reject only non-finite or negative state.
        if helper
            .accumulators
            .iter()
            .any(|value| !value.is_finite() || *value < 0.0)
        {
            return Err(serde::de::Error::custom(
                "accumulators must be finite and non-negative",
            ));
        }
        let n = helper
            .accumulators
            .len()
            .max(helper.pending_spikes.len());
        encoder.phases = vec![0.0; n];
        encoder.pending_spikes = vec![0; n];
        for i in 0..n {
            let combined = helper.accumulators.get(i).copied().unwrap_or(0.0);
            let (whole_from_acc, phase) = Self::split_whole_and_frac(combined);
            let pending = helper.pending_spikes.get(i).copied().unwrap_or(0);
            encoder.phases[i] = phase;
            encoder.pending_spikes[i] = pending.saturating_add(whole_from_acc);
        }
        Ok(encoder)
    }
}

impl Encoder for RateEncoder {
    fn encode(&mut self, input: &[f32]) -> EncodedOutput {
        self.encode_with_rate_scale(input, 1.0)
    }

    fn encode_step(&mut self, input: &[f32]) -> EncodedOutput {
        self.encode_step_with_rate_scale(input, 1.0)
    }

    fn reset(&mut self) {
        for phase in self.phases.iter_mut() {
            *phase = 0.0;
        }
        for pending in self.pending_spikes.iter_mut() {
            *pending = 0;
        }
    }
}

impl ModulatedEncoder for RateEncoder {
    fn encode_with_gains(&mut self, input: &[f32], gains: EncodingGains) -> EncodedOutput {
        self.encode_with_rate_scale(input, gains.sanitize().firing_rate_scale)
    }

    fn encode_step_with_gains(&mut self, input: &[f32], gains: EncodingGains) -> EncodedOutput {
        self.encode_step_with_rate_scale(input, gains.sanitize().firing_rate_scale)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rate_encoder_basic() {
        let mut encoder = RateEncoder::new(0.0, 10.0, (0.0, 100.0));
        let input = [0.0, 50.0, 100.0];
        let output = encoder.encode(&input);
        assert!(output.spikes.len() <= 3);
    }

    #[test]
    fn test_rate_encoder_encode_step() {
        let mut encoder = RateEncoder::new(0.0, 10.0, (0.0, 1.0));
        // max_rate 10.0 -> increment = (0.0 + 1.0 * 10.0) / 10.0 = 1.0
        let output = encoder.encode_step(&[1.0]);
        assert_eq!(output.spikes.len(), 1);

        let output2 = encoder.encode_step(&[0.5]);
        // 0.5 * 10.0 / 10.0 = 0.5 increment
        assert_eq!(output2.spikes.len(), 0);
        let output3 = encoder.encode_step(&[0.5]);
        // another 0.5 -> 1.0 -> spike
        assert_eq!(output3.spikes.len(), 1);
    }

    #[test]
    fn test_rate_encoder_empty_input() {
        let mut encoder = RateEncoder::new(0.0, 10.0, (0.0, 100.0));
        let input: [f32; 0] = [];
        let output = encoder.encode(&input);
        assert_eq!(output.spikes.len(), 0);
        let output_step = encoder.encode_step(&input);
        assert_eq!(output_step.spikes.len(), 0);
    }

    #[test]
    fn test_rate_encoder_single_channel() {
        let mut encoder = RateEncoder::new(5.0, 10.0, (0.0, 1.0));
        let input = [0.5];
        let output = encoder.encode(&input);
        assert!(output.spikes.len() <= 1);
    }

    #[test]
    fn test_rate_encoder_below_min() {
        let mut encoder = RateEncoder::new(0.0, 10.0, (0.0, 100.0));
        let input = [-50.0, -100.0, -1.0];
        let output = encoder.encode(&input);
        assert!(
            output.spikes.is_empty(),
            "Below-min inputs should produce no spikes"
        );
    }

    #[test]
    fn test_rate_encoder_above_max() {
        let mut encoder = RateEncoder::new(0.0, 10.0, (0.0, 100.0));
        let input = [150.0, 200.0, 101.0];
        let output = encoder.encode(&input);
        assert!(output.spikes.len() <= 3);
        for spike in &output.spikes {
            assert!(u32::from(spike.channel) < 3);
        }
    }

    #[test]
    fn test_rate_encoder_reset_does_not_panic() {
        let mut encoder = RateEncoder::new(5.0, 10.0, (0.0, 1.0));
        let input = [0.5; 10];
        encoder.encode(&input);
        encoder.reset();
        encoder.encode(&input);
    }

    #[test]
    fn test_rate_encoder_never_panics() {
        let mut encoder = RateEncoder::new(0.0, 10.0, (0.0, 100.0));
        let inputs: [&[f32]; 4] = [&[], &[0.0], &[50.0, 100.0], &[f32::MIN, f32::MAX]];
        for input in inputs {
            let result =
                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| encoder.encode(input)));
            assert!(result.is_ok());
        }
    }

    #[test]
    fn test_rate_encoder_modulated_step_scales_firing_rate() {
        let mut encoder = RateEncoder::new(0.0, 5.0, (0.0, 1.0));
        let modulators = NeuroModulators {
            dopamine: 1.0,
            ..Default::default()
        };
        let gain_curves = NeuromodulatorGainCurves {
            dopamine: ModulatorGainCurves {
                firing_rate: Some(GainCurve::new((0.0, 1.0), (1.0, 2.0))),
                ..Default::default()
            },
            ..Default::default()
        };

        let baseline = encoder.encode_step(&[1.0]);
        assert!(baseline.spikes.is_empty());

        encoder.reset();

        let boosted = encoder.encode_step_with_modulators(&[1.0], &modulators, &gain_curves);
        assert_eq!(boosted.spikes.len(), 1);
    }

    #[test]
    fn test_rate_encoder_encode_with_modulators() {
        let mut encoder = RateEncoder::new(0.0, 10.0, (0.0, 1.0));
        let modulators = NeuroModulators {
            dopamine: 1.0,
            ..Default::default()
        };
        let gain_curves = NeuromodulatorGainCurves {
            dopamine: ModulatorGainCurves {
                firing_rate: Some(GainCurve::new((0.0, 1.0), (1.0, 2.0))),
                ..Default::default()
            },
            ..Default::default()
        };

        // Use the deterministic streaming path: dopamine doubles the 10 Hz max
        // rate to 20 Hz, so at dt=0.1s the accumulator advances by 2.0 and emits
        // two spikes. Batch `encode_with_modulators` is stochastic (p ≈ 0.865)
        // and flaky under CI, so it is not used here.
        let boosted = encoder.encode_step_with_modulators(&[1.0], &modulators, &gain_curves);
        assert_eq!(boosted.spikes.len(), 2);
        assert!(boosted.spikes.iter().all(|s| s.channel == 0));

        // Baseline (identity gains) advances by 1.0 and emits a single spike.
        let mut baseline = RateEncoder::new(0.0, 10.0, (0.0, 1.0));
        let identity = baseline.encode_step_with_modulators(
            &[1.0],
            &NeuroModulators::default(),
            &NeuromodulatorGainCurves::default(),
        );
        assert_eq!(identity.spikes.len(), 1);
    }

    #[test]
    fn test_rate_encoder_step_shorter_input() {
        let mut encoder = RateEncoder::new(0.0, 10.0, (0.0, 1.0));
        // Grow accumulators to two channels, then step with a shorter slice so only
        // channel 0 is updated; channel 1 state is left untouched.
        let _ = encoder.encode_step(&[0.0, 0.0]);
        let output = encoder.encode_step(&[1.0]);
        assert_eq!(output.spikes.len(), 1);
        // Channel 1 still at zero accumulation: another zero-only step on both
        // channels must not invent a ch1 spike.
        let quiet = encoder.encode_step(&[0.0, 0.0]);
        assert!(quiet.spikes.is_empty());
    }

    #[test]
    fn test_rate_encoder_zero_rate_scale_never_accumulates() {
        let mut encoder = RateEncoder::new(0.0, 10.0, (0.0, 1.0));
        for _ in 0..10_000 {
            let output = encoder.encode_step_with_rate_scale(&[1.0], 0.0);
            assert!(
                output.spikes.is_empty(),
                "zero firing-rate scale must fully silence streaming output"
            );
        }
    }

    #[test]
    fn test_rate_encoder_non_finite_rate_scale_silences() {
        let mut encoder = RateEncoder::new(0.0, 10.0, (0.0, 1.0));
        for scale in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, -1.0] {
            let batch = encoder.encode_with_rate_scale(&[1.0], scale);
            assert!(
                batch.spikes.is_empty(),
                "non-finite/negative rate_scale ({scale}) must silence batch encode"
            );
            let step = encoder.encode_step_with_rate_scale(&[1.0], scale);
            assert!(
                step.spikes.is_empty(),
                "non-finite/negative rate_scale ({scale}) must silence streaming encode"
            );
        }
        // Accumulators must not be poisoned: a normal step after NaN still works.
        encoder.reset();
        let recovered = encoder.encode_step_with_rate_scale(&[1.0], 1.0);
        assert_eq!(recovered.spikes.len(), 1);
    }

    #[test]
    fn test_rate_encoder_try_new_validation() {
        let dt = RateEncoder::DEFAULT_DT_SECONDS;
        assert_eq!(
            RateEncoder::try_new(f32::NAN, 1.0, (0.0, 1.0), dt).err(),
            Some(EncoderError::NonNegativeFinite {
                parameter: "base_rate"
            })
        );
        assert_eq!(
            RateEncoder::try_new(0.0, f32::INFINITY, (0.0, 1.0), dt).err(),
            Some(EncoderError::NonNegativeFinite {
                parameter: "max_rate"
            })
        );
        assert_eq!(
            RateEncoder::try_new(-5.0, 10.0, (0.0, 1.0), dt).err(),
            Some(EncoderError::NonNegativeFinite {
                parameter: "base_rate"
            })
        );
        assert_eq!(
            RateEncoder::try_new(2.0, 1.0, (0.0, 1.0), dt).err(),
            Some(EncoderError::RateOrder)
        );
        assert_eq!(
            RateEncoder::try_new(0.0, 1.0, (1.0, 1.0), dt).err(),
            Some(EncoderError::InvalidRange { parameter: "range" })
        );
        assert_eq!(
            RateEncoder::try_new(0.0, 1.0, (f32::MIN, f32::MAX), dt).err(),
            Some(EncoderError::InvalidRange { parameter: "range" })
        );
        assert_eq!(
            RateEncoder::try_new(0.0, 10.0, (0.0, 1.0), 0.0).err(),
            Some(EncoderError::NonPositiveOrNonFinite {
                parameter: "dt_seconds"
            })
        );
    }

    #[test]
    fn test_rate_encoder_try_new_validates_dt_seconds() {
        assert!(RateEncoder::try_new(0.0, 10.0, (0.0, 1.0), 0.001).is_ok());
        for dt in [0.0, -0.001, f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            assert!(
                RateEncoder::try_new(0.0, 10.0, (0.0, 1.0), dt).is_err(),
                "dt_seconds={dt:?} should be rejected"
            );
        }
    }

    #[cfg(feature = "serde")]
    #[test]
    fn test_rate_encoder_serde_rejects_out_of_range_accumulators() {
        // Backlog whole spikes (>= 1.0) must round-trip after a capped step.
        let backlog = r#"{"base_rate":0.0,"max_rate":10.0,"range":[0.0,1.0],"accumulators":[5.0]}"#;
        let res: Result<RateEncoder, _> = serde_json::from_str(backlog);
        assert!(res.is_ok());

        let negative =
            r#"{"base_rate":0.0,"max_rate":10.0,"range":[0.0,1.0],"accumulators":[-0.1]}"#;
        let res: Result<RateEncoder, _> = serde_json::from_str(negative);
        assert!(res.is_err());

        let non_finite =
            r#"{"base_rate":0.0,"max_rate":10.0,"range":[0.0,1.0],"accumulators":[null]}"#;
        // JSON null is a type error
        let res: Result<RateEncoder, _> = serde_json::from_str(non_finite);
        assert!(res.is_err());

        let ok = r#"{"base_rate":0.0,"max_rate":10.0,"range":[0.0,1.0],"dt_seconds":0.1,"accumulators":[0.5]}"#;
        let res: Result<RateEncoder, _> = serde_json::from_str(ok);
        assert!(res.is_ok());
    }

    #[test]
    fn test_rate_encoder_large_backlog_drains_exactly() {
        // Above ~2^24, f32 cannot subtract 1.0; u64 pending must still drain.
        let mut encoder = RateEncoder::try_new(0.0, 20_000_000.0, (0.0, 1.0), 1.0).unwrap();
        let first = encoder.encode_step(&[1.0]);
        assert_eq!(
            first.spikes.len(),
            RateEncoder::MAX_SPIKES_PER_CHANNEL_PER_STEP
        );
        // Quiet step must continue draining the same cap amount.
        let second = encoder.encode_step(&[0.0]);
        assert_eq!(
            second.spikes.len(),
            RateEncoder::MAX_SPIKES_PER_CHANNEL_PER_STEP
        );
        // After many drain steps, total emitted should exceed a single cap.
        let mut total = first.spikes.len() + second.spikes.len();
        for _ in 0..10 {
            total += encoder.encode_step(&[0.0]).spikes.len();
        }
        assert!(
            total > RateEncoder::MAX_SPIKES_PER_CHANNEL_PER_STEP * 2,
            "backlog should keep draining across steps, total={total}"
        );
    }

    #[test]
    fn test_rate_encoder_backlog_drains_above_f64_precision() {
        // Above 2^53, f64 `acc -= 1.0` is a no-op. Exact u64 pending must still
        // decrease so quiet steps eventually exhaust the queue instead of
        // emitting the 1024-spike cap forever.
        //
        // Seed a modest backlog via serde (above one cap, well below 2^53 so the
        // test finishes quickly) and a huge runtime increment past 2^53.
        #[cfg(feature = "serde")]
        {
            let seeded: RateEncoder = serde_json::from_str(
                r#"{"base_rate":0.0,"max_rate":1.0,"range":[0.0,1.0],"dt_seconds":1.0,"accumulators":[2500.0]}"#,
            )
            .unwrap();
            let mut encoder = seeded;
            let first = encoder.encode_step(&[0.0]);
            assert_eq!(
                first.spikes.len(),
                RateEncoder::MAX_SPIKES_PER_CHANNEL_PER_STEP
            );
            let second = encoder.encode_step(&[0.0]);
            assert_eq!(
                second.spikes.len(),
                RateEncoder::MAX_SPIKES_PER_CHANNEL_PER_STEP
            );
            // 2500 - 2*1024 = 452 remaining
            let third = encoder.encode_step(&[0.0]);
            assert_eq!(third.spikes.len(), 452);
            let fourth = encoder.encode_step(&[0.0]);
            assert!(
                fourth.spikes.is_empty(),
                "backlog must fully drain rather than emit forever"
            );
        }

        // Runtime path: 1e16 Hz × 1 s is past f64's exact integer range.
        let mut encoder = RateEncoder::try_new(0.0, 1.0e16, (0.0, 1.0), 1.0).unwrap();
        let first = encoder.encode_step(&[1.0]);
        assert_eq!(
            first.spikes.len(),
            RateEncoder::MAX_SPIKES_PER_CHANNEL_PER_STEP
        );
        // Pending must decrease exactly by the cap each quiet step (not stall).
        let before = encoder.pending_spikes[0];
        assert!(
            before > RateEncoder::MAX_SPIKES_PER_CHANNEL_PER_STEP as u64,
            "expected a large exact backlog, got {before}"
        );
        let quiet = encoder.encode_step(&[0.0]);
        assert_eq!(
            quiet.spikes.len(),
            RateEncoder::MAX_SPIKES_PER_CHANNEL_PER_STEP
        );
        assert_eq!(
            encoder.pending_spikes[0],
            before - RateEncoder::MAX_SPIKES_PER_CHANNEL_PER_STEP as u64,
            "u64 pending must decrement exactly past the f64 precision cliff"
        );
    }

    #[test]
    fn test_rate_encoder_streaming_bounds_extreme_dt() {
        // f32::MAX is a valid finite dt, but rate * dt overflows to infinity.
        // The step must remain silent and terminate (no OOM / hang).
        let mut encoder = RateEncoder::try_new(0.0, 10.0, (0.0, 1.0), f32::MAX).unwrap();
        let output = encoder.encode_step(&[1.0]);
        assert!(output.spikes.is_empty());

        // Huge but finite expected count is capped per step; remainder is queued.
        let mut encoder = RateEncoder::try_new(0.0, 1.0e6, (0.0, 1.0), 1.0).unwrap();
        let output = encoder.encode_step(&[1.0]);
        assert_eq!(
            output.spikes.len(),
            RateEncoder::MAX_SPIKES_PER_CHANNEL_PER_STEP
        );
        // Undispatched whole spikes remain and drain on later quiet steps.
        let next = encoder.encode_step(&[0.0]);
        assert_eq!(
            next.spikes.len(),
            RateEncoder::MAX_SPIKES_PER_CHANNEL_PER_STEP
        );
    }

    #[test]
    fn test_rate_encoder_default_dt_preserves_streaming_compatibility() {
        let mut encoder = RateEncoder::new(0.0, 10.0, (0.0, 1.0));
        assert_eq!(encoder.dt_seconds(), RateEncoder::DEFAULT_DT_SECONDS);
        assert_eq!(encoder.encode_step(&[1.0]).spikes.len(), 1);
    }

    #[test]
    fn test_rate_encoder_streaming_uses_hz_times_dt() {
        let cases = [(5.0, 0.2, 10), (20.0, 0.05, 20), (7.5, 0.1, 40)];
        for (rate_hz, dt_seconds, steps) in cases {
            let mut encoder = RateEncoder::try_new(0.0, rate_hz, (0.0, 1.0), dt_seconds).unwrap();
            let spikes: usize = (0..steps)
                .map(|_| encoder.encode_step(&[1.0]).spikes.len())
                .sum();
            let elapsed_seconds = dt_seconds * steps as f32;
            let observed_hz = spikes as f32 / elapsed_seconds;
            assert!(
                (observed_hz - rate_hz).abs() <= 1.0 / elapsed_seconds,
                "rate_hz={rate_hz}, dt={dt_seconds}, observed={observed_hz}"
            );
        }
    }

    #[test]
    fn test_rate_encoder_stochastic_mean_matches_poisson_probability() {
        let cases = [(2.0, 0.01), (10.0, 0.005), (25.0, 0.002)];
        let trials = 50_000;
        for (rate_hz, dt_seconds) in cases {
            let mut encoder = RateEncoder::try_new(0.0, rate_hz, (0.0, 1.0), dt_seconds).unwrap();
            let spikes: usize = (0..trials)
                .map(|_| encoder.encode(&[1.0]).spikes.len())
                .sum();
            let observed_probability = spikes as f32 / trials as f32;
            let expected_probability =
                crate::poisson::probability_from_rate_hz(rate_hz, dt_seconds);
            assert!(
                (observed_probability - expected_probability).abs() < 0.01,
                "rate_hz={rate_hz}, dt={dt_seconds}, observed_p={observed_probability}, expected_p={expected_probability}"
            );
        }
    }
}
