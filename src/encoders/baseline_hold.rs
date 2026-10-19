use crate::prelude::*;

/// Baseline-hold ternary encoder (Corinth `TelemetryEncoder` semantics).
///
/// Each channel keeps a baseline that moves **only** when a signed threshold
/// crossing fires. The first non-empty encode after construction or
/// [`reset`](Encoder::reset) seeds baselines from the input and emits zero for
/// every channel (no spike / no ternary event).
///
/// Crossing rules (inclusive boundaries):
///
/// - positive event when `current - baseline >= threshold`
/// - negative event when `current - baseline <= -threshold`
/// - otherwise zero; baseline unchanged
///
/// Use [`encode_ternary`](Self::encode_ternary) for dense `{-1, 0, +1}` output.
/// The [`Encoder`] implementation maps nonzero ternary values to
/// [`SpikeEvent`] polarity at [`TickOffset::ZERO`](crate::time::TickOffset::ZERO).
///
/// Extracted from [corinth-canal#158](https://github.com/rmems/corinth-canal/issues/158);
/// not interchangeable with [`DeltaEncoder`] (absolute delta) or
/// [`DerivativeEncoder`] (baseline advances every step, strict `>` / `<` bounds).
///
/// [`DeltaEncoder`]: crate::encoders::DeltaEncoder
/// [`DerivativeEncoder`]: crate::encoders::DerivativeEncoder
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(try_from = "BaselineHoldTernaryEncoderRepr"))]
pub struct BaselineHoldTernaryEncoder {
    baselines: Vec<f32>,
    thresholds: Vec<f32>,
    initialized: bool,
}

#[cfg(feature = "serde")]
#[derive(serde::Deserialize)]
struct BaselineHoldTernaryEncoderRepr {
    baselines: Vec<f32>,
    thresholds: Vec<f32>,
    initialized: bool,
}

#[cfg(feature = "serde")]
impl TryFrom<BaselineHoldTernaryEncoderRepr> for BaselineHoldTernaryEncoder {
    type Error = String;

    fn try_from(r: BaselineHoldTernaryEncoderRepr) -> Result<Self, String> {
        if r.baselines.len() != r.thresholds.len() {
            return Err(format!(
                "mismatched baselines length ({}) and thresholds length ({})",
                r.baselines.len(),
                r.thresholds.len()
            ));
        }
        if r.baselines.iter().any(|v| !v.is_finite()) {
            return Err("baselines must be finite".into());
        }
        let mut encoder = Self::try_new(r.thresholds).map_err(|error| error.to_string())?;
        encoder.baselines = r.baselines;
        encoder.initialized = r.initialized;
        Ok(encoder)
    }
}

impl BaselineHoldTernaryEncoder {
    /// Creates a new encoder, panicking if configuration is invalid.
    ///
    /// Prefer [`try_new`](Self::try_new) for typed validation errors.
    pub fn new(thresholds: Vec<f32>) -> Self {
        Self::try_new(thresholds).expect("invalid BaselineHoldTernaryEncoder configuration")
    }

    /// Creates a new encoder with one threshold per channel.
    pub fn try_new(thresholds: Vec<f32>) -> Result<Self, EncoderError> {
        crate::error::validate_channel_count(thresholds.len())?;
        for &threshold in &thresholds {
            crate::error::validate_non_negative_finite("threshold", threshold)?;
        }
        let num_channels = thresholds.len();
        Ok(Self {
            baselines: vec![0.0; num_channels],
            thresholds,
            initialized: false,
        })
    }

    /// Returns the per-channel baselines (same length as configured thresholds).
    pub fn baselines(&self) -> &[f32] {
        &self.baselines
    }

    /// Whether the first-sample seeding pass has completed.
    pub fn is_initialized(&self) -> bool {
        self.initialized
    }

    /// Dense ternary encoding: `-1`, `0`, or `+1` per processed channel.
    ///
    /// Input is truncated to the configured channel count. An empty slice is a
    /// no-op and returns an empty vector.
    pub fn encode_ternary(&mut self, input: &[f32]) -> Vec<i8> {
        self.encode_ternary_values(Self::active_values(input, self.thresholds.len()))
    }

    fn encode_ternary_values(&mut self, values: &[f32]) -> Vec<i8> {
        if values.is_empty() {
            return Vec::new();
        }
        if !self.initialized {
            return self.seed_baselines(values);
        }
        self.ternary_with_held_baselines(values)
    }

    fn active_values<'a>(input: &'a [f32], channel_count: usize) -> &'a [f32] {
        if channel_count == 0 {
            &[]
        } else {
            &input[..input.len().min(channel_count)]
        }
    }

    fn seed_baselines(&mut self, values: &[f32]) -> Vec<i8> {
        for (i, &value) in values.iter().enumerate() {
            self.baselines[i] = value;
        }
        self.initialized = true;
        vec![0; values.len()]
    }

    fn ternary_with_held_baselines(&mut self, values: &[f32]) -> Vec<i8> {
        let mut out = vec![0i8; values.len()];
        for (i, &value) in values.iter().enumerate() {
            let event = Self::classify_crossing(value - self.baselines[i], self.thresholds[i]);
            if event != 0 {
                self.baselines[i] = value;
            }
            out[i] = event;
        }
        out
    }

    fn classify_crossing(delta: f32, threshold: f32) -> i8 {
        if delta >= threshold {
            1
        } else if delta <= -threshold {
            -1
        } else {
            0
        }
    }

    fn encode_step_into_sink<S: SpikeSink + ?Sized>(&mut self, input: &[f32], sink: &mut S) {
        let values = Self::active_values(input, self.thresholds.len());
        sink.reserve(values.len());
        let ternary = self.encode_ternary_values(values);
        for (i, &t) in ternary.iter().enumerate() {
            if t == 0 {
                continue;
            }
            sink.push(SpikeEvent::at_step_start(
                u16::try_from(i).expect("channel index exceeds u16::MAX"),
                t > 0,
            ));
        }
    }

    fn clamp_to_channels<'a>(&self, input: &'a [f32]) -> &'a [f32] {
        if input.len() > self.thresholds.len() {
            &input[..self.thresholds.len()]
        } else {
            input
        }
    }
}

impl Encoder for BaselineHoldTernaryEncoder {
    fn encode(&mut self, input: &[f32]) -> EncodedOutput {
        self.encode_step(input)
    }

    fn encode_step(&mut self, input: &[f32]) -> EncodedOutput {
        let safe_input = self.clamp_to_channels(input);
        let mut output = EncodedOutput::new();
        self.encode_step_into_sink(safe_input, &mut output.spikes);
        output
    }

    fn encode_into(&mut self, input: &[f32], sink: &mut dyn SpikeSink) {
        crate::sink::through_chunks(sink, |sink| self.encode_step_into_sink(input, sink));
    }

    fn encode_step_into(&mut self, input: &[f32], sink: &mut dyn SpikeSink) {
        let safe_input = self.clamp_to_channels(input);
        crate::sink::through_chunks(sink, |sink| self.encode_step_into_sink(safe_input, sink));
    }

    fn time_model(&self) -> TimeModel {
        TimeModel::INSTANT
    }

    fn reset(&mut self) {
        for baseline in self.baselines.iter_mut() {
            *baseline = 0.0;
        }
        self.initialized = false;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const CORINTH_THRESHOLDS: [f32; 4] = [1.0, 5.0, 1.0, 5.0];

    fn corinth_encoder() -> BaselineHoldTernaryEncoder {
        BaselineHoldTernaryEncoder::new(CORINTH_THRESHOLDS.to_vec())
    }

    #[test]
    fn first_encode_seeds_baseline_and_emits_zeroes() {
        let mut encoder = corinth_encoder();
        let snap = [60.0, 260.0, 77.0, 153.0];

        assert_eq!(encoder.encode_ternary(&snap), vec![0, 0, 0, 0]);
        assert!(encoder.is_initialized());
        assert_eq!(encoder.baselines(), &[60.0, 260.0, 77.0, 153.0]);
    }

    #[test]
    fn positive_threshold_crossing_emits_positive_spike_and_updates_baseline() {
        let mut encoder = corinth_encoder();
        encoder.encode_ternary(&[60.0, 260.0, 77.0, 153.0]);

        let output = encoder.encode_ternary(&[61.0, 260.0, 77.0, 153.0]);

        assert_eq!(output, vec![1, 0, 0, 0]);
        assert_eq!(encoder.baselines(), &[61.0, 260.0, 77.0, 153.0]);
    }

    #[test]
    fn negative_threshold_crossing_emits_negative_spike_and_updates_baseline() {
        let mut encoder = corinth_encoder();
        encoder.encode_ternary(&[60.0, 260.0, 77.0, 153.0]);

        let output = encoder.encode_ternary(&[60.0, 254.5, 77.0, 153.0]);

        assert_eq!(output, vec![0, -1, 0, 0]);
        assert_eq!(encoder.baselines(), &[60.0, 254.5, 77.0, 153.0]);
    }

    #[test]
    fn subthreshold_change_emits_zero_and_preserves_baseline() {
        let mut encoder = corinth_encoder();
        encoder.encode_ternary(&[60.0, 260.0, 77.0, 153.0]);

        let output = encoder.encode_ternary(&[60.5, 264.9, 77.9, 157.9]);

        assert_eq!(output, vec![0, 0, 0, 0]);
        assert_eq!(encoder.baselines(), &[60.0, 260.0, 77.0, 153.0]);
    }

    #[test]
    fn channels_use_independent_thresholds_and_baselines() {
        let mut encoder = BaselineHoldTernaryEncoder::new(vec![1.0, 10.0, 0.5, 7.0]);
        encoder.encode_ternary(&[60.0, 260.0, 77.0, 153.0]);

        let output = encoder.encode_ternary(&[61.0, 269.0, 76.0, 160.0]);

        assert_eq!(output, vec![1, 0, -1, 1]);
        assert_eq!(encoder.baselines(), &[61.0, 260.0, 76.0, 160.0]);
    }

    #[test]
    fn positive_boundary_is_inclusive() {
        let mut encoder = BaselineHoldTernaryEncoder::new(vec![1.0]);
        encoder.encode_ternary(&[10.0]);
        let output = encoder.encode_ternary(&[11.0]);
        assert_eq!(output, vec![1]);
    }

    #[test]
    fn negative_boundary_is_inclusive() {
        let mut encoder = BaselineHoldTernaryEncoder::new(vec![1.0]);
        encoder.encode_ternary(&[10.0]);
        let output = encoder.encode_ternary(&[9.0]);
        assert_eq!(output, vec![-1]);
    }

    #[test]
    fn encoder_trait_maps_ternary_sign_to_polarity() {
        let mut encoder = corinth_encoder();
        encoder.encode_ternary(&[60.0, 260.0, 77.0, 153.0]);
        let out = encoder.encode(&[60.0, 254.5, 77.0, 153.0]);
        assert_eq!(out.spikes.len(), 1);
        assert_eq!(out.spikes[0].channel, 1);
        assert!(!out.spikes[0].polarity);
    }

    #[test]
    fn reset_clears_seeded_baselines() {
        let mut encoder = corinth_encoder();
        encoder.encode_ternary(&[60.0, 260.0, 77.0, 153.0]);
        encoder.reset();
        assert!(!encoder.is_initialized());
        assert_eq!(encoder.baselines(), &[0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn try_new_rejects_invalid_thresholds() {
        assert!(BaselineHoldTernaryEncoder::try_new(vec![0.0]).is_ok());
        assert_eq!(
            BaselineHoldTernaryEncoder::try_new(vec![-1.0]).err(),
            Some(EncoderError::NonNegativeFinite {
                parameter: "threshold"
            })
        );
        assert_eq!(
            BaselineHoldTernaryEncoder::try_new(vec![1.0; u16::MAX as usize + 2]).err(),
            Some(EncoderError::NumChannelsTooLarge)
        );
    }

    #[cfg(feature = "serde")]
    #[test]
    fn serde_rejects_mismatched_state_lengths() {
        let value = serde_json::json!({
            "baselines": [1.0, 2.0],
            "thresholds": [1.0],
            "initialized": true,
        });
        let res: Result<BaselineHoldTernaryEncoder, _> = serde_json::from_value(value);
        assert!(res.is_err());
    }

    #[cfg(feature = "serde")]
    #[test]
    fn serde_rejects_non_finite_baselines() {
        let value = serde_json::json!({
            "baselines": [1.0, "NaN"],
            "thresholds": [1.0, 1.0],
            "initialized": true,
        });
        let res: Result<BaselineHoldTernaryEncoder, _> = serde_json::from_value(value);
        assert!(res.is_err());
    }

    #[cfg(feature = "serde")]
    #[test]
    fn serde_round_trips_live_state() {
        let mut encoder = BaselineHoldTernaryEncoder::new(vec![1.0, 2.0]);
        encoder.encode_ternary(&[10.0, 20.0]);
        encoder.encode_ternary(&[12.0, 20.0]);
        let json = serde_json::to_string(&encoder).expect("serialize");
        let restored: BaselineHoldTernaryEncoder =
            serde_json::from_str(&json).expect("deserialize");
        assert_eq!(restored, encoder);
    }

    #[test]
    fn empty_input_and_zero_channels_are_no_ops() {
        let mut zero = BaselineHoldTernaryEncoder::new(vec![]);
        assert!(zero.encode_ternary(&[1.0]).is_empty());

        let mut encoder = BaselineHoldTernaryEncoder::new(vec![1.0]);
        assert!(encoder.encode_ternary(&[]).is_empty());
        assert!(!encoder.is_initialized());
    }

    #[test]
    fn encode_step_truncates_excess_channels() {
        let mut encoder = BaselineHoldTernaryEncoder::new(vec![1.0, 1.0]);
        encoder.encode_step(&[10.0, 20.0, 99.0]);
        let out = encoder.encode_step(&[12.0, 20.0, 99.0]);
        assert_eq!(out.spikes.len(), 1);
        assert_eq!(out.spikes[0].channel, 0);
    }
}
