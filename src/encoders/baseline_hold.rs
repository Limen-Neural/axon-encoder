use crate::prelude::*;

/// Baseline-hold ternary encoder (Corinth `TelemetryEncoder` semantics).
///
/// Each channel keeps a baseline that moves **only** when a signed threshold
/// crossing fires. Each channel seeds independently on its first observed
/// finite sample: that sample is copied into the baseline and the channel emits zero.
/// A shorter input seeds only the channels it contains; the tail stays unseeded.
///
/// Crossing rules for an already-seeded channel (inclusive boundaries):
///
/// - positive event when `current - baseline >= threshold`
/// - negative event when `current - baseline <= -threshold`
/// - otherwise zero; baseline unchanged
///
/// A `NaN` or `±inf` sample is **skipped**: it does not seed an unseeded
/// channel, does not snap a seeded baseline, and emits `0` / no spike. The
/// next finite sample behaves as if the non-finite step never happened. This
/// is the crate-wide rule for stateful encoders (see [`DeltaEncoder`],
/// [`DerivativeEncoder`], [`TemporalEncoder`], [`PredictiveEncoder`]); serde
/// restore still rejects non-finite `baselines`.
///
/// Use [`encode_ternary`](Self::encode_ternary) for dense `{-1, 0, +1}` output.
/// The [`Encoder`] implementation maps nonzero ternary values to
/// [`SpikeEvent`] polarity at [`TickOffset::ZERO`](crate::time::TickOffset::ZERO).
///
/// [`Encoder::reset`] returns every channel to unseeded with baseline `0.0`.
/// A Corinth `TelemetryFunnel::reset` must **not** call this `reset`: Corinth
/// keeps the telemetry baseline across GIF reset.
///
/// Extracted from [corinth-canal#158](https://github.com/rmems/corinth-canal/issues/158).
/// Joint Corinth adoption of this crate together with neuromod is still blocked
/// by the existing [`NeuroModulators`] / [`PoissonEncoder`] name collision
/// ([corinth-canal#143](https://github.com/rmems/corinth-canal/issues/143)).
/// Not interchangeable with [`DeltaEncoder`] (absolute delta) or
/// [`DerivativeEncoder`] (baseline advances every step, strict `>` / `<` bounds).
///
/// [`DeltaEncoder`]: crate::encoders::DeltaEncoder
/// [`DerivativeEncoder`]: crate::encoders::DerivativeEncoder
/// [`TemporalEncoder`]: crate::encoders::TemporalEncoder
/// [`PredictiveEncoder`]: crate::encoders::PredictiveEncoder
/// [`NeuroModulators`]: crate::modulators::NeuroModulators
/// [`PoissonEncoder`]: crate::poisson::PoissonEncoder
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(try_from = "BaselineHoldTernaryEncoderRepr"))]
pub struct BaselineHoldTernaryEncoder {
    baselines: Vec<f32>,
    thresholds: Vec<f32>,
    seeded: Vec<bool>,
}

/// Sample observed on one configured channel.
#[derive(Clone, Copy)]
struct ChannelSample {
    index: usize,
    value: f32,
}

#[cfg(feature = "serde")]
#[derive(serde::Deserialize)]
struct BaselineHoldTernaryEncoderRepr {
    baselines: Vec<f32>,
    thresholds: Vec<f32>,
    seeded: Vec<bool>,
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
        if r.seeded.len() != r.thresholds.len() {
            return Err(format!(
                "mismatched seeded length ({}) and thresholds length ({})",
                r.seeded.len(),
                r.thresholds.len()
            ));
        }
        if r.baselines.iter().any(|v| !v.is_finite()) {
            return Err("baselines must be finite".into());
        }
        let mut encoder = Self::try_new(r.thresholds).map_err(|error| error.to_string())?;
        encoder.baselines = r.baselines;
        encoder.seeded = r.seeded;
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
            seeded: vec![false; num_channels],
        })
    }

    /// Returns the per-channel baselines (same length as configured thresholds).
    pub fn baselines(&self) -> &[f32] {
        &self.baselines
    }

    /// Whether every configured channel has observed a sample.
    ///
    /// Stays false while any channel is unseeded, including after a short frame
    /// and for a zero-channel encoder.
    pub fn is_initialized(&self) -> bool {
        !self.seeded.is_empty() && self.seeded.iter().all(|seeded| *seeded)
    }

    /// Dense ternary encoding: `-1`, `0`, or `+1` per processed channel.
    ///
    /// Input is truncated to the configured channel count. An empty slice is a
    /// no-op and returns an empty vector. A channel that has never been seeded
    /// copies the sample into its baseline and emits `0`.
    pub fn encode_ternary(&mut self, input: &[f32]) -> Vec<i8> {
        let values = self.active_values(input);
        self.encode_ternary_values(values)
    }

    fn encode_ternary_values(&mut self, values: &[f32]) -> Vec<i8> {
        if values.is_empty() {
            return Vec::new();
        }
        let mut out = vec![0i8; values.len()];
        for (index, &value) in values.iter().enumerate() {
            // Skip rule (crate-wide for stateful encoders): a non-finite
            // sample neither seeds, snaps, nor fires; the channel is untouched.
            if !value.is_finite() {
                continue;
            }
            let sample = ChannelSample { index, value };
            out[index] = if self.seeded[index] {
                self.classify_and_snap(sample)
            } else {
                self.seed_channel(sample)
            };
        }
        out
    }

    fn active_values<'a>(&self, input: &'a [f32]) -> &'a [f32] {
        let channel_count = self.thresholds.len();
        if channel_count == 0 {
            &[]
        } else {
            &input[..input.len().min(channel_count)]
        }
    }

    fn seed_channel(&mut self, sample: ChannelSample) -> i8 {
        self.baselines[sample.index] = sample.value;
        self.seeded[sample.index] = true;
        0
    }

    fn classify_and_snap(&mut self, sample: ChannelSample) -> i8 {
        let event = self.crossing_event(sample);
        if event != 0 {
            self.baselines[sample.index] = sample.value;
        }
        event
    }

    fn crossing_event(&self, sample: ChannelSample) -> i8 {
        let delta = sample.value - self.baselines[sample.index];
        let threshold = self.thresholds[sample.index];
        if delta >= threshold {
            1
        } else if delta <= -threshold {
            -1
        } else {
            0
        }
    }

    fn encode_step_into_sink<S: SpikeSink + ?Sized>(&mut self, input: &[f32], sink: &mut S) {
        let values = self.active_values(input);
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
}

impl Encoder for BaselineHoldTernaryEncoder {
    fn encode(&mut self, input: &[f32]) -> EncodedOutput {
        self.encode_step(input)
    }

    fn encode_step(&mut self, input: &[f32]) -> EncodedOutput {
        let safe_input = self.active_values(input);
        let mut output = EncodedOutput::new();
        self.encode_step_into_sink(safe_input, &mut output.spikes);
        output
    }

    fn encode_into(&mut self, input: &[f32], sink: &mut dyn SpikeSink) {
        crate::sink::through_chunks(sink, |sink| self.encode_step_into_sink(input, sink));
    }

    fn encode_step_into(&mut self, input: &[f32], sink: &mut dyn SpikeSink) {
        let safe_input = self.active_values(input);
        crate::sink::through_chunks(sink, |sink| self.encode_step_into_sink(safe_input, sink));
    }

    fn time_model(&self) -> TimeModel {
        TimeModel::INSTANT
    }

    fn reset(&mut self) {
        self.baselines.fill(0.0);
        self.seeded.fill(false);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const CORINTH_THRESHOLDS: [f32; 4] = [1.0, 5.0, 1.0, 5.0];

    fn corinth_encoder() -> BaselineHoldTernaryEncoder {
        BaselineHoldTernaryEncoder::new(CORINTH_THRESHOLDS.to_vec())
    }

    fn seeded_prefix() -> BaselineHoldTernaryEncoder {
        let mut encoder = BaselineHoldTernaryEncoder::new(vec![1.0; 4]);
        encoder.encode_ternary(&[10.0, 20.0]);
        encoder
    }

    const NON_FINITE: [f32; 3] = [f32::NAN, f32::INFINITY, f32::NEG_INFINITY];

    /// Skip rule: a non-finite first sample does not seed the channel.
    #[test]
    fn non_finite_sample_does_not_seed() {
        for bad in NON_FINITE {
            let mut encoder = BaselineHoldTernaryEncoder::new(vec![1.0, 1.0]);
            assert_eq!(encoder.encode_ternary(&[bad, 5.0]), vec![0, 0], "{bad}");
            assert_eq!(encoder.seeded, vec![false, true], "{bad}: must not seed");
            assert_eq!(encoder.baselines(), &[0.0, 5.0], "{bad}: must not store");
        }
    }

    /// Skip rule: a non-finite sample on a seeded channel neither fires nor
    /// snaps, and the next finite sample is judged against the old baseline.
    #[test]
    fn non_finite_sample_does_not_snap_or_fire() {
        for bad in NON_FINITE {
            let mut encoder = BaselineHoldTernaryEncoder::new(vec![1.0]);
            encoder.encode_ternary(&[10.0]);
            assert_eq!(encoder.encode_ternary(&[bad]), vec![0], "{bad}: no fire");
            assert_eq!(encoder.baselines(), &[10.0], "{bad}: baseline untouched");
            assert_eq!(
                encoder.encode_ternary(&[11.0]),
                vec![1],
                "{bad}: finite recovery"
            );
        }
    }

    #[test]
    fn non_finite_sample_is_silent_on_spike_paths() {
        for bad in NON_FINITE {
            let mut encoder = BaselineHoldTernaryEncoder::new(vec![1.0]);
            encoder.encode_ternary(&[10.0]);
            let mut sink = Vec::new();
            encoder.encode_into(&[bad], &mut sink);
            encoder.encode_step_into(&[bad], &mut sink);
            assert!(encoder.encode(&[bad]).spikes.is_empty(), "{bad}");
            assert!(encoder.encode_step(&[bad]).spikes.is_empty(), "{bad}");
            assert!(sink.is_empty(), "{bad}: sink paths honour the skip rule");
        }
    }

    #[test]
    fn first_encode_emits_zeroes_until_every_channel_is_seeded() {
        let mut encoder = corinth_encoder();
        let snap = [60.0, 260.0, 77.0, 153.0];
        assert_eq!(encoder.encode_ternary(&snap), vec![0, 0, 0, 0]);
        assert!(encoder.is_initialized());
    }

    #[test]
    fn first_encode_copies_each_sample_into_its_baseline() {
        let mut encoder = corinth_encoder();
        encoder.encode_ternary(&[60.0, 260.0, 77.0, 153.0]);
        assert_eq!(encoder.seeded, vec![true; 4]);
        assert_eq!(encoder.baselines(), &[60.0, 260.0, 77.0, 153.0]);
    }

    #[test]
    fn short_first_frame_seeds_only_observed_channels() {
        let mut encoder = BaselineHoldTernaryEncoder::new(vec![1.0; 4]);
        assert_eq!(encoder.encode_ternary(&[10.0, 20.0]), vec![0, 0]);
        assert_eq!(encoder.baselines(), &[10.0, 20.0, 0.0, 0.0]);
    }

    #[test]
    fn short_first_frame_leaves_the_encoder_uninitialized() {
        let encoder = seeded_prefix();
        assert_eq!(encoder.seeded, vec![true, true, false, false]);
        assert!(!encoder.is_initialized());
    }

    #[test]
    fn unseeded_tail_adopts_the_sample_instead_of_firing() {
        // Channels 0 and 1 follow crossing rules. Channels 2 and 3 would cross
        // a zero baseline, but they have never been seeded, so they emit 0.
        let mut encoder = seeded_prefix();
        assert_eq!(
            encoder.encode_ternary(&[12.0, 18.0, 50.0, -40.0]),
            vec![1, -1, 0, 0]
        );
        assert_eq!(encoder.seeded, vec![true; 4]);
    }

    #[test]
    fn seeding_sample_becomes_the_next_baseline() {
        let mut encoder = seeded_prefix();
        encoder.encode_ternary(&[12.0, 18.0, 50.0, -40.0]);
        assert_eq!(encoder.baselines(), &[12.0, 18.0, 50.0, -40.0]);
    }

    #[test]
    fn seeded_tail_uses_the_inclusive_crossing_rule() {
        let mut encoder = seeded_prefix();
        encoder.encode_ternary(&[12.0, 18.0, 50.0, -40.0]);
        assert_eq!(
            encoder.encode_ternary(&[12.0, 18.0, 51.0, -40.5]),
            vec![0, 0, 1, 0]
        );
        assert_eq!(encoder.baselines(), &[12.0, 18.0, 51.0, -40.0]);
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
        encoder.encode_ternary(&[60.0, 260.0]);
        encoder.reset();
        assert_eq!(encoder.seeded, vec![false; 4]);
        assert_eq!(encoder.baselines(), &[0.0; 4]);
    }

    #[test]
    fn reset_makes_the_next_sample_seed_again() {
        let mut encoder = corinth_encoder();
        encoder.encode_ternary(&[60.0, 260.0]);
        encoder.reset();
        assert!(!encoder.is_initialized());
        assert_eq!(
            encoder.encode_ternary(&[60.0, 260.0, 77.0, 153.0]),
            vec![0, 0, 0, 0]
        );
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
    fn serde_rejects_invalid_state() {
        let too_many_channels = (u16::MAX as usize) + 2;
        let payloads = [
            serde_json::json!({
                "baselines": [1.0, 2.0],
                "thresholds": [1.0],
                "seeded": [true],
            }),
            serde_json::json!({
                "baselines": [1.0, 2.0],
                "thresholds": [1.0, 1.0],
                "seeded": [true],
            }),
            serde_json::json!({
                "baselines": [0.0],
                "thresholds": [-1.0],
                "seeded": [false],
            }),
            serde_json::json!({
                "baselines": vec![0.0; too_many_channels],
                "thresholds": vec![1.0; too_many_channels],
                "seeded": vec![false; too_many_channels],
            }),
        ];
        let json_rejected = payloads
            .into_iter()
            .all(|payload| serde_json::from_value::<BaselineHoldTernaryEncoder>(payload).is_err());
        let non_finite = [
            BaselineHoldTernaryEncoderRepr {
                baselines: vec![1.0, f32::NAN],
                thresholds: vec![1.0, 1.0],
                seeded: vec![true, false],
            },
            BaselineHoldTernaryEncoderRepr {
                baselines: vec![f32::INFINITY],
                thresholds: vec![1.0],
                seeded: vec![true],
            },
        ];
        let non_finite_rejected = non_finite
            .into_iter()
            .all(|repr| BaselineHoldTernaryEncoder::try_from(repr).is_err());
        assert!(json_rejected);
        assert!(non_finite_rejected);
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

    #[cfg(feature = "serde")]
    #[test]
    fn serde_round_trips_partial_channel_seeding() {
        let mut encoder = BaselineHoldTernaryEncoder::new(vec![1.0; 4]);
        encoder.encode_ternary(&[10.0, 20.0]);
        let json = serde_json::to_string(&encoder).expect("serialize");
        let mut restored: BaselineHoldTernaryEncoder =
            serde_json::from_str(&json).expect("deserialize");
        assert_eq!(restored, encoder);
        let frame = [12.0, 18.0, 50.0, -40.0];
        assert_eq!(
            encoder.encode_ternary(&frame),
            restored.encode_ternary(&frame)
        );
        assert_eq!(encoder, restored);
    }

    #[test]
    fn zero_channel_encoder_ignores_input() {
        let mut zero = BaselineHoldTernaryEncoder::new(vec![]);
        assert!(zero.encode_ternary(&[1.0]).is_empty());
        assert!(!zero.is_initialized());
    }

    #[test]
    fn empty_input_does_not_seed_a_channel() {
        let mut encoder = BaselineHoldTernaryEncoder::new(vec![1.0]);
        assert!(encoder.encode_ternary(&[]).is_empty());
        assert_eq!(encoder.baselines(), &[0.0]);
        assert_eq!(encoder.seeded, vec![false]);
    }

    #[test]
    fn encode_step_truncates_excess_channels() {
        let mut encoder = BaselineHoldTernaryEncoder::new(vec![1.0, 1.0]);
        encoder.encode_step(&[10.0, 20.0, 99.0]);
        let out = encoder.encode_step(&[12.0, 20.0, 99.0]);
        assert_eq!(out.spikes.len(), 1);
        assert_eq!(out.spikes[0].channel, 0);
        assert_eq!(encoder.baselines(), &[12.0, 20.0]);
    }
}
