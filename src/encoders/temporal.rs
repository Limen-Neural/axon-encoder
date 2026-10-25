use crate::prelude::*;
use std::collections::VecDeque;

/// Dual-window change detector over a per-channel sample history.
///
/// Despite the name, this is **not** a temporal / time-to-first-spike code:
/// every spike lands at [`TickOffset::ZERO`](crate::time::TickOffset::ZERO)
/// with `polarity: true`, and the only information carried is *whether* the
/// recent mean moved away from the older mean on this step. For codes that put
/// information into spike *timing* use [`LatencyEncoder`] or [`PhaseEncoder`].
///
/// # Mathematical Model
///
/// Each channel keeps the newest [`Self::WINDOW`] (= 6) finite samples. Once
/// six are held, the step compares the mean of the newest three against the
/// mean of the three before them and fires when the absolute difference clears
/// a threshold:
///
/// ```text
/// change = |mean(history[-3:]) - mean(history[-6:-3])|
/// spike if change > threshold          (one spike per channel per step)
/// ```
///
/// `change_thresholds` is scanned from the largest configured value down and
/// the first one the change clears fires; since every band produces the same
/// spike, this is equivalent to `change > min(change_thresholds)`. Multiple
/// bands are accepted for configuration compatibility only.
///
/// # `history_depth` is buffer-only
///
/// `history_depth` sets how many finite samples each channel *retains*; it
/// must be at least [`Self::WINDOW`]. The detector always reads the newest six,
/// so `history_depth = 6` and `history_depth = 32` emit identical spikes on
/// identical input (locked by a test). The extra retention only affects what a
/// serde checkpoint carries. A configurable window is a possible future
/// extension; it is not implied by a larger depth today.
///
/// # Non-finite input
///
/// A `NaN` or `±inf` sample is **skipped**: it is not pushed into the channel's
/// history, the channel emits nothing for that step, and the window is left
/// exactly as it was. The next finite sample extends the history as if the
/// non-finite step never happened. Without the skip a `NaN` would sit in the
/// 3+3 window for six steps, silencing the channel and then producing a
/// delayed spurious spike as it slid out. This is the crate-wide rule for
/// stateful encoders (see also [`DerivativeEncoder`] and [`PredictiveEncoder`]);
/// constructor and serde validation of non-finite thresholds is unchanged.
///
/// [`DerivativeEncoder`]: crate::encoders::DerivativeEncoder
/// [`PredictiveEncoder`]: crate::encoders::PredictiveEncoder
/// [`LatencyEncoder`]: crate::encoders::LatencyEncoder
/// [`PhaseEncoder`]: crate::encoders::PhaseEncoder
///
/// # When to Use
///
/// - Detecting sudden changes in signal (edge detection) with some smoothing
///   over the 3+3 window, where [`DeltaEncoder`] would react to every sample
/// - Motion detection in video or sensor streams
/// - Event-based encoding where changes are more important than absolute values
///
/// [`DeltaEncoder`]: crate::encoders::DeltaEncoder
///
/// # Parameters
///
/// - `history_depth`: finite samples retained per channel (`>= 6`, buffer-only)
/// - `change_thresholds`: finite non-negative change magnitudes; the channel
///   fires when the change exceeds the smallest of them
/// - `num_channels`: number of input channels
///
/// # Migration (0.7)
///
/// Before 0.7 `change_thresholds` was `Vec<(f32, u16)>`, documented as
/// `(threshold, spike_value)`. The `u16` was never read: it did not set
/// polarity, channel, or anything else. It is removed; pass `vec![0.5]` where
/// you passed `vec![0.5]`. Checkpoints written with the pair form still
/// deserialize (the labels are dropped); new checkpoints write plain floats.
/// `PredictiveEncoder` keeps its pair form for now.
///
/// # Examples
///
/// ```rust
/// use axon_encoder::prelude::*;
/// # fn main() -> Result<(), EncoderError> {
/// // history_depth must be at least 6 for the dual-window change detector.
/// let mut enc = TemporalEncoder::try_new(6, vec![0.5], 1)?;
/// let mut fired = Vec::new();
/// for v in [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0] {
///     fired.push(enc.encode_step(&[v]).spikes.len());
/// }
/// // Six quiet samples fill the window; the jump then moves the recent mean.
/// assert_eq!(fired, [0, 0, 0, 0, 0, 0, 1]);
/// # Ok(())
/// # }
/// ```
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
pub struct TemporalEncoder {
    history: Vec<VecDeque<f32>>,
    history_depth: usize,
    change_thresholds: Vec<f32>,
}

impl TemporalEncoder {
    /// Samples the detector reads: three recent plus three older.
    ///
    /// `history_depth` must be at least this; anything beyond it is retained
    /// but never read.
    pub const WINDOW: usize = 6;

    /// Creates a new `TemporalEncoder`, panicking if configuration is invalid.
    ///
    /// Prefer [`try_new`](Self::try_new) for typed validation errors.
    ///
    /// # Panics
    ///
    /// Panics if `history_depth < 6` or `num_channels` is unsupported.
    pub fn new(history_depth: usize, change_thresholds: Vec<f32>, num_channels: usize) -> Self {
        Self::try_new(history_depth, change_thresholds, num_channels)
            .expect("invalid TemporalEncoder configuration")
    }

    /// Creates a new `TemporalEncoder`, returning an [`EncoderError`] for invalid configuration.
    ///
    /// Each threshold in `change_thresholds` must be finite and non-negative.
    pub fn try_new(
        history_depth: usize,
        change_thresholds: Vec<f32>,
        num_channels: usize,
    ) -> Result<Self, EncoderError> {
        if history_depth < Self::WINDOW {
            return Err(EncoderError::HistoryDepthTooSmall {
                minimum: Self::WINDOW,
            });
        }
        for &threshold in &change_thresholds {
            crate::error::validate_non_negative_finite("change_threshold", threshold)?;
        }
        crate::error::validate_channel_count(num_channels)?;
        Ok(Self {
            history: vec![VecDeque::with_capacity(history_depth); num_channels],
            history_depth,
            change_thresholds,
        })
    }

    /// Spike-emitting core: writes straight into `sink`, allocating nothing.
    ///
    /// Every public encoding path on this encoder routes through here, so the
    /// returning and sink-based APIs cannot drift apart.
    fn encode_with_threshold_scale_into<S: SpikeSink + ?Sized>(
        &mut self,
        input: &[f32],
        threshold_scale: f32,
        sink: &mut S,
    ) {
        for (i, &value) in input.iter().enumerate() {
            if i >= self.history.len() {
                break;
            }
            let Ok(channel) = u16::try_from(i) else {
                // Remaining channels exceed u16::MAX; stop rather than wrap.
                break;
            };
            // Skip rule: a non-finite sample never enters the history window.
            if !value.is_finite() {
                continue;
            }
            let Some(change) = self.push_and_measure_change(i, value) else {
                continue;
            };

            for &threshold in self.change_thresholds.iter().rev() {
                if change > (threshold * threshold_scale).max(0.0) {
                    sink.push(SpikeEvent::at_step_start(channel, true));
                    break; // Only fire one spike per channel per step
                }
            }
        }
    }

    /// Pushes a finite sample into the channel's buffer and returns the
    /// dual-window change over the newest [`Self::WINDOW`] samples, or `None`
    /// while fewer than that are held.
    fn push_and_measure_change(&mut self, channel_idx: usize, value: f32) -> Option<f32> {
        let channel_history = &mut self.history[channel_idx];
        if channel_history.len() == self.history_depth {
            channel_history.pop_front();
        }
        channel_history.push_back(value);

        if channel_history.len() < Self::WINDOW {
            return None;
        }

        // Sum in f64 so a window of `f32::MAX` samples cannot overflow to
        // `+inf` on the way to a representable mean; ordinary magnitudes round
        // to the same f32 as the direct f32 sum.
        let mut newest = channel_history.iter().rev().map(|&v| f64::from(v));
        let recent_avg = newest.by_ref().take(3).sum::<f64>() / 3.0;
        let older_avg = newest.take(3).sum::<f64>() / 3.0;
        Some((recent_avg - older_avg).abs() as f32)
    }

    fn encode_with_threshold_scale(
        &mut self,
        input: &[f32],
        threshold_scale: f32,
    ) -> EncodedOutput {
        let mut output = EncodedOutput::new();
        self.encode_with_threshold_scale_into(input, threshold_scale, &mut output.spikes);
        output
    }

    /// Streaming inputs are truncated to the configured channel count.
    fn clamp_to_channels<'a>(&self, input: &'a [f32]) -> &'a [f32] {
        if input.len() > self.history.len() {
            &input[..self.history.len()]
        } else {
            input
        }
    }
}

impl Encoder for TemporalEncoder {
    fn encode(&mut self, input: &[f32]) -> EncodedOutput {
        self.encode_with_threshold_scale(input, 1.0)
    }

    fn encode_step(&mut self, input: &[f32]) -> EncodedOutput {
        let safe_input = self.clamp_to_channels(input);
        self.encode_with_threshold_scale(safe_input, 1.0)
    }

    fn encode_into(&mut self, input: &[f32], sink: &mut dyn SpikeSink) {
        crate::sink::through_chunks(sink, |sink| {
            self.encode_with_threshold_scale_into(input, 1.0, sink)
        });
    }

    fn encode_step_into(&mut self, input: &[f32], sink: &mut dyn SpikeSink) {
        let safe_input = self.clamp_to_channels(input);
        crate::sink::through_chunks(sink, |sink| {
            self.encode_with_threshold_scale_into(safe_input, 1.0, sink)
        });
    }

    /// One call is one tick; batch and streaming are identical here.
    ///
    /// A channel emits at most one spike per call, at
    /// [`TickOffset::ZERO`](crate::time::TickOffset::ZERO) — the pattern lives
    /// in the *sequence* of steps, not in offsets within one. Ticks are
    /// dimensionless: one tick is one sample of the history window.
    fn time_model(&self) -> TimeModel {
        TimeModel::INSTANT
    }

    fn reset(&mut self) {
        for history in self.history.iter_mut() {
            history.clear();
        }
    }
}

impl ModulatedEncoder for TemporalEncoder {
    fn encode_with_gains(&mut self, input: &[f32], gains: EncodingGains) -> EncodedOutput {
        let safe_input = self.clamp_to_channels(input);
        self.encode_with_threshold_scale(safe_input, gains.sanitize().threshold_scale)
    }

    fn encode_with_gains_into(
        &mut self,
        input: &[f32],
        gains: EncodingGains,
        sink: &mut dyn SpikeSink,
    ) {
        let safe_input = self.clamp_to_channels(input);
        let threshold_scale = gains.sanitize().threshold_scale;
        crate::sink::through_chunks(sink, |sink| {
            self.encode_with_threshold_scale_into(safe_input, threshold_scale, sink)
        });
    }

    /// Mirrors [`encode_step_with_gains`], which this encoder leaves at its
    /// default of [`encode_with_gains`]. Without this the trait default would
    /// build and drain an intermediate `EncodedOutput`, so the streaming
    /// modulated path would allocate on every step.
    ///
    /// [`encode_step_with_gains`]: ModulatedEncoder::encode_step_with_gains
    /// [`encode_with_gains`]: ModulatedEncoder::encode_with_gains
    fn encode_step_with_gains_into(
        &mut self,
        input: &[f32],
        gains: EncodingGains,
        sink: &mut dyn SpikeSink,
    ) {
        self.encode_with_gains_into(input, gains, sink);
    }

    /// Skips the intermediate [`EncodedOutput`] the trait default builds.
    ///
    /// The default mirrors the returning [`encode_with_modulators`], which
    /// allocates. This encoder's modulator layer *is* its gains layer, so it
    /// can evaluate the curves and write straight into `sink`.
    ///
    /// [`encode_with_modulators`]: ModulatedEncoder::encode_with_modulators
    fn encode_with_modulators_into(
        &mut self,
        input: &[f32],
        modulators: &NeuroModulators,
        gain_curves: &NeuromodulatorGainCurves,
        sink: &mut dyn SpikeSink,
    ) {
        self.encode_with_gains_into(input, gain_curves.evaluate(modulators), sink);
    }

    /// Streaming counterpart of [`encode_with_modulators_into`], allocation-free
    /// for the same reason.
    ///
    /// [`encode_with_modulators_into`]: ModulatedEncoder::encode_with_modulators_into
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

#[cfg(feature = "serde")]
impl<'de> serde::Deserialize<'de> for TemporalEncoder {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        use std::collections::VecDeque;

        /// Pre-0.7 checkpoints wrote `(threshold, spike_value)` pairs; the
        /// label was never read, so it is dropped on load.
        #[derive(serde::Deserialize)]
        #[serde(untagged)]
        enum Threshold {
            Plain(f32),
            Legacy(f32, #[allow(dead_code)] u16),
        }

        impl From<Threshold> for f32 {
            fn from(value: Threshold) -> f32 {
                match value {
                    Threshold::Plain(threshold) | Threshold::Legacy(threshold, _) => threshold,
                }
            }
        }

        #[derive(serde::Deserialize)]
        struct Helper {
            history: Vec<VecDeque<f32>>,
            history_depth: usize,
            change_thresholds: Vec<Threshold>,
        }

        let helper = Helper::deserialize(deserializer)?;
        let change_thresholds: Vec<f32> = helper
            .change_thresholds
            .into_iter()
            .map(f32::from)
            .collect();

        if helper.history_depth < Self::WINDOW {
            return Err(serde::de::Error::custom(
                EncoderError::HistoryDepthTooSmall {
                    minimum: Self::WINDOW,
                },
            ));
        }
        crate::error::validate_channel_count(helper.history.len())
            .map_err(serde::de::Error::custom)?;

        // Match try_new: reject non-finite / negative thresholds on load.
        for &threshold in &change_thresholds {
            crate::error::validate_non_negative_finite("change_threshold", threshold)
                .map_err(serde::de::Error::custom)?;
        }

        validate_restored_history(&helper.history, helper.history_depth)
            .map_err(serde::de::Error::custom)?;

        Ok(Self {
            history: helper.history,
            history_depth: helper.history_depth,
            change_thresholds,
        })
    }
}

/// Per-channel checks for a restored checkpoint: history within
/// `history_depth`, and — mirroring `DerivativeEncoder`/`DeltaEncoder` — no
/// non-finite entry, so a checkpoint cannot re-poison the state the
/// encode-time skip rule keeps finite.
#[cfg(feature = "serde")]
fn validate_restored_history(
    history: &[VecDeque<f32>],
    history_depth: usize,
) -> Result<(), String> {
    for (i, deque) in history.iter().enumerate() {
        if deque.len() > history_depth {
            return Err(format!(
                "history channel {i} length ({}) exceeds history_depth ({history_depth})",
                deque.len()
            ));
        }
        if deque.iter().any(|value| !value.is_finite()) {
            return Err(format!("history channel {i} must be finite"));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ModulatedEncoder;

    #[test]
    fn test_temporal_encoder() {
        let mut encoder = TemporalEncoder::new(6, vec![2.0, 5.0], 1);
        let _output = encoder.encode(&[1.0]);
        let _output = encoder.encode(&[1.0]);
        let _output = encoder.encode(&[1.0]);
        let _output = encoder.encode(&[8.0]);
        let _output = encoder.encode(&[8.0]);
        let output = encoder.encode(&[8.0]);
        assert!(!output.spikes.is_empty());
    }

    #[test]
    fn test_temporal_encoder_modulators_reduce_threshold() {
        let mut encoder = TemporalEncoder::new(6, vec![4.5], 1);
        let modulators = NeuroModulators {
            tempo: 1.0,
            ..Default::default()
        };
        let gain_curves = NeuromodulatorGainCurves {
            tempo: ModulatorGainCurves {
                threshold: Some(GainCurve::new((0.0, 1.0), (1.0, 0.5))),
                ..Default::default()
            },
            ..Default::default()
        };

        for _ in 0..3 {
            encoder.encode(&[1.0]);
        }
        for _ in 0..2 {
            encoder.encode(&[5.0]);
        }
        assert!(encoder.encode(&[5.0]).spikes.is_empty());

        encoder.reset();

        for _ in 0..3 {
            encoder.encode_step_with_modulators(&[1.0], &modulators, &gain_curves);
        }
        for _ in 0..2 {
            encoder.encode_step_with_modulators(&[5.0], &modulators, &gain_curves);
        }
        let output = encoder.encode_step_with_modulators(&[5.0], &modulators, &gain_curves);
        assert_eq!(output.spikes.len(), 1);
    }

    #[test]
    fn test_temporal_encoder_encode_with_modulators() {
        let mut encoder = TemporalEncoder::new(6, vec![4.5], 1);
        let modulators = NeuroModulators {
            tempo: 1.0,
            ..Default::default()
        };
        let gain_curves = NeuromodulatorGainCurves {
            tempo: ModulatorGainCurves {
                threshold: Some(GainCurve::new((0.0, 1.0), (1.0, 0.5))),
                ..Default::default()
            },
            ..Default::default()
        };

        for _ in 0..3 {
            encoder.encode_with_modulators(&[1.0], &modulators, &gain_curves);
        }
        for _ in 0..2 {
            encoder.encode_with_modulators(&[5.0], &modulators, &gain_curves);
        }
        let output = encoder.encode_with_modulators(&[5.0], &modulators, &gain_curves);
        assert_eq!(output.spikes.len(), 1);
    }

    /// Skip rule: NaN / ±inf never enter the history, emit nothing, and the
    /// live encoder matches a fresh one fed only the finite samples.
    #[test]
    fn test_temporal_encoder_skips_non_finite_without_poisoning_history() {
        for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let mut live = TemporalEncoder::new(6, vec![0.5], 2);
            let mut fresh = TemporalEncoder::new(6, vec![0.5], 1);

            for _ in 0..3 {
                assert!(live.encode_step(&[0.0, 0.0]).spikes.is_empty());
                assert!(fresh.encode_step(&[0.0]).spikes.is_empty());
            }
            // Mixed-channel slice: channel 0 poisoned, channel 1 finite.
            let out = live.encode_step(&[bad, 0.0]);
            assert!(out.spikes.is_empty(), "{bad}: no fire on non-finite step");
            assert_eq!(live.history[0].len(), 3, "{bad}: must not be stored");
            assert_eq!(live.history[1].len(), 4);
            assert!(live.history.iter().flatten().all(|v| v.is_finite()));

            // Channel 0 of the live encoder behaves exactly like a fresh
            // encoder that never saw the non-finite step: the jump fires on the
            // same steps, and once the window is flat neither emits a delayed
            // spurious spike as the old samples slide out.
            for step in 0..12 {
                let live_ch0: Vec<_> = live
                    .encode_step(&[2.0, 2.0])
                    .spikes
                    .into_iter()
                    .filter(|s| s.channel == 0)
                    .collect();
                let fresh_out = fresh.encode_step(&[2.0]).spikes;
                assert_eq!(live_ch0, fresh_out, "{bad}: finite recovery at step {step}");
                assert_eq!(
                    live_ch0.len(),
                    usize::from((2..5).contains(&step)),
                    "{bad}: step {step}"
                );
            }
        }
    }

    /// Finite-but-huge samples: the window means must not overflow, so a
    /// flat `f32::MAX` signal stays silent and the checkpoint round-trips.
    #[test]
    fn test_temporal_encoder_f32_max_inputs_keep_means_finite() {
        let mut enc = TemporalEncoder::new(6, vec![0.5], 1);
        for step in 0..8 {
            let out = enc.encode_step(&[f32::MAX]);
            assert!(
                out.spikes.is_empty(),
                "step {step}: flat signal must stay silent"
            );
        }
        assert!(enc.history[0].iter().all(|v| v.is_finite()));
        #[cfg(feature = "serde")]
        {
            let json = serde_json::to_string(&enc).expect("serialize");
            let restored: TemporalEncoder = serde_json::from_str(&json).expect("round-trip");
            assert_eq!(restored, enc);
        }
    }

    #[test]
    fn test_temporal_encoder_skip_rule_holds_on_every_public_path() {
        let run = |f: &mut dyn FnMut(&mut TemporalEncoder, &[f32]) -> Vec<SpikeEvent>| {
            let mut enc = TemporalEncoder::new(6, vec![0.5], 1);
            for _ in 0..3 {
                assert!(f(&mut enc, &[0.0]).is_empty());
            }
            assert!(f(&mut enc, &[f32::NAN]).is_empty());
            assert!(f(&mut enc, &[f32::NEG_INFINITY]).is_empty());
            assert_eq!(enc.history[0].len(), 3);
            for _ in 0..2 {
                assert!(f(&mut enc, &[2.0]).is_empty());
            }
            assert_eq!(f(&mut enc, &[2.0]).len(), 1);
        };
        run(&mut |e, x| e.encode(x).spikes);
        run(&mut |e, x| e.encode_step(x).spikes);
        run(&mut |e, x| {
            let mut sink = Vec::new();
            e.encode_into(x, &mut sink);
            sink
        });
        run(&mut |e, x| {
            let mut sink = Vec::new();
            e.encode_step_into(x, &mut sink);
            sink
        });
    }

    #[test]
    fn test_temporal_encoder_step_longer_input() {
        let mut encoder = TemporalEncoder::new(6, vec![4.5], 2);
        let output = encoder.encode_step(&[1.0, 2.0, 3.0]);
        assert!(output.spikes.len() <= 2);
    }

    #[cfg(feature = "serde")]
    #[test]
    fn test_temporal_serde_history_channel_too_long() {
        let json = r#"{
            "history": [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
            "history_depth": 6,
            "change_thresholds": []
        }"#;
        let res: Result<TemporalEncoder, _> = serde_json::from_str(json);
        assert!(res.is_err());
    }

    #[cfg(feature = "serde")]
    #[test]
    fn test_temporal_serde_rejects_invalid_thresholds() {
        let negative = r#"{
            "history": [[]],
            "history_depth": 6,
            "change_thresholds": [-0.5]
        }"#;
        let res: Result<TemporalEncoder, _> = serde_json::from_str(negative);
        assert!(
            res.is_err(),
            "negative change_threshold must fail deserialize"
        );

        let ok = r#"{
            "history": [[]],
            "history_depth": 6,
            "change_thresholds": [0.0, 1.5]
        }"#;
        let res: Result<TemporalEncoder, _> = serde_json::from_str(ok);
        assert!(res.is_ok());
    }
    /// A checkpoint carrying NaN/±inf in `history` must not restore: it would
    /// re-poison the state the encode-time skip rule keeps finite. JSON cannot
    /// carry those values (`serde_json` writes `null`, rejected as a type
    /// error before our check), so this goes through RON, which can, and
    /// asserts the explicit finiteness guard fires with its own message.
    #[cfg(feature = "serde")]
    #[test]
    fn test_temporal_serde_rejects_non_finite_history() {
        for bad in ["NaN", "inf", "-inf"] {
            let payload = format!(
                "(history: [[0.0], [0.0, {bad}]], history_depth: 6, change_thresholds: [0.5])"
            );
            let err = ron::from_str::<TemporalEncoder>(&payload)
                .expect_err("non-finite history must not restore");
            assert!(
                err.to_string().contains("history channel 1 must be finite"),
                "{bad}: unexpected error {err}"
            );
        }
        let finite =
            r#"{"history": [[1e30, -1e30]], "history_depth": 6, "change_thresholds": [0.5]}"#;
        assert!(serde_json::from_str::<TemporalEncoder>(finite).is_ok());
    }

    #[cfg(feature = "serde")]
    #[test]
    fn test_temporal_validate_restored_history_reports_each_reason() {
        let ok = vec![VecDeque::from(vec![1.0f32; 6])];
        assert!(validate_restored_history(&ok, 6).is_ok());
        assert!(
            validate_restored_history(&ok, 5)
                .unwrap_err()
                .contains("exceeds")
        );
        let poisoned = vec![VecDeque::new(), VecDeque::from(vec![0.0, f32::NAN])];
        assert_eq!(
            validate_restored_history(&poisoned, 6),
            Err("history channel 1 must be finite".to_string())
        );
    }

    /// `history_depth` is buffer-only: depth 6 and depth 32 emit identical
    /// spikes on identical input, because the detector reads the newest
    /// [`TemporalEncoder::WINDOW`] samples regardless of retention.
    #[test]
    fn test_temporal_encoder_history_depth_beyond_window_is_buffer_only() {
        let signal: Vec<f32> = (0..40)
            .map(|i| match i % 10 {
                0..=4 => 0.0,
                _ => 3.0,
            })
            .collect();
        let mut shallow = TemporalEncoder::new(TemporalEncoder::WINDOW, vec![0.5], 1);
        let mut deep = TemporalEncoder::new(32, vec![0.5], 1);
        let shallow_out: Vec<_> = signal
            .iter()
            .map(|&v| shallow.encode_step(&[v]).spikes)
            .collect();
        let deep_out: Vec<_> = signal
            .iter()
            .map(|&v| deep.encode_step(&[v]).spikes)
            .collect();
        assert_eq!(shallow_out, deep_out, "depth must not change the algorithm");
        assert!(
            shallow_out.iter().any(|s| !s.is_empty()),
            "fixture must fire"
        );
        assert_eq!((shallow.history[0].len(), deep.history[0].len()), (6, 32));
    }

    /// Several thresholds behave as the smallest one: every band emits the
    /// same spike, so the extra bands change nothing.
    #[test]
    fn test_temporal_encoder_multiple_thresholds_equal_the_smallest() {
        let drive = |thresholds: Vec<f32>| {
            let mut enc = TemporalEncoder::new(6, thresholds, 1);
            [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
                .iter()
                .map(|&v| enc.encode_step(&[v]).spikes.len())
                .collect::<Vec<_>>()
        };
        assert_eq!(drive(vec![0.5, 2.0, 5.0]), drive(vec![0.5]));
    }

    /// Pre-0.7 checkpoints wrote `(threshold, spike_value)` pairs; they still
    /// load with the label dropped, and HEAD writes plain floats.
    #[cfg(feature = "serde")]
    #[test]
    fn test_temporal_serde_accepts_legacy_threshold_pairs() {
        let legacy =
            r#"{"history": [[]], "history_depth": 6, "change_thresholds": [[0.5, 1], [1.5, 3]]}"#;
        let restored: TemporalEncoder = serde_json::from_str(legacy).expect("legacy pairs load");
        assert_eq!(restored, TemporalEncoder::new(6, vec![0.5, 1.5], 1));
        let written = serde_json::to_string(&restored).expect("serialize");
        assert!(
            written.contains(r#""change_thresholds":[0.5,1.5]"#),
            "{written}"
        );
    }

    #[test]
    fn test_temporal_encoder_try_new_validation() {
        assert_eq!(
            TemporalEncoder::try_new(5, vec![1.0], 1).err(),
            Some(EncoderError::HistoryDepthTooSmall { minimum: 6 })
        );
        assert_eq!(
            TemporalEncoder::try_new(6, vec![1.0], u16::MAX as usize + 2).err(),
            Some(EncoderError::NumChannelsTooLarge)
        );
        assert_eq!(
            TemporalEncoder::try_new(6, vec![f32::NAN], 1).err(),
            Some(EncoderError::NonNegativeFinite {
                parameter: "change_threshold"
            })
        );
        assert_eq!(
            TemporalEncoder::try_new(6, vec![-0.5], 1).err(),
            Some(EncoderError::NonNegativeFinite {
                parameter: "change_threshold"
            })
        );
        assert!(TemporalEncoder::try_new(6, vec![0.0], 1).is_ok());
    }
}
