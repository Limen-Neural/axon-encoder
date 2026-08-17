use crate::prelude::*;

/// Encodes analog values into latency-coded spike times
///
/// Each input channel produces exactly one positive spike whose timestamp is
/// determined by the input strength within the configured range. Stronger
/// inputs fire earlier. Values below the range minimum map to the latest
/// possible spike at `max_latency`, and values above the range maximum map to
/// timestamp `0`.
///
/// # Time semantics
///
/// One call is one *presentation* spanning `max_latency + 1` ticks, and offsets
/// are relative to the start of that call — so repeated calls with the same
/// input produce the same offsets, and it is the caller's
/// [`TimeCursor`](crate::time::TimeCursor) that separates them in absolute
/// time. `encode` and `encode_step` are the same stateless path.
///
/// Spikes are emitted in channel order, *not* in time order: a later channel
/// can carry an earlier offset. Sort by `timestamp` if a consumer needs a
/// chronological stream.
///
/// # Examples
///
/// ```rust
/// use axon_encoder::prelude::*;
/// # fn main() -> Result<(), EncoderError> {
/// let mut enc = LatencyEncoder::try_new(10, (0.0, 1.0))?;
/// let out = enc.encode(&[1.0, 0.0]); // strong → early, weak → late
/// assert_eq!(out.spikes.len(), 2);
/// assert!(out.spikes[0].timestamp <= out.spikes[1].timestamp);
///
/// // The window is max_latency + 1 ticks wide, so presentations do not overlap.
/// assert_eq!(enc.time_model().span_ticks(), 11);
/// assert_eq!(enc.time_model().step_ticks(), 11);
/// # Ok(())
/// # }
/// ```
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
pub struct LatencyEncoder {
    max_latency: u64,
    range: (f32, f32),
}

impl LatencyEncoder {
    /// Creates a new `LatencyEncoder`, panicking if configuration is invalid.
    ///
    /// Prefer [`try_new`](Self::try_new) for typed validation errors.
    ///
    /// # Panics
    ///
    /// Panics if `range.0 >= range.1` or either bound is non-finite.
    ///
    /// `max_latency == 0` is valid and emits every spike at timestamp `0`
    /// (instantaneous response). `max_latency == u64::MAX` is rejected.
    pub fn new(max_latency: u64, range: (f32, f32)) -> Self {
        Self::try_new(max_latency, range).unwrap_or_else(|error| panic!("{error}"))
    }

    /// Creates a new `LatencyEncoder`, returning an [`EncoderError`] for invalid configuration.
    ///
    /// `max_latency == 0` is accepted and maps every input to timestamp `0`.
    ///
    /// `max_latency == u64::MAX` is rejected with
    /// [`EncoderError::WindowTooLarge`]: the presentation window is
    /// `max_latency + 1` ticks, so that value would leave the declared
    /// [`time_model`](Encoder::time_model) span unable to contain a spike at
    /// `max_latency`.
    pub fn try_new(max_latency: u64, range: (f32, f32)) -> Result<Self, EncoderError> {
        crate::error::validate_range("range", range)?;
        if max_latency == u64::MAX {
            return Err(EncoderError::WindowTooLarge {
                parameter: "max_latency",
            });
        }
        Ok(Self { max_latency, range })
    }

    fn normalize(&self, value: f32) -> f64 {
        // Use f64 to prevent overflow for valid f32 ranges (e.g., f32::MIN..f32::MAX).
        let clamped = value.clamp(self.range.0, self.range.1) as f64;
        let lo = self.range.0 as f64;
        let hi = self.range.1 as f64;
        (clamped - lo) / (hi - lo)
    }

    fn timestamp_for(&self, value: f32) -> TickOffset {
        if self.max_latency == 0 {
            return TickOffset::ZERO;
        }
        if value.is_nan() {
            return TickOffset::new(self.max_latency);
        }

        let normalized = self.normalize(value);
        TickOffset::new(((1.0 - normalized) * self.max_latency as f64).round() as u64)
    }

    fn timestamp_for_with_latency_scale(&self, value: f32, latency_scale: f32) -> TickOffset {
        // Clamp to `max_latency` so a gain above 1.0 cannot push a spike past
        // the configured window — `time_model().span_ticks()` is a hard bound.
        let scaled_latency = ((self.max_latency as f64) * (latency_scale as f64)).round() as u64;
        let scaled_latency = scaled_latency.min(self.max_latency);
        if scaled_latency == 0 {
            return TickOffset::ZERO;
        }
        if value.is_nan() {
            return TickOffset::new(scaled_latency);
        }

        let normalized = self.normalize(value);
        TickOffset::new(((1.0 - normalized) * scaled_latency as f64).round() as u64)
    }

    fn encode_with_latency_scale(&mut self, input: &[f32], latency_scale: f32) -> EncodedOutput {
        let mut output = EncodedOutput::new();
        output.spikes.reserve(input.len());

        for (channel, &value) in input.iter().enumerate() {
            let Ok(channel) = u16::try_from(channel) else {
                // Remaining channels exceed u16::MAX; stop rather than wrap.
                break;
            };
            output.spikes.push(SpikeEvent::new(
                channel,
                self.timestamp_for_with_latency_scale(value, latency_scale),
                true,
            ));
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

impl Encoder for LatencyEncoder {
    fn encode(&mut self, input: &[f32]) -> EncodedOutput {
        let mut output = EncodedOutput::new();
        output.spikes.reserve(input.len());

        for (channel, &value) in input.iter().enumerate() {
            let Ok(channel) = u16::try_from(channel) else {
                // Remaining channels exceed u16::MAX; stop rather than wrap.
                break;
            };
            output
                .spikes
                .push(SpikeEvent::new(channel, self.timestamp_for(value), true));
        }

        output
    }

    fn encode_step(&mut self, input: &[f32]) -> EncodedOutput {
        self.encode(input)
    }

    /// A non-overlapping presentation window of `max_latency + 1` ticks.
    ///
    /// Offsets span `0..=max_latency`, so the window is one tick wider than
    /// `max_latency`, and the caller's origin advances by the whole window per
    /// call. Ticks are dimensionless: `max_latency` is a tick budget, not a
    /// duration, so the caller sets the physical tick length.
    ///
    /// Neuromodulated latency gains shorten the window but never stretch it past
    /// `max_latency`, so `span_ticks` bounds modulated output too.
    fn time_model(&self) -> TimeModel {
        // `try_new` rejects `u64::MAX`, so the window is always representable.
        TimeModel::window(self.max_latency + 1)
    }

    fn reset(&mut self) {
        // Stateless encoder.
    }
}

impl ModulatedEncoder for LatencyEncoder {
    fn encode_with_gains(&mut self, input: &[f32], gains: EncodingGains) -> EncodedOutput {
        self.encode_with_latency_scale(input, gains.sanitize().latency_scale)
    }
}

#[cfg(feature = "serde")]
impl<'de> serde::Deserialize<'de> for LatencyEncoder {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(serde::Deserialize)]
        struct Helper {
            max_latency: u64,
            range: (f32, f32),
        }

        let helper = Helper::deserialize(deserializer)?;

        Self::try_new(helper.max_latency, helper.range).map_err(serde::de::Error::custom)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn latency_encoder_emits_one_positive_spike_per_channel() {
        let mut encoder = LatencyEncoder::new(10, (0.0, 1.0));

        let output = encoder.encode(&[0.0, 0.5, 1.0]);

        assert_eq!(output.spikes.len(), 3);
        assert_eq!(
            output.spikes,
            vec![
                SpikeEvent::new(0, 10u64, true),
                SpikeEvent::new(1, 5u64, true),
                SpikeEvent::new(2, 0u64, true),
            ]
        );
    }

    #[test]
    fn latency_encoder_time_model_matches_window() {
        let encoder = LatencyEncoder::new(10, (0.0, 1.0));
        let model = encoder.time_model();

        assert_eq!(model.span_ticks(), 11);
        assert_eq!(model.step_ticks(), 11);
        assert!(!model.is_overlapping());
        // Latency ticks are a budget, not a duration.
        assert!(model.timebase().is_none());

        // max_latency == 0 still yields a one-tick window.
        assert_eq!(
            LatencyEncoder::new(0, (0.0, 1.0)).time_model().span_ticks(),
            1
        );
    }

    #[test]
    fn latency_encoder_rejects_an_unrepresentable_window() {
        // max_latency + 1 must fit in u64, or the declared span could not
        // contain a spike emitted at max_latency itself.
        assert_eq!(
            LatencyEncoder::try_new(u64::MAX, (0.0, 1.0)),
            Err(EncoderError::WindowTooLarge {
                parameter: "max_latency"
            })
        );

        // One below the cap is accepted, and its own span still contains it.
        let mut encoder = LatencyEncoder::new(u64::MAX - 1, (0.0, 1.0));
        let model = encoder.time_model();
        // NaN maps to the latest possible spike — the tightest case for the bound.
        let latest = encoder.encode(&[f32::NAN]).spikes[0].timestamp;
        assert_eq!(latest, u64::MAX - 1);
        assert!(model.contains(latest), "latest spike escapes the span");
    }

    #[test]
    fn latency_encoder_offsets_stay_inside_the_window_under_gain() {
        let mut encoder = LatencyEncoder::new(10, (0.0, 1.0));
        let span = encoder.time_model().span_ticks();

        // A latency gain above 1.0 stretches nothing: max_latency is a hard cap.
        let stretched = encoder.encode_with_gains(
            &[0.0, 0.5, 1.0, f32::NAN],
            EncodingGains {
                latency_scale: 4.0,
                ..EncodingGains::identity()
            },
        );
        assert!(stretched.spikes.iter().all(|spike| spike.timestamp < span));
        assert_eq!(stretched.spikes[0].timestamp, 10);
        assert_eq!(stretched.spikes[3].timestamp, 10);

        // A gain below 1.0 compresses the window as before.
        let compressed = encoder.encode_with_gains(
            &[0.0],
            EncodingGains {
                latency_scale: 0.5,
                ..EncodingGains::identity()
            },
        );
        assert_eq!(compressed.spikes[0].timestamp, 5);
    }

    #[test]
    fn latency_encoder_offsets_are_call_relative() {
        let mut encoder = LatencyEncoder::new(10, (0.0, 1.0));
        let mut cursor = TimeCursor::new(encoder.time_model());

        let first = encoder.encode(&[0.0]);
        cursor.advance();
        let second = encoder.encode(&[0.0]);

        // Identical offsets; the caller's cursor is what separates the calls.
        assert_eq!(first.spikes[0].timestamp, second.spikes[0].timestamp);
        assert_eq!(cursor.absolute(second.spikes[0].timestamp), 21);
    }

    #[test]
    fn test_latency_encoder_nan() {
        let mut encoder = LatencyEncoder::new(10, (0.0, 1.0));
        let output = encoder.encode(&[f32::NAN]);
        assert_eq!(output.spikes[0].timestamp, 10);
    }

    #[test]
    fn test_latency_encoder_reset() {
        let mut encoder = LatencyEncoder::new(10, (0.0, 1.0));
        encoder.reset(); // Should do nothing
    }

    #[test]
    fn latency_encoder_stronger_inputs_fire_earlier() {
        let mut encoder = LatencyEncoder::new(12, (0.0, 3.0));

        let output = encoder.encode(&[0.5, 1.5, 2.5]);

        assert_eq!(output.spikes.len(), 3);
        assert!(output.spikes[0].timestamp > output.spikes[1].timestamp);
        assert!(output.spikes[1].timestamp > output.spikes[2].timestamp);
    }

    #[test]
    fn latency_encoder_clamps_inputs_to_range() {
        let mut encoder = LatencyEncoder::new(8, (2.0, 6.0));

        let output = encoder.encode(&[0.0, 2.0, 4.0, 6.0, 9.0]);

        assert_eq!(
            output
                .spikes
                .iter()
                .map(|spike| spike.timestamp)
                .collect::<Vec<_>>(),
            vec![8, 8, 4, 0, 0]
        );
    }

    #[test]
    fn latency_encoder_encode_step_matches_encode() {
        let mut encoder = LatencyEncoder::new(20, (-1.0, 1.0));
        let input = [-1.0, -0.25, 0.75, 1.5];

        let batch = encoder.encode(&input);
        let step = encoder.encode_step(&input);

        assert_eq!(batch, step);
    }

    #[test]
    fn latency_encoder_handles_empty_input() {
        let mut encoder = LatencyEncoder::new(5, (0.0, 1.0));

        let output = encoder.encode(&[]);

        assert!(output.spikes.is_empty());
    }

    #[test]
    fn latency_encoder_nan_maps_to_max_latency() {
        let mut encoder = LatencyEncoder::new(7, (0.0, 1.0));

        let output = encoder.encode(&[f32::NAN, 1.0]);

        assert_eq!(output.spikes[0].timestamp, 7);
        assert_eq!(output.spikes[1].timestamp, 0);
    }

    #[test]
    #[should_panic(expected = "range must be finite and min must be less than max")]
    fn latency_encoder_rejects_invalid_range() {
        let _ = LatencyEncoder::new(5, (1.0, 1.0));
    }

    #[test]
    #[should_panic(expected = "range must be finite and min must be less than max")]
    fn latency_encoder_rejects_infinite_range() {
        let _ = LatencyEncoder::new(10, (f32::NEG_INFINITY, f32::INFINITY));
    }

    #[test]
    fn latency_encoder_truncates_channel_overflow() {
        let mut encoder = LatencyEncoder::new(1, (0.0, 1.0));
        let input = vec![0.0f32; (u16::MAX as usize) + 2];
        let output = encoder.encode(&input);
        assert_eq!(output.spikes.len(), u16::MAX as usize + 1);
    }

    #[test]
    fn latency_encoder_encode_with_modulators_identity() {
        let mut encoder = LatencyEncoder::new(10, (0.0, 1.0));
        let curves = NeuromodulatorGainCurves::default();
        let mods = NeuroModulators::default();

        let plain = encoder.encode(&[0.5]);
        let modulated = encoder.encode_with_modulators(&[0.5], &mods, &curves);

        assert_eq!(plain.spikes[0].timestamp, modulated.spikes[0].timestamp);
    }

    #[test]
    fn latency_encoder_encode_with_modulators_latency_scale() {
        let mut encoder = LatencyEncoder::new(10, (0.0, 1.0));
        let curves = NeuromodulatorGainCurves {
            dopamine: ModulatorGainCurves {
                latency: Some(GainCurve::new((0.0, 1.0), (0.5, 0.5))),
                ..Default::default()
            },
            ..Default::default()
        };
        let mods = NeuroModulators {
            dopamine: 1.0,
            ..Default::default()
        };

        let output = encoder.encode_with_modulators(&[0.5], &mods, &curves);
        // latency_scale = 0.5, so max_latency = 10 * 0.5 = 5
        // normalized(0.5) = 0.5, timestamp = (1.0 - 0.5) * 5 = 2.5 → 3
        assert_eq!(output.spikes[0].timestamp, 3);
    }

    #[test]
    fn latency_encoder_encode_step_with_modulators_matches_encode() {
        let mut encoder = LatencyEncoder::new(10, (0.0, 1.0));
        let curves = NeuromodulatorGainCurves::default();
        let mods = NeuroModulators::default();

        let batch = encoder.encode_with_modulators(&[0.5], &mods, &curves);
        let step = encoder.encode_step_with_modulators(&[0.5], &mods, &curves);

        assert_eq!(batch, step);
    }

    #[test]
    fn latency_encoder_modulators_zero_scale_maps_to_zero() {
        let mut encoder = LatencyEncoder::new(10, (0.0, 1.0));
        let curves = NeuromodulatorGainCurves {
            dopamine: ModulatorGainCurves {
                latency: Some(GainCurve::new((0.0, 1.0), (1.0, 0.0))),
                ..Default::default()
            },
            ..Default::default()
        };
        let mods = NeuroModulators {
            dopamine: 1.0,
            ..Default::default()
        };

        let output = encoder.encode_with_modulators(&[0.5, f32::NAN], &mods, &curves);
        assert_eq!(output.spikes.len(), 2);
        assert!(output.spikes.iter().all(|s| s.timestamp == 0));
    }
    #[test]
    fn latency_encoder_supports_zero_max_latency() {
        let mut encoder = LatencyEncoder::new(0, (0.0, 1.0));
        let output = encoder.encode(&[0.0, 0.5, 1.0]);
        assert_eq!(output.spikes.len(), 3);
        assert!(output.spikes.iter().all(|s| s.timestamp == 0));
    }

    #[test]
    fn latency_encoder_try_new_rejects_invalid_configuration() {
        assert!(LatencyEncoder::try_new(0, (0.0, 1.0)).is_ok());
        assert_eq!(
            LatencyEncoder::try_new(1, (1.0, 1.0)).err(),
            Some(EncoderError::InvalidRange { parameter: "range" })
        );
    }
}
