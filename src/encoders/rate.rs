use crate::prelude::*;

/// Encodes analog values as spike rates based on input intensity.
///
/// Each input channel is mapped to a firing rate between `base_rate` and `max_rate`.
/// In batch mode (`encode`), each call generates independent probabilistic spikes.
/// In streaming mode (`encode_step`), accumulates probability and fires deterministically
/// when the accumulated value exceeds a threshold per channel.
///
/// # Mathematical Model
///
/// For batch encoding:
/// ```text
/// rate = base_rate + normalized * (max_rate - base_rate)
/// probability = rate / 10.0
/// spike if random() < probability
/// ```
///
/// For streaming (`encode_step`):
/// ```text
/// accumulator[i] += normalized[i] * (max_rate - base_rate) / 10.0
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
/// - `base_rate`: Minimum firing rate (Hz equivalent) when input is at range minimum
/// - `max_rate`: Maximum firing rate (Hz equivalent) when input is at range maximum
/// - `range`: Tuple of (min, max) input values
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct RateEncoder {
    base_rate: f32,
    max_rate: f32,
    range: (f32, f32),
    accumulators: Vec<f32>,
}

impl RateEncoder {
    pub fn new(base_rate: f32, max_rate: f32, range: (f32, f32)) -> Self {
        Self {
            base_rate,
            max_rate,
            range,
            accumulators: Vec::new(),
        }
    }

    fn normalize(&self, value: f32) -> f32 {
        ((value - self.range.0) / (self.range.1 - self.range.0)).clamp(0.0, 1.0)
    }

    fn ensure_accumulators(&mut self, num_channels: usize) {
        if self.accumulators.len() < num_channels {
            self.accumulators.resize(num_channels, 0.0);
        }
    }

    fn encode_with_rate_scale(&mut self, input: &[f32], rate_scale: f32) -> EncodedOutput {
        let mut output = EncodedOutput::new();
        if input.is_empty() {
            return output;
        }

        for (i, &value) in input.iter().enumerate() {
            let Ok(channel) = u16::try_from(i) else {
                // Remaining channels exceed u16::MAX; stop rather than wrap.
                break;
            };
            let normalized = self.normalize(value);
            let rate =
                (self.base_rate + normalized * (self.max_rate - self.base_rate)) * rate_scale;
            let probability = (rate / 10.0).clamp(0.0, 1.0);

            if crate::rng::gen_unit_f32() < probability {
                output.spikes.push(SpikeEvent {
                    channel,
                    timestamp: 0,
                    polarity: true,
                });
            }
        }

        output
    }

    fn encode_step_with_rate_scale(&mut self, input: &[f32], rate_scale: f32) -> EncodedOutput {
        let mut output = EncodedOutput::new();
        if input.is_empty() {
            return output;
        }

        self.ensure_accumulators(input.len());

        for (i, &value) in input.iter().enumerate() {
            let Ok(channel) = u16::try_from(i) else {
                // Remaining channels exceed u16::MAX; stop rather than wrap.
                break;
            };
            let normalized = self.normalize(value);
            let rate_increment = ((self.base_rate + normalized * (self.max_rate - self.base_rate))
                * rate_scale)
                / 10.0;
            self.accumulators[i] += rate_increment.max(0.0);

            while self.accumulators[i] >= 1.0 {
                output.spikes.push(SpikeEvent {
                    channel,
                    timestamp: 0,
                    polarity: true,
                });
                self.accumulators[i] -= 1.0;
            }
        }

        output
    }

    /// Encode input using neuromodulator-driven gain curves.
    ///
    /// Evaluates `gain_curves` against the current `modulators` to produce
    /// an [`EncodingGains`], then uses the `firing_rate_scale` component to
    /// modulate the base firing rate. Values > 1.0 increase spike frequency;
    /// values < 1.0 decrease it.
    ///
    /// Expected modulator range: any finite f32. Expected gain range after
    /// sanitization: `[0.0, 10,000.0]`.
    pub fn encode_with_modulators(
        &mut self,
        input: &[f32],
        modulators: &NeuroModulators,
        gain_curves: &NeuromodulatorGainCurves,
    ) -> EncodedOutput {
        let gains = gain_curves.evaluate(modulators);
        self.encode_with_rate_scale(input, gains.firing_rate_scale)
    }

    /// Step-wise variant of [`encode_with_modulators`](Self::encode_with_modulators).
    /// Uses [`encode_step_with_rate_scale`](Self::encode_step_with_rate_scale)
    /// for accumulator-based spike generation.
    pub fn encode_step_with_modulators(
        &mut self,
        input: &[f32],
        modulators: &NeuroModulators,
        gain_curves: &NeuromodulatorGainCurves,
    ) -> EncodedOutput {
        let gains = gain_curves.evaluate(modulators);
        self.encode_step_with_rate_scale(input, gains.firing_rate_scale)
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
        for acc in self.accumulators.iter_mut() {
            *acc = 0.0;
        }
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
}
