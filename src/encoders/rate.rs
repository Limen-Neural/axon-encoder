use crate::prelude::*;

/// Encodes analog values as spike rates based on input intensity
///
/// Each input channel is mapped to a firing rate between `base_rate` and `max_rate`
/// In batch mode (`encode`), each call generates independent probabilistic spikes
/// In streaming mode (`encode_step`), accumulates probability and fires deterministically
/// when the accumulated value exceeds a threshold per channel
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
            let probability = (rate / 10.0).clamp(0.0, 1.0);

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

    fn encode_step_with_rate_scale(&mut self, input: &[f32], rate_scale: f32) -> EncodedOutput {
        let mut output = EncodedOutput::new();
        if input.is_empty() {
            return output;
        }
        // Non-finite / non-positive scales must not poison accumulators with NaN.
        if !rate_scale.is_finite() || rate_scale <= 0.0 {
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

        let boosted = encoder.encode_with_modulators(&[1.0], &modulators, &gain_curves);
        assert_eq!(boosted.spikes.len(), 1);
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
}
