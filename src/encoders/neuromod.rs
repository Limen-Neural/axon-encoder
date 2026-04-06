use crate::prelude::*;
use crate::modulators::NeuroModulators;
use rand::Rng;

pub struct NeuromodSensoryEncoder {
    neuromodulators: NeuroModulators,
    channel_gains: Vec<f32>,
    channel_biases: Vec<f32>,
    adaptation_rates: Vec<f32>,
    input_channels: usize,
    output_channels: usize,
}

impl NeuromodSensoryEncoder {
    pub fn new(input_channels: usize, output_channels: usize) -> Self {
        assert!(output_channels % input_channels == 0, "Output channels must be a multiple of input channels.");
        Self {
            neuromodulators: NeuroModulators::default(),
            channel_gains: vec![1.0; output_channels],
            channel_biases: vec![0.0; output_channels],
            adaptation_rates: vec![0.01; output_channels],
            input_channels,
            output_channels,
        }
    }

    pub fn update_neuromodulators(&mut self, modulators: NeuroModulators) {
        self.neuromodulators = modulators;
    }

    fn poisson_encode_channel(&self, value: f32, channel_idx: usize) -> (f32, f32) {
        let gain = self.channel_gains[channel_idx];
        let bias = self.channel_biases[channel_idx];
        let scaled_value = (value * gain + bias).clamp(-3.0, 3.0);

        if scaled_value >= 0.0 {
            let bull_rate = (scaled_value / 3.0).clamp(0.0, 1.0);
            (0.01, bull_rate)
        } else {
            let bear_rate = (-scaled_value / 3.0).clamp(0.0, 1.0);
            (bear_rate, 0.01)
        }
    }

    fn update_adaptation(&mut self, stimuli: &[f32]) {
        for i in 0..self.output_channels {
            let rate = self.adaptation_rates[i];
            let activity = stimuli[i];
            let target_rate = 0.1;
            let error = activity - target_rate;

            self.channel_gains[i] -= rate * error * 0.1;
            self.channel_gains[i] = self.channel_gains[i].clamp(0.1, 3.0);

            if activity < 0.05 {
                self.channel_biases[i] += rate * 0.01;
            } else if activity > 0.5 {
                self.channel_biases[i] -= rate * 0.01;
            }
            self.channel_biases[i] = self.channel_biases[i].clamp(-0.5, 0.5);
        }
    }
}

impl Encoder for NeuromodSensoryEncoder {
    fn encode(&mut self, input: &[f32]) -> EncodedOutput {
        let mut output = EncodedOutput::new();
        let mut stimuli = vec![0.0f32; self.output_channels];
        let channels_per_input = self.output_channels / self.input_channels;

        for (i, &value) in input.iter().enumerate() {
            for j in 0..channels_per_input {
                let output_channel = i * channels_per_input + j;
                let (bear_rate, bull_rate) = self.poisson_encode_channel(value, output_channel);
                
                if rand::thread_rng().gen_range(0.0f32..1.0) < bear_rate {
                    output.spikes.push(SpikeEvent {
                        channel: output_channel as u16,
                        timestamp: 0, // Simplified
                        polarity: false, // Bearish
                    });
                }
                if rand::thread_rng().gen_range(0.0f32..1.0) < bull_rate {
                    output.spikes.push(SpikeEvent {
                        channel: output_channel as u16,
                        timestamp: 0, // Simplified
                        polarity: true, // Bullish
                    });
                }
                stimuli[output_channel] = bear_rate + bull_rate; // For adaptation
            }
        }

        self.update_adaptation(&stimuli);
        output
    }

    fn reset(&mut self) {
        self.neuromodulators = NeuroModulators::default();
        self.channel_gains = vec![1.0; self.output_channels];
        self.channel_biases = vec![0.0; self.output_channels];
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_neuromod_sensory_encoder() {
        let mut encoder = NeuromodSensoryEncoder::new(2, 4);
        let input = [1.0, -1.0];
        let output = encoder.encode(&input);
        // Probabilistic, but we should get some spikes
        assert!(output.spikes.len() > 0);
    }
}
