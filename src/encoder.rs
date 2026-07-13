use crate::types::{EncodedOutput, SpikeEvent};

#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct EmbeddingEncoderConfig {
    pub v_th: f32,
}

#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct EncoderState {
    pub membrane_potentials: Vec<f32>,
}

impl EncoderState {
    pub fn new_zeros(len: usize) -> Self {
        Self {
            membrane_potentials: vec![0.0; len],
        }
    }
}

pub struct EmbeddingRateEncoder {
    pub config: EmbeddingEncoderConfig,
    pub normalized_embeddings: Vec<f32>,
}

impl EmbeddingRateEncoder {
    pub fn new(embeddings: &[f32], config: EmbeddingEncoderConfig) -> Self {
        let min_val = embeddings.iter().copied().fold(f32::INFINITY, f32::min);
        let max_val = embeddings.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let range = max_val - min_val;
        let epsilon = 1e-5f32;
        let safe_range = range + epsilon;

        let normalized: Vec<f32> = embeddings
            .iter()
            .map(|&x| (x - min_val) / safe_range)
            .collect();

        Self {
            config,
            normalized_embeddings: normalized,
        }
    }

    pub fn forward(&self, prev_state: &EncoderState) -> (EncodedOutput, EncoderState) {
        let mut new_potentials = prev_state.membrane_potentials.clone();
        let mut output = EncodedOutput::new();

        for (i, (pot, &emb)) in new_potentials
            .iter_mut()
            .zip(self.normalized_embeddings.iter())
            .enumerate()
        {
            *pot += emb;

            if *pot >= self.config.v_th {
                output.spikes.push(SpikeEvent {
                    channel: i as u16,
                    timestamp: 0,
                    polarity: true,
                });
                *pot -= self.config.v_th; // Soft reset
            }
        }

        (
            output,
            EncoderState {
                membrane_potentials: new_potentials,
            },
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_embedding_rate_encoder_init() {
        let embeddings = vec![1.0, 2.0, 3.0, 5.0];
        let config = EmbeddingEncoderConfig { v_th: 1.0 };
        let encoder = EmbeddingRateEncoder::new(&embeddings, config);

        // Min: 1.0, Max: 5.0, Range: 4.0. Epsilon: 1e-5.
        // Safe Range: 4.00001
        // Expected normalized: (x - 1.0) / 4.00001
        assert_eq!(encoder.normalized_embeddings.len(), 4);
        assert!((encoder.normalized_embeddings[0] - 0.0).abs() < 1e-5);
        assert!((encoder.normalized_embeddings[3] - (4.0 / 4.00001)).abs() < 1e-5);
    }

    #[test]
    fn test_embedding_rate_encoder_forward_no_spikes() {
        let embeddings = vec![1.0, 2.0, 3.0];
        let config = EmbeddingEncoderConfig { v_th: 10.0 }; // High threshold
        let encoder = EmbeddingRateEncoder::new(&embeddings, config);

        let state = EncoderState::new_zeros(3);
        let (output, next_state) = encoder.forward(&state);

        assert!(output.spikes.is_empty());
        assert_eq!(next_state.membrane_potentials.len(), 3);
        for i in 0..3 {
            assert_eq!(
                next_state.membrane_potentials[i],
                encoder.normalized_embeddings[i]
            );
        }
    }

    #[test]
    fn test_embedding_rate_encoder_forward_with_spikes_and_soft_reset() {
        let embeddings = vec![1.0, 2.0, 3.0];
        // v_th is 0.4.
        // Normalized embeddings:
        // Min: 1.0, Max: 3.0, Range: 2.0. Safe Range: 2.00001.
        // Norm: [0.0, 1.0 / 2.00001 (~0.5), 2.0 / 2.00001 (~1.0)]
        let config = EmbeddingEncoderConfig { v_th: 0.4 };
        let encoder = EmbeddingRateEncoder::new(&embeddings, config);

        let state = EncoderState::new_zeros(3);
        let (output, next_state) = encoder.forward(&state);

        // Expected spike channels:
        // Channel 0: potential = 0.0 < 0.4 -> No spike. Potential remaining: 0.0
        // Channel 1: potential = ~0.5 >= 0.4 -> Spike! Soft reset: ~0.5 - 0.4 = ~0.1
        // Channel 2: potential = ~1.0 >= 0.4 -> Spike! Soft reset: ~1.0 - 0.4 = ~0.6
        assert_eq!(output.spikes.len(), 2);

        // Spikes should be in channel order
        assert_eq!(output.spikes[0].channel, 1);
        assert_eq!(output.spikes[0].timestamp, 0);
        assert!(output.spikes[0].polarity);

        assert_eq!(output.spikes[1].channel, 2);
        assert_eq!(output.spikes[1].timestamp, 0);
        assert!(output.spikes[1].polarity);

        assert!((next_state.membrane_potentials[0] - 0.0).abs() < 1e-5);
        assert!(
            (next_state.membrane_potentials[1] - (encoder.normalized_embeddings[1] - 0.4)).abs()
                < 1e-5
        );
        assert!(
            (next_state.membrane_potentials[2] - (encoder.normalized_embeddings[2] - 0.4)).abs()
                < 1e-5
        );
    }

    #[test]
    fn test_embedding_rate_encoder_multi_step_evolution() {
        let embeddings = vec![1.0, 3.0]; // Norm: [0.0, ~1.0]
        let config = EmbeddingEncoderConfig { v_th: 0.6 };
        let encoder = EmbeddingRateEncoder::new(&embeddings, config);

        let mut state = EncoderState::new_zeros(2);

        // Step 1:
        // Channel 0: potential = 0.0. No spike.
        // Channel 1: potential = ~1.0. Spike! Soft reset to ~0.4.
        let (output1, state1) = encoder.forward(&state);
        assert_eq!(output1.spikes.len(), 1);
        assert_eq!(output1.spikes[0].channel, 1);
        state = state1;

        // Step 2:
        // Channel 0: potential = 0.0 + 0.0 = 0.0. No spike.
        // Channel 1: potential = ~0.4 + ~1.0 = ~1.4. Spike! Soft reset to ~0.8.
        let (output2, state2) = encoder.forward(&state);
        assert_eq!(output2.spikes.len(), 1);
        assert_eq!(output2.spikes[0].channel, 1);
        state = state2;

        // Step 3:
        // Channel 0: potential = 0.0. No spike.
        // Channel 1: potential = ~0.8 + ~1.0 = ~1.8. Spike! Soft reset to ~1.2.
        let (output3, state3) = encoder.forward(&state);
        assert_eq!(output3.spikes.len(), 1);
        assert_eq!(output3.spikes[0].channel, 1);
        state = state3;

        assert!((state.membrane_potentials[0] - 0.0).abs() < 1e-5);
        assert!(
            (state.membrane_potentials[1] - (3.0 * encoder.normalized_embeddings[1] - 3.0 * 0.6))
                .abs()
                < 1e-5
        );
    }

    #[test]
    fn test_embedding_rate_encoder_edge_cases() {
        // Equal embeddings
        let embeddings = vec![2.5, 2.5, 2.5];
        let config = EmbeddingEncoderConfig { v_th: 0.5 };
        let encoder = EmbeddingRateEncoder::new(&embeddings, config);

        // Since range is 0.0, safe range is 1e-5.
        // All normalized values are (2.5 - 2.5) / 1e-5 = 0.0.
        for &val in &encoder.normalized_embeddings {
            assert_eq!(val, 0.0);
        }

        let state = EncoderState::new_zeros(3);
        let (output, next_state) = encoder.forward(&state);
        assert!(output.spikes.is_empty());
        assert_eq!(next_state.membrane_potentials, vec![0.0, 0.0, 0.0]);
    }
}
