use crate::types::{EncodedOutput, SpikeEvent};

#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
pub struct EmbeddingEncoderConfig {
    pub v_th: f32,
}

#[cfg(feature = "serde")]
impl<'de> serde::Deserialize<'de> for EmbeddingEncoderConfig {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(serde::Deserialize)]
        struct Helper {
            v_th: f32,
        }

        let helper = Helper::deserialize(deserializer)?;
        if !matches!(
            helper.v_th.partial_cmp(&0.0),
            Some(std::cmp::Ordering::Greater)
        ) {
            return Err(serde::de::Error::custom(
                "EmbeddingEncoderConfig.v_th must be > 0",
            ));
        }
        Ok(Self { v_th: helper.v_th })
    }
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

#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
pub struct EmbeddingRateEncoder {
    pub config: EmbeddingEncoderConfig,
    pub normalized_embeddings: Vec<f32>,
}

#[cfg(feature = "serde")]
impl<'de> serde::Deserialize<'de> for EmbeddingRateEncoder {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(serde::Deserialize)]
        struct Helper {
            config: EmbeddingEncoderConfig,
            normalized_embeddings: Vec<f32>,
        }

        // EmbeddingEncoderConfig's Deserialize already enforces v_th > 0.
        let helper = Helper::deserialize(deserializer)?;
        Ok(Self {
            config: helper.config,
            normalized_embeddings: helper.normalized_embeddings,
        })
    }
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
                    channel: u16::try_from(i).expect("channel index exceeds u16::MAX"),
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
