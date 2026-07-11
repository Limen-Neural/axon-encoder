use crate::types::{EncodedOutput, SpikeEvent};

#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
#[cfg_attr(feature = "serde", serde(try_from = "EmbeddingEncoderConfigRepr"))]
pub struct EmbeddingEncoderConfig {
    pub v_th: f32,
}

#[cfg(feature = "serde")]
#[derive(serde::Deserialize)]
struct EmbeddingEncoderConfigRepr {
    v_th: f32,
}

#[cfg(feature = "serde")]
impl TryFrom<EmbeddingEncoderConfigRepr> for EmbeddingEncoderConfig {
    type Error = String;

    fn try_from(r: EmbeddingEncoderConfigRepr) -> Result<Self, String> {
        if !(r.v_th > 0.0) {
            return Err("v_th must be positive".into());
        }
        Ok(Self { v_th: r.v_th })
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
#[cfg_attr(feature = "serde", serde(try_from = "EmbeddingRateEncoderRepr"))]
pub struct EmbeddingRateEncoder {
    pub config: EmbeddingEncoderConfig,
    pub normalized_embeddings: Vec<f32>,
}

#[cfg(feature = "serde")]
#[derive(serde::Deserialize)]
struct EmbeddingRateEncoderRepr {
    config: EmbeddingEncoderConfig,
    normalized_embeddings: Vec<f32>,
}

#[cfg(feature = "serde")]
impl TryFrom<EmbeddingRateEncoderRepr> for EmbeddingRateEncoder {
    type Error = String;

    fn try_from(r: EmbeddingRateEncoderRepr) -> Result<Self, String> {
        if r.normalized_embeddings.iter().any(|v| !v.is_finite()) {
            return Err("normalized_embeddings must be finite".into());
        }
        if r.normalized_embeddings.len() > u16::MAX as usize + 1 {
            return Err("too many channels (max 65536)".into());
        }
        Ok(Self {
            config: r.config,
            normalized_embeddings: r.normalized_embeddings,
        })
    }
}

impl EmbeddingRateEncoder {
    pub fn new(embeddings: &[f32], config: EmbeddingEncoderConfig) -> Self {
        assert!(config.v_th > 0.0, "v_th must be positive");
        assert!(
            embeddings.len() <= u16::MAX as usize + 1,
            "too many channels (max 65536)"
        );

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
