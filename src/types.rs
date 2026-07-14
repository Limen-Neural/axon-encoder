//! Standardized types for encoder inputs and outputs.

/// A single spike event.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SpikeEvent {
    pub channel: u16,
    pub timestamp: u64, // or relative step
    pub polarity: bool, // or strength
}

/// Optional metadata about the encoding process.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct EncodingMetadata {
    // Add any relevant metadata fields here, e.g.:
    // pub source_sample_index: u64,
}

/// The standardized output of an encoder.
#[derive(Clone, Debug, Default, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct EncodedOutput {
    pub spikes: Vec<SpikeEvent>,
    pub embeddings: Option<Vec<f32>>,
    pub metadata: Option<EncodingMetadata>,
}

impl EncodedOutput {
    pub fn new() -> Self {
        Self::default()
    }
}

/// General-purpose configuration for encoders.
#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct EncoderConfig {
    pub input_channels: usize,
    pub output_channels: usize,
}

impl Default for EncoderConfig {
    fn default() -> Self {
        Self {
            input_channels: 256,
            output_channels: 256,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encoded_output_new() {
        let output = EncodedOutput::new();
        assert!(output.spikes.is_empty());
        assert!(output.embeddings.is_none());
        assert!(output.metadata.is_none());
    }

    #[test]
    fn test_encoder_config_default() {
        let config = EncoderConfig::default();
        assert_eq!(config.input_channels, 256);
        assert_eq!(config.output_channels, 256);
    }
}
