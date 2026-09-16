use axon_encoder::prelude::{EmbeddingEncoderConfig, EmbeddingRateEncoder, EncodedOutput, Encoder};

#[test]
fn encoded_output_defaults_are_stable() {
    let output: EncodedOutput = EncodedOutput::new();

    assert!(output.spikes.is_empty());
    assert!(output.embeddings.is_none());
}

#[test]
fn encoded_output_supports_optional_dense_embedding() {
    let mut output = EncodedOutput::new();
    output.embeddings = Some(vec![0.1, 0.2, 0.3]);

    assert_eq!(
        output.embeddings.as_deref(),
        Some([0.1, 0.2, 0.3].as_slice())
    );
}

#[test]
fn embedding_rate_encoder_produces_standardized_output() {
    let mut encoder = EmbeddingRateEncoder::new(3, EmbeddingEncoderConfig { v_th: 0.4 });
    let output = encoder.encode(&[0.5, 0.1, 0.5]);

    assert!(!output.spikes.is_empty());
    assert!(output.embeddings.is_none());
}
