#[cfg(feature = "serde")]
#[test]
fn test_serde_serialization_deserialization() {
    use axon_encoder::prelude::*;

    // 1. Test SpikeEvent
    let spike = SpikeEvent {
        channel: 12,
        timestamp: 42,
        polarity: true,
    };
    let serialized_spike = serde_json::to_string(&spike).unwrap();
    let deserialized_spike: SpikeEvent = serde_json::from_str(&serialized_spike).unwrap();
    assert_eq!(spike.channel, deserialized_spike.channel);
    assert_eq!(spike.timestamp, deserialized_spike.timestamp);
    assert_eq!(spike.polarity, deserialized_spike.polarity);

    // 2. Test EncoderConfig
    let config = EncoderConfig {
        input_channels: 10,
        output_channels: 20,
    };
    let serialized_config = serde_json::to_string(&config).unwrap();
    let deserialized_config: EncoderConfig = serde_json::from_str(&serialized_config).unwrap();
    assert_eq!(config.input_channels, deserialized_config.input_channels);
    assert_eq!(config.output_channels, deserialized_config.output_channels);

    // 3. Test EncodedOutput
    let mut output = EncodedOutput::new();
    output.spikes.push(spike);
    output.embeddings = Some(vec![1.0, 2.0, 3.0]);
    output.metadata = Some(EncodingMetadata::default());

    let serialized_output = serde_json::to_string(&output).unwrap();
    let deserialized_output: EncodedOutput = serde_json::from_str(&serialized_output).unwrap();
    assert_eq!(output.spikes.len(), deserialized_output.spikes.len());
    assert_eq!(
        output.spikes[0].channel,
        deserialized_output.spikes[0].channel
    );
    assert_eq!(output.embeddings, deserialized_output.embeddings);
}
