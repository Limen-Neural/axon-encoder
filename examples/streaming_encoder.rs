//! Buffer streaming calls and convert each batch's offsets to absolute ticks.
use axon_encoder::prelude::*;

fn main() -> Result<(), EncoderError> {
    let mut encoder = LatencyEncoder::try_new(9, (0.0, 1.0))?;
    let dynamic: &mut dyn Encoder = &mut encoder;
    let mut stream = StreamingEncoder::new(dynamic, 3, FlushPolicy::OnCapacity)?;
    let mut spikes: Vec<SpikeEvent> = Vec::new();
    let mut absolute_ticks = Vec::new();
    let mut sink = |batch: SpikeBatch<'_>| {
        for spike in batch.spikes() {
            absolute_ticks.push(batch.origin().saturating_add(spike.timestamp.ticks()));
        }
        spikes.extend_from_slice(batch.spikes());
    };
    for input in [&[1.0, 0.0][..], &[0.5, 0.2][..]] {
        stream
            .encode_step(input, &mut sink)
            .expect("capacity policy does not block");
    }
    stream.flush_into(&mut sink);
    println!(
        "{} spikes at absolute ticks {absolute_ticks:?}",
        spikes.len()
    );
    Ok(())
}
