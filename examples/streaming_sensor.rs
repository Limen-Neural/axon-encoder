//! Streaming Sensor Example
//!
//! Encodes one three-axis sensor frame per call.  `SpikeEvent` timestamps are
//! offsets from the current call, so the application owns absolute time with a
//! `TimeCursor` and advances it once after each frame.
//!
//! ```
//! cargo run --example streaming_sensor
//! ```

use axon_encoder::prelude::*;

fn main() -> Result<(), EncoderError> {
    // 100 Hz sampling: each streaming call represents a 10 ms tick.
    let mut encoder = RateEncoder::try_new(0.0, 200.0, (-1.0, 1.0), 0.010)?;
    let mut cursor = TimeCursor::new(encoder.time_model());
    let mut spikes = Vec::<SpikeEvent>::new();

    // Each element is one accelerometer frame, not a batch window.
    let frames = [
        [0.0, 0.2, -0.1],
        [0.4, 0.5, -0.2],
        [0.8, 0.1, 0.0],
        [0.2, -0.4, 0.3],
    ];

    for (frame_number, frame) in frames.iter().enumerate() {
        spikes.clear(); // retain capacity across sensor callbacks
        encoder.encode_step_into(frame, &mut spikes);

        for spike in &spikes {
            // `timestamp` is a TickOffset, never an absolute timestamp.
            let absolute_tick = cursor.absolute(spike.timestamp);
            let absolute_nanos = cursor
                .absolute_nanos(spike.timestamp)
                .expect("RateEncoder provides a physical timebase");
            println!(
                "frame {frame_number}: channel {} at tick {absolute_tick} ({absolute_nanos} ns)",
                spike.channel
            );
        }

        cursor.advance(); // exactly once for the call just encoded
    }

    Ok(())
}
