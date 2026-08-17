//! Spike Timebase Example
//!
//! Shows the crate's spike time contract end to end: encoders emit
//! **call-relative** tick offsets, and the caller owns absolute time via a
//! `TimeCursor`. This is the integration path for a downstream simulator or a
//! hardware adapter that needs one merged, absolutely-timed event stream.
//!
//! The key step for merging two encoders is putting them in the same unit.
//! `RateEncoder` is configured in seconds, so it reports its own `Timebase`;
//! `LatencyEncoder` counts dimensionless ticks, so the caller declares what one
//! of its ticks lasts before the two streams can share a timeline.
//!
//! ```
//! cargo run --example spike_timebase
//! ```

use axon_encoder::prelude::*;

/// One absolutely-timed event, as a downstream consumer would store it.
#[derive(Debug)]
struct TimedSpike {
    source: &'static str,
    channel: u16,
    nanos: u64,
}

fn main() {
    // One call is an 8-tick presentation window of dimensionless ticks.
    let mut latency = LatencyEncoder::try_new(7, (0.0, 1.0)).expect("valid LatencyEncoder");
    // One call is one 8 ms tick — configured in physical time. Matching the two
    // encoders' per-call duration (8 ms either way, once the latency tick is
    // declared as 1 ms below) is what makes the merged stream meaningful: spikes
    // from the same sample interleave instead of drifting apart.
    let mut rate = RateEncoder::try_new(50.0, 900.0, (0.0, 1.0), 0.008).expect("valid RateEncoder");

    println!("=== Spike Timebase ===");
    for (name, model) in [
        ("latency", latency.time_model()),
        ("rate", rate.time_model()),
    ] {
        println!(
            "{name:>8}: step {} tick(s), span {} tick(s), timebase {}",
            model.step_ticks(),
            model.span_ticks(),
            model
                .timebase()
                .map_or_else(|| "none (caller decides)".to_string(), |tb| tb.to_string()),
        );
    }

    // The rate encoder knows its tick duration; the latency encoder does not, so
    // the caller supplies one. Both cursors now speak nanoseconds.
    let mut latency_cursor =
        TimeCursor::new(latency.time_model().with_timebase(Timebase::MILLISECOND));
    let mut rate_cursor = TimeCursor::new(rate.time_model());
    println!(
        "\nlatency ticks are dimensionless — declaring {} for this pipeline\n",
        Timebase::MILLISECOND
    );

    let mut timeline: Vec<TimedSpike> = Vec::new();
    let samples = [0.9_f32, 0.2, 0.6];
    for &sample in &samples {
        collect(
            "latency",
            &latency.encode_step(&[sample, 1.0 - sample]),
            &latency_cursor,
            &mut timeline,
        );
        latency_cursor.advance();

        collect(
            "rate",
            &rate.encode_step(&[sample]),
            &rate_cursor,
            &mut timeline,
        );
        rate_cursor.advance();
    }

    // A shared unit is what makes a merged, chronological stream possible.
    timeline.sort_by_key(|spike| spike.nanos);

    for spike in &timeline {
        println!(
            "{:>8} channel {} at {:.3} ms",
            spike.source,
            spike.channel,
            spike.nanos as f64 / 1e6
        );
    }

    println!(
        "\nafter {} calls: latency cursor at tick {}, rate cursor at tick {}",
        samples.len(),
        latency_cursor.origin(),
        rate_cursor.origin()
    );
}

/// Converts one call's call-relative spikes onto the caller's timeline.
fn collect(
    source: &'static str,
    output: &EncodedOutput,
    cursor: &TimeCursor,
    timeline: &mut Vec<TimedSpike>,
) {
    for spike in &output.spikes {
        let nanos = cursor
            .absolute_nanos(spike.timestamp)
            .expect("cursor was built with a timebase");
        timeline.push(TimedSpike {
            source,
            channel: spike.channel,
            nanos,
        });
    }
}
