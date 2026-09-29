//! Buffered Streaming Example
//!
//! Shows the deferred-delivery integration path: instead of taking each call's
//! spikes directly, the caller wraps a borrowed `&mut dyn Encoder` in a
//! `StreamingEncoder`, buffers output in a bounded queue, and receives whole
//! `SpikeBatch`es through a closure `BatchSink` under an explicit `FlushPolicy`.
//!
//! It also reconstructs absolute time the crate's usual way: each batch carries
//! its `origin` (the absolute tick the emitting call started at) and the spikes
//! keep their unchanged call-relative offsets, so the absolute tick of a spike
//! is `batch.origin() + spike.timestamp.ticks()`.
//!
//! Feature-independent: builds and runs under default features.

use axon_encoder::prelude::*;

fn main() {
    println!("=== Buffered Streaming Example ===");

    // A concrete encoder, then a borrowed trait object. `StreamingEncoder`
    // holds `&'a mut E where E: Encoder + ?Sized`, so a `&mut dyn Encoder`
    // works directly and the caller keeps ownership of `concrete`.
    let mut concrete =
        RateEncoder::try_new(0.0, 100.0, (0.0, 1.0), 0.1).expect("valid RateEncoder configuration");
    let encoder: &mut dyn Encoder = &mut concrete;

    // Bounded queue of 6 spikes, delivering automatically when a new call would
    // overflow it (OnCapacity). Nothing is delivered while output still fits.
    let capacity = 6;
    let mut streaming = StreamingEncoder::try_new(encoder, capacity, FlushPolicy::OnCapacity)
        .expect("capacity is non-zero");

    println!(
        "Encoder: RateEncoder behind &mut dyn Encoder, queue capacity {} spikes, policy {:?}\n",
        streaming.capacity(),
        streaming.policy(),
    );

    // Caller-owned delivery target: a plain Vec the closure sink writes into.
    // Each delivered spike is stored as (sequence, channel, absolute_tick).
    let mut collected: Vec<(u64, u16, u64)> = Vec::new();

    // Drive several steps. A moderate input maps to a moderate rate, so each
    // step deterministically produces a few spikes that accumulate in the queue
    // until a later step would overflow it and trigger an OnCapacity flush.
    let steps = 6;
    for step in 0..steps {
        // The closure is the BatchSink (blanket impl for FnMut(SpikeBatch)).
        let report = streaming
            .encode_step(&[0.25], &mut |batch: SpikeBatch<'_>| {
                for spike in batch.spikes() {
                    // origin is absolute; the offset is left exactly as emitted.
                    let absolute_tick = batch.origin() + spike.timestamp.ticks();
                    collected.push((batch.sequence(), spike.channel, absolute_tick));
                }
            })
            .expect("OnCapacity never blocks");

        println!(
            "Step {step}: delivered {} batch(es) ({:?}), {} spike(s) still queued",
            report.delivered_batches(),
            report.reason(),
            report.queued_spikes(),
        );
    }

    // Deliver whatever remains queued after the loop.
    let flushed = streaming.flush_into(&mut |batch: SpikeBatch<'_>| {
        for spike in batch.spikes() {
            let absolute_tick = batch.origin() + spike.timestamp.ticks();
            collected.push((batch.sequence(), spike.channel, absolute_tick));
        }
    });
    println!(
        "\nFinal flush: delivered {} batch(es) ({:?})",
        flushed.delivered_batches(),
        flushed.reason(),
    );
    assert!(streaming.is_empty(), "flush drains everything");

    // Every spike arrived exactly once, ordered by call sequence, with an
    // absolute tick reconstructed from the batch origin.
    let earliest = collected.iter().map(|(_, _, tick)| *tick).min();
    let latest = collected.iter().map(|(_, _, tick)| *tick).max();
    println!(
        "\nCollected {} spike(s) across {steps} step(s); absolute ticks span {:?}..={:?}",
        collected.len(),
        earliest,
        latest,
    );
}
