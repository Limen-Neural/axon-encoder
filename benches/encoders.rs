use axon_encoder::encoders::{
    DeltaEncoder, PopulationEncoder, PredictiveEncoder, RateEncoder, TemporalEncoder,
};
use axon_encoder::prelude::*;
use axon_encoder::PoissonEncoder;
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

fn bench_rate_encoder(c: &mut Criterion) {
    let mut group = c.benchmark_group("RateEncoder::encode");
    for size in [10, 100, 1000, 10000].iter() {
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, size| {
            let mut encoder = RateEncoder::new(5.0, 100.0, (0.0, 1.0));
            let input: Vec<f32> = (0..*size).map(|i| i as f32 / *size as f32).collect();
            b.iter(|| encoder.encode(black_box(&input)));
        });
    }
    group.finish();
}

fn bench_population_encoder(c: &mut Criterion) {
    let mut group = c.benchmark_group("PopulationEncoder::encode");
    for size in [10, 100, 1000, 10000].iter() {
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, size| {
            let mut encoder = PopulationEncoder::new(*size, (50.0, 100.0), 10.0);
            let input: Vec<f32> = (0..*size).map(|i| i as f32 / *size as f32).collect();
            b.iter(|| encoder.encode(black_box(&input)));
        });
    }
    group.finish();
}

fn bench_delta_encoder(c: &mut Criterion) {
    let mut group = c.benchmark_group("DeltaEncoder::encode");
    for size in [10, 100, 1000, 10000].iter() {
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, size| {
            let mut encoder = DeltaEncoder::new(0.1, *size);
            let input: Vec<f32> = (0..*size).map(|i| i as f32 / *size as f32 * 10.0).collect();
            b.iter(|| encoder.encode(black_box(&input)));
        });
    }
    group.finish();
}

fn bench_temporal_encoder(c: &mut Criterion) {
    let mut group = c.benchmark_group("TemporalEncoder::encode");
    for size in [10, 100, 1000, 10000].iter() {
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, size| {
            let mut encoder = TemporalEncoder::new(5, vec![(0.5, 1)], *size);
            let input: Vec<f32> = (0..*size).map(|i| i as f32 / *size as f32).collect();
            b.iter(|| encoder.encode(black_box(&input)));
        });
    }
    group.finish();
}

fn bench_predictive_encoder(c: &mut Criterion) {
    let mut group = c.benchmark_group("PredictiveEncoder::encode");
    for size in [10, 100, 1000, 10000].iter() {
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, size| {
            let mut encoder = PredictiveEncoder::new(5, vec![(0.5, 1)], *size);
            let input: Vec<f32> = (0..*size).map(|i| i as f32 / *size as f32 * 10.0).collect();
            b.iter(|| encoder.encode(black_box(&input)));
        });
    }
    group.finish();
}

fn bench_poisson_encoder(c: &mut Criterion) {
    let mut group = c.benchmark_group("PoissonEncoder::encode");
    for steps in [10, 100, 1000].iter() {
        group.bench_with_input(BenchmarkId::from_parameter(steps), steps, |b, steps| {
            let enc = PoissonEncoder::new(*steps);
            b.iter(|| enc.encode(black_box(0.5)));
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_rate_encoder,
    bench_population_encoder,
    bench_delta_encoder,
    bench_temporal_encoder,
    bench_predictive_encoder,
    bench_poisson_encoder
);
criterion_main!(benches);