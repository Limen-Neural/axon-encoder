use axon_encoder::encoders::{
    DeltaEncoder, PopulationEncoder, PredictiveEncoder, RateEncoder, TemporalEncoder,
};
use axon_encoder::prelude::*;
use std::alloc::{GlobalAlloc, Layout, System};
use std::hint::black_box;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

const SCALES: [usize; 3] = [256, 1024, 10_000];
const POISSON_STEPS: [usize; 3] = [10, 100, 1000];

struct CountingAllocator;

static COUNTING_ENABLED: AtomicBool = AtomicBool::new(false);
static ALLOCATION_COUNT: AtomicUsize = AtomicUsize::new(0);
static ALLOCATION_BYTES: AtomicUsize = AtomicUsize::new(0);

#[global_allocator]
static GLOBAL_ALLOCATOR: CountingAllocator = CountingAllocator;

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // Using SeqCst to ensure measurement boundaries are strictly respected.
        let ptr = unsafe { System.alloc(layout) };
        if COUNTING_ENABLED.load(Ordering::SeqCst) && !ptr.is_null() {
            ALLOCATION_COUNT.fetch_add(1, Ordering::SeqCst);
            ALLOCATION_BYTES.fetch_add(layout.size(), Ordering::SeqCst);
        }
        ptr
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        // Using SeqCst to ensure measurement boundaries are strictly respected.
        let ptr = unsafe { System.alloc_zeroed(layout) };
        if COUNTING_ENABLED.load(Ordering::SeqCst) && !ptr.is_null() {
            ALLOCATION_COUNT.fetch_add(1, Ordering::SeqCst);
            ALLOCATION_BYTES.fetch_add(layout.size(), Ordering::SeqCst);
        }
        ptr
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        // Using SeqCst to ensure measurement boundaries are strictly respected.
        let new_ptr = unsafe { System.realloc(ptr, layout, new_size) };
        if COUNTING_ENABLED.load(Ordering::SeqCst) && !new_ptr.is_null() {
            // Only count the net growth to avoid double-counting existing allocations.
            ALLOCATION_COUNT.fetch_add(1, Ordering::SeqCst);
            ALLOCATION_BYTES.fetch_add(new_size.saturating_sub(layout.size()), Ordering::SeqCst);
        }
        new_ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // dealloc is not tracked to focus on net growth metrics during the measured operation.
        unsafe { System.dealloc(ptr, layout) };
    }
}

#[derive(Clone, Copy)]
struct AllocationStats {
    allocations: usize,
    bytes: usize,
}

fn normalized_input(size: usize) -> Vec<f32> {
    (0..size)
        .map(|i| i as f32 / (size.saturating_sub(1).max(1) as f32))
        .collect()
}

fn shifted_input(size: usize, offset: f32) -> Vec<f32> {
    normalized_input(size)
        .into_iter()
        .map(|value| (value + offset).clamp(0.0, 10.0))
        .collect()
}

fn constant_input(size: usize, value: f32) -> Vec<f32> {
    vec![value; size]
}

fn measure_operation<T>(operation: impl FnOnce() -> T) -> AllocationStats {
    ALLOCATION_COUNT.store(0, Ordering::SeqCst);
    ALLOCATION_BYTES.store(0, Ordering::SeqCst);
    COUNTING_ENABLED.store(true, Ordering::SeqCst);
    let result = operation();
    COUNTING_ENABLED.store(false, Ordering::SeqCst);
    black_box(result);

    AllocationStats {
        allocations: ALLOCATION_COUNT.load(Ordering::SeqCst),
        bytes: ALLOCATION_BYTES.load(Ordering::SeqCst),
    }
}

fn print_stats(
    encoder: &str,
    operation: &str,
    scale_label: &str,
    scale: usize,
    stats: AllocationStats,
) {
    println!(
        "{encoder},{operation},{scale_label},{scale},{},{}",
        stats.allocations, stats.bytes
    );
}

fn report_rate_encoder() {
    for scale in SCALES {
        let mut encoder = RateEncoder::try_new(5.0, 100.0, (0.0, 1.0)).expect("valid RateEncoder");
        let input = normalized_input(scale);
        encoder.encode_step(&input);

        let stats = measure_operation(|| encoder.encode_step(&input));
        print_stats("RateEncoder", "encode_step", "scale", scale, stats);
    }
}

fn report_population_encoder() {
    for neurons in SCALES {
        let mut encoder = PopulationEncoder::try_new(neurons, (50.0, 100.0), 10.0)
            .expect("valid PopulationEncoder");
        let input = [75.0_f32];

        let stats = measure_operation(|| encoder.encode(&input));
        print_stats("PopulationEncoder", "encode", "neurons", neurons, stats);
    }
}

fn report_delta_encoder() {
    for scale in SCALES {
        let mut encoder = DeltaEncoder::try_new(0.1, scale).expect("valid DeltaEncoder");
        let baseline = normalized_input(scale);
        let shifted = shifted_input(scale, 0.25);
        encoder.encode_step(&baseline);

        let stats = measure_operation(|| encoder.encode_step(&shifted));
        print_stats("DeltaEncoder", "encode_step", "scale", scale, stats);
    }
}

fn report_temporal_encoder() {
    for scale in SCALES {
        let mut encoder =
            TemporalEncoder::try_new(6, vec![(0.2, 1)], scale).expect("valid TemporalEncoder");
        let low = constant_input(scale, 0.0);
        let high = constant_input(scale, 1.0);

        for input in [&low, &low, &low, &high, &high, &high] {
            encoder.encode_step(input);
        }

        let stats = measure_operation(|| encoder.encode_step(&high));
        print_stats("TemporalEncoder", "encode_step", "scale", scale, stats);
    }
}

fn report_predictive_encoder() {
    for scale in SCALES {
        let mut encoder =
            PredictiveEncoder::try_new(5, vec![(0.2, 1)], scale).expect("valid PredictiveEncoder");
        let low = constant_input(scale, 0.0);
        let high = constant_input(scale, 1.0);

        for _ in 0..5 {
            encoder.encode_step(&low);
        }

        let stats = measure_operation(|| encoder.encode_step(&high));
        print_stats("PredictiveEncoder", "encode_step", "scale", scale, stats);
    }
}

fn report_poisson_encoder() {
    for steps in POISSON_STEPS {
        let encoder = PoissonEncoder::new(steps);
        let stats = measure_operation(|| encoder.encode(0.5));
        print_stats("PoissonEncoder", "encode", "steps", steps, stats);
    }
}

fn main() {
    println!("encoder,operation,scale_type,scale,allocations,bytes");
    report_rate_encoder();
    report_population_encoder();
    report_delta_encoder();
    report_temporal_encoder();
    report_predictive_encoder();
    report_poisson_encoder();
}
