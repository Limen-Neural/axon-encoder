use std::sync::atomic::{AtomicU64, Ordering};

static SEED: AtomicU64 = AtomicU64::new(0x9E3779B97F4A7C15);

#[inline]
fn next_u64() -> u64 {
    let mut x = SEED.load(Ordering::Relaxed);
    // xorshift64*
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    let next = x.wrapping_mul(0x2545F4914F6CDD1D);
    SEED.store(next, Ordering::Relaxed);
    next
}

#[inline]
pub fn gen_unit_f32() -> f32 {
    // [0,1)
    const SCALE: f32 = 1.0 / ((u32::MAX as f32) + 1.0);
    let v = (next_u64() >> 32) as u32;
    (v as f32) * SCALE
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gen_unit_f32() {
        for _ in 0..100 {
            let v = gen_unit_f32();
            assert!((0.0..1.0).contains(&v));
        }
    }
}
