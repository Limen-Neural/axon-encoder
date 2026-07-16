//! Random number generation for stochastic encoders.
//!
//! Uses `rand::rng()` (thread-local `ThreadRng`) backed by OS entropy via
//! `getrandom` / `sys_rng`. Values are suitable for spike sampling and
//! stochastic encoding — **not** a general-purpose API for cryptographic
//! secrets or key material.
//!
//! When compiling for `wasm32-unknown-unknown`, downstream crates must enable
//! the appropriate `getrandom` JS/browser backend for their toolchain — see
//! the `getrandom` crate docs for the target you ship.

use rand::{Rng, RngExt};

/// Generates a random floating-point value in the range `[0, 1)`.
///
/// Uses the thread-local generator from `rand::rng()`. Prefer
/// [`gen_unit_f32_with_rng`] when drawing many samples in a loop so a single
/// handle can be reused (and when you need a seeded RNG for reproducible
/// experiments).
#[inline]
pub fn gen_unit_f32() -> f32 {
    let mut rng = rand::rng();
    gen_unit_f32_with_rng(&mut rng)
}

/// Generates a random floating-point value in the range `[0, 1)` from a
/// caller-provided RNG.
///
/// Reusing the same RNG produces a deterministic sequence for a given seed
/// and algorithm, which is useful for experiments and tests.
#[inline]
pub fn gen_unit_f32_with_rng<R>(rng: &mut R) -> f32
where
    R: Rng + ?Sized,
{
    rng.random::<f32>()
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

    #[test]
    fn test_gen_unit_f32_range() {
        for _ in 0..10_000 {
            let val = gen_unit_f32();
            assert!(val >= 0.0);
            assert!(val < 1.0);
        }
    }

    #[test]
    fn test_gen_unit_f32_multithreaded() {
        use std::thread;
        let mut handles = vec![];
        for _ in 0..8 {
            let handle = thread::spawn(|| {
                for _ in 0..1000 {
                    let val = gen_unit_f32();
                    assert!(val >= 0.0);
                    assert!(val < 1.0);
                }
            });
            handles.push(handle);
        }
        for handle in handles {
            handle.join().unwrap();
        }
    }

    #[test]
    fn test_gen_unit_f32_with_rng_seeded_is_deterministic() {
        use rand::SeedableRng;
        use rand::rngs::StdRng;

        let mut a = StdRng::seed_from_u64(42);
        let mut b = StdRng::seed_from_u64(42);
        let seq_a: Vec<f32> = (0..16).map(|_| gen_unit_f32_with_rng(&mut a)).collect();
        let seq_b: Vec<f32> = (0..16).map(|_| gen_unit_f32_with_rng(&mut b)).collect();
        assert_eq!(seq_a, seq_b);
        assert!(seq_a.iter().all(|&v| (0.0..1.0).contains(&v)));
    }
}
