//! Shared helpers for kernel tests: deterministic data and plain-Rust references.

use engine_cerebras::bench::round_bf16;

/// Hashed values in `[-1, 1)`, rounded to bf16.
pub fn data(n: usize, seed: u32) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let mut h = (i as u32)
                .wrapping_mul(2654435761)
                .wrapping_add(seed.wrapping_mul(97));
            h ^= h >> 13;
            h = h.wrapping_mul(0x5bd1e995);
            h ^= h >> 15;
            round_bf16((h % 20000) as f32 / 10000.0 - 1.0)
        })
        .collect()
}
