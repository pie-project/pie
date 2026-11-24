//! What the engine's integration tests share: a load's boot with the
//! defaults a test states nothing of ([`boot`]), and the hashes and
//! roundings their fixtures are seeded by.

mod boot;
pub use boot::{POOL, boot};

/// FNV-1a of a plane's name: the seed its fixture's values are drawn by.
pub fn fnv(name: &str) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for byte in name.as_bytes() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

/// `value` rounded to the nearest bf16, ties to even.
pub fn bf16_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    let rounding = 0x7fff + ((bits >> 16) & 1);
    ((bits + rounding) >> 16) as u16
}
