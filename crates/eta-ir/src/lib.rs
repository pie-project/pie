#![cfg_attr(not(feature = "std"), no_std)]
#![deny(clippy::cast_possible_truncation)]

extern crate alloc;

pub mod container;
pub mod expand;
pub mod infer;
pub mod op;
pub mod read;
pub mod registry;
pub mod rng;
pub mod tagged;
pub mod types;
pub mod validate;
pub mod wire;

pub use types::{Dtype, Literal, MAX_RANK, Predicate, RngKind, Shape, ValueId, ValueType};

pub const ETA_MAGIC: [u8; 4] = *b"ETA\0";

pub const ETA_VERSION: u16 = 1;

pub const ETA_VERSION_EXTERN: u16 = 2;

#[derive(Clone, Copy, Debug)]
pub struct Fnv1a(u64);

impl Default for Fnv1a {
    fn default() -> Self {
        Self::new()
    }
}

impl Fnv1a {
    const OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
    const PRIME: u64 = 0x0000_0100_0000_01b3;

    pub const fn new() -> Self {
        Self(Self::OFFSET)
    }

    pub fn byte(&mut self, byte: u8) {
        self.0 ^= u64::from(byte);
        self.0 = self.0.wrapping_mul(Self::PRIME);
    }

    pub fn u32_le(&mut self, value: u32) {
        for byte in value.to_le_bytes() {
            self.byte(byte);
        }
    }

    pub fn bytes(&mut self, bytes: &[u8]) {
        for &byte in bytes {
            self.byte(byte);
        }
    }

    pub const fn finish(self) -> u64 {
        self.0
    }
}

pub fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut hash = Fnv1a::new();
    hash.bytes(bytes);
    hash.finish()
}

pub fn container_hash(container_bytes: &[u8]) -> u64 {
    fnv1a64(container_bytes)
}

#[cfg(test)]
mod fnv_tests {

    #[test]
    fn the_hash_is_fnv_1a_64() {
        assert_eq!(super::fnv1a64(b""), 0xcbf2_9ce4_8422_2325);
        assert_eq!(super::fnv1a64(b"a"), 0xaf63_dc4c_8601_ec8c);
        assert_eq!(super::fnv1a64(b"foobar"), 0x8594_4171_f739_67e8);
    }
}
