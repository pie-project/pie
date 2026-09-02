//! `eta-ir`: ETA (Embedded Tensor Algebra), the representation layer of Pie's programmable dataflow — stage-tagged programs, channels, and a versioned trace container, `no_std` so a wasm inferlet can import it.

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

/// Container magic: ASCII `"ETA"` plus a NUL pad — four bytes, as the
/// header has always been.
pub const ETA_MAGIC: [u8; 4] = *b"ETA\0";

/// Container format version written + read by this crate.
pub const ETA_VERSION: u16 = 1;

/// v1.1: the extern-channel extension (SPSC pairs may span pipelines). Encoded as
/// wire-version 2 ONLY when the extern table is non-empty, so every version-1
/// container's bytes — and therefore every existing hash — are unchanged.
pub const ETA_VERSION_EXTERN: u16 = 2;

/// FNV-1a 64 — the one implementation, byte-identical to the CUDA engine's
/// FNV-1a. Not itself an identity: what a hash means depends on what was fed
/// to it, so callers must not assume agreement just because they share this function.
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

    /// A hasher at the FNV-1a offset basis.
    pub const fn new() -> Self {
        Self(Self::OFFSET)
    }

    /// Folds one byte in.
    pub fn byte(&mut self, byte: u8) {
        self.0 ^= u64::from(byte);
        self.0 = self.0.wrapping_mul(Self::PRIME);
    }

    /// Little-endian, four [`byte`](Self::byte) steps — the order the C++ walk uses.
    pub fn u32_le(&mut self, value: u32) {
        for byte in value.to_le_bytes() {
            self.byte(byte);
        }
    }

    /// Folds each byte of `bytes` in, in order.
    pub fn bytes(&mut self, bytes: &[u8]) {
        for &byte in bytes {
            self.byte(byte);
        }
    }

    /// The accumulated hash.
    pub const fn finish(self) -> u64 {
        self.0
    }
}

/// FNV-1a 64 over a byte slice. Prefer [`container_hash`] when the bytes are
/// a container.
pub fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut hash = Fnv1a::new();
    hash.bytes(bytes);
    hash.finish()
}

/// FNV-1a 64 over the canonical container bytes — the traced pass's
/// identity. Seeds and per-instance data are not in the container, so
/// identity is instance-independent by construction.
pub fn container_hash(container_bytes: &[u8]) -> u64 {
    fnv1a64(container_bytes)
}

#[cfg(test)]
mod fnv_tests {
    
    

    /// The published FNV-1a 64 vectors. Without these the two forms below could
    /// agree with each other on some *other* function and the engine would be
    /// the one to notice.
    #[test]
    fn the_hash_is_fnv_1a_64() {
        assert_eq!(super::fnv1a64(b""), 0xcbf2_9ce4_8422_2325);
        assert_eq!(super::fnv1a64(b"a"), 0xaf63_dc4c_8601_ec8c);
        assert_eq!(super::fnv1a64(b"foobar"), 0x8594_4171_f739_67e8);
    }

}
