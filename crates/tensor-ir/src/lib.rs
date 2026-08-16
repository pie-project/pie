//! # `tensor-ir` — PTIR, the Pie Tensor IR
//!
//! The representation layer of Pie's programmable dataflow: **stage-tagged
//! programs** over a closed first-party op set, **channels** (the only stateful
//! construct), descriptor-port bindings, configuration sinks, and named
//! second-party kernels — carried in one versioned **trace container**
//! ([`container`], magic `"PTIR"`). It is `no_std` (+ `alloc`) so a program
//! can be authored/lowered without `std`; the host uses the default `std`
//! feature to parse and validate.
//!
//! This crate is the **dependency floor** the toolchain sits on:
//! `tensor-compiler`'s `plan` decides how to execute a program, its `eval`
//! computes what it produces, and its `codegen` emits backend source — all
//! three read the vocabulary defined here. It stays a crate of its own, and
//! did not fold with them, because it is the half a GUEST imports: `no_std`,
//! and reached by every wasm inferlet through `tensor-dsl`.
//!
//! ## Layers (each its own module)
//!
//! * [`types`] — the shape-typed primitive vocabulary ([`DType`], [`Shape`],
//!   [`ValueType`], [`Literal`], [`Predicate`], [`RngKind`]). A value's type is
//!   `{ shape: list<u32>, dtype }`.
//! * [`op`] — the PTIR op enum + the **op table** (ids, names, families,
//!   arities): the single source of truth every backend dispatches from.
//! * [`container`] — the trace container data model + canonical LE
//!   encode/decode. Program identity = [`container_hash`] (FNV-1a 64) over the
//!   canonical container bytes (contract C3).
//! * [`read`] — the bounds-checked little-endian cursor the container decoder
//!   reads through.
//! * [`registry`] — stages, descriptor ports, first-party value intrinsics,
//!   well-known sink names, and the bind-time model profile.
//! * [`infer`] — per-op shape/dtype inference over a stage body.
//! * [`validate`] — bind + the T-rule checks (SPSC, readiness direction table,
//!   sink stage-precedence, model gating) producing a bound trace.
//! * [`expand`] — the composed ops (`gumbel`, `mask_apply`, `softmax`,
//!   `log_softmax`, `l2norm`) as expansions over the core.
//! * [`wire`] — the flat wire record an op projects onto, and the two
//!   projections between it and the typed enum. [`container`]'s encoder and
//!   decoder are both walks over the layout declared in the op table.
//! * [`rng`] — the canonical keyed-RNG formula. Its deterministic CUDA/C++ and
//!   MSL projections live in `tensor-compiler`'s `codegen`.
//!

#![cfg_attr(not(feature = "std"), no_std)]
// This crate turns bytes it did not write into the vocabulary every other
// crate trusts, and it is the one crate in the toolchain with no truncating
// cast left in it. A narrowing `as` here would silently re-describe a decoded
// count, extent or index as a smaller one, and the program that results is
// well-formed -- so the failure surfaces as a wrong answer rather than a
// rejected container. Narrow through `try_from` and say what happens when it
// does not fit.
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

pub use types::{DType, Literal, MAX_RANK, Predicate, RngKind, Shape, ValueId, ValueType};

/// Container magic: ASCII `"PTIR"`.
pub const PTIR_MAGIC: [u8; 4] = *b"PTIR";

/// Container format version written + read by this crate.
pub const PTIR_VERSION: u16 = 1;

/// v1.1: the extern-channel extension (SPSC pairs may span pipelines). Encoded as
/// wire-version 2 ONLY when the extern table is non-empty, so every version-1
/// container's bytes — and therefore every existing hash — are unchanged.
pub const PTIR_VERSION_EXTERN: u16 = 2;

/// FNV-1a 64 — the one implementation, byte-identical to the CUDA driver's
/// `jit::fnv1a64`.
///
/// This is the *primitive*. It is not an identity: what a hash means depends on
/// what was fed to it, so the named contracts below (and
/// [`compile::stage_identity`](../tensor_tensor-compiler's plan/compile/fn.stage_identity.html)) say
/// which bytes they walk. Two callers hashing different byte sets must not be
/// read as agreeing just because they share this function.
///
/// Streaming callers — those hashing structured fields rather than one slice —
/// use [`Fnv1a`] so the constants stay written once.
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

/// FNV-1a 64 over a byte slice.
///
/// Prefer the named contract ([`container_hash`]) when the bytes are a
/// container; reach for this only where the input is some other canonical
/// byte string, so the call site does not claim an identity it does not have.
pub fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut hash = Fnv1a::new();
    hash.bytes(bytes);
    hash.finish()
}

/// FNV-1a 64 over the canonical container bytes — the traced pass's identity
/// (contract C3: the batching/graph key component, the compile-cache key). Seeds
/// and per-instance data are NOT in the container, so identity is
/// instance-independent by construction.
///
/// Because the container encoder is canonical (same trace ⟺ same bytes) this is
/// a *sound* identity key, and because it is byte-identical to the driver's
/// `jit::fnv1a64` the host program cache, the driver compile cache, and the
/// cross-request group key are one mechanism rather than three.
pub fn container_hash(container_bytes: &[u8]) -> u64 {
    fnv1a64(container_bytes)
}

#[cfg(test)]
mod fnv_tests {
    use super::Fnv1a;
    use alloc::vec::Vec;

    /// The published FNV-1a 64 vectors. Without these the two forms below could
    /// agree with each other on some *other* function and the driver would be
    /// the one to notice.
    #[test]
    fn the_hash_is_fnv_1a_64() {
        assert_eq!(super::fnv1a64(b""), 0xcbf2_9ce4_8422_2325);
        assert_eq!(super::fnv1a64(b"a"), 0xaf63_dc4c_8601_ec8c);
        assert_eq!(super::fnv1a64(b"foobar"), 0x8594_4171_f739_67e8);
    }

    /// Structured callers feed fields; slice callers feed bytes. They are the
    /// same walk, so a `u32_le` field must equal its four little-endian bytes —
    /// otherwise a host/driver pair hashing the same plan two ways disagrees.
    #[test]
    fn the_streaming_form_is_the_slice_form() {
        let mut streamed = Fnv1a::new();
        streamed.byte(0x7f);
        streamed.u32_le(0xdead_beef);
        streamed.bytes(b"tail");

        let mut flat: Vec<u8> = alloc::vec![0x7f];
        flat.extend_from_slice(&0xdead_beefu32.to_le_bytes());
        flat.extend_from_slice(b"tail");

        assert_eq!(streamed.finish(), super::fnv1a64(&flat));
    }

    /// FNV-1a is order-sensitive; a commutative mistake would make plans that
    /// differ only in field order share an identity.
    #[test]
    fn order_changes_the_hash() {
        let mut forward = Fnv1a::new();
        forward.byte(1);
        forward.byte(2);
        let mut backward = Fnv1a::new();
        backward.byte(2);
        backward.byte(1);
        assert_ne!(forward.finish(), backward.finish());
    }
}
