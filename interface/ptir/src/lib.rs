//! # `pie-ptir` — PTIR, the Pie Tensor IR
//!
//! The single cross-language seam for Pie's programmable dataflow: **stage-tagged
//! programs** over a closed first-party op set, **channels** (the only stateful
//! construct), descriptor-port bindings, configuration sinks, and named
//! second-party kernels — carried in one versioned **trace container**
//! (`PTIR-CONTAINER.md`, magic `"PTIR"`). It is `no_std` (+ `alloc`) so a program
//! can be authored/lowered without `std`; the host uses the default `std`
//! feature to parse and validate.
//!
//! ## Layers (each its own module)
//!
//! * [`types`] — the shape-typed primitive vocabulary ([`DType`], [`Shape`],
//!   [`ValueType`], [`Literal`], [`Predicate`], [`RngKind`]). A value's type is
//!   `{ shape: list<u32>, dtype }`.
//! * [`op`] — the PTIR op enum + the **op table** (ids, names, families,
//!   arities): the single source of truth the generated C++ header
//!   (`include/ptir_abi.h`, [`header`]) is emitted from.
//! * [`container`] — the trace container data model + canonical LE
//!   encode/decode. Program identity = [`container_hash`] (FNV-1a 64) over the
//!   canonical container bytes (contract C3).
//! * [`registry`] — stages, descriptor ports, first-party value intrinsics,
//!   well-known sink names, and the bind-time model profile.
//! * [`infer`] — per-op shape/dtype inference over a stage body.
//! * [`validate`] — bind + the T-rule checks (SPSC, readiness direction table,
//!   sink stage-precedence, model gating) producing a bound trace.
//! * [`expand`] — the composed ops (`gumbel`, `mask_apply`, `softmax`,
//!   `log_softmax`, `l2norm`) as expansions over the core.
//! * [`rng`] — the canonical keyed-RNG formula and generated CUDA/C++ and MSL
//!   projections.
//! * [`interp`] (feature `eval`) — the **tier-0 reference interpreter**: the
//!   golden model every backend diffs against.
//! * [`header`] (feature `std`) — deterministic C header generator.

#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

pub mod compiler;
pub mod container;
pub mod expand;
pub mod infer;
pub mod op;
pub mod registry;
pub mod rng;
pub mod sidecar;
pub mod stage_key;
pub mod types;
pub mod validate;

#[cfg(feature = "std")]
pub mod header;

#[cfg(feature = "eval")]
pub mod interp;

#[cfg(feature = "eval")]
pub mod pareval;

pub use types::{DType, Literal, MAX_RANK, Predicate, RngKind, Shape, ValueId, ValueType};

/// Container magic: ASCII `"PTIR"`.
pub const PTIR_MAGIC: [u8; 4] = *b"PTIR";

/// Container format version written + read by this crate.
pub const PTIR_VERSION: u16 = 1;

/// v1.1: the extern-channel extension (SPSC pairs may span pipelines). Encoded as
/// wire-version 2 ONLY when the extern table is non-empty, so every version-1
/// container's bytes — and therefore every existing hash — are unchanged.
pub const PTIR_VERSION_EXTERN: u16 = 2;

/// FNV-1a 64-bit hash of a container's canonical bytes — its identity.
///
/// Byte-identical to the CUDA driver's `jit::fnv1a64`, so a container's hash
/// equals the driver's handle for the same bytes. Because the container encoder
/// is canonical (same trace ⟺ same bytes), this is a *sound* identity key — the
/// same value serves the host program cache, the driver compile cache, and the
/// cross-request group key (one mechanism, one FNV-1a impl).
pub fn program_hash(bytecode: &[u8]) -> u64 {
    const OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
    const PRIME: u64 = 0x0000_0100_0000_01b3;
    let mut h = OFFSET;
    for &b in bytecode {
        h ^= b as u64;
        h = h.wrapping_mul(PRIME);
    }
    h
}

/// FNV-1a 64 over the canonical container bytes — the traced pass's identity
/// (contract C3: the batching/graph key component, the compile-cache key). Seeds
/// and per-instance data are NOT in the container, so identity is
/// instance-independent by construction (D2).
pub fn container_hash(container_bytes: &[u8]) -> u64 {
    program_hash(container_bytes)
}
