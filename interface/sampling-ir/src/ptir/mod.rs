//! # PTIR — the pie tensor IR (thrust-3 P0 substrate)
//!
//! This module generalizes the Sampling IR (one JIT'd program at one stage) to
//! the full PTIR model of `docs/ptir/overview.md`: **stage-tagged programs**
//! over the closed first-party op set, **channels** (the only stateful
//! construct), descriptor-port bindings, configuration sinks, and named
//! second-party kernels — carried in one versioned **trace container**
//! (`PTIR-CONTAINER.md`, magic `"PTIR"`).
//!
//! It deliberately does NOT touch the shipped PSIR v4 surface
//! ([`crate::types`], [`crate::bytecode`]): v4 programs stay byte-identical
//! (hashes stable), and the PTIR body op set is a **superset sharing v4's op
//! tags** where the ops coincide, so the CUDA driver's decoder extends rather
//! than forks.
//!
//! Layers (each its own submodule, mirroring the crate's three-layer shape):
//!
//! * [`op`] — the PTIR op enum + the **op table** (ids, names, families,
//!   arities): the single source of truth the generated C++ header
//!   (`include/ptir_abi.h`, [`header`]) is emitted from.
//! * [`container`] — the trace container data model + canonical LE
//!   encode/decode. Program identity = [`crate::program_hash`] (FNV-1a 64)
//!   over the canonical container bytes (C3).
//! * [`registry`] — stages, descriptor ports, first-party value intrinsics,
//!   well-known sink names, and the bind-time [`registry::ModelProfile`].
//! * [`infer`] — per-op shape/dtype inference over a stage body.
//! * [`validate`] — bind + the T-rule checks (SPSC, readiness direction
//!   table, sink stage-precedence, model gating, T10) producing a
//!   [`validate::BoundTrace`].
//! * [`expand`] — the composed ops (`gumbel`, `mask_apply`, `softmax`,
//!   `log_softmax`, `l2norm`) as expansions over the core.
//! * [`interp`] (feature `eval`) — the **tier-0 reference interpreter**: the
//!   golden model every backend diffs against. Executes bound traces
//!   cell-accurately per overview §1 + §7.1.
//! * [`header`] (feature `std`) — deterministic C header generator.

pub mod container;
pub mod expand;
pub mod infer;
pub mod op;
pub mod registry;
pub mod sidecar;
pub mod validate;

#[cfg(feature = "std")]
pub mod header;

#[cfg(feature = "eval")]
pub mod interp;

/// Container magic: ASCII `"PTIR"`.
pub const PTIR_MAGIC: [u8; 4] = *b"PTIR";

/// Container format version written + read by this module.
pub const PTIR_VERSION: u16 = 1;

/// v1.1: the extern-channel extension (§1 "SPSC pairs may span pipelines" in
/// the registration surface — Pentathlon gap G6). Encoded as wire-version 2
/// ONLY when the extern table is non-empty, so every version-1 container's
/// bytes — and therefore every existing hash — are unchanged.
pub const PTIR_VERSION_EXTERN: u16 = 2;

/// FNV-1a 64 over the canonical container bytes — the traced pass's identity
/// (contract C3: the batching/graph key component, the compile-cache key).
/// Seeds and per-instance data are NOT in the container, so identity is
/// instance-independent by construction (D2).
pub fn container_hash(container_bytes: &[u8]) -> u64 {
    crate::program_hash(container_bytes)
}
