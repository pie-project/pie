//! # `pie-sampling-ir` — the canonical Sampling IR for Pie programmable samplers
//!
//! This crate is the single source of truth for the **Sampling IR**: a small,
//! closed set of array primitives (map / reduce / scan / sort / pivot-threshold /
//! gather / scatter / rng) authored as a typed **shape-typed** SSA program,
//! validated, and lowered to a flat **versioned bytecode** that the CUDA driver
//! compiles to fused kernels.
//!
//! It is the **internal host↔driver encoding**: the inferlet front door is the
//! structured WIT `op-kind` surface, which the host lowers to this IR (1:1) →
//! `encode()` → bytecode → bridge → driver. It is `no_std` (+ `alloc`) so the
//! wasm SDK can author/lower without `std`; the host uses the default `std`
//! feature to parse and validate inferlet-supplied programs.
//!
//! ## The three layers
//!
//! * [`types`] — the typed SSA IR data model ([`SamplingProgram`], [`Op`],
//!   [`Shape`], …). A value's type is `{ shape: list<u32>, dtype }`.
//! * [`validate`] — SSA well-formedness + per-op shape/dtype checking.
//! * [`bytecode`] — the flat little-endian wire format ([`encode`]/[`decode`]),
//!   documented byte-for-byte in `BYTECODE.md` (the C++ driver's contract).
//!
//! ## SSA value-id model
//!
//! A program is **one flat SSA op list** plus its declared outputs. Every value
//! id is produced by an [`Op`] — including [`Op::Input`] (an external binding)
//! and [`Op::Const`] (a literal). Op at list position `p` defines `next_id ..
//! next_id + op.result_count()`, `next_id` starting at 0 (2 ids for
//! [`Op::SortDesc`], 1 otherwise). Operands reference earlier ids (no forward
//! refs).
//!
//! ```
//! use pie_sampling_ir::types::*;
//! use pie_sampling_ir::bytecode;
//!
//! // greedy argmax over the logits intrinsic vector (slot 0)
//! let prog = SamplingProgram {
//!     inputs: vec![InputDecl::new(Shape::vector(32_000), DType::F32)],
//!     ops: vec![
//!         Op::Input(0),            // id 0 : materialize slot 0 (logits)
//!         Op::ReduceArgmax(0),     // id 1 : scalar i32 token
//!     ],
//!     outputs: vec![OutputDecl::new(1, OutputKind::Token)],
//! };
//! prog.validate().unwrap();
//! let bytes = bytecode::encode(&prog);
//! assert_eq!(bytecode::decode(&bytes).unwrap(), prog);
//! ```

#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

pub mod bytecode;
pub mod types;
pub mod validate;
pub mod witmap;

/// CPU reference interpreter for the Sampling IR. Off by default; enable with the
/// `eval` feature (implies `std`). The canonical executable semantics the GPU
/// codegen must match — authored/owned by the L7 (test/parity) lane.
#[cfg(feature = "eval")]
pub mod eval;

pub use bytecode::{DecodeError, decode, decode_validated, encode};
pub use types::{
    Binding, DType, InputDecl, InputIndex, Literal, MAX_RANK, Op, OutputDecl, OutputKind,
    Predicate, Readiness, RngKind, SamplingProgram, Shape, TensorKey, ValueId, ValueType,
};
pub use validate::{
    ValidationError, input_first_use, late_input_barriers, output_kinds, output_types,
    program_from_parts, program_from_parts_typed, validate, value_types,
};
pub use witmap::OpKind;

/// Bytecode magic: ASCII `"PSIR"` (Pie Sampling IR).
pub const MAGIC: [u8; 4] = *b"PSIR";

/// Bytecode format version written + read by this crate.
///
/// v4 = the **shape-typed** layout (the new-interface convergence): a flat SSA
/// op list (no separate inputs vector / slots), `input`/`const` as op tags,
/// shapes as `rank:u8 | dims[]`. The structure differs from v1–v3, so a v4
/// reader accepts only `version == 4` (older streams are rejected cleanly).
pub const VERSION: u16 = 4;

/// FNV-1a 64-bit hash of a program's canonical bytecode — the program's identity.
///
/// Byte-identical to the CUDA driver's `jit::fnv1a64`, so `program_hash(&encode(p))`
/// equals the driver's `ProgramHandle` for the same program. Because [`encode`] is
/// canonical (same program ⟺ same bytes), this is a *sound* program-identity key —
/// the same value serves the host program cache, the driver compile cache, the
/// cross-request group key, and the program→kind hash-match recognizer (one
/// mechanism). It is the single FNV-1a impl: the runtime cache and the EDSL's
/// standard-program reference set both hash through here, so they cannot drift.
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

/// Canonicalize a program's op-shape for the `#12` recognizer.
///
/// Historically this zeroed a baked `RankLe(k)` immediate so top-k programs
/// differing only in `k` hashed alike. Under **#25**, `k` is a host-submit input
/// value-id (de-hardwired like top-p `p` / min-p `thr`), so top-k bytecode is
/// *already* k-invariant — there is no baked immediate to normalize. Retained as
/// a no-op for API stability across the recognizer lanes (EDSL / driver) pending
/// their drop of the call in the coordinated #25 land.
pub fn canonicalize_op_shape(_program: &mut SamplingProgram) {}

#[cfg(test)]
mod hash_tests {
    use super::program_hash;

    #[test]
    fn program_hash_matches_driver_fnv1a64_vectors() {
        // Standard FNV-1a-64 vectors == the driver's `jit::fnv1a64` (offset
        // 0xcbf29ce484222325, prime 0x100000001b3).
        assert_eq!(program_hash(b""), 0xcbf2_9ce4_8422_2325);
        assert_eq!(program_hash(b"a"), 0xaf63_dc4c_8601_ec8c);
        assert_eq!(program_hash(b"foobar"), 0x8594_4171_f739_67e8);
    }
}

#[cfg(test)]
mod canon_tests {
    use super::canonicalize_op_shape;
    use crate::types::*;
    use crate::{encode, program_hash};
    use alloc::vec;

    // #25: top-k declares `k` as a host-submit `U32` scalar input (slot 1), so the
    // bytecode is identical for every runtime `k`.
    fn topk() -> SamplingProgram {
        SamplingProgram {
            inputs: vec![
                InputDecl::new(Shape::vector(8), DType::F32),
                InputDecl::new(Shape::SCALAR, DType::U32),
            ],
            ops: vec![
                Op::Input(0),
                Op::Input(1),
                Op::PivotThreshold { input: 0, predicate: Predicate::RankLe(1) },
                Op::ReduceArgmax(2),
            ],
            outputs: vec![OutputDecl::new(3, OutputKind::Token)],
        }
    }

    #[test]
    fn topk_bytecode_is_k_invariant_by_construction() {
        // #25: `k` is a host-submit value-id, not a baked immediate — every top-k
        // program has identical bytecode regardless of the runtime `k`, so the #12
        // recognizer keys on the raw bytecode with no canonicalization.
        let (a, b) = (topk(), topk());
        assert_eq!(encode(&a), encode(&b));
        assert_eq!(program_hash(&encode(&a)), program_hash(&encode(&b)));
    }

    #[test]
    fn canonicalize_op_shape_is_noop() {
        // The #25 cleanup retires the RankLe-immediate hack: nothing to normalize.
        let mut p = topk();
        let before = encode(&p);
        canonicalize_op_shape(&mut p);
        assert_eq!(encode(&p), before);
    }
}
