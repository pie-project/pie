//! The runtime vocabulary — every driver-owned object a routine may take.
//!
//! The third channel of the no-ask contract (`.wiki/designs/design-no-ask.md`
//! §0): a routine that reads driver-owned state takes it as an OPERAND —
//! `In<Struct<KvCache>>`, positional, counted by `arity_problem` — and the
//! driver answers a name from THIS list, never a free-form key. What
//! `keys.rs` was to `ctx.ask`, this module is to the operand form, with the
//! difference that made the migration: an operand is enumerable in the
//! derived column, so *"does every driver answer everything its plane's
//! routines name"* is a walkable test again.
//!
//! # Identity here, carrier in the plane, answer in the driver
//!
//! This module holds NAMES. The carrier a routine receives (`PagedKvView` on
//! CUDA, a bind-group entry on wgpu) is the plane's own type, declared beside
//! the plane's kernels with [`crate::resident!`] — the same split
//! [`crate::raises`] documents for per-fire objects, for the same reason:
//! this crate has no dependencies and cannot spell a plane's pointer.
//!
//! Tier-2 objects (`fa2.prefill`, dsv4's state views) are declared wholly in
//! their plane; only names every backend must answer live here.

/// One entry of the tier-1 vocabulary.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RuntimeEntry {
    /// The wire name a trace states and a driver answers.
    pub name: &'static str,
    /// Lives across fires (driver-allocated) rather than being raised per
    /// fire by a stated prep.
    pub resident: bool,
}

/// The tier-1 vocabulary: names every backend must answer.
pub const TIER1: &[RuntimeEntry] = &[
    // The paged KV cache, one view per (fire, layer): base planes, page
    // tables, geometry, quantisation scheme, and the write descriptor half.
    RuntimeEntry { name: "kv_cache", resident: true },
    // The recurrent-state slabs, one view per (fire, layer): state slab,
    // slot ids, strides, and the conv-window half.
    RuntimeEntry { name: "recurrent_state", resident: true },
    // Per-fire staged streams. Resident is false: the driver stages these
    // for the fire being bound, and a captured replay re-stages them.
    RuntimeEntry { name: "positions", resident: false },
    RuntimeEntry { name: "token_ids", resident: false },
    RuntimeEntry { name: "request_of_token", resident: false },
    RuntimeEntry { name: "qo_indptr", resident: false },
    RuntimeEntry { name: "row_valid", resident: false },
    RuntimeEntry { name: "attention_mask", resident: false },
    RuntimeEntry { name: "sampling_indices", resident: false },
    RuntimeEntry { name: "first_token", resident: false },
];

/// Whether `name` is in the tier-1 vocabulary.
#[must_use]
pub fn tier1(name: &str) -> Option<&'static RuntimeEntry> {
    TIER1.iter().find(|e| e.name == name)
}
