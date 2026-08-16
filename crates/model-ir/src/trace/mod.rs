//! The traced form: what one forward pass computes, as data.
//!
//! Values are SSA — each is produced by exactly one op — and shapes are
//! symbolic in the fire's extents (`Dim::Tokens`, `Dim::Requests`), because
//! the trace is taken once per model load, not per fire. Weights appear by
//! declaration name (`layer.3.qkv`); resolving names to device tensors is
//! the driver contract's job, exactly as it is for the loader.
//!
//! The op vocabulary is deliberately the *operation* vocabulary of the
//! hand-written passes, not their kernel vocabulary: `Matmul` + `SplitQkv` +
//! `RmsnormQk` + `Rope` is what the fused decode kernel computes, and
//! whether those four ops become one launch is the emitter's choice, made
//! per fire — the hook-free prefix taking the fused kernel while the tail
//! runs unfused (stage1-notes.md) is exactly that choice, and it is not
//! expressible if the trace bakes the fusion in.
//!
//! # `dyn`: the first per-token axis
//!
//! Everything above is resolved at trace time. The MoE expert axis is the
//! first thing that is not: `TopK` produces a per-token expert assignment
//! whose CONTENT exists only at fire time, and the expert-indexed `Matmul`s
//! downstream of it name a weight *template* (`layer.0.expert.{e}.gate_up`)
//! whose `{e}` the selector resolves per token. This is the first trace
//! whose lowering is not fixed at trace time — the expert dimension is
//! data — and, per the tart prototype's `ir.py`, per-token weight selection
//! IS `Div::Weight` at token granularity: gather → grouped GEMM → scatter is
//! its lowering, and `matmul(x, W[i])` with `i` per-token being MoE grouped
//! GEMM (with `i` per-request, SGMV) is the syntactic identity that
//! motivated this work (plan.md Part 1). The trace states the selection;
//! which grouped-GEMM strategy fires (cuBLAS batched, aligned blocks,
//! CUTLASS fused) stays the emitter's per-fire choice, exactly as fusion
//! does. The [`DynAxis`] marker on values and the `selector` field on
//! [`OpKind::Matmul`] are that syntax — present exactly where cost is
//! incurred, absent everywhere else.
//!
//! # The per-request state axis
//!
//! The GDN ops (`CausalConv1d`, `GatedDelta`) are the first whose semantics
//! include a store that is per-layer AND per-request: each request owns a
//! conv-window slab and a recurrent-state slab that the op reads and
//! advances in place, across fires (pie-application-plan.md §5.4's
//! `state[l] is per-request` — the axis the sketch left unmarked, and the
//! reason RS-touching fires are forced solo today, `touches_rs_buffer()`).
//! The trace marks it the way the KV cache is already marked: the ops carry
//! `layer` and the store stays implicit, NOT a traced value. That is a
//! deliberate design call, justified by the hand-written pass: state never
//! appears as an activation there — every state-touching kernel takes the
//! cache base plus a per-request slot indirection (`slot_ids_d`) and
//! mutates the slab in place — and a traced SSA value is per-fire and
//! single-assignment, so a first-class state value would misstate both the
//! lifetime (state outlives the fire) and the dataflow (state is not
//! produced by any op of this pass). What the planner needs is the FACT
//! that an op addresses such a store; [`OpKind::state_ref`] derives exactly
//! that from the vocabulary, so "does this trace touch per-request
//! recurrent state" is a query, not a name-match. (`DynAxis::PerRequest`
//! stays un-introduced: `dyn` marks values whose CONTENT selects structure,
//! and no state value exists to mark.)
//!
//! # By file
//!
//! `trace.rs` was 1,759 lines in one module. The cut is the order a reader
//! meets them in: the words a shape is written in (`types.rs`), the op
//! vocabulary built from them (`op.rs`), the plan that carries a list of
//! those (`plan.rs`), and the recorder that produces one (`builder.rs`).
//! Every name is re-exported flat -- `model_ir::trace::OpKind` is what a
//! driver spells, and a file boundary is not an API.

mod builder;
mod op;
mod plan;
mod types;

pub use builder::*;
pub use op::*;
pub use plan::*;
pub use types::*;
