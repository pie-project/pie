//! Forward-pass declarations.
//!
//! A model family's forward pass is a **declaration**: ordinary Rust that
//! runs at *model-load time*, with the checkpoint's config facts in hand,
//! and records what one pass computes. Static control flow — layer kinds,
//! rope variant, qk-norm, whether the deployment bound a fused QKV — executes
//! during tracing and leaves no trace. What remains is the **traced form**:
//! the operation sequence a driver executes, with shapes symbolic in the
//! fire's extents and weights referenced by declaration name.
//!
//! The shape mirrors `loader/`:
//!
//! ```text
//! declaration  ──trace──▶  forward plan  ──(C ABI, `ffi/`)──▶  driver executes
//! (what a pass    (the ops to run,          (committed header,
//!  computes)       in what order)            generated)
//! ```
//!
//! Two rules, the first REVISED by north-star-dsl.md (2026-08-02):
//!
//! * **The declaration states the computation and the kernel choice in
//!   one text, and the driver is dumb.** A SEMANTIC trace (`llama_like`)
//!   names operations, never kernels — it is the general arm, the thing
//!   parity holds everything to. A LOWERED trace
//!   (`family::llama_like_cuda`) is the same declaration traced once per
//!   [`trace::FireClass`] with the backend facts in hand: its class arms
//!   state fusions ([`OpKind::QkvDecodeFusedPost`]) and kernels
//!   ([`trace::AttnKernel`]) as ordinary trace-time matches, and its
//!   traced form IS the launch form — statically convertible to the C++
//!   the driver runs. The driver never chooses between two kernels for
//!   semantic reasons; every choice is spelled in the program it
//!   received. (The prior reading — fusion as the C++ executor's peephole
//!   — put that choice on the wrong side of the ABI; the peephole's days
//!   are numbered by the migration ladder in north-star-dsl.md.)
//! * **Syntax is required exactly where cost is incurred.** A declaration
//!   with no structural divergence is an ordinary forward pass; the first
//!   family here (`llama_like`) has none, so nothing in it is `dyn`. The
//!   qwen3_5_moe MLP fragment (`family::qwen3_5_moe_mlp_block`) carries the
//!   first `dyn`: the per-token expert axis, whose lowering (grouped GEMM)
//!   is data-dependent and therefore the one thing the trace cannot fix —
//!   see the `trace` module doc's "`dyn`: the first per-token axis". The
//!   qwen3_5 GDN fragment (`family::qwen3_5_gdn_block`) carries the first
//!   PER-REQUEST state: the conv/recurrent slabs its ops address, implicit
//!   behind `layer` exactly as the KV cache is, marked by vocabulary
//!   (`OpKind::state_ref`) — the trace module doc's "the per-request state
//!   axis". The qwen3_5 full-attention fragment
//!   (`family::qwen3_5_full_attn_block`) completes the layer kinds — gated
//!   attention, partial rope — and `family::qwen3_5_hybrid` composes all
//!   three bodies into the first whole-model declaration beyond
//!   llama_like, its layer schedule a static match over the facts.

pub mod dsl;
pub mod emit_cuda;
pub mod emit_qwen35;
pub mod facts;
pub mod kernels;
pub mod lower;
pub mod family;
pub mod ffi;
pub mod trace;

pub use facts::{
    Gemma4Facts, LlamaLikeCudaFacts, LlamaLikeFacts, LlamaLikeMetalFacts, Qwen35CudaFacts, Qwen35FullAttnFacts, Qwen35GdnFacts,
    Qwen35HybridFacts, Qwen35MlpKind, Qwen35MoeMlpFacts,
};
pub use trace::{
    DType, Dim, DynAxis, FireClass, ForwardPlan, HookStage, Op, OpKind, Shape, StateRef,
    StateStore, TraceBuilder, ValueId,
};
