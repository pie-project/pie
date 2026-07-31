//! The forward crate's C entry points.
//!
//! The driver calls in here; nothing calls out. The rules are the loader's
//! (`loader/src/ffi/entry.rs:1-19`), restated for this smaller surface:
//!
//! * **Never unwind.** A panic crossing into C++ is not something the driver
//!   can act on. The loader relies on `extern "C"`'s abort-on-unwind; here
//!   the catch is explicit — [`abort_on_panic`] — which changes nothing
//!   about the outcome and states the intent where the reader is. Nothing in
//!   this module converts a panic into a status code: a panic here is a
//!   tracer bug, and the shipping profile is `panic = "abort"` anyway
//!   (workspace `Cargo.toml` `[profile.release-min]`).
//! * **Never hold global state.** Tracing allocates nothing reachable except
//!   through the plan header it fills in, so concurrent traces (one per
//!   model, or per rank) cannot observe each other.
//! * **Status answers *did it work*.** The loader splits status from a
//!   diagnostics list because verification produces many findings; tracing
//!   produces none — the only failure mode is a malformed argument — so
//!   there is no diagnostics channel to carry, and the status is the whole
//!   answer.

use crate::facts::{
    LlamaLikeFacts, NormPlacement, QkNorm, Qwen35FullAttnFacts, Qwen35GdnFacts, Qwen35HybridFacts,
    Qwen35MlpKind, Qwen35MoeMlpFacts,
};
use crate::trace::{NormVariant, RopeKind};

use super::arena;
use super::types::PieForwardPlan;

#[repr(u32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PieForwardStatus {
    Ok = 0,
    /// The request was malformed: a null pointer or an out-of-range enum.
    /// The caller built the request wrong.
    InvalidArgument = 1,
}

/// The llama_like facts, as C states them. Mirrors
/// [`crate::facts::LlamaLikeFacts`] field for field.
///
/// `rope` and `norm_variant` are plain `uint32_t` rather than their enum
/// types for the input-side rule `loader/src/ffi/entry.rs` states on
/// `PieLoaderTargetSpec`: C++ lets a caller store any integer in an
/// enum-typed field, and reading such a field as a Rust enum is undefined
/// behaviour *before* any check could reject it. Assign
/// `static_cast<uint32_t>(PieForwardRopeKind::Standard)`; the tracer
/// validates the value and answers [`PieForwardStatus::InvalidArgument`] if
/// it is not one of them. The bools are `uint8_t` (zero is false) for the
/// same reason C ABIs avoid `bool` in input structs: every bit pattern is a
/// valid `u8`.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct PieForwardLlamaLikeFacts {
    pub hidden: u32,
    pub layers: u32,
    pub q_heads: u32,
    pub kv_heads: u32,
    pub head_dim: u32,
    pub intermediate: u32,
    pub vocab: u32,
    /// A [`super::types::PieForwardRopeKind`] value.
    pub rope: u32,
    /// A [`super::types::PieForwardNormVariant`] value.
    pub norm_variant: u32,
    /// A [`super::types::PieForwardNormPlacement`] value.
    pub norm_placement: u32,
    /// A [`super::types::PieForwardQkNorm`] value. (Formerly a 0/1 bool;
    /// `Off`/`PerHead` keep those wire values.)
    pub qk_norm: u32,
    /// The deployment bound one packed QKV projection; non-zero is true.
    pub fused_qkv: u8,
    /// The lm_head weight is the embedding table; non-zero is true.
    pub tied_embeddings: u8,
}

/// Validate the C facts into the tracer's own type.
///
/// Only the enums can be malformed; the widths are facts the caller states
/// and the tracer has no basis to second-guess (the loader takes the same
/// stance on the driver-measured target fields it copies straight through).
fn read_facts(facts: &PieForwardLlamaLikeFacts) -> Result<LlamaLikeFacts, PieForwardStatus> {
    let rope = RopeKind::try_from(facts.rope).map_err(|_| PieForwardStatus::InvalidArgument)?;
    let norm_variant =
        NormVariant::try_from(facts.norm_variant).map_err(|_| PieForwardStatus::InvalidArgument)?;
    let norm_placement = NormPlacement::try_from(facts.norm_placement)
        .map_err(|_| PieForwardStatus::InvalidArgument)?;
    let qk_norm =
        QkNorm::try_from(facts.qk_norm).map_err(|_| PieForwardStatus::InvalidArgument)?;
    Ok(LlamaLikeFacts {
        hidden: facts.hidden,
        layers: facts.layers,
        q_heads: facts.q_heads,
        kv_heads: facts.kv_heads,
        head_dim: facts.head_dim,
        intermediate: facts.intermediate,
        vocab: facts.vocab,
        rope,
        norm_variant,
        norm_placement,
        qk_norm,
        fused_qkv: facts.fused_qkv != 0,
        tied_embeddings: facts.tied_embeddings != 0,
    })
}

/// The qwen3_5_moe MLP-block facts, as C states them. Mirrors
/// [`crate::facts::Qwen35MoeMlpFacts`] field for field; same input-side
/// rules as [`PieForwardLlamaLikeFacts`].
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct PieForwardQwen35MoeMlpFacts {
    pub hidden: u32,
    pub num_experts: u32,
    pub top_k: u32,
    pub moe_intermediate: u32,
    /// 0 means no shared expert (the qwen3_moe shape).
    pub shared_expert_intermediate: u32,
    /// A [`super::types::PieForwardNormVariant`] value.
    pub norm_variant: u32,
}

fn read_moe_facts(
    facts: &PieForwardQwen35MoeMlpFacts,
) -> Result<Qwen35MoeMlpFacts, PieForwardStatus> {
    let norm_variant =
        NormVariant::try_from(facts.norm_variant).map_err(|_| PieForwardStatus::InvalidArgument)?;
    // A router with no experts or no routes is not a smaller MoE, it is a
    // malformed request: the tracer would emit ops whose k or E dimension
    // is zero, which no kernel means anything by.
    if facts.num_experts == 0 || facts.top_k == 0 || facts.moe_intermediate == 0 {
        return Err(PieForwardStatus::InvalidArgument);
    }
    Ok(Qwen35MoeMlpFacts {
        hidden: facts.hidden,
        num_experts: facts.num_experts,
        top_k: facts.top_k,
        moe_intermediate: facts.moe_intermediate,
        shared_expert_intermediate: facts.shared_expert_intermediate,
        norm_variant,
    })
}

/// The qwen3_5 GDN-block facts, as C states them. Mirrors
/// [`crate::facts::Qwen35GdnFacts`] field for field; same input-side rules
/// as [`PieForwardLlamaLikeFacts`].
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct PieForwardQwen35GdnFacts {
    pub hidden: u32,
    pub key_heads: u32,
    pub value_heads: u32,
    pub key_head_dim: u32,
    pub value_head_dim: u32,
    pub conv_kernel: u32,
    /// The deployment bound the fused `in_proj_qkvz`/`in_proj_ba` banks
    /// (`PIE_QWEN35_FUSED_GDN_PROJ`); non-zero is true.
    pub fused_in_proj: u8,
    /// A [`super::types::PieForwardNormVariant`] value.
    pub norm_variant: u32,
}

fn read_gdn_facts(facts: &PieForwardQwen35GdnFacts) -> Result<Qwen35GdnFacts, PieForwardStatus> {
    let norm_variant =
        NormVariant::try_from(facts.norm_variant).map_err(|_| PieForwardStatus::InvalidArgument)?;
    // A GDN block with no heads, zero-width heads or an empty conv window
    // is not a smaller block, it is a malformed request: the tracer would
    // emit ops whose dimensions no kernel means anything by. The GQA share
    // constraint the driver checks at engine load (value heads divide into
    // key heads) is validated here for the same reason.
    if facts.key_heads == 0
        || facts.value_heads == 0
        || facts.key_head_dim == 0
        || facts.value_head_dim == 0
        || facts.conv_kernel == 0
        || !facts.value_heads.is_multiple_of(facts.key_heads)
    {
        return Err(PieForwardStatus::InvalidArgument);
    }
    Ok(Qwen35GdnFacts {
        hidden: facts.hidden,
        key_heads: facts.key_heads,
        value_heads: facts.value_heads,
        key_head_dim: facts.key_head_dim,
        value_head_dim: facts.value_head_dim,
        conv_kernel: facts.conv_kernel,
        fused_in_proj: facts.fused_in_proj != 0,
        norm_variant,
    })
}

/// The qwen3_5 full-attention block facts, as C states them. Mirrors
/// [`crate::facts::Qwen35FullAttnFacts`] field for field; same input-side
/// rules as [`PieForwardLlamaLikeFacts`].
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct PieForwardQwen35FullAttnFacts {
    pub hidden: u32,
    pub q_heads: u32,
    pub kv_heads: u32,
    pub head_dim: u32,
    /// Partial-rotary width: the leading channels of each head that
    /// rotate. Must be in `1..=head_dim`.
    pub rotary_dim: u32,
    /// The deployment bound the fused `[2q | k | v]` bank
    /// (`PIE_QWEN35_FUSED_FULL_ATTN_QGKV`); non-zero is true.
    pub fused_qkv: u8,
    /// A [`super::types::PieForwardNormVariant`] value.
    pub norm_variant: u32,
}

fn read_full_attn_facts(
    facts: &PieForwardQwen35FullAttnFacts,
) -> Result<Qwen35FullAttnFacts, PieForwardStatus> {
    let norm_variant =
        NormVariant::try_from(facts.norm_variant).map_err(|_| PieForwardStatus::InvalidArgument)?;
    // No heads, zero-width heads, a GQA share that does not divide, or a
    // rotary width outside the head are not smaller blocks, they are
    // malformed requests (the same stance as the GDN validation).
    if facts.q_heads == 0
        || facts.kv_heads == 0
        || facts.head_dim == 0
        || !facts.q_heads.is_multiple_of(facts.kv_heads)
        || facts.rotary_dim == 0
        || facts.rotary_dim > facts.head_dim
    {
        return Err(PieForwardStatus::InvalidArgument);
    }
    Ok(Qwen35FullAttnFacts {
        hidden: facts.hidden,
        q_heads: facts.q_heads,
        kv_heads: facts.kv_heads,
        head_dim: facts.head_dim,
        rotary_dim: facts.rotary_dim,
        fused_qkv: facts.fused_qkv != 0,
        norm_variant,
    })
}

/// The qwen3_5 HYBRID model facts, as C states them. Mirrors
/// [`crate::facts::Qwen35HybridFacts`], with the MLP enum flattened the way
/// C states a sum type: `mlp_is_moe` selects which of
/// `dense_intermediate` / `moe` is read (the other is ignored). Same
/// input-side rules as [`PieForwardLlamaLikeFacts`].
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct PieForwardQwen35HybridFacts {
    pub layers: u32,
    /// One full-attention layer every Nth (at the end of each block);
    /// `1` means every layer. Must be non-zero.
    pub full_attn_interval: u32,
    pub vocab: u32,
    /// The lm_head weight is the embedding table; non-zero is true.
    pub tied_embeddings: u8,
    /// A [`super::types::PieForwardNormVariant`] value (the final norm's
    /// fold; block norms carry their own inside the sub-facts).
    pub norm_variant: u32,
    /// The full-attention layer kind.
    pub attn: PieForwardQwen35FullAttnFacts,
    /// The GDN linear-attention layer kind.
    pub gdn: PieForwardQwen35GdnFacts,
    /// Non-zero: the MLP is the MoE block (`moe`); zero: dense
    /// (`dense_intermediate`).
    pub mlp_is_moe: u8,
    /// The dense MLP's intermediate width; read only when `mlp_is_moe`
    /// is zero, and must then be non-zero.
    pub dense_intermediate: u32,
    /// The MoE MLP facts; read only when `mlp_is_moe` is non-zero.
    pub moe: PieForwardQwen35MoeMlpFacts,
}

fn read_hybrid_facts(
    facts: &PieForwardQwen35HybridFacts,
) -> Result<Qwen35HybridFacts, PieForwardStatus> {
    let norm_variant =
        NormVariant::try_from(facts.norm_variant).map_err(|_| PieForwardStatus::InvalidArgument)?;
    if facts.layers == 0 || facts.full_attn_interval == 0 || facts.vocab == 0 {
        return Err(PieForwardStatus::InvalidArgument);
    }
    let attn = read_full_attn_facts(&facts.attn)?;
    let gdn = read_gdn_facts(&facts.gdn)?;
    let mlp = if facts.mlp_is_moe != 0 {
        Qwen35MlpKind::Moe(read_moe_facts(&facts.moe)?)
    } else {
        if facts.dense_intermediate == 0 {
            return Err(PieForwardStatus::InvalidArgument);
        }
        Qwen35MlpKind::Dense {
            intermediate: facts.dense_intermediate,
        }
    };
    // The sub-facts each state hidden (they trace standalone too); a
    // disagreement is a malformed request, answered here rather than as
    // the tracer's assert-abort.
    let hidden = attn.hidden;
    if gdn.hidden != hidden {
        return Err(PieForwardStatus::InvalidArgument);
    }
    if let Qwen35MlpKind::Moe(moe) = &mlp
        && moe.hidden != hidden
    {
        return Err(PieForwardStatus::InvalidArgument);
    }
    Ok(Qwen35HybridFacts {
        layers: facts.layers,
        full_attn_interval: facts.full_attn_interval,
        vocab: facts.vocab,
        tied_embeddings: facts.tied_embeddings != 0,
        norm_variant,
        attn,
        gdn,
        mlp,
    })
}

/// Run `f`, aborting the process if it panics.
///
/// Equivalent to letting the unwind hit the `extern "C"` boundary — the
/// default panic hook has already printed the message by the time the catch
/// sees it — but explicit, so the "never unwind" rule is enforced by code
/// rather than by the edition's abort-on-unwind semantics.
fn abort_on_panic<T>(f: impl FnOnce() -> T) -> T {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(f))
        .unwrap_or_else(|_| std::process::abort())
}

/// Trace the llama_like family against `facts` and publish the traced form
/// into `*out_plan`.
///
/// The header is written into caller-owned storage; the slices inside it are
/// owned by Rust and must be reclaimed with [`pie_forward_release`]. On any
/// failure `*out_plan` (if writable) is left empty, never dangling.
///
/// # Safety
///
/// `facts` is null or points at a readable [`PieForwardLlamaLikeFacts`];
/// `out_plan` is null or a writable slot.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pie_forward_trace_llama_like(
    facts: *const PieForwardLlamaLikeFacts,
    out_plan: *mut PieForwardPlan,
) -> PieForwardStatus {
    abort_on_panic(|| {
        if out_plan.is_null() {
            return PieForwardStatus::InvalidArgument;
        }
        // Empty before anything can fail, so a caller that ignores the
        // status and releases anyway frees nothing rather than garbage.
        unsafe { *out_plan = PieForwardPlan::default() };
        if facts.is_null() {
            return PieForwardStatus::InvalidArgument;
        }
        let facts = match read_facts(unsafe { &*facts }) {
            Ok(facts) => facts,
            Err(status) => return status,
        };
        let plan = crate::family::llama_like(&facts);
        unsafe { *out_plan = arena::build(&plan) };
        PieForwardStatus::Ok
    })
}

/// Trace the qwen3_5_moe MoE MLP-block FRAGMENT against `facts` and publish
/// the traced form into `*out_plan`.
///
/// The result is the first traced form carrying `dyn` ops (`TopK`,
/// expert-selecting `Matmul`s, `WeightedSum`, `SigmoidGateAdd`). The
/// declared executors do NOT consume these — their op-kind switches throw
/// on any kind past their vocabulary — so this entry point exists for the
/// toolchain side (planning, tests, cross-language pinning), not for
/// emission; the grouped-GEMM emission is a later, much larger lift.
///
/// # Safety
///
/// `facts` is null or points at a readable [`PieForwardQwen35MoeMlpFacts`];
/// `out_plan` is null or a writable slot.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pie_forward_trace_qwen3_5_moe_mlp(
    facts: *const PieForwardQwen35MoeMlpFacts,
    out_plan: *mut PieForwardPlan,
) -> PieForwardStatus {
    abort_on_panic(|| {
        if out_plan.is_null() {
            return PieForwardStatus::InvalidArgument;
        }
        unsafe { *out_plan = PieForwardPlan::default() };
        if facts.is_null() {
            return PieForwardStatus::InvalidArgument;
        }
        let facts = match read_moe_facts(unsafe { &*facts }) {
            Ok(facts) => facts,
            Err(status) => return status,
        };
        let plan = crate::family::qwen3_5_moe_mlp_block(&facts);
        unsafe { *out_plan = arena::build(&plan) };
        PieForwardStatus::Ok
    })
}

/// Trace the qwen3_5 GDN (gated-deltanet) linear-attention block FRAGMENT
/// against `facts` and publish the traced form into `*out_plan`.
///
/// The result carries the GDN vocabulary (`SplitGdn`, `CausalConv1d`,
/// `GdnPrep`, `GatedDelta`, `RmsnormGated`) — and the first ops that
/// address PER-REQUEST state (the conv/recurrent slabs, implicit behind
/// `CausalConv1d`/`GatedDelta`'s layer, exactly as the KV cache is behind
/// `KvAppend`'s). The declared executors do NOT consume these — their
/// op-kind switches throw on any kind past their vocabulary — so this
/// entry point exists for the toolchain side (planning, tests,
/// cross-language pinning), not for emission.
///
/// # Safety
///
/// `facts` is null or points at a readable [`PieForwardQwen35GdnFacts`];
/// `out_plan` is null or a writable slot.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pie_forward_trace_qwen3_5_gdn(
    facts: *const PieForwardQwen35GdnFacts,
    out_plan: *mut PieForwardPlan,
) -> PieForwardStatus {
    abort_on_panic(|| {
        if out_plan.is_null() {
            return PieForwardStatus::InvalidArgument;
        }
        unsafe { *out_plan = PieForwardPlan::default() };
        if facts.is_null() {
            return PieForwardStatus::InvalidArgument;
        }
        let facts = match read_gdn_facts(unsafe { &*facts }) {
            Ok(facts) => facts,
            Err(status) => return status,
        };
        let plan = crate::family::qwen3_5_gdn_block(&facts);
        unsafe { *out_plan = arena::build(&plan) };
        PieForwardStatus::Ok
    })
}

/// Trace the qwen3_5 FULL-attention block FRAGMENT against `facts` and
/// publish the traced form into `*out_plan`.
///
/// The result carries the full-attention vocabulary (`SplitQGate`,
/// `SigmoidGateMul`, the partial `Rope` and the Gemma-fold
/// `RmsnormPerHead`) alongside `KvAppend`/`Attention` marking the layer's
/// KV cache. The declared executors do NOT consume these — their op-kind
/// switches throw on any kind past their vocabulary — so, like the MoE and
/// GDN fragments, this entry point exists for the toolchain side (planning,
/// tests, cross-language pinning), not for emission.
///
/// # Safety
///
/// `facts` is null or points at a readable [`PieForwardQwen35FullAttnFacts`];
/// `out_plan` is null or a writable slot.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pie_forward_trace_qwen3_5_full_attn(
    facts: *const PieForwardQwen35FullAttnFacts,
    out_plan: *mut PieForwardPlan,
) -> PieForwardStatus {
    abort_on_panic(|| {
        if out_plan.is_null() {
            return PieForwardStatus::InvalidArgument;
        }
        unsafe { *out_plan = PieForwardPlan::default() };
        if facts.is_null() {
            return PieForwardStatus::InvalidArgument;
        }
        let facts = match read_full_attn_facts(unsafe { &*facts }) {
            Ok(facts) => facts,
            Err(status) => return status,
        };
        let plan = crate::family::qwen3_5_full_attn_block(&facts);
        unsafe { *out_plan = arena::build(&plan) };
        PieForwardStatus::Ok
    })
}

/// Trace the full qwen3_5 HYBRID model against `facts` and publish the
/// traced form into `*out_plan` — the first whole-model entry point beyond
/// llama_like: embed → per-layer {GDN or full attention, per the
/// checkpoint's layer schedule; dense or MoE MLP} → final norm → lm_head.
///
/// The result composes every vocabulary the fragments introduced (dyn MoE
/// ops when the facts say MoE, the GDN per-request-state ops, the
/// full-attention gate/partial-rope ops), so the declared executors refuse
/// it loudly; the entry point serves the toolchain side.
///
/// # Safety
///
/// `facts` is null or points at a readable [`PieForwardQwen35HybridFacts`];
/// `out_plan` is null or a writable slot.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pie_forward_trace_qwen3_5_hybrid(
    facts: *const PieForwardQwen35HybridFacts,
    out_plan: *mut PieForwardPlan,
) -> PieForwardStatus {
    abort_on_panic(|| {
        if out_plan.is_null() {
            return PieForwardStatus::InvalidArgument;
        }
        unsafe { *out_plan = PieForwardPlan::default() };
        if facts.is_null() {
            return PieForwardStatus::InvalidArgument;
        }
        let facts = match read_hybrid_facts(unsafe { &*facts }) {
            Ok(facts) => facts,
            Err(status) => return status,
        };
        let plan = crate::family::qwen3_5_hybrid(&facts);
        unsafe { *out_plan = arena::build(&plan) };
        PieForwardStatus::Ok
    })
}

/// Free the storage behind a plan header filled by
/// [`pie_forward_trace_llama_like`], [`pie_forward_trace_qwen3_5_moe_mlp`],
/// [`pie_forward_trace_qwen3_5_gdn`],
/// [`pie_forward_trace_qwen3_5_full_attn`] or
/// [`pie_forward_trace_qwen3_5_hybrid`], and reset the header to empty.
///
/// Safe to call with null, with a header that was never filled, or twice
/// with the same header (the first call empties it).
///
/// # Safety
///
/// `plan` is null, or points at a writable header that is empty or was
/// filled by [`pie_forward_trace_llama_like`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pie_forward_release(plan: *mut PieForwardPlan) {
    abort_on_panic(|| unsafe { arena::release(plan) })
}

/// A function pointer that is `Sync` by fiat, so the table below can be a
/// `static`. Function addresses are immutable; the wrapper exists only
/// because raw pointers are not `Sync` by default.
#[repr(transparent)]
struct EntryAddr(*const ());
unsafe impl Sync for EntryAddr {}

/// Anchors the entry points against dead-code elimination.
///
/// `pie-forward` is an rlib, and nothing in Rust calls these functions — the
/// only caller is the C++ driver, linked afterwards. Without a reference
/// from a reachable item, `rustc` and the linker are free to drop
/// `#[no_mangle]` functions from an rlib, and the failure surfaces as an
/// undefined symbol at final link rather than anywhere near this file
/// (`loader/src/ffi/entry.rs:637-652`, `loader/architecture.md` §3.4).
/// `#[used]` keeps the table, and the table keeps the functions.
#[used]
static KEEP_ALIVE: [EntryAddr; 6] = [
    EntryAddr(pie_forward_trace_llama_like as *const ()),
    EntryAddr(pie_forward_trace_qwen3_5_moe_mlp as *const ()),
    EntryAddr(pie_forward_trace_qwen3_5_gdn as *const ()),
    EntryAddr(pie_forward_trace_qwen3_5_full_attn as *const ()),
    EntryAddr(pie_forward_trace_qwen3_5_hybrid as *const ()),
    EntryAddr(pie_forward_release as *const ()),
];
