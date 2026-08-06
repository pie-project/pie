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
    LlamaLikeCudaFacts, LlamaLikeFacts, NormPlacement, QkNorm, Qwen35CudaFacts,
    Qwen35FullAttnFacts, Qwen35GdnFacts, Qwen35HybridFacts, Qwen35MlpKind, Qwen35MoeMlpFacts,
};
use crate::trace::{FireClass, NormVariant, RopeKind};

use super::arena;
use super::types::{PieForwardLowered, PieForwardPlan, PieForwardRow};

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
    /// Qwen-2 family attention biases (`{q,k,v}_proj.bias` bound and
    /// added to the raw projections); non-zero is true. Appended field:
    /// existing zero-initialized C callers read as false.
    pub qkv_bias: u8,
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
        qkv_bias: facts.qkv_bias != 0,
    })
}

/// The CUDA backend facts for a LOWERED llama_like trace, as C states
/// them. Mirrors [`crate::facts::LlamaLikeCudaFacts`] field for field;
/// same input-side rules as [`PieForwardLlamaLikeFacts`] (the bools are
/// `uint8_t`, non-zero is true).
///
/// The driver fills this from its OWN derivation (env, kernel-support
/// predicates, binding, cache format) at cold start — the same terms its
/// executor booleans compute today — and the returned class traces then
/// STATE every kernel, which is what lets the executor go dumb
/// (north-star-dsl.md, migration rung 2).
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct PieForwardLlamaLikeCudaFacts {
    /// XQA decode eligibility; non-zero is true.
    pub xqa_decode: u8,
    /// The fused decode-QKV epilogue's load-time terms hold.
    pub decode_fused_post: u8,
    /// The workspace carries a rope table.
    pub rope_table: u8,
    /// FlashInfer's decode set lacks this GQA ratio.
    pub force_prefill_path: u8,
    /// Attention runs at a padded kernel head dim (Phi-3's 96 → 128);
    /// non-zero is true. Appended field: existing zero-initialized C
    /// callers read as false.
    pub head_dim_padded: u8,
    /// The checkpoint bound a packed gate‖up bank, so the MLP activation
    /// is the chunked swiglu over one buffer rather than the pair form
    /// over two. Appended field, same zero-init rule — and note the
    /// default is the UNFUSED form, which is the conservative one: it
    /// reads the two narrow buffers a decliner writes.
    pub gate_up_fused: u8,
}

fn read_cuda_facts(facts: &PieForwardLlamaLikeCudaFacts) -> LlamaLikeCudaFacts {
    LlamaLikeCudaFacts {
        xqa_decode: facts.xqa_decode != 0,
        decode_fused_post: facts.decode_fused_post != 0,
        rope_table: facts.rope_table != 0,
        head_dim_padded: facts.head_dim_padded != 0,
        force_prefill_path: facts.force_prefill_path != 0,
        gate_up_fused: facts.gate_up_fused != 0,
    }
}

/// Mirrors [`crate::trace::FireClass`]; same appended-only discriminant
/// rule as [`PieForwardOpKind`], same input-side `uint32_t` crossing rule
/// as every enum here.
#[repr(u32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PieForwardFireClass {
    Decode = 0,
    Prefill = 1,
    /// qwen3_5 MTP service classes (2/3/4); llama_like rejects them.
    CommitAdvance = 2,
    StateOnly = 3,
    /// Reserved (the frozen-verify slice); both entries reject it today.
    FrozenVerify = 4,
    /// RETIRED (A1/A2, the class-collapse amendment): a custom mask is
    /// a HasCustomMask guard arm and attached stage hooks are a
    /// HasStageHooks guard arm of classes 0/1. The discriminants stay
    /// reserved (append-only rule); both entries reject them.
    MaskedDecode = 5,
    MaskedPrefill = 6,
    HookedDecode = 7,
    HookedPrefill = 8,
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

/// The CUDA backend facts for a LOWERED qwen3_5 hybrid trace, as C
/// states them. Mirrors [`crate::facts::Qwen35CudaFacts`] field for
/// field; same input-side rules as [`PieForwardLlamaLikeFacts`] (the
/// bools are `uint8_t`, non-zero is true; the thresholds are plain
/// `uint32_t` values the tracer has no basis to second-guess).
///
/// The driver fills this from its OWN derivation (the env gates
/// `PIE_QWEN35_GDN_WARP_TILED_STATE_PERSIST` /
/// `..._WARP_TILED_MAX_TOKENS` / `..._CACHED_PREFILL_MAX_TOKENS`, the
/// state dtype, the K_d bound) at cold start — the same terms
/// `declared_forward.cpp`'s hoisted predicates compute today.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct PieForwardQwen35CudaFacts {
    /// The recurrent-state store dtype is bf16; non-zero is true.
    pub state_bf16: u8,
    /// The warp-tiled prefill arm exists at all (K_d bound && the
    /// state-persist env gate); non-zero is true.
    pub warp_tiled: u8,
    /// `qwen35_gdn_warp_tiled_max_tokens()` — the warp-tiled arm's
    /// `TokensLE` payload.
    pub warp_tiled_max: u32,
    /// `qwen35_gdn_cached_prefill_max_tokens()` — the cached arm's
    /// `TokensLE` payload.
    pub cached_max: u32,
    /// The deployment configures the verify hidden stash
    /// (`RecurrentStateCache::configure_verify_hidden_stash`): the
    /// CommitAdvance class replays the linear layers' in-proj outputs
    /// from the stash instead of re-running the GEMMs. Non-zero is true.
    /// Appended field (4c-iv) — the append-only struct discipline.
    pub verify_stash: u8,
}

fn read_qwen35_cuda_facts(facts: &PieForwardQwen35CudaFacts) -> Qwen35CudaFacts {
    Qwen35CudaFacts {
        state_bf16: facts.state_bf16 != 0,
        warp_tiled: facts.warp_tiled != 0,
        warp_tiled_max: facts.warp_tiled_max,
        cached_max: facts.cached_max,
        verify_stash: facts.verify_stash != 0,
    }
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

/// Trace the LOWERED llama_like — the same text as
/// [`pie_forward_trace_llama_like`], with the CUDA backend facts and a
/// fire class in hand, so the class arms run and the traced form states
/// kernels (`QkvDecodeFusedPost`, `RopeTableBuild`, `Attention.param1`;
/// north-star-dsl.md). Call once per class the deployment fires; the
/// semantic entry remains the parity reference.
///
/// # Safety
///
/// `facts` / `cuda` are null or point at readable
/// [`PieForwardLlamaLikeFacts`] / [`PieForwardLlamaLikeCudaFacts`];
/// `out_plan` is null or a writable slot. `class` is a
/// [`PieForwardFireClass`] value crossed as `uint32_t` (the input-side
/// enum rule); anything else answers `InvalidArgument`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pie_forward_trace_llama_like_cuda(
    facts: *const PieForwardLlamaLikeFacts,
    cuda: *const PieForwardLlamaLikeCudaFacts,
    class: u32,
    out_plan: *mut PieForwardPlan,
) -> PieForwardStatus {
    abort_on_panic(|| {
        if out_plan.is_null() {
            return PieForwardStatus::InvalidArgument;
        }
        unsafe { *out_plan = PieForwardPlan::default() };
        if facts.is_null() || cuda.is_null() {
            return PieForwardStatus::InvalidArgument;
        }
        let facts = match read_facts(unsafe { &*facts }) {
            Ok(facts) => facts,
            Err(status) => return status,
        };
        let cuda = read_cuda_facts(unsafe { &*cuda });
        let class = match class {
            0 => FireClass::Decode,
            1 => FireClass::Prefill,
            // 5/6 (the masked classes) and 7/8 (the hooked classes) are
            // RETIRED (A1/A2, the class-collapse amendment): a custom
            // mask is a HasCustomMask guard arm and attached hooks are
            // a HasStageHooks guard arm of classes 0/1 now. The wire
            // numbers stay reserved; requesting them is malformed.
            // 2/3/4 are qwen3_5's service classes; llama_like has no
            // MTP, so they stay malformed requests here too.
            _ => return PieForwardStatus::InvalidArgument,
        };
        let plan = crate::family::llama_like_cuda(&facts, &cuda, class);
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

/// Trace the LOWERED qwen3_5 hybrid — the same text as
/// [`pie_forward_trace_qwen3_5_hybrid`], with the CUDA backend facts and
/// a fire class in hand, so the class arms run and the traced form
/// states its kernels as `Launch` ops with the recurrence three-way
/// behind value-producing `Guard` chains (north-star-dsl.md rung 4c).
/// Call once per class the deployment fires; the semantic entry remains
/// the parity reference.
///
/// # Safety
///
/// `facts` / `cuda` are null or point at readable
/// [`PieForwardQwen35HybridFacts`] / [`PieForwardQwen35CudaFacts`];
/// `out_plan` is null or a writable slot. `class` is a
/// [`PieForwardFireClass`] value crossed as `uint32_t` (the input-side
/// enum rule); anything else answers `InvalidArgument`. ALL FOUR classes
/// are traceable here (4c-iv): the service classes CommitAdvance (2 —
/// family `qwen3_5_hybrid.cuda.commit_advance`) and StateOnly (3 —
/// `...state_only`) alongside Decode/Prefill. They remain qwen3_5's:
/// the llama_like entry keeps refusing them.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pie_forward_trace_qwen3_5_hybrid_cuda(
    facts: *const PieForwardQwen35HybridFacts,
    cuda: *const PieForwardQwen35CudaFacts,
    class: u32,
    out_plan: *mut PieForwardPlan,
) -> PieForwardStatus {
    abort_on_panic(|| {
        if out_plan.is_null() {
            return PieForwardStatus::InvalidArgument;
        }
        unsafe { *out_plan = PieForwardPlan::default() };
        if facts.is_null() || cuda.is_null() {
            return PieForwardStatus::InvalidArgument;
        }
        let facts = match read_hybrid_facts(unsafe { &*facts }) {
            Ok(facts) => facts,
            Err(status) => return status,
        };
        let cuda = read_qwen35_cuda_facts(unsafe { &*cuda });
        let class = match class {
            0 => FireClass::Decode,
            1 => FireClass::Prefill,
            2 => FireClass::CommitAdvance,
            3 => FireClass::StateOnly,
            4 => FireClass::FrozenVerify,
            _ => return PieForwardStatus::InvalidArgument,
        };
        let plan = crate::family::qwen3_5_hybrid_cuda(&facts, &cuda, class);
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
static KEEP_ALIVE: [EntryAddr; 8] = [
    EntryAddr(pie_forward_trace_llama_like as *const ()),
    EntryAddr(pie_forward_trace_llama_like_cuda as *const ()),
    EntryAddr(pie_forward_trace_qwen3_5_moe_mlp as *const ()),
    EntryAddr(pie_forward_trace_qwen3_5_gdn as *const ()),
    EntryAddr(pie_forward_trace_qwen3_5_full_attn as *const ()),
    EntryAddr(pie_forward_trace_qwen3_5_hybrid as *const ()),
    EntryAddr(pie_forward_trace_qwen3_5_hybrid_cuda as *const ()),
    EntryAddr(pie_forward_release as *const ()),
];

/// Lower a traced plan over one fire's rows — the SHADOW comparison's
/// Rust half (`.wiki/tart/dsl.md` migration step 6).
///
/// The driver walks a nested region IR; `lower` produces the flat launch
/// list that is meant to replace it. Calling both on the same fire and
/// comparing is how the replacement earns the right to happen, and this
/// is the entry point for the comparing.
///
/// It EXECUTES NOTHING. The result is a description of what would run.
///
/// `*out` points into storage the plan owns, valid until the next call
/// on the SAME plan (one slot). Copy or compare before calling again; do
/// not free it — `pie_forward_release` does, with the plan.
///
/// # Safety
///
/// `plan` is null or points at a writable header built by one of the
/// trace entry points and not yet released; `rows` is null or points at
/// `rows_len` readable [`PieForwardRow`]s; `out` is null or a writable
/// slot.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pie_forward_lower(
    plan: *mut PieForwardPlan,
    rows: *const PieForwardRow,
    rows_len: usize,
    out: *mut PieForwardLowered,
) -> PieForwardStatus {
    abort_on_panic(|| {
        if out.is_null() {
            return PieForwardStatus::InvalidArgument;
        }
        unsafe { *out = PieForwardLowered::default() };
        if plan.is_null() || (rows.is_null() && rows_len != 0) {
            return PieForwardStatus::InvalidArgument;
        }
        let wire = if rows_len == 0 {
            &[][..]
        } else {
            unsafe { std::slice::from_raw_parts(rows, rows_len) }
        };
        let rows: Vec<crate::lower::Row> = wire
            .iter()
            .map(|r| crate::lower::Row {
                multi_token: r.multi_token != 0,
                custom_mask: r.custom_mask != 0,
                hooked: r.hooked != 0,
                lora: r.lora != 0,
                depth_k: (r.depth_k >= 0).then_some(r.depth_k as u32),
                write_desc: r.write_desc != 0,
                wants_scores: r.wants_scores != 0,
                samples: r.samples != 0,
            })
            .collect();
        unsafe { *out = arena::lower(&mut *plan, &rows) };
        PieForwardStatus::Ok
    })
}
