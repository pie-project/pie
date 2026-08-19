//! `qwen3_5` — the hybrid: GDN layers, full-attention layers, MoE or dense
//! MLP, composed by a static layer schedule.
//!
//! Three fragments trace standalone ([`qwen3_5_moe_mlp_block`],
//! [`qwen3_5_gdn_block`], [`qwen3_5_full_attn_block`]) and
//! [`qwen3_5_hybrid`] composes their bodies per layer, so a fragment's test
//! and the whole model's test read the same ops.

pub mod facts;
pub mod metal;

/// The MoE aligned path's block size and block ceiling, as the driver picks
/// them (`kernels::moe::moe_aligned_block`, `kMoeAlignedBlockMin/Max`).
///
/// Load-time constants, so a declaration may state them: the driver's own
/// choice varies with the route count, and the MINIMUM is what a declaration
/// must assume -- it yields the most blocks, so a plan sized against it fits
/// whatever the driver picks.
const MOE_ALIGNED_BLOCK: u32 = 16;
const MOE_MAX_BLOCKS: u32 = 1024;

use self::facts::{
    Qwen35CudaFacts, Qwen35FullAttnFacts, Qwen35GdnFacts, Qwen35HybridFacts, Qwen35MlpKind,
    Qwen35MoeMlpFacts,
};
use model_dsl::{
    self as dsl, ConvW, GdnPrepW, Kv, MatW, NormW, Rs, Trace, Val, WeightRepr, attention,
    causal_conv1d, cuda, gated_delta, gdn_prep, matmul, matmul_per_token, rmsnorm, rmsnorm_gated,
    rope_partial, sigmoid_gate_add, sigmoid_gate_mul, split_gdn, split_q_gate, split_qkv, swiglu,
    topk, weighted_sum,
};
use model_ir::trace::{
    DType, Dim, FireClass, ForwardPlan, GuardPred, NormVariant, RopeKind, Shape,
};

/// One qwen3_5_moe MoE MLP block, traced standalone — the first `dyn`
/// fragment.
///
/// This is a FRAGMENT, not a model: the unit the future qwen3_5 declaration
/// composes per layer (`y += moe_mlp(l, rmsnorm(y, mlp_norm))`), traced
/// against layer 0 with the residual stream as a fragment parameter
/// ([`dsl::input`]). A full qwen3_5 declaration also needs the
/// hybrid GDN attention vocabulary (`causal_conv1d`, `gated_delta`, gated
/// rmsnorm, per-request recurrent state) — out of scope here, a separate
/// rung; see [`Qwen35MoeMlpFacts`].
///
/// Mirrors `qwen3_5_moe_forward.cpp::run_moe_mlp` launch for launch, in the
/// decode fast path's one-launch-per-op form (the canonical granularity;
/// the prefill path's host-routed per-expert gather/GEMM/scatter loop is a
/// LOWERING of ops 4–7, as is the CUTLASS fused pipeline — both the
/// emitter's per-fire choice):
///
/// | trace op                       | hand-written kernel(s)                      |
/// |--------------------------------|---------------------------------------------|
/// | Rmsnorm(mlp_norm)              | kernels::norm::rmsnorm_gemma_bf16                   |
/// | Matmul(router)                 | kernels::gemm::act_x_wt_bf16 (router logits)     |
/// | TopK                           | kernels::moe::topk_softmax_bf16                    |
/// | Matmul(expert.{e}.gate_up, sel)| grouped GEMM (batched/aligned/CUTLASS)      |
/// | Swiglu                         | kernels::mlp::chunked_swiglu_bf16 over N*k rows    |
/// | Matmul(expert.{e}.down, sel)   | grouped GEMM (batched/aligned/CUTLASS)      |
/// | WeightedSum                    | kernels::moe::token_batched_weighted_sum_bf16      |
/// | Matmul(shared_expert.gate_up)  | kernels::gemm::act_x_w                           |
/// | Swiglu                         | kernels::mlp::chunked_swiglu_bf16                  |
/// | Matmul(shared_expert.down)     | kernels::gemm::act_x_w                           |
/// | Matmul(shared_expert_gate)     | kernels::gemm::act_x_w ([Tokens, 1] logit)       |
/// | SigmoidGateAdd                 | kernels::mlp::sigmoid_scalar_gate_add_bf16         |
/// | ResidualAdd                    | kernels::norm::residual_add_bf16                    |
///
/// The five shared-expert ops fold away when the facts say the checkpoint
/// has none (`shared_expert_intermediate == 0`, the qwen3_moe shape), the
/// same way llama_like's branches fold: at trace time, leaving no trace.
pub fn qwen3_5_moe_mlp_block(facts: &Qwen35MoeMlpFacts) -> ForwardPlan {
    dsl::trace_named("qwen3_5_moe_mlp_block", |t| {
        // The fragment's parameter: the residual stream entering the block.
        let y = dsl::input(t, facts.hidden);
        moe_mlp_body(0, facts, &y);
    })
}

/// The MoE MLP block's weight namespace at layer `l`: the router, the
/// `{e}`-templated expert banks, and the shared expert's three handles —
/// eager strings, [`model_dsl::Layer`]-style, so a checkpoint without a
/// shared expert simply never reads them.
struct MoeLayerW {
    mlp_norm: NormW,
    router: MatW,
    expert_gate_up: MatW,
    expert_down: MatW,
    shared_gate_up: MatW,
    shared_down: MatW,
    shared_gate: MatW,
}

impl MoeLayerW {
    /// `repr` reaches ONE handle — the shared expert's `down`, which is
    /// the only projection in this block the driver ever carried a quant
    /// descriptor for (`shared_down_proj_quant`). The router is a tiny
    /// dense GEMM, the expert BANKS are addressed through pointer arrays
    /// by their own kernels, and the shared `gate_up` is a packed join.
    /// The semantic text passes `Bf16`.
    fn new(l: u32, f: &Qwen35MoeMlpFacts, repr: WeightRepr) -> Self {
        let w = |name: &str| format!("layer.{l}.{name}");
        let mat = |name: &str, width: u32| MatW {
            name: w(name),
            width,
            layer: Some(l),
            repr: WeightRepr::Bf16,
        };
        MoeLayerW {
            mlp_norm: NormW {
                name: w("mlp_norm"),
                variant: f.norm_variant,
                per_head: None,
                layer: Some(l),
            },
            router: mat("router", f.num_experts),
            expert_gate_up: mat("expert.{e}.gate_up", 2 * f.moe_intermediate),
            expert_down: mat("expert.{e}.down", f.hidden),
            shared_gate_up: mat("shared_expert.gate_up", 2 * f.shared_expert_intermediate),
            shared_down: mat("shared_expert.down", f.hidden).with_repr(repr),
            shared_gate: mat("shared_expert_gate", 1),
        }
    }
}

/// The MoE MLP block's op emission at layer `l` — the unit
/// [`qwen3_5_moe_mlp_block`] traces standalone (at layer 0) and
/// [`qwen3_5_hybrid`] composes per layer. One body so the hybrid's MLP ops
/// ARE the fragment's, by construction rather than by parallel maintenance.
/// The MoE block's ALIGNED CUDA reading — the leg every fire outside the
/// fused CUTLASS bound actually takes.
///
/// # The extent that used to make this unstatable
///
/// The aligned path buckets every (token, expert) route by expert and pads
/// each bucket to a whole block, so one batched GEMM covers all experts. Its
/// intermediates are therefore
/// `ceil((N·k + min(E, N·k)·(block-1)) / block) · block` rows tall -- not
/// `Tokens`, not a `Const`, and the north-star doc named exactly this as "an
/// extent no `Dim` spells". [`Dim::MoeAlignedRoutes`] spells it: every input
/// but `N` is load-time, so the extent is a function of the fire's own token
/// count, which is what a symbolic dim has to be.
///
/// # What it states
///
/// The permutation, the gather into block-major order, the pointer arrays,
/// the two grouped GEMMs with the activation between them, and the reorder
/// back to route order. Then the combine -- and WHICH combine is a binding,
/// so the text states it: a deployment that folds the residual takes the
/// token-batched aligned form, one that does not takes the per-expert
/// scatter-add.
fn moe_mlp_body_aligned_cuda(l: u32, facts: &Qwen35MoeMlpFacts, y: &Val, repr: WeightRepr) -> Val {
    let w = MoeLayerW::new(l, facts, repr);
    let y = y.clone();
    let m = dsl::cuda::rmsnorm(&y, &w.mlp_norm);

    let aligned = model_ir::trace::Dim::MoeAlignedRoutes {
        top_k: facts.top_k,
        experts: facts.num_experts,
        block: MOE_ALIGNED_BLOCK,
    };

    let logits = matmul(&m, &w.router);
    let (experts, weights) = dsl::cuda::topk(&logits, facts.top_k);

    // The permutation, and the three arrays it produces: the sorted route
    // order, which expert each block belongs to, and the inverse map the
    // reorder reads back.
    let (sorted, expert_ids, _inverse) = dsl::cuda::moe_align(
        &experts,
        MOE_MAX_BLOCKS,
        MOE_ALIGNED_BLOCK,
        facts.top_k,
        facts.num_experts,
    );
    let aligned_in =
        dsl::cuda::gather_moe_aligned_inputs(&m, &sorted, aligned, facts.hidden, facts.top_k);
    // The pointer build DECLARES the aligned staging, because it bakes
    // those buffers' base addresses into the device pointer arrays the
    // batched-cuBLAS fallback dereferences. Everything below fills a
    // buffer it named -- each takes its destination as an operand and
    // writes it in place -- which is what the executor was standing in
    // for with `mw.aligned_gate_up` / `_act` / `_out`.
    let (gu_stage, act_stage, out_stage) = dsl::cuda::build_moe_ptrs_aligned(
        &expert_ids,
        &aligned_in,
        l,
        &w.expert_gate_up.name,
        &w.expert_down.name,
        aligned,
        facts.hidden,
        facts.moe_intermediate,
    );

    // Both projections are the SAME statement -- a grouped GEMM over the
    // block-major operand -- which is why the selector matmul lowers to one
    // kernel rather than to a per-expert loop. The second operand is
    // `expert_ids`, which is what the kernel indexes the bank by; it used
    // to be `sorted`, which the kernel never reads.
    let gate_up = dsl::cuda::moe_grouped_gemm(
        &aligned_in,
        &expert_ids,
        &gu_stage,
        aligned,
        2 * facts.moe_intermediate,
        &w.expert_gate_up.name,
        MOE_ALIGNED_BLOCK,
        MOE_MAX_BLOCKS,
    );
    let act = dsl::cuda::swiglu_aligned(&gate_up, &act_stage, aligned, facts.moe_intermediate);
    let down = dsl::cuda::moe_grouped_gemm(
        &act,
        &expert_ids,
        &out_stage,
        aligned,
        facts.hidden,
        &w.expert_down.name,
        MOE_ALIGNED_BLOCK,
        MOE_MAX_BLOCKS,
    );

    let route_out =
        dsl::cuda::reorder_moe_aligned_output(&down, &sorted, facts.top_k, facts.hidden);

    // The combine, in the form the aligned leg actually fires: the reorder
    // above already put the rows back in ROUTE order, so what follows is the
    // ordinary token-batched sum, not the aligned one. Read off
    // `qwen3_5_moe_forward.cpp`'s aligned block rather than inferred from the
    // name -- `_aligned_` names a kernel that reads block-major rows, and
    // this one no longer has any.
    //
    // The residual FOLDS into it. At tp=1 the aligned leg is reached only
    // through the decode fast path, and there `add_to_residual` is set, so
    // `moe_out` IS the residual stream: the combine fires the `_add_` form
    // straight onto `y` and the shared expert's gate lands on top of it.
    // There is no trailing add. An earlier reading of this block stated the
    // plain combine and a closing `y += combined`, which is the tp>1 /
    // general-path shape -- one this leg never takes.
    let routed = dsl::cuda::weighted_sum(&weights, &route_out, facts.hidden, Some(&y));

    if facts.shared_expert_intermediate > 0 {
        let inter = facts.shared_expert_intermediate;
        let act = dsl::cuda::swiglu(&matmul(&m, &w.shared_gate_up), inter);
        let shared = matmul(&act, &w.shared_down);
        dsl::cuda::sigmoid_dot_scalar_gate_add(&m, &w.shared_gate, &shared, &routed, facts.hidden)
    } else {
        routed
    }
}

fn moe_mlp_body(l: u32, facts: &Qwen35MoeMlpFacts, y: &Val) -> Val {
    // Semantic: no backend, so no scaled kernel to name.
    let w = MoeLayerW::new(l, facts, WeightRepr::Bf16);
    let mut y = y.clone();

    let m = rmsnorm(&y, &w.mlp_norm);

    // Routed experts: router -> topk -> grouped gate_up -> swiglu ->
    // grouped down -> per-token weighted combine.
    let logits = matmul(&m, &w.router);
    let (experts, weights) = topk(&logits, facts.top_k);
    let gate_up = matmul_per_token(&m, &w.expert_gate_up, &experts);
    let act = swiglu(&gate_up, facts.moe_intermediate);
    let down = matmul_per_token(&act, &w.expert_down, &experts);
    let routed = weighted_sum(&weights, &down);

    // Shared expert (qwen3.5/3.6-MoE: always-on dense MLP behind a
    // per-token sigmoid scalar gate; absent on qwen3_moe).
    let combined = if facts.shared_expert_intermediate > 0 {
        let inter = facts.shared_expert_intermediate;
        let act = swiglu(&matmul(&m, &w.shared_gate_up), inter);
        let shared = matmul(&act, &w.shared_down);
        let gate = matmul(&m, &w.shared_gate);
        sigmoid_gate_add(&shared, &gate, &routed)
    } else {
        routed
    };

    // Not a fresh matmul, so `+=` records the explicit ResidualAdd.
    y += combined;
    y
}

/// The MoE MLP fragment's CUDA reading, traced standalone at layer 0 —
/// [`qwen3_5_moe_mlp_block`]'s peer, and the only place the MoE block's
/// stated form is pinned on its own.
pub fn qwen3_5_moe_mlp_block_cuda(
    facts: &Qwen35MoeMlpFacts,
    cuda: &Qwen35CudaFacts,
) -> ForwardPlan {
    dsl::trace_named("qwen3_5_moe_mlp_block.cuda.decode", |t| {
        let y = dsl::input(t, facts.hidden);
        moe_mlp_body_cuda(0, facts, cuda, &y, FireClass::Decode);
    })
}

/// The MoE MLP block's CUDA reading — [`moe_mlp_body`]'s peer, naming
/// the kernels the hand-written pass fires instead of leaving the
/// selector and the combine opaque.
///
/// # Which leg this states, and why only one
///
/// `run_moe_mlp` reaches the same numbers four ways. Three of them are
/// not rectangles:
///
/// - the ALIGNED/grouped leg pads routes into blocks, giving its
///   intermediates `ceil((N*k + min(E, N*k)*(block-1)) / block) * block`
///   rows — an extent no [`model_ir::trace::Dim`] spells;
/// - the decode GEMV leg is a rectangle, but the aligned block size is
///   8 or 16 and never 1, so the aligned leg always exists and the GEMV
///   arm covers only `N * k < 64` (N <= 7 at top_k 8);
/// - the HOST-routed general path reads the router back to the CPU and
///   issues one gather/GEMM/scatter per expert, so its launch COUNT is a
///   device-derived number.
///
/// The fused CUTLASS call is the fourth and the one decode actually
/// takes: permute, both grouped GEMMs, the activation and the weighted
/// finalize in ONE call producing `[Tokens, hidden]`. Its `bool` return
/// is decided before the fire (see [`dsl::cuda::moe_fused_cutlass`]), so
/// the leg is a fact plus a row bound.
///
/// Fires outside that bound do not get a guarded arm — a guard whose
/// other arm cannot be stated refuses the whole plan. They DECLINE, the
/// llama_like way: the plan states one rectangle and the driver's
/// eligibility sends the rest to the hand-written path.
///
/// Everything this body refuses returns [`moe_mlp_body`] unchanged, so
/// the refusal shows up where every other refusal does — as residue in
/// the coverage ledger, naming its own cause.
fn moe_mlp_body_cuda(
    l: u32,
    facts: &Qwen35MoeMlpFacts,
    cuda: &Qwen35CudaFacts,
    y: &Val,
    class: FireClass,
) -> Val {
    // The fused leg is the decode fast path's. Prefill and the service
    // classes take the host-routed path, as do a streamed expert cache
    // (no fused slab to stride) and the force-general env; and a
    // deployment that sized no CUTLASS workspace has no fused leg at
    // all. tp>1 writes to scratch and follows with an allreduce, which
    // is a different shape than the one stated here.
    if class != FireClass::Decode
        || cuda.moe_cutlass_max_rows == 0
        || cuda.moe_streamed_experts
        || cuda.moe_force_general
        || !cuda.moe_residual_fold
        || (facts.shared_expert_intermediate > 0 && !cuda.moe_shared_gate_dot)
    {
        return moe_mlp_body_aligned_cuda(l, facts, y, cuda.proj_repr);
    }

    let w = MoeLayerW::new(l, facts, cuda.proj_repr);
    // Semantic: the lowering reads the variant and names the fold's
    // kernel, so there is nothing here for a CUDA reading to add.
    let m = dsl::cuda::rmsnorm(y, &w.mlp_norm);

    // The router stays two ops — a plain GEMM for the logits, then the
    // fused top-k/softmax/renormalize — because the fused call takes the
    // routing as operands rather than computing it.
    let logits = matmul(&m, &w.router);
    let (experts, weights) = dsl::cuda::topk(&logits, facts.top_k);
    let routed = dsl::cuda::moe_fused_cutlass(
        &m,
        &experts,
        &weights,
        &w.expert_gate_up,
        &w.expert_down,
        facts.hidden,
    );

    // The fused runner overwrites its output, so a folded residual costs
    // a separate add — still one launch, and it is why the CUDA reading
    // has no ResidualAdd at the end where the semantic body does.
    let y = dsl::cuda::residual_add(&routed, y, facts.hidden);

    if facts.shared_expert_intermediate == 0 {
        return y;
    }

    // The shared expert is dense: two cuBLAS GEMMs around the chunked
    // activation, then the landing accumulates into the stream the
    // routed block already wrote.
    let inter = facts.shared_expert_intermediate;
    let act = dsl::cuda::swiglu(&matmul(&m, &w.shared_gate_up), inter);
    let shared = matmul(&act, &w.shared_down);
    dsl::cuda::sigmoid_dot_scalar_gate_add(&m, &w.shared_gate, &shared, &y, facts.hidden)
}

/// One qwen3_5 GDN (gated-deltanet) linear-attention block, traced
/// standalone — the second fragment, and the other layer kind of the
/// qwen3.5 hybrid.
///
/// This is a FRAGMENT, not a model: the unit the qwen3_5 declaration
/// composes on a `Linear` layer (`y += gdn(l, dsl::cuda::rmsnorm(y, attn_norm))`,
/// plan.md Part 1's `match layers[l]`), traced against layer 0 with the
/// residual stream as a fragment parameter ([`dsl::input`]),
/// exactly the MoE fragment's shape. The FULL-attention layer kind of this
/// family — not llama_like's: q_proj 2× wide with the per-head
/// `[query | gate]` split, sigmoid output gate, partial rope, Gemma-fold
/// per-head norms — is its own fragment, [`qwen3_5_full_attn_block`], and
/// [`qwen3_5_hybrid`] composes all three bodies into the full model.
///
/// Mirrors `qwen3_5_forward.cpp::linear_attn_layer_body` launch for launch
/// on the TP=1 decode fast path (the canonical granularity; the prefill
/// conv/recurrence walks, the batched slot-indirected variants, the
/// warp-tiled/cached/FLA recurrence kernels and the GQA
/// `repeat_interleave` materialization are all LOWERINGS of ops 5–7, the
/// emitter's per-fire choice — as are the verify-stash and rs-buffer
/// scatter/gather paths, which are speculative-decode services around the
/// same ops, not ops of the pass):
///
/// | trace op                | hand-written kernel(s)                          |
/// |-------------------------|--------------------------------------------------|
/// | Rmsnorm(attn_norm)      | kernels::norm::rmsnorm_gemma_bf16                        |
/// | Matmul(in_proj_qkv)     | kernels::gemm::act_x_w                                |
/// | Matmul(in_proj_z)       | kernels::gemm::act_x_w                                |
/// | Matmul(in_proj_a)       | kernels::gemm::act_x_w                                |
/// | Matmul(in_proj_b)       | kernels::gemm::act_x_w                                |
/// | CausalConv1d            | launch_causal_conv1d_update[_batched]_bf16       |
/// | GdnPrep                 | kernels::ssm::qwen_gdn_post_conv_prep_bf16              |
/// | GatedDelta              | launch_recurrent_gated_delta_step_* (decode)     |
/// | RmsnormGated            | kernels::norm::rmsnorm_gated_fp32_in_bf16                |
/// | Matmul(o_proj)+res      | kernels::gemm::act_x_w beta=1                         |
///
/// With the fused binding (`fused_in_proj`, `PIE_QWEN35_FUSED_GDN_PROJ`)
/// the four projections become two matmuls + two [`SplitGdn`] launches
/// (`kernels::layout::split_bf16_rows`, `kernels::layout::split_qwen_gdn_ba_bf16`) — same op
/// count, different ops, resolved at trace time like llama_like's
/// `fused_qkv`.
///
/// `CausalConv1d` and `GatedDelta` address the layer's PER-REQUEST
/// conv/recurrent state — implicit, marked by the op kinds themselves
/// ([`model_ir::trace::OpKind::state_ref`]); see the trace module doc's "the
/// per-request state axis" for why the state is not a traced value.
///
/// [`SplitGdn`]: model_ir::trace::OpKind::SplitGdn
pub fn qwen3_5_gdn_block(facts: &Qwen35GdnFacts) -> ForwardPlan {
    dsl::trace_named("qwen3_5_gdn_block", |t| {
        // The fragment's parameter: the residual stream entering the block.
        let y = dsl::input(t, facts.hidden);
        gdn_attn_body(t, 0, facts, &y);
    })
}

/// The GDN block's weight namespace at layer `l`: both in-projection
/// bindings (fused and unfused — eager strings, only the traced branch
/// reads its handles), the conv/prep weight pairs, and the layer's
/// per-request recurrent state.
struct GdnLayerW {
    attn_norm: NormW,
    in_proj_qkvz: MatW,
    in_proj_ba: MatW,
    in_proj_qkv: MatW,
    in_proj_z: MatW,
    in_proj_a: MatW,
    in_proj_b: MatW,
    conv: ConvW,
    prep: GdnPrepW,
    gate_norm: NormW,
    o_proj: MatW,
    rs: Rs,
}

impl GdnLayerW {
    fn new(t: &Trace, l: u32, f: &Qwen35GdnFacts) -> Self {
        let conv_dim = f.conv_dim();
        let v_dim = f.value_width();
        let w = |name: &str| format!("layer.{l}.{name}");
        let mat = |name: &str, width: u32| MatW {
            name: w(name),
            width,
            layer: Some(l),
            repr: WeightRepr::Bf16,
        };
        GdnLayerW {
            attn_norm: NormW {
                name: w("attn_norm"),
                variant: f.norm_variant,
                per_head: None,
                layer: Some(l),
            },
            in_proj_qkvz: mat("in_proj_qkvz", conv_dim + v_dim),
            in_proj_ba: mat("in_proj_ba", 2 * f.value_heads),
            in_proj_qkv: mat("in_proj_qkv", conv_dim),
            in_proj_z: mat("in_proj_z", v_dim),
            in_proj_a: mat("in_proj_a", f.value_heads),
            in_proj_b: mat("in_proj_b", f.value_heads),
            conv: ConvW {
                name: w("conv"),
                // `Qwen3NextGatedDeltaNet` builds its conv with `bias=False`.
                bias: None,
                kernel: f.conv_kernel,
                layer: l,
            },
            prep: GdnPrepW {
                a_log: w("a_log"),
                dt_bias: w("dt_bias"),
                layer: l,
            },
            // The gated norm's fold is Plain by construction (the op
            // carries no variant); the handle contributes name and layer.
            gate_norm: NormW {
                name: w("gate_norm"),
                variant: NormVariant::Plain,
                per_head: None,
                layer: Some(l),
            },
            o_proj: mat("o_proj", f.hidden),
            rs: Rs::at(t, l),
        }
    }
}

/// The GDN linear-attention block's op emission at layer `l` — the unit
/// [`qwen3_5_gdn_block`] traces standalone (at layer 0) and
/// [`qwen3_5_hybrid`] composes on every `Linear` layer. One body so the
/// hybrid's GDN ops ARE the fragment's, by construction.
///
/// ONLY the kernel CHOICES lower under `Some(lower)`: the conv (decode
/// update vs prefill walk) and the recurrence (the decode step's four
/// name variants; the prefill three-way behind the first value-producing
/// guard chain). Everything else — the norms, the in-projections and
/// their fused/unfused splits, `gdn_prep`, the gated norm, the o_proj
/// fold — is a 1:1-kernel semantic op and stays semantic in every form.
/// The GDN in-projections against `w`'s layer: the fused/unfused branch
/// resolves at trace time (a binding fact); operand packing mirrors the
/// driver's: qkvz = [mixed_qkv | z], ba = [b | a]. Returns
/// `(qkv, z, a, b)`. One function so the CommitAdvance pass's no-stash
/// arm (the retired commit-advance pass) runs EXACTLY the normal body's GEMMs
/// and splits, by construction rather than by parallel maintenance.
fn gdn_in_proj(
    x: &Val,
    w: &GdnLayerW,
    facts: &Qwen35GdnFacts,
    stated: bool,
) -> (Val, Val, Val, Val) {
    if facts.fused_in_proj {
        // `stated`: the two splits NAME their kernels rather than
        // leaving a semantic `SplitGdn` whose widths the executor
        // compared against `conv_dim` / `V_dim` / `V_h` to pick one.
        //
        // A row split and an INTERLEAVED b/a split are different
        // arithmetic over the same shapes, so telling them apart by
        // extent was a kernel choice made from a coincidence of
        // numbers — and one a family whose `V_h` ever equalled its
        // `V_dim` would get wrong. Two symbols, no comparison.
        let qkvz = matmul(x, &w.in_proj_qkvz);
        let (qkv, z) = if stated {
            dsl::cuda::split_rows(&qkvz, facts.conv_dim(), facts.value_width())
        } else {
            split_gdn(&qkvz, facts.conv_dim(), facts.value_width())
        };
        let ba = matmul(x, &w.in_proj_ba);
        let (b, a) = if stated {
            dsl::cuda::split_qwen_gdn_ba(&ba, facts.value_heads)
        } else {
            split_gdn(&ba, facts.value_heads, facts.value_heads)
        };
        (qkv, z, a, b)
    } else {
        (
            matmul(x, &w.in_proj_qkv),
            matmul(x, &w.in_proj_z),
            matmul(x, &w.in_proj_a),
            matmul(x, &w.in_proj_b),
        )
    }
}

fn gdn_attn_body(t: &Trace, l: u32, facts: &Qwen35GdnFacts, y: &Val) -> Val {
    let w = GdnLayerW::new(t, l, facts);
    let mut y = y.clone();

    let x = rmsnorm(&y, &w.attn_norm);

    let (qkv, z, a, b) = gdn_in_proj(&x, &w, facts, /*stated=*/ false);

    // Conv → prep → recurrence: the GDN core, against the layer's
    // per-request conv/recurrent state. Both stay opaque here — the
    // consumer owns kernel choice.
    let qkv = causal_conv1d(&qkv, &w.conv);
    let (q, k, v, g, beta) = gdn_prep(
        &qkv,
        &a,
        &b,
        &w.prep,
        facts.key_heads,
        facts.key_head_dim,
        facts.value_heads,
        facts.value_head_dim,
    );
    let core = gated_delta(&w.rs, &q, &k, &v, &g, &beta);

    // Gated norm (z-gated, per-head, plain fold) → o_proj landed on
    // the residual (`+=` of a fresh matmul IS the beta=1 fold).
    let o = rmsnorm_gated(&core, &z, &w.gate_norm);
    y += matmul(&o, &w.o_proj);
    y
}

/// The GDN block's CUDA text — [`gdn_attn_body`]'s kernel-stating
/// counterpart, one per non-CommitAdvance [`FireClass`]. Only the conv
/// and the recurrence differ by class; everything else states the same
/// 1:1-kernel ops the semantic text does.
fn gdn_attn_body_cuda(
    t: &Trace,
    l: u32,
    facts: &Qwen35GdnFacts,
    y: &Val,
    c: &Qwen35CudaFacts,
    class: FireClass,
) -> Val {
    let w = GdnLayerW::new(t, l, facts);
    let mut y = y.clone();

    let x = dsl::cuda::rmsnorm(&y, &w.attn_norm);

    let (qkv, z, a, b) = gdn_in_proj(&x, &w, facts, /*stated=*/ true);

    // THE VERIFY STASH IS GONE (`.wiki/driver/graph.md` §4.2). It cached
    // the in-proj activations so a later commit-advance pass could replay
    // them — but a speculative decode buffers its own tokens and folds
    // only the accepted prefix, so the BUFFER is the stash and there is
    // no replay to feed.

    // Conv → prep → recurrence: the GDN core, against the layer's
    // per-request conv/recurrent state.
    // StateOnly is prefill-shaped throughout the backbone — it takes the
    // Prefill arm in every kernel choice here; only the model epilogue
    // differs, and that class match lives in `qwen3_5_hybrid_cuda_text`.
    // CommitAdvance never enters this body at all: it is its own pass
    // (the retired commit-advance pass), not a variant of the layer loop.
    let qkv = match class {
        FireClass::Decode => cuda::gdn_conv_update_batched(&qkv, &w.conv, &w.rs),
        FireClass::Prefill => cuda::gdn_conv_prefill_batched(&qkv, &w.conv, &w.rs),
    };
    let (q, k, v, g, beta) = gdn_prep(
        &qkv,
        &a,
        &b,
        &w.prep,
        facts.key_heads,
        facts.key_head_dim,
        facts.value_heads,
        facts.value_head_dim,
    );
    // The OnAttnProj site (A4): the hand-written GDN body invokes the
    // fire's programs here observing q_pre (fp32; qwen3_5's sites are
    // OBSERVATION-only — no page-mask sink, no score capture). Lowered
    // traces only; a fire with nothing attached passes by argument.
    // (The hand-written invoke sits after the cached family's GQA
    // repeats; the repeats read q_pre and never write it, so observing
    // before the recurrence guard sees the same bytes.)
    dsl::seam(q.trace(), &dsl::seam::ATTN_Q, &[&q], Some(l));
    // GQA (value heads sharing fewer key heads) picks the `_gqa` decode
    // step; the prefill kernels state their own layout handling.
    let gqa = facts.value_heads != facts.key_heads;
    let core = match class {
        FireClass::Decode => {
            cuda::gdn_step_batched(&q, &k, &v, &g, &beta, &w.rs, gqa, c.state_bf16)
        }
        FireClass::Prefill => {
            // The prefill recurrence three-way, as the first
            // VALUE-PRODUCING guard chain (north-star-dsl.md 4b): the
            // guard's output is the recurrence core — the same
            // `[Tokens, Vh, Vd]` f32 the semantic `gated_delta`
            // produces — and each arm's launch binds that buffer,
            // recording no SSA outputs of its own. Arm order is the
            // hand-written probe order: warp-tiled (when eligible at
            // all — a fact), then the cached family (whose kernels
            // index the REPEATED head layout, so the GQA repeats
            // materialize INSIDE its arm and nowhere else — launch
            // order matches the hand-written stream order: prep,
            // [repeats], recurrence), else the batched GQA-aware FLA.
            let out_shape = (
                Shape(vec![
                    Dim::Tokens,
                    Dim::Const(facts.value_heads),
                    Dim::Const(facts.value_head_dim),
                ]),
                DType::F32,
            );
            // A CONDITIONAL arm, which is why `regions` takes `&mut`
            // rather than a builder chain: whether the warp-tiled leg
            // exists at all is a deployment fact, and a chain would have
            // to rebind itself to say so.
            dsl::regions(
                t,
                Some(l),
                Some(out_shape),
                |ctx| {
                    if c.warp_tiled {
                        ctx.arm(
                            dsl::Region::Fire(GuardPred::TokensLE(c.warp_tiled_max)),
                            || {
                                cuda::gdn_prefill_warp_tiled(
                                    &q,
                                    &k,
                                    &v,
                                    &g,
                                    &beta,
                                    &w.rs,
                                    c.state_bf16,
                                );
                            },
                        );
                    }
                    ctx.arm(dsl::Region::Fire(GuardPred::TokensLE(c.cached_max)), || {
                        // The repeats declare their results now, so the
                        // recurrence in this arm takes THEM — which is
                        // what the arm always meant and could not say
                        // while the repeat was output-less.
                        if gqa {
                            let qr = cuda::repeat_interleave_heads(
                                &q,
                                facts.value_heads,
                                facts.key_head_dim,
                            );
                            let kr = cuda::repeat_interleave_heads(
                                &k,
                                facts.value_heads,
                                facts.key_head_dim,
                            );
                            cuda::gdn_prefill_cached(&qr, &kr, &v, &g, &beta, &w.rs, c.state_bf16);
                        } else {
                            cuda::gdn_prefill_cached(&q, &k, &v, &g, &beta, &w.rs, c.state_bf16);
                        }
                    });
                },
                || {
                    cuda::gdn_prefill_fla(&q, &k, &v, &g, &beta, &w.rs, c.state_bf16);
                },
            )
            .expect("the guarded recurrence produces its value")
        }
    };
    // The OnAttn site: after the recurrence core, before the gated norm
    // — the hand-written invoke's position (observing q_pre again).
    dsl::seam(q.trace(), &dsl::seam::ATTN_OUT, &[&q], Some(l));

    // Gated norm (z-gated, per-head, plain fold) → o_proj landed on
    // the residual (`+=` of a fresh matmul IS the beta=1 fold).
    let o = rmsnorm_gated(&core, &z, &w.gate_norm);
    y += matmul(&o, &w.o_proj);
    y
}

/// One qwen3_5 FULL-attention block, traced standalone — the third
/// fragment, and the last layer kind the qwen3.5 hybrid needed.
///
/// This is a FRAGMENT, not a model: the unit [`qwen3_5_hybrid`] composes on
/// a `Full` layer (plan.md Part 1's `match layers[l] { Full => full_attn(l,
/// x), .. }`), traced against layer 0 with the residual stream as a
/// fragment parameter ([`dsl::input`]), exactly the MoE and GDN
/// fragments' shape.
///
/// Mirrors `qwen3_5_forward.cpp::full_attn_layer_body` launch for launch on
/// the TP=1 path (the canonical granularity; decode vs prefill vs
/// small-naive attention plans, the explicit KV-write descriptor branch and
/// the TP all-reduce/residual split are all LOWERINGS the emitter picks per
/// fire), on the default (unfused) binding:
///
/// | trace op                  | hand-written kernel(s)                       |
/// |---------------------------|-----------------------------------------------|
/// | Rmsnorm(attn_norm)        | kernels::norm::rmsnorm_gemma_bf16                     |
/// | Matmul(q_proj) [2×-wide]  | kernels::gemm::act_x_w → [N, 2·Hq]                 |
/// | Matmul(k_proj)            | kernels::gemm::act_x_w → [N, Hk]                   |
/// | Matmul(v_proj)            | kernels::gemm::act_x_w → [N, Hk]                   |
/// | SplitQGate                | kernels::layout::split_q_gate_bf16 (per-head q‖gate)    |
/// | RmsnormPerHead(q, Gemma)  | kernels::norm::rmsnorm_gemma_bf16 over N·Hq rows of d |
/// | RmsnormPerHead(k, Gemma)  | kernels::norm::rmsnorm_gemma_bf16 over N·Hkv rows of d|
/// | Rope(partial)             | kernels::rope::rope_partial_bf16 (rotary_dim chans)   |
/// | KvAppend                  | kernels::attn::write_kv_to_pages / _explicit          |
/// | Attention                 | dispatch_attention_flashinfer_{decode,prefill}|
/// | SigmoidGateMul            | kernels::mlp::sigmoid_gate_inplace_bf16              |
/// | Matmul(o_proj)+res        | kernels::gemm::act_x_w beta=1                      |
///
/// With the fused binding (`fused_qkv`, `PIE_QWEN35_FUSED_FULL_ATTN_QGKV`)
/// the three projections become Matmul(qgkv) + [`SplitQkv`] whose "q" leg
/// is the 2×-wide `[query | gate]` bank (`use_fused_qgkv`:
/// `kernels::attn::split_qkv_bf16(packed, qg, k, v, N, 2*Hq, Hk)`) — the
/// [`SplitQGate`] de-interleave still follows, exactly as in the
/// hand-written body. `KvAppend`/`Attention` mark the layer's KV cache
/// ([`model_ir::trace::StateStore::KvCache`] via
/// [`model_ir::trace::OpKind::state_ref`]), the same marking llama_like
/// carries.
///
/// [`SplitQkv`]: model_ir::trace::OpKind::SplitQkv
/// [`SplitQGate`]: model_ir::trace::OpKind::SplitQGate
pub fn qwen3_5_full_attn_block(facts: &Qwen35FullAttnFacts) -> ForwardPlan {
    dsl::trace_named("qwen3_5_full_attn_block", |t| {
        // The fragment's parameter: the residual stream entering the block.
        let y = dsl::input(t, facts.hidden);
        full_attn_body(t, 0, facts, &y);
    })
}

/// The full-attention block's weight namespace at layer `l`: both
/// projection bindings (the fused `qgkv` bank and the unfused three), the
/// per-head qk-norm handles, and the layer's KV cache. The q handles are
/// 2× wide (per-head `[query | gate]`).
struct FullAttnLayerW {
    attn_norm: NormW,
    qgkv: MatW,
    q_proj: MatW,
    k_proj: MatW,
    v_proj: MatW,
    q_norm: NormW,
    k_norm: NormW,
    o_proj: MatW,
    kv: Kv,
}

impl FullAttnLayerW {
    /// `repr` is how the deployment STORES the four attention
    /// projections ([`Qwen35CudaFacts::proj_repr`]). The semantic text
    /// passes `Bf16` — a trace with no backend cannot name the kernel a
    /// scaled weight needs.
    ///
    /// It reaches q/k/v/o and NOT `qgkv`: the fused bank is the loader's
    /// dense join, and that contract declines quantized groups, so a
    /// quantized deployment binds the three separately. `mat` therefore
    /// stays the dense builder and the four say so.
    fn new(t: &Trace, l: u32, f: &Qwen35FullAttnFacts, repr: WeightRepr) -> Self {
        let q2_w = 2 * f.q_width();
        let kv_w = f.kv_width();
        let w = |name: &str| format!("layer.{l}.{name}");
        let mat = |name: &str, width: u32| MatW {
            name: w(name),
            width,
            layer: Some(l),
            repr: WeightRepr::Bf16,
        };
        let proj = |name: &str, width: u32| mat(name, width).with_repr(repr);
        // Per-head convention throughout this family — the weight knows,
        // so `rmsnorm(q, &w.q_norm)` needs no variant arguments.
        let qk_norm = |name: &str| NormW {
            name: w(name),
            variant: f.norm_variant,
            per_head: Some(f.head_dim),
            layer: Some(l),
        };
        FullAttnLayerW {
            attn_norm: NormW {
                name: w("attn_norm"),
                variant: f.norm_variant,
                per_head: None,
                layer: Some(l),
            },
            qgkv: mat("qgkv", q2_w + 2 * kv_w),
            q_proj: proj("q_proj", q2_w),
            k_proj: proj("k_proj", kv_w),
            v_proj: proj("v_proj", kv_w),
            q_norm: qk_norm("q_norm"),
            k_norm: qk_norm("k_norm"),
            o_proj: proj("o_proj", f.hidden),
            kv: Kv::at(t, l),
        }
    }
}

/// The full-attention block's op emission at layer `l` — the unit
/// [`qwen3_5_full_attn_block`] traces standalone (at layer 0) and
/// [`qwen3_5_hybrid`] composes on every `Full` layer.
///
/// `KvAppend`/`Attention` carry the MODEL layer `l`. The driver's compact
/// KV slot (`Qwen3_5LayerWeights::kv_layer`, assigned `kv_slot++` over the
/// full-attention layers in `qwen3_5.cpp::bind_qwen3_5`) is storage
/// knowledge derived from the layer-kind schedule — the count of full
/// layers before `l` — not a fact of what the pass computes, so the trace
/// states the layer and the emitter derives the slot, exactly as the GDN
/// ops state `l` while the driver keys its stash on the compact
/// `linear_idx`.
///
/// ONLY the kernel CHOICES lower under `Some(lower)`: the KV write (the
/// per-fire `HasWriteDesc` guard, both arms stated — llama_like's 4a
/// form) and the attention kernel (FlashInfer decode vs the planned
/// prefill dispatch). Everything else — the norms (incl. the Gemma
/// per-head pair), the projections and splits, the partial rope, the
/// sigmoid output gate, the o_proj fold — is a 1:1-kernel semantic op
/// and stays semantic in every form.
fn full_attn_body(t: &Trace, l: u32, facts: &Qwen35FullAttnFacts, y: &Val) -> Val {
    let w = FullAttnLayerW::new(t, l, facts, WeightRepr::Bf16);
    let mut y = y.clone();

    let x = rmsnorm(&y, &w.attn_norm);

    // Projections: q is 2× wide (per-head [query | gate]). The fused
    // binding packs [2q | k | v] into one bank (`qkv_proj.fused`, joined
    // behind PIE_QWEN35_FUSED_FULL_ATTN_QGKV); the split widths mirror the
    // driver's kernels::attn::split_qkv_bf16(N, 2*Hq, Hk).
    let (qg, k, v) = if facts.fused_qkv {
        split_qkv(&matmul(&x, &w.qgkv), 2 * facts.q_width(), facts.kv_width())
    } else {
        (
            matmul(&x, &w.q_proj),
            matmul(&x, &w.k_proj),
            matmul(&x, &w.v_proj),
        )
    };
    let (q, gate) = split_q_gate(&qg, facts.q_heads, facts.head_dim);

    // Per-head q/k norms (the weight knows: Gemma fold, per-head), then
    // partial rope: only the first rotary_dim channels of each head rotate.
    let q = rmsnorm(&q, &w.q_norm);
    let k = rmsnorm(&k, &w.k_norm);
    let (q, k) = rope_partial(&q, &k, RopeKind::Standard, facts.rotary_dim);
    w.kv.append(&k, &v);

    // Attention stays opaque — the backend owns plan choice — then the
    // multiply-only output gate and the o_proj accumulate (`+=` of a
    // fresh matmul IS the beta=1 fold).
    let attn = attention(&q, &w.kv, facts.q_width());
    let gated = sigmoid_gate_mul(&attn, &gate);
    y += matmul(&gated, &w.o_proj);
    y
}

/// The full-attention block's CUDA text — [`full_attn_body`]'s
/// kernel-stating counterpart, one per non-CommitAdvance [`FireClass`].
///
/// ONLY the kernel CHOICES differ from the semantic text: the KV write
/// (the per-fire `HasWriteDesc` guard, both arms stated — llama_like's 4a
/// form) and the attention kernel (FlashInfer decode vs the planned
/// prefill dispatch). Everything else — the norms (incl. the Gemma
/// per-head pair), the projections and splits, the partial rope, the
/// sigmoid output gate, the o_proj fold — is a 1:1-kernel op stated the
/// same way in both texts.
fn full_attn_body_cuda(
    t: &Trace,
    l: u32,
    facts: &Qwen35FullAttnFacts,
    c: &Qwen35CudaFacts,
    y: &Val,
    class: FireClass,
) -> Val {
    let w = FullAttnLayerW::new(t, l, facts, c.proj_repr);
    // THIS LAYER's sliding window, `-1` for none — a load-time fact the
    // dispatch statements carry.
    let window_left = model_ir::facts::window_left_at(&c.window_left, l);
    let mut y = y.clone();

    let x = dsl::cuda::rmsnorm(&y, &w.attn_norm);

    let (qg, k, v) = if facts.fused_qkv {
        split_qkv(&matmul(&x, &w.qgkv), 2 * facts.q_width(), facts.kv_width())
    } else {
        (
            matmul(&x, &w.q_proj),
            matmul(&x, &w.k_proj),
            matmul(&x, &w.v_proj),
        )
    };
    let (q, gate) = split_q_gate(&qg, facts.q_heads, facts.head_dim);

    let q = dsl::cuda::rmsnorm(&q, &w.q_norm);
    let k = dsl::cuda::rmsnorm(&k, &w.k_norm);
    let (q, k) = dsl::cuda::rope_partial(&q, &k, facts.rotary_dim);
    // The OnAttnProj site (A4): post-rope, pre-KV-write — the
    // hand-written full-attn invoke's position, observing the roped q
    // (bf16). Observation-only, like the GDN sites.
    dsl::seam(q.trace(), &dsl::seam::ATTN_Q, &[&q], Some(l));

    // The KV-write mechanism is a per-fire runtime input (explicit
    // descriptors when the fire steers a graph replay, page-derived
    // otherwise) — the same HasWriteDesc guard llama_like's CUDA text
    // carries, both arms stated.
    dsl::regions(
        t,
        None,
        None,
        |c| {
            c.arm(dsl::Region::Fire(GuardPred::HasWriteDesc), || {
                cuda::write_kv_explicit(&k, &v, &w.kv);
            });
        },
        || {
            cuda::write_kv_to_pages(&k, &v, &w.kv);
        },
    );

    // qwen3_5's cache is bf16-gated, so the prefill arm is the
    // dequant-less planned dispatch.
    // StateOnly runs the full backbone, prefill-shaped — the Prefill arm;
    // CommitAdvance skips full-attention layers entirely and never enters
    // this body (the retired commit-advance pass).
    let attn = match class {
        // The single-request redirect is a GUARD, not a second class: at
        // `prefill_decode` the prepare plans the prefill path for R == 1
        // and leaves `decode_plan` null, and in a pure-decode fire one
        // token per request makes `num_requests == 1` exactly
        // `TokensLE(1)`. Both arms stated, so neither reading is the
        // executor's to guess.
        FireClass::Decode if c.prefill_decode => {
            let out_shape = (
                Shape(vec![Dim::Tokens, Dim::Const(facts.q_width())]),
                DType::BF16,
            );
            dsl::regions(
                t,
                Some(l),
                Some(out_shape),
                |c| {
                    c.arm(dsl::Region::Fire(GuardPred::TokensLE(1)), || {
                        cuda::attention_flashinfer_prefill(&q, &w.kv, window_left, facts.head_dim, 0.0, 0.0);
                    });
                },
                || {
                    cuda::attention_flashinfer_decode(&q, &w.kv, window_left, facts.head_dim);
                },
            )
        }
        FireClass::Decode => cuda::attention_flashinfer_decode(&q, &w.kv, window_left, facts.head_dim),
        FireClass::Prefill => {
            // No dequant statement beside it: qwen3_5's full-attention
            // path gates on a native-bf16 cache.
            cuda::attention_flashinfer_prefill(&q, &w.kv, window_left, facts.head_dim, 0.0, 0.0)
        }
    };
    let attn = attn.expect("a plain attention statement produces its value");
    let gated = sigmoid_gate_mul(&attn, &gate);
    // The OnAttn site: after the output gate, before the o_proj — the
    // hand-written invoke's position (observing q).
    dsl::seam(q.trace(), &dsl::seam::ATTN_OUT, &[&q], Some(l));
    y += matmul(&gated, &w.o_proj);
    y
}

/// The dense SwiGLU MLP block's op emission at layer `l`
/// (`qwen3_5_forward.cpp::qwen35_dense_mlp_block`): pre-norm → gate‖up →
/// swiglu → down landed on the residual (the beta=1 GEMM). The driver's
/// fused-vs-unfused gate/up banks are emitter dispatch on the single traced
/// `gate_up` matmul, not a fact — the same call llama_like's olmo2 comment
/// records for its unfused gate/up binding.
fn dense_mlp_body(l: u32, hidden: u32, intermediate: u32, variant: NormVariant, y: &Val) -> Val {
    let w = |name: &str| format!("layer.{l}.{name}");
    let mlp_norm = NormW {
        name: w("mlp_norm"),
        variant,
        per_head: None,
        layer: Some(l),
    };
    let gate_up = MatW {
        name: w("gate_up"),
        width: 2 * intermediate,
        layer: Some(l),
        repr: WeightRepr::Bf16,
    };
    let down = MatW {
        name: w("down"),
        width: hidden,
        layer: Some(l),
        repr: WeightRepr::Bf16,
    };
    let mut y = y.clone();
    let m = rmsnorm(&y, &mlp_norm);
    let act = swiglu(&matmul(&m, &gate_up), intermediate);
    y += matmul(&act, &down);
    y
}

/// The dense MLP block's CUDA reading — [`dense_mlp_body`]'s peer,
/// differing in exactly one statement: the activation names its kernel.
///
/// `packed` is [`Qwen35CudaFacts::gate_up_fused`], and the reasoning is
/// llama_like's verbatim — a checkpoint that bound the packed gate‖up
/// bank lands the projection in one buffer and takes the CHUNKED kernel,
/// one that did not lands two and takes the pair form. The trace
/// declares ONE packed matmul either way, because whether the binding
/// materialised it as one buffer or two is a BUFFER question.
fn dense_mlp_body_cuda(
    l: u32,
    hidden: u32,
    intermediate: u32,
    variant: NormVariant,
    y: &Val,
    packed: bool,
    repr: WeightRepr,
) -> Val {
    let w = |name: &str| format!("layer.{l}.{name}");
    let mlp_norm = NormW {
        name: w("mlp_norm"),
        variant,
        per_head: None,
        layer: Some(l),
    };
    // The PACKED bank is what the loader's dense join built, and that
    // join declines quantized groups -- so this handle is BF16 by the
    // same contract that makes it exist. The unfused pair below carries
    // the deployment's repr, which is where a quantized checkpoint's
    // gate and up actually live.
    let gate_up = MatW {
        name: w("gate_up"),
        width: 2 * intermediate,
        layer: Some(l),
        repr: WeightRepr::Bf16,
    };
    let half = |name: &str| MatW {
        name: w(name),
        width: intermediate,
        layer: Some(l),
        repr,
    };
    let down = MatW {
        name: w("down"),
        width: hidden,
        layer: Some(l),
        repr,
    };
    let mut y = y.clone();
    let m = dsl::cuda::rmsnorm(&y, &mlp_norm);
    // 2d: the binding's answer, STATED. llama_like's `mlp` helper
    // verbatim -- one packed matmul into the chunked kernel, or two
    // matmuls into the pair form.
    let act = if packed {
        dsl::cuda::swiglu(&matmul(&m, &gate_up), intermediate)
    } else {
        dsl::cuda::swiglu_pair(
            &matmul(&m, &half("gate_proj")),
            &matmul(&m, &half("up_proj")),
            intermediate,
        )
    };
    y += matmul(&act, &down);
    y
}

/// The full qwen3_5 HYBRID declaration — the first whole-model trace beyond
/// llama_like, composing the three fragment bodies exactly as plan.md Part
/// 1 sketches:
///
/// ```text
/// let mut y = embed[tok];
/// for l in 0..layers {
///     y += match layers[l] {          // static match, resolved at trace time
///         Full   => full_attn(l, dsl::cuda::rmsnorm(y, attn_norm)),
///         Linear => gdn(l, dsl::cuda::rmsnorm(y, attn_norm)),
///     };
///     y += mlp(l, dsl::cuda::rmsnorm(y, mlp_norm));   // dense or MoE, per the facts
/// }
/// lm_head(dsl::cuda::rmsnorm(y, final_norm))
/// ```
///
/// The `match layers[l]` runs over [`Qwen35HybridFacts::is_full_attn`] —
/// the checkpoint's `layer_types` schedule stated as the regular interval
/// (see the facts doc for the provenance chain) — and, like every fact
/// branch, executes at trace time and vanishes: the traced form is a flat
/// op list whose layer kinds are baked in. Each layer's attention ops are
/// EXACTLY the standalone fragment's ([`qwen3_5_gdn_block`] /
/// [`qwen3_5_full_attn_block`] — one shared body each, pinned by test), so
/// everything those fragments state about lowerings, per-request state
/// marking and binding facts holds here per layer.
///
/// Mirrors `qwen3_5_forward.cpp::qwen3_5_forward_paged`'s walk: embed
/// (`kernels::layout::embed_bf16`) → per layer {pre-attn norm + attention body,
/// pre-MLP norm + MLP body} → final norm (`kernels::norm::rmsnorm_gemma_bf16`) →
/// lm_head (`kernels::gemm::act_x_w`). The compact-logit gather, the state-only and
/// commit-advance fires, MTP and the verify/rs-buffer services are
/// per-fire services around this one pass, not ops of it.
pub fn qwen3_5_hybrid(facts: &Qwen35HybridFacts) -> ForwardPlan {
    let hidden = hybrid_hidden(facts);
    dsl::trace_named("qwen3_5_hybrid", |t| {
        dsl::seam(t, &dsl::seam::IN, &[], None);
        let mut y = dsl::embed_with(t, "embed", hidden);

        for l in 0..facts.layers {
            let y_attn = if facts.is_full_attn(l) {
                full_attn_body(t, l, &facts.attn, &y)
            } else {
                gdn_attn_body(t, l, &facts.gdn, &y)
            };
            y = match &facts.mlp {
                Qwen35MlpKind::Dense { intermediate } => {
                    dense_mlp_body(l, hidden, *intermediate, facts.norm_variant, &y_attn)
                }
                Qwen35MlpKind::Moe(moe) => moe_mlp_body(l, moe, &y_attn),
            };
        }

        hybrid_epilogue(t, facts, &y, /*stated=*/ false);
    })
}

/// The qwen3_5 hybrid's CUDA text — [`qwen3_5_hybrid`]'s kernel-stating
/// counterpart, traced with the CUDA backend facts and a fire class, so
/// the traced form states its kernels as raw
/// signatures ([`model_dsl::cuda`]; north-star-dsl.md rung 4c). One
/// trace per [`FireClass`] the deployment fires; family names
/// `qwen3_5_hybrid.cuda.decode` / `.prefill` — the [`llama_like_cuda`]
/// naming, verbatim — plus the two SERVICE classes (rung 4c-iv):
/// `.state_only` (the whole backbone, prefill-shaped, minus the
/// final-norm/lm_head epilogue) and `.commit_advance` (the spec-decode
/// repair: a genuinely different pass — the retired commit-advance pass).
pub fn qwen3_5_hybrid_cuda(
    facts: &Qwen35HybridFacts,
    cuda: &Qwen35CudaFacts,
    class: FireClass,
) -> ForwardPlan {
    let hidden = hybrid_hidden(facts);
    let family = format!(
        "qwen3_5_hybrid.cuda.{}",
        match class {
            FireClass::Decode => "decode",
            FireClass::Prefill => "prefill",
        }
    );
    dsl::trace_named(&family, |t| {
        dsl::seam(t, &dsl::seam::IN, &[], None);
        let mut y = dsl::embed_with(t, "embed", hidden);

        for l in 0..facts.layers {
            let y_attn = if facts.is_full_attn(l) {
                full_attn_body_cuda(t, l, &facts.attn, cuda, &y, class)
            } else {
                gdn_attn_body_cuda(t, l, &facts.gdn, &y, cuda, class)
            };
            y = match &facts.mlp {
                Qwen35MlpKind::Dense { intermediate } => dense_mlp_body_cuda(
                    l,
                    hidden,
                    *intermediate,
                    facts.norm_variant,
                    &y_attn,
                    cuda.gate_up_fused,
                    cuda.proj_repr,
                ),
                Qwen35MlpKind::Moe(moe) => moe_mlp_body_cuda(l, moe, cuda, &y_attn, class),
            };
        }

        hybrid_epilogue(t, facts, &y, /*stated=*/ true);
    })
}

/// The hybrid's cross-facts check, shared by the two texts: the sub-facts
/// are separate structs, so a deployment that disagrees with itself about
/// `hidden` is caught before any op is recorded.
fn hybrid_hidden(facts: &Qwen35HybridFacts) -> u32 {
    let hidden = facts.hidden();
    assert_eq!(
        facts.gdn.hidden, hidden,
        "hybrid sub-facts disagree on hidden (gdn)"
    );
    if let Qwen35MlpKind::Moe(moe) = &facts.mlp {
        assert_eq!(
            moe.hidden, hidden,
            "hybrid sub-facts disagree on hidden (moe)"
        );
    }
    hidden
}

/// Final norm → lm_head, resolving the tied-embedding fact. No kernel
/// choice lives here (both ops are 1:1), so both texts state it the same
/// way and it is written once.
/// `stated`: this epilogue is being traced for a BACKEND, so its final
/// norm names its kernel. The semantic caller passes false, because a
/// backend-independent trace has no kernel to name — the same split
/// `dsl::rmsnorm` and `dsl::cuda::rmsnorm` are.
fn hybrid_epilogue(t: &Trace, facts: &Qwen35HybridFacts, y: &Val, stated: bool) {
    let final_norm = NormW {
        name: "final_norm".to_string(),
        variant: facts.norm_variant,
        per_head: None,
        layer: None,
    };
    let normed = if stated {
        dsl::cuda::rmsnorm(y, &final_norm)
    } else {
        rmsnorm(y, &final_norm)
    };
    let logits = dsl::lm_head_tied(t, &normed, facts.tied_embeddings, facts.vocab);
    dsl::seam(t, &dsl::seam::OUT, &[&logits], None);
}
// THE COMMIT-ADVANCE PASS IS GONE (`.wiki/driver/graph.md` §4.2).
//
// It re-advanced each linear layer's conv window and recurrent state over
// a confirmed prefix, fed from a verify hidden stash -- a REPAIR, and
// directive 4.2 says there are none. A speculative decode writes its
// tokens into a buffer and folds only the accepted prefix into the linear
// state; a rejected token is never folded, so the state is never wrong
// and there is nothing to re-advance. The fold length IS the confirmed
// prefix, and the driver resolves it (`PIE_RS_FLAG_FOLD`, and v24's
// `_FOLD_LEN_DEVICE` for a length only the device knows).

#[cfg(test)]
mod tests {
    use super::*;
    // A handful of tests here compare against the dense family -- "the hybrid's
    // KV marks match llama_like's" is a statement about both, so it is named
    // across the module boundary rather than duplicated.
    use crate::shared::llama_like::forward::facts::LlamaLikeFacts;
    use crate::shared::llama_like::forward::llama_like;
    use model_ir::trace::{DType, Dim, NormVariant, OpKind, StateRef, StateStore};

    /// The MoE block fragment's op sequence, mapped launch for launch to
    /// `run_moe_mlp`'s decode fast path (the table on
    /// [`qwen3_5_moe_mlp_block`]).
    #[test]
    fn moe_block_op_sequence() {
        let plan = qwen3_5_moe_mlp_block(&Qwen35MoeMlpFacts::qwen3_5_35b_a3b());
        let kinds: Vec<&'static str> = plan
            .layer_ops(0)
            .map(|op| match &op.kind {
                OpKind::Rmsnorm { .. } => "rmsnorm",
                OpKind::Matmul {
                    selector: Some(_), ..
                } => "matmul_per_token",
                OpKind::Matmul { .. } => "matmul",
                OpKind::TopK { .. } => "topk",
                OpKind::Swiglu { .. } => "swiglu",
                OpKind::WeightedSum { .. } => "weighted_sum",
                OpKind::SigmoidGateAdd => "sigmoid_gate_add",
                OpKind::ResidualAdd => "residual_add",
                _ => "other",
            })
            .collect();
        assert_eq!(
            kinds,
            [
                "rmsnorm",          // mlp_norm (gemma fold)
                "matmul",           // router logits [Tokens, E]
                "topk",             // launch_topk_softmax: idx + renormed weights
                "matmul_per_token", // grouped gate_up over the selected experts
                "swiglu",           // chunked swiglu over [Tokens, k, Im]
                "matmul_per_token", // grouped down
                "weighted_sum",     // [Tokens, k, H] -> [Tokens, H]
                "matmul",           // shared_expert.gate_up
                "swiglu",
                "matmul",           // shared_expert.down
                "matmul",           // shared_expert_gate: [Tokens, 1] logit
                "sigmoid_gate_add", // routed + sigmoid(gate) * shared
                "residual_add",     // y += moe_out
            ]
        );
    }

    /// Without a shared expert (qwen3_moe: `shared_expert_intermediate` 0)
    /// the five shared ops fold away at trace time, llama_like-branch
    /// style, and the routed combine lands on the residual directly.
    #[test]
    fn moe_block_without_shared_expert_folds_the_shared_ops() {
        let facts = Qwen35MoeMlpFacts {
            shared_expert_intermediate: 0,
            norm_variant: NormVariant::Plain,
            ..Qwen35MoeMlpFacts::qwen3_5_35b_a3b()
        };
        let plan = qwen3_5_moe_mlp_block(&facts);
        assert_eq!(plan.ops.len(), 8);
        assert!(
            !plan
                .ops
                .iter()
                .any(|op| matches!(op.kind, OpKind::SigmoidGateAdd))
        );
        assert!(!plan.ops.iter().any(|op| {
            matches!(&op.kind, OpKind::Matmul { weight, .. } if weight.contains("shared"))
        }));
        // The residual add consumes the weighted sum's output directly.
        let add = plan.ops.last().unwrap();
        assert!(matches!(add.kind, OpKind::ResidualAdd));
        let combine = plan
            .ops
            .iter()
            .find(|op| matches!(op.kind, OpKind::WeightedSum { .. }))
            .unwrap();
        assert_eq!(add.inputs[0], combine.outputs[0]);
    }

    /// The dyn dataflow: TopK's index output is the fragment's only
    /// dyn-marked value, both expert matmuls name it as their selector AND
    /// their last input, and their weight names are `{e}` templates.
    #[test]
    fn moe_block_selector_dataflow() {
        let plan = qwen3_5_moe_mlp_block(&Qwen35MoeMlpFacts::qwen3_5_35b_a3b());
        let topk = plan
            .ops
            .iter()
            .find(|op| matches!(op.kind, OpKind::TopK { .. }))
            .unwrap();
        let idx = topk.outputs[0];
        let dyn_values: Vec<_> = plan
            .values
            .iter()
            .enumerate()
            .filter(|(_, v)| v.dyn_axis.is_some())
            .map(|(i, _)| i as u32)
            .collect();
        assert_eq!(dyn_values, vec![idx]);
        assert_eq!(plan.values[idx as usize].dtype, DType::I32);

        let grouped: Vec<_> = plan
            .ops
            .iter()
            .filter(|op| {
                matches!(
                    &op.kind,
                    OpKind::Matmul {
                        selector: Some(_),
                        ..
                    }
                )
            })
            .collect();
        assert_eq!(grouped.len(), 2);
        for op in &grouped {
            let OpKind::Matmul {
                weight, selector, ..
            } = &op.kind
            else {
                unreachable!()
            };
            assert_eq!(*selector, Some(idx));
            assert_eq!(*op.inputs.last().unwrap(), idx);
            assert!(weight.contains("{e}"), "not a template: {weight}");
        }
        assert!(matches!(
            &grouped[0].kind,
            OpKind::Matmul { weight, .. } if weight == "layer.0.expert.{e}.gate_up"
        ));
        assert!(matches!(
            &grouped[1].kind,
            OpKind::Matmul { weight, .. } if weight == "layer.0.expert.{e}.down"
        ));
    }

    /// Route-expanded shapes: the grouped matmuls and the swiglu between
    /// them carry the `[Tokens, k, ...]` factored form of the driver's
    /// `[N*K, ...]` scratch, and the weighted sum collapses it back.
    #[test]
    fn moe_block_route_expanded_shapes() {
        let facts = Qwen35MoeMlpFacts::qwen3_5_35b_a3b();
        let plan = qwen3_5_moe_mlp_block(&facts);
        let k = Dim::Const(facts.top_k);
        let by_kind = |pred: fn(&OpKind) -> bool| {
            plan.ops
                .iter()
                .filter(move |op| pred(&op.kind))
                .collect::<Vec<_>>()
        };

        let grouped = by_kind(|k| {
            matches!(
                k,
                OpKind::Matmul {
                    selector: Some(_),
                    ..
                }
            )
        });
        assert_eq!(
            plan.values[grouped[0].outputs[0] as usize].shape.0,
            vec![Dim::Tokens, k, Dim::Const(2 * facts.moe_intermediate)]
        );
        assert_eq!(
            plan.values[grouped[1].outputs[0] as usize].shape.0,
            vec![Dim::Tokens, k, Dim::Const(facts.hidden)]
        );

        // The routed swiglu keeps the route dims; the shared one is the
        // ordinary dense shape.
        let swiglus = by_kind(|k| matches!(k, OpKind::Swiglu { .. }));
        assert_eq!(
            plan.values[swiglus[0].outputs[0] as usize].shape.0,
            vec![Dim::Tokens, k, Dim::Const(facts.moe_intermediate)]
        );
        assert_eq!(
            plan.values[swiglus[1].outputs[0] as usize].shape.0,
            vec![Dim::Tokens, Dim::Const(facts.shared_expert_intermediate)]
        );

        let combine = by_kind(|k| matches!(k, OpKind::WeightedSum { .. }));
        assert!(matches!(combine[0].kind, OpKind::WeightedSum { k } if k == facts.top_k));
        assert_eq!(
            plan.values[combine[0].outputs[0] as usize].shape.0,
            vec![Dim::Tokens, Dim::Const(facts.hidden)]
        );

        // The shared gate logit is the [Tokens, 1] scalar-gate GEMM.
        let gate = plan
            .ops
            .iter()
            .find(|op| {
                matches!(&op.kind, OpKind::Matmul { weight, .. }
                    if weight.ends_with("shared_expert_gate"))
            })
            .unwrap();
        assert_eq!(
            plan.values[gate.outputs[0] as usize].shape.0,
            vec![Dim::Tokens, Dim::Const(1)]
        );
    }

    /// The fragment parameter is honest dataflow: value 0 is produced by no
    /// op, read by the block's first norm, and landed on by the final
    /// residual add.
    #[test]
    fn moe_block_residual_stream_is_a_fragment_parameter() {
        let plan = qwen3_5_moe_mlp_block(&Qwen35MoeMlpFacts::qwen3_5_35b_a3b());
        assert!(!plan.ops.iter().any(|op| op.outputs.contains(&0)));
        assert!(matches!(&plan.ops[0].kind, OpKind::Rmsnorm { weight, .. }
            if weight == "layer.0.mlp_norm"));
        assert_eq!(plan.ops[0].inputs, vec![0]);
        let add = plan.ops.last().unwrap();
        assert!(matches!(add.kind, OpKind::ResidualAdd));
        assert_eq!(*add.inputs.last().unwrap(), 0);
    }

    /// The dyn vocabulary survives serde — selector fields, dyn markers,
    /// rank-3 shapes — and, per the additive rule, none of it appears in a
    /// dyn-free plan's serialization (the goldens pin that byte-for-byte;
    /// this pins the reason).
    #[test]
    fn moe_traced_form_round_trips() {
        let plan = qwen3_5_moe_mlp_block(&Qwen35MoeMlpFacts::qwen3_5_35b_a3b());
        let json = serde_json::to_string(&plan).unwrap();
        let back: ForwardPlan = serde_json::from_str(&json).unwrap();
        assert_eq!(plan, back);

        let dense = serde_json::to_string(&llama_like(&LlamaLikeFacts::qwen3_0_6b())).unwrap();
        assert!(!dense.contains("selector"));
        assert!(!dense.contains("dyn_axis"));
    }

    /// The GDN block fragment's op sequence, mapped launch for launch to
    /// `linear_attn_layer_body`'s decode fast path (the table on
    /// [`qwen3_5_gdn_block`]), on the default (unfused) binding.
    #[test]
    fn gdn_block_op_sequence() {
        let plan = qwen3_5_gdn_block(&Qwen35GdnFacts::qwen3_5_0_8b());
        let kinds: Vec<&'static str> = plan
            .layer_ops(0)
            .map(|op| match &op.kind {
                OpKind::Rmsnorm { .. } => "rmsnorm",
                OpKind::Matmul {
                    beta_one: false, ..
                } => "matmul",
                OpKind::Matmul { beta_one: true, .. } => "matmul+res",
                OpKind::SplitGdn { .. } => "split_gdn",
                OpKind::CausalConv1d { .. } => "causal_conv1d",
                OpKind::GdnPrep { .. } => "gdn_prep",
                OpKind::GatedDelta { .. } => "gated_delta",
                OpKind::RmsnormGated { .. } => "rmsnorm_gated",
                _ => "other",
            })
            .collect();
        assert_eq!(
            kinds,
            [
                "rmsnorm",       // attn_norm (gemma fold)
                "matmul",        // in_proj_qkv [Tokens, conv_dim]
                "matmul",        // in_proj_z  [Tokens, v_dim]
                "matmul",        // in_proj_a  [Tokens, Vh]
                "matmul",        // in_proj_b  [Tokens, Vh]
                "causal_conv1d", // per-request conv state, fused silu
                "gdn_prep",      // q/k/v/g/beta from qkv+a+b (+a_log, dt_bias)
                "gated_delta",   // per-request recurrent state -> core
                "rmsnorm_gated", // z-gated per-head norm, plain fold
                "matmul+res",    // o_proj, beta=1
            ]
        );
        assert_eq!(plan.ops.len(), 10);
    }

    /// The fused in-proj binding (`PIE_QWEN35_FUSED_GDN_PROJ`) trades the
    /// four projection matmuls for two matmuls + two SplitGdn launches —
    /// same count, resolved at trace time — and the ba split's outputs are
    /// (b, a) in the driver's packing order, so `a` is the split's SECOND
    /// output while gdn_prep consumes `[qkv, a, b]`.
    #[test]
    fn gdn_block_fused_binding_traces_two_splits() {
        let facts = Qwen35GdnFacts {
            fused_in_proj: true,
            ..Qwen35GdnFacts::qwen3_5_0_8b()
        };
        let plan = qwen3_5_gdn_block(&facts);
        assert_eq!(plan.ops.len(), 10);
        let matmuls = plan
            .ops
            .iter()
            .filter(|op| matches!(op.kind, OpKind::Matmul { .. }))
            .count();
        assert_eq!(matmuls, 3); // qkvz, ba, o_proj
        let splits: Vec<_> = plan
            .ops
            .iter()
            .filter(|op| matches!(op.kind, OpKind::SplitGdn { .. }))
            .collect();
        assert_eq!(splits.len(), 2);
        assert!(matches!(
            splits[0].kind,
            OpKind::SplitGdn { width0, width1 }
                if width0 == facts.conv_dim() && width1 == facts.value_width()
        ));
        assert!(matches!(
            splits[1].kind,
            OpKind::SplitGdn { width0, width1 }
                if width0 == facts.value_heads && width1 == facts.value_heads
        ));
        // gdn_prep's a operand is the ba split's SECOND output ([b | a]).
        let prep = plan
            .ops
            .iter()
            .find(|op| matches!(op.kind, OpKind::GdnPrep { .. }))
            .unwrap();
        assert_eq!(prep.inputs[1], splits[1].outputs[1]); // a
        assert_eq!(prep.inputs[2], splits[1].outputs[0]); // b
    }

    /// Dataflow and shapes of the GDN core: conv is shape-preserving over
    /// the packed `[Tokens, conv_dim]`, prep emits the compact per-head
    /// rank-3 f32 forms, the recurrence keeps v's shape, and the gated
    /// norm flattens to the z gate's `[Tokens, v_dim]` bf16.
    #[test]
    fn gdn_block_core_shapes() {
        let facts = Qwen35GdnFacts::qwen3_5_0_8b();
        let plan = qwen3_5_gdn_block(&facts);
        let shape_of = |id: u32| plan.values[id as usize].shape.0.clone();
        let dtype_of = |id: u32| plan.values[id as usize].dtype;

        // 0.8B geometry sanity, against the metal driver's stated launch
        // geometry (decode_consts.cpp): 1024 -> 6144, z 1024 -> 2048.
        assert_eq!(facts.conv_dim(), 6144);
        assert_eq!(facts.value_width(), 2048);

        let conv = plan
            .ops
            .iter()
            .find(|op| matches!(op.kind, OpKind::CausalConv1d { .. }))
            .unwrap();
        assert!(matches!(
            &conv.kind,
            OpKind::CausalConv1d { weight, bias: None, layer: 0, kernel: 4 }
                if weight == "layer.0.conv"
        ));
        assert_eq!(
            shape_of(conv.outputs[0]),
            vec![Dim::Tokens, Dim::Const(facts.conv_dim())]
        );

        let prep = plan
            .ops
            .iter()
            .find(|op| matches!(op.kind, OpKind::GdnPrep { .. }))
            .unwrap();
        assert!(matches!(
            &prep.kind,
            OpKind::GdnPrep { a_log, dt_bias }
                if a_log == "layer.0.a_log" && dt_bias == "layer.0.dt_bias"
        ));
        assert_eq!(prep.inputs[0], conv.outputs[0]);
        assert_eq!(prep.outputs.len(), 5);
        let kh = Dim::Const(facts.key_heads);
        let kd = Dim::Const(facts.key_head_dim);
        let vh = Dim::Const(facts.value_heads);
        let vd = Dim::Const(facts.value_head_dim);
        assert_eq!(shape_of(prep.outputs[0]), vec![Dim::Tokens, kh, kd]); // q
        assert_eq!(shape_of(prep.outputs[1]), vec![Dim::Tokens, kh, kd]); // k
        assert_eq!(shape_of(prep.outputs[2]), vec![Dim::Tokens, vh, vd]); // v
        assert_eq!(shape_of(prep.outputs[3]), vec![Dim::Tokens, vh]); // g
        assert_eq!(shape_of(prep.outputs[4]), vec![Dim::Tokens, vh]); // beta
        for &out in &prep.outputs {
            assert_eq!(dtype_of(out), DType::F32);
        }

        let delta = plan
            .ops
            .iter()
            .find(|op| matches!(op.kind, OpKind::GatedDelta { .. }))
            .unwrap();
        assert_eq!(delta.inputs, prep.outputs); // [q, k, v, g, beta]
        assert_eq!(shape_of(delta.outputs[0]), vec![Dim::Tokens, vh, vd]);
        assert_eq!(dtype_of(delta.outputs[0]), DType::F32);

        // The gated norm consumes the rank-3 core and the z gate, and
        // lands the flat bf16 form the o_proj GEMM reads.
        let gated = plan
            .ops
            .iter()
            .find(|op| matches!(op.kind, OpKind::RmsnormGated { .. }))
            .unwrap();
        assert_eq!(gated.inputs[0], delta.outputs[0]);
        let z = plan
            .ops
            .iter()
            .find(|op| {
                matches!(&op.kind, OpKind::Matmul { weight, .. } if weight == "layer.0.in_proj_z")
            })
            .unwrap();
        assert_eq!(gated.inputs[1], z.outputs[0]);
        assert_eq!(
            shape_of(gated.outputs[0]),
            vec![Dim::Tokens, Dim::Const(facts.value_width())]
        );
        assert_eq!(dtype_of(gated.outputs[0]), DType::BF16);

        // o_proj accumulates onto the fragment parameter (value 0).
        let o_proj = plan.ops.last().unwrap();
        assert!(matches!(
            &o_proj.kind,
            OpKind::Matmul { beta_one: true, weight, .. } if weight == "layer.0.o_proj"
        ));
        assert_eq!(o_proj.inputs, vec![gated.outputs[0], 0]);
    }

    /// The per-request state axis (§5.4), marked by vocabulary: exactly the
    /// conv and the recurrence address the RecurrentState store at the
    /// block's layer — the traced-form statement of `touches_rs_buffer` —
    /// while llama_like's KvAppend/Attention mark KvCache and the MoE
    /// fragment marks nothing.
    #[test]
    fn gdn_block_marks_the_per_request_state() {
        let plan = qwen3_5_gdn_block(&Qwen35GdnFacts::qwen3_5_0_8b());
        let marks: Vec<_> = plan
            .ops
            .iter()
            .filter_map(|op| op.kind.state_ref())
            .collect();
        assert_eq!(
            marks,
            vec![
                StateRef {
                    store: StateStore::RecurrentState,
                    layer: 0
                },
                StateRef {
                    store: StateStore::RecurrentState,
                    layer: 0
                },
            ]
        );

        let kv_marks: Vec<_> = llama_like(&LlamaLikeFacts::qwen3_0_6b())
            .layer_ops(3)
            .filter_map(|op| op.kind.state_ref())
            .collect();
        assert_eq!(
            kv_marks,
            vec![
                StateRef {
                    store: StateStore::KvCache,
                    layer: 3
                },
                StateRef {
                    store: StateStore::KvCache,
                    layer: 3
                },
            ]
        );

        let moe = qwen3_5_moe_mlp_block(&Qwen35MoeMlpFacts::qwen3_5_35b_a3b());
        assert!(moe.ops.iter().all(|op| op.kind.state_ref().is_none()));
    }

    /// The fragment parameter is honest dataflow, MoE-fragment style: value
    /// 0 is produced by no op, read first by the block norm, and landed on
    /// by the o_proj accumulate.
    #[test]
    fn gdn_block_residual_stream_is_a_fragment_parameter() {
        let plan = qwen3_5_gdn_block(&Qwen35GdnFacts::qwen3_5_0_8b());
        assert!(!plan.ops.iter().any(|op| op.outputs.contains(&0)));
        assert!(matches!(&plan.ops[0].kind, OpKind::Rmsnorm { weight, .. }
            if weight == "layer.0.attn_norm"));
        assert_eq!(plan.ops[0].inputs, vec![0]);
        assert_eq!(*plan.ops.last().unwrap().inputs.last().unwrap(), 0);
    }

    /// The full-attention block fragment's op sequence, mapped launch for
    /// launch to `full_attn_layer_body` (the table on
    /// [`qwen3_5_full_attn_block`]), on the default (unfused) binding.
    #[test]
    fn full_attn_block_op_sequence() {
        let plan = qwen3_5_full_attn_block(&Qwen35FullAttnFacts::qwen3_5_0_8b());
        let kinds: Vec<&'static str> = plan
            .layer_ops(0)
            .map(|op| match &op.kind {
                OpKind::Rmsnorm { .. } => "rmsnorm",
                OpKind::Matmul {
                    beta_one: false, ..
                } => "matmul",
                OpKind::Matmul { beta_one: true, .. } => "matmul+res",
                OpKind::SplitQkv { .. } => "split_qkv",
                OpKind::SplitQGate { .. } => "split_q_gate",
                OpKind::RmsnormPerHead { .. } => "rmsnorm_per_head",
                OpKind::Rope { .. } => "rope",
                OpKind::KvAppend { .. } => "kv_append",
                OpKind::Attention { .. } => "attention",
                OpKind::SigmoidGateMul => "sigmoid_gate_mul",
                _ => "other",
            })
            .collect();
        assert_eq!(
            kinds,
            [
                "rmsnorm",          // attn_norm (gemma fold)
                "matmul",           // q_proj, 2x wide: [Tokens, 2*Hq]
                "matmul",           // k_proj [Tokens, Hk]
                "matmul",           // v_proj [Tokens, Hk]
                "split_q_gate",     // per-head [query | gate] de-interleave
                "rmsnorm_per_head", // q_norm (gemma fold)
                "rmsnorm_per_head", // k_norm
                "rope",             // partial: first rotary_dim channels
                "kv_append",
                "attention",
                "sigmoid_gate_mul", // attn_out *= sigmoid(gate)
                "matmul+res",       // o_proj, beta=1
            ]
        );
        assert_eq!(plan.ops.len(), 12);
    }

    /// The fused qgkv binding (`PIE_QWEN35_FUSED_FULL_ATTN_QGKV`) trades
    /// the three projections for Matmul(qgkv) + SplitQkv whose "q" leg is
    /// the 2×-wide `[query | gate]` bank — and the SplitQGate de-interleave
    /// still follows, consuming that leg, exactly as the hand-written
    /// `use_fused_qgkv` branch.
    #[test]
    fn full_attn_block_fused_binding_traces_qgkv_split() {
        let facts = Qwen35FullAttnFacts {
            fused_qkv: true,
            ..Qwen35FullAttnFacts::qwen3_5_0_8b()
        };
        let plan = qwen3_5_full_attn_block(&facts);
        assert_eq!(plan.ops.len(), 11);
        let matmuls = plan
            .ops
            .iter()
            .filter(|op| matches!(op.kind, OpKind::Matmul { .. }))
            .count();
        assert_eq!(matmuls, 2); // qgkv, o_proj
        let split = plan
            .ops
            .iter()
            .find(|op| matches!(op.kind, OpKind::SplitQkv { .. }))
            .unwrap();
        assert!(matches!(
            split.kind,
            OpKind::SplitQkv { q_width, kv_width }
                if q_width == 2 * facts.q_width() && kv_width == facts.kv_width()
        ));
        // SplitQGate consumes the split's first (2x-wide q|gate) leg.
        let qg_split = plan
            .ops
            .iter()
            .find(|op| matches!(op.kind, OpKind::SplitQGate { .. }))
            .unwrap();
        assert_eq!(qg_split.inputs, vec![split.outputs[0]]);
        // KvAppend consumes the k and v legs.
        let append = plan
            .ops
            .iter()
            .find(|op| matches!(op.kind, OpKind::KvAppend { .. }))
            .unwrap();
        assert_eq!(append.inputs[1], split.outputs[2]); // v (pre-rope)
    }

    /// Dataflow and params of the gated attention: the interleaved split
    /// carries head geometry and halves the 2×-wide projection, the
    /// per-head norms fold Gemma, rope is partial at the fixture's 64
    /// channels, the output gate multiplies attention's output by the
    /// split's GATE leg, and o_proj lands the gated value on the residual.
    #[test]
    fn full_attn_block_gate_dataflow_and_shapes() {
        let facts = Qwen35FullAttnFacts::qwen3_5_0_8b();
        let plan = qwen3_5_full_attn_block(&facts);
        let shape_of = |id: u32| plan.values[id as usize].shape.0.clone();

        // 0.8B geometry sanity, against the metal driver's stated launch
        // geometry (decode_consts.cpp): q 1024 -> 4096 (2x-wide), k/v
        // 1024 -> 512, o 2048 -> 1024.
        assert_eq!(2 * facts.q_width(), 4096);
        assert_eq!(facts.kv_width(), 512);
        assert_eq!(facts.q_width(), 2048);

        let qg_split = plan
            .ops
            .iter()
            .find(|op| matches!(op.kind, OpKind::SplitQGate { .. }))
            .unwrap();
        assert!(matches!(
            qg_split.kind,
            OpKind::SplitQGate {
                heads: 8,
                head_dim: 256
            }
        ));
        let q_proj = &plan.ops[1];
        assert!(matches!(
            &q_proj.kind,
            OpKind::Matmul { weight, .. } if weight == "layer.0.q_proj"
        ));
        assert_eq!(qg_split.inputs, vec![q_proj.outputs[0]]);
        assert_eq!(
            shape_of(q_proj.outputs[0]),
            vec![Dim::Tokens, Dim::Const(4096)]
        );
        for &out in &qg_split.outputs {
            assert_eq!(shape_of(out), vec![Dim::Tokens, Dim::Const(2048)]);
        }

        // Per-head norms: Gemma fold, head_dim 256, on the QUERY leg.
        let per_head: Vec<_> = plan
            .ops
            .iter()
            .filter(|op| matches!(op.kind, OpKind::RmsnormPerHead { .. }))
            .collect();
        assert_eq!(per_head.len(), 2);
        assert!(matches!(
            &per_head[0].kind,
            OpKind::RmsnormPerHead { weight, head_dim: 256, variant: NormVariant::Gemma }
                if weight == "layer.0.q_norm"
        ));
        assert_eq!(per_head[0].inputs, vec![qg_split.outputs[0]]);

        // Partial rope: the fixture's 64 channels (0.25 x 256).
        let rope = plan
            .ops
            .iter()
            .find(|op| matches!(op.kind, OpKind::Rope { .. }))
            .unwrap();
        assert!(matches!(
            rope.kind,
            OpKind::Rope {
                kind: RopeKind::Standard,
                partial: Some(64)
            }
        ));

        // The output gate: attention's output times the GATE leg — the
        // gate flows AROUND the norm/rope/attention chain, untouched.
        let attn = plan
            .ops
            .iter()
            .find(|op| matches!(op.kind, OpKind::Attention { .. }))
            .unwrap();
        let gate_mul = plan
            .ops
            .iter()
            .find(|op| matches!(op.kind, OpKind::SigmoidGateMul))
            .unwrap();
        assert_eq!(gate_mul.inputs, vec![attn.outputs[0], qg_split.outputs[1]]);
        assert_eq!(
            shape_of(gate_mul.outputs[0]),
            vec![Dim::Tokens, Dim::Const(2048)]
        );

        // o_proj accumulates the GATED value onto the fragment parameter.
        let o_proj = plan.ops.last().unwrap();
        assert!(matches!(
            &o_proj.kind,
            OpKind::Matmul { beta_one: true, weight, .. } if weight == "layer.0.o_proj"
        ));
        assert_eq!(o_proj.inputs, vec![gate_mul.outputs[0], 0]);

        // KvCache marking: exactly KvAppend + Attention, at the block's
        // layer — the same marks llama_like carries, none of the GDN ones.
        let marks: Vec<_> = plan
            .ops
            .iter()
            .filter_map(|op| op.kind.state_ref())
            .collect();
        assert_eq!(
            marks,
            vec![
                StateRef {
                    store: StateStore::KvCache,
                    layer: 0
                },
                StateRef {
                    store: StateStore::KvCache,
                    layer: 0
                },
            ]
        );
    }

    /// The fragment parameter is honest dataflow, MoE/GDN-fragment style.
    #[test]
    fn full_attn_block_residual_stream_is_a_fragment_parameter() {
        let plan = qwen3_5_full_attn_block(&Qwen35FullAttnFacts::qwen3_5_0_8b());
        assert!(!plan.ops.iter().any(|op| op.outputs.contains(&0)));
        assert!(matches!(&plan.ops[0].kind, OpKind::Rmsnorm { weight, .. }
            if weight == "layer.0.attn_norm"));
        assert_eq!(plan.ops[0].inputs, vec![0]);
        assert_eq!(*plan.ops.last().unwrap().inputs.last().unwrap(), 0);
    }

    /// Rewrite a fragment op's kind from layer 0 to layer `l`: weight names
    /// re-prefixed, state-layer params re-pointed. What "the hybrid's layer
    /// ops equal the fragment's" means, made precise.
    fn relayer(kind: &OpKind, l: u32) -> OpKind {
        let re = |w: &str| w.replacen("layer.0.", &format!("layer.{l}."), 1);
        let mut kind = kind.clone();
        match &mut kind {
            OpKind::Matmul { weight, .. }
            | OpKind::Rmsnorm { weight, .. }
            | OpKind::RmsnormPerHead { weight, .. }
            | OpKind::CausalConv1d { weight, .. }
            | OpKind::RmsnormGated { weight }
            | OpKind::AddBias { weight }
            | OpKind::Embed { weight }
            | OpKind::LmHead { weight } => *weight = re(weight),
            OpKind::GdnPrep { a_log, dt_bias } => {
                *a_log = re(a_log);
                *dt_bias = re(dt_bias);
            }
            _ => {}
        }
        match &mut kind {
            OpKind::KvAppend { layer }
            | OpKind::Attention { layer, .. }
            | OpKind::CausalConv1d { layer, .. }
            | OpKind::GatedDelta { layer } => *layer = l,
            _ => {}
        }
        kind
    }

    /// Assert the hybrid's layer-`l` ATTENTION ops are the standalone
    /// fragment's, op for op: same kinds (modulo the layer rewrite) and the
    /// same SSA dataflow under the id mapping {fragment 0 → the layer's
    /// incoming residual, fragment i → the layer's i-th fresh value}.
    fn assert_layer_head_matches_fragment(
        hybrid: &model_ir::trace::ForwardPlan,
        l: u32,
        fragment: &model_ir::trace::ForwardPlan,
    ) {
        let h_ops: Vec<_> = hybrid.layer_ops(l).collect();
        let f_ops: Vec<_> = fragment.layer_ops(0).collect();
        assert!(h_ops.len() > f_ops.len(), "layer {l} shorter than fragment");
        // Fragment value 0 is the parameter; its fresh values start at 1.
        // The hybrid's layer reads the stream as the first op's input and
        // allocates fresh values from the first op's output on.
        let y_in = h_ops[0].inputs[0];
        let base = h_ops[0].outputs[0];
        let map = |id: u32| if id == 0 { y_in } else { base + (id - 1) };
        for (f, h) in f_ops.iter().zip(&h_ops) {
            assert_eq!(h.kind, relayer(&f.kind, l), "kind at layer {l}");
            let mapped_in: Vec<u32> = f.inputs.iter().map(|&i| map(i)).collect();
            let mapped_out: Vec<u32> = f.outputs.iter().map(|&i| map(i)).collect();
            assert_eq!(h.inputs, mapped_in, "inputs of {:?} at layer {l}", f.kind);
            assert_eq!(
                h.outputs, mapped_out,
                "outputs of {:?} at layer {l}",
                f.kind
            );
        }
    }

    /// The hybrid's layer-kind schedule is the checkpoint's 3:1 pattern:
    /// full attention exactly on layers 3, 7, 11, 15, 19, 23 (interval 4,
    /// end of each block — the Metal geometry's `is_full_attn`), GDN
    /// everywhere else, and every layer carries the dense MLP.
    #[test]
    fn hybrid_layer_kind_sequence_matches_the_pattern() {
        let facts = Qwen35HybridFacts::qwen3_5_0_8b();
        let plan = qwen3_5_hybrid(&facts);
        for l in 0..facts.layers {
            let ops: Vec<_> = plan.layer_ops(l).collect();
            let full = ops
                .iter()
                .any(|op| matches!(op.kind, OpKind::Attention { .. }));
            let linear = ops
                .iter()
                .any(|op| matches!(op.kind, OpKind::GatedDelta { .. }));
            assert_eq!(full, l % 4 == 3, "layer {l} full-attention");
            assert_eq!(linear, l % 4 != 3, "layer {l} linear-attention");
            assert!(!(full && linear), "layer {l} mixes kinds");
            // The uniform dense MLP: gate_up + down on every layer.
            assert!(ops.iter().any(|op| matches!(&op.kind,
                OpKind::Matmul { weight, .. } if weight.ends_with("gate_up"))));
        }
    }

    /// The hybrid's GDN layers ARE the standalone GDN fragment, op for op
    /// and edge for edge — the shared-body refactor pinned as behaviour.
    #[test]
    fn hybrid_gdn_layers_equal_the_standalone_fragment() {
        let facts = Qwen35HybridFacts::qwen3_5_0_8b();
        let hybrid = qwen3_5_hybrid(&facts);
        let fragment = qwen3_5_gdn_block(&facts.gdn);
        for l in (0..facts.layers).filter(|&l| !facts.is_full_attn(l)) {
            assert_layer_head_matches_fragment(&hybrid, l, &fragment);
        }
    }

    /// The hybrid's full-attention layers ARE the standalone full-attention
    /// fragment, same pinning.
    #[test]
    fn hybrid_full_attn_layers_equal_the_standalone_fragment() {
        let facts = Qwen35HybridFacts::qwen3_5_0_8b();
        let hybrid = qwen3_5_hybrid(&facts);
        let fragment = qwen3_5_full_attn_block(&facts.attn);
        for l in (0..facts.layers).filter(|&l| facts.is_full_attn(l)) {
            assert_layer_head_matches_fragment(&hybrid, l, &fragment);
        }
    }

    /// The op-count formula: 18 GDN layers x (10 attn + 4 mlp) + 6 full
    /// layers x (12 attn + 4 mlp) + embed + final norm + lm_head — and the
    /// epilogue: tied lm_head over the 0.8B vocab.
    #[test]
    fn hybrid_full_plan_shape() {
        let facts = Qwen35HybridFacts::qwen3_5_0_8b();
        let plan = qwen3_5_hybrid(&facts);
        assert_eq!(plan.ops.len(), 18 * 14 + 6 * 16 + 3);
        assert!(matches!(&plan.ops[0].kind, OpKind::Embed { weight } if weight == "embed"));
        assert!(matches!(
            &plan.ops.last().unwrap().kind,
            OpKind::LmHead { weight } if weight == "embed"
        ));
        let logits = plan.ops.last().unwrap().outputs[0];
        assert_eq!(
            plan.values[logits as usize].shape.0,
            vec![Dim::Requests, Dim::Const(facts.vocab)]
        );
        // Both state stores are marked, on disjoint layer sets: the KV
        // cache exactly on the full-attention layers (twice each: append +
        // attention), the recurrent store exactly on the GDN layers
        // (twice each: conv + recurrence).
        for l in 0..facts.layers {
            let stores: Vec<_> = plan
                .layer_ops(l)
                .filter_map(|op| op.kind.state_ref())
                .collect();
            let store = if facts.is_full_attn(l) {
                StateStore::KvCache
            } else {
                StateStore::RecurrentState
            };
            assert_eq!(
                stores,
                vec![StateRef { store, layer: l }, StateRef { store, layer: l }]
            );
        }
    }

    /// A MoE-MLP hybrid composes the MoE fragment body per layer (the
    /// qwen3.5/3.6-MoE shape): every layer carries the router → topk →
    /// grouped GEMMs → combine block in place of the dense four.
    #[test]
    fn hybrid_with_moe_mlp_composes_the_moe_fragment() {
        let moe = Qwen35MoeMlpFacts {
            hidden: 1024,
            ..Qwen35MoeMlpFacts::qwen3_5_35b_a3b()
        };
        let facts = Qwen35HybridFacts {
            layers: 4,
            mlp: Qwen35MlpKind::Moe(moe),
            ..Qwen35HybridFacts::qwen3_5_0_8b()
        };
        let plan = qwen3_5_hybrid(&facts);
        // 3 GDN layers x (10 + 13) + 1 full layer x (12 + 13) + 3.
        assert_eq!(plan.ops.len(), 3 * 23 + 25 + 3);
        for l in 0..facts.layers {
            assert_eq!(
                plan.layer_ops(l)
                    .filter(|op| matches!(op.kind, OpKind::TopK { .. }))
                    .count(),
                1,
                "layer {l} routes"
            );
        }
    }

    /// The lowered GDN prefill recurrence under a GQA share (the 0.8B
    /// fixture has Kh == Vh, so the golden cannot show this): the
    /// repeat_interleave launches materialize INSIDE the cached arm only
    /// — the warp-tiled and FLA arms index the compact layout directly —
    /// and every arm binds the guard's output, which the gated norm
    /// consumes as its core. The decode class under the same share
    /// states the `_gqa` step variant.
    #[test]
    fn lowered_gdn_prefill_gqa_repeats_live_inside_the_cached_arm() {
        let mut facts = Qwen35HybridFacts::qwen3_5_0_8b();
        facts.gdn.key_heads = 8; // 16 value heads sharing 8 key heads
        let cuda = Qwen35CudaFacts::qwen3_5_0_8b_synthetic();
        let plan = qwen3_5_hybrid_cuda(&facts, &cuda, FireClass::Prefill);

        let idx = plan
            .ops
            .iter()
            .position(|op| matches!(op.kind, OpKind::Guard { .. }) && op.layer == Some(0))
            .expect("layer 0 (GDN) carries the recurrence guard");
        let OpKind::Guard { arms, else_ops } = &plan.ops[idx].kind else {
            unreachable!()
        };
        assert_eq!(arms.len(), 2);
        assert_eq!(arms[0].pred, GuardPred::TokensLE(64));
        assert_eq!(arms[0].ops, 1); // warp-tiled alone
        assert_eq!(arms[1].pred, GuardPred::TokensLE(4096));
        assert_eq!(arms[1].ops, 3); // 2 repeats + cached
        assert_eq!(*else_ops, 1); // FLA alone

        let kernels: Vec<&str> = plan.ops[idx + 1..idx + 6]
            .iter()
            .map(|op| match &op.kind {
                OpKind::Launch { kernel, .. } => kernel.as_str(),
                other => panic!("guard region holds a non-launch: {other:?}"),
            })
            .collect();
        assert_eq!(
            kernels,
            [
                "ssm::chunk_gated_delta_prefill_batched_warp_tiled_gqa_state_bf16",
                "ssm::repeat_interleave_heads_fp32",
                "ssm::repeat_interleave_heads_fp32",
                "ssm::chunk_gated_delta_prefill_batched_cached_state_bf16",
                "ssm::chunk_gated_delta_prefill_batched_state_bf16",
            ]
        );
        // Region launches are output-less lowerings of the guard's value,
        // and that value is the core the gated norm consumes. The
        // REPEATS are not region launches: they are ordinary dataflow
        // that happens to live inside an arm, so they declare results of
        // their own and the cached recurrence takes them as operands —
        // which is what lets the driver stop deciding by launch order
        // which of two workspace buffers a repeat meant.
        for op in &plan.ops[idx + 1..idx + 6] {
            let OpKind::Launch { kernel, .. } = &op.kind else {
                unreachable!()
            };
            if kernel == "ssm::repeat_interleave_heads_fp32" {
                assert_eq!(op.outputs.len(), 1, "a repeat states its result: {op:?}");
                continue;
            }
            assert!(op.outputs.is_empty(), "region launch grew outputs: {op:?}");
        }
        // The cached arm's recurrence reads the REPEATED pair, not the
        // prep's compact q/k — the whole point of stating the repeats.
        let cached = &plan.ops[idx + 4];
        assert_eq!(cached.inputs[0], plan.ops[idx + 2].outputs[0]);
        assert_eq!(cached.inputs[1], plan.ops[idx + 3].outputs[0]);
        let core = plan.ops[idx].outputs[0];
        let gated = plan
            .ops
            .iter()
            .find(|op| matches!(op.kind, OpKind::RmsnormGated { .. }) && op.layer == Some(0))
            .unwrap();
        assert_eq!(gated.inputs[0], core);

        let decode = qwen3_5_hybrid_cuda(&facts, &cuda, FireClass::Decode);
        assert!(decode.ops.iter().any(|op| matches!(
            &op.kind,
            OpKind::Launch { kernel, .. }
                if kernel == "ssm::recurrent_gated_delta_step_batched_gqa_state_bf16"
        )));
    }

    /// The CUDA text NAMES both in-proj splits, and names them
    /// differently.
    ///
    /// `gdn_block_fused_binding_traces_two_splits` pins the semantic form,
    /// where both come out as one `SplitGdn` op and the executor picked a
    /// kernel by comparing the widths against `conv_dim` / `V_dim` /
    /// `V_h`. That comparison is the bug this arm exists to remove: a row
    /// split and an INTERLEAVED b/a split are different arithmetic over
    /// the same shapes, so any family whose `V_h` happened to equal its
    /// `V_dim` got whichever the comparison hit first -- silently, since
    /// both produce two tensors of the right size.
    ///
    /// So the assertion is on the two symbols being present and distinct,
    /// not on their widths. `a` is still read from the ba split's SECOND
    /// output, because the driver packs `ba` as `[b | a]` while `gdn_prep`
    /// takes `[qkv, a, b]`.
    #[test]
    fn the_cuda_fused_in_proj_states_a_row_split_and_a_ba_split_by_name() {
        let mut facts = Qwen35HybridFacts::qwen3_5_0_8b();
        facts.gdn.fused_in_proj = true;
        let cuda = Qwen35CudaFacts::qwen3_5_0_8b_synthetic();
        let plan = qwen3_5_hybrid_cuda(&facts, &cuda, FireClass::Prefill);

        let splits: Vec<&str> = plan
            .ops
            .iter()
            .filter(|op| op.layer == Some(0))
            .filter_map(|op| match &op.kind {
                OpKind::Launch { kernel, .. } if kernel.starts_with("layout::split") => {
                    Some(kernel.as_str())
                }
                _ => None,
            })
            .collect();
        assert_eq!(
            splits,
            vec!["layout::split_bf16_rows", "layout::split_qwen_gdn_ba_bf16"],
            "the fused binding states one row split for qkv/z and the \
             interleaved b/a split for the gates, in that order"
        );
        assert!(
            !plan
                .ops
                .iter()
                .any(|op| matches!(op.kind, OpKind::SplitGdn { .. })),
            "no semantic split survives into the CUDA text, or the \
             executor is back to choosing a kernel from the widths"
        );

        let ba = plan
            .ops
            .iter()
            .find(|op| {
                matches!(&op.kind, OpKind::Launch { kernel, .. }
                if kernel == "layout::split_qwen_gdn_ba_bf16")
            })
            .expect("the ba split was just asserted present");
        let prep = plan
            .ops
            .iter()
            .find(|op| matches!(op.kind, OpKind::GdnPrep { .. }) && op.layer == Some(0))
            .expect("layer 0 is the GDN layer");
        assert_eq!(prep.inputs[1], ba.outputs[1], "a is the split's second");
        assert_eq!(prep.inputs[2], ba.outputs[0], "b is the split's first");
    }

    /// A CUDA MoE with no shared expert folds the shared block away, on
    /// BOTH the aligned fast path and the general one.    ///
    /// The semantic text is already held to this; the two CUDA texts are
    /// separate statements and could drift from it independently. A text
    /// that kept the block would bind `shared_gate_up`, `shared_down` and
    /// `shared_gate` -- three weights the checkpoint of a
    /// no-shared-expert row does not contain -- and the load would fail
    /// with a missing-tensor message about a block the model does not
    /// have.
    #[test]
    fn a_cuda_moe_with_no_shared_expert_folds_the_shared_block_on_both_paths() {
        let with_shared = Qwen35MoeMlpFacts::qwen3_5_35b_a3b();
        let without = Qwen35MoeMlpFacts {
            shared_expert_intermediate: 0,
            ..with_shared.clone()
        };
        // `moe_residual_fold` picks the ALIGNED fast path over the general
        // one, and both have their own shared-expert arm.
        for aligned in [true, false] {
            let cuda = Qwen35CudaFacts {
                moe_residual_fold: aligned,
                ..Qwen35CudaFacts::qwen3_5_0_8b_synthetic()
            };
            let names = |facts: &Qwen35MoeMlpFacts| {
                qwen3_5_moe_mlp_block_cuda(facts, &cuda)
                    .ops
                    .iter()
                    .filter_map(|op| match &op.kind {
                        OpKind::Matmul { weight, .. } => Some(weight.clone()),
                        _ => None,
                    })
                    .filter(|w| w.contains("shared"))
                    .count()
            };
            assert!(
                names(&with_shared) > 0,
                "aligned={aligned}: the shared block was never stated"
            );
            assert_eq!(
                names(&without),
                0,
                "aligned={aligned}: a row with no shared expert still binds \
                 its weights"
            );
        }
    }

    /// The UNFUSED full-attention binding states three projections where
    /// the fused one states a bank and a split.
    ///
    /// The fused sibling is already held to its own shape. This is the
    /// other half of the same claim, and it is the DEFAULT: a checkpoint
    /// that ships `q_proj`/`k_proj`/`v_proj` separately is what the fused
    /// path's environment switch turns off. A text that stated the fused
    /// form anyway would name a `qgkv` bank no file contains.
    #[test]
    fn the_unfused_full_attention_binding_states_three_projections() {
        let cuda = Qwen35CudaFacts::qwen3_5_0_8b_synthetic();
        let plan = |fused_qkv| {
            let mut facts = Qwen35HybridFacts::qwen3_5_0_8b();
            facts.attn.fused_qkv = fused_qkv;
            qwen3_5_hybrid_cuda(&facts, &cuda, FireClass::Decode)
        };
        let full_attn_layer = {
            let facts = Qwen35HybridFacts::qwen3_5_0_8b();
            (0..facts.layers)
                .find(|&l| facts.is_full_attn(l))
                .expect("the hybrid has a full-attention layer")
        };
        let projections = |fused_qkv| {
            plan(fused_qkv)
                .layer_ops(full_attn_layer)
                .filter_map(|op| match &op.kind {
                    OpKind::Matmul { weight, .. } => Some(weight.clone()),
                    _ => None,
                })
                .filter(|w| w.contains("proj") || w.contains("qgkv"))
                .count()
        };
        assert_eq!(
            projections(false),
            projections(true) + 2,
            "the unfused binding did not state q, k and v separately"
        );
        assert!(
            !plan(false)
                .layer_ops(full_attn_layer)
                .any(|op| matches!(&op.kind, OpKind::SplitQkv { .. })),
            "an unfused binding has nothing to split"
        );
    }

    /// The UNFUSED dense MLP states two matmuls into the pair form, and
    /// names the two halves rather than the bank.
    ///
    /// Same claim as the attention's, one block down, and it is the
    /// binding that decides -- not the checkpoint and not the kernel. A
    /// text that stated the packed form for an unfused binding would bind
    /// a `gate_up_proj` the file does not have, and the chunked kernel
    /// would read the second half of a bank that is only half as wide.
    #[test]
    fn the_unfused_dense_mlp_states_both_halves_by_name() {
        let mut facts = Qwen35HybridFacts::qwen3_5_0_8b();
        facts.mlp = Qwen35MlpKind::Dense { intermediate: 512 };
        let weights = |gate_up_fused| {
            let cuda = Qwen35CudaFacts {
                gate_up_fused,
                ..Qwen35CudaFacts::qwen3_5_0_8b_synthetic()
            };
            qwen3_5_hybrid_cuda(&facts, &cuda, FireClass::Decode)
                .layer_ops(0)
                .filter_map(|op| match &op.kind {
                    OpKind::Matmul { weight, .. } => Some(weight.clone()),
                    _ => None,
                })
                .collect::<Vec<_>>()
        };
        let unfused = weights(false);
        let fused = weights(true);
        assert!(
            unfused.iter().any(|w| w.ends_with("gate_proj"))
                && unfused.iter().any(|w| w.ends_with("up_proj")),
            "the unfused binding did not name the two halves: {unfused:?}"
        );
        assert!(
            !unfused.iter().any(|w| w.contains("gate_up")),
            "an unfused binding named the bank anyway: {unfused:?}"
        );
        assert!(
            fused.iter().any(|w| w.contains("gate_up")),
            "the fused binding did not name the bank: {fused:?}"
        );
        assert_eq!(unfused.len(), fused.len() + 1);
    }

    /// The full-attention and hybrid traced forms survive serde — the new
    /// kinds, the partial rope, the per-head Gemma variant — and, per the
    /// additive rule, none of the new vocabulary appears in any pre-hybrid
    /// plan's serialization: the seven existing goldens stay byte-identical.
    #[test]
    fn full_attn_and_hybrid_traced_forms_round_trip() {
        for plan in [
            qwen3_5_full_attn_block(&Qwen35FullAttnFacts::qwen3_5_0_8b()),
            qwen3_5_hybrid(&Qwen35HybridFacts::qwen3_5_0_8b()),
        ] {
            let json = serde_json::to_string(&plan).unwrap();
            let back: model_ir::trace::ForwardPlan = serde_json::from_str(&json).unwrap();
            assert_eq!(plan, back);
        }

        for old in [
            serde_json::to_string(&llama_like(&LlamaLikeFacts::qwen3_0_6b())).unwrap(),
            serde_json::to_string(&llama_like(&LlamaLikeFacts::olmo2_1b())).unwrap(),
            serde_json::to_string(&qwen3_5_moe_mlp_block(&Qwen35MoeMlpFacts::qwen3_5_35b_a3b()))
                .unwrap(),
            serde_json::to_string(&qwen3_5_gdn_block(&Qwen35GdnFacts::qwen3_5_0_8b())).unwrap(),
        ] {
            for token in ["SplitQGate", "SigmoidGateMul", "partial"] {
                assert!(
                    !old.contains(token),
                    "{token} leaked into a pre-hybrid plan"
                );
            }
            // RmsnormPerHead's variant field is serde-skipped at its Plain
            // default, so pre-variant serializations carry no per-head
            // variant key (Rmsnorm's own always-present variant remains).
            assert!(!old.contains(r#""head_dim":128,"variant""#));
        }
    }

    /// The GDN vocabulary survives serde — new op kinds, rank-3 f32 values,
    /// two-name GdnPrep — and, per the additive rule, none of it (nor the
    /// dyn vocabulary) appears in a pre-GDN plan's serialization: the
    /// existing goldens stay byte-identical.
    #[test]
    fn gdn_traced_form_round_trips() {
        let plan = qwen3_5_gdn_block(&Qwen35GdnFacts::qwen3_5_0_8b());
        let json = serde_json::to_string(&plan).unwrap();
        let back: ForwardPlan = serde_json::from_str(&json).unwrap();
        assert_eq!(plan, back);

        for dense in [
            serde_json::to_string(&llama_like(&LlamaLikeFacts::qwen3_0_6b())).unwrap(),
            serde_json::to_string(&qwen3_5_moe_mlp_block(&Qwen35MoeMlpFacts::qwen3_5_35b_a3b()))
                .unwrap(),
        ] {
            for token in [
                "SplitGdn",
                "CausalConv1d",
                "GdnPrep",
                "GatedDelta",
                "RmsnormGated",
            ] {
                assert!(!dense.contains(token), "{token} leaked into a pre-GDN plan");
            }
        }
    }
}
