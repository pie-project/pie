pub mod facts;
pub mod metal;

const MOE_ALIGNED_BLOCK: u32 = 16;
const MOE_MAX_BLOCKS: u32 = 1024;

use self::facts::{
    Qwen35CudaFacts, Qwen35FullAttnFacts, Qwen35GdnFacts, Qwen35HybridFacts, Qwen35MlpKind,
    Qwen35MoeMlpFacts,
};
use model_dsl::axes::{Bf16Ax, DtypeAxis, KvAxis, NativeKv};
use model_dsl::{
    self as dsl, ConvW, GdnPrepW, Kv, MatW, NormW, Rs, Trace, Val, WeightRepr, attention,
    causal_conv1d, cuda, gated_delta, gdn_prep, matmul, matmul_per_token, rmsnorm, rmsnorm_gated,
    rope_partial, sigmoid_gate_add, sigmoid_gate_mul, split_gdn, split_q_gate, split_qkv, swiglu,
    topk, weighted_sum,
};
use model_ir::trace::{
    DType, Dim, FireClass, ForwardPlan, GuardPred, NormVariant, RopeKind, Shape,
};

pub fn qwen3_5_moe_mlp_block(facts: &Qwen35MoeMlpFacts, norm_eps: f32) -> ForwardPlan {
    dsl::trace_named("qwen3_5_moe_mlp_block", |t| {
        let y = dsl::input(t, facts.hidden);
        moe_mlp_body(0, facts, &y, norm_eps);
    })
}

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
    fn new(
        l: u32,
        f: &Qwen35MoeMlpFacts,
        repr: WeightRepr,
        expert_repr: WeightRepr,
        norm_eps: f32,
    ) -> Self {
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
                eps: norm_eps,
            },
            router: mat("router", f.num_experts),

            expert_gate_up: MatW {
                repr: expert_repr,
                ..mat("expert.{e}.gate_up", 2 * f.moe_intermediate)
            },
            expert_down: MatW {
                repr: expert_repr,
                ..mat("expert.{e}.down", f.hidden)
            },
            shared_gate_up: mat("shared_expert.gate_up", 2 * f.shared_expert_intermediate),
            shared_down: mat("shared_expert.down", f.hidden).with_repr(repr),
            shared_gate: mat("shared_expert_gate", 1),
        }
    }
}

fn moe_mlp_body_aligned_cuda(
    l: u32,
    facts: &Qwen35MoeMlpFacts,
    y: &Val,
    repr: WeightRepr,
    expert_repr: WeightRepr,
    norm_eps: f32,
) -> Val {
    let w = MoeLayerW::new(l, facts, repr, expert_repr, norm_eps);
    let y = y.clone();
    let m = dsl::cuda::rmsnorm(&y, &w.mlp_norm);

    let aligned = model_ir::trace::Dim::MoeAlignedRoutes {
        top_k: facts.top_k,
        experts: facts.num_experts,
        block: MOE_ALIGNED_BLOCK,
    };

    let logits = matmul(&m, &w.router);
    let (experts, weights) = dsl::cuda::generated::topk_softmax(
        &logits,
        (
            Shape(vec![Dim::Tokens, Dim::Const(facts.top_k)]),
            DType::I32,
        ),
        (
            Shape(vec![Dim::Tokens, Dim::Const(facts.top_k)]),
            DType::F32,
        ),
        logits.layer(),
        None,
    );

    let (sorted, expert_ids, _inverse) = dsl::cuda::generated::moe_align_decode(
        &experts,
        (
            Shape(vec![Dim::Const(MOE_MAX_BLOCKS * MOE_ALIGNED_BLOCK)]),
            DType::I32,
        ),
        (Shape(vec![Dim::Const(MOE_MAX_BLOCKS)]), DType::I32),
        (
            Shape(vec![Dim::Tokens, Dim::Const(facts.top_k)]),
            DType::I32,
        ),
        facts.num_experts as i32,
        MOE_ALIGNED_BLOCK as i32,
        MOE_MAX_BLOCKS as i32,
        experts.layer(),
        None,
    );
    let aligned_in =
        dsl::cuda::gather_moe_aligned_inputs(&m, &sorted, aligned, facts.hidden, facts.top_k);

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
    let act =
        dsl::cuda::generated::chunked_swiglu_into(&gate_up, &act_stage, gate_up.layer(), None);
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

    let route_out = dsl::cuda::generated::reorder_moe_aligned_output(
        &down,
        &sorted,
        (
            Shape(vec![
                Dim::Tokens,
                Dim::Const(facts.top_k),
                Dim::Const(facts.hidden),
            ]),
            DType::BF16,
        ),
        down.layer(),
        None,
    );

    let routed = dsl::cuda::weighted_sum(&weights, &route_out, facts.hidden, Some(&y));

    if facts.shared_expert_intermediate > 0 {
        let gate_up = matmul(&m, &w.shared_gate_up);
        let act = dsl::cuda::generated::chunked_swiglu(&gate_up, gate_up.layer(), None);
        let shared = matmul(&act, &w.shared_down);
        dsl::cuda::generated::sigmoid_dot_scalar_gate_add(
            &m,
            &w.shared_gate.name,
            &routed,
            &shared,
            w.shared_gate.layer,
            None,
        )
    } else {
        routed
    }
}

fn moe_mlp_body(l: u32, facts: &Qwen35MoeMlpFacts, y: &Val, norm_eps: f32) -> Val {
    let w = MoeLayerW::new(l, facts, WeightRepr::Bf16, WeightRepr::Bf16, norm_eps);
    let mut y = y.clone();

    let m = rmsnorm(&y, &w.mlp_norm);

    let logits = matmul(&m, &w.router);
    let (experts, weights) = topk(&logits, facts.top_k);
    let gate_up = matmul_per_token(&m, &w.expert_gate_up, &experts);
    let act = swiglu(&gate_up, facts.moe_intermediate);
    let down = matmul_per_token(&act, &w.expert_down, &experts);
    let routed = weighted_sum(&weights, &down);

    let combined = if facts.shared_expert_intermediate > 0 {
        let inter = facts.shared_expert_intermediate;
        let act = swiglu(&matmul(&m, &w.shared_gate_up), inter);
        let shared = matmul(&act, &w.shared_down);
        let gate = matmul(&m, &w.shared_gate);
        sigmoid_gate_add(&shared, &gate, &routed)
    } else {
        routed
    };

    y += combined;
    y
}

pub fn qwen3_5_moe_mlp_block_cuda(
    facts: &Qwen35MoeMlpFacts,
    cuda: &Qwen35CudaFacts,
    norm_eps: f32,
) -> ForwardPlan {
    dsl::trace_named("qwen3_5_moe_mlp_block.cuda.decode", |t| {
        let y = dsl::input(t, facts.hidden);
        moe_mlp_body_cuda(
            0,
            facts,
            cuda,
            &y,
            FireClass::Decode,
            cuda.proj_repr,
            WeightRepr::Bf16,
            norm_eps,
        );
    })
}

#[allow(
    clippy::too_many_arguments,
    reason = "a body's arguments are the facts its statement reads, not a bundle waiting for a struct"
)]
fn moe_mlp_body_cuda(
    l: u32,
    facts: &Qwen35MoeMlpFacts,
    cuda: &Qwen35CudaFacts,
    y: &Val,
    class: FireClass,
    repr: WeightRepr,
    expert_repr: WeightRepr,
    norm_eps: f32,
) -> Val {
    if class != FireClass::Decode
        || cuda.moe_cutlass_max_rows == 0
        || cuda.moe_streamed_experts
        || cuda.moe_force_general
        || !cuda.moe_residual_fold
        || (facts.shared_expert_intermediate > 0 && !cuda.moe_shared_gate_dot)
    {
        return moe_mlp_body_aligned_cuda(l, facts, y, repr, expert_repr, norm_eps);
    }

    let w = MoeLayerW::new(l, facts, repr, expert_repr, norm_eps);

    let m = dsl::cuda::rmsnorm(y, &w.mlp_norm);

    let logits = matmul(&m, &w.router);
    let (experts, weights) = dsl::cuda::generated::topk_softmax(
        &logits,
        (
            Shape(vec![Dim::Tokens, Dim::Const(facts.top_k)]),
            DType::I32,
        ),
        (
            Shape(vec![Dim::Tokens, Dim::Const(facts.top_k)]),
            DType::F32,
        ),
        logits.layer(),
        None,
    );
    let routed = dsl::cuda::moe_fused_cutlass(
        &m,
        &experts,
        &weights,
        &w.expert_gate_up,
        &w.expert_down,
        facts.hidden,
    );

    let y = dsl::cuda::generated::residual_add(&routed, y, routed.layer(), None);

    if facts.shared_expert_intermediate == 0 {
        return y;
    }

    let gate_up = matmul(&m, &w.shared_gate_up);
    let act = dsl::cuda::generated::chunked_swiglu(&gate_up, gate_up.layer(), None);
    let shared = matmul(&act, &w.shared_down);
    dsl::cuda::generated::sigmoid_dot_scalar_gate_add(
        &m,
        &w.shared_gate.name,
        &y,
        &shared,
        w.shared_gate.layer,
        None,
    )
}

pub fn qwen3_5_gdn_block(facts: &Qwen35GdnFacts, norm_eps: f32) -> ForwardPlan {
    dsl::trace_named("qwen3_5_gdn_block", |t| {
        let y = dsl::input(t, facts.hidden);
        gdn_attn_body(t, 0, facts, &y, norm_eps);
    })
}

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
    fn new(t: &Trace, l: u32, f: &Qwen35GdnFacts, norm_eps: f32) -> Self {
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
                eps: norm_eps,
            },
            in_proj_qkvz: mat("in_proj_qkvz", conv_dim + v_dim),
            in_proj_ba: mat("in_proj_ba", 2 * f.value_heads),
            in_proj_qkv: mat("in_proj_qkv", conv_dim),
            in_proj_z: mat("in_proj_z", v_dim),
            in_proj_a: mat("in_proj_a", f.value_heads),
            in_proj_b: mat("in_proj_b", f.value_heads),
            conv: ConvW {
                name: w("conv"),

                bias: None,
                kernel: f.conv_kernel,
                layer: l,
            },
            prep: GdnPrepW {
                a_log: w("a_log"),
                dt_bias: w("dt_bias"),
                layer: l,
            },

            gate_norm: NormW {
                name: w("gate_norm"),
                // The ONE plain norm in this family. RMSNormGated is its own
                // class and initialises its weight to ones, where every other
                // norm here initialises to zeros and folds `1 + w` — see the
                // note on `VARIANTS`.
                variant: NormVariant::Plain,
                // RMSNormGated is a PER-HEAD norm: its weight is one row of
                // `value_head_dim` floats and the mean of the square is taken
                // across a single head's channels, not across the block's
                // whole `value_heads · value_head_dim` output. Leaving this
                // `None` said "whole row", which both reduced over sixteen
                // heads at once and walked `weight[i]` off the end of a
                // 128-float buffer.
                per_head: Some(f.value_head_dim),
                layer: Some(l),
                eps: norm_eps,
            },
            o_proj: mat("o_proj", f.hidden),
            rs: Rs::at(t, l),
        }
    }
}

fn gdn_in_proj(
    x: &Val,
    w: &GdnLayerW,
    facts: &Qwen35GdnFacts,
    stated: bool,
) -> (Val, Val, Val, Val) {
    if facts.fused_in_proj {
        let qkvz = matmul(x, &w.in_proj_qkvz);
        let (qkv, z) = if stated {
            dsl::cuda::generated::split_bf16_rows(
                &qkvz,
                (
                    Shape(vec![Dim::Tokens, Dim::Const(facts.conv_dim())]),
                    DType::BF16,
                ),
                (
                    Shape(vec![Dim::Tokens, Dim::Const(facts.value_width())]),
                    DType::BF16,
                ),
                qkvz.layer(),
                None,
            )
        } else {
            split_gdn(&qkvz, facts.conv_dim(), facts.value_width())
        };
        let ba = matmul(x, &w.in_proj_ba);
        let (b, a) = if stated {
            dsl::cuda::generated::split_qwen_gdn_ba(&ba, ba.layer(), None)
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

fn gdn_attn_body(t: &Trace, l: u32, facts: &Qwen35GdnFacts, y: &Val, norm_eps: f32) -> Val {
    let w = GdnLayerW::new(t, l, facts, norm_eps);
    let mut y = y.clone();

    let x = rmsnorm(&y, &w.attn_norm);

    let (qkv, z, a, b) = gdn_in_proj(&x, &w, facts, false);

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
        facts.conv_dim(),
    );
    let core = gated_delta(&w.rs, &q, &k, &v, &g, &beta);

    let o = rmsnorm_gated(&core, &z, &w.gate_norm);
    y += matmul(&o, &w.o_proj);
    y
}

fn gdn_attn_body_cuda(
    t: &Trace,
    l: u32,
    facts: &Qwen35GdnFacts,
    y: &Val,
    c: &Qwen35CudaFacts,
    class: FireClass,
    norm_eps: f32,
) -> Val {
    let w = GdnLayerW::new(t, l, facts, norm_eps);
    let mut y = y.clone();

    let x = dsl::cuda::rmsnorm(&y, &w.attn_norm);

    let (qkv, z, a, b) = gdn_in_proj(&x, &w, facts, true);

    let conv_geom = cuda::GdnShape {
        k_heads: facts.key_heads,
        v_heads: facts.value_heads,
        k_dim: facts.key_head_dim,
        v_dim: facts.value_head_dim,
        conv_dim: facts.conv_dim(),
        conv_k: facts.conv_kernel,
    };
    let rt = dsl::rt(qkv.trace());
    let qkv = match class {
        FireClass::Decode => cuda::generated::causal_conv1d_update_batched(
            &qkv,
            &w.conv.name,
            w.conv.bias.as_deref(),
            conv_geom.conv_dim as i32,
            conv_geom.conv_k as i32,
            &w.rs.view(),
            Some(w.conv.layer),
            Some(w.rs.state()),
        ),

        FireClass::Prefill => {
            let qo_indptr = rt.qo_indptr();
            cuda::generated::causal_conv1d_prefill_batched(
                &qkv,
                &w.conv.name,
                w.conv.bias.as_deref(),
                conv_geom.conv_dim as i32,
                conv_geom.conv_k as i32,
                &w.rs.view(),
                true,
                &qo_indptr,
                Some(w.conv.layer),
                Some(w.rs.state()),
            )
        }
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
        facts.conv_dim(),
    );

    dsl::seam(q.trace(), &dsl::seam::ATTN_Q, &[&q], Some(l));

    let gqa = facts.value_heads != facts.key_heads;

    let geom = cuda::GdnShape {
        k_heads: facts.key_heads,
        v_heads: facts.value_heads,
        k_dim: facts.key_head_dim,
        v_dim: facts.value_head_dim,
        conv_dim: facts.conv_dim(),
        conv_k: facts.conv_kernel,
    };
    let core = match class {
        FireClass::Decode => {
            cuda::gdn_step_batched(&q, &k, &v, &g, &beta, &w.rs, gqa, c.state_bf16, geom)
        }
        FireClass::Prefill => {
            let out_shape = (
                Shape(vec![
                    Dim::Tokens,
                    Dim::Const(facts.value_heads),
                    Dim::Const(facts.value_head_dim),
                ]),
                DType::F32,
            );

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
                                    geom,
                                    true,
                                );
                            },
                        );
                    }
                    ctx.arm(dsl::Region::Fire(GuardPred::TokensLE(c.cached_max)), || {
                        if gqa {
                            let repeat_shape = || {
                                (
                                    Shape(vec![
                                        Dim::Tokens,
                                        Dim::Const(facts.value_heads),
                                        Dim::Const(facts.key_head_dim),
                                    ]),
                                    DType::F32,
                                )
                            };
                            let qr = cuda::generated::repeat_interleave_heads_fp32(
                                &q,
                                repeat_shape(),
                                facts.key_heads as i32,
                                facts.value_heads as i32,
                                facts.key_head_dim as i32,
                                q.layer(),
                                None,
                            );
                            let kr = cuda::generated::repeat_interleave_heads_fp32(
                                &k,
                                repeat_shape(),
                                facts.key_heads as i32,
                                facts.value_heads as i32,
                                facts.key_head_dim as i32,
                                k.layer(),
                                None,
                            );
                            cuda::gdn_prefill_cached(
                                &qr,
                                &kr,
                                &v,
                                &g,
                                &beta,
                                &w.rs,
                                c.state_bf16,
                                geom,
                                true,
                            );
                        } else {
                            cuda::gdn_prefill_cached(
                                &q,
                                &k,
                                &v,
                                &g,
                                &beta,
                                &w.rs,
                                c.state_bf16,
                                geom,
                                true,
                            );
                        }
                    });
                },
                || {
                    cuda::gdn_prefill_fla(&q, &k, &v, &g, &beta, &w.rs, c.state_bf16, geom, true);
                },
            )
            .expect("the guarded recurrence produces its value")
        }
    };

    dsl::seam(q.trace(), &dsl::seam::ATTN_OUT, &[&q], Some(l));

    let o = rmsnorm_gated(&core, &z, &w.gate_norm);
    y += matmul(&o, &w.o_proj);
    y
}

pub fn qwen3_5_full_attn_block(
    facts: &Qwen35FullAttnFacts,
    norm_eps: f32,
    rope_theta: f32,
) -> ForwardPlan {
    dsl::trace_named("qwen3_5_full_attn_block", |t| {
        let y = dsl::input(t, facts.hidden);
        full_attn_body(t, 0, facts, &y, norm_eps, rope_theta);
    })
}

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
    fn new(t: &Trace, l: u32, f: &Qwen35FullAttnFacts, repr: WeightRepr, norm_eps: f32) -> Self {
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

        let qk_norm = |name: &str| NormW {
            name: w(name),
            variant: f.norm_variant,
            per_head: Some(f.head_dim),
            layer: Some(l),
            eps: norm_eps,
        };
        FullAttnLayerW {
            attn_norm: NormW {
                name: w("attn_norm"),
                variant: f.norm_variant,
                per_head: None,
                layer: Some(l),
                eps: norm_eps,
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

fn full_attn_body(
    t: &Trace,
    l: u32,
    facts: &Qwen35FullAttnFacts,
    y: &Val,
    norm_eps: f32,
    rope_theta: f32,
) -> Val {
    let w = FullAttnLayerW::new(t, l, facts, WeightRepr::Bf16, norm_eps);
    let mut y = y.clone();

    let x = rmsnorm(&y, &w.attn_norm);

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

    let q = rmsnorm(&q, &w.q_norm);
    let k = rmsnorm(&k, &w.k_norm);
    let (q, k) = rope_partial(
        &q,
        &k,
        RopeKind::Standard,
        facts.rotary_dim,
        facts.head_dim,
        rope_theta,
    );
    w.kv.append(&k, &v);

    let attn = attention(&q, &w.kv, facts.q_width());
    let gated = sigmoid_gate_mul(&attn, &gate);
    y += matmul(&gated, &w.o_proj);
    y
}

#[allow(
    clippy::too_many_arguments,
    reason = "a body's arguments are the facts its statement reads, not a bundle waiting for a struct"
)]
fn full_attn_body_cuda(
    t: &Trace,
    l: u32,
    facts: &Qwen35FullAttnFacts,
    c: &Qwen35CudaFacts,
    y: &Val,
    class: FireClass,
    repr: WeightRepr,
    norm_eps: f32,
    rope_theta: f32,
) -> Val {
    let w = FullAttnLayerW::new(t, l, facts, repr, norm_eps);

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
    let rt = dsl::rt(q.trace());
    let (q, k) = dsl::cuda::generated::rope_partial_bf16(
        &q,
        &k,
        facts.rotary_dim as i32,
        facts.head_dim as i32,
        rope_theta,
        &rt.positions(),
        q.layer(),
        None,
    );

    dsl::seam(q.trace(), &dsl::seam::ATTN_Q, &[&q], Some(l));

    dsl::regions(
        t,
        None,
        None,
        |c| {
            c.arm(dsl::Region::Fire(GuardPred::HasWriteDesc), || {
                cuda::generated::write_kv_explicit_bf16(
                    &k,
                    &v,
                    &w.kv.cache(),
                    facts.kv_heads as i32,
                    facts.head_dim as i32,
                    &rt.row_valid(),
                    Some(w.kv.l),
                    Some(w.kv.state()),
                );
            });
        },
        || {
            cuda::write_kv_to_pages(&k, &v, &w.kv, facts.kv_heads, facts.head_dim);
        },
    );

    let attn = match class {
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
                        cuda::attention_flashinfer_prefill(
                            &dsl::runtime::query_windows(&q),
                            &w.kv,
                            window_left,
                            facts.head_dim,
                            0.0,
                            0.0,
                        );
                    });
                },
                || {
                    cuda::attention_flashinfer_decode(
                        &q,
                        &w.kv,
                        window_left,
                        facts.head_dim,
                        0.0,
                        0.0,
                    );
                },
            )
        }
        FireClass::Decode => {
            cuda::attention_flashinfer_decode(&q, &w.kv, window_left, facts.head_dim, 0.0, 0.0)
        }
        FireClass::Prefill => cuda::attention_flashinfer_prefill(
            &dsl::runtime::query_windows(&q),
            &w.kv,
            window_left,
            facts.head_dim,
            0.0,
            0.0,
        ),
    };
    let attn = attn.expect("a plain attention statement produces its value");
    let gated = sigmoid_gate_mul(&attn, &gate);

    dsl::seam(q.trace(), &dsl::seam::ATTN_OUT, &[&q], Some(l));
    y += matmul(&gated, &w.o_proj);
    y
}

fn dense_mlp_body(
    l: u32,
    hidden: u32,
    intermediate: u32,
    variant: NormVariant,
    y: &Val,
    norm_eps: f32,
) -> Val {
    let w = |name: &str| format!("layer.{l}.{name}");
    let mlp_norm = NormW {
        name: w("mlp_norm"),
        variant,
        per_head: None,
        layer: Some(l),
        eps: norm_eps,
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

#[allow(
    clippy::too_many_arguments,
    reason = "a body's arguments are the facts its statement reads, not a bundle waiting for a struct"
)]
fn dense_mlp_body_cuda(
    l: u32,
    hidden: u32,
    intermediate: u32,
    variant: NormVariant,
    y: &Val,
    packed: bool,
    repr: WeightRepr,
    norm_eps: f32,
) -> Val {
    let w = |name: &str| format!("layer.{l}.{name}");
    let mlp_norm = NormW {
        name: w("mlp_norm"),
        variant,
        per_head: None,
        layer: Some(l),
        eps: norm_eps,
    };

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

    let act = if packed {
        let gu = matmul(&m, &gate_up);
        dsl::cuda::generated::chunked_swiglu(&gu, gu.layer(), None)
    } else {
        let gate = matmul(&m, &half("gate_proj"));
        let up = matmul(&m, &half("up_proj"));
        dsl::cuda::generated::swiglu(&gate, &up, gate.layer(), None)
    };
    y += matmul(&act, &down);
    y
}

pub fn qwen3_5_hybrid(facts: &Qwen35HybridFacts, norm_eps: f32, rope_theta: f32) -> ForwardPlan {
    let hidden = hybrid_hidden(facts);
    dsl::trace_named("qwen3_5_hybrid", |t| {
        dsl::seam(t, &dsl::seam::IN, &[], None);
        let mut y = dsl::embed_with(t, "embed", hidden, facts.vocab);

        for l in 0..facts.layers {
            let y_attn = if facts.is_full_attn(l) {
                full_attn_body(t, l, &facts.attn, &y, norm_eps, rope_theta)
            } else {
                gdn_attn_body(t, l, &facts.gdn, &y, norm_eps)
            };
            y = match &facts.mlp {
                Qwen35MlpKind::Dense { intermediate } => dense_mlp_body(
                    l,
                    hidden,
                    *intermediate,
                    facts.norm_variant,
                    &y_attn,
                    norm_eps,
                ),
                Qwen35MlpKind::Moe(moe) => moe_mlp_body(l, moe, &y_attn, norm_eps),
            };
        }

        hybrid_epilogue(t, facts, &y, false, norm_eps);
    })
}

pub fn qwen3_5_hybrid_cuda<W1: DtypeAxis, W2: DtypeAxis, A: DtypeAxis, K: KvAxis>(
    facts: &Qwen35HybridFacts,
    cuda: &Qwen35CudaFacts,
    class: FireClass,
    norm_eps: f32,
    rope_theta: f32,
) -> ForwardPlan {
    const {
        assert!(matches!(A::DTYPE, model_ir::trace::DType::BF16));
        assert!(K::NATIVE_BF16);
    }
    let hidden = hybrid_hidden(facts);

    let family = format!(
        "qwen3_5_hybrid-{}-{}-{}.cuda.{}",
        W1::NAME,
        W2::NAME,
        K::NAME,
        match class {
            FireClass::Decode => "decode",
            FireClass::Prefill => "prefill",
        }
    );
    dsl::trace_named(&family, |t| {
        dsl::seam(t, &dsl::seam::IN, &[], None);
        let mut y = dsl::embed_with(t, "embed", hidden, facts.vocab);

        for l in 0..facts.layers {
            let y_attn = if facts.is_full_attn(l) {
                full_attn_body_cuda(
                    t,
                    l,
                    &facts.attn,
                    cuda,
                    &y,
                    class,
                    W1::REPR,
                    norm_eps,
                    rope_theta,
                )
            } else {
                gdn_attn_body_cuda(t, l, &facts.gdn, &y, cuda, class, norm_eps)
            };
            y = match &facts.mlp {
                Qwen35MlpKind::Dense { intermediate } => dense_mlp_body_cuda(
                    l,
                    hidden,
                    *intermediate,
                    facts.norm_variant,
                    &y_attn,
                    cuda.gate_up_fused,
                    W1::REPR,
                    norm_eps,
                ),
                Qwen35MlpKind::Moe(moe) => {
                    moe_mlp_body_cuda(l, moe, cuda, &y_attn, class, W1::REPR, W2::REPR, norm_eps)
                }
            };
        }

        hybrid_epilogue(t, facts, &y, true, norm_eps);
    })
}

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

fn hybrid_epilogue(t: &Trace, facts: &Qwen35HybridFacts, y: &Val, stated: bool, norm_eps: f32) {
    let final_norm = NormW {
        name: "final_norm".to_string(),
        variant: facts.norm_variant,
        per_head: None,
        layer: None,
        eps: norm_eps,
    };
    let normed = if stated {
        dsl::cuda::rmsnorm(y, &final_norm)
    } else {
        rmsnorm(y, &final_norm)
    };
    let logits = dsl::lm_head_tied(t, &normed, facts.tied_embeddings, facts.vocab);
    dsl::seam(t, &dsl::seam::OUT, &[&logits], None);
}

pub type TraceFn = fn(&Qwen35HybridFacts, &Qwen35CudaFacts, FireClass, f32, f32) -> ForwardPlan;

pub type ShippedW1 = Bf16Ax;

pub type ShippedW2 = Bf16Ax;

pub type ShippedA = Bf16Ax;

pub type ShippedKv = NativeKv;

pub const CATALOG: &[(&str, TraceFn)] = model_dsl::catalogue![(
    "qwen3_5_hybrid-bf16-bf16-kv-bf16",
    qwen3_5_hybrid_cuda::<ShippedW1, ShippedW2, ShippedA, ShippedKv>,
),];
