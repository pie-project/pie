pub mod facts;

use self::facts::KimiK3Facts;
use model_dsl::axes::{Bf16Ax, DtypeAxis, KvAxis, Mxfp4Ax, NativeKv};
use model_dsl::{self as dsl, MatW, NormW, WeightRepr, matmul};
use model_ir::trace::{DType, Dim, FireClass, ForwardPlan, NormVariant, Shape};

const SITU_BETA: f32 = 1.0;

const SITU_LINEAR_BETA: f32 = 0.0;

struct K3LayerW {
    attn_norm: NormW,
    mlp_norm: NormW,

    q_a_proj: MatW,
    q_a_norm: NormW,
    q_b_proj: MatW,
    kv_a_proj: MatW,
    o_proj: MatW,

    kda_q: MatW,
    kda_k: MatW,
    kda_v: MatW,
    kda_f_a: MatW,
    kda_f_b: MatW,
    kda_b: MatW,
    kda_g: MatW,
    kda_o: MatW,
    kda_o_norm: NormW,

    dense_gate: MatW,
    dense_up: MatW,
    dense_down: MatW,
    router: MatW,
    shared_gate: MatW,
    shared_up: MatW,
    shared_down: MatW,
}

impl K3LayerW {
    fn new(l: u32, f: &KimiK3Facts, norm_eps: f32, repr: WeightRepr) -> Self {
        let w = |name: &str| format!("layer.{l}.{name}");
        let m = |name: &str, width: u32| MatW {
            name: w(name),
            width,
            layer: Some(l),
            repr,
        };
        let n = |name: &str| NormW {
            name: w(name),
            variant: NormVariant::Plain,
            per_head: None,
            layer: Some(l),
            eps: norm_eps,
        };
        let a = &f.attn;
        let k = &f.kda;
        Self {
            attn_norm: n("attn_norm"),
            mlp_norm: n("mlp_norm"),
            q_a_proj: m("q_a_proj", a.q_lora_rank),
            q_a_norm: n("q_a_norm"),
            q_b_proj: m("q_b_proj", a.q_b_width()),
            kv_a_proj: m("kv_a_proj_with_mqa", a.kv_a_width()),
            o_proj: m("o_proj", a.hidden),
            kda_q: m("kda_q_proj", k.width()),
            kda_k: m("kda_k_proj", k.width()),
            kda_v: m("kda_v_proj", k.width()),
            kda_f_a: m("kda_f_a_proj", k.value_head_dim),
            kda_f_b: m("kda_f_b_proj", k.width()),
            kda_b: m("kda_b_proj", k.value_heads),
            kda_g: m("kda_g_proj", k.width()),
            kda_o: m("kda_o_proj", f.hidden),
            kda_o_norm: n("kda_o_norm"),
            dense_gate: m("dense_gate_proj", f.dense_intermediate),
            dense_up: m("dense_up_proj", f.dense_intermediate),
            dense_down: m("dense_down_proj", f.hidden),
            router: m("router", f.moe.num_experts),
            shared_gate: m("shared_expert.gate", f.moe.shared_intermediate),
            shared_up: m("shared_expert.up", f.moe.shared_intermediate),
            shared_down: m("shared_expert.down", f.hidden),
        }
    }
}

pub fn kimi_k3_cuda<W1: DtypeAxis, W2: DtypeAxis, A: DtypeAxis, K: KvAxis>(
    facts: &KimiK3Facts,
    class: FireClass,
    norm_eps: f32,
) -> ForwardPlan {

    const {
        assert!(matches!(A::DTYPE, model_ir::trace::DType::BF16));
        assert!(K::NATIVE_BF16);
        assert!(matches!(W2::REPR, WeightRepr::Mxfp4Marlin));
    }

    let family = format!(
        "kimi_k3-{}-{}-{}.cuda.{}",
        W1::NAME,
        W2::NAME,
        K::NAME,
        class.suffix()
    );
    let a = facts.attn.clone();
    let kd = facts.kda.clone();
    dsl::trace_named(&family, |t| {
        let mut y = dsl::embedded_prologue(t, facts.hidden, facts.vocab);

        let rt = dsl::rt(t);
        for l in 0..facts.layers {
            let w = K3LayerW::new(l, facts, norm_eps, W1::REPR);

            if facts.attn_res_block > 0 && l > 0 && l % facts.attn_res_block == 0 {
                y = dsl::cuda::generated::attn_res_blend(
                    &y,
                    &y,
                    &format!("layer.{l}.attn_res_norm"),
                    &format!("layer.{l}.attn_res_proj"),
                    norm_eps,
                    y.layer(),
                    None,
                );
            }
            let x = dsl::cuda::rmsnorm(&y, &w.attn_norm);

            if facts.is_full_attn(l) {
                let (q_b, kv_a, _q_a_n) = dsl::mla_latents(
                    &x,
                    None,
                    &w.q_a_proj,
                    &w.q_a_norm,
                    &w.q_b_proj,
                    &w.kv_a_proj,
                    a.q_lora_rank,
                );

                let (kv_c, k_pe) = dsl::cuda::generated::kimi_split_kv_a_norm(
                    &kv_a,
                    &format!("layer.{l}.kv_a_norm"),
                    (
                        Shape(vec![Dim::Tokens, Dim::Const(a.kv_lora_rank)]),
                        DType::BF16,
                    ),
                    (
                        Shape(vec![Dim::Tokens, Dim::Const(a.qk_rope_head_dim)]),
                        DType::BF16,
                    ),
                    norm_eps,
                    kv_a.layer(),
                    None,
                );
                let (q_nope, q_pe) = dsl::cuda::generated::kimi_split_q_b(
                    &q_b,
                    (
                        Shape(vec![
                            Dim::Tokens,
                            Dim::Const(a.heads),
                            Dim::Const(a.qk_nope_head_dim),
                        ]),
                        DType::BF16,
                    ),
                    (
                        Shape(vec![
                            Dim::Tokens,
                            Dim::Const(a.heads),
                            Dim::Const(a.qk_rope_head_dim),
                        ]),
                        DType::BF16,
                    ),
                    a.heads as i32,
                    a.qk_nope_head_dim as i32,
                    a.qk_rope_head_dim as i32,
                    q_b.layer(),
                    None,
                );
                dsl::cuda::write_mla_to_pages(&kv_c, &k_pe, l);
                let attn_v = dsl::mla_absorbed_attention(
                    &q_nope,
                    &q_pe,
                    &format!("layer.{l}.kv_b_proj"),
                    l,
                    dsl::MlaWidths {
                        heads: a.heads,
                        kv_lora_rank: a.kv_lora_rank,
                        qk_nope_head_dim: a.qk_nope_head_dim,
                        v_head_dim: a.v_head_dim,
                    },
                );

                assert!(
                    !a.output_gate,
                    "kimi_k3: `mla_output_gate` is not stated yet — the \
                     semantic SigmoidGateMul wants equal Shapes and MLA's \
                     absorb is rank-3. See this arm's comment."
                );
                y += dsl::attention_landing(&attn_v, &w.o_proj, l);
            } else {

                let q = matmul(&x, &w.kda_q);
                let k = matmul(&x, &w.kda_k);
                let v = matmul(&x, &w.kda_v);

                let rs = dsl::Rs::at(t, l);
                let conv = |x: &dsl::Val, name: &str| {

                    dsl::cuda::generated::causal_conv1d_update_batched(
                        x,
                        &format!("layer.{l}.{name}"),
                        None,
                        kd.value_heads as i32,
                        kd.conv_kernel as i32,
                        &rs.view(),
                        Some(l),
                        Some(rs.state()),
                    )
                };
                let q = conv(&q, "kda_q_conv");
                let k = conv(&k, "kda_k_conv");
                let v = conv(&v, "kda_v_conv");

                let f_a = matmul(&x, &w.kda_f_a);
                let f_b = matmul(&f_a, &w.kda_f_b);
                let b = matmul(&x, &w.kda_b);
                let (gate, beta) = dsl::cuda::generated::kda_gate_beta(
                    &f_b,
                    &b,
                    &format!("layer.{l}.kda_a_log"),
                    &format!("layer.{l}.kda_dt_bias"),
                    (
                        Shape(vec![
                            Dim::Tokens,
                            Dim::Const(kd.value_heads * kd.value_head_dim),
                        ]),
                        DType::F32,
                    ),
                    (
                        Shape(vec![Dim::Tokens, Dim::Const(kd.value_heads)]),
                        DType::F32,
                    ),
                    kd.value_head_dim as i32,
                    f_b.layer(),
                    None,
                );
                let q = dsl::cuda::generated::l2norm_scale_bf16_to_fp32(
                    &q,
                    (Shape(vec![Dim::Tokens, Dim::Const(kd.width())]), DType::F32),
                    kd.norm_eps(),
                    q.layer(),
                    None,
                );
                let k = dsl::cuda::generated::l2norm_scale_bf16_to_fp32(
                    &k,
                    (Shape(vec![Dim::Tokens, Dim::Const(kd.width())]), DType::F32),
                    kd.norm_eps(),
                    k.layer(),
                    None,
                );
                let v = dsl::cuda::generated::bf16_to_fp32(
                    &v,
                    (Shape(vec![Dim::Tokens, Dim::Const(kd.width())]), DType::F32),
                    v.layer(),
                    None,
                );
                let core = dsl::cuda::generated::kda_recurrent_step_batched(
                    &q,
                    &k,
                    &v,
                    &gate,
                    &beta,
                    (
                        Shape(vec![
                            Dim::Requests,
                            Dim::Const(kd.value_heads),
                            Dim::Const(kd.value_head_dim),
                        ]),
                        DType::F32,
                    ),
                    kd.value_heads as i32,
                    kd.value_head_dim as i32,
                    &rs.view(),
                    Some(l),
                    Some(rs.state()),
                );
                let g = matmul(&x, &w.kda_g);
                let o = dsl::cuda::generated::kda_o_norm_gated(
                    &core,
                    &g,
                    &w.kda_o_norm.name,
                    kd.value_heads as i32,
                    kd.value_head_dim as i32,
                    kd.norm_eps(),
                    core.layer(),
                    None,
                );
                dsl::seam(o.trace(), &dsl::seam::ATTN_OUT, &[&o], Some(l));
                y += matmul(&o, &w.kda_o);
            }

            let m = dsl::cuda::rmsnorm(&y, &w.mlp_norm);
            if !facts.is_moe_layer(l) {
                y += dsl::dense_gated_mlp(
                    &m,
                    &w.dense_gate,
                    &w.dense_up,
                    &w.dense_down,
                    facts.dense_intermediate,
                    dsl::GatedAct::Situ {
                        beta: SITU_BETA,
                        linear_beta: SITU_LINEAR_BETA,
                    },
                );
                continue;
            }

            let logits = matmul(&m, &w.router);
            let (experts, weights) = dsl::cuda::generated::topk_sigmoid(
                &logits,
                (
                    Shape(vec![Dim::Tokens, Dim::Const(facts.moe.top_k)]),
                    DType::I32,
                ),
                (
                    Shape(vec![Dim::Tokens, Dim::Const(facts.moe.top_k)]),
                    DType::F32,
                ),
                None,
                facts.moe.norm_topk_prob,
                facts.moe.routed_scaling,
                logits.layer(),
                None,
            );
            let routed_shape = || {
                (
                    Shape(vec![
                        Dim::Tokens,
                        Dim::Const(facts.moe.top_k),
                        Dim::Const(facts.moe.moe_intermediate),
                    ]),
                    DType::BF16,
                )
            };
            let (gate_up, _up) = dsl::cuda::generated::mxfp4_moe_gate_up_decode_bf16(
                &experts,
                &m,
                &format!("layer.{l}.expert.{{e}}.gate_up"),
                routed_shape(),
                routed_shape(),

                0.0,
                0.0,
                &rt.object("moe.expert_weights", Some(l)),
                Some(l),
                None,
            );
            let act = dsl::cuda::generated::chunked_situ(
                &gate_up,
                SITU_BETA,
                SITU_LINEAR_BETA,
                gate_up.layer(),
                None,
            );
            let route_out = dsl::cuda::generated::mxfp4_moe_down_decode_bf16(
                &experts,
                &act,
                &format!("layer.{l}.expert.{{e}}.down"),
                (
                    Shape(vec![
                        Dim::Tokens,
                        Dim::Const(facts.moe.top_k),
                        Dim::Const(facts.hidden),
                    ]),
                    DType::BF16,
                ),
                &rt.object("moe.expert_weights", Some(l)),
                Some(l),
                None,
            );
            let routed = dsl::cuda::weighted_sum(&weights, &route_out, facts.hidden, None);

            let moe_out = if facts.moe.shared_intermediate > 0 {
                let sgate = matmul(&m, &w.shared_gate);
                let sup = matmul(&m, &w.shared_up);
                let sact = dsl::cuda::generated::situ(
                    &sgate,
                    &sup,
                    SITU_BETA,
                    SITU_LINEAR_BETA,
                    sgate.layer(),
                    None,
                );
                let shared = matmul(&sact, &w.shared_down);
                dsl::cuda::generated::residual_add(&routed, &shared, routed.layer(), None)
            } else {
                routed
            };
            y = dsl::cuda::generated::residual_add(&y, &moe_out, y.layer(), None);
        }

        dsl::logits_epilogue(
            t,
            &y,
            NormVariant::Plain,
            false,
            facts.vocab,
            None,
            norm_eps,
        );
    })
}

pub type TraceFn = fn(&KimiK3Facts, FireClass, f32) -> ForwardPlan;

pub type ShippedW1 = Bf16Ax;

pub type ShippedW2 = Mxfp4Ax;

pub type ShippedA = Bf16Ax;

pub type ShippedKv = NativeKv;

pub const CATALOG: &[(&str, TraceFn)] = model_dsl::catalogue![(
    "kimi_k3-bf16-mxfp4-kv-bf16",
    kimi_k3_cuda::<ShippedW1, ShippedW2, ShippedA, ShippedKv>,
),];
