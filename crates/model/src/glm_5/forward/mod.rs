pub mod facts;

use self::facts::Glm5Facts;
use model_dsl::axes::{Bf16Ax, DtypeAxis, KvAxis, NativeKv};
use model_dsl::{self as dsl, MatW, NormW, WeightRepr, matmul};
use model_ir::trace::{DType, Dim, FireClass, ForwardPlan, NormVariant, Shape};

struct Glm5LayerW {
    attn_norm: NormW,
    mlp_norm: NormW,
    q_a_proj: MatW,
    q_a_norm: NormW,
    q_b_proj: MatW,
    kv_a_proj: MatW,
    o_proj: MatW,
    idx_wq_b: MatW,
    idx_wk: MatW,
    idx_weights: MatW,

    idx_k_norm_weight: String,
    idx_k_norm_bias: String,
    dense_gate: MatW,
    dense_up: MatW,
    dense_down: MatW,
    router: MatW,
    shared_gate: MatW,
    shared_up: MatW,
    shared_down: MatW,
}

impl Glm5LayerW {
    fn new(l: u32, f: &Glm5Facts, norm_eps: f32, repr: WeightRepr) -> Self {
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
        Self {
            attn_norm: n("attn_norm"),
            mlp_norm: n("mlp_norm"),
            q_a_proj: m("q_a_proj", a.q_lora_rank),
            q_a_norm: n("q_a_norm"),
            q_b_proj: m("q_b_proj", a.q_b_width()),
            kv_a_proj: m("kv_a_proj_with_mqa", a.kv_a_width()),
            o_proj: m("o_proj", a.hidden),
            idx_wq_b: m("idx_wq_b", f.dsa.index_n_heads * f.dsa.index_head_dim),
            idx_wk: m("idx_wk", f.dsa.index_head_dim),
            idx_weights: m("idx_weights_proj", f.dsa.index_n_heads),
            idx_k_norm_weight: w("idx_k_norm"),
            idx_k_norm_bias: w("idx_k_norm_bias"),
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

pub fn glm5_cuda<W1: DtypeAxis, W2: DtypeAxis, A: DtypeAxis, K: KvAxis>(
    facts: &Glm5Facts,
    class: FireClass,
    norm_eps: f32,
    rope_theta: f32,
) -> ForwardPlan {

    const {
        assert!(matches!(A::DTYPE, model_ir::trace::DType::BF16));
        assert!(K::NATIVE_BF16);
    }

    let family = format!(
        "glm5-{}-{}-{}.cuda.{}",
        W1::NAME,
        W2::NAME,
        K::NAME,
        class.suffix()
    );
    let a = facts.attn.clone();
    dsl::trace_named(&family, |t| {
        let mut y = dsl::embedded_prologue(t, facts.hidden, facts.vocab);

        let rt = dsl::rt(t);
        for l in 0..facts.layers {
            let w = Glm5LayerW::new(l, facts, norm_eps, W1::REPR);
            let x = dsl::cuda::rmsnorm(&y, &w.attn_norm);

            let (q_b, kv_a, q_a_n) = dsl::mla_latents(
                &x,
                None,
                &w.q_a_proj,
                &w.q_a_norm,
                &w.q_b_proj,
                &w.kv_a_proj,
                a.q_lora_rank,
            );

            let idx_q = dsl::cuda::generated::act_x_wt_bf16(
                &q_a_n,
                &w.idx_wq_b.name,
                (
                    Shape(vec![
                        Dim::Tokens,
                        Dim::Const(facts.dsa.index_n_heads * facts.dsa.index_head_dim),
                    ]),
                    DType::BF16,
                ),
                q_a_n.layer(),
                None,
            );
            let idx_k = dsl::cuda::generated::act_x_wt_bf16(
                &x,
                &w.idx_wk.name,
                (
                    Shape(vec![Dim::Tokens, Dim::Const(facts.dsa.index_head_dim)]),
                    DType::BF16,
                ),
                x.layer(),
                None,
            );
            let idx_w = dsl::cuda::generated::act_x_wt_bf16(
                &q_a_n,
                &w.idx_weights.name,
                (
                    Shape(vec![Dim::Tokens, Dim::Const(facts.dsa.index_n_heads)]),
                    DType::BF16,
                ),
                q_a_n.layer(),
                None,
            );

            let idx_k = dsl::cuda::generated::dsa_index_knorm_rope(
                &idx_k,
                &w.idx_k_norm_weight,
                &w.idx_k_norm_bias,
                a.qk_rope_head_dim as i32,
                rope_theta,
                norm_eps,
                &rt.positions(),
                idx_k.layer(),
                None,
            );
            let idx_q = dsl::cuda::generated::dsa_index_q_rope(
                &idx_q,
                facts.dsa.index_n_heads as i32,
                facts.dsa.index_head_dim as i32,
                a.qk_rope_head_dim as i32,
                rope_theta,
                &rt.positions(),
                idx_q.layer(),
                None,
            );

            let _index_mask = dsl::cuda::generated::dsa_index_topk_mask(
                &idx_q,
                &idx_k,
                &idx_w,
                (Shape(vec![Dim::Tokens, Dim::Tokens]), DType::I32),
                facts.dsa.index_n_heads as i32,
                facts.dsa.index_head_dim as i32,
                facts.dsa.index_topk as i32,
                idx_q.layer(),
                None,
            );

            let (_kv_c, _k_pe, q_nope, q_pe) = dsl::cuda::mla_prepare(
                &kv_a,
                &q_b,
                a.heads,
                a.kv_lora_rank,
                a.qk_nope_head_dim,
                a.qk_rope_head_dim,
            );
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

            y += dsl::attention_landing(&attn_v, &w.o_proj, l);

            let m = dsl::cuda::rmsnorm(&y, &w.mlp_norm);
            if !facts.is_moe_layer(l) {
                y += dsl::dense_gated_mlp(
                    &m,
                    &w.dense_gate,
                    &w.dense_up,
                    &w.dense_down,
                    facts.dense_intermediate,
                    dsl::GatedAct::SwiGlu,
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

            let gate_up = dsl::cuda::generated::moe_gate_up_decode_gemv(
                &experts,
                &m,
                &format!("layer.{l}.expert.{{e}}.gate_up"),
                (
                    Shape(vec![
                        Dim::Tokens,
                        Dim::Const(facts.moe.top_k),
                        Dim::Const(2 * facts.moe.moe_intermediate),
                    ]),
                    DType::BF16,
                ),
                Some(l),
                None,
            );
            let act = dsl::cuda::generated::chunked_swiglu(&gate_up, gate_up.layer(), None);
            let route_out = dsl::cuda::generated::moe_down_decode_gemv(
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
                Some(l),
                None,
            );
            let routed = dsl::cuda::weighted_sum(&weights, &route_out, facts.hidden, None);

            let moe_out = if facts.moe.shared_intermediate > 0 {
                let sgate = matmul(&m, &w.shared_gate);
                let sup = matmul(&m, &w.shared_up);
                let sact = dsl::cuda::generated::swiglu(&sgate, &sup, sgate.layer(), None);
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

pub type TraceFn = fn(&Glm5Facts, FireClass, f32, f32) -> ForwardPlan;

pub type ShippedW1 = Bf16Ax;

pub type ShippedW2 = Bf16Ax;

pub type ShippedA = Bf16Ax;

pub type ShippedKv = NativeKv;

pub const CATALOG: &[(&str, TraceFn)] = model_dsl::catalogue![(
    "glm5-bf16-bf16-kv-bf16",
    glm5_cuda::<ShippedW1, ShippedW2, ShippedA, ShippedKv>,
),];
