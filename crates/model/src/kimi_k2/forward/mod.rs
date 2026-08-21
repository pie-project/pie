pub mod facts;

use self::facts::{KimiCudaFacts, KimiFacts};
use model_dsl::axes::{Bf16Ax, DtypeAxis, KvAxis, NativeKv, Wna16Ax};
use model_dsl::{self as dsl, MatW, NormW, WeightRepr, matmul};
use model_ir::trace::{DType, Dim, FireClass, ForwardPlan, NormVariant, Shape};

struct KimiLayerW {
    attn_norm: NormW,
    mlp_norm: NormW,
    q_kv_a: MatW,
    q_a_proj: MatW,
    kv_a_proj: MatW,
    q_a_norm: NormW,
    q_b_proj: MatW,
    o_proj: MatW,
    dense_gate: MatW,
    dense_up: MatW,
    dense_down: MatW,
    router: MatW,
    shared_gate: MatW,
    shared_up: MatW,
    shared_down: MatW,
}

impl KimiLayerW {
    fn new(l: u32, f: &KimiFacts, norm_eps: f32, repr: WeightRepr) -> Self {
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
            q_kv_a: m("q_kv_a_fused", a.q_kv_a_width()),
            q_a_proj: m("q_a_proj", a.q_lora_rank),
            kv_a_proj: m("kv_a_proj_with_mqa", a.kv_a_width()),
            q_a_norm: n("q_a_norm"),
            q_b_proj: m("q_b_proj", a.q_b_width()),
            o_proj: m("o_proj", a.hidden),
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

pub fn kimi_cuda<W1: DtypeAxis, W2: DtypeAxis, A: DtypeAxis, K: KvAxis>(
    facts: &KimiFacts,
    cuda: &KimiCudaFacts,
    class: FireClass,
    norm_eps: f32,
) -> ForwardPlan {

    const {
        assert!(matches!(A::DTYPE, model_ir::trace::DType::BF16));
        assert!(K::NATIVE_BF16);
    }

    let wna16_group = const {
        match W2::REPR {
            WeightRepr::Scaled { group, .. } => group,
            _ => panic!("kimi's routed experts are WNA16; W2 must be a Scaled axis"),
        }
    };

    let family = format!(
        "kimi-{}-{}-{}.cuda.{}",
        W1::NAME,
        W2::NAME,
        K::NAME,
        class.suffix()
    );
    let a = facts.attn.clone();
    dsl::trace_named(&family, |t| {
        let mut y = dsl::embedded_prologue(t, facts.hidden, facts.vocab);

        for l in 0..facts.layers {
            let w = KimiLayerW::new(l, facts, norm_eps, W1::REPR);
            let x = dsl::cuda::rmsnorm(&y, &w.attn_norm);

            let (q_b, kv_a, _q_a_n) = dsl::mla_latents(
                &x,
                cuda.q_kv_a_fused.then_some(&w.q_kv_a),
                &w.q_a_proj,
                &w.q_a_norm,
                &w.q_b_proj,
                &w.kv_a_proj,
                a.q_lora_rank,
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

            let m_fp16 = dsl::cuda::generated::bf16_to_fp16(
                &m,
                (
                    Shape(vec![Dim::Tokens, Dim::Const(facts.hidden)]),
                    DType::F16,
                ),
                m.layer(),
                None,
            );
            let expert_shape = || {
                (
                    Shape(vec![Dim::Tokens, Dim::Const(facts.moe.moe_intermediate)]),
                    DType::BF16,
                )
            };
            let bank = format!("layer.{l}.experts");
            let (gate, up) = dsl::cuda::generated::wna16_gate_up_decode_bf16(
                &m_fp16,
                &experts,
                &format!("{bank}.gate_packed"),
                &format!("{bank}.gate_scale"),
                &format!("{bank}.up_packed"),
                &format!("{bank}.up_scale"),
                expert_shape(),
                expert_shape(),
                wna16_group as i32,
                m_fp16.layer(),
                None,
            );
            let act = dsl::cuda::generated::swiglu(&gate, &up, gate.layer(), None);
            let act_fp16 = dsl::cuda::generated::bf16_to_fp16(
                &act,
                (
                    Shape(vec![Dim::Tokens, Dim::Const(facts.moe.moe_intermediate)]),
                    DType::F16,
                ),
                act.layer(),
                None,
            );
            let route_out = dsl::cuda::generated::wna16_down_decode_bf16(
                &act_fp16,
                &experts,
                &format!("{bank}.down_packed"),
                &format!("{bank}.down_scale"),
                (
                    Shape(vec![Dim::Tokens, Dim::Const(facts.hidden)]),
                    DType::BF16,
                ),
                wna16_group as i32,
                act_fp16.layer(),
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

pub type TraceFn = fn(&KimiFacts, &KimiCudaFacts, FireClass, f32) -> ForwardPlan;

pub type ShippedW1 = Bf16Ax;

pub type ShippedW2 = Wna16Ax;

pub type ShippedA = Bf16Ax;

pub type ShippedKv = NativeKv;

pub const CATALOG: &[(&str, TraceFn)] = model_dsl::catalogue![(
    "kimi-bf16-wna16-kv-bf16",
    kimi_cuda::<ShippedW1, ShippedW2, ShippedA, ShippedKv>,
),];
