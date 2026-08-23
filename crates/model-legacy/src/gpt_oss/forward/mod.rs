pub mod facts;

use self::facts::{GptOssCudaFacts, GptOssFacts};
use model_dsl::axes::{Bf16Ax, DtypeAxis, KvAxis, Mxfp4Ax, NativeKv};
use model_dsl::{self as dsl, MatW, NormW, Val, WeightRepr, matmul};
use model_ir::trace::{DType, Dim, FireClass, ForwardPlan, NormVariant, Shape};

struct GptOssLayerW {
    attn_norm: NormW,
    q_proj: MatW,
    k_proj: MatW,
    v_proj: MatW,
    q_bias: MatW,
    k_bias: MatW,
    v_bias: MatW,
    o_proj: MatW,
    o_bias: MatW,
    sinks: MatW,
    mlp_norm: NormW,
    router: MatW,
    router_bias: MatW,
    expert_gate_up: MatW,
    expert_down: MatW,
}

impl GptOssLayerW {
    fn new(
        l: u32,
        f: &GptOssFacts,
        norm_eps: f32,
        repr: WeightRepr,
        expert_repr: WeightRepr,
    ) -> Self {
        let w = |name: &str| format!("layer.{l}.{name}");
        let m = |name: &str, width: u32| MatW {
            name: w(name),
            width,
            layer: Some(l),
            repr,
        };
        let d = f.head_dim;
        Self {
            attn_norm: NormW {
                name: w("attn_norm"),
                variant: NormVariant::Plain,
                per_head: None,
                layer: Some(l),
                eps: norm_eps,
            },
            q_proj: m("q_proj", f.q_heads * d),
            k_proj: m("k_proj", f.kv_heads * d),
            v_proj: m("v_proj", f.kv_heads * d),
            q_bias: m("q_bias", f.q_heads * d),
            k_bias: m("k_bias", f.kv_heads * d),
            v_bias: m("v_bias", f.kv_heads * d),
            o_proj: m("o_proj", f.hidden),
            o_bias: m("o_bias", f.hidden),
            sinks: m("attn_sinks", f.q_heads),
            mlp_norm: NormW {
                name: w("mlp_norm"),
                variant: NormVariant::Plain,
                per_head: None,
                layer: Some(l),
                eps: norm_eps,
            },
            router: m("router", f.experts),
            router_bias: m("router_bias", f.experts),

            expert_gate_up: MatW {
                repr: expert_repr,
                ..m("expert_gate_up_bank", f.intermediate)
            },
            expert_down: MatW {
                repr: expert_repr,
                ..m("expert_down_bank", f.hidden)
            },
        }
    }
}

pub fn gpt_oss_cuda<W1: DtypeAxis, W2: DtypeAxis, A: DtypeAxis, K: KvAxis>(
    facts: &GptOssFacts,
    cuda: &GptOssCudaFacts,
    class: FireClass,
    norm_eps: f32,
    rope_theta: f32,
    sliding_window: i32,
) -> ForwardPlan {
    const {
        assert!(matches!(A::DTYPE, model_ir::trace::DType::BF16));
        assert!(K::NATIVE_BF16);
        assert!(matches!(W2::REPR, WeightRepr::Mxfp4Marlin));
    }

    let crate::deployment::RopeScaling::Yarn {
        factor: yarn_factor,
        beta_fast: yarn_beta_fast,
        beta_slow: yarn_beta_slow,
        attention_factor: yarn_attention_factor,
        original_max_position: yarn_original_max_position,
        ..
    } = crate::gpt_oss::ROPE_SCALING
    else {
        unreachable!("gpt-oss rescales by YaRN; the row's constant says so")
    };
    assert!(
        cuda.mxfp4_decode_gemv,
        "gpt_oss states the fused MXFP4 decode leg; a deployment without \
         the per-expert pointer arrays reaches the experts by a host walk \
         this declaration refuses"
    );
    assert!(
        !cuda.streamed_experts,
        "gpt_oss states the resident bank; a streamed one reaches the same \
         kernels only after a host round-trip that decides what to page in"
    );

    let family = format!(
        "gpt_oss-{}-{}-{}.cuda.{}",
        W1::NAME,
        W2::NAME,
        K::NAME,
        class.suffix()
    );
    let hidden = facts.hidden;
    dsl::trace_named(&family, |t| {
        dsl::seam(t, &dsl::seam::IN, &[], None);
        let mut y = dsl::embed_with(t, "embed", hidden, facts.vocab);

        let rt = dsl::rt(t);
        for l in 0..facts.layers {
            let w = GptOssLayerW::new(l, facts, norm_eps, W1::REPR, W2::REPR);
            let kv = dsl::Kv::at(t, l);
            let normed = dsl::cuda::rmsnorm(&y, &w.attn_norm);

            let proj = |x: &Val, w: &MatW, b: &MatW| {
                dsl::cuda::generated::act_x_wt_bias_bf16(
                    x,
                    &w.name,
                    &b.name,
                    (Shape(vec![Dim::Tokens, Dim::Const(w.width)]), DType::BF16),
                    w.layer,
                    None,
                )
            };
            let q = proj(&normed, &w.q_proj, &w.q_bias);
            let k = proj(&normed, &w.k_proj, &w.k_bias);
            let v = proj(&normed, &w.v_proj, &w.v_bias);

            let (q, k) = dsl::cuda::generated::rope_yarn_original_bf16(
                &q,
                &k,
                facts.head_dim as i32,
                rope_theta,
                yarn_factor,
                yarn_beta_fast,
                yarn_beta_slow,
                yarn_attention_factor,
                yarn_original_max_position as i32,
                false,
                &rt.positions(),
                q.layer(),
                None,
            );
            dsl::cuda::write_kv_to_pages(&k, &v, &kv, facts.kv_heads, facts.head_dim);

            dsl::seam(q.trace(), &dsl::seam::ATTN_Q, &[&q], Some(l));

            let window_left = if facts.is_sliding(l) {
                sliding_window
            } else {
                -1
            };
            let (o, lse) = dsl::cuda::attention_for_lse(
                class,
                &q,
                &kv,
                facts.q_heads,
                facts.head_dim,
                facts.kv_heads,
                window_left,
                0.0,
                0.0,
            );
            let a = dsl::cuda::generated::attention_sink_rescale(
                &o,
                &lse,
                &w.sinks.name,
                facts.q_heads as i32,
                facts.head_dim as i32,
                w.sinks.layer,
                None,
            );

            dsl::seam(a.trace(), &dsl::seam::ATTN_OUT, &[&a], Some(l));

            y += matmul(&a, &w.o_proj);
            y = dsl::add_bias(&y, &w.o_bias);

            let mlp_in = dsl::cuda::rmsnorm(&y, &w.mlp_norm);
            let logits = proj(&mlp_in, &w.router, &w.router_bias);
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

            let act = dsl::cuda::generated::bf16_to_fp16(
                &mlp_in,
                (Shape(vec![Dim::Tokens, Dim::Const(hidden)]), DType::F16),
                mlp_in.layer(),
                None,
            );
            let routed_shape = || {
                (
                    Shape(vec![
                        Dim::Tokens,
                        Dim::Const(facts.top_k),
                        Dim::Const(facts.intermediate),
                    ]),
                    DType::BF16,
                )
            };
            let (gate, up) = dsl::cuda::generated::mxfp4_moe_gate_up_decode_bf16(
                &experts,
                &act,
                &w.expert_gate_up.name,
                routed_shape(),
                routed_shape(),
                facts.swiglu_limit,
                crate::gpt_oss::project::GATE_ALPHA,
                &rt.object("moe.expert_weights", w.expert_gate_up.layer),
                w.expert_gate_up.layer,
                None,
            );

            assert!(
                facts.swiglu_limit > 0.0,
                "gpt_oss without a swiglu limit states no activation yet"
            );
            let routed = dsl::cuda::generated::gpt_oss_glu(
                &gate,
                &up,
                facts.swiglu_limit,
                crate::gpt_oss::project::GATE_ALPHA,
                gate.layer(),
                None,
            );
            let routed = dsl::cuda::generated::bf16_to_fp16(
                &routed,
                (
                    Shape(vec![
                        Dim::Tokens,
                        Dim::Const(facts.top_k),
                        Dim::Const(facts.intermediate),
                    ]),
                    DType::F16,
                ),
                routed.layer(),
                None,
            );
            let out = dsl::cuda::generated::mxfp4_moe_down_decode_bf16(
                &experts,
                &routed,
                &w.expert_down.name,
                (
                    Shape(vec![
                        Dim::Tokens,
                        Dim::Const(facts.top_k),
                        Dim::Const(hidden),
                    ]),
                    DType::BF16,
                ),
                &rt.object("moe.expert_weights", w.expert_down.layer),
                w.expert_down.layer,
                None,
            );

            let combined = dsl::cuda::weighted_sum(&weights, &out, hidden, None);

            y = dsl::cuda::generated::residual_add(&y, &combined, y.layer(), None);
        }

        dsl::logits_epilogue(
            t,
            &y,
            NormVariant::Plain,
            facts.tied_embeddings,
            facts.vocab,
            None,
            norm_eps,
        );
    })
}

pub type TraceFn = fn(&GptOssFacts, &GptOssCudaFacts, FireClass, f32, f32, i32) -> ForwardPlan;

pub type ShippedW1 = Bf16Ax;

pub type ShippedW2 = Mxfp4Ax;

pub type ShippedA = Bf16Ax;

pub type ShippedKv = NativeKv;

pub const CATALOG: &[(&str, TraceFn)] = model_dsl::catalogue![(
    "gpt_oss-bf16-mxfp4-kv-bf16",
    gpt_oss_cuda::<ShippedW1, ShippedW2, ShippedA, ShippedKv>,
),];
