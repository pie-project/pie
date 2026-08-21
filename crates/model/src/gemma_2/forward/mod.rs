pub mod facts;

use self::facts::Gemma2Facts;
use model_dsl::axes::{Bf16Ax, DtypeAxis, KvAxis, NativeKv};
use model_dsl::{self as dsl, MatW, NormW, matmul};
use model_ir::trace::{FireClass, ForwardPlan, NormVariant};

struct G2LayerW {
    attn_norm: NormW,
    post_attn_norm: NormW,
    q_proj: MatW,
    k_proj: MatW,
    v_proj: MatW,
    o_proj: MatW,
    mlp_norm: NormW,
    post_mlp_norm: NormW,
    gate_proj: MatW,
    up_proj: MatW,
    down_proj: MatW,
}

impl G2LayerW {
    fn new(l: u32, f: &Gemma2Facts, norm_eps: f32, repr: model_dsl::WeightRepr) -> Self {
        let w = |name: &str| format!("layer.{l}.{name}");
        let m = |name: &str, width: u32| MatW {
            name: w(name),
            width,
            layer: Some(l),
            repr,
        };
        let n = |name: &str| NormW {
            name: w(name),
            variant: NormVariant::Gemma,
            per_head: None,
            layer: Some(l),
            eps: norm_eps,
        };
        Self {
            attn_norm: n("attn_norm"),
            post_attn_norm: n("post_attn_norm"),
            q_proj: m("q_proj", f.attn.q_width()),
            k_proj: m("k_proj", f.attn.kv_width()),
            v_proj: m("v_proj", f.attn.kv_width()),
            o_proj: m("o_proj", f.hidden),
            mlp_norm: n("mlp_norm"),
            post_mlp_norm: n("post_mlp_norm"),
            gate_proj: m("gate_proj", f.intermediate),
            up_proj: m("up_proj", f.intermediate),
            down_proj: m("down_proj", f.hidden),
        }
    }
}

pub fn gemma2_cuda<W1: DtypeAxis, A: DtypeAxis, K: KvAxis>(
    facts: &Gemma2Facts,
    class: FireClass,
    norm_eps: f32,
    rope_theta: f32,
) -> ForwardPlan {

    const {
        assert!(matches!(A::DTYPE, model_ir::trace::DType::BF16));
        assert!(K::NATIVE_BF16);
    }

    let family = format!("gemma_2-{}-{}.cuda.{}", W1::NAME, K::NAME, class.suffix());
    dsl::trace_named(&family, |t| {
        let embedded = dsl::embedded_prologue(t, facts.hidden, facts.vocab);

        let mut y =
            dsl::cuda::scalar_mul(&embedded, "embed_scale", Some((facts.hidden as f32).sqrt()));

        let rt = dsl::rt(t);
        for l in 0..facts.layers {

            let window_left = facts.window_left_at(l);
            let w = G2LayerW::new(l, facts, norm_eps, W1::REPR);
            let x = dsl::cuda::rmsnorm(&y, &w.attn_norm);

            let q = matmul(&x, &w.q_proj);
            let k = matmul(&x, &w.k_proj);
            let v = matmul(&x, &w.v_proj);

            let q = dsl::cuda::scalar_mul(&q, &format!("layer.{l}.query_scale"), None);

            let (q, k) = dsl::cuda::generated::rope_bf16(
                &q,
                &k,
                facts.attn.heads as i32,
                facts.attn.kv_heads as i32,
                facts.attn.head_dim as i32,
                rope_theta,
                false,
                &rt.positions(),
                q.layer(),
                None,
            );
            let kv = dsl::Kv::at(t, l);
            dsl::cuda::write_kv_to_pages(&k, &v, &kv, facts.attn.kv_heads, facts.attn.head_dim);
            dsl::seam(q.trace(), &dsl::seam::ATTN_Q, &[&q], Some(l));

            let o = dsl::cuda::attention_for(
                class,
                &q,
                &kv,
                window_left,
                facts.attn.head_dim,

                if facts.attn.attn_logit_softcap {
                    crate::gemma_2::project::ATTN_LOGIT_SOFTCAP
                } else {
                    0.0
                },
                0.0,
                facts.attn.kv_heads,
            )
            .expect("a plain attention statement produces its value");
            let o = dsl::attention_landing(&o, &w.o_proj, l);

            let o = dsl::cuda::rmsnorm(&o, &w.post_attn_norm);
            y = dsl::cuda::generated::residual_add(&y, &o, y.layer(), None);

            let m = dsl::cuda::rmsnorm(&y, &w.mlp_norm);
            let gate = matmul(&m, &w.gate_proj);
            let up = matmul(&m, &w.up_proj);
            let act = dsl::cuda::generated::geglu_tanh(&gate, &up, gate.layer(), None);
            let mlp = matmul(&act, &w.down_proj);
            let mlp = dsl::cuda::rmsnorm(&mlp, &w.post_mlp_norm);
            y = dsl::cuda::generated::residual_add(&y, &mlp, y.layer(), None);
        }

        dsl::logits_epilogue(
            t,
            &y,
            NormVariant::Gemma,
            facts.tied_embeddings,
            facts.vocab,
            facts
                .final_logit_softcap
                .then_some(crate::gemma_2::project::FINAL_LOGIT_SOFTCAP),
            norm_eps,
        );
    })
}

pub type TraceFn = fn(&Gemma2Facts, FireClass, f32, f32) -> ForwardPlan;

pub type ShippedW1 = Bf16Ax;

pub type ShippedA = Bf16Ax;

pub type ShippedKv = NativeKv;

pub const CATALOG: &[(&str, TraceFn)] = model_dsl::catalogue![(
    "gemma_2-bf16-kv-bf16",
    gemma2_cuda::<ShippedW1, ShippedA, ShippedKv>
),];
