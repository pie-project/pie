//! gemma-2's forward, declared.
//!
//! Transcribed from `gemma/gemma2.cpp`. The last family that existed only
//! as hand-written C++, and the simplest — which makes the three things
//! that ARE gemma's easy to see:
//!
//! * **A norm PAIR around each block.** Every other declared family
//!   norms before its block and folds the residual after. gemma norms
//!   before AND after, then adds — so `post_attn_norm` and
//!   `post_mlp_norm` are statements, and the residual add is its own
//!   launch rather than a `beta=1` fold.
//!
//! * **An alternating window.** Layers take turns attending the whole
//!   context and a 4096-token suffix. It is a per-layer LIST, not an
//!   interval, because that is what the driver reads.
//!
//! * **Two softcaps, and only one is a launch.** The attention logit cap
//!   is a DISPATCH parameter — the kernel takes it — so nothing states
//!   it. The final one is `launch_logit_softcap_bf16`, at the end. A
//!   text that stated both as launches would claim a kernel the fire
//!   never runs.
//!
//! The embedding is scaled once, before the layers: `sqrt(hidden)` folded
//! into a `scalar_mul`, which is a launch and says so.

pub mod facts;

use self::facts::Gemma2Facts;
use model_compiler::dsl::{
    WeightRepr,self, matmul, MatW, NormW};
use model_compiler::trace::{FireClass, ForwardPlan, NormVariant};

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
    fn new(l: u32, f: &Gemma2Facts) -> Self {
        let w = |name: &str| format!("layer.{l}.{name}");
        let m = |name: &str, width: u32| MatW {
            name: w(name),
            width,
            layer: Some(l),
            repr: WeightRepr::Bf16,
        };
        let n = |name: &str| NormW {
            name: w(name),
            variant: NormVariant::Gemma,
            per_head: None,
            layer: Some(l),
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

/// gemma-2's CUDA text for one fire class.
pub fn gemma2_cuda(facts: &Gemma2Facts, class: FireClass) -> ForwardPlan {
    let family = format!(
        "gemma_2.cuda.{}",
        match class {
            FireClass::Decode => "decode",
            other => panic!("gemma_2 states no {other:?} class yet"),
        }
    );
    let a = facts.attn.clone();
    dsl::trace_named(&family, |t| {
        let embedded = dsl::embedded_prologue(t, facts.hidden);
        // `sqrt(hidden)` on the embedding — a launch, not a fold.
        let mut y = dsl::cuda::scalar_mul(
            &embedded,
            "embed_scale",
            Some((facts.hidden as f32).sqrt()),
        );

        for l in 0..facts.layers {
            // THIS LAYER's sliding window, `-1` for none — a
            // load-time fact the dispatch statements carry, where four
            // executors used to re-derive it per launch.
            let window_left =
                model_compiler::facts::window_left_at(&facts.window_left, l);
            let w = G2LayerW::new(l, facts);
            let x = dsl::cuda::rmsnorm(&y, &w.attn_norm);

            let q = matmul(&x, &w.q_proj);
            let k = matmul(&x, &w.k_proj);
            let v = matmul(&x, &w.v_proj);
            let (q, k) = if a.qk_norm {
                let qn = NormW {
                    name: format!("layer.{l}.q_norm"),
                    variant: NormVariant::Gemma,
                    per_head: Some(a.head_dim),
                    layer: Some(l),
                };
                let kn = NormW {
                    name: format!("layer.{l}.k_norm"),
                    variant: NormVariant::Gemma,
                    per_head: Some(a.head_dim),
                    layer: Some(l),
                };
                (dsl::cuda::rmsnorm(&q, &qn), dsl::cuda::rmsnorm(&k, &kn))
            } else {
                (q, k)
            };
            // The pre-attention query scale is its OWN launch.
            let q = if a.query_pre_attn_scale {
                dsl::cuda::scalar_mul(&q, &format!("layer.{l}.query_scale"), None)
            } else {
                q
            };
            let (q, k) = dsl::cuda::rope(&q, &k);
            let kv = dsl::Kv::at(t, l);
            dsl::cuda::write_kv_to_pages(&k, &v, &kv);
            dsl::seam(q.trace(), &dsl::seam::ATTN_Q, &[&q], Some(l));
            // The attention logit softcap rides HERE, as a dispatch
            // parameter — see the module doc on why it is not a
            // statement of its own.
            let o = dsl::cuda::attention_flashinfer_decode(&q, &kv, window_left)
                .expect("a plain attention statement produces its value");
            let o = dsl::attention_landing(&o, &w.o_proj, l);
            // The POST norm, then an explicit add — gemma's pair.
            let o = dsl::cuda::rmsnorm(&o, &w.post_attn_norm);
            y = dsl::cuda::residual_add(&y, &o, facts.hidden);

            let m = dsl::cuda::rmsnorm(&y, &w.mlp_norm);
            let gate = matmul(&m, &w.gate_proj);
            let up = matmul(&m, &w.up_proj);
            let act = dsl::cuda::geglu_tanh_pair(&gate, &up, facts.intermediate);
            let mlp = matmul(&act, &w.down_proj);
            let mlp = dsl::cuda::rmsnorm(&mlp, &w.post_mlp_norm);
            y = dsl::cuda::residual_add(&y, &mlp, facts.hidden);
        }

        dsl::logits_epilogue(
            t,
            &y,
            NormVariant::Gemma,
            facts.tied_embeddings,
            facts.vocab,
            facts.final_logit_softcap,
        );
    })
}
