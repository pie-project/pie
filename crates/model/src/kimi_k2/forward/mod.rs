//! kimi's forward, declared.
//!
//! Transcribed from `driver-cuda/csrc/src/model/kimi/kimi_forward.cpp`.
//! The second MLA family; what differs from [`crate::glm5`] is worth
//! naming, because the shapes are otherwise the same statement:
//!
//! * **No DSA.** kimi's attention reads the whole context; there is no
//!   lightning indexer and no page mask.
//!
//! * **The latents may be ONE projection.** With `q_kv_a_fused` bound,
//!   `[q_lora | kv_lora | rope]` land row-major in one buffer, so the
//!   query half is normed with a STRIDED kernel rather than a plain one
//!   — neither latent is a contiguous block of the result. That is a
//!   BINDING fact, so it is a fact here and both readings are stated.
//!
//! * **The experts are WNA16.** The decode leg is
//!   `launch_wna16_{gate_up,down}_decode_bf16` over packed weights and
//!   scales, with `bf16_to_fp16` casts on the activation either side —
//!   the kernel reads fp16. Same rectangle shape as glm5's GEMV leg,
//!   different kernels, and the casts are real launches that a text
//!   omitting them would be wrong about.

pub mod facts;

use self::facts::{KimiCudaFacts, KimiFacts};
use model_compiler::dsl::{
    WeightRepr,self, matmul, MatW, NormW};
use model_compiler::trace::{FireClass, ForwardPlan, NormVariant};

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
    fn new(l: u32, f: &KimiFacts) -> Self {
        let w = |name: &str| format!("layer.{l}.{name}");
        let m = |name: &str, width: u32| MatW {
            name: w(name),
            width,
            layer: Some(l),
            repr: WeightRepr::Bf16,
        };
        let n = |name: &str| NormW {
            name: w(name),
            variant: NormVariant::Plain,
            per_head: None,
            layer: Some(l),
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

/// kimi's CUDA text for one fire class.
pub fn kimi_cuda(facts: &KimiFacts, cuda: &KimiCudaFacts, class: FireClass) -> ForwardPlan {
    let family = format!(
        "kimi.cuda.{}",
        match class {
            FireClass::Decode => "decode",
            other => panic!("kimi states no {other:?} class yet"),
        }
    );
    let a = facts.attn.clone();
    dsl::trace_named(&family, |t| {
        dsl::seam(t, &dsl::seam::IN, &[], None);
        let mut y = dsl::embed_with(t, "embed", facts.hidden);

        for l in 0..facts.layers {
            let w = KimiLayerW::new(l, facts);
            let x = dsl::cuda::rmsnorm(&y, &w.attn_norm);

            // The two latents, fused or not. The FUSED arm norms the
            // query half in place with a pitch, which is a different
            // kernel and not a buffer detail — `kernels::norm::rmsnorm_strided_bf16`
            // reads a row stride the plain one has no parameter for.
            let (q_a_n, kv_a) = if cuda.q_kv_a_fused {
                let qkv_a = matmul(&x, &w.q_kv_a);
                // The pitch is the fused width; the statement carries the
                // NARROW extent it produces, and the stride is the buffer
                // question  owns.
                let q_a_n =
                    dsl::cuda::rmsnorm_strided(&qkv_a, &w.q_a_norm.name, a.q_lora_rank);
                (q_a_n, qkv_a)
            } else {
                let q_a = matmul(&x, &w.q_a_proj);
                (dsl::cuda::rmsnorm(&q_a, &w.q_a_norm), matmul(&x, &w.kv_a_proj))
            };
            let q_b = matmul(&q_a_n, &w.q_b_proj);

            let (_kv_c, _k_pe, q_nope, q_pe) = dsl::cuda::mla_prepare(
                &kv_a,
                &q_b,
                a.heads,
                a.kv_lora_rank,
                a.qk_nope_head_dim,
                a.qk_rope_head_dim,
            );
            let kv_b = format!("layer.{l}.kv_b_proj");
            let q_latent =
                dsl::cuda::mla_absorb_q_to_latent(&q_nope, &kv_b, a.heads, a.kv_lora_rank);
            let attn_latent =
                dsl::cuda::attention_mla(&q_latent, &q_pe, l, a.heads, a.kv_lora_rank);
            let attn_v =
                dsl::cuda::mla_absorb_latent_to_v(&attn_latent, &kv_b, a.heads, a.v_head_dim);
            dsl::seam(attn_v.trace(), &dsl::seam::ATTN_OUT, &[&attn_v], Some(l));
            y += matmul(&attn_v, &w.o_proj);

            let m = dsl::cuda::rmsnorm(&y, &w.mlp_norm);
            if !facts.is_moe_layer(l) {
                let gate = matmul(&m, &w.dense_gate);
                let _up = matmul(&m, &w.dense_up);
                let act = dsl::cuda::swiglu(&gate, facts.dense_intermediate, false);
                y += matmul(&act, &w.dense_down);
                continue;
            }

            let logits = matmul(&m, &w.router);
            let (experts, weights) = dsl::cuda::topk_sigmoid(&logits, facts.moe.top_k);

            // WNA16: the kernel reads fp16, so the cast either side is a
            // real launch. Omitting it would make the text claim a
            // dtype the deployment never has at that point.
            let m_fp16 = dsl::cuda::bf16_to_fp16(&m);
            let (gate, up) = dsl::cuda::wna16_gate_up_decode(
                &m_fp16,
                &experts,
                facts.moe.moe_intermediate,
                &format!("layer.{l}.experts"),
            );
            let _ = up;
            let act = dsl::cuda::swiglu(&gate, facts.moe.moe_intermediate, false);
            let act_fp16 = dsl::cuda::bf16_to_fp16(&act);
            let route_out =
                dsl::cuda::wna16_down_decode(
                    &act_fp16,
                    &experts,
                    facts.hidden,
                    &format!("layer.{l}.experts"),
                );
            let routed = dsl::cuda::weighted_sum(&weights, &route_out, facts.hidden, None);

            let moe_out = if facts.moe.shared_intermediate > 0 {
                let sgate = matmul(&m, &w.shared_gate);
                let _sup = matmul(&m, &w.shared_up);
                let sact = dsl::cuda::swiglu(&sgate, facts.moe.shared_intermediate, false);
                let shared = matmul(&sact, &w.shared_down);
                dsl::cuda::residual_add(&routed, &shared, facts.hidden)
            } else {
                routed
            };
            y = dsl::cuda::residual_add(&y, &moe_out, facts.hidden);
        }

        let normed = dsl::cuda::rmsnorm(
            &y,
            &NormW {
                name: "final_norm".to_string(),
                variant: NormVariant::Plain,
                per_head: None,
                layer: None,
            },
        );
        let logits = dsl::lm_head_at(t, &normed, "lm_head", facts.vocab);
        dsl::seam(t, &dsl::seam::OUT, &[&logits], None);
    })
}
