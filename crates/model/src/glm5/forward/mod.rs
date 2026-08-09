//! glm5's forward, declared.
//!
//! Transcribed from `driver-cuda/csrc/src/model/glm5/glm5_forward.cpp`,
//! which is the only description of this family that has ever run. Three
//! things about the reading are worth stating before the body, because
//! each is a place a transcription could go wrong quietly:
//!
//! * **The MLA prepare is FUSED.** `kernels::attn::mla_prepare_bf16` does what
//!   the four launches beside it do (`kimi_split_kv_a_norm`,
//!   `kimi_split_q_b`, `rope`, `write_mla_to_pages`), and the driver
//!   takes it whenever `mla_prepare_supported(q_rope)` holds. This text
//!   states the fused form. The four-launch arm is not a guard's other
//!   half: it is a different rectangle count for a rope width the fused
//!   kernel refuses, which makes it a different declaration.
//!
//! * **The DSA indexer produces no activation the layer reads.** Its
//!   output is a page mask the attention dispatch takes as a sideband.
//!   It is stated anyway: a reader following the dataflow has to see
//!   where the mask comes from, and its three projections are real
//!   weights a binding must resolve.
//!
//! * **The decode MoE leg is the GEMV one.** `kGlm5MoeGemvMaxTokens` is
//!   1, so a decode fire takes `launch_moe_{gate_up,down}_decode_gemv_bf16`
//!   and never the aligned or CUTLASS legs. Stating the leg a decode fire
//!   ACTUALLY takes is the same rule qwen3_5's text follows.

pub mod facts;

use self::facts::Glm5Facts;
use model_compiler::dsl::{
    WeightRepr,self, matmul, MatW, NormW};
use model_compiler::trace::{FireClass, ForwardPlan, NormVariant};

/// One glm5 layer's weight handles, under the tree-wide
/// `layer.{l}.{field}` convention every executor's `parse_name` reads.
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
    dense_gate: MatW,
    dense_up: MatW,
    dense_down: MatW,
    router: MatW,
    shared_gate: MatW,
    shared_up: MatW,
    shared_down: MatW,
}

impl Glm5LayerW {
    fn new(l: u32, f: &Glm5Facts) -> Self {
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
            q_a_proj: m("q_a_proj", a.q_lora_rank),
            q_a_norm: n("q_a_norm"),
            q_b_proj: m("q_b_proj", a.q_b_width()),
            kv_a_proj: m("kv_a_proj_with_mqa", a.kv_a_width()),
            o_proj: m("o_proj", a.hidden),
            idx_wq_b: m("idx_wq_b", f.dsa.index_n_heads * f.dsa.index_head_dim),
            idx_wk: m("idx_wk", f.dsa.index_head_dim),
            idx_weights: m("idx_weights_proj", f.dsa.index_n_heads),
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

/// glm5's CUDA text for one fire class.
pub fn glm5_cuda(facts: &Glm5Facts, class: FireClass) -> ForwardPlan {
    let family = format!(
        "glm5.cuda.{}",
        match class {
            FireClass::Decode => "decode",
            other => panic!("glm5 states no {other:?} class yet"),
        }
    );
    let a = facts.attn.clone();
    dsl::trace_named(&family, |t| {
        let mut y = dsl::embedded_prologue(t, facts.hidden);

        for l in 0..facts.layers {
            let w = Glm5LayerW::new(l, facts);
            let x = dsl::cuda::rmsnorm(&y, &w.attn_norm);

            // The query's own latent, normed, then expanded. `hidden`
            // appears nowhere between here and `o_proj` — that is what
            // makes this MLA rather than a wide attention.
            let (q_b, kv_a, q_a_n) = dsl::mla_latents(
                &x,
                None,
                &w.q_a_proj,
                &w.q_a_norm,
                &w.q_b_proj,
                &w.kv_a_proj,
                a.q_lora_rank,
            );

            // ── DSA lightning indexer ────────────────────────────────
            // A second, smaller attention whose only product is a top-k
            // page mask. `gemm_xwt` rather than `matmul`: these three
            // weights are named directly rather than through a layer
            // field the binder resolves by role.
            let idx_q = dsl::cuda::gemm_xwt(
                &q_a_n,
                &w.idx_wq_b.name,
                facts.dsa.index_n_heads * facts.dsa.index_head_dim,
            );
            let idx_k = dsl::cuda::gemm_xwt(&x, &w.idx_wk.name, facts.dsa.index_head_dim);
            let idx_w =
                dsl::cuda::gemm_xwt(&q_a_n, &w.idx_weights.name, facts.dsa.index_n_heads);
            let idx_k = dsl::cuda::dsa_index_knorm_rope(&idx_k, facts.dsa.index_head_dim);
            let idx_q = dsl::cuda::dsa_index_q_rope(
                &idx_q,
                facts.dsa.index_n_heads,
                facts.dsa.index_head_dim,
            );
            dsl::cuda::dsa_index_topk_mask(&idx_q, &idx_k, &idx_w, facts.dsa.index_n_heads, facts.dsa.index_head_dim, facts.dsa.index_topk);

            // ── MLA ──────────────────────────────────────────────────
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
            // The OnAttn site: after the core, before `o_proj` — the
            // hand-written invoke's position.
            // `+=` of a fresh matmul IS the beta=1 fold the T==1 arm makes.
            y += dsl::attention_landing(&attn_v, &w.o_proj, l);

            // ── MLP / MoE ────────────────────────────────────────────
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
            let (experts, weights) = dsl::cuda::topk_sigmoid(&logits, facts.moe.top_k);

            // The decode leg: one warp per output row, because at M == 1
            // the routed GEMMs are streaming reads with no weight reuse.
            let gate_up = dsl::cuda::moe_gate_up_gemv(
                &m,
                &MatW {
                    name: format!("layer.{l}.expert.{{e}}.gate_up"),
                    width: 2 * facts.moe.moe_intermediate,
                    layer: Some(l),
                    repr: WeightRepr::Bf16,
                },
                &experts,
                facts.moe.top_k,
            );
            let act = dsl::cuda::swiglu(&gate_up, facts.moe.moe_intermediate, true);
            let route_out = dsl::cuda::moe_down_gemv(
                &act,
                &MatW {
                    name: format!("layer.{l}.expert.{{e}}.down"),
                    width: facts.hidden,
                    layer: Some(l),
                    repr: WeightRepr::Bf16,
                },
                &experts,
                facts.moe.top_k,
            );
            let routed =
                dsl::cuda::weighted_sum(&weights, &route_out, facts.hidden, None);

            // The shared expert lands on the routed sum, and the SUM
            // lands on the residual — two explicit adds, because glm5's
            // `moe_out` is scratch and nothing folded into it.
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

        dsl::logits_epilogue(t, &y, NormVariant::Plain, false, facts.vocab, false);
    })
}
