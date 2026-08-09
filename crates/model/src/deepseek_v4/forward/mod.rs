//! deepseek_v4's forward, declared.
//!
//! Transcribed from `deepseek_v4_forward.cpp`. Two schemes here belong to
//! no other declared family, and both change the SHAPE of the body rather
//! than a kernel inside it:
//!
//! * **Hyper-connections.** The residual is rank-K: `hc_expand` opens the
//!   body into `hc_mult` streams, each layer reads a MIX of them
//!   (`hc_pre`) and writes a mix back (`hc_post`), and `hc_head` folds
//!   them into one at the end. `y += ...` never appears in this text,
//!   which is the whole point — there is no single residual to add onto.
//!
//! * **Compressed attention.** Distant KV is compressed into per-block
//!   entries; a fire attends the sliding WINDOW uncompressed and the
//!   compressed history separately, then combines the two outputs by
//!   their LSEs. `combine_attn_outputs` is that combine, and
//!   `lse_log2_to_ln` exists because the two passes report the LSE in
//!   different bases — a unit mismatch that would be invisible in the
//!   output and catastrophic in the weighting.
//!
//! The attention sink correction rides the combined output, as in
//! gpt-oss; the difference is that here the LSE it needs is the COMBINED
//! one, which is why the correction is stated after the combine and not
//! beside either pass.

pub mod facts;

use self::facts::Dsv4Facts;
use model_compiler::dsl::{
    WeightRepr,self, matmul, MatW, NormW};
use model_compiler::trace::{FireClass, ForwardPlan, NormVariant};

struct Dsv4LayerW {
    attn_norm: NormW,
    mlp_norm: NormW,
    wq_a: MatW,
    q_norm: NormW,
    wq_b: MatW,
    wkv: MatW,
    kv_norm: NormW,
    o_a: MatW,
    o_b: MatW,
    dense_gate: MatW,
    dense_up: MatW,
    dense_down: MatW,
    router: MatW,
}

impl Dsv4LayerW {
    fn new(l: u32, f: &Dsv4Facts) -> Self {
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
            wq_a: m("wq_a", a.q_lora_rank),
            q_norm: n("q_norm"),
            wq_b: m("wq_b", a.q_width()),
            wkv: m("wkv", a.q_width()),
            kv_norm: n("kv_norm"),
            // The output projection is itself low-rank and grouped.
            o_a: m("wo_a", a.o_lora_rank),
            o_b: m("wo_b", f.hidden),
            dense_gate: m("dense_gate_proj", f.dense_intermediate),
            dense_up: m("dense_up_proj", f.dense_intermediate),
            dense_down: m("dense_down_proj", f.hidden),
            router: m("router", f.moe.num_experts),
        }
    }
}

/// deepseek_v4's CUDA text for one fire class.
pub fn dsv4_cuda(facts: &Dsv4Facts, class: FireClass) -> ForwardPlan {
    let family = format!(
        "deepseek_v4.cuda.{}",
        match class {
            FireClass::Decode => "decode",
            other => panic!("deepseek_v4 states no {other:?} class yet"),
        }
    );
    let a = facts.attn.clone();
    let k = facts.hc.mult;
    dsl::trace_named(&family, |t| {
        let embedded = dsl::embedded_prologue(t, facts.hidden);
        // The rank-K residual opens here and stays open to `hc_head`.
        let mut streams = dsl::cuda::hc_expand(&embedded, k, facts.hidden);

        // The compressed pass needs the block boundaries this fire's
        // positions imply, and they are a FIRE fact — one statement,
        // outside the layer loop, exactly as the hand-written pass has it.
        let (boundary_pos, _meta, _counts) =
            dsl::cuda::dsv4_boundary_meta_decode(&embedded);

        for l in 0..facts.layers {
            let w = Dsv4LayerW::new(l, facts);

            // Read a mix of the streams. `hc_pre` produces the layer's
            // input and the two mixes `hc_post` will need to write back.
            let normed_f32 =
                dsl::cuda::hc_rmsnorm_to_f32(&streams, &w.attn_norm.name, facts.hidden);
            let (x, post_mix, comb_mix) =
                dsl::cuda::hc_pre(&normed_f32, &streams, k, facts.hidden);

            // Q through its latent, then a per-head norm with NO gamma —
            // the reference's `q *= rsqrt(...)`, which is a different
            // statement from an rmsnorm with a weight.
            let q_a = matmul(&x, &w.wq_a);
            let q_a = dsl::cuda::rmsnorm(&q_a, &w.q_norm);
            let q = matmul(&q_a, &w.wq_b);
            let q = dsl::cuda::per_head_rmsnorm(&q, "", a.heads, a.head_dim);
            let kv = matmul(&x, &w.wkv);
            let kv = dsl::cuda::rmsnorm(&kv, &w.kv_norm);
            // Partial rope on the LAST channels of each head.
            let q = dsl::cuda::rope_partial_last(&q, a.heads, a.qk_rope_head_dim);
            let kv = dsl::cuda::rope_partial_last(&kv, a.heads, a.qk_rope_head_dim);
            dsl::seam(q.trace(), &dsl::seam::ATTN_Q, &[&q], Some(l));

            let kvh = dsl::Kv::at(t, l);
            dsl::cuda::write_kv_to_pages(&kv, &kv, &kvh);

            // The window pass, uncompressed.
            let (o_win, lse_win) = dsl::cuda::attention_flashinfer_prefill_lse(&q, &kvh, a.q_width());
            let lse_win = dsl::cuda::lse_log2_to_ln(&lse_win, a.heads);

            // The compressed pass: gather this layer's block entries,
            // rope them the same way, store them, then attend.
            let entries = dsl::cuda::dsv4_compress_gather_paged(&boundary_pos, l, a.head_dim);
            let entries = dsl::cuda::rope_partial_last(&entries, a.heads, a.qk_rope_head_dim);
            dsl::cuda::dsv4_store_comp_entries(&entries, &boundary_pos, l);
            let (o_comp, lse_comp) =
                dsl::cuda::attention_compressed_paged(&q, l, a.heads, a.head_dim);

            // One output, weighted by the two LSEs.
            let (o, lse) = dsl::cuda::combine_attn_outputs(
                &o_win,
                &lse_win,
                &o_comp,
                &lse_comp,
                a.heads,
                a.head_dim,
            );
            let o = dsl::cuda::attn_sink_correction(
                &o,
                &lse,
                &format!("layer.{l}.attn_sink"),
                a.heads,
                a.head_dim,
            );
            dsl::seam(o.trace(), &dsl::seam::ATTN_OUT, &[&o], Some(l));

            // The grouped low-rank output projection, then back into the
            // streams — never onto a single residual.
            let o = matmul(&o, &w.o_a);
            let o = matmul(&o, &w.o_b);
            streams = dsl::cuda::hc_post(&o, &streams, &post_mix, &comb_mix, k, facts.hidden);

            // ── MLP / MoE, over the same rank-K residual ─────────────
            let normed_f32 =
                dsl::cuda::hc_rmsnorm_to_f32(&streams, &w.mlp_norm.name, facts.hidden);
            let (m, post_mix, comb_mix) =
                dsl::cuda::hc_pre(&normed_f32, &streams, k, facts.hidden);

            let out = if !facts.is_moe_layer(l) {
                dsl::dense_gated_mlp(
                    &m,
                    &w.dense_gate,
                    &w.dense_up,
                    &w.dense_down,
                    facts.dense_intermediate,
                    dsl::GatedAct::SwiGluClamp,
                )
            } else {
                let logits = matmul(&m, &w.router);
                let (experts, weights) = dsl::cuda::topk_sqrtsoftplus(
                    &logits,
                    &format!("layer.{l}.router_bias"),
                    facts.moe.top_k,
                );
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
                let act = dsl::cuda::swiglu_clamp(&gate_up, facts.moe.moe_intermediate, true);
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
                dsl::cuda::weighted_sum(&weights, &route_out, facts.hidden, None)
            };
            streams = dsl::cuda::hc_post(&out, &streams, &post_mix, &comb_mix, k, facts.hidden);
        }

        // The streams fold into one.
        let y = dsl::cuda::hc_head(&streams, &streams, facts.hidden);
        dsl::logits_epilogue(t, &y, NormVariant::Plain, false, facts.vocab, false);
    })
}
