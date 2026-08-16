//! `gpt-oss`.

pub mod facts;

use self::facts::{GptOssCudaFacts, GptOssFacts};
use model_dsl::{self as dsl, MatW, NormW, Val, WeightRepr, matmul};
use model_ir::trace::{FireClass, ForwardPlan, NormVariant};

// ── gpt-oss ────────────────────────────────────────────────────────────

/// One gpt-oss layer's weight handles. The family rides `mixtral.cpp`,
/// so these are that file's names.
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
    fn new(l: u32, f: &GptOssFacts) -> Self {
        // `layer.{l}.{field}` — the tree-wide convention every executor's
        // `parse_name` reads. Naming them bare made the drive's first live
        // fire throw on the very first weight it looked up.
        let w = |name: &str| format!("layer.{l}.{name}");
        let m = |name: &str, width: u32| MatW {
            name: w(name),
            width,
            layer: Some(l),
            repr: WeightRepr::Bf16,
        };
        let d = f.head_dim;
        Self {
            attn_norm: NormW {
                name: w("attn_norm"),
                variant: NormVariant::Plain,
                per_head: None,
                layer: Some(l),
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
            },
            router: m("router", f.experts),
            router_bias: m("router_bias", f.experts),
            expert_gate_up: m("expert_gate_up_bank", f.intermediate),
            expert_down: m("expert_down_bank", f.hidden),
        }
    }
}

/// gpt-oss's CUDA text — the DECODE class, and the first family whose
/// MoE block is stated end to end.
///
/// Mirrors `mixtral.cpp::mixtral_forward_paged` at `tp_size == 1`, which
/// is what `bind_gpt_oss` returns weights for: gpt-oss has no forward of
/// its own, only a binder.
///
/// Three things here exist in no other family's text.
///
/// **The sink.** gpt-oss learns a per-head logit that joins the softmax
/// denominator without contributing a value. flashinfer's
/// DefaultAttention will not emit it, so the driver asks the dispatch
/// for its LSE and rescales the output by `sigmoid(lse - sink)`. The
/// declaration says this by having the attention statement produce TWO
/// values on a sink-carrying layer — the `lse_out` argument is the whole
/// difference, and the symbol does not change.
///
/// **The MXFP4 routed leg.** The expert weights are never materialized:
/// two GEMVs read the packed nibbles out of HBM and index the experts
/// through a device pointer array. That leg is SEVEN rectangles and no
/// host sync. The alternative — a serial per-expert walk whose launch
/// count depends on what the router picked, behind a D2H that drains the
/// stream — is refused by name below, not stated.
///
/// **The clamped GLU.** `swiglu_limit` is a config constant, so gpt-oss
/// states a different activation kernel rather than passing a limit.
///
/// Yarn was NOT stated here at first, and deliberately: the config asked
/// for it while `mixtral.cpp` passed a plain `rope_theta`, so declaring
/// it would have made this text disagree with the pass it mirrors. The
/// fix went to that line instead, and the fact followed it — which is the
/// order that keeps a declaration honest about a driver bug rather than
/// laundering one.
pub fn gpt_oss_cuda(facts: &GptOssFacts, cuda: &GptOssCudaFacts, class: FireClass) -> ForwardPlan {
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
    let family = format!("gpt_oss.cuda.{}", class.suffix());
    let hidden = facts.hidden;
    dsl::trace_named(&family, |t| {
        // The entry boundary, where device puts and channel reads attach.
        dsl::seam(t, &dsl::seam::IN, &[], None);
        let mut y = dsl::embed_with(t, "embed", hidden);

        for l in 0..facts.layers {
            // THIS LAYER's sliding window, `-1` for none — a
            // load-time fact the dispatch statements carry, where four
            // executors used to re-derive it per launch.
            let w = GptOssLayerW::new(l, facts);
            let kv = dsl::Kv::at(t, l);
            let normed = dsl::cuda::rmsnorm(&y, &w.attn_norm);

            // The q/k/v biases FOLD INTO the projection's epilogue
            // (`kernels::gemm::act_x_wt_bias_bf16`): at decode these route to the
            // warp-per-row GEMV, which absorbs the bias for free. Stating
            // them as separate AddBias ops — which this text did until a
            // census of its own golden was read against the driver — is
            // three extra launches per layer and a different accumulation
            // order, and nothing that only asks whether the trace LOWERS
            // would have said so.
            // Every gpt-oss biases q/k/v/o, the router and the experts, so
            // every projection here is the fused epilogue and not a matmul.
            let proj = |x: &Val, w: &MatW, b: &MatW| dsl::cuda::gemm_bias(x, w, b);
            let q = proj(&normed, &w.q_proj, &w.q_bias);
            let k = proj(&normed, &w.k_proj, &w.k_bias);
            let v = proj(&normed, &w.v_proj, &w.v_bias);

            // NO q/v adapter seam here, and that is the honest answer
            // rather than a limitation being hidden.
            //
            // `attn.qv`'s position rule is that the correction lands on the
            // BASE projection, not on base + bias. This family folds the
            // bias into the GEMM's own epilogue
            // (`kernels::gemm::act_x_wt_bias_bf16`), so there is no point
            // in the trace where the base projection exists as a value: the
            // first thing that exists is already base + bias.
            //
            // The rule found this -- `seam::check_plan` refused the
            // statement with "value 2 comes from Launch", because a
            // bias-folded projection is a `Launch` and not a `Matmul`.
            // Stating it anyway would have put every adapter's delta on the
            // wrong operand.

            // gpt-oss scales, and the driver had to be TAUGHT to: this
            // family shares llama_like's cfg, where `apply_rope_config`
            // had already resolved the scaling, and `mixtral.cpp` spelled
            // a plain `kernels::rope::rope_bf16` anyway. The declaration states
            // the kernel the fixed pass fires.
            let (q, k) = dsl::cuda::rope_yarn_original(&q, &k);
            dsl::cuda::write_kv_to_pages(&k, &v, &kv);

            // The dispatch is the ONLY thing the two classes disagree
            // about. The MoE leg below is admitted by ROUTES
            // (`N * top_k <= max_routes`), not by class, so a prefill
            // under the cap takes the same fused GEMVs — which is why
            // this family has a prefill class at all.
            //
            // EVERY gpt-oss layer carries sinks, so every layer asks for
            // the LSE and pays the per-layer write.
            //
            // The sliding window is not stated here. It is a per-layer
            // ARGUMENT the driver reads out of the deployment
            // (`LayerAttention::window`, reached as `window_left_by_layer`)
            // rather than a kernel this text selects, which is why the
            // alternation `is_sliding` describes leaves no mark on the plan.
            dsl::seam(q.trace(), &dsl::seam::ATTN_Q, &[&q], Some(l));
            let (o, lse) = dsl::cuda::attention_for_lse(class, &q, &kv, facts.q_heads, facts.head_dim);
            let a = dsl::cuda::attention_sink_rescale(&o, &lse, &w.sinks);

            // Post-attention observation. On the sink layers this sees the
            // RESCALED output, not the raw dispatch result -- the rescale
            // is part of what attention computes here.
            dsl::seam(a.trace(), &dsl::seam::ATTN_OUT, &[&a], Some(l));

            // o_proj folds the RESIDUAL (beta=1) and not its bias: the
            // hand-written tp=1 arm calls the plain gemm and then
            // `kernels::norm::add_bias_bf16`. The one place in this layer where
            // the split spelling is the truthful one.
            y += matmul(&a, &w.o_proj);
            y = dsl::add_bias(&y, &w.o_bias);

            // ── The MoE block ───────────────────────────────────────
            let mlp_in = dsl::cuda::rmsnorm(&y, &w.mlp_norm);
            let logits = proj(&mlp_in, &w.router, &w.router_bias);
            let (experts, weights) = dsl::cuda::topk(&logits, facts.top_k);

            let act = dsl::cuda::bf16_to_fp16(&mlp_in);
            let (gate, up) = dsl::cuda::mxfp4_moe_gate_up_decode(
                &act,
                &experts,
                &w.expert_gate_up,
                facts.top_k,
                facts.intermediate,
            );
            // The clamp is the whole fork, and a checkpoint without one
            // takes `kernels::mlp::swiglu_bf16`'s PAIR form — a spelling no
            // statement carries yet. Refused by name rather than guessed:
            // every gpt-oss release so far clamps, so an unclamped one
            // would be the first thing this text had never seen.
            assert!(
                facts.swiglu_limit > 0.0,
                "gpt_oss without a swiglu limit states no activation yet"
            );
            let routed = dsl::cuda::gpt_oss_glu(
                &gate,
                &up,
                facts.top_k,
                facts.intermediate,
                facts.swiglu_limit,
            );
            let routed = dsl::cuda::bf16_to_fp16(&routed);
            let out = dsl::cuda::mxfp4_moe_down_decode(
                &routed,
                &experts,
                &w.expert_down,
                facts.top_k,
                hidden,
            );
            // The combine writes to scratch and the landing is its own
            // launch — mixtral's tp=1 shape. (The `_add` fused form
            // exists, and this pass does not take it.)
            let combined = dsl::cuda::weighted_sum(&weights, &out, hidden, None);
            // STREAM FIRST. `residual_add` lands on operand 0 -- the
            // `kernel!` row aliases output 0 over input 0 -- so the
            // order is not a caller's preference, it says which buffer
            // holds the sum. Written the other way round this claimed
            // the stream lands on the MoE output's bytes, which no
            // driver does and which the stream does not mean; it went
            // unnoticed while both operands were pinned workspace
            // fields and the arm added into `ws.y` regardless. Every
            // other family already spells it this way.
            y = dsl::cuda::residual_add(&y, &combined, hidden);
        }

        dsl::logits_epilogue(
            t,
            &y,
            NormVariant::Plain,
            facts.tied_embeddings,
            facts.vocab,
            false,
        );
    })
}
