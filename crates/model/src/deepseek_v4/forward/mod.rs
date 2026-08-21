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
use model_dsl::axes::{Bf16Ax, DtypeAxis, KvAxis, NativeKv};
use model_dsl::{self as dsl, MatW, NormW, WeightRepr, matmul};
use model_ir::trace::{FireClass, ForwardPlan, NormVariant};

/// The hyper-connection's OWN epsilon — the sigmoid/sinkhorn floor inside
/// `hc_pre`/`hc_head`, NOT the RMS norm's (`dsv4_hc.cuh` norms the MIX and
/// the two are separate numbers in the checkpoint). The hand-written pass
/// read it as `cfg.dsv4_hc_eps` (config key `hc_eps`, default `1e-6`), and
/// no committed V4 config states one, so this is that default — the same
/// honest reading `rope_theta` and `norm_eps` give on the row.
const HC_EPS: f32 = 1e-6;
/// The post-mix blend gain. Hard-coded `2.0f` in the hand-written pass
/// (`deepseek_v4_forward.cpp`), which is the only statement of it anywhere.
const HC_POST_ALPHA: f32 = 2.0;
/// Sinkhorn normalization sweeps over the comb mix. Hard-coded `20` in the
/// hand-written pass ("from config hc_sinkhorn_iters", never read).
const HC_SINKHORN_ITERS: u32 = 20;

struct Dsv4LayerW {
    // DECLARED AND NOT APPLIED, which is a finding and not an oversight of
    // this struct. Both names used to be passed to `dsl::cuda::hc_rmsnorm_to_f32`,
    // whose kernel (`kernels/norm/dsv4_hc.cuh:354`) takes no weight pointer
    // at all -- the binder resolved the name, the arm never bound it, and
    // §6.2's arity rule found the gap by counting one pointer read against
    // two placed operands. The statement dropped them; the CHECKPOINT still
    // carries the tensors, so they stay declared here and in the weight
    // contract. Whether the hyper-connection residual should carry a learned
    // gain is a question about `hc_rmsnorm_to_f32`, and nothing in this
    // refactor changes what any fire computes.
    #[expect(dead_code, reason = "declared for the checkpoint; see above")]
    attn_norm: NormW,
    #[expect(dead_code, reason = "declared for the checkpoint; see above")]
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
    /// THE HYPER-CONNECTION'S OWN AFFINE, TWICE: a layer reads a mix of the
    /// streams before its attention and again before its MLP, and each read
    /// is a separate `hc_pre` with its own learned `scale` and `base`. Four
    /// names, not two, because the two reads are two mixes -- sharing one
    /// pair would make the MLP's transport plan the attention's.
    ///
    /// Trace names with no witnessed checkpoint spelling, exactly like
    /// `attn_sink` and `router_bias`; `project.rs` says why none of them is
    /// in the manifest.
    hc_attn_scale: String,
    hc_attn_base: String,
    hc_mlp_scale: String,
    hc_mlp_base: String,
}

impl Dsv4LayerW {
    fn new(l: u32, f: &Dsv4Facts, norm_eps: f32, repr: WeightRepr) -> Self {
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
            hc_attn_scale: w("hc_attn_scale"),
            hc_attn_base: w("hc_attn_base"),
            hc_mlp_scale: w("hc_mlp_scale"),
            hc_mlp_base: w("hc_mlp_base"),
        }
    }
}

/// deepseek_v4's CUDA text for one fire class.
///
/// **Both shaped classes.** The compressed pass needs the block boundaries
/// this fire's positions imply, and that is a per-TOKEN fact: whether a
/// position closes a compression window is a fact about the position. What
/// used to be decode-only was the request index beside it —
/// `dsv4_boundary_meta_decode` may shortcut it to the token index because a
/// decode brings one row per request, and a prefill has to read it out of
/// `qo_indptr`. Two launchers, one statement here.
pub fn dsv4_cuda<W1: DtypeAxis, W2: DtypeAxis, A: DtypeAxis, K: KvAxis>(
    facts: &Dsv4Facts,
    class: FireClass,
    norm_eps: f32,
    rope_theta: f32,
) -> ForwardPlan {
    // The activation axis is DECLARED but pinned until the launch wrappers
    // take a dtype: every statement below states BF16 outs, so a point
    // instantiated at another A would lie. The pin is a compile refusal,
    // not a comment. Same for K: the compressed cache is stated bf16 and
    // nothing here forks on the scheme yet.
    const {
        assert!(matches!(A::DTYPE, model_ir::trace::DType::BF16));
        assert!(K::NATIVE_BF16);
    }
    // The SKU joins the family's FIRST segment ('.'-separated segment two
    // stays the backend, which `Backend::of_family` parses).
    let family = format!(
        "deepseek_v4-{}-{}-{}.cuda.{}",
        W1::NAME,
        W2::NAME,
        K::NAME,
        class.suffix()
    );
    let a = facts.attn.clone();
    let k = facts.hc.mult;
    dsl::trace_named(&family, |t| {
        let embedded = dsl::embedded_prologue(t, facts.hidden, facts.vocab);
        // The rank-K residual opens here and stays open to `hc_head`.
        let mut streams = dsl::cuda::hc_expand(&embedded, k, facts.hidden);

        // The compressed pass needs the block boundaries this fire's
        // positions imply, and they are a FIRE fact — one statement,
        // outside the layer loop, exactly as the hand-written pass has it.
        // ONE ratio for a fire-wide statement, and the schedule states one
        // PER LAYER (`ratios: &[1, 2, 4]`). The coarsest positive stride is
        // the only single number the schedule can stand behind — its
        // boundaries close every finer window where the ratios nest, as
        // this row's do. A schedule whose ratios did not nest could not be
        // served by one fire-wide meta at all; see the sweep report.
        let fire_ratio = facts
            .ratios
            .iter()
            .copied()
            .max()
            .unwrap_or(0)
            .max(0)
            .unsigned_abs();
        let (boundary_pos, boundary_req, _counts) =
            dsl::cuda::dsv4_boundary_meta(&embedded, class, fire_ratio);

        for l in 0..facts.layers {
            let w = Dsv4LayerW::new(l, facts, norm_eps, W1::REPR);

            // Read a mix of the streams. `hc_pre` produces the layer's
            // input and the two mixes `hc_post` will need to write back.
            let normed_f32 = dsl::cuda::hc_rmsnorm_to_f32(&streams, facts.hidden, norm_eps);
            let (x, post_mix, comb_mix) = dsl::cuda::hc_pre(
                &normed_f32,
                &streams,
                &w.hc_attn_scale,
                &w.hc_attn_base,
                k,
                facts.hidden,
                HC_EPS,
                HC_POST_ALPHA,
                HC_SINKHORN_ITERS,
            );

            // Q through its latent, then a per-head norm with NO gamma —
            // the reference's `q *= rsqrt(...)`, which is a different
            // statement from an rmsnorm with a weight.
            let q_a = matmul(&x, &w.wq_a);
            let q_a = dsl::cuda::rmsnorm(&q_a, &w.q_norm);
            let q = matmul(&q_a, &w.wq_b);
            let q = dsl::cuda::per_head_rmsnorm(&q, a.heads, a.head_dim, norm_eps);
            let kv = matmul(&x, &w.wkv);
            let kv = dsl::cuda::rmsnorm(&kv, &w.kv_norm);
            // Partial rope on the LAST channels of each head.
            let q = dsl::cuda::rope_partial_last(
                &q,
                a.heads,
                a.head_dim,
                a.qk_rope_head_dim,
                rope_theta,
                // GPT-J pairing — the rope kernel's own doc names
                // DeepSeek-V4 (`is_neox_style=False` in vLLM's
                // `build_deepseek_v4_rope`).
                true,
                // No rescaling: the row states `rope_scaling: None`
                // (project.rs), and zero is the kernel's word for it.
                0.0,
                0.0,
                0.0,
                0,
            );
            let kv = dsl::cuda::rope_partial_last(
                &kv,
                a.heads,
                a.head_dim,
                a.qk_rope_head_dim,
                rope_theta,
                true,
                0.0,
                0.0,
                0.0,
                0,
            );
            dsl::seam(q.trace(), &dsl::seam::ATTN_Q, &[&q], Some(l));

            let kvh = dsl::Kv::at(t, l);
            dsl::cuda::write_kv_to_pages(&kv, &kv, &kvh);

            // The window pass, uncompressed.
            // The UNCOMPRESSED reach: everything older is the compressed
            // pass's — project.rs states the same rule for the deployment.
            let window_left = i32::try_from(a.sliding_window).unwrap_or(i32::MAX);
            let window_left = if window_left > 0 { window_left } else { -1 };
            let (o_win, lse_win) = dsl::cuda::attention_flashinfer_prefill_lse(
                &dsl::runtime::query_windows(&q),
                &kvh,
                a.heads,
                a.head_dim,
                // K and V are ONE projection: every query head has its own
                // KV head (project.rs's Geometry says the same).
                a.heads,
                window_left,
                0.0,
                0.0,
            );
            let lse_win = dsl::cuda::lse_log2_to_ln(&lse_win, a.heads);

            // The compressed pass: gather this layer's block entries,
            // rope them the same way, store them, then attend.
            let layer_ratio = facts.compress_ratio_at(l);
            let entries = dsl::cuda::dsv4_compress_gather_paged(
                &boundary_pos,
                &boundary_req,
                l,
                a.head_dim,
                layer_ratio.max(0).unsigned_abs(),
            );
            // THE ENTRIES' OWN RECTANGLE, not the query's. `rope_partial_last`
            // rotates IN PLACE -- its row aliases result 0 with operand 0 --
            // so the result it declares has to be the buffer it was handed,
            // and the gather declares one entry of `head_dim` per boundary
            // token (`[Dim::Tokens, Dim::Const(head_dim)]`). Passing the
            // query's `heads * qk_rope_head_dim` here declared one buffer at
            // two sizes: the arena placed the alias outside its owner.
            let entries = dsl::cuda::rope_partial_last(
                &entries,
                1,
                a.head_dim,
                a.qk_rope_head_dim,
                // The compressed pass's own base. The one committed V4
                // config states `compress_rope_theta: 10000.0`, which the
                // row does not carry as a separate number; at the only
                // measurement in this tree the two bases agree.
                rope_theta,
                true,
                0.0,
                0.0,
                0.0,
                0,
            );
            dsl::cuda::dsv4_store_comp_entries(&entries, &boundary_pos, &boundary_req, l);
            let (o_comp, lse_comp) = dsl::cuda::attention_compressed_paged(
                &q,
                l,
                a.heads,
                a.head_dim,
                layer_ratio.max(0).unsigned_abs(),
                // Over the head's own width — project.rs's sm_scale.
                1.0 / (a.head_dim as f32).sqrt(),
            );

            // One output, weighted by the two LSEs.
            let (o, lse) = dsl::cuda::combine_attn_outputs(
                &o_win, &lse_win, &o_comp, &lse_comp, a.heads, a.head_dim,
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
            let normed_f32 = dsl::cuda::hc_rmsnorm_to_f32(&streams, facts.hidden, norm_eps);
            let (m, post_mix, comb_mix) = dsl::cuda::hc_pre(
                &normed_f32,
                &streams,
                &w.hc_mlp_scale,
                &w.hc_mlp_base,
                k,
                facts.hidden,
                HC_EPS,
                HC_POST_ALPHA,
                HC_SINKHORN_ITERS,
            );

            let out = if !facts.is_moe_layer(l) {
                dsl::dense_gated_mlp(
                    &m,
                    &w.dense_gate,
                    &w.dense_up,
                    &w.dense_down,
                    facts.dense_intermediate,
                    dsl::GatedAct::SwiGluClamp {
                        // `cfg.swiglu_limit`, stored in thousandths so the
                        // facts stay `Eq`.
                        limit: facts.moe.swiglu_limit_milli as f32 / 1000.0,
                    },
                )
            } else {
                let logits = matmul(&m, &w.router);
                let (experts, weights) = dsl::cuda::topk_sqrtsoftplus(
                    &logits,
                    &format!("layer.{l}.router_bias"),
                    facts.moe.top_k,
                    facts.moe.norm_topk_prob,
                    facts.moe.routed_scaling,
                );
                let gate_up = dsl::cuda::moe_gate_up_gemv(
                    &m,
                    &MatW {
                        name: format!("layer.{l}.expert.{{e}}.gate_up"),
                        width: 2 * facts.moe.moe_intermediate,
                        layer: Some(l),
                        repr: W2::REPR,
                    },
                    &experts,
                    facts.moe.top_k,
                );
                let act = dsl::cuda::swiglu_clamp(
                    &gate_up,
                    facts.moe.moe_intermediate,
                    facts.moe.swiglu_limit_milli as f32 / 1000.0,
                );
                let route_out = dsl::cuda::moe_down_gemv(
                    &act,
                    &MatW {
                        name: format!("layer.{l}.expert.{{e}}.down"),
                        width: facts.hidden,
                        layer: Some(l),
                        repr: W2::REPR,
                    },
                    &experts,
                    facts.moe.top_k,
                );
                dsl::cuda::weighted_sum(&weights, &route_out, facts.hidden, None)
            };
            streams = dsl::cuda::hc_post(&out, &streams, &post_mix, &comb_mix, k, facts.hidden);
        }

        // The streams fold into one.
        // NOT LAYER-SCOPED, and that is the collapse being one statement
        // for the whole tower rather than one per layer: it runs once, after
        // the loop, so its affine pair belongs to the model and carries no
        // `layer.{l}.` prefix.
        let y = dsl::cuda::hc_head(
            &streams,
            &streams,
            "hc_head_scale",
            "hc_head_base",
            facts.hidden,
            HC_EPS,
        );
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

/// One shipping SKU: its name and the monomorphized trace it instantiates.
pub type TraceFn = fn(&Dsv4Facts, FireClass, f32, f32) -> ForwardPlan;

/// The family's catalogue — every SKU this build ships, enumerated. The
/// coverage test (`model/tests/catalogue_coverage.rs`) traces each row at
/// both fire classes; `TraceBuilder::finish`'s `check_plan` then refuses a
/// row whose statements reach a routine point that does not exist, which
/// is how the demand set closes: checked, never hoped.
pub const CATALOG: &[(&str, TraceFn)] = model_dsl::catalogue![
    (
        "deepseek_v4-bf16-bf16-kv-bf16",
        dsv4_cuda::<Bf16Ax, Bf16Ax, Bf16Ax, NativeKv>,
    ),
];
