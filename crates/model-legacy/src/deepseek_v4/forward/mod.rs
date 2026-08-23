pub mod facts;

use self::facts::Dsv4Facts;
use model_dsl::axes::{Bf16Ax, DtypeAxis, KvAxis, NativeKv};
use model_dsl::{self as dsl, MatW, NormW, WeightRepr, matmul};
use model_ir::trace::{DType, Dim, FireClass, ForwardPlan, NormVariant, Shape};

const HC_EPS: f32 = 1e-6;

const HC_POST_ALPHA: f32 = 2.0;

const HC_SINKHORN_ITERS: u32 = 20;

struct Dsv4LayerW {
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

pub fn dsv4_cuda<W1: DtypeAxis, W2: DtypeAxis, A: DtypeAxis, K: KvAxis>(
    facts: &Dsv4Facts,
    class: FireClass,
    norm_eps: f32,
    rope_theta: f32,
) -> ForwardPlan {
    const {
        assert!(matches!(A::DTYPE, model_ir::trace::DType::BF16));
        assert!(K::NATIVE_BF16);
    }

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

        let rt = dsl::rt(t);
        let mut streams = dsl::cuda::generated::hc_expand(
            &embedded,
            (
                Shape(vec![Dim::Tokens, Dim::Const(k), Dim::Const(facts.hidden)]),
                DType::BF16,
            ),
            embedded.layer(),
            None,
        );

        let mut boundary_by_ratio = std::collections::BTreeMap::new();
        for r in facts.ratios.iter().copied().filter(|r| *r > 0) {
            boundary_by_ratio
                .entry(r.unsigned_abs())
                .or_insert_with(|| {
                    dsl::cuda::dsv4_boundary_meta(&embedded, class, r.unsigned_abs())
                });
        }

        for l in 0..facts.layers {
            let w = Dsv4LayerW::new(l, facts, norm_eps, W1::REPR);

            let normed_f32 = dsl::cuda::generated::hc_rmsnorm_to_f32(
                &streams,
                (
                    Shape(vec![Dim::Tokens, Dim::Const(facts.hidden)]),
                    DType::F32,
                ),
                norm_eps,
                streams.layer(),
                None,
            );
            let (x, post_mix, comb_mix) = dsl::cuda::generated::hc_pre_postprocess(
                &normed_f32,
                &w.hc_attn_scale,
                &w.hc_attn_base,
                &streams,
                (Shape(vec![Dim::Tokens, Dim::Const(k)]), DType::F32),
                (
                    Shape(vec![Dim::Tokens, Dim::Const(k), Dim::Const(k)]),
                    DType::F32,
                ),
                (
                    Shape(vec![Dim::Tokens, Dim::Const(facts.hidden)]),
                    DType::BF16,
                ),
                HC_EPS,
                HC_POST_ALPHA,
                HC_SINKHORN_ITERS as i32,
                normed_f32.layer(),
                None,
            );

            let q_a = matmul(&x, &w.wq_a);
            let q_a = dsl::cuda::rmsnorm(&q_a, &w.q_norm);
            let q = matmul(&q_a, &w.wq_b);
            let q = dsl::cuda::generated::per_head_rmsnorm(
                &q,
                a.head_dim as i32,
                norm_eps,
                q.layer(),
                None,
            );
            let kv = matmul(&x, &w.wkv);
            let kv = dsl::cuda::rmsnorm(&kv, &w.kv_norm);

            let q = dsl::cuda::generated::rope_partial_last_q_bf16(
                &q,
                a.head_dim as i32,
                a.qk_rope_head_dim as i32,
                rope_theta,
                true,
                0.0,
                0.0,
                0.0,
                0,
                &rt.positions(),
                q.layer(),
                None,
            );
            let kv = dsl::cuda::generated::rope_partial_last_q_bf16(
                &kv,
                a.head_dim as i32,
                a.qk_rope_head_dim as i32,
                rope_theta,
                true,
                0.0,
                0.0,
                0.0,
                0,
                &rt.positions(),
                kv.layer(),
                None,
            );
            dsl::seam(q.trace(), &dsl::seam::ATTN_Q, &[&q], Some(l));

            let kvh = dsl::Kv::at(t, l);
            dsl::cuda::write_kv_to_pages(&kv, &kv, &kvh, a.heads, a.head_dim);

            let window_left = i32::try_from(a.sliding_window).unwrap_or(i32::MAX);
            let window_left = if window_left > 0 { window_left } else { -1 };
            let (o_win, lse_win) = dsl::cuda::attention_flashinfer_prefill_lse(
                &dsl::runtime::query_windows(&q),
                &kvh,
                a.heads,
                a.head_dim,
                a.heads,
                window_left,
                0.0,
                0.0,
            );
            let lse_win = dsl::cuda::generated::lse_log2_to_ln(&lse_win, lse_win.layer(), None);

            let layer_ratio = facts.compress_ratio_at(l);
            let comp = (layer_ratio > 0).then(|| {
                let ratio = layer_ratio.unsigned_abs();
                let (boundary_pos, boundary_req, _counts) = boundary_by_ratio
                    .get(&ratio)
                    .expect("a compressing layer's ratio is in the schedule");
                let entries = dsl::cuda::dsv4_compress_gather_paged(
                    boundary_pos,
                    boundary_req,
                    l,
                    a.head_dim,
                    ratio,
                );

                let entries = dsl::cuda::generated::rope_partial_last_q_bf16(
                    &entries,
                    a.head_dim as i32,
                    a.qk_rope_head_dim as i32,
                    rope_theta,
                    true,
                    0.0,
                    0.0,
                    0.0,
                    0,
                    &rt.positions(),
                    entries.layer(),
                    None,
                );
                {
                    let kvh = rt.kv(l);
                    dsl::cuda::generated::dsv4_store_comp_entries_bf16(
                        &entries,
                        boundary_pos,
                        boundary_req,
                        &kvh.cache(),
                        &rt.object("dsv4.comp_kv_pages", Some(l)),
                        Some(l),
                        Some(kvh.state()),
                    );
                }
                let (o_comp, lse_comp) = dsl::cuda::generated::attention_compressed_paged_bf16(
                    &q,
                    (
                        Shape(vec![
                            Dim::Tokens,
                            Dim::Const(a.heads),
                            Dim::Const(a.head_dim),
                        ]),
                        DType::BF16,
                    ),
                    (Shape(vec![Dim::Tokens, Dim::Const(a.heads)]), DType::F32),
                    ratio as i32,
                    a.heads as i32,
                    a.head_dim as i32,
                    &rt.kv(l).cache(),
                    1.0 / (a.head_dim as f32).sqrt(),
                    &rt.positions(),
                    &rt.request_of_token(),
                    &rt.object("dsv4.comp_kv_pages", Some(l)),
                    Some(l),
                    None,
                );
                (o_comp, lse_comp)
            });

            let (o, lse) = match &comp {
                Some((o_comp, lse_comp)) => dsl::cuda::generated::combine_attn_outputs(
                    &o_win,
                    &lse_win,
                    o_comp,
                    lse_comp,
                    a.heads as i32,
                    a.head_dim as i32,
                    o_win.layer(),
                    None,
                ),
                None => (o_win, lse_win),
            };
            let o = dsl::cuda::generated::attn_sink_correction(
                &o,
                &lse,
                &format!("layer.{l}.attn_sink"),
                a.head_dim as i32,
                o.layer(),
                None,
            );
            dsl::seam(o.trace(), &dsl::seam::ATTN_OUT, &[&o], Some(l));

            let o = matmul(&o, &w.o_a);
            let o = matmul(&o, &w.o_b);
            streams =
                dsl::cuda::generated::hc_post(&o, &streams, &post_mix, &comb_mix, o.layer(), None);

            let normed_f32 = dsl::cuda::generated::hc_rmsnorm_to_f32(
                &streams,
                (
                    Shape(vec![Dim::Tokens, Dim::Const(facts.hidden)]),
                    DType::F32,
                ),
                norm_eps,
                streams.layer(),
                None,
            );
            let (m, post_mix, comb_mix) = dsl::cuda::generated::hc_pre_postprocess(
                &normed_f32,
                &w.hc_mlp_scale,
                &w.hc_mlp_base,
                &streams,
                (Shape(vec![Dim::Tokens, Dim::Const(k)]), DType::F32),
                (
                    Shape(vec![Dim::Tokens, Dim::Const(k), Dim::Const(k)]),
                    DType::F32,
                ),
                (
                    Shape(vec![Dim::Tokens, Dim::Const(facts.hidden)]),
                    DType::BF16,
                ),
                HC_EPS,
                HC_POST_ALPHA,
                HC_SINKHORN_ITERS as i32,
                normed_f32.layer(),
                None,
            );

            let out = if !facts.is_moe_layer(l) {
                dsl::dense_gated_mlp(
                    &m,
                    &w.dense_gate,
                    &w.dense_up,
                    &w.dense_down,
                    facts.dense_intermediate,
                    dsl::GatedAct::SwiGluClamp {
                        limit: facts.moe.swiglu_limit_milli as f32 / 1000.0,
                    },
                )
            } else {
                let logits = matmul(&m, &w.router);
                let (experts, weights) = dsl::cuda::generated::topk_sqrtsoftplus(
                    &logits,
                    (
                        Shape(vec![Dim::Tokens, Dim::Const(facts.moe.top_k)]),
                        DType::I32,
                    ),
                    (
                        Shape(vec![Dim::Tokens, Dim::Const(facts.moe.top_k)]),
                        DType::F32,
                    ),
                    Some(&format!("layer.{l}.router_bias")),
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
                let act = dsl::cuda::generated::chunked_swiglu_clamp(
                    &gate_up,
                    facts.moe.swiglu_limit_milli as f32 / 1000.0,
                    gate_up.layer(),
                    None,
                );
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
                dsl::cuda::weighted_sum(&weights, &route_out, facts.hidden, None)
            };
            streams = dsl::cuda::generated::hc_post(
                &out,
                &streams,
                &post_mix,
                &comb_mix,
                out.layer(),
                None,
            );
        }

        let y = dsl::cuda::generated::hc_head_postprocess(
            &streams,
            "hc_head_scale",
            "hc_head_base",
            &streams,
            (
                Shape(vec![Dim::Tokens, Dim::Const(facts.hidden)]),
                DType::BF16,
            ),
            HC_EPS,
            streams.layer(),
            None,
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

pub type TraceFn = fn(&Dsv4Facts, FireClass, f32, f32) -> ForwardPlan;

pub type ShippedW1 = Bf16Ax;

pub type ShippedW2 = Bf16Ax;

pub type ShippedA = Bf16Ax;

pub type ShippedKv = NativeKv;

pub const CATALOG: &[(&str, TraceFn)] = model_dsl::catalogue![(
    "deepseek_v4-bf16-bf16-kv-bf16",
    dsv4_cuda::<ShippedW1, ShippedW2, ShippedA, ShippedKv>,
),];
