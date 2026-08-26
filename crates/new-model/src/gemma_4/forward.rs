//! The Gemma 4 forward pass over the typed IR — the old forward minus the
//! machinery the design killed (design §10): attention plans are built once
//! up front and shared visibly across layers (§6), kv-append geometry is a
//! declared input fetched once per forward (§7), raggedness is ambient so the
//! masked/prefill arms lose their `query_windows` plumbing (§5), and tensor
//! parallelism is plain control flow on `m.tp` (§9, decision #18). The
//! logit-softcap tail and the local/global theta split transcribe verbatim.

use new_model_dsl::{
    Classify, Facts, ForwardHybrid, HybridSpec, Input, Norm, Predicate, Request, Value, ValueId,
    Weight, kernels, merge, seam,
};
use new_model_ir::GeomKind;

use super::model::{Attn, AttnBanks, AttnKind, Model};

#[derive(Facts)]
pub struct Facts {
    pub qo_one: bool,
    pub masked: bool,
}

impl Classify for Facts {
    fn of(r: &Request) -> Self {
        Self {
            qo_one: r.query_len() == 1,
            masked: r.has_custom_mask(),
        }
    }
}

impl ForwardHybrid for Model {
    type Facts = Facts;

    fn caches(&self) -> HybridSpec {
        let mut c = HybridSpec::new();
        for (l, w) in self.layers.iter().enumerate() {
            if let AttnBanks::Owned { .. } = &w.attn.banks {
                let ki = &w.attn.kind;
                c.kv(
                    format!("kv.{l}"),
                    [2, ki.kv_heads() as u64 * ki.head_dim() as u64],
                );
            }
        }
        c
    }

    fn forward(&self, inputs: Input<Facts>) -> Value {
        let m = self;
        let attn = AttnShared::of(m, &inputs);
        let ids = inputs.tokens();
        let mut y = kernels::layout::embed(&ids, &m.embed, m.vocab) * (m.hidden as f32).sqrt();

        let relay = m.ple.as_ref().map(|ple| {
            let proj =
                kernels::gemm::matmul(&y, &ple.model_proj) * (m.hidden as f32).sqrt().recip();
            let n = &ple.model_norm;
            (
                ple,
                kernels::norm::rmsnorm_per_head(&proj, &n.weight, ple.dim, n.eps),
            )
        });

        for (l, w) in inputs.layers(&m.layers) {
            let an = &w.attn_norm;
            let normed = kernels::norm::rmsnorm(&y, &an.weight, an.eps);
            let at = &w.attn;
            let d = at.kind.head_dim();
            let pages = inputs.kv(at.kv.name());

            let q = match &at.banks {
                AttnBanks::Shared { q_proj } => q_only(&normed, &attn.positions, at, q_proj),
                AttnBanks::Owned { qkv, k_norm } => {
                    // The old guard also asked `K::NATIVE_BF16`; a cache row's
                    // element layout is load-time business now (design §5),
                    // and the shipped SKUs store native bf16 kv, so the plane
                    // is the whole question.
                    if inputs.cuda() && at.kind.sliding() {
                        let fused = Facts::qo_one() & !Facts::masked();
                        let (fast_x, rest_x) = normed.split(&fused);
                        let (fast_pos, rest_pos) = attn.positions.split(&fused);
                        let qf = kernels::cuda::qkv_fused_qknorm_rope_vnorm_write(
                            &kernels::gemm::matmul(&fast_x, qkv),
                            &at.q_norm,
                            k_norm,
                            at.kind.kv_heads(),
                            d,
                            pages,
                            &attn.kv_indices,
                            at.kind.theta(),
                            &fast_pos,
                        );
                        let qr = qkv_unfused(&rest_x, &rest_pos, &attn, at, qkv, k_norm, pages);
                        merge![qf, qr]
                    } else {
                        qkv_unfused(&normed, &attn.positions, &attn, at, qkv, k_norm, pages)
                    }
                }
            };

            seam::at(seam::ATTN_Q, (&q,));

            let win = at.kind.window();
            let [mq, dq, p] = q.split([Facts::masked(), Facts::qo_one(), Predicate::rest()]);
            let a = merge![
                kernels::attention::masked(&mq, &attn.plan_p, pages, win, d, at.sm_scale),
                kernels::attention::decode(&dq, &attn.plan_d, pages, win, d, at.sm_scale),
                kernels::attention::prefill(
                    &p,
                    &attn.plan_p,
                    pages,
                    win,
                    d,
                    at.kind.kv_heads(),
                    at.sm_scale
                ),
            ];
            seam::at(seam::ATTN_OUT, (&a,));
            let o = kernels::gemm::attention_landing(&a, &w.o_proj, l);
            let o = if m.tp > 1 {
                kernels::dist::all_reduce(&o)
            } else {
                o
            };

            let pan = &w.post_attn_norm;
            y = kernels::norm::residual_add(&kernels::norm::rmsnorm(&o, &pan.weight, pan.eps), &y);
            let pff = &w.pre_ffw_norm;
            let mlp_in = kernels::norm::rmsnorm(&y, &pff.weight, pff.eps);

            let act = kernels::mlp::geglu_tanh_packed(
                &kernels::gemm::matmul(&mlp_in, &w.gate_up),
                w.inter,
            );
            let f = kernels::gemm::matmul(&act, &w.down);
            let f = if m.tp > 1 {
                kernels::dist::all_reduce(&f)
            } else {
                f
            };
            let pfn = &w.post_ffw_norm;
            y = kernels::norm::residual_add(&kernels::norm::rmsnorm(&f, &pfn.weight, pfn.eps), &y);

            if let Some((ple, proj)) = &relay {
                let lp = &ple.per_layer[l as usize];
                let table =
                    kernels::layout::embed(&ids, &lp.table, m.vocab) * (ple.dim as f32).sqrt();
                let relay =
                    kernels::norm::residual_add(&table, &kernels::layout::select(proj, l, ple.dim))
                        * std::f32::consts::FRAC_1_SQRT_2;
                let gated = kernels::mlp::geglu_tanh(&kernels::gemm::matmul(&y, &lp.gate), &relay);
                let out = kernels::gemm::matmul(&gated, &lp.proj);
                let out = kernels::norm::rmsnorm(&out, &lp.norm.weight, lp.norm.eps);

                y = kernels::norm::scale(&lp.scalar, &kernels::norm::residual_add(&out, &y));
            }
        }

        let fin = &m.final_norm;
        let x = kernels::norm::rmsnorm(&y, &fin.weight, fin.eps);
        let logits = kernels::gemm::lm_head(&x, &m.embed);
        if let Some(cap) = m.softcap {
            kernels::attention::logit_softcap(&logits, cap)
        } else {
            logits
        }
    }
}

/// What every attention layer shares, stated once per forward: the decode and
/// prefill plans (§6) — the masked arm rides the prefill plan, its queries
/// being just as ragged — the page indices kv writes go through (§7), and the
/// positions rope reads. The first owned kv space's geometry stands in for
/// all of them: the fire lays every layer's kv pages out identically, and the
/// kv-sharing tail reads through the same pages it would have named.
struct AttnShared {
    positions: Value,
    kv_indices: Value,
    plan_d: Value,
    plan_p: Value,
}

impl AttnShared {
    fn of(m: &Model, inputs: &Input<Facts>) -> AttnShared {
        let kv = m
            .layers
            .iter()
            .find_map(|w| match &w.attn.banks {
                AttnBanks::Owned { .. } => Some(inputs.cache_index(w.attn.kv.name())),
                AttnBanks::Shared { .. } => None,
            })
            .expect("gemma 4 owns its leading kv spaces");
        let positions = inputs.positions();
        AttnShared {
            plan_d: kernels::attention::plan_decode(positions.rec(), kv),
            plan_p: kernels::attention::plan_prefill(positions.rec(), kv),
            kv_indices: inputs.geometry(kv, GeomKind::Indices),
            positions,
        }
    }
}

fn qkv_unfused(
    x: &Value,
    pos: &Value,
    s: &AttnShared,
    at: &Attn,
    qkv: &Weight,
    k_norm: &Norm,
    pages: ValueId,
) -> Value {
    let d = at.kind.head_dim();
    let (q, k, v) = kernels::layout::split_qkv(
        &kernels::gemm::matmul(x, qkv),
        at.q_heads * d,
        at.kind.kv_heads() * d,
    );
    seam::at(seam::ATTN_QV, (&q, &v));
    let v = kernels::norm::rmsnorm_no_scale(&v, d, at.q_norm.eps);
    let q = kernels::norm::rmsnorm_per_head(&q, &at.q_norm.weight, d, at.q_norm.eps);
    let k = kernels::norm::rmsnorm_per_head(&k, &k_norm.weight, d, k_norm.eps);
    let (q, k) = match &at.kind {
        AttnKind::Full {
            rotary_dim, theta, ..
        } => kernels::rope::partial(&q, &k, pos, *rotary_dim, d, *theta),

        AttnKind::Sliding { theta, .. } => kernels::rope::full(&q, &k, pos, d, *theta, false),
    };
    kernels::attention::kv_append(&k, &v, pages, &s.kv_indices, pos);
    q
}

fn q_only(x: &Value, pos: &Value, at: &Attn, q_proj: &Weight) -> Value {
    let d = at.kind.head_dim();
    let q = kernels::norm::rmsnorm_per_head(
        &kernels::gemm::matmul(x, q_proj),
        &at.q_norm.weight,
        d,
        at.q_norm.eps,
    );
    match &at.kind {
        AttnKind::Full {
            rotary_dim, theta, ..
        } => kernels::rope::partial_q(&q, pos, *rotary_dim, d, *theta),
        AttnKind::Sliding { theta, .. } => kernels::rope::partial_q(&q, pos, d, d, *theta),
    }
}
