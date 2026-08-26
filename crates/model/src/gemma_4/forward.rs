//! The Gemma 4 forward pass over the typed IR — the old forward minus the
//! machinery the design killed (design §10): attention plans are built once
//! up front and shared visibly across layers (§6), kv-append geometry is a
//! declared input fetched where it is used — the recorder declares each input
//! once no matter how many layers ask (§7), raggedness is ambient so the
//! masked/prefill arms lose their `query_windows` plumbing (§5), and tensor
//! parallelism is plain control flow on `m.tp` (§9, decision #18). The
//! logit-softcap tail and the local/global theta split transcribe verbatim.

use model_dsl::{
    Classify, ForwardHybrid, GeomKind, HybridSpec, Input, Predicate, Request, Value, ValueId,
    Weight, kernels, merge, seam,
};

use super::model::{Attn, AttnBanks, AttnKind, Model};

model_dsl::facts! {
    pub struct Facts { qo_one, masked }
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
        // One paged space for every owned kv row — the fire lays every
        // layer's kv pages out identically, and the kv-sharing tail reads
        // through the same pages it would have named.
        let kv = c.kv_space(self.kv);
        for (l, w) in self.layers.iter().enumerate() {
            if let AttnBanks::Owned { .. } = &w.attn.banks {
                let ki = &w.attn.kind;
                c.kv(
                    kv,
                    format!("kv.{l}"),
                    [2, ki.kv_heads() as u64 * ki.head_dim() as u64],
                );
            }
        }
        c
    }

    fn forward(&self, inputs: Input<Facts>) -> Value {
        let m = self;
        // The decode and prefill plans, built once and shared visibly by
        // every layer (§6) — the masked arm rides the prefill plan, its
        // queries being just as ragged, and names the mask it applies.
        let positions = inputs.positions();
        let plan_d = kernels::attn::plan_decode(positions.rec(), inputs.kv_space());
        let plan_p = kernels::attn::plan_prefill(positions.rec(), inputs.kv_space());
        let mask = kernels::mask(positions.rec(), inputs.kv_space());
        let ids = inputs.tokens();
        let mut y = kernels::layout::embed(&ids, &m.embed, m.vocab) * (m.hidden as f32).sqrt();

        let relay = m.ple.as_ref().map(|ple| {
            let proj =
                kernels::linear::matmul(&y, &ple.model_proj) * (m.hidden as f32).sqrt().recip();
            (
                ple,
                kernels::elemwise::rmsnorm_per_head(
                    &proj,
                    &ple.model_norm,
                    ple.dim,
                    ple.model_norm_eps,
                ),
            )
        });

        for (l, w) in inputs.layers(&m.layers) {
            let normed = kernels::elemwise::rmsnorm(&y, &w.attn_norm, w.attn_norm_eps);
            let at = &w.attn;
            let d = at.kind.head_dim();
            let pages = inputs.kv(&at.kv);

            let q = match &at.banks {
                AttnBanks::Shared { q_proj } => q_only(&normed, &positions, at, q_proj),
                AttnBanks::Owned {
                    qkv,
                    k_norm,
                    k_norm_eps,
                } => {
                    // The old guard also asked `K::NATIVE_BF16`; a cache row's
                    // element layout is the model's own `kv` declaration now
                    // (design §5), and the shipped SKUs store native bf16 kv,
                    // so the plane is the whole question.
                    if inputs.cuda() && at.kind.sliding() {
                        let fused = Facts::qo_one() & !Facts::masked();
                        let (fast_x, rest_x) = normed.split(&fused);
                        let (fast_pos, rest_pos) = positions.split(&fused);
                        let qf = kernels::custom::qkv_fused_qknorm_rope_vnorm_write(
                            &kernels::linear::matmul(&fast_x, qkv),
                            &at.q_norm,
                            at.q_norm_eps,
                            k_norm,
                            *k_norm_eps,
                            at.kind.kv_heads(),
                            d,
                            pages,
                            &inputs.geometry(inputs.kv_space(), GeomKind::WritePage),
                            &inputs.geometry(inputs.kv_space(), GeomKind::WriteOffset),
                            at.kind.theta(),
                            &fast_pos,
                        );
                        let qr = qkv_unfused(
                            &rest_x,
                            &rest_pos,
                            &inputs,
                            at,
                            qkv,
                            k_norm,
                            *k_norm_eps,
                            pages,
                        );
                        merge![qf, qr]
                    } else {
                        qkv_unfused(
                            &normed,
                            &positions,
                            &inputs,
                            at,
                            qkv,
                            k_norm,
                            *k_norm_eps,
                            pages,
                        )
                    }
                }
            };

            seam::at(seam::ATTN_Q, (&q,));

            let win = at.kind.window();
            let [mq, dq, p] = q.split([Facts::masked(), Facts::qo_one(), Predicate::rest()]);
            let a = merge![
                kernels::attn::masked(&mq, &plan_p, &mask, pages, win, d, at.sm_scale),
                kernels::attn::decode(&dq, &plan_d, pages, win, d, at.sm_scale),
                kernels::attn::prefill(&p, &plan_p, pages, win, d, at.kind.kv_heads(), at.sm_scale),
            ];
            seam::at(seam::ATTN_OUT, (&a,));
            let o = kernels::linear::attention_landing(&a, &w.o_proj, l);
            let o = if m.tp > 1 {
                kernels::collective::all_reduce(&o)
            } else {
                o
            };

            y = kernels::elemwise::residual_add(
                &kernels::elemwise::rmsnorm(&o, &w.post_attn_norm, w.post_attn_norm_eps),
                &y,
            );
            let mlp_in = kernels::elemwise::rmsnorm(&y, &w.pre_ffw_norm, w.pre_ffw_norm_eps);

            let act = kernels::linear::mlp_geglu_tanh_packed(
                &kernels::linear::matmul(&mlp_in, &w.gate_up),
                w.inter,
            );
            let f = kernels::linear::matmul(&act, &w.down);
            let f = if m.tp > 1 {
                kernels::collective::all_reduce(&f)
            } else {
                f
            };
            y = kernels::elemwise::residual_add(
                &kernels::elemwise::rmsnorm(&f, &w.post_ffw_norm, w.post_ffw_norm_eps),
                &y,
            );

            if let Some((ple, proj)) = &relay {
                let lp = &ple.per_layer[l as usize];
                let table =
                    kernels::layout::embed(&ids, &lp.table, m.vocab) * (ple.dim as f32).sqrt();
                let relay = kernels::elemwise::residual_add(
                    &table,
                    &kernels::layout::select(proj, l, ple.dim),
                ) * std::f32::consts::FRAC_1_SQRT_2;
                let gated =
                    kernels::linear::mlp_geglu_tanh(&kernels::linear::matmul(&y, &lp.gate), &relay);
                let out = kernels::linear::matmul(&gated, &lp.proj);
                let out = kernels::elemwise::rmsnorm(&out, &lp.norm, lp.norm_eps);

                y = kernels::elemwise::scale(
                    &lp.scalar,
                    &kernels::elemwise::residual_add(&out, &y),
                );
            }
        }

        let x = kernels::elemwise::rmsnorm(&y, &m.final_norm, m.final_norm_eps);
        let logits = kernels::linear::lm_head(&x, &m.embed);
        if let Some(cap) = m.softcap {
            kernels::attn::logit_softcap(&logits, cap)
        } else {
            logits
        }
    }
}

fn qkv_unfused(
    x: &Value,
    pos: &Value,
    inputs: &Input<Facts>,
    at: &Attn,
    qkv: &Weight,
    k_norm: &Weight,
    k_norm_eps: f32,
    pages: ValueId,
) -> Value {
    let write_page = inputs.geometry(inputs.kv_space(), GeomKind::WritePage);
    let write_offset = inputs.geometry(inputs.kv_space(), GeomKind::WriteOffset);
    let d = at.kind.head_dim();
    let (q, k, v) = kernels::layout::split_qkv(
        &kernels::linear::matmul(x, qkv),
        at.q_heads * d,
        at.kind.kv_heads() * d,
    );
    seam::at(seam::ATTN_QV, (&q, &v));
    let v = kernels::elemwise::rmsnorm_no_scale(&v, d, at.q_norm_eps);
    let q = kernels::elemwise::rmsnorm_per_head(&q, &at.q_norm, d, at.q_norm_eps);
    let k = kernels::elemwise::rmsnorm_per_head(&k, k_norm, d, k_norm_eps);
    let (q, k) = match &at.kind {
        AttnKind::Full {
            rotary_dim, theta, ..
        } => kernels::elemwise::rope_partial(&q, &k, pos, *rotary_dim, d, *theta),

        AttnKind::Sliding { theta, .. } => {
            kernels::elemwise::rope_full(&q, &k, pos, d, *theta, false)
        }
    };
    kernels::attn::kv_append(&k, &v, pages, &write_page, &write_offset);
    q
}

fn q_only(x: &Value, pos: &Value, at: &Attn, q_proj: &Weight) -> Value {
    let d = at.kind.head_dim();
    let q = kernels::elemwise::rmsnorm_per_head(
        &kernels::linear::matmul(x, q_proj),
        &at.q_norm,
        d,
        at.q_norm_eps,
    );
    match &at.kind {
        AttnKind::Full {
            rotary_dim, theta, ..
        } => kernels::elemwise::rope_partial_q(&q, pos, *rotary_dim, d, *theta),
        AttnKind::Sliding { theta, .. } => kernels::elemwise::rope_partial_q(&q, pos, d, d, *theta),
    }
}
