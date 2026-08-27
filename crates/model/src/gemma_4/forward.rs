use model_dsl::{
    Classify, ForwardHybrid, GeomKind, HybridSpec, Input, Predicate, Request, Value, ValueId,
    Weight, merge, ops, seam,
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

        let positions = inputs.positions();
        let plan_d = ops::attn::plan_decode(positions.rec(), inputs.kv_space());
        let plan_p = ops::attn::plan_prefill(positions.rec(), inputs.kv_space());
        let mask = ops::mask(positions.rec(), inputs.kv_space());
        let ids = inputs.tokens();
        let mut y = ops::layout::embed(&ids, &m.embed, m.vocab) * (m.hidden as f32).sqrt();

        let relay = m.ple.as_ref().map(|ple| {
            let proj = ops::linear::matmul(&y, &ple.model_proj) * (m.hidden as f32).sqrt().recip();
            (
                ple,
                ops::elemwise::rmsnorm_per_head(
                    &proj,
                    &ple.model_norm,
                    ple.dim,
                    ple.model_norm_eps,
                ),
            )
        });

        for (l, w) in inputs.layers(&m.layers) {
            let normed = ops::elemwise::rmsnorm(&y, &w.attn_norm, w.attn_norm_eps);
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
                    if inputs.cuda() && at.kind.sliding() {
                        let fused = Facts::qo_one() & !Facts::masked();
                        let (fast_x, rest_x) = normed.split(&fused);
                        let (fast_pos, rest_pos) = positions.split(&fused);
                        let qf = ops::custom::qkv_fused_qknorm_rope_vnorm_write(
                            &ops::linear::matmul(&fast_x, qkv),
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
                ops::attn::masked(&mq, &plan_p, &mask, pages, win, d, at.sm_scale),
                ops::attn::decode(&dq, &plan_d, pages, win, d, at.sm_scale),
                ops::attn::prefill(&p, &plan_p, pages, win, d, at.kind.kv_heads(), at.sm_scale),
            ];
            seam::at(seam::ATTN_OUT, (&a,));
            let o = ops::linear::attention_landing(&a, &w.o_proj, l);
            let o = if m.tp > 1 {
                ops::collective::all_reduce(&o)
            } else {
                o
            };

            y = ops::elemwise::residual_add(
                &ops::elemwise::rmsnorm(&o, &w.post_attn_norm, w.post_attn_norm_eps),
                &y,
            );
            let mlp_in = ops::elemwise::rmsnorm(&y, &w.pre_ffw_norm, w.pre_ffw_norm_eps);

            let act = ops::linear::mlp_geglu_tanh_packed(
                &ops::linear::matmul(&mlp_in, &w.gate_up),
                w.inter,
            );
            let f = ops::linear::matmul(&act, &w.down);
            let f = if m.tp > 1 {
                ops::collective::all_reduce(&f)
            } else {
                f
            };
            y = ops::elemwise::residual_add(
                &ops::elemwise::rmsnorm(&f, &w.post_ffw_norm, w.post_ffw_norm_eps),
                &y,
            );

            if let Some((ple, proj)) = &relay {
                let lp = &ple.per_layer[l as usize];
                let table = ops::layout::embed(&ids, &lp.table, m.vocab) * (ple.dim as f32).sqrt();
                let relay =
                    ops::elemwise::residual_add(&table, &ops::layout::select(proj, l, ple.dim))
                        * std::f32::consts::FRAC_1_SQRT_2;
                let gated = ops::linear::mlp_geglu_tanh(&ops::linear::matmul(&y, &lp.gate), &relay);
                let out = ops::linear::matmul(&gated, &lp.proj);
                let out = ops::elemwise::rmsnorm(&out, &lp.norm, lp.norm_eps);

                y = ops::elemwise::scale(&lp.scalar, &ops::elemwise::residual_add(&out, &y));
            }
        }

        let x = ops::elemwise::rmsnorm(&y, &m.final_norm, m.final_norm_eps);
        let logits = ops::linear::lm_head(&x, &m.embed);
        if let Some(cap) = m.softcap {
            ops::attn::logit_softcap(&logits, cap)
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
    let (q, k, v) = ops::layout::split_qkv(
        &ops::linear::matmul(x, qkv),
        at.q_heads * d,
        at.kind.kv_heads() * d,
    );
    seam::at(seam::ATTN_QV, (&q, &v));
    let v = ops::elemwise::rmsnorm_no_scale(&v, d, at.q_norm_eps);
    let q = ops::elemwise::rmsnorm_per_head(&q, &at.q_norm, d, at.q_norm_eps);
    let k = ops::elemwise::rmsnorm_per_head(&k, k_norm, d, k_norm_eps);
    let (q, k) = match &at.kind {
        AttnKind::Full {
            rotary_dim, theta, ..
        } => ops::elemwise::rope_partial(&q, &k, pos, *rotary_dim, d, *theta),

        AttnKind::Sliding { theta, .. } => ops::elemwise::rope_full(&q, &k, pos, d, *theta, false),
    };
    ops::attn::kv_append(&k, &v, pages, &write_page, &write_offset);
    q
}

fn q_only(x: &Value, pos: &Value, at: &Attn, q_proj: &Weight) -> Value {
    let d = at.kind.head_dim();
    let q = ops::elemwise::rmsnorm_per_head(
        &ops::linear::matmul(x, q_proj),
        &at.q_norm,
        d,
        at.q_norm_eps,
    );
    match &at.kind {
        AttnKind::Full {
            rotary_dim, theta, ..
        } => ops::elemwise::rope_partial_q(&q, pos, *rotary_dim, d, *theta),
        AttnKind::Sliding { theta, .. } => ops::elemwise::rope_partial_q(&q, pos, d, d, *theta),
    }
}
