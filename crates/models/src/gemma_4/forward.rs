use model_dsl::{
    Classify, Dtype, ForwardHybrid, HybridSpec, Input, MropeForm, Platform, Predicate, Request,
    Value, ValueId, Weight, ops, seam,
};

use super::model::{Attn, AttnBanks, Clippable, Draft, Model, Reading, Tower};

pub struct Facts {
    pub qo_one: bool,
    pub masked: bool,
    pub has_adapter: bool,
    pub media: bool,
    pub drafts: bool,
}

impl Facts {
    pub fn qo_one() -> Predicate {
        Predicate::fact(0)
    }

    pub fn masked() -> Predicate {
        Predicate::fact(1)
    }

    /// Rows routed to a registered adapter. Bit 2 of the fact word; a fire
    /// with no adapter rows costs nothing (the walk skips the empty class).
    pub fn has_adapter() -> Predicate {
        Predicate::fact(2)
    }

    /// Rows the embed merge writes tower output into. Bit 3 of the fact word.
    pub fn media() -> Predicate {
        Predicate::fact(3)
    }

    /// Rows that run the aux draft head. A second readout with its own
    /// column (`seam::MTP`), not a correction to the trunk's. Bit 4 of the
    /// fact word.
    pub fn drafts() -> Predicate {
        Predicate::fact(4)
    }
}

impl Classify for Facts {
    fn of(r: &Request) -> Facts {
        Facts {
            qo_one: r.query_len() == 1,
            masked: r.has_custom_mask(),
            has_adapter: r.has_adapter(),
            media: r.has_media(),
            drafts: r.drafts(),
        }
    }

    fn word(&self) -> u64 {
        u64::from(self.qo_one)
            | (u64::from(self.masked) << 1)
            | (u64::from(self.has_adapter) << 2)
            | (u64::from(self.media) << 3)
            | (u64::from(self.drafts) << 4)
    }
}

impl ForwardHybrid for Model {
    type Facts = Facts;

    fn caches(&self) -> HybridSpec {
        let mut c = HybridSpec::new();

        let kv = c.kv_space(self.kv);
        for w in &self.layers {
            if let AttnBanks::Owned { .. } = &w.attn.banks {
                let (head_dim, kv_heads) = match w.attn.reading {
                    Reading::Sliding => (self.sliding.head_dim, self.sliding.kv_heads),
                    Reading::Global => (self.global.head_dim, self.global.kv_heads),
                };
                let plane = kv_heads as u64 * head_dim as u64;
                c.kv(kv, w.attn.kv.clone(), [plane, plane]);
            }
        }
        // The draft head's kv lives in the same page-id space as the trunk
        // (same sequence, same lengths); its planes use the global reading.
        if let Some(a) = &self.draft {
            let plane = self.global.kv_heads as u64 * self.global.head_dim as u64;
            c.kv(kv, a.attn.kv.clone(), [plane, plane]);
        }
        c
    }

    fn forward(&self, inputs: Input<Facts>) -> Value {
        let m = self;

        let positions = inputs.positions();
        let qo_one = Facts::qo_one();
        let fused = qo_one.clone() & !Facts::masked();

        // Sliding/global readings need separate plans (head width, kv chunk
        // size, window differ); classes need separate plans too, or sharing
        // one would straddle rebased boundaries (`Fault::Straddled`).
        let classes = [Facts::masked(), qo_one.clone(), Predicate::rest()];
        let [input_m, input_d, input_p] = inputs.split(classes.clone());
        let plan_m = [
            ops::attn::plan_prefill(
                &input_m,
                m.q_heads,
                m.sliding.kv_heads,
                m.sliding.head_dim,
                Some(m.sliding.window),
            ),
            ops::attn::plan_prefill(
                &input_m,
                m.q_heads,
                m.global.kv_heads,
                m.global.head_dim,
                None,
            ),
        ];
        let plan_d = [
            ops::attn::plan_decode(
                &input_d,
                m.q_heads,
                m.sliding.kv_heads,
                m.sliding.head_dim,
                Some(m.sliding.window),
            ),
            ops::attn::plan_decode(
                &input_d,
                m.q_heads,
                m.global.kv_heads,
                m.global.head_dim,
                None,
            ),
        ];
        let plan_p = [
            ops::attn::plan_prefill(
                &input_p,
                m.q_heads,
                m.sliding.kv_heads,
                m.sliding.head_dim,
                Some(m.sliding.window),
            ),
            ops::attn::plan_prefill(
                &input_p,
                m.q_heads,
                m.global.kv_heads,
                m.global.head_dim,
                None,
            ),
        ];
        let mask = inputs.mask();

        // Emitted before any trunk node: interleaving the tower's node runs
        // with the trunk's would trigger `Error::UnitsInterleave`.
        let towered = m.tower.as_ref().map(|t| tower(&inputs, t));

        let ids = inputs.tokens();
        let mut y = ops::layout::embed(&ids, &m.embed, m.vocab) * (m.hidden as f32).sqrt();

        // Embed merge runs only over media-window rows; other rows keep the
        // token embedding already written.
        if let Some(t) = &towered {
            let (imaged, _) = y.split(&Facts::media());
            y = ops::layout::scatter_live_rows(t, &inputs.patch_routes(), &imaged).everywhere();
        }

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

        let routes = inputs.adapter_routes();
        for (l, w) in inputs.walk_layers(&m.layers) {
            let normed = ops::elemwise::rmsnorm(&y, &w.attn_norm, w.attn_norm_eps);
            let at = &w.attn;
            // Head width, kv heads, and window all come from this layer's reading.
            let (d, kv_heads, win) = match at.reading {
                Reading::Sliding => (
                    m.sliding.head_dim,
                    m.sliding.kv_heads,
                    Some(m.sliding.window),
                ),
                Reading::Global => (m.global.head_dim, m.global.kv_heads, None),
            };
            let pages = inputs.kv(&at.kv);

            let q = match &at.banks {
                AttnBanks::Shared { q_proj } => q_only(&normed, &positions, m, at, d, q_proj),
                AttnBanks::Owned {
                    qkv,
                    k_norm,
                    k_norm_eps,
                } => {
                    if model_dsl::platform() == Platform::Cuda && at.reading == Reading::Sliding {
                        let (fast_x, rest_x) = normed.split(&fused);
                        let (fast_pos, rest_pos) = positions.split(&fused);
                        let qf = ops::custom::qkv_fused_qknorm_rope_vnorm_write(
                            &ops::linear::matmul(&fast_x, qkv),
                            &at.q_norm,
                            at.q_norm_eps,
                            k_norm,
                            *k_norm_eps,
                            kv_heads,
                            d,
                            pages,
                            &inputs.write_page(&at.kv),
                            &inputs.write_offset(&at.kv),
                            m.sliding.theta,
                            &fast_pos,
                        );
                        let qr = qkv_unfused(
                            &rest_x,
                            &rest_pos,
                            &inputs,
                            m,
                            at,
                            d,
                            kv_heads,
                            qkv,
                            k_norm,
                            *k_norm_eps,
                            pages,
                        );
                        Value::merge(vec![qf, qr])
                    } else {
                        qkv_unfused(
                            &normed,
                            &positions,
                            &inputs,
                            m,
                            at,
                            d,
                            kv_heads,
                            qkv,
                            k_norm,
                            *k_norm_eps,
                            pages,
                        )
                    }
                }
            };

            seam::at(seam::ATTN_Q, &[&q]);

            let [mq, dq, p] = q.split(classes.clone());
            let a = Value::merge(vec![
                ops::attn::masked(
                    &mq,
                    &plan_m[at.reading as usize],
                    &mask,
                    pages,
                    win,
                    d,
                    at.sm_scale,
                ),
                ops::attn::decode(
                    &dq,
                    &plan_d[at.reading as usize],
                    pages,
                    win,
                    d,
                    at.sm_scale,
                ),
                ops::attn::prefill(
                    &p,
                    &plan_p[at.reading as usize],
                    pages,
                    win,
                    d,
                    kv_heads,
                    at.sm_scale,
                ),
            ]);
            seam::at(seam::ATTN_OUT, &[&a]);
            let o = ops::linear::matmul(&a, &w.o_proj);
            let o = if m.tp > 1 {
                ops::collective::all_reduce(&o)
            } else {
                o
            };
            // Adds the adapter's `B*(A*x)` in place, over the adapter window;
            // rows outside the window keep the unmodified value (write-through,
            // no merge). Applied after the reduce, before `post_attn_norm`.
            let o = {
                let (adapted, _) = o.split(&Facts::has_adapter());
                let (px, _) = normed.split(&Facts::has_adapter());
                ops::linear::lora_correct(&px, &w.lora_a, &w.lora_b, &routes, &adapted)
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
            // `None` on a dense layer: the whole MoE block is skipped and `f`
            // passes through unchanged.
            let f = match &w.moe {
                None => f,
                Some(x) => {
                    // Dense-layer FFN output would go straight to
                    // `post_ffw_norm`; here it has a routed sibling to sum with first.
                    let h1 = ops::elemwise::rmsnorm(
                        &f,
                        &x.post_ffw_norm_1,
                        x.post_ffw_norm_1_eps,
                    );
                    // Both branches and the router read `y` (the
                    // post-attention residual), not the dense branch's norm output.
                    let (routes, weights) = ops::linear::moe_topk_softmax_scaled(
                        &ops::linear::matmul(
                            &ops::elemwise::rmsnorm(
                                &y,
                                &x.router_norm,
                                x.router_norm_eps,
                            ),
                            &x.router,
                        ),
                        &x.per_expert_scale,
                        x.experts,
                        x.top_k,
                    );
                    let moe_in =
                        ops::elemwise::rmsnorm(&y, &x.pre_ffw_norm_2, x.pre_ffw_norm_2_eps);
                    // Routed activation is GeGLU (`gelu_approx(gate) * up`),
                    // same as the dense MLP's, just over `tokens * top_k` rows.
                    //
                    // Matches against dense dtypes (not quantized ones): an
                    // unrecognized quantized dtype should fall through to the
                    // quantized path, not the dense one.
                    let select = |act: &Value, bank: &Weight| {
                        if matches!(bank.dtype, Dtype::Bf16 | Dtype::F16 | Dtype::F32) {
                            ops::linear::moe_matmul_select(act, bank, &routes, x.top_k)
                        } else {
                            ops::linear::moe_matmul_select_quant(act, bank, &routes, x.top_k)
                        }
                    };
                    let hidden =
                        ops::linear::mlp_geglu_tanh_packed(&select(&moe_in, &x.gate_up), x.inter);
                    let routed =
                        ops::linear::moe_weighted_sum(&select(&hidden, &x.down), &weights);
                    // `down` is rows-cut so each rank holds a partial; weights
                    // are replicated, so summing then reducing equals
                    // reducing then summing.
                    let routed = if m.tp > 1 {
                        ops::collective::all_reduce(&routed)
                    } else {
                        routed
                    };
                    let h2 = ops::elemwise::rmsnorm(
                        &routed,
                        &x.post_ffw_norm_2,
                        x.post_ffw_norm_2_eps,
                    );
                    ops::elemwise::residual_add(&h1, &h2)
                }
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

            // Applied after the PLE relay's own addition, at the site the
            // branch above doesn't run. See `model::Layer::scalar`.
            if let Some(scalar) = &w.scalar {
                y = ops::elemwise::scale(scalar, &y);
            }
        }

        let x = ops::elemwise::rmsnorm(&y, &m.final_norm, m.final_norm_eps);
        let logits = ops::linear::lm_head(&x, &m.embed);
        let logits = if let Some(cap) = m.softcap {
            ops::attn::logit_softcap(&logits, cap)
        } else {
            logits
        };

        // Stated after the trunk's own readout: the export arena gives the
        // delivery tail to the last-stated node, so an earlier draft column
        // would let a vocabulary-wide GEMM be carved on top of the sampler's bytes.
        if let Some(a) = &m.draft {
            // Built inside the arm so a SKU without a draft head builds no plan for it.
            let (input_draft, _) = inputs.split(&Facts::drafts());
            let plan_draft = ops::attn::plan_prefill(
                &input_draft,
                m.q_heads,
                m.global.kv_heads,
                m.global.head_dim,
                None,
            );
            let (dx, _) = x.split(&Facts::drafts());
            let (dids, _) = ids.split(&Facts::drafts());

            // `[a|b]·[We|Wh]^T = a·We^T + b·Wh^T`: two matmuls and an add
            // since this IR has no concat op. No pre-fusion norms (EAGLE's
            // design). Embedding scale matches the trunk's own.
            let e = ops::layout::embed(&dids, &m.embed, m.vocab) * (m.hidden as f32).sqrt();
            let mut dy = ops::elemwise::residual_add(
                &ops::linear::matmul(&e, &a.fc_embed),
                &ops::linear::matmul(&dx, &a.fc_hidden),
            );

            let normed = ops::elemwise::rmsnorm(&dy, &a.attn_norm, a.norm_eps);
            let o = draft_attn(&normed, &inputs, m, &plan_draft, a);
            let o = if m.tp > 1 {
                ops::collective::all_reduce(&o)
            } else {
                o
            };
            dy = ops::elemwise::residual_add(
                &ops::elemwise::rmsnorm(&o, &a.post_attn_norm, a.norm_eps),
                &dy,
            );
            let mlp_in = ops::elemwise::rmsnorm(&dy, &a.pre_ffw_norm, a.norm_eps);
            let act =
                ops::linear::mlp_geglu_tanh_packed(&ops::linear::matmul(&mlp_in, &a.gate_up), a.inter);
            let f = ops::linear::matmul(&act, &a.down);
            let f = if m.tp > 1 {
                ops::collective::all_reduce(&f)
            } else {
                f
            };
            dy = ops::elemwise::residual_add(
                &ops::elemwise::rmsnorm(&f, &a.post_ffw_norm, a.norm_eps),
                &dy,
            );

            // Uses the trunk's own lm_head and softcap.
            let draft = ops::linear::lm_head(
                &ops::elemwise::rmsnorm(&dy, &m.final_norm, m.final_norm_eps),
                &m.embed,
            );
            let draft = match m.softcap {
                Some(cap) => ops::attn::logit_softcap(&draft, cap),
                None => draft,
            };
            seam::at(seam::MTP, &[&draft]);
        }

        logits
    }
}

/// The aux head's attention: this family's own global site, over the head's
/// own kv row, on one prefill schedule.
fn draft_attn(x: &Value, inputs: &Input<Facts>, m: &Model, plan: &Value, a: &Draft) -> Value {
    let at = &a.attn;
    let d = m.global.head_dim;
    let pages = inputs.kv(&at.kv);
    let AttnBanks::Owned {
        qkv,
        k_norm,
        k_norm_eps,
    } = &at.banks
    else {
        panic!("an aux head owns its own qkv and borrows nobody's kv row");
    };
    // `x` already carries the draft window; a `Guard::Always` input narrows
    // to whatever window it's read beside.
    let q = qkv_unfused(
        x,
        &inputs.positions(),
        inputs,
        m,
        at,
        d,
        m.global.kv_heads,
        qkv,
        k_norm,
        *k_norm_eps,
        pages,
    );
    let o = ops::attn::prefill(&q, plan, pages, None, d, m.global.kv_heads, at.sm_scale);
    ops::linear::matmul(&o, &a.o_proj)
}

/// Clamp input to checkpoint-stated bounds, apply the bank, clamp output —
/// or just the bank, when the tower has no clip bounds.
///
/// Bounds are weights, not plan constants: trace building has no checkpoint
/// present yet. The `None` arm emits no clamp at all (not a ±∞ one), since
/// an unclipped tower has no clip planes to read.
fn clipped(x: &Value, c: &Clippable) -> Value {
    let Some(k) = &c.clip else {
        return ops::linear::matmul(x, &c.bank);
    };
    let held = ops::elemwise::clamp_learned(x, &k.in_lo, &k.in_hi);
    ops::elemwise::clamp_learned(&ops::linear::matmul(&held, &c.bank), &k.out_lo, &k.out_hi)
}

/// Gemma's vision tower, as one capture unit (emitted before any trunk
/// node). Returns the pooled `[Dim::Patches, trunk hidden]` rectangle whose
/// leading `rows / pool^2` rows are live.
fn tower(inputs: &Input<Facts>, t: &Tower) -> Value {
    let d = t.head_dim;
    let x = inputs.patches(t.patch_width);
    let segments = inputs.patch_segments();
    let grid = inputs.patch_positions();

    // Position table is two separable lookups: one `embed_weighted` over
    // both axis tables laid end to end, weights of one.
    let mut y = ops::linear::matmul(&x, &t.patch_embed);
    let pos = ops::layout::embed_weighted(
        &inputs.patch_embed_rows(2),
        &inputs.patch_embed_weights(2),
        &t.pos_embed,
        t.positions,
    );
    y = ops::elemwise::residual_add(&pos, &y);

    for b in &t.blocks {
        let n = ops::elemwise::rmsnorm(&y, &b.attn_norm, t.norm_eps);
        let q = ops::elemwise::rmsnorm_per_head(&clipped(&n, &b.q), &b.q_norm, d, t.norm_eps);
        let k = ops::elemwise::rmsnorm_per_head(&clipped(&n, &b.k), &b.k_norm, d, t.norm_eps);
        // A per-head norm with no learned weight (same as the trunk's `v` leg).
        let v = ops::elemwise::rmsnorm_no_scale(&clipped(&n, &b.v), d, t.norm_eps);
        // RoPE restarts per spatial axis (`MropeForm::Blocked`), matching
        // upstream's `compute_default_rope_parameters`.
        let (q, k) = ops::elemwise::rope_mrope(
            &q,
            &k,
            &grid,
            [0, d / 4, d / 4],
            MropeForm::Blocked,
            d,
            d,
            t.theta,
        );
        let o = ops::attn::dense(&q, &k, &v, &segments, d, t.sm_scale);
        y = ops::elemwise::residual_add(
            &ops::elemwise::rmsnorm(&clipped(&o, &b.o), &b.post_attn_norm, t.norm_eps),
            &y,
        );

        let n = ops::elemwise::rmsnorm(&y, &b.pre_ffw_norm, t.norm_eps);
        // Unfused: gate and up clamp to different learned bounds, so they
        // can't share a packed bank.
        let act = ops::linear::mlp_geglu_tanh(&clipped(&n, &b.gate), &clipped(&n, &b.up));
        y = ops::elemwise::residual_add(
            &ops::elemwise::rmsnorm(&clipped(&act, &b.down), &b.post_ffw_norm, t.norm_eps),
            &y,
        );
    }

    // Order: pool, scale by sqrt(hidden), standardize, project — matching
    // `Gemma4VisionModel.forward`. Standardize brings the magnitude back
    // down after the scale.
    let pooled = ops::layout::pool_rows(&y, t.pool);
    let pooled = pooled * (t.hidden as f32).sqrt();
    let pooled = match &t.std {
        None => pooled,
        Some(std) => ops::elemwise::standardize(&pooled, &std.bias, &std.scale),
    };
    ops::linear::matmul(&pooled, &t.projection)
}

fn qkv_unfused(
    x: &Value,
    pos: &Value,
    inputs: &Input<Facts>,
    m: &Model,
    at: &Attn,
    d: u32,
    kv_heads: u32,
    qkv: &Weight,
    k_norm: &Weight,
    k_norm_eps: f32,
    pages: ValueId,
) -> Value {
    let write_page = inputs.write_page(&at.kv);
    let write_offset = inputs.write_offset(&at.kv);
    let (q, k, v) =
        ops::layout::split_qkv(&ops::linear::matmul(x, qkv), m.q_heads * d, kv_heads * d);
    let v = ops::elemwise::rmsnorm_no_scale(&v, d, at.q_norm_eps);
    let q = ops::elemwise::rmsnorm_per_head(&q, &at.q_norm, d, at.q_norm_eps);
    let k = ops::elemwise::rmsnorm_per_head(&k, k_norm, d, k_norm_eps);
    let (q, k) = match at.reading {
        Reading::Global => {
            ops::elemwise::rope_partial(&q, &k, pos, m.global.rotary_dim, d, m.global.theta)
        }

        Reading::Sliding => ops::elemwise::rope_full(&q, &k, pos, d, m.sliding.theta, false),
    };
    ops::attn::kv_append(&k, &v, pages, &write_page, &write_offset);
    q
}

fn q_only(x: &Value, pos: &Value, m: &Model, at: &Attn, d: u32, q_proj: &Weight) -> Value {
    let q = ops::elemwise::rmsnorm_per_head(
        &ops::linear::matmul(x, q_proj),
        &at.q_norm,
        d,
        at.q_norm_eps,
    );
    match at.reading {
        Reading::Global => {
            ops::elemwise::rope_partial_q(&q, pos, m.global.rotary_dim, d, m.global.theta)
        }
        Reading::Sliding => ops::elemwise::rope_partial_q(&q, pos, d, d, m.sliding.theta),
    }
}
