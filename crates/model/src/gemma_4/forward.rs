use model_dsl::{
    Classify, ForwardHybrid, HybridSpec, Input, Platform, Predicate, Request, Value, ValueId,
    Weight, ops, seam,
};

use super::model::{Attn, AttnBanks, Model, Reading};

pub struct Facts {
    pub qo_one: bool,
    pub masked: bool,
    pub has_adapter: bool,
}

impl Facts {
    pub fn qo_one() -> Predicate {
        Predicate::fact(0)
    }

    pub fn masked() -> Predicate {
        Predicate::fact(1)
    }

    /// **THE ADAPTER WINDOW** (palo design §8, §0; campaign A-6).
    ///
    /// A lane that routed its rows to a registered adapter. §8 puts the
    /// correction "over the adapter window", and §0 defines a window as the
    /// rows of the lanes whose word satisfies the guard — so the axis needs a
    /// bit, and the bit is what makes it FREE when nobody uses it: a fire no
    /// lane routed has zero rows in this class, `engine::fire::walk` skips a
    /// zero-row region before it dispatches anything, and the correction costs
    /// that fire no launch, no empty grid and no instruction. A `Guard::Always`
    /// correction would instead launch two kernels per layer over every row of
    /// every fire to add zero to them, which is 1.0x nothing.
    ///
    /// **BIT TWO, AND THE POSITION IS DELIBERATE.** The masked window was
    /// declared at bit one before this axis existed, and a word is what the
    /// runtime hands the shell — renumbering it would move every masked lane's
    /// class for no reason but tidiness. The correction is orthogonal to the
    /// attention split anyway: it reads no plan and takes no arm of the merge,
    /// so where its bit sits changes nothing but the arithmetic of the word.
    pub fn has_adapter() -> Predicate {
        Predicate::fact(2)
    }
}

impl Classify for Facts {
    fn of(r: &Request) -> Facts {
        Facts {
            qo_one: r.query_len() == 1,
            masked: r.has_custom_mask(),
            has_adapter: r.has_adapter(),
        }
    }

    fn word(&self) -> u64 {
        u64::from(self.qo_one)
            | (u64::from(self.masked) << 1)
            | (u64::from(self.has_adapter) << 2)
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
        c
    }

    fn forward(&self, inputs: Input<Facts>) -> Value {
        let m = self;

        let positions = inputs.positions();
        let qo_one = Facts::qo_one();
        let fused = qo_one.clone() & !Facts::masked();

        // SIX SCHEDULES, AND EVERY ONE OF THEM IS A DIFFERENT CARVING.
        //
        // A gemma layer is one of two READINGS of the one sequence — sliding
        // layers are 2 kv heads of 256 under a 512-wide window, global layers
        // 2 heads of 512 with none — and an attention schedule is carved for
        // exactly one of those: the head width picks the CTA tile, the window
        // sizes the kv chunking, and the kernel recomputes both from the same
        // numbers when it merges the partials. So the reading axis is the
        // literal pair below, whose two lines differ by the reading's numbers
        // — the ones each plan op now states — and the layer indexes it.
        //
        // The class axis is the split. One `plan_prefill` read by both the
        // prefill arm (the rest class) and the masked arm (its own) would be
        // carved over both windows, and each arm would then hand it its own
        // rebased boundaries, which end before the schedule's work items do —
        // the shell refuses that by name at load (`Fault::Straddled`). So each
        // plan is built off ONE class's arm of `inputs` and carries that class
        // as its guard. The layer loop cuts `q` with the SAME `classes` array,
        // so the arms carry structurally equal conds and the recorder refuses
        // a reader from another arm at the line that mixes them.
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

        let routes = inputs.adapter_routes();
        for (l, w) in inputs.walk_layers(&m.layers) {
            let normed = ops::elemwise::rmsnorm(&y, &w.attn_norm, w.attn_norm_eps);
            let at = &w.attn;
            // Every number this layer's attention is carved for comes off its
            // reading, once: the head width, the kv heads, and the window.
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
            // **THE CORRECTION, OVER ITS WINDOW** (design §8, campaign A-6).
            // One statement: the sublayer's output, plus this row's adapter's
            // `B·(A·x)`, in place. No merge and no arm — the op writes THROUGH
            // `o`'s arena column, so a class outside the window never runs the
            // node and reads the uncorrected value at the same address, which
            // is the identity for free.
            //
            // After the reduce and before `post_attn_norm`; `Layer::lora_a`'s
            // own note argues both.
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

            // **THE PER-LAYER SCALAR, AFTER EVERYTHING THE LAYER DID.**
            // `gemma4_text`'s decoder ends `h = h * self.layer_scalar`, after
            // the PLE relay's own addition and not before it — so this is the
            // same statement as the one closing the branch above, at the one
            // site the branch does not run. See `model::Layer::scalar` for why
            // exactly one of the two fires.
            if let Some(scalar) = &w.scalar {
                y = ops::elemwise::scale(scalar, &y);
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
