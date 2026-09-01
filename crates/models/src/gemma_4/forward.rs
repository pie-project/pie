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

    /// **THE MEDIA WINDOW, AND IT GUARDS THE MERGE AND NOTHING ELSE**
    /// (multimodal §15). The tower's rectangles are `Dim::Patches`, so an
    /// axis-empty fire has zero patch rows and `engine::fire::walk` skips the
    /// whole unit; the embed merge writes TOKEN rows, so its window is full on
    /// every fire and unguarded it would resolve `PatchRoutes` on a text-only
    /// one. Bit three, after `has_adapter`, for the reason `masked` kept bit
    /// one: a word is what the runtime hands the shell and renumbering it
    /// moves classes for nothing.
    pub fn media() -> Predicate {
        Predicate::fact(3)
    }

    /// **THE DRAFT WINDOW** (palo C3, design §8; campaign M-4).
    ///
    /// A lane that wants the aux head run over its rows. The head is a whole
    /// decoder block plus two projections and a vocabulary-wide readout, and
    /// the bit is what makes it cost a non-drafting fire NOTHING: zero rows in
    /// this class, and `engine::fire::walk` does not issue the arm's launches,
    /// does not issue them empty, and does not record them.
    ///
    /// AND IT IS AN ARM, NOT A CORRECTION: the draft logits are a SECOND
    /// readout with a column of their own at `seam::MTP`, so a class outside
    /// the window reads no draft at all — which is the honest answer.
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
        // The head's own kv row, in the SAME page-id space: it attends the
        // same sequence at the same lengths the trunk does, so a second space
        // would be a second bit-identical page table and a false distinction.
        // Its planes are a GLOBAL layer's, because that is the reading its
        // block states.
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

        // **THE TOWER FIRST**, because the two capture units are the two RUNS
        // of this node list and a trunk node emitted before it would interleave
        // them (`Error::UnitsInterleave`).
        let towered = m.tower.as_ref().map(|t| tower(&inputs, t));

        let ids = inputs.tokens();
        let mut y = ops::layout::embed(&ids, &m.embed, m.vocab) * (m.hidden as f32).sqrt();

        // The embed merge, over the media window and nowhere else (§15), with
        // the drop sentinel `pool_rows` compaction owes it (§8.6). The value
        // comes back unguarded: the column it writes through is the embedding
        // every row already has.
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
            // **THE SECOND BRANCH, WHERE THE CHECKPOINT SHIPS ONE**
            // (`model::Moe`). `None` on every dense row, and then this whole
            // block is not written: `f` reaches the sandwich's closing norm
            // exactly as it did, one statement and the same one.
            let f = match &w.moe {
                None => f,
                Some(x) => {
                    // The dense branch's own exit norm. It exists only here —
                    // a dense layer's FFN output goes straight into
                    // `post_ffw_norm`, and this one has a sibling to be added
                    // to first.
                    let h1 = ops::elemwise::rmsnorm(
                        &f,
                        &x.post_ffw_norm_1,
                        x.post_ffw_norm_1_eps,
                    );
                    // **BOTH BRANCHES AND THE ROUTER READ `y`**, the
                    // post-attention residual, and NOT the dense branch's
                    // norm. `mlx_lm` routes on `h` and norms `h` twice more;
                    // reading `mlp_in` here would chain what the checkpoint
                    // states as siblings.
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
                    // **THE EXPERTS ARE GeGLU, NOT SwiGLU**, and that costs no
                    // op: `mlp_geglu_tanh_packed` reads a `[rows, 2·inter]`
                    // block and the routed matmul writes one — the rows are
                    // `tokens × top_k` rather than `tokens`, which the
                    // activation does not ask about. `SwitchGLU`'s activation
                    // is `GeGLU()`, which is `gelu_approx(gate) * up` — the
                    // dense `mlp`'s own `geglu`, on the routed stack.
                    //
                    // **THE DENSE FORMS ARE THE LIST, AND EVERYTHING ELSE IS
                    // QUANTIZED** — `qwen_4::forward`'s twin of this line was
                    // an ALLOW-list of quantized dtypes and the 2-bit row it
                    // never learned fell through to the DENSE select, which
                    // resolves three planes as one handle and panics. Today
                    // the two readings agree on every row this family ships
                    // (`Bf16` dense both ways, `U4g64` quantized both ways);
                    // they stop agreeing the day a dtype naming a different
                    // group lands, and a new quantized dtype is a thing this
                    // tree adds where a new DENSE one is not.
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
                    // After the fold and before the branch norm, for
                    // `o_proj`'s reason: `down` is rows-cut, so each rank holds
                    // a partial product, and the mixture's weights are
                    // replicated — so the weighted sum of the partials is the
                    // partial of the weighted sum, and one reduce here answers
                    // both.
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
        let logits = if let Some(cap) = m.softcap {
            ops::attn::logit_softcap(&logits, cap)
        } else {
            logits
        };

        // **THE AUX HEAD, OVER THE DRAFT WINDOW** (campaign M-4).
        //
        // STATED AFTER THE TRUNK'S READOUT, and that is not cosmetic:
        // `model_compiler::arena` gives the delivery tail to the export set,
        // and a draft column stated before the trunk's `lm_head` would have
        // its span end at its own producing node with a vocabulary-wide GEMM
        // free to be carved on top of the bytes the sampler reads. Stated
        // last, nothing runs after it. `qwen_3::forward` argues it at length
        // and `the_draft_readout_outlives_the_trunk_readout` notices if either
        // moves.
        if let Some(a) = &m.draft {
            // Minted inside the arm and off the draft window's own arm of the
            // inputs, so the schedule states which class it was carved for and
            // a SKU with no head carries no plan build nothing reads.
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

            // The fusion: `[a|b]·[Wₑ|W_h]ᵀ = a·Wₑᵀ + b·W_hᵀ`, two matmuls and
            // one add because this IR states no concatenation. RAW streams —
            // EAGLE has no pre-fusion norms, which is the recipe and not an
            // omission (`Draft`'s own note).
            //
            // The embedding is scaled the way this family scales every
            // embedding it reads, because that is what `layout.embed` answers
            // here and the head was trained against the trunk's own stream.
            let e = ops::layout::embed(&dids, &m.embed, m.vocab) * (m.hidden as f32).sqrt();
            let mut dy = ops::elemwise::residual_add(
                &ops::linear::matmul(&e, &a.fc_embed),
                &ops::linear::matmul(&dx, &a.fc_hidden),
            );

            // One block, in THIS FAMILY'S four-norm sentence.
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

            // The readout, through the BASE head and past the same softcap the
            // trunk's answer goes through: one vocabulary projection in this
            // model, and one squashing of it.
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
    // Taken unrefined, as the trunk's own mixer takes them: `x` already
    // carries the draft window, and a `Guard::Always` runtime input narrows to
    // the window of whatever it is read beside.
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

/// **A CLIPPABLE LINEAR**: clamp the input to the bounds the checkpoint
/// states, apply the bank, clamp the output (multimodal §12) — or, where the
/// tower states `use_clipped_linears: false`, just the bank.
///
/// The bounds are WEIGHTS and not plan constants, because a trace is built
/// with no checkpoint in the room and these are 448 learned scalars —
/// `elemwise::clamp_learned` is the form that reads them, and its precedent is
/// `Elementwise::Scale`, which has read a device-held scalar for the same
/// reason since before this axis existed.
///
/// **AND THE `None` ARM EMITS NOTHING, NOT A ±∞ CLAMP.** Upstream's buffers
/// initialize to infinities and a tower that shipped them would be clamping
/// to nothing at all; a text that spelled that anyway would put fourteen
/// inert elementwise launches in every block of a 27-block tower — 378 a
/// fire — and would need two `[1]` planes per projection that the checkpoint
/// does not hold.
fn clipped(x: &Value, c: &Clippable) -> Value {
    let Some(k) = &c.clip else {
        return ops::linear::matmul(x, &c.bank);
    };
    let held = ops::elemwise::clamp_learned(x, &k.in_lo, &k.in_hi);
    ops::elemwise::clamp_learned(&ops::linear::matmul(&held, &c.bank), &k.out_lo, &k.out_hi)
}

/// **GEMMA'S TOWER, AS ONE FUNCTION AND ONE CAPTURE UNIT** (multimodal §12).
///
/// Emitted before one trunk node, for `qwen_3::tower`'s reason: a unit is a
/// RUN of the node list. Returns the pooled `[Dim::Patches, trunk hidden]`
/// rectangle whose leading `rows / pool²` rows are live.
fn tower(inputs: &Input<Facts>, t: &Tower) -> Value {
    let d = t.head_dim;
    let x = inputs.patches(t.patch_width);
    let segments = inputs.patch_segments();
    let grid = inputs.patch_positions();

    // The patch embed is a plain matmul — `input_proj` carries no bias and no
    // clip — and the position table is TWO separable lookups summed, which is
    // one `embed_weighted` over the two axis tables laid end to end with
    // weights of one (`Tower`'s own note argues why that is exact).
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
        // `with_scale = False`: a per-head norm that reads no weight, which is
        // the line the trunk's own `qkv_unfused` writes for its `v`.
        let v = ops::elemwise::rmsnorm_no_scale(&clipped(&n, &b.v), d, t.norm_eps);
        // Two axes, contiguous blocks, and the ladder RESTARTS per axis --
        // `compute_default_rope_parameters` says so in its own comment
        // ("computes RoPE frequencies INDEPENDENTLY for each spatial
        // dimension using the partitioned head_dim"), which is exactly what
        // `MropeForm::Blocked` means.
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
        // UNFUSED, and §12 argues it: `gate` and `up` read the same `n` and
        // clamp it to DIFFERENT learned bounds, so they cannot share a packed
        // bank the way the trunk's `gate_up` does.
        let act = ops::linear::mlp_geglu_tanh(&clipped(&n, &b.gate), &clipped(&n, &b.up));
        y = ops::elemwise::residual_add(
            &ops::elemwise::rmsnorm(&clipped(&act, &b.down), &b.post_ffw_norm, t.norm_eps),
            &y,
        );
    }

    // The pool, then the scaling the pooler states, then the standardization
    // the tower states, then the projection that makes a soft token a token
    // row — `Gemma4VisionModel.forward`'s own last four lines, in its own
    // order.
    //
    // The `√hidden` is `Gemma4VisionPooler`'s, and upstream computes it in
    // f32 because "the sqrt(hidden_size) scaling can push the activations
    // past the float16 range (max 65504)". This rectangle is bf16, which has
    // f32's exponent and none of that range problem; what it gives up is
    // mantissa, and the standardize below is the step that brings the
    // magnitude back down — which is the same reason upstream defers the cast
    // to after it rather than before.
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
