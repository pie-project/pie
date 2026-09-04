use model_dsl::{
    Classify, Dtype, ForwardHybrid, GateActivation, HybridSpec, Input, MropeForm, Predicate,
    Request, Value, Weight, ops, seam,
};

use super::model::{Attn, DFlash, DRAFT_DEPTH, Gdn, Head, Mixer, Mlp, Model, Tower};

/// The trunk's mrope section split; both qwen SKUs share `[11, 11, 10]`,
/// summing to half `rotary_dim`.
const MROPE_SECTIONS: [u32; 3] = [11, 11, 10];

pub struct Facts {
    pub qo_one: bool,
    pub has_adapter: bool,
    pub drafts: bool,
    pub captures_scores: bool,
    pub masked: bool,
    pub media: bool,
    /// The rows are a block drafter's proposal — see [`Facts::block_draft`].
    pub block_draft: bool,
}

impl Facts {
    #[must_use]
    pub fn qo_one() -> Predicate {
        Predicate::fact(0)
    }

    /// Lanes with an adapter routed. Guards the LoRA correction so a fire
    /// with no adapter lane issues no launch for it.
    #[must_use]
    pub fn has_adapter() -> Predicate {
        Predicate::fact(1)
    }

    /// Lanes that want the MTP draft head run over their rows. The draft
    /// logits are a separate `[rows, vocab]` export, not an in-place
    /// correction.
    #[must_use]
    pub fn drafts() -> Predicate {
        Predicate::fact(2)
    }

    /// Lanes that want their attention's per-query LSE kept. Ordered before
    /// `qo_one` in the split so a capturing lane takes the capture arm
    /// regardless of row count; only `masked` outranks it.
    #[must_use]
    pub fn captures_scores() -> Predicate {
        Predicate::fact(3)
    }

    /// Lanes that brought their own attention mask instead of the causal
    /// one. Applies at attention layers only — GDN layers recur with no
    /// per-key score for a mask to veto.
    #[must_use]
    pub fn masked() -> Predicate {
        Predicate::fact(4)
    }

    /// Lanes that submitted images. Guards the embed merge only, not the
    /// tower: the tower's rows are already zero on an image-free fire, but
    /// the merge writes into the token axis, which is never empty.
    #[must_use]
    pub fn media() -> Predicate {
        Predicate::fact(5)
    }

    /// Lanes whose rows are a block drafter's proposal rather than the
    /// sequence's own — `[anchor, MASK x block-1]`, which the trunk must not
    /// run over. Declared ahead of the arm that reads it, so the fact a lane
    /// carries and the plan that guards on it land in one place.
    #[must_use]
    pub fn block_draft() -> Predicate {
        Predicate::fact(6)
    }
}

impl Classify for Facts {
    fn of(r: &Request) -> Facts {
        Facts {
            qo_one: r.query_len() == 1,
            has_adapter: r.has_adapter(),
            drafts: r.drafts(),
            captures_scores: r.captures_scores(),
            masked: r.has_custom_mask(),
            media: r.has_media(),
            block_draft: r.drafts_a_block(),
        }
    }

    fn word(&self) -> u64 {
        u64::from(self.qo_one)
            | (u64::from(self.has_adapter) << 1)
            | (u64::from(self.drafts) << 2)
            | (u64::from(self.captures_scores) << 3)
            | (u64::from(self.masked) << 4)
            | (u64::from(self.media) << 5)
            | (u64::from(self.block_draft) << 6)
    }
}

impl ForwardHybrid for Model {
    type Facts = Facts;

    fn caches(&self) -> HybridSpec {
        let mut c = HybridSpec::new();
        let kv = c.kv_space(self.kv);
        let plane = self.kv_heads as u64 * self.head_dim as u64;
        for w in &self.layers {
            match &w.mixer {
                Mixer::Attn(a) => {
                    c.kv(kv, a.kv.clone(), [plane, plane]);
                }

                Mixer::Gdn(g) => {
                    let conv_ch = u64::from(Gdn::qkv_width(g.k_heads, g.v_heads, g.k_dim, g.v_dim));
                    c.state(g.conv_state.clone(), [g.conv_kernel as u64, conv_ch], Dtype::Bf16);
                    c.state(
                        g.delta_state.clone(),
                        [g.v_heads as u64, g.k_dim as u64, g.v_dim as u64],
                        Dtype::Bf16,
                    );
                }
            }
        }
        // The draft head's kv row shares the trunk's page-id space and plane
        // width: it attends the same sequence at the same lengths.
        if let Some(mtp) = &self.mtp {
            let a = &mtp.attn;
            c.kv(kv, a.kv.clone(), [plane, plane]);
        }
        // The block drafter brings five layers, so five rows — in the same
        // page-id space for the same reason, and at the same plane width,
        // which its own geometry happens to land on (8 kv heads x 128
        // against the trunk's 4 x 256; see `DFlash`).
        if let Some(dflash) = &self.dflash {
            for b in &dflash.blocks {
                let a = &b.attn;
                let dplane = a.kv_heads as u64 * a.head_dim as u64;
                c.kv(kv, a.kv.clone(), [dplane, dplane]);
            }
        }
        c
    }

    fn forward(&self, inputs: Input<Facts>) -> Value {
        let m = self;

        // One schedule per class, each built off the one arm that reads it.
        // Masked is ordered before captures: a lane asking for both gets the
        // masked arm with no scores.
        let classes = [
            Facts::masked(),
            Facts::captures_scores(),
            Facts::qo_one(),
            Predicate::rest(),
        ];
        // A plan is CONSUMED inside the trunk's own arm, so it must be built
        // there too: `compatible` lets `Always` meet anything, but two
        // stated guards must be equal, and a plan spelled `captures` cannot
        // meet a query spelled `block-draft AND captures`. Splitting the
        // drafter's rows off first makes both sides say the same thing.
        let (_, trunk_inputs) = match &m.dflash {
            Some(_) => inputs.split(&Facts::block_draft()),
            None => (inputs.clone(), inputs.clone()),
        };
        let [input_m, input_s, input_d, input_p] = trunk_inputs.split(classes);
        let plan_m = ops::attn::plan_prefill(&input_m, m.q_heads, m.kv_heads, m.head_dim, None);
        let plan_d = ops::attn::plan_decode(&input_d, m.q_heads, m.kv_heads, m.head_dim, None);
        let plan_p = ops::attn::plan_prefill(&input_p, m.q_heads, m.kv_heads, m.head_dim, None);
        let plan_s = ops::attn::plan_prefill(&input_s, m.q_heads, m.kv_heads, m.head_dim, None);
        let mask = inputs.mask();

        // Tower nodes must all be emitted before any trunk node, or
        // `model_compiler` refuses the plan (`Error::UnitsInterleave`).
        let towered = m.tower.as_ref().map(|t| tower(&inputs, t));

        let ids = inputs.tokens();
        let mut y = ops::layout::embed(&ids, &m.embed, m.vocab);

        // Tower rows written over the token rows the image placeholders
        // occupy. Uses `scatter_live_rows`, not a plain scatter, since
        // `merge_rows` compacts and the tail routes carry a `-1` sentinel.
        if let Some(t) = &towered {
            let (imaged, _) = y.split(&Facts::media());
            y = ops::layout::scatter_live_rows(t, &inputs.patch_routes(), &imaged).everywhere();
        }

        // **THE BLOCK DRAFTER'S ROWS LEAVE HERE.** They carry
        // `[anchor, MASK x block-1]`, which is not this sequence — the 64
        // layers below must not run over them. Every trunk node downstream
        // is guarded by this split, so a fire of nothing but draft rows
        // leaves each of their classes empty, and an empty class is not
        // walked at all (`engine_metal::window`): a draft fire pays for no
        // trunk layer. `h_block` is already the drafter's input, since the
        // drafter shares the target's embedding.
        let (h_block, mut y) = match &m.dflash {
            Some(_) => {
                let (block, rest) = y.split(&Facts::block_draft());
                (Some(block), rest)
            }
            None => (None, y),
        };

        // The trunk hidden states the drafter was trained against, kept in
        // tap order as the loop passes them.
        // **THE TAPS ARE FUSED WHERE THEY ARE TAKEN, NOT COLLECTED.** A residual
    // add ALIASES its output onto the stream it folds into
    // (`Elementwise::aliases`), so the trunk's hidden state is ONE buffer and
    // a handle held across a later layer reads that layer's value, not the
    // tapped one. The fusion's `[hidden, taps·hidden]` bank is its column
    // slices summed, and a slice's matmul allocates — so taking the tap's
    // product here is both the fusion and the snapshot, at no extra cost.
    let mut fused: Option<Value> = None;
        let routes = inputs.adapter_routes();
        for (l, w) in inputs.walk_layers(&m.layers) {
            let x = ops::elemwise::rmsnorm_plus_one(&y, &w.mixer_norm, w.mixer_norm_eps);
            let o = match &w.mixer {
                Mixer::Attn(a) => {
                    attn_mixer(&x, &inputs, m, &plan_m, &plan_d, &plan_p, &plan_s, &mask, a)
                }
                Mixer::Gdn(g) => gdn_mixer(&x, &inputs, g),
            };
            let o = if m.tp > 1 {
                ops::collective::all_reduce(&o)
            } else {
                o
            };
            // The LoRA correction, in place over the adapter window, applied
            // after the tp reduce so a rows-cut partial product isn't summed
            // `tp` times.
            let o = {
                let (adapted, _) = o.split(&Facts::has_adapter());
                let (px, _) = x.split(&Facts::has_adapter());
                ops::linear::lora_correct(&px, &w.lora_a, &w.lora_b, &routes, &adapted)
            };
            y = ops::elemwise::residual_add(&o, &y);

            let x = ops::elemwise::rmsnorm_plus_one(&y, &w.mlp_norm, w.mlp_norm_eps);
            let f = match &w.mlp {
                Mlp::Dense {
                    gate_up,
                    down,
                    inter,
                } => ops::linear::matmul(
                    &ops::linear::mlp_swiglu(&ops::linear::matmul(&x, gate_up), *inter),
                    down,
                ),
                Mlp::Routed {
                    router,
                    gate_up,
                    down,
                    shared_gate_up,
                    shared_down,
                    shared_gate,
                    experts,
                    top_k,
                    inter,
                    shared_inter,
                } => {
                    let (routes, weights) = ops::linear::moe_topk_softmax(
                        &ops::linear::matmul(&x, router),
                        *experts,
                        *top_k,
                    );
                    // A packed expert bank is two or three device planes, not
                    // one dense handle, so the select op is chosen off the
                    // bank's own dtype: dense forms are the explicit list,
                    // everything else goes through the quantized reader.
                    let select = |act: &Value, bank: &Weight| {
                        if matches!(bank.dtype, Dtype::Bf16 | Dtype::F16 | Dtype::F32) {
                            ops::linear::moe_matmul_select(act, bank, &routes, *top_k)
                        } else {
                            ops::linear::moe_matmul_select_quant(act, bank, &routes, *top_k)
                        }
                    };
                    let hidden = ops::linear::mlp_swiglu(&select(&x, gate_up), *inter);
                    let routed = ops::linear::moe_weighted_sum(&select(&hidden, down), &weights);
                    let shared = ops::linear::matmul(
                        &ops::linear::mlp_swiglu(
                            &ops::linear::matmul(&x, shared_gate_up),
                            *shared_inter,
                        ),
                        shared_down,
                    );
                    ops::linear::moe_sigmoid_gate_add(
                        &routed,
                        &shared,
                        &ops::linear::matmul(&x, shared_gate),
                    )
                }
            };
            let f = if m.tp > 1 {
                ops::collective::all_reduce(&f)
            } else {
                f
            };
            y = ops::elemwise::residual_add(&f, &y);
            if let Some(at) = m
                .dflash
                .as_ref()
                .and_then(|d| d.taps.iter().position(|t| *t == l))
            {
                let d = m.dflash.as_ref().expect("a tap index came from it");
                let part = ops::linear::matmul(&y, &d.fc[at]);
                fused = Some(match fused {
                    Some(sum) => ops::elemwise::residual_add(&part, &sum),
                    None => part,
                });
            }
        }

        let x = ops::elemwise::rmsnorm_plus_one(&y, &m.final_norm, m.final_norm_eps);
        let head = match &m.head {
            Head::Tied => &m.embed,
            Head::Bank(bank) => bank,
        };

        // The drafter reads out through the TARGET's head, so its rows join
        // the trunk's before the one `lm_head` rather than after it — which
        // is what sharing the head means. Merge order is the split's.
        let x = match (&m.dflash, h_block) {
            (Some(d), Some(block)) => {
                let fused = fused.as_ref().expect("a block drafter tapped the trunk");
                let hb = dflash_arm(d, &inputs, fused, &block, &mask);
                Value::merge(vec![hb, x])
            }
            _ => x,
        };

        let logits = ops::linear::lm_head(&x, head);

        // The block's proposals: one draft a block row, the same
        // `[rows, depth]` seam shape the chained heads plant at depth one.
        if m.dflash.is_some() {
            let (dlogits, _) = logits.split(&Facts::block_draft());
            seam::at(seam::MTP, &[&dlogits]);
            seam::at(seam::MTP_DRAFTS, &[&ops::layout::argmax(&[&dlogits])]);
        }

        // Stated after the trunk's readout, so a draft column doesn't share
        // address space with the trunk's `lm_head` output.
        if let Some(mtp) = &m.mtp {
            // Minted inside the arm and off the draft window's own inputs, so
            // a SKU with no draft head carries no plan build for it.
            let (input_mtp, _) = inputs.split(&Facts::drafts());
            let plan_mtp =
                ops::attn::plan_prefill(&input_mtp, m.q_heads, m.kv_heads, m.head_dim, None);
            let (dx, _) = x.split(&Facts::drafts());
            // The token the head pairs with row r's hidden is THE TRUNK'S
            // ARGMAX there — the module is trained on `(h_t, x_{t+1}) → x_{t+2}`,
            // and the verifier's window opens on that same argmax, so the
            // draft is the token after the correction (as the dsv4, gemma4
            // and qwen4 arms read it). Row r's own id pairs it one position
            // early and the head drafts nothing the trunk accepts.
            let (dlogits, _) = logits.split(&Facts::drafts());
            let mut chosen = ops::layout::argmax(&[&dlogits]);
            let mut hidden = dx;
            // The chain: `DRAFT_DEPTH` passes of the one shipped block, each
            // fed the previous pass's argmax and residual — the shape the
            // qwen4 head runs. The checkpoint trains one step; every step
            // past it is the head run past its training, and pays only if
            // its acceptance clears the round's extra row (model.rs).
            let mut chain: Vec<Value> = Vec::with_capacity(DRAFT_DEPTH as usize);
            for step in 0..DRAFT_DEPTH {
                // `[a|b]*[We|Wh]^T = a*We^T + b*Wh^T`, as two matmuls and one
                // add since the IR has no concatenation. The pre-norms are the
                // recipe's: MTP normalizes each stream first, EAGLE fuses the
                // raw pair (`None` here is a recipe with no norm, not a skip).
                let e = ops::layout::embed(&chosen, &m.embed, m.vocab);
                let (e, h) = match &mtp.pre_fc {
                    Some(pre) => (
                        ops::elemwise::rmsnorm_plus_one(&e, &pre.embedding, pre.eps),
                        ops::elemwise::rmsnorm_plus_one(&hidden, &pre.hidden, pre.eps),
                    ),
                    None => (e, hidden.clone()),
                };
                let mut dy = ops::elemwise::residual_add(
                    &ops::linear::matmul(&e, &mtp.fc_embed),
                    &ops::linear::matmul(&h, &mtp.fc_hidden),
                );

                // One attention arm, not a decode/prefill split: the head only
                // ever runs small speculative forwards, where a batched-prefill
                // read is the same numbers as a decode read. A chained step
                // reads the kv the first step appended and appends nothing.
                let a = &mtp.attn;
                let nx = ops::elemwise::rmsnorm_plus_one(&dy, &mtp.mixer_norm, mtp.mixer_norm_eps);
                let o = mtp_attn(&nx, &inputs, m, &plan_mtp, a, step > 0);
                let o = if m.tp > 1 {
                    ops::collective::all_reduce(&o)
                } else {
                    o
                };
                dy = ops::elemwise::residual_add(&o, &dy);

                let nx = ops::elemwise::rmsnorm_plus_one(&dy, &mtp.mlp_norm, mtp.mlp_norm_eps);
                let Mlp::Dense {
                    gate_up,
                    down,
                    inter,
                } = &mtp.mlp
                else {
                    panic!("a draft head is one block and routes to no experts");
                };
                let f = ops::linear::matmul(
                    &ops::linear::mlp_swiglu(&ops::linear::matmul(&nx, gate_up), *inter),
                    down,
                );
                let f = if m.tp > 1 {
                    ops::collective::all_reduce(&f)
                } else {
                    f
                };
                dy = ops::elemwise::residual_add(&f, &dy);

                // Readout through the base head (no dedicated mtp.lm_head), past
                // this recipe's own final norm when it has one; EAGLE has none.
                let read = match &mtp.norm {
                    Some(norm) => ops::elemwise::rmsnorm_plus_one(&dy, norm, mtp.norm_eps),
                    None => dy.clone(),
                };
                let draft = ops::linear::lm_head(&read, head);
                if step == 0 {
                    seam::at(seam::MTP, &[&draft]);
                }
                chosen = ops::layout::argmax(&[&draft]);
                hidden = dy;
                chain.push(draft);
            }
            // The token plane the device-resident loop reads
            // (`intrinsics::mtp_drafts`): every step's argmax side by side,
            // `[rows, DRAFT_DEPTH]` — what `mtp_depth` advertises.
            let steps: Vec<&Value> = chain.iter().collect();
            seam::at(seam::MTP_DRAFTS, &[&ops::layout::argmax(&steps)]);
        }

        logits
    }
}

/// Picks the rotation by whether the model has a tower, not by a per-lane
/// fact bit: a text-only model keeps the scalar `rope_partial`; a tower model
/// always uses mrope, since an image-free row's position is just `(p, p, p)`.
/// **THE BLOCK DRAFTER'S TWO ARMS** — the context it caches, and the block
/// it proposes.
///
/// The reference (`z-lab/dflash`'s `model_mlx.py`) runs
/// `h = layer(h, h_ctx, rope, cache)` over five layers, where `h_ctx` is
/// **passed to every layer and updated by none**, and all a layer does with
/// it is `cache.update_and_fetch(k_proj(h_ctx), v_proj(h_ctx))`. A fixed row
/// set that contributes keys and values at every layer, cached, IS a kv row
/// — so there is no second stream to carry here. The context is written into
/// the drafter's kv rows over the TRUNK's own rows, and the draft pass reads
/// it back out of the cache in a later fire.
///
/// `fused` is the trunk's tapped hidden states already summed through their
/// slices of the fusion bank — taken at the tap sites, because the residual
/// stream is one aliased buffer. Returns the block rows' final hidden for
/// the caller to merge before the shared readout.

fn dflash_arm(
    d: &DFlash,
    inputs: &Input<Facts>,
    fused: &Value,
    h_block: &Value,
    mask: &Value,
) -> Value {
    // ── THE CONTEXT, WHICH IS THE CACHE ──────────────────────────────────
    // **ON EVERY TRUNK FIRE, NOT ONLY A DRAFTING ONE.** The taps carry the
    // trunk's own arm and that is the whole guard this wants: a fire whose
    // rows the trunk ran must leave the drafter's context behind, or the
    // drafter attends over a sequence with holes in it the next time it
    // drafts. (The chained heads guard their work on `Facts::drafts`, but
    // that fact is INFERRED from a program reading the draft seam, and a
    // block drafter plants that seam on its block rows alone — so a verify
    // fire could never set it, and the context would never be written.)
    // It costs a five-way fusion and ten projections a fire, under a percent
    // of a decode.
    //
    let h_ctx = ops::elemwise::rmsnorm_plus_one(fused, &d.hidden_norm, d.hidden_norm_eps);
    // Spelled the way the keys beside them are: the taps these positions go
    // with are already inside the trunk's arm.
    let (_, ctx_positions) = inputs.positions().split(&Facts::block_draft());
    for b in &d.blocks {
        let a = &b.attn;
        let hd = a.head_dim;
        let k = ops::linear::matmul(&h_ctx, &a.k_proj);
        let v = ops::linear::matmul(&h_ctx, &a.v_proj);
        let k = ops::elemwise::rmsnorm_per_head_plus_one(&k, &a.k_norm, hd, a.k_norm_eps);
        // One tensor, not a q/k pair: there is no query here, only the
        // context's keys on their way into the row.
        let k = ops::elemwise::rope_partial_q(&k, &ctx_positions, a.rotary_dim, hd, a.theta);
        ops::attn::kv_append(
            &k,
            &v,
            inputs.kv(&a.kv),
            &inputs.write_page(&a.kv),
            &inputs.write_offset(&a.kv),
        );
    }

    // ── THE BLOCK ────────────────────────────────────────────────────────
    let (input_block, _) = inputs.split(&Facts::block_draft());
    let (block_positions, _) = inputs.positions().split(&Facts::block_draft());
    let mut h = h_block.clone();
    for b in &d.blocks {
        let a = &b.attn;
        let hd = a.head_dim;
        let plan = ops::attn::plan_prefill(&input_block, a.q_heads, a.kv_heads, hd, None);
        let x = ops::elemwise::rmsnorm_plus_one(&h, &b.mixer_norm, b.mixer_norm_eps);
        let q = ops::linear::matmul(&x, &a.q_proj);
        let k = ops::linear::matmul(&x, &a.k_proj);
        let v = ops::linear::matmul(&x, &a.v_proj);
        let q = ops::elemwise::rmsnorm_per_head_plus_one(&q, &a.q_norm, hd, a.q_norm_eps);
        let k = ops::elemwise::rmsnorm_per_head_plus_one(&k, &a.k_norm, hd, a.k_norm_eps);
        let (q, k) =
            ops::elemwise::rope_partial(&q, &k, &block_positions, a.rotary_dim, hd, a.theta);
        // The block's own kv joins the row the context is already in, and
        // the guest rolls it back with `kv_len` next round — the transient
        // half of `keys = concat(cache.fetch(ctx), prop)`.
        ops::attn::kv_append(
            &k,
            &v,
            inputs.kv(&a.kv),
            &inputs.write_page(&a.kv),
            &inputs.write_offset(&a.kv),
        );
        // A sliding layer is causal within the block and windowed over the
        // context, which is what a windowed prefill IS. The last layer is
        // full attention and BIDIRECTIONAL over the block — only a stated
        // mask says that, and it is the guest's (`inputs.mask()`).
        let o = match b.window {
            Some(w) => ops::attn::prefill(
                &q,
                &plan,
                inputs.kv(&a.kv),
                Some(w),
                hd,
                a.kv_heads,
                a.sm_scale,
            ),
            // **NOT CAUSAL.** The drafter's last layer is full attention:
            // the reference skips `create_causal_mask` outright when
            // `is_causal` is false, so a mask row sees the whole block, its
            // own future included. Stated here because a mask alone cannot
            // say it — pie's masked read is causal AND mask unless the op
            // says otherwise.
            None => ops::attn::masked(
                &q,
                &plan,
                mask,
                inputs.kv(&a.kv),
                None,
                hd,
                a.kv_heads,
                false,
                a.sm_scale,
            ),
        };
        h = ops::elemwise::residual_add(&ops::linear::matmul(&o, &a.o_proj), &h);

        let x = ops::elemwise::rmsnorm_plus_one(&h, &b.mlp_norm, b.mlp_norm_eps);
        let Mlp::Dense {
            gate_up,
            down,
            inter,
        } = &b.mlp
        else {
            panic!("a draft block routes to no experts");
        };
        let f = ops::linear::matmul(
            &ops::linear::mlp_swiglu(&ops::linear::matmul(&x, gate_up), *inter),
            down,
        );
        h = ops::elemwise::residual_add(&f, &h);
    }
    ops::elemwise::rmsnorm_plus_one(&h, &d.norm, d.norm_eps)
}

fn rotate(
    q: &Value,
    k: &Value,
    inputs: &Input<Facts>,
    m: &Model,
    a: &Attn,
    d: u32,
) -> (Value, Value) {
    match &m.tower {
        None => ops::elemwise::rope_partial(q, k, &inputs.positions(), a.rotary_dim, d, a.theta),
        Some(_) => ops::elemwise::rope_mrope(
            q,
            k,
            &inputs.mrope_positions(),
            MROPE_SECTIONS,
            MropeForm::Interleaved,
            a.rotary_dim,
            d,
            a.theta,
        ),
    }
}

/// The tower, as one function and one capture unit on the patch axis. Every
/// rectangle here is unguarded (`Dim::Patches` is empty on an image-free
/// fire, so it costs nothing) and must be emitted before any trunk node, or
/// `model_compiler` refuses the plan (`Error::UnitsInterleave`).
///
/// Returns the merged `[Dim::Patches, trunk hidden]` rectangle whose leading
/// `rows / merge^2` rows are live; the caller scatters it with
/// [`ops::layout::scatter_live_rows`] and a `-1`-sentinel route vector.
fn tower(inputs: &Input<Facts>, t: &Tower) -> Value {
    let d = t.head_dim;
    // Pre-unfolded patch vectors, the per-image indptr the bidirectional
    // attention is block-diagonal over, and each patch's (t, h, w).
    let x = inputs.patches(t.patch_width);
    let segments = inputs.patch_segments();
    let grid = inputs.patch_positions();

    // The patch embed is a matmul over pre-unfolded patch vectors; the
    // position table is read with a weighted gather over four bilinear taps.
    let mut y = ops::elemwise::add_bias(
        &t.patch_embed_bias,
        &ops::linear::matmul(&x, &t.patch_embed),
    );
    let ids = inputs.patch_embed_rows(t.taps);
    let pos = if t.taps == 1 {
        ops::layout::embed(&ids, &t.pos_embed, t.positions)
    } else {
        let weights = inputs.patch_embed_weights(t.taps);
        ops::layout::embed_weighted(&ids, &weights, &t.pos_embed, t.positions)
    };
    y = ops::elemwise::residual_add(&pos, &y);

    // `nn.LayerNorm` is one runtime op: its scale/bias fold into the
    // following GEMM is only half expressible and the halves don't compose,
    // so the import contract keeps it a plain copy.
    for b in &t.blocks {
        let n = ops::elemwise::layernorm(&y, &b.norm1, &b.norm1_bias, t.norm_eps);
        let (q, k, v) = ops::layout::split_qkv(
            &ops::elemwise::add_bias(&b.qkv_bias, &ops::linear::matmul(&n, &b.qkv)),
            t.hidden,
            t.hidden,
        );
        // Two axes and no time: a zero section rather than a two-wide stream.
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
            &ops::elemwise::add_bias(&b.proj_bias, &ops::linear::matmul(&o, &b.proj)),
            &y,
        );

        let n = ops::elemwise::layernorm(&y, &b.norm2, &b.norm2_bias, t.norm_eps);
        let h = ops::elemwise::add_bias(&b.fc1_bias, &ops::linear::matmul(&n, &b.fc1));
        let a = ops::linear::mlp_gelu_tanh(&h);
        y = ops::elemwise::residual_add(
            &ops::elemwise::add_bias(&b.fc2_bias, &ops::linear::matmul(&a, &b.fc2)),
            &y,
        );
    }

    // The merger: norm runs on unmerged rows (`merger.norm` is `[hidden]`,
    // not `[merge^2*hidden]`), and the fold comes after it.
    let m = &t.merger;
    let n = ops::elemwise::layernorm(&y, &m.norm, &m.norm_bias, t.norm_eps);
    let folded = ops::layout::merge_rows(&n, t.merge);
    let h = ops::elemwise::add_bias(&m.fc1_bias, &ops::linear::matmul(&folded, &m.fc1));
    let a = ops::linear::mlp_gelu_tanh(&h);
    ops::elemwise::add_bias(&m.fc2_bias, &ops::linear::matmul(&a, &m.fc2))
}

fn attn_mixer(
    x: &Value,
    inputs: &Input<Facts>,
    m: &Model,
    plan_m: &Value,
    plan_d: &Value,
    plan_p: &Value,
    plan_s: &Value,
    mask: &Value,
    a: &Attn,
) -> Value {
    let pages = inputs.kv(&a.kv);
    let write_page = inputs.write_page(&a.kv);
    let write_offset = inputs.write_offset(&a.kv);
    let d = m.head_dim;
    let (q, gate) = ops::layout::split_q_gate(&ops::linear::matmul(x, &a.qg_proj), d);
    let k = ops::linear::matmul(x, &a.k_proj);
    let v = ops::linear::matmul(x, &a.v_proj);
    seam::at(seam::ATTN_QV, &[&q, &v]);
    let q = ops::elemwise::rmsnorm_per_head_plus_one(&q, &a.q_norm, d, a.q_norm_eps);
    let k = ops::elemwise::rmsnorm_per_head_plus_one(&k, &a.k_norm, d, a.k_norm_eps);
    let (q, k) = rotate(&q, &k, inputs, m, a, d);
    ops::attn::kv_append(&k, &v, pages, &write_page, &write_offset);
    seam::at(seam::ATTN_Q, &[&q]);

    // Four arms of one merge: masked, score-capturing, decode, prefill. The
    // split order here must match `forward`'s class order.
    let [mq, sq, dq, p] = q.split([
        Facts::masked(),
        Facts::captures_scores(),
        Facts::qo_one(),
        Predicate::rest(),
    ]);
    let (so, lse) = ops::attn::prefill_lse(&sq, plan_s, pages, None, d, m.kv_heads, a.sm_scale);
    seam::at(seam::SCORES, &[&lse]);
    let o = Value::merge(vec![
        ops::attn::masked(&mq, plan_m, mask, pages, None, d, m.kv_heads, true, a.sm_scale),
        so,
        ops::attn::decode(&dq, plan_d, pages, None, d, a.sm_scale),
        ops::attn::prefill(&p, plan_p, pages, None, d, m.kv_heads, a.sm_scale),
    ]);
    seam::at(seam::ATTN_OUT, &[&o]);
    ops::linear::matmul(&ops::elemwise::gate_sigmoid_mul(&o, &gate), &a.o_proj)
}

/// The draft head's attention: the family's gated full-attention site, over
/// the head's own kv row (in the trunk's page-id space), on one prefill
/// schedule.
/// `chain`: a step past the first, which reads the kv the first step wrote
/// for this row and appends none of its own (as the qwen4 head chains).
fn mtp_attn(x: &Value, inputs: &Input<Facts>, m: &Model, plan: &Value, a: &Attn, chain: bool) -> Value {
    let pages = inputs.kv(&a.kv);
    let write_page = inputs.write_page(&a.kv);
    let write_offset = inputs.write_offset(&a.kv);
    let d = m.head_dim;
    let (q, gate) = ops::layout::split_q_gate(&ops::linear::matmul(x, &a.qg_proj), d);
    let k = ops::linear::matmul(x, &a.k_proj);
    let v = ops::linear::matmul(x, &a.v_proj);
    let q = ops::elemwise::rmsnorm_per_head_plus_one(&q, &a.q_norm, d, a.q_norm_eps);
    let k = ops::elemwise::rmsnorm_per_head_plus_one(&k, &a.k_norm, d, a.k_norm_eps);
    let (q, k) = rotate(&q, &k, inputs, m, a, d);
    if !chain {
        ops::attn::kv_append(&k, &v, pages, &write_page, &write_offset);
    }
    let o = ops::attn::prefill(&q, plan, pages, None, d, m.kv_heads, a.sm_scale);
    ops::linear::matmul(&ops::elemwise::gate_sigmoid_mul(&o, &gate), &a.o_proj)
}

fn gdn_mixer(x: &Value, inputs: &Input<Facts>, g: &Gdn) -> Value {
    let conv_state = inputs.state(&g.conv_state);
    let delta_state = inputs.state(&g.delta_state);
    let qkvz = ops::linear::matmul(x, &g.in_qkvz);
    let ba = ops::linear::matmul(x, &g.in_ba);
    seam::at(seam::RECURRENT, &[&qkvz]);
    let width = Gdn::qkv_width(g.k_heads, g.v_heads, g.k_dim, g.v_dim);
    let one = Facts::qo_one();
    let (qkvz_d, qkvz_p) = qkvz.split(&one);
    let (ba_d, ba_p) = ba.split(&one);
    let (core_d, z_d) = {
        let (qkv, z) = ops::layout::split_rows(&qkvz_d, width);
        let qkv = ops::attn::ssm_causal_conv1d(&qkv, &g.conv, conv_state, g.conv_kernel);
        let gates = ops::attn::ssm_gdn_prep(&ba_d, &g.dt_bias, &g.a_log);
        let core = ops::attn::ssm_gated_delta(
            &qkv,
            &z,
            &gates,
            delta_state,
            g.k_heads,
            g.v_heads,
            g.k_dim,
            g.v_dim,
        );
        (core, z)
    };
    let (core_p, z_p) = {
        let (qkv, z) = ops::layout::split_rows(&qkvz_p, width);
        let qkv = ops::attn::ssm_causal_conv1d_chunked(&qkv, &g.conv, conv_state, g.conv_kernel);
        let gates = ops::attn::ssm_gdn_prep(&ba_p, &g.dt_bias, &g.a_log);
        let core = ops::attn::ssm_gated_delta_chunked(
            &qkv,
            &z,
            &gates,
            delta_state,
            g.k_heads,
            g.v_heads,
            g.k_dim,
            g.v_dim,
        );
        (core, z)
    };
    let o = Value::merge(vec![core_d, core_p]);
    let z = Value::merge(vec![z_d, z_p]);

    let o =
        ops::elemwise::rmsnorm_gated(&o, &z, &g.norm, g.v_dim, g.norm_eps, GateActivation::Silu);
    ops::linear::matmul(&o, &g.out_proj)
}
