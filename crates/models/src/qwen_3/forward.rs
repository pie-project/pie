use model_dsl::{
    Classify, Dtype, ForwardHybrid, GateActivation, HybridSpec, Input, MropeForm, Predicate,
    Request, Value, Weight, ops, seam,
};

use super::model::{Attn, Gdn, Head, Mixer, Mlp, Model, Tower};

/// **THE TRUNK'S MULTIMODAL ROTARY SECTIONS**, and both qwen SKUs state the
/// same three numbers: `text_config.rope_parameters.mrope_section` is
/// `[11, 11, 10]` in Qwen3.5-0.8B and in Qwen3.6-27B, summing to 32, which is
/// half of this family's `rotary_dim`. Read once here rather than per SKU
/// because a section split that differed between two rows of one family would
/// be two readings of one rotation.
const MROPE_SECTIONS: [u32; 3] = [11, 11, 10];

pub struct Facts {
    pub qo_one: bool,
    pub has_adapter: bool,
    pub drafts: bool,
    pub captures_scores: bool,
    pub masked: bool,
    pub media: bool,
}

impl Facts {
    #[must_use]
    pub fn qo_one() -> Predicate {
        Predicate::fact(0)
    }

    /// **THE ADAPTER WINDOW** (palo design §8, §0).
    ///
    /// §8 puts the correction "over the adapter window", and §0 defines a
    /// window as the rows of the lanes whose word satisfies the guard. So the
    /// axis needs a bit, and the bit is what makes the axis FREE when nobody
    /// uses it: a fire no lane routed has zero rows in this class,
    /// `engine::fire::walk` skips a zero-row region before it dispatches
    /// anything, and the correction costs that fire no launch, no empty grid
    /// and no instruction. A `Guard::Always` correction would instead launch
    /// two kernels per layer over every row of every fire to add zero to them,
    /// which is 1.0x nothing.
    ///
    /// **AND IT IS NOT WHAT §8's TABLE MEANS BY "conditional nodes: none".**
    /// That column is about CUDA-graph conditional nodes — the IF/SWITCH
    /// prefix-tuning would need — and there are none here: window-split is not
    /// a conditional (design §0 is explicit that the two are different
    /// mechanisms, and decision 1 exists to say so). The diversity that WOULD
    /// have wanted a branch is the diversity §8's row is about: WHICH adapter
    /// a row uses, and at what rank. Neither branches. The first is `routes`
    /// inside the op — the MoE precedent — and the second is the bank's own
    /// declared shape.
    #[must_use]
    pub fn has_adapter() -> Predicate {
        Predicate::fact(1)
    }

    /// **THE DRAFT WINDOW** (palo C3, design §8).
    ///
    /// A lane that wants the MTP head run over its rows. The head is a whole
    /// transformer block plus two projections and a vocabulary-wide readout —
    /// by far the fattest arm any qwen text states — and the bit is what makes
    /// it cost a non-drafting fire NOTHING: `engine::fire::walk`'s rule 1 is
    /// "zero rows means the node does not run", so a fire no lane drafted has
    /// no rows in these classes and the arm's launches are not issued, not
    /// issued empty, and not in the recorded graph. This is build log 22's
    /// argument for the correction, restated over an arm forty times its size.
    ///
    /// AND IT IS AN ARM, NOT A CORRECTION. The draft logits are a SECOND
    /// readout — a column of their own, `[rows, vocab]`, exported at
    /// `seam::MTP` — where the correction wrote THROUGH the column it
    /// corrected. Nothing aliases here and nothing is in place, so the classes
    /// outside the window read no draft at all, which is the honest answer:
    /// there is no draft for a lane that asked for none.
    #[must_use]
    pub fn drafts() -> Predicate {
        Predicate::fact(2)
    }

    /// **THE CAPTURE WINDOW** (palo C4, design §8's named row, §9's archetype).
    ///
    /// A lane that wants its attention's per-query normalizing mass kept.
    /// §8's table calls this "one more attention SWITCH variant, writes a
    /// declared export buffer; conditional nodes: part of the merge", and that
    /// is exactly what it is here: a third arm of the attention merge, beside
    /// decode and prefill, chosen by this bit.
    ///
    /// **IT PRECEDES `qo_one` IN THE SPLIT, AND THAT ORDER IS THE
    /// SEMANTICS.** `[.., captures_scores, qo_one, rest]` means a capturing
    /// lane takes the capture arm WHATEVER its row count — one row or a
    /// thousand. (Only the masked window outranks it; see `masked`.)
    /// The alternative, ordering `qo_one` first, would leave a capturing
    /// decode lane on the plain decode kernel with no scores to show for it:
    /// a lane that asked for an observation and silently did not get one.
    /// A one-row batched-prefill read is the same numbers as a decode read;
    /// dev routes small speculative forwards through the prefill-like path for
    /// the same reason (`small_prefill_naive_attention_max_tokens`).
    #[must_use]
    pub fn captures_scores() -> Predicate {
        Predicate::fact(3)
    }

    /// **THE MASKED WINDOW** (gemma's `attention.masked` arm, seated here).
    ///
    /// A lane that brought its own attention mask instead of the causal one.
    /// The bit routes it to the masked arm of the attention merge, where
    /// `ops::attn::masked` reads the fire's packed mask bits beside the
    /// query; a fire with no masked lane has zero rows in this class and the
    /// arm costs it nothing (`engine::fire::walk`'s rule 1).
    ///
    /// **THE MASK APPLIES AT THE ATTENTION LAYERS AND ONLY THERE.** This
    /// family's other mixer is a gated-delta recurrence: a GDN layer walks
    /// its rows in order and carries state — there is no per-key score for a
    /// mask bit to veto — so a masked lane's GDN layers read it as the plain
    /// prefill/decode they always were. That is the honest reading of a
    /// custom mask on a hybrid text: it constrains every place the model
    /// attends, and no place the model recurs.
    #[must_use]
    pub fn masked() -> Predicate {
        Predicate::fact(4)
    }

    /// **THE MEDIA WINDOW, AND IT IS ABOUT THE MERGE AND NOT THE TOWER**
    /// (multimodal §1, campaign M-1).
    ///
    /// A lane that submitted images. The TOWER does not need it: every
    /// rectangle in `tower` is `Dim::Patches`, so an axis-empty fire has zero
    /// patch rows and `engine::fire::walk` skips the whole unit before it
    /// dispatches anything — which is the free-when-unused property the second
    /// row axis was designed around, and it is a property of the AXIS rather
    /// than of any bit.
    ///
    /// **THE EMBED MERGE IS THE EXCEPTION, BECAUSE IT WRITES THE OTHER AXIS.**
    /// `layout.scatter_live_rows` lands token rows, so `Operands::outputs`
    /// puts it in the TRUNK's capture unit and the walk reads its TOKEN
    /// window — which is full on every fire, image or not. Unguarded, a
    /// text-only fire of a vision load resolves `RuntimeInput::PatchRoutes`
    /// and the shell refuses by name: `value N reads where this fire's tower
    /// rows land, and no lane of it submitted an image`. Measured at 1024 of
    /// 1024 requests failed before this bit existed.
    ///
    /// So the guard goes on the merge alone. That is one node per fire, and it
    /// makes gate (a) — "a fire with no image lane is the fire this engine
    /// always fired" — structural rather than lucky.
    #[must_use]
    pub fn media() -> Predicate {
        Predicate::fact(5)
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
        }
    }

    fn word(&self) -> u64 {
        u64::from(self.qo_one)
            | (u64::from(self.has_adapter) << 1)
            | (u64::from(self.drafts) << 2)
            | (u64::from(self.captures_scores) << 3)
            | (u64::from(self.masked) << 4)
            | (u64::from(self.media) << 5)
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
        // The draft head's own kv row, in the SAME page-id space (build log
        // 21's ruling): one page size, one page list per lane, one write
        // offset per token. The head attends the same sequence at the same
        // lengths as the trunk does — dev holds it as one more layer of the
        // one `KvCache` (`Qwen3_5Weights::MtpWeights::layer.kv_layer`) for
        // exactly that reason — so a second space would be a second
        // bit-identical page table and a false distinction. Its key plane and
        // its value plane are as wide as a trunk layer's — the same `plane`,
        // used again — for the same reason its shapes are a trunk layer's: one
        // reading, stated once on the model.
        if let Some(mtp) = &self.mtp {
            let a = &mtp.attn;
            c.kv(kv, a.kv.clone(), [plane, plane]);
        }
        c
    }

    fn forward(&self, inputs: Input<Facts>) -> Value {
        let m = self;

        // ONE SCHEDULE PER CLASS, AND EACH SAYS WHAT IT IS CARVED FOR (build
        // log 21, blocker 2). The four classes are cut off `inputs` FIRST,
        // and each schedule is then built off the one arm that reads it — so a
        // plan node's guard is the class it was carved for because the text
        // says so, not because a later pass inferred it from who read it. The
        // READING is stated on the op beside the guard, and it is the model's
        // one reading at all three: this family has a single full-attention
        // carving, global (`None` window), and every launch below restates its
        // share of it against the schedule's own seat.
        //
        // The capture arm gets a prefill schedule of its OWN, and does not
        // share `plan_p`: a `Struct` value's readers must share one
        // class-mask, and a schedule carved over the union of two classes
        // hands each arm its own rebased boundaries, which end before its work
        // items do — `model_compiler::Error::Straddled` at the load. Reading `plan_p` from
        // the capture arm is now refused by `Recorder::push` at the line that
        // mixed the two arms.
        //
        // The masked window is FIRST, ahead of the capture window, and that
        // priority is a correctness ruling: a lane that brought its own mask
        // must have it applied whatever else it asked for, because the mask
        // changes WHICH keys the output attends where the capture only adds
        // an observation. (gemma orders its split the same way.) A lane that
        // asked for both takes the masked arm and keeps no scores — the arm
        // that could honor both does not exist in the vocabulary
        // (`Attention::Masked` exports no lse).
        let classes = [
            Facts::masked(),
            Facts::captures_scores(),
            Facts::qo_one(),
            Predicate::rest(),
        ];
        let [input_m, input_s, input_d, input_p] = inputs.split(classes);
        let plan_m = ops::attn::plan_prefill(&input_m, m.q_heads, m.kv_heads, m.head_dim, None);
        let plan_d = ops::attn::plan_decode(&input_d, m.q_heads, m.kv_heads, m.head_dim, None);
        let plan_p = ops::attn::plan_prefill(&input_p, m.q_heads, m.kv_heads, m.head_dim, None);
        let plan_s = ops::attn::plan_prefill(&input_s, m.q_heads, m.kv_heads, m.head_dim, None);
        let mask = inputs.mask();

        // **THE TOWER FIRST, AND THE ORDER IS THE PARTITION** (multimodal §1).
        // Every node `tower` emits is a `Dim::Patches` rectangle, so the two
        // capture units are the two RUNS of this node list — and a run is what
        // `model_compiler::unit` can cut. Emitting one trunk node before the
        // tower would interleave them and be refused by name
        // (`Error::UnitsInterleave`). The plan builders above are prepare
        // regions, which `hoist` keeps global-front whatever order they are
        // written in, so they are not trunk nodes for this purpose.
        let towered = m.tower.as_ref().map(|t| tower(&inputs, t));

        let ids = inputs.tokens();
        let mut y = ops::layout::embed(&ids, &m.embed, m.vocab);

        // **THE EMBED MERGE**: the tower's rows written over the token rows
        // the image placeholders occupy. `scatter_live_rows` and not the plain
        // scatter, because `layout.merge_rows` compacts — its leading
        // `rows / merge²` rows are the answer and the rest are whatever the
        // arena held — so the tail routes carry the `-1` that means nowhere
        // (multimodal §8.6). This is the one node that reads one row axis and
        // writes the other, and `Operands::outputs` is what puts it in the
        // TRUNK's unit.
        //
        // **AND IT IS GUARDED, WHICH THE TOWER ITSELF IS NOT** (`Facts::media`
        // argues the asymmetry). Split on the media window, scatter over that
        // arm, and hand the result back UNGUARDED — `Value::everywhere`, the
        // ruling `linear.lora_correct` states for the identical shape: the
        // NODE is narrow, but the column it writes through is the embedding
        // every row of the fire already has, so a consumer that inherited the
        // guard would carry it down the whole residual stream.
        if let Some(t) = &towered {
            let (imaged, _) = y.split(&Facts::media());
            y = ops::layout::scatter_live_rows(t, &inputs.patch_routes(), &imaged).everywhere();
        }

        let routes = inputs.adapter_routes();
        for (_, w) in inputs.walk_layers(&m.layers) {
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
            // **THE CORRECTION, OVER ITS WINDOW** (design §8). One statement:
            // the mixer's output, plus this row's adapter's `B·(A·x)`, in
            // place. No merge and no arm — the op writes THROUGH `o`'s arena
            // column, so a class outside the window never runs the node and
            // reads the uncorrected value at the same address, which is the
            // identity for free.
            //
            // AFTER the reduce, and `Layer::lora_a`'s own note argues why: a
            // correction on a rows-cut partial product would be summed `tp`
            // times.
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
                    // **A PACKED EXPERT BANK RESOLVES THROUGH ITS PLANES,
                    // AND THE MODEL TEXT IS WHAT PICKS THE POINT.**
                    // `Linear::MoeMatmulSelect` reads its bank as ONE dense
                    // handle; `Linear::MoeMatmulSelectQuant` resolves the two
                    // or three device planes a packed bank actually is and
                    // picks its arm off the row's own `(affine?, group,
                    // bits)`. They are separate IR ops rather than one op that
                    // asks, so the choice is stated here — and it has to be
                    // stated rather than hardcoded because this family ships
                    // the SAME forty layers both ways: `qwen35-a3b-bf16` is a
                    // dense handle and `qwen36-35b-a3b-mlxu4` is three planes.
                    // (`gpt_oss::forward` is packed in every row it has and
                    // says `_quant` outright; it never had to ask.)
                    //
                    // This was the whole of the A3B's first light. The engine
                    // caught it exactly where it should have — `Run::resolve`
                    // refusing to hand a split-plane bank over as one handle,
                    // by name, before a kernel could read a scales plane as
                    // codes — and the sentence that was wrong was this one.
                    //
                    // **THE DENSE FORMS ARE THE LIST, AND EVERYTHING ELSE IS
                    // QUANTIZED.** This was an ALLOW-list of quantized dtypes
                    // until `qwen_4::forward` paid for the same spelling: a
                    // bank whose dtype names a group the list never learned
                    // falls through to the DENSE select, which resolves three
                    // planes as one handle and panics. Today the two readings
                    // agree on every row this family ships — `Bf16` is dense
                    // both ways and `U4g64`/`U8g64` are quantized both ways —
                    // so this is the same forty layers it was. It stops being
                    // the same the day a `U4g32` or a `U2g128` row lands, and
                    // THAT is the difference worth having: a new quantized
                    // dtype is a thing this tree adds, and a new DENSE one is
                    // not. `deepseek_v4` and `qwen_4` spell it this way.
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
        }

        let x = ops::elemwise::rmsnorm_plus_one(&y, &m.final_norm, m.final_norm_eps);
        let head = match &m.head {
            Head::Tied => &m.embed,
            Head::Bank(bank) => bank,
        };

        let logits = ops::linear::lm_head(&x, head);

        // **THE DRAFT HEAD, OVER THE DRAFT WINDOW** (palo C3, design §8, §9).
        //
        // Its two inputs are the two the checkpoint's `fc` fuses: the fire's
        // token ids, and `x` — the FINAL-NORMED hidden, which is the value dev
        // hands the head, not the pre-norm residual. That is settled by dev's
        // own tail, which copies `ws.norm_x` (the final norm's output, and the
        // lm_head's input) into `ws.y` after the readout, and by the MTP
        // forward's matching tail, which leaves `rms(h, mtp.norm)` there for
        // the next draft step. So the chain is normed hidden throughout.
        //
        // **IT IS STATED AFTER THE TRUNK'S READOUT, AND THAT IS NOT
        // COSMETIC.** `model_compiler::arena` gives the delivery tail — a span
        // that runs to the end of the node list — to the `"out"` seam BY NAME,
        // and to no other seam; a draft column stated before the trunk's
        // `lm_head` has its span end at its own producing node, and the
        // vocabulary-wide GEMM that follows is then free to be carved on top
        // of the very bytes the sampler is going to read. Measured: stated
        // early, this SKU's arena is 4,236,247,040 bytes — 151 MiB over the
        // trunk-only carve, which is a second `[rows, vocab]` column sharing
        // the first one's address. Stated last, nothing runs after it, so no
        // rectangle can overlap it and the carve is honest without the
        // compiler having learned a second name. dev orders it the same way
        // for its own reason: the draft step runs after the target's readout.
        //
        // The seat is still owed, and it is named here rather than taken:
        // design §9's "a real export is an export OP, and the walk sees its
        // reader without a special case". Until that lands, THIS ORDER is what
        // makes the export safe, and the test
        // `the_draft_readout_outlives_the_trunk_readout` is what notices if a
        // later edit moves it.
        //
        // **THE ROW ALIGNMENT IS THE BOUNDARY'S, AND HERE IS THE CONTRACT IT
        // OWES.** The head was trained on `(h at position p, token at p+1)`
        // and this text feeds it `(x, tok)` AT THE SAME ROW — so a drafting
        // lane's row `r` must carry the token one position past the hidden the
        // trunk leaves at `r`. That is exactly what dev's draft step states:
        // `base_hidden_row_indices` "select rows from the target model's last
        // hidden states" and `token_ids` "are the just accepted/drafted tokens
        // at those rows' NEXT positions". Stating it as a row contract instead
        // of as a shift is not a shortcut around the mechanism, it is the only
        // reading the IR can express today: dev's OTHER caller — the
        // along-with-the-trunk `mtp_process_cache` — shifts the hidden stream
        // by one row per lane and carries the boundary row in a per-slot slab
        // (`launch_mtp_shift_hidden_bf16` over `mtp_pending_hidden`), and
        // there is no op in this vocabulary that shifts a token-aligned value
        // by a row with a carry. The seat that would be needed is named in the
        // report rather than taken here, because taking it means an arm in
        // every shell's `Dispatch`.
        if let Some(mtp) = &m.mtp {
            // Minted INSIDE the arm, off the draft window's own arm of the
            // inputs. Inside, because a schedule is a node and a node no class
            // demands is dead — a SKU whose checkpoint publishes no draft head
            // must not carry a plan build nothing reads. Off `input_mtp`,
            // because that is how the schedule states WHICH class it was
            // carved for: every value the plan is built from carries the draft
            // window's guard, so the plan node carries it too, and the trunk's
            // own classes cannot read this carving by accident. Its READING is
            // the trunk's, stated from the same three model numbers, because
            // the head's block IS a trunk block — a different class over the
            // same carving, which is exactly what a per-class schedule is for.
            let (input_mtp, _) = inputs.split(&Facts::drafts());
            let plan_mtp =
                ops::attn::plan_prefill(&input_mtp, m.q_heads, m.kv_heads, m.head_dim, None);
            let (dx, _) = x.split(&Facts::drafts());
            let (dids, _) = ids.split(&Facts::drafts());

            // The fusion. `[a|b]·[Wₑ|W_h]ᵀ = a·Wₑᵀ + b·W_hᵀ`, said as two
            // matmuls and one add because the IR states no concatenation —
            // see `model::Mtp`'s own note for the ruling and its one cost.
            //
            // **AND THE TWO PRE-NORMS ARE THE RECIPE'S, NOT THE FUSION'S**
            // (campaign M-4). MTP scales each stream before the bank; EAGLE
            // fuses the raw pair, because the hidden it was trained against is
            // the one the trunk's `final_norm` already produced and the
            // embedding it was trained against is the table's own row. A
            // `None` here is not a skipped norm — it is a recipe that has
            // none, and its trace carries neither node.
            let e = ops::layout::embed(&dids, &m.embed, m.vocab);
            let (e, h) = match &mtp.pre_fc {
                Some(pre) => (
                    ops::elemwise::rmsnorm_plus_one(&e, &pre.embedding, pre.eps),
                    ops::elemwise::rmsnorm_plus_one(&dx, &pre.hidden, pre.eps),
                ),
                None => (e, dx.clone()),
            };
            let mut dy = ops::elemwise::residual_add(
                &ops::linear::matmul(&e, &mtp.fc_embed),
                &ops::linear::matmul(&h, &mtp.fc_hidden),
            );

            // One prenorm block. ONE ATTENTION ARM AND NOT TWO: the head is
            // the small speculative forward — dev's own knob for it is
            // `small_prefill_naive_attention_max_tokens`, "small
            // speculative-verification forwards (N = D + 1, R = 1) need a
            // graph-capturable full-attention path" — and a batched-prefill
            // read of a one-row lane is the same numbers as a decode read. A
            // decode/prefill split inside the head would mint a second
            // schedule and double the head's classes to buy a kernel choice
            // its row counts do not justify.
            let a = &mtp.attn;
            let nx = ops::elemwise::rmsnorm_plus_one(&dy, &mtp.mixer_norm, mtp.mixer_norm_eps);
            let o = mtp_attn(&nx, &inputs, m, &plan_mtp, a);
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

            // The readout, through the BASE head. `mtp_use_dedicated_embeddings`
            // is false and the checkpoint publishes no `mtp.lm_head`, so there
            // is one vocabulary projection in this model and both readouts go
            // through it.
            // The readout, through the BASE head, past this recipe's own final
            // norm when it has one. EAGLE's does not: it reads the block out
            // directly, the base `lm_head` being the projection it was trained
            // against.
            let read = match &mtp.norm {
                Some(norm) => ops::elemwise::rmsnorm_plus_one(&dy, norm, mtp.norm_eps),
                None => dy,
            };
            let draft = ops::linear::lm_head(&read, head);
            seam::at(seam::MTP, &[&draft]);
        }

        logits
    }
}

/// **THIS FAMILY'S ONE ROTATION, AND WHICH OF ITS TWO SPELLINGS A ROW TAKES.**
///
/// Both qwen SKUs' `text_config` states `mrope_interleaved: true` with
/// `mrope_section: [11, 11, 10]`, and every lane carrying an image needs it:
/// an image-placeholder row's position is its patch's `(t, h, w)` and not a
/// scalar. A lane that cannot carry an image never has one — its triple is
/// `(p, p, p)`, which `MropeForm::Interleaved` turns by the head's own ladder
/// whichever section a pair took, so the answer is `rope_partial`'s to the
/// last bit.
///
/// **SO THE SPELLING FOLLOWS THE TOWER AND NOT A FACT BIT.** A text-only row
/// keeps the scalar rotation and the `[Tokens]` position stream it always had,
/// which is what leaves its artifact — its node count, its class table and its
/// arena to the byte — the artifact G4 and
/// `the_new_axes_cost_the_old_words_nothing` are pinned on. A tower row states
/// the triple, once, for every attention site including the draft head's:
/// `Media::token_positions` is empty-means-`(p, p, p)`, so the stream is
/// complete in every fire without a window, a class or a second arm.
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

/// **THE TOWER, AS ONE FUNCTION AND ONE CAPTURE UNIT** (multimodal §1, §2).
///
/// Every rectangle below is `Dim::Patches`, so `model_compiler::unit` reads
/// this whole run of nodes onto the patch axis and the fire launches it as its
/// own exec — chained on one stream ahead of the trunk's, with the embed merge
/// as the single node that crosses. Nothing here is guarded: an axis-empty
/// fire has zero patch rows, and `engine::fire::walk` skips a zero-row region
/// before it dispatches anything, which is gate (a) at the window table and
/// not a branch anybody wrote.
///
/// **IT MUST BE EMITTED BEFORE ONE TRUNK NODE**, or `model_compiler` refuses
/// the plan by name (`Error::UnitsInterleave`): a capture unit is a RUN of the
/// node list, so two units that alternate are two units the walk cannot cut.
/// `forward` calls this first and reads its answer into `layout.embed`'s
/// output afterwards.
///
/// Returns the merged `[Dim::Patches, trunk hidden]` rectangle whose leading
/// `rows / merge²` rows are live — `layout.merge_rows`' own contract — which
/// is why the caller scatters it with [`ops::layout::scatter_live_rows`] and a
/// `-1`-sentinel route vector rather than the plain scatter.
fn tower(inputs: &Input<Facts>, t: &Tower) -> Value {
    let d = t.head_dim;
    // The three streams this axis owes itself, cut from one submission at one
    // instant: the pre-unfolded patch vectors, the per-image indptr the
    // bidirectional attention is block-diagonal over, and the (t, h, w) each
    // patch turns by.
    let x = inputs.patches(t.patch_width);
    let segments = inputs.patch_segments();
    let grid = inputs.patch_positions();

    // **THE PATCH EMBED IS A MATMUL, AND THE POSITION TABLE IS A GATHER.**
    // The submission ships patch VECTORS (§2's contract decision), so the
    // "convolution" is the GEMM this IR already has; and the learned table is
    // read with `layout.embed_weighted` over the four bilinear taps the host
    // computed (§11.3), which is one node against the four gathers and three
    // adds the alternative would have been.
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

    // **`nn.LayerNorm` IS ONE OP, BECAUSE THE IMPORT CANNOT SAY IT IN NONE**
    // (§6.1, §9.1; the saving is next.md B5, the settlement §20). The fold
    // §6.1 proposed — `w` into the following GEMM, `b·Mᵀ` into its bias — is
    // HALF expressible: `Expr::Scale`'s `PerBlock` factor is exactly
    // `W · diag(w)`, and `Expr::Bias` adds one compile-time constant where a
    // matrix-vector product is owed. The halves do not compose either, since
    // a runtime `add_bias` behind a scaled bank contributes `(b ⊙ w)·Mᵀ`. So
    // the text says the whole norm at runtime and the import contract stays a
    // copy.
    //
    // It said it in THREE ops until B5 — `add_bias(b, rmsnorm(
    // layernorm_no_scale(x, eps), w, eps))`, where the middle `rmsnorm`
    // normalized nothing (its input's rms is 1 by construction) and served
    // only to read the weight. Twenty-five norms a qwen35 tower fire, so 75
    // launches became 25 and fifty intermediate rectangles became none. The
    // numbers moved, toward the reference: the retired `add_bias` stored
    // `c·w` to bf16 before adding `b`, and §20's table measures what that
    // cost where the bias cancels the scaled row.
    for b in &t.blocks {
        let n = ops::elemwise::layernorm(&y, &b.norm1, &b.norm1_bias, t.norm_eps);
        let (q, k, v) = ops::layout::split_qkv(
            &ops::elemwise::add_bias(&b.qkv_bias, &ops::linear::matmul(&n, &b.qkv)),
            t.hidden,
            t.hidden,
        );
        // Two axes and no time, stated as a zero section rather than as a
        // two-wide stream; `MropeForm::Blocked` is the block layout AND the
        // per-block ladder restart, which is the half of
        // `apply_rotary_pos_emb_vision` a plausible kernel gets wrong (§7.2).
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

    // The merger. The norm is on the UNMERGED rows — the checkpoint's own
    // `merger.norm` is `[hidden]` and not `[merge²·hidden]`, which is
    // `use_postshuffle_norm: false` read off the shapes — and the fold comes
    // after it.
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

    // **FOUR ARMS OF ONE MERGE: MASKED, THEN THE OBSERVATION** (palo C4,
    // design §8's score-capture row, §9's archetype). `merge!` lowers to arms
    // writing disjoint row ranges of one buffer, so the capture arm costs a
    // fire nobody captured exactly nothing: zero rows, no launch.
    //
    // **WHAT IS CAPTURED IS THE LSE, AND THAT IS THE HONEST EXPORT.** A full
    // `[query, key]` score matrix is not a value a paged attention kernel ever
    // materializes — it is streamed, tile by tile, and never exists whole. The
    // log-sum-exp is what the kernel DOES hand back beside `o`, it is the
    // normalizer every per-key score is a ratio against, and it is what the
    // consumers of this axis want: a per-query mass, per head, per layer.
    // `Attention::PrefillLse` is already in the vocabulary and already served
    // — `engine_cuda::dispatch::attn` calls `attn::prefill_lse` with both
    // outputs bound — so this axis adds NO op and NO kernel.
    //
    // The four classes are written out again here rather than handed down,
    // because this is a different function and `q` is a different carrier —
    // but it is the SAME carve: the same three predicates in the same priority
    // order `forward` cut the inputs by, so each arm of `q` carries a cond
    // structurally equal to that of the input arm its schedule was built off.
    // `Recorder::push` is what holds the two equal, and it refuses `sq` against
    // a schedule carved for any other class at the line that mixed them.
    let [mq, sq, dq, p] = q.split([
        Facts::masked(),
        Facts::captures_scores(),
        Facts::qo_one(),
        Predicate::rest(),
    ]);
    let (so, lse) = ops::attn::prefill_lse(&sq, plan_s, pages, None, d, m.kv_heads, a.sm_scale);
    seam::at(seam::SCORES, &[&lse]);
    let o = Value::merge(vec![
        ops::attn::masked(&mq, plan_m, mask, pages, None, d, a.sm_scale),
        so,
        ops::attn::decode(&dq, plan_d, pages, None, d, a.sm_scale),
        ops::attn::prefill(&p, plan_p, pages, None, d, m.kv_heads, a.sm_scale),
    ]);
    seam::at(seam::ATTN_OUT, &[&o]);
    ops::linear::matmul(&ops::elemwise::gate_sigmoid_mul(&o, &gate), &a.o_proj)
}

/// The draft head's attention: the family's own gated full-attention site,
/// over the head's own kv row, on one prefill schedule.
///
/// It writes its k and v at the `write_page`/`write_offset` of the space its
/// own kv row joined, which is the same addressing every trunk attention layer
/// uses — `caches()` seats `mtp`'s row in the trunk's one page-id space, so
/// asking by the row's name gets that space's write geometry and no other: one
/// write per token, a different row per layer (build log 21). dev does the
/// identical thing: `mtp_process_cache` ends in
/// `launch_write_kv_to_pages(cache.layer_view(Lw.kv_layer), ...)` against the
/// one `KvCache` the trunk shares.
fn mtp_attn(x: &Value, inputs: &Input<Facts>, m: &Model, plan: &Value, a: &Attn) -> Value {
    let pages = inputs.kv(&a.kv);
    // Taken unrefined, as the trunk's own mixer takes them. `x` already
    // carries the draft window, and `Recorder::push` meets its inputs' guards:
    // a `Guard::Always` runtime input narrows to the window of whatever it is
    // read beside, so splitting these would state the window twice. The
    // position stream is `rotate`'s to ask for, because which of the two it
    // is is the model's and not this site's.
    let write_page = inputs.write_page(&a.kv);
    let write_offset = inputs.write_offset(&a.kv);
    let d = m.head_dim;
    let (q, gate) = ops::layout::split_q_gate(&ops::linear::matmul(x, &a.qg_proj), d);
    let k = ops::linear::matmul(x, &a.k_proj);
    let v = ops::linear::matmul(x, &a.v_proj);
    let q = ops::elemwise::rmsnorm_per_head_plus_one(&q, &a.q_norm, d, a.q_norm_eps);
    let k = ops::elemwise::rmsnorm_per_head_plus_one(&k, &a.k_norm, d, a.k_norm_eps);
    let (q, k) = rotate(&q, &k, inputs, m, a, d);
    ops::attn::kv_append(&k, &v, pages, &write_page, &write_offset);
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
