//! qwen3.5 / qwen3.6's METAL text — the hybrid, stated in this backend's
//! symbols.
//!
//! The third text for this family, after the semantic one and CUDA's, and
//! the first that reaches a gated-deltanet layer on a GPU this workspace
//! can run. Its reference is `mlx_lm.models.qwen3_5`, read line for line:
//! `Qwen3.6-27B-4bit` and `Qwen3.6-35B-A3B-4bit` both publish
//! `model_type: qwen3_5_moe`, so "qwen3.6" is this architecture and not a
//! successor to it.
//!
//! # What differs from `llama_like`'s Metal text
//!
//! The MLP is the same mixture — softmax over all experts, top-k, renormalize,
//! plus a sigmoid-gated shared expert — and the projections take the same
//! guarded GEMM/GEMV pair. Two things are this family's own:
//!
//! - The full-attention layers project q at TWICE the head width and cut a
//!   per-head output gate out of it (`q_gate_split`, then `sigmoid_gate`
//!   after the softmax). llama_like has no such bank.
//! - Three layers in four are not attention at all. They run a gated
//!   deltanet: a depthwise causal convolution over a packed `q‖k‖v`, an l2
//!   norm per head, a decay/beta gate, and a recurrence against per-request
//!   state that lives in a slab rather than in a page table.
//!
//! # The two classes are two recurrences, not two kernels
//!
//! A decode fires `gdn_core`, which fuses the convolution, the norm, the
//! gates and the scan into one dispatch because there is one token per
//! request and nothing to share. A prefill splits: `gdn_prep` computes the
//! q/k path once per KEY head, and `gdn_core_recurrent` scans it once per
//! VALUE head. The split exists because qwen3.5 runs 32 value heads over 16
//! key heads, so the fused form recomputes each q and k twice.
//!
//! # The seat table was one entry long, and it hung the GPU
//!
//! **Closed.** On Qwen3.6-35B-A3B-4bit a fire sequence of prefill, anything,
//! prefill used to wedge the GPU on the third fire — "the GPU did not reach
//! event 3 within 60000 ms". Four prefills in a row were clean, and so was
//! one prefill followed by sixty-four decodes, so it looked like the ORDER
//! and not the count. It was neither. It was an out-of-bounds read three
//! kernels upstream, and the whole shape of the symptom is worth keeping,
//! because the next defect of this class will look exactly the same.
//!
//! ## What it was
//!
//! `gdn_prep_slotted` and `gdn_core_recurrent_slotted` both read their state
//! seat as `slot_ids[b_idx]`, where `b_idx = tpig.z / Hv` runs over the
//! FIRE'S ROWS. The wire numbers `rs_slot_ids` per REQUEST — `driver_api`
//! validates it against `qo_indptr`, one entry per resolved qo row — and
//! `serve::launch` passed that vector straight through. So a 128-token
//! prefill of ONE request staged a seat table of ONE `u32` and then read a
//! hundred and twenty-seven entries past the end of it.
//!
//! Past the end of that table is the rest of the fire-tables region and then
//! whatever the allocator put next to it. The garbage those reads returned
//! became `slot`, and `slot` is not checked against anything: it indexes
//! `rstate` and `new_conv_state`, both of them DEVICE WRITES. An unbounded
//! read feeding an unbounded write.
//!
//! ## Why it presented as a hang in a kernel that cannot loop
//!
//! The writes landed on the fire's own tables — the region is small, live and
//! adjacent — and what they wrote was `0x7fc00000`, a float NaN, over the
//! POSITIONS. `sdpa_paged_tiled` then read a position of 2143289344 and ran
//! its staging loop `for (base = kp_lo; base <= kp_hi; base += KT)` over two
//! billion key tiles. Nothing faulted, because every address it touched was
//! resident; nothing answered, because the loop was a hundred million passes
//! deep. Given twenty minutes instead of sixty seconds, it still did not
//! retire.
//!
//! That is why every property of the symptom pointed away from the cause:
//!
//! - **Not deterministic.** Held at sixty-four rows at position 96 the same
//!   binary on the same checkpoint passed three runs in five. It depended on
//!   what happened to be in memory past a one-element table.
//! - **Not the middle fire's text.** Truncating it to ZERO encoded dispatches
//!   still wedged the third. What the middle fire did was irrelevant; that it
//!   moved the allocator was the whole of it.
//! - **Not the kernel that stalled.** Bisecting the measured prefill put the
//!   stall on dispatch 845, `sdpa_paged_tiled_bfloat16_d_256` at layer 31 —
//!   the eighth of ten attention layers, with the seven before it retiring in
//!   the same command buffer. The corruption was dispatch 789,
//!   `gdn_core_recurrent_slotted` at layer 29, and the fifty-six dispatches
//!   between them read tables nobody had touched yet.
//!
//! ## How it was found
//!
//! Two probes, and the second only made sense because of the first.
//!
//! Capping the pass loop's trip count in `sdpa_paged.metal` made the hang go
//! away three runs in three, which said the loop bound was garbage rather
//! than the memory system being stuck. Replacing `position_ids[row]` with the
//! row index did the same, which named the garbage.
//!
//! `PIE_BENCH_VERIFY_TABLES` then read the fire's tables BACK off the device
//! after each fire and compared them to what was staged: intact after the
//! first prefill, intact after the middle fire, `2143289344 != 0` after the
//! third. Bisecting the truncation against that check — rather than against
//! the hang — walked straight to the writer, because a clobber is visible
//! fifty dispatches before it is fatal.
//!
//! ## What holds it closed
//!
//! `serve::launch` expands the wire's per-request seats to one per row
//! through `req_of_token`, and `bind::tables::stage` REFUSES any
//! `recurrent_slots` that is neither empty nor one entry per token. The
//! second is the one that matters: the kernels index it per row, so any other
//! length is a read past the region, and a refusal at staging time is the
//! only place that can be said before the GPU has it.
//!
//! With that, the sequence needs no warm-up: the tier-one bench measures this
//! model with no `PIE_BENCH_PREDECODE` at 242.6 tok/s prefill and 60.3 tok/s
//! decode, the same numbers the workaround used to buy.
//!
//! ## It was all three of this family's failures
//!
//! The wedge was the one this hunt started on, and the other two went with
//! it, which is what a shared cause looks like from the outside:
//!
//! - **The batched routed GEMM.** `moe_tile` was `None` because the batched
//!   arm hung before the first fire retired, at 38 tokens and not at 37. It
//!   is on again, clean, and worth 242.5 -> 348.6 tok/s on a 128-token
//!   prefill.
//! - **The dense 27B.** `Qwen3.6-27B-4bit` faulted with
//!   `MTL4CommandQueueErrorDomain error 1` on its very first prefill -- a
//!   fault rather than a stall, because its garbage seat landed on an
//!   address that was not resident rather than on one that was. It now
//!   prefills 128 tokens at 82.1 tok/s and decodes at 17.6.
//!
//! Three failures, one family, one out-of-bounds read. None of them reached
//! any other family because no other family in this tree has a slotted
//! recurrence to read the table wrong.
//!
//! ## Ruled out along the way
//!
//! Kept because each was measured and each is now known-good ground: the
//! recurrent planes zeroed whole; the paged K and V zeroed whole; the
//! weights' strided fingerprint, identical across the middle fire; the
//! lowering, which is one cached `Lowered` both times; allocation layout,
//! since padding the params region and the arena moved nothing; the working
//! set, which held 18.3 GiB of 24.96 with no leak; the encoder's ordering,
//! where narrowing the hazard extents took the fire from 1084 barriers to
//! 813 and did not move the hang; the command allocator ring, where
//! `ALLOCATOR_COUNT = 8` wedged exactly as before; record-and-replay, which
//! this harness does not enable; and the elastic pager, which it does not
//! use.
//!
use super::facts::{Qwen35FullAttnFacts, Qwen35GdnFacts, Qwen35HybridFacts, Qwen35MlpKind};
use model_dsl::metal::{GdnShape, GdnW};
use model_dsl::{self as dsl, Kv, MatW, NormW, Val, WeightRepr};
use model_ir::trace::{DType, Dim, FireClass, ForwardPlan, GuardPred, NormVariant, Shape};

/// What the METAL deployment of a qwen3.5 hybrid decided, beyond the model.
///
/// Same role [`LlamaLikeMetalFacts`] plays for that family and populated the
/// same way: everything here is a load-time answer about the CHECKPOINT or
/// about this backend's kernels, not a property of the architecture.
///
/// [`LlamaLikeMetalFacts`]: crate::shared::llama_like::forward::facts::LlamaLikeMetalFacts
/// The routed GEMM's tile when a deployment file does not name one.
///
/// The llama-like tile, same as every other mixture in this tree, and the
/// history is worth a paragraph because this fact was `None` for a while and
/// the reason was wrong.
///
/// The batched arm used to hang the GPU on Qwen3.6-35B-A3B-4bit before its
/// first fire retired — "the GPU did not reach event 1 within 60000 ms" — and
/// it looked like a property of the arm: 32, 33, 34, 36 and 44 tokens ran in
/// ~0.5 s, 37 took 29 s, and 38, 39, 40, 48 and 56 never came back. That
/// gradient is the tell in hindsight. A dispatch that takes 29 s and then one
/// that takes forever is not a deadlock; it is a loop whose bound came from
/// memory nobody wrote. It did: the slotted GDN kernels were reading a seat
/// table one entry long and the garbage they read went on to index a device
/// write, which landed on the fire's own position table. The module docs
/// carry the whole account.
///
/// With the seat table one entry per ROW, the arm is clean at 38 and at 128
/// tokens and worth 242.5 -> 348.6 tok/s on a 128-token prefill. The two
/// other failures this family carried went with it: the decode-then-prefill
/// wedge, and the DENSE 27B faulting with `MTL4CommandQueueErrorDomain error
/// 1` on its very first prefill, which now prefills at 82.1 tok/s.
///
/// A free function because `#[serde(default = ...)]` takes a path.
fn default_moe_tile() -> Option<(u32, u32)> {
    Some(crate::shared::llama_like::project::ROUTED_QMM_TILE)
}

#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Qwen35MetalFacts {
    /// How this deployment's dense projections are stored.
    pub proj_repr: WeightRepr,
    /// Bits per packed weight element — 4 or 8.
    pub affine_bits: u32,
    /// How the EXPERT BANKS are stored, when that is not how the dense
    /// projections are.
    pub moe_repr: Option<WeightRepr>,
    /// See [`Self::moe_repr`].
    pub moe_bits: u32,
    /// The ROUTED GEMM's tile — row tile and, necessarily, the sort's
    /// padding block.
    ///
    /// See
    /// [`LlamaLikeMetalFacts::moe_tile`](crate::shared::llama_like::forward::facts::LlamaLikeMetalFacts::moe_tile);
    /// the two families answer the same question and answer it the same
    /// way, and the fact is duplicated rather than shared because these
    /// structs share nothing else.
    ///
    /// Serde-defaulted so no deployment file changes meaning.
    #[serde(default = "default_moe_tile")]
    pub moe_tile: Option<(u32, u32)>,
    /// How the ROUTER GATE is stored, when that is not how the dense
    /// projections are.
    ///
    /// qwen3.6's `mlx-community` builds publish `mlp.gate` and
    /// `mlp.shared_expert_gate` at EIGHT bits inside a four-bit stack —
    /// forty layers of it, listed one by one in `quantization`. Reading a
    /// gate at the wrong width is the failure gpt-oss measured: a fluent
    /// model routing every token to almost the right experts.
    pub router_repr: Option<WeightRepr>,
    /// See [`Self::router_repr`].
    pub router_bits: u32,
    /// The steel GEMM's tile, as `(bm, bn)`.
    pub qmm_tile: (u32, u32),
    /// The batched arm stages its input to `half` and multiplies there.
    ///
    /// See [`LlamaLikeMetalFacts::qmm_fp16_precast`]: the checkpoint and the
    /// output stay bfloat and only the GEMM's tiles and matrix instruction
    /// move, which on a device with no native bfloat matrix unit is the
    /// difference between an emulated sequence and one instruction.
    ///
    /// `gs = 64, b = 4` alone -- `affine_qmm_t_fp16_precast` is stamped at
    /// that point and no other. Both qwen3.6 builds are g64/b4 at the stack
    /// level; the eighty 8-bit tensors `mlx-community` publishes are the
    /// router gates, which never reach a GEMM at all (`[hidden, 256]` is
    /// narrower than any tile, so the gate is a `qmv` by construction).
    ///
    /// [`LlamaLikeMetalFacts::qmm_fp16_precast`]:
    ///     crate::shared::llama_like::forward::LlamaLikeMetalFacts::qmm_fp16_precast
    pub qmm_fp16_precast: bool,
    /// The ROUTED GEMM runs its tiles and its MMA in half.
    ///
    /// Separate from [`Self::qmm_fp16_precast`] and not implied by it. A
    /// mixture's next layer reads this layer's output through a TOP-K, and a
    /// top-k is a comparison two logits can swap under -- which is a
    /// different model downstream and not a tolerance. See
    /// [`LlamaLikeMetalFacts::routed_qmm_fp16`], where llama's moved and
    /// gemma-4's held.
    ///
    /// [`LlamaLikeMetalFacts::routed_qmm_fp16`]:
    ///     crate::shared::llama_like::forward::LlamaLikeMetalFacts::routed_qmm_fp16
    pub routed_qmm_fp16: bool,
    /// The M>1 projections take the GEMM rather than the GEMV.
    pub qmm_multi_batch: bool,
    /// The projection GEMV folds the block residual in its epilogue.
    pub fuse_residual_gemv: bool,
    /// Every norm's epsilon.
    pub rms_eps: f32,
    /// The rope base.
    pub rope_theta: f32,
    /// The attention softmax's scale, `head_dim ** -0.5`.
    pub attn_scale: f32,
    /// Whether the top-k routing weights are renormalized to sum to one.
    ///
    /// True for this family, and it decides the router's DENOMINATOR:
    /// `softmax(all)` then `scores / scores.sum()` over the chosen k is
    /// exactly a softmax over the chosen k, which is what
    /// `router_topk`'s `softmax_over_all = false` computes.
    pub norm_topk_prob: bool,
}

/// The residual stream's width, and the cross-facts check with it.
fn hidden_of(facts: &Qwen35HybridFacts) -> u32 {
    let hidden = facts.hidden();
    assert_eq!(
        facts.gdn.hidden, hidden,
        "hybrid sub-facts disagree on hidden (gdn)"
    );
    if let Qwen35MlpKind::Moe(moe) = &facts.mlp {
        assert_eq!(
            moe.hidden, hidden,
            "hybrid sub-facts disagree on hidden (moe)"
        );
    }
    hidden
}

/// The projections and the two closures that state them, gathered so the
/// three bodies below take one argument instead of eight.
struct Ctx<'a> {
    metal: &'a Qwen35MetalFacts,
    /// The affine point every dense projection names.
    point: String,
    /// The affine point the GEMM names, tile included.
    gemm_point: String,
    /// This fire runs more than one row per request.
    multi_batch: bool,
    /// The FP16 staging pass's memo, keyed on the SSA id of the ACTIVATION.
    ///
    /// One cast per source value and not per projection: q, k and v are three
    /// projections of one normed activation, so a cast apiece would pay three
    /// times for one conversion. Ids are unique for the whole trace, so an
    /// entry cannot go stale and the map is never cleared.
    ///
    /// The cast is a statement of its own and lives OUTSIDE the guard, which
    /// is not a preference: a guard arm records no value -- `region_out`
    /// returns `None` inside one and every launch there binds the guard's
    /// output buffer -- so a cast written in the arm would put half-precision
    /// activations where the projection's result belongs.
    staged: std::cell::RefCell<std::collections::HashMap<model_ir::trace::ValueId, Val>>,
    /// The weight fold every `rms_single_row` in this text applies, read from
    /// the facts rather than assumed. See [`Ctx::norm`].
    norm_variant: NormVariant,
}

impl Ctx<'_> {
    /// One projection, guarded between the GEMM and the GEMV.
    ///
    /// The guard is the FIRE's question and not the deployment's:
    /// `qmm_t.metal` has no `M` argument, so a prefill whose row count the
    /// tile does not divide cannot take the GEMM at all. See
    /// `llama_like`'s Metal text, where the same guard is argued at length.
    /// This activation in `half`, cast once however many projections ask.
    fn stage(&self, x: &Val) -> Val {
        if let Some(v) = self.staged.borrow().get(&x.key()) {
            return v.clone();
        }
        let v = dsl::metal::cast_qmm_input(x);
        self.staged.borrow_mut().insert(x.key(), v.clone());
        v
    }

    fn gemm_at(&self, x: &Val, w: &MatW, pt: &str, gpt: &str) -> Val {
        // `gemm_fits` is the COLUMN half of the tiled GEMM's contract and the
        // guard below is the row half. This family is the one that needs it:
        // `in_proj_a` and `in_proj_b` are one scalar per value head, 48 wide
        // on qwen3.6-27B, and the deployment's tile is 32.
        if !(self.multi_batch
            && self.metal.qmm_multi_batch
            && dsl::metal::gemm_fits(w.width, self.metal.qmm_tile))
        {
            return dsl::metal::qmv(x, w, pt);
        }
        let half = self.metal.qmm_fp16_precast.then(|| self.stage(x));
        let shape = (Shape(vec![Dim::Tokens, Dim::Const(w.width)]), DType::BF16);
        let (g, v) = dsl::guarded_value(x.trace(), w.layer, shape);
        g.arm(
            GuardPred::TokensMultipleOf(self.metal.qmm_tile.0.max(1)),
            || match &half {
                Some(h) => {
                    dsl::metal::qmm_fp16(h, w, gpt);
                }
                None => {
                    dsl::metal::qmm(x, w, gpt);
                }
            },
        )
        .otherwise(|| {
            dsl::metal::qmv(x, w, pt);
        });
        v
    }

    fn gemm(&self, x: &Val, w: &MatW) -> Val {
        self.gemm_at(x, w, &self.point, &self.gemm_point)
    }

    /// A projection that lands on the residual stream.
    fn gemm_add(&self, x: &Val, w: &MatW, residual: &Val) -> Val {
        if !self.metal.fuse_residual_gemv {
            return dsl::metal::residual_add(&self.gemm(x, w), residual);
        }
        if !(self.multi_batch
            && self.metal.qmm_multi_batch
            && dsl::metal::gemm_fits(w.width, self.metal.qmm_tile))
        {
            return dsl::metal::qmv_residual(x, w, residual, &self.point);
        }
        let half = self.metal.qmm_fp16_precast.then(|| self.stage(x));
        let shape = (Shape(vec![Dim::Tokens, Dim::Const(w.width)]), DType::BF16);
        let (g, v) = dsl::guarded_value(x.trace(), w.layer, shape);
        g.arm(
            GuardPred::TokensMultipleOf(self.metal.qmm_tile.0.max(1)),
            || match &half {
                Some(h) => {
                    dsl::metal::qmm_residual_fp16(h, w, residual, &self.gemm_point);
                }
                None => {
                    dsl::metal::qmm_residual(x, w, residual, &self.gemm_point);
                }
            },
        )
        .otherwise(|| {
            dsl::metal::qmv_residual(x, w, residual, &self.point);
        });
        v
    }

    /// A dense projection's handle at this deployment's representation.
    fn mat(&self, l: Option<u32>, name: &str, width: u32) -> MatW {
        MatW {
            name: match l {
                Some(l) => format!("layer.{l}.{name}"),
                None => name.to_string(),
            },
            width,
            layer: l,
            repr: self.metal.proj_repr,
        }
    }

    /// An RMS norm at the fold THE FACTS STATE, which for this family is
    /// gemma's `(1 + w)`.
    ///
    /// This text used to hardcode [`NormVariant::Plain`] and say so: *"qwen3.5
    /// states `Plain` everywhere, unlike its CUDA text's comment"*. The facts
    /// disagreed with it in the same tree -- every `Qwen35*Facts` fixture
    /// carries `norm_variant: Gemma`, and `Qwen35GdnFacts`'s own doc cites
    /// `qwen3_5_forward.cpp` launching `rmsnorm_gemma_bf16` for every block
    /// norm -- so one of the two was wrong and the field was dead either way.
    ///
    /// Three measurements say which, and none of them is a reading of anyone's
    /// source:
    ///
    /// - **The weights.** A plain norm's gain is trained from ones and stays
    ///   near one; a gemma norm's is trained from ZEROS because the fold
    ///   supplies the one. Qwen3.5-0.8B-Base ships `layer.0.mlp_norm` at mean
    ///   0.0855 and `layer.0.attn_norm` at 0.2382, with `q_norm` and `k_norm`
    ///   carrying negative channels outright. Multiplied directly, channels at
    ///   ±0.01 come out with OPPOSITE SIGNS where `1 + w` has them both at
    ///   nearly one.
    /// - **The sibling that is genuinely plain.** `linear_attn.norm.weight`,
    ///   the gated-DeltaNet output gain, ships at 0.96-0.99 -- near one, and it
    ///   is the one norm here that does not come through this function.
    ///   `gated_rms.wgsl` has no `plus_one` and needs none.
    /// - **The answer.** Folded plainly, this model replies a SPACE to every
    ///   prompt, with comma, period and newline behind it. Folded as gemma, on
    ///   the same weights and the same fire, "The capital of France is Paris.
    ///   The capital of France is" replies ` Paris` at the top of the
    ///   distribution.
    ///
    /// `driver-wgpu`'s `which_fold_the_final_norm_applies_and_what_each_one_answers`
    /// holds the third and prints both.
    fn norm(&self, x: &Val, l: Option<u32>, name: &str, row: u32) -> Val {
        let w = NormW {
            name: match l {
                Some(l) => format!("layer.{l}.{name}"),
                None => name.to_string(),
            },
            variant: self.norm_variant,
            per_head: None,
            layer: l,
        };
        dsl::metal::rms_norm(x, &w, row, self.metal.rms_eps)
    }
}

/// One FULL-attention layer: the 2×-wide query bank, its gate, and the
/// paged softmax between them.
fn full_attn(c: &Ctx<'_>, l: u32, f: &Qwen35FullAttnFacts, y: &Val) -> Val {
    let q_width = f.q_heads * f.head_dim;
    let kv_width = f.kv_heads * f.head_dim;
    let x = c.norm(y, Some(l), "attn_norm", f.hidden);

    // TWICE the query width, and the second half is not more queries: the
    // bank is `[rows, heads, 2, head_dim]`, each head's queries followed by
    // that head's gate. A row split at `q_width` would take the first half
    // of the HEADS rather than the first half of each head.
    let qg = c.gemm(&x, &c.mat(Some(l), "q_proj", 2 * q_width));
    let k = c.gemm(&x, &c.mat(Some(l), "k_proj", kv_width));
    let v = c.gemm(&x, &c.mat(Some(l), "v_proj", kv_width));
    let (q, gate) = dsl::metal::q_gate_split(&qg, f.q_heads, f.head_dim);

    let q = c.norm(&q, Some(l), "q_norm", f.head_dim);
    let k = c.norm(&k, Some(l), "k_norm", f.head_dim);
    // A QUARTER of each head rotates (`partial_rotary_factor: 0.25`), and
    // the rest passes through. The extent reaches the GRID, not the kernel.
    let (q, k) = dsl::metal::rope(
        &q,
        &k,
        c.multi_batch,
        c.metal.rope_theta,
        1.0,
        f.head_dim,
        f.rotary_dim,
        false,
    );
    let kv = Kv::at(x.trace(), l);
    dsl::metal::kv_append(&k, &v, &kv, /*paged=*/ true, f.head_dim, f.kv_heads);
    let a = dsl::metal::sdpa(
        &q,
        &kv,
        q_width,
        f.head_dim,
        /*paged=*/ true,
        f.q_heads / f.kv_heads.max(1),
        f.kv_heads,
        // No sliding window: qwen3.5's full layers attend the whole context,
        // and the linear layers beside them are not attention at all.
        -1,
        None,
        c.metal.attn_scale,
        c.multi_batch,
    )
    .expect("a plain attention statement produces its value");
    // The output gate, AFTER the softmax and before the o_proj — the
    // reference's `output * sigmoid(gate)`.
    let gated = dsl::metal::sigmoid_gate(&a, &gate, q_width);
    c.gemm_add(&gated, &c.mat(Some(l), "o_proj", f.hidden), y)
}

/// One LINEAR-attention layer: the gated deltanet.
fn gdn(c: &Ctx<'_>, l: u32, f: &Qwen35GdnFacts, y: &Val, class: FireClass) -> Val {
    let key_width = f.key_width();
    let v_width = f.value_width();
    let x = c.norm(y, Some(l), "attn_norm", f.hidden);

    // FOUR projections, not two. `mlx_lm` publishes `in_proj_qkv`,
    // `in_proj_z`, `in_proj_a` and `in_proj_b` as separate tensors, and the
    // fused pair the CUDA text can bind is a join this backend's loader does
    // not perform. Stating four is stating what the checkpoint ships.
    let qkv = c.gemm(&x, &c.mat(Some(l), "in_proj_qkv", f.conv_dim()));
    let z = c.gemm(&x, &c.mat(Some(l), "in_proj_z", v_width));
    let a = c.gemm(&x, &c.mat(Some(l), "in_proj_a", f.value_heads));
    let b = c.gemm(&x, &c.mat(Some(l), "in_proj_b", f.value_heads));

    // The offsets are where the reference splits the convolution's output:
    // `mx.split(conv_out, [key_dim, 2 * key_dim], -1)`. They are params
    // rather than a layout rule because the shader indexes a PACKED row and
    // has no other way to know where v begins.
    let shape = GdnShape {
        k_dim: f.key_head_dim,
        v_dim: f.value_head_dim,
        k_heads: f.key_heads,
        v_heads: f.value_heads,
        conv_dim: f.conv_dim(),
        conv_k: f.conv_kernel,
        q_off: 0,
        k_off: key_width,
        v_off: 2 * key_width,
        eps: c.metal.rms_eps,
    };
    let w = GdnW {
        conv_w: format!("layer.{l}.conv_w"),
        conv_b: format!("layer.{l}.conv_b"),
        a_log: format!("layer.{l}.a_log"),
        dt_bias: format!("layer.{l}.dt"),
    };
    let core = match class {
        // One token per request: nothing to share, so the fused form is
        // strictly fewer dispatches.
        FireClass::Decode => dsl::metal::gdn_core(&qkv, &a, &b, shape, &w, l),
        // Many: the q/k path is per KEY head and the scan is per VALUE head,
        // and this family runs twice as many of the latter.
        //
        // The PREFILL pair, not the slotted one. The slotted scan indexes the
        // recurrent state by slot and runs its grid over rows, so a prompt
        // fires one thread per token at one state -- the recurrence is over
        // tokens and that dispatch runs it in parallel over them. It cost
        // this stack a logit per fire and cost nothing to notice, because
        // nothing compared a fire to itself until the qwen references
        // existed. See `dsl::metal::gdn_prep_prefill`.
        FireClass::Prefill => {
            let (pre_q, pre_k, pre_gate) = dsl::metal::gdn_prep_prefill(&qkv, &a, &b, shape, &w, l);
            dsl::metal::gdn_core_recurrent_prefill(
                &qkv,
                &pre_q,
                &pre_k,
                &pre_gate,
                shape,
                &w,
                l,
                dsl::metal::GDN_SCAN_TILE,
            )
        }
    };
    // The z gate and the per-head norm in one dispatch, and the weight is
    // RAW: `gated_rms.metal` multiplies by it directly, because the
    // reference's `Qwen3NextRMSNormGated` is `nn.RMSNorm`-shaped and folds
    // nothing.
    let o = dsl::metal::gated_rms(
        &core,
        &z,
        &format!("layer.{l}.gate_norm"),
        f.value_heads,
        f.value_head_dim,
        c.metal.rms_eps,
    );
    c.gemm_add(&o, &c.mat(Some(l), "o_proj", f.hidden), y)
}

/// The MLP, dense or routed, landed on the residual stream.
fn mlp(
    c: &Ctx<'_>,
    l: u32,
    facts: &Qwen35HybridFacts,
    hidden: u32,
    y: &Val,
    class: FireClass,
) -> Val {
    let x = c.norm(y, Some(l), "mlp_norm", hidden);
    match &facts.mlp {
        Qwen35MlpKind::Dense { intermediate } => {
            let h = dsl::metal::silu_mul(
                &c.gemm(&x, &c.mat(Some(l), "gate_proj", *intermediate)),
                &c.gemm(&x, &c.mat(Some(l), "up_proj", *intermediate)),
                *intermediate,
            );
            c.gemm_add(&h, &c.mat(Some(l), "down", hidden), y)
        }
        Qwen35MlpKind::Moe(moe) => {
            let k = moe.top_k.max(1);
            // The gate at its OWN point. `qmv` directly rather than through
            // `gemm`: the gate is `[hidden, 256]` and no GEMM tile is that
            // narrow, so the guard would carry an arm that can never run.
            let gate_repr = c.metal.router_repr.unwrap_or(c.metal.proj_repr);
            let gate_bits = if c.metal.router_repr.is_some() {
                c.metal.router_bits
            } else {
                c.metal.affine_bits
            };
            let logits = dsl::metal::qmv(
                &x,
                &MatW {
                    repr: gate_repr,
                    ..c.mat(Some(l), "router", moe.num_experts)
                },
                &dsl::metal::affine_point(gate_repr, gate_bits),
            );
            let (ids, weights) = dsl::metal::router_topk(
                &logits,
                moe.num_experts,
                k,
                // No per-expert gain: that is gemma-4's, and this family
                // publishes no such tensor.
                None,
                c.metal.norm_topk_prob,
            );
            // WHICH ARM. A decode routes one token, so a GEMM tile would be
            // one live row in sixteen; a prefill routes `tokens * k`, and a
            // matvec there reads every expert's whole bank once per row. The
            // sort's block must be the GEMM's row tile or a tile spans two
            // experts and silently gets one of their banks.
            //
            // `moe_tile` is the llama-like tile for this family again -- see
            // `default_moe_tile` for the hang that had it off, which was
            // never the arm's.
            let tile = if class == FireClass::Prefill {
                c.metal.moe_tile
            } else {
                None
            };
            let block = tile.map_or(dsl::metal::ROUTE_BLOCK_MATVEC, |t| t.0);
            let (perm, row_expert, tile_expert, inv) =
                dsl::metal::route_sort(&ids, moe.num_experts, k, hidden, block);
            let rows = dsl::metal::route_gather(&x, &perm, moe.num_experts, k, hidden, block);
            let bank = |name: &str, width: u32| MatW {
                repr: c.metal.moe_repr.unwrap_or(c.metal.proj_repr),
                ..c.mat(Some(l), name, width)
            };
            let bits = if c.metal.moe_repr.is_some() {
                c.metal.moe_bits
            } else {
                c.metal.affine_bits
            };
            let project = |x: &Val, name: &str, width: u32, in_vec: u32| {
                if let Some(tile) = tile {
                    dsl::metal::routed_qmm(
                        x,
                        &row_expert,
                        &tile_expert,
                        &bank(name, width),
                        moe.num_experts,
                        k,
                        in_vec,
                        bits,
                        tile,
                        c.metal.routed_qmm_fp16 && bits == 4,
                    )
                } else {
                    dsl::metal::routed_qmv(
                        x,
                        &row_expert,
                        &bank(name, width),
                        k,
                        in_vec,
                        false,
                        bits,
                    )
                }
            };
            // `k * moe_intermediate` FOR THE MATVEC: each of its values is
            // one token's k expert results end to end, and the activation is
            // elementwise over all of it. The GEMM's rows are already the
            // sorted stack, one result each, so its width is one run.
            let h = dsl::metal::silu_mul(
                &project(&rows, "expert_gate", moe.moe_intermediate, hidden),
                &project(&rows, "expert_up", moe.moe_intermediate, hidden),
                if tile.is_some() {
                    moe.moe_intermediate
                } else {
                    moe.moe_intermediate * k
                },
            );
            let routed = dsl::metal::combine_sorted(
                &project(&h, "expert_down", hidden, moe.moe_intermediate),
                &weights,
                &inv,
                k,
                hidden,
            );
            let blended = if moe.shared_expert_intermediate == 0 {
                routed
            } else {
                let shared = dsl::metal::silu_mul(
                    &c.gemm(
                        &x,
                        &c.mat(Some(l), "shared_gate", moe.shared_expert_intermediate),
                    ),
                    &c.gemm(
                        &x,
                        &c.mat(Some(l), "shared_up", moe.shared_expert_intermediate),
                    ),
                    moe.shared_expert_intermediate,
                );
                // The shared expert's gate is a `[hidden, 1]` projection and
                // takes the gate's own point for the same reason the router
                // does: `mlx-community` quantizes both at eight bits.
                dsl::metal::shared_expert_combine(
                    &routed,
                    &c.gemm(&shared, &c.mat(Some(l), "shared_down", hidden)),
                    &dsl::metal::qmv(
                        &x,
                        &MatW {
                            repr: gate_repr,
                            ..c.mat(Some(l), "shared_gate_proj", 1)
                        },
                        &dsl::metal::affine_point(gate_repr, gate_bits),
                    ),
                    hidden,
                )
            };
            dsl::metal::residual_add(&blended, y)
        }
    }
}

/// The qwen3.5 hybrid, stated in Metal's symbols.
///
/// One text per [`FireClass`], as every Metal text is: the class picks the
/// recurrence's shape and the projections' lane, and both are decisions a
/// plan has to carry rather than discover.
#[must_use]
pub fn qwen3_5_hybrid_metal(
    facts: &Qwen35HybridFacts,
    metal: &Qwen35MetalFacts,
    class: FireClass,
) -> ForwardPlan {
    let hidden = hidden_of(facts);
    let (n_experts, moe_intermediate, shared_intermediate, intermediate) = match &facts.mlp {
        Qwen35MlpKind::Dense { intermediate } => (0, 0, 0, *intermediate),
        Qwen35MlpKind::Moe(moe) => (
            moe.num_experts,
            moe.moe_intermediate,
            moe.shared_expert_intermediate,
            0,
        ),
    };
    let shape = dsl::ModelShape {
        hidden,
        intermediate,
        n_experts,
        moe_intermediate,
        shared_intermediate,
        vocab: facts.vocab,
        head_dim: facts.attn.head_dim,
        q_width: facts.attn.q_width(),
        kv_width: facts.attn.kv_width(),
        qk_norm: model_ir::facts::QkNorm::PerHead,
        norm_variant: facts.norm_variant,
        tied_embeddings: facts.tied_embeddings,
        proj_repr: metal.proj_repr,
    };
    dsl::trace_metal("qwen3_5_hybrid", &shape, class, |m| {
        // Every layer-tagged statement below becomes implicitly
        // `rows(depth > layer)`, so a fire whose rows truncate at different
        // layers lowers to rectangles that narrow.
        m.depth_window();
        let c = Ctx {
            metal,
            point: dsl::metal::affine_point(metal.proj_repr, metal.affine_bits),
            gemm_point: dsl::metal::affine_gemm_point(
                metal.proj_repr,
                metal.affine_bits,
                metal.qmm_tile,
            ),
            multi_batch: class != FireClass::Decode,
            staged: std::cell::RefCell::new(std::collections::HashMap::new()),
            norm_variant: facts.norm_variant,
        };
        let t = m.trace();
        let mut y =
            dsl::metal::embed_gather(t, "embed", hidden, c.multi_batch, metal.proj_repr, &c.point);
        for l in 0..facts.layers {
            let after = if facts.is_full_attn(l) {
                full_attn(&c, l, &facts.attn, &y)
            } else {
                gdn(&c, l, &facts.gdn, &y, class)
            };
            y = mlp(&c, l, facts, hidden, &after, class);
        }
        let normed = c.norm(&y, None, "final_norm", hidden);
        // The readout's rows, and only those: a prefill computes one
        // distribution per REQUEST, not per token.
        let sampled = dsl::metal::sample_rows(&normed, hidden);
        let head = if facts.tied_embeddings {
            "embed"
        } else {
            "lm_head"
        };
        let logits = dsl::metal::lm_head(&sampled, head, facts.vocab, metal.proj_repr, &c.point);
        // THE EXIT SEAM, which this text did not declare.
        //
        // `lm_head` returns the distribution and this discarded it, so the plan
        // named no `seam::OUT` -- and a driver reading a fire's answer looks
        // for exactly that. `driver-wgpu` refuses with *"this text states no
        // exit, so it has no logits"*, and nothing had noticed because the
        // Metal suite measures this family's ACTIVATIONS: `device_real_weights`
        // asks whether they are finite and varied and where the first NaN is,
        // none of which needs a readout.
        //
        // `llama_like_metal` ends the same way and does declare it; so does
        // this family's own non-Metal epilogue, which is why the generic
        // `qwen3_5_hybrid` and the CUDA text both answer. The Metal one was
        // the odd text out.
        dsl::seam(t, &dsl::seam::OUT, &[&logits], None);
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::qwen_3_5::forward::facts::Qwen35MoeMlpFacts;

    /// `Qwen3.6-35B-A3B-4bit`, as `config.json` publishes it.
    fn a3b() -> Qwen35HybridFacts {
        Qwen35HybridFacts {
            layers: 40,
            full_attn_interval: 4,
            vocab: 248_320,
            tied_embeddings: false,
            norm_variant: NormVariant::Plain,
            attn: Qwen35FullAttnFacts {
                hidden: 2048,
                q_heads: 16,
                kv_heads: 2,
                head_dim: 256,
                // `max(2, 2 * int(0.5 * 0.25 * 256))` = 64.
                rotary_dim: 64,
                fused_qkv: false,
                norm_variant: NormVariant::Plain,
            },
            gdn: Qwen35GdnFacts {
                hidden: 2048,
                key_heads: 16,
                value_heads: 32,
                key_head_dim: 128,
                value_head_dim: 128,
                conv_kernel: 4,
                fused_in_proj: false,
                norm_variant: NormVariant::Plain,
            },
            mlp: Qwen35MlpKind::Moe(Qwen35MoeMlpFacts {
                hidden: 2048,
                num_experts: 256,
                top_k: 8,
                moe_intermediate: 512,
                shared_expert_intermediate: 512,
                norm_variant: NormVariant::Plain,
            }),
        }
    }

    fn metal() -> Qwen35MetalFacts {
        Qwen35MetalFacts {
            proj_repr: WeightRepr::Scaled {
                layout: model_dsl::ScaleLayout::PerGroup,
                group: 64,
                axis: 0,
                zero_point: true,
            },
            affine_bits: 4,
            moe_repr: None,
            moe_bits: 4,
            moe_tile: default_moe_tile(),
            router_repr: Some(WeightRepr::Scaled {
                layout: model_dsl::ScaleLayout::PerGroup,
                group: 64,
                axis: 0,
                zero_point: true,
            }),
            router_bits: 8,
            qmm_tile: (16, 32),
            qmm_fp16_precast: true,
            routed_qmm_fp16: true,
            qmm_multi_batch: true,
            fuse_residual_gemv: true,
            rms_eps: 1e-6,
            rope_theta: 10_000_000.0,
            attn_scale: 0.0625,
            norm_topk_prob: true,
        }
    }

    fn runs(plan: &ForwardPlan, kernel: &str) -> usize {
        plan.ops
            .iter()
            .filter(|op| match &op.kind {
                model_ir::trace::OpKind::Launch { kernel: k, .. } => k == kernel,
                _ => false,
            })
            .count()
    }

    /// The schedule reaches both layer kinds, in the reference's proportion.
    ///
    /// `full_attention_interval: 4` over 40 layers is ten full layers and
    /// thirty linear ones, and `is_full_attn` puts the full one LAST in each
    /// block (`layer_types[3] == "full_attention"`).
    #[test]
    fn thirty_layers_recur_and_ten_attend() {
        let plan = qwen3_5_hybrid_metal(&a3b(), &metal(), FireClass::Decode);
        assert_eq!(runs(&plan, "gdn_core_slotted_bfloat16"), 30);
        assert_eq!(runs(&plan, "gated_rms_bfloat16"), 30);
        assert_eq!(runs(&plan, "q_gate_split_bfloat16"), 10);
        assert_eq!(runs(&plan, "gate_bfloat16"), 10);
        assert_eq!(runs(&plan, "kv_append_paged_bfloat16"), 10);
    }

    /// A prefill splits the recurrence and a decode fuses it.
    ///
    /// Not a spelling difference: the split computes q and k once per KEY
    /// head where the fused form computes them once per VALUE head, and this
    /// family runs twice as many of the latter.
    #[test]
    fn the_prefill_splits_the_scan_the_decode_fuses() {
        let d = qwen3_5_hybrid_metal(&a3b(), &metal(), FireClass::Decode);
        let p = qwen3_5_hybrid_metal(&a3b(), &metal(), FireClass::Prefill);
        assert_eq!(runs(&d, "gdn_core_slotted_bfloat16"), 30);
        assert_eq!(runs(&d, "gdn_prep_slotted_bfloat16"), 0);
        assert_eq!(runs(&p, "gdn_core_slotted_bfloat16"), 0);
        assert_eq!(runs(&p, "gdn_prep_prefill_bfloat16"), 30);
        let (lanes, vrows) = dsl::metal::GDN_SCAN_TILE;
        assert_eq!(
            runs(
                &p,
                &format!("gdn_core_recurrent_prefill_bfloat16_l_{lanes}_v_{vrows}")
            ),
            30
        );
        assert_eq!(
            runs(&p, "gdn_prep_slotted_bfloat16"),
            0,
            "the slotted scan indexes state by SLOT and grids over ROWS, so a \
             prompt would race its own tokens"
        );
        assert_eq!(runs(&p, "gdn_core_recurrent_slotted_bfloat16"), 0);
    }

    /// Every mixture layer routes, and the shared expert lands beside it.
    #[test]
    fn every_layer_routes_and_blends_a_shared_expert() {
        let plan = qwen3_5_hybrid_metal(&a3b(), &metal(), FireClass::Decode);
        assert_eq!(runs(&plan, "router_topk_bfloat16"), 40);
        // No per-expert gain: that tensor is gemma-4's.
        assert_eq!(runs(&plan, "router_topk_scaled_bfloat16"), 0);
        assert_eq!(runs(&plan, "shared_expert_combine"), 40);
    }

    /// The router gate is read at EIGHT bits inside a four-bit stack.
    ///
    /// `mlx-community` lists `mlp.gate` and `mlp.shared_expert_gate` at
    /// `bits: 8` for all forty layers. Reading them at the stack's width is
    /// gpt-oss's measured failure: a fluent model routing every token to
    /// almost the right experts.
    #[test]
    fn the_gate_is_read_at_its_own_width() {
        let plan = qwen3_5_hybrid_metal(&a3b(), &metal(), FireClass::Decode);
        assert_eq!(runs(&plan, "affine_qmv_fast_bfloat16_gs_64_b_8"), 80);
    }

    /// Every symbol this text names is one this build compiles.
    #[test]
    fn every_symbol_names_a_kernel_this_backend_has() {
        for class in [FireClass::Decode, FireClass::Prefill] {
            let plan = qwen3_5_hybrid_metal(&a3b(), &metal(), class);
            for op in &plan.ops {
                let model_ir::trace::OpKind::Launch { kernel, .. } = &op.kind else {
                    continue;
                };
                assert!(
                    model_ir::kernels::stated_in(model_ir::kernels::Backend::Metal, kernel)
                        .is_some(),
                    "no metal routine states `{kernel}`"
                );
            }
        }
    }
}
