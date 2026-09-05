//! **DFlash, DFlash2 and DSpark** — the published block drafters, as model
//! text any family can carry. A head's own numbers are a [`Head`] descriptor
//! (one constant per published checkpoint); the family names one and spells
//! the four hooks the module doc lists. This file is everything behind them.

use checkpoint::contract::{Expr, TensorType};
use checkpoint_dsl::{Builder, Error, extents};
use model_dsl::{
    BlockDrafter, Dtype, HybridSpec, Input, KvSpace, Predicate, Value, Weight, ops, seam,
};

/// What the drafter reads off the trunk it is declared beside: the numbers
/// its banks are shaped by, and nothing about the trunk's own architecture.
#[derive(Clone, Copy, Debug)]
pub struct Trunk {
    pub hidden: u64,
    pub vocab: u64,
    pub norm_eps: f32,
    /// The element type the drafter's projections are carried at — the
    /// trunk's, since a drafter is quantized like its trunk (see
    /// [`DFlash::declare`]).
    pub weights: Dtype,
    /// The element type of norms and codebooks.
    pub dense: Dtype,
    pub tp: u32,
}

/// **A DFlash block drafter** — the shape `z-lab/Qwen3.6-27B-DFlash` ships.
///
/// Where a chained head fuses one hidden state with one token embedding and
/// runs a single block chained a token at a time, this fuses
/// [`taps`](DFlash::taps) TRUNK HIDDEN STATES and runs
/// [`blocks`](DFlash::blocks) layers ONCE over a whole block of rows whose
/// tail is the mask token, so a single pass proposes [`block`](DFlash::block)
/// tokens:
///
/// ```text
/// h  = rms(Σᵢ tapᵢ · fcᵢ)                       the taps, fused
/// for each block:  h += attn(rms(h));  h += mlp(rms(h))
/// draft = lm_head(rms(h))                       through the TARGET's head
/// ```
///
/// Three things differ from a chained head, and each is why this is its own
/// declaration rather than a flag on one:
///
/// * **The fusion is taps-wide.** The stored `fc.weight` is one
///   `[hidden, taps·hidden]` bank; the IR has no concat, so it is sliced into
///   `[hidden, hidden]` banks summed with `residual_add`, the same way a
///   chained head splits its two-wide bank.
/// * **The attention is NOT the family's site.** Its `q_proj` is
///   `[q_heads·head_dim, hidden]` with no gate to split off, and its
///   geometry is its own (32 q heads, 8 kv, head dim 128 against a 27B
///   trunk's 24/4/256). Its kv rows live in the trunk's page-id space at
///   their own plane width — a space admits rows of any width.
/// * **The v1 pass is bidirectional over the block.** Four of five layers
///   are sliding-window; the mask that makes the block see itself is the
///   guest's (`inputs.mask()`), not the model's. v2's layers are all sliding
///   and causal inside the block, and it needs no mask.
pub struct DFlash {
    /// Which trunk layers feed the fusion, in the order their banks are
    /// sliced out of `fc` — `[1, 16, 31, 46, 61]` for the v1 head, `[5, 19,
    /// 33, 47, 61]` for v2.
    pub taps: Vec<u32>,
    /// One `[hidden, hidden]` column slice of the stored `fc.weight` per tap.
    pub fc: Vec<Weight>,
    /// Scales the fused stream before the first block.
    pub hidden_norm: Weight,
    pub hidden_norm_eps: f32,
    pub blocks: Vec<DFlashBlock>,
    /// The final norm before the readout through the target's `lm_head`.
    pub norm: Weight,
    pub norm_eps: f32,
    /// Rows one draft pass proposes — the width of the `mtp.drafts` seam.
    pub block: u32,
    /// The token every block row but the first carries on the way in.
    pub mask_token: u32,
    /// DFlash2's candidate selector; `None` on a v1 head, whose readout is
    /// the per-slot argmax.
    pub selector: Option<Selector>,
    /// The first block row whose readout is a proposal — 1 for both DFlash
    /// heads (the anchor row proposes nothing), 0 for DSpark.
    pub proposals_from: u32,
    /// The published head this was declared from.
    pub head: &'static Head,
}

/// **DFlash2's candidate selector** — the head's readout. Each mask slot
/// keeps its `top_k` candidates off the target's head, and a path through
/// them is walked from the anchor: slot `s` picks
/// `argmax_c unary[c] + ⟨pred[prev] ⊙ hidden_projection(h_s), succ[c]⟩`,
/// `prev` the previous slot's pick (the anchor's id first). Two `[vocab,
/// rank]` codebooks and one projection; the walk is
/// `attention.selector_walk`, planted at the `mtp.drafts` seam where v1
/// plants its argmax.
pub struct Selector {
    /// `[rank, hidden]`; `None` for a plain bigram lattice (DSpark), whose
    /// score is `unary + ⟨pred[prev], succ[c]⟩` with no hidden term.
    pub hidden_projection: Option<Weight>,
    /// `[vocab, rank]`, indexed by the predecessor's id.
    pub pred: Weight,
    /// `[vocab, rank]`, indexed by the candidate's id.
    pub succ: Weight,
    pub top_k: u32,
}

/// One layer of a [`DFlash`] drafter: the standard pre-norm decoder block,
/// with its own ungated attention and a sliding window on all but the last.
pub struct DFlashBlock {
    pub mixer_norm: Weight,
    pub mixer_norm_eps: f32,
    pub attn: DraftAttn,
    pub mlp_norm: Weight,
    pub mlp_norm_eps: f32,
    pub mlp: DraftMlp,
    /// Keys older than this many positions are not attended; `None` on the
    /// one full-attention layer the v1 head ends with (v2 has none).
    pub window: Option<u32>,
    /// DFlash2's dynamic convolution around the attention sublayer; `None`
    /// on a v1 head.
    pub attn_conv: Option<DynConv>,
    /// The same around the MLP sublayer.
    pub mlp_conv: Option<DynConv>,
}

/// A draft block's gated MLP: one packed `[2·inter, hidden]` gate|up bank and
/// a `[hidden, inter]` down bank. A draft block routes to no experts.
pub struct DraftMlp {
    pub gate_up: Weight,
    pub down: Weight,
    pub inter: u32,
}

/// **DFlash2's two-tap grouped dynamic convolution** around one sublayer
/// (`attention_conv` / `mlp_conv`). One projection of the sublayer's normed
/// input yields the coefficients for BOTH sides — the input convolved before
/// the sublayer and its output convolved after — as a learned per-channel
/// base plus a per-group correction:
///
/// ```text
/// coeff  = proj(x)                        [rows, 2·taps·groups], (side, tap, group)
/// x'     = Σ_t (base[0, t] + δ[i, 0, t, g]) ⊙ x[i − t]      before the sublayer
/// y'     = Σ_t (base[1, t] + δ[i, 1, t, g]) ⊙ y[i − t]      after it
/// ```
///
/// along the block axis, `x[−1] = 0`. It runs `attention.block_dyn_conv`
/// twice a sublayer; the reference is `DFlashGroupedConv` in `mlx_dspark`.
pub struct DynConv {
    /// `[2·taps, hidden]`, row `side·taps + tap`: the stored
    /// `base_kernel [2, taps, hidden]` read flat.
    pub base: Weight,
    /// `[2·taps·groups, hidden]`: `kernel_projection.weight`.
    pub proj: Weight,
    pub taps: u32,
    /// Channels sharing one correction.
    pub group: u32,
}

/// A [`DFlash`] layer's attention site: plain (ungated) attention at the
/// head's own geometry, with per-head q and k norms and rope over the full
/// head.
pub struct DraftAttn {
    pub q_heads: u32,
    pub kv_heads: u32,
    pub head_dim: u32,
    pub rotary_dim: u32,
    pub theta: f32,
    pub sm_scale: f32,
    pub q_proj: Weight,
    pub k_proj: Weight,
    pub v_proj: Weight,
    pub o_proj: Weight,
    pub q_norm: Weight,
    pub q_norm_eps: f32,
    pub k_norm: Weight,
    pub k_norm_eps: f32,
    /// This layer's kv row, in the trunk's page-id space.
    pub kv: String,
}

/// **A PUBLISHED HEAD'S OWN NUMBERS** — its `config.json`. Every field is a
/// fact about one published checkpoint and not a knob, so the heads this
/// build knows are constants ([`QWEN36_27B_DFLASH`] and the rest below) and
/// a family names one when it declares the drafter. Nothing here is about
/// the trunk: a head's taps index the trunk's layers, and its `hidden` is
/// the trunk's, read off [`Trunk`].
#[derive(Debug, PartialEq)]
pub struct Head {
    /// The trunk layers whose hidden states feed the fusion, in the order
    /// their banks are sliced out of `fc` (`dflash_config.target_layer_ids`).
    pub taps: &'static [u32],
    /// One entry a layer (`layer_types`): the window a sliding layer attends
    /// the context through, or `None` for full attention — which is
    /// BIDIRECTIONAL over the block, since the reference skips the causal
    /// mask outright for such a head.
    pub windows: &'static [Option<u32>],
    pub q_heads: u32,
    pub kv_heads: u32,
    pub head_dim: u32,
    pub inter: u32,
    pub theta: f32,
    /// Rows one draft pass proposes (`block_size`) — the width of the
    /// `mtp.drafts` seam.
    pub block: u32,
    /// `dflash_config.mask_token_id`: the id every block row but the first
    /// carries into the draft pass.
    pub mask_token: u32,
    /// The first block row whose readout is a proposal: 1 when the anchor
    /// row proposes nothing, 0 when every row does (DSpark's `logits_start`).
    pub proposals_from: u32,
    /// The dynamic convolution around every sublayer, where the head has one
    /// (`conv_kernel_size` / `conv_group_size`).
    pub conv: Option<Conv>,
    pub readout: Readout,
}

/// A [`DynConv`]'s two numbers: taps of the convolution and channels
/// sharing one correction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Conv {
    pub taps: u32,
    pub group: u32,
}

/// How a head reads its proposals off the shared `lm_head`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Readout {
    /// The per-slot argmax.
    Argmax,
    /// DFlash2's candidate selector (`selector_rank` / `selector_top_k`):
    /// `unary + ⟨pred[prev] ⊙ proj(h), succ[c]⟩` over the slot's `top_k`.
    Selector { rank: u32, top_k: u32 },
    /// DSpark's markov bigram head (`markov_rank`): the same lattice with no
    /// hidden term. It biases the WHOLE vocabulary in the reference; here it
    /// biases the slot's top-`K` — the approximation DFlash2's selector was
    /// trained with, taken so one kernel serves both, and paid for in
    /// acceptance only (the verify is what makes a round lossless).
    Markov { rank: u32, top_k: u32 },
}

const SLIDING: Option<u32> = Some(2_048);

/// `z-lab/Qwen3.6-27B-DFlash`: block sixteen, four sliding layers then one
/// full, bidirectional over the block, argmax readout.
pub const QWEN36_27B_DFLASH: Head = Head {
    taps: &[1, 16, 31, 46, 61],
    windows: &[SLIDING, SLIDING, SLIDING, SLIDING, None],
    q_heads: 32,
    kv_heads: 8,
    head_dim: 128,
    inter: 17_408,
    theta: 10_000_000.0,
    block: 16,
    mask_token: 248_070,
    proposals_from: 1,
    conv: None,
    readout: Readout::Argmax,
};

/// `z-lab/Qwen3.8-27B-DFlash2`: block eight, five sliding layers causal
/// inside the block, a dynamic convolution around every sublayer, a
/// candidate selector for the readout. Heads, widths, window, theta and the
/// mask token are v1's.
pub const QWEN38_27B_DFLASH2: Head = Head {
    taps: &[5, 19, 33, 47, 61],
    windows: &[SLIDING; 5],
    q_heads: 32,
    kv_heads: 8,
    head_dim: 128,
    inter: 17_408,
    theta: 10_000_000.0,
    block: 8,
    mask_token: 248_070,
    proposals_from: 1,
    conv: Some(Conv { taps: 2, group: 16 }),
    readout: Readout::Selector { rank: 256, top_k: 16 },
};

/// `DimInfer/Qwen3.8-27B-Dspark-v1`: the v1 backbone (its taps, five plain
/// layers) with every layer full attention and the block bidirectional, a
/// block of fifteen whose EVERY row proposes (row `i` predicts position
/// `i + 1`, the anchor's row included), its own mask id, and a markov bigram
/// head for the readout. Its confidence head is not read yet.
pub const QWEN38_27B_DSPARK: Head = Head {
    taps: &[1, 16, 31, 46, 61],
    windows: &[None; 5],
    q_heads: 32,
    kv_heads: 8,
    head_dim: 128,
    inter: 17_408,
    theta: 10_000_000.0,
    block: 15,
    mask_token: 248_200,
    proposals_from: 0,
    conv: None,
    readout: Readout::Markov { rank: 256, top_k: 16 },
};

/// `z-lab/Qwen3.6-35B-A3B-DFlash`: the v1 shape against the 40-layer
/// mixture — eight taps, six layers (five sliding at 4096, then one full),
/// hidden 2048, MLP 6144, its own mask id.
pub const QWEN36_35B_A3B_DFLASH: Head = Head {
    taps: &[1, 6, 11, 16, 22, 27, 32, 37],
    windows: &[Some(4_096), Some(4_096), Some(4_096), Some(4_096), Some(4_096), None],
    q_heads: 32,
    kv_heads: 8,
    head_dim: 128,
    inter: 6_144,
    theta: 10_000_000.0,
    block: 16,
    mask_token: 248_077,
    proposals_from: 1,
    conv: None,
    readout: Readout::Argmax,
};

/// `z-lab/gemma-4-26B-A4B-it-DFlash`: the v1 shape against a 30-layer
/// trunk — six taps, a narrower MLP, theta 1e6, and the mask id 4 — read
/// out through gemma's softcapped head (monotone, so the argmax is the
/// head's own). The head is a Qwen3-style stack whatever the target.
pub const GEMMA4_26B_A4B_DFLASH: Head = Head {
    taps: &[1, 6, 11, 17, 22, 27],
    windows: &[SLIDING, SLIDING, SLIDING, SLIDING, None],
    q_heads: 32,
    kv_heads: 8,
    head_dim: 128,
    inter: 5_632,
    theta: 1_000_000.0,
    block: 16,
    mask_token: 4,
    proposals_from: 1,
    conv: None,
    readout: Readout::Argmax,
};

impl DFlash {
    /// Declare the head's planes under `prefix` (`aux` for an `--aux`
    /// overlay), shaped by the trunk's hidden width and element types.
    ///
    /// **THE DRAFTER IS QUANTIZED LIKE THE TRUNK, AND IT COSTS NOTHING TO
    /// BE.** The shipped head is bf16 and only 1.75 G parameters against the
    /// trunk's 27, so carrying it unquantized is affordable — the obvious
    /// place to look for acceptance. It was tried, on this text with the
    /// projections at `dense`: the artifact went 15.0 -> 17.3 GiB, a draft
    /// fire went 17.8 -> 27.8 ms, and the accepted prefix DID NOT MOVE —
    /// 13.500 / 10.000 / 4.375 of fifteen on counting / code / recall against
    /// 13.500 / 10.125 / 5.250 at four bits, the one differing round being
    /// noise over eight. What a block drafter accepts is a property of the
    /// HEAD, not of the precision it is carried at.
    #[must_use]
    pub fn declare(head: &'static Head, prefix: &str, trunk: &Trunk) -> DFlash {
        let (hidden, w, dense, tp) = (trunk.hidden, trunk.weights, trunk.dense, trunk.tp);
        let n = |s: &str| format!("{prefix}.{s}");
        // Heads differ in their taps, their layers' windows, the block, the
        // convolution and the readout — the descriptor's fields; everything
        // else below is one text.
        let conv = |l: u32, which: &str| {
            head.conv.map(|c| DynConv {
                base: Weight::sym(
                    format!("{prefix}.layers.{l}.{which}.base_kernel"),
                    [2 * u64::from(c.taps), hidden],
                    dense,
                ),
                proj: Weight::sym(
                    format!("{prefix}.layers.{l}.{which}.kernel_projection"),
                    [2 * u64::from(c.taps) * (hidden / u64::from(c.group)), hidden],
                    w,
                )
                .columns(),
                taps: c.taps,
                group: c.group,
            })
        };
        let (dq, dkv, dhd) = (head.q_heads / tp, head.kv_heads / tp, head.head_dim);
        let hd = u64::from(dhd);
        let inter = head.inter / tp;
        // Read seven times sixteen rows a fire: carried dense, where
        // precision costs nothing and the gather is exact.
        let codebook = |s: &str, rank: u32| Weight::sym(n(s), [trunk.vocab, u64::from(rank)], dense);
        DFlash {
            taps: head.taps.to_vec(),
            // One column slice of the stored `[hidden, taps·hidden]` bank per
            // tap; replicated, since a trunk hidden state is.
            fc: (0..head.taps.len())
                .map(|i| Weight::sym(n(&format!("fc_tap{i}")), [hidden, hidden], w))
                .collect(),
            hidden_norm: Weight::sym(n("hidden_norm"), [hidden], dense),
            hidden_norm_eps: trunk.norm_eps,
            blocks: head
                .windows
                .iter()
                .zip(0u32..)
                .map(|(&window, l)| {
                    let b = |s: &str| format!("{prefix}.layers.{l}.{s}");
                    DFlashBlock {
                        mixer_norm: Weight::sym(b("mixer_norm"), [hidden], dense),
                        mixer_norm_eps: trunk.norm_eps,
                        attn: DraftAttn {
                            q_heads: dq,
                            kv_heads: dkv,
                            head_dim: dhd,
                            rotary_dim: dhd,
                            theta: head.theta,
                            sm_scale: (dhd as f32).sqrt().recip(),
                            q_proj: Weight::sym(b("q_proj"), [u64::from(dq) * hd, hidden], w).columns(),
                            k_proj: Weight::sym(b("k_proj"), [u64::from(dkv) * hd, hidden], w).columns(),
                            v_proj: Weight::sym(b("v_proj"), [u64::from(dkv) * hd, hidden], w).columns(),
                            o_proj: Weight::sym(b("o_proj"), [hidden, u64::from(dq) * hd], w).rows(),
                            q_norm: Weight::sym(b("q_norm"), [hd], dense),
                            q_norm_eps: trunk.norm_eps,
                            k_norm: Weight::sym(b("k_norm"), [hd], dense),
                            k_norm_eps: trunk.norm_eps,
                            kv: format!("kv.dflash.{l}"),
                        },
                        mlp_norm: Weight::sym(b("mlp_norm"), [hidden], dense),
                        mlp_norm_eps: trunk.norm_eps,
                        mlp: DraftMlp {
                            gate_up: Weight::sym(b("gate_up"), [2 * u64::from(inter), hidden], w)
                                .packed([u64::from(inter), u64::from(inter)]),
                            down: Weight::sym(b("down"), [hidden, u64::from(inter)], w).rows(),
                            inter,
                        },
                        window,
                        attn_conv: conv(l, "attention_conv"),
                        mlp_conv: conv(l, "mlp_conv"),
                    }
                })
                .collect(),
            norm: Weight::sym(n("norm"), [hidden], dense),
            norm_eps: trunk.norm_eps,
            block: head.block,
            mask_token: head.mask_token,
            proposals_from: head.proposals_from,
            selector: match head.readout {
                Readout::Argmax => None,
                Readout::Selector { rank, top_k } => Some(Selector {
                    hidden_projection: Some(
                        Weight::sym(
                            n("candidate_selector.hidden_projection"),
                            [u64::from(rank), hidden],
                            w,
                        )
                        .columns(),
                    ),
                    pred: codebook("candidate_selector.predecessor_codebook", rank),
                    succ: codebook("candidate_selector.successor_codebook", rank),
                    top_k,
                }),
                Readout::Markov { rank, top_k } => Some(Selector {
                    hidden_projection: None,
                    pred: codebook("markov_w1", rank),
                    succ: codebook("markov_w2", rank),
                    top_k,
                }),
            },
            head,
        }
    }

    /// Register the drafter's kv rows — one per block — in `space`, the
    /// trunk's page-id space: the drafter attends the same sequence at the
    /// same lengths, and a space admits rows of any plane width.
    pub fn declare_caches(&self, c: &mut HybridSpec, space: KvSpace) {
        for b in &self.blocks {
            let a = &b.attn;
            let plane = u64::from(a.kv_heads) * u64::from(a.head_dim);
            c.kv(space, a.attn_kv(), [plane, plane]);
        }
    }

    /// **THE TAPS ARE FUSED WHERE THEY ARE TAKEN, NOT COLLECTED.** Call this
    /// with the residual stream `y` after every trunk layer `layer`; at a
    /// tapped layer it folds `y · fcᵢ` into `fused`. A residual add ALIASES
    /// its output onto the stream it folds into, so the trunk's hidden state
    /// is ONE buffer and a handle held across a later layer reads that
    /// layer's value, not the tapped one. The fusion's `[hidden, taps·hidden]`
    /// bank is its column slices summed, and a slice's matmul allocates — so
    /// taking the tap's product here is both the fusion and the snapshot, at
    /// no extra cost.
    pub fn tap(&self, layer: u32, y: &Value, fused: &mut Option<Value>) {
        let Some(at) = self.taps.iter().position(|t| *t == layer) else {
            return;
        };
        let part = ops::linear::matmul(y, &self.fc[at]);
        *fused = Some(match fused.take() {
            Some(sum) => ops::elemwise::residual_add(&part, &sum),
            None => part,
        });
    }

    /// **THE BLOCK DRAFTER'S TWO ARMS** — the context it caches, and the
    /// block it proposes.
    ///
    /// The reference (`z-lab/dflash`'s `model_mlx.py`) runs
    /// `h = layer(h, h_ctx, rope, cache)` over five layers, where `h_ctx` is
    /// **passed to every layer and updated by none**, and all a layer does
    /// with it is `cache.update_and_fetch(k_proj(h_ctx), v_proj(h_ctx))`. A
    /// fixed row set that contributes keys and values at every layer,
    /// cached, IS a kv row — so there is no second stream to carry here. The
    /// context is written into the drafter's kv rows over the TRUNK's own
    /// rows, and the draft pass reads it back out of the cache in a later
    /// fire.
    ///
    /// `fused` is what [`tap`](DFlash::tap) accumulated; `h_block` the draft
    /// rows' embedding; `mask` the guest's mask channel (read by v1's full
    /// layer alone); `block_draft` the family's fact for the draft rows.
    /// Returns the block rows' final hidden for the caller to merge before
    /// the shared readout.
    pub fn arm<F>(
        &self,
        inputs: &Input<F>,
        fused: &Value,
        h_block: &Value,
        mask: &Value,
        block_draft: &Predicate,
    ) -> Value {
        let d = self;
        // ── THE CONTEXT, WHICH IS THE CACHE ──────────────────────────────
        // **ON EVERY TRUNK FIRE, NOT ONLY A DRAFTING ONE.** The taps carry
        // the trunk's own arm and that is the whole guard this wants: a fire
        // whose rows the trunk ran must leave the drafter's context behind,
        // or the drafter attends over a sequence with holes in it the next
        // time it drafts. (The chained heads guard their work on a `drafts`
        // fact, but that fact is INFERRED from a program reading the draft
        // seam, and a block drafter plants that seam on its block rows alone
        // — so a verify fire could never set it, and the context would never
        // be written.) It costs a five-way fusion and ten projections a
        // fire, under a percent of a decode.
        let h_ctx = ops::elemwise::rmsnorm_plus_one(fused, &d.hidden_norm, d.hidden_norm_eps);
        // Spelled the way the keys beside them are: the taps these positions
        // go with are already inside the trunk's arm.
        let (_, ctx_positions) = inputs.positions().split(block_draft);
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

        // ── THE BLOCK ────────────────────────────────────────────────────
        let (input_block, _) = inputs.split(block_draft);
        let (block_positions, _) = inputs.positions().split(block_draft);
        let mut h = h_block.clone();
        for b in &d.blocks {
            let a = &b.attn;
            let hd = a.head_dim;
            let plan = ops::attn::plan_prefill(&input_block, a.q_heads, a.kv_heads, hd, None);
            let x = ops::elemwise::rmsnorm_plus_one(&h, &b.mixer_norm, b.mixer_norm_eps);
            // DFlash2: the dynamic convolution's two sides, both from this
            // one input (`DynConv`); a v1 head has none and `x` passes through.
            let (x, attn_coeff) = conv_prepare(&x, b.attn_conv.as_ref());
            let q = ops::linear::matmul(&x, &a.q_proj);
            let k = ops::linear::matmul(&x, &a.k_proj);
            let v = ops::linear::matmul(&x, &a.v_proj);
            let q = ops::elemwise::rmsnorm_per_head_plus_one(&q, &a.q_norm, hd, a.q_norm_eps);
            let k = ops::elemwise::rmsnorm_per_head_plus_one(&k, &a.k_norm, hd, a.k_norm_eps);
            let (q, k) =
                ops::elemwise::rope_partial(&q, &k, &block_positions, a.rotary_dim, hd, a.theta);
            // The block's own kv joins the row the context is already in,
            // and the guest rolls it back with `kv_len` next round — the
            // transient half of `keys = concat(cache.fetch(ctx), prop)`.
            ops::attn::kv_append(
                &k,
                &v,
                inputs.kv(&a.kv),
                &inputs.write_page(&a.kv),
                &inputs.write_offset(&a.kv),
            );
            // A sliding layer is causal within the block and windowed over
            // the context, which is what a windowed prefill IS. v1's last
            // layer is full attention and BIDIRECTIONAL over the block —
            // only a stated mask says that, and it is the guest's.
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
                // **NOT CAUSAL.** The reference skips `create_causal_mask`
                // outright when `is_causal` is false, so a mask row sees the
                // whole block, its own future included. Stated here because
                // a mask alone cannot say it — pie's masked read is causal
                // AND mask unless the op says otherwise.
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
            let o = conv_finish(&ops::linear::matmul(&o, &a.o_proj), b.attn_conv.as_ref(), attn_coeff.as_ref());
            h = ops::elemwise::residual_add(&o, &h);

            let x = ops::elemwise::rmsnorm_plus_one(&h, &b.mlp_norm, b.mlp_norm_eps);
            let (x, mlp_coeff) = conv_prepare(&x, b.mlp_conv.as_ref());
            let f = ops::linear::matmul(
                &ops::linear::mlp_swiglu(&ops::linear::matmul(&x, &b.mlp.gate_up), b.mlp.inter),
                &b.mlp.down,
            );
            let f = conv_finish(&f, b.mlp_conv.as_ref(), mlp_coeff.as_ref());
            h = ops::elemwise::residual_add(&f, &h);
        }
        ops::elemwise::rmsnorm_plus_one(&h, &d.norm, d.norm_eps)
    }

    /// Plant the head's proposals on the seams a guest reads: the draft
    /// rows' logits on `mtp`, and one id a row on `mtp.drafts` — a v1 head's
    /// per-slot argmax, or DFlash2's selector walk over the slot's top
    /// candidates. The head's own readout either way, so the guest reads one
    /// seam whichever head the load carries.
    ///
    /// `logits` is the shared `lm_head`'s output over trunk and draft rows;
    /// `hb` the block rows' final hidden [`arm`](DFlash::arm) returned.
    pub fn plant_readout<F>(
        &self,
        logits: &Value,
        inputs: &Input<F>,
        hb: Option<&Value>,
        block_draft: &Predicate,
    ) {
        // The facts a guest needs to seed this head's block, on the trace for
        // the load to advertise: v1 is bidirectional over the block (its full
        // layer), v2 is causal inside it.
        logits.rec().block_drafter(BlockDrafter {
            rows: self.block,
            mask_token: self.mask_token,
            bidirectional: self.blocks.iter().any(|b| b.window.is_none()),
            proposals_from: self.proposals_from,
        });
        let (dlogits, _) = logits.split(block_draft);
        seam::at(seam::MTP, &[&dlogits]);
        let picks = match (&self.selector, hb) {
            (Some(sel), Some(hb)) => {
                let (unary, cand) = ops::layout::topk(&dlogits, sel.top_k);
                let hp = sel
                    .hidden_projection
                    .as_ref()
                    .map(|proj| ops::linear::matmul(hb, proj));
                let (toks, _) = inputs.tokens().split(block_draft);
                ops::attn::selector_walk(
                    &cand,
                    &unary,
                    hp.as_ref(),
                    &toks,
                    &sel.pred,
                    &sel.succ,
                    self.proposals_from,
                )
            }
            _ => ops::layout::argmax(&[&dlogits]),
        };
        seam::at(seam::MTP_DRAFTS, &[&picks]);
    }

    /// Bind the head's `--aux` planes. The head is published on its own, so
    /// it always rides in under `aux.`; its layers are the standard pre-norm
    /// decoder spelling — `input_layernorm`, `self_attn.*`,
    /// `post_attention_layernorm`, `mlp.*` — and its `q_proj` is the plain
    /// one. `norm` is the family's rule for reading an RMSNorm weight (the
    /// mlx_lm `+1` fold, where the layout carries it).
    ///
    /// **THE DRAFTER'S NORMS ARE FOLDED LIKE THE TRUNK'S**, measured rather
    /// than assumed. Whether an `--aux` checkpoint carries mlx_lm's `+1` fold
    /// is a question about its publisher, not about the target, so it was
    /// A/B'd on the artifact with everything else fixed: folded, the drafter
    /// keeps 0.33 / 0.33 / 0.67 of a block on counting / code / recall;
    /// unfolded, 0.33 / 0.00 / 0.33.
    ///
    /// # Errors
    ///
    /// Whatever the builder refuses: a plane the source lacks, a shape that
    /// does not read as declared.
    pub fn bind_aux(
        &self,
        b: &mut Builder,
        src: &ztensor::Source,
        norm: &dyn Fn(String) -> Expr,
    ) -> Result<(), Error> {
        b.read_expr(&self.hidden_norm, norm("aux.hidden_norm.weight".to_string()))?;
        // `fc.weight` is one `[hidden, taps·hidden]` bank; tap `i` is columns
        // `i·hidden .. (i+1)·hidden`, sliced as a chained head's two-wide
        // bank is.
        let span = extents(&self.fc[0])[1];
        for (i, bank) in self.fc.iter().enumerate() {
            let at = span * i as i64;
            b.read_expr(bank, Expr::src("aux.fc.weight".to_string()).slice(1, at, span))?;
        }
        for (l, block) in self.blocks.iter().enumerate() {
            let n = |s: &str| format!("aux.layers.{l}.{s}");
            let a = &block.attn;
            b.read_expr(&block.mixer_norm, norm(n("input_layernorm.weight")))?;
            b.read(&a.q_proj, n("self_attn.q_proj.weight"))?;
            b.read(&a.k_proj, n("self_attn.k_proj.weight"))?;
            b.read(&a.v_proj, n("self_attn.v_proj.weight"))?;
            b.read(&a.o_proj, n("self_attn.o_proj.weight"))?;
            b.read_expr(&a.q_norm, norm(n("self_attn.q_norm.weight")))?;
            b.read_expr(&a.k_norm, norm(n("self_attn.k_norm.weight")))?;
            b.read_expr(&block.mlp_norm, norm(n("post_attention_layernorm.weight")))?;
            // DFlash2's dynamic convolutions: the stored `[2, taps, hidden]`
            // base read flat as `[2·taps, hidden]`, and the coefficient
            // projection as any other bank.
            for (conv, which) in [(&block.attn_conv, "attention_conv"), (&block.mlp_conv, "mlp_conv")] {
                if let Some(c) = conv {
                    let want: Vec<i64> = extents(&c.base);
                    b.read_expr(&c.base, flat(src, n(&format!("{which}.base_kernel")), want)?)?;
                    b.read(&c.proj, n(&format!("{which}.kernel_projection.weight")))?;
                }
            }
            b.read_concat(
                &block.mlp.gate_up,
                [n("mlp.gate_proj.weight"), n("mlp.up_proj.weight")],
            )?;
            b.read(&block.mlp.down, n("mlp.down_proj.weight"))?;
        }
        b.read_expr(&self.norm, norm("aux.norm.weight".to_string()))?;
        // DFlash2's selector: a projection bank and two codebooks, the
        // codebooks stored as plain arrays (no `.weight`).
        match (&self.selector, self.head.readout) {
            (Some(sel), Readout::Selector { .. }) => {
                if let Some(proj) = &sel.hidden_projection {
                    b.read(proj, "aux.candidate_selector.hidden_projection.weight".to_string())?;
                }
                b.read(&sel.pred, "aux.candidate_selector.predecessor_codebook".to_string())?;
                b.read(&sel.succ, "aux.candidate_selector.successor_codebook".to_string())?;
            }
            // DSpark's markov head: `w1` an embedding indexed by the previous
            // id, `w2` a linear whose weight is `[vocab, rank]` — the same
            // shape, read as the successor codebook. Its confidence head
            // (`confidence_head.proj.*`) is left out until a guest reads it.
            (Some(sel), Readout::Markov { .. }) => {
                b.read(&sel.pred, "aux.markov_head.markov_w1.weight".to_string())?;
                b.read(&sel.succ, "aux.markov_head.markov_w2.weight".to_string())?;
            }
            _ => {}
        }
        Ok(())
    }
}

impl DraftAttn {
    /// The kv row's name, owned, for a cache declaration.
    fn attn_kv(&self) -> String {
        self.kv.clone()
    }
}

/// The input side of a [`DynConv`]: project the coefficients off the normed
/// input and convolve it with side 0; the coefficients come back for the
/// output side. A block without one passes `x` through.
fn conv_prepare(x: &Value, conv: Option<&DynConv>) -> (Value, Option<Value>) {
    match conv {
        Some(c) => {
            let coeff = ops::linear::matmul(x, &c.proj);
            let x = ops::attn::block_dyn_conv(x, &coeff, &c.base, 0, c.taps, c.group);
            (x, Some(coeff))
        }
        None => (x.clone(), None),
    }
}

/// The output side of a [`DynConv`], with the coefficients its input side
/// projected.
fn conv_finish(y: &Value, conv: Option<&DynConv>, coeff: Option<&Value>) -> Value {
    match (conv, coeff) {
        (Some(c), Some(coeff)) => ops::attn::block_dyn_conv(y, coeff, &c.base, 1, c.taps, c.group),
        _ => y.clone(),
    }
}

/// The same bytes read as `want`: a stored rank-N kernel as a rank-2 bank.
/// The checkpoint DSL has no reshape node, only a transmute of the type, so
/// this checks the element count and re-states it.
fn flat(src: &ztensor::Source, from: String, want: Vec<i64>) -> Result<Expr, Error> {
    let Some(tensor) = src.get(&from) else {
        return Err(Error::Missing(from));
    };
    let illegible = |why: &dyn std::fmt::Display| Error::Illegible {
        name: from.clone(),
        detail: why.to_string(),
    };
    let shape = tensor.shape();
    let stored: i128 = shape.iter().map(|&n| i128::from(n)).product();
    let asked: i128 = want.iter().map(|&n| i128::from(n)).product();
    if stored != asked {
        return Err(illegible(&format!(
            "is stored {shape:?} ({stored} elements) and the plan reads it as {want:?} \
             ({asked} elements)"
        )));
    }
    let stored = checkpoint::file::encoding_of(&tensor).map_err(|why| illegible(&why))?;
    Ok(Expr::src(from).transmute(TensorType::new(want, stored)))
}
