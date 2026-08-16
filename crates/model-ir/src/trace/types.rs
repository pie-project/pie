//! THE WORDS A SHAPE, A CLASS AND A PREDICATE ARE WRITTEN IN.
//!
//! Nothing here refers to an op: these are the types an [`Op`](super::Op)'s
//! fields are made OF, and they are separated for that reason -- a reader
//! asking what `Dim::Requests` means should not have to scroll past the op
//! vocabulary to find it.

use serde::{Deserialize, Serialize};

/// One symbolic extent of a value's shape.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Dim {
    /// The fire's token rows (`N`; equals `Requests` on a pure-decode fire).
    Tokens,
    /// The fire's request rows (`R`).
    Requests,
    /// A load-time constant: hidden size, head count x head dim, vocab.
    Const(u32),
    /// The MoE ALIGNED path's padded route count.
    ///
    /// `ceil((N·k + min(E, N·k)·(block-1)) / block) · block` — routes
    /// bucketed by expert and each bucket padded to a whole block, so one
    /// batched GEMM covers every expert. The padding is what makes it not
    /// `Tokens` and not a `Const`: it grows with the fire AND with how
    /// many experts a fire happens to touch.
    ///
    /// Every input but `N` is load-time (`top_k`, `experts` and the block
    /// size the driver picks from the route count), so the extent is a
    /// function of the fire's own token count and nothing else — which is
    /// exactly what a symbolic dim has to be.
    ///
    /// This is the extent the north-star doc said "no `Dim` spells", and
    /// it is why the aligned leg could not be stated.
    MoeAlignedRoutes {
        top_k: u32,
        experts: u32,
        block: u32,
    },
}

impl Dim {
    /// The aligned route count for a fire of `n` tokens.
    ///
    /// The driver computes the same number from `moe_aligned_block`; this
    /// is the host-side reading, used when a lowering needs the extent to
    /// size a rectangle.
    pub fn moe_aligned_rows(n: u32, top_k: u32, experts: u32, block: u32) -> u32 {
        let routes = n * top_k;
        let padded = routes + experts.min(routes) * block.saturating_sub(1);
        padded.div_ceil(block.max(1)) * block.max(1)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Shape(pub Vec<Dim>);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DType {
    BF16,
    /// Half. Only gpt-oss's MXFP4 routed GEMVs consume it: their
    /// activation operand is cast from bf16 first, and typing that cast's
    /// output BF16 would say the cast did nothing.
    F16,
    F32,
    I32,
}

/// Index into [`ForwardPlan::values`](super::ForwardPlan::values).
pub type ValueId = u32;

/// The `dyn` marker: which fire extent a value's *selection* varies over.
///
/// Marks values whose content chooses lowering-relevant structure per
/// element of an extent — today only the per-token expert assignment a
/// [`OpKind::TopK`](super::OpKind::TopK) produces. Ordinary activations are per-token *data* and
/// carry no marker; the marker means "the planner must look at this value's
/// content to know which weights a downstream op reads" (plan.md Part 1's
/// `dyn PerToken<Expert>`). `PerRequest` (adapters, depth) is the same
/// grammar at request granularity and lands with its own axes later.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DynAxis {
    /// Varies per token row of the fire (`Dim::Tokens` granularity).
    PerToken,
}

/// RMSNorm weight conventions that change the arithmetic, not the kernel
/// choice. `Gemma` folds `(1 + w)`; `Plain` multiplies `w` directly.
///
/// `Default` is `Plain` so the field can ride serde-additively on ops that
/// predate it ([`OpKind::RmsnormPerHead`](super::OpKind::RmsnormPerHead)): a golden that never stated a
/// variant reads back as the plain fold it always meant.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum NormVariant {
    #[default]
    Plain,
    Gemma,
}

impl NormVariant {
    /// Serde helper: `Plain` is the resting value and is skipped on
    /// serialization, the discipline that keeps pre-variant goldens
    /// byte-identical (the same rule as `selector`/`dyn_axis`).
    pub fn is_plain(&self) -> bool {
        *self == NormVariant::Plain
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RopeKind {
    Standard,
    /// Llama3/YaRN-style frequency scaling; parameters live in the facts.
    Yarn,
}

/// The fire-shape class a LOWERED trace is specialized to (north-star-dsl.md).
///
/// What varies after model load and CHANGES WHICH OPS RUN: the toolchain
/// traces a lowered declaration once per class, so inside the
/// declaration a class arm is an ordinary trace-time `match` — the same
/// mechanism that erases static facts, applied to the axes that used to
/// be the drivers' `is_pure_decode` / `commit_advance` / `state_only`
/// booleans. Anything that changes only a kernel's PARAMETER
/// (`verify_frozen`'s write_state) is not a class; anything that changes
/// a kernel choice per fire within one op list is a [`GuardPred`].
///
/// Semantic traces (`llama_like` (the model text)) have no class: they
/// serve every fire shape, and kernel choice stays with their consumer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum FireClass {
    /// Pure decode: every request contributes one token row.
    Decode,
    /// Prefill-shaped (anything else — the hand-written bodies treat
    /// mixed fires as one qo_indptr-windowed prefill, a decode row being
    /// an `Nr == 1` window).
    Prefill,
    // THE REPAIR CLASSES ARE GONE (`.wiki/driver/graph.md` §4.2).
    //
    // `CommitAdvance`, `StateOnly` and `FrozenVerify` were spec-decode's
    // repair passes, and they existed only because the driver refused the
    // ABI mechanism that makes repair unnecessary. A speculative decode
    // writes its tokens into a BUFFER and folds only the accepted prefix
    // into the linear state; a rejected token is never folded, so nothing
    // is ever wrong and nothing needs repairing. `FrozenVerify` was
    // "prefill plus a verify-stash store" — the buffer IS the stash.
    // `CommitAdvance` was "replay the confirmed prefix" — the fold length
    // IS that prefix. `StateOnly` was the backbone with the epilogue cut
    // off, which is a readout question, not a pass.
    //
    // The driver accepts `PIE_RS_FLAG_FOLD` / `_BUFFER_WRITE` /
    // `_FOLD_LEN_DEVICE` now, so all three lost their reason to exist.
    //
    // The masked classes (wire 5/6) and the hooked classes (wire 7/8)
    // are RETIRED (A1/A2, the class-collapse amendment): a custom mask
    // is a GuardPred::HasCustomMask arm and attached stage hooks are a
    // GuardPred::HasStageHooks arm of the Decode/Prefill traces now —
    // the op-list deltas are local, so they live at op granularity.
    // What remains a class is what changes the PASS wholesale: the
    // fire's shape and the MTP services. The wire numbers stay
    // reserved (append-only ABI); the trace entries answer
    // InvalidArgument.
}

impl FireClass {
    /// The suffix a trace's family name carries for this class.
    ///
    /// Every family spelled this match itself — nine of them, identically,
    /// beside a `panic!` arm for whatever they had not written yet — which
    /// is how seven families came to refuse Prefill by omission rather than
    /// by decision. One statement, so a family names its class instead of
    /// re-deciding what the classes are called.
    #[must_use]
    pub const fn suffix(self) -> &'static str {
        match self {
            Self::Decode => "decode",
            Self::Prefill => "prefill",
        }
    }
}

// (The short-lived `AttnKernel` enum — rung 1's `Attention.param1` tag —
// is gone: a lowered trace states its attention kernel the way it states
// every kernel, as an [`OpKind::Launch`] with the launcher's name. Raw
// signatures, not enum tags; north-star-dsl.md.)

/// A [`OpKind::Guard`](super::OpKind::Guard) arm's predicate: the ONE kind of branch a lowered
/// trace may carry — over a per-fire RUNTIME INPUT, closed-vocabulary so
/// every predicate is emittable as a fixed C++ condition (rung 3's
/// generated form spells it; the interpreter evaluates it). Trace-time
/// facts never appear here — they resolved during tracing. Predicates
/// may carry a load-time VALUE (the token thresholds: env-tunable
/// driver constants, resolved into the trace like every fact) but never
/// an open-ended expression: a declaration cannot smuggle an arbitrary
/// runtime choice past the toolchain. Wire form: a (kind, payload) u32
/// pair; kinds are appended-only.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GuardPred {
    /// The fire carries explicit KV-write descriptors (`w_page`/`w_off`),
    /// the graph-replay steering path — `has_write_desc` in every driver
    /// signature. Wire kind 0, payload unused.
    HasWriteDesc,
    /// `N <= k`: the fire's token rows within a threshold (the
    /// warp-tiled prefill ceiling). Wire kind 1, payload `k`.
    TokensLE(u32),
    /// `N > k` (the cached-prefill floor). Wire kind 2, payload `k`.
    TokensGT(u32),
    /// `N % k == 0` with `k != 0`: the fire's token rows are a WHOLE NUMBER
    /// of `k`-row tiles. Wire kind 10, payload `k`.
    ///
    /// # Why a threshold could not stand in for this
    ///
    /// [`Self::TokensGT`] guarded `llama_like`'s projection for as long as
    /// that guard existed, on the reasoning that a GEMM whose tile is `BM`
    /// needs at least `BM` rows. It needs more than that: `qmm_t`'s header
    /// states *"the driver only selects this kernel when `M % BM == 0` … so
    /// every tile is full and the row count lives in the grid"*, and there is
    /// no `M` argument to shorten a tile with. `TokensGT(BM - 1)` therefore
    /// admitted every count above the tile that the tile does not divide —
    /// fifteen in sixteen — to an arm no driver can launch:
    ///
    /// ```text
    /// PartialTile { rows: 35, tile: 16 }
    /// ```
    ///
    /// on a 35-token prompt, on Metal, Vulkan and wgpu alike. This is the
    /// predicate that guard wanted, and it needs no companion: for a non-zero
    /// token count, `N % k == 0` already implies `N >= k`.
    TokensMultipleOf(u32),
    /// The fire's attached programs read attention scores at `OnAttn`
    /// (`StageHooks::wants_attn_score`) — the score-capturing attention
    /// dispatch runs instead of the plain one. Wire kind 3, payload
    /// unused.
    WantsAttnScore,
    /// The fire carries a custom attention mask (`custom_mask_d !=
    /// nullptr`): the masked arm runs the custom-mask prefill dispatch
    /// and, in the fused-decode deployment, the general QKV sequence —
    /// the class-collapse amendment's first predicate (a mask is a
    /// guard, not a class). Wire kind 4, payload unused.
    HasCustomMask,
    /// The fire carries attached stage-hook programs (`stage_hooks !=
    /// nullptr`) — A2 of the class-collapse amendment: the hooked arm
    /// holds the general QKV sequence, the two per-layer HookSites and
    /// the WantsAttnScore-guarded attention. The caller's gate admits
    /// only ALL-hooked fires (fast_rows == 0), so presence ⇔ every row
    /// is hooked; A3's Peel op replaces this all-or-nothing arm with
    /// the fast_rows row split. Wire kind 5, payload unused. RETIRED
    /// vocabulary since A3 (reserved, unstated).
    HasStageHooks,
    /// The fire carries usable lora lanes (`lora != nullptr &&
    /// lora->usable()`) — the §5.1 correction: the adapter delta lands
    /// on the materialized q/v projections before anything consumes
    /// them, and the fused decode-QKV epilogue (which writes V straight
    /// to the paged cache, so there is nothing to correct into) must
    /// not run. Wire kind 6, payload unused.
    HasLora,
    /// EVERY ROW IS A ONE-TOKEN QUERY WINDOW — what `FireClass::Decode`
    /// used to mean, said as a property of the rows instead of as a class.
    ///
    /// This is directive 4.1 of `.wiki/driver/graph.md`. Decode and
    /// Prefill were already ONE body whose only difference was
    /// `let window_one = class == FireClass::Decode`, and the goldens
    /// pinned the collapse as byte-identical; the class survived only
    /// because nothing else could carry the boolean. A guard can, and a
    /// guard is what the masked and hooked classes were already retired
    /// into (A1/A2, the class-collapse amendment).
    ///
    /// A MIXED fire — some rows one token, some many — answers false and
    /// takes the ragged arm, which is correct: a ragged qo window serves a
    /// one-token request as the degenerate case. That is the property
    /// that makes the merge sound rather than merely convenient.
    ///
    /// Wire kind 7, payload unused.
    WindowOne,
}

impl GuardPred {
    /// The ABI (kind, payload) pair.
    pub fn wire(&self) -> (u32, u32) {
        match *self {
            GuardPred::HasWriteDesc => (0, 0),
            GuardPred::TokensLE(k) => (1, k),
            GuardPred::TokensGT(k) => (2, k),
            GuardPred::WantsAttnScore => (3, 0),
            GuardPred::HasCustomMask => (4, 0),
            GuardPred::HasStageHooks => (5, 0),
            GuardPred::HasLora => (6, 0),
            GuardPred::WindowOne => (7, 0),
            // 10 and not 8, because 8 and 9 are `driver-cuda`'s two Peel
            // slots. They were placed "above the GuardPred wire range" when
            // that range ended at 7; a guard added at 8 would have quietly
            // become a Peel in the device predicate word.
            GuardPred::TokensMultipleOf(k) => (10, k),
        }
    }
}

/// A model-body hook site's stage (the HookSite slice,
/// north-star-dsl.md): the TWO points at which a fire's attached PTIR
/// programs observe and intervene inside the forward. PTIR's Prologue
/// and Epilogue are dispatch-side post-logits machinery and never trace
/// ops. Wire values are the ABI (`HookSite.param0`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HookStage {
    /// Before the layer's attention: observes q, intervenes through the
    /// page-mask sink (the driver brackets `begin_layer` → invoke →
    /// compact/pointer-swap; a narrowed page list feeds the SAME stated
    /// attention kernel as substituted arguments).
    OnAttnProj,
    /// After the layer's attention: observes the scores the (possibly
    /// capturing) attention published through the sideband.
    OnAttn,
}

/// One arm of a [`OpKind::Guard`](super::OpKind::Guard) chain: the first arm whose predicate
/// holds runs; `ops` is the arm's flat region length.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct GuardArm {
    pub pred: GuardPred,
    pub ops: u32,
}

/// ONE SEAM STATEMENT the model text made, in text order
/// (`.wiki/tart/dsl.md` ①, migration step 4).
///
/// Three of the five seams lower to ops today (two `HookSite`s and the
/// adapter's `HasLora` guard); the two BOUNDARY seams lower to nothing
/// at all, which is why prologue and epilogue live in a different world
/// from the rest — the traced form does not record that the text has
/// them. This list records every seam the text stated, whichever way it
/// lowered, so "what does this declaration expose?" has one answer.
///
/// `op` is the index of the op carrying the seam when one does. A
/// boundary seam has none: it is a statement about the trace, not a
/// point inside it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SeamStatement {
    pub seam: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub layer: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub op: Option<u32>,
    /// The values this seam EXPOSES — exactly the ones the statement
    /// names, which is what `seam::check_plan` already validates the
    /// arity of.
    ///
    /// Recorded because buffer assignment needs to know which values
    /// machinery outside the walk reaches by name, and inferring it from
    /// the neighbouring op gets the set wrong in BOTH directions: it
    /// takes that op's inputs, so it over-pins the operands a construct
    /// happens to share (`attn.qv` names q and v; the attention op's
    /// inputs are q, k AND v) and it misses any exposed value that is an
    /// OUTPUT — the sampler reads the logit softcap's result, not its
    /// operand.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub values: Vec<ValueId>,
}
