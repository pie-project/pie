//! The op vocabulary — every kind of thing a traced form can say.
//!
//! An op names a weight by string and a layer by index, so a driver brackets
//! its layer loop without re-deriving structure from names. Row counts that
//! vary per fire are runtime inputs, never trace values; per-request state is
//! reached through [`StateRef`], not by matching a name.

use super::*;
use serde::{Deserialize, Serialize};

/// One operation of the traced form.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum OpKind {
    /// Token ids -> hidden rows, via the embedding table.
    Embed { weight: String },
    /// `out = act @ weight^T (+ beta * out)`; `beta_one` folds the residual
    /// accumulate into cuBLAS. With `selector` set, `weight` is a template
    /// whose `{e}` a per-token `[Tokens, k]` expert assignment resolves, and
    /// the selector is the op's last input.
    Matmul {
        weight: String,
        beta_one: bool,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        selector: Option<ValueId>,
    },
    /// Row RMSNorm over the trailing dim.
    Rmsnorm {
        weight: String,
        variant: NormVariant,
    },
    /// `x[r, :] += bias` over `[rows, width]`. Sits after the lora
    /// correction and before norms/rope.
    AddBias { weight: String },
    /// Per-head RMSNorm of packed `[rows, heads * head_dim]` Q or K.
    /// `variant` folds the weight as on [`OpKind::Rmsnorm`]: `Plain`
    /// multiplies `w`, `Gemma` folds `(1 + w)`.
    RmsnormPerHead {
        weight: String,
        head_dim: u32,
        #[serde(default, skip_serializing_if = "NormVariant::is_plain")]
        variant: NormVariant,
    },
    /// Split packed QKV `[rows, q + 2kv]` into Q, K, V (three results).
    SplitQkv { q_width: u32, kv_width: u32 },
    /// Rotary embedding applied in place to Q and K (two operands).
    /// `partial` rotates only the leading `rotary_dim` channels of each head,
    /// as a resolved channel count — not HF's `partial_rotary_factor`.
    Rope {
        kind: RopeKind,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        partial: Option<u32>,
    },
    /// Append this fire's K/V rows to the layer's paged cache.
    KvAppend { layer: u32 },
    /// Paged attention over the layer's cache. Opaque in a semantic trace; a
    /// lowered one states its kernel as an [`OpKind::Launch`].
    Attention { layer: u32 },
    /// SwiGLU over packed `[rows, 2 * inter]` gate‖up.
    Swiglu { inter: u32 },
    /// Gather the sampled rows and project to logits.
    LmHead { weight: String },
    /// `residual += x`. A separate op because the norm between the projection
    /// GEMM and the add makes `beta=1` impossible.
    ResidualAdd,
    /// `x[index]` along the leading dim. Launches nothing: `Buffers` gives
    /// its value an offset into the source's.
    Select { index: u32 },
    /// Router top-k over per-token logits, softmaxed and renormalized.
    /// Results are expert indices (`[Tokens, k]` i32, [`DynAxis::PerToken`])
    /// then weights, in that order.
    TopK { k: u32 },
    /// Per-token combine of the k routed expert outputs:
    /// `out[t] = sum_j w[t, j] * x[t, j, :]`, `[Tokens, k, d]` → `[Tokens, d]`.
    WeightedSum { k: u32 },
    /// `out = base + sigmoid(gate) * x`, the scalar per-token gate broadcast
    /// over the hidden dim. Operands are `[x, gate, base]`: fresh value
    /// first, the stream it lands on last.
    SigmoidGateAdd,
    /// Split a packed `[rows, w0 + w1]` value at `w0` into two, for the GDN
    /// in-projection. Two-way, unlike [`OpKind::SplitQkv`].
    SplitGdn { width0: u32, width1: u32 },
    /// Depthwise causal conv1d over the packed `[rows, conv_dim]` qkv with
    /// fused SiLU. `weight` names the conv binding, under which the driver
    /// binds both the conv weight and its bias; `kernel` is the window width;
    /// `layer` picks the per-request conv-state slab it advances.
    CausalConv1d {
        weight: String,
        bias: Option<String>,
        layer: u32,
        kernel: u32,
    },
    /// The post-conv GDN prep: unpacks `[q_raw|k_raw|v_raw]`, L2-normalizes
    /// q/k into compact per-head fp32, converts v to fp32, and folds `a`/`b`
    /// with `a_log`/`dt_bias` into log-decay `g` and mixing `beta`. The GQA
    /// `repeat_interleave` of q/k is no op here: the recurrence kernels index
    /// the compact layout directly.
    GdnPrep { a_log: String, dt_bias: String },
    /// Folds this fire's tokens into the layer's per-request recurrent state,
    /// producing `[Tokens, Vh, Vd]` f32 from operands `[q, k, v, g, beta]`.
    GatedDelta { layer: u32 },
    /// Per (row, head), `out = w * rmsnorm(x) * silu(gate)` over the trailing
    /// head dim of the rank-3 f32 core output, flattened to the gate's bf16
    /// shape. Not a [`NormVariant`], since gating adds an operand.
    RmsnormGated { weight: String },
    /// The interleaved per-head `[query | gate]` split of qwen3.5's 2×-wide
    /// gated q projection: `q[n, h*d + i] = packed[n, h*2d + i]`,
    /// `gate[n, h*d + i] = packed[n, h*2d + d + i]`. Not a row split.
    SplitQGate { heads: u32, head_dim: u32 },
    /// `out = x * sigmoid(gate)`, qwen3.5's output gate before o_proj. No
    /// residual and no landing, unlike [`OpKind::SigmoidGateAdd`].
    SigmoidGateMul,
    /// A stated kernel launch, which only a lowered trace carries. `kernel`
    /// is the driver's launcher symbol, resolved through a name→launcher
    /// registry so the ABI stops growing per kernel.
    Launch {
        kernel: String,
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        weights: Vec<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        state: Option<StateRef>,
        /// Scalar arguments no operand shape gives — a rotary width, a
        /// padded head dim. A number belongs here only when it is a property
        /// of this statement that no shape spells; `eps`, `rope_theta` and
        /// vocab size stay the arm's parameters.
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        params: Vec<u32>,
        /// Params the fire decides: `(index, Shape)` means the scalar at
        /// `params[index]` is that shape's element count for the fire being
        /// lowered, not the constant sitting there.
        ///
        /// A number a shape does spell must go here rather than be frozen
        /// into `params` — a deployment-time count that outruns the fire's
        /// reads past the live data silently.
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        param_extents: Vec<(u8, Shape)>,
    },
    /// The one branch a lowered trace may carry: a chain of arms over
    /// per-fire runtime inputs ([`GuardPred`], a closed vocabulary). The first
    /// arm whose predicate holds runs, and the trailing `else_ops` region when
    /// none does. Regions are flat and consecutive, never nested. A guard may
    /// produce values, in which case every region binds one output buffer and
    /// records no SSA output of its own.
    Guard { arms: Vec<GuardArm>, else_ops: u32 },
    /// Where the fire's attached PTIR programs run against this layer.
    /// Produces nothing: interventions travel through sidebands and are
    /// argument-driven, so an unattached site is a no-op by argument rather
    /// than by branch.
    HookSite { stage: HookStage, layer: u32 },
    /// Fire-time HOST work whose result the launches below read by name.
    ///
    /// The attention schedulers are the case it exists for: a FlashInfer plan
    /// is walked on the CPU from THIS batch's page CSR, lands in a workspace
    /// the dispatch addresses, and is re-walked every fire — so it is neither
    /// a launch nor a value. The driver used to infer it, by looking for the
    /// dispatch symbol in the lowered kernel table; a text that states the
    /// preparation says so instead.
    ///
    /// Produces nothing. What it publishes is reached through [`PrepKind`],
    /// which is the whole vocabulary: a `Prep` cannot name an arbitrary host
    /// program, only one the backend already knows how to run.
    Prep { prep: PrepKind },
    /// Two regions that both run, over complementary row ranges: prefix over
    /// `[0, fast_rows)`, tail over `[fast_rows, N)`. `fast_rows` is a runtime
    /// input of the fire, never a trace value, and `window` names which row
    /// count splits them. An empty range skips its region's launches.
    Peel {
        prefix_ops: u32,
        tail_ops: u32,
        #[serde(default, skip_serializing_if = "PeelWindow::is_hook_free")]
        window: PeelWindow,
    },
}

/// Which runtime row count a [`OpKind::Peel`] splits on. Both regions run
/// over complementary windows whichever it is.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PeelWindow {
    /// `fast_rows`: the fused fast path, with the hook-visible general
    /// sequence in the tail.
    #[default]
    HookFreePrefix,
    /// The spatial mask split: plain decode rows in the prefix, masked suffix
    /// rows in the tail. `unmasked_prefix_rows == u32::MAX` means unplanned,
    /// and the tail runs full-N.
    UnmaskedPrefix,
}

impl PeelWindow {
    /// Whether this window is the hook-free fast path.
    pub fn is_hook_free(&self) -> bool {
        matches!(self, PeelWindow::HookFreePrefix)
    }
}

/// Which implicit store an op addresses. Both are per-layer and per-request,
/// but the KV cache grows and is page-table-indirected while the recurrent
/// store is fixed slabs advanced in place — which is why RS fires run solo.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum StateStore {
    /// The paged KV cache (`KvAppend` writes, `Attention` reads).
    KvCache,
    /// The GDN conv-window and recurrent-state slabs; `CausalConv1d` and
    /// `GatedDelta` each read and advance their half.
    RecurrentState,
}

/// The state an op addresses, derived by [`OpKind::state_ref`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct StateRef {
    /// Which implicit store.
    pub store: StateStore,
    /// Which layer's instance of it.
    pub layer: u32,
}

impl OpKind {
    /// How the planner learns a trace touches state without name-matching.
    pub fn state_ref(&self) -> Option<StateRef> {
        match *self {
            OpKind::KvAppend { layer } | OpKind::Attention { layer, .. } => Some(StateRef {
                store: StateStore::KvCache,
                layer,
            }),
            OpKind::Launch { state, .. } => state,
            OpKind::CausalConv1d { layer, .. } | OpKind::GatedDelta { layer } => Some(StateRef {
                store: StateStore::RecurrentState,
                layer,
            }),
            _ => None,
        }
    }
}

/// One op of the traced form: a kind, its operands and its results.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Op {
    /// What the op does.
    pub kind: OpKind,
    /// Operand values, in the kind's declared order.
    pub inputs: Vec<ValueId>,
    /// Result values, in the kind's declared order.
    pub outputs: Vec<ValueId>,
    /// The layer this op belongs to, or `None` for prologue/epilogue.
    pub layer: Option<u32>,
    /// Where this op's write lands when it is not the value's producer.
    ///
    /// [`Self::outputs`] says which values an op produces. A launch inside a
    /// guard's value region produces none — the guard mints the value up
    /// front, so dataflow sees one producer whichever arm runs — yet the
    /// kernel still writes, into the region's buffer. This names that buffer.
    ///
    /// Empty on every op that publishes what it writes.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub dest: Vec<ValueId>,
}

/// What a [`OpKind::Prep`] asks the backend to raise.
///
/// A closed vocabulary, deliberately. An open one would make `Prep` a hook for
/// running host code from a model text, which is the thing the trace exists
/// not to be.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PrepKind {
    /// The paged-decode attention schedule, over this fire's KV page CSR.
    ///
    /// `head_dim` is stated because the planner BAKES it in, so a stack whose
    /// layers disagree needs one schedule per width. `full_attention` is the
    /// window read as a claim: a layer attending the whole context states
    /// `true`, and it is what picks between the two when a stack has two.
    ///
    /// Stated per attention statement rather than once per fire, and that is
    /// not redundancy: a union trace carries both fire classes as guard arms,
    /// so which schedules a FIRE needs is the set its arms state, deduplicated.
    DecodeAttention { head_dim: u32, full_attention: bool },
    /// The paged-prefill schedule.
    ///
    /// A text whose prefill plans INSIDE the fire states none — that is what
    /// the planless prefill is, and its schedule is walked from the host CSR
    /// mirrors when the statement runs.
    PrefillAttention { head_dim: u32 },
}

/// The type of one SSA value in the trace.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ValueInfo {
    /// Logical shape.
    pub shape: Shape,
    /// Element type.
    pub dtype: DType,
    /// Set on values whose content selects per-element structure, such as a
    /// [`OpKind::TopK`] expert assignment.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dyn_axis: Option<DynAxis>,
}
