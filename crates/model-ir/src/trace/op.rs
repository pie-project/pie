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
    // ── RETIRED WIRE POSITIONS ──────────────────────────────────────────
    //
    // The semantic vocabulary lived here: `Embed`, `Matmul`, `Rmsnorm`, ...
    // — twenty-two ops that said what a statement MEANS and left a driver
    // table to decide what runs. The no-ask contract retires them: a tier-1
    // statement is a [`OpKind::Launch`] whose symbol the DSL resolved
    // through `kernels::canon` at trace time (or spelled as `canon::<role>`
    // in a backend-less description trace). The stubs hold the wire
    // positions — discriminants are ABI, append-only, never reused.
    #[doc(hidden)]
    Retired0,
    #[doc(hidden)]
    Retired1,
    #[doc(hidden)]
    Retired2,
    #[doc(hidden)]
    Retired3,
    #[doc(hidden)]
    Retired4,
    #[doc(hidden)]
    Retired5,
    #[doc(hidden)]
    Retired6,
    #[doc(hidden)]
    Retired7,
    #[doc(hidden)]
    Retired8,
    #[doc(hidden)]
    Retired9,
    /// Gather the sampled rows and project to logits. STRUCTURAL, like
    /// [`OpKind::Select`]: the lowering's readout/epilogue machinery owns it
    /// (`Buffers::assign`'s epilogue gather), so it survives the semantic
    /// retirement — a launch could not say "these rows are the readout".
    LmHead { weight: String },
    #[doc(hidden)]
    Retired11,
    /// `x[index]` along the leading dim. Launches nothing: `Buffers` gives
    /// its value an offset into the source's.
    Select { index: u32 },
    #[doc(hidden)]
    Retired13,
    #[doc(hidden)]
    Retired14,
    #[doc(hidden)]
    Retired15,
    #[doc(hidden)]
    Retired16,
    #[doc(hidden)]
    Retired17,
    #[doc(hidden)]
    Retired18,
    #[doc(hidden)]
    Retired19,
    #[doc(hidden)]
    Retired20,
    #[doc(hidden)]
    Retired21,
    #[doc(hidden)]
    Retired22,
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
        /// Params the LOWERING fills with this launch's own row window:
        /// `(start_slot, len_slot)` receive the rectangle's start and
        /// length. For a launch inside a peel region that is the region's
        /// split — a number no statement can state and no shape can spell,
        /// because the split is the fire's; on an unpeeled fire the window
        /// is `(0, N)`, which is the same reading. The statement carries
        /// zeros at those slots and the walk overwrites them.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        peel_slots: Option<(u8, u8)>,
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
    /// the dispatch addresses, and is re-walked every fire — so it is not a
    /// launch. The driver used to infer it, by looking for the dispatch symbol
    /// in the lowered kernel table; a text that states the preparation says so
    /// instead.
    ///
    /// # It produces ONE VALUE, and it used to produce none
    ///
    /// *"Produces nothing. What it publishes is reached through `PrepKind`"* —
    /// that was this doc, and everything downstream followed from it
    /// mechanically. With no edge from the prep to the statements that execute
    /// what it raised, the edge had to be rebuilt by hand in four places:
    /// `bind::attn_plan` recovers it from a family string and, for decode,
    /// GUESSES from the window on the `LaunchSpec`; `raise_attn_plans` cannot
    /// tell which plan a statement wants so it raises one at the widest head
    /// dim any layer stated; the object's fields scatter into 45 of
    /// `kernels::keys`'s 182 keys, read back at 130 `ask` sites; and the
    /// upload fence is ordered by hand where an edge would order itself.
    ///
    /// So it produces a value. `outputs[0]` is a RAISE — see
    /// [`ValueInfo::raised`] — and the statements that read what this raised
    /// take it as an operand, positionally, the way they take an activation.
    /// `.wiki/designs/design-struct.md` carries the whole of it.
    ///
    /// [`PrepKind`] is still the whole vocabulary and is still closed: a
    /// `Prep` cannot name an arbitrary host program, only one the backend
    /// already knows how to run. Producing a value does not widen that by one
    /// entry.
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
    /// The paged KV cache.
    KvCache,
    /// The GDN conv-window and recurrent-state slabs.
    RecurrentState,
}

impl StateStore {
    /// The `kernels::runtime` name this store answers to — the identity the
    /// driver's resolver is keyed by. The enum is the WIRE form; the name is
    /// the vocabulary's.
    #[must_use]
    pub fn runtime_name(&self) -> &'static str {
        match self {
            StateStore::KvCache => "kv_cache",
            StateStore::RecurrentState => "recurrent_state",
        }
    }
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
            OpKind::Launch { state, .. } => state,
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

impl PrepKind {
    /// The raise this prep publishes, by the word `raise!` wrote.
    ///
    /// Read through [`kernels::raises::Raise`] rather than spelled here: the
    /// key is declared once, in `kernels-cuda/src/raises.rs`, and a literal in
    /// this file would be a second copy able to drift from it. `keys.rs`'s
    /// preamble states the same rule for facts — *"the word appears once in
    /// the tree"* — and this crate can honour it because it already depends on
    /// `kernels-cuda`.
    #[must_use]
    pub const fn key(self) -> &'static str {
        use kernels::raises::Raise;
        match self {
            Self::DecodeAttention { .. } => kernels_cuda::raises::Fa2Decode::KEY,
            Self::PrefillAttention { .. } => kernels_cuda::raises::Fa2Prefill::KEY,
        }
    }
}

/// The type of one SSA value in the trace.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ValueInfo {
    /// Logical shape. **Not this value's when [`Self::raised`] is set** — see
    /// there.
    pub shape: Shape,
    /// Element type. **Not this value's when [`Self::raised`] is set.**
    pub dtype: DType,
    /// Set on values whose content selects per-element structure, such as a
    /// [`OpKind::TopK`] expert assignment.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dyn_axis: Option<DynAxis>,
    /// Set when this value is NOT A TENSOR: one object the fire raised, by the
    /// word its `raise!` declared ([`PrepKind::key`]).
    ///
    /// # Why the other two fields are not `Option`
    ///
    /// Because a raise has neither a rectangle nor an element type, and making
    /// [`Self::shape`] and [`Self::dtype`] optional to say so would rewrite
    /// every reader of both. The invariant is instead: **a reader checks
    /// [`Self::is_raised`] first**, and the one accessor that could silently
    /// hand back a degenerate answer — `TraceBuilder::value_shape` — refuses
    /// rather than returning the empty shape stored here.
    ///
    /// A `String` and not a type parameter because a trace is serialized: the
    /// key is what survives to disk and back, and it is the same string the
    /// binder resolves.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub raised: Option<String>,
}

impl ValueInfo {
    /// One raised object, as a value.
    ///
    /// The shape is empty and the dtype is arbitrary; neither is read for a
    /// raise, and [`Self::is_raised`] is what a reader must consult before
    /// either. See [`Self::raised`].
    #[must_use]
    pub fn raise(key: &str) -> Self {
        Self {
            shape: Shape(Vec::new()),
            dtype: DType::I32,
            dyn_axis: None,
            raised: Some(key.to_string()),
        }
    }

    /// Whether this value is a raise rather than a tensor.
    #[must_use]
    pub fn is_raised(&self) -> bool {
        self.raised.is_some()
    }
}
