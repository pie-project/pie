//! `#[repr(C)]` mirror of the traced form.
//!
//! These types are the forward crate's published vocabulary: the driver walks
//! them in place, so every field here is part of the ABI. The design copies
//! `loader/src/ffi/types.rs` deliberately — flat POD, read-only, consumed by
//! C++ walking `PieForwardPlan::ops` in execution order — and the same two
//! rules govern this module:
//!
//! * **No owning types.** Every aggregate is a pointer + length pair into an
//!   arena owned by [`crate::ffi::arena::PlanArena`]. Slices stay valid
//!   exactly as long as the `PieForwardPlan` that produced them.
//! * **No `Option`, no niches.** `Option<u32>` is spelled as a sentinel
//!   ([`PIE_FORWARD_NO_LAYER`], [`PIE_FORWARD_NO_NAME`]) so the layout is
//!   legible from C without knowing Rust's niche rules.
//!
//! Where the loader's plan needed a tagged union (`PieLoaderStorageOp`
//! carries per-operation operand structs), the traced form does not: every
//! op fits `kind + layer + weight name + two u32 params + selector +
//! operand ranges`, so [`PieForwardOp`] stays a plain struct and the
//! per-kind meaning of the params is documented on it.

use crate::facts::{NormPlacement, QkNorm};
use crate::trace::{DType, Dim, NormVariant, RopeKind};

/// `PieForwardOp::weight_name` when the op references no weight.
pub const PIE_FORWARD_NO_NAME: u32 = u32::MAX;

/// `PieForwardOp::selector` when the op selects no per-token weights —
/// every op except the expert-indexed `Matmul`s of an MoE trace.
pub const PIE_FORWARD_NO_VALUE: u32 = u32::MAX;

/// `PieForwardOp::layer` for prologue/epilogue ops (embed, final norm,
/// lm_head). Signed so the resting value cannot collide with a real layer.
pub const PIE_FORWARD_NO_LAYER: i32 = -1;

/// Inline dim capacity of [`PieForwardValue`]. The tracer emits rank-2
/// shapes plus the MoE trace's rank-3 route-expanded `[Tokens, k, d]`
/// values; 4 leaves headroom without an arena run per value.
pub const PIE_FORWARD_MAX_DIMS: usize = 4;

/// The op vocabulary, as stable wire values.
///
/// Discriminants are the ABI, not a declaration order: append new kinds
/// after the last value and never renumber, exactly the discipline
/// `loader/src/ffi/types.rs` states on `PieLoaderDType`.
#[repr(u32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PieForwardOpKind {
    Embed = 0,
    Matmul = 1,
    Rmsnorm = 2,
    RmsnormPerHead = 3,
    SplitQkv = 4,
    Rope = 5,
    KvAppend = 6,
    Attention = 7,
    Swiglu = 8,
    LmHead = 9,
    /// `residual += x` (post-norm placement's separate landing). Appended
    /// per the discipline above — the nine kinds before it keep their
    /// wire values.
    ResidualAdd = 10,
    /// Router top-k + softmax + renormalize (one launch in the hand-written
    /// MoE pass). First of the `dyn` kinds — the declared executors do NOT
    /// consume these; their op-kind switches throw on them via the loud
    /// default arm, which is the intended v0 behaviour (the grouped-GEMM
    /// emission is a later, much larger lift).
    TopK = 11,
    /// Per-token combine of the k routed expert outputs.
    WeightedSum = 12,
    /// `out = base + sigmoid(gate) * x` (shared-expert landing).
    SigmoidGateAdd = 13,
    /// Two-way GDN split (`[rows, w0 + w1]` → two results). First of the
    /// GDN kinds (the `pie_forward_trace_qwen3_5_gdn` fragment) — like the
    /// dyn kinds above, the declared executors do NOT consume these; their
    /// op-kind switches throw on them via the loud default arm.
    SplitGdn = 14,
    /// Depthwise causal conv1d + fused SiLU against the layer's implicit
    /// PER-REQUEST conv state.
    CausalConv1d = 15,
    /// Post-conv GDN prep: q/k/v/g/beta from the packed conv output and
    /// the a/b projections plus the a_log/dt_bias parameters (five
    /// results; the one kind that names TWO weights — see the op table).
    GdnPrep = 16,
    /// The gated-delta recurrence against the layer's implicit PER-REQUEST
    /// recurrent state. Opaque like `Attention`.
    GatedDelta = 17,
    /// Per-head gated RMSNorm: `w * rmsnorm(x) * silu(gate)`, plain fold.
    RmsnormGated = 18,
    /// Interleaved per-head `[query | gate]` split of the 2×-wide gated q
    /// projection (qwen3.5 full attention; NOT a row split — the halves
    /// interleave at head granularity). First of the full-attention kinds
    /// (the `pie_forward_trace_qwen3_5_full_attn` fragment / the
    /// `pie_forward_trace_qwen3_5_hybrid` model) — like the dyn and GDN
    /// kinds above, the declared executors do NOT consume these; their
    /// op-kind switches throw on them via the loud default arm.
    SplitQGate = 19,
    /// `out = x * sigmoid(gate)`, elementwise — the full-attention output
    /// gate. A multiply with NO residual: distinct from `SigmoidGateAdd`.
    SigmoidGateMul = 20,
    /// RETIRED (rung 1's per-kernel kind, absorbed into [`Self::Launch`]
    /// within the same unreleased arc). Never emitted; the discriminant
    /// stays reserved per the appended-only rule.
    QkvDecodeFusedPost = 21,
    /// RETIRED — see [`Self::QkvDecodeFusedPost`].
    RopeTableBuild = 22,
    /// A STATED kernel launch — the ONE kind every lowered trace uses for
    /// every kernel its class arms call (north-star-dsl.md; raw kernel
    /// signatures, `dsl::cuda`). The kernel's launcher symbol rides the
    /// weight slot as a name index; the weight names it consumes ride
    /// `aux_names` (name indices, signature order); param0 is the
    /// implicit-state store it addresses (0 none, 1 kv-cache,
    /// 2 recurrent) and param1 that state's layer. A dumb consumer
    /// resolves the symbol in its name→launcher registry and launches —
    /// adding a kernel never grows this enum again.
    Launch = 23,
    /// The lowered branch CHAIN over per-fire RUNTIME inputs: arm count
    /// in param0; the `aux_names` run is [pred kind, pred payload, region
    /// len] per arm plus a trailing else-region length. Regions are flat,
    /// consecutive, in arm order then else; the first arm whose predicate
    /// holds runs. May produce values: the guard's outputs are the ONE
    /// producer whichever region runs — region launches bind the same
    /// output buffer and record no outputs of their own. The ONLY branch
    /// a class trace carries.
    Guard = 24,
    /// A model-body hook site (the HookSite slice): stage wire value in
    /// param0 (0 = OnAttnProj, 1 = OnAttn), layer in param1. The
    /// executor brackets the site's mechanics (page-mask begin/compact,
    /// score sideband) and invokes the fire's attached programs; a fire
    /// with nothing attached passes through by argument.
    HookSite = 25,
    /// Loop peeling (A3, the class-collapse amendment): two regions
    /// that BOTH run over complementary row ranges — prefix `[0,
    /// split)`, tail `[split, N)`. Prefix-region op count in `param0`,
    /// tail-region count in `param1`; the split is a runtime input.
    /// WHICH runtime row count ([`crate::trace::PeelWindow`]) rides
    /// the aux run: EMPTY = the hook-free prefix (`fast_rows`, A3),
    /// `[1]` = the unmasked prefix (the spatial mask split — prefix
    /// region serves the plain decode rows, tail the masked suffix;
    /// UNPLANNED collapses to tail-only full-N fire-level).
    Peel = 26,
    /// Broadcast bias add over `[rows, width]` (Qwen-2 family qkv
    /// biases): weight name in `name`, width from the value's shape.
    AddBias = 27,
}

/// Mirrors [`crate::trace::GuardPred`]'s wire KINDS (each arm crosses as
/// a (kind, payload) pair in the guard's aux run); same appended-only
/// discriminant rule as [`PieForwardOpKind`].
#[repr(u32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PieForwardGuardPred {
    /// The fire carries explicit KV-write descriptors (`has_write_desc`).
    /// Payload unused.
    HasWriteDesc = 0,
    /// `N <= payload` (token rows within a threshold).
    TokensLE = 1,
    /// `N > payload`.
    TokensGT = 2,
    /// The fire's programs read attention scores at OnAttn
    /// (`StageHooks::wants_attn_score`). Payload unused.
    WantsAttnScore = 3,
    /// The fire carries a custom attention mask (`custom_mask_d !=
    /// nullptr`) — A1, the class-collapse amendment. Payload unused.
    HasCustomMask = 4,
    /// The fire carries attached stage-hook programs (`stage_hooks !=
    /// nullptr`) — A2, the class-collapse amendment. Payload unused.
    /// Retired since A3 (reserved, unstated).
    HasStageHooks = 5,
    /// The fire carries usable lora lanes (the §5.1 correction).
    /// Payload unused.
    HasLora = 6,
}

/// Mirrors [`crate::trace::DType`]; same appended-only discriminant rule as
/// [`PieForwardOpKind`].
#[repr(u32)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum PieForwardDType {
    #[default]
    BF16 = 0,
    F32 = 1,
    I32 = 2,
}

impl From<DType> for PieForwardDType {
    fn from(value: DType) -> Self {
        match value {
            DType::BF16 => Self::BF16,
            DType::F32 => Self::F32,
            DType::I32 => Self::I32,
        }
    }
}

/// The tag of one [`PieForwardDim`]: which extent a dim is symbolic in, or
/// that it is a load-time constant. Mirrors [`crate::trace::Dim`].
#[repr(u32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PieForwardDimKind {
    /// The fire's token rows (`N`).
    Tokens = 0,
    /// The fire's request rows (`R`).
    Requests = 1,
    /// A load-time constant; the extent is [`PieForwardDim::value`].
    Const = 2,
}

/// Mirrors [`crate::trace::NormVariant`].
#[repr(u32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PieForwardNormVariant {
    Plain = 0,
    Gemma = 1,
}

impl From<NormVariant> for PieForwardNormVariant {
    fn from(value: NormVariant) -> Self {
        match value {
            NormVariant::Plain => Self::Plain,
            NormVariant::Gemma => Self::Gemma,
        }
    }
}

/// Facts fields cross as `uint32_t` (see `PieForwardLlamaLikeFacts`), so an
/// out-of-range value is a diagnosable request rather than an invalid Rust
/// enum — the same input-side rule `loader/src/ffi/entry.rs` states on
/// `PieLoaderTargetSpec`. These are the inverses.
impl TryFrom<u32> for NormVariant {
    type Error = u32;
    fn try_from(value: u32) -> Result<Self, u32> {
        Ok(match value {
            0 => Self::Plain,
            1 => Self::Gemma,
            other => return Err(other),
        })
    }
}

/// Mirrors [`crate::facts::NormPlacement`].
#[repr(u32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PieForwardNormPlacement {
    Pre = 0,
    /// OLMo-2/3: norm the sub-layer OUTPUT, then a separate residual add.
    Post = 1,
}

impl From<NormPlacement> for PieForwardNormPlacement {
    fn from(value: NormPlacement) -> Self {
        match value {
            NormPlacement::Pre => Self::Pre,
            NormPlacement::Post => Self::Post,
        }
    }
}

impl TryFrom<u32> for NormPlacement {
    type Error = u32;
    fn try_from(value: u32) -> Result<Self, u32> {
        Ok(match value {
            0 => Self::Pre,
            1 => Self::Post,
            other => return Err(other),
        })
    }
}

/// Mirrors [`crate::facts::QkNorm`]. `Off`/`PerHead` keep the wire values
/// the field had as a bool (0/1), so a caller that treated it as "non-zero
/// is per-head qk-norm" still states the same facts.
#[repr(u32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PieForwardQkNorm {
    Off = 0,
    PerHead = 1,
    /// One RMSNorm over the flattened `[heads * head_dim]` q/k projection
    /// (OLMo-2) — different arithmetic from per-head.
    Global = 2,
}

impl From<QkNorm> for PieForwardQkNorm {
    fn from(value: QkNorm) -> Self {
        match value {
            QkNorm::Off => Self::Off,
            QkNorm::PerHead => Self::PerHead,
            QkNorm::Global => Self::Global,
        }
    }
}

impl TryFrom<u32> for QkNorm {
    type Error = u32;
    fn try_from(value: u32) -> Result<Self, u32> {
        Ok(match value {
            0 => Self::Off,
            1 => Self::PerHead,
            2 => Self::Global,
            other => return Err(other),
        })
    }
}

/// Mirrors [`crate::trace::RopeKind`].
#[repr(u32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PieForwardRopeKind {
    Standard = 0,
    /// Llama3/YaRN-style frequency scaling.
    Yarn = 1,
}

impl From<RopeKind> for PieForwardRopeKind {
    fn from(value: RopeKind) -> Self {
        match value {
            RopeKind::Standard => Self::Standard,
            RopeKind::Yarn => Self::Yarn,
        }
    }
}

impl TryFrom<u32> for RopeKind {
    type Error = u32;
    fn try_from(value: u32) -> Result<Self, u32> {
        Ok(match value {
            0 => Self::Standard,
            1 => Self::Yarn,
            other => return Err(other),
        })
    }
}

/// A borrowed byte run. Not NUL-terminated, for the reason
/// `loader/src/ffi/types.rs` gives on `PieLoaderBytes`: the bytes come from
/// Rust `String`s and C++ consumers build a `string_view` from the pair.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct PieForwardBytes {
    pub ptr: *const u8,
    pub len: usize,
}

impl Default for PieForwardBytes {
    fn default() -> Self {
        Self {
            ptr: std::ptr::null(),
            len: 0,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct PieForwardSlice<T> {
    pub ptr: *const T,
    pub len: usize,
}

impl<T> Default for PieForwardSlice<T> {
    fn default() -> Self {
        Self {
            ptr: std::ptr::null(),
            len: 0,
        }
    }
}

pub type PieForwardU32Slice = PieForwardSlice<u32>;

/// One entry of the name table: a substring of
/// [`PieForwardPlan::name_bytes`]. Names are interned, so ops that share a
/// weight (tied embeddings name `embed` twice) share an entry.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PieForwardName {
    pub offset: u32,
    pub len: u32,
}

pub type PieForwardNameSlice = PieForwardSlice<PieForwardName>;

/// One symbolic extent. `value` is meaningful only under
/// [`PieForwardDimKind::Const`] and rests at zero otherwise.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PieForwardDim {
    pub kind: PieForwardDimKind,
    pub value: u32,
}

impl Default for PieForwardDim {
    /// The resting value for dim slots past a value's rank: a zero-extent
    /// constant, which no traced shape produces, so a reader that forgets to
    /// stop at `rank` sees an impossible dim rather than a plausible one.
    fn default() -> Self {
        Self {
            kind: PieForwardDimKind::Const,
            value: 0,
        }
    }
}

impl From<Dim> for PieForwardDim {
    fn from(value: Dim) -> Self {
        match value {
            Dim::Tokens => Self {
                kind: PieForwardDimKind::Tokens,
                value: 0,
            },
            Dim::Requests => Self {
                kind: PieForwardDimKind::Requests,
                value: 0,
            },
            Dim::Const(extent) => Self {
                kind: PieForwardDimKind::Const,
                value: extent,
            },
        }
    }
}

/// One SSA value of the traced form. Dims are inline rather than an arena
/// run because rank is tiny and bounded ([`PIE_FORWARD_MAX_DIMS`]); slots at
/// `rank` and beyond hold [`PieForwardDim::default`].
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct PieForwardValue {
    pub rank: u32,
    pub dims: [PieForwardDim; PIE_FORWARD_MAX_DIMS],
    pub dtype: PieForwardDType,
}

pub type PieForwardValueSlice = PieForwardSlice<PieForwardValue>;

/// A run of value ids inside [`PieForwardPlan::value_ids`].
///
/// An {offset, len} pair rather than a pointer slice so an op stays a plain
/// record over one flat array — the driver walks
/// `plan.value_ids.ptr[range.offset .. range.offset + range.len]`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PieForwardIdRange {
    pub offset: u32,
    pub len: u32,
}

/// One operation of the traced form, flattened.
///
/// The two `param` fields carry what the corresponding
/// [`crate::trace::OpKind`] variant carries; everything a kind does not use
/// rests at zero:
///
/// | `kind`           | `weight_name`        | `param0`                     | `param1`   |
/// |------------------|----------------------|------------------------------|------------|
/// | `Embed`          | embedding table      | —                            | —          |
/// | `Matmul`         | weight               | `beta_one` (0/1)             | —          |
/// | `Rmsnorm`        | weight               | [`PieForwardNormVariant`]    | —          |
/// | `RmsnormPerHead` | weight               | `head_dim`                   | [`PieForwardNormVariant`] |
/// | `SplitQkv`       | none                 | `q_width`                    | `kv_width` |
/// | `Rope`           | none                 | [`PieForwardRopeKind`]       | partial rotary width (0 = full) |
/// | `KvAppend`       | none                 | cache layer                  | —          |
/// | `Attention`      | none                 | cache layer                  | —          |
/// | `Swiglu`         | none                 | `inter`                      | —          |
/// | `LmHead`         | weight               | —                            | —          |
/// | `ResidualAdd`    | none                 | —                            | —          |
/// | `TopK`           | none                 | `k`                          | —          |
/// | `WeightedSum`    | none                 | `k`                          | —          |
/// | `SigmoidGateAdd` | none                 | —                            | —          |
/// | `SplitGdn`       | none                 | `width0`                     | `width1`   |
/// | `CausalConv1d`   | conv (weight + bias) | state layer                  | `kernel`   |
/// | `GdnPrep`        | a_log                | dt_bias NAME index           | —          |
/// | `GatedDelta`     | none                 | state layer                  | —          |
/// | `RmsnormGated`   | weight               | —                            | —          |
/// | `SplitQGate`     | none                 | `heads`                      | `head_dim` |
/// | `SigmoidGateMul` | none                 | —                            | —          |
/// | `Launch`         | KERNEL symbol        | state store (0/1/2)          | state layer |
///
/// `Launch` additionally carries its consumed weight names in
/// `aux_names` — see the field.
///
/// `RmsnormPerHead`'s param1 and `Rope`'s param1 are serde-additive on the
/// Rust side (default `Plain` / absent) and appended-param-additive here:
/// both rest at 0 on every trace that predates them, so a pre-qwen3.5
/// consumer reading only param0 still reads what it always did. A partial
/// `Rope` (param1 != 0) rotates only the first param1 channels of each
/// head (`launch_rope_partial_bf16`'s `rotary_dim`).
///
/// `KvAppend`/`Attention` restate the layer their kind addresses even though
/// `layer` carries the bracketing layer, because the trace states both
/// separately and the flattening does not get to decide they coincide; the
/// GDN state ops (`CausalConv1d`, `GatedDelta`) restate theirs for the same
/// reason — param0 is the layer of the implicit PER-REQUEST conv/recurrent
/// slab the op reads and advances (the trace crate's `OpKind::state_ref`
/// marking, pie-application-plan.md §5.4). `GdnPrep` is the one kind whose
/// launch reads two parameter tensors, so its param0 is a SECOND
/// [`PieForwardPlan::names`] index (the dt_bias name), not a width.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct PieForwardOp {
    pub kind: PieForwardOpKind,
    /// The layer this op belongs to, or [`PIE_FORWARD_NO_LAYER`] for
    /// prologue/epilogue — so the driver can bracket its layer loop without
    /// re-deriving structure from names.
    pub layer: i32,
    /// Index into [`PieForwardPlan::names`], or [`PIE_FORWARD_NO_NAME`].
    pub weight_name: u32,
    pub param0: u32,
    pub param1: u32,
    /// The per-token selector: for an expert-indexed `Matmul` (whose
    /// `weight_name` is then a template, `layer.0.expert.{e}.gate_up`) the
    /// value id of the `TopK` index output that resolves `{e}` per token;
    /// [`PIE_FORWARD_NO_VALUE`] for every other op. The selector is also
    /// the op's last input, so a dataflow walk needs no special case; this
    /// field states which input selects rather than flows. The dyn marker
    /// crosses the ABI only here and as the producing `TopK` op — a
    /// per-value flag would duplicate what these two already state.
    pub selector: u32,
    /// `Launch` only: the weight names the stated kernel consumes, as a
    /// range of NAME indices in the flat id array (the same array the
    /// operand ranges index — ids are just u32s; what a range means is
    /// the field's contract). Empty for every other kind.
    pub aux_names: PieForwardIdRange,
    /// Values consumed, in operand order.
    /// The op's role under the DEPTH axis ([`crate::trace::DepthRole`]
    /// as wire values: 0 = none, 1 = windowed, 2 = prefix-plan-swap).
    /// Appended per the ABI discipline; pre-role consumers read 0.
    pub depth_role: u32,
    pub inputs: PieForwardIdRange,
    /// Values produced (`SplitQkv` produces three, `KvAppend` none).
    pub outputs: PieForwardIdRange,
}

pub type PieForwardOpSlice = PieForwardSlice<PieForwardOp>;

/// The traced form of one family's forward pass, as the driver sees it.
///
/// Read-only by construction: the driver walks `ops` in order, resolves
/// weight names against its own tensor store, and never writes back.
///
/// `owner` is the opaque handle to the arena keeping every slice alive. It
/// is consumed by `pie_forward_release`; the driver must not dereference it.
#[repr(C)]
#[derive(Debug)]
pub struct PieForwardPlan {
    /// The family that traced this, as a name-table index — a cache key, and
    /// the first thing a mismatch report prints.
    pub family: u32,
    pub values: PieForwardValueSlice,
    pub ops: PieForwardOpSlice,
    /// The flat operand array every [`PieForwardIdRange`] indexes; ids index
    /// `values`.
    pub value_ids: PieForwardU32Slice,
    /// The name table; entries substring `name_bytes`.
    pub names: PieForwardNameSlice,
    /// The UTF-8 blob behind the name table.
    pub name_bytes: PieForwardBytes,
    /// The tracer's content hash ([`crate::ffi::compiler_version`]), so two
    /// plans compare as stale-vs-fresh without re-tracing.
    pub compiler_version: u64,
    /// STRUCTURAL S-3: non-zero when the declaration states the depth
    /// axis ([`crate::trace::ForwardPlan::depth_window`]) — layer-tagged
    /// ops may run over the full-depth prefix window (or be skipped on a
    /// uniform truncated fire), keyed on each op's own layer tag.
    pub depth_window: u8,
    pub owner: *mut std::ffi::c_void,
}

impl Default for PieForwardPlan {
    /// The empty plan: every slice null, `owner` null. This is what
    /// `pie_forward_trace_llama_like` writes before it can fail and what
    /// `pie_forward_release` leaves behind, so a released or never-filled
    /// header reads as empty instead of dangling.
    fn default() -> Self {
        Self {
            family: PIE_FORWARD_NO_NAME,
            values: PieForwardValueSlice::default(),
            ops: PieForwardOpSlice::default(),
            value_ids: PieForwardU32Slice::default(),
            names: PieForwardNameSlice::default(),
            name_bytes: PieForwardBytes::default(),
            compiler_version: 0,
            depth_window: 0,
            owner: std::ptr::null_mut(),
        }
    }
}

// ── The lowering, for the shadow comparison ────────────────────────────

/// One row of a fire as the engine's seriation ordered them — the input
/// side of [`crate::lower::lower`], as C states it.
///
/// Flags rather than a bitfield because the driver fills this per row per
/// fire and a named field is what keeps a filler honest; `depth_k` is
/// negative for a full-depth row, which is the same "no truncation"
/// spelling the wire already uses elsewhere.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct PieForwardRow {
    pub multi_token: u8,
    pub custom_mask: u8,
    pub hooked: u8,
    pub lora: u8,
    pub write_desc: u8,
    pub wants_scores: u8,
    /// This row's logits are read (the fire's sampled set).
    pub samples: u8,
    pub _pad: u8,
    /// Truncated at this layer, or negative for full depth.
    pub depth_k: i32,
}

/// One rectangle of the flat launch list.
///
/// `kernel_name` indexes a table handed back beside the launches, NOT the
/// plan's name table: a lowering names launcher SYMBOLS, and the plan's
/// names are weights.
///
/// `rows` is read in the op's own row space — `Dim::Tokens` for the body,
/// `Dim::Requests` for the epilogue (see [`crate::lower::Launch`]).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PieForwardLaunch {
    /// The statement this rectangle came from — an index into the plan's
    /// ops, and what a shadow comparison keys on.
    pub at_op: u32,
    pub kernel_name: u32,
    pub row_lo: u32,
    pub row_hi: u32,
    pub layer_lo: u16,
    pub layer_hi: u16,
    /// Which row partition this rectangle sits in: 0 = none,
    /// 1 = the hook-free prefix's axis, 2 = the unmasked prefix's.
    pub peel_axis: u8,
    /// Non-zero for the SUFFIX region rather than the prefix — the
    /// executor's mask region, and what decides whether a statement
    /// addresses rows at absolute offsets.
    pub peel_tail: u8,
    /// Non-zero when `row_lo/row_hi` are the HOST's belief and the
    /// executing form must read the fire's runtime split instead. Set
    /// only inside a CAPTURED fire's peel — the one place a rectangle
    /// is not a pair of numbers.
    pub rows_device: u8,
    pub _pad: u8,
}

/// What [`crate::lower::Uncovered`] crosses as: zero is a lowering, and
/// every other value is a group that should not have been formed.
#[repr(u32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PieForwardUncovered {
    None = 0,
    Rows = 1,
    WholeKernelSplit = 2,
    Discontiguous = 3,
    UnknownBackend = 4,
}

/// The flat launch list for one fire, pointing into storage the plan owns
/// until the next [`crate::ffi::pie_forward_lower`] on the same plan.
#[repr(C)]
#[derive(Debug)]
pub struct PieForwardLowered {
    pub launches: *const PieForwardLaunch,
    pub launches_len: usize,
    /// The distinct launcher symbols, in first-launch order; entries
    /// substring `kernel_name_bytes`.
    pub kernel_names: *const PieForwardName,
    pub kernel_names_len: usize,
    pub kernel_name_bytes: PieForwardBytes,
    /// Peak activation bytes the frame would need.
    pub arena_bytes: usize,
    /// Non-zero when the fire could not be lowered; `launches` is then
    /// empty and the value says which rule refused.
    pub uncovered: PieForwardUncovered,
}

impl Default for PieForwardLowered {
    fn default() -> Self {
        Self {
            launches: std::ptr::null(),
            launches_len: 0,
            kernel_names: std::ptr::null(),
            kernel_names_len: 0,
            kernel_name_bytes: PieForwardBytes::default(),
            arena_bytes: 0,
            uncovered: PieForwardUncovered::None,
        }
    }
}
