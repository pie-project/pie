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
/// | `RmsnormPerHead` | weight               | `head_dim`                   | —          |
/// | `SplitQkv`       | none                 | `q_width`                    | `kv_width` |
/// | `Rope`           | none                 | [`PieForwardRopeKind`]       | —          |
/// | `KvAppend`       | none                 | cache layer                  | —          |
/// | `Attention`      | none                 | cache layer                  | —          |
/// | `Swiglu`         | none                 | `inter`                      | —          |
/// | `LmHead`         | weight               | —                            | —          |
/// | `ResidualAdd`    | none                 | —                            | —          |
/// | `TopK`           | none                 | `k`                          | —          |
/// | `WeightedSum`    | none                 | `k`                          | —          |
/// | `SigmoidGateAdd` | none                 | —                            | —          |
///
/// `KvAppend`/`Attention` restate the layer their kind addresses even though
/// `layer` carries the bracketing layer, because the trace states both
/// separately and the flattening does not get to decide they coincide.
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
    /// Values consumed, in operand order.
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
            owner: std::ptr::null_mut(),
        }
    }
}
