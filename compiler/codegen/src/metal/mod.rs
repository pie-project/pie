//! # Metal (MSL) region emitters
//!
//! The only producer of Pie's generated MSL. Emission is a pure function of
//! the plan — no device-architecture inputs — so the same stage emits the same
//! bytes every time, and `compiler/tests/golden-msl/` pins them.
//!
//! Those goldens started as a dump of an in-driver C++ emitter that no longer
//! exists, which is why they are formatted as a foreign dump and why
//! regenerating one is guarded. They are now the contract itself rather than a
//! transcript of one: nothing can re-derive them, so a diff is a decision to be
//! justified, not a comparison to be re-run.
//!
//! Most emitters return [`Result<String, EmitError>`] and refuse rather than
//! emit a kernel they cannot justify. The three that take no plan —
//! [`singleton::emit_singleton_region`], [`effects::emit_grouped_readiness`]
//! and [`effects::emit_grouped_commit`] — return a bare `String`, because their
//! inputs are a name and a closed-enum tag and there is nothing left to refuse.
//!
//! Three refusals the earlier design needed have no counterpart here — an
//! out-of-range symbolic extent role, an out-of-range dtype, and an unknown op
//! tag — because `SymbolicExtent`, `DType` and `Op` are closed enums whose
//! variants are exactly the legal values. A refusal that the types already make
//! unrepresentable is dead code that reads like a live guard.
//!
//! [`EmitError`]: crate::error::EmitError
//!
//! ## Modules
//!
//! * [`preamble`] — the embedded runtime and the shared MSL struct preambles.
//! * [`validate`] — `validate_singleton_plan` and the region ABI checks.
//! * [`singleton`] — the one-op-per-dispatch kernel.
//! * [`effects`] — readiness/commit kernels, single-lane and grouped.
//! * [`fused`] — whole-region kernels, single-lane and grouped.
//! * [`nucleus`] — the grouped nucleus-sampling library kernel.
//! * [`topk`] — the grouped top-k library kernel.

pub mod effects;
pub mod fused;
pub mod nucleus;
pub mod preamble;
pub mod singleton;
pub mod topk;
pub mod validate;

pub use crate::op_view::OpView;
pub use effects::{
    channel_effects, emit_commit, emit_grouped_commit, emit_grouped_readiness, emit_readiness,
};
pub use fused::{emit_fused_region, emit_grouped_fused_region};
pub use nucleus::emit_grouped_nucleus;
pub use preamble::RUNTIME_TEMPLATE;
pub use singleton::emit_singleton_region;
pub use topk::emit_grouped_topk;
pub use validate::validate_singleton_plan;

/// `kMetalM1EmitterVersion` — bumped whenever emitted MSL changes, so the
/// driver's pipeline cache keys on it.
pub const METAL_M1_EMITTER_VERSION: u16 = 36;

/// `kMetalM1MaxChannels` — the single-lane readiness/commit kernels bind one
/// `words_N` buffer per channel starting at buffer 2, and Metal's highest
/// buffer index is 30. Enforced by `emit_readiness` / `emit_commit`; it used to
/// be a comment, and a program with one channel more emitted `[[buffer(31)]]`.
pub const METAL_M1_MAX_CHANNELS: usize = 29;

/// `kMetalM2MaxFusedChannels` — a fused region binds committed/pending pairs
/// from buffer 7, which caps the direct-binding form at 12 channels.
pub const METAL_M2_MAX_FUSED_CHANNELS: usize = 12;

/// `M1ChannelEffect` — what one channel needs before a lane may run, and what
/// the lane does to it on commit.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct M1ChannelEffect {
    /// The lane may run only when this channel's ring is non-empty — a
    /// `take`/`read` precondition.
    pub requires_full: bool,
    /// The lane may run only when this channel's ring has room — a `put`
    /// precondition.
    pub requires_empty: bool,
    /// On commit the lane pops one committed cell from this channel.
    pub take: bool,
    /// On commit the lane pushes one value to this channel.
    pub put: bool,
    /// The channel ring's capacity — its bound on committed cells.
    pub capacity: u32,
}

/// `M1OpMeta` — one accepted singleton op: where it sits in the stage, the
/// SSA id its first result defines, and the `COp` view the driver dispatches
/// on.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct M1OpMeta {
    /// The op's position in the stage op list.
    pub node: u32,
    /// The SSA id this op's first result defines.
    pub result_base: u32,
    /// The decoded [`OpView`] the driver dispatches on.
    pub op: OpView,
}
