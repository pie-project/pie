//! # Metal (MSL) region emitters
//!
//! A direct port of `driver/metal/src/pipeline/m1_codegen.cpp` (namespace
//! `pie::metal::pipeline`). The C++ original is the oracle: every function
//! here emits the same bytes for the same input, and
//! `compiler/tests/golden-msl/` is the checked-in proof (see
//! `compiler/tests/oracle/README.md`).
//!
//! The port keeps the C++ names and control flow so the two can be read
//! side by side. The differences are deliberate and mechanical:
//!
//! * out-params + `bool` become [`Result<String, String>`];
//! * the `runtime_template` argument is gone — the runtime is embedded with
//!   `include_str!` ([`RUNTIME_TEMPLATE`]) so the emitter is self-contained;
//! * the wire plan (`pie_native::ptir::plan::StagePlan`) becomes the native
//!   [`pie_plan::CompiledStage`], with [`op_view::OpView`] standing in for the
//!   decoded `container::COp` the C++ switches on.
//!
//! Three of the C++ validator's rejections are unreachable from the Rust
//! types and have no counterpart here: an out-of-range symbolic extent role,
//! an out-of-range dtype, and an unknown op tag. `SymbolicExtent`, `DType` and
//! `Op` are closed enums whose variants are exactly the legal values.
//!
//! ## Modules
//!
//! * [`op_view`] — the `COp` projection of a [`pie_ir::op::Op`].
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
pub mod op_view;
pub mod preamble;
pub mod singleton;
pub mod topk;
pub mod validate;

pub use effects::{emit_commit, emit_grouped_commit, emit_grouped_readiness, emit_readiness};
pub use fused::{emit_fused_region, emit_grouped_fused_region};
pub use nucleus::emit_grouped_nucleus;
pub use op_view::OpView;
pub use preamble::RUNTIME_TEMPLATE;
pub use singleton::emit_singleton_region;
pub use topk::emit_grouped_topk;
pub use validate::validate_singleton_plan;

/// `kMetalM1EmitterVersion` — bumped whenever emitted MSL changes, so the
/// driver's pipeline cache keys on it.
pub const METAL_M1_EMITTER_VERSION: u16 = 23;

/// `kMetalM1MaxChannels` — the single-lane readiness/commit kernels bind one
/// `words_N` buffer per channel starting at buffer 2.
pub const METAL_M1_MAX_CHANNELS: usize = 29;

/// `kMetalM2MaxFusedChannels` — a fused region binds committed/pending pairs
/// from buffer 7, which caps the direct-binding form at 12 channels.
pub const METAL_M2_MAX_FUSED_CHANNELS: usize = 12;

/// `M1ChannelEffect` — what one channel needs before a lane may run, and what
/// the lane does to it on commit.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct M1ChannelEffect {
    pub requires_full: bool,
    pub requires_empty: bool,
    pub take: bool,
    pub put: bool,
    pub capacity: u32,
}

/// `M1OpMeta` — one accepted singleton op: where it sits in the stage, the
/// SSA id its first result defines, and the `COp` view the driver dispatches
/// on.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct M1OpMeta {
    pub node: u32,
    pub result_base: u32,
    pub op: OpView,
}
