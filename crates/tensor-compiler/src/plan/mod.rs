//! # `tensor-compiler` — PTIR execution planning
//!
//! The backend-neutral middle end. Given a bound trace ([`tensor_ir::validate`]),
//! this crate decides **how** a program executes: it normalizes each stage,
//! derives its canonical signature, classifies value domains, partitions the op
//! DAG into generated / library / second-party regions, and lays out the
//! lane-table ABI. Backends consume the serialized result and supply only code
//! generation and library implementations.
//!
//! "Plan" here is the cuDNN/FFTW sense — a reusable, shape-parameterized
//! execution strategy keyed by a [`StageSignature`] — **not** an LLVM-style
//! optimization pass pipeline. Runtime-varying extents stay symbolic
//! ([`SymbolicExtent`]) so one plan serves many batch shapes.
//!
//! Nothing is serialized on the way out. A [`CompiledStage`] is handed to
//! `tensor-compiler` as a Rust value and reaches a driver as generated source plus
//! the launch package's typed records ([`LaneRecord`]); the driver is told what
//! to run rather than given bytes to parse. [`debug_stage_plan`] renders a plan
//! for humans, and [`stage_identity`] hashes one for cache keys, but neither is
//! a format anything decodes.
//!
//! ## Why this crate returns no `Result`
//!
//! Every entry point here is infallible — [`compile_bound`] returns
//! `Vec<CompiledStage>`, not `Result`. That is a precondition, not an
//! oversight: the input is a [`BoundTrace`](tensor_ir::validate::BoundTrace), and producing one requires
//! [`tensor_ir::validate::bind`], which is where a malformed guest container is
//! rejected. Arity, SSA dominance, value-id range, channel-slot validity,
//! stage ordering and readiness are all settled before a plan is asked for, so
//! there is no untrusted input left for this crate to refuse.
//!
//! What remains are *invariants*: places where the code relies on something
//! `bind` guarantees. Those are written as `expect` with the invariant named,
//! never as a silent fallback — a wrong default here becomes a wrong plan,
//! which becomes a kernel computing the wrong answer, and that is far worse
//! than a panic naming the broken assumption. Each such site cites the check
//! that establishes it, so the claim can be audited rather than trusted. The
//! full list is short by design; if it grows, that is a signal that something
//! untrusted has reached this crate and the entry points need to change shape.
//!
//! The backends are the opposite case and are fallible for the opposite
//! reason: `tensor-compiler` refuses plans it cannot emit (`EmitError`) because
//! "this backend has no lowering for that" is a normal outcome, not a bug.

mod compile;
pub mod lane_table;

// Spelled out rather than `pub use compile::*`. The glob made every `pub` item
// anywhere under `compile` part of this crate's API whether or not anything
// called it, so the surface grew silently and nothing ever shrank it.
pub use compile::{
    COMPILER_VERSION, ChannelSink, ChannelSlot, CompiledStage, Dimension, LibraryOp, NodeIndex,
    NormalizedStage, PartitionKind, PlanMetrics, REGION_PLAN_VERSION, Region, RegionKind,
    RegionPartition, ScheduleTemplate, StageSignature, SymbolicExtent, SymbolicType, ValueDomain,
    compile_bound, compile_stage, compile_stage_at, debug_stage_plan, library_op_for_tag,
    stage_identity,
};
pub use lane_table::{
    LANE_TABLE_ABI_VERSION, LaneChannelSlot, LaneRecord, LaneTableHeader, RuntimeExtents,
};
