//! # `pie-plan` — PTIR execution planning
//!
//! The backend-neutral middle end. Given a bound trace ([`pie_ir::validate`]),
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
//! Two wire formats leave this crate:
//!
//! * the **region plan** (`PTRP`, [`encode_stage_plan`]) — per-stage regions,
//!   schedules, and lane records;
//! * the **bound-trace sidecar** (`PTIB`, [`sidecar`]) — the typed lowering that
//!   carries those plans, so a driver never re-infers shapes.

#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

mod compile;
pub mod sidecar;

// Spelled out rather than `pub use compile::*`. The glob made every `pub` item
// anywhere under `compile` part of this crate's API whether or not anything
// called it, so the surface grew silently and nothing ever shrank it.
pub use compile::{
    COMPILER_VERSION, ChannelSink, CompiledStage, Dimension, EncodedPlanHeader,
    LANE_TABLE_ABI_VERSION, LaneChannelSlot, LaneRecord, LaneTableHeader, LibraryOp,
    NormalizedStage, PartitionKind, PlanDecodeError, PlanMetrics, REGION_PLAN_VERSION, Region,
    RegionKind, RegionPartition, RuntimeExtents, ScheduleTemplate, StageSignature, SymbolicExtent,
    SymbolicType, ValueDomain, compile_bound, compile_stage, compile_stage_at, debug_stage_plan,
    decode_plan_header, encode_stage_plan, library_op_for_tag, stage_identity,
};
