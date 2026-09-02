//! ETA execution planning: the backend-neutral middle end. Given a bound
//! trace, normalizes each stage, derives its signature, classifies value
//! domains, partitions the op DAG into regions, and lays out the lane-table
//! ABI. Entry points are infallible since [`eta_ir::validate::bind`] has
//! already settled arity, SSA dominance, value-id range and stage ordering.

mod compile;
pub mod lane_table;

// Spelled out rather than `pub use compile::*`, which made every `pub` item
// under `compile` part of this crate's API whether or not anything used it.
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
