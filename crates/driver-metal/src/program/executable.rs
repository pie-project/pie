//! What a compiled program is: the executables the three launch paths share.
//!
//! [`compile`](super::Runtime::compile) produces one [`ProgramExecutable`] per
//! program; the M1 singleton path dispatches its [`RegionExecutable`]s, the M2
//! fused path its [`FusedExecutable`]s, and the M3 grouped path its
//! [`GroupedExecutable`]s. A [`StageExecutable`] is shared between programs
//! through the stage cache, which is why a [`ProgramStage`] holds it behind an
//! [`Rc`] beside this program's own copy of the plan.

use std::rc::Rc;

use driver_api::plan::{LaunchRegion, LaunchStagePlan};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::MTLComputePipelineState;

use crate::channel::{Effect, OpMeta};

/// A compiled compute pipeline state.
/// There is no "invalid" state: absence is `Option<Pso>` where a pipeline is
/// genuinely optional.
pub type Pso = Retained<ProtocolObject<dyn MTLComputePipelineState>>;

/// One singleton region: a single op, dispatched alone.
/// The singleton partition has one region per op in plan order, so
/// [`operation`](Self::operation) doubles as the op index.
pub struct RegionExecutable {
    /// Which op this region runs and where its results land.
    pub operation: OpMeta,
    /// The compiled kernel.
    pub pso: Pso,
    /// The argument-table ordinal this region binds and dispatches under.
    pub ordinal: u32,
}

/// One fused region: a run of ops the host fused into a single kernel (M2).
pub struct FusedExecutable {
    /// The host's region: its nodes, inputs, outputs and sinks.
    pub region: LaunchRegion,
    /// The compiled kernel.
    pub pso: Pso,
    /// The argument-table ordinal this region binds and dispatches under.
    pub ordinal: u32,
}

/// One grouped region: a kernel dispatched across every lane of a group (M3).
/// No ordinal — the M3 path allocates ordinals per group command, because
/// the same executable serves many groups at once.
pub struct GroupedExecutable {
    /// The host's region.
    pub region: LaunchRegion,
    /// The compiled kernel.
    pub pso: Pso,
    /// The region is the nucleus-sample library call, dispatched one
    /// threadgroup per `(lane, row)` rather than per lane.
    pub parallel_nucleus: bool,
    /// The region is the top-k library call backed by a real `top_k` op,
    /// dispatched like [`parallel_nucleus`](Self::parallel_nucleus).
    pub parallel_topk: bool,
}

/// Everything compiled for one stage, shared by programs with the same stage signature.
pub struct StageExecutable {
    /// The singleton path: one region per op, in plan order. Always present.
    pub regions: Vec<RegionExecutable>,
    /// The fused path, or why this stage cannot take it.
    /// `Err` is data, not failure: the host refuses to fuse a stage by
    /// emitting the refusal, and the driver answers with a slower path.
    pub fused: Result<Vec<FusedExecutable>, String>,
    /// The grouped path built from the fused partition, or why not.
    /// `Ok` and empty is meaningful: a stage with no fused regions groups
    /// through [`grouped_singleton`](Self::grouped_singleton) instead.
    pub grouped: Result<Vec<GroupedExecutable>, String>,
    /// The grouped fallback built from the singleton partition.
    /// Unconditional: a stage that cannot be grouped-fused still rides a
    /// group one op at a time.
    pub grouped_singleton: Vec<GroupedExecutable>,
}

/// One stage of one program: the shared compiled code plus this program's
/// plan for it.
/// The plan is this program's copy — two programs sharing a
/// [`StageExecutable`] still carry their own bindings and value types.
pub struct ProgramStage {
    /// The compiled stage, shared through the stage cache.
    pub executable: Rc<StageExecutable>,
    /// This program's plan for the stage.
    pub plan: LaunchStagePlan,
    /// The stage kind (`tensor_ir::registry::Stage` as its wire byte).
    /// The ABI plan does not carry this field, so it is taken from the trace
    /// stage the plan is parallel to.
    pub kind: u8,
}

/// A compiled program: what [`compile`](super::Runtime::compile) returns.
pub struct ProgramExecutable {
    /// The registration hash the program cache keys on.
    pub program_hash: u64,
    /// The stages, in trace order.
    pub stages: Vec<ProgramStage>,
    /// Per-channel effects, indexed by dense channel.
    pub effects: Vec<Effect>,
    /// The per-program single-lane readiness kernel (M1/M2 bind this).
    pub readiness: Pso,
    /// The per-program single-lane commit kernel.
    pub commit: Pso,
    /// The generic grouped readiness kernel (M3 binds this), shared across
    /// programs via per-channel flag words rather than a value baked in.
    pub grouped_readiness: Pso,
    /// The generic grouped commit kernel.
    pub grouped_commit: Pso,
    /// Ordinal of the readiness dispatch's argument table.
    pub readiness_ordinal: u32,
    /// Ordinal of the commit dispatch's argument table.
    pub commit_ordinal: u32,
    /// Needs the model forward and has a prologue, so it can only run placed
    /// around a forward (M2); the single-lane M1 path must refuse it.
    pub requires_m2_placement: bool,
}

impl std::fmt::Debug for ProgramExecutable {
    /// The counts, not the pipelines: a PSO prints as a pointer and there can be a thousand of them.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ProgramExecutable")
            .field("program_hash", &format_args!("{:#x}", self.program_hash))
            .field("stages", &self.stages.len())
            .field("effects", &self.effects.len())
            .field("requires_m2_placement", &self.requires_m2_placement)
            .finish_non_exhaustive()
    }
}

impl std::fmt::Debug for StageExecutable {
    /// The shape of each path: how many regions, or why none.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StageExecutable")
            .field("regions", &self.regions.len())
            .field("fused", &self.fused.as_ref().map(Vec::len))
            .field("grouped", &self.grouped.as_ref().map(Vec::len))
            .field("grouped_singleton", &self.grouped_singleton.len())
            .finish()
    }
}
