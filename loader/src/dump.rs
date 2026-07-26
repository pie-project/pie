use std::collections::BTreeMap;
use std::fmt::Write as _;

use serde::Serialize;

use crate::error::CompileError;
use crate::ir::LayoutPlan;
use crate::load_plan::{LoadPlan, StorageInstr, TileMapKind};
use crate::verify::ContractCoverage;

/// A combined debug dump of both compiler stages: the optimized
/// [`LayoutPlan`] and the [`LoadPlan`] it lowered to.
#[derive(Clone, Debug, Serialize)]
pub struct CompilerDump<'a> {
    /// `LOAD_PLAN_VERSION` — the plan *format* version, not the compiler hash
    /// (the compiler hash travels inside `load_plan.compiler_version`).
    pub plan_version: u32,
    pub layout: &'a LayoutPlan,
    pub load_plan: &'a LoadPlan,
}

pub fn dump_load_plan_json(plan: &LoadPlan) -> Result<String, CompileError> {
    serde_json::to_string_pretty(plan)
        .map_err(|err| CompileError::Internal(format!("load-plan dump failed: {err}")))
}

pub fn dump_compiler_json(
    layout: &LayoutPlan,
    load_plan: &LoadPlan,
) -> Result<String, CompileError> {
    serde_json::to_string_pretty(&CompilerDump {
        plan_version: crate::load_plan::LOAD_PLAN_VERSION,
        layout,
        load_plan,
    })
    .map_err(|err| CompileError::Internal(format!("compiler dump failed: {err}")))
}

/// The plan's own name for an instruction.
///
/// A driver that renders a plan needs these names, and the table it used to keep
/// was a `switch` with a `"Unknown"` fallthrough: adding a variant here left that
/// table quietly wrong. Matching exhaustively on the enum makes the same change a
/// compile error instead.
fn instr_name(instr: &StorageInstr) -> &'static str {
    match instr {
        StorageInstr::Allocate { .. } => "Allocate",
        StorageInstr::ExtentWrite { .. } => "ExtentWrite",
        StorageInstr::BulkExtentWrite { .. } => "BulkExtentWrite",
        StorageInstr::SlabScatter { .. } => "SlabScatter",
        StorageInstr::TileMap { .. } => "TileMap",
        StorageInstr::CreateView { .. } => "CreateView",
        StorageInstr::Attach { .. } => "Attach",
        StorageInstr::Release { .. } => "Release",
        StorageInstr::Finalize { .. } => "Finalize",
    }
}

/// The plan's own name for a tile transform. See [`instr_name`].
fn tile_map_name(kind: TileMapKind) -> &'static str {
    match kind {
        TileMapKind::Cast => "Cast",
        TileMapKind::Decode => "Decode",
        TileMapKind::Encode => "Encode",
        TileMapKind::Transcode => "Transcode",
        TileMapKind::Reblock => "Reblock",
        TileMapKind::Reorder => "Reorder",
        TileMapKind::Repack => "Repack",
    }
}

/// One line describing a compiled plan, for the driver's boot log.
pub fn describe(plan: &LoadPlan, coverage: ContractCoverage) -> String {
    let rewrites: usize = plan.optimizer.passes.iter().map(|pass| pass.rewrites).sum();
    let mut out = String::new();
    let _ = write!(
        out,
        "load_plan(version={}, source_tensors={}, contracts={}/{}, tensors={}, \
         buffers={}, instrs={}, schedule={}, optimizer_passes={}, \
         optimizer_rewrites={}, persistent_bytes={}, read_bytes={}, write_bytes={})",
        plan.version,
        plan.sources.len(),
        coverage.covered,
        coverage.demanded,
        plan.tensors.len(),
        plan.buffers.len(),
        plan.instrs.len(),
        plan.schedule.len(),
        plan.optimizer.passes.len(),
        rewrites,
        plan.memory.persistent_bytes,
        plan.memory.checkpoint_read_bytes,
        plan.memory.device_write_bytes,
    );
    out
}

#[derive(Serialize)]
struct PlanStats<'a> {
    summary: &'a str,
    version: u32,
    source_tensor_count: usize,
    covered_contract_count: usize,
    runtime_tensor_count: usize,
    tensor_count: usize,
    buffer_count: usize,
    instruction_count: usize,
    schedule_count: usize,
    /// Sorted, so two dumps of the same plan compare as text.
    instruction_kinds: BTreeMap<&'static str, usize>,
    tile_map_kinds: BTreeMap<&'static str, usize>,
}

/// A compiled plan's shape as JSON: the counts plus instruction and transform
/// histograms.
///
/// This is the small operator-facing dump, not [`dump_load_plan_json`]'s full
/// serialization of every instruction.
pub fn plan_stats_json(plan: &LoadPlan, coverage: ContractCoverage) -> String {
    let mut instruction_kinds: BTreeMap<&'static str, usize> = BTreeMap::new();
    let mut tile_map_kinds: BTreeMap<&'static str, usize> = BTreeMap::new();
    for instr in &plan.instrs {
        *instruction_kinds.entry(instr_name(instr)).or_default() += 1;
        if let StorageInstr::TileMap { kind, .. } = instr {
            *tile_map_kinds.entry(tile_map_name(*kind)).or_default() += 1;
        }
    }
    let summary = describe(plan, coverage);
    let stats = PlanStats {
        summary: &summary,
        version: plan.version,
        source_tensor_count: plan.sources.len(),
        covered_contract_count: coverage.covered,
        runtime_tensor_count: coverage.demanded,
        tensor_count: plan.tensors.len(),
        buffer_count: plan.buffers.len(),
        instruction_count: plan.instrs.len(),
        schedule_count: plan.schedule.len(),
        instruction_kinds,
        tile_map_kinds,
    };
    // A histogram of fixed-name counters cannot fail to serialize.
    serde_json::to_string_pretty(&stats).unwrap_or_default()
}
