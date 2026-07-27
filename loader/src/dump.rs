use std::collections::BTreeMap;
use std::fmt::Write as _;

use serde::Serialize;

use crate::error::Error;
use crate::plan::{LoadPlan, StorageInstr, TileMapKind};

pub fn dump_load_plan_json(plan: &LoadPlan) -> Result<String, Error> {
    serde_json::to_string_pretty(plan)
        .map_err(|err| Error::Internal(format!("load-plan dump failed: {err}")))
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
pub fn describe(plan: &LoadPlan) -> String {
    let rewrites: usize = plan.passes.iter().map(|pass| pass.rewrites).sum();
    let mut out = String::new();
    let _ = write!(
        out,
        "load_plan(source_tensors={}, tensors={}, \
         buffers={}, instrs={}, schedule={}, passes={}, \
         rewrites={}, persistent_bytes={}, read_bytes={}, write_bytes={})",
        plan.sources.len(),
        plan.tensors.len(),
        plan.buffers.len(),
        plan.instrs.len(),
        plan.schedule.len(),
        plan.passes.len(),
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
    source_tensor_count: usize,
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
pub fn plan_stats_json(plan: &LoadPlan) -> String {
    let mut instruction_kinds: BTreeMap<&'static str, usize> = BTreeMap::new();
    let mut tile_map_kinds: BTreeMap<&'static str, usize> = BTreeMap::new();
    for instr in &plan.instrs {
        *instruction_kinds.entry(instr_name(instr)).or_default() += 1;
        if let StorageInstr::TileMap { kind, .. } = instr {
            *tile_map_kinds.entry(tile_map_name(*kind)).or_default() += 1;
        }
    }
    let summary = describe(plan);
    let stats = PlanStats {
        summary: &summary,
        source_tensor_count: plan.sources.len(),
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
