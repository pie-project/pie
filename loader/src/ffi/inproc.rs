//! In-process load-planning entry: read a directory, then compile.
//!
//! The runtime links this crate as an `rlib` and calls
//! [`compile_snapshot_to_plan_bytes`] directly — no FFI — to turn a checkpoint
//! directory into a serialized [`LoadPlan`] shipped to the driver. The **bulk
//! weight bytes never cross this boundary**: the plan only records where each
//! tensor lives (file + offset + span) and how to place it; the driver reads
//! the payloads from its own copy of the checkpoint during execution.
//!
//! Every function here is a two-liner, and that is the point. The read is
//! [`crate::checkpoint::read`] and the compile is
//! [`crate::planner::compile_load_plan`]; this module is only the seam where a
//! caller who happens to have a directory meets a compiler that only ever sees
//! values.

use std::path::Path;

use crate::checkpoint::read::parse_checkpoint_metadata;
use crate::config::ModelConfig;
use crate::error::CompileError;
use crate::load_plan::{LOAD_PLAN_VERSION, LoadPlan, StorageTarget, compiler_version};
use crate::planner::compile_load_plan;
/// Compile a checkpoint directory into a [`LoadPlan`] entirely in-process.
///
/// `model` and `target` carry the same coarse configuration the C++ bridge
/// builds from the HF config + backend caps. The RuntimeABI is derived with
/// [`crate::arch::default_contract`], exactly as the FFI compile does when the
/// caller supplies no explicit contracts.
pub fn compile_snapshot(
    snapshot_dir: &Path,
    model: &ModelConfig,
    target: StorageTarget,
) -> Result<LoadPlan, CompileError> {
    let metadata = parse_checkpoint_metadata(snapshot_dir)?;
    let contract = crate::arch::default_contract(&metadata, model, &target)?;
    compile_load_plan(&metadata, &contract, target)
}

/// Compile a checkpoint directory and serialize the resulting plan as the
/// versioned JSON document consumed by native and host executors.
pub fn compile_snapshot_to_plan_bytes(
    snapshot_dir: &Path,
    model: &ModelConfig,
    target: StorageTarget,
) -> Result<Vec<u8>, CompileError> {
    let program = compile_snapshot(snapshot_dir, model, target)?;
    serialize_load_plan(&program)
}

pub fn serialize_load_plan(plan: &LoadPlan) -> Result<Vec<u8>, CompileError> {
    serde_json::to_vec(plan)
        .map_err(|err| CompileError::Internal(format!("load plan serialize failed: {err}")))
}

pub fn deserialize_load_plan(bytes: &[u8]) -> Result<LoadPlan, CompileError> {
    let plan: LoadPlan = serde_json::from_slice(bytes).map_err(|err| {
        CompileError::InvalidInput(format!("load plan deserialize failed: {err}"))
    })?;
    if plan.version != LOAD_PLAN_VERSION {
        return Err(CompileError::InvalidInput(format!(
            "load plan version {} does not match executor version {}",
            plan.version, LOAD_PLAN_VERSION
        )));
    }
    let expected = compiler_version();
    if plan.compiler_version != expected {
        return Err(CompileError::InvalidInput(format!(
            "load planner version {:#x} does not match executor version {expected:#x}",
            plan.compiler_version
        )));
    }
    Ok(plan)
}
