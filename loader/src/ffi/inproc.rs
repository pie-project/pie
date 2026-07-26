//! The plan as a document.
//!
//! A [`LoadPlan`] travels as JSON to the host executor and to any tool that
//! wants to read one back. Both directions check the two versions the plan
//! carries, because a plan compiled by a different build of this crate
//! describes a different machine.

use crate::error::CompileError;
use crate::load_plan::{LOAD_PLAN_VERSION, LoadPlan, compiler_version};
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
