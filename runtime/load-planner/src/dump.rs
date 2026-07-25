use serde::Serialize;

use crate::error::CompileError;
use crate::ir::LayoutPlan;
use crate::load_plan::LoadPlan;

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
