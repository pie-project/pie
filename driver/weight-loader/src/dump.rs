use serde::Serialize;

use crate::error::CompileError;
use crate::ir::LayoutPlan;
use crate::storage::StorageProgram;

#[derive(Clone, Debug, Serialize)]
pub struct CompilerDump<'a> {
    pub compiler_version: u32,
    pub layout: &'a LayoutPlan,
    pub storage: &'a StorageProgram,
}

pub fn dump_storage_program_json(program: &StorageProgram) -> Result<String, CompileError> {
    serde_json::to_string_pretty(program)
        .map_err(|err| CompileError::Internal(format!("storage dump failed: {err}")))
}

pub fn dump_compiler_json(
    layout: &LayoutPlan,
    storage: &StorageProgram,
) -> Result<String, CompileError> {
    serde_json::to_string_pretty(&CompilerDump {
        compiler_version: crate::storage::STORAGE_PROGRAM_VERSION,
        layout,
        storage,
    })
    .map_err(|err| CompileError::Internal(format!("compiler dump failed: {err}")))
}
