use std::collections::HashMap;

use crate::error::{Error, Result};
use crate::plan::{BufferDecl, LoadPlan, SourceTensorDecl, StorageInstr};
use crate::types::{BufferId, InstrId, TensorDecl, TensorId};

impl LoadPlan {
    pub fn buffer(&self, id: BufferId) -> Result<&BufferDecl> {
        let decl = self
            .buffers
            .get(id.0 as usize)
            .ok_or_else(|| Error::Internal(format!("buffer {} is not declared", id.0)))?;
        dense(decl.id.0, id.0, "buffer")?;
        Ok(decl)
    }

    pub fn instr(&self, id: InstrId) -> Result<&StorageInstr> {
        let instr = self
            .instrs
            .get(id.0 as usize)
            .ok_or_else(|| Error::Internal(format!("instruction {} is not in the plan", id.0)))?;
        dense(instr_id_of(instr).0, id.0, "instruction")?;
        Ok(instr)
    }
}

pub(crate) fn instr_by_id(instrs: &[StorageInstr], id: InstrId) -> Result<&StorageInstr> {
    let found = instrs
        .get(id.0 as usize)
        .ok_or_else(|| Error::Internal(format!("scheduled instr {} is missing", id.0)))?;
    if instr_id_of(found) != id {
        return Err(Error::Internal(format!(
            "instruction ids are not dense: position {} holds {}",
            id.0,
            instr_id_of(found).0
        )));
    }
    Ok(found)
}

pub(crate) fn instr_id_of(instr: &StorageInstr) -> InstrId {
    match instr {
        StorageInstr::Allocate { id, .. }
        | StorageInstr::Fill { id, .. }
        | StorageInstr::ExtentWrite { id, .. }
        | StorageInstr::BulkExtentWrite { id, .. }
        | StorageInstr::GatherWrite { id, .. }
        | StorageInstr::TileMap { id, .. }
        | StorageInstr::CreateView { id, .. }
        | StorageInstr::Finalize { id, .. } => *id,
    }
}

pub(crate) fn set_instr_id(instr: &mut StorageInstr, new_id: InstrId) {
    match instr {
        StorageInstr::Allocate { id, .. }
        | StorageInstr::Fill { id, .. }
        | StorageInstr::ExtentWrite { id, .. }
        | StorageInstr::BulkExtentWrite { id, .. }
        | StorageInstr::GatherWrite { id, .. }
        | StorageInstr::TileMap { id, .. }
        | StorageInstr::CreateView { id, .. }
        | StorageInstr::Finalize { id, .. } => *id = new_id,
    }
}

fn dense(found: u32, wanted: u32, what: &str) -> Result<()> {
    if found == wanted {
        return Ok(());
    }
    Err(Error::Internal(format!(
        "{what} ids are not dense: position {wanted} holds {found}"
    )))
}

pub struct PlanIndex {
    tensor: HashMap<TensorId, u32>,
    source: HashMap<TensorId, u32>,
}

impl PlanIndex {
    pub fn new(plan: &LoadPlan) -> Self {
        Self {
            tensor: position_by_id(plan.tensors.iter().map(|decl| decl.id)),
            source: position_by_id(plan.sources.iter().map(|decl| decl.id)),
        }
    }

    pub fn tensor<'a>(&self, plan: &'a LoadPlan, id: TensorId) -> Option<&'a TensorDecl> {
        plan.tensors.get(*self.tensor.get(&id)? as usize)
    }

    pub fn source<'a>(&self, plan: &'a LoadPlan, id: TensorId) -> Option<&'a SourceTensorDecl> {
        plan.sources.get(*self.source.get(&id)? as usize)
    }
}

fn position_by_id(ids: impl Iterator<Item = TensorId>) -> HashMap<TensorId, u32> {
    ids.enumerate()
        .filter_map(|(at, id)| u32::try_from(at).ok().map(|at| (id, at)))
        .collect()
}
