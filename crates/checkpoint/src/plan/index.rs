//! Finding things in a plan. Every id a plan carries is dense and assigned
//! in push order, so a lookup is an array index, checked on every access: if
//! `buffers[i].id != BufferId(i)` something built the plan wrong, which is
//! an `Internal` error rather than a silently slow path.

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

/// Resolve a scheduled instruction by id, against a slice the caller owns
/// (passes clone `instrs` before rewriting it, so they can't go through
/// [`LoadPlan::instr`]). Same invariant: ids are dense.
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

/// The id every instruction carries, whichever variant it is.
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

/// Lookups that are *not* an array index: tensor ids interleave two
/// allocators (a contract's own tensors, then generated scale tensors), so
/// the tensor table is sparse where the buffer table is not. Built once by
/// a pass that needs it rather than carried on the plan.
pub struct PlanIndex {
    tensor: HashMap<TensorId, u32>,
    source: HashMap<TensorId, u32>,
    /// What each `CreateView` output looks at. See [`PlanIndex::buffer_tensor`].
    view_input: HashMap<BufferId, BufferId>,
}

impl PlanIndex {
    pub fn new(plan: &LoadPlan) -> Self {
        Self {
            tensor: position_by_id(plan.tensors.iter().map(|decl| decl.id)),
            source: position_by_id(plan.sources.iter().map(|decl| decl.id)),
            view_input: plan
                .instrs
                .iter()
                .filter_map(|instr| match instr {
                    StorageInstr::CreateView { input, output, .. } => Some((*output, *input)),
                    _ => None,
                })
                .collect(),
        }
    }

    pub fn tensor<'a>(&self, plan: &'a LoadPlan, id: TensorId) -> Option<&'a TensorDecl> {
        plan.tensors.get(*self.tensor.get(&id)? as usize)
    }

    pub fn source<'a>(&self, plan: &'a LoadPlan, id: TensorId) -> Option<&'a SourceTensorDecl> {
        plan.sources.get(*self.source.get(&id)? as usize)
    }

    /// The declaration behind a buffer, chasing views.
    ///
    /// A `CreateView` output declares no tensor: it is a window onto one, and
    /// takes the elements of what it looks at, following the same chain
    /// `resolve` walks for bytes. The chain terminates because `CreateView`
    /// names an input that is already allocated.
    pub fn buffer_tensor<'a>(&self, plan: &'a LoadPlan, id: BufferId) -> Option<&'a TensorDecl> {
        let mut at = id;
        loop {
            if let Some(tensor) = plan.buffer(at).ok()?.tensor {
                return self.tensor(plan, tensor);
            }
            at = *self.view_input.get(&at)?;
        }
    }
}

fn position_by_id(ids: impl Iterator<Item = TensorId>) -> HashMap<TensorId, u32> {
    ids.enumerate()
        .filter_map(|(at, id)| u32::try_from(at).ok().map(|at| (id, at)))
        .collect()
}
