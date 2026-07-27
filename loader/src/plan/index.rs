//! Finding things in a plan.
//!
//! Every id a plan carries is dense and assigned in push order, so a lookup is
//! an array index. It was not written that way: `buffers.iter().find()` and
//! `tensors.iter().find()` appeared in the passes, in the memory accounting and
//! in the backend lowering, which made compilation quadratic in tensor count —
//! 2.1 s for a 32k-tensor checkpoint, of which two scans were most of it.
//!
//! The fix is not a cache but an invariant, checked on every access: if
//! `buffers[i].id != BufferId(i)` something built the plan wrong, and that is
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

    pub fn buffer_mut(&mut self, id: BufferId) -> Result<&mut BufferDecl> {
        let decl = self
            .buffers
            .get_mut(id.0 as usize)
            .ok_or_else(|| Error::Internal(format!("buffer {} is not declared", id.0)))?;
        dense(decl.id.0, id.0, "buffer")?;
        Ok(decl)
    }

    pub fn instr(&self, id: InstrId) -> Result<&StorageInstr> {
        let instr = self
            .instrs
            .get(id.0 as usize)
            .ok_or_else(|| Error::Internal(format!("instruction {} is not in the plan", id.0)))?;
        dense(
            crate::plan::passes::instr_id_of(instr).0,
            id.0,
            "instruction",
        )?;
        Ok(instr)
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

/// Lookups that are *not* an array index.
///
/// Tensor ids interleave two allocators — a contract's own tensors take their
/// declaration order, and generated scale tensors continue past the end — so
/// the tensor table is sparse where the buffer table is not. Built once by a
/// pass that needs it rather than carried on the plan, because a plan that
/// owned an index would have to keep it right through every rewrite.
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

    /// The declaration behind a buffer, if the buffer names one.
    pub fn buffer_tensor<'a>(&self, plan: &'a LoadPlan, id: BufferId) -> Option<&'a TensorDecl> {
        self.tensor(plan, plan.buffer(id).ok()?.tensor?)
    }
}

fn position_by_id(ids: impl Iterator<Item = TensorId>) -> HashMap<TensorId, u32> {
    ids.enumerate()
        .filter_map(|(at, id)| u32::try_from(at).ok().map(|at| (id, at)))
        .collect()
}
