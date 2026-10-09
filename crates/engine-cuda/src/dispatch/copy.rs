use kernels_cuda::{Tensor, layout};
use poem_compiler::Region;
use poem_exec::KernelError;
use poem_exec::fire::Serve;
use poem_ir::{Def, Dim, Operands, Operation, Ty, ValueId};

use crate::run::Run;

const SLAB: &str = "fallback.copy";

#[derive(Debug, Clone, Copy)]
pub(crate) struct CopySlot {
    pub(crate) wide: Tensor,
    pub(crate) offset: u64,
    pub(crate) tight: Tensor,
    pub(crate) read: bool,
    pub(crate) written: bool,
}

#[derive(Debug, Clone)]
pub(crate) struct CopyPlan {
    pub(crate) region: u32,
    pub(crate) slots: Vec<CopySlot>,
    pub(crate) bytes: u64,
}

impl Default for CopyPlan {
    fn default() -> CopyPlan {
        CopyPlan {
            region: u32::MAX,
            slots: Vec::new(),
            bytes: 0,
        }
    }
}

impl CopyPlan {
    pub(crate) fn tight(&self, wide: u64) -> Option<Tensor> {
        self.slots
            .iter()
            .find(|slot| slot.wide.ptr == wide)
            .map(|slot| slot.tight)
    }
}

fn row_bytes(handle: Tensor) -> u64 {
    u64::from(handle.width) * poem_compiler::arena::elem_bytes(handle.dtype).unwrap_or(0)
}

fn align(at: u64) -> u64 {
    at.next_multiple_of(16)
}

impl Run<'_> {
    fn copy_plan(&self, region: &Region) -> CopyPlan {
        let mut plan = CopyPlan {
            region: self.at_region(),
            ..CopyPlan::default()
        };
        let rows = self.window().span().rows;
        let mut at = 0u64;
        let mut note = |plan: &mut CopyPlan, id: ValueId, written: bool| {
            let Some(decl) = self.values().get(id.0 as usize) else {
                return;
            };
            if matches!(decl.def, Def::Cache(_)) || matches!(decl.ty, Ty::Struct(_)) {
                return;
            }
            let Ty::Tensor { shape, .. } = &decl.ty else {
                return;
            };
            let rows = match shape.first() {
                Some(Dim::Tokens) => rows,
                Some(Dim::TokensTimes(k)) => rows * k,
                _ => return,
            };
            let wide = self.uncut(id);
            if let Some(held) = plan.slots.iter_mut().find(|slot| slot.wide.ptr == wide.ptr) {
                held.read |= !written;
                held.written |= written;
                return;
            }
            let offset = align(at);
            at = offset + u64::from(rows) * row_bytes(wide);
            plan.slots.push(CopySlot {
                wide,
                offset,
                tight: Tensor::new(offset, rows, wide.width, wide.dtype),
                read: !written,
                written,
            });
        };

        let mut ins: Vec<ValueId> = Vec::new();
        let mut outs: Vec<ValueId> = Vec::new();
        for node in region.nodes.clone() {
            let Some(node) = self.nodes().get(node as usize) else {
                continue;
            };
            ins.clear();
            outs.clear();
            macro_rules! collect {
                ($op:expr) => {{
                    $op.inputs(&mut ins);
                    $op.outputs(&mut outs);
                }};
            }
            match &node.op {
                Operation::Attention(op) => collect!(op),
                Operation::Linear(op) => collect!(op),
                Operation::Elementwise(op) => collect!(op),
                Operation::Layout(op) => collect!(op),
                Operation::Collective(op) => collect!(op),
                Operation::Fused(op) => collect!(op),
                Operation::Spatial(op) => collect!(op),
            }
            for &id in &ins {
                note(&mut plan, id, false);
            }
            for &id in &outs {
                note(&mut plan, id, true);
            }
        }
        plan.bytes = at;
        plan
    }

    fn move_rows(&mut self, region: &Region, out: bool) -> Result<(), kernels_cuda::Error> {
        let mut plan = self.copy_plan(region);
        if plan.slots.is_empty() {
            self.set_copy(plan);
            return Ok(());
        }
        let slab = self
            .ctx()
            .scratch("fallback.copy", SLAB, plan.bytes as usize)? as u64;
        for slot in &mut plan.slots {
            slot.tight.ptr = slab + slot.offset;
        }

        let index = self.gathered_rows(region);
        for slot in &plan.slots {
            let moving = if out { slot.written } else { slot.read };
            if !moving {
                continue;
            }
            let mut wide = slot.wide;
            let mut tight = slot.tight;
            if out {
                layout::scatter_rows(self.ctx(), tight, index, &mut wide)?;
            } else {
                layout::gather_rows(self.ctx(), wide, index, &mut tight)?;
            }
        }
        self.set_copy(plan);
        Ok(())
    }

    fn gathered_rows(&self, region: &Region) -> Tensor {
        self.window()
            .gathered
            .as_ref()
            .unwrap_or_else(|| {
                panic!(
                    "region {} is being copied and its window carries no row map; \
                     `Windows::of` and `Serve::copies` read the same table and cannot \
                     disagree",
                    region.nodes.start
                )
            })
            .rows
    }
}

impl Serve for Run<'_> {
    fn copies(&self, _region: &Region) -> bool {
        self.window().gathered.is_some()
    }

    fn gather(&mut self, region: &Region) -> Result<(), KernelError> {
        self.move_rows(region, false).map_err(crate::error::kernel)
    }

    fn scatter(&mut self, region: &Region) -> Result<(), KernelError> {
        self.move_rows(region, true).map_err(crate::error::kernel)
    }
}
