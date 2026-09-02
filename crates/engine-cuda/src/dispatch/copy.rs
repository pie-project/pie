//! `Fallback::Copy`: gathers a region's row-shaped operands into a scratch
//! slab, dispatches the region's nodes against the compacted rows, then
//! scatters written rows back to their fire-wide positions.

use kernels_cuda::{Tensor, layout};
use model_compiler::Region;
use model_exec::KernelError;
use model_exec::fire::Serve;
use model_ir::{Def, Dim, Operands, Operation, Ty, ValueId};

use crate::run::Run;

/// Name the copy slab is keyed by inside a context's scratch arena.
const SLAB: &str = "fallback.copy";

/// One rectangle a copied region compacts.
///
/// Keyed by the fire-wide tensor address (not `ValueId`), so two plan values
/// aliased onto one arena column share a single compacted slot.
#[derive(Debug, Clone, Copy)]
pub(crate) struct CopySlot {
    /// The fire-wide rectangle. Its `ptr` is the key.
    pub(crate) wide: Tensor,
    /// Where the compacted rectangle sits inside the slab.
    pub(crate) offset: u64,
    /// The compacted rectangle, at the window's row count.
    pub(crate) tight: Tensor,
    /// Does some node of the region read it? Then it is gathered in.
    pub(crate) read: bool,
    /// Does some node of the region write it? Then it is scattered back.
    pub(crate) written: bool,
}

/// A copied region's whole plan: which rectangles move, and where in the slab.
#[derive(Debug, Clone)]
pub(crate) struct CopyPlan {
    /// Region index this plan was built for; `u32::MAX` means none.
    /// `Run::cut` checks it so a stale plan panics rather than misreads.
    pub(crate) region: u32,
    pub(crate) slots: Vec<CopySlot>,
    /// The slab bytes this region needs.
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
    /// The compacted rectangle a fire-wide address was gathered into, if this
    /// region moves it.
    pub(crate) fn tight(&self, wide: u64) -> Option<Tensor> {
        self.slots
            .iter()
            .find(|slot| slot.wide.ptr == wide)
            .map(|slot| slot.tight)
    }
}

/// How many bytes one row of this handle takes.
fn row_bytes(handle: Tensor) -> u64 {
    u64::from(handle.width) * model_compiler::arena::elem_bytes(handle.dtype).unwrap_or(0)
}

/// Rounds up to 16-byte alignment so row copies can use their widest unit.
fn align(at: u64) -> u64 {
    at.next_multiple_of(16)
}

impl Run<'_> {
    /// Which rectangles this region moves, in operand order.
    ///
    /// A value is row-shaped iff its declared first dim is `Dim::Tokens` or
    /// `TokensTimes` (a fixed multiple); everything else is passed whole.
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
            // Cache spaces and struct payloads aren't rectangles; skip them.
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
                Operation::CustomCuda(op) => collect!(op),
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

    /// Moves this region's rectangles in one direction (gather in / scatter
    /// out). The slab is fetched once so plan and launch addresses agree.
    fn move_rows(&mut self, region: &Region, out: bool) -> Result<(), kernels_cuda::Error> {
        let mut plan = self.copy_plan(region);
        if plan.slots.is_empty() {
            // record even an empty plan so `Run::cut` finds one for this region.
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

    /// The row map this region's window was gathered by.
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
    /// Reads the decision `Windows::of` already made: a region copies iff
    /// its window carries a [`Gathered`] row map.
    ///
    /// [`Gathered`]: crate::window::Gathered
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
