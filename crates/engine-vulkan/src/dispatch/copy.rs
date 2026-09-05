use kernels_vulkan::{Tensor, layout};
use model_compiler::Region;
use model_exec::KernelError;
use model_exec::fire::Serve;
use model_ir::{Def, Dim, Ty, ValueId};

use crate::run::Run;

#[derive(Debug, Clone, Copy)]
pub(crate) struct CopySlot {
    key: (u64, u64),

    offset: u64,

    wide: Tensor,

    tight: Tensor,

    read: bool,

    written: bool,
}

#[derive(Debug, Clone)]
pub(crate) struct CopyPlan {
    pub(crate) region: u32,
    slots: Vec<CopySlot>,

    bytes: u64,
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
    pub(crate) fn tight(&self, key: (u64, u64)) -> Option<Tensor> {
        self.slots
            .iter()
            .find(|slot| slot.key == key)
            .map(|slot| slot.tight)
    }
}

fn row_bytes(handle: Tensor) -> u64 {
    u64::from(handle.width) * handle.dtype.bytes_ceil()
}

fn align(at: u64) -> u64 {
    at.next_multiple_of(16)
}

fn overflowed(bytes: u64) -> KernelError {
    KernelError::Backend {
        op: "fallback.copy",
        detail: format!(
            "this region's staging rectangles come to {bytes} bytes and the copy role \
             this load reserved is smaller; `scratch::copy_ceiling` and this plan walk \
             the same regions and cannot disagree"
        ),
    }
}

impl Run<'_> {
    fn copy_plan(&self, region: &Region) -> CopyPlan {
        let mut plan = CopyPlan {
            region: self.at_region(),
            ..CopyPlan::default()
        };
        let rows = self.window().span.rows;
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
            if !matches!(shape.first(), Some(Dim::Tokens)) {
                return;
            }
            let wide = self.uncut(id);
            let Some(key) = self.address(wide.buf) else {
                return;
            };
            if let Some(held) = plan.slots.iter_mut().find(|slot| slot.key == key) {
                held.read |= !written;
                held.written |= written;
                return;
            }
            let offset = align(at);
            at = offset + u64::from(rows) * row_bytes(wide);
            plan.slots.push(CopySlot {
                key,
                offset,
                wide,
                tight: Tensor::new(crate::device::handles::NIL, rows, wide.width, wide.dtype),
                read: !written,
                written,
            });
        };

        let Some((ins, outs)) = crate::window::operands(self.nodes(), region) else {
            return plan;
        };
        for id in ins {
            note(&mut plan, id, false);
        }
        for id in outs {
            note(&mut plan, id, true);
        }
        plan.bytes = at;
        plan
    }

    fn mint_copy(&self, plan: &mut CopyPlan) -> Result<(), KernelError> {
        for slot in &mut plan.slots {
            let bytes = u64::from(slot.tight.rows) * row_bytes(slot.tight);
            let Some(handle) = self.copy_room(slot.offset, bytes) else {
                return Err(overflowed(plan.bytes));
            };
            slot.tight.buf = handle;
        }
        Ok(())
    }

    fn move_rows(&self, region: &Region, out: bool) -> Result<(), kernels_vulkan::Error> {
        let index = self.gathered_rows(region);
        for slot in &self.staged_copy().slots {
            let moving = if out { slot.written } else { slot.read };
            if !moving {
                continue;
            }
            if out {
                layout::scatter_rows(self.ctx(), slot.tight, index, slot.wide)?;
            } else {
                layout::gather_rows(self.ctx(), slot.wide, index, slot.tight)?;
            }
        }
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
        let mut plan = self.copy_plan(region);

        if !plan.slots.is_empty() {
            self.mint_copy(&mut plan)?;
        }
        self.set_copy(plan);
        self.move_rows(region, false).map_err(crate::error::kernel)
    }

    fn scatter(&mut self, region: &Region) -> Result<(), KernelError> {
        assert_eq!(
            self.staged_copy().region,
            self.at_region(),
            "region {} is being scattered and the seated copy plan is another \
             region's; `model_exec::fire::walk` brackets a copied region's nodes with \
             the pair and this is what says the bracket was lost",
            region.nodes.start
        );
        self.move_rows(region, true).map_err(crate::error::kernel)
    }
}
