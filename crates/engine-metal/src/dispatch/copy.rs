use kernels_metal::Tensor;
use kernels_metal::gather;
use poem_compiler::Region;
use poem_exec::KernelError;
use poem_exec::fire::{Serve, fallback};
use poem_ir::{Def, Dim, Ty, ValueId};

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
    fn copy_plan(&self, region: &Region) -> crate::run::Ran<CopyPlan> {
        let mut plan = CopyPlan {
            region: self.at_region(),
            ..CopyPlan::default()
        };
        let rows = self.window()?.span.rows;
        let mut at = 0u64;
        let mut note = |plan: &mut CopyPlan, id: ValueId, written: bool| -> crate::run::Ran<()> {
            let Some(decl) = self.values().get(id.0 as usize) else {
                return Ok(());
            };
            if matches!(decl.def, Def::Cache(_)) || matches!(decl.ty, Ty::Struct(_)) {
                return Ok(());
            }
            let Ty::Tensor { shape, .. } = &decl.ty else {
                return Ok(());
            };
            if !matches!(shape.first(), Some(Dim::Tokens)) {
                return Ok(());
            }
            let wide = self.uncut(id)?;
            let Some(key) = self.address(wide.buf) else {
                return Ok(());
            };
            if let Some(held) = plan.slots.iter_mut().find(|slot| slot.key == key) {
                held.read |= !written;
                held.written |= written;
                return Ok(());
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
            Ok(())
        };

        let Some((ins, outs)) = fallback::operands(self.nodes(), region) else {
            return Ok(plan);
        };
        for id in ins {
            note(&mut plan, id, false)?;
        }
        for id in outs {
            note(&mut plan, id, true)?;
        }
        // The fire's masks are per row: where it stages any, the copy carries
        // each gathered row's word and mask row, as 4-byte words, which the
        // window's plans read in place of the fire's (`Run::cut_rows`).
        let tables = self.fire_tables();
        if tables.mask_stride > 0 {
            for (handle, width) in [
                (tables.mask, tables.mask_stride / 4),
                (tables.mask_enabled, 1),
            ] {
                let wide = Tensor::new(handle.buf, handle.rows, width, poem_ir::Dtype::I32);
                let Some(key) = self.address(handle.buf) else {
                    continue;
                };
                let offset = align(at);
                at = offset + u64::from(rows) * row_bytes(wide);
                plan.slots.push(CopySlot {
                    key,
                    offset,
                    wide,
                    tight: Tensor::new(
                        crate::device::handles::NIL,
                        rows,
                        width,
                        poem_ir::Dtype::I32,
                    ),
                    read: true,
                    written: false,
                });
            }
        }
        plan.bytes = at;
        Ok(plan)
    }

    fn mint_copy(&self, plan: &mut CopyPlan) -> Result<(), KernelError> {
        for slot in &mut plan.slots {
            let bytes = u64::from(slot.tight.rows) * row_bytes(slot.tight);
            let Some(handle) = self
                .copy_room(slot.offset, bytes)
                .map_err(crate::error::kernel)?
            else {
                return Err(overflowed(plan.bytes));
            };
            slot.tight.buf = handle;
        }
        Ok(())
    }

    fn move_rows(&self, region: &Region, out: bool) -> Result<(), kernels_metal::Error> {
        let index = self.gathered_rows(region)?;
        for slot in &self.staged_copy().slots {
            let moving = if out { slot.written } else { slot.read };
            if !moving {
                continue;
            }
            if out {
                gather::scatter_rows(self.ctx(), slot.tight, index, slot.wide)?;
            } else {
                gather::gather_rows(self.ctx(), slot.wide, index, slot.tight)?;
            }
        }
        Ok(())
    }

    /// The row map a copied region's window carries. `Windows::of` and
    /// `Serve::copies` read the same table, so a copied window without one
    /// is a mis-carved fire, refused by name.
    fn gathered_rows(&self, region: &Region) -> Result<Tensor, kernels_metal::Error> {
        self.window()?
            .gathered
            .as_ref()
            .map(|gathered| gathered.rows)
            .ok_or_else(|| kernels_metal::Error::Backend {
                op: "copy",
                detail: format!(
                    "region {} is copied and its window carries no row map",
                    region.nodes.start
                ),
            })
    }
}

impl Serve for Run<'_> {
    fn copies(&self, _region: &Region) -> bool {
        // A run past the fire's windows refuses at its gather (`Run::window`).
        self.window().is_ok_and(|window| window.gathered.is_some())
    }

    fn gather(&mut self, region: &Region) -> Result<(), KernelError> {
        let mut plan = self.copy_plan(region).map_err(crate::error::kernel)?;
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
             region's; `poem_exec::fire::walk` brackets a copied region's nodes with \
             the pair and this is what says the bracket was lost",
            region.nodes.start
        );
        self.move_rows(region, true).map_err(crate::error::kernel)
    }
}
