//! `Fallback::Copy`: gathers a region's row-shaped operands into scratch, runs the region's nodes unchanged, then scatters the written rectangles back to their fire rows.

use kernels_metal::{Tensor, layout};
use model_compiler::Region;
use model_exec::KernelError;
use model_exec::fire::Serve;
use model_ir::{Def, Dim, Ty, ValueId};

use crate::run::Run;

/// One rectangle a copied region compacts. Keyed by the resolved binding `(reservation, offset)`, not the handle, so two aliased values compact to one slot.
#[derive(Debug, Clone, Copy)]
pub(crate) struct CopySlot {
    key: (u64, u64),
    /// Where the compacted rectangle sits inside the copy role.
    offset: u64,
    /// The fire-wide rectangle, at the fire's own rows.
    wide: Tensor,
    /// The compacted rectangle, at the window's row count. Handle is [`NIL`](crate::device::handles::NIL) until [`Run::mint_copy`] cuts the role.
    tight: Tensor,
    /// Read by some node of the region, so gathered in.
    read: bool,
    /// Written by some node of the region, so scattered back.
    written: bool,
}

/// A copied region's whole plan: which rectangles move, and where in the copy role (one scratch slab, sub-divided per region).
#[derive(Debug, Clone)]
pub(crate) struct CopyPlan {
    /// Which region of the template this was built for; `u32::MAX` is the default, which no cursor ever names.
    pub(crate) region: u32,
    slots: Vec<CopySlot>,
    /// The copy-role bytes this region needs.
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
    /// The compacted rectangle a fire-wide binding was gathered into, if this
    /// region moves it.
    pub(crate) fn tight(&self, key: (u64, u64)) -> Option<Tensor> {
        self.slots
            .iter()
            .find(|slot| slot.key == key)
            .map(|slot| slot.tight)
    }
}

/// How many bytes one row of this rectangle takes. `Dtype::bytes_ceil`, not
/// the arena's `elem_bytes`: the row move only ever sees bf16 or f32.
fn row_bytes(handle: Tensor) -> u64 {
    u64::from(handle.width) * handle.dtype.bytes_ceil()
}

/// Round a slab offset up so every compacted rectangle starts 16-byte
/// aligned, which is what lets the row move pick its widest copy unit.
fn align(at: u64) -> u64 {
    at.next_multiple_of(16)
}

/// The refusal a copy answers when the load's reservation does not hold this fire's rectangles.
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
    /// Which rectangles this region moves, in operand order. A value is
    /// row-shaped iff its first declared dim is `Dim::Tokens` — the same test
    /// `Run::cut` uses to decide what a split windows.
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
            // Cache spaces and plan payloads are not rectangles.
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

    /// Cut the copy role into this plan's rectangles. Errs [`overflowed`] when the load's reservation does not hold them.
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

    /// Move this region's rectangles, one direction. Answers
    /// [`kernels_metal::Error`], lifted to `KernelError` by its callers below.
    fn move_rows(&self, region: &Region, out: bool) -> Result<(), kernels_metal::Error> {
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
    /// Checks whether `Windows::of` built a [`Gathered`](crate::window::Gathered) window for this region.
    fn copies(&self, _region: &Region) -> bool {
        self.window().gathered.is_some()
    }

    /// Errs [`overflowed`] for a reservation this region's rectangles don't fit, or whatever `gather_rows` answered.
    fn gather(&mut self, region: &Region) -> Result<(), KernelError> {
        let mut plan = self.copy_plan(region);
        // A prepare region's copy moves nothing, but is still seated so
        // Run::compacted finds a plan for the region it is inside.
        if !plan.slots.is_empty() {
            self.mint_copy(&mut plan)?;
        }
        self.set_copy(plan);
        self.move_rows(region, false).map_err(crate::error::kernel)
    }

    /// As [`gather`](Serve::gather), minus the reservation: reads the plan gather already seated.
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
