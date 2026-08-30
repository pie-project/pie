//! `Fallback::Copy`: `impl model_exec::fire::Serve for Run<'_>` — the shell's
//! half of the answer P4's menu asks for below the copy/split crossover
//! (design §3, `model_compiler::layout`'s `CROSSOVER_ROWS`).
//!
//! ```text
//! fire rows:      [ 4 : 1 | 0 : 1 | 5 : 1 ]      mask {4,5,6,7}
//! split           encode over [0,1)  ·  encode over [2,3)     2 encodes
//! copy            gather 0,2 -> slab | ONE encode | scatter   1 encode + 2 moves
//! ```
//!
//! # The three steps, and which of them is a dispatch
//!
//! 1. **gather** — every row-shaped operand of the region's nodes is read out
//!    of its fire-wide arena column at the rows the window covers and laid
//!    down contiguously in the scratch plane's copy role
//!    (`kernels_metal::layout::gather_rows`). One dispatch per distinct
//!    rectangle, and only for rectangles the region READS.
//! 2. **the nodes**, dispatched exactly as they would have been. They cannot
//!    tell: `Run::cut` re-points every row-shaped operand at the staging
//!    rectangle and `Run::compacted` hands out the re-cut lane tables, so an
//!    arm resolves a copied window through the same call it resolves every
//!    other one through.
//! 3. **scatter** — every rectangle the region WRITES goes back to the fire
//!    rows it was gathered from. Rows outside the window are never written,
//!    which is what keeps a copy one consumer's slow path rather than a fact
//!    about the arena.
//!
//! # There is no fork, no join and no side stream — the ordering IS the encoder
//!
//! **THE ONE PLACE THIS FILE IS SHORTER THAN ITS CUDA TWIN, AND THE REASON IS
//! DESIGN §6.** There a copied region rides its own stream, and the gather has
//! to be ordered against the region's producers by the same events the region
//! is — `engine_cuda::dispatch::copy` inherits that from `walk`'s brackets and
//! the shell's `Cursor::across`. This shell is eager from end to end: one
//! command buffer, one `MTLDispatchTypeSerial` compute pass, encoded in walk
//! order. Every dispatch observes the writes of every dispatch before it
//! (`crate::scratch`'s second bullet states the property and where it is
//! measured), so `gather -> the region's nodes -> scatter` means what it reads
//! as by construction. There is nothing to record and nothing to wait on.
//!
//! # Why the slab is `crate::scratch` and not a carve
//!
//! `model_compiler::arena` carves the busiest instant exactly and leaves no
//! spare rectangle — deliberately; a spare would be bytes every load pays for
//! a path most fires never take. The scratch plane is the mechanism this
//! shell already has for "a workspace an entry may not allocate per fire":
//! one reservation at load, sized at the ceiling, minted per fire and never
//! grown. What it is NOT is the three roles beside it: a copy's bytes are live
//! across the whole region rather than inside one dispatch chain, so the copy
//! role is ADDED to their union and not aliased onto it — `crate::scratch`'s
//! header argues that at length and prices it.
//!
//! **ONE SLAB, SUB-DIVIDED PER REGION**, rather than one per rectangle: the
//! regions are encoded in sequence, so a region's gather is ordered after the
//! previous region's scatter and the bytes are free to be reused. What must
//! not be reused is a slot WITHIN a region, and the plan below is keyed by the
//! fire-wide rectangle's RESOLVED BINDING so that two plan values the carve
//! aliased onto one column compact to one slot — an in-place op reading its
//! input from one staging rectangle and writing its output to another would
//! not be in place any more.

use kernels_metal::{Tensor, layout};
use model_compiler::Region;
use model_exec::KernelError;
use model_exec::fire::Serve;
use model_ir::{Def, Dim, Ty, ValueId};

use crate::run::Run;

/// One rectangle a copied region compacts.
///
/// **KEYED BY THE RESOLVED BINDING, NOT BY THE HANDLE.** `Run::address` is
/// where that is argued: on this plane `crate::arena::carve` mints a row per
/// VALUE, so two values the compiler aliased onto one column answer two
/// `u32`s at one offset, and the pair `(reservation, offset)` is the fact the
/// CUDA twin's address was.
#[derive(Debug, Clone, Copy)]
pub(crate) struct CopySlot {
    /// The fire-wide rectangle's resolved binding — the key.
    key: (u64, u64),
    /// Where the compacted rectangle sits inside the copy role.
    offset: u64,
    /// The fire-wide rectangle, at the fire's own rows.
    wide: Tensor,
    /// The compacted rectangle, at the window's row count. Its handle is
    /// [`NIL`](crate::device::handles::NIL) until [`Run::mint_copy`] has cut
    /// the role.
    tight: Tensor,
    /// Does some node of the region read it? Then it is gathered in.
    read: bool,
    /// Does some node of the region write it? Then it is scattered back.
    written: bool,
}

/// A copied region's whole plan: which rectangles move, and where in the copy
/// role.
#[derive(Debug, Clone)]
pub(crate) struct CopyPlan {
    /// Which region of the template this was built for — the walk's own
    /// index, read back by `Run::compacted` so a stale plan is a panic with a
    /// sentence rather than a silent read of another region's offsets.
    /// `u32::MAX` is the default: no region, which no cursor ever names.
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

/// How many bytes one row of this rectangle takes.
///
/// `Dtype::bytes_ceil` and not the arena's `elem_bytes`, because the two
/// disagree only for a sub-byte element and `crate::window::copyable` admits
/// none: the row move is stamped for bf16 and f32 alone.
fn row_bytes(handle: Tensor) -> u64 {
    u64::from(handle.width) * handle.dtype.bytes_ceil()
}

/// Round a slab offset up so every compacted rectangle starts 16-byte
/// aligned, which is what lets the row move pick its widest copy unit.
fn align(at: u64) -> u64 {
    at.next_multiple_of(16)
}

/// The refusal a copy answers when the load's reservation does not hold this
/// fire's rectangles.
///
/// **A `KernelError` AND NOT A PANIC, EVEN THOUGH IT CANNOT HAPPEN.**
/// `crate::scratch`'s `copy_ceiling` walks the same regions this does and
/// sizes them at the largest bucket a `Fallback::Copy` row covers, and a fire
/// that gathers is a fire in one of those buckets — so the two agree by
/// construction. What makes this a refusal rather than an assertion is that
/// the answer is not recoverable by the shell: `Windows::of` has already cut
/// the region into ONE window, so there is no split left to fall back to, and
/// the honest thing is a fire that fails by name.
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
    /// Which rectangles this region moves, in operand order.
    ///
    /// **THE SHAPE IS THE IR'S AND THE BINDING IS THE CARVE'S.** A value is
    /// row-shaped iff its declared first dim is `Dim::Tokens` — the same test
    /// `Run::cut` slices by, asked of the same table, so a value that would
    /// have been WINDOWED by a split is exactly a value that is COMPACTED by
    /// a copy. Everything else is handed over whole by both.
    ///
    /// `TokensTimes(k)` is absent where the CUDA twin has it, and
    /// `crate::window::copyable` is why: this plane declines to gather a
    /// region naming one at all, because the row map indexes TOKEN rows and
    /// the move entries would refuse the mismatch. A shape the window
    /// declined cannot arrive here, so it is not answered twice.
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
            // A cache space and a plan payload are not rectangles; the pool
            // tables and the plan's own carriers are gathered elsewhere.
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

    /// Cut the copy role into this plan's rectangles.
    ///
    /// One handle per slot, minted here rather than in [`Run::copy_plan`] so
    /// the plan stays arithmetic a test can hold to account without a device.
    ///
    /// # Errors
    ///
    /// [`overflowed`] when the load's reservation does not hold them.
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

    /// Move this region's rectangles, one direction.
    ///
    /// Answers [`kernels_metal::Error`] and not the contract's `KernelError`,
    /// for the reason the `Dispatch*` families answer it: every call in the
    /// body is a `kernels-metal` entry, and `?` converts to the enclosing
    /// function's error type. [`Serve::gather`] and [`Serve::scatter`] below
    /// are the two lines that lift it (`crate::error::kernel`).
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
    /// **THE ANSWER IS ALREADY DECIDED**, and this reads it rather than
    /// making it. `Windows::of` asked P4's table at this fire's bucket, asked
    /// whether the region's operands admit a copy, and built a
    /// [`Gathered`](crate::window::Gathered) window for the regions that
    /// passed — because the window is what a copy IS. So the walk's question
    /// is answered by whether the window it is about to resolve through has
    /// one, which is the only reading that cannot disagree with the table the
    /// encodes will be cut by.
    fn copies(&self, _region: &Region) -> bool {
        self.window().gathered.is_some()
    }

    /// # Errors
    ///
    /// [`overflowed`] for a reservation this region's rectangles do not fit,
    /// and whatever `kernels_metal::layout::gather_rows` answered.
    fn gather(&mut self, region: &Region) -> Result<(), KernelError> {
        let mut plan = self.copy_plan(region);
        // A prepare region's copy is not a movement at all: its builder
        // simply carries ONE set of tables over the union
        // (`crate::window::Gathered`'s ambient twins). Seated anyway, so that
        // `Run::compacted` finds a plan for the region it is inside.
        if !plan.slots.is_empty() {
            self.mint_copy(&mut plan)?;
        }
        self.seat_copy(plan);
        self.move_rows(region, false).map_err(crate::error::kernel)
    }

    /// # Errors
    ///
    /// As [`gather`](Serve::gather), minus the reservation: the plan the
    /// gather seated is the plan this reads, so the rectangles are the ones
    /// already minted and the two halves cannot address different bytes.
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
