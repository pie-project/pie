//! `Fallback::Copy`: `impl model_exec::fire::Serve for Run<'_>` — the shell's
//! half of the answer P4's menu asks for below the copy/split crossover
//! (design §3, `model_compiler::layout`'s `CROSSOVER_ROWS`).
//!
//! ```text
//! fire rows:      [ 4 : 1 | 0 : 1 | 5 : 1 ]      mask {4,5,6,7}
//! split           launch over [0,1)  ·  launch over [2,3)     2 launches
//! copy            gather 0,2 -> slab | ONE launch | scatter   1 launch + 2 moves
//! ```
//!
//! # The three steps, and which of them is a kernel
//!
//! 1. **gather** — every row-shaped operand of the region's nodes is read out
//!    of its fire-wide arena column at the rows the window covers and laid
//!    down contiguously in a scratch slab
//!    (`kernels_cuda::layout::gather_rows`). One launch per distinct
//!    rectangle, and only for rectangles the region READS.
//! 2. **the nodes**, dispatched exactly as they would have been. They cannot
//!    tell: `Run::cut` re-points every row-shaped operand at the slab and
//!    `Run::pool` / `Run::planning` hand out the gathered lane tables, so an
//!    arm resolves a copied window through the same three calls it resolves
//!    every other one through.
//! 3. **scatter** — every rectangle the region WRITES goes back to the fire
//!    rows it was gathered from. Rows outside the window are never written,
//!    which is what keeps a copy one consumer's slow path rather than a fact
//!    about the arena.
//!
//! # Why the slab is `Ctx::scratch` and not a carve
//!
//! `model_compiler::arena` carves the busiest instant exactly and leaves no
//! spare rectangle — deliberately; a spare would be bytes every load pays for
//! a path most fires never take. The scratch plane is the mechanism this
//! stack already has for "a workspace an entry may not allocate per fire":
//! grown but never shrunk, keyed by `(arena, name, stream)` so two arms of a
//! P6 fork group do not share one, and freed when the context is. Its
//! contract is that a capture may not grow it, which the record path already
//! satisfies for every other consumer — a key's first two fires are eager and
//! at the same shape (`record::WARM_FIRES`), so by the time a capture pass
//! asks, the slab is already the size that pass will ask for.
//!
//! **ONE SLAB PER STREAM, SUB-DIVIDED PER REGION**, rather than one per
//! rectangle: the regions of a stream are sequential, so a region's gather is
//! ordered after the previous region's scatter and the bytes are free to be
//! reused. What must not be reused is a slot WITHIN a region, and the plan
//! below is keyed by the fire-wide rectangle's ADDRESS so that two plan values
//! the carve aliased onto one column compact to one slot — an in-place op
//! reading its input from one slab rectangle and writing its output to
//! another would not be in place any more.

use kernels_cuda::{Tensor, layout};
use model_compiler::Region;
use model_exec::KernelError;
use model_exec::fire::Serve;
use model_ir::{Def, Dim, Operands, Operation, Ty, ValueId};

use crate::run::Run;

/// The name the copy slab is keyed by inside a context's scratch arena.
///
/// One name and not one per region: see the module header. The `(arena,
/// stream)` halves of the key are the context's own.
const SLAB: &str = "fallback.copy";

/// One rectangle a copied region compacts.
///
/// **KEYED BY THE FIRE-WIDE ADDRESS**, which is the field that makes aliasing
/// work: `Run::tensor` resolves every plan value through the arena's root, so
/// two values the carve folded onto one column answer the same `wide`, find
/// the same slot, and share the same compacted rectangle. Keying by
/// `ValueId` would give an in-place op two.
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
    /// Which region of the template this was built for — the walk's own
    /// index, read back by `Run::cut` so a stale plan is a panic with a
    /// sentence rather than a silent read of another region's slab offsets.
    /// `u32::MAX` is the default: no region, which no cursor ever names.
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

/// Round a slab offset up so every compacted rectangle starts 16-byte
/// aligned, which is what lets the row movement pick its widest copy unit.
fn align(at: u64) -> u64 {
    at.next_multiple_of(16)
}

impl Run<'_> {
    /// Which rectangles this region moves, in operand order.
    ///
    /// **THE SHAPE IS THE IR'S AND THE ADDRESS IS THE CARVE'S.** A value is
    /// row-shaped iff its declared first dim is `Dim::Tokens` (or
    /// `TokensTimes`, whose rows are a fixed multiple of them) — that is the
    /// same test `Run::cut` slices by, asked of the same table, so a value
    /// that would have been WINDOWED by a split is exactly a value that is
    /// COMPACTED by a copy. Everything else is handed over whole by both.
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
            // tables and the schedule are gathered elsewhere.
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

    /// Move this region's rectangles, one direction.
    ///
    /// The slab is taken first and the plan's offsets are resolved against
    /// it, so the addresses `Run::cut` hands the nodes and the addresses
    /// these launches write are one arithmetic — asking the scratch plane
    /// twice for the same name in one fire answers the same pointer, but
    /// deriving both from one call is what makes that not need to be true.
    ///
    /// Answers [`kernels_cuda::Error`] and not the contract's `KernelError`,
    /// for the reason the `Dispatch*` families answer it: every call in the
    /// body is a `kernels-cuda` entry, and `?` converts to the enclosing
    /// function's error type. [`Serve::gather`] and [`Serve::scatter`] below
    /// are the two lines that lift it (`crate::error::kernel`).
    fn move_rows(&mut self, region: &Region, out: bool) -> Result<(), kernels_cuda::Error> {
        let mut plan = self.copy_plan(region);
        if plan.slots.is_empty() {
            // A prepare region's copy is not a movement at all: its builder
            // simply carves ONE schedule over the union. Recorded anyway, so
            // that `Run::cut` finds a plan for the region it is inside.
            self.seat_copy(plan);
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
        self.seat_copy(plan);
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
    /// whether the region's operands admit a copy, and built a [`Gathered`]
    /// window for the regions that passed — because the window is what a copy
    /// IS. So the walk's question is answered by whether the window it is
    /// about to resolve through has one, which is the only reading that
    /// cannot disagree with the table the launches will be cut by.
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
