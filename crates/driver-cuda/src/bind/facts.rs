//! One launch's world, as the derived binder reads it.
//!
//! This file was the driver's answer to every fact a bind body could ask for
//! — 96 keys, ~40 accessors over `DispatchCtx`/`AttnCtx`/`GdnCtx`. The no-ask
//! sweep (`.wiki/designs/design-no-ask.md` §B7) retires the question: a
//! routine's signature carries its constants as `Const` marks, its geometry
//! as operand extents, and the driver-owned objects as runtime operands the
//! per-fire arena answers (`super::views`). What remains here is the CURSOR —
//! the launch's own operands, its statement scalars, its named weights — the
//! things `table::operand` still reads to bind a column.

use core::ffi::c_void;
use core::ptr::NonNull;

use kernels_cuda::attn::Rows;

use super::{AttnCtx, BoundLaunch, DispatchCtx, GdnCtx, LaunchSpec};

/// One launch's whole world, as the derived binder reads it.
pub struct Fire<'a> {
    /// The launch, with every operand resolved.
    pub bound: &'a BoundLaunch<'a>,
    /// The op join: the operand split, the statement's params, the weight.
    pub spec: &'a LaunchSpec,
    /// The fire's model-wide facts.
    pub ctx: &'a DispatchCtx,
    /// The attention half of the fire, when it has one. Kept for the driver
    /// ops (`bind::dispatch`'s own match) that read it whole; no per-key
    /// accessor remains.
    pub attn: Option<&'a AttnCtx>,
    /// The gated-delta-net half, when it has one. As [`Self::attn`].
    pub gdn: Option<&'a GdnCtx>,
    /// This region's row count, already narrowed.
    pub rows: i32,
    /// The named weight, resolved ONCE by the caller: `Facts` is not `&mut`.
    pub w_named: *const c_void,
    /// The second named weight, resolved the same way.
    pub w_named2: *const c_void,
}

impl Fire<'_> {
    /// This launch's layer, as an index.
    pub fn layer_index(&self) -> usize {
        usize::from(self.bound.layers.start)
    }

    /// The weights a statement names by NAME: `0` is `spec.weight`, `1` is
    /// `spec.weight2`. Null is absence, and which kind is the caller's job.
    pub fn weight_named(&self, i: usize) -> Option<*mut c_void> {
        let p = match i {
            0 => self.w_named,
            1 => self.w_named2,
            _ => return None,
        };
        NonNull::new(p.cast_mut()).map(NonNull::as_ptr)
    }

    /// The region, and the lane space it sits in: `total` is the fire's rows.
    pub fn rows(&self) -> Rows {
        Rows {
            start: i32::try_from(self.bound.rows.start).unwrap_or(0),
            count: self.rows,
            total: self.ctx.rows_total,
        }
    }

    pub(super) fn layer(&self) -> usize {
        self.layer_index()
    }

    /// `OpKind::Launch::params` — the wire scalars the statement carries.
    pub fn param(&self, i: usize) -> Option<u32> {
        self.spec.params.get(i).copied()
    }

    /// The engine's cuBLAS handle, with THIS fire's stream already bound.
    pub fn cublas(&self) -> *mut c_void {
        self.ctx.cublas
    }
}
