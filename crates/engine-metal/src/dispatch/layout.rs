//! The `layout` family: `impl DispatchLayout for Run<'_>`.

use kernels_metal::layout;
use model_exec::{DispatchLayout, KernelError};
use model_ir::Layout;

use crate::run::Run;

impl DispatchLayout for Run<'_> {
    fn dispatch(&mut self, op: &Layout) -> Result<(), KernelError> {
        self.layout(op).map_err(crate::error::kernel)
    }
}

impl Run<'_> {
    /// Errors here are `kernels-metal`'s vocabulary; [`kernel`](crate::error::kernel)
    /// lifts them at the call site.
    fn layout(&mut self, op: &Layout) -> Result<(), kernels_metal::Error> {
        match op {
            // Bank vs dense mirrors `linear.matmul`: the weight row's storage
            // form decides. Both arms are bound by `vocab`.
            Layout::Embed {
                ids,
                table,
                vocab,
                y,
            } => match self.banked(*table) {
                Some(bank) => layout::embed_gather_mb_4bit(
                    self.ctx(),
                    self.tensor(*ids),
                    bank,
                    *vocab,
                    self.tensor(*y),
                ),
                None => layout::embed(
                    self.ctx(),
                    self.tensor(*ids),
                    self.tensor(*table),
                    *vocab,
                    self.tensor(*y),
                ),
            },
            // Concatenating gather (qwen4 PLE): several hashed ids per token,
            // each gathering one row, laid side by side.
            //
            // Dense arm (`embed_concat_bfloat16`) contributes zero for an id
            // the table can't answer, where `Embed`'s dense arm clamps to
            // row zero.
            //
            // `guard` is the height of the bound slab, not `vocab`: an
            // out-of-range id maps to `seats`, landing on this same zero arm.
            Layout::EmbedConcat {
                ids,
                table,
                vocab,
                y,
            } => match self.banked(*table) {
                Some(bank) => {
                    let guard = bank.codes.rows.min(*vocab);
                    layout::embed_concat_mb_4bit(
                        self.ctx(),
                        self.tensor(*ids),
                        bank,
                        guard,
                        self.tensor(*y),
                    )
                }
                None => {
                    let plane = self.tensor(*table);
                    let guard = plane.rows.min(*vocab);
                    layout::embed_concat(
                        self.ctx(),
                        self.tensor(*ids),
                        plane,
                        guard,
                        self.tensor(*y),
                    )
                }
            },
            Layout::SplitQkv {
                packed,
                q_width,
                kv_width,
                q,
                k,
                v,
            } => layout::split_qkv(
                self.ctx(),
                self.tensor(*packed),
                *q_width,
                *kv_width,
                self.tensor(*q),
                self.tensor(*k),
                self.tensor(*v),
            ),
            Layout::SplitQGate {
                packed,
                head_dim,
                q,
                gate,
            } => layout::split_q_gate(
                self.ctx(),
                self.tensor(*packed),
                *head_dim,
                self.tensor(*q),
                self.tensor(*gate),
            ),
            Layout::SplitRows {
                x,
                width,
                left,
                right,
            } => layout::split_rows(
                self.ctx(),
                self.tensor(*x),
                *width,
                self.tensor(*left),
                self.tensor(*right),
            ),
            // gemma4's k×k soft-token fold: side² consecutive patch rows
            // averaged into one, computed in f32.
            Layout::PoolRows { x, side, y } => layout::pool_rows(
                self.ctx(),
                self.tensor(*x),
                *side,
                self.tensor(*y),
            ),
            // side² rows concatenated, not averaged: the one op here whose
            // destination is wider than its source (qwen's spatial merger).
            Layout::MergeRows { x, side, y } => layout::merge_rows(
                self.ctx(),
                self.tensor(*x),
                *side,
                self.tensor(*y),
            ),
            // Like `scatter_rows` below, but admits a `-1` route as a drop
            // sentinel; only this op's plans use `-1` in `patch_routes`.
            Layout::ScatterLiveRows {
                src,
                routes,
                y,
                y_out: _,
            } => layout::scatter_live_rows(
                self.ctx(),
                self.tensor(*src),
                self.tensor(*routes),
                // routes are absolute fire rows; a window-cut `y` would
                // double-count the offset.
                self.uncut(*y),
            ),
            // Interpolating gather: position table reads 2 taps, resample
            // reads 4 or 16 — `layout::embed` reads one row per id, unweighted.
            Layout::EmbedWeighted {
                ids,
                weights,
                table,
                vocab,
                y,
            } => layout::embed_weighted(
                self.ctx(),
                self.tensor(*ids),
                self.tensor(*weights),
                self.tensor(*table),
                *vocab,
                self.tensor(*y),
            ),
            // `src` resolves in the patch window, `y` in the token window.
            // `routes` is pre-checked host-side (`Fault::PatchRoute`): an
            // out-of-range entry here is an OOB device write the kernel
            // can't catch.
            Layout::ScatterRows {
                src,
                routes,
                y,
                y_out: _,
            } => layout::scatter_rows(
                self.ctx(),
                self.tensor(*src),
                self.tensor(*routes),
                // Whole, for `scatter_live_rows`' reason one arm up.
                self.uncut(*y),
            ),
            // One launch per operand: each writes its own column of the i32
            // plane, so the plane is whole once the last has run.
            Layout::Argmax { xs, y } => {
                for (column, x) in xs.iter().enumerate() {
                    layout::argmax(
                        self.ctx(),
                        self.tensor(*x),
                        u32::try_from(column).expect("a draft depth inside u32"),
                        self.tensor(*y),
                    )?;
                }
                Ok(())
            }
            Layout::Select {
                table,
                layer,
                width,
                y,
            } => layout::select(
                self.ctx(),
                self.tensor(*table),
                *layer,
                *width,
                self.tensor(*y),
            ),
        }
    }
}
