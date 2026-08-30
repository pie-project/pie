//! `Layout`: the embedding table read and the packed-row splits.

use kernels_cuda::layout;
use model_exec::{DispatchLayout, KernelError};
use model_ir::Layout;

use crate::run::Run;

impl DispatchLayout for Run<'_> {
    fn dispatch(&mut self, op: &Layout) -> Result<(), KernelError> {
        self.layout(op).map_err(crate::error::kernel)
    }
}

impl Run<'_> {
    /// The arms themselves, in `kernels-cuda`'s error vocabulary and not
    /// the contract's — which is what keeps each one a plain tail call with
    /// a plain `?`. [`kernel`](crate::error::kernel) is the single line
    /// above that lifts the family, and says why it is a call and not a
    /// `From` impl.
    fn layout(&mut self, op: &Layout) -> Result<(), kernels_cuda::Error> {
        match op {
            Layout::Embed {
                ids,
                table,
                vocab,
                y,
            } => layout::embed(
                self.ctx(),
                self.tensor(*ids),
                self.tensor(*table),
                *vocab,
                &mut self.tensor(*y),
            ),
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
                &mut self.tensor(*q),
                &mut self.tensor(*k),
                &mut self.tensor(*v),
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
                &mut self.tensor(*q),
                &mut self.tensor(*gate),
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
                &mut self.tensor(*left),
                &mut self.tensor(*right),
            ),
            // **THE SPATIAL POOL** (multimodal §6.5, §7.4), through the
            // `#[path]`-rehomed unit: the file is `layout/pool.rs`, the module
            // is `kernels_cuda::layout_pool`, and `layout.rs` is closed to the
            // wave that wrote it.
            // **THE GATHER THAT INTERPOLATES** (multimodal §9.2), through its
            // own `#[path]`-rehomed unit. The plain `layout.embed` arm above
            // serves the native grid; this one serves the resampled table, and
            // a text picks by which op it writes.
            Layout::EmbedWeighted {
                ids,
                weights,
                table,
                vocab,
                y,
            } => kernels_cuda::layout_embed_weighted::embed_weighted(
                self.ctx(),
                self.tensor(*ids),
                self.tensor(*weights),
                self.tensor(*table),
                *vocab,
                &mut self.tensor(*y),
            ),
            Layout::PoolRows { x, side, y } => kernels_cuda::layout_fold::pool_rows(
                self.ctx(),
                self.tensor(*x),
                *side,
                &mut self.tensor(*y),
            ),
            // **THE MERGING FOLD** (multimodal §8.1), the same unit's other
            // entry: `side²` rows concatenated rather than averaged, and the
            // one op here whose destination is wider than its source.
            Layout::MergeRows { x, side, y } => kernels_cuda::layout_fold::merge_rows(
                self.ctx(),
                self.tensor(*x),
                *side,
                &mut self.tensor(*y),
            ),
            // **THE EMBED MERGE WITH A DROP SENTINEL** (multimodal §8.6),
            // through its own `#[path]`-rehomed unit: `scatter_rows` below,
            // plus a guard on a negative route. The shell admits a `-1` in
            // `patch_routes` only for a plan that names THIS op, so the arm
            // below keeps the contract it always had.
            Layout::ScatterLiveRows {
                src,
                routes,
                y,
                y_out: _,
            } => kernels_cuda::layout_scatter_live::scatter_live_rows(
                self.ctx(),
                self.tensor(*src),
                self.tensor(*routes),
                &mut self.tensor(*y),
            ),
            // THE EMBED MERGE. `src` resolves at the PATCH window and `y` at
            // the token one — the one node in a tower plan whose two operands
            // come out of two seriations — and the routes vector has already
            // been checked against this fire's token row count host-side
            // (`Fault::PatchRoute`), because an out-of-range entry here is an
            // out-of-bounds device write the kernel cannot see.
            Layout::ScatterRows {
                src,
                routes,
                y,
                y_out: _,
            } => layout::scatter_rows(
                self.ctx(),
                self.tensor(*src),
                self.tensor(*routes),
                &mut self.tensor(*y),
            ),
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
                &mut self.tensor(*y),
            ),
        }
    }
}
