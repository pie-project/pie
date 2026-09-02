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
    /// Arms in `kernels-cuda`'s error vocabulary, lifted by
    /// [`kernel`](crate::error::kernel) above the match.
    fn layout(&mut self, op: &Layout) -> Result<(), kernels_cuda::Error> {
        match op {
            Layout::Embed {
                ids,
                table,
                vocab,
                y,
            } => match self.maybe_planes(*table) {
                // Affine-quantized table (e.g. qwen4's 8-bit): dequantized
                // only for the rows this step touches.
                Some((codes, scales, biases, seat)) => {
                    kernels_cuda::layout_embed_concat::embed_mlx_affine(
                        self.ctx(),
                        self.tensor(*ids),
                        codes,
                        scales,
                        biases,
                        *vocab,
                        seat,
                        &mut self.tensor(*y),
                    )
                }
                None => layout::embed(
                    self.ctx(),
                    self.tensor(*ids),
                    self.tensor(*table),
                    *vocab,
                    &mut self.tensor(*y),
                ),
            },
            Layout::EmbedConcat {
                ids,
                table,
                vocab,
                y,
            } => match self.maybe_planes(*table) {
                // Large table lands as its affine triplet; the gather
                // dequantizes only the rows touched per token.
                Some((codes, scales, biases, seat)) => {
                    kernels_cuda::layout_embed_concat::embed_concat_mlxu4(
                        self.ctx(),
                        self.tensor(*ids),
                        codes,
                        scales,
                        biases,
                        *vocab,
                        seat,
                        &mut self.tensor(*y),
                    )
                }
                None => kernels_cuda::layout_embed_concat::embed_concat(
                    self.ctx(),
                    self.tensor(*ids),
                    self.tensor(*table),
                    *vocab,
                    &mut self.tensor(*y),
                ),
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
            // Interpolating gather: the resampled table, vs the
            // native-grid embed arm above.
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
            // side² rows concatenated rather than averaged; the one op here
            // whose destination is wider than its source.
            Layout::MergeRows { x, side, y } => kernels_cuda::layout_fold::merge_rows(
                self.ctx(),
                self.tensor(*x),
                *side,
                &mut self.tensor(*y),
            ),
            // Embed merge with a drop sentinel: the shell admits -1 in
            // patch_routes only for a plan that names this op.
            Layout::ScatterLiveRows {
                src,
                routes,
                y,
                y_out: _,
            } => kernels_cuda::layout_scatter_live::scatter_live_rows(
                self.ctx(),
                self.tensor(*src),
                self.tensor(*routes),
                // Routes are absolute fire rows; a window-cut `y` would
                // double-count the region's offset.
                &mut self.fire_wide(*y),
            ),
            // src resolves at the patch window, y at the token window — the
            // one node whose operands come from two seriations. routes are
            // already validated host-side (Fault::PatchRoute).
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
                &mut self.fire_wide(*y),
            ),
            Layout::Argmax { .. } => {
                return Err(kernels_cuda::Error::Backend {
                    op: "layout.argmax",
                    detail: "the per-row argmax a draft chain feeds itself is not yet read on \
                             the CUDA arm"
                        .to_string(),
                });
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
                &mut self.tensor(*y),
            ),
        }
    }
}
