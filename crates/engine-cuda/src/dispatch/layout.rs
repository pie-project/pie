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
    fn layout(&mut self, op: &Layout) -> Result<(), kernels_cuda::Error> {
        match op {
            Layout::Embed {
                ids,
                table,
                vocab,
                y,
            } => match self.maybe_planes(*table) {
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
                None if self.tensor(*table).rows < *vocab => layout::embed_vocab_shard(
                    self.ctx(),
                    self.tensor(*ids),
                    self.tensor(*table),
                    &mut self.tensor(*y),
                ),
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
            Layout::MergeRows { x, side, y } => kernels_cuda::layout_fold::merge_rows(
                self.ctx(),
                self.tensor(*x),
                *side,
                &mut self.tensor(*y),
            ),
            Layout::ScatterLiveRows {
                src,
                routes,
                y,
                y_out: _,
            } => kernels_cuda::layout_scatter_live::scatter_live_rows(
                self.ctx(),
                self.tensor(*src),
                self.tensor(*routes),
                &mut self.fire_wide(*y),
            ),
            Layout::ScatterRows {
                src,
                routes,
                y,
                y_out: _,
            } => layout::scatter_rows(
                self.ctx(),
                self.tensor(*src),
                self.tensor(*routes),
                &mut self.fire_wide(*y),
            ),
            Layout::PackRows { x, perm, y } => layout::pack_rows(
                self.ctx(),
                self.fire_wide(*x),
                self.tensor(*perm),
                &mut self.tensor(*y),
            ),
            Layout::UnpackRows { x, perm, y } => {
                let packed = self.tensor(*x);
                let whole = self.fire_wide(*y);
                layout::unpack_rows(
                    self.ctx(),
                    packed,
                    self.tensor(*perm),
                    &mut kernels_cuda::Tensor::new(
                        whole.ptr,
                        packed.rows,
                        whole.width,
                        whole.dtype,
                    ),
                )
            }
            Layout::TopK {
                x,
                k,
                values,
                indices,
            } => layout::topk(
                self.ctx(),
                self.tensor(*x),
                *k,
                &mut self.tensor(*values),
                &mut self.tensor(*indices),
            ),
            Layout::Argmax { xs, y } => {
                for (column, x) in xs.iter().enumerate() {
                    layout::argmax(
                        self.ctx(),
                        self.tensor(*x),
                        u32::try_from(column).expect("a draft depth inside u32"),
                        &mut self.tensor(*y),
                    )?;
                }
                Ok(())
            }
            Layout::GatherRows { x, rows, y } => layout::gather_rows(
                self.ctx(),
                self.tensor(*x),
                self.tensor(*rows),
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
