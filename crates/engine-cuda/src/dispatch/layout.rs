use kernels_cuda::gather;
use poem_exec::{DispatchLayout, KernelError};
use poem_ir::Layout;

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
                Some((codes, ..))
                    if matches!(
                        codes.dtype,
                        poem_ir::Dtype::Nvfp4 | poem_ir::Dtype::E4m3tensor
                    ) =>
                {
                    Err(kernels_cuda::Error::Backend {
                        op: "layout.embed",
                        detail: "an nvfp4 table has no cuda gather; its rows are read as \
                                 bf16 or affine codes"
                            .to_string(),
                    })
                }
                Some((codes, scales, biases, seat)) => kernels_cuda::gather::embed_mlx_affine(
                    self.ctx(),
                    self.tensor(*ids),
                    codes,
                    scales,
                    biases,
                    *vocab,
                    seat,
                    &mut self.tensor(*y),
                ),
                None if self.tensor(*table).rows < *vocab => gather::embed_vocab_shard(
                    self.ctx(),
                    self.tensor(*ids),
                    self.tensor(*table),
                    &mut self.tensor(*y),
                ),
                None => gather::embed(
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
                scales: Some(scales),
                vocab,
                y,
            } => kernels_cuda::gather::embed_concat_mxfp8(
                self.ctx(),
                self.tensor(*ids),
                self.tensor(*table),
                self.tensor(*scales),
                *vocab,
                &mut self.tensor(*y),
            ),
            Layout::EmbedConcat {
                ids,
                table,
                scales: None,
                vocab,
                y,
            } => match self.maybe_planes(*table) {
                #[cfg(not(feature = "rocm"))]
                Some((codes, ..)) if codes.dtype == poem_ir::Dtype::Nvfp4 => {
                    Err(kernels_cuda::Error::Backend {
                        op: "layout.embed_concat",
                        detail: "an nvfp4 table has no cuda gather; its rows are read as \
                                 bf16 or affine codes"
                            .to_string(),
                    })
                }
                #[cfg(not(feature = "rocm"))]
                Some((codes, scales, _, _)) if codes.dtype == poem_ir::Dtype::E4m3tensor => {
                    kernels_cuda::gather::embed_concat_e4m3(
                        self.ctx(),
                        self.tensor(*ids),
                        codes,
                        scales,
                        *vocab,
                        &mut self.tensor(*y),
                    )
                }
                Some((codes, scales, biases, seat)) => kernels_cuda::gather::embed_concat_mlxu4(
                    self.ctx(),
                    self.tensor(*ids),
                    codes,
                    scales,
                    biases,
                    *vocab,
                    seat,
                    &mut self.tensor(*y),
                ),
                None => kernels_cuda::gather::embed_concat(
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
            } => gather::split_qkv(
                self.ctx(),
                self.tensor(*packed),
                *q_width,
                *kv_width,
                &mut self.tensor(*q),
                &mut self.tensor(*k),
                &mut self.tensor(*v),
            ),
            Layout::SplitQkvPadded {
                packed,
                heads,
                widths,
                head_dim,
                q,
                k,
                v,
            } => gather::split_qkv_padded(
                self.ctx(),
                self.tensor(*packed),
                *heads,
                *widths,
                *head_dim,
                &mut self.tensor(*q),
                &mut self.tensor(*k),
                &mut self.tensor(*v),
            ),
            Layout::SplitQGate {
                packed,
                head_dim,
                q,
                gate,
            } => gather::split_q_gate(
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
            } => gather::split_rows(
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
            } => kernels_cuda::gather::embed_weighted(
                self.ctx(),
                self.tensor(*ids),
                self.tensor(*weights),
                self.tensor(*table),
                *vocab,
                &mut self.tensor(*y),
            ),
            Layout::PoolRows { x, side, y } => kernels_cuda::gather::pool_rows(
                self.ctx(),
                self.tensor(*x),
                *side,
                &mut self.tensor(*y),
            ),
            Layout::MergeRows { x, side, y } => kernels_cuda::gather::merge_rows(
                self.ctx(),
                self.tensor(*x),
                *side,
                &mut self.tensor(*y),
            ),
            Layout::Pixels {
                x,
                patch,
                mean,
                std,
                order,
                temporal,
                y,
            } => kernels_cuda::gather::pixels(
                self.ctx(),
                self.tensor(*x),
                *patch,
                *mean,
                *std,
                *order == poem_ir::PixelOrder::Chw,
                *temporal,
                &mut self.tensor(*y),
            ),
            Layout::ConcatCols { xs, y } => {
                let xs: Vec<_> = xs.iter().map(|x| self.tensor(*x)).collect();
                kernels_cuda::gather::concat_cols(self.ctx(), &xs, &mut self.tensor(*y))
            }
            Layout::GridTaps {
                positions,
                grids,
                segments,
                kind,
                side,
                ids,
                weights,
            } => kernels_cuda::gather::grid_taps(
                self.ctx(),
                self.tensor(*positions),
                self.tensor(*grids),
                self.tensor(*segments),
                *kind == poem_ir::TapKind::Bilinear,
                *side,
                &mut self.tensor(*ids),
                &mut self.tensor(*weights),
            ),
            Layout::ScatterLiveRows {
                src,
                routes,
                y,
                y_out: _,
            } => kernels_cuda::gather::scatter_live_rows(
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
            } => gather::scatter_rows(
                self.ctx(),
                self.tensor(*src),
                self.tensor(*routes),
                &mut self.fire_wide(*y),
            ),
            Layout::PackRows { x, perm, y } => gather::pack_rows(
                self.ctx(),
                self.fire_wide(*x),
                self.tensor(*perm),
                &mut self.tensor(*y),
            ),
            Layout::UnpackRows { x, perm, y } => {
                let packed = self.tensor(*x);
                let whole = self.fire_wide(*y);
                gather::unpack_rows(
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
            } => kernels_cuda::sampling::topk(
                self.ctx(),
                self.tensor(*x),
                *k,
                &mut self.tensor(*values),
                &mut self.tensor(*indices),
            ),
            Layout::Argmax { xs, y } => {
                for (column, x) in xs.iter().enumerate() {
                    kernels_cuda::sampling::argmax(
                        self.ctx(),
                        self.tensor(*x),
                        u32::try_from(column).expect("a draft depth inside u32"),
                        &mut self.tensor(*y),
                    )?;
                }
                Ok(())
            }
            Layout::GatherRows { x, rows, y } => gather::gather_rows(
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
            } => gather::select(
                self.ctx(),
                self.tensor(*table),
                *layer,
                *width,
                &mut self.tensor(*y),
            ),
        }
    }
}
