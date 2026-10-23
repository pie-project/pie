use kernels_cuda::gather;
use poem_exec::{DispatchGather, KernelError};
use poem_ir::{Gather, Operands};

use crate::run::Run;

impl DispatchGather for Run<'_> {
    fn dispatch(&mut self, op: &Gather) -> Result<(), KernelError> {
        self.gather(op).map_err(crate::error::kernel)
    }
}

impl Run<'_> {
    fn gather(&mut self, op: &Gather) -> Result<(), kernels_cuda::Error> {
        match op {
            Gather::NgramIds {
                ids,
                state,
                eos,
                mults,
                primes,
                offsets,
                heads_per_ngram,
                map,
                ngram_ids,
            } => kernels_cuda::gather::ngram_ids(
                self.ctx(),
                self.tensor(*ids),
                &self.recurrent(*state),
                *eos,
                mults,
                primes,
                offsets,
                *heads_per_ngram,
                map.map(|id| self.tensor(id)),
                &mut self.tensor(*ngram_ids),
            ),
            Gather::NgramIdsChunked {
                ids,
                state,
                eos,
                mults,
                primes,
                offsets,
                heads_per_ngram,
                map,
                ngram_ids,
            } => {
                let tail = self.recurrent_tail_absolute(*state);
                kernels_cuda::gather::ngram_ids_chunked(
                    self.ctx(),
                    self.ragged_lanes(*ids),
                    &self.recurrent_absolute(*state),
                    *eos,
                    mults,
                    primes,
                    offsets,
                    *heads_per_ngram,
                    map.map(|id| self.tensor(id)),
                    &mut self.tensor(*ngram_ids),
                )?;
                let Some(tail) = tail else { return Ok(()) };
                kernels_cuda::gather::ngram_ids_chunked(
                    self.ctx(),
                    self.ragged_lanes(*ids),
                    &tail,
                    *eos,
                    mults,
                    primes,
                    offsets,
                    *heads_per_ngram,
                    map.map(|id| self.tensor(id)),
                    &mut self.tensor(*ngram_ids),
                )
            }
            Gather::Embed {
                ids,
                table,
                vocab,
                y,
                weights: None,
                scales: None,
                concat: false,
                ..
            } => match self.maybe_planes(*table) {
                Some((codes, ..))
                    if matches!(
                        codes.dtype,
                        poem_ir::Dtype::Nvfp4 | poem_ir::Dtype::E4m3tensor
                    ) =>
                {
                    Err(kernels_cuda::Error::Backend {
                        op: "gather.embed",
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
            Gather::Embed {
                ids,
                table,
                scales: Some(scales),
                vocab,
                y,
                weights: None,
                concat: true,
            } => kernels_cuda::gather::embed_concat_mxfp8(
                self.ctx(),
                self.tensor(*ids),
                self.tensor(*table),
                self.tensor(*scales),
                *vocab,
                &mut self.tensor(*y),
            ),
            Gather::Embed {
                ids,
                table,
                scales: None,
                vocab,
                y,
                weights: None,
                concat: true,
            } => match self.maybe_planes(*table) {
                #[cfg(not(feature = "rocm"))]
                Some((codes, ..)) if codes.dtype == poem_ir::Dtype::Nvfp4 => {
                    Err(kernels_cuda::Error::Backend {
                        op: "gather.embed",
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
            Gather::SplitQkv {
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
            Gather::SplitQkvPadded {
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
            Gather::SplitQGate {
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
            Gather::SplitRows {
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
            Gather::Embed {
                ids,
                weights: Some(weights),
                table,
                vocab,
                y,
                scales: None,
                concat: false,
                ..
            } => kernels_cuda::gather::embed_weighted(
                self.ctx(),
                self.tensor(*ids),
                self.tensor(*weights),
                self.tensor(*table),
                *vocab,
                &mut self.tensor(*y),
            ),
            // The forms no builder makes.
            Gather::Embed { .. } => Err(kernels_cuda::Error::Unsupported { op: op.name() }),
            Gather::PoolRows { x, side, y } => kernels_cuda::gather::pool_rows(
                self.ctx(),
                self.tensor(*x),
                *side,
                &mut self.tensor(*y),
            ),
            Gather::MergeRows { x, side, y } => kernels_cuda::gather::merge_rows(
                self.ctx(),
                self.tensor(*x),
                *side,
                &mut self.tensor(*y),
            ),
            Gather::Pixels {
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
            Gather::ConcatCols { xs, y } => {
                let xs: Vec<_> = xs.iter().map(|x| self.tensor(*x)).collect();
                kernels_cuda::gather::concat_cols(self.ctx(), &xs, &mut self.tensor(*y))
            }
            Gather::GridTaps {
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
            Gather::ScatterLiveRows {
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
            Gather::ScatterRows {
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
            Gather::PackRows { x, perm, y } => gather::pack_rows(
                self.ctx(),
                self.fire_wide(*x),
                self.tensor(*perm),
                &mut self.tensor(*y),
            ),
            Gather::UnpackRows { x, perm, y } => {
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
            Gather::Rows { x, rows, y } => gather::gather_rows(
                self.ctx(),
                self.tensor(*x),
                self.tensor(*rows),
                &mut self.tensor(*y),
            ),
            Gather::Select {
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
