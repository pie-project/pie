use kernels_vulkan::attn;
use kernels_vulkan::layout;
use poem_exec::{DispatchGather, KernelError};
use poem_ir::Gather;
use poem_ir::Operands;

use crate::run::Run;

impl DispatchGather for Run<'_> {
    fn dispatch(&mut self, op: &Gather) -> Result<(), KernelError> {
        self.gather(op).map_err(crate::error::kernel)
    }
}

impl Run<'_> {
    fn gather(&mut self, op: &Gather) -> Result<(), kernels_vulkan::Error> {
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
            } => {
                if map.is_some() {
                    // Engram's id map is not carried by this backend's hasher yet.
                    return Err(kernels_vulkan::Error::Unsupported { op: op.name() });
                }
                let Some(hash) = self.ple_hash(mults, primes, offsets) else {
                    return Err(kernels_vulkan::Error::Unsupported { op: op.name() });
                };
                attn::ple::ngram_ids(
                    self.ctx(),
                    self.tensor(*ids),
                    &self.recurrent(*state),
                    hash,
                    *eos,
                    mults,
                    primes,
                    offsets,
                    *heads_per_ngram,
                    self.tensor(*ngram_ids),
                )
            }
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
                if map.is_some() {
                    // Engram's id map is not carried by this backend's hasher yet.
                    return Err(kernels_vulkan::Error::Unsupported { op: op.name() });
                }
                let Some(hash) = self.ple_hash(mults, primes, offsets) else {
                    return Err(kernels_vulkan::Error::Unsupported { op: op.name() });
                };
                attn::ple::ngram_ids_chunked(
                    self.ctx(),
                    self.ragged(*ids),
                    &self.recurrent(*state),
                    hash,
                    *eos,
                    mults,
                    primes,
                    offsets,
                    *heads_per_ngram,
                    self.tensor(*ngram_ids),
                )
            }
            Gather::Embed {
                ids,
                table,
                vocab,
                y,
            } => match self.banked(*table) {
                Some(bank) => kernels_slang::gather::embed::embed_banked(
                    &self.slang(),
                    crate::encode::slang_tensor(self.tensor(*ids)),
                    crate::encode::slang_bank(bank),
                    *vocab,
                    crate::encode::slang_tensor(self.tensor(*y)),
                )
                .map_err(crate::encode::from_slang),
                None => kernels_slang::gather::embed::embed(
                    &self.slang(),
                    crate::encode::slang_tensor(self.tensor(*ids)),
                    crate::encode::slang_tensor(self.tensor(*table)),
                    *vocab,
                    crate::encode::slang_tensor(self.tensor(*y)),
                )
                .map_err(crate::encode::from_slang),
            },

            Gather::EmbedConcat {
                scales: Some(_), ..
            } => Err(kernels_vulkan::Error::Unsupported {
                op: "gather.embed_concat(mxfp8)",
            }),
            Gather::EmbedConcat {
                ids,
                table,
                scales: None,
                vocab,
                y,
            } => match self.banked(*table) {
                Some(bank) => {
                    let (bank, ids) = match self.gathered_table(bank, self.tensor(*ids))? {
                        Some(seated) => seated,
                        None => (bank, self.tensor(*ids)),
                    };
                    let guard = bank.codes.rows.min(*vocab);
                    kernels_slang::gather::embed::embed_concat_banked(
                        &self.slang(),
                        crate::encode::slang_tensor(ids),
                        crate::encode::slang_bank(bank),
                        guard,
                        crate::encode::slang_tensor(self.tensor(*y)),
                    )
                    .map_err(crate::encode::from_slang)
                }
                None => {
                    let plane = self.tensor(*table);
                    let guard = plane.rows.min(*vocab);
                    kernels_slang::gather::embed::embed_concat(
                        &self.slang(),
                        crate::encode::slang_tensor(self.tensor(*ids)),
                        crate::encode::slang_tensor(plane),
                        guard,
                        crate::encode::slang_tensor(self.tensor(*y)),
                    )
                    .map_err(crate::encode::from_slang)
                }
            },
            Gather::SplitQkv {
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
            Gather::SplitQGate {
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
            Gather::SplitRows {
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

            Gather::PoolRows { x, side, y } => {
                layout::pool_rows(self.ctx(), self.tensor(*x), *side, self.tensor(*y))
            }

            Gather::MergeRows { x, side, y } => {
                layout::merge_rows(self.ctx(), self.tensor(*x), *side, self.tensor(*y))
            }

            Gather::Pixels {
                x,
                patch,
                mean,
                std,
                order,
                temporal,
                y,
            } => layout::pixels(
                self.ctx(),
                self.tensor(*x),
                *patch,
                *mean,
                *std,
                *order == poem_ir::PixelOrder::Chw,
                *temporal,
                self.tensor(*y),
            ),

            // DSpark's taps side by side are DeepSeek-V4.1's, served on CUDA.
            Gather::ConcatCols { .. } => Err(kernels_vulkan::Error::Unsupported {
                op: "gather.concat_cols",
            }),
            Gather::SplitQkvPadded { .. } => Err(kernels_vulkan::Error::Unsupported {
                op: "gather.split_qkv_padded",
            }),
            Gather::GridTaps {
                positions,
                grids,
                segments,
                kind,
                side,
                ids,
                weights,
            } => layout::grid_taps(
                self.ctx(),
                self.tensor(*positions),
                self.tensor(*grids),
                self.tensor(*segments),
                *kind == poem_ir::TapKind::Bilinear,
                *side,
                self.tensor(*ids),
                self.tensor(*weights),
            ),

            Gather::ScatterLiveRows {
                src,
                routes,
                y,
                y_out: _,
            } => layout::scatter_live_rows(
                self.ctx(),
                self.tensor(*src),
                self.tensor(*routes),
                self.uncut(*y),
            ),

            Gather::EmbedWeighted {
                ids,
                weights,
                table,
                vocab,
                y,
            } => kernels_slang::gather::embed::embed_weighted(
                &self.slang(),
                crate::encode::slang_tensor(self.tensor(*ids)),
                crate::encode::slang_tensor(self.tensor(*weights)),
                crate::encode::slang_tensor(self.tensor(*table)),
                *vocab,
                crate::encode::slang_tensor(self.tensor(*y)),
            )
            .map_err(crate::encode::from_slang),

            Gather::ScatterRows {
                src,
                routes,
                y,
                y_out: _,
            } => layout::scatter_rows(
                self.ctx(),
                self.tensor(*src),
                self.tensor(*routes),
                self.uncut(*y),
            ),

            Gather::PackRows { x, perm, y } => layout::pack_rows(
                self.ctx(),
                self.uncut(*x),
                self.tensor(*perm),
                self.tensor(*y),
            ),
            Gather::UnpackRows { x, perm, y } => layout::unpack_rows(
                self.ctx(),
                self.tensor(*x),
                self.tensor(*perm),
                self.uncut(*y),
            ),
            Gather::Rows { x, rows, y } => layout::gather_rows(
                self.ctx(),
                self.uncut(*x),
                self.tensor(*rows),
                self.tensor(*y),
            ),
            Gather::Select {
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
