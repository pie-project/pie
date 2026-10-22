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
            } => kernels_slang::gather::rows::split_qkv(
                &self.slang(),
                crate::encode::slang_tensor(self.tensor(*packed)),
                *q_width,
                *kv_width,
                crate::encode::slang_tensor(self.tensor(*q)),
                crate::encode::slang_tensor(self.tensor(*k)),
                crate::encode::slang_tensor(self.tensor(*v)),
            )
            .map_err(crate::encode::from_slang),
            Gather::SplitQGate {
                packed,
                head_dim,
                q,
                gate,
            } => kernels_slang::gather::rows::split_q_gate(
                &self.slang(),
                crate::encode::slang_tensor(self.tensor(*packed)),
                *head_dim,
                crate::encode::slang_tensor(self.tensor(*q)),
                crate::encode::slang_tensor(self.tensor(*gate)),
            )
            .map_err(crate::encode::from_slang),
            Gather::SplitRows {
                x,
                width,
                left,
                right,
            } => kernels_slang::gather::rows::split_rows(
                &self.slang(),
                crate::encode::slang_tensor(self.tensor(*x)),
                *width,
                crate::encode::slang_tensor(self.tensor(*left)),
                crate::encode::slang_tensor(self.tensor(*right)),
            )
            .map_err(crate::encode::from_slang),

            Gather::PoolRows { x, side, y } => kernels_slang::gather::rows::pool_rows(
                &self.slang(),
                crate::encode::slang_tensor(self.tensor(*x)),
                *side,
                crate::encode::slang_tensor(self.tensor(*y)),
            )
            .map_err(crate::encode::from_slang),

            Gather::MergeRows { x, side, y } => kernels_slang::gather::rows::merge_rows(
                &self.slang(),
                crate::encode::slang_tensor(self.tensor(*x)),
                *side,
                crate::encode::slang_tensor(self.tensor(*y)),
            )
            .map_err(crate::encode::from_slang),

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
            } => kernels_slang::gather::rows::scatter_live_rows(
                &self.slang(),
                crate::encode::slang_tensor(self.tensor(*src)),
                crate::encode::slang_tensor(self.tensor(*routes)),
                crate::encode::slang_tensor(self.uncut(*y)),
            )
            .map_err(crate::encode::from_slang),

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
            } => kernels_slang::gather::rows::scatter_rows(
                &self.slang(),
                crate::encode::slang_tensor(self.tensor(*src)),
                crate::encode::slang_tensor(self.tensor(*routes)),
                crate::encode::slang_tensor(self.uncut(*y)),
            )
            .map_err(crate::encode::from_slang),

            Gather::PackRows { x, perm, y } => kernels_slang::gather::rows::pack_rows(
                &self.slang(),
                crate::encode::slang_tensor(self.uncut(*x)),
                crate::encode::slang_tensor(self.tensor(*perm)),
                crate::encode::slang_tensor(self.tensor(*y)),
            )
            .map_err(crate::encode::from_slang),
            Gather::UnpackRows { x, perm, y } => kernels_slang::gather::rows::unpack_rows(
                &self.slang(),
                crate::encode::slang_tensor(self.tensor(*x)),
                crate::encode::slang_tensor(self.tensor(*perm)),
                crate::encode::slang_tensor(self.uncut(*y)),
            )
            .map_err(crate::encode::from_slang),
            Gather::Rows { x, rows, y } => kernels_slang::gather::rows::gather_rows(
                &self.slang(),
                crate::encode::slang_tensor(self.uncut(*x)),
                crate::encode::slang_tensor(self.tensor(*rows)),
                crate::encode::slang_tensor(self.tensor(*y)),
            )
            .map_err(crate::encode::from_slang),
            Gather::Select {
                table,
                layer,
                width,
                y,
            } => kernels_slang::gather::rows::select(
                &self.slang(),
                crate::encode::slang_tensor(self.tensor(*table)),
                *layer,
                *width,
                crate::encode::slang_tensor(self.tensor(*y)),
            )
            .map_err(crate::encode::from_slang),
        }
    }
}
