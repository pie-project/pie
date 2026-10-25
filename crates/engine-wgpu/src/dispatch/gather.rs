use poem_exec::{DispatchGather, KernelError};
use poem_ir::{Gather, Operands};

use crate::run::Run;

impl DispatchGather for Run<'_> {
    fn dispatch(&mut self, op: &Gather) -> Result<(), KernelError> {
        self.gather(op).map_err(crate::error::kernel)
    }
}

impl Run<'_> {
    fn gather(&mut self, op: &Gather) -> Result<(), kernels_slang::Error> {
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
                    return Err(kernels_slang::Error::Unsupported { op: op.name() });
                }
                let Some(hash) = self.ple_hash(mults, primes, offsets) else {
                    return Err(kernels_slang::Error::Unsupported { op: op.name() });
                };
                let pool = self.recurrent(*state);
                kernels_slang::gather::ngram::ngram_ids(
                    self.ctx,
                    self.tensor(*ids),
                    pool.state,
                    pool.slots,
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
                    return Err(kernels_slang::Error::Unsupported { op: op.name() });
                }
                let Some(hash) = self.ple_hash(mults, primes, offsets) else {
                    return Err(kernels_slang::Error::Unsupported { op: op.name() });
                };
                let ids = self.ragged(*ids);
                let pool = self.recurrent(*state);
                kernels_slang::gather::ngram::ngram_ids_chunked(
                    self.ctx,
                    ids.data,
                    ids.indptr,
                    pool.state,
                    pool.slots,
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
                weights: None,
                scales: None,
                concat: false,
                ..
            } => match self.banked(*table) {
                Some(bank) => kernels_slang::gather::embed::embed_banked(
                    self.ctx,
                    self.tensor(*ids),
                    bank,
                    *vocab,
                    self.tensor(*y),
                ),
                None => kernels_slang::gather::embed::embed(
                    self.ctx,
                    self.tensor(*ids),
                    self.tensor(*table),
                    *vocab,
                    self.tensor(*y),
                ),
            },

            Gather::Embed {
                scales: Some(_),
                weights: None,
                concat: true,
                ..
            } => Err(kernels_slang::Error::Unsupported {
                op: "gather.embed(mxfp8)",
            }),
            Gather::Embed {
                ids,
                table,
                scales: None,
                vocab,
                y,
                weights: None,
                concat: true,
            } => match self.banked(*table) {
                Some(bank) => {
                    let (bank, ids) = match self.gathered_table(bank, self.tensor(*ids))? {
                        Some(seated) => seated,
                        None => (bank, self.tensor(*ids)),
                    };
                    let guard = bank.codes.rows.min(*vocab);
                    kernels_slang::gather::embed::embed_concat_banked(
                        self.ctx,
                        ids,
                        bank,
                        guard,
                        self.tensor(*y),
                    )
                }
                None => {
                    let plane = self.tensor(*table);
                    let guard = plane.rows.min(*vocab);
                    kernels_slang::gather::embed::embed_concat(
                        self.ctx,
                        self.tensor(*ids),
                        plane,
                        guard,
                        self.tensor(*y),
                    )
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
                self.ctx,
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
            } => kernels_slang::gather::rows::split_q_gate(
                self.ctx,
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
            } => kernels_slang::gather::rows::split_rows(
                self.ctx,
                self.tensor(*x),
                *width,
                self.tensor(*left),
                self.tensor(*right),
            ),

            Gather::PoolRows { x, side, y } => kernels_slang::gather::rows::pool_rows(
                self.ctx,
                self.tensor(*x),
                *side,
                self.tensor(*y),
            ),

            Gather::MergeRows { x, side, y } => kernels_slang::gather::rows::merge_rows(
                self.ctx,
                self.tensor(*x),
                *side,
                self.tensor(*y),
            ),

            Gather::Pixels {
                x,
                patch,
                mean,
                std,
                order,
                temporal,
                y,
            } => kernels_slang::gather::pixels::pixels(
                self.ctx,
                self.tensor(*x),
                *patch,
                *mean,
                *std,
                *order == poem_ir::PixelOrder::Chw,
                *temporal,
                self.tensor(*y),
            ),

            Gather::GroupRows {
                csr,
                groups,
                lanes,
                y,
            } => kernels_slang::conv::rule::group_rows(
                self.ctx,
                self.tensor(*csr),
                self.tensor(*groups),
                self.tensor(*lanes),
                self.tensor(*y),
            ),
            Gather::ConcatCols { xs, y } => {
                let xs: Vec<_> = xs.iter().map(|x| self.tensor(*x)).collect();
                kernels_slang::gather::rows::concat_cols(self.ctx, &xs, self.tensor(*y))
            }
            Gather::SplitQkvPadded { .. } => {
                Err(kernels_slang::Error::Unsupported { op: op.name() })
            }
            Gather::GridTaps {
                positions,
                grids,
                segments,
                kind,
                side,
                ids,
                weights,
            } => kernels_slang::gather::pixels::grid_taps(
                self.ctx,
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
                self.ctx,
                self.tensor(*src),
                self.tensor(*routes),
                self.uncut(*y),
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
            } => kernels_slang::gather::embed::embed_weighted(
                self.ctx,
                self.tensor(*ids),
                self.tensor(*weights),
                self.tensor(*table),
                *vocab,
                self.tensor(*y),
            ),

            // The forms no builder makes.
            Gather::Embed { .. } => Err(kernels_slang::Error::Unsupported { op: op.name() }),
            Gather::ScatterRows {
                src,
                routes,
                y,
                y_out: _,
            } => kernels_slang::gather::rows::scatter_rows(
                self.ctx,
                self.tensor(*src),
                self.tensor(*routes),
                self.uncut(*y),
            ),

            Gather::PackRows { x, perm, y } => kernels_slang::gather::rows::pack_rows(
                self.ctx,
                self.uncut(*x),
                self.tensor(*perm),
                self.tensor(*y),
            ),
            Gather::UnpackRows { x, perm, y } => kernels_slang::gather::rows::unpack_rows(
                self.ctx,
                self.tensor(*x),
                self.tensor(*perm),
                self.uncut(*y),
            ),
            Gather::Rows { x, rows, y } => kernels_slang::gather::rows::gather_rows(
                self.ctx,
                self.uncut(*x),
                self.tensor(*rows),
                self.tensor(*y),
            ),
            Gather::Select {
                table,
                layer,
                width,
                y,
            } => kernels_slang::gather::rows::select(
                self.ctx,
                self.tensor(*table),
                *layer,
                *width,
                self.tensor(*y),
            ),
        }
    }
}
