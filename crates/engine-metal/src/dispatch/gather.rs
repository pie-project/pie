use kernels_metal::elemwise::gate;
use kernels_metal::gather;
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
    fn gather(&mut self, op: &Gather) -> Result<(), kernels_metal::Error> {
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
            }
            | Gather::NgramIdsChunked {
                ids,
                state,
                eos,
                mults,
                primes,
                offsets,
                heads_per_ngram,
                map,
                ngram_ids,
            } if self.rs_seat().is_some() => {
                const OP: &str = "gather.ngram_ids_committed";
                let seat = self.rs_seat().expect("guarded");
                let Some(hash) = self.ple_hash(mults, primes, offsets) else {
                    return Err(kernels_metal::Error::Unsupported { op: op.name() });
                };
                let ext_ids = self.rs_extend(OP, &seat, *ids)?;
                gather::ngram::ngram_ids_committed(
                    self.ctx(),
                    ext_ids,
                    self.qo_indptr(),
                    &self.rs_committed(&seat),
                    &self.recurrent(*state),
                    hash,
                    *eos,
                    mults,
                    primes,
                    offsets,
                    *heads_per_ngram,
                    map.map(|id| self.tensor(id)),
                    self.tensor(*ngram_ids),
                )
            }
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
                let Some(hash) = self.ple_hash(mults, primes, offsets) else {
                    return Err(kernels_metal::Error::Unsupported { op: op.name() });
                };
                gather::ngram::ngram_ids(
                    self.ctx(),
                    self.tensor(*ids),
                    &self.recurrent(*state),
                    hash,
                    *eos,
                    mults,
                    primes,
                    offsets,
                    *heads_per_ngram,
                    map.map(|id| self.tensor(id)),
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
                let Some(hash) = self.ple_hash(mults, primes, offsets) else {
                    return Err(kernels_metal::Error::Unsupported { op: op.name() });
                };
                gather::ngram::ngram_ids_chunked(
                    self.ctx(),
                    self.ragged(*ids),
                    &self.recurrent(*state),
                    hash,
                    *eos,
                    mults,
                    primes,
                    offsets,
                    *heads_per_ngram,
                    map.map(|id| self.tensor(id)),
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
                Some(bank) => gather::embed_gather_mb_4bit(
                    self.ctx(),
                    self.tensor(*ids),
                    bank,
                    *vocab,
                    self.tensor(*y),
                ),
                None => gather::embed(
                    self.ctx(),
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
            } => Err(kernels_metal::Error::Unsupported {
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
                    let guard = bank.codes.rows.min(*vocab);
                    gather::embed_concat_mb_4bit(
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
                    gather::embed_concat(
                        self.ctx(),
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
            } => gather::split_qkv(
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
            } => gate::split_q_gate(
                self.ctx(),
                self.tensor(*packed),
                *head_dim,
                self.tensor(*q),
                self.tensor(*gate),
            ),
            // A segment launch of the projection it splits landed both halves.
            Gather::SplitRows { left, .. } if self.landed_ahead(*left) => Ok(()),
            Gather::SplitRows {
                x,
                width,
                left,
                right,
            } => gather::split_rows(
                self.ctx(),
                self.tensor(*x),
                *width,
                self.tensor(*left),
                self.tensor(*right),
            ),
            Gather::PoolRows { x, side, y } => {
                gather::pool_rows(self.ctx(), self.tensor(*x), *side, self.tensor(*y))
            }
            Gather::MergeRows { x, side, y } => {
                gather::merge_rows(self.ctx(), self.tensor(*x), *side, self.tensor(*y))
            }
            Gather::Pixels {
                x,
                patch,
                mean,
                std,
                order,
                temporal,
                y,
            } => gather::pixels(
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
            Gather::ConcatCols { .. } => Err(kernels_metal::Error::Unsupported {
                op: "gather.concat_cols",
            }),
            Gather::SplitQkvPadded { .. } => Err(kernels_metal::Error::Unsupported {
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
            } => gather::grid_taps(
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
            } => gather::scatter_live_rows(
                self.ctx(),
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
            } => gather::embed_weighted(
                self.ctx(),
                self.tensor(*ids),
                self.tensor(*weights),
                self.tensor(*table),
                *vocab,
                self.tensor(*y),
            ),
            // The forms no builder makes.
            Gather::Embed { .. } => Err(kernels_metal::Error::Unsupported { op: op.name() }),
            Gather::ScatterRows {
                src,
                routes,
                y,
                y_out: _,
            } => gather::scatter_rows(
                self.ctx(),
                self.tensor(*src),
                self.tensor(*routes),
                self.uncut(*y),
            ),
            Gather::PackRows { x, perm, y } => gather::pack_rows(
                self.ctx(),
                self.uncut(*x),
                self.tensor(*perm),
                self.tensor(*y),
            ),
            Gather::UnpackRows { x, perm, y } => gather::unpack_rows(
                self.ctx(),
                self.tensor(*x),
                self.tensor(*perm),
                self.uncut(*y),
            ),
            Gather::Rows { x, rows, y } => gather::gather_rows(
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
            } => gather::select(
                self.ctx(),
                self.tensor(*table),
                *layer,
                *width,
                self.tensor(*y),
            ),
        }
    }
}
