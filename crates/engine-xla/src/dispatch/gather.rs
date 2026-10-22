use kernels_xla::attn;
use kernels_xla::layout;
use poem_exec::{DispatchGather, KernelError};
use poem_ir::Gather;

use poem_ir::Operands;

use crate::run::Run;

impl DispatchGather for Run<'_> {
    fn dispatch(&mut self, op: &Gather) -> Result<(), KernelError> {
        self.ctx().scope(op.name());
        self.gather(op).map_err(crate::error::kernel)
    }
}

impl Run<'_> {
    fn gather(&mut self, op: &Gather) -> Result<(), kernels_xla::Error> {
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
            } => attn::ple::ngram_ids(
                self.ctx(),
                self.tensor(*ids),
                &self.recurrent(*state),
                *eos,
                mults,
                primes,
                offsets,
                *heads_per_ngram,
                map.map(|m| self.tensor(m)),
                self.tensor(*ngram_ids),
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
            } => attn::ple::ngram_ids_chunked(
                self.ctx(),
                self.ragged(*ids),
                &self.recurrent(*state),
                *eos,
                mults,
                primes,
                offsets,
                *heads_per_ngram,
                map.map(|m| self.tensor(m)),
                self.tensor(*ngram_ids),
            ),
            Gather::Embed {
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
                None if self.tensor(*table).rows < *vocab => layout::embed_vocab_shard(
                    self.ctx(),
                    self.tensor(*ids),
                    self.tensor(*table),
                    self.rank(),
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

            Gather::EmbedConcat {
                scales: Some(_), ..
            } => Err(kernels_xla::Error::Unsupported {
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
                    let ids = self.tensor(*ids);
                    let width = self.tensor(*y).width / ids.width.max(1);
                    let guard = layout::table_rows(&bank, width).min(*vocab);
                    layout::embed_concat_mb_4bit(self.ctx(), ids, bank, guard, self.tensor(*y))
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

            Gather::ConcatCols { .. } | Gather::SplitQkvPadded { .. } => {
                Err(kernels_xla::Error::Unsupported { op: op.name() })
            }
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
            } => layout::embed_weighted(
                self.ctx(),
                self.tensor(*ids),
                self.tensor(*weights),
                self.tensor(*table),
                *vocab,
                self.tensor(*y),
            ),

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
