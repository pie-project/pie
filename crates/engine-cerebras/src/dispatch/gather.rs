use kernels_cerebras::attn;
use kernels_cerebras::layout;
use poem_exec::{DispatchGather, KernelError};
use poem_ir::{Gather, Operands};

use crate::run::Run;

impl DispatchGather for Run<'_> {
    fn dispatch(&mut self, op: &Gather) -> Result<(), KernelError> {
        self.ctx().scope(op.name());
        self.gather(op).map_err(crate::error::kernel)
    }
}

impl Run<'_> {
    fn gather(&mut self, op: &Gather) -> Result<(), kernels_cerebras::Error> {
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
            Gather::EmbedConcat {
                ids,
                table,
                scales: None,
                vocab,
                y,
            } => layout::embed_concat(
                self.ctx(),
                self.tensor(*ids),
                self.tensor(*table),
                *vocab,
                self.tensor(*y),
            ),
            Gather::Embed {
                ids,
                table,
                vocab,
                y,
            } => layout::embed(
                self.ctx(),
                self.tensor(*ids),
                self.tensor(*table),
                *vocab,
                self.tensor(*y),
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
            Gather::MergeRows { x, side, y } => {
                layout::merge_rows(self.ctx(), self.tensor(*x), *side, self.tensor(*y))
            }
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
            Gather::Rows { x, rows, y } => layout::gather_rows(
                self.ctx(),
                self.uncut(*x),
                self.tensor(*rows),
                self.tensor(*y),
            ),
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
            other => Err(kernels_cerebras::Error::Unsupported { op: other.name() }),
        }
    }
}
