use kernels_cerebras::layout;
use model_exec::{DispatchLayout, KernelError};
use model_ir::{Layout, Operands};

use crate::run::Run;

impl DispatchLayout for Run<'_> {
    fn dispatch(&mut self, op: &Layout) -> Result<(), KernelError> {
        self.ctx().scope(op.name());
        self.layout(op).map_err(crate::error::kernel)
    }
}

impl Run<'_> {
    fn layout(&mut self, op: &Layout) -> Result<(), kernels_cerebras::Error> {
        match op {
            Layout::TopK {
                x,
                k,
                values,
                indices,
            } => layout::topk(
                self.ctx(),
                self.tensor(*x),
                *k,
                self.tensor(*values),
                self.tensor(*indices),
            ),
            Layout::EmbedConcat {
                ids,
                table,
                vocab,
                y,
            } => layout::embed_concat(
                self.ctx(),
                self.tensor(*ids),
                self.tensor(*table),
                *vocab,
                self.tensor(*y),
            ),
            Layout::Embed {
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
            Layout::SplitQGate {
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
            Layout::SplitRows {
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
            Layout::MergeRows { x, side, y } => {
                layout::merge_rows(self.ctx(), self.tensor(*x), *side, self.tensor(*y))
            }
            Layout::ScatterLiveRows {
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
            Layout::GatherRows { x, rows, y } => layout::gather_rows(
                self.ctx(),
                self.uncut(*x),
                self.tensor(*rows),
                self.tensor(*y),
            ),
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
                self.tensor(*q),
                self.tensor(*k),
                self.tensor(*v),
            ),
            Layout::EmbedWeighted {
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
            Layout::Argmax { xs, y } => {
                for (column, x) in xs.iter().enumerate() {
                    layout::argmax(
                        self.ctx(),
                        self.tensor(*x),
                        u32::try_from(column).expect("a draft depth inside u32"),
                        self.tensor(*y),
                    )?;
                }
                Ok(())
            }
            other => Err(kernels_cerebras::Error::Unsupported { op: other.name() }),
        }
    }
}
