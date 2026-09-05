use kernels_wgpu::layout;
use model_exec::{DispatchLayout, KernelError};
use model_ir::Layout;

use crate::run::Run;

impl DispatchLayout for Run<'_> {
    fn dispatch(&mut self, op: &Layout) -> Result<(), KernelError> {
        self.layout(op).map_err(crate::error::kernel)
    }
}

impl Run<'_> {
    fn layout(&mut self, op: &Layout) -> Result<(), kernels_wgpu::Error> {
        match op {
            Layout::Embed {
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
                None => layout::embed(
                    self.ctx(),
                    self.tensor(*ids),
                    self.tensor(*table),
                    *vocab,
                    self.tensor(*y),
                ),
            },

            Layout::EmbedConcat {
                ids,
                table,
                vocab,
                y,
            } => match self.banked(*table) {
                Some(bank) => {
                    let (bank, ids) = match self.gathered_table(bank, self.tensor(*ids))? {
                        Some(seated) => seated,
                        None => (bank, self.tensor(*ids)),
                    };
                    let guard = bank.codes.rows.min(*vocab);
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

            Layout::PoolRows { x, side, y } => {
                layout::pool_rows(self.ctx(), self.tensor(*x), *side, self.tensor(*y))
            }

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

            Layout::ScatterRows {
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

            // The per-row top-k a candidate selector reads has no wgpu kernel yet.
            Layout::TopK { .. } => Err(kernels_wgpu::Error::Unsupported { op: "layout.topk" }),
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
                self.tensor(*y),
            ),
        }
    }
}
