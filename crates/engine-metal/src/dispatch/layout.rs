//! The `layout` family: `impl DispatchLayout for Run<'_>`.

use kernels_metal::layout;
use model_exec::{DispatchLayout, KernelError};
use model_ir::{Layout, Operands};

use crate::run::Run;

impl DispatchLayout for Run<'_> {
    fn dispatch(&mut self, op: &Layout) -> Result<(), KernelError> {
        self.layout(op).map_err(crate::error::kernel)
    }
}

impl Run<'_> {
    /// The arms themselves, in `kernels-metal`'s error vocabulary and not
    /// the contract's — which is what keeps each one a plain tail call with
    /// a plain `?`. [`kernel`](crate::error::kernel) is the single line
    /// above that lifts the family, and says why it is a call and not a
    /// `From` impl.
    fn layout(&mut self, op: &Layout) -> Result<(), kernels_metal::Error> {
        match op {
            // The same choice `linear.matmul` makes, for the same reason:
            // the op names a table and the weight row names its form.
            //
            // The quantized gather takes no `vocab`. That is the shader's
            // own shape and not an omission — `embed_gather.metal` indexes
            // the codes by the id it was handed, where `embed.metal` clamps
            // against a row count first. A table whose rows a plan can
            // address past is one this arm cannot narrow, so the guard the
            // dense entry carries stays the dense entry's.
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
            // **THE SPATIAL POOL, REFUSED BY NAME** (multimodal §6.5). This
            // plane's `layout` unit moves rows and never folds them — there is
            // no `pool_rows` shader — and a fold is not something a gather can
            // be talked into. One entry beside `scatter_rows` retires this.
            // **THE TWO FOLDS AND THE DROPPING SCATTER, REFUSED BY NAME**
            // (multimodal §7.4, §8.1, §8.6). This plane's `layout` unit moves
            // rows and never folds them, and its `scatter_rows` reads every
            // route as a destination; a fold is not something a gather can be
            // talked into, and a guard this plane does not have is a write
            // below the base of a rectangle. Three entries beside
            // `scatter_rows` retire these.
            Layout::PoolRows { .. }
            | Layout::MergeRows { .. }
            | Layout::ScatterLiveRows { .. }
            // And the interpolating gather (multimodal §9.2): this plane's
            // `layout::embed` reads one row per id and weights none of them.
            | Layout::EmbedWeighted { .. } => {
                Err(kernels_metal::Error::Unsupported { op: op.name() })
            }
            // THE EMBED MERGE, forwarded like every other arm here: this
            // plane's `layout::scatter_rows` is the same body its
            // `gather_rows` twin is, so there is nothing to refuse. What this
            // plane cannot do is RESOLVE `src` — `Run::cut` refuses a
            // patch-axis rectangle by name one file over — so a tower plan
            // stops before it reaches here.
            Layout::ScatterRows {
                src,
                routes,
                y,
                y_out: _,
            } => layout::scatter_rows(
                self.ctx(),
                self.tensor(*src),
                self.tensor(*routes),
                self.tensor(*y),
            ),
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
