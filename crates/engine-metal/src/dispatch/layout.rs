//! The `layout` family: `impl DispatchLayout for Run<'_>`.

use kernels_metal::layout;
use model_exec::{DispatchLayout, KernelError};
use model_ir::Layout;

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
            // **BOTH ARMS STATE `vocab`, BECAUSE BOTH ARMS ADDRESS BY THE
            // ID.** The banked gather reads three planes off it where the
            // dense one reads a single row, so an id the table cannot answer
            // is the worse read of the two, not the exempt one. The CUDA twin
            // hands `*vocab` to `embed_mlx_affine` and to `layout::embed`
            // alike (`engine-cuda/src/dispatch/layout.rs`), and this arm is
            // that arm.
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
            // **THE CONCATENATING GATHER** (qwen4's PLE): sixteen hashed ids
            // per token, each gathering one row of a twenty-million-row table,
            // laid side by side into the wide row the PLE's projections read.
            //
            // The same choice `Embed` above makes, and the same two shader
            // FILES: an id and a slice are the same address at a different
            // stride, so the head axis folds into the row axis in the GRID and
            // neither point needs a second kernel body. The bank arm is the
            // one this SKU fires — the miniature's table is a `(4, 32)`
            // triplet and a dense landing of it would be two gigabytes of bf16
            // nothing reads twice.
            //
            // The dense arm fires `embed_concat_bfloat16` and not `Embed`'s
            // `embed_bfloat16`: one body, two stamps, because an id this table
            // cannot answer contributes ZERO to the concatenated row here
            // where it clamps to row zero there — the twin's own split
            // (`embed_concat.cuh` against `layout.cuh`).
            Layout::EmbedConcat {
                ids,
                table,
                vocab,
                y,
            } => match self.banked(*table) {
                Some(bank) => layout::embed_concat_mb_4bit(
                    self.ctx(),
                    self.tensor(*ids),
                    bank,
                    *vocab,
                    self.tensor(*y),
                ),
                None => layout::embed_concat(
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
            // **THE SPATIAL POOL** (multimodal §6.5, §7.4): gemma4's `k × k`
            // soft-token fold. `side²` consecutive patch rows averaged into
            // one, in f32 whatever the element is, over the pool-block-major
            // patch order §2's statute already asks the submission for.
            Layout::PoolRows { x, side, y } => layout::pool_rows(
                self.ctx(),
                self.tensor(*x),
                *side,
                self.tensor(*y),
            ),
            // **THE MERGING FOLD** (multimodal §8.1, §8.3), the same unit's
            // other entry: `side²` rows concatenated rather than averaged, and
            // the one op here whose destination is wider than its source.
            // qwen's spatial merger, which is the dev vehicle's own fold.
            Layout::MergeRows { x, side, y } => layout::merge_rows(
                self.ctx(),
                self.tensor(*x),
                *side,
                self.tensor(*y),
            ),
            // **THE EMBED MERGE WITH A DROP SENTINEL** (multimodal §8.6):
            // `scatter_rows` below, plus a guard on a negative route. The
            // shell admits a `-1` in `patch_routes` only for a plan that names
            // THIS op, so the arm below keeps the contract it always had.
            Layout::ScatterLiveRows {
                src,
                routes,
                y,
                y_out: _,
            } => layout::scatter_live_rows(
                self.ctx(),
                self.tensor(*src),
                self.tensor(*routes),
                // **THE DESTINATION GOES OVER WHOLE** (multimodal §18): the
                // routes are absolute fire rows, so a window-cut `y` would
                // count the region's offset twice. `Run::uncut` carries the
                // argument, and it is the same call `fire_wide` is on the
                // CUDA plane.
                self.uncut(*y),
            ),
            // The interpolating gather (multimodal §9.2): the separable
            // position table read at two taps with weights of one, and the
            // resample read at four or sixteen. `layout::embed` reads one row
            // per id and weights none of them, so this is its own entry.
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
            // THE EMBED MERGE. `src` resolves at the PATCH window and `y` at
            // the token one — the one node in a tower plan whose two operands
            // come out of two seriations — and the routes vector has already
            // been checked against this fire's token row count host-side
            // (`Fault::PatchRoute`), because an out-of-range entry here is an
            // out-of-bounds device write the kernel cannot see.
            Layout::ScatterRows {
                src,
                routes,
                y,
                y_out: _,
            } => layout::scatter_rows(
                self.ctx(),
                self.tensor(*src),
                self.tensor(*routes),
                // Whole, for `scatter_live_rows`' reason one arm up.
                self.uncut(*y),
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
