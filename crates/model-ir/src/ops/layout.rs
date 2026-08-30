use serde::{Deserialize, Serialize};

use crate::operands::Operands;
use crate::value::ValueId;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Layout {
    Embed {
        ids: ValueId,
        table: ValueId,
        vocab: u32,
        y: ValueId,
    },
    /// **THE GATHER THAT INTERPOLATES**: `y[r] = Σₜ weights[r, t] ·
    /// table[ids[r, t]]` (multimodal §9.2).
    ///
    /// The learned position grid is stored once at `num_grid_per_side²` and
    /// resampled to each image's own grid, so a patch row needs `taps` rows of
    /// it under `taps` weights — four for bilinear, sixteen for bicubic, and
    /// `_interpolation_axis_taps_weights`' outer product is where both come
    /// from. `ids` and `weights` are both `[rows, taps]`; `taps` is read off
    /// their width rather than stated, because the operands carry it and a
    /// second spelling could disagree with them.
    ///
    /// **ONE OP AND NOT FOUR GATHERS AND THREE ADDS**, which was the other
    /// shape on offer. The deciding fact is not op count: this IR has NO
    /// row-broadcast multiply. [`Elementwise::Scale`](crate::Elementwise::Scale)
    /// reads one device scalar for a whole rectangle and
    /// [`MulScalar`](crate::Elementwise::MulScalar) is a plan constant, so
    /// "four `embed`s and a weighted sum" would have needed a NEW op anyway —
    /// a `[rows]`-broadcast scale — plus eight nodes and four times the gather
    /// traffic, against this one node.
    ///
    /// **AND THE NATIVE GRID DOES NOT COME HERE AT ALL.** A tower whose image
    /// grid is the stored grid resamples by the identity — one tap, weight one
    /// — and writes [`Embed`](Layout::Embed) with a `[Dim::Patches]` id
    /// vector. The cheap path is the absence of this op, not a degenerate use
    /// of it.
    EmbedWeighted {
        ids: ValueId,
        weights: ValueId,
        table: ValueId,
        vocab: u32,
        y: ValueId,
    },
    SplitQkv {
        packed: ValueId,
        q_width: u32,
        kv_width: u32,
        q: ValueId,
        k: ValueId,
        v: ValueId,
    },
    /// Deinterleaves per-head (q, gate) pairs from the packed projection.
    SplitQGate {
        packed: ValueId,
        head_dim: u32,
        q: ValueId,
        gate: ValueId,
    },
    /// Splits each row at column `width`.
    SplitRows {
        x: ValueId,
        width: u32,
        left: ValueId,
        right: ValueId,
    },
    /// Views layer `layer`'s `width`-wide slice of a stacked table.
    Select {
        table: ValueId,
        layer: u32,
        width: u32,
        y: ValueId,
    },
    /// **THE EMBED MERGE**: the tower's rows written into the token rows the
    /// image placeholders occupy (multimodal §2's third op).
    ///
    /// `src` is a PATCH rectangle, `y` a TOKEN one, and `routes` — the
    /// `RuntimeInput::PatchRoutes` vector, `i32`, one entry per `src` row —
    /// says which token row each tower row lands in. This is the one node in
    /// a tower plan that reads one row axis and writes the other, and
    /// [`Operands::outputs`] is what the capture-unit partition asks: it
    /// writes token rows, so it belongs to the TRUNK's exec and reads its row
    /// count from the token window.
    ///
    /// **`y_out` AND NOT A BARE `y`, BECAUSE A SCATTER DOES NOT WRITE EVERY
    /// ROW IT OWNS.** The design sketch spells the op `scatter_rows(y, src,
    /// routes)` with three operands; a three-operand form would make `y` a
    /// fresh rectangle whose unrouted rows — every text row of the fire — are
    /// whatever the arena last left there. The merge is an IN-PLACE edit of
    /// the embedding `layout.embed` already wrote, so it takes `y` as an
    /// input and aliases `y_out` onto it, exactly as `elementwise.add_bias`
    /// and `elementwise.residual_add` do.
    ScatterRows {
        src: ValueId,
        routes: ValueId,
        y: ValueId,
        y_out: ValueId,
    },
    /// **THE SPATIAL POOL** (multimodal §6.5, §7.4): `y[j]` is the mean of
    /// rows `[j·side², (j+1)·side²)` of `x`.
    ///
    /// gemma4's tower averages each `k × k` square of the patch grid down to
    /// one soft token — `vision_soft_tokens_per_image` of them per image,
    /// `pooling_kernel_size` being `k`. §6.5 calls that a row-mixing GEMM,
    /// `[soft_tokens, patches] · [patches, hidden]`, and says the vocabulary
    /// has none. It does not need one: the matrix is an `O(patches²)`
    /// constant, per rung, whose operand has a symbolic dim no weight
    /// declaration here carries, and what it computes is a strided reduction.
    ///
    /// **A `Layout` ROW AND NOT A `Linear` ONE**, and that is what the shape
    /// says: this moves rows and folds them, it contracts no channel, it reads
    /// no bank. `layout.split_rows` cuts a rectangle in two and this one folds
    /// it by a stride; they are the same family of question.
    ///
    /// **`side` AND NOT `side²`**, because `pooling_kernel_size` is the
    /// checkpoint's own number and a text should be able to write it down.
    /// `side == 1` is the identity, which is a real case: the pooler skips
    /// itself when an image's patch count already equals its soft-token count.
    ///
    /// # The operand contract
    ///
    /// `x` and `y` are both `[Dim::Patches, hidden]`. The output is the SAME
    /// symbolic row space — so it is cut at the same patch window and belongs
    /// to the same capture unit — and only its leading `x.rows / side²` rows
    /// are written. A narrower dim (`Patches / k`) would be a fourth row space
    /// for one op, and the rows past the fold are the rung padding that was
    /// already there.
    ///
    /// **AND IT READS NO GEOMETRY**, which is the whole of why it is one op.
    /// Pooling by POSITION needs each patch's grid coordinate and its image's
    /// grid width; pooling by STRIDE needs the submission to order an image's
    /// patches so that each `side × side` square is contiguous — which is §2's
    /// merge-block-major statute at `side` instead of 2. The statute already
    /// exists for qwen's 2×2 merge; this is the same sentence at `k = 3`. Image
    /// boundaries then need no indptr either: `get_aspect_ratio_preserving_size`
    /// rounds each image to a multiple of `pooling_kernel_size · patch_size`,
    /// so every run is a whole number of blocks and no block straddles two.
    PoolRows {
        x: ValueId,
        side: u32,
        y: ValueId,
    },
    /// **THE MERGING FOLD** (multimodal §8.1, §8.3): `y[j]` is rows
    /// `[j·side², (j+1)·side²)` of `x` laid END TO END — `side²` rows of
    /// `width` becoming one row of `side²·width`.
    ///
    /// qwen's spatial merger, and the statement §8.1 found the tower stopping
    /// one short of. §2 called the merge "a view feeding existing GEMMs" and
    /// the checkpoints say otherwise about the ROW COUNT:
    /// `Qwen3_5VisionPatchMerger.forward` opens `x.view(-1, hidden_size ·
    /// spatial_merge_size²)`, and `merger.linear_fc1.weight` is
    /// `[4·hidden, 4·hidden]` — four rows in, one out.
    ///
    /// **§2 WAS RIGHT ABOUT THE BYTES AND WRONG ONLY ABOUT THE COUNT.** On a
    /// dense row-major rectangle `[rows, width]` and `[rows/side², side²·width]`
    /// hold the same element at the same offset, so this op is the identity
    /// copy. It is a NODE because the IR gives one value one type, and a
    /// second type needs a second value; a compiler that later resolves it to
    /// a placement alias changes no arithmetic, because there is none.
    ///
    /// # The operand contract
    ///
    /// `x` is `[Dim::Patches, width]` and `y` is `[Dim::Patches, side²·width]`
    /// — the ONE fold that cannot reuse `x`'s type. Same row space, so the
    /// same patch window and the same capture unit; leading `x.rows / side²`
    /// rows written, and the tail is [`PoolRows`](Layout::PoolRows)' tail with
    /// [`ScatterLiveRows`](Layout::ScatterLiveRows)' answer to it.
    MergeRows {
        x: ValueId,
        side: u32,
        y: ValueId,
    },
    /// **THE EMBED MERGE, WITH A DROP SENTINEL** (multimodal §8.6):
    /// [`ScatterRows`](Layout::ScatterRows) exactly, plus a NEGATIVE `routes`
    /// entry meaning "this row has no destination".
    ///
    /// A compacting fold answers `rows / side²` rows and leaves the rest of the
    /// patch rectangle as whatever the arena held; `RuntimeInput::PatchRoutes`
    /// is `[Dim::Patches]`, one destination per row of the FULL rectangle, so
    /// those rows have route entries and there was no legal value to put in
    /// them — the shell refuses `route < 0` and `layout.scatter_rows` reads a
    /// negative index as a write below the base of the token rectangle. `-1`
    /// is the spelling `RuntimeInput::AdapterRoutes` already uses for "no
    /// bank".
    ///
    /// **A SECOND ROW AND NOT A WIDENED FIRST ONE.** The plain scatter's
    /// contract — every route names a row, checked host-side before the launch
    /// — is one its `Fallback::Copy` consumers rely on, and a fire that
    /// reached this variant's leniency by accident would scatter the arena's
    /// leftovers over real token rows silently. Two ops, two contracts; the
    /// shell admits the sentinel only for a plan that declares THIS one.
    ///
    /// `y_out` aliases `y`, for [`ScatterRows`](Layout::ScatterRows)' reason.
    ScatterLiveRows {
        src: ValueId,
        routes: ValueId,
        y: ValueId,
        y_out: ValueId,
    },
}

impl Operands for Layout {
    fn inputs(&self, sink: &mut Vec<ValueId>) {
        match self {
            Self::Embed { ids, table, .. } => sink.extend([*ids, *table]),
            Self::EmbedWeighted { ids, weights, table, .. } => {
                sink.extend([*ids, *weights, *table]);
            }
            Self::SplitQkv { packed, .. } => sink.push(*packed),
            Self::SplitQGate { packed, .. } => sink.push(*packed),
            Self::SplitRows { x, .. } => sink.push(*x),
            Self::Select { table, .. } => sink.push(*table),
            Self::ScatterRows { src, routes, y, .. } => sink.extend([*src, *routes, *y]),
            Self::PoolRows { x, .. } => sink.push(*x),
            Self::MergeRows { x, .. } => sink.push(*x),
            Self::ScatterLiveRows { src, routes, y, .. } => sink.extend([*src, *routes, *y]),
        }
    }
    fn outputs(&self, sink: &mut Vec<ValueId>) {
        match self {
            Self::Embed { y, .. } => sink.push(*y),
            Self::EmbedWeighted { y, .. } => sink.push(*y),
            Self::SplitQkv { q, k, v, .. } => sink.extend([*q, *k, *v]),
            Self::SplitQGate { q, gate, .. } => sink.extend([*q, *gate]),
            Self::SplitRows { left, right, .. } => sink.extend([*left, *right]),
            Self::Select { y, .. } => sink.push(*y),
            Self::ScatterRows { y_out, .. } => sink.push(*y_out),
            Self::PoolRows { y, .. } => sink.push(*y),
            Self::MergeRows { y, .. } => sink.push(*y),
            Self::ScatterLiveRows { y_out, .. } => sink.push(*y_out),
        }
    }
    /// The one aliasing row this family has, and it is the scatter's: every
    /// other `Layout` variant either reads a table or cuts a packed row into
    /// pieces that are views of it, and neither is an in-place edit the
    /// compiler has to fold into one slot.
    fn aliases(&self, sink: &mut Vec<(ValueId, ValueId)>) {
        match self {
            Self::Embed { .. }
            | Self::EmbedWeighted { .. }
            | Self::SplitQkv { .. }
            | Self::SplitQGate { .. }
            | Self::SplitRows { .. }
            | Self::Select { .. }
            // The pool writes a FRESH rectangle: `y` is not `x` narrowed in
            // place, it is a different number of rows of different numbers.
            | Self::PoolRows { .. }
            | Self::MergeRows { .. } => {}
            Self::ScatterRows { y_out, y, .. }
            | Self::ScatterLiveRows { y_out, y, .. } => sink.push((*y_out, *y)),
        }
    }
    fn name(&self) -> &'static str {
        match self {
            Self::Embed { .. } => "layout.embed",
            Self::EmbedWeighted { .. } => "layout.embed_weighted",
            Self::SplitQkv { .. } => "layout.split_qkv",
            Self::SplitQGate { .. } => "layout.split_q_gate",
            Self::SplitRows { .. } => "layout.split_rows",
            Self::Select { .. } => "layout.select",
            Self::ScatterRows { .. } => "layout.scatter_rows",
            Self::PoolRows { .. } => "layout.pool_rows",
            Self::MergeRows { .. } => "layout.merge_rows",
            Self::ScatterLiveRows { .. } => "layout.scatter_live_rows",
        }
    }
}
