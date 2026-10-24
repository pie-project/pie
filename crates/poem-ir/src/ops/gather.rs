use serde::{Deserialize, Serialize};

use crate::Op;
use crate::value::ValueId;

/// Moving rows and columns: embedding lookups, splits and concatenations,
/// row gathers, scatters and permutations, and the ids those read.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Op)]
#[op(group = "gather")]
pub enum Gather {
    /// Each id's row of `table` (`vocab` rows), times its f32 `weights`
    /// where given; `concat` lands the rows of a token's several ids side by
    /// side, and `scales` are an MXFP8 table's ue8m0 exponents, one a 32 of
    /// each row.
    Embed {
        #[input(I32)]
        ids: ValueId,
        #[input(F32)]
        weights: Option<ValueId>,
        #[input]
        table: ValueId,
        #[serde(default)]
        #[input]
        scales: Option<ValueId>,
        vocab: u32,
        concat: bool,
        #[output]
        y: ValueId,
    },
    SplitQkv {
        #[input]
        packed: ValueId,
        q_width: u32,
        kv_width: u32,
        #[output]
        q: ValueId,
        #[output]
        k: ValueId,
        #[output]
        v: ValueId,
    },
    SplitQGate {
        #[input]
        packed: ValueId,
        head_dim: u32,
        #[output]
        q: ValueId,
        #[output]
        gate: ValueId,
    },
    /// A packed row of `heads[0]` query heads and `heads[1]` key and value
    /// heads, queries and keys `widths[0]` wide and values `widths[1]`, cut
    /// into q, k and v of `head_dim`-wide heads, each zero past its width.
    SplitQkvPadded {
        #[input]
        packed: ValueId,
        heads: [u32; 2],
        widths: [u32; 2],
        head_dim: u32,
        #[output]
        q: ValueId,
        #[output]
        k: ValueId,
        #[output]
        v: ValueId,
    },
    SplitRows {
        #[input]
        x: ValueId,
        width: u32,
        #[output]
        left: ValueId,
        #[output]
        right: ValueId,
    },
    /// `xs` side by side, a row each's columns after the one before's.
    ConcatCols {
        #[input]
        xs: Vec<ValueId>,
        #[output]
        y: ValueId,
    },
    /// The rows of `x` that `rows` names, in its order.
    Rows {
        #[input]
        x: ValueId,
        #[input(I32)]
        rows: ValueId,
        #[output]
        y: ValueId,
    },
    /// Layer `layer`'s `width` columns of a stacked table.
    Select {
        #[input]
        table: ValueId,
        layer: u32,
        width: u32,
        #[output]
        y: ValueId,
    },
    ScatterRows {
        #[input]
        src: ValueId,
        #[input(I32)]
        routes: ValueId,
        #[input]
        y: ValueId,
        #[output(alias = y)]
        y_out: ValueId,
    },
    ScatterLiveRows {
        #[input]
        src: ValueId,
        #[input(I32)]
        routes: ValueId,
        #[input]
        y: ValueId,
        #[output(alias = y)]
        y_out: ValueId,
    },
    PoolRows {
        #[input]
        x: ValueId,
        side: u32,
        #[output]
        y: ValueId,
    },
    MergeRows {
        #[input]
        x: ValueId,
        side: u32,
        #[output]
        y: ValueId,
    },
    PackRows {
        #[input]
        x: ValueId,
        #[input(I32)]
        perm: ValueId,
        #[output]
        y: ValueId,
    },
    UnpackRows {
        #[input]
        x: ValueId,
        #[input(I32)]
        perm: ValueId,
        #[output]
        y: ValueId,
    },
    /// Patch rows of raw RGB bytes, `patch` × `patch` pixels in HWC order,
    /// to the tower's rows: `(v / 255 - mean[c]) / std[c]`, columns in
    /// `order`, the frame repeated `temporal` times.
    Pixels {
        #[input(U8)]
        x: ValueId,
        patch: u32,
        mean: [f32; 3],
        std: [f32; 3],
        order: PixelOrder,
        temporal: u32,
        #[output]
        y: ValueId,
    },
    /// Position-table taps for each patch from its grid position and its
    /// image's grid: `ids` and `weights` for `gather.embed`.
    GridTaps {
        #[input(I32)]
        positions: ValueId,
        #[input(I32)]
        grids: ValueId,
        #[input(I32)]
        segments: ValueId,
        kind: TapKind,
        side: u32,
        #[output(I32)]
        ids: ValueId,
        #[output(F32)]
        weights: ValueId,
    },
    /// Each row's hashed n-gram ids over its lane's last ids, a step a row.
    NgramIds {
        #[input(I32)]
        ids: ValueId,
        /// The window of each lane's last ids the hashes span.
        #[input(CACHE)]
        state: ValueId,
        eos: u32,
        mults: Vec<u64>,
        primes: Vec<u64>,
        offsets: Vec<u64>,
        heads_per_ngram: u32,
        /// An i64 table the ids are mapped through before hashing (Engram's
        /// tokenizer-compressed ids); the window keeps raw ids.
        #[input]
        map: Option<ValueId>,
        #[output(I32)]
        ngram_ids: ValueId,
    },
    /// [`Gather::NgramIds`] over a prompt's rows.
    NgramIdsChunked {
        #[input(I32)]
        ids: ValueId,
        /// The window of each lane's last ids the hashes span.
        #[input(CACHE)]
        state: ValueId,
        eos: u32,
        mults: Vec<u64>,
        primes: Vec<u64>,
        offsets: Vec<u64>,
        heads_per_ngram: u32,
        /// An i64 table the ids are mapped through before hashing (Engram's
        /// tokenizer-compressed ids); the window keeps raw ids.
        #[input]
        map: Option<ValueId>,
        #[output(I32)]
        ngram_ids: ValueId,
    },
}

/// How a patch row's columns are laid out: channel-major over the pixels,
/// or pixel-major with the channels interleaved.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PixelOrder {
    Chw,
    Hwc,
}

/// Which taps [`Gather::GridTaps`] takes into a position table.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TapKind {
    /// Four bilinear taps into a `side` × `side` table, the patch grid
    /// stretched over it.
    Bilinear,
    /// Two unit taps: the column into the first `side` rows, the row into
    /// the next `side`.
    Axes,
}

impl TapKind {
    #[must_use]
    pub const fn taps(self) -> u32 {
        match self {
            TapKind::Bilinear => 4,
            TapKind::Axes => 2,
        }
    }
}
