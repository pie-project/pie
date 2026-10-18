use crate::error::Error;
use dtype::Dtype;

use crate::jit::{
    Arg, ArgValue, Ctx, Fire, Launch, dtype_dispatch, nonzero, refuse, stated, symbol,
};
use crate::linear::gemm;
use crate::linear::moe::GroupSeat;
use crate::tensor::Tensor;

const FILE: &str = "linear/quant.hip";

const BLOCK: u32 = 256;

const WAVE: u32 = 64;

/// The bf16 tile a plane is decoded into before the dense gemm reads it:
/// one a stream, the size of the widest plane fired on it, since the decode
/// and the gemm are consecutive on the stream. Keyed by region it was one a
/// region, the whole model in bf16 across a load.
pub(crate) const DECODED_WEIGHT: &str = "linear.quant.decoded_weight";

fn route_rows(rows: u32, width: u32) -> Launch {
    const MAX_BLOCK: u32 = 1024;

    Launch::per_row(
        rows,
        width
            .div_ceil(WAVE)
            .max(1)
            .saturating_mul(WAVE)
            .min(MAX_BLOCK),
    )
}

fn extent(op: &'static str, n: u64) -> Result<u32, Error> {
    u32::try_from(n).map_err(|_| {
        refuse(
            op,
            format!("{n} elements do not fit a 32-bit launch extent"),
        )
    })
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum OffsetKind {
    Post,
    PreInt,
    PreReal,
    PreConst,
}

impl OffsetKind {
    const fn axis(self) -> &'static str {
        match self {
            Self::Post => "::pie::linear::kOffPost",
            Self::PreInt => "::pie::linear::kOffPreInt",
            Self::PreReal => "::pie::linear::kOffPreReal",
            Self::PreConst => "::pie::linear::kOffPreConst",
        }
    }

    const fn spelling(self) -> &'static str {
        match self {
            Self::Post => "a post-offset arm (`s·c + b`)",
            Self::PreInt => "an integer pre-offset arm (`s·(c − z)`)",
            Self::PreReal => "a real pre-offset arm (`s·(c − z)`)",
            Self::PreConst => "a constant pre-offset arm (`s·(c − 2^(bits−1))`)",
        }
    }
}

/// Takes this stream's decoded-weight tile at `bytes` now, so the first
/// fire that decodes a plane finds it and no capture ever has to grow it —
/// a load whose bodies are held to nothing arms no body at load, and its
/// first fire is a capture.
pub fn warm_decoded_weight(ctx: &Ctx, bytes: u64) -> Result<(), Error> {
    if bytes == 0 {
        return Ok(());
    }
    ctx.scratch_stream(
        "linear.matmul",
        DECODED_WEIGHT,
        usize::try_from(bytes).unwrap_or(usize::MAX),
    )
    .map(|_| ())
}

#[allow(clippy::too_many_arguments)]
pub fn matmul(
    ctx: &Ctx,
    act: Tensor,
    codes: Tensor,
    scales: Tensor,
    offset: OffsetKind,
    biases: Option<Tensor>,
    factor: Dtype,
    y: &mut Tensor,
    seat: GroupSeat,
) -> Result<(), Error> {
    dense_affine(
        ctx,
        "linear.matmul",
        act,
        codes,
        scales,
        offset,
        biases,
        factor,
        y,
        seat,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn lm_head(
    ctx: &Ctx,
    act: Tensor,
    codes: Tensor,
    scales: Tensor,
    offset: OffsetKind,
    biases: Option<Tensor>,
    factor: Dtype,
    y: &mut Tensor,
    seat: GroupSeat,
) -> Result<(), Error> {
    dense_affine(
        ctx,
        "linear.lm_head",
        act,
        codes,
        scales,
        offset,
        biases,
        factor,
        y,
        seat,
    )
}

#[derive(Clone, Copy)]
struct Affine {
    bits: u32,
    group: u32,
    n: u32,
    k: u32,
}

fn affine(
    op: &'static str,
    act: Tensor,
    codes: Tensor,
    scales: Tensor,
    offset: OffsetKind,
    biases: Option<Tensor>,
    y: &Tensor,
) -> Result<Affine, Error> {
    debug_assert_eq!(codes.dtype, Dtype::U8, "a packed plane binds as bytes");
    debug_assert_eq!(scales.dtype, Dtype::U8, "a packed plane binds as bytes");
    debug_assert!(
        biases.is_none_or(|b| b.dtype == Dtype::U8),
        "a packed plane binds as bytes"
    );
    match (offset, biases) {
        (OffsetKind::PreConst, Some(_)) => {
            return Err(refuse(
                op,
                format!(
                    "{} reads no offset plane, and one was bound",
                    offset.spelling()
                ),
            ));
        }
        (OffsetKind::Post | OffsetKind::PreInt | OffsetKind::PreReal, None) => {
            return Err(refuse(
                op,
                format!(
                    "{} was declared with no offset plane to read",
                    offset.spelling()
                ),
            ));
        }
        _ => {}
    }
    debug_assert_eq!(
        act.rows, y.rows,
        "the activation's rows are the rows the result lands"
    );
    let n = nonzero(op, "N, the columns this projection lands", y.width)?;
    let k = nonzero(op, "K, the contraction this projection walks", act.width)?;
    if scales.width == 0 || !scales.width.is_multiple_of(2) {
        return Err(refuse(
            op,
            format!(
                "a {}-byte factor row is not a whole number of two-byte factors",
                scales.width
            ),
        ));
    }
    let groups = scales.width / 2;
    if !k.is_multiple_of(groups) {
        return Err(refuse(
            op,
            format!("{groups} factors do not group a {k}-wide row into whole groups"),
        ));
    }
    let group = k / groups;
    let bits: u32 = if codes.width == k {
        8
    } else if codes.width * 2 == k {
        4
    } else if codes.width * 4 == k {
        2
    } else {
        return Err(refuse(
            op,
            format!(
                "a {}-byte code row stores a {k}-wide row at neither two, four nor eight bits",
                codes.width
            ),
        ));
    };
    let per_word = 32 / bits;
    if !group.is_multiple_of(per_word) {
        return Err(refuse(
            op,
            format!("a {group}-code group is not a whole number of {per_word}-code words"),
        ));
    }
    if let Some(plane) = biases {
        let want = if offset == OffsetKind::PreInt {
            groups
        } else {
            scales.width
        };
        if plane.width != want {
            return Err(refuse(
                op,
                format!(
                    "{} states a {want}-byte offset row over a {k}-wide row, and the \
                     plane holds {}",
                    offset.spelling(),
                    plane.width
                ),
            ));
        }
    }
    Ok(Affine { bits, group, n, k })
}

#[allow(clippy::too_many_arguments)]
fn dense_affine(
    ctx: &Ctx,
    op: &'static str,
    act: Tensor,
    codes: Tensor,
    scales: Tensor,
    offset: OffsetKind,
    biases: Option<Tensor>,
    factor: Dtype,
    y: &mut Tensor,
    seat: GroupSeat,
) -> Result<(), Error> {
    let t = dtype_dispatch!(op, act.dtype, { Bf16 => "::pie::bf16", F16 => "::pie::f16" });
    let f = dtype_dispatch!(op, factor, { Bf16 => "::pie::bf16", F16 => "::pie::f16" });
    let Affine { bits, group, n, k } = affine(op, act, codes, scales, offset, biases, y)?;
    if y.rows == 0 {
        return Ok(());
    }
    let cus = ctx.multiprocessors().unwrap_or(304).max(1);
    let axis = offset.axis();
    if y.rows > Stream::ROWS
        && let Some(tile) = Tile::plan(bits, group, offset, k, y.rows)
    {
        let blocks = y.rows.div_ceil(tile.bm()) * n.div_ceil(tile.bn());
        let split = GemmSplit::plan_stages(ctx, op, y.rows, n, k / tile.ks, 2, blocks, cus)?;
        ctx.fire(
            op,
            Fire::at(
                FILE,
                symbol(&format!(
                    "::pie::linear::matmul_affine_tile<{t}, {f}, ::pie::i32({bits}), {axis}, \
                     ::pie::i32({group}), ::pie::i32({}), ::pie::i32({}), ::pie::i32({}), \
                     ::pie::i32({}), ::pie::i32({})>",
                    tile.ks, tile.wm, tile.rs, tile.nt, tile.depth
                )),
            )
            .apply(Launch::grid(
                [
                    y.rows.div_ceil(tile.bm()),
                    n.div_ceil(tile.bn()),
                    split.splits,
                ],
                [TILE_THREADS, 1, 1],
            )),
            &[
                act.arg(),
                codes.arg(),
                scales.arg(),
                biases.map_or(ArgValue::ABSENT, |b| b.arg()),
                y.arg(),
                split.part_arg(),
                split.per_arg(op)?,
                stated(op, y.rows)?.arg(),
                stated(op, n)?.arg(),
                stated(op, k)?.arg(),
                ArgValue::Ptr(seat.cell),
                ctx.stage(),
            ],
        )?;
        return split.finalize(ctx, op, FILE, t, y);
    }
    if y.rows > DECODE_ROWS && k.is_multiple_of(GEMM_KC) {
        // Activation rows a block: the tallest tile that still lands a
        // block on every compute unit, so a short prefill is not one row
        // of blocks.
        let blocks = |bm: u32| y.rows.div_ceil(bm) * n.div_ceil(GEMM_BN);
        let bm = [128, 64]
            .into_iter()
            .find(|&bm| blocks(bm) >= cus)
            .unwrap_or(32);
        let split = GemmSplit::plan(ctx, op, y.rows, n, k, blocks(bm), cus)?;
        ctx.fire(
            op,
            Fire::at(
                FILE,
                symbol(&format!(
                    "::pie::linear::matmul_affine_gemm<{t}, {f}, ::pie::i32({bits}), {axis}, \
                     ::pie::i32({group}), ::pie::i32({bm})>"
                )),
            )
            .apply(Launch::grid(
                [y.rows.div_ceil(bm), n.div_ceil(GEMM_BN), split.splits],
                [GEMM_THREADS, 1, 1],
            )),
            &[
                act.arg(),
                codes.arg(),
                scales.arg(),
                biases.map_or(ArgValue::ABSENT, |b| b.arg()),
                y.arg(),
                split.part_arg(),
                split.per_arg(op)?,
                stated(op, y.rows)?.arg(),
                stated(op, n)?.arg(),
                stated(op, k)?.arg(),
                ArgValue::Ptr(seat.cell),
                ctx.stage(),
            ],
        )?;
        return split.finalize(ctx, op, FILE, t, y);
    }
    if let Some(s) = Stream::plan(bits, group, offset, n, k, y.rows, cus) {
        let split = GemmSplit::fixed(ctx, op, y.rows, n, s.slices)?;
        ctx.fire(
            op,
            Fire::at(
                FILE,
                symbol(&format!(
                    "::pie::linear::matmul_affine_stream<{t}, {f}, ::pie::i32({bits}), {axis}, \
                     ::pie::i32({group}), ::pie::i32({}), ::pie::i32({}), ::pie::i32({}), \
                     ::pie::i32({}), ::pie::i32({}), ::pie::i32({}), ::pie::i32({})>",
                    s.lane, s.nt, s.rs, s.rows, s.pieces, s.split, s.depth
                )),
            )
            .apply(Launch::grid(
                [s.blocks, s.slices, 1],
                [WAVE * s.split, 1, 1],
            )),
            &[
                act.arg(),
                codes.arg(),
                scales.arg(),
                biases.map_or(ArgValue::ABSENT, |b| b.arg()),
                y.arg(),
                split.part_arg(),
                stated(op, y.rows)?.arg(),
                stated(op, n)?.arg(),
                stated(op, k)?.arg(),
                ArgValue::Ptr(seat.cell),
                ctx.stage(),
            ],
        )?;
        return split.finalize(ctx, op, FILE, t, y);
    }
    let d = Decode::plan(bits, group, n, k, y.rows, cus);
    ctx.fire(
        op,
        Fire::at(
            FILE,
            symbol(&format!(
                "::pie::linear::matmul_affine_decode<{t}, {f}, ::pie::i32({bits}), {axis}, \
                 ::pie::i32({group}), ::pie::i32({}), ::pie::i32({}), ::pie::i32({}), \
                 ::pie::i32({}), ::pie::i32({}), ::pie::i32({})>",
                d.shape, d.lane, d.nt, d.rs, d.split, d.round
            )),
        )
        .apply(Launch::grid(
            [
                n.div_ceil(d.shape * d.nt),
                y.rows.div_ceil(d.shape * d.rs),
                1,
            ],
            [64 * d.split, 1, 1],
        )),
        &[
            act.arg(),
            codes.arg(),
            scales.arg(),
            biases.map_or(ArgValue::ABSENT, |b| b.arg()),
            y.arg(),
            stated(op, y.rows)?.arg(),
            stated(op, n)?.arg(),
            stated(op, k)?.arg(),
            ArgValue::Ptr(seat.cell),
            ctx.stage(),
        ],
    )
}

/// The most rows the streaming decode kernel takes in one launch row; past
/// it a K in whole stages takes the GEMM.
const DECODE_ROWS: u32 = 64;

/// The GEMM's K stage, output columns a block, and threads a block.
const GEMM_KC: u32 = 64;
const GEMM_BN: u32 = 128;
const GEMM_THREADS: u32 = 256;

/// The affine GEMM's K slicing (linear/affine.hip): a grid short of the
/// device slices K over blockIdx.z, each slice's f32 partial landing in
/// scratch for the finalize to sum in order.
#[derive(Clone, Copy, Debug)]
pub(crate) struct GemmSplit {
    pub splits: u32,
    per: u32,
    part: u64,
    rows: u32,
    n: u32,
}

impl GemmSplit {
    pub(crate) fn plan(
        ctx: &Ctx,
        op: &'static str,
        rows: u32,
        n: u32,
        k: u32,
        blocks: u32,
        cus: u32,
    ) -> Result<Self, Error> {
        // Eight 64-code stages a slice at least.
        Self::plan_stages(ctx, op, rows, n, k / GEMM_KC, 8, blocks, cus)
    }

    /// The slicing of `stages` K stages, `least` a slice at least: twice
    /// the device in blocks, since a partial is written and read once
    /// more, so no further.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn plan_stages(
        ctx: &Ctx,
        op: &'static str,
        rows: u32,
        n: u32,
        stages: u32,
        least: u32,
        blocks: u32,
        cus: u32,
    ) -> Result<Self, Error> {
        let mut splits = 1;
        while blocks * splits < 2 * cus && stages / (2 * splits) >= least && splits < 16 {
            splits *= 2;
        }
        let per = stages.div_ceil(splits);
        let splits = stages.div_ceil(per);
        let part = if splits > 1 {
            let bytes = splits as usize * rows as usize * n as usize * 4;
            ctx.scratch(op, "linear.quant.gemm_part", bytes)? as u64
        } else {
            0
        };
        Ok(Self {
            splits,
            per,
            part,
            rows,
            n,
        })
    }

    /// `splits` slices a kernel walks on its own (no stages a slice).
    fn fixed(ctx: &Ctx, op: &'static str, rows: u32, n: u32, splits: u32) -> Result<Self, Error> {
        let part = if splits > 1 {
            let bytes = splits as usize * rows as usize * n as usize * 4;
            ctx.scratch(op, "linear.quant.gemm_part", bytes)? as u64
        } else {
            0
        };
        Ok(Self {
            splits,
            per: 0,
            part,
            rows,
            n,
        })
    }

    pub(crate) fn part_arg(self) -> ArgValue {
        if self.splits > 1 {
            ArgValue::Ptr(self.part)
        } else {
            ArgValue::ABSENT
        }
    }

    pub(crate) fn per_arg(self, op: &'static str) -> Result<ArgValue, Error> {
        Ok(stated(op, self.per)?.arg())
    }

    /// The slices' sum into `y`, when K was sliced.
    pub(crate) fn finalize(
        self,
        ctx: &Ctx,
        op: &'static str,
        file: &'static str,
        t: &str,
        y: &Tensor,
    ) -> Result<(), Error> {
        if self.splits <= 1 {
            return Ok(());
        }
        let cells = self.rows * self.n;
        ctx.fire(
            op,
            Fire::at(
                file,
                symbol(&format!("::pie::linear::affine_gemm_finalize<{t}>")),
            )
            .apply(Launch::grid(
                [cells.div_ceil(256).min(4096), 1, 1],
                [256, 1, 1],
            )),
            &[
                ArgValue::Ptr(self.part),
                y.arg(),
                stated(op, self.rows)?.arg(),
                stated(op, self.n)?.arg(),
                stated(op, self.splits)?.arg(),
                ctx.stage(),
            ],
        )
    }
}

/// The tile kernel's threads a block (four wavefronts).
const TILE_THREADS: u32 = 256;

/// The tile kernel's tiling (linear/quant.hip `matmul_affine_tile`): `ks`
/// codes a K stage, `wm` x `4 / wm` wavefronts of `rs` x `nt` 16 x 16
/// tiles, `depth` stages in flight.
#[derive(Clone, Copy, Debug)]
struct Tile {
    ks: u32,
    wm: u32,
    rs: u32,
    nt: u32,
    depth: u32,
}

impl Tile {
    const fn bm(self) -> u32 {
        self.wm * 16 * self.rs
    }

    const fn bn(self) -> u32 {
        4 / self.wm * 16 * self.nt
    }

    fn plan(bits: u32, group: u32, offset: OffsetKind, k: u32, rows: u32) -> Option<Self> {
        // A lane group's quarter of a group is whole words, at most four;
        // a group is two to eight 16-code segments.
        if !matches!(group * bits / 32, 4 | 8 | 16) || !matches!(group / 16, 2 | 4 | 8) {
            return None;
        }
        // Activation rows a block by what the rows fill; the stage's code
        // bytes a weight row shrink as the activation tile grows, to keep
        // LDS within a block.
        let (wm, rs, nt, row_bytes) = if rows <= 16 {
            (1, 1, 1, 128)
        } else if rows <= 32 {
            (2, 1, 2, 128)
        } else {
            (2, 2, 2, 64)
        };
        let tile = Self {
            ks: row_bytes * 8 / bits,
            wm,
            rs,
            nt,
            depth: 2,
        };
        let (bm, bn, ks) = (tile.bm(), tile.bn(), tile.ks);
        let groups = ks / group;
        let off_bytes = match offset {
            OffsetKind::PreConst => 0,
            OffsetKind::PreInt => 1,
            OffsetKind::Post | OffsetKind::PreReal => 2,
        };
        let lds = bn * (row_bytes + 16)
            + bm * (ks + 8) * 2
            + bn * groups * (2 + off_bytes)
            + groups * bm * 4;
        let whole = |bytes: u32| bytes.is_multiple_of(16 * TILE_THREADS);
        (ks.is_multiple_of(group)
            && k.is_multiple_of(ks)
            && (groups * 2).is_multiple_of(4)
            && (groups * off_bytes).is_multiple_of(4)
            && whole(bn * row_bytes)
            && whole(bm * ks)
            && lds <= 64 << 10)
            .then_some(tile)
    }
}

/// The streaming kernel's tiling (linear/quant.hip `matmul_affine_stream`):
/// `lane` codes a lane reads a piece, `nt` and `rs` tiles of four weight and
/// activation rows, `rows` of the four an activation tile lands, `pieces`
/// a wavefront holds the activation of and `split` wavefronts a block, so
/// a block walks a slice of K a tile, `slices` of K over the grid, `depth`
/// tiles in flight a block, `blocks` persistent blocks a slice.
#[derive(Clone, Copy, Debug)]
struct Stream {
    lane: u32,
    nt: u32,
    rs: u32,
    rows: u32,
    pieces: u32,
    split: u32,
    slices: u32,
    depth: u32,
    blocks: u32,
}

impl Stream {
    /// The most activation rows the kernel takes.
    const ROWS: u32 = 8;

    fn plan(
        bits: u32,
        group: u32,
        offset: OffsetKind,
        n: u32,
        k: u32,
        rows: u32,
        cus: u32,
    ) -> Option<Self> {
        if rows > Self::ROWS {
            return None;
        }
        // Sixteen bytes a lane, or the most whole words of one group.
        let mut lane = 128 / bits;
        while lane > 8 && !group.is_multiple_of(lane) {
            lane /= 2;
        }
        let piece = 16 * lane;
        let piece_groups = piece / group;
        // K in whole pieces; a piece's factor rows whole 4-byte words.
        let off_bytes = if offset == OffsetKind::PreInt { 1 } else { 2 };
        let words = |bytes: u32| piece_groups * bytes / 4;
        if !group.is_multiple_of(lane)
            || !k.is_multiple_of(piece)
            || !piece.is_multiple_of(group)
            || words(2) == 0
            || (offset != OffsetKind::PreConst && words(off_bytes) == 0)
        {
            return None;
        }
        let rs = rows.div_ceil(4);
        let rows_used = if rs == 1 { rows } else { 4 };
        // Two tiles of four weight rows (four streamed no faster and hold
        // more registers), one where a piece's factors take two words a lane.
        let most_words = words(2).max(if offset == OffsetKind::PreConst {
            0
        } else {
            words(off_bytes)
        });
        let mut nt = 2;
        while nt > 1 && 4 * nt * most_words > 64 {
            nt /= 2;
        }
        if 4 * nt * most_words > 64 {
            return None;
        }
        // A wavefront holds the activation of two pieces for one activation
        // tile, of one for two (more spill out of the registers); sixteen
        // wavefronts a block, and K past what they hold sliced over the grid.
        let total = k / piece;
        let most = if rs == 1 { 2 } else { 1 };
        let slices = total.div_ceil(16 * most);
        let per = total.div_ceil(slices);
        let pieces = per.div_ceil(16);
        let tiles = n.div_ceil(4 * nt);
        Some(Self {
            lane,
            nt,
            rs,
            rows: rows_used,
            pieces,
            split: per.div_ceil(pieces),
            slices,
            // Two tiles in flight, one where the activation fills the
            // registers.
            depth: if pieces > 1 { 1 } else { 2 },
            // Up to two blocks a compute unit (their tiles' reads in flight,
            // more only queue), as few as land the same tiles each.
            blocks: tiles.div_ceil(tiles.div_ceil((2 * cus).div_ceil(slices))),
        })
    }
}

/// The decode kernel's tiling: the matrix-core `shape` (4 x 4 x 4 or
/// 16 x 16 x 16), `lane` codes a lane reads at once, `nt` and `rs` tiles
/// of weight and activation rows a wavefront, `split` wavefronts a block
/// over K, `round` pieces a wavefront reads at once.
#[derive(Clone, Copy, Debug)]
struct Decode {
    shape: u32,
    lane: u32,
    nt: u32,
    rs: u32,
    split: u32,
    round: u32,
}

impl Decode {
    fn plan(bits: u32, group: u32, n: u32, k: u32, rows: u32, cus: u32) -> Self {
        // Up to eight rows, the 4 x 4 x 4 core: sixteen dots a block of four
        // lanes, four weight rows by four activation rows, and a lane's
        // codes one group's. Past eight, the 16 x 16 x 16 core, whose
        // product spans four lanes' codes, so a lane holds a quarter group
        // at most.
        let (shape, rs, nt, most) = if rows <= 8 {
            (4, rows.div_ceil(4), 4, group)
        } else {
            let rs = rows.div_ceil(16).min(4);
            (16, rs, if rs <= 2 { 2 } else { 1 }, group / 4)
        };
        // Sixteen bytes a lane, or the largest power of two of codes that
        // fits: a group is whole words, so four bytes at least. The
        // activation's share (two bytes a code a row) stays within 128 bytes
        // a lane.
        let mut lane = (128 / bits).min(if rs > 2 { 16 } else { 64 / rs });
        while lane > 4 && !most.is_multiple_of(lane) {
            lane /= 2;
        }
        // Pieces a round: as many as about a hundred registers hold, since
        // the hand-issued reads must fit the VGPRs outright.
        let regs = nt * (lane * bits / 32).max(1) + rs * lane / 2 + 2 * nt;
        let round = (112 / regs).clamp(1, 4).min(if shape == 4 { 2 } else { 4 });
        let pieces = k.div_ceil(64 / shape * lane);
        let blocks = n.div_ceil(shape * nt);
        // Eight wavefronts a compute unit to keep the codes in flight, but
        // a round of pieces a wavefront at least, sixteen a block at most,
        // and the block's partial sums within 32 KiB of LDS.
        let cells = nt * rs * shape * shape * if shape == 4 { 4 } else { 1 };
        let want = (8 * cus).div_ceil(blocks).next_power_of_two();
        let mut split = want.min(16);
        while split > 1 && (round * split > pieces || split * cells * 4 > 32 << 10) {
            split /= 2;
        }
        Self {
            shape,
            lane,
            nt,
            rs,
            split,
            round,
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub fn decoded_plane(
    ctx: &Ctx,
    op: &'static str,
    codes: Tensor,
    scales: Tensor,
    offset: OffsetKind,
    biases: Option<Tensor>,
    factor: Dtype,
    n: u32,
    k: u32,
    seat: GroupSeat,
) -> Result<Tensor, Error> {
    let f = dtype_dispatch!(op, factor, { Bf16 => "::pie::bf16", F16 => "::pie::f16" });
    if seat.streams() {
        return Err(refuse(
            op,
            "a decoded plane serves resident projections only, and these planes are \
             seated by a streaming tier",
        ));
    }
    if codes.rows != n {
        return Err(refuse(
            op,
            format!(
                "the code plane holds {} rows and the entry states {n}",
                codes.rows
            ),
        ));
    }
    let bits: u32 = if codes.width == k {
        8
    } else if codes.width * 2 == k {
        4
    } else if codes.width * 4 == k {
        2
    } else {
        return Err(refuse(
            op,
            format!(
                "a {}-byte code row stores a {k}-wide row at neither two, four nor eight bits",
                codes.width
            ),
        ));
    };
    let groups = scales.width / 2;
    if groups == 0 || !k.is_multiple_of(groups) {
        return Err(refuse(
            op,
            format!("{groups} factors do not group a {k}-wide row into whole groups"),
        ));
    }
    let group = k / groups;
    let bytes = (n as usize).saturating_mul(k as usize).saturating_mul(2);
    let tile = ctx.scratch_stream(op, DECODED_WEIGHT, bytes)? as usize as u64;
    let words = extent(op, u64::from(n) * u64::from(k) / u64::from(32 / bits))?;
    ctx.fire(
        op,
        Fire::at(
            FILE,
            symbol(&format!(
                "::pie::linear::dequant_affine<{f}, ::pie::i32({bits}), {}, \
                 ::pie::i32({group})>",
                offset.axis()
            )),
        )
        .apply(Launch::flat(words, BLOCK)),
        &[
            codes.arg(),
            scales.arg(),
            biases.map_or(ArgValue::ABSENT, |b| b.arg()),
            ArgValue::Ptr(tile),
            stated(op, n)?.arg(),
            stated(op, k)?.arg(),
        ],
    )?;
    Ok(Tensor::new(tile, n, k, Dtype::Bf16))
}

#[allow(clippy::too_many_arguments)]
pub fn decode_into(
    ctx: &Ctx,
    op: &'static str,
    codes: Tensor,
    scales: Tensor,
    offset: OffsetKind,
    biases: Option<Tensor>,
    factor: Dtype,
    dst: u64,
    n: u32,
    k: u32,
) -> Result<Tensor, Error> {
    let f = dtype_dispatch!(op, factor, { Bf16 => "::pie::bf16", F16 => "::pie::f16" });
    if codes.rows != n {
        return Err(refuse(
            op,
            format!(
                "the code plane holds {} rows and the entry states {n}",
                codes.rows
            ),
        ));
    }
    let bits: u32 = if codes.width == k {
        8
    } else if codes.width * 2 == k {
        4
    } else if codes.width * 4 == k {
        2
    } else {
        return Err(refuse(
            op,
            format!(
                "a {}-byte code row stores a {k}-wide row at neither two, four nor eight bits",
                codes.width
            ),
        ));
    };
    let groups = scales.width / 2;
    if groups == 0 || !k.is_multiple_of(groups) {
        return Err(refuse(
            op,
            format!("{groups} factors do not group a {k}-wide row into whole groups"),
        ));
    }
    let group = k / groups;
    let words = extent(op, u64::from(n) * u64::from(k) / u64::from(32 / bits))?;
    ctx.fire(
        op,
        Fire::at(
            FILE,
            symbol(&format!(
                "::pie::linear::dequant_affine<{f}, ::pie::i32({bits}), {}, \
                 ::pie::i32({group})>",
                offset.axis()
            )),
        )
        .apply(Launch::flat(words, BLOCK)),
        &[
            codes.arg(),
            scales.arg(),
            biases.map_or(ArgValue::ABSENT, |b| b.arg()),
            ArgValue::Ptr(dst),
            stated(op, n)?.arg(),
            stated(op, k)?.arg(),
        ],
    )?;
    Ok(Tensor::new(dst, n, k, Dtype::Bf16))
}

#[allow(clippy::too_many_arguments)]
pub fn matmul_via_dense(
    ctx: &Ctx,
    act: Tensor,
    codes: Tensor,
    scales: Tensor,
    offset: OffsetKind,
    biases: Option<Tensor>,
    factor: Dtype,
    y: &mut Tensor,
    seat: GroupSeat,
) -> Result<(), Error> {
    dense_affine_via_dense(
        ctx,
        "linear.matmul",
        act,
        codes,
        scales,
        offset,
        biases,
        factor,
        y,
        seat,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn lm_head_via_dense(
    ctx: &Ctx,
    act: Tensor,
    codes: Tensor,
    scales: Tensor,
    offset: OffsetKind,
    biases: Option<Tensor>,
    factor: Dtype,
    y: &mut Tensor,
    seat: GroupSeat,
) -> Result<(), Error> {
    dense_affine_via_dense(
        ctx,
        "linear.lm_head",
        act,
        codes,
        scales,
        offset,
        biases,
        factor,
        y,
        seat,
    )
}

#[allow(clippy::too_many_arguments)]
fn dense_affine_via_dense(
    ctx: &Ctx,
    op: &'static str,
    act: Tensor,
    codes: Tensor,
    scales: Tensor,
    offset: OffsetKind,
    biases: Option<Tensor>,
    factor: Dtype,
    y: &mut Tensor,
    seat: GroupSeat,
) -> Result<(), Error> {
    dtype_dispatch!(op, act.dtype, { Bf16 => () });
    let f = dtype_dispatch!(op, factor, { Bf16 => "::pie::bf16", F16 => "::pie::f16" });
    if seat.streams() {
        return Err(refuse(
            op,
            "the decoded-tile arm serves resident dense projections only, and these \
             planes are seated by a streaming tier",
        ));
    }
    let Affine { bits, group, n, k } = affine(op, act, codes, scales, offset, biases, y)?;
    if y.rows == 0 {
        return Ok(());
    }
    let bytes = (n as usize).saturating_mul(k as usize).saturating_mul(2);
    let tile = ctx.scratch_stream(op, DECODED_WEIGHT, bytes)? as usize as u64;
    let words = extent(op, u64::from(n) * u64::from(k) / u64::from(32 / bits))?;
    ctx.fire(
        op,
        Fire::at(
            FILE,
            symbol(&format!(
                "::pie::linear::dequant_affine<{f}, ::pie::i32({bits}), {}, \
                 ::pie::i32({group})>",
                offset.axis()
            )),
        )
        .apply(Launch::flat(words, BLOCK)),
        &[
            codes.arg(),
            scales.arg(),
            biases.map_or(ArgValue::ABSENT, |b| b.arg()),
            ArgValue::Ptr(tile),
            stated(op, n)?.arg(),
            stated(op, k)?.arg(),
        ],
    )?;
    gemm::act_x_wt(ctx, op, act, Tensor::new(tile, n, k, Dtype::Bf16), None, y)
}

pub fn cast_fp32_to(ctx: &Ctx, src: Tensor, dst: &mut Tensor) -> Result<(), Error> {
    const OP: &str = "linear.quant_cast_fp32_to";
    debug_assert_eq!(src.dtype, Dtype::F32, "`{OP}` casts an f32 source");
    let t = dtype_dispatch!(OP, dst.dtype, { Bf16 => "::pie::bf16", F16 => "::pie::f16" });
    let n = dst.elements();
    let lanes = nonzero(OP, "the cast's element count", extent(OP, n)?)?;
    ctx.fire(
        OP,
        Fire::at(FILE, symbol(&format!("::pie::linear::cast_f32_to<{t}>")))
            .apply(Launch::flat(lanes, BLOCK)),
        &[src.arg(), dst.arg(), n.arg()],
    )
}

pub fn scale_rows(ctx: &Ctx, l: Tensor, buf: &mut Tensor) -> Result<(), Error> {
    const OP: &str = "linear.quant_scale_rows";
    let t = dtype_dispatch!(OP, buf.dtype, { Bf16 => "::pie::bf16", F16 => "::pie::f16" });
    nonzero(OP, "rows", buf.rows)?;
    let width = stated(OP, nonzero(OP, "width", buf.width)?)?;
    ctx.fire(
        OP,
        Fire::at(FILE, symbol(&format!("::pie::linear::scale_rows<{t}>")))
            .apply(route_rows(buf.rows, buf.width)),
        &[buf.arg(), l.arg(), width.arg()],
    )
}
