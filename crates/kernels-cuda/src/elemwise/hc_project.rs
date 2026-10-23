use crate::error::Error;
use dtype::Dtype;

use crate::elemwise::hc::MAX_HC_MULT;
use crate::jit::{Arg, Ctx, Fire, Launch, nonzero, refuse, stated, symbol};
use crate::tensor::Tensor;

const HC_PROJECT: &str = "elemwise/hc_project.cu";

const BLOCK: u32 = 256;

pub fn project(
    ctx: &Ctx,
    normed: Tensor,
    hc_fn: Tensor,
    stream_count: u32,
    mixes: &mut Tensor,
) -> Result<(), Error> {
    const OP: &str = "elemwise.hc_project";
    debug_assert!(
        normed.dtype == Dtype::F32 && hc_fn.dtype == Dtype::F32 && mixes.dtype == Dtype::F32,
        "`{OP}` projects an f32 row through an f32 plane into an f32 row"
    );
    debug_assert_eq!(
        mixes.rows, normed.rows,
        "the projection lands one mix row per stream row"
    );
    let fan = nonzero(OP, "the stream row this projection contracts", normed.width)?;
    if hc_fn.width != fan {
        return Err(refuse(
            OP,
            format!(
                "the dynamic plane contracts {} and the stream row is {fan} wide",
                hc_fn.width
            ),
        ));
    }
    if stream_count == 0 || stream_count > MAX_HC_MULT {
        return Err(refuse(
            OP,
            format!(
                "the stream count is {stream_count}, not one of the {MAX_HC_MULT} the \
                 mixers unroll"
            ),
        ));
    }
    let mix_hc = hc_fn.rows;
    let layer_row = 2 * stream_count + stream_count * stream_count;
    let independent_row = 2 * stream_count;
    if mixes.width != mix_hc
        || (mix_hc != layer_row && mix_hc != independent_row && mix_hc != stream_count)
    {
        return Err(refuse(
            OP,
            format!(
                "a {stream_count}-stream mix row is {layer_row} wide ({independent_row} without \
                 its combine) and a trunk collapse row is {stream_count}; the plane lands {} \
                 rows into a {}-wide row",
                hc_fn.rows, mixes.width
            ),
        ));
    }
    let rows = nonzero(OP, "rows", mixes.rows)?;
    // Four rows a block, a warp's mix columns every eighth (three of a
    // layer's 24), the row's width cut into slices across blocks where the
    // rows alone would leave the SMs idle (a decode fire). The kernel below
    // read a row once a mix column and walked it alone (DeepSeek-V4-Flash:
    // 8.3 s of a 32x4096 prefill, 3.3 ms a decode8 step).
    if ctx.tuning().hc_tile
        && fan.is_multiple_of(TILE_K)
        && rows >= TILE_FROM
        && mix_hc <= TILE_COLS
        && normed.ptr.is_multiple_of(16)
        && hc_fn.ptr.is_multiple_of(16)
    {
        return project_sliced(ctx, OP, normed, hc_fn, mixes, rows, fan, mix_hc, true);
    }
    if fan.is_multiple_of(4) && mix_hc <= 8 * PROJECT_COLS {
        return project_sliced(ctx, OP, normed, hc_fn, mixes, rows, fan, mix_hc, false);
    }
    ctx.fire(
        OP,
        Fire::at(HC_PROJECT, "::pie::elemwise::hc_project<256>")
            .apply(Launch::grid([rows * mix_hc, 1, 1], [BLOCK, 1, 1])),
        &[
            normed.arg(),
            hc_fn.arg(),
            mixes.arg(),
            stated(OP, fan)?.arg(),
            stated(OP, mix_hc)?.arg(),
            ctx.stage(),
        ],
    )
}

const PROJECT_ROWS: u32 = 4;
const PROJECT_COLS: u32 = 3;

/// `hc_project_tile`'s rows a block, column ceiling, shared tile, thread
/// groups over the tile and tiles in flight: 64 x 4 x 64 x 2 projects
/// DeepSeek-V4-Flash's 16384 rows of 16384 into 24 in 427 us where the
/// four-row split walk took 1339 (64 x 4 x 32 x 3: 506; 64 x 4 x 128 x 2,
/// past the default shared memory: 409).
const TILE_ROWS: u32 = 64;
const TILE_COLS: u32 = 24;
const TILE_K: u32 = 64;
const TILE_STAGES: u32 = 2;
const TILE_GROUPS: u32 = 4;
/// Below this many rows the four-row split walk serves: a tile of 64 would
/// mostly repeat its last row.
pub(crate) const TILE_FROM: u32 = 128;

/// Whether `hc_prenorm_project_tile` (and its tensor-core form) take
/// these planes: bf16 streams whole 64-element tiles wide, an f32 plane of
/// at most 24 mix columns over them, 16-byte aligned.
pub(crate) fn prenorm_fits(
    streams: &Tensor,
    hc_fn: &Tensor,
    mixes: &Tensor,
    stream_count: u32,
) -> bool {
    streams.dtype == Dtype::Bf16
        && hc_fn.dtype == Dtype::F32
        && mixes.dtype == Dtype::F32
        && hc_fn.width == streams.width
        && streams.width.is_multiple_of(TILE_K)
        && hc_fn.rows <= TILE_COLS
        && mixes.width == hc_fn.rows
        && mixes.rows == streams.rows
        && stream_count != 0
        && stream_count <= MAX_HC_MULT
        && [streams.ptr, hc_fn.ptr].iter().all(|at| at % 16 == 0)
}

/// Where `prenorm_launch` landed the projection: the mixes themselves, or
/// `slices` partial slices at `out`.
pub(crate) struct Landed {
    pub(crate) out: u64,
    pub(crate) slices: u32,
}

/// The projection's tile launch: a prefill's rows whole, a decode's few cut
/// across the row.
pub(crate) fn prenorm_launch(
    ctx: &Ctx,
    op: &'static str,
    streams: &Tensor,
    eps: f32,
    hc_fn: &Tensor,
    mixes: &Tensor,
) -> Result<Landed, Error> {
    // At most 8192 partial rows (800 KiB at 24 mixes), in shared scratch.
    const PARTIAL_ROWS: u32 = 8192;
    let fan = streams.width;
    let mix_hc = hc_fn.rows;
    let rows = nonzero(op, "rows", mixes.rows)?;
    let wide = rows >= TILE_FROM;
    // A decode's few rows (one block of them) cut the row 128 columns a
    // block: the 128 slices that leaves read the projection plane, cold at a
    // decode, over as many SMs (DeepSeek-V4 at 8 rows, planes cold: 7.0 ->
    // 5.0 us), and the fold's warps (or the gates', under `pre`) take them.
    // Past one block of rows, 256 columns a block, the 64 slices measured
    // no slower.
    let floor = if wide {
        512
    } else if rows <= 32 {
        128
    } else {
        256
    };
    if wide && ctx.tuning().hc_mma {
        return prenorm_launch_mma(ctx, op, streams, eps, hc_fn, mixes);
    }
    let (br, kt) = (if wide { TILE_ROWS } else { 32 }, TILE_K);
    let row_blocks = rows.div_ceil(br);
    let want = 2 * ctx.multiprocessors().unwrap_or(132);
    let slices = want
        .div_ceil(row_blocks)
        .clamp(1, (fan / floor).max(1))
        .min((PARTIAL_ROWS / rows).max(1));
    let slice = fan.div_ceil(slices).next_multiple_of(kt);
    let slices = fan.div_ceil(slice);
    let out = if slices > 1 {
        ctx.scratch_shared(
            op,
            "elemwise.hc_project_partial",
            (slices * rows * (mix_hc + 1)) as usize * core::mem::size_of::<f32>(),
        )? as usize as u64
    } else {
        mixes.ptr
    };
    // A prefill's rows 64 a block over four thread groups, a decode's 32 over
    // eight.
    let (kg, stages) = (if wide { TILE_GROUPS } else { 8 }, TILE_STAGES);
    let tiles = stages * (br * (kt + 8) * 2 + TILE_COLS * (kt + 4) * 4);
    let fold = (kg - 1) * br * (TILE_COLS + 4) * 4;
    let (entry, threads, smem) = (
        format!("::pie::elemwise::hc_prenorm_project_tile<{br}, {kg}, {kt}, {stages}>"),
        br * kg,
        tiles.max(fold),
    );
    ctx.fire(
        op,
        Fire::at(HC_PROJECT, symbol(&entry)).apply(
            Launch {
                smem,
                ..Launch::grid([row_blocks, slices, 1], [threads, 1, 1])
            }
            .programmatic(ctx.tuning().pdl),
        ),
        &[
            streams.arg(),
            hc_fn.arg(),
            crate::jit::ArgValue::Ptr(out),
            stated(op, fan)?.arg(),
            stated(op, mix_hc)?.arg(),
            stated(op, rows)?.arg(),
            stated(op, slice)?.arg(),
            eps.arg(),
            ctx.stage(),
        ],
    )?;
    Ok(Landed { out, slices })
}

/// [`prenorm_launch`] on the tensor cores (`hc_prenorm_project_mma_t`): the
/// weights split into two bf16 halves first (`hc_fn_split`), then 256 rows
/// a block, the row cut into as many slices as fill the multiprocessors
/// four times over (DeepSeek-V4-Flash, 16384 rows: 64 rows a warp and two
/// stages, 134 us where 16 rows a warp and three stages took 152).
fn prenorm_launch_mma(
    ctx: &Ctx,
    op: &'static str,
    streams: &Tensor,
    eps: f32,
    hc_fn: &Tensor,
    mixes: &Tensor,
) -> Result<Landed, Error> {
    const STAGES: u32 = 2;
    // A warp's row tiles of 8: 64 rows a warp, four warps a block.
    const TILES: u32 = 8;
    const ROWS: u32 = 32 * TILES;
    const PARTIAL_ROWS: u32 = 65536;
    let fan = streams.width;
    let mix_hc = hc_fn.rows;
    let rows = nonzero(op, "rows", mixes.rows)?;
    let split =
        ctx.scratch_shared(op, "elemwise.hc_fn_split", (48 * fan) as usize * 2)? as usize as u64;
    ctx.fire(
        op,
        Fire::at(HC_PROJECT, "::pie::elemwise::hc_fn_split").apply(Launch::grid(
            [(24 * fan).div_ceil(4 * 256), 1, 1],
            [256, 1, 1],
        )),
        &[
            hc_fn.arg(),
            crate::jit::ArgValue::Ptr(split),
            stated(op, fan)?.arg(),
            stated(op, mix_hc)?.arg(),
        ],
    )?;
    let row_blocks = rows.div_ceil(ROWS);
    let want = 4 * ctx.multiprocessors().unwrap_or(132);
    let slices = want
        .div_ceil(row_blocks)
        .clamp(1, (fan / 1024).max(1))
        .min((PARTIAL_ROWS / rows).max(1));
    let slice = fan.div_ceil(slices).next_multiple_of(TILE_K);
    let slices = fan.div_ceil(slice);
    let out = if slices > 1 {
        ctx.scratch_shared(
            op,
            "elemwise.hc_project_partial",
            (slices * rows * (mix_hc + 1)) as usize * core::mem::size_of::<f32>(),
        )? as usize as u64
    } else {
        mixes.ptr
    };
    let smem = STAGES * (ROWS + 48) * (TILE_K + 8) * 2;
    ctx.fire(
        op,
        Fire::at(
            HC_PROJECT,
            symbol(&format!(
                "::pie::elemwise::hc_prenorm_project_mma_t<{STAGES}, {TILES}>"
            )),
        )
        .apply(Launch {
            smem,
            ..Launch::grid([row_blocks, slices, 1], [128, 1, 1])
        }),
        &[
            streams.arg(),
            crate::jit::ArgValue::Ptr(split),
            crate::jit::ArgValue::Ptr(out),
            stated(op, fan)?.arg(),
            stated(op, mix_hc)?.arg(),
            stated(op, rows)?.arg(),
            stated(op, slice)?.arg(),
            eps.arg(),
            ctx.stage(),
        ],
    )?;
    Ok(Landed { out, slices })
}

/// The stream norm and the mix projection over it in one pass over the bf16
/// `streams` (`hc_prenorm_project_tile`, and
/// `hc_prenorm_project_fold` past one slice): `mixes = (x . w) *
/// rsqrt(mean(x^2) + eps)`. `Ok(false)` where the shapes are not the
/// tile's: the caller runs the two.
pub fn prenorm_project(
    ctx: &Ctx,
    streams: Tensor,
    eps: f32,
    hc_fn: Tensor,
    stream_count: u32,
    mixes: &mut Tensor,
) -> Result<bool, Error> {
    const OP: &str = "fused.hc_prenorm_project";
    if !prenorm_fits(&streams, &hc_fn, mixes, stream_count) {
        return Ok(false);
    }
    let landed = prenorm_launch(ctx, OP, &streams, eps, &hc_fn, mixes)?;
    if landed.slices > 1 {
        let (rows, mix_hc) = (mixes.rows, hc_fn.rows);
        // A few slices over many rows (the tensor-core form's) fold a
        // thread a row; a decode's many slices a warp a mix.
        let fold = if ctx.tuning().hc_mma && landed.slices <= 8 && rows >= 1024 {
            Fire::at(HC_PROJECT, "::pie::elemwise::hc_prenorm_project_fold_rows")
                .apply(Launch::grid([rows.div_ceil(256), 1, 1], [256, 1, 1]))
        } else {
            Fire::at(HC_PROJECT, "::pie::elemwise::hc_prenorm_project_fold").apply(Launch::grid(
                [(rows * mix_hc).div_ceil(8), 1, 1],
                [BLOCK, 1, 1],
            ))
        };
        ctx.fire(
            OP,
            fold,
            &[
                crate::jit::ArgValue::Ptr(landed.out),
                mixes.arg(),
                stated(OP, mix_hc)?.arg(),
                stated(OP, rows)?.arg(),
                stated(OP, landed.slices)?.arg(),
                stated(OP, streams.width)?.arg(),
                eps.arg(),
                ctx.stage(),
            ],
        )?;
    }
    Ok(true)
}

/// `hc_project` on `hc_project_tile` (`tile`) or `hc_project_split`, the
/// row cut into slices across blocks for a decode's few rows, and
/// `hc_project_fold` past one slice. At most 2048 partial rows: a region
/// captured on its own holds its own partial slab.
#[allow(clippy::too_many_arguments)]
fn project_sliced(
    ctx: &Ctx,
    op: &'static str,
    normed: Tensor,
    hc_fn: Tensor,
    mixes: &mut Tensor,
    rows: u32,
    fan: u32,
    mix_hc: u32,
    tile: bool,
) -> Result<(), Error> {
    const PARTIAL_ROWS: u32 = 2048;
    let want = 2 * ctx.multiprocessors().unwrap_or(132);
    let (br, kg, kt, stages) = (TILE_ROWS, TILE_GROUPS, TILE_K, TILE_STAGES);
    let (block_rows, floor, most, step) = if tile {
        (br, 512, u32::MAX, kt)
    } else {
        (PROJECT_ROWS, 256, 64, 4)
    };
    let row_blocks = rows.div_ceil(block_rows);
    let slices = want
        .div_ceil(row_blocks)
        .clamp(1, (fan / floor).max(1))
        .min(most)
        .min((PARTIAL_ROWS / rows).max(1));
    let slice = fan.div_ceil(slices).next_multiple_of(step);
    let slices = fan.div_ceil(slice);
    let out = if slices > 1 {
        ctx.scratch_shared(
            op,
            "elemwise.hc_project_partial",
            (slices * rows * mix_hc) as usize * core::mem::size_of::<f32>(),
        )? as usize as u64
    } else {
        mixes.ptr
    };
    let fire = if tile {
        Fire::at(
            HC_PROJECT,
            symbol(&format!(
                "::pie::elemwise::hc_project_tile<{br}, {kg}, {kt}, {stages}>"
            )),
        )
        .apply(Launch {
            smem: stages * (br + TILE_COLS) * (kt + 4) * 4,
            ..Launch::grid([row_blocks, slices, 1], [br * kg, 1, 1])
        })
    } else {
        Fire::at(
            HC_PROJECT,
            symbol(&format!(
                "::pie::elemwise::hc_project_split<{PROJECT_ROWS}, {PROJECT_COLS}>"
            )),
        )
        .apply(Launch::grid([row_blocks, slices, 1], [BLOCK, 1, 1]))
    };
    ctx.fire(
        op,
        fire,
        &[
            normed.arg(),
            hc_fn.arg(),
            crate::jit::ArgValue::Ptr(out),
            stated(op, fan)?.arg(),
            stated(op, mix_hc)?.arg(),
            stated(op, rows)?.arg(),
            stated(op, slice)?.arg(),
            ctx.stage(),
        ],
    )?;
    if slices == 1 {
        return Ok(());
    }
    ctx.fire(
        op,
        Fire::at(HC_PROJECT, "::pie::elemwise::hc_project_fold")
            .apply(Launch::flat(rows * mix_hc, BLOCK)),
        &[
            crate::jit::ArgValue::Ptr(out),
            mixes.arg(),
            stated(op, mix_hc)?.arg(),
            stated(op, rows)?.arg(),
            stated(op, slices)?.arg(),
            ctx.stage(),
        ],
    )
}
