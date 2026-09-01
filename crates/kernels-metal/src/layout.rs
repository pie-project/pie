//! `Layout`: gathers, cuts, and slices — data movement with no arithmetic.
//! One entry per IR variant, plus the quantized embed-gather the driver
//! selects when the table is an affine bank.

use crate::error::Error;
use dtype::Dtype;

use crate::encode::{
    Arg, Ctx, Fire, Grid, dtype_dispatch, elementwise_rows, head_grid, head_group, nonzero,
    refuse, stated,
};
use crate::tensor::{Bank, Tensor};

pub fn embed(
    ctx: &Ctx<'_>,
    ids: Tensor,
    table: Tensor,
    vocab: u32,
    y: Tensor,
) -> Result<(), Error> {
    const OP: &str = "layout.embed";
    debug_assert_eq!(ids.dtype, Dtype::I32, "`{OP}` gathers by i32 token ids");
    let entry = dtype_dispatch!(OP, table.dtype, { Bf16 => "embed_bfloat16" });
    nonzero(OP, "the row count this embedding table states", vocab)?;
    debug_assert_eq!(
        ids.rows, y.rows,
        "the token ids handed over are the rows this gather lands"
    );
    ctx.fire(
        Fire::at("layout/embed.metal", entry).apply(Grid::of(
            elementwise_rows(OP, y.width, y.rows)?,
            [256, 1, 1],
        )),
        &[
            ids.arg(),
            table.arg(),
            y.arg_mut(),
            stated(OP, y.width)?.arg(),
            stated(OP, vocab)?.arg(),
        ],
    )
}

/// **THE CONCATENATING GATHER'S GEOMETRY** (qwen4's PLE). `ids` is one row of
/// `heads` ids per token and `y` is one row of `heads` table rows laid SIDE BY
/// SIDE, so the gather is [`embed`]'s own with the head axis folded into the
/// row axis: `(rows · heads)` slices of `y.width / heads`.
///
/// **THAT FOLD IS WHY THIS PLANE NEEDS NO SECOND SHADER BODY.** Both embed
/// points here already index their ids and their output by a flat slice number
/// (`embed.metal`'s `tid.y`, `embed_gather.metal`'s `gid.y`), and a
/// concatenation is exactly the same addressing at a different stride. The
/// CUDA twin writes two more kernels because its launch is one flat index that
/// has to divide `heads` back out; this one carves the axis in the grid.
///
/// **WHAT THE TWO POINTS DO NOT SHARE IS THE UNADDRESSABLE ID'S ANSWER**, and
/// there the twin's per-op split is the one this plane copies: `layout.embed`
/// clamps to row zero (`layout.cuh`'s `embed`) and `layout.embed_concat`
/// writes zero (`embed_concat.cuh`'s `embed_concat`). That is one `bool` stamp
/// over one body — `embed_bfloat16` and `embed_concat_bfloat16` out of
/// `embed.metal` — not a second shader.
fn concat_slices(op: &'static str, ids: Tensor, y: Tensor) -> Result<(u32, u32), Error> {
    let heads = nonzero(op, "the ids per row", ids.width)?;
    if y.width == 0 || y.width % heads != 0 {
        return Err(refuse(
            op,
            format!(
                "the {}-wide landing is not {heads} table rows side by side",
                y.width
            ),
        ));
    }
    debug_assert_eq!(
        ids.rows, y.rows,
        "the token rows handed over are the rows this gather lands"
    );
    let slices = ids.rows.checked_mul(heads).ok_or_else(|| {
        refuse(
            op,
            format!("the grid will not launch: {} rows x {heads} ids", ids.rows),
        )
    })?;
    Ok((slices, y.width / heads))
}

/// `heads` gathers per row, concatenated — the dense table.
pub fn embed_concat(
    ctx: &Ctx<'_>,
    ids: Tensor,
    table: Tensor,
    vocab: u32,
    y: Tensor,
) -> Result<(), Error> {
    const OP: &str = "layout.embed_concat";
    debug_assert_eq!(ids.dtype, Dtype::I32, "`{OP}` gathers by i32 token ids");
    let entry = dtype_dispatch!(OP, table.dtype, { Bf16 => "embed_concat_bfloat16" });
    nonzero(OP, "the row count this embedding table states", vocab)?;
    let (slices, width) = concat_slices(OP, ids, y)?;
    ctx.fire(
        Fire::at("layout/embed.metal", entry).apply(Grid::of(
            elementwise_rows(OP, width, slices)?,
            [256, 1, 1],
        )),
        &[
            ids.arg(),
            table.arg(),
            y.arg_mut(),
            stated(OP, width)?.arg(),
            stated(OP, vocab)?.arg(),
        ],
    )
}

/// `heads` gathers per row, concatenated — the affine bank, dequantized for
/// exactly the rows the step touches and never landed dense. qwen4's n-gram
/// table is twenty million rows of a `(4, 32)` triplet; nothing else about
/// this entry is different from [`embed_gather_mb_4bit`].
pub fn embed_concat_mb_4bit(
    ctx: &Ctx<'_>,
    ids: Tensor,
    table: Bank,
    vocab: u32,
    y: Tensor,
) -> Result<(), Error> {
    const OP: &str = "layout.embed_concat";
    debug_assert_eq!(ids.dtype, Dtype::I32, "`{OP}` gathers by i32 token ids");
    let (slices, width) = concat_slices(OP, ids, y)?;
    gather_mb_4bit(ctx, OP, ids, table, vocab, y, slices, width)
}

pub fn split_qkv(
    ctx: &Ctx<'_>,
    packed: Tensor,
    q_width: u32,
    kv_width: u32,
    q: Tensor,
    k: Tensor,
    v: Tensor,
) -> Result<(), Error> {
    const OP: &str = "layout.split_qkv";
    let entry = dtype_dispatch!(OP, packed.dtype, { Bf16 => "split_qkv_bf16" });
    ctx.fire(
        Fire::at("attn/split_qkv.metal", entry).apply(Grid::of(
            elementwise_rows(OP, packed.width, packed.rows)?,
            [256, 1, 1],
        )),
        &[
            packed.arg(),
            q.arg_mut(),
            k.arg_mut(),
            v.arg_mut(),
            q_width.arg(),
            kv_width.arg(),
        ],
    )
}

/// Deinterleaves per-head `(q, gate)` pairs from the packed projection.
pub fn split_q_gate(
    ctx: &Ctx<'_>,
    packed: Tensor,
    head_dim: u32,
    q: Tensor,
    gate: Tensor,
) -> Result<(), Error> {
    const OP: &str = "layout.split_q_gate";
    let entry = dtype_dispatch!(OP, packed.dtype, { Bf16 => "q_gate_split_bfloat16" });
    nonzero(OP, "the head width this cut walks", head_dim)?;
    if q.width == 0 || q.width % head_dim != 0 {
        return Err(refuse(
            OP,
            format!(
                "the {}-wide query half does not divide by the stated head width {head_dim}",
                q.width
            ),
        ));
    }
    let lanes = head_grid(OP, head_dim, q.width / head_dim, packed.rows)?;
    ctx.fire(
        Fire::at("elemwise/gate.metal", entry).apply(Grid::of(lanes, head_group(lanes))),
        &[
            packed.arg(),
            q.arg_mut(),
            gate.arg_mut(),
            stated(OP, head_dim)?.arg(),
            stated(OP, packed.width)?.arg(),
            stated(OP, q.width)?.arg(),
        ],
    )
}

/// Splits each row at column `width`.
pub fn split_rows(
    ctx: &Ctx<'_>,
    x: Tensor,
    width: u32,
    left: Tensor,
    right: Tensor,
) -> Result<(), Error> {
    const OP: &str = "layout.split_rows";
    let entry = dtype_dispatch!(OP, x.dtype, { Bf16 => "split_rows_bfloat16" });
    nonzero(OP, "the left half of this cut", left.width)?;
    nonzero(OP, "the right half of this cut", right.width)?;
    debug_assert_eq!(left.width, width, "the left half is the width this cut states");
    debug_assert_eq!(
        left.width + right.width,
        x.width,
        "the two halves cover the packed row"
    );
    ctx.fire(
        Fire::at("layout/deinterleave.metal", entry).apply(Grid::of(
            elementwise_rows(OP, x.width, x.rows)?,
            [256, 1, 1],
        )),
        &[
            x.arg(),
            left.arg_mut(),
            right.arg_mut(),
            stated(OP, left.width)?.arg(),
            stated(OP, right.width)?.arg(),
        ],
    )
}

/// Copies layer `layer`'s `width`-wide slice out of a stacked table.
pub fn select(
    ctx: &Ctx<'_>,
    table: Tensor,
    layer: u32,
    width: u32,
    y: Tensor,
) -> Result<(), Error> {
    const OP: &str = "layout.select";
    let entry = dtype_dispatch!(OP, table.dtype, { Bf16 => "select_slice_bfloat16" });
    nonzero(OP, "the slice width this select states", width)?;
    debug_assert_eq!(
        y.width, width,
        "the selected slice is the width the statement states"
    );
    let offset = layer.checked_mul(width).ok_or_else(|| {
        refuse(
            OP,
            format!("layer {layer}'s slice starts beyond any column: {layer} x {width}"),
        )
    })?;
    if offset
        .checked_add(width)
        .is_none_or(|end| end > table.width)
    {
        return Err(refuse(
            OP,
            format!(
                "the {}-wide relayed row does not reach layer {layer}'s slice at {offset}",
                table.width
            ),
        ));
    }
    ctx.fire(
        Fire::at("layout/deinterleave.metal", entry).apply(Grid::of(
            elementwise_rows(OP, y.width, y.rows)?,
            [256, 1, 1],
        )),
        &[
            table.arg(),
            y.arg_mut(),
            stated(OP, table.width)?.arg(),
            stated(OP, offset)?.arg(),
            stated(OP, width)?.arg(),
        ],
    )
}

fn affine_point(op: &'static str, group: i32, bits: i32) -> Result<usize, Error> {
    let g = match group {
        32 => 0,
        64 => 1,
        128 => 2,
        _ => {
            return Err(refuse(op, format!("no affine point at group size {group}")));
        }
    };
    let b = match bits {
        4 => 0,
        8 => 1,
        _ => {
            return Err(refuse(op, format!("no affine point at bit width {bits}")));
        }
    };
    Ok(g * 2 + b)
}

/// The embed gather over an affine-quantized table — what `layout.embed`
/// becomes when the driver resolves the table to a bank instead of a dense
/// weight.
///
/// The gather dequantizes ONE ROW per token rather than a rectangle, so
/// there is no tile to divide and no vector-vs-tile choice: the six stamped
/// points differ only in the `(group, bits)` pair the row carries.
pub fn embed_gather_mb_4bit(
    ctx: &Ctx<'_>,
    ids: Tensor,
    table: Bank,
    vocab: u32,
    y: Tensor,
) -> Result<(), Error> {
    const OP: &str = "layout.embed";
    debug_assert_eq!(
        ids.rows, y.rows,
        "the token ids handed over are the rows this gather lands"
    );
    gather_mb_4bit(ctx, OP, ids, table, vocab, y, y.rows, y.width)
}

/// The banked gather both embed points fire, over a slice count and a slice
/// width the caller carved — one row apiece for [`embed_gather_mb_4bit`],
/// `rows · heads` of `y.width / heads` for [`embed_concat_mb_4bit`].
///
/// **`vocab` IS AN OPERAND OF THIS GATHER AND NOT ONLY OF THE DENSE ONE.** A
/// bank is three planes read at one id — codes, scales, biases — so an id past
/// the table's rows is three reads off the end of the checkpoint, where the
/// dense entry's would be one. Both quantized embed points on the CUDA plane
/// state it (`embed_concat_mlxu4`'s `vocab`, reached from `layout.embed` via
/// `embed_mlx_affine` and from `layout.embed_concat` directly) and both zero
/// the row rather than read it; this states it too, and `embed_gather.metal`
/// answers the same way.
fn gather_mb_4bit(
    ctx: &Ctx<'_>,
    op: &'static str,
    ids: Tensor,
    table: Bank,
    vocab: u32,
    y: Tensor,
    slices: u32,
    width: u32,
) -> Result<(), Error> {
    const ENTRIES: [&str; 6] = [
        "embed_gather_mb_4bit_bfloat16_gs_32_b_4",
        "embed_gather_mb_4bit_bfloat16_gs_32_b_8",
        "embed_gather_mb_4bit_bfloat16_gs_64_b_4",
        "embed_gather_mb_4bit_bfloat16_gs_64_b_8",
        "embed_gather_mb_4bit_bfloat16_gs_128_b_4",
        "embed_gather_mb_4bit_bfloat16_gs_128_b_8",
    ];
    debug_assert_eq!(ids.dtype, Dtype::I32, "`{op}` gathers by i32 token ids");
    nonzero(op, "the row count this embedding table states", vocab)?;
    // Every one of the six is affine: the shader adds `bias` to `code *
    // scale` unconditionally, so a symmetric table read through it takes
    // whatever the unbound seat happens to hold.
    let Some(biases) = table.biases else {
        return Err(refuse(
            op,
            format!(
                "the table is a symmetric {}-bit bank in groups of {}, and \
                 `embed_gather.metal` stamps the affine gather alone",
                table.bits, table.group
            ),
        ));
    };
    let point = affine_point(
        op,
        i32::try_from(table.group).unwrap_or(i32::MAX),
        i32::try_from(table.bits).unwrap_or(i32::MAX),
    )?;
    let _ = y;
    ctx.fire(
        Fire::at("layout/embed_gather.metal", ENTRIES[point]).apply(Grid::of(
            elementwise_rows(op, width, slices)?,
            [256, 1, 1],
        )),
        &[
            table.codes.arg(),
            table.scales.arg(),
            biases.arg(),
            ids.arg(),
            y.arg_mut(),
            stated(op, width)?.arg(),
            stated(op, vocab)?.arg(),
        ],
    )
}

/// The two halves of one row permutation, which differ only in which way the
/// index is read — so they are one body, and the pair cannot drift apart into
/// a gather and a scatter that disagree about what the map means.
///
/// The row map is a fire table a shell assembles; no op names it, so the
/// trace-time validator never sees it and its dtype is refused on the same
/// footing as its length — the boundary rule at [`refuse`].
///
/// **A NON-NEGATIVE `i32` AND A `u32` ARE THE SAME BITS, WHICH IS WHY THE
/// SHADER READS `uint` AND THIS ENTRY DEMANDS `i32`.** The map's element
/// type is the CUDA twin's (`layout::gather_rows` refuses anything but i32,
/// and `engine-cuda` assembles it as i32), so a Metal shell handing this
/// entry the same vector must not have to transcode it; `row_gather.metal`
/// was written reading `uint` before either entry existed and reads the same
/// row numbers either way. The refusal is on the dtype the PLANE agrees on,
/// not on the one the shader spells.
fn move_rows(
    ctx: &Ctx<'_>,
    op: &'static str,
    entry: &'static str,
    wide: Tensor,
    tight: Tensor,
    index: Tensor,
    args: [Tensor; 2],
) -> Result<(), Error> {
    if index.dtype != Dtype::I32 {
        return Err(refuse(
            op,
            format!(
                "the fire rows this copy is handed are {:?}, and it reads an i32 row map",
                index.dtype
            ),
        ));
    }
    if index.rows != tight.rows {
        return Err(refuse(
            op,
            format!("{} rows to move and {} rows named", tight.rows, index.rows),
        ));
    }
    if wide.dtype != tight.dtype || wide.width != tight.width {
        return Err(refuse(
            op,
            format!(
                "the fire-wide rectangle is {} x {:?} and the compacted one {} x {:?}; \
                 a row copy does not reshape",
                wide.width, wide.dtype, tight.width, tight.dtype
            ),
        ));
    }
    let rows = nonzero(op, "rows to move", tight.rows)?;
    let width = nonzero(op, "the width of a row this copy moves", tight.width)?;
    ctx.fire(
        Fire::at("layout/row_gather.metal", entry)
            .apply(Grid::of(elementwise_rows(op, width, rows)?, [256, 1, 1])),
        &[
            args[0].arg(),
            args[1].arg_mut(),
            index.arg(),
            width.arg(),
            rows.arg(),
        ],
    )
}

/// **GATHER: the rows a fragmented window covers, laid down as one.**
///
/// `Fallback::Copy`'s first half. A windowed consumer P4 could not seat
/// stands over several intervals of the fire's rows; this reads them out of
/// `wide` in the order `index` names and writes them contiguously into
/// `tight`, so the consumer behind it is ONE launch over a rectangle rather
/// than one launch per interval.
///
/// `index` is `i32`, one entry per row of `tight`: the FIRE row that row
/// stands at. It is the caller's span list flattened, and the caller is the
/// only one who knows it — this entry checks the shapes agree and moves rows.
///
/// **THE ENTRY EXISTS; THE SHELL STILL SPLITS.** `engine_metal`'s window
/// machinery answers every fragmented window with `Fallback::Split` and its
/// `model_exec::fire::Serve` impl is the default one. That is no longer
/// because this plane publishes no gather — it publishes both halves, for
/// both elements a copied region moves — but because a copy is a WINDOW
/// before it is a kernel: the shell owes a union window, a row map staged
/// where the host may write it, rebased qo boundaries over that union and
/// re-cut per-space pool tables, and `engine_metal::window::Windows::of` is
/// handed neither the trace nor the fire's geometry it would take to build
/// them. Publishing the kernel entry is the half that lives here; the other
/// half is named where it is owed (`engine_metal::window::Windows::of`).
///
/// # Errors
///
/// [`Error::DtypeUnsupported`] for anything but bf16 and f32 — the two
/// elements a copied region's rectangles are, where the CUDA twin is
/// dtype-blind and moves bytes. An attention region owed a fallback row
/// reads bf16 activations and writes an f32 log-sum-exp column, so a pair
/// stamped for one of them would refuse the very window the copy exists for;
/// a third element is a third `instantiate_row_gather` line and nothing more.
/// A refusal for an index vector or a rectangle that does not match the one
/// beside it.
pub fn gather_rows(
    ctx: &Ctx<'_>,
    wide: Tensor,
    index: Tensor,
    tight: Tensor,
) -> Result<(), Error> {
    const OP: &str = "layout.gather_rows";
    let entry = dtype_dispatch!(OP, tight.dtype, {
        Bf16 => "row_gather_bfloat16",
        F32 => "row_gather_float32",
    });
    move_rows(ctx, OP, entry, wide, tight, index, [wide, tight])
}

/// **SCATTER: the answers put back where their rows came from.**
///
/// `Fallback::Copy`'s second half, and the same map as [`gather_rows`] read
/// the other way: row `i` of `tight` lands at fire row `index[i]` of `wide`.
/// The rows the window does NOT cover are not written, which is what keeps a
/// copy one consumer's slow path rather than a fact about the arena.
///
/// This is also the design's third multimodal op
/// (`.wiki/alto/multimodal.md` §2: `layout::scatter_rows(y, src, routes)` —
/// the tower's output landing in the image-placeholder token rows), which on
/// the CUDA plane needed no new kernel because the copy fallback had already
/// written one. This plane had written neither half, so both are here.
///
/// # Errors
///
/// As [`gather_rows`].
pub fn scatter_rows(
    ctx: &Ctx<'_>,
    tight: Tensor,
    index: Tensor,
    wide: Tensor,
) -> Result<(), Error> {
    const OP: &str = "layout.scatter_rows";
    let entry = dtype_dispatch!(OP, tight.dtype, {
        Bf16 => "row_scatter_bfloat16",
        F32 => "row_scatter_float32",
    });
    move_rows(ctx, OP, entry, wide, tight, index, [tight, wide])
}

/// The two folds' shared arithmetic: how many rows one block is, and how many
/// whole blocks the source has.
///
/// **FLOOR, AND THE TAIL IT DROPS IS RUNG PADDING.** `x.rows` is a RUNG
/// count and a patch ladder's rungs are not multiples of nine, so this folds
/// `x.rows / side²` whole blocks and reads nothing past them. That is safe
/// rather than lossy because every image's patch run is a whole number of
/// blocks — `get_aspect_ratio_preserving_size` rounds an image's height and
/// width DOWN to a multiple of `pooling_kernel_size · patch_size` — so the
/// real rows are a prefix of the rectangle and everything after the leading
/// `Σ patches_i / side²` output rows is padding folded with padding.
///
/// Both entries take it from here so the two can never disagree about where a
/// fold stops. It is the twin's `fold_extent`, transcribed.
fn fold_extent(op: &'static str, x: Tensor, side: u32) -> Result<(u32, u32), Error> {
    nonzero(op, "the folding square's side", side)?;
    nonzero(op, "the folded row's width", x.width)?;
    let block = side.checked_mul(side).ok_or_else(|| {
        refuse(
            op,
            format!("a {side}-wide folding square has no row count that fits a u32"),
        )
    })?;
    if x.rows < block {
        return Err(refuse(
            op,
            format!(
                "{} rows do not fill one {side}x{side} fold, and a fold with no whole block \
                 would leave the destination unwritten",
                x.rows
            ),
        ));
    }
    Ok((block, x.rows / block))
}

/// **THE SPATIAL POOL**: `y[j]` is the mean of rows `[j·side², (j+1)·side²)`
/// of `x`, over `x.rows / side²` output rows — gemma4's
/// `pooling_kernel_size: 3` (`.wiki/alto/multimodal.md` §6.5, §7.4), and this
/// plane's mirror of `kernels_cuda::layout_fold::pool_rows`.
///
/// # What this asks of the submission, and what it asks of the fire
///
/// Of the submission: **POOL-BLOCK-MAJOR PATCH ORDER** — an image's patches
/// are ordered so that each `side × side` square of its grid is contiguous.
/// That is multimodal §2's merge-block-major statute at `side` instead of 2,
/// and it is what turns a 2-D pool into a row reduction. Of the fire:
/// nothing. No position stream, no image indptr, no per-image grid width —
/// image runs are whole numbers of blocks by the preprocessor's own resize
/// rule, so a block never straddles two images and two images pool as one
/// concatenation.
///
/// **THE FOLD IS COMPACTING AND THE TAIL OF `y` IS NOT WRITTEN.** What says
/// "this row has no destination" downstream is a `-1` in `patch_routes` and
/// the scatter that honours it ([`scatter_live_rows`]) — not a rule this
/// entry could enforce, because it does not own the route vector.
///
/// # Errors
///
/// [`Error::DtypeUnsupported`] for anything but bf16; a refusal for a zero
/// `side`, a zero-wide row, a destination whose rows are a different width, a
/// rectangle with fewer rows than one block, and a destination too short to
/// hold the blocks the source has.
pub fn pool_rows(ctx: &Ctx<'_>, x: Tensor, side: u32, y: Tensor) -> Result<(), Error> {
    const OP: &str = "layout.pool_rows";
    let entry = dtype_dispatch!(OP, x.dtype, { Bf16 => "pool_rows_bfloat16" });
    debug_assert_eq!(y.dtype, x.dtype, "`{OP}` pools into the element it reads");

    let (block, out) = fold_extent(OP, x, side)?;
    if y.width != x.width {
        return Err(refuse(
            OP,
            format!(
                "the source rows are {} wide and the destination's are {}; a pool folds rows \
                 and never a row",
                x.width, y.width
            ),
        ));
    }
    if y.rows < out {
        return Err(refuse(
            OP,
            format!(
                "{} source rows fold into {out} pooled rows and the destination holds {}",
                x.rows, y.rows
            ),
        ));
    }
    let lanes = elementwise_rows(OP, x.width, out)?;
    ctx.fire(
        Fire::at("layout/fold.metal", entry).apply(Grid::of(lanes, [lanes[0].min(256), 1, 1])),
        &[
            x.arg(),
            y.arg_mut(),
            stated(OP, x.width)?.arg(),
            stated(OP, block)?.arg(),
        ],
    )
}

/// **THE MERGING FOLD**: `y[j]` is rows `[j·side², (j+1)·side²)` of `x` laid
/// end to end — `side²` rows of `width` becoming one row of `side²·width`
/// (`.wiki/alto/multimodal.md` §8.1, §8.3), and this plane's mirror of
/// `kernels_cuda::layout_fold::merge_rows`.
///
/// qwen's spatial merger: `Qwen3_5VisionPatchMerger.forward` opens
/// `x.view(-1, hidden_size · spatial_merge_size²)`, which is why
/// `merger.linear_fc1.weight` is `[4·hidden, 4·hidden]` on the 768-wide dev
/// tower.
///
/// # It asks exactly what [`pool_rows`] asks
///
/// The same MERGE-BLOCK-MAJOR patch order — which for this op is not a new
/// statute at all but the one multimodal §2 already made — the same nothing
/// from the fire, and the same tail rule: `x.rows / side²` rows written, the
/// rest of `y` untouched.
///
/// **AND ON A DENSE RECTANGLE IT IS THE IDENTITY COPY.** `[rows, width]` and
/// `[rows/block, block·width]` are the same bytes in the same order; the
/// launch exists because the IR gives one value one type, so a second type
/// needs a second value and a node to define it. Recorded so that a later
/// wave can retire it into a placement alias in the compiler without anyone
/// wondering whether the arithmetic changed. It did not; there is none.
///
/// # Errors
///
/// As [`pool_rows`], plus a refusal for a destination whose width is not
/// `side²` times the source's — the one shape mistake this op can make that
/// the pool cannot.
pub fn merge_rows(ctx: &Ctx<'_>, x: Tensor, side: u32, y: Tensor) -> Result<(), Error> {
    const OP: &str = "layout.merge_rows";
    let entry = dtype_dispatch!(OP, x.dtype, { Bf16 => "merge_rows_bfloat16" });
    debug_assert_eq!(y.dtype, x.dtype, "`{OP}` merges into the element it reads");

    let (block, out) = fold_extent(OP, x, side)?;
    let merged = block.checked_mul(x.width).ok_or_else(|| {
        refuse(
            OP,
            format!(
                "{block} rows of {} do not concatenate into a row that fits a u32",
                x.width
            ),
        )
    })?;
    if y.width != merged {
        return Err(refuse(
            OP,
            format!(
                "{block} rows of {} concatenate into {merged}, and the destination's rows \
                 are {} wide",
                x.width, y.width
            ),
        ));
    }
    if y.rows < out {
        return Err(refuse(
            OP,
            format!(
                "{} source rows fold into {out} merged rows and the destination holds {}",
                x.rows, y.rows
            ),
        ));
    }
    let lanes = elementwise_rows(OP, merged, out)?;
    ctx.fire(
        Fire::at("layout/fold.metal", entry).apply(Grid::of(lanes, [lanes[0].min(256), 1, 1])),
        &[x.arg(), y.arg_mut(), stated(OP, merged)?.arg()],
    )
}

/// **THE EMBED MERGE, WITH A DROP SENTINEL**: row `i` of `src` lands at token
/// row `routes[i]` of `y`, and a NEGATIVE `routes[i]` places it nowhere
/// (`.wiki/alto/multimodal.md` §8.6), mirroring
/// `kernels_cuda::layout_scatter_live::scatter_live_rows`.
///
/// [`scatter_rows`]' launch plus one comparison, and it is a second entry
/// rather than a guard inside the first because the existing op's contract —
/// every route names a row — is one its consumers rely on and should not be
/// widened underneath them. Two ops, two contracts, one body apart.
///
/// **WHY THE SENTINEL IS OWED AT ALL.** A compacting fold ([`pool_rows`],
/// [`merge_rows`]) answers `rows / side²` rows and leaves the rest of the
/// patch rectangle as whatever the arena held. `RuntimeInput::PatchRoutes` is
/// `[Dim::Patches]` — one destination per row of the FULL rectangle — so those
/// tail rows have route entries, and before this op there was no legal value
/// to put in them: the shell refuses `route < 0` by name and
/// [`scatter_rows`] would take a negative index as a write below the base of
/// the token rectangle.
///
/// `-1` is the value a submission writes; any negative is dropped, because a
/// kernel that distinguished `-1` from `-2` would be inventing a second
/// sentinel nobody declared. The bound at the OTHER end is still not this
/// entry's to check — a route past `y.rows` is a device write no arena faults
/// on, and the fire path validates the vector against the token row count
/// before the launch, exactly as it does for the unguarded twin.
///
/// # Errors
///
/// As [`scatter_rows`].
pub fn scatter_live_rows(
    ctx: &Ctx<'_>,
    src: Tensor,
    routes: Tensor,
    y: Tensor,
) -> Result<(), Error> {
    const OP: &str = "layout.scatter_live_rows";
    let entry = dtype_dispatch!(OP, src.dtype, {
        Bf16 => "row_scatter_live_bfloat16",
        F32 => "row_scatter_live_float32",
    });
    move_rows(ctx, OP, entry, y, src, routes, [src, y])
}

/// **THE GATHER THAT INTERPOLATES**: `y[r] = Σₜ weights[r, t] ·
/// table[ids[r, t]]` (`.wiki/alto/multimodal.md` §9.2), and this plane's
/// mirror of `kernels_cuda::layout_embed_weighted::embed_weighted`.
///
/// `ids` is `[rows, taps]` `i32` and `weights` is `[rows, taps]` `f32`; `taps`
/// is read off their width — 2 for gemma's separable table read at two taps,
/// 4 for a bilinear resample, 16 for bicubic — because the operands carry it
/// and a stated second spelling could disagree with them. `vocab` is the
/// table's row count, stated as [`embed`] states it.
///
/// # Why the position embedding is a gather at all
///
/// The towers store one learned position grid at `num_grid_per_side²` and
/// RESAMPLE it to each image's own grid, which an import cannot compute —
/// `checkpoint::contract::Expr` places bytes and does not do arithmetic. On
/// the native grid that is [`embed`] unchanged; off it, it is this.
///
/// # Errors
///
/// [`Error::DtypeUnsupported`] for a table that is not bf16; a refusal for an
/// `ids` that is not `i32` or a `weights` that is not `f32`, for the two
/// geometry rectangles disagreeing with each other or with `y`, for a zero
/// tap count, and for an empty output.
pub fn embed_weighted(
    ctx: &Ctx<'_>,
    ids: Tensor,
    weights: Tensor,
    table: Tensor,
    vocab: u32,
    y: Tensor,
) -> Result<(), Error> {
    const OP: &str = "layout.embed_weighted";
    let entry = dtype_dispatch!(OP, table.dtype, { Bf16 => "embed_weighted_bfloat16" });
    debug_assert_eq!(y.dtype, table.dtype, "`{OP}` gathers into the table's element");

    if ids.dtype != Dtype::I32 {
        return Err(refuse(
            OP,
            format!(
                "the taps this gather is handed are {:?}, and it reads i32 rows",
                ids.dtype
            ),
        ));
    }
    // **THE WEIGHTS ARE THE PREPROCESSOR'S ARITHMETIC AND NOT THE
    // ACTIVATION'S.** A bf16 weight would move the resample by more than the
    // gather it feeds, so the element is refused rather than converted.
    if weights.dtype != Dtype::F32 {
        return Err(refuse(
            OP,
            format!(
                "the interpolation weights are {:?}, and this gather reads f32 — they are the \
                 preprocessor's arithmetic and not the activation's",
                weights.dtype
            ),
        ));
    }
    if ids.rows != weights.rows || ids.width != weights.width {
        return Err(refuse(
            OP,
            format!(
                "{} x {} taps and {} x {} weights; every tap is weighted and every weight taps",
                ids.rows, ids.width, weights.rows, weights.width
            ),
        ));
    }
    if ids.rows != y.rows {
        return Err(refuse(
            OP,
            format!(
                "{} rows of taps land {} rows; a gather answers one row per index row",
                ids.rows, y.rows
            ),
        ));
    }
    let taps = stated(OP, nonzero(OP, "the taps per row", ids.width)?)?;
    let vocab = stated(OP, nonzero(OP, "the table's row count", vocab)?)?;
    ctx.fire(
        Fire::at("layout/embed_weighted.metal", entry).apply(Grid::of(
            elementwise_rows(OP, y.width, y.rows)?,
            [256, 1, 1],
        )),
        &[
            ids.arg(),
            weights.arg(),
            table.arg(),
            y.arg_mut(),
            stated(OP, y.width)?.arg(),
            vocab.arg(),
            taps.arg(),
        ],
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::encode::ArgValue;
    use crate::probe::Probe;

    fn bf16(buf: u32, rows: u32, width: u32) -> Tensor {
        Tensor::new(buf, rows, width, Dtype::Bf16)
    }

    fn map(rows: u32) -> Tensor {
        Tensor::new(7, rows, 1, Dtype::I32)
    }

    /// A four-bit affine bank of `vocab` rows at `width`, in groups of 32 —
    /// the shape qwen4's PLE table lands as.
    fn u4_bank(vocab: u32, width: u32) -> Bank {
        Bank {
            codes: Tensor::new(10, vocab, width / 8, Dtype::U32),
            scales: Tensor::new(11, vocab, width / 32, Dtype::Bf16),
            biases: Some(Tensor::new(12, vocab, width / 32, Dtype::Bf16)),
            group: 32,
            bits: 4,
        }
    }

    /// **THE TWO DENSE EMBED POINTS ARE ONE BODY AND TWO STAMPS.** Same file,
    /// same argument list, same grid arithmetic with the head axis folded into
    /// the row axis — and two entrypoints, because the twin answers an
    /// unaddressable id differently per op: `layout::embed` clamps to row zero
    /// (`layout.cuh`) where `layout::embed_concat` writes zero
    /// (`embed_concat.cuh`). A single stamp here would have to pick one of the
    /// twin's two answers and be wrong about the other op.
    #[test]
    fn the_dense_embed_points_share_a_body_and_split_on_the_unaddressable_id() {
        let plain = Probe::default();
        embed(&plain, Tensor::new(3, 8, 1, Dtype::I32), bf16(1, 256, 64), 256, bf16(2, 8, 64))
            .expect("the dense gather enqueues");
        let (pf, pa) = plain.only();
        assert_eq!(pf.file, "layout/embed.metal");
        assert_eq!(pf.entrypoint, "embed_bfloat16");
        assert_eq!(pf.lanes, [64, 8, 1]);

        let concat = Probe::default();
        embed_concat(
            &concat,
            Tensor::new(3, 8, 16, Dtype::I32),
            bf16(1, 256, 64),
            256,
            bf16(2, 8, 1024),
        )
        .expect("the concatenating gather enqueues");
        let (cf, ca) = concat.only();
        assert_eq!(cf.file, pf.file, "one shader file serves both points");
        assert_eq!(
            cf.entrypoint, "embed_concat_bfloat16",
            "the concat writes zero where the plain embed clamps, so it is its own stamp"
        );
        // The fold: sixteen ids per row over eight rows is 128 slices of 64,
        // which is the plain point's own launch at another stride.
        assert_eq!(cf.lanes, [64, 128, 1]);
        assert_eq!(cf.group, pf.group);
        // And the argument list does not move: the split is in the stamp.
        assert_eq!(ca[0], ArgValue::Buffer(3));
        assert_eq!(ca[1], ArgValue::Buffer(1));
        assert_eq!(ca[2], ArgValue::BufferMut(2));
        assert_eq!(ca[3], ArgValue::I32(64));
        assert_eq!(ca[4], ArgValue::I32(256), "the row count is stated");
        assert_eq!(pa, ca, "the two stamps take the same six arguments");
    }

    /// **THE BANKED GATHER STATES `vocab` TOO**, at both points. It reads
    /// three planes off one id — codes, scales and biases — where the dense
    /// entry reads one row, so an id past the table is the worse read of the
    /// two and not the exempt one. The CUDA twin's `embed_concat_mlxu4` takes
    /// `vocab` and both of its quantized embed points hand it over; so do
    /// these.
    #[test]
    fn the_banked_gather_states_the_row_count_at_both_points() {
        let one = Probe::default();
        embed_gather_mb_4bit(
            &one,
            Tensor::new(3, 8, 1, Dtype::I32),
            u4_bank(4096, 64),
            4096,
            bf16(2, 8, 64),
        )
        .expect("the one-row banked gather enqueues");
        let (f, a) = one.only();
        assert_eq!(f.file, "layout/embed_gather.metal");
        assert_eq!(f.entrypoint, "embed_gather_mb_4bit_bfloat16_gs_32_b_4");
        assert_eq!(f.lanes, [64, 8, 1]);
        assert_eq!(a[3], ArgValue::Buffer(3), "the ids");
        assert_eq!(a[4], ArgValue::BufferMut(2));
        assert_eq!(a[5], ArgValue::I32(64), "the slice width");
        assert_eq!(a[6], ArgValue::I32(4096), "the row count the guard reads");

        let concat = Probe::default();
        embed_concat_mb_4bit(
            &concat,
            Tensor::new(3, 8, 16, Dtype::I32),
            u4_bank(4096, 64),
            4096,
            bf16(2, 8, 1024),
        )
        .expect("the concatenating banked gather enqueues");
        let (cf, ca) = concat.only();
        assert_eq!(cf.entrypoint, f.entrypoint, "one banked shader, two callers");
        assert_eq!(cf.lanes, [64, 128, 1], "the head axis is folded into the grid");
        assert_eq!(ca[5], ArgValue::I32(64));
        assert_eq!(ca[6], ArgValue::I32(4096), "and the concat states it too");
    }

    /// A table of no rows cannot answer any id, and the banked point says so
    /// in the dense point's own words rather than firing a guard that refuses
    /// everything on the device.
    #[test]
    fn a_banked_table_of_no_rows_is_refused_by_name() {
        let probe = Probe::default();
        let why = embed_gather_mb_4bit(
            &probe,
            Tensor::new(3, 8, 1, Dtype::I32),
            u4_bank(4096, 64),
            0,
            bf16(2, 8, 64),
        )
        .expect_err("a table of no rows");
        assert!(format!("{why}").contains("row count this embedding table states"), "{why}");
        assert!(probe.fires().is_empty());
    }

    /// The pair is a permutation and its inverse: same file, same map, same
    /// grid — only the two buffers change places, and which one is written.
    #[test]
    fn the_two_halves_are_one_grid_read_two_ways() {
        let (wide, tight, index) = (bf16(1, 512, 2048), bf16(2, 40, 2048), map(40));

        let gather = Probe::default();
        gather_rows(&gather, wide, index, tight).expect("the gather enqueues");
        let (gf, ga) = gather.only();
        assert_eq!(gf.file, "layout/row_gather.metal");
        assert_eq!(gf.entrypoint, "row_gather_bfloat16");
        assert_eq!(gf.lanes, [2048, 40, 1]);
        assert_eq!(gf.group, [256, 1, 1]);
        assert_eq!(ga[0], ArgValue::Buffer(1));
        assert_eq!(ga[1], ArgValue::BufferMut(2));
        assert_eq!(ga[3], ArgValue::U32(2048));
        assert_eq!(ga[4], ArgValue::U32(40));

        let scatter = Probe::default();
        scatter_rows(&scatter, tight, index, wide).expect("the scatter enqueues");
        let (sf, sa) = scatter.only();
        assert_eq!(sf.file, gf.file);
        assert_eq!(sf.entrypoint, "row_scatter_bfloat16");
        assert_eq!(sf.lanes, gf.lanes);
        assert_eq!(sf.group, gf.group);
        // The one difference: the compacted rectangle is read and the
        // fire-wide one written.
        assert_eq!(sa[0], ArgValue::Buffer(2));
        assert_eq!(sa[1], ArgValue::BufferMut(1));
        assert_eq!(sa[2..], ga[2..]);
    }

    #[test]
    fn a_row_map_that_is_not_an_i32_vector_is_refused_by_name() {
        let probe = Probe::default();
        let why = scatter_rows(
            &probe,
            bf16(2, 8, 64),
            Tensor::new(7, 8, 1, Dtype::U32),
            bf16(1, 64, 64),
        )
        .expect_err("the row map is i32 on both planes");
        assert!(format!("{why}").contains("i32 row map"), "{why}");
        assert!(probe.fires().is_empty());
    }

    #[test]
    fn a_map_that_names_a_different_number_of_rows_is_refused_by_name() {
        let probe = Probe::default();
        let why = gather_rows(&probe, bf16(1, 64, 64), map(7), bf16(2, 8, 64))
            .expect_err("eight rows to move and seven named");
        assert!(format!("{why}").contains("rows named"), "{why}");
    }

    #[test]
    fn a_copy_does_not_reshape() {
        let probe = Probe::default();
        let why = gather_rows(&probe, bf16(1, 64, 128), map(8), bf16(2, 8, 64))
            .expect_err("a row copy does not reshape");
        assert!(format!("{why}").contains("does not reshape"), "{why}");
    }

    /// **THE LSE PLANE MOVES TOO, AND IT IS f32.** An attention region owed
    /// a fallback row reads bf16 activations and writes an f32 log-sum-exp
    /// column, so a pair stamped for one element would refuse half of the
    /// very window `Fallback::Copy` exists for. Same file, same grid, same
    /// map — only the instantiation the entry names changes.
    #[test]
    fn the_log_sum_exp_plane_moves_on_its_own_instantiation() {
        let lse = |buf: u32, rows: u32| Tensor::new(buf, rows, 1, Dtype::F32);

        let gather = Probe::default();
        gather_rows(&gather, lse(1, 512), map(40), lse(2, 40)).expect("the gather enqueues");
        let (gf, _) = gather.only();
        assert_eq!(gf.file, "layout/row_gather.metal");
        assert_eq!(gf.entrypoint, "row_gather_float32");

        let scatter = Probe::default();
        scatter_rows(&scatter, lse(2, 40), map(40), lse(1, 512)).expect("the scatter enqueues");
        let (sf, _) = scatter.only();
        assert_eq!(sf.file, gf.file);
        assert_eq!(sf.entrypoint, "row_scatter_float32");
    }

    /// The gap this plane carries and the CUDA twin does not: the twin moves
    /// BYTES and is blind to the element, so any rectangle is a rectangle it
    /// can move; here an element is a template argument and only the two a
    /// copied region actually holds are stamped.
    #[test]
    fn an_element_with_no_instantiation_is_refused_by_dtype() {
        let probe = Probe::default();
        let why = gather_rows(
            &probe,
            Tensor::new(1, 64, 8, Dtype::F16),
            map(8),
            Tensor::new(2, 8, 8, Dtype::F16),
        )
        .expect_err("this plane stamps the row copy for bf16 and f32");
        assert!(matches!(why, Error::DtypeUnsupported { .. }), "{why}");
    }

    /// **THE POOL FOLDS ROWS AND NEVER A ROW**, and the fold is the whole
    /// difference: the destination is `rows / side²` rows of the SAME width,
    /// and the grid is the destination's.
    #[test]
    fn the_pool_launches_over_the_folded_rows_at_the_unfolded_width() {
        let probe = Probe::default();
        // gemma's `pooling_kernel_size: 3` over its wide tower's hidden.
        pool_rows(&probe, bf16(1, 4096, 1152), 3, bf16(2, 512, 1152))
            .expect("the pool enqueues");
        let (f, a) = probe.only();
        assert_eq!(f.file, "layout/fold.metal");
        assert_eq!(f.entrypoint, "pool_rows_bfloat16");
        // 4096 / 9 = 455 whole blocks, and the tail is rung padding.
        assert_eq!(f.lanes, [1152, 455, 1]);
        assert_eq!(f.group, [256, 1, 1]);
        assert_eq!(a[0], ArgValue::Buffer(1));
        assert_eq!(a[1], ArgValue::BufferMut(2));
        assert_eq!(a[2], ArgValue::I32(1152));
        // The divisor the shader averages by is `side²` and not the live rows.
        assert_eq!(a[3], ArgValue::I32(9));
    }

    /// **THE MERGE FOLDS ROWS INTO A ROW**, which is the one shape the pool
    /// cannot make: `side²` rows of `width` become one row of `side²·width`,
    /// and the grid is that wider row's.
    #[test]
    fn the_merge_launches_over_the_concatenated_row() {
        let probe = Probe::default();
        // qwen35-d0.8b's tower: 768 wide, `spatial_merge_size: 2`.
        merge_rows(&probe, bf16(1, 1024, 768), 2, bf16(2, 256, 4 * 768))
            .expect("the merge enqueues");
        let (f, a) = probe.only();
        assert_eq!(f.file, "layout/fold.metal");
        assert_eq!(f.entrypoint, "merge_rows_bfloat16");
        assert_eq!(f.lanes, [3072, 256, 1]);
        assert_eq!(f.group, [256, 1, 1]);
        assert_eq!(a[2], ArgValue::I32(3072));
    }

    /// **THE TAIL IS FLOORED AND NOT REFUSED.** A patch ladder's rungs are
    /// not multiples of nine, and the rows the fold drops are provably rung
    /// padding — every image's run is a whole number of blocks.
    #[test]
    fn a_rung_that_is_not_a_whole_number_of_blocks_folds_its_prefix() {
        let probe = Probe::default();
        pool_rows(&probe, bf16(1, 100, 64), 3, bf16(2, 16, 64)).expect("the pool enqueues");
        // 100 / 9 = 11, and rows 99..100 are never read.
        assert_eq!(probe.only().0.lanes, [64, 11, 1]);
    }

    /// A fold with no whole block would leave the destination unwritten, so
    /// it is refused rather than launched at zero rows — the twin's rule,
    /// which is `fold_extent`'s and therefore both entries'.
    #[test]
    fn a_rectangle_thinner_than_one_block_is_refused_by_name() {
        let probe = Probe::default();
        let why = pool_rows(&probe, bf16(1, 8, 64), 3, bf16(2, 8, 64))
            .expect_err("eight rows do not fill one 3x3 fold");
        assert!(format!("{why}").contains("do not fill one 3x3 fold"), "{why}");

        let merged = merge_rows(&probe, bf16(1, 3, 64), 2, bf16(2, 8, 256))
            .expect_err("three rows do not fill one 2x2 fold");
        assert!(format!("{merged}").contains("do not fill one 2x2 fold"), "{merged}");
        assert!(probe.fires().is_empty());
    }

    /// The two folds disagree about the destination's WIDTH and the refusals
    /// say so in each one's own words.
    #[test]
    fn a_destination_of_the_wrong_width_is_refused_in_the_folds_own_words() {
        let probe = Probe::default();
        let pooled = pool_rows(&probe, bf16(1, 36, 64), 3, bf16(2, 4, 128))
            .expect_err("a pool does not widen a row");
        assert!(format!("{pooled}").contains("never a row"), "{pooled}");

        let merged = merge_rows(&probe, bf16(1, 36, 64), 3, bf16(2, 4, 64))
            .expect_err("a merge does widen a row, by exactly side²");
        assert!(format!("{merged}").contains("concatenate into 576"), "{merged}");
    }

    /// A destination too short to hold the blocks the source has is a fold
    /// that would write past it.
    #[test]
    fn a_destination_too_short_for_the_blocks_is_refused_by_name() {
        let probe = Probe::default();
        let why = pool_rows(&probe, bf16(1, 90, 64), 3, bf16(2, 4, 64))
            .expect_err("ten pooled rows do not fit four");
        assert!(format!("{why}").contains("the destination holds 4"), "{why}");
    }

    /// **THE LIVE SCATTER IS THE PLAIN ONE'S LAUNCH, ON ITS OWN POINT.**
    /// Same file, same grid, same argument list — the sentinel lives in the
    /// shader, which is why the two entries can share `move_rows` and still
    /// not share a contract.
    #[test]
    fn the_live_scatter_is_the_plain_scatters_launch_at_another_point() {
        let (wide, tight, index) = (bf16(1, 512, 768), bf16(2, 40, 768), map(40));

        let plain = Probe::default();
        scatter_rows(&plain, tight, index, wide).expect("the plain scatter enqueues");
        let (pf, pa) = plain.only();

        let live = Probe::default();
        scatter_live_rows(&live, tight, index, wide).expect("the live scatter enqueues");
        let (lf, la) = live.only();

        assert_eq!(lf.file, pf.file);
        assert_eq!(lf.entrypoint, "row_scatter_live_bfloat16");
        assert_eq!(lf.lanes, pf.lanes);
        assert_eq!(lf.group, pf.group);
        assert_eq!(la, pa);
    }

    /// The route vector's ELEMENT is the one thing this entry must not be
    /// talked out of: the shader reads it as `int` because the sign is the
    /// sentinel, where the plain scatter reads `uint` because its sign never
    /// mattered.
    #[test]
    fn a_route_vector_that_is_not_an_i32_vector_is_refused_by_name() {
        let probe = Probe::default();
        let why = scatter_live_rows(
            &probe,
            bf16(2, 8, 64),
            Tensor::new(7, 8, 1, Dtype::U32),
            bf16(1, 64, 64),
        )
        .expect_err("the route vector is i32 on both planes");
        assert!(format!("{why}").contains("i32 row map"), "{why}");
        assert!(probe.fires().is_empty());
    }

    /// **THE WEIGHTED GATHER READS THE TAP COUNT OFF ITS OPERANDS**, so a
    /// two-tap separable read and a four-tap bilinear one are the same entry
    /// at two grids' worth of the same shape.
    #[test]
    fn the_weighted_gather_reads_its_taps_off_the_operand() {
        let probe = Probe::default();
        let taps = |width| Tensor::new(3, 96, width, Dtype::I32);
        let weights = |width| Tensor::new(4, 96, width, Dtype::F32);

        embed_weighted(&probe, taps(2), weights(2), bf16(1, 10240, 1152), 10240, bf16(2, 96, 1152))
            .expect("gemma's two-tap separable read enqueues");
        let (f, a) = probe.only();
        assert_eq!(f.file, "layout/embed_weighted.metal");
        assert_eq!(f.entrypoint, "embed_weighted_bfloat16");
        assert_eq!(f.lanes, [1152, 96, 1]);
        assert_eq!(f.group, [256, 1, 1]);
        assert_eq!(a[0], ArgValue::Buffer(3));
        assert_eq!(a[1], ArgValue::Buffer(4));
        assert_eq!(a[2], ArgValue::Buffer(1));
        assert_eq!(a[3], ArgValue::BufferMut(2));
        assert_eq!(a[4], ArgValue::I32(1152));
        assert_eq!(a[5], ArgValue::I32(10240));
        assert_eq!(a[6], ArgValue::I32(2), "the taps are read off the operand");

        let bilinear = Probe::default();
        embed_weighted(
            &bilinear,
            taps(4),
            weights(4),
            bf16(1, 10240, 1152),
            10240,
            bf16(2, 96, 1152),
        )
        .expect("a four-tap bilinear resample enqueues");
        assert_eq!(bilinear.only().1[6], ArgValue::I32(4));
    }

    /// **THE WEIGHTS ARE THE PREPROCESSOR'S ARITHMETIC.** A bf16 weight would
    /// move the resample by more than the gather it feeds, so the element is
    /// refused rather than converted — and the taps stay `i32`.
    #[test]
    fn a_weight_plane_in_the_activations_element_is_refused_by_name() {
        let probe = Probe::default();
        let why = embed_weighted(
            &probe,
            Tensor::new(3, 8, 4, Dtype::I32),
            bf16(4, 8, 4),
            bf16(1, 256, 64),
            256,
            bf16(2, 8, 64),
        )
        .expect_err("the interpolation weights are f32");
        assert!(format!("{why}").contains("preprocessor's arithmetic"), "{why}");

        let ids = embed_weighted(
            &probe,
            Tensor::new(3, 8, 4, Dtype::F32),
            Tensor::new(4, 8, 4, Dtype::F32),
            bf16(1, 256, 64),
            256,
            bf16(2, 8, 64),
        )
        .expect_err("the taps are i32");
        assert!(format!("{ids}").contains("i32 rows"), "{ids}");
        assert!(probe.fires().is_empty());
    }

    /// Every tap is weighted and every weight taps, and a gather answers one
    /// row per index row.
    #[test]
    fn tap_geometry_that_disagrees_with_itself_is_refused_by_name() {
        let probe = Probe::default();
        let ragged = embed_weighted(
            &probe,
            Tensor::new(3, 8, 4, Dtype::I32),
            Tensor::new(4, 8, 2, Dtype::F32),
            bf16(1, 256, 64),
            256,
            bf16(2, 8, 64),
        )
        .expect_err("four taps and two weights");
        assert!(format!("{ragged}").contains("every tap is weighted"), "{ragged}");

        let rows = embed_weighted(
            &probe,
            Tensor::new(3, 8, 4, Dtype::I32),
            Tensor::new(4, 8, 4, Dtype::F32),
            bf16(1, 256, 64),
            256,
            bf16(2, 9, 64),
        )
        .expect_err("eight rows of taps do not land nine rows");
        assert!(format!("{rows}").contains("one row per index row"), "{rows}");
    }
}
