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
    y: Tensor,
) -> Result<(), Error> {
    const OP: &str = "layout.embed";
    const ENTRIES: [&str; 6] = [
        "embed_gather_mb_4bit_bfloat16_gs_32_b_4",
        "embed_gather_mb_4bit_bfloat16_gs_32_b_8",
        "embed_gather_mb_4bit_bfloat16_gs_64_b_4",
        "embed_gather_mb_4bit_bfloat16_gs_64_b_8",
        "embed_gather_mb_4bit_bfloat16_gs_128_b_4",
        "embed_gather_mb_4bit_bfloat16_gs_128_b_8",
    ];
    debug_assert_eq!(ids.dtype, Dtype::I32, "`{OP}` gathers by i32 token ids");
    debug_assert_eq!(
        ids.rows, y.rows,
        "the token ids handed over are the rows this gather lands"
    );
    // Every one of the six is affine: the shader adds `bias` to `code *
    // scale` unconditionally, so a symmetric table read through it takes
    // whatever the unbound seat happens to hold.
    let Some(biases) = table.biases else {
        return Err(refuse(
            OP,
            format!(
                "the table is a symmetric {}-bit bank in groups of {}, and \
                 `embed_gather.metal` stamps the affine gather alone",
                table.bits, table.group
            ),
        ));
    };
    let point = affine_point(
        OP,
        i32::try_from(table.group).unwrap_or(i32::MAX),
        i32::try_from(table.bits).unwrap_or(i32::MAX),
    )?;
    ctx.fire(
        Fire::at("layout/embed_gather.metal", ENTRIES[point]).apply(Grid::of(
            elementwise_rows(OP, y.width, y.rows)?,
            [256, 1, 1],
        )),
        &[
            table.codes.arg(),
            table.scales.arg(),
            biases.arg(),
            ids.arg(),
            y.arg_mut(),
            stated(OP, y.width)?.arg(),
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
}
