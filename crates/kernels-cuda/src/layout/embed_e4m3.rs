//! A hashed table stored e4m3 under one f32 scale (`Dtype::E4m3tensor`) or
//! one ue8m0 exponent a 32 of each row (`Dtype::Mxfp8`), gathered and
//! widened only at the rows its ids name, so the table stays at a byte a
//! value on the device.

use crate::error::Error;
use dtype::Dtype;

use crate::jit::{Arg, Ctx, Fire, Launch, dtype_dispatch, nonzero, refuse, stated, symbol};
use crate::tensor::Tensor;

const FILE: &str = "layout/embed_e4m3.cuh";

const BLOCK: u32 = 256;

/// The heads `ids` names a row and each one's width of `y`.
fn head_width(op: &'static str, ids: Tensor, y: &Tensor) -> Result<(u32, u32), Error> {
    debug_assert_eq!(ids.dtype, Dtype::I32, "`{op}` gathers by i32 rows");
    let heads = nonzero(op, "the hashed head count", ids.width)?;
    if y.width == 0 || !y.width.is_multiple_of(heads) {
        return Err(refuse(
            op,
            format!(
                "the {}-wide output is not a whole number of {heads} head slices",
                y.width
            ),
        ));
    }
    debug_assert_eq!(y.rows, ids.rows, "one output row per id row");
    Ok((heads, y.width / heads))
}

/// The gather a thread `per` of each head's slice of a row.
#[allow(clippy::too_many_arguments)]
fn gather(
    ctx: &Ctx,
    op: &'static str,
    entry: &str,
    ids: Tensor,
    codes: Tensor,
    scale: Tensor,
    vocab: u32,
    (heads, width, per): (u32, u32, u32),
    y: &mut Tensor,
) -> Result<(), Error> {
    let t = dtype_dispatch!(op, y.dtype, { Bf16 => "::pie::bf16", F16 => "::pie::f16" });
    let rows = nonzero(op, "rows", ids.rows)?;
    let total = u64::from(rows) * u64::from(heads) * u64::from(per);
    let lanes = u32::try_from(total).map_err(|_| {
        refuse(
            op,
            format!("{total} elements do not fit a 32-bit launch extent"),
        )
    })?;
    let entry = entry.replace('T', t);
    ctx.fire(
        op,
        Fire::at(FILE, symbol(&format!("::pie::layout::{entry}")))
            .apply(Launch::flat(lanes, BLOCK)),
        &[
            ids.arg(),
            codes.arg(),
            scale.arg(),
            y.arg(),
            stated(op, rows)?.arg(),
            stated(op, heads)?.arg(),
            stated(op, width)?.arg(),
            stated(op, vocab)?.arg(),
            ctx.stage(),
        ],
    )
}

/// `y[r, h * width ..]` is `codes[ids[r, h]] · scale`: `codes` the `[vocab,
/// width]` table bound as bytes, `scale` its one f32 bound as four bytes.
pub fn embed_concat_e4m3(
    ctx: &Ctx,
    ids: Tensor,
    codes: Tensor,
    scale: Tensor,
    vocab: u32,
    y: &mut Tensor,
) -> Result<(), Error> {
    const OP: &str = "layout.embed_concat";
    let (heads, width) = head_width(OP, ids, y)?;
    let held = u64::from(codes.rows) * u64::from(codes.width);
    if held != u64::from(vocab) * u64::from(width) || scale.rows * scale.width != 4 {
        return Err(refuse(
            OP,
            format!(
                "a {held}-byte table under a {}-byte scale is not [{vocab} x {width}] e4m3 \
                 under one f32",
                scale.rows * scale.width
            ),
        ));
    }
    // Sixteen codes a thread where every row is whole 16-byte runs of an
    // aligned table: the same widening in a sixteenth of the requests.
    let (entry, per) = if width.is_multiple_of(16) && codes.ptr.is_multiple_of(16) {
        ("embed_concat_e4m3_vec16<T, false>", width / 16)
    } else {
        ("embed_concat_e4m3<T>", width)
    };
    gather(
        ctx,
        OP,
        entry,
        ids,
        codes,
        scale,
        vocab,
        (heads, width, per),
        y,
    )
}

/// `y[r, h * width ..]` is `codes[ids[r, h]]` under its row's exponents:
/// `codes` the `[vocab, width]` MXFP8 table's e4m3 codes, `exps` its
/// `[vocab, width / 32]` ue8m0 exponents.
pub fn embed_concat_mxfp8(
    ctx: &Ctx,
    ids: Tensor,
    codes: Tensor,
    exps: Tensor,
    vocab: u32,
    y: &mut Tensor,
) -> Result<(), Error> {
    const OP: &str = "layout.embed_concat";
    let (heads, width) = head_width(OP, ids, y)?;
    if !width.is_multiple_of(32) || !codes.ptr.is_multiple_of(16) {
        return Err(refuse(
            OP,
            format!("an MXFP8 row is whole 32-wide blocks on 16-byte runs, not {width} wide"),
        ));
    }
    let held = u64::from(codes.rows) * u64::from(codes.width);
    let scales = u64::from(exps.rows) * u64::from(exps.width);
    if held != u64::from(vocab) * u64::from(width) || scales != held / 32 {
        return Err(refuse(
            OP,
            format!(
                "{held} codes under {scales} exponents are not a [{vocab} x {width}] MXFP8 table"
            ),
        ));
    }
    let entry = "embed_concat_e4m3_vec16<T, true>";
    gather(
        ctx,
        OP,
        entry,
        ids,
        codes,
        exps,
        vocab,
        (heads, width, width / 16),
        y,
    )
}
