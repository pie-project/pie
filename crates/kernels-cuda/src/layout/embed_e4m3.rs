//! A hashed table stored e4m3 under one f32 scale (`Dtype::E4m3tensor`),
//! gathered and widened only at the rows its ids name, so the table stays
//! at a byte a value on the device.

use crate::error::Error;
use dtype::Dtype;

use crate::jit::{Arg, Ctx, Fire, Launch, dtype_dispatch, nonzero, refuse, stated, symbol};
use crate::tensor::Tensor;

const FILE: &str = "layout/embed_e4m3.cuh";

const BLOCK: u32 = 256;

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
    debug_assert_eq!(ids.dtype, Dtype::I32, "`{OP}` gathers by i32 rows");
    let t = dtype_dispatch!(OP, y.dtype, { Bf16 => "::pie::bf16", F16 => "::pie::f16" });
    let heads = nonzero(OP, "the hashed head count", ids.width)?;
    if y.width == 0 || !y.width.is_multiple_of(heads) {
        return Err(refuse(
            OP,
            format!(
                "the {}-wide output is not a whole number of {heads} head slices",
                y.width
            ),
        ));
    }
    let width = y.width / heads;
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
    debug_assert_eq!(y.rows, ids.rows, "one output row per id row");
    let rows = nonzero(OP, "rows", ids.rows)?;
    // Sixteen codes a thread where every row is whole 16-byte runs of an
    // aligned table: the same widening, code for code, in a sixteenth of
    // the requests (which is what counts where the table is mapped host
    // memory).
    let vec16 = width.is_multiple_of(16) && codes.ptr % 16 == 0;
    let (entry, per) = if vec16 {
        ("embed_concat_e4m3_vec16", width / 16)
    } else {
        ("embed_concat_e4m3", width)
    };
    let total = u64::from(rows) * u64::from(heads) * u64::from(per);
    let lanes = u32::try_from(total).map_err(|_| {
        refuse(
            OP,
            format!("{total} elements do not fit a 32-bit launch extent"),
        )
    })?;
    ctx.fire(
        OP,
        Fire::at(FILE, symbol(&format!("::pie::layout::{entry}<{t}>")))
            .apply(Launch::flat(lanes, BLOCK)),
        &[
            ids.arg(),
            codes.arg(),
            scale.arg(),
            y.arg(),
            stated(OP, rows)?.arg(),
            stated(OP, heads)?.arg(),
            stated(OP, width)?.arg(),
            stated(OP, vocab)?.arg(),
            ctx.stage(),
        ],
    )
}
