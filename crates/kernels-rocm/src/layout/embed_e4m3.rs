use crate::error::Error;
use dtype::Dtype;

use crate::jit::{Arg, Ctx, Fire, Launch, dtype_dispatch, nonzero, refuse, stated, symbol};
use crate::tensor::Tensor;

const FILE: &str = "layout/embed_concat.hip";

const BLOCK: u32 = 256;

/// The rows of an MXFP8 table (`codes` e4m3 `[vocab, width]` beside
/// `scales`, one e8m0 exponent a 1x32 block) the ids name, side by side.
pub fn embed_concat_mxfp8(
    ctx: &Ctx,
    ids: Tensor,
    codes: Tensor,
    scales: Tensor,
    vocab: u32,
    y: &mut Tensor,
) -> Result<(), Error> {
    const OP: &str = "layout.embed_concat";
    debug_assert_eq!(ids.dtype, Dtype::I32, "`{OP}` gathers by i32 rows");
    let t = dtype_dispatch!(OP, y.dtype, { Bf16 => "::pie::bf16", F16 => "::pie::f16" });
    let heads = nonzero(OP, "the hashed head count", ids.width)?;
    let width = nonzero(OP, "the table row width", codes.width)?;
    if !width.is_multiple_of(32)
        || scales.width != width / 32
        || scales.rows != codes.rows
        || codes.rows < vocab
    {
        return Err(refuse(
            OP,
            format!(
                "a [{} x {}] exponent plane does not block a [{} x {width}] MXFP8 table of {vocab} rows in 1x32s",
                scales.rows, scales.width, codes.rows
            ),
        ));
    }
    if y.width != heads * width || y.rows != ids.rows {
        return Err(refuse(
            OP,
            "the output row is every head's table row, side by side",
        ));
    }
    let rows = nonzero(OP, "rows", ids.rows)?;
    let total = u64::from(rows) * u64::from(heads) * u64::from(width);
    let lanes = u32::try_from(total).map_err(|_| {
        refuse(
            OP,
            format!("{total} elements do not fit a 32-bit launch extent"),
        )
    })?;
    ctx.fire(
        OP,
        Fire::at(
            FILE,
            symbol(&format!("::pie::layout::embed_concat_mxfp8<{t}>")),
        )
        .apply(Launch::flat(lanes, BLOCK)),
        &[
            ids.arg(),
            codes.arg(),
            scales.arg(),
            y.arg(),
            stated(OP, rows)?.arg(),
            stated(OP, heads)?.arg(),
            stated(OP, width)?.arg(),
            stated(OP, vocab)?.arg(),
            ctx.stage(),
        ],
    )
}
