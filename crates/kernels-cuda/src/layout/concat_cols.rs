//! `layout.concat_cols`: inputs side by side, a row each's columns after
//! the one before's.

use crate::error::Error;

use crate::jit::{Arg, Ctx, Fire, Launch, refuse};
use crate::tensor::Tensor;

const FILE: &str = "layout/concat_cols.cuh";

const BLOCK: u32 = 256;

/// `xs` side by side into `y`: each a row's columns after the one before's.
pub fn concat_cols(ctx: &Ctx, xs: &[Tensor], y: &mut Tensor) -> Result<(), Error> {
    const OP: &str = "layout.concat_cols";
    let elem = y.dtype.bits();
    if elem % 8 != 0 {
        return Err(refuse(OP, "an element of whole bytes"));
    }
    let elem = elem / 8;
    let stride = u64::from(y.width) * elem;
    let int = |v: u64| i32::try_from(v).map_err(|_| refuse(OP, "a row past 32-bit offsets"));
    let mut at = 0u64;
    for x in xs {
        if x.dtype != y.dtype || x.rows != y.rows {
            return Err(refuse(OP, "the inputs share the output's element and rows"));
        }
        let width = u64::from(x.width) * elem;
        let wide = width % 16 == 0
            && stride % 16 == 0
            && at % 16 == 0
            && x.ptr % 16 == 0
            && y.ptr % 16 == 0;
        let per = if wide { width / 16 } else { width };
        let threads = u32::try_from(u64::from(y.rows) * per)
            .map_err(|_| refuse(OP, "the rows pass a 32-bit launch"))?;
        if threads > 0 {
            ctx.fire(
                OP,
                Fire::at(FILE, "::pie::layout::concat_cols").apply(Launch::flat(threads, BLOCK)),
                &[
                    x.arg(),
                    y.arg(),
                    int(u64::from(y.rows))?.arg(),
                    int(width)?.arg(),
                    int(stride)?.arg(),
                    int(at)?.arg(),
                    i32::from(wide).arg(),
                ],
            )?;
        }
        at += width;
    }
    if at != stride {
        return Err(refuse(
            OP,
            format!("the inputs fill {at} bytes of a {stride}-byte row"),
        ));
    }
    Ok(())
}
