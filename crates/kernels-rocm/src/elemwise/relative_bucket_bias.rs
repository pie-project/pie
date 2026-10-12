use crate::error::Error;
use dtype::Dtype;

use crate::jit::{Arg, Ctx, Fire, Launch, dtype_dispatch, nonzero, refuse, stated, symbol};
use crate::tensor::Tensor;

const FILE: &str = "elemwise/relative_bucket_bias.cuh";

const BLOCK: u32 = 256;

pub fn relative_bucket_bias(
    ctx: &Ctx,
    embedding: Tensor,
    max_len: u32,
    num_buckets: u32,
    max_distance: f32,
    bidirectional: bool,
    y: &mut Tensor,
) -> Result<(), Error> {
    const OP: &str = "elementwise.relative_bucket_bias";
    let t = dtype_dispatch!(OP, embedding.dtype, { Bf16 => "::pie::bf16", F32 => "float" });
    let heads = nonzero(OP, "heads", y.rows)?;
    nonzero(OP, "max_len", max_len)?;
    let span = 2 * max_len - 1;
    if y.dtype != Dtype::F32 || y.width != span {
        return Err(refuse(
            OP,
            format!(
                "the table is {} x {} {:?}; this entry writes one f32 row of {span} \
                 (2 · {max_len} − 1) per head",
                y.rows, y.width, y.dtype
            ),
        ));
    }
    if embedding.rows < num_buckets || embedding.width < heads {
        return Err(refuse(
            OP,
            format!(
                "the bucket embedding is {} x {}, and the table reads {num_buckets} buckets of \
                 {heads} heads",
                embedding.rows, embedding.width
            ),
        ));
    }
    let directional_buckets = if bidirectional {
        num_buckets / 2
    } else {
        num_buckets
    };
    let max_exact = directional_buckets / 2;
    if max_exact == 0 {
        return Err(refuse(
            OP,
            format!("{num_buckets} buckets leave no exact band; the function needs at least 4"),
        ));
    }
    let ratio = f64::from(max_distance) / f64::from(max_exact);
    if ratio.is_nan() || ratio <= 1.0 || !ratio.is_finite() {
        return Err(refuse(
            OP,
            format!(
                "max_distance {max_distance} is at or below max_exact {max_exact}; the large \
                 band's logarithm has no answer"
            ),
        ));
    }
    let log_ratio = ratio.ln() as f32;
    let cells = u64::from(heads) * u64::from(span);
    let lanes = u32::try_from(cells).map_err(|_| {
        refuse(
            OP,
            format!("{cells} table cells do not fit a 32-bit launch extent"),
        )
    })?;
    ctx.fire(
        OP,
        Fire::at(
            FILE,
            symbol(&format!("::pie::elemwise::relative_bucket_bias<{t}>")),
        )
        .apply(Launch::flat(lanes, BLOCK)),
        &[
            embedding.arg(),
            y.arg(),
            stated(OP, heads)?.arg(),
            stated(OP, span)?.arg(),
            stated(OP, max_len)?.arg(),
            stated(OP, embedding.width)?.arg(),
            stated(OP, num_buckets)?.arg(),
            i32::from(bidirectional).arg(),
            log_ratio.arg(),
        ],
    )
}
