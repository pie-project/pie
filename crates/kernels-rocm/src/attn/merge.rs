use crate::error::Error;
use dtype::Dtype;

use crate::attn::elementwise;
use crate::attn::lse_plane;
use crate::attn::per_head_elementwise;
use crate::attn::row_heads;
use crate::jit::{Arg, Ctx, Fire, count, dtype_dispatch, kernel, nonzero, refuse, stated};
use crate::tensor::Tensor;

pub fn sink(
    ctx: &Ctx,
    o: &mut Tensor,
    lse: Tensor,
    sink: Tensor,
    head_dim: u32,
) -> Result<(), Error> {
    const OP: &str = "attn.sink";
    let t = dtype_dispatch!(OP, o.dtype, { Bf16 => "::pie::bf16", F16 => "::pie::f16" });
    // The sinks are the scores' own element, or f32 as a checkpoint keeps them.
    let s = match sink.dtype {
        Dtype::F32 => "float",
        held if held == o.dtype => t,
        held => {
            return Err(refuse(
                OP,
                format!("the sinks are {held:?} and the scores {:?}", o.dtype),
            ));
        }
    };
    let heads = row_heads(OP, o.width, head_dim)?;
    lse_plane(OP, &lse, o.rows, heads);
    let rows = count(OP, "rows", o.rows)?;
    let (vec, launch) = per_head_elementwise(o.rows, heads, head_dim);
    ctx.fire(
        OP,
        Fire::at(kernel(
            "attn/attention.hip",
            &format!("::pie::attn::sink_rescale<{t}, {s}, {vec}>"),
        ))
        .apply(launch),
        &[
            o.arg(),
            lse.arg(),
            sink.arg(),
            rows.arg(),
            stated(OP, heads)?.arg(),
            stated(OP, head_dim)?.arg(),
            ctx.stage(),
        ],
    )
}

#[allow(clippy::too_many_arguments)]
pub fn merge_lse(
    ctx: &Ctx,
    o1: Tensor,
    lse1: Tensor,
    o2: Tensor,
    lse2: Tensor,
    heads: u32,
    head_dim: u32,
    o: &mut Tensor,
    lse: &mut Tensor,
) -> Result<(), Error> {
    const OP: &str = "attn.merge_lse";

    let t = dtype_dispatch!(OP, o.dtype, { Bf16 => "::pie::bf16", F16 => "::pie::f16" });
    debug_assert!(
        o1.rows == o.rows && o2.rows == o.rows,
        "the merged outputs are one row per query row"
    );
    lse_plane(OP, lse, o.rows, heads);
    let (vec, launch) = per_head_elementwise(o.rows, heads, head_dim);
    let heads = count(OP, "the head count this merge states", heads)?;
    let head_dim = count(OP, "the head width this merge states", head_dim)?;
    ctx.fire(
        OP,
        Fire::at(kernel(
            "attn/attention.hip",
            &format!("::pie::attn::merge_lse_combine<{t}, {vec}>"),
        ))
        .apply(launch),
        &[
            o1.arg(),
            lse1.arg(),
            o2.arg(),
            lse2.arg(),
            o.arg(),
            lse.arg(),
            heads.arg(),
            head_dim.arg(),
            ctx.stage(),
        ],
    )
}

pub fn logit_softcap(ctx: &Ctx, x: &mut Tensor, cap: f32) -> Result<(), Error> {
    const OP: &str = "elemwise.softcap";
    let t = dtype_dispatch!(OP, x.dtype, { Bf16 => "::pie::bf16", F16 => "::pie::f16" });
    if cap.is_nan() || cap <= 0.0 {
        return Err(refuse(OP, format!("{cap} is not a logit soft cap")));
    }
    let n = x.elements();
    let lanes = u32::try_from(n).map_err(|_| {
        refuse(
            OP,
            format!("{n} elements do not fit a 32-bit launch extent"),
        )
    })?;
    nonzero(OP, "the element count", lanes)?;
    ctx.fire(
        OP,
        Fire::at(kernel(
            "attn/attention.hip",
            &format!("::pie::attn::logit_softcap<{t}>"),
        ))
        .apply(elementwise(lanes)),
        &[x.arg(), cap.arg(), n.arg()],
    )
}
