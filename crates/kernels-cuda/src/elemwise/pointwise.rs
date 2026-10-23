use super::*;
use crate::error::Error;
use crate::jit::{Arg, Ctx, Fire, Launch, dtype_dispatch, nonzero, refuse, stated, symbol};
use crate::tensor::Tensor;
use dtype::Dtype;

const POINTWISE: &str = "elemwise/pointwise.cu";

pub fn residual_add(ctx: &Ctx, x: Tensor, y: &mut Tensor) -> Result<(), Error> {
    const OP: &str = "elemwise.residual_add";
    let t = dtype_dispatch!(OP, y.dtype, { Bf16 => "::pie::bf16", F16 => "::pie::f16" });
    let (launch, n) = elementwise(OP, *y)?;
    // Rows of whole 16-byte groups add eight lanes a thread.
    if y.dtype == Dtype::Bf16 && y.width.is_multiple_of(8) && (y.ptr | x.ptr).is_multiple_of(16) {
        let lanes = u32::try_from(n / 8).unwrap_or(u32::MAX);
        return ctx.fire(
            OP,
            Fire::at(POINTWISE, "::pie::elemwise::residual_add_vec8")
                .apply(Launch::flat(lanes, BLOCK).programmatic(ctx.tuning().pdl)),
            &[
                y.arg(),
                x.arg(),
                (n / 8).arg(),
                stated(OP, y.width / 8)?.arg(),
                ctx.stage(),
            ],
        );
    }
    ctx.fire(
        OP,
        Fire::at(
            POINTWISE,
            symbol(&format!("::pie::elemwise::residual_add<{t}>")),
        )
        .apply(launch),
        &[
            y.arg(),
            x.arg(),
            n.arg(),
            stated(OP, y.width)?.arg(),
            ctx.stage(),
        ],
    )
}

pub fn add_bias(ctx: &Ctx, bias: Tensor, out: &mut Tensor) -> Result<(), Error> {
    const OP: &str = "elemwise.add_bias";
    let t = dtype_dispatch!(OP, out.dtype, { Bf16 => "::pie::bf16", F16 => "::pie::f16", F32 => "float" });
    let tb = dtype_dispatch!(OP, bias.dtype, { Bf16 => "::pie::bf16", F16 => "::pie::f16", F32 => "float" });
    nonzero(OP, "rows", out.rows)?;
    let width = stated(OP, nonzero(OP, "the biased row's width", out.width)?)?;
    ctx.fire(
        OP,
        Fire::at(
            POINTWISE,
            symbol(&format!("::pie::elemwise::add_bias<{t}, {tb}>")),
        )
        .apply(route_rows(out.rows, out.width)),
        &[out.arg(), bias.arg(), width.arg(), ctx.stage()],
    )
}

pub fn silu_scaled(ctx: &Ctx, s: f32, x: &mut Tensor) -> Result<(), Error> {
    const OP: &str = "elemwise.act";
    let t = dtype_dispatch!(OP, x.dtype, { Bf16 => "::pie::bf16", F16 => "::pie::f16" });
    let (launch, n) = elementwise(OP, *x)?;
    ctx.fire(
        OP,
        Fire::at(
            POINTWISE,
            symbol(&format!("::pie::elemwise::silu_scaled<{t}>")),
        )
        .apply(launch),
        &[
            x.arg(),
            s.arg(),
            n.arg(),
            stated(OP, x.width)?.arg(),
            ctx.stage(),
        ],
    )
}

pub fn mul_scalar(ctx: &Ctx, s: f32, x: &mut Tensor) -> Result<(), Error> {
    const OP: &str = "elemwise.act";
    let t = dtype_dispatch!(OP, x.dtype, { Bf16 => "::pie::bf16", F16 => "::pie::f16", F32 => "float" });
    let (launch, n) = elementwise(OP, *x)?;
    ctx.fire(
        OP,
        Fire::at(
            POINTWISE,
            symbol(&format!("::pie::elemwise::mul_scalar<{t}>")),
        )
        .apply(launch),
        &[
            x.arg(),
            s.arg(),
            n.arg(),
            stated(OP, x.width)?.arg(),
            ctx.stage(),
        ],
    )
}

pub fn scale(ctx: &Ctx, s: Tensor, x: &mut Tensor) -> Result<(), Error> {
    const OP: &str = "elemwise.act";
    let t = dtype_dispatch!(OP, x.dtype, { Bf16 => "::pie::bf16", F16 => "::pie::f16" });
    let (launch, n) = elementwise(OP, *x)?;
    ctx.fire(
        OP,
        Fire::at(POINTWISE, symbol(&format!("::pie::elemwise::scale<{t}>")))
            .apply(launch.programmatic(ctx.tuning().pdl)),
        &[
            x.arg(),
            s.arg(),
            n.arg(),
            stated(OP, x.width)?.arg(),
            ctx.stage(),
        ],
    )
}

const BLOCK: u32 = 256;

pub fn silu(ctx: &Ctx, x: Tensor, o: &mut Tensor) -> Result<(), Error> {
    fire_unary(ctx, "elemwise.act", "0", x, o)
}

pub fn tanh(ctx: &Ctx, x: Tensor, o: &mut Tensor) -> Result<(), Error> {
    fire_unary(ctx, "elemwise.act", "1", x, o)
}

pub fn gelu_tanh(ctx: &Ctx, x: Tensor, o: &mut Tensor) -> Result<(), Error> {
    crate::linear::glu::gelu_tanh(ctx, x, 1, o)
}

fn fire_unary(
    ctx: &Ctx,
    op: &'static str,
    stamp: &str,
    x: Tensor,
    o: &mut Tensor,
) -> Result<(), Error> {
    let t = dtype_dispatch!(op, o.dtype, { Bf16 => "::pie::bf16", F16 => "::pie::f16", F32 => "float" });
    if x.dtype != o.dtype || x.rows < o.rows || x.width != o.width {
        return Err(refuse(
            op,
            format!(
                "the operand is {} x {} {:?} and the answer is {} x {} {:?}; an activation \
                 does not reshape",
                x.rows, x.width, x.dtype, o.rows, o.width, o.dtype
            ),
        ));
    }
    let n = o.elements();
    let lanes = u32::try_from(n).map_err(|_| {
        refuse(
            op,
            format!("{n} elements do not fit a 32-bit launch extent"),
        )
    })?;
    nonzero(op, "the element count", lanes)?;
    ctx.fire(
        op,
        Fire::at(
            POINTWISE,
            symbol(&format!("::pie::elemwise::activation<{t}, {stamp}>")),
        )
        .apply(Launch::flat(lanes, BLOCK)),
        &[
            x.arg(),
            o.arg(),
            stated(op, lanes)?.arg(),
            stated(op, o.width)?.arg(),
            ctx.stage(),
        ],
    )
}

pub fn add(ctx: &Ctx, x: Tensor, y: Tensor, o: &mut Tensor) -> Result<(), Error> {
    fire_binary(ctx, "elemwise.binary", "0", x, y, o)
}

pub fn mul(ctx: &Ctx, x: Tensor, y: Tensor, o: &mut Tensor) -> Result<(), Error> {
    fire_binary(ctx, "elemwise.binary", "1", x, y, o)
}

fn fire_binary(
    ctx: &Ctx,
    op: &'static str,
    stamp: &str,
    x: Tensor,
    y: Tensor,
    o: &mut Tensor,
) -> Result<(), Error> {
    let t = dtype_dispatch!(op, o.dtype, { Bf16 => "::pie::bf16", F16 => "::pie::f16", F32 => "float" });
    for (what, plane) in [("left", x), ("right", y)] {
        if plane.dtype != o.dtype || plane.rows < o.rows || plane.width != o.width {
            return Err(refuse(
                op,
                format!(
                    "the {what} operand is {} x {} {:?} and the answer is {} x {} {:?}; \
                     this op does not broadcast",
                    plane.rows, plane.width, plane.dtype, o.rows, o.width, o.dtype
                ),
            ));
        }
    }
    let n = o.elements();
    let lanes = u32::try_from(n).map_err(|_| {
        refuse(
            op,
            format!("{n} elements do not fit a 32-bit launch extent"),
        )
    })?;
    nonzero(op, "the element count", lanes)?;
    ctx.fire(
        op,
        Fire::at(
            POINTWISE,
            symbol(&format!("::pie::elemwise::binary<{t}, {stamp}>")),
        )
        .apply(Launch::flat(lanes, BLOCK)),
        &[
            x.arg(),
            y.arg(),
            o.arg(),
            stated(op, lanes)?.arg(),
            stated(op, o.width)?.arg(),
            ctx.stage(),
        ],
    )
}

pub fn clamp(ctx: &Ctx, lo: f32, hi: f32, x: &mut Tensor) -> Result<(), Error> {
    const OP: &str = "elemwise.clamp";
    let t = dtype_dispatch!(OP, x.dtype, { Bf16 => "::pie::bf16", F16 => "::pie::f16" });
    #[allow(clippy::neg_cmp_op_on_partial_ord)]
    if !(lo <= hi) {
        return Err(refuse(
            OP,
            format!(
                "the bounds {lo} and {hi} cross, and a clamp between them is the constant {hi}"
            ),
        ));
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
        Fire::at(POINTWISE, symbol(&format!("::pie::elemwise::clamp<{t}>")))
            .apply(Launch::flat(lanes, BLOCK)),
        &[
            x.arg(),
            lo.arg(),
            hi.arg(),
            n.arg(),
            stated(OP, x.width)?.arg(),
            ctx.stage(),
        ],
    )
}

pub fn clamp_learned(ctx: &Ctx, lo: Tensor, hi: Tensor, x: &mut Tensor) -> Result<(), Error> {
    const OP: &str = "elemwise.clamp";
    let t = dtype_dispatch!(OP, x.dtype, { Bf16 => "::pie::bf16", F16 => "::pie::f16" });
    for (what, bound) in [("lower", lo), ("upper", hi)] {
        if bound.dtype != x.dtype {
            return Err(refuse(
                OP,
                format!(
                    "the {what} bound is {:?} and the rows it clamps are {:?}; a learned bound \
                     rides the activation's element",
                    bound.dtype, x.dtype
                ),
            ));
        }
        if bound.elements() != 1 {
            return Err(refuse(
                OP,
                format!(
                    "the {what} bound is a {} x {} plane, and this clamp reads one scalar",
                    bound.rows, bound.width
                ),
            ));
        }
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
        Fire::at(
            POINTWISE,
            symbol(&format!("::pie::elemwise::clamp_learned<{t}>")),
        )
        .apply(Launch::flat(lanes, BLOCK)),
        &[
            x.arg(),
            lo.arg(),
            hi.arg(),
            n.arg(),
            stated(OP, x.width)?.arg(),
            ctx.stage(),
        ],
    )
}
