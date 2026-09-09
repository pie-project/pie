use crate::encode::{Arg, Ctx, Fire, Grid, dtype_dispatch, elementwise, nonzero, refuse};
use crate::error::Error;
use crate::tensor::Tensor;

const FILE: &str = "elemwise/pointwise.metal";

const GROUP: [u32; 3] = [256, 1, 1];

fn binary(
    ctx: &Ctx<'_>,
    op: &'static str,
    entry: &'static str,
    x: Tensor,
    y: Tensor,
    z: Tensor,
) -> Result<(), Error> {
    debug_assert!(
        x.rows == z.rows && x.width == z.width && y.rows == z.rows && y.width == z.width,
        "`{op}` walks one rectangle"
    );
    ctx.fire(
        Fire::at(FILE, entry).apply(Grid::of(elementwise(op, z.width, z.rows)?, GROUP)),
        &[x.arg(), y.arg(), z.arg_mut()],
    )
}

pub fn add(ctx: &Ctx<'_>, x: Tensor, y: Tensor, z: Tensor) -> Result<(), Error> {
    const OP: &str = "elementwise.add";
    let entry = dtype_dispatch!(OP, z.dtype, {
        Bf16 => "binary_add_bfloat16",
        F32 => "binary_add_float32",
    });
    binary(ctx, OP, entry, x, y, z)
}

pub fn mul(ctx: &Ctx<'_>, x: Tensor, y: Tensor, z: Tensor) -> Result<(), Error> {
    const OP: &str = "elementwise.mul";
    let entry = dtype_dispatch!(OP, z.dtype, {
        Bf16 => "binary_mul_bfloat16",
        F32 => "binary_mul_float32",
    });
    binary(ctx, OP, entry, x, y, z)
}

fn activation(
    ctx: &Ctx<'_>,
    op: &'static str,
    entry: &'static str,
    x: Tensor,
    o: Tensor,
) -> Result<(), Error> {
    debug_assert!(
        x.rows == o.rows && x.width == o.width,
        "`{op}` writes the rectangle it reads"
    );
    ctx.fire(
        Fire::at(FILE, entry).apply(Grid::of(elementwise(op, o.width, o.rows)?, GROUP)),
        &[x.arg(), o.arg_mut()],
    )
}

pub fn silu(ctx: &Ctx<'_>, x: Tensor, o: Tensor) -> Result<(), Error> {
    const OP: &str = "elementwise.silu";
    let entry = dtype_dispatch!(OP, o.dtype, {
        Bf16 => "act_silu_bfloat16",
        F32 => "act_silu_float32",
    });
    activation(ctx, OP, entry, x, o)
}

pub fn tanh(ctx: &Ctx<'_>, x: Tensor, o: Tensor) -> Result<(), Error> {
    const OP: &str = "elementwise.tanh";
    let entry = dtype_dispatch!(OP, o.dtype, {
        Bf16 => "act_tanh_bfloat16",
        F32 => "act_tanh_float32",
    });
    activation(ctx, OP, entry, x, o)
}

pub fn gelu_tanh(ctx: &Ctx<'_>, x: Tensor, o: Tensor) -> Result<(), Error> {
    const OP: &str = "elementwise.gelu";
    let entry = dtype_dispatch!(OP, o.dtype, {
        Bf16 => "act_gelu_tanh_bfloat16",
        F32 => "act_gelu_tanh_float32",
    });
    activation(ctx, OP, entry, x, o)
}

#[allow(clippy::neg_cmp_op_on_partial_ord)]
pub fn clamp(ctx: &Ctx<'_>, lo: f32, hi: f32, x: Tensor) -> Result<(), Error> {
    const OP: &str = "elementwise.clamp";
    let entry = dtype_dispatch!(OP, x.dtype, {
        Bf16 => "clamp_bounds_bfloat16",
        F32 => "clamp_bounds_float32",
    });
    if !(lo <= hi) {
        return Err(refuse(
            OP,
            format!("the bound is [{lo}, {hi}], which admits no value"),
        ));
    }
    ctx.fire(
        Fire::at(FILE, entry).apply(Grid::of(elementwise(OP, x.width, x.rows)?, GROUP)),
        &[x.arg_mut(), lo.arg(), hi.arg()],
    )
}

pub fn clamp_learned(ctx: &Ctx<'_>, lo: Tensor, hi: Tensor, x: Tensor) -> Result<(), Error> {
    const OP: &str = "elementwise.clamp_learned";
    let entry = dtype_dispatch!(OP, x.dtype, {
        Bf16 => "clamp_learned_bfloat16",
        F32 => "clamp_learned_float32",
    });
    for (what, bound) in [("lower", lo), ("upper", hi)] {
        if bound.dtype != x.dtype {
            return Err(refuse(
                OP,
                format!(
                    "the {what} bound is {:?} and the rows it clamps are {:?}; a learned \
                     bound rides the activation's element",
                    bound.dtype, x.dtype
                ),
            ));
        }
        if u64::from(bound.rows) * u64::from(bound.width) != 1 {
            return Err(refuse(
                OP,
                format!(
                    "the {what} bound is a {} x {} plane, and this clamp reads one scalar",
                    bound.rows, bound.width
                ),
            ));
        }
    }
    ctx.fire(
        Fire::at(FILE, entry).apply(Grid::of(elementwise(OP, x.width, x.rows)?, GROUP)),
        &[x.arg_mut(), lo.arg(), hi.arg()],
    )
}

pub fn cast_f32_to_bf16(ctx: &Ctx<'_>, x: Tensor, o: Tensor) -> Result<(), Error> {
    const OP: &str = "elementwise.cast_f32_to_bf16";
    if x.dtype != dtype::Dtype::F32 || o.dtype != dtype::Dtype::Bf16 {
        return Err(refuse(
            OP,
            format!(
                "this cast reads f32 into bf16, and was handed {:?} into {:?}",
                x.dtype, o.dtype
            ),
        ));
    }
    activation(ctx, OP, "cast_f32_to_bf16", x, o)
}

pub fn relative_bucket_bias(
    ctx: &Ctx<'_>,
    embedding: Tensor,
    max_len: u32,
    num_buckets: u32,
    max_distance: f32,
    bidirectional: bool,
    y: Tensor,
) -> Result<(), Error> {
    const OP: &str = "elementwise.relative_bucket_bias";
    let entry = dtype_dispatch!(OP, embedding.dtype, {
        Bf16 => "relative_bucket_bias_bfloat16",
        F32 => "relative_bucket_bias_float32",
    });
    let heads = nonzero(OP, "heads", y.rows)?;
    nonzero(OP, "max_len", max_len)?;
    let span = 2 * max_len - 1;
    if y.dtype != dtype::Dtype::F32 || y.width != span {
        return Err(refuse(
            OP,
            format!(
                "the table is {} x {} {:?}; this entry writes one f32 row of {span} \
                 (2 x {max_len} - 1) per head",
                y.rows, y.width, y.dtype
            ),
        ));
    }
    if embedding.rows < num_buckets || embedding.width < heads {
        return Err(refuse(
            OP,
            format!(
                "the bucket embedding is {} x {}, and the table reads {num_buckets} \
                 bucket(s) of {heads} head(s)",
                embedding.rows, embedding.width
            ),
        ));
    }
    let directional = if bidirectional {
        num_buckets / 2
    } else {
        num_buckets
    };
    let max_exact = directional / 2;
    if max_exact == 0 {
        return Err(refuse(
            OP,
            format!("{num_buckets} bucket(s) leave no exact band; the function needs at least 4"),
        ));
    }
    let ratio = f64::from(max_distance) / f64::from(max_exact);
    if ratio.is_nan() || ratio <= 1.0 || !ratio.is_finite() {
        return Err(refuse(
            OP,
            format!(
                "max_distance {max_distance} is at or below max_exact {max_exact}; the \
                 large band's logarithm has no answer"
            ),
        ));
    }
    #[allow(clippy::cast_possible_truncation)]
    let log_ratio = ratio.ln() as f32;
    ctx.fire(
        Fire::at(FILE, entry).apply(Grid::of([span, heads, 1], [256.min(span), 1, 1])),
        &[
            embedding.arg(),
            y.arg_mut(),
            heads.arg(),
            span.arg(),
            max_len.arg(),
            embedding.width.arg(),
            num_buckets.arg(),
            u32::from(bidirectional).arg(),
            log_ratio.arg(),
        ],
    )
}
