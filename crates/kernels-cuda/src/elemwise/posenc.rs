use crate::error::Error;
use crate::jit::{
    Arg, ArgValue, Ctx, Fire, Launch, dtype_dispatch, kernel, nonzero, refuse, stated,
};
use crate::tensor::Tensor;
use crate::voxels::lane_pair;
use dtype::Dtype;

const POSENC: &str = "elemwise/posenc.cu";

const BLOCK: u32 = 256;

pub fn sinusoid(
    ctx: &Ctx,
    t: Tensor,
    dim: u32,
    max_period: f32,
    flip_sin_cos: bool,
    scale: f32,
    o: &mut Tensor,
) -> Result<(), Error> {
    const OP: &str = "elemwise.sinusoid";
    let rows = nonzero(OP, "rows", o.rows)?;
    nonzero(OP, "the embedding width", dim)?;
    for (what, plane) in [("timestep", t), ("embedding", *o)] {
        if plane.dtype != Dtype::F32 {
            return Err(refuse(
                OP,
                format!(
                    "the {what} plane is {:?}, and this embedding is computed and \
                     published in f32",
                    plane.dtype
                ),
            ));
        }
    }
    if t.width != 1 || t.rows < rows {
        return Err(refuse(
            OP,
            format!(
                "the timestep stream is {} x {}, and this embedding reads one scalar for \
                 each of {rows} rows",
                t.rows, t.width
            ),
        ));
    }
    if o.width != dim {
        return Err(refuse(
            OP,
            format!(
                "the destination is {} wide and the embedding is {dim}",
                o.width
            ),
        ));
    }
    if max_period <= 0.0 || !max_period.is_finite() {
        return Err(refuse(
            OP,
            format!("the period is {max_period}, and its logarithm sets the ladder"),
        ));
    }
    ctx.fire(
        OP,
        Fire::at(kernel(POSENC, "::pie::elemwise::sinusoid")).apply(Launch::per_row(rows, BLOCK)),
        &[
            t.arg(),
            o.arg(),
            stated(OP, dim)?.arg(),
            max_period.arg(),
            i32::from(flip_sin_cos).arg(),
            scale.arg(),
            ctx.stage(),
        ],
    )
}

pub fn relative_bucket_bias(
    ctx: &Ctx,
    embedding: Tensor,
    max_len: u32,
    num_buckets: u32,
    max_distance: f32,
    bidirectional: bool,
    y: &mut Tensor,
) -> Result<(), Error> {
    const OP: &str = "elemwise.relative_bucket_bias";
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
        Fire::at(kernel(
            POSENC,
            &format!("::pie::elemwise::relative_bucket_bias<{t}>"),
        ))
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

pub fn rel_bias(
    ctx: &Ctx,
    x: Tensor,
    w: Tensor,
    heads: u32,
    d_rel: u32,
    extent: u32,
    bias: &mut Tensor,
) -> Result<(), Error> {
    const OP: &str = "linear.rel_bias";
    let t = dtype_dispatch!(OP, x.dtype, { Bf16 => "::pie::bf16", F16 => "::pie::f16" });
    let rows = nonzero(OP, "rows", x.rows)?;
    nonzero(OP, "heads", heads)?;
    nonzero(OP, "the relative feature width", d_rel)?;
    nonzero(OP, "the relative extent", extent)?;
    if x.width != heads * d_rel {
        return Err(refuse(
            OP,
            format!(
                "the relative features are {} wide and the statement names {heads} x {d_rel}",
                x.width
            ),
        ));
    }
    if w.dtype != x.dtype || w.rows != d_rel || w.width != extent {
        return Err(refuse(
            OP,
            format!(
                "the profile bank is [{}, {}] {:?} and the statement names [{d_rel}, {extent}] {:?}",
                w.rows, w.width, w.dtype, x.dtype
            ),
        ));
    }
    if bias.dtype != Dtype::F32 || bias.rows != x.rows || bias.width != heads * extent {
        return Err(refuse(
            OP,
            format!(
                "the bias lands [{}, {}] {:?} and the statement names [{}, {heads} x {extent}] f32",
                bias.rows, bias.width, bias.dtype, x.rows
            ),
        ));
    }
    ctx.fire(
        OP,
        Fire::at(kernel(POSENC, &format!("::pie::elemwise::rel_bias<{t}>"))).apply(Launch::grid(
            [extent.div_ceil(BLOCK), heads, rows],
            [BLOCK, 1, 1],
        )),
        &[
            x.arg(),
            w.arg(),
            bias.arg(),
            stated(OP, rows)?.arg(),
            stated(OP, heads)?.arg(),
            stated(OP, d_rel)?.arg(),
            stated(OP, extent)?.arg(),
            ctx.stage(),
        ],
    )
}

const OP: &str = "conv.grid";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GridRule {
    Conv {
        k: [u32; 3],
        stride: [u32; 3],
        pad: [u32; 3],
        pad_back: [u32; 3],
        causal_t: bool,
    },
    Upsample {
        factor: [u32; 3],
        keep_first_frame: bool,
    },
    Shuffle {
        r: [u32; 3],
        trim_t: u32,
    },
    Unshuffle {
        r: [u32; 3],
    },
    AvgDown {
        factor: [u32; 3],
    },
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct Geom {
    kind: i32,
    a: [i32; 3],
    b: [i32; 3],
    c: [i32; 3],
    d: [i32; 3],
    flag: i32,
    lanes: i32,
}

fn triple(op: &'static str, v: [u32; 3]) -> Result<[i32; 3], Error> {
    Ok([stated(op, v[0])?, stated(op, v[1])?, stated(op, v[2])?])
}

pub fn derive_grid(
    ctx: &Ctx,
    grid: Tensor,
    rule: GridRule,
    o_grid: &mut Tensor,
) -> Result<(), Error> {
    let lanes = lane_pair(OP, grid, *o_grid)?;
    let geom = match rule {
        GridRule::Conv {
            k,
            stride,
            pad,
            pad_back,
            causal_t,
        } => Geom {
            kind: 0,
            a: triple(OP, k)?,
            b: triple(OP, stride)?,
            c: triple(OP, pad)?,
            d: triple(OP, pad_back)?,
            flag: i32::from(causal_t),
            lanes,
        },
        GridRule::Upsample {
            factor,
            keep_first_frame,
        } => Geom {
            kind: 1,
            a: triple(OP, factor)?,
            b: [0; 3],
            c: [0; 3],
            d: [0; 3],
            flag: i32::from(keep_first_frame),
            lanes,
        },
        GridRule::Shuffle { r, trim_t } => Geom {
            kind: 2,
            a: triple(OP, r)?,
            b: [0; 3],
            c: [0; 3],
            d: [0; 3],
            flag: stated(OP, trim_t)?,
            lanes,
        },
        GridRule::Unshuffle { r } => Geom {
            kind: 3,
            a: triple(OP, r)?,
            b: [0; 3],
            c: [0; 3],
            d: [0; 3],
            flag: 0,
            lanes,
        },
        GridRule::AvgDown { factor } => Geom {
            kind: 4,
            a: triple(OP, factor)?,
            b: [0; 3],
            c: [0; 3],
            d: [0; 3],
            flag: 0,
            lanes,
        },
    };
    ctx.fire(
        OP,
        Fire::at(kernel(POSENC, "::pie::elemwise::grid_rule"))
            .apply(Launch::grid([1, 1, 1], [32, 1, 1])),
        &[grid.arg(), o_grid.arg(), ArgValue::of(&geom)],
    )
}

/// Each token row's cell (`ops::conv::cells`): a row of a clip of `tgrid`
/// its patch's `(t, h, w)` there, any other row `positions[row]` in column
/// `axis`; zeros elsewhere.
pub fn cells(
    ctx: &Ctx,
    tgrid: Tensor,
    positions: Tensor,
    axis: u32,
    y: &mut Tensor,
) -> Result<(), Error> {
    const OP: &str = "conv.cells";
    let clips = crate::voxels::lanes_of(OP, "token", tgrid)?;
    if positions.dtype != Dtype::I32 || y.dtype != Dtype::F32 || y.width != axis.max(2) + 1 {
        return Err(refuse(
            OP,
            format!(
                "the positions are {:?} and the cells {} x {:?}; this reads i32 positions and \
                 writes max(3, {axis} + 1) f32 columns",
                positions.dtype, y.width, y.dtype
            ),
        ));
    }
    if positions.rows < y.rows {
        return Err(refuse(
            OP,
            format!("{} positions for {} rows", positions.rows, y.rows),
        ));
    }
    let rows = nonzero(OP, "rows", y.rows)?;
    ctx.fire(
        OP,
        Fire::at(kernel(POSENC, "::pie::elemwise::cells")).apply(Launch::flat(rows, BLOCK)),
        &[
            tgrid.arg(),
            positions.arg(),
            y.arg(),
            clips.arg(),
            stated(OP, axis)?.arg(),
            stated(OP, y.width)?.arg(),
            stated(OP, rows)?.arg(),
        ],
    )
}

/// Each row's count of `csr`'s rows in its lane's group
/// (`ops::gather::group_rows`).
pub fn group_rows(
    ctx: &Ctx,
    csr: Tensor,
    groups: Tensor,
    lanes: Tensor,
    y: &mut Tensor,
) -> Result<(), Error> {
    const OP: &str = "gather.group_rows";
    if [csr.dtype, groups.dtype, lanes.dtype] != [Dtype::I32; 3]
        || y.dtype != Dtype::F32
        || y.width != 1
    {
        return Err(refuse(
            OP,
            "the tables are i32 and the counts one f32 column",
        ));
    }
    if lanes.rows < y.rows || csr.rows == 0 {
        return Err(refuse(
            OP,
            format!(
                "{} row lanes for {} rows over a {}-entry indptr",
                lanes.rows, y.rows, csr.rows
            ),
        ));
    }
    let rows = nonzero(OP, "rows", y.rows)?;
    ctx.fire(
        OP,
        Fire::at(kernel(POSENC, "::pie::elemwise::group_rows")).apply(Launch::flat(rows, BLOCK)),
        &[
            csr.arg(),
            groups.arg(),
            lanes.arg(),
            y.arg(),
            stated(OP, groups.rows)?.arg(),
            stated(OP, csr.rows - 1)?.arg(),
            stated(OP, rows)?.arg(),
        ],
    )
}
