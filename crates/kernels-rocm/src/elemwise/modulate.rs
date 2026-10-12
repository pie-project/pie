use crate::error::Error;
use dtype::Dtype;

use crate::jit::{
    Arg, ArgValue, Ctx, Fire, Launch, dtype_dispatch, nonzero, refuse, stated, symbol,
};
use crate::tensor::Tensor;

const FILE: &str = "elemwise/modulate.cuh";

const BLOCK: u32 = 256;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Form {
    ScaleShift,
    Scale,
    TanhGate,
}

impl Form {
    const fn stamp(self) -> &'static str {
        match self {
            Form::ScaleShift => "0",
            Form::Scale => "1",
            Form::TanhGate => "2",
        }
    }

    const fn vectors(self) -> u32 {
        match self {
            Form::ScaleShift => 2,
            Form::Scale | Form::TanhGate => 1,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum NormKind {
    LayerNormNoAffine { eps: f32 },
    RmsNormNoScale { eps: f32 },
    RmsNorm { weight: Tensor, eps: f32 },
}

impl NormKind {
    const fn stamp(self) -> &'static str {
        match self {
            NormKind::LayerNormNoAffine { .. } => "0",
            NormKind::RmsNormNoScale { .. } => "1",
            NormKind::RmsNorm { .. } => "2",
        }
    }

    const fn eps(self) -> f32 {
        match self {
            NormKind::LayerNormNoAffine { eps }
            | NormKind::RmsNormNoScale { eps }
            | NormKind::RmsNorm { eps, .. } => eps,
        }
    }

    fn weight(self) -> ArgValue {
        match self {
            NormKind::RmsNorm { weight, .. } => weight.arg(),
            _ => ArgValue::ABSENT,
        }
    }
}

fn lane_map(op: &'static str, lane_of_row: Option<Tensor>, rows: u32) -> Result<ArgValue, Error> {
    let Some(map) = lane_of_row else {
        return Ok(ArgValue::ABSENT);
    };
    if map.dtype != Dtype::I32 {
        return Err(refuse(
            op,
            format!(
                "the lane map is {:?}, and this modulation reads an i32 lane per row",
                map.dtype
            ),
        ));
    }
    if map.elements() != u64::from(rows) {
        return Err(refuse(
            op,
            format!(
                "the lane map is {} x {}, and this modulation reads one lane for each of \
                 {rows} rows",
                map.rows, map.width
            ),
        ));
    }
    Ok(map.arg())
}

fn modulation(
    op: &'static str,
    m: Tensor,
    width: u32,
    vectors: u32,
    dtype: Dtype,
) -> Result<&'static str, Error> {
    let tm = if m.dtype == dtype {
        dtype_dispatch!(op, dtype, { Bf16 => "::pie::bf16", F16 => "::pie::f16" })
    } else if m.dtype == Dtype::F32 {
        "float"
    } else {
        return Err(refuse(
            op,
            format!(
                "the modulation plane is {:?} and the rows it modulates are {dtype:?}; \
                 a vector rides the activation's element or stays f32",
                m.dtype
            ),
        ));
    };
    if m.width != vectors * width {
        return Err(refuse(
            op,
            format!(
                "the modulation plane is {} wide, and this form reads {vectors} x {width}",
                m.width
            ),
        ));
    }
    Ok(tm)
}

fn fire(
    ctx: &Ctx,
    op: &'static str,
    form: Form,
    x: Tensor,
    m: Tensor,
    lane_of_row: Option<Tensor>,
    o: &mut Tensor,
) -> Result<(), Error> {
    let t = dtype_dispatch!(op, x.dtype, { Bf16 => "::pie::bf16", F16 => "::pie::f16" });
    debug_assert!(
        x.rows == o.rows && x.width == o.width,
        "`{op}` writes the rectangle it reads"
    );
    let width = nonzero(op, "the modulated width", o.width)?;
    let rows = nonzero(op, "rows", o.rows)?;
    let tm = modulation(op, m, width, form.vectors(), x.dtype)?;
    if lane_of_row.is_none() && m.rows < rows {
        return Err(refuse(
            op,
            format!("{} per-token modulation vectors for {rows} rows", m.rows),
        ));
    }
    ctx.fire(
        op,
        Fire::at(
            FILE,
            symbol(&format!(
                "::pie::elemwise::modulate<{t}, {tm}, {}>",
                form.stamp()
            )),
        )
        .apply(Launch::per_row(rows, BLOCK)),
        &[
            x.arg(),
            m.arg(),
            lane_map(op, lane_of_row, rows)?,
            o.arg(),
            stated(op, width)?.arg(),
            stated(op, form.vectors() * width)?.arg(),
            ctx.stage(),
        ],
    )
}

pub fn scale_shift(
    ctx: &Ctx,
    x: Tensor,
    m: Tensor,
    lane_of_row: Option<Tensor>,
    o: &mut Tensor,
) -> Result<(), Error> {
    fire(
        ctx,
        "elementwise.modulate",
        Form::ScaleShift,
        x,
        m,
        lane_of_row,
        o,
    )
}

pub fn scale(
    ctx: &Ctx,
    x: Tensor,
    m: Tensor,
    lane_of_row: Option<Tensor>,
    o: &mut Tensor,
) -> Result<(), Error> {
    fire(
        ctx,
        "elementwise.modulate",
        Form::Scale,
        x,
        m,
        lane_of_row,
        o,
    )
}

pub fn tanh_gate(
    ctx: &Ctx,
    x: Tensor,
    m: Tensor,
    lane_of_row: Option<Tensor>,
    o: &mut Tensor,
) -> Result<(), Error> {
    fire(
        ctx,
        "elementwise.modulate",
        Form::TanhGate,
        x,
        m,
        lane_of_row,
        o,
    )
}

pub fn gated_residual_add(
    ctx: &Ctx,
    r: Tensor,
    g: Tensor,
    y: Tensor,
    lane_of_row: Option<Tensor>,
    r_out: &mut Tensor,
) -> Result<(), Error> {
    const OP: &str = "elementwise.gated_residual_add";
    let t = dtype_dispatch!(OP, r.dtype, { Bf16 => "::pie::bf16", F16 => "::pie::f16" });
    debug_assert!(
        r.rows == y.rows && r.width == y.width && r.rows == r_out.rows && r.width == r_out.width,
        "`{OP}` folds three rectangles of one shape"
    );
    let width = nonzero(OP, "the folded width", r_out.width)?;
    let rows = nonzero(OP, "rows", r_out.rows)?;
    let tm = modulation(OP, g, width, 1, r.dtype)?;
    if lane_of_row.is_none() && g.rows < rows {
        return Err(refuse(
            OP,
            format!("{} per-token gate vectors for {rows} rows", g.rows),
        ));
    }
    ctx.fire(
        OP,
        Fire::at(
            FILE,
            symbol(&format!("::pie::elemwise::gated_residual_add<{t}, {tm}>")),
        )
        .apply(Launch::per_row(rows, BLOCK)),
        &[
            r.arg(),
            g.arg(),
            y.arg(),
            lane_map(OP, lane_of_row, rows)?,
            r_out.arg(),
            stated(OP, width)?.arg(),
            ctx.stage(),
        ],
    )
}

pub fn norm_modulate(
    ctx: &Ctx,
    x: Tensor,
    m: Tensor,
    lane_of_row: Option<Tensor>,
    norm: NormKind,
    o: &mut Tensor,
) -> Result<(), Error> {
    const OP: &str = "elementwise.norm_modulate";
    let (t, tm, width, rows) = fused_shapes(OP, x, m, lane_of_row, norm, o)?;
    ctx.fire(
        OP,
        Fire::at(
            FILE,
            symbol(&format!(
                "::pie::elemwise::norm_modulate<{t}, {tm}, {BLOCK}, {}>",
                norm.stamp()
            )),
        )
        .apply(Launch::per_row(rows, BLOCK)),
        &[
            x.arg(),
            norm.weight(),
            m.arg(),
            lane_map(OP, lane_of_row, rows)?,
            o.arg(),
            stated(OP, width)?.arg(),
            stated(OP, 2 * width)?.arg(),
            norm.eps().arg(),
            ctx.stage(),
        ],
    )
}

#[allow(clippy::too_many_arguments)]
pub fn gated_residual_norm_modulate(
    ctx: &Ctx,
    r: Tensor,
    g: Tensor,
    y: Tensor,
    m: Tensor,
    lane_of_row: Option<Tensor>,
    norm: NormKind,
    r_out: &mut Tensor,
    o: &mut Tensor,
) -> Result<(), Error> {
    const OP: &str = "elementwise.gated_residual_norm_modulate";
    let (t, tm, width, rows) = fused_shapes(OP, r, m, lane_of_row, norm, o)?;
    debug_assert!(
        r.rows == y.rows && r.width == y.width && r.rows == r_out.rows && r.width == r_out.width,
        "`{OP}` folds three rectangles of one shape"
    );
    let tg = modulation(OP, g, width, 1, r.dtype)?;
    if tg != tm {
        return Err(refuse(
            OP,
            format!(
                "the gate plane is {:?} and the modulation plane {:?}; one fused pass reads \
                 both in one element",
                g.dtype, m.dtype
            ),
        ));
    }
    if lane_of_row.is_none() && g.rows < rows {
        return Err(refuse(
            OP,
            format!("{} per-token gate vectors for {rows} rows", g.rows),
        ));
    }
    ctx.fire(
        OP,
        Fire::at(
            FILE,
            symbol(&format!(
                "::pie::elemwise::gated_residual_norm_modulate<{t}, {tm}, {BLOCK}, {}>",
                norm.stamp()
            )),
        )
        .apply(Launch::per_row(rows, BLOCK)),
        &[
            r.arg(),
            g.arg(),
            y.arg(),
            r_out.arg(),
            norm.weight(),
            m.arg(),
            lane_map(OP, lane_of_row, rows)?,
            o.arg(),
            stated(OP, width)?.arg(),
            stated(OP, 2 * width)?.arg(),
            norm.eps().arg(),
            ctx.stage(),
        ],
    )
}

fn fused_shapes(
    op: &'static str,
    x: Tensor,
    m: Tensor,
    lane_of_row: Option<Tensor>,
    norm: NormKind,
    o: &mut Tensor,
) -> Result<(&'static str, &'static str, u32, u32), Error> {
    let t = dtype_dispatch!(op, x.dtype, { Bf16 => "::pie::bf16", F16 => "::pie::f16" });
    debug_assert!(
        x.rows == o.rows && x.width == o.width,
        "`{op}` writes the rectangle it reads"
    );
    let width = nonzero(op, "the normed width", o.width)?;
    let rows = nonzero(op, "rows", o.rows)?;
    let tm = modulation(op, m, width, 2, x.dtype)?;
    if lane_of_row.is_none() && m.rows < rows {
        return Err(refuse(
            op,
            format!("{} per-token modulation vectors for {rows} rows", m.rows),
        ));
    }
    if let NormKind::RmsNorm { weight, .. } = norm
        && weight.elements() != u64::from(width)
    {
        return Err(refuse(
            op,
            format!(
                "the norm's weight is a {} x {} plane, and this row reads one scalar per \
                 column of a {width}-wide rectangle",
                weight.rows, weight.width
            ),
        ));
    }
    Ok((t, tm, width, rows))
}
