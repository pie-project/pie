use super::norm::Next;
use crate::error::Error;
use crate::jit::{
    Arg, ArgValue, Ctx, Fire, Launch, aligned16, count, dtype_dispatch, nonzero, refuse, stated,
    symbol,
};
use crate::tensor::Tensor;
use dtype::Dtype;

const MODULATE: &str = "elemwise/modulate.cu";

/// Muse Spark's post-norm and residual gate (`elemwise.norm_gate`): a
/// row's `x` times `scale` (rounded), RMS-normed unscaled (rounded), and with
/// `res` and its f32 `gate` the residual's gated sum; with `next`, the next
/// sublayer's input norm over it in the same pass.
#[allow(clippy::too_many_arguments)]
pub fn norm_gate(
    ctx: &Ctx,
    x: Tensor,
    scale: Option<Tensor>,
    res: Option<Tensor>,
    gate: Option<Tensor>,
    eps: f32,
    temperature: f32,
    fan: u32,
    next: Option<Next>,
    y: &mut Tensor,
) -> Result<(), Error> {
    const OP: &str = "elemwise.norm_gate";
    let t = dtype_dispatch!(OP, x.dtype, { Bf16 => "::pie::bf16", F16 => "::pie::f16" });
    if res.is_some() != gate.is_some() || gate.is_some_and(|g| g.dtype != Dtype::F32) {
        return Err(refuse(OP, "a residual comes with its f32 gate"));
    }
    let rows = nonzero(OP, "rows", y.rows)?;
    let absent = |t: Option<Tensor>| t.map_or(crate::jit::ArgValue::ABSENT, |t| t.arg());
    // A 16-byte run a thread, the row in the block's registers.
    let threads = y.width / 8;
    let planes = [
        Some(x),
        scale,
        res,
        gate,
        Some(*y),
        next.map(|n| n.weight),
        next.map(|n| n.y),
    ];
    if !y.width.is_multiple_of(8)
        || !threads.is_power_of_two()
        || !(32..=1024).contains(&threads)
        || !planes.into_iter().flatten().all(|t| aligned16(t.ptr))
    {
        return Err(refuse(
            OP,
            "a row of 256 to 8192 elements, a power of two, 16-byte aligned",
        ));
    }
    let tail = next.is_some();
    // A long fire's block walks sixteen rows; a short one's keeps a row.
    const WALK: u32 = 16;
    let walk = rows >= 8192;
    let (entry, launch) = if walk {
        (
            format!("::pie::elemwise::norm_gate_rows<{t}, {threads}, {tail}, {WALK}>"),
            Launch::per_row(rows.div_ceil(WALK), threads),
        )
    } else {
        (
            format!("::pie::elemwise::norm_gate<{t}, {threads}, {tail}>"),
            Launch::per_row(rows, threads),
        )
    };
    let mut args = vec![
        x.arg(),
        absent(scale),
        absent(res),
        absent(gate),
        y.arg(),
        eps.arg(),
        temperature.arg(),
        stated(OP, fan)?.arg(),
        absent(next.map(|n| n.weight)),
        next.map_or(0.0, |n| n.eps).arg(),
        i32::from(next.is_some_and(|n| n.plus_one)).arg(),
        absent(next.map(|n| n.y)),
    ];
    if walk {
        args.push(stated(OP, rows)?.arg());
    }
    args.push(ctx.stage());
    ctx.fire(OP, Fire::at(MODULATE, symbol(&entry)).apply(launch), &args)
}

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

    fn weight(self) -> ArgValue<'static> {
        match self {
            NormKind::RmsNorm { weight, .. } => weight.arg(),
            _ => ArgValue::ABSENT,
        }
    }
}

fn lane_map(
    op: &'static str,
    lane_of_row: Option<Tensor>,
    rows: u32,
) -> Result<ArgValue<'static>, Error> {
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
            MODULATE,
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
        "elemwise.modulate",
        Form::ScaleShift,
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
        "elemwise.modulate",
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
    const OP: &str = "elemwise.gated_residual_add";
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
            MODULATE,
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
    const OP: &str = "fused.norm_modulate";
    let (t, tm, width, rows) = fused_shapes(OP, x, m, lane_of_row, norm, o)?;
    ctx.fire(
        OP,
        Fire::at(
            MODULATE,
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
    const OP: &str = "fused.gated_residual_norm_modulate";
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
            MODULATE,
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

pub fn scale(
    ctx: &Ctx,
    x: Tensor,
    m: Tensor,
    lane_of_row: Option<Tensor>,
    o: &mut Tensor,
) -> Result<(), Error> {
    fire(ctx, "elemwise.modulate", Form::Scale, x, m, lane_of_row, o)
}

pub fn res_blend(
    ctx: &Ctx,
    prefix: Tensor,
    blocks: &[Tensor],
    weight: Tensor,
    eps: f32,
    proj: Tensor,
    y: &mut Tensor,
) -> Result<(), Error> {
    const OP: &str = "elemwise.res_blend";

    const MAX_BLOCKS: usize = 32;

    let t = dtype_dispatch!(OP, y.dtype, { Bf16 => "::pie::bf16", F16 => "::pie::f16" });
    let rows = count(OP, "rows", y.rows)?;
    let hidden = count(OP, "the blended row's width", y.width)?;
    if blocks.len() > MAX_BLOCKS {
        return Err(refuse(
            OP,
            format!(
                "{} candidate blocks exceed the kernel's softmax scratch bound of {MAX_BLOCKS}",
                blocks.len()
            ),
        ));
    }
    // Each closed block is read where it lies (they are residual snapshots
    // taken layers apart), its address passed by value.
    #[repr(C)]
    struct ResBlocks {
        at: [u64; MAX_BLOCKS],
    }
    let mut at = ResBlocks {
        at: [0; MAX_BLOCKS],
    };
    for (slot, block) in at.at.iter_mut().zip(blocks) {
        *slot = block.ptr;
    }
    ctx.fire(
        OP,
        Fire::at(
            MODULATE,
            symbol(&format!("::pie::elemwise::res_blend<{t}>")),
        )
        .apply(Launch::per_row(y.rows, BLOCK)),
        &[
            prefix.arg(),
            crate::jit::ArgValue::of(&at),
            weight.arg(),
            proj.arg(),
            y.arg(),
            stated(OP, u32::try_from(blocks.len()).unwrap_or(u32::MAX))?.arg(),
            hidden.arg(),
            rows.arg(),
            eps.arg(),
            ctx.stage(),
        ],
    )
}

/// [`res_blend`] and the RMS norm of its row under `out_weight` in one pass
/// (`res_blend_norm`, 512 threads a row or 256 for a prefill's many, the
/// blended row held in registers): `y` the normed row. `Ok(false)` where the
/// row is not whole 16-byte runs, wider than the registers hold, or a plane
/// is unaligned.
#[allow(clippy::too_many_arguments)]
pub fn res_blend_norm(
    ctx: &Ctx,
    prefix: Tensor,
    blocks: &[Tensor],
    weight: Tensor,
    eps: f32,
    proj: Tensor,
    out_weight: Tensor,
    out_eps: f32,
    y: &mut Tensor,
) -> Result<bool, Error> {
    const OP: &str = "fused.res_blend_rmsnorm";
    const MAX_BLOCKS: usize = 32;
    let t = dtype_dispatch!(OP, y.dtype, { Bf16 => "::pie::bf16", F16 => "::pie::f16" });
    let hidden = y.width;
    // A prefill's many rows keep more of themselves resident at 256 threads a row.
    let threads: u32 = if y.rows >= 1024 && hidden <= 256 * 8 * 4 {
        256
    } else {
        512
    };
    let aligned = |p: u64| p.is_multiple_of(16);
    let row_bytes = u64::from(hidden) * 2;
    if blocks.len() > MAX_BLOCKS
        || hidden == 0
        || !hidden.is_multiple_of(8)
        || hidden > threads * 8 * 4
        || !row_bytes.is_multiple_of(16)
        || ![prefix.ptr, weight.ptr, proj.ptr, out_weight.ptr, y.ptr]
            .into_iter()
            .chain(blocks.iter().map(|b| b.ptr))
            .all(aligned)
    {
        return Ok(false);
    }
    #[repr(C)]
    struct ResBlocks {
        at: [u64; MAX_BLOCKS],
    }
    let mut at = ResBlocks {
        at: [0; MAX_BLOCKS],
    };
    for (slot, block) in at.at.iter_mut().zip(blocks) {
        *slot = block.ptr;
    }
    ctx.fire(
        OP,
        Fire::at(
            MODULATE,
            symbol(&format!("::pie::elemwise::res_blend_norm<{t}, {threads}>")),
        )
        .apply(Launch::per_row(y.rows, threads)),
        &[
            prefix.arg(),
            crate::jit::ArgValue::of(&at),
            weight.arg(),
            proj.arg(),
            out_weight.arg(),
            y.arg(),
            stated(OP, u32::try_from(blocks.len()).unwrap_or(u32::MAX))?.arg(),
            stated(OP, hidden)?.arg(),
            eps.arg(),
            out_eps.arg(),
            ctx.stage(),
        ],
    )?;
    Ok(true)
}
