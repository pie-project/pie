use crate::error::Error;
use dtype::Dtype;

use crate::elemwise::hc::MAX_HC_MULT;
use crate::elemwise::hc::rmsnorm_f32;
use crate::jit::{
    Arg, ArgValue, Ctx, Fire, Launch, dtype_dispatch, kernel, nonzero, refuse, stated,
};
use crate::tensor::Tensor;

const FILE: &str = "elemwise/hc.hip";

const BLOCK: u32 = 256;

/// The rows a tile of the tiled projection takes.
const PROJECT_ROWS: u32 = 16;
/// The tiled projection's wavefronts a block; each walks 64 k a step.
const PROJECT_WAVES: u32 = 4;
/// The most K slices it sums (`hc_project_sum` holds one a register).
const PROJECT_SLICES: u32 = 64;
/// The blocks a compute unit the slicing fills: one block leaves a SIMD a
/// single wavefront, whose f32 MFMA chains and stream reads then stall it.
const PROJECT_BLOCKS_PER_CU: u32 = 2;

fn compute_units() -> u32 {
    #[cfg(feature = "rocm")]
    {
        crate::linear::compute_units()
    }
    #[cfg(not(feature = "rocm"))]
    {
        304
    }
}

/// How the tiled projection cuts its rows and K: `nt` 16-row tiles a
/// wavefront (two past one tile), `tiles` row blocks, and K in `slices`
/// slices of `per`, sliced until the blocks fill every compute unit
/// `PROJECT_BLOCKS_PER_CU` deep, a slice at least one step of every
/// wavefront.
#[derive(Clone, Copy)]
struct Slicing {
    nt: u32,
    tiles: u32,
    per: u32,
    slices: u32,
}

impl Slicing {
    fn of(rows: u32, fan: u32) -> Self {
        let nt = if rows > PROJECT_ROWS { 2 } else { 1 };
        let tiles = rows.div_ceil(PROJECT_ROWS * nt);
        let grain = 64 * PROJECT_WAVES;
        let want = (compute_units() * PROJECT_BLOCKS_PER_CU)
            .div_ceil(tiles)
            .clamp(1, fan.div_ceil(grain).min(PROJECT_SLICES));
        let per = fan.div_ceil(want).div_ceil(grain) * grain;
        Self {
            nt,
            tiles,
            per,
            slices: fan.div_ceil(per),
        }
    }

    fn launch(self) -> Launch {
        Launch::grid([self.slices, self.tiles, 1], [64 * PROJECT_WAVES, 1, 1])
    }

    /// The slices' partial sums, `[slices][rows][width]` f32, when there
    /// is more than one slice (or `always`).
    fn part(
        self,
        ctx: &Ctx,
        op: &'static str,
        rows: u32,
        width: u32,
        always: bool,
    ) -> Result<ArgValue<'static>, Error> {
        if self.slices == 1 && !always {
            return Ok(ArgValue::ABSENT);
        }
        let bytes = self.slices as usize * rows as usize * width as usize * 4;
        Ok(ArgValue::Ptr(
            ctx.scratch(op, "elemwise.hc_project_part", bytes)? as usize as u64,
        ))
    }
}

pub fn project(
    ctx: &Ctx,
    normed: Tensor,
    hc_fn: Tensor,
    stream_count: u32,
    mixes: &mut Tensor,
) -> Result<(), Error> {
    const OP: &str = "elemwise.hc_project";
    debug_assert!(
        normed.dtype == Dtype::F32 && hc_fn.dtype == Dtype::F32 && mixes.dtype == Dtype::F32,
        "`{OP}` projects an f32 row through an f32 plane into an f32 row"
    );
    debug_assert_eq!(
        mixes.rows, normed.rows,
        "the projection lands one mix row per stream row"
    );
    let fan = nonzero(OP, "the stream row this projection contracts", normed.width)?;
    if hc_fn.width != fan {
        return Err(refuse(
            OP,
            format!(
                "the dynamic plane contracts {} and the stream row is {fan} wide",
                hc_fn.width
            ),
        ));
    }
    if stream_count == 0 || stream_count > MAX_HC_MULT {
        return Err(refuse(
            OP,
            format!(
                "the stream count is {stream_count}, not one of the {MAX_HC_MULT} the \
                 mixers unroll"
            ),
        ));
    }
    let mix_hc = hc_fn.rows;
    let layer_row = 2 * stream_count + stream_count * stream_count;
    let independent_row = 2 * stream_count;
    if mixes.width != mix_hc
        || (mix_hc != layer_row && mix_hc != independent_row && mix_hc != stream_count)
    {
        return Err(refuse(
            OP,
            format!(
                "a {stream_count}-stream mix row is {layer_row} wide ({independent_row} without \
                 its combine) and a trunk collapse row is {stream_count}; the plane lands {} \
                 rows into a {}-wide row",
                hc_fn.rows, mixes.width
            ),
        ));
    }
    let rows = nonzero(OP, "rows", mixes.rows)?;
    if !fan.is_multiple_of(64) || !normed.ptr.is_multiple_of(16) || !hc_fn.ptr.is_multiple_of(16) {
        return ctx.fire(
            OP,
            Fire::at(kernel(FILE, "::pie::elemwise::hc_project<256>"))
                .apply(Launch::grid([rows * mix_hc, 1, 1], [BLOCK, 1, 1])),
            &[
                normed.arg(),
                hc_fn.arg(),
                mixes.arg(),
                stated(OP, fan)?.arg(),
                stated(OP, mix_hc)?.arg(),
                ctx.stage(),
            ],
        );
    }

    let plan = Slicing::of(rows, fan);
    let part = plan.part(ctx, OP, rows, mix_hc, false)?;
    ctx.fire(
        OP,
        Fire::at(kernel(
            FILE,
            &format!(
                "::pie::elemwise::hc_project_tiled<{mix_hc}, {}, {PROJECT_WAVES}>",
                plan.nt
            ),
        ))
        .apply(plan.launch()),
        &[
            normed.arg(),
            hc_fn.arg(),
            part,
            mixes.arg(),
            stated(OP, rows)?.arg(),
            stated(OP, fan)?.arg(),
            stated(OP, plan.per)?.arg(),
            ctx.stage(),
        ],
    )?;
    if plan.slices == 1 {
        return Ok(());
    }
    ctx.fire(
        OP,
        Fire::at(kernel(
            FILE,
            &format!("::pie::elemwise::hc_project_sum<{mix_hc}, {}>", plan.slices),
        ))
        .apply(Launch::flat(rows * mix_hc, 64)),
        &[part, mixes.arg(), stated(OP, rows)?.arg(), ctx.stage()],
    )
}

/// `project` of `rmsnorm_f32` of the streams, the f32 normed row never
/// landed: the norm has no weight, so it scales the projection's sums.
pub fn norm_project(
    ctx: &Ctx,
    streams: Tensor,
    hc_fn: Tensor,
    eps: f32,
    stream_count: u32,
    mixes: &mut Tensor,
) -> Result<(), Error> {
    const OP: &str = "fused.hc_norm_project";
    let t = dtype_dispatch!(OP, streams.dtype, { Bf16 => "::pie::bf16" });
    debug_assert!(
        hc_fn.dtype == Dtype::F32 && mixes.dtype == Dtype::F32,
        "`{OP}` projects through an f32 plane into an f32 row"
    );
    let rows = nonzero(OP, "rows", mixes.rows)?;
    let fan = nonzero(
        OP,
        "the stream row this projection contracts",
        streams.width,
    )?;
    let tiled = fan.is_multiple_of(64)
        && streams.ptr.is_multiple_of(16)
        && hc_fn.ptr.is_multiple_of(16)
        && hc_fn.width == fan
        && hc_fn.rows == mixes.width;
    if !tiled {
        // The primitives, over a normed row of scratch.
        let bytes = rows as usize * fan as usize * 4;
        let normed = ctx.scratch(OP, "elemwise.hc_norm_project_normed", bytes)? as usize as u64;
        let mut normed = Tensor::new(normed, streams.rows, fan, Dtype::F32);
        rmsnorm_f32(ctx, streams, eps, &mut normed)?;
        return project(ctx, normed, hc_fn, stream_count, mixes);
    }
    let mix_hc = hc_fn.rows;
    let layer_row = 2 * stream_count + stream_count * stream_count;
    let independent_row = 2 * stream_count;
    if stream_count == 0
        || stream_count > MAX_HC_MULT
        || (mix_hc != layer_row && mix_hc != independent_row && mix_hc != stream_count)
    {
        return Err(refuse(
            OP,
            format!(
                "a {stream_count}-stream mix row is {layer_row} wide ({independent_row} without \
                 its combine) and a trunk collapse row is {stream_count}; the plane lands \
                 {mix_hc} rows"
            ),
        ));
    }
    let plan = Slicing::of(rows, fan);
    let part = plan.part(ctx, OP, rows, mix_hc + 1, true)?;
    ctx.fire(
        OP,
        Fire::at(kernel(
            FILE,
            &format!(
                "::pie::elemwise::hc_norm_project_tiled<{t}, {mix_hc}, {}, {PROJECT_WAVES}>",
                plan.nt
            ),
        ))
        .apply(plan.launch()),
        &[
            streams.arg(),
            hc_fn.arg(),
            part,
            stated(OP, rows)?.arg(),
            stated(OP, fan)?.arg(),
            stated(OP, plan.per)?.arg(),
            ctx.stage(),
        ],
    )?;
    ctx.fire(
        OP,
        Fire::at(kernel(
            FILE,
            &format!(
                "::pie::elemwise::hc_norm_project_sum<{mix_hc}, {}>",
                plan.slices
            ),
        ))
        .apply(Launch::per_row(rows, (mix_hc + 64) / 64 * 64)),
        &[
            part,
            mixes.arg(),
            stated(OP, rows)?.arg(),
            stated(OP, fan)?.arg(),
            eps.arg(),
            ctx.stage(),
        ],
    )
}
