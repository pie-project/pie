use crate::error::Error;
use dtype::Dtype;

use crate::jit::{Arg, ArgValue, Ctx, Fire, Launch, dtype_dispatch, nonzero, refuse, stated, symbol};
use crate::tensor::Tensor;

const FILE: &str = "linear/lora.cuh";

const BLOCK: u32 = 256;

const WAIST: &str = "linear.lora.waist";

#[derive(Debug, Clone, Copy)]
pub struct Segments {
    pub list: Tensor,
    pub count: u32,
    pub cap: u32,
    pub max_rows: u32,
}

pub fn correct(
    ctx: &Ctx,
    x: Tensor,
    bank_a: Tensor,
    bank_b: Tensor,
    routes: Tensor,
    y: &mut Tensor,
    segments: Option<Segments>,
) -> Result<(), Error> {
    const OP: &str = "linear.lora_correct";

    let t = dtype_dispatch!(OP, x.dtype, { Bf16 => "::pie::bf16", F16 => "::pie::f16" });
    debug_assert_eq!(routes.dtype, Dtype::I32, "`{OP}` walks i32 adapter ids");
    debug_assert_eq!(bank_a.dtype, x.dtype, "the adapter bank rides the activation's dtype");
    debug_assert_eq!(bank_b.dtype, x.dtype, "the adapter bank rides the activation's dtype");

    let rows = nonzero(OP, "rows", x.rows)?;
    let in_width = nonzero(OP, "the correction's input width", x.width)?;
    let out_width = nonzero(OP, "the correction's output width", y.width)?;
    if !bank_a.width.is_multiple_of(in_width) {
        return Err(refuse(
            OP,
            format!(
                "the down bank is {} wide over an input of {in_width}, which is not a \
                 whole number of ranks",
                bank_a.width
            ),
        ));
    }
    let rank = nonzero(OP, "the adapter bank's rank", bank_a.width / in_width)?;
    if bank_b.width != out_width.saturating_mul(rank) {
        return Err(refuse(
            OP,
            format!(
                "the up bank is {} wide where {out_width} x {rank} is {}; the two \
                 planes of one bank state two ranks",
                bank_b.width,
                out_width.saturating_mul(rank),
            ),
        ));
    }
    if bank_a.rows != bank_b.rows {
        return Err(refuse(
            OP,
            format!(
                "the bank's two planes seat {} and {} adapters",
                bank_a.rows, bank_b.rows
            ),
        ));
    }
    debug_assert_eq!(y.rows, x.rows, "a correction lands one row per input row");
    debug_assert_eq!(routes.rows, x.rows, "one adapter id per token row");

    let bytes = (rows as usize)
        .saturating_mul(rank as usize)
        .saturating_mul(2);
    let waist = ctx.scratch(OP, WAIST, bytes)?;
    let mut projected = Tensor::new(waist as u64, rows, rank, x.dtype);

    super::moe::select_gemv(
        ctx,
        OP,
        x,
        bank_a,
        routes,
        &mut projected,
        super::moe::ExpertTable::RESIDENT,
    )?;

    let rank_i = stated(OP, rank)?;
    let out_i = stated(OP, out_width)?;
    let stride = i64::from(out_i) * i64::from(rank_i);
    let (grid, list, segs) = match segments {
        None => ([out_width.div_ceil(BLOCK), rows, 1], ArgValue::ABSENT, 0i32),
        Some(segments) => (
            [
                out_width.div_ceil(BLOCK),
                segments.max_rows.max(1),
                segments.cap.max(segments.count).max(1),
            ],
            segments.list.arg(),
            stated(OP, segments.count)?,
        ),
    };
    ctx.fire(
        OP,
        Fire::at(FILE, symbol(&format!("::pie::linear::lora_combine<{t}>")))
            .apply(Launch::grid(grid, [BLOCK, 1, 1])),
        &[
            routes.arg(),
            projected.arg(),
            bank_b.arg(),
            y.arg(),
            list,
            segs.arg(),
            rank_i.arg(),
            out_i.arg(),
            stride.arg(),
            ctx.stage(),
        ],
    )
}
