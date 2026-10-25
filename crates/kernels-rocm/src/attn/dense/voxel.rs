use crate::error::Error;
use crate::jit::{
    Arg, Ctx, Fire, Launch, aligned16, count, dtype_dispatch, kernel, refuse, stated,
};
use crate::tensor::Tensor;
use crate::voxels::lanes_of;

const FILE: &str = "attn/dense.hip";

const OP: &str = "attn.voxel";

/// Two wavefronts a block.
const BLOCK: u32 = 128;

const WAVE: u32 = 64;

/// The row widths the kernel is stamped at, and the query rows a wavefront
/// answers at each.
const WIDTHS: [(u32, u32); 4] = [(256, 4), (512, 2), (640, 1), (1024, 1)];

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum Segment {
    #[default]
    Lane,
    Frames(u32),
}

impl Segment {
    fn frames(self) -> u32 {
        match self {
            Segment::Lane => 0,
            Segment::Frames(n) => n,
        }
    }
}

/// `base`: the row `grid` addresses the rectangles' row 0 at, nonzero in a
/// window of a segmented fire whose grid holds fire-wide rows.
#[allow(clippy::too_many_arguments)]
pub fn attention(
    ctx: &Ctx,
    q: Tensor,
    k: Tensor,
    v: Tensor,
    grid: Tensor,
    segment: Segment,
    sm_scale: f32,
    o: &mut Tensor,
    base: u32,
) -> Result<(), Error> {
    dtype_dispatch!(OP, q.dtype, { Bf16 => () });
    for (what, t) in [("key", k), ("value", v), ("output", *o)] {
        if t.rows != q.rows || t.width != q.width || t.dtype != q.dtype {
            return Err(refuse(
                OP,
                format!(
                    "the {what} is {}x{} {:?}; the query is {}x{} {:?}, and all four rectangles share one shape",
                    t.rows, t.width, t.dtype, q.rows, q.width, q.dtype
                ),
            ));
        }
    }
    let Some(&(_, qpw)) = WIDTHS.iter().find(|(width, _)| *width == q.width) else {
        return Err(refuse(
            OP,
            format!(
                "no attention unit is stamped at row width {}; the arm holds {:?}",
                q.width,
                WIDTHS.map(|(width, _)| width)
            ),
        ));
    };
    if segment == Segment::Frames(0) {
        return Err(refuse(
            OP,
            "a block of zero frames holds no keys; state `Segment::Lane` for the whole clip",
        ));
    }
    if !(aligned16(q.ptr) && aligned16(k.ptr) && aligned16(v.ptr) && aligned16(o.ptr)) {
        return Err(refuse(OP, "the rectangles must be 16-byte aligned"));
    }
    let lanes = lanes_of(OP, "input", grid)?;
    let rows = count(OP, "rows", q.rows)?;
    let seg = stated(OP, segment.frames())?;
    let per_block = BLOCK / WAVE * qpw;
    let blocks = q.rows.div_ceil(per_block);
    let scale_log2 = sm_scale * core::f32::consts::LOG2_E;
    let entry = format!("::pie::attn::voxel::attention<{}, {qpw}>", q.width);
    ctx.fire(
        OP,
        Fire::at(kernel(FILE, &entry)).apply(Launch::grid([blocks, 1, 1], [BLOCK, 1, 1])),
        &[
            q.arg(),
            k.arg(),
            v.arg(),
            grid.arg(),
            o.arg(),
            lanes.arg(),
            rows.arg(),
            seg.arg(),
            stated(OP, base)?.arg(),
            scale_log2.arg(),
        ],
    )
}
