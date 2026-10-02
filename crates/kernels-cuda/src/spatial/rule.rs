use crate::error::Error;
use crate::jit::{ArgValue, Ctx, Fire, Launch, stated};
use crate::spatial::lane_pair;
use crate::tensor::Tensor;

const FILE: &str = "spatial/rule.cuh";

const OP: &str = "spatial.grid";

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
        Fire::at(FILE, "::pie::spatial::grid_rule").apply(Launch::grid([1, 1, 1], [32, 1, 1])),
        &[
            grid.arg(),
            o_grid.arg(),
            ArgValue::Bytes {
                ptr: std::ptr::from_ref(&geom).cast(),
                len: size_of::<Geom>(),
            },
        ],
    )
}
