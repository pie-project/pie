use crate::error::Error;
use dtype::Dtype;

use crate::jit::{Arg, Ctx, Fire, Launch, dtype_dispatch, nonzero, refuse, stated, symbol};
use crate::tensor::Tensor;

const FILE: &str = "linear/kquant.cuh";

const WARP: u32 = 32;

const SUPER: u32 = 256;

const Q2K_BYTES: u32 = 84;

const Q3K_BYTES: u32 = 110;

const Q4K_BYTES: u32 = 144;

const Q5K_BYTES: u32 = 176;

const Q6K_BYTES: u32 = 210;

const ROWS_PER_WARP: u32 = 4;
const BLOCK_LANES: u32 = 128;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Scheme {
    Q2K,
    Q3K,
    Q4K,
    Q5K,
    Q6K,
}

impl Scheme {
    const fn point(self) -> &'static str {
        match self {
            Self::Q2K => "matmul_q2k",
            Self::Q3K => "matmul_q3k",
            Self::Q4K => "matmul_q4k",
            Self::Q5K => "matmul_q5k",
            Self::Q6K => "matmul_q6k",
        }
    }

    const fn name(self) -> &'static str {
        match self {
            Self::Q2K => "q2_k",
            Self::Q3K => "q3_k",
            Self::Q4K => "q4_k",
            Self::Q5K => "q5_k",
            Self::Q6K => "q6_k",
        }
    }
}

const FAMILY: [(u32, Scheme); 5] = [
    (Q2K_BYTES, Scheme::Q2K),
    (Q3K_BYTES, Scheme::Q3K),
    (Q4K_BYTES, Scheme::Q4K),
    (Q5K_BYTES, Scheme::Q5K),
    (Q6K_BYTES, Scheme::Q6K),
];

fn scheme(op: &'static str, k: u32, row_bytes: u32) -> Result<Scheme, Error> {
    let blocks = k / SUPER;
    for (width, scheme) in FAMILY {
        if row_bytes == blocks * width {
            return Ok(scheme);
        }
    }
    let mut ladder = String::new();
    for (at, (width, scheme)) in FAMILY.iter().enumerate() {
        if at > 0 {
            ladder.push_str(", ");
        }
        ladder.push_str(&format!("{} ({})", blocks * width, scheme.name()));
    }
    Err(refuse(
        op,
        format!(
            "a {row_bytes}-byte weight row is none of the five K-quant widths over a \
             {k}-wide contraction ({blocks} super-blocks): {ladder}"
        ),
    ))
}

pub fn matmul(ctx: &Ctx, act: Tensor, w: Tensor, y: &mut Tensor) -> Result<(), Error> {
    kquant(ctx, "linear.matmul", act, w, y)
}

pub fn lm_head(ctx: &Ctx, act: Tensor, w: Tensor, y: &mut Tensor) -> Result<(), Error> {
    kquant(ctx, "linear.lm_head", act, w, y)
}

fn kquant(
    ctx: &Ctx,
    op: &'static str,
    act: Tensor,
    w: Tensor,
    y: &mut Tensor,
) -> Result<(), Error> {
    let t = dtype_dispatch!(op, act.dtype, { Bf16 => "::pie::bf16", F16 => "::pie::f16" });
    debug_assert_eq!(w.dtype, Dtype::U8, "a stored K-quant plane binds as bytes");
    debug_assert_eq!(
        act.rows, y.rows,
        "the activation's rows are the rows the result lands"
    );
    let n = nonzero(op, "N, the columns this projection lands", y.width)?;
    let k = nonzero(op, "K, the contraction this projection walks", act.width)?;
    debug_assert_eq!(w.rows, n, "one weight row per column this projection lands");
    if !k.is_multiple_of(SUPER) {
        return Err(refuse(
            op,
            format!("K is {k}, not a whole number of {SUPER}-element K-quant super-blocks"),
        ));
    }
    let scheme = scheme(op, k, w.width)?;
    if y.rows == 0 {
        return Ok(());
    }
    let tile = (BLOCK_LANES / WARP) * ROWS_PER_WARP;
    ctx.fire(
        op,
        Fire::at(
            FILE,
            symbol(&format!(
                "::pie::linear::{}<{t}, ::pie::i32({ROWS_PER_WARP})>",
                scheme.point()
            )),
        )
        .apply(Launch::grid(
            [y.rows, n.div_ceil(tile), 1],
            [BLOCK_LANES, 1, 1],
        )),
        &[
            act.arg(),
            w.arg(),
            y.arg(),
            stated(op, n)?.arg(),
            stated(op, k)?.arg(),
            ctx.stage(),
        ],
    )
}
