use crate::encode::{Arg, Ctx, Fire, dtype_dispatch, refuse, stated};
use crate::error::Error;
use crate::tensor::Tensor;

const FILE: &str = "quant/kquant.wgsl";

const SUPER: u32 = 256;

const Q2K_BYTES: u32 = 84;

const Q3K_BYTES: u32 = 110;

const Q4K_BYTES: u32 = 144;

const Q5K_BYTES: u32 = 176;

const Q6K_BYTES: u32 = 210;

const ROWS_PER_GROUP: u32 = 4;
const GROUP: [u32; 3] = [32, 2, 1];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Scheme {
    Q2K,
    Q3K,
    Q4K,
    Q5K,
    Q6K,
}

impl Scheme {
    const fn point(self) -> &'static str {
        match self {
            Self::Q2K => "kquant_q2k_bf16",
            Self::Q3K => "kquant_q3k_bf16",
            Self::Q4K => "kquant_q4k_bf16",
            Self::Q5K => "kquant_q5k_bf16",
            Self::Q6K => "kquant_q6k_bf16",
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

    #[must_use]
    pub const fn block_bytes(self) -> u32 {
        match self {
            Self::Q2K => Q2K_BYTES,
            Self::Q3K => Q3K_BYTES,
            Self::Q4K => Q4K_BYTES,
            Self::Q5K => Q5K_BYTES,
            Self::Q6K => Q6K_BYTES,
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

pub fn scheme(op: &'static str, k: u32, row_bytes: u32) -> Result<Scheme, Error> {
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

pub fn matmul(ctx: &Ctx<'_>, act: Tensor, w: Tensor, y: Tensor) -> Result<(), Error> {
    fire(ctx, "linear.matmul", act, w, y)
}

pub fn lm_head(ctx: &Ctx<'_>, act: Tensor, w: Tensor, y: Tensor) -> Result<(), Error> {
    fire(ctx, "linear.lm_head", act, w, y)
}

fn fire(ctx: &Ctx<'_>, op: &'static str, act: Tensor, w: Tensor, y: Tensor) -> Result<(), Error> {
    dtype_dispatch!(op, act.dtype, { Bf16 => () });
    debug_assert_eq!(
        act.rows, y.rows,
        "the activation's rows are the rows the result lands"
    );
    if y.width == 0 {
        return Err(refuse(op, "the columns this projection lands are zero"));
    }
    if act.width == 0 {
        return Err(refuse(op, "the contraction this projection walks is zero"));
    }
    let (m, n, k) = (y.rows, y.width, act.width);
    if !k.is_multiple_of(SUPER) {
        return Err(refuse(
            op,
            format!("K is {k}, not a whole number of {SUPER}-element K-quant super-blocks"),
        ));
    }
    if w.rows != n {
        return Err(refuse(
            op,
            format!(
                "the weight has {} rows and this projection lands {n} columns; a stored \
                 K-quant plane is one row per column",
                w.rows
            ),
        ));
    }
    if !n.is_multiple_of(2) {
        return Err(refuse(
            op,
            format!(
                "this projection lands {n} columns; the point writes bf16 pairs and \
                 an odd column count would tear the last word"
            ),
        ));
    }
    let scheme = scheme(op, k, w.width)?;
    if m == 0 {
        return Ok(());
    }
    ctx.fire(
        Fire::at(FILE, scheme.point())
            .groups([n.div_ceil(ROWS_PER_GROUP), m, 1])
            .group(GROUP),
        &[
            w.arg(),
            act.arg(),
            y.arg_mut(),
            stated(op, n)?.arg(),
            stated(op, k)?.arg(),
            stated(op, m)?.arg(),
        ],
    )
}
