//! GGUF K-quant matmuls (`linear.matmul`, `linear.lm_head` over a stored
//! Q2_K to Q6_K plane) from `kernels/linear/kquant.slang`: a 32-lane strip
//! lands two columns of a row, walking the row's super-blocks.

use crate::encode::{Arg, Ctx, Fire, dtype_dispatch, instantiated, refuse};
use crate::error::Error;
use crate::tensor::Tensor;

const KQUANT: &str = "linear/kquant.slang";

/// The elements a super-block holds.
const SUPER: u32 = 256;

/// How an engine runs the K-quant matmuls on its device.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Tier {
    /// A strip folds its sums inside a subgroup (SPIR-V, subgroups at least
    /// 32 wide).
    pub subgroup: bool,
    /// Strips a workgroup: 2 or 4.
    pub slices: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Scheme {
    Q2K,
    Q3K,
    Q4K,
    Q5K,
    Q6K,
}

impl Scheme {
    const ALL: [Self; 5] = [Self::Q2K, Self::Q3K, Self::Q4K, Self::Q5K, Self::Q6K];

    const fn tag(self) -> &'static str {
        match self {
            Self::Q2K => "q2k",
            Self::Q3K => "q3k",
            Self::Q4K => "q4k",
            Self::Q5K => "q5k",
            Self::Q6K => "q6k",
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

    /// The bytes a super-block takes.
    #[must_use]
    pub const fn block_bytes(self) -> u32 {
        match self {
            Self::Q2K => 84,
            Self::Q3K => 110,
            Self::Q4K => 144,
            Self::Q5K => 176,
            Self::Q6K => 210,
        }
    }
}

/// The scheme whose rows are `row_bytes` wide over a `k`-wide contraction.
pub fn scheme(op: &'static str, k: u32, row_bytes: u32) -> Result<Scheme, Error> {
    let blocks = k / SUPER;
    if let Some(scheme) = Scheme::ALL
        .into_iter()
        .find(|s| row_bytes == blocks * s.block_bytes())
    {
        return Ok(scheme);
    }
    let ladder: Vec<String> = Scheme::ALL
        .iter()
        .map(|s| format!("{} ({})", blocks * s.block_bytes(), s.name()))
        .collect();
    Err(refuse(
        op,
        format!(
            "a {row_bytes}-byte weight row is none of the five K-quant widths over a \
             {k}-wide contraction ({blocks} super-blocks): {}",
            ladder.join(", ")
        ),
    ))
}

pub fn matmul(ctx: &Ctx<'_>, tier: Tier, act: Tensor, w: Tensor, y: Tensor) -> Result<(), Error> {
    fire(ctx, "linear.matmul", tier, act, w, y)
}

pub fn lm_head(ctx: &Ctx<'_>, tier: Tier, act: Tensor, w: Tensor, y: Tensor) -> Result<(), Error> {
    fire(ctx, "linear.lm_head", tier, act, w, y)
}

fn fire(
    ctx: &Ctx<'_>,
    op: &'static str,
    tier: Tier,
    act: Tensor,
    w: Tensor,
    y: Tensor,
) -> Result<(), Error> {
    dtype_dispatch!(op, act.dtype, { Bf16 => () });
    debug_assert_eq!(
        act.rows, y.rows,
        "the activation's rows are the rows the result lands"
    );
    let (m, n, k) = (y.rows, y.width, act.width);
    if n == 0 {
        return Err(refuse(op, "the columns this projection lands are zero"));
    }
    if k == 0 {
        return Err(refuse(op, "the contraction this projection walks is zero"));
    }
    if !k.is_multiple_of(SUPER) {
        return Err(refuse(
            op,
            format!("K is {k}, not a whole number of {SUPER}-element K-quant super-blocks"),
        ));
    }
    if !n.is_multiple_of(2) {
        return Err(refuse(
            op,
            format!("the projection lands {n} columns: a strip lands a bf16 word, two"),
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
    if !matches!(tier.slices, 2 | 4) {
        return Err(refuse(
            op,
            format!("{} strips a workgroup is not 2 or 4", tier.slices),
        ));
    }
    let scheme = scheme(op, k, w.width)?;
    if m == 0 {
        return Ok(());
    }
    let entry = instantiated(
        op,
        format!(
            "linear_kquant{}{}_{}_bf16",
            if tier.subgroup { "_sg" } else { "" },
            if tier.slices == 4 { "_s4" } else { "" },
            scheme.tag()
        ),
    )?;
    ctx.fire(
        Fire::at(KQUANT, entry)
            .groups([n.div_ceil(2 * tier.slices), m, 1])
            .group([32, tier.slices, 1]),
        &[w.arg(), act.arg(), y.arg_mut(), n.arg(), k.arg()],
    )
}
