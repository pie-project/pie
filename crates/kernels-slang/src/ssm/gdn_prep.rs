//! The gated delta rule's gates (`ssm.gdn_gates`), from
//! `kernels/ssm/gdn_prep.slang`.

use dtype::Dtype;

use super::rows;
use crate::encode::{Arg, Ctx, Fire, Grid, dtype_dispatch, refuse, stated};
use crate::error::Error;
use crate::tensor::Tensor;

const GDN_PREP: &str = "ssm/gdn_prep.slang";

/// The f32 `[g | beta]` row of each `[b | a]` projection row: `g =
/// -exp(a_log) softplus(a + dt_bias)`, `beta = sigmoid(b)`, a value head an
/// entry.
pub fn gdn_prep(
    ctx: &Ctx<'_>,
    ba: Tensor,
    dt_bias: Tensor,
    a_log: Tensor,
    gates: Tensor,
) -> Result<(), Error> {
    const OP: &str = "ssm.gdn_gates";
    let entry = dtype_dispatch!(OP, ba.dtype, { Bf16 => "ssm_gdn_prep_bf16" });
    debug_assert_eq!(a_log.dtype, Dtype::F32, "`{OP}` reads an f32 decay bank");
    debug_assert_eq!(gates.dtype, Dtype::F32, "`{OP}` lands an f32 gate row");
    if ba.width == 0 || !ba.width.is_multiple_of(2) {
        return Err(refuse(
            OP,
            format!(
                "the {}-wide `[b | a]` projection does not halve into value heads",
                ba.width
            ),
        ));
    }
    let v_heads = ba.width / 2;
    debug_assert!(
        gates.rows == ba.rows && gates.width == ba.width,
        "the fused `[g | beta]` row rides the projection it is derived from"
    );
    ctx.fire(
        Fire::at(GDN_PREP, entry).apply(Grid::of([v_heads, rows(OP, ba)?, 1], [256, 1, 1])),
        &[
            ba.arg(),
            a_log.arg(),
            dt_bias.arg(),
            gates.arg_mut(),
            stated(OP, v_heads)?.arg(),
        ],
    )
}
