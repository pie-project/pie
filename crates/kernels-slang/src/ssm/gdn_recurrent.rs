//! The gated delta rule, token by token (`ssm.gated_delta`, its chunked and
//! committed forms), from `kernels/ssm/gdn_recurrent.slang`.

use dtype::Dtype;

use super::{Committed, StatePool, head_width, requests, rows, window_lanes};
use crate::encode::{Arg, Ctx, Fire, Grid, Tier, dtype_dispatch, nonzero, refuse, stated};
use crate::error::Error;
use crate::tensor::{RaggedTensor, Tensor};

const GDN_RECURRENT: &str = "ssm/gdn_recurrent.slang";

const GROUP: u32 = 128;

/// The key width the register-state kernels hold.
const REGISTER_K: u32 = 128;

/// How an engine runs the recurrence on its device.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Tactic {
    /// At `Tier::Subgroup` the norms fold through subgroups (SPIR-V); else
    /// through workgroup memory.
    pub tier: Tier,
    /// The fewest rows a chunked or committed fire holds its state in
    /// registers at (128-wide keys, at most 128-wide values); a decode step
    /// always does.
    pub register_min_rows: u32,
}

/// The four stated head numbers, checked against the planes.
struct Heads {
    k_heads: u32,
    v_heads: u32,
    k_dim: u32,
    v_dim: u32,
}

impl Heads {
    #[allow(clippy::too_many_arguments)]
    fn of(
        op: &'static str,
        qkv: Tensor,
        gates: Tensor,
        y: Tensor,
        k_heads: u32,
        v_heads: u32,
        k_dim: u32,
        v_dim: u32,
    ) -> Result<Self, Error> {
        nonzero(op, "the key heads this statement states", k_heads)?;
        nonzero(op, "the value heads this statement states", v_heads)?;
        nonzero(op, "the key head width this statement states", k_dim)?;
        nonzero(op, "the value head width this statement states", v_dim)?;
        if !v_heads.is_multiple_of(k_heads) {
            return Err(refuse(
                op,
                format!(
                    "the {v_heads} value heads are not a whole number of the {k_heads} key heads"
                ),
            ));
        }
        head_width(op, k_dim, "the key head width")?;
        debug_assert_eq!(gates.dtype, Dtype::F32, "`{op}` reads an f32 gate row");
        debug_assert_eq!(y.dtype, Dtype::F32, "`{op}` lands an f32 accumulator");
        debug_assert_eq!(
            u64::from(qkv.width),
            2 * u64::from(k_heads) * u64::from(k_dim) + u64::from(v_heads) * u64::from(v_dim),
            "the post-convolution qkv's row is the four stated head numbers"
        );
        debug_assert!(
            gates.rows == qkv.rows && gates.width == 2 * v_heads,
            "the fused `[g | beta]` row is two entries per value head"
        );
        debug_assert!(
            y.rows == qkv.rows && u64::from(y.width) == u64::from(v_heads) * u64::from(v_dim),
            "the recurrence lands one value plane per row"
        );
        Ok(Self {
            k_heads,
            v_heads,
            k_dim,
            v_dim,
        })
    }

    /// Whether the state fits a lane's registers.
    fn registers(&self) -> bool {
        self.k_dim == REGISTER_K && self.v_dim <= GROUP
    }

    fn grid(&self, rows: u32) -> Grid {
        Grid::of([GROUP, self.v_heads, rows], [GROUP, 1, 1])
    }

    fn args(&self, op: &'static str) -> Result<[crate::ArgValue; 4], Error> {
        Ok([
            stated(op, self.k_heads)?.arg(),
            stated(op, self.v_heads)?.arg(),
            stated(op, self.k_dim)?.arg(),
            stated(op, self.v_dim)?.arg(),
        ])
    }
}

/// One decode step a row, each value head's state in its slot.
#[allow(clippy::too_many_arguments)]
pub fn gated_delta(
    ctx: &Ctx<'_>,
    tactic: Tactic,
    qkv: Tensor,
    gates: Tensor,
    state: &StatePool,
    k_heads: u32,
    v_heads: u32,
    k_dim: u32,
    v_dim: u32,
    y: Tensor,
) -> Result<(), Error> {
    const OP: &str = "ssm.gated_delta";
    dtype_dispatch!(OP, qkv.dtype, { Bf16 => () });
    let heads = Heads::of(OP, qkv, gates, y, k_heads, v_heads, k_dim, v_dim)?;
    let entry = if heads.registers() {
        "ssm_gdn_recurrent_r128_bf16"
    } else {
        "ssm_gdn_recurrent_bf16"
    };
    let [kh, vh, kd, vd] = heads.args(OP)?;
    ctx.fire(
        Fire::at(GDN_RECURRENT, entry)
            .tier(tactic.tier)
            .apply(heads.grid(rows(OP, qkv)?)),
        &[
            qkv.arg(),
            gates.arg(),
            state.bank.arg_mut(),
            state.slots.arg(),
            y.arg_mut(),
            kh,
            vh,
            kd,
            vd,
        ],
    )
}

/// Each request's rows of the query CSR, in order over its slot.
#[allow(clippy::too_many_arguments)]
pub fn gated_delta_chunked(
    ctx: &Ctx<'_>,
    tactic: Tactic,
    qkv: RaggedTensor,
    gates: Tensor,
    state: &StatePool,
    k_heads: u32,
    v_heads: u32,
    k_dim: u32,
    v_dim: u32,
    y: Tensor,
) -> Result<(), Error> {
    const OP: &str = "ssm.gated_delta_chunked";
    dtype_dispatch!(OP, qkv.data.dtype, { Bf16 => () });
    let heads = Heads::of(OP, qkv.data, gates, y, k_heads, v_heads, k_dim, v_dim)?;
    let entry = if heads.registers() && qkv.data.rows >= tactic.register_min_rows {
        "ssm_gdn_recurrent_chunked_r128_bf16"
    } else {
        "ssm_gdn_recurrent_chunked_bf16"
    };
    let [kh, vh, kd, vd] = heads.args(OP)?;
    ctx.fire(
        Fire::at(GDN_RECURRENT, entry)
            .tier(tactic.tier)
            .apply(heads.grid(requests(OP, qkv)?)),
        &[
            qkv.data.arg(),
            qkv.indptr.arg(),
            gates.arg(),
            state.bank.arg_mut(),
            state.slots.arg(),
            y.arg_mut(),
            kh,
            vh,
            kd,
            vd,
        ],
    )
}

/// A speculative window's rows (the replayed ones first) on a copy of each
/// request's state in `work`, the slot kept as of its committed row.
#[allow(clippy::too_many_arguments)]
pub fn gated_delta_committed(
    ctx: &Ctx<'_>,
    tactic: Tactic,
    qkv: Tensor,
    indptr: Tensor,
    committed: &Committed,
    gates: Tensor,
    state: &StatePool,
    work: Tensor,
    k_heads: u32,
    v_heads: u32,
    k_dim: u32,
    v_dim: u32,
    y: Tensor,
) -> Result<(), Error> {
    const OP: &str = "ssm.gated_delta_committed";
    dtype_dispatch!(OP, qkv.dtype, { Bf16 => () });
    let heads = Heads::of(OP, qkv, gates, y, k_heads, v_heads, k_dim, v_dim)?;
    let entry = if heads.registers() && qkv.rows >= tactic.register_min_rows {
        "ssm_gdn_recurrent_committed_r128_bf16"
    } else {
        "ssm_gdn_recurrent_committed_bf16"
    };
    let [kh, vh, kd, vd] = heads.args(OP)?;
    ctx.fire(
        Fire::at(GDN_RECURRENT, entry)
            .tier(tactic.tier)
            .apply(heads.grid(window_lanes(OP, indptr)?)),
        &[
            qkv.arg(),
            indptr.arg(),
            committed.replay.arg(),
            committed.commit.arg(),
            committed.slots.arg(),
            gates.arg(),
            state.bank.arg_mut(),
            work.arg_mut(),
            y.arg_mut(),
            stated(OP, committed.lane0)?.arg(),
            kh,
            vh,
            kd,
            vd,
        ],
    )
}
