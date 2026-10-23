//! Kimi delta attention, token by token (`ssm.kda`, its chunked and
//! committed forms), from `kernels/ssm/kda.slang`.

use dtype::Dtype;

use super::{Committed, StatePool, head_width, requests, rows, window_lanes};
use crate::encode::{Arg, ArgValue, Ctx, Fire, Grid, Tier, dtype_dispatch, nonzero, stated};
use crate::error::Error;
use crate::tensor::{RaggedTensor, Tensor};

const KDA: &str = "ssm/kda.slang";

const GROUP: u32 = 128;

/// The per-token planes and the stated head shape and gate constants.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Gates {
    /// The forget projection, one head plane a row.
    pub f: Tensor,
    /// The beta projection, one entry a head a row.
    pub b: Tensor,
    /// The f32 per-channel decay bias.
    pub dt_bias: Tensor,
    /// The f32 per-head decay log.
    pub a_log: Tensor,

    pub heads: u32,

    pub head_dim: u32,

    pub norm_eps: f32,
    /// Nonzero: the gate is `exp(floor * sigmoid(...))`.
    pub gate_floor: f32,
}

impl Gates {
    fn check(&self, op: &'static str, mixed: Tensor, y: Tensor) -> Result<(), Error> {
        nonzero(op, "the KDA heads this statement states", self.heads)?;
        nonzero(
            op,
            "the KDA head width this statement states",
            self.head_dim,
        )?;
        head_width(op, self.head_dim, "the KDA head width")?;
        let plane = u64::from(self.heads) * u64::from(self.head_dim);
        debug_assert_eq!(
            self.dt_bias.dtype,
            Dtype::F32,
            "`{op}` reads an f32 decay bias"
        );
        debug_assert_eq!(
            self.a_log.dtype,
            Dtype::F32,
            "`{op}` reads an f32 decay bank"
        );
        debug_assert_eq!(y.dtype, Dtype::F32, "`{op}` lands an f32 accumulator");
        debug_assert_eq!(
            u64::from(mixed.width),
            3 * plane,
            "the post-convolution `[q | k | v]` row is three head planes"
        );
        debug_assert!(
            self.f.rows == mixed.rows && u64::from(self.f.width) == plane,
            "the forget projection's row is one head plane"
        );
        debug_assert!(
            self.b.rows == mixed.rows && self.b.width == self.heads,
            "the beta projection's row is one entry per head"
        );
        debug_assert!(
            y.rows == mixed.rows && u64::from(y.width) == plane,
            "the accumulator lands one head plane per row"
        );
        Ok(())
    }

    fn grid(&self, rows: u32) -> Grid {
        Grid::of([GROUP, self.heads, rows], [GROUP, 1, 1])
    }

    fn planes(&self) -> [ArgValue; 4] {
        [
            self.f.arg(),
            self.b.arg(),
            self.dt_bias.arg(),
            self.a_log.arg(),
        ]
    }

    fn scalars(&self, op: &'static str) -> Result<[ArgValue; 4], Error> {
        Ok([
            stated(op, self.heads)?.arg(),
            stated(op, self.head_dim)?.arg(),
            self.norm_eps.arg(),
            self.gate_floor.arg(),
        ])
    }
}

/// One decode step a row, each head's state in its slot.
pub fn kda(
    ctx: &Ctx<'_>,
    tier: Tier,
    mixed: Tensor,
    gates: &Gates,
    state: &StatePool,
    y: Tensor,
) -> Result<(), Error> {
    const OP: &str = "ssm.kda";
    let entry = dtype_dispatch!(OP, mixed.dtype, { Bf16 => "ssm_kda_bf16" });
    gates.check(OP, mixed, y)?;
    let [f, b, dt_bias, a_log] = gates.planes();
    let [heads, head_dim, eps, floor] = gates.scalars(OP)?;
    ctx.fire(
        Fire::at(KDA, entry)
            .tier(tier)
            .apply(gates.grid(rows(OP, mixed)?)),
        &[
            mixed.arg(),
            f,
            b,
            dt_bias,
            a_log,
            state.bank.arg_mut(),
            state.slots.arg(),
            y.arg_mut(),
            heads,
            head_dim,
            eps,
            floor,
        ],
    )
}

/// Each request's rows of the query CSR, in order over its slot.
pub fn kda_chunked(
    ctx: &Ctx<'_>,
    tier: Tier,
    mixed: RaggedTensor,
    gates: &Gates,
    state: &StatePool,
    y: Tensor,
) -> Result<(), Error> {
    const OP: &str = "ssm.kda_chunked";
    let entry = dtype_dispatch!(OP, mixed.data.dtype, { Bf16 => "ssm_kda_chunked_bf16" });
    gates.check(OP, mixed.data, y)?;
    let [f, b, dt_bias, a_log] = gates.planes();
    let [heads, head_dim, eps, floor] = gates.scalars(OP)?;
    ctx.fire(
        Fire::at(KDA, entry)
            .tier(tier)
            .apply(gates.grid(requests(OP, mixed)?)),
        &[
            mixed.data.arg(),
            mixed.indptr.arg(),
            f,
            b,
            dt_bias,
            a_log,
            state.bank.arg_mut(),
            state.slots.arg(),
            y.arg_mut(),
            heads,
            head_dim,
            eps,
            floor,
        ],
    )
}

/// A speculative window's rows (the replayed ones first) on a copy of each
/// request's state in `work`, the slot kept as of its committed row.
#[allow(clippy::too_many_arguments)]
pub fn kda_committed(
    ctx: &Ctx<'_>,
    tier: Tier,
    mixed: Tensor,
    indptr: Tensor,
    committed: &Committed,
    gates: &Gates,
    state: &StatePool,
    work: Tensor,
    y: Tensor,
) -> Result<(), Error> {
    const OP: &str = "ssm.kda_committed";
    let entry = dtype_dispatch!(OP, mixed.dtype, { Bf16 => "ssm_kda_committed_bf16" });
    gates.check(OP, mixed, y)?;
    let [f, b, dt_bias, a_log] = gates.planes();
    let [heads, head_dim, eps, floor] = gates.scalars(OP)?;
    ctx.fire(
        Fire::at(KDA, entry)
            .tier(tier)
            .apply(gates.grid(window_lanes(OP, indptr)?)),
        &[
            mixed.arg(),
            indptr.arg(),
            committed.replay.arg(),
            committed.commit.arg(),
            committed.slots.arg(),
            f,
            b,
            dt_bias,
            a_log,
            state.bank.arg_mut(),
            work.arg_mut(),
            y.arg_mut(),
            stated(OP, committed.lane0)?.arg(),
            heads,
            head_dim,
            eps,
            floor,
        ],
    )
}
