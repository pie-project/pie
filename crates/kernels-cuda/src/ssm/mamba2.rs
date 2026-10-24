use super::*;
use crate::jit::kernel;

const MAMBA2: &str = "ssm/mamba2.cu";

/// Mamba-2's per-head weights: `A_log`, `dt_bias` and `D` (f32 `[heads]`),
/// and the gated group norm's bf16 `[heads * 64]` weight and epsilon.
#[derive(Clone, Copy)]
pub struct Mamba2 {
    pub a_log: Tensor,
    pub dt_bias: Tensor,
    pub d: Tensor,
    pub norm: Tensor,
    pub eps: f32,
    pub heads: u32,
    pub groups: u32,
}

impl Mamba2 {
    /// Refuses all but 64-wide heads over a 128-wide state, the width of
    /// `xbc` ( `x`, then each group's `B`, then its `C`) and `dt` against
    /// the heads, and a group count that does not split them.
    fn check(
        &self,
        op: &'static str,
        xbc: Tensor,
        dt: Tensor,
        z: Tensor,
        y: &Tensor,
        state_dim: u32,
    ) -> Result<(), Error> {
        dtype_dispatch!(op, xbc.dtype, { Bf16 => () });
        let inner = self.heads * 64;
        if state_dim != 128
            || self.groups == 0
            || !self.heads.is_multiple_of(self.groups)
            || xbc.width != inner + 2 * self.groups * state_dim
            || dt.width != self.heads
            || z.width != inner
            || y.width != inner
        {
            return Err(refuse(
                op,
                format!(
                    "Mamba-2 runs 64-wide heads over a 128-wide state in whole groups; this \
                     fire states {} heads in {} groups of a {}-wide state, a {}-wide xbc, \
                     {}-wide dt, a {}-wide z and a {}-wide output",
                    self.heads, self.groups, state_dim, xbc.width, dt.width, z.width, y.width
                ),
            ));
        }
        for (what, t) in [("dt", dt.dtype), ("z", z.dtype), ("output", y.dtype)] {
            if t != Dtype::Bf16 {
                return Err(refuse(
                    op,
                    format!("Mamba-2 reads and lands bf16, and its {what} is {t:?}"),
                ));
            }
        }
        Ok(())
    }

    /// The gated output's group norm, in place.
    fn norm(&self, ctx: &Ctx, op: &'static str, y: &mut Tensor) -> Result<(), Error> {
        let group = y.width / self.groups;
        crate::elemwise::norm::rmsnorm_grouped(ctx, op, *y, self.norm, group, self.eps, false, y)
    }
}

/// One Mamba-2 step a request over the convolved `xbc` rows: lands
/// `(C S + D x) * silu(z)`, group-normed, in `y`.
#[allow(clippy::too_many_arguments)]
pub fn mamba2(
    ctx: &Ctx,
    xbc: Tensor,
    dt: Tensor,
    z: Tensor,
    m: Mamba2,
    state: &RecurrentPool,
    state_dim: u32,
    y: &mut Tensor,
) -> Result<(), Error> {
    const OP: &str = "attn.mamba2_step";
    m.check(OP, xbc, dt, z, y, state_dim)?;
    f32_state(OP, state)?;
    seated(OP, state, "mamba2_step", true, false, false)?;
    let rows = nonzero(OP, "rows", xbc.rows)?;
    ctx.fire(
        OP,
        Fire::at(kernel(MAMBA2, "::pie::ssm::mamba2_step"))
            .apply(Launch::grid([rows, m.heads, 1], [BLOCK, 1, 1])),
        &[
            xbc.arg(),
            dt.arg(),
            z.arg(),
            m.a_log.arg(),
            m.dt_bias.arg(),
            m.d.arg(),
            state.slab.arg(),
            state.slot_ids.arg(),
            state.slot_stride_elems.arg(),
            y.arg(),
            stated(OP, m.heads)?.arg(),
            stated(OP, m.groups)?.arg(),
            state.write_state.arg(),
            state.write_state_mask.arg(),
            ctx.stage(),
        ],
    )?;
    m.norm(ctx, OP, y)
}

/// A prompt's Mamba-2 walk in sixteen-token chunks: a block a (request,
/// head) lands `(C S + D x) * silu(z)` for each row, then the group norm.
#[allow(clippy::too_many_arguments)]
pub fn mamba2_chunked(
    ctx: &Ctx,
    xbc: RaggedTensor,
    dt: Tensor,
    z: Tensor,
    m: Mamba2,
    state: &RecurrentPool,
    state_dim: u32,
    y: &mut Tensor,
) -> Result<(), Error> {
    const OP: &str = "ssm.mamba2_chunked";
    // `M2_SCAN_SMEM`: two stages of C, B, x and dt, the bf16 state, `dt x`
    // twice, `P`, and the chunk's steps and decay sums.
    const CH: u32 = 16;
    const LDV: u32 = 72;
    const STAGE: u32 = 2 * CH * 136 * 2 + CH * LDV * 2 + CH * FLOAT;
    const SMEM: u32 = 2 * STAGE + 128 * LDV * 2 + 2 * CH * LDV * 2 + CH * 24 * 2 + 2 * CH * FLOAT;
    m.check(OP, xbc.data, dt, z, y, state_dim)?;
    let lanes = requests(OP, xbc)?;
    f32_state(OP, state)?;
    seated(OP, state, "mamba2_scan", true, true, true)?;
    ctx.fire(
        OP,
        Fire::at(kernel(MAMBA2, "::pie::ssm::mamba2_scan"))
            .apply(Launch::grid([lanes * m.heads, 1, 1], [128, 1, 1]).smem(SMEM)),
        &[
            xbc.data.arg(),
            dt.arg(),
            z.arg(),
            m.a_log.arg(),
            m.dt_bias.arg(),
            m.d.arg(),
            state.slab.arg(),
            state.slot_ids.arg(),
            xbc.indptr.arg(),
            state.slot_stride_elems.arg(),
            state.commit_len.arg(),
            state.begin_at.arg(),
            state.write_state.arg(),
            state.write_state_mask.arg(),
            y.arg(),
            stated(OP, lanes)?.arg(),
            stated(OP, m.heads)?.arg(),
            stated(OP, m.groups)?.arg(),
            ctx.stage(),
        ],
    )?;
    m.norm(ctx, OP, y)
}
