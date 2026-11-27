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
    let n = nonzero(OP, "rows", xbc.rows)?;
    let rows = state.rows(OP, state.slab, state.slot_stride_elems, m.heads)?;
    ctx.fire(
        OP,
        Fire::at(kernel(MAMBA2, "::pie::ssm::mamba2_step"))
            .apply(Launch::grid([n, m.heads, 1], [BLOCK, 1, 1])),
        &[
            xbc.arg(),
            dt.arg(),
            z.arg(),
            m.a_log.arg(),
            m.dt_bias.arg(),
            m.d.arg(),
            rows.arg(),
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

/// [`mamba2`] over each lane's carried rows then its own ([`Carry`]), the
/// carried rows read as a linear attention from the committed state: their
/// step sizes (plane 0) and convolved `x | B` (plane 1) as one dot and one
/// decay a row; each head's state folded through the lane's commit length.
/// The own row's step size lands in the lane's buffer.
#[allow(clippy::too_many_arguments)]
pub fn mamba2_carried(
    ctx: &Ctx,
    xbc: Tensor,
    dt: Tensor,
    z: Tensor,
    m: Mamba2,
    state: &RecurrentPool,
    state_dim: u32,
    y: &mut Tensor,
    carry: &Carry,
) -> Result<(), Error> {
    const OP: &str = "attn.mamba2_step";
    m.check(OP, xbc, dt, z, y, state_dim)?;
    f32_state(OP, state)?;
    seated(OP, state, "mamba2_step_carried", true, false, false)?;
    let n = nonzero(OP, "rows", xbc.rows)?;
    let rows = state.rows(OP, state.slab, state.slot_stride_elems, m.heads)?;
    ctx.fire(
        OP,
        Fire::at(kernel(MAMBA2, "::pie::ssm::mamba2_step_carried"))
            .apply(Launch::grid([n, m.heads, 1], [BLOCK, 1, 1])),
        &[
            xbc.arg(),
            dt.arg(),
            z.arg(),
            m.a_log.arg(),
            m.dt_bias.arg(),
            m.d.arg(),
            rows.arg(),
            y.arg(),
            stated(OP, m.heads)?.arg(),
            stated(OP, m.groups)?.arg(),
            state.write_state.arg(),
            state.write_state_mask.arg(),
            carry.arg(),
            ctx.stage(),
        ],
    )?;
    m.norm(ctx, OP, y)
}

/// The rows a chunked Mamba-2 fire's lanes land in their buffer past their
/// commit, made ready for a later decode ([`mamba2_carried`]): each row's
/// convolved `x | B` (plane 1; its step size, plane 0, the fire lands).
/// `xbc` is the fire's own rows by `qo_indptr`, or with `ext` its extended
/// rows (row `RsLane::ext + j`).
#[allow(clippy::too_many_arguments)]
pub fn mamba2_tail(
    ctx: &Ctx,
    xbc: Tensor,
    m: Mamba2,
    state: &RecurrentPool,
    qo_indptr: Tensor,
    ext: bool,
    state_dim: u32,
    carry: &Carry,
) -> Result<(), Error> {
    const OP: &str = "ssm.mamba2_chunked";
    if !carry.buffers() {
        return Ok(());
    }
    dtype_dispatch!(OP, xbc.dtype, { Bf16 => () });
    let lanes = qo_indptr.rows.saturating_sub(1);
    if lanes == 0 {
        return Ok(());
    }
    let xb = xbc.width - m.groups * state_dim;
    if !xb.is_multiple_of(8) {
        return Err(refuse(
            OP,
            format!("a {xb}-wide `x | B` row the tail copies in 16-byte runs"),
        ));
    }
    ctx.fire(
        OP,
        Fire::at(kernel(MAMBA2, "::pie::ssm::mamba2_tail_rows"))
            .apply(Launch::grid([lanes, 1, 1], [BLOCK, 1, 1])),
        &[
            xbc.arg(),
            qo_indptr.arg(),
            ext.arg(),
            stated(OP, xbc.width)?.arg(),
            stated(OP, xb)?.arg(),
            state.write_state.arg(),
            state.write_state_mask.arg(),
            stated(OP, lanes)?.arg(),
            carry.arg(),
            ctx.stage(),
        ],
    )
}

/// A prompt's Mamba-2 walk in sixteen-token chunks: a block a (request,
/// head) lands `(C S + D x) * silu(z)` for each row, then the group norm.
/// A `tail` pass (rows past a lane's commit, from the state its first pass
/// folded) walks before the norm, which every row then takes once.
#[allow(clippy::too_many_arguments)]
pub fn mamba2_chunked(
    ctx: &Ctx,
    xbc: RaggedTensor,
    dt: Tensor,
    z: Tensor,
    m: Mamba2,
    state: &RecurrentPool,
    tail: Option<&RecurrentPool>,
    state_dim: u32,
    y: &mut Tensor,
) -> Result<(), Error> {
    const OP: &str = "ssm.mamba2_chunked";
    for pass in core::iter::once(state).chain(tail) {
        mamba2_scan(ctx, xbc, dt, z, m, pass, state_dim, y)?;
    }
    m.norm(ctx, OP, y)
}

/// One pass of [`mamba2_chunked`]'s walk, the norm not taken.
#[allow(clippy::too_many_arguments)]
fn mamba2_scan(
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
    let rows = state.rows(OP, state.slab, state.slot_stride_elems, m.heads)?;
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
            rows.arg(),
            xbc.indptr.arg(),
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
    )
}

/// Folds each beam group's fold rows (`carry`'s
/// planes step size and `x | B`) into its Mamba-2 state, as a decode step
/// folds a committed row, the state stored once.
/// The groups that fold lead `folds` (`groups` entries at most, the first of
/// no rows ending them); a block folds each in turn, so a fire none of whose
/// groups folds costs one read a block.
pub fn mamba2_fold(
    ctx: &Ctx,
    state: &RecurrentPool,
    m: Mamba2,
    carry: &Carry,
    folds: Tensor,
    groups: u32,
) -> Result<(), Error> {
    const OP: &str = "ssm.mamba2 group fold";
    const BLOCK: u32 = 256;
    if groups == 0 {
        return Ok(());
    }
    let rows = state.rows(OP, state.slab, state.slot_stride_elems, m.heads)?;
    ctx.fire(
        OP,
        Fire::at(fold_kernel()).apply(Launch::grid([m.heads, 1, 1], [BLOCK, 1, 1])),
        &[
            rows.arg(),
            m.a_log.arg(),
            m.dt_bias.arg(),
            stated(OP, m.heads)?.arg(),
            stated(OP, m.groups)?.arg(),
            carry.arg(),
            folds.arg(),
            stated(OP, groups)?.arg(),
        ],
    )
}

/// Mamba-2's group fold kernel ([`mamba2_fold`]).
pub(super) fn fold_kernel() -> &'static crate::jit::Kernel {
    kernel(MAMBA2, "::pie::ssm::mamba2_fold")
}
