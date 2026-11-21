//! The gated delta rule, token by token (`ssm.gated_delta`, its chunked and
//! committed forms), from `kernels/ssm/gdn_recurrent.slang`.

use dtype::Dtype;

use super::{Carry, CarryPlane, Committed, Csr, RowSpan, StatePool, head_width};
use crate::encode::{Arg, Ctx, Fire, Grid, Tier, dtype_dispatch, every, nonzero, refuse, stated};
use crate::error::Error;
use crate::tensor::Tensor;

const GDN_RECURRENT: &str = "ssm/gdn_recurrent.slang";
const GDN_CARRIED: &str = "ssm/gdn_carried.slang";

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

/// Refuses a pool a register-state kernel cannot read its state rows from
/// four words a vector: its segments, the state's origin and its slot stride
/// whole vectors.
fn vectors(op: &'static str, state: &StatePool) -> Result<(), Error> {
    if [state.pool.words, state.origin, state.slot_stride]
        .iter()
        .any(|words| !words.is_multiple_of(4))
    {
        return Err(refuse(
            op,
            format!(
                "the state pool's segments of {} words, origin {} and slot stride {} are not \
                 whole vectors of four words, which a register-state scan reads its state rows by",
                state.pool.words, state.origin, state.slot_stride
            ),
        ));
    }
    Ok(())
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
    span: RowSpan,
) -> Result<(), Error> {
    const OP: &str = "ssm.gated_delta";
    dtype_dispatch!(OP, qkv.dtype, { Bf16 => () });
    let heads = Heads::of(OP, qkv, gates, y, k_heads, v_heads, k_dim, v_dim)?;
    let launched = span.rows(OP, &[qkv, gates, y, state.slots])?;
    let entry = if heads.registers() {
        vectors(OP, state)?;
        "ssm_gdn_recurrent_r128_bf16"
    } else {
        "ssm_gdn_recurrent_bf16"
    };
    let [kh, vh, kd, vd] = heads.args(OP)?;
    ctx.fire(
        Fire::at(GDN_RECURRENT, entry)
            .tier(tactic.tier)
            .apply(heads.grid(launched)),
        &[
            &[qkv.arg(), gates.arg(), state.slots.arg(), y.arg_mut()][..],
            &state.piece_args(true),
            &[kh, vh, kd, vd, span.row0.arg()],
        ]
        .concat(),
    )
}

/// The row counts a scan's kernel changes at: one row, and the fewest that
/// hold their state in registers.
fn scan_breaks(tactic: Tactic) -> [u32; 2] {
    [1, tactic.register_min_rows.max(1)]
}

/// A scan's kernel over `state`: `registers` where its rows hold their state
/// in registers, else `memory`.
fn scan_kernel(
    op: &'static str,
    tactic: Tactic,
    state: &StatePool,
    in_registers: bool,
    registers: &'static str,
    memory: &'static str,
) -> Result<Fire, Error> {
    if in_registers {
        vectors(op, state)?;
    }
    let entry = if in_registers { registers } else { memory };
    Ok(Fire::at(GDN_RECURRENT, state.entry(op, entry)?).tier(tactic.tier))
}

/// Each request's rows of the query CSR, in order over its slot.
#[allow(clippy::too_many_arguments)]
pub fn gated_delta_chunked(
    ctx: &Ctx<'_>,
    tactic: Tactic,
    qkv: Tensor,
    csr: Csr,
    gates: Tensor,
    state: &StatePool,
    k_heads: u32,
    v_heads: u32,
    k_dim: u32,
    v_dim: u32,
    y: Tensor,
    span: RowSpan,
) -> Result<(), Error> {
    const OP: &str = "ssm.gated_delta_chunked";
    dtype_dispatch!(OP, qkv.dtype, { Bf16 => () });
    let heads = Heads::of(OP, qkv, gates, y, k_heads, v_heads, k_dim, v_dim)?;
    let kernel = |rows: u32| {
        scan_kernel(
            OP,
            tactic,
            state,
            heads.registers() && rows >= tactic.register_min_rows,
            "ssm_gdn_recurrent_chunked_r128_bf16",
            "ssm_gdn_recurrent_chunked_bf16",
        )
    };
    if let Some(names) = ctx.naming() {
        return names.either(&every(&scan_breaks(tactic), kernel)?);
    }
    let [kh, vh, kd, vd] = heads.args(OP)?;
    ctx.fire(
        kernel(span.rows(OP, &[qkv, gates, y])?)?.apply(heads.grid(csr.lanes(OP)?)),
        &[
            &[
                qkv.arg(),
                csr.table.arg(),
                gates.arg(),
                state.slots.arg(),
                y.arg_mut(),
            ][..],
            &state.snap_table(),
            &state.piece_args(true),
            &[kh, vh, kd, vd, span.row0.arg()],
            &[csr.at.arg()],
            &state.snap_lane0(OP)?,
        ]
        .concat(),
    )
}

/// A speculative window's rows (the replayed ones first) on a copy of each
/// request's state in `work`, the slot kept as of its committed row.
#[allow(clippy::too_many_arguments)]
pub fn gated_delta_committed(
    ctx: &Ctx<'_>,
    tactic: Tactic,
    qkv: Tensor,
    csr: Csr,
    committed: &Committed,
    gates: Tensor,
    state: &StatePool,
    work: Tensor,
    k_heads: u32,
    v_heads: u32,
    k_dim: u32,
    v_dim: u32,
    y: Tensor,
    row0: u32,
    carry: &Carry,
    [k_plane, u_plane]: [CarryPlane; 2],
) -> Result<(), Error> {
    const OP: &str = "ssm.gated_delta_committed";
    dtype_dispatch!(OP, qkv.dtype, { Bf16 => () });
    let heads = Heads::of(OP, qkv, gates, y, k_heads, v_heads, k_dim, v_dim)?;
    super::work_holds(OP, work, csr)?;
    let kernel = |rows: u32| {
        scan_kernel(
            OP,
            tactic,
            state,
            heads.registers() && rows >= tactic.register_min_rows,
            "ssm_gdn_recurrent_committed_r128_bf16",
            "ssm_gdn_recurrent_committed_bf16",
        )
    };
    if let Some(names) = ctx.naming() {
        return names.either(&every(&scan_breaks(tactic), kernel)?);
    }
    let [kh, vh, kd, vd] = heads.args(OP)?;
    ctx.fire(
        kernel(qkv.rows)?.apply(heads.grid(csr.lanes(OP)?)),
        &[
            &[
                qkv.arg(),
                csr.table.arg(),
                committed.replay.arg(),
                committed.commit.arg(),
                committed.slots.arg(),
                gates.arg(),
                work.arg_mut(),
                y.arg_mut(),
                carry.table.arg(),
            ][..],
            &state.snap_table(),
            &state.piece_args(true),
            &[
                stated(OP, committed.lane0)?.arg(),
                kh,
                vh,
                kd,
                vd,
                row0.arg(),
            ],
            &[csr.at.arg()],
            &state.snap_lane0(OP)?,
            &carry.page_args(OP)?,
            &k_plane.args(),
            &u_plane.args(),
        ]
        .concat(),
    )
}

/// A gated-delta layer's gate prep: its `[b | a]` rows (bf16), `dt_bias`
/// (bf16) and `A_log` (f32), which a carried step forms its rows' gates
/// from.
#[derive(Clone, Copy, Debug)]
pub struct Gated {
    pub ba: Tensor,
    pub dt_bias: Tensor,
    pub a_log: Tensor,
}

/// One decode step a lane over its buffered rows then its own
/// (`kernels/ssm/gdn_carried.slang`): the committed state read once, each
/// buffered row read where it lies in the lane's rs buffer pages (its
/// `[b | a]`, keys and corrected value: `planes`, in that order) as a linear
/// attention, the rows the lane commits folded, the own row's keys, `u` and
/// `[b | a]` landed in its buffer.
#[allow(clippy::too_many_arguments)]
pub fn gated_delta_carried(
    ctx: &Ctx<'_>,
    tactic: Tactic,
    qkv: Tensor,
    gated: Gated,
    state: &StatePool,
    carry: &Carry,
    planes: [CarryPlane; 3],
    k_heads: u32,
    v_heads: u32,
    k_dim: u32,
    v_dim: u32,
    y: Tensor,
    span: RowSpan,
) -> Result<(), Error> {
    gated_delta_carried_as(
        "ssm_gdn_carried_r128_bf16",
        ctx, tactic, qkv, gated, state, carry, planes, k_heads, v_heads, k_dim, v_dim, y, span,
    )
}

/// A beam group's fold (`BEAM-PAGE-LISTS.md` §11.3): each of `carry`'s
/// `folds` lanes from its `lane0` is a group's set, its carried rows the
/// ones every live beam shares, folded into its state ahead of the step
/// that reads it; no own row, no input read, no output written. `qkv`, the
/// gates and `y` are the step's, bound as it binds them.
#[allow(clippy::too_many_arguments)]
pub fn gated_delta_carried_fold(
    ctx: &Ctx<'_>,
    tactic: Tactic,
    qkv: Tensor,
    gated: Gated,
    state: &StatePool,
    carry: &Carry,
    planes: [CarryPlane; 3],
    k_heads: u32,
    v_heads: u32,
    k_dim: u32,
    v_dim: u32,
    y: Tensor,
    folds: u32,
) -> Result<(), Error> {
    gated_delta_carried_as(
        "ssm_gdn_carried_fold_r128_bf16",
        ctx,
        tactic,
        qkv,
        gated,
        state,
        carry,
        planes,
        k_heads,
        v_heads,
        k_dim,
        v_dim,
        y,
        RowSpan { row0: 0, rows: folds },
    )
}

#[allow(clippy::too_many_arguments)]
fn gated_delta_carried_as(
    entry: &'static str,
    ctx: &Ctx<'_>,
    tactic: Tactic,
    qkv: Tensor,
    gated: Gated,
    state: &StatePool,
    carry: &Carry,
    planes: [CarryPlane; 3],
    k_heads: u32,
    v_heads: u32,
    k_dim: u32,
    v_dim: u32,
    y: Tensor,
    span: RowSpan,
) -> Result<(), Error> {
    const OP: &str = "ssm.gated_delta_carried";
    dtype_dispatch!(OP, qkv.dtype, { Bf16 => () });
    nonzero(OP, "the key heads this statement states", k_heads)?;
    nonzero(OP, "the value heads this statement states", v_heads)?;
    if !v_heads.is_multiple_of(k_heads) {
        return Err(refuse(
            OP,
            format!("the {v_heads} value heads are not a whole number of the {k_heads} key heads"),
        ));
    }
    if k_dim != REGISTER_K || v_dim > GROUP {
        return Err(refuse(
            OP,
            format!(
                "the carried step holds a {REGISTER_K}-wide key row a value column in \
                 registers, at most {GROUP} columns; this layer's keys are {k_dim} wide and \
                 its values {v_dim}"
            ),
        ));
    }
    vectors(OP, state)?;
    let heads = Heads {
        k_heads,
        v_heads,
        k_dim,
        v_dim,
    };
    let launched = span.rows(OP, &[qkv, y])?;
    let [kh, vh, kd, vd] = heads.args(OP)?;
    let [ba_plane, k_plane, u_plane] = planes;
    ctx.fire(
        Fire::at(GDN_CARRIED, entry)
            .tier(tactic.tier)
            .apply(heads.grid(launched)),
        &[
            &[
                qkv.arg(),
                gated.ba.arg(),
                gated.dt_bias.arg(),
                gated.a_log.arg(),
                carry.slots.arg(),
                carry.table.arg(),
                y.arg_mut(),
            ][..],
            &state.piece_args(true),
            &[stated(OP, carry.lane0)?.arg(), kh, vh, kd, vd, span.row0.arg()],
            &carry.page_args(OP)?,
            &ba_plane.args(),
            &k_plane.args(),
            &u_plane.args(),
        ]
        .concat(),
    )
}
