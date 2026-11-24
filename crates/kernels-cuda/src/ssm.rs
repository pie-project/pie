use crate::error::Error;
use dtype::Dtype;

use crate::jit::{
    Arg, ArgValue, Ctx, Fire, Launch, aligned16, dtype_dispatch, nonzero, refuse, stated,
};
use crate::tensor::{RaggedTensor, RecurrentPool, StateRows, Tensor};

const GDN_PREP: &str = "ssm/gdn_prep.cu";

mod conv1d;
mod dyn_conv;
mod gdn_chunk;
mod gdn_prep;
mod gdn_recurrent;
mod kda;
mod kda_chunk;
mod mamba2;

pub use conv1d::conv1d_fold;
pub use conv1d::{
    causal_conv1d, causal_conv1d_carried, causal_conv1d_chunked, short_conv,
    short_conv_chunked,
};
pub use dyn_conv::dyn_conv;

/// The kernel of a beam group's fold by the step it folds
/// for: launched eagerly ahead of a fire's body, so no arming fire reaches
/// it, and a load makes its kernel beforehand ([`FoldKernel::load`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FoldKernel {
    Conv1d,
    GatedDelta,
    Mamba2,
    Kda { head_dim: u32 },
}

impl FoldKernel {
    /// Compiles (or reads from the disk cache) and loads this fold's kernel
    /// on the current device, so that no serving fire does.
    pub fn load(self) -> Result<(), Error> {
        match self {
            FoldKernel::Conv1d => conv1d::fold_kernel(),
            FoldKernel::GatedDelta => gdn_recurrent::fold_kernel(),
            FoldKernel::Mamba2 => mamba2::fold_kernel(),
            FoldKernel::Kda { head_dim } => kda::fold_kernel(head_dim),
        }
        .load()
    }
}
pub use gdn_chunk::{gated_delta_chunked, gated_delta_chunked_serial, gdn_lands_snapshots};
pub use gdn_prep::gdn_gates;
pub use gdn_recurrent::{
    GdnGated, carry_split_bytes, gated_delta, gated_delta_carried, gated_delta_fold, gated_delta_tail,
};
pub use kda::{kda, kda_carried, kda_chunked, kda_fold, kda_tail};
pub use mamba2::{Mamba2, mamba2, mamba2_carried, mamba2_chunked, mamba2_fold, mamba2_tail};

const BLOCK: u32 = 256;

/// A decode step's carried rows (`ssm/carry.cuh`'s `Carry`), passed by
/// value: the staged rs tables a lane's buffered rows are read through
/// (`runtime::channel::RsTables`), each lane's commit length over its run
/// (carried rows, then its own), and where the op's buffered planes lie in
/// a page's slot. [`Carry::NONE`]: no lane carries a row, lands one, or
/// commits short of its own.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Carry {
    pub lanes: u64,
    pub pages: u64,
    pub replay_spans: u64,
    pub commit_len: u64,
    pub plane_off: [u64; 4],
    pub plane_stride: [u64; 4],
    pub page_tokens: u32,
    pub pad: u32,
    /// Where a gated-delta decode step may split a lane's carried rows
    /// across blocks ([`gated_delta_carried`]): a counter a unit, then each
    /// part's partial sums, [`CARRY_SPLIT_BYTES_PER_BLOCK`] bytes a block of
    /// its grid; 0 where it splits none.
    pub split: u64,
    /// The fire's beams' rows ([`crate::runtime::channel::RsRows`]), a
    /// fire lane each, read by a lane whose page offset names one
    /// ([`crate::runtime::channel::RsLane::ROWS`]); 0 where no lane is a beam.
    pub lane_rows: u64,
}

const _: () = assert!(size_of::<Carry>() == 120);

/// The carried rows a decode step stages at once (`gdn_recurrent.cu`'s
/// `kStageTile`): a lane carrying more spans chunks, which a split shares
/// out.
pub const CARRY_CHUNK: u32 = 16;

/// The bytes a gated-delta decode step's split takes a block of its grid:
/// its unit's counter and its part's partial sums (`a` and `o`, 128 values
/// each, f32).
pub const CARRY_SPLIT_BYTES_PER_BLOCK: u64 = 4 + (2 * 128 + 4) * 4;

impl Carry {
    pub const NONE: Carry = Carry {
        lanes: 0,
        pages: 0,
        replay_spans: 0,
        commit_len: 0,
        plane_off: [0; 4],
        plane_stride: [0; 4],
        page_tokens: 1,
        pad: 0,
        split: 0,
        lane_rows: 0,
    };

    /// The kernels' argument.
    #[must_use]
    pub fn arg(&self) -> ArgValue<'_> {
        ArgValue::of(self)
    }

    /// Whether lanes may carry rows: the fire buffers.
    #[must_use]
    pub const fn buffers(&self) -> bool {
        self.lanes != 0
    }
}

const WARP: u32 = 32;

const FLOAT: u32 = 4;

const PREP_BLOCK: u32 = 128;

const GDN_BLOCK: u32 = 128;

fn requests(op: &'static str, x: RaggedTensor) -> Result<u32, Error> {
    if x.indptr.dtype != Dtype::I32 {
        return Err(refuse(
            op,
            format!(
                "the query CSR's boundaries are {:?}, and this scan walks an i32 indptr",
                x.indptr.dtype
            ),
        ));
    }
    match x.indptr.rows.checked_sub(1) {
        Some(lanes) if lanes > 0 => Ok(lanes),
        _ => Err(refuse(op, "the query CSR this fire names spans no request")),
    }
}

#[derive(Clone, Copy)]
enum StagePlanes {
    Step,
    Chunk,
}

impl StagePlanes {
    const fn names(self) -> (&'static str, &'static str, &'static str) {
        match self {
            StagePlanes::Step => (
                "attn.ssm_gdn_step_qk",
                "attn.ssm_gdn_step_v",
                "attn.ssm_gdn_step_gates",
            ),
            StagePlanes::Chunk => (
                "attn.ssm_gdn_chunk_qk",
                "attn.ssm_gdn_chunk_v",
                "attn.ssm_gdn_chunk_gates",
            ),
        }
    }
}

fn plane(ctx: &Ctx, op: &'static str, name: &'static str, elems: u64) -> Result<u64, Error> {
    let bytes = elems.checked_mul(u64::from(FLOAT)).ok_or_else(|| {
        refuse(
            op,
            format!("the {elems}-element staging plane will not size"),
        )
    })?;
    let bytes = usize::try_from(bytes)
        .map_err(|_| refuse(op, format!("{bytes} staging bytes do not fit this host")))?;
    Ok(ctx.scratch_shared(op, name, bytes)? as u64)
}

fn seated(
    op: &'static str,
    state: &RecurrentPool,
    arm: &'static str,
    mask: bool,
    commit: bool,
    begin: bool,
) -> Result<(), Error> {
    if !mask && !state.write_state_mask.is_absent() {
        return Err(refuse(
            op,
            format!(
                "this fire carries a per-request fold predicate and `{arm}` has no seat for \
                 one, so a refused pass would fold anyway"
            ),
        ));
    }
    if !commit && !state.commit_len.is_absent() {
        return Err(refuse(
            op,
            format!(
                "this fire carries a commit length and `{arm}` has no seat for one, so the \
                 replay would fold the whole buffered window instead of its accepted prefix"
            ),
        ));
    }
    if !begin && !state.begin_at.is_absent() {
        return Err(refuse(
            op,
            format!(
                "this fire cuts a row at an interior fold boundary and `{arm}` has no seat for \
                 the segment's origin, so the tail would replay the head's tokens from the state \
                 the head just folded"
            ),
        ));
    }
    if !state.write_state && !mask {
        return Err(refuse(
            op,
            format!("`{arm}` folds its boundary unconditionally and this fire asked it not to"),
        ));
    }
    Ok(())
}

#[derive(Clone, Copy)]
struct Delta {
    n: u32,
    k_heads: u32,
    v_heads: u32,
    k_dim: u32,
    v_dim: u32,
    conv_dim: u32,
}

#[derive(Clone, Copy)]
struct DeltaStaged {
    q_norm: u64,
    k_norm: u64,
    v: u64,
    g_log: u64,
    beta: u64,
}

fn plane_bf16(ctx: &Ctx, op: &'static str, name: &'static str, elems: u64) -> Result<u64, Error> {
    let bytes = usize::try_from(elems.saturating_mul(2))
        .map_err(|_| refuse(op, format!("{elems} bf16 elements do not fit this host")))?;
    Ok(ctx.scratch_shared(op, name, bytes)? as u64)
}

fn f32_state(op: &'static str, state: &RecurrentPool) -> Result<(), Error> {
    if state.slab.dtype != Dtype::F32 {
        return Err(refuse(
            op,
            format!(
                "the KDA recurrence keeps an f32 state and this cache row is declared {:?}",
                state.slab.dtype
            ),
        ));
    }
    Ok(())
}

/// A beam group's fold (as `kernels-rocm`'s
/// `ssm::GroupFold`): its representative lane (any of its lanes; they share
/// the set's state), where its fold rows start in the replay table, and how
/// many. The fold kernels read a device table of them: the groups that fold
/// lead it, and the first entry of no rows ends it (a block folds each group
/// in turn, so an empty table costs a read, not a block a group).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct GroupFold {
    pub lane: u32,
    pub fold_off: u32,
    pub rows: u32,
}

const _: () = assert!(size_of::<GroupFold>() == 12);
