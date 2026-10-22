use crate::error::Error;
use dtype::Dtype;

use crate::jit::{
    Arg, ArgValue, Ctx, Fire, Launch, aligned16, dtype_dispatch, nonzero, refuse, stated, symbol,
};
use crate::tensor::{RaggedTensor, RecurrentPool, Tensor};

const GDN_PREP: &str = "ssm/gdn_prep.cu";

mod conv1d;
mod dyn_conv;
mod gdn_chunk;
mod gdn_prep;
mod gdn_recurrent;
mod kda;
mod mamba2;

pub use conv1d::{causal_conv1d, causal_conv1d_chunked, short_conv, short_conv_chunked};
pub use dyn_conv::dyn_conv;
pub use gdn_chunk::{gated_delta_chunked, gated_delta_chunked_serial, gdn_lands_snapshots};
pub use gdn_prep::gdn_gates;
pub use gdn_recurrent::gated_delta;
pub use kda::{kda, kda_chunked};
pub use mamba2::{Mamba2, mamba2, mamba2_chunked};

const BLOCK: u32 = 256;

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
