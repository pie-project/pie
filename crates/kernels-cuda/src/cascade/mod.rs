#![allow(clippy::too_many_arguments)]

pub mod merge_states;

use core::ptr::NonNull;
use kernels::{Bind, Fire};

use crate::jit::abi::bf16;
use crate::jit::{Ctx, Launch};
use kernels::Refusal;

const NO_ROW: Refusal = Refusal::Unstated {
    what: "a cascade merge at this head dim -- 64, 128, 256 and 512 are here",
};

pub const HEAD_DIMS: &[u32] = &[64, 128, 256, 512];

pub const NUM_SMEM_STAGES: u32 = 4;

pub const NUM_THREADS: u32 = 128;

#[must_use]
pub const fn geometry(head_dim: u32) -> Option<(u32, u32, u32)> {
    let vec_size = match head_dim {
        64 | 128 | 256 => 8,
        512 => 16,
        _ => return None,
    };
    let bdx = head_dim / vec_size;
    Some((vec_size, bdx, NUM_THREADS / bdx))
}

#[must_use]
pub const fn smem_bytes(head_dim: u32) -> Option<u32> {
    let Some((_, _, bdy)) = geometry(head_dim) else {
        return None;
    };
    Some(NUM_SMEM_STAGES * bdy * head_dim * 2 + NUM_THREADS * 4)
}

const fn merge_inst(head_dim: u32) -> Option<&'static str> {
    match head_dim {
        64 | 128 | 256 => Some(
            "::flashinfer::MergeStatesKernel<\
                                    8, ::pie::cascade::DTypeIn, ::pie::cascade::DTypeO>",
        ),
        512 => Some(
            "::flashinfer::MergeStatesKernel<\
                         16, ::pie::cascade::DTypeIn, ::pie::cascade::DTypeO>",
        ),
        _ => None,
    }
}

const fn merge_large_inst(head_dim: u32) -> Option<&'static str> {
    match head_dim {
        64 => Some(
            "::flashinfer::MergeStatesLargeNumIndexSetsKernel<\
                        8, 8, 16, 4, ::pie::cascade::DTypeIn, ::pie::cascade::DTypeO>",
        ),
        128 => Some(
            "::flashinfer::MergeStatesLargeNumIndexSetsKernel<\
                         8, 16, 8, 4, ::pie::cascade::DTypeIn, ::pie::cascade::DTypeO>",
        ),
        256 => Some(
            "::flashinfer::MergeStatesLargeNumIndexSetsKernel<\
                         8, 32, 4, 4, ::pie::cascade::DTypeIn, ::pie::cascade::DTypeO>",
        ),
        512 => Some(
            "::flashinfer::MergeStatesLargeNumIndexSetsKernel<\
                         16, 32, 4, 4, ::pie::cascade::DTypeIn, ::pie::cascade::DTypeO>",
        ),
        _ => None,
    }
}

const fn merge_varlen_inst(head_dim: u32) -> Option<&'static str> {
    match head_dim {
        64 => Some(
            "::flashinfer::PersistentVariableLengthMergeStatesKernel<\
                        8, 8, 16, 4, ::pie::cascade::DTypeIn, ::pie::cascade::DTypeO, ::pie::cascade::IdType>",
        ),
        128 => Some(
            "::flashinfer::PersistentVariableLengthMergeStatesKernel<\
                         8, 16, 8, 4, ::pie::cascade::DTypeIn, ::pie::cascade::DTypeO, ::pie::cascade::IdType>",
        ),
        256 => Some(
            "::flashinfer::PersistentVariableLengthMergeStatesKernel<\
                         8, 32, 4, 4, ::pie::cascade::DTypeIn, ::pie::cascade::DTypeO, ::pie::cascade::IdType>",
        ),
        512 => Some(
            "::flashinfer::PersistentVariableLengthMergeStatesKernel<\
                         16, 32, 4, 4, ::pie::cascade::DTypeIn, ::pie::cascade::DTypeO, ::pie::cascade::IdType>",
        ),
        _ => None,
    }
}

pub fn merge_states(
    ctx: &Ctx<'_>,
    v: *mut bf16,
    s: *mut f32,
    v_merged: *mut bf16,
    s_merged: *mut f32,
    num_index_sets: u32,
    seq_len: u32,
    num_heads: u32,
    head_dim: u32,
) -> Result<(), Refusal> {
    const MAX_BLOCK_THREADS: u32 = 1024;

    let (_, bdx, bdy) = geometry(head_dim).ok_or(NO_ROW)?;
    null_check(v.is_null(), "v")?;
    null_check(s.is_null(), "s")?;
    null_check(v_merged.is_null(), "v_merged")?;

    if num_index_sets >= seq_len {
        let smem = smem_bytes(head_dim).ok_or(NO_ROW)?;
        let instantiation = merge_large_inst(head_dim).ok_or(NO_ROW)?;

        return ctx.fire(
            Fire::at("cascade/merge_states.cuh", instantiation)
                .apply(Launch::grid([seq_len, num_heads, 1], [bdx, bdy, 1]).smem(smem)),
            &[
                v.arg(),
                s.arg(),
                v_merged.arg(),
                NonNull::new(s_merged).arg(),
                num_index_sets.arg(),
                num_heads.arg(),
            ],
        );
    }

    let threads = bdx.saturating_mul(num_heads);
    if threads > MAX_BLOCK_THREADS {
        return Err(Refusal::Wide {
            what: "threads per block, which `MergeStatesKernel` sizes by num_heads",
            at: i64::from(threads),
            max: i64::from(MAX_BLOCK_THREADS),
        });
    }
    let instantiation = merge_inst(head_dim).ok_or(NO_ROW)?;

    ctx.fire(
        Fire::at("cascade/merge_states.cuh", instantiation)
            .apply(Launch::grid([seq_len, 1, 1], [bdx, num_heads, 1])),
        &[
            v.arg(),
            s.arg(),
            v_merged.arg(),
            NonNull::new(s_merged).arg(),
            num_index_sets.arg(),
            num_heads.arg(),
            head_dim.arg(),
        ],
    )
}

pub fn merge_states_varlen(
    ctx: &Ctx<'_>,
    v: *mut bf16,
    s: *mut f32,
    indptr: *mut i32,
    v_merged: *mut bf16,
    s_merged: *mut f32,
    max_seq_len: u32,
    seq_len: *mut u32,
    num_heads: u32,
    head_dim: u32,
) -> Result<(), Refusal> {
    let (_, bdx, bdy) = geometry(head_dim).ok_or(NO_ROW)?;
    let smem = smem_bytes(head_dim).ok_or(NO_ROW)?;
    let instantiation = merge_varlen_inst(head_dim).ok_or(NO_ROW)?;
    null_check(v.is_null(), "v")?;
    null_check(s.is_null(), "s")?;
    null_check(indptr.is_null(), "indptr")?;
    null_check(v_merged.is_null(), "v_merged")?;

    let num_sms = ctx.multiprocessors()?.max(1);
    let blocks = grid_blocks(
        blocks_per_sm(instantiation, smem),
        max_seq_len,
        num_heads,
        num_sms,
    );

    ctx.fire(
        Fire::at("cascade/merge_states.cuh", instantiation)
            .apply(Launch::grid([blocks, 1, 1], [bdx, bdy, 1]).smem(smem)),
        &[
            v.arg(),
            s.arg(),
            indptr.arg(),
            v_merged.arg(),
            NonNull::new(s_merged).arg(),
            max_seq_len.arg(),
            NonNull::new(seq_len).arg(),
            num_heads.arg(),
        ],
    )
}

fn null_check(is_null: bool, which: &'static str) -> Result<(), Refusal> {
    if is_null {
        Err(Refusal::Null { what: which })
    } else {
        Ok(())
    }
}

fn grid_blocks(per_sm: u32, max_seq_len: u32, num_heads: u32, num_sms: u32) -> u32 {
    let work_bound = max_seq_len
        .saturating_mul(num_heads)
        .div_ceil(num_sms)
        .max(1);
    per_sm.min(work_bound).saturating_mul(num_sms).max(num_sms)
}

#[cfg(feature = "_cuda")]
fn blocks_per_sm(instantiation: &str, smem: u32) -> u32 {
    use cudarc::driver::sys as dr;

    let Ok(resolved) = crate::jit::cache::resolve(
        &crate::jit::Root::new("cascade/merge_states.cuh"),
        instantiation,
    ) else {
        return 1;
    };
    let mut blocks: core::ffi::c_int = 0;

    let code = unsafe {
        dr::cuOccupancyMaxActiveBlocksPerMultiprocessor(
            &raw mut blocks,
            resolved.function,
            i32::try_from(NUM_THREADS).unwrap_or(i32::MAX),
            usize::try_from(smem).unwrap_or(usize::MAX),
        )
    };
    if code != dr::CUresult::CUDA_SUCCESS {
        return 1;
    }
    u32::try_from(blocks).unwrap_or(1).max(1)
}

#[cfg(not(feature = "_cuda"))]
fn blocks_per_sm(_instantiation: &str, _smem: u32) -> u32 {
    1
}
