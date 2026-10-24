//! Partial attention states folded into one (`attn/merge.cu`): the split-KV
//! fold, the LSE merge, the sink rescale and the logit softcap.
#![allow(unsafe_code)]

use crate::error::Error;
use crate::jit::{
    Arg, ArgValue, Ctx, Fire, Kernel, Launch, count, dtype_dispatch, kernel, nonzero, refuse,
    stated,
};
use crate::tensor::Tensor;
use dtype::Dtype;

use crate::attn::elementwise;
use crate::attn::fa2::NUM_THREADS;
use crate::attn::fa2::abi::Partials;
use crate::attn::lse_plane;
use crate::attn::per_head_elementwise;
use crate::attn::row_heads;

const MERGE: &str = "attn/merge.cu";

const NUM_SMEM_STAGES: u32 = 4;

#[must_use]
const fn merge_geometry(head_dim: u32) -> Option<(u32, u32, u32)> {
    let vec_size = match head_dim {
        64 | 128 | 256 => 8,
        512 => 16,
        _ => return None,
    };
    let bdx = head_dim / vec_size;
    Some((vec_size, bdx, NUM_THREADS / bdx))
}

#[must_use]
const fn merge_smem_bytes(head_dim: u32) -> Option<u32> {
    let Some((_, _, bdy)) = merge_geometry(head_dim) else {
        return None;
    };
    Some(NUM_SMEM_STAGES * bdy * head_dim * 2 + NUM_THREADS * 4)
}

fn merge_varlen_inst(head_dim: u32) -> Option<String> {
    let (vec_size, bdx, bdy) = merge_geometry(head_dim)?;
    Some(format!(
        "::pie::attn::fa2::PersistentVariableLengthMergeStatesKernel<\
         {vec_size}, {bdx}, {bdy}, {NUM_SMEM_STAGES}, \
         ::pie::attn::merge_lse::DTypeIn, ::pie::attn::merge_lse::DTypeO, \
         ::pie::attn::merge_lse::IdType>",
    ))
}

fn no_merge_row(op: &'static str) -> Error {
    refuse(
        op,
        "no cascade merge is stamped at this head width -- 64, 128, 256 and 512 are here",
    )
}

fn grid_blocks(per_sm: u32, max_seq_len: u32, num_heads: u32, num_sms: u32) -> u32 {
    let work_bound = max_seq_len
        .saturating_mul(num_heads)
        .div_ceil(num_sms)
        .max(1);
    per_sm.min(work_bound).saturating_mul(num_sms).max(num_sms)
}

#[cfg(feature = "cuda")]
fn merge_blocks_per_sm(merge: &'static Kernel, smem: u32) -> u32 {
    use cudarc::driver::sys as dr;

    let Ok(resolved) = merge.resolve() else {
        return 1;
    };
    let mut blocks: core::ffi::c_int = 0;

    // SAFETY: `blocks` is a local out-pointer and `resolved.function` is a
    // handle the cache keeps loaded for the process; the query launches
    // nothing.
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

#[cfg(not(feature = "cuda"))]
fn merge_blocks_per_sm(_merge: &'static Kernel, _smem: u32) -> u32 {
    1
}

/// The split's partials folded a warp a (row, head), straight from global
/// memory: a decode's rows hold two or three partials each, which the
/// staged persistent merge spends most of its time synchronizing over.
fn fold_warp(ctx: &Ctx, op: &'static str, split: &Partials, num_sms: u32) -> Result<(), Error> {
    let sink = if split.sinks_f32 {
        "float"
    } else {
        "::pie::bf16"
    };
    const WARPS: u32 = 8;
    let pairs = split.max_seq_len.saturating_mul(split.num_heads);
    let blocks = pairs.div_ceil(WARPS).clamp(1, num_sms.saturating_mul(8));
    // A narrower live width folds its columns alone.
    let live = match split.live {
        live if live != 0 && live < split.head_dim && live.is_multiple_of(64) => {
            format!(", {live}")
        }
        _ => String::new(),
    };
    ctx.fire(
        op,
        Fire::at(kernel(
            MERGE,
            &format!(
                "::pie::attn::merge_partials_warp<{}, {sink}{live}>",
                split.head_dim
            ),
        ))
        .apply(Launch::grid([blocks, 1, 1], [32 * WARPS, 1, 1])),
        &[
            ArgValue::Ptr(split.tmp_v),
            ArgValue::Ptr(split.tmp_s),
            ArgValue::Ptr(split.indptr),
            ArgValue::Ptr(split.o),
            ArgValue::Ptr(split.lse),
            split.max_seq_len.arg(),
            ArgValue::Ptr(split.seq_len),
            split.num_heads.arg(),
            ArgValue::Ptr(split.sinks),
            ctx.stage(),
        ],
    )
}

/// Folds the split's partials; `Ok(true)` where the fold also scaled the
/// output by the split's sinks (the warp fold, where the profile takes it),
/// `Ok(false)` where the caller still owes them.
pub(crate) fn fold(ctx: &Ctx, op: &'static str, split: &Partials) -> Result<bool, Error> {
    let head_dim = split.head_dim;
    let (_, bdx, bdy) = merge_geometry(head_dim).ok_or_else(|| no_merge_row(op))?;
    let smem = merge_smem_bytes(head_dim).ok_or_else(|| no_merge_row(op))?;
    let merge = kernel(
        MERGE,
        &merge_varlen_inst(head_dim).ok_or_else(|| no_merge_row(op))?,
    );
    for (ptr, which) in [
        (split.tmp_v, "the partial value plane"),
        (split.tmp_s, "the partial state plane"),
        (split.indptr, "the merge indptr"),
        (split.o, "the folded output"),
    ] {
        if ptr == 0 {
            return Err(refuse(op, format!("{which} the fold reads is null")));
        }
    }

    let num_sms = ctx.multiprocessors().unwrap_or(1).max(1);
    // A wide batch's rows hold a few partials each and fold a warp a row; a
    // narrow one's rows hold many, which the staged merge pipelines.
    // With sinks the warp fold takes every batch: it lands them in the same
    // pass, which outweighs the staged merge's edge on a narrow batch.
    let pairs = split.max_seq_len.saturating_mul(split.num_heads);
    if ctx.tuning().merge_warp
        && head_dim.is_multiple_of(64)
        && (pairs >= num_sms * 8 || split.sinks != 0)
    {
        return fold_warp(ctx, op, split, num_sms).map(|()| split.sinks != 0);
    }
    let unsinked = Partials { sinks: 0, ..*split };
    let split = &unsinked;
    let blocks = grid_blocks(
        merge_blocks_per_sm(merge, smem),
        split.max_seq_len,
        split.num_heads,
        num_sms,
    );

    ctx.fire(
        op,
        Fire::at(merge).apply(
            Launch::grid([blocks, 1, 1], [bdx, bdy, 1])
                .smem(smem)
                .programmatic(ctx.tuning().pdl),
        ),
        &[
            ArgValue::Ptr(split.tmp_v),
            ArgValue::Ptr(split.tmp_s),
            ArgValue::Ptr(split.indptr),
            ArgValue::Ptr(split.o),
            ArgValue::Ptr(split.lse),
            split.max_seq_len.arg(),
            ArgValue::Ptr(split.seq_len),
            split.num_heads.arg(),
            ctx.stage(),
        ],
    )
    .map(|()| false)
}

pub fn sink(
    ctx: &Ctx,
    o: &mut Tensor,
    lse: Tensor,
    sink: Tensor,
    head_dim: u32,
) -> Result<(), Error> {
    const OP: &str = "attn.sink";
    let t = dtype_dispatch!(OP, o.dtype, { Bf16 => "::pie::bf16", F16 => "::pie::f16" });
    // The sinks are the scores' own element, or f32 as a checkpoint keeps them.
    let s = match sink.dtype {
        Dtype::F32 => "float",
        held if held == o.dtype => t,
        held => {
            return Err(refuse(
                OP,
                format!("the sinks are {held:?} and the scores {:?}", o.dtype),
            ));
        }
    };
    let heads = row_heads(OP, o.width, head_dim)?;
    lse_plane(OP, &lse, o.rows, heads);
    let rows = count(OP, "rows", o.rows)?;
    ctx.fire(
        OP,
        Fire::at(kernel(
            MERGE,
            &format!("::pie::attn::sink_rescale<{t}, {s}>"),
        ))
        .apply(per_head_elementwise(o.rows, heads, head_dim)),
        &[
            o.arg(),
            lse.arg(),
            sink.arg(),
            rows.arg(),
            stated(OP, heads)?.arg(),
            stated(OP, head_dim)?.arg(),
            ctx.stage(),
        ],
    )
}

#[allow(clippy::too_many_arguments)]
pub fn merge_lse(
    ctx: &Ctx,
    o1: Tensor,
    lse1: Tensor,
    o2: Tensor,
    lse2: Tensor,
    heads: u32,
    head_dim: u32,
    o: &mut Tensor,
    lse: &mut Tensor,
) -> Result<(), Error> {
    const OP: &str = "attn.merge_lse";

    const COMBINE_BLOCK_MIN: u32 = 32;

    const COMBINE_BLOCK_MAX: u32 = 256;

    let t = dtype_dispatch!(OP, o.dtype, { Bf16 => "::pie::bf16", F16 => "::pie::f16" });
    debug_assert!(
        o1.rows == o.rows && o2.rows == o.rows,
        "the merged outputs are one row per query row"
    );
    lse_plane(OP, lse, o.rows, heads);
    let heads = count(OP, "the head count this merge states", heads)?;
    let head_dim = count(OP, "the head width this merge states", head_dim)?;
    ctx.fire(
        OP,
        Fire::at(kernel(
            MERGE,
            &format!("::pie::attn::merge_lse_combine<{t}>"),
        ))
        .apply(Launch::grid(
            [o.rows, heads.unsigned_abs(), 1],
            [
                head_dim
                    .unsigned_abs()
                    .clamp(COMBINE_BLOCK_MIN, COMBINE_BLOCK_MAX),
                1,
                1,
            ],
        )),
        &[
            o1.arg(),
            lse1.arg(),
            o2.arg(),
            lse2.arg(),
            o.arg(),
            lse.arg(),
            heads.arg(),
            head_dim.arg(),
            ctx.stage(),
        ],
    )
}

pub fn logit_softcap(ctx: &Ctx, x: &mut Tensor, cap: f32) -> Result<(), Error> {
    const OP: &str = "elemwise.softcap";
    let t = dtype_dispatch!(OP, x.dtype, { Bf16 => "::pie::bf16", F16 => "::pie::f16" });
    if cap.is_nan() || cap <= 0.0 {
        return Err(refuse(OP, format!("{cap} is not a logit soft cap")));
    }
    let n = x.elements();
    let lanes = u32::try_from(n).map_err(|_| {
        refuse(
            OP,
            format!("{n} elements do not fit a 32-bit launch extent"),
        )
    })?;
    nonzero(OP, "the element count", lanes)?;
    // Eight elements a thread where the plane takes whole 16-byte runs: the
    // same values, element for element.
    let (entry, extent) = if lanes.is_multiple_of(8) && x.ptr.is_multiple_of(16) {
        ("logit_softcap_vec8", lanes / 8)
    } else {
        ("logit_softcap", lanes)
    };
    ctx.fire(
        OP,
        Fire::at(kernel(
            "attn/merge.cu",
            &format!("::pie::attn::{entry}<{t}>"),
        ))
        .apply(elementwise(extent)),
        &[x.arg(), cap.arg(), (extent as usize).arg()],
    )
}
