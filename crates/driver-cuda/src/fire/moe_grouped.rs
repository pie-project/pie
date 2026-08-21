//! `moe::moe_grouped_gemm_bf16`: the aligned MoE leg's two grouped GEMMs.
//! Picks the WMMA kernel when `x::moe::supported`, else batched cuBLAS; the
//! choice lives here, not a `bind!` dispatch, since a refusal is final.

use core::ffi::c_void;

use kernels_cuda::Refusal;

use kernels_cuda::jit::abi::bf16;

use crate::fire::moe_ptrs::Arrays;

/// `moe::moe_grouped_gemm_bf16` — pick the implementation, then launch it.
///
/// # Safety
///
/// `a`, `c`, `expert_ids` must be live device allocations sized for the
/// rectangle; `bank` is the `[E, N, K]` weight base. Batched-leg `arrays`
/// must be this fire's build output, live until the next sync.
#[cfg(feature = "_cuda")]
#[allow(clippy::too_many_arguments)]
pub unsafe fn grouped_gemm_bf16(
    handle: *mut c_void,
    arrays: Option<Arrays>,
    a: *const c_void,
    bank: *const c_void,
    c: *mut c_void,
    expert_ids: *const c_void,
    max_blocks: i32,
    m: i32,
    n: i32,
    k: i32,
    stream: *mut c_void,
) -> Result<(), Refusal> {
    // `batched_act_x_wt_bf16` returns silently on `batch_count <= 0`.
    if max_blocks <= 0 {
        return Err(Refusal::Empty {
            what: "the padded block count",
        });
    }

    if kernels_cuda::moe::supported(m, n, k).is_ok() {
        // SAFETY: caller's obligation per this fn's `# Safety` section.
        let ctx = unsafe { kernels_cuda::jit::Ctx::on(stream) };
        return match kernels_cuda::moe::moe_grouped_gemm::<bf16>(
            &ctx,
            kernels::routine::In {
                ptr: a.cast::<bf16>(),
                rows: 0,
                width: k,
            },
            kernels::routine::Const {
                v: bank.cast::<bf16>(),
            },
            kernels::routine::In {
                ptr: expert_ids.cast::<i32>(),
                rows: 0,
                width: 0,
            },
            // ONE ADDRESS IN BOTH RUNS, and LAST: the grouped GEMM
            // accumulates into the statement's third input, which is what
            // `in_place = &[(0, 2)]` used to say beside the row.
            kernels::routine::InOut {
                ptr: c.cast::<bf16>(),
                rows: 0,
                width: n,
            },
            // The alignment's own two numbers, which the signature takes as
            // `Const` because no driver answers `keys::MoeMaxBlocks` or
            // `keys::MoeAlignedRows` — see `kernels-cuda/src/moe.rs`. This
            // caller is the alignment, so it is the one that has them.
            //
            // Adjacent, same-typed and swappable: `max_blocks` is the block
            // ceiling and `m` the ALIGNED row count, in that order.
            kernels::routine::Const { v: max_blocks },
            kernels::routine::Const { v: m },
        ) {
            Ok(()) => Ok(()),
            Err(why) => Err(why),
        };
    }

    let Some(arrays) = arrays else {
        return Err(Refusal::Absent {
            what: "the six pointer arrays, which `moe::build_moe_ptrs_aligned_bf16` fills \
                   as step 3 of the aligned leg",
        });
    };
    let Some(half) = arrays.select(bank) else {
        return Err(Refusal::Absent {
            what: "this GEMM's bank among the two the pointer build carved for",
        });
    };
    let (a_ptrs, b_ptrs, c_ptrs) = arrays.triple(half);
    // `a`, `c`, `expert_ids` are unused; addresses are already in `arrays`.
    // SAFETY: the arrays hold `max_blocks` device addresses, per this fn's
    // safety doc.
    unsafe {
        kernels_cuda::gemm::dense::batched_act_x_wt_bf16(
            handle, a_ptrs, b_ptrs, c_ptrs, m, n, k, max_blocks, 0.0,
        );
    }
    Ok(())
}
