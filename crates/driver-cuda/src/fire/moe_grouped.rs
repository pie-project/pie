//! `moe::moe_grouped_gemm_bf16` — the aligned MoE leg's two grouped GEMMs,
//! and the choice between the two implementations that serve them.
//!
//! # One symbol, two implementations, and why that makes it a driver op
//!
//! `x::moe::supported(m, n, k)` is the predicate: `M` exactly one 16-row
//! fragment, `N` in whole 64-wide tiles, and `K` at or under `SHORT_K` — 512,
//! **"above which cuBLAS wins"**, which is that refusal's own words. Inside
//! the predicate the WMMA kernel is 3.0x the batched library call at both of
//! qwen3.5's shapes (`x/moe.rs`'s decode census). Outside it there is no
//! kernel, and the library call is the only arithmetic there is.
//!
//! qwen3.5 is outside it on one of the two halves and inside on the other:
//!
//! ```text
//!   gate_up   M=16  N=2*I=1024  K=H=2048   K > 512  ->  batched cuBLAS
//!   down      M=16  N=H=2048    K=I=512    supported ->  WMMA
//! ```
//!
//! So the symbol needs both, and the choice is made **inside one host
//! program** rather than by a dispatch that tries one and catches the other.
//! `bind/mod.rs`'s floor says why that is not a matter of taste: *"A refusal
//! is not a fallthrough. A symbol the registry HOLDS is dispatched here or
//! not at all. If its bind refuses, that is the answer — `NoArm` carrying the
//! sentence the bind wrote, not a walk down to a hand arm that would fire it
//! with different arithmetic."* A `bind!` that returned `Refusal::Wide` for
//! `K = 2048` is therefore final, and the only place the second
//! implementation can live is the shape where the driver, not the registry,
//! owns the body.
//!
//! **Name the resource** — `x/mod.rs`'s one-line discriminator for a driver
//! op. Two, and both are the driver's: the **cuBLAS handle** on `ctx.cublas`,
//! which §3.3 forbids a `Cx` to hand over (it has a settable stream, a math
//! mode and a workspace), and the **pointer-array arena** on
//! `ctx.moe_ptrs`, which is not a stated operand of anything. Neither is a
//! fact `Cx` is missing; both are surfaces `Cx` must not grow.
//!
//! # The fallback did not need writing. It needed a CALLER.
//!
//! [`kernels_cuda::gemm::dense::batched_act_x_wt_bf16`] is this
//! module's second leg, and it was already in the tree, already ported,
//! already carrying the capture latch that makes the grouped form safe under
//! stream capture. Its own doc says what it was waiting for: *"**This symbol
//! has no row.** `table::gemm` struck `gemm::batched_act_x_wt_bf16` (§38)
//! because its whole consumer set was one unreachable inline"*, ported anyway
//! under §45.2's *"porting them unfaithfully is how you get 99.83% of the
//! right answer"*.
//!
//! It is the C++'s `gemm.cpp:1145-1241` verbatim — `cublasGemmGroupedBatchedEx`
//! falling back to `cublasGemmBatchedEx` — which is the same pair of calls the
//! deleted `Control::Switch` walk described for this symbol. The gap that has
//! stood since the aligned leg was written was never the arithmetic; it was
//! that nothing named the function. **A body with no caller and a caller with
//! no body were four hundred lines apart in two crates for the whole of it.**
//!
//! # What this module does not check, stated because it cannot
//!
//! The batched leg writes where the **pointer build** baked its destinations,
//! not to the `c` this statement carries. They are the same address because
//! the contract states `in_place: &[(0, 2)]` — the GEMM's result IS the
//! staging the build declared — and because both statements resolve that
//! value through the same window in the same frame. If a planner ignored the
//! pair, the WMMA leg would still be right (it writes `c`) and this one would
//! write bytes the swiglu never reads. Comparing `c` against a stage base
//! here would catch that, and it is deliberately not done: the two statements
//! take their row windows from different tensors, so an address that differs
//! for a reason having nothing to do with `in_place` would refuse a live
//! path. The invariant is the contract's to state and the planner's to keep.

use core::ffi::c_void;

use kernels_cuda::Refusal;
use kernels_cuda::jit::abi::bf16;

use crate::fire::moe_ptrs::Arrays;

/// `moe::moe_grouped_gemm_bf16` — pick the implementation, then launch it.
///
/// `arrays` is the fire's [`Arrays`] if `moe::build_moe_ptrs_aligned_bf16`
/// ran and `None` if it did not. **The `Option` is resolved here and not by
/// the caller**, and that is the whole reason it is a parameter: the WMMA leg
/// does not read the arrays at all, so an arm that refused on `None` would
/// refuse the `down` half — which is supported, fires today, and needs
/// nothing the build produces.
///
/// Returns `Fired` rather than a two-state type of its own, unlike
/// [`crate::fire::moe_ptrs::Built`] next door: that one carries the arrays
/// out on its `Ready` arm and has an arena failure to name, and this one has
/// neither. Every outcome here is "it went to the device" or "a `Refusal`",
/// which is `Fired` exactly.
///
/// # Safety
///
/// `a`, `c` and `expert_ids` must be live device allocations of the aligned
/// rectangle's shapes on `stream`, and `bank` the base of the `[E, N, K]`
/// weight the statement names. On the batched leg the six arrays in `arrays`
/// must be the ones the build filled for THIS fire — cuBLAS dereferences
/// them on the device, so they and everything they address must outlive the
/// launch, which ends at the next synchronisation and not at this return.
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
    // `moe_grouped_gemm.cu:37`, and it is the host program's own first test:
    // an empty padded batch has nothing to launch OVER, which is a different
    // fact from a rectangle no implementation can compute. Made here as well
    // as there because the batched leg does not reach the host program —
    // `batched_act_x_wt_bf16` returns silently on `batch_count <= 0`, and a
    // silent return is not an answer this caller can report.
    if max_blocks <= 0 {
        return Err(Refusal::Empty { what: "the padded block count" });
    }

    if kernels_cuda::moe::supported(m, n, k).is_ok() {
        // SAFETY: the caller's obligation. The host program repeats the
        // predicate and would decline on its own; it is asked here first
        // because the answer selects the implementation rather than reporting
        // one.
        let ctx = unsafe { kernels_cuda::jit::Ctx::on(stream) };
        return match kernels_cuda::moe::moe_grouped_gemm::<bf16>(
            &ctx,
            a.cast::<bf16>(),
            bank.cast::<bf16>(),
            c.cast::<bf16>(),
            expert_ids.cast::<i32>(),
            max_blocks,
            m,
            n,
            k,
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
    // `a`, `c` and `expert_ids` are unused on this leg and that is the point
    // of the leg: every block's three addresses are already IN the arrays,
    // computed on the device from `expert_ids` by the build. The host has
    // `max_blocks` uniform blocks of `M` rows each and nothing else to say.
    //
    // `beta = 0` — the destination is written, not accumulated. The row this
    // replaced carried no beta and the C++ passed the same literal.
    //
    // No stream is passed and none is set: `ctx.cublas`'s stream is bound
    // per fire to the same `ctx.stream` the WMMA leg above launches on, and
    // a `cublasSetStream` here would be a second place for that to be true.
    //
    // SAFETY: the caller's obligation. The three arrays are device arrays of
    // `max_blocks` device addresses, which is what this entry point's own
    // safety note requires.
    unsafe {
        kernels_cuda::gemm::dense::batched_act_x_wt_bf16(
            handle, a_ptrs, b_ptrs, c_ptrs, m, n, k, max_blocks, 0.0,
        );
    }
    Ok(())
}
