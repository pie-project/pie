//===-- expert_offsets.cuh - the fused MoE's routing front-end -----------===//
//
// Four `__global__`s and one block collective. No host code at all: the grid
// arithmetic, the width ladders and the `cudaLaunchKernelEx` attribute that
// used to sit beside these kernels are `driver-cuda`'s
// `fire::flashinfer_moe`, in Rust.
//
// # What this computes
//
// The CUTLASS fused MoE cannot start until it knows, for every expert, where
// that expert's rows begin in the permuted activation buffer. That is the
// `expert_first_token_offset` array the grouped GEMM's `Params` is built
// from, and producing it is a three-phase segmented count:
//
// ```text
//   block_expert_prefix_sum     per (expert, token-block): how many of this
//                               block's tokens chose this expert, and which
//                               unpermuted rows they were
//   global_expert_prefix_sum    exclusive scan of that whole count matrix,
//   (or ..._large)              which turns per-block counts into per-block
//                               bases and, every `num_blocks_per_seq`-th
//                               element, into `expert_first_token_offset`
//   merge_expert_prefix_sum     scatter: each surviving row is written at its
//                               base + rank, and both permutation maps are
//                               filled
// ```
//
// The order is preserved within an expert, which is what makes the inverse
// map `unpermuted_row_to_permuted_row` a function rather than a relation, and
// it is why the middle phase is an *exclusive* scan of a matrix laid out
// expert-major.
//
// # Provenance
//
// Transcribed from FlashInfer's CPM-fetched
// `csrc/fused_moe/cutlass_backend/cutlass_fused_moe_kernels.cuh`
// (Apache-2.0, NVIDIA CORPORATION), which is TensorRT-LLM's MoE front-end
// vendored into FlashInfer. Line citations below are against that file at the
// revision this tree fetches. The kernel bodies are upstream's; what changed
// is the namespace, the integer spellings, and `cub::BlockScan` -- see below.
//
// # Why `<cub/cub.cuh>` is not here, and what replaced it
//
// The same answer `attn/page_compact.cuh` already gave, and this file is the
// second measurement of it rather than a new decision. Walking the include
// closure of the two cub headers these kernels need --
// `cub/block/block_scan.cuh` and `cub/block/block_radix_rank.cuh` -- against
// CCCL, the toolkit and this crate's shim resolves **429 files and 4,376,255
// bytes (4.2 MB)**. The carried set is `include_str!`, so that is 4.2 MB of
// binary for four kernels whose entire use of it is an exclusive sum of an
// `int`. §13.5 closed this door and the number is why.
//
// So `cub::BlockScan<int, BLOCK>::ExclusiveSum` is written out below against
// `__shfl_up_sync`, which NVRTC has as a builtin.
//
// **That is a body change and §8 wants evidence.** The evidence is the
// element type, exactly as it was for `page_compact.cuh`: every scan here
// folds `int` under `+`, which is exact and associative modulo 2^32, so any
// correct fold order produces the same bits. It is not "close enough to cub";
// it is the same integer. Note the difference from a float reduction in this
// shape, which would have to carry the original's order -- `attn_res.cuh`
// does, and that asymmetry is the rule, not an exception granted here.
//
// # What did NOT come across, and it is a fifth kernel
//
// `fusedBuildExpertMapsSortFirstTokenKernel` (`:350`) is upstream's
// single-block fast path for the same job -- it does all three phases in one
// launch. It is **not** here, and the reason is not NVRTC:
//
//  * it is built on `cub::BlockRadixRank<BLOCK_SIZE, LOG2_NUM_EXPERTS,
//    false>` (`:398`), not `BlockScan`, and it depends on that type's
//    *internal* layout in three separate ways -- `BINS_TRACKED_PER_THREAD`
//    sizes a per-thread array (`:406`), is `static_assert`ed against the
//    expert count (`:403`), and decides which thread writes which element of
//    `expert_first_token_offset` (`:428-434`). A hand-written rank that
//    produced the same *ranking* but a different bin-to-thread mapping would
//    write the offsets to the wrong places. That is not the exact-integer
//    argument above; it is a layout contract, and re-deriving it is a real
//    piece of work rather than twenty-six lines.
//  * its own dispatcher refuses any batch it cannot do in one block --
//    `TLLM_CHECK_WITH_INFO(blocks == 1, "Current implementation requires
//    single block")` (`:451`) -- so it is an optimisation over this file's
//    path, never the only path. Everything it can do, these four can do.
//
// So its absence costs a launch and a fast path at small batch, and costs no
// capability. It is written down here rather than in a report because the
// next reader's first question is *"where is the fifth one"*.
//
// # The PDL guard is upstream's and is kept
//
// Every kernel here opens with `cudaGridDependencySynchronize()` and closes
// with `cudaTriggerProgrammaticLaunchCompletion()`, both under upstream's
// `#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))`. NVRTC supplies
// neither name; `shim/cuda_runtime.h` does, as one inline-PTX
// instruction each (measured rc=0, `nvrtc-probes/
// cutlass_moe_c9_griddepcontrol.py`). The arch guard is kept exactly as
// upstream wrote it, which is what makes this text correct on the sm_89 box
// it is being written on -- `griddepcontrol` is an sm_90 instruction and the
// guard, not the shim, is what keeps it out of an sm_89 compile.
//
// The launcher's other half of that contract is now Rust: the guard decides
// whether the *instruction* is compiled, and
// `CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION` on the launch
// decides whether the *runtime* honours it. Both have to agree, and upstream
// carried the second in a `cudaLaunchConfig_t` field
// (`programmaticStreamSerializationAllowed`) threaded through five launchers
// as a `bool enable_pdl`.
//
//===--------------------------------------------------------------------===//

#include "prelude/device.cuh"

namespace pie::moe {


/// The widest block any of these kernels is launched at, in warps, plus one
/// slot for the block aggregate.
///
/// 1024 threads is upstream's ceiling in both places that pick a width --
/// `computeNumTokensPerBlock` (`:585`) walks `32, 64, ... 1024` and returns
/// 1024 when nothing fits, and `globalExpertPrefixSum` (`:764`) ladders the
/// same six values and uses 1024 for the strided form. So 32 warps is the
/// most any scan below folds, and 132 bytes of static shared memory covers
/// every launch. `dynamicSmemBytes` stays 0, exactly as upstream's five
/// `cudaLaunchConfig_t`s set it.
constexpr int kScanScratch = 1024 / 32 + 1;

/// Block-wide exclusive prefix sum of `x`, plus each thread's own value back,
/// through `scratch[kScanScratch]`.
///
/// `cub::BlockScan<int, BLOCK>::ExclusiveSum(x, excl)` by hand: every thread
/// gets its own `excl`. `attn/page_compact.cuh::block_exclusive_sum_u32` is
/// the same routine over `u32` and states the argument for why a rewrite of
/// an integer scan needs no tolerance; this is its `i32` twin, kept separate
/// rather than shared because the two live in different units and the carried
/// set has no notion of a common header below `pie_device.cuh`.
///
/// # The width is a run-time value here and was a template parameter upstream
///
/// That difference is a CONSEQUENCE of cub being gone, not an independent
/// choice, and it is the reason this file states four rows where a faithful
/// port would have stated fourteen. `cub::BlockScan` takes its width as a
/// template parameter because the width sizes its `TempStorage` type, so
/// upstream had to ladder six instantiations of three kernels by hand
/// (`:664-676`, `:781-797`) and select among them with a host `if` chain over
/// function pointers. Nothing here needs a compile-time width: the scratch is
/// sized for the maximum, which is 132 bytes, and the warp count is
/// `blockDim.x / 32`. The ladder disappears and the launch width becomes what
/// it should always have been -- a number on the `Launch`, not a name in the
/// symbol.
///
/// **`blockDim.x` must be a multiple of 32**, and is at every call site:
/// `computeNumTokensPerBlock` returns a power of two in `[32, 1024]`, and the
/// global-scan launcher sets `blockDim` from the same six values. This is
/// stricter than it looks -- the `__shfl_up_sync` below passes a FULL mask,
/// so a partial last warp is undefined behaviour rather than a wrong answer.
/// cub carries the identical requirement for the identical reason; upstream
/// simply spelled it in a template argument where this spells it in a launch.
///
/// The trailing `__syncthreads()` lets a caller reuse `scratch`; none does
/// today, and it is cheaper than the note explaining why a future one may
/// not.
///
/// Signed rather than unsigned because upstream's counts are `int` and the
/// values are non-negative by construction (a count of matches, at most the
/// block width), so the sum cannot reach the sign bit at any width here. The
/// spelling follows the source it replaces rather than the type it could have
/// used.
__device__ __forceinline__ i32 block_exclusive_sum_i32(i32 x, i32* scratch) {
    const int warps = static_cast<int>(blockDim.x) >> 5;
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int warp = static_cast<int>(threadIdx.x) >> 5;

    i32 inclusive = x;
    for (int off = 1; off < 32; off <<= 1) {
        const i32 up = __shfl_up_sync(0xffffffffu, inclusive, off);
        if (lane >= off) inclusive += up;
    }
    if (lane == 31) scratch[warp] = inclusive;
    __syncthreads();
    if (threadIdx.x == 0) {
        i32 running = 0;
        for (int w = 0; w < warps; ++w) {
            const i32 t = scratch[w];
            scratch[w] = running;
            running += t;
        }
        scratch[warps] = running;
    }
    __syncthreads();
    const i32 excl = inclusive - x + scratch[warp];
    __syncthreads();
    return excl;
}

/// Phase one: per (expert, token-block) counts, and the rows that produced
/// them.
///
/// Upstream `blockExpertPrefixSumKernel` (`:597-645`). One block per
/// (expert, token-block) pair -- `gridDim.x` is `num_experts_per_node`,
/// `gridDim.y` is `num_blocks_per_seq` -- and one thread per token in the
/// block's slice.
///
/// Each thread scans its token's `num_experts_per_token` choices for
/// `blockIdx.x` and, on a hit, records the *expanded* row index
/// `i * num_tokens + token_id`, which is the encoding the rest of the
/// pipeline uses: expanded row `r` came from unpermuted token
/// `r % num_tokens` under its `r / num_tokens`-th choice. The `break` makes a
/// duplicate choice of the same expert count once, which is upstream's
/// behaviour and not an accident -- a router that emits an expert twice for
/// one token contributes one row.
///
/// `blocked_row_to_unpermuted_row` is written at
/// `expert * num_tokens + block_base + rank`, so it is expert-major with a
/// full `num_tokens` stride per expert and is deliberately sparse: only the
/// first `count` entries of each block's slice are live, and phase three
/// reads exactly that many.
///
/// `blockDim.x` is the tokens-per-block width and must be a multiple of 32;
/// `block_exclusive_sum_i32` says why, and
/// `fire::flashinfer_moe::tokens_per_block` is the Rust that picks it.
/// Upstream made it a template parameter and laddered six instantiations
/// (`:664-676`) because `cub::BlockScan` needed a compile-time width; it does
/// not need to be one, and the one place it was load-bearing -- the
/// `threadIdx.x == kTokensPerBlock - 1` guard that elects the count writer --
/// reads `blockDim.x - 1` and elects the same thread.
__global__ void block_expert_prefix_sum(i32 const* token_selected_experts,
                                        i32* blocked_expert_counts,
                                        i32* blocked_row_to_unpermuted_row,
                                        i64 const num_tokens,
                                        i64 const num_experts_per_token,
                                        i32 const start_expert_id) {
    __shared__ i32 scratch[kScanScratch];

    // `target_expert_id` and `expert_id` are both offset by `start_expert_id`,
    // so this is a node-local expert index throughout.
    const i32 target_expert_id = static_cast<i32>(blockIdx.x);
    const i32 block_id = static_cast<i32>(blockIdx.y);
    const i32 num_blocks_per_seq = static_cast<i32>(gridDim.y);
    const i64 tokens_per_block = static_cast<i64>(blockDim.x);
    const i64 token_id = static_cast<i64>(block_id) * tokens_per_block + threadIdx.x;

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    cudaGridDependencySynchronize();
#endif

    i32 expanded_token_id = -1;
    if (token_id < num_tokens) {
        for (i64 i = 0; i < num_experts_per_token; ++i) {
            const i32 expert_id =
                token_selected_experts[token_id * num_experts_per_token + i] - start_expert_id;
            if (expert_id == target_expert_id) {
                expanded_token_id = static_cast<i32>(i * num_tokens + token_id);
                break;
            }
        }
    }

    const i32 has_matched = expanded_token_id >= 0 ? 1 : 0;
    const i32 index = block_exclusive_sum_i32(has_matched, scratch);

    if (has_matched) {
        blocked_row_to_unpermuted_row[static_cast<i64>(target_expert_id) * num_tokens +
                                      static_cast<i64>(block_id) * tokens_per_block + index] =
            expanded_token_id;
    }
    if (threadIdx.x == blockDim.x - 1) {
        blocked_expert_counts[target_expert_id * num_blocks_per_seq + block_id] =
            index + has_matched;
    }

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    cudaTriggerProgrammaticLaunchCompletion();
#endif
}

/// Phase two, the one-element-per-thread form.
///
/// Upstream `globalExpertPrefixSumKernel` (`:730-762`). A single block of
/// `blockDim.x` threads, exclusive-scanning the whole `num_experts_per_node *
/// num_blocks_per_seq` count matrix, which is why the launcher picks the
/// smallest power-of-two width that covers it and falls to the `_large` form
/// above 1024.
///
/// Two arrays come out of one scan and they are not the same thing.
/// `blocked_expert_counts_cumsum` is the base for every (expert, block) cell
/// and phase three adds a thread's rank to it. `expert_first_token_offset` is
/// the *expert* boundary, so it takes only the cells where `threadIdx.x %
/// num_blocks_per_seq == 0` -- the first block of each expert -- and gains a
/// final `num_experts_per_node`-th entry from the last thread, which is the
/// total row count and the one element the grouped GEMM reads as its extent.
///
/// The last thread writes `cumsum + cnt`, an *inclusive* value, and it is the
/// only inclusive read in the file. `_large` below spells the same fact
/// differently because it carries a running total; the two must agree and
/// they do.
__global__ void global_expert_prefix_sum(i32 const* blocked_expert_counts,
                                         i32* blocked_expert_counts_cumsum,
                                         i64* expert_first_token_offset,
                                         i64 const num_experts_per_node,
                                         i64 const num_blocks_per_seq) {
    __shared__ i32 scratch[kScanScratch];

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    cudaGridDependencySynchronize();
#endif

    const i64 num_elements = num_experts_per_node * num_blocks_per_seq;
    const i32 cnt =
        static_cast<i64>(threadIdx.x) < num_elements ? blocked_expert_counts[threadIdx.x] : 0;
    const i32 cumsum = block_exclusive_sum_i32(cnt, scratch);

    if (static_cast<i64>(threadIdx.x) < num_elements) {
        blocked_expert_counts_cumsum[threadIdx.x] = cumsum;
        if (threadIdx.x % num_blocks_per_seq == 0) {
            expert_first_token_offset[threadIdx.x / num_blocks_per_seq] = cumsum;
        }
        if (static_cast<i64>(threadIdx.x) == num_elements - 1) {
            expert_first_token_offset[num_experts_per_node] = cumsum + cnt;
        }
    }

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    cudaTriggerProgrammaticLaunchCompletion();
#endif
}

/// Phase two, the strided form, for count matrices above 1024 elements.
///
/// Upstream `globalExpertPrefixSumLargeKernel` (`:683-728`). Each thread owns
/// `num_elem_per_thread` *contiguous* elements, sums them into one value,
/// takes part in a single block scan, and then walks its run again adding as
/// it goes.
///
/// Upstream's own comment on the second loop, verbatim, because it explains a
/// shape that otherwise reads as an oversight:
///
/// ```text
///   // Note: Because of limited registers, cannot store thread-level prefix
///   // sum or enable #pragma unroll
/// ```
///
/// So the elements are read twice from global memory rather than held. The
/// TODO beside it -- *"Fix uncoalesced access with shared memory"* -- is
/// upstream's and is preserved here as a standing question, unmeasured in
/// this tree: a contiguous-per-thread partition is exactly the layout that
/// does not coalesce, and a strided one would, at the cost of the running
/// total in the second loop.
///
/// The width is always 1024 -- the launcher does not ladder this form -- and
/// `num_elem_per_thread` is `ceil(num_elements / 1024)`. That constant is on
/// the `Launch` in Rust now rather than in the symbol; see
/// `block_exclusive_sum_i32` for why the ladder went.
__global__ void global_expert_prefix_sum_large(i32 const* blocked_expert_counts,
                                               i32* blocked_expert_counts_cumsum,
                                               i64* expert_first_token_offset,
                                               i64 const num_experts_per_node,
                                               i64 const num_blocks_per_seq,
                                               i64 const num_elem_per_thread) {
    __shared__ i32 scratch[kScanScratch];

    const i64 offset = static_cast<i64>(threadIdx.x) * num_elem_per_thread;
    const i64 num_elements = num_experts_per_node * num_blocks_per_seq;
    i32 cnt = 0;

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    cudaGridDependencySynchronize();
#endif

    for (i64 i = 0; i < num_elem_per_thread; ++i) {
        if (offset + i < num_elements) {
            cnt += blocked_expert_counts[offset + i];
        }
    }

    i32 cumsum = block_exclusive_sum_i32(cnt, scratch);

    for (i64 i = 0; i < num_elem_per_thread; ++i) {
        if (offset + i < num_elements) {
            blocked_expert_counts_cumsum[offset + i] = cumsum;
            if ((offset + i) % num_blocks_per_seq == 0) {
                expert_first_token_offset[(offset + i) / num_blocks_per_seq] = cumsum;
            }
            cumsum += blocked_expert_counts[offset + i];
            if ((offset + i) == num_elements - 1) {
                expert_first_token_offset[num_experts_per_node] = cumsum;
            }
        }
    }

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    cudaTriggerProgrammaticLaunchCompletion();
#endif
}

/// Phase three: scatter each surviving row to its permuted position, and fill
/// both permutation maps.
///
/// Upstream `mergeExpertPrefixSumKernel` (`:810-840`). The grid is phase
/// one's -- `(num_experts_per_node, num_blocks_per_seq)` -- and it is
/// launched at the same width, so `threadIdx.x` is a *rank within this
/// (expert, block) cell* rather than a token index. The guard is `threadIdx.x
/// < cnt`, which is why phase one's sparse `blocked_row_to_unpermuted_row`
/// slice needs no compaction: the live prefix is exactly `cnt` long.
///
/// `token_id` is `block_id * blockDim.x + threadIdx.x` and indexes phase
/// one's output, NOT the token array -- the name is upstream's and is kept so
/// the two files read as one, but it is the reason this kernel takes
/// `blockDim.x` off the launch instead of a template parameter: it has no
/// shared memory and no collective, so the width is free.
///
/// Three writes, and the third is the inverse map that
/// `finalizeMoeRoutingKernel` reads to un-permute. `permuted_row` is dense
/// and increasing within an expert, which is the ordering invariant the whole
/// front-end exists to establish.
__global__ void merge_expert_prefix_sum(i32 const* blocked_expert_counts,
                                        i32 const* blocked_expert_counts_cumsum,
                                        i32 const* blocked_row_to_unpermuted_row,
                                        i32* permuted_token_selected_experts,
                                        i32* permuted_row_to_unpermuted_row,
                                        i32* unpermuted_row_to_permuted_row,
                                        i32 const num_tokens) {
    const i32 target_expert_id = static_cast<i32>(blockIdx.x);
    const i32 block_id = static_cast<i32>(blockIdx.y);
    const i32 num_blocks_per_seq = static_cast<i32>(gridDim.y);
    const i32 token_id = block_id * static_cast<i32>(blockDim.x) + static_cast<i32>(threadIdx.x);

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    cudaGridDependencySynchronize();
#endif

    const i32 cnt = blocked_expert_counts[target_expert_id * num_blocks_per_seq + block_id];
    const i32 offset =
        blocked_expert_counts_cumsum[target_expert_id * num_blocks_per_seq + block_id];
    if (static_cast<i32>(threadIdx.x) < cnt) {
        const i32 unpermuted_row =
            blocked_row_to_unpermuted_row[target_expert_id * num_tokens + token_id];
        const i32 permuted_row = offset + static_cast<i32>(threadIdx.x);
        permuted_row_to_unpermuted_row[permuted_row] = unpermuted_row;
        permuted_token_selected_experts[permuted_row] = target_expert_id;
        unpermuted_row_to_permuted_row[unpermuted_row] = permuted_row;
    }

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    cudaTriggerProgrammaticLaunchCompletion();
#endif
}

}  // namespace pie::moe
