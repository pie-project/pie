//===-- moe_grouped_gemm.cuh - the short-K grouped GEMM -------------------===//
//
// One `__global__` template, over the WMMA shape `pie_mma.cuh` implements.
// No launcher: `moe_grouped_gemm.cu` includes this file and keeps the
// support predicate and the `<<<>>>`, so the kernel has exactly ONE
// definition that nvcc and NVRTC both read.
//
// # This file is why the wmma shim exists
//
// `pie_mma.cuh`'s header names two call sites in the whole tree and this is
// one of them. NVRTC 13.0 on an L40S refused `mma.h` — *"could not open
// source file 'mma.h' (no directories in search list)"* — so a `wmma`
// fragment could not be spelled under the JIT at all, and this family was
// blocked on that until the shim was written and proved bit-identical to
// `nvcuda::wmma` on four cases with a transposed-store control caught at
// max |Δ| 4.17.
//
// The shape asked for here is the shape the shim implements, exactly:
// 16×16×16, bf16 × bf16 → f32, A `row_major`, B `col_major`, store
// `mem_row_major`. `kFrag` is 16 and always was.
//
// # Why the include is conditional, now that both files are in one tree
//
// `pie_mma.cuh` is beside this one — the `.cuh` files moved into this crate's
// `kernels/` — and the archive crate's `kernels-cuda/csrc/CMakeLists.txt`
// reaches this directory with `-Xcompiler=-iquote`, so nvcc CAN resolve
// `#include "prelude/mma.cuh"` now. It still must not: `-iquote` answers our own
// quoted spellings and leaves `#include <mma.h>` to the toolkit, and taking
// the shim ahead of time would swap NVIDIA's `nvcuda::wmma` out of the
// archive for a reimplementation of one fragment shape — a change to what
// the archive computes, made by a build flag, which is not a thing an
// include path should be able to do.
//
// So the branch stays, and it is now a statement of WHICH IMPLEMENTATION
// each compiler gets rather than which one it can find: NVRTC 13.0 refused
// `mma.h` outright, so it gets the shim; nvcc has the real header and keeps
// it. Both spellings produce the same fragment types because the shim's
// `__nv_bfloat16` typedef names the prelude's `bf16`, which is what
// the pointer casts below go through — measured bit-identical on an L40S.
//
// # The element type is a template parameter and bf16 is the only one
//
// The row states an element and the kernel checks it: the shim implements
// bf16 fragments and nothing else, and an f16 instantiation would otherwise
// reinterpret f16 bits through a bf16 lane map — a wrong answer that
// compiles. The `static_assert` turns that into a message naming this file.
//
//===----------------------------------------------------------------------===//
#pragma once

#include "prelude/device.cuh"

#ifdef __CUDACC_RTC__
// NVRTC: the shim, which resolves and typedefs `__nv_bfloat16` to the
// prelude's `bf16`.
#include "prelude/mma.cuh"
#else
// nvcc: the vendor's headers, which are on the ahead-of-time include path and
// are what the shim was written against.
#include <cuda_bf16.h>
#include <mma.h>
#endif

namespace pie::moe {


namespace wmma = ::nvcuda::wmma;

/// The WMMA tile, and the aligned block's row count.
constexpr int kFrag = 16;
/// Warps per block, one n-fragment each.
constexpr int kGemmWarps = 4;
/// Columns of C one block covers.
constexpr int kNTile = kFrag * kGemmWarps;

/// C = A @ W^T for a batch of `kFrag`-row blocks, skipping padding blocks.
///
/// A is [M, K] row-major, so it loads as a row_major matrix_a. W is [N, K]
/// row-major, and W^T is [K, N]; a [K, N] column-major view of W^T is exactly
/// W's own memory with leading dimension K, so the b-fragment needs no
/// staging pass -- it is col_major with ld = K.
///
/// That costs coalescing: a fragment is 16 rows of 32 bytes at a stride of
/// 2*K, so a quarter of each cache line is used. It is affordable only while
/// K is short, which is why `moe_grouped_gemm_bf16_supported` bounds K.
///
/// The long-K case was pursued across seven kernels and cuBLAS keeps it.
/// On Qwen3.6's gate_up (M=16, N=512, K=2048), against cuBLAS's 10.57 ms:
///   direct (this kernel)                              14.60
///   + both operands staged through shared memory      13.71
///   + cp.async double buffering                       12.75
///   + 4 stages at kChunk=32                           11.98
///   + 4 n-fragments per warp sharing one a-fragment   11.22
///   + narrower tile for 4x the CTAs                   11.17
///   + kChunk=64 so each row read is a full cache line 11.13
///
/// The plateau is the point. Skipping the padding blocks removes about 65%
/// of the *batch entries* but almost none of the DRAM traffic: roughly 106
/// of 256 experts are live at 128 rows, so the unique weight bytes are
/// ~212 MB per layer either way, and the padding entries were already being
/// served from L2 (they repeat live experts, and inactive ones collapse onto
/// expert 0). Both kernels end up streaming the same bytes at ~780 GB/s.
///
/// That also corrects the clamp probe that motivated this: clamping the
/// batch count to 128 measured 6.78 ms, but it cut the unique bytes too, so
/// it was never an achievable target for a correct kernel.
///
/// Where the early exit does pay is a short K, because there the per-entry
/// fixed cost is large relative to the mainloop -- which is exactly where
/// this kernel is used.
template <class T>
__global__ __launch_bounds__(kGemmWarps * 32) void moe_grouped_gemm(
    const T* __restrict__ a,
    const T* __restrict__ weight_base,
    T* __restrict__ c,
    const i32* __restrict__ expert_ids,
    int N,
    int K)
{
    static_assert(is_same<T, bf16>::value,
                  "pie_mma.cuh implements bf16 fragments only -- see its "
                  "static_assert, and do not extend it without a parity run");

    const int b = blockIdx.y;
    const int e = expert_ids[b];
    if (e < 0) return;  // padding block: the whole point of this kernel

    const int warp = static_cast<int>(threadIdx.x) >> 5;
    const int n_warp = blockIdx.x * kNTile + warp * kFrag;
    // The fragment element type is the shim's (NVRTC) or the vendor's (nvcc);
    // both are 16 bits with the prelude's layout, and the cast is what lets
    // one body serve both. See this file's header.
    const auto* a_row = reinterpret_cast<const __nv_bfloat16*>(a) +
        static_cast<long long>(b) * kFrag * K;
    const auto* w = reinterpret_cast<const __nv_bfloat16*>(weight_base) +
        static_cast<long long>(e) * N * K + static_cast<long long>(n_warp) * K;

    wmma::fragment<wmma::accumulator, kFrag, kFrag, kFrag, float> acc;
    wmma::fill_fragment(acc, 0.f);
    wmma::fragment<wmma::matrix_a, kFrag, kFrag, kFrag,
                   __nv_bfloat16, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, kFrag, kFrag, kFrag,
                   __nv_bfloat16, wmma::col_major> b_frag;
    for (int k = 0; k < K; k += kFrag) {
        wmma::load_matrix_sync(a_frag, a_row + k, K);
        wmma::load_matrix_sync(b_frag, w + k, K);
        wmma::mma_sync(acc, a_frag, b_frag, acc);
    }

    // Round to bf16 on the way out; the accumulator stayed fp32, matching
    // cuBLAS's compute type for this call.
    __shared__ float staged[kGemmWarps][kFrag * kFrag];
    wmma::store_matrix_sync(staged[warp], acc, kFrag, wmma::mem_row_major);
    __syncwarp();
    T* c_row = c + static_cast<long long>(b) * kFrag * N + n_warp;
    for (int idx = static_cast<int>(threadIdx.x) & 31; idx < kFrag * kFrag;
         idx += 32) {
        c_row[static_cast<long long>(idx / kFrag) * N + (idx % kFrag)] =
            Elem<T>::from_f32(staged[warp][idx]);
    }
}

}  // namespace pie::moe
