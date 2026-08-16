//===-- gemv.cuh - the M=1 bf16 GEMV, as `__global__` templates ---*- CUDA -*-===//
//
// TWO `__global__` templates and nothing else. No host function, no `<<<>>>`,
// no stream, no `cudaDeviceGetAttribute`. `gemm/gemv.cu` and `gemm/gemv.hpp`
// are DELETED, not split: everything above the launches was a host decision,
// and every host decision is now Rust in `driver-cuda`'s `fire/gemv.rs`.
//
// This is a MOVE and not a copy, which is the distinction
// `tests/device_sources.rs::no_global_is_defined_twice` exists to
// enforce: there is exactly one definition of each of these two templates in
// the tree, and it is here. `norm/altup_aux` shipped a release with two
// definitions of six kernels; they agreed the day they were written and
// drifted after, each right for whichever half of the tests exercised it.
//
// # What moved to Rust, and where each piece landed
//
// Every one of these landed in `crates/driver-cuda/src/fire/gemv.rs`, which
// is where host code lives — in Rust, permanently. This crate holds device
// text and the tables that describe it and nothing else; it must contain no
// `.cu` at all, which is why this file is a `.cuh`: a `.cu` is a translation
// unit for nvcc, and a `.cuh` is text carried into NVRTC at run time. The
// specification of the host program as a whole — units fired, buffers between
// them, host decisions, and what `Source`/`LaunchRule`/`Specialisation`/
// `Execution` cannot yet state — is in `src/families/gemm.rs`'s module header
// and restated against the code in `fire/gemv.rs`.
//
// `gemv_bf16` was one host launcher over four launches. Its parts:
//
//   `gemv_unroll_depth()`     a `cudaDevAttrComputeCapabilityMajor` read,
//                             cached in a function-local `static`, answering
//                             2 on sm_100+ and 4 below. DEVICE-SPECIFIC
//                             TUNING, so Rust by the owner's principle:
//                             `fire::gemv::unroll_depth`, which asks
//                             `driver_cuda::device::Device::compute_capability`
//                             rather than opening a second way to ask.
//   `N <= kSplitKMaxRows`     a shape threshold, `constexpr int` 4096 since
//                             §36 folded the `getenv` away at its unchanged
//                             default. `fire::gemv::SPLIT_K_MAX_ROWS`.
//   `K % 8`, `aligned16(..)`  the REFUSAL. `false` meant "I did not launch --
//                             use cuBLAS", and it still does:
//                             `fire::gemv::Gemv::Declined`.
//   `blocks = ceil(N/4)`      grid arithmetic, and its `> 2147483647` refusal.
//
// # The four `<<<>>>` these kernels were launched by
//
// Recorded VERBATIM because the file that held them is deleted and every
// `Launch` in `fire/gemv.rs` cites one of these lines. `gemm/gemv.cu` at
// deletion, in launcher order:
//
//   :344  gemv_splitk_bf16_kernel<kSplitWarpsB, /*kUnrollP=*/2>
//   :345      <<<dim3(static_cast<unsigned>(N)), dim3(32, kSplitWarpsB), 0,
//   :346         stream>>>(
//
//   :355  gemv_splitk_bf16_kernel<kSplitWarps>
//   :356      <<<dim3(static_cast<unsigned>(N)), dim3(32, kSplitWarps), 0,
//   :357         stream>>>(
//
//   :372  gemv_bf16_kernel<kWarps, 2>
//   :373      <<<dim3(static_cast<unsigned>(blocks)), dim3(32, kWarps), 0,
//   :374         stream>>>(
//
//   :382  gemv_bf16_kernel<kWarps, 4>
//   :383      <<<dim3(static_cast<unsigned>(blocks)), dim3(32, kWarps), 0, stream>>>(
//
// with `kWarps = kSplitWarpsB = 4`, `kSplitWarps = 8` and
// `blocks = (N + kWarps - 1) / kWarps`. Every block is `dim3(32, kWarps)` --
// a warp per row, `kWarps` rows per block -- which is a TWO-DIMENSIONAL block
// and the reason no `kernels::LaunchRule` states these launches. All four rows
// in `kernels_cuda::families::gemm` are `LaunchRule::Unstated` and the
// driver builds the `Launch` by hand, exactly as `attn_score_fold_heads`
// already does for its literal `gridDim.y`.
//
// # The one thing that is not byte-for-byte what `gemv.cu` held
//
// The kernels are in `pie::gemm` now, where
// `gemv.cu` had them in an anonymous namespace inside
// `pie::gemm`. From inside a namespace called `device`,
// the spelling `bf16` looks up `device` in the enclosing scope, finds
// THIS namespace, and fails to find `bf16` in it. So the prelude's names are
// brought in by `using` and used unqualified, which is the idiom
// `rope/rope.cuh` records at its own `using` block. That is a requalification
// and not an edit: no arithmetic, no order, no type and no rounding changed,
// and the `float4` walk, the shuffle tree and the double-rounded epilogue are
// the text `gemv.cu` shipped.
//
//===----------------------------------------------------------------------===//
#pragma once

// The scalar layer and the fixed-width integer names. What used to be
// `pie_device.cuh` plus `<cstdint>` -- and `<cstdint>` is gone with the host
// half that wanted it: its one use was `std::uintptr_t` in `aligned16`, which
// is a HOST alignment test made before any launch and is Rust now.
#include "prelude/device.cuh"

namespace pie::gemm {

// The scalar layer is the PRELUDE's, not this family's -- see the header's
// last section for why these have to be declared rather than spelled
// `bf16` inside a namespace of that name.

// One warp per output row. Each lane walks the row in float4 strides (8
// bf16 = 16 B, so a warp step covers 512 B — four full cache lines) and
// accumulates in fp32; a single shuffle tree finishes the dot product.
//
// The walk is unrolled by `kUnroll` and the loads are hoisted above the math
// that consumes them. That is the whole reason for the unroll: written the
// obvious way each lane has exactly ONE load in flight, because the FMA on
// `w4[i]` is the next instruction after it, and a warp that is waiting on a
// single HBM round trip cannot cover the latency no matter how many warps the
// SM holds. Measured on an H100 at gpt-oss's o_proj shape (N=2880, K=4096,
// 23.6 MB per layer) the one-at-a-time version sustained about 963 GB/s.
//
// WHICH `kUnrollP` a fire instantiates is a device fact and therefore not this
// file's: `fire::gemv::unroll_depth` reads the compute capability and picks
// between the `<4, 2>` and `<4, 4>` rows, and carries the measurement that
// decides it.
template <int kWarps, int kUnrollP = 4>
__global__ void gemv_bf16_kernel(
    const bf16* __restrict__ weight,
    const bf16* __restrict__ act,
    const bf16* __restrict__ bias,
    bf16* __restrict__ out,
    int N, int K, float beta)
{
    const int row = blockIdx.x * kWarps + threadIdx.y;
    if (row >= N) return;
    const float4* w4 =
        reinterpret_cast<const float4*>(weight + (long long)row * K);
    const float4* x4 = reinterpret_cast<const float4*>(act);
    const int vectors = K / 8;
    constexpr int kUnroll = kUnrollP;
    float acc = 0.f;
    int i = threadIdx.x;
    for (; i + 32 * (kUnroll - 1) < vectors; i += 32 * kUnroll) {
        float4 wv[kUnroll];
        float4 xv[kUnroll];
        #pragma unroll
        for (int u = 0; u < kUnroll; ++u) {
            wv[u] = w4[i + 32 * u];
            xv[u] = x4[i + 32 * u];
        }
        #pragma unroll
        for (int u = 0; u < kUnroll; ++u) {
            const bf16* wb = reinterpret_cast<const bf16*>(&wv[u]);
            const bf16* xb = reinterpret_cast<const bf16*>(&xv[u]);
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                acc += bf16_to_f32(wb[j]) * bf16_to_f32(xb[j]);
            }
        }
    }
    for (; i < vectors; i += 32) {
        float4 wv = w4[i];
        float4 xv = x4[i];
        const bf16* wb = reinterpret_cast<const bf16*>(&wv);
        const bf16* xb = reinterpret_cast<const bf16*>(&xv);
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            acc += bf16_to_f32(wb[j]) * bf16_to_f32(xb[j]);
        }
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        acc += __shfl_down_sync(0xffffffffu, acc, off);
    }
    if (threadIdx.x == 0) {
        if (beta != 0.f) acc += beta * bf16_to_f32(out[row]);
        // Round to bf16 *before* adding the bias, then round again. That is
        // a redundant-looking double rounding, and it is deliberate: it is
        // exactly what the separate `add_bias_bf16_kernel` did when it read
        // this kernel's bf16 output back. Folding the two launches into one
        // is a launch-count optimization, not an arithmetic change, so it
        // has to stay bit-identical or it stops being free to validate.
        bf16 v = f32_to_bf16(acc);
        if (bias != nullptr) {
            v = f32_to_bf16(bf16_to_f32(v) + bf16_to_f32(bias[row]));
        }
        out[row] = v;
    }
}

// One BLOCK per output row, its warps splitting K between them and reducing
// through shared memory.
//
// The kernel above gives one warp to each row, so the grid is N/4 blocks and
// the machine is only busy if N is large. gpt-oss decodes through three shapes
// where it is not: k_proj and v_proj are N=512, which is 128 blocks on 132 SMs,
// and the MoE router is N=32, which is EIGHT. vLLM runs the same projections
// through cuBLAS's `splitK` nvjet kernels at 2.3-2.5 TB/s for exactly this
// reason. Splitting inside the block keeps it to one launch and needs no
// scratch, which the reduce-across-blocks form would.
//
// WHERE the crossover is -- `N <= 4096` -- is a host decision and is
// `fire::gemv::SPLIT_K_MAX_ROWS`, which carries the L40S table that measured
// it and the reason the number was not moved.

// MEASURED DEAD END, recorded because the reasoning looks sound and someone
// will try it again: staging the activation in shared memory before the dot
// product is SLOWER. Every block reads the same activation vector, so the
// loop below issues two global loads per iteration -- one weight, one
// activation -- and repeats the activation read for all N blocks, which looks
// like an obvious thing to hoist. It is not: those re-reads are already
// served by L2, and paying a __syncthreads plus a shared-memory round trip to
// avoid them costs more than they do. Measured cold on B200 (bf16, one token):
// gpt-oss q_proj 10.3 -> 13.7 us, o_proj 8.3 -> 11.5, gemma-4-31B kv_proj
// 12.5 -> 16.4. Results were bit-identical, so this is purely a loss.
template <int kWarps, int kUnrollP = 1>
__global__ void gemv_splitk_bf16_kernel(
    const bf16* __restrict__ weight,
    const bf16* __restrict__ act,
    const bf16* __restrict__ bias,
    bf16* __restrict__ out,
    int N, int K, float beta)
{
    const int row = blockIdx.x;
    if (row >= N) return;
    const int warp = threadIdx.y;
    const float4* w4 =
        reinterpret_cast<const float4*>(weight + (long long)row * K);
    const float4* x4 = reinterpret_cast<const float4*>(act);
    const int vectors = K / 8;

    float acc = 0.f;
    // Warp `warp` walks a strided share of the row, so the whole block still
    // reads it in the same coalesced order the one-warp kernel does.
    // `kUnrollP` hoists loads above the math for the same reason the
    // row-per-warp kernel does it: written flat, each lane has ONE load in
    // flight. Default 1 keeps the shipping code byte-identical.
    constexpr int kU = kUnrollP;
    const int stride = kWarps * 32;
    int i = warp * 32 + threadIdx.x;
    for (; i + stride * (kU - 1) < vectors; i += stride * kU) {
        float4 wv[kU];
        float4 xv[kU];
        #pragma unroll
        for (int u = 0; u < kU; ++u) {
            wv[u] = w4[i + stride * u];
            xv[u] = x4[i + stride * u];
        }
        #pragma unroll
        for (int u = 0; u < kU; ++u) {
            const bf16* wb = reinterpret_cast<const bf16*>(&wv[u]);
            const bf16* xb = reinterpret_cast<const bf16*>(&xv[u]);
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                acc += bf16_to_f32(wb[j]) * bf16_to_f32(xb[j]);
            }
        }
    }
    for (; i < vectors; i += stride) {
        float4 wv = w4[i];
        float4 xv = x4[i];
        const bf16* wb = reinterpret_cast<const bf16*>(&wv);
        const bf16* xb = reinterpret_cast<const bf16*>(&xv);
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            acc += bf16_to_f32(wb[j]) * bf16_to_f32(xb[j]);
        }
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        acc += __shfl_down_sync(0xffffffffu, acc, off);
    }

    __shared__ float partial[kWarps];
    if (threadIdx.x == 0) partial[warp] = acc;
    __syncthreads();
    if (warp != 0 || threadIdx.x != 0) return;
    float total = 0.f;
    #pragma unroll
    for (int w = 0; w < kWarps; ++w) total += partial[w];
    if (beta != 0.f) total += beta * bf16_to_f32(out[row]);
    // Same double rounding as the kernel above, for the same reason: it is
    // what the separate bias kernel used to do, so the fold stays bit-exact.
    bf16 v = f32_to_bf16(total);
    if (bias != nullptr) {
        v = f32_to_bf16(bf16_to_f32(v) + bf16_to_f32(bias[row]));
    }
    out[row] = v;
}

}  // namespace pie::gemm
