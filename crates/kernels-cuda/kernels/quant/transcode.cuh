//===-- transcode.cuh - the fused quant->quant kernel, as device text ----===//
//
// One `__global__` template, two Decode functors and one Encode functor.
// `transcode.cu` was the dispatch that bound a (source, target) pair to an
// instantiation; this file is what the GPU runs. It was already a header
// before this migration -- the only file in the family that was -- which is
// why the split had nothing to do here and the conversion had everything.
//
// `2ef431d02` deleted that dispatch, and the two pairs it bound are now two
// `fn`s: `driver_internal::{transcode_bf16_to_mxfp4,
// transcode_fp8_e4m3_per_group_to_mxfp4}`, over the root `src/quant.rs`
// declares for this file. Each names one template-id, which is the dispatch's
// `switch` unrolled into the type system -- a driver-internal caller knows
// its source format at compile time, so the arm it took at run time is a
// choice the Rust compiler can make instead.
//
// # Why it exists at all
//
// The loader's quant->quant transcode (an FP8 checkpoint into an MXFP4
// runtime quant, say) is a single IR `Transcode` node, but the executor
// lowered it to two device kernels through a BF16 scratch buffer in HBM:
// dequant into BF16, then encode out of it. That round trip triples the HBM
// traffic for a value nothing else ever reads.
//
// Hand-writing one fused kernel per (source, target) pair is an `a x B`
// explosion. This composes `a` Decode functors with `B` Encode functors in
// one kernel template instead, and the compiler emits the specialisations.
// The intermediate lives in REGISTERS and never reaches HBM: the narrow waist
// moved from "BF16 in global memory" to "float[32] in registers", which is
// the same composition contract the two-step relied on, minus the traffic.
//
// The Decode functors round through BF16 deliberately, so a fused transcode
// is BIT-IDENTICAL to the two-step it replaces -- that is what preserves the
// validated model numerics, and it is what let `tests/test_transcode_fused.cu`
// assert exact equality rather than a tolerance. That test was
// `driver-cuda/csrc/tests/test_transcode_fused.cu` and went with that tree at
// `4569b9e4b`; nothing has replaced it. So the bit-identity is an argument
// about the arithmetic below and is checked by nothing today -- which is
// worth knowing before the fusion bit is turned back on.
//
// # No rows, and exactly why -- one reason, where there were two
//
// This file carries no `DeviceKernel` rows and is not a `Unit`. Two
// independent reasons were recorded, and only the first still holds:
//
//   * `transcode_rowmajor_kernel<GROUP, Decode, Encode>` has THREE template
//     parameters, one of them an `int`. `DeviceKernel::instantiation` spells
//     `path<elem>` with a single type path, so no row can name it. Note what
//     that is an argument about: the row world's SPELLING. A `mod inst`
//     writes the whole template-id out, so `quant::transcode::inst` names both
//     pairs and NVRTC lowers them.
//   * its operands are the Decode and Encode functors BY VALUE -- aggregates
//     of pointers and extents -- and *"`kernels::Ty` has no kind for a
//     by-value aggregate, and `runtime::args` marshals pointers, `I32`,
//     `U32`, `F32` and `Usize` and refuses everything else"*. THAT IS NO
//     LONGER TRUE. `jit::ArgValue::Bytes` carries an aggregate whole, and
//     `by_value!` declares a Rust mirror of one with its `sizeof`, `alignof`
//     and every `offsetof` asserted against a measurement -- in Rust by the
//     macro, in C++ by `tests/typecheck_tu.rs` over this very text. The three
//     mirrors are in `src/quant.rs` and the numbers came out of NVRTC's PTX.
//
// So what keeps these kernels out of a `Family` is neither of the two: it is
// that no `Source` produces a Decode or an Encode. They are assembled from a
// loader plan's tile facts, and a trace statement has no way to name one.
//
// Its conversion off `<cstdint>` was never conditional on any of that. Every
// `.cuh` under `kernels/` is carried into `DEVICE_HEADERS` by
// `kernels-cuda/build.rs`, so a file that reaches for a standard header
// is a compile error waiting for whichever root first includes it. This one
// includes `quant/quant_bf16_to_mxfp4.cuh` rather than the other way about,
// and until the root above existed nothing included IT -- which is exactly
// how a carried file with a `<cstdint>` in it would have gone unnoticed.
//
// # The duplicate that is gone
//
// `encode_fp4_e2m1` and `encode_e8m0` used to be copied into this file from
// `quant_bf16_to_mxfp4.cu`, with a comment saying so. That is precisely the
// failure this migration exists to remove: two E2M1 rounding ladders that
// agree today and drift the first time one of them learns about NVFP4. They
// are now included from `quant/quant_bf16_to_mxfp4.cuh`, which is where the
// kernel that writes the checkpoint format keeps them, and there is one
// boundary table in the tree.
//
// # NVRTC, and what left with it
//
// `<cstdint>` and `<math.h>` are gone: NVRTC ships no C++ standard library --
// 0 of 31 standard headers answered when it was measured -- and it does not
// need to, since `fabsf`, `log2f`, `ceilf` and `ldexpf` are device builtins
// under both compilers and the prelude names the fixed-width integers.
// `<cuda_bf16.h>` is gone because the prelude carries `bf16` and both exact
// widenings. `<cuda_fp8.h>` stays, guarded: `__nv_cvt_fp8_to_halfraw` is a
// hardware conversion and the shim reproduces it bit-identically over all 256
// E4M3 byte patterns, so the intrinsic keeps its spelling and only the
// include is chosen per compiler.
//
//===----------------------------------------------------------------------===//
#pragma once

#include "prelude/device.cuh"
#include "quant/quant_bf16_to_mxfp4.cuh"

#ifdef __CUDACC_RTC__
#include "prelude/fp8.cuh"
#else
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#endif

namespace pie::transcode {

// E8M0 stores a biased power-of-two exponent: value = 2^(byte - kE8M0Bias).
inline constexpr int kE8M0Bias = 127;

// The MXFP4 encoders, from the kernel that writes the same format.
//
// `quant/quant_bf16_to_mxfp4.cuh` holds the E2M1 boundary ladder and the
// E8M0 exponent rule; a second copy here is a second rounding rule with
// nobody to notice when they diverge.
using ::pie::quant::encode_e8m0;
using ::pie::quant::encode_fp4_e2m1;


// ---- Decode functors: source element -> float (rounded through BF16) -------

// Raw BF16 source (no source scale).
struct DecodeBf16 {
    const bf16* __restrict__ src;
    int cols;
    __device__ __forceinline__ float load(int row, int col) const
    {
        return bf16_to_f32(src[static_cast<usize>(row) * cols + col]);
    }
};

// FP8 E4M3 with a per-group (block) FP32 scale, matching
// dequant_fp8_e4m3_per_group: scale index = [row/gs][col/gs]. The value is
// rounded through BF16 so the result matches the FP8->BF16->MXFP4 two-step.
struct DecodeFp8E4m3PerGroup {
    const u8* __restrict__ src;
    const float* __restrict__ scales;
    int cols;
    int scale_cols;
    int group_size;
    __device__ __forceinline__ float load(int row, int col) const
    {
        const int sr = row / group_size;
        const int sc = col / group_size;
        const float s = scales[static_cast<usize>(sr) * scale_cols + sc];
        const __half h = __nv_cvt_fp8_to_halfraw(
            src[static_cast<usize>(row) * cols + col], __NV_E4M3);
        return bf16_to_f32(f32_to_bf16(__half2float(h) * s));
    }
};

// ---- Encode functor: 32 floats -> E8M0 scale byte + 16 packed E2M1 bytes ---
// Each Encode advertises its group width as `kGroup`; the dispatch instantiates
// the kernel template with it, so encode_group's array size always matches.
struct EncodeMxfp4 {
    static constexpr int kGroup = 32;        // values per E8M0 scale block
    static constexpr int kPackedPerByte = 2;  // E2M1 nibbles packed per output byte
    static constexpr int kBytesPerGroup = kGroup / kPackedPerByte;  // 16
    u8* __restrict__ packed;  // [rows, cols/kPackedPerByte]
    u8* __restrict__ scales;  // [rows, cols/kGroup]
    int cols;
    __device__ __forceinline__ void encode_group(
        const float (&vals)[kGroup], int row, int g) const
    {
        float absmax = 0.0f;
        #pragma unroll
        for (int k = 0; k < kGroup; ++k) {
            const float a = fabsf(vals[k]);
            if (a > absmax) absmax = a;
        }
        const u8 sb = encode_e8m0(absmax);
        scales[static_cast<usize>(row) * (cols / kGroup) + g] = sb;
        const float s = ldexpf(1.0f, static_cast<int>(sb) - kE8M0Bias);
        const float inv_s = (s == 0.0f) ? 0.0f : (1.0f / s);
        const usize po = static_cast<usize>(row) * (cols / kPackedPerByte)
            + static_cast<usize>(g) * kBytesPerGroup;
        #pragma unroll
        for (int k = 0; k < kBytesPerGroup; ++k) {
            const unsigned lo = encode_fp4_e2m1(vals[kPackedPerByte * k]     * inv_s);
            const unsigned hi = encode_fp4_e2m1(vals[kPackedPerByte * k + 1] * inv_s);
            packed[po + k] = static_cast<u8>((hi << 4) | (lo & 0xFu));
        }
    }
};

// ---- Composable kernel: Decode -> register group of GROUP -> Encode --------
// One block per row; threads stride over the GROUP-wide blocks. The intermediate
// float[GROUP] never leaves registers. GROUP is the target's group width
// (Encode::kGroup) — 32 for MXFP4, 16 for NVFP4, etc.
template <int GROUP, typename Decode, typename Encode>
__global__ void transcode_rowmajor_kernel(Decode dec, Encode enc, int cols)
{
    const int row = blockIdx.x;
    const int groups = cols / GROUP;
    for (int g = threadIdx.x; g < groups; g += blockDim.x) {
        const int base = g * GROUP;
        float vals[GROUP];
        #pragma unroll
        for (int k = 0; k < GROUP; ++k) {
            vals[k] = dec.load(row, base + k);
        }
        enc.encode_group(vals, row, g);
    }
}

}  // namespace pie::transcode
