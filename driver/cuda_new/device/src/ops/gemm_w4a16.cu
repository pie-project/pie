#include "gemm_w4a16.cuh"

// ---------------------------------------------------------------------------
// Fused W4A16 GEMM — attempted on CUTLASS 3.x Hopper "mixed input" GEMM, with a
// working CUTLASS-tensor-core fallback. Read the header for the data format.
//
// What we tried (the intended fully-fused path)
// ---------------------------------------------------------------------------
// The goal was the SM90 mixed-input warp-specialized GEMM (the kernel behind
// examples/55_hopper_mixed_dtype_gemm): int4 weight kept packed, loaded into
// registers, dequantized in-flight (de-scaled by the group scale) just before
// the GMMA. We assembled it with
//   cutlass::gemm::collective::CollectiveBuilder<arch::Sm90, OpClassTensorOp,
//       ElementA = {int4b_t, bfloat16_t} /* weight + group scale */, RowMajor,
//       ElementB = bfloat16_t            /* activations */,          RowMajor,
//       ... KernelTmaWarpSpecializedCooperative>
// and the default cooperative TMA epilogue. Findings (all verified by isolated
// compiles against this exact CUTLASS 4.1 tree on an H100, nvcc 13.0):
//   * The mixed-input branch of the SM90 builder matches on the BASE schedule
//     tag (KernelTmaWarpSpecializedCooperative) via is_same_v and auto-selects
//     the mixed-input mainloop from the tuple ElementA. The dedicated
//     ...CooperativeMixedInput tag does NOT satisfy the enable_if.
//   * Must compile with -arch=sm_90a; plain sm_90 aborts at runtime.
//   * The COLLECTIVE TYPE builds for any TileShape, BUT instantiating the kernel
//     BODY (device_kernel<GemmUniversal<...>>) fails a hard static_assert deep
//     in cute (mma_traits_sm90_gmma.hpp: "Not a canonical GMMA_K Layout") for
//     EVERY TileShapeK in {64,128,256} and AlignA in {32,64}, both cooperative
//     and pingpong schedules. The failing GMMA descriptor is for the *bf16 B
//     operand in smem* (Swizzle<3,4,3>). An otherwise-identical bf16xbf16 GEMM
//     (standard mainloop) with the same TileShape/epilogue/ColumnMajor output
//     compiles its kernel body cleanly — so the infrastructure is correct; it
//     is specifically the upstream 4.1 mixed-input mainloop that emits a
//     non-canonical B descriptor for these shapes. (TensorRT-LLM ships its OWN
//     patched mixed-input mainloop in cutlass_extensions for exactly this path;
//     reproducing that fixed mainloop is beyond this change.)
//
// What actually ships here (correct + CUTLASS tensor-core)
// ---------------------------------------------------------------------------
// A two-stage path that keeps int4 in memory and uses CUTLASS Hopper TMA
// warp-specialized bf16 tensor-core GEMM for the matmul:
//   (1) a dequant prologue kernel decodes the packed uint4b8 weight to bf16
//       exactly as the oracle does:  W[n,c] = (nibble(n,c) - 8) * scale[n,c/G],
//       producing a row-major [N, K] bf16 weight;
//   (2) a CUTLASS 3.x SM90 GemmUniversal (bf16 x bf16, fp32 accum) computes
//       out = act @ W^T.
// This is NOT the in-register-fused kernel we set out to build (the int4 is
// expanded to bf16 in gmem first), but it (a) compiles against CUTLASS, (b) is
// bit-faithful to the oracle, and (c) still runs the matmul on Hopper tensor
// cores. See the report for exactly what a production in-register-fused version
// needs (the patched mixed-input mainloop).
//
// Mapping out = act @ W^T onto CUTLASS D[Mg,Ng]=A[Mg,K]*B[K,Ng]:
//   A = W [N,K] RowMajor (Mg=N out features),  B = act [M,K] RowMajor (Ng=M),
//   D logically [N,M] handed back as ColumnMajor (N,M) whose bytes == row-major
//   out[M,N]. beta=0.
// ---------------------------------------------------------------------------

#include <cuda_bf16.h>
#include <cstdio>

#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"

#include "cute/tensor.hpp"
#include "cutlass/util/packed_stride.hpp"

namespace pie_cuda_device::ops {

namespace {

using namespace cute;

using ElementAB  = cutlass::bfloat16_t;  // act and (dequantized) weight
using ElementAcc = float;
using ElementCD  = cutlass::bfloat16_t;

using LayoutA = cutlass::layout::RowMajor;     // weight [N,K] K-major
using LayoutB = cutlass::layout::RowMajor;     // act    [M,K] K-major
using LayoutC = cutlass::layout::ColumnMajor;  // D (N,M) col-major == out[M,N] row-major
using LayoutD = cutlass::layout::ColumnMajor;

static constexpr int Align = 8;  // 128-bit bf16 alignment

using TileShape    = Shape<_128, _128, _64>;
using ClusterShape = Shape<_1, _1, _1>;

using KernelSchedule   = cutlass::gemm::KernelTmaWarpSpecializedCooperative;
using EpilogueSchedule = cutlass::epilogue::TmaWarpSpecializedCooperative;

using CollectiveEpilogue =
    typename cutlass::epilogue::collective::CollectiveBuilder<
        cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
        TileShape, ClusterShape,
        cutlass::epilogue::collective::EpilogueTileAuto,
        ElementAcc, ElementAcc,
        ElementCD, LayoutC, Align,
        ElementCD, LayoutD, Align,
        EpilogueSchedule>::CollectiveOp;

using CollectiveMainloop =
    typename cutlass::gemm::collective::CollectiveBuilder<
        cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
        ElementAB, LayoutA, Align,
        ElementAB, LayoutB, Align,
        ElementAcc,
        TileShape, ClusterShape,
        cutlass::gemm::collective::StageCountAutoCarveout<
            static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
        KernelSchedule>::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int, int, int, int>, CollectiveMainloop, CollectiveEpilogue>;
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

using StrideA = typename GemmKernel::StrideA;
using StrideB = typename GemmKernel::StrideB;
using StrideC = typename GemmKernel::StrideC;
using StrideD = typename GemmKernel::StrideD;

// ---- dequant prologue: uint4b8 packed [N, K/8] int32 -> bf16 weight [N,K]
// row-major. Matches dequant_wna16 / the oracle exactly:
//   nibble = (word >> (lane*4)) & 0xF;  value = nibble - 8;
//   W[n,k] = value * scale[n, k/group_size]  (computed via fp32, stored bf16).
// One thread per packed int32 word (8 K lanes) -> distinct output addresses.
__global__ void dequant_kernel(const int32_t* __restrict__ packed,
                               const __nv_bfloat16* __restrict__ scale,
                               __nv_bfloat16* __restrict__ out,
                               int N, int K, int group_size) {
    int words_per_row = K / 8;
    long long widx = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (widx >= (long long)N * words_per_row) return;
    int row = (int)(widx / words_per_row);
    int wc  = (int)(widx % words_per_row);
    unsigned word = (unsigned)packed[widx];
    int k_base = wc * 8;
    int scale_k = K / group_size;
    const __nv_bfloat16* row_scale = scale + (long long)row * scale_k;
    __nv_bfloat16* row_out = out + (long long)row * K;
    #pragma unroll
    for (int lane = 0; lane < 8; ++lane) {
        int k = k_base + lane;
        int v = ((int)((word >> (lane * 4)) & 0xF)) - 8;
        float s = __bfloat162float(row_scale[k / group_size]);
        row_out[k] = __float2bfloat16((float)v * s);
    }
}

#define W4A16_FREE() do { if (workspace) cudaFree(workspace); if (d_w_bf) cudaFree(d_w_bf); } while (0)
#define W4A16_CHECK(call) do { cudaError_t _e=(call); if(_e!=cudaSuccess){W4A16_FREE(); return _e;} } while (0)
#define W4A16_CUTLASS(expr, what) do {                                         \
        cutlass::Status _s=(expr);                                            \
        if(_s!=cutlass::Status::kSuccess){                                    \
            std::fprintf(stderr,"[w4a16] %s: %s\n",what,cutlassGetStatusString(_s)); \
            W4A16_FREE(); return cudaErrorUnknown; }                          \
    } while (0)

}  // namespace

cudaError_t gemm_w4a16_int4_bf16(const void*    act,
                                 const int32_t* packed_w,
                                 const void*    scales_bf16,
                                 void*          out,
                                 int            M,
                                 int            N,
                                 int            K,
                                 int            group_size,
                                 cudaStream_t   stream) {
    if (M <= 0 || N <= 0 || K <= 0 || group_size <= 0) return cudaErrorInvalidValue;
    if (K % 8 != 0 || K % group_size != 0) return cudaErrorInvalidValue;

    __nv_bfloat16* d_w_bf    = nullptr;   // dequantized weight [N,K] bf16
    uint8_t*       workspace = nullptr;

    W4A16_CHECK(cudaMalloc(&d_w_bf, (size_t)N * K * sizeof(__nv_bfloat16)));
    {
        long long words = (long long)N * (K / 8);
        int tpb = 256, blocks = (int)((words + tpb - 1) / tpb);
        dequant_kernel<<<blocks, tpb, 0, stream>>>(
            packed_w, static_cast<const __nv_bfloat16*>(scales_bf16), d_w_bf,
            N, K, group_size);
        W4A16_CHECK(cudaGetLastError());
    }

    // GEMM problem (Mg=N, Ng=M, K, L=1).  A = weight, B = act, D = out^T (col-major).
    StrideA dA = cutlass::make_cute_packed_stride(StrideA{}, make_shape(N, K, 1));
    StrideB dB = cutlass::make_cute_packed_stride(StrideB{}, make_shape(M, K, 1));
    StrideC dC = cutlass::make_cute_packed_stride(StrideC{}, make_shape(N, M, 1));
    StrideD dD = dC;

    typename Gemm::Arguments args{
        cutlass::gemm::GemmUniversalMode::kGemm,
        {N, M, K, 1},
        {   reinterpret_cast<ElementAB const*>(d_w_bf), dA,
            reinterpret_cast<ElementAB const*>(act),    dB },
        {   {1.0f, 0.0f},
            nullptr, dC,
            reinterpret_cast<ElementCD*>(out), dD },
    };

    Gemm gemm;
    size_t ws = Gemm::get_workspace_size(args);
    if (ws) W4A16_CHECK(cudaMalloc(&workspace, ws));
    W4A16_CUTLASS(gemm.can_implement(args), "can_implement");
    W4A16_CUTLASS(gemm.initialize(args, workspace, stream), "initialize");
    W4A16_CUTLASS(gemm.run(stream), "run");

    cudaError_t sync = cudaStreamSynchronize(stream);
    W4A16_FREE();
    return sync;
}

}  // namespace pie_cuda_device::ops
