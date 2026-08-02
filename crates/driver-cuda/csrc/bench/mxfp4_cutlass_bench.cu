// MXFP4 on Blackwell tensor cores, timed against pie's unpack GEMV.
//
// The whole MoE gap is the FP4->bf16 unpack. At gpt-oss's gate/up shape
// (mxfp4_roof_bench.cu, cold-read): streaming the 33.2 MB of packed weights
// costs 8.2 us, pie's PRMT unpack takes that to 21.5, and the FMA chain on
// top is free. Three attempts to make the unpack cheaper all failed --
// Blackwell's native cvt is 2x SLOWER than the hand-rolled __byte_perm, extra
// accumulators do nothing, and there are no register spills to recover. The
// unpack is already good; it just should not exist. tcgen05 eats E2M1
// operands directly, which is the entire 1.8x SGLang holds here.
//
// So: does a CUTLASS block-scaled tensor-core GEMM deliver that at OUR shape
// and batch? SGLang's ~10.5 us is deflated from a CUPTI trace of a different
// call pattern, and the TRT-LLM cubin path SGLang uses refuses gpt-oss's
// shape from a standalone caller. CUTLASS compiles from source, so it can
// answer for the shape we actually run.
//
// TWO THINGS THIS BENCH ESTABLISHED, each of which cost a build to learn:
//
// 1. K MUST BE A MULTIPLE OF 128. gpt-oss's hidden and intermediate are both
//    2880 = 22.5 * 128, and `can_implement` refuses it. Swept: 2688 ok, 2816
//    ok, 2880 REFUSED, 2944 ok, 3072 ok -- so it is K % 128, matching
//    AlignmentB = 128 elements of e2m1. N is unconstrained (2880 and 23008
//    both run). This is near-certainly why TRT-LLM answered "No valid config
//    found for the given problem shape" for the same model: it is gpt-oss's
//    odd hidden size, not the kernel. pie must pad K 2880 -> 2944, reading
//    2.2% more bytes. That is the price of admission to tensor cores.
//
// 2. BOTH OPERANDS MUST BE BLOCK-SCALED. OpClassBlockScaledTensorOp has no
//    specialisation taking a plain bf16 operand. Example 72c is named
//    "mixed_mxfp8_bf16" for its bf16 OUTPUT, not its activations -- its A is
//    mx_float8_t. The FP4-weights-against-unquantised-bf16 case
//    (`use_wfp4a16`) is a TRT-LLM *runner* feature, not a CUTLASS collective
//    one. So pie does need a bf16 -> MXFP8 activation quantiser, the same one
//    SGLang pays 4.9% of its decode for. Cheap at decode -- M x K is 2880
//    values at M=1 against 33 MB of weights -- but not free, and an earlier
//    read of that filename claimed it was.
//
// What is swept here is the TILE. The first working config used 72c's
// _256,_256,_256 with a 2x4 cluster and read 0.33 TB/s -- 115 us, six times
// slower than pie. That is a prefill tile: at N=23040 it makes 90 CTAs to
// fill 148 SMs, so a third of the GPU idles while the rest wait on HBM.
// Decode wants many narrow tiles, not few fat ones.
//
// Build (~4 min; one CUTLASS instantiation per config is not cheap):
//   FI=/root/.cache/cpm/flashinfer/e3c8
//   nvcc -O3 -std=c++20 -arch=sm_100a --expt-relaxed-constexpr \
//        --extended-lambda -I $FI/3rdparty/cutlass/include \
//        -I $FI/3rdparty/cutlass/tools/util/include \
//        -o /tmp/mxfp4_cutlass driver/cuda/bench/mxfp4_cutlass_bench.cu
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/util/device_memory.h"
#include "cutlass/util/packed_stride.hpp"

using namespace cute;

namespace {

using ElementA   = cutlass::mx_float8_t<cutlass::float_e4m3_t>;
using LayoutATag = cutlass::layout::RowMajor;
constexpr int AlignmentA = 16;

using ElementB   = cutlass::mx_float4_t<cutlass::float_e2m1_t>;
using LayoutBTag = cutlass::layout::ColumnMajor;
constexpr int AlignmentB = 128;

using ElementC   = cutlass::bfloat16_t;
using ElementD   = cutlass::bfloat16_t;
using LayoutCTag = cutlass::layout::RowMajor;
using LayoutDTag = cutlass::layout::RowMajor;
constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;
constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;

using ElementAccumulator = float;
using ArchTag       = cutlass::arch::Sm100;
using OperatorClass = cutlass::arch::OpClassBlockScaledTensorOp;

template <class MmaTileShape, class ClusterShape>
struct Cfg {
    using CollectiveEpilogue =
        typename cutlass::epilogue::collective::CollectiveBuilder<
            ArchTag, OperatorClass, MmaTileShape, ClusterShape,
            cutlass::epilogue::collective::EpilogueTileAuto,
            ElementAccumulator, ElementAccumulator,
            ElementC, LayoutCTag, AlignmentC,
            ElementD, LayoutDTag, AlignmentD,
            cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;

    using CollectiveMainloop =
        typename cutlass::gemm::collective::CollectiveBuilder<
            ArchTag, OperatorClass,
            ElementA, LayoutATag, AlignmentA,
            ElementB, LayoutBTag, AlignmentB,
            ElementAccumulator, MmaTileShape, ClusterShape,
            cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
                sizeof(typename CollectiveEpilogue::SharedStorage))>,
            cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;

    using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
        Shape<int, int, int, int>, CollectiveMainloop, CollectiveEpilogue>;
    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
};

constexpr int kWarmup = 10;
constexpr int kIters = 50;
// B200's L2 is 132.6 MB and one weight buffer is ~34 MB, so a single
// allocation would sit in cache for the whole loop and report L2 bandwidth.
// The engine never gets that -- gpt-oss streams ~1.3 GB of expert weights per
// token. Rotate enough buffers that returning to one has evicted it.
constexpr int kBufs = 8;

constexpr double kPieUs = 19.0;

template <class C>
void bench(const char* tag, int m, int n, int k) {
    using Gemm = typename C::Gemm;
    using StrideA = typename Gemm::GemmKernel::StrideA;
    using StrideB = typename Gemm::GemmKernel::StrideB;
    using StrideC = typename Gemm::GemmKernel::StrideC;
    using StrideD = typename Gemm::GemmKernel::StrideD;
    using Mainloop = typename Gemm::GemmKernel::CollectiveMainloop;
    using LayoutSFA = typename Mainloop::LayoutSFA;
    using LayoutSFB = typename Mainloop::LayoutSFB;
    using Blk = typename Mainloop::Sm1xxBlkScaledConfig;
    using ElementSF = typename Mainloop::ElementSF;

    auto shape = cute::make_shape(m, n, k, 1);
    StrideA sA = cutlass::make_cute_packed_stride(StrideA{}, {m, k, 1});
    StrideB sB = cutlass::make_cute_packed_stride(StrideB{}, {n, k, 1});
    StrideC sC = cutlass::make_cute_packed_stride(StrideC{}, {m, n, 1});
    StrideD sD = cutlass::make_cute_packed_stride(StrideD{}, {m, n, 1});
    LayoutSFA lSFA = Blk::tile_atom_to_shape_SFA(shape);
    LayoutSFB lSFB = Blk::tile_atom_to_shape_SFB(shape);

    cutlass::DeviceAllocation<typename Gemm::ElementA> A(
        static_cast<size_t>(m) * k);
    cutlass::DeviceAllocation<ElementC> Cbuf(static_cast<size_t>(m) * n);
    cutlass::DeviceAllocation<ElementD> D(static_cast<size_t>(m) * n);
    cutlass::DeviceAllocation<ElementSF> SFA(cosize(lSFA));

    // One weight set per rotation slot. Each Gemm keeps its own params, so
    // swapping buffers costs nothing at launch time.
    std::vector<cutlass::DeviceAllocation<typename Gemm::ElementB>> B(kBufs);
    std::vector<cutlass::DeviceAllocation<ElementSF>> SFB(kBufs);
    std::vector<cutlass::DeviceAllocation<uint8_t>> ws(kBufs);
    std::vector<Gemm> gemm(kBufs);
    for (int i = 0; i < kBufs; ++i) {
        B[i].reset(static_cast<size_t>(n) * k);
        SFB[i].reset(cosize(lSFB));
        typename Gemm::Arguments args{
            cutlass::gemm::GemmUniversalMode::kGemm, shape,
            {A.get(), sA, B[i].get(), sB, SFA.get(), lSFA, SFB[i].get(), lSFB},
            {{1.0f, 0.0f}, Cbuf.get(), sC, D.get(), sD}};
        if (i == 0) {
            cutlass::Status st = gemm[0].can_implement(args);
            if (st != cutlass::Status::kSuccess) {
                std::printf("  %-24s m=%-4d REFUSED (%s)\n", tag, m,
                            cutlass::cutlassGetStatusString(st));
                return;
            }
        }
        ws[i].reset(Gemm::get_workspace_size(args));
        if (gemm[i].initialize(args, ws[i].get()) != cutlass::Status::kSuccess) {
            std::printf("  %-24s m=%-4d initialize failed\n", tag, m);
            return;
        }
    }

    for (int i = 0; i < kWarmup; ++i) gemm[i % kBufs].run();
    cudaDeviceSynchronize();
    cudaEvent_t a, b;
    cudaEventCreate(&a);
    cudaEventCreate(&b);
    cudaEventRecord(a);
    for (int i = 0; i < kIters; ++i) gemm[i % kBufs].run();
    cudaEventRecord(b);
    cudaEventSynchronize(b);
    float ms = 0.f;
    cudaEventElapsedTime(&ms, a, b);
    const double us = ms * 1e3 / kIters;
    // Weight bytes only: at these M the read of B is the entire cost.
    const double bytes = static_cast<double>(n) * k * 0.5 +   // E2M1
                         static_cast<double>(n) * k / 32.0;   // UE8M0
    std::printf("  %-24s m=%-4d %8.2f us  %5.2f TB/s  %5.2fx pie\n", tag, m, us,
                bytes / (us * 1e-6) / 1e12, kPieUs / us);
    cudaEventDestroy(a);
    cudaEventDestroy(b);
}

// Decode is M=1..64 against a very wide N, so the tile wants to make MANY
// CTAs. N=23040 over a 256-wide tile is 90 CTAs on 148 SMs; over 128 it is
// 180. The 2-SM variants (cluster M=2) need MmaTile M=256.
using T128_128 = Cfg<Shape<_128, _128, _256>, Shape<_1, _1, _1>>;
using T128_256 = Cfg<Shape<_128, _256, _256>, Shape<_1, _1, _1>>;
using T256_128 = Cfg<Shape<_256, _128, _256>, Shape<_2, _1, _1>>;
using T256_256 = Cfg<Shape<_256, _256, _256>, Shape<_2, _4, _1>>;  // 72c's

}  // namespace

int main(int argc, char** argv) {
    // gpt-oss gate/up for the 4 experts one token routes to, flattened into
    // one GEMM: N = topk * 2 * intermediate, K = hidden padded to a multiple
    // of 128 (see note 1 in the header -- 2880 is refused).
    const int H_true = 2880, H = 2944, I = 2880, topk = 4;
    const int N = topk * 2 * I;
    std::printf("gpt-oss gate/up as one GEMM: N=%d K=%d "
                "(K padded from %d, +%.1f%% bytes, %.1f MB of MXFP4)\n",
                N, H, H_true, 100.0 * (H - H_true) / H_true,
                (N * static_cast<double>(H) * 0.5 + N * H / 32.0) / 1e6);
    std::printf("  pie's unpack GEMV: %.1f us | cold stream-only roof: 8.2 us"
                " | SGLang est: ~10.5 us\n"
                "  (%d buffers rotated to defeat the 132.6 MB L2)\n\n",
                kPieUs, kBufs);

    if (argc == 4) {
        bench<T128_128>("128x128x256 c1x1", std::atoi(argv[1]),
                        std::atoi(argv[2]), std::atoi(argv[3]));
        return 0;
    }
    for (int m : {1, 8, 128}) {
        bench<T128_128>("128x128x256 c1x1", m, N, H);
        bench<T128_256>("128x256x256 c1x1", m, N, H);
        bench<T256_128>("256x128x256 c2x1", m, N, H);
        bench<T256_256>("256x256x256 c2x4(72c)", m, N, H);
        std::printf("\n");
    }
    return 0;
}
