// roofline_probe.cpp — where the decode step actually sits against the machine.
//
// The decode is a 4-bit quantized GEMM over the whole checkpoint once per step.
// Two numbers bound it: the bytes it must stream (weights + scales + biases) and
// the FMAs it must issue (M rows x 2 x params).  This tool measures the machine's
// achievable streaming roof with a kernel that does nothing but read, then runs
// the real `affine_qmm_t` / `affine_qmv_fast` at the model's projection shapes so
// each one can be reported as a fraction of that roof.
//
// Timing is data-independent for these kernels (fixed loop bounds), so zero-filled
// buffers are representative; bit-exactness is owned by the parity tests.

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "decode_abi.hpp"
#include "harness.hpp"
#include "mtl4_context.hpp"

#ifndef PIE_METAL_TOOL_KERNELS_DIR
#define PIE_METAL_TOOL_KERNELS_DIR "."
#endif

using namespace pie::metal;

namespace {
constexpr int GROUP = 64;
constexpr size_t TSZ = 2;  // bfloat16

// Amortize over `repeats` dispatches in one command buffer so the per-CB sync
// floor drops out and what is left is what a fused decode step actually pays.
double amortized_ms(RawMetalContext& ctx, Pso pso, Kernel k, int ord, Grid grid,
                    Threadgroup tg, int repeats = 64) {
    LatencyHarness h(ctx);
    auto fn = [&](StepEncoder& se) {
        se.set_pso(pso);
        for (int i = 0; i < repeats; ++i) {
            se.set_argtable(k, ord);
            se.dispatch(grid, tg);
            se.barrier();
        }
    };
    return h.time_step("b", fn, 60, 15).median.gpu_exec_ms / repeats;
}

struct Shape { const char* label; uint32_t K; uint32_t N; int count; };
// qwen3.5-0.8B, 24 layers = 18 GDN + 6 full-attn.  `count` is how many of each
// the model runs per token, so the per-step totals below are the real ones.
const Shape kShapes[] = {
    {"gdn_in    K1024 N6144", 1024, 6144, 18},
    {"gdn_out   K4096 N1024", 4096, 1024, 18},
    {"gate/up   K1024 N3584", 1024, 3584, 48},
    {"down_proj K3584 N1024", 3584, 1024, 24},
    {"q_proj    K1024 N2048", 1024, 2048, 6},
    {"kv_proj   K1024 N512 ", 1024, 512, 12},
    {"o_proj    K2048 N1024", 2048, 1024, 6},
    {"lm_head   K1024 N248320", 1024, 248320, 1},
};

size_t weight_bytes(uint32_t K, uint32_t N) {
    return size_t(N) * (K / 2) + 2 * size_t(N) * (K / GROUP) * TSZ;
}
}  // namespace

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    std::string dir = PIE_METAL_TOOL_KERNELS_DIR;
    if (argc > 1) dir = argv[1];
    const int M = argc > 2 ? atoi(argv[2]) : 16;
    const int BN = argc > 3 ? atoi(argv[3]) : 32;

    auto ctx = RawMetalContext::create(/*heap_bytes=*/3072ull << 20);
    if (!ctx) { printf("FAIL: no context\n"); return 1; }
    std::string err;

    Pso stream = ctx->compile_pso_from_file(dir + "/roofline_stream.metal",
                                            "stream_read_bf16", &err);
    if (!stream.valid()) { printf("FAIL stream compile: %s\n", err.c_str()); return 1; }
    Pso qmm = ctx->compile_pso_from_file(
        dir + "/quantized_qmm_t.metal",
        "affine_qmm_t_bfloat16_gs_64_b_4_bm_16_bn_" + std::to_string(BN), &err);
    if (!qmm.valid()) { printf("FAIL qmm compile: %s\n", err.c_str()); return 1; }
    Pso qmv = ctx->compile_pso_from_file(dir + "/quantized_qmv.metal",
                                         "affine_qmv_fast_bfloat16_gs_64_b_4", &err);
    if (!qmv.valid()) { printf("FAIL qmv compile: %s\n", err.c_str()); return 1; }

    // ── the machine's streaming roof ────────────────────────────────────────
    double roof_gbs = 0;
    {
        const size_t BYTES = 512ull << 20;
        SlotHandle src = ctx->heap_alloc(BYTES);
        SlotHandle dst = ctx->heap_alloc(1024 * TSZ);
        if (!src.valid()) { printf("FAIL: roof alloc\n"); return 1; }
        memset(src.contents(), 0, BYTES);
        ctx->arg_bind(Kernel::Rms, 0, 0, src);
        ctx->arg_bind(Kernel::Rms, 0, 1, dst);
        ctx->make_resident();
        // one thread per 8 bf16 (uint4 loads), 256-wide threadgroups
        const uint32_t threads = uint32_t(BYTES / TSZ / 8);
        double ms = amortized_ms(*ctx, stream, Kernel::Rms, 0, Grid{threads, 1, 1},
                                 Threadgroup{256, 1, 1}, 8);
        roof_gbs = double(BYTES) / (ms * 1e-3) / 1e9;
        printf("streaming roof (read-only, %zu MB): %.3f ms -> %.1f GB/s\n\n",
               BYTES >> 20, ms, roof_gbs);
    }

    printf("M=%d BN=%d  4-bit affine g64, per-dispatch amortized\n", M, BN);
    printf("  (threadgroups per dispatch = N/BN x ceil(M/16))\n");
    printf("%-24s %9s %9s %8s %9s %8s\n", "shape", "ms", "GB/s", "%roof",
           "GFLOP/s", "n_tg");
    double total_ms = 0, total_bytes = 0, total_flop = 0;
    int ord = 1;
    for (const auto& s : kShapes) {
        const uint32_t K = s.K, N = s.N;
        const size_t wb = size_t(N) * (K / 2);
        const size_t sb = size_t(N) * (K / GROUP) * TSZ;
        SlotHandle w = ctx->heap_alloc(wb);
        SlotHandle sc = ctx->heap_alloc(sb);
        SlotHandle bi = ctx->heap_alloc(sb);
        SlotHandle x = ctx->heap_alloc(size_t(M) * K * TSZ);
        SlotHandle y = ctx->heap_alloc(size_t(M) * N * TSZ);
        SlotHandle ks = ctx->heap_alloc(sizeof(int32_t));
        SlotHandle ns = ctx->heap_alloc(sizeof(int32_t));
        if (!w.valid() || !y.valid()) { printf("  %s: heap OOM\n", s.label); continue; }
        memset(w.contents(), 0, wb);
        memset(sc.contents(), 0, sb);
        memset(bi.contents(), 0, sb);
        memset(x.contents(), 0, size_t(M) * K * TSZ);
        *static_cast<int32_t*>(ks.contents()) = int32_t(K);
        *static_cast<int32_t*>(ns.contents()) = int32_t(N);
        const Kernel kk = Kernel::QmvIn;
        ctx->arg_bind(kk, ord, uint8_t(bind::Qmv::W), w);
        ctx->arg_bind(kk, ord, uint8_t(bind::Qmv::Scales), sc);
        ctx->arg_bind(kk, ord, uint8_t(bind::Qmv::Biases), bi);
        ctx->arg_bind(kk, ord, uint8_t(bind::Qmv::X), x);
        ctx->arg_bind(kk, ord, uint8_t(bind::Qmv::Out), y);
        ctx->arg_bind(kk, ord, uint8_t(bind::Qmv::K), ks);
        ctx->arg_bind(kk, ord, uint8_t(bind::Qmv::N), ns);
        ctx->make_resident();

        Grid g;
        Threadgroup tg;
        Pso pso;
        if (M == 1) {
            pso = qmv;
            g = Grid{32u * (N / 4u), 1, 1};
            tg = Threadgroup{32, 2, 1};
        } else {
            pso = qmm;
            const uint32_t rows = uint32_t((M + 15) / 16 * 16);
            if (N % uint32_t(BN) != 0) { printf("  %s: N %% BN != 0\n", s.label); continue; }
            g = Grid{32u * (N / uint32_t(BN)), 2u * (rows / 16u), 2};
            tg = Threadgroup{32, 2, 2};
        }
        const double ms = amortized_ms(*ctx, pso, kk, ord, g, tg);
        const double bytes = double(weight_bytes(K, N));
        const double flop = 2.0 * double(M) * double(K) * double(N);
        printf("%-24s %9.4f %9.1f %7.0f%% %9.1f %8u\n", s.label, ms,
               bytes / (ms * 1e-3) / 1e9, 100.0 * (bytes / (ms * 1e-3) / 1e9) / roof_gbs,
               flop / (ms * 1e-3) / 1e9,
               M == 1 ? uint32_t(N / 4) : (N / uint32_t(BN)) * uint32_t((M + 15) / 16));
        total_ms += ms * s.count;
        total_bytes += bytes * s.count;
        total_flop += flop * s.count;
        ++ord;
    }
    printf("\nwhole step (projections only, x count):\n");
    printf("  %.2f ms   %.0f MB   %.1f GB/s (%.0f%% of roof)   %.2f TFLOP/s\n",
           total_ms, total_bytes / 1e6, total_bytes / (total_ms * 1e-3) / 1e9,
           100.0 * (total_bytes / (total_ms * 1e-3) / 1e9) / roof_gbs,
           total_flop / (total_ms * 1e-3) / 1e12);
    printf("  bandwidth floor at the roof: %.2f ms\n", total_bytes / (roof_gbs * 1e9) * 1e3);
    return 0;
}
