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
#include <algorithm>
#include <vector>

#include "decode_abi.hpp"
#include "harness.hpp"
#include "mtl4_context.hpp"

#ifndef PIE_METAL_TOOL_KERNELS_DIR
#define PIE_METAL_TOOL_KERNELS_DIR "."
#endif
#ifndef PIE_METAL_TOOL_LOCAL_KERNELS_DIR
#define PIE_METAL_TOOL_LOCAL_KERNELS_DIR "."
#endif

using namespace pie::metal;

namespace {
constexpr int GROUP = 64;
constexpr size_t TSZ = 2;  // bfloat16

// Amortize over `repeats` dispatches in one command buffer so the per-CB sync
// floor drops out and what is left is what a fused decode step actually pays.
double amortized_ms(RawMetalContext& ctx, Pso pso, Kernel k, int ord, Grid grid,
                    Threadgroup tg, int repeats = 64, int copies = 1) {
    LatencyHarness h(ctx);
    auto fn = [&](StepEncoder& se) {
        se.set_pso(pso);
        for (int i = 0; i < repeats; ++i) {
            // Rotate the argument table across distinct weight copies: repeating
            // one dispatch measures a cache-hot GEMM, and a real decode step
            // reads each projection's weights exactly once, cold.
            se.set_argtable(k, ord + (copies > 1 ? i % copies : 0));
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
    const int SPLIT = argc > 4 ? atoi(argv[4]) : 1;
    const bool COLD = getenv("PROBE_COLD") != nullptr;
    const int BM = getenv("PROBE_BM") ? atoi(getenv("PROBE_BM")) : 16;

    auto ctx = RawMetalContext::create(/*heap_bytes=*/3072ull << 20);
    if (!ctx) { printf("FAIL: no context\n"); return 1; }
    std::string err;

    Pso stream = ctx->compile_pso_from_file(
                                            std::string(PIE_METAL_TOOL_LOCAL_KERNELS_DIR) +
                                                "/roofline_stream.metal",
                                            "stream_read_bf16", &err);
    if (!stream.valid()) { printf("FAIL stream compile: %s\n", err.c_str()); return 1; }
    Pso qmm = ctx->compile_pso_from_file(
        dir + "/quantized_qmm_t.metal",
        "affine_qmm_t_bfloat16_gs_64_b_4_bm_" + std::to_string(BM) + "_bn_" +
            std::to_string(BN), &err);
    if (!qmm.valid()) { printf("FAIL qmm compile: %s\n", err.c_str()); return 1; }
    Pso qmv = ctx->compile_pso_from_file(dir + "/quantized_qmv.metal",
                                         "affine_qmv_fast_bfloat16_gs_64_b_4", &err);
    if (!qmv.valid()) { printf("FAIL qmv compile: %s\n", err.c_str()); return 1; }
    Pso qsk, qred;
    if (SPLIT > 1) {
        const std::string sp = std::to_string(SPLIT);
        qsk = ctx->compile_pso_from_file(
            dir + "/quantized_qmm_t.metal",
            "affine_qmm_t_splitk_bfloat16_gs_64_b_4_bm_16_bn_" +
                std::to_string(BN) + "_sp_" + sp, &err);
        qred = ctx->compile_pso_from_file(dir + "/quantized_qmm_t.metal",
                                          "qmm_splitk_reduce_bfloat16_sp_" + sp, &err);
        if (!qsk.valid() || !qred.valid()) {
            printf("FAIL splitk compile: %s\n", err.c_str());
            return 1;
        }
    }

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

    // What one more barrier-separated dispatch costs, with no compute and no
    // memory traffic in it -- the floor every dispatch in the step pays.
    {
        Pso nop = ctx->compile_pso_from_file(
            std::string(PIE_METAL_TOOL_LOCAL_KERNELS_DIR) + "/nop_probe.metal",
            "nop_probe", &err);
        if (nop.valid()) {
            LatencyHarness h(*ctx);
            for (int reps : {64, 256, 1024}) {
                auto fn = [&](StepEncoder& se) {
                    se.set_pso(nop);
                    for (int i = 0; i < reps; ++i) {
                        se.dispatch(Grid{1, 1, 1}, Threadgroup{1, 1, 1});
                        se.barrier();
                    }
                };
                const double ms = h.time_step("nop", fn, 40, 10).median.gpu_exec_ms;
                printf("nop dispatch+barrier x%-5d %7.3f ms -> %.2f us each\n", reps, ms,
                       ms * 1000.0 / reps);
            }
        }
    }
    printf("\nM=%d BN=%d SPLIT=%d  4-bit affine g64, per-dispatch amortized\n", M, BN,
           SPLIT);
    printf("  (threadgroups per dispatch = N/BN x ceil(M/16))\n");
    printf("%-24s %9s %9s %8s %9s %8s\n", "shape", "ms", "GB/s", "%roof",
           "GFLOP/s", "n_tg");
    double total_ms = 0, total_bytes = 0, total_flop = 0;
    int ord = 1;
    for (const auto& s : kShapes) {
        const uint32_t K = s.K, N = s.N;
        const size_t wb = size_t(N) * (K / 2);
        const size_t sb = size_t(N) * (K / GROUP) * TSZ;
        // Enough distinct weight copies to exceed any cache the machine has, so
        // the rotation above actually reaches memory.
        const int copies = COLD ? int(std::min<size_t>(16, (192ull << 20) / (wb + 2 * sb) + 1)) : 1;
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
        std::vector<SlotHandle> extra;
        for (int c = 1; c < copies; ++c) {
            SlotHandle w2 = ctx->heap_alloc(wb);
            if (!w2.valid()) break;
            memset(w2.contents(), 0, wb);
            extra.push_back(w2);
            ctx->arg_bind(kk, ord + c, uint8_t(bind::Qmv::W), w2);
            ctx->arg_bind(kk, ord + c, uint8_t(bind::Qmv::Scales), sc);
            ctx->arg_bind(kk, ord + c, uint8_t(bind::Qmv::Biases), bi);
            ctx->arg_bind(kk, ord + c, uint8_t(bind::Qmv::X), x);
            ctx->arg_bind(kk, ord + c, uint8_t(bind::Qmv::Out), y);
            ctx->arg_bind(kk, ord + c, uint8_t(bind::Qmv::K), ks);
            ctx->arg_bind(kk, ord + c, uint8_t(bind::Qmv::N), ns);
        }
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
            const uint32_t rows = uint32_t((M + BM - 1) / BM * BM);
            if (N % uint32_t(BN) != 0) { printf("  %s: N %% BN != 0\n", s.label); continue; }
            g = Grid{32u * (N / uint32_t(BN)), 2u * (rows / uint32_t(BM)), 2};
            tg = Threadgroup{32, 2, 2};
        }
        double ms;
        if (SPLIT > 1) {
            const uint32_t rows = uint32_t((M + 15) / 16 * 16);
            SlotHandle part = ctx->heap_alloc(size_t(SPLIT) * rows * N * sizeof(float));
            SlotHandle ms_ = ctx->heap_alloc(sizeof(int32_t));
            if (!part.valid()) { printf("  %s: partial OOM\n", s.label); continue; }
            *static_cast<int32_t*>(ms_.contents()) = int32_t(rows);
            // split-K argtable: partial at 4, M at 7 (K/N keep the Qmv slots)
            ctx->arg_bind(kk, ord + 100, 0, w);
            ctx->arg_bind(kk, ord + 100, 1, sc);
            ctx->arg_bind(kk, ord + 100, 2, bi);
            ctx->arg_bind(kk, ord + 100, 3, x);
            ctx->arg_bind(kk, ord + 100, 4, y);
            ctx->arg_bind(kk, ord + 100, 5, ks);
            ctx->arg_bind(kk, ord + 100, 6, ns);
            ctx->arg_bind(kk, ord + 100, 7, y);
            ctx->arg_bind(kk, ord + 100, 8, part);
            ctx->arg_bind(kk, ord + 100, 9, ms_);
            ctx->make_resident();
            const Grid gk{32u * (N / uint32_t(BN)), 2u * (rows / 16u),
                          2u * uint32_t(SPLIT)};
            const Threadgroup tgk{32, 2, 2};
            const Grid gr{N, rows, 1};
            const Threadgroup tgr{256, 1, 1};
            LatencyHarness h2(*ctx);
            const int reps = 32;
            auto fn = [&](StepEncoder& se) {
                for (int i = 0; i < reps; ++i) {
                    se.set_pso(qsk);
                    se.set_argtable(kk, ord + 100);
                    se.dispatch(gk, tgk);
                    se.barrier();
                    se.set_pso(qred);
                    se.set_argtable(kk, ord + 100);
                    se.dispatch(gr, tgr);
                    se.barrier();
                }
            };
            ms = h2.time_step("sk", fn, 40, 10).median.gpu_exec_ms / reps;
        } else {
            ms = amortized_ms(*ctx, pso, kk, ord, g, tg, 64, int(extra.size()) + 1);
        }
        const double bytes = double(weight_bytes(K, N));
        const double flop = 2.0 * double(M) * double(K) * double(N);
        printf("%-24s %9.4f %9.1f %7.0f%% %9.1f %8u\n", s.label, ms,
               bytes / (ms * 1e-3) / 1e9, 100.0 * (bytes / (ms * 1e-3) / 1e9) / roof_gbs,
               flop / (ms * 1e-3) / 1e9,
               M == 1 ? uint32_t(N / 4) : (N / uint32_t(BN)) * uint32_t((M + 15) / 16));
        total_ms += ms * s.count;
        total_bytes += bytes * s.count;
        total_flop += flop * s.count;
        ord += 32;  // leave room for this shape's weight copies
    }
    printf("\nwhole step (projections only, x count):\n");
    printf("  %.2f ms   %.0f MB   %.1f GB/s (%.0f%% of roof)   %.2f TFLOP/s\n",
           total_ms, total_bytes / 1e6, total_bytes / (total_ms * 1e-3) / 1e9,
           100.0 * (total_bytes / (total_ms * 1e-3) / 1e9) / roof_gbs,
           total_flop / (total_ms * 1e-3) / 1e12);
    printf("  bandwidth floor at the roof: %.2f ms\n", total_bytes / (roof_gbs * 1e9) * 1e3);
    return 0;
}
