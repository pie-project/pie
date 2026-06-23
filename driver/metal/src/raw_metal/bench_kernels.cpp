// bench_kernels.cpp — micro-bench delta's REAL ported kernels (rms_single_row +
// affine_qmv_fast) at qwen3.6 decode shapes. Produces the first gpu-exec-ms datapoints
// (manager's critical-path convergence). Timing is data-independent for these kernels
// (fixed loop bounds), so zero-filled buffers give representative exec-ms; delta owns
// bit-exact correctness separately (cosine 1.0).
//
// bf16 variants for apples-to-apples vs mlx_lm (qwen3.6 runs bfloat16).

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>

#include "decode_abi.hpp"
#include "harness.hpp"
#include "mtl4_context.hpp"

#ifndef RAW_METAL_KERNELS_DIR
#define RAW_METAL_KERNELS_DIR "."
#endif

using namespace pie_metal_driver::raw_metal;

namespace {
constexpr int GROUP = 64;  // affine g64

// Per-dispatch exec amortized over R dispatches in ONE command buffer (one commit/wait):
// strips the ~0.16ms per-CB sync floor, leaving compute + inter-dispatch barrier — what
// the fused ~322-dispatch decode step actually pays per kernel.
double batched_exec_ms(RawMetalContext& ctx, Pso pso, Kernel k, int layer,
                       Grid grid, Threadgroup tg, int repeats = 128) {
    LatencyHarness h(ctx);
    auto fn = [&](StepEncoder& se) {
        se.set_pso(pso);
        for (int i = 0; i < repeats; ++i) {
            se.set_argtable(k, layer);
            se.dispatch(grid, tg);
            se.barrier();
        }
    };
    BenchResult r = h.time_step("batch", fn, 80, 20);
    return r.median.gpu_exec_ms / repeats;
}

// qwen3.6 (hidden=1024) projection shapes: {label, K(in), N(out)}.
struct Shape { const char* label; uint32_t K; uint32_t N; };
const Shape kQmvShapes[] = {
    {"q_proj    K1024 N2048", 1024, 2048},
    {"kv_proj   K1024 N512 ", 1024, 512},
    {"gate/up   K1024 N3584", 1024, 3584},
    {"down_proj K3584 N1024", 3584, 1024},
    {"gdn_in    K1024 N6144", 1024, 6144},
};
}  // namespace

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    std::string kernels_dir = RAW_METAL_KERNELS_DIR;
    if (argc > 1) kernels_dir = argv[1];

    printf("raw-Metal micro-bench — delta's ported kernels (bf16), M1 Max\n");
    printf("(gpu-exec-ms = single-dispatch isolation incl. launch+sync; decode fuses ~322)\n\n");

    auto ctx = RawMetalContext::create(/*heap_bytes=*/512ull << 20);
    if (!ctx) { printf("FAIL: no context\n"); return 1; }

    std::string err;
    Pso rms = ctx->compile_pso_from_file(kernels_dir + "/rms_norm.metal",
                                         "rms_single_row_bfloat16", &err);
    if (!rms.valid()) { printf("FAIL rms compile: %s\n", err.c_str()); return 1; }
    Pso qmv = ctx->compile_pso_from_file(kernels_dir + "/quantized_qmv.metal",
                                         "affine_qmv_fast_bfloat16_gs_64_b_4", &err);
    if (!qmv.valid()) { printf("FAIL qmv compile: %s\n", err.c_str()); return 1; }
    printf("runtime MSL compile OK (rms_single_row_bfloat16, affine_qmv_fast_bf16_g64)\n\n");

    const size_t TSZ = 2;  // bfloat16

    // ── rms_single_row @ hidden=1024 ──
    {
        const uint32_t H = 1024;
        SlotHandle x   = ctx->heap_alloc(H * TSZ);
        SlotHandle w   = ctx->heap_alloc(H * TSZ);
        SlotHandle out = ctx->heap_alloc(H * TSZ);
        SlotHandle pp  = ctx->heap_alloc(sizeof(float) + 2 * sizeof(uint32_t));
        memset(x.contents(), 0, H * TSZ);
        memset(w.contents(), 0, H * TSZ);
        // RmsParams { float eps; uint axis_size; uint w_stride; }
        auto* f = static_cast<float*>(pp.contents());
        f[0] = 1e-6f;
        auto* u = reinterpret_cast<uint32_t*>(static_cast<char*>(pp.contents()) + sizeof(float));
        u[0] = H; u[1] = 1;
        ctx->arg_bind(Kernel::Rms, 0, 0, x);
        ctx->arg_bind(Kernel::Rms, 0, 1, w);
        ctx->arg_bind(Kernel::Rms, 0, 2, out);
        ctx->arg_bind(Kernel::Rms, 0, 3, pp);
        ctx->make_resident();
        LatencyHarness h(*ctx);
        // 256 threads (=H/N_READS), 1 threadgroup (1 decode row).
        BenchResult r = h.bench_kernel("rms_single_row H1024", rms, Kernel::Rms, 0,
                                       Grid{256, 1, 1}, Threadgroup{256, 1, 1});
        double amort = batched_exec_ms(*ctx, rms, Kernel::Rms, 0,
                                       Grid{256, 1, 1}, Threadgroup{256, 1, 1});
        printf("rms:\n");
        print_result(r);
        printf("    -> amortized compute+barrier: %.4f ms/dispatch\n\n", amort);
    }

    // ── affine_qmv_fast across projection shapes ──
    printf("affine_qmv_fast (4-bit g64):\n");
    LatencyHarness h(*ctx);
    int layer = 1;
    for (const auto& s : kQmvShapes) {
        const uint32_t K = s.K, N = s.N;
        // packed 4-bit: K/2 bytes per row; scales/biases: K/GROUP per row.
        SlotHandle w  = ctx->heap_alloc((size_t)N * (K / 2));
        SlotHandle sc = ctx->heap_alloc((size_t)N * (K / GROUP) * TSZ);
        SlotHandle bi = ctx->heap_alloc((size_t)N * (K / GROUP) * TSZ);
        SlotHandle x  = ctx->heap_alloc((size_t)K * TSZ);
        SlotHandle y  = ctx->heap_alloc((size_t)N * TSZ);
        SlotHandle ks = ctx->heap_alloc(sizeof(int32_t));
        SlotHandle ns = ctx->heap_alloc(sizeof(int32_t));
        if (!w.valid() || !y.valid()) { printf("  %s: heap OOM\n", s.label); continue; }
        memset(w.contents(), 0, (size_t)N * (K / 2));
        memset(sc.contents(), 0, (size_t)N * (K / GROUP) * TSZ);
        memset(bi.contents(), 0, (size_t)N * (K / GROUP) * TSZ);
        memset(x.contents(), 0, (size_t)K * TSZ);
        *static_cast<int32_t*>(ks.contents()) = (int32_t)K;
        *static_cast<int32_t*>(ns.contents()) = (int32_t)N;
        ctx->arg_bind(Kernel::QmvIn, layer, (uint8_t)bind::Qmv::W,      w);
        ctx->arg_bind(Kernel::QmvIn, layer, (uint8_t)bind::Qmv::Scales, sc);
        ctx->arg_bind(Kernel::QmvIn, layer, (uint8_t)bind::Qmv::Biases, bi);
        ctx->arg_bind(Kernel::QmvIn, layer, (uint8_t)bind::Qmv::X,      x);
        ctx->arg_bind(Kernel::QmvIn, layer, (uint8_t)bind::Qmv::Out,    y);
        ctx->arg_bind(Kernel::QmvIn, layer, (uint8_t)bind::Qmv::K,      ks);
        ctx->arg_bind(Kernel::QmvIn, layer, (uint8_t)bind::Qmv::N,      ns);
        ctx->make_resident();
        // qmv_fast: tg=(32,2,1)=64 threads; grid threads=(32, N/4, 1) -> N/8 threadgroups.
        BenchResult r = h.bench_kernel(s.label, qmv, Kernel::QmvIn, layer,
                                       Grid{32, N / 4, 1}, Threadgroup{32, 2, 1});
        double amort = batched_exec_ms(*ctx, qmv, Kernel::QmvIn, layer,
                                       Grid{32, N / 4, 1}, Threadgroup{32, 2, 1});
        print_result(r);
        printf("    -> amortized compute+barrier: %.4f ms/dispatch\n", amort);
        ++layer;
    }

    printf("\nMICROBENCH_OK\n");
    return 0;
}
