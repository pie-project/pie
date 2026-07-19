// harness_main.cpp — Phase-0 smoke + demo for the raw-Metal scaffold.
//
// Proves the wrapper + harness run end-to-end on THIS box (M1 Max, runtime MSL compile)
// BEFORE delta's real kernel ports land:
//   1. create context + single resident heap
//   2. heap_alloc weight/input/output/scalar slots (delta's signature)
//   3. arg_bind keyed by decode_abi.hpp bind::Qmv enums (delta's signature)
//   4. make_resident ONCE
//   5. micro-bench the demo GEMV (encode-ms vs gpu-exec-ms split)
//   6. multi-dispatch encode-scaling check (reproduces beta's ~0.05ms/322-disp finding)

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

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    std::string kernels_dir = PIE_METAL_TOOL_KERNELS_DIR;
    if (argc > 1) kernels_dir = argv[1];

    // Realistic decode qmv shape: x[K] @ W[N,K] -> out[N], K=hidden, N=q_dim.
    const uint32_t K = 1024;
    const uint32_t N = 2048;

    printf("raw-Metal Phase-0 harness — M1 Max, runtime MSL compile\n");

    auto ctx = RawMetalContext::create(/*heap_bytes=*/256ull << 20);  // 256 MB
    if (!ctx) { printf("FAIL: no context\n"); return 1; }

    // (1) heap_alloc — delta's signature, deterministic bump offsets.
    SlotHandle w   = ctx->heap_alloc((size_t)N * K * sizeof(float));
    SlotHandle x   = ctx->heap_alloc((size_t)K * sizeof(float));
    SlotHandle out = ctx->heap_alloc((size_t)N * sizeof(float));
    SlotHandle ks  = ctx->heap_alloc(sizeof(uint32_t));
    SlotHandle ns  = ctx->heap_alloc(sizeof(uint32_t));
    if (!w.valid() || !x.valid() || !out.valid() || !ks.valid() || !ns.valid()) {
        printf("FAIL: heap_alloc\n");
        return 1;
    }
    printf("heap_alloc OK: W@%zu (%zuB)  X@%zu  Out@%zu  K@%zu  N@%zu\n",
           w.offset, w.size, x.offset, out.offset, ks.offset, ns.offset);

    // stage weights/input via contents() (Shared storage, UMA)
    float* wp = static_cast<float*>(w.contents());
    float* xp = static_cast<float*>(x.contents());
    for (uint32_t i = 0; i < N * K; ++i) wp[i] = 1.0f / (float)((i % 7) + 1);
    for (uint32_t k = 0; k < K; ++k) xp[k] = (float)((k % 3) + 1);
    *static_cast<uint32_t*>(ks.contents()) = K;
    *static_cast<uint32_t*>(ns.contents()) = N;

    // (3) arg_bind — keyed by decode_abi.hpp bind::Qmv enums (delta's signature).
    const Kernel kQmv = Kernel::QmvIn;
    ctx->arg_bind(kQmv, 0, (uint8_t)bind::Qmv::W,   w);
    ctx->arg_bind(kQmv, 0, (uint8_t)bind::Qmv::X,   x);
    ctx->arg_bind(kQmv, 0, (uint8_t)bind::Qmv::Out, out);
    ctx->arg_bind(kQmv, 0, (uint8_t)bind::Qmv::K,   ks);
    ctx->arg_bind(kQmv, 0, (uint8_t)bind::Qmv::N,   ns);

    // (4) make resident ONCE (I2)
    ctx->make_resident();

    // compile the demo kernel at runtime
    std::string err;
    Pso pso = ctx->compile_pso_from_file(kernels_dir + "/gemv_demo.metal", "gemv_demo", &err);
    if (!pso.valid()) { printf("FAIL: pso compile: %s\n", err.c_str()); return 1; }
    printf("runtime MSL compile OK (gemv_demo)\n");

    LatencyHarness h(*ctx);

    // (5) micro-bench the single kernel (encode vs gpu-exec split)
    Grid grid{N, 1, 1};
    Threadgroup tg{256, 1, 1};
    BenchResult mb = h.bench_kernel("gemv_demo[N=2048,K=1024]", pso, kQmv, 0, grid, tg);
    printf("micro-bench (single dispatch):\n");
    print_result(mb);

    // correctness sanity: out[gid] = sum_k W[gid,k]*x[k]
    {
        h.bench_kernel("warm", pso, kQmv, 0, grid, tg, /*iters=*/1, /*warmup=*/0);
        float* op = static_cast<float*>(out.contents());
        double ref = 0.0;
        for (uint32_t k = 0; k < K; ++k) ref += (1.0 / ((k % 7) + 1)) * xp[k];  // row 0: W[0,k]
        double diff = std::abs((double)op[0] - ref);
        printf("correctness: out[0]=%.4f ref=%.4f diff=%.2e  %s\n",
               op[0], ref, diff, diff < 1e-1 ? "OK" : "MISMATCH");
    }

    // (6) multi-dispatch encode-scaling (beta's ~322-disp re-encode regime)
    printf("encode scaling (reproduces beta's MTL4 re-encode finding):\n");
    for (int ndisp : {50, 100, 200, 300, 322}) {
        auto encode_fn = [&](StepEncoder& se) {
            se.set_pso(pso);
            for (int d = 0; d < ndisp; ++d) {
                se.set_argtable(kQmv, 0);
                se.dispatch(grid, tg);
                se.barrier();
            }
        };
        BenchResult r = h.time_step("ndisp=" + std::to_string(ndisp), encode_fn, 150, 40);
        print_result(r);
    }

    printf("PHASE0_HARNESS_OK\n");
    return 0;
}
