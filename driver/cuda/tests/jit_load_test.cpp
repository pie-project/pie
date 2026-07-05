// #11 JIT compile-axis load-test (lane L3 / delta).
//
// Quantifies the coefficients echo's M-batch cost model + alpha's #10
// accumulation policy consume:
//   * compile µs/program   — the cold NVRTC PTX-gen wall (regime A headline).
//   * finalize µs/program  — the on-context cuModuleLoadData+cuMemAlloc that
//                            prefetch CANNOT hide (the residual at fire).
//   * fire µs/program      — the N-SEQUENTIAL cache-hit launch baseline (the
//                            denominator echo's M-batch fire-collapse measures
//                            against).
//   * dedup               — compiles_run() == K for M requests over K distinct
//                            programs (B regime: identical → 1 compile).
//   * pool-width knee     — regime-A compile wall vs PIE_JIT_POOL_THREADS
//                            (charlie's ~1.9x-at-2 NVRTC bench, on the 4090).
//
// Drives JitEngine directly (no codegen) — builds K distinct programs by baking
// a per-variant constant into a trivial kernel (distinct source → distinct hash
// → fresh NVRTC compile, no dedup). The PTX-gen runs on the off-context pool;
// finalize + launch stay on this (context) thread, per the #11 invariant.

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <string>
#include <thread>
#include <vector>

#include <cuda.h>

#include "sampling_ir/jit.hpp"

using namespace pie_cuda_driver::sampling_ir::jit;
using Clock = std::chrono::steady_clock;

namespace {

void cu_check(CUresult res, const char* expr, int line) {
    if (res != CUDA_SUCCESS) {
        const char* name = nullptr;
        cuGetErrorName(res, &name);
        std::fprintf(stderr, "FATAL %d: %s -> %s\n", line, expr,
                     name ? name : "?");
        std::exit(2);
    }
}
#define CU(expr) cu_check((expr), #expr, __LINE__)

double ms_since(Clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

enum Buf : BufferId { kIn = 0, kOut = 1 };

constexpr unsigned kC = 256;  // vector length (small — compile, not compute, is the cost)

// A DISTINCT program per `variant`: the baked `+ variant` constant makes the
// source, the entry-point name, and the program hash all unique → a fresh NVRTC
// compile every time (no dedup), exactly the burst-of-distinct stress.
KernelDAG build_variant(unsigned variant) {
    KernelDAG dag;
    dag.buffers = {
        {kIn, kC * sizeof(float), ScalarType::F32, /*external=*/false},
        {kOut, kC * sizeof(float), ScalarType::F32, false},
    };

    const std::string name = "psir_load_v" + std::to_string(variant);
    std::string src = "extern \"C\" __global__ void " + name +
                      "(const float* in, float* out, unsigned c) {\n"
                      "  unsigned k = blockIdx.x * blockDim.x + threadIdx.x;\n"
                      "  if (k < c) out[k] = in[k] + " +
                      std::to_string(variant) + ".0f;\n"
                      "}\n";

    KernelDef k;
    k.name = name;
    k.source = src;
    k.grid = {(kC + 255) / 256, 1, 1};
    k.block = {256, 1, 1};
    k.args = {KernelArg::buffer_arg(kIn), KernelArg::buffer_arg(kOut),
              KernelArg::u32(kC)};
    dag.kernels = {k};

    dag.hash = fnv1a64(src.data(), src.size());
    return dag;
}

// Regime A — burst of K DISTINCT-new programs: prefetch all (parallel NVRTC on
// the pool), then drain (each get_or_compile waits for its PTX + finalizes on
// this context thread). Returns the total compile-wall (ms) and asserts dedup.
double regime_A_compile_wall(JitEngine& engine, unsigned K, std::uint64_t base_compiles) {
    std::vector<KernelDAG> dags;
    dags.reserve(K);
    for (unsigned i = 0; i < K; ++i) dags.push_back(build_variant(1000000u + i));

    const auto t0 = Clock::now();
    for (const auto& d : dags) engine.prefetch_compile(d);  // kick parallel NVRTC
    for (const auto& d : dags) engine.get_or_compile(d);    // drain: wait + finalize
    const double wall = ms_since(t0);

    const std::uint64_t ran = engine.compiles_run() - base_compiles;
    std::fprintf(stderr, "    [dedup] compiles_run delta = %llu (expect K=%u) %s\n",
                 static_cast<unsigned long long>(ran), K,
                 ran == K ? "OK" : "MISMATCH");
    if (ran != K) std::exit(3);
    return wall;
}

}  // namespace

int main() {
    CU(cuInit(0));
    CUdevice dev = 0;
    CU(cuDeviceGet(&dev, 0));
    CUcontext ctx = nullptr;
    CU(cuCtxCreate(&ctx, nullptr, 0, dev));

    std::fprintf(stderr, "==== #11 JIT compile-axis load-test (RTX 4090 / sm_89) ====\n");

    // ── Coefficient 1+2: cold-compile vs finalize-only (the prefetch-hideable
    //    split), averaged over N distinct programs after a warmup that absorbs
    //    one-time NVRTC-library init. Pool default (2). COLD get_or_compile (no
    //    prefetch) = compile+finalize; PREFETCHED get_or_compile (PTX already
    //    produced) = finalize only (the on-context cuModuleLoadData PTX->SASS
    //    JIT, which prefetch cannot hide). The difference ≈ the hideable NVRTC.
    {
        JitEngine engine;
        std::fprintf(stderr, "arch=%s pool(default)\n", engine.arch().c_str());

        // Warmup: one throwaway compile absorbs NVRTC lib init + GPU clock ramp.
        engine.get_or_compile(build_variant(900000));

        constexpr unsigned N = 8;
        double cold_sum = 0.0, fin_sum = 0.0;
        for (unsigned i = 0; i < N; ++i) {
            KernelDAG cold = build_variant(100 + i);
            const auto tc = Clock::now();
            engine.get_or_compile(cold);  // cold: NVRTC compile + finalize
            cold_sum += ms_since(tc);

            KernelDAG warm = build_variant(200 + i);
            engine.prefetch_compile(warm);  // PTX-gen off-thread
            std::this_thread::sleep_for(std::chrono::milliseconds(150));  // let PTX finish
            const auto tf = Clock::now();
            engine.get_or_compile(warm);  // PTX ready → finalize only
            fin_sum += ms_since(tf);
        }
        const double cold_ms = cold_sum / N, finalize_ms = fin_sum / N;
        std::fprintf(stderr, "  [C1] cold get_or_compile (compile+finalize) = %.2f ms/prog (avg of %u)\n", cold_ms, N);
        std::fprintf(stderr, "  [C2] finalize-only (prefetched, on-context)  = %.2f ms/prog (avg of %u)\n", finalize_ms, N);
        std::fprintf(stderr, "  [C1-C2] NVRTC compile (prefetch-hideable)    ~ %.2f ms/prog\n",
                     cold_ms - finalize_ms);
    }

    // ── Coefficient 3: fire µs/program — the N-SEQUENTIAL cache-hit launch
    //    baseline (echo's M-batch fire-collapse numerator is measured vs this).
    {
        JitEngine engine;
        KernelDAG d = build_variant(3);
        CompiledProgram& prog = engine.get_or_compile(d);
        // warm one launch (module/context warmup), then time a sequential burst.
        engine.launch(prog, /*stream=*/0, /*param_values=*/{1, kC, 0});
        CU(cuCtxSynchronize());
        constexpr unsigned kFires = 200;
        const auto t0 = Clock::now();
        for (unsigned i = 0; i < kFires; ++i)
            engine.launch(prog, /*stream=*/0, /*param_values=*/{1, kC, 0});
        CU(cuCtxSynchronize());
        const double per_fire_us = (ms_since(t0) / kFires) * 1000.0;
        std::fprintf(stderr, "  [C3] fire (cache-hit launch, N-sequential)  = %.2f us/prog\n",
                     per_fire_us);
    }

    // ── Coefficient 4: dedup — M prefetch_compile of the SAME program → exactly
    //    ONE NVRTC run (the B-regime / identical-program collapse).
    {
        JitEngine engine;
        KernelDAG same = build_variant(42);
        constexpr unsigned M = 64;
        for (unsigned i = 0; i < M; ++i) engine.prefetch_compile(same);  // M identical
        engine.get_or_compile(same);
        const std::uint64_t ran = engine.compiles_run();
        std::fprintf(stderr, "  [C4] dedup: %u identical prefetch → compiles_run=%llu (expect 1) %s\n",
                     M, static_cast<unsigned long long>(ran),
                     ran == 1 ? "OK" : "MISMATCH");
        if (ran != 1) return 4;
    }

    // ── Coefficient 5: pool-width knee — regime-A compile wall for K distinct
    //    programs vs PIE_JIT_POOL_THREADS ∈ {1,2,4,8}. A fresh JitEngine per
    //    width re-reads the env at construction.
    {
        constexpr unsigned K = 16;
        std::fprintf(stderr, "  [C5] pool-width knee (regime A, K=%u distinct):\n", K);
        double wall1 = 0.0;
        for (unsigned n : {1u, 2u, 4u, 8u}) {
            setenv("PIE_JIT_POOL_THREADS", std::to_string(n).c_str(), /*overwrite=*/1);
            JitEngine engine;
            const double wall = regime_A_compile_wall(engine, K, /*base=*/0);
            if (n == 1) wall1 = wall;
            std::fprintf(stderr, "      pool=%u: wall=%.1f ms (%.2f ms/prog)  speedup vs n=1 = %.2fx\n",
                         n, wall, wall / K, wall1 / wall);
        }
    }

    std::fprintf(stderr, "==== JIT_LOAD_OK ====\n");
    cuCtxDestroy(ctx);
    return 0;
}
