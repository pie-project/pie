// gemma4_encode_probe.cpp — non-GPU-exclusive verification of the gemma4 executor:
//   1. build_gemma4_dag → dispatch count + stats.
//   2. load_gemma4_psos → runtime-compile EVERY gemma4 kernel (proves each Kernel kind
//      maps to a real, compilable entrypoint — closes the wiki enum/.metal columns).
//   3. walk the DAG → assert every dispatch has a valid PSO + non-degenerate launch dims.
// Runtime newLibraryWithSource compile is non-exclusive (no dispatch) → safe to run while
// the GPU is busy. The bit-exact parity gate vs charlie's golden is the separate GPU step.

#include <cstdio>
#include <string>

#include "gemma4_encode.hpp"
#include "gemma4_decode_step.hpp"

using namespace pie_metal_driver::raw_metal;
using namespace pie_metal_driver::raw_metal::gemma4;

int main() {
    Gemma4Geometry g;  // gemma4-E2B defaults, q_bits=4
    auto dag = build_gemma4_dag(g);
    auto st  = dag_stats(dag, g);
    printf("[gemma4] DAG dispatches=%d  shared_layers=%d  full_attn=%d sliding=%d  gemv=%d\n",
           st.total, st.n_shared_layers, st.n_full_attn, st.n_sliding_attn, st.n_gemv);

    // Concurrency pre-flight (pure, no GPU): barriers emitted per policy vs barrier-each.
    const int b_each   = gemma4_plan_barrier_count(dag, g, 0);
    const int b_gateup = gemma4_plan_barrier_count(dag, g, 1);
    const int b_greedy = gemma4_plan_barrier_count(dag, g, 2);
    printf("[gemma4] barriers: each=%d  +gate||up=%d  greedy=%d  (greedy drops %d, %.1f%%)\n",
           b_each, b_gateup, b_greedy, b_each - b_greedy,
           b_each ? 100.0 * (b_each - b_greedy) / b_each : 0.0);

    auto ctx = RawMetalContext::create(1u << 20);
    if (!ctx) { printf("FAIL: no Metal context\n"); return 1; }

    const char* kdir = RAW_METAL_KERNELS_DIR;
    Gemma4StepPsos psos;
    std::string err;
    if (!load_gemma4_psos(*ctx, kdir, g, psos, /*with_argmax=*/false, &err)) {
        printf("FAIL: load_gemma4_psos: %s\n", err.c_str());
        return 1;
    }
    printf("[gemma4] all PSOs compiled OK\n");

    // Walk: every dispatch must have a valid PSO (except optional Argmax) + sane dims.
    int bad = 0;
    for (const auto& d : dag) {
        if (d.kind == gemma4::Kernel::Argmax) continue;  // optional, not yet ported
        if (!psos[d.kind].valid()) {
            printf("  MISSING PSO for kind=%d (ord=%d layer=%d)\n",
                   int(d.kind), d.ordinal, d.layer);
            ++bad; continue;
        }
        Grid grid; Threadgroup tg;
        gemma4_launch_dims(d.kind, d.layer, g, grid, tg);
        const bool dims_ok = grid.x >= 1 && grid.y >= 1 && grid.z >= 1 &&
                             tg.x >= 1 && tg.y >= 1 && tg.z >= 1 &&
                             tg.x * tg.y * tg.z <= 1024;
        if (!dims_ok) {
            printf("  BAD DIMS kind=%d ord=%d grid(%u,%u,%u) tg(%u,%u,%u)\n",
                   int(d.kind), d.ordinal, grid.x, grid.y, grid.z, tg.x, tg.y, tg.z);
            ++bad;
        }
    }
    if (bad) { printf("GEMMA4_ENCODE_FAIL %d\n", bad); return 1; }

    // Per-layer-type dim spot-checks (catch a regression to uniform geometry): full-attn
    // layers must launch wider q/o (head_dim 512) than sliding (256), and the double-wide
    // MLP range (>=first_kv_shared) must launch 2x the gate/up width.
    auto qmv_grid = [&](gemma4::Kernel k, int L) {
        Grid gr; Threadgroup t; gemma4_launch_dims(k, L, g, gr, t);
        return gr.y;  // qmv grid = (1, ceil(N/bn), 1)
    };
    const uint32_t q_full = qmv_grid(gemma4::Kernel::QmvQ, 4);   // head_dim 512 -> q 4096
    const uint32_t q_slide = qmv_grid(gemma4::Kernel::QmvQ, 0);  // head_dim 256 -> q 2048
    const uint32_t mlp_wide = qmv_grid(gemma4::Kernel::QmvGate, 15);  // double-wide 12288
    const uint32_t mlp_narrow = qmv_grid(gemma4::Kernel::QmvGate, 0); // 6144
    printf("[gemma4] per-layer dims: q_full.y=%u q_slide.y=%u (expect 2x); "
           "mlp_wide.y=%u mlp_narrow.y=%u (expect 2x)\n",
           q_full, q_slide, mlp_wide, mlp_narrow);
    if (q_full != 2 * q_slide || mlp_wide != 2 * mlp_narrow) {
        printf("GEMMA4_ENCODE_FAIL per-layer dims not differentiated\n");
        return 1;
    }
    printf("GEMMA4_ENCODE_OK (every dispatch: valid PSO + sane launch dims; per-layer dims OK)\n");
    return 0;
}
