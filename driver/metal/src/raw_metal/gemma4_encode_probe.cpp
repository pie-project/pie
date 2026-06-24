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
        gemma4_launch_dims(d.kind, g, grid, tg);
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
    printf("GEMMA4_ENCODE_OK (every dispatch: valid PSO + sane launch dims)\n");
    return 0;
}
