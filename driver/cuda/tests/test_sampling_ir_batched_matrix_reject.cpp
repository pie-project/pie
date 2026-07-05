// Lane L2 / charlie (#35-A regression guard): the batched (M>1) lowering path
// must REJECT matrix (per-row / cross-row) programs so the backend falls back to
// the M=1 per-row matrix path. Batched lowering treats every op as per-batch-row
// independent work (one block per batch row) — folding a `[k, vocab]` matrix
// program into that conflates the matrix's per-row (k draft positions) axis with
// the batch axis and silently miscompiles cross-row ops (e.g. spec-verify's
// `cumprod` accept-scan collapses to per-block, leaking `[d0,d1,-1,d3]` instead
// of the prefix `[d0,d1,-1,-1]`). The guard: matrix goldens reject batched (→ M=1
// fallback gives the correct Custom-grid=k DAG); vector samplers still batch.
#include <cstdio>

#include "sampling_ir/codegen.hpp"
#include "sampling_ir_golden_bytecode.h"

using namespace pie_cuda_driver::sampling_ir;

namespace {
int g_fail = 0;
#define EXPECT(cond, msg)                                              \
    do {                                                              \
        if (!(cond)) { std::fprintf(stderr, "FAIL: %s\n", (msg)); ++g_fail; } \
    } while (0)

LowerResult lw(const unsigned char* d, std::size_t n, bool batched) {
    return lower_bytecode(reinterpret_cast<const std::uint8_t*>(d), n,
                          LowerOptions{/*batched=*/batched});
}
}  // namespace

int main() {
    // Matrix spec-verify (cross-row argmax→eq→cumprod→select): batched REJECTED;
    // M=1 lowers to 2 kernels (matrix per-row argmax + the cross-row accept-scan).
    {
        LowerResult b = lw(GV_SPECGREEDY, sizeof(GV_SPECGREEDY), /*batched=*/true);
        EXPECT(!b.ok, "GV_SPECGREEDY must be REJECTED by batched lowering");
        LowerResult m = lw(GV_SPECGREEDY, sizeof(GV_SPECGREEDY), /*batched=*/false);
        EXPECT(m.ok, "GV_SPECGREEDY must lower on the M=1 path");
        EXPECT(m.dag.kernels.size() == 2,
               "GV_SPECGREEDY M=1 = 2 kernels (matrix argmax + cross-row scan)");
    }
    // Matrix per-row argmax (embarrassingly parallel, but still a matrix program):
    // batched REJECTED; M=1 lowers to a single Custom-grid=rows matrix kernel.
    {
        LowerResult b = lw(GV_MATARGMAX, sizeof(GV_MATARGMAX), /*batched=*/true);
        EXPECT(!b.ok, "GV_MATARGMAX must be REJECTED by batched lowering");
        LowerResult m = lw(GV_MATARGMAX, sizeof(GV_MATARGMAX), /*batched=*/false);
        EXPECT(m.ok, "GV_MATARGMAX must lower on the M=1 path");
        EXPECT(m.dag.kernels.size() == 1, "GV_MATARGMAX M=1 = 1 matrix kernel");
        EXPECT(!m.dag.kernels.empty() &&
                   m.dag.kernels[0].shape == LaunchShape::Custom,
               "GV_MATARGMAX M=1 kernel is Custom (grid = matrix rows)");
    }
    // No regression: vector samplers ([vocab] intrinsic, no matrix work) must
    // STILL lower on the batched (M>1) fast path.
    {
        LowerResult a = lw(GV_ARGMAX, sizeof(GV_ARGMAX), /*batched=*/true);
        EXPECT(a.ok, "GV_ARGMAX (vector) must still lower batched");
        LowerResult t = lw(GV_TEMP, sizeof(GV_TEMP), /*batched=*/true);
        EXPECT(t.ok, "GV_TEMP (vector) must still lower batched");
    }

    if (g_fail == 0) {
        std::fprintf(stderr, "sampling_ir_batched_matrix_reject: OK\n");
        return 0;
    }
    std::fprintf(stderr, "sampling_ir_batched_matrix_reject: %d failure(s)\n", g_fail);
    return 1;
}
