// CPU unit test for the #10 cross-request grouping partition (no GPU needed —
// pure host logic). Verifies group-by-program-handle: first-seen ordering,
// invalid-row skipping, the min-gather threshold, and degenerate cases.

#include <cstdio>
#include <cstdlib>
#include <vector>

#include "sampling_ir/group.hpp"

using namespace pie_cuda_driver::sampling_ir;

namespace {
int g_failures = 0;
#define CHECK(cond)                                                            \
    do {                                                                       \
        if (!(cond)) {                                                         \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
            ++g_failures;                                                      \
        }                                                                      \
    } while (0)
}  // namespace

int main() {
    const ProgramHandle A = 111, B = 222, C = 333;

    // Mixed fire: rows from 3 programs interleaved (A,B,A,C,B,A) + a skipped
    // (non-IR / DedicatedKernel) row at the end.
    std::vector<ProgramHandle> h = {A, B, A, C, B, A, kInvalidProgram};
    auto g = partition_by_program(std::span<const ProgramHandle>(h.data(), h.size()));

    CHECK(g.size() == 3);  // 3 distinct programs
    // First-seen order: A, B, C.
    CHECK(g[0].handle == A);
    CHECK((g[0].rows == std::vector<std::uint32_t>{0, 2, 5}));
    CHECK(g[1].handle == B);
    CHECK((g[1].rows == std::vector<std::uint32_t>{1, 4}));
    CHECK(g[2].handle == C);
    CHECK((g[2].rows == std::vector<std::uint32_t>{3}));

    // The kInvalidProgram row (index 6) is skipped → 6 IR rows total.
    std::size_t total = 0;
    for (const auto& gr : g) total += gr.rows.size();
    CHECK(total == 6);

    // Threshold: C (size 1) launches in place; A/B (>=2) gather + batch.
    CHECK(!should_gather(g[2]));
    CHECK(should_gather(g[0]));
    CHECK(should_gather(g[1]));

    // Contiguity (contiguous → in-place fast path; scattered → gather): the
    // interleaved groups A {0,2,5} / B {1,4} are scattered; a size-1 group is
    // trivially contiguous.
    CHECK(!is_contiguous(g[0]));
    CHECK(!is_contiguous(g[1]));
    CHECK(is_contiguous(g[2]));

    // All-same → one group (the high-concurrency common case: many requests,
    // same sampler → one big batched launch).
    std::vector<ProgramHandle> h2 = {A, A, A, A};
    auto g2 = partition_by_program(std::span<const ProgramHandle>(h2.data(), h2.size()));
    CHECK(g2.size() == 1);
    CHECK(g2[0].rows.size() == 4);
    CHECK(should_gather(g2[0]));
    CHECK(is_contiguous(g2[0]));  // {0,1,2,3} → contiguous, in-place fast path

    // Empty fire → no groups.
    auto g3 = partition_by_program(std::span<const ProgramHandle>{});
    CHECK(g3.empty());

    // All-invalid (every row on the legacy ladder) → no IR groups.
    std::vector<ProgramHandle> h4 = {kInvalidProgram, kInvalidProgram};
    auto g4 = partition_by_program(std::span<const ProgramHandle>(h4.data(), h4.size()));
    CHECK(g4.empty());

    if (g_failures == 0) {
        std::fprintf(stderr, "sampling_ir_group: OK\n");
        return 0;
    }
    std::fprintf(stderr, "sampling_ir_group: %d failure(s)\n", g_failures);
    return 1;
}
