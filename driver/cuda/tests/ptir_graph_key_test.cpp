// PTIR tier-2 graph-key extension test. Pure-host unit
// test for the program-set hash (contract C3) added to ForwardGraphKey: proves
// (a) the non-PTIR path is unchanged (default 0), (b) different program sets key
// to different graphs, (c) the fold is order-independent + duplicate-insensitive
// (batching identity is a SET of stage-program traces, T5). No GPU.

#include <cstdio>
#include <cstdint>
#include <vector>

#include "batch/forward_graph.hpp"

using namespace pie_cuda_driver;

namespace {
int g_pass = 0, g_fail = 0;
void expect(bool ok, const char* what) {
    if (ok) { ++g_pass; std::printf("  PASS  %s\n", what); }
    else    { ++g_fail; std::printf("  FAIL  %s\n", what); }
}
}

int main() {
    std::printf("PTIR tier-2 graph-key (program-set hash / C3)\n");
    ForwardGraphKeyHash H;

    // (a) backward-compat: 3-field aggregate init leaves program_set_hash = 0,
    // identical to today's key.
    ForwardGraphKey legacy{4, 4, 7};
    ForwardGraphKey explicit0{4, 4, 7, 0};
    expect(legacy == explicit0, "3-field init defaults program_set_hash to 0");
    expect(legacy.program_set_hash == 0, "non-PTIR key has zero program-set hash");
    expect(H(legacy) == H(explicit0), "hash agrees for the default");

    // (b) same shape, different program set → distinct key + (very likely) hash.
    std::uint64_t setA = make_program_set_hash({0x1111, 0x2222});
    std::uint64_t setB = make_program_set_hash({0x1111, 0x3333});
    ForwardGraphKey ka{4, 4, 7, setA};
    ForwardGraphKey kb{4, 4, 7, setB};
    expect(!(ka == kb), "different program sets → different keys");
    expect(H(ka) != H(kb), "different program sets → different hashes");
    expect(!(ka == legacy), "PTIR key != non-PTIR key of same shape");

    // (c) order-independent + duplicate-insensitive (it's a SET).
    expect(make_program_set_hash({0xAA, 0xBB, 0xCC}) ==
           make_program_set_hash({0xCC, 0xAA, 0xBB}), "fold is order-independent");
    expect(make_program_set_hash({0xAA, 0xBB, 0xAA}) ==
           make_program_set_hash({0xAA, 0xBB}), "fold dedups duplicate programs");
    expect(make_program_set_hash({}) == 0, "empty set folds to 0 (== non-PTIR)");
    expect(make_program_set_hash({0xAA}) != make_program_set_hash({0xBB}),
           "distinct singletons fold distinctly");
    // a single-program fleet vs a two-program fleet must differ.
    expect(make_program_set_hash({0xAA}) != make_program_set_hash({0xAA, 0xBB}),
           "adding a program changes the set identity");

    std::printf("\n==== graph-key C3: %d passed, %d failed ====\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
