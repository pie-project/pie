// #12 driver-side program → kind recognizer test (program_recognizer.hpp).
// Host-only: pure hashing + table lookup, no GPU. Proves the driver
// self-recognizes its OWN baked standard programs (round-trip) with distinct
// hashes, and that a non-standard program falls through to CustomJIT (nullopt).

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <vector>

#include "sampling_ir/program_recognizer.hpp"

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
    const std::uint32_t V = 151936;  // the baked qwen3 vocab
    const std::vector<StandardKindEntry> table = build_standard_kind_table(V);

    // The k-invariant kinds baked for this V (Argmax/Temperature/MinP now; TopP
    // joins automatically once its program is baked). At least the 3 current.
    CHECK(table.size() >= 3);

    // All baked-program hashes are distinct — no collision aliases two kinds.
    for (std::size_t i = 0; i < table.size(); ++i)
        for (std::size_t j = i + 1; j < table.size(); ++j)
            CHECK(table[i].hash != table[j].hash);

    // Round-trip: each baked k-invariant program recognizes back to its own kind
    // (the enum→graph behavior-preserving property, driver-internal).
    for (StandardSamplerKind k : {StandardSamplerKind::Argmax,
                                  StandardSamplerKind::Temperature,
                                  StandardSamplerKind::MinP,
                                  StandardSamplerKind::TopP}) {
        const StandardSamplerProgram p = standard_sampler_program(k, V);
        if (!p.valid) continue;  // not baked for this V yet (e.g. TopP) — skip
        const auto got = recognize_standard_kind(table, p.bytecode, p.len);
        CHECK(got.has_value());
        if (got) CHECK(*got == k);
    }

    // Negative: a syntactically-plausible but non-standard program → CustomJIT.
    const std::uint8_t junk[] = {'P', 'S', 'I', 'R', 4, 0, 0, 0,
                                 1, 0, 0, 0, 7, 7, 7, 7};
    CHECK(!recognize_standard_kind(table, junk, sizeof(junk)).has_value());

    // Empty / null → CustomJIT (no spurious match).
    CHECK(!recognize_standard_kind(table, nullptr, 0).has_value());
    CHECK(!recognize_standard_kind(table, junk, 0).has_value());

    if (g_failures == 0) {
        std::fprintf(stderr, "sampling_ir_recognizer: OK (%zu baked kinds)\n",
                     table.size());
        return 0;
    }
    std::fprintf(stderr, "sampling_ir_recognizer: %d failure(s)\n", g_failures);
    return 1;
}
