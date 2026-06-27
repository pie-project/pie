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

    // ── #12 phase-2: canonical (op-shape) recognition of the k-bearing kinds ──
    const std::vector<StandardKindEntry> canon_table = build_canonical_kind_table(V);
    CHECK(canon_table.size() == 2);  // TopK + TopKTopP baked for V=151936

    // Cross-language oracle: the baked CANONICAL hashes == foxtrot's fixture rows.
    for (const StandardKindEntry& e : canon_table) {
        if (e.kind == StandardSamplerKind::TopK)
            CHECK(e.hash == 0x455494d83191da69ull);
        if (e.kind == StandardSamplerKind::TopKTopP)
            CHECK(e.hash == 0x95f84cc8d4e175ddull);
    }

    // Round-trip + k-invariance: the canonical (k=0) program recognizes; AND a
    // synthesized k=40 variant (patch the RankLe payload at its known offset)
    // canonicalizes to the SAME kind — proving the RankLe-zeroing finds the right
    // immediate — with extract_rank_le_k reading 0 / 40 respectively.
    struct KCase { StandardSamplerKind kind; std::size_t rank_le_off; };
    for (KCase kc : {KCase{StandardSamplerKind::TopK, 58u},
                     KCase{StandardSamplerKind::TopKTopP, 98u}}) {
        const StandardSamplerProgram p = standard_canonical_program(kc.kind, V);
        CHECK(p.valid);
        if (!p.valid) continue;
        auto got0 = recognize_canonical_kind(canon_table, p.bytecode, p.len);
        CHECK(got0.has_value() && got0 == kc.kind);
        CHECK(extract_rank_le_k(p.bytecode, p.len) == std::optional<std::uint32_t>(0u));

        std::vector<std::uint8_t> k40(p.bytecode, p.bytecode + p.len);
        k40[kc.rank_le_off] = 40; k40[kc.rank_le_off + 1] = 0;
        k40[kc.rank_le_off + 2] = 0; k40[kc.rank_le_off + 3] = 0;
        auto got40 = recognize_canonical_kind(canon_table, k40.data(), k40.size());
        CHECK(got40.has_value() && got40 == kc.kind);
        CHECK(extract_rank_le_k(k40.data(), k40.size()) == std::optional<std::uint32_t>(40u));
        // a k-bearing program never EXACT-matches (it's not in the exact table).
        CHECK(!recognize_standard_kind(table, k40.data(), k40.size()).has_value());
    }

    // No cross-table false-match: a k-invariant exact program has no RankLe →
    // canonicalize is a no-op → its hash ≠ the canonical table; and a canonical
    // k-bearing program is not in the exact table.
    {
        const StandardSamplerProgram tp = standard_sampler_program(StandardSamplerKind::TopP, V);
        if (tp.valid)
            CHECK(!recognize_canonical_kind(canon_table, tp.bytecode, tp.len).has_value());
        const StandardSamplerProgram tk = standard_canonical_program(StandardSamplerKind::TopK, V);
        if (tk.valid)
            CHECK(!recognize_standard_kind(table, tk.bytecode, tk.len).has_value());
    }

    if (g_failures == 0) {
        std::fprintf(stderr,
                     "sampling_ir_recognizer: OK (%zu exact + %zu canonical kinds)\n",
                     table.size(), canon_table.size());
        return 0;
    }
    std::fprintf(stderr, "sampling_ir_recognizer: %d failure(s)\n", g_failures);
    return 1;
}
