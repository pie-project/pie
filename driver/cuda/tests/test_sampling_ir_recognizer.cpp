// driver-side program → kind recognizer test (program_recognizer.hpp).
// Host-only: pure hashing + table lookup, no GPU. Proves the driver
// self-recognizes its OWN baked standard programs (round-trip) with distinct
// hashes, and that a non-standard program falls through to CustomJIT (nullopt).
// #25: ALL six standard kinds are k-invariant → recognized by EXACT hash.

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <span>
#include <vector>

#include "sampling_ir/program_identity.hpp"
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

// Cross-language oracle: each baked program's fnv1a64 == foxtrot's V=151936
// standard_program_bytecode.txt fixture row (the SDK↔driver byte-parity pin).
std::uint64_t oracle_hash(StandardSamplerKind k) {
    switch (k) {
        case StandardSamplerKind::Argmax:      return 0x91069124d6dda1c5ull;
        case StandardSamplerKind::Temperature: return 0x7d841977776bbb2dull;
        case StandardSamplerKind::MinP:        return 0x9445890b01721734ull;
        case StandardSamplerKind::TopP:        return 0xfdebb8135fe248e7ull;
        case StandardSamplerKind::TopK:        return 0x6f4df2250039a15aull;  // #25 value-id form
        case StandardSamplerKind::TopKTopP:    return 0x0a22af3db5c47474ull;  // #25 value-id form
    }
    return 0;
}
}  // namespace

int main() {
    const std::uint32_t V = 151936;  // the baked qwen3 vocab
    const std::vector<StandardKindEntry> table = build_standard_kind_table(V);

    // #25: all six standard kinds are baked + k-invariant for this V.
    CHECK(table.size() == 6);

    // All baked-program hashes are distinct — no collision aliases two kinds.
    for (std::size_t i = 0; i < table.size(); ++i)
        for (std::size_t j = i + 1; j < table.size(); ++j)
            CHECK(table[i].hash != table[j].hash);

    // Round-trip + cross-language oracle: each baked program recognizes back to
    // its own kind by EXACT hash, and that hash == foxtrot's fixture row.
    for (StandardSamplerKind k : {StandardSamplerKind::Argmax,
                                  StandardSamplerKind::Temperature,
                                  StandardSamplerKind::MinP,
                                  StandardSamplerKind::TopP,
                                  StandardSamplerKind::TopK,
                                  StandardSamplerKind::TopKTopP}) {
        const StandardSamplerProgram p = standard_sampler_program(k, V);
        CHECK(p.valid);
        if (!p.valid) continue;
        CHECK(jit::fnv1a64(p.bytecode, p.len) == oracle_hash(k));
        const auto got = recognize_standard_kind(table, p.bytecode, p.len);
        CHECK(got.has_value());
        if (got) CHECK(*got == k);
    }

    // k-invariance: a per-k wire top-k program (any k value) is byte-identical to
    // the baked one — k rides the submit binding, not the bytecode — so it always
    // EXACT-matches TopK regardless of k. (The bytecode carries the k value-id, not
    // the k value; nothing to vary here. Proven cross-language in the EDSL fixture.)
    {
        const StandardSamplerProgram tk = standard_sampler_program(StandardSamplerKind::TopK, V);
        CHECK(tk.valid);
        const auto got = recognize_standard_kind(table, tk.bytecode, tk.len);
        CHECK(got.has_value() && got == StandardSamplerKind::TopK);
    }

    // Negative: a syntactically-plausible but non-standard program → CustomJIT.
    const std::uint8_t junk[] = {'P', 'S', 'I', 'R', 4, 0, 0, 0,
                                 1, 0, 0, 0, 7, 7, 7, 7};
    CHECK(!recognize_standard_kind(table, junk, sizeof(junk)).has_value());

    // Empty / null → CustomJIT (no spurious match).
    CHECK(!recognize_standard_kind(table, nullptr, 0).has_value());
    CHECK(!recognize_standard_kind(table, junk, 0).has_value());

    // ── #10 cross-lang PIN: program_identity_hash goldens (program_identity.hpp) ──
    // The single-source key shared by #11 compile-dedup / echo M-batch grouping /
    // alpha's #10 distinct-count. alpha's Rust mirror pins to these EXACT vectors
    // (no silent C++↔Rust drift — especially the load-bearing intrinsic row).
    {
        const std::uint8_t bc[] = {0x50, 0x53, 0x49, 0x52, 0x04, 0x00, 0x00, 0x00,
                                   0x01, 0x00, 0x00, 0x00, 0x33, 0x00, 0x00, 0x00};
        std::span<const std::uint8_t> bytes(bc, sizeof(bc));
        const InputBind logits{BindKind::Logits, 0, HostAvailability::SubmitBound};
        InputBind mtp{BindKind::Logits, 0, HostAvailability::SubmitBound};
        mtp.intrinsic_kind = Intrinsic::MtpLogits;
        const InputBind ht0{BindKind::HostTensor, 0, HostAvailability::SubmitBound};
        const InputBind ht1{BindKind::HostTensor, 1, HostAvailability::SubmitBound};
        const InputBind ht1_late{BindKind::HostTensor, 1, HostAvailability::LateBound};

        CHECK(program_identity_hash(bytes, ProgramManifest{}) == 0x28bfe0a3f1d0e019ull);
        CHECK(program_identity_hash(bytes, ProgramManifest{logits}) == 0xff5b1c59d84d9124ull);
        // gotcha-(a): MtpLogits ≡ Logits (intrinsic_kind NOT hashed → dedup to one).
        CHECK(program_identity_hash(bytes, ProgramManifest{mtp}) == 0xff5b1c59d84d9124ull);
        CHECK(program_identity_hash(bytes, ProgramManifest{logits}) ==
              program_identity_hash(bytes, ProgramManifest{mtp}));
        CHECK(program_identity_hash(bytes, ProgramManifest{logits, ht0}) == 0x5f6eac04dece7e45ull);
        CHECK(program_identity_hash(bytes, ProgramManifest{logits, ht0, ht1}) ==
              0x97ff4a9ff574943dull);
        CHECK(program_identity_hash(bytes, ProgramManifest{logits, ht1_late}) ==
              0xcc3a636aa7e8ac37ull);
    }

    if (g_failures == 0) {
        std::fprintf(stderr,
                     "sampling_ir_recognizer: OK (%zu exact kinds)\n", table.size());
        return 0;
    }
    std::fprintf(stderr, "sampling_ir_recognizer: %d failure(s)\n", g_failures);
    return 1;
}
