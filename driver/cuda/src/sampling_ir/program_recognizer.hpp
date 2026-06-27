#pragma once

// #12 driver-side program → kind recognizer (WS5 / #15).
//
// The driver SELF-recognizes a carrier-attached sampling program: it hashes the
// incoming program bytecode with the same FNV-1a that is the canonical
// `program_hash` (== `ProgramHandle` == #9 compile-cache key; alpha `8a45d83a`)
// and matches it against the hashes of its OWN baked
// `standard_sampler_program(k,V)` programs. A match → the kind `k` →
// `dispatch_target(k)` (the dispatch core is unchanged; only the kind SOURCE
// swaps params→program). No match → `std::nullopt` → CustomJIT (a genuine custom
// program, or a k-bearing kind whose `RankLe(k)` immediate the op-shape
// recognizer handles).
//
// Behavior-preserving by canonical-encode: the SDK's `build_standard(k,V)` emits
// byte-identical bytecode to the driver's `standard_sampler_program(k,V)` (one
// EDSL source), so its hash matches the baked hash → same kind → same
// `dispatch_target` → same kernel → token-identical. The round-trip test proves
// it driver-internally; an additive cross-language test (SDK-emitted bytecode
// hash == baked hash) closes the loop once the EDSL `standard_programs` slice
// lands on dev.
//
// SCOPE — the **4 k-INVARIANT** kinds only (Argmax / Temperature / MinP / TopP):
// their continuous params are host-submit inputs, so each has ONE canonical
// bytecode per V → a stable hash entry. The **2 k-BEARING** kinds
// (TopK / TopKTopP) bake `k` into a `RankLe(k)` immediate → per-k bytecode → no
// single hash → they belong to foxtrot's op-shape recognizer, NOT this table.
// (A kind not yet baked for this V — `standard_sampler_program` invalid — is
// skipped, so the table grows automatically as bakes land, e.g. TopP.)

#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

#include "sampling_ir/pie_standard_samplers.h"  // StandardSamplerKind, standard_sampler_program
#include "sampling_ir/program_hash.hpp"          // jit::fnv1a64 (inline, canonical)

namespace pie_cuda_driver::sampling_ir {

// One `{program hash → kind}` table entry for a baked standard program.
struct StandardKindEntry {
    std::uint64_t       hash;
    StandardSamplerKind kind;
};

// Build the recognizer table for vocab `V`: the FNV-1a of each baked k-invariant
// standard program. Built once at model init (V is fixed per model); the
// per-fire `recognize_standard_kind` is then an O(table) hash compare.
inline std::vector<StandardKindEntry> build_standard_kind_table(std::uint32_t vocab) {
    std::vector<StandardKindEntry> table;
    for (StandardSamplerKind k : {StandardSamplerKind::Argmax,
                                  StandardSamplerKind::Temperature,
                                  StandardSamplerKind::MinP,
                                  StandardSamplerKind::TopP}) {
        const StandardSamplerProgram p = standard_sampler_program(k, vocab);
        if (p.valid) table.push_back({jit::fnv1a64(p.bytecode, p.len), k});
    }
    return table;
}

// Recognize a carrier program by hash. Match → its kind; no match → nullopt
// (→ CustomJIT). `table` is built once via `build_standard_kind_table(V)`.
inline std::optional<StandardSamplerKind>
recognize_standard_kind(const std::vector<StandardKindEntry>& table,
                        const std::uint8_t* bytecode, std::size_t len) {
    if (bytecode == nullptr || len == 0) return std::nullopt;
    const std::uint64_t h = jit::fnv1a64(bytecode, len);
    for (const StandardKindEntry& e : table) {
        if (e.hash == h) return e.kind;
    }
    return std::nullopt;
}

}  // namespace pie_cuda_driver::sampling_ir
