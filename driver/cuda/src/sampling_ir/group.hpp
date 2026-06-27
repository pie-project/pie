#pragma once

// Cross-request batching (#10) grouping helpers — a SEPARABLE module, decoupled
// from the executor mode-select (echo's keystone) so the two land without
// colliding. echo wires these into the per-fire sampling dispatch after the
// keystone establishes `dispatch_target`.
//
// The grouping key is the `get_or_compile(bytecode, manifest)` ProgramHandle =
// the canonical bytecode-hash (`hash(bc) ^ manifest_hash`), which is ALSO the #9
// cache key and the #8 dispatch fast-path key — one mechanism, three uses. Rows
// whose programs hash-equal form one group → one batched `[N,V]` launch.
//
//   partition_by_program (CPU) — group a fire's rows by program identity.
//   gather_logits_bf16 / scatter_tokens_i32 (device) — marshal a group's
//   SCATTERED rows into a compact `[Ng,V]` for the batched launch, then scatter
//   the `[Ng]` result back to the original rows.
//
// #10 ships WITH the gather (it wins on launch-overhead anyway: N tiny `[1,V]`
// launches → one `[N,V]`, dominating ~one extra N·V bf16 read+write). The codegen
// row-index indirection (`logits[idx[r]*V+j]`, no copy) that elides the gather is
// #11's first perf item, not #10.

#include <cstddef>
#include <cstdint>
#include <span>
#include <unordered_map>
#include <vector>

#include <cuda_runtime.h>

#include "sampling_ir/runtime.hpp"  // ProgramHandle, kInvalidProgram

namespace pie_cuda_driver::sampling_ir {

// A set of fire rows sharing one program (one batched launch target).
struct ProgramGroup {
    ProgramHandle handle = kInvalidProgram;
    std::vector<std::uint32_t> rows;  // original fire-row indices, in fire order
};

// Partition sampling rows by program identity. `row_handles[r]` is the handle
// from `get_or_compile` for row r's resolved program; `kInvalidProgram` skips the
// row (e.g. a DedicatedKernel row routed to the legacy ladder, not the IR path).
// Groups, and rows within each group, are in FIRST-SEEN order — deterministic, so
// the scatter preserves a stable token order across runs.
inline std::vector<ProgramGroup>
partition_by_program(std::span<const ProgramHandle> row_handles) {
    std::vector<ProgramGroup> groups;
    std::unordered_map<ProgramHandle, std::size_t> handle_to_group;
    for (std::uint32_t r = 0; r < row_handles.size(); ++r) {
        const ProgramHandle h = row_handles[r];
        if (h == kInvalidProgram) continue;  // non-IR row
        auto it = handle_to_group.find(h);
        std::size_t gi;
        if (it == handle_to_group.end()) {
            gi = groups.size();
            handle_to_group.emplace(h, gi);
            groups.push_back(ProgramGroup{h, {}});
        } else {
            gi = it->second;
        }
        groups[gi].rows.push_back(r);
    }
    return groups;
}

// A 1-row group gathers/batches nothing — launch it directly at its source row
// offset (no copy). Groups at/above this gather to compact `[Ng,V]` + one batched
// launch, amortizing the copy. The exact crossover is profile-tuned on #10's
// SCATTERED-group bench (gather cost included); 2 is the safe floor — a size-1
// group can never benefit from gather+batch.
constexpr std::size_t kMinGatherGroup = 2;

// True when a group should be gathered into a compact `[Ng,V]` and launched
// batched; false for a degenerate group that should launch in place.
inline bool should_gather(const ProgramGroup& g) {
    return g.rows.size() >= kMinGatherGroup;
}

// True when a group's rows are already contiguous in the source `[*, V]` (rows ==
// base, base+1, … base+Ng-1) — the keystone/pure-decode fast path: pass the source
// base + `rows[0]*V` directly to the batched launch, no gather copy. A scattered
// group (cross-request interleaving) is false → gather to compact `[Ng,V]`.
inline bool is_contiguous(const ProgramGroup& g) {
    for (std::size_t i = 1; i < g.rows.size(); ++i) {
        if (g.rows[i] != g.rows[i - 1] + 1) return false;
    }
    return true;
}

// Device: gather a group's bf16 logits rows from a scattered `[*, V]` source into
// a compact `[rows.size(), V]` destination (one D2D copy per row). The #11
// indirection (`logits[idx[r]*V+j]` in codegen) elides this copy. Async on
// `stream`; the caller orders the batched launch after it on the same stream.
// Pointers are device addresses (runtime-API `void*`, matching cudaMemcpyAsync).
void gather_logits_bf16(const void* src_base, std::span<const std::uint32_t> rows,
                        std::uint32_t vocab, void* dst, cudaStream_t stream);

// Device: gather a group's per-row SCALAR params from a scattered `[*]` source
// into a compact `[rows.size()]` destination (one 4-byte element per row) — the
// #10-phase-2 marshalling kit. A scattered group launched over compact `[Ng,V]`
// logits needs its per-row params (seed/temp/top_p/min_p) gathered to the SAME
// compact order, else compact row r reads the wrong row's param (the input half
// of the logits gather). `gather_f32` for temp/top_p/min_p; `gather_u32` for
// seed (and the bit-compatible i32 top_k). Async on `stream`. (FlashInfer
// top-k/p self-gather via `sample_idx`, so they take no explicit gather.)
void gather_f32(const void* src_base, std::span<const std::uint32_t> rows,
                void* dst, cudaStream_t stream);
void gather_u32(const void* src_base, std::span<const std::uint32_t> rows,
                void* dst, cudaStream_t stream);

// Device: scatter a group's compact `[rows.size()]` i32 tokens back to the
// original rows of a `[*]` destination (e.g. `pi.sampled`). Async on `stream`.
void scatter_tokens_i32(const void* src_compact, std::span<const std::uint32_t> rows,
                        void* dst_base, cudaStream_t stream);

}  // namespace pie_cuda_driver::sampling_ir
