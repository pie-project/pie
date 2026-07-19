#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "decode_psos.hpp"
#include "decode_step.hpp"
#include "heap_bind_metal.hpp"

namespace pie::metal {

inline constexpr int kMultiBatchOrdinalBase = 1024;
inline constexpr int kPrefillOrdinalBase = 2048;
inline constexpr int kPrefillOrdinalStride = 512;

// Per-token table offsets for the sequential prompt stream.  CSR page arrays
// stay global; scalar/token-row ports and logits advance together (the caller
// binds scratch with the matching fixed-width row offset).
struct MbBindOffsets {
    size_t token_row = 0;
    size_t logits_bytes = 0;
};

std::size_t paged_attention_mask_pitch_bytes(
    const DecodeGeometry& geometry);
bool paged_pool_size_supported(
    const DecodeGeometry& geometry,
    std::uint32_t pages);

// The paged DAG has the same dataflow/order as build_decode_dag.  Only kernels
// whose ABI changes for token rows/page tables/state slots use append-only kinds.
std::vector<Dispatch> build_decode_dag_mb(const DecodeGeometry& g, int n_tokens,
                                          int ordinal_base = kMultiBatchOrdinalBase,
                                          bool fuse_residual = false,
                                          bool gdn_prep = true);

std::vector<std::vector<Dispatch>> build_decode_prefill_dags(
    const DecodeGeometry& g, int n_tokens, bool fuse_residual = false,
    bool gdn_prep = true);

void bind_decode_dag_mb(RawMetalContext& ctx, const BoundDecode& b,
                        const std::vector<Dispatch>& dag, const DecodeGeometry& g,
                        const std::vector<SlotHandle>& k_pages,
                        const std::vector<SlotHandle>& v_pages,
                        bool gdn_prep = true,
                        const MbBindOffsets& offsets = {});

void encode_decode_step_mb(StepEncoder& se, const std::vector<Dispatch>& dag,
                           const DecodeStepPsos& base_psos, const MultiBatchPsos& mb_psos,
                           bool force_barriers = false);

void bind_prefill_gdn_state(RawMetalContext& ctx, const BoundDecode& b,
                            const std::vector<Dispatch>& dag, uint32_t slot,
                            bool even);

}  // namespace pie::metal
