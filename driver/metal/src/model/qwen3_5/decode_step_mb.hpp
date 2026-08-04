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
// Ordinals claimed by ONE prefill row's DAG. It must exceed the DAG's own
// length, and the DAG's length is the model's: a 24-layer dense Qwen3.5 emits
// about 260 dispatches per row, and the 40-layer 256-expert MoE emits 1083 --
// which walked straight through a stride of 512 and refused to prefill at all.
// Twenty-seven dispatches per layer is the rate, so 4096 holds a decoder of
// about 150 layers. The space itself is free: argument tables live in a map
// keyed by ordinal and are created on demand, so a wide stride costs nothing
// that is not used. `qwen3_5_geometry_test` asserts the routed DAG fits.
inline constexpr int kPrefillOrdinalStride = 4096;
// One past the highest ordinal a prefill row can claim.  The PTIR runtime
// allocates its own ordinals and must start at or above this: the two spaces
// were separated only by both being small, and raising the rows-per-fire bound
// from 64 to 512 walked the prefill range straight through PTIR's base at
// 100000, which showed up as "no argument table bound for ordinal=100038".
// Deriving one from the other means they cannot silently overlap again.
//
// Raised 512 -> 1024 with `kPagedMaxForwardTokensCeiling`, because rows are the
// longest prompt this driver accepts and a 650-token one was being refused. The
// ordinal space itself is free -- argument tables are created on demand, keyed
// by ordinal -- and the per-row prefill DAGs measured free too: 566 rows and
// 1024 rows give the same 2.35GB peak RSS and the same wall clock.
inline constexpr int kPrefillOrdinalMaxRows = 1024;  // == executor::kPagedMaxForwardTokensCeiling
inline constexpr int kPrefillOrdinalLimit =
    kPrefillOrdinalBase + kPrefillOrdinalMaxRows * kPrefillOrdinalStride;

// Per-token table offsets for the sequential prompt stream.  CSR page arrays
// stay global; scalar/token-row ports and logits advance together (the caller
// binds scratch with the matching fixed-width row offset).
struct MbBindOffsets {
    size_t token_row = 0;
    size_t logits_bytes = 0;
};

// Output width of a projection kernel, or 0 if the kind is not one.
int qmv_out_size(Kernel k, const DecodeGeometry& g);

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

// Encodes every prompt token's DAG dispatch-major rather than token-major, so
// each weight tensor is read once for the whole prompt. Falls back to
// token-major if the per-token DAGs are not the same shape.
// `row_needs_logits`, when non-empty, is one byte per token: rows whose logits
// nothing will read skip the lm_head projection and the argmax that follows it.
// A run of prompt rows that share one recurrent slot, and how many of them the
// GDN scan covers.  A grouped prefill carries several requests, and each one's
// recurrence is independent, so each gets its own scan.
struct GdnScanSegment {
    int start = 0;  // first prompt row of the request
    int rows = 0;   // rows the scan covers (the rest stay per-token)
};

void encode_prefill_dags_mb(StepEncoder& se,
                            const std::vector<std::vector<Dispatch>>& dags,
                            int n_tokens,
                            const DecodeStepPsos& base_psos,
                            const MultiBatchPsos& mb_psos,
                            bool force_barriers,
                            const std::vector<std::uint8_t>& row_needs_logits = {},
                            const DecodeGeometry* geometry = nullptr,
                            int max_rows = 0,
                            const std::vector<GdnScanSegment>& gdn_scans = {});

// Point ConvStateOut at ConvState so a paged decode shifts the conv history in
// place; the prefill re-binds its own ordinals per fire and is unaffected.
void alias_decode_conv_state_out(RawMetalContext& ctx, const BoundDecode& b,
                                 const std::vector<Dispatch>& dag);

// Interleaved A/B.  This machine is permanently contended -- the agent process
// alone runs at ~250% CPU and macOS daemons spike on top of it -- so the same
// binary measures anywhere from 11.5s to 23s and no before/after comparison
// across runs means anything.  Alternating the arms fire by fire inside ONE run
// puts both under identical conditions at millisecond granularity, which is the
// only way left to decide a change here.  `ab_arm()` is read wherever the two
// arms differ; `ab_enabled()` says whether to alternate at all.
bool ab_enabled();
// Whether arm B of the A/B also serializes every dispatch (PIE_METAL_AB_BARRIERS).
bool ab_all_barriers();
bool ab_arm();
void ab_set_arm(bool b);

void encode_decode_step_mb(StepEncoder& se, const std::vector<Dispatch>& dag,
                           const DecodeStepPsos& base_psos, const MultiBatchPsos& mb_psos,
                           bool force_barriers = false);

void bind_prefill_gdn_state(RawMetalContext& ctx, const BoundDecode& b,
                            const std::vector<Dispatch>& dag, uint32_t slot,
                            bool even);

}  // namespace pie::metal
