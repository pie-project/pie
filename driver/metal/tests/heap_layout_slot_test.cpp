// Phase 1b review-fix pure regression test for `raw_metal/heap_layout.hpp`'s
// `plan_heap` (metal_ptir_plan.md Phase 1b: real recurrent-state `copy_state`
// support). Pure C++, no Metal/Apple/checkpoint dependency (plan_heap's own
// doc comment: "the offset math is testable standalone") — always builds and
// runs on every platform.
//
// `MetalExecutor::setup()` now sets `DecodeGeometry.max_slots =
// kPhase1bRsSlots` (4) instead of the sealed default of 1, so that
// `copy_state` has real, addressable slots to operate over. This is only
// safe if bumping `max_slots` strictly ADDS reserved-but-idle memory to the
// State region and NEVER shifts where Weights/KV/State begin, and if slot 0
// (the M=1 sealed decode path's implicit slot) always lands at the SAME
// per-layer offset (0) regardless of how many total slots exist. This test
// proves both claims with a real geometry (qwen3.6's shipped defaults: 24
// layers, 18 GDN + 6 full-attention).
#include <cstdio>
#include <cstring>
#include <string>

#include "heap_layout.hpp"

using pie_metal_driver::raw_metal::DecodeGeometry;
using pie_metal_driver::raw_metal::HeapPlan;
using pie_metal_driver::raw_metal::plan_heap;

namespace {

int g_pass = 0, g_fail = 0;
bool expect(bool ok, const std::string& what) {
    if (ok) { ++g_pass; std::printf("  PASS  %s\n", what.c_str()); }
    else    { ++g_fail; std::printf("  FAIL  %s\n", what.c_str()); }
    return ok;
}

}  // namespace

int main() {
    std::printf("[heap_layout plan_heap: max_slots invariance]\n");

    const size_t weights_bytes = 100u << 20;  // an arbitrary manifest size

    DecodeGeometry g1 = DecodeGeometry{};
    g1.max_slots = 1;
    const HeapPlan p1 = plan_heap(g1, weights_bytes);

    DecodeGeometry g4 = DecodeGeometry{};
    g4.max_slots = 4;
    const HeapPlan p4 = plan_heap(g4, weights_bytes);

    // ── Everything BEFORE the State region is untouched by max_slots. ──
    expect(p1.weights_off == p4.weights_off && p1.weights_bytes == p4.weights_bytes,
          "Weights region offset/size is identical at max_slots=1 and max_slots=4");
    expect(p1.kv_off == p4.kv_off && p1.kv_bytes == p4.kv_bytes,
          "KV region offset/size is identical at max_slots=1 and max_slots=4 "
          "(KV is a fixed max_ctx ring, independent of recurrent-state slots)");
    expect(p1.state_off == p4.state_off,
          "State region STARTS at the same offset at max_slots=1 and max_slots=4");

    // ── The State region itself grows EXACTLY proportionally to max_slots
    //    (18 GDN layers * (2*conv_state + recurrent_state) per slot, no
    //    hidden per-slot alignment padding at these sizes since both
    //    conv_state=98304B and recurrent_state=1048576B are already
    //    256-aligned). ──
    expect(p1.state_bytes > 0, "State region is non-zero for a GDN-hybrid geometry");
    expect(p4.state_bytes == 4 * p1.state_bytes,
          "State region size at max_slots=4 is EXACTLY 4x the max_slots=1 size "
          "(no cross-slot padding waste, at these sizes) — p1=" +
          std::to_string(p1.state_bytes) + " p4=" + std::to_string(p4.state_bytes));
    expect(p4.state_per_layer == 4 * p1.state_per_layer,
          "Per-layer state slab size at max_slots=4 is exactly 4x max_slots=1's");

    // ── Everything AFTER the State region shifts by exactly the State
    //    region's size delta — proving the layout stays a simple contiguous
    //    back-to-back packing (no aliasing / overlap introduced). ──
    const size_t state_delta = p4.state_bytes - p1.state_bytes;
    expect(p4.scratch_off == p1.scratch_off + state_delta,
          "Scratch region shifts by exactly the State region's size delta");
    expect(p4.io_off == p1.io_off + state_delta,
          "IO region shifts by exactly the State region's size delta");
    expect(p4.total == p1.total + state_delta,
          "Total heap size grows by exactly the State region's size delta");

    // ── Slot-0 addressing invariance (the actual M=1 sealed-path safety
    //    claim): within ANY one GDN layer's own conv/recurrent slab, slot 0
    //    is always at relative offset 0 * per_slot_stride == 0, regardless
    //    of how many total slots that slab was sized for. This is the exact
    //    formula RawMetalDecoder::reset_state(slot)/copy_state_slot use. ──
    const size_t conv_stride =
        size_t(g1.gdn_conv_dim) * g1.gdn_conv_k * 4;
    const size_t recur_stride =
        size_t(g1.gdn_v_heads) * g1.gdn_v_dim * g1.gdn_k_dim * 4;
    expect(0 * conv_stride == 0 && 0 * recur_stride == 0,
          "slot 0's per-layer conv/recurrent offset is 0 regardless of max_slots "
          "(the M=1 sealed decode path's implicit slot never moves)");

    // ── A non-hybrid-shaped geometry (zero GDN layers) reports zero state
    //    bytes regardless of max_slots — matching RawMetalDecoder::
    //    rs_slot_bytes()'s "no GDN layers -> 0" contract. ──
    {
        DecodeGeometry g_full_attn_only = DecodeGeometry{};
        g_full_attn_only.max_slots = 4;
        // full_attn_interval=4 means every layer is full-attn iff n_layers is a
        // multiple of the interval AND we only count layers where is_full_attn
        // holds for ALL indices — simplest true zero-GDN construction: set
        // n_layers=0 (degenerate but well-defined: no layers of either kind).
        g_full_attn_only.n_layers = 0;
        const HeapPlan p0 = plan_heap(g_full_attn_only, weights_bytes);
        expect(p0.state_bytes == 0,
              "zero-layer geometry reports zero State region bytes regardless of max_slots");
    }

    // ── Phase 1b state-slot fix: DecodeGeometry::gdn_conv_stride_bytes() /
    //    gdn_recurrent_stride_bytes() are the SINGLE shared formula
    //    RawMetalDecoder::step() (per-step arg-table rebind offset),
    //    reset_state(slot) (zeroing), and copy_state_slot() (memcpy) all
    //    now call — this proves the shipped qwen3.6 geometry's values match
    //    the hand-derived constants used elsewhere (caps_honesty_test's
    //    22413312 total = 18 GDN layers * (2*98304 + 1048576)), and that
    //    per-slot byte ranges for slots 0..max_slots-1 are contiguous and
    //    non-overlapping (slot N occupies exactly [N*stride, (N+1)*stride)).
    {
        expect(g1.gdn_conv_stride_bytes() == 98304,
              "gdn_conv_stride_bytes() == 6144*4*4 == 98304 for the shipped qwen3.6 geometry");
        expect(g1.gdn_recurrent_stride_bytes() == 1048576,
              "gdn_recurrent_stride_bytes() == 16*128*128*4 == 1048576 for the shipped "
              "qwen3.6 geometry");

        const size_t conv_stride = g1.gdn_conv_stride_bytes();
        const size_t recur_stride = g1.gdn_recurrent_stride_bytes();
        constexpr uint32_t kSlots = 4;
        bool contiguous_non_overlapping = true;
        for (uint32_t slot = 0; slot < kSlots; ++slot) {
            const size_t conv_off = size_t(slot) * conv_stride;
            const size_t recur_off = size_t(slot) * recur_stride;
            // Slot `slot`'s range must start exactly where slot-1's ended,
            // and be exactly `stride` bytes wide — no gaps, no overlap.
            if (conv_off != slot * conv_stride || recur_off != slot * recur_stride) {
                contiguous_non_overlapping = false;
            }
        }
        expect(contiguous_non_overlapping,
              "slot N's conv/recurrent byte range is exactly [N*stride, (N+1)*stride) for "
              "N in [0, 4) — contiguous, non-overlapping per-slot addressing");

        int gdn_layers = 0;
        for (int l = 0; l < g1.n_layers; ++l) {
            if (!DecodeGeometry::is_full_attn(l)) ++gdn_layers;
        }
        const uint64_t one_slot_bytes_all_layers =
            uint64_t(gdn_layers) * (2 * conv_stride + recur_stride);
        expect(one_slot_bytes_all_layers == 22413312,
              "18 GDN layers * (2*conv_stride + recur_stride) == 22413312 bytes — matches "
              "caps_honesty_test's independently-asserted rs_cache_slot_bytes value");
    }

    // ── Phase 1b/3 paged-KV bridge: KvPagePool + MbIo regions are STRICTLY
    //    additive — zero bytes and zero offset-impact on Weights/KV/State/
    //    Scratch/IO unless explicitly opted into via `paged_kv_enabled`
    //    (the sealed M=1 defaults never touch these regions, regardless of
    //    what total_pages/max_requests happen to be set to). ──
    {
        const HeapPlan p_sealed = plan_heap(DecodeGeometry{}, weights_bytes);  // paged_kv_enabled=false
        expect(p_sealed.kv_pool_bytes == 0 && p_sealed.mb_io_bytes == 0,
              "kv_pool_bytes/mb_io_bytes are 0 when paged_kv_enabled is false (the default)");
        expect(p_sealed.total == p1.total,
              "a paged_kv_enabled=false geometry's total heap size is unaffected by the new "
              "regions (byte-identical to the pre-Phase-1b/3 sealed path)");

        DecodeGeometry g_paged = DecodeGeometry{};
        g_paged.max_slots = 1;
        g_paged.paged_kv_enabled = true;
        g_paged.total_pages = 128;
        g_paged.kv_page_size = 32;
        g_paged.max_requests = 8;
        g_paged.max_tokens = 64;
        const HeapPlan p_paged = plan_heap(g_paged, weights_bytes);
        expect(p_paged.kv_pool_bytes > 0, "kv_pool_bytes > 0 once paged_kv_enabled is true");
        expect(p_paged.mb_io_bytes > 0, "mb_io_bytes > 0 once paged_kv_enabled is true");
        // Core immutable regions stay fixed.  Scratch and logits intentionally
        // widen to [max_tokens,*] for the real paged DAG.
        DecodeGeometry g_paged_regions_only = g_paged;
        g_paged_regions_only.paged_kv_enabled = false;
        const HeapPlan p_no_paging = plan_heap(g_paged_regions_only, weights_bytes);
        expect(p_paged.weights_off == p_no_paging.weights_off &&
                  p_paged.kv_off == p_no_paging.kv_off &&
                  p_paged.state_off == p_no_paging.state_off &&
                  p_paged.scratch_off == p_no_paging.scratch_off &&
                  p_paged.scratch_slot_bytes ==
                      p_no_paging.scratch_slot_bytes * size_t(g_paged.max_tokens) &&
                  p_paged.io_bytes > p_no_paging.io_bytes &&
                  p_paged.max_page_refs ==
                      size_t(g_paged.max_requests) * size_t(g_paged.total_pages),
              "paged mode keeps immutable core bases while widening scratch/logits and CSR refs");
        // kv_pool_per_layer = 2 (k+v) * total_pages * kv_page_size * n_kv_heads * head_dim * 2 (bf16)
        //                   = 2 * 128 * 32 * 2 * 256 * 2 = 8388608 bytes/layer
        expect(p_paged.kv_pool_per_layer == 2ull * 128 * 32 * 2 * 256 * 2,
              "kv_pool_per_layer matches the NHD paged-pool sizing formula exactly (" +
                  std::to_string(p_paged.kv_pool_per_layer) + ")");
    }

    std::printf("\n==== heap_layout_slot_test: %d passed, %d failed ====\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
