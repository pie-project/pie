// Oracle for `model::FirePageMask` and `prepare_page_mask_capture`.
//
// Compiles the REAL `model/attn_page_mask.cu` and the REAL
// `model/hook_sideband_arena.cpp` and drives them over a grid of fire
// geometries. Everything else is stubbed: the four CUDA entry points and the
// compaction kernel are defined here, so the run needs no GPU and — more
// usefully — every device-facing call becomes an observable event.
//
// # What the transcript is for
//
// The class is a carve: five buffers cut from one arena slot at fixed offsets.
// Two separate call paths perform that carve — `FirePageMask`'s constructor at
// fire time, and `prepare_page_mask_capture` during the hook-graph prepare
// pass — and the prepare pass BAKES the addresses it computes into a captured
// CUDA graph. If the two ever disagree by a byte, the replayed graph writes
// its compacted page table somewhere the attention does not read, and the
// model silently attends over stale pages. So the transcript records both
// carves for every shape, as offsets from the arena base, and the two are
// compared row by row.
//
// Offsets rather than addresses: the property is the layout, and a golden full
// of malloc's return values would be a golden about malloc.

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "attn/page_compact.hpp"
#include "model/attn_observation.hpp"
#include "model/attn_page_mask.hpp"
#include "model/hook_sideband_arena.hpp"
#include "model/stage_hooks.hpp"

namespace {

constexpr char kSep = '\x1f';

// ---------------------------------------------------------------------------
// The device side, replaced.
// ---------------------------------------------------------------------------

// One slab, handed out at a known base so every recorded pointer can be
// reported as an offset. 64 MiB is past anything the grid asks for.
alignas(256) unsigned char g_slab[64u << 20];
std::size_t g_slab_used = 0;
int g_malloc_calls = 0;

std::vector<std::string>* g_events = nullptr;

void record(const std::string& e) {
    if (g_events != nullptr) g_events->push_back(e);
}

// The mask slot's base, so pointers can be printed relative to it.
const unsigned char* g_base = nullptr;

std::string at(const void* p) {
    if (p == nullptr) return "null";
    if (g_base == nullptr) return "?";
    const auto* c = static_cast<const unsigned char*>(p);
    if (c < g_base) return "before_base";
    return "+" + std::to_string(static_cast<std::size_t>(c - g_base));
}

// The compaction kernel, replaced by a recorder. Recording its ARGUMENTS is
// the point: `compact` is the one method that hands the carve to a kernel, and
// a buffer carved correctly but passed in the wrong slot is invisible to a
// layout-only check.
struct CompactCall {
    bool seen = false;
    std::string keep;
    std::string counts;
    std::string out_indices;
    std::string out_indptr;
    std::string out_last_lens;
    std::uint32_t keep_stride = 0;
    int num_requests = 0;
    const std::uint32_t* in_indices = nullptr;
    const std::uint32_t* in_indptr = nullptr;
    const std::uint32_t* in_lens = nullptr;
};

CompactCall g_compact;

struct MemsetCall {
    bool seen = false;
    std::string dst;
    int value = 0;
    std::size_t bytes = 0;
};

MemsetCall g_memset;

}  // namespace

cudaError_t cudaMalloc(void** ptr, std::size_t bytes) {
    const std::size_t aligned = (g_slab_used + 255u) & ~std::size_t{255};
    if (aligned + bytes > sizeof(g_slab)) return cudaErrorMemoryAllocation;
    *ptr = g_slab + aligned;
    g_slab_used = aligned + bytes;
    ++g_malloc_calls;
    record("malloc:" + std::to_string(bytes));
    return cudaSuccess;
}

cudaError_t cudaFree(void* ptr) {
    if (ptr != nullptr) record("free");
    return cudaSuccess;
}

cudaError_t cudaStreamSynchronize(cudaStream_t) {
    record("sync");
    return cudaSuccess;
}

cudaError_t cudaMemsetAsync(
    void* ptr, int value, std::size_t bytes, cudaStream_t) {
    g_memset.seen = true;
    g_memset.dst = at(ptr);
    g_memset.value = value;
    g_memset.bytes = bytes;
    return cudaSuccess;
}

namespace pie_cuda_driver::kernels::attn {

void compact_page_csr(
    const std::uint32_t* page_indices_in,
    const std::uint32_t* page_indptr_in,
    const std::uint32_t* last_page_lens_in,
    const std::uint8_t* keep,
    std::uint32_t* scratch_counts,
    std::uint32_t keep_stride,
    int num_requests,
    std::uint32_t* page_indices_out,
    std::uint32_t* page_indptr_out,
    std::uint32_t* last_page_lens_out,
    cudaStream_t) {
    g_compact.seen = true;
    g_compact.in_indices = page_indices_in;
    g_compact.in_indptr = page_indptr_in;
    g_compact.in_lens = last_page_lens_in;
    g_compact.keep = at(keep);
    g_compact.counts = at(scratch_counts);
    g_compact.keep_stride = keep_stride;
    g_compact.num_requests = num_requests;
    g_compact.out_indices = at(page_indices_out);
    g_compact.out_indptr = at(page_indptr_out);
    g_compact.out_last_lens = at(last_page_lens_out);
}

}  // namespace pie_cuda_driver::kernels::attn

namespace {

using pie_cuda_driver::model::AttentionObservation;
using pie_cuda_driver::model::FirePageMask;
using pie_cuda_driver::model::HookSidebandArena;
using pie_cuda_driver::model::PageMaskCapturePlan;
using pie_cuda_driver::model::prepare_page_mask_capture;
using pie_cuda_driver::model::StageHooks;

// ---------------------------------------------------------------------------
// The grid.
// ---------------------------------------------------------------------------

struct Shape {
    const char* name;
    // Host page CSR, `num_requests + 1` entries.
    std::vector<std::uint32_t> indptr;
};

// Realistic decode/prefill geometries plus every way the CSR validation can
// reject one. The malformed cases are not padding: `mask_slot_layout` is the
// only thing standing between a bad CSR and a stride that under-covers some
// request's page list, which the compaction would then read past.
const std::vector<Shape>& shapes() {
    static const std::vector<Shape> s = {
        {"single_page", {0, 1}},
        {"single_request_deep", {0, 129}},
        {"uniform_decode_4", {0, 8, 16, 24, 32}},
        {"ragged_decode_5", {0, 3, 3, 40, 41, 97}},
        {"leading_empty", {0, 0, 7}},
        {"trailing_empty", {0, 7, 7}},
        {"wide_batch_16",
         {0, 2, 5, 5, 9, 14, 20, 27, 35, 44, 54, 65, 77, 90, 104, 119, 135}},
        {"all_empty", {0, 0, 0}},
        {"zero_total", {0}},
        {"non_monotonic", {0, 9, 4, 12}},
        {"end_past_total", {0, 5, 99, 12}},
    };
    return s;
}

// A usable observation over a host CSR. The device pointers are never
// dereferenced by anything under test — the header is emphatic that the host
// CSR sizes and the device CSR addresses — so they only have to be non-null
// for `usable()`.
struct Fire {
    AttentionObservation obs;
    std::vector<std::uint32_t> indptr;
    std::uint32_t scratch[8] = {};

    explicit Fire(const std::vector<std::uint32_t>& csr) : indptr(csr) {
        obs.kv = reinterpret_cast<pie_cuda_driver::model::KvCache*>(this);
        obs.kv_page_indices_d = scratch;
        obs.kv_page_indptr_d = scratch;
        obs.kv_last_page_lens_d = scratch;
        obs.qo_indptr_h = scratch;
        obs.kv_page_indptr_h = indptr.data();
        obs.kv_last_page_lens_h = scratch;
        obs.num_requests = static_cast<int>(indptr.size()) - 1;
        obs.total_tokens = obs.num_requests;
    }
};

std::string join(const std::vector<std::string>& v) {
    std::string out;
    for (const std::string& e : v) {
        if (!out.empty()) out += ",";
        out += e;
    }
    return out;
}

// Whether the mask slot is still held, observed the way a caller observes it:
// an overlapping acquire is refused. The C++ arena exposes no `is_held`, and
// probing is the more honest check anyway — it asks whether the NEXT fire can
// proceed, which is the thing a leaked hold would break.
bool slot_is_held(HookSidebandArena& arena) {
    void* p = arena.acquire(HookSidebandArena::Region::Mask, 1, nullptr);
    if (p == nullptr) return true;
    arena.release(HookSidebandArena::Region::Mask);
    return false;
}

std::string outcome_of(const std::exception* e) {
    return e == nullptr ? std::string("ok") : std::string("throw:") + e->what();
}

// ---------------------------------------------------------------------------
// Script 1 — the carve, from both call paths.
// ---------------------------------------------------------------------------

void script_carve() {
    for (const Shape& shape : shapes()) {
        Fire fire(shape.indptr);
        HookSidebandArena arena;

        // The prepare pass first, as it runs first in production: it is what
        // pre-grows the slot so the fire-time acquire cannot allocate inside a
        // captured region.
        PageMaskCapturePlan plan =
            prepare_page_mask_capture(&arena, fire.obs, nullptr);
        // `out_indices` sits at offset 0 of the slot, so it IS the base.
        g_base = reinterpret_cast<const unsigned char*>(plan.out_indices);

        std::printf(
            "carve%c%s%cplan%c%d%c%u%c%u%c%s%c%s%c%s%c%s\n", kSep, shape.name,
            kSep, kSep, plan.ok ? 1 : 0, kSep, plan.num_requests, kSep,
            plan.stride, kSep, at(plan.out_indices).c_str(), kSep,
            at(plan.out_indptr).c_str(), kSep, at(plan.out_last_lens).c_str(),
            kSep, at(plan.keep).c_str());

        StageHooks hooks;
        hooks.wants_page_mask = true;
        hooks.observation = &fire.obs;
        hooks.sideband_arena = &arena;

        std::string result;
        std::string sink_keep = "null";
        std::string idx = "null";
        std::string indptr = "null";
        std::string lens = "null";
        std::uint32_t requests = 0;
        std::uint32_t stride = 0;
        int active = 0;
        try {
            FirePageMask mask(&hooks, nullptr);
            result = "ok";
            // A carve the prepare pass refused still has to be reportable, so
            // the base can also come from the fire-time path — and when both
            // paths ran, they hand out the same one, which is the claim.
            if (g_base == nullptr) {
                g_base = reinterpret_cast<const unsigned char*>(
                    mask.page_indices());
            }
            active = mask.active() ? 1 : 0;
            idx = at(mask.page_indices());
            indptr = at(mask.page_indptr());
            lens = at(mask.last_page_lens());
            if (mask.sink() != nullptr) {
                sink_keep = at(mask.sink()->keep);
                requests = mask.sink()->num_requests;
                stride = mask.sink()->stride;
            }
        } catch (const std::exception& e) {
            result = outcome_of(&e);
        }

        std::printf(
            "carve%c%s%cfire%c%d%c%u%c%u%c%s%c%s%c%s%c%s%c%s\n", kSep,
            shape.name, kSep, kSep, active, kSep, requests, kSep, stride, kSep,
            idx.c_str(), kSep, indptr.c_str(), kSep, lens.c_str(), kSep,
            sink_keep.c_str(), kSep, result.c_str());

        // The invariant the whole file exists to protect, asserted in the
        // transcript rather than left for a reader to diff by eye.
        const bool agree = (plan.ok ? 1 : 0) == active &&
                           plan.num_requests == requests &&
                           plan.stride == stride &&
                           at(plan.out_indices) == idx &&
                           at(plan.out_indptr) == indptr &&
                           at(plan.out_last_lens) == lens &&
                           at(plan.keep) == sink_keep;
        std::printf(
            "carve%c%s%cagree%c%d\n", kSep, shape.name, kSep, kSep,
            agree ? 1 : 0);

        g_base = nullptr;
    }
}

// ---------------------------------------------------------------------------
// Script 2 — the layer loop.
// ---------------------------------------------------------------------------

void script_layer_loop() {
    Fire fire({0, 3, 3, 40, 41, 97});
    HookSidebandArena arena;
    StageHooks hooks;
    hooks.wants_page_mask = true;
    hooks.observation = &fire.obs;
    hooks.sideband_arena = &arena;

    FirePageMask mask(&hooks, nullptr);
    g_base = reinterpret_cast<const unsigned char*>(mask.page_indices());

    std::uint32_t in_idx[128] = {};
    std::uint32_t in_indptr[8] = {};
    std::uint32_t in_lens[8] = {};

    for (std::uint32_t layer = 0; layer < 3; ++layer) {
        g_memset = MemsetCall{};
        mask.begin_layer(nullptr);
        std::printf(
            "loop%clayer%u%cbegin%c%s%c%d%c%zu%c%d%c%d\n", kSep, layer, kSep,
            kSep, g_memset.dst.c_str(), kSep, g_memset.value, kSep,
            g_memset.bytes, kSep, mask.written_for(layer) ? 1 : 0, kSep,
            mask.written_for(layer + 1) ? 1 : 0);

        // Layer 1 is the one whose program writes the sink.
        if (layer == 1) mask.sink()->written_layer = static_cast<int>(layer);

        // `written_for` is the stale-view guard: after the sink writes for
        // layer 1, only layer 1 may compact.
        std::printf(
            "loop%clayer%u%cwritten%c%d%c%d%c%d\n", kSep, layer, kSep, kSep,
            mask.written_for(0) ? 1 : 0, kSep, mask.written_for(1) ? 1 : 0,
            kSep, mask.written_for(2) ? 1 : 0);

        if (mask.written_for(layer)) {
            g_compact = CompactCall{};
            mask.compact(in_idx, in_indptr, in_lens, 5, nullptr);
            std::printf(
                "loop%clayer%u%ccompact%c%d%c%s%c%s%c%s%c%s%c%s%c%u%c%d%c%d\n",
                kSep, layer, kSep, kSep, g_compact.seen ? 1 : 0, kSep,
                g_compact.keep.c_str(), kSep, g_compact.counts.c_str(), kSep,
                g_compact.out_indices.c_str(), kSep,
                g_compact.out_indptr.c_str(), kSep,
                g_compact.out_last_lens.c_str(), kSep, g_compact.keep_stride,
                kSep, g_compact.num_requests, kSep,
                (g_compact.in_indices == in_idx &&
                 g_compact.in_indptr == in_indptr &&
                 g_compact.in_lens == in_lens)
                    ? 1
                    : 0);
        }
    }

    // A compaction whose request count disagrees with the fire's must be
    // refused: the kernel would otherwise walk `keep` with the wrong row
    // count.
    g_compact = CompactCall{};
    std::string mismatch;
    try {
        mask.compact(in_idx, in_indptr, in_lens, 4, nullptr);
        mismatch = "ok";
    } catch (const std::exception& e) {
        mismatch = outcome_of(&e);
    }
    std::printf(
        "loop%cmismatch%c-%c%d%c%s\n", kSep, kSep, kSep,
        g_compact.seen ? 1 : 0, kSep, mismatch.c_str());

    g_base = nullptr;
}

// ---------------------------------------------------------------------------
// Script 3 — the inactive fire.
// ---------------------------------------------------------------------------

void script_inactive() {
    Fire fire({0, 8, 16});
    HookSidebandArena arena;

    struct Case {
        const char* name;
        bool null_hooks;
        bool wants;
        bool with_obs;
        bool with_arena;
    };
    const Case cases[] = {
        {"null_hooks", true, false, false, false},
        {"wants_false", false, false, true, true},
        {"wants_false_no_obs", false, false, false, false},
        {"no_observation", false, true, false, true},
        {"no_arena", false, true, true, false},
    };

    for (const Case& c : cases) {
        StageHooks hooks;
        hooks.wants_page_mask = c.wants;
        hooks.observation = c.with_obs ? &fire.obs : nullptr;
        hooks.sideband_arena = c.with_arena ? &arena : nullptr;

        std::string result;
        int active = -1;
        int sink_null = -1;
        int wrote = -1;
        try {
            FirePageMask mask(c.null_hooks ? nullptr : &hooks, nullptr);
            result = "ok";
            active = mask.active() ? 1 : 0;
            sink_null = mask.sink() == nullptr ? 1 : 0;
            // Every accessor must stay safe on an inactive mask: the layer
            // loop calls `begin_layer` unconditionally.
            g_memset = MemsetCall{};
            mask.begin_layer(nullptr);
            wrote = g_memset.seen ? 1 : 0;
            g_compact = CompactCall{};
            mask.compact(nullptr, nullptr, nullptr, 99, nullptr);
        } catch (const std::exception& e) {
            result = outcome_of(&e);
        }
        std::printf(
            "inactive%c%s%c%d%c%d%c%d%c%d%c%s\n", kSep, c.name, kSep, active,
            kSep, sink_null, kSep, wrote, kSep, g_compact.seen ? 1 : 0, kSep,
            result.c_str());
    }
}

// ---------------------------------------------------------------------------
// Script 4 — across fires.
// ---------------------------------------------------------------------------

// The graph-capture precondition, end to end: once the arena has grown to the
// workload's widest fire, every narrower fire reuses the same addresses and
// allocates nothing. This is what makes a captured hook body replayable, and
// it is a claim about `FirePageMask` and the arena TOGETHER, so neither
// component's own tests can make it.
void script_across_fires() {
    HookSidebandArena arena;
    const std::vector<std::vector<std::uint32_t>> fires = {
        {0, 8, 16, 24, 32},
        {0, 4, 8},
        {0, 1},
        {0, 8, 16, 24, 32},
        // Past the 64 KiB base capacity, so this one must grow. Its whole
        // point is the fire AFTER it.
        {0, 8000, 16000, 24000},
        {0, 8, 16, 24, 32},
    };

    const unsigned char* prev_base = nullptr;
    for (std::size_t i = 0; i < fires.size(); ++i) {
        Fire fire(fires[i]);
        StageHooks hooks;
        hooks.wants_page_mask = true;
        hooks.observation = &fire.obs;
        hooks.sideband_arena = &arena;

        std::vector<std::string> events;
        g_events = &events;
        {
            FirePageMask mask(&hooks, nullptr);
            const auto* base =
                reinterpret_cast<const unsigned char*>(mask.page_indices());
            const bool moved = prev_base != nullptr && base != prev_base;
            prev_base = base;
            g_base = base;
            // The base is reported as same/moved against the PREVIOUS fire
            // rather than as an address. That localises the move: it must
            // happen on exactly the fires whose event column shows a growth,
            // and on no others. An address column could not say that, and a
            // hash of addresses would be a hash of malloc.
            std::printf(
                "fires%c%zu%c%u%c%u%c%s%c%s%c%s\n", kSep, i, kSep,
                mask.sink()->num_requests, kSep, mask.sink()->stride, kSep,
                at(mask.sink()->keep).c_str(), kSep,
                moved ? "base_moved" : "base_same", kSep,
                events.empty() ? "-" : join(events).c_str());
            g_base = nullptr;
        }
        g_events = nullptr;

        // The slot must be back for the next fire; a leaked hold turns every
        // subsequent fire's acquire into a busy refusal.
        std::printf(
            "fires%c%zu%cheld_after%c%d\n", kSep, i, kSep, kSep,
            slot_is_held(arena) ? 1 : 0);
    }
}

}  // namespace

int main() {
    script_carve();
    script_layer_loop();
    script_inactive();
    script_across_fires();
    return 0;
}
