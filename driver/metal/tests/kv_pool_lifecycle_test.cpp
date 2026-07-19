// Phase 3 (metal_ptir_plan.md §7, review item 4) standalone-buffer allocation
// lifecycle gate — REAL Metal allocation, but NO checkpoint (RawMetalContext
// only needs a Metal device, not model weights). Apple-only (RawMetalContext
// is Metal). Proves that create_standalone_buffer / release_standalone_buffer
// keep an exact allocation count + byte total, and — critically — that a
// repeated grow/shrink-style cycle (allocate new, release old) returns to a
// bounded baseline instead of leaking the old paged-KV buffers unbounded
// (which is exactly the resize_pool leak the review flagged).

#include <cstdio>
#include <string>
#include <vector>

#include "mtl4_context.hpp"

using pie::metal::RawMetalContext;
using pie::metal::SlotHandle;

namespace {
int g_pass = 0, g_fail = 0;
bool expect(bool ok, const std::string& what) {
    if (ok) { ++g_pass; std::printf("  PASS  %s\n", what.c_str()); }
    else    { ++g_fail; std::printf("  FAIL  %s\n", what.c_str()); }
    return ok;
}
}  // namespace

int main() {
    std::printf("[standalone-buffer allocation lifecycle]\n");

    auto ctx = RawMetalContext::create(16u << 20, 512u << 20);
    if (!expect(ctx != nullptr, "RawMetalContext::create succeeds")) {
        std::printf("\n==== kv_pool_lifecycle_test: %d passed, %d failed ====\n", g_pass, g_fail);
        return g_fail == 0 ? 0 : 1;
    }

    expect(ctx->standalone_buffer_count() == 0 && ctx->standalone_bytes() == 0,
           "baseline: zero standalone buffers");

    // Allocate a batch of buffers; count + bytes track exactly.
    constexpr size_t kBufBytes = 1u << 20;  // 1 MiB
    constexpr size_t kN = 6;
    std::vector<SlotHandle> handles;
    for (size_t i = 0; i < kN; ++i) handles.push_back(ctx->create_standalone_buffer(kBufBytes));
    bool all_valid = true;
    for (const auto& h : handles) all_valid = all_valid && h.valid();
    expect(all_valid, "all standalone buffers allocated");
    expect(ctx->standalone_buffer_count() == kN && ctx->standalone_bytes() == kN * kBufBytes,
           "count + bytes track allocations exactly");

    // Release them all; count + bytes return to zero (ARC frees them).
    for (const auto& h : handles) ctx->release_standalone_buffer(h);
    expect(ctx->standalone_buffer_count() == 0 && ctx->standalone_bytes() == 0,
           "release returns count + bytes to zero (no leak)");

    // A double-release / invalid-handle release is a harmless no-op.
    ctx->release_standalone_buffer(handles[0]);   // already released
    ctx->release_standalone_buffer(SlotHandle{});  // invalid
    expect(ctx->standalone_buffer_count() == 0 && ctx->standalone_bytes() == 0,
           "double/invalid release is a no-op");

    // The decisive check: simulate the resize_pool grow/shrink pattern — each
    // cycle allocates a NEW (bigger) buffer and releases the OLD one. Over many
    // cycles the live allocation must stay bounded at exactly ONE buffer, not
    // grow with the cycle count (the leak the review flagged).
    SlotHandle live = ctx->create_standalone_buffer(kBufBytes);
    size_t peak_count = ctx->standalone_buffer_count();
    for (int cycle = 0; cycle < 64; ++cycle) {
        const size_t sz = (cycle % 2 == 0) ? (kBufBytes * 2) : kBufBytes;  // grow/shrink
        SlotHandle next = ctx->create_standalone_buffer(sz);
        // Both briefly live (mirrors resize: new allocated + copy before old freed).
        peak_count = std::max(peak_count, ctx->standalone_buffer_count());
        ctx->release_standalone_buffer(live);
        live = next;
    }
    expect(peak_count <= 2,
           "grow/shrink keeps at most 2 buffers live at once (peak=" +
               std::to_string(peak_count) + ")");
    expect(ctx->standalone_buffer_count() == 1,
           "after 64 grow/shrink cycles exactly ONE buffer is live (bounded, no leak; count=" +
               std::to_string(ctx->standalone_buffer_count()) + ")");
    ctx->release_standalone_buffer(live);
    expect(ctx->standalone_buffer_count() == 0, "final release returns to zero");

    std::printf("\n[placement-sparse elastic lifecycle]\n");
    SlotHandle elastic = ctx->create_elastic_buffer(64u << 20, 2u << 20);
    const std::uint64_t elastic_address = elastic.gpu_address;
    expect(elastic.valid() && elastic.elastic,
           "elastic buffer reserves stable sparse VA");
    expect(ctx->elastic_committed_pages() > 0,
           "initial sparse prefix is physically committed");
    expect(ctx->ensure_elastic_buffer(elastic, 32u << 20) &&
               elastic.gpu_address == elastic_address,
           "elastic grow preserves GPU address");
    expect(ctx->trim_elastic_buffer(elastic, 0),
           "elastic trim completes");
    ctx->drain_elastic_mappings();
    expect(ctx->elastic_committed_pages() == 0,
           "elastic trim releases physical commitment");
    expect(ctx->pending_elastic_release_count() == 0,
           "elastic trim retires final heap without another mapping");
    expect(ctx->ensure_elastic_buffer(elastic, 4u << 20) &&
               elastic.gpu_address == elastic_address,
           "elastic regrow preserves GPU address");
    ctx->release_elastic_buffer(elastic);
    expect(ctx->pending_elastic_release_count() == 0,
           "elastic final release leaves no pending heap");

    auto tight = RawMetalContext::create(4u << 20, 4u << 20);
    SlotHandle tight_a = tight->create_elastic_buffer(4u << 20);
    SlotHandle tight_b = tight->create_elastic_buffer(4u << 20);
    expect(
        !tight->ensure_elastic_buffers_atomically({
            {tight_a, 4u << 20},
            {tight_b, 4u << 20},
        }),
        "atomic elastic grow rejects an over-budget batch");
    expect(
        tight->elastic_committed_pages() == 0,
        "failed atomic grow leaves every buffer uncommitted");
    tight->release_elastic_buffer(tight_a);
    tight->release_elastic_buffer(tight_b);

    auto pressure = RawMetalContext::create(4u << 20, 8u << 20);
    SlotHandle pressure_a = pressure->create_elastic_buffer(4u << 20);
    SlotHandle pressure_b = pressure->create_elastic_buffer(4u << 20);
    pressure->set_memory_pressure_level_for_test(1);
    expect(
        pressure->elastic_budget_pages() == 2 &&
            !pressure->ensure_elastic_buffers_atomically({
                {pressure_a, 4u << 20},
                {pressure_b, 4u << 20},
            }),
        "warning pressure halves admission budget");
    expect(
        pressure->elastic_committed_pages() == 0,
        "pressure rejection leaves every buffer uncommitted");
    pressure->set_memory_pressure_level_for_test(0);
    expect(
        pressure->ensure_elastic_buffer(pressure_a, 4u << 20),
        "normal pressure restores the configured budget");
    const auto committed_before_critical =
        pressure->elastic_committed_pages();
    pressure->set_memory_pressure_level_for_test(2);
    expect(
        !pressure->ensure_elastic_buffer(pressure_b, 4u << 20) &&
            pressure->elastic_committed_pages() ==
                committed_before_critical,
        "critical pressure blocks growth without disturbing committed storage");
    pressure->release_elastic_buffer(pressure_a);
    pressure->release_elastic_buffer(pressure_b);

    std::printf("\n==== kv_pool_lifecycle_test: %d passed, %d failed ====\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
