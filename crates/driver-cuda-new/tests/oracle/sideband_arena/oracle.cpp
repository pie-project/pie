// Drives the real `HookSidebandArena` through a scripted sequence of
// acquire/release/begin_fire calls and reports what each one returned.
//
// The transcript records the returned pointer SYMBOLICALLY — `block#K`, the
// K-th distinct device allocation the arena has ever made — rather than as an
// address. That is deliberate. The property this class exists to provide is
// stated in its header as a graph-capture precondition:
//
//   > while a region's capacity suffices, the addresses it hands out are
//   > STABLE across fires of the same geometry
//
// and stability is a statement about pointer IDENTITY, not about any
// particular value. A transcript of raw addresses would be reproducible only
// by an allocator that happened to hand out the same numbers, which proves
// nothing about the arena and would make the golden a property of malloc.
//
// The three CUDA entry points are defined here so the allocator can be told to
// fail: the growth path frees the old block before it learns whether a
// replacement exists, and that ordering is only observable if the failure can
// be scheduled.

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "model/hook_sideband_arena.hpp"

using pie_cuda_driver::model::HookSidebandArena;
using Region = HookSidebandArena::Region;

namespace {

// --- the failable device allocator -----------------------------------------

std::uintptr_t g_next_block = 0x100000;
int g_blocks_handed_out = 0;
// When positive, the next N cudaMalloc calls fail.
int g_fail_next_mallocs = 0;
// When positive, the next N cudaStreamSynchronize calls fail.
int g_fail_next_syncs = 0;
std::map<std::uintptr_t, int> g_block_id;   // address -> block number
std::vector<std::string> g_events;          // allocator-level side effects

}  // namespace

cudaError_t cudaMalloc(void** ptr, std::size_t bytes) {
    if (g_fail_next_mallocs > 0) {
        --g_fail_next_mallocs;
        g_events.push_back("malloc_failed:" + std::to_string(bytes));
        *ptr = nullptr;
        return cudaErrorMemoryAllocation;
    }
    const std::uintptr_t addr = g_next_block;
    // Spaced far enough apart that a stale pointer cannot land inside a later
    // block and be mistaken for a live one.
    g_next_block += 1u << 24;
    g_block_id[addr] = ++g_blocks_handed_out;
    g_events.push_back("malloc:" + std::to_string(bytes) + ":block#" +
                       std::to_string(g_blocks_handed_out));
    *ptr = reinterpret_cast<void*>(addr);
    return cudaSuccess;
}

cudaError_t cudaFree(void* ptr) {
    const auto it = g_block_id.find(reinterpret_cast<std::uintptr_t>(ptr));
    g_events.push_back(
        "free:block#" +
        (it == g_block_id.end() ? std::string("?") : std::to_string(it->second)));
    return cudaSuccess;
}

cudaError_t cudaStreamSynchronize(cudaStream_t) {
    if (g_fail_next_syncs > 0) {
        --g_fail_next_syncs;
        g_events.push_back("sync_failed");
        return cudaErrorMemoryAllocation;
    }
    g_events.push_back("sync");
    return cudaSuccess;
}

namespace {

constexpr char kSep = '\x1f';

const char* region_name(Region r) {
    switch (r) {
        case Region::Score: return "score";
        case Region::Mask: return "mask";
        case Region::ScoreRows: return "score_rows";
    }
    return "?";
}

std::string block_of(void* p) {
    if (p == nullptr) return "null";
    const auto it = g_block_id.find(reinterpret_cast<std::uintptr_t>(p));
    return it == g_block_id.end() ? "unknown"
                                  : "block#" + std::to_string(it->second);
}

std::string drain_events() {
    std::string out;
    for (std::size_t i = 0; i < g_events.size(); ++i) {
        if (i != 0) out += ',';
        out += g_events[i];
    }
    g_events.clear();
    return out.empty() ? "-" : out;
}

struct Harness {
    HookSidebandArena arena;
    const char* label;

    void acquire(Region r, std::size_t bytes) {
        void* p = arena.acquire(r, bytes, nullptr);
        std::cout << label << kSep << "acquire" << kSep << region_name(r)
                  << kSep << bytes << kSep << block_of(p) << kSep << "gen="
                  << arena.generation() << kSep << drain_events() << "\n";
    }
    void release(Region r) {
        arena.release(r);
        std::cout << label << kSep << "release" << kSep << region_name(r)
                  << kSep << 0 << kSep << "-" << kSep << "gen="
                  << arena.generation() << kSep << drain_events() << "\n";
    }
    void begin_fire() {
        arena.begin_fire();
        std::cout << label << kSep << "begin_fire" << kSep << "-" << kSep << 0
                  << kSep << "-" << kSep << "gen=" << arena.generation() << kSep
                  << drain_events() << "\n";
    }
};

void reset_allocator() {
    g_next_block = 0x100000;
    g_blocks_handed_out = 0;
    g_fail_next_mallocs = 0;
    g_fail_next_syncs = 0;
    g_block_id.clear();
    g_events.clear();
}

// --- the scripts ------------------------------------------------------------

// The steady state the header describes: the region reaches the fire's max on
// the first layer and every later acquire is a pointer return.
void script_steady_state() {
    reset_allocator();
    Harness h{{}, "steady"};
    for (int fire = 0; fire < 3; ++fire) {
        h.begin_fire();
        for (int layer = 0; layer < 4; ++layer) {
            h.acquire(Region::Score, 4096);
            h.release(Region::Score);
        }
    }
}

// The growth ladder: 64 KiB base, doubling, and `bytes > capacity` rather than
// `>=` — so a request of exactly the capacity must NOT grow.
void script_growth_ladder() {
    reset_allocator();
    Harness h{{}, "ladder"};
    const std::size_t asks[] = {
        1,               // -> 64 KiB
        64u * 1024u,     // exactly capacity: no growth
        64u * 1024u + 1, // -> 128 KiB
        100u * 1024u,    // fits
        1u << 20,        // -> 1 MiB
        1,               // fits; must return the SAME block
    };
    for (std::size_t bytes : asks) {
        h.acquire(Region::Score, bytes);
        h.release(Region::Score);
    }
}

// The three regions are independent slots but share one generation counter.
void script_regions_are_independent() {
    reset_allocator();
    Harness h{{}, "regions"};
    h.acquire(Region::Score, 1024);
    h.acquire(Region::Mask, 1024);
    h.acquire(Region::ScoreRows, 1024);
    // All three held at once: the mask is held for the whole layer loop while
    // score is re-acquired per layer, so this overlap is the normal case.
    h.release(Region::Score);
    h.acquire(Region::Score, 1024);
    h.release(Region::Score);
    h.release(Region::Mask);
    h.release(Region::ScoreRows);
    // A growth in one region bumps the shared counter.
    h.acquire(Region::Mask, 1u << 20);
    h.release(Region::Mask);
}

// Overlapping acquisition is refused rather than shared, and the refusal does
// not disturb the slot.
void script_busy_refusal() {
    reset_allocator();
    Harness h{{}, "busy"};
    h.acquire(Region::Score, 2048);
    h.acquire(Region::Score, 2048);   // refused
    h.acquire(Region::Score, 1u << 20);  // refused BEFORE the size check
    h.release(Region::Score);
    h.acquire(Region::Score, 2048);   // same block again
    h.release(Region::Score);
}

// A zero-byte request is refused, and — the part worth pinning — it does NOT
// mark the slot busy, so the next real acquire still succeeds.
void script_zero_bytes() {
    reset_allocator();
    Harness h{{}, "zero"};
    h.acquire(Region::Score, 0);
    h.acquire(Region::Score, 2048);
    h.release(Region::Score);
    // Zero on a slot that already has capacity is still refused.
    h.acquire(Region::Score, 0);
    h.acquire(Region::Score, 2048);
    h.release(Region::Score);
}

// The failure paths, in the order they can happen.
void script_failures() {
    reset_allocator();
    Harness h{{}, "fail"};
    // A sync failure aborts before anything is freed: the old block survives.
    h.acquire(Region::Score, 1024);
    h.release(Region::Score);
    g_fail_next_syncs = 1;
    h.acquire(Region::Score, 1u << 20);   // refused, block#1 intact
    h.acquire(Region::Score, 1024);       // still block#1
    h.release(Region::Score);
    // A malloc failure happens AFTER the old block is freed, so the slot is
    // left empty and the next acquire must allocate afresh. Note the
    // generation does NOT move across this, even though the address did --
    // the region's addresses are re-checked by the caller's fingerprint, not
    // by the counter, which is why nothing reads `generation()`.
    g_fail_next_mallocs = 1;
    h.acquire(Region::Score, 1u << 20);   // refused, block#1 already freed
    h.acquire(Region::Score, 1024);       // fresh block
    h.release(Region::Score);
}

}  // namespace

int main() {
    script_steady_state();
    script_growth_ladder();
    script_regions_are_independent();
    script_busy_refusal();
    script_zero_bytes();
    script_failures();
    return 0;
}
