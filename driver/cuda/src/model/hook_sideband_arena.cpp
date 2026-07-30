#include "model/hook_sideband_arena.hpp"

#include <cstdio>
#include <cstdlib>

namespace pie_cuda_driver::model {

namespace {

bool sideband_trace_enabled() noexcept {
    static const bool enabled = [] {
        const char* v = std::getenv("PIE_SIDEBAND_TRACE");
        return v != nullptr && v[0] != '\0' && v[0] != '0';
    }();
    return enabled;
}

// Growth ladder: powers of two from 64 KiB. Keeps growths logarithmic in the
// largest fire ever seen, so the "rare after warmup" claim in the header is
// structural rather than hopeful.
std::size_t round_capacity(std::size_t bytes) noexcept {
    std::size_t cap = 64u * 1024u;
    while (cap < bytes) cap *= 2;
    return cap;
}

}  // namespace

const char* HookSidebandArena::region_name(Region region) noexcept {
    return region == Region::Score ? "score" : "mask";
}

HookSidebandArena::~HookSidebandArena() {
    for (Slot& slot : slots_) {
        if (slot.base != nullptr) cudaFree(slot.base);
        slot = Slot{};
    }
}

void* HookSidebandArena::acquire(
    Region region, std::size_t bytes, cudaStream_t stream) noexcept {
    Slot& slot = slots_[static_cast<int>(region)];
    if (slot.busy) {
        std::fprintf(
            stderr,
            "[pie-driver-cuda] hook sideband arena: overlapping %s "
            "acquisitions; the second is refused\n",
            region_name(region));
        return nullptr;
    }
    if (bytes == 0) return nullptr;
    if (bytes > slot.capacity) {
        // Growth path: retire everything in flight that may still read the
        // old block, then free+realloc. Synchronous and logged because it is
        // meant to be rare — steady state is the capacity check above.
        const std::size_t new_capacity = round_capacity(bytes);
        if (cudaStreamSynchronize(stream) != cudaSuccess) {
            return nullptr;
        }
        if (slot.base != nullptr) {
            cudaFree(slot.base);
            slot.base = nullptr;
            slot.capacity = 0;
        }
        void* fresh = nullptr;
        if (cudaMalloc(&fresh, new_capacity) != cudaSuccess) {
            return nullptr;
        }
        slot.base = static_cast<std::uint8_t*>(fresh);
        slot.capacity = new_capacity;
        ++generation_;
        ++fire_grows_;
        ++total_grows_;
        std::fprintf(
            stderr,
            "[pie-driver-cuda] hook sideband arena grew: region=%s to %zu KiB "
            "(need %zu B, generation %llu)\n",
            region_name(region), new_capacity >> 10, bytes,
            static_cast<unsigned long long>(generation_));
    }
    slot.busy = true;
    ++fire_acquires_;
    ++total_acquires_;
    return slot.base;
}

void HookSidebandArena::release(Region region) noexcept {
    slots_[static_cast<int>(region)].busy = false;
}

void HookSidebandArena::begin_fire() noexcept {
    if (sideband_trace_enabled() && fire_index_ > 0) {
        std::fprintf(
            stderr,
            "[pie-sideband] fire=%llu acquires=%u device_allocs=%u "
            "(totals: acquires=%llu grows=%llu, generation=%llu)\n",
            static_cast<unsigned long long>(fire_index_),
            fire_acquires_, fire_grows_,
            static_cast<unsigned long long>(total_acquires_),
            static_cast<unsigned long long>(total_grows_),
            static_cast<unsigned long long>(generation_));
    }
    ++fire_index_;
    fire_acquires_ = 0;
    fire_grows_ = 0;
}

}  // namespace pie_cuda_driver::model
