#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <new>

namespace pie::metal::pipeline {

// Stable CPU-visible storage with an optional Metal Shared-storage backing.
// The host fallback is used by non-Apple builds and pure interpreter tests.
struct SharedStorage {
    std::shared_ptr<void> owner;
    std::uint8_t* contents = nullptr;
    void* native_buffer = nullptr;
    std::uint64_t gpu_address = 0;
    std::size_t size = 0;

    bool valid() const { return contents != nullptr && size != 0; }
    bool device_visible() const {
        return native_buffer != nullptr && gpu_address != 0;
    }
};

inline SharedStorage make_host_shared_storage(std::size_t size) {
    SharedStorage storage;
    if (size == 0) return storage;
    void* allocation = ::operator new(size, std::align_val_t{alignof(std::max_align_t)});
    std::memset(allocation, 0, size);
    storage.owner = std::shared_ptr<void>(
        allocation,
        [](void* ptr) {
            ::operator delete(
                ptr, std::align_val_t{alignof(std::max_align_t)});
        });
    storage.contents = static_cast<std::uint8_t*>(allocation);
    storage.size = size;
    return storage;
}

// Apple builds return an MTLStorageModeShared buffer. Other builds return the
// stable aligned host fallback above.
SharedStorage make_platform_shared_storage(std::size_t size);

}  // namespace pie::metal::pipeline
