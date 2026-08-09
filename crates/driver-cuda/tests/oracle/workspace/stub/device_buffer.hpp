#pragma once
// Stub for csrc/src/device_buffer.hpp.
//
// `Workspace` holds a `LoraStageArena`, which holds `DeviceBuffer`s. The arena
// allocates NOTHING at construction — it bump-allocates on first use, which
// the forward pass does and `allocate_full` never does — so the oracle needs
// the type to exist and nothing more. `alloc` is left undefined on purpose: if
// a future `allocate_full` starts touching the arena, this fails to LINK
// rather than silently recording an allocation the transcript cannot see.
#include <cstddef>
#include <cstdint>
#include <utility>

namespace pie_cuda_driver {

template <typename T>
class DeviceBuffer {
public:
    DeviceBuffer() = default;
    static DeviceBuffer alloc(std::size_t n);
    std::size_t size() const { return n_; }
    T* data() const { return ptr_; }

private:
    T* ptr_ = nullptr;
    std::size_t n_ = 0;
};

}  // namespace pie_cuda_driver
