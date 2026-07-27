#pragma once

#include <cstddef>

/// Page accounting for the Metal elastic allocator.
///
/// Only the page size and the rounding rule. The CUDA driver states the same
/// two and adds `PhysicalPool`/`Arena` on top, because it reserves through a
/// virtual address range it can commit into; Metal reserves through a
/// `MTLHeap`, which is a different mechanism with a different lifetime, and
/// the abstract classes never applied to it. Sharing this file with CUDA meant
/// carrying an interface Metal does not implement.
namespace pie::elastic {

inline constexpr std::size_t kLogicalPageBytes = 2ull * 1024 * 1024;

inline constexpr std::size_t pages_for_bytes(
    std::size_t bytes,
    std::size_t page_bytes = kLogicalPageBytes) noexcept {
    return bytes == 0 ? 0 : (bytes + page_bytes - 1) / page_bytes;
}

}  // namespace pie::elastic
