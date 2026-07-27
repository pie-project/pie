#pragma once

#include <cstddef>
#include <cstdint>

namespace pie::elastic {

inline constexpr std::size_t kLogicalPageBytes = 2ull * 1024 * 1024;

inline constexpr std::size_t pages_for_bytes(
    std::size_t bytes,
    std::size_t page_bytes = kLogicalPageBytes) noexcept {
    return bytes == 0 ? 0 : (bytes + page_bytes - 1) / page_bytes;
}

class PhysicalPool {
  public:
    virtual ~PhysicalPool() = default;

    virtual bool try_reserve(std::size_t pages) = 0;
    virtual void unreserve(std::size_t pages) noexcept = 0;
    virtual std::size_t page_bytes() const noexcept = 0;
    virtual std::size_t budget_pages() const noexcept = 0;
    virtual std::size_t committed_pages() const noexcept = 0;
};

class Arena {
  public:
    virtual ~Arena() = default;

    virtual std::uint64_t base() const noexcept = 0;
    virtual std::size_t committed_bytes() const noexcept = 0;
    virtual void ensure_committed(std::size_t bytes) = 0;
    virtual void trim_committed(std::size_t bytes) = 0;
};

}  // namespace pie::elastic
