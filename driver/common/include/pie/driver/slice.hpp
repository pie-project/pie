#pragma once

#include <cstddef>
#include <cstdint>
#include <span>

#include <pie_driver_abi.h>

namespace pie::driver {

template <typename T>
struct Slice {
    const T* ptr = nullptr;
    std::size_t len = 0;

    constexpr std::size_t size() const noexcept { return len; }
    constexpr bool empty() const noexcept { return len == 0; }
    constexpr const T* data() const noexcept { return ptr; }
    template <typename U>
    std::span<const U> as() const noexcept {
        return std::span<const U>(reinterpret_cast<const U*>(ptr), len);
    }
};

struct ByteSlice {
    const void* ptr = nullptr;
    std::size_t len = 0;

    constexpr std::size_t size() const noexcept { return len; }
    constexpr bool empty() const noexcept { return len == 0; }
    template <typename U>
    std::span<const U> as() const noexcept {
        return std::span<const U>(reinterpret_cast<const U*>(ptr), len);
    }
};

template <typename T>
constexpr Slice<T> slice_from(const T* ptr, std::size_t len) noexcept {
    return Slice<T>{ptr, len};
}
inline Slice<std::uint32_t> slice_from_u32(const std::uint32_t* p, std::size_t n) noexcept {
    return Slice<std::uint32_t>{p, n};
}
inline Slice<std::uint64_t> slice_from_u64(const std::uint64_t* p, std::size_t n) noexcept {
    return Slice<std::uint64_t>{p, n};
}
inline Slice<std::uint8_t> slice_from_u8(const std::uint8_t* p, std::size_t n) noexcept {
    return Slice<std::uint8_t>{p, n};
}

}  // namespace pie::driver
