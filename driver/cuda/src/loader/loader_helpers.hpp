#pragma once

// Stateless arithmetic / checked-narrowing helpers factored out of
// load_plan_executor.hpp so the executor body stays materialize logic, not
// utility math. Free functions in the driver namespace, so the executor's
// existing unqualified call sites resolve here unchanged.

#include <climits>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "loader_config.hpp"

namespace pie_cuda_driver {

inline std::uint64_t next_power_of_two(std::uint64_t value)
{
    if (value <= 1) {
        return 1;
    }
    --value;
    for (std::size_t shift = 1; shift < 64; shift <<= 1) {
        value |= value >> shift;
    }
    return value + 1;
}

inline std::uint64_t checked_mul_u64(std::uint64_t lhs, std::uint64_t rhs, const char* context)
{
    if (rhs != 0 && lhs > UINT64_MAX / rhs) {
        throw std::runtime_error(
            std::string("rust storage executor: ") + context + " byte size overflow");
    }
    return lhs * rhs;
}

// Narrow an int64 dimension to int for the tile/scale kernels (which take int),
// throwing loudly rather than silently truncating a >INT_MAX value.
inline int checked_int(std::int64_t v, const char* what)
{
    if (v < 0 || v > INT_MAX) {
        throw std::runtime_error(
            std::string("rust storage executor: ") + what + " exceeds int range");
    }
    return static_cast<int>(v);
}

inline std::uint64_t checked_nibble_bytes(std::uint64_t rows, std::uint64_t cols,
                                          const char* context)
{
    const std::uint64_t elements = checked_mul_u64(rows, cols, context);
    if (elements % 2 != 0) {
        throw std::runtime_error(
            std::string("rust storage executor: ") + context + " has odd nibble element count");
    }
    return elements / 2;
}

inline std::vector<float> expand_e8m0_to_f32(const std::uint8_t* e8m0, std::size_t n)
{
    // E8M0 stores a biased power-of-two exponent: value = 2^(byte - bias).
    std::vector<float> out(n);
    for (std::size_t i = 0; i < n; ++i) {
        out[i] = std::ldexp(1.0f, static_cast<int>(e8m0[i]) - loader_config::kE8M0Bias);
    }
    return out;
}

}  // namespace pie_cuda_driver
