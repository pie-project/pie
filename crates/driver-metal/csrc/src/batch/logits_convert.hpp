#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>

#include "observability.hpp"

namespace pie::metal::batch {

inline float bf16_to_f32(std::uint16_t value) {
    const std::uint32_t bits = std::uint32_t(value) << 16;
    float result;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

inline void copy_bf16_to_f32(
    const std::uint16_t* input,
    float* output,
    std::size_t count) {
    const auto begin = M0TimingCounters::Clock::now();
    for (std::size_t index = 0; index < count; ++index) {
        output[index] = bf16_to_f32(input[index]);
    }
    m0_timing_counters().record_bf16_conversion(
        M0TimingCounters::Clock::now() - begin);
}

}  // namespace pie::metal::batch
