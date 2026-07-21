#pragma once

// GPT-OSS section-index map for streamed experts (RoutedDequant path).
//
// Order must match `GPT_OSS_EXPERT_SECTIONS` in
// `driver/weight_loader/src/stream_arch.rs`:
//   gate_up.weight, gate_up.scale, down.weight, down.scale
// Biases stay resident and are not streamed.

#include <cstdint>
#include <stdexcept>
#include <string>

#include "expert_stream_cache.hpp"

namespace pie_cuda_driver {
namespace model {

inline constexpr int kGptOssExpertSectionCount = 4;
enum GptOssExpertSection : int {
    kGptOssGateUp = 0,
    kGptOssGateUpScale = 1,
    kGptOssDown = 2,
    kGptOssDownScale = 3,
};

inline void require_gpt_oss_sections(const ExpertSectionPointers& p)
{
    if (p.num_sections() != kGptOssExpertSectionCount) {
        throw std::runtime_error(
            "gpt_oss expert streaming: expected " +
            std::to_string(kGptOssExpertSectionCount) + " sections, got " +
            std::to_string(p.num_sections()));
    }
}

inline const std::uint8_t* gpt_oss_gate_up(const ExpertSectionPointers& p)
{
    return p.at(kGptOssGateUp);
}
inline const std::uint8_t* gpt_oss_gate_up_scale(const ExpertSectionPointers& p)
{
    return p.at(kGptOssGateUpScale);
}
inline const std::uint8_t* gpt_oss_down(const ExpertSectionPointers& p)
{
    return p.at(kGptOssDown);
}
inline const std::uint8_t* gpt_oss_down_scale(const ExpertSectionPointers& p)
{
    return p.at(kGptOssDownScale);
}

}  // namespace model
}  // namespace pie_cuda_driver
