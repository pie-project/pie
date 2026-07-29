#pragma once

// GPT-OSS section-index maps for streamed experts.
//
// RoutedDequant (4): HF MXFP4 packs — must match `GPT_OSS_EXPERT_SECTIONS`.
// Native Marlin (6): offline pack — must match `GPT_OSS_NATIVE_EXPERT_SECTIONS`.
// Eager BF16 (3): offline dequant pack — must match `GPT_OSS_EAGER_BF16_EXPERT_SECTIONS`.
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

inline constexpr int kGptOssNativeExpertSectionCount = 6;
enum GptOssNativeExpertSection : int {
    kGptOssNativeGate = 0,
    kGptOssNativeGateScale = 1,
    kGptOssNativeUp = 2,
    kGptOssNativeUpScale = 3,
    kGptOssNativeDown = 4,
    kGptOssNativeDownScale = 5,
};

inline void require_gpt_oss_native_sections(const ExpertSectionPointers& p)
{
    if (p.num_sections() != kGptOssNativeExpertSectionCount) {
        throw std::runtime_error(
            "gpt_oss native expert streaming: expected " +
            std::to_string(kGptOssNativeExpertSectionCount) +
            " sections, got " + std::to_string(p.num_sections()));
    }
}

inline const std::uint8_t* gpt_oss_native_gate(const ExpertSectionPointers& p)
{
    return p.at(kGptOssNativeGate);
}
inline const std::uint8_t* gpt_oss_native_gate_scale(const ExpertSectionPointers& p)
{
    return p.at(kGptOssNativeGateScale);
}
inline const std::uint8_t* gpt_oss_native_up(const ExpertSectionPointers& p)
{
    return p.at(kGptOssNativeUp);
}
inline const std::uint8_t* gpt_oss_native_up_scale(const ExpertSectionPointers& p)
{
    return p.at(kGptOssNativeUpScale);
}
inline const std::uint8_t* gpt_oss_native_down(const ExpertSectionPointers& p)
{
    return p.at(kGptOssNativeDown);
}
inline const std::uint8_t* gpt_oss_native_down_scale(const ExpertSectionPointers& p)
{
    return p.at(kGptOssNativeDownScale);
}

inline constexpr int kGptOssEagerBf16ExpertSectionCount = 3;
enum GptOssEagerBf16ExpertSection : int {
    kGptOssEagerGate = 0,
    kGptOssEagerUp = 1,
    kGptOssEagerDown = 2,
};

inline void require_gpt_oss_eager_bf16_sections(const ExpertSectionPointers& p)
{
    if (p.num_sections() != kGptOssEagerBf16ExpertSectionCount) {
        throw std::runtime_error(
            "gpt_oss eager BF16 expert streaming: expected " +
            std::to_string(kGptOssEagerBf16ExpertSectionCount) +
            " sections, got " + std::to_string(p.num_sections()));
    }
}

inline const std::uint8_t* gpt_oss_eager_gate(const ExpertSectionPointers& p)
{
    return p.at(kGptOssEagerGate);
}
inline const std::uint8_t* gpt_oss_eager_up(const ExpertSectionPointers& p)
{
    return p.at(kGptOssEagerUp);
}
inline const std::uint8_t* gpt_oss_eager_down(const ExpertSectionPointers& p)
{
    return p.at(kGptOssEagerDown);
}

}  // namespace model
}  // namespace pie_cuda_driver
