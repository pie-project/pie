#pragma once

// Qwen MoE section-index maps for streamed BF16 experts.
//
// Orders must match `QWEN3_MOE_EXPERT_SECTIONS` / `QWEN35_MOE_EXPERT_SECTIONS`
// in `driver/weight_loader/src/stream_arch.rs`.

#include <cstdint>
#include <stdexcept>
#include <string>

#include "expert_stream_cache.hpp"

namespace pie_cuda_driver {
namespace model {

// Plain Qwen3-MoE: per-expert gate / up / down.
inline constexpr int kQwen3MoeExpertSectionCount = 3;
enum Qwen3MoeExpertSection : int {
    kQwen3MoeGate = 0,
    kQwen3MoeUp = 1,
    kQwen3MoeDown = 2,
};

inline void require_qwen3_moe_sections(const ExpertSectionPointers& p)
{
    if (p.num_sections() != kQwen3MoeExpertSectionCount) {
        throw std::runtime_error(
            "qwen3_moe expert streaming: expected " +
            std::to_string(kQwen3MoeExpertSectionCount) + " sections, got " +
            std::to_string(p.num_sections()));
    }
}

inline const std::uint8_t* qwen3_moe_gate(const ExpertSectionPointers& p)
{
    return p.at(kQwen3MoeGate);
}
inline const std::uint8_t* qwen3_moe_up(const ExpertSectionPointers& p)
{
    return p.at(kQwen3MoeUp);
}
inline const std::uint8_t* qwen3_moe_down(const ExpertSectionPointers& p)
{
    return p.at(kQwen3MoeDown);
}

// Qwen3.5/3.6-MoE: fused gate_up + down banks.
inline constexpr int kQwen35MoeExpertSectionCount = 2;
enum Qwen35MoeExpertSection : int {
    kQwen35MoeGateUp = 0,
    kQwen35MoeDown = 1,
};

inline void require_qwen35_moe_sections(const ExpertSectionPointers& p)
{
    if (p.num_sections() != kQwen35MoeExpertSectionCount) {
        throw std::runtime_error(
            "qwen3_5_moe expert streaming: expected " +
            std::to_string(kQwen35MoeExpertSectionCount) + " sections, got " +
            std::to_string(p.num_sections()));
    }
}

inline const std::uint8_t* qwen35_moe_gate_up(const ExpertSectionPointers& p)
{
    return p.at(kQwen35MoeGateUp);
}
inline const std::uint8_t* qwen35_moe_down(const ExpertSectionPointers& p)
{
    return p.at(kQwen35MoeDown);
}

}  // namespace model
}  // namespace pie_cuda_driver
