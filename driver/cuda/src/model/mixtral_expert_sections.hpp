#pragma once

// Mixtral section-index map for streamed BF16 experts.
//
// Order must match `MIXTRAL_EXPERT_SECTIONS` in
// `driver/weight_loader/src/stream_arch.rs`:
//   w1.weight, w2.weight, w3.weight
// HF layout: w1=gate, w2=down, w3=up. Router stays resident.

#include <cstdint>
#include <stdexcept>
#include <string>

#include "expert_stream_cache.hpp"

namespace pie_cuda_driver {
namespace model {

inline constexpr int kMixtralExpertSectionCount = 3;
enum MixtralExpertSection : int {
    kMixtralW1 = 0,  // gate
    kMixtralW2 = 1,  // down
    kMixtralW3 = 2,  // up
};

inline void require_mixtral_sections(const ExpertSectionPointers& p)
{
    if (p.num_sections() != kMixtralExpertSectionCount) {
        throw std::runtime_error(
            "mixtral expert streaming: expected " +
            std::to_string(kMixtralExpertSectionCount) + " sections, got " +
            std::to_string(p.num_sections()));
    }
}

inline const std::uint8_t* mixtral_w1(const ExpertSectionPointers& p)
{
    return p.at(kMixtralW1);
}
inline const std::uint8_t* mixtral_w2(const ExpertSectionPointers& p)
{
    return p.at(kMixtralW2);
}
inline const std::uint8_t* mixtral_w3(const ExpertSectionPointers& p)
{
    return p.at(kMixtralW3);
}

}  // namespace model
}  // namespace pie_cuda_driver
