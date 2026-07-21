#pragma once

// DeepSeek-V4 section-index map for streamed experts.
//
// Order must match `DSV4_EXPERT_SECTIONS` in
// `driver/weight_loader/src/abi.rs`:
//   w1.weight, w1.scale, w2.weight, w2.scale, w3.weight, w3.scale

#include <cstdint>
#include <stdexcept>

#include "expert_stream_cache.hpp"

namespace pie_cuda_driver {
namespace model {

inline constexpr int kDsv4ExpertSectionCount = 6;
enum Dsv4ExpertSection : int {
    kDsv4W1 = 0,
    kDsv4W1Scale = 1,
    kDsv4W2 = 2,
    kDsv4W2Scale = 3,
    kDsv4W3 = 4,
    kDsv4W3Scale = 5,
};

inline void require_dsv4_sections(const ExpertSectionPointers& p)
{
    if (p.num_sections() != kDsv4ExpertSectionCount) {
        throw std::runtime_error(
            "dsv4 expert streaming: expected " +
            std::to_string(kDsv4ExpertSectionCount) + " sections, got " +
            std::to_string(p.num_sections()));
    }
}

inline const std::uint8_t* dsv4_w1(const ExpertSectionPointers& p)
{
    return p.at(kDsv4W1);
}
inline const std::uint8_t* dsv4_w1_scale(const ExpertSectionPointers& p)
{
    return p.at(kDsv4W1Scale);
}
inline const std::uint8_t* dsv4_w2(const ExpertSectionPointers& p)
{
    return p.at(kDsv4W2);
}
inline const std::uint8_t* dsv4_w2_scale(const ExpertSectionPointers& p)
{
    return p.at(kDsv4W2Scale);
}
inline const std::uint8_t* dsv4_w3(const ExpertSectionPointers& p)
{
    return p.at(kDsv4W3);
}
inline const std::uint8_t* dsv4_w3_scale(const ExpertSectionPointers& p)
{
    return p.at(kDsv4W3Scale);
}

}  // namespace model
}  // namespace pie_cuda_driver
