#pragma once

// GLM-5 section-index map for streamed routed experts.
//
// Order must match `GLM_EXPERT_SECTIONS` in
// `driver/weight_loader/src/abi.rs`:
//   gate_proj.weight, gate_proj.weight_scale_inv,
//   up_proj.weight,   up_proj.weight_scale_inv,
//   down_proj.weight, down_proj.weight_scale_inv
//
// Streamed experts are native FP8 E4M3 + FP32 block scale_inv (group 128).

#include <cstddef>
#include <cstdint>
#include <stdexcept>

#include "expert_stream_cache.hpp"
#include "ops/gemm.hpp"
#include "model/weight_store.hpp"
#include "tensor.hpp"

namespace pie_cuda_driver {
namespace model {

inline constexpr int kGlm5ExpertSectionCount = 6;
enum Glm5ExpertSection : int {
    kGlm5Gate = 0,
    kGlm5GateScale = 1,
    kGlm5Up = 2,
    kGlm5UpScale = 3,
    kGlm5Down = 4,
    kGlm5DownScale = 5,
};

inline void require_glm5_sections(const ExpertSectionPointers& p)
{
    if (p.num_sections() != kGlm5ExpertSectionCount) {
        throw std::runtime_error(
            "glm5 expert streaming: expected " +
            std::to_string(kGlm5ExpertSectionCount) + " sections, got " +
            std::to_string(p.num_sections()));
    }
}

// Build an FP8 PerGroup/128 WeightView from streamed weight + scale_inv
// section pointers (matches resident GLM QuantMeta attachments).
inline ops::WeightView glm5_fp8_weight_view(
    const std::uint8_t* weight,
    std::uint64_t weight_bytes,
    const std::uint8_t* scale_inv,
    std::uint64_t scale_bytes)
{
    ops::WeightView v;
    v.data = weight;
    v.dtype = DType::FP8_E4M3;
    v.nbytes = static_cast<std::size_t>(weight_bytes);
    v.scale_data = scale_inv;
    v.scale_dtype = DType::FP32;
    v.scale_numel = static_cast<std::size_t>(scale_bytes / sizeof(float));
    v.quant_kind = QuantMeta::Kind::PerGroup;
    v.group_size = 128;
    v.channel_axis = 0;
    return v;
}

inline ops::WeightView glm5_gate_view(
    const ExpertSectionPointers& p,
    const StreamedExpertTable& table)
{
    return glm5_fp8_weight_view(
        p.at(kGlm5Gate), table.section_bytes[kGlm5Gate],
        p.at(kGlm5GateScale), table.section_bytes[kGlm5GateScale]);
}

inline ops::WeightView glm5_up_view(
    const ExpertSectionPointers& p,
    const StreamedExpertTable& table)
{
    return glm5_fp8_weight_view(
        p.at(kGlm5Up), table.section_bytes[kGlm5Up],
        p.at(kGlm5UpScale), table.section_bytes[kGlm5UpScale]);
}

inline ops::WeightView glm5_down_view(
    const ExpertSectionPointers& p,
    const StreamedExpertTable& table)
{
    return glm5_fp8_weight_view(
        p.at(kGlm5Down), table.section_bytes[kGlm5Down],
        p.at(kGlm5DownScale), table.section_bytes[kGlm5DownScale]);
}

}  // namespace model
}  // namespace pie_cuda_driver
