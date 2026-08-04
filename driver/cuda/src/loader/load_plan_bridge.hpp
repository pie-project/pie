#pragma once

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "pie_loader/source_checkpoint.hpp"

#include "loader/rust_quant_attachment.hpp"
#include "loader/load_plan.hpp"

#include "model/config.hpp"

namespace pie_cuda_driver {

struct LoadPlanResult {
    LoadPlan plan;
    std::vector<RustQuantAttachment> quant_attachments;
    /// Tensors the plan produces. The only count anything reads: it sizes the
    /// weight store's reservation.
    std::size_t planned_tensor_count = 0;
    std::string cache_key;
};

/// Resolve the plan's quant attachments into the name-keyed form the executor's
/// `WeightStore` is addressed by.
///
/// Only a translation. The pairing itself used to be *inferred* here by matching
/// name suffixes over the plan's tensor list, and then, for a while, by matching
/// them one layer earlier inside the loader. It is now recorded by whoever
/// declares the scale tensor: `plan/build.rs::quant_metadata_outputs` for scales
/// the loader writes, and the contract's `scales` field — see
/// `dsv4_block_scales_to_fp32` -- for scales the checkpoint shipped.
inline std::vector<RustQuantAttachment> resolve_quant_attachments(
    const pie_loader::LoadPlanView& view) {
    std::unordered_map<std::uint32_t, pie_loader::PieLoaderBytes> names;
    names.reserve(view.tensors.len);
    for (std::size_t i = 0; i < view.tensors.len; ++i) {
        names.emplace(view.tensors.ptr[i].id, view.tensors.ptr[i].name);
    }

    std::vector<RustQuantAttachment> out;
    out.reserve(view.attachments.len);
    for (std::size_t i = 0; i < view.attachments.len; ++i) {
        const auto& attachment = view.attachments.ptr[i];
        const auto tensor = names.find(attachment.tensor_id);
        const auto scale = names.find(attachment.scale_tensor_id);
        if (tensor == names.end() || scale == names.end()) {
            // Both ids index the same table the loader built this from, so a
            // miss is a marshalling fault rather than a model that lacks scales.
            throw std::runtime_error(
                "engine: Rust loader attached a quant scale to a tensor id the "
                "plan does not declare");
        }
        out.push_back({
            .tensor_name = pie_loader::bytes_to_string(tensor->second),
            .scale_tensor_name = pie_loader::bytes_to_string(scale->second),
            .granularity = attachment.granularity,
            .group_size = static_cast<int>(attachment.group_size),
            .channel_axis = static_cast<int>(attachment.channel_axis),
            .scale_form = attachment.scale_form,
        });
    }
    return out;
}

// `prepare_load_plan` — the contract-request compile — was harvested with
// the C++ authors it served (`plan/model-in-rust.md` §8-5). The boot
// compiles through `loader/rust_author.hpp`; what this header still owns is
// the plan-side translation below.

}  // namespace pie_cuda_driver
