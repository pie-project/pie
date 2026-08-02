#pragma once

#include <string>

#include "pie_loader/plan.hpp"

namespace pie_cuda_driver {

struct RustQuantAttachment {
    std::string tensor_name;
    std::string scale_tensor_name;
    pie_loader::PieLoaderQuantGranularity granularity =
        pie_loader::PieLoaderQuantGranularity::PerChannel;
    int group_size = 0;
    int channel_axis = 0;
    /// Whether the kernels want the scale bytes as-is or expanded to F32.
    /// Stated by the loader, which chose the encoding.
    pie_loader::PieLoaderScaleForm scale_form =
        pie_loader::PieLoaderScaleForm::F32Factors;
};

}  // namespace pie_cuda_driver
