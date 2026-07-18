#pragma once

#include <string>

#include "loader/tensor_spec.hpp"

namespace pie_cuda_driver {

struct RustQuantAttachment {
    std::string tensor_name;
    std::string scale_tensor_name;
    std::string zero_point_tensor_name;
    QuantGranularity granularity = QuantGranularity::None;
    int group_size = 0;
    int channel_axis = 0;
    bool preserve_scale_dtype = false;
};

}  // namespace pie_cuda_driver
