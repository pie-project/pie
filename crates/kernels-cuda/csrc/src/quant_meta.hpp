#pragma once

// Per-weight quantization metadata, threaded from the loaded model's weight
// store through to the GEMM dispatcher (`gemm/gemm.hpp`'s `WeightView`).
// Kernel-owned: it describes how a GEMM operand should be scaled or
// dequantized, not how the weight is stored. model/weight_store.hpp re-exports this under
// its historical `pie_cuda_driver::QuantMeta` name for its own map/lookup
// API; the pointers reference DeviceTensors registered separately under
// their own names in that store — QuantMeta does not own those tensors.

#include <string>

#include "tensor.hpp"

namespace pie_cuda_driver {

struct QuantMeta {
    enum class Kind { PerTensor, PerChannel, PerGroup };
    Kind kind = Kind::PerTensor;
    std::string scale_name;
    std::string zero_point_name;
    const DeviceTensor* scale = nullptr;
    const DeviceTensor* zero_point = nullptr;
    int group_size = 0;
    int channel_axis = 0;
};

}  // namespace pie_cuda_driver
