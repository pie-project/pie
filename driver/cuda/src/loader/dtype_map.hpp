#pragma once

// Pure mappings between the Rust loader's wire enums and the driver's runtime
// types. No state, no CUDA — factored out of the storage executor so its body
// stays materialize logic.

#include "../../../weight_loader/include/weight_loader.h"
#include "tensor.hpp"
#include "loader/tensor_spec.hpp"
#include "model/weight_store.hpp"

#include <stdexcept>
#include <vector>

namespace pie_cuda_driver {

inline QuantMeta::Kind quant_meta_kind(QuantGranularity granularity)
{
    switch (granularity) {
    case QuantGranularity::PerTensor: return QuantMeta::Kind::PerTensor;
    case QuantGranularity::PerChannel: return QuantMeta::Kind::PerChannel;
    case QuantGranularity::PerGroup: return QuantMeta::Kind::PerGroup;
    case QuantGranularity::None: break;
    }
    return QuantMeta::Kind::PerTensor;
}

inline DType dtype_from_rust(pie_weight_loader::PieLoaderDType dtype)
{
    switch (dtype) {
    case pie_weight_loader::PieLoaderDType::F32: return DType::FP32;
    case pie_weight_loader::PieLoaderDType::F16: return DType::FP16;
    case pie_weight_loader::PieLoaderDType::BF16: return DType::BF16;
    case pie_weight_loader::PieLoaderDType::F8E4M3: return DType::FP8_E4M3;
    case pie_weight_loader::PieLoaderDType::F8E5M2: return DType::FP8_E5M2;
    case pie_weight_loader::PieLoaderDType::I32: return DType::INT32;
    case pie_weight_loader::PieLoaderDType::I8: return DType::INT8;
    case pie_weight_loader::PieLoaderDType::U8:
    case pie_weight_loader::PieLoaderDType::Bool:
    case pie_weight_loader::PieLoaderDType::I16:
    case pie_weight_loader::PieLoaderDType::U16:
    case pie_weight_loader::PieLoaderDType::U32:
        return DType::UINT8;
    }
    return DType::UINT8;
}

inline DType quant_physical_dtype(
    const pie_weight_loader::PieLoaderTensorDeclView& tensor)
{
    switch (tensor.quant_scheme) {
    case pie_weight_loader::PieLoaderQuantScheme::Fp8E4M3: return DType::FP8_E4M3;
    case pie_weight_loader::PieLoaderQuantScheme::Int8Symmetric: return DType::INT8;
    case pie_weight_loader::PieLoaderQuantScheme::GptqInt4:
    case pie_weight_loader::PieLoaderQuantScheme::AwqInt4:
        return DType::INT4_PACKED;
    default: return DType::UINT8;
    }
}

inline std::vector<std::int64_t> quant_physical_shape(
    const pie_weight_loader::PieLoaderTensorDeclView& tensor,
    DType physical)
{
    auto logical = pie_weight_loader::cpp::i64_slice_to_vector(tensor.shape);
    if (physical != DType::INT4_PACKED) return logical;
    if (logical.size() != 2 || logical[0] <= 0 || logical[1] <= 0 ||
        logical[1] % 16 != 0 || logical[0] % 64 != 0) {
        throw std::runtime_error(
            "rust storage executor: Marlin INT4 logical weight shape must be [N,K] with N%64=0 and K%16=0");
    }
    return {logical[1] / 16, logical[0] * 8};
}

}  // namespace pie_cuda_driver
