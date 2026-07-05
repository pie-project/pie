#pragma once

// Pure mappings between the Rust loader's wire enums and the driver's runtime
// types. No state, no CUDA — factored out of the storage executor so its body
// stays materialize logic.

#include "../../../weight-loader/include/weight_loader.h"
#include "tensor.hpp"
#include "loader/tensor_spec.hpp"
#include "model/weight_store.hpp"

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
    default: return DType::UINT8;
    }
}

}  // namespace pie_cuda_driver
