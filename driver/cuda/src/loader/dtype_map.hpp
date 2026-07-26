#pragma once

// Pure mappings between the Rust loader's wire enums and the driver's runtime
// types. No state, no CUDA — factored out of the storage executor so its body
// stays materialize logic.

#include "pie_loader/plan.hpp"
#include "tensor.hpp"
#include "loader/tensor_spec.hpp"
#include "model/weight_store.hpp"

namespace pie_cuda_driver {

inline QuantMeta::Kind quant_meta_kind(
    pie_loader::PieLoaderQuantGranularity granularity)
{
    switch (granularity) {
    case pie_loader::PieLoaderQuantGranularity::PerChannel:
        return QuantMeta::Kind::PerChannel;
    case pie_loader::PieLoaderQuantGranularity::PerGroup:
        return QuantMeta::Kind::PerGroup;
    }
    return QuantMeta::Kind::PerTensor;
}

inline DType dtype_from_rust(pie_loader::PieLoaderDType dtype)
{
    switch (dtype) {
    case pie_loader::PieLoaderDType::F32: return DType::FP32;
    case pie_loader::PieLoaderDType::F16: return DType::FP16;
    case pie_loader::PieLoaderDType::BF16: return DType::BF16;
    case pie_loader::PieLoaderDType::F8E4M3: return DType::FP8_E4M3;
    case pie_loader::PieLoaderDType::F8E5M2: return DType::FP8_E5M2;
    case pie_loader::PieLoaderDType::I32: return DType::INT32;
    // 8-byte integer index tables (DeepSeek-V4 `tid2eid`). The driver has no
    // unsigned 64-bit tag; INT64 is the same width and the kernels reinterpret
    // the bytes anyway.
    case pie_loader::PieLoaderDType::I64:
    case pie_loader::PieLoaderDType::U64: return DType::INT64;
    case pie_loader::PieLoaderDType::I8: return DType::INT8;
    case pie_loader::PieLoaderDType::U8:
    case pie_loader::PieLoaderDType::Bool:
    case pie_loader::PieLoaderDType::I16:
    case pie_loader::PieLoaderDType::U16:
    case pie_loader::PieLoaderDType::U32:
        return DType::UINT8;
    }
    return DType::UINT8;
}

inline DType quant_physical_dtype(
    const pie_loader::PieLoaderTensorDeclView& tensor)
{
    switch (tensor.quant_scheme) {
    case pie_loader::PieLoaderQuantScheme::Fp8E4M3: return DType::FP8_E4M3;
    case pie_loader::PieLoaderQuantScheme::Int8Symmetric: return DType::INT8;
    default: return DType::UINT8;
    }
}

}  // namespace pie_cuda_driver
