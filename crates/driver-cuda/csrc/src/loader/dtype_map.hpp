#pragma once

// Pure mappings between the Rust loader's wire enums and the driver's runtime
// types. No state, no CUDA — factored out of the storage executor so its body
// stays materialize logic.

#include <string_view>

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
    case pie_loader::PieLoaderDType::E8M0: return DType::E8M0;
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

// The container type an artifact stores this driver type's bytes as.
//
// Not the inverse of `dtype_from_rust`, and it cannot be: the driver's set is
// larger. Its tile-packed types are a *layout* over bytes, which the loader's
// vocabulary has no name for and should not grow one for -- what an artifact
// records instead is the width, so any reader can see a well-formed tensor,
// with `dtype_name` carried alongside for the kernels that need the layout.
//
// Every arm here must agree with `dtype_bytes` on width, since that equation is
// what makes an object's declared shape describe its payload.
inline pie_loader::PieLoaderDType dtype_to_rust(DType dtype)
{
    switch (dtype) {
    case DType::BF16: return pie_loader::PieLoaderDType::BF16;
    case DType::FP16: return pie_loader::PieLoaderDType::F16;
    case DType::FP32: return pie_loader::PieLoaderDType::F32;
    case DType::INT8: return pie_loader::PieLoaderDType::I8;
    case DType::INT32: return pie_loader::PieLoaderDType::I32;
    case DType::INT64: return pie_loader::PieLoaderDType::I64;
    case DType::FP8_E4M3: return pie_loader::PieLoaderDType::F8E4M3;
    case DType::FP8_E5M2: return pie_loader::PieLoaderDType::F8E5M2;
    case DType::E8M0: return pie_loader::PieLoaderDType::E8M0;
    case DType::UINT8:
    case DType::INT4_PACKED:
    case DType::MXFP4_PACKED:
        return pie_loader::PieLoaderDType::U8;
    }
    return pie_loader::PieLoaderDType::U8;
}

// The driver type `dtype_name` printed. Returns false for a name this build
// does not know, which is how an artifact from a different driver is refused
// rather than misread.
inline bool dtype_by_name(std::string_view name, DType& out)
{
    for (const DType candidate : {
             DType::BF16, DType::FP16, DType::FP32, DType::INT8, DType::INT32,
             DType::INT64, DType::UINT8, DType::FP8_E4M3, DType::FP8_E5M2,
             DType::INT4_PACKED, DType::MXFP4_PACKED, DType::E8M0}) {
        if (name == dtype_name(candidate)) {
            out = candidate;
            return true;
        }
    }
    return false;
}

// How a quantized weight's scales are applied, as an artifact spells it.
inline const char* quant_kind_name(QuantMeta::Kind kind)
{
    switch (kind) {
    case QuantMeta::Kind::PerChannel: return "per_channel";
    case QuantMeta::Kind::PerGroup: return "per_group";
    case QuantMeta::Kind::PerTensor: break;
    }
    return "per_tensor";
}

inline bool quant_kind_by_name(std::string_view name, QuantMeta::Kind& out)
{
    for (const QuantMeta::Kind candidate : {QuantMeta::Kind::PerTensor,
                                            QuantMeta::Kind::PerChannel,
                                            QuantMeta::Kind::PerGroup}) {
        if (name == quant_kind_name(candidate)) {
            out = candidate;
            return true;
        }
    }
    return false;
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
