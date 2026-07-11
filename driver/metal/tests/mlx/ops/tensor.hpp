#pragma once

// Tensor type for the Metal (MLX) driver.
//
// Unlike the CUDA driver's owning `DeviceTensor` (a RAII wrapper over a
// `cudaMalloc`'d buffer), the Metal driver uses MLX's `mlx::core::array`
// directly as its tensor type. MLX arrays are:
//   * reference-counted (cheap to copy/pass by value),
//   * lazily evaluated (ops build a graph; `eval()` materializes),
//   * carriers of their own dtype + shape.
//
// So there is no separate owning wrapper here — `Tensor` is an alias for
// `mlx::core::array`. This header adds the small glue the rest of the
// driver needs: a driver-side `DType` enum that mirrors the CUDA driver's,
// conversions to/from MLX's `Dtype`, and a couple of eval helpers.

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include <mlx/array.h>
#include <mlx/dtype.h>
#include <mlx/ops.h>
#include <mlx/transforms.h>

namespace pie_metal_driver {

// The driver tensor type. A lazily-evaluated, reference-counted MLX array.
using Tensor = mlx::core::array;

// Driver-side dtype tag. Mirrors `pie_cuda_driver::DType` so loader/model
// code that was written against the CUDA naming reads the same. Only the
// subset MLX can represent on Metal is wired through to a real MLX dtype;
// the packed-quant tags are carried for metadata but stored as raw bytes.
enum class DType : std::uint8_t {
    BF16  = 0,
    FP16  = 1,
    FP32  = 2,
    INT8  = 3,
    INT32 = 4,
    INT64 = 5,
    UINT8 = 6,
    UINT32 = 7,
};

inline std::size_t dtype_bytes(DType d) {
    switch (d) {
        case DType::BF16:   return 2;
        case DType::FP16:   return 2;
        case DType::FP32:   return 4;
        case DType::INT8:   return 1;
        case DType::INT32:  return 4;
        case DType::INT64:  return 8;
        case DType::UINT8:  return 1;
        case DType::UINT32: return 4;
    }
    throw std::runtime_error("pie_metal_driver: unknown dtype");
}

inline const char* dtype_name(DType d) {
    switch (d) {
        case DType::BF16:   return "bf16";
        case DType::FP16:   return "fp16";
        case DType::FP32:   return "fp32";
        case DType::INT8:   return "int8";
        case DType::INT32:  return "int32";
        case DType::INT64:  return "int64";
        case DType::UINT8:  return "u8";
        case DType::UINT32: return "u32";
    }
    return "?";
}

// Driver DType -> MLX Dtype.
inline mlx::core::Dtype to_mlx_dtype(DType d) {
    switch (d) {
        case DType::BF16:   return mlx::core::bfloat16;
        case DType::FP16:   return mlx::core::float16;
        case DType::FP32:   return mlx::core::float32;
        case DType::INT8:   return mlx::core::int8;
        case DType::INT32:  return mlx::core::int32;
        case DType::INT64:  return mlx::core::int64;
        case DType::UINT8:  return mlx::core::uint8;
        case DType::UINT32: return mlx::core::uint32;
    }
    throw std::runtime_error("pie_metal_driver: unmappable dtype");
}

// MLX Dtype -> driver DType. Throws on a dtype the driver doesn't tag.
inline DType from_mlx_dtype(mlx::core::Dtype d) {
    if (d == mlx::core::bfloat16) return DType::BF16;
    if (d == mlx::core::float16)  return DType::FP16;
    if (d == mlx::core::float32)  return DType::FP32;
    if (d == mlx::core::int8)     return DType::INT8;
    if (d == mlx::core::int32)    return DType::INT32;
    if (d == mlx::core::int64)    return DType::INT64;
    if (d == mlx::core::uint8)    return DType::UINT8;
    if (d == mlx::core::uint32)   return DType::UINT32;
    throw std::runtime_error("pie_metal_driver: unmappable mlx dtype");
}

// Parse the safetensors dtype string ("BF16", "F16", "F32", "I8", ...).
inline DType dtype_from_safetensors(const std::string& s) {
    if (s == "BF16") return DType::BF16;
    if (s == "F16")  return DType::FP16;
    if (s == "F32")  return DType::FP32;
    if (s == "I8")   return DType::INT8;
    if (s == "I32")  return DType::INT32;
    if (s == "I64")  return DType::INT64;
    if (s == "U8")   return DType::UINT8;
    if (s == "U32")  return DType::UINT32;
    throw std::runtime_error("pie_metal_driver: unsupported safetensors dtype " + s);
}

// Force materialization of one tensor (and anything it depends on).
inline void eval(const Tensor& t) { mlx::core::eval(t); }

// Force materialization of several tensors in one graph evaluation.
inline void eval(std::vector<Tensor> ts) { mlx::core::eval(std::move(ts)); }

namespace ops {

// Canonical placeholder for a not-yet-bound Tensor. MLX `array` has no default
// constructor, so structs with non-optional `Tensor` members (e.g. model
// weights) aren't default-constructible. Use this as an in-class initializer
// (`Tensor w = ops::empty_tensor();`) so such structs can be default-built and
// the real value assigned later. Returns an empty [0] float32 array (no alloc).
inline Tensor empty_tensor() { return mlx::core::zeros({0}); }

}  // namespace ops

}  // namespace pie_metal_driver
