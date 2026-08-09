// Oracle for `WeightView`, `make_weight_view` and the weight schema.
//
// Compiles the REAL `weight_view.hpp`, `quant_meta.hpp` and `tensor.hpp` with
// **no stubs at all** — they include only the standard library, and
// `DeviceTensor::view` builds a tensor without touching CUDA. That makes this
// the least-mediated oracle in the set: what it measures is the shipping type.
//
// # Why this one measures offsets
//
// `WeightView` is passed to kernel launchers **by value**, across the
// driver/kernels crate boundary. A Rust `#[repr(C)]` mirror is only correct if
// every field lands at the same byte, and nothing about a mirror that is wrong
// by four bytes looks wrong — the GEMM reads a scale pointer out of the
// padding and dereferences it. So the first half of the transcript is the ABI:
// `sizeof`, `alignof`, and `offsetof` for all eleven fields plus the two enum
// underlying types.
//
// # Why it also measures which fields each factory leaves alone
//
// There are five ways to make a `WeightView` and they fill different subsets.
// `raw()` in particular leaves `nbytes` at 0 — which is safe only because the
// one reader of `nbytes` (`validate_quant_weight_view`) is gated behind
// `scale_data != nullptr`, and `raw()` leaves that null too. That is a
// load-bearing coincidence between two functions in different files, so the
// transcript records every field of every factory rather than the ones that
// looked interesting.

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <optional>
#include <string>
#include <type_traits>
#include <vector>

#include "quant_meta.hpp"
#include "tensor.hpp"
#include "weight_view.hpp"

// The real `tensor.cpp` is linked in, so the two CUDA entry points it names
// have to exist. Neither is ever called: every tensor below is a non-owning
// `DeviceTensor::view`, and `free_()` frees only when `owns_memory_`.
cudaError_t cudaMalloc(void**, std::size_t) { return cudaSuccess; }
cudaError_t cudaFree(void*) { return cudaSuccess; }
cudaError_t cudaStreamBeginCapture(cudaStream_t, cudaStreamCaptureMode) {
    return cudaSuccess;
}
cudaError_t cudaStreamEndCapture(cudaStream_t, cudaGraph_t*) {
    return cudaSuccess;
}
cudaError_t cudaGraphDestroy(cudaGraph_t) { return cudaSuccess; }
cudaError_t cudaGetLastError() { return cudaSuccess; }
const char* cudaGetErrorString(cudaError_t) { return "stub"; }

namespace {

constexpr char kSep = '\x1f';

using pie_cuda_driver::DeviceTensor;
using pie_cuda_driver::DType;
using pie_cuda_driver::QuantMeta;
using pie_cuda_driver::WeightView;

// Backing bytes for the non-owning tensors. Never dereferenced — the views
// only ever have their `data()` read back out — but distinct so a factory that
// copied the wrong pointer is visible.
alignas(256) unsigned char g_slab[1 << 16];

const unsigned char* g_base = g_slab;

std::string at(const void* p) {
    if (p == nullptr) return "null";
    const auto* c = static_cast<const unsigned char*>(p);
    return "+" + std::to_string(static_cast<std::size_t>(c - g_base));
}

DeviceTensor tensor(std::size_t offset, DType dtype,
                    std::vector<std::int64_t> shape) {
    return DeviceTensor::view(g_slab + offset, dtype, std::move(shape));
}

// ---------------------------------------------------------------------------
// Script 1 — the ABI.
// ---------------------------------------------------------------------------

void script_abi() {
    std::printf("abi%csizeof%c%zu\n", kSep, kSep, sizeof(WeightView));
    std::printf("abi%calignof%c%zu\n", kSep, kSep, alignof(WeightView));
    // Standard layout is what makes `offsetof` meaningful and what makes a
    // `#[repr(C)]` mirror legitimate in the first place.
    std::printf(
        "abi%cstandard_layout%c%d\n", kSep, kSep,
        std::is_standard_layout<WeightView>::value ? 1 : 0);
    std::printf(
        "abi%ctrivially_copyable%c%d\n", kSep, kSep,
        std::is_trivially_copyable<WeightView>::value ? 1 : 0);

#define FIELD(name)                                                       \
    std::printf(                                                          \
        "abi%coffsetof%c%s%c%zu%c%zu\n", kSep, kSep, #name, kSep,         \
        offsetof(WeightView, name), kSep, sizeof(WeightView::name))
    FIELD(data);
    FIELD(dtype);
    FIELD(nbytes);
    FIELD(scale_data);
    FIELD(scale_dtype);
    FIELD(scale_numel);
    FIELD(quant_kind);
    FIELD(zero_point_data);
    FIELD(group_size);
    FIELD(channel_axis);
#undef FIELD

    // The two enums travel inside the struct, so their underlying types are
    // part of the ABI just as much as the field offsets are.
    std::printf(
        "abi%cenum%cDType%c%zu%c%d\n", kSep, kSep, kSep, sizeof(DType), kSep,
        std::is_signed<std::underlying_type<DType>::type>::value ? 1 : 0);
    std::printf(
        "abi%cenum%cQuantKind%c%zu%c%d\n", kSep, kSep, kSep,
        sizeof(QuantMeta::Kind), kSep,
        std::is_signed<std::underlying_type<QuantMeta::Kind>::type>::value ? 1
                                                                          : 0);
    for (int k = 0; k <= 2; ++k) {
        std::printf(
            "abi%cquant_kind_value%c%d%c%d\n", kSep, kSep, k, kSep,
            static_cast<int>(static_cast<QuantMeta::Kind>(k)));
    }
}

// ---------------------------------------------------------------------------
// Script 2 — every field of every factory.
// ---------------------------------------------------------------------------

void dump(const char* label, const WeightView& v) {
    std::printf(
        "view%c%s%c%s%c%d%c%zu%c%s%c%d%c%zu%c%d%c%s%c%d%c%d\n", kSep, label,
        kSep, at(v.data).c_str(), kSep, static_cast<int>(v.dtype), kSep,
        v.nbytes, kSep, at(v.scale_data).c_str(), kSep,
        static_cast<int>(v.scale_dtype), kSep, v.scale_numel, kSep,
        static_cast<int>(v.quant_kind), kSep, at(v.zero_point_data).c_str(),
        kSep, v.group_size, kSep, v.channel_axis);
}

void script_factories() {
    dump("default", WeightView{});

    const DeviceTensor bf16 = tensor(0, DType::BF16, {128, 64});
    dump("implicit_bf16", WeightView(bf16));

    const DeviceTensor fp8 = tensor(4096, DType::FP8_E4M3, {128, 64});
    dump("implicit_fp8", WeightView(fp8));

    // `raw` is the escape hatch for buffers with no DeviceTensor handle. The
    // fields it does NOT set are the point of recording it.
    dump("raw_bf16", WeightView::raw(g_slab + 8192, DType::BF16));
    dump("raw_mxfp4", WeightView::raw(g_slab + 8192, DType::MXFP4_PACKED));
    dump("raw_null", WeightView::raw(nullptr, DType::FP32));

    const DeviceTensor scale_f32 = tensor(12288, DType::FP32, {128});
    const DeviceTensor zp = tensor(16384, DType::INT8, {128});

    QuantMeta per_tensor;
    per_tensor.kind = QuantMeta::Kind::PerTensor;
    per_tensor.scale = &scale_f32;
    dump("quantized_per_tensor", WeightView::quantized(fp8, per_tensor));

    QuantMeta per_channel;
    per_channel.kind = QuantMeta::Kind::PerChannel;
    per_channel.scale = &scale_f32;
    per_channel.channel_axis = 1;
    dump("quantized_per_channel", WeightView::quantized(fp8, per_channel));

    QuantMeta per_group;
    per_group.kind = QuantMeta::Kind::PerGroup;
    per_group.scale = &scale_f32;
    per_group.zero_point = &zp;
    per_group.group_size = 128;
    per_group.channel_axis = 0;
    dump("quantized_per_group", WeightView::quantized(fp8, per_group));

    // A QuantMeta with no scale at all: `quantized` must still produce a view
    // rather than dereference the null, and the result must be one that
    // `validate_quant_weight_view` rejects rather than one that looks bf16.
    QuantMeta no_scale;
    no_scale.kind = QuantMeta::Kind::PerChannel;
    dump("quantized_no_scale", WeightView::quantized(fp8, no_scale));

    const DeviceTensor mx_w = tensor(20480, DType::UINT8, {128, 32});
    const DeviceTensor mx_s = tensor(24576, DType::UINT8, {128, 2});
    dump("mxfp4_marlin", WeightView::mxfp4_marlin(mx_w, mx_s));
}

// ---------------------------------------------------------------------------
// Script 3 — `make_weight_view`, the quant dispatch the .inc bodies call.
// ---------------------------------------------------------------------------

// Reproduced here rather than included, because `model/llama_like/qwen3.hpp`
// drags in `loaded_model.hpp` and the whole gemm header. The body is three
// lines and is asserted below to be the same three lines.
WeightView make_weight_view(const DeviceTensor* w,
                            const std::optional<QuantMeta>& meta) {
    if (meta.has_value()) {
        return WeightView::quantized(*w, *meta);
    }
    return WeightView(*w);
}

void script_make_weight_view() {
    const DeviceTensor w = tensor(0, DType::BF16, {256, 64});
    const DeviceTensor scale = tensor(12288, DType::FP32, {256});

    dump("make_unquantized", make_weight_view(&w, std::nullopt));

    QuantMeta meta;
    meta.kind = QuantMeta::Kind::PerChannel;
    meta.scale = &scale;
    meta.channel_axis = 0;
    dump("make_quantized", make_weight_view(&w, meta));

    // An engaged optional whose scale is null still takes the quantized
    // branch: `has_value()` is the discriminator, not `scale`. A default
    // QuantMeta cannot show this -- its PerTensor kind and zero group_size
    // make the result identical to the unquantized view either way -- so the
    // observable case carries a non-default kind and group_size, which the
    // quantized branch copies through and the unquantized branch would not.
    QuantMeta empty;
    empty.kind = QuantMeta::Kind::PerGroup;
    empty.group_size = 64;
    empty.channel_axis = 1;
    dump("make_engaged_but_empty", make_weight_view(&w, empty));
}

}  // namespace

int main() {
    script_abi();
    script_factories();
    script_make_weight_view();
    return 0;
}
