#pragma once

// `WeightView` — how every kernel in this crate is handed a weight.
//
// It was declared in `gemm/gemm.hpp` because that was the first header to need
// it, and it never belonged to gemm: the driver names it `WeightView` in
// 31 hand-written lines and 1068 generated ones, and half the kernel families
// take one. Moving gemm's launchers into `kernels::gemm` therefore could not
// take this with them, and for three commits it stayed reachable through a
// `using` declaration left behind in `ops`. This header replaces that shim.
//
// The namespace is `pie_cuda_driver`, the same one `tensor.hpp`,
// `cache_root.hpp` and `attention_workspace_view.hpp` sit in beside it. An earlier
// revision of this comment argued for keeping `ops` -- "1099 edits for a nicer
// word and buys nothing else" -- and that was wrong on the second half. `ops`
// was a directory name from when this tree meant "vendored-wrapper kernels".
// The directory is gone; a namespace whose only remaining job is to be a
// leftover reads like a distinction, and the reader has to go find out that it
// is not one. `DeviceTensor` is named unqualified in 358 model lines; so is
// this now.
//
// A POD plus two factories, no out-of-line members — which is exactly why this
// one lifts out and `CublasHandle` and `RuntimeQuantContext` do not. Those are
// genuinely gemm's: one wraps its cuBLAS handle, the other is bound to the
// cuBLASLt scratch context and a thread-local in `gemm.cpp`.

#include <cstddef>

#include "quant_meta.hpp"
#include "tensor.hpp"

namespace pie_cuda_driver {

// Lightweight reference to a weight tensor + (optional) quantization
// metadata, threaded through the GEMM dispatcher. The bf16 path takes the
// implicit `WeightView(const DeviceTensor&)` constructor and pays nothing;
// the quantized path uses `WeightView::quantized(...)`.
//
// We deliberately don't store a `const DeviceTensor*` — bind functions may
// hand us raw pointers (e.g. into a fused expert table) that don't have a
// DeviceTensor wrapper. Carrying just `(data, dtype)` covers both cases.
struct WeightView {
    const void* data = nullptr;
    DType       dtype = DType::BF16;
    std::size_t nbytes = 0;

    // Quant metadata. `scale_data == nullptr` means "no quant — bf16 path".
    // For per-channel / per-group quant the dispatcher reads the layout
    // hints from `kind`, `group_size`, `channel_axis`.
    const void*       scale_data = nullptr;
    DType             scale_dtype = DType::FP32;
    std::size_t       scale_numel = 0;
    QuantMeta::Kind   quant_kind = QuantMeta::Kind::PerTensor;
    const void*       zero_point_data = nullptr;
    int               group_size = 0;
    int               channel_axis = 0;

    WeightView() = default;

    // Implicit conversion from a plain DeviceTensor — preserves call-site
    // terseness for the unquantized path (the 99% case in M0).
    WeightView(const DeviceTensor& t)
        : data(t.data()), dtype(t.dtype()), nbytes(t.nbytes()) {}

    // Raw pointer + dtype, for buffers without a DeviceTensor handle
    // (deinterleaved MoE scratch, expert pointer arrays).
    static WeightView raw(const void* p, DType d) {
        WeightView v; v.data = p; v.dtype = d; return v;
    }

    // Quantized weight: ties together a weight DeviceTensor and a
    // `QuantMeta` snapshot pulled from `LoadedModel::quant_meta`.
    static WeightView quantized(const DeviceTensor& weight, const QuantMeta& meta) {
        WeightView v;
        v.data = weight.data();
        v.dtype = weight.dtype();
        v.nbytes = weight.nbytes();
        v.scale_data = meta.scale ? meta.scale->data() : nullptr;
        v.scale_dtype = meta.scale ? meta.scale->dtype() : DType::FP32;
        v.scale_numel = meta.scale ? meta.scale->numel() : 0;
        v.quant_kind = meta.kind;
        v.zero_point_data = meta.zero_point ? meta.zero_point->data() : nullptr;
        v.group_size = meta.group_size;
        v.channel_axis = meta.channel_axis;
        return v;
    }

    // `channel_axis` is defaulted for the Marlin repack path (mixtral),
    // whose scales are already in the axis-0 layout the repack produced.
    // glm5's routed experts carry the checkpoint's own axis in their
    // QuantMeta and pass it through -- before the parameter existed, that
    // one caller hand-built this view field by field.
    static WeightView mxfp4_marlin(
        const DeviceTensor& weight,
        const DeviceTensor& scale,
        int channel_axis = 0)
    {
        WeightView v;
        v.data = weight.data();
        v.dtype = DType::MXFP4_PACKED;
        v.nbytes = weight.nbytes();
        v.scale_data = scale.data();
        v.scale_dtype = DType::UINT8;
        v.scale_numel = scale.numel();
        v.quant_kind = QuantMeta::Kind::PerGroup;
        v.group_size = 32;
        v.channel_axis = channel_axis;
        return v;
    }
};

}  // namespace pie_cuda_driver
