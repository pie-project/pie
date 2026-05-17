#pragma once

namespace pie_cuda_driver {

enum class Mxfp4MoeLowering {
    Bf16Dequant,
    RoutedDequant,
    NativeGemm,
};

struct BackendTarget {
    int device_major = 0;
    int device_minor = 0;
    bool fp8_native = false;
    bool gptq_marlin_int4 = true;
    bool mxfp4_native_gemm = false;

    // RoutedDequant keeps QuantPacked MXFP4 expert weights resident and lets
    // the MoE runtime dequantize only routed experts into bounded BF16 scratch.
    // NativeGemm is reserved for a backend that consumes MXFP4 directly inside
    // expert GEMM kernels; selecting it requires `mxfp4_native_gemm`.
    Mxfp4MoeLowering mxfp4_moe = Mxfp4MoeLowering::Bf16Dequant;
};

}  // namespace pie_cuda_driver

