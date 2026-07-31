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

    // RoutedDequant keeps packed MXFP4 expert weights resident and lets the
    // MoE runtime dequantize only routed experts into bounded BF16 scratch.
    // NativeGemm consumes backend-repacked FE2M1/E8M0 weights directly inside
    // expert GEMM kernels; auto selects it only when `mxfp4_native_gemm` is
    // true for the current build and GPU.
    Mxfp4MoeLowering mxfp4_moe = Mxfp4MoeLowering::Bf16Dequant;

    // SSD expert streaming: routed MoE expert weights are omitted from the
    // resident schedule (no VRAM at load) and described by the program's
    // deferred stream plan, which the expert stream cache executes into a
    // bounded GPU slab on demand at forward time.
    bool stream_routed_experts = false;
};

}  // namespace pie_cuda_driver
