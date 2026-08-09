//===-- altup_aux_entry.cu - the entry points, and the manifest -------===//
//
// One `extern "C" __global__` per row in `kernels_cuda::norm_device::ENTRIES`.
// This file is the whole of what `kernels.def` does for the attention head
// dims -- it names the instantiations -- with the difference that the set is
// not a build-time secret: the rows state it, and a row is what
// `model-compiler` and `driver-cuda` already read.
//
// Nothing here computes a grid, takes a stream, guards an extent or casts a
// dimension. An entry is a NAME for a template argument list, and it is the
// only thing in Tier A that a human writes twice.
//
// # Why this is hand-written and the typecheck is generated
//
// It is the shim's argument, in the other direction. `abi::emit_c_shim`
// generates a body that CALLS the launcher, so C++ decides whether the row
// is right. Here the call is inside the entry -- `device::compute_rms` is a
// template and the compiler resolves it against the header -- so the entry
// proves the BODY, and `abi::emit_device_typecheck` proves the entry against
// the row with a function pointer that admits no conversions. Generate both
// halves and the check is a tautology; hand-write both and nothing checks
// anything. One each is what makes the pair a proof.
//
//===----------------------------------------------------------------------===//

#include "norm/altup_aux_device.cuh"

namespace device = ::pie_cuda_driver::kernels::norm::device;

extern "C" {

__global__ void pie_g_norm_compute_rms_bf16(const void* reference, float* target_rms_out, int h,
                                            float eps) {
    device::compute_rms<__nv_bfloat16>(static_cast<const __nv_bfloat16*>(reference), target_rms_out,
                                       h, eps);
}

__global__ void pie_g_norm_magnitude_rescale_bf16(void* x, const float* target_rms, int h,
                                                  float eps) {
    device::magnitude_rescale<__nv_bfloat16>(static_cast<__nv_bfloat16*>(x), target_rms, h, eps);
}

__global__ void pie_g_norm_mean_streams_bf16(const void* streams, void* out, int k, int t, int h) {
    device::mean_streams<__nv_bfloat16>(static_cast<const __nv_bfloat16*>(streams),
                                        static_cast<__nv_bfloat16*>(out), k, t, h);
}

__global__ void pie_g_norm_altup_unpack_predict_coefs(const void* in_bf16, float* out_fp32, int k) {
    device::unpack_predict_coefs<__nv_bfloat16>(static_cast<const __nv_bfloat16*>(in_bf16),
                                                out_fp32, k);
}

__global__ void pie_g_norm_altup_unpack_correct_coefs(const void* in_bf16, float* out_fp32, int k) {
    device::unpack_correct_coefs<__nv_bfloat16>(static_cast<const __nv_bfloat16*>(in_bf16),
                                                out_fp32, k);
}

__global__ void pie_g_norm_tanh_bf16(void* x, int numel) {
    device::tanh_inplace<__nv_bfloat16>(static_cast<__nv_bfloat16*>(x), numel);
}

// The fp16 half of the same six, which the bf16 tree never had and which
// costs one line each because the bodies are templates. It is here to make
// the instantiation claim checkable rather than asserted: `kernels.def`
// would have needed a new macro, a new `#include` list and a new CMake
// regex to say this much.
__global__ void pie_g_norm_tanh_f16(void* x, int numel) {
    device::tanh_inplace<__half>(static_cast<__half*>(x), numel);
}

}  // extern "C"
