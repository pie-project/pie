//===-- vision/gemma4_naive_kernels.cuh - the gemma-4 towers' shared one -===//
//
// The one `__global__` the two gemma-4 tower translation units share, as a
// template over the storage format.
//
// # Why this exists
//
// The cross-tower half (`k_matmul`, `k_rms`, `k_add`, `k_f32_to_bf16`,
// `k_gelu_erf`, `k_layernorm`) is in `vision/tower_naive_kernels.cuh`; this is
// the one only gemma-4's vision and audio translation units share. Same way
// station, same limit -- see that header for both, and for why these are named
// templates rather than the anonymous-namespace `__global__`s they were.
//
// Splitting it out is what keeps `gemma4_vision.cu` and `gemma4_audio.cu` from
// each carrying a copy. They did, once, in the old driver; the copies agreed,
// and `tests/sources.rs::no_global_is_defined_twice` exists because
// `norm/altup_aux` proved that two copies which agree today ship a release
// where one of them has drifted and every test still passes.
//
//===---------------------------------------------------------------------===//
#pragma once

#include "vision/tower_naive_kernels.cuh"

namespace pie_cuda_driver::kernels::vision::device {

/// Clamp into `[*lo, *hi]`, either bound OPTIONAL.
///
/// `Rule::Elementwise`: `k_clamp<<<(t+255)/256, 256, 0, S>>>` at all four call
/// sites -- gemma-4 vision's `clin` fires it twice per clipped linear, gemma-4
/// audio's does the same -- and `elementwise` evaluates `rows * width` to the
/// same `ceil(t/256)` blocks of 256.
///
/// The bounds are DEVICE pointers, not floats, because they are per-layer
/// weights that live in the checkpoint next to the matrix they clip; reading
/// them on the host would be a synchronising copy per linear per layer.
/// `nullptr` means unbounded on that side, which is why the infinities appear
/// at all.
///
/// `CUDART_INF_F` became `device::pos_inf()` -- the same `0x7f800000` bit
/// pattern, stated as bits. `math_constants.h` was one of the 31 headers NVRTC
/// 13.0 was probed for and answered 0 of, so the constant had nowhere to come
/// from and the kernel would not have compiled at all.
template <class T>
__global__ void k_clamp(const T* x, T* o, const T* lo, const T* hi, usize t) {
    usize i = blockIdx.x * (usize)blockDim.x + threadIdx.x;
    if (i >= t) return;
    float v = F(x[i]);
    float l = lo ? F(*lo) : ::pie_cuda_driver::kernels::device::neg_inf();
    float h = hi ? F(*hi) : ::pie_cuda_driver::kernels::device::pos_inf();
    o[i] = Bf<T>(v < l ? l : (v > h ? h : v));
}

}  // namespace pie_cuda_driver::kernels::vision::device
