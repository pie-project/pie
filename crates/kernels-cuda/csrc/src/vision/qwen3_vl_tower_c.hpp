#pragma once

// The qwen3_vl vision TOWER as one launcher — the launch-bridge philosophy
// at tower granularity (see `.wiki/driver-cuda-retirement.md`, the VL
// judgment). The C++ walk (`run_qwen3vl_vision` + the host prep inside
// `scatter_qwen3vl_vision`: bilinear pos-embed interpolation, 2-D rope
// position ids, the spatial-merge reorder, the f32→bf16 pixel cast) stays
// C++ and parity-anchored; this entry only flattens the `QwenVisRawWeights`
// struct — whose `std::vector` members no C ABI can carry — into pointer
// TABLES, exactly how the grouped-GEMM rows carry their per-expert banks.
//
// Table layouts (HOST arrays of DEVICE pointers, like every `Ty::Bufs`):
//   block_w   [depth * 12]: per block, in order
//             [norm1.g, norm1.b, norm2.g, norm2.b, qkv.w, qkv.b,
//              o.w, o.b, fc1.w, fc1.b, fc2.w, fc2.b]
//   merger_w  [6]:  [norm.g, norm.b, fc1.w, fc1.b, fc2.w, fc2.b]
//   deepstack_w [num_deep * 6]: the same six per deepstack merger
//
// `pixels_h`, `pixel_byte_indptr_h`, `grids_h`, `anchor_rows_h` and
// `deepstack_layers` are HOST pointers — the step hands them over host-side
// and the scatter uploads what it needs, the C++ shape.

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace pie_cuda_driver::kernels::vision {

void qwen3vl_scatter(
    const void* patch_w, const void* patch_b, const void* pos_embed,
    const void* const* block_w, int depth,
    const void* const* merger_w,
    const void* const* deepstack_w, const int* deepstack_layers,
    int hidden, int heads, int intermediate, int patch_size,
    int temporal_patch, int merge_size, int in_channels, int out_hidden,
    int num_pos_embed, float ln_eps, float rope_theta,
    const float* pixels_h, const std::uint32_t* pixel_byte_indptr_h,
    const std::uint32_t* grids_h, const std::uint32_t* anchor_rows_h,
    int num_images,
    void* hidden_rows, int n_rows,
    void* deepstack_scratch, int num_deep,
    cublasHandle_t blas, cudaStream_t stream);

}  // namespace pie_cuda_driver::kernels::vision
