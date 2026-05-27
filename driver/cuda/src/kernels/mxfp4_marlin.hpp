#pragma once

#include <cuda_runtime.h>

namespace pie_cuda_driver::kernels {

enum class Mxfp4RowSelect : int {
    Identity = 0,
    Even = 1,
    Odd = 2,
};

// Convert a GPT-OSS row-major MXFP4 weight matrix from a logical
// `[source_rows, source_stride_k / 2]` byte table into the GPTQ-style
// `[target_k / 8, selected_rows]` int32 staging layout consumed by Marlin's
// existing W4 repacker. Row and column offsets describe tensor-parallel
// slices without requiring a separate copy of the checkpoint tensor.
void launch_mxfp4_weight_to_gptq_w4(
    const void* raw_mxfp4,      // uint8 [source_rows, K / 2]
    void*       gptq_w4_out,    // uint32 [K / 8, selected_rows]
    int         source_rows,
    int         source_row_offset,
    int         selected_rows,
    int         valid_rows,
    int         source_stride_k,
    int         source_col_offset,
    int         source_k,
    int         target_k,
    Mxfp4RowSelect row_select,
    cudaStream_t stream);

// Convert raw GPT-OSS E8M0 block scales from a logical
// `[source_rows, source_stride_groups]` table to Marlin's
// `[target_groups, selected_rows]` byte layout, including Marlin's 64-wide
// scale permutation and the MXFP4 four-lane post-permutation used by
// vLLM/SGLang.
void launch_mxfp4_scales_to_marlin_e8m0(
    const void* raw_e8m0,       // uint8 [source_rows, source_groups]
    void*       marlin_e8m0,    // uint8 [target_groups, selected_rows]
    int         source_rows,
    int         source_row_offset,
    int         selected_rows,
    int         valid_rows,
    int         source_stride_groups,
    int         source_group_offset,
    int         source_groups,
    int         target_groups,
    Mxfp4RowSelect row_select,
    cudaStream_t stream);

// Gather a batched BF16 vector table by row map. This is used for fused
// GPT-OSS gate/up bias tensors; it lives with the MXFP4 repack kernels
// because it is the bias side of the same backend layout contract.
void launch_bf16_row_map_to_dense(
    const void* raw_bf16,       // bf16 [batch, source_rows]
    void*       out_bf16,       // bf16 [batch, selected_rows]
    int         batch,
    int         source_rows,
    int         source_row_offset,
    int         selected_rows,
    int         valid_rows,
    Mxfp4RowSelect row_select,
    cudaStream_t stream);

}  // namespace pie_cuda_driver::kernels
