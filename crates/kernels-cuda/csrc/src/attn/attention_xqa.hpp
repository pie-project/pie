#pragma once

#include <cstdint>

#include <cuda_runtime.h>

#include "attention_workspace_view.hpp"

namespace pie_cuda_driver::kernels::attn {

// FlashInfer XQA decode specializations currently compiled into Pie.
// This is a torch-free wrapper over FlashInfer's csrc/xqa decode kernel.
// It is intentionally narrow: unsupported shapes fall back to the existing
// FlashInfer decode/prefill paths.
bool xqa_decode_bf16_supported(int num_q_heads,
                               int num_kv_heads,
                               int head_dim,
                               int page_size,
                               int window_left,
                               float logits_soft_cap,
                               float sm_scale);

int xqa_decode_page_bucket(int max_pages_per_seq);
// Set the per-device max-dynamic-smem attribute for the selected XQA kernel on
// the *current* CUDA device. FlashInfer's xqa csrc does this via a
// once-per-process static initializer, which only covers whichever device is
// current when that static runs; under TP>1, other ranks' devices may never get
// the attribute set. Call this after `cudaSetDevice` on each rank, before any
// graph capture.
// `xqa_decode_bf16_warmup_current_device`,
// `xqa_decode_bf16_gqa5_warmup_current_device` and the unprepared
// `attention_xqa_decode_bf16` WERE declared here. All three launchers are
// deleted: no shim entry is generated for any of them, so the driver cannot
// call them, and nothing in `csrc` does either. The `.cu` carries each
// measurement.

void prepare_attention_xqa_decode_bf16(
    const std::uint32_t* kv_page_indices_d,
    const std::uint32_t* kv_page_indptr_d,
    const std::uint32_t* kv_last_page_lens_d,
    int num_requests,
    int page_size,
    int max_pages_per_seq,
    AttentionWorkspaceView workspace,
    cudaStream_t stream);

void attention_xqa_decode_bf16_prepared(
    const void* q,
    void* k_pages,
    void* v_pages,
    void* o,
    int num_requests,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    int page_size,
    int max_pages_per_seq,
    AttentionWorkspaceView workspace,
    cudaStream_t stream,
    float sm_scale = -1.f);

}  // namespace pie_cuda_driver::kernels::attn
