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
// `xqa_decode_bf16_warmup_current_device`,
// `xqa_decode_bf16_gqa5_warmup_current_device` and the unprepared
// `attention_xqa_decode_bf16` WERE declared here. All three launchers are
// deleted: no shim entry is generated for any of them, so the driver cannot
// call them, and nothing in `csrc` does either. The `.cu` carries each
// measurement.
//
// The paragraph that stood here explained the warmups -- that FlashInfer's xqa
// csrc sets the per-device max-dynamic-smem attribute from a once-per-process
// static initializer, which only covers whichever device is current when that
// static runs, so under TP>1 other ranks' devices may never get it, and that
// the fix is to call the warmup after `cudaSetDevice` on each rank before any
// graph capture. That is a real hazard and it is now UNADDRESSED IN THIS TREE:
// the two warmups with bodies went with §44.4, the five per-ratio ones in the
// `.cu`'s `detail` block were declarations over nothing, and no Rust does it
// either. Kept as prose so the port that finishes XQA knows there is a
// `cudaFuncSetAttribute` per device owed. `new-horizon.md` §50.9.
//
// `prepare_attention_xqa_decode_bf16` WAS declared here too, and it is the
// last thing to leave this archive that had a `__global__` behind it. Its
// kernel is `kernels-cuda-new/csrc/src/attn/attention_xqa.cuh`, JIT-compiled
// as the `attn/attention_xqa` unit; its host half is
// `driver-cuda/src/fire/xqa.rs::prepare_decode`. The row below still states
// `needs = Prepare::FireWide`, and the Rust is now the only thing that can
// discharge it -- as the C++ was, since it had no shim entry either.

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
