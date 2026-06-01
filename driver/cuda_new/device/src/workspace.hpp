// PieWorkspace — pre-allocated per-fire activation scratch, reused across
// layers (and, later, fires). This is the architecture's pointer-stable
// workspace that replaces the per-call cudaMalloc in the layer slice; its
// fixed addresses are what make CUDA-graph replay safe. Opaque to the ABI;
// concrete here so abi.cpp (alloc/free) and forward/ (reads) can use it.
#pragma once

#include "ops/attention_paged.hpp"      // plan caches (opaque) + workspace
#include "ops/attention_workspace.hpp"

namespace pie_cuda_device {}

// Sized from PieWorkspaceDims at pie_ws_alloc time. All buffers are bf16,
// width-by-max_tokens. `hidden_buf` is the residual stream threaded through
// the decoder layers; the rest are per-layer scratch.
struct PieWorkspace {
    int max_tokens = 0;
    int hidden = 0;
    int n_q_heads = 0;
    int n_kv_heads = 0;
    int head_dim = 0;
    int intermediate = 0;
    int vocab = 0;

    void* hidden_buf = nullptr;  // [max_tokens, hidden]
    void* normed = nullptr;      // [max_tokens, hidden]
    void* q = nullptr;           // [max_tokens, n_q_heads*head_dim]
    void* k = nullptr;           // [max_tokens, n_kv_heads*head_dim]
    void* v = nullptr;           // [max_tokens, n_kv_heads*head_dim]
    void* attn = nullptr;        // [max_tokens, n_q_heads*head_dim]
    void* o = nullptr;           // [max_tokens, hidden]
    void* gate = nullptr;        // [max_tokens, intermediate]
    void* up = nullptr;          // [max_tokens, intermediate]
    void* mlp = nullptr;         // [max_tokens, intermediate]
    void* mlp_out = nullptr;     // [max_tokens, hidden]

    // FlashInfer-backed attention: the persistent plan scratch (device float/
    // int + pinned host int) and the per-fire plan caches, reused across all
    // layers. Allocated lazily on first forward that needs the fast path (the
    // ws geometry alone doesn't say which attention kernel an arch uses). The
    // AttentionWorkspace destructor + unique_ptr free on `delete ws`.
    // FlashInfer plan scratch + per-fire plan caches (device-side scratch,
    // legitimately in the workspace). Persistent INPUT buffers + the graph
    // cache live in the RUST control plane (executor.rs) — C++ stays thin.
    bool attn_ready = false;
    pie_cuda_device::ops::AttentionWorkspace attn_ws;
    pie_cuda_device::ops::PrefillPlanCachePtr prefill_plan;
    pie_cuda_device::ops::DecodePlanCachePtr decode_plan;
};
