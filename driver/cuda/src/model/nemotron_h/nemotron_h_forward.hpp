#pragma once

#include <cstdint>
#include <vector>

#include "ops/attention_workspace.hpp"
#include "device_buffer.hpp"
#include "store/kv_cache.hpp"
#include "model/llama_like/llama_like.hpp"
#include "model/nemotron_h/nemotron_h.hpp"
#include "model/workspace.hpp"
#include "ops/gemm.hpp"
#include "store/recurrent_state_cache.hpp"

namespace pie_cuda_driver::model {

struct NemotronHWorkspace {
    DeviceBuffer<std::uint16_t> mamba_projected;
    DeviceBuffer<std::uint16_t> mamba_gate;
    DeviceBuffer<std::uint16_t> mamba_conv_in;
    DeviceBuffer<std::uint16_t> mamba_conv_out;
    DeviceBuffer<std::uint16_t> mamba_core;
    DeviceBuffer<std::uint16_t> mamba_dt;
    DeviceBuffer<float> mamba_dt_f32;
    DeviceBuffer<float> mamba_dA_f32;

    DeviceBuffer<float> router_logits;
    DeviceBuffer<std::int32_t> topk_idx;
    DeviceBuffer<float> topk_weights;
    DeviceBuffer<std::uint16_t> expert_in;
    DeviceBuffer<std::uint16_t> expert_up;
    DeviceBuffer<std::uint16_t> expert_act;
    DeviceBuffer<std::uint16_t> expert_out;
    DeviceBuffer<std::int32_t> expert_idx;
    DeviceBuffer<float> expert_w;
    DeviceBuffer<std::uint16_t> shared_up;
    DeviceBuffer<std::uint16_t> shared_act;
    DeviceBuffer<std::uint16_t> shared_out;

    DeviceBuffer<const std::uint16_t*> a_up_ptrs;
    DeviceBuffer<const std::uint16_t*> b_up_ptrs;
    DeviceBuffer<std::uint16_t*> c_up_ptrs;
    DeviceBuffer<const std::uint16_t*> a_down_ptrs;
    DeviceBuffer<const std::uint16_t*> b_down_ptrs;
    DeviceBuffer<std::uint16_t*> c_down_ptrs;
    DeviceBuffer<float> route_weights;
    DeviceBuffer<std::uint8_t> flashinfer_moe_workspace;
    DeviceBuffer<std::int32_t> flashinfer_moe_map;
    std::size_t flashinfer_moe_workspace_bytes = 0;

    static NemotronHWorkspace allocate(
        const HfConfig& cfg, int max_tokens, int tp_size);
};

std::size_t nemotron_h_workspace_bytes(
    const HfConfig& cfg, int max_tokens, int tp_size);

int nemotron_h_attention_layers(const HfConfig& cfg);

// Number of mamba layers in the hybrid stack — the cheap "count == mamba"
// scan callers use for memory planning.
int nemotron_h_mamba_layers(const HfConfig& cfg);

// KV page bytes for a Nemotron-H stack — only the "attention" layer slice
// of the hybrid contributes to the paged KV cache. Mamba state is held
// separately by the recurrent state cache.
std::size_t kv_page_bytes_nemotron_h(const HfConfig& cfg,
                                     int tp_size,
                                     const ::pie_cuda_driver::KvCacheFormat& format);

// Per-slot bytes for the per-layer mamba state cache (conv + recurrent),
// accounting for TP sharding when nemotron_h_tp_mamba_sharding_enabled is
// true. Returns 0 if there are no mamba layers.
std::size_t nemotron_h_state_slot_bytes(const HfConfig& cfg,
                                        int mamba_layers,
                                        int tp_size);

void nemotron_h_forward_paged(
    const NemotronHWeights& w,
    const HfConfig& cfg,
    const LlamaLikeForwardCfg& fwd_cfg,
    const LlamaLikePlanState& plan_state,
    Workspace& ws,
    NemotronHWorkspace& nem_ws,
    KvCache& cache,
    RecurrentStateCache& state_cache,
    AttentionWorkspace& attn_ws,
    ops::CublasHandle& cublas,
    const std::int32_t* token_ids,
    const std::int32_t* positions,
    const std::uint32_t* qo_indptr,
    const std::uint32_t* kv_page_indices,
    const std::uint32_t* kv_page_indptr,
    const std::uint32_t* kv_last_page_lens,
    const std::uint32_t* qo_indptr_h,
    const std::uint32_t* kv_page_indptr_h,
    int total_tokens,
    int num_requests,
    bool is_pure_decode,
    const std::uint8_t* row_valid_d,
    const std::uint8_t* custom_mask_d,
    const std::int32_t* custom_mask_indptr_d,
    const std::int32_t* slot_ids_h,
    const std::uint8_t* is_fresh_h,
    const std::int32_t* slot_ids_d,
    const std::uint8_t* is_fresh_d,
    const std::int32_t* logit_row_indices_d,
    int num_logit_rows);

}  // namespace pie_cuda_driver::model
