#pragma once

#include <cstdint>
#include <filesystem>
#include <span>
#include <string>
#include <vector>

#include "attention_workspace.hpp"
#include "device_buffer.hpp"
#include "executor/executor.hpp"
#include "model/gemma4.hpp"
#include "model/weight_store.hpp"
#include "ops/attention_flashinfer.hpp"
#include "tensor.hpp"

namespace pie_cuda_driver::model {

struct Gemma4MtpRuntimeConfig {
    bool adaptive_drafts = false;
    int initial_drafts = 3;
    int min_drafts = 2;
    std::string compact_draft_rows = "auto";
    bool cuda_graph = true;
    int max_draft_batch_rows = 0;
};

struct Gemma4MtpLayerWeights {
    const DeviceTensor* input_norm = nullptr;
    const DeviceTensor* post_attn_norm = nullptr;
    const DeviceTensor* pre_mlp_norm = nullptr;
    const DeviceTensor* post_mlp_norm = nullptr;

    const DeviceTensor* q_proj = nullptr;
    const DeviceTensor* o_proj = nullptr;
    const DeviceTensor* q_norm = nullptr;

    const DeviceTensor* gate_proj = nullptr;
    const DeviceTensor* up_proj = nullptr;
    const DeviceTensor* down_proj = nullptr;
    const DeviceTensor* gate_up_proj_fused = nullptr;

    float layer_scalar_value = 1.f;
    int head_dim = 0;
    int target_kv_layer = 0;
    int window_left = -1;
    float rope_theta = 10000.f;
    float partial_rotary_factor = 1.0f;
};

struct Gemma4MtpWeights {
    HfConfig cfg;
    WeightStore store;

    const DeviceTensor* pre_projection = nullptr;
    const DeviceTensor* post_projection = nullptr;
    const DeviceTensor* final_norm = nullptr;
    const DeviceTensor* lm_head = nullptr;
    const DeviceTensor* masked_centroids = nullptr;
    const DeviceTensor* token_ordering = nullptr;

    int backbone_hidden_size = 0;
    int backbone_vocab_size = 0;
    bool use_ordered_embeddings = false;
    int num_centroids = 0;
    int centroid_top_k = 0;
    int vocab_size_per_centroid = 0;
    int max_hq = 0;
    int max_intermediate = 0;
    std::vector<Gemma4MtpLayerWeights> layers;
    std::vector<DeviceTensor> owned_gate_up_fused;
};

struct Gemma4MtpWorkspace {
    DeviceTensor hidden;       // [max_requests, backbone_hidden]
    DeviceTensor combined;     // [max_requests, 2 * backbone_hidden]
    DeviceTensor y;            // [max_requests, draft_hidden]
    DeviceTensor norm_x;       // [max_requests, draft_hidden]
    DeviceTensor norm_y;       // [max_requests, draft_hidden]
    DeviceTensor q;            // [max_requests, max_hq]
    DeviceTensor attn_out;     // [max_requests, max_hq]
    DeviceTensor gate;         // [max_requests, max_intermediate]
    DeviceTensor up;           // [max_requests, max_intermediate]
    DeviceTensor gate_up_fused; // [max_requests, 2 * max_intermediate]
    DeviceTensor logits;       // [max_requests, vocab]
    DeviceTensor centroid_logits; // [max_requests, num_centroids]
    DeviceTensor sparse_argmax_pairs; // [num_sparse_tiles, max_requests]
    DeviceTensor greedy_pairs_all; // [8, max_requests] packed argmax scratch

    DeviceBuffer<std::int32_t> row_indices;
    DeviceBuffer<std::int32_t> input_ids;
    DeviceBuffer<std::int32_t> positions;
    DeviceBuffer<std::int32_t> sampled;
    DeviceBuffer<std::int32_t> draft_tokens;
    DeviceBuffer<std::int32_t> top_centroids;
    DeviceBuffer<std::uint32_t> qo_indptr;
    DeviceBuffer<std::uint32_t> kv_page_indices;
    DeviceBuffer<std::uint32_t> kv_page_indptr;
    DeviceBuffer<std::uint32_t> kv_last_page_lens;

    std::vector<AttentionWorkspace> attn_workspaces;
    std::vector<ops::DecodePlanCachePtr> decode_plans;

    std::vector<std::int32_t> h_row_indices;
    std::vector<std::int32_t> h_input_ids;
    std::vector<std::int32_t> h_positions;
    PinnedHostBuffer<std::int32_t> h_sampled;
    std::vector<std::uint32_t> h_qo_indptr;
    std::vector<std::uint32_t> h_kv_page_indices;
    std::vector<std::uint32_t> h_kv_page_indptr;
    std::vector<std::uint32_t> h_kv_last_page_lens;

    int max_requests = 0;
    int max_page_refs = 0;
    int max_drafts = 0;

    static Gemma4MtpWorkspace allocate(
        const Gemma4MtpWeights& w,
        int max_requests,
        int max_page_refs,
        int max_drafts);
};

Gemma4MtpWeights load_gemma4_mtp_weights(
    const std::filesystem::path& snapshot_dir,
    const std::string& device,
    const HfConfig& target_cfg,
    const Gemma4Weights& target_weights,
    const Gemma4MtpRuntimeConfig& runtime,
    bool verbose = false);

void gemma4_mtp_draft(
    const Gemma4MtpWeights& w,
    const Gemma4Weights& target_weights,
    Gemma4MtpWorkspace& ws,
    const Gemma4MtpRuntimeConfig& runtime,
    const SystemSpecDraftInputs& in,
    std::span<pie_driver::PerRequestOutput> per_request);

// Single-step MTP forward matching the executor's MtpFn signature.
// Gathers hidden states from target_ws.norm_x at base_hidden_row_indices,
// runs one draft step, writes logits to target_ws.logits for executor argmax.
void gemma4_mtp_forward_step(
    const Gemma4MtpWeights& w,
    const Gemma4Weights& target_weights,
    Gemma4MtpWorkspace& mtp_ws,
    Qwen3Workspace& target_ws,
    KvCache& kv_cache,
    ops::CublasHandle& cublas,
    const std::int32_t* token_ids,
    const std::int32_t* position_ids,
    const std::int32_t* base_hidden_row_indices,
    const std::int32_t* request_ids,
    const std::uint32_t* kv_page_indices,
    const std::uint32_t* kv_page_indptr,
    const std::uint32_t* kv_last_page_lens,
    int num_tokens,
    int draft_step,
    int max_global_tokens);

}  // namespace pie_cuda_driver::model
