#pragma once

// store/: long-lived device pools and their capacity planning.
// Given the loaded HF config, the
// active KV cache format, and the desired memory profile, search the
// (kv_page_size × decode_target × prefill_target) lattice and pick the
// shape that maximises a unified throughput/latency score subject to the
// device memory budget.
//
// Everything that the driver consumes from the planner — the chosen
// `kv_page_size`, capacity limits, KV/state byte budgets — is bundled in
// `CudaMemoryPlan`. The bounded `PlannedForwardLimits` substructure is
// what the executor passes downstream so per-fire allocators know the
// maximum shape they need to support.

#include <cstddef>
#include <vector>

#include <cuda_runtime.h>

namespace pie_cuda_driver {

struct Config;
struct HfConfig;
class KvCacheFormat;

namespace ops { struct RuntimeQuantScratchSpec; }

// Upper bounds on per-fire shapes. Sized by the planner so persistent
// device buffers can be reserved once and shared across all calls.
struct PlannedForwardLimits {
    int max_forward_tokens = 0;
    int max_forward_requests = 0;
    int max_page_refs = 0;
    int max_logit_rows = 0;
    int max_prob_rows = 0;
    int max_custom_mask_bytes = 0;
    int max_sampler_rows = 0;
    int max_logprob_labels = 0;
};

// One end-to-end memory plan for the CUDA driver. Fields are all that
// the engine needs to size persistent allocations + advertise capacity.
struct CudaMemoryPlan {
    int kv_page_size = 0;
    int max_workspace_tokens = 0;
    int max_requests = 0;
    int max_page_refs = 0;
    int kv_pages = 0;
    int state_slots = 0;
    std::size_t attn_float_workspace_bytes = 0;
    std::size_t runtime_quant_scratch_bytes = 0;
    std::size_t persistent_input_bytes = 0;
    std::size_t arena_bytes = 0;
    std::size_t kv_bytes = 0;
    std::size_t state_bytes = 0;
    PlannedForwardLimits capacity;
};

// Default KV page size for the active memory profile (cheap heuristic).
// Used by the parity harness which sizes its own KV cache; the main
// planner sweeps multiple page sizes through `plan_cuda_memory`.
int derive_kv_page_size(const Config& cfg,
                        const HfConfig& hf,
                        const cudaDeviceProp& prop);

// Search the candidate (kv_page_size × decode × prefill) lattice and
// return the highest-scoring plan that fits in device memory. Throws if
// no candidate fits or the model leaves no budget after weights.
CudaMemoryPlan plan_cuda_memory(
    const Config& cfg,
    const HfConfig& hf,
    int max_intermediate,
    int max_Hq,
    int max_Hk,
    bool gemma4_selected,
    const std::vector<int>& gemma4_per_layer_head_dim,
    const std::vector<int>& gemma4_kv_source_layer,
    bool qwen3_5_selected,
    bool qwen3_5_moe_selected,
    int qwen3_5_linear_layers,
    bool nemotron_h_selected,
    int nemotron_h_mamba_layers,
    bool deepseek_v4_selected,
    bool kimi_selected,
    bool glm5_selected,
    const KvCacheFormat& kv_format,
    const ops::RuntimeQuantScratchSpec& runtime_quant_scratch_base,
    bool verbose);

}  // namespace pie_cuda_driver
