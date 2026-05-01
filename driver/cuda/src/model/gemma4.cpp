#include "model/gemma4.hpp"

#include <stdexcept>

namespace pie_cuda_driver::model {

Qwen3Weights bind_gemma4(Engine&) {
    throw std::runtime_error(
        "bind_gemma4: not yet implemented; see model/gemma4.hpp for the "
        "list of architectural deltas (PLE, KV-cache sharing, dual head_dim, "
        "proportional RoPE) and the flashinfer-side prerequisite "
        "(trtllm_batch_context_with_kv_cache for head_dim=512 prefill).");
}

}  // namespace pie_cuda_driver::model
