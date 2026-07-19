#pragma once

// Env-var-derived runtime knobs shared between Qwen3.5 (non-MoE) and
// Qwen3.5-MoE. Values are cached on first call.

namespace pie_cuda_driver::model {

int  qwen35_small_spec_graph_tokens();
bool qwen35_forward_profile_enabled();
int  qwen35_mtp_draft_position_offset();
bool qwen35_mtp_fused_gemv_enabled();
bool qwen35_mtp_prefix_global_cache();

}  // namespace pie_cuda_driver::model
