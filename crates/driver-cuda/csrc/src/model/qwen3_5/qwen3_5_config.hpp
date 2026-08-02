#pragma once

// Fixed model constants shared between Qwen3.5 (non-MoE) and Qwen3.5-MoE.

namespace pie_cuda_driver::model {

int  qwen35_small_spec_graph_tokens();
int  qwen35_mtp_draft_position_offset();
bool qwen35_mtp_prefix_global_cache();

}  // namespace pie_cuda_driver::model
