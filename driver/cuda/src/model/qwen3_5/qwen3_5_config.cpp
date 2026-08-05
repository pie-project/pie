#include "model/qwen3_5/qwen3_5_config.hpp"

#include <algorithm>
#include <cstdlib>

namespace pie_cuda_driver::model {

int qwen35_small_spec_graph_tokens() { return 17; }

int qwen35_mtp_draft_position_offset() { return 0; }

bool qwen35_mtp_prefix_global_cache() { return true; }

}  // namespace pie_cuda_driver::model
