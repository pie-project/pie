#pragma once
#include <cstddef>
namespace pie_cuda_driver::model {
struct HfConfigFwd;
}
namespace pie_cuda_driver {
struct HfConfig;
namespace model {
std::size_t workspace_bytes(const HfConfig& cfg, int max_tokens,
                            int max_output_rows, int max_intermediate,
                            int max_Hq, int max_Hk, int mtp_draft_rows);
}
}  // namespace pie_cuda_driver
