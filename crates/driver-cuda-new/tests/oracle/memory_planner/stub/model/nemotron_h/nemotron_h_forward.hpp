#pragma once
#include <cstddef>
namespace pie_cuda_driver {
struct HfConfig;
class KvCacheFormat;
namespace model {
std::size_t nemotron_h_workspace_bytes(const HfConfig& cfg, int N, int tp);
std::size_t nemotron_h_state_slot_bytes(const HfConfig& cfg, int mamba_layers, int tp);
std::size_t kv_page_bytes_nemotron_h(const HfConfig& cfg, int tp,
                                     const KvCacheFormat& format);
}
}  // namespace pie_cuda_driver
