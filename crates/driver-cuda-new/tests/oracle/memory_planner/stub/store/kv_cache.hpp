#pragma once
#include <cstddef>
#include <vector>
#include "kv_cache_format.hpp"

namespace pie_cuda_driver {

struct HfConfig;

// Only the one static the planner calls. `KvCache::envelopes_requested()` and
// the planner MUST agree or the cache overruns its budget, which is why it is
// a switch the driver reads rather than a constant.
class KvCache {
public:
    static bool envelopes_requested();
};

std::size_t kv_cache_device_bytes_per_page(const KvCacheFormat& format,
                                           int page_size, int kv_heads,
                                           int head_dim);
std::size_t kv_page_bytes_homogeneous(const HfConfig& cfg, int tp_size,
                                      const KvCacheFormat& format);
std::size_t kv_page_bytes_per_layer(const HfConfig& cfg,
                                    const std::vector<int>& per_layer_head_dim,
                                    const std::vector<int>& per_layer_num_kv_heads,
                                    const std::vector<int>& kv_source_layer,
                                    int tp_size,
                                    const KvCacheFormat& format);

}  // namespace pie_cuda_driver
