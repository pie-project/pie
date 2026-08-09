#pragma once
// Stub for csrc/src/store/kv_cache_format.hpp. Only `.name` is read by
// `make_planner_profile_key`; the real header drags in the tensor and
// attention-view headers, which have nothing to do with this file.
#include <string>
namespace pie_cuda_driver {
struct KvCacheFormat { std::string name = "bf16"; };
}
