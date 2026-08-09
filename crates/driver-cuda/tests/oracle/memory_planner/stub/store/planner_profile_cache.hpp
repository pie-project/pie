#pragma once
#include <filesystem>
#include <optional>
#include <string>
#include <cuda_runtime.h>

namespace pie_cuda_driver {

// Relative budget change past which a measured profile stops describing this
// machine. Lives in the real header, so the stub must carry it too.
constexpr double kPlannerBudgetTolerance = 0.05;

struct HfConfig;
class KvCacheFormat;

// The planner's read side of the cache. Stubbed so the transcript can drive
// the SELECTION path -- pinned, drifted, unmatched -- from the driver rather
// than from a file on disk, which is what the profile_cache oracle already
// proves byte for byte.
struct PlannerProfileKey {
    std::string gpu_name;
    int compute_major = 0;
    int compute_minor = 0;
    int sm_count = 0;
    std::string kv_cache_dtype;
    int tp_size = 1;
    std::string model_type;
    int hidden_size = 0;
    int num_hidden_layers = 0;
    int num_attention_heads = 0;
    int num_key_value_heads = 0;
    int head_dim = 0;
};

struct PlannerProfileShape {
    std::string policy_profile;
    int kv_page_size = 0;
    int max_forward_tokens = 0;
    int max_forward_requests = 0;
    std::size_t budget_bytes = 0;
};

PlannerProfileKey make_planner_profile_key(const cudaDeviceProp& prop,
                                           const HfConfig& hf, int tp_size,
                                           const KvCacheFormat& format);
std::optional<PlannerProfileShape> planner_profile_cache_lookup(
    const PlannerProfileKey& key, std::string* error);
std::filesystem::path planner_profile_cache_path();
void set_planner_budget_bytes(std::size_t budget);
std::size_t planner_budget_bytes();
}  // namespace pie_cuda_driver
