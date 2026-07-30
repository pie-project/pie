#pragma once

// store/: the on-disk cache of MEASURED planner shapes.
//
// `plan_cuda_memory` scores its candidate lattice analytically. The score is a
// model of how a shape will perform, and a model is a guess: where the guess
// disagreed with reality the tree accumulated per-(model, GPU) overrides with
// hand-measured constants baked in. This cache is the mechanism that lets the
// driver replace those constants with its own measurement — `calibrate_memory_planner`
// (batch/planner_calibration.hpp) times the real forward step across the shape
// ladder and stores the winner here; the planner reads it back and selects by
// evidence instead of by score.
//
// Reader and writer share `PlannerProfileKey` on purpose. The key has twelve
// fields, and if the two sides ever built it independently a single
// disagreement would make every lookup miss silently — the cache would look
// empty rather than broken.

#include <cstddef>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

#include <cuda_runtime.h>

namespace pie_cuda_driver {

struct HfConfig;
class KvCacheFormat;

// The identity under which a measured shape is valid. A shape measured on one
// (device, model, parallelism, kv format) says nothing about another.
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

// The cached selection. Zero/empty fields are wildcards on lookup, so a cache
// written by a calibrator that only swept `max_forward_tokens` does not pin the
// fields it never measured.
struct PlannerProfileShape {
    std::string policy_profile;
    int kv_page_size = 0;
    int max_forward_tokens = 0;
    int max_forward_requests = 0;
};

// One measured point on the shape ladder, retained beside the selection so a
// cache entry can be audited without re-running the calibration.
struct PlannerShapeSample {
    int max_forward_tokens = 0;
    int max_forward_requests = 0;
    int tokens_per_request = 0;
    double step_ms = 0.0;
    double step_ms_stddev = 0.0;
    double tokens_per_s = 0.0;
};

// `$HOME/.cache/pie/cuda_memory_profiles.json`, or empty when $HOME is unset.
std::filesystem::path planner_profile_cache_path();

PlannerProfileKey make_planner_profile_key(const cudaDeviceProp& prop,
                                           const HfConfig& hf,
                                           int tp_size,
                                           const KvCacheFormat& kv_format);

// The measured shape for `key`, or nullopt when the cache is absent,
// unreadable, or holds no entry for this key. Never throws: a corrupt cache
// degrades to "no measurement", and `error` (when non-null) receives why.
std::optional<PlannerProfileShape> planner_profile_cache_lookup(
    const PlannerProfileKey& key,
    std::string* error);

// Replace (or append) the entry for `key`, preserving every other entry.
// Serialised and atomic: an exclusive `flock` on a sibling `.lock` file covers
// the whole read -> merge -> rename, so two processes calibrating at once
// cannot drop each other's entries, and the rename itself means a concurrent
// reader observes either the old cache or the new one, never a partial write.
bool planner_profile_cache_store(const PlannerProfileKey& key,
                                 const PlannerProfileShape& shape,
                                 const std::vector<PlannerShapeSample>& samples,
                                 std::string* error);

}  // namespace pie_cuda_driver
