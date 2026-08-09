// The Config subset plan_cuda_memory reads. Fields it never touches are
// omitted rather than defaulted: an omitted field cannot silently acquire a
// value that diverges from the Rust's PlannerConfig.
#pragma once
#include <cstdint>
#include <string>

namespace pie_cuda_driver {

struct BatchingConfig {
    double gpu_mem_utilization = 0.9;
    std::string memory_profile = "auto";
    std::uint32_t max_forward_tokens = 0;
    std::uint32_t max_forward_requests = 0;
    std::uint32_t kv_page_size = 0;
};

struct DistributedConfig {
    int tp_size = 1;
    std::string nccl_unique_id_hex;
};

struct ModelConfig {
    int mtp_num_drafts = 0;
};

struct Config {
    BatchingConfig batching;
    DistributedConfig distributed;
    ModelConfig model;
};

}  // namespace pie_cuda_driver
