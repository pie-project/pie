#pragma once

#include <cstdint>
#include <filesystem>
#include <stdexcept>
#include <string>

#include <toml++/toml.hpp>

namespace pie_cuda_driver {

struct ShmemConfig {
    std::string name = "/pie-fwd-0";
    std::size_t num_slots = 8;
    std::size_t req_buf = 4 * 1024 * 1024;
    std::size_t resp_buf = 1 * 1024 * 1024;
    std::uint64_t spin_us = 0;
};

struct ModelConfig {
    std::string hf_repo;
    std::string snapshot_dir;     // local path to weights + config.json
    std::string device = "cuda:0";
    std::string dtype = "bfloat16";
};

struct BatchingConfig {
    std::uint32_t kv_page_size = 32;
    std::uint32_t max_num_kv_pages = 1024;
    std::uint32_t max_batch_tokens = 10240;
    std::uint32_t max_batch_size = 512;
};

struct Config {
    ShmemConfig shmem;
    ModelConfig model;
    BatchingConfig batching;
};

inline Config load_config(const std::filesystem::path& path) {
    if (!std::filesystem::exists(path)) {
        throw std::runtime_error("config not found: " + path.string());
    }

    auto tbl = toml::parse_file(path.string());
    Config c;

    if (auto s = tbl["shmem"].as_table()) {
        c.shmem.name      = (*s)["name"].value_or(c.shmem.name);
        c.shmem.num_slots = (*s)["num_slots"].value_or<int64_t>(c.shmem.num_slots);
        c.shmem.req_buf   = (*s)["req_buf"].value_or<int64_t>(c.shmem.req_buf);
        c.shmem.resp_buf  = (*s)["resp_buf"].value_or<int64_t>(c.shmem.resp_buf);
        c.shmem.spin_us   = (*s)["spin_us"].value_or<int64_t>(c.shmem.spin_us);
    }
    if (auto m = tbl["model"].as_table()) {
        c.model.hf_repo      = (*m)["hf_repo"].value_or(std::string{});
        c.model.snapshot_dir = (*m)["snapshot_dir"].value_or(std::string{});
        c.model.device       = (*m)["device"].value_or(c.model.device);
        c.model.dtype        = (*m)["dtype"].value_or(c.model.dtype);
    }
    if (auto b = tbl["batching"].as_table()) {
        c.batching.kv_page_size     = (*b)["kv_page_size"].value_or<int64_t>(c.batching.kv_page_size);
        c.batching.max_num_kv_pages = (*b)["max_num_kv_pages"].value_or<int64_t>(c.batching.max_num_kv_pages);
        c.batching.max_batch_tokens = (*b)["max_batch_tokens"].value_or<int64_t>(c.batching.max_batch_tokens);
        c.batching.max_batch_size   = (*b)["max_batch_size"].value_or<int64_t>(c.batching.max_batch_size);
    }

    if (c.model.hf_repo.empty()) {
        throw std::runtime_error("config: [model].hf_repo is required");
    }
    return c;
}

}  // namespace pie_cuda_driver
