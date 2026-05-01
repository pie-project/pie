#include "engine.hpp"

#include <algorithm>
#include <chrono>
#include <iostream>
#include <stdexcept>

#include <cuda_runtime.h>

#include "cuda_check.hpp"

namespace pie_cuda_driver {

Engine Engine::load(const Config& boot_cfg) {
    if (boot_cfg.model.snapshot_dir.empty()) {
        throw std::runtime_error(
            "engine: model.snapshot_dir is empty — pass it in dev.toml or "
            "let the wrapper resolve it via pie_driver.hf_utils");
    }

    Engine e;
    e.boot_ = boot_cfg;

    const std::filesystem::path snapshot{boot_cfg.model.snapshot_dir};
    e.hf_ = parse_hf_config(snapshot / "config.json");

    // Bind to the requested CUDA device before we allocate anything.
    int dev_id = 0;
    {
        const auto& d = boot_cfg.model.device;
        const auto colon = d.find(':');
        if (colon != std::string::npos) {
            dev_id = std::stoi(d.substr(colon + 1));
        }
    }
    CUDA_CHECK(cudaSetDevice(dev_id));

    auto loader = SafetensorsLoader::open(snapshot);

    const auto t0 = std::chrono::steady_clock::now();

    e.weights_.reserve(loader.num_tensors());
    for (const auto& name : loader.tensor_names()) {
        e.weights_.emplace(name, loader.load_to_device(name));
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    const auto t1 = std::chrono::steady_clock::now();
    const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    const double mib = static_cast<double>(loader.total_bytes()) / (1024.0 * 1024.0);

    std::cerr << "[pie-driver-cuda] loaded " << e.weights_.size() << " tensors ("
              << static_cast<std::uint64_t>(mib) << " MiB) in " << static_cast<int>(ms)
              << " ms; arch=" << e.hf_.arch_name << " (" << e.hf_.model_type << ")\n";

    return e;
}

EngineCapabilities Engine::capabilities() const {
    EngineCapabilities c;
    c.total_pages = 0;  // populated in M1.2.2 once kv_cache lands
    c.kv_page_size = static_cast<int>(boot_.batching.kv_page_size);
    c.swap_pool_size = 0;
    c.max_batch_tokens = static_cast<int>(boot_.batching.max_batch_tokens);
    c.max_batch_size = static_cast<int>(boot_.batching.max_batch_size);
    c.arch_name = hf_.arch_name;
    c.vocab_size = hf_.vocab_size;
    c.max_model_len = hf_.max_position_embeddings;
    c.activation_dtype = boot_.model.dtype;
    c.snapshot_dir = boot_.model.snapshot_dir;
    return c;
}

std::uint64_t Engine::total_weight_bytes() const noexcept {
    std::uint64_t n = 0;
    for (const auto& [_, t] : weights_) n += t.nbytes();
    return n;
}

const DeviceTensor& Engine::get(const std::string& name) const {
    auto it = weights_.find(name);
    if (it == weights_.end()) {
        throw std::runtime_error("engine: weight not loaded: " + name);
    }
    return it->second;
}

void Engine::insert(std::string name, DeviceTensor tensor) {
    auto [it, inserted] = weights_.emplace(std::move(name), std::move(tensor));
    if (!inserted) {
        throw std::runtime_error("engine: weight already registered: " + it->first);
    }
}

}  // namespace pie_cuda_driver
