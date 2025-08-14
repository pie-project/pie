// NOTE: This translation unit is compiled with NVCC. Avoid including heavy C++ template
// libraries (like toml++) directly here to prevent CUDA compilation issues. We delegate
// TOML parsing to a separate C++ source file.

#include <cassert>
#include <iostream>
#include <vector>
#include <fstream>
#include <filesystem>
#include <optional>
#include <cstdlib>

#include "model.hpp"
#include "config.hpp"
// Include implementation unit (not added separately in target sources) to instantiate templates.
#include "../../src/l4ma.cu"

// Minimal stub to construct AppConfig & ModelMetadata similar to runtime.
static AppConfig make_test_config() {
    AppConfig cfg;
    cfg.model_name = "llama-3.2-1b-instruct"; // assumes weights exist in cache for full forward
    cfg.cache_dir = std::filesystem::path(std::getenv("HOME")) / ".cache" / "pie";
    cfg.kv_page_size = 32;
    cfg.dist_size = 16; // small for faster test
    cfg.max_num_kv_pages = 4;
    cfg.max_num_embeds = 4096;
    cfg.device = "cuda:0";
    cfg.dtype = "bfloat16";
    return cfg;
}

// Extern loader implemented in a regular C++ file.
ModelMetadata load_model_metadata_for_test(const AppConfig& cfg);

int main() {
    auto cfg = make_test_config();
    auto meta = load_model_metadata_for_test(cfg);

    Model model(cfg, meta);

    // Allocate a KV block for context.
    uint32_t block_id = 10;
    model.handle_allocate({Model::AllocateCommand{Model::ObjectKind::KV_BLOCK, block_id, 1}});

    // Prepare a simple forward text command with 4 tokens, request distributions for last 2 tokens.
    Model::ForwardTextCommand cmd;
    cmd.kv_page_last_len = 0; // no previous tokens in block, treat as fresh context
    cmd.kv_page_ids = {}; // empty context pages for first forward
    cmd.token_ids = {1,2,3,4};
    cmd.position_ids = {0,1,2,3};
    cmd.output_indices = {2,3};

    auto results = model.handle_forward_text({cmd});
    assert(results.size() == 1);
    auto& dists = results[0];
    assert(dists.size() == 2);

    for (auto& dist : dists) {
        assert(dist.token_ids.size() == cfg.dist_size);
        assert(dist.probabilities.size() == cfg.dist_size);
    }

    std::cout << "ForwardText test passed with " << dists.size() << " distributions." << std::endl;
    return 0;
}
