// Smoke test for delta's loader seam: parse config.json, load safetensors into
// MLX arrays, bind weights, build the graph, allocate the paged-KV cache, and
// run a tiny real forward on the default (Metal GPU) device.
//
// Usage: smoke_load <hf_path>

#include <iostream>
#include <string>

#include <mlx/mlx.h>

#include "loader/model_loader.hpp"
#include "model/model_graph.hpp"

namespace mx = mlx::core;
using namespace pie_metal_driver;

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "usage: smoke_load <hf_path>\n";
        return 2;
    }
    const std::string hf_path = argv[1];

    std::cerr << "[smoke] metal::is_available = " << std::boolalpha
              << mx::metal::is_available() << "\n";
    std::cerr << "[smoke] default_device = "
              << (mx::default_device() == mx::Device::gpu ? "gpu" : "cpu") << "\n";

    // 1-4: parse + load + bind + graph + KV.
    BatchingConfig batching;   // defaults: page=32, total_pages=1024
    auto model = loader::load_model(hf_path, batching);

    const auto& c = model.caps;
    std::cerr << "[smoke] LOADED arch=" << c.arch_name
              << " layers=" << c.num_hidden_layers
              << " heads=" << c.num_attention_heads
              << " kv_heads=" << c.num_key_value_heads
              << " head_dim=" << c.head_dim
              << " hidden=" << c.hidden_size
              << " vocab=" << c.vocab_size
              << " max_len=" << c.max_model_len
              << " act=" << c.activation_dtype << "\n";

    // Tiny forward: a 4-token prefill for a single request. ForwardBatch has
    // no default ctor (MLX arrays), so aggregate-initialise every field.
    const int n = 4;
    model::ForwardBatch batch{
        /*token_ids*/        mx::array({1, 2, 3, 4}, {n}, mx::int32),
        /*positions*/        mx::array({0, 1, 2, 3}, {n}, mx::int32),
        /*logit_rows*/       mx::array({n - 1}, {1}, mx::int32),
        /*kv_page_indices*/  mx::array({0}, {1}, mx::int32),
        /*kv_page_indptr*/   mx::array({0, 1}, {2}, mx::int32),
        /*kv_last_page_lens*/mx::array({n}, {1}, mx::int32),
        /*qo_indptr*/        mx::array({0, n}, {2}, mx::int32),
        /*kv_write_indices*/ mx::array({0, 1, 2, 3}, {n}, mx::int32),
        /*n_total*/    n,
        /*n_requests*/ 1,
        /*n_slots*/    1,
        /*pure_decode*/false,
    };

    // Hybrid linear-attention (qwen3.6): thread delta's conv/recurrent state
    // cache + this request's persistent slot. Null/ignored for every other arch.
    // The 4-token prefill routes linear layers through gated_delta_net_varlen,
    // which needs the host CSR token spans.
    batch.lin_cache      = model.lin_cache.get();
    batch.slot_ids       = mx::array({0}, {1}, mx::int32);
    batch.qo_indptr_host = {0, n};
    if (model.lin_cache) {
        std::cerr << "[smoke] hybrid linear-attn: lin_cache slots="
                  << model.lin_cache->num_slots()
                  << " layers=" << model.lin_cache->n_linear_layers() << "\n";
    }

    mx::array logits = model.graph->forward(batch, *model.kv);
    mx::eval(logits);
    model.kv->eval();

    std::cerr << "[smoke] forward OK, logits shape = [";
    for (size_t i = 0; i < logits.shape().size(); ++i) {
        std::cerr << logits.shape()[i] << (i + 1 < logits.shape().size() ? "," : "");
    }
    std::cerr << "]\n";

    // Logits are [n_slots, vocab] (token-major) — argmax over the vocab axis.
    mx::array next = mx::argmax(logits, /*axis=*/1);
    mx::eval(next);
    std::cerr << "[smoke] argmax token id = " << next.item<int>() << "\n";
    std::cerr << "[smoke] PASS\n";
    return 0;
}
