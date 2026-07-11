// st_probe.cpp — validate SafetensorsView against a real checkpoint. Prints tensor
// count + a few qwen3.6 entries (name / dtype / shape / nbytes) so delta can confirm
// the weight-staging handoff before wiring heap_bind.
//   ./st_probe ~/models/4bit-tied/Qwen3.5-0.8B

#include <cstdio>
#include <string>

#include "safetensors_view.hpp"

using namespace pie::metal;

static void show(const SafetensorsView& v, const std::string& name) {
    auto rt = v.try_get(name);
    if (!rt) { printf("  %-60s  ABSENT\n", name.c_str()); return; }
    std::string shape = "[";
    for (size_t i = 0; i < rt->shape.size(); ++i)
        shape += std::to_string(rt->shape[i]) + (i + 1 < rt->shape.size() ? "," : "");
    shape += "]";
    printf("  %-58s %-5s %-14s %zu B\n", name.c_str(), rt->dtype.c_str(), shape.c_str(),
           rt->nbytes);
}

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    if (argc < 2) { printf("usage: %s <hf_snapshot_dir>\n", argv[0]); return 2; }
    try {
        SafetensorsView v(argv[1]);
        printf("SafetensorsView OK: %zu tensors\n", v.size());

        // qwen3.6 layer-0 quantized q_proj triple + embed (tied lm_head).
        printf("sample tensors:\n");
        show(v, "model.embed_tokens.weight");
        show(v, "model.layers.0.self_attn.q_proj.weight");
        show(v, "model.layers.0.self_attn.q_proj.scales");
        show(v, "model.layers.0.self_attn.q_proj.biases");
        show(v, "model.layers.0.input_layernorm.weight");

        // total staged bytes (= heap Weights region size).
        size_t total = 0;
        for (const auto& n : v.names()) total += v.get(n).nbytes;
        printf("total weight bytes: %.1f MB\n", total / (1024.0 * 1024.0));
        printf("ST_PROBE_OK\n");
        return 0;
    } catch (const std::exception& e) {
        printf("FAIL: %s\n", e.what());
        return 1;
    }
}
