// The DeepSeek-V4 eager expert names bind as a pair after the native Marlin
// banks moved under `experts.marlin.*`.
//
// Device-free on purpose. `DeviceTensor::view` supplies metadata-only borrowed
// tensors, so this exercises the real `bind_deepseek_v4` name lookup without a
// CUDA allocation or a synthetic copy of the binding decision.

#include "model/deepseek_v4/deepseek_v4.hpp"

#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

namespace {

using pie_cuda_driver::DType;
using pie_cuda_driver::DeviceTensor;
using pie_cuda_driver::HfConfig;
using pie_cuda_driver::LoadedModel;
using pie_cuda_driver::TensorDecl;
using pie_cuda_driver::TensorOwnershipKind;
using pie_cuda_driver::WeightStore;
using pie_cuda_driver::WeightStoreBuilder;
using pie_cuda_driver::model::bind_deepseek_v4;

int failures = 0;
std::uintptr_t next_address = 0x1000;

void check(bool ok, const std::string& what) {
    if (!ok) {
        std::cerr << "FAIL: " << what << "\n";
        ++failures;
    }
}

void add(WeightStoreBuilder& weights, const std::string& name,
         std::vector<std::int64_t> shape) {
    void* data = reinterpret_cast<void*>(next_address);
    next_address += 0x100;
    TensorDecl spec;
    spec.name = name;
    spec.dtype = DType::BF16;
    spec.shape = shape;
    spec.ownership = TensorOwnershipKind::External;
    weights.insert(name,
                   DeviceTensor::view(data, DType::BF16, std::move(shape)),
                   std::move(spec));
}

LoadedModel eager_model() {
    LoadedModel engine;
    auto& cfg = const_cast<HfConfig&>(engine.hf_config());
    cfg = HfConfig{};
    cfg.hidden_size = 64;
    cfg.vocab_size = 128;
    cfg.num_hidden_layers = 1;
    cfg.num_experts = 2;
    cfg.tie_word_embeddings = true;

    auto& store = const_cast<WeightStore&>(engine.weight_store());
    WeightStoreBuilder weights(store);
    add(weights, "embed.weight", {cfg.vocab_size, cfg.hidden_size});
    add(weights, "norm.weight", {cfg.hidden_size});

    const std::string layer = "layers.0.";
    add(weights, layer + "attn_norm.weight", {cfg.hidden_size});
    add(weights, layer + "ffn_norm.weight", {cfg.hidden_size});
    for (const char* projection : {
             "wq_a.weight", "wq_b.weight", "q_norm.weight", "wkv.weight",
             "kv_norm.weight", "wo_a.weight", "wo_b.weight"}) {
        add(weights, layer + "attn." + projection, {1});
    }
    add(weights, layer + "ffn.gate.weight", {cfg.num_experts, cfg.hidden_size});
    add(weights, layer + "ffn.experts.gate_up.weight",
        {cfg.num_experts, 64, cfg.hidden_size});
    add(weights, layer + "ffn.experts.down.weight",
        {cfg.num_experts, cfg.hidden_size, 32});
    weights.finalize();
    return engine;
}

}  // namespace

int main() {
    try {
        LoadedModel engine = eager_model();
        const auto bound = bind_deepseek_v4(engine);
        check(bound.layers.size() == 1, "the fixture binds one layer");
        if (bound.layers.size() == 1) {
            const auto& layer = bound.layers[0];
            check(layer.moe_gate_up_bf16 ==
                      &engine.get("layers.0.ffn.experts.gate_up.weight"),
                  "gate_up BF16 stack binds by its eager name");
            check(layer.moe_down_bf16 ==
                      &engine.get("layers.0.ffn.experts.down.weight"),
                  "down BF16 stack binds by its eager name");
            check(layer.moe_gate_mxfp4 == nullptr &&
                      layer.moe_up_mxfp4 == nullptr &&
                      layer.moe_down_mxfp4 == nullptr,
                  "eager names do not collide with native Marlin banks");
            check(layer.expert_cache == nullptr && layer.experts.empty(),
                  "complete BF16 stacks suppress packed fallbacks");
        }
    } catch (const std::exception& error) {
        check(false, std::string("binding threw: ") + error.what());
    }

    if (failures == 0) {
        std::cout << "deepseek_v4_expert_binding: all checks passed\n";
    }
    return failures == 0 ? 0 : 1;
}
