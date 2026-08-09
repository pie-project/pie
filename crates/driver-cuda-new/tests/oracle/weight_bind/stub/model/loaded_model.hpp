// Minimal stand-in for `model/loaded_model.hpp`.
//
// `bind_llama_like` / `bind_phi3` / `bind_olmo3` touch exactly five things on
// a LoadedModel: `hf_config()`, `has()`, `get()`, `quant_meta()`, and the
// DeviceTensor type. Everything else in the real header — the loader, the
// device pool, the CUDA graph state — is irrelevant to what those functions
// decide, so it is left out rather than stubbed.
//
// `has()` here is a *recorder*: it appends every name the bind function asks
// about, in order. That is half the transcript. A bind function that resolved
// all the right pointers by asking for the wrong names first, or by probing in
// a different order, would produce identical weights and a different
// transcript — which is the point, because the probe order is what decides
// which of two present tensors wins.

#pragma once

#include <cstddef>
#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <vector>

#include "quant_meta.hpp"
#include "tensor.hpp"

namespace pie_cuda_driver {

struct HfConfig {
    int num_hidden_layers = 2;
    bool tie_word_embeddings = false;
    bool attention_bias = false;
    bool use_qk_norm = false;
};

namespace model {

class LoadedModel {
public:
    // The probe log. Every `has()` the bind function performs lands here.
    mutable std::vector<std::string> probes;

    HfConfig cfg;

    void add(const std::string& name) {
        // Identity is the DeviceTensor's own address, because that is what the
        // binders compare (`w.lm_head == w.embed`) and what the forward path
        // holds. `std::map` guarantees reference stability, so these stay
        // valid as the set grows.
        auto& t = tensors_[name];
        t = DeviceTensor();
        names_by_ptr_[&t] = name;
    }

    void add_quant(const std::string& weight, QuantMeta meta) {
        quant_[weight] = std::move(meta);
    }

    const HfConfig& hf_config() const { return cfg; }

    bool has(const std::string& name) const {
        probes.push_back(name);
        return tensors_.count(name) != 0;
    }

    const DeviceTensor& get(const std::string& name) const {
        return tensors_.at(name);
    }

    std::optional<QuantMeta> quant_meta(const std::string& name) const {
        auto it = quant_.find(name);
        if (it == quant_.end()) return std::nullopt;
        return it->second;
    }

    // Resolve a bound pointer back to the name it was bound from, so the
    // transcript can say `q_proj=model.layers.0.self_attn.q_proj.weight`
    // rather than an address.
    std::string name_of(const DeviceTensor* t) const {
        if (t == nullptr) return "null";
        auto it = names_by_ptr_.find(t);
        return it == names_by_ptr_.end() ? "unknown" : it->second;
    }

private:
    std::map<std::string, DeviceTensor> tensors_;
    std::map<const DeviceTensor*, std::string> names_by_ptr_;
    std::map<std::string, QuantMeta> quant_;
};

}  // namespace model
}  // namespace pie_cuda_driver
