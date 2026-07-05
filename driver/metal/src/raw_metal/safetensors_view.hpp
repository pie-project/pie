#pragma once
// safetensors_view.hpp — MLX-free, zero-copy safetensors reader for the raw-Metal
// weight-staging handoff. The existing loader (SafetensorsWeightSource) returns
// mlx::core::array and is part of the MLX dependency being removed; this exposes raw
// per-tensor byte spans instead, so delta's heap_bind can stage straight into a
// heap slot's contents():
//
//     SafetensorsView view(hf_path);
//     RawTensor rt = view.get("model.layers.0.self_attn.q_proj.weight");
//     SlotHandle s = ctx.heap_alloc(rt.nbytes);
//     memcpy(s.contents(), rt.data, rt.nbytes);          // stage into the resident heap
//     ctx.arg_bind(Kernel::QmvQ, 0, (uint8_t)bind::Qmv::W, s);
//
// Quantized linears are three entries (`.weight` u32-packed / `.scales` / `.biases`);
// stage each into its own slot per bind::Qmv. The view mmaps each shard read-only and
// keeps it mapped for its lifetime — `data` is a borrowed pointer into the mapping.

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace pie_metal_driver::raw_metal {

struct RawTensor {
    const uint8_t*       data   = nullptr;  // borrowed: into the shard mmap
    size_t               nbytes = 0;
    std::string          dtype;             // safetensors dtype: "BF16"/"F16"/"F32"/"U32"/...
    std::vector<int64_t> shape;

    bool valid() const { return data != nullptr; }
};

class SafetensorsView {
  public:
    // Maps every shard under `hf_path` (honors model.safetensors.index.json, else
    // model.safetensors, else any *.safetensors). Throws on missing/corrupt weights.
    explicit SafetensorsView(const std::string& hf_path);
    ~SafetensorsView();

    SafetensorsView(const SafetensorsView&)            = delete;
    SafetensorsView& operator=(const SafetensorsView&) = delete;

    RawTensor                get(const std::string& name) const;  // throws if absent
    std::optional<RawTensor> try_get(const std::string& name) const;
    bool                     has(const std::string& name) const;

    size_t                   size() const;          // number of tensors
    std::vector<std::string> names() const;

  private:
    struct Impl;
    Impl* impl_;
};

}  // namespace pie_metal_driver::raw_metal
