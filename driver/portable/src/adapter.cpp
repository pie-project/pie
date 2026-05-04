#include "adapter.hpp"

#include <cstring>
#include <stdexcept>
#include <string>

#include "safetensors.hpp"

namespace pie_portable_driver {

namespace {

ggml_type st_to_ggml(StDtype dt, const std::string& dbg) {
    switch (dt) {
        case StDtype::F32:  return GGML_TYPE_F32;
        case StDtype::F16:  return GGML_TYPE_F16;
        case StDtype::BF16: return GGML_TYPE_BF16;
        default:
            throw std::runtime_error(
                "adapter: unsupported safetensors dtype for '" + dbg +
                "': " + std::string(st_dtype_name(dt)));
    }
}

// HF/PyTorch shape `[d_n, ..., d_0]` → ggml ne `[d_0, ..., d_n]`.
ggml_tensor* new_tensor_from_st(ggml_context* ctx,
                                const StTensor& t,
                                const std::string& dbg) {
    if (t.shape.size() != 2) {
        throw std::runtime_error(
            "adapter: expected 2D tensor for '" + dbg +
            "', got rank " + std::to_string(t.shape.size()));
    }
    const auto type = st_to_ggml(t.dtype, dbg);
    auto* out = ggml_new_tensor_2d(ctx, type, t.shape[1], t.shape[0]);
    ggml_set_name(out, dbg.c_str());
    return out;
}

}  // namespace

Adapter::Adapter(ggml_backend_t backend,
                 std::int32_t   n_layers,
                 std::int32_t   /*rank*/,
                 float          scale,
                 const std::filesystem::path& safetensors_path,
                 const Hparams& /*hparams*/)
    : scale_(scale) {
    SafetensorsShard shard(safetensors_path);
    const auto& tensors = shard.tensors();

    constexpr std::size_t MAX_TENSORS = 4096;
    ggml_init_params ip{
        /*.mem_size   =*/ ggml_tensor_overhead() * MAX_TENSORS,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ctx_ = ggml_init(ip);
    if (!ctx_) {
        throw std::runtime_error("adapter: ggml_init failed");
    }

    layers_.resize(n_layers);

    // Per-layer A/B for q/k/v/o targets. Tensors that the adapter doesn't
    // target stay nullptr.
    struct Decl {
        std::string hf_name;
        ggml_tensor** out;
    };
    std::vector<Decl> decls;

    auto try_decl_pair = [&](std::int32_t layer, const std::string& proj,
                             ggml_tensor** a_out, ggml_tensor** b_out) {
        const std::string base =
            "base_model.model.model.layers." + std::to_string(layer) +
            ".self_attn." + proj + "_proj.";
        const std::string a_name = base + "lora_A.weight";
        const std::string b_name = base + "lora_B.weight";
        if (tensors.count(a_name) && tensors.count(b_name)) {
            decls.push_back({a_name, a_out});
            decls.push_back({b_name, b_out});
        }
    };

    for (std::int32_t i = 0; i < n_layers; ++i) {
        try_decl_pair(i, "q", &layers_[i].q_a, &layers_[i].q_b);
        try_decl_pair(i, "k", &layers_[i].k_a, &layers_[i].k_b);
        try_decl_pair(i, "v", &layers_[i].v_a, &layers_[i].v_b);
        try_decl_pair(i, "o", &layers_[i].o_a, &layers_[i].o_b);
    }

    std::vector<std::pair<std::string, ggml_tensor*>> ordered;
    ordered.reserve(decls.size());
    for (auto& d : decls) {
        const auto& src = tensors.at(d.hf_name);
        auto* t = new_tensor_from_st(ctx_, src, d.hf_name);
        *d.out = t;
        ordered.emplace_back(d.hf_name, t);
    }

    buf_ = ggml_backend_alloc_ctx_tensors(ctx_, backend);
    if (!buf_) {
        ggml_free(ctx_);
        ctx_ = nullptr;
        throw std::runtime_error(
            "adapter: ggml_backend_alloc_ctx_tensors failed");
    }

    for (auto& [name, t] : ordered) {
        const auto& src = tensors.at(name);
        const std::size_t need = ggml_nbytes(t);
        if (need != src.nbytes) {
            throw std::runtime_error(
                "adapter: byte-size mismatch for '" + name + "'");
        }
        ggml_backend_tensor_set(t, src.data, 0, src.nbytes);
    }
}

Adapter::~Adapter() {
    if (buf_) ggml_backend_buffer_free(buf_);
    if (ctx_) ggml_free(ctx_);
}

void AdapterPool::insert(std::uint64_t id, std::unique_ptr<Adapter> a) {
    std::lock_guard<std::mutex> g(mu_);
    map_[id] = std::move(a);
}

const Adapter* AdapterPool::get(std::uint64_t id) const {
    std::lock_guard<std::mutex> g(mu_);
    auto it = map_.find(id);
    return it == map_.end() ? nullptr : it->second.get();
}

}  // namespace pie_portable_driver
