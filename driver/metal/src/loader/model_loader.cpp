#include "model_loader.hpp"

#include <stdexcept>

#include "hf_config.hpp"
#include "safetensors_source.hpp"
#include "../model/arch.hpp"
#include "../model/arch_spec.hpp"
#include "../model/weights.hpp"

namespace pie_metal_driver::loader {

namespace {

// Map an HF torch_dtype string onto the driver DType (activation/KV element).
DType dtype_from_torch(const std::string& torch_dtype) {
    if (torch_dtype == "float16" || torch_dtype == "fp16" || torch_dtype == "half")
        return DType::FP16;
    if (torch_dtype == "float32" || torch_dtype == "fp32" || torch_dtype == "float")
        return DType::FP32;
    // bfloat16 (and anything unrecognised) → bf16, the MLX/Metal default.
    return DType::BF16;
}

const char* dtype_caps_name(DType d) {
    switch (d) {
        case DType::FP16: return "fp16";
        case DType::FP32: return "fp32";
        default:          return "bf16";
    }
}

// Resolve the KV-cache element dtype: "auto" tracks the activation dtype;
// otherwise honour an explicit override from [batching].kv_cache_dtype.
DType resolve_kv_dtype(const std::string& kv_cache_dtype, DType activation) {
    if (kv_cache_dtype == "auto" || kv_cache_dtype.empty()) return activation;
    if (kv_cache_dtype == "bfloat16" || kv_cache_dtype == "bf16") return DType::BF16;
    if (kv_cache_dtype == "float16" || kv_cache_dtype == "fp16") return DType::FP16;
    if (kv_cache_dtype == "float32" || kv_cache_dtype == "fp32") return DType::FP32;
    return activation;
}

}  // namespace

ModelCapabilities derive_capabilities(const model::ModelConfig& cfg) {
    ModelCapabilities caps;
    caps.arch_name           = model::pie_arch_name(cfg.arch);
    caps.vocab_size          = cfg.vocab_size;
    caps.max_model_len       = cfg.max_position_embeddings;
    caps.num_hidden_layers   = cfg.num_hidden_layers;
    caps.num_attention_heads = cfg.num_attention_heads;
    caps.num_key_value_heads = cfg.num_key_value_heads;
    caps.head_dim            = cfg.head_dim;
    caps.hidden_size         = cfg.hidden_size;
    caps.activation_dtype    = dtype_caps_name(dtype_from_torch(cfg.torch_dtype));
    return caps;
}

LoadedWeights load_weights(const std::string& hf_path) {
    LoadedWeights out;
    out.config = parse_hf_config(hf_path);
    if (out.config.arch == model::PieArch::Unknown) {
        throw std::runtime_error(
            "unsupported architecture (model_type='" + out.config.hf_model_type +
            "'): no metal graph builder for this checkpoint yet");
    }
    SafetensorsWeightSource src(hf_path);
    out.weights = model::bind_weights(src, out.config);
    return out;
}

LoadedModel load_model(const std::string& hf_path, const BatchingConfig& batching) {
    LoadedModel out;

    // 1-2. Parse config + load/bind weights (shared with the lower-level path).
    LoadedWeights lw = load_weights(hf_path);
    out.config = lw.config;
    out.caps   = derive_capabilities(out.config);

    // 3. Build the runtime forward graph from the bound weights.
    out.graph = model::make_model_graph(out.config, std::move(lw.weights));

    // 4. Allocate the paged-KV cache from the real model geometry.
    const DType activation = dtype_from_torch(out.config.torch_dtype);
    const DType kv_dtype = resolve_kv_dtype(batching.kv_cache_dtype, activation);
    PagedKvGeometry geo;
    geo.n_layers   = out.config.num_hidden_layers;
    geo.n_pages    = static_cast<int>(batching.total_pages);
    geo.page_size  = static_cast<int>(batching.kv_page_size);
    geo.n_kv_heads = out.config.num_key_value_heads;
    geo.head_dim   = out.config.head_dim;
    out.kv = std::make_unique<PagedKvCache>(geo, kv_dtype);

    // 5. Hybrid linear-attention models (qwen3.6): allocate the separate,
    //    fixed-size-per-slot conv + recurrent Gated-DeltaNet state cache. The
    //    number of linear layers comes from the per-layer attention schedule
    //    ('l' entries); geometry from the config's linear_* fields.
    const model::ArchSpec spec = model::arch_spec_for(out.config.arch, out.config);
    int n_linear = 0;
    for (char k : spec.layer_pattern)
        if (k == 'l') ++n_linear;
    if (n_linear > 0 && out.config.linear_num_value_heads > 0) {
        const int k_heads = out.config.linear_num_key_heads;
        const int v_heads = out.config.linear_num_value_heads;
        const int k_hd    = out.config.linear_key_head_dim;
        const int v_hd    = out.config.linear_value_head_dim;
        LinearStateGeometry lg;
        lg.n_linear_layers = n_linear;
        lg.num_slots       = static_cast<int>(batching.max_forward_requests);
        lg.conv_kernel     = out.config.linear_conv_kernel_dim;
        lg.conv_dim        = 2 * k_heads * k_hd + v_heads * v_hd;
        lg.v_heads         = v_heads;
        lg.k_head_dim      = k_hd;
        lg.v_head_dim      = v_hd;
        out.lin_cache = std::make_unique<LinearStateCache>(lg);
    }

    return out;
}

}  // namespace pie_metal_driver::loader
