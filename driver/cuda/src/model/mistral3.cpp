#include "model/mistral3.hpp"

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <cuda_runtime.h>

#include "cuda_check.hpp"
#include "kernels/dequant_fp8.hpp"

namespace pie_cuda_driver::model {

namespace {

const DeviceTensor& must(const Engine& e, const std::string& name) {
    if (!e.has(name)) {
        throw std::runtime_error("mistral3: missing weight '" + name + "'");
    }
    return e.get(name);
}

// Dequantize a single FP8 tensor under `fp8_name` (with scalar host
// `scale = *scale_name`) into a fresh bf16 allocation, registered as
// `bf16_name`. If `fp8_name` doesn't exist (e.g. a model variant that
// already shipped this projection in bf16), it's a no-op.
void dequant_fp8_in_place(
    Engine& engine,
    const std::string& fp8_name,
    const std::string& scale_name,
    const std::string& bf16_name)
{
    if (!engine.has(fp8_name)) return;
    const auto& fp8 = engine.get(fp8_name);
    if (fp8.dtype() != DType::UINT8) {
        // The safetensors loader maps F8_E4M3 to UINT8 (raw bits) — if
        // we see something else under this name, the checkpoint is
        // already dequantized and there's nothing to do.
        return;
    }

    // Pull the scalar scale to host. HF stores `weight_scale_inv` as a
    // 1-element fp32 (sometimes bf16) tensor.
    const auto& scale_tensor = must(engine, scale_name);
    float scale = 1.f;
    if (scale_tensor.dtype() == DType::FP32) {
        CUDA_CHECK(cudaMemcpy(&scale, scale_tensor.data(), sizeof(float),
                              cudaMemcpyDeviceToHost));
    } else if (scale_tensor.dtype() == DType::BF16) {
        std::uint16_t bits = 0;
        CUDA_CHECK(cudaMemcpy(&bits, scale_tensor.data(), sizeof(std::uint16_t),
                              cudaMemcpyDeviceToHost));
        const std::uint32_t f32_bits = static_cast<std::uint32_t>(bits) << 16;
        std::memcpy(&scale, &f32_bits, sizeof(float));
    } else {
        throw std::runtime_error(
            "mistral3: unsupported scale dtype for '" + scale_name + "'");
    }

    // Allocate the bf16 destination matching the FP8 source's logical
    // shape.
    auto bf16 = DeviceTensor::allocate(DType::BF16, fp8.shape());
    kernels::launch_dequant_fp8_e4m3_to_bf16(
        static_cast<const std::uint8_t*>(fp8.data()),
        bf16.data(), scale, fp8.numel(), /*stream=*/nullptr);
    engine.insert(bf16_name, std::move(bf16));
}

}  // namespace

Qwen3Weights bind_mistral3(Engine& engine) {
    const auto& cfg = engine.hf_config();

    // Mistral-Small-3.1 keeps norms and the embedding in bf16 (or
    // fp16); only Q/K/V/O and gate/up/down projections come as FP8.
    // For each layer, materialize bf16 copies under the canonical names
    // bind_llama_like expects.
    const auto suffix_pairs = std::vector<std::pair<std::string, std::string>>{
        {"self_attn.q_proj",    "self_attn.q_proj"},
        {"self_attn.k_proj",    "self_attn.k_proj"},
        {"self_attn.v_proj",    "self_attn.v_proj"},
        {"self_attn.o_proj",    "self_attn.o_proj"},
        {"mlp.gate_proj",       "mlp.gate_proj"},
        {"mlp.up_proj",         "mlp.up_proj"},
        {"mlp.down_proj",       "mlp.down_proj"},
    };

    for (int i = 0; i < cfg.num_hidden_layers; ++i) {
        const std::string p = "model.layers." + std::to_string(i) + ".";
        for (const auto& [src, dst] : suffix_pairs) {
            const std::string fp8_name   = p + src + ".weight";
            const std::string scale_name = p + src + ".weight_scale_inv";
            const std::string bf16_name  = p + dst + ".weight";

            // If the engine already has the bf16 version registered
            // (e.g. the safetensors loader saw F8_E4M3 and went down
            // a different path), we don't double-dequantize.
            if (engine.has(bf16_name) && engine.get(bf16_name).dtype() == DType::BF16) {
                continue;
            }
            dequant_fp8_in_place(engine, fp8_name, scale_name, bf16_name);
        }
    }

    return bind_llama_like(engine);
}

}  // namespace pie_cuda_driver::model
