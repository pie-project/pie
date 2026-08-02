#include "model/nemotron_h/nemotron_h.hpp"

#include "model/nemotron_h/nemotron_h.hpp"

#include <algorithm>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <cuda_runtime.h>

#include "cuda_check.hpp"
#include "kernels/nemotron_h.hpp"

namespace pie_cuda_driver::model {

namespace {

const DeviceTensor& must(const LoadedModel& e, const std::string& name) {
    if (!e.has(name)) {
        throw std::runtime_error("nemotron_h: missing weight '" + name + "'");
    }
    return e.get(name);
}

void upload_expert_ptrs(NemotronHLayerWeights& Lw) {
    const int E = static_cast<int>(Lw.expert_up.size());
    if (E == 0) return;
    std::vector<const std::uint16_t*> up(static_cast<std::size_t>(E));
    std::vector<const std::uint16_t*> down(static_cast<std::size_t>(E));
    for (int e = 0; e < E; ++e) {
        up[static_cast<std::size_t>(e)] =
            static_cast<const std::uint16_t*>(Lw.expert_up[e]->data());
        down[static_cast<std::size_t>(e)] =
            static_cast<const std::uint16_t*>(Lw.expert_down[e]->data());
    }
    Lw.expert_up_ptrs = DeviceBuffer<const std::uint16_t*>::alloc(E);
    Lw.expert_down_ptrs = DeviceBuffer<const std::uint16_t*>::alloc(E);
    CUDA_CHECK(cudaMemcpy(Lw.expert_up_ptrs.data(), up.data(),
                          up.size() * sizeof(const std::uint16_t*),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(Lw.expert_down_ptrs.data(), down.data(),
                          down.size() * sizeof(const std::uint16_t*),
                          cudaMemcpyHostToDevice));
}

}  // namespace

NemotronHWeights bind_nemotron_h(const LoadedModel& engine) {
    const auto& cfg = engine.hf_config();
    if (cfg.model_type != "nemotron_h") {
        throw std::runtime_error("bind_nemotron_h called for model_type='" +
                                 cfg.model_type + "'");
    }
    if (cfg.layer_types.size() != static_cast<std::size_t>(cfg.num_hidden_layers)) {
        throw std::runtime_error("nemotron_h: layer_types not parsed");
    }

    const std::string p = "language_model.";
    NemotronHWeights w;
    w.embed = &must(engine, p + "backbone.embeddings.weight");
    w.final_norm = &must(engine, p + "backbone.norm_f.weight");
    w.lm_head = &must(engine, p + "lm_head.weight");
    w.layers.resize(static_cast<std::size_t>(cfg.num_hidden_layers));

    int kv_slot = 0;
    bool prepared_mamba_params = false;
    const int tp_size = std::max(1, engine.distributed().tp_size);
    const bool shard_mamba = nemotron_h_tp_mamba_sharding_enabled(tp_size);
    for (int li = 0; li < cfg.num_hidden_layers; ++li) {
        const std::string lp =
            p + "backbone.layers." + std::to_string(li) + ".";
        const std::string mp = lp + "mixer.";
        auto& Lw = w.layers[static_cast<std::size_t>(li)];
        Lw.norm = &must(engine, lp + "norm.weight");
        const auto& kind = cfg.layer_types[static_cast<std::size_t>(li)];
        if (kind == "mamba") {
            Lw.kind = NemotronHLayerWeights::Kind::Mamba;
            Lw.mamba_in_proj = &must(engine, mp + "in_proj.weight");
            Lw.mamba_conv_w = &must(engine, mp + "conv1d.weight");
            Lw.mamba_conv_b = &must(engine, mp + "conv1d.bias");
            Lw.mamba_A_log = &must(engine, mp + "A_log");
            Lw.mamba_D = &must(engine, mp + "D");
            Lw.mamba_dt_bias = &must(engine, mp + "dt_bias");
            Lw.mamba_norm_w = &must(engine, mp + "norm.weight");
            Lw.mamba_out_proj = &must(engine, mp + "out_proj.weight");
            // The contract already split these (`nemotron_h_mamba_tp_shards`),
            // so what arrived is this rank's share and its length says so.
            Lw.mamba_tp_sharded = shard_mamba;
            const int local_heads = shard_mamba
                ? cfg.mamba_num_heads / tp_size
                : cfg.mamba_num_heads;
            Lw.mamba_A = DeviceBuffer<float>::alloc(local_heads);
            Lw.mamba_D_f32 = DeviceBuffer<float>::alloc(local_heads);
            Lw.mamba_dt_bias_f32 =
                DeviceBuffer<float>::alloc(local_heads);
            kernels::launch_nemotron_prepare_mamba_params(
                Lw.mamba_A_log->data(),
                Lw.mamba_D->data(),
                Lw.mamba_dt_bias->data(),
                Lw.mamba_A.data(),
                Lw.mamba_D_f32.data(),
                Lw.mamba_dt_bias_f32.data(),
                local_heads,
                /*stream=*/0);
            prepared_mamba_params = true;
        } else if (kind == "attention") {
            Lw.kind = NemotronHLayerWeights::Kind::Attention;
            Lw.q_proj = &must(engine, mp + "q_proj.weight");
            Lw.k_proj = &must(engine, mp + "k_proj.weight");
            Lw.v_proj = &must(engine, mp + "v_proj.weight");
            Lw.o_proj = &must(engine, mp + "o_proj.weight");
            Lw.kv_layer = kv_slot++;
        } else if (kind == "moe") {
            Lw.kind = NemotronHLayerWeights::Kind::MoE;
            Lw.router = &must(engine, mp + "gate.weight");
            Lw.router_correction_bias =
                &must(engine, mp + "gate.e_score_correction_bias");
            const std::string packed_up =
                mp + "experts.up_proj.packed.weight";
            const std::string packed_down =
                mp + "experts.down_proj.packed.weight";
            if (engine.has(packed_up)) {
                Lw.expert_up_packed = &must(engine, packed_up);
            }
            if (engine.has(packed_down)) {
                Lw.expert_down_packed = &must(engine, packed_down);
            }
            Lw.expert_up.resize(static_cast<std::size_t>(cfg.num_experts));
            Lw.expert_down.resize(static_cast<std::size_t>(cfg.num_experts));
            for (int e = 0; e < cfg.num_experts; ++e) {
                const std::string ep = mp + "experts." + std::to_string(e) + ".";
                Lw.expert_up[static_cast<std::size_t>(e)] =
                    &must(engine, ep + "up_proj.weight");
                Lw.expert_down[static_cast<std::size_t>(e)] =
                    &must(engine, ep + "down_proj.weight");
            }
            Lw.shared_up = &must(engine, mp + "shared_experts.up_proj.weight");
            Lw.shared_down = &must(engine, mp + "shared_experts.down_proj.weight");
            upload_expert_ptrs(Lw);
        } else {
            throw std::runtime_error("nemotron_h: unsupported layer kind '" +
                                     kind + "'");
        }
    }
    if (prepared_mamba_params) {
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    return w;
}

}  // namespace pie_cuda_driver::model
