#include "model/nemotron_h.hpp"

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

void copy_rows_bf16(
    const DeviceTensor& src,
    DeviceTensor& dst,
    int cols,
    const std::vector<std::pair<int, int>>& segments)
{
    auto* dst_base = static_cast<std::uint8_t*>(dst.data());
    const auto* src_base = static_cast<const std::uint8_t*>(src.data());
    std::size_t dst_row = 0;
    for (const auto& [src_row, rows] : segments) {
        if (rows <= 0) continue;
        const std::size_t row_bytes =
            static_cast<std::size_t>(cols) * sizeof(std::uint16_t);
        CUDA_CHECK(cudaMemcpyAsync(
            dst_base + dst_row * row_bytes,
            src_base + static_cast<std::size_t>(src_row) * row_bytes,
            static_cast<std::size_t>(rows) * row_bytes,
            cudaMemcpyDeviceToDevice));
        dst_row += static_cast<std::size_t>(rows);
    }
}

void copy_cols_bf16(
    const DeviceTensor& src,
    DeviceTensor& dst,
    int rows,
    int src_cols,
    int col_start,
    int col_count)
{
    const auto* src_base = static_cast<const std::uint8_t*>(src.data()) +
        static_cast<std::size_t>(col_start) * sizeof(std::uint16_t);
    auto* dst_base = static_cast<std::uint8_t*>(dst.data());
    CUDA_CHECK(cudaMemcpy2DAsync(
        dst_base,
        static_cast<std::size_t>(col_count) * sizeof(std::uint16_t),
        src_base,
        static_cast<std::size_t>(src_cols) * sizeof(std::uint16_t),
        static_cast<std::size_t>(col_count) * sizeof(std::uint16_t),
        static_cast<std::size_t>(rows),
        cudaMemcpyDeviceToDevice));
}

void materialize_mamba_tp_shard(
    NemotronHLayerWeights& Lw,
    const HfConfig& cfg,
    int tp_rank,
    int tp_size)
{
    const int H = cfg.hidden_size;
    const int heads = cfg.mamba_num_heads;
    const int head_dim = cfg.mamba_head_dim;
    const int groups = cfg.mamba_n_groups;
    const int state = cfg.mamba_state_size;
    const int conv_kernel = cfg.mamba_conv_kernel;
    if (tp_size <= 1) return;
    if (heads % tp_size != 0 || groups % tp_size != 0) {
        throw std::runtime_error(
            "nemotron_h: Mamba TP sharding requires mamba heads/groups to be "
            "divisible by tp_size");
    }

    const int local_heads = heads / tp_size;
    const int local_groups = groups / tp_size;
    const int full_intermediate = heads * head_dim;
    const int local_intermediate = local_heads * head_dim;
    const int full_group_state = groups * state;
    const int local_group_state = local_groups * state;
    const int full_conv_dim = full_intermediate + 2 * full_group_state;
    const int local_conv_dim = local_intermediate + 2 * local_group_state;
    const int local_projection_dim =
        local_intermediate + local_conv_dim + local_heads;

    const int rank_head = tp_rank * local_heads;
    const int rank_intermediate = tp_rank * local_intermediate;
    const int rank_group_state = tp_rank * local_group_state;

    Lw.mamba_in_proj_tp = DeviceTensor::allocate(
        DType::BF16, {local_projection_dim, H});
    copy_rows_bf16(
        *Lw.mamba_in_proj, Lw.mamba_in_proj_tp, H,
        {
            {rank_intermediate, local_intermediate},
            {full_intermediate + rank_intermediate, local_intermediate},
            {2 * full_intermediate + rank_group_state, local_group_state},
            {2 * full_intermediate + full_group_state + rank_group_state,
             local_group_state},
            {2 * full_intermediate + 2 * full_group_state + rank_head,
             local_heads},
        });

    Lw.mamba_conv_w_tp = DeviceTensor::allocate(
        DType::BF16, {local_conv_dim, conv_kernel});
    copy_rows_bf16(
        *Lw.mamba_conv_w, Lw.mamba_conv_w_tp, conv_kernel,
        {
            {rank_intermediate, local_intermediate},
            {full_intermediate + rank_group_state, local_group_state},
            {full_intermediate + full_group_state + rank_group_state,
             local_group_state},
        });
    Lw.mamba_conv_b_tp = DeviceTensor::allocate(
        DType::BF16, {local_conv_dim});
    copy_rows_bf16(
        *Lw.mamba_conv_b, Lw.mamba_conv_b_tp, 1,
        {
            {rank_intermediate, local_intermediate},
            {full_intermediate + rank_group_state, local_group_state},
            {full_intermediate + full_group_state + rank_group_state,
             local_group_state},
        });

    Lw.mamba_A_log_tp = DeviceTensor::allocate(DType::BF16, {local_heads});
    Lw.mamba_D_tp = DeviceTensor::allocate(DType::BF16, {local_heads});
    Lw.mamba_dt_bias_tp = DeviceTensor::allocate(DType::BF16, {local_heads});
    copy_rows_bf16(*Lw.mamba_A_log, Lw.mamba_A_log_tp, 1,
                   {{rank_head, local_heads}});
    copy_rows_bf16(*Lw.mamba_D, Lw.mamba_D_tp, 1,
                   {{rank_head, local_heads}});
    copy_rows_bf16(*Lw.mamba_dt_bias, Lw.mamba_dt_bias_tp, 1,
                   {{rank_head, local_heads}});

    Lw.mamba_norm_w_tp = DeviceTensor::allocate(
        DType::BF16, {local_intermediate});
    copy_rows_bf16(*Lw.mamba_norm_w, Lw.mamba_norm_w_tp, 1,
                   {{rank_intermediate, local_intermediate}});

    Lw.mamba_out_proj_tp = DeviceTensor::allocate(
        DType::BF16, {H, local_intermediate});
    copy_cols_bf16(
        *Lw.mamba_out_proj, Lw.mamba_out_proj_tp,
        H, full_intermediate, rank_intermediate, local_intermediate);

    Lw.mamba_tp_sharded = true;
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

bool nemotron_h_tp_mamba_sharding_enabled(int tp_size) {
    if (tp_size <= 1) return false;
    const char* disabled = std::getenv("PIE_NEMOTRON_DISABLE_TP_MAMBA_SHARD");
    return disabled == nullptr || disabled[0] == '\0' || disabled[0] == '0';
}

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
    const int tp_rank = engine.distributed().tp_rank;
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
            if (shard_mamba) {
                materialize_mamba_tp_shard(Lw, cfg, tp_rank, tp_size);
            }
            const int local_heads = Lw.mamba_tp_sharded
                ? cfg.mamba_num_heads / tp_size
                : cfg.mamba_num_heads;
            Lw.mamba_A = DeviceBuffer<float>::alloc(local_heads);
            Lw.mamba_D_f32 = DeviceBuffer<float>::alloc(local_heads);
            Lw.mamba_dt_bias_f32 =
                DeviceBuffer<float>::alloc(local_heads);
            kernels::launch_nemotron_prepare_mamba_params(
                Lw.mamba_tp_sharded ? Lw.mamba_A_log_tp.data()
                                     : Lw.mamba_A_log->data(),
                Lw.mamba_tp_sharded ? Lw.mamba_D_tp.data()
                                     : Lw.mamba_D->data(),
                Lw.mamba_tp_sharded ? Lw.mamba_dt_bias_tp.data()
                                     : Lw.mamba_dt_bias->data(),
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
