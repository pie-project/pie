#pragma once

#include <string>
#include <vector>

#include "config.hpp"
#include "loader/backend_target.hpp"
#include "loader/hf_config.hpp"
#include "loader/layout_plan.hpp"
#include "loader/semantic_graph.hpp"

namespace pie_cuda_driver {

enum class ModelSchemaFamily {
    Generic,
    DenseLlamaLike,
    Phi3,
    QwenMoe,
};

struct ModelSchema {
    std::string name;
    ModelSchemaFamily family = ModelSchemaFamily::Generic;
    bool pack_dense_qkv_and_gate_up = false;
    bool unfuse_phi3_for_tp = false;
    bool shard_fused_moe_experts_for_tp = false;
    bool fuse_per_expert_moe_after_load = false;
};

ModelSchema resolve_model_schema(
    const HfConfig& hf,
    const Config& boot_cfg,
    int tp_size);

LayoutPlan build_model_layout_plan(
    const HfConfig& hf,
    const Config& boot_cfg,
    const CheckpointSource& metadata,
    int tp_size,
    const BackendTarget& target);

}  // namespace pie_cuda_driver
