#include "loader/model_schema.hpp"

#include "loader/model_family.hpp"

namespace pie_cuda_driver {

ModelSchema resolve_model_schema(
    const HfConfig& hf,
    const Config& boot_cfg,
    int tp_size)
{
    ModelSchema schema;
    schema.name = hf.model_type.empty() ? hf.arch_name : hf.model_type;
    schema.pack_dense_qkv_and_gate_up =
        supports_dense_llama_packed_load(hf, boot_cfg);
    schema.unfuse_phi3_for_tp = (hf.model_type == "phi3");
    schema.shard_fused_moe_experts_for_tp =
        (tp_size > 1) && is_qwen_moe_model_type(hf.model_type);
    schema.fuse_per_expert_moe_after_load =
        is_qwen_moe_model_type(hf.model_type);
    schema.family = model_schema_family_for_type(hf.model_type);
    return schema;
}

}  // namespace pie_cuda_driver
