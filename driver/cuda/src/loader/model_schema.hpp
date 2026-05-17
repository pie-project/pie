#pragma once

#include <string>
#include <vector>

#include "config.hpp"
#include "loader/hf_config.hpp"
#include "loader/load_plan.hpp"
#include "loader/safetensors.hpp"

namespace pie_cuda_driver {

int llama_like_shard_axis(const std::string& name);

enum class ModelSchemaFamily {
    Generic,
    DenseLlamaLike,
    Phi3,
    QwenMoe,
};

enum class LogicalTensorRole {
    Unknown,
    Embedding,
    LmHead,
    Norm,
    AttentionQ,
    AttentionK,
    AttentionV,
    AttentionO,
    AttentionQkv,
    MlpGate,
    MlpUp,
    MlpDown,
    MlpGateUp,
    MoeExpertGate,
    MoeExpertUp,
    MoeExpertDown,
    MoeExpertsGateUp,
    MoeExpertsDown,
    QuantPackedData,
    QuantScale,
    QuantZeroPoint,
    Bias,
};

enum class LogicalTensorGroupKind {
    PackedQkv,
    PackedGateUp,
    RowRangeSplit,
    PerExpertMoe,
    FusedMoeExperts,
    GptOssMxfp4,
    Fp8ScaleInv,
    OfflineInt4,
    CompressedQuant,
};

enum class Mxfp4MoeLowering {
    Bf16Dequant,
    RoutedDequant,
    NativeGemm,
};

struct LoadTarget {
    int device_major = 0;
    int device_minor = 0;
    bool fp8_native = false;
    bool gptq_marlin_int4 = true;
    bool mxfp4_native_gemm = false;

    // RoutedDequant keeps QuantPacked MXFP4 expert weights resident and lets
    // the MoE runtime dequantize only routed experts into bounded BF16 scratch.
    // NativeGemm is reserved for a backend that consumes MXFP4 directly inside
    // expert GEMM kernels; selecting it requires `mxfp4_native_gemm`.
    Mxfp4MoeLowering mxfp4_moe = Mxfp4MoeLowering::Bf16Dequant;
};

struct LogicalTensor {
    std::string raw_name;
    std::string runtime_name;
    LogicalTensorRole role = LogicalTensorRole::Unknown;
    DType checkpoint_dtype;
    std::vector<std::int64_t> checkpoint_shape;
    // Tensor-parallel shard axis declared by the schema adapter. Lowering
    // consumes this value directly instead of reparsing architecture suffixes.
    int shard_axis = -1;
};

struct LogicalTensorGroup {
    LogicalTensorGroupKind kind = LogicalTensorGroupKind::PackedQkv;
    std::string runtime_base;
    std::vector<std::string> raw_names;
    std::vector<std::string> runtime_names;
    // Adapters declare the semantic role of each source and runtime output.
    // Lowering uses these roles instead of reparsing architecture suffixes.
    std::vector<LogicalTensorRole> raw_roles;
    std::vector<LogicalTensorRole> runtime_roles;
};

struct LogicalTensorGraph {
    std::vector<LogicalTensor> tensors;
    std::vector<LogicalTensorGroup> groups;
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

LogicalTensorRole infer_logical_tensor_role(const std::string& runtime_name);
LogicalTensorGraph build_logical_tensor_graph(
    const HfConfig& hf,
    const TensorMetadataSource& metadata);

LoadPlan build_model_load_plan(
    const HfConfig& hf,
    const Config& boot_cfg,
    const TensorMetadataSource& metadata,
    int tp_size,
    const LoadTarget& target);

}  // namespace pie_cuda_driver
