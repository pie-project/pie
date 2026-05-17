#pragma once

#include <string>
#include <vector>

#include "loader/checkpoint_source.hpp"
#include "loader/hf_config.hpp"
#include "tensor.hpp"

namespace pie_cuda_driver {

enum class SemanticRole {
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

enum class SemanticGroupKind {
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

struct SemanticTensor {
    std::string raw_name;
    std::string runtime_name;
    SemanticRole role = SemanticRole::Unknown;
    DType checkpoint_dtype;
    std::vector<std::int64_t> checkpoint_shape;
    int shard_axis = -1;
};

struct SemanticGroup {
    SemanticGroupKind kind = SemanticGroupKind::PackedQkv;
    std::string runtime_base;
    std::vector<std::string> raw_names;
    std::vector<std::string> runtime_names;
    std::vector<SemanticRole> raw_roles;
    std::vector<SemanticRole> runtime_roles;
};

struct SemanticGraph {
    std::vector<SemanticTensor> tensors;
    std::vector<SemanticGroup> groups;
};

int llama_like_shard_axis(const std::string& name);
SemanticRole infer_semantic_role(const std::string& runtime_name);
SemanticGraph build_semantic_graph(
    const HfConfig& hf,
    const CheckpointSource& metadata);
bool semantic_graph_has_group(
    const SemanticGraph& graph,
    SemanticGroupKind kind);

}  // namespace pie_cuda_driver

