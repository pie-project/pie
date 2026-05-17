#pragma once

#include <memory>

#include "config.hpp"
#include "loader/checkpoint_source.hpp"
#include "loader/hf_config.hpp"
#include "loader/semantic_graph.hpp"

namespace pie_cuda_driver {

class ModelAdapter {
public:
    virtual ~ModelAdapter() = default;

    virtual SemanticGraph build(
        const HfConfig& hf,
        const Config& boot_cfg,
        const CheckpointSource& source) const = 0;
};

std::unique_ptr<ModelAdapter> make_model_adapter(
    const HfConfig& hf,
    const Config& boot_cfg);

SemanticGraph build_model_semantic_graph(
    const HfConfig& hf,
    const Config& boot_cfg,
    const CheckpointSource& source);

}  // namespace pie_cuda_driver

