#include "loader/model_adapter.hpp"

#include <memory>

namespace pie_cuda_driver {

namespace {

class HfModelAdapter final : public ModelAdapter {
public:
    SemanticGraph build(
        const HfConfig& hf,
        const Config&,
        const CheckpointSource& source) const override
    {
        return build_semantic_graph(hf, source);
    }
};

}  // namespace

std::unique_ptr<ModelAdapter> make_model_adapter(
    const HfConfig&,
    const Config&)
{
    return std::make_unique<HfModelAdapter>();
}

SemanticGraph build_model_semantic_graph(
    const HfConfig& hf,
    const Config& boot_cfg,
    const CheckpointSource& source)
{
    return make_model_adapter(hf, boot_cfg)->build(hf, boot_cfg, source);
}

}  // namespace pie_cuda_driver

