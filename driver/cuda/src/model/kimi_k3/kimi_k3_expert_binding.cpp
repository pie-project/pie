#include "model/kimi_k3/kimi_k3_expert_binding.hpp"

#include <stdexcept>

namespace pie_cuda_driver::model {

bool kimi_k3_use_streamed_experts(
    bool gate_up_stack_present,
    bool down_stack_present,
    bool group_found,
    std::uint32_t group_arity,
    int num_experts,
    int layer,
    const std::string& group_name)
{
    if (gate_up_stack_present != down_stack_present) {
        throw std::runtime_error(
            "kimi_k3: layer " + std::to_string(layer) + " published " +
            (gate_up_stack_present ? "experts.gate_up_proj without "
                                     "experts.down_proj"
                                   : "experts.down_proj without "
                                     "experts.gate_up_proj"));
    }
    if (gate_up_stack_present) {
        return false;
    }
    if (!group_found) {
        throw std::runtime_error(
            "kimi_k3: layer " + std::to_string(layer) +
            " has neither stacked routed experts nor a '" + group_name +
            "' group");
    }
    if (group_arity != static_cast<std::uint32_t>(num_experts)) {
        throw std::runtime_error(
            "kimi_k3: group '" + group_name + "' holds " +
            std::to_string(group_arity) + " experts but the config says " +
            std::to_string(num_experts));
    }
    return true;
}

}  // namespace pie_cuda_driver::model
