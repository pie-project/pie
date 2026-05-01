#include "model/qwen3_5.hpp"

#include <stdexcept>

namespace pie_cuda_driver::model {

Qwen3Weights bind_qwen3_5(Engine&) {
    throw std::runtime_error(
        "bind_qwen3_5: not yet implemented; see model/qwen3_5.hpp. The "
        "blocker is a GatedDeltaNet (linear-attention SSM) CUDA kernel "
        "with no flashinfer counterpart.");
}

}  // namespace pie_cuda_driver::model
