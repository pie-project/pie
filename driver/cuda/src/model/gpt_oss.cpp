#include "model/gpt_oss.hpp"

#include <stdexcept>

namespace pie_cuda_driver::model {

MixtralWeights bind_gpt_oss(Engine&) {
    // See header for the implementation roadmap. The kernels and the
    // sparse-MoE forward are in place; the missing pieces are:
    //   1. Per-layer alternating sliding/full attention dispatch.
    //   2. Sink-aware attention variant (custom flashinfer
    //      AttentionVariant or virtual sink-token injection).
    //   3. MXFP4 → bf16 staging during bind, or — better — a fused
    //      MXFP4-aware grouped-gemm path for the experts.
    throw std::runtime_error(
        "bind_gpt_oss: not yet implemented; see model/gpt_oss.hpp for the "
        "build-out plan and currently-available primitives.");
}

}  // namespace pie_cuda_driver::model
