#include "batch/logits.hpp"

#include "batch/forward.hpp"
#include "device_buffer.hpp"
#include "kernels/dtype_cast.hpp"
#include "kernels/gather_rows.hpp"
#include "model/workspace.hpp"
#include "ops/gemm.hpp"

namespace pie_cuda_driver {

const float* gather_selected_logits_f32(
    BatchEngine& engine, int num_sampling, std::uint32_t vocab)
{
    const std::size_t n_conv =
        static_cast<std::size_t>(num_sampling) * vocab;
    if (n_conv == 0) return nullptr;

    if (engine.ptir_logits_bf16.size() < n_conv) {
        engine.ptir_logits_bf16 = DeviceBuffer<std::uint16_t>::alloc(n_conv);
    }
    if (engine.ptir_logits_f32.size() < n_conv) {
        engine.ptir_logits_f32 = DeviceBuffer<float>::alloc(n_conv);
    }
    kernels::launch_gather_bf16_rows(
        static_cast<const std::uint16_t*>(engine.ws.logits.data()),
        engine.inputs.sample_idx.data(),
        engine.ptir_logits_bf16.data(),
        num_sampling,
        static_cast<int>(vocab),
        engine.cublas.stream());
    kernels::launch_cast_bf16_to_fp32(
        engine.ptir_logits_bf16.data(),
        engine.ptir_logits_f32.data(),
        n_conv,
        engine.cublas.stream());
    return engine.ptir_logits_f32.data();
}

}  // namespace pie_cuda_driver
