// The kernels live in the header; this file is the two entry points that
// still launch them ahead of time. The scalar layer comes out of the prelude:
// NVRTC has no CUDA device headers, and `add_bias.cuh` compiles under both it
// and nvcc.
#include "pie_device.cuh"
#include "norm/add_bias.cuh"
#include "norm/add_bias.hpp"


namespace pie_cuda_driver::kernels::norm {

void add_bias_bf16(
    void* out, const void* bias,
    int num_rows, int dim,
    cudaStream_t stream)
{
    if (num_rows <= 0 || dim <= 0) return;
    constexpr int BLOCK = 256;
    device::add_bias<device::bf16><<<num_rows, BLOCK, 0, stream>>>(
        static_cast<device::bf16*>(out),
        static_cast<const device::bf16*>(bias),
        dim);
}

// `add_bias_bf16_strided` was deleted here by §43. `launch_abi.rs` carried it
// as `NormNoRow::Orphaned` -- nothing calls it: not the driver, not a sibling
// `.cu`, not `lower.rs`, which chooses the contiguous `add_bias_bf16` above.
// Its stated last consumer was `sources.rs`' `<<<>>>` census, and a census is
// not a consumer.

}  // namespace pie_cuda_driver::kernels::norm
