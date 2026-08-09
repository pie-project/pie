// The kernel lives in `norm/elementwise.cuh` -- this file is the one entry
// point that still launches it ahead of time, and nothing else.
//
// It used to carry its own `residual_add_bf16_kernel`, a second copy of
// `device::residual_add<T>` with the same four lines of arithmetic. Two
// copies that agree today are two kernels that drift, and each is right for
// whichever half of the tree its tests exercise: the JIT rows fired the
// template, this file fired the copy, and nothing compared them. The copy is
// gone; the launcher spells the instantiation.
#include "pie_device.cuh"
#include "norm/elementwise.cuh"
#include "norm/residual_add.hpp"


namespace pie_cuda_driver::kernels::norm {

void residual_add_bf16(
    void* y, const void* x,
    device::usize n,
    cudaStream_t stream)
{
    if (n == 0) return;
    constexpr int BLOCK = 256;
    const auto blocks = static_cast<unsigned>((n + BLOCK - 1) / BLOCK);
    device::residual_add<device::bf16><<<blocks, BLOCK, 0, stream>>>(
        static_cast<device::bf16*>(y),
        static_cast<const device::bf16*>(x),
        n);
}

}  // namespace pie_cuda_driver::kernels::norm
