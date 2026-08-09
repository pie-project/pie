//===-- deinterleave.cu - the packed-bank launchers ------------------===//
//
// Seven host launchers and not one `__global__`: the device text is in
// `layout/deinterleave.cuh`, which this file includes so the archive and the
// JIT header set hold the SAME definition rather than two that drift.
//
// Five of the seven kernels are also rows in
// `kernels-cuda-new/src/families/layout.rs`. The launchers stay anyway --
// this migration extracts device text and adds rows, it deletes nothing, and
// `new-horizon.md` §10.10 fixes that order so the two paths can be measured
// against each other.
//
//===----------------------------------------------------------------------===//

// The scalar layer and the fixed-width integer names, out of the prelude.
#include "pie_device.cuh"
#include "layout/deinterleave.hpp"

// The `__global__`s these launchers fire. ONE definition of each.
#include "layout/deinterleave.cuh"

namespace pie_cuda_driver::kernels::layout {

void deinterleave_rows_bf16(
    const void* fused, void* gate_out, void* up_out,
    int I, int H, cudaStream_t stream)
{
    if (I <= 0 || H <= 0) return;
    const int block = (H < 128) ? 32 : (H > 256 ? 256 : 128);
    device::deinterleave_rows<device::bf16><<<I, block, 0, stream>>>(
        static_cast<const device::bf16*>(fused),
        static_cast<device::bf16*>(gate_out),
        static_cast<device::bf16*>(up_out),
        H);
}

void deinterleave_vec_bf16(
    const void* fused, void* gate_out, void* up_out,
    int I, cudaStream_t stream)
{
    if (I <= 0) return;
    constexpr int BLOCK = 256;
    const int grid = (I + BLOCK - 1) / BLOCK;
    device::deinterleave_vec<device::bf16><<<grid, BLOCK, 0, stream>>>(
        static_cast<const device::bf16*>(fused),
        static_cast<device::bf16*>(gate_out),
        static_cast<device::bf16*>(up_out),
        I);
}

void split_q_gate_bf16(
    const void* packed, void* q_out, void* gate_out,
    int N, int num_heads, int head_dim, cudaStream_t stream)
{
    if (N <= 0 || num_heads <= 0 || head_dim <= 0) return;
    const int block = (head_dim < 128) ? 64 : 128;
    dim3 grid(N, num_heads);
    device::split_q_gate<device::bf16><<<grid, block, 0, stream>>>(
        static_cast<const device::bf16*>(packed),
        static_cast<device::bf16*>(q_out),
        static_cast<device::bf16*>(gate_out),
        N, num_heads, head_dim);
}

void concat_bf16_rows(
    const void* left, const void* right, void* out,
    int N, int left_dim, int right_dim, cudaStream_t stream)
{
    if (N <= 0 || left_dim <= 0 || right_dim <= 0) return;
    device::concat_rows<device::bf16><<<N, 256, 0, stream>>>(
        static_cast<const device::bf16*>(left),
        static_cast<const device::bf16*>(right),
        static_cast<device::bf16*>(out),
        left_dim, right_dim);
}

// `split_bf16_rows`, `split_qwen_gdn_ba_bf16` and
// `repeat_interleave_heads_bf16` were deleted here by §43. The first two are
// still jobs -- `device.rs`'s `JIT_DISPATCHED` routes both rows to NVRTC out
// of `layout/deinterleave.cuh`, so what went is the ahead-of-time launcher
// and not the kernel: §10.10 step 5, the launcher goes AFTER the row moves.
// The third had no row and no caller in any language, which the `.cuh` said
// in prose long before the audit measured it.

}  // namespace pie_cuda_driver::kernels::layout
