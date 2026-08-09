// Host launchers for the short causal convolution. The `__global__`
// templates they fire live in `ssm/causal_conv1d.cuh` and are defined ONCE,
// there: this file includes that header rather than carrying a second copy,
// so the archive nvcc builds and the cubin NVRTC builds are the same text.
// `new-horizon.md` §10.10 records what the copy costs — `norm/altup_aux`
// shipped two definitions for a release, each right for whichever half its
// tests exercised.
//
// Every launcher below stays. `causal_conv1d_prefill_noact_bf16` is Gemma-4's
// audio lconv1d — `vision/gemma4_audio.cu` calls it across family lines — and
// four of the five kernels have no `LaunchRule` that states their geometry, so
// the ahead-of-time path is still the only path that fires them.
#include "pie_device.cuh"
#include "ssm/causal_conv1d.cuh"
#include "ssm/causal_conv1d.hpp"


namespace pie_cuda_driver::kernels::ssm {

// `causal_conv1d_prefill_bf16`, `causal_conv1d_prefill_noact_bf16` and
// `causal_conv1d_update_bf16` were deleted here by §43, and the
// `prefill_dispatch<SILU>` template that existed only for the first two went
// with them. `families::ssm` §28.9 had already measured all three as second
// names for jobs the `_batched` launchers below do; §41 measured the rest of
// the consumer set and found it empty. `driver-cuda/csrc/vision/gemma4_audio.cu`
// used to call the noact form and now launches the template itself, which is
// the last hold released.

void causal_conv1d_update_batched_bf16(
    const void* x, const void* weight, const void* bias,
    void* state_base,
    const device::i32* slot_ids,
    long long slot_stride_elems,
    void* y,
    int R, int C, int K, cudaStream_t stream)
{
    if (R <= 0 || C <= 0 || K <= 0) return;
    constexpr int BLOCK = 128;
    dim3 grid((C + BLOCK - 1) / BLOCK, R);
    dim3 block(BLOCK);
    device::causal_conv1d_update_batched<device::bf16><<<grid, block, 0, stream>>>(
        static_cast<const device::bf16*>(x),
        static_cast<const device::bf16*>(weight),
        static_cast<const device::bf16*>(bias),
        static_cast<device::bf16*>(state_base),
        slot_ids,
        slot_stride_elems,
        static_cast<device::bf16*>(y),
        R, C, K);
}

void causal_conv1d_prefill_batched_bf16(
    const void* x, const void* weight, const void* bias,
    void* y, void* state_out_base,
    const device::i32* slot_ids,
    const device::u32* qo_indptr,
    long long slot_stride_elems,
    int R, int C, int K, cudaStream_t stream, bool write_state,
    const int* commit_len,
    const device::u8* write_state_mask)
{
    if (R <= 0 || C <= 0 || K <= 0) return;
    if (R >= 8) {
        constexpr int TILE = 128;
        dim3 grid((C + TILE - 1) / TILE, R);
        dim3 block(TILE);
        device::causal_conv1d_prefill_batched_channel_tile<device::bf16><<<grid, block, 0, stream>>>(
            static_cast<const device::bf16*>(x),
            static_cast<const device::bf16*>(weight),
            static_cast<const device::bf16*>(bias),
            static_cast<device::bf16*>(y),
            static_cast<device::bf16*>(state_out_base),
            slot_ids, qo_indptr,
            slot_stride_elems,
            C, K, write_state, write_state_mask, commit_len);
        return;
    }
    constexpr int BLOCK = 64;
    dim3 grid(C, R);
    dim3 block(BLOCK);
    device::causal_conv1d_prefill_batched<device::bf16><<<grid, block, 0, stream>>>(
        static_cast<const device::bf16*>(x),
        static_cast<const device::bf16*>(weight),
        static_cast<const device::bf16*>(bias),
        static_cast<device::bf16*>(y),
        static_cast<device::bf16*>(state_out_base),
        slot_ids, qo_indptr,
        slot_stride_elems,
        C, K, write_state, write_state_mask, commit_len);
}

}  // namespace pie_cuda_driver::kernels::ssm
