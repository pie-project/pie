//===-- quant_bf16_to_fp8.cu - the ahead-of-time entry points ------------===//
//
// Fourteen launchers and no device text. Every `__global__` this file fires
// lives in `quant_bf16_to_fp8.cuh`, which the JIT compiles from the same
// bytes -- see the header for why the split exists, why twelve kernels became
// nine templates, what each `<<<>>>` became, and which three have no launch
// rule.
//
// `model-loader` calls `quantize_bf16_to_fp8_e4m3_per_tensor`,
// `quantize_bf16_to_fp8_e4m3_per_channel` and their int8 siblings by name
// from Rust, so none of these entry points is going away.
//
//===----------------------------------------------------------------------===//
#include "quant/quant_bf16_to_fp8.hpp"

#include "quant/quant_bf16_to_fp8.cuh"
#include "cuda_check.hpp"

namespace pie_cuda_driver::kernels::quant {

namespace {

constexpr int BLOCK = 256;

// `LaunchRule::Rms` requests exactly this: one float per warp of a 256-wide
// block. The rule and this expression are the same arithmetic, stated in the
// two places that must agree while both paths run.
constexpr std::size_t ROW_REDUCE_SHMEM = (BLOCK / 32) * sizeof(float);

}  // namespace

void launch_absmax_bf16(
    const void* W_bf16, float* absmax_dev,
    std::size_t n, cudaStream_t stream)
{
    if (n == 0) return;
    CUDA_CHECK(cudaMemsetAsync(absmax_dev, 0, sizeof(float), stream));
    // Cap grid at 1024 blocks — enough parallelism for >256k elements
    // and keeps the atomic contention bounded.
    const unsigned blocks_full = static_cast<unsigned>((n + BLOCK - 1) / BLOCK);
    const unsigned blocks = blocks_full < 1024u ? blocks_full : 1024u;
    device::absmax_bf16<<<blocks, BLOCK, 0, stream>>>(
        static_cast<const device::bf16*>(W_bf16), absmax_dev, n);
}

void launch_quant_bf16_to_fp8_e4m3(
    const void* W_bf16, device::u8* W_fp8,
    float scale_inv, std::size_t n, cudaStream_t stream)
{
    if (n == 0) return;
    const auto blocks = static_cast<unsigned>((n + BLOCK - 1) / BLOCK);
    device::quant_flat<device::fp8_e4m3><<<blocks, BLOCK, 0, stream>>>(
        static_cast<const device::bf16*>(W_bf16), W_fp8, scale_inv, n);
}

void quantize_bf16_to_fp8_e4m3_per_channel(
    const void* W_bf16, device::u8* W_fp8,
    float* scale_inv_dev, int rows, int cols, cudaStream_t stream)
{
    if (rows == 0 || cols == 0) return;
    device::quant_per_channel<device::fp8_e4m3>
        <<<rows, BLOCK, ROW_REDUCE_SHMEM, stream>>>(
            static_cast<const device::bf16*>(W_bf16),
            W_fp8, scale_inv_dev, cols);
}

void quantize_bf16_to_int8_per_channel(
    const void* W_bf16, device::i8* W_int8,
    float* scale_inv_dev, int rows, int cols, cudaStream_t stream)
{
    if (rows == 0 || cols == 0) return;
    device::quant_per_channel<device::int8_sym>
        <<<rows, BLOCK, ROW_REDUCE_SHMEM, stream>>>(
            static_cast<const device::bf16*>(W_bf16),
            W_int8, scale_inv_dev, cols);
}

// Per-token activation INT8 quant is mathematically the same op as
// per-channel weight INT8 quant: per-row symmetric absmax over a 2-D
// row-major buffer, producing one scale_inv per row. Reuse the same
// kernel — only the semantic naming differs.
void quantize_bf16_to_int8_per_token(
    const void* act_bf16, device::i8* act_int8,
    float* act_scale_inv, int n_tokens, int k, cudaStream_t stream)
{
    quantize_bf16_to_int8_per_channel(
        act_bf16, act_int8, act_scale_inv, n_tokens, k, stream);
}

void launch_absmax_to_scale_inv_int8(
    float* absmax_inout, int rows, cudaStream_t stream)
{
    if (rows == 0) return;
    const int blocks = (rows + BLOCK - 1) / BLOCK;
    device::absmax_to_scale_inv<device::int8_sym><<<blocks, BLOCK, 0, stream>>>(
        absmax_inout, rows);
}

void launch_cast_bf16_to_int8_per_channel(
    const void* W_bf16, device::i8* W_int8,
    const float* scale_inv_dev, int rows, int cols, cudaStream_t stream)
{
    if (rows == 0 || cols == 0) return;
    device::cast_per_channel<device::int8_sym><<<rows, BLOCK, 0, stream>>>(
        static_cast<const device::bf16*>(W_bf16),
        W_int8, scale_inv_dev, cols);
}

void launch_dequant_int8_to_bf16_per_channel(
    const device::i8* W_int8, void* W_bf16,
    const float* scale_inv_dev, int rows, int cols, cudaStream_t stream)
{
    if (rows == 0 || cols == 0) return;
    const std::size_t n =
        static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols);
    const auto blocks = static_cast<unsigned>((n + BLOCK - 1) / BLOCK);
    device::dequant_int8_per_channel<device::bf16><<<blocks, BLOCK, 0, stream>>>(
        W_int8, static_cast<device::bf16*>(W_bf16), scale_inv_dev, cols, n);
}

void dequant_int32_w8a8_to_bf16(
    const device::i32* acc_int32, const float* act_scale_inv,
    const float* w_scale_inv, void* out_bf16,
    int M, int N, cudaStream_t stream)
{
    if (M == 0 || N == 0) return;
    constexpr int BX = 32, BY = 8;
    const dim3 block(BX, BY);
    const dim3 grid((N + BX - 1) / BX, (M + BY - 1) / BY);
    device::w8a8_dequant<<<grid, block, 0, stream>>>(
        acc_int32, act_scale_inv, w_scale_inv,
        static_cast<device::bf16*>(out_bf16), M, N);
}

void launch_absmax_per_row_bf16(
    const void* W_bf16, float* absmax_dev,
    int rows, int cols, cudaStream_t stream)
{
    if (rows == 0 || cols == 0) return;
    device::absmax_per_row<device::bf16>
        <<<rows, BLOCK, ROW_REDUCE_SHMEM, stream>>>(
            static_cast<const device::bf16*>(W_bf16), absmax_dev, cols);
}

void launch_absmax_to_scale_inv(
    float* absmax_inout, int rows, cudaStream_t stream)
{
    if (rows == 0) return;
    const auto blocks = static_cast<unsigned>((rows + BLOCK - 1) / BLOCK);
    device::absmax_to_scale_inv<device::fp8_e4m3><<<blocks, BLOCK, 0, stream>>>(
        absmax_inout, rows);
}

void launch_cast_bf16_to_fp8_e4m3_per_channel(
    const void* W_bf16, device::u8* W_fp8,
    const float* scale_inv_dev, int rows, int cols, cudaStream_t stream)
{
    if (rows == 0 || cols == 0) return;
    device::cast_per_channel<device::fp8_e4m3><<<rows, BLOCK, 0, stream>>>(
        static_cast<const device::bf16*>(W_bf16),
        W_fp8, scale_inv_dev, cols);
}

float quantize_bf16_to_fp8_e4m3_per_tensor(
    const void* W_bf16, device::u8* W_fp8,
    std::size_t n, cudaStream_t stream)
{
    if (n == 0) return 1.f;
    // 1) absmax → tmp scalar
    float* tmp = nullptr;
    CUDA_CHECK(cudaMalloc(&tmp, sizeof(float)));
    launch_absmax_bf16(W_bf16, tmp, n, stream);

    // 2) Pull absmax to host (one sync per quant call — load-time
    // operation, not the hot path).
    float absmax = 0.f;
    CUDA_CHECK(cudaMemcpyAsync(&absmax, tmp, sizeof(float),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaFree(tmp));

    // Degenerate weights (all-zero) — pick scale=1.0 and let the cast
    // produce zeros. Returns weight_scale_inv = 1.0 (caller stores this
    // in the QuantMeta scale tensor; cuBLASLt treats it as a no-op).
    if (absmax == 0.f) {
        launch_quant_bf16_to_fp8_e4m3(W_bf16, W_fp8, 1.f, n, stream);
        return 1.f;
    }

    // We pick `weight_scale_inv` such that bf16 ≈ fp8 * weight_scale_inv,
    // i.e. weight_scale_inv = absmax / fp8_max. The cast multiplies by
    // the reciprocal:  fp8 = round(bf16 * (fp8_max / absmax)).
    //
    // The saturation point comes from the format tag rather than a local
    // constant, so the host's scale and the device's clamp cannot disagree.
    const float fp8_max = device::fp8_e4m3::max_abs();
    const float weight_scale_inv = absmax / fp8_max;
    const float scale_inv        = fp8_max / absmax;
    launch_quant_bf16_to_fp8_e4m3(W_bf16, W_fp8, scale_inv, n, stream);
    return weight_scale_inv;
}

void quantize_bf16_to_fp8_e4m3_per_token_group(
    const void*    act_bf16,
    device::u8*  act_fp8,
    float*         act_scale,
    int            m,
    int            k,
    int            group_size,
    cudaStream_t   stream)
{
    if (m <= 0 || k <= 0 || group_size <= 0) return;
    const int n_groups = (k + group_size - 1) / group_size;
    const dim3 grid(static_cast<unsigned>(n_groups), static_cast<unsigned>(m));
    device::quant_act_fp8_per_group<<<grid, 128, 0, stream>>>(
        static_cast<const device::bf16*>(act_bf16),
        act_fp8, act_scale, m, k, group_size, n_groups);
    CUDA_CHECK(cudaGetLastError());
}

}  // namespace pie_cuda_driver::kernels::quant
