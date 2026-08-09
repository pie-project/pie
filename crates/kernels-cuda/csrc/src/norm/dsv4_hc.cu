// The kernels live in the header; this file is the seven entry points that
// still launch them ahead of time. Three of the seven are also JIT rows in
// `kernels_cuda_new::families::norm` -- `hc_pre_postprocess`,
// `hc_head_postprocess` and `hc_rmsnorm_to_f32`, all of them `<<<N, 256>>>`,
// which is `LaunchRule::Rms`. The other four launch grids no ported rule
// produces and these launchers are the only way to reach them; `dsv4_hc.cuh`
// says which and why.
#include "pie_device.cuh"
#include "norm/dsv4_hc.cuh"
#include "norm/dsv4_hc.hpp"


namespace pie_cuda_driver::kernels::norm {

namespace {

constexpr int BLOCK = 256;

}  // namespace

void hc_pre_postprocess_bf16(
    const float* mixes,
    const float* scale,
    const float* base,
    const void* residual,
    float* post_mix,
    float* comb_mix,
    void* layer_input,
    int N,
    int hc_mult,
    int hidden_size,
    float hc_eps,
    float hc_post_alpha,
    int sinkhorn_iters,
    cudaStream_t stream)
{
    if (N <= 0) return;
    device::hc_pre_postprocess<device::bf16, BLOCK><<<N, BLOCK, 0, stream>>>(
        mixes, scale, base,
        static_cast<const device::bf16*>(residual),
        post_mix, comb_mix,
        static_cast<device::bf16*>(layer_input),
        hc_mult, hidden_size, hc_eps, hc_post_alpha, sinkhorn_iters);
}

void hc_post_bf16(
    const void* x,
    const void* residual,
    const float* post_mix,
    const float* comb_mix,
    void* out_residual,
    int N,
    int hc_mult,
    int hidden_size,
    cudaStream_t stream)
{
    const long long total = static_cast<long long>(N) * hidden_size;
    if (total <= 0 || hc_mult > device::MAX_HC_MULT) return;
    const int grid = static_cast<int>((total + BLOCK - 1) / BLOCK);
    device::hc_post<device::bf16><<<grid, BLOCK, 0, stream>>>(
        static_cast<const device::bf16*>(x),
        static_cast<const device::bf16*>(residual),
        post_mix, comb_mix,
        static_cast<device::bf16*>(out_residual),
        N, hc_mult, hidden_size);
}

void hc_head_postprocess_bf16(
    const float* mixes,
    const float* scale,
    const float* base,
    const void* residual,
    void* out,
    int N,
    int hc_mult,
    int hidden_size,
    cudaStream_t stream,
    float hc_eps)
{
    if (N <= 0) return;
    device::hc_head_postprocess<device::bf16, BLOCK><<<N, BLOCK, 0, stream>>>(
        mixes, scale, base,
        static_cast<const device::bf16*>(residual),
        static_cast<device::bf16*>(out),
        hc_mult, hidden_size, hc_eps);
}

void hc_expand_bf16(
    const void* input,
    void* output,
    int N,
    int hc_mult,
    int hidden_size,
    cudaStream_t stream)
{
    const long long total = static_cast<long long>(N) * hidden_size;
    if (total <= 0) return;
    const int grid = static_cast<int>((total + BLOCK - 1) / BLOCK);
    device::hc_expand<device::bf16><<<grid, BLOCK, 0, stream>>>(
        static_cast<const device::bf16*>(input),
        static_cast<device::bf16*>(output),
        N, hc_mult, hidden_size);
}

void hc_rmsnorm_to_f32(
    const void* input,
    float* output,
    int N,
    int dim,
    float eps,
    cudaStream_t stream)
{
    if (N <= 0) return;
    device::hc_rmsnorm_to_f32<device::bf16, BLOCK><<<N, BLOCK, 0, stream>>>(
        static_cast<const device::bf16*>(input),
        output, dim, eps);
}

void attn_sink_correction_bf16(
    void* attn_out,
    const float* lse,
    const float* sink,
    int N, int num_heads, int head_dim,
    cudaStream_t stream)
{
    if (N <= 0 || !sink) return;
    dim3 grid(N, num_heads);
    dim3 block(BLOCK);
    device::attn_sink_correction<device::bf16><<<grid, block, 0, stream>>>(
        static_cast<device::bf16*>(attn_out),
        lse, sink, num_heads, head_dim);
}

void per_head_rmsnorm_bf16(
    void* q, int N, int num_heads, int head_dim,
    float eps, cudaStream_t stream)
{
    if (N <= 0 || num_heads <= 0 || head_dim <= 0) return;
    dim3 grid(N, num_heads);
    dim3 block(BLOCK);
    device::per_head_rmsnorm<device::bf16><<<grid, block, 0, stream>>>(
        static_cast<device::bf16*>(q), head_dim, eps);
}

}  // namespace pie_cuda_driver::kernels::norm
