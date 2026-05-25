#include "kernels/deinterleave.hpp"

#include <cuda_bf16.h>

namespace pie_cuda_driver::kernels {

namespace {

__global__ void deinterleave_rows_bf16_kernel(
    const __nv_bfloat16* __restrict__ fused,
    __nv_bfloat16* __restrict__       gate_out,
    __nv_bfloat16* __restrict__       up_out,
    int I, int H)
{
    const int row = blockIdx.x;  // 0 .. I-1
    if (row >= I) return;
    const __nv_bfloat16* gate_src = fused + (2 * row    ) * H;
    const __nv_bfloat16* up_src   = fused + (2 * row + 1) * H;
    __nv_bfloat16* gate_dst = gate_out + row * H;
    __nv_bfloat16* up_dst   = up_out   + row * H;
    for (int j = threadIdx.x; j < H; j += blockDim.x) {
        gate_dst[j] = gate_src[j];
        up_dst[j]   = up_src[j];
    }
}

__global__ void deinterleave_vec_bf16_kernel(
    const __nv_bfloat16* __restrict__ fused,
    __nv_bfloat16* __restrict__       gate_out,
    __nv_bfloat16* __restrict__       up_out,
    int I)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= I) return;
    gate_out[i] = fused[2 * i];
    up_out[i]   = fused[2 * i + 1];
}

}  // namespace

void launch_deinterleave_rows_bf16(
    const void* fused, void* gate_out, void* up_out,
    int I, int H, cudaStream_t stream)
{
    if (I <= 0 || H <= 0) return;
    const int block = (H < 128) ? 32 : (H > 256 ? 256 : 128);
    deinterleave_rows_bf16_kernel<<<I, block, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(fused),
        static_cast<__nv_bfloat16*>(gate_out),
        static_cast<__nv_bfloat16*>(up_out),
        I, H);
}

void launch_deinterleave_vec_bf16(
    const void* fused, void* gate_out, void* up_out,
    int I, cudaStream_t stream)
{
    if (I <= 0) return;
    constexpr int BLOCK = 256;
    const int grid = (I + BLOCK - 1) / BLOCK;
    deinterleave_vec_bf16_kernel<<<grid, BLOCK, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(fused),
        static_cast<__nv_bfloat16*>(gate_out),
        static_cast<__nv_bfloat16*>(up_out),
        I);
}

namespace {

__global__ void split_q_gate_bf16_kernel(
    const __nv_bfloat16* __restrict__ packed,  // [N, num_heads, 2*head_dim]
    __nv_bfloat16* __restrict__ q_out,         // [N, num_heads, head_dim]
    __nv_bfloat16* __restrict__ gate_out,      // [N, num_heads, head_dim]
    int N, int num_heads, int head_dim)
{
    const int n = blockIdx.x;
    const int h = blockIdx.y;
    if (n >= N || h >= num_heads) return;

    const int twod = 2 * head_dim;
    const __nv_bfloat16* row = packed + ((long long)n * num_heads + h) * twod;
    __nv_bfloat16* q_row     = q_out   + ((long long)n * num_heads + h) * head_dim;
    __nv_bfloat16* gate_row  = gate_out + ((long long)n * num_heads + h) * head_dim;
    for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
        q_row[i]    = row[i];
        gate_row[i] = row[head_dim + i];
    }
}

}  // namespace

void launch_split_q_gate_bf16(
    const void* packed, void* q_out, void* gate_out,
    int N, int num_heads, int head_dim, cudaStream_t stream)
{
    if (N <= 0 || num_heads <= 0 || head_dim <= 0) return;
    const int block = (head_dim < 128) ? 64 : 128;
    dim3 grid(N, num_heads);
    split_q_gate_bf16_kernel<<<grid, block, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(packed),
        static_cast<__nv_bfloat16*>(q_out),
        static_cast<__nv_bfloat16*>(gate_out),
        N, num_heads, head_dim);
}

namespace {

__global__ void concat_bf16_rows_kernel(
    const __nv_bfloat16* __restrict__ left,
    const __nv_bfloat16* __restrict__ right,
    __nv_bfloat16* __restrict__ out,
    int N, int left_dim, int right_dim)
{
    const int n = blockIdx.x;
    const int total_dim = left_dim + right_dim;
    if (n >= N) return;
    const __nv_bfloat16* l = left + (long long)n * left_dim;
    const __nv_bfloat16* r = right + (long long)n * right_dim;
    __nv_bfloat16* o = out + (long long)n * total_dim;
    for (int i = threadIdx.x; i < total_dim; i += blockDim.x) {
        o[i] = (i < left_dim) ? l[i] : r[i - left_dim];
    }
}

__global__ void split_qwen_gdn_projections_kernel(
    const __nv_bfloat16* __restrict__ qkvz,
    const __nv_bfloat16* __restrict__ ba,
    __nv_bfloat16* __restrict__ qkv_out,
    __nv_bfloat16* __restrict__ z_out,
    __nv_bfloat16* __restrict__ b_out,
    __nv_bfloat16* __restrict__ a_out,
    int conv_dim, int v_dim, int v_h)
{
    const int n = blockIdx.x;
    const int qkvz_dim = conv_dim + v_dim;
    const int ba_dim = 2 * v_h;
    const int max_dim = qkvz_dim > ba_dim ? qkvz_dim : ba_dim;

    const __nv_bfloat16* qkvz_row = qkvz + (long long)n * qkvz_dim;
    const __nv_bfloat16* ba_row = ba + (long long)n * ba_dim;
    __nv_bfloat16* qkv_row = qkv_out + (long long)n * conv_dim;
    __nv_bfloat16* z_row = z_out + (long long)n * v_dim;
    __nv_bfloat16* b_row = b_out + (long long)n * v_h;
    __nv_bfloat16* a_row = a_out + (long long)n * v_h;

    for (int i = threadIdx.x; i < max_dim; i += blockDim.x) {
        if (i < conv_dim) {
            qkv_row[i] = qkvz_row[i];
        } else if (i < qkvz_dim) {
            z_row[i - conv_dim] = qkvz_row[i];
        }
        if (i < v_h) {
            b_row[i] = ba_row[i];
        } else if (i < ba_dim) {
            a_row[i - v_h] = ba_row[i];
        }
    }
}

__global__ void repeat_interleave_heads_bf16_kernel(
    const __nv_bfloat16* __restrict__ in,
    __nv_bfloat16* __restrict__ out,
    int N, int kv_heads, int q_heads, int head_dim)
{
    const int n = blockIdx.x;
    const int qh = blockIdx.y;
    if (n >= N || qh >= q_heads) return;
    const int repeat = q_heads / kv_heads;
    const int kh = qh / repeat;
    const __nv_bfloat16* src =
        in + ((long long)n * kv_heads + kh) * head_dim;
    __nv_bfloat16* dst =
        out + ((long long)n * q_heads + qh) * head_dim;
    for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
        dst[i] = src[i];
    }
}

}  // namespace

void launch_concat_bf16_rows(
    const void* left, const void* right, void* out,
    int N, int left_dim, int right_dim, cudaStream_t stream)
{
    if (N <= 0 || left_dim <= 0 || right_dim <= 0) return;
    concat_bf16_rows_kernel<<<N, 256, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(left),
        static_cast<const __nv_bfloat16*>(right),
        static_cast<__nv_bfloat16*>(out),
        N, left_dim, right_dim);
}

void launch_split_qwen_gdn_projections_bf16(
    const void* qkvz, const void* ba,
    void* qkv_out, void* z_out, void* b_out, void* a_out,
    int N, int conv_dim, int v_dim, int v_h, cudaStream_t stream)
{
    if (N <= 0 || conv_dim <= 0 || v_dim <= 0 || v_h <= 0) return;
    split_qwen_gdn_projections_kernel<<<N, 256, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(qkvz),
        static_cast<const __nv_bfloat16*>(ba),
        static_cast<__nv_bfloat16*>(qkv_out),
        static_cast<__nv_bfloat16*>(z_out),
        static_cast<__nv_bfloat16*>(b_out),
        static_cast<__nv_bfloat16*>(a_out),
        conv_dim, v_dim, v_h);
}

void launch_repeat_interleave_heads_bf16(
    const void* in, void* out,
    int N, int kv_heads, int q_heads, int head_dim, cudaStream_t stream)
{
    if (N <= 0 || kv_heads <= 0 || q_heads <= 0 || head_dim <= 0) return;
    if (q_heads % kv_heads != 0) return;
    const int block = (head_dim < 128) ? 64 : 128;
    dim3 grid(N, q_heads);
    repeat_interleave_heads_bf16_kernel<<<grid, block, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(in),
        static_cast<__nv_bfloat16*>(out),
        N, kv_heads, q_heads, head_dim);
}

}  // namespace pie_cuda_driver::kernels
