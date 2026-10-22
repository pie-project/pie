#include "ssm/state.cuh"

namespace pie::attn {

template <class T>
__global__ void widen(
    const T* __restrict__ x, float* __restrict__ y, usize n)
{
    const usize i = blockIdx.x * (usize)blockDim.x + threadIdx.x;
    if (i < n) y[i] = Elem<T>::to_f32(x[i]);
}

template <class T>
__global__ void narrow(
    const float* __restrict__ x, T* __restrict__ y, usize n)
{
    const usize i = blockIdx.x * (usize)blockDim.x + threadIdx.x;
    if (i < n) y[i] = Elem<T>::from_f32(x[i]);
}

template <class T>
__global__ void repeat_interleave_heads_fp32(
    const T* __restrict__ in, T* __restrict__ out,
    int K_h, int V_h, int D, int repeat,
    const u32* __restrict__ win)
{
    const int n   = blockIdx.x;

    if (win != nullptr && n >= static_cast<int>(win[0])) return;
    const int h_v = blockIdx.y;
    const int d   = threadIdx.x;
    if (h_v >= V_h || d >= D) return;
    const int h_k = h_v / repeat;
    const long long src = ((long long)n * K_h + h_k) * D + d;
    const long long dst = ((long long)n * V_h + h_v) * D + d;
    if (d < D) out[dst] = in[src];

    for (int dd = d + blockDim.x; dd < D; dd += blockDim.x) {
        out[((long long)n * V_h + h_v) * D + dd] =
            in[((long long)n * K_h + h_k) * D + dd];
    }
}

template <class T, int BLOCK>
__global__ void ssm_gdn_prep_qk_norm(
    const T* __restrict__ qkv_post,
    float* __restrict__ q_out,
    float* __restrict__ k_out,
    int K_h, int K_d, int conv_dim,
    float q_scale,
    const u32* __restrict__ win)
{
    const int n = blockIdx.x;

    if (win != nullptr && n >= static_cast<int>(win[0])) return;

    const int n_row = win != nullptr ? n + static_cast<int>(win[1]) : n;
    const int h = blockIdx.y;
    const int tid = threadIdx.x;
    const int K_dim = K_h * K_d;
    const T* q_base =
        qkv_post + (long long)n_row * conv_dim + (long long)h * K_d;
    const T* k_base =
        qkv_post + (long long)n_row * conv_dim + K_dim + (long long)h * K_d;

    float q_sum = 0.f;
    float k_sum = 0.f;
    for (int i = tid; i < K_d; i += BLOCK) {
        const float qv = Elem<T>::to_f32(q_base[i]);
        const float kv = Elem<T>::to_f32(k_base[i]);
        q_sum += qv * qv;
        k_sum += kv * kv;
    }

    __shared__ float q_buf[BLOCK];
    __shared__ float k_buf[BLOCK];
    q_buf[tid] = q_sum;
    k_buf[tid] = k_sum;
    __syncthreads();
    for (int off = BLOCK / 2; off > 0; off >>= 1) {
        if (tid < off) {
            q_buf[tid] += q_buf[tid + off];
            k_buf[tid] += k_buf[tid + off];
        }
        __syncthreads();
    }

    const float q_inv = rsqrtf(q_buf[0] + 1e-6f) * q_scale;
    const float k_inv = rsqrtf(k_buf[0] + 1e-6f);
    float* q_dst = q_out + ((long long)n * K_h + h) * K_d;
    float* k_dst = k_out + ((long long)n * K_h + h) * K_d;
    for (int i = tid; i < K_d; i += BLOCK) {
        q_dst[i] = Elem<T>::to_f32(q_base[i]) * q_inv;
        k_dst[i] = Elem<T>::to_f32(k_base[i]) * k_inv;
    }
}

template <class T>
__global__ void ssm_gdn_prep_ba_gates(
    const T* __restrict__ ba,
    const float* __restrict__ A_log,
    const T* __restrict__ dt_bias,
    float* __restrict__ gates,
    int N, int V_h,
    const u32* __restrict__ win)
{
    const int t = blockIdx.x;

    if (win != nullptr && t >= static_cast<int>(win[0])) return;

    const int t_row = win != nullptr ? t + static_cast<int>(win[1]) : t;
    const int h = blockIdx.y * blockDim.x + threadIdx.x;
    if (t >= N || h >= V_h) return;

    const T* row = ba + (long long)t_row * 2 * V_h;
    const float bv = Elem<T>::to_f32(row[h]);
    const float av = Elem<T>::to_f32(row[V_h + h]);

    const float z = av + Elem<T>::to_f32(dt_bias[h]);
    const float sp = (z > 20.f) ? z : log1pf(__expf(z));

    float* out = gates + (long long)t_row * 2 * V_h;
    out[h] = -__expf(A_log[h]) * sp;
    out[V_h + h] = 1.f / (1.f + __expf(-bv));
}

template <class T, int BLOCK>
__global__ void ssm_gdn_prep_v_gates(
    const T* __restrict__ qkv_post,
    const float* __restrict__ gates,
    float* __restrict__ v_out,
    float* __restrict__ g_log_out,
    float* __restrict__ beta_out,
    int K_h, int V_h, int K_d, int V_d, int conv_dim,
    const u32* __restrict__ win)
{
    const int n = blockIdx.x;

    if (win != nullptr && n >= static_cast<int>(win[0])) return;

    const int n_row = win != nullptr ? n + static_cast<int>(win[1]) : n;
    const int h = blockIdx.y;
    const int tid = threadIdx.x;
    const int K_dim = K_h * K_d;
    const T* v_base =
        qkv_post + (long long)n_row * conv_dim + 2 * K_dim + (long long)h * V_d;
    float* v_dst = v_out + ((long long)n * V_h + h) * V_d;
    for (int i = tid; i < V_d; i += BLOCK) {
        v_dst[i] = Elem<T>::to_f32(v_base[i]);
    }

    if (tid == 0) {
        const long long gh = (long long)n * V_h + h;
        const float* row = gates + (long long)n_row * 2 * V_h;
        g_log_out[gh] = row[h];
        beta_out[gh] = row[V_h + h];
    }
}

}
