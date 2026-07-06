// Qwen3-0.6B decoder-layer Metal kernels (parity build). Each mirrors the
// corresponding CUDA kernel formula; numeric ops use precise math (fast-math is
// disabled in the harness) and the same accumulation order as the CPU reference.
// Runtime-compiled (CLT-only box).

#include <metal_stdlib>
using namespace metal;

// ── GEMM: y[m,n] = sum_k x[m,k] * w[n,k]  (y = x · Wᵀ, W row-major [N,K]) ──────
// One thread per output element; sequential-k f32 accumulation. grid = M*N.
kernel void matmul_xwt(
    device const float* x   [[buffer(0)]],
    device const float* w   [[buffer(1)]],
    device float*       y   [[buffer(2)]],
    constant int&       M   [[buffer(3)]],
    constant int&       N   [[buffer(4)]],
    constant int&       K   [[buffer(5)]],
    constant uint&      total [[buffer(6)]],  // M*N
    uint gid [[thread_position_in_grid]]) {
    if (gid >= total) return;
    const int m = (int)(gid / (uint)N);
    const int n = (int)(gid % (uint)N);
    device const float* xr = x + (ulong)m * K;
    device const float* wr = w + (ulong)n * K;
    float acc = 0.0f;
    for (int k = 0; k < K; ++k) acc += xr[k] * wr[k];
    y[gid] = acc;
}

// ── RMSNorm: y = x * rsqrt(mean(x²)+eps) * w  (per-row over `dim`) ─────────────
// One thread per row; sequential f32 sum-of-squares. grid = rows.
kernel void rmsnorm(
    device const float* x    [[buffer(0)]],
    device const float* w    [[buffer(1)]],
    device float*       y    [[buffer(2)]],
    constant int&       rows [[buffer(3)]],
    constant int&       dim  [[buffer(4)]],
    constant float&     eps  [[buffer(5)]],
    uint r [[thread_position_in_grid]]) {
    if (r >= (uint)rows) return;
    device const float* xr = x + (ulong)r * dim;
    device float* yr = y + (ulong)r * dim;
    float acc = 0.0f;
    for (int i = 0; i < dim; ++i) { float v = xr[i]; acc += v * v; }
    float inv_rms = rsqrt(acc / (float)dim + eps);
    for (int i = 0; i < dim; ++i) yr[i] = xr[i] * inv_rms * w[i];
}

// ── RoPE (Qwen half-rotation) applied in place over [rows, n_heads, head_dim] ──
// One thread per (row, head, pair i<half). freq = theta^(-2i/head_dim).
kernel void rope_qwen(
    device float*       h         [[buffer(0)]],
    device const int*   positions [[buffer(1)]],  // [rows]
    constant int&       n_heads   [[buffer(2)]],
    constant int&       head_dim  [[buffer(3)]],
    constant float&     theta     [[buffer(4)]],
    constant uint&      total     [[buffer(5)]],  // rows * n_heads * (head_dim/2)
    uint gid [[thread_position_in_grid]]) {
    if (gid >= total) return;
    const int hhalf = head_dim / 2;
    const int i = (int)(gid % (uint)hhalf);
    const uint hg = gid / (uint)hhalf;              // row*n_heads + head
    const int head = (int)(hg % (uint)n_heads);
    const int row = (int)(hg / (uint)n_heads);
    (void)head;
    const int pos = positions[row];
    device float* base = h + (ulong)hg * head_dim;
    float freq = pow(theta, -2.0f * (float)i / (float)head_dim);
    float ang = (float)pos * freq;
    float c = cos(ang), s = sin(ang);
    float a = base[i], b = base[i + hhalf];
    base[i] = a * c - b * s;
    base[i + hhalf] = b * c + a * s;
}

// ── SwiGLU: y = silu(gate) * up, silu(g) = g/(1+e^-g). grid = n. ──────────────
kernel void swiglu(
    device const float* gate [[buffer(0)]],
    device const float* up   [[buffer(1)]],
    device float*       y    [[buffer(2)]],
    constant uint&      n    [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= n) return;
    float g = gate[gid];
    float silu = g / (1.0f + exp(-g));
    y[gid] = silu * up[gid];
}

// ── paged decode/prefill attention (same kernel; per-row causal bound) ────────
// One thread per (query_row, q_head). Online flash-softmax over kv positions
// [0, position_ids[row]] with NHD paged KV. Prefill = multi query rows per
// request, each with its own q_pos. grid = N * n_q_heads. (Mirrors kvattn.)
constant constexpr int MAX_D = 128;
kernel void paged_attention(
    device const float* queries          [[buffer(0)]],
    device const float* k_pages          [[buffer(1)]],
    device const float* v_pages          [[buffer(2)]],
    device float*       out              [[buffer(3)]],
    device const int*   position_ids     [[buffer(4)]],
    device const int*   req_of_token     [[buffer(5)]],
    device const uint*  kv_page_indices  [[buffer(6)]],
    device const uint*  kv_page_indptr   [[buffer(7)]],
    constant int&       n_q_heads        [[buffer(8)]],
    constant int&       n_kv_heads       [[buffer(9)]],
    constant int&       d                [[buffer(10)]],
    constant int&       page_size        [[buffer(11)]],
    constant int&       gqa_factor       [[buffer(12)]],
    constant float&     scale            [[buffer(13)]],
    constant uint&      total            [[buffer(14)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= total) return;
    const int row = (int)(gid / (uint)n_q_heads);
    const int qh = (int)(gid % (uint)n_q_heads);
    const int kv_head = qh / gqa_factor;
    const int r = req_of_token[row];
    const int q_pos = position_ids[row];
    const int page_base = (int)kv_page_indptr[r];
    const ulong qbase = ((ulong)row * n_q_heads + qh) * d;
    float m = -3.0e38f, l = 0.0f;
    float acc[MAX_D];
    for (int i = 0; i < d; ++i) acc[i] = 0.0f;
    for (int kp = 0; kp <= q_pos; ++kp) {
        const int page = (int)kv_page_indices[page_base + kp / page_size];
        const ulong slot = (ulong)page * page_size + (kp % page_size);
        const ulong kb = (slot * n_kv_heads + kv_head) * d;
        float score = 0.0f;
        for (int i = 0; i < d; ++i) score += queries[qbase + i] * k_pages[kb + i];
        score *= scale;
        const float nm = max(m, score);
        const float factor = exp(m - nm);
        const float e = exp(score - nm);
        l = l * factor + e;
        for (int i = 0; i < d; ++i) acc[i] = acc[i] * factor + e * v_pages[kb + i];
        m = nm;
    }
    for (int i = 0; i < d; ++i) out[qbase + i] = (l == 0.0f) ? 0.0f : acc[i] / l;
}

// Add a residual: out = a + b, grid = n.
kernel void add_inplace(
    device float*       a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    constant uint&      n [[buffer(2)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= n) return;
    a[gid] = a[gid] + b[gid];
}
