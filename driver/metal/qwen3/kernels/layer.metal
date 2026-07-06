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
