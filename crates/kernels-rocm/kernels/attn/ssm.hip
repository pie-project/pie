#pragma once

// Gated DeltaNet (the linear-attention mixer of Qwen 3.5 / Qwen3-Next): the
// causal depthwise conv in front of it, the gates, and the delta-rule
// recurrence over a per-slot state; and Kimi Delta Attention, the same
// recurrence with a decay per key channel (at the end).
//
// The recurrent state of value head h is S[i][j] (i over the key head, j
// over the value head), bf16 at slab + slot·stride + h·K_d·V_d + i·V_d + j.
// One token, with k̂ and q̂ the l2-normed key and query (q̂ also scaled by
// 1/√K_d), g = exp(g_log) and β:
//
//   kv[j]   = Σ_i S[i][j]·g·k̂[i]
//   δ[j]    = (v[j] − kv[j])·β
//   S'[i][j] = bf16(S[i][j]·g) + k̂[i]·δ[j]      (stored in bf16)
//   o[j]    = Σ_i S'[i][j]·q̂[i]                (from S' before its rounding)
//
// The conv state of a slot holds the last (K−1)·dil + 1 inputs, oldest
// first, row-major over the C channels.
//
// A captured launch reads its extent from the window {rows, row_offset,
// lanes, lane_offset}: per-token kernels gate on rows, per-request kernels on
// lanes, and a request's slot, mask and lengths are read at lane_offset + r.

#include "prelude/device.hip"
#include "prelude/mfma.hip"

namespace pie::attn {

using f32x4 = float __attribute__((ext_vector_type(4)));
using f32x2 = float __attribute__((ext_vector_type(2)));

__device__ __forceinline__ float silu(float z) { return z / (1.f + __expf(-z)); }

__device__ __forceinline__ bool folds(bool write_state, const u8* mask, int r) {
    return write_state && (mask == nullptr || mask[r] != 0);
}

// `folds` through a scalar load of the aligned word that holds the byte
// (a uniform byte load is a vector load, and waiting on it waits on every
// load issued before it): for a row a block, read where the stores need it.
__device__ __forceinline__ bool folds_uniform(bool write_state, const u8* mask, int r) {
    if (!write_state) return false;
    if (mask == nullptr) return true;
    const int off = static_cast<int>((reinterpret_cast<usize>(mask) + static_cast<usize>(r)) & 3);
    const u32 word = *reinterpret_cast<const u32*>(mask + r - off);
    return ((word >> (8 * off)) & 0xffu) != 0;
}

template <bool SILU, bool RESIDUAL>
__device__ __forceinline__ float conv_out(float acc, float x) {
    if constexpr (RESIDUAL) return acc + x;
    if constexpr (SILU) return silu(acc);
    return acc;
}

// One request's span of rows after its begin offset and commit cap.
struct Span {
    int t0;
    int n;
};

__device__ __forceinline__ Span span_of(const u32* indptr, int r, int rl, const int* commit_len,
                                        const int* begin_at) {
    Span s{static_cast<int>(indptr[r]), static_cast<int>(indptr[r + 1] - indptr[r])};
    if (begin_at != nullptr) {
        const int b = min(begin_at[rl], s.n);
        if (b > 0) {
            s.t0 += b;
            s.n -= b;
        }
    }
    if (commit_len != nullptr) s.n = min(s.n, commit_len[rl]);
    return s;
}

// ------------------------------------------------------------------ conv --

// Decode: each row r convolves its new input with its slot's window, then
// (when it folds) shifts the window by one. A thread owns one channel, so a
// wavefront's reads of x, of each window row and of the taps are adjacent,
// and every read of the step is issued before the first is used (KMAX taps
// at most; a wider conv takes the loop).
template <class T, bool SILU = true, bool RESIDUAL = false>
__global__ void ssm_conv1d_update(const T* __restrict__ x, const T* __restrict__ weight,
                                  const T* __restrict__ bias, T* __restrict__ state_base,
                                  const i32* __restrict__ slot_ids, long long slot_stride,
                                  T* __restrict__ y, int R, int C, int K, int dil, bool write_state,
                                  const u8* __restrict__ mask, const u32* __restrict__ win) {
    constexpr int KMAX = 8;
    const int r = blockIdx.y;
    if (r >= R || (win != nullptr && r >= static_cast<int>(win[0]))) return;
    const long long row = win != nullptr ? r + static_cast<long long>(win[1]) : r;
    const int rl = win != nullptr ? r + static_cast<int>(win[3]) : r;
    const int slot = slot_ids[rl];
    if (slot < 0) return;
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    T* state = state_base + static_cast<long long>(slot) * slot_stride;
    const int span = (K - 1) * dil;
    const bool fold = folds(write_state, mask, rl);
    const float xn = to_f32(x[row * C + c]);
    float acc = bias != nullptr ? to_f32(bias[c]) : 0.f;
    if (K <= KMAX && dil == 1) {
        float wk[KMAX], hist[KMAX];
#pragma unroll
        for (int k = 0; k < KMAX; ++k) {
            wk[k] = k < K ? to_f32(weight[c * K + k]) : 0.f;
            hist[k] = k + 1 <= span ? to_f32(state[(k + 1) * C + c]) : 0.f;
        }
#pragma unroll
        for (int k = 0; k < KMAX - 1; ++k)
            if (k < K - 1) acc += wk[k] * hist[k];
        acc += wk[K - 1 < KMAX ? K - 1 : 0] * xn;
        y[row * C + c] = from_f32<T>(conv_out<SILU, RESIDUAL>(acc, xn));
        if (fold) {
            // The window shifted by one: row k takes row k + 1's input.
#pragma unroll
            for (int k = 0; k < KMAX; ++k)
                if (k < span) state[k * C + c] = from_f32<T>(hist[k]);
            state[span * C + c] = from_f32<T>(xn);
        }
        return;
    }
    for (int k = 0; k < K - 1; ++k) acc += to_f32(weight[c * K + k]) * to_f32(state[(k * dil + 1) * C + c]);
    acc += to_f32(weight[c * K + K - 1]) * xn;
    y[row * C + c] = from_f32<T>(conv_out<SILU, RESIDUAL>(acc, xn));
    if (fold) {
        for (int k = 0; k < span; ++k) state[k * C + c] = state[(k + 1) * C + c];
        state[span * C + c] = from_f32<T>(xn);
    }
}

constexpr int kConvRows = 32;

// Prefill: grid (⌈C / block⌉, ⌈rows / 32⌉); a block convolves 32 rows of its
// channels, whichever requests they belong to. Inputs before a request's
// first row come from its slot's window. The window is not written here;
// `ssm_conv1d_state_commit` folds it after.
template <class T, bool SILU = true, bool RESIDUAL = false>
__global__ void ssm_conv1d_chunked(const T* __restrict__ x, const T* __restrict__ weight,
                                   const T* __restrict__ bias, T* __restrict__ y,
                                   const T* __restrict__ state_base, const i32* __restrict__ slot_ids,
                                   const u32* __restrict__ indptr, long long slot_stride, int C,
                                   int K, int dil, bool write_state, const u8* __restrict__ mask,
                                   const int* __restrict__ commit_len, const int* __restrict__ begin_at,
                                   const u32* __restrict__ win, int lanes_stated) {
    (void)write_state;
    (void)mask;
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    const int lanes = win != nullptr ? static_cast<int>(win[2]) : lanes_stated;
    if (lanes <= 0) return;
    const int total = static_cast<int>(indptr[lanes]);
    const int tile0 = blockIdx.y * kConvRows;
    if (tile0 >= total) return;
    const int tile_end = min(tile0 + kConvRows, total);
    const long long row0 = win != nullptr ? static_cast<long long>(win[1]) : 0;
    const int window = (K - 1) * dil + 1;
    const float b = bias != nullptr ? to_f32(bias[c]) : 0.f;

    // The first request whose rows reach into this tile.
    int lo = 0, hi = lanes - 1;
    while (lo < hi) {
        const int mid = (lo + hi + 1) / 2;
        if (static_cast<int>(indptr[mid]) <= tile0) lo = mid; else hi = mid - 1;
    }
    for (int r = lo; r < lanes; ++r) {
        if (static_cast<int>(indptr[r]) >= tile_end) break;
        const int rl = win != nullptr ? r + static_cast<int>(win[3]) : r;
        const Span s = span_of(indptr, r, rl, commit_len, begin_at);
        if (s.n <= 0) continue;
        const int slot = slot_ids[rl];
        if (slot < 0) continue;
        const T* xr = x + (s.t0 + row0) * C;
        T* yr = y + (s.t0 + row0) * C;
        const T* state = state_base + static_cast<long long>(slot) * slot_stride;
        for (int t = max(tile0, s.t0) - s.t0; t < min(tile_end, s.t0 + s.n) - s.t0; ++t) {
            float acc = b;
            for (int k = 0; k < K; ++k) {
                const int src = t - (K - 1 - k) * dil;
                const float xv = src < 0 ? to_f32(state[(window + src) * C + c])
                                         : to_f32(xr[static_cast<long long>(src) * C + c]);
                acc += to_f32(weight[c * K + k]) * xv;
            }
            yr[static_cast<long long>(t) * C + c] = from_f32<T>(conv_out<SILU, RESIDUAL>(acc, to_f32(xr[static_cast<long long>(t) * C + c])));
        }
    }
}

// The prefill conv at dilation 1 and K taps, eight channels a thread: each
// input row is read once as one 16-byte load and slides through a window of
// K rows in registers, where the per-channel walk above reads every input K
// times, two bytes a lane. The same products in the same order, so the same
// bits. Grid (⌈C / (8·block)⌉, ⌈rows / 32⌉); C a multiple of 8.
template <class T, int K, bool SILU = true, bool RESIDUAL = false>
__global__ void ssm_conv1d_chunked_v8(const T* __restrict__ x, const T* __restrict__ weight,
                                      const T* __restrict__ bias, T* __restrict__ y,
                                      const T* __restrict__ state_base, const i32* __restrict__ slot_ids,
                                      const u32* __restrict__ indptr, long long slot_stride, int C,
                                      const int* __restrict__ commit_len, const int* __restrict__ begin_at,
                                      const u32* __restrict__ win, int lanes_stated) {
    constexpr int V = 8;
    const int c0 = (blockIdx.x * blockDim.x + threadIdx.x) * V;
    if (c0 >= C) return;
    const int lanes = win != nullptr ? static_cast<int>(win[2]) : lanes_stated;
    if (lanes <= 0) return;
    const int total = static_cast<int>(indptr[lanes]);
    const int tile0 = blockIdx.y * kConvRows;
    if (tile0 >= total) return;
    const int tile_end = min(tile0 + kConvRows, total);
    const long long row0 = win != nullptr ? static_cast<long long>(win[1]) : 0;
    float w[V][K], b[V];
#pragma unroll
    for (int v = 0; v < V; ++v) {
        b[v] = bias != nullptr ? to_f32(bias[c0 + v]) : 0.f;
#pragma unroll
        for (int k = 0; k < K; ++k) w[v][k] = to_f32(weight[(c0 + v) * K + k]);
    }
    auto load = [](const T* at, float (&out)[V]) {
        const Pack8<T> p = *reinterpret_cast<const Pack8<T>*>(at);
#pragma unroll
        for (int v = 0; v < V; ++v) out[v] = to_f32(p.v[v]);
    };

    int lo = 0, hi = lanes - 1;
    while (lo < hi) {
        const int mid = (lo + hi + 1) / 2;
        if (static_cast<int>(indptr[mid]) <= tile0) lo = mid; else hi = mid - 1;
    }
    for (int r = lo; r < lanes; ++r) {
        if (static_cast<int>(indptr[r]) >= tile_end) break;
        const int rl = win != nullptr ? r + static_cast<int>(win[3]) : r;
        const Span s = span_of(indptr, r, rl, commit_len, begin_at);
        if (s.n <= 0) continue;
        const int slot = slot_ids[rl];
        if (slot < 0) continue;
        const T* xr = x + (s.t0 + row0) * C + c0;
        T* yr = y + (s.t0 + row0) * C + c0;
        const T* state = state_base + static_cast<long long>(slot) * slot_stride + c0;
        const int t_begin = max(tile0, s.t0) - s.t0, t_end = min(tile_end, s.t0 + s.n) - s.t0;
        if (t_begin >= t_end) continue;
        // win_[j] holds input t - (K - 1) + j.
        float xw[K][V];
#pragma unroll
        for (int j = 0; j < K - 1; ++j) {
            const int src = t_begin - (K - 1) + j;
            load(src < 0 ? state + static_cast<long long>(K + src) * C : xr + static_cast<long long>(src) * C, xw[j]);
        }
        for (int t = t_begin; t < t_end; ++t) {
            load(xr + static_cast<long long>(t) * C, xw[K - 1]);
            Pack8<T> out;
#pragma unroll
            for (int v = 0; v < V; ++v) {
                float acc = b[v];
#pragma unroll
                for (int k = 0; k < K; ++k) acc += w[v][k] * xw[k][v];
                out.v[v] = from_f32<T>(conv_out<SILU, RESIDUAL>(acc, xw[K - 1][v]));
            }
            *reinterpret_cast<Pack8<T>*>(yr + static_cast<long long>(t) * C) = out;
#pragma unroll
            for (int j = 0; j < K - 1; ++j)
#pragma unroll
                for (int v = 0; v < V; ++v) xw[j][v] = xw[j + 1][v];
        }
    }
}

// The window after a request's rows: its last (K−1)·dil + 1 inputs, the
// older ones from the window it started with. Grid (⌈C / block⌉, lanes).
template <class T>
__global__ void ssm_conv1d_state_commit(const T* __restrict__ x, T* __restrict__ state_base,
                                        const i32* __restrict__ slot_ids,
                                        const u32* __restrict__ indptr, long long slot_stride, int C,
                                        int K, int dil, const u8* __restrict__ mask,
                                        const int* __restrict__ commit_len,
                                        const int* __restrict__ begin_at, const u32* __restrict__ win,
                                        int lanes_stated) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    const int lanes = win != nullptr ? static_cast<int>(win[2]) : lanes_stated;
    const int r = blockIdx.y;
    if (r >= lanes) return;
    const int rl = win != nullptr ? r + static_cast<int>(win[3]) : r;
    if (mask != nullptr && mask[rl] == 0) return;
    const int slot = slot_ids[rl];
    if (slot < 0) return;
    const Span s = span_of(indptr, r, rl, commit_len, begin_at);
    if (s.n <= 0) return;
    const long long row0 = win != nullptr ? static_cast<long long>(win[1]) : 0;
    const int window = (K - 1) * dil + 1;
    const T* xr = x + (s.t0 + row0) * C;
    T* state = state_base + static_cast<long long>(slot) * slot_stride;
    // Ascending s reads only rows at or after the one it overwrites.
    for (int i = 0; i < window; ++i) {
        const int src = s.n - window + i;
        state[i * C + c] = src < 0 ? state[(window + src) * C + c] : xr[static_cast<long long>(src) * C + c];
    }
}

// ----------------------------------------------------------------- gates --

// [b | a] → [g_log | β]: g_log = −exp(A_log)·softplus(a + dt_bias),
// β = sigmoid(b). Grid (rows, ⌈V_h / block⌉).
template <class T>
__global__ void ssm_gdn_prep_ba_gates(const T* __restrict__ ba, const float* __restrict__ a_log,
                                      const T* __restrict__ dt_bias, float* __restrict__ gates, int N,
                                      int V_h, const u32* __restrict__ win) {
    const int t = blockIdx.x;
    const int h = blockIdx.y * blockDim.x + threadIdx.x;
    if (t >= N || h >= V_h || (win != nullptr && t >= static_cast<int>(win[0]))) return;
    const long long row = win != nullptr ? t + static_cast<long long>(win[1]) : t;
    const float b = to_f32(ba[row * 2 * V_h + h]);
    const float z = to_f32(ba[row * 2 * V_h + V_h + h]) + to_f32(dt_bias[h]);
    const float softplus = z > 20.f ? z : log1pf(__expf(z));
    gates[row * 2 * V_h + h] = -__expf(a_log[h]) * softplus;
    gates[row * 2 * V_h + V_h + h] = 1.f / (1.f + __expf(-b));
}

// The l2-normed q (scaled) and k of each key head, staged in f32 at the
// launch row. Grid (rows, K_h), block BLOCK.
template <class T, int BLOCK>
__global__ void ssm_gdn_prep_qk_norm(const T* __restrict__ qkv, float* __restrict__ q_out,
                                     float* __restrict__ k_out, int K_h, int K_d, int conv_dim,
                                     float q_scale, const u32* __restrict__ win) {
    __shared__ float smem[2][BLOCK / kWave];
    const int n = blockIdx.x;
    if (win != nullptr && n >= static_cast<int>(win[0])) return;
    const long long row = win != nullptr ? n + static_cast<long long>(win[1]) : n;
    const int h = blockIdx.y;
    const T* q = qkv + row * conv_dim + static_cast<long long>(h) * K_d;
    const T* k = q + K_h * K_d;
    float qs = 0.f, ks = 0.f;
    for (int i = threadIdx.x; i < K_d; i += BLOCK) {
        qs += to_f32(q[i]) * to_f32(q[i]);
        ks += to_f32(k[i]) * to_f32(k[i]);
    }
    const float q_inv = rsqrtf(block_sum(qs, smem[0]) + 1e-6f) * q_scale;
    const float k_inv = rsqrtf(block_sum(ks, smem[1]) + 1e-6f);
    float* qd = q_out + (static_cast<long long>(n) * K_h + h) * K_d;
    float* kd = k_out + (static_cast<long long>(n) * K_h + h) * K_d;
    for (int i = threadIdx.x; i < K_d; i += BLOCK) {
        qd[i] = to_f32(q[i]) * q_inv;
        kd[i] = to_f32(k[i]) * k_inv;
    }
}

// v and the gates of each value head, staged in f32 at the launch row.
// Grid (rows, V_h), block BLOCK.
template <class T, int BLOCK>
__global__ void ssm_gdn_prep_v_gates(const T* __restrict__ qkv, const float* __restrict__ gates,
                                     float* __restrict__ v_out, float* __restrict__ g_out,
                                     float* __restrict__ beta_out, int K_h, int V_h, int K_d, int V_d,
                                     int conv_dim, const u32* __restrict__ win) {
    const int n = blockIdx.x;
    if (win != nullptr && n >= static_cast<int>(win[0])) return;
    const long long row = win != nullptr ? n + static_cast<long long>(win[1]) : n;
    const int h = blockIdx.y;
    const T* v = qkv + row * conv_dim + 2LL * K_h * K_d + static_cast<long long>(h) * V_d;
    float* vd = v_out + (static_cast<long long>(n) * V_h + h) * V_d;
    for (int i = threadIdx.x; i < V_d; i += BLOCK) vd[i] = to_f32(v[i]);
    if (threadIdx.x == 0) {
        const long long at = static_cast<long long>(n) * V_h + h;
        g_out[at] = gates[row * 2 * V_h + h];
        beta_out[at] = gates[row * 2 * V_h + V_h + h];
    }
}

// --------------------------------------------------------- decode (step) --

// One token per row, straight from the post-conv qkv row and the gates.
// Grid (V_d / BV, rows, V_h), block 256: thread t owns state row i = t / TPR
// and 8 columns j0.. of the block's BV-column tile, so K_d ≤ 256 / TPR.
template <int BV>
__global__ __launch_bounds__(256) void ssm_gdn_decode_step(
    const bf16* __restrict__ qkv, const float* __restrict__ gates, bf16* __restrict__ state_base,
    const i32* __restrict__ slot_ids, long long slot_stride, float* __restrict__ out, int K_h, int V_h,
    int K_d, int V_d, int conv_dim, float q_scale, bool write_state, const u8* __restrict__ mask,
    const u32* __restrict__ win) {
    constexpr int BLOCK = 256, VEC = 8, TPR = BV / VEC, WAVES = BLOCK / kWave;
    static_assert(BV % VEC == 0 && kWave % TPR == 0, "a tile row is whole 8-wide packs");
    __shared__ float sq[BLOCK / TPR], sk[BLOCK / TPR];
    __shared__ float red[WAVES][BV];
    __shared__ float norm_part[2][WAVES];

    const int r = blockIdx.y, h = blockIdx.z;
    if (win != nullptr && r >= static_cast<int>(win[0])) return;
    const long long row = win != nullptr ? r + static_cast<long long>(win[1]) : r;
    const int rl = win != nullptr ? r + static_cast<int>(win[3]) : r;
    const int slot = slot_ids[rl];
    if (slot < 0) return;
    const bool fold = folds(write_state, mask, rl);

    const int tid = threadIdx.x, wave = tid / kWave;
    const int h_k = h / (V_h / K_h);
    const bf16* qkv_row = qkv + row * conv_dim;
    const int i = tid / TPR;
    const int j0 = blockIdx.x * BV + (tid % TPR) * VEC;
    const bool live = i < K_d;

    // Every read of the step issued before the norms' barriers.
    bf16* cell = state_base + static_cast<long long>(slot) * slot_stride + static_cast<long long>(h) * K_d * V_d +
                 static_cast<long long>(i) * V_d + j0;
    Pack8<bf16> sp{};
    if (live) sp = *reinterpret_cast<const Pack8<bf16>*>(cell);
    const Pack8<bf16> vp = *reinterpret_cast<const Pack8<bf16>*>(qkv_row + 2LL * K_h * K_d + static_cast<long long>(h) * V_d + j0);
    const float g_log = gates[row * 2 * V_h + h];
    const float beta = gates[row * 2 * V_h + V_h + h];

    // The norms of q and k over the key head.
    const float qv = tid < K_d ? to_f32(qkv_row[h_k * K_d + tid]) : 0.f;
    const float kv = tid < K_d ? to_f32(qkv_row[K_h * K_d + h_k * K_d + tid]) : 0.f;
    const float qw = wave_sum(qv * qv), kw = wave_sum(kv * kv);
    if (lane() == 0) {
        norm_part[0][wave] = qw;
        norm_part[1][wave] = kw;
    }
    lds_barrier();
    float qn = 0.f, kn = 0.f;
#pragma unroll
    for (int w = 0; w < WAVES; ++w) {
        qn += norm_part[0][w];
        kn += norm_part[1][w];
    }
    if (tid < K_d) {
        sq[tid] = qv * rsqrtf(qn + 1e-6f) * q_scale;
        sk[tid] = kv * rsqrtf(kn + 1e-6f);
    }
    lds_barrier();
    const float qi = live ? sq[i] : 0.f, ki = live ? sk[i] : 0.f;


    // Column sums over the state rows: lanes that share tid % TPR within a
    // wavefront, then the wavefronts through LDS.
    auto column_sum = [&](float (&x)[VEC]) {
#pragma unroll
        for (int u = 0; u < VEC; ++u) {
#pragma unroll
            for (int off = TPR; off < kWave; off <<= 1) x[u] += __shfl_xor(x[u], off, kWave);
        }
        if (lane() < TPR) {
#pragma unroll
            for (int u = 0; u < VEC; ++u) red[wave][lane() * VEC + u] = x[u];
        }
        lds_barrier();
#pragma unroll
        for (int u = 0; u < VEC; ++u) {
            float total = 0.f;
#pragma unroll
            for (int w = 0; w < WAVES; ++w) total += red[w][(tid % TPR) * VEC + u];
            x[u] = total;
        }
        lds_barrier();
    };

    const float g = __expf(g_log);
    float s[VEC], x[VEC];
#pragma unroll
    for (int u = 0; u < VEC; ++u) {
        s[u] = to_f32(sp.v[u]);
        x[u] = s[u] * g * ki;
    }
    column_sum(x);
#pragma unroll
    for (int u = 0; u < VEC; ++u) {
        const float delta = (to_f32(vp.v[u]) - x[u]) * beta;
        s[u] = to_f32(from_f32<bf16>(s[u] * g)) + ki * delta;
        x[u] = s[u] * qi;
    }
    if (live && fold) {
        Pack8<bf16> st;
#pragma unroll
        for (int u = 0; u < VEC; ++u) st.v[u] = from_f32<bf16>(s[u]);
        *reinterpret_cast<Pack8<bf16>*>(cell) = st;
    }
    column_sum(x);
    if (tid < TPR) {
        float* o = out + (row * V_h + h) * V_d + j0;
#pragma unroll
        for (int u = 0; u < VEC; ++u) o[u] = x[u];
    }
}

// The decode steps' state traffic: read once and written once a step, and
// not again until the next step, past every other layer's. Non-temporal, so
// it streams past the caches it would only evict from (a tenth faster at a
// batch's 200-500 MiB of state).
using u32x4 = u32 __attribute__((ext_vector_type(4)));
__device__ __forceinline__ u32x4 state_load(const bf16* p) {
    return __builtin_nontemporal_load(reinterpret_cast<const u32x4*>(p));
}
__device__ __forceinline__ void state_store(bf16* p, u32x4 v) {
    __builtin_nontemporal_store(v, reinterpret_cast<u32x4*>(p));
}
__device__ __forceinline__ f32x4 state_load(const f32x4* p) { return __builtin_nontemporal_load(p); }
__device__ __forceinline__ void state_store(f32x4* p, f32x4 v) { __builtin_nontemporal_store(v, p); }

// bf16 pairs packed in a word, low element first.
__device__ __forceinline__ float bf16_lo(u32 w) { return __builtin_bit_cast(float, w << 16); }
__device__ __forceinline__ float bf16_hi(u32 w) { return __builtin_bit_cast(float, w & 0xffff0000u); }

// Round to nearest even into the upper half, as `from_f32<bf16>` rounds every
// finite value and the NaNs arithmetic makes (canonical, or widened from
// bf16): two integer ops where the conversion spends five, and a pair packs
// in one more.
__device__ __forceinline__ u32 bf16_rne_bits(float x) {
    const u32 u = __builtin_bit_cast(u32, x);
    return u + 0x7fffu + ((u >> 16) & 1u);
}
__device__ __forceinline__ float bf16_round(float x) {
    return __builtin_bit_cast(float, bf16_rne_bits(x) & 0xffff0000u);
}
__device__ __forceinline__ u32 bf16_pack_rne(float lo, float hi) {
    return __builtin_amdgcn_perm(bf16_rne_bits(hi), bf16_rne_bits(lo), 0x07060302u);
}

// The step for a batch: grid (V_h, rows), block BLOCK, a block per (head,
// row), consecutive blocks on consecutive heads of a slot so the device
// sweeps each slot's state in order. Thread t owns columns 8(t % 16).. of
// rows t / 16 + (BLOCK / 16)q: a wavefront read is four whole consecutive
// 256-byte state rows, the block's reads its head's 32 KiB in order. The
// norms of q and k factor out of the sums,
//
//   kv[j] = g·|k|⁻¹·Σ_i S[i][j]·k[i]      o[j] = q_scale·|q|⁻¹·Σ_i S'[i][j]·q[i]
//
// so the block meets at two barriers: one for the kv sums and the norms
// together, one for the outputs (the column sums fold the lane's rows, the
// row groups of a wavefront by shuffles, then the wavefronts through LDS).
// The arithmetic of `ssm_gdn_decode_step` in fused pairs and another order.
template <int KD, int VD, int BLOCK>
__global__ __launch_bounds__(BLOCK) void ssm_gdn_decode_rows(
    const bf16* __restrict__ qkv, const float* __restrict__ gates, bf16* __restrict__ state_base,
    const i32* __restrict__ slot_ids, long long slot_stride, float* __restrict__ out, int K_h, int V_h,
    int conv_dim, float q_scale, bool write_state, const u8* __restrict__ mask, const u32* __restrict__ win) {
    constexpr int WAVES = BLOCK / kWave, VEC = 8, CG = VD / VEC, RG = BLOCK / CG, Q = KD / RG;
    static_assert(VD % VEC == 0 && BLOCK % CG == 0 && KD % RG == 0 && kWave % CG == 0,
                  "whole packs, whole row groups");
    __shared__ alignas(16) float red[2][WAVES][VD];
    __shared__ float norms[WAVES][2];

    const int h = blockIdx.x, r = blockIdx.y;
    if (win != nullptr && r >= static_cast<int>(win[0])) return;
    const long long row = win != nullptr ? r + static_cast<long long>(win[1]) : r;
    const int rl = win != nullptr ? r + static_cast<int>(win[3]) : r;
    const int slot = slot_ids[rl];
    if (slot < 0) return;
    const int tid = threadIdx.x, wave = tid / kWave, l = lane();
    const int h_k = h / (V_h / K_h);
    const bf16* qkv_row = qkv + row * conv_dim;
    const int cg = tid % CG, rg = tid / CG, j0 = cg * VEC;

    // The token's reads, then the state's: every read of the step in flight
    // before any is used.
    float qi[Q], ki[Q];
#pragma unroll
    for (int q = 0; q < Q; ++q) {
        qi[q] = to_f32(qkv_row[h_k * KD + rg + RG * q]);
        ki[q] = to_f32(qkv_row[K_h * KD + h_k * KD + rg + RG * q]);
    }
    const Pack8<bf16> vp = *reinterpret_cast<const Pack8<bf16>*>(qkv_row + 2LL * K_h * KD + static_cast<long long>(h) * VD + j0);
    const float g_log = gates[row * 2 * V_h + h];
    const float beta = gates[row * 2 * V_h + V_h + h];
    bf16* head = state_base + static_cast<long long>(slot) * slot_stride + static_cast<long long>(h) * KD * VD;
    u32x4 sp[Q];
#pragma unroll
    for (int q = 0; q < Q; ++q) sp[q] = state_load(head + (rg + RG * q) * VD + j0);
    const bool fold = folds_uniform(write_state, mask, rl);

    // Over the row groups of a column: the wavefront's lanes CG apart, then
    // the wavefronts. Every thread of a column gets the same total.
    auto lanes = [](float v) {
#pragma unroll
        for (int off = CG; off < kWave; off <<= 1) v += __shfl_xor(v, off, kWave);
        return v;
    };
    auto column_sums = [&](float (&x)[VEC], int buf) {
#pragma unroll
        for (int u = 0; u < VEC; ++u) x[u] = lanes(x[u]);
        if (l < CG) {
            *reinterpret_cast<f32x4*>(&red[buf][wave][j0]) = f32x4{x[0], x[1], x[2], x[3]};
            *reinterpret_cast<f32x4*>(&red[buf][wave][j0 + 4]) = f32x4{x[4], x[5], x[6], x[7]};
        }
    };
    auto column_totals = [&](float (&x)[VEC], int buf) {
#pragma unroll
        for (int u = 0; u < VEC; ++u) x[u] = 0.f;
#pragma unroll
        for (int w = 0; w < WAVES; ++w) {
            const f32x4 a = *reinterpret_cast<const f32x4*>(&red[buf][w][j0]);
            const f32x4 b = *reinterpret_cast<const f32x4*>(&red[buf][w][j0 + 4]);
#pragma unroll
            for (int u = 0; u < 4; ++u) {
                x[u] += a[u];
                x[4 + u] += b[u];
            }
        }
    };

    // Σ_i S[i][j]·k[i] and the norms' partial sums.
    f32x2 x2[VEC / 2];
#pragma unroll
    for (int w = 0; w < VEC / 2; ++w) x2[w] = f32x2{0.f, 0.f};
    float qq = 0.f, kk = 0.f;
#pragma unroll
    for (int q = 0; q < Q; ++q) {
        qq += qi[q] * qi[q];
        kk += ki[q] * ki[q];
        const f32x2 k2 = {ki[q], ki[q]};
#pragma unroll
        for (int w = 0; w < VEC / 2; ++w)
            x2[w] = __builtin_elementwise_fma(f32x2{bf16_lo(sp[q][w]), bf16_hi(sp[q][w])}, k2, x2[w]);
    }
    // Unpacked again for the update rather than held: half the registers,
    // a fifth more wavefronts resident.
#pragma unroll
    for (int q = 0; q < Q; ++q) asm volatile("" : "+v"(sp[q]));
    float x[VEC];
#pragma unroll
    for (int w = 0; w < VEC / 2; ++w) {
        x[2 * w] = x2[w].x;
        x[2 * w + 1] = x2[w].y;
    }
    column_sums(x, 0);
    qq = lanes(qq);
    kk = lanes(kk);
    if (l == 0) {
        norms[wave][0] = qq;
        norms[wave][1] = kk;
    }
    lds_barrier();
    column_totals(x, 0);
    float qn = 0.f, kn = 0.f;
#pragma unroll
    for (int w = 0; w < WAVES; ++w) {
        qn += norms[w][0];
        kn += norms[w][1];
    }
    const float g = __expf(g_log);
    const float rk = rsqrtf(kn + 1e-6f), rq = rsqrtf(qn + 1e-6f) * q_scale;

    // δ[j], the update, Σ_i S'[i][j]·q[i].
    f32x2 d2[VEC / 2];
#pragma unroll
    for (int w = 0; w < VEC / 2; ++w) {
        d2[w] = f32x2{(to_f32(vp.v[2 * w]) - x[2 * w] * (g * rk)) * beta,
                      (to_f32(vp.v[2 * w + 1]) - x[2 * w + 1] * (g * rk)) * beta};
        x2[w] = f32x2{0.f, 0.f};
    }
    const f32x2 g2 = {g, g};
#pragma unroll
    for (int q = 0; q < Q; ++q) {
        const float kh = ki[q] * rk;
        const f32x2 k2 = {kh, kh}, q2 = {qi[q], qi[q]};
        u32x4 st;
#pragma unroll
        for (int w = 0; w < VEC / 2; ++w) {
            const f32x2 dec = f32x2{bf16_lo(sp[q][w]), bf16_hi(sp[q][w])} * g2;
            const f32x2 next = __builtin_elementwise_fma(k2, d2[w], f32x2{bf16_round(dec.x), bf16_round(dec.y)});
            st[w] = bf16_pack_rne(next.x, next.y);
            x2[w] = __builtin_elementwise_fma(next, q2, x2[w]);
        }
        if (fold) state_store(head + (rg + RG * q) * VD + j0, st);
    }
#pragma unroll
    for (int w = 0; w < VEC / 2; ++w) {
        x[2 * w] = x2[w].x;
        x[2 * w + 1] = x2[w].y;
    }
    column_sums(x, 1);
    lds_barrier();
    if (tid < CG) {
        column_totals(x, 1);
        float* o = out + (row * V_h + h) * VD + j0;
        *reinterpret_cast<f32x4*>(o) = f32x4{x[0] * rq, x[1] * rq, x[2] * rq, x[3] * rq};
        *reinterpret_cast<f32x4*>(o + 4) = f32x4{x[4] * rq, x[5] * rq, x[6] * rq, x[7] * rq};
    }
}

// The whole decode step of a gated-delta layer in one launch, from the
// in-projections' rows: split [qkv | z], the causal conv's update of the
// head's q, k and v channels, the gate prep, and the recurrence of
// `ssm_gdn_decode_rows`, each rounding where its unfused op stores. Grid
// (V_h, rows), block BLOCK, a block per (value head, row).
//
// The q and k channels of a key head are shared by its G = V_h / K_h value
// heads, so each of their blocks convolves them; their conv window may only
// shift once all G have read it. Each block reads, then draws a ticket
// (`tickets[r · K_h + h_k]`, zero between launches); the block that draws
// the last shifts the shared channels and puts the ticket back to zero.
// Each block shifts its own value channels. `qkv` and `z` land the split's
// halves, which the step's other readers (the replay planes, the output
// norm) read.
template <int KD, int VD, int BLOCK, int KMAX>
__global__ __launch_bounds__(BLOCK) void ssm_gdn_step_fused(
    const bf16* __restrict__ qkvz, const bf16* __restrict__ conv_w, bf16* __restrict__ conv_base,
    const i32* __restrict__ conv_slots, long long conv_stride, bool conv_write, const u8* __restrict__ conv_mask,
    const bf16* __restrict__ ba, const float* __restrict__ a_log, const bf16* __restrict__ dt_bias,
    bf16* __restrict__ state_base, const i32* __restrict__ slot_ids, long long slot_stride, bool write_state,
    const u8* __restrict__ mask, bf16* __restrict__ qkv_out, bf16* __restrict__ z_out, float* __restrict__ out,
    int* __restrict__ tickets, int K_h, int V_h, int K, float q_scale, const u32* __restrict__ win) {
    constexpr int WAVES = BLOCK / kWave, VEC = 8, CG = VD / VEC, RG = BLOCK / CG, Q = KD / RG;
    constexpr int CH = 2 * KD + VD, CPT = (CH + BLOCK - 1) / BLOCK;
    static_assert(VD % VEC == 0 && BLOCK % CG == 0 && KD % RG == 0 && kWave % CG == 0,
                  "whole packs, whole row groups");
    __shared__ alignas(16) float red[2][WAVES][VD];
    __shared__ float norms[WAVES][2];
    __shared__ float cv[CH];
    __shared__ int last_s;

    const int h = blockIdx.x, r = blockIdx.y;
    if (win != nullptr && r >= static_cast<int>(win[0])) return;
    const long long row = win != nullptr ? r + static_cast<long long>(win[1]) : r;
    const int rl = win != nullptr ? r + static_cast<int>(win[3]) : r;
    const int slot = slot_ids[rl];
    const int cslot = conv_slots[rl];
    if (slot < 0 || cslot < 0) return;
    const int tid = threadIdx.x, wave = tid / kWave, l = lane();
    const int G = V_h / K_h;
    const int h_k = h / G;
    const int C = 2 * K_h * KD + V_h * VD;  // the conv's channels: [q | k | v]
    const long long QW = static_cast<long long>(C) + static_cast<long long>(V_h) * VD;
    const bf16* in_row = qkvz + row * QW;
    const int span = K - 1;

    // The conv over this block's channels: q and k of its key head, v of
    // its own; every read in flight before any is used.
    int ch[CPT];
    float xn[CPT], hist[CPT][KMAX], wk[CPT][KMAX];
    bf16* cstate = conv_base + static_cast<long long>(cslot) * conv_stride;
#pragma unroll
    for (int u = 0; u < CPT; ++u) {
        const int ii = min(tid + u * BLOCK, CH - 1);
        ch[u] = ii < KD       ? h_k * KD + ii
                : ii < 2 * KD ? K_h * KD + h_k * KD + (ii - KD)
                              : 2 * K_h * KD + h * VD + (ii - 2 * KD);
        xn[u] = to_f32(in_row[ch[u]]);
#pragma unroll
        for (int k = 0; k < KMAX; ++k) {
            wk[u][k] = k < K ? to_f32(conv_w[ch[u] * K + k]) : 0.f;
            hist[u][k] = k + 1 <= span ? to_f32(cstate[(k + 1) * C + ch[u]]) : 0.f;
        }
    }
    const bool cfold = folds_uniform(conv_write, conv_mask, rl);
#pragma unroll
    for (int u = 0; u < CPT; ++u) {
        const int i = tid + u * BLOCK;
        if (i >= CH) continue;
        float acc = 0.f;
#pragma unroll
        for (int k = 0; k < KMAX - 1; ++k)
            if (k < K - 1) acc += wk[u][k] * hist[u][k];
        acc += wk[u][K - 1 < KMAX ? K - 1 : 0] * xn[u];
        cv[i] = to_f32(from_f32<bf16>(silu(acc)));
        const bool own = i >= 2 * KD;
        if (own || h % G == 0) qkv_out[row * C + ch[u]] = from_f32<bf16>(xn[u]);
        if (own && cfold) {
#pragma unroll
            for (int k = 0; k < KMAX; ++k)
                if (k < span) cstate[k * C + ch[u]] = from_f32<bf16>(hist[u][k]);
            cstate[span * C + ch[u]] = from_f32<bf16>(xn[u]);
        }
    }
    if (tid < VD) z_out[row * (V_h * VD) + h * VD + tid] = in_row[C + h * VD + tid];
    // The ticket, once every read of the shared window has landed.
    __syncthreads();
    if (tid == 0) {
        // The last block writes only what every block has read already
        // (the barrier), so the ticket alone orders them.
        const int at = r * K_h + h_k;
        const bool last = atomicAdd(&tickets[at], 1) == G - 1;
        if (last) atomicExch(&tickets[at], 0);
        last_s = last;
    }
    __syncthreads();
    if (last_s && cfold) {
#pragma unroll
        for (int u = 0; u < CPT; ++u) {
            const int i = tid + u * BLOCK;
            if (i >= 2 * KD) continue;
#pragma unroll
            for (int k = 0; k < KMAX; ++k)
                if (k < span) cstate[k * C + ch[u]] = from_f32<bf16>(hist[u][k]);
            cstate[span * C + ch[u]] = from_f32<bf16>(xn[u]);
        }
    }

    // The gates, as `ssm_gdn_prep_ba_gates` lands them.
    const float bb = to_f32(ba[row * 2 * V_h + h]);
    const float za = to_f32(ba[row * 2 * V_h + V_h + h]) + to_f32(dt_bias[h]);
    const float softplus = za > 20.f ? za : log1pf(__expf(za));
    const float g_log = -__expf(a_log[h]) * softplus;
    const float beta = 1.f / (1.f + __expf(-bb));

    // The recurrence, as `ssm_gdn_decode_rows` walks it.
    const int cg = tid % CG, rg = tid / CG, j0 = cg * VEC;
    bf16* head = state_base + static_cast<long long>(slot) * slot_stride + static_cast<long long>(h) * KD * VD;
    u32x4 sp[Q];
#pragma unroll
    for (int q = 0; q < Q; ++q) sp[q] = state_load(head + (rg + RG * q) * VD + j0);
    const bool fold = folds_uniform(write_state, mask, rl);
    float qi[Q], ki[Q], vv[VEC];
#pragma unroll
    for (int q = 0; q < Q; ++q) {
        qi[q] = cv[rg + RG * q];
        ki[q] = cv[KD + rg + RG * q];
    }
#pragma unroll
    for (int u = 0; u < VEC; ++u) vv[u] = cv[2 * KD + j0 + u];

    auto lanes = [](float v) {
#pragma unroll
        for (int off = CG; off < kWave; off <<= 1) v += __shfl_xor(v, off, kWave);
        return v;
    };
    auto column_sums = [&](float (&x)[VEC], int buf) {
#pragma unroll
        for (int u = 0; u < VEC; ++u) x[u] = lanes(x[u]);
        if (l < CG) {
            *reinterpret_cast<f32x4*>(&red[buf][wave][j0]) = f32x4{x[0], x[1], x[2], x[3]};
            *reinterpret_cast<f32x4*>(&red[buf][wave][j0 + 4]) = f32x4{x[4], x[5], x[6], x[7]};
        }
    };
    auto column_totals = [&](float (&x)[VEC], int buf) {
#pragma unroll
        for (int u = 0; u < VEC; ++u) x[u] = 0.f;
#pragma unroll
        for (int w = 0; w < WAVES; ++w) {
            const f32x4 a = *reinterpret_cast<const f32x4*>(&red[buf][w][j0]);
            const f32x4 b = *reinterpret_cast<const f32x4*>(&red[buf][w][j0 + 4]);
#pragma unroll
            for (int u = 0; u < 4; ++u) {
                x[u] += a[u];
                x[4 + u] += b[u];
            }
        }
    };

    f32x2 x2[VEC / 2];
#pragma unroll
    for (int w = 0; w < VEC / 2; ++w) x2[w] = f32x2{0.f, 0.f};
    float qq = 0.f, kk = 0.f;
#pragma unroll
    for (int q = 0; q < Q; ++q) {
        qq += qi[q] * qi[q];
        kk += ki[q] * ki[q];
        const f32x2 k2 = {ki[q], ki[q]};
#pragma unroll
        for (int w = 0; w < VEC / 2; ++w)
            x2[w] = __builtin_elementwise_fma(f32x2{bf16_lo(sp[q][w]), bf16_hi(sp[q][w])}, k2, x2[w]);
    }
#pragma unroll
    for (int q = 0; q < Q; ++q) asm volatile("" : "+v"(sp[q]));
    float x[VEC];
#pragma unroll
    for (int w = 0; w < VEC / 2; ++w) {
        x[2 * w] = x2[w].x;
        x[2 * w + 1] = x2[w].y;
    }
    column_sums(x, 0);
    qq = lanes(qq);
    kk = lanes(kk);
    if (l == 0) {
        norms[wave][0] = qq;
        norms[wave][1] = kk;
    }
    lds_barrier();
    column_totals(x, 0);
    float qn = 0.f, kn = 0.f;
#pragma unroll
    for (int w = 0; w < WAVES; ++w) {
        qn += norms[w][0];
        kn += norms[w][1];
    }
    const float g = __expf(g_log);
    const float rk = rsqrtf(kn + 1e-6f), rq = rsqrtf(qn + 1e-6f) * q_scale;

    f32x2 d2[VEC / 2];
#pragma unroll
    for (int w = 0; w < VEC / 2; ++w) {
        d2[w] = f32x2{(vv[2 * w] - x[2 * w] * (g * rk)) * beta, (vv[2 * w + 1] - x[2 * w + 1] * (g * rk)) * beta};
        x2[w] = f32x2{0.f, 0.f};
    }
    const f32x2 g2 = {g, g};
#pragma unroll
    for (int q = 0; q < Q; ++q) {
        const float kh = ki[q] * rk;
        const f32x2 k2 = {kh, kh}, q2 = {qi[q], qi[q]};
        u32x4 st;
#pragma unroll
        for (int w = 0; w < VEC / 2; ++w) {
            const f32x2 dec = f32x2{bf16_lo(sp[q][w]), bf16_hi(sp[q][w])} * g2;
            const f32x2 next = __builtin_elementwise_fma(k2, d2[w], f32x2{bf16_round(dec.x), bf16_round(dec.y)});
            st[w] = bf16_pack_rne(next.x, next.y);
            x2[w] = __builtin_elementwise_fma(next, q2, x2[w]);
        }
        if (fold) state_store(head + (rg + RG * q) * VD + j0, st);
    }
#pragma unroll
    for (int w = 0; w < VEC / 2; ++w) {
        x[2 * w] = x2[w].x;
        x[2 * w + 1] = x2[w].y;
    }
    column_sums(x, 1);
    lds_barrier();
    if (tid < CG) {
        column_totals(x, 1);
        float* o = out + (row * V_h + h) * VD + j0;
        *reinterpret_cast<f32x4*>(o) = f32x4{x[0] * rq, x[1] * rq, x[2] * rq, x[3] * rq};
        *reinterpret_cast<f32x4*>(o + 4) = f32x4{x[4] * rq, x[5] * rq, x[6] * rq, x[7] * rq};
    }
}

// ---------------------------------------------------- prefill (sequential) --

// The recurrence over each request's rows, token by token, from the staged
// f32 planes (indexed by launch row; the output lands at row + row_offset).
// Grid (⌈V_d / BV⌉, lanes, V_h), block BV: a thread owns one state column,
// K_d ≤ BK entries of it held in registers in bf16, as the slab holds them.
// `fused_decay` rounds the decayed state only once per token, with the update.
template <int BK, int BV>
__global__ __launch_bounds__(BV) void ssm_gdn_scan(
    const float* __restrict__ q_norm, const float* __restrict__ k_norm, const float* __restrict__ v,
    const float* __restrict__ g_log, const float* __restrict__ beta, bf16* __restrict__ state_base,
    const i32* __restrict__ slot_ids, const u32* __restrict__ indptr, long long slot_stride,
    float* __restrict__ out, int K_h, int V_h, int K_d, int V_d, bool write_state,
    const int* __restrict__ commit_len, const u8* __restrict__ mask, const int* __restrict__ begin_at,
    bool fused_decay, const u32* __restrict__ win) {
    __shared__ float sq[BK], sk[BK];
    const int r = blockIdx.y, h = blockIdx.z;
    const int j = blockIdx.x * BV + threadIdx.x;
    if (win != nullptr && r >= static_cast<int>(win[2])) return;
    const int rl = win != nullptr ? r + static_cast<int>(win[3]) : r;
    const long long row0 = win != nullptr ? static_cast<long long>(win[1]) : 0;
    const Span sp = span_of(indptr, r, rl, commit_len, begin_at);
    if (sp.n <= 0) return;
    const int slot = slot_ids[rl];
    if (slot < 0) return;
    const bool col = j < V_d;
    const int h_k = h / (V_h / K_h);
    bf16* state = state_base + static_cast<long long>(slot) * slot_stride + static_cast<long long>(h) * K_d * V_d;

    bf16 s[BK];
#pragma unroll
    for (int i = 0; i < BK; ++i) s[i] = (col && i < K_d) ? state[static_cast<long long>(i) * V_d + j] : from_f32<bf16>(0.f);

    for (int t = 0; t < sp.n; ++t) {
        const long long at = static_cast<long long>(sp.t0 + t);
        const float g = __expf(g_log[at * V_h + h]);
        const float b = beta[at * V_h + h];
        const float* qt = q_norm + (at * K_h + h_k) * K_d;
        const float* kt = k_norm + (at * K_h + h_k) * K_d;
        for (int i = threadIdx.x; i < K_d; i += blockDim.x) {
            sq[i] = qt[i];
            sk[i] = kt[i];
        }
        __syncthreads();
        float kv = 0.f;
#pragma unroll
        for (int i = 0; i < BK; ++i) {
            if (i >= K_d) break;
            const float decayed = to_f32(s[i]) * g;
            if (!fused_decay) s[i] = from_f32<bf16>(decayed);
            kv += decayed * sk[i];
        }
        const float delta = col ? (v[at * V_h * V_d + static_cast<long long>(h) * V_d + j] - kv) * b : 0.f;
        float o = 0.f;
#pragma unroll
        for (int i = 0; i < BK; ++i) {
            if (i >= K_d) break;
            const float next = (fused_decay ? to_f32(s[i]) * g : to_f32(s[i])) + sk[i] * delta;
            s[i] = from_f32<bf16>(next);
            o += next * sq[i];
        }
        if (col) out[((sp.t0 + t + row0) * V_h + h) * V_d + j] = o;
        __syncthreads();
    }

    if (!col || !folds(write_state, mask, rl)) return;
#pragma unroll
    for (int i = 0; i < BK; ++i) {
        if (i >= K_d) break;
        state[static_cast<long long>(i) * V_d + j] = s[i];
    }
}


// ------------------------------------------------- prefill (chunk-parallel) --

// The same recurrence over 64-token chunks (the WY form of the delta rule).
// Per value head, chunk tokens 0 ≤ i < L, Γ_i = Σ_{t ≤ i} g_log_t:
//
//   A[i][j] = β_i·exp(Γ_i − Γ_j)·k̂_i·k̂_j  (j < i),   T = (I + A)⁻¹
//   W = T·diag(β·exp Γ)·K                 U = T·diag(β)·V
//   V' = U − W·S                          (S the state entering the chunk)
//   O  = exp(Γ)·Q·S + (mask(exp(Γ_i − Γ_j))∘Q·Kᵀ)·V'
//   S ← exp(Γ_L)·S + Kᵀ·diag(exp(Γ_L − Γ))·V'
//
// in three passes: `wu` builds W and U per chunk (all chunks at once), `h`
// walks each request's chunks in order carrying S in f32 (a block per
// 16-column slice of a head's state), and `o` lands each chunk's output from
// the S that `h` recorded at its start. Matrix products run on the matrix
// core in bf16 with f32 accumulation; the state is rounded to its cell only
// where it is stored.
//
// Scratch is indexed by a chunk slot s = ⌊t0 / CH⌋ + r + j for request r's
// j-th chunk, which keeps requests' chunks disjoint without a prefix scan.

constexpr int GDN_CH = 64;

struct GdnLane {
    int t0;
    int T;
    int rl;
    int slot;
};

__device__ __forceinline__ GdnLane gdn_lane(int r, const u32* indptr, const i32* slot_ids, const int* commit_len,
                                            const int* begin_at, const u32* win) {
    GdnLane lane;
    lane.rl = win != nullptr ? r + static_cast<int>(win[3]) : r;
    const Span sp = span_of(indptr, r, lane.rl, commit_len, begin_at);
    lane.t0 = sp.t0;
    lane.T = sp.n;
    lane.slot = lane.T > 0 ? slot_ids[lane.rl] : -1;
    return lane;
}

__device__ __forceinline__ int gdn_chunk_base(int r, const u32* indptr, const int* begin_at, const u32* win) {
    int t0 = static_cast<int>(indptr[r]);
    if (begin_at != nullptr) {
        const int rl = win != nullptr ? r + static_cast<int>(win[3]) : r;
        const int b = min(begin_at[rl], static_cast<int>(indptr[r + 1]) - t0);
        if (b > 0) t0 += b;
    }
    return t0 / GDN_CH + r;
}

// The request whose chunk slots hold s.
__device__ __forceinline__ int gdn_slot_lane(int s, int lanes, const u32* indptr, const int* begin_at,
                                             const u32* win) {
    int lo = 0, hi = lanes - 1;
    while (lo < hi) {
        const int mid = (lo + hi + 1) >> 1;
        if (gdn_chunk_base(mid, indptr, begin_at, win) <= s) lo = mid;
        else hi = mid - 1;
    }
    return lo;
}

// T = (I + A)⁻¹ for a strictly lower CH x CH A in LDS, in 16 x 16 blocks:
// each diagonal block by forward substitution, T[i][c] = δ_ic − Σ_{j<i}
// A[i][j]·T[j][c] with lane c keeping column c in registers; then the blocks
// below it a block-row at a time, T_ij = −T_ii·Σ_{j≤k<i} A_ik·T_kj, in f32
// on the matrix core. T overwrites A in place (the blocks above the diagonal
// are left as they are). X is CH/16 16 x 16 f32 tiles of LDS scratch, one a
// wavefront; every thread of a block of at least CH/16 wavefronts calls it.
template <int CH>
__device__ __forceinline__ void unit_lower_inverse(float (*A)[CH], float* X) {
    constexpr int NB = CH / 16;
    const int wave = threadIdx.x / kWave, l = lane();
    if (wave < NB && l < 16) {
        const int d = 16 * wave;
        float col[16];
#pragma unroll
        for (int i = 0; i < 16; ++i) {
            float t = i == l ? 1.f : 0.f;
#pragma unroll
            for (int jj = 0; jj < i; ++jj) t -= A[d + i][d + jj] * col[jj];
            col[i] = t;
            A[d + i][d + l] = t;
        }
    }
    lds_barrier();
    // 16 x 16 · 16 x 16 in f32 from LDS rows at stride `ld`.
    auto product = [&](const float* m, int ldm, const float* n, int ldn, mfma_cd c) {
#pragma unroll
        for (int k0 = 0; k0 < 16; k0 += 4) {
            const int k = k0 + l / 16;
            c = mfma4_f32(m[(l % 16) * ldm + k], n[k * ldn + l % 16], c);
        }
        return c;
    };
    float* Xw = X + (wave < NB ? wave : 0) * 256;
    for (int bi = 1; bi < NB; ++bi) {
        const int bj = wave;  // the block-row's blocks, one per wavefront
        mfma_cd t = mfma_zero();
        const bool mine = bj < bi;
        if (mine) {
            mfma_cd x = mfma_zero();
            for (int bk = bj; bk < bi; ++bk) x = product(&A[16 * bi][16 * bk], CH, &A[16 * bk][16 * bj], CH, x);
            mfma_each(x, [&](int i, int c, float v) { Xw[i * 16 + c] = v; });
            // (the wavefront reads back only what it wrote)
            __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront");
            __builtin_amdgcn_wave_barrier();
            __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront");
            t = product(&A[16 * bi][16 * bi], CH, Xw, 16, mfma_zero());
        }
        lds_barrier();
        if (mine) mfma_each(t, [&](int i, int c, float v) { A[16 * bi + i][16 * bj + c] = -v; });
        lds_barrier();
    }
}

// Grid (slots, V_h), block 256, straight from the post-conv qkv rows and the
// fused [g_log | β] gates (both at launch row + row_offset). Lands Γ (gcum,
// f32 [n][V_h]), the bf16 q̂ and k̂ rows (qb, kb [n][K_h][BK]), k̂ᵀ per chunk
// (kT [s][K_h][BK][CH], zero past the chunk's rows), W ([n][V_h][BK]) and Uᵀ
// per chunk (uT [s][V_h][BV][CH]); q̂, k̂ are l2-normed as the staging
// kernels norm them.
template <int BK, int BV>
__global__ __launch_bounds__(512) void ssm_gdn_chunk_wu(
    const bf16* __restrict__ qkv, const float* __restrict__ gates, const i32* __restrict__ slot_ids,
    const u32* __restrict__ indptr, const int* __restrict__ commit_len, const int* __restrict__ begin_at,
    float* __restrict__ gcum, bf16* __restrict__ qb, bf16* __restrict__ kb_out, bf16* __restrict__ kT,
    bf16* __restrict__ w_out, bf16* __restrict__ u_out, int lanes, int K_h, int V_h, int conv_dim, float q_scale,
    const u32* __restrict__ win) {
    static_assert(BK == BV, "the value rows reuse the key rows' buffer");
    constexpr int CH = GDN_CH, THREADS = 512, WAVES = THREADS / 64, VEC = 8, RowLanes = BK / VEC;
    // Row strides padded by 8 bytes so the matrix core's 16-row reads fall
    // in distinct banks.
    constexpr int KP = BK + 4, TP = CH + 4;
    constexpr int PER = CH * BK / VEC / THREADS;  // 16-byte loads per thread per plane
    static_assert(kWave % RowLanes == 0, "a row's lanes share a wavefront");
    __shared__ float A[CH][CH];
    __shared__ alignas(16) bf16 kb[CH * KP];  // k̂ rows; later diag(β)·V
    __shared__ alignas(16) bf16 Tb[CH * TP];
    __shared__ float G[CH], B[CH], BE[CH];

    const int h = blockIdx.y, s = blockIdx.x;
    if (win != nullptr) lanes = min(lanes, static_cast<int>(win[2]));
    if (lanes <= 0) return;
    const int r = gdn_slot_lane(s, lanes, indptr, begin_at, win);
    const GdnLane ln = gdn_lane(r, indptr, slot_ids, commit_len, begin_at, win);
    if (ln.T <= 0 || ln.slot < 0) return;
    const int j = s - gdn_chunk_base(r, indptr, begin_at, win);
    if (j < 0 || j * CH >= ln.T) return;
    const int ta = ln.t0 + j * CH;
    const int L = min(CH, ln.t0 + ln.T - ta);
    const long long row0 = win != nullptr ? static_cast<long long>(win[1]) : 0;
    const int group = V_h / K_h, hk = h / group;
    const bool leader = h % group == 0;  // one value head per group lands the key planes
    const int tid = threadIdx.x, wave = tid / kWave, l = lane();

    // Every load of the chunk issued before any is used.
    b128 kx[PER], qx[PER], vx[PER];
#pragma unroll
    for (int p = 0; p < PER; ++p) {
        const int e = tid + p * THREADS, i = e / RowLanes, c = VEC * (e % RowLanes);
        const bf16* row = qkv + (row0 + ta + i) * conv_dim;
        const bool live = i < L;
        kx[p] = live ? *reinterpret_cast<const b128*>(row + (K_h + hk) * BK + c) : b128{};
        qx[p] = live && leader ? *reinterpret_cast<const b128*>(row + hk * BK + c) : b128{};
        vx[p] = live ? *reinterpret_cast<const b128*>(row + 2 * K_h * BK + h * BV + c) : b128{};
    }
    if (wave == 0) {
        // Γ: an inclusive scan across the wavefront.
        const float* gr = gates + (row0 + ta + l) * 2 * V_h;
        float x = l < L ? gr[h] : 0.f;
#pragma unroll
        for (int off = 1; off < kWave; off <<= 1) {
            const float y = __shfl_up(x, off, kWave);
            if (l >= off) x += y;
        }
        G[l] = x;
        const float b = l < L ? gr[V_h + h] : 0.f;
        B[l] = b;
        BE[l] = b * __expf(x);
        if (l < L) gcum[static_cast<long long>(ta + l) * V_h + h] = x;
    }

    // The row's l2 norm over the RowLanes lanes that hold it.
    auto normed = [&](const b128& x, float scale) {
        const bf16* e = reinterpret_cast<const bf16*>(&x);
        float f[VEC], ss = 0.f;
#pragma unroll
        for (int u = 0; u < VEC; ++u) {
            f[u] = to_f32(e[u]);
            ss += f[u] * f[u];
        }
#pragma unroll
        for (int off = 1; off < RowLanes; off <<= 1) ss += __shfl_xor(ss, off, kWave);
        const float inv = rsqrtf(ss + 1e-6f) * scale;
        b128 out;
        bf16* o = reinterpret_cast<bf16*>(&out);
#pragma unroll
        for (int u = 0; u < VEC; ++u) o[u] = from_f32<bf16>(f[u] * inv);
        return out;
    };
#pragma unroll
    for (int p = 0; p < PER; ++p) {
        const int e = tid + p * THREADS, i = e / RowLanes, c = VEC * (e % RowLanes);
        const b128 k8 = normed(kx[p], 1.f);
        const mfma_ab* halves = reinterpret_cast<const mfma_ab*>(&k8);
        *reinterpret_cast<mfma_ab*>(kb + i * KP + c) = halves[0];
        *reinterpret_cast<mfma_ab*>(kb + i * KP + c + 4) = halves[1];
        if (leader && i < L) {
            const long long at = (static_cast<long long>(ta + i) * K_h + hk) * BK + c;
            *reinterpret_cast<b128*>(kb_out + at) = k8;
            *reinterpret_cast<b128*>(qb + at) = normed(qx[p], q_scale);
        }
    }
    lds_barrier();

    // k̂ᵀ for the state pass, a 64-row column of it per 16 lanes.
    if (leader) {
        bf16* kT_s = kT + (static_cast<long long>(s) * K_h + hk) * BK * CH;
        for (int e = tid; e < BK * CH / 4; e += THREADS) {
            const int kk = e / (CH / 4), i = 4 * (e % (CH / 4));
            bf16 x[4];
#pragma unroll
            for (int t = 0; t < 4; ++t) x[t] = kb[(i + t) * KP + kk];
            *reinterpret_cast<mfma_ab*>(kT_s + static_cast<long long>(kk) * CH + i) = *reinterpret_cast<const mfma_ab*>(x);
        }
    }

    // A = K̂·K̂ᵀ, masked and weighted.
    for (int tile = wave; tile < 16; tile += WAVES) {
        const int m0 = (tile / 4) * 16, n0 = (tile % 4) * 16;
        const mfma_cd c = mfma_tile<BK>(kb + m0 * KP, KP, kb + n0 * KP, KP, mfma_zero());
        mfma_each(c, [&](int i, int jj, float x) {
            i += m0;
            jj += n0;
            A[i][jj] = (i < L && jj < i) ? x * B[i] * __expf(G[i] - G[jj]) : 0.f;
        });
    }
    lds_barrier();

    // T = (I + A)⁻¹ in place, its scratch in Tb before Tb is built.
    unit_lower_inverse<CH>(A, reinterpret_cast<float*>(Tb));

    // The B operand (k j, column n) from row-major rows at stride KP.
    auto b_rows = [&](const bf16* m, int k0, int n0) {
        const int k = k0 + 4 * (l / 16), n = n0 + l % 16;
        bf16 x[4];
#pragma unroll
        for (int t = 0; t < 4; ++t) x[t] = m[(k + t) * KP + n];
        return *reinterpret_cast<const mfma_ab*>(x);
    };

    // W = T·diag(β·exp Γ)·K̂.
    for (int e = tid; e < CH * CH; e += THREADS) {
        const int i = e / CH, jj = e % CH;
        Tb[i * TP + jj] = from_f32<bf16>((i < L && jj < L) ? A[i][jj] * BE[jj] : 0.f);
    }
    lds_barrier();
    // As Wᵀ = K̂ᵀ·(T·diag)ᵀ, so a lane's four cells are adjacent keys of one row.
    for (int tile = wave; tile < 4 * (BK / 16); tile += WAVES) {
        const int k0r = (tile / 4) * 16, i0 = (tile % 4) * 16;
        mfma_cd c = mfma_zero();
#pragma unroll
        for (int k0 = 0; k0 < CH; k0 += 16)
            c = mfma16(b_rows(kb, k0, k0r), load4(Tb + (i0 + l % 16) * TP + k0 + 4 * (l / 16)), c);
        const int i = i0 + l % 16;
        if (i < L)
            *reinterpret_cast<mfma_ab*>(w_out + (static_cast<long long>(ta + i) * V_h + h) * BK + k0r + 4 * (l / 16)) =
                pack4(c[0], c[1], c[2], c[3]);
    }
    lds_barrier();

    // U = T·diag(β)·V.
    for (int e = tid; e < CH * CH; e += THREADS) {
        const int i = e / CH, jj = e % CH;
        Tb[i * TP + jj] = from_f32<bf16>((i < L && jj < L) ? A[i][jj] : 0.f);
    }
#pragma unroll
    for (int p = 0; p < PER; ++p) {
        const int e = tid + p * THREADS, i = e / RowLanes, c = VEC * (e % RowLanes);
        const float b = B[i];
        const bf16* x = reinterpret_cast<const bf16*>(&vx[p]);
#pragma unroll
        for (int u = 0; u < VEC; u += 4)
            *reinterpret_cast<mfma_ab*>(kb + i * KP + c + u) =
                pack4(to_f32(x[u]) * b, to_f32(x[u + 1]) * b, to_f32(x[u + 2]) * b, to_f32(x[u + 3]) * b);
    }
    lds_barrier();
    for (int tile = wave; tile < 4 * (BV / 16); tile += WAVES) {
        const int m0 = (tile / (BV / 16)) * 16, n0 = (tile % (BV / 16)) * 16;
        mfma_cd c = mfma_zero();
#pragma unroll
        for (int k0 = 0; k0 < CH; k0 += 16)
            c = mfma16(load4(Tb + (m0 + l % 16) * TP + k0 + 4 * (l / 16)), b_rows(kb, k0, n0), c);
        // The lane's four rows of one column are adjacent in Uᵀ.
        bf16* uT_s = u_out + ((static_cast<long long>(s) * V_h + h) * BV + n0 + l % 16) * CH + m0 + 4 * (l / 16);
        *reinterpret_cast<mfma_ab*>(uT_s) = pack4(c[0], c[1], c[2], c[3]);
    }
}

// Grid (lanes · V_h · BV/16), block 256: one 16-column slice of one head's
// state through every chunk of one request (state columns evolve
// independently), in f32 registers in the matrix core's D layout:
// wavefront w holds key rows [16w, 16w+16) and [16(w+4), 16(w+4)+16).
// Records each chunk's entering state (hcT [s][V_h][BV][BK]) and V' (vnT
// [s][V_h][BV][CH], zero past the chunk's rows). The next chunks' operands
// are loaded while this one computes, in few enough reads that three
// chunks' worth stay under the 63 the memory counter can wait on.
template <int BK, int BV>
__global__ __launch_bounds__(256) void ssm_gdn_chunk_h(
    const float* __restrict__ gcum, const bf16* __restrict__ kT, const bf16* __restrict__ w,
    const bf16* __restrict__ uT, bf16* __restrict__ vnT, bf16* __restrict__ hcT, bf16* __restrict__ state_base,
    const i32* __restrict__ slot_ids, const u32* __restrict__ indptr, long long slot_stride,
    const int* __restrict__ commit_len, const int* __restrict__ begin_at, bool write_state,
    const u8* __restrict__ mask, int lanes, int K_h, int V_h, const u32* __restrict__ win) {
    constexpr int CH = GDN_CH, BVT = 16, KT = BK / 16 / 4, KS = BK / 32, CS = CH / 32;
    // LDS rows padded by 16 bytes against the 16-row reads' bank conflicts.
    constexpr int HP = BK + 8, VP = CH + 8;
    static_assert(BK % 64 == 0, "four wavefronts split the key rows evenly");
    __shared__ alignas(16) bf16 hT[BVT * HP];
    __shared__ alignas(16) bf16 vnTs[BVT * VP];  // V'·exp(Γ_L − Γ), the update's operand
    __shared__ alignas(16) bf16 vnL[BVT * CH];   // V' as recorded
    __shared__ float sE[CH];

    const int vt = static_cast<int>(blockIdx.x) % (BV / BVT);
    const int h = (static_cast<int>(blockIdx.x) / (BV / BVT)) % V_h;
    const int r = static_cast<int>(blockIdx.x) / ((BV / BVT) * V_h);
    if (win != nullptr) lanes = min(lanes, static_cast<int>(win[2]));
    if (r >= lanes) return;
    const GdnLane ln = gdn_lane(r, indptr, slot_ids, commit_len, begin_at, win);
    if (ln.T <= 0 || ln.slot < 0) return;
    const int hk = h / (V_h / K_h);
    const int v0 = vt * BVT;
    const int tid = threadIdx.x, wave = tid / kWave, l = lane();
    bf16* state = state_base + static_cast<long long>(ln.slot) * slot_stride + static_cast<long long>(h) * BK * BV;

    float hf[KT][4];
#pragma unroll
    for (int q = 0; q < KT; ++q)
        mfma_each(mfma_zero(), [&](int i, int vv, float) {
            hf[q][i % 4] = to_f32(state[static_cast<long long>((wave + 4 * q) * 16 + i) * BV + v0 + vv]);
        });

    // One chunk's operands for this lane: W's A fragments for the
    // wavefront's 16 rows, U at its D cells, k̂ᵀ's A fragments for its key
    // rows, and (wavefront 0) Γ at its row and at the chunk's last.
    struct Ops {
        mfma_ab2 wf[KS];
        mfma_ab uv;
        mfma_ab2 kf[KT][CS];
        float g, gl;
    };
    const int base = gdn_chunk_base(r, indptr, begin_at, win);
    const int chunks = (ln.T + CH - 1) / CH;
    const int g8 = 8 * (l / 16);
    auto fetch = [&](int jc, Ops& o) {
        const int sj = base + jc, ta = ln.t0 + jc * CH, L = min(CH, ln.t0 + ln.T - ta);
        const int m0 = wave * 16;
        const bf16* wr = w + (static_cast<long long>(ta + m0 + l % 16) * V_h + h) * BK + g8;
#pragma unroll
        for (int st = 0; st < KS; ++st) o.wf[st] = load8(wr + st * 32);
        // Uᵀ: the lane's four rows of its column, adjacent.
        o.uv = load4(uT + ((static_cast<long long>(sj) * V_h + h) * BV + v0 + l % 16) * CH + m0 + 4 * (l / 16));
        const bf16* kT_s = kT + (static_cast<long long>(sj) * K_h + hk) * BK * CH;
#pragma unroll
        for (int q = 0; q < KT; ++q)
#pragma unroll
            for (int st = 0; st < CS; ++st) o.kf[q][st] = load8(kT_s + ((wave + 4 * q) * 16 + l % 16) * CH + st * 32 + g8);
        o.gl = gcum[static_cast<long long>(ta + L - 1) * V_h + h];
        o.g = gcum[static_cast<long long>(ta + min(l, L - 1)) * V_h + h];
    };

    // A ring of DEPTH chunks in flight: chunk c's operands land in ring[c % DEPTH]
    // while the DEPTH − 1 before it compute. Every step refills its slot
    // (past the last chunk, with the last chunk again) so each slot's loads
    // stay the oldest outstanding when the step that reads them comes round.
    constexpr int DEPTH = 3;
    Ops ring[DEPTH];
#pragma unroll
    for (int d = 0; d < DEPTH; ++d) fetch(min(d, chunks - 1), ring[d]);
    auto step = [&](int jc, Ops& o) {
        const int sj = base + jc;
        const int ta = ln.t0 + jc * CH;
        const int L = min(CH, ln.t0 + ln.T - ta);

#pragma unroll
        for (int q = 0; q < KT; ++q)
            mfma_each(mfma_zero(), [&](int i, int vv, float) { hT[vv * HP + (wave + 4 * q) * 16 + i] = from_f32<bf16>(hf[q][i % 4]); });
        if (wave == 0) sE[l] = l < L ? __expf(o.gl - o.g) : 0.f;
        lds_barrier();

        // The entering state, recorded as a contiguous slice.
        bf16* hcT_s = hcT + ((static_cast<long long>(sj) * V_h + h) * BV + v0) * BK;
        static_assert(BVT * BK / 8 == 256, "a 16-byte piece per thread");
        *reinterpret_cast<b128*>(hcT_s + (tid / 16) * BK + (tid % 16) * 8) =
            *reinterpret_cast<const b128*>(hT + (tid / 16) * HP + (tid % 16) * 8);

        // V' = U − W·S for the wavefront's 16 chunk rows.
        {
            const int m0 = wave * 16;
            mfma_cd c = mfma_zero();
#pragma unroll
            for (int st = 0; st < KS; ++st) c = mfma16x2(o.wf[st], load8(hT + (l % 16) * HP + st * 32 + g8), c);
            const bf16* uv = reinterpret_cast<const bf16*>(&o.uv);
            float vn[4], vs[4];
#pragma unroll
            for (int t = 0; t < 4; ++t) {
                const int i = m0 + 4 * (l / 16) + t;
                vn[t] = i < L ? to_f32(uv[t]) - c[t] : 0.f;
                vs[t] = vn[t] * sE[i];
            }
            const int vv = l % 16, i0 = m0 + 4 * (l / 16);
            *reinterpret_cast<mfma_ab*>(vnL + vv * CH + i0) = pack4(vn[0], vn[1], vn[2], vn[3]);
            *reinterpret_cast<mfma_ab*>(vnTs + vv * VP + i0) = pack4(vs[0], vs[1], vs[2], vs[3]);
        }
        lds_barrier();

        bf16* vnT_s = vnT + ((static_cast<long long>(sj) * V_h + h) * BV + v0) * CH;
        static_assert(BVT * CH / 8 == 128, "a 16-byte piece per thread of the first half");
        if (tid < BVT * CH / 8) reinterpret_cast<b128*>(vnT_s)[tid] = reinterpret_cast<const b128*>(vnL)[tid];

        // S ← exp(Γ_L)·S + K̂ᵀ·diag(exp(Γ_L − Γ))·V'.
        const float decay = __expf(o.gl);
#pragma unroll
        for (int q = 0; q < KT; ++q) {
            mfma_cd c = mfma_zero();
#pragma unroll
            for (int st = 0; st < CS; ++st) c = mfma16x2(o.kf[q][st], load8(vnTs + (l % 16) * VP + st * 32 + g8), c);
#pragma unroll
            for (int t = 0; t < 4; ++t) hf[q][t] = hf[q][t] * decay + c[t];
        }
        fetch(min(jc + DEPTH, chunks - 1), o);
    };
    int jc = 0;
    for (; jc + DEPTH <= chunks; jc += DEPTH) {
#pragma unroll
        for (int d = 0; d < DEPTH; ++d) step(jc + d, ring[d]);
    }
#pragma unroll
    for (int d = 0; d < DEPTH - 1; ++d)
        if (jc + d < chunks) step(jc + d, ring[d]);

    if (!folds(write_state, mask, ln.rl)) return;
#pragma unroll
    for (int q = 0; q < KT; ++q)
        mfma_each(mfma_zero(), [&](int i, int vv, float) {
            state[static_cast<long long>((wave + 4 * q) * 16 + i) * BV + v0 + vv] = from_f32<bf16>(hf[q][i % 4]);
        });
}

// Grid (slots, V_h), block 256: the chunk's output rows, f32 [n][V_h][BV]
// at row + row_offset. Wavefront w lands chunk rows [16w, 16w+16).
template <int BK, int BV>
__global__ __launch_bounds__(256) void ssm_gdn_chunk_o(
    const float* __restrict__ gcum, const bf16* __restrict__ qb, const bf16* __restrict__ kb,
    const bf16* __restrict__ vnT, const bf16* __restrict__ hcT, const i32* __restrict__ slot_ids,
    const u32* __restrict__ indptr, const int* __restrict__ commit_len, const int* __restrict__ begin_at,
    float* __restrict__ out, int lanes, int K_h, int V_h, const u32* __restrict__ win) {
    constexpr int CH = GDN_CH, BVT = 16, KS = BK / 32, CS = CH / 32;
    constexpr int SP = CH + 8;  // padded against the 16-row reads' bank conflicts
    __shared__ alignas(16) bf16 Sb[CH * SP];
    __shared__ float sG[CH];

    const int h = blockIdx.y, s = blockIdx.x;
    if (win != nullptr) lanes = min(lanes, static_cast<int>(win[2]));
    if (lanes <= 0) return;
    const int r = gdn_slot_lane(s, lanes, indptr, begin_at, win);
    const GdnLane ln = gdn_lane(r, indptr, slot_ids, commit_len, begin_at, win);
    if (ln.T <= 0 || ln.slot < 0) return;
    const int j = s - gdn_chunk_base(r, indptr, begin_at, win);
    if (j < 0 || j * CH >= ln.T) return;
    const int ta = ln.t0 + j * CH;
    const int L = min(CH, ln.t0 + ln.T - ta);
    const long long row0 = win != nullptr ? static_cast<long long>(win[1]) : 0;
    const int hk = h / (V_h / K_h);
    const int tid = threadIdx.x, wave = tid / kWave, l = lane(), g8 = 8 * (l / 16);
    if (tid < CH) sG[tid] = gcum[static_cast<long long>(ta + min(tid, L - 1)) * V_h + h];

    const int ld = K_h * BK;
    const bf16* q_s = qb + (static_cast<long long>(ta) * K_h + hk) * BK;
    const bf16* k_s = kb + (static_cast<long long>(ta) * K_h + hk) * BK;
    const bf16* hcT_s = hcT + (static_cast<long long>(s) * V_h + h) * BV * BK;
    const bf16* vnT_s = vnT + (static_cast<long long>(s) * V_h + h) * BV * CH;
    // The wavefront's 16 query rows as A fragments, kept across the slices.
    const int m0 = wave * 16;
    mfma_ab2 qf[KS];
#pragma unroll
    for (int st = 0; st < KS; ++st) qf[st] = load8(q_s + static_cast<long long>(m0 + l % 16) * ld + st * 32 + g8);
    mfma_cd sc[CH / 16];
#pragma unroll
    for (int n = 0; n < CH / 16; ++n) {
        sc[n] = mfma_zero();
#pragma unroll
        for (int st = 0; st < KS; ++st)
            sc[n] = mfma16x2(qf[st], load8(k_s + static_cast<long long>(n * 16 + l % 16) * ld + st * 32 + g8), sc[n]);
    }
    lds_barrier();

    // S = mask(exp(Γ_i − Γ_j)) ∘ Q̂·K̂ᵀ for the wavefront's rows.
#pragma unroll
    for (int n = 0; n < CH / 16; ++n)
        mfma_each(sc[n], [&](int i, int jj, float x) {
            i += m0;
            jj += n * 16;
            Sb[i * SP + jj] = from_f32<bf16>((i < L && jj <= i) ? x * __expf(sG[i] - sG[jj]) : 0.f);
        });
    // (each wavefront reads back only its own rows of S)
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront");
    __builtin_amdgcn_wave_barrier();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront");
    mfma_ab2 sf[CS];
#pragma unroll
    for (int st = 0; st < CS; ++st) sf[st] = load8(Sb + (m0 + l % 16) * SP + st * 32 + g8);

    float eg[4];
#pragma unroll
    for (int t = 0; t < 4; ++t) eg[t] = __expf(sG[m0 + 4 * (l / 16) + t]);
#pragma unroll 2
    for (int v0 = 0; v0 < BV; v0 += BVT) {
        mfma_cd o1 = mfma_zero(), o2 = mfma_zero();
#pragma unroll
        for (int st = 0; st < KS; ++st)
            o1 = mfma16x2(qf[st], load8(hcT_s + static_cast<long long>(v0 + l % 16) * BK + st * 32 + g8), o1);
#pragma unroll
        for (int st = 0; st < CS; ++st)
            o2 = mfma16x2(sf[st], load8(vnT_s + static_cast<long long>(v0 + l % 16) * CH + st * 32 + g8), o2);
#pragma unroll
        for (int t = 0; t < 4; ++t) {
            const int i = m0 + 4 * (l / 16) + t;
            if (i < L) out[((ta + i + row0) * V_h + h) * BV + v0 + l % 16] = o1[t] * eg[t] + o2[t];
        }
    }
}

// ------------------------------------------------- Kimi Delta Attention --

// KDA (Kimi Linear's mixer, and GLM's): the delta rule above with a decay
// per key channel instead of per head. Every head has its own q, k and v,
// all D wide; head h's state is St[v][k] (v over the value head, k over the
// key head), f32 at slab + slot·stride + h·D·D + v·D + k. One token, with q̂
// and k̂ the l2-normed query (also scaled by 1/√D) and key and a = exp(gate)
// per channel:
//
//   St[v][k] ← St[v][k]·a[k]
//   δ[v]     = (v[v] − Σ_k St[v][k]·k̂[k])·β
//   St[v][k] ← St[v][k] + k̂[k]·δ[v]
//   o[v]     = Σ_k St[v][k]·q̂[k]
//
// The state stays f32 throughout. The decode step and the chunk passes read
// the post-conv [q | k | v] row, the forget projection f and the β
// projection b where they stand (launch row + row_offset) and norm and gate
// them as they load; the token-by-token prefill reads them staged in f32
// (q̂, k̂, v, the gate [n][H][D] and β [n][H], at the launch row). Every
// recurrence lands its output at row + row_offset.


__device__ __forceinline__ float sigmoid(float x) { return 1.f / (1.f + __expf(-x)); }

// The log decay of one channel: with g = f + dt_bias and a = exp(A_log[h]),
// floor·sigmoid(a·g) where the decay has a floor (floor < 0), else
// −a·softplus(g).
__device__ __forceinline__ float kda_gate(float g, float a, float floor) {
    return floor < 0.f ? floor * sigmoid(a * g) : -a * (g > 20.f ? g : __logf(1.f + __expf(g)));
}

// [q | k | v] → q̂, k̂ (each head l2-normed over its D entries, eps inside the
// root) and v. Grid (rows, 3, ⌈H / 4·waves⌉): a plane a block row, four
// heads a wavefront.
constexpr int kKdaPrepHeads = 4;

template <class T, int BLOCK>
__global__ __launch_bounds__(BLOCK) void ssm_kda_qkv_prep(const T* __restrict__ mixed, float* __restrict__ q_out,
                                                          float* __restrict__ k_out, float* __restrict__ v_out,
                                                          int width, int D, float eps, const u32* __restrict__ win) {
    constexpr int WAVES = BLOCK / kWave;
    const int n = blockIdx.x, plane = blockIdx.y, l = lane();
    if (win != nullptr && n >= static_cast<int>(win[0])) return;
    const long long row = win != nullptr ? n + static_cast<long long>(win[1]) : n;
    const float scale = plane == 0 ? rsqrtf(static_cast<float>(D)) : 1.f;
    const T* src = mixed + row * 3 * width + static_cast<long long>(plane) * width;
    float* dst = (plane == 0 ? q_out : plane == 1 ? k_out : v_out) + static_cast<long long>(n) * width;
    for (int i = 0; i < kKdaPrepHeads; ++i) {
        const int h = (blockIdx.z * kKdaPrepHeads + i) * WAVES + static_cast<int>(threadIdx.x) / kWave;
        if (h >= width / D) return;
        const T* x = src + static_cast<long long>(h) * D;
        float* o = dst + static_cast<long long>(h) * D;
        if (plane == 2) {
            for (int d = l; d < D; d += kWave) o[d] = to_f32(x[d]);
            continue;
        }
        float ss = 0.f;
        for (int d = l; d < D; d += kWave) ss += to_f32(x[d]) * to_f32(x[d]);
        const float inv = rsqrtf(wave_sum(ss) + eps);
        for (int d = l; d < D; d += kWave) o[d] = to_f32(x[d]) * inv * scale;
    }
}

// The gates of each channel and β = sigmoid(b). Grid (rows, ⌈H·D / 4·block⌉),
// four adjacent channels a thread.
template <class T>
__global__ void ssm_kda_gate_beta(const T* __restrict__ f, const T* __restrict__ b, const float* __restrict__ a_log,
                                  const float* __restrict__ dt_bias, float* __restrict__ gate_out,
                                  float* __restrict__ beta_out, int N, int H, int D, float floor,
                                  const u32* __restrict__ win) {
    const int t = blockIdx.x;
    if (t >= N || (win != nullptr && t >= static_cast<int>(win[0]))) return;
    const long long row = win != nullptr ? t + static_cast<long long>(win[1]) : t;
    const int width = H * D;
    const int i = 4 * (blockIdx.y * blockDim.x + threadIdx.x);
    if (i < width) {
        const float a = __expf(a_log[i / D]);
#pragma unroll
        for (int u = 0; u < 4; ++u)
            gate_out[static_cast<long long>(t) * width + i + u] =
                kda_gate(to_f32(f[row * width + i + u]) + dt_bias[i + u], a, floor);
    }
    if (blockIdx.y == 0)
        for (int h = threadIdx.x; h < H; h += blockDim.x)
            beta_out[static_cast<long long>(t) * H + h] = sigmoid(to_f32(b[row * H + h]));
}

// The token kernels hold a state row on eight adjacent lanes, lane c the
// four-wide pieces at k = 4c + 32e, so a wavefront's read is eight whole
// 128-byte lines and a row's sums fold in three moves.
constexpr int kKdaRowLanes = 8;

template <int CTRL>
__device__ __forceinline__ float dpp_mov(float v) {
    return __builtin_bit_cast(float, __builtin_amdgcn_update_dpp(0, __builtin_bit_cast(int, v), CTRL, 0xf, 0xf, true));
}

// The sum over the eight lanes of a row, in every one of them.
__device__ __forceinline__ float row8_sum(float x) {
    x += dpp_mov<0xb1>(x);   // quad_perm [1, 0, 3, 2]
    x += dpp_mov<0x4e>(x);   // quad_perm [2, 3, 0, 1]
    x += dpp_mov<0x141>(x);  // row_half_mirror
    return x;
}

// One token of a row's recurrence: s the row's pieces, q, k, a the token's
// at the same k; returns o[v] (in every lane of the row).
template <int E>
__device__ __forceinline__ float kda_row_step(f32x4 (&s)[E], const f32x4 (&q)[E], const f32x4 (&k)[E],
                                              const f32x4 (&a)[E], float v, float beta) {
    float kv = 0.f;
#pragma unroll
    for (int e = 0; e < E; ++e)
#pragma unroll
        for (int u = 0; u < 4; ++u) {
            s[e][u] *= a[e][u];
            kv += s[e][u] * k[e][u];
        }
    const float delta = (v - row8_sum(kv)) * beta;
    float o = 0.f;
#pragma unroll
    for (int e = 0; e < E; ++e)
#pragma unroll
        for (int u = 0; u < 4; ++u) {
            s[e][u] += k[e][u] * delta;
            o += s[e][u] * q[e][u];
        }
    return row8_sum(o);
}

// A token's q̂, k̂ and exp(gate) at the lane's pieces.
template <int E>
struct KdaToken {
    f32x4 q[E], k[E], a[E];
};

// From rows of q̂, k̂ and the gate (`log`) or its exponential.
template <int E>
__device__ __forceinline__ KdaToken<E> kda_token(const float* q, const float* k, const float* g, int c,
                                                 bool log = true) {
    KdaToken<E> t;
#pragma unroll
    for (int e = 0; e < E; ++e) {
        const int at = 4 * c + 32 * e;
        t.q[e] = *reinterpret_cast<const f32x4*>(q + at);
        t.k[e] = *reinterpret_cast<const f32x4*>(k + at);
        const f32x4 x = *reinterpret_cast<const f32x4*>(g + at);
        t.a[e] = log ? f32x4{__expf(x[0]), __expf(x[1]), __expf(x[2]), __expf(x[3])} : x;
    }
    return t;
}

// The decode step's rows: `kda_row_step` over R rows at once, in fused pairs
// (v_pk_mul / v_pk_fma), each phase across every row before the next so the
// rows' chains interleave. The sums over a lane's pieces run in E pair
// accumulators, then over the row's eight lanes. Rounds once where the
// token kernels round twice.
template <int R, int E>
__device__ __forceinline__ void kda_rows_step(f32x4 (&s)[R][E], const f32x4 (&q)[E], const f32x4 (&k)[E],
                                              const f32x4 (&a)[E], const float (&v)[R], float beta,
                                              float (&y)[R]) {
    auto pairs = [](f32x2 (&x)[E]) {
#pragma unroll
        for (int w = 1; w < E; w <<= 1)
#pragma unroll
            for (int e = 0; e + w < E; e += 2 * w) x[e] += x[e + w];
        return x[0].x + x[0].y;
    };
    float d[R];
#pragma unroll
    for (int p = 0; p < R; ++p) {
        f32x2 kv[E];
#pragma unroll
        for (int e = 0; e < E; ++e) {
            const f32x2 lo = s[p][e].xy * a[e].xy, hi = s[p][e].zw * a[e].zw;
            s[p][e].xy = lo;
            s[p][e].zw = hi;
            kv[e] = __builtin_elementwise_fma(hi, k[e].zw, lo * k[e].xy);
        }
        d[p] = pairs(kv);
    }
#pragma unroll
    for (int p = 0; p < R; ++p) d[p] = (v[p] - row8_sum(d[p])) * beta;
#pragma unroll
    for (int p = 0; p < R; ++p) {
        const f32x2 d2 = {d[p], d[p]};
        f32x2 o[E];
#pragma unroll
        for (int e = 0; e < E; ++e) {
            const f32x2 lo = __builtin_elementwise_fma(k[e].xy, d2, s[p][e].xy);
            const f32x2 hi = __builtin_elementwise_fma(k[e].zw, d2, s[p][e].zw);
            s[p][e].xy = lo;
            s[p][e].zw = hi;
            o[e] = __builtin_elementwise_fma(hi, q[e].zw, lo * q[e].xy);
        }
        y[p] = pairs(o);
    }
#pragma unroll
    for (int p = 0; p < R; ++p) y[p] = row8_sum(y[p]);
}

// Decode: one token per row, straight from its projections. Grid (H, rows,
// D / ROWS), block 256: a block walks ROWS value rows of one head, eight
// lanes a row (lane c the pieces at k = 4c + 32e) and each lane group
// R = ROWS / 32 rows 32 apart, so a wavefront's read is eight whole
// consecutive rows. Each wavefront norms and gates the token's channels
// itself, two a lane, and hands them to its rows' lanes through its own
// slice of LDS: the wavefronts share nothing and meet at no barrier, each
// running from its reads to its stores on its own.
template <int D, int ROWS>
__global__ __launch_bounds__(256) void ssm_kda_step_batched(
    const bf16* __restrict__ mixed, const bf16* __restrict__ f, const bf16* __restrict__ b,
    const float* __restrict__ a_log, const float* __restrict__ dt_bias, float* __restrict__ state_base,
    const i32* __restrict__ slot_ids, long long slot_stride, float* __restrict__ out, int H, float eps, float floor,
    bool write_state, const u8* __restrict__ mask, const u32* __restrict__ win) {
    constexpr int E = D / 32, RPP = 256 / kKdaRowLanes, R = ROWS / RPP, WAVES = 256 / kWave;
    static_assert(D % 32 == 0 && D <= 2 * kWave && ROWS % RPP == 0 && D % ROWS == 0,
                  "whole pieces, two channels a lane, whole passes");
    const int h = blockIdx.x, r = blockIdx.y;
    if (win != nullptr && r >= static_cast<int>(win[0])) return;
    const long long row = win != nullptr ? r + static_cast<long long>(win[1]) : r;
    const int rl = win != nullptr ? r + static_cast<int>(win[3]) : r;
    const int slot = slot_ids[rl];
    if (slot < 0) return;
    const int tid = threadIdx.x, wave = tid / kWave, l = lane();
    const int c = tid % kKdaRowLanes;
    const int v0 = blockIdx.z * ROWS + tid / kKdaRowLanes;
    const long long width = static_cast<long long>(H) * D;
    float* head = state_base + static_cast<long long>(slot) * slot_stride + static_cast<long long>(h) * D * D;
    const bf16* q = mixed + row * 3 * width + static_cast<long long>(h) * D;

    // The token's reads first, so waiting on them waits on no state read;
    // then the state's. Lane l holds channels 2l, 2l + 1.
    const bool mine = 2 * l < D;
    const int ch = mine ? 2 * l : 0;
    const u32 qw = *reinterpret_cast<const u32*>(q + ch), kw = *reinterpret_cast<const u32*>(q + width + ch);
    const u32 fw = *reinterpret_cast<const u32*>(f + row * width + static_cast<long long>(h) * D + ch);
    const float2 dt = *reinterpret_cast<const float2*>(dt_bias + h * D + ch);
    float vv[R];
#pragma unroll
    for (int p = 0; p < R; ++p) vv[p] = to_f32(q[2 * width + v0 + RPP * p]);
    const float braw = to_f32(b[row * H + h]), al = a_log[h];
    f32x4 s[R][E];
#pragma unroll
    for (int p = 0; p < R; ++p)
#pragma unroll
        for (int e = 0; e < E; ++e)
            s[p][e] = state_load(reinterpret_cast<const f32x4*>(head + static_cast<long long>(v0 + RPP * p) * D + 4 * c + 32 * e));
    const bool fold = folds_uniform(write_state, mask, rl);

    // q̂, k̂ and the decays as the staging kernels compute them, into this
    // wavefront's slice.
    __shared__ alignas(16) float tq[WAVES][D], tk[WAVES][D], ta[WAVES][D];
    const float q0 = mine ? bf16_lo(qw) : 0.f, q1 = mine ? bf16_hi(qw) : 0.f;
    const float k0 = mine ? bf16_lo(kw) : 0.f, k1 = mine ? bf16_hi(kw) : 0.f;
    const float rq = rsqrtf(wave_sum(q0 * q0 + q1 * q1) + eps), rk = rsqrtf(wave_sum(k0 * k0 + k1 * k1) + eps);
    const float a = __expf(al), rd = rsqrtf(static_cast<float>(D));
    if (mine) {
        *reinterpret_cast<float2*>(&tq[wave][ch]) = float2{q0 * rq * rd, q1 * rq * rd};
        *reinterpret_cast<float2*>(&tk[wave][ch]) = float2{k0 * rk, k1 * rk};
        *reinterpret_cast<float2*>(&ta[wave][ch]) = float2{__expf(kda_gate(bf16_lo(fw) + dt.x, a, floor)),
                                                          __expf(kda_gate(bf16_hi(fw) + dt.y, a, floor))};
    }
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront", "local");
    __builtin_amdgcn_wave_barrier();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront", "local");
    const KdaToken<E> tok = kda_token<E>(tq[wave], tk[wave], ta[wave], c, false);
    const float beta = sigmoid(braw);

    float* o = out + (row * H + h) * D;
    float y[R];
    kda_rows_step<R, E>(s, tok.q, tok.k, tok.a, vv, beta, y);
#pragma unroll
    for (int p = 0; p < R; ++p) {
        if (c == 0) o[v0 + RPP * p] = y[p];
        if (fold) {
#pragma unroll
            for (int e = 0; e < E; ++e)
                state_store(reinterpret_cast<f32x4*>(head + static_cast<long long>(v0 + RPP * p) * D + 4 * c + 32 * e), s[p][e]);
        }
    }
}

// Prefill, token by token: grid (lanes, H), block 8·D, the head's state in
// registers a value row to eight lanes, the next token's operands loaded
// while this one folds. The fallback for heads the chunk passes are not
// built for, and their reference.
template <int D>
__global__ __launch_bounds__(8 * D) void ssm_kda_chunked_batched(
    const float* __restrict__ q_norm, const float* __restrict__ k_norm, const float* __restrict__ v,
    const float* __restrict__ gate, const float* __restrict__ beta, float* __restrict__ state_base,
    const i32* __restrict__ slot_ids, const u32* __restrict__ indptr, long long slot_stride,
    float* __restrict__ out, int H, int D_stated, bool write_state, const int* __restrict__ commit_len,
    const u8* __restrict__ mask, const int* __restrict__ begin_at, const u32* __restrict__ win) {
    constexpr int E = D / 32;
    static_assert(D % 32 == 0 && 8 * D <= 1024, "whole pieces, a row a lane group");
    (void)D_stated;
    const int r = blockIdx.x, h = blockIdx.y;
    if (win != nullptr && r >= static_cast<int>(win[2])) return;
    const int rl = win != nullptr ? r + static_cast<int>(win[3]) : r;
    const long long row0 = win != nullptr ? static_cast<long long>(win[1]) : 0;
    const Span sp = span_of(indptr, r, rl, commit_len, begin_at);
    if (sp.n <= 0) return;
    const int slot = slot_ids[rl];
    if (slot < 0) return;
    const int c = threadIdx.x % kKdaRowLanes, vr = threadIdx.x / kKdaRowLanes;
    float* st = state_base + static_cast<long long>(slot) * slot_stride + static_cast<long long>(h) * D * D +
                static_cast<long long>(vr) * D;

    f32x4 s[E];
#pragma unroll
    for (int e = 0; e < E; ++e) s[e] = *reinterpret_cast<const f32x4*>(st + 4 * c + 32 * e);

    auto fetch = [&](int t, KdaToken<E>& tk, float& vt, float& bt) {
        const long long th = static_cast<long long>(sp.t0 + t) * H + h;
        tk = kda_token<E>(q_norm + th * D, k_norm + th * D, gate + th * D, c);
        vt = v[th * D + vr];
        bt = beta[th];
    };
    KdaToken<E> tk;
    float vt, bt;
    fetch(0, tk, vt, bt);
    for (int t = 0; t < sp.n; ++t) {
        KdaToken<E> nx;
        float vn, bn;
        fetch(min(t + 1, sp.n - 1), nx, vn, bn);
        const float y = kda_row_step<E>(s, tk.q, tk.k, tk.a, vt, bt);
        if (c == 0) out[((sp.t0 + t + row0) * H + h) * D + vr] = y;
        tk = nx;
        vt = vn;
        bt = bn;
    }
    if (!folds(write_state, mask, rl)) return;
#pragma unroll
    for (int e = 0; e < E; ++e) *reinterpret_cast<f32x4*>(st + 4 * c + 32 * e) = s[e];
}

// ------------------------------------------- KDA prefill (chunk-parallel) --

// The gated delta rule's three chunk passes and chunk slots, with the decay
// per channel (FLA's chunk_kda). Per head, chunk tokens 0 ≤ i < L, and Γ_i =
// Σ_{t ≤ i} gate_t a vector over the key channels:
//
//   A[i][j] = β_i·Σ_c k̂_ic·k̂_jc·exp(Γ_ic − Γ_jc)   (j < i),   T = (I + A)⁻¹
//   P[i][j] =     Σ_c q̂_ic·k̂_jc·exp(Γ_ic − Γ_jc)   (j ≤ i)
//   W = T·diag(β)·(K̂ ∘ exp Γ)        U = T·diag(β)·V
//   V' = U − W·S                     (S = Stᵀ, the state entering the chunk)
//   O  = (Q̂ ∘ exp Γ)·S + P·V'
//   S ← diag(exp Γ_L)·S + (K̂ ∘ exp(Γ_L − Γ))ᵀ·V'
//
// A split exp(Γ_i)·exp(−Γ_j) would overflow under a steep decay, so A and P
// are matrix-core products of factors referenced to the first token n of
// row i's 16-row block, (x_i ∘ exp(Γ_i − Γ_n))·(k̂_j ∘ exp(Γ_n − Γ_j))ᵀ:
// both at most one between blocks, and on the diagonal the second at most
// the block's own decay. A chunk where some channel decays past e^80 within
// a block (as a decay without a floor may) sums its diagonal blocks per
// channel instead, every exponential then of a difference no greater than
// zero. `wu` builds W, U, P and the gated operands per chunk; `ho` walks
// each request's chunks carrying S in f32, landing the output as it goes.

// f(Index<0>{}) .. f(Index<N − 1>{}), in order: a loop whose index must be
// a constant (it picks registers) however long the body.
template <int N>
struct Index {
    static constexpr int value = N;
};

template <int N, int I = 0, class F>
__device__ __forceinline__ void unrolled(F&& f) {
    if constexpr (I < N) {
        f(Index<I>{});
        unrolled<N, I + 1>(f);
    }
}

// Cell j of the row pair (p, 15 − p) of a diagonal block: j < 15 the 15 of A
// (m < i), the rest the 17 of P (m ≤ i).
struct KdaCell {
    int i, m;
    bool p;
};

__device__ __forceinline__ constexpr KdaCell kda_cell(int pair, int j) {
    if (j < 15) return j < pair ? KdaCell{pair, j, false} : KdaCell{15 - pair, j - pair, false};
    j -= 15;
    return j <= pair ? KdaCell{pair, j, true} : KdaCell{15 - pair, j - pair - 1, true};
}

// The widest decay, in nats, within a 16-row block that the matrix core's
// referenced factors carry on the diagonal (e^80 against bf16's e^88).
constexpr float kKdaSteep = 80.f;

// The lane's partner across bit HALF of the lane index for the halving
// exchange: any pairing across that bit within the group the earlier steps
// left alike will do, so the steps within a row of 16 are DPP moves.
template <int HALF>
__device__ __forceinline__ float scatter_partner(float v) {
    if constexpr (HALF == 16) return __shfl_xor(v, 16, kWave);
    else if constexpr (HALF == 8) return dpp_mov<0x140>(v);  // row_mirror: l ↔ 15 − l
    else if constexpr (HALF == 4) return dpp_mov<0x141>(v);  // row_half_mirror: l ↔ 7 − l
    else if constexpr (HALF == 2) return dpp_mov<0x4e>(v);   // quad_perm [2, 3, 0, 1]
    else return dpp_mov<0xb1>(v);                            // quad_perm [1, 0, 3, 2]
}

// The wavefront's sum of each of 32 values, lane l < 32 (and l + 32) left
// holding value l: a halving exchange, each step sending the half the lane
// gives up to the partner that keeps it. (Constant indices from the start,
// so the halves are selected between registers and not by an index into
// them.)
__device__ __forceinline__ float wave_scatter_sum(float (&x)[32]) {
    const int l = lane();
    unrolled<5>([&](auto sc) {
        constexpr int half = 16 >> decltype(sc)::value;
        const bool up = (l & half) != 0;
        unrolled<half>([&](auto jc) {
            constexpr int j = decltype(jc)::value;
            const float give = up ? x[j] : x[j + half];
            const float keep = up ? x[j + half] : x[j];
            x[j] = keep + scatter_partner<half>(give);
        });
    });
    return x[0] + __shfl_xor(x[0], 32, kWave);
}

// Grid (slots, H), block 512, straight from the post-conv rows and the
// projections: thread (b, c) holds rows 16b..16b+15 of key channel c. Lands,
// per chunk slot, the bf16 planes (K̂ ∘ exp(Γ_L − Γ))ᵀ (kT [s][H][D][CH]), W
// ([s][H][CH][D]), Uᵀ ([s][H][D][CH]), Q̂ ∘ exp Γ (qg [s][H][CH][D]) and P
// ([s][H][CH][CH]), all zero past the chunk's rows, and exp(Γ_L) (gl, f32
// [s][H][D]). The planes leave through LDS in whole 16-byte pieces.
template <int D>
__global__ __launch_bounds__(512) void ssm_kda_chunk_wu(
    const bf16* __restrict__ mixed, const bf16* __restrict__ f, const bf16* __restrict__ b,
    const float* __restrict__ a_log, const float* __restrict__ dt_bias, const i32* __restrict__ slot_ids,
    const u32* __restrict__ indptr, const int* __restrict__ commit_len, const int* __restrict__ begin_at,
    bf16* __restrict__ kT, bf16* __restrict__ w_out, bf16* __restrict__ uT, bf16* __restrict__ qg,
    bf16* __restrict__ p_out, float* __restrict__ gl, int lanes, int H, float eps, float floor,
    const u32* __restrict__ win) {
    constexpr int CH = GDN_CH, THREADS = 512, WAVES = THREADS / kWave, NB = CH / 16;
    static_assert(THREADS == NB * D, "a thread a channel of a 16-row block");
    static_assert(kWave != 64 || D == 2 * kWave, "two wavefronts a block row");
    // Row strides padded by 8 bytes so the matrix core's 16-row reads fall
    // in distinct banks; kT's staging rows by 16 so its pieces stay aligned.
    constexpr int KP = D + 4, TP = CH + 4, KTP = CH + 8;
    static_assert(D * KTP <= 6 * 16 * KP && CH * D * 2 <= CH * CH * 4, "the staging fits its buffers");
    __shared__ float A[CH][CH];                  // first q̂ ∘ exp Γ in bf16
    __shared__ alignas(16) bf16 X[6 * 16 * KP];  // kTᵀ; a block row's factors; K̂ ∘ exp Γ; diag(β)·V
    __shared__ alignas(16) bf16 Tb[CH * TP];     // P; T's scratch; T·diag(β); T
    __shared__ float Gsum[NB][D], Gref[NB][D];
    __shared__ float Pd[NB][16][16];
    __shared__ float NS[WAVES][32];
    __shared__ float B[CH];
    __shared__ int Steep[WAVES];

    const int h = blockIdx.y, s = blockIdx.x;
    if (win != nullptr) lanes = min(lanes, static_cast<int>(win[2]));
    if (lanes <= 0) return;
    const int r = gdn_slot_lane(s, lanes, indptr, begin_at, win);
    const GdnLane ln = gdn_lane(r, indptr, slot_ids, commit_len, begin_at, win);
    if (ln.T <= 0 || ln.slot < 0) return;
    const int j = s - gdn_chunk_base(r, indptr, begin_at, win);
    if (j < 0 || j * CH >= ln.T) return;
    const int ta = ln.t0 + j * CH;
    const int L = min(CH, ln.t0 + ln.T - ta);
    const long long row0 = win != nullptr ? static_cast<long long>(win[1]) : 0;
    const int tid = threadIdx.x, wave = tid / kWave, l = lane();
    const int rq = tid / D, c = tid % D;
    const long long sh = static_cast<long long>(s) * H + h, width = static_cast<long long>(H) * D;

    // Every load of the chunk issued before any is used (the gate's
    // branches would otherwise hold each row's loads behind the last row's
    // math); rows past L read the last row and keep zero (a zero gate holds
    // Γ at Γ_L).
    bf16 qx[16], kx[16], fx[16], vr[16];
#pragma unroll
    for (int t = 0; t < 16; ++t) {
        const long long row = row0 + ta + min(16 * rq + t, L - 1);
        const bf16* x = mixed + row * 3 * width + h * D + c;
        qx[t] = x[0];
        kx[t] = x[width];
        vr[t] = x[2 * width];
        fx[t] = f[row * width + h * D + c];
    }
    const float a = __expf(a_log[h]), dt = dt_bias[h * D + c];
    const float bx = to_f32(b[(row0 + ta + min(tid % CH, L - 1)) * H + h]);
    float kr[16], qr[16], G[16];
#pragma unroll
    for (int t = 0; t < 16; ++t) {
        const bool live = 16 * rq + t < L;
        qr[t] = live ? to_f32(qx[t]) : 0.f;
        kr[t] = live ? to_f32(kx[t]) : 0.f;
        G[t] = live ? kda_gate(to_f32(fx[t]) + dt, a, floor) : 0.f;
        if (!live) vr[t] = from_f32<bf16>(0.f);
    }
    if (tid < CH) B[tid] = tid < L ? sigmoid(bx) : 0.f;

    // The rows' norms: the wavefront's 64 channels of each, then the
    // block's two wavefronts.
    {
        float x[32];
#pragma unroll
        for (int t = 0; t < 16; ++t) {
            x[t] = qr[t] * qr[t];
            x[16 + t] = kr[t] * kr[t];
        }
        const float ss = wave_scatter_sum(x);
        if (l < 32) NS[wave][l] = ss;
    }
    // Γ: the block's own scan, then the blocks before it.
#pragma unroll
    for (int t = 1; t < 16; ++t) G[t] += G[t - 1];
    Gsum[rq][c] = G[15];
    lds_barrier();
    float pre = 0.f, last = 0.f;
#pragma unroll
    for (int bb = 0; bb < NB; ++bb) {
        if (bb == rq) pre = last;
        last += Gsum[bb][c];
    }
    const float scale = rsqrtf(static_cast<float>(D));
#pragma unroll
    for (int t = 0; t < 16; ++t) {
        G[t] += pre;
        qr[t] = qr[t] * rsqrtf(NS[2 * rq][t] + NS[2 * rq + 1][t] + eps) * scale;
        kr[t] = kr[t] * rsqrtf(NS[2 * rq][16 + t] + NS[2 * rq + 1][16 + t] + eps);
    }
    Gref[rq][c] = G[0];
    // Whether some channel decays more than e^80 within a 16-row block, past
    // which a factor referenced to the block's first row would overflow on
    // the diagonal: such a chunk sums its diagonal blocks per channel.
    {
        const bool steep = __any(G[0] - G[15] > kKdaSteep);
        if (l == 0) Steep[wave] = steep;
    }

    // The gated planes: kT staged as its rows (the thread's 16 cells of row c
    // adjacent), q̂ ∘ exp Γ as its.
    {
        float x[16];
#pragma unroll
        for (int t = 0; t < 16; ++t) x[t] = kr[t] * __expf(last - G[t]);
        mfma_ab* dst = reinterpret_cast<mfma_ab*>(X + c * KTP + 16 * rq);
#pragma unroll
        for (int t = 0; t < 4; ++t) dst[t] = pack4(x[4 * t], x[4 * t + 1], x[4 * t + 2], x[4 * t + 3]);
    }
    bf16* Qs = reinterpret_cast<bf16*>(&A[0][0]);
#pragma unroll
    for (int t = 0; t < 16; ++t) Qs[(16 * rq + t) * D + c] = from_f32<bf16>(qr[t] * __expf(G[t]));
    if (rq == 0) gl[sh * D + c] = __expf(last);
    lds_barrier();
    bool steep = false;
#pragma unroll
    for (int w = 0; w < WAVES; ++w) steep |= Steep[w] != 0;
    static_assert(D * CH / 8 == 2 * THREADS, "two pieces a thread of each plane");
#pragma unroll
    for (int u = 0; u < 2; ++u) {
        const int e = tid + u * THREADS;
        *reinterpret_cast<b128*>(kT + (sh * D + e / 8) * CH + (e % 8) * 8) =
            *reinterpret_cast<const b128*>(X + (e / 8) * KTP + (e % 8) * 8);
        *reinterpret_cast<b128*>(qg + (sh * CH + e / (D / 8)) * D + (e % (D / 8)) * 8) =
            *reinterpret_cast<const b128*>(Qs + (e / (D / 8)) * D + (e % (D / 8)) * 8);
    }

    // A steep chunk's diagonal blocks, a row pair at a time: the thread's
    // channel's share of the pair's 32 cells, summed over the wavefront's 64
    // channels, then the block's two wavefronts in turn.
    float diag[8] = {};
    if (steep) unrolled<8>([&](auto pc) {
        constexpr int p = decltype(pc)::value;
        float x[32];
        unrolled<32>([&](auto jc) {
            constexpr KdaCell e = kda_cell(p, decltype(jc)::value);
            const float d = e.i == e.m ? 1.f : __expf(G[e.i] - G[e.m]);
            x[decltype(jc)::value] = (e.p ? qr[e.i] : kr[e.i]) * kr[e.m] * d;
        });
        diag[p] = wave_scatter_sum(x);
    });
    lds_barrier();
    for (int e = tid; e < CH * CH; e += THREADS) (&A[0][0])[e] = 0.f;
    for (int e = tid; e < NB * 256; e += THREADS) (&Pd[0][0][0])[e] = 0.f;
    lds_barrier();
    auto land = [&](bool add) {
        if (!steep || l >= 32 || (wave % 2 == 1) != add) return;
#pragma unroll
        for (int p = 0; p < 8; ++p) {
            const KdaCell e = kda_cell(p, l);
            const int i = 16 * rq + e.i;
            float& cell = e.p ? Pd[rq][e.i][e.m] : A[i][16 * rq + e.m];
            const float x = e.p ? diag[p] : diag[p] * B[i];
            cell = add ? cell + x : x;
        }
    };
    land(false);
    lds_barrier();
    land(true);

    // Block row bi's factors referenced to its first token, then its A and P
    // tiles against each block before it (and, unless the chunk is steep,
    // its own), a tile a wavefront. P lands in Tb.
    bf16* Lk = X;
    bf16* Lq = X + 16 * KP;
    bf16* R = X + 32 * KP;
#pragma unroll
    for (int bi = 0; bi < NB; ++bi) {
        if (bi == 0 && steep) continue;
        const int tiles = steep ? bi : bi + 1;
        if (rq == bi) {
#pragma unroll
            for (int t = 0; t < 16; ++t) {
                const float d = __expf(G[t] - G[0]);
                Lk[t * KP + c] = from_f32<bf16>(kr[t] * d);
                Lq[t * KP + c] = from_f32<bf16>(qr[t] * d);
                if (!steep) R[(16 * rq + t) * KP + c] = from_f32<bf16>(kr[t] * __expf(G[0] - G[t]));
            }
        } else if (rq < bi) {
            const float gn = Gref[bi][c];
#pragma unroll
            for (int t = 0; t < 16; ++t) R[(16 * rq + t) * KP + c] = from_f32<bf16>(kr[t] * __expf(gn - G[t]));
        }
        lds_barrier();
        if (wave < 2 * tiles) {
            const bool pk = wave >= tiles;
            const int bj = wave % tiles;
            const mfma_cd x = mfma_tile<D>(pk ? Lq : Lk, KP, R + 16 * bj * KP, KP, mfma_zero());
            mfma_each(x, [&](int i, int m, float y) {
                const bool keep = bj < bi || (pk ? m <= i : m < i);
                i += 16 * bi;
                m += 16 * bj;
                if (pk) Tb[i * TP + m] = from_f32<bf16>(keep ? y : 0.f);
                else A[i][m] = keep ? y * B[i] : 0.f;
            });
        }
        lds_barrier();
    }

    // P's diagonal blocks (of a steep chunk) and the zeros above them, then P out.
    for (int e = tid; e < CH * CH; e += THREADS) {
        const int i = e / CH, m = e % CH;
        if (m / 16 == i / 16) {
            if (steep) Tb[i * TP + m] = from_f32<bf16>(Pd[i / 16][i % 16][m % 16]);
        } else if (m > i) {
            Tb[i * TP + m] = from_f32<bf16>(0.f);
        }
    }
    lds_barrier();
    static_assert(CH * CH / 4 == 2 * THREADS, "two 8-byte pieces a thread");
#pragma unroll
    for (int u = 0; u < 2; ++u) {
        const int e = tid + u * THREADS, i = e / (CH / 4), m = (e % (CH / 4)) * 4;
        *reinterpret_cast<mfma_ab*>(p_out + (sh * CH + i) * CH + m) = load4(Tb + i * TP + m);
    }

    // T = (I + A)⁻¹ in place, its scratch in Tb (once P has left it).
    unit_lower_inverse<CH>(A, reinterpret_cast<float*>(Tb));

    // The B operand (k j, column n) from row-major rows at stride KP.
    auto b_rows = [&](const bf16* m, int k0, int n0) {
        const int k = k0 + 4 * (l / 16), n = n0 + l % 16;
        bf16 x[4];
#pragma unroll
        for (int t = 0; t < 4; ++t) x[t] = m[(k + t) * KP + n];
        return *reinterpret_cast<const mfma_ab*>(x);
    };

    // W = T·diag(β)·(K̂ ∘ exp Γ).
    for (int e = tid; e < CH * CH; e += THREADS) {
        const int i = e / CH, jj = e % CH;
        Tb[i * TP + jj] = from_f32<bf16>((i < L && jj < L) ? A[i][jj] * B[jj] : 0.f);
    }
#pragma unroll
    for (int t = 0; t < 16; ++t) X[(16 * rq + t) * KP + c] = from_f32<bf16>(kr[t] * __expf(G[t]));
    lds_barrier();
    // As Wᵀ = (K̂ ∘ exp Γ)ᵀ·(T·diag)ᵀ, so a lane's four cells are adjacent keys of one row.
    for (int tile = wave; tile < 4 * (D / 16); tile += WAVES) {
        const int k0r = (tile / 4) * 16, i0 = (tile % 4) * 16;
        mfma_cd x = mfma_zero();
#pragma unroll
        for (int k0 = 0; k0 < CH; k0 += 16)
            x = mfma16(b_rows(X, k0, k0r), load4(Tb + (i0 + l % 16) * TP + k0 + 4 * (l / 16)), x);
        *reinterpret_cast<mfma_ab*>(w_out + (sh * CH + i0 + l % 16) * D + k0r + 4 * (l / 16)) =
            pack4(x[0], x[1], x[2], x[3]);
    }
    lds_barrier();

    // U = T·diag(β)·V.
    for (int e = tid; e < CH * CH; e += THREADS) {
        const int i = e / CH, jj = e % CH;
        Tb[i * TP + jj] = from_f32<bf16>((i < L && jj < L) ? A[i][jj] : 0.f);
    }
#pragma unroll
    for (int t = 0; t < 16; ++t) X[(16 * rq + t) * KP + c] = from_f32<bf16>(to_f32(vr[t]) * B[16 * rq + t]);
    lds_barrier();
    for (int tile = wave; tile < 4 * (D / 16); tile += WAVES) {
        const int m0 = (tile / (D / 16)) * 16, n0 = (tile % (D / 16)) * 16;
        mfma_cd x = mfma_zero();
#pragma unroll
        for (int k0 = 0; k0 < CH; k0 += 16)
            x = mfma16(load4(Tb + (m0 + l % 16) * TP + k0 + 4 * (l / 16)), b_rows(X, k0, n0), x);
        // The lane's four rows of one column are adjacent in Uᵀ.
        *reinterpret_cast<mfma_ab*>(uT + (sh * D + n0 + l % 16) * CH + m0 + 4 * (l / 16)) =
            pack4(x[0], x[1], x[2], x[3]);
    }
}

// Grid (lanes · H · D/16), block 256: one 16-column slice of one head's
// state S through every chunk of one request, in f32 registers in the matrix
// core's D layout (wavefront w holds key rows [16w, 16w+16) and
// [16(w+4), 16(w+4)+16)), as `ssm_gdn_chunk_h` walks it, with the decay a
// row's own, landing each chunk's output for its slice on the way: O =
// (Q̂ ∘ exp Γ)·S + P·V' from the entering S and V' it holds in LDS,
// wavefront w's chunk rows [16w, 16w+16), so neither is recorded (a separate
// output pass wrote and read back 790 MiB of them at a 16K-token prefill). f32
// [n][H][D] at row + row_offset.
template <int D>
__global__ __launch_bounds__(256) void ssm_kda_chunk_ho(
    const bf16* __restrict__ kT, const bf16* __restrict__ w, const bf16* __restrict__ uT,
    const float* __restrict__ gl, const bf16* __restrict__ qg, const bf16* __restrict__ p,
    float* __restrict__ out, float* __restrict__ state_base,
    const i32* __restrict__ slot_ids, const u32* __restrict__ indptr, long long slot_stride,
    const int* __restrict__ commit_len, const int* __restrict__ begin_at, bool write_state,
    const u8* __restrict__ mask, int lanes, int H, const u32* __restrict__ win) {
    constexpr int CH = GDN_CH, BVT = 16, KT = D / 16 / 4, KS = D / 32, CS = CH / 32;
    // LDS rows padded by 16 bytes against the 16-row reads' bank conflicts.
    constexpr int HP = D + 8, VP = CH + 8;
    static_assert(D % 64 == 0 && BVT * D / 8 == 256, "four wavefronts split the key rows; a piece a thread");
    __shared__ alignas(16) bf16 hT[BVT * HP];
    __shared__ alignas(16) bf16 vn[BVT * VP];

    const int vt = static_cast<int>(blockIdx.x) % (D / BVT);
    const int h = (static_cast<int>(blockIdx.x) / (D / BVT)) % H;
    const int r = static_cast<int>(blockIdx.x) / ((D / BVT) * H);
    if (win != nullptr) lanes = min(lanes, static_cast<int>(win[2]));
    if (r >= lanes) return;
    const GdnLane ln = gdn_lane(r, indptr, slot_ids, commit_len, begin_at, win);
    if (ln.T <= 0 || ln.slot < 0) return;
    const int v0 = vt * BVT;
    const long long row0 = win != nullptr ? static_cast<long long>(win[1]) : 0;
    const int tid = threadIdx.x, wave = tid / kWave, l = lane();
    // The lane's cells (key rows k(q)..k(q)+3, value column v0 + l % 16) are
    // four adjacent floats of St.
    float* state = state_base + static_cast<long long>(ln.slot) * slot_stride + static_cast<long long>(h) * D * D +
                   static_cast<long long>(v0 + l % 16) * D;
    auto key = [&](int q) { return (wave + 4 * q) * 16 + 4 * (l / 16); };

    float hf[KT][4];
#pragma unroll
    for (int q = 0; q < KT; ++q) {
        const f32x4 x = *reinterpret_cast<const f32x4*>(state + key(q));
#pragma unroll
        for (int t = 0; t < 4; ++t) hf[q][t] = x[t];
    }

    // One chunk's operands for this lane: W's A fragments for the
    // wavefront's 16 rows, U at its D cells, the gated k̂ᵀ's A fragments for
    // its key rows and the decay of those rows.
    struct Ops {
        mfma_ab2 wf[KS];
        mfma_ab uv;
        mfma_ab2 kf[KT][CS];
        f32x4 dk[KT];
    };
    const int base = gdn_chunk_base(r, indptr, begin_at, win);
    const int chunks = (ln.T + CH - 1) / CH;
    const int g8 = 8 * (l / 16), m0 = wave * 16;
    auto fetch = [&](int jc, Ops& o) {
        const long long sh = static_cast<long long>(base + jc) * H + h;
        const bf16* wr = w + (sh * CH + m0 + l % 16) * D + g8;
#pragma unroll
        for (int st = 0; st < KS; ++st) o.wf[st] = load8(wr + st * 32);
        o.uv = load4(uT + (sh * D + v0 + l % 16) * CH + m0 + 4 * (l / 16));
        const bf16* kT_s = kT + sh * D * CH;
#pragma unroll
        for (int q = 0; q < KT; ++q) {
#pragma unroll
            for (int st = 0; st < CS; ++st) o.kf[q][st] = load8(kT_s + ((wave + 4 * q) * 16 + l % 16) * CH + st * 32 + g8);
            o.dk[q] = *reinterpret_cast<const f32x4*>(gl + sh * D + key(q));
        }
    };

    // A ring of DEPTH chunks in flight: two keep the walk at two wavefronts a
    // SIMD (three needed more registers than that leaves).
    constexpr int DEPTH = 2;
    Ops ring[DEPTH];
#pragma unroll
    for (int d = 0; d < DEPTH; ++d) fetch(min(d, chunks - 1), ring[d]);
    auto step = [&](int jc, Ops& o) {
        const long long sh = static_cast<long long>(base + jc) * H + h;
        const int ta = ln.t0 + jc * CH;
        const int L = min(CH, ln.t0 + ln.T - ta);

#pragma unroll
        for (int q = 0; q < KT; ++q)
            mfma_each(mfma_zero(), [&](int i, int vv, float) { hT[vv * HP + (wave + 4 * q) * 16 + i] = from_f32<bf16>(hf[q][i % 4]); });
        lds_barrier();

        // The wavefront's rows of Q̂ ∘ exp Γ and of P as A fragments.
        mfma_ab2 qf[KS], pf[CS];
#pragma unroll
        for (int st = 0; st < KS; ++st) qf[st] = load8(qg + (sh * CH + m0 + l % 16) * D + st * 32 + g8);
#pragma unroll
        for (int st = 0; st < CS; ++st) pf[st] = load8(p + (sh * CH + m0 + l % 16) * CH + st * 32 + g8);

        // V' = U − W·S for the wavefront's 16 chunk rows.
        {
            mfma_cd x = mfma_zero();
#pragma unroll
            for (int st = 0; st < KS; ++st) x = mfma16x2(o.wf[st], load8(hT + (l % 16) * HP + st * 32 + g8), x);
            const bf16* uv = reinterpret_cast<const bf16*>(&o.uv);
            const int i0 = m0 + 4 * (l / 16);
            float y[4];
#pragma unroll
            for (int t = 0; t < 4; ++t) y[t] = i0 + t < L ? to_f32(uv[t]) - x[t] : 0.f;
            *reinterpret_cast<mfma_ab*>(vn + (l % 16) * VP + i0) = pack4(y[0], y[1], y[2], y[3]);
        }
        // (Q̂ ∘ exp Γ)·S, from hT before the barrier lets the next chunk at it.
        mfma_cd out_c = mfma_zero();
#pragma unroll
        for (int st = 0; st < KS; ++st) out_c = mfma16x2(qf[st], load8(hT + (l % 16) * HP + st * 32 + g8), out_c);
        lds_barrier();

        // + P·V'.
#pragma unroll
        for (int st = 0; st < CS; ++st) out_c = mfma16x2(pf[st], load8(vn + (l % 16) * VP + st * 32 + g8), out_c);
#pragma unroll
        for (int t = 0; t < 4; ++t) {
            const int i = m0 + 4 * (l / 16) + t;
            if (i < L) out[((ta + i + row0) * H + h) * D + v0 + l % 16] = out_c[t];
        }

        // S ← diag(exp Γ_L)·S + (K̂ ∘ exp(Γ_L − Γ))ᵀ·V'.
#pragma unroll
        for (int q = 0; q < KT; ++q) {
            mfma_cd x = mfma_zero();
#pragma unroll
            for (int st = 0; st < CS; ++st) x = mfma16x2(o.kf[q][st], load8(vn + (l % 16) * VP + st * 32 + g8), x);
#pragma unroll
            for (int t = 0; t < 4; ++t) hf[q][t] = hf[q][t] * o.dk[q][t] + x[t];
        }
        fetch(min(jc + DEPTH, chunks - 1), o);
    };
    int jc = 0;
    for (; jc + DEPTH <= chunks; jc += DEPTH) {
#pragma unroll
        for (int d = 0; d < DEPTH; ++d) step(jc + d, ring[d]);
    }
#pragma unroll
    for (int d = 0; d < DEPTH - 1; ++d)
        if (jc + d < chunks) step(jc + d, ring[d]);

    if (!folds(write_state, mask, ln.rl)) return;
#pragma unroll
    for (int q = 0; q < KT; ++q)
        *reinterpret_cast<f32x4*>(state + key(q)) = f32x4{hf[q][0], hf[q][1], hf[q][2], hf[q][3]};
}

}  // namespace pie::attn
