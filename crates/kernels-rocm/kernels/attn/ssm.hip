#pragma once

// Gated DeltaNet (the linear-attention mixer of Qwen 3.5 / Qwen3-Next): the
// causal depthwise conv in front of it, the gates, and the delta-rule
// recurrence over a per-slot state.
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

namespace pie::attn {

__device__ __forceinline__ float silu(float z) { return z / (1.f + __expf(-z)); }

__device__ __forceinline__ bool folds(bool write_state, const u8* mask, int r) {
    return write_state && (mask == nullptr || mask[r] != 0);
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
// (when it folds) shifts the window by one. A thread owns 8 channels.
template <class T, bool SILU = true, bool RESIDUAL = false>
__global__ void ssm_conv1d_update(const T* __restrict__ x, const T* __restrict__ weight,
                                  const T* __restrict__ bias, T* __restrict__ state_base,
                                  const i32* __restrict__ slot_ids, long long slot_stride,
                                  T* __restrict__ y, int R, int C, int K, int dil, bool write_state,
                                  const u8* __restrict__ mask, const u32* __restrict__ win) {
    const int r = blockIdx.y;
    if (r >= R || (win != nullptr && r >= static_cast<int>(win[0]))) return;
    const long long row = win != nullptr ? r + static_cast<long long>(win[1]) : r;
    const int rl = win != nullptr ? r + static_cast<int>(win[3]) : r;
    const int slot = slot_ids[rl];
    if (slot < 0) return;
    const int c0 = (blockIdx.x * blockDim.x + threadIdx.x) * 8;
    if (c0 >= C) return;
    T* state = state_base + static_cast<long long>(slot) * slot_stride;
    const int span = (K - 1) * dil;
    const bool fold = folds(write_state, mask, rl);
    for (int c = c0; c < min(c0 + 8, C); ++c) {
        const float xn = to_f32(x[row * C + c]);
        float acc = bias != nullptr ? to_f32(bias[c]) : 0.f;
        for (int k = 0; k < K - 1; ++k) acc += to_f32(weight[c * K + k]) * to_f32(state[(k * dil + 1) * C + c]);
        acc += to_f32(weight[c * K + K - 1]) * xn;
        y[row * C + c] = from_f32<T>(conv_out<SILU, RESIDUAL>(acc, xn));
        if (fold) {
            for (int k = 0; k < span; ++k) state[k * C + c] = state[(k + 1) * C + c];
            state[span * C + c] = from_f32<T>(xn);
        }
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

    // The norms of q and k over the key head.
    const float qv = tid < K_d ? to_f32(qkv_row[h_k * K_d + tid]) : 0.f;
    const float kv = tid < K_d ? to_f32(qkv_row[K_h * K_d + h_k * K_d + tid]) : 0.f;
    const float qw = wave_sum(qv * qv), kw = wave_sum(kv * kv);
    if (lane() == 0) {
        norm_part[0][wave] = qw;
        norm_part[1][wave] = kw;
    }
    __syncthreads();
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
    __syncthreads();
    const float qi = live ? sq[i] : 0.f, ki = live ? sk[i] : 0.f;

    bf16* cell = state_base + static_cast<long long>(slot) * slot_stride + static_cast<long long>(h) * K_d * V_d +
                 static_cast<long long>(i) * V_d + j0;
    Pack8<bf16> sp{};
    if (live) sp = *reinterpret_cast<const Pack8<bf16>*>(cell);
    const Pack8<bf16> vp = *reinterpret_cast<const Pack8<bf16>*>(qkv_row + 2LL * K_h * K_d + static_cast<long long>(h) * V_d + j0);
    const float g = __expf(gates[row * 2 * V_h + h]);
    const float beta = gates[row * 2 * V_h + V_h + h];

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
        __syncthreads();
#pragma unroll
        for (int u = 0; u < VEC; ++u) {
            float total = 0.f;
#pragma unroll
            for (int w = 0; w < WAVES; ++w) total += red[w][(tid % TPR) * VEC + u];
            x[u] = total;
        }
        __syncthreads();
    };

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

}  // namespace pie::attn
