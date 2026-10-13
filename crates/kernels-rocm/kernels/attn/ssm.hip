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
#include "prelude/mfma.hip"

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

// The step for a batch: grid (rows, V_h), block BLOCK, a block per (row, head)
// over the head's whole K × V state. Thread t owns columns 8(t%16).. of rows
// t/16 + (BLOCK/16)q, so each wavefront read is four whole 256-byte state
// rows; the column sums fold the row groups through two shuffles and LDS. The
// same arithmetic as `ssm_gdn_decode_step`, summed in another order.
template <int KD, int VD, int BLOCK = 512>
__global__ __launch_bounds__(BLOCK) void ssm_gdn_decode_rows(
    const bf16* __restrict__ qkv, const float* __restrict__ gates, bf16* __restrict__ state_base,
    const i32* __restrict__ slot_ids, long long slot_stride, float* __restrict__ out, int K_h, int V_h,
    int conv_dim, float q_scale, bool write_state, const u8* __restrict__ mask, const u32* __restrict__ win) {
    constexpr int WAVES = BLOCK / kWave, VEC = 8, CG = VD / VEC, RG = BLOCK / CG, Q = KD / RG;
    static_assert(VD % VEC == 0 && BLOCK % CG == 0 && KD % RG == 0 && kWave % CG == 0 && KD == 2 * kWave &&
                      BLOCK >= 2 * KD,
                  "whole packs, whole row groups, and q, k a wavefront pair each");
    __shared__ float sq[KD], sk[KD];
    __shared__ float red[WAVES][VD];
    __shared__ float norm_part[4];

    const int r = blockIdx.x, h = blockIdx.y;
    if (win != nullptr && r >= static_cast<int>(win[0])) return;
    const long long row = win != nullptr ? r + static_cast<long long>(win[1]) : r;
    const int rl = win != nullptr ? r + static_cast<int>(win[3]) : r;
    const int slot = slot_ids[rl];
    if (slot < 0) return;
    const bool fold = folds(write_state, mask, rl);
    const int tid = threadIdx.x, wave = tid / kWave, l = lane();
    const int h_k = h / (V_h / K_h);
    const bf16* qkv_row = qkv + row * conv_dim;
    const int cg = tid % CG, rg = tid / CG, j0 = cg * VEC;

    // Every read of the step issued first.
    bf16* head = state_base + static_cast<long long>(slot) * slot_stride + static_cast<long long>(h) * KD * VD;
    Pack8<bf16> sp[Q];
#pragma unroll
    for (int q = 0; q < Q; ++q) sp[q] = *reinterpret_cast<const Pack8<bf16>*>(head + static_cast<long long>(rg + RG * q) * VD + j0);
    const Pack8<bf16> vp = *reinterpret_cast<const Pack8<bf16>*>(qkv_row + 2LL * K_h * KD + static_cast<long long>(h) * VD + j0);
    const float g_log = gates[row * 2 * V_h + h];
    const float beta = gates[row * 2 * V_h + V_h + h];
    // Wavefronts 0-1 hold q's key head, 2-3 k's.
    const int e = tid % KD;
    const float qk = tid < 2 * KD ? to_f32(qkv_row[(tid < KD ? 0 : K_h * KD) + h_k * KD + e]) : 0.f;

    const float part = wave_sum(qk * qk);
    if (l == 0 && wave < 4) norm_part[wave] = part;
    lds_barrier();
    const float qn = norm_part[0] + norm_part[1], kn = norm_part[2] + norm_part[3];
    if (tid < KD) sq[e] = qk * rsqrtf(qn + 1e-6f) * q_scale;
    else if (tid < 2 * KD) sk[e] = qk * rsqrtf(kn + 1e-6f);
    lds_barrier();

    const float g = __expf(g_log);
    // Column sums over the KD rows: this thread's Q rows, then the four row
    // groups a wavefront holds (lanes cg, cg ± 16, ± 32), then the wavefronts.
    auto column_sum = [&](float (&x)[VEC]) {
#pragma unroll
        for (int u = 0; u < VEC; ++u) {
#pragma unroll
            for (int off = CG; off < kWave; off <<= 1) x[u] += __shfl_xor(x[u], off, kWave);
        }
        if (l < CG) {
#pragma unroll
            for (int u = 0; u < VEC; ++u) red[wave][l * VEC + u] = x[u];
        }
        lds_barrier();
#pragma unroll
        for (int u = 0; u < VEC; ++u) {
            float total = 0.f;
#pragma unroll
            for (int w = 0; w < WAVES; ++w) total += red[w][cg * VEC + u];
            x[u] = total;
        }
        lds_barrier();
    };

    float x[VEC];
#pragma unroll
    for (int u = 0; u < VEC; ++u) x[u] = 0.f;
#pragma unroll
    for (int q = 0; q < Q; ++q) {
        const float ki = sk[rg + RG * q];
#pragma unroll
        for (int u = 0; u < VEC; ++u) x[u] += to_f32(sp[q].v[u]) * g * ki;
    }
    column_sum(x);
    float delta[VEC];
#pragma unroll
    for (int u = 0; u < VEC; ++u) {
        delta[u] = (to_f32(vp.v[u]) - x[u]) * beta;
        x[u] = 0.f;
    }
#pragma unroll
    for (int q = 0; q < Q; ++q) {
        const int i = rg + RG * q;
        const float ki = sk[i], qi = sq[i];
        Pack8<bf16> st;
#pragma unroll
        for (int u = 0; u < VEC; ++u) {
            const float next = to_f32(from_f32<bf16>(to_f32(sp[q].v[u]) * g)) + ki * delta[u];
            st.v[u] = from_f32<bf16>(next);
            x[u] += next * qi;
        }
        if (fold) *reinterpret_cast<Pack8<bf16>*>(head + static_cast<long long>(i) * VD + j0) = st;
    }
    column_sum(x);
    if (tid < CG) {
        float* o = out + (row * V_h + h) * VD + j0;
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

    // T = (I + A)⁻¹, in 16 x 16 blocks (A is strictly lower): each diagonal
    // block by forward substitution, T[i][c] = δ_ic − Σ_{j<i} A[i][j]·T[j][c]
    // with lane c keeping column c in registers; then the blocks below it a
    // block-row at a time, T_ij = −T_ii·Σ_{j≤k<i} A_ik·T_kj, in f32 on the
    // matrix core. T overwrites A in place.
    constexpr int NB = CH / 16;
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
    {
        // 16 x 16 · 16 x 16 in f32 from LDS rows at stride `ld`.
        auto product = [&](const float* m, int ldm, const float* n, int ldn, mfma_cd c) {
#pragma unroll
            for (int k0 = 0; k0 < 16; k0 += 4) {
                const int k = k0 + l / 16;
                c = mfma4_f32(m[(l % 16) * ldm + k], n[k * ldn + l % 16], c);
            }
            return c;
        };
        float* X = reinterpret_cast<float*>(Tb) + wave * 256;  // per-wavefront 16 x 16, in Tb before Tb is built
        for (int bi = 1; bi < NB; ++bi) {
            const int bj = wave;  // the block-row's blocks, one per wavefront
            mfma_cd t = mfma_zero();
            const bool mine = bj < bi;
            if (mine) {
                mfma_cd x = mfma_zero();
                for (int bk = bj; bk < bi; ++bk) x = product(&A[16 * bi][16 * bk], CH, &A[16 * bk][16 * bj], CH, x);
                mfma_each(x, [&](int i, int c, float v) { X[i * 16 + c] = v; });
                // (the wavefront reads back only what it wrote)
                __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront");
                __builtin_amdgcn_wave_barrier();
                __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront");
                t = product(&A[16 * bi][16 * bi], CH, X, 16, mfma_zero());
            }
            lds_barrier();
            if (mine) mfma_each(t, [&](int i, int c, float v) { A[16 * bi + i][16 * bj + c] = -v; });
            lds_barrier();
        }
    }

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

}  // namespace pie::attn
