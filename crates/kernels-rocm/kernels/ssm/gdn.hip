// Gated DeltaNet: the gates, the decode steps (plain, fused with the conv,
// over carried rows), the folds and tails, and the sequential and
// chunk-parallel prefills.

#include "ssm/common.hip"

namespace pie::attn {

// ----------------------------------------------------------------- gates --

// [b | a] → [g_log | β]: g_log = −exp(A_log)·softplus(a + dt_bias),
// β = sigmoid(b). Grid (rows, ⌈V_h / block⌉).
__global__ void ssm_gdn_prep_ba_gates(const bf16* __restrict__ ba, const float* __restrict__ a_log,
                                      const bf16* __restrict__ dt_bias, float* __restrict__ gates, int N,
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
template <int BLOCK>
__global__ void ssm_gdn_prep_qk_norm(const bf16* __restrict__ qkv, float* __restrict__ q_out,
                                     float* __restrict__ k_out, int K_h, int K_d, int conv_dim,
                                     float q_scale, const u32* __restrict__ win) {
    __shared__ float smem[2][BLOCK / kWave];
    const int n = blockIdx.x;
    if (win != nullptr && n >= static_cast<int>(win[0])) return;
    const long long row = win != nullptr ? n + static_cast<long long>(win[1]) : n;
    const int h = blockIdx.y;
    const bf16* q = qkv + row * conv_dim + static_cast<long long>(h) * K_d;
    const bf16* k = q + K_h * K_d;
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
template <int BLOCK>
__global__ void ssm_gdn_prep_v_gates(const bf16* __restrict__ qkv, const float* __restrict__ gates,
                                     float* __restrict__ v_out, float* __restrict__ g_out,
                                     float* __restrict__ beta_out, int K_h, int V_h, int K_d, int V_d,
                                     int conv_dim, const u32* __restrict__ win) {
    const int n = blockIdx.x;
    if (win != nullptr && n >= static_cast<int>(win[0])) return;
    const long long row = win != nullptr ? n + static_cast<long long>(win[1]) : n;
    const int h = blockIdx.y;
    const bf16* v = qkv + row * conv_dim + 2LL * K_h * K_d + static_cast<long long>(h) * V_d;
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
    const bf16* __restrict__ qkv, const float* __restrict__ gates, StateRows rows, float* __restrict__ out,
    int K_h, int V_h, int K_d, int V_d, int conv_dim, float q_scale, bool write_state, const u8* __restrict__ mask,
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
    const i32* rid = rows.row(rl);
    if (rid[0] < 0) return;
    const bool fold = folds(write_state, mask, rl);

    const int tid = threadIdx.x, wave = tid / kWave;
    const int h_k = h / (V_h / K_h);
    const bf16* qkv_row = qkv + row * conv_dim;
    const int i = tid / TPR;
    const int j0 = blockIdx.x * BV + (tid % TPR) * VEC;
    const bool live = i < K_d;

    // Every read of the step issued before the norms' barriers.
    bf16* cell = rows.atom<bf16>(rid, h, static_cast<long long>(K_d) * V_d) + static_cast<long long>(i) * V_d + j0;
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

// ------------------------------------------------- decode over carried rows --

// A buffered row's log-gate from its `[b | a]`, as `ssm_gdn_prep_ba_gates`
// forms it.
__device__ __forceinline__ float gdn_log_gate(const bf16* __restrict__ ba, int h, int V_h,
                                              const bf16* __restrict__ dt_bias, const float* __restrict__ a_log) {
    const float z = to_f32(ba[V_h + h]) + to_f32(dt_bias[h]);
    const float softplus = z > 20.f ? z : log1pf(__expf(z));
    return -__expf(a_log[h]) * softplus;
}

// One state row's eight columns folded as `ssm_gdn_decode_step` folds a token
// and stores it: S' = bf16(bf16(S·g) + k̂·u).
__device__ __forceinline__ void gdn_fold8(float (&s)[8], float g, float ki, const float* __restrict__ u) {
#pragma unroll
    for (int c = 0; c < 8; ++c) s[c] = to_f32(from_f32<bf16>(to_f32(from_f32<bf16>(s[c] * g)) + ki * u[c]));
}

// `ssm_gdn_decode_step` over a lane's buffered rows then its own (`Carry`:
// planes `[b | a]`, keys, corrected values `u = β(v − g·Sᵀk̂)`, each written
// once, when the row was a fire's own). The rows the lane commits among its
// carried ones fold first, in order and rounded as a step a token rounds
// them; the rest read as a linear attention over that state,
//   S_t = G S_c + Σ_s G_s k̂_s u_sᵀ,   G_s = Π_{s<i≤t} g_i,
// so the own row's u and output take each such row as two dots, its decay
// formed from the log-gates after it chunk by chunk from the end (never as a
// difference of sums). Where none is left past the commit, the own row is
// `ssm_gdn_decode_step`'s arithmetic. The state is stored as it stands after
// the lane's commit; the own row's `[b | a]` and u land in its buffer (its
// keys land with the conv's output). Grid (V_d / BV, rows, V_h), block 256.
template <int BV>
__global__ __launch_bounds__(256) void ssm_gdn_step_carried(
    const bf16* __restrict__ qkv, const float* __restrict__ gates, const bf16* __restrict__ ba,
    const bf16* __restrict__ dt_bias, const float* __restrict__ a_log, StateRows rows, float* __restrict__ out,
    int K_h, int V_h, int K_d, int V_d, int conv_dim, float q_scale, bool write_state, const u8* __restrict__ mask,
    Carry carry, const u32* __restrict__ win) {
    constexpr int BLOCK = 256, VEC = 8, TPR = BV / VEC, WAVES = BLOCK / kWave, NI = BLOCK / TPR, CH = kWave;
    static_assert(BV % VEC == 0 && kWave % TPR == 0 && CH <= NI, "a tile row is whole 8-wide packs");
    __shared__ float sq[NI], sk[NI];
    __shared__ float red[WAVES][BV];
    __shared__ float part[3][WAVES];
    // A chunk of buffered rows: each row's log-gate, key norm and dots with
    // the own row's q̂ and k̂, then its weights into u and into the output.
    __shared__ float c_glog[CH], c_nj[CH], c_dq[CH], c_dk[CH], c_wa[CH], c_wo[CH];
    __shared__ float g_all;

    const int r = blockIdx.y, h = blockIdx.z;
    if (win != nullptr && r >= static_cast<int>(win[0])) return;
    const long long row = win != nullptr ? r + static_cast<long long>(win[1]) : r;
    const int rl = win != nullptr ? r + static_cast<int>(win[3]) : r;
    const i32* rid = rows.row(rl);
    if (rid[0] < 0) return;
    const CarryLane cl = carry.lane(rl);
    const int n = carry.carried(cl);
    const int commit = carry.commit(rl, n + 1);
    const bool fold = folds(write_state, mask, rl) && commit > 0;
    // The carried rows folded first: those the lane commits.
    const int c = min(commit, n);

    const int tid = threadIdx.x, wave = tid / kWave, ln = lane();
    const int h_k = h / (V_h / K_h);
    const bf16* qkv_row = qkv + row * conv_dim;
    const int i = tid / TPR;
    const int j0 = blockIdx.x * BV + (tid % TPR) * VEC;
    const bool live = i < K_d;

    bf16* cell = rows.atom<bf16>(rid, h, static_cast<long long>(K_d) * V_d) + static_cast<long long>(i) * V_d + j0;
    Pack8<bf16> sp{};
    if (live) sp = *reinterpret_cast<const Pack8<bf16>*>(cell);
    const Pack8<bf16> vp =
        *reinterpret_cast<const Pack8<bf16>*>(qkv_row + 2LL * K_h * K_d + static_cast<long long>(h) * V_d + j0);
    const float g_log = gates[row * 2 * V_h + h];
    const float beta = gates[row * 2 * V_h + V_h + h];

    // The own row's q̂ and k̂ over the key head, and k̂·q̂.
    const float qv = tid < K_d ? to_f32(qkv_row[h_k * K_d + tid]) : 0.f;
    const float kv = tid < K_d ? to_f32(qkv_row[K_h * K_d + h_k * K_d + tid]) : 0.f;
    const float qw = wave_sum(qv * qv), kw = wave_sum(kv * kv), qkw = wave_sum(qv * kv);
    if (ln == 0) {
        part[0][wave] = qw;
        part[1][wave] = kw;
        part[2][wave] = qkw;
    }
    lds_barrier();
    float qn = 0.f, kn = 0.f, qk = 0.f;
#pragma unroll
    for (int w = 0; w < WAVES; ++w) {
        qn += part[0][w];
        kn += part[1][w];
        qk += part[2][w];
    }
    const float nq = rsqrtf(qn + 1e-6f) * q_scale, nk = rsqrtf(kn + 1e-6f);
    if (tid < K_d) {
        sq[tid] = qv * nq;
        sk[tid] = kv * nk;
    }
    lds_barrier();
    const float qi = live ? sq[i] : 0.f, ki = live ? sk[i] : 0.f;

    auto column_sum = [&](float (&x)[VEC]) {
#pragma unroll
        for (int u = 0; u < VEC; ++u) {
#pragma unroll
            for (int off = TPR; off < kWave; off <<= 1) x[u] += __shfl_xor(x[u], off, kWave);
        }
        if (ln < TPR) {
#pragma unroll
            for (int u = 0; u < VEC; ++u) red[wave][ln * VEC + u] = x[u];
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
    // Rows [lo, lo + m) of the lane's buffered rows into the chunk tables, a
    // wavefront a row: each one's log-gate, key norm and raw dots.
    auto chunk = [&](int lo, int m) {
        for (int e = wave; e < m; e += WAVES) {
            u64 slot;
            u32 at;
            carry.carried_at(cl, lo + e, slot, at);
            const bf16* krow = carry.plane_at<const bf16>(slot, at, 1) + static_cast<long long>(h_k) * K_d;
            float kk = 0.f, dq = 0.f, dk = 0.f;
            for (int x = ln; x < K_d; x += kWave) {
                const float k = to_f32(krow[x]);
                kk += k * k;
                dq += k * sq[x];
                dk += k * sk[x];
            }
            kk = wave_sum(kk);
            dq = wave_sum(dq);
            dk = wave_sum(dk);
            if (ln == 0) {
                const float nj = rsqrtf(kk + 1e-6f);
                c_nj[e] = nj;
                c_dq[e] = dq * nj;
                c_dk[e] = dk * nj;
                c_glog[e] = gdn_log_gate(carry.plane_at<const bf16>(slot, at, 0), h, V_h, dt_bias, a_log);
            }
        }
        lds_barrier();
    };
    auto u_of = [&](int j, float (&uu)[VEC]) {
        const float* u = carry.carried_row<float>(cl, j, 2) + static_cast<long long>(h) * V_d + j0;
        const f32x4 a = reinterpret_cast<const f32x4*>(u)[0];
        const f32x4 b = reinterpret_cast<const f32x4*>(u)[1];
        uu[0] = a[0]; uu[1] = a[1]; uu[2] = a[2]; uu[3] = a[3];
        uu[4] = b[0]; uu[5] = b[1]; uu[6] = b[2]; uu[7] = b[3];
    };

    float s[VEC];
#pragma unroll
    for (int u = 0; u < VEC; ++u) s[u] = to_f32(sp.v[u]);

    // The committed carried rows, folded in order.
    for (int lo = 0; lo < c; lo += CH) {
        const int m = min(CH, c - lo);
        chunk(lo, m);
        for (int e = 0; e < m; ++e) {
            float uu[VEC];
            u_of(lo + e, uu);
            const float kj = live ? to_f32((carry.carried_row<bf16>(cl, lo + e, 1) + static_cast<long long>(h_k) * K_d)[i]) * c_nj[e] : 0.f;
            gdn_fold8(s, __expf(c_glog[e]), kj, uu);
        }
        lds_barrier();
    }
    auto store = [&]() {
        if (!live) return;
        Pack8<bf16> st;
#pragma unroll
        for (int u = 0; u < VEC; ++u) st.v[u] = from_f32<bf16>(s[u]);
        *reinterpret_cast<Pack8<bf16>*>(cell) = st;
    };
    if (fold && commit <= n) store();

    float delta[VEC], x[VEC];
    if (c == n) {
        // Nothing carried past the commit: a token's step.
        const float g = __expf(g_log);
#pragma unroll
        for (int u = 0; u < VEC; ++u) x[u] = s[u] * g * ki;
        column_sum(x);
#pragma unroll
        for (int u = 0; u < VEC; ++u) {
            delta[u] = (to_f32(vp.v[u]) - x[u]) * beta;
            s[u] = to_f32(from_f32<bf16>(s[u] * g)) + ki * delta[u];
            x[u] = s[u] * qi;
        }
        if (fold && commit == n + 1) store();
        column_sum(x);
    } else {
        // The rows past the commit as a linear attention, last chunk first.
        float acc_a[VEC], acc_o[VEC];
#pragma unroll
        for (int u = 0; u < VEC; ++u) {
            acc_a[u] = 0.f;
            acc_o[u] = 0.f;
        }
        float after = g_log;
        for (int hi = n; hi > c;) {
            const int lo = max(c, hi - CH);
            const int m = hi - lo;
            chunk(lo, m);
            if (wave == 0) {
                const float gl = ln < m ? c_glog[ln] : 0.f;
                float later = gl;
#pragma unroll
                for (int off = 1; off < kWave; off <<= 1) {
                    const float down = __shfl_down(later, off, kWave);
                    if (ln + off < kWave) later += down;
                }
                const float past = __shfl_down(later, 1, kWave);
                const float excl = ln + 1 < m ? past : 0.f;
                if (ln < m) {
                    const float w = __expf(after + excl);
                    c_wa[ln] = w * c_dk[ln];
                    c_wo[ln] = w * c_dq[ln];
                }
                after += __shfl(later, 0, kWave);
            }
            lds_barrier();
            if (i < m) {
                float uu[VEC];
                u_of(lo + i, uu);
                const float wa = c_wa[i], wo = c_wo[i];
#pragma unroll
                for (int u = 0; u < VEC; ++u) {
                    acc_a[u] += wa * uu[u];
                    acc_o[u] += wo * uu[u];
                }
            }
            lds_barrier();
            hi = lo;
        }
        if (tid == 0) g_all = __expf(after);
        lds_barrier();
        const float g0 = g_all;
        float xq[VEC];
#pragma unroll
        for (int u = 0; u < VEC; ++u) {
            x[u] = g0 * s[u] * ki + acc_a[u];
            xq[u] = g0 * s[u] * qi + acc_o[u];
        }
        column_sum(x);
        column_sum(xq);
        const float kq = qk * nq * nk;
#pragma unroll
        for (int u = 0; u < VEC; ++u) {
            delta[u] = (to_f32(vp.v[u]) - x[u]) * beta;
            x[u] = xq[u] + kq * delta[u];
        }
    }
    if (tid < TPR) {
        float* o = out + (row * V_h + h) * V_d + j0;
#pragma unroll
        for (int u = 0; u < VEC; ++u) o[u] = x[u];
        if (float* dst = carry.own_row<float>(cl, 0, 2)) {
            float* d = dst + static_cast<long long>(h) * V_d + j0;
#pragma unroll
            for (int u = 0; u < VEC; ++u) d[u] = delta[u];
        }
    }
    // The head's first tile lands the own row's `[b | a]` pair.
    if (blockIdx.x == 0 && tid < 2) {
        if (bf16* dst = carry.own_row<bf16>(cl, 0, 0)) {
            const int at = tid * V_h + h;
            dst[at] = ba[row * 2 * V_h + at];
        }
    }
}

// The buffered rows a chunked fire's lanes land past their commit, made
// ready for a later decode's linear attention: each row's u (plane 2) and
// keys (plane 1), from the state as the fire left it (folded through the
// lane's commit, or as it stood where the lane folds nothing), stepping the
// rows past the commit in order (a carried one by its stored u, an own one
// forming its u as `ssm_gdn_decode_step` does). The state stays as it was.
// The own rows' `q k v` rows and gates are read where the fire lays them:
// the lane's extended rows (`ext`: row `RsLane::ext + j`) or its own rows
// (`qo_indptr`, the window's row offset `win[1]`). Grid (V_d / BV, lanes,
// V_h), block 256.
template <int BV>
__global__ __launch_bounds__(256) void ssm_gdn_tail_corrected(
    const bf16* __restrict__ qkv, const float* __restrict__ gates, const bf16* __restrict__ dt_bias,
    const float* __restrict__ a_log, StateRows rows, const u32* __restrict__ qo_indptr, bool ext, int K_h, int V_h,
    int K_d, int V_d, int conv_dim, bool write_state, const u8* __restrict__ mask, int lanes_stated, Carry carry,
    const u32* __restrict__ win) {
    constexpr int BLOCK = 256, VEC = 8, TPR = BV / VEC, WAVES = BLOCK / kWave, NI = BLOCK / TPR;
    __shared__ float sk[NI];
    __shared__ float red[WAVES][BV];
    __shared__ float part[WAVES];
    const int lanes = win != nullptr ? static_cast<int>(win[2]) : lanes_stated;
    const int r = blockIdx.y, h = blockIdx.z;
    if (r >= lanes) return;
    const int rl = win != nullptr ? r + static_cast<int>(win[3]) : r;
    const long long row0 = win != nullptr ? static_cast<long long>(win[1]) : 0;
    const CarryLane cl = carry.lane(rl);
    if (cl.at == CARRY_NONE) return;
    const i32* rid = rows.row(rl);
    if (rid[0] < 0) return;
    const int n = carry.carried(cl);
    const int own = static_cast<int>(qo_indptr[r + 1] - qo_indptr[r]);
    const int run = n + own;
    const int start = folds(write_state, mask, rl) ? carry.commit(rl, run) : 0;
    if (start >= run) return;

    const int tid = threadIdx.x, wave = tid / kWave, ln = lane();
    const int group = V_h / K_h;
    const int h_k = h / group;
    const int K_dim = K_h * K_d;
    const int i = tid / TPR;
    const int j0 = blockIdx.x * BV + (tid % TPR) * VEC;
    const bool live = i < K_d;
    const bf16* cell = rows.atom<bf16>(rid, h, static_cast<long long>(K_d) * V_d) + static_cast<long long>(i) * V_d + j0;
    float s[VEC];
    {
        Pack8<bf16> sp{};
        if (live) sp = *reinterpret_cast<const Pack8<bf16>*>(cell);
#pragma unroll
        for (int u = 0; u < VEC; ++u) s[u] = to_f32(sp.v[u]);
    }
    auto column_sum = [&](float (&x)[VEC]) {
#pragma unroll
        for (int u = 0; u < VEC; ++u) {
#pragma unroll
            for (int off = TPR; off < kWave; off <<= 1) x[u] += __shfl_xor(x[u], off, kWave);
        }
        if (ln < TPR) {
#pragma unroll
            for (int u = 0; u < VEC; ++u) red[wave][ln * VEC + u] = x[u];
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
    // k̂ of a key row into `sk`, its element `i` returned.
    auto keys = [&](const bf16* krow) {
        const float kv = tid < K_d ? to_f32(krow[tid]) : 0.f;
        const float kw = wave_sum(kv * kv);
        if (ln == 0) part[wave] = kw;
        lds_barrier();
        float kn = 0.f;
#pragma unroll
        for (int w = 0; w < WAVES; ++w) kn += part[w];
        if (tid < K_d) sk[tid] = kv * rsqrtf(kn + 1e-6f);
        lds_barrier();
        const float k = live ? sk[i] : 0.f;
        lds_barrier();
        return k;
    };
    for (int j = start; j < run; ++j) {
        float uu[VEC], g, ki;
        if (j < n) {
            ki = keys(carry.carried_row<bf16>(cl, j, 1) + static_cast<long long>(h_k) * K_d);
            const float* u = carry.carried_row<float>(cl, j, 2) + static_cast<long long>(h) * V_d + j0;
#pragma unroll
            for (int x = 0; x < VEC; ++x) uu[x] = u[x];
            g = __expf(gdn_log_gate(carry.carried_row<bf16>(cl, j, 0), h, V_h, dt_bias, a_log));
        } else {
            const int e = j - n;
            const long long at = ext ? static_cast<long long>(cl.ext) + j : row0 + qo_indptr[r] + e;
            const bf16* qrow = qkv + at * conv_dim;
            const bf16* krow = qrow + K_dim + static_cast<long long>(h_k) * K_d;
            ki = keys(krow);
            // The key head's first tile lands the row's keys.
            if (blockIdx.x == 0 && h % group == 0 && tid < K_d) {
                if (bf16* dst = carry.own_row<bf16>(cl, e, 1)) dst[static_cast<long long>(h_k) * K_d + tid] = krow[tid];
            }
            const float glog = gates[at * 2 * V_h + h];
            const float beta = gates[at * 2 * V_h + V_h + h];
            g = __expf(glog);
            const Pack8<bf16> vp =
                *reinterpret_cast<const Pack8<bf16>*>(qrow + 2LL * K_dim + static_cast<long long>(h) * V_d + j0);
            float x[VEC];
#pragma unroll
            for (int u = 0; u < VEC; ++u) x[u] = s[u] * g * ki;
            column_sum(x);
#pragma unroll
            for (int u = 0; u < VEC; ++u) uu[u] = (to_f32(vp.v[u]) - x[u]) * beta;
            if (tid < TPR) {
                if (float* dst = carry.own_row<float>(cl, e, 2)) {
                    float* d = dst + static_cast<long long>(h) * V_d + j0;
#pragma unroll
                    for (int u = 0; u < VEC; ++u) d[u] = uu[u];
                }
            }
        }
        gdn_fold8(s, g, ki, uu);
    }
}

// A beam group's fold of its gated-delta state (its rows below the common ancestor): its
// fold rows (keys, corrected values and gates from the kept planes `[b | a]`,
// keys, u, through the representative lane's pages) folded in order, rounded
// as a token's step stores them, and the state stored once. Grid (V_d / BV,
// V_h), block 256; a block folds each group in turn.
template <int BV>
__global__ __launch_bounds__(256) void ssm_gdn_fold(
    StateRows rows, const bf16* __restrict__ dt_bias, const float* __restrict__ a_log, int K_h, int V_h, int K_d,
    int V_d, Carry carry, const GroupFold* __restrict__ folds, int groups) {
    constexpr int BLOCK = 256, VEC = 8, TPR = BV / VEC, WAVES = BLOCK / kWave, NI = BLOCK / TPR;
    __shared__ float sk[NI];
    __shared__ float part[WAVES];
    const int h = blockIdx.y;
    // The groups that fold lead the table; the first that folds none ends it.
    for (int g = 0; g < groups; ++g) {
        const GroupFold f = folds[g];
        const int n = static_cast<int>(f.rows);
        if (n <= 0) break;
        const int rl = static_cast<int>(f.lane);
        const i32* rid = rows.row(rl);
        if (rid[0] < 0) continue;
        const CarryLane cl = carry.lane(rl);
        const int tid = threadIdx.x, wave = tid / kWave, ln = lane();
        const int h_k = h / (V_h / K_h);
        const int i = tid / TPR;
        const int j0 = blockIdx.x * BV + (tid % TPR) * VEC;
        const bool live = i < K_d;
        bf16* cell = rows.atom<bf16>(rid, h, static_cast<long long>(K_d) * V_d) + static_cast<long long>(i) * V_d + j0;
        float s[VEC];
        {
            Pack8<bf16> sp{};
            if (live) sp = *reinterpret_cast<const Pack8<bf16>*>(cell);
    #pragma unroll
            for (int u = 0; u < VEC; ++u) s[u] = to_f32(sp.v[u]);
        }
        for (int j = 0; j < n; ++j) {
            u64 slot;
            u32 pos;
            carry.fold_at(cl, f.fold_off, j, slot, pos);
            const bf16* krow = carry.plane_at<const bf16>(slot, pos, 1) + static_cast<long long>(h_k) * K_d;
            lds_barrier();
            const float kv = tid < K_d ? to_f32(krow[tid]) : 0.f;
            const float kw = wave_sum(kv * kv);
            if (ln == 0) part[wave] = kw;
            lds_barrier();
            float kn = 0.f;
    #pragma unroll
            for (int w = 0; w < WAVES; ++w) kn += part[w];
            if (tid < K_d) sk[tid] = kv * rsqrtf(kn + 1e-6f);
            lds_barrier();
            const float ki = live ? sk[i] : 0.f;
            const float gj = __expf(gdn_log_gate(carry.plane_at<const bf16>(slot, pos, 0), h, V_h, dt_bias, a_log));
            const float* u = carry.plane_at<const float>(slot, pos, 2) + static_cast<long long>(h) * V_d + j0;
            float uu[VEC];
    #pragma unroll
            for (int x = 0; x < VEC; ++x) uu[x] = u[x];
            gdn_fold8(s, gj, ki, uu);
        }
        if (!live) continue;
        Pack8<bf16> st;
    #pragma unroll
        for (int u = 0; u < VEC; ++u) st.v[u] = from_f32<bf16>(s[u]);
        *reinterpret_cast<Pack8<bf16>*>(cell) = st;
    }
}


// The step for a batch: a block a (head, row), consecutive blocks on one
// slot's consecutive heads; a wavefront reads four whole state rows. The
// q and k norms factor out of the sums, so the block meets at two barriers
// (the kv sums with the norms, then the outputs). `ssm_gdn_decode_step`'s
// arithmetic, fused in pairs and in another order.
template <int KD, int VD, int BLOCK>
__global__ __launch_bounds__(BLOCK) void ssm_gdn_decode_rows(
    const bf16* __restrict__ qkv, const float* __restrict__ gates, StateRows rows, float* __restrict__ out,
    int K_h, int V_h, int conv_dim, float q_scale, bool write_state, const u8* __restrict__ mask, const u32* __restrict__ win) {
    constexpr int WAVES = BLOCK / kWave, VEC = 8, CG = VD / VEC, RG = BLOCK / CG, Q = KD / RG;
    static_assert(VD % VEC == 0 && BLOCK % CG == 0 && KD % RG == 0 && kWave % CG == 0,
                  "whole packs, whole row groups");
    __shared__ alignas(16) float red[2][WAVES][VD];
    __shared__ float norms[WAVES][2];

    const int h = blockIdx.x, r = blockIdx.y;
    if (win != nullptr && r >= static_cast<int>(win[0])) return;
    const long long row = win != nullptr ? r + static_cast<long long>(win[1]) : r;
    const int rl = win != nullptr ? r + static_cast<int>(win[3]) : r;
    const i32* rid = rows.row(rl);
    if (rid[0] < 0) return;
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
    bf16* head = rows.atom<bf16>(rid, h, static_cast<long long>(KD) * VD);
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

// `ssm_gdn_step_carried` in `ssm_gdn_decode_rows`' layout: a block a (head,
// row) holds the head's whole state, a thread Q = KD / RG state rows of
// eight columns. The committed carried rows fold first, in order and rounded
// as a token's step stores them; the rest read as a linear attention over
// that state, each row's dots formed once a block, a wavefront a row. Where
// none is left past the commit, the own row is `ssm_gdn_decode_rows`'
// arithmetic, bit for bit. The state is stored at the lane's commit; the own
// row's `[b | a]` and u land in its buffer.
template <int KD, int VD, int BLOCK>
__global__ __launch_bounds__(BLOCK) void ssm_gdn_rows_carried(
    const bf16* __restrict__ qkv, const float* __restrict__ gates, const bf16* __restrict__ ba,
    const bf16* __restrict__ dt_bias, const float* __restrict__ a_log, StateRows rows, float* __restrict__ out,
    int K_h, int V_h, int conv_dim, float q_scale, bool write_state, const u8* __restrict__ mask, Carry carry,
    const u32* __restrict__ win) {
    constexpr int WAVES = BLOCK / kWave, VEC = 8, CG = VD / VEC, RG = BLOCK / CG, Q = KD / RG, CH = kWave;
    static_assert(VD % VEC == 0 && BLOCK % CG == 0 && KD % RG == 0 && kWave % CG == 0 && KD <= 2 * kWave,
                  "whole packs, whole row groups, a key row two elements a lane");
    __shared__ alignas(16) float red[2][WAVES][VD];
    __shared__ float norms[WAVES][3];
    __shared__ float sq[KD], sk[KD];
    __shared__ float c_glog[CH], c_nj[CH], c_dq[CH], c_dk[CH], c_wa[CH], c_wo[CH];
    __shared__ float g_all;

    const int h = blockIdx.x, r = blockIdx.y;
    if (win != nullptr && r >= static_cast<int>(win[0])) return;
    const long long row = win != nullptr ? r + static_cast<long long>(win[1]) : r;
    const int rl = win != nullptr ? r + static_cast<int>(win[3]) : r;
    const i32* rid = rows.row(rl);
    if (rid[0] < 0) return;
    const int tid = threadIdx.x, wave = tid / kWave, l = lane();
    const int h_k = h / (V_h / K_h);
    const bf16* qkv_row = qkv + row * conv_dim;
    const int cg = tid % CG, rg = tid / CG, j0 = cg * VEC;
    const CarryLane cl = carry.lane(rl);
    const int n = carry.carried(cl);
    const int commit = carry.commit(rl, n + 1);
    const bool fold = folds_uniform(write_state, mask, rl) && commit > 0;
    const int c = min(commit, n);

    float qi[Q], ki[Q];
#pragma unroll
    for (int q = 0; q < Q; ++q) {
        qi[q] = to_f32(qkv_row[h_k * KD + rg + RG * q]);
        ki[q] = to_f32(qkv_row[K_h * KD + h_k * KD + rg + RG * q]);
    }
    const Pack8<bf16> vp = *reinterpret_cast<const Pack8<bf16>*>(qkv_row + 2LL * K_h * KD + static_cast<long long>(h) * VD + j0);
    const float g_log = gates[row * 2 * V_h + h];
    const float beta = gates[row * 2 * V_h + V_h + h];
    bf16* head = rows.atom<bf16>(rid, h, static_cast<long long>(KD) * VD);
    u32x4 sp[Q];
#pragma unroll
    for (int q = 0; q < Q; ++q) sp[q] = state_load(head + (rg + RG * q) * VD + j0);

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

    // The own row's norms (and its raw q·k), as `ssm_gdn_decode_rows` sums them.
    float qq = 0.f, kk = 0.f, qk = 0.f;
#pragma unroll
    for (int q = 0; q < Q; ++q) {
        qq += qi[q] * qi[q];
        kk += ki[q] * ki[q];
        qk += qi[q] * ki[q];
    }
    qq = lanes(qq);
    kk = lanes(kk);
    qk = lanes(qk);
    if (l == 0) {
        norms[wave][0] = qq;
        norms[wave][1] = kk;
        norms[wave][2] = qk;
    }
    lds_barrier();
    float qn = 0.f, kn = 0.f, qkn = 0.f;
#pragma unroll
    for (int w = 0; w < WAVES; ++w) {
        qn += norms[w][0];
        kn += norms[w][1];
        qkn += norms[w][2];
    }
    const float rk = rsqrtf(kn + 1e-6f), rq = rsqrtf(qn + 1e-6f) * q_scale;

    // The own row's q̂ and k̂ for the buffered rows' dots.
    if (n > 0 && cg == 0) {
#pragma unroll
        for (int q = 0; q < Q; ++q) {
            sq[rg + RG * q] = qi[q] * rq;
            sk[rg + RG * q] = ki[q] * rk;
        }
    }
    // Rows [lo, lo + m) of the lane's buffered rows into the chunk tables, a
    // wavefront a row.
    auto chunk = [&](int lo, int m) {
        lds_barrier();
        for (int e = wave; e < m; e += WAVES) {
            u64 slot;
            u32 at;
            carry.carried_at(cl, lo + e, slot, at);
            const bf16* krow = carry.plane_at<const bf16>(slot, at, 1) + static_cast<long long>(h_k) * KD;
            float a2 = 0.f, dq = 0.f, dk = 0.f;
#pragma unroll
            for (int x = l; x < KD; x += kWave) {
                const float k = to_f32(krow[x]);
                a2 += k * k;
                dq += k * sq[x];
                dk += k * sk[x];
            }
            a2 = wave_sum(a2);
            dq = wave_sum(dq);
            dk = wave_sum(dk);
            if (l == 0) {
                const float nj = rsqrtf(a2 + 1e-6f);
                c_nj[e] = nj;
                c_dq[e] = dq * nj;
                c_dk[e] = dk * nj;
                c_glog[e] = gdn_log_gate(carry.plane_at<const bf16>(slot, at, 0), h, V_h, dt_bias, a_log);
            }
        }
        lds_barrier();
    };
    auto u_of = [&](int j, float (&uu)[VEC]) {
        const float* u = carry.carried_row<float>(cl, j, 2) + static_cast<long long>(h) * VD + j0;
        const f32x4 a = reinterpret_cast<const f32x4*>(u)[0];
        const f32x4 b = reinterpret_cast<const f32x4*>(u)[1];
        uu[0] = a[0]; uu[1] = a[1]; uu[2] = a[2]; uu[3] = a[3];
        uu[4] = b[0]; uu[5] = b[1]; uu[6] = b[2]; uu[7] = b[3];
    };

    // The committed carried rows, folded in order: S' = bf16(bf16(S·g) + k̂·u).
    for (int lo = 0; lo < c; lo += CH) {
        const int m = min(CH, c - lo);
        chunk(lo, m);
        for (int e = 0; e < m; ++e) {
            float uu[VEC];
            u_of(lo + e, uu);
            const bf16* krow = carry.carried_row<bf16>(cl, lo + e, 1) + static_cast<long long>(h_k) * KD;
            const float g = __expf(c_glog[e]), nj = c_nj[e];
#pragma unroll
            for (int q = 0; q < Q; ++q) {
                const float kj = to_f32(krow[rg + RG * q]) * nj;
                u32x4 st;
#pragma unroll
                for (int w = 0; w < VEC / 2; ++w) {
                    const float lo2 = bf16_round(bf16_round(bf16_lo(sp[q][w]) * g) + kj * uu[2 * w]);
                    const float hi2 = bf16_round(bf16_round(bf16_hi(sp[q][w]) * g) + kj * uu[2 * w + 1]);
                    st[w] = bf16_pack_rne(lo2, hi2);
                }
                sp[q] = st;
            }
        }
    }
    if (fold && commit <= n) {
#pragma unroll
        for (int q = 0; q < Q; ++q) state_store(head + (rg + RG * q) * VD + j0, sp[q]);
    }

    float x[VEC];
    f32x2 d2[VEC / 2];
    if (c == n) {
        // `ssm_gdn_decode_rows`' step, as it orders it.
        f32x2 x2[VEC / 2];
#pragma unroll
        for (int w = 0; w < VEC / 2; ++w) x2[w] = f32x2{0.f, 0.f};
#pragma unroll
        for (int q = 0; q < Q; ++q) {
            const f32x2 k2 = {ki[q], ki[q]};
#pragma unroll
            for (int w = 0; w < VEC / 2; ++w)
                x2[w] = __builtin_elementwise_fma(f32x2{bf16_lo(sp[q][w]), bf16_hi(sp[q][w])}, k2, x2[w]);
        }
#pragma unroll
        for (int w = 0; w < VEC / 2; ++w) {
            x[2 * w] = x2[w].x;
            x[2 * w + 1] = x2[w].y;
        }
        column_sums(x, 0);
        lds_barrier();
        column_totals(x, 0);
        const float g = __expf(g_log);
#pragma unroll
        for (int w = 0; w < VEC / 2; ++w) {
            d2[w] = f32x2{(to_f32(vp.v[2 * w]) - x[2 * w] * (g * rk)) * beta,
                          (to_f32(vp.v[2 * w + 1]) - x[2 * w + 1] * (g * rk)) * beta};
            x2[w] = f32x2{0.f, 0.f};
        }
        const f32x2 g2 = {g, g};
        const bool own = fold && commit == n + 1;
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
            if (own) state_store(head + (rg + RG * q) * VD + j0, st);
        }
#pragma unroll
        for (int w = 0; w < VEC / 2; ++w) {
            x[2 * w] = x2[w].x;
            x[2 * w + 1] = x2[w].y;
        }
        column_sums(x, 1);
        lds_barrier();
        column_totals(x, 1);
#pragma unroll
        for (int u = 0; u < VEC; ++u) x[u] *= rq;
    } else {
        // The rows past the commit as a linear attention, last chunk first;
        // row e of a chunk accumulates in row group e mod RG.
        float acc_a[VEC], acc_o[VEC];
#pragma unroll
        for (int u = 0; u < VEC; ++u) {
            acc_a[u] = 0.f;
            acc_o[u] = 0.f;
        }
        float after = g_log;
        for (int hi = n; hi > c; ) {
            const int lo = max(c, hi - CH);
            const int m = hi - lo;
            chunk(lo, m);
            if (wave == 0) {
                const float gl = l < m ? c_glog[l] : 0.f;
                float later = gl;
#pragma unroll
                for (int off = 1; off < kWave; off <<= 1) {
                    const float down = __shfl_down(later, off, kWave);
                    if (l + off < kWave) later += down;
                }
                const float past = __shfl_down(later, 1, kWave);
                const float excl = l + 1 < m ? past : 0.f;
                if (l < m) {
                    const float w = __expf(after + excl);
                    c_wa[l] = w * c_dk[l];
                    c_wo[l] = w * c_dq[l];
                }
                after += __shfl(later, 0, kWave);
            }
            lds_barrier();
            for (int e = rg; e < m; e += RG) {
                float uu[VEC];
                u_of(lo + e, uu);
                const float wa = c_wa[e], wo = c_wo[e];
#pragma unroll
                for (int u = 0; u < VEC; ++u) {
                    acc_a[u] += wa * uu[u];
                    acc_o[u] += wo * uu[u];
                }
            }
            hi = lo;
        }
        if (tid == 0) g_all = __expf(after);
        lds_barrier();
        const float g0 = g_all;
        // The state's part over this thread's rows, then the column totals.
        float xq[VEC];
#pragma unroll
        for (int u = 0; u < VEC; ++u) {
            x[u] = 0.f;
            xq[u] = 0.f;
        }
#pragma unroll
        for (int q = 0; q < Q; ++q) {
#pragma unroll
            for (int w = 0; w < VEC / 2; ++w) {
                const float s0 = bf16_lo(sp[q][w]), s1 = bf16_hi(sp[q][w]);
                x[2 * w] += s0 * ki[q];
                x[2 * w + 1] += s1 * ki[q];
                xq[2 * w] += s0 * qi[q];
                xq[2 * w + 1] += s1 * qi[q];
            }
        }
#pragma unroll
        for (int u = 0; u < VEC; ++u) {
            x[u] = g0 * rk * x[u] + acc_a[u];
            xq[u] = g0 * rq * xq[u] + acc_o[u];
        }
        column_sums(x, 0);
        column_sums(xq, 1);
        lds_barrier();
        column_totals(x, 0);
        column_totals(xq, 1);
        const float kq = qkn * rk * rq;
#pragma unroll
        for (int w = 0; w < VEC / 2; ++w) {
            d2[w] = f32x2{(to_f32(vp.v[2 * w]) - x[2 * w]) * beta, (to_f32(vp.v[2 * w + 1]) - x[2 * w + 1]) * beta};
            x[2 * w] = xq[2 * w] + kq * d2[w].x;
            x[2 * w + 1] = xq[2 * w + 1] + kq * d2[w].y;
        }
    }
    if (tid < CG) {
        float* o = out + (row * V_h + h) * VD + j0;
        *reinterpret_cast<f32x4*>(o) = f32x4{x[0], x[1], x[2], x[3]};
        *reinterpret_cast<f32x4*>(o + 4) = f32x4{x[4], x[5], x[6], x[7]};
        if (float* dst = carry.own_row<float>(cl, 0, 2)) {
            float* d = dst + static_cast<long long>(h) * VD + j0;
            *reinterpret_cast<f32x4*>(d) = f32x4{d2[0].x, d2[0].y, d2[1].x, d2[1].y};
            *reinterpret_cast<f32x4*>(d + 4) = f32x4{d2[2].x, d2[2].y, d2[3].x, d2[3].y};
        }
    }
    if (tid < 2) {
        if (bf16* dst = carry.own_row<bf16>(cl, 0, 0)) {
            const int at = tid * V_h + h;
            dst[at] = ba[row * 2 * V_h + at];
        }
    }
}

// A gated-delta layer's whole decode step in one launch: the [qkv | z]
// split, the conv update, the gate prep and the recurrence, each rounding
// where its unfused op stores. A key head's q and k channels are shared by
// its G value heads' blocks: each block reads them and draws a ticket
// (`tickets[r·K_h + h_k]`, zero between launches); the last shifts the
// shared window and zeroes the ticket.
template <int KD, int VD, int BLOCK, int KMAX>
__global__ __launch_bounds__(BLOCK) void ssm_gdn_step_fused(
    const bf16* __restrict__ qkvz, const bf16* __restrict__ conv_w, StateRows conv_rows, bool conv_write,
    const u8* __restrict__ conv_mask, const bf16* __restrict__ ba, const float* __restrict__ a_log,
    const bf16* __restrict__ dt_bias, StateRows rows, bool write_state,
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
    const i32* rid = rows.row(rl);
    const i32* crid = conv_rows.row(rl);
    if (rid[0] < 0 || crid[0] < 0) return;
    const int tid = threadIdx.x, wave = tid / kWave, l = lane();
    const int G = V_h / K_h;
    const int h_k = h / G;
    const int C = 2 * K_h * KD + V_h * VD;  // the conv's channels: [q | k | v]
    const long long QW = static_cast<long long>(C) + static_cast<long long>(V_h) * VD;
    const bf16* in_row = qkvz + row * QW;
    const int span = K - 1;

    // This head's state, in flight under the conv and the ticket.
    const int cg = tid % CG, rg = tid / CG, j0 = cg * VEC;
    bf16* head = rows.atom<bf16>(rid, h, static_cast<long long>(KD) * VD);
    u32x4 sp[Q];
#pragma unroll
    for (int q = 0; q < Q; ++q) sp[q] = state_load(head + (rg + RG * q) * VD + j0);

    // The gates, as `ssm_gdn_prep_ba_gates` lands them.
    const float bb = to_f32(ba[row * 2 * V_h + h]);
    const float za = to_f32(ba[row * 2 * V_h + V_h + h]) + to_f32(dt_bias[h]);
    const float softplus = za > 20.f ? za : log1pf(__expf(za));
    const float g_log = -__expf(a_log[h]) * softplus;
    const float beta = 1.f / (1.f + __expf(-bb));

    // The conv over this block's channels: q and k of its key head, v of
    // its own; every read in flight before any is used. Channel ch[u]'s
    // window column: cc[u] in the piece at cst[u], `ccs` channels a piece.
    int ch[CPT], cc[CPT];
    bf16* cst[CPT];
    float xn[CPT], hist[CPT][KMAX], wk[CPT][KMAX];
    const int ccs = conv_rows.atoms;
#pragma unroll
    for (int u = 0; u < CPT; ++u) {
        const int ii = min(tid + u * BLOCK, CH - 1);
        ch[u] = ii < KD       ? h_k * KD + ii
                : ii < 2 * KD ? K_h * KD + h_k * KD + (ii - KD)
                              : 2 * K_h * KD + h * VD + (ii - 2 * KD);
        cc[u] = ch[u] % ccs;
        cst[u] = conv_rows.piece<bf16>(crid, ch[u] / ccs);
        xn[u] = to_f32(in_row[ch[u]]);
#pragma unroll
        for (int k = 0; k < KMAX; ++k) {
            wk[u][k] = k < K ? to_f32(conv_w[ch[u] * K + k]) : 0.f;
            hist[u][k] = k + 1 <= span ? to_f32(cst[u][(k + 1) * ccs + cc[u]]) : 0.f;
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
                if (k < span) cst[u][k * ccs + cc[u]] = from_f32<bf16>(hist[u][k]);
            cst[u][span * ccs + cc[u]] = from_f32<bf16>(xn[u]);
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
                if (k < span) cst[u][k * ccs + cc[u]] = from_f32<bf16>(hist[u][k]);
            cst[u][span * ccs + cc[u]] = from_f32<bf16>(xn[u]);
        }
    }

    // The recurrence, as `ssm_gdn_decode_rows` walks it.
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

// `ssm_gdn_step_fused` over a lane's carried rows then its own: the conv's
// window positions before the own row read the committed cells, then the
// lane's carried rows' inputs (`conv`: plane 0 the conv's input rows, plane
// 1 its output's keys), and the window is stored as it stands after the
// lane's commit; the recurrence is `ssm_gdn_rows_carried`'s over `step`'s
// planes (`[b | a]`, keys, corrected values). Where the lane carries nothing
// and commits its row, it is `ssm_gdn_step_fused`, bit for bit. The own
// row's conv input, keys, `[b | a]` and u land in its buffer.
template <int KD, int VD, int BLOCK, int KMAX>
__global__ __launch_bounds__(BLOCK) void ssm_gdn_step_fused_carried(
    const bf16* __restrict__ qkvz, const bf16* __restrict__ conv_w, StateRows conv_rows, bool conv_write,
    const u8* __restrict__ conv_mask, const bf16* __restrict__ ba, const float* __restrict__ a_log,
    const bf16* __restrict__ dt_bias, StateRows rows, bool write_state,
    const u8* __restrict__ mask, bf16* __restrict__ qkv_out, bf16* __restrict__ z_out, float* __restrict__ out,
    int* __restrict__ tickets, int K_h, int V_h, int K, float q_scale, Carry conv, Carry step, const u32* __restrict__ win) {
    constexpr int WAVES = BLOCK / kWave, VEC = 8, CG = VD / VEC, RG = BLOCK / CG, Q = KD / RG;
    constexpr int CH = 2 * KD + VD, CPT = (CH + BLOCK - 1) / BLOCK, RCH = kWave;
    static_assert(VD % VEC == 0 && BLOCK % CG == 0 && KD % RG == 0 && kWave % CG == 0 && KD <= 2 * kWave,
                  "whole packs, whole row groups, a key row two elements a lane");
    __shared__ alignas(16) float red[2][WAVES][VD];
    __shared__ float norms[WAVES][3];
    __shared__ float cv[CH];
    __shared__ float sq[KD], sk[KD];
    __shared__ float c_glog[RCH], c_nj[RCH], c_dq[RCH], c_dk[RCH], c_wa[RCH], c_wo[RCH];
    __shared__ float g_all;
    __shared__ int last_s;
    // The page slot and row of the lane's carried rows [base, n), resolved
    // once: every later read of a carried row is one load.
    __shared__ u64 t_slot[RCH];
    __shared__ u32 t_pos[RCH];
    // A chunk of carried rows staged whole: keys (bf16), corrected values
    // (this head's columns, f32) and `a` gate words, one load a chunk.
    constexpr int SCH = 16;
    __shared__ alignas(16) bf16 s_k[SCH][KD];
    __shared__ alignas(16) float s_u[SCH][VD];
    __shared__ float s_a[SCH];

    const int h = blockIdx.x, r = blockIdx.y;
    if (win != nullptr && r >= static_cast<int>(win[0])) return;
    const long long row = win != nullptr ? r + static_cast<long long>(win[1]) : r;
    const int rl = win != nullptr ? r + static_cast<int>(win[3]) : r;
    const i32* rid = rows.row(rl);
    const i32* crid = conv_rows.row(rl);
    if (rid[0] < 0 || crid[0] < 0) return;
    const int tid = threadIdx.x, wave = tid / kWave, l = lane();
    const int G = V_h / K_h;
    const int h_k = h / G;
    const int C = 2 * K_h * KD + V_h * VD;
    const long long QW = static_cast<long long>(C) + static_cast<long long>(V_h) * VD;
    const bf16* in_row = qkvz + row * QW;
    const CarryLane cl = step.lane(rl);
    const CarryLane xl = conv.lane(rl);
    const int n = step.carried(cl);
    // A lane that writes no state (a beam lane, its write masked off)
    // commits none of its carried rows: it reads them all as the linear
    // attention from its state, as CUDA's carried step does, not as folds
    // into registers it then drops. A lane that writes folds its committed
    // rows as the reference does.
    const int commit = folds_uniform(write_state, mask, rl) ? step.commit(rl, n + 1) : 0;
    const int c = min(commit, n);

    const int base = max(0, n - RCH);
    if (tid < n - base) {
        u64 slot;
        u32 pos;
        step.carried_at(cl, base + tid, slot, pos);
        t_slot[tid] = slot;
        t_pos[tid] = pos;
    }
    // Carried row t's page slot and row: from the table, or resolved where
    // it lies below it (a search deeper than the table).
    auto where = [&](const Carry& cr, const CarryLane& lane_, int t, u64& slot, u32& pos) {
        if (t >= base) {
            slot = t_slot[t - base];
            pos = t_pos[t - base];
        } else {
            cr.carried_at(lane_, t, slot, pos);
        }
    };
    const int cg = tid % CG, rg = tid / CG, j0 = cg * VEC;
    bf16* head = rows.atom<bf16>(rid, h, static_cast<long long>(KD) * VD);
    u32x4 sp[Q];
#pragma unroll
    for (int q = 0; q < Q; ++q) sp[q] = state_load(head + (rg + RG * q) * VD + j0);

    const float bb = to_f32(ba[row * 2 * V_h + h]);
    const float za = to_f32(ba[row * 2 * V_h + V_h + h]) + to_f32(dt_bias[h]);
    const float softplus = za > 20.f ? za : log1pf(__expf(za));
    const float g_log = -__expf(a_log[h]) * softplus;
    const float beta = 1.f / (1.f + __expf(-bb));

    // The conv over this block's channels, its window from the committed
    // cells and the carried rows' inputs.
    int ch[CPT], cc[CPT];
    bf16* cst[CPT];
    float xn[CPT], wk[CPT][KMAX];
    const int ccs = conv_rows.atoms;
#pragma unroll
    for (int u = 0; u < CPT; ++u) {
        const int ii = min(tid + u * BLOCK, CH - 1);
        ch[u] = ii < KD       ? h_k * KD + ii
                : ii < 2 * KD ? K_h * KD + h_k * KD + (ii - KD)
                              : 2 * K_h * KD + h * VD + (ii - 2 * KD);
        cc[u] = ch[u] % ccs;
        cst[u] = conv_rows.piece<bf16>(crid, ch[u] / ccs);
        xn[u] = to_f32(in_row[ch[u]]);
#pragma unroll
        for (int k = 0; k < KMAX; ++k) wk[u][k] = k < K ? to_f32(conv_w[ch[u] * K + k]) : 0.f;
    }
    __syncthreads();
    // Every carried row fits one stage at a natural lag: its keys, values
    // and gates are read now, under the conv and the ticket.
    const bool whole = n <= SCH;
    if (n > 0 && whole) {
        constexpr int KP = KD / 8, UP = VD / 4;
        for (int x = tid; x < n * KP; x += BLOCK) {
            const int e = x / KP, p8 = x % KP;
            *reinterpret_cast<u32x4*>(&s_k[e][p8 * 8]) = *reinterpret_cast<const u32x4*>(
                step.plane_at<const bf16>(t_slot[e], t_pos[e], 1) + static_cast<long long>(h_k) * KD + p8 * 8);
        }
        for (int x = tid; x < n * UP; x += BLOCK) {
            const int e = x / UP, p4 = x % UP;
            *reinterpret_cast<f32x4*>(&s_u[e][p4 * 4]) = *reinterpret_cast<const f32x4*>(
                step.plane_at<const float>(t_slot[e], t_pos[e], 2) + static_cast<long long>(h) * VD + p4 * 4);
        }
        if (tid < n) s_a[tid] = gdn_log_gate(step.plane_at<const bf16>(t_slot[tid], t_pos[tid], 0), h, V_h, dt_bias, a_log);
    }
    const bool cfold = folds_uniform(conv_write, conv_mask, rl) && commit > 0;
    // Where run position t's input of channel slot u lies: a committed
    // cell or a carried row's input; null for the own row. Every tap's and
    // every stored cell's source is placed first and all of them read at
    // once, never one dependent load a tap.
    // `table`: t is one of the run's last K - 1 rows, which the slot table
    // always holds (`base` keeps the last RCH >= K - 1), so its row is one
    // shared load, not the lane's page walk inlined at each of the taps.
    auto src = [&](int u, int t, bool table) -> const bf16* {
        if (t < 0) return &cst[u][(K + t) * ccs + cc[u]];
        if (t < n) {
            u64 slot;
            u32 pos;
            if (table) {
                slot = t_slot[t - base];
                pos = t_pos[t - base];
            } else {
                where(conv, xl, t, slot, pos);
            }
            return conv.plane_at<const bf16>(slot, pos, 0) + ch[u];
        }
        return nullptr;
    };
    static_assert(RCH >= KMAX - 1, "the slot table holds every window tap's row");
    float hv[CPT][KMAX], wv[CPT][KMAX];
    {
        const bf16* hp[CPT][KMAX];
        const bf16* wp[CPT][KMAX];
#pragma unroll
        for (int u = 0; u < CPT; ++u) {
#pragma unroll
            for (int k = 0; k < KMAX; ++k) {
                hp[u][k] = k < K ? src(u, n - (K - 1 - k), true) : nullptr;
                // The committed cells may lie below the table (a run past RCH
                // rows committing few): the lane's page walk.
                wp[u][k] = cfold && k < K ? src(u, commit - K + k, false) : nullptr;
            }
        }
#pragma unroll
        for (int u = 0; u < CPT; ++u) {
#pragma unroll
            for (int k = 0; k < KMAX; ++k) {
                hv[u][k] = hp[u][k] != nullptr ? to_f32(*hp[u][k]) : xn[u];
                wv[u][k] = wp[u][k] != nullptr ? to_f32(*wp[u][k]) : xn[u];
            }
        }
    }
    bf16* own_x = conv.own_row<bf16>(xl, 0, 0);
    bf16* own_k = conv.own_row<bf16>(xl, 0, 1);
#pragma unroll
    for (int u = 0; u < CPT; ++u) {
        const int i = tid + u * BLOCK;
        if (i >= CH) continue;
        float acc = 0.f;
#pragma unroll
        for (int k = 0; k < KMAX; ++k)
            if (k < K) acc += wk[u][k] * hv[u][k];
        const bf16 out_c = from_f32<bf16>(silu(acc));
        cv[i] = to_f32(out_c);
        const bool own = i >= 2 * KD;
        if (own || h % G == 0) {
            qkv_out[row * C + ch[u]] = from_f32<bf16>(xn[u]);
            if (own_x != nullptr) own_x[ch[u]] = from_f32<bf16>(xn[u]);
        }
        // The row's keys, as the conv's output lands them.
        if (own_k != nullptr && h % G == 0 && i >= KD && i < 2 * KD) own_k[h_k * KD + (i - KD)] = out_c;
    }
    if (tid < VD) z_out[row * (V_h * VD) + h * VD + tid] = in_row[C + h * VD + tid];
    // Every block has read the shared window before any writes it: the
    // ticket's last block writes the key head's q and k channels.
    __syncthreads();
    if (tid == 0) {
        const int tk = r * K_h + h_k;
        const bool last = atomicAdd(&tickets[tk], 1) == G - 1;
        if (last) atomicExch(&tickets[tk], 0);
        last_s = last;
    }
    __syncthreads();
    if (cfold) {
#pragma unroll
        for (int u = 0; u < CPT; ++u) {
            const int i = tid + u * BLOCK;
            if (i >= CH || (i < 2 * KD && !last_s)) continue;
#pragma unroll
            for (int k = 0; k < KMAX; ++k)
                if (k < K) cst[u][k * ccs + cc[u]] = from_f32<bf16>(wv[u][k]);
        }
    }

    const bool fold = folds_uniform(write_state, mask, rl) && commit > 0;
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
    float qq = 0.f, kk = 0.f, qk = 0.f;
#pragma unroll
    for (int q = 0; q < Q; ++q) {
        qq += qi[q] * qi[q];
        kk += ki[q] * ki[q];
        qk += qi[q] * ki[q];
    }
    qq = lanes(qq);
    kk = lanes(kk);
    qk = lanes(qk);
    if (l == 0) {
        norms[wave][0] = qq;
        norms[wave][1] = kk;
        norms[wave][2] = qk;
    }
    lds_barrier();
    float qn = 0.f, kn = 0.f, qkn = 0.f;
#pragma unroll
    for (int w = 0; w < WAVES; ++w) {
        qn += norms[w][0];
        kn += norms[w][1];
        qkn += norms[w][2];
    }
    const float rk = rsqrtf(kn + 1e-6f), rq = rsqrtf(qn + 1e-6f) * q_scale;
    if (n > 0 && cg == 0) {
#pragma unroll
        for (int q = 0; q < Q; ++q) {
            sq[rg + RG * q] = qi[q] * rq;
            sk[rg + RG * q] = ki[q] * rk;
        }
    }
    // Rows [lo, lo + m) staged (m <= SCH): each thread a 16-byte piece of
    // the keys and two of the values, then a wavefront a row forms its dots,
    // norm and log-gate from the stage.
    // Rows [lo, lo + m) into the stage (m <= SCH): each thread a 16-byte
    // piece of the keys and two of the values. Their gates and dots wait for
    // the own row ([`dots`]).
    auto stage_load = [&](int lo, int m) {
        constexpr int KP = KD / 8, UP = VD / 4;
        for (int x = tid; x < m * KP; x += BLOCK) {
            const int e = x / KP, p8 = x % KP;
            u64 slot;
            u32 pos;
            where(step, cl, lo + e, slot, pos);
            *reinterpret_cast<u32x4*>(&s_k[e][p8 * 8]) = *reinterpret_cast<const u32x4*>(
                step.plane_at<const bf16>(slot, pos, 1) + static_cast<long long>(h_k) * KD + p8 * 8);
        }
        for (int x = tid; x < m * UP; x += BLOCK) {
            const int e = x / UP, p4 = x % UP;
            u64 slot;
            u32 pos;
            where(step, cl, lo + e, slot, pos);
            *reinterpret_cast<f32x4*>(&s_u[e][p4 * 4]) = *reinterpret_cast<const f32x4*>(
                step.plane_at<const float>(slot, pos, 2) + static_cast<long long>(h) * VD + p4 * 4);
        }
        if (tid < m) {
            u64 slot;
            u32 pos;
            where(step, cl, lo + tid, slot, pos);
            s_a[tid] = gdn_log_gate(step.plane_at<const bf16>(slot, pos, 0), h, V_h, dt_bias, a_log);
        }
    };
    // The staged rows' dots with the own row's q̂ and k̂, their norms and
    // gates: a wavefront a row, every row at once.
    auto dots = [&](int m) {
        lds_barrier();
        for (int e = wave; e < m; e += WAVES) {
            float a2 = 0.f, dq = 0.f, dk = 0.f;
#pragma unroll
            for (int x = l; x < KD; x += kWave) {
                const float k = to_f32(s_k[e][x]);
                a2 += k * k;
                dq += k * sq[x];
                dk += k * sk[x];
            }
            a2 = wave_sum(a2);
            dq = wave_sum(dq);
            dk = wave_sum(dk);
            if (l == 0) {
                const float nj = rsqrtf(a2 + 1e-6f);
                c_nj[e] = nj;
                c_dq[e] = dq * nj;
                c_dk[e] = dk * nj;
                c_glog[e] = s_a[e];
            }
        }
        lds_barrier();
    };
    auto stage = [&](int lo, int m) {
        lds_barrier();
        stage_load(lo, m);
        dots(m);
    };
    // Staged row e folded into this thread's rows: S' = bf16(bf16(S·g) + k̂·u).
    auto fold_row = [&](int e) {
        const float g = __expf(c_glog[e]), nj = c_nj[e];
#pragma unroll
        for (int q = 0; q < Q; ++q) {
            const float kj = to_f32(s_k[e][rg + RG * q]) * nj;
            u32x4 st;
#pragma unroll
            for (int w = 0; w < VEC / 2; ++w) {
                const float lo2 = bf16_round(bf16_round(bf16_lo(sp[q][w]) * g) + kj * s_u[e][j0 + 2 * w]);
                const float hi2 = bf16_round(bf16_round(bf16_hi(sp[q][w]) * g) + kj * s_u[e][j0 + 2 * w + 1]);
                st[w] = bf16_pack_rne(lo2, hi2);
            }
            sp[q] = st;
        }
    };
    // Where every carried row fits one stage, it is staged once for the
    // fold and the linear attention both.
    if (n > 0 && whole) dots(n);
    for (int lo = 0; lo < c; lo += SCH) {
        const int m = min(SCH, c - lo);
        if (!whole) stage(lo, m);
        for (int e = 0; e < m; ++e) fold_row(whole ? lo + e : e);
    }
    if (fold && commit <= n) {
#pragma unroll
        for (int q = 0; q < Q; ++q) state_store(head + (rg + RG * q) * VD + j0, sp[q]);
    }
    float x[VEC];
    f32x2 d2[VEC / 2];
    if (c == n) {
        f32x2 x2[VEC / 2];
#pragma unroll
        for (int w = 0; w < VEC / 2; ++w) x2[w] = f32x2{0.f, 0.f};
#pragma unroll
        for (int q = 0; q < Q; ++q) {
            const f32x2 k2 = {ki[q], ki[q]};
#pragma unroll
            for (int w = 0; w < VEC / 2; ++w)
                x2[w] = __builtin_elementwise_fma(f32x2{bf16_lo(sp[q][w]), bf16_hi(sp[q][w])}, k2, x2[w]);
        }
#pragma unroll
        for (int w = 0; w < VEC / 2; ++w) {
            x[2 * w] = x2[w].x;
            x[2 * w + 1] = x2[w].y;
        }
        column_sums(x, 0);
        lds_barrier();
        column_totals(x, 0);
        const float g = __expf(g_log);
#pragma unroll
        for (int w = 0; w < VEC / 2; ++w) {
            d2[w] = f32x2{(vv[2 * w] - x[2 * w] * (g * rk)) * beta, (vv[2 * w + 1] - x[2 * w + 1] * (g * rk)) * beta};
            x2[w] = f32x2{0.f, 0.f};
        }
        const f32x2 g2 = {g, g};
        const bool own = fold && commit == n + 1;
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
            if (own) state_store(head + (rg + RG * q) * VD + j0, st);
        }
#pragma unroll
        for (int w = 0; w < VEC / 2; ++w) {
            x[2 * w] = x2[w].x;
            x[2 * w + 1] = x2[w].y;
        }
        column_sums(x, 1);
        lds_barrier();
        column_totals(x, 1);
#pragma unroll
        for (int u = 0; u < VEC; ++u) x[u] *= rq;
    } else {
        float acc_a[VEC], acc_o[VEC];
#pragma unroll
        for (int u = 0; u < VEC; ++u) {
            acc_a[u] = 0.f;
            acc_o[u] = 0.f;
        }
        float after = g_log;
        for (int hi = n; hi > c; ) {
            const int lo = whole ? c : max(c, hi - SCH);
            const int m = hi - lo;
            // The staged index of row lo.
            const int off = whole ? lo : 0;
            if (!whole) stage(lo, m);
            if (wave == 0) {
                const float gl = l < m ? c_glog[off + l] : 0.f;
                float later = gl;
#pragma unroll
                for (int off = 1; off < kWave; off <<= 1) {
                    const float down = __shfl_down(later, off, kWave);
                    if (l + off < kWave) later += down;
                }
                const float past = __shfl_down(later, 1, kWave);
                const float excl = l + 1 < m ? past : 0.f;
                if (l < m) {
                    const float w = __expf(after + excl);
                    c_wa[l] = w * c_dk[off + l];
                    c_wo[l] = w * c_dq[off + l];
                }
                after += __shfl(later, 0, kWave);
            }
            lds_barrier();
            for (int e = rg; e < m; e += RG) {
                const float wa = c_wa[e], wo = c_wo[e];
#pragma unroll
                for (int u = 0; u < VEC; ++u) {
                    acc_a[u] += wa * s_u[off + e][j0 + u];
                    acc_o[u] += wo * s_u[off + e][j0 + u];
                }
            }
            lds_barrier();
            hi = lo;
        }
        if (tid == 0) g_all = __expf(after);
        lds_barrier();
        const float g0 = g_all;
        float xq[VEC];
#pragma unroll
        for (int u = 0; u < VEC; ++u) {
            x[u] = 0.f;
            xq[u] = 0.f;
        }
#pragma unroll
        for (int q = 0; q < Q; ++q) {
#pragma unroll
            for (int w = 0; w < VEC / 2; ++w) {
                const float s0 = bf16_lo(sp[q][w]), s1 = bf16_hi(sp[q][w]);
                x[2 * w] += s0 * ki[q];
                x[2 * w + 1] += s1 * ki[q];
                xq[2 * w] += s0 * qi[q];
                xq[2 * w + 1] += s1 * qi[q];
            }
        }
#pragma unroll
        for (int u = 0; u < VEC; ++u) {
            x[u] = g0 * rk * x[u] + acc_a[u];
            xq[u] = g0 * rq * xq[u] + acc_o[u];
        }
        column_sums(x, 0);
        column_sums(xq, 1);
        lds_barrier();
        column_totals(x, 0);
        column_totals(xq, 1);
        const float kq = qkn * rk * rq;
#pragma unroll
        for (int w = 0; w < VEC / 2; ++w) {
            d2[w] = f32x2{(vv[2 * w] - x[2 * w]) * beta, (vv[2 * w + 1] - x[2 * w + 1]) * beta};
            x[2 * w] = xq[2 * w] + kq * d2[w].x;
            x[2 * w + 1] = xq[2 * w + 1] + kq * d2[w].y;
        }
    }
    if (tid < CG) {
        float* o = out + (row * V_h + h) * VD + j0;
        *reinterpret_cast<f32x4*>(o) = f32x4{x[0], x[1], x[2], x[3]};
        *reinterpret_cast<f32x4*>(o + 4) = f32x4{x[4], x[5], x[6], x[7]};
        if (float* dst = step.own_row<float>(cl, 0, 2)) {
            float* d = dst + static_cast<long long>(h) * VD + j0;
            *reinterpret_cast<f32x4*>(d) = f32x4{d2[0].x, d2[0].y, d2[1].x, d2[1].y};
            *reinterpret_cast<f32x4*>(d + 4) = f32x4{d2[2].x, d2[2].y, d2[3].x, d2[3].y};
        }
    }
    if (tid < 2) {
        if (bf16* dst = step.own_row<bf16>(cl, 0, 0)) {
            const int at2 = tid * V_h + h;
            dst[at2] = ba[row * 2 * V_h + at2];
        }
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
    const float* __restrict__ g_log, const float* __restrict__ beta, StateRows rows,
    const u32* __restrict__ indptr, float* __restrict__ out, int K_h, int V_h, int K_d, int V_d, bool write_state,
    const int* __restrict__ commit_len, const u8* __restrict__ mask, const int* __restrict__ begin_at,
    bool fused_decay, const int* __restrict__ snap, const int* __restrict__ snap_rows,
    const i32* __restrict__ snap_slots, const u32* __restrict__ win) {
    __shared__ float sq[BK], sk[BK];
    const int r = blockIdx.y, h = blockIdx.z;
    const int j = blockIdx.x * BV + threadIdx.x;
    if (win != nullptr && r >= static_cast<int>(win[2])) return;
    const int rl = win != nullptr ? r + static_cast<int>(win[3]) : r;
    const long long row0 = win != nullptr ? static_cast<long long>(win[1]) : 0;
    const Span sp = span_of(indptr, r, rl, commit_len, begin_at);
    if (sp.n <= 0) return;
    const i32* rid = rows.row(rl);
    if (rid[0] < 0) return;
    const bool col = j < V_d;
    const int h_k = h / (V_h / K_h);
    bf16* state = rows.atom<bf16>(rid, h, static_cast<long long>(K_d) * V_d);

    bf16 s[BK];
#pragma unroll
    for (int i = 0; i < BK; ++i) s[i] = (col && i < K_d) ? state[static_cast<long long>(i) * V_d + j] : from_f32<bf16>(0.f);
    Snaps sn = snaps_of(snap, rl);

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
        // Every loop over the state runs all BK entries, those past K_d
        // predicated off: a loop that breaks at K_d does not unroll, and the
        // state it indexes is then kept in scratch.
        float kv = 0.f;
#pragma unroll
        for (int i = 0; i < BK; ++i) {
            if (i < K_d) {
                const float decayed = to_f32(s[i]) * g;
                if (!fused_decay) s[i] = from_f32<bf16>(decayed);
                kv += decayed * sk[i];
            }
        }
        const float delta = col ? (v[at * V_h * V_d + static_cast<long long>(h) * V_d + j] - kv) * b : 0.f;
        float o = 0.f;
#pragma unroll
        for (int i = 0; i < BK; ++i) {
            if (i < K_d) {
                const float next = (fused_decay ? to_f32(s[i]) * g : to_f32(s[i])) + sk[i] * delta;
                s[i] = from_f32<bf16>(next);
                o += next * sq[i];
            }
        }
        if (col) out[((sp.t0 + t + row0) * V_h + h) * V_d + j] = o;
        if (sn.e0 < sn.e1 && snap_rows[sn.e0] == t + 1) {
            bf16* at = rows.atom<bf16>(snap_slots + static_cast<long long>(sn.e0) * rows.pieces, h,
                                       static_cast<long long>(K_d) * V_d);
            if (col && folds(write_state, mask, rl)) {
#pragma unroll
                for (int i = 0; i < BK; ++i) {
                    if (i < K_d) at[static_cast<long long>(i) * V_d + j] = s[i];
                }
            }
            ++sn.e0;
        }
        __syncthreads();
    }

    if (!col || !folds(write_state, mask, rl)) return;
#pragma unroll
    for (int i = 0; i < BK; ++i) {
        if (i < K_d) state[static_cast<long long>(i) * V_d + j] = s[i];
    }
}


// ------------------------------------------------- prefill (chunk-parallel) --

// The same recurrence over 64-token chunks (the delta rule's WY form): `wu`
// builds W and U per chunk, `h` walks a request's chunks carrying S in f32
// (a block a 16-column slice), `o` lands each chunk's output from the S `h`
// recorded at its start. bf16 MFMA, f32 accumulation; the state rounds only
// where stored. Scratch chunk slot s = ⌊t0 / CH⌋ + r + j keeps requests'
// chunks disjoint without a prefix scan.


// Grid (slots, V_h), block 256, straight from the post-conv qkv rows and the
// fused [g_log | β] gates (both at launch row + row_offset). Lands Γ (gcum,
// f32 [n][V_h]), the bf16 q̂ and k̂ rows (qb, kb [n][K_h][BK]), k̂ᵀ per chunk
// (kT [s][K_h][BK][CH], zero past the chunk's rows), W ([n][V_h][BK]) and Uᵀ
// per chunk (uT [s][V_h][BV][CH]); q̂, k̂ are l2-normed as the staging
// kernels norm them.
template <int BK, int BV>
__global__ __launch_bounds__(512) void ssm_gdn_chunk_wu(
    const bf16* __restrict__ qkv, const float* __restrict__ gates, StateRows rows,
    const u32* __restrict__ indptr, const int* __restrict__ commit_len, const int* __restrict__ begin_at,
    float* __restrict__ gcum, bf16* __restrict__ qb, bf16* __restrict__ kb_out, bf16* __restrict__ kT,
    bf16* __restrict__ w_out, bf16* __restrict__ u_out, int lanes, int K_h, int V_h, int conv_dim, float q_scale,
    const int* __restrict__ snap, const int* __restrict__ snap_rows, const u32* __restrict__ win) {
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
    const int r = gdn_slot_lane(s, lanes, indptr, begin_at, snap, win);
    const GdnLane ln = gdn_lane(r, indptr, rows, commit_len, begin_at, snap, win);
    if (ln.T <= 0 || ln.slot < 0) return;
    const int j = s - gdn_chunk_base(r, indptr, begin_at, snap, win);
    if (j < 0) return;
    const GdnChunk ck = gdn_chunk(ln, j, snap_rows);
    if (ck.L <= 0) return;
    const int ta = ck.ta;
    const int L = ck.L;
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

// Grid (lanes · V_h · BV/BVT), block 256: one BVT-column slice of one
// head's state through every chunk of one request (state columns evolve
// independently), in f32 registers in the matrix core's D layout:
// wavefront w holds key rows [16w, 16w+16) and [16(w+4), 16(w+4)+16) of
// each 16-column tile. Records each chunk's entering state (hcT
// [s][V_h][BV][BK]) and V' (vnT [s][V_h][BV][CH], zero past the chunk's
// rows). The next chunk's operands are loaded while this one computes, two
// wavefronts a SIMD (a deeper ring spills them). A wider slice reads a
// chunk's W and k̂ᵀ for more columns; a narrower one fills the device at
// fewer requests.
template <int BK, int BV, int BVT>
__global__ __launch_bounds__(256) __attribute__((amdgpu_waves_per_eu(2, 2))) void ssm_gdn_chunk_h(
    const float* __restrict__ gcum, const bf16* __restrict__ kT, const bf16* __restrict__ w,
    const bf16* __restrict__ uT, bf16* __restrict__ vnT, bf16* __restrict__ hcT, StateRows rows,
    const u32* __restrict__ indptr, const int* __restrict__ commit_len, const int* __restrict__ begin_at, bool write_state,
    const u8* __restrict__ mask, int lanes, int K_h, int V_h, const int* __restrict__ snap,
    const int* __restrict__ snap_rows, const i32* __restrict__ snap_slots, const u32* __restrict__ win) {
    constexpr int CH = GDN_CH, NT = BVT / 16, KT = BK / 16 / 4, KS = BK / 32, CS = CH / 32;
    // LDS rows padded by 16 bytes against the 16-row reads' bank conflicts.
    constexpr int HP = BK + 8, VP = CH + 8;
    static_assert(BK % 64 == 0, "four wavefronts split the key rows evenly");
    static_assert(BVT % 16 == 0 && BV % BVT == 0, "whole 16-column tiles");
    __shared__ alignas(16) bf16 hT[BVT * HP];
    __shared__ alignas(16) bf16 vnTs[BVT * VP];  // V'·exp(Γ_L − Γ), the update's operand
    __shared__ alignas(16) bf16 vnL[BVT * CH];   // V' as recorded
    __shared__ float sE[CH];

    // Each XCD (block b runs on XCD b mod 8) takes a run of the live blocks
    // (a captured grid's tail past the live lanes exits), so a head's
    // slices, which read the same W and k̂ᵀ chunks, share its L2.
    if (win != nullptr) lanes = min(lanes, static_cast<int>(win[2]));
    const int live = lanes * (BV / BVT) * V_h, b = blockIdx.x;
    if (b >= live) return;
    const int xcd = b % 8, q8 = live / 8, rem = live % 8;
    const int at = xcd * q8 + min(xcd, rem) + b / 8;
    const int vt = at % (BV / BVT);
    const int h = (at / (BV / BVT)) % V_h;
    const int r = at / ((BV / BVT) * V_h);
    const GdnLane ln = gdn_lane(r, indptr, rows, commit_len, begin_at, snap, win);
    if (ln.T <= 0 || ln.slot < 0) return;
    const int hk = h / (V_h / K_h);
    const int v0 = vt * BVT;
    const int tid = threadIdx.x, wave = tid / kWave, l = lane();
    bf16* state = rows.atom<bf16>(rows.row(ln.rl), h, static_cast<long long>(BK) * BV);

    float hf[KT][NT][4];
#pragma unroll
    for (int q = 0; q < KT; ++q)
#pragma unroll
        for (int n = 0; n < NT; ++n)
            mfma_each(mfma_zero(), [&](int i, int vv, float) {
                hf[q][n][i % 4] = to_f32(state[static_cast<long long>((wave + 4 * q) * 16 + i) * BV + v0 + 16 * n + vv]);
            });

    // One chunk's operands for this lane: W's A fragments for the
    // wavefront's 16 rows, U at its D cells, k̂ᵀ's A fragments for its key
    // rows, and (wavefront 0) Γ at its row and at the chunk's last.
    struct Ops {
        mfma_ab2 wf[KS];
        mfma_ab uv[NT];
        mfma_ab2 kf[KT][CS];
        float g, gl;
    };
    const int base = gdn_chunk_base(r, indptr, begin_at, snap, win);
    const int chunks = gdn_chunks(ln, snap_rows);
    const int g8 = 8 * (l / 16);
    const bool folding = folds(write_state, mask, ln.rl);
    auto fetch = [&](int jc, Ops& o) {
        const GdnChunk ck = gdn_chunk(ln, jc, snap_rows);
        const int sj = base + jc, ta = ck.ta, L = ck.L;
        const int m0 = wave * 16;
        const bf16* wr = w + (static_cast<long long>(ta + m0 + l % 16) * V_h + h) * BK + g8;
#pragma unroll
        for (int st = 0; st < KS; ++st) o.wf[st] = load8(wr + st * 32);
        // Uᵀ: the lane's four rows of its column, adjacent.
#pragma unroll
        for (int n = 0; n < NT; ++n)
            o.uv[n] = load4(uT + ((static_cast<long long>(sj) * V_h + h) * BV + v0 + 16 * n + l % 16) * CH + m0 + 4 * (l / 16));
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
    constexpr int DEPTH = 2;
    Ops ring[DEPTH];
#pragma unroll
    for (int d = 0; d < DEPTH; ++d) fetch(min(d, chunks - 1), ring[d]);
    auto step = [&](int jc, Ops& o) {
        const int sj = base + jc;
        const GdnChunk ck = gdn_chunk(ln, jc, snap_rows);
        const int L = ck.L;

#pragma unroll
        for (int q = 0; q < KT; ++q)
#pragma unroll
            for (int n = 0; n < NT; ++n)
                mfma_each(mfma_zero(), [&](int i, int vv, float) {
                    hT[(16 * n + vv) * HP + (wave + 4 * q) * 16 + i] = from_f32<bf16>(hf[q][n][i % 4]);
                });
        if (wave == 0) sE[l] = l < L ? __expf(o.gl - o.g) : 0.f;
        lds_barrier();

        // The entering state, recorded as a contiguous slice.
        bf16* hcT_s = hcT + ((static_cast<long long>(sj) * V_h + h) * BV + v0) * BK;
        static_assert(BVT * BK / 8 % 256 == 0, "whole 16-byte pieces a thread");
#pragma unroll
        for (int e = tid; e < BVT * BK / 8; e += 256)
            *reinterpret_cast<b128*>(hcT_s + (e / 16) * BK + (e % 16) * 8) =
                *reinterpret_cast<const b128*>(hT + (e / 16) * HP + (e % 16) * 8);

        // V' = U − W·S for the wavefront's 16 chunk rows.
        {
            const int m0 = wave * 16;
#pragma unroll
            for (int n = 0; n < NT; ++n) {
                mfma_cd c = mfma_zero();
#pragma unroll
                for (int st = 0; st < KS; ++st)
                    c = mfma16x2(o.wf[st], load8(hT + (16 * n + l % 16) * HP + st * 32 + g8), c);
                const bf16* uv = reinterpret_cast<const bf16*>(&o.uv[n]);
                float vn[4], vs[4];
#pragma unroll
                for (int t = 0; t < 4; ++t) {
                    const int i = m0 + 4 * (l / 16) + t;
                    vn[t] = i < L ? to_f32(uv[t]) - c[t] : 0.f;
                    vs[t] = vn[t] * sE[i];
                }
                const int vv = 16 * n + l % 16, i0 = m0 + 4 * (l / 16);
                *reinterpret_cast<mfma_ab*>(vnL + vv * CH + i0) = pack4(vn[0], vn[1], vn[2], vn[3]);
                *reinterpret_cast<mfma_ab*>(vnTs + vv * VP + i0) = pack4(vs[0], vs[1], vs[2], vs[3]);
            }
        }
        lds_barrier();

        bf16* vnT_s = vnT + ((static_cast<long long>(sj) * V_h + h) * BV + v0) * CH;
        for (int e = tid; e < BVT * CH / 8; e += 256) reinterpret_cast<b128*>(vnT_s)[e] = reinterpret_cast<const b128*>(vnL)[e];

        // S ← exp(Γ_L)·S + K̂ᵀ·diag(exp(Γ_L − Γ))·V'.
        const float decay = __expf(o.gl);
#pragma unroll
        for (int q = 0; q < KT; ++q)
#pragma unroll
            for (int n = 0; n < NT; ++n) {
                mfma_cd c = mfma_zero();
#pragma unroll
                for (int st = 0; st < CS; ++st)
                    c = mfma16x2(o.kf[q][st], load8(vnTs + (16 * n + l % 16) * VP + st * 32 + g8), c);
#pragma unroll
                for (int t = 0; t < 4; ++t) hf[q][n][t] = hf[q][n][t] * decay + c[t];
            }
        // A snapshot lands the state as stored and the walk carries on from
        // it rounded, as a pass cut there reloads it.
        if (ck.snap >= 0) {
            bf16* sat = rows.atom<bf16>(snap_slots + static_cast<long long>(ck.snap) * rows.pieces, h,
                                        static_cast<long long>(BK) * BV);
#pragma unroll
            for (int q = 0; q < KT; ++q)
#pragma unroll
                for (int n = 0; n < NT; ++n)
                    mfma_each(mfma_zero(), [&](int i, int vv, float) {
                        const bf16 x = from_f32<bf16>(hf[q][n][i % 4]);
                        if (folding) sat[static_cast<long long>((wave + 4 * q) * 16 + i) * BV + v0 + 16 * n + vv] = x;
                        hf[q][n][i % 4] = to_f32(x);
                    });
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
#pragma unroll
        for (int n = 0; n < NT; ++n)
            mfma_each(mfma_zero(), [&](int i, int vv, float) {
                state[static_cast<long long>((wave + 4 * q) * 16 + i) * BV + v0 + 16 * n + vv] = from_f32<bf16>(hf[q][n][i % 4]);
            });
}

// Grid (slots, V_h), block 256: the chunk's output rows, f32 [n][V_h][BV]
// at row + row_offset. Wavefront w lands chunk rows [16w, 16w+16).
template <int BK, int BV>
__global__ __launch_bounds__(256) void ssm_gdn_chunk_o(
    const float* __restrict__ gcum, const bf16* __restrict__ qb, const bf16* __restrict__ kb,
    const bf16* __restrict__ vnT, const bf16* __restrict__ hcT, StateRows rows,
    const u32* __restrict__ indptr, const int* __restrict__ commit_len, const int* __restrict__ begin_at,
    float* __restrict__ out, int lanes, int K_h, int V_h, const int* __restrict__ snap,
    const int* __restrict__ snap_rows, const i32* __restrict__ snap_slots, const u32* __restrict__ win) {
    constexpr int CH = GDN_CH, BVT = 16, KS = BK / 32, CS = CH / 32;
    constexpr int SP = CH + 8;  // padded against the 16-row reads' bank conflicts
    __shared__ alignas(16) bf16 Sb[CH * SP];
    __shared__ float sG[CH];

    const int h = blockIdx.y, s = blockIdx.x;
    if (win != nullptr) lanes = min(lanes, static_cast<int>(win[2]));
    if (lanes <= 0) return;
    const int r = gdn_slot_lane(s, lanes, indptr, begin_at, snap, win);
    const GdnLane ln = gdn_lane(r, indptr, rows, commit_len, begin_at, snap, win);
    if (ln.T <= 0 || ln.slot < 0) return;
    const int j = s - gdn_chunk_base(r, indptr, begin_at, snap, win);
    if (j < 0) return;
    const GdnChunk ck = gdn_chunk(ln, j, snap_rows);
    if (ck.L <= 0) return;
    const int ta = ck.ta;
    const int L = ck.L;
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
