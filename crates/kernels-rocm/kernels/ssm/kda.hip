// Kimi Delta Attention: Gated DeltaNet with a decay per key channel.

#include "ssm/common.hip"

namespace pie::attn {

// ------------------------------------------------- Kimi Delta Attention --

// KDA: the delta rule with a decay per key channel. Head h's state St[v][k]
// is f32 at slab + slot·stride + h·D·D + v·D + k: St ← St·a, δ = (v −
// St·k̂)·β, St ← St + δ k̂ᵀ, o = St·q̂. The decode step and chunk passes norm
// and gate the post-conv [q | k | v], f and b rows where they stand (row +
// row_offset); the token-by-token prefill reads them staged in f32.


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

template <int BLOCK>
__global__ __launch_bounds__(BLOCK) void ssm_kda_qkv_prep(const bf16* __restrict__ mixed, float* __restrict__ q_out,
                                                          float* __restrict__ k_out, float* __restrict__ v_out,
                                                          int width, int D, float eps, const u32* __restrict__ win) {
    constexpr int WAVES = BLOCK / kWave;
    const int n = blockIdx.x, plane = blockIdx.y, l = lane();
    if (win != nullptr && n >= static_cast<int>(win[0])) return;
    const long long row = win != nullptr ? n + static_cast<long long>(win[1]) : n;
    const float scale = plane == 0 ? rsqrtf(static_cast<float>(D)) : 1.f;
    const bf16* src = mixed + row * 3 * width + static_cast<long long>(plane) * width;
    float* dst = (plane == 0 ? q_out : plane == 1 ? k_out : v_out) + static_cast<long long>(n) * width;
    for (int i = 0; i < kKdaPrepHeads; ++i) {
        const int h = (blockIdx.z * kKdaPrepHeads + i) * WAVES + static_cast<int>(threadIdx.x) / kWave;
        if (h >= width / D) return;
        const bf16* x = src + static_cast<long long>(h) * D;
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
__global__ void ssm_kda_gate_beta(const bf16* __restrict__ f, const bf16* __restrict__ b, const float* __restrict__ a_log,
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
    const float* __restrict__ a_log, const float* __restrict__ dt_bias, StateRows rows,
    float* __restrict__ out, int H, float eps, float floor,
    bool write_state, const u8* __restrict__ mask, const u32* __restrict__ win) {
    constexpr int E = D / 32, RPP = 256 / kKdaRowLanes, R = ROWS / RPP, WAVES = 256 / kWave;
    static_assert(D % 32 == 0 && D <= 2 * kWave && ROWS % RPP == 0 && D % ROWS == 0,
                  "whole pieces, two channels a lane, whole passes");
    const int h = blockIdx.x, r = blockIdx.y;
    if (win != nullptr && r >= static_cast<int>(win[0])) return;
    const long long row = win != nullptr ? r + static_cast<long long>(win[1]) : r;
    const int rl = win != nullptr ? r + static_cast<int>(win[3]) : r;
    const i32* rid = rows.row(rl);
    if (rid[0] < 0) return;
    const int tid = threadIdx.x, wave = tid / kWave, l = lane();
    const int c = tid % kKdaRowLanes;
    const int v0 = blockIdx.z * ROWS + tid / kKdaRowLanes;
    const long long width = static_cast<long long>(H) * D;
    float* head = rows.atom<float>(rid, h, static_cast<long long>(D) * D);
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

// `ssm_kda_step_batched` over a lane's buffered rows then its own, the
// buffered rows read as a linear attention over the committed state (each
// keeps its forget and beta rows, convolved keys and corrected value u:
// planes 0 to 3): M_t diag(a_{t+1}) = M0 diag(G_{0,t+1}) + sum_s u_s
// (k_s . G_{s,t+1})^T, so the own row's u and output read the state once,
// its q and k decayed to it, and each buffered row as two decay-weighted
// dots. The per-channel decays are suffix sums of f32 log-gates down the
// buffered rows, a thread a channel, a chunk of rows staged at a time. A
// fold applies M <- M diag(a) + u k^T a committed row, then the own row's
// where it commits that too; the rows are stored once. The own row's forget
// and beta rows and its u land in its buffer.
template <int D, int ROWS>
__global__ __launch_bounds__(256) void ssm_kda_step_carried(
    const bf16* __restrict__ mixed, const bf16* __restrict__ f, const bf16* __restrict__ b,
    const float* __restrict__ a_log, const float* __restrict__ dt_bias, StateRows rows,
    float* __restrict__ out, int H, float eps, float floor,
    bool write_state, const u8* __restrict__ mask, Carry carry, const u32* __restrict__ win) {
    constexpr int E = D / 32, RPP = 256 / kKdaRowLanes, R = ROWS / RPP, WAVES = 256 / kWave, CH = 16;
    static_assert(D % 32 == 0 && D <= 2 * kWave && ROWS % RPP == 0 && D % ROWS == 0,
                  "whole pieces, two channels a lane, whole passes");
    __shared__ alignas(16) float sq[D], sk[D], sgl[D], sg0[D];
    __shared__ alignas(16) float ck[CH][D], cg[CH][D];
    __shared__ float cdq[CH], cdk[CH];
    __shared__ float part[3][WAVES];
    const int h = blockIdx.x, r = blockIdx.y;
    if (win != nullptr && r >= static_cast<int>(win[0])) return;
    const long long row = win != nullptr ? r + static_cast<long long>(win[1]) : r;
    const int rl = win != nullptr ? r + static_cast<int>(win[3]) : r;
    const int tid = threadIdx.x, wave = tid / kWave, l = lane();
    const long long width = static_cast<long long>(H) * D;
    const CarryLane cl = carry.lane(rl);
    if (blockIdx.z == 0) {
        if (bf16* dst = carry.own_row<bf16>(cl, 0, 0)) {
            for (int d = tid; d < D; d += 256) dst[static_cast<long long>(h) * D + d] = f[row * width + static_cast<long long>(h) * D + d];
        }
        if (tid == 0) {
            if (bf16* dst = carry.own_row<bf16>(cl, 0, 1)) dst[h] = b[row * H + h];
        }
    }
    const i32* rid = rows.row(rl);
    if (rid[0] < 0) return;
    const int c = tid % kKdaRowLanes;
    const int v0 = blockIdx.z * ROWS + tid / kKdaRowLanes;
    float* head = rows.atom<float>(rid, h, static_cast<long long>(D) * D);
    f32x4 s[R][E];
#pragma unroll
    for (int p = 0; p < R; ++p)
#pragma unroll
        for (int e = 0; e < E; ++e)
            s[p][e] = state_load(reinterpret_cast<const f32x4*>(head + static_cast<long long>(v0 + RPP * p) * D + 4 * c + 32 * e));
    const bf16* q = mixed + row * 3 * width + static_cast<long long>(h) * D;
    float vv[R];
#pragma unroll
    for (int p = 0; p < R; ++p) vv[p] = to_f32(q[2 * width + v0 + RPP * p]);
    const float a = __expf(a_log[h]);
    const float beta = sigmoid(to_f32(b[row * H + h]));
    // The own row's q̂, k̂ and log-gates, and k̂·q̂.
    const float qv = tid < D ? to_f32(q[tid]) : 0.f, kv = tid < D ? to_f32(q[width + tid]) : 0.f;
    const float w0 = wave_sum(qv * qv), w1 = wave_sum(kv * kv), w2 = wave_sum(qv * kv);
    if (l == 0) {
        part[0][wave] = w0;
        part[1][wave] = w1;
        part[2][wave] = w2;
    }
    lds_barrier();
    float qn = 0.f, kn = 0.f, qkn = 0.f;
#pragma unroll
    for (int w = 0; w < WAVES; ++w) {
        qn += part[0][w];
        kn += part[1][w];
        qkn += part[2][w];
    }
    const float rq = rsqrtf(qn + eps) * rsqrtf(static_cast<float>(D)), rk = rsqrtf(kn + eps);
    if (tid < D) {
        sq[tid] = qv * rq;
        sk[tid] = kv * rk;
        sgl[tid] = kda_gate(to_f32(f[row * width + static_cast<long long>(h) * D + tid]) + dt_bias[h * D + tid], a, floor);
    }
    lds_barrier();
    float after = tid < D ? sgl[tid] : 0.f;
    float pq[R], pk[R];
#pragma unroll
    for (int p = 0; p < R; ++p) {
        pq[p] = 0.f;
        pk[p] = 0.f;
    }
    const int n = carry.carried(cl);
    for (int cb = ((n + CH - 1) / CH - 1) * CH; cb >= 0; cb -= CH) {
        const int m = min(CH, n - cb);
        // The chunk's keys and log-gates staged, then each row's keys normed.
        for (int x = tid; x < m * D; x += 256) {
            const int e = x / D, d = x % D;
            u64 slot;
            u32 pos;
            carry.carried_at(cl, cb + e, slot, pos);
            ck[e][d] = to_f32(carry.plane_at<const bf16>(slot, pos, 2)[static_cast<long long>(h) * D + d]);
            cg[e][d] = kda_gate(to_f32(carry.plane_at<const bf16>(slot, pos, 0)[static_cast<long long>(h) * D + d]) +
                                    dt_bias[h * D + d], a, floor);
        }
        lds_barrier();
        for (int e = wave; e < m; e += WAVES) {
            float ss = 0.f;
            for (int d = l; d < D; d += kWave) ss += ck[e][d] * ck[e][d];
            const float inv = rsqrtf(wave_sum(ss) + eps);
            for (int d = l; d < D; d += kWave) ck[e][d] *= inv;
        }
        lds_barrier();
        // A thread a channel decays each row's key to the own row.
        if (tid < D) {
            float run = after;
            for (int e = m - 1; e >= 0; --e) {
                ck[e][tid] *= __expf(run);
                run += cg[e][tid];
            }
            after = run;
        }
        lds_barrier();
        // A wavefront a row: its decayed keys' dots with the own q̂ and k̂.
        for (int e = wave; e < m; e += WAVES) {
            float dq = 0.f, dk = 0.f;
            for (int d = l; d < D; d += kWave) {
                dq += ck[e][d] * sq[d];
                dk += ck[e][d] * sk[d];
            }
            dq = wave_sum(dq);
            dk = wave_sum(dk);
            if (l == 0) {
                cdq[e] = dq;
                cdk[e] = dk;
            }
        }
        lds_barrier();
        // A row's lane 0 weighs each buffered row's u on its value rows.
        if (c == 0) {
            for (int e = 0; e < m; ++e) {
                u64 slot;
                u32 pos;
                carry.carried_at(cl, cb + e, slot, pos);
                const float* ur = carry.plane_at<const float>(slot, pos, 3) + static_cast<long long>(h) * D;
                const float dq = cdq[e], dk = cdk[e];
#pragma unroll
                for (int p = 0; p < R; ++p) {
                    const float u = ur[v0 + RPP * p];
                    pq[p] += dq * u;
                    pk[p] += dk * u;
                }
            }
        }
        lds_barrier();
    }
    if (tid < D) sg0[tid] = __expf(after);
    lds_barrier();
    // The committed state's part, decayed to the own row.
    float mem[R], acc[R];
#pragma unroll
    for (int p = 0; p < R; ++p) {
        float km = 0.f, qm = 0.f;
#pragma unroll
        for (int e = 0; e < E; ++e) {
            const int at = 4 * c + 32 * e;
#pragma unroll
            for (int u = 0; u < 4; ++u) {
                const float sd = s[p][e][u] * sg0[at + u];
                km += sd * sk[at + u];
                qm += sd * sq[at + u];
            }
        }
        mem[p] = row8_sum(km + pk[p]);
        acc[p] = row8_sum(qm + pq[p]);
    }
    const float kq = qkn * rq * rk;
    float un[R];
    float* o = out + (row * H + h) * D;
    float* own_u = carry.own_row<float>(cl, 0, 3);
#pragma unroll
    for (int p = 0; p < R; ++p) {
        un[p] = (vv[p] - mem[p]) * beta;
        if (c == 0) {
            o[v0 + RPP * p] = acc[p] + kq * un[p];
            if (own_u != nullptr) own_u[static_cast<long long>(h) * D + v0 + RPP * p] = un[p];
        }
    }
    const int commit = carry.commit(rl, n + 1);
    if (!folds_uniform(write_state, mask, rl) || commit <= 0) return;
    // The rows the lane commits fold into the state: its buffered ones, then
    // its own where it commits that too.
    const int folded = min(commit, n);
    for (int t = 0; t < folded; ++t) {
        u64 slot;
        u32 pos;
        carry.carried_at(cl, t, slot, pos);
        lds_barrier();
        const float kx = tid < D ? to_f32(carry.plane_at<const bf16>(slot, pos, 2)[static_cast<long long>(h) * D + tid]) : 0.f;
        const float ws = wave_sum(kx * kx);
        if (l == 0) part[0][wave] = ws;
        if (tid < D)
            cg[0][tid] = __expf(kda_gate(to_f32(carry.plane_at<const bf16>(slot, pos, 0)[static_cast<long long>(h) * D + tid]) +
                                             dt_bias[h * D + tid], a, floor));
        lds_barrier();
        float kn2 = 0.f;
#pragma unroll
        for (int w = 0; w < WAVES; ++w) kn2 += part[0][w];
        if (tid < D) ck[0][tid] = kx * rsqrtf(kn2 + eps);
        lds_barrier();
        const float* ur = carry.plane_at<const float>(slot, pos, 3) + static_cast<long long>(h) * D;
#pragma unroll
        for (int p = 0; p < R; ++p) {
            const float u = ur[v0 + RPP * p];
#pragma unroll
            for (int e = 0; e < E; ++e) {
                const int at = 4 * c + 32 * e;
#pragma unroll
                for (int x = 0; x < 4; ++x) s[p][e][x] = s[p][e][x] * cg[0][at + x] + u * ck[0][at + x];
            }
        }
    }
    if (commit == n + 1) {
#pragma unroll
        for (int p = 0; p < R; ++p) {
#pragma unroll
            for (int e = 0; e < E; ++e) {
                const int at = 4 * c + 32 * e;
#pragma unroll
                for (int x = 0; x < 4; ++x) s[p][e][x] = s[p][e][x] * __expf(sgl[at + x]) + un[p] * sk[at + x];
            }
        }
    }
#pragma unroll
    for (int p = 0; p < R; ++p)
#pragma unroll
        for (int e = 0; e < E; ++e)
            state_store(reinterpret_cast<f32x4*>(head + static_cast<long long>(v0 + RPP * p) * D + 4 * c + 32 * e), s[p][e]);
}

// The buffered rows a chunked KDA fire's lanes land past their commit,
// made ready for a later decode's linear attention: each row's convolved
// keys (plane 2) and corrected value u (plane 3), from the state as the
// fire left it (folded through the lane's commit, or as it stood where the
// lane folds nothing), stepping the lane's rows past the commit in order (a
// buffered one by its stored keys and u, an own one forming its u). The
// state stays as it was. The own rows' [q | k | v], forget and beta rows are
// read where the fire lays them: the lane's extended rows (`ext`: row
// `RsLane::ext + j`) or its own (`qo_indptr`, the window's row offset
// `win[1]`). Grid (H, lanes, D / ROWS), block 256.
template <int D, int ROWS>
__global__ __launch_bounds__(256) void ssm_kda_tail_corrected(
    const bf16* __restrict__ mixed, const bf16* __restrict__ f, const bf16* __restrict__ b,
    const float* __restrict__ a_log, const float* __restrict__ dt_bias, StateRows rows,
    const u32* __restrict__ qo_indptr, bool ext, int H, float eps, float floor,
    bool write_state, const u8* __restrict__ mask, int lanes_stated, Carry carry, const u32* __restrict__ win) {
    constexpr int E = D / 32, RPP = 256 / kKdaRowLanes, R = ROWS / RPP, WAVES = 256 / kWave;
    __shared__ alignas(16) float sk[D], sa[D];
    __shared__ float part[WAVES];
    const int lanes = win != nullptr ? static_cast<int>(win[2]) : lanes_stated;
    const int h = blockIdx.x, r = blockIdx.y;
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
    const int tid = threadIdx.x, wave = tid / kWave, l = lane();
    const int c = tid % kKdaRowLanes;
    const int v0 = blockIdx.z * ROWS + tid / kKdaRowLanes;
    const long long width = static_cast<long long>(H) * D;
    const float* head = rows.atom<float>(rid, h, static_cast<long long>(D) * D);
    f32x4 s[R][E];
#pragma unroll
    for (int p = 0; p < R; ++p)
#pragma unroll
        for (int e = 0; e < E; ++e)
            s[p][e] = *reinterpret_cast<const f32x4*>(head + static_cast<long long>(v0 + RPP * p) * D + 4 * c + 32 * e);
    const float a = __expf(a_log[h]);
    for (int j = start; j < run; ++j) {
        const bool carried = j < n;
        const int e_own = j - n;
        const long long at = ext ? static_cast<long long>(cl.ext) + j : row0 + qo_indptr[r] + e_own;
        const bf16* krow;
        const bf16* frow;
        const float* urow = nullptr;
        const bf16* qrow = nullptr;
        if (carried) {
            u64 slot;
            u32 pos;
            carry.carried_at(cl, j, slot, pos);
            krow = carry.plane_at<const bf16>(slot, pos, 2) + static_cast<long long>(h) * D;
            frow = carry.plane_at<const bf16>(slot, pos, 0) + static_cast<long long>(h) * D;
            urow = carry.plane_at<const float>(slot, pos, 3) + static_cast<long long>(h) * D;
        } else {
            qrow = mixed + at * 3 * width + static_cast<long long>(h) * D;
            krow = qrow + width;
            frow = f + at * width + static_cast<long long>(h) * D;
        }
        lds_barrier();
        const float kx = tid < D ? to_f32(krow[tid]) : 0.f;
        const float ws = wave_sum(kx * kx);
        if (l == 0) part[wave] = ws;
        if (tid < D) sa[tid] = __expf(kda_gate(to_f32(frow[tid]) + dt_bias[h * D + tid], a, floor));
        // The own row's keys land in its buffer, a head's from its first slice.
        if (!carried && blockIdx.z == 0 && tid < D) {
            if (bf16* dst = carry.own_row<bf16>(cl, e_own, 2)) dst[static_cast<long long>(h) * D + tid] = krow[tid];
        }
        lds_barrier();
        float kn = 0.f;
#pragma unroll
        for (int w = 0; w < WAVES; ++w) kn += part[w];
        if (tid < D) sk[tid] = kx * rsqrtf(kn + eps);
        lds_barrier();
        const float beta = carried ? 0.f : sigmoid(to_f32(b[at * H + h]));
        float* own_u = carried ? nullptr : carry.own_row<float>(cl, e_own, 3);
#pragma unroll
        for (int p = 0; p < R; ++p) {
            float km = 0.f;
#pragma unroll
            for (int e = 0; e < E; ++e) {
                const int at4 = 4 * c + 32 * e;
#pragma unroll
                for (int x = 0; x < 4; ++x) {
                    s[p][e][x] *= sa[at4 + x];
                    km += s[p][e][x] * sk[at4 + x];
                }
            }
            const int v = v0 + RPP * p;
            const float u = carried ? urow[v] : (to_f32(qrow[2 * width + v]) - row8_sum(km)) * beta;
#pragma unroll
            for (int e = 0; e < E; ++e) {
                const int at4 = 4 * c + 32 * e;
#pragma unroll
                for (int x = 0; x < 4; ++x) s[p][e][x] += sk[at4 + x] * u;
            }
            if (c == 0 && own_u != nullptr) own_u[static_cast<long long>(h) * D + v] = u;
        }
    }
}

// A beam group's fold of its KDA state (BEAM-PAGE-LISTS §11.3): its fold rows
// (forget rows, keys and u from the kept planes 0, 2 and 3) folded in order,
// M <- M diag(a) + u k̂^T, and the rows stored once. Grid (H,
// D / ROWS), block 256.
template <int D, int ROWS>
__global__ __launch_bounds__(256) void ssm_kda_fold(
    StateRows rows, const float* __restrict__ a_log, const float* __restrict__ dt_bias, int H, float eps, float floor,
    Carry carry, const GroupFold* __restrict__ folds, int groups) {
    constexpr int E = D / 32, RPP = 256 / kKdaRowLanes, R = ROWS / RPP, WAVES = 256 / kWave;
    __shared__ alignas(16) float sk[D], sa[D];
    __shared__ float part[WAVES];
    const int h = blockIdx.x;
    // The groups that fold lead the table; the first that folds none ends it.
    for (int g = 0; g < groups; ++g) {
        const GroupFold f = folds[g];
        const int n = static_cast<int>(f.rows);
        if (n <= 0) break;
        const int rl = static_cast<int>(f.lane);
        const i32* rid = rows.row(rl);
        if (rid[0] < 0) continue;
        const CarryLane cl = carry.lane(rl);
        const int tid = threadIdx.x, wave = tid / kWave, l = lane();
        const int c = tid % kKdaRowLanes;
        const int v0 = blockIdx.y * ROWS + tid / kKdaRowLanes;
        float* head = rows.atom<float>(rid, h, static_cast<long long>(D) * D);
        f32x4 s[R][E];
    #pragma unroll
        for (int p = 0; p < R; ++p)
    #pragma unroll
            for (int e = 0; e < E; ++e)
                s[p][e] = state_load(reinterpret_cast<const f32x4*>(head + static_cast<long long>(v0 + RPP * p) * D + 4 * c + 32 * e));
        const float a = __expf(a_log[h]);
        for (int j = 0; j < n; ++j) {
            u64 slot;
            u32 pos;
            carry.fold_at(cl, f.fold_off, j, slot, pos);
            lds_barrier();
            const float kx = tid < D ? to_f32(carry.plane_at<const bf16>(slot, pos, 2)[static_cast<long long>(h) * D + tid]) : 0.f;
            const float ws = wave_sum(kx * kx);
            if (l == 0) part[wave] = ws;
            if (tid < D)
                sa[tid] = __expf(kda_gate(to_f32(carry.plane_at<const bf16>(slot, pos, 0)[static_cast<long long>(h) * D + tid]) +
                                              dt_bias[h * D + tid], a, floor));
            lds_barrier();
            float kn = 0.f;
    #pragma unroll
            for (int w = 0; w < WAVES; ++w) kn += part[w];
            if (tid < D) sk[tid] = kx * rsqrtf(kn + eps);
            lds_barrier();
            const float* ur = carry.plane_at<const float>(slot, pos, 3) + static_cast<long long>(h) * D;
    #pragma unroll
            for (int p = 0; p < R; ++p) {
                const float u = ur[v0 + RPP * p];
    #pragma unroll
                for (int e = 0; e < E; ++e) {
                    const int at = 4 * c + 32 * e;
    #pragma unroll
                    for (int x = 0; x < 4; ++x) s[p][e][x] = s[p][e][x] * sa[at + x] + u * sk[at + x];
                }
            }
        }
    #pragma unroll
        for (int p = 0; p < R; ++p)
    #pragma unroll
            for (int e = 0; e < E; ++e)
                state_store(reinterpret_cast<f32x4*>(head + static_cast<long long>(v0 + RPP * p) * D + 4 * c + 32 * e), s[p][e]);
    }
}

// Prefill, token by token: grid (lanes, H), block 8·D, the head's state in
// registers a value row to eight lanes, the next token's operands loaded
// while this one folds. The fallback for heads the chunk passes are not
// built for, and their reference.
template <int D>
__global__ __launch_bounds__(8 * D) void ssm_kda_chunked_batched(
    const float* __restrict__ q_norm, const float* __restrict__ k_norm, const float* __restrict__ v,
    const float* __restrict__ gate, const float* __restrict__ beta, StateRows rows,
    const u32* __restrict__ indptr,
    float* __restrict__ out, int H, int D_stated, bool write_state, const int* __restrict__ commit_len,
    const u8* __restrict__ mask, const int* __restrict__ begin_at, const int* __restrict__ snap, const int* __restrict__ snap_rows,
    const i32* __restrict__ snap_slots, const u32* __restrict__ win) {
    constexpr int E = D / 32;
    static_assert(D % 32 == 0 && 8 * D <= 1024, "whole pieces, a row a lane group");
    (void)D_stated;
    const int r = blockIdx.x, h = blockIdx.y;
    if (win != nullptr && r >= static_cast<int>(win[2])) return;
    const int rl = win != nullptr ? r + static_cast<int>(win[3]) : r;
    const long long row0 = win != nullptr ? static_cast<long long>(win[1]) : 0;
    const Span sp = span_of(indptr, r, rl, commit_len, begin_at);
    if (sp.n <= 0) return;
    const i32* rid = rows.row(rl);
    if (rid[0] < 0) return;
    const int c = threadIdx.x % kKdaRowLanes, vr = threadIdx.x / kKdaRowLanes;
    float* st = rows.atom<float>(rid, h, static_cast<long long>(D) * D) + static_cast<long long>(vr) * D;

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
    Snaps sn = snaps_of(snap, rl);
    for (int t = 0; t < sp.n; ++t) {
        KdaToken<E> nx;
        float vn, bn;
        fetch(min(t + 1, sp.n - 1), nx, vn, bn);
        const float y = kda_row_step<E>(s, tk.q, tk.k, tk.a, vt, bt);
        if (c == 0) out[((sp.t0 + t + row0) * H + h) * D + vr] = y;
        if (sn.e0 < sn.e1 && snap_rows[sn.e0] == t + 1) {
            float* at = rows.atom<float>(snap_slots + static_cast<long long>(sn.e0) * rows.pieces, h,
                                         static_cast<long long>(D) * D) +
                        static_cast<long long>(vr) * D;
            if (folds(write_state, mask, rl))
#pragma unroll
                for (int e = 0; e < E; ++e) *reinterpret_cast<f32x4*>(at + 4 * c + 32 * e) = s[e];
            ++sn.e0;
        }
        tk = nx;
        vt = vn;
        bt = bn;
    }
    if (!folds(write_state, mask, rl)) return;
#pragma unroll
    for (int e = 0; e < E; ++e) *reinterpret_cast<f32x4*>(st + 4 * c + 32 * e) = s[e];
}

// ------------------------------------------- KDA prefill (chunk-parallel) --

// KDA's chunk passes (FLA's chunk_kda), Γ a vector over key channels. A
// split exp(Γ_i)·exp(−Γ_j) overflows under a steep decay, so A and P are
// products of factors referenced to the first token of each 16-row block,
// each at most one; a chunk where a channel decays past e^80 within a block
// sums its diagonal blocks per channel instead. `wu` builds W, U, P and the
// gated operands; `ho` carries S in f32 and lands the output as it goes.

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
    const float* __restrict__ a_log, const float* __restrict__ dt_bias, StateRows rows,
    const u32* __restrict__ indptr, const int* __restrict__ commit_len, const int* __restrict__ begin_at,
    bf16* __restrict__ kT, bf16* __restrict__ w_out, bf16* __restrict__ uT, bf16* __restrict__ qg,
    bf16* __restrict__ p_out, float* __restrict__ gl, int lanes, int H, float eps, float floor,
    const int* __restrict__ snap, const int* __restrict__ snap_rows, const u32* __restrict__ win) {
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

// One 16-column slice of a head's state through every chunk of a request,
// f32 in the MFMA D layout as `ssm_gdn_chunk_h` walks it, landing each
// chunk's output on the way from the S and V' held in LDS (an output pass
// would write and reread them).
template <int D>
__global__ __launch_bounds__(256) void ssm_kda_chunk_ho(
    const bf16* __restrict__ kT, const bf16* __restrict__ w, const bf16* __restrict__ uT,
    const float* __restrict__ gl, const bf16* __restrict__ qg, const bf16* __restrict__ p,
    float* __restrict__ out, StateRows rows, const u32* __restrict__ indptr,
    const int* __restrict__ commit_len, const int* __restrict__ begin_at, bool write_state,
    const u8* __restrict__ mask, int lanes, int H, const int* __restrict__ snap,
    const int* __restrict__ snap_rows, const i32* __restrict__ snap_slots, const u32* __restrict__ win) {
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
    const GdnLane ln = gdn_lane(r, indptr, rows, commit_len, begin_at, snap, win);
    if (ln.T <= 0 || ln.slot < 0) return;
    const int v0 = vt * BVT;
    const long long row0 = win != nullptr ? static_cast<long long>(win[1]) : 0;
    const int tid = threadIdx.x, wave = tid / kWave, l = lane();
    // The lane's cells (key rows k(q)..k(q)+3, value column v0 + l % 16) are
    // four adjacent floats of St.
    float* state = rows.atom<float>(rows.row(ln.rl), h, static_cast<long long>(D) * D) +
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
    const int base = gdn_chunk_base(r, indptr, begin_at, snap, win);
    const int chunks = gdn_chunks(ln, snap_rows);
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

    // A ring of DEPTH chunks in flight: a single request's walk has no second
    // block to hide behind.
    constexpr int DEPTH = 3;
    Ops ring[DEPTH];
#pragma unroll
    for (int d = 0; d < DEPTH; ++d) fetch(min(d, chunks - 1), ring[d]);
    auto step = [&](int jc, Ops& o) {
        const long long sh = static_cast<long long>(base + jc) * H + h;
        const GdnChunk ck = gdn_chunk(ln, jc, snap_rows);
        const int ta = ck.ta;
        const int L = ck.L;

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
        if (ck.snap >= 0 && folds(write_state, mask, ln.rl)) {
            float* at = rows.atom<float>(snap_slots + static_cast<long long>(ck.snap) * rows.pieces, h,
                                         static_cast<long long>(D) * D) +
                        static_cast<long long>(v0 + l % 16) * D;
#pragma unroll
            for (int q = 0; q < KT; ++q)
                *reinterpret_cast<f32x4*>(at + key(q)) = f32x4{hf[q][0], hf[q][1], hf[q][2], hf[q][3]};
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
