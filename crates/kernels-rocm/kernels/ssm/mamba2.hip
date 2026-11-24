// Mamba-2: the step, the carried step, the tail rows, the fold and the scan.

#include "ssm/common.hip"

namespace pie::attn {

// ------------------------------------------------------------- Mamba-2 --

// Mamba-2: a head's state S[p][n] (64 head dims by a 128-wide state) in f32
// at the row's head h (`StateRows::atom`), its operands off one convolved `xbc` row
// (`x`, then each group's `B`, then its `C`), its decay exp(−exp(A_log)·dt)
// one scalar a step. A token stores S = S·decay + dt·x·Bᵀ and lands
// (S·C + D·x)·silu(z), the group norm after it. A block of four wavefronts
// holds the state in registers: a half-wavefront a state row, lane c its
// columns 4c..4c+3, eight rows a lane.
constexpr int M2_P = 64, M2_N = 128, M2_Q = M2_P / 8;

__device__ __forceinline__ float m2_dt(float raw, float bias) {
    const float x = raw + bias;
    return x > 20.f ? x : log1pf(__expf(x));
}

// The state row of the lane's q-th run, and the run's index among the head's.
__device__ __forceinline__ int m2_row(int q) {
    return static_cast<int>(threadIdx.x / kWave) * (2 * M2_Q) + 2 * q + (lane() >> 5);
}
__device__ __forceinline__ int m2_run(int q) { return m2_row(q) * (M2_N / 4) + (lane() & 31); }

// One token through the lane's runs: `at` its xbc row, `zr` and `yr` its z
// and output rows.
__device__ __forceinline__ void m2_token(f32x4 (&s)[M2_Q], const bf16* at, const bf16* zr, bf16* yr, float step,
                                         float decay, float skip, int h, int inner, int g, int G) {
    const int c = 4 * (lane() & 31);
    const bf16* bp = at + inner + g * M2_N + c;
    const bf16* cp = at + inner + (G + g) * M2_N + c;
    const f32x4 b = {to_f32(bp[0]), to_f32(bp[1]), to_f32(bp[2]), to_f32(bp[3])};
    const f32x4 cc = {to_f32(cp[0]), to_f32(cp[1]), to_f32(cp[2]), to_f32(cp[3])};
    float x[M2_Q];
#pragma unroll
    for (int q = 0; q < M2_Q; ++q) x[q] = to_f32(at[h * M2_P + m2_row(q)]);
#pragma unroll
    for (int q = 0; q < M2_Q; ++q) {
        s[q] = s[q] * decay + (step * x[q]) * b;
        float acc = s[q].x * cc.x + s[q].y * cc.y + s[q].z * cc.z + s[q].w * cc.w;
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) acc += __shfl_xor(acc, off, kWave);
        if ((lane() & 31) == 0) {
            const int o = h * M2_P + m2_row(q);
            yr[o] = from_f32<bf16>((acc + skip * x[q]) * silu(to_f32(zr[o])));
        }
    }
}

// Decode: grid (rows, H), a block a (row, head).
__global__ __launch_bounds__(256) void ssm_mamba2_step(
    const bf16* __restrict__ xbc, const bf16* __restrict__ dt, const bf16* __restrict__ z,
    const float* __restrict__ a_log, const float* __restrict__ dt_bias, const float* __restrict__ D,
    StateRows rows, bf16* __restrict__ y, int H, int G, bool write_state, const u8* __restrict__ mask, const u32* __restrict__ win) {
    const int r = blockIdx.x, h = blockIdx.y;
    if (win != nullptr && r >= static_cast<int>(win[0])) return;
    const long long row = win != nullptr ? r + static_cast<long long>(win[1]) : r;
    const int rl = win != nullptr ? r + static_cast<int>(win[3]) : r;
    const i32* rid = rows.row(rl);
    if (rid[0] < 0) return;
    const int inner = H * M2_P;
    f32x4* st = reinterpret_cast<f32x4*>(rows.atom<float>(rid, h, static_cast<long long>(M2_P) * M2_N));
    f32x4 s[M2_Q];
#pragma unroll
    for (int q = 0; q < M2_Q; ++q) s[q] = st[m2_run(q)];
    const float step = m2_dt(to_f32(dt[row * H + h]), dt_bias[h]);
    m2_token(s, xbc + row * (inner + 2 * G * M2_N), z + row * inner, y + row * inner, step,
             __expf(-__expf(a_log[h]) * step), D[h], h, inner, h / (H / G), G);
    if (!folds(write_state, mask, rl)) return;
#pragma unroll
    for (int q = 0; q < M2_Q; ++q) st[m2_run(q)] = s[q];
}

// `ssm_mamba2_step` over a lane's buffered rows then its own, the buffered
// rows read as a linear attention over the committed state (a row keeps its
// step size and its convolved x | B: planes 0 and 1):
//   y_t = A_{0,t} C_t S + sum_s A_{s,t} dt_s (B_s . C_t) x_s + D x_t,
// A_{s,t} = prod_{s<i<=t} a_i, so a buffered row is one dot B_s . C_t (every
// half-wave forms it over its 128 columns, no barrier) and its x_s, weighed
// by the decays after it as the rows are walked from the last. A fold
// applies S <- a S + (dt x) B^T a committed row, then the own row's where it
// commits that too; the state is stored once. The own row's step size lands
// in its buffer (its x | B lands with the conv's output).
__global__ __launch_bounds__(256) void ssm_mamba2_step_carried(
    const bf16* __restrict__ xbc, const bf16* __restrict__ dt, const bf16* __restrict__ z,
    const float* __restrict__ a_log, const float* __restrict__ dt_bias, const float* __restrict__ D,
    StateRows rows, bf16* __restrict__ y, int H, int G, bool write_state, const u8* __restrict__ mask, Carry carry,
    const u32* __restrict__ win) {
    const int r = blockIdx.x, h = blockIdx.y;
    if (win != nullptr && r >= static_cast<int>(win[0])) return;
    const long long row = win != nullptr ? r + static_cast<long long>(win[1]) : r;
    const int rl = win != nullptr ? r + static_cast<int>(win[3]) : r;
    const CarryLane cl = carry.lane(rl);
    if (threadIdx.x == 0) {
        if (bf16* dst = carry.own_row<bf16>(cl, 0, 0)) dst[h] = dt[row * H + h];
    }
    const i32* rid = rows.row(rl);
    if (rid[0] < 0) return;
    const int inner = H * M2_P;
    const int width = inner + 2 * G * M2_N;
    const int g = h / (H / G);
    f32x4* st = reinterpret_cast<f32x4*>(rows.atom<float>(rid, h, static_cast<long long>(M2_P) * M2_N));
    f32x4 s[M2_Q];
#pragma unroll
    for (int q = 0; q < M2_Q; ++q) s[q] = st[m2_run(q)];
    const int cc = 4 * (lane() & 31);
    auto half = [](float v) {
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) v += __shfl_xor(v, off, kWave);
        return v;
    };
    auto four = [](const bf16* p) { return f32x4{to_f32(p[0]), to_f32(p[1]), to_f32(p[2]), to_f32(p[3])}; };
    const bf16* at = xbc + row * width;
    const f32x4 bo = four(at + inner + g * M2_N + cc), co = four(at + inner + (G + g) * M2_N + cc);
    float xo[M2_Q];
#pragma unroll
    for (int q = 0; q < M2_Q; ++q) xo[q] = to_f32(at[h * M2_P + m2_row(q)]);
    const float a = __expf(a_log[h]), bias = dt_bias[h];
    const float step = m2_dt(to_f32(dt[row * H + h]), bias);
    const float la_n = -a * step;
    const float bc_n = half(bo[0] * co[0] + bo[1] * co[1] + bo[2] * co[2] + bo[3] * co[3]);
    const int n = carry.carried(cl);
    float ya[M2_Q];
#pragma unroll
    for (int q = 0; q < M2_Q; ++q) ya[q] = 0.f;
    float after = la_n;
    for (int j = n - 1; j >= 0; --j) {
        u64 slot;
        u32 pos;
        carry.carried_at(cl, j, slot, pos);
        const bf16* xb = carry.plane_at<const bf16>(slot, pos, 1);
        const f32x4 bj = four(xb + inner + g * M2_N + cc);
        const float dot = half(bj[0] * co[0] + bj[1] * co[1] + bj[2] * co[2] + bj[3] * co[3]);
        const float sj = m2_dt(to_f32(carry.plane_at<const bf16>(slot, pos, 0)[h]), bias);
        const float w = __expf(after) * sj * dot;
#pragma unroll
        for (int q = 0; q < M2_Q; ++q) ya[q] += w * to_f32(xb[h * M2_P + m2_row(q)]);
        after += -a * sj;
    }
    const float g0 = __expf(after);
#pragma unroll
    for (int q = 0; q < M2_Q; ++q) {
        const float acc = half(s[q][0] * co[0] + s[q][1] * co[1] + s[q][2] * co[2] + s[q][3] * co[3]);
        if ((lane() & 31) == 0) {
            const long long o = row * inner + h * M2_P + m2_row(q);
            const float yo = g0 * acc + ya[q] + step * bc_n * xo[q] + D[h] * xo[q];
            y[o] = from_f32<bf16>(yo * silu(to_f32(z[o])));
        }
    }
    const int commit = carry.commit(rl, n + 1);
    if (!folds(write_state, mask, rl) || commit <= 0) return;
    const int folded = min(commit, n);
    for (int j = 0; j < folded; ++j) {
        u64 slot;
        u32 pos;
        carry.carried_at(cl, j, slot, pos);
        const bf16* xb = carry.plane_at<const bf16>(slot, pos, 1);
        const f32x4 bj = four(xb + inner + g * M2_N + cc);
        const float sj = m2_dt(to_f32(carry.plane_at<const bf16>(slot, pos, 0)[h]), bias);
        const float decay = __expf(-a * sj);
#pragma unroll
        for (int q = 0; q < M2_Q; ++q) s[q] = s[q] * decay + (sj * to_f32(xb[h * M2_P + m2_row(q)])) * bj;
    }
    if (commit == n + 1) {
        const float decay = __expf(la_n);
#pragma unroll
        for (int q = 0; q < M2_Q; ++q) s[q] = s[q] * decay + (step * xo[q]) * bo;
    }
#pragma unroll
    for (int q = 0; q < M2_Q; ++q) st[m2_run(q)] = s[q];
}

// The rows a chunked Mamba-2 fire's lanes land past their commit, made
// ready for a later decode's linear attention: each row's convolved x | B
// (the first `xb` channels of its xbc row) into plane 1 of the lane's
// buffer (its step size, plane 0, the fire lands). A block a lane; the rows
// read where the fire lays them: the lane's extended rows (`ext`: row
// `RsLane::ext + j`) or its own (`qo_indptr`, the window's row offset
// `win[1]`).
__global__ void ssm_mamba2_tail_rows(const bf16* __restrict__ xbc, const u32* __restrict__ qo_indptr, bool ext,
                                     int width, int xb, bool write_state, const u8* __restrict__ mask,
                                     int lanes_stated, Carry carry, const u32* __restrict__ win) {
    const int lanes = win != nullptr ? static_cast<int>(win[2]) : lanes_stated;
    const int r = blockIdx.x;
    if (r >= lanes) return;
    const int rl = win != nullptr ? r + static_cast<int>(win[3]) : r;
    const long long row0 = win != nullptr ? static_cast<long long>(win[1]) : 0;
    const CarryLane cl = carry.lane(rl);
    if (cl.at == CARRY_NONE) return;
    const int n = carry.carried(cl);
    const int own = static_cast<int>(qo_indptr[r + 1] - qo_indptr[r]);
    const int run = n + own;
    const int start = folds(write_state, mask, rl) ? carry.commit(rl, run) : 0;
    for (int e = max(start - n, 0); e < own; ++e) {
        const long long at = ext ? static_cast<long long>(cl.ext) + n + e : row0 + qo_indptr[r] + e;
        const bf16* src = xbc + at * width;
        bf16* dst = carry.own_row<bf16>(cl, e, 1);
        if (dst == nullptr) continue;
        for (int i = threadIdx.x; i < xb; i += blockDim.x) dst[i] = src[i];
    }
}

// A beam group's fold of its Mamba-2 state (BEAM-PAGE-LISTS §11.3): its fold
// rows (step sizes and convolved x | B from the kept planes 0 and 1) folded in
// order, S <- a S + (dt x) B^T, and the state stored once. Grid H, a block folding each group in turn,
// block 256.
__global__ __launch_bounds__(256) void ssm_mamba2_fold(StateRows rows, const float* __restrict__ a_log,
                                                       const float* __restrict__ dt_bias, int H, int G, Carry carry,
                                                       const GroupFold* __restrict__ folds, int groups) {
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
        const int inner = H * M2_P;
        const int gg = h / (H / G);
        f32x4* st = reinterpret_cast<f32x4*>(rows.atom<float>(rid, h, static_cast<long long>(M2_P) * M2_N));
        f32x4 s[M2_Q];
    #pragma unroll
        for (int q = 0; q < M2_Q; ++q) s[q] = st[m2_run(q)];
        const float a = __expf(a_log[h]), bias = dt_bias[h];
        const int cc = 4 * (lane() & 31);
        for (int j = 0; j < n; ++j) {
            u64 slot;
            u32 pos;
            carry.fold_at(cl, f.fold_off, j, slot, pos);
            const bf16* xb = carry.plane_at<const bf16>(slot, pos, 1);
            const bf16* bp = xb + inner + gg * M2_N + cc;
            const f32x4 bj = {to_f32(bp[0]), to_f32(bp[1]), to_f32(bp[2]), to_f32(bp[3])};
            const float sj = m2_dt(to_f32(carry.plane_at<const bf16>(slot, pos, 0)[h]), bias);
            const float decay = __expf(-a * sj);
    #pragma unroll
            for (int q = 0; q < M2_Q; ++q) s[q] = s[q] * decay + (sj * to_f32(xb[h * M2_P + m2_row(q)])) * bj;
        }
    #pragma unroll
        for (int q = 0; q < M2_Q; ++q) st[m2_run(q)] = s[q];
    }
}

// Prefill, token by token: grid (lanes, H), the head's state in registers
// across the walk.
__global__ __launch_bounds__(256) void ssm_mamba2_scan(
    const bf16* __restrict__ xbc, const bf16* __restrict__ dt, const bf16* __restrict__ z,
    const float* __restrict__ a_log, const float* __restrict__ dt_bias, const float* __restrict__ D,
    StateRows rows, const u32* __restrict__ indptr, const int* __restrict__ commit_len, const int* __restrict__ begin_at, bool write_state,
    const u8* __restrict__ mask, const int* __restrict__ snap, const int* __restrict__ snap_rows,
    const i32* __restrict__ snap_slots, bf16* __restrict__ y, int H, int G, const u32* __restrict__ win) {
    const int r = blockIdx.x, h = blockIdx.y;
    if (win != nullptr && r >= static_cast<int>(win[2])) return;
    const int rl = win != nullptr ? r + static_cast<int>(win[3]) : r;
    const long long row0 = win != nullptr ? static_cast<long long>(win[1]) : 0;
    const Span sp = span_of(indptr, r, rl, commit_len, begin_at);
    if (sp.n <= 0) return;
    const i32* rid = rows.row(rl);
    if (rid[0] < 0) return;
    const long long head = static_cast<long long>(M2_P) * M2_N;
    f32x4* st = reinterpret_cast<f32x4*>(rows.atom<float>(rid, h, head));
    f32x4 s[M2_Q];
#pragma unroll
    for (int q = 0; q < M2_Q; ++q) s[q] = st[m2_run(q)];
    const int inner = H * M2_P, g = h / (H / G);
    const long long width = inner + 2 * G * M2_N;
    const float neg_a = -__expf(a_log[h]), bias = dt_bias[h], skip = D[h];
    const bool fold = folds(write_state, mask, rl);
    Snaps sn = snaps_of(snap, rl);
    for (int t = 0; t < sp.n; ++t) {
        const long long row = sp.t0 + t + row0;
        const float step = m2_dt(to_f32(dt[row * H + h]), bias);
        m2_token(s, xbc + row * width, z + row * inner, y + row * inner, step, __expf(neg_a * step), skip, h, inner,
                 g, G);
        if (sn.e0 < sn.e1 && snap_rows[sn.e0] == t + 1) {
            f32x4* at = reinterpret_cast<f32x4*>(
                rows.atom<float>(snap_slots + static_cast<long long>(sn.e0) * rows.pieces, h, head));
            if (fold)
#pragma unroll
                for (int q = 0; q < M2_Q; ++q) at[m2_run(q)] = s[q];
            ++sn.e0;
        }
    }
    if (!fold) return;
#pragma unroll
    for (int q = 0; q < M2_Q; ++q) st[m2_run(q)] = s[q];
}

}  // namespace pie::attn
