#include "ssm/tile_sm80.cuh"
#include "prim/state_rows.cuh"
#include "ssm/carry.cuh"

namespace pie::ssm {

// Mamba-2: a head's state `[head_dim = 64][state_dim = 128]` in f32, its
// rows read off one convolved `xbc` row (`x`, then the groups' `B`, then
// their `C`), its decay `exp(-exp(A_log) dt)` one scalar a step.
constexpr int M2_P = 64;
constexpr int M2_N = 128;

__device__ __forceinline__ float m2_dt(float raw, float bias) {
    const float x = raw + bias;
    return x > 20.f ? x : log1pf(__expf(x));
}

// One step a request: a block a (row, head), its state read and written in
// whole 16-byte runs a lane, a warp a state row at a time (lane `l` its
// columns `4l..4l+3`), eight rows a warp. Lands `(C S + D x) * silu(z)`, the
// group norm after it.
__global__ void __launch_bounds__(256) mamba2_step(
    const bf16* __restrict__ xbc,
    const bf16* __restrict__ dt,
    const bf16* __restrict__ z,
    const float* __restrict__ A_log,
    const float* __restrict__ dt_bias,
    const float* __restrict__ D,
    StateRows rows,
    bf16* __restrict__ y,
    int H, int G,
    bool write_state,
    const u8* __restrict__ write_state_mask,
    const u32* __restrict__ win)
{
    constexpr int ROWS = M2_P / 8;
    const int r = blockIdx.x;
    if (win != nullptr && r >= static_cast<int>(win[0])) return;
    const long long row = win != nullptr ? r + static_cast<long long>(win[1]) : r;
    const int h = blockIdx.y;
    const i32* rid = rows.row(r);
    if (rid[0] < 0) return;
    const int inner = H * M2_P;
    const int g = h / (H / G);
    const int warp = threadIdx.x >> 5;
    const int l = threadIdx.x & 31;
    const bf16* at = xbc + row * (inner + 2 * G * M2_N);
    float4* st = reinterpret_cast<float4*>(
        rows.atom<float>(rid, h, static_cast<long long>(M2_P) * M2_N)) + threadIdx.x;
    // Every load out before any is used: the state's eight runs, the lane's
    // columns of B and C, the warp's rows of x.
    float4 s[ROWS];
    #pragma unroll
    for (int q = 0; q < ROWS; ++q) s[q] = __ldcs(st + q * 256);
    const uint2 braw = *reinterpret_cast<const uint2*>(at + inner + g * M2_N + 4 * l);
    const uint2 craw = *reinterpret_cast<const uint2*>(at + inner + (G + g) * M2_N + 4 * l);
    const float xl = l < ROWS ? __bfloat162float(at[h * M2_P + l * 8 + warp]) : 0.f;
    const float step = m2_dt(__bfloat162float(dt[row * H + h]), dt_bias[h]);
    const float decay = __expf(-__expf(A_log[h]) * step);
    const float2 b01 = gdn_unpack(braw.x), b23 = gdn_unpack(braw.y);
    const float2 c01 = gdn_unpack(craw.x), c23 = gdn_unpack(craw.y);
    float acc[ROWS];
    #pragma unroll
    for (int q = 0; q < ROWS; ++q) {
        const float dx = step * __shfl_sync(0xffffffffu, xl, q);
        s[q].x = s[q].x * decay + dx * b01.x;
        s[q].y = s[q].y * decay + dx * b01.y;
        s[q].z = s[q].z * decay + dx * b23.x;
        s[q].w = s[q].w * decay + dx * b23.y;
        acc[q] = s[q].x * c01.x + s[q].y * c01.y + s[q].z * c23.x + s[q].w * c23.y;
    }
    if (folds(write_state, write_state_mask, r)) {
        #pragma unroll
        for (int q = 0; q < ROWS; ++q) __stcs(st + q * 256, s[q]);
    }
    #pragma unroll
    for (int q = 0; q < ROWS; ++q) {
        acc[q] = ::pie::warp_sum(acc[q]);
    }
    if (l < ROWS) {
        float mine = acc[0];
        #pragma unroll
        for (int q = 1; q < ROWS; ++q) mine = l == q ? acc[q] : mine;
        const long long o = row * inner + h * M2_P + l * 8 + warp;
        y[o] = __float2bfloat16((mine + D[h] * xl) * silu_f(__bfloat162float(z[o])));
    }
}

// `mamba2_step` over a lane's buffered rows then its own, read as a linear
// attention from the committed state `S`. With `a_i = exp(-exp(A_log) dt_i)`,
//   h_t = A_{0,t} S + sum_s A_{s,t} (dt_s x_s) B_s^T,  A_{s,t} = prod_{s<i<=t} a_i,
//   y_t = A_{0,t} C_t S + sum_s A_{s,t} dt_s (B_s . C_t) x_s + D x_t,
// so the own row reads the state once and each buffered row as one dot
// (`B_s . C_t`, a warp a row) and its `x_s`. The decays are formed in f32
// from per-row log-decays, a suffix sum, never as a ratio. A buffered row
// keeps its step size (plane 0) and its convolved `x | B` (plane 1, the
// conv lands it); a fold applies `S <- a S + (dt x) B^T` a committed row,
// the only state write. The own row's step size lands in its buffer. Lanes
// index the fire's tables (`win[3]` on).
__global__ void __launch_bounds__(256) mamba2_step_carried(
    const bf16* __restrict__ xbc,
    const bf16* __restrict__ dt,
    const bf16* __restrict__ z,
    const float* __restrict__ A_log,
    const float* __restrict__ dt_bias,
    const float* __restrict__ D,
    StateRows rows,
    bf16* __restrict__ y,
    int H, int G,
    bool write_state,
    const u8* __restrict__ write_state_mask,
    Carry carry,
    const u32* __restrict__ win)
{
    constexpr int ROWS = M2_P / 8;
    constexpr int WARPS = 8;
    // A chunk of buffered rows a pass: each row's weight to the own row
    // (before its decay), its log-decay and its `x`.
    constexpr int CH = 32;
    __shared__ float s_w[CH];
    __shared__ float s_la[CH];
    __shared__ __align__(16) float s_x[CH][M2_P];
    const int r = blockIdx.x;
    if (win != nullptr && r >= static_cast<int>(win[0])) return;
    const long long row = win != nullptr ? r + static_cast<long long>(win[1]) : r;
    const int rl = win != nullptr ? r + static_cast<int>(win[3]) : r;
    const int h = blockIdx.y;
    const CarryLane cl = carry.lane(rl);
    if (threadIdx.x == 0) {
        if (bf16* dst = carry.own_row<bf16>(cl, 0, 0)) dst[h] = dt[row * H + h];
    }
    const i32* rid = rows.row(rl);
    if (rid[0] < 0) return;
    const int inner = H * M2_P;
    const int width = inner + 2 * G * M2_N;
    const int g = h / (H / G);
    const int warp = threadIdx.x >> 5;
    const int l = threadIdx.x & 31;
    float4* st = reinterpret_cast<float4*>(
        rows.atom<float>(rid, h, static_cast<long long>(M2_P) * M2_N)) + threadIdx.x;
    float4 s[ROWS];
    #pragma unroll
    for (int q = 0; q < ROWS; ++q) s[q] = __ldcs(st + q * 256);
    const bf16* at = xbc + row * width;
    const uint2 braw = *reinterpret_cast<const uint2*>(at + inner + g * M2_N + 4 * l);
    const uint2 craw = *reinterpret_cast<const uint2*>(at + inner + (G + g) * M2_N + 4 * l);
    const float xl = l < ROWS ? __bfloat162float(at[h * M2_P + l * 8 + warp]) : 0.f;
    const float ea = __expf(A_log[h]);
    const float bias = dt_bias[h];
    const float step = m2_dt(__bfloat162float(dt[row * H + h]), bias);
    const float la_n = -ea * step;
    const float2 b01 = gdn_unpack(braw.x), b23 = gdn_unpack(braw.y);
    const float2 c01 = gdn_unpack(craw.x), c23 = gdn_unpack(craw.y);
    // The committed state's part, `C S` a state row, and the own row's
    // `B . C`.
    float acc[ROWS];
    #pragma unroll
    for (int q = 0; q < ROWS; ++q) {
        acc[q] = s[q].x * c01.x + s[q].y * c01.y + s[q].z * c23.x + s[q].w * c23.y;
    }
    const float bc_n = ::pie::warp_sum(b01.x * c01.x + b01.y * c01.y + b23.x * c23.x + b23.y * c23.y);
    const int n = carry.carried(cl);
    const int commit = carry.commit(rl, n + 1);
    const bool fold = folds(write_state, write_state_mask, rl) && commit > 0;
    // The buffered rows, a chunk at a time from the last: lane `l < 8` of
    // warp `w` sums row `p = 8 l + w` of the output. `after` is the
    // log-decay of the rows after the chunk, the own row's with them.
    float ya = 0.f;
    float after = la_n;
    const int p = l * 8 + warp;
    for (int c = (n + CH - 1) / CH - 1; c >= 0; --c) {
        const int base = CH * c;
        const int rows_c = min(CH, n - base);
        for (int e = warp; e < rows_c; e += WARPS) {
            u64 slot;
            u32 tok;
            carry.carried_at(cl, base + e, slot, tok);
            const bf16* xb = carry.plane_at<const bf16>(slot, tok, 1);
            const uint2 bj = *reinterpret_cast<const uint2*>(xb + inner + g * M2_N + 4 * l);
            const float2 j01 = gdn_unpack(bj.x), j23 = gdn_unpack(bj.y);
            const float dot = ::pie::warp_sum(j01.x * c01.x + j01.y * c01.y + j23.x * c23.x + j23.y * c23.y);
            const float2 xv = gdn_unpack(*reinterpret_cast<const unsigned*>(xb + h * M2_P + 2 * l));
            s_x[e][2 * l] = xv.x;
            s_x[e][2 * l + 1] = xv.y;
            if (l == 0) {
                const float sj = m2_dt(__bfloat162float(carry.plane_at<const bf16>(slot, tok, 0)[h]), bias);
                s_w[e] = sj * dot;
                s_la[e] = -ea * sj;
            }
        }
        __syncthreads();
        const float la = l < rows_c ? s_la[l] : 0.f;
        float later = la;
        #pragma unroll
        for (int off = 1; off < 32; off <<= 1) {
            const float down = __shfl_down_sync(0xffffffffu, later, off);
            if (l + off < 32) later += down;
        }
        const float total = __shfl_sync(0xffffffffu, later, 0);
        const float w = l < rows_c ? __expf(after + later - la) * s_w[l] : 0.f;
        after += total;
        for (int e = 0; e < rows_c; ++e) {
            const float we = __shfl_sync(0xffffffffu, w, e);
            if (l < ROWS) ya += we * s_x[e][p];
        }
        __syncthreads();
    }
    const float g0 = __expf(after);
    #pragma unroll
    for (int q = 0; q < ROWS; ++q) acc[q] = ::pie::warp_sum(acc[q]);
    if (l < ROWS) {
        float mine = acc[0];
        #pragma unroll
        for (int q = 1; q < ROWS; ++q) mine = l == q ? acc[q] : mine;
        const long long o = row * inner + h * M2_P + p;
        const float yo = g0 * mine + ya + step * bc_n * xl + D[h] * xl;
        y[o] = __float2bfloat16(yo * silu_f(__bfloat162float(z[o])));
    }
    if (!fold) return;
    // The rows the lane commits fold into the state: its buffered ones, then
    // its own where it commits that too.
    const int folded = min(commit, n);
    for (int j = 0; j < folded; ++j) {
        u64 slot;
        u32 tok;
        carry.carried_at(cl, j, slot, tok);
        const bf16* xb = carry.plane_at<const bf16>(slot, tok, 1);
        const uint2 bj = *reinterpret_cast<const uint2*>(xb + inner + g * M2_N + 4 * l);
        const float xj = l < ROWS ? __bfloat162float(xb[h * M2_P + l * 8 + warp]) : 0.f;
        const float sj = m2_dt(__bfloat162float(carry.plane_at<const bf16>(slot, tok, 0)[h]), bias);
        const float decay = __expf(-ea * sj);
        const float2 j01 = gdn_unpack(bj.x), j23 = gdn_unpack(bj.y);
        #pragma unroll
        for (int q = 0; q < ROWS; ++q) {
            const float dx = sj * __shfl_sync(0xffffffffu, xj, q);
            s[q].x = s[q].x * decay + dx * j01.x;
            s[q].y = s[q].y * decay + dx * j01.y;
            s[q].z = s[q].z * decay + dx * j23.x;
            s[q].w = s[q].w * decay + dx * j23.y;
        }
    }
    if (commit == n + 1) {
        const float decay = __expf(la_n);
        #pragma unroll
        for (int q = 0; q < ROWS; ++q) {
            const float dx = step * __shfl_sync(0xffffffffu, xl, q);
            s[q].x = s[q].x * decay + dx * b01.x;
            s[q].y = s[q].y * decay + dx * b01.y;
            s[q].z = s[q].z * decay + dx * b23.x;
            s[q].w = s[q].w * decay + dx * b23.y;
        }
    }
    #pragma unroll
    for (int q = 0; q < ROWS; ++q) __stcs(st + q * 256, s[q]);
}

// The rows a chunked Mamba-2 fire's lanes land past their commit, made
// ready for a later decode's linear attention: each row's convolved
// `x | B` (the first `xb` channels of its `xbc` row) into plane 1 of the
// lane's buffer (its step size, plane 0, the fire lands). A block a lane;
// the rows read where the fire lays them: the lane's extended rows (`ext`:
// row `RsLane::ext + j`) or its own (`qo_indptr`, the window's row offset
// `win[1]`).
__global__ void mamba2_tail_rows(
    const bf16* __restrict__ xbc,
    const u32* __restrict__ qo_indptr,
    bool ext,
    int width, int xb,
    bool write_state,
    const u8* __restrict__ write_state_mask,
    int lanes_stated,
    Carry carry,
    const u32* __restrict__ win)
{
    const int lanes = win != nullptr ? static_cast<int>(win[2]) : lanes_stated;
    const int r = blockIdx.x;
    if (r >= lanes) return;
    const int rl = win != nullptr ? r + static_cast<int>(win[3]) : r;
    const int row0 = win != nullptr ? static_cast<int>(win[1]) : 0;
    const CarryLane cl = carry.lane(rl);
    if (cl.at == CARRY_NONE) return;
    const int n = carry.carried(cl);
    const int own = static_cast<int>(qo_indptr[r + 1] - qo_indptr[r]);
    const int run = n + own;
    const int start = folds(write_state, write_state_mask, rl) ? carry.commit(rl, run) : 0;
    for (int e = max(start - n, 0); e < own; ++e) {
        const long long at = ext ? static_cast<long long>(cl.ext) + n + e
                                 : static_cast<long long>(row0) + qo_indptr[r] + e;
        const uint4* src = reinterpret_cast<const uint4*>(xbc + at * width);
        uint4* dst = reinterpret_cast<uint4*>(carry.own_row<bf16>(cl, e, 1));
        for (int i = threadIdx.x; i < xb / 8; i += blockDim.x) dst[i] = src[i];
    }
}

// A prompt's walk in sixteen-token chunks, as `kda_chunk_scan` walks KDA's:
// a block a (request, head), a warp 32 state columns in f32 fragments, the
// state in shared memory in bf16 for each chunk's products. Its decay is one
// scalar a step, so a chunk's causal scores `P = tril(C_i . B_j exp(G_i -
// G_j))` form in place, and with no delta rule each chunk lands
// `exp(G) (C S) + P (dt x) + D x`, gated, then folds
// `S = exp(G_L) S + B^T (exp(G_L - G) dt x)`.
constexpr int M2_LDV = M2_P + 8;
constexpr int M2_STAGE = 2 * KDA_CH * GDN_LD * 2 + KDA_CH * M2_LDV * 2 + KDA_CH * 4;
constexpr int M2_SCAN_SMEM = 2 * M2_STAGE + M2_N * M2_LDV * 2 + 2 * KDA_CH * M2_LDV * 2 +
                             KDA_CH * KDA16_LDT * 2 + 2 * KDA_CH * 4;

__global__ void __launch_bounds__(128) mamba2_scan(
    const bf16* __restrict__ xbc,
    const bf16* __restrict__ dt,
    const bf16* __restrict__ z,
    const float* __restrict__ A_log,
    const float* __restrict__ dt_bias,
    const float* __restrict__ D,
    StateRows rows,
    const u32* __restrict__ qo_indptr,
    const int* __restrict__ commit_len,
    const int* __restrict__ begin_at,
    bool write_state,
    const u8* __restrict__ write_state_mask,
    bf16* __restrict__ y,
    int lanes, int H, int G,
    const u32* __restrict__ win)
{
    constexpr int CH = KDA_CH;
    constexpr int N = M2_N;
    constexpr int P = M2_P;
    constexpr int LDV = M2_LDV;
    extern __shared__ __align__(16) unsigned char m2_smem[];
    bf16* hs = reinterpret_cast<bf16*>(m2_smem + 2 * M2_STAGE);
    bf16* su = hs + N * LDV;
    bf16* su2 = su + CH * LDV;
    bf16* sp = su2 + CH * LDV;
    float* sG = reinterpret_cast<float*>(sp + CH * KDA16_LDT);
    float* sdt = sG + CH;
    auto stage = [&](int b, bf16*& sc, bf16*& sb, bf16*& sx, float*& sraw) {
        unsigned char* at = m2_smem + b * M2_STAGE;
        sc = reinterpret_cast<bf16*>(at);
        sb = sc + CH * GDN_LD;
        sx = sb + CH * GDN_LD;
        sraw = reinterpret_cast<float*>(sx + CH * LDV);
    };

    const int h = static_cast<int>(blockIdx.x) % H;
    const int r = static_cast<int>(blockIdx.x) / H;
    if (win != nullptr) lanes = min(lanes, static_cast<int>(win[2]));
    if (r >= lanes) return;
    const GdnLane lane = gdn_lane(r, qo_indptr, rows, commit_len, begin_at, win);
    if (lane.T <= 0 || lane.slot < 0) return;
    const int row0 = win != nullptr ? static_cast<int>(win[1]) : 0;
    const int inner = H * P;
    const long long width = inner + 2 * G * N;
    const int grp = h / (H / G);
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int l = tid & 31;
    const int g = l >> 2;
    const int t = l & 3;
    const int c0 = warp * (P / 4);
    const int kr0 = warp * 32;
    const float neg_a = -__expf(A_log[h]);
    const float bias = dt_bias[h];
    const float skip = D[h];
    float* state = rows.atom<float>(rows.row(lane.rl), h, static_cast<long long>(N) * P);

    // The state fragments: two sixteen-row tiles of state columns by the
    // eight eight-wide tiles of the head's dims.
    float hacc[2][P / 8][4];
    #pragma unroll
    for (int mt = 0; mt < 2; ++mt) {
        #pragma unroll
        for (int nt = 0; nt < P / 8; ++nt) {
            #pragma unroll
            for (int e = 0; e < 4; ++e) {
                const int kk = kr0 + mt * 16 + g + 8 * (e >> 1);
                const int vv = nt * 8 + 2 * t + (e & 1);
                hacc[mt][nt][e] = state[static_cast<long long>(vv) * N + kk];
            }
        }
    }
    auto put_state = [&]() {
        #pragma unroll
        for (int mt = 0; mt < 2; ++mt) {
            #pragma unroll
            for (int nt = 0; nt < P / 8; ++nt) {
                #pragma unroll
                for (int half = 0; half < 2; ++half) {
                    *reinterpret_cast<unsigned*>(hs + (kr0 + mt * 16 + g + 8 * half) * LDV + nt * 8 + 2 * t) =
                        pack_bf16x2(hacc[mt][nt][2 * half], hacc[mt][nt][2 * half + 1]);
                }
            }
        }
    };

    const int nchunks = (lane.T + CH - 1) / CH;
    auto load = [&](int jc) {
        bf16 *sc, *sb, *sx;
        float* sraw;
        stage(jc & 1, sc, sb, sx, sraw);
        const int ta = lane.t0 + jc * CH;
        const int L = min(CH, lane.t0 + lane.T - ta);
        auto row = [&](int i) { return static_cast<long long>(ta + (i < L ? i : 0) + row0) * width; };
        #pragma unroll
        for (int e = tid; e < CH * (N / 8); e += 128) {
            const int i = e / (N / 8);
            const int c = (e % (N / 8)) * 8;
            gdn_cp16(sb + i * GDN_LD + c, xbc + row(i) + inner + grp * N + c, i < L);
            gdn_cp16(sc + i * GDN_LD + c, xbc + row(i) + inner + (G + grp) * N + c, i < L);
        }
        {
            const int i = tid / (P / 8);
            const int c = (tid % (P / 8)) * 8;
            gdn_cp16(sx + i * LDV + c, xbc + row(i) + h * P + c, i < L);
        }
        if (tid < CH) {
            sraw[tid] = tid < L ? __bfloat162float(dt[(static_cast<long long>(ta + tid) + row0) * H + h]) : 0.f;
        }
        gdn_cp_commit();
    };

    load(0);
    put_state();
    for (int jc = 0; jc < nchunks; ++jc) {
        if (jc + 1 < nchunks) {
            load(jc + 1);
            asm volatile("cp.async.wait_group 1;\n" ::: "memory");
        } else {
            gdn_cp_wait_all();
        }
        __syncthreads();
        bf16 *sc, *sb, *sx;
        float* sraw;
        stage(jc & 1, sc, sb, sx, sraw);
        const int ta = lane.t0 + jc * CH;
        const int L = min(CH, lane.t0 + lane.T - ta);

        // The steps and the decay's running sum down the chunk, a row past
        // it neither.
        if (warp == 0) {
            float step = 0.f;
            float a = 0.f;
            if (l < L) {
                step = m2_dt(sraw[l], bias);
                a = neg_a * step;
            }
            #pragma unroll
            for (int off = 1; off < CH; off <<= 1) {
                const float v = __shfl_up_sync(0xffffffffu, a, off);
                if (l >= off) a += v;
            }
            if (l < CH) {
                sG[l] = a;
                sdt[l] = step;
            }
        }
        __syncthreads();
        const float gl = sG[CH - 1];

        // `dt x` for the scores and, decayed to the chunk's end, for the
        // fold; the first two warps form `P` from `C B^T`.
        {
            const int i = tid / (P / 8);
            const int c = (tid % (P / 8)) * 8;
            float x[8], u[8];
            #pragma unroll
            for (int e = 0; e < 8; ++e) x[e] = __bfloat162float(sx[i * LDV + c + e]);
            const float s = sdt[i];
            const float s2 = s * __expf(gl - sG[i]);
            #pragma unroll
            for (int e = 0; e < 8; ++e) u[e] = s * x[e];
            gdn_store8(su + i * LDV + c, u);
            #pragma unroll
            for (int e = 0; e < 8; ++e) u[e] = s2 * x[e];
            gdn_store8(su2 + i * LDV + c, u);
        }
        if (warp < 2) {
            float acc[4] = {};
            #pragma unroll
            for (int kk = 0; kk < N / 16; ++kk) {
                unsigned a[4], b[2];
                ldmatrix_x4(a, gdn_at_a(sc, GDN_LD, 0, kk * 16, l));
                ld_b(b, sb + warp * 8 * GDN_LD + kk * 16, GDN_LD, l);
                mma_m16n8k16(acc, a, b);
            }
            #pragma unroll
            for (int half = 0; half < 2; ++half) {
                const int i = g + 8 * half;
                const int j = warp * 8 + 2 * t;
                const bool live = i < L;
                const float p0 = live && j <= i ? acc[2 * half] * __expf(sG[i] - sG[j]) : 0.f;
                const float p1 = live && j + 1 <= i ? acc[2 * half + 1] * __expf(sG[i] - sG[j + 1]) : 0.f;
                *reinterpret_cast<unsigned*>(sp + i * KDA16_LDT + j) = pack_bf16x2(p0, p1);
            }
        }
        __syncthreads();

        // `exp(G) (C S) + P (dt x)` over the warp's dims, then `+ D x`, gated.
        {
            float qh[2][4] = {};
            #pragma unroll
            for (int kk = 0; kk < N / 16; ++kk) {
                unsigned aq[4], b[4];
                ldmatrix_x4(aq, gdn_at_a(sc, GDN_LD, 0, kk * 16, l));
                ldmatrix_x4_trans(b, gdn_at_b(hs, LDV, kk * 16, c0, l));
                mma_m16n8k16(qh[0], aq, {b[0], b[1]});
                mma_m16n8k16(qh[1], aq, {b[2], b[3]});
            }
            #pragma unroll
            for (int half = 0; half < 2; ++half) {
                const float e = __expf(sG[g + 8 * half]);
                #pragma unroll
                for (int nt = 0; nt < 2; ++nt) {
                    qh[nt][2 * half] *= e;
                    qh[nt][2 * half + 1] *= e;
                }
            }
            unsigned pa[4], b[4];
            ldmatrix_x4(pa, gdn_at_a(sp, KDA16_LDT, 0, 0, l));
            ldmatrix_x4_trans(b, gdn_at_b(su, LDV, 0, c0, l));
            mma_m16n8k16(qh[0], pa, {b[0], b[1]});
            mma_m16n8k16(qh[1], pa, {b[2], b[3]});
            #pragma unroll
            for (int half = 0; half < 2; ++half) {
                const int i = g + 8 * half;
                if (i >= L) continue;
                const long long o = static_cast<long long>(ta + i + row0) * inner + h * P;
                #pragma unroll
                for (int nt = 0; nt < 2; ++nt) {
                    const int c = c0 + nt * 8 + 2 * t;
                    const float2 xv = gdn_unpack(*reinterpret_cast<const unsigned*>(sx + i * LDV + c));
                    const float2 zv = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(z + o + c));
                    *reinterpret_cast<unsigned*>(y + o + c) =
                        pack_bf16x2((qh[nt][2 * half] + skip * xv.x) * silu_f(zv.x),
                                 (qh[nt][2 * half + 1] + skip * xv.y) * silu_f(zv.y));
                }
            }
        }

        // The fold: `S = exp(G_L) S + B^T (exp(G_L - G) dt x)`.
        const float d = __expf(gl);
        #pragma unroll
        for (int mt = 0; mt < 2; ++mt) {
            #pragma unroll
            for (int nt = 0; nt < P / 8; ++nt) {
                #pragma unroll
                for (int e = 0; e < 4; ++e) hacc[mt][nt][e] *= d;
            }
        }
        {
            unsigned a[2][4];
            #pragma unroll
            for (int mt = 0; mt < 2; ++mt) ldmatrix_x4_trans(a[mt], gdn_at_at(sb, GDN_LD, 0, kr0 + mt * 16, l));
            #pragma unroll
            for (int np = 0; np < P / 16; ++np) {
                unsigned b[4];
                ldmatrix_x4_trans(b, gdn_at_b(su2, LDV, 0, np * 16, l));
                #pragma unroll
                for (int mt = 0; mt < 2; ++mt) {
                    mma_m16n8k16(hacc[mt][2 * np], a[mt], {b[0], b[1]});
                    mma_m16n8k16(hacc[mt][2 * np + 1], a[mt], {b[2], b[3]});
                }
            }
        }
        __syncthreads();
        put_state();
    }
    gdn_cp_wait_all();

    if (write_state && row_persists(write_state_mask, lane.rl)) {
        #pragma unroll
        for (int mt = 0; mt < 2; ++mt) {
            #pragma unroll
            for (int nt = 0; nt < P / 8; ++nt) {
                #pragma unroll
                for (int e = 0; e < 4; ++e) {
                    const int kk = kr0 + mt * 16 + g + 8 * (e >> 1);
                    const int vv = nt * 8 + 2 * t + (e & 1);
                    state[static_cast<long long>(vv) * N + kk] = hacc[mt][nt][e];
                }
            }
        }
    }
}

// A beam group's fold of its Mamba-2 state: its fold
// rows (step sizes and `x | B`: planes 0 and 1, through the representative
// lane's pages) folded in order, `S <- a S + (dt x) B^T`, `a = exp(-exp(A_log)
// dt)`, as a decode step folds a committed row, and the state stored once. A
// block a head, folding the groups in turn (the folding ones lead the table;
// the first of no rows ends it), `mamba2_step_carried`'s layout.
__global__ void __launch_bounds__(256) mamba2_fold(
    StateRows rows,
    const float* __restrict__ A_log,
    const float* __restrict__ dt_bias,
    int H, int G,
    Carry carry,
    const GroupFold* __restrict__ folds,
    int groups)
{
    constexpr int ROWS = M2_P / 8;
    const int h = blockIdx.x;
    const int inner = H * M2_P;
    const int gi = h / (H / G);
    const int warp = threadIdx.x >> 5;
    const int l = threadIdx.x & 31;
    const float ea = __expf(A_log[h]);
    const float bias = dt_bias[h];
    for (int g = 0; g < groups; ++g) {
        const GroupFold f = folds[g];
        const int n = static_cast<int>(f.rows);
        if (n <= 0) break;
        const int rl = static_cast<int>(f.lane);
        const i32* rid = rows.row(rl);
        if (rid[0] < 0) continue;
        const CarryLane cl = carry.lane(rl);
        float4* st = reinterpret_cast<float4*>(
            rows.atom<float>(rid, h, static_cast<long long>(M2_P) * M2_N)) + threadIdx.x;
        float4 s[ROWS];
        #pragma unroll
        for (int q = 0; q < ROWS; ++q) s[q] = __ldcs(st + q * 256);
        for (int j = 0; j < n; ++j) {
            u64 slot;
            u32 tok;
            carry.fold_at(cl, f.fold_off, j, slot, tok);
            const bf16* xb = carry.plane_at<const bf16>(slot, tok, 1);
            const uint2 bj = *reinterpret_cast<const uint2*>(xb + inner + gi * M2_N + 4 * l);
            const float xj = l < ROWS ? __bfloat162float(xb[h * M2_P + l * 8 + warp]) : 0.f;
            const float sj = m2_dt(__bfloat162float(carry.plane_at<const bf16>(slot, tok, 0)[h]), bias);
            const float decay = __expf(-ea * sj);
            const float2 j01 = gdn_unpack(bj.x), j23 = gdn_unpack(bj.y);
            #pragma unroll
            for (int q = 0; q < ROWS; ++q) {
                const float dx = sj * __shfl_sync(0xffffffffu, xj, q);
                s[q].x = s[q].x * decay + dx * j01.x;
                s[q].y = s[q].y * decay + dx * j01.y;
                s[q].z = s[q].z * decay + dx * j23.x;
                s[q].w = s[q].w * decay + dx * j23.y;
            }
        }
        #pragma unroll
        for (int q = 0; q < ROWS; ++q) __stcs(st + q * 256, s[q]);
    }
}

}
