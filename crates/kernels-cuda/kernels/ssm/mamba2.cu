#include "ssm/tile_sm80.cuh"

namespace pie::attn {

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
__global__ void __launch_bounds__(256) ssm_mamba2_step(
    const bf16* __restrict__ xbc,
    const bf16* __restrict__ dt,
    const bf16* __restrict__ z,
    const float* __restrict__ A_log,
    const float* __restrict__ dt_bias,
    const float* __restrict__ D,
    float* __restrict__ state_base,
    const i32* __restrict__ slot_ids,
    long long slot_stride_elems,
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
    const int slot = slot_ids[r];
    if (slot < 0) return;
    const int inner = H * M2_P;
    const int g = h / (H / G);
    const int warp = threadIdx.x >> 5;
    const int l = threadIdx.x & 31;
    const bf16* at = xbc + row * (inner + 2 * G * M2_N);
    float4* st = reinterpret_cast<float4*>(state_base + static_cast<long long>(slot) * slot_stride_elems +
                                           static_cast<long long>(h) * M2_P * M2_N) + threadIdx.x;
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
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) acc[q] += __shfl_xor_sync(0xffffffffu, acc[q], off);
    }
    if (l < ROWS) {
        float mine = acc[0];
        #pragma unroll
        for (int q = 1; q < ROWS; ++q) mine = l == q ? acc[q] : mine;
        const long long o = row * inner + h * M2_P + l * 8 + warp;
        y[o] = __float2bfloat16((mine + D[h] * xl) * silu_f(__bfloat162float(z[o])));
    }
}

// A prompt's walk in sixteen-token chunks, as `ssm_kda16_scan` walks KDA's:
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

__global__ void __launch_bounds__(128) ssm_mamba2_scan(
    const bf16* __restrict__ xbc,
    const bf16* __restrict__ dt,
    const bf16* __restrict__ z,
    const float* __restrict__ A_log,
    const float* __restrict__ dt_bias,
    const float* __restrict__ D,
    float* __restrict__ state_base,
    const int* __restrict__ slot_ids,
    const u32* __restrict__ qo_indptr,
    long long slot_stride_elems,
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
    const GdnLane lane = gdn_lane(r, qo_indptr, slot_ids, commit_len, begin_at, win);
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
    float* state = state_base + static_cast<long long>(lane.slot) * slot_stride_elems +
                   static_cast<long long>(h) * N * P;

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
                        gdn_pack(hacc[mt][nt][2 * half], hacc[mt][nt][2 * half + 1]);
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
                unsigned a[4], b0, b1;
                gdn_ldsm(a, gdn_at_a(sc, GDN_LD, 0, kk * 16, l));
                const bf16* brow = sb + (warp * 8 + (l & 7)) * GDN_LD + kk * 16 + (((l >> 3) & 1) << 3);
                const unsigned addr = static_cast<unsigned>(__cvta_generic_to_shared(brow));
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                             : "=r"(b0), "=r"(b1)
                             : "r"(addr));
                gdn_mma(acc, a, b0, b1);
            }
            #pragma unroll
            for (int half = 0; half < 2; ++half) {
                const int i = g + 8 * half;
                const int j = warp * 8 + 2 * t;
                const bool live = i < L;
                const float p0 = live && j <= i ? acc[2 * half] * __expf(sG[i] - sG[j]) : 0.f;
                const float p1 = live && j + 1 <= i ? acc[2 * half + 1] * __expf(sG[i] - sG[j + 1]) : 0.f;
                *reinterpret_cast<unsigned*>(sp + i * KDA16_LDT + j) = gdn_pack(p0, p1);
            }
        }
        __syncthreads();

        // `exp(G) (C S) + P (dt x)` over the warp's dims, then `+ D x`, gated.
        {
            float qh[2][4] = {};
            #pragma unroll
            for (int kk = 0; kk < N / 16; ++kk) {
                unsigned aq[4], b[4];
                gdn_ldsm(aq, gdn_at_a(sc, GDN_LD, 0, kk * 16, l));
                gdn_ldsm_t(b, gdn_at_b(hs, LDV, kk * 16, c0, l));
                gdn_mma(qh[0], aq, b[0], b[1]);
                gdn_mma(qh[1], aq, b[2], b[3]);
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
            gdn_ldsm(pa, gdn_at_a(sp, KDA16_LDT, 0, 0, l));
            gdn_ldsm_t(b, gdn_at_b(su, LDV, 0, c0, l));
            gdn_mma(qh[0], pa, b[0], b[1]);
            gdn_mma(qh[1], pa, b[2], b[3]);
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
                        gdn_pack((qh[nt][2 * half] + skip * xv.x) * silu_f(zv.x),
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
            for (int mt = 0; mt < 2; ++mt) gdn_ldsm_t(a[mt], gdn_at_at(sb, GDN_LD, 0, kr0 + mt * 16, l));
            #pragma unroll
            for (int np = 0; np < P / 16; ++np) {
                unsigned b[4];
                gdn_ldsm_t(b, gdn_at_b(su2, LDV, 0, np * 16, l));
                #pragma unroll
                for (int mt = 0; mt < 2; ++mt) {
                    gdn_mma(hacc[mt][2 * np], a[mt], b[0], b[1]);
                    gdn_mma(hacc[mt][2 * np + 1], a[mt], b[2], b[3]);
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

}
