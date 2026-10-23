#include "ssm/tile_sm80.cuh"

namespace pie::ssm {

// A gated delta walk's f32 rows narrowed to bf16 where its trace asked for
// a bf16 output and the walk could not land one itself: a block a row and
// span of `blockDim.x` runs of four (a few rows' fire is otherwise a few
// blocks walking a long row), the staged window's rows only.
__global__ void gdn_narrow_rows(
    const float* __restrict__ src, __nv_bfloat16* __restrict__ dst, int width,
    const u32* __restrict__ win)
{
    const int r = blockIdx.x;
    if (win != nullptr && r >= static_cast<int>(win[0])) return;
    const long long row = win != nullptr ? r + static_cast<long long>(win[1]) : r;
    const float* s = src + row * width;
    __nv_bfloat16* d = dst + row * width;
    const int span = 4 * static_cast<int>(blockDim.x);
    const int c0 = static_cast<int>(blockIdx.y) * span;
    if ((width & 3) == 0) {
        const int c = c0 + 4 * static_cast<int>(threadIdx.x);
        if (c < width) {
            const float4 v = *reinterpret_cast<const float4*>(s + c);
            __nv_bfloat162 lo = __floats2bfloat162_rn(v.x, v.y);
            __nv_bfloat162 hi = __floats2bfloat162_rn(v.z, v.w);
            *reinterpret_cast<__nv_bfloat162*>(d + c) = lo;
            *reinterpret_cast<__nv_bfloat162*>(d + c + 2) = hi;
        }
        return;
    }
    for (int c = c0 + static_cast<int>(threadIdx.x); c < min(width, c0 + span); c += blockDim.x) {
        d[c] = __float2bfloat16(s[c]);
    }
}

// The chunk-parallel half of the chunked gated delta rule: one block per
// (chunk, value head) normalizes its chunk's key rows, scans the decay
// within the chunk, inverts `I + A` with `A = tril(beta_i k_i . k_j
// exp(G_i - G_j), -1)` and lands `w = T (beta exp(G) k)` and
// `u = T (beta v)`, the chunk's update directions, and the chunk's decayed
// causal scores `P = tril(q_i . k_j exp(G_i - G_j))`, beside the normalized
// queries and keys (written once per key head) and the decay's running sum.
template <int BK, int BV>
__global__ void __launch_bounds__(256) gdn_chunk_prep(
    const bf16* __restrict__ qkv_post,
    const float* __restrict__ gates,
    const int* __restrict__ slot_ids,
    const u32* __restrict__ qo_indptr,
    const int* __restrict__ commit_len,
    const int* __restrict__ begin_at,
    float* __restrict__ gcum,
    bf16* __restrict__ qn_out,
    bf16* __restrict__ kn_out,
    bf16* __restrict__ w_out,
    bf16* __restrict__ u_out,
    bf16* __restrict__ p_out,
    int lanes, int K_h, int V_h, int conv_dim,
    float q_scale,
    const int* __restrict__ snap,
    const int* __restrict__ snap_rows,
    const u32* __restrict__ win)
{
    static_assert(BK == 128 && BV == 128, "a lane carries four of a 128-wide row");
    constexpr int THREADS = 256;
    extern __shared__ __align__(16) unsigned char gdn_smem[];
    bf16* ks = reinterpret_cast<bf16*>(gdn_smem);
    bf16* vs = ks + GDN_CH * GDN_LD;
    bf16* t1 = vs + GDN_CH * GDN_LD;
    bf16* t2 = t1 + GDN_CH * GDN_LDC;
    float* A = reinterpret_cast<float*>(t2 + GDN_CH * GDN_LDC);
    float* G = A + GDN_CH * GDN_LDA;
    float* B = G + GDN_CH;

    const int h = blockIdx.x;
    const int s = blockIdx.y;
    if (win != nullptr) lanes = min(lanes, static_cast<int>(win[2]));
    if (lanes <= 0) return;
    const int r = gdn_slot_lane_snap(s, lanes, qo_indptr, begin_at, snap, win);
    const GdnLane lane = gdn_lane(r, qo_indptr, slot_ids, commit_len, begin_at, win);
    if (lane.T <= 0 || lane.slot < 0) return;
    const int j = s - gdn_chunk_base_snap(r, qo_indptr, begin_at, snap, win);
    if (j < 0) return;
    const GdnChunk ck = gdn_chunk(lane, snaps_of(snap, lane.rl), j, snap_rows);
    if (ck.L <= 0) return;
    const int ta = ck.ta;
    const int L = ck.L;
    const int row0 = win != nullptr ? static_cast<int>(win[1]) : 0;
    const int group = V_h / K_h;
    const int hk = h / group;
    const bool writer = h % group == 0;
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int l = tid & 31;
    const int K_dim = K_h * BK;

    // The chunk's raw key, value and query rows land whole before any is
    // read; the queries borrow the decay tile's space until the scores are
    // formed.
    bf16* qs = reinterpret_cast<bf16*>(A);
    for (int e = tid; e < GDN_CH * (BK / 8); e += THREADS) {
        const int i = e / (BK / 8);
        const int c = (e % (BK / 8)) * 8;
        const bool in = i < L;
        const bf16* row = qkv_post + static_cast<long long>(ta + (in ? i : 0) + row0) * conv_dim;
        gdn_cp16(ks + i * GDN_LD + c, row + K_dim + hk * BK + c, in);
        gdn_cp16(vs + i * GDN_LD + c, row + 2 * K_dim + h * BV + c, in);
        gdn_cp16(qs + i * GDN_LD + c, row + hk * BK + c, in);
    }
    gdn_cp_commit();
    if (tid < GDN_CH) {
        const bool in = tid < L;
        const float* grow = gates + static_cast<long long>(ta + (in ? tid : 0) + row0) * 2 * V_h;
        G[tid] = in ? grow[h] : 0.f;
        B[tid] = in ? grow[V_h + h] : 0.f;
    }
    gdn_cp_wait_all();
    __syncthreads();
    #pragma unroll 4
    for (int rr = 0; rr < 8; ++rr) {
        const int i = warp * 8 + rr;
        const bool in = i < L;
        uint2* kp = reinterpret_cast<uint2*>(ks + i * GDN_LD + l * 4);
        const uint2 kr = *kp;
        const float2 k01 = gdn_unpack(kr.x);
        const float2 k23 = gdn_unpack(kr.y);
        const float kinv = rsqrtf(
            warp_sum(k01.x * k01.x + k01.y * k01.y + k23.x * k23.x + k23.y * k23.y) + 1e-6f);
        const uint2 kw = make_uint2(pack_bf16x2(k01.x * kinv, k01.y * kinv),
                                    pack_bf16x2(k23.x * kinv, k23.y * kinv));
        *kp = kw;
        uint2* vp = reinterpret_cast<uint2*>(vs + i * GDN_LD + l * 4);
        const uint2 vr = *vp;
        const float beta = B[i];
        const float2 v01 = gdn_unpack(vr.x);
        const float2 v23 = gdn_unpack(vr.y);
        *vp = make_uint2(pack_bf16x2(v01.x * beta, v01.y * beta), pack_bf16x2(v23.x * beta, v23.y * beta));
        uint2* qp = reinterpret_cast<uint2*>(qs + i * GDN_LD + l * 4);
        const uint2 qr = *qp;
        const float2 q01 = gdn_unpack(qr.x);
        const float2 q23 = gdn_unpack(qr.y);
        const float qinv = rsqrtf(
            warp_sum(q01.x * q01.x + q01.y * q01.y + q23.x * q23.x + q23.y * q23.y) + 1e-6f)
            * q_scale;
        const uint2 qw = make_uint2(pack_bf16x2(q01.x * qinv, q01.y * qinv),
                                    pack_bf16x2(q23.x * qinv, q23.y * qinv));
        *qp = qw;
        if (writer && in) {
            const long long at = (static_cast<long long>(ta + i) * K_h + hk) * BK + l * 4;
            *reinterpret_cast<uint2*>(kn_out + at) = kw;
            *reinterpret_cast<uint2*>(qn_out + at) = qw;
        }
    }
    __syncthreads();
    if (warp == 0) {
        const float a = G[2 * l];
        const float b = a + G[2 * l + 1];
        float run = b;
        #pragma unroll
        for (int off = 1; off < 32; off <<= 1) {
            const float up = __shfl_up_sync(0xffffffffu, run, off);
            if (l >= off) run += up;
        }
        const float before = run - b;
        G[2 * l] = before + a;
        G[2 * l + 1] = run;
        #pragma unroll
        for (int e = 0; e < 2; ++e) {
            const int i = 2 * l + e;
            if (i < L) gcum[static_cast<long long>(ta + i) * V_h + h] = G[i];
        }
    }
    __syncthreads();

    // The warps split the chunk's rows four ways and the columns two ways.
    const int g = l >> 2;
    const int t = l & 3;
    const int m0 = (warp & 3) * 16;
    const int half_n = warp >> 2;
    {
        // `k k^T` and `q k^T` share their key operand.
        const int j0 = half_n * 32;
        float acc[4][4];
        float qkc[4][4];
        #pragma unroll
        for (int nt = 0; nt < 4; ++nt) {
            #pragma unroll
            for (int e = 0; e < 4; ++e) {
                acc[nt][e] = 0.f;
                qkc[nt][e] = 0.f;
            }
        }
        #pragma unroll
        for (int kk = 0; kk < BK / 16; ++kk) {
            unsigned a[4];
            unsigned aq[4];
            ldmatrix_x4(a, gdn_at_a(ks, GDN_LD, m0, kk * 16, l));
            ldmatrix_x4(aq, gdn_at_a(qs, GDN_LD, m0, kk * 16, l));
            #pragma unroll
            for (int np = 0; np < 2; ++np) {
                unsigned b[4];
                ldmatrix_x4(b, gdn_at_bt(ks, GDN_LD, j0 + np * 16, kk * 16, l));
                mma_m16n8k16(acc[2 * np], a, {b[0], b[1]});
                mma_m16n8k16(acc[2 * np + 1], a, {b[2], b[3]});
                mma_m16n8k16(qkc[2 * np], aq, {b[0], b[1]});
                mma_m16n8k16(qkc[2 * np + 1], aq, {b[2], b[3]});
            }
        }
        bf16* pd = p_out + (static_cast<long long>(s) * V_h + h) * GDN_CH * GDN_CH;
        #pragma unroll
        for (int nt = 0; nt < 4; ++nt) {
            #pragma unroll
            for (int half = 0; half < 2; ++half) {
                const int i = m0 + g + 8 * half;
                const int jj = j0 + nt * 8 + 2 * t;
                const float p0 = (jj <= i && i < L)
                    ? qkc[nt][2 * half] * __expf(G[i] - G[jj]) : 0.f;
                const float p1 = (jj + 1 <= i && i < L)
                    ? qkc[nt][2 * half + 1] * __expf(G[i] - G[jj + 1]) : 0.f;
                *reinterpret_cast<unsigned*>(pd + i * GDN_CH + jj) = pack_bf16x2(p0, p1);
            }
        }
        __syncthreads();
        #pragma unroll
        for (int nt = 0; nt < 4; ++nt) {
            #pragma unroll
            for (int e = 0; e < 4; ++e) {
                const int i = m0 + g + 8 * (e >> 1);
                const int jj = j0 + nt * 8 + 2 * t + (e & 1);
                A[i * GDN_LDA + jj] = (jj < i && i < L)
                    ? acc[nt][e] * B[i] * __expf(G[i] - G[jj])
                    : 0.f;
            }
        }
    }
    __syncthreads();

    // `T = (I + A)^-1` by sixteen-row blocks, in f32 throughout: each warp
    // inverts one diagonal block `D_b` by forward substitution, then each
    // later block row solves its off-diagonal blocks from the rows above it,
    // `T_ib = -D_i sum_{b<=k<i} A_ik T_kb`, one block a warp. `T` lands in
    // the space its two bf16 scalings take after it.
    float* Tm = reinterpret_cast<float*>(t1);
    float* E = B + GDN_CH;
    float* X = E + GDN_CH + warp * 256;
    if (tid < GDN_CH) E[tid] = B[tid] * __expf(G[tid]);
    if (warp < 4 && l < 16) {
        const int d0 = warp * 16;
        float tcol[16];
        #pragma unroll
        for (int rr = 0; rr < 16; ++rr) {
            const float* ar = A + (d0 + rr) * GDN_LDA + d0;
            float sum = 0.f;
            #pragma unroll
            for (int m = 0; m < rr; ++m) sum = __fmaf_rn(ar[m], tcol[m], sum);
            tcol[rr] = rr == l ? 1.f : (rr < l ? 0.f : -sum);
        }
        #pragma unroll
        for (int rr = 0; rr < 16; ++rr) Tm[(d0 + rr) * GDN_LDA + d0 + l] = tcol[rr];
    }
    {
        const int rr = l >> 1;
        const int cc = (l & 1) * 8;
        for (int bi = 1; bi < 4; ++bi) {
            __syncthreads();
            if (warp >= bi) continue;
            const int bb = warp;
            float x[8];
            #pragma unroll
            for (int u = 0; u < 8; ++u) x[u] = 0.f;
            for (int kb = bb; kb < bi; ++kb) {
                const float* ar = A + (bi * 16 + rr) * GDN_LDA + kb * 16;
                #pragma unroll
                for (int m = 0; m < 16; ++m) {
                    const float av = ar[m];
                    const float* tr = Tm + (kb * 16 + m) * GDN_LDA + bb * 16 + cc;
                    #pragma unroll
                    for (int u = 0; u < 8; ++u) x[u] = __fmaf_rn(av, tr[u], x[u]);
                }
            }
            #pragma unroll
            for (int u = 0; u < 8; ++u) X[rr * 16 + cc + u] = x[u];
            __syncwarp();
            float y[8];
            #pragma unroll
            for (int u = 0; u < 8; ++u) y[u] = 0.f;
            const float* dr = Tm + (bi * 16 + rr) * GDN_LDA + bi * 16;
            #pragma unroll
            for (int m = 0; m < 16; ++m) {
                const float dv = dr[m];
                #pragma unroll
                for (int u = 0; u < 8; ++u) y[u] = __fmaf_rn(dv, X[m * 16 + cc + u], y[u]);
            }
            float* tw = Tm + (bi * 16 + rr) * GDN_LDA + bb * 16 + cc;
            #pragma unroll
            for (int u = 0; u < 8; ++u) tw[u] = -y[u];
            __syncwarp();
        }
    }
    __syncthreads();
    {
        constexpr int PER = GDN_CH * GDN_CH / THREADS;
        float tv[PER];
        #pragma unroll
        for (int q = 0; q < PER; ++q) {
            const int e = q * THREADS + tid;
            const int i = e / GDN_CH;
            const int c = e % GDN_CH;
            tv[q] = (c >> 4) > (i >> 4) ? 0.f : Tm[i * GDN_LDA + c];
        }
        __syncthreads();
        #pragma unroll
        for (int q = 0; q < PER; ++q) {
            const int e = q * THREADS + tid;
            const int i = e / GDN_CH;
            const int c = e % GDN_CH;
            t1[i * GDN_LDC + c] = bf16(tv[q] * E[c]);
            t2[i * GDN_LDC + c] = bf16(tv[q]);
        }
    }
    __syncthreads();

    #pragma unroll
    for (int pass = 0; pass < 2; ++pass) {
        const bf16* tm = pass == 0 ? t1 : t2;
        const bf16* rhs = pass == 0 ? ks : vs;
        bf16* dst = pass == 0 ? w_out : u_out;
        const int n0 = half_n * 64;
        float acc[8][4];
        #pragma unroll
        for (int nt = 0; nt < 8; ++nt) {
            #pragma unroll
            for (int e = 0; e < 4; ++e) acc[nt][e] = 0.f;
        }
        #pragma unroll
        for (int kk = 0; kk < GDN_CH / 16; ++kk) {
            unsigned a[4];
            ldmatrix_x4(a, gdn_at_a(tm, GDN_LDC, m0, kk * 16, l));
            #pragma unroll
            for (int np = 0; np < 4; ++np) {
                unsigned b[4];
                ldmatrix_x4_trans(b, gdn_at_b(rhs, GDN_LD, kk * 16, n0 + np * 16, l));
                mma_m16n8k16(acc[2 * np], a, {b[0], b[1]});
                mma_m16n8k16(acc[2 * np + 1], a, {b[2], b[3]});
            }
        }
        #pragma unroll
        for (int half = 0; half < 2; ++half) {
            const int i = m0 + g + 8 * half;
            if (i >= L) continue;
            bf16* o = dst + (static_cast<long long>(ta + i) * V_h + h) * BV + n0 + 2 * t;
            #pragma unroll
            for (int nt = 0; nt < 8; ++nt) {
                *reinterpret_cast<unsigned*>(o + nt * 8) =
                    pack_bf16x2(acc[nt][2 * half], acc[nt][2 * half + 1]);
            }
        }
    }
}

// The sequential half: one block per (request, value head, `BV` values)
// walks the request's chunks with the `BK x BV` state in f32 accumulators,
// landing `q_i S + P v_new` (`v_new = u - w S`) and folding
// `S = D S + k^T v_new`, decaying by a scalar a row: `q_i` times
// `exp(G_i)`, `D = exp(G_L)`, the fold's `v_new` rows times
// `exp(G_L - G_j)`, its state bf16 `[keys][values]`. Rows split four ways
// and value columns two over the eight warps; each next-chunk operand
// streams in as soon as its buffer frees, so two blocks fit an SM.
template <int BV>
__global__ void __launch_bounds__(256) gdn_chunk_scan(
    const float* __restrict__ decay,
    const bf16* __restrict__ qn,
    const bf16* __restrict__ kn,
    const bf16* __restrict__ w,
    const bf16* __restrict__ u,
    const bf16* __restrict__ pm,
    void* __restrict__ state_base,
    const int* __restrict__ slot_ids,
    const u32* __restrict__ qo_indptr,
    long long slot_stride_elems,
    const int* __restrict__ commit_len,
    const int* __restrict__ begin_at,
    bool write_state,
    const u8* __restrict__ write_state_mask,
    float* __restrict__ out,
    int lanes, int K_h, int V_h,
    const int* __restrict__ snap,
    const int* __restrict__ snap_rows,
    const int* __restrict__ snap_slots,
    const u32* __restrict__ win)
{
    constexpr int BK = 128;
    constexpr int V = 128;
    constexpr int THREADS = 256;
    constexpr int NT = BV / 8;
    constexpr int CN = BV / 2;
    constexpr int CT = CN / 8;
    constexpr int LDV = BV + 8;
    // A chunk's decay: a row a step.
    constexpr int GD = GDN_CH;
    using State = __nv_bfloat16;
    static_assert(CN % 16 == 0 && V % BV == 0, "a warp's value columns are whole sixteen-wide pairs");
    extern __shared__ __align__(16) unsigned char gdn_smem[];
    bf16* sq = reinterpret_cast<bf16*>(gdn_smem);
    bf16* sw = sq + GDN_CH * GDN_LD;
    bf16* sk = sw + GDN_CH * GDN_LD;
    bf16* su = sk + GDN_CH * GDN_LD;
    bf16* sp = su + GDN_CH * LDV;
    bf16* hs = sp + GDN_CH * GDN_LDC;
    bf16* vn = hs + BK * LDV;
    // The fold's decayed rows.
    bf16* vsc = vn + GDN_CH * LDV;
    float* sg0 = reinterpret_cast<float*>(vsc + GDN_CH * LDV);

    const int nvt = V / BV;
    const int vt = static_cast<int>(blockIdx.x) % nvt;
    const int h = (static_cast<int>(blockIdx.x) / nvt) % V_h;
    const int r = static_cast<int>(blockIdx.x) / (nvt * V_h);
    if (win != nullptr) lanes = min(lanes, static_cast<int>(win[2]));
    if (r >= lanes) return;
    const GdnLane lane = gdn_lane(r, qo_indptr, slot_ids, commit_len, begin_at, win);
    if (lane.T <= 0 || lane.slot < 0) return;
    const int row0 = win != nullptr ? static_cast<int>(win[1]) : 0;
    const int hk = h / (V_h / K_h);
    const int v0 = vt * BV;
    const int base = gdn_chunk_base_snap(r, qo_indptr, begin_at, snap, win);
    const Snaps sn = snaps_of(snap, lane.rl);
    const bool lands = write_state && row_persists(write_state_mask, lane.rl);
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int l = tid & 31;
    const int g = l >> 2;
    const int t = l & 3;
    const int m0 = (warp & 3) * 16;
    const int c0 = (warp >> 2) * CN;
    // The state splits over the warps as the output does: 32 key rows by
    // the warp's value half, two row tiles of `CT` column tiles each.
    const int kr0 = (warp & 3) * 32;
    auto h_row = [&](int q, int half) { return kr0 + (q / CT) * 16 + g + 8 * half; };
    auto h_col = [&](int q) { return c0 + (q % CT) * 8 + 2 * t; };
    State* state = static_cast<State*>(state_base)
        + static_cast<long long>(lane.slot) * slot_stride_elems
        + static_cast<long long>(h) * BK * V;

    float hacc[NT][4];
    #pragma unroll
    for (int q = 0; q < NT; ++q) {
        #pragma unroll
        for (int half = 0; half < 2; ++half) {
            const int kk = h_row(q, half);
            const int vv = v0 + h_col(q);
            const float2 f = gdn_unpack(*reinterpret_cast<const unsigned*>(
                state + static_cast<long long>(kk) * V + vv));
            hacc[q][2 * half] = f.x;
            hacc[q][2 * half + 1] = f.y;
        }
    }

    const int nchunks = gdn_chunks(lane, sn, snap_rows);
    auto span = [&](int jc, int& ta, int& L) {
        const GdnChunk ck = gdn_chunk(lane, sn, jc, snap_rows);
        ta = ck.ta;
        L = ck.L;
    };
    auto load_k = [&](int jc) {
        int ta, L;
        span(jc, ta, L);
        #pragma unroll
        for (int e = tid; e < GDN_CH * (BK / 8); e += THREADS) {
            const int i = e / (BK / 8);
            const int c = (e % (BK / 8)) * 8;
            const bool in = i < L;
            gdn_cp16(sk + i * GDN_LD + c,
                     kn + (static_cast<long long>(ta + (in ? i : 0)) * K_h + hk) * BK + c, in);
        }
    };
    auto load_g = [&](int jc) {
        float* sg = sg0 + (jc & 1) * GD;
        int ta, L;
        span(jc, ta, L);
        if (tid < GDN_CH) {
            const bool in = tid < L;
            gdn_cp4(sg + tid, decay + static_cast<long long>(ta + (in ? tid : 0)) * V_h + h, in);
        }
    };
    auto load_qwu = [&](int jc) {
        int ta, L;
        span(jc, ta, L);
        #pragma unroll
        for (int e = tid; e < GDN_CH * (BK / 8); e += THREADS) {
            const int i = e / (BK / 8);
            const int c = (e % (BK / 8)) * 8;
            const bool in = i < L;
            const long long row = ta + (in ? i : 0);
            gdn_cp16(sq + i * GDN_LD + c, qn + (row * K_h + hk) * BK + c, in);
            gdn_cp16(sw + i * GDN_LD + c, w + (row * V_h + h) * BK + c, in);
        }
        #pragma unroll
        for (int e = tid; e < GDN_CH * (BV / 8); e += THREADS) {
            const int i = e / (BV / 8);
            const int c = (e % (BV / 8)) * 8;
            const bool in = i < L;
            gdn_cp16(su + i * LDV + c,
                     u + (static_cast<long long>(ta + (in ? i : 0)) * V_h + h) * V + v0 + c, in);
        }
    };
    auto load_p = [&](int jc) {
        const bf16* src = pm + (static_cast<long long>(base + jc) * V_h + h) * GDN_CH * GDN_CH;
        #pragma unroll
        for (int e = tid; e < GDN_CH * (GDN_CH / 8); e += THREADS) {
            const int i = e / (GDN_CH / 8);
            const int c = (e % (GDN_CH / 8)) * 8;
            gdn_cp16(sp + i * GDN_LDC + c, src + i * GDN_CH + c, true);
        }
    };

    load_g(0);
    load_qwu(0);
    load_p(0);
    gdn_cp_commit();
    #pragma unroll
    for (int q = 0; q < NT; ++q) {
        #pragma unroll
        for (int half = 0; half < 2; ++half) {
            *reinterpret_cast<unsigned*>(hs + h_row(q, half) * LDV + h_col(q)) =
                pack_bf16x2(hacc[q][2 * half], hacc[q][2 * half + 1]);
        }
    }

    for (int jc = 0; jc < nchunks; ++jc) {
        gdn_cp_wait_all();
        __syncthreads();
        const bool more = jc + 1 < nchunks;
        // This chunk's keys, read only by its state fold, into the one key
        // buffer the last fold is done with; the next chunk's decay ahead.
        load_k(jc);
        if (more) load_g(jc + 1);
        gdn_cp_commit();

        int ta, L;
        span(jc, ta, L);
        const float* sg = sg0 + (jc & 1) * GD;
        const float g_last = sg[L - 1];

        float qh[CT][4];
        float wh[CT][4];
        #pragma unroll
        for (int nt = 0; nt < CT; ++nt) {
            #pragma unroll
            for (int e = 0; e < 4; ++e) {
                qh[nt][e] = 0.f;
                wh[nt][e] = 0.f;
            }
        }
        #pragma unroll
        for (int kk = 0; kk < BK / 16; ++kk) {
            unsigned aq[4];
            unsigned aw[4];
            ldmatrix_x4(aq, gdn_at_a(sq, GDN_LD, m0, kk * 16, l));
            ldmatrix_x4(aw, gdn_at_a(sw, GDN_LD, m0, kk * 16, l));
            #pragma unroll
            for (int np = 0; np < CT / 2; ++np) {
                unsigned b[4];
                ldmatrix_x4_trans(b, gdn_at_b(hs, LDV, kk * 16, c0 + np * 16, l));
                mma_m16n8k16(qh[2 * np], aq, {b[0], b[1]});
                mma_m16n8k16(qh[2 * np + 1], aq, {b[2], b[3]});
                mma_m16n8k16(wh[2 * np], aw, {b[0], b[1]});
                mma_m16n8k16(wh[2 * np + 1], aw, {b[2], b[3]});
            }
        }
        #pragma unroll
        for (int half = 0; half < 2; ++half) {
            const int i = m0 + g + 8 * half;
            const bool in = i < L;
            #pragma unroll
            for (int nt = 0; nt < CT; ++nt) {
                const int c = c0 + nt * 8 + 2 * t;
                const float2 uu = gdn_unpack(*reinterpret_cast<const unsigned*>(su + i * LDV + c));
                const float n0 = in ? uu.x - wh[nt][2 * half] : 0.f;
                const float n1 = in ? uu.y - wh[nt][2 * half + 1] : 0.f;
                *reinterpret_cast<unsigned*>(vn + i * LDV + c) = pack_bf16x2(n0, n1);
                const float d = in ? __expf(g_last - sg[i]) : 0.f;
                *reinterpret_cast<unsigned*>(vsc + i * LDV + c) = pack_bf16x2(n0 * d, n1 * d);
            }
        }
        __syncthreads();
        if (more) load_qwu(jc + 1);
        gdn_cp_commit();

        #pragma unroll
        for (int half = 0; half < 2; ++half) {
            const int i = m0 + g + 8 * half;
            const float eg = __expf(sg[i]);
            #pragma unroll
            for (int nt = 0; nt < CT; ++nt) {
                qh[nt][2 * half] *= eg;
                qh[nt][2 * half + 1] *= eg;
            }
        }
        #pragma unroll
        for (int kk = 0; kk < GDN_CH / 16; ++kk) {
            unsigned pa[4];
            ldmatrix_x4(pa, gdn_at_a(sp, GDN_LDC, m0, kk * 16, l));
            #pragma unroll
            for (int np = 0; np < CT / 2; ++np) {
                unsigned b[4];
                ldmatrix_x4_trans(b, gdn_at_b(vn, LDV, kk * 16, c0 + np * 16, l));
                mma_m16n8k16(qh[2 * np], pa, {b[0], b[1]});
                mma_m16n8k16(qh[2 * np + 1], pa, {b[2], b[3]});
            }
        }
        #pragma unroll
        for (int half = 0; half < 2; ++half) {
            const int i = m0 + g + 8 * half;
            if (i >= L) continue;
            float* o = out + (static_cast<long long>(ta + i + row0) * V_h + h) * V + v0 + c0 + 2 * t;
            #pragma unroll
            for (int nt = 0; nt < CT; ++nt) {
                *reinterpret_cast<float2*>(o + nt * 8) =
                    make_float2(qh[nt][2 * half], qh[nt][2 * half + 1]);
            }
        }
        __syncthreads();
        if (more) load_p(jc + 1);
        gdn_cp_commit();

        // The keys' group is in once the two after it alone may be pending.
        asm volatile("cp.async.wait_group 2;\n" ::: "memory");
        __syncthreads();
        {
            const float el = __expf(g_last);
            #pragma unroll
            for (int nt = 0; nt < NT; ++nt) {
                #pragma unroll
                for (int e = 0; e < 4; ++e) hacc[nt][e] *= el;
            }
        }
        #pragma unroll
        for (int kk = 0; kk < GDN_CH / 16; ++kk) {
            unsigned a[2][4];
            #pragma unroll
            for (int mt = 0; mt < 2; ++mt) {
                ldmatrix_x4_trans(a[mt], gdn_at_at(sk, GDN_LD, kk * 16, kr0 + mt * 16, l));
            }
            #pragma unroll
            for (int np = 0; np < CT / 2; ++np) {
                unsigned b[4];
                ldmatrix_x4_trans(b, gdn_at_b(vsc, LDV, kk * 16, c0 + np * 16, l));
                #pragma unroll
                for (int mt = 0; mt < 2; ++mt) {
                    mma_m16n8k16(hacc[mt * CT + 2 * np], a[mt], {b[0], b[1]});
                    mma_m16n8k16(hacc[mt * CT + 2 * np + 1], a[mt], {b[2], b[3]});
                }
            }
        }
        // A snapshot lands the state as stored and the walk carries on from
        // it rounded, as a pass cut there reloads it.
        const int at_snap = sn.e0 < sn.e1 ? gdn_chunk(lane, sn, jc, snap_rows).snap : -1;
        #pragma unroll
        for (int q = 0; q < NT; ++q) {
            #pragma unroll
            for (int half = 0; half < 2; ++half) {
                const unsigned packed = pack_bf16x2(hacc[q][2 * half], hacc[q][2 * half + 1]);
                *reinterpret_cast<unsigned*>(hs + h_row(q, half) * LDV + h_col(q)) = packed;
                if (at_snap >= 0) {
                    const float2 f = gdn_unpack(packed);
                    hacc[q][2 * half] = f.x;
                    hacc[q][2 * half + 1] = f.y;
                    if (lands) {
                        State* sat = static_cast<State*>(state_base)
                            + static_cast<long long>(snap_slots[at_snap]) * slot_stride_elems
                            + static_cast<long long>(h) * BK * V;
                        *reinterpret_cast<unsigned*>(
                            sat + static_cast<long long>(h_row(q, half)) * V + v0 + h_col(q)) = packed;
                    }
                }
            }
        }
    }
    gdn_cp_wait_all();

    if (lands) {
        #pragma unroll
        for (int q = 0; q < NT; ++q) {
            #pragma unroll
            for (int half = 0; half < 2; ++half) {
                const int kk = h_row(q, half);
                const int vv = v0 + h_col(q);
                *reinterpret_cast<unsigned*>(state + static_cast<long long>(kk) * V + vv) =
                    pack_bf16x2(hacc[q][2 * half], hacc[q][2 * half + 1]);
            }
        }
    }
}

// The chunked rule in three wmma launches, for a card whose blocks cannot
// hold the scan's shared memory: per (chunk, value head) the update
// directions `w` and `u`; per (request, value head, sixteen value columns)
// the chunk-by-chunk state walk; per (chunk, value head) the output.
constexpr int GDN_WARPS = 8;

template <int BK, int BV>
__global__ void __launch_bounds__(256) gdn_chunk_wu(
    const float* __restrict__ q_norm,
    const float* __restrict__ k_norm,
    const float* __restrict__ v,
    const float* __restrict__ g_log,
    const float* __restrict__ beta,
    const int* __restrict__ slot_ids,
    const u32* __restrict__ qo_indptr,
    const int* __restrict__ commit_len,
    const int* __restrict__ begin_at,
    float* __restrict__ gcum,
    __nv_bfloat16* __restrict__ qb,
    __nv_bfloat16* __restrict__ kb_out,
    __nv_bfloat16* __restrict__ kT,
    __nv_bfloat16* __restrict__ w_out,
    __nv_bfloat16* __restrict__ u_out,
    int lanes, int K_h, int V_h,
    const u32* __restrict__ win)
{
    static_assert(BV == BK, "the value rows reuse the key buffer");
    namespace wmma = ::nvcuda::wmma;
    extern __shared__ float smem[];
    float* A = smem;
    __nv_bfloat16* kb = reinterpret_cast<__nv_bfloat16*>(smem + GDN_CH * GDN_CH);
    __nv_bfloat16* Tb = kb + GDN_CH * BK;
    float* G = reinterpret_cast<float*>(Tb + GDN_CH * GDN_CH);
    float* B = G + GDN_CH;
    float* BE = B + GDN_CH;
    float* arow = BE + GDN_CH;
    const int h = blockIdx.y;
    const int s = blockIdx.x;
    if (win != nullptr) lanes = min(lanes, static_cast<int>(win[2]));
    if (lanes <= 0) return;
    const int r = gdn_slot_lane(s, lanes, qo_indptr, begin_at, win);
    const GdnLane lane = gdn_lane(r, qo_indptr, slot_ids, commit_len, begin_at, win);
    if (lane.T <= 0 || lane.slot < 0) return;
    const int j = s - (lane.t0 / GDN_CH + r);
    if (j < 0 || j * GDN_CH >= lane.T) return;
    const int ta = lane.t0 + j * GDN_CH;
    const int L = min(GDN_CH, lane.t0 + lane.T - ta);
    const int hk = h / (V_h / K_h);
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    if (tid == 0) {
        float acc = 0.f;
        for (int i = 0; i < L; ++i) {
            acc += g_log[static_cast<long long>(ta + i) * V_h + h];
            G[i] = acc;
        }
    }
    for (int i = tid; i < L; i += blockDim.x) {
        B[i] = beta[static_cast<long long>(ta + i) * V_h + h];
    }
    __nv_bfloat16* kT_s = kT + (static_cast<long long>(s) * K_h + hk) * BK * GDN_CH;
    for (int e = tid; e < GDN_CH * BK; e += blockDim.x) {
        const int i = e / BK, kk = e % BK;
        const long long row = (static_cast<long long>(ta + i) * K_h + hk) * BK + kk;
        const __nv_bfloat16 kv = i < L ? __float2bfloat16(k_norm[row]) : __float2bfloat16(0.f);
        kb[e] = kv;
        kT_s[static_cast<long long>(kk) * GDN_CH + i] = kv;
        if (i < L) {
            kb_out[row] = kv;
            qb[row] = __float2bfloat16(q_norm[row]);
        }
    }
    __syncthreads();
    for (int i = tid; i < L; i += blockDim.x) {
        gcum[static_cast<long long>(ta + i) * V_h + h] = G[i];
        BE[i] = B[i] * __expf(G[i]);
    }
    {
        wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> a;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> b;
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> c;
        for (int tile = warp; tile < 16; tile += GDN_WARPS) {
            const int m0 = (tile >> 2) * 16, n0 = (tile & 3) * 16;
            wmma::fill_fragment(c, 0.f);
            for (int k0 = 0; k0 < BK; k0 += 16) {
                wmma::load_matrix_sync(a, kb + m0 * BK + k0, BK);
                wmma::load_matrix_sync(b, kb + n0 * BK + k0, BK);
                wmma::mma_sync(c, a, b, c);
            }
            wmma::store_matrix_sync(A + m0 * GDN_CH + n0, c, GDN_CH, wmma::mem_row_major);
        }
    }
    __syncthreads();
    for (int e = tid; e < GDN_CH * GDN_CH; e += blockDim.x) {
        const int i = e / GDN_CH, jj = e % GDN_CH;
        A[e] = (i < L && jj < i) ? A[e] * B[i] * __expf(G[i] - G[jj]) : 0.f;
    }
    __syncthreads();
    for (int i = 0; i < L; ++i) {
        if (tid < i) arow[tid] = A[i * GDN_CH + tid];
        __syncthreads();
        if (tid < GDN_CH) {
            const int c = tid;
            float t = (c == i) ? 1.f : 0.f;
            if (c < i) {
                for (int jj = c; jj < i; ++jj) t -= arow[jj] * A[jj * GDN_CH + c];
            }
            A[i * GDN_CH + c] = t;
        }
        __syncthreads();
    }
    for (int e = tid; e < GDN_CH * GDN_CH; e += blockDim.x) {
        const int i = e / GDN_CH, jj = e % GDN_CH;
        Tb[e] = __float2bfloat16((i < L && jj < L) ? A[e] * BE[jj] : 0.f);
    }
    __syncthreads();
    {
        wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> a;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> b;
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> c;
        float* mine = reinterpret_cast<float*>(kb) + warp * 256;
        const int lane_id = tid & 31;
        for (int tile = warp; tile < 4 * (BK / 16); tile += GDN_WARPS) {
            const int m0 = (tile / (BK / 16)) * 16, n0 = (tile % (BK / 16)) * 16;
            wmma::fill_fragment(c, 0.f);
            for (int k0 = 0; k0 < GDN_CH; k0 += 16) {
                wmma::load_matrix_sync(a, Tb + m0 * GDN_CH + k0, GDN_CH);
                wmma::load_matrix_sync(b, kT_s + n0 * GDN_CH + k0, GDN_CH);
                wmma::mma_sync(c, a, b, c);
            }
            wmma::store_matrix_sync(mine, c, 16, wmma::mem_row_major);
            __syncwarp();
            for (int e = lane_id; e < 256; e += 32) {
                const int i = m0 + e / 16, kk = n0 + e % 16;
                if (i < L) {
                    w_out[(static_cast<long long>(ta + i) * V_h + h) * BK + kk] = __float2bfloat16(mine[e]);
                }
            }
            __syncwarp();
        }
    }
    __syncthreads();
    for (int e = tid; e < GDN_CH * GDN_CH; e += blockDim.x) {
        const int i = e / GDN_CH, jj = e % GDN_CH;
        Tb[e] = __float2bfloat16((i < L && jj < L) ? A[e] : 0.f);
    }
    __nv_bfloat16* vbT = kb;
    for (int e = tid; e < GDN_CH * BV; e += blockDim.x) {
        const int i = e / BV, vv = e % BV;
        const float val = i < L ? v[(static_cast<long long>(ta + i) * V_h + h) * BV + vv] * B[i] : 0.f;
        vbT[vv * GDN_CH + i] = __float2bfloat16(val);
    }
    __syncthreads();
    {
        wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> a;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> b;
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> c;
        float* mine = A + warp * 256;
        const int lane_id = tid & 31;
        for (int tile = warp; tile < 4 * (BV / 16); tile += GDN_WARPS) {
            const int m0 = (tile / (BV / 16)) * 16, n0 = (tile % (BV / 16)) * 16;
            wmma::fill_fragment(c, 0.f);
            for (int k0 = 0; k0 < GDN_CH; k0 += 16) {
                wmma::load_matrix_sync(a, Tb + m0 * GDN_CH + k0, GDN_CH);
                wmma::load_matrix_sync(b, vbT + n0 * GDN_CH + k0, GDN_CH);
                wmma::mma_sync(c, a, b, c);
            }
            wmma::store_matrix_sync(mine, c, 16, wmma::mem_row_major);
            __syncwarp();
            for (int e = lane_id; e < 256; e += 32) {
                const int i = m0 + e / 16, vv = n0 + e % 16;
                if (i < L) {
                    u_out[(static_cast<long long>(ta + i) * V_h + h) * BV + vv] = __float2bfloat16(mine[e]);
                }
            }
            __syncwarp();
        }
    }
}

template <typename StateT, int BK, int BV, int BVT>
__global__ void __launch_bounds__(256) gdn_chunk_h(
    const float* __restrict__ gcum,
    const __nv_bfloat16* __restrict__ kT,
    const __nv_bfloat16* __restrict__ w,
    __nv_bfloat16* __restrict__ u,
    __nv_bfloat16* __restrict__ vnT,
    __nv_bfloat16* __restrict__ hcT,
    StateT* __restrict__ state_base,
    const int* __restrict__ slot_ids,
    const u32* __restrict__ qo_indptr,
    long long slot_stride_elems,
    const int* __restrict__ commit_len,
    const int* __restrict__ begin_at,
    bool write_state,
    const u8* __restrict__ write_state_mask,
    int lanes, int K_h, int V_h,
    const u32* __restrict__ win)
{
    static_assert(BVT == 16, "one value tile per mma column");
    namespace wmma = ::nvcuda::wmma;
    __shared__ float hf[BK * BVT];
    __shared__ float dl[BK * BVT];
    __shared__ float vn[GDN_CH * BVT];
    __shared__ __nv_bfloat16 hT[BVT * BK];
    __shared__ __nv_bfloat16 vnTs[BVT * GDN_CH];
    __shared__ float sE[GDN_CH];
    const int vt = static_cast<int>(blockIdx.x) % (BV / BVT);
    const int h = (static_cast<int>(blockIdx.x) / (BV / BVT)) % V_h;
    const int r = static_cast<int>(blockIdx.x) / ((BV / BVT) * V_h);
    if (win != nullptr) lanes = min(lanes, static_cast<int>(win[2]));
    if (r >= lanes) return;
    const GdnLane lane = gdn_lane(r, qo_indptr, slot_ids, commit_len, begin_at, win);
    if (lane.T <= 0 || lane.slot < 0) return;
    const int hk = h / (V_h / K_h);
    const int v0 = vt * BVT;
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    StateT* state = state_base
        + static_cast<long long>(lane.slot) * slot_stride_elems
        + static_cast<long long>(h) * BK * BV;
    for (int e = tid; e < BK * BVT; e += blockDim.x) {
        const int kk = e / BVT, vv = e % BVT;
        hf[e] = state_load(state + static_cast<long long>(kk) * BV + v0 + vv);
    }
    __syncthreads();
    const int base = lane.t0 / GDN_CH + r;
    const int nchunks = (lane.T + GDN_CH - 1) / GDN_CH;
    wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> a;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> b;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c;
    for (int j = 0; j < nchunks; ++j) {
        const int sj = base + j;
        const int ta = lane.t0 + j * GDN_CH;
        const int L = min(GDN_CH, lane.t0 + lane.T - ta);
        __nv_bfloat16* hcT_s = hcT + (static_cast<long long>(sj) * V_h + h) * BV * BK;
        for (int e = tid; e < BVT * BK; e += blockDim.x) {
            const int vv = e / BK, kk = e % BK;
            const __nv_bfloat16 hv = __float2bfloat16(hf[kk * BVT + vv]);
            hT[e] = hv;
            hcT_s[static_cast<long long>(v0 + vv) * BK + kk] = hv;
        }
        if (tid < GDN_CH) {
            const float gl = gcum[static_cast<long long>(ta + L - 1) * V_h + h];
            sE[tid] = tid < L ? __expf(gl - gcum[static_cast<long long>(ta + tid) * V_h + h]) : 0.f;
        }
        __syncthreads();
        if (warp < 4) {
            const int m0 = warp * 16;
            wmma::fill_fragment(c, 0.f);
            for (int k0 = 0; k0 < BK; k0 += 16) {
                wmma::load_matrix_sync(a, w + (static_cast<long long>(ta + m0) * V_h + h) * BK + k0, V_h * BK);
                wmma::load_matrix_sync(b, hT + k0, BK);
                wmma::mma_sync(c, a, b, c);
            }
            wmma::store_matrix_sync(vn + m0 * BVT, c, BVT, wmma::mem_row_major);
        }
        __syncthreads();
        __nv_bfloat16* vnT_s = vnT + (static_cast<long long>(sj) * V_h + h) * BV * GDN_CH;
        for (int e = tid; e < GDN_CH * BVT; e += blockDim.x) {
            const int i = e / BVT, vv = e % BVT;
            float vnew = 0.f;
            if (i < L) {
                const long long ui = (static_cast<long long>(ta + i) * V_h + h) * BV + v0 + vv;
                vnew = __bfloat162float(u[ui]) - vn[e];
                u[ui] = __float2bfloat16(vnew);
            }
            const __nv_bfloat16 vb = __float2bfloat16(vnew);
            vnT_s[static_cast<long long>(v0 + vv) * GDN_CH + i] = vb;
            vnTs[vv * GDN_CH + i] = __float2bfloat16(vnew * sE[i]);
        }
        __syncthreads();
        {
            const int m0 = warp * 16;
            wmma::fill_fragment(c, 0.f);
            const __nv_bfloat16* kT_s = kT + (static_cast<long long>(sj) * K_h + hk) * BK * GDN_CH;
            for (int k0 = 0; k0 < GDN_CH; k0 += 16) {
                wmma::load_matrix_sync(a, kT_s + m0 * GDN_CH + k0, GDN_CH);
                wmma::load_matrix_sync(b, vnTs + k0, GDN_CH);
                wmma::mma_sync(c, a, b, c);
            }
            wmma::store_matrix_sync(dl + m0 * BVT, c, BVT, wmma::mem_row_major);
        }
        __syncthreads();
        const float gl = __expf(gcum[static_cast<long long>(ta + L - 1) * V_h + h]);
        for (int e = tid; e < BK * BVT; e += blockDim.x) {
            hf[e] = hf[e] * gl + dl[e];
        }
        __syncthreads();
    }
    if (write_state && row_persists(write_state_mask, lane.rl)) {
        for (int e = tid; e < BK * BVT; e += blockDim.x) {
            const int kk = e / BVT, vv = e % BVT;
            state_store(state + static_cast<long long>(kk) * BV + v0 + vv, hf[e]);
        }
    }
}

template <int BK, int BV, int BVT>
__global__ void __launch_bounds__(256) gdn_chunk_o(
    const float* __restrict__ gcum,
    const __nv_bfloat16* __restrict__ qb,
    const __nv_bfloat16* __restrict__ kb,
    const __nv_bfloat16* __restrict__ vnT,
    const __nv_bfloat16* __restrict__ hcT,
    const int* __restrict__ slot_ids,
    const u32* __restrict__ qo_indptr,
    const int* __restrict__ commit_len,
    const int* __restrict__ begin_at,
    float* __restrict__ out,
    int lanes, int K_h, int V_h,
    const u32* __restrict__ win)
{
    static_assert(BVT == 16, "one value tile per mma column");
    namespace wmma = ::nvcuda::wmma;
    __shared__ float S[GDN_CH * GDN_CH];
    __shared__ __nv_bfloat16 Sb[GDN_CH * GDN_CH];
    __shared__ float o1[GDN_CH * BVT];
    __shared__ float o2[GDN_CH * BVT];
    __shared__ float sG[GDN_CH];
    const int h = blockIdx.y;
    const int s = blockIdx.x;
    if (win != nullptr) lanes = min(lanes, static_cast<int>(win[2]));
    if (lanes <= 0) return;
    const int r = gdn_slot_lane(s, lanes, qo_indptr, begin_at, win);
    const GdnLane lane = gdn_lane(r, qo_indptr, slot_ids, commit_len, begin_at, win);
    if (lane.T <= 0 || lane.slot < 0) return;
    const int j = s - (lane.t0 / GDN_CH + r);
    if (j < 0 || j * GDN_CH >= lane.T) return;
    const int ta = lane.t0 + j * GDN_CH;
    const int L = min(GDN_CH, lane.t0 + lane.T - ta);
    const int row0 = win != nullptr ? static_cast<int>(win[1]) : 0;
    const int hk = h / (V_h / K_h);
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    for (int i = tid; i < GDN_CH; i += blockDim.x) {
        sG[i] = i < L ? gcum[static_cast<long long>(ta + i) * V_h + h] : 0.f;
    }
    wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> a;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> b;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c;
    const __nv_bfloat16* q_s = qb + (static_cast<long long>(ta) * K_h + hk) * BK;
    const __nv_bfloat16* k_s = kb + (static_cast<long long>(ta) * K_h + hk) * BK;
    for (int tile = warp; tile < 16; tile += GDN_WARPS) {
        const int m0 = (tile >> 2) * 16, n0 = (tile & 3) * 16;
        wmma::fill_fragment(c, 0.f);
        for (int k0 = 0; k0 < BK; k0 += 16) {
            wmma::load_matrix_sync(a, q_s + static_cast<long long>(m0) * K_h * BK + k0, K_h * BK);
            wmma::load_matrix_sync(b, k_s + static_cast<long long>(n0) * K_h * BK + k0, K_h * BK);
            wmma::mma_sync(c, a, b, c);
        }
        wmma::store_matrix_sync(S + m0 * GDN_CH + n0, c, GDN_CH, wmma::mem_row_major);
    }
    __syncthreads();
    for (int e = tid; e < GDN_CH * GDN_CH; e += blockDim.x) {
        const int i = e / GDN_CH, jj = e % GDN_CH;
        Sb[e] = __float2bfloat16((i < L && jj <= i) ? S[e] * __expf(sG[i] - sG[jj]) : 0.f);
    }
    __syncthreads();
    const __nv_bfloat16* hcT_s = hcT + (static_cast<long long>(s) * V_h + h) * BV * BK;
    const __nv_bfloat16* vnT_s = vnT + (static_cast<long long>(s) * V_h + h) * BV * GDN_CH;
    for (int vt = 0; vt < BV / BVT; ++vt) {
        const int v0 = vt * BVT;
        if (warp < 4) {
            const int m0 = warp * 16;
            wmma::fill_fragment(c, 0.f);
            for (int k0 = 0; k0 < BK; k0 += 16) {
                wmma::load_matrix_sync(a, q_s + static_cast<long long>(m0) * K_h * BK + k0, K_h * BK);
                wmma::load_matrix_sync(b, hcT_s + static_cast<long long>(v0) * BK + k0, BK);
                wmma::mma_sync(c, a, b, c);
            }
            wmma::store_matrix_sync(o1 + m0 * BVT, c, BVT, wmma::mem_row_major);
        } else {
            const int m0 = (warp - 4) * 16;
            wmma::fill_fragment(c, 0.f);
            for (int k0 = 0; k0 < GDN_CH; k0 += 16) {
                wmma::load_matrix_sync(a, Sb + m0 * GDN_CH + k0, GDN_CH);
                wmma::load_matrix_sync(b, vnT_s + static_cast<long long>(v0) * GDN_CH + k0, GDN_CH);
                wmma::mma_sync(c, a, b, c);
            }
            wmma::store_matrix_sync(o2 + m0 * BVT, c, BVT, wmma::mem_row_major);
        }
        __syncthreads();
        for (int e = tid; e < GDN_CH * BVT; e += blockDim.x) {
            const int i = e / BVT, vv = e % BVT;
            if (i < L) {
                out[(static_cast<long long>(ta + i + row0) * V_h + h) * BV + v0 + vv]
                    = o1[e] * __expf(sG[i]) + o2[e];
            }
        }
        __syncthreads();
    }
}

}
