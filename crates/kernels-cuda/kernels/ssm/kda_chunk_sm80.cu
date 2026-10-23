#include "ssm/tile_sm80.cuh"

namespace pie::ssm {

__global__ void __launch_bounds__(KDA16_TASKS * 64, 4) kda_chunk_prep(
    const bf16* __restrict__ mixed,
    float eps,
    const bf16* __restrict__ raw_g,
    const bf16* __restrict__ raw_beta,
    const float* __restrict__ A_log,
    const float* __restrict__ dt_bias,
    float lower_bound,
    const int* __restrict__ slot_ids,
    const u32* __restrict__ qo_indptr,
    const int* __restrict__ commit_len,
    const int* __restrict__ begin_at,
    bf16* __restrict__ qg_out,
    bf16* __restrict__ kg_out,
    bf16* __restrict__ ke_out,
    bf16* __restrict__ tb_out,
    bf16* __restrict__ p_out,
    float* __restrict__ gl_out,
    int lanes, int H,
    const u32* __restrict__ win)
{
    constexpr int D = 128;
    constexpr int R = KDA_CH / 2;  // a warp's rows
    const int warp = threadIdx.x >> 5;
    const int l = threadIdx.x & 31;
    const int task = warp >> 1;
    const int hw = warp & 1;
    const int h = blockIdx.x * KDA16_TASKS + task;
    const int s = blockIdx.y;
    if (win != nullptr) lanes = min(lanes, static_cast<int>(win[2]));
    if (lanes <= 0 || h >= H) return;
    const int r = gdn_slot_lane<KDA_CH>(s, lanes, qo_indptr, begin_at, win);
    const GdnLane lane = gdn_lane(r, qo_indptr, slot_ids, commit_len, begin_at, win);
    if (lane.T <= 0 || lane.slot < 0) return;
    const int j = s - (lane.t0 / KDA_CH + r);
    if (j < 0 || j * KDA_CH >= lane.T) return;
    const int ta = lane.t0 + j * KDA_CH;
    const int L = min(KDA_CH, lane.t0 + lane.T - ta);
    const int row0 = win != nullptr ? static_cast<int>(win[1]) : 0;
    const long long width = static_cast<long long>(H) * D;
    const int c = 4 * l;
    const int i0 = hw * R;
    // The task's two warps meet at their own barrier.
    auto pair = [&]() { asm volatile("bar.sync %0, 64;" ::"r"(1 + task) : "memory"); };

    extern __shared__ __align__(16) unsigned char kda16_smem[];
    unsigned char* mine = kda16_smem + task * KDA16_TASK_SMEM;
    bf16* kr = reinterpret_cast<bf16*>(mine);
    bf16* kc = kr + KDA_CH * GDN_LD;
    bf16* qr = kc + KDA_CH * GDN_LD;
    float* As = reinterpret_cast<float*>(qr + KDA_CH * GDN_LD);
    float* xch = As + KDA_CH * (KDA_CH + 1);

    auto store4 = [&](bf16* at, float a, float b, float cc, float d) {
        *reinterpret_cast<uint2*>(at) = make_uint2(pack_bf16x2(a, b), pack_bf16x2(cc, d));
    };
    auto row_at = [&](int i) { return static_cast<long long>(ta + (i < L ? i : 0) + row0); };
    // The task's rows land in shared memory at once, each in the tile it
    // turns into: the gates in the scaled keys', the keys in their twins',
    // the queries in the scaled queries'. A lane reads back only its own
    // row's channels, so each is overwritten in place.
    #pragma unroll
    for (int e = hw * 32 + l; e < 3 * KDA_CH * (D / 8); e += 64) {
        const int which = e / (KDA_CH * (D / 8));
        const int i = (e / (D / 8)) % KDA_CH;
        const int cc = (e % (D / 8)) * 8;
        const long long rw = row_at(i);
        const bf16* src = which == 0 ? raw_g + (rw * H + h) * D + cc
                                     : mixed + rw * 3 * width + (which == 1 ? width : 0) + h * D + cc;
        gdn_cp16((which == 0 ? kr : which == 1 ? kc : qr) + i * GDN_LD + cc, src, i < L);
    }
    gdn_cp_commit();
    auto unpack4 = [&](const bf16* tile, int i, float (&x)[4]) {
        const uint2 v = *reinterpret_cast<const uint2*>(tile + i * GDN_LD + c);
        const float2 a = gdn_unpack(v.x), b = gdn_unpack(v.y);
        x[0] = a.x; x[1] = a.y; x[2] = b.x; x[3] = b.y;
    };
    const float beta = l < L ? sigmoidf(__bfloat162float(raw_beta[row_at(l) * H + h])) : 0.f;
    gdn_cp_wait_all();
    pair();

    // The gates and their running sum `G` down the warp's rows, in log2
    // units, a row past the chunk no decay; the second warp's offset by the
    // first's total.
    constexpr float kLog2e = 1.4426950408889634f;
    float G[R][4];
    {
        const float a = __expf(A_log[h]);
        float bias[4];
        #pragma unroll
        for (int u = 0; u < 4; ++u) bias[u] = dt_bias[h * D + c + u];
        // `lower_bound sigmoid(a x)` as `lb + lb tanh(a x / 2)` (`lb` half
        // the bound), else `-a softplus(x)`.
        const float lb = 0.5f * kLog2e * lower_bound;
        const float ha = 0.5f * a;
        float run[4] = {0.f, 0.f, 0.f, 0.f};
        #pragma unroll
        for (int ii = 0; ii < R; ++ii) {
            float g[4];
            unpack4(kr, i0 + ii, g);
            #pragma unroll
            for (int u = 0; u < 4; ++u) {
                const float x = g[u] + bias[u];
                const float gate = lower_bound < 0.f
                    ? __fmaf_rn(lb, kda_tanh(ha * x), lb)
                    : -a * (x > 20.f ? x * kLog2e : kda_lg2(1.f + kda_ex2(x * kLog2e)));
                run[u] += i0 + ii < L ? gate : 0.f;
                G[ii][u] = run[u];
            }
        }
        if (hw == 0) {
            *reinterpret_cast<float4*>(xch + c) = make_float4(run[0], run[1], run[2], run[3]);
            *reinterpret_cast<float4*>(xch + 128 + c) = make_float4(G[0][0], G[0][1], G[0][2], G[0][3]);
        }
        pair();
        if (hw == 1) {
            const float4 off = *reinterpret_cast<const float4*>(xch + c);
            const float o[4] = {off.x, off.y, off.z, off.w};
            #pragma unroll
            for (int ii = 0; ii < R; ++ii) {
                #pragma unroll
                for (int u = 0; u < 4; ++u) G[ii][u] += o[u];
            }
            *reinterpret_cast<float4*>(xch + 256 + c) =
                make_float4(G[R - 1][0], G[R - 1][1], G[R - 1][2], G[R - 1][3]);
        }
        pair();
    }
    // From here `exp(G)`: each factor below a product of it, its inverse,
    // and the chunk's first and last rows'.
    float e_0[4], einv_0[4], e_l[4];
    {
        const float4 g0 = *reinterpret_cast<const float4*>(xch + 128 + c);
        const float4 gl = *reinterpret_cast<const float4*>(xch + 256 + c);
        const float a0[4] = {g0.x, g0.y, g0.z, g0.w};
        const float al[4] = {gl.x, gl.y, gl.z, gl.w};
        #pragma unroll
        for (int u = 0; u < 4; ++u) {
            e_0[u] = kda_ex2(a0[u]);
            einv_0[u] = kda_ex2(-a0[u]);
            e_l[u] = kda_ex2(al[u]);
        }
    }
    if (hw == 1) {
        *reinterpret_cast<float4*>(gl_out + (static_cast<long long>(s) * H + h) * D + c) =
            make_float4(e_l[0], e_l[1], e_l[2], e_l[3]);
    }

    // A row's inverse length (a warp's channels summed in f32).
    auto inv_len = [&](const float (&x)[4], float scale) {
        return rsqrtf(warp_sum(x[0] * x[0] + x[1] * x[1] + x[2] * x[2] + x[3] * x[3]) + eps) * scale;
    };
    // The keys and queries: their split-scaled twins for `A` and `P`, the
    // scan's. Every row's lengths first, so their shuffles overlap.
    const float q_scale = rsqrtf(static_cast<float>(D));
    float kinv[R], qinv[R];
    #pragma unroll
    for (int ii = 0; ii < R; ++ii) {
        float k[4], q[4];
        unpack4(kc, i0 + ii, k);
        unpack4(qr, i0 + ii, q);
        kinv[ii] = inv_len(k, 1.f);
        qinv[ii] = inv_len(q, q_scale);
    }
    #pragma unroll
    for (int ii = 0; ii < R; ++ii) {
        const int i = i0 + ii;
        float k[4], q[4];
        unpack4(kc, i, k);
        unpack4(qr, i, q);
        float ke[4], kd[4], kz[4], kl[4], qe[4], qd[4];
        #pragma unroll
        for (int u = 0; u < 4; u += 2) {
            const float2 kk = kda_mul2(make_float2(k[u], k[u + 1]), make_float2(kinv[ii], kinv[ii]));
            const float2 up =
                kda_mul2(kk, make_float2(kda_ex2(-G[ii][u]), kda_ex2(-G[ii][u + 1])));
            const float2 e = make_float2(kda_ex2(G[ii][u]), kda_ex2(G[ii][u + 1]));
            const float2 ke2 = kda_mul2(kk, e);
            const float2 kd2 = kda_mul2(ke2, make_float2(einv_0[u], einv_0[u + 1]));
            const float2 kz2 = kda_mul2(up, make_float2(e_0[u], e_0[u + 1]));
            const float2 kl2 = kda_mul2(up, make_float2(e_l[u], e_l[u + 1]));
            const float2 qe2 =
                kda_mul2(kda_mul2(make_float2(q[u], q[u + 1]), make_float2(qinv[ii], qinv[ii])), e);
            const float2 qd2 = kda_mul2(qe2, make_float2(einv_0[u], einv_0[u + 1]));
            ke[u] = ke2.x; ke[u + 1] = ke2.y;
            kd[u] = kd2.x; kd[u + 1] = kd2.y;
            kz[u] = kz2.x; kz[u + 1] = kz2.y;
            kl[u] = kl2.x; kl[u + 1] = kl2.y;
            qe[u] = qe2.x; qe[u + 1] = qe2.y;
            qd[u] = qd2.x; qd[u + 1] = qd2.y;
        }
        store4(kr + i * GDN_LD + c, kd[0], kd[1], kd[2], kd[3]);
        store4(kc + i * GDN_LD + c, kz[0], kz[1], kz[2], kz[3]);
        store4(qr + i * GDN_LD + c, qd[0], qd[1], qd[2], qd[3]);
        if (i < L) {
            const long long at = (static_cast<long long>(ta + i) * H + h) * D + c;
            store4(kg_out + at, kl[0], kl[1], kl[2], kl[3]);
            store4(ke_out + at, ke[0], ke[1], ke[2], ke[3]);
            store4(qg_out + at, qe[0], qe[1], qe[2], qe[3]);
        }
    }
    pair();

    // `k k^T` and `q k^T` on the tensor cores, sharing the key operand: a
    // warp an eight-wide column tile.
    const int g = l >> 2;
    const int t = l & 3;
    {
        float acc[4] = {};
        float qkc[4] = {};
        #pragma unroll
        for (int kk = 0; kk < D / 16; ++kk) {
            unsigned a[4], aq[4], b[2];
            ldmatrix_x4(a, gdn_at_a(kr, GDN_LD, 0, kk * 16, l));
            ldmatrix_x4(aq, gdn_at_a(qr, GDN_LD, 0, kk * 16, l));
            ld_b(b, kc + hw * 8 * GDN_LD + kk * 16, GDN_LD, l);
            gdn_mma(acc, a, b[0], b[1]);
            gdn_mma(qkc, aq, b[0], b[1]);
        }
        bf16* pd = p_out + (static_cast<long long>(s) * H + h) * KDA_CH * KDA_CH;
        #pragma unroll
        for (int half = 0; half < 2; ++half) {
            const int i = g + 8 * half;
            const int jj = hw * 8 + 2 * t;
            const float bi = __shfl_sync(0xffffffffu, beta, i);
            const bool live = i < L;
            const float p0 = live && jj <= i ? qkc[2 * half] : 0.f;
            const float p1 = live && jj + 1 <= i ? qkc[2 * half + 1] : 0.f;
            *reinterpret_cast<unsigned*>(pd + i * KDA_CH + jj) = pack_bf16x2(p0, p1);
            As[i * (KDA_CH + 1) + jj] = live && jj < i ? acc[2 * half] * bi : 0.f;
            As[i * (KDA_CH + 1) + jj + 1] = live && jj + 1 < i ? acc[2 * half + 1] * bi : 0.f;
        }
    }
    pair();

    // `T = (I + A)^-1` by forward substitution, a column a lane of the first
    // warp, out as `T diag(beta)` in bf16.
    if (hw == 0) {
        float tcol[KDA_CH];
        #pragma unroll
        for (int rr = 0; rr < KDA_CH; ++rr) {
            float sum = 0.f;
            #pragma unroll
            for (int m = 0; m < rr; ++m) sum = __fmaf_rn(As[rr * (KDA_CH + 1) + m], tcol[m], sum);
            tcol[rr] = rr == l ? 1.f : (rr < l ? 0.f : -sum);
        }
        if (l < KDA_CH) {
            bf16* td = tb_out + (static_cast<long long>(s) * H + h) * KDA_CH * KDA_CH;
            #pragma unroll
            for (int rr = 0; rr < KDA_CH; ++rr) td[rr * KDA_CH + l] = bf16(tcol[rr] * beta);
        }
    }
}

// The sequential half: a block per (request, head, `64 NB` value columns)
// walks the request's chunks; a warp carries `16 NB` value columns of the
// state over all its key rows, held transposed, `[values][keys]`, in f32
// fragments. Transposed, the state is its own A operand for the chunk's
// `v_new^T = (v^T - S^T k_e^T) (T diag(beta))^T` and
// `out^T = S^T q_g^T + v_new^T P^T`, and
// `v_new^T` is the A operand of the fold `S^T = S^T diag(exp(G_L)) +
// v_new^T k_g`, so a warp waits on the others only to refill a stage; its
// `NB` sixteen-column blocks share each B operand. The operands stream in
// two chunks ahead, each stage with a barrier its loads complete and one
// its four warps release. Two blocks a (request, head), each a warp sixteen
// columns, while they fit two a multiprocessor (`wide_at` the
// multiprocessors); past that a block a (request, head), each warp 32.
template <int NB>
struct Kda16Scan {
    static constexpr int BV = 64 * NB;
    static constexpr int LDV = BV + 8;
    static constexpr int STAGE =
        3 * KDA_CH * GDN_LD * 2 + KDA_CH * LDV * 2 + 2 * KDA_CH * KDA16_LDT * 2 + 128 * 4;
};

__device__ __forceinline__ void kda_bar_init(unsigned long long* bar, unsigned count) {
    asm volatile("mbarrier.init.shared.b64 [%0], %1;" ::"r"(static_cast<unsigned>(__cvta_generic_to_shared(bar))),
                 "r"(count));
}
__device__ __forceinline__ void kda_bar_arrive(unsigned long long* bar) {
    asm volatile("mbarrier.arrive.shared.b64 _, [%0];" ::"r"(static_cast<unsigned>(__cvta_generic_to_shared(bar)))
                 : "memory");
}
// One arrival once this thread's `cp.async`s land.
__device__ __forceinline__ void kda_bar_arrive_cp(unsigned long long* bar) {
    asm volatile("cp.async.mbarrier.arrive.noinc.shared.b64 [%0];" ::"r"(
                     static_cast<unsigned>(__cvta_generic_to_shared(bar)))
                 : "memory");
}
__device__ __forceinline__ void kda_bar_wait(unsigned long long* bar, unsigned parity) {
    const unsigned at = static_cast<unsigned>(__cvta_generic_to_shared(bar));
    unsigned done = 0;
    while (!done) {
        asm volatile(
            "{\n\t.reg .pred p;\n\t"
            "mbarrier.test_wait.parity.shared.b64 p, [%1], %2;\n\t"
            "selp.u32 %0, 1, 0, p;\n\t}"
            : "=r"(done)
            : "r"(at), "r"(parity)
            : "memory");
    }
}

template <int NB>
__device__ __forceinline__ void kda16_walk(
    const float* __restrict__ decay,
    const bf16* __restrict__ mixed,
    const bf16* __restrict__ qg,
    const bf16* __restrict__ kg,
    const bf16* __restrict__ ke,
    const bf16* __restrict__ tb,
    const bf16* __restrict__ pm,
    float* __restrict__ state_base,
    const int* __restrict__ slot_ids,
    const u32* __restrict__ qo_indptr,
    long long slot_stride_elems,
    const int* __restrict__ commit_len,
    const int* __restrict__ begin_at,
    bool write_state,
    const u8* __restrict__ write_state_mask,
    float* __restrict__ out,
    int H, int r, int h, int vt,
    const u32* __restrict__ win)
{
    constexpr int BK = 128;
    constexpr int V = 128;
    constexpr int BV = Kda16Scan<NB>::BV;
    constexpr int LDV = Kda16Scan<NB>::LDV;
    constexpr int STAGE = Kda16Scan<NB>::STAGE;
    extern __shared__ __align__(16) unsigned char kda16_smem[];
    auto stage = [&](int b, bf16*& sq, bf16*& sw, bf16*& sk, bf16*& su, bf16*& sp, bf16*& st, float*& sg) {
        unsigned char* at = kda16_smem + b * STAGE;
        sq = reinterpret_cast<bf16*>(at);
        sw = sq + KDA_CH * GDN_LD;
        sk = sw + KDA_CH * GDN_LD;
        su = sk + KDA_CH * GDN_LD;
        sp = su + KDA_CH * LDV;
        st = sp + KDA_CH * KDA16_LDT;
        sg = reinterpret_cast<float*>(st + KDA_CH * KDA16_LDT);
    };

    const GdnLane lane = gdn_lane(r, qo_indptr, slot_ids, commit_len, begin_at, win);
    if (lane.T <= 0 || lane.slot < 0) return;
    const int row0 = win != nullptr ? static_cast<int>(win[1]) : 0;
    const int base = lane.t0 / KDA_CH + r;
    const int tid = threadIdx.x;
    const int l = tid & 31;
    const int g = l >> 2;
    const int t = l & 3;
    // The warp's value columns, within the block's and overall.
    const int cw = (tid >> 5) * 16 * NB;
    const int c0 = vt * BV + cw;
    float* state = state_base + static_cast<long long>(lane.slot) * slot_stride_elems
                   + static_cast<long long>(h) * BK * V;

    // The state fragments: per sixteen value rows, sixteen eight-wide key
    // tiles.
    float hacc[NB][BK / 8][4];
    #pragma unroll
    for (int nb = 0; nb < NB; ++nb) {
        #pragma unroll
        for (int nt = 0; nt < BK / 8; ++nt) {
            #pragma unroll
            for (int half = 0; half < 2; ++half) {
                const float2 s2 = *reinterpret_cast<const float2*>(
                    state + static_cast<long long>(c0 + 16 * nb + g + 8 * half) * BK + nt * 8 + 2 * t);
                hacc[nb][nt][2 * half] = s2.x;
                hacc[nb][nt][2 * half + 1] = s2.y;
            }
        }
    }

    // Per stage: its loads landed (each thread's arrival), its warps are
    // done with it (each warp's).
    unsigned long long* full = reinterpret_cast<unsigned long long*>(kda16_smem + 3 * STAGE);
    unsigned long long* empty = full + 3;
    if (tid < 3) {
        kda_bar_init(full + tid, 128);
        kda_bar_init(empty + tid, 4);
    }
    __syncthreads();

    const int nchunks = (lane.T + KDA_CH - 1) / KDA_CH;
    auto load = [&](int jc) {
        if (jc < nchunks) {
            if (jc >= 3) kda_bar_wait(empty + jc % 3, (jc / 3 - 1) & 1);
            bf16 *sq, *sw, *sk, *su, *sp, *st;
            float* sg;
            stage(jc % 3, sq, sw, sk, su, sp, st, sg);
            const int ta = lane.t0 + jc * KDA_CH;
            const int L = min(KDA_CH, lane.t0 + lane.T - ta);
            #pragma unroll
            for (int e = tid; e < KDA_CH * (BK / 8); e += 128) {
                const int i = e / (BK / 8);
                const int c = (e % (BK / 8)) * 8;
                const bool in = i < L;
                const long long row = (static_cast<long long>(ta + (in ? i : 0)) * H + h) * BK + c;
                gdn_cp16(sq + i * GDN_LD + c, qg + row, in);
                gdn_cp16(sw + i * GDN_LD + c, ke + row, in);
                gdn_cp16(sk + i * GDN_LD + c, kg + row, in);
            }
            #pragma unroll
            for (int e = tid; e < KDA_CH * (BV / 8); e += 128) {
                const int i = e / (BV / 8);
                const int c = (e % (BV / 8)) * 8;
                const bool in = i < L;
                gdn_cp16(su + i * LDV + c,
                         mixed + static_cast<long long>(ta + (in ? i : 0) + row0) * 3 * H * V + 2 * H * V + h * V +
                             vt * BV + c,
                         in);
            }
            if (tid < KDA_CH * KDA_CH / 4) {
                const int i = (tid / 2) % KDA_CH;
                const int c = (tid % 2) * 8;
                const long long at = (static_cast<long long>(base + jc) * H + h) * KDA_CH * KDA_CH + i * KDA_CH + c;
                if (tid < KDA_CH * KDA_CH / 8) gdn_cp16(sp + i * KDA16_LDT + c, pm + at, true);
                else gdn_cp16(st + i * KDA16_LDT + c, tb + at, true);
            }
            if (tid < BK / 4) {
                gdn_cp16(sg + tid * 4, decay + (static_cast<long long>(base + jc) * H + h) * BK + tid * 4, true);
            }
            kda_bar_arrive_cp(full + jc % 3);
        }
    };
    // Lane `l`'s row address for an x4 `ldmatrix` over a `[token][k]` tile:
    // as is, the B operand of two eight-token tiles; transposed, a `[k][token]`
    // tile's C fragments.
    auto at_tok = [&](const bf16* s, int ld, int k0) {
        return s + ((l & 7) + ((l >> 4) << 3)) * ld + k0 + (((l >> 3) & 1) << 3);
    };

    load(0);
    load(1);
    for (int jc = 0; jc < nchunks; ++jc) {
        kda_bar_wait(full + jc % 3, (jc / 3) & 1);
        bf16 *sq, *sw, *sk, *su, *sp, *st;
        float* sg;
        stage(jc % 3, sq, sw, sk, su, sp, st, sg);
        const int ta = lane.t0 + jc * KDA_CH;
        const int L = min(KDA_CH, lane.t0 + lane.T - ta);

        // `S^T q_g^T` and `S^T k_e^T`, two eight-token tiles each.
        float oh[NB][2][4] = {};
        float wh[NB][2][4] = {};
        #pragma unroll
        for (int kk = 0; kk < BK / 16; ++kk) {
            unsigned bq[4], bw[4];
            ldmatrix_x4(bq, at_tok(sq, GDN_LD, kk * 16));
            ldmatrix_x4(bw, at_tok(sw, GDN_LD, kk * 16));
            #pragma unroll
            for (int nb = 0; nb < NB; ++nb) {
                const unsigned a[4] = {pack_bf16x2(hacc[nb][2 * kk][0], hacc[nb][2 * kk][1]),
                                       pack_bf16x2(hacc[nb][2 * kk][2], hacc[nb][2 * kk][3]),
                                       pack_bf16x2(hacc[nb][2 * kk + 1][0], hacc[nb][2 * kk + 1][1]),
                                       pack_bf16x2(hacc[nb][2 * kk + 1][2], hacc[nb][2 * kk + 1][3])};
                gdn_mma(oh[nb][0], a, bq[0], bq[1]);
                gdn_mma(oh[nb][1], a, bq[2], bq[3]);
                gdn_mma(wh[nb][0], a, bw[0], bw[1]);
                gdn_mma(wh[nb][1], a, bw[2], bw[3]);
            }
        }

        // `v_new^T = (v^T - S^T k_e^T) (T diag(beta))^T`, in A-operand order.
        unsigned vn[NB][4];
        {
            unsigned bt[4];
            ldmatrix_x4(bt, at_tok(st, KDA16_LDT, 0));
            #pragma unroll
            for (int nb = 0; nb < NB; ++nb) {
                unsigned x[4];
                ldmatrix_x4_trans(x, at_tok(su, LDV, cw + 16 * nb));
                #pragma unroll
                for (int nt = 0; nt < 2; ++nt) {
                    #pragma unroll
                    for (int half = 0; half < 2; ++half) {
                        const float2 vv = gdn_unpack(x[2 * nt + half]);
                        x[2 * nt + half] = pack_bf16x2(vv.x - wh[nb][nt][2 * half], vv.y - wh[nb][nt][2 * half + 1]);
                    }
                }
                float o[2][4] = {};
                gdn_mma(o[0], x, bt[0], bt[1]);
                gdn_mma(o[1], x, bt[2], bt[3]);
                #pragma unroll
                for (int nt = 0; nt < 2; ++nt) {
                    #pragma unroll
                    for (int half = 0; half < 2; ++half) {
                        vn[nb][2 * nt + half] = pack_bf16x2(o[nt][2 * half], o[nt][2 * half + 1]);
                    }
                }
            }
        }

        // `+ v_new^T P^T`, out.
        {
            unsigned bp[4];
            ldmatrix_x4(bp, at_tok(sp, KDA16_LDT, 0));
            #pragma unroll
            for (int nb = 0; nb < NB; ++nb) {
                gdn_mma(oh[nb][0], vn[nb], bp[0], bp[1]);
                gdn_mma(oh[nb][1], vn[nb], bp[2], bp[3]);
                #pragma unroll
                for (int nt = 0; nt < 2; ++nt) {
                    #pragma unroll
                    for (int e = 0; e < 4; ++e) {
                        const int i = nt * 8 + 2 * t + (e & 1);
                        if (i < L) {
                            out[(static_cast<long long>(ta + i + row0) * H + h) * V + c0 + 16 * nb + g + 8 * (e >> 1)] =
                                oh[nb][nt][e];
                        }
                    }
                }
            }
        }

        // The fold: `S^T = S^T diag(exp(G_L)) + v_new^T k_g`.
        #pragma unroll
        for (int np = 0; np < BK / 16; ++np) {
            unsigned b[4];
            ldmatrix_x4_trans(b, gdn_at_b(sk, GDN_LD, 0, np * 16, l));
            #pragma unroll
            for (int j = 0; j < 2; ++j) {
                const float2 d = *reinterpret_cast<const float2*>(sg + (2 * np + j) * 8 + 2 * t);
                #pragma unroll
                for (int nb = 0; nb < NB; ++nb) {
                    float* s = hacc[nb][2 * np + j];
                    s[0] *= d.x; s[1] *= d.y; s[2] *= d.x; s[3] *= d.y;
                    gdn_mma(hacc[nb][2 * np + j], vn[nb], b[2 * j], b[2 * j + 1]);
                }
            }
        }
        __syncwarp();
        if (l == 0) kda_bar_arrive(empty + jc % 3);
        load(jc + 2);
    }

    if (write_state && row_persists(write_state_mask, lane.rl)) {
        #pragma unroll
        for (int nb = 0; nb < NB; ++nb) {
            #pragma unroll
            for (int nt = 0; nt < BK / 8; ++nt) {
                #pragma unroll
                for (int half = 0; half < 2; ++half) {
                    *reinterpret_cast<float2*>(
                        state + static_cast<long long>(c0 + 16 * nb + g + 8 * half) * BK + nt * 8 + 2 * t) =
                        make_float2(hacc[nb][nt][2 * half], hacc[nb][nt][2 * half + 1]);
                }
            }
        }
    }
}

__global__ void __launch_bounds__(128) kda_chunk_scan(
    const float* __restrict__ decay,
    const bf16* __restrict__ mixed,
    const bf16* __restrict__ qg,
    const bf16* __restrict__ kg,
    const bf16* __restrict__ ke,
    const bf16* __restrict__ tb,
    const bf16* __restrict__ pm,
    float* __restrict__ state_base,
    const int* __restrict__ slot_ids,
    const u32* __restrict__ qo_indptr,
    long long slot_stride_elems,
    const int* __restrict__ commit_len,
    const int* __restrict__ begin_at,
    bool write_state,
    const u8* __restrict__ write_state_mask,
    float* __restrict__ out,
    int lanes, int H, int wide_at,
    const u32* __restrict__ win)
{
    if (win != nullptr) lanes = min(lanes, static_cast<int>(win[2]));
    const int b = static_cast<int>(blockIdx.x);
    if (lanes * H > wide_at) {
        if (b < lanes * H) {
            kda16_walk<2>(decay, mixed, qg, kg, ke, tb, pm, state_base, slot_ids, qo_indptr, slot_stride_elems,
                          commit_len, begin_at, write_state, write_state_mask, out, H, b / H, b % H, 0, win);
        }
    } else {
        const int vt = b & 1;
        const int h = (b >> 1) % H;
        const int r = b / (2 * H);
        if (r >= lanes) return;
        kda16_walk<1>(decay, mixed, qg, kg, ke, tb, pm, state_base, slot_ids, qo_indptr, slot_stride_elems,
                      commit_len, begin_at, write_state, write_state_mask, out, H, r, h, vt, win);
    }
}

}
