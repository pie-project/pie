#include "ssm/copy.cuh"
#include "prim/state_rows.cuh"
#include "ssm/carry.cuh"

namespace pie::ssm {

constexpr int gqa_smem_bv = 128;

template <typename StateT, bool KLast>
__global__ void gdn_step(
    const float* __restrict__ q_norm_kh,
    const float* __restrict__ k_norm_kh,
    const float* __restrict__ v,
    const float* __restrict__ g_log,
    const float* __restrict__ beta,
    StateRows rows,
    float*       __restrict__ out,
    int K_h, int V_h, int K_d, int V_d,
    bool write_state,
    const u8* __restrict__ write_state_mask,
    const u32* __restrict__ win)
{
    const int r = blockIdx.x;

    if (win != nullptr && r >= static_cast<int>(win[0])) return;

    const int r_row = win != nullptr ? r + static_cast<int>(win[1]) : r;
    const int h = blockIdx.y;
    const int repeat = V_h / K_h;
    const int h_k = h / repeat;
    const int rl = win != nullptr ? r + static_cast<int>(win[3]) : r;
    const i32* rid = rows.row(rl);
    if (rid[0] < 0) return;
    const bool fold = folds(write_state, write_state_mask, rl);

    const long long qh = ((long long)r * K_h + h_k) * K_d;
    const long long vh = (long long)r * V_h + h;
    const float* q_h = q_norm_kh + qh;
    const float* k_h = k_norm_kh + qh;
    const float* v_h = v + vh * V_d;
    const float  g_h = __expf(g_log[vh]);
    const float  beta_h = beta[vh];

    StateT* state = rows.atom<StateT>(rid, h, (long long)K_d * V_d);
    float* out_bh = out + ((long long)r_row * V_h + h) * V_d;

    extern __shared__ float smem[];
    float* sq = smem;
    float* sk = smem + K_d;

    for (int i = threadIdx.x; i < K_d; i += blockDim.x) {
        sq[i] = q_h[i];
        sk[i] = k_h[i];
    }
    __syncthreads();

    for (int v_idx = threadIdx.x; v_idx < V_d; v_idx += blockDim.x) {
        float kv_mem = 0.f;
        for (int k_idx = 0; k_idx < K_d; ++k_idx) {
            const long long off =
                state_offset<KLast>(k_idx, v_idx, K_d, V_d);
            const float s = state_load(state + off) * g_h;
            if (fold) state_store(state + off, s);
            kv_mem += s * sk[k_idx];
        }

        const float v_t = v_h[v_idx];
        const float delta = (v_t - kv_mem) * beta_h;

        float out_v = 0.f;
        for (int k_idx = 0; k_idx < K_d; ++k_idx) {
            const long long off =
                state_offset<KLast>(k_idx, v_idx, K_d, V_d);
            const float decayed = fold
                ? state_load(state + off)
                : state_round<StateT>(state_load(state + off) * g_h);
            const float s = decayed + sk[k_idx] * delta;
            if (fold) state_store(state + off, s);
            out_v += s * sq[k_idx];
        }
        out_bh[v_idx] = out_v;
    }
}

template <int BV>
__global__ void gdn_step_smem(
    const float* __restrict__ q_norm_kh,
    const float* __restrict__ k_norm_kh,
    const float* __restrict__ v,
    const float* __restrict__ g_log,
    const float* __restrict__ beta,
    StateRows rows,
    float*       __restrict__ out,
    int K_h, int V_h, int K_d, int V_d,
    bool write_state,
    const u8* __restrict__ write_state_mask,
    const u32* __restrict__ win)
{
    const int vt = blockIdx.x;
    const int r  = blockIdx.y;

    if (win != nullptr && r >= static_cast<int>(win[0])) return;

    const int r_row = win != nullptr ? r + static_cast<int>(win[1]) : r;
    const int h  = blockIdx.z;
    const int v_idx = vt * BV + threadIdx.x;
    if (v_idx >= V_d) return;
    const int repeat = V_h / K_h;
    const int h_k = h / repeat;
    const int rl = win != nullptr ? r + static_cast<int>(win[3]) : r;
    const i32* rid = rows.row(rl);
    if (rid[0] < 0) return;
    const bool fold = folds(write_state, write_state_mask, rl);

    const long long qh = ((long long)r * K_h + h_k) * K_d;
    const long long vh = (long long)r * V_h + h;
    const float* v_h = v + vh * V_d;
    const float g_h = __expf(g_log[vh]);
    const float beta_h = beta[vh];

    __nv_bfloat16* state = rows.atom<__nv_bfloat16>(rid, h, (long long)K_d * V_d);
    float* out_bh = out + ((long long)r_row * V_h + h) * V_d;

    extern __shared__ __nv_bfloat16 smem_smem_step[];
    __nv_bfloat16* s_state = smem_smem_step;
    float* sq = (float*)(smem_smem_step + K_d * BV);
    float* sk = sq + K_d;

    const bool vec_tile =
        (BV == V_d) && ((V_d & 7) == 0) &&
        ((reinterpret_cast<usize>(state) & 15) == 0);
    const int n_vec = (K_d * V_d) >> 3;
    if (vec_tile) {
        const uint4* __restrict__ src = reinterpret_cast<const uint4*>(state);
        uint4* __restrict__ dst = reinterpret_cast<uint4*>(s_state);
        for (int i = threadIdx.x; i < n_vec; i += BV) dst[i] = src[i];
    } else {
        constexpr int kStageTile = 16;
        __nv_bfloat16 staged[kStageTile];
        int k = 0;
        for (; k + kStageTile <= K_d; k += kStageTile) {
            #pragma unroll
            for (int u = 0; u < kStageTile; ++u) {
                staged[u] = state[(long long)(k + u) * V_d + v_idx];
            }
            #pragma unroll
            for (int u = 0; u < kStageTile; ++u) {
                s_state[(k + u) * BV + threadIdx.x] = staged[u];
            }
        }
        for (; k < K_d; ++k) {
            s_state[k * BV + threadIdx.x] =
                state[(long long)k * V_d + v_idx];
        }
    }
    const float* q_h = q_norm_kh + qh;
    const float* k_h = k_norm_kh + qh;
    for (int i = threadIdx.x; i < K_d; i += blockDim.x) {
        sq[i] = q_h[i];
        sk[i] = k_h[i];
    }
    __syncthreads();

    float kv_mem = 0.f;
    for (int k = 0; k < K_d; ++k) {
        float s = __bfloat162float(s_state[k * BV + threadIdx.x]) * g_h;
        kv_mem += s * sk[k];
    }
    const float delta = (v_h[v_idx] - kv_mem) * beta_h;

    float out_v = 0.f;
    for (int k = 0; k < K_d; ++k) {
        const float sg = __bfloat162float(__float2bfloat16(
            __bfloat162float(s_state[k * BV + threadIdx.x]) * g_h));
        float s = sg + sk[k] * delta;
        out_v += s * sq[k];

        if (vec_tile) {
            s_state[k * BV + threadIdx.x] = __float2bfloat16(s);
        } else if (fold) {
            state[(long long)k * V_d + v_idx] = __float2bfloat16(s);
        }
    }
    out_bh[v_idx] = out_v;
    if (vec_tile && fold) {
        __syncthreads();
        const uint4* __restrict__ src = reinterpret_cast<const uint4*>(s_state);
        uint4* __restrict__ dst = reinterpret_cast<uint4*>(state);
        for (int i = threadIdx.x; i < n_vec; i += BV) dst[i] = src[i];
    }
}

// One decode step of the gated delta rule, streamed: a warp owns a 16-wide
// value tile of one head and keeps its `K_d x 16` state tile in registers
// (lane `l` holds columns `(l & 1) * 8` of key rows `l / 2 + 16 m`), so the
// norms, column sums and update stay in the warp. Persistent warps prefetch
// the next tile's operands into their other shared stage. The host keeps the
// planes 16-byte aligned and the key width a multiple of 16 up to 128.
constexpr int GDN_STREAM_STAGE = 4096 + 512 + 32 + 32;

template <int W>
__global__ void __launch_bounds__(W * 32) gdn_step_fused(
    const bf16* __restrict__ qkv_post,
    const float* __restrict__ gates,
    StateRows rows,
    float* __restrict__ out,
    int K_h, int V_h, int K_d, int V_d, int conv_dim,
    float q_scale,
    bool write_state,
    const u8* __restrict__ write_state_mask,
    int tiles,
    const u32* __restrict__ win)
{
    constexpr int MR = 8;
    constexpr int VEC = 8;
    extern __shared__ __align__(16) unsigned char gdn_ring[];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    unsigned char* ring = gdn_ring + warp * 2 * GDN_STREAM_STAGE;
    const int i0 = lane >> 1;
    const int vts = V_d / 16;
    const int group = V_h / K_h;
    const int K_dim = K_h * K_d;
    const int live_rows = win != nullptr ? static_cast<int>(win[0]) : 0x7fffffff;

    struct Tile {
        bool live;
        bool fold;
        int r_row;
        int h;
        int j0;
        __nv_bfloat16* cell;
    };
    auto plan = [&](int tile) {
        Tile tl;
        tl.live = false;
        if (tile >= tiles) return tl;
        const int vt = tile % vts;
        const int rest = tile / vts;
        tl.h = rest % V_h;
        const int r = rest / V_h;
        if (r >= live_rows) return tl;
        tl.r_row = win != nullptr ? r + static_cast<int>(win[1]) : r;
        const int rl = win != nullptr ? r + static_cast<int>(win[3]) : r;
        const i32* rid = rows.row(rl);
        if (rid[0] < 0) return tl;
        tl.live = true;
        tl.fold = folds(write_state, write_state_mask, rl);
        tl.j0 = vt * 16 + (lane & 1) * VEC;
        tl.cell = rows.atom<__nv_bfloat16>(rid, tl.h, (long long)K_d * V_d) + tl.j0;
        return tl;
    };
    auto issue = [&](const Tile& tl, int stage) {
        if (!tl.live) return;
        unsigned char* st = ring + stage * GDN_STREAM_STAGE;
        #pragma unroll
        for (int m = 0; m < MR; ++m) {
            if (16 * m < K_d) {
                gdn_cp16(st + (m * 32 + lane) * 16, tl.cell + (long long)(i0 + 16 * m) * V_d, true);
            }
        }
        const bf16* row = qkv_post + (long long)tl.r_row * conv_dim;
        const int h_k = tl.h / group;
        const int c = lane & 15;
        if (c * 8 < K_d) {
            const bf16* src = row + (lane < 16 ? 0 : K_dim) + (long long)h_k * K_d + c * 8;
            gdn_cp16(st + 4096 + (lane >> 4) * 256 + c * 16, src, true);
        }
        if (lane < 2) {
            gdn_cp16(st + 4608 + lane * 16, row + 2 * K_dim + (long long)tl.h * V_d + tl.j0, true);
            gdn_cp4(st + 4640 + lane * 4, gates + (long long)tl.r_row * 2 * V_h + lane * V_h + tl.h, true);
        }
    };

    const int stride = gridDim.x * W;
    int t = blockIdx.x * W + warp;
    Tile cur = plan(t);
    issue(cur, 0);
    gdn_cp_commit();
    for (int it = 0; t < tiles; t += stride, ++it) {
        const Tile nxt = plan(t + stride);
        issue(nxt, (it + 1) & 1);
        gdn_cp_commit();
        asm volatile("cp.async.wait_group 1;\n" ::: "memory");
        __syncwarp();
        if (cur.live) {
            const unsigned char* st = ring + (it & 1) * GDN_STREAM_STAGE;
            float s[MR][VEC];
            #pragma unroll
            for (int m = 0; m < MR; ++m) {
                const uint4 raw = 16 * m < K_d
                    ? *reinterpret_cast<const uint4*>(st + (m * 32 + lane) * 16)
                    : make_uint4(0u, 0u, 0u, 0u);
                const unsigned* w32 = reinterpret_cast<const unsigned*>(&raw);
                #pragma unroll
                for (int u = 0; u < 4; ++u) {
                    const float2 f = gdn_unpack(w32[u]);
                    s[m][2 * u] = f.x;
                    s[m][2 * u + 1] = f.y;
                }
            }
            const bf16* sq = reinterpret_cast<const bf16*>(st + 4096);
            const bf16* sk = reinterpret_cast<const bf16*>(st + 4096 + 256);
            float qi[MR];
            float ki[MR];
            float qs = 0.f;
            float ks = 0.f;
            #pragma unroll
            for (int m = 0; m < MR; ++m) {
                const int i = i0 + 16 * m;
                qi[m] = 16 * m < K_d ? Elem<bf16>::to_f32(sq[i]) : 0.f;
                ki[m] = 16 * m < K_d ? Elem<bf16>::to_f32(sk[i]) : 0.f;
                qs += qi[m] * qi[m];
                ks += ki[m] * ki[m];
            }
            float vv[VEC];
            {
                const uint4 vraw = *reinterpret_cast<const uint4*>(st + 4608 + (lane & 1) * 16);
                const unsigned* w32 = reinterpret_cast<const unsigned*>(&vraw);
                #pragma unroll
                for (int u = 0; u < 4; ++u) {
                    const float2 f = gdn_unpack(w32[u]);
                    vv[2 * u] = f.x;
                    vv[2 * u + 1] = f.y;
                }
            }
            const float* sgate = reinterpret_cast<const float*>(st + 4640);
            const float g = __expf(sgate[0]);
            const float beta = sgate[1];

            // Both lanes of a row pair hold the row, so the warp sums it twice.
            const float nq = rsqrtf(0.5f * warp_sum(qs) + 1e-6f) * q_scale;
            const float nk = rsqrtf(0.5f * warp_sum(ks) + 1e-6f);
            #pragma unroll
            for (int m = 0; m < MR; ++m) {
                qi[m] *= nq;
                ki[m] *= nk;
            }

            float x[VEC];
            #pragma unroll
            for (int u = 0; u < VEC; ++u) x[u] = 0.f;
            #pragma unroll
            for (int m = 0; m < MR; ++m) {
                #pragma unroll
                for (int u = 0; u < VEC; ++u) x[u] += s[m][u] * g * ki[m];
            }
            gdn_rows_sum(x);

            float o[VEC];
            #pragma unroll
            for (int u = 0; u < VEC; ++u) {
                x[u] = (vv[u] - x[u]) * beta;
                o[u] = 0.f;
            }
            #pragma unroll
            for (int m = 0; m < MR; ++m) {
                #pragma unroll
                for (int u = 0; u < VEC; ++u) {
                    const float sg = __bfloat162float(__float2bfloat16(s[m][u] * g));
                    const float sn = sg + ki[m] * x[u];
                    s[m][u] = sn;
                    o[u] += sn * qi[m];
                }
            }
            if (cur.fold) {
                #pragma unroll
                for (int m = 0; m < MR; ++m) {
                    if (16 * m >= K_d) break;
                    uint4 packed;
                    unsigned* w32 = reinterpret_cast<unsigned*>(&packed);
                    #pragma unroll
                    for (int u = 0; u < 4; ++u) w32[u] = pack_bf16x2(s[m][2 * u], s[m][2 * u + 1]);
                    *reinterpret_cast<uint4*>(cur.cell + (long long)(i0 + 16 * m) * V_d) = packed;
                }
            }
            gdn_rows_sum(o);
            if (lane < 2) {
                float4* dst = reinterpret_cast<float4*>(
                    out + ((long long)cur.r_row * V_h + cur.h) * V_d + cur.j0);
                dst[0] = make_float4(o[0], o[1], o[2], o[3]);
                dst[1] = make_float4(o[4], o[5], o[6], o[7]);
            }
        }
        __syncwarp();
        cur = nxt;
    }
    gdn_cp_wait_all();
}

// The gated delta rule over a lane's buffered rows as a linear attention
// (`Carry`). A row's step is `S_s = g_s S_{s-1} + k_s u_s^T`, with its
// corrected value `u_s = beta_s (v_s - g_s S_{s-1}^T k_s)`; from the
// committed state `S`, after buffered rows `1..m`,
//   S_t = G_{0,t} S + sum_s G_{s,t} k_s u_s^T,   G_{s,t} = prod_{s<i<=t} g_i,
// so a row's output and its own `u` read the state once and each buffered
// row as two dots: `S_t^T q = G_{0,t} S^T q + sum_s G_{s,t} (k_s . q) u_s`.
// A buffered row keeps its keys and its `u` (planes 1 and 2 of the lane's rs
// buffer pages; its `[b | a]`, plane 0, gives its gate), written once, when
// it is a fire's own row; a fold applies `S <- g S + k u^T` a committed row.
// The decays are formed in f32 from per-row log-gates, never as a ratio.

// A warp's tile keys from a key row (lane `l` its rows `l / 2 + 16 m`),
// normalized as the fused step normalizes them, times `scale`.
__device__ __forceinline__ float gdn_keys(const bf16* __restrict__ row, int i0, int K_d,
                                          float scale, float (&ki)[8])
{
    float ks = 0.f;
    #pragma unroll
    for (int m = 0; m < 8; ++m) {
        ki[m] = 16 * m < K_d ? Elem<bf16>::to_f32(row[i0 + 16 * m]) : 0.f;
        ks += ki[m] * ki[m];
    }
    const float nk = rsqrtf(0.5f * warp_sum(ks) + 1e-6f) * scale;
    #pragma unroll
    for (int m = 0; m < 8; ++m) ki[m] *= nk;
    return nk;
}

// One lane's buffered row's dots against the own row's raw q and k (bf16,
// `K_d` each, in shared memory): its keys' norm folded in, and the own
// rows' scales `nq`, `nk`.
__device__ __forceinline__ void gdn_row_dots(const bf16* __restrict__ krow,
                                             const bf16* __restrict__ sq,
                                             const bf16* __restrict__ sk, int K_d, float nq,
                                             float nk, float& dq, float& dk)
{
    float kk = 0.f;
    float q = 0.f;
    float k = 0.f;
    // Every load of the row in flight at once: the key width is at most 128.
    #pragma unroll
    for (int i = 0; i < 128; i += 8) {
        if (i >= K_d) break;
        const uint4 kr = *reinterpret_cast<const uint4*>(krow + i);
        const uint4 qr = *reinterpret_cast<const uint4*>(sq + i);
        const uint4 sr = *reinterpret_cast<const uint4*>(sk + i);
        const unsigned* kw = reinterpret_cast<const unsigned*>(&kr);
        const unsigned* qw = reinterpret_cast<const unsigned*>(&qr);
        const unsigned* sw = reinterpret_cast<const unsigned*>(&sr);
        #pragma unroll
        for (int u = 0; u < 4; ++u) {
            const float2 a = gdn_unpack(kw[u]);
            const float2 b = gdn_unpack(qw[u]);
            const float2 c = gdn_unpack(sw[u]);
            kk += a.x * a.x + a.y * a.y;
            q += a.x * b.x + a.y * b.y;
            k += a.x * c.x + a.y * c.y;
        }
    }
    const float nj = rsqrtf(kk + 1e-6f);
    dq = q * nj * nq;
    dk = k * nj * nk;
}

// A buffered row's log-gate from its `[b | a]`, as `gdn_prep_ba_gates`
// forms it.
__device__ __forceinline__ float gdn_log_gate(const bf16* __restrict__ ba, int h, int V_h,
                                              const bf16* __restrict__ dt_bias,
                                              const float* __restrict__ a_log)
{
    const float z = Elem<bf16>::to_f32(ba[V_h + h]) + Elem<bf16>::to_f32(dt_bias[h]);
    const float sp = (z > 20.f) ? z : log1pf(__expf(z));
    return -__expf(a_log[h]) * sp;
}

// A warp's columns of a buffered row's `u` (f32, eight from `j0`).
__device__ __forceinline__ void gdn_u8(const float* __restrict__ u, float (&uu)[8])
{
    const float4 a = reinterpret_cast<const float4*>(u)[0];
    const float4 b = reinterpret_cast<const float4*>(u)[1];
    uu[0] = a.x; uu[1] = a.y; uu[2] = a.z; uu[3] = a.w;
    uu[4] = b.x; uu[5] = b.y; uu[6] = b.z; uu[7] = b.w;
}

// The fold of one row into the warp's tile: `S <- g S + k u^T`.
__device__ __forceinline__ void gdn_fold_row(float (&s)[8][8], float g, const float (&ki)[8],
                                             const float (&uu)[8])
{
    #pragma unroll
    for (int m = 0; m < 8; ++m) {
        #pragma unroll
        for (int u = 0; u < 8; ++u) s[m][u] = g * s[m][u] + ki[m] * uu[u];
    }
}

// The gate prep of a buffering decode fire (`gdn_prep_ba_gates`'s gates for
// the own row, which lands its `[b | a]` in the lane's buffer), and the
// attention weights of the lane's buffered rows for the step: a warp a
// (lane, key head) sums each buffered row's keys' dots against the own
// row's q and k (normalized, q scaled) and, a lane a row for each value head
// of the group, its decay to the own row (the log-gates after it, a suffix
// scan, and the own row's), into the carried scratch: row `ext + j` holds
// the weights `(w k_j . k, w k_j . q)` a value head, row `ext + n` the
// state's decay to the own row. The step reads them, not the rows' keys.
__global__ void __launch_bounds__(32) gdn_gates_carried(
    const bf16* __restrict__ ba,
    const bf16* __restrict__ dt_bias,
    const float* __restrict__ a_log,
    float* __restrict__ gates,
    const bf16* __restrict__ qkv_post,
    int K_h, int V_h, int K_d, int conv_dim,
    float q_scale,
    Carry carry,
    const u32* __restrict__ win)
{
    constexpr int B = 16;
    const int r = blockIdx.x;
    if (win != nullptr && r >= static_cast<int>(win[0])) return;
    const int r_row = win != nullptr ? r + static_cast<int>(win[1]) : r;
    const int rl = win != nullptr ? r + static_cast<int>(win[3]) : r;
    const int h_k = blockIdx.y;
    const int lane = threadIdx.x;
    const int group = V_h / K_h;
    const int K_dim = K_h * K_d;
    const CarryLane cl = carry.lane(rl);
    const int n = carry.dec_count(rl);
    float2* weights = reinterpret_cast<float2*>(carry.scratch);
    // Every load the warp needs, issued together: the own row's `[b | a]`,
    // q and k (four elements a lane), and each buffered row's page slot (a
    // lane a row).
    const bf16* own_ba = ba + (long long)r_row * 2 * V_h;
    const bf16* row = qkv_post + (long long)r_row * conv_dim + (long long)h_k * K_d;
    float oq[4] = {0.f, 0.f, 0.f, 0.f};
    float ok[4] = {0.f, 0.f, 0.f, 0.f};
    if (n > 0 && 4 * lane < K_d) {
        const uint2 qr = *reinterpret_cast<const uint2*>(row + 4 * lane);
        const uint2 kr = *reinterpret_cast<const uint2*>(row + K_dim + 4 * lane);
        const float2 q0 = gdn_unpack(qr.x), q1 = gdn_unpack(qr.y);
        const float2 k0 = gdn_unpack(kr.x), k1 = gdn_unpack(kr.y);
        oq[0] = q0.x; oq[1] = q0.y; oq[2] = q1.x; oq[3] = q1.y;
        ok[0] = k0.x; ok[1] = k0.y; ok[2] = k1.x; ok[3] = k1.y;
    }
    // The own row's gates, a value head a lane, its `[b | a]` landed.
    float glog_own = 0.f;
    if (lane < group) {
        const int h = h_k * group + lane;
        glog_own = gdn_log_gate(own_ba, h, V_h, dt_bias, a_log);
        gates[(long long)r_row * 2 * V_h + h] = glog_own;
        gates[(long long)r_row * 2 * V_h + V_h + h] =
            1.f / (1.f + __expf(-Elem<bf16>::to_f32(own_ba[h])));
    }
    // The own row's `[b | a]` lands in its buffer, at the end: its place
    // waits on the lane's table.
    auto land = [&]() {
        if (lane < group) {
            const int h = h_k * group + lane;
            if (bf16* dst = carry.own_row<bf16>(cl, 0, 0)) {
                dst[h] = own_ba[h];
                dst[V_h + h] = own_ba[V_h + h];
            }
        }
    };
    if (carry.scratch == nullptr) {
        land();
        return;
    }
    if (n == 0) {
        if (lane < group) {
            weights[carry.dec_row(rl, carry.dec_stride) * V_h + h_k * group + lane] =
                make_float2(__expf(glog_own), 0.f);
        }
        land();
        return;
    }
    float qs = 0.f;
    float ks = 0.f;
    #pragma unroll
    for (int x = 0; x < 4; ++x) {
        qs += oq[x] * oq[x];
        ks += ok[x] * ok[x];
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        qs += __shfl_xor_sync(0xffffffffu, qs, off);
        ks += __shfl_xor_sync(0xffffffffu, ks, off);
    }
    const float nq = rsqrtf(qs + 1e-6f) * q_scale;
    const float nk = rsqrtf(ks + 1e-6f);
    // Each value head's decay through the later rows, from the own row's.
    float after[4];
    #pragma unroll
    for (int gi = 0; gi < 4; ++gi) after[gi] = __shfl_sync(0xffffffffu, glog_own, gi < group ? gi : 0);
    const int chunks = (n + B - 1) / B;
    for (int c = chunks - 1; c >= 0; --c) {
        const int base = B * c;
        const int rows = min(B, n - base);
        u64 slot = 0;
        u32 tok = 0;
        if (lane < rows) carry.dec_at(rl, base + lane, slot, tok);
        // The rows' keys, a row's four elements a lane, all in flight.
        float sq[B], sk[B], kk[B];
        #pragma unroll
        for (int b = 0; b < B; ++b) {
            const unsigned lo = __shfl_sync(0xffffffffu, static_cast<unsigned>(slot), b);
            const unsigned hi = __shfl_sync(0xffffffffu, static_cast<unsigned>(slot >> 32), b);
            const u32 tb = __shfl_sync(0xffffffffu, tok, b);
            const u64 sb = (static_cast<u64>(hi) << 32) | lo;
            sq[b] = 0.f;
            sk[b] = 0.f;
            kk[b] = 0.f;
            if (b < rows && 4 * lane < K_d) {
                const uint2 kr = *reinterpret_cast<const uint2*>(
                    carry.plane_at<const bf16>(sb, tb, 1) + (long long)h_k * K_d + 4 * lane);
                const float2 k0 = gdn_unpack(kr.x), k1 = gdn_unpack(kr.y);
                const float kv[4] = {k0.x, k0.y, k1.x, k1.y};
                #pragma unroll
                for (int x = 0; x < 4; ++x) {
                    sq[b] += kv[x] * oq[x];
                    sk[b] += kv[x] * ok[x];
                    kk[b] += kv[x] * kv[x];
                }
            }
        }
        // Each row's log-gates (a lane a row) while the sums reduce.
        float glog[4] = {0.f, 0.f, 0.f, 0.f};
        if (lane < rows) {
            const bf16* rb = carry.plane_at<const bf16>(slot, tok, 0);
            for (int gi = 0; gi < group && gi < 4; ++gi) {
                glog[gi] = gdn_log_gate(rb, h_k * group + gi, V_h, dt_bias, a_log);
            }
        }
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            #pragma unroll
            for (int b = 0; b < B; ++b) {
                sq[b] += __shfl_xor_sync(0xffffffffu, sq[b], off);
                sk[b] += __shfl_xor_sync(0xffffffffu, sk[b], off);
                kk[b] += __shfl_xor_sync(0xffffffffu, kk[b], off);
            }
        }
        float dq = 0.f;
        float dk = 0.f;
        #pragma unroll
        for (int b = 0; b < B; ++b) {
            if (lane == b) {
                const float nj = rsqrtf(kk[b] + 1e-6f);
                dq = sq[b] * nj * nq;
                dk = sk[b] * nj * nk;
            }
        }
        for (int gi = 0; gi < group && gi < 4; ++gi) {
            float later = glog[gi];
            #pragma unroll
            for (int off = 1; off < 32; off <<= 1) {
                const float down = __shfl_down_sync(0xffffffffu, later, off);
                if (lane + off < 32) later += down;
            }
            const float total = __shfl_sync(0xffffffffu, later, 0);
            if (lane < rows) {
                const float w = __expf(after[gi] + later - glog[gi]);
                weights[carry.dec_row(rl, base + lane) * V_h + h_k * group + gi] =
                    make_float2(w * dk, w * dq);
            }
            after[gi] += total;
        }
    }
    if (lane < group && lane < 4) {
        float a = after[0];
        #pragma unroll
        for (int gi = 1; gi < 4; ++gi) a = lane == gi ? after[gi] : a;
        weights[carry.dec_row(rl, carry.dec_stride) * V_h + h_k * group + lane] = make_float2(__expf(a), 0.f);
    }
    land();
}

// `gdn_step_fused` over a lane's buffered rows then its own: the warp loads
// its state tile once, folds the rows the lane commits among its carried
// ones (`commit`), reads the rest as a linear attention for its own row's
// `u` and output, folds the own row where the lane commits it too, and
// stores the tile once if it folded any. The own row's `[b | a]` and `u`
// land in its buffer (its keys land with the conv's output). The state is
// rounded to bf16 only as it is stored.
template <int W>
__global__ void __launch_bounds__(W * 32) gdn_step_carried(
    const bf16* __restrict__ qkv_post,
    const float* __restrict__ gates,
    const bf16* __restrict__ ba,
    const bf16* __restrict__ dt_bias,
    const float* __restrict__ a_log,
    StateRows rows,
    float* __restrict__ out,
    int K_h, int V_h, int K_d, int V_d, int conv_dim,
    float q_scale,
    bool write_state,
    const u8* __restrict__ write_state_mask,
    int tiles,
    Carry carry,
    const u32* __restrict__ win)
{
    constexpr int MR = 8;
    constexpr int VEC = 8;
    extern __shared__ __align__(16) unsigned char gdn_ring[];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    unsigned char* ring = gdn_ring + warp * 2 * GDN_STREAM_STAGE;
    const int i0 = lane >> 1;
    const int vts = V_d / 16;
    const int group = V_h / K_h;
    const int K_dim = K_h * K_d;
    const int live_rows = win != nullptr ? static_cast<int>(win[0]) : 0x7fffffff;
    // A tile's buffered rows go in chunks of CH; the last chunk's corrected
    // values and weights are prefetched with the tile's state, into the
    // warp's carried stage of the state's stage.
    constexpr int CH = 16;
    constexpr int CSTAGE = CH * 16 * 4 + CH * 8;
    unsigned char* cring = gdn_ring + W * 2 * GDN_STREAM_STAGE + warp * 2 * CSTAGE;
    const float2* weights = reinterpret_cast<const float2*>(carry.scratch);

    struct Tile {
        bool live;
        bool fold;
        int r_row;
        int rl;
        int h;
        int j0;
        __nv_bfloat16* cell;
        // The lane's buffered rows, its commit length, its place in the
        // extended run, the state's decay to the own row, and (a lane a row)
        // the page slot of each row of its last chunk: resolved a tile ahead,
        // with the state's prefetch.
        int n;
        int commit;
        u32 ext;
        float g0;
        u64 slot;
        u32 tok;
    };
    auto plan = [&](int tile) {
        Tile tl;
        tl.live = false;
        tl.n = 0;
        tl.commit = 0;
        tl.ext = 0;
        tl.g0 = 0.f;
        tl.slot = 0;
        tl.tok = 0;
        if (tile >= tiles) return tl;
        const int vt = tile % vts;
        const int rest = tile / vts;
        tl.h = rest % V_h;
        const int r = rest / V_h;
        if (r >= live_rows) return tl;
        tl.r_row = win != nullptr ? r + static_cast<int>(win[1]) : r;
        tl.rl = win != nullptr ? r + static_cast<int>(win[3]) : r;
        const i32* rid = rows.row(tl.rl);
        if (rid[0] < 0) return tl;
        tl.live = true;
        tl.fold = folds(write_state, write_state_mask, tl.rl);
        tl.j0 = vt * 16 + (lane & 1) * VEC;
        tl.cell = rows.atom<__nv_bfloat16>(rid, tl.h, (long long)K_d * V_d) + tl.j0;
        tl.n = carry.dec_count(tl.rl);
        tl.commit = carry.commit(tl.rl, tl.n + 1);
        tl.g0 = weights[carry.dec_row(tl.rl, carry.dec_stride) * V_h + tl.h].x;
        const int j = CH * ((tl.n + CH - 1) / CH - 1) + lane;
        if (tl.n > 0 && lane < CH && j < tl.n) carry.dec_at(tl.rl, j, tl.slot, tl.tok);
        return tl;
    };
    auto issue = [&](const Tile& tl, int stage) {
        if (!tl.live) return;
        unsigned char* st = ring + stage * GDN_STREAM_STAGE;
        #pragma unroll
        for (int m = 0; m < MR; ++m) {
            if (16 * m < K_d) {
                gdn_cp16(st + (m * 32 + lane) * 16, tl.cell + (long long)(i0 + 16 * m) * V_d, true);
            }
        }
        const bf16* row = qkv_post + (long long)tl.r_row * conv_dim;
        const int h_k = tl.h / group;
        const int c = lane & 15;
        if (c * 8 < K_d) {
            const bf16* src = row + (lane < 16 ? 0 : K_dim) + (long long)h_k * K_d + c * 8;
            gdn_cp16(st + 4096 + (lane >> 4) * 256 + c * 16, src, true);
        }
        if (lane < 2) {
            gdn_cp16(st + 4608 + lane * 16, row + 2 * K_dim + (long long)tl.h * V_d + tl.j0, true);
            gdn_cp4(st + 4640 + lane * 4, gates + (long long)tl.r_row * 2 * V_h + lane * V_h + tl.h, true);
        }
        // The last chunk of the lane's buffered rows, a lane a row.
        if (tl.slot != 0) {
            unsigned char* cs = cring + stage * CSTAGE;
            const float* ur = carry.plane_at<const float>(tl.slot, tl.tok, 2) + (long long)tl.h * V_d +
                              (tl.j0 & ~15);
            #pragma unroll
            for (int q = 0; q < 4; ++q) gdn_cp16(cs + lane * 64 + q * 16, ur + 4 * q, true);
            const int j = CH * ((tl.n + CH - 1) / CH - 1) + lane;
            const float* wsrc = reinterpret_cast<const float*>(
                weights + carry.dec_row(tl.rl, j) * V_h + tl.h);
            gdn_cp4(cs + CH * 64 + lane * 8, wsrc, true);
            gdn_cp4(cs + CH * 64 + lane * 8 + 4, wsrc + 1, true);
        }
    };
    auto unpack8 = [&](const uint4& raw, float (&f8)[8]) {
        const unsigned* w32 = reinterpret_cast<const unsigned*>(&raw);
        #pragma unroll
        for (int u = 0; u < 4; ++u) {
            const float2 f = gdn_unpack(w32[u]);
            f8[2 * u] = f.x;
            f8[2 * u + 1] = f.y;
        }
    };

    const int stride = gridDim.x * W;
    int t = blockIdx.x * W + warp;
    Tile cur = plan(t);
    issue(cur, 0);
    gdn_cp_commit();
    for (int it = 0; t < tiles; t += stride, ++it) {
        const Tile nxt = plan(t + stride);
        issue(nxt, (it + 1) & 1);
        gdn_cp_commit();
        // The tile's buffered rows, in chunks of CH, a lane a row: the
        // last chunk's corrected values (the tile's sixteen columns) and
        // attention weights (the gate prep's, a value head each) came with
        // the state; an earlier chunk's are read into the same stage.
        unsigned char* cs = cring + (it & 1) * CSTAGE;
        float* su = reinterpret_cast<float*>(cs);
        const float2* sw = reinterpret_cast<const float2*>(cs + CH * 64);
        auto shfl64 = [&](u64 v, int from) {
            const unsigned lo = __shfl_sync(0xffffffffu, static_cast<unsigned>(v), from);
            const unsigned hi = __shfl_sync(0xffffffffu, static_cast<unsigned>(v >> 32), from);
            return (static_cast<u64>(hi) << 32) | lo;
        };
        auto resolve = [&](const CarryLane& cl, int n, int j, u64& slot, u32& tok) {
            slot = 0;
            tok = 0;
            if (j < n) carry.dec_at(cur.rl, j, slot, tok);
        };
        auto read_chunk = [&](const CarryLane& cl, int n, int base) {
            const int j = base + lane;
            if (lane >= CH || j >= n) return;
            u64 slot;
            u32 tok;
            carry.dec_at(cur.rl, j, slot, tok);
            reinterpret_cast<float2*>(cs + CH * 64)[lane] = weights[carry.dec_row(cur.rl, j) * V_h + cur.h];
            const float4* ur = reinterpret_cast<const float4*>(
                carry.plane_at<const float>(slot, tok, 2) + (long long)cur.h * V_d + (cur.j0 & ~15));
            float4* sr = reinterpret_cast<float4*>(su + lane * 16);
            sr[0] = ur[0];
            sr[1] = ur[1];
            sr[2] = ur[2];
            sr[3] = ur[3];
        };
        const CarryLane cl = carry.lane(cur.live ? cur.rl : 0);
        const int n = cur.n;
        const int chunks = (n + CH - 1) / CH;
        const float g0 = cur.g0;
        asm volatile("cp.async.wait_group 1;\n" ::: "memory");
        __syncwarp();
        if (cur.live) {
            const unsigned char* st = ring + (it & 1) * GDN_STREAM_STAGE;
            float s[MR][VEC];
            #pragma unroll
            for (int m = 0; m < MR; ++m) {
                const uint4 raw = 16 * m < K_d
                    ? *reinterpret_cast<const uint4*>(st + (m * 32 + lane) * 16)
                    : make_uint4(0u, 0u, 0u, 0u);
                unpack8(raw, s[m]);
            }
            const int h_k = cur.h / group;
            const int commit = cur.commit;
            const bool fold = cur.fold && commit > 0;
            // The own row: q (scaled), k, v, its log-gate and beta.
            const bf16* sq = reinterpret_cast<const bf16*>(st + 4096);
            const bf16* sk = reinterpret_cast<const bf16*>(st + 4096 + 256);
            float qi[MR];
            float ki[MR];
            gdn_keys(sq, i0, K_d, q_scale, qi);
            gdn_keys(sk, i0, K_d, 1.f, ki);
            float vv[VEC];
            unpack8(*reinterpret_cast<const uint4*>(st + 4608 + (lane & 1) * 16), vv);
            const float* sgate = reinterpret_cast<const float*>(st + 4640);
            const float glog_n = sgate[0];
            const float beta = sgate[1];
            // The buffered rows read as a linear attention from the committed
            // state.
            float a[VEC];
            float o[VEC];
            #pragma unroll
            for (int u = 0; u < VEC; ++u) {
                a[u] = 0.f;
                o[u] = 0.f;
            }
            for (int c = chunks - 1; c >= 0; --c) {
                const int base = CH * c;
                if (c != chunks - 1) {
                    __syncwarp();
                    read_chunk(cl, n, base);
                }
                __syncwarp();
                const int rows = min(CH, n - base);
                #pragma unroll 4
                for (int e = 0; e < rows; ++e) {
                    const float2 wc = sw[e];
                    const float4* ue = reinterpret_cast<const float4*>(su + e * 16 + (lane & 1) * 8);
                    const float4 u0 = ue[0];
                    const float4 u1 = ue[1];
                    const float uu[VEC] = {u0.x, u0.y, u0.z, u0.w, u1.x, u1.y, u1.z, u1.w};
                    #pragma unroll
                    for (int u = 0; u < VEC; ++u) {
                        a[u] += wc.x * uu[u];
                        o[u] += wc.y * uu[u];
                    }
                }
            }
            // The state's part: S^T k and S^T q over the tile, decayed to the
            // own row.
            float xq[VEC];
            float xk[VEC];
            #pragma unroll
            for (int u = 0; u < VEC; ++u) {
                xq[u] = 0.f;
                xk[u] = 0.f;
            }
            #pragma unroll
            for (int m = 0; m < MR; ++m) {
                #pragma unroll
                for (int u = 0; u < VEC; ++u) {
                    xq[u] += s[m][u] * qi[m];
                    xk[u] += s[m][u] * ki[m];
                }
            }
            gdn_rows_sum(xq);
            gdn_rows_sum(xk);
            #pragma unroll
            for (int u = 0; u < VEC; ++u) {
                a[u] += g0 * xk[u];
                o[u] += g0 * xq[u];
            }
            // The carried rows the lane commits fold into the state.
            const int folded = fold ? min(commit, n) : 0;
            for (int base = 0; base < folded; base += 32) {
                u64 slot;
                u32 tok;
                resolve(cl, folded, base + lane, slot, tok);
                const float glog = slot != 0
                    ? gdn_log_gate(carry.plane_at<const bf16>(slot, tok, 0), cur.h, V_h, dt_bias, a_log)
                    : 0.f;
                const int rows = min(32, folded - base);
                for (int e = 0; e < rows; ++e) {
                    const u64 se = shfl64(slot, e);
                    const u32 te = __shfl_sync(0xffffffffu, tok, e);
                    const float g = __expf(__shfl_sync(0xffffffffu, glog, e));
                    float kj[MR];
                    gdn_keys(carry.plane_at<const bf16>(se, te, 1) + (long long)h_k * K_d, i0, K_d, 1.f, kj);
                    float uu[VEC];
                    gdn_u8(carry.plane_at<const float>(se, te, 2) + (long long)cur.h * V_d + cur.j0, uu);
                    gdn_fold_row(s, g, kj, uu);
                }
            }
            float kq = 0.f;
            #pragma unroll
            for (int m = 0; m < MR; ++m) kq += ki[m] * qi[m];
            kq = 0.5f * warp_sum(kq);
            float un[VEC];
            #pragma unroll
            for (int u = 0; u < VEC; ++u) {
                un[u] = beta * (vv[u] - a[u]);
                o[u] += kq * un[u];
            }
            if (lane < 2) {
                if (float* dst = carry.own_row<float>(cl, 0, 2)) {
                    float4* d4 = reinterpret_cast<float4*>(dst + (long long)cur.h * V_d + cur.j0);
                    d4[0] = make_float4(un[0], un[1], un[2], un[3]);
                    d4[1] = make_float4(un[4], un[5], un[6], un[7]);
                }
            }
            if (fold) {
                if (commit == n + 1) gdn_fold_row(s, __expf(glog_n), ki, un);
                #pragma unroll
                for (int m = 0; m < MR; ++m) {
                    if (16 * m >= K_d) break;
                    uint4 packed;
                    unsigned* w32 = reinterpret_cast<unsigned*>(&packed);
                    #pragma unroll
                    for (int u = 0; u < 4; ++u) w32[u] = pack_bf16x2(s[m][2 * u], s[m][2 * u + 1]);
                    *reinterpret_cast<uint4*>(cur.cell + (long long)(i0 + 16 * m) * V_d) = packed;
                }
            }
            if (lane < 2) {
                float4* dst = reinterpret_cast<float4*>(
                    out + ((long long)cur.r_row * V_h + cur.h) * V_d + cur.j0);
                dst[0] = make_float4(o[0], o[1], o[2], o[3]);
                dst[1] = make_float4(o[4], o[5], o[6], o[7]);
            }
            // The head's first tile lands the own row's `[b | a]` pair.
            if ((cur.j0 & ~15) == 0 && lane < 2) {
                if (bf16* dst = carry.own_row<bf16>(cl, 0, 0)) {
                    const int at = lane * V_h + cur.h;
                    dst[at] = ba[(long long)cur.r_row * 2 * V_h + at];
                }
            }
        }
        __syncwarp();
        cur = nxt;
    }
    gdn_cp_wait_all();
}

// The buffered rows a chunked fire's lanes land past their commit, made
// ready for a later decode's linear attention: each row's corrected value
// `u` (plane 2) and its keys (plane 1), from the state as the fire left it
// (folded through the lane's commit, or as it stood where the lane folds
// nothing), stepping the lane's rows past the commit in order (a carried
// one by its stored `u`, an own one forming its `u`). The state stays as it
// was. A warp a 16-column tile of a (lane, head), the own rows' `q k v`
// rows (`qkv_post`) and gates read where the fire lays them: the lane's
// extended rows (`ext`: row `RsLane::ext + j`) or its own rows
// (`qo_indptr`, the window's row offset `win[1]`).
template <int W>
__global__ void __launch_bounds__(W * 32) gdn_tail_corrected(
    const bf16* __restrict__ qkv_post,
    const float* __restrict__ gates,
    const bf16* __restrict__ dt_bias,
    const float* __restrict__ a_log,
    StateRows rows,
    const u32* __restrict__ qo_indptr,
    bool ext,
    int K_h, int V_h, int K_d, int V_d, int conv_dim,
    bool write_state,
    const u8* __restrict__ write_state_mask,
    int lanes_stated,
    Carry carry,
    const u32* __restrict__ win)
{
    constexpr int MR = 8;
    constexpr int VEC = 8;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int i0 = lane >> 1;
    const int vts = V_d / 16;
    const int group = V_h / K_h;
    const int K_dim = K_h * K_d;
    const int lanes = win != nullptr ? static_cast<int>(win[2]) : lanes_stated;
    const int r = blockIdx.y;
    if (r >= lanes) return;
    const int tile = blockIdx.x * W + warp;
    if (tile >= V_h * vts) return;
    const int vt = tile % vts;
    const int h = tile / vts;
    const int h_k = h / group;
    const int j0 = vt * 16 + (lane & 1) * VEC;
    const int rl = win != nullptr ? r + static_cast<int>(win[3]) : r;
    const int row0 = win != nullptr ? static_cast<int>(win[1]) : 0;
    const CarryLane cl = carry.lane(rl);
    if (cl.at == CARRY_NONE) return;
    const i32* rid = rows.row(rl);
    if (rid[0] < 0) return;
    const int n = carry.carried(cl);
    const int own = static_cast<int>(qo_indptr[r + 1] - qo_indptr[r]);
    const int run = n + own;
    const int start = folds(write_state, write_state_mask, rl) ? carry.commit(rl, run) : 0;
    if (start >= run) return;
    const __nv_bfloat16* cell = rows.atom<__nv_bfloat16>(rid, h, (long long)K_d * V_d) + j0;
    float s[MR][VEC];
    #pragma unroll
    for (int m = 0; m < MR; ++m) {
        if (16 * m < K_d) {
            const uint4 raw = *reinterpret_cast<const uint4*>(cell + (long long)(i0 + 16 * m) * V_d);
            const unsigned* w32 = reinterpret_cast<const unsigned*>(&raw);
            #pragma unroll
            for (int u = 0; u < 4; ++u) {
                const float2 f = gdn_unpack(w32[u]);
                s[m][2 * u] = f.x;
                s[m][2 * u + 1] = f.y;
            }
        } else {
            #pragma unroll
            for (int u = 0; u < VEC; ++u) s[m][u] = 0.f;
        }
    }
    for (int j = start; j < run; ++j) {
        float ki[MR];
        float uu[VEC];
        float g;
        if (j < n) {
            gdn_keys(carry.carried_row<bf16>(cl, j, 1) + (long long)h_k * K_d, i0, K_d, 1.f, ki);
            gdn_u8(carry.carried_row<float>(cl, j, 2) + (long long)h * V_d + j0, uu);
            g = __expf(gdn_log_gate(carry.carried_row<bf16>(cl, j, 0), h, V_h, dt_bias, a_log));
        } else {
            const int e = j - n;
            const long long at = ext ? static_cast<long long>(cl.ext) + j
                                     : static_cast<long long>(row0) + qo_indptr[r] + e;
            const bf16* row = qkv_post + at * conv_dim;
            const bf16* krow = row + K_dim + (long long)h_k * K_d;
            gdn_keys(krow, i0, K_d, 1.f, ki);
            // The key head's first tile lands the row's keys.
            if (vt == 0 && h % group == 0 && (lane & 1) == 0) {
                if (bf16* dst = carry.own_row<bf16>(cl, e, 1)) {
                    #pragma unroll
                    for (int m = 0; m < MR; ++m) {
                        if (16 * m < K_d) dst[(long long)h_k * K_d + i0 + 16 * m] = krow[i0 + 16 * m];
                    }
                }
            }
            const float* gr = gates + at * 2 * V_h;
            const float glog = gr[h];
            const float beta = gr[V_h + h];
            g = __expf(glog);
            float vv[VEC];
            const uint4 vraw = *reinterpret_cast<const uint4*>(row + 2 * K_dim + (long long)h * V_d + j0);
            const unsigned* w32 = reinterpret_cast<const unsigned*>(&vraw);
            #pragma unroll
            for (int u = 0; u < 4; ++u) {
                const float2 f = gdn_unpack(w32[u]);
                vv[2 * u] = f.x;
                vv[2 * u + 1] = f.y;
            }
            float xk[VEC];
            #pragma unroll
            for (int u = 0; u < VEC; ++u) xk[u] = 0.f;
            #pragma unroll
            for (int m = 0; m < MR; ++m) {
                #pragma unroll
                for (int u = 0; u < VEC; ++u) xk[u] += s[m][u] * ki[m];
            }
            gdn_rows_sum(xk);
            #pragma unroll
            for (int u = 0; u < VEC; ++u) uu[u] = beta * (vv[u] - g * xk[u]);
            if (lane < 2) {
                if (float* dst = carry.own_row<float>(cl, e, 2)) {
                    float4* d4 = reinterpret_cast<float4*>(dst + (long long)h * V_d + j0);
                    d4[0] = make_float4(uu[0], uu[1], uu[2], uu[3]);
                    d4[1] = make_float4(uu[4], uu[5], uu[6], uu[7]);
                }
            }
        }
        gdn_fold_row(s, g, ki, uu);
    }
}

}
