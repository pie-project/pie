#include "ssm/copy.cuh"

namespace pie::ssm {

constexpr int gqa_smem_bv = 128;

template <typename StateT, bool KLast>
__global__ void gdn_step(
    const float* __restrict__ q_norm_kh,
    const float* __restrict__ k_norm_kh,
    const float* __restrict__ v,
    const float* __restrict__ g_log,
    const float* __restrict__ beta,
    StateT*      __restrict__ state_base,
    const i32* __restrict__ slot_ids,
    long long slot_stride_elems,
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
    const int slot = slot_ids[rl];
    if (slot < 0) return;
    const bool fold = folds(write_state, write_state_mask, rl);

    const long long qh = ((long long)r * K_h + h_k) * K_d;
    const long long vh = (long long)r * V_h + h;
    const float* q_h = q_norm_kh + qh;
    const float* k_h = k_norm_kh + qh;
    const float* v_h = v + vh * V_d;
    const float  g_h = __expf(g_log[vh]);
    const float  beta_h = beta[vh];

    StateT* state = state_base
        + (long long)slot * slot_stride_elems
        + (long long)h * K_d * V_d;
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
    __nv_bfloat16* __restrict__ state_base,
    const i32* __restrict__ slot_ids,
    long long slot_stride_elems,
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
    const int slot = slot_ids[rl];
    if (slot < 0) return;
    const bool fold = folds(write_state, write_state_mask, rl);

    const long long qh = ((long long)r * K_h + h_k) * K_d;
    const long long vh = (long long)r * V_h + h;
    const float* v_h = v + vh * V_d;
    const float g_h = __expf(g_log[vh]);
    const float beta_h = beta[vh];

    __nv_bfloat16* state = state_base
        + (long long)slot * slot_stride_elems
        + (long long)h * K_d * V_d;
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
    __nv_bfloat16* __restrict__ state_base,
    const i32* __restrict__ slot_ids,
    long long slot_stride_elems,
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
        const int slot = slot_ids[rl];
        if (slot < 0) return tl;
        tl.live = true;
        tl.fold = folds(write_state, write_state_mask, rl);
        tl.j0 = vt * 16 + (lane & 1) * VEC;
        tl.cell = state_base + (long long)slot * slot_stride_elems
            + (long long)tl.h * K_d * V_d + tl.j0;
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
                    for (int u = 0; u < 4; ++u) w32[u] = gdn_pack(s[m][2 * u], s[m][2 * u + 1]);
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

}
