#include "ssm/state.cuh"

namespace pie::ssm {

template <typename StateT, bool KLast>
__global__ void gdn_chunked(
    const float* __restrict__ q_norm,
    const float* __restrict__ k_norm,
    const float* __restrict__ v,
    const float* __restrict__ g_log,
    const float* __restrict__ beta,
    StateT*      __restrict__ state_base,
    const int*       __restrict__ slot_ids,
    const u32* __restrict__ qo_indptr,
    long long slot_stride_elems,
    float*       __restrict__ out,
    int V_h, int K_d, int V_d,
    const u32* __restrict__ win)
{
    const int r = blockIdx.x;
    const int h = blockIdx.y;

    if (win != nullptr && r >= static_cast<int>(win[2])) return;
    const int rl = win != nullptr ? r + static_cast<int>(win[3]) : r;

    const int row0 = win != nullptr ? static_cast<int>(win[1]) : 0;
    const int t0 = static_cast<int>(qo_indptr[r]);
    const int T  = static_cast<int>(qo_indptr[r + 1]) - t0;
    if (T <= 0) return;

    const int slot = slot_ids[rl];
    if (slot < 0) return;
    StateT* state = state_base
        + (long long)slot * slot_stride_elems
        + (long long)h * K_d * V_d;

    extern __shared__ float smem[];
    float* sq = smem;
    float* sk = smem + K_d;

    for (int t = 0; t < T; ++t) {
        const long long bh = (long long)(t0 + t) * V_h + h;
        const float* q_h = q_norm + bh * K_d;
        const float* k_h = k_norm + bh * K_d;
        const float* v_h = v      + bh * V_d;
        const float  g_h = __expf(g_log[bh]);
        const float  beta_h = beta[bh];

        float* out_bh = out + ((long long)(t0 + t + row0) * V_h + h) * V_d;

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
                state_store(state + off, s);
                kv_mem += s * sk[k_idx];
            }

            const float v_t   = v_h[v_idx];
            const float delta = (v_t - kv_mem) * beta_h;

            float out_v = 0.f;
            for (int k_idx = 0; k_idx < K_d; ++k_idx) {
                const long long off =
                    state_offset<KLast>(k_idx, v_idx, K_d, V_d);
                const float s = state_load(state + off) + sk[k_idx] * delta;
                state_store(state + off, s);
                out_v += s * sq[k_idx];
            }
            out_bh[v_idx] = out_v;
        }

        __syncthreads();
    }
}

template <typename StateT, bool KLast>
__global__ void gdn_chunked_warp_tiled(
    const float* __restrict__ q_norm_kh,
    const float* __restrict__ k_norm_kh,
    const float* __restrict__ v,
    const float* __restrict__ g_log,
    const float* __restrict__ beta,
    StateT*      __restrict__ state_base,
    const int*       __restrict__ slot_ids,
    const u32* __restrict__ qo_indptr,
    long long slot_stride_elems,
    float*       __restrict__ out,
    int K_h, int V_h, int K_d, int V_d,
    bool write_state,
    const u8* __restrict__ write_state_mask,
    const u32* __restrict__ win)
{
    constexpr int WARPS = 4;
    constexpr int MAX_K_PER_LANE = 8;
    const int r = blockIdx.x;
    const int h = blockIdx.y;
    const int v_tile = blockIdx.z * WARPS;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int v_idx = v_tile + warp;
    if (warp >= WARPS || v_idx >= V_d) return;

    if (win != nullptr && r >= static_cast<int>(win[2])) return;
    const int rl = win != nullptr ? r + static_cast<int>(win[3]) : r;
    const int row0 = win != nullptr ? static_cast<int>(win[1]) : 0;

    const int repeat = V_h / K_h;
    const int qk_h = h / repeat;
    const int t0 = static_cast<int>(qo_indptr[r]);
    const int T  = static_cast<int>(qo_indptr[r + 1]) - t0;
    if (T <= 0) return;

    const int slot = slot_ids[rl];
    if (slot < 0) return;
    StateT* state = state_base
        + (long long)slot * slot_stride_elems
        + (long long)h * K_d * V_d;

    float s_vals[MAX_K_PER_LANE];
    int k_vals[MAX_K_PER_LANE];
    int n_k = 0;
    for (int k_idx = lane; k_idx < K_d && n_k < MAX_K_PER_LANE; k_idx += 32) {
        k_vals[n_k] = k_idx;
        s_vals[n_k] = state_load(
            state + state_offset<KLast>(k_idx, v_idx, K_d, V_d));
        ++n_k;
    }

    for (int t = 0; t < T; ++t) {
        const long long qk_bh = ((long long)(t0 + t) * K_h + qk_h);
        const long long vh = (long long)(t0 + t) * V_h + h;
        const float* q_h = q_norm_kh + qk_bh * K_d;
        const float* k_h = k_norm_kh + qk_bh * K_d;
        const float* v_h = v + vh * V_d;
        const float g_h = __expf(g_log[vh]);
        const float beta_h = beta[vh];

        float kv_part = 0.f;
        #pragma unroll
        for (int i = 0; i < MAX_K_PER_LANE; ++i) {
            if (i < n_k) {
                const int k_idx = k_vals[i];
                const float s = s_vals[i] * g_h;
                s_vals[i] = s;
                kv_part += s * k_h[k_idx];
            }
        }
        const float kv_mem = warp_sum(kv_part);
        const float delta = (v_h[v_idx] - kv_mem) * beta_h;

        float out_part = 0.f;
        #pragma unroll
        for (int i = 0; i < MAX_K_PER_LANE; ++i) {
            if (i < n_k) {
                const int k_idx = k_vals[i];
                const float s = s_vals[i] + k_h[k_idx] * delta;
                s_vals[i] = s;
                out_part += s * q_h[k_idx];
            }
        }
        const float out_v = warp_sum(out_part);
        if (lane == 0) {

            const long long out_vh = (long long)(t0 + t + row0) * V_h + h;
            out[out_vh * (long long)V_d + v_idx] = out_v;
        }
    }

    if (write_state && row_persists(write_state_mask, rl)) {
        #pragma unroll
        for (int i = 0; i < MAX_K_PER_LANE; ++i) {
            if (i < n_k) {
                state_store(
                    state + state_offset<KLast>(
                        k_vals[i], v_idx, K_d, V_d),
                    s_vals[i]);
            }
        }
    }
}

template <typename StateT, int BV, int BK_MAX>
__global__ void gdn_chunked_fla(
    const float* __restrict__ q_norm,
    const float* __restrict__ k_norm,
    const float* __restrict__ v,
    const float* __restrict__ g_log,
    const float* __restrict__ beta,
    StateT*      __restrict__ state_base,
    const int*       __restrict__ slot_ids,
    const u32* __restrict__ qo_indptr,
    long long slot_stride_elems,
    float*       __restrict__ out,
    int K_h, int V_h, int K_d, int V_d,
    bool write_state,
    const int* __restrict__ commit_len,
    const u8* __restrict__ write_state_mask,
    const int* __restrict__ begin_at,
    bool fused_decay,
    const int* __restrict__ snap,
    const int* __restrict__ snap_rows,
    const int* __restrict__ snap_slots,
    const u32* __restrict__ win)
{

    const int gqa_repeat = V_h / K_h;

    const int vt = blockIdx.x;
    const int r  = blockIdx.y;
    const int h  = blockIdx.z;
    const int h_k = h / gqa_repeat;
    const int v_idx = vt * BV + threadIdx.x;
    if (v_idx >= V_d) return;

    if (win != nullptr && r >= static_cast<int>(win[2])) return;
    const int rl = win != nullptr ? r + static_cast<int>(win[3]) : r;
    const int row0 = win != nullptr ? static_cast<int>(win[1]) : 0;

    int t0 = static_cast<int>(qo_indptr[r]);
    int T  = static_cast<int>(qo_indptr[r + 1]) - t0;

    if (begin_at != nullptr) {
        int b = begin_at[rl];
        if (b > T) b = T;
        if (b > 0) { t0 += b; T -= b; }
    }
    if (commit_len != nullptr) {
        const int c = commit_len[rl];
        if (c < T) T = c;
    }
    if (T <= 0) return;

    const int slot = slot_ids[rl];
    if (slot < 0) return;
    StateT* state = state_base
        + (long long)slot * slot_stride_elems
        + (long long)h * K_d * V_d;

    extern __shared__ float smem[];
    float* sq = smem;
    float* sk = smem + BK_MAX;

    __nv_bfloat162 bh_state[BK_MAX / 2];
    #pragma unroll
    for (int j = 0; j < BK_MAX / 2; ++j) {
        const int k0 = 2 * j;
        if (k0 >= K_d) break;
        const int k1 = k0 + 1;
        const float s0 = state_load(state + (long long)k0 * V_d + v_idx);
        const float s1 = (k1 < K_d)
            ? state_load(state + (long long)k1 * V_d + v_idx)
            : 0.f;
        bh_state[j] = __floats2bfloat162_rn(s0, s1);
    }

    const bool single_round = fused_decay;
    Snaps sn = snaps_of(snap, rl);

    for (int t = 0; t < T; ++t) {
        const long long bh = (long long)(t0 + t) * V_h + h;
        const float  g_h = __expf(g_log[bh]);
        const float  beta_h = beta[bh];

        const long long bh_qk = (long long)(t0 + t) * K_h + h_k;
        const float* q_h_t = q_norm + bh_qk * K_d;
        const float* k_h_t = k_norm + bh_qk * K_d;
        for (int i = threadIdx.x; i < K_d; i += blockDim.x) {
            sq[i] = q_h_t[i];
            sk[i] = k_h_t[i];
        }
        __syncthreads();

        float kv_mem = 0.f;
        #pragma unroll
        for (int j = 0; j < BK_MAX / 2; ++j) {
            const int k0 = 2 * j;
            if (k0 >= K_d) break;
            const int k1 = k0 + 1;
            float2 s = __bfloat1622float2(bh_state[j]);
            s.x *= g_h;
            if (k1 < K_d) s.y *= g_h;
            if (!single_round) bh_state[j] = __floats2bfloat162_rn(s.x, s.y);
            kv_mem += s.x * sk[k0];
            if (k1 < K_d) kv_mem += s.y * sk[k1];
        }
        const float v_t   = v[bh * V_d + v_idx];
        const float delta = (v_t - kv_mem) * beta_h;

        float out_v = 0.f;
        #pragma unroll
        for (int j = 0; j < BK_MAX / 2; ++j) {
            const int k0 = 2 * j;
            if (k0 >= K_d) break;
            const int k1 = k0 + 1;
            float2 s = __bfloat1622float2(bh_state[j]);
            float sx, sy;
            if (single_round) {
                sx = s.x * g_h + sk[k0] * delta;
                sy = (k1 < K_d) ? (s.y * g_h + sk[k1] * delta) : s.y;
            } else {

                sx = s.x + sk[k0] * delta;
                sy = (k1 < K_d) ? (s.y + sk[k1] * delta) : s.y;
            }
            bh_state[j] = __floats2bfloat162_rn(sx, sy);
            out_v += sx * sq[k0];
            if (k1 < K_d) out_v += sy * sq[k1];
        }

        out[((long long)(t0 + t + row0) * V_h + h) * V_d + v_idx] = out_v;
        // A snapshot lands the state the walk carries, as a pass cut there
        // stores it.
        if (sn.e0 < sn.e1 && snap_rows[sn.e0] == t + 1) {
            if (write_state && row_persists(write_state_mask, rl)) {
                StateT* sat = state_base
                    + (long long)snap_slots[sn.e0] * slot_stride_elems
                    + (long long)h * K_d * V_d;
                #pragma unroll
                for (int j = 0; j < BK_MAX / 2; ++j) {
                    const int k0 = 2 * j;
                    if (k0 >= K_d) break;
                    const int k1 = k0 + 1;
                    const float2 sv = __bfloat1622float2(bh_state[j]);
                    state_store(sat + (long long)k0 * V_d + v_idx, sv.x);
                    if (k1 < K_d) state_store(sat + (long long)k1 * V_d + v_idx, sv.y);
                }
            }
            ++sn.e0;
        }
        __syncthreads();
    }

    if (!write_state || !row_persists(write_state_mask, rl)) return;
    #pragma unroll
    for (int j = 0; j < BK_MAX / 2; ++j) {
        const int k0 = 2 * j;
        if (k0 >= K_d) break;
        const int k1 = k0 + 1;
        const float2 s = __bfloat1622float2(bh_state[j]);
        state_store(state + (long long)k0 * V_d + v_idx, s.x);
        if (k1 < K_d) state_store(state + (long long)k1 * V_d + v_idx, s.y);
    }
}

}
