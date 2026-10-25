#include "prim/device.cuh"
#include "prim/voxels.cuh"

namespace pie::attn {

template <int HEAD_DIM_MAX, int WARPS>
__global__ void dense_bidirectional(
    const bf16* __restrict__ q,
    const bf16* __restrict__ k,
    const bf16* __restrict__ v,
    bf16* __restrict__ o,
    const i32* __restrict__ segments,
    int num_segments,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    float sm_scale)
{
    constexpr int VPT = HEAD_DIM_MAX / 32;
    const int threads = WARPS * 32;
    const float neg_inf = __int_as_float(0xff800000u);

    const int row = static_cast<int>(blockIdx.x);
    const int head = static_cast<int>(blockIdx.y);
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int warp = static_cast<int>(threadIdx.x) >> 5;

    extern __shared__ float smem[];
    float* q_s = smem;
    float* wacc = q_s + head_dim;
    float* wm = wacc + WARPS * head_dim;
    float* wl = wm + WARPS;

    __shared__ int span[2];

    if (threadIdx.x == 0) {
        int begin = -1;
        int end = -1;
        const int first = segments[0];
        const int total = segments[num_segments];
        if (row >= first && row < total) {
            int lo = 0;
            int hi = num_segments - 1;
            while (lo < hi) {
                const int mid = (lo + hi + 1) >> 1;
                if (segments[mid] <= row) {
                    lo = mid;
                } else {
                    hi = mid - 1;
                }
            }
            begin = segments[lo];
            end = segments[lo + 1];
        }
        span[0] = begin;
        span[1] = end;
    }
    __syncthreads();

    const int begin = span[0];
    const int end = span[1];

    bf16* out = o + (static_cast<long long>(row) * num_q_heads + head) * head_dim;
    if (end <= begin) {

        for (int d = static_cast<int>(threadIdx.x); d < head_dim; d += threads) {
            out[d] = f32_to_bf16(0.f);
        }
        return;
    }

    const bf16* q_row = q + (static_cast<long long>(row) * num_q_heads + head) * head_dim;
    for (int d = static_cast<int>(threadIdx.x); d < head_dim; d += threads) {
        q_s[d] = bf16_to_f32(q_row[d]);
    }
    __syncthreads();

    const int group = num_q_heads / num_kv_heads;
    const int kv_head = head / group;

    float acc[VPT];
#pragma unroll
    for (int u = 0; u < VPT; ++u) {
        acc[u] = 0.f;
    }
    float running_max = neg_inf;
    float running_sum = 0.f;

    for (int j = begin + warp; j < end; j += WARPS) {
        const bf16* k_row =
            k + (static_cast<long long>(j) * num_kv_heads + kv_head) * head_dim;
        float dot = 0.f;
#pragma unroll
        for (int u = 0; u < VPT; ++u) {
            const int d = lane + u * 32;
            if (d < head_dim) {
                dot += q_s[d] * bf16_to_f32(k_row[d]);
            }
        }

        dot = ::pie::warp_sum(dot);

        const float score = dot * sm_scale;
        const float widened = fmaxf(running_max, score);
        const float rescale = __expf(running_max - widened);
        const float weight = __expf(score - widened);

        const bf16* v_row =
            v + (static_cast<long long>(j) * num_kv_heads + kv_head) * head_dim;
#pragma unroll
        for (int u = 0; u < VPT; ++u) {
            const int d = lane + u * 32;
            if (d < head_dim) {
                acc[u] = acc[u] * rescale + weight * bf16_to_f32(v_row[d]);
            }
        }
        running_sum = running_sum * rescale + weight;
        running_max = widened;
    }

    if (lane == 0) {
        wm[warp] = running_max;
        wl[warp] = running_sum;
    }
#pragma unroll
    for (int u = 0; u < VPT; ++u) {
        const int d = lane + u * 32;
        if (d < head_dim) {
            wacc[warp * head_dim + d] = acc[u];
        }
    }
    __syncthreads();

    float folded_max = neg_inf;
#pragma unroll
    for (int w = 0; w < WARPS; ++w) {
        folded_max = fmaxf(folded_max, wm[w]);
    }
    float denominator = 0.f;
#pragma unroll
    for (int w = 0; w < WARPS; ++w) {
        denominator += wl[w] * __expf(wm[w] - folded_max);
    }
    const float inv = denominator > 0.f ? 1.f / denominator : 0.f;

    for (int d = static_cast<int>(threadIdx.x); d < head_dim; d += threads) {
        float sum = 0.f;
#pragma unroll
        for (int w = 0; w < WARPS; ++w) {
            sum += wacc[w * head_dim + d] * __expf(wm[w] - folded_max);
        }
        out[d] = f32_to_bf16(sum * inv);
    }
}

}

namespace pie::attn::voxel {

using namespace ::pie::voxels;

namespace detail {

__device__ __forceinline__ void unpack8(uint4 word, float out[8]) {
    const unsigned int u[4] = {word.x, word.y, word.z, word.w};
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        out[2 * i] = __uint_as_float(u[i] << 16);
        out[2 * i + 1] = __uint_as_float(u[i] & 0xffff0000u);
    }
}

}

template <int C, int QPW>
__global__ __launch_bounds__(128) void attention(
    const bf16* __restrict__ q,
    const bf16* __restrict__ k,
    const bf16* __restrict__ v,
    const int* __restrict__ grid,
    bf16* __restrict__ y,
    int lanes,
    int rows,
    int seg_frames,
    int base,
    float scale_log2)
{
    constexpr int CPL = C / 32;
    constexpr bool VEC = (CPL % 8 == 0);
    constexpr int WORDS = CPL / 8;
    static_assert(C % 32 == 0, "every lane holds a whole slice of the row");

    auto read_slice = [](const bf16* __restrict__ base, int row, int col, float (&out)[CPL]) {
        const bf16* at = base + static_cast<size_t>(row) * C + col;
        if constexpr (VEC) {
            const uint4* src = reinterpret_cast<const uint4*>(at);
#pragma unroll
            for (int w = 0; w < WORDS; ++w) {
                float tmp[8];
                detail::unpack8(__ldg(src + w), tmp);
#pragma unroll
                for (int e = 0; e < 8; ++e) out[w * 8 + e] = tmp[e];
            }
        } else {
#pragma unroll
            for (int e = 0; e < CPL; ++e) out[e] = bf16_to_f32(ldg(at + e));
        }
    };

    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int first = (blockIdx.x * 4 + warp) * QPW;
    if (first >= rows) return;
    const int col = lane * CPL;

    int owner[QPW];
    int seg_begin[QPW], seg_end[QPW];
    Lane box[QPW];
    float qreg[QPW][CPL];
#pragma unroll
    for (int i = 0; i < QPW; ++i) {
        const int r = first + i;
        // `base`: the row `grid` addresses row 0 at; segments land local.
        owner[i] = r < rows ? lane_of(grid, lanes, r + base, box[i]) : -2;
        seg_begin[i] = 0;
        seg_end[i] = 0;
        if (owner[i] >= 0) {
            box[i].off -= base;
            segment_of(box[i], r - box[i].off, seg_frames, seg_begin[i], seg_end[i]);
            read_slice(q, r, col, qreg[i]);
#pragma unroll
            for (int e = 0; e < CPL; ++e) qreg[i][e] *= scale_log2;
        } else {
#pragma unroll
            for (int c = 0; c < CPL; ++c) qreg[i][c] = 0.f;
        }
    }

    float acc[QPW][CPL];
    float m[QPW], l[QPW];
#pragma unroll
    for (int i = 0; i < QPW; ++i) {
        m[i] = __int_as_float(0xff800000);
        l[i] = 0.f;
#pragma unroll
        for (int c = 0; c < CPL; ++c) acc[i][c] = 0.f;
    }

    bool done[QPW];
#pragma unroll
    for (int i = 0; i < QPW; ++i) done[i] = owner[i] < 0;
    for (;;) {
        int pick = -1;
#pragma unroll
        for (int i = 0; i < QPW; ++i) if (pick < 0 && !done[i]) pick = i;
        if (pick < 0) break;
        const int who = seg_begin[pick];
        bool active[QPW];
#pragma unroll
        for (int i = 0; i < QPW; ++i) {
            active[i] = !done[i] && seg_begin[i] == who;
            if (active[i]) done[i] = true;
        }
        const int begin = who;
        const int end = seg_end[pick];
        for (int j = begin; j < end; ++j) {
            float kk[CPL];
            read_slice(k, j, col, kk);
            float s[QPW];
#pragma unroll
            for (int i = 0; i < QPW; ++i) s[i] = 0.f;
#pragma unroll
            for (int e = 0; e < CPL; ++e) {
#pragma unroll
                for (int i = 0; i < QPW; ++i) s[i] = fmaf(qreg[i][e], kk[e], s[i]);
            }
            float p[QPW];
#pragma unroll
            for (int i = 0; i < QPW; ++i) {
                s[i] = ::pie::warp_sum(s[i]);
                if (active[i]) {
                    const float m_new = fmaxf(m[i], s[i]);
                    const float rescale = exp2f(m[i] - m_new);
                    p[i] = exp2f(s[i] - m_new);
                    l[i] = l[i] * rescale + p[i];
                    m[i] = m_new;
#pragma unroll
                    for (int c = 0; c < CPL; ++c) acc[i][c] *= rescale;
                } else {
                    p[i] = 0.f;
                }
            }
            float vv[CPL];
            read_slice(v, j, col, vv);
#pragma unroll
            for (int e = 0; e < CPL; ++e) {
#pragma unroll
                for (int i = 0; i < QPW; ++i) acc[i][e] = fmaf(p[i], vv[e], acc[i][e]);
            }
        }
    }

#pragma unroll
    for (int i = 0; i < QPW; ++i) {
        const int r = first + i;
        if (r >= rows) continue;
        const float inv = owner[i] >= 0 && l[i] > 0.f ? 1.f / l[i] : 0.f;
        bf16* at = y + static_cast<size_t>(r) * C + col;
        if constexpr (VEC) {
            uint4* dst = reinterpret_cast<uint4*>(at);
#pragma unroll
            for (int w = 0; w < WORDS; ++w) {
                uint4 word;
                word.x = pack_bf16x2(acc[i][w * 8 + 0] * inv, acc[i][w * 8 + 1] * inv);
                word.y = pack_bf16x2(acc[i][w * 8 + 2] * inv, acc[i][w * 8 + 3] * inv);
                word.z = pack_bf16x2(acc[i][w * 8 + 4] * inv, acc[i][w * 8 + 5] * inv);
                word.w = pack_bf16x2(acc[i][w * 8 + 6] * inv, acc[i][w * 8 + 7] * inv);
                dst[w] = word;
            }
        } else {
#pragma unroll
            for (int e = 0; e < CPL; ++e) at[e] = f32_to_bf16(acc[i][e] * inv);
        }
    }
}

}
