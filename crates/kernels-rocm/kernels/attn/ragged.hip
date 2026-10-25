#pragma once

// Ragged attention over packed q, k and v (the vision towers' form): group g
// holds query rows [q_indptr[g], q_indptr[g+1]) and key rows [kv_indptr[g],
// kv_indptr[g+1]); a row attends to every key of its group the arm keeps,
// with no causal cut. A row before the first group or past the last keeps the
// bytes it had. MFMA flash attention as dense.hip: a block owns WAVES·16
// consecutive rows of one head and walks the keys of every group its rows
// fall in, BC at a time through LDS (V transposed); each lane masks to its own
// row's group. The head is zero-padded to the stamp D; scores in the log2
// domain, a group-local index on each side as kernels-cuda's fa2 variants.

#include "prim/device.hip"
#include "prim/mfma.hip"

namespace pie::attn {

// The arms, one per kernels-cuda fa2 ragged variant.
constexpr int RAGGED_FULL = 0;
constexpr int RAGGED_CUSTOM = 1;   // keep bit qo·kv_len + kv of group g's bits, from byte indptr[g]
constexpr int RAGGED_BIAS = 2;     // logit · scale + bias[head][clamp(kv − qo + max_len − 1)]

// An arm reads only its own operands; group g's `bits` start at byte `mask_indptr[g]`.
struct RaggedMaskArgs {
    const unsigned char* bits;
    const i32* mask_indptr;
    const float* bias;
    int max_len;
};

// The group holding query `row`, or -1 for a row no group holds. With empty
// groups the boundaries repeat; the last group starting at or before the row
// is the one that holds it.
__device__ __forceinline__ int ragged_group(const i32* __restrict__ q_indptr, int groups, int row) {
    if (row < q_indptr[0] || row >= q_indptr[groups]) return -1;
    int lo = 0, hi = groups - 1;
    while (lo < hi) {
        const int mid = (lo + hi + 1) >> 1;
        if (q_indptr[mid] <= row) lo = mid;
        else hi = mid - 1;
    }
    return lo;
}

template <int ARM>
__device__ __forceinline__ bool ragged_keeps(const RaggedMaskArgs& m, int g, int qo, int kv, int kv_len) {
    if constexpr (ARM == RAGGED_CUSTOM) {
        const unsigned long long cell =
            static_cast<unsigned long long>(qo) * static_cast<unsigned long long>(kv_len) +
            static_cast<unsigned long long>(kv);
        return (m.bits[m.mask_indptr[g] + static_cast<long long>(cell >> 3)] >> (cell & 7)) & 1;
    } else {
        return true;
    }
}

template <int D, int WAVES, int ARM>
__global__ __launch_bounds__(WAVES * 64) void ragged(
    const bf16* __restrict__ q, const bf16* __restrict__ k, const bf16* __restrict__ v,
    bf16* __restrict__ o, const i32* __restrict__ q_indptr, const i32* __restrict__ kv_indptr,
    int groups, int num_q_heads, int num_kv_heads, int head_dim, float sm_scale, int rows,
    const unsigned char* __restrict__ bits, const i32* __restrict__ mask_indptr,
    const float* __restrict__ bias, int max_len) {
    static_assert(D % 16 == 0, "the stamp is whole 16-wide MFMA steps");
    constexpr int BC = 32, KT = BC / 16, KS = D / 16;
    constexpr int TILE = WAVES * 16;
    constexpr float LOG2E = 1.4426950408889634f;
    // 8 bytes of padding a row puts an operand read's 16 rows in distinct banks.
    constexpr int KP = D + 4, VP = BC + 4;
    __shared__ bf16 ks[BC * KP];
    __shared__ bf16 vt[D * VP];

    const RaggedMaskArgs mask{bits, mask_indptr, bias, max_len};
    const int head = static_cast<int>(blockIdx.y);
    const int kv_head = head / (num_q_heads / num_kv_heads);
    const int first_row = static_cast<int>(blockIdx.x) * TILE;
    const int wave = static_cast<int>(threadIdx.x) / kWave;
    const int ln = lane();
    const int quad = ln / 16;
    const int row = first_row + wave * 16 + ln % 16;

    // The keys the block's rows reach: the groups are ordered, so from the
    // first key of its first row's group to the last key of its last row's.
    const int lo_row = max(first_row, q_indptr[0]);
    const int hi_row = min(first_row + TILE, min(rows, q_indptr[groups])) - 1;
    if (lo_row > hi_row) return;
    const int key_lo = kv_indptr[ragged_group(q_indptr, groups, lo_row)];
    const int key_hi = kv_indptr[ragged_group(q_indptr, groups, hi_row) + 1];

    const int g = row < rows ? ragged_group(q_indptr, groups, row) : -1;
    const int q_base = g < 0 ? 0 : q_indptr[g];
    const int seg_lo = g < 0 ? 0 : kv_indptr[g];
    const int seg_hi = g < 0 ? 0 : kv_indptr[g + 1];
    const int qo = row - q_base;

    const long long kv_stride = static_cast<long long>(num_kv_heads) * head_dim;
    const bool packed = head_dim % 8 == 0 && aligned16(k) && aligned16(v);

    // Qᵀ as B operands: column = this lane's row, k = 4·quad + t.
    mfma_ab qf[KS];
    {
        const bf16* qrow = q + (static_cast<long long>(row) * num_q_heads + head) * head_dim;
        const bool live = g >= 0;
#pragma unroll
        for (int s = 0; s < KS; ++s) {
            const int d = s * 16 + 4 * quad;
            float x[4];
#pragma unroll
            for (int t = 0; t < 4; ++t) x[t] = live && d + t < head_dim ? to_f32(qrow[d + t]) : 0.f;
            qf[s] = pack4(x[0], x[1], x[2], x[3]);
        }
    }

    mfma_cd acc[KS];
#pragma unroll
    for (int s = 0; s < KS; ++s) acc[s] = mfma_zero();
    float m = neg_inf(), l = 0.f;
    const float scale_log2 = sm_scale * LOG2E;
    const int span = 2 * mask.max_len - 1;
    const float* bias_row = ARM == RAGGED_BIAS ? mask.bias + static_cast<long long>(head) * span : nullptr;

    for (int kv0 = key_lo; kv0 < key_hi; kv0 += BC) {
        if (packed) {
            for (int c = threadIdx.x; c < BC * (D / 8); c += blockDim.x) {
                const int key = c / (D / 8), j = (c % (D / 8)) * 8;
                Pack8<bf16> kv8{}, vv8{};
                if (kv0 + key < key_hi && j < head_dim) {
                    const long long at = (kv0 + key) * kv_stride + static_cast<long long>(kv_head) * head_dim + j;
                    kv8 = *reinterpret_cast<const Pack8<bf16>*>(k + at);
                    vv8 = *reinterpret_cast<const Pack8<bf16>*>(v + at);
                }
#pragma unroll
                for (int e = 0; e < 8; ++e) {
                    ks[key * KP + j + e] = kv8.v[e];
                    vt[(j + e) * VP + key] = vv8.v[e];
                }
            }
        } else {
            for (int c = threadIdx.x; c < BC * D; c += blockDim.x) {
                const int key = c / D, j = c % D;
                bf16 kx = from_f32<bf16>(0.f), vx = kx;
                if (kv0 + key < key_hi && j < head_dim) {
                    const long long at = (kv0 + key) * kv_stride + static_cast<long long>(kv_head) * head_dim + j;
                    kx = k[at];
                    vx = v[at];
                }
                ks[key * KP + j] = kx;
                vt[j * VP + key] = vx;
            }
        }
        lds_barrier();

        // Sᵀ: the lane holds keys 4·quad + t of each 16-key tile, for its row.
        mfma_cd s[KT];
#pragma unroll
        for (int kt = 0; kt < KT; ++kt) {
            s[kt] = mfma_zero();
#pragma unroll
            for (int st = 0; st < KS; ++st)
                s[kt] = mfma16(load4(&ks[(kt * 16 + ln % 16) * KP + st * 16 + 4 * quad]), qf[st], s[kt]);
        }
        float tile_max = neg_inf();
#pragma unroll
        for (int kt = 0; kt < KT; ++kt)
#pragma unroll
            for (int t = 0; t < 4; ++t) {
                const int key = kv0 + kt * 16 + 4 * quad + t;
                float x = neg_inf();
                if (key >= seg_lo && key < seg_hi) {
                    const int kv = key - seg_lo;
                    if constexpr (ARM == RAGGED_BIAS) {
                        const int d = min(max(kv - qo + mask.max_len - 1, 0), span - 1);
                        x = (s[kt][t] * sm_scale + bias_row[d]) * LOG2E;
                    } else if (ragged_keeps<ARM>(mask, g, qo, kv, seg_hi - seg_lo)) {
                        x = s[kt][t] * scale_log2;
                    }
                }
                s[kt][t] = x;
                tile_max = fmaxf(tile_max, x);
            }
        // A row's 32 keys sit in the four lanes ln % 16 + 16·quad.
        tile_max = fmaxf(tile_max, __shfl_xor(tile_max, 16, kWave));
        tile_max = fmaxf(tile_max, __shfl_xor(tile_max, 32, kWave));
        const float m_new = fmaxf(m, tile_max);
        const float alpha = m_new == neg_inf() ? 1.f : exp2f(m - m_new);
        float sum = 0.f;
        mfma_ab pt[KT];
#pragma unroll
        for (int kt = 0; kt < KT; ++kt) {
            float e[4];
#pragma unroll
            for (int t = 0; t < 4; ++t) {
                e[t] = m_new == neg_inf() ? 0.f : exp2f(s[kt][t] - m_new);
                sum += e[t];
            }
            pt[kt] = pack4(e[0], e[1], e[2], e[3]);
        }
        l = l * alpha + sum;
        m = m_new;
#pragma unroll
        for (int dt = 0; dt < KS; ++dt) {
            acc[dt] *= alpha;
#pragma unroll
            for (int kt = 0; kt < KT; ++kt)
                acc[dt] = mfma16(load4(&vt[(dt * 16 + ln % 16) * VP + kt * 16 + 4 * quad]), pt[kt], acc[dt]);
        }
        lds_barrier();
    }

    // Each lane summed only its own keys of the row; the row's four lanes
    // (ln, ln ± 16, ± 32) share m, so their sums add.
    l += __shfl_xor(l, 16, kWave);
    l += __shfl_xor(l, 32, kWave);
    if (g < 0) return;

    bf16* out = o + (static_cast<long long>(row) * num_q_heads + head) * head_dim;
    const float inv = l > 0.f ? 1.f / l : 0.f;
#pragma unroll
    for (int dt = 0; dt < KS; ++dt) {
        const int d = dt * 16 + 4 * quad;
        if (head_dim % 4 == 0) {
            if (d < head_dim)
                *reinterpret_cast<mfma_ab*>(out + d) =
                    pack4(acc[dt][0] * inv, acc[dt][1] * inv, acc[dt][2] * inv, acc[dt][3] * inv);
        } else {
#pragma unroll
            for (int t = 0; t < 4; ++t)
                if (d + t < head_dim) out[d + t] = from_f32<bf16>(acc[dt][t] * inv);
        }
    }
}

}  // namespace pie::attn
