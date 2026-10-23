#include "prim/bf16.slang"
#include "prim/params.slang"
#include "prim/shuffle.slang"

// GGUF K-quant matmuls (Q2_K to Q6_K) on bf16 activations: `y[t, c]` is row
// `t` of `x` against weight row `c`, a run of 256-wide super-blocks of
// `PIE_BLOCK_BYTES` bytes, the bytes four to a word. A workgroup is
// `PIE_SLICES` strips of 32 lanes; a strip lands two columns (one bf16 word)
// of one row, its lanes walking the super-blocks, four codes at a time. The
// strip's sums fold through workgroup memory, or (`PIE_SUBGROUP`, subgroups
// at least 32 wide) with a butterfly inside the subgroup.

#define LANES 32u
#define SUPER 256u

[[vk::binding(0)]] StructuredBuffer<uint> w;
[[vk::binding(1)]] StructuredBuffer<uint> x;
[[vk::binding(2)]] RWStructuredBuffer<uint> y;

struct Params {
    uint n;
    uint k;
};
PIE_PARAMS(3, Params, params);

// Byte `at` of the weights.
uint w1(uint at) {
    return (w[at >> 2] >> ((at & 3u) * 8u)) & 0xffu;
}

// Bytes `at` to `at + 3`, the first in the low byte; `at` need not fall on a
// word.
uint w4(uint at) {
    uint lo = w[at >> 2];
    uint shift = (at & 3u) * 8u;
    if (shift == 0u) {
        return lo;
    }
    return (lo >> shift) | (w[(at >> 2) + 1u] << (32u - shift));
}

// The `mask`-wide field `shift` bits up in each of four bytes.
float4 field4(uint bytes, uint shift, uint mask) {
    return float4(
        float((bytes >> shift) & mask),
        float((bytes >> (8u + shift)) & mask),
        float((bytes >> (16u + shift)) & mask),
        float((bytes >> (24u + shift)) & mask));
}

// Activations `i` to `i + 3`, `i` a multiple of four.
float4 x4(uint i) {
    uint a = x[i >> 1];
    uint b = x[(i >> 1) + 1u];
    return float4(bf16_to_f32(a & 0xffffu), bf16_to_f32(a >> 16), bf16_to_f32(b & 0xffffu), bf16_to_f32(b >> 16));
}

float sum4(float4 v) {
    return v.x + v.y + v.z + v.w;
}

// An IEEE half from two weight bytes, subnormals and infinities kept.
float half_at(uint at) {
    uint h = w1(at) | (w1(at + 1u) << 8u);
    uint s = (h & 0x8000u) << 16u;
    uint e = (h >> 10u) & 0x1fu;
    uint m = h & 0x3ffu;
    if (e == 0u) {
        float v = float(m) * 5.9604645e-8;
        return (s != 0u) ? -v : v;
    }
    if (e == 31u) {
        return asfloat(s | 0x7f800000u | (m << 13u));
    }
    return asfloat(s | ((e + 112u) << 23u) | (m << 13u));
}

// Q4_K and Q5_K: sub-block `sub`'s six-bit scale and minimum.
float2 scale_min(uint at, uint sub) {
    if (sub < 4u) {
        return float2(float(w1(at + sub) & 63u), float(w1(at + sub + 4u) & 63u));
    }
    uint a = w1(at + sub + 4u);
    uint b = w1(at + sub - 4u);
    uint c = w1(at + sub);
    return float2(float((a & 0x0fu) | ((b >> 6u) << 4u)), float((a >> 4u) | ((c >> 6u) << 4u)));
}

// Q3_K: sub-block `sub`'s six-bit scale.
int q3_scale(uint at, uint sub) {
    uint grp = sub >> 2u;
    uint j = sub & 3u;
    uint src = w1(at + (((grp & 1u) != 0u) ? 4u : 0u) + j);
    uint low = (grp < 2u) ? (src & 0x0fu) : (src >> 4u);
    uint top = (w1(at + 8u + j) >> (2u * grp)) & 3u;
    return int(low | (top << 4u));
}

// Super-block `base`'s dot with the activations from `xb`.
float block_dot(uint base, uint xb) {
    float acc = 0.0;
#if PIE_SCHEME == 2
    float d = half_at(base + 80u);
    float dmin = half_at(base + 82u);
    for (uint b = 0u; b < 16u; b++) {
        uint shift = 2u * ((b >> 1u) & 3u);
        uint at = base + 16u + (b >> 3u) * 32u + (b & 1u) * 16u;
        float part = 0.0;
        float xsum = 0.0;
        [ForceUnroll] for (uint q = 0u; q < 4u; q++) {
            float4 xv = x4(xb + b * 16u + q * 4u);
            xsum += sum4(xv);
            part += dot(field4(w4(at + q * 4u), shift, 3u), xv);
        }
        uint packed = w1(base + b);
        acc += d * float(packed & 0x0fu) * part - dmin * float(packed >> 4u) * xsum;
    }
#elif PIE_SCHEME == 3
    float d = half_at(base + 108u);
    for (uint b = 0u; b < 16u; b++) {
        uint step = (b >> 1u) & 3u;
        uint high_bit = (b >> 3u) * 4u + step;
        uint at = base + 32u + (b >> 3u) * 32u + (b & 1u) * 16u;
        uint mask_at = base + (b & 1u) * 16u;
        float part = 0.0;
        [ForceUnroll] for (uint q = 0u; q < 4u; q++) {
            float4 xv = x4(xb + b * 16u + q * 4u);
            float4 codes = field4(w4(at + q * 4u), 2u * step, 3u);
            // A clear high bit borrows 4.
            float4 high = field4(w4(mask_at + q * 4u), high_bit, 1u);
            part += dot(codes + 4.0 * high - 4.0, xv);
        }
        acc += d * float(q3_scale(base + 96u, b) - 32) * part;
    }
#elif PIE_SCHEME == 4 || PIE_SCHEME == 5
    float d = half_at(base);
    float dmin = half_at(base + 2u);
#if PIE_SCHEME == 4
    uint codes_at = base + 16u;
#else
    uint codes_at = base + 48u;
#endif
    for (uint b = 0u; b < 8u; b++) {
        uint shift = (b & 1u) * 4u;
        uint at = codes_at + (b >> 1u) * 32u;
        float part = 0.0;
        float xsum = 0.0;
        [ForceUnroll] for (uint q = 0u; q < 8u; q++) {
            float4 xv = x4(xb + b * 32u + q * 4u);
            xsum += sum4(xv);
            float4 codes = field4(w4(at + q * 4u), shift, 15u);
#if PIE_SCHEME == 5
            codes += 16.0 * field4(w4(base + 16u + q * 4u), b, 1u);
#endif
            part += dot(codes, xv);
        }
        float2 sm = scale_min(base + 4u, b);
        acc += d * sm.x * part - dmin * sm.y * xsum;
    }
#elif PIE_SCHEME == 6
    float d = half_at(base + 208u);
    for (uint half_ = 0u; half_ < 2u; half_++) {
        for (uint quarter = 0u; quarter < 4u; quarter++) {
            uint low_at = base + half_ * 64u + 32u * (quarter & 1u);
            uint low_shift = (quarter < 2u) ? 0u : 4u;
            uint top_at = base + 128u + half_ * 32u;
            for (uint sub = 0u; sub < 2u; sub++) {
                float part = 0.0;
                [ForceUnroll] for (uint q = 0u; q < 4u; q++) {
                    uint i = sub * 16u + q * 4u;
                    float4 xv = x4(xb + half_ * 128u + quarter * 32u + i);
                    float4 low = field4(w4(low_at + i), low_shift, 15u);
                    float4 top = field4(w4(top_at + i), 2u * quarter, 3u);
                    part += dot(low + 16.0 * top - 32.0, xv);
                }
                int raw = int(w1(base + 192u + half_ * 8u + sub + 2u * quarter));
                acc += d * float((raw > 127) ? (raw - 256) : raw) * part;
            }
        }
    }
#else
#error "PIE_SCHEME must be one of 2, 3, 4, 5, 6"
#endif
    return acc;
}

#if !defined(PIE_SUBGROUP)
groupshared float folds[PIE_SLICES][2][LANES];
#endif

// The strip's two sums, folded; every lane holds the totals after.
float2 strip_sum(float2 v, uint lane, uint slice) {
#if defined(PIE_SUBGROUP)
    [ForceUnroll] for (uint off = LANES / 2u; off > 0u; off >>= 1u) {
        v.x += pie_shuffle_xor(v.x, off);
        v.y += pie_shuffle_xor(v.y, off);
    }
    return v;
#else
    folds[slice][0][lane] = v.x;
    folds[slice][1][lane] = v.y;
    GroupMemoryBarrierWithGroupSync();
    for (uint half_ = LANES / 2u; half_ > 0u; half_ >>= 1u) {
        if (lane < half_) {
            folds[slice][0][lane] += folds[slice][0][lane + half_];
            folds[slice][1][lane] += folds[slice][1][lane + half_];
        }
        GroupMemoryBarrierWithGroupSync();
    }
    return float2(folds[slice][0][0], folds[slice][1][0]);
#endif
}

[numthreads(32, PIE_SLICES, 1)]
void main(uint3 group : SV_GroupID, uint3 local : SV_GroupThreadID) {
    uint lane = local.x;
    uint slice = local.y;
    uint token = group.y;
    uint n = params.n;
    uint k = params.k;
    uint blocks = k / SUPER;
    uint row_bytes = blocks * uint(PIE_BLOCK_BYTES);
    uint col = (group.x * uint(PIE_SLICES) + slice) * 2u;

    float2 acc = float2(0.0, 0.0);
    if (col < n) {
        for (uint g = lane; g < blocks; g += LANES) {
            uint xb = token * k + g * SUPER;
            uint base = col * row_bytes + g * uint(PIE_BLOCK_BYTES);
            acc.x += block_dot(base, xb);
            acc.y += block_dot(base + row_bytes, xb);
        }
    }
    float2 total = strip_sum(acc, lane, slice);
    if (lane == 0u && col < n) {
        y[(token * n + col) >> 1] = pack_bf16(total.x, total.y);
    }
}

// pie:instantiate linear_kquant_q2k_bf16 PIE_SCHEME=2 PIE_BLOCK_BYTES=84 PIE_SLICES=2
// pie:instantiate linear_kquant_q3k_bf16 PIE_SCHEME=3 PIE_BLOCK_BYTES=110 PIE_SLICES=2
// pie:instantiate linear_kquant_q4k_bf16 PIE_SCHEME=4 PIE_BLOCK_BYTES=144 PIE_SLICES=2
// pie:instantiate linear_kquant_q5k_bf16 PIE_SCHEME=5 PIE_BLOCK_BYTES=176 PIE_SLICES=2
// pie:instantiate linear_kquant_q6k_bf16 PIE_SCHEME=6 PIE_BLOCK_BYTES=210 PIE_SLICES=2
// pie:instantiate linear_kquant_s4_q2k_bf16 PIE_SCHEME=2 PIE_BLOCK_BYTES=84 PIE_SLICES=4
// pie:instantiate linear_kquant_s4_q3k_bf16 PIE_SCHEME=3 PIE_BLOCK_BYTES=110 PIE_SLICES=4
// pie:instantiate linear_kquant_s4_q4k_bf16 PIE_SCHEME=4 PIE_BLOCK_BYTES=144 PIE_SLICES=4
// pie:instantiate linear_kquant_s4_q5k_bf16 PIE_SCHEME=5 PIE_BLOCK_BYTES=176 PIE_SLICES=4
// pie:instantiate linear_kquant_s4_q6k_bf16 PIE_SCHEME=6 PIE_BLOCK_BYTES=210 PIE_SLICES=4
// pie:instantiate linear_kquant_q2k_bf16 PIE_SCHEME=2 PIE_BLOCK_BYTES=84 PIE_SLICES=2 PIE_TIER=subgroup PIE_TARGET=spirv
// pie:instantiate linear_kquant_q3k_bf16 PIE_SCHEME=3 PIE_BLOCK_BYTES=110 PIE_SLICES=2 PIE_TIER=subgroup PIE_TARGET=spirv
// pie:instantiate linear_kquant_q4k_bf16 PIE_SCHEME=4 PIE_BLOCK_BYTES=144 PIE_SLICES=2 PIE_TIER=subgroup PIE_TARGET=spirv
// pie:instantiate linear_kquant_q5k_bf16 PIE_SCHEME=5 PIE_BLOCK_BYTES=176 PIE_SLICES=2 PIE_TIER=subgroup PIE_TARGET=spirv
// pie:instantiate linear_kquant_q6k_bf16 PIE_SCHEME=6 PIE_BLOCK_BYTES=210 PIE_SLICES=2 PIE_TIER=subgroup PIE_TARGET=spirv
// pie:instantiate linear_kquant_s4_q2k_bf16 PIE_SCHEME=2 PIE_BLOCK_BYTES=84 PIE_SLICES=4 PIE_TIER=subgroup PIE_TARGET=spirv
// pie:instantiate linear_kquant_s4_q3k_bf16 PIE_SCHEME=3 PIE_BLOCK_BYTES=110 PIE_SLICES=4 PIE_TIER=subgroup PIE_TARGET=spirv
// pie:instantiate linear_kquant_s4_q4k_bf16 PIE_SCHEME=4 PIE_BLOCK_BYTES=144 PIE_SLICES=4 PIE_TIER=subgroup PIE_TARGET=spirv
// pie:instantiate linear_kquant_s4_q5k_bf16 PIE_SCHEME=5 PIE_BLOCK_BYTES=176 PIE_SLICES=4 PIE_TIER=subgroup PIE_TARGET=spirv
// pie:instantiate linear_kquant_s4_q6k_bf16 PIE_SCHEME=6 PIE_BLOCK_BYTES=210 PIE_SLICES=4 PIE_TIER=subgroup PIE_TARGET=spirv
