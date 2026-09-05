#include "common/bf16.slang"

#define PIE_LANES 32u
#define PIE_SLICES 2u
#define PIE_ROWS_PER_SLICE 2u
#define PIE_ROWS (PIE_SLICES * PIE_ROWS_PER_SLICE)

#define PIE_SUPER 256u

[[vk::binding(0)]] StructuredBuffer<uint8_t> w;
PIE_BUFFER_RO(1, x);
PIE_BUFFER_RW(2, y);

struct Push {
    int out_vec_size;
    int in_vec_size;
    int vecs;
};
[[vk::push_constant]] ConstantBuffer<Push> pc;

groupshared float partials[PIE_ROWS * PIE_LANES];

float gguf_f16(uint h) {
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

void q4k_scale_min(uint base, uint sub, out float scale, out float min_) {
    if (sub < 4u) {
        scale = float(uint(w[base + sub]) & 63u);
        min_ = float(uint(w[base + sub + 4u]) & 63u);
    } else {
        uint a = uint(w[base + sub + 4u]);
        uint b = uint(w[base + sub - 4u]);
        uint c = uint(w[base + sub]);
        scale = float((a & 0x0fu) | ((b >> 6u) << 4u));
        min_ = float((a >> 4u) | ((c >> 6u) << 4u));
    }
}

int q3k_scale(uint base, uint sub) {
    uint grp = sub >> 2u;
    uint j = sub & 3u;
    uint src = uint(w[base + (((grp & 1u) != 0u) ? 4u : 0u) + j]);
    uint low = (grp < 2u) ? (src & 0x0fu) : (src >> 4u);
    uint top = (uint(w[base + 8u + j]) >> (2u * grp)) & 3u;
    return int(low | (top << 4u));
}

[numthreads(32, 2, 1)]
void main(uint3 group : SV_GroupID, uint3 local : SV_GroupThreadID) {
    uint lid = local.x;
    uint ly = local.y;
    uint token = group.y;
    uint n = uint(pc.out_vec_size);
    uint k = uint(pc.in_vec_size);
    uint blocks = k / PIE_SUPER;
    uint row_bytes = blocks * uint(PIE_BLOCK_BYTES);
    uint out0 = group.x * PIE_ROWS + ly * PIE_ROWS_PER_SLICE;

    float acc[PIE_ROWS_PER_SLICE];
    [ForceUnroll] for (uint r = 0u; r < PIE_ROWS_PER_SLICE; r++) acc[r] = 0.0;

    for (uint g = lid; g < blocks; g += PIE_LANES) {
        uint xb = token * k + g * PIE_SUPER;
        [ForceUnroll] for (uint r = 0u; r < PIE_ROWS_PER_SLICE; r++) {
            uint row = out0 + r;
            if (row >= n) continue;
            uint base = row * row_bytes + g * uint(PIE_BLOCK_BYTES);

#if PIE_SCHEME == 2
            float d = gguf_f16(uint(w[base + 80u]) | (uint(w[base + 81u]) << 8u));
            float dmin = gguf_f16(uint(w[base + 82u]) | (uint(w[base + 83u]) << 8u));
            for (uint b = 0u; b < 16u; b++) {
                uint shift = 2u * ((b >> 1u) & 3u);
                uint at = 16u + (b >> 3u) * 32u + (b & 1u) * 16u;
                float part = 0.0;
                float xsum = 0.0;
                [ForceUnroll] for (uint l = 0u; l < 16u; l++) {
                    float xv = PIE_LOAD(x[xb + b * 16u + l]);
                    xsum += xv;
                    part += float((uint(w[base + at + l]) >> shift) & 3u) * xv;
                }
                uint packed = uint(w[base + b]);
                acc[r] += d * float(packed & 0x0fu) * part
                        - dmin * float(packed >> 4u) * xsum;
            }
#elif PIE_SCHEME == 3
            float d = gguf_f16(uint(w[base + 108u]) | (uint(w[base + 109u]) << 8u));
            for (uint b = 0u; b < 16u; b++) {
                uint step = (b >> 1u) & 3u;
                uint shift = 2u * step;
                uint selector = 1u << ((b >> 3u) * 4u + step);
                uint at = 32u + (b >> 3u) * 32u + (b & 1u) * 16u;
                uint mask_at = (b & 1u) * 16u;
                float part = 0.0;
                [ForceUnroll] for (uint l = 0u; l < 16u; l++) {
                    float xv = PIE_LOAD(x[xb + b * 16u + l]);
                    int code = int((uint(w[base + at + l]) >> shift) & 3u);
                    int borrow = ((uint(w[base + mask_at + l]) & selector) != 0u) ? 0 : 4;
                    part += float(code - borrow) * xv;
                }
                acc[r] += d * float(q3k_scale(base + 96u, b) - 32) * part;
            }
#elif PIE_SCHEME == 4
            float d = gguf_f16(uint(w[base]) | (uint(w[base + 1u]) << 8u));
            float dmin = gguf_f16(uint(w[base + 2u]) | (uint(w[base + 3u]) << 8u));
            for (uint b = 0u; b < 8u; b++) {
                uint pair = b >> 1u;
                bool high = (b & 1u) != 0u;
                float part = 0.0;
                float xsum = 0.0;
                [ForceUnroll] for (uint i = 0u; i < 32u; i++) {
                    float xv = PIE_LOAD(x[xb + b * 32u + i]);
                    xsum += xv;
                    uint byte_ = uint(w[base + 16u + pair * 32u + i]);
                    part += float(high ? (byte_ >> 4u) : (byte_ & 0x0fu)) * xv;
                }
                float sc;
                float mn;
                q4k_scale_min(base + 4u, b, sc, mn);
                acc[r] += d * sc * part - dmin * mn * xsum;
            }
#elif PIE_SCHEME == 5
            float d = gguf_f16(uint(w[base]) | (uint(w[base + 1u]) << 8u));
            float dmin = gguf_f16(uint(w[base + 2u]) | (uint(w[base + 3u]) << 8u));
            for (uint b = 0u; b < 8u; b++) {
                uint pair = b >> 1u;
                bool high = (b & 1u) != 0u;
                float part = 0.0;
                float xsum = 0.0;
                [ForceUnroll] for (uint i = 0u; i < 32u; i++) {
                    float xv = PIE_LOAD(x[xb + b * 32u + i]);
                    xsum += xv;
                    uint byte_ = uint(w[base + 48u + pair * 32u + i]);
                    uint low = high ? (byte_ >> 4u) : (byte_ & 0x0fu);
                    uint fifth = (uint(w[base + 16u + i]) >> b) & 1u;
                    part += float(low | (fifth << 4u)) * xv;
                }
                float sc;
                float mn;
                q4k_scale_min(base + 4u, b, sc, mn);
                acc[r] += d * sc * part - dmin * mn * xsum;
            }
#elif PIE_SCHEME == 6
            float d = gguf_f16(uint(w[base + 208u]) | (uint(w[base + 209u]) << 8u));
            for (uint half_ = 0u; half_ < 2u; half_++) {
                for (uint quarter = 0u; quarter < 4u; quarter++) {
                    for (uint sub = 0u; sub < 2u; sub++) {
                        float part = 0.0;
                        [ForceUnroll] for (uint t = 0u; t < 16u; t++) {
                            uint i = sub * 16u + t;
                            float xv = PIE_LOAD(x[xb + half_ * 128u + quarter * 32u + i]);
                            uint byte_ = uint(w[base + half_ * 64u + i + 32u * (quarter & 1u)]);
                            uint low = (quarter < 2u) ? (byte_ & 0x0fu) : (byte_ >> 4u);
                            uint top = (uint(w[base + 128u + half_ * 32u + i]) >> (2u * quarter)) & 3u;
                            part += float(int(low | (top << 4u)) - 32) * xv;
                        }
                        int raw = int(uint(w[base + 192u + half_ * 8u + sub + 2u * quarter]));
                        acc[r] += d * float((raw > 127) ? (raw - 256) : raw) * part;
                    }
                }
            }
#else
#error "PIE_SCHEME must be one of 2, 3, 4, 5, 6"
#endif
        }
    }

    [ForceUnroll] for (uint r = 0u; r < PIE_ROWS_PER_SLICE; r++)
        partials[(ly * PIE_ROWS_PER_SLICE + r) * PIE_LANES + lid] = acc[r];
    GroupMemoryBarrierWithGroupSync();
    for (uint s = PIE_LANES / 2u; s > 0u; s >>= 1u) {
        if (lid < s) {
            [ForceUnroll] for (uint r = 0u; r < PIE_ROWS_PER_SLICE; r++) {
                uint at = (ly * PIE_ROWS_PER_SLICE + r) * PIE_LANES;
                partials[at + lid] += partials[at + lid + s];
            }
        }
        GroupMemoryBarrierWithGroupSync();
    }
    if (lid == 0u) {
        [ForceUnroll] for (uint r = 0u; r < PIE_ROWS_PER_SLICE; r++) {
            uint row = out0 + r;
            if (row < n) {
                y[token * n + row] =
                    PIE_STORE(partials[(ly * PIE_ROWS_PER_SLICE + r) * PIE_LANES]);
            }
        }
    }
}

// pie:instantiate kquant_q2k_bf16 PIE_SCHEME=2 PIE_BLOCK_BYTES=84
// pie:instantiate kquant_q3k_bf16 PIE_SCHEME=3 PIE_BLOCK_BYTES=110
// pie:instantiate kquant_q4k_bf16 PIE_SCHEME=4 PIE_BLOCK_BYTES=144
// pie:instantiate kquant_q5k_bf16 PIE_SCHEME=5 PIE_BLOCK_BYTES=176
// pie:instantiate kquant_q6k_bf16 PIE_SCHEME=6 PIE_BLOCK_BYTES=210
