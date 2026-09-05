#include "common/bf16.slang"

#define PIE_LANES 32u
#define PIE_SLICES 2u
#define PIE_ROWS_PER_SLICE 2u
#define PIE_ROWS (PIE_SLICES * PIE_ROWS_PER_SLICE)

#define PIE_NVFP4_GROUP 16u

[[vk::binding(0)]] StructuredBuffer<uint8_t> codes;
[[vk::binding(1)]] StructuredBuffer<uint8_t> scales;
PIE_BUFFER_RO(2, x);
PIE_BUFFER_RW(3, y);

struct Push {
    int out_vec_size;
    int in_vec_size;
    int vecs;
    float tensor_scale;
};
[[vk::push_constant]] ConstantBuffer<Push> pc;

groupshared float partials[PIE_ROWS * PIE_LANES];

static const float kNvfp4Lut[16] = {
     0.0,  0.5,  1.0,  1.5,  2.0,  3.0,  4.0,  6.0,
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0};

float e4m3_to_f32(uint byte_) {
    int exp = int((byte_ >> 3u) & 0xfu);
    int mant = int(byte_ & 0x7u);
    float mag;
    if (exp == 0) {
        mag = float(mant) * 0.001953125;
    } else if (exp == 0xf && mant == 0x7) {
        mag = asfloat(0x7fc00000u);
    } else {
        mag = (1.0 + float(mant) * 0.125) * asfloat(uint((exp - 7 + 127) << 23));
    }
    return ((byte_ & 0x80u) != 0u) ? -mag : mag;
}

[numthreads(32, 2, 1)]
void main(uint3 group : SV_GroupID, uint3 local : SV_GroupThreadID) {
    uint lid = local.x;
    uint ly = local.y;
    uint token = group.y;
    uint n = uint(pc.out_vec_size);
    uint k = uint(pc.in_vec_size);
    uint groups = k / PIE_NVFP4_GROUP;
    uint code_bytes = k / 2u;
    uint out0 = group.x * PIE_ROWS + ly * PIE_ROWS_PER_SLICE;

    float acc[PIE_ROWS_PER_SLICE];
    [ForceUnroll] for (uint r = 0u; r < PIE_ROWS_PER_SLICE; r++) acc[r] = 0.0;

    for (uint g = lid; g < groups; g += PIE_LANES) {
        uint xb = token * k + g * PIE_NVFP4_GROUP;
        [ForceUnroll] for (uint r = 0u; r < PIE_ROWS_PER_SLICE; r++) {
            uint row = out0 + r;
            if (row >= n) continue;
            float sc = e4m3_to_f32(uint(scales[row * groups + g]));
            uint base = row * code_bytes + g * (PIE_NVFP4_GROUP / 2u);
            float part = 0.0;
            [ForceUnroll] for (uint j = 0u; j < PIE_NVFP4_GROUP / 2u; j++) {
                uint byte_ = uint(codes[base + j]);
                part += kNvfp4Lut[byte_ & 0x0fu] * PIE_LOAD(x[xb + 2u * j]);
                part += kNvfp4Lut[byte_ >> 4u] * PIE_LOAD(x[xb + 2u * j + 1u]);
            }
            acc[r] += sc * part;
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
                y[token * n + row] = PIE_STORE(
                    pc.tensor_scale * partials[(ly * PIE_ROWS_PER_SLICE + r) * PIE_LANES]);
            }
        }
    }
}

// pie:instantiate nvfp4_qmv_bf16
