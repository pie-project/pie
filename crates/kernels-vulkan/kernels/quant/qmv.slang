#include "common/bf16.slang"
#include "common/affine.slang"

#define PIE_LANES 32u
#define PIE_SLICES 4u
#define PIE_ROWS_PER_SLICE 2u
#define PIE_VROWS 4u
#if PIE_BITS == 2
#define PIE_CHUNK_WORDS 2u
#else
#define PIE_CHUNK_WORDS 4u
#endif
#define PIE_CHUNK_CODES (PIE_CHUNK_WORDS * uint(PIE_CODES_PER_WORD))

[[vk::binding(0)]] StructuredBuffer<uint> w;
PIE_BUFFER_RO(1, scales);
PIE_BUFFER_RO(2, biases);
[[vk::binding(3)]] StructuredBuffer<uint> x2;
PIE_BUFFER_RW(4, y);

struct Push {
    int in_vec_size;
    int out_vec_size;
    int vecs;
};
[[vk::push_constant]] ConstantBuffer<Push> pc;

groupshared float partials[PIE_SLICES * PIE_ROWS_PER_SLICE * PIE_VROWS * PIE_LANES];

[numthreads(32, 4, 1)]
void main(uint3 group : SV_GroupID, uint3 local : SV_GroupThreadID) {
    uint lid = local.x;
    uint ly = local.y;
    uint out0 = group.x * (PIE_SLICES * PIE_ROWS_PER_SLICE) + ly * PIE_ROWS_PER_SLICE;
    uint k = uint(pc.in_vec_size);
    uint n = uint(pc.out_vec_size);

    uint vec0 = group.y * PIE_VROWS;
    uint m = min(uint(pc.vecs) - min(uint(pc.vecs), vec0), PIE_VROWS);
    uint cpw = uint(PIE_CODES_PER_WORD);
    uint wpr = k * uint(PIE_BITS) / 32u;
    uint gpr = k / uint(PIE_GROUP);
    uint chunks = wpr / PIE_CHUNK_WORDS;
    uint xw = k / 2u;

    float acc[PIE_ROWS_PER_SLICE][PIE_VROWS];
    [ForceUnroll] for (uint r = 0u; r < PIE_ROWS_PER_SLICE; r++)
        [ForceUnroll] for (uint v = 0u; v < PIE_VROWS; v++) acc[r][v] = 0.0;

    for (uint c = lid; c < chunks; c += PIE_LANES) {
        uint kb = c * PIE_CHUNK_CODES;
        uint g = kb / uint(PIE_GROUP);

        float xs[PIE_VROWS][PIE_CHUNK_CODES];
        float xsum[PIE_VROWS];
        [ForceUnroll] for (uint v = 0u; v < PIE_VROWS; v++) {
            xsum[v] = 0.0;
            if (v < m) {
                [ForceUnroll] for (uint p = 0u; p < PIE_CHUNK_CODES / 2u; p++) {
                    uint pair = x2[(vec0 + v) * xw + kb / 2u + p];
                    xs[v][2u * p] = asfloat(pair << 16u);
                    xs[v][2u * p + 1u] = asfloat(pair & 0xffff0000u);
                    xsum[v] += xs[v][2u * p] + xs[v][2u * p + 1u];
                }
            }
        }
        [ForceUnroll] for (uint r = 0u; r < PIE_ROWS_PER_SLICE; r++) {
            uint row = out0 + r;
            if (row >= n) continue;
            float sc = PIE_LOAD(scales[row * gpr + g]);
            float bi = PIE_LOAD(biases[row * gpr + g]);
            uint wat = row * wpr + c * PIE_CHUNK_WORDS;
            float dots[PIE_VROWS];
            [ForceUnroll] for (uint v = 0u; v < PIE_VROWS; v++) dots[v] = 0.0;
            [ForceUnroll] for (uint wi = 0u; wi < PIE_CHUNK_WORDS; wi++) {
                uint word = w[wat + wi];
                [ForceUnroll] for (uint q = 0u; q < cpw; q++) {
                    float code = pie_affine_code(word, q);
                    [ForceUnroll] for (uint v = 0u; v < PIE_VROWS; v++) {
                        dots[v] += code * xs[v][wi * cpw + q];
                    }
                }
            }
            [ForceUnroll] for (uint v = 0u; v < PIE_VROWS; v++) {
                acc[r][v] += sc * dots[v] + bi * xsum[v];
            }
        }
    }
    [ForceUnroll] for (uint r = 0u; r < PIE_ROWS_PER_SLICE; r++)
        [ForceUnroll] for (uint v = 0u; v < PIE_VROWS; v++)
            partials[((ly * PIE_ROWS_PER_SLICE + r) * PIE_VROWS + v) * PIE_LANES + lid] = acc[r][v];
    GroupMemoryBarrierWithGroupSync();
    [ForceUnroll] for (uint step = PIE_LANES / 2u; step > 0u; step >>= 1u) {
        if (lid < step) {
            [ForceUnroll] for (uint r = 0u; r < PIE_ROWS_PER_SLICE; r++)
                [ForceUnroll] for (uint v = 0u; v < PIE_VROWS; v++) {
                    uint at = ((ly * PIE_ROWS_PER_SLICE + r) * PIE_VROWS + v) * PIE_LANES;
                    partials[at + lid] += partials[at + lid + step];
                }
        }
        GroupMemoryBarrierWithGroupSync();
    }
    if (lid == 0u) {
        [ForceUnroll] for (uint r = 0u; r < PIE_ROWS_PER_SLICE; r++) {
            uint row = out0 + r;
            if (row >= n) continue;
            [ForceUnroll] for (uint v = 0u; v < PIE_VROWS; v++) {
                if (v < m) y[(vec0 + v) * n + row] = PIE_STORE(partials[((ly * PIE_ROWS_PER_SLICE + r) * PIE_VROWS + v) * PIE_LANES]);
            }
        }
    }
}

// pie:instantiate affine_qmv_bf16_gs_32_b_2 PIE_GROUP=32 PIE_BITS=2
// pie:instantiate affine_qmv_bf16_gs_32_b_4 PIE_GROUP=32 PIE_BITS=4
// pie:instantiate affine_qmv_bf16_gs_32_b_8 PIE_GROUP=32 PIE_BITS=8
// pie:instantiate affine_qmv_bf16_gs_64_b_2 PIE_GROUP=64 PIE_BITS=2
// pie:instantiate affine_qmv_bf16_gs_64_b_4 PIE_GROUP=64 PIE_BITS=4
// pie:instantiate affine_qmv_bf16_gs_64_b_8 PIE_GROUP=64 PIE_BITS=8
// pie:instantiate affine_qmv_bf16_gs_128_b_2 PIE_GROUP=128 PIE_BITS=2
// pie:instantiate affine_qmv_bf16_gs_128_b_4 PIE_GROUP=128 PIE_BITS=4
// pie:instantiate affine_qmv_bf16_gs_128_b_8 PIE_GROUP=128 PIE_BITS=8
