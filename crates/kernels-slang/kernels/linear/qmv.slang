#include "prim/affine.slang"
#include "prim/bf16.slang"
#include "prim/params.slang"
#include "prim/slot16.slang"

// y = dequant(w) x over a few activation rows, the decode-batch shape, from
// an affine bank. A workgroup is four 32-lane strips; a strip lands two
// output columns for `PIE_VROWS` activation rows, its lanes walking the
// contraction in chunks of `PIE_CHUNK_WORDS` code words and folding through
// workgroup memory. More rows a workgroup re-read the bank fewer times and
// hold more registers. SPIR-V lands a value a 16-bit slot; WGSL lands the
// strip's two columns as one word, so its host refuses an odd width.

#define PIE_LANES 32u
#define PIE_SLICES 4u
#define PIE_ROWS_PER_SLICE 2u
#if !defined(PIE_VROWS)
#define PIE_VROWS 4u
#endif
#if PIE_BITS == 2
#define PIE_CHUNK_WORDS 2u
#else
#define PIE_CHUNK_WORDS 4u
#endif
#define PIE_CHUNK_CODES (PIE_CHUNK_WORDS * (32u / uint(PIE_BITS)))

[[vk::binding(0)]] StructuredBuffer<uint> w;
[[vk::binding(1)]] StructuredBuffer<uint> scales;
[[vk::binding(2)]] StructuredBuffer<uint> biases;
[[vk::binding(3)]] StructuredBuffer<uint> x2;
#if defined(PIE_WGSL)
[[vk::binding(4)]] RWStructuredBuffer<uint> y;
#else
PIE_SLOTS_RW(4, y);
#endif

struct Params {
    int in_vec_size;
    int out_vec_size;
    int vecs;
};
PIE_PARAMS(5, Params, params);

groupshared float partials[PIE_SLICES * PIE_ROWS_PER_SLICE * PIE_VROWS * PIE_LANES];

[numthreads(32, 4, 1)]
void main(uint3 group : SV_GroupID, uint3 local : SV_GroupThreadID) {
    uint lid = local.x;
    uint ly = local.y;
    uint out0 = group.x * (PIE_SLICES * PIE_ROWS_PER_SLICE) + ly * PIE_ROWS_PER_SLICE;
    uint k = uint(params.in_vec_size);
    uint n = uint(params.out_vec_size);

    uint vec0 = group.y * PIE_VROWS;
    uint m = min(uint(params.vecs) - min(uint(params.vecs), vec0), PIE_VROWS);
    uint cpw = PIE_CODES_PER_WORD;
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
            float sc = bf16_at(scales, row * gpr + g);
            float bi = bf16_at(biases, row * gpr + g);
            uint wat = row * wpr + c * PIE_CHUNK_WORDS;
            float dots[PIE_VROWS];
            [ForceUnroll] for (uint v = 0u; v < PIE_VROWS; v++) dots[v] = 0.0;
            [ForceUnroll] for (uint wi = 0u; wi < PIE_CHUNK_WORDS; wi++) {
                uint word = w[wat + wi];
                [ForceUnroll] for (uint q = 0u; q < cpw; q++) {
                    float code = affine_code(word, q);
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
#if defined(PIE_WGSL)
        if (out0 < n) {
            [ForceUnroll] for (uint v = 0u; v < PIE_VROWS; v++) {
                if (v < m) {
                    uint at = ly * PIE_ROWS_PER_SLICE * PIE_VROWS + v;
                    y[((vec0 + v) * n + out0) >> 1] = pack_bf16(
                        partials[at * PIE_LANES], partials[(at + PIE_VROWS) * PIE_LANES]);
                }
            }
        }
#else
        [ForceUnroll] for (uint r = 0u; r < PIE_ROWS_PER_SLICE; r++) {
            uint row = out0 + r;
            if (row >= n) continue;
            [ForceUnroll] for (uint v = 0u; v < PIE_VROWS; v++) {
                if (v < m) {
                    slot_put(y, (vec0 + v) * n + row,
                             partials[((ly * PIE_ROWS_PER_SLICE + r) * PIE_VROWS + v) * PIE_LANES]);
                }
            }
        }
#endif
    }
}

// pie:instantiate linear_qmv_bf16_gs_32_b_2 PIE_GROUP=32 PIE_BITS=2
// pie:instantiate linear_qmv_bf16_gs_32_b_4 PIE_GROUP=32 PIE_BITS=4
// pie:instantiate linear_qmv_bf16_gs_32_b_8 PIE_GROUP=32 PIE_BITS=8
// pie:instantiate linear_qmv_bf16_gs_64_b_2 PIE_GROUP=64 PIE_BITS=2
// pie:instantiate linear_qmv_bf16_gs_64_b_4 PIE_GROUP=64 PIE_BITS=4
// pie:instantiate linear_qmv_bf16_gs_64_b_8 PIE_GROUP=64 PIE_BITS=8
// pie:instantiate linear_qmv_bf16_gs_128_b_2 PIE_GROUP=128 PIE_BITS=2
// pie:instantiate linear_qmv_bf16_gs_128_b_4 PIE_GROUP=128 PIE_BITS=4
// pie:instantiate linear_qmv_bf16_gs_128_b_8 PIE_GROUP=128 PIE_BITS=8
// The same at 1, 2 and 8 activation rows a workgroup.
// pie:instantiate linear_qmv_bf16_gs_32_b_2_v1 PIE_GROUP=32 PIE_BITS=2 PIE_VROWS=1u
// pie:instantiate linear_qmv_bf16_gs_32_b_2_v2 PIE_GROUP=32 PIE_BITS=2 PIE_VROWS=2u
// pie:instantiate linear_qmv_bf16_gs_32_b_2_v8 PIE_GROUP=32 PIE_BITS=2 PIE_VROWS=8u
// pie:instantiate linear_qmv_bf16_gs_32_b_4_v1 PIE_GROUP=32 PIE_BITS=4 PIE_VROWS=1u
// pie:instantiate linear_qmv_bf16_gs_32_b_4_v2 PIE_GROUP=32 PIE_BITS=4 PIE_VROWS=2u
// pie:instantiate linear_qmv_bf16_gs_32_b_4_v8 PIE_GROUP=32 PIE_BITS=4 PIE_VROWS=8u
// pie:instantiate linear_qmv_bf16_gs_32_b_8_v1 PIE_GROUP=32 PIE_BITS=8 PIE_VROWS=1u
// pie:instantiate linear_qmv_bf16_gs_32_b_8_v2 PIE_GROUP=32 PIE_BITS=8 PIE_VROWS=2u
// pie:instantiate linear_qmv_bf16_gs_32_b_8_v8 PIE_GROUP=32 PIE_BITS=8 PIE_VROWS=8u
// pie:instantiate linear_qmv_bf16_gs_64_b_2_v1 PIE_GROUP=64 PIE_BITS=2 PIE_VROWS=1u
// pie:instantiate linear_qmv_bf16_gs_64_b_2_v2 PIE_GROUP=64 PIE_BITS=2 PIE_VROWS=2u
// pie:instantiate linear_qmv_bf16_gs_64_b_2_v8 PIE_GROUP=64 PIE_BITS=2 PIE_VROWS=8u
// pie:instantiate linear_qmv_bf16_gs_64_b_4_v1 PIE_GROUP=64 PIE_BITS=4 PIE_VROWS=1u
// pie:instantiate linear_qmv_bf16_gs_64_b_4_v2 PIE_GROUP=64 PIE_BITS=4 PIE_VROWS=2u
// pie:instantiate linear_qmv_bf16_gs_64_b_4_v8 PIE_GROUP=64 PIE_BITS=4 PIE_VROWS=8u
// pie:instantiate linear_qmv_bf16_gs_64_b_8_v1 PIE_GROUP=64 PIE_BITS=8 PIE_VROWS=1u
// pie:instantiate linear_qmv_bf16_gs_64_b_8_v2 PIE_GROUP=64 PIE_BITS=8 PIE_VROWS=2u
// pie:instantiate linear_qmv_bf16_gs_64_b_8_v8 PIE_GROUP=64 PIE_BITS=8 PIE_VROWS=8u
// pie:instantiate linear_qmv_bf16_gs_128_b_2_v1 PIE_GROUP=128 PIE_BITS=2 PIE_VROWS=1u
// pie:instantiate linear_qmv_bf16_gs_128_b_2_v2 PIE_GROUP=128 PIE_BITS=2 PIE_VROWS=2u
// pie:instantiate linear_qmv_bf16_gs_128_b_2_v8 PIE_GROUP=128 PIE_BITS=2 PIE_VROWS=8u
// pie:instantiate linear_qmv_bf16_gs_128_b_4_v1 PIE_GROUP=128 PIE_BITS=4 PIE_VROWS=1u
// pie:instantiate linear_qmv_bf16_gs_128_b_4_v2 PIE_GROUP=128 PIE_BITS=4 PIE_VROWS=2u
// pie:instantiate linear_qmv_bf16_gs_128_b_4_v8 PIE_GROUP=128 PIE_BITS=4 PIE_VROWS=8u
// pie:instantiate linear_qmv_bf16_gs_128_b_8_v1 PIE_GROUP=128 PIE_BITS=8 PIE_VROWS=1u
// pie:instantiate linear_qmv_bf16_gs_128_b_8_v2 PIE_GROUP=128 PIE_BITS=8 PIE_VROWS=2u
// pie:instantiate linear_qmv_bf16_gs_128_b_8_v8 PIE_GROUP=128 PIE_BITS=8 PIE_VROWS=8u
