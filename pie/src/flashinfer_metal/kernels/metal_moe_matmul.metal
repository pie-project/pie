// MoE FP4 Matmul Kernels for Apple Silicon
//
// Performs GEMM with inline FP4 dequantization from MXFP4 packed weights.
// Weight layout: blocks [rows, cols/2] uint8, scales [rows, cols/32] uint8 (E8M0)
//
// Four kernel variants:
//   1. moe_matmul_swiglu  — decode GEMM1 (per-token, fused SwiGLU)
//   2. moe_matmul         — decode GEMM2 (per-token, plain matmul)
//   3. moe_dense_matmul_swiglu — prefill GEMM1 (batched tokens, tiled)
//   4. moe_dense_matmul        — prefill GEMM2 (batched tokens, tiled)
//
// FP4 (E2M1) value table:
//   index  0: +0.0   1: +0.5   2: +1.0   3: +1.5
//         4: +2.0   5: +3.0   6: +4.0   7: +6.0
//         8: -0.0   9: -0.5  10: -1.0  11: -1.5
//        12: -2.0  13: -3.0  14: -4.0  15: -6.0
//
// E8M0 scale: value = 2^(exponent - 127), applied via as_type<float>((uint)exp << 23)

#include <metal_stdlib>
using namespace metal;

// FP4 LUT in constant memory
constant constexpr float fp4_lut[16] = {
    +0.0f, +0.5f, +1.0f, +1.5f, +2.0f, +3.0f, +4.0f, +6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f,
};

// Convert E8M0 exponent byte to float scale factor: 2^(exp - 127)
// This works by placing the exponent byte directly into the IEEE 754
// exponent field (bits 30..23).
inline float e8m0_to_float(uint8_t exp_byte) {
    uint32_t bits = (uint32_t)exp_byte << 23;
    return as_type<float>(bits);
}

// Dequantize one FP4 nibble with E8M0 scale
inline float dequant_fp4(uint8_t nibble, float scale) {
    return fp4_lut[nibble] * scale;
}

// Tile sizes for tiled kernels
constant constexpr int TILE_M = 16;
constant constexpr int TILE_N = 16;
constant constexpr int TILE_K = 32;  // Must be multiple of 32 for FP4 block alignment

// =============================================================================
// Kernel 1: moe_matmul_swiglu — per-token GEMM1 with fused GPT-OSS activation
// =============================================================================
// Computes: for each token t in [0, count):
//   g1 = x[t] @ W1^T   where W1 is [2*I, H] in FP4 packed
//   After deinterleave: first I rows = up, second I rows = gate
//   gate_proj = clamp(g1[I:], max=limit)
//   up_proj   = clamp(g1[:I], -limit, limit)
//   output[t] = (up_proj + 1) * gate_proj * σ(gate_proj * alpha)
//
// Params: [count, hidden_dim, intermediate_size, alpha, beta, clamp_limit,
//          output1_scale_gate, output1_scale_up]
// Grid: (count, intermediate_size, 1), Group: (1, min(I, 256), 1)

kernel void moe_matmul_swiglu(
    device const bfloat*    input       [[buffer(0)]],  // [count, hidden_dim]
    device const uint8_t*   w_blocks    [[buffer(1)]],  // [2*I, H/2] — one expert
    device const uint8_t*   w_scales    [[buffer(2)]],  // [2*I, H/32] — one expert
    device const bfloat*    bias        [[buffer(3)]],  // [2*I] or nullptr
    device bfloat*          output      [[buffer(4)]],  // [count, I]
    device const float*     params_raw  [[buffer(5)]],
    uint3 gid [[thread_position_in_grid]]
) {
    const int count     = (int)params_raw[0];
    const int H         = (int)params_raw[1];  // hidden_dim
    const int I         = (int)params_raw[2];  // intermediate_size
    const float alpha   = params_raw[3];
    const float beta    = params_raw[4];
    const float clamp_l = params_raw[5];
    const float scale_gate = params_raw[6];
    const float scale_up   = params_raw[7];
    const int has_bias  = (int)params_raw[8];

    const int token = gid.x;
    const int out_col = gid.y;  // output column in [0, I)

    if (token >= count || out_col >= I) return;

    const int H_half = H / 2;
    const int H_scale = H / 32;

    // After deinterleave: first I rows = up, second I rows = gate
    float dot_up = 0.0f;
    float dot_gate = 0.0f;

    const int up_row = out_col;         // first I rows = up projection
    const int gate_row = I + out_col;   // second I rows = gate projection

    for (int blk = 0; blk < H_scale; ++blk) {
        float scale_u = e8m0_to_float(w_scales[up_row * H_scale + blk]);
        float scale_g = e8m0_to_float(w_scales[gate_row * H_scale + blk]);

        int byte_start = blk * 16;  // 16 bytes = 32 FP4 values
        for (int b = 0; b < 16; ++b) {
            uint8_t packed_u = w_blocks[up_row * H_half + byte_start + b];
            uint8_t packed_g = w_blocks[gate_row * H_half + byte_start + b];

            float u_lo = fp4_lut[packed_u & 0x0F] * scale_u;
            float u_hi = fp4_lut[packed_u >> 4] * scale_u;
            float g_lo = fp4_lut[packed_g & 0x0F] * scale_g;
            float g_hi = fp4_lut[packed_g >> 4] * scale_g;

            int col = blk * 32 + b * 2;
            float x_lo = float(input[token * H + col]);
            float x_hi = float(input[token * H + col + 1]);

            dot_up   += x_lo * u_lo + x_hi * u_hi;
            dot_gate += x_lo * g_lo + x_hi * g_hi;
        }
    }

    // Add bias
    if (has_bias) {
        dot_up   += float(bias[up_row]);
        dot_gate += float(bias[gate_row]);
    }

    // Apply output scales
    dot_up *= scale_gate;
    dot_gate *= scale_up;

    // GPT-OSS activation (matching HuggingFace reference):
    // 1. Pre-activation clamp: gate max=limit, up ±limit
    float gate_c = min(dot_gate, clamp_l);
    float up_c = clamp(dot_up, -clamp_l, clamp_l);

    // 2. gate * σ(gate * α)
    float glu = gate_c / (1.0f + exp(-gate_c * alpha));

    // 3. output = (up + 1) * glu
    output[token * I + out_col] = bfloat((up_c + 1.0f) * glu);
}

// =============================================================================
// Kernel 2: moe_matmul — per-token GEMM2 (plain matmul)
// =============================================================================
// Computes: output[t] = input[t] @ W2^T  where W2 is [H_out, I_in] in FP4
//
// Params: [count, in_dim, out_dim, scale, has_bias]
// Grid: (count, out_dim, 1)

kernel void moe_matmul(
    device const bfloat*    input       [[buffer(0)]],  // [count, in_dim]
    device const uint8_t*   w_blocks    [[buffer(1)]],  // [out_dim, in_dim/2]
    device const uint8_t*   w_scales    [[buffer(2)]],  // [out_dim, in_dim/32]
    device const bfloat*    bias        [[buffer(3)]],  // [out_dim] or nullptr
    device bfloat*          output      [[buffer(4)]],  // [count, out_dim]
    device const float*     params_raw  [[buffer(5)]],
    uint3 gid [[thread_position_in_grid]]
) {
    const int count   = (int)params_raw[0];
    const int in_dim  = (int)params_raw[1];
    const int out_dim = (int)params_raw[2];
    const float scale = params_raw[3];
    const int has_bias = (int)params_raw[4];

    const int token = gid.x;
    const int out_col = gid.y;

    if (token >= count || out_col >= out_dim) return;

    const int in_half = in_dim / 2;
    const int in_scale = in_dim / 32;

    float dot = 0.0f;
    for (int blk = 0; blk < in_scale; ++blk) {
        float s = e8m0_to_float(w_scales[out_col * in_scale + blk]);
        int byte_start = blk * 16;

        for (int b = 0; b < 16; ++b) {
            uint8_t packed = w_blocks[out_col * in_half + byte_start + b];
            float w_lo = fp4_lut[packed & 0x0F] * s;
            float w_hi = fp4_lut[packed >> 4] * s;

            int col = blk * 32 + b * 2;
            dot += float(input[token * in_dim + col]) * w_lo;
            dot += float(input[token * in_dim + col + 1]) * w_hi;
        }
    }

    if (has_bias) {
        dot += float(bias[out_col]);
    }

    output[token * out_dim + out_col] = bfloat(dot * scale);
}

// =============================================================================
// Kernel 3: moe_dense_matmul_swiglu — prefill GEMM1 (tiled, fused GPT-OSS act)
// =============================================================================
// Same computation as moe_matmul_swiglu but uses shared memory tiling for
// better performance when count > 1.
//
// Grid: (ceil(count/TILE_M) * TILE_M, ceil(I/TILE_N) * TILE_N, 1)
// Group: (TILE_M, TILE_N, 1)

kernel void moe_dense_matmul_swiglu(
    device const bfloat*    input       [[buffer(0)]],
    device const uint8_t*   w_blocks    [[buffer(1)]],
    device const uint8_t*   w_scales    [[buffer(2)]],
    device const bfloat*    bias        [[buffer(3)]],
    device bfloat*          output      [[buffer(4)]],
    device const float*     params_raw  [[buffer(5)]],
    uint3 gid [[thread_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]],
    uint3 tid [[threadgroup_position_in_grid]]
) {
    const int count     = (int)params_raw[0];
    const int H         = (int)params_raw[1];
    const int I         = (int)params_raw[2];
    const float alpha   = params_raw[3];
    const float beta    = params_raw[4];
    const float clamp_l = params_raw[5];
    const float scale_gate = params_raw[6];
    const float scale_up   = params_raw[7];
    const int has_bias  = (int)params_raw[8];

    const int row = tid.x * TILE_M + lid.x;  // token index
    const int out_col = tid.y * TILE_N + lid.y;  // output column in [0, I)

    if (row >= count || out_col >= I) return;

    const int H_half = H / 2;
    const int H_scale = H / 32;

    // After deinterleave: first I rows = up, second I rows = gate
    float dot_up = 0.0f;
    float dot_gate = 0.0f;

    const int up_row = out_col;         // first I rows = up projection
    const int gate_row = I + out_col;   // second I rows = gate projection

    for (int blk = 0; blk < H_scale; ++blk) {
        float scale_u = e8m0_to_float(w_scales[up_row * H_scale + blk]);
        float scale_g = e8m0_to_float(w_scales[gate_row * H_scale + blk]);

        int byte_start = blk * 16;
        for (int b = 0; b < 16; ++b) {
            uint8_t packed_u = w_blocks[up_row * H_half + byte_start + b];
            uint8_t packed_g = w_blocks[gate_row * H_half + byte_start + b];

            float u_lo = fp4_lut[packed_u & 0x0F] * scale_u;
            float u_hi = fp4_lut[packed_u >> 4] * scale_u;
            float g_lo = fp4_lut[packed_g & 0x0F] * scale_g;
            float g_hi = fp4_lut[packed_g >> 4] * scale_g;

            int col = blk * 32 + b * 2;
            float x_lo = float(input[row * H + col]);
            float x_hi = float(input[row * H + col + 1]);

            dot_up   += x_lo * u_lo + x_hi * u_hi;
            dot_gate += x_lo * g_lo + x_hi * g_hi;
        }
    }

    if (has_bias) {
        dot_up   += float(bias[up_row]);
        dot_gate += float(bias[gate_row]);
    }

    dot_up *= scale_gate;
    dot_gate *= scale_up;

    // GPT-OSS activation (matching HuggingFace reference)
    float gate_c = min(dot_gate, clamp_l);
    float up_c = clamp(dot_up, -clamp_l, clamp_l);
    float glu = gate_c / (1.0f + exp(-gate_c * alpha));
    output[row * I + out_col] = bfloat((up_c + 1.0f) * glu);
}

// =============================================================================
// Kernel 4: moe_dense_matmul — prefill GEMM2 (tiled, plain matmul)
// =============================================================================

kernel void moe_dense_matmul(
    device const bfloat*    input       [[buffer(0)]],
    device const uint8_t*   w_blocks    [[buffer(1)]],
    device const uint8_t*   w_scales    [[buffer(2)]],
    device const bfloat*    bias        [[buffer(3)]],
    device bfloat*          output      [[buffer(4)]],
    device const float*     params_raw  [[buffer(5)]],
    uint3 gid [[thread_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]],
    uint3 tid [[threadgroup_position_in_grid]]
) {
    const int count   = (int)params_raw[0];
    const int in_dim  = (int)params_raw[1];
    const int out_dim = (int)params_raw[2];
    const float scale = params_raw[3];
    const int has_bias = (int)params_raw[4];

    const int row = tid.x * TILE_M + lid.x;
    const int out_col = tid.y * TILE_N + lid.y;

    if (row >= count || out_col >= out_dim) return;

    const int in_half = in_dim / 2;
    const int in_scale = in_dim / 32;

    float dot = 0.0f;
    for (int blk = 0; blk < in_scale; ++blk) {
        float s = e8m0_to_float(w_scales[out_col * in_scale + blk]);
        int byte_start = blk * 16;

        for (int b = 0; b < 16; ++b) {
            uint8_t packed = w_blocks[out_col * in_half + byte_start + b];
            float w_lo = fp4_lut[packed & 0x0F] * s;
            float w_hi = fp4_lut[packed >> 4] * s;

            int col = blk * 32 + b * 2;
            dot += float(input[row * in_dim + col]) * w_lo;
            dot += float(input[row * in_dim + col + 1]) * w_hi;
        }
    }

    if (has_bias) {
        dot += float(bias[out_col]);
    }

    output[row * out_dim + out_col] = bfloat(dot * scale);
}
