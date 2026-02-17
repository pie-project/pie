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
//   5. moe_batched_matmul_swiglu — decode GEMM1 batched over K active experts
//   6. moe_batched_matmul        — decode GEMM2 batched over K active experts
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

// =============================================================================
// Kernel 5: moe_batched_matmul_swiglu — batched decode GEMM1
// =============================================================================
// Processes K active experts in a single dispatch (single token).
// Grid: (K, I, 1), Group: (1, min(I, 256), 1)
//
// Params: [H, I, has_bias, K]
// expert_params: [K, 5] with [alpha, beta, clamp_limit, scale_gate, scale_up]

kernel void moe_batched_matmul_swiglu(
    device const bfloat*    input          [[buffer(0)]],  // [1, H]
    device const uint8_t*   all_w_blocks   [[buffer(1)]],  // [E, 2*I, H/2] flattened
    device const uint8_t*   all_w_scales   [[buffer(2)]],  // [E, 2*I, H/32] flattened
    device const bfloat*    all_bias       [[buffer(3)]],  // [E, 2*I] flattened or dummy
    device bfloat*          output         [[buffer(4)]],  // [K, I]
    device const float*     params_raw     [[buffer(5)]],  // [H, I, has_bias, K]
    device const int*       expert_ids     [[buffer(6)]],  // [K]
    device const float*     expert_params  [[buffer(7)]],  // [K, 5]
    uint3 gid [[thread_position_in_grid]]
) {
    const int H         = (int)params_raw[0];
    const int I         = (int)params_raw[1];
    const int has_bias  = (int)params_raw[2];
    const int K         = (int)params_raw[3];

    const int k_idx   = gid.x;   // which active expert [0, K)
    const int out_col = gid.y;   // output column [0, I)

    if (k_idx >= K || out_col >= I) return;

    const int expert = expert_ids[k_idx];

    // Per-expert params
    const float alpha      = expert_params[k_idx * 5 + 0];
    const float clamp_l    = expert_params[k_idx * 5 + 2];
    const float scale_gate = expert_params[k_idx * 5 + 3];
    const float scale_up   = expert_params[k_idx * 5 + 4];

    const int H_half  = H / 2;
    const int H_scale = H / 32;

    // Expert's weight strides
    const long expert_blocks_stride = (long)(2 * I) * H_half;
    const long expert_scales_stride = (long)(2 * I) * H_scale;
    const long expert_bias_stride   = 2 * I;

    device const uint8_t* w_blocks = all_w_blocks + expert * expert_blocks_stride;
    device const uint8_t* w_scales = all_w_scales + expert * expert_scales_stride;

    const int up_row   = out_col;
    const int gate_row = I + out_col;

    float dot_up = 0.0f;
    float dot_gate = 0.0f;

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
            float x_lo = float(input[col]);
            float x_hi = float(input[col + 1]);

            dot_up   += x_lo * u_lo + x_hi * u_hi;
            dot_gate += x_lo * g_lo + x_hi * g_hi;
        }
    }

    if (has_bias) {
        device const bfloat* bias = all_bias + expert * expert_bias_stride;
        dot_up   += float(bias[up_row]);
        dot_gate += float(bias[gate_row]);
    }

    dot_up *= scale_gate;
    dot_gate *= scale_up;

    float gate_c = min(dot_gate, clamp_l);
    float up_c = clamp(dot_up, -clamp_l, clamp_l);
    float glu = gate_c / (1.0f + exp(-gate_c * alpha));

    output[k_idx * I + out_col] = bfloat((up_c + 1.0f) * glu);
}

// =============================================================================
// Kernel 6: moe_batched_matmul — batched decode GEMM2
// =============================================================================
// Processes K active experts in a single dispatch.
// Input is [K, in_dim] (one row per active expert from GEMM1 output).
// Grid: (K, out_dim, 1), Group: (1, min(out_dim, 256), 1)
//
// Params: [in_dim, out_dim, has_bias, K]
// expert_scales: [K] float32

kernel void moe_batched_matmul(
    device const bfloat*    input          [[buffer(0)]],  // [K, in_dim]
    device const uint8_t*   all_w_blocks   [[buffer(1)]],  // [E, out_dim, in_dim/2] flattened
    device const uint8_t*   all_w_scales   [[buffer(2)]],  // [E, out_dim, in_dim/32] flattened
    device const bfloat*    all_bias       [[buffer(3)]],  // [E, out_dim] flattened or dummy
    device bfloat*          output         [[buffer(4)]],  // [K, out_dim]
    device const float*     params_raw     [[buffer(5)]],  // [in_dim, out_dim, has_bias, K]
    device const int*       expert_ids     [[buffer(6)]],  // [K]
    device const float*     expert_scales  [[buffer(7)]],  // [K]
    uint3 gid [[thread_position_in_grid]]
) {
    const int in_dim   = (int)params_raw[0];
    const int out_dim  = (int)params_raw[1];
    const int has_bias = (int)params_raw[2];
    const int K        = (int)params_raw[3];

    const int k_idx   = gid.x;   // which active expert [0, K)
    const int out_col = gid.y;   // output column [0, out_dim)

    if (k_idx >= K || out_col >= out_dim) return;

    const int expert = expert_ids[k_idx];
    const float scale = expert_scales[k_idx];

    const int in_half  = in_dim / 2;
    const int in_scale = in_dim / 32;

    const long expert_blocks_stride = (long)out_dim * in_half;
    const long expert_scales_stride = (long)out_dim * in_scale;
    const long expert_bias_stride   = out_dim;

    device const uint8_t* w_blocks = all_w_blocks + expert * expert_blocks_stride;
    device const uint8_t* w_scales = all_w_scales + expert * expert_scales_stride;

    float dot = 0.0f;
    for (int blk = 0; blk < in_scale; ++blk) {
        float s = e8m0_to_float(w_scales[out_col * in_scale + blk]);
        int byte_start = blk * 16;

        for (int b = 0; b < 16; ++b) {
            uint8_t packed = w_blocks[out_col * in_half + byte_start + b];
            float w_lo = fp4_lut[packed & 0x0F] * s;
            float w_hi = fp4_lut[packed >> 4] * s;

            int col = blk * 32 + b * 2;
            dot += float(input[k_idx * in_dim + col]) * w_lo;
            dot += float(input[k_idx * in_dim + col + 1]) * w_hi;
        }
    }

    if (has_bias) {
        device const bfloat* bias = all_bias + expert * expert_bias_stride;
        dot += float(bias[out_col]);
    }

    output[k_idx * out_dim + out_col] = bfloat(dot * scale);
}

// =============================================================================
// Kernel 7: moe_batched_matmul_swiglu_simd — SIMD-cooperative batched GEMM1
// =============================================================================
// Uses SIMD-cooperative reduction: 32 lanes split H_scale blocks per output col.
// Threadgroup memory caches input vector (shared across all simdgroups).
// 8 simdgroups = 8 output columns per threadgroup.
//
// Grid: (K, ceil(I/8), 1), Group: (1, 256, 1)
// Threadgroup memory: H * sizeof(float)
//
// Params: [H, I, has_bias, K]
// expert_params: [K, 5]

constant constexpr int SIMD_WIDTH = 32;
constant constexpr int COLS_PER_TG = 8;
constant constexpr int TG_SIZE = COLS_PER_TG * SIMD_WIDTH;  // 256

kernel void moe_batched_matmul_swiglu_simd(
    device const bfloat*    input          [[buffer(0)]],  // [1, H]
    device const uint8_t*   all_w_blocks   [[buffer(1)]],  // [E, 2*I, H/2]
    device const uint8_t*   all_w_scales   [[buffer(2)]],  // [E, 2*I, H/32]
    device const bfloat*    all_bias       [[buffer(3)]],  // [E, 2*I] or dummy
    device bfloat*          output         [[buffer(4)]],  // [K, I]
    device const float*     params_raw     [[buffer(5)]],  // [H, I, has_bias, K]
    device const int*       expert_ids     [[buffer(6)]],  // [K]
    device const float*     expert_params  [[buffer(7)]],  // [K, 5]
    uint3 tgid  [[threadgroup_position_in_grid]],
    uint  sgid  [[simdgroup_index_in_threadgroup]],
    uint  slid  [[thread_index_in_simdgroup]],
    uint  lid   [[thread_index_in_threadgroup]]
) {
    const int H        = (int)params_raw[0];
    const int I        = (int)params_raw[1];
    const int has_bias = (int)params_raw[2];
    const int K_count  = (int)params_raw[3];

    const int k_idx   = tgid.x;
    const int out_col = tgid.y * COLS_PER_TG + sgid;

    if (k_idx >= K_count) return;

    const int expert = expert_ids[k_idx];

    // Cooperatively load input[H] into threadgroup memory
    threadgroup float shared_input[3072];  // max H supported (>2944)
    for (int i = (int)lid; i < H; i += TG_SIZE) {
        shared_input[i] = float(input[i]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (out_col >= I) return;

    // Per-expert params
    const float alpha      = expert_params[k_idx * 5 + 0];
    const float clamp_l    = expert_params[k_idx * 5 + 2];
    const float scale_gate = expert_params[k_idx * 5 + 3];
    const float scale_up   = expert_params[k_idx * 5 + 4];

    const int H_half  = H / 2;
    const int H_scale = H / 32;

    const long expert_blocks_stride = (long)(2 * I) * H_half;
    const long expert_scales_stride = (long)(2 * I) * H_scale;

    device const uint8_t* w_blocks = all_w_blocks + expert * expert_blocks_stride;
    device const uint8_t* w_scales = all_w_scales + expert * expert_scales_stride;

    const int up_row   = out_col;
    const int gate_row = I + out_col;

    // Vectorized weight pointers (uint4 = 16 bytes = one FP4 block)
    device const uint4* w_up_vec   = (device const uint4*)(w_blocks + up_row * H_half);
    device const uint4* w_gate_vec = (device const uint4*)(w_blocks + gate_row * H_half);

    // SIMD-cooperative: each lane handles blocks stride-32 apart
    float partial_up = 0.0f;
    float partial_gate = 0.0f;

    for (int blk = (int)slid; blk < H_scale; blk += SIMD_WIDTH) {
        float scale_u = e8m0_to_float(w_scales[up_row * H_scale + blk]);
        float scale_g = e8m0_to_float(w_scales[gate_row * H_scale + blk]);

        // Single 16-byte load per row (32 FP4 values)
        uint4 chunk_u = w_up_vec[blk];
        uint4 chunk_g = w_gate_vec[blk];

        int col = blk * 32;
        // Process 4 words × 4 bytes × 2 nibbles = 32 FP4 values
        uint u_words[4] = {chunk_u.x, chunk_u.y, chunk_u.z, chunk_u.w};
        uint g_words[4] = {chunk_g.x, chunk_g.y, chunk_g.z, chunk_g.w};
        for (int w = 0; w < 4; ++w) {
            for (int i = 0; i < 4; ++i) {
                uint8_t pu = (u_words[w] >> (i * 8)) & 0xFF;
                uint8_t pg = (g_words[w] >> (i * 8)) & 0xFF;
                float u_lo = fp4_lut[pu & 0x0F] * scale_u;
                float u_hi = fp4_lut[pu >> 4] * scale_u;
                float g_lo = fp4_lut[pg & 0x0F] * scale_g;
                float g_hi = fp4_lut[pg >> 4] * scale_g;
                int c = col + (w * 4 + i) * 2;
                partial_up   += shared_input[c] * u_lo + shared_input[c + 1] * u_hi;
                partial_gate += shared_input[c] * g_lo + shared_input[c + 1] * g_hi;
            }
        }
    }

    float dot_up   = simd_sum(partial_up);
    float dot_gate = simd_sum(partial_gate);

    if (slid == 0) {
        if (has_bias) {
            device const bfloat* bias = all_bias + expert * (long)(2 * I);
            dot_up   += float(bias[up_row]);
            dot_gate += float(bias[gate_row]);
        }

        dot_up *= scale_gate;
        dot_gate *= scale_up;

        float gate_c = min(dot_gate, clamp_l);
        float up_c = clamp(dot_up, -clamp_l, clamp_l);
        float glu = gate_c / (1.0f + exp(-gate_c * alpha));

        output[k_idx * I + out_col] = bfloat((up_c + 1.0f) * glu);
    }
}

// =============================================================================
// Kernel 8: moe_batched_matmul_simd — SIMD-cooperative batched GEMM2
// =============================================================================
// Grid: (K, ceil(out_dim/8), 1), Group: (1, 256, 1)
// Threadgroup memory: in_dim * sizeof(float)
//
// Params: [in_dim, out_dim, has_bias, K]
// expert_scales: [K]

kernel void moe_batched_matmul_simd(
    device const bfloat*    input          [[buffer(0)]],  // [K, in_dim]
    device const uint8_t*   all_w_blocks   [[buffer(1)]],  // [E, out_dim, in_dim/2]
    device const uint8_t*   all_w_scales   [[buffer(2)]],  // [E, out_dim, in_dim/32]
    device const bfloat*    all_bias       [[buffer(3)]],  // [E, out_dim] or dummy
    device bfloat*          output         [[buffer(4)]],  // [K, out_dim]
    device const float*     params_raw     [[buffer(5)]],  // [in_dim, out_dim, has_bias, K]
    device const int*       expert_ids     [[buffer(6)]],  // [K]
    device const float*     expert_scales  [[buffer(7)]],  // [K]
    uint3 tgid  [[threadgroup_position_in_grid]],
    uint  sgid  [[simdgroup_index_in_threadgroup]],
    uint  slid  [[thread_index_in_simdgroup]],
    uint  lid   [[thread_index_in_threadgroup]]
) {
    const int in_dim   = (int)params_raw[0];
    const int out_dim  = (int)params_raw[1];
    const int has_bias = (int)params_raw[2];
    const int K_count  = (int)params_raw[3];

    const int k_idx   = tgid.x;
    const int out_col = tgid.y * COLS_PER_TG + sgid;

    if (k_idx >= K_count) return;

    const int expert = expert_ids[k_idx];
    const float scale = expert_scales[k_idx];

    // Cooperatively load this expert's input row into threadgroup memory
    threadgroup float shared_input[3072];  // max in_dim supported
    for (int i = (int)lid; i < in_dim; i += TG_SIZE) {
        shared_input[i] = float(input[k_idx * in_dim + i]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (out_col >= out_dim) return;

    const int in_half  = in_dim / 2;
    const int in_scale = in_dim / 32;

    const long expert_blocks_stride = (long)out_dim * in_half;
    const long expert_scales_stride = (long)out_dim * in_scale;

    device const uint8_t* w_blocks = all_w_blocks + expert * expert_blocks_stride;
    device const uint8_t* w_scales = all_w_scales + expert * expert_scales_stride;

    // Vectorized weight pointer (uint4 = 16 bytes = one FP4 block)
    device const uint4* w_vec = (device const uint4*)(w_blocks + out_col * in_half);

    float partial = 0.0f;
    for (int blk = (int)slid; blk < in_scale; blk += SIMD_WIDTH) {
        float s = e8m0_to_float(w_scales[out_col * in_scale + blk]);

        // Single 16-byte load (32 FP4 values)
        uint4 chunk = w_vec[blk];
        int col = blk * 32;
        uint words[4] = {chunk.x, chunk.y, chunk.z, chunk.w};
        for (int w = 0; w < 4; ++w) {
            for (int i = 0; i < 4; ++i) {
                uint8_t packed = (words[w] >> (i * 8)) & 0xFF;
                float w_lo = fp4_lut[packed & 0x0F] * s;
                float w_hi = fp4_lut[packed >> 4] * s;
                int c = col + (w * 4 + i) * 2;
                partial += shared_input[c] * w_lo + shared_input[c + 1] * w_hi;
            }
        }
    }

    float dot = simd_sum(partial);

    if (slid == 0) {
        if (has_bias) {
            device const bfloat* bias = all_bias + expert * (long)out_dim;
            dot += float(bias[out_col]);
        }
        output[k_idx * out_dim + out_col] = bfloat(dot * scale);
    }
}

// =============================================================================
// Kernel 9: moe_batched_matmul_simd_fused — GEMM2 with fused weighted scatter
// =============================================================================
// Iterates over K experts per threadgroup, accumulates weighted results directly
// into [1, out_dim] output. Eliminates separate weighted-sum step.
//
// Grid: (1, ceil(out_dim/8), 1), Group: (1, 256, 1)
// Threadgroup memory: in_dim * sizeof(float)
//
// Params: [in_dim, out_dim, has_bias, K]
// expert_scales: [K] — pre-multiplied (output_scale * routing_weight)

kernel void moe_batched_matmul_simd_fused(
    device const bfloat*    input          [[buffer(0)]],  // [K, in_dim]
    device const uint8_t*   all_w_blocks   [[buffer(1)]],  // [E, out_dim, in_dim/2]
    device const uint8_t*   all_w_scales   [[buffer(2)]],  // [E, out_dim, in_dim/32]
    device const bfloat*    all_bias       [[buffer(3)]],  // [E, out_dim] or dummy
    device float*           output         [[buffer(4)]],  // [1, out_dim] float32
    device const float*     params_raw     [[buffer(5)]],  // [in_dim, out_dim, has_bias, K]
    device const int*       expert_ids     [[buffer(6)]],  // [K]
    device const float*     expert_scales  [[buffer(7)]],  // [K]
    uint3 tgid  [[threadgroup_position_in_grid]],
    uint  sgid  [[simdgroup_index_in_threadgroup]],
    uint  slid  [[thread_index_in_simdgroup]],
    uint  lid   [[thread_index_in_threadgroup]]
) {
    const int in_dim   = (int)params_raw[0];
    const int out_dim  = (int)params_raw[1];
    const int has_bias = (int)params_raw[2];
    const int K_count  = (int)params_raw[3];

    const int out_col = tgid.y * COLS_PER_TG + sgid;

    const int in_half  = in_dim / 2;
    const int in_scale = in_dim / 32;

    float accumulated = 0.0f;

    threadgroup float shared_input[3072];

    for (int k_idx = 0; k_idx < K_count; ++k_idx) {
        const int expert = expert_ids[k_idx];
        const float scale = expert_scales[k_idx];

        // Cooperatively load this expert's input row
        for (int i = (int)lid; i < in_dim; i += TG_SIZE) {
            shared_input[i] = float(input[k_idx * in_dim + i]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (out_col < out_dim) {
            const long expert_blocks_stride = (long)out_dim * in_half;
            const long expert_scales_stride = (long)out_dim * in_scale;

            device const uint8_t* w_blocks = all_w_blocks + expert * expert_blocks_stride;
            device const uint8_t* w_scales_ptr = all_w_scales + expert * expert_scales_stride;

            device const uint4* w_vec = (device const uint4*)(w_blocks + out_col * in_half);

            float partial = 0.0f;
            for (int blk = (int)slid; blk < in_scale; blk += SIMD_WIDTH) {
                float s = e8m0_to_float(w_scales_ptr[out_col * in_scale + blk]);

                uint4 chunk = w_vec[blk];
                int col = blk * 32;
                uint words[4] = {chunk.x, chunk.y, chunk.z, chunk.w};
                for (int w = 0; w < 4; ++w) {
                    for (int i = 0; i < 4; ++i) {
                        uint8_t packed = (words[w] >> (i * 8)) & 0xFF;
                        float w_lo = fp4_lut[packed & 0x0F] * s;
                        float w_hi = fp4_lut[packed >> 4] * s;
                        int c = col + (w * 4 + i) * 2;
                        partial += shared_input[c] * w_lo + shared_input[c + 1] * w_hi;
                    }
                }
            }

            float dot = simd_sum(partial);

            if (slid == 0) {
                if (has_bias) {
                    device const bfloat* bias = all_bias + expert * (long)out_dim;
                    dot += float(bias[out_col]);
                }
                accumulated += dot * scale;
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (slid == 0 && out_col < out_dim) {
        output[out_col] = accumulated;
    }
}
