// MoE FP4 Matmul Kernels for Apple Silicon
//
// Performs GEMM with inline FP4 dequantization from MXFP4 packed weights.
// Weight layout: blocks [rows, cols/2] uint8, scales [rows, cols/32] uint8 (E8M0)
//
// Kernel variants:
//   Prefill (multi-token, per-expert dispatch):
//     moe_prefill_gemm1_swiglu        — GEMM1 with fused GPT-OSS activation
//     moe_prefill_gemm2               — GEMM2 plain matmul
//     moe_prefill_gemm1_swiglu_tiled  — GEMM1 tiled (shared memory, count > 1)
//     moe_prefill_gemm2_tiled         — GEMM2 tiled (shared memory, count > 1)
//
//   Decode (single-token, all-K-experts in one dispatch, SIMD K-split):
//     moe_decode_gemm1_swiglu   — GEMM1 with fused SwiGLU activation
//     moe_decode_gemm1          — GEMM1 raw matmul (bench/debug)
//     moe_decode_gemm2_fused    — GEMM2 fused across experts with routing scales
//
// FP4 (E2M1) value table:
//   index  0: +0.0   1: +0.5   2: +1.0   3: +1.5
//         4: +2.0   5: +3.0   6: +4.0   7: +6.0
//         8: -0.0   9: -0.5  10: -1.0  11: -1.5
//        12: -2.0  13: -3.0  14: -4.0  15: -6.0
//
// E8M0 scale: value = 2^(exponent - 127), applied via as_type<float>((uint)exp << 23)
//
// See MOE_OPTIMIZATION_HISTORY.md for the evolution of these kernels.

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

// Arithmetic FP4 dequant — no LUT, pure bit manipulation.
// Converts E2M1 nibble to float32 using IEEE 754 bit construction.
//   E2M1 format: [sign(1)][exp(2)][mant(1)]
//   Normal (E>0): (-1)^S × 2^(E-1) × (1 + M×0.5)
//   Subnormal (E=0): (-1)^S × M × 0.5
inline float fp4_to_float(uint nibble) {
    uint sign = nibble >> 3;
    uint exp  = (nibble >> 1) & 3u;
    uint mant = nibble & 1u;
    // Normal: fp32 exponent = E + 126, mantissa = M << 22
    uint normal_bits = ((exp + 126u) << 23) | (mant << 22);
    // Subnormal: only 0.0 (M=0) or 0.5 (M=1)
    uint subnorm_bits = mant * 0x3F000000u;
    uint result = select(subnorm_bits, normal_bits, exp > 0u);
    result |= (sign << 31);
    return as_type<float>(result);
}

// Tile sizes for tiled kernels
constant constexpr int TILE_M = 16;
constant constexpr int TILE_N = 16;
constant constexpr int TILE_K = 32;  // Must be multiple of 32 for FP4 block alignment


// =============================================================================
// PREFILL KERNELS (per-expert dispatch, multi-token)
// =============================================================================


// =============================================================================
// Kernel 1: moe_prefill_gemm1_swiglu — per-token GEMM1 with fused GPT-OSS activation
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

kernel void moe_prefill_gemm1_swiglu(
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
// Kernel 2: moe_prefill_gemm2 — per-token GEMM2 (plain matmul)
// =============================================================================
// Computes: output[t] = input[t] @ W2^T  where W2 is [H_out, I_in] in FP4
//
// Params: [count, in_dim, out_dim, scale, has_bias]
// Grid: (count, out_dim, 1)

kernel void moe_prefill_gemm2(
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
// Kernel 3: moe_prefill_gemm1_swiglu_tiled — prefill GEMM1 (tiled, fused GPT-OSS act)
// =============================================================================
// Same computation as moe_prefill_gemm1_swiglu but uses shared memory tiling for
// better performance when count > 1.
//
// Grid: (ceil(count/TILE_M) * TILE_M, ceil(I/TILE_N) * TILE_N, 1)
// Group: (TILE_M, TILE_N, 1)

kernel void moe_prefill_gemm1_swiglu_tiled(
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
// Kernel 4: moe_prefill_gemm2_tiled — prefill GEMM2 (tiled, plain matmul)
// =============================================================================

kernel void moe_prefill_gemm2_tiled(
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
// DECODE KERNELS: SIMD-parallel FP4 matvec (inspired by llama.cpp/ggml)
// =============================================================================
//
// These kernels split the K (reduction) dimension across 32 SIMD lanes
// dimension rather than the N (output) dimension. This distributes the FP4
// dequantization compute across threads, allowing memory latency hiding.
//
// Key techniques (adapted from llama.cpp's ggml-metal.metal):
//   - Threadgroup LUT: 16-float FP4 lookup table in threadgroup memory
//   - SIMD K-split: ix=tiisg/2 (block), it=tiisg%2 (half-block)
//   - Deferred scale: one E8M0 multiply per 32-element block
//   - float4 dot products: hardware dot() for 4-element accumulation
//   - simd_sum reduction: combines all 32 lanes after K loop
//
// See MOE_OPTIMIZATION_HISTORY.md for why this approach was chosen over
// alternatives (1-thread-per-output, simdgroup_multiply_accumulate, etc.)
//


// =============================================================================
// Kernel 14: moe_decode_gemm1_swiglu — SIMD K-split GEMM1 with fused SwiGLU
// =============================================================================
//
// Fuses GEMM1 matmul + SwiGLU activation in one kernel.
// Each simdgroup computes one output column by reading TWO weight rows:
//   up_row = out_col, gate_row = I + out_col
// Then applies: output = (clamp(up, -cl, cl) + 1) * gate_c * sigmoid(alpha * gate_c)
//   where gate_c = min(gate, cl)
//
// Grid: (ceil(I / 2), K_active, 1)
// ThreadsPerTG: (32, 2, 1)  [32 per SG, 2 SGs → 2 output cols per TG]
//
// Params buffer: [H_PAD, I, alpha, clamp_limit, has_bias]
//
kernel void moe_decode_gemm1_swiglu(
    device const bfloat*    input          [[buffer(0)]],  // [1, H_PAD]
    device const uint8_t*   all_w_blocks   [[buffer(1)]],  // [E, 2*I, H_PAD/2]
    device const uint8_t*   all_w_scales   [[buffer(2)]],  // [E, 2*I, H_PAD/32]
    device const bfloat*    all_bias       [[buffer(3)]],  // [E, 2*I] or dummy
    device bfloat*          output         [[buffer(4)]],  // [K, I]
    constant uint*          expert_ids     [[buffer(5)]],  // [K]
    constant float*         params         [[buffer(6)]],  // [H_PAD, I, alpha, clamp_limit, has_bias]
    uint3  tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]]
) {
    const uint H = (uint)params[0];           // H_PAD (input dim)
    const uint I = (uint)params[1];           // intermediate_size (output cols)
    const float alpha = params[2];            // swiglu alpha (e.g. 1.702)
    const float clamp_l = params[3];          // swiglu clamp limit
    const uint has_bias = (uint)params[4];    // 1.0 if bias present, 0.0 otherwise
    const uint N = 2 * I;                     // total weight rows per expert
    const uint H_half = H / 2;
    const uint H_scale = H / 32;
    const uint nb = H / 32;

    threadgroup float lut[16];
    if (sgitg == 0 && tiisg < 16) {
        lut[tiisg] = fp4_lut[tiisg];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint expert_idx = tgpig.y;
    const uint eid = expert_ids[expert_idx];
    const uint out_col = tgpig.x * 2 + sgitg;  // 2 output cols per TG

    if (out_col >= I) return;

    const uint up_row = out_col;
    const uint gate_row = I + out_col;

    device const uint8_t* wb = all_w_blocks + (long)eid * N * H_half;
    device const uint8_t* ws = all_w_scales + (long)eid * N * H_scale;

    const ushort ix = tiisg / 2;
    const ushort it = tiisg % 2;

    float sumf_up = 0.0f;
    float sumf_gate = 0.0f;

    for (uint ib = ix; ib < nb; ib += 16) {
        const uint k_base = ib * 32 + it * 16;
        device const bfloat* xp = input + k_base;

        float4 xf[4];
        xf[0] = float4(float(xp[0]),  float(xp[1]),  float(xp[2]),  float(xp[3]));
        xf[1] = float4(float(xp[4]),  float(xp[5]),  float(xp[6]),  float(xp[7]));
        xf[2] = float4(float(xp[8]),  float(xp[9]),  float(xp[10]), float(xp[11]));
        xf[3] = float4(float(xp[12]), float(xp[13]), float(xp[14]), float(xp[15]));

        // Up row weights
        device const uint8_t* qu = wb + (long)up_row * H_half + ib * 16 + it * 8;
        float4 up0 = float4(lut[qu[0]&0xF], lut[qu[0]>>4], lut[qu[1]&0xF], lut[qu[1]>>4]);
        float4 up1 = float4(lut[qu[2]&0xF], lut[qu[2]>>4], lut[qu[3]&0xF], lut[qu[3]>>4]);
        float4 up2 = float4(lut[qu[4]&0xF], lut[qu[4]>>4], lut[qu[5]&0xF], lut[qu[5]>>4]);
        float4 up3 = float4(lut[qu[6]&0xF], lut[qu[6]>>4], lut[qu[7]&0xF], lut[qu[7]>>4]);

        float up_dot = dot(xf[0], up0) + dot(xf[1], up1)
                     + dot(xf[2], up2) + dot(xf[3], up3);

        // Gate row weights
        device const uint8_t* qg = wb + (long)gate_row * H_half + ib * 16 + it * 8;
        float4 gt0 = float4(lut[qg[0]&0xF], lut[qg[0]>>4], lut[qg[1]&0xF], lut[qg[1]>>4]);
        float4 gt1 = float4(lut[qg[2]&0xF], lut[qg[2]>>4], lut[qg[3]&0xF], lut[qg[3]>>4]);
        float4 gt2 = float4(lut[qg[4]&0xF], lut[qg[4]>>4], lut[qg[5]&0xF], lut[qg[5]>>4]);
        float4 gt3 = float4(lut[qg[6]&0xF], lut[qg[6]>>4], lut[qg[7]&0xF], lut[qg[7]>>4]);

        float gate_dot = dot(xf[0], gt0) + dot(xf[1], gt1)
                       + dot(xf[2], gt2) + dot(xf[3], gt3);

        float scale_up = e8m0_to_float(ws[up_row * H_scale + ib]);
        float scale_gate = e8m0_to_float(ws[gate_row * H_scale + ib]);

        sumf_up += scale_up * up_dot;
        sumf_gate += scale_gate * gate_dot;
    }

    float total_up = simd_sum(sumf_up);
    float total_gate = simd_sum(sumf_gate);

    if (tiisg == 0) {
        if (has_bias) {
            device const bfloat* bias = all_bias + eid * (long)N;
            total_up += float(bias[up_row]);
            total_gate += float(bias[gate_row]);
        }

        float gate_c = min(total_gate, clamp_l);
        float up_c = clamp(total_up, -clamp_l, clamp_l);
        float glu = gate_c / (1.0f + exp(-gate_c * alpha));
        output[expert_idx * I + out_col] = static_cast<bfloat>((up_c + 1.0f) * glu);
    }
}


// Kernel 12: moe_decode_gemm1 — SIMD K-split FP4 matvec for GEMM1
// =============================================================================
//
// Key insight: 32 SIMD lanes split the K dimension (not the N dimension).
// This distributes the FP4 dequant compute across threads, hiding latency.
//
// Thread mapping: ix = tiisg/2 (block 0..15), it = tiisg%2 (half-block 0/1)
// Each pair (ix, it) processes one 32-element FP4 block (16 nibbles each).
// 16 blocks = 512 elements processed per stride.
// After K loop, simd_sum reduces across all 32 lanes.
//
// NR0=2: each simdgroup handles 2 output rows, reusing activation loads.
// N_SG=2: 2 simdgroups per threadgroup → 4 output rows per TG.
//
// Grid: (ceil(N / 4), K_active, 1)
// ThreadsPerTG: (32, 2, 1)  [32 per SG, 2 SGs]
// Threadgroup memory: 64 bytes (16-float FP4 LUT)
//
// Params buffer: [H_PAD, N_per_expert]
//
kernel void moe_decode_gemm1(
    device const bfloat*    input          [[buffer(0)]],  // [1, H_PAD]
    device const uint8_t*   all_w_blocks   [[buffer(1)]],  // [E, N, H_PAD/2]
    device const uint8_t*   all_w_scales   [[buffer(2)]],  // [E, N, H_PAD/32]
    device bfloat*          output         [[buffer(3)]],  // [K, N]
    constant uint*          expert_ids     [[buffer(4)]],  // [K]
    constant float*         params         [[buffer(5)]],  // [H_PAD, N]
    uint3  tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]]
) {
    const uint H = (uint)params[0];       // H_PAD (input dim = K in matmul)
    const uint N = (uint)params[1];       // 2*I (output dim per expert)
    const uint H_half = H / 2;
    const uint H_scale = H / 32;
    const uint nb = H / 32;              // blocks per weight row

    // Load FP4 LUT into threadgroup memory (lower latency than constant mem)
    threadgroup float lut[16];
    if (sgitg == 0 && tiisg < 16) {
        lut[tiisg] = fp4_lut[tiisg];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Grid mapping: tgpig.x = row tile, tgpig.y = expert index
    const uint expert_idx = tgpig.y;
    const uint eid = expert_ids[expert_idx];
    const uint first_row = tgpig.x * 4 + sgitg * 2;  // 4 rows/TG, 2 rows/SG

    // Expert weight pointers
    device const uint8_t* wb = all_w_blocks + (long)eid * N * H_half;
    device const uint8_t* ws = all_w_scales + (long)eid * N * H_scale;

    // SIMD lane assignment
    const ushort ix = tiisg / 2;   // block index within stride (0..15)
    const ushort it = tiisg % 2;   // half-block index (0 or 1)

    float sumf[2] = {0.0f, 0.0f};

    // K-dimension loop: stride of 16 blocks = 512 elements
    for (uint ib = ix; ib < nb; ib += 16) {
        // Activation: 16 bf16 values for this half-block
        const uint k_base = ib * 32 + it * 16;
        device const bfloat* xp = input + k_base;

        // Load 16 activation values as 4 × float4
        float4 xf[4];
        xf[0] = float4(float(xp[0]),  float(xp[1]),  float(xp[2]),  float(xp[3]));
        xf[1] = float4(float(xp[4]),  float(xp[5]),  float(xp[6]),  float(xp[7]));
        xf[2] = float4(float(xp[8]),  float(xp[9]),  float(xp[10]), float(xp[11]));
        xf[3] = float4(float(xp[12]), float(xp[13]), float(xp[14]), float(xp[15]));

        // Process 2 output rows (NR0=2)
        for (short r = 0; r < 2; r++) {
            uint row = first_row + r;
            if (row >= N) break;

            // 8 weight bytes for this half-block
            device const uint8_t* q = wb + (long)row * H_half + ib * 16 + it * 8;

            // Dequant 16 nibbles via threadgroup LUT, multiply with activation
            // byte i → lo nibble = element 2i, hi nibble = element 2i+1
            float4 wp0 = float4(lut[q[0]&0xF], lut[q[0]>>4], lut[q[1]&0xF], lut[q[1]>>4]);
            float4 wp1 = float4(lut[q[2]&0xF], lut[q[2]>>4], lut[q[3]&0xF], lut[q[3]>>4]);
            float4 wp2 = float4(lut[q[4]&0xF], lut[q[4]>>4], lut[q[5]&0xF], lut[q[5]>>4]);
            float4 wp3 = float4(lut[q[6]&0xF], lut[q[6]>>4], lut[q[7]&0xF], lut[q[7]>>4]);

            float block_dot = dot(xf[0], wp0) + dot(xf[1], wp1)
                            + dot(xf[2], wp2) + dot(xf[3], wp3);

            // Deferred scale: 1 multiply per 32-element block
            sumf[r] += e8m0_to_float(ws[row * H_scale + ib]) * block_dot;
        }
    }

    // SIMD reduction: combine all 32 lanes
    for (short r = 0; r < 2; r++) {
        float total = simd_sum(sumf[r]);
        uint row = first_row + r;
        if (tiisg == 0 && row < N) {
            output[expert_idx * N + row] = static_cast<bfloat>(total);
        }
    }
}


// =============================================================================
// Kernel 13: moe_decode_gemm2_fused — SIMD K-split FP4 GEMM2
// =============================================================================
//
// Fused GEMM2 across K active experts with routing scale accumulation.
// output[1, out_dim] = sum_k routing_scale[k] * act_k[1, in_dim] × W_k[out_dim, in_dim]^T
//
// Same SIMD-parallel K-split approach as Kernel 12.
// Iterates over K experts within each threadgroup, accumulating scaled results.
//
// Grid: (ceil(out_dim / 4), 1, 1)
// ThreadsPerTG: (32, 2, 1)
// Threadgroup memory: 64 bytes (16-float FP4 LUT)
//
// Params buffer: [out_dim, in_dim, K_active, has_bias]
//
kernel void moe_decode_gemm2_fused(
    device const bfloat*    all_activations [[buffer(0)]],  // [K, in_dim]
    device const uint8_t*   all_w_blocks    [[buffer(1)]],  // [E, out_dim, in_dim/2]
    device const uint8_t*   all_w_scales    [[buffer(2)]],  // [E, out_dim, in_dim/32]
    device const bfloat*    all_bias        [[buffer(3)]],  // [E, out_dim] or dummy
    device float*           output          [[buffer(4)]],  // [1, out_dim] float32
    constant uint*          expert_ids      [[buffer(5)]],  // [K]
    constant float*         routing_scales  [[buffer(6)]],  // [K]
    constant float*         params          [[buffer(7)]],  // [out_dim, in_dim, K_active, has_bias]
    uint3  tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]]
) {
    const uint out_dim = (uint)params[0];   // H_PAD (output dim)
    const uint in_dim  = (uint)params[1];   // I (input dim per expert)
    const uint K_active = (uint)params[2];
    const uint has_bias = (uint)params[3];  // 1.0 if bias present
    const uint in_half = in_dim / 2;
    const uint in_scale = in_dim / 32;
    const uint nb = in_dim / 32;

    // Load LUT
    threadgroup float lut[16];
    if (sgitg == 0 && tiisg < 16) {
        lut[tiisg] = fp4_lut[tiisg];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint first_row = tgpig.x * 4 + sgitg * 2;  // 4 rows/TG, 2 rows/SG
    const ushort ix = tiisg / 2;
    const ushort it = tiisg % 2;

    float acc[2] = {0.0f, 0.0f};  // accumulated output across all experts

    // Loop over K active experts
    for (uint k = 0; k < K_active; k++) {
        uint eid = expert_ids[k];
        float rscale = routing_scales[k];
        device const bfloat* act = all_activations + k * in_dim;
        device const uint8_t* wb = all_w_blocks + (long)eid * out_dim * in_half;
        device const uint8_t* ws = all_w_scales + (long)eid * out_dim * in_scale;

        float sumf[2] = {0.0f, 0.0f};

        // K-dimension loop
        for (uint ib = ix; ib < nb; ib += 16) {
            const uint k_base = ib * 32 + it * 16;
            device const bfloat* xp = act + k_base;

            float4 xf[4];
            xf[0] = float4(float(xp[0]),  float(xp[1]),  float(xp[2]),  float(xp[3]));
            xf[1] = float4(float(xp[4]),  float(xp[5]),  float(xp[6]),  float(xp[7]));
            xf[2] = float4(float(xp[8]),  float(xp[9]),  float(xp[10]), float(xp[11]));
            xf[3] = float4(float(xp[12]), float(xp[13]), float(xp[14]), float(xp[15]));

            for (short r = 0; r < 2; r++) {
                uint row = first_row + r;
                if (row >= out_dim) break;

                device const uint8_t* q = wb + (long)row * in_half + ib * 16 + it * 8;

                float4 wp0 = float4(lut[q[0]&0xF], lut[q[0]>>4], lut[q[1]&0xF], lut[q[1]>>4]);
                float4 wp1 = float4(lut[q[2]&0xF], lut[q[2]>>4], lut[q[3]&0xF], lut[q[3]>>4]);
                float4 wp2 = float4(lut[q[4]&0xF], lut[q[4]>>4], lut[q[5]&0xF], lut[q[5]>>4]);
                float4 wp3 = float4(lut[q[6]&0xF], lut[q[6]>>4], lut[q[7]&0xF], lut[q[7]>>4]);

                float block_dot = dot(xf[0], wp0) + dot(xf[1], wp1)
                                + dot(xf[2], wp2) + dot(xf[3], wp3);

                sumf[r] += e8m0_to_float(ws[row * in_scale + ib]) * block_dot;
            }
        }

        // SIMD reduction and accumulate with routing scale
        for (short r = 0; r < 2; r++) {
            float dot_val = simd_sum(sumf[r]);
            if (has_bias) {
                uint row = first_row + r;
                if (row < out_dim) {
                    dot_val += float(all_bias[eid * out_dim + row]);
                }
            }
            acc[r] += rscale * dot_val;
        }
    }

    // Write final output (float32 for precision during residual add)
    for (short r = 0; r < 2; r++) {
        uint row = first_row + r;
        if (tiisg == 0 && row < out_dim) {
            output[row] = acc[r];
        }
    }
}
