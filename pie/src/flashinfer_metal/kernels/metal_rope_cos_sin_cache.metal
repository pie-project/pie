#include <metal_stdlib>
using namespace metal;

// RoPE with precomputed cos/sin cache — Metal implementation
// Matches flashinfer.apply_rope_with_cos_sin_cache_inplace()
//
// cos_sin_cache layout: [max_pos, head_dim]
//   columns [0, half_dim)  = cos values
//   columns [half_dim, head_dim) = sin values
//
// Thread grid: (num_tokens, num_heads, half_dim)
//
// params_raw[0] = num_tokens
// params_raw[1] = num_heads
// params_raw[2] = head_size (= head_dim)
// params_raw[3] = is_neox (0 or 1)

kernel void metal_rope_cos_sin_cache_bfloat16(
    device bfloat* input_qk           [[buffer(0)]],  // [num_tokens, num_heads, head_size]
    device const int* positions        [[buffer(1)]],  // [num_tokens]
    device const float* cos_sin_cache  [[buffer(2)]],  // [max_pos, head_dim]
    device const float* params_raw     [[buffer(3)]],
    uint3 gid                          [[thread_position_in_grid]]
) {
    const uint32_t token_idx = gid.x;
    const uint32_t head_idx = gid.y;
    const uint32_t pair_idx = gid.z;

    const uint32_t num_tokens = (uint32_t)params_raw[0];
    const uint32_t num_heads = (uint32_t)params_raw[1];
    const uint32_t head_size = (uint32_t)params_raw[2];
    const uint32_t half_dim = head_size / 2;

    if (token_idx >= num_tokens || head_idx >= num_heads || pair_idx >= half_dim) {
        return;
    }

    const uint32_t base_idx = token_idx * num_heads * head_size + head_idx * head_size;

    uint32_t x_idx, y_idx;
    if (((int)params_raw[3]) != 0) {
        // NeoX: split halves
        x_idx = base_idx + pair_idx;
        y_idx = base_idx + pair_idx + half_dim;
    } else {
        // Interleaved: pairs (0,1), (2,3), ...
        x_idx = base_idx + pair_idx * 2;
        y_idx = base_idx + pair_idx * 2 + 1;
    }

    const int pos = positions[token_idx];
    const float c = cos_sin_cache[pos * head_size + pair_idx];
    const float s = cos_sin_cache[pos * head_size + half_dim + pair_idx];

    const float x = float(input_qk[x_idx]);
    const float y = float(input_qk[y_idx]);

    input_qk[x_idx] = bfloat(x * c - y * s);
    input_qk[y_idx] = bfloat(x * s + y * c);
}

kernel void metal_rope_cos_sin_cache_float16(
    device half* input_qk              [[buffer(0)]],
    device const int* positions        [[buffer(1)]],
    device const float* cos_sin_cache  [[buffer(2)]],
    device const float* params_raw     [[buffer(3)]],
    uint3 gid                          [[thread_position_in_grid]]
) {
    const uint32_t token_idx = gid.x;
    const uint32_t head_idx = gid.y;
    const uint32_t pair_idx = gid.z;

    const uint32_t num_tokens = (uint32_t)params_raw[0];
    const uint32_t num_heads = (uint32_t)params_raw[1];
    const uint32_t head_size = (uint32_t)params_raw[2];
    const uint32_t half_dim = head_size / 2;

    if (token_idx >= num_tokens || head_idx >= num_heads || pair_idx >= half_dim) {
        return;
    }

    const uint32_t base_idx = token_idx * num_heads * head_size + head_idx * head_size;

    uint32_t x_idx, y_idx;
    if (((int)params_raw[3]) != 0) {
        x_idx = base_idx + pair_idx;
        y_idx = base_idx + pair_idx + half_dim;
    } else {
        x_idx = base_idx + pair_idx * 2;
        y_idx = base_idx + pair_idx * 2 + 1;
    }

    const int pos = positions[token_idx];
    const float c = cos_sin_cache[pos * head_size + pair_idx];
    const float s = cos_sin_cache[pos * head_size + half_dim + pair_idx];

    const float x = float(input_qk[x_idx]);
    const float y = float(input_qk[y_idx]);

    input_qk[x_idx] = half(x * c - y * s);
    input_qk[y_idx] = half(x * s + y * c);
}

kernel void metal_rope_cos_sin_cache_float32(
    device float* input_qk             [[buffer(0)]],
    device const int* positions        [[buffer(1)]],
    device const float* cos_sin_cache  [[buffer(2)]],
    device const float* params_raw     [[buffer(3)]],
    uint3 gid                          [[thread_position_in_grid]]
) {
    const uint32_t token_idx = gid.x;
    const uint32_t head_idx = gid.y;
    const uint32_t pair_idx = gid.z;

    const uint32_t num_tokens = (uint32_t)params_raw[0];
    const uint32_t num_heads = (uint32_t)params_raw[1];
    const uint32_t head_size = (uint32_t)params_raw[2];
    const uint32_t half_dim = head_size / 2;

    if (token_idx >= num_tokens || head_idx >= num_heads || pair_idx >= half_dim) {
        return;
    }

    const uint32_t base_idx = token_idx * num_heads * head_size + head_idx * head_size;

    uint32_t x_idx, y_idx;
    if (((int)params_raw[3]) != 0) {
        x_idx = base_idx + pair_idx;
        y_idx = base_idx + pair_idx + half_dim;
    } else {
        x_idx = base_idx + pair_idx * 2;
        y_idx = base_idx + pair_idx * 2 + 1;
    }

    const int pos = positions[token_idx];
    const float c = cos_sin_cache[pos * head_size + pair_idx];
    const float s = cos_sin_cache[pos * head_size + half_dim + pair_idx];

    const float x = input_qk[x_idx];
    const float y = input_qk[y_idx];

    input_qk[x_idx] = x * c - y * s;
    input_qk[y_idx] = x * s + y * c;
}
