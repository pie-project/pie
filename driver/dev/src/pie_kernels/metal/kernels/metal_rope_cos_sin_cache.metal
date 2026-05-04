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

// =============================================================================
// Fused Q+K RoPE kernels — process both buffers in a single dispatch
// =============================================================================
//
// params_raw[0] = num_tokens
// params_raw[1] = num_q_heads
// params_raw[2] = num_kv_heads
// params_raw[3] = head_size (= head_dim)
// params_raw[4] = is_neox (0 or 1)
//
// Thread grid: (num_tokens, max(num_q_heads, num_kv_heads), half_dim)

kernel void metal_rope_cos_sin_cache_fused_bfloat16(
    device bfloat* query              [[buffer(0)]],  // [num_tokens, num_q_heads, head_size]
    device bfloat* key                [[buffer(1)]],  // [num_tokens, num_kv_heads, head_size]
    device const int* positions       [[buffer(2)]],  // [num_tokens]
    device const float* cos_sin_cache [[buffer(3)]],  // [max_pos, head_dim]
    device const float* params_raw    [[buffer(4)]],
    uint3 gid                         [[thread_position_in_grid]]
) {
    const uint32_t token_idx = gid.x;
    const uint32_t head_idx = gid.y;
    const uint32_t pair_idx = gid.z;

    const uint32_t num_tokens = (uint32_t)params_raw[0];
    const uint32_t num_q_heads = (uint32_t)params_raw[1];
    const uint32_t num_kv_heads = (uint32_t)params_raw[2];
    const uint32_t head_size = (uint32_t)params_raw[3];
    const uint32_t half_dim = head_size / 2;
    const uint32_t max_heads = max(num_q_heads, num_kv_heads);

    if (token_idx >= num_tokens || head_idx >= max_heads || pair_idx >= half_dim) {
        return;
    }

    const int pos = positions[token_idx];
    const float c = cos_sin_cache[pos * head_size + pair_idx];
    const float s = cos_sin_cache[pos * head_size + half_dim + pair_idx];

    const int is_neox = (int)params_raw[4];

    // Apply to Q
    if (head_idx < num_q_heads) {
        const uint32_t q_base = token_idx * num_q_heads * head_size + head_idx * head_size;
        uint32_t qx, qy;
        if (is_neox) { qx = q_base + pair_idx; qy = q_base + pair_idx + half_dim; }
        else         { qx = q_base + pair_idx * 2; qy = q_base + pair_idx * 2 + 1; }
        const float x = float(query[qx]);
        const float y = float(query[qy]);
        query[qx] = bfloat(x * c - y * s);
        query[qy] = bfloat(x * s + y * c);
    }

    // Apply to K
    if (head_idx < num_kv_heads) {
        const uint32_t k_base = token_idx * num_kv_heads * head_size + head_idx * head_size;
        uint32_t kx, ky;
        if (is_neox) { kx = k_base + pair_idx; ky = k_base + pair_idx + half_dim; }
        else         { kx = k_base + pair_idx * 2; ky = k_base + pair_idx * 2 + 1; }
        const float x = float(key[kx]);
        const float y = float(key[ky]);
        key[kx] = bfloat(x * c - y * s);
        key[ky] = bfloat(x * s + y * c);
    }
}

kernel void metal_rope_cos_sin_cache_fused_float16(
    device half* query                [[buffer(0)]],
    device half* key                  [[buffer(1)]],
    device const int* positions       [[buffer(2)]],
    device const float* cos_sin_cache [[buffer(3)]],
    device const float* params_raw    [[buffer(4)]],
    uint3 gid                         [[thread_position_in_grid]]
) {
    const uint32_t token_idx = gid.x;
    const uint32_t head_idx = gid.y;
    const uint32_t pair_idx = gid.z;

    const uint32_t num_tokens = (uint32_t)params_raw[0];
    const uint32_t num_q_heads = (uint32_t)params_raw[1];
    const uint32_t num_kv_heads = (uint32_t)params_raw[2];
    const uint32_t head_size = (uint32_t)params_raw[3];
    const uint32_t half_dim = head_size / 2;
    const uint32_t max_heads = max(num_q_heads, num_kv_heads);

    if (token_idx >= num_tokens || head_idx >= max_heads || pair_idx >= half_dim) {
        return;
    }

    const int pos = positions[token_idx];
    const float c = cos_sin_cache[pos * head_size + pair_idx];
    const float s = cos_sin_cache[pos * head_size + half_dim + pair_idx];

    const int is_neox = (int)params_raw[4];

    if (head_idx < num_q_heads) {
        const uint32_t q_base = token_idx * num_q_heads * head_size + head_idx * head_size;
        uint32_t qx, qy;
        if (is_neox) { qx = q_base + pair_idx; qy = q_base + pair_idx + half_dim; }
        else         { qx = q_base + pair_idx * 2; qy = q_base + pair_idx * 2 + 1; }
        const float x = float(query[qx]);
        const float y = float(query[qy]);
        query[qx] = half(x * c - y * s);
        query[qy] = half(x * s + y * c);
    }

    if (head_idx < num_kv_heads) {
        const uint32_t k_base = token_idx * num_kv_heads * head_size + head_idx * head_size;
        uint32_t kx, ky;
        if (is_neox) { kx = k_base + pair_idx; ky = k_base + pair_idx + half_dim; }
        else         { kx = k_base + pair_idx * 2; ky = k_base + pair_idx * 2 + 1; }
        const float x = float(key[kx]);
        const float y = float(key[ky]);
        key[kx] = half(x * c - y * s);
        key[ky] = half(x * s + y * c);
    }
}

kernel void metal_rope_cos_sin_cache_fused_float32(
    device float* query               [[buffer(0)]],
    device float* key                 [[buffer(1)]],
    device const int* positions       [[buffer(2)]],
    device const float* cos_sin_cache [[buffer(3)]],
    device const float* params_raw    [[buffer(4)]],
    uint3 gid                         [[thread_position_in_grid]]
) {
    const uint32_t token_idx = gid.x;
    const uint32_t head_idx = gid.y;
    const uint32_t pair_idx = gid.z;

    const uint32_t num_tokens = (uint32_t)params_raw[0];
    const uint32_t num_q_heads = (uint32_t)params_raw[1];
    const uint32_t num_kv_heads = (uint32_t)params_raw[2];
    const uint32_t head_size = (uint32_t)params_raw[3];
    const uint32_t half_dim = head_size / 2;
    const uint32_t max_heads = max(num_q_heads, num_kv_heads);

    if (token_idx >= num_tokens || head_idx >= max_heads || pair_idx >= half_dim) {
        return;
    }

    const int pos = positions[token_idx];
    const float c = cos_sin_cache[pos * head_size + pair_idx];
    const float s = cos_sin_cache[pos * head_size + half_dim + pair_idx];

    const int is_neox = (int)params_raw[4];

    if (head_idx < num_q_heads) {
        const uint32_t q_base = token_idx * num_q_heads * head_size + head_idx * head_size;
        uint32_t qx, qy;
        if (is_neox) { qx = q_base + pair_idx; qy = q_base + pair_idx + half_dim; }
        else         { qx = q_base + pair_idx * 2; qy = q_base + pair_idx * 2 + 1; }
        const float x = query[qx];
        const float y = query[qy];
        query[qx] = x * c - y * s;
        query[qy] = x * s + y * c;
    }

    if (head_idx < num_kv_heads) {
        const uint32_t k_base = token_idx * num_kv_heads * head_size + head_idx * head_size;
        uint32_t kx, ky;
        if (is_neox) { kx = k_base + pair_idx; ky = k_base + pair_idx + half_dim; }
        else         { kx = k_base + pair_idx * 2; ky = k_base + pair_idx * 2 + 1; }
        const float x = key[kx];
        const float y = key[ky];
        key[kx] = x * c - y * s;
        key[ky] = x * s + y * c;
    }
}
