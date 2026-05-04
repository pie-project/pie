#include <metal_stdlib>
using namespace metal;

struct AppendPagedKVCacheParams {
    uint num_tokens;
    uint num_kv_heads;
    uint head_size;
    uint page_size;
    uint max_num_pages;
    uint batch_size;
    uint k_stride_token;  // num_kv_heads * head_size
    uint k_stride_head;   // head_size
    uint v_stride_token;  // num_kv_heads * head_size
    uint v_stride_head;   // head_size
};

/**
 * Metal kernel for appending key-value pairs to paged KV cache
 * Equivalent to FlashInfer's AppendPagedKVCache CUDA operation
 *
 * Each thread processes one token's K or V data for one head
 */
kernel void metal_append_paged_kv_cache_bfloat16(
    device const bfloat *k_input [[buffer(0)]],        // [num_tokens, num_kv_heads * head_size]
    device const bfloat *v_input [[buffer(1)]],        // [num_tokens, num_kv_heads * head_size]
    device bfloat *paged_kv_cache [[buffer(2)]],       // Unified buffer: [num_pages, 2, page_size, num_kv_heads, head_size]
    device const uint *kv_batch_indices [[buffer(3)]], // [num_tokens] - which batch each token belongs to
    device const uint *kv_positions [[buffer(4)]],     // [num_tokens] - position within sequence
    device const uint *kv_page_indices [[buffer(5)]],  // [max_num_pages] - mapping of logical to physical pages
    device const uint *kv_page_indptr [[buffer(6)]],   // [batch_size + 1] - page range per batch
    device const uint *kv_last_page_lens [[buffer(7)]], // [batch_size] - length of last page per batch
    device const float* params_raw [[buffer(8)]],
    uint3 thread_position_in_grid [[thread_position_in_grid]]
) {
    const uint token_idx = thread_position_in_grid.x;
    const uint head_idx = thread_position_in_grid.y;
    const uint head_offset = thread_position_in_grid.z;

    if (token_idx >= ((uint)params_raw[0]) ||
        head_idx >= ((uint)params_raw[1]) ||
        head_offset >= ((uint)params_raw[2])) {
        return;
    }

    const uint batch_idx = kv_batch_indices[token_idx];
    const uint token_position = kv_positions[token_idx];

    // Validate batch index
    if (batch_idx >= ((uint)params_raw[5])) {
        return;
    }

    // Find which page this token belongs to within the batch
    const uint page_start = kv_page_indptr[batch_idx];
    const uint page_end = kv_page_indptr[batch_idx + 1];

    // Calculate which page within the batch sequence to use
    const uint page_offset_in_batch = token_position / ((uint)params_raw[3]);
    const uint position_in_page = token_position % ((uint)params_raw[3]);

    // Get physical page index
    const uint page_logical = page_start + page_offset_in_batch;
    if (page_logical >= page_end) {
        return; // Token position exceeds allocated pages for this batch
    }

    const uint physical_page = kv_page_indices[page_logical];
    if (physical_page >= ((uint)params_raw[4])) {
        return;
    }

    // Calculate input offsets
    const uint input_offset = token_idx * ((uint)params_raw[6]) +
                             head_idx * ((uint)params_raw[7]) +
                             head_offset;

    // Calculate output offsets in paged cache with INTERLEAVED K/V layout
    // Layout: [page_idx, 2, page_size, num_kv_heads, head_size]
    // Within each page: K at offset 0, V at offset (page_size * num_kv_heads * head_size)
    const uint page_size = (uint)params_raw[3];
    const uint num_kv_heads = (uint)params_raw[1];
    const uint head_size = (uint)params_raw[2];

    // Elements per page for both K and V combined
    const uint elements_per_page = 2 * page_size * num_kv_heads * head_size;

    // K cache offset (at offset 0 within page)
    const uint k_cache_offset = physical_page * elements_per_page +
                                0 * (page_size * num_kv_heads * head_size) +  // K is at offset 0
                                position_in_page * (num_kv_heads * head_size) +
                                head_idx * head_size +
                                head_offset;

    // V cache offset (at offset page_size * num_kv_heads * head_size within page)
    const uint v_cache_offset = physical_page * elements_per_page +
                                1 * (page_size * num_kv_heads * head_size) +  // V is after K
                                position_in_page * (num_kv_heads * head_size) +
                                head_idx * head_size +
                                head_offset;

    // Copy K and V data from input to unified paged cache at correct interleaved positions
    if (input_offset < ((uint)params_raw[0]) * num_kv_heads * head_size) {
        paged_kv_cache[k_cache_offset] = k_input[input_offset];
        paged_kv_cache[v_cache_offset] = v_input[input_offset];
    }
}

/**
 * Metal kernel for float32 version
 */
kernel void metal_append_paged_kv_cache_float32(
    device const float *k_input [[buffer(0)]],
    device const float *v_input [[buffer(1)]],
    device float *paged_kv_cache [[buffer(2)]],
    device const uint *kv_batch_indices [[buffer(3)]],
    device const uint *kv_positions [[buffer(4)]],
    device const uint *kv_page_indices [[buffer(5)]],
    device const uint *kv_page_indptr [[buffer(6)]],
    device const uint *kv_last_page_lens [[buffer(7)]],
    device const float* params_raw [[buffer(8)]],
    uint3 thread_position_in_grid [[thread_position_in_grid]]
) {
    const uint token_idx = thread_position_in_grid.x;
    const uint head_idx = thread_position_in_grid.y;
    const uint head_offset = thread_position_in_grid.z;

    if (token_idx >= ((uint)params_raw[0]) ||
        head_idx >= ((uint)params_raw[1]) ||
        head_offset >= ((uint)params_raw[2])) {
        return;
    }

    const uint batch_idx = kv_batch_indices[token_idx];
    const uint token_position = kv_positions[token_idx];

    if (batch_idx >= ((uint)params_raw[5])) {
        return;
    }

    const uint page_start = kv_page_indptr[batch_idx];
    const uint page_end = kv_page_indptr[batch_idx + 1];

    const uint page_offset_in_batch = token_position / ((uint)params_raw[3]);
    const uint position_in_page = token_position % ((uint)params_raw[3]);

    const uint page_logical = page_start + page_offset_in_batch;
    if (page_logical >= page_end) {
        return;
    }

    const uint physical_page = kv_page_indices[page_logical];
    if (physical_page >= ((uint)params_raw[4])) {
        return;
    }

    const uint input_offset = token_idx * ((uint)params_raw[6]) +
                             head_idx * ((uint)params_raw[7]) +
                             head_offset;

    // INTERLEAVED K/V layout: [page_idx, 2, page_size, num_kv_heads, head_size]
    const uint page_size = (uint)params_raw[3];
    const uint num_kv_heads = (uint)params_raw[1];
    const uint head_size = (uint)params_raw[2];
    const uint elements_per_page = 2 * page_size * num_kv_heads * head_size;

    const uint k_cache_offset = physical_page * elements_per_page +
                                0 * (page_size * num_kv_heads * head_size) +
                                position_in_page * (num_kv_heads * head_size) +
                                head_idx * head_size +
                                head_offset;

    const uint v_cache_offset = physical_page * elements_per_page +
                                1 * (page_size * num_kv_heads * head_size) +
                                position_in_page * (num_kv_heads * head_size) +
                                head_idx * head_size +
                                head_offset;

    if (input_offset < ((uint)params_raw[0]) * num_kv_heads * head_size) {
        paged_kv_cache[k_cache_offset] = k_input[input_offset];
        paged_kv_cache[v_cache_offset] = v_input[input_offset];
    }
}

/**
 * Metal kernel for float16 version (optimized with vectorized writes)
 */
kernel void metal_append_paged_kv_cache_float16(
    device const half *k_input [[buffer(0)]],
    device const half *v_input [[buffer(1)]],
    device half *paged_kv_cache [[buffer(2)]],
    device const uint *kv_batch_indices [[buffer(3)]],
    device const uint *kv_positions [[buffer(4)]],
    device const uint *kv_page_indices [[buffer(5)]],
    device const uint *kv_page_indptr [[buffer(6)]],
    device const uint *kv_last_page_lens [[buffer(7)]],
    device const float* params_raw [[buffer(8)]],
    uint3 thread_position_in_grid [[thread_position_in_grid]],
    uint3 threads_per_threadgroup [[threads_per_threadgroup]],
    uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]]
) {
    // Pre-load params (avoid repeated float->uint casts)
    const uint num_tokens = (uint)params_raw[0];
    const uint num_kv_heads = (uint)params_raw[1];
    const uint head_size = (uint)params_raw[2];
    const uint page_size = (uint)params_raw[3];
    const uint max_num_pages = (uint)params_raw[4];
    const uint batch_size = (uint)params_raw[5];
    const uint k_stride_token = (uint)params_raw[6];
    const uint head_size_stride = (uint)params_raw[7];

    const uint token_idx = thread_position_in_grid.x;
    const uint head_idx = thread_position_in_grid.y;
    const uint head_offset = thread_position_in_grid.z;

    if (token_idx >= num_tokens || head_idx >= num_kv_heads || head_offset >= head_size) {
        return;
    }

    const uint batch_idx = kv_batch_indices[token_idx];
    const uint token_position = kv_positions[token_idx];

    if (batch_idx >= batch_size) {
        return;
    }

    const uint page_start = kv_page_indptr[batch_idx];
    const uint page_end = kv_page_indptr[batch_idx + 1];

    const uint page_offset_in_batch = token_position / page_size;
    const uint position_in_page = token_position % page_size;

    const uint page_logical = page_start + page_offset_in_batch;
    if (page_logical >= page_end) {
        return;
    }

    const uint physical_page = kv_page_indices[page_logical];
    if (physical_page >= max_num_pages) {
        return;
    }

    const uint input_offset = token_idx * k_stride_token +
                             head_idx * head_size_stride +
                             head_offset;

    // INTERLEAVED K/V layout: [page_idx, 2, page_size, num_kv_heads, head_size]
    const uint elements_per_page = 2 * page_size * num_kv_heads * head_size;
    const uint page_kv_stride = page_size * num_kv_heads * head_size;

    // Calculate base offset for this token's position in the page
    const uint base_offset = physical_page * elements_per_page +
                            position_in_page * (num_kv_heads * head_size) +
                            head_idx * head_size +
                            head_offset;

    // K at offset 0, V at offset page_kv_stride
    const uint k_cache_offset = base_offset;
    const uint v_cache_offset = base_offset + page_kv_stride;

    // Write K and V (bounds check already done above)
    paged_kv_cache[k_cache_offset] = k_input[input_offset];
    paged_kv_cache[v_cache_offset] = v_input[input_offset];
}