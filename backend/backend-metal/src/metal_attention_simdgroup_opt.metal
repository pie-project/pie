#include <metal_stdlib>
using namespace metal;

// Include common constants and utilities
#include "metal_attention_common.metal"

// --- Simdgroup Optimizations ---
// 1. Simdgroup reductions for block max/sum (reduces barriers from O(log2(128)) to ~1)
// 2. Split-d parallel score computation (better utilization across head dimension)
// 3. Fused weight computation with V accumulation (eliminates w_block storage and barriers)
// 4. One TG per (qo, head) mapping (improves GPU occupancy for small batches)
// 5. Vectorization improvements (better memory bandwidth utilization)

kernel void batch_prefill_attention_unified_bf16_simdgroup_kernel(
    device const half* q_input [[buffer(0)]],
    device const half* paged_k_cache [[buffer(1)]],
    device const half* paged_v_cache [[buffer(2)]],
    device const int* qo_indptr [[buffer(3)]],
    device const int* kv_page_indptr [[buffer(4)]],
    device const int* kv_page_indices [[buffer(5)]],
    device const int* kv_last_page_lens [[buffer(6)]],
    device half* output [[buffer(7)]],
    constant Params& params [[buffer(8)]],
    device float* debug_out [[buffer(9)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint tid_in_tgp [[thread_index_in_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]
) {
    // Read parameters from the uniform buffer
    const int num_qo = params.num_qo;
    const int head_dim = params.head_dim;
    const int kv_head_dim = params.kv_head_dim;
    const int head_size = params.head_size;
    const int page_size = params.page_size;
    const int num_query_heads = params.num_query_heads;
    const int num_kv_heads = params.num_kv_heads;
    const float scale = params.scale;

    // Each threadgroup handles one query token.
    uint qo_idx = tgid.x;
    if (qo_idx >= uint(num_qo)) return;

    // Check if head_size (dimension per head) exceeds our threadgroup memory limit
    if (head_size > MAX_HEAD_DIM) return;

    const int num_simd_groups = TGP_SIZE / SIMD_SIZE;

    // --- Shared Memory Declaration ---
    threadgroup half q_s[MAX_HEAD_DIM];
    threadgroup half k_block[KERNEL_BLOCK_SIZE][MAX_HEAD_DIM];
    threadgroup half v_block[KERNEL_BLOCK_SIZE][MAX_HEAD_DIM];

    // Online softmax accumulators (allocated once, reused per head)
    threadgroup float m_i;
    threadgroup float l_i;
    threadgroup float acc_i[MAX_HEAD_DIM];

    // Simdgroup reduction scratchpad - much smaller than temp_reduce[TGP_SIZE]
    threadgroup float simd_scratch[4];  // Max 4 simdgroups for TGP_SIZE=128

    // Use the explicitly provided number of query heads
    const int num_heads = num_query_heads;

    // --- Get Sequence and KV Page Information ---
    int seq_id = find_sequence_id(qo_indptr, int(qo_idx));
    int kv_start_page_pos = kv_page_indptr[seq_id];
    int kv_end_page_pos = kv_page_indptr[seq_id + 1];
    int num_pages = kv_end_page_pos - kv_start_page_pos;

    if (qo_idx == 0 && tid_in_tgp == 0 && debug_out != nullptr) {
        debug_out[0] = scale;
        debug_out[1] = (float)head_dim;
        debug_out[2] = (float)page_size;
        debug_out[3] = (float)num_qo;
        debug_out[5] = (float)num_pages;
    }

    if (num_pages <= 0) {
        for (int d = tid_in_tgp; d < head_dim; d += TGP_SIZE) {
            output[qo_idx * head_dim + d] = 0.0h;
        }
        return;
    }
    int last_page_len = kv_last_page_lens[seq_id];
    int total_kv_len = (num_pages - 1) * page_size + last_page_len;
    int seq_start = qo_indptr[seq_id];
    int seq_pos = int(qo_idx) - seq_start;
    seq_pos = max(seq_pos, 0);
    int effective_kv_len = min(total_kv_len, seq_pos + 1);
    if (qo_idx == 0 && tid_in_tgp == 0 && debug_out != nullptr) {
        debug_out[4] = (float)total_kv_len;
        debug_out[6] = (float)last_page_len;
        // DEBUG: KV page indexing values
        debug_out[10] = (float)seq_id;
        debug_out[11] = (float)kv_start_page_pos;
        debug_out[12] = (float)kv_end_page_pos;
        if (num_pages > 0) {
            debug_out[13] = (float)kv_page_indices[kv_start_page_pos];  // First page index
        }
    }

    if (effective_kv_len <= 0) {
        for (int d = tid_in_tgp; d < head_dim; d += TGP_SIZE) {
            output[qo_idx * head_dim + d] = 0.0h;
        }
        return;
    }

    // --- Per-head processing: compute attention independently for each head ---
    for (int h = 0; h < num_heads; ++h) {
        // --- OPTIMIZATION 5: Combined initialization and query loading ---
        // Initialize and load query data in parallel to reduce barriers
        if (tid_in_tgp == 0) {
            m_i = -INFINITY;
            l_i = 0.0f;
        }

        int q_base = int(qo_idx) * head_dim + h * head_size;
        for (int d = tid_in_tgp; d < head_size; d += TGP_SIZE) {
            acc_i[d] = 0.0f;
            q_s[d] = q_input[q_base + d];  // Load query data simultaneously
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // --- Main Loop: Process KV Cache in Parallel Blocks ---
        for (int block_start = 0; block_start < effective_kv_len; block_start += KERNEL_BLOCK_SIZE) {
            // Load K/V for this block and head slice
            if (tid_in_tgp < KERNEL_BLOCK_SIZE) {
                int global_key_idx = block_start + tid_in_tgp;
                if (global_key_idx < effective_kv_len) {
                    int page_offset = global_key_idx / page_size;
                    int in_page_offset = global_key_idx % page_size;
                    int page_idx = kv_page_indices[kv_start_page_pos + page_offset];
                    int kv_head = map_query_to_kv_head(h, num_query_heads, num_kv_heads);
                    uint base_addr = calculate_kv_address(in_page_offset, page_size, kv_head_dim, head_size, page_idx, kv_head);

                    // Debug: Show address calculation for first few keys
                    if (qo_idx == 0 && tid_in_tgp == 0 && h == 0 && global_key_idx < 3 && debug_out != nullptr) {
                        int debug_idx = 15 + global_key_idx * 3;  // Use debug[15-23] for address info
                        debug_out[debug_idx] = (float)base_addr;
                        debug_out[debug_idx + 1] = (float)page_idx;
                        debug_out[debug_idx + 2] = (global_key_idx < total_kv_len) ? paged_k_cache[base_addr] : -999.0f;
                    }

                    for (int d = 0; d < head_size; ++d) {
                        k_block[tid_in_tgp][d] = paged_k_cache[base_addr + d];
                        v_block[tid_in_tgp][d] = paged_v_cache[base_addr + d];
                    }
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // --- Manual loop unrolling for dot product ---
            // Use manual unrolling instead of simdgroup operations to avoid cross-key issues
            float score = 0.0f;
            int global_key_idx_score = block_start + tid_in_tgp;

            if (tid_in_tgp < KERNEL_BLOCK_SIZE && global_key_idx_score < effective_kv_len) {
                // --- OPTIMIZATION 2: Vectorized memory access with float4 ---
                int d = 0;
                // Process 4 dimensions at a time using vectorized operations for BF16
                if ((head_size & 3) == 0) {  // Head size is multiple of 4
                    for (; d < head_size; d += 4) {
                        // Load 4 half values and convert to float4
                        half4 q_vec = *reinterpret_cast<threadgroup const half4*>(&q_s[d]);
                        half4 k_vec = *reinterpret_cast<threadgroup const half4*>(&k_block[tid_in_tgp][d]);

                        float4 q_f = float4(q_vec);
                        float4 k_f = float4(k_vec);

                        // Vector dot product
                        float4 prod = q_f * k_f;
                        score += prod.x + prod.y + prod.z + prod.w;
                    }
                } else {
                    // Fallback to manual unrolling for non-multiple-of-4 dimensions
                    for (; d < (head_size & ~3); d += 4) {
                        score += float(q_s[d]) * float(k_block[tid_in_tgp][d]);
                        score += float(q_s[d+1]) * float(k_block[tid_in_tgp][d+1]);
                        score += float(q_s[d+2]) * float(k_block[tid_in_tgp][d+2]);
                        score += float(q_s[d+3]) * float(k_block[tid_in_tgp][d+3]);
                    }
                }

                // Handle remaining dimensions
                for (; d < head_size; ++d) {
                    score += float(q_s[d]) * float(k_block[tid_in_tgp][d]);
                }

                score *= scale;
            } else {
                score = -INFINITY;
            }

            // --- Simdgroup reduction for block max ---
            float m_j = simdgroup_max_reduction(
                (tid_in_tgp < KERNEL_BLOCK_SIZE) ? score : -INFINITY,
                simd_scratch, tid_in_tgp, simd_lane_id, simd_group_id, KERNEL_BLOCK_SIZE
            );

            // Update global max and rescale accumulators
            threadgroup float m_prev;
            if (tid_in_tgp == 0) {
                m_prev = m_i;
                m_i = max(m_i, m_j);
                simd_scratch[0] = m_prev; // Store for broadcast
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            m_prev = simd_scratch[0]; // Broadcast previous max

            float scale_factor = fast::exp(m_prev - m_i);
            if (tid_in_tgp == 0) {
                l_i *= scale_factor;
            }
            for (int d = tid_in_tgp; d < head_size; d += TGP_SIZE) {
                acc_i[d] *= scale_factor;
            }

            // --- OPTIMIZATION 3: Fused weight computation with V accumulation ---
            float w = (score > -INFINITY) ? fast::exp(score - m_i) : 0.0f;

            // Simdgroup sum for weights
            float l_j = simdgroup_sum_reduction(
                (tid_in_tgp < KERNEL_BLOCK_SIZE) ? w : 0.0f,
                simd_scratch, tid_in_tgp, simd_lane_id, simd_group_id, KERNEL_BLOCK_SIZE
            );

            if (tid_in_tgp == 0) {
                l_i += l_j;
            }

            // Store weights for this block (needed for correct V accumulation)
            threadgroup float w_block[KERNEL_BLOCK_SIZE];
            if (tid_in_tgp < KERNEL_BLOCK_SIZE) {
                w_block[tid_in_tgp] = w;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // --- Vectorized V accumulation with loop unrolling ---
            int num_keys_in_block = min((int)KERNEL_BLOCK_SIZE, effective_kv_len - block_start);

            // Process dimensions in batches for better cache utilization
            for (int d = tid_in_tgp; d < head_size; d += TGP_SIZE) {
                float sum_wv_d = 0.0f;

                // --- Vectorized weight-value multiplication ---
                int j = 0;
                // Process 4 weights and values at once using vector operations
                if ((num_keys_in_block & 3) == 0 && KERNEL_BLOCK_SIZE >= 4) {
                    for (; j < num_keys_in_block; j += 4) {
                        // Load 4 weights as vector
                        float4 w_vec = *reinterpret_cast<threadgroup const float4*>(&w_block[j]);

                        // Load 4 V values for current dimension
                        half4 v_vec = half4(v_block[j][d], v_block[j+1][d], v_block[j+2][d], v_block[j+3][d]);
                        float4 v_f = float4(v_vec);

                        // Vector multiply and sum
                        float4 prod = w_vec * v_f;
                        sum_wv_d += prod.x + prod.y + prod.z + prod.w;
                    }
                } else {
                    // Fallback to manual unrolling for non-multiple-of-4 cases
                    for (; j < (num_keys_in_block & ~3); j += 4) {
                        sum_wv_d += w_block[j] * float(v_block[j][d]);
                        sum_wv_d += w_block[j+1] * float(v_block[j+1][d]);
                        sum_wv_d += w_block[j+2] * float(v_block[j+2][d]);
                        sum_wv_d += w_block[j+3] * float(v_block[j+3][d]);
                    }
                }

                // Handle remaining keys
                for (; j < num_keys_in_block; ++j) {
                    sum_wv_d += w_block[j] * float(v_block[j][d]);
                }

                acc_i[d] += sum_wv_d;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        // --- Finalization for this head ---
        int out_base = int(qo_idx) * head_dim + h * head_size;
        for (int d = tid_in_tgp; d < head_size; d += TGP_SIZE) {
            if (l_i > 1e-9f) {
                output[out_base + d] = half(acc_i[d] / l_i);
            } else {
                output[out_base + d] = 0.0h;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

kernel void batch_prefill_attention_unified_f32_simdgroup_kernel(
    device const float* q_input [[buffer(0)]],
    device const float* paged_k_cache [[buffer(1)]],
    device const float* paged_v_cache [[buffer(2)]],
    device const int* qo_indptr [[buffer(3)]],
    device const int* kv_page_indptr [[buffer(4)]],
    device const int* kv_page_indices [[buffer(5)]],
    device const int* kv_last_page_lens [[buffer(6)]],
    device float* output [[buffer(7)]],
    constant Params& params [[buffer(8)]],
    device float* debug_out [[buffer(9)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint tid_in_tgp [[thread_index_in_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]
) {
    // Read parameters from the uniform buffer
    const int num_qo = params.num_qo;
    const int head_dim = params.head_dim;
    const int kv_head_dim = params.kv_head_dim;
    const int head_size = params.head_size;
    const int page_size = params.page_size;
    const int num_query_heads = params.num_query_heads;
    const int num_kv_heads = params.num_kv_heads;
    const float scale = params.scale;

    // Each threadgroup handles one query token.
    uint qo_idx = tgid.x;
    if (qo_idx >= uint(num_qo)) return;

    // F32 kernels use 2x memory compared to BF16, so stricter limits apply
    // Conservative limit for F32 to fit in 32KB threadgroup memory
    const int MAX_F32_HEAD_DIM = 128;
    if (head_size > MAX_F32_HEAD_DIM) return;

    const int num_simd_groups = TGP_SIZE / SIMD_SIZE;

    // --- Shared Memory Declaration ---
    // Use smaller arrays for F32 to fit in 32KB threadgroup memory limit
    threadgroup float q_s[MAX_F32_HEAD_DIM];
    threadgroup float k_block[KERNEL_BLOCK_SIZE][MAX_F32_HEAD_DIM];
    threadgroup float v_block[KERNEL_BLOCK_SIZE][MAX_F32_HEAD_DIM];

    // Online softmax accumulators (allocated once, reused per head)
    threadgroup float m_i;
    threadgroup float l_i;
    threadgroup float acc_i[MAX_F32_HEAD_DIM];

    // Simdgroup reduction scratchpad - much smaller than temp_reduce[TGP_SIZE]
    threadgroup float simd_scratch[4];  // Max 4 simdgroups for TGP_SIZE=128

    // Use the explicitly provided number of query heads
    const int num_heads = num_query_heads;

    // --- Get Sequence and KV Page Information ---
    int seq_id = find_sequence_id(qo_indptr, int(qo_idx));
    int kv_start_page_pos = kv_page_indptr[seq_id];
    int kv_end_page_pos = kv_page_indptr[seq_id + 1];
    int num_pages = kv_end_page_pos - kv_start_page_pos;

    if (qo_idx == 0 && tid_in_tgp == 0 && debug_out != nullptr) {
        debug_out[0] = scale;
        debug_out[1] = (float)head_dim;
        debug_out[2] = (float)page_size;
        debug_out[3] = (float)num_qo;
        debug_out[5] = (float)num_pages;
    }

    if (num_pages <= 0) {
        for (int d = tid_in_tgp; d < head_dim; d += TGP_SIZE) {
            output[qo_idx * head_dim + d] = 0.0f;
        }
        return;
    }
    int last_page_len = kv_last_page_lens[seq_id];
    int total_kv_len = (num_pages - 1) * page_size + last_page_len;
    int seq_start = qo_indptr[seq_id];
    int seq_pos = int(qo_idx) - seq_start;
    seq_pos = max(seq_pos, 0);
    int effective_kv_len = min(total_kv_len, seq_pos + 1);
    if (qo_idx == 0 && tid_in_tgp == 0 && debug_out != nullptr) {
        debug_out[4] = (float)total_kv_len;
        debug_out[6] = (float)last_page_len;
        // DEBUG: KV page indexing values
        debug_out[10] = (float)seq_id;
        debug_out[11] = (float)kv_start_page_pos;
        debug_out[12] = (float)kv_end_page_pos;
        if (num_pages > 0) {
            debug_out[13] = (float)kv_page_indices[kv_start_page_pos];  // First page index
        }
    }

    if (effective_kv_len <= 0) {
        for (int d = tid_in_tgp; d < head_dim; d += TGP_SIZE) {
            output[qo_idx * head_dim + d] = 0.0f;
        }
        return;
    }

    // --- Per-head processing: compute attention independently for each head ---
    for (int h = 0; h < num_heads; ++h) {
        // --- OPTIMIZATION 5: Combined initialization and query loading ---
        // Initialize and load query data in parallel to reduce barriers
        if (tid_in_tgp == 0) {
            m_i = -INFINITY;
            l_i = 0.0f;
        }

        int q_base = int(qo_idx) * head_dim + h * head_size;
        for (int d = tid_in_tgp; d < head_size; d += TGP_SIZE) {
            acc_i[d] = 0.0f;
            q_s[d] = q_input[q_base + d];  // Load query data simultaneously
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // --- Main Loop: Process KV Cache in Parallel Blocks ---
        for (int block_start = 0; block_start < effective_kv_len; block_start += KERNEL_BLOCK_SIZE) {
            // Load K/V for this block and head slice
            if (tid_in_tgp < KERNEL_BLOCK_SIZE) {
                int global_key_idx = block_start + tid_in_tgp;
                if (global_key_idx < effective_kv_len) {
                    int page_offset = global_key_idx / page_size;
                    int in_page_offset = global_key_idx % page_size;
                    int page_idx = kv_page_indices[kv_start_page_pos + page_offset];
                    int kv_head = map_query_to_kv_head(h, num_query_heads, num_kv_heads);
                    uint base_addr = calculate_kv_address(in_page_offset, page_size, kv_head_dim, head_size, page_idx, kv_head);

                    // DEBUG: Log first key access details (F32 kernel)
                    if (qo_idx == 0 && h == 0 && block_start == 0 && tid_in_tgp == 0 && debug_out != nullptr) {
                        debug_out[19] = (float)page_idx;          // Page ID being accessed
                        debug_out[20] = (float)base_addr;         // Calculated base address
                        debug_out[21] = paged_k_cache[base_addr]; // First key value accessed
                        debug_out[22] = (float)in_page_offset;    // Position within page
                        debug_out[23] = (float)page_offset;       // Which page in sequence
                    }

                    for (int d = 0; d < head_size; ++d) {
                        k_block[tid_in_tgp][d] = paged_k_cache[base_addr + d];
                        v_block[tid_in_tgp][d] = paged_v_cache[base_addr + d];
                    }
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // --- Manual loop unrolling for dot product (F32) ---
            // Use manual unrolling instead of simdgroup operations to avoid cross-key issues
            float score = 0.0f;
            int global_key_idx_score = block_start + tid_in_tgp;

            if (tid_in_tgp < KERNEL_BLOCK_SIZE && global_key_idx_score < effective_kv_len) {
                // --- OPTIMIZATION 2: Vectorized memory access with float4 (F32) ---
                int d = 0;
                // Process 4 dimensions at a time using vectorized operations for F32
                if ((head_size & 3) == 0) {  // Head size is multiple of 4
                    for (; d < head_size; d += 4) {
                        // Load 4 float values as vector
                        float4 q_vec = *reinterpret_cast<threadgroup const float4*>(&q_s[d]);
                        float4 k_vec = *reinterpret_cast<threadgroup const float4*>(&k_block[tid_in_tgp][d]);

                        // Vector dot product
                        float4 prod = q_vec * k_vec;
                        score += prod.x + prod.y + prod.z + prod.w;
                    }
                } else {
                    // Fallback to manual unrolling for non-multiple-of-4 dimensions
                    for (; d < (head_size & ~3); d += 4) {
                        score += q_s[d] * k_block[tid_in_tgp][d];
                        score += q_s[d+1] * k_block[tid_in_tgp][d+1];
                        score += q_s[d+2] * k_block[tid_in_tgp][d+2];
                        score += q_s[d+3] * k_block[tid_in_tgp][d+3];
                    }
                }

                // Handle remaining dimensions
                for (; d < head_size; ++d) {
                    score += q_s[d] * k_block[tid_in_tgp][d];
                }

                score *= scale;
            } else {
                score = -INFINITY;
            }

            // --- Simdgroup reduction for block max ---
            float m_j = simdgroup_max_reduction(
                (tid_in_tgp < KERNEL_BLOCK_SIZE) ? score : -INFINITY,
                simd_scratch, tid_in_tgp, simd_lane_id, simd_group_id, KERNEL_BLOCK_SIZE
            );

            // Update global max and rescale accumulators
            threadgroup float m_prev;
            if (tid_in_tgp == 0) {
                m_prev = m_i;
                m_i = max(m_i, m_j);
                simd_scratch[0] = m_prev; // Store for broadcast
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            m_prev = simd_scratch[0]; // Broadcast previous max

            float scale_factor = fast::exp(m_prev - m_i);
            if (tid_in_tgp == 0) {
                l_i *= scale_factor;
            }
            for (int d = tid_in_tgp; d < head_size; d += TGP_SIZE) {
                acc_i[d] *= scale_factor;
            }

            // --- Fused weight computation with V accumulation ---
            float w = (score > -INFINITY) ? fast::exp(score - m_i) : 0.0f;

            // Simdgroup sum for weights
            float l_j = simdgroup_sum_reduction(
                (tid_in_tgp < KERNEL_BLOCK_SIZE) ? w : 0.0f,
                simd_scratch, tid_in_tgp, simd_lane_id, simd_group_id, KERNEL_BLOCK_SIZE
            );

            if (tid_in_tgp == 0) {
                l_i += l_j;
            }

            // Store weights for this block (needed for correct V accumulation)
            threadgroup float w_block[KERNEL_BLOCK_SIZE];
            if (tid_in_tgp < KERNEL_BLOCK_SIZE) {
                w_block[tid_in_tgp] = w;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // --- Vectorized V accumulation with loop unrolling for F32 ---
            int num_keys_in_block = min((int)KERNEL_BLOCK_SIZE, effective_kv_len - block_start);

            // Process dimensions in batches for better cache utilization
            for (int d = tid_in_tgp; d < head_size; d += TGP_SIZE) {
                float sum_wv_d = 0.0f;

                // --- Vectorized weight-value multiplication (F32) ---
                int j = 0;
                // Process 4 weights and values at once using vector operations
                if ((num_keys_in_block & 3) == 0 && KERNEL_BLOCK_SIZE >= 4) {
                    for (; j < num_keys_in_block; j += 4) {
                        // Load 4 weights as vector
                        float4 w_vec = *reinterpret_cast<threadgroup const float4*>(&w_block[j]);

                        // Load 4 V values for current dimension
                        float4 v_vec = float4(v_block[j][d], v_block[j+1][d], v_block[j+2][d], v_block[j+3][d]);

                        // Vector multiply and sum
                        float4 prod = w_vec * v_vec;
                        sum_wv_d += prod.x + prod.y + prod.z + prod.w;
                    }
                } else {
                    // Fallback to manual unrolling for non-multiple-of-4 cases
                    for (; j < (num_keys_in_block & ~3); j += 4) {
                        sum_wv_d += w_block[j] * v_block[j][d];
                        sum_wv_d += w_block[j+1] * v_block[j+1][d];
                        sum_wv_d += w_block[j+2] * v_block[j+2][d];
                        sum_wv_d += w_block[j+3] * v_block[j+3][d];
                    }
                }

                // Handle remaining keys
                for (; j < num_keys_in_block; ++j) {
                    sum_wv_d += w_block[j] * v_block[j][d];
                }

                acc_i[d] += sum_wv_d;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        // --- Finalization for this head ---
        int out_base = int(qo_idx) * head_dim + h * head_size;
        for (int d = tid_in_tgp; d < head_size; d += TGP_SIZE) {
            if (l_i > 1e-9f) {
                output[out_base + d] = acc_i[d] / l_i;
            } else {
                output[out_base + d] = 0.0f;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// --- One TG per (qo, head) Mapping ---
// Each threadgroup handles exactly one (query_token, head) pair instead of looping over heads
// Benefits: Better GPU occupancy for small batches, multiplies TG count by num_query_heads

kernel void batch_prefill_attention_unified_bf16_per_head_kernel(
    device const half* q_input [[buffer(0)]],
    device const half* paged_k_cache [[buffer(1)]],
    device const half* paged_v_cache [[buffer(2)]],
    device const int* qo_indptr [[buffer(3)]],
    device const int* kv_page_indptr [[buffer(4)]],
    device const int* kv_page_indices [[buffer(5)]],
    device const int* kv_last_page_lens [[buffer(6)]],
    device half* output [[buffer(7)]],
    constant Params& params [[buffer(8)]],
    device float* debug_out [[buffer(9)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint tid_in_tgp [[thread_index_in_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]
) {
    // Read parameters from the uniform buffer
    const int num_qo = params.num_qo;
    const int head_dim = params.head_dim;
    const int kv_head_dim = params.kv_head_dim;
    const int head_size = params.head_size;
    const int page_size = params.page_size;
    const int num_query_heads = params.num_query_heads;
    const int num_kv_heads = params.num_kv_heads;
    const float scale = params.scale;

    // NEW: Each threadgroup handles one (qo, head) pair
    // Grid dimensions: (num_qo * num_query_heads, 1, 1)
    uint linear_tid = tgid.x;
    uint qo_idx = linear_tid / num_query_heads;
    uint h = linear_tid % num_query_heads;

    if (qo_idx >= uint(num_qo)) return;

    // Check if head_size exceeds our threadgroup memory limit
    if (head_size > MAX_HEAD_DIM) return;

    const int num_simd_groups = TGP_SIZE / SIMD_SIZE;

    // --- Shared Memory Declaration ---
    // Same memory layout as Priority 0 kernel but only for single head
    threadgroup half q_s[MAX_HEAD_DIM];
    threadgroup half k_block[KERNEL_BLOCK_SIZE][MAX_HEAD_DIM];
    threadgroup half v_block[KERNEL_BLOCK_SIZE][MAX_HEAD_DIM];

    // Online softmax accumulators (single head only)
    threadgroup float m_i;
    threadgroup float l_i;
    threadgroup float acc_i[MAX_HEAD_DIM];

    // Simdgroup reduction scratchpad
    threadgroup float simd_scratch[4];

    // --- Get Sequence and KV Page Information ---
    int seq_id = find_sequence_id(qo_indptr, int(qo_idx));
    int kv_start_page_pos = kv_page_indptr[seq_id];
    int kv_end_page_pos = kv_page_indptr[seq_id + 1];
    int num_pages = kv_end_page_pos - kv_start_page_pos;

    if (num_pages <= 0) {
        // Zero out this head's output
        int output_base = int(qo_idx) * head_dim + int(h) * head_size;
        for (int d = tid_in_tgp; d < head_size; d += TGP_SIZE) {
            output[output_base + d] = 0.0h;
        }
        return;
    }

    int last_page_len = kv_last_page_lens[seq_id];
    int total_kv_len = (num_pages - 1) * page_size + last_page_len;
    int seq_start = qo_indptr[seq_id];
    int seq_pos = int(qo_idx) - seq_start;
    seq_pos = max(seq_pos, 0);
    int effective_kv_len = min(total_kv_len, seq_pos + 1);

    if (effective_kv_len <= 0) {
        int output_base = int(qo_idx) * head_dim + int(h) * head_size;
        for (int d = tid_in_tgp; d < head_size; d += TGP_SIZE) {
            output[output_base + d] = 0.0h;
        }
        return;
    }

    // --- Single Head Processing (no loop needed) ---
    // Initialize online softmax state
    if (tid_in_tgp == 0) {
        m_i = -INFINITY;
        l_i = 0.0f;
    }
    for (int d = tid_in_tgp; d < head_size; d += TGP_SIZE) {
        acc_i[d] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // --- Load Query slice for this head into Shared Memory ---
    int q_base = int(qo_idx) * head_dim + int(h) * head_size;
    for (int d = tid_in_tgp; d < head_size; d += TGP_SIZE) {
        q_s[d] = q_input[q_base + d];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // --- Main Loop: Process KV Cache in Blocks ---
    for (int block_start = 0; block_start < effective_kv_len; block_start += KERNEL_BLOCK_SIZE) {
        // Load K/V for this block and head slice - VECTORIZED
        if (tid_in_tgp < KERNEL_BLOCK_SIZE) {
            int global_key_idx = block_start + tid_in_tgp;
            if (global_key_idx < effective_kv_len) {
                int page_offset = global_key_idx / page_size;
                int in_page_offset = global_key_idx % page_size;
                int page_idx = kv_page_indices[kv_start_page_pos + page_offset];
                int kv_head = map_query_to_kv_head(int(h), num_query_heads, num_kv_heads);
                uint base_addr = calculate_kv_address(in_page_offset, page_size, kv_head_dim, head_size, page_idx, kv_head);

                // VECTORIZATION: Use half4 loads where possible for better bandwidth
                int d = 0;
                for (; d + 3 < head_size; d += 4) {
                    half4 k_vec = *((device half4*)(paged_k_cache + base_addr + d));
                    half4 v_vec = *((device half4*)(paged_v_cache + base_addr + d));

                    k_block[tid_in_tgp][d] = k_vec.x;
                    k_block[tid_in_tgp][d+1] = k_vec.y;
                    k_block[tid_in_tgp][d+2] = k_vec.z;
                    k_block[tid_in_tgp][d+3] = k_vec.w;

                    v_block[tid_in_tgp][d] = v_vec.x;
                    v_block[tid_in_tgp][d+1] = v_vec.y;
                    v_block[tid_in_tgp][d+2] = v_vec.z;
                    v_block[tid_in_tgp][d+3] = v_vec.w;
                }
                // Handle remaining elements
                for (; d < head_size; ++d) {
                    k_block[tid_in_tgp][d] = paged_k_cache[base_addr + d];
                    v_block[tid_in_tgp][d] = paged_v_cache[base_addr + d];
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // --- Split-d parallel score computation with vectorization ---
        float score = 0.0f;
        int global_key_idx_score = block_start + tid_in_tgp;

        if (tid_in_tgp < KERNEL_BLOCK_SIZE && global_key_idx_score < effective_kv_len) {
            // VECTORIZED dot product computation
            float acc = 0.0f;
            int d = 0;
            for (; d + 3 < head_size; d += 4) {
                half4 q_vec = half4(q_s[d], q_s[d+1], q_s[d+2], q_s[d+3]);
                half4 k_vec = half4(k_block[tid_in_tgp][d], k_block[tid_in_tgp][d+1],
                                   k_block[tid_in_tgp][d+2], k_block[tid_in_tgp][d+3]);

                acc += float(q_vec.x) * float(k_vec.x);
                acc += float(q_vec.y) * float(k_vec.y);
                acc += float(q_vec.z) * float(k_vec.z);
                acc += float(q_vec.w) * float(k_vec.w);
            }
            // Handle remaining elements
            for (; d < head_size; ++d) {
                acc += float(q_s[d]) * float(k_block[tid_in_tgp][d]);
            }
            score = acc * scale;
        } else {
            score = -INFINITY;
        }

        // --- Simdgroup reduction for block max ---
        float m_j = simdgroup_max_reduction(
            (tid_in_tgp < KERNEL_BLOCK_SIZE) ? score : -INFINITY,
            simd_scratch, tid_in_tgp, simd_lane_id, simd_group_id, KERNEL_BLOCK_SIZE
        );

        // Update global max and rescale accumulators
        threadgroup float m_prev;
        if (tid_in_tgp == 0) {
            m_prev = m_i;
            m_i = max(m_i, m_j);
            simd_scratch[0] = m_prev;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        m_prev = simd_scratch[0];

        float scale_factor = fast::exp(m_prev - m_i);
        if (tid_in_tgp == 0) {
            l_i *= scale_factor;
        }
        for (int d = tid_in_tgp; d < head_size; d += TGP_SIZE) {
            acc_i[d] *= scale_factor;
        }

        // --- Fused weight computation with V accumulation ---
        float w = (score > -INFINITY) ? fast::exp(score - m_i) : 0.0f;

        // Simdgroup sum for weights
        float l_j = simdgroup_sum_reduction(
            (tid_in_tgp < KERNEL_BLOCK_SIZE) ? w : 0.0f,
            simd_scratch, tid_in_tgp, simd_lane_id, simd_group_id, KERNEL_BLOCK_SIZE
        );

        if (tid_in_tgp == 0) {
            l_i += l_j;
        }

        // Store weights for correct V accumulation
        threadgroup float w_block[KERNEL_BLOCK_SIZE];
        if (tid_in_tgp < KERNEL_BLOCK_SIZE) {
            w_block[tid_in_tgp] = w;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // VECTORIZED V accumulation with correct weights
        int dims_per_thread = (head_size + TGP_SIZE - 1) / TGP_SIZE;
        for (int i = 0; i < dims_per_thread; ++i) {
            int d = tid_in_tgp * dims_per_thread + i;
            if (d < head_size) {
                float sum_wv_d = 0.0f;
        int num_keys_in_block = min((int)KERNEL_BLOCK_SIZE, effective_kv_len - block_start);

                // Use correct per-key weights stored in w_block
                for (int j = 0; j < num_keys_in_block; ++j) {
                    sum_wv_d += w_block[j] * float(v_block[j][d]);
                }

                acc_i[d] += sum_wv_d;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // --- Finalization and Output ---
    if (tid_in_tgp == 0 && l_i == 0.0f) {
        l_i = 1.0f; // Avoid division by zero
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float inv_l_i = 1.0f / l_i;
    int output_base = int(qo_idx) * head_dim + int(h) * head_size;

    // VECTORIZED output writing
    int d = 0;
    for (; d + 3 < head_size && d + 3 < MAX_HEAD_DIM; d += 4) {
        if (tid_in_tgp * 4 + 3 < head_size) {
            half4 result = half4(
                half(acc_i[tid_in_tgp * 4] * inv_l_i),
                half(acc_i[tid_in_tgp * 4 + 1] * inv_l_i),
                half(acc_i[tid_in_tgp * 4 + 2] * inv_l_i),
                half(acc_i[tid_in_tgp * 4 + 3] * inv_l_i)
            );
            *((device half4*)(output + output_base + tid_in_tgp * 4)) = result;
        }
    }
    // Handle remaining elements with regular stride
    for (int d = tid_in_tgp; d < head_size; d += TGP_SIZE) {
        output[output_base + d] = half(acc_i[d] * inv_l_i);
    }
}
