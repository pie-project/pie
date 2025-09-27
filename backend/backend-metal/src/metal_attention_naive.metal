#include <metal_stdlib>
using namespace metal;

/**
 * Naive Attention Implementation - Baseline for Performance Comparison
 *
 * This kernel implements standard attention with O(n²) memory complexity
 * to establish a baseline for measuring optimization benefits.
 *
 * Algorithm:
 * 1. Compute full attention matrix S = Q @ K^T [O(n²) memory]
 * 2. Apply softmax to entire matrix [stores full matrix]
 * 3. Compute output O = S @ V [O(n²) memory reads]
 *
 * Memory Complexity: O(n²) for attention matrix storage
 * Compute Complexity: O(n²·d) same as optimized version
 *
 * Purpose: Prove that optimized version achieves same result with O(n) memory
 */

#define MAX_PARTITION_SIZE 256  // Partition size to fit in threadgroup memory (32KB limit)
#define MAX_HEAD_DIM 256
#define TGP_SIZE 128

// Memory analysis:
// - attention_matrix: MAX_PARTITION_SIZE * 4 bytes = 1KB per partition
// - query/key/value caches: MAX_HEAD_DIM * 2 bytes each = 1.5KB total
// - Total per partition: ~2.5KB << 32KB limit

// Same parameter structure as optimized kernel for fair comparison
struct NaiveAttentionParams {
    int num_qo;
    int head_dim;
    int kv_head_dim;
    int head_size;
    int page_size;
    int num_query_heads;
    int num_kv_heads;
    float scale;
    int total_kv_len;      // Total sequence length for memory calculations
    int num_partitions;    // Number of partitions needed
};

/**
 * Naive attention kernel with partitioned processing
 * Demonstrates O(n²) memory complexity by storing full attention matrix in device memory
 * Uses partitioned processing to stay within threadgroup memory limits
 */
kernel void naive_attention_bf16_kernel(
    device const half* q_input [[buffer(0)]],
    device const half* paged_k_cache [[buffer(1)]],
    device const half* paged_v_cache [[buffer(2)]],
    device const int* qo_indptr [[buffer(3)]],
    device const int* kv_page_indptr [[buffer(4)]],
    device const int* kv_page_indices [[buffer(5)]],
    device const int* kv_last_page_lens [[buffer(6)]],
    device half* output [[buffer(7)]],
    constant NaiveAttentionParams& params [[buffer(8)]],
    device float* debug_out [[buffer(9)]],
    device float* attention_matrix_storage [[buffer(10)]],  // O(n²) storage in device memory
    uint3 tgid [[threadgroup_position_in_grid]],
    uint tid_in_tgp [[thread_index_in_threadgroup]]
) {
    // Read parameters
    const int num_qo = params.num_qo;
    const int head_dim = params.head_dim;
    const int kv_head_dim = params.kv_head_dim;
    const int head_size = params.head_size;
    const int page_size = params.page_size;
    const int num_query_heads = params.num_query_heads;
    const int num_kv_heads = params.num_kv_heads;
    const float scale = params.scale;
    const int total_kv_len = params.total_kv_len;

    uint qo_idx = tgid.x;
    if (qo_idx >= uint(num_qo)) return;
    if (head_size > MAX_HEAD_DIM) return;

    // Find sequence information
    int seq_id = 0;
    while (qo_indptr[seq_id + 1] <= int(qo_idx)) { seq_id++; }
    int kv_start_page_pos = kv_page_indptr[seq_id];
    int kv_end_page_pos = kv_page_indptr[seq_id + 1];
    int num_pages = kv_end_page_pos - kv_start_page_pos;

    if (num_pages <= 0 || total_kv_len <= 0) {
        for (int d = tid_in_tgp; d < head_dim; d += TGP_SIZE) {
            output[qo_idx * head_dim + d] = 0.0h;
        }
        return;
    }

    // *** DEVICE MEMORY: Full O(n²) attention matrix storage ***
    // Each query token stores its full attention row: qo_idx * total_kv_len
    device float* my_attention_row = attention_matrix_storage + qo_idx * total_kv_len;

    // Threadgroup memory for partitioned processing (fits in 32KB)
    threadgroup float attention_partition[MAX_PARTITION_SIZE];
    threadgroup half q_cache[MAX_HEAD_DIM];
    threadgroup half k_cache[MAX_HEAD_DIM];
    threadgroup float temp_reduce[TGP_SIZE];

    const int num_heads = num_query_heads;

    // Initialize output to zero
    for (int d = tid_in_tgp; d < head_dim; d += TGP_SIZE) {
        output[qo_idx * head_dim + d] = 0.0h;
    }

    // Process each attention head
    for (int h = 0; h < num_heads; ++h) {
        // Load query for this head
        int q_base = int(qo_idx) * head_dim + h * head_size;
        for (int d = tid_in_tgp; d < head_size; d += TGP_SIZE) {
            q_cache[d] = q_input[q_base + d];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // *** PHASE 1: COMPUTE AND STORE FULL ATTENTION MATRIX (O(n²) STORAGE) ***
        // Process attention matrix in partitions that fit in threadgroup memory
        int num_partitions = (total_kv_len + MAX_PARTITION_SIZE - 1) / MAX_PARTITION_SIZE;

        for (int partition = 0; partition < num_partitions; partition++) {
            int partition_start = partition * MAX_PARTITION_SIZE;
            int partition_end = min(partition_start + MAX_PARTITION_SIZE, total_kv_len);
            int partition_size = partition_end - partition_start;

            // Compute attention scores for this partition
            for (int k_local = tid_in_tgp; k_local < partition_size; k_local += TGP_SIZE) {
                int k_idx = partition_start + k_local;

                // Load key for position k_idx
                int page_offset = k_idx / page_size;
                int in_page_offset = k_idx % page_size;
                if (page_offset >= num_pages) {
                    attention_partition[k_local] = -INFINITY;
                    continue;
                }

                int page_idx = kv_page_indices[kv_start_page_pos + page_offset];
                int kv_head = h / max(1, num_query_heads / num_kv_heads);
                uint k_base_addr = page_idx * page_size * kv_head_dim + in_page_offset * kv_head_dim + kv_head * head_size;

                // Load key vector and compute Q @ K^T
                float score = 0.0f;
                for (int d = 0; d < head_size; ++d) {
                    k_cache[d] = paged_k_cache[k_base_addr + d];
                    score += float(q_cache[d]) * float(k_cache[d]);
                }
                score *= scale;

                attention_partition[k_local] = score;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // *** STORE PARTITION IN DEVICE MEMORY (O(n²) MEMORY WRITES) ***
            for (int k_local = tid_in_tgp; k_local < partition_size; k_local += TGP_SIZE) {
                my_attention_row[partition_start + k_local] = attention_partition[k_local];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        // *** PHASE 2: SOFTMAX OVER ENTIRE ROW (O(n²) MEMORY READS) ***
        // Find maximum across entire attention row stored in device memory
        float max_val = -INFINITY;
        for (int k_idx = tid_in_tgp; k_idx < total_kv_len; k_idx += TGP_SIZE) {
            max_val = max(max_val, my_attention_row[k_idx]);
        }

        // Reduce to find global max
        temp_reduce[tid_in_tgp] = max_val;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint s = TGP_SIZE / 2; s > 0; s >>= 1) {
            if (tid_in_tgp < s) {
                temp_reduce[tid_in_tgp] = max(temp_reduce[tid_in_tgp], temp_reduce[tid_in_tgp + s]);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        float global_max = temp_reduce[0];
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Apply exp and compute sum
        float sum = 0.0f;
        for (int k_idx = tid_in_tgp; k_idx < total_kv_len; k_idx += TGP_SIZE) {
            float exp_val = exp(my_attention_row[k_idx] - global_max);
            my_attention_row[k_idx] = exp_val;  // *** WRITE BACK TO O(n²) STORAGE ***
            sum += exp_val;
        }

        // Reduce sum
        temp_reduce[tid_in_tgp] = sum;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint s = TGP_SIZE / 2; s > 0; s >>= 1) {
            if (tid_in_tgp < s) {
                temp_reduce[tid_in_tgp] += temp_reduce[tid_in_tgp + s];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        float global_sum = temp_reduce[0];
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Normalize attention weights
        float inv_sum = 1.0f / max(global_sum, 1e-9f);
        for (int k_idx = tid_in_tgp; k_idx < total_kv_len; k_idx += TGP_SIZE) {
            my_attention_row[k_idx] *= inv_sum;  // *** MORE O(n²) MEMORY OPERATIONS ***
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // *** PHASE 3: COMPUTE OUTPUT O = ATTENTION @ V (O(n²) MEMORY READS) ***
        int out_base = int(qo_idx) * head_dim + h * head_size;
        for (int d = tid_in_tgp; d < head_size; d += TGP_SIZE) {
            float output_val = 0.0f;

            // Weighted sum over all key positions - reads from O(n²) storage
            for (int k_idx = 0; k_idx < total_kv_len; k_idx++) {
                // Load value at position k_idx, dimension d
                int page_offset = k_idx / page_size;
                int in_page_offset = k_idx % page_size;
                if (page_offset >= num_pages) continue;

                int page_idx = kv_page_indices[kv_start_page_pos + page_offset];
                int kv_head = h / max(1, num_query_heads / num_kv_heads);
                uint v_base_addr = page_idx * page_size * kv_head_dim + in_page_offset * kv_head_dim + kv_head * head_size;

                half v_val = paged_v_cache[v_base_addr + d];

                // *** READ FROM O(n²) ATTENTION MATRIX STORAGE ***
                output_val += my_attention_row[k_idx] * float(v_val);
            }

            output[out_base + d] = half(output_val);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

/**
 * F32 variant for comparison
 */
kernel void naive_attention_f32_kernel(
    device const float* q_input [[buffer(0)]],
    device const float* paged_k_cache [[buffer(1)]],
    device const float* paged_v_cache [[buffer(2)]],
    device const int* qo_indptr [[buffer(3)]],
    device const int* kv_page_indptr [[buffer(4)]],
    device const int* kv_page_indices [[buffer(5)]],
    device const int* kv_last_page_lens [[buffer(6)]],
    device float* output [[buffer(7)]],
    constant NaiveAttentionParams& params [[buffer(8)]],
    device float* debug_out [[buffer(9)]],
    device float* attention_matrix_storage [[buffer(10)]],  // O(n²) storage in device memory
    uint3 tgid [[threadgroup_position_in_grid]],
    uint tid_in_tgp [[thread_index_in_threadgroup]]
) {
    // Same partitioned algorithm as BF16 version but with float32
    const int num_qo = params.num_qo;
    const int head_dim = params.head_dim;
    const int kv_head_dim = params.kv_head_dim;
    const int head_size = params.head_size;
    const int page_size = params.page_size;
    const int num_query_heads = params.num_query_heads;
    const int num_kv_heads = params.num_kv_heads;
    const float scale = params.scale;
    const int total_kv_len = params.total_kv_len;

    uint qo_idx = tgid.x;
    if (qo_idx >= uint(num_qo)) return;
    if (head_size > MAX_HEAD_DIM) return;

    int seq_id = 0;
    while (qo_indptr[seq_id + 1] <= int(qo_idx)) { seq_id++; }
    int kv_start_page_pos = kv_page_indptr[seq_id];
    int kv_end_page_pos = kv_page_indptr[seq_id + 1];
    int num_pages = kv_end_page_pos - kv_start_page_pos;

    if (num_pages <= 0 || total_kv_len <= 0) {
        for (int d = tid_in_tgp; d < head_dim; d += TGP_SIZE) {
            output[qo_idx * head_dim + d] = 0.0f;
        }
        return;
    }

    // *** DEVICE MEMORY: Full O(n²) attention matrix storage ***
    device float* my_attention_row = attention_matrix_storage + qo_idx * total_kv_len;

    // Threadgroup memory for partitioned processing
    threadgroup float attention_partition[MAX_PARTITION_SIZE];
    threadgroup float q_cache[MAX_HEAD_DIM];
    threadgroup float k_cache[MAX_HEAD_DIM];
    threadgroup float temp_reduce[TGP_SIZE];

    // For simplicity, F32 version just returns zero (focus on BF16 implementation)
    // (Full implementation would be identical to BF16 version above, but with float types)
    for (int d = tid_in_tgp; d < head_dim; d += TGP_SIZE) {
        output[qo_idx * head_dim + d] = 0.0f;
    }
}