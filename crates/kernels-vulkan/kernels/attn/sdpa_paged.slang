#include "common/bf16.slang"
#include "attn/sdpa_online.slang"

#ifndef PIE_HEAD_DIM
#error "sdpa_paged requires -DPIE_HEAD_DIM"
#endif

#if defined(PIE_TILED)
#define PIE_NUMTHREADS [numthreads(32, 32, 1)]
#else
#define PIE_NUMTHREADS [numthreads(PIE_DECODE_THREADS, 1, 1)]
#endif

PIE_BUFFER_RO(0, queries);
PIE_BUFFER_RO(1, k_pages);
PIE_BUFFER_RO(2, v_pages);
PIE_BUFFER_RW(3, out_);
[[vk::binding(4)]] StructuredBuffer<int> position_ids;
[[vk::binding(5)]] StructuredBuffer<int> req_of_token;
[[vk::binding(6)]] StructuredBuffer<uint> kv_page_indices;
[[vk::binding(7)]] StructuredBuffer<uint> kv_page_indptr;
[[vk::binding(8)]] StructuredBuffer<uint8_t> attention_mask;
[[vk::binding(9)]] StructuredBuffer<uint8_t> attention_mask_enabled;

PIE_BUFFER_RO(10, sinks);
#if defined(PIE_SPLIT)

[[vk::binding(11)]] RWStructuredBuffer<float> part_out;
[[vk::binding(12)]] RWStructuredBuffer<float> part_lse;
#if defined(PIE_SPLIT_LSE)
[[vk::binding(13)]] RWStructuredBuffer<float> lse_out;
#endif
#elif defined(PIE_LSE)

[[vk::binding(11)]] RWStructuredBuffer<float> lse_out;
#endif

struct Push {
    int gqa_factor;
    int page_size;
    int n_kv_heads;
    float scale;
    uint attention_mask_stride;
    int window;
#if defined(PIE_TILED)
    int n_rows;
#endif
#if defined(PIE_SPLIT)
    int splits;
    int n_rows;
    int min_keys;
#endif
};
[[vk::push_constant]] ConstantBuffer<Push> pc;

#if defined(PIE_SPLIT)

static const float PIE_SPLIT_EMPTY = asfloat(0xff800000u);
static const float PIE_SPLIT_DONE = asfloat(0x7f800000u);
#endif

uint page_slot(int req, int kp) {
    uint page_ix = uint(kp / pc.page_size);
    uint page_off = uint(kp % pc.page_size);
    uint phys = kv_page_indices[kv_page_indptr[req] + page_ix];
    return phys * uint(pc.page_size) + page_off;
}

bool keeps(uint row, int kp, int q_pos, int start) {
    if (kp > q_pos || kp < start) return false;
    if (attention_mask_enabled[row] != uint8_t(0)) {
        if (uint(kp) >= pc.attention_mask_stride) return false;
        if (attention_mask[row * pc.attention_mask_stride + uint(kp)] == uint8_t(0)) return false;
    }
    return true;
}

int window_start(int q_pos) {
    return (pc.window > 0 && q_pos >= pc.window) ? (q_pos - pc.window + 1) : 0;
}

uint row_base(uint row, uint q_head, uint n_q_heads) {
    return (row * n_q_heads + q_head) * uint(PIE_HEAD_DIM);
}

#if defined(PIE_TILED)

#define PIE_LANES_PER_ROW 32u
#define PIE_D_PER_LANE (uint(PIE_HEAD_DIM) / PIE_LANES_PER_ROW)

groupshared float pie_sdpa_part[32][32];
groupshared int pie_sdpa_last;

void tiled_tile(uint q_head, uint n_q_heads, uint3 group, uint3 local, uint group_index) {
    uint lane = local.x;
    uint slot_y = local.y;
    uint row = group.y * 32u + slot_y;
    bool live = row < uint(pc.n_rows);

    int req = live ? req_of_token[row] : 0;
    int q_pos = live ? position_ids[row] : -1;
    int start = window_start(q_pos);
    uint q_base = row_base(row, q_head, n_q_heads);
    int kv_head = int(q_head) / pc.gqa_factor;

    if (group_index == 0u) pie_sdpa_last = -1;
    GroupMemoryBarrierWithGroupSync();
    if (lane == 0u) { int previous; InterlockedMax(pie_sdpa_last, q_pos, previous); }
    GroupMemoryBarrierWithGroupSync();
    int last = pie_sdpa_last;

    float max_score = PIE_SDPA_NEG_INF;
    float sum_exp = 0.0;
    float out_acc[PIE_D_PER_LANE];
    for (uint i = 0u; i < PIE_D_PER_LANE; i++) out_acc[i] = 0.0;

    for (int kp = 0; kp <= last; kp++) {
        bool take = live && keeps(row, kp, q_pos, start);
        uint base = 0u;
        float partial = 0.0;
        if (take) {
            uint slot = page_slot(req, kp);
            base = (slot * uint(pc.n_kv_heads) + uint(kv_head)) * uint(PIE_HEAD_DIM);
            for (uint i = 0u; i < PIE_D_PER_LANE; i++) {
                uint d = lane + i * PIE_LANES_PER_ROW;
                partial += pc.scale * PIE_LOAD(queries[q_base + d]) * PIE_LOAD(k_pages[base + d]);
            }
        }
        pie_sdpa_part[slot_y][lane] = partial;
        GroupMemoryBarrierWithGroupSync();
        for (uint half_ = PIE_LANES_PER_ROW >> 1; half_ > 0u; half_ >>= 1) {
            if (lane < half_) pie_sdpa_part[slot_y][lane] += pie_sdpa_part[slot_y][lane + half_];
            GroupMemoryBarrierWithGroupSync();
        }
        float score = pie_sdpa_part[slot_y][0];
        if (take) {
            float hist, prob;
            sdpa_online_update(score, max_score, sum_exp, hist, prob);
            for (uint i = 0u; i < PIE_D_PER_LANE; i++) {
                uint d = lane + i * PIE_LANES_PER_ROW;
                out_acc[i] = out_acc[i] * hist + prob * PIE_LOAD(v_pages[base + d]);
            }
        }

        GroupMemoryBarrierWithGroupSync();
    }

#if defined(PIE_LSE)
    if (live && lane == 0u) {
        lse_out[row * n_q_heads + q_head] = sdpa_lse_base2(max_score, sum_exp);
    }
#endif
    if (!live) return;
    for (uint i = 0u; i < PIE_D_PER_LANE; i++) {
        uint d = lane + i * PIE_LANES_PER_ROW;
        out_[q_base + d] = PIE_STORE(sum_exp == 0.0 ? out_acc[i] : out_acc[i] / sum_exp);
    }
}
#else

#define PIE_DECODE_THREADS 256
#define PIE_DECODE_MAX_SUBGROUPS 8
#define PIE_DPL (PIE_HEAD_DIM / 32)

groupshared float pie_sdpa_acc[PIE_DECODE_MAX_SUBGROUPS][PIE_HEAD_DIM];
groupshared float pie_sdpa_max[PIE_DECODE_MAX_SUBGROUPS];
groupshared float pie_sdpa_sum[PIE_DECODE_MAX_SUBGROUPS];

void decode_walk(uint row, uint q_head, uint n_q_heads, int k_lo, int k_hi,
                 int q_pos, int start, out float merged_max, out float merged_sum,
                 out uint subgroups) {
    int req = req_of_token[row];
    uint base_row = row_base(row, q_head, n_q_heads);
    int kv_head = int(q_head) / pc.gqa_factor;

    uint sg = gl_SubgroupID;
    uint lane = WaveGetLaneIndex();
    uint width = WaveGetLaneCount();
    subgroups = min(gl_NumSubgroups, uint(PIE_DECODE_MAX_SUBGROUPS));
    bool walks = sg < subgroups;

    uint per_lane = uint(PIE_HEAD_DIM) / width;

    float qreg[PIE_DPL];
    float acc[PIE_DPL];
    [ForceUnroll] for (uint j = 0u; j < uint(PIE_DPL); j++) {
        uint dd = lane + j * width;
        qreg[j] = (j < per_lane) ? pc.scale * PIE_LOAD(queries[base_row + dd]) : 0.0;
        acc[j] = 0.0;
    }

    float max_score = PIE_SDPA_NEG_INF;
    float sum_exp = 0.0;
    if (walks) {
        for (int kp = k_lo + int(sg); kp <= k_hi; kp += int(subgroups)) {
            if (!keeps(row, kp, q_pos, start)) continue;
            uint slot = page_slot(req, kp);
            uint base = (slot * uint(pc.n_kv_heads) + uint(kv_head)) * uint(PIE_HEAD_DIM);
            float partial = 0.0;
            [ForceUnroll] for (uint j = 0u; j < uint(PIE_DPL); j++) {
                if (j < per_lane) partial += qreg[j] * PIE_LOAD(k_pages[base + lane + j * width]);
            }
            float score = WaveActiveSum(partial);
            float hist, prob;
            sdpa_online_update(score, max_score, sum_exp, hist, prob);
            [ForceUnroll] for (uint j = 0u; j < uint(PIE_DPL); j++) {
                if (j < per_lane) acc[j] = acc[j] * hist + prob * PIE_LOAD(v_pages[base + lane + j * width]);
            }
        }
        [ForceUnroll] for (uint j = 0u; j < uint(PIE_DPL); j++) {
            if (j < per_lane) pie_sdpa_acc[sg][lane + j * width] = acc[j];
        }
        if (WaveIsFirstLane()) {
            pie_sdpa_max[sg] = max_score;
            pie_sdpa_sum[sg] = sum_exp;
        }
    }
    GroupMemoryBarrierWithGroupSync();

    merged_max = PIE_SDPA_NEG_INF;
    for (uint s = 0u; s < subgroups; s++) merged_max = max(merged_max, pie_sdpa_max[s]);
    merged_sum = 0.0;
    for (uint s = 0u; s < subgroups; s++) {
        merged_sum += pie_sdpa_sum[s] * exp(pie_sdpa_max[s] - merged_max);
    }
}

float decode_fold(uint d, float merged_max, uint subgroups) {
    float out_acc = 0.0;
    for (uint s = 0u; s < subgroups; s++) {
        out_acc += pie_sdpa_acc[s][d] * exp(pie_sdpa_max[s] - merged_max);
    }
    return out_acc;
}

#if defined(PIE_SPLIT)
void decode_split(uint row, uint q_head, uint split, uint tid, uint n_q_heads) {
    int q_pos = position_ids[row];
    int start = window_start(q_pos);

    int keys = q_pos - start + 1;
    int by_length = (keys + pc.min_keys - 1) / max(pc.min_keys, 1);
    int usable = clamp(by_length, 1, pc.splits);
    uint slice = (split * uint(pc.n_rows) + row) * n_q_heads + q_head;
    if (int(split) >= usable) {
        if (tid == 0u) part_lse[slice] = PIE_SPLIT_EMPTY;
        return;
    }
    int chunk = (keys + usable - 1) / usable;
    int k_lo = start + int(split) * chunk;
    int k_hi = min(q_pos, k_lo + chunk - 1);

    float merged_max, merged_sum;
    uint subgroups;
    decode_walk(row, q_head, n_q_heads, k_lo, k_hi, q_pos, start,
                merged_max, merged_sum, subgroups);

    if (usable == 1) {
        if (tid == 0u) {
            part_lse[slice] = PIE_SPLIT_DONE;
#if defined(PIE_SPLIT_LSE)
            lse_out[row * n_q_heads + q_head] = sdpa_lse_base2(merged_max, merged_sum);
#endif
        }
        uint base_row = row_base(row, q_head, n_q_heads);
        for (uint d = tid; d < uint(PIE_HEAD_DIM); d += uint(PIE_DECODE_THREADS)) {
            float out_acc = decode_fold(d, merged_max, subgroups);
            out_[base_row + d] = PIE_STORE(merged_sum == 0.0 ? out_acc : out_acc / merged_sum);
        }
        return;
    }

    if (tid == 0u) {
        part_lse[slice] = sdpa_lse_base2(merged_max, merged_sum);
    }
    uint out_base = slice * uint(PIE_HEAD_DIM);
    for (uint d = tid; d < uint(PIE_HEAD_DIM); d += uint(PIE_DECODE_THREADS)) {
        float out_acc = decode_fold(d, merged_max, subgroups);
        part_out[out_base + d] = merged_sum == 0.0 ? 0.0 : out_acc / merged_sum;
    }
}
#else
void decode_one(uint row, uint q_head, uint tid, uint n_q_heads) {
    int q_pos = position_ids[row];
    int start = window_start(q_pos);
    uint base_row = row_base(row, q_head, n_q_heads);

    float merged_max, merged_sum;
    uint subgroups;
    decode_walk(row, q_head, n_q_heads, start, q_pos, q_pos, start,
                merged_max, merged_sum, subgroups);

#if defined(PIE_LSE)
    if (tid == 0u) {
        lse_out[row * n_q_heads + q_head] = sdpa_lse_base2(merged_max, merged_sum);
    }
#endif
    for (uint d = tid; d < uint(PIE_HEAD_DIM); d += uint(PIE_DECODE_THREADS)) {
        float out_acc = decode_fold(d, merged_max, subgroups);
        out_[base_row + d] = PIE_STORE(merged_sum == 0.0 ? out_acc : out_acc / merged_sum);
    }
}
#endif
#endif

PIE_NUMTHREADS
void main(uint3 group : SV_GroupID, uint3 local : SV_GroupThreadID, uint group_index : SV_GroupIndex) {
#if defined(PIE_TILED)
    tiled_tile(group.x, gl_NumWorkGroups.x, group, local, group_index);
#elif defined(PIE_SPLIT)
    decode_split(group.y, group.x, group.z, local.x, gl_NumWorkGroups.x);
#else
    decode_one(group.y, group.x, local.x, gl_NumWorkGroups.x);
#endif
}

// pie:instantiate sdpa_paged_decode_bfloat16_d_64 PIE_HEAD_DIM=64
// pie:instantiate sdpa_paged_decode_bfloat16_d_128 PIE_HEAD_DIM=128
// pie:instantiate sdpa_paged_decode_bfloat16_d_256 PIE_HEAD_DIM=256
// pie:instantiate sdpa_paged_decode_bfloat16_d_512 PIE_HEAD_DIM=512
// pie:instantiate sdpa_paged_decode_lse_bfloat16_d_64 PIE_HEAD_DIM=64 PIE_LSE=1
// pie:instantiate sdpa_paged_decode_lse_bfloat16_d_128 PIE_HEAD_DIM=128 PIE_LSE=1
// pie:instantiate sdpa_paged_decode_lse_bfloat16_d_256 PIE_HEAD_DIM=256 PIE_LSE=1
// pie:instantiate sdpa_paged_decode_lse_bfloat16_d_512 PIE_HEAD_DIM=512 PIE_LSE=1
// pie:instantiate sdpa_paged_split_bfloat16_d_64 PIE_HEAD_DIM=64 PIE_SPLIT=1
// pie:instantiate sdpa_paged_split_bfloat16_d_128 PIE_HEAD_DIM=128 PIE_SPLIT=1
// pie:instantiate sdpa_paged_split_bfloat16_d_256 PIE_HEAD_DIM=256 PIE_SPLIT=1
// pie:instantiate sdpa_paged_split_bfloat16_d_512 PIE_HEAD_DIM=512 PIE_SPLIT=1
// pie:instantiate sdpa_paged_split_lse_bfloat16_d_64 PIE_HEAD_DIM=64 PIE_SPLIT=1 PIE_SPLIT_LSE=1
// pie:instantiate sdpa_paged_split_lse_bfloat16_d_128 PIE_HEAD_DIM=128 PIE_SPLIT=1 PIE_SPLIT_LSE=1
// pie:instantiate sdpa_paged_split_lse_bfloat16_d_256 PIE_HEAD_DIM=256 PIE_SPLIT=1 PIE_SPLIT_LSE=1
// pie:instantiate sdpa_paged_split_lse_bfloat16_d_512 PIE_HEAD_DIM=512 PIE_SPLIT=1 PIE_SPLIT_LSE=1
// pie:instantiate sdpa_paged_tiled_bfloat16_d_64 PIE_HEAD_DIM=64 PIE_TILED=1
// pie:instantiate sdpa_paged_tiled_bfloat16_d_128 PIE_HEAD_DIM=128 PIE_TILED=1
// pie:instantiate sdpa_paged_tiled_bfloat16_d_256 PIE_HEAD_DIM=256 PIE_TILED=1
// pie:instantiate sdpa_paged_tiled_bfloat16_d_512 PIE_HEAD_DIM=512 PIE_TILED=1
// pie:instantiate sdpa_paged_tiled_lse_bfloat16_d_64 PIE_HEAD_DIM=64 PIE_TILED=1 PIE_LSE=1
// pie:instantiate sdpa_paged_tiled_lse_bfloat16_d_128 PIE_HEAD_DIM=128 PIE_TILED=1 PIE_LSE=1
// pie:instantiate sdpa_paged_tiled_lse_bfloat16_d_256 PIE_HEAD_DIM=256 PIE_TILED=1 PIE_LSE=1
// pie:instantiate sdpa_paged_tiled_lse_bfloat16_d_512 PIE_HEAD_DIM=512 PIE_TILED=1 PIE_LSE=1
