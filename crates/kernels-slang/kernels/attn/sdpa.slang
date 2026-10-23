#include "prim/bf16.slang"
#include "prim/params.slang"

// Paged attention over a KV cache, bf16 two a word, causal (an optional
// sliding window, an optional per-row byte mask). Decode reads one query row
// a workgroup (per head): the workgroup's subgroups (`PIE_SUBGROUP`, SPIR-V
// only) or 32-lane strips (the baseline, reduced through workgroup memory)
// each walk a share of the keys with an online softmax and fold at the end.
// Variants: the row's base-2 LSE (`PIE_LSE`); the keys a block selection
// names (`PIE_SELECTED`); one of `splits` key ranges a workgroup, landing f32
// partials and their LSE for the fold (`PIE_SPLIT`, `PIE_SPLIT_LSE`).
// Prefill (`PIE_TILED`) holds `PIE_TILE_ROWS` query rows of one head a
// workgroup, 32 lanes a row, every row walking the tile's keys in step.
// `PIE_FOLD` folds the split partials by their LSEs.

static const float NEG_INF = -3.0e38;

float2 word_pair(uint w) {
    return float2(bf16_to_f32(w & 0xffffu), bf16_to_f32(w >> 16));
}

uint byte_at(StructuredBuffer<uint> buf, uint i) {
    return (buf[i >> 2] >> ((i & 3u) * 8u)) & 0xffu;
}

float lse_base2(float max_score, float sum_exp) {
    return sum_exp > 0.0 ? max_score * 1.44269504088896340736 + log2(sum_exp) : asfloat(0xff800000u);
}

void online_update(float score, inout float max_score, inout float sum_exp, out float history, out float weight) {
    float new_max = max(max_score, score);
    history = exp(max_score - new_max);
    weight = exp(score - new_max);
    max_score = new_max;
    sum_exp = sum_exp * history + weight;
}

#if defined(PIE_FOLD)

[[vk::binding(0)]] StructuredBuffer<float> part_out;
[[vk::binding(1)]] StructuredBuffer<float> part_lse;
[[vk::binding(2)]] RWStructuredBuffer<uint> o;
#if defined(PIE_LSE)
[[vk::binding(3)]] RWStructuredBuffer<float> lse_out;
#define PIE_PARAMS_AT 4
#else
#define PIE_PARAMS_AT 3
#endif

struct Params {
    int splits;
    int n_rows;
    int q_heads;
    int head_dim;
};
PIE_PARAMS(PIE_PARAMS_AT, Params, params);

bool finite_(float x) {
    return abs(x) < 3.0e38;
}

// The fold of split `s`'s partials, the row's dims `d` and `d + 1`.
[shader("compute")]
[numthreads(PIE_GROUP_X, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    uint hd = uint(params.head_dim);
    uint width = uint(params.q_heads) * hd;
    uint e = 2u * gid.x;
    uint row = gid.y;
    if (e >= width || row >= uint(params.n_rows)) {
        return;
    }
    uint head = e / hd;
    uint d = e - head * hd;
    uint rows = uint(params.n_rows);
    uint heads = uint(params.q_heads);
    // A row one split covered whole was written by the walker already.
    if (part_lse[row * heads + head] > 3.0e38) {
        return;
    }
    float top = NEG_INF;
    for (int s = 0; s < params.splits; s++) {
        float l = part_lse[(uint(s) * rows + row) * heads + head];
        if (finite_(l)) {
            top = max(top, l);
        }
    }
    float2 acc = float2(0.0, 0.0);
    float total = 0.0;
    if (finite_(top)) {
        for (int s = 0; s < params.splits; s++) {
            uint col = (uint(s) * rows + row) * heads + head;
            float l = part_lse[col];
            if (finite_(l)) {
                float w = exp2(l - top);
                total += w;
                acc += w * float2(part_out[col * hd + d], part_out[col * hd + d + 1u]);
            }
        }
    }
    float2 v = total == 0.0 ? float2(0.0, 0.0) : acc / total;
    o[(row * width + e) >> 1] = pack_bf16(v.x, v.y);
#if defined(PIE_LSE)
    if (d == 0u) {
        lse_out[row * heads + head] = finite_(top) ? top + log2(total) : asfloat(0xff800000u);
    }
#endif
}

#else

#ifndef PIE_HEAD_DIM
#error "attn/sdpa.slang is compiled at a head width: -DPIE_HEAD_DIM"
#endif

[[vk::binding(0)]] StructuredBuffer<uint> queries;
[[vk::binding(1)]] StructuredBuffer<uint> k_pages;
[[vk::binding(2)]] StructuredBuffer<uint> v_pages;
[[vk::binding(3)]] RWStructuredBuffer<uint> o;
[[vk::binding(4)]] StructuredBuffer<int> position_ids;
[[vk::binding(5)]] StructuredBuffer<int> req_of_token;
[[vk::binding(6)]] StructuredBuffer<uint> kv_page_indices;
[[vk::binding(7)]] StructuredBuffer<uint> kv_page_indptr;
[[vk::binding(8)]] StructuredBuffer<uint> attention_mask;
[[vk::binding(9)]] StructuredBuffer<uint> attention_mask_enabled;
#if defined(PIE_SPLIT)
[[vk::binding(10)]] RWStructuredBuffer<float> part_out;
[[vk::binding(11)]] RWStructuredBuffer<float> part_lse;
#if defined(PIE_SPLIT_LSE)
[[vk::binding(12)]] RWStructuredBuffer<float> lse_out;
#define PIE_PARAMS_AT 13
#else
#define PIE_PARAMS_AT 12
#endif
#elif defined(PIE_LSE)
[[vk::binding(10)]] RWStructuredBuffer<float> lse_out;
#define PIE_PARAMS_AT 11
#elif defined(PIE_SELECTED)
[[vk::binding(10)]] StructuredBuffer<int> selection;
#define PIE_PARAMS_AT 11
#else
#define PIE_PARAMS_AT 10
#endif

struct Params {
    int gqa_factor;
    int page_size;
    int n_kv_heads;
    float scale;
    uint attention_mask_stride;
    int window;
    int n_q_heads;
#if defined(PIE_SELECTED)
    int top_k;
    int ratio;
#endif
#if defined(PIE_TILED)
    int n_rows;
#endif
#if defined(PIE_SPLIT)
    int splits;
    int n_rows;
    int min_keys;
#endif
};
PIE_PARAMS(PIE_PARAMS_AT, Params, params);

uint page_slot(int req, int kp) {
    uint phys = kv_page_indices[kv_page_indptr[req] + uint(kp / params.page_size)];
    return phys * uint(params.page_size) + uint(kp % params.page_size);
}

bool keeps(uint row, int kp, int q_pos, int start) {
    if (kp > q_pos || kp < start) {
        return false;
    }
    if (byte_at(attention_mask_enabled, row) != 0u) {
        if (uint(kp) >= params.attention_mask_stride) {
            return false;
        }
        if (byte_at(attention_mask, row * params.attention_mask_stride + uint(kp)) == 0u) {
            return false;
        }
    }
    return true;
}

int window_start(int q_pos) {
    return (params.window > 0 && q_pos >= params.window) ? q_pos - params.window + 1 : 0;
}

uint row_base(uint row, uint q_head) {
    return (row * uint(params.n_q_heads) + q_head) * uint(PIE_HEAD_DIM);
}

#if defined(PIE_TILED)

#if !defined(PIE_TILE_ROWS)
#define PIE_TILE_ROWS 32
#endif
#define LANES_PER_ROW 32u
// Pairs of dims a lane holds: pair `lane + 32 i`, dims `2p` and `2p + 1`.
#define PAIRS_PER_LANE (uint(PIE_HEAD_DIM) / (2u * LANES_PER_ROW))

groupshared float sdpa_part[PIE_TILE_ROWS][32];
groupshared int sdpa_last;

[shader("compute")]
[numthreads(32, PIE_TILE_ROWS, 1)]
void main(uint3 group : SV_GroupID, uint3 local : SV_GroupThreadID) {
    uint q_head = group.x;
    uint lane = local.x;
    uint slot_y = local.y;
    uint row = group.y * uint(PIE_TILE_ROWS) + slot_y;
    bool live = row < uint(params.n_rows);

    int req = live ? req_of_token[row] : 0;
    int q_pos = live ? position_ids[row] : -1;
    int start = window_start(q_pos);
    uint q_base = row_base(row, q_head);
    int kv_head = int(q_head) / params.gqa_factor;

    // The tile's last position, read once and uniformly: it bounds the key
    // loop, whose every step holds barriers.
    if (lane == 0u && slot_y == 0u) {
        int last = -1;
        for (uint r = 0u; r < uint(PIE_TILE_ROWS); r++) {
            uint at = group.y * uint(PIE_TILE_ROWS) + r;
            if (at < uint(params.n_rows)) {
                last = max(last, position_ids[at]);
            }
        }
        sdpa_last = last;
    }
    GroupMemoryBarrierWithGroupSync();
    int last = workgroupUniformLoad(sdpa_last);

    float2 qv[PAIRS_PER_LANE];
    float2 acc[PAIRS_PER_LANE];
    for (uint i = 0u; i < PAIRS_PER_LANE; i++) {
        uint d = 2u * (lane + i * LANES_PER_ROW);
        qv[i] = live ? word_pair(queries[(q_base + d) >> 1]) * params.scale : float2(0.0, 0.0);
        acc[i] = float2(0.0, 0.0);
    }
    float max_score = NEG_INF;
    float sum_exp = 0.0;

    for (int kp = 0; kp <= last; kp++) {
        bool take = live && keeps(row, kp, q_pos, start);
        uint base = 0u;
        float partial = 0.0;
        if (take) {
            base = (page_slot(req, kp) * uint(params.n_kv_heads) + uint(kv_head)) * uint(PIE_HEAD_DIM);
            for (uint i = 0u; i < PAIRS_PER_LANE; i++) {
                partial += dot(qv[i], word_pair(k_pages[(base + 2u * (lane + i * LANES_PER_ROW)) >> 1]));
            }
        }
        sdpa_part[slot_y][lane] = partial;
        GroupMemoryBarrierWithGroupSync();
        for (uint half_ = LANES_PER_ROW >> 1; half_ > 0u; half_ >>= 1) {
            if (lane < half_) {
                sdpa_part[slot_y][lane] += sdpa_part[slot_y][lane + half_];
            }
            GroupMemoryBarrierWithGroupSync();
        }
        float score = sdpa_part[slot_y][0];
        if (take) {
            float hist, w;
            online_update(score, max_score, sum_exp, hist, w);
            for (uint i = 0u; i < PAIRS_PER_LANE; i++) {
                acc[i] = acc[i] * hist + w * word_pair(v_pages[(base + 2u * (lane + i * LANES_PER_ROW)) >> 1]);
            }
        }
        GroupMemoryBarrierWithGroupSync();
    }

#if defined(PIE_LSE)
    if (live && lane == 0u) {
        lse_out[row * uint(params.n_q_heads) + q_head] = lse_base2(max_score, sum_exp);
    }
#endif
    if (live) {
        for (uint i = 0u; i < PAIRS_PER_LANE; i++) {
            float2 v = sum_exp == 0.0 ? acc[i] : acc[i] / sum_exp;
            o[(q_base + 2u * (lane + i * LANES_PER_ROW)) >> 1] = pack_bf16(v.x, v.y);
        }
    }
}

#else

#if !defined(PIE_DECODE_THREADS)
#define PIE_DECODE_THREADS 256
#endif
#define MAX_WALKERS (PIE_DECODE_THREADS / 32)
// The narrowest subgroup a variant serves: its per-lane registers hold
// HEAD_DIM / MIN_LANES dims (`_narrow`, 8, for devices narrower than 32).
#if !defined(PIE_MIN_LANES)
#define PIE_MIN_LANES 32
#endif
// Dim pairs a lane holds: pair `lane + j * width`, dims `2p` and `2p + 1`,
// so one word load serves two dims.
#define PPL ((PIE_HEAD_DIM / 2 + PIE_MIN_LANES - 1) / PIE_MIN_LANES)

groupshared float sdpa_acc[MAX_WALKERS][PIE_HEAD_DIM];
groupshared float sdpa_max[MAX_WALKERS];
groupshared float sdpa_sum[MAX_WALKERS];
#if !defined(PIE_SUBGROUP)
groupshared float sdpa_fold[MAX_WALKERS][32];
#endif
groupshared int sdpa_pos;

// The walkers' share of the row's keys: each walker (a subgroup, or a
// 32-lane strip) takes every `walkers`-th step from its own.
void decode_walk(uint row, uint q_head, uint tid, int k_lo, int k_hi, int q_pos, int start,
                 out float merged_max, out float merged_sum, out uint walkers) {
    int req = req_of_token[row];
    uint base_row = row_base(row, q_head);
    int kv_head = int(q_head) / params.gqa_factor;

#if defined(PIE_SUBGROUP)
    uint width = WaveGetLaneCount();
    uint walker = tid / width;
    uint lane = WaveGetLaneIndex();
    walkers = min((uint(PIE_DECODE_THREADS) + width - 1u) / width, uint(MAX_WALKERS));
#else
    uint width = 32u;
    uint walker = tid / 32u;
    uint lane = tid % 32u;
    walkers = uint(MAX_WALKERS);
#endif
    bool walks = walker < walkers;
    // Pairs a lane covers, rounded up: a subgroup wider than the head's
    // pairs leaves its upper lanes idle.
    uint pairs = uint(PIE_HEAD_DIM) / 2u;
    uint per_lane = (pairs + width - 1u) / width;

    float2 qreg[PPL];
    float2 acc[PPL];
    [ForceUnroll] for (uint j = 0u; j < uint(PPL); j++) {
        uint pp = lane + j * width;
        qreg[j] = (j < per_lane && pp < pairs) ? params.scale * word_pair(queries[(base_row >> 1) + pp]) : float2(0.0, 0.0);
        acc[j] = float2(0.0, 0.0);
    }

    float max_score = NEG_INF;
    float sum_exp = 0.0;
#if defined(PIE_SELECTED)
    int nblocks = (q_pos + 1) / params.ratio;
    int sel_end = min(params.top_k, nblocks) * params.ratio;
    int steps = sel_end + (params.ratio - 1);
#else
    int steps = k_hi - k_lo + 1;
#endif
#if defined(PIE_SUBGROUP)
    // A subgroup reduces within itself: each walks its own steps.
    for (int n = int(walker); walks && n < steps; n += int(walkers)) {
#else
    // A strip reduces through workgroup memory, so every lane takes every
    // round, a strip past the keys with nothing to add.
    int rounds = (max(steps, 0) + int(walkers) - 1) / int(walkers);
    for (int round = 0; round < rounds; round++) {
        int n = round * int(walkers) + int(walker);
#endif
#if defined(PIE_SELECTED)
        int kp = -1;
        if (n < sel_end) {
            int c = selection[row * uint(params.top_k) + uint(n / params.ratio)];
            if (c >= 0 && c < nblocks) {
                kp = c * params.ratio + (n % params.ratio);
            }
        } else {
            kp = nblocks * params.ratio + (n - sel_end);
        }
        bool take = n < steps && kp >= 0 && keeps(row, kp, q_pos, start);
#else
        int kp = k_lo + n;
        bool take = n < steps && keeps(row, kp, q_pos, start);
#endif
        uint base = 0u;
        float partial = 0.0;
        if (take) {
            base = (page_slot(req, kp) * uint(params.n_kv_heads) + uint(kv_head)) * uint(PIE_HEAD_DIM);
            [ForceUnroll] for (uint j = 0u; j < uint(PPL); j++) {
                uint pp = lane + j * width;
                if (j < per_lane && pp < pairs) {
                    partial += dot(qreg[j], word_pair(k_pages[(base >> 1) + pp]));
                }
            }
        }
#if defined(PIE_SUBGROUP)
        float score = WaveActiveSum(partial);
#else
        sdpa_fold[walker][lane] = partial;
        GroupMemoryBarrierWithGroupSync();
        for (uint half_ = 16u; half_ > 0u; half_ >>= 1) {
            if (lane < half_) {
                sdpa_fold[walker][lane] += sdpa_fold[walker][lane + half_];
            }
            GroupMemoryBarrierWithGroupSync();
        }
        float score = sdpa_fold[walker][0];
        GroupMemoryBarrierWithGroupSync();
#endif
        if (take) {
            float hist, w;
            online_update(score, max_score, sum_exp, hist, w);
            [ForceUnroll] for (uint j = 0u; j < uint(PPL); j++) {
                uint pp = lane + j * width;
                if (j < per_lane && pp < pairs) {
                    acc[j] = acc[j] * hist + w * word_pair(v_pages[(base >> 1) + pp]);
                }
            }
        }
    }
    if (walks) {
        [ForceUnroll] for (uint j = 0u; j < uint(PPL); j++) {
            uint pp = lane + j * width;
            if (j < per_lane && pp < pairs) {
                sdpa_acc[walker][2u * pp] = acc[j].x;
                sdpa_acc[walker][2u * pp + 1u] = acc[j].y;
            }
        }
        if (lane == 0u) {
            sdpa_max[walker] = max_score;
            sdpa_sum[walker] = sum_exp;
        }
    }
    GroupMemoryBarrierWithGroupSync();

    merged_max = NEG_INF;
    for (uint s = 0u; s < walkers; s++) {
        merged_max = max(merged_max, sdpa_max[s]);
    }
    merged_sum = 0.0;
    for (uint s = 0u; s < walkers; s++) {
        merged_sum += sdpa_sum[s] * exp(sdpa_max[s] - merged_max);
    }
}

float decode_fold(uint d, float merged_max, uint walkers) {
    float out_acc = 0.0;
    for (uint s = 0u; s < walkers; s++) {
        out_acc += sdpa_acc[s][d] * exp(sdpa_max[s] - merged_max);
    }
    return out_acc;
}

// Dims `d`, `d + 1` of the row, folded and normed.
float2 folded_pair(uint d, float merged_max, float merged_sum, uint walkers) {
    float2 v = float2(decode_fold(d, merged_max, walkers), decode_fold(d + 1u, merged_max, walkers));
    return merged_sum == 0.0 ? v : v / merged_sum;
}

[shader("compute")]
[numthreads(PIE_DECODE_THREADS, 1, 1)]
void main(uint3 group : SV_GroupID, uint3 local : SV_GroupThreadID) {
    uint q_head = group.x;
    uint row = group.y;
    uint tid = local.x;
    // Every lane reads the same row; staged and read uniformly, as the key
    // bound below decides loops of barriers and an early return.
    if (tid == 0u) {
        sdpa_pos = position_ids[row];
    }
    GroupMemoryBarrierWithGroupSync();
    int q_pos = workgroupUniformLoad(sdpa_pos);
    int start = window_start(q_pos);
    uint base_row = row_base(row, q_head);
    float merged_max, merged_sum;
    uint walkers;
#if defined(PIE_SPLIT)
    uint split = group.z;
    int keys = q_pos - start + 1;
    int usable = clamp((keys + params.min_keys - 1) / max(params.min_keys, 1), 1, params.splits);
    uint slice = (split * uint(params.n_rows) + row) * uint(params.n_q_heads) + q_head;
    // Uniform across the workgroup: every lane of it reads the same row.
    if (int(split) >= usable) {
        if (tid == 0u) {
            part_lse[slice] = asfloat(0xff800000u);
        }
        return;
    }
    int chunk = (keys + usable - 1) / usable;
    int k_lo = start + int(split) * chunk;
    int k_hi = min(q_pos, k_lo + chunk - 1);
    decode_walk(row, q_head, tid, k_lo, k_hi, q_pos, start, merged_max, merged_sum, walkers);
    if (usable == 1) {
        // One split covered the row: land it whole, and mark it done.
        if (tid == 0u) {
            part_lse[slice] = asfloat(0x7f800000u);
#if defined(PIE_SPLIT_LSE)
            lse_out[row * uint(params.n_q_heads) + q_head] = lse_base2(merged_max, merged_sum);
#endif
        }
        for (uint d = 2u * tid; d < uint(PIE_HEAD_DIM); d += 2u * uint(PIE_DECODE_THREADS)) {
            float2 v = folded_pair(d, merged_max, merged_sum, walkers);
            o[(base_row + d) >> 1] = pack_bf16(v.x, v.y);
        }
        return;
    }
    if (tid == 0u) {
        part_lse[slice] = lse_base2(merged_max, merged_sum);
    }
    for (uint d = tid; d < uint(PIE_HEAD_DIM); d += uint(PIE_DECODE_THREADS)) {
        float v = decode_fold(d, merged_max, walkers);
        part_out[slice * uint(PIE_HEAD_DIM) + d] = merged_sum == 0.0 ? 0.0 : v / merged_sum;
    }
#else
    decode_walk(row, q_head, tid, start, q_pos, q_pos, start, merged_max, merged_sum, walkers);
#if defined(PIE_LSE)
    if (tid == 0u) {
        lse_out[row * uint(params.n_q_heads) + q_head] = lse_base2(merged_max, merged_sum);
    }
#endif
    for (uint d = 2u * tid; d < uint(PIE_HEAD_DIM); d += 2u * uint(PIE_DECODE_THREADS)) {
        float2 v = folded_pair(d, merged_max, merged_sum, walkers);
        o[(base_row + d) >> 1] = pack_bf16(v.x, v.y);
    }
#endif
}

#endif
#endif

// pie:instantiate attn_sdpa_decode_bf16_d_64 PIE_HEAD_DIM=64 PIE_TIER=subgroup PIE_TARGET=spirv
// pie:instantiate attn_sdpa_decode_t128_bf16_d_64 PIE_HEAD_DIM=64 PIE_TIER=subgroup PIE_TARGET=spirv PIE_DECODE_THREADS=128
// pie:instantiate attn_sdpa_decode_lanes_8_bf16_d_64 PIE_HEAD_DIM=64 PIE_TIER=subgroup PIE_TARGET=spirv PIE_MIN_LANES=8
// pie:instantiate attn_sdpa_decode_t128_lanes_8_bf16_d_64 PIE_HEAD_DIM=64 PIE_TIER=subgroup PIE_TARGET=spirv PIE_DECODE_THREADS=128 PIE_MIN_LANES=8
// pie:instantiate attn_sdpa_decode_bf16_d_64 PIE_HEAD_DIM=64
// pie:instantiate attn_sdpa_decode_t128_bf16_d_64 PIE_HEAD_DIM=64 PIE_DECODE_THREADS=128
// pie:instantiate attn_sdpa_decode_lse_bf16_d_64 PIE_HEAD_DIM=64 PIE_LSE=1 PIE_TIER=subgroup PIE_TARGET=spirv
// pie:instantiate attn_sdpa_decode_lse_t128_bf16_d_64 PIE_HEAD_DIM=64 PIE_LSE=1 PIE_TIER=subgroup PIE_TARGET=spirv PIE_DECODE_THREADS=128
// pie:instantiate attn_sdpa_decode_lse_lanes_8_bf16_d_64 PIE_HEAD_DIM=64 PIE_LSE=1 PIE_TIER=subgroup PIE_TARGET=spirv PIE_MIN_LANES=8
// pie:instantiate attn_sdpa_decode_lse_t128_lanes_8_bf16_d_64 PIE_HEAD_DIM=64 PIE_LSE=1 PIE_TIER=subgroup PIE_TARGET=spirv PIE_DECODE_THREADS=128 PIE_MIN_LANES=8
// pie:instantiate attn_sdpa_decode_lse_bf16_d_64 PIE_HEAD_DIM=64 PIE_LSE=1
// pie:instantiate attn_sdpa_decode_lse_t128_bf16_d_64 PIE_HEAD_DIM=64 PIE_LSE=1 PIE_DECODE_THREADS=128
// pie:instantiate attn_sdpa_tiled_bf16_d_64 PIE_HEAD_DIM=64 PIE_TILED=1 PIE_TILE_ROWS=32
// pie:instantiate attn_sdpa_tiled_r16_bf16_d_64 PIE_HEAD_DIM=64 PIE_TILED=1 PIE_TILE_ROWS=16
// pie:instantiate attn_sdpa_tiled_r8_bf16_d_64 PIE_HEAD_DIM=64 PIE_TILED=1 PIE_TILE_ROWS=8
// pie:instantiate attn_sdpa_tiled_lse_bf16_d_64 PIE_HEAD_DIM=64 PIE_TILED=1 PIE_LSE=1 PIE_TILE_ROWS=32
// pie:instantiate attn_sdpa_tiled_lse_r16_bf16_d_64 PIE_HEAD_DIM=64 PIE_TILED=1 PIE_LSE=1 PIE_TILE_ROWS=16
// pie:instantiate attn_sdpa_tiled_lse_r8_bf16_d_64 PIE_HEAD_DIM=64 PIE_TILED=1 PIE_LSE=1 PIE_TILE_ROWS=8
// pie:instantiate attn_sdpa_decode_bf16_d_128 PIE_HEAD_DIM=128 PIE_TIER=subgroup PIE_TARGET=spirv
// pie:instantiate attn_sdpa_decode_t128_bf16_d_128 PIE_HEAD_DIM=128 PIE_TIER=subgroup PIE_TARGET=spirv PIE_DECODE_THREADS=128
// pie:instantiate attn_sdpa_decode_lanes_8_bf16_d_128 PIE_HEAD_DIM=128 PIE_TIER=subgroup PIE_TARGET=spirv PIE_MIN_LANES=8
// pie:instantiate attn_sdpa_decode_t128_lanes_8_bf16_d_128 PIE_HEAD_DIM=128 PIE_TIER=subgroup PIE_TARGET=spirv PIE_DECODE_THREADS=128 PIE_MIN_LANES=8
// pie:instantiate attn_sdpa_decode_bf16_d_128 PIE_HEAD_DIM=128
// pie:instantiate attn_sdpa_decode_t128_bf16_d_128 PIE_HEAD_DIM=128 PIE_DECODE_THREADS=128
// pie:instantiate attn_sdpa_decode_lse_bf16_d_128 PIE_HEAD_DIM=128 PIE_LSE=1 PIE_TIER=subgroup PIE_TARGET=spirv
// pie:instantiate attn_sdpa_decode_lse_t128_bf16_d_128 PIE_HEAD_DIM=128 PIE_LSE=1 PIE_TIER=subgroup PIE_TARGET=spirv PIE_DECODE_THREADS=128
// pie:instantiate attn_sdpa_decode_lse_lanes_8_bf16_d_128 PIE_HEAD_DIM=128 PIE_LSE=1 PIE_TIER=subgroup PIE_TARGET=spirv PIE_MIN_LANES=8
// pie:instantiate attn_sdpa_decode_lse_t128_lanes_8_bf16_d_128 PIE_HEAD_DIM=128 PIE_LSE=1 PIE_TIER=subgroup PIE_TARGET=spirv PIE_DECODE_THREADS=128 PIE_MIN_LANES=8
// pie:instantiate attn_sdpa_decode_lse_bf16_d_128 PIE_HEAD_DIM=128 PIE_LSE=1
// pie:instantiate attn_sdpa_decode_lse_t128_bf16_d_128 PIE_HEAD_DIM=128 PIE_LSE=1 PIE_DECODE_THREADS=128
// pie:instantiate attn_sdpa_tiled_bf16_d_128 PIE_HEAD_DIM=128 PIE_TILED=1 PIE_TILE_ROWS=32
// pie:instantiate attn_sdpa_tiled_r16_bf16_d_128 PIE_HEAD_DIM=128 PIE_TILED=1 PIE_TILE_ROWS=16
// pie:instantiate attn_sdpa_tiled_r8_bf16_d_128 PIE_HEAD_DIM=128 PIE_TILED=1 PIE_TILE_ROWS=8
// pie:instantiate attn_sdpa_tiled_lse_bf16_d_128 PIE_HEAD_DIM=128 PIE_TILED=1 PIE_LSE=1 PIE_TILE_ROWS=32
// pie:instantiate attn_sdpa_tiled_lse_r16_bf16_d_128 PIE_HEAD_DIM=128 PIE_TILED=1 PIE_LSE=1 PIE_TILE_ROWS=16
// pie:instantiate attn_sdpa_tiled_lse_r8_bf16_d_128 PIE_HEAD_DIM=128 PIE_TILED=1 PIE_LSE=1 PIE_TILE_ROWS=8
// pie:instantiate attn_sdpa_decode_bf16_d_256 PIE_HEAD_DIM=256 PIE_TIER=subgroup PIE_TARGET=spirv
// pie:instantiate attn_sdpa_decode_t128_bf16_d_256 PIE_HEAD_DIM=256 PIE_TIER=subgroup PIE_TARGET=spirv PIE_DECODE_THREADS=128
// pie:instantiate attn_sdpa_decode_lanes_8_bf16_d_256 PIE_HEAD_DIM=256 PIE_TIER=subgroup PIE_TARGET=spirv PIE_MIN_LANES=8
// pie:instantiate attn_sdpa_decode_t128_lanes_8_bf16_d_256 PIE_HEAD_DIM=256 PIE_TIER=subgroup PIE_TARGET=spirv PIE_DECODE_THREADS=128 PIE_MIN_LANES=8
// pie:instantiate attn_sdpa_decode_bf16_d_256 PIE_HEAD_DIM=256
// pie:instantiate attn_sdpa_decode_t128_bf16_d_256 PIE_HEAD_DIM=256 PIE_DECODE_THREADS=128
// pie:instantiate attn_sdpa_decode_lse_bf16_d_256 PIE_HEAD_DIM=256 PIE_LSE=1 PIE_TIER=subgroup PIE_TARGET=spirv
// pie:instantiate attn_sdpa_decode_lse_t128_bf16_d_256 PIE_HEAD_DIM=256 PIE_LSE=1 PIE_TIER=subgroup PIE_TARGET=spirv PIE_DECODE_THREADS=128
// pie:instantiate attn_sdpa_decode_lse_lanes_8_bf16_d_256 PIE_HEAD_DIM=256 PIE_LSE=1 PIE_TIER=subgroup PIE_TARGET=spirv PIE_MIN_LANES=8
// pie:instantiate attn_sdpa_decode_lse_t128_lanes_8_bf16_d_256 PIE_HEAD_DIM=256 PIE_LSE=1 PIE_TIER=subgroup PIE_TARGET=spirv PIE_DECODE_THREADS=128 PIE_MIN_LANES=8
// pie:instantiate attn_sdpa_decode_lse_bf16_d_256 PIE_HEAD_DIM=256 PIE_LSE=1
// pie:instantiate attn_sdpa_decode_lse_t128_bf16_d_256 PIE_HEAD_DIM=256 PIE_LSE=1 PIE_DECODE_THREADS=128
// pie:instantiate attn_sdpa_tiled_bf16_d_256 PIE_HEAD_DIM=256 PIE_TILED=1 PIE_TILE_ROWS=32
// pie:instantiate attn_sdpa_tiled_r16_bf16_d_256 PIE_HEAD_DIM=256 PIE_TILED=1 PIE_TILE_ROWS=16
// pie:instantiate attn_sdpa_tiled_r8_bf16_d_256 PIE_HEAD_DIM=256 PIE_TILED=1 PIE_TILE_ROWS=8
// pie:instantiate attn_sdpa_tiled_lse_bf16_d_256 PIE_HEAD_DIM=256 PIE_TILED=1 PIE_LSE=1 PIE_TILE_ROWS=32
// pie:instantiate attn_sdpa_tiled_lse_r16_bf16_d_256 PIE_HEAD_DIM=256 PIE_TILED=1 PIE_LSE=1 PIE_TILE_ROWS=16
// pie:instantiate attn_sdpa_tiled_lse_r8_bf16_d_256 PIE_HEAD_DIM=256 PIE_TILED=1 PIE_LSE=1 PIE_TILE_ROWS=8
// pie:instantiate attn_sdpa_decode_bf16_d_512 PIE_HEAD_DIM=512 PIE_TIER=subgroup PIE_TARGET=spirv
// pie:instantiate attn_sdpa_decode_t128_bf16_d_512 PIE_HEAD_DIM=512 PIE_TIER=subgroup PIE_TARGET=spirv PIE_DECODE_THREADS=128
// pie:instantiate attn_sdpa_decode_lanes_8_bf16_d_512 PIE_HEAD_DIM=512 PIE_TIER=subgroup PIE_TARGET=spirv PIE_MIN_LANES=8
// pie:instantiate attn_sdpa_decode_t128_lanes_8_bf16_d_512 PIE_HEAD_DIM=512 PIE_TIER=subgroup PIE_TARGET=spirv PIE_DECODE_THREADS=128 PIE_MIN_LANES=8
// pie:instantiate attn_sdpa_decode_bf16_d_512 PIE_HEAD_DIM=512
// pie:instantiate attn_sdpa_decode_t128_bf16_d_512 PIE_HEAD_DIM=512 PIE_DECODE_THREADS=128
// pie:instantiate attn_sdpa_decode_lse_bf16_d_512 PIE_HEAD_DIM=512 PIE_LSE=1 PIE_TIER=subgroup PIE_TARGET=spirv
// pie:instantiate attn_sdpa_decode_lse_t128_bf16_d_512 PIE_HEAD_DIM=512 PIE_LSE=1 PIE_TIER=subgroup PIE_TARGET=spirv PIE_DECODE_THREADS=128
// pie:instantiate attn_sdpa_decode_lse_lanes_8_bf16_d_512 PIE_HEAD_DIM=512 PIE_LSE=1 PIE_TIER=subgroup PIE_TARGET=spirv PIE_MIN_LANES=8
// pie:instantiate attn_sdpa_decode_lse_t128_lanes_8_bf16_d_512 PIE_HEAD_DIM=512 PIE_LSE=1 PIE_TIER=subgroup PIE_TARGET=spirv PIE_DECODE_THREADS=128 PIE_MIN_LANES=8
// pie:instantiate attn_sdpa_decode_lse_bf16_d_512 PIE_HEAD_DIM=512 PIE_LSE=1
// pie:instantiate attn_sdpa_decode_lse_t128_bf16_d_512 PIE_HEAD_DIM=512 PIE_LSE=1 PIE_DECODE_THREADS=128
// pie:instantiate attn_sdpa_tiled_bf16_d_512 PIE_HEAD_DIM=512 PIE_TILED=1 PIE_TILE_ROWS=32
// pie:instantiate attn_sdpa_tiled_r16_bf16_d_512 PIE_HEAD_DIM=512 PIE_TILED=1 PIE_TILE_ROWS=16
// pie:instantiate attn_sdpa_tiled_r8_bf16_d_512 PIE_HEAD_DIM=512 PIE_TILED=1 PIE_TILE_ROWS=8
// pie:instantiate attn_sdpa_tiled_lse_bf16_d_512 PIE_HEAD_DIM=512 PIE_TILED=1 PIE_LSE=1 PIE_TILE_ROWS=32
// pie:instantiate attn_sdpa_tiled_lse_r16_bf16_d_512 PIE_HEAD_DIM=512 PIE_TILED=1 PIE_LSE=1 PIE_TILE_ROWS=16
// pie:instantiate attn_sdpa_tiled_lse_r8_bf16_d_512 PIE_HEAD_DIM=512 PIE_TILED=1 PIE_LSE=1 PIE_TILE_ROWS=8
