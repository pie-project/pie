#include "prim/bf16.slang"
#include "prim/params.slang"
#include "prim/shuffle.slang"
#include "prim/ieee.slang"
#include "prim/pool.slang"

// Paged attention over a KV cache, bf16 two a word, causal (an optional
// sliding window, each row's byte mask from its lane's committed boundary on,
// per-row data: a row stating none reads every key in the same kernel). Decode reads one query row a workgroup (per head): the
// workgroup's subgroups (the subgroup tier, SPIR-V only) or 32-lane strips
// (the baseline, reduced through workgroup memory) each walk a share of the
// keys with an online softmax and fold at the end. The subgroup tier reads a
// bf16 value a lane and the mask a byte at a time as 16- and 8-bit storage
// (the same bytes as the packed words), one load a dim: engine-vulkan's
// decode, kept as it was measured.
// Variants: the row's base-2 LSE (`PIE_LSE`); the keys a block selection
// names (`PIE_SELECTED`); one of `splits` key ranges a workgroup, landing f32
// partials and their LSE for the fold (`PIE_SPLIT`, `PIE_SPLIT_LSE`).
// Prefill (`PIE_TILED`) holds `PIE_TILE_ROWS` query rows of one head a
// workgroup, `PIE_TILE_LANES` lanes a row (32, reduced as a tree; or 16,
// each lane summing its row's partials after one barrier, as
// kernels-wgpu's tile did; at the subgroup tier the 16 lanes fold by
// shuffles, no barrier, on subgroups of at least 16 whose lanes are a row's
// in order), every row walking the tile's keys in step.
// `PIE_FOLD` folds the split partials by their LSEs.

static const float NEG_INF = -3.0e38;

float2 word_pair(uint w) {
    return float2(bf16_to_f32(w & 0xffffu), bf16_to_f32(w >> 16));
}

uint byte_at(StructuredBuffer<uint> buf, uint i) {
    return (buf[i >> 2] >> ((i & 3u) * 8u)) & 0xffu;
}

float lse_base2(float max_score, float sum_exp) {
    return sum_exp > 0.0 ? max_score * 1.44269504088896340736 + log2(sum_exp) : neg_inf();
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
    // The fire row the window's row 0 is: `o` and the LSE are the fire's
    // whole planes, the partials the window's own.
    uint row0;
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
    uint at = params.row0 + row;
    o[(at * width + e) >> 1] = pack_bf16(v.x, v.y);
#if defined(PIE_LSE)
    if (d == 0u) {
        lse_out[at * heads + head] = finite_(top) ? top + log2(total) : neg_inf();
    }
#endif
}

#else

#ifndef PIE_HEAD_DIM
#error "attn/sdpa.slang is compiled at a head width: -DPIE_HEAD_DIM"
#endif

// The subgroup decode reads bf16 and mask bytes through 16- and 8-bit
// storage; the tile (and WGSL, which has neither) reads packed words.
#if defined(PIE_SUBGROUP) && !defined(PIE_TILED)
#define SDPA_NARROW_STORAGE 1
#endif

#if defined(SDPA_NARROW_STORAGE)
#define SDPA_KV uint16_t
#define SDPA_KV_SCALE 2u
[[vk::binding(0)]] StructuredBuffer<uint16_t> queries;
[[vk::binding(1)]] RWStructuredBuffer<uint16_t> o;
#else
#define SDPA_KV uint
#define SDPA_KV_SCALE 1u
[[vk::binding(0)]] StructuredBuffer<uint> queries;
[[vk::binding(1)]] RWStructuredBuffer<uint> o;
#endif
[[vk::binding(2)]] StructuredBuffer<int> position_ids;
[[vk::binding(3)]] StructuredBuffer<int> req_of_token;
[[vk::binding(4)]] StructuredBuffer<uint> kv_page_indices;
[[vk::binding(5)]] StructuredBuffer<uint> kv_page_indptr;
// A row's mask entry, two words at `2 row`: the key its plane's column 0 is
// (its lane's committed boundary; `MASK_NONE`, past every key, states no
// mask), then the byte its columns start at in the plane, read only by a key
// past the boundary.
static const uint MASK_NONE = 0x7fffffffu;
[[vk::binding(7)]] StructuredBuffer<uint> attention_mask_words;
#if defined(SDPA_NARROW_STORAGE)
[[vk::binding(6)]] StructuredBuffer<uint8_t> attention_mask;

uint mask_byte(uint i) {
    return uint(attention_mask[i]);
}
#else
[[vk::binding(6)]] StructuredBuffer<uint> attention_mask;

uint mask_byte(uint i) {
    return byte_at(attention_mask, i);
}
#endif
#if defined(PIE_SPLIT)
[[vk::binding(8)]] RWStructuredBuffer<float> part_out;
[[vk::binding(9)]] RWStructuredBuffer<float> part_lse;
#if defined(PIE_SPLIT_LSE)
[[vk::binding(10)]] RWStructuredBuffer<float> lse_out;
#define SDPA_SEGMENTS_AT 11
#else
#define SDPA_SEGMENTS_AT 10
#endif
#elif defined(PIE_LSE)
[[vk::binding(8)]] RWStructuredBuffer<float> lse_out;
#define SDPA_SEGMENTS_AT 9
#elif defined(PIE_SELECTED)
[[vk::binding(8)]] StructuredBuffer<int> selection;
#define SDPA_SEGMENTS_AT 9
#else
#define SDPA_SEGMENTS_AT 8
#endif
#if defined(PIE_RING)
#if defined(PIE_SELECTED)
#error "a ring is a window's, and a selection is no window"
#endif
// A windowed layer as state + buffer (CONTRACT-BC 19, 21): a row's keys
// below its lane's folded index lie in the ring in the lane's record, row
// `kp` at ring place `kp % ring_rows`; keys from there to the lane's first
// row of the fire in its window-buffer slots, entry `i` rows `[base + i T,
// base + (i + 1) T)`; the fire's own rows in the fire's slots, laid as
// buffer slots, the lane's first row of the fire at row 0 of its first.
// `ring_lanes` holds, a lane, its folded index, its buffer base (a slot's
// first row), its first row of the fire, its folded index after the fire
// (the fold's), the buffered rows it does not read and the first slot of
// its one run of buffered rows (its lineage, `ring_start`), then its
// record's units, `pieces` of them. The page
// table holds, a lane, its buffer slots of rows below its first row of the
// fire, then its fire slots.
[[vk::binding(SDPA_SEGMENTS_AT)]] StructuredBuffer<int> ring_lanes;
#define SDPA_POOL_AT (SDPA_SEGMENTS_AT + 1)
#else
#define SDPA_POOL_AT SDPA_SEGMENTS_AT
#endif
// The kv pages, in the chunk pool's segments (`prim/pool.slang`).
PIE_POOL_SEGMENTS(StructuredBuffer, SDPA_KV, SDPA_POOL_AT)
#define PIE_PARAMS_AT (SDPA_POOL_AT + 4)

struct Params {
    uint seg_words;
    uint k_origin;
    uint v_origin;
    uint page_stride;
    int gqa_factor;
    int page_size;
    int n_kv_heads;
    float scale;
    int window;
    int n_q_heads;
    // The fire row the window's row 0 is: the queries, the output, the LSE,
    // the tables and the selection are the fire's whole, bound once at load
    // whatever window a fire cuts; a dispatch's `row` is the window's.
    uint row0;
#if defined(PIE_SELECTED)
    int top_k;
    int ratio;
#endif
#if defined(PIE_TILED)
    int n_rows;
    int first_tile;
#endif
#if defined(PIE_SPLIT)
    int splits;
    int n_rows;
    int min_keys;
#endif
#if defined(PIE_RING)
    // The rows the ring holds; the record's pieces the layer's ring keys
    // and values lie in and the pieces a record takes; the ring's keys and
    // values, in u32 words past their piece's first; u32 words a record
    // unit.
    uint ring_rows;
    uint ring_piece;
    uint ring_piece_v;
    uint pieces;
    uint ring_k;
    uint ring_v;
    uint slot_stride;
#endif
};
PIE_PARAMS(PIE_PARAMS_AT, Params, params);
PIE_POOL_READ(SDPA_KV, SDPA_KV_SCALE * params.seg_words)

// The fire row a window's row `row` is.
uint fire_row(uint row) {
    return params.row0 + row;
}

// The bf16 element key `kp` of request `req` starts at in its plane: its
// page, `page_stride` words a page, then the token's row in it.
uint page_row(int req, int kp) {
    uint phys = kv_page_indices[kv_page_indptr[req] + uint(kp / params.page_size)];
    return phys * (2u * params.page_stride)
        + uint(kp % params.page_size) * uint(params.n_kv_heads) * uint(PIE_HEAD_DIM);
}

// Where a row's keys lie: its lane's buffer base and first row of the
// fire, and its ring's keys and values as u32 words of the pool.
struct Keys {
    int folded;
    int base;
    int first;
    // The buffered rows the lane does not read (its lineage runs this many
    // rows behind the set's slots), and the first slot of its one run of
    // buffered rows.
    int shift;
    int run_at;
    uint ring_k;
    uint ring_v;
    // A fire row's place in the lane's page table counted in rows: its
    // buffer's whole pages, then its own (`kp + fire_at` from `first` on).
    int fire_at;
};

Keys keys_of(int req) {
    Keys at;
#if defined(PIE_RING)
    uint lane = (6u + params.pieces) * uint(req);
    at.folded = ring_lanes[lane];
    at.base = ring_lanes[lane + 1u];
    at.first = ring_lanes[lane + 2u];
    at.shift = ring_lanes[lane + 4u];
    at.run_at = ring_lanes[lane + 5u];
    uint rec_k = uint(ring_lanes[lane + 6u + params.ring_piece]) * params.slot_stride;
    uint rec_v = uint(ring_lanes[lane + 6u + params.ring_piece_v]) * params.slot_stride;
    at.ring_k = rec_k + params.ring_k;
    at.ring_v = rec_v + params.ring_v;
    int buffered = max(at.first - at.base + params.page_size - 1, 0) / params.page_size;
    at.fire_at = buffered * params.page_size - at.first;
#else
    at.folded = 0;
    at.base = 0;
    at.first = 0x7fffffff;
    at.shift = 0;
    at.run_at = 0;
    at.ring_k = 0u;
    at.ring_v = 0u;
    at.fire_at = 0;
#endif
    return at;
}

// The page-table entry of request `req` key `kp` and its row in that page
// (`-1` where the key lies in the ring, `row` then its ring place).
int key_page(Keys at, int kp, out uint row) {
#if defined(PIE_RING)
    if (kp < at.folded) {
        row = uint(kp) % params.ring_rows;
        return -1;
    }
    // The row's place in the table: past the buffer's rows, the fire's.
    int idx = kp >= at.first ? kp + at.fire_at : kp - at.base;
    int page = idx / params.page_size;
    row = uint(idx - page * params.page_size);
    return page;
#else
    row = uint(kp % params.page_size);
    return kp / params.page_size;
#endif
}

// The u32 word of the pool key `kp` of request `req` starts at, its key
// then its value (the head's offset not added). Words, not bf16 elements:
// a pool past 8 GiB (the fire's own slots lie at its top) passes a u32 of
// elements.
void key_at(int req, Keys at, int kp, out uint k_w, out uint v_w) {
    uint row;
    int page = key_page(at, kp, row);
    uint cell = (row * uint(params.n_kv_heads) * uint(PIE_HEAD_DIM)) >> 1;
    if (page < 0) {
        k_w = at.ring_k + cell;
        v_w = at.ring_v + cell;
        return;
    }
    uint phys = kv_page_indices[kv_page_indptr[req] + uint(page)];
    uint w = phys * params.page_stride + cell;
    k_w = params.k_origin + w;
    v_w = params.v_origin + w;
}

// The segment pool word `w` lies in and its index there as the kernel's
// storage counts it (`SDPA_KV_SCALE` a word): `pie_pool_find` over words.
void ring_find(uint w, out uint seg, out uint local) {
    uint s = params.seg_words;
    seg = 0u;
    if (w >= s) {
        w -= s;
        seg = 1u;
        if (w >= s) {
            w -= s;
            seg = 2u;
            if (w >= s) {
                w -= s;
                seg = 3u;
            }
        }
    }
    local = SDPA_KV_SCALE * w;
}

// Row `row`'s boundary, read once a row: `MASK_NONE` where it states no mask.
uint mask_origin(uint row) {
    return attention_mask_words[2u * fire_row(row)];
}

// Whether row `row`'s mask admits key `kp`, `origin` its boundary. A row's
// mask speaks of the keys from its lane's committed boundary on: every key
// below it is committed and read under the model's own bounds alone, and the
// row's column `c` is key `origin + c`, which its causal bound keeps inside
// the row's columns. Only a key past the boundary reads the row's offset and
// its byte, so a row stating no mask touches no plane.
bool mask_admits(uint row, uint origin, uint kp) {
    if (kp < origin) {
        return true;
    }
    return mask_byte(attention_mask_words[2u * fire_row(row) + 1u] + (kp - origin)) != 0u;
}

bool keeps(uint row, uint origin, int kp, int q_pos, int start) {
    if (kp > q_pos || kp < start) {
        return false;
    }
    return mask_admits(row, origin, uint(kp));
}

int window_start(int q_pos) {
    return (params.window > 0 && q_pos >= params.window) ? q_pos - params.window + 1 : 0;
}

#if defined(PIE_RING)
// A ring lane's row's first key: its window counted along its lineage (the
// buffered rows it reads, then its own: `q_pos - shift`), mapped back to
// the set's slots. Below the folded index the two agree; among its buffered
// rows they lie in its one run from `run_at`; past them, among its own.
int ring_start(Keys at, int q_pos) {
    int s = window_start(q_pos - at.shift);
    if (s >= at.folded) {
        int admitted = at.first - at.folded - at.shift;
        s = s >= at.folded + admitted ? s + at.shift : at.run_at + (s - at.folded);
    }
    return s;
}
#endif

uint row_base(uint row, uint q_head) {
    return (fire_row(row) * uint(params.n_q_heads) + q_head) * uint(PIE_HEAD_DIM);
}

#if defined(PIE_TILED)

#if !defined(PIE_TILE_ROWS)
#define PIE_TILE_ROWS 32
#endif
#if !defined(PIE_TILE_LANES)
#define PIE_TILE_LANES 32
#endif
#define LANES_PER_ROW uint(PIE_TILE_LANES)
// Pairs of dims a lane holds: pair `lane + LANES_PER_ROW i`, dims `2p` and
// `2p + 1`.
#define PAIRS_PER_LANE (uint(PIE_HEAD_DIM) / (2u * LANES_PER_ROW))

#if defined(PIE_SUBGROUP) && PIE_TILE_LANES != 16
#error "the subgroup tile folds 16 lanes a row"
#endif
#if !defined(PIE_SUBGROUP)
groupshared float sdpa_part[PIE_TILE_ROWS][PIE_TILE_LANES];
#endif
groupshared int sdpa_last;
groupshared int sdpa_first;

[shader("compute")]
[numthreads(PIE_TILE_LANES, PIE_TILE_ROWS, 1)]
void main(uint3 group : SV_GroupID, uint3 local : SV_GroupThreadID) {
    uint q_head = group.x;
    uint lane = local.x;
    uint slot_y = local.y;
    uint tile = group.y + uint(params.first_tile);
    uint row = tile * uint(PIE_TILE_ROWS) + slot_y;
    bool live = row < uint(params.n_rows);

    int req = live ? req_of_token[fire_row(row)] : 0;
#if defined(PIE_RING)
    Keys ring = keys_of(req);
#endif
    int q_pos = live ? position_ids[fire_row(row)] : -1;
    uint origin = live ? mask_origin(row) : MASK_NONE;
#if defined(PIE_RING)
    int start = live ? ring_start(ring, q_pos) : 0;
#else
    int start = window_start(q_pos);
#endif
    uint q_base = row_base(row, q_head);
    int kv_head = int(q_head) / params.gqa_factor;

    // The tile's first and last keys, read once and uniformly: they bound
    // the key loop, whose every step holds barriers. No row keeps a key
    // below its window's first, so the loop starts at the earliest row's.
    if (lane == 0u && slot_y == 0u) {
        int last = -1;
        int first = 0x7fffffff;
        for (uint r = 0u; r < uint(PIE_TILE_ROWS); r++) {
            uint at = tile * uint(PIE_TILE_ROWS) + r;
            if (at < uint(params.n_rows)) {
                int pos = position_ids[fire_row(at)];
                last = max(last, pos);
#if defined(PIE_RING)
                first = min(first, ring_start(keys_of(req_of_token[fire_row(at)]), pos));
#else
                first = min(first, window_start(pos));
#endif
            }
        }
        sdpa_last = last;
        sdpa_first = max(min(first, last + 1), 0);
    }
    GroupMemoryBarrierWithGroupSync();
    int last = workgroupUniformLoad(sdpa_last);
    int first_key = workgroupUniformLoad(sdpa_first);

    float2 qv[PAIRS_PER_LANE];
    float2 acc[PAIRS_PER_LANE];
    for (uint i = 0u; i < PAIRS_PER_LANE; i++) {
        uint d = 2u * (lane + i * LANES_PER_ROW);
        qv[i] = live ? word_pair(queries[(q_base + d) >> 1]) * params.scale : float2(0.0, 0.0);
        acc[i] = float2(0.0, 0.0);
    }
    float max_score = NEG_INF;
    float sum_exp = 0.0;

    // The page the keys walk is in, its segment and where its words start
    // there (a page lies in one segment): found once a page, not a key.
    int page_at = -1;
    uint k_seg = 0u, k_page = 0u, v_seg = 0u, v_page = 0u;
#if defined(PIE_RING) && !defined(PIE_ONE_SEGMENT)
    // The lane's ring lies in its record's one chunk, so in one segment:
    // found once, its rows' words then past its first.
    uint ring_k_seg = 0u, ring_k_at = 0u, ring_v_seg = 0u, ring_v_at = 0u;
    pie_pool_at(SDPA_KV_SCALE * ring.ring_k, ring_k_seg, ring_k_at);
    pie_pool_at(SDPA_KV_SCALE * ring.ring_v, ring_v_seg, ring_v_at);
#endif
    for (int kp = first_key; kp <= last; kp++) {
        bool take = live && keeps(row, origin, kp, q_pos, start);
        float partial = 0.0;
        uint k_at = 0u, v_at = 0u;
        uint vw[PAIRS_PER_LANE];
        if (take) {
#if defined(PIE_RING)
            uint in_page;
            int page = key_page(ring, kp, in_page);
            if (page < 0) {
                // A ring row: its keys and values lie in the record's one
                // chunk, so in one segment; the page cache is left for the
                // next table entry.
                uint cell_words = (in_page * uint(params.n_kv_heads) * uint(PIE_HEAD_DIM)) >> 1;
#if defined(PIE_ONE_SEGMENT)
                k_page = SDPA_KV_SCALE * (ring.ring_k + cell_words);
                v_page = SDPA_KV_SCALE * (ring.ring_v + cell_words);
#else
                k_seg = ring_k_seg;
                v_seg = ring_v_seg;
                k_page = ring_k_at + SDPA_KV_SCALE * cell_words;
                v_page = ring_v_at + SDPA_KV_SCALE * cell_words;
#endif
                page_at = -1;
                in_page = 0u;
            } else if (page != page_at) {
#else
            int page = kp / params.page_size;
            if (page != page_at) {
#endif
                uint phys = kv_page_indices[kv_page_indptr[req] + uint(page)];
#if defined(PIE_ONE_SEGMENT)
                k_page = SDPA_KV_SCALE * params.k_origin + phys * params.page_stride;
                v_page = SDPA_KV_SCALE * params.v_origin + phys * params.page_stride;
#else
                pie_pool_at(SDPA_KV_SCALE * params.k_origin + phys * params.page_stride, k_seg, k_page);
                pie_pool_at(SDPA_KV_SCALE * params.v_origin + phys * params.page_stride, v_seg, v_page);
#endif
                page_at = page;
            }
            // The token's row in the page, then the head's: `page_row`'s
            // words past the page's first.
#if defined(PIE_RING)
            uint row_words = (in_page * uint(params.n_kv_heads) * uint(PIE_HEAD_DIM)
                + uint(kv_head) * uint(PIE_HEAD_DIM)) >> 1;
#else
            uint row_words = (uint(kp % params.page_size) * uint(params.n_kv_heads) * uint(PIE_HEAD_DIM)
                + uint(kv_head) * uint(PIE_HEAD_DIM)) >> 1;
#endif
            k_at = k_page + row_words;
            v_at = v_page + row_words;
            // The row's key and value words first, from their one segment (a
            // slot holds a page's keys and values); then the sum as the pool
            // view read it.
            uint kw[PAIRS_PER_LANE];
#define SDPA_TILE_K(buf_) \
            [ForceUnroll] for (uint i = 0u; i < PAIRS_PER_LANE; i++) { \
                kw[i] = buf_[k_at + lane + i * LANES_PER_ROW]; \
            }
#define SDPA_TILE_V(buf_) \
            [ForceUnroll] for (uint i = 0u; i < PAIRS_PER_LANE; i++) { \
                vw[i] = buf_[v_at + lane + i * LANES_PER_ROW]; \
            }
#define SDPA_TILE_KV(buf_) SDPA_TILE_K(buf_) SDPA_TILE_V(buf_)
#if defined(PIE_ONE_SEGMENT)
            // A pool of one segment (`_seg1`, which the host picks for one, SPIR-V)
            // reads it with no copy of the loads a segment.
            SDPA_TILE_KV(pie_seg0)
#else
            if (k_seg == v_seg) {
                PIE_POOL_ON(k_seg, SDPA_TILE_KV)
            } else {
                PIE_POOL_ON(k_seg, SDPA_TILE_K)
                PIE_POOL_ON(v_seg, SDPA_TILE_V)
            }
#endif
            [ForceUnroll] for (uint i = 0u; i < PAIRS_PER_LANE; i++) {
                partial += dot(qv[i], word_pair(kw[i]));
            }
        }
#if defined(PIE_SUBGROUP)
        float score = partial;
        for (uint mask = LANES_PER_ROW >> 1; mask > 0u; mask >>= 1) {
            score += pie_shuffle_xor(score, mask);
        }
#else
        sdpa_part[slot_y][lane] = partial;
        GroupMemoryBarrierWithGroupSync();
#if PIE_TILE_LANES == 32
        for (uint half_ = LANES_PER_ROW >> 1; half_ > 0u; half_ >>= 1) {
            if (lane < half_) {
                sdpa_part[slot_y][lane] += sdpa_part[slot_y][lane + half_];
            }
            GroupMemoryBarrierWithGroupSync();
        }
        float score = sdpa_part[slot_y][0];
#else
        float score = 0.0;
        for (uint l = 0u; l < LANES_PER_ROW; l++) {
            score += sdpa_part[slot_y][l];
        }
#endif
#endif
        if (take) {
            float hist, w;
            online_update(score, max_score, sum_exp, hist, w);
            [ForceUnroll] for (uint i = 0u; i < PAIRS_PER_LANE; i++) {
                acc[i] = acc[i] * hist + w * word_pair(vw[i]);
            }
        }
#if !defined(PIE_SUBGROUP)
        GroupMemoryBarrierWithGroupSync();
#endif
    }

#if defined(PIE_LSE)
    if (live && lane == 0u) {
        lse_out[fire_row(row) * uint(params.n_q_heads) + q_head] = lse_base2(max_score, sum_exp);
    }
#endif
    if (live) {
        for (uint i = 0u; i < PAIRS_PER_LANE; i++) {
            float2 v = sum_exp == 0.0 ? acc[i] : acc[i] / sum_exp;
            o[(q_base + 2u * (lane + i * LANES_PER_ROW)) >> 1] = pack_bf16(v.x, v.y);
        }
    }
}

#elif defined(PIE_SUBGROUP)

// Invocations a decode workgroup walks a row's keys with; `_t128` halves
// them, and their workgroup memory, where the device's limits or the
// profile ask.
#if !defined(PIE_DECODE_THREADS)
#define PIE_DECODE_THREADS 256
#endif
#define MAX_SUBGROUPS (PIE_DECODE_THREADS / 32)
// The narrowest subgroup this variant serves: its per-lane registers hold
// HEAD_DIM / MIN_LANES dims. A device narrower than 32 lanes that cannot be
// pinned to 32 takes `_lanes_8`.
#if !defined(PIE_MIN_LANES)
#define PIE_MIN_LANES 32
#endif
#define DPL ((PIE_HEAD_DIM + PIE_MIN_LANES - 1) / PIE_MIN_LANES)

groupshared float sdpa_acc[MAX_SUBGROUPS][PIE_HEAD_DIM];
groupshared float sdpa_max[MAX_SUBGROUPS];
groupshared float sdpa_sum[MAX_SUBGROUPS];

float bf16_of(uint16_t h) {
    return bf16_to_f32(uint(h));
}

// The workgroup's subgroups' share of the row's keys: each takes every
// `subgroups`-th step from its own and reduces within itself.
void decode_walk(uint row, uint q_head, uint tid, int k_lo, int k_hi, int q_pos, int start,
                 out float merged_max, out float merged_sum, out uint subgroups) {
    int req = req_of_token[fire_row(row)];
#if defined(PIE_RING)
    Keys ring = keys_of(req);
#endif
    uint origin = mask_origin(row);
    uint base_row = row_base(row, q_head);
    int kv_head = int(q_head) / params.gqa_factor;

    uint width = WaveGetLaneCount();
    // From the invocation's index: Adreno reads the SubgroupId builtin as 0
    // in every subgroup.
    uint sg = tid / width;
    uint lane = WaveGetLaneIndex();
    subgroups = min((uint(PIE_DECODE_THREADS) + width - 1u) / width, uint(MAX_SUBGROUPS));
    bool walks = sg < subgroups;

    // Dims a lane covers, rounded up: a subgroup wider than the head leaves
    // its upper lanes idle rather than covering nothing.
    uint per_lane = (uint(PIE_HEAD_DIM) + width - 1u) / width;

    float qreg[DPL];
    float acc[DPL];
    [ForceUnroll] for (uint j = 0u; j < uint(DPL); j++) {
        uint dd = lane + j * width;
        qreg[j] = (j < per_lane && dd < uint(PIE_HEAD_DIM))
            ? params.scale * bf16_of(queries[base_row + dd]) : 0.0;
        acc[j] = 0.0;
    }

    float max_score = NEG_INF;
    float sum_exp = 0.0;
    if (walks) {
        // `keeps` carries the causal and mask tests, so the selected walk
        // only has to name cells: a block below `nblocks`, then the open one.
#if defined(PIE_SELECTED)
        int nblocks = (q_pos + 1) / params.ratio;
        int sel_end = min(params.top_k, nblocks) * params.ratio;
        int steps = sel_end + (params.ratio - 1);
#else
        int steps = k_hi - k_lo + 1;
#endif
        for (int n = int(sg); n < steps; n += int(subgroups)) {
#if defined(PIE_SELECTED)
            int kp;
            if (n < sel_end) {
                int c = selection[fire_row(row) * uint(params.top_k) + uint(n / params.ratio)];
                if (c < 0 || c >= nblocks) {
                    continue;
                }
                kp = c * params.ratio + (n % params.ratio);
            } else {
                kp = nblocks * params.ratio + (n - sel_end);
            }
#else
            int kp = k_lo + n;
#endif
            if (!keeps(row, origin, kp, q_pos, start)) {
                continue;
            }
            uint k_seg, k_at, v_seg, v_at;
#if defined(PIE_RING)
            uint k_w, v_w;
            key_at(req, ring, kp, k_w, v_w);
            ring_find(k_w + ((uint(kv_head) * uint(PIE_HEAD_DIM)) >> 1), k_seg, k_at);
            ring_find(v_w + ((uint(kv_head) * uint(PIE_HEAD_DIM)) >> 1), v_seg, v_at);
#else
            uint base = page_row(req, kp) + uint(kv_head) * uint(PIE_HEAD_DIM);
            pie_pool_find(SDPA_KV_SCALE * params.k_origin + base, k_seg, k_at);
            pie_pool_find(SDPA_KV_SCALE * params.v_origin + base, v_seg, v_at);
#endif
            // The row's elements first, from its one segment; then the sum
            // as the pool view read it.
            uint16_t kh[DPL];
#define SDPA_DECODE_K(buf_) \
            [ForceUnroll] for (uint j = 0u; j < uint(DPL); j++) { \
                uint dd = lane + j * width; \
                if (j < per_lane && dd < uint(PIE_HEAD_DIM)) { kh[j] = buf_[k_at + dd]; } else { kh[j] = uint16_t(0); } \
            }
            PIE_POOL_BRANCH(k_seg, SDPA_DECODE_K)
            // The fused forms the driver contracted the pool view's sums to,
            // stated: the loads moved, the bits stay.
            float partial = 0.0;
            [ForceUnroll] for (uint j = 0u; j < uint(DPL); j++) {
                uint dd = lane + j * width;
                if (j < per_lane && dd < uint(PIE_HEAD_DIM)) {
                    partial = fma(qreg[j], bf16_of(kh[j]), partial);
                }
            }
            float score = WaveActiveSum(partial);
            float hist, w;
            online_update(score, max_score, sum_exp, hist, w);
            uint16_t vh[DPL];
#define SDPA_DECODE_V(buf_) \
            [ForceUnroll] for (uint j = 0u; j < uint(DPL); j++) { \
                uint dd = lane + j * width; \
                if (j < per_lane && dd < uint(PIE_HEAD_DIM)) { vh[j] = buf_[v_at + dd]; } else { vh[j] = uint16_t(0); } \
            }
            PIE_POOL_BRANCH(v_seg, SDPA_DECODE_V)
            [ForceUnroll] for (uint j = 0u; j < uint(DPL); j++) {
                uint dd = lane + j * width;
                if (j < per_lane && dd < uint(PIE_HEAD_DIM)) {
                    acc[j] = fma(acc[j], hist, w * bf16_of(vh[j]));
                }
            }
        }
        [ForceUnroll] for (uint j = 0u; j < uint(DPL); j++) {
            uint dd = lane + j * width;
            if (j < per_lane && dd < uint(PIE_HEAD_DIM)) {
                sdpa_acc[sg][dd] = acc[j];
            }
        }
        if (WaveIsFirstLane()) {
            sdpa_max[sg] = max_score;
            sdpa_sum[sg] = sum_exp;
        }
    }
    GroupMemoryBarrierWithGroupSync();

    merged_max = NEG_INF;
    for (uint s = 0u; s < subgroups; s++) {
        merged_max = max(merged_max, sdpa_max[s]);
    }
    merged_sum = 0.0;
    for (uint s = 0u; s < subgroups; s++) {
        merged_sum += sdpa_sum[s] * exp(sdpa_max[s] - merged_max);
    }
}

float decode_fold(uint d, float merged_max, uint subgroups) {
    float out_acc = 0.0;
    for (uint s = 0u; s < subgroups; s++) {
        out_acc += sdpa_acc[s][d] * exp(sdpa_max[s] - merged_max);
    }
    return out_acc;
}

[shader("compute")]
[numthreads(PIE_DECODE_THREADS, 1, 1)]
void main(uint3 group : SV_GroupID, uint3 local : SV_GroupThreadID) {
    uint q_head = group.x;
    uint row = group.y;
    uint tid = local.x;
    int q_pos = position_ids[fire_row(row)];
#if defined(PIE_RING)
    int start = ring_start(keys_of(req_of_token[fire_row(row)]), q_pos);
#else
    int start = window_start(q_pos);
#endif
    uint base_row = row_base(row, q_head);
    float merged_max, merged_sum;
    uint subgroups;
#if defined(PIE_SPLIT)
    uint split = group.z;
    int keys = q_pos - start + 1;
    int usable = clamp((keys + params.min_keys - 1) / max(params.min_keys, 1), 1, params.splits);
    uint slice = (split * uint(params.n_rows) + row) * uint(params.n_q_heads) + q_head;
    if (int(split) >= usable) {
        if (tid == 0u) {
            part_lse[slice] = neg_inf();
        }
        return;
    }
    int chunk = (keys + usable - 1) / usable;
    int k_lo = start + int(split) * chunk;
    int k_hi = min(q_pos, k_lo + chunk - 1);
    decode_walk(row, q_head, tid, k_lo, k_hi, q_pos, start, merged_max, merged_sum, subgroups);
    if (usable == 1) {
        // One split covered the row: land it whole, and mark it done.
        if (tid == 0u) {
            part_lse[slice] = pos_inf();
#if defined(PIE_SPLIT_LSE)
            lse_out[fire_row(row) * uint(params.n_q_heads) + q_head] = lse_base2(merged_max, merged_sum);
#endif
        }
        for (uint d = tid; d < uint(PIE_HEAD_DIM); d += uint(PIE_DECODE_THREADS)) {
            float v = decode_fold(d, merged_max, subgroups);
            o[base_row + d] = uint16_t(f32_to_bf16(merged_sum == 0.0 ? v : v / merged_sum));
        }
        return;
    }
    if (tid == 0u) {
        part_lse[slice] = lse_base2(merged_max, merged_sum);
    }
    for (uint d = tid; d < uint(PIE_HEAD_DIM); d += uint(PIE_DECODE_THREADS)) {
        float v = decode_fold(d, merged_max, subgroups);
        part_out[slice * uint(PIE_HEAD_DIM) + d] = merged_sum == 0.0 ? 0.0 : v / merged_sum;
    }
#else
    decode_walk(row, q_head, tid, start, q_pos, q_pos, start, merged_max, merged_sum, subgroups);
#if defined(PIE_LSE)
    if (tid == 0u) {
        lse_out[fire_row(row) * uint(params.n_q_heads) + q_head] = lse_base2(merged_max, merged_sum);
    }
#endif
    for (uint d = tid; d < uint(PIE_HEAD_DIM); d += uint(PIE_DECODE_THREADS)) {
        float v = decode_fold(d, merged_max, subgroups);
        o[base_row + d] = uint16_t(f32_to_bf16(merged_sum == 0.0 ? v : v / merged_sum));
    }
#endif
}

#else

#if !defined(PIE_DECODE_THREADS)
#define PIE_DECODE_THREADS 256
#endif
#define MAX_WALKERS (PIE_DECODE_THREADS / 32)
// Dim pairs a lane holds: pair `lane + 32 j`, dims `2p` and `2p + 1`, so
// one word load serves two dims.
#define PPL ((PIE_HEAD_DIM / 2 + 31) / 32)

groupshared float sdpa_acc[MAX_WALKERS][PIE_HEAD_DIM];
groupshared float sdpa_max[MAX_WALKERS];
groupshared float sdpa_sum[MAX_WALKERS];
groupshared float sdpa_fold[MAX_WALKERS][32];
groupshared int sdpa_pos;
#if defined(PIE_RING)
groupshared int sdpa_start;
#endif

// The walkers' share of the row's keys: each 32-lane strip takes every
// `walkers`-th step from its own.
void decode_walk(uint row, uint q_head, uint tid, int k_lo, int k_hi, int q_pos, int start,
                 out float merged_max, out float merged_sum, out uint walkers) {
    int req = req_of_token[fire_row(row)];
#if defined(PIE_RING)
    Keys ring = keys_of(req);
#endif
    uint origin = mask_origin(row);
    uint base_row = row_base(row, q_head);
    int kv_head = int(q_head) / params.gqa_factor;

    uint width = 32u;
    uint walker = tid / 32u;
    uint lane = tid % 32u;
    walkers = uint(MAX_WALKERS);
    // Pairs a lane covers, rounded up: a strip wider than the head's pairs
    // leaves its upper lanes idle.
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
    // A strip reduces through workgroup memory, so every lane takes every
    // round, a strip past the keys with nothing to add.
    int rounds = (max(steps, 0) + int(walkers) - 1) / int(walkers);
    for (int round = 0; round < rounds; round++) {
        int n = round * int(walkers) + int(walker);
#if defined(PIE_SELECTED)
        int kp = -1;
        if (n < sel_end) {
            int c = selection[fire_row(row) * uint(params.top_k) + uint(n / params.ratio)];
            if (c >= 0 && c < nblocks) {
                kp = c * params.ratio + (n % params.ratio);
            }
        } else {
            kp = nblocks * params.ratio + (n - sel_end);
        }
        bool take = n < steps && kp >= 0 && keeps(row, origin, kp, q_pos, start);
#else
        int kp = k_lo + n;
        bool take = n < steps && keeps(row, origin, kp, q_pos, start);
#endif
        uint base = 0u;
        float partial = 0.0;
        uint k_seg = 0u, k_at = 0u, v_seg = 0u, v_at = 0u;
        if (take) {
#if defined(PIE_RING)
            uint k_w, v_w;
            key_at(req, ring, kp, k_w, v_w);
            ring_find(k_w + ((uint(kv_head) * uint(PIE_HEAD_DIM)) >> 1), k_seg, k_at);
            ring_find(v_w + ((uint(kv_head) * uint(PIE_HEAD_DIM)) >> 1), v_seg, v_at);
#else
            base = page_row(req, kp) + uint(kv_head) * uint(PIE_HEAD_DIM);
            pie_pool_find(SDPA_KV_SCALE * params.k_origin + (base >> 1), k_seg, k_at);
            pie_pool_find(SDPA_KV_SCALE * params.v_origin + (base >> 1), v_seg, v_at);
#endif
            // The row's words first, from its one segment; then the sum as
            // the pool view read it.
            uint kw[PPL];
#define SDPA_WORDS_K(buf_) \
            [ForceUnroll] for (uint j = 0u; j < uint(PPL); j++) { \
                uint pp = lane + j * width; \
                if (j < per_lane && pp < pairs) { kw[j] = buf_[k_at + pp]; } else { kw[j] = 0u; } \
            }
            PIE_POOL_BRANCH(k_seg, SDPA_WORDS_K)
            [ForceUnroll] for (uint j = 0u; j < uint(PPL); j++) {
                uint pp = lane + j * width;
                if (j < per_lane && pp < pairs) {
                    partial += dot(qreg[j], word_pair(kw[j]));
                }
            }
        }
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
        if (take) {
            float hist, w;
            online_update(score, max_score, sum_exp, hist, w);
            uint vw[PPL];
#define SDPA_WORDS_V(buf_) \
            [ForceUnroll] for (uint j = 0u; j < uint(PPL); j++) { \
                uint pp = lane + j * width; \
                if (j < per_lane && pp < pairs) { vw[j] = buf_[v_at + pp]; } else { vw[j] = 0u; } \
            }
            PIE_POOL_BRANCH(v_seg, SDPA_WORDS_V)
            [ForceUnroll] for (uint j = 0u; j < uint(PPL); j++) {
                uint pp = lane + j * width;
                if (j < per_lane && pp < pairs) {
                    acc[j] = acc[j] * hist + w * word_pair(vw[j]);
                }
            }
        }
    }
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
        sdpa_pos = position_ids[fire_row(row)];
#if defined(PIE_RING)
        sdpa_start = ring_start(keys_of(req_of_token[fire_row(row)]), sdpa_pos);
#endif
    }
    GroupMemoryBarrierWithGroupSync();
    int q_pos = workgroupUniformLoad(sdpa_pos);
#if defined(PIE_RING)
    int start = workgroupUniformLoad(sdpa_start);
#else
    int start = window_start(q_pos);
#endif
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
            part_lse[slice] = neg_inf();
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
            part_lse[slice] = pos_inf();
#if defined(PIE_SPLIT_LSE)
            lse_out[fire_row(row) * uint(params.n_q_heads) + q_head] = lse_base2(merged_max, merged_sum);
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
        lse_out[fire_row(row) * uint(params.n_q_heads) + q_head] = lse_base2(merged_max, merged_sum);
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
// pie:instantiate attn_sdpa_decode_lanes_8_bf16_d_64 PIE_HEAD_DIM=64 PIE_TIER=subgroup PIE_TARGET=spirv PIE_MIN_LANES=8
// pie:instantiate attn_sdpa_decode_t128_bf16_d_64 PIE_HEAD_DIM=64 PIE_TIER=subgroup PIE_TARGET=spirv PIE_DECODE_THREADS=128
// pie:instantiate attn_sdpa_decode_t128_lanes_8_bf16_d_64 PIE_HEAD_DIM=64 PIE_TIER=subgroup PIE_TARGET=spirv PIE_DECODE_THREADS=128 PIE_MIN_LANES=8
// pie:instantiate attn_sdpa_decode_bf16_d_64 PIE_HEAD_DIM=64
// pie:instantiate attn_sdpa_decode_t128_bf16_d_64 PIE_HEAD_DIM=64 PIE_DECODE_THREADS=128
// pie:instantiate attn_sdpa_decode_lse_bf16_d_64 PIE_HEAD_DIM=64 PIE_LSE=1 PIE_TIER=subgroup PIE_TARGET=spirv
// pie:instantiate attn_sdpa_decode_lse_lanes_8_bf16_d_64 PIE_HEAD_DIM=64 PIE_LSE=1 PIE_TIER=subgroup PIE_TARGET=spirv PIE_MIN_LANES=8
// pie:instantiate attn_sdpa_decode_lse_t128_bf16_d_64 PIE_HEAD_DIM=64 PIE_LSE=1 PIE_TIER=subgroup PIE_TARGET=spirv PIE_DECODE_THREADS=128
// pie:instantiate attn_sdpa_decode_lse_t128_lanes_8_bf16_d_64 PIE_HEAD_DIM=64 PIE_LSE=1 PIE_TIER=subgroup PIE_TARGET=spirv PIE_DECODE_THREADS=128 PIE_MIN_LANES=8
// pie:instantiate attn_sdpa_decode_lse_bf16_d_64 PIE_HEAD_DIM=64 PIE_LSE=1
// pie:instantiate attn_sdpa_decode_lse_t128_bf16_d_64 PIE_HEAD_DIM=64 PIE_LSE=1 PIE_DECODE_THREADS=128
// pie:instantiate attn_sdpa_selected_bf16_d_64 PIE_HEAD_DIM=64 PIE_SELECTED=1 PIE_TIER=subgroup PIE_TARGET=spirv
// pie:instantiate attn_sdpa_selected_lanes_8_bf16_d_64 PIE_HEAD_DIM=64 PIE_SELECTED=1 PIE_TIER=subgroup PIE_TARGET=spirv PIE_MIN_LANES=8
// pie:instantiate attn_sdpa_selected_t128_bf16_d_64 PIE_HEAD_DIM=64 PIE_SELECTED=1 PIE_TIER=subgroup PIE_TARGET=spirv PIE_DECODE_THREADS=128
// pie:instantiate attn_sdpa_selected_t128_lanes_8_bf16_d_64 PIE_HEAD_DIM=64 PIE_SELECTED=1 PIE_TIER=subgroup PIE_TARGET=spirv PIE_DECODE_THREADS=128 PIE_MIN_LANES=8
// pie:instantiate attn_sdpa_selected_bf16_d_64 PIE_HEAD_DIM=64 PIE_SELECTED=1
// pie:instantiate attn_sdpa_selected_t128_bf16_d_64 PIE_HEAD_DIM=64 PIE_SELECTED=1 PIE_DECODE_THREADS=128
// pie:instantiate attn_sdpa_split_bf16_d_64 PIE_HEAD_DIM=64 PIE_SPLIT=1 PIE_TIER=subgroup PIE_TARGET=spirv
// pie:instantiate attn_sdpa_split_lanes_8_bf16_d_64 PIE_HEAD_DIM=64 PIE_SPLIT=1 PIE_TIER=subgroup PIE_TARGET=spirv PIE_MIN_LANES=8
// pie:instantiate attn_sdpa_split_t128_bf16_d_64 PIE_HEAD_DIM=64 PIE_SPLIT=1 PIE_TIER=subgroup PIE_TARGET=spirv PIE_DECODE_THREADS=128
// pie:instantiate attn_sdpa_split_t128_lanes_8_bf16_d_64 PIE_HEAD_DIM=64 PIE_SPLIT=1 PIE_TIER=subgroup PIE_TARGET=spirv PIE_DECODE_THREADS=128 PIE_MIN_LANES=8
// pie:instantiate attn_sdpa_split_bf16_d_64 PIE_HEAD_DIM=64 PIE_SPLIT=1
// pie:instantiate attn_sdpa_split_t128_bf16_d_64 PIE_HEAD_DIM=64 PIE_SPLIT=1 PIE_DECODE_THREADS=128
// pie:instantiate attn_sdpa_split_lse_bf16_d_64 PIE_HEAD_DIM=64 PIE_SPLIT=1 PIE_SPLIT_LSE=1 PIE_TIER=subgroup PIE_TARGET=spirv
// pie:instantiate attn_sdpa_split_lse_lanes_8_bf16_d_64 PIE_HEAD_DIM=64 PIE_SPLIT=1 PIE_SPLIT_LSE=1 PIE_TIER=subgroup PIE_TARGET=spirv PIE_MIN_LANES=8
// pie:instantiate attn_sdpa_split_lse_t128_bf16_d_64 PIE_HEAD_DIM=64 PIE_SPLIT=1 PIE_SPLIT_LSE=1 PIE_TIER=subgroup PIE_TARGET=spirv PIE_DECODE_THREADS=128
// pie:instantiate attn_sdpa_split_lse_t128_lanes_8_bf16_d_64 PIE_HEAD_DIM=64 PIE_SPLIT=1 PIE_SPLIT_LSE=1 PIE_TIER=subgroup PIE_TARGET=spirv PIE_DECODE_THREADS=128 PIE_MIN_LANES=8
// pie:instantiate attn_sdpa_split_lse_bf16_d_64 PIE_HEAD_DIM=64 PIE_SPLIT=1 PIE_SPLIT_LSE=1
// pie:instantiate attn_sdpa_split_lse_t128_bf16_d_64 PIE_HEAD_DIM=64 PIE_SPLIT=1 PIE_SPLIT_LSE=1 PIE_DECODE_THREADS=128
// pie:instantiate attn_sdpa_tiled_bf16_d_64 PIE_HEAD_DIM=64 PIE_TILED=1 PIE_TILE_ROWS=32
// pie:instantiate attn_sdpa_tiled_r16_bf16_d_64 PIE_HEAD_DIM=64 PIE_TILED=1 PIE_TILE_ROWS=16
// pie:instantiate attn_sdpa_tiled_r8_bf16_d_64 PIE_HEAD_DIM=64 PIE_TILED=1 PIE_TILE_ROWS=8
// pie:instantiate attn_sdpa_tiled_lse_bf16_d_64 PIE_HEAD_DIM=64 PIE_TILED=1 PIE_LSE=1 PIE_TILE_ROWS=32
// pie:instantiate attn_sdpa_tiled_lse_r16_bf16_d_64 PIE_HEAD_DIM=64 PIE_TILED=1 PIE_LSE=1 PIE_TILE_ROWS=16
// pie:instantiate attn_sdpa_tiled_lse_r8_bf16_d_64 PIE_HEAD_DIM=64 PIE_TILED=1 PIE_LSE=1 PIE_TILE_ROWS=8
// pie:instantiate attn_sdpa_tiled_l16_r16_bf16_d_64 PIE_HEAD_DIM=64 PIE_TILED=1 PIE_TILE_LANES=16 PIE_TILE_ROWS=16
// pie:instantiate attn_sdpa_tiled_l16_r16_bf16_d_64 PIE_HEAD_DIM=64 PIE_TILED=1 PIE_TILE_LANES=16 PIE_TILE_ROWS=16 PIE_TIER=subgroup
// pie:instantiate attn_sdpa_tiled_lse_l16_r16_bf16_d_64 PIE_HEAD_DIM=64 PIE_TILED=1 PIE_LSE=1 PIE_TILE_LANES=16 PIE_TILE_ROWS=16
// pie:instantiate attn_sdpa_tiled_lse_l16_r16_bf16_d_64 PIE_HEAD_DIM=64 PIE_TILED=1 PIE_LSE=1 PIE_TILE_LANES=16 PIE_TILE_ROWS=16 PIE_TIER=subgroup
// pie:instantiate attn_sdpa_decode_bf16_d_128 PIE_HEAD_DIM=128 PIE_TIER=subgroup PIE_TARGET=spirv
// pie:instantiate attn_sdpa_decode_lanes_8_bf16_d_128 PIE_HEAD_DIM=128 PIE_TIER=subgroup PIE_TARGET=spirv PIE_MIN_LANES=8
// pie:instantiate attn_sdpa_decode_t128_bf16_d_128 PIE_HEAD_DIM=128 PIE_TIER=subgroup PIE_TARGET=spirv PIE_DECODE_THREADS=128
// pie:instantiate attn_sdpa_decode_t128_lanes_8_bf16_d_128 PIE_HEAD_DIM=128 PIE_TIER=subgroup PIE_TARGET=spirv PIE_DECODE_THREADS=128 PIE_MIN_LANES=8
// pie:instantiate attn_sdpa_decode_bf16_d_128 PIE_HEAD_DIM=128
// pie:instantiate attn_sdpa_decode_t128_bf16_d_128 PIE_HEAD_DIM=128 PIE_DECODE_THREADS=128
// pie:instantiate attn_sdpa_decode_lse_bf16_d_128 PIE_HEAD_DIM=128 PIE_LSE=1 PIE_TIER=subgroup PIE_TARGET=spirv
// pie:instantiate attn_sdpa_decode_lse_lanes_8_bf16_d_128 PIE_HEAD_DIM=128 PIE_LSE=1 PIE_TIER=subgroup PIE_TARGET=spirv PIE_MIN_LANES=8
// pie:instantiate attn_sdpa_decode_lse_t128_bf16_d_128 PIE_HEAD_DIM=128 PIE_LSE=1 PIE_TIER=subgroup PIE_TARGET=spirv PIE_DECODE_THREADS=128
// pie:instantiate attn_sdpa_decode_lse_t128_lanes_8_bf16_d_128 PIE_HEAD_DIM=128 PIE_LSE=1 PIE_TIER=subgroup PIE_TARGET=spirv PIE_DECODE_THREADS=128 PIE_MIN_LANES=8
// pie:instantiate attn_sdpa_decode_lse_bf16_d_128 PIE_HEAD_DIM=128 PIE_LSE=1
// pie:instantiate attn_sdpa_decode_lse_t128_bf16_d_128 PIE_HEAD_DIM=128 PIE_LSE=1 PIE_DECODE_THREADS=128
// pie:instantiate attn_sdpa_selected_bf16_d_128 PIE_HEAD_DIM=128 PIE_SELECTED=1 PIE_TIER=subgroup PIE_TARGET=spirv
// pie:instantiate attn_sdpa_selected_lanes_8_bf16_d_128 PIE_HEAD_DIM=128 PIE_SELECTED=1 PIE_TIER=subgroup PIE_TARGET=spirv PIE_MIN_LANES=8
// pie:instantiate attn_sdpa_selected_t128_bf16_d_128 PIE_HEAD_DIM=128 PIE_SELECTED=1 PIE_TIER=subgroup PIE_TARGET=spirv PIE_DECODE_THREADS=128
// pie:instantiate attn_sdpa_selected_t128_lanes_8_bf16_d_128 PIE_HEAD_DIM=128 PIE_SELECTED=1 PIE_TIER=subgroup PIE_TARGET=spirv PIE_DECODE_THREADS=128 PIE_MIN_LANES=8
// pie:instantiate attn_sdpa_selected_bf16_d_128 PIE_HEAD_DIM=128 PIE_SELECTED=1
// pie:instantiate attn_sdpa_selected_t128_bf16_d_128 PIE_HEAD_DIM=128 PIE_SELECTED=1 PIE_DECODE_THREADS=128
// pie:instantiate attn_sdpa_split_bf16_d_128 PIE_HEAD_DIM=128 PIE_SPLIT=1 PIE_TIER=subgroup PIE_TARGET=spirv
// pie:instantiate attn_sdpa_split_lanes_8_bf16_d_128 PIE_HEAD_DIM=128 PIE_SPLIT=1 PIE_TIER=subgroup PIE_TARGET=spirv PIE_MIN_LANES=8
// pie:instantiate attn_sdpa_split_t128_bf16_d_128 PIE_HEAD_DIM=128 PIE_SPLIT=1 PIE_TIER=subgroup PIE_TARGET=spirv PIE_DECODE_THREADS=128
// pie:instantiate attn_sdpa_split_t128_lanes_8_bf16_d_128 PIE_HEAD_DIM=128 PIE_SPLIT=1 PIE_TIER=subgroup PIE_TARGET=spirv PIE_DECODE_THREADS=128 PIE_MIN_LANES=8
// pie:instantiate attn_sdpa_split_bf16_d_128 PIE_HEAD_DIM=128 PIE_SPLIT=1
// pie:instantiate attn_sdpa_split_t128_bf16_d_128 PIE_HEAD_DIM=128 PIE_SPLIT=1 PIE_DECODE_THREADS=128
// pie:instantiate attn_sdpa_split_lse_bf16_d_128 PIE_HEAD_DIM=128 PIE_SPLIT=1 PIE_SPLIT_LSE=1 PIE_TIER=subgroup PIE_TARGET=spirv
// pie:instantiate attn_sdpa_split_lse_lanes_8_bf16_d_128 PIE_HEAD_DIM=128 PIE_SPLIT=1 PIE_SPLIT_LSE=1 PIE_TIER=subgroup PIE_TARGET=spirv PIE_MIN_LANES=8
// pie:instantiate attn_sdpa_split_lse_t128_bf16_d_128 PIE_HEAD_DIM=128 PIE_SPLIT=1 PIE_SPLIT_LSE=1 PIE_TIER=subgroup PIE_TARGET=spirv PIE_DECODE_THREADS=128
// pie:instantiate attn_sdpa_split_lse_t128_lanes_8_bf16_d_128 PIE_HEAD_DIM=128 PIE_SPLIT=1 PIE_SPLIT_LSE=1 PIE_TIER=subgroup PIE_TARGET=spirv PIE_DECODE_THREADS=128 PIE_MIN_LANES=8
// pie:instantiate attn_sdpa_split_lse_bf16_d_128 PIE_HEAD_DIM=128 PIE_SPLIT=1 PIE_SPLIT_LSE=1
// pie:instantiate attn_sdpa_split_lse_t128_bf16_d_128 PIE_HEAD_DIM=128 PIE_SPLIT=1 PIE_SPLIT_LSE=1 PIE_DECODE_THREADS=128
// pie:instantiate attn_sdpa_tiled_bf16_d_128 PIE_HEAD_DIM=128 PIE_TILED=1 PIE_TILE_ROWS=32
// pie:instantiate attn_sdpa_tiled_r16_bf16_d_128 PIE_HEAD_DIM=128 PIE_TILED=1 PIE_TILE_ROWS=16
// pie:instantiate attn_sdpa_tiled_r8_bf16_d_128 PIE_HEAD_DIM=128 PIE_TILED=1 PIE_TILE_ROWS=8
// pie:instantiate attn_sdpa_tiled_lse_bf16_d_128 PIE_HEAD_DIM=128 PIE_TILED=1 PIE_LSE=1 PIE_TILE_ROWS=32
// pie:instantiate attn_sdpa_tiled_lse_r16_bf16_d_128 PIE_HEAD_DIM=128 PIE_TILED=1 PIE_LSE=1 PIE_TILE_ROWS=16
// pie:instantiate attn_sdpa_tiled_lse_r8_bf16_d_128 PIE_HEAD_DIM=128 PIE_TILED=1 PIE_LSE=1 PIE_TILE_ROWS=8
// pie:instantiate attn_sdpa_tiled_l16_r16_bf16_d_128 PIE_HEAD_DIM=128 PIE_TILED=1 PIE_TILE_LANES=16 PIE_TILE_ROWS=16
// pie:instantiate attn_sdpa_tiled_l16_r16_bf16_d_128 PIE_HEAD_DIM=128 PIE_TILED=1 PIE_TILE_LANES=16 PIE_TILE_ROWS=16 PIE_TIER=subgroup
// pie:instantiate attn_sdpa_tiled_lse_l16_r16_bf16_d_128 PIE_HEAD_DIM=128 PIE_TILED=1 PIE_LSE=1 PIE_TILE_LANES=16 PIE_TILE_ROWS=16
// pie:instantiate attn_sdpa_tiled_lse_l16_r16_bf16_d_128 PIE_HEAD_DIM=128 PIE_TILED=1 PIE_LSE=1 PIE_TILE_LANES=16 PIE_TILE_ROWS=16 PIE_TIER=subgroup
// pie:instantiate attn_sdpa_decode_bf16_d_256 PIE_HEAD_DIM=256 PIE_TIER=subgroup PIE_TARGET=spirv
// pie:instantiate attn_sdpa_decode_lanes_8_bf16_d_256 PIE_HEAD_DIM=256 PIE_TIER=subgroup PIE_TARGET=spirv PIE_MIN_LANES=8
// pie:instantiate attn_sdpa_decode_t128_bf16_d_256 PIE_HEAD_DIM=256 PIE_TIER=subgroup PIE_TARGET=spirv PIE_DECODE_THREADS=128
// pie:instantiate attn_sdpa_decode_t128_lanes_8_bf16_d_256 PIE_HEAD_DIM=256 PIE_TIER=subgroup PIE_TARGET=spirv PIE_DECODE_THREADS=128 PIE_MIN_LANES=8
// pie:instantiate attn_sdpa_decode_bf16_d_256 PIE_HEAD_DIM=256
// pie:instantiate attn_sdpa_decode_t128_bf16_d_256 PIE_HEAD_DIM=256 PIE_DECODE_THREADS=128
// pie:instantiate attn_sdpa_decode_lse_bf16_d_256 PIE_HEAD_DIM=256 PIE_LSE=1 PIE_TIER=subgroup PIE_TARGET=spirv
// pie:instantiate attn_sdpa_decode_lse_lanes_8_bf16_d_256 PIE_HEAD_DIM=256 PIE_LSE=1 PIE_TIER=subgroup PIE_TARGET=spirv PIE_MIN_LANES=8
// pie:instantiate attn_sdpa_decode_lse_t128_bf16_d_256 PIE_HEAD_DIM=256 PIE_LSE=1 PIE_TIER=subgroup PIE_TARGET=spirv PIE_DECODE_THREADS=128
// pie:instantiate attn_sdpa_decode_lse_t128_lanes_8_bf16_d_256 PIE_HEAD_DIM=256 PIE_LSE=1 PIE_TIER=subgroup PIE_TARGET=spirv PIE_DECODE_THREADS=128 PIE_MIN_LANES=8
// pie:instantiate attn_sdpa_decode_lse_bf16_d_256 PIE_HEAD_DIM=256 PIE_LSE=1
// pie:instantiate attn_sdpa_decode_lse_t128_bf16_d_256 PIE_HEAD_DIM=256 PIE_LSE=1 PIE_DECODE_THREADS=128
// pie:instantiate attn_sdpa_selected_bf16_d_256 PIE_HEAD_DIM=256 PIE_SELECTED=1 PIE_TIER=subgroup PIE_TARGET=spirv
// pie:instantiate attn_sdpa_selected_lanes_8_bf16_d_256 PIE_HEAD_DIM=256 PIE_SELECTED=1 PIE_TIER=subgroup PIE_TARGET=spirv PIE_MIN_LANES=8
// pie:instantiate attn_sdpa_selected_t128_bf16_d_256 PIE_HEAD_DIM=256 PIE_SELECTED=1 PIE_TIER=subgroup PIE_TARGET=spirv PIE_DECODE_THREADS=128
// pie:instantiate attn_sdpa_selected_t128_lanes_8_bf16_d_256 PIE_HEAD_DIM=256 PIE_SELECTED=1 PIE_TIER=subgroup PIE_TARGET=spirv PIE_DECODE_THREADS=128 PIE_MIN_LANES=8
// pie:instantiate attn_sdpa_selected_bf16_d_256 PIE_HEAD_DIM=256 PIE_SELECTED=1
// pie:instantiate attn_sdpa_selected_t128_bf16_d_256 PIE_HEAD_DIM=256 PIE_SELECTED=1 PIE_DECODE_THREADS=128
// pie:instantiate attn_sdpa_split_bf16_d_256 PIE_HEAD_DIM=256 PIE_SPLIT=1 PIE_TIER=subgroup PIE_TARGET=spirv
// pie:instantiate attn_sdpa_split_lanes_8_bf16_d_256 PIE_HEAD_DIM=256 PIE_SPLIT=1 PIE_TIER=subgroup PIE_TARGET=spirv PIE_MIN_LANES=8
// pie:instantiate attn_sdpa_split_t128_bf16_d_256 PIE_HEAD_DIM=256 PIE_SPLIT=1 PIE_TIER=subgroup PIE_TARGET=spirv PIE_DECODE_THREADS=128
// pie:instantiate attn_sdpa_split_t128_lanes_8_bf16_d_256 PIE_HEAD_DIM=256 PIE_SPLIT=1 PIE_TIER=subgroup PIE_TARGET=spirv PIE_DECODE_THREADS=128 PIE_MIN_LANES=8
// pie:instantiate attn_sdpa_split_bf16_d_256 PIE_HEAD_DIM=256 PIE_SPLIT=1
// pie:instantiate attn_sdpa_split_t128_bf16_d_256 PIE_HEAD_DIM=256 PIE_SPLIT=1 PIE_DECODE_THREADS=128
// pie:instantiate attn_sdpa_split_lse_bf16_d_256 PIE_HEAD_DIM=256 PIE_SPLIT=1 PIE_SPLIT_LSE=1 PIE_TIER=subgroup PIE_TARGET=spirv
// pie:instantiate attn_sdpa_split_lse_lanes_8_bf16_d_256 PIE_HEAD_DIM=256 PIE_SPLIT=1 PIE_SPLIT_LSE=1 PIE_TIER=subgroup PIE_TARGET=spirv PIE_MIN_LANES=8
// pie:instantiate attn_sdpa_split_lse_t128_bf16_d_256 PIE_HEAD_DIM=256 PIE_SPLIT=1 PIE_SPLIT_LSE=1 PIE_TIER=subgroup PIE_TARGET=spirv PIE_DECODE_THREADS=128
// pie:instantiate attn_sdpa_split_lse_t128_lanes_8_bf16_d_256 PIE_HEAD_DIM=256 PIE_SPLIT=1 PIE_SPLIT_LSE=1 PIE_TIER=subgroup PIE_TARGET=spirv PIE_DECODE_THREADS=128 PIE_MIN_LANES=8
// pie:instantiate attn_sdpa_split_lse_bf16_d_256 PIE_HEAD_DIM=256 PIE_SPLIT=1 PIE_SPLIT_LSE=1
// pie:instantiate attn_sdpa_split_lse_t128_bf16_d_256 PIE_HEAD_DIM=256 PIE_SPLIT=1 PIE_SPLIT_LSE=1 PIE_DECODE_THREADS=128
// pie:instantiate attn_sdpa_tiled_bf16_d_256 PIE_HEAD_DIM=256 PIE_TILED=1 PIE_TILE_ROWS=32
// pie:instantiate attn_sdpa_tiled_r16_bf16_d_256 PIE_HEAD_DIM=256 PIE_TILED=1 PIE_TILE_ROWS=16
// pie:instantiate attn_sdpa_tiled_r8_bf16_d_256 PIE_HEAD_DIM=256 PIE_TILED=1 PIE_TILE_ROWS=8
// pie:instantiate attn_sdpa_tiled_lse_bf16_d_256 PIE_HEAD_DIM=256 PIE_TILED=1 PIE_LSE=1 PIE_TILE_ROWS=32
// pie:instantiate attn_sdpa_tiled_lse_r16_bf16_d_256 PIE_HEAD_DIM=256 PIE_TILED=1 PIE_LSE=1 PIE_TILE_ROWS=16
// pie:instantiate attn_sdpa_tiled_lse_r8_bf16_d_256 PIE_HEAD_DIM=256 PIE_TILED=1 PIE_LSE=1 PIE_TILE_ROWS=8
// pie:instantiate attn_sdpa_tiled_l16_r16_bf16_d_256 PIE_HEAD_DIM=256 PIE_TILED=1 PIE_TILE_LANES=16 PIE_TILE_ROWS=16
// pie:instantiate attn_sdpa_tiled_l16_r16_bf16_d_256 PIE_HEAD_DIM=256 PIE_TILED=1 PIE_TILE_LANES=16 PIE_TILE_ROWS=16 PIE_TIER=subgroup
// pie:instantiate attn_sdpa_tiled_lse_l16_r16_bf16_d_256 PIE_HEAD_DIM=256 PIE_TILED=1 PIE_LSE=1 PIE_TILE_LANES=16 PIE_TILE_ROWS=16
// pie:instantiate attn_sdpa_tiled_lse_l16_r16_bf16_d_256 PIE_HEAD_DIM=256 PIE_TILED=1 PIE_LSE=1 PIE_TILE_LANES=16 PIE_TILE_ROWS=16 PIE_TIER=subgroup
// pie:instantiate attn_sdpa_decode_bf16_d_512 PIE_HEAD_DIM=512 PIE_TIER=subgroup PIE_TARGET=spirv
// pie:instantiate attn_sdpa_decode_lanes_8_bf16_d_512 PIE_HEAD_DIM=512 PIE_TIER=subgroup PIE_TARGET=spirv PIE_MIN_LANES=8
// pie:instantiate attn_sdpa_decode_t128_bf16_d_512 PIE_HEAD_DIM=512 PIE_TIER=subgroup PIE_TARGET=spirv PIE_DECODE_THREADS=128
// pie:instantiate attn_sdpa_decode_t128_lanes_8_bf16_d_512 PIE_HEAD_DIM=512 PIE_TIER=subgroup PIE_TARGET=spirv PIE_DECODE_THREADS=128 PIE_MIN_LANES=8
// pie:instantiate attn_sdpa_decode_bf16_d_512 PIE_HEAD_DIM=512
// pie:instantiate attn_sdpa_decode_t128_bf16_d_512 PIE_HEAD_DIM=512 PIE_DECODE_THREADS=128
// pie:instantiate attn_sdpa_decode_lse_bf16_d_512 PIE_HEAD_DIM=512 PIE_LSE=1 PIE_TIER=subgroup PIE_TARGET=spirv
// pie:instantiate attn_sdpa_decode_lse_lanes_8_bf16_d_512 PIE_HEAD_DIM=512 PIE_LSE=1 PIE_TIER=subgroup PIE_TARGET=spirv PIE_MIN_LANES=8
// pie:instantiate attn_sdpa_decode_lse_t128_bf16_d_512 PIE_HEAD_DIM=512 PIE_LSE=1 PIE_TIER=subgroup PIE_TARGET=spirv PIE_DECODE_THREADS=128
// pie:instantiate attn_sdpa_decode_lse_t128_lanes_8_bf16_d_512 PIE_HEAD_DIM=512 PIE_LSE=1 PIE_TIER=subgroup PIE_TARGET=spirv PIE_DECODE_THREADS=128 PIE_MIN_LANES=8
// pie:instantiate attn_sdpa_decode_lse_bf16_d_512 PIE_HEAD_DIM=512 PIE_LSE=1
// pie:instantiate attn_sdpa_decode_lse_t128_bf16_d_512 PIE_HEAD_DIM=512 PIE_LSE=1 PIE_DECODE_THREADS=128
// pie:instantiate attn_sdpa_selected_bf16_d_512 PIE_HEAD_DIM=512 PIE_SELECTED=1 PIE_TIER=subgroup PIE_TARGET=spirv
// pie:instantiate attn_sdpa_selected_lanes_8_bf16_d_512 PIE_HEAD_DIM=512 PIE_SELECTED=1 PIE_TIER=subgroup PIE_TARGET=spirv PIE_MIN_LANES=8
// pie:instantiate attn_sdpa_selected_t128_bf16_d_512 PIE_HEAD_DIM=512 PIE_SELECTED=1 PIE_TIER=subgroup PIE_TARGET=spirv PIE_DECODE_THREADS=128
// pie:instantiate attn_sdpa_selected_t128_lanes_8_bf16_d_512 PIE_HEAD_DIM=512 PIE_SELECTED=1 PIE_TIER=subgroup PIE_TARGET=spirv PIE_DECODE_THREADS=128 PIE_MIN_LANES=8
// pie:instantiate attn_sdpa_selected_bf16_d_512 PIE_HEAD_DIM=512 PIE_SELECTED=1
// pie:instantiate attn_sdpa_selected_t128_bf16_d_512 PIE_HEAD_DIM=512 PIE_SELECTED=1 PIE_DECODE_THREADS=128
// pie:instantiate attn_sdpa_split_bf16_d_512 PIE_HEAD_DIM=512 PIE_SPLIT=1 PIE_TIER=subgroup PIE_TARGET=spirv
// pie:instantiate attn_sdpa_split_lanes_8_bf16_d_512 PIE_HEAD_DIM=512 PIE_SPLIT=1 PIE_TIER=subgroup PIE_TARGET=spirv PIE_MIN_LANES=8
// pie:instantiate attn_sdpa_split_t128_bf16_d_512 PIE_HEAD_DIM=512 PIE_SPLIT=1 PIE_TIER=subgroup PIE_TARGET=spirv PIE_DECODE_THREADS=128
// pie:instantiate attn_sdpa_split_t128_lanes_8_bf16_d_512 PIE_HEAD_DIM=512 PIE_SPLIT=1 PIE_TIER=subgroup PIE_TARGET=spirv PIE_DECODE_THREADS=128 PIE_MIN_LANES=8
// pie:instantiate attn_sdpa_split_bf16_d_512 PIE_HEAD_DIM=512 PIE_SPLIT=1
// pie:instantiate attn_sdpa_split_t128_bf16_d_512 PIE_HEAD_DIM=512 PIE_SPLIT=1 PIE_DECODE_THREADS=128
// pie:instantiate attn_sdpa_split_lse_bf16_d_512 PIE_HEAD_DIM=512 PIE_SPLIT=1 PIE_SPLIT_LSE=1 PIE_TIER=subgroup PIE_TARGET=spirv
// pie:instantiate attn_sdpa_split_lse_lanes_8_bf16_d_512 PIE_HEAD_DIM=512 PIE_SPLIT=1 PIE_SPLIT_LSE=1 PIE_TIER=subgroup PIE_TARGET=spirv PIE_MIN_LANES=8
// pie:instantiate attn_sdpa_split_lse_t128_bf16_d_512 PIE_HEAD_DIM=512 PIE_SPLIT=1 PIE_SPLIT_LSE=1 PIE_TIER=subgroup PIE_TARGET=spirv PIE_DECODE_THREADS=128
// pie:instantiate attn_sdpa_split_lse_t128_lanes_8_bf16_d_512 PIE_HEAD_DIM=512 PIE_SPLIT=1 PIE_SPLIT_LSE=1 PIE_TIER=subgroup PIE_TARGET=spirv PIE_DECODE_THREADS=128 PIE_MIN_LANES=8
// pie:instantiate attn_sdpa_split_lse_bf16_d_512 PIE_HEAD_DIM=512 PIE_SPLIT=1 PIE_SPLIT_LSE=1
// pie:instantiate attn_sdpa_split_lse_t128_bf16_d_512 PIE_HEAD_DIM=512 PIE_SPLIT=1 PIE_SPLIT_LSE=1 PIE_DECODE_THREADS=128
// pie:instantiate attn_sdpa_tiled_bf16_d_512 PIE_HEAD_DIM=512 PIE_TILED=1 PIE_TILE_ROWS=32
// pie:instantiate attn_sdpa_tiled_r16_bf16_d_512 PIE_HEAD_DIM=512 PIE_TILED=1 PIE_TILE_ROWS=16
// pie:instantiate attn_sdpa_tiled_r8_bf16_d_512 PIE_HEAD_DIM=512 PIE_TILED=1 PIE_TILE_ROWS=8
// pie:instantiate attn_sdpa_tiled_lse_bf16_d_512 PIE_HEAD_DIM=512 PIE_TILED=1 PIE_LSE=1 PIE_TILE_ROWS=32
// pie:instantiate attn_sdpa_tiled_lse_r16_bf16_d_512 PIE_HEAD_DIM=512 PIE_TILED=1 PIE_LSE=1 PIE_TILE_ROWS=16
// pie:instantiate attn_sdpa_tiled_lse_r8_bf16_d_512 PIE_HEAD_DIM=512 PIE_TILED=1 PIE_LSE=1 PIE_TILE_ROWS=8
// pie:instantiate attn_sdpa_tiled_l16_r16_bf16_d_512 PIE_HEAD_DIM=512 PIE_TILED=1 PIE_TILE_LANES=16 PIE_TILE_ROWS=16
// pie:instantiate attn_sdpa_tiled_l16_r16_bf16_d_512 PIE_HEAD_DIM=512 PIE_TILED=1 PIE_TILE_LANES=16 PIE_TILE_ROWS=16 PIE_TIER=subgroup
// pie:instantiate attn_sdpa_tiled_lse_l16_r16_bf16_d_512 PIE_HEAD_DIM=512 PIE_TILED=1 PIE_LSE=1 PIE_TILE_LANES=16 PIE_TILE_ROWS=16
// pie:instantiate attn_sdpa_tiled_lse_l16_r16_bf16_d_512 PIE_HEAD_DIM=512 PIE_TILED=1 PIE_LSE=1 PIE_TILE_LANES=16 PIE_TILE_ROWS=16 PIE_TIER=subgroup
// pie:instantiate attn_sdpa_fold_bf16 PIE_FOLD=1 PIE_GROUP_X=256
// pie:instantiate attn_sdpa_fold_lse_bf16 PIE_FOLD=1 PIE_LSE=1 PIE_GROUP_X=256

// pie:instantiate attn_sdpa_tiled_seg1_bf16_d_64 PIE_HEAD_DIM=64 PIE_TILED=1 PIE_TILE_ROWS=32 PIE_TARGET=spirv PIE_ONE_SEGMENT=1
// pie:instantiate attn_sdpa_tiled_r16_seg1_bf16_d_64 PIE_HEAD_DIM=64 PIE_TILED=1 PIE_TILE_ROWS=16 PIE_TARGET=spirv PIE_ONE_SEGMENT=1
// pie:instantiate attn_sdpa_tiled_r8_seg1_bf16_d_64 PIE_HEAD_DIM=64 PIE_TILED=1 PIE_TILE_ROWS=8 PIE_TARGET=spirv PIE_ONE_SEGMENT=1
// pie:instantiate attn_sdpa_tiled_lse_seg1_bf16_d_64 PIE_HEAD_DIM=64 PIE_TILED=1 PIE_LSE=1 PIE_TILE_ROWS=32 PIE_TARGET=spirv PIE_ONE_SEGMENT=1
// pie:instantiate attn_sdpa_tiled_lse_r16_seg1_bf16_d_64 PIE_HEAD_DIM=64 PIE_TILED=1 PIE_LSE=1 PIE_TILE_ROWS=16 PIE_TARGET=spirv PIE_ONE_SEGMENT=1
// pie:instantiate attn_sdpa_tiled_lse_r8_seg1_bf16_d_64 PIE_HEAD_DIM=64 PIE_TILED=1 PIE_LSE=1 PIE_TILE_ROWS=8 PIE_TARGET=spirv PIE_ONE_SEGMENT=1
// pie:instantiate attn_sdpa_tiled_l16_r16_seg1_bf16_d_64 PIE_HEAD_DIM=64 PIE_TILED=1 PIE_TILE_LANES=16 PIE_TILE_ROWS=16 PIE_TARGET=spirv PIE_ONE_SEGMENT=1
// pie:instantiate attn_sdpa_tiled_l16_r16_seg1_bf16_d_64 PIE_HEAD_DIM=64 PIE_TILED=1 PIE_TILE_LANES=16 PIE_TILE_ROWS=16 PIE_TIER=subgroup PIE_TARGET=spirv PIE_ONE_SEGMENT=1
// pie:instantiate attn_sdpa_tiled_lse_l16_r16_seg1_bf16_d_64 PIE_HEAD_DIM=64 PIE_TILED=1 PIE_LSE=1 PIE_TILE_LANES=16 PIE_TILE_ROWS=16 PIE_TARGET=spirv PIE_ONE_SEGMENT=1
// pie:instantiate attn_sdpa_tiled_lse_l16_r16_seg1_bf16_d_64 PIE_HEAD_DIM=64 PIE_TILED=1 PIE_LSE=1 PIE_TILE_LANES=16 PIE_TILE_ROWS=16 PIE_TIER=subgroup PIE_TARGET=spirv PIE_ONE_SEGMENT=1
// pie:instantiate attn_sdpa_tiled_seg1_bf16_d_128 PIE_HEAD_DIM=128 PIE_TILED=1 PIE_TILE_ROWS=32 PIE_TARGET=spirv PIE_ONE_SEGMENT=1
// pie:instantiate attn_sdpa_tiled_r16_seg1_bf16_d_128 PIE_HEAD_DIM=128 PIE_TILED=1 PIE_TILE_ROWS=16 PIE_TARGET=spirv PIE_ONE_SEGMENT=1
// pie:instantiate attn_sdpa_tiled_r8_seg1_bf16_d_128 PIE_HEAD_DIM=128 PIE_TILED=1 PIE_TILE_ROWS=8 PIE_TARGET=spirv PIE_ONE_SEGMENT=1
// pie:instantiate attn_sdpa_tiled_lse_seg1_bf16_d_128 PIE_HEAD_DIM=128 PIE_TILED=1 PIE_LSE=1 PIE_TILE_ROWS=32 PIE_TARGET=spirv PIE_ONE_SEGMENT=1
// pie:instantiate attn_sdpa_tiled_lse_r16_seg1_bf16_d_128 PIE_HEAD_DIM=128 PIE_TILED=1 PIE_LSE=1 PIE_TILE_ROWS=16 PIE_TARGET=spirv PIE_ONE_SEGMENT=1
// pie:instantiate attn_sdpa_tiled_lse_r8_seg1_bf16_d_128 PIE_HEAD_DIM=128 PIE_TILED=1 PIE_LSE=1 PIE_TILE_ROWS=8 PIE_TARGET=spirv PIE_ONE_SEGMENT=1
// pie:instantiate attn_sdpa_tiled_l16_r16_seg1_bf16_d_128 PIE_HEAD_DIM=128 PIE_TILED=1 PIE_TILE_LANES=16 PIE_TILE_ROWS=16 PIE_TARGET=spirv PIE_ONE_SEGMENT=1
// pie:instantiate attn_sdpa_tiled_l16_r16_seg1_bf16_d_128 PIE_HEAD_DIM=128 PIE_TILED=1 PIE_TILE_LANES=16 PIE_TILE_ROWS=16 PIE_TIER=subgroup PIE_TARGET=spirv PIE_ONE_SEGMENT=1
// pie:instantiate attn_sdpa_tiled_lse_l16_r16_seg1_bf16_d_128 PIE_HEAD_DIM=128 PIE_TILED=1 PIE_LSE=1 PIE_TILE_LANES=16 PIE_TILE_ROWS=16 PIE_TARGET=spirv PIE_ONE_SEGMENT=1
// pie:instantiate attn_sdpa_tiled_lse_l16_r16_seg1_bf16_d_128 PIE_HEAD_DIM=128 PIE_TILED=1 PIE_LSE=1 PIE_TILE_LANES=16 PIE_TILE_ROWS=16 PIE_TIER=subgroup PIE_TARGET=spirv PIE_ONE_SEGMENT=1
// pie:instantiate attn_sdpa_tiled_seg1_bf16_d_256 PIE_HEAD_DIM=256 PIE_TILED=1 PIE_TILE_ROWS=32 PIE_TARGET=spirv PIE_ONE_SEGMENT=1
// pie:instantiate attn_sdpa_tiled_r16_seg1_bf16_d_256 PIE_HEAD_DIM=256 PIE_TILED=1 PIE_TILE_ROWS=16 PIE_TARGET=spirv PIE_ONE_SEGMENT=1
// pie:instantiate attn_sdpa_tiled_r8_seg1_bf16_d_256 PIE_HEAD_DIM=256 PIE_TILED=1 PIE_TILE_ROWS=8 PIE_TARGET=spirv PIE_ONE_SEGMENT=1
// pie:instantiate attn_sdpa_tiled_lse_seg1_bf16_d_256 PIE_HEAD_DIM=256 PIE_TILED=1 PIE_LSE=1 PIE_TILE_ROWS=32 PIE_TARGET=spirv PIE_ONE_SEGMENT=1
// pie:instantiate attn_sdpa_tiled_lse_r16_seg1_bf16_d_256 PIE_HEAD_DIM=256 PIE_TILED=1 PIE_LSE=1 PIE_TILE_ROWS=16 PIE_TARGET=spirv PIE_ONE_SEGMENT=1
// pie:instantiate attn_sdpa_tiled_lse_r8_seg1_bf16_d_256 PIE_HEAD_DIM=256 PIE_TILED=1 PIE_LSE=1 PIE_TILE_ROWS=8 PIE_TARGET=spirv PIE_ONE_SEGMENT=1
// pie:instantiate attn_sdpa_tiled_l16_r16_seg1_bf16_d_256 PIE_HEAD_DIM=256 PIE_TILED=1 PIE_TILE_LANES=16 PIE_TILE_ROWS=16 PIE_TARGET=spirv PIE_ONE_SEGMENT=1
// pie:instantiate attn_sdpa_tiled_l16_r16_seg1_bf16_d_256 PIE_HEAD_DIM=256 PIE_TILED=1 PIE_TILE_LANES=16 PIE_TILE_ROWS=16 PIE_TIER=subgroup PIE_TARGET=spirv PIE_ONE_SEGMENT=1
// pie:instantiate attn_sdpa_tiled_lse_l16_r16_seg1_bf16_d_256 PIE_HEAD_DIM=256 PIE_TILED=1 PIE_LSE=1 PIE_TILE_LANES=16 PIE_TILE_ROWS=16 PIE_TARGET=spirv PIE_ONE_SEGMENT=1
// pie:instantiate attn_sdpa_tiled_lse_l16_r16_seg1_bf16_d_256 PIE_HEAD_DIM=256 PIE_TILED=1 PIE_LSE=1 PIE_TILE_LANES=16 PIE_TILE_ROWS=16 PIE_TIER=subgroup PIE_TARGET=spirv PIE_ONE_SEGMENT=1
// pie:instantiate attn_sdpa_tiled_seg1_bf16_d_512 PIE_HEAD_DIM=512 PIE_TILED=1 PIE_TILE_ROWS=32 PIE_TARGET=spirv PIE_ONE_SEGMENT=1
// pie:instantiate attn_sdpa_tiled_r16_seg1_bf16_d_512 PIE_HEAD_DIM=512 PIE_TILED=1 PIE_TILE_ROWS=16 PIE_TARGET=spirv PIE_ONE_SEGMENT=1
// pie:instantiate attn_sdpa_tiled_r8_seg1_bf16_d_512 PIE_HEAD_DIM=512 PIE_TILED=1 PIE_TILE_ROWS=8 PIE_TARGET=spirv PIE_ONE_SEGMENT=1
// pie:instantiate attn_sdpa_tiled_lse_seg1_bf16_d_512 PIE_HEAD_DIM=512 PIE_TILED=1 PIE_LSE=1 PIE_TILE_ROWS=32 PIE_TARGET=spirv PIE_ONE_SEGMENT=1
// pie:instantiate attn_sdpa_tiled_lse_r16_seg1_bf16_d_512 PIE_HEAD_DIM=512 PIE_TILED=1 PIE_LSE=1 PIE_TILE_ROWS=16 PIE_TARGET=spirv PIE_ONE_SEGMENT=1
// pie:instantiate attn_sdpa_tiled_lse_r8_seg1_bf16_d_512 PIE_HEAD_DIM=512 PIE_TILED=1 PIE_LSE=1 PIE_TILE_ROWS=8 PIE_TARGET=spirv PIE_ONE_SEGMENT=1
// pie:instantiate attn_sdpa_tiled_l16_r16_seg1_bf16_d_512 PIE_HEAD_DIM=512 PIE_TILED=1 PIE_TILE_LANES=16 PIE_TILE_ROWS=16 PIE_TARGET=spirv PIE_ONE_SEGMENT=1
// pie:instantiate attn_sdpa_tiled_l16_r16_seg1_bf16_d_512 PIE_HEAD_DIM=512 PIE_TILED=1 PIE_TILE_LANES=16 PIE_TILE_ROWS=16 PIE_TIER=subgroup PIE_TARGET=spirv PIE_ONE_SEGMENT=1
// pie:instantiate attn_sdpa_tiled_lse_l16_r16_seg1_bf16_d_512 PIE_HEAD_DIM=512 PIE_TILED=1 PIE_LSE=1 PIE_TILE_LANES=16 PIE_TILE_ROWS=16 PIE_TARGET=spirv PIE_ONE_SEGMENT=1
// pie:instantiate attn_sdpa_tiled_lse_l16_r16_seg1_bf16_d_512 PIE_HEAD_DIM=512 PIE_TILED=1 PIE_LSE=1 PIE_TILE_LANES=16 PIE_TILE_ROWS=16 PIE_TIER=subgroup PIE_TARGET=spirv PIE_ONE_SEGMENT=1
// pie:instantiate attn_sdpa_decode_ring_bf16_d_64 PIE_HEAD_DIM=64 PIE_TIER=subgroup PIE_TARGET=spirv PIE_RING=1
// pie:instantiate attn_sdpa_decode_ring_lanes_8_bf16_d_64 PIE_HEAD_DIM=64 PIE_TIER=subgroup PIE_TARGET=spirv PIE_MIN_LANES=8 PIE_RING=1
// pie:instantiate attn_sdpa_decode_ring_t128_bf16_d_64 PIE_HEAD_DIM=64 PIE_TIER=subgroup PIE_TARGET=spirv PIE_DECODE_THREADS=128 PIE_RING=1
// pie:instantiate attn_sdpa_decode_ring_t128_lanes_8_bf16_d_64 PIE_HEAD_DIM=64 PIE_TIER=subgroup PIE_TARGET=spirv PIE_DECODE_THREADS=128 PIE_MIN_LANES=8 PIE_RING=1
// pie:instantiate attn_sdpa_decode_ring_bf16_d_64 PIE_HEAD_DIM=64 PIE_RING=1
// pie:instantiate attn_sdpa_decode_ring_t128_bf16_d_64 PIE_HEAD_DIM=64 PIE_DECODE_THREADS=128 PIE_RING=1
// pie:instantiate attn_sdpa_decode_lse_ring_bf16_d_64 PIE_HEAD_DIM=64 PIE_LSE=1 PIE_TIER=subgroup PIE_TARGET=spirv PIE_RING=1
// pie:instantiate attn_sdpa_decode_lse_ring_lanes_8_bf16_d_64 PIE_HEAD_DIM=64 PIE_LSE=1 PIE_TIER=subgroup PIE_TARGET=spirv PIE_MIN_LANES=8 PIE_RING=1
// pie:instantiate attn_sdpa_decode_lse_ring_t128_bf16_d_64 PIE_HEAD_DIM=64 PIE_LSE=1 PIE_TIER=subgroup PIE_TARGET=spirv PIE_DECODE_THREADS=128 PIE_RING=1
// pie:instantiate attn_sdpa_decode_lse_ring_t128_lanes_8_bf16_d_64 PIE_HEAD_DIM=64 PIE_LSE=1 PIE_TIER=subgroup PIE_TARGET=spirv PIE_DECODE_THREADS=128 PIE_MIN_LANES=8 PIE_RING=1
// pie:instantiate attn_sdpa_decode_lse_ring_bf16_d_64 PIE_HEAD_DIM=64 PIE_LSE=1 PIE_RING=1
// pie:instantiate attn_sdpa_decode_lse_ring_t128_bf16_d_64 PIE_HEAD_DIM=64 PIE_LSE=1 PIE_DECODE_THREADS=128 PIE_RING=1
// pie:instantiate attn_sdpa_split_ring_bf16_d_64 PIE_HEAD_DIM=64 PIE_SPLIT=1 PIE_TIER=subgroup PIE_TARGET=spirv PIE_RING=1
// pie:instantiate attn_sdpa_split_ring_lanes_8_bf16_d_64 PIE_HEAD_DIM=64 PIE_SPLIT=1 PIE_TIER=subgroup PIE_TARGET=spirv PIE_MIN_LANES=8 PIE_RING=1
// pie:instantiate attn_sdpa_split_ring_t128_bf16_d_64 PIE_HEAD_DIM=64 PIE_SPLIT=1 PIE_TIER=subgroup PIE_TARGET=spirv PIE_DECODE_THREADS=128 PIE_RING=1
// pie:instantiate attn_sdpa_split_ring_t128_lanes_8_bf16_d_64 PIE_HEAD_DIM=64 PIE_SPLIT=1 PIE_TIER=subgroup PIE_TARGET=spirv PIE_DECODE_THREADS=128 PIE_MIN_LANES=8 PIE_RING=1
// pie:instantiate attn_sdpa_split_ring_bf16_d_64 PIE_HEAD_DIM=64 PIE_SPLIT=1 PIE_RING=1
// pie:instantiate attn_sdpa_split_ring_t128_bf16_d_64 PIE_HEAD_DIM=64 PIE_SPLIT=1 PIE_DECODE_THREADS=128 PIE_RING=1
// pie:instantiate attn_sdpa_split_lse_ring_bf16_d_64 PIE_HEAD_DIM=64 PIE_SPLIT=1 PIE_SPLIT_LSE=1 PIE_TIER=subgroup PIE_TARGET=spirv PIE_RING=1
// pie:instantiate attn_sdpa_split_lse_ring_lanes_8_bf16_d_64 PIE_HEAD_DIM=64 PIE_SPLIT=1 PIE_SPLIT_LSE=1 PIE_TIER=subgroup PIE_TARGET=spirv PIE_MIN_LANES=8 PIE_RING=1
// pie:instantiate attn_sdpa_split_lse_ring_t128_bf16_d_64 PIE_HEAD_DIM=64 PIE_SPLIT=1 PIE_SPLIT_LSE=1 PIE_TIER=subgroup PIE_TARGET=spirv PIE_DECODE_THREADS=128 PIE_RING=1
// pie:instantiate attn_sdpa_split_lse_ring_t128_lanes_8_bf16_d_64 PIE_HEAD_DIM=64 PIE_SPLIT=1 PIE_SPLIT_LSE=1 PIE_TIER=subgroup PIE_TARGET=spirv PIE_DECODE_THREADS=128 PIE_MIN_LANES=8 PIE_RING=1
// pie:instantiate attn_sdpa_split_lse_ring_bf16_d_64 PIE_HEAD_DIM=64 PIE_SPLIT=1 PIE_SPLIT_LSE=1 PIE_RING=1
// pie:instantiate attn_sdpa_split_lse_ring_t128_bf16_d_64 PIE_HEAD_DIM=64 PIE_SPLIT=1 PIE_SPLIT_LSE=1 PIE_DECODE_THREADS=128 PIE_RING=1
// pie:instantiate attn_sdpa_tiled_ring_bf16_d_64 PIE_HEAD_DIM=64 PIE_TILED=1 PIE_TILE_ROWS=32 PIE_RING=1
// pie:instantiate attn_sdpa_tiled_ring_r16_bf16_d_64 PIE_HEAD_DIM=64 PIE_TILED=1 PIE_TILE_ROWS=16 PIE_RING=1
// pie:instantiate attn_sdpa_tiled_ring_r8_bf16_d_64 PIE_HEAD_DIM=64 PIE_TILED=1 PIE_TILE_ROWS=8 PIE_RING=1
// pie:instantiate attn_sdpa_tiled_lse_ring_bf16_d_64 PIE_HEAD_DIM=64 PIE_TILED=1 PIE_LSE=1 PIE_TILE_ROWS=32 PIE_RING=1
// pie:instantiate attn_sdpa_tiled_lse_ring_r16_bf16_d_64 PIE_HEAD_DIM=64 PIE_TILED=1 PIE_LSE=1 PIE_TILE_ROWS=16 PIE_RING=1
// pie:instantiate attn_sdpa_tiled_lse_ring_r8_bf16_d_64 PIE_HEAD_DIM=64 PIE_TILED=1 PIE_LSE=1 PIE_TILE_ROWS=8 PIE_RING=1
// pie:instantiate attn_sdpa_tiled_ring_l16_r16_bf16_d_64 PIE_HEAD_DIM=64 PIE_TILED=1 PIE_TILE_LANES=16 PIE_TILE_ROWS=16 PIE_RING=1
// pie:instantiate attn_sdpa_tiled_ring_l16_r16_bf16_d_64 PIE_HEAD_DIM=64 PIE_TILED=1 PIE_TILE_LANES=16 PIE_TILE_ROWS=16 PIE_TIER=subgroup PIE_RING=1
// pie:instantiate attn_sdpa_tiled_lse_ring_l16_r16_bf16_d_64 PIE_HEAD_DIM=64 PIE_TILED=1 PIE_LSE=1 PIE_TILE_LANES=16 PIE_TILE_ROWS=16 PIE_RING=1
// pie:instantiate attn_sdpa_tiled_lse_ring_l16_r16_bf16_d_64 PIE_HEAD_DIM=64 PIE_TILED=1 PIE_LSE=1 PIE_TILE_LANES=16 PIE_TILE_ROWS=16 PIE_TIER=subgroup PIE_RING=1
// pie:instantiate attn_sdpa_decode_ring_bf16_d_128 PIE_HEAD_DIM=128 PIE_TIER=subgroup PIE_TARGET=spirv PIE_RING=1
// pie:instantiate attn_sdpa_decode_ring_lanes_8_bf16_d_128 PIE_HEAD_DIM=128 PIE_TIER=subgroup PIE_TARGET=spirv PIE_MIN_LANES=8 PIE_RING=1
// pie:instantiate attn_sdpa_decode_ring_t128_bf16_d_128 PIE_HEAD_DIM=128 PIE_TIER=subgroup PIE_TARGET=spirv PIE_DECODE_THREADS=128 PIE_RING=1
// pie:instantiate attn_sdpa_decode_ring_t128_lanes_8_bf16_d_128 PIE_HEAD_DIM=128 PIE_TIER=subgroup PIE_TARGET=spirv PIE_DECODE_THREADS=128 PIE_MIN_LANES=8 PIE_RING=1
// pie:instantiate attn_sdpa_decode_ring_bf16_d_128 PIE_HEAD_DIM=128 PIE_RING=1
// pie:instantiate attn_sdpa_decode_ring_t128_bf16_d_128 PIE_HEAD_DIM=128 PIE_DECODE_THREADS=128 PIE_RING=1
// pie:instantiate attn_sdpa_decode_lse_ring_bf16_d_128 PIE_HEAD_DIM=128 PIE_LSE=1 PIE_TIER=subgroup PIE_TARGET=spirv PIE_RING=1
// pie:instantiate attn_sdpa_decode_lse_ring_lanes_8_bf16_d_128 PIE_HEAD_DIM=128 PIE_LSE=1 PIE_TIER=subgroup PIE_TARGET=spirv PIE_MIN_LANES=8 PIE_RING=1
// pie:instantiate attn_sdpa_decode_lse_ring_t128_bf16_d_128 PIE_HEAD_DIM=128 PIE_LSE=1 PIE_TIER=subgroup PIE_TARGET=spirv PIE_DECODE_THREADS=128 PIE_RING=1
// pie:instantiate attn_sdpa_decode_lse_ring_t128_lanes_8_bf16_d_128 PIE_HEAD_DIM=128 PIE_LSE=1 PIE_TIER=subgroup PIE_TARGET=spirv PIE_DECODE_THREADS=128 PIE_MIN_LANES=8 PIE_RING=1
// pie:instantiate attn_sdpa_decode_lse_ring_bf16_d_128 PIE_HEAD_DIM=128 PIE_LSE=1 PIE_RING=1
// pie:instantiate attn_sdpa_decode_lse_ring_t128_bf16_d_128 PIE_HEAD_DIM=128 PIE_LSE=1 PIE_DECODE_THREADS=128 PIE_RING=1
// pie:instantiate attn_sdpa_split_ring_bf16_d_128 PIE_HEAD_DIM=128 PIE_SPLIT=1 PIE_TIER=subgroup PIE_TARGET=spirv PIE_RING=1
// pie:instantiate attn_sdpa_split_ring_lanes_8_bf16_d_128 PIE_HEAD_DIM=128 PIE_SPLIT=1 PIE_TIER=subgroup PIE_TARGET=spirv PIE_MIN_LANES=8 PIE_RING=1
// pie:instantiate attn_sdpa_split_ring_t128_bf16_d_128 PIE_HEAD_DIM=128 PIE_SPLIT=1 PIE_TIER=subgroup PIE_TARGET=spirv PIE_DECODE_THREADS=128 PIE_RING=1
// pie:instantiate attn_sdpa_split_ring_t128_lanes_8_bf16_d_128 PIE_HEAD_DIM=128 PIE_SPLIT=1 PIE_TIER=subgroup PIE_TARGET=spirv PIE_DECODE_THREADS=128 PIE_MIN_LANES=8 PIE_RING=1
// pie:instantiate attn_sdpa_split_ring_bf16_d_128 PIE_HEAD_DIM=128 PIE_SPLIT=1 PIE_RING=1
// pie:instantiate attn_sdpa_split_ring_t128_bf16_d_128 PIE_HEAD_DIM=128 PIE_SPLIT=1 PIE_DECODE_THREADS=128 PIE_RING=1
// pie:instantiate attn_sdpa_split_lse_ring_bf16_d_128 PIE_HEAD_DIM=128 PIE_SPLIT=1 PIE_SPLIT_LSE=1 PIE_TIER=subgroup PIE_TARGET=spirv PIE_RING=1
// pie:instantiate attn_sdpa_split_lse_ring_lanes_8_bf16_d_128 PIE_HEAD_DIM=128 PIE_SPLIT=1 PIE_SPLIT_LSE=1 PIE_TIER=subgroup PIE_TARGET=spirv PIE_MIN_LANES=8 PIE_RING=1
// pie:instantiate attn_sdpa_split_lse_ring_t128_bf16_d_128 PIE_HEAD_DIM=128 PIE_SPLIT=1 PIE_SPLIT_LSE=1 PIE_TIER=subgroup PIE_TARGET=spirv PIE_DECODE_THREADS=128 PIE_RING=1
// pie:instantiate attn_sdpa_split_lse_ring_t128_lanes_8_bf16_d_128 PIE_HEAD_DIM=128 PIE_SPLIT=1 PIE_SPLIT_LSE=1 PIE_TIER=subgroup PIE_TARGET=spirv PIE_DECODE_THREADS=128 PIE_MIN_LANES=8 PIE_RING=1
// pie:instantiate attn_sdpa_split_lse_ring_bf16_d_128 PIE_HEAD_DIM=128 PIE_SPLIT=1 PIE_SPLIT_LSE=1 PIE_RING=1
// pie:instantiate attn_sdpa_split_lse_ring_t128_bf16_d_128 PIE_HEAD_DIM=128 PIE_SPLIT=1 PIE_SPLIT_LSE=1 PIE_DECODE_THREADS=128 PIE_RING=1
// pie:instantiate attn_sdpa_tiled_ring_bf16_d_128 PIE_HEAD_DIM=128 PIE_TILED=1 PIE_TILE_ROWS=32 PIE_RING=1
// pie:instantiate attn_sdpa_tiled_ring_r16_bf16_d_128 PIE_HEAD_DIM=128 PIE_TILED=1 PIE_TILE_ROWS=16 PIE_RING=1
// pie:instantiate attn_sdpa_tiled_ring_r8_bf16_d_128 PIE_HEAD_DIM=128 PIE_TILED=1 PIE_TILE_ROWS=8 PIE_RING=1
// pie:instantiate attn_sdpa_tiled_lse_ring_bf16_d_128 PIE_HEAD_DIM=128 PIE_TILED=1 PIE_LSE=1 PIE_TILE_ROWS=32 PIE_RING=1
// pie:instantiate attn_sdpa_tiled_lse_ring_r16_bf16_d_128 PIE_HEAD_DIM=128 PIE_TILED=1 PIE_LSE=1 PIE_TILE_ROWS=16 PIE_RING=1
// pie:instantiate attn_sdpa_tiled_lse_ring_r8_bf16_d_128 PIE_HEAD_DIM=128 PIE_TILED=1 PIE_LSE=1 PIE_TILE_ROWS=8 PIE_RING=1
// pie:instantiate attn_sdpa_tiled_ring_l16_r16_bf16_d_128 PIE_HEAD_DIM=128 PIE_TILED=1 PIE_TILE_LANES=16 PIE_TILE_ROWS=16 PIE_RING=1
// pie:instantiate attn_sdpa_tiled_ring_l16_r16_bf16_d_128 PIE_HEAD_DIM=128 PIE_TILED=1 PIE_TILE_LANES=16 PIE_TILE_ROWS=16 PIE_TIER=subgroup PIE_RING=1
// pie:instantiate attn_sdpa_tiled_lse_ring_l16_r16_bf16_d_128 PIE_HEAD_DIM=128 PIE_TILED=1 PIE_LSE=1 PIE_TILE_LANES=16 PIE_TILE_ROWS=16 PIE_RING=1
// pie:instantiate attn_sdpa_tiled_lse_ring_l16_r16_bf16_d_128 PIE_HEAD_DIM=128 PIE_TILED=1 PIE_LSE=1 PIE_TILE_LANES=16 PIE_TILE_ROWS=16 PIE_TIER=subgroup PIE_RING=1
// pie:instantiate attn_sdpa_decode_ring_bf16_d_256 PIE_HEAD_DIM=256 PIE_TIER=subgroup PIE_TARGET=spirv PIE_RING=1
// pie:instantiate attn_sdpa_decode_ring_lanes_8_bf16_d_256 PIE_HEAD_DIM=256 PIE_TIER=subgroup PIE_TARGET=spirv PIE_MIN_LANES=8 PIE_RING=1
// pie:instantiate attn_sdpa_decode_ring_t128_bf16_d_256 PIE_HEAD_DIM=256 PIE_TIER=subgroup PIE_TARGET=spirv PIE_DECODE_THREADS=128 PIE_RING=1
// pie:instantiate attn_sdpa_decode_ring_t128_lanes_8_bf16_d_256 PIE_HEAD_DIM=256 PIE_TIER=subgroup PIE_TARGET=spirv PIE_DECODE_THREADS=128 PIE_MIN_LANES=8 PIE_RING=1
// pie:instantiate attn_sdpa_decode_ring_bf16_d_256 PIE_HEAD_DIM=256 PIE_RING=1
// pie:instantiate attn_sdpa_decode_ring_t128_bf16_d_256 PIE_HEAD_DIM=256 PIE_DECODE_THREADS=128 PIE_RING=1
// pie:instantiate attn_sdpa_decode_lse_ring_bf16_d_256 PIE_HEAD_DIM=256 PIE_LSE=1 PIE_TIER=subgroup PIE_TARGET=spirv PIE_RING=1
// pie:instantiate attn_sdpa_decode_lse_ring_lanes_8_bf16_d_256 PIE_HEAD_DIM=256 PIE_LSE=1 PIE_TIER=subgroup PIE_TARGET=spirv PIE_MIN_LANES=8 PIE_RING=1
// pie:instantiate attn_sdpa_decode_lse_ring_t128_bf16_d_256 PIE_HEAD_DIM=256 PIE_LSE=1 PIE_TIER=subgroup PIE_TARGET=spirv PIE_DECODE_THREADS=128 PIE_RING=1
// pie:instantiate attn_sdpa_decode_lse_ring_t128_lanes_8_bf16_d_256 PIE_HEAD_DIM=256 PIE_LSE=1 PIE_TIER=subgroup PIE_TARGET=spirv PIE_DECODE_THREADS=128 PIE_MIN_LANES=8 PIE_RING=1
// pie:instantiate attn_sdpa_decode_lse_ring_bf16_d_256 PIE_HEAD_DIM=256 PIE_LSE=1 PIE_RING=1
// pie:instantiate attn_sdpa_decode_lse_ring_t128_bf16_d_256 PIE_HEAD_DIM=256 PIE_LSE=1 PIE_DECODE_THREADS=128 PIE_RING=1
// pie:instantiate attn_sdpa_split_ring_bf16_d_256 PIE_HEAD_DIM=256 PIE_SPLIT=1 PIE_TIER=subgroup PIE_TARGET=spirv PIE_RING=1
// pie:instantiate attn_sdpa_split_ring_lanes_8_bf16_d_256 PIE_HEAD_DIM=256 PIE_SPLIT=1 PIE_TIER=subgroup PIE_TARGET=spirv PIE_MIN_LANES=8 PIE_RING=1
// pie:instantiate attn_sdpa_split_ring_t128_bf16_d_256 PIE_HEAD_DIM=256 PIE_SPLIT=1 PIE_TIER=subgroup PIE_TARGET=spirv PIE_DECODE_THREADS=128 PIE_RING=1
// pie:instantiate attn_sdpa_split_ring_t128_lanes_8_bf16_d_256 PIE_HEAD_DIM=256 PIE_SPLIT=1 PIE_TIER=subgroup PIE_TARGET=spirv PIE_DECODE_THREADS=128 PIE_MIN_LANES=8 PIE_RING=1
// pie:instantiate attn_sdpa_split_ring_bf16_d_256 PIE_HEAD_DIM=256 PIE_SPLIT=1 PIE_RING=1
// pie:instantiate attn_sdpa_split_ring_t128_bf16_d_256 PIE_HEAD_DIM=256 PIE_SPLIT=1 PIE_DECODE_THREADS=128 PIE_RING=1
// pie:instantiate attn_sdpa_split_lse_ring_bf16_d_256 PIE_HEAD_DIM=256 PIE_SPLIT=1 PIE_SPLIT_LSE=1 PIE_TIER=subgroup PIE_TARGET=spirv PIE_RING=1
// pie:instantiate attn_sdpa_split_lse_ring_lanes_8_bf16_d_256 PIE_HEAD_DIM=256 PIE_SPLIT=1 PIE_SPLIT_LSE=1 PIE_TIER=subgroup PIE_TARGET=spirv PIE_MIN_LANES=8 PIE_RING=1
// pie:instantiate attn_sdpa_split_lse_ring_t128_bf16_d_256 PIE_HEAD_DIM=256 PIE_SPLIT=1 PIE_SPLIT_LSE=1 PIE_TIER=subgroup PIE_TARGET=spirv PIE_DECODE_THREADS=128 PIE_RING=1
// pie:instantiate attn_sdpa_split_lse_ring_t128_lanes_8_bf16_d_256 PIE_HEAD_DIM=256 PIE_SPLIT=1 PIE_SPLIT_LSE=1 PIE_TIER=subgroup PIE_TARGET=spirv PIE_DECODE_THREADS=128 PIE_MIN_LANES=8 PIE_RING=1
// pie:instantiate attn_sdpa_split_lse_ring_bf16_d_256 PIE_HEAD_DIM=256 PIE_SPLIT=1 PIE_SPLIT_LSE=1 PIE_RING=1
// pie:instantiate attn_sdpa_split_lse_ring_t128_bf16_d_256 PIE_HEAD_DIM=256 PIE_SPLIT=1 PIE_SPLIT_LSE=1 PIE_DECODE_THREADS=128 PIE_RING=1
// pie:instantiate attn_sdpa_tiled_ring_bf16_d_256 PIE_HEAD_DIM=256 PIE_TILED=1 PIE_TILE_ROWS=32 PIE_RING=1
// pie:instantiate attn_sdpa_tiled_ring_r16_bf16_d_256 PIE_HEAD_DIM=256 PIE_TILED=1 PIE_TILE_ROWS=16 PIE_RING=1
// pie:instantiate attn_sdpa_tiled_ring_r8_bf16_d_256 PIE_HEAD_DIM=256 PIE_TILED=1 PIE_TILE_ROWS=8 PIE_RING=1
// pie:instantiate attn_sdpa_tiled_lse_ring_bf16_d_256 PIE_HEAD_DIM=256 PIE_TILED=1 PIE_LSE=1 PIE_TILE_ROWS=32 PIE_RING=1
// pie:instantiate attn_sdpa_tiled_lse_ring_r16_bf16_d_256 PIE_HEAD_DIM=256 PIE_TILED=1 PIE_LSE=1 PIE_TILE_ROWS=16 PIE_RING=1
// pie:instantiate attn_sdpa_tiled_lse_ring_r8_bf16_d_256 PIE_HEAD_DIM=256 PIE_TILED=1 PIE_LSE=1 PIE_TILE_ROWS=8 PIE_RING=1
// pie:instantiate attn_sdpa_tiled_ring_l16_r16_bf16_d_256 PIE_HEAD_DIM=256 PIE_TILED=1 PIE_TILE_LANES=16 PIE_TILE_ROWS=16 PIE_RING=1
// pie:instantiate attn_sdpa_tiled_ring_l16_r16_bf16_d_256 PIE_HEAD_DIM=256 PIE_TILED=1 PIE_TILE_LANES=16 PIE_TILE_ROWS=16 PIE_TIER=subgroup PIE_RING=1
// pie:instantiate attn_sdpa_tiled_lse_ring_l16_r16_bf16_d_256 PIE_HEAD_DIM=256 PIE_TILED=1 PIE_LSE=1 PIE_TILE_LANES=16 PIE_TILE_ROWS=16 PIE_RING=1
// pie:instantiate attn_sdpa_tiled_lse_ring_l16_r16_bf16_d_256 PIE_HEAD_DIM=256 PIE_TILED=1 PIE_LSE=1 PIE_TILE_LANES=16 PIE_TILE_ROWS=16 PIE_TIER=subgroup PIE_RING=1
// pie:instantiate attn_sdpa_decode_ring_bf16_d_512 PIE_HEAD_DIM=512 PIE_TIER=subgroup PIE_TARGET=spirv PIE_RING=1
// pie:instantiate attn_sdpa_decode_ring_lanes_8_bf16_d_512 PIE_HEAD_DIM=512 PIE_TIER=subgroup PIE_TARGET=spirv PIE_MIN_LANES=8 PIE_RING=1
// pie:instantiate attn_sdpa_decode_ring_t128_bf16_d_512 PIE_HEAD_DIM=512 PIE_TIER=subgroup PIE_TARGET=spirv PIE_DECODE_THREADS=128 PIE_RING=1
// pie:instantiate attn_sdpa_decode_ring_t128_lanes_8_bf16_d_512 PIE_HEAD_DIM=512 PIE_TIER=subgroup PIE_TARGET=spirv PIE_DECODE_THREADS=128 PIE_MIN_LANES=8 PIE_RING=1
// pie:instantiate attn_sdpa_decode_ring_bf16_d_512 PIE_HEAD_DIM=512 PIE_RING=1
// pie:instantiate attn_sdpa_decode_ring_t128_bf16_d_512 PIE_HEAD_DIM=512 PIE_DECODE_THREADS=128 PIE_RING=1
// pie:instantiate attn_sdpa_decode_lse_ring_bf16_d_512 PIE_HEAD_DIM=512 PIE_LSE=1 PIE_TIER=subgroup PIE_TARGET=spirv PIE_RING=1
// pie:instantiate attn_sdpa_decode_lse_ring_lanes_8_bf16_d_512 PIE_HEAD_DIM=512 PIE_LSE=1 PIE_TIER=subgroup PIE_TARGET=spirv PIE_MIN_LANES=8 PIE_RING=1
// pie:instantiate attn_sdpa_decode_lse_ring_t128_bf16_d_512 PIE_HEAD_DIM=512 PIE_LSE=1 PIE_TIER=subgroup PIE_TARGET=spirv PIE_DECODE_THREADS=128 PIE_RING=1
// pie:instantiate attn_sdpa_decode_lse_ring_t128_lanes_8_bf16_d_512 PIE_HEAD_DIM=512 PIE_LSE=1 PIE_TIER=subgroup PIE_TARGET=spirv PIE_DECODE_THREADS=128 PIE_MIN_LANES=8 PIE_RING=1
// pie:instantiate attn_sdpa_decode_lse_ring_bf16_d_512 PIE_HEAD_DIM=512 PIE_LSE=1 PIE_RING=1
// pie:instantiate attn_sdpa_decode_lse_ring_t128_bf16_d_512 PIE_HEAD_DIM=512 PIE_LSE=1 PIE_DECODE_THREADS=128 PIE_RING=1
// pie:instantiate attn_sdpa_split_ring_bf16_d_512 PIE_HEAD_DIM=512 PIE_SPLIT=1 PIE_TIER=subgroup PIE_TARGET=spirv PIE_RING=1
// pie:instantiate attn_sdpa_split_ring_lanes_8_bf16_d_512 PIE_HEAD_DIM=512 PIE_SPLIT=1 PIE_TIER=subgroup PIE_TARGET=spirv PIE_MIN_LANES=8 PIE_RING=1
// pie:instantiate attn_sdpa_split_ring_t128_bf16_d_512 PIE_HEAD_DIM=512 PIE_SPLIT=1 PIE_TIER=subgroup PIE_TARGET=spirv PIE_DECODE_THREADS=128 PIE_RING=1
// pie:instantiate attn_sdpa_split_ring_t128_lanes_8_bf16_d_512 PIE_HEAD_DIM=512 PIE_SPLIT=1 PIE_TIER=subgroup PIE_TARGET=spirv PIE_DECODE_THREADS=128 PIE_MIN_LANES=8 PIE_RING=1
// pie:instantiate attn_sdpa_split_ring_bf16_d_512 PIE_HEAD_DIM=512 PIE_SPLIT=1 PIE_RING=1
// pie:instantiate attn_sdpa_split_ring_t128_bf16_d_512 PIE_HEAD_DIM=512 PIE_SPLIT=1 PIE_DECODE_THREADS=128 PIE_RING=1
// pie:instantiate attn_sdpa_split_lse_ring_bf16_d_512 PIE_HEAD_DIM=512 PIE_SPLIT=1 PIE_SPLIT_LSE=1 PIE_TIER=subgroup PIE_TARGET=spirv PIE_RING=1
// pie:instantiate attn_sdpa_split_lse_ring_lanes_8_bf16_d_512 PIE_HEAD_DIM=512 PIE_SPLIT=1 PIE_SPLIT_LSE=1 PIE_TIER=subgroup PIE_TARGET=spirv PIE_MIN_LANES=8 PIE_RING=1
// pie:instantiate attn_sdpa_split_lse_ring_t128_bf16_d_512 PIE_HEAD_DIM=512 PIE_SPLIT=1 PIE_SPLIT_LSE=1 PIE_TIER=subgroup PIE_TARGET=spirv PIE_DECODE_THREADS=128 PIE_RING=1
// pie:instantiate attn_sdpa_split_lse_ring_t128_lanes_8_bf16_d_512 PIE_HEAD_DIM=512 PIE_SPLIT=1 PIE_SPLIT_LSE=1 PIE_TIER=subgroup PIE_TARGET=spirv PIE_DECODE_THREADS=128 PIE_MIN_LANES=8 PIE_RING=1
// pie:instantiate attn_sdpa_split_lse_ring_bf16_d_512 PIE_HEAD_DIM=512 PIE_SPLIT=1 PIE_SPLIT_LSE=1 PIE_RING=1
// pie:instantiate attn_sdpa_split_lse_ring_t128_bf16_d_512 PIE_HEAD_DIM=512 PIE_SPLIT=1 PIE_SPLIT_LSE=1 PIE_DECODE_THREADS=128 PIE_RING=1
// pie:instantiate attn_sdpa_tiled_ring_bf16_d_512 PIE_HEAD_DIM=512 PIE_TILED=1 PIE_TILE_ROWS=32 PIE_RING=1
// pie:instantiate attn_sdpa_tiled_ring_r16_bf16_d_512 PIE_HEAD_DIM=512 PIE_TILED=1 PIE_TILE_ROWS=16 PIE_RING=1
// pie:instantiate attn_sdpa_tiled_ring_r8_bf16_d_512 PIE_HEAD_DIM=512 PIE_TILED=1 PIE_TILE_ROWS=8 PIE_RING=1
// pie:instantiate attn_sdpa_tiled_lse_ring_bf16_d_512 PIE_HEAD_DIM=512 PIE_TILED=1 PIE_LSE=1 PIE_TILE_ROWS=32 PIE_RING=1
// pie:instantiate attn_sdpa_tiled_lse_ring_r16_bf16_d_512 PIE_HEAD_DIM=512 PIE_TILED=1 PIE_LSE=1 PIE_TILE_ROWS=16 PIE_RING=1
// pie:instantiate attn_sdpa_tiled_lse_ring_r8_bf16_d_512 PIE_HEAD_DIM=512 PIE_TILED=1 PIE_LSE=1 PIE_TILE_ROWS=8 PIE_RING=1
// pie:instantiate attn_sdpa_tiled_ring_l16_r16_bf16_d_512 PIE_HEAD_DIM=512 PIE_TILED=1 PIE_TILE_LANES=16 PIE_TILE_ROWS=16 PIE_RING=1
// pie:instantiate attn_sdpa_tiled_ring_l16_r16_bf16_d_512 PIE_HEAD_DIM=512 PIE_TILED=1 PIE_TILE_LANES=16 PIE_TILE_ROWS=16 PIE_TIER=subgroup PIE_RING=1
// pie:instantiate attn_sdpa_tiled_lse_ring_l16_r16_bf16_d_512 PIE_HEAD_DIM=512 PIE_TILED=1 PIE_LSE=1 PIE_TILE_LANES=16 PIE_TILE_ROWS=16 PIE_RING=1
// pie:instantiate attn_sdpa_tiled_lse_ring_l16_r16_bf16_d_512 PIE_HEAD_DIM=512 PIE_TILED=1 PIE_LSE=1 PIE_TILE_LANES=16 PIE_TILE_ROWS=16 PIE_TIER=subgroup PIE_RING=1
// pie:instantiate attn_sdpa_tiled_ring_seg1_bf16_d_64 PIE_HEAD_DIM=64 PIE_TILED=1 PIE_TILE_ROWS=32 PIE_TARGET=spirv PIE_ONE_SEGMENT=1 PIE_RING=1
// pie:instantiate attn_sdpa_tiled_ring_r16_seg1_bf16_d_64 PIE_HEAD_DIM=64 PIE_TILED=1 PIE_TILE_ROWS=16 PIE_TARGET=spirv PIE_ONE_SEGMENT=1 PIE_RING=1
// pie:instantiate attn_sdpa_tiled_ring_r8_seg1_bf16_d_64 PIE_HEAD_DIM=64 PIE_TILED=1 PIE_TILE_ROWS=8 PIE_TARGET=spirv PIE_ONE_SEGMENT=1 PIE_RING=1
// pie:instantiate attn_sdpa_tiled_lse_ring_seg1_bf16_d_64 PIE_HEAD_DIM=64 PIE_TILED=1 PIE_LSE=1 PIE_TILE_ROWS=32 PIE_TARGET=spirv PIE_ONE_SEGMENT=1 PIE_RING=1
// pie:instantiate attn_sdpa_tiled_lse_ring_r16_seg1_bf16_d_64 PIE_HEAD_DIM=64 PIE_TILED=1 PIE_LSE=1 PIE_TILE_ROWS=16 PIE_TARGET=spirv PIE_ONE_SEGMENT=1 PIE_RING=1
// pie:instantiate attn_sdpa_tiled_lse_ring_r8_seg1_bf16_d_64 PIE_HEAD_DIM=64 PIE_TILED=1 PIE_LSE=1 PIE_TILE_ROWS=8 PIE_TARGET=spirv PIE_ONE_SEGMENT=1 PIE_RING=1
// pie:instantiate attn_sdpa_tiled_ring_l16_r16_seg1_bf16_d_64 PIE_HEAD_DIM=64 PIE_TILED=1 PIE_TILE_LANES=16 PIE_TILE_ROWS=16 PIE_TARGET=spirv PIE_ONE_SEGMENT=1 PIE_RING=1
// pie:instantiate attn_sdpa_tiled_ring_l16_r16_seg1_bf16_d_64 PIE_HEAD_DIM=64 PIE_TILED=1 PIE_TILE_LANES=16 PIE_TILE_ROWS=16 PIE_TIER=subgroup PIE_TARGET=spirv PIE_ONE_SEGMENT=1 PIE_RING=1
// pie:instantiate attn_sdpa_tiled_lse_ring_l16_r16_seg1_bf16_d_64 PIE_HEAD_DIM=64 PIE_TILED=1 PIE_LSE=1 PIE_TILE_LANES=16 PIE_TILE_ROWS=16 PIE_TARGET=spirv PIE_ONE_SEGMENT=1 PIE_RING=1
// pie:instantiate attn_sdpa_tiled_lse_ring_l16_r16_seg1_bf16_d_64 PIE_HEAD_DIM=64 PIE_TILED=1 PIE_LSE=1 PIE_TILE_LANES=16 PIE_TILE_ROWS=16 PIE_TIER=subgroup PIE_TARGET=spirv PIE_ONE_SEGMENT=1 PIE_RING=1
// pie:instantiate attn_sdpa_tiled_ring_seg1_bf16_d_128 PIE_HEAD_DIM=128 PIE_TILED=1 PIE_TILE_ROWS=32 PIE_TARGET=spirv PIE_ONE_SEGMENT=1 PIE_RING=1
// pie:instantiate attn_sdpa_tiled_ring_r16_seg1_bf16_d_128 PIE_HEAD_DIM=128 PIE_TILED=1 PIE_TILE_ROWS=16 PIE_TARGET=spirv PIE_ONE_SEGMENT=1 PIE_RING=1
// pie:instantiate attn_sdpa_tiled_ring_r8_seg1_bf16_d_128 PIE_HEAD_DIM=128 PIE_TILED=1 PIE_TILE_ROWS=8 PIE_TARGET=spirv PIE_ONE_SEGMENT=1 PIE_RING=1
// pie:instantiate attn_sdpa_tiled_lse_ring_seg1_bf16_d_128 PIE_HEAD_DIM=128 PIE_TILED=1 PIE_LSE=1 PIE_TILE_ROWS=32 PIE_TARGET=spirv PIE_ONE_SEGMENT=1 PIE_RING=1
// pie:instantiate attn_sdpa_tiled_lse_ring_r16_seg1_bf16_d_128 PIE_HEAD_DIM=128 PIE_TILED=1 PIE_LSE=1 PIE_TILE_ROWS=16 PIE_TARGET=spirv PIE_ONE_SEGMENT=1 PIE_RING=1
// pie:instantiate attn_sdpa_tiled_lse_ring_r8_seg1_bf16_d_128 PIE_HEAD_DIM=128 PIE_TILED=1 PIE_LSE=1 PIE_TILE_ROWS=8 PIE_TARGET=spirv PIE_ONE_SEGMENT=1 PIE_RING=1
// pie:instantiate attn_sdpa_tiled_ring_l16_r16_seg1_bf16_d_128 PIE_HEAD_DIM=128 PIE_TILED=1 PIE_TILE_LANES=16 PIE_TILE_ROWS=16 PIE_TARGET=spirv PIE_ONE_SEGMENT=1 PIE_RING=1
// pie:instantiate attn_sdpa_tiled_ring_l16_r16_seg1_bf16_d_128 PIE_HEAD_DIM=128 PIE_TILED=1 PIE_TILE_LANES=16 PIE_TILE_ROWS=16 PIE_TIER=subgroup PIE_TARGET=spirv PIE_ONE_SEGMENT=1 PIE_RING=1
// pie:instantiate attn_sdpa_tiled_lse_ring_l16_r16_seg1_bf16_d_128 PIE_HEAD_DIM=128 PIE_TILED=1 PIE_LSE=1 PIE_TILE_LANES=16 PIE_TILE_ROWS=16 PIE_TARGET=spirv PIE_ONE_SEGMENT=1 PIE_RING=1
// pie:instantiate attn_sdpa_tiled_lse_ring_l16_r16_seg1_bf16_d_128 PIE_HEAD_DIM=128 PIE_TILED=1 PIE_LSE=1 PIE_TILE_LANES=16 PIE_TILE_ROWS=16 PIE_TIER=subgroup PIE_TARGET=spirv PIE_ONE_SEGMENT=1 PIE_RING=1
// pie:instantiate attn_sdpa_tiled_ring_seg1_bf16_d_256 PIE_HEAD_DIM=256 PIE_TILED=1 PIE_TILE_ROWS=32 PIE_TARGET=spirv PIE_ONE_SEGMENT=1 PIE_RING=1
// pie:instantiate attn_sdpa_tiled_ring_r16_seg1_bf16_d_256 PIE_HEAD_DIM=256 PIE_TILED=1 PIE_TILE_ROWS=16 PIE_TARGET=spirv PIE_ONE_SEGMENT=1 PIE_RING=1
// pie:instantiate attn_sdpa_tiled_ring_r8_seg1_bf16_d_256 PIE_HEAD_DIM=256 PIE_TILED=1 PIE_TILE_ROWS=8 PIE_TARGET=spirv PIE_ONE_SEGMENT=1 PIE_RING=1
// pie:instantiate attn_sdpa_tiled_lse_ring_seg1_bf16_d_256 PIE_HEAD_DIM=256 PIE_TILED=1 PIE_LSE=1 PIE_TILE_ROWS=32 PIE_TARGET=spirv PIE_ONE_SEGMENT=1 PIE_RING=1
// pie:instantiate attn_sdpa_tiled_lse_ring_r16_seg1_bf16_d_256 PIE_HEAD_DIM=256 PIE_TILED=1 PIE_LSE=1 PIE_TILE_ROWS=16 PIE_TARGET=spirv PIE_ONE_SEGMENT=1 PIE_RING=1
// pie:instantiate attn_sdpa_tiled_lse_ring_r8_seg1_bf16_d_256 PIE_HEAD_DIM=256 PIE_TILED=1 PIE_LSE=1 PIE_TILE_ROWS=8 PIE_TARGET=spirv PIE_ONE_SEGMENT=1 PIE_RING=1
// pie:instantiate attn_sdpa_tiled_ring_l16_r16_seg1_bf16_d_256 PIE_HEAD_DIM=256 PIE_TILED=1 PIE_TILE_LANES=16 PIE_TILE_ROWS=16 PIE_TARGET=spirv PIE_ONE_SEGMENT=1 PIE_RING=1
// pie:instantiate attn_sdpa_tiled_ring_l16_r16_seg1_bf16_d_256 PIE_HEAD_DIM=256 PIE_TILED=1 PIE_TILE_LANES=16 PIE_TILE_ROWS=16 PIE_TIER=subgroup PIE_TARGET=spirv PIE_ONE_SEGMENT=1 PIE_RING=1
// pie:instantiate attn_sdpa_tiled_lse_ring_l16_r16_seg1_bf16_d_256 PIE_HEAD_DIM=256 PIE_TILED=1 PIE_LSE=1 PIE_TILE_LANES=16 PIE_TILE_ROWS=16 PIE_TARGET=spirv PIE_ONE_SEGMENT=1 PIE_RING=1
// pie:instantiate attn_sdpa_tiled_lse_ring_l16_r16_seg1_bf16_d_256 PIE_HEAD_DIM=256 PIE_TILED=1 PIE_LSE=1 PIE_TILE_LANES=16 PIE_TILE_ROWS=16 PIE_TIER=subgroup PIE_TARGET=spirv PIE_ONE_SEGMENT=1 PIE_RING=1
// pie:instantiate attn_sdpa_tiled_ring_seg1_bf16_d_512 PIE_HEAD_DIM=512 PIE_TILED=1 PIE_TILE_ROWS=32 PIE_TARGET=spirv PIE_ONE_SEGMENT=1 PIE_RING=1
// pie:instantiate attn_sdpa_tiled_ring_r16_seg1_bf16_d_512 PIE_HEAD_DIM=512 PIE_TILED=1 PIE_TILE_ROWS=16 PIE_TARGET=spirv PIE_ONE_SEGMENT=1 PIE_RING=1
// pie:instantiate attn_sdpa_tiled_ring_r8_seg1_bf16_d_512 PIE_HEAD_DIM=512 PIE_TILED=1 PIE_TILE_ROWS=8 PIE_TARGET=spirv PIE_ONE_SEGMENT=1 PIE_RING=1
// pie:instantiate attn_sdpa_tiled_lse_ring_seg1_bf16_d_512 PIE_HEAD_DIM=512 PIE_TILED=1 PIE_LSE=1 PIE_TILE_ROWS=32 PIE_TARGET=spirv PIE_ONE_SEGMENT=1 PIE_RING=1
// pie:instantiate attn_sdpa_tiled_lse_ring_r16_seg1_bf16_d_512 PIE_HEAD_DIM=512 PIE_TILED=1 PIE_LSE=1 PIE_TILE_ROWS=16 PIE_TARGET=spirv PIE_ONE_SEGMENT=1 PIE_RING=1
// pie:instantiate attn_sdpa_tiled_lse_ring_r8_seg1_bf16_d_512 PIE_HEAD_DIM=512 PIE_TILED=1 PIE_LSE=1 PIE_TILE_ROWS=8 PIE_TARGET=spirv PIE_ONE_SEGMENT=1 PIE_RING=1
// pie:instantiate attn_sdpa_tiled_ring_l16_r16_seg1_bf16_d_512 PIE_HEAD_DIM=512 PIE_TILED=1 PIE_TILE_LANES=16 PIE_TILE_ROWS=16 PIE_TARGET=spirv PIE_ONE_SEGMENT=1 PIE_RING=1
// pie:instantiate attn_sdpa_tiled_ring_l16_r16_seg1_bf16_d_512 PIE_HEAD_DIM=512 PIE_TILED=1 PIE_TILE_LANES=16 PIE_TILE_ROWS=16 PIE_TIER=subgroup PIE_TARGET=spirv PIE_ONE_SEGMENT=1 PIE_RING=1
// pie:instantiate attn_sdpa_tiled_lse_ring_l16_r16_seg1_bf16_d_512 PIE_HEAD_DIM=512 PIE_TILED=1 PIE_LSE=1 PIE_TILE_LANES=16 PIE_TILE_ROWS=16 PIE_TARGET=spirv PIE_ONE_SEGMENT=1 PIE_RING=1
// pie:instantiate attn_sdpa_tiled_lse_ring_l16_r16_seg1_bf16_d_512 PIE_HEAD_DIM=512 PIE_TILED=1 PIE_LSE=1 PIE_TILE_LANES=16 PIE_TILE_ROWS=16 PIE_TIER=subgroup PIE_TARGET=spirv PIE_ONE_SEGMENT=1 PIE_RING=1
