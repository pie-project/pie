#include "prim/params.slang"
#include "prim/pool.slang"

// A windowed layer's fold (CONTRACT-BC 19, 21): rows a lane folds move from
// the slots that hold them (its window-buffer slots, or the fire's own
// slots, laid as buffer slots) into the ring in its record, and rows of the
// fire it does not fold into its buffer slots. A job moves one row's keys
// and values, a u32 word a lane: `jobs` holds the jobs' count, then four
// ints a job: the source slot (units of `page_stride` words from the
// layer's origins) and its row there; the destination, a slot unit (`>= 0`)
// or the ring of lane `-1 - dst` (its record's piece `piece` at its unit
// `lanes[(4 + pieces) lane + 4 + piece]`, units of `slot_stride` words into
// the pool, the ring's keys at word
// `ring_k` past it, its values at `ring_v` of piece `ring_piece_v`), and
// its row there: a slot's
// row, or the ring's row `r` itself, which lands at place `r % ring_rows`
// unless it lies `ring_rows` or more below the lane's folded index after
// the fire (`lanes[(4 + pieces) lane + 3]`), so a layer of a narrower ring than the
// jobs were laid for drops the rows past it. No two jobs of a fire land on
// one destination row, and none reads a row another writes.

[[vk::binding(0)]] StructuredBuffer<int> jobs;
[[vk::binding(1)]] StructuredBuffer<int> lanes;
PIE_POOL_SEGMENTS(RWStructuredBuffer, uint, 2)

struct Params {
    uint seg_words;
    uint k_origin;
    uint v_origin;
    uint page_stride;
    // u32 words of one row's keys (and of its values).
    uint row_words;
    uint ring_piece;
    uint ring_piece_v;
    uint pieces;
    uint ring_k;
    uint ring_v;
    uint slot_stride;
    uint ring_rows;
};
PIE_PARAMS(6, Params, params);
PIE_POOL_READ(uint, params.seg_words)
PIE_POOL_WRITE(uint, params.seg_words)

[shader("compute")]
[numthreads(PIE_GROUP_X, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    uint w = gid.x;
    uint job = gid.y;
    uint row_words = params.row_words;
    if (w >= 2u * row_words || int(job) >= jobs[0]) {
        return;
    }
    bool values = w >= row_words;
    uint i = values ? w - row_words : w;
    uint at = 1u + 4u * job;
    uint src_unit = uint(jobs[at]);
    uint src_row = uint(jobs[at + 1u]);
    int dst = jobs[at + 2u];
    uint dst_row = uint(jobs[at + 3u]);
    uint origin = values ? params.v_origin : params.k_origin;
    uint src = origin + src_unit * params.page_stride + src_row * row_words + i;
    uint to;
    if (dst < 0) {
        uint lane = uint(-1 - dst);
        uint at_lane = (4u + params.pieces) * lane;
        if (dst_row + params.ring_rows < uint(lanes[at_lane + 3u])) {
            return;
        }
        dst_row = dst_row % params.ring_rows;
        uint piece = values ? params.ring_piece_v : params.ring_piece;
        uint rec = uint(lanes[at_lane + 4u + piece]) * params.slot_stride;
        to = rec + (values ? params.ring_v : params.ring_k) + dst_row * row_words + i;
    } else {
        to = origin + uint(dst) * params.page_stride + dst_row * row_words + i;
    }
    pie_pool_set(to, pie_pool_get(src));
}

// pie:instantiate attn_ring_fold PIE_GROUP_X=64
