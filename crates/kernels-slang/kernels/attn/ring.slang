#include "prim/params.slang"
#include "prim/pool.slang"

// Every windowed layer's fold (CONTRACT-BC 19, 21), one dispatch a fire:
// rows a lane folds move from the slots that hold them (its window-buffer
// slots, or the fire's own slots, laid as buffer slots) into the ring in its
// record, and rows of the fire it does not fold into its buffer slots. The
// grid's z is the layer: `jobs` opens with `layers` entries of
// `LAYER_INTS` ints, a layer's geometry (its keys' and values' origins, its
// slot unit and row words, the pieces and words its ring's keys and values
// lie at, its ring rows), then the jobs' count and four ints a job, which
// every layer moves alike: the source slot (units of the layer's
// `page_stride` words from its origins) and its row there; the destination,
// a slot unit (`>= 0`) or the ring of lane `-1 - dst` (its record's piece
// at its unit `lanes[(6 + pieces) lane + 6 + piece]`, units of
// `slot_stride` words into the pool), and its row there: a slot's row, or
// the ring's row `r` itself, which lands at place `r % ring_rows` unless it
// lies `ring_rows` or more below the lane's folded index after the fire
// (`lanes[(6 + pieces) lane + 3]`), so a layer of a narrower ring than the
// jobs were laid for drops the rows past it. A job moves one row's keys and
// values, a u32 word a lane. No two jobs of a fire land on one destination
// row, and none reads a row another writes.

[[vk::binding(0)]] StructuredBuffer<int> jobs;
[[vk::binding(1)]] StructuredBuffer<int> lanes;
PIE_POOL_SEGMENTS(RWStructuredBuffer, uint, 2)

static const uint LAYER_INTS = 9u;

struct Params {
    uint pieces;
    uint slot_stride;
    uint layers;
};
PIE_PARAMS(6, Params, params);
PIE_POOL_READ(uint)
PIE_POOL_WRITE(uint)

[shader("compute")]
[numthreads(PIE_GROUP_X, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    uint w = gid.x;
    uint job = gid.y;
    uint layer = gid.z;
    uint jobs_at = params.layers * LAYER_INTS;
    if (layer >= params.layers || int(job) >= jobs[jobs_at]) {
        return;
    }
    uint l = layer * LAYER_INTS;
    uint row_words = uint(jobs[l + 3u]);
    if (w >= 2u * row_words) {
        return;
    }
    uint page_stride = uint(jobs[l + 2u]);
    uint ring_rows = uint(jobs[l + 8u]);
    bool values = w >= row_words;
    uint i = values ? w - row_words : w;
    uint at = jobs_at + 1u + 4u * job;
    uint src_unit = uint(jobs[at]);
    uint src_row = uint(jobs[at + 1u]);
    int dst = jobs[at + 2u];
    uint dst_row = uint(jobs[at + 3u]);
    uint origin = uint(jobs[l + (values ? 1u : 0u)]);
    uint src = origin + pie_unit_at(src_unit, page_stride) + src_row * row_words + i;
    uint to;
    if (dst < 0) {
        uint lane = uint(-1 - dst);
        uint at_lane = (6u + params.pieces) * lane;
        if (dst_row + ring_rows < uint(lanes[at_lane + 3u])) {
            return;
        }
        dst_row = dst_row % ring_rows;
        uint piece = uint(jobs[l + (values ? 5u : 4u)]);
        uint rec = pie_unit_at(uint(lanes[at_lane + 6u + piece]), params.slot_stride);
        to = rec + uint(jobs[l + (values ? 7u : 6u)]) + dst_row * row_words + i;
    } else {
        to = origin + pie_unit_at(uint(dst), page_stride) + dst_row * row_words + i;
    }
    pie_pool_set(to, pie_pool_get(src));
}

// pie:instantiate attn_ring_fold PIE_GROUP_X=64
