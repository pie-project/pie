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
// values, a u32 word a lane. A source unit below 0 is the ring of lane
// `-1 - unit`, its row `r` at place `r % ring_rows` (a snapshot's rows
// below the lane's folded index, read before the fold writes the ring). No
// two jobs of a phase land on one destination row, and none reads a row
// another writes.
//
// A dispatch runs one phase: the jobs whose count lies at `jobs_at` (the
// snapshots' before the fold's), its workgroups striding over them by the
// grid's rows, so a fire's grid is one whatever its jobs.

[[vk::binding(0)]] StructuredBuffer<int> jobs;
[[vk::binding(1)]] StructuredBuffer<int> lanes;
PIE_POOL_SEGMENTS(RWStructuredBuffer, uint, 2)

static const uint LAYER_INTS = 9u;

struct Params {
    uint seg_words;
    uint pieces;
    uint slot_stride;
    uint layers;
    // The phase's count, an int index of `jobs`; the grid's job rows.
    uint jobs_at;
    uint stride;
};
PIE_PARAMS(6, Params, params);
PIE_POOL_READ(uint, params.seg_words)
PIE_POOL_WRITE(uint, params.seg_words)

[shader("compute")]
[numthreads(PIE_GROUP_X, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    uint w = gid.x;
    uint layer = gid.z;
    uint jobs_at = params.jobs_at;
    if (layer >= params.layers) {
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
    uint origin = uint(jobs[l + (values ? 1u : 0u)]);
    uint piece = uint(jobs[l + (values ? 5u : 4u)]);
    uint ring_at = uint(jobs[l + (values ? 7u : 6u)]);
    int count = jobs[jobs_at];
    for (uint job = gid.y; int(job) < count; job += params.stride) {
        uint at = jobs_at + 1u + 4u * job;
        int src_unit = jobs[at];
        uint src_row = uint(jobs[at + 1u]);
        int dst = jobs[at + 2u];
        uint dst_row = uint(jobs[at + 3u]);
        uint src;
        if (src_unit < 0) {
            uint from = (6u + params.pieces) * uint(-1 - src_unit);
            uint rec = uint(lanes[from + 6u + piece]) * params.slot_stride;
            src = rec + ring_at + (src_row % ring_rows) * row_words + i;
        } else {
            src = origin + uint(src_unit) * page_stride + src_row * row_words + i;
        }
        uint to;
        if (dst < 0) {
            uint lane = uint(-1 - dst);
            uint at_lane = (6u + params.pieces) * lane;
            if (dst_row + ring_rows < uint(lanes[at_lane + 3u])) {
                continue;
            }
            uint rec = uint(lanes[at_lane + 6u + piece]) * params.slot_stride;
            to = rec + ring_at + (dst_row % ring_rows) * row_words + i;
        } else {
            to = origin + uint(dst) * page_stride + dst_row * row_words + i;
        }
        pie_pool_set(to, pie_pool_get(src));
    }
}

// pie:instantiate attn_ring_fold PIE_GROUP_X=64
