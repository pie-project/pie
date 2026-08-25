#include "common/bf16.slang"
#include "attn/sdpa_online.slang"

// Paged attention read side. The port keeps the Metal page-table arithmetic:
// request -> CSR page list -> physical page + in-page offset, then NHD element
// addressing. The scalar body below is intentionally subgroup-width agnostic;
// the `_p32` and `_sg8` names are ABI points, not assumptions about Vulkan's
// hardware subgroup size.

#ifndef PIE_HEAD_DIM
#error "sdpa_paged requires -DPIE_HEAD_DIM"
#endif

#if defined(PIE_TILED)
#define PIE_NUMTHREADS [numthreads(32, 32, 1)]
#else
#define PIE_NUMTHREADS [numthreads(PIE_HEAD_DIM, 1, 1)]
#endif

#if defined(PIE_COMBINE)
// The fold pass of a flash decode reads NOTHING the split pass read. It does
// not touch the queries, the pages, the page table, the positions or the
// mask -- all of that was consumed by the splits -- so it declares its own
// three bindings rather than inheriting eleven of which nine would be holes.
// The partials are `float` and not `PIE_ACT`: a partial is an unnormalised
// weighted sum whose scale is `exp(score - split_max)`, and rounding that to
// bfloat16 between the two passes would throw away eight bits of mantissa the
// single-pass path never loses.
// Numbered to match the ORDER `attn::sdpa_paged_decode` takes them in, not
// the order the fold reads them: `tests/routines.rs` holds every body to
// passing a subsequence of its signature, and the signature's output comes
// before its sinks which come before its partials. A sinkless fold never
// reads binding 1, slangc drops it, and the descriptor set carries a hole
// there -- the same shape the split pass has at `out_`.
PIE_BUFFER_RW(0, out_);
PIE_BUFFER_RO(1, sinks);
[[vk::binding(2)]] StructuredBuffer<float> partials;
#else
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
#if defined(PIE_LSE)
// THE PUBLISHED DENOMINATOR, and the one buffer in this header that is neither
// the model's nor the pool's: it is accumulator state that the point promotes
// to an operand, which is why it is `float` and not `PIE_ACT`. An lse is
// produced and consumed inside one fire; rounding it to bfloat16 would throw
// away the precision `attention.sink`'s sigmoid argument is then asked for.
//
// AN `_lse` ARM READS TEN OF THE ELEVEN IT INHERITS, and the hole is `sinks` at
// binding 10 -- an `_lse` reading reaches no sink, so nothing in this variant
// touches that declaration and `slangc` drops it. The descriptor set carries a
// hole there, which is the same shape `sdpa_paged_decode_split` has at `out_`
// and the sinkless `_combine` modules have at `sinks`, and it is what
// `Declared::holes` is for. Two holes is what `tests/rules.rs` counts for the
// two stamps below.
//
// Binding 11 is `partials`' number under `PIE_SPLIT`. The two are never
// compiled together and could not be: a split holds only its own slice's
// denominator, and the one this point states exists after the fold. See the
// claim body, which takes the unsplit arm on purpose.
[[vk::binding(11)]] RWStructuredBuffer<float> lse_out;
#endif
#if defined(PIE_SPLIT)
// Where a split leaves its `(sum over its keys of exp(s - max) * v)`, its
// `max` and its `sum_exp`. `out_` at binding 3 is declared above and never
// written by a split, so slangc drops it and the descriptor set carries a
// hole there -- which is what `Declared::holes` is for.
[[vk::binding(11)]] RWStructuredBuffer<float> partials;
#endif
#endif

#if defined(PIE_TILED)
struct Push {
    int gqa_factor;
    int page_size;
    int n_kv_heads;
    float scale;
    uint attention_mask_stride;
    int window;
    int n_rows;
#if defined(PIE_STRIDED)
    int q_row_pitch;
    int o_row_pitch;
#endif
};
[[vk::push_constant]] ConstantBuffer<Push> pc;
#elif defined(PIE_COMBINE)
// One number, and it is the one the grid cannot carry: the fold's grid is
// `(head, row)` with no third axis, so how many splits there were to fold has
// to be pushed.
struct Push {
    int n_splits;
};
[[vk::push_constant]] ConstantBuffer<Push> pc;
#else
struct Push {
    int gqa_factor;
    int page_size;
    int n_kv_heads;
    float scale;
    uint attention_mask_stride;
    int window;
};
[[vk::push_constant]] ConstantBuffer<Push> pc;
#endif

// A page size the entrypoint did not state is zero, and zero is not 32.
//
// Spelt as a default rather than as `defined(PIE_PAGE_SIZE) && PIE_PAGE_SIZE
// == 32`, which is what the GLSL said and which Slang's preprocessor warns
// about: it evaluates the second operand's identifiers even when the first is
// false, so every module compiled without the define reported an undefined
// identifier. The condition was always correct -- `&&` still short-circuits --
// but a warning that fires on most of the tree is a warning nobody reads.
#ifndef PIE_PAGE_SIZE
#define PIE_PAGE_SIZE 0
#endif

#if !defined(PIE_COMBINE)
uint page_slot(int req, int kp) {
#if PIE_PAGE_SIZE == 32
    uint page_ix = uint(kp) >> 5;
    uint page_off = uint(kp) & 31u;
    uint phys = kv_page_indices[kv_page_indptr[req] + page_ix];
    return phys * 32u + page_off;
#else
    uint page_ix = uint(kp / pc.page_size);
    uint page_off = uint(kp % pc.page_size);
    uint phys = kv_page_indices[kv_page_indptr[req] + page_ix];
    return phys * uint(pc.page_size) + page_off;
#endif
}

bool keeps(uint row, int kp, int q_pos, int start) {
    if (kp > q_pos || kp < start) return false;
    if (attention_mask_enabled[row] != uint8_t(0)) {
        if (uint(kp) >= pc.attention_mask_stride) return false;
        if (attention_mask[row * pc.attention_mask_stride + uint(kp)] == uint8_t(0)) return false;
    }
    return true;
}
#endif

uint q_base_for(uint row, uint q_head, uint n_q_heads) {
#if defined(PIE_TILED) && defined(PIE_STRIDED)
    return (pc.q_row_pitch > 0 ? row * uint(pc.q_row_pitch) : row * n_q_heads * uint(PIE_HEAD_DIM)) + q_head * uint(PIE_HEAD_DIM);
#else
    return (row * n_q_heads + q_head) * uint(PIE_HEAD_DIM);
#endif
}

uint o_base_for(uint row, uint q_head, uint n_q_heads) {
#if defined(PIE_TILED) && defined(PIE_STRIDED)
    return (pc.o_row_pitch > 0 ? row * uint(pc.o_row_pitch) : row * n_q_heads * uint(PIE_HEAD_DIM)) + q_head * uint(PIE_HEAD_DIM);
#else
    return (row * n_q_heads + q_head) * uint(PIE_HEAD_DIM);
#endif
}

#if defined(PIE_TILED)
// The prefill path, with the score computed once per key instead of once per
// thread-and-dimension.
//
// This kernel used to be `compute_one` in a loop: thirty-two threads across a
// row, each walking `PIE_HEAD_DIM / 32` output dimensions, and each of those
// walks recomputing the entire key loop from scratch. That is `PIE_HEAD_DIM`
// full dot products per key per row -- the same redundancy the decode path
// had, arrived at by a different route.
//
// Making it cooperative is harder here than it was for decode, because a
// tile spans thirty-two DIFFERENT rows, and rows have different positions,
// different windows and different masks. A barrier is workgroup-wide, so a
// reduction inside a per-row loop would have threads waiting at barriers
// their neighbours never reach. Three things make the loop uniform:
//
//   - the bound is the largest position in the whole tile, found once with a
//     shared atomic, not each row's own position;
//   - rows past `n_rows` stay in the loop with `q_pos = -1` instead of
//     returning, so they reach every barrier and contribute zero;
//   - the mask test becomes a predicate on the body, not a `continue`.
//
// A row therefore runs some iterations that do nothing. In a prefill the
// tile's rows are consecutive positions, so the waste is bounded by the tile
// height -- thirty-one keys against a history of hundreds -- and it buys back
// a factor of `PIE_HEAD_DIM` on the part that dominates.
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
#if defined(PIE_FAST_FULL)
    int start = 0;
#else
    int start = (pc.window > 0 && q_pos >= pc.window) ? (q_pos - pc.window + 1) : 0;
#endif
    uint q_base = q_base_for(row, q_head, n_q_heads);
    uint o_base = o_base_for(row, q_head, n_q_heads);
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
        // The next key overwrites the scratch, so no thread may run ahead of
        // the slowest reader of `pie_sdpa_part[slot_y][0]`.
        GroupMemoryBarrierWithGroupSync();
    }

#if defined(PIE_LSE)
    // ONE writer per `(row, query head)`, and it can be any of the 32 x lanes:
    // every one of them ran `sdpa_online_update` on the same `score` -- the
    // tree fold above publishes it at `pie_sdpa_part[slot_y][0]` and all 32
    // read it -- so they all hold the same `(max_score, sum_exp)`. `lane == 0`
    // picks one, and `live` is the same guard the writeout below takes: a row
    // past `pc.n_rows` belongs to the tile's padding and has no lse slot.
    //
    // ABOVE the `PIE_WITH_SINK` arm and not below it. The two are never
    // compiled together, but the ORDER is the statement:
    // `attention.prefill_lse` publishes the log-sum-exp of the softmax it
    // normalised by, and `attention.sink` is the separate statement that
    // extends that denominator afterwards. A sink folded in first would publish
    // a denominator no downstream `sigmoid(lse - sink)` could undo.
    if (live && lane == 0u) {
        lse_out[row * n_q_heads + q_head] = sdpa_lse_base2(max_score, sum_exp);
    }
#endif
#if defined(PIE_WITH_SINK)
    float orescale = sdpa_merge_sink(PIE_LOAD(sinks[q_head]), max_score, sum_exp);
    for (uint i = 0u; i < PIE_D_PER_LANE; i++) out_acc[i] *= orescale;
#endif
    if (!live) return;
    for (uint i = 0u; i < PIE_D_PER_LANE; i++) {
        uint d = lane + i * PIE_LANES_PER_ROW;
        out_[o_base + d] = PIE_STORE(sum_exp == 0.0 ? out_acc[i] : out_acc[i] / sum_exp);
    }
}
#endif

#if !defined(PIE_TILED) && !defined(PIE_COMBINE)
// The decode path, with each SUBGROUP walking its own keys.
//
// Two rewrites have happened here and the second undoes the cost of the
// first. The original shape -- every thread owns one output dimension and
// works out the whole score for itself -- paid the dot product `PIE_HEAD_DIM`
// times over. Replacing it with a cooperative score removed that arithmetic
// and replaced it with SYNCHRONISATION: one shared-memory tree fold per key,
// which at `PIE_HEAD_DIM == 128` is `log2(128) = 7` barriered rounds plus one
// barrier to publish the partial and one to protect the scratch from the next
// key. Nine whole-workgroup barriers per key, ~3,456 of them per (head,
// layer) at a 384-key history, with four warps resident to hide them.
//
// The measurement said what that costs. At 384 keys, 8 KV heads and a 128-wide
// head, one dispatch moves 1.5 MiB and took 288 us on a card independently
// measured at ~1000 GB/s -- 5.5 GB/s, a factor of 180 off. The rows sweep in
// `tests/sdpa_bench.rs` settled the cause: one decode row and EIGHT cost the
// same 265 us, so the card was not busy and the time was latency, not work.
//
// So the workgroup stops cooperating on one key. Subgroup `s` of
// `gl_NumSubgroups` walks keys `start + s, start + s + n, ...`; within it a
// lane owns dimensions `WaveGetLaneIndex(), + width, ...` across the head; and
// the score is one `WaveActiveSum`, which is internally synchronised, needs no
// `GroupMemoryBarrierWithGroupSync` and lands already broadcast to every lane
// that will use it. Each subgroup keeps its OWN online-softmax state and its
// own output accumulator, and the `gl_NumSubgroups` states are merged ONCE at
// the end. The inner loop contains no group barrier at all: the whole kernel
// has two, one after the query is staged and one before the merge reads what
// the other subgroups accumulated.
//
// The merge is the standard online-softmax combine and is exact:
// `pie_sdpa_acc[s][d]` holds `sum over s's keys of exp(score - max_s) * v`, so
// rescaling it by `exp(max_s - max_global)` restores a common base. Only the
// ASSOCIATION of the sum changes against the sequential fold, not the value it
// approximates.
//
// ## Why the accumulator is shared and bounded by a constant
//
// A lane's dimension count is `PIE_HEAD_DIM / gl_SubgroupSize`, which is not a
// compile-time constant -- the width is 32 here, 64 on AMD and as little as 8
// on Intel -- so it cannot size a register array. The accumulator therefore
// lives in groupshared, indexed `[subgroup][dimension]`, and each lane touches
// only entries no other invocation touches, which is why the loop needs no
// barrier despite writing shared memory every key.
//
// That array must be bounded at compile time too, and `gl_NumSubgroups` can be
// as large as `PIE_HEAD_DIM` on a hypothetical one-lane subgroup. The bound
// below is a shared-memory budget of 16 KiB; a device that reports more
// subgroups than fit simply leaves the extra ones idle through the key loop
// (they still reach both barriers and still write their own output
// dimensions), which is slower than it could be and never wrong.
#if PIE_HEAD_DIM >= 512
#define PIE_SDPA_MAX_SUBGROUPS 8
#else
#define PIE_SDPA_MAX_SUBGROUPS 16
#endif

groupshared float pie_sdpa_q[PIE_HEAD_DIM];
groupshared float pie_sdpa_acc[PIE_SDPA_MAX_SUBGROUPS][PIE_HEAD_DIM];
groupshared float pie_sdpa_max[PIE_SDPA_MAX_SUBGROUPS];
groupshared float pie_sdpa_sum[PIE_SDPA_MAX_SUBGROUPS];

void decode_one(uint row, uint q_head, uint d, uint n_q_heads) {
    int req = req_of_token[row];
    int q_pos = position_ids[row];
#if defined(PIE_FAST_FULL)
    int start = 0;
#else
    int start = (pc.window > 0 && q_pos >= pc.window) ? (q_pos - pc.window + 1) : 0;
#endif
    uint q_base = q_base_for(row, q_head, n_q_heads);
    uint o_base = o_base_for(row, q_head, n_q_heads);
    int kv_head = int(q_head) / pc.gqa_factor;

    uint sg = gl_SubgroupID;
    uint lane = WaveGetLaneIndex();
    uint width = WaveGetLaneCount();
    uint subgroups = min(gl_NumSubgroups, uint(PIE_SDPA_MAX_SUBGROUPS));
    bool walks = sg < subgroups;

    // The query is read once per workgroup rather than once per key, scaled
    // on the way in so the inner product is a bare FMA. One thread per
    // dimension, because the workgroup is exactly `PIE_HEAD_DIM` wide.
    pie_sdpa_q[d] = pc.scale * PIE_LOAD(queries[q_base + d]);
    if (walks) {
        for (uint i = lane; i < uint(PIE_HEAD_DIM); i += width) pie_sdpa_acc[sg][i] = 0.0;
    }
    GroupMemoryBarrierWithGroupSync();

    float max_score = PIE_SDPA_NEG_INF;
    float sum_exp = 0.0;
    if (walks) {
        // `keeps` and `page_slot` depend only on the row and the key, both of
        // which are uniform across the subgroup, so every lane reaches the
        // `WaveActiveSum` below on exactly the same iterations.
        //
        // One key per iteration, which is not obviously right and was
        // measured. What is left after the barriers are gone is ~700 ns per
        // key per subgroup at a 128-wide head and ~690 ns at a 64-wide one --
        // a figure that barely moves with the arithmetic, so it is one memory
        // round trip per iteration that a 16-workgroup launch has nothing to
        // hide behind. Two attempts to shorten it failed and are recorded so
        // they are not retried: hoisting `page_slot`'s two dependent loads out
        // of the loop (the CSR base is invariant and the physical page changes
        // only every `page_size` keys) measured 68.5 us against 67.8, because
        // those arrays are small and stay in L1; and taking FOUR keys per
        // iteration, to put four independent key loads in flight, measured
        // 134 us -- twice as slow, the per-key `float`/`uint` arrays it needs
        // spilling to scratch memory because the dimension walk inside them
        // is not a compile-time trip count. The latency is real and the fix
        // for it is more workgroups, not a longer loop body.
        for (int kp = start + int(sg); kp <= q_pos; kp += int(subgroups)) {
            if (!keeps(row, kp, q_pos, start)) continue;
            uint slot = page_slot(req, kp);
            uint base = (slot * uint(pc.n_kv_heads) + uint(kv_head)) * uint(PIE_HEAD_DIM);
            float partial = 0.0;
            for (uint i = lane; i < uint(PIE_HEAD_DIM); i += width) {
                partial += pie_sdpa_q[i] * PIE_LOAD(k_pages[base + i]);
            }
            float score = WaveActiveSum(partial);
            float hist, prob;
            sdpa_online_update(score, max_score, sum_exp, hist, prob);
            for (uint i = lane; i < uint(PIE_HEAD_DIM); i += width) {
                pie_sdpa_acc[sg][i] = pie_sdpa_acc[sg][i] * hist + prob * PIE_LOAD(v_pages[base + i]);
            }
        }
        if (WaveIsFirstLane()) {
            pie_sdpa_max[sg] = max_score;
            pie_sdpa_sum[sg] = sum_exp;
        }
    }
    // Both the accumulators and the two state arrays are read below by
    // invocations that did not write them, and this is the barrier that
    // orders that. There is no second one -- `common/reduce.slang` needs its
    // because its scratch is reused on the next call, and nothing here writes
    // the scratch again.
    GroupMemoryBarrierWithGroupSync();

    // The merge. A subgroup that drew no key at all keeps `PIE_SDPA_NEG_INF`
    // and a zero sum; its weight is `exp(-3e38 - max)`, which is zero for any
    // real maximum and exactly one when EVERY subgroup was empty -- and in
    // that case the sum is still zero, so the guarded division below returns
    // the zero accumulator rather than a NaN.
    float merged_max = PIE_SDPA_NEG_INF;
    for (uint s = 0u; s < subgroups; s++) merged_max = max(merged_max, pie_sdpa_max[s]);
    float merged_sum = 0.0;
    float out_acc = 0.0;
    for (uint s = 0u; s < subgroups; s++) {
        float w = exp(pie_sdpa_max[s] - merged_max);
        merged_sum += pie_sdpa_sum[s] * w;
        out_acc += pie_sdpa_acc[s][d] * w;
    }

#if defined(PIE_LSE)
    // The decode twin of the tiled arm's block, and the redundancy it rests on
    // is the same one the merge above already rests on: every invocation of
    // this workgroup walked the whole `subgroups` fold and therefore holds the
    // same `(merged_max, merged_sum)` -- only `out_acc` is per-`d`. `d == 0`
    // picks one. See the tiled twin for why this stands above the sink fold.
    if (d == 0u) {
        lse_out[row * n_q_heads + q_head] = sdpa_lse_base2(merged_max, merged_sum);
    }
#endif
#if defined(PIE_WITH_SINK)
    float orescale = sdpa_merge_sink(PIE_LOAD(sinks[q_head]), merged_max, merged_sum);
    out_acc *= orescale;
#endif
    out_[o_base + d] = PIE_STORE(merged_sum == 0.0 ? out_acc : out_acc / merged_sum);
}
#endif

#if defined(PIE_SPLIT)
// The first pass of a FLASH DECODE: the same body as `decode_one`, over one
// contiguous slice of the key range, leaving a partial nobody has normalised.
//
// ## Why this exists
//
// `decode_one` is occupancy-bound and the sweep in `tests/sdpa_bench.rs` is
// what says so: at a 384-key history one decode row costs 67.65 us and
// THIRTY-TWO rows cost 76.03 -- thirty-two times the work for twelve percent
// more time, and 662 GB/s against a card measured at 996. A single row of
// qwen3-0.6b launches `q_heads * rows` = 16 workgroups on a 128-SM card, so
// fifteen sixteenths of the machine is idle and the ~700 ns per key per
// subgroup that is left is one memory round trip with nothing to hide behind.
// Two attempts to shorten that round trip were measured and rejected (see
// `decode_one`); the fix is more workgroups, and the only axis left to make
// them out of is the key range itself.
//
// ## The decomposition
//
// Split `[start, q_pos]` into `gl_NumWorkGroups.z` CONTIGUOUS chunks, one per
// workgroup, and give the split its own online-softmax state. Contiguous and
// not strided: a strided assignment at `S = 8` and four subgroups walks with
// a stride of 32, which is exactly a page, so every workgroup would touch
// every page. A chunk of 48 keys touches two.
//
// What a split leaves behind is `(max_s, sum_s, acc_s)` where
// `acc_s = sum over its keys of exp(score - max_s) * v` -- unnormalised, and
// in `float` rather than bfloat16, because the fold has to rescale it by
// `exp(max_s - max)` before it means anything.
//
// ## What is deliberately NOT done here
//
// The sink. `sdpa_merge_sink` adds one more term to the softmax, and adding
// it once per split would add it `S` times. It belongs in the fold, which is
// why there is no `PIE_WITH_SINK` in this body and why the sink and non-sink
// decodes share ONE split module.
void decode_split(uint row, uint q_head, uint d, uint n_q_heads, uint n_rows,
                  uint split, uint n_splits) {
    int req = req_of_token[row];
    int q_pos = position_ids[row];
#if defined(PIE_FAST_FULL)
    int start = 0;
#else
    int start = (pc.window > 0 && q_pos >= pc.window) ? (q_pos - pc.window + 1) : 0;
#endif
    uint q_base = q_base_for(row, q_head, n_q_heads);
    int kv_head = int(q_head) / pc.gqa_factor;

    // The slice. `keys` can be zero or negative for a row whose window has
    // not opened, and `chunk` is then zero and every split's `lo > hi`, so
    // the loop below never runs and the whole grid writes the empty partial
    // the fold reads as "contributed nothing".
    int keys = q_pos - start + 1;
    int chunk = keys > 0 ? (keys + int(n_splits) - 1) / int(n_splits) : 0;
    int lo = start + int(split) * chunk;
    int hi = min(q_pos, lo + chunk - 1);

    uint sg = gl_SubgroupID;
    uint lane = WaveGetLaneIndex();
    uint width = WaveGetLaneCount();
    uint subgroups = min(gl_NumSubgroups, uint(PIE_SDPA_MAX_SUBGROUPS));
    bool walks = sg < subgroups;

    pie_sdpa_q[d] = pc.scale * PIE_LOAD(queries[q_base + d]);
    if (walks) {
        for (uint i = lane; i < uint(PIE_HEAD_DIM); i += width) pie_sdpa_acc[sg][i] = 0.0;
    }
    GroupMemoryBarrierWithGroupSync();

    float max_score = PIE_SDPA_NEG_INF;
    float sum_exp = 0.0;
    if (walks) {
        for (int kp = lo + int(sg); kp <= hi; kp += int(subgroups)) {
            if (!keeps(row, kp, q_pos, start)) continue;
            uint slot = page_slot(req, kp);
            uint base = (slot * uint(pc.n_kv_heads) + uint(kv_head)) * uint(PIE_HEAD_DIM);
            float partial = 0.0;
            for (uint i = lane; i < uint(PIE_HEAD_DIM); i += width) {
                partial += pie_sdpa_q[i] * PIE_LOAD(k_pages[base + i]);
            }
            float score = WaveActiveSum(partial);
            float hist, prob;
            sdpa_online_update(score, max_score, sum_exp, hist, prob);
            for (uint i = lane; i < uint(PIE_HEAD_DIM); i += width) {
                pie_sdpa_acc[sg][i] = pie_sdpa_acc[sg][i] * hist + prob * PIE_LOAD(v_pages[base + i]);
            }
        }
        if (WaveIsFirstLane()) {
            pie_sdpa_max[sg] = max_score;
            pie_sdpa_sum[sg] = sum_exp;
        }
    }
    GroupMemoryBarrierWithGroupSync();

    // The within-workgroup merge is `decode_one`'s, unchanged and for the
    // same reason: `subgroups` states are folded once, and a subgroup that
    // drew no key weighs `exp(-3e38 - max)`, which is zero against any real
    // maximum and one when every state is empty -- in which case the sum is
    // zero too and the fold's guard is what answers.
    float merged_max = PIE_SDPA_NEG_INF;
    for (uint s = 0u; s < subgroups; s++) merged_max = max(merged_max, pie_sdpa_max[s]);
    float merged_sum = 0.0;
    float out_acc = 0.0;
    for (uint s = 0u; s < subgroups; s++) {
        float w = exp(pie_sdpa_max[s] - merged_max);
        merged_sum += pie_sdpa_sum[s] * w;
        out_acc += pie_sdpa_acc[s][d] * w;
    }

    // Written UNCONDITIONALLY, by every workgroup of the grid, including one
    // whose slice was empty. The fold reads all `S` entries and nothing
    // zeroes this buffer between fires, so a split that skipped its write
    // would hand the fold the previous token's numbers.
    uint at = (split * n_rows + row) * n_q_heads + q_head;
    partials[at * uint(PIE_HEAD_DIM) + d] = out_acc;
    if (d == 0u) {
        uint stats = n_splits * n_rows * n_q_heads * uint(PIE_HEAD_DIM) + at * 2u;
        partials[stats] = merged_max;
        partials[stats + 1u] = merged_sum;
    }
}
#endif

#if defined(PIE_COMBINE)
// The second pass of a flash decode: fold `S` partials into one row.
//
// One workgroup per `(query head, row)` and one lane per dimension, which is
// the grid `decode_one` had -- so this pass launches exactly the workgroups
// the single-pass kernel did and reads `S * (PIE_HEAD_DIM + 2)` floats each.
// At qwen3-0.6b's shape with `S = 8` that is 8 * 130 * 4 = 4 KiB a workgroup
// and 66 KiB a dispatch, against the 1.5 MiB the split pass moves.
//
// The merge is the standard online-softmax combine and is exact: `acc_s` is
// `sum over s's keys of exp(score - max_s) * v`, so `exp(max_s - max)` puts
// every split on one base. Only the ASSOCIATION of the sum differs from the
// sequential fold.
void decode_combine(uint row, uint q_head, uint d, uint n_q_heads, uint n_rows) {
    uint n_splits = uint(pc.n_splits);
    uint stats0 = n_splits * n_rows * n_q_heads * uint(PIE_HEAD_DIM);

    float merged_max = PIE_SDPA_NEG_INF;
    for (uint s = 0u; s < n_splits; s++) {
        uint at = (s * n_rows + row) * n_q_heads + q_head;
        merged_max = max(merged_max, partials[stats0 + at * 2u]);
    }
    float merged_sum = 0.0;
    float out_acc = 0.0;
    for (uint s = 0u; s < n_splits; s++) {
        uint at = (s * n_rows + row) * n_q_heads + q_head;
        // `exp(-3e38 - merged_max)` is zero for any real maximum, so an empty
        // split contributes nothing without an `if`; when EVERY split was
        // empty the weight is `exp(0) == 1` and the sums are all zero, so the
        // guarded division below returns the zero accumulator and not a NaN.
        float w = exp(partials[stats0 + at * 2u] - merged_max);
        merged_sum += partials[stats0 + at * 2u + 1u] * w;
        out_acc += partials[at * uint(PIE_HEAD_DIM) + d] * w;
    }

#if defined(PIE_WITH_SINK)
    // ONCE, over the whole key range, which is the reason the sink is not in
    // the split pass: a per-split sink would join the softmax `S` times.
    float orescale = sdpa_merge_sink(PIE_LOAD(sinks[q_head]), merged_max, merged_sum);
    out_acc *= orescale;
#endif
    uint o_base = o_base_for(row, q_head, n_q_heads);
    out_[o_base + d] = PIE_STORE(merged_sum == 0.0 ? out_acc : out_acc / merged_sum);
}
#endif

PIE_NUMTHREADS
void main(uint3 group : SV_GroupID, uint3 local : SV_GroupThreadID, uint group_index : SV_GroupIndex) {
#if defined(PIE_TILED)
    tiled_tile(group.x, gl_NumWorkGroups.x, group, local, group_index);
#elif defined(PIE_COMBINE)
    decode_combine(group.y, group.x, local.x, gl_NumWorkGroups.x, gl_NumWorkGroups.y);
#elif defined(PIE_SPLIT)
    decode_split(group.y, group.x, local.x, gl_NumWorkGroups.x, gl_NumWorkGroups.y,
                 group.z, gl_NumWorkGroups.z);
#else
    decode_one(group.y, group.x, local.x, gl_NumWorkGroups.x);
#endif
}

// pie:instantiate sdpa_paged_decode_bfloat16_d_64 PIE_HEAD_DIM=64
// pie:instantiate sdpa_paged_decode_bfloat16_d_128 PIE_HEAD_DIM=128
// pie:instantiate sdpa_paged_decode_bfloat16_d_256 PIE_HEAD_DIM=256
// pie:instantiate sdpa_paged_decode_bfloat16_d_512 PIE_HEAD_DIM=512
// pie:instantiate sdpa_paged_decode_bfloat16_d_64_p32 PIE_HEAD_DIM=64 PIE_PAGE_SIZE=32 PIE_FAST_FULL=1
// pie:instantiate sdpa_paged_decode_bfloat16_d_128_p32 PIE_HEAD_DIM=128 PIE_PAGE_SIZE=32 PIE_FAST_FULL=1
// pie:instantiate sdpa_paged_decode_bfloat16_d_64_p32_sg8 PIE_HEAD_DIM=64 PIE_PAGE_SIZE=32 PIE_FAST_FULL=1 PIE_SHORT_GROUP=8
// pie:instantiate sdpa_paged_decode_sink_bfloat16_d_64 PIE_HEAD_DIM=64 PIE_WITH_SINK=1
// THE PUBLISHED DENOMINATOR, at gpt-oss's width and no other.
//
// `attention.decode_lse` is stated by one family on this plane and its heads
// are 64 wide, so one stamp is what the tree carries; `head_point` refuses
// every other width BY NAME at the fire, which is a better answer than three
// more pipelines nothing states. Add one the day a model states one.
//
// A SEPARATE ENTRY POINT AND NOT AN ARGUMENT THE UNSPLIT ARM LEARNS TO IGNORE:
// a variant's buffer list IS its binding contract here -- `Declared` reads the
// compiled module's `OpDecorate Binding` set, and `encode::fire` holds a
// launch's operand run against it -- so a twelfth operand is a new entry point
// by construction.
// pie:instantiate sdpa_paged_decode_lse_bfloat16_d_64 PIE_HEAD_DIM=64 PIE_LSE=1
// The flash-decode pair. One split module serves the sink and non-sink forms
// both, because the sink is merged in the fold and not per split.
// pie:instantiate sdpa_paged_decode_split_bfloat16_d_64 PIE_HEAD_DIM=64 PIE_SPLIT=1
// pie:instantiate sdpa_paged_decode_split_bfloat16_d_128 PIE_HEAD_DIM=128 PIE_SPLIT=1
// pie:instantiate sdpa_paged_decode_split_bfloat16_d_256 PIE_HEAD_DIM=256 PIE_SPLIT=1
// pie:instantiate sdpa_paged_decode_split_bfloat16_d_512 PIE_HEAD_DIM=512 PIE_SPLIT=1
// pie:instantiate sdpa_paged_decode_combine_bfloat16_d_64 PIE_HEAD_DIM=64 PIE_COMBINE=1
// pie:instantiate sdpa_paged_decode_combine_bfloat16_d_128 PIE_HEAD_DIM=128 PIE_COMBINE=1
// pie:instantiate sdpa_paged_decode_combine_bfloat16_d_256 PIE_HEAD_DIM=256 PIE_COMBINE=1
// pie:instantiate sdpa_paged_decode_combine_bfloat16_d_512 PIE_HEAD_DIM=512 PIE_COMBINE=1
// pie:instantiate sdpa_paged_decode_combine_sink_bfloat16_d_64 PIE_HEAD_DIM=64 PIE_COMBINE=1 PIE_WITH_SINK=1
// pie:instantiate sdpa_paged_tiled_bfloat16_d_64 PIE_HEAD_DIM=64 PIE_TILED=1
// pie:instantiate sdpa_paged_tiled_bfloat16_d_128 PIE_HEAD_DIM=128 PIE_TILED=1
// pie:instantiate sdpa_paged_tiled_bfloat16_d_256 PIE_HEAD_DIM=256 PIE_TILED=1
// pie:instantiate sdpa_paged_tiled_bfloat16_d_512 PIE_HEAD_DIM=512 PIE_TILED=1
// pie:instantiate sdpa_paged_tiled_sink_bfloat16_d_64 PIE_HEAD_DIM=64 PIE_TILED=1 PIE_WITH_SINK=1
// `attention.prefill_lse`, at the one width that states it. See the decode
// twin above for why there is one stamp and not four.
// pie:instantiate sdpa_paged_tiled_lse_bfloat16_d_64 PIE_HEAD_DIM=64 PIE_TILED=1 PIE_LSE=1
// pie:instantiate sdpa_paged_tiled_strided_bfloat16_d_256 PIE_HEAD_DIM=256 PIE_TILED=1 PIE_STRIDED=1
