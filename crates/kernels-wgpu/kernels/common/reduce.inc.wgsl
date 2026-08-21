// The row reduction every norm and every GEMV ends in.
//
// Metal's `rms_reduce.h` is two functions over `simd_sum` and a threadgroup
// array. `kernels-vulkan` is the same two over `subgroupAdd`. This is the same
// two again, with one structural difference that is worth reading before
// touching anything here.
//
// ## Why the baseline is a tree and not a subgroup reduction
//
// `subgroupAdd` is `wgpu::Features::SUBGROUP`, and WebGPU does not guarantee
// it: a browser, and a fair number of native adapters, report it absent. So the
// BASELINE body is a shared-memory tree over the whole workgroup, which needs
// nothing, and the `@subgroup` tier replaces the inner level with
// `subgroupAdd`. Both are here, chosen by `PIE_SUBGROUP`, and both answer the
// same recurrence.
//
// A tree costs `log2(n)` barriers where a subgroup reduction costs one. That is
// the price of running everywhere, and it is why the tier exists.
//
// ## The array is sized for the widest workgroup, not for a subgroup count
//
// `pie_partials` holds one f32 per LANE of the widest workgroup this tree
// launches -- 256 -- rather than one per subgroup. WGSL sizes a
// `var<workgroup>` at compile time from a const-expression, and the subgroup
// width is not one: `subgroup_size` is a runtime builtin. Sizing by lane is
// what makes the array a constant, and it costs 1 KiB of a 16 KiB budget.
//
// The Vulkan tree's own note about this is the mirror image and still worth
// knowing: it sizes by `1024 / 8` because it CAN index by subgroup, and gets it
// wrong on Intel if it assumes 32.
//
// ## Both barriers are load-bearing
//
// The second one is the one that is easy to drop: without it, an invocation
// that reaches the next call of this function can overwrite a partial another
// invocation has not yet read. Every caller here reduces more than once per
// launch, so "it happened to work" is what a missing second barrier looks like
// until the workgroup is wide enough to straddle two scheduling waves.
//
// And `workgroupBarrier()` must sit in control flow UNIFORM across the
// workgroup. A caller that returns early from a tail guard before reducing has
// written a hang, not a wrong number. Every body in this tree guards its
// STORES rather than its entry for that reason.

// NO `enable subgroups;`, AND THAT IS WHY THIS ARM NEVER RAN. naga 30 refuses
// the enable-extension outright -- "specifies standard functionality which is
// not yet implemented in Naga" -- while parsing and lowering the subgroup
// builtins perfectly well, gating them on `wgpu::Features::SUBGROUP` instead.
// This file wrote the enable and no `pie:instantiate` line carried an
// `@subgroup` tag, so the arm below had never been compiled by anything until
// `sdpa_paged.wgsl` and `qmv.wgsl` found what a reduction level costs on Metal
// and came looking for it.
//
// # AND HAVING REPAIRED IT, NOTHING MINTS IT, WHICH IS A MEASUREMENT
//
// `rms_single_row_bfloat16` and `rms_rope_bfloat16` are the two callers worth
// the trouble -- 5.7% and 5.5% of a decode, both `grid=[1,1,1]`, one workgroup
// on a twenty-core GPU and therefore as latency-bound as anything in the tree.
// Both were minted `@subgroup` and both measured a TIE: 7.750/7.796/7.857 ms a
// token against 7.730 without them, three interleaved rounds, inside a
// repeatability of about 1.7%.
//
// So the mint was reverted and the arm is kept. That is not a contradiction of
// the "a level is what costs" finding -- it is its boundary. The two kernels
// that paid were paying per KEY and per output ROW, hundreds of ladders a
// launch; an rms runs ONE ladder for the whole dispatch, and one ladder is
// worth about what three barriers are worth. The arm is correct, compiles, and
// is here for the caller that runs a ladder in a loop.

// One slot per lane of the widest workgroup this tree launches.
const PIE_REDUCE_LANES = 256u;

var<workgroup> pie_partials: array<f32, PIE_REDUCE_LANES>;

// The workgroup's sum of `v`, broadcast to every invocation.
fn pie_workgroup_sum(lane: u32, lanes: u32, v: f32) -> f32 {
//#if defined(PIE_SUBGROUP)
    // One partial per subgroup, then EVERY lane folds them. `subgroup_size` is
    // a runtime value, so the fold is a loop over a bound the host cannot see
    // -- which is why the slots are cleared first: a lane that belongs to no
    // subgroup of this launch must contribute an identity, not stale memory.
    //
    // What this buys is the barrier count: THREE, against `log2(lanes) + 2` for
    // the ladder below, which is ten at the 256 lanes an rms launches. On this
    // machine a reduction LEVEL is what costs and the adds in it are not --
    // see `sdpa_paged.wgsl`, where deleting three levels of a 63-add tree
    // removed eleven percent of the adds and a third of the kernel.
    //
    // THE FOLD MUST STRIDE BY THE SUBGROUP WIDTH, and the first version of
    // this arm did not. Walking all `lanes` slots and adding mostly zeros
    // needs no width and looked free -- a few hundred cycles of a kernel
    // spending microseconds. It measured **+2.7 ms a token**, worse than the
    // ladder it replaced and worse by more than every subgroup arm in the tree
    // had won. 256 f32 adds out of workgroup memory are not a few hundred
    // cycles when each one depends on the last, and this kernel is a SINGLE
    // workgroup on a twenty-core GPU with nothing to hide the chain behind.
    //
    // `subgroupAdd(1.0)` is the width, and it needs no `subgroup_size` builtin
    // and therefore no change to any caller's entry point -- every lane of a
    // subgroup contributes one, so the sum IS the count of active lanes. The
    // fold then walks the `lanes / width` slots an elected lane actually
    // wrote, which is eight at 256 lanes instead of 256.
    let s = subgroupAdd(v);
    let width = max(1u, u32(subgroupAdd(1.0)));
    pie_partials[lane] = 0.0;
    workgroupBarrier();
    // `subgroupElect()` is what this used to say, and naga 30 does not have it
    // -- it lowers `subgroupAdd`, the shuffles and `subgroupBallot`, but the
    // elect is not an identifier in scope. `lane == subgroupBroadcastFirst(lane)`
    // is the same predicate built out of what IS there: the broadcast answers
    // the lowest ACTIVE lane's value, so exactly one lane compares equal, and
    // it needs no `subgroup_invocation_id` builtin and therefore no change to
    // any caller's entry point.
    if (lane == subgroupBroadcastFirst(lane)) {
        pie_partials[lane] = s;
    }
    workgroupBarrier();
    var total = 0.0;
    for (var i = 0u; i < lanes; i = i + width) {
        total = total + pie_partials[i];
    }
    workgroupBarrier();
    return total;
//#else
    pie_partials[lane] = v;
    workgroupBarrier();
    // A power-of-two tree over `lanes`, which is a power of two in every
    // launch this tree makes. `stride` starting at the next power of two below
    // `lanes` and the `lane + stride < lanes` guard together handle the case
    // where it is not, rather than reading a slot nobody wrote.
    var stride = lanes >> 1u;
    loop {
        if (stride == 0u) { break; }
        if (lane < stride && lane + stride < lanes) {
            pie_partials[lane] = pie_partials[lane] + pie_partials[lane + stride];
        }
        workgroupBarrier();
        stride = stride >> 1u;
    }
    let total = pie_partials[0];
    workgroupBarrier();
    return total;
//#endif
}

// The workgroup's maximum of `v`, broadcast. The online softmax's first half.
fn pie_workgroup_max(lane: u32, lanes: u32, v: f32) -> f32 {
//#if defined(PIE_SUBGROUP)
    let s = subgroupMax(v);
    let width = max(1u, u32(subgroupAdd(1.0)));
    // The identity for a max is not zero. A cleared slot that read 0.0 would
    // clamp every negative row to zero, which is a plausible number and so the
    // worst kind of wrong.
    pie_partials[lane] = -3.4028235e38;
    workgroupBarrier();
    if (lane == subgroupBroadcastFirst(lane)) {
        pie_partials[lane] = s;
    }
    workgroupBarrier();
    var total = -3.4028235e38;
    for (var i = 0u; i < lanes; i = i + width) {
        total = max(total, pie_partials[i]);
    }
    workgroupBarrier();
    return total;
//#else
    pie_partials[lane] = v;
    workgroupBarrier();
    var stride = lanes >> 1u;
    loop {
        if (stride == 0u) { break; }
        if (lane < stride && lane + stride < lanes) {
            pie_partials[lane] = max(pie_partials[lane], pie_partials[lane + stride]);
        }
        workgroupBarrier();
        stride = stride >> 1u;
    }
    let total = pie_partials[0];
    workgroupBarrier();
    return total;
//#endif
}

// `1 / sqrt(mean(x^2) + eps)` from a lane's partial sum of squares.
//
// `inverseSqrt` and not `1.0 / sqrt`: the two are not the same function.
// `inverseSqrt` is allowed a couple of ULP, the division form is correctly
// rounded, and Metal's side of this reads `precise::rsqrt`. Matching the
// sibling's precision is what makes a parity walk between backends mean
// something.
fn pie_inv_rms(lane: u32, lanes: u32, sum_sq: f32, axis_size: u32, eps: f32) -> f32 {
    let total = pie_workgroup_sum(lane, lanes, sum_sq);
    return inverseSqrt(total / f32(axis_size) + eps);
}
