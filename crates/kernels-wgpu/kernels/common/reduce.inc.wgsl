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

//#if defined(PIE_SUBGROUP)
enable subgroups;
//#endif

// One slot per lane of the widest workgroup this tree launches.
const PIE_REDUCE_LANES = 256u;

var<workgroup> pie_partials: array<f32, PIE_REDUCE_LANES>;

// The workgroup's sum of `v`, broadcast to every invocation.
fn pie_workgroup_sum(lane: u32, lanes: u32, v: f32) -> f32 {
//#if defined(PIE_SUBGROUP)
    // One partial per subgroup, then lane 0 folds them. `subgroup_size` is a
    // runtime value, so the fold is a loop over a bound the host cannot see --
    // which is why the slots are cleared first: a lane that belongs to no
    // subgroup of this launch must contribute an identity, not stale memory.
    let s = subgroupAdd(v);
    pie_partials[lane] = 0.0;
    workgroupBarrier();
    if (subgroupElect()) {
        pie_partials[lane] = s;
    }
    workgroupBarrier();
    var total = 0.0;
    for (var i = 0u; i < lanes; i = i + 1u) {
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
    // The identity for a max is not zero. A cleared slot that read 0.0 would
    // clamp every negative row to zero, which is a plausible number and so the
    // worst kind of wrong.
    pie_partials[lane] = -3.4028235e38;
    workgroupBarrier();
    if (subgroupElect()) {
        pie_partials[lane] = s;
    }
    workgroupBarrier();
    var total = -3.4028235e38;
    for (var i = 0u; i < lanes; i = i + 1u) {
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
