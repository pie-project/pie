// The row reduction every norm and every GEMV ends in. The Slang half of
// `common/reduce.glsl`.
//
// Everything the GLSL header says about the SHAPE still holds and is not
// repeated here: reduce inside a subgroup, land one partial per subgroup in
// shared memory, have the rest of the workgroup fold the partials; the subgroup
// width is not assumed, and `PIE_REDUCE_MAX_SUBGROUPS` is 128 because 1024
// threads over an 8-wide Intel subgroup is 128 of them and sizing it per-lane
// instead is the bug that only appears on Intel. Read that file for the
// reasoning; this one records only what the change of language forced.
//
// ## What the change of language forced
//
// The subgroup calls become wave calls: `subgroupAdd` is `WaveActiveSum`,
// `subgroupMax` is `WaveActiveMax`, `subgroupElect` is `WaveIsFirstLane`, and
// `barrier` is `GroupMemoryBarrierWithGroupSync`. Same instructions, same
// `GroupNonUniformArithmetic` capability, same `OpControlBarrier`.
//
// Two things do NOT become wave calls, because Slang has nothing to become.
// `gl_SubgroupID` -- which subgroup of this workgroup am I -- and
// `gl_NumSubgroups` have no HLSL or Slang spelling; HLSL's wave model has no
// notion of a workgroup's subgroups at all. They could be computed as
// `groupIndex / WaveGetLaneCount()` and `(threads + width - 1) / width`, and
// that would be a DIFFERENT program: it assumes the implementation packs lanes
// into waves in index order and fills every wave but the last, neither of which
// Vulkan promises. So the two builtins stay, which Slang accepts under
// `-allow-glsl`, and they compile to the same `SubgroupId` / `NumSubgroups`
// builtins `glslc` emitted.
//
// The `#extension` lines are gone. Slang derives the SPIR-V capabilities from
// the intrinsics used, so `GL_KHR_shader_subgroup_basic` and
// `_arithmetic` are requested by `WaveActiveSum` existing rather than by a
// line at the top that it is possible to forget to update.

#ifndef PIE_VULKAN_REDUCE_SLANG
#define PIE_VULKAN_REDUCE_SLANG

#define PIE_REDUCE_MAX_SUBGROUPS 128

groupshared float pie_partials[PIE_REDUCE_MAX_SUBGROUPS];

/// The workgroup's sum of `v`, broadcast to every invocation.
///
/// Both barriers are load-bearing and the second one is the one that is easy to
/// drop: without it, an invocation that reaches the next call of this function
/// can overwrite a partial another invocation has not yet read. Every caller
/// here reduces more than once per launch, so "it happened to work" is what a
/// missing second barrier looks like until the workgroup is wide enough to
/// straddle two scheduling waves.
float pie_workgroup_sum(float v) {
    float s = WaveActiveSum(v);
    if (WaveIsFirstLane()) {
        pie_partials[gl_SubgroupID] = s;
    }
    GroupMemoryBarrierWithGroupSync();

    float total = 0.0;
    for (uint i = 0; i < gl_NumSubgroups; i++) {
        total += pie_partials[i];
    }
    GroupMemoryBarrierWithGroupSync();
    return total;
}

/// The workgroup's maximum of `v`, broadcast. The online softmax's first half.
float pie_workgroup_max(float v) {
    float m = WaveActiveMax(v);
    if (WaveIsFirstLane()) {
        pie_partials[gl_SubgroupID] = m;
    }
    GroupMemoryBarrierWithGroupSync();

    float total = pie_partials[0];
    for (uint i = 1; i < gl_NumSubgroups; i++) {
        total = max(total, pie_partials[i]);
    }
    GroupMemoryBarrierWithGroupSync();
    return total;
}

/// `1 / sqrt(mean(x^2) + eps)` from a lane's partial sum of squares.
///
/// The reciprocal square root is spelled `rsqrt` and not `1.0 / sqrt` because
/// the two are not the same function: the reciprocal-square-root instruction is
/// allowed 2 ULP, the division form is correctly rounded, and Metal's side of
/// this reads `precise::rsqrt`. Matching the sibling's precision is what makes
/// a parity walk between the two backends mean something. Slang lowers `rsqrt`
/// to the same `GLSL.std.450 InverseSqrt` that GLSL's `inversesqrt` did.
float pie_inv_rms(float sum_sq, uint axis_size, float eps) {
    float total = pie_workgroup_sum(sum_sq);
    return rsqrt(total / float(axis_size) + eps);
}

#endif  // PIE_VULKAN_REDUCE_SLANG
