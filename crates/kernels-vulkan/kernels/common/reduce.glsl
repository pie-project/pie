// The row reduction every norm and every GEMV ends in.
//
// Metal's `rms_reduce.h` is two functions over `simd_sum` and a threadgroup
// array; this is the same two, and the shape is llama.cpp's: reduce inside a
// subgroup with `subgroupAdd`, land one partial per subgroup in shared memory,
// then have subgroup 0 reduce the partials.
//
// The subgroup width is NOT assumed. Metal can hard-code 32 because every Apple
// GPU is 32; a Vulkan shell runs on AMD (64), Intel (8/16/32) and NVIDIA (32),
// so the partial array is sized for the widest workgroup this tree launches
// divided by the NARROWEST subgroup, and `gl_NumSubgroups` is what the second
// stage iterates. Sizing it the other way -- one slot per lane of a 32-wide
// subgroup -- is the bug that only appears on Intel, where a 1024-thread
// workgroup has 128 subgroups and writes 96 slots past the array.
//
// `PIE_REDUCE_MAX_SUBGROUPS` is 128 for that reason: 1024 threads, the widest
// workgroup the tree asks for, over an 8-wide subgroup, the narrowest any
// desktop implementation reports.

#ifndef PIE_VULKAN_REDUCE_GLSL
#define PIE_VULKAN_REDUCE_GLSL

#extension GL_KHR_shader_subgroup_basic : require
#extension GL_KHR_shader_subgroup_arithmetic : require

#define PIE_REDUCE_MAX_SUBGROUPS 128

shared float pie_partials[PIE_REDUCE_MAX_SUBGROUPS];

/// The workgroup's sum of `v`, broadcast to every invocation.
///
/// Both barriers are load-bearing and the second one is the one that is easy to
/// drop: without it, an invocation that reaches the next call of this function
/// can overwrite a partial another invocation has not yet read. Every caller
/// here reduces more than once per launch, so "it happened to work" is what a
/// missing second barrier looks like until the workgroup is wide enough to
/// straddle two scheduling waves.
float pie_workgroup_sum(float v) {
    float s = subgroupAdd(v);
    if (subgroupElect()) {
        pie_partials[gl_SubgroupID] = s;
    }
    barrier();

    float total = 0.0;
    for (uint i = 0; i < gl_NumSubgroups; i++) {
        total += pie_partials[i];
    }
    barrier();
    return total;
}

/// The workgroup's maximum of `v`, broadcast. The online softmax's first half.
float pie_workgroup_max(float v) {
    float m = subgroupMax(v);
    if (subgroupElect()) {
        pie_partials[gl_SubgroupID] = m;
    }
    barrier();

    float total = pie_partials[0];
    for (uint i = 1; i < gl_NumSubgroups; i++) {
        total = max(total, pie_partials[i]);
    }
    barrier();
    return total;
}

/// `1 / sqrt(mean(x^2) + eps)` from a lane's partial sum of squares.
///
/// The reciprocal square root is spelled `inversesqrt` and not `1.0 / sqrt`
/// because the two are not the same function: `inversesqrt` is allowed 2 ULP,
/// the division form is correctly rounded, and Metal's side of this reads
/// `precise::rsqrt`. Matching the sibling's precision is what makes a parity
/// walk between the two backends mean something.
float pie_inv_rms(float sum_sq, uint axis_size, float eps) {
    float total = pie_workgroup_sum(sum_sq);
    return inversesqrt(total / float(axis_size) + eps);
}

#endif  // PIE_VULKAN_REDUCE_GLSL
