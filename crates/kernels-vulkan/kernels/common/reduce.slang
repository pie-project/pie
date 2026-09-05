#ifndef PIE_VULKAN_REDUCE_SLANG
#define PIE_VULKAN_REDUCE_SLANG

#define PIE_REDUCE_MAX_SUBGROUPS 128

groupshared float pie_partials[PIE_REDUCE_MAX_SUBGROUPS];

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

float pie_inv_rms(float sum_sq, uint axis_size, float eps) {
    float total = pie_workgroup_sum(sum_sq);
    return rsqrt(total / float(axis_size) + eps);
}

#endif
