#pragma once

// Workgroup reductions over a one-dimensional workgroup of `PIE_GROUP_X`
// lanes, a power of two. The baseline folds through shared memory, no
// subgroup operation, so the same source runs wherever WGSL does. The
// `subgroup` tier folds each subgroup first and then sums the subgroups'
// partials in order, any subgroup width.

groupshared float pie_partials[PIE_GROUP_X];

#if defined(PIE_SUBGROUP)

float pie_workgroup_sum(float v, uint lid) {
    float s = WaveActiveSum(v);
    uint width = WaveGetLaneCount();
    if (WaveGetLaneIndex() == 0u) {
        pie_partials[lid / width] = s;
    }
    GroupMemoryBarrierWithGroupSync();
    float total = 0.0;
    for (uint i = 0u; i < (PIE_GROUP_X + width - 1u) / width; i++) {
        total += pie_partials[i];
    }
    GroupMemoryBarrierWithGroupSync();
    return total;
}

float pie_workgroup_max(float v, uint lid) {
    float m = WaveActiveMax(v);
    uint width = WaveGetLaneCount();
    if (WaveGetLaneIndex() == 0u) {
        pie_partials[lid / width] = m;
    }
    GroupMemoryBarrierWithGroupSync();
    float top = pie_partials[0];
    for (uint i = 1u; i < (PIE_GROUP_X + width - 1u) / width; i++) {
        top = max(top, pie_partials[i]);
    }
    GroupMemoryBarrierWithGroupSync();
    return top;
}

#else

float pie_workgroup_sum(float v, uint lid) {
    pie_partials[lid] = v;
    GroupMemoryBarrierWithGroupSync();
    for (uint half = PIE_GROUP_X / 2; half > 0; half >>= 1) {
        if (lid < half) {
            pie_partials[lid] += pie_partials[lid + half];
        }
        GroupMemoryBarrierWithGroupSync();
    }
    float total = pie_partials[0];
    GroupMemoryBarrierWithGroupSync();
    return total;
}

float pie_workgroup_max(float v, uint lid) {
    pie_partials[lid] = v;
    GroupMemoryBarrierWithGroupSync();
    for (uint half = PIE_GROUP_X / 2; half > 0; half >>= 1) {
        if (lid < half) {
            pie_partials[lid] = max(pie_partials[lid], pie_partials[lid + half]);
        }
        GroupMemoryBarrierWithGroupSync();
    }
    float top = pie_partials[0];
    GroupMemoryBarrierWithGroupSync();
    return top;
}

#endif

float pie_inv_rms(float sum_sq, uint lid, uint axis_size, float eps) {
    return rsqrt(pie_workgroup_sum(sum_sq, lid) / float(axis_size) + eps);
}
