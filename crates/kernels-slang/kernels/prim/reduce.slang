#pragma once

// Workgroup reductions through shared memory: no subgroup operation, so the
// same source runs wherever WGSL does. `PIE_GROUP_X` is the workgroup width,
// a power of two.

groupshared float pie_partials[PIE_GROUP_X];

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

float pie_inv_rms(float sum_sq, uint lid, uint axis_size, float eps) {
    return rsqrt(pie_workgroup_sum(sum_sq, lid) / float(axis_size) + eps);
}
