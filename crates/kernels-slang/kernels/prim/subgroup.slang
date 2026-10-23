#pragma once

// The invocation's subgroup and the workgroup's count of them, from its index
// and the lane count: Adreno reads the SubgroupId builtin as 0 in every one.
// SPIR-V's subgroup tier.

uint pie_subgroup_id() {
    return gl_LocalInvocationIndex / WaveGetLaneCount();
}

uint pie_num_subgroups() {
    uint3 g = WorkgroupSize();
    return (g.x * g.y * g.z + WaveGetLaneCount() - 1u) / WaveGetLaneCount();
}
