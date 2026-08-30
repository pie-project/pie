#pragma once

#include "prelude/device.cuh"

namespace pie::layout {

/// **THE SCATTER THAT LETS A ROW HAVE NO DESTINATION**
/// (`.wiki/alto/multimodal.md` §8.6).
///
/// `scatter_rows`' body plus one comparison. `index[n] < 0` means row `n` of
/// `src` is not placed; every other entry is the destination row it is in the
/// unguarded twin.
///
/// **WHY THE COMPARISON IS OWED AT ALL.** A compacting fold — `pool_rows`,
/// `merge_rows` — writes `rows / side²` rows and leaves the rest of the
/// rectangle as whatever the arena held. `patch_routes` is `[Dim::Patches]`,
/// one destination per row of the FULL rectangle, so those tail rows have
/// entries too, and there was no legal way to spell "nowhere": the shell
/// refuses `route < 0` by name and the unguarded kernel would take a negative
/// index as a device write below the base of the rectangle. `-1` is the
/// spelling `AdapterRoutes` already uses for "no bank"; this reads it as "no
/// row".
///
/// **AND THE GUARD IS A RETURN AND NOT A PREDICATED STORE**, so a dropped row
/// costs one load of one integer and the whole block exits — which is what
/// makes the tail of a folded rectangle cost the copy nothing rather than a
/// quarter of it.
template <class U>
__global__ void scatter_live_rows(
    const U* __restrict__ tight,
    U* __restrict__ wide,
    const i32* __restrict__ index,
    int units)
{
    const int n = static_cast<int>(blockIdx.x);
    const i32 at = index[n];
    if (at < 0) return;

    const U* src = tight + static_cast<long long>(n) * units;
    U* dst = wide + static_cast<long long>(at) * units;
    for (int i = static_cast<int>(threadIdx.x); i < units;
         i += static_cast<int>(blockDim.x)) {
        dst[i] = src[i];
    }
}

}
