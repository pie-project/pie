#pragma once

#include "prelude/device.cuh"

// Bounded replay's floors (DeepSeek-V4.1's Causal Encoder-Decoder): a
// lane's least position whose window keys its decoder layers hold. A block
// a lane. With `mark` the lane's rows are a prompt's encoder-only ones: its
// state takes the position past its last row. Without, every row reads the
// floor its lane's state holds (0 for a sequence that never skipped rows,
// its state zeroed where it began).
namespace pie::attn {

__global__ void replay_floor(
    const int* __restrict__ positions,
    int* __restrict__ state_base,
    const int* __restrict__ slot_ids,
    const u32* __restrict__ qo_indptr,
    long long slot_stride_elems,
    int* __restrict__ floors,
    int mark,
    bool write_state,
    const u8* __restrict__ write_state_mask,
    const u32* __restrict__ win)
{
    const int r = blockIdx.x;
    if (win != nullptr && r >= static_cast<int>(win[2])) return;
    const int rl = win != nullptr ? r + static_cast<int>(win[3]) : r;
    const int row0 = win != nullptr ? static_cast<int>(win[1]) : 0;
    const int t0 = static_cast<int>(qo_indptr[r]) + row0;
    const int n = static_cast<int>(qo_indptr[r + 1]) - static_cast<int>(qo_indptr[r]);
    if (n <= 0) return;
    const int slot = slot_ids[rl];
    int* state = slot >= 0 ? state_base + static_cast<long long>(slot) * slot_stride_elems : nullptr;
    const int floor = mark ? positions[t0 + n - 1] + 1 : (state != nullptr ? state[0] : 0);
    for (int t = static_cast<int>(threadIdx.x); t < n; t += static_cast<int>(blockDim.x)) {
        floors[t0 + t] = floor;
    }
    if (mark && state != nullptr && write_state &&
        (write_state_mask == nullptr || write_state_mask[rl] != 0) && threadIdx.x == 0) {
        state[0] = floor;
    }
}

}  // namespace pie::attn
