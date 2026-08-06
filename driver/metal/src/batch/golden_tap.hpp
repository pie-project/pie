#pragma once
// golden_tap.hpp — env-gated per-kernel activation dump for the accuracy gate.
//
// The MLX reference under tests/mlx writes `<PIE_METAL_GOLDEN_DIR>/<layer>.<kernel>.npy`
// for every tapped intermediate (see qwen3_5.cpp's dump_kernel). This is the raw-Metal
// counterpart: identical file names, identical shapes, so the two trees can be diffed
// tap by tap and the FIRST diverging (kernel, layer) named.
//
// It is off unless PIE_METAL_GOLDEN_DIR names a directory. When on, the scratch
// schedule switches to `no_recycle` (every activation value gets its own pool buffer,
// so nothing is overwritten before the dump) and the pool is allocated CPU-visible.
// Both are diagnostic-only; the shipped path is untouched.

#include <string>
#include <vector>

#include "decode_abi.hpp"
#include "decode_step.hpp"
#include "scratch.hpp"

namespace pie::metal {

// The directory taps are written to, or empty when the dump is off.
const std::string& golden_tap_dir();
inline bool golden_taps_enabled() { return !golden_tap_dir().empty(); }

/// Whether a tap dump keeps the ordinary scratch recycling.
///
/// A dump normally gives every value its own pool buffer, because a recycled
/// producer's buffer holds someone else's value by the time the fire retires
/// and there would be nothing to read. The cost is that the taps then describe
/// an allocation no fire outside the dump ever runs against -- so a colouring
/// that reuses a buffer while its value is still live is the ONE class of
/// defect this dump structurally cannot see. `PIE_METAL_TAPS_RECYCLE=1` dumps
/// against the real allocation instead: values whose buffer was reused read as
/// whatever overwrote them, which is exactly what makes the two dumps
/// comparable and the first diverging tap the answer.
bool golden_taps_recycle();

// Write every tapped activation of `dag` as `<dir>/<layer>.<kernel>.npy`, float32,
// shape [n_rows, width]. Row t is read at `t * row_stride_bytes` inside its pool slot,
// which is how bind_scratch lays the per-token prefill rows out.
void dump_golden_taps(const std::vector<Dispatch>& dag,
                      const ScratchSchedule& sched,
                      const SlotHandle* pool,
                      int pool_n,
                      const DecodeGeometry& g,
                      int n_rows,
                      std::size_t row_stride_bytes);

// Write one already-materialized bf16 tensor (the lm_head logits live in IO, not
// scratch, so they never pass through the schedule).
void dump_golden_bf16(const std::string& name,
                      const void* bf16,
                      int rows,
                      int width,
                      std::size_t row_stride_elems);

// Write one bf16 tensor that the expert sort REORDERED, restoring the layout it
// had before the sort: `[rows, slots * width]`, slot-major, which is what every
// reference and every bisect already speaks.
//
// `perm` is the sort's own output -- its within-expert order is decided by
// atomics and is not reproducible on the host, so it is read rather than
// recomputed. `perm[p]` is the (token, slot) pair stored at row `p`, or -1 for
// a padding row. Pairs the sort never wrote stay zero.
void dump_golden_bf16_sorted(const std::string& name,
                             const void* bf16,
                             const std::int32_t* perm,
                             int stored_rows,
                             int rows,
                             int slots,
                             int width);

// The exact token ids this pass ran, so the reference can be regenerated on them.
void dump_golden_tokens(const std::uint32_t* ids, int n);

}  // namespace pie::metal
