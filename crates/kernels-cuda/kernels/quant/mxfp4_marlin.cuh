//===-- mxfp4_marlin.cuh - the GPT-OSS repackers, as device text ---------===//
//
// Three `__global__`s -- two of them now templates -- and the row selector
// they share. `mxfp4_marlin.cu` is the three host entry points and the
// validation that throws before any of them launches; this file is what the
// GPU runs. There is exactly one definition of each, because two copies that
// agree today drift tomorrow and `norm/altup_aux` shipped a release proving
// it.
//
// # What the launchers were doing, and where it went
//
// All three are the same shape:
//
//     const int grid = (total + 255) / 256;
//     kernel<<<grid, 256, 0, stream>>>(...);
//
// which is `LaunchRule::Elementwise`. There is no `total == 0` guard because
// `validate_row_select` has already thrown on a zero row count -- the
// ahead-of-time path refuses the empty case louder than the rule does, and
// the rule's own `Ungeometric::Empty` covers the JIT path.
//
// # Why `Mxfp4RowSelect` became an `int`
//
// The enum lives in `mxfp4_marlin.hpp`, which is a HOST interface header --
// it includes `<cuda_runtime.h>` and is compiled by the host compiler for
// every caller. `new-horizon.md` §10.5's rule is that `.hpp` files do not
// convert, and NVRTC has no include path that could reach one anyway. A
// device header that RESTATED the enum would be a second definition of a
// three-value contract, which is the drift this whole split exists to
// prevent.
//
// So the kernels take the underlying type. `enum class Mxfp4RowSelect : int`
// has a fixed underlying type, so `static_cast<int>` is the identity on its
// value representation and the launcher's cast costs nothing. It is also what
// a row could state in any case: `runtime::args` marshals pointers, `I32`,
// `U32`, `F32` and `Usize`, and refuses everything else -- a row carrying
// `Ty::Mxfp4RowSelect` would be one no caller could ever fire.
//
// # Why two of them are templates and one is not
//
// The scale permutation and the row gather are LAYOUT transforms: they move
// payload without looking at it, so the payload is the natural type
// parameter and the fp16 variant of the row gather costs a row.
//
// `mxfp4_weight_to_gptq_w4` is not. Its `k_pack` arithmetic hard-codes eight
// 4-bit nibbles to a 32-bit word -- `source_k / 8`, `sizeof(uint32_t)` in the
// byte offset -- so a width parameter would be a lie that compiles. A
// `DeviceKernel` row needs exactly one type parameter and this kernel has no
// honest one, so it stays a plain `__global__` with no row and keeps its
// ahead-of-time launcher.
//
//===----------------------------------------------------------------------===//
#pragma once

#include "prelude/device.cuh"

namespace pie::quant {


/// `Mxfp4RowSelect`'s three values, as the underlying `int` the kernels see.
///
/// Named constants rather than bare literals so the mapping between this file
/// and `mxfp4_marlin.hpp` is one grep and not three.
constexpr int kRowSelectIdentity = 0;
constexpr int kRowSelectEven = 1;
constexpr int kRowSelectOdd = 2;

/// Which source row a destination row reads.
///
/// GPT-OSS stores a fused gate/up projection interleaved, so a
/// tensor-parallel slice of either half is every other row. `Identity` is the
/// unfused case.
__device__ __forceinline__ int select_row(int row, int mode) {
    switch (mode) {
        case kRowSelectIdentity:
            return row;
        case kRowSelectEven:
            return 2 * row;
        case kRowSelectOdd:
            return 2 * row + 1;
        default:
            break;
    }
    return row;
}

/// Row-major MXFP4 bytes into Marlin's `[target_k / 8, selected_rows]` int32
/// staging layout.
///
/// Out-of-range destinations are ZEROED rather than skipped: the staging
/// buffer is consumed whole by Marlin's repacker, and a padded row that kept
/// whatever was in the allocation would decode to weights.
__global__ void mxfp4_weight_to_gptq_w4(
    const u8* __restrict__ raw,
    u32* __restrict__ out,
    int source_rows,
    int source_row_offset,
    int selected_rows,
    int valid_rows,
    int source_stride_k,
    int source_col_offset,
    int source_k,
    int target_k,
    int row_select) {
    const int k_packs = target_k / 8;
    const int source_k_packs = source_k / 8;
    const usize total = static_cast<usize>(k_packs) * static_cast<usize>(selected_rows);
    const usize idx = static_cast<usize>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    const int k_pack = static_cast<int>(idx / selected_rows);
    const int dst_row =
        static_cast<int>(idx - static_cast<usize>(k_pack) * selected_rows);
    const int logical_row = source_row_offset + dst_row;
    const int src_row = select_row(logical_row, row_select);
    if (dst_row >= valid_rows || src_row < 0 || src_row >= source_rows ||
        k_pack >= source_k_packs) {
        out[idx] = 0;
        return;
    }

    const usize row_stride_bytes = static_cast<usize>(source_stride_k) / 2;
    const auto* src = reinterpret_cast<const u32*>(
        raw + static_cast<usize>(src_row) * row_stride_bytes +
        static_cast<usize>(source_col_offset) / 2 +
        static_cast<usize>(k_pack) * sizeof(u32));
    out[idx] = *src;
}

/// Raw E8M0 block scales into Marlin's `[target_groups, selected_rows]`
/// layout, including the 64-wide scale permutation and the MXFP4 four-lane
/// post-permutation vLLM and SGLang both apply.
///
/// The index arithmetic runs BACKWARDS -- each thread owns one output slot
/// and inverts both permutations to find its source -- because a forward
/// scatter would need the output to be written by whichever thread happened
/// to own the input, and the two permutations do not compose into a stride.
template <class T>
__global__ void mxfp4_scales_to_marlin_e8m0(
    const T* __restrict__ raw,
    T* __restrict__ out,
    int source_rows,
    int source_row_offset,
    int selected_rows,
    int valid_rows,
    int source_stride_groups,
    int source_group_offset,
    int source_groups,
    int target_groups,
    int row_select) {
    const usize total =
        static_cast<usize>(target_groups) * static_cast<usize>(selected_rows);
    const usize out_idx = static_cast<usize>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (out_idx >= total) return;

    // Inverse of mxfp4_marlin_process_scales: view(-1, 4)[:, [0, 2, 1, 3]].
    const int lane4 = static_cast<int>(out_idx & 3);
    const usize base4 = out_idx & ~usize{3};
    const int pre_lane4 = (lane4 == 1) ? 2 : ((lane4 == 2) ? 1 : lane4);
    const usize after_marlin = base4 + static_cast<usize>(pre_lane4);

    // Inverse of marlin_permute_scales' 64-wide transpose.
    const usize block64 = after_marlin & ~usize{63};
    const int tid64 = static_cast<int>(after_marlin & 63);
    const int src64 = (tid64 % 8) * 8 + (tid64 / 8);
    const usize transposed_idx = block64 + static_cast<usize>(src64);

    const int group = static_cast<int>(transposed_idx / selected_rows);
    const int dst_row =
        static_cast<int>(transposed_idx - static_cast<usize>(group) * selected_rows);
    if (group < 0 || group >= target_groups) return;
    const int logical_row = source_row_offset + dst_row;
    const int src_row = select_row(logical_row, row_select);
    if (dst_row >= valid_rows || src_row < 0 || src_row >= source_rows ||
        group >= source_groups) {
        out[out_idx] = T{};
        return;
    }

    out[out_idx] = raw[static_cast<usize>(src_row) *
                           static_cast<usize>(source_stride_groups) +
                       static_cast<usize>(source_group_offset + group)];
}

/// A `[batch, source_rows]` table gathered down to `[batch, selected_rows]`.
///
/// The payload is never interpreted, only moved, which is why it is the type
/// parameter -- the bias vector this gathers is bf16 today and an fp16 build
/// costs a row rather than a kernel.
template <class T>
__global__ void row_map_to_dense(
    const T* __restrict__ raw,
    T* __restrict__ out,
    int batch,
    int source_rows,
    int source_row_offset,
    int selected_rows,
    int valid_rows,
    int row_select) {
    const usize total = static_cast<usize>(batch) * static_cast<usize>(selected_rows);
    const usize idx = static_cast<usize>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    const int b = static_cast<int>(idx / selected_rows);
    const int dst_row = static_cast<int>(idx - static_cast<usize>(b) * selected_rows);
    const int logical_row = source_row_offset + dst_row;
    const int src_row = select_row(logical_row, row_select);
    if (dst_row >= valid_rows || src_row < 0 || src_row >= source_rows) {
        out[idx] = T{};
        return;
    }
    out[idx] = raw[static_cast<usize>(b) * static_cast<usize>(source_rows) +
                   static_cast<usize>(src_row)];
}

}  // namespace pie::quant
