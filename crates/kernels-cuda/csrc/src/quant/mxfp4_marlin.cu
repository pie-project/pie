//===-- mxfp4_marlin.cu - the ahead-of-time entry points -----------------===//
//
// Three launchers, the validation they share, and no device text. Every
// `__global__` this file fires lives in `mxfp4_marlin.cuh`, which the JIT
// compiles from the same bytes -- see the header for why the split exists,
// what each `<<<>>>` became, why the row selector crosses as an `int`, and
// why one of the three has no row.
//
// `validate_row_select` stays here because it THROWS. A `std::runtime_error`
// is host machinery -- NVRTC has no C++ standard library and no exceptions --
// and refusing a bad shape before the launch is the launcher's job, not the
// kernel's.
//
//===----------------------------------------------------------------------===//
#include "quant/mxfp4_marlin.hpp"

#include "quant/mxfp4_marlin.cuh"

#include <cstddef>
#include <stdexcept>
#include <string>

namespace pie_cuda_driver::kernels::quant {

namespace {

constexpr int BLOCK = 256;

void validate_row_select(
    const char* op,
    int source_rows,
    int source_row_offset,
    int selected_rows,
    int valid_rows,
    Mxfp4RowSelect row_select)
{
    if (source_rows <= 0 || selected_rows <= 0 || valid_rows <= 0 ||
        valid_rows > selected_rows || source_row_offset < 0) {
        throw std::runtime_error(std::string(op) + ": row counts must be positive");
    }
    const long long logical_end =
        static_cast<long long>(source_row_offset) + valid_rows;
    const long long required = row_select == Mxfp4RowSelect::Identity
        ? logical_end
        : logical_end * 2;
    if (required > source_rows) {
        throw std::runtime_error(
            std::string(op) + ": row offset exceeds source row table");
    }
}

}  // namespace

// `mxfp4_weight_to_gptq_w4` was deleted here by §43: no shim entry, no row,
// no C++ caller, no `ffi::` fire, no golden. The `__global__` stays in
// `quant/mxfp4_marlin.cuh`, where the header records why it carries no launch
// rule -- its `k_pack` arithmetic hard-codes eight -- and `attn`'s
// `pack_dense_mask.cuh` still cites that refusal by name.

void mxfp4_scales_to_marlin_e8m0(
    const void* raw_e8m0,
    void*       marlin_e8m0,
    int         source_rows,
    int         source_row_offset,
    int         selected_rows,
    int         valid_rows,
    int         source_stride_groups,
    int         source_group_offset,
    int         source_groups,
    int         target_groups,
    Mxfp4RowSelect row_select,
    cudaStream_t stream)
{
    validate_row_select(
        "mxfp4_scales_to_marlin_e8m0",
        source_rows, source_row_offset, selected_rows, valid_rows, row_select);
    if (source_groups <= 0 || target_groups <= 0 ||
        source_stride_groups <= 0 || source_group_offset < 0 ||
        target_groups < source_groups ||
        static_cast<long long>(source_group_offset) + source_groups >
            source_stride_groups) {
        throw std::runtime_error(
            "mxfp4_scales_to_marlin_e8m0: source/target groups, "
            "stride, and group offset must be positive; target groups must "
            "cover source groups; and the source slice must fit in stride");
    }
    const std::size_t total =
        static_cast<std::size_t>(target_groups) *
        static_cast<std::size_t>(selected_rows);
    if (total % 64 != 0) {
        throw std::runtime_error(
            "mxfp4_scales_to_marlin_e8m0: scale layout requires total "
            "scale count divisible by 64");
    }
    const int grid = static_cast<int>((total + BLOCK - 1) / BLOCK);
    device::mxfp4_scales_to_marlin_e8m0<device::u8><<<grid, BLOCK, 0, stream>>>(
        static_cast<const device::u8*>(raw_e8m0),
        static_cast<device::u8*>(marlin_e8m0),
        source_rows, source_row_offset, selected_rows, valid_rows,
        source_stride_groups, source_group_offset, source_groups,
        target_groups, static_cast<int>(row_select));
}

// `bf16_row_map_to_dense` was deleted here by §43. Its row survives it:
// `families::quant` states it as a device row over `quant/mxfp4_marlin.cuh`,
// so NVRTC still compiles the kernel and only the ahead-of-time launcher
// went (§10.10 step 5).

}  // namespace pie_cuda_driver::kernels::quant
