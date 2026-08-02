#pragma once

// Region support: the host-side analysis and validation the CUDA launch path
// performs on a decoded plan.
//
// This is what survived the in-driver emitters (`singleton_codegen.hpp` and
// `fused_codegen.hpp`, deleted in the compiler/ consolidation). Those files
// were pure string builders and their output now comes from the host
// (`compiler/codegen`, shipped in `PieProgramDesc::emitted_kernels`); the
// predicates and the packer analysis below are NOT emission and are still
// called at bind and at launch:
//
//   * `GeneratedValueDesc` / `GeneratedOpParams` — the device-side ABI the
//     host packer in `fused_runtime.cuh` fills and the emitted kernel reads.
//   * `second_party_region_supported`, `validate_generated_region`,
//     `detail::library_region_valid` — bind-time gates.
//   * `analyze_direct_argmax` — the launch packer's intrinsic side-table
//     analysis.
//
// Under the north star these all become launch-package data and this header
// goes away; until then it is the only place they live.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <optional>
#include <sstream>
#include <utility>
#include <string>
#include <type_traits>
#include <vector>

#include "pie/driver/launch/op_table.hpp"
#include "pie/driver/launch/plan.hpp"
#include <rng_contract.generated.h>

namespace pie_cuda_driver::pipeline::generated {

struct GeneratedStatus {
    std::uint32_t state;
    std::uint32_t fault;
    std::uint32_t reserved0;
    std::uint32_t reserved1;
};

struct GeneratedValueDesc {
    std::uint32_t len;
    std::uint32_t rows;
    std::uint32_t last;
    std::uint32_t rank;
    std::uint32_t dtype;
    std::uint32_t dims[4];
};

enum class IntrinsicStorageMode : std::uint32_t {
    F32 = 0,
    RawBf16 = 1,
};

enum class BoolStorageMode : std::uint32_t {
    NativeBytes = 0,
    WirePacked = 1,
    Unpacked = NativeBytes,
    Packed = WirePacked,
};

struct GeneratedOpParams {
    std::uint32_t tag;
    std::uint32_t a0;
    std::uint32_t a1;
    std::uint32_t a2;
    std::uint32_t o0;
    std::uint32_t o1;
    std::uint32_t imm;
    std::uint32_t imm2;
    std::uint32_t imm3;
    std::uint32_t kind;
    std::uint32_t pred_tag;
    std::uint32_t lit_dtype;
    std::uint32_t lit_bits;
    std::uint32_t channel_slot;
    std::uint32_t intr;
    std::uint32_t sink_bytes;
    std::uint32_t intrinsic_dtype;
    std::uint32_t bool_storage;
    std::uint32_t intrinsic_row_stride;
    std::uint32_t intrinsic_row_offset;
    std::uint64_t rng_seed;
};

struct GeneratedKernelSource {
    bool ok = false;
    std::string error;
    std::string entry_name;
    std::string source;
    std::uint8_t op_tag = 0;
};

using M1Status = GeneratedStatus;
using M1ValueDesc = GeneratedValueDesc;
using M1OpParams = GeneratedOpParams;
using SingletonSource = GeneratedKernelSource;

static_assert(std::is_standard_layout_v<GeneratedStatus>);
static_assert(std::is_trivial_v<GeneratedStatus>);
static_assert(sizeof(GeneratedStatus) == 16);
static_assert(alignof(GeneratedStatus) == 4);
static_assert(offsetof(GeneratedStatus, reserved1) == 12);
static_assert(std::is_standard_layout_v<GeneratedValueDesc>);
static_assert(std::is_trivial_v<GeneratedValueDesc>);
static_assert(sizeof(GeneratedValueDesc) == 36);
static_assert(alignof(GeneratedValueDesc) == 4);
static_assert(offsetof(GeneratedValueDesc, dims) == 20);
static_assert(std::is_standard_layout_v<GeneratedOpParams>);
static_assert(std::is_trivial_v<GeneratedOpParams>);
static_assert(sizeof(GeneratedOpParams) == 88);
static_assert(alignof(GeneratedOpParams) == 8);
static_assert(offsetof(GeneratedOpParams, sink_bytes) == 60);
static_assert(offsetof(GeneratedOpParams, intrinsic_dtype) == 64);
static_assert(offsetof(GeneratedOpParams, bool_storage) == 68);
static_assert(offsetof(GeneratedOpParams, intrinsic_row_stride) == 72);
static_assert(offsetof(GeneratedOpParams, intrinsic_row_offset) == 76);
static_assert(offsetof(GeneratedOpParams, rng_seed) == 80);

namespace detail {

inline constexpr bool supported_tag(std::uint8_t tag) {
    switch (tag) {
#define PTIR_CUDA_SINGLETON_TAG(name, value, arity, results) case value:
        PTIR_OP_LIST(PTIR_CUDA_SINGLETON_TAG)
#undef PTIR_CUDA_SINGLETON_TAG
            return true;
        default:
            return false;
    }
}

inline bool same_type(
    const pie::driver::launch::plan::ValueType& left,
    const pie::driver::launch::plan::ValueType& right) {
    return left.dtype == right.dtype &&
        left.dims.size() == right.dims.size() &&
        std::equal(
            left.dims.begin(),
            left.dims.end(),
            right.dims.begin(),
            [](const auto& a, const auto& b) {
                return a.symbolic == b.symbolic && a.value == b.value;
            });
}

inline bool nucleus_library_region_valid(
    const pie::driver::launch::plan::StagePlan& stage,
    const pie::driver::launch::plan::Region& region) {
    if (!region.library ||
        region.library_op != PTIR_LIBRARY_NUCLEUS_SAMPLE ||
        region.schedule != PTIR_SCHEDULE_LIBRARY ||
        (region.inputs.size() != 3 && region.inputs.size() != 5) ||
        region.nodes.size() != 13 ||
        region.outputs.size() != 1 || !region.sinks.empty() ||
        std::any_of(
            region.inputs.begin(),
            region.inputs.end(),
            [&](std::uint32_t value) {
                return value >= stage.value_types.size();
            }) ||
        region.outputs[0] >= stage.value_types.size()) {
        return false;
    }
    const bool scaled = region.inputs.size() == 5;
    const auto& raw_logits_type =
        stage.value_types[region.inputs[0]];
    const auto& logits_type = stage.value_types[
        region.inputs[scaled ? 2 : 0]];
    const auto& scale_type = stage.value_types[
        region.inputs[scaled ? 1 : 0]];
    const auto& top_p_type = stage.value_types[
        region.inputs[scaled ? 3 : 1]];
    const auto& state_type = stage.value_types[
        region.inputs[scaled ? 4 : 2]];
    const auto& output_type = stage.value_types[region.outputs[0]];
    auto same_dims = [](const auto& left, const auto& right) {
        return left.size() == right.size() &&
            std::equal(
                left.begin(),
                left.end(),
                right.begin(),
                [](const auto& a, const auto& b) {
                    return a.symbolic == b.symbolic &&
                        a.value == b.value;
                });
    };
    if (logits_type.dtype != PTIR_DT_F32 ||
        logits_type.dims.empty() || logits_type.dims.size() > 2) {
        return false;
    }
    if (raw_logits_type.dtype != PTIR_DT_F32 ||
        raw_logits_type.dims.empty() ||
        raw_logits_type.dims.back().symbolic !=
            logits_type.dims.back().symbolic ||
        raw_logits_type.dims.back().value !=
            logits_type.dims.back().value) {
        return false;
    }
    const std::vector<pie::driver::launch::plan::Dimension> row_dims(
        logits_type.dims.begin(), logits_type.dims.end() - 1);
    return top_p_type.dtype == PTIR_DT_F32 &&
        (top_p_type.dims.empty() ||
         same_dims(top_p_type.dims, row_dims)) &&
        (!scaled ||
         (scale_type.dtype == PTIR_DT_F32 &&
          (scale_type.dims.empty() ||
           same_dims(scale_type.dims, row_dims)))) &&
        state_type.dtype == PTIR_DT_U32 &&
        state_type.dims.size() == 1 &&
        !state_type.dims[0].symbolic &&
        state_type.dims[0].value == 2 &&
        output_type.dtype == PTIR_DT_I32 &&
        same_dims(output_type.dims, row_dims);
}

inline bool library_region_valid(
    const pie::driver::launch::plan::StagePlan& stage,
    const pie::driver::launch::plan::Region& region) {
    if (!region.library) {
        return region.schedule != PTIR_SCHEDULE_LIBRARY;
    }
    if (region.schedule != PTIR_SCHEDULE_LIBRARY) return false;
    if (region.library_op == PTIR_LIBRARY_NUCLEUS_SAMPLE) {
        return nucleus_library_region_valid(stage, region);
    }
    if (region.nodes.size() != 1 || region.nodes[0] >= stage.ops.size()) {
        return false;
    }
    const std::uint8_t tag = stage.ops[region.nodes[0]].op.tag;
    switch (region.library_op) {
        case PTIR_LIBRARY_TOP_K:
            return tag == PTIR_OP_TOP_K;
        case PTIR_LIBRARY_SORT:
            return tag == PTIR_OP_SORT_DESC;
        case PTIR_LIBRARY_SCAN:
            return tag == PTIR_OP_CUMSUM || tag == PTIR_OP_CUMPROD;
        case PTIR_LIBRARY_MATMUL:
            return tag == PTIR_OP_MATMUL;
        case PTIR_LIBRARY_SECOND_PARTY:
            return tag == PTIR_OP_KERNEL_CALL ||
                tag == PTIR_OP_SINK_CALL;
        default:
            return false;
    }
}

inline bool valid_identifier(const std::string& name) {
    if (name.empty()) return false;
    auto alpha = [](unsigned char value) {
        return (value >= 'A' && value <= 'Z') ||
            (value >= 'a' && value <= 'z') || value == '_';
    };
    auto digit = [](unsigned char value) {
        return value >= '0' && value <= '9';
    };
    if (!alpha(static_cast<unsigned char>(name.front()))) return false;
    return std::all_of(
        name.begin() + 1,
        name.end(),
        [&](char value) {
            const auto byte = static_cast<unsigned char>(value);
            return alpha(byte) || digit(byte);
        });
}

}  // namespace detail

// The three per-region analyses that used to live here -- the bind gates
// `second_party_region_supported` and `validate_generated_region`, and the
// intrinsic side-table analysis `analyze_direct_argmax` -- are gone
// (`ptir-refactor.md` §4.2, fields 4 and 5). They were a second
// implementation of decisions `compiler/codegen` had already made to emit the
// kernel that consumes them, and the two could disagree without failing to
// compile: the kernel would read a side-table slot the packer never wrote.
// The host now ships its answers as `PieProgramDesc::region_analysis` and the
// driver carries them on `FusedStageExecutable::region_analysis`. They were
// deleted at `region_divergent == 0` with `region_host_supplied != 0`, over
// the vendored corpus and the curated e2e matrix (`824421813`).

inline constexpr std::uint16_t kCudaGeneratedEmitterVersion = 19;

// Per-lane stride of the intrinsic side tables (bases / modes / widths /
// strides / offsets), in slots. Indexed by `IntrinsicId`, so it must be one
// past the largest id -- an id that overflows this stride does not fault, it
// silently reads the NEXT lane's slot 0. Both the host packer
// (`fused_runtime.cuh`) and the emitted device source below index with it, and
// `module_cache.hpp` keys the disk cache on `kCudaGeneratedEmitterVersion`, so
// widening it must bump that version or stale cubins keep the old stride.
//
// Projected from `ptir_abi.h`, which the compiler generates from the same
// `declare_intrinsics!` rows the enum comes from -- the emitter writes this
// stride into the device source it hands us, so a locally derived second
// answer is a second answer. The assertion below keeps the local knowledge as
// a check on the projection rather than as its source.
inline constexpr std::uint32_t kPtirIntrinsicSlots = PTIR_INTRINSIC_SLOTS;
static_assert(
    kPtirIntrinsicSlots == static_cast<std::uint32_t>(PTIR_INTR_ATTN_SCORE) + 1u,
    "PTIR_INTRINSIC_SLOTS must be one past the largest PtirIntrinsic id");

}  // namespace pie_cuda_driver::pipeline::generated
