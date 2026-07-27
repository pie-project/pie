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

#include "pie_native/ptir/op_table.hpp"
#include "pie_native/ptir/plan.hpp"
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

struct GeneratedOpMeta {
    std::uint32_t node = 0;
    std::uint32_t result_base = 0;
    pie_native::ptir::container::COp op;
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
using SingletonOpMeta = GeneratedOpMeta;
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
    const pie_native::ptir::plan::ValueType& left,
    const pie_native::ptir::plan::ValueType& right) {
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
    const pie_native::ptir::plan::StagePlan& stage,
    const pie_native::ptir::plan::Region& region) {
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
    const std::vector<pie_native::ptir::plan::Dimension> row_dims(
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
    const pie_native::ptir::plan::StagePlan& stage,
    const pie_native::ptir::plan::Region& region) {
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

inline bool second_party_region_supported(
    const pie_native::ptir::plan::StagePlan& stage,
    const pie_native::ptir::plan::Region& region) {
    if (region.nodes.size() != 1) return false;
    const std::uint32_t node = region.nodes.front();
    if (node >= stage.ops.size()) return false;
    const auto& op = stage.ops[node].op;
    if (op.tag == PTIR_OP_SINK_CALL) {
        // `attn_page_mask(mask)` -- a configuration sink. One argument, no
        // result. The mask is a per-page vector over the request's page list,
        // so the only structural claim that holds is rank 1; its extent is the
        // program's own page ceiling, which the runtime checks against the
        // lane's actual page count.
        if (op.name_idx >= stage.names.size()) return false;
        if (stage.names[op.name_idx] != "attn_page_mask") return false;
        if (op.args.size() != 1 || op.results != 0) return false;
        if (!region.outputs.empty()) return false;
        if (region.inputs.size() != 1) return false;
        const auto& mask_type = stage.value_types[region.inputs.front()];
        if (mask_type.dims.size() != 1) return false;
        return stage.stage == PTIR_STAGE_ON_ATTN_PROJ;
    }
    if (op.tag != PTIR_OP_KERNEL_CALL) return false;
    if (op.name_idx >= stage.names.size()) return false;
    if (stage.names[op.name_idx] != "envelope_dot") return false;
    if (op.args.size() != 1 || op.results != 1) return false;
    // The score is a per-page f32 vector. A different rank or dtype means the
    // program disagrees with the kernel's ABI.
    if (region.outputs.size() != 1) return false;
    const auto& result_type = stage.value_types[region.outputs.front()];
    if (result_type.dtype != PTIR_DT_F32 || result_type.dims.size() != 1) {
        return false;
    }
    return stage.stage == PTIR_STAGE_ON_ATTN_PROJ ||
           stage.stage == PTIR_STAGE_ON_ATTN;
}

inline constexpr std::uint16_t kCudaGeneratedEmitterVersion = 19;

// Per-lane stride of the intrinsic side tables (bases / modes / widths /
// strides / offsets), in slots. Indexed by `IntrinsicId`, so it must be one
// past the largest id -- an id that overflows this stride does not fault, it
// silently reads the NEXT lane's slot 0. Both the host packer
// (`fused_runtime.cuh`) and the emitted device source below index with it, and
// `module_cache.hpp` keys the disk cache on `kCudaGeneratedEmitterVersion`, so
// widening it must bump that version or stale cubins keep the old stride.
inline constexpr std::uint32_t kPtirIntrinsicSlots =
    static_cast<std::uint32_t>(PTIR_INTR_ATTN_SCORE) + 1u;
static_assert(kPtirIntrinsicSlots == 8u);

inline bool validate_generated_region(
    const pie_native::ptir::plan::StagePlan& stage,
    const pie_native::ptir::plan::Region& region,
    std::string& error) {
    error.clear();
    if (region.library || region.schedule == PTIR_SCHEDULE_LIBRARY ||
        region.nodes.empty()) {
        error = "fused CUDA emitter requires a non-library generated region";
        return false;
    }
    std::uint32_t previous = 0;
    bool have_previous = false;
    for (const std::uint32_t node : region.nodes) {
        if (node >= stage.ops.size() ||
            (have_previous && node <= previous)) {
            error = "generated region nodes are invalid or unordered";
            return false;
        }
        const auto& op = stage.ops[node].op;
        if (!detail::supported_tag(op.tag) ||
            op.tag == PTIR_OP_KERNEL_CALL ||
            op.tag == PTIR_OP_SINK_CALL) {
            error = "generated region contains a non-generated boundary";
            return false;
        }
        previous = node;
        have_previous = true;
    }
    return true;
}

struct DirectArgmaxAnalysis {
    std::vector<std::uint16_t> intrinsic;
    std::vector<std::uint8_t> skipped;
    std::vector<std::uint32_t> source_value;
    std::vector<std::uint8_t> requires_single_row;
};

inline DirectArgmaxAnalysis analyze_direct_argmax(
    const pie_native::ptir::plan::StagePlan& stage,
    const pie_native::ptir::plan::Region& region,
    const std::vector<std::uint32_t>& bases) {
    const std::uint32_t value_count =
        static_cast<std::uint32_t>(stage.value_types.size());
    std::vector<std::uint32_t> producers(
        value_count, std::numeric_limits<std::uint32_t>::max());
    std::vector<std::vector<std::uint32_t>> consumers(value_count);
    for (std::size_t node = 0; node < stage.ops.size(); ++node) {
        const auto& op = stage.ops[node].op;
        for (std::uint32_t result = 0; result < op.results; ++result) {
            producers[bases[node] + result] =
                static_cast<std::uint32_t>(node);
        }
        for (const std::uint32_t argument : op.args) {
            consumers[argument].push_back(
                static_cast<std::uint32_t>(node));
        }
    }
    DirectArgmaxAnalysis analysis{
        std::vector<std::uint16_t>(
            stage.ops.size(), std::numeric_limits<std::uint16_t>::max()),
        std::vector<std::uint8_t>(stage.ops.size(), 0),
        std::vector<std::uint32_t>(
            stage.ops.size(), std::numeric_limits<std::uint32_t>::max()),
        std::vector<std::uint8_t>(stage.ops.size(), 0),
    };
    struct RowShape {
        std::uint64_t fixed_rows = 1;
        std::uint32_t row_extent = UINT32_MAX;
        std::uint32_t width = 1;
        bool operator==(const RowShape&) const = default;
    };
    auto row_shape = [](const auto& type) -> std::optional<RowShape> {
        RowShape shape;
        if (type.dims.size() >= 2) {
            for (std::size_t dimension = 0;
                 dimension + 1 < type.dims.size();
                 ++dimension) {
                if (type.dims[dimension].symbolic) {
                    if (shape.row_extent != UINT32_MAX) {
                        return std::nullopt;
                    }
                    shape.row_extent = type.dims[dimension].value;
                } else {
                    if (type.dims[dimension].value == 0 ||
                        shape.fixed_rows >
                            std::numeric_limits<std::uint64_t>::max() /
                                type.dims[dimension].value) {
                        return std::nullopt;
                    }
                    shape.fixed_rows *= type.dims[dimension].value;
                }
            }
        }
        if (!type.dims.empty()) {
            if (type.dims.back().symbolic) return std::nullopt;
            shape.width = type.dims.back().value;
            if (shape.width == 0) return std::nullopt;
        }
        return shape;
    };
    for (const std::uint32_t node : region.nodes) {
        const auto& reduction = stage.ops[node].op;
        if (reduction.tag != PTIR_OP_REDUCE_ARGMAX ||
            reduction.args.empty()) {
            continue;
        }
        std::uint32_t value = reduction.args[0];
        std::uint32_t expected_consumer = node;
        std::vector<std::uint32_t> chain;
        while (value < producers.size() &&
               producers[value] != std::numeric_limits<std::uint32_t>::max() &&
               consumers[value].size() == 1 &&
               consumers[value][0] == expected_consumer) {
            const std::uint32_t producer = producers[value];
            const auto& op = stage.ops[producer].op;
            chain.push_back(producer);
            if (op.tag == PTIR_OP_RESHAPE && !op.args.empty()) {
                expected_consumer = producer;
                value = op.args[0];
                continue;
            }
            if (op.tag != PTIR_OP_INTRINSIC_VAL ||
                (op.intr != PTIR_INTR_LOGITS &&
                 op.intr != PTIR_INTR_MTP_LOGITS)) {
                break;
            }
            const auto source_shape =
                row_shape(stage.value_types[bases[producer]]);
            const auto reduction_shape =
                row_shape(stage.value_types[reduction.args[0]]);
            const bool exact_shape =
                source_shape.has_value() &&
                reduction_shape == source_shape;
            const bool runtime_single_row =
                source_shape.has_value() &&
                reduction_shape.has_value() &&
                source_shape->width == reduction_shape->width &&
                source_shape->fixed_rows == 1 &&
                reduction_shape->fixed_rows == 1 &&
                source_shape->row_extent != UINT32_MAX &&
                reduction_shape->row_extent == UINT32_MAX;
            if (exact_shape || runtime_single_row) {
                analysis.intrinsic[node] = op.intr;
                analysis.source_value[node] = bases[producer];
                analysis.requires_single_row[node] =
                    runtime_single_row ? 1 : 0;
                for (const std::uint32_t skipped : chain) {
                    analysis.skipped[skipped] = 1;
                }
            }
            break;
        }
    }
    return analysis;
}


}  // namespace pie_cuda_driver::pipeline::generated
