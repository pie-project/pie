#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <string>
#include <type_traits>
#include <vector>

#include "pie_native/ptir/op_table.hpp"
#include "pie_native/ptir/plan.hpp"
#include "pie_native/ptir/rng_contract.generated.h"

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

inline bool validate_singleton_plan(
    const pie_native::ptir::plan::StagePlan& plan,
    std::vector<GeneratedOpMeta>& operations,
    std::string& error) {
    operations.clear();
    error.clear();
    if (plan.stage > PTIR_STAGE_EPILOGUE ||
        plan.signature_hash == 0 ||
        pie_native::ptir::container::fnv1a64(
            plan.signature.data(), plan.signature.size()) !=
            plan.signature_hash ||
        plan.singleton.kind != 0 || plan.fused.kind != 1) {
        error = "invalid singleton plan identity";
        return false;
    }
    for (const auto& type : plan.value_types) {
        if (type.dims.size() > 4 || type.dtype > PTIR_DT_BOOL ||
            type.domain > 7) {
            error = "invalid normalized value type";
            return false;
        }
        std::uint64_t product = 1;
        for (const auto& dimension : type.dims) {
            if (dimension.symbolic) {
                if (dimension.value > PTIR_EXTENT_KEY_LEN) {
                    error = "invalid symbolic extent role";
                    return false;
                }
                continue;
            }
            if (dimension.value == 0 ||
                product >
                    std::numeric_limits<std::uint32_t>::max() /
                        dimension.value) {
                error = "normalized value shape product exceeds u32";
                return false;
            }
            product *= dimension.value;
        }
    }
    for (const auto* partition : {&plan.singleton, &plan.fused}) {
        std::vector<bool> covered(plan.ops.size(), false);
        std::uint32_t previous_node = 0;
        bool have_previous = false;
        for (const auto& region : partition->regions) {
            if (region.schedule > PTIR_SCHEDULE_LIBRARY ||
                region.nodes.empty()) {
                error = "invalid region schedule or empty node list";
                return false;
            }
            if (std::any_of(
                    region.nodes.begin(),
                    region.nodes.end(),
                    [&](std::uint32_t node) {
                        return node >= plan.ops.size();
                    })) {
                error = "region node out of range";
                return false;
            }
            if (std::adjacent_find(
                    region.nodes.begin(),
                    region.nodes.end(),
                    std::greater_equal<std::uint32_t>()) !=
                region.nodes.end()) {
                error = "region node indices are not strictly ordered";
                return false;
            }
            if (have_previous && region.nodes.front() <= previous_node) {
                error = "partition region nodes are not globally ordered";
                return false;
            }
            previous_node = region.nodes.back();
            have_previous = true;
            for (const std::uint32_t node : region.nodes) {
                if (covered[node]) {
                    error = "partition contains a duplicate region node";
                    return false;
                }
                covered[node] = true;
            }
            if (std::any_of(
                    region.inputs.begin(),
                    region.inputs.end(),
                    [&](std::uint32_t value) {
                        return value >= plan.value_types.size();
                    })) {
                error = "region input out of range";
                return false;
            }
            if (std::any_of(
                    region.outputs.begin(),
                    region.outputs.end(),
                    [&](std::uint32_t value) {
                        return value >= plan.value_types.size();
                    })) {
                error = "region output out of range";
                return false;
            }
            for (const auto& sink : region.sinks) {
                if (sink.channel_slot >= plan.channel_bindings.size() ||
                    sink.value >= plan.value_types.size()) {
                    error = "region sink out of range";
                    return false;
                }
            }
            if (!detail::library_region_valid(plan, region)) {
                error = "library region ABI is invalid";
                return false;
            }
        }
        if (std::any_of(
                covered.begin(), covered.end(), [](bool value) {
                    return !value;
                })) {
            error = "partition does not cover every normalized op";
            return false;
        }
    }
    if (plan.singleton.regions.size() != plan.ops.size()) {
        error = "singleton partition must contain one region per normalized op";
        return false;
    }
    std::uint32_t result_base = 0;
    for (std::size_t node = 0; node < plan.ops.size(); ++node) {
        const auto& region = plan.singleton.regions[node];
        const auto& op = plan.ops[node].op;
        if (region.nodes.size() != 1 || region.nodes[0] != node) {
            error = "singleton region/node ordering mismatch";
            return false;
        }
        if (!detail::supported_tag(op.tag)) {
            error = "unsupported singleton opcode tag " +
                std::to_string(op.tag);
            return false;
        }
        const auto info = pie_native::ptir::op_info(
            static_cast<pie_native::ptir::OpCode>(op.tag));
        const std::size_t expected_arity =
            op.tag == PTIR_OP_PIVOT_THRESHOLD ? 1 : info.arity;
        if (expected_arity != 0xfe &&
            op.args.size() != expected_arity) {
            error = "normalized op arity mismatch";
            return false;
        }
        if (op.results != info.results ||
            result_base >
                std::numeric_limits<std::uint32_t>::max() - op.results ||
            result_base + op.results > plan.value_types.size()) {
            error = "normalized op result range is invalid";
            return false;
        }
        for (const std::uint32_t argument : op.args) {
            if (argument >= result_base) {
                error = "normalized SSA operand is not a prior value";
                return false;
            }
        }
        if (op.tag == PTIR_OP_PIVOT_THRESHOLD &&
            (op.pred_tag > 2 || op.pred_payload >= result_base)) {
            error = "pivot predicate payload is out of range";
            return false;
        }
        if ((op.tag == PTIR_OP_RNG ||
             op.tag == PTIR_OP_RNG_KEYED) &&
            op.kind > 1) {
            error = "normalized RNG kind is invalid";
            return false;
        }
        if (op.tag == PTIR_OP_INTRINSIC_VAL &&
            op.intr > PTIR_INTR_MTP_DRAFTS) {
            error = "normalized intrinsic id is invalid";
            return false;
        }
        const bool channel_op =
            op.tag == PTIR_OP_CHAN_TAKE ||
            op.tag == PTIR_OP_CHAN_READ ||
            op.tag == PTIR_OP_CHAN_PUT;
        if ((channel_op &&
             (op.chan < 0 ||
              static_cast<std::size_t>(op.chan) >=
                  plan.channel_bindings.size())) ||
            (!channel_op && op.chan != -1)) {
            error = "normalized channel slot is invalid";
            return false;
        }
        if (op.tag == PTIR_OP_KERNEL_CALL) {
            const bool identity =
                op.name_idx < plan.names.size() &&
                plan.names[op.name_idx] == "cuda.identity";
            const bool envelope_dot =
                op.name_idx < plan.names.size() &&
                plan.names[op.name_idx] == "envelope_dot";
            if (op.name_idx >= plan.names.size() ||
                (!identity && !envelope_dot) ||
                op.args.size() != 1 ||
                result_base >= plan.value_types.size() ||
                !detail::same_type(
                    plan.value_types[op.args[0]],
                    plan.value_types[result_base])) {
                error =
                    "unsupported CUDA semantic kernel boundary " +
                    (op.name_idx < plan.names.size()
                         ? plan.names[op.name_idx]
                         : std::string("<out-of-range>"));
                return false;
            }
        } else if (op.tag == PTIR_OP_SINK_CALL) {
            if (op.name_idx >= plan.names.size() ||
                (plan.names[op.name_idx] != "cuda.discard" &&
                 plan.names[op.name_idx] != PTIR_SINK_ATTN_PAGE_MASK)) {
                error =
                    "unsupported CUDA semantic sink boundary " +
                    (op.name_idx < plan.names.size()
                         ? plan.names[op.name_idx]
                         : std::string("<out-of-range>"));
                return false;
            }
        }
        operations.push_back(
            {static_cast<std::uint32_t>(node), result_base, op});
        result_base += op.results;
    }
    if (plan.singleton.whole_stage_fallback ||
        plan.fused.whole_stage_fallback) {
        error = "generated singleton plan must not request fallback execution";
        return false;
    }
    if (result_base != plan.value_types.size()) {
        error = "normalized value layout does not match op results";
        return false;
    }
    return true;
}

inline std::string singleton_runtime_cuda_source() {
    std::string source = R"PTIR_CUDA(
typedef unsigned char m1_u8;
typedef unsigned short m1_u16;
typedef unsigned int m1_u32;
typedef unsigned long long m1_u64;

struct M1Status {
  m1_u32 state;
  m1_u32 fault;
  m1_u32 reserved0;
  m1_u32 reserved1;
};

struct M1ValueDesc {
  m1_u32 len;
  m1_u32 rows;
  m1_u32 last;
  m1_u32 rank;
  m1_u32 dtype;
  m1_u32 dims[4];
};

struct M1OpParams {
  m1_u32 tag;
  m1_u32 a0;
  m1_u32 a1;
  m1_u32 a2;
  m1_u32 o0;
  m1_u32 o1;
  m1_u32 imm;
  m1_u32 imm2;
  m1_u32 imm3;
  m1_u32 kind;
  m1_u32 pred_tag;
  m1_u32 lit_dtype;
  m1_u32 lit_bits;
  m1_u32 channel_slot;
  m1_u32 intr;
  m1_u32 sink_bytes;
  m1_u32 intrinsic_dtype;
  m1_u32 bool_storage;
  m1_u32 intrinsic_row_stride;
  m1_u32 intrinsic_row_offset;
  m1_u64 rng_seed;
};

static_assert(sizeof(M1Status) == 16, "M1Status ABI");
static_assert(sizeof(M1ValueDesc) == 36, "M1ValueDesc ABI");
static_assert(sizeof(M1OpParams) == 88, "M1OpParams ABI");

struct M1ArgmaxCandidate {
  float value;
  m1_u32 index;
  m1_u32 have;
  m1_u32 reserved;
};

struct M1IntArgmaxCandidate {
  long long value;
  m1_u32 index;
  m1_u32 have;
};

__device__ __forceinline__ float m1_pos_inf() {
  return __int_as_float(0x7f800000);
}

__device__ __forceinline__ float m1_neg_inf() {
  return __int_as_float((int)0xff800000u);
}

__device__ __forceinline__ float m1_nan() {
  return __int_as_float(0x7fc00000);
}

__device__ __forceinline__ bool m1_isnan(float value) {
  return value != value;
}

__device__ __forceinline__ bool m1_signbit(float value) {
  return (__float_as_uint(value) >> 31) != 0;
}

__device__ __forceinline__ int m1_bits_i32(m1_u32 value) {
  union Bits {
    m1_u32 u;
    int i;
  } bits;
  bits.u = value;
  return bits.i;
}

__device__ __forceinline__ float m1_bits_f32(m1_u32 value) {
  return __uint_as_float(value);
}

__device__ __forceinline__ int m1_float_to_i32(float value) {
  if (m1_isnan(value)) return 0;
  if (value >= 2147483647.0f) return 2147483647;
  if (value <= -2147483648.0f) return m1_bits_i32(0x80000000u);
  return (int)value;
}

__device__ __forceinline__ m1_u32 m1_float_to_u32(float value) {
  if (m1_isnan(value) || value <= 0.0f) return 0u;
  if (value >= 4294967295.0f) return 0xffffffffu;
  return (m1_u32)value;
}

__device__ __forceinline__ float m1_load_f(
    const m1_u8* data, m1_u32 index, m1_u32 dtype) {
  if (dtype == 0) return reinterpret_cast<const float*>(data)[index];
  if (dtype == 1) return float(reinterpret_cast<const int*>(data)[index]);
  if (dtype == 2) return float(reinterpret_cast<const m1_u32*>(data)[index]);
  return data[index] != 0 ? 1.0f : 0.0f;
}

__device__ __forceinline__ int m1_load_i(
    const m1_u8* data, m1_u32 index, m1_u32 dtype) {
  if (dtype == 0)
    return m1_float_to_i32(reinterpret_cast<const float*>(data)[index]);
  if (dtype == 1) return reinterpret_cast<const int*>(data)[index];
  if (dtype == 2)
    return m1_bits_i32(reinterpret_cast<const m1_u32*>(data)[index]);
  return data[index] != 0 ? 1 : 0;
}

__device__ __forceinline__ m1_u32 m1_load_u(
    const m1_u8* data, m1_u32 index, m1_u32 dtype) {
  if (dtype == 0)
    return m1_float_to_u32(reinterpret_cast<const float*>(data)[index]);
  if (dtype == 1)
    return (m1_u32)reinterpret_cast<const int*>(data)[index];
  if (dtype == 2) return reinterpret_cast<const m1_u32*>(data)[index];
  return data[index] != 0 ? 1u : 0u;
}

__device__ __forceinline__ bool m1_load_b(
    const m1_u8* data, m1_u32 index, m1_u32 dtype) {
  if (dtype == 0)
    return reinterpret_cast<const float*>(data)[index] != 0.0f;
  if (dtype == 1)
    return reinterpret_cast<const int*>(data)[index] != 0;
  if (dtype == 2)
    return reinterpret_cast<const m1_u32*>(data)[index] != 0u;
  return data[index] != 0;
}

__device__ __forceinline__ void m1_store_f(
    m1_u8* data, m1_u32 index, float value) {
  reinterpret_cast<float*>(data)[index] = value;
}

__device__ __forceinline__ void m1_store_i(
    m1_u8* data, m1_u32 index, int value) {
  reinterpret_cast<int*>(data)[index] = value;
}

__device__ __forceinline__ void m1_store_u(
    m1_u8* data, m1_u32 index, m1_u32 value) {
  reinterpret_cast<m1_u32*>(data)[index] = value;
}

__device__ __forceinline__ void m1_store_b(
    m1_u8* data, m1_u32 index, bool value) {
  data[index] = value ? 1 : 0;
}

__device__ __forceinline__ float m1_canonical_max(
    float left, float right) {
  const bool ln = m1_isnan(left);
  const bool rn = m1_isnan(right);
  if (ln && rn) return m1_neg_inf();
  if (ln) return right;
  if (rn) return left;
  if (left == 0.0f && right == 0.0f)
    return m1_signbit(left) && m1_signbit(right) ? -0.0f : 0.0f;
  return left > right ? left : right;
}

__device__ __forceinline__ float m1_canonical_min(
    float left, float right) {
  const bool ln = m1_isnan(left);
  const bool rn = m1_isnan(right);
  if (ln && rn) return m1_pos_inf();
  if (ln) return right;
  if (rn) return left;
  if (left == 0.0f && right == 0.0f)
    return m1_signbit(left) || m1_signbit(right) ? -0.0f : 0.0f;
  return left < right ? left : right;
}

__device__ __forceinline__ float m1_element_max(
    float left, float right) {
  const bool ln = m1_isnan(left);
  const bool rn = m1_isnan(right);
  if (ln && rn) return left;
  if (ln) return right;
  if (rn) return left;
  if (left == 0.0f && right == 0.0f)
    return m1_signbit(left) && m1_signbit(right) ? -0.0f : 0.0f;
  return left > right ? left : right;
}

__device__ __forceinline__ float m1_element_min(
    float left, float right) {
  const bool ln = m1_isnan(left);
  const bool rn = m1_isnan(right);
  if (ln && rn) return left;
  if (ln) return right;
  if (rn) return left;
  if (left == 0.0f && right == 0.0f)
    return m1_signbit(left) || m1_signbit(right) ? -0.0f : 0.0f;
  return left < right ? left : right;
}

__device__ __forceinline__ long long m1_load_index(
    const m1_u8* data, m1_u32 index, m1_u32 dtype) {
  if (dtype == 1)
    return (long long)reinterpret_cast<const int*>(data)[index];
  if (dtype == 2)
    return (long long)reinterpret_cast<const m1_u32*>(data)[index];
  if (dtype == 3) return data[index] != 0 ? 1ll : 0ll;
  return (long long)m1_float_to_i32(
      reinterpret_cast<const float*>(data)[index]);
}

__device__ __forceinline__ M1ArgmaxCandidate m1_argmax_combine(
    M1ArgmaxCandidate left, M1ArgmaxCandidate right) {
  if (right.have == 0) return left;
  if (left.have == 0 || right.value > left.value ||
      (right.value == left.value && right.index < left.index)) {
    return right;
  }
  return left;
}

__device__ __forceinline__ M1IntArgmaxCandidate m1_int_argmax_combine(
    M1IntArgmaxCandidate left, M1IntArgmaxCandidate right) {
  if (right.have == 0) return left;
  if (left.have == 0 || right.value > left.value ||
      (right.value == left.value && right.index < left.index)) {
    return right;
  }
  return left;
}

__device__ __forceinline__ bool m1_sort_better(
    float value, m1_u32 index, float best, m1_u32 best_index) {
  const bool value_nan = m1_isnan(value);
  const bool best_nan = m1_isnan(best);
  if (value_nan != best_nan) return best_nan;
  if (value_nan) return index < best_index;
  if (value != best) return value > best;
  return index < best_index;
}

__device__ __forceinline__ m1_u32 m1_pick(
    m1_u32 len, m1_u32 index) {
  return len == 1 ? 0u : index;
}

__device__ __forceinline__ void m1_fault(
    M1Status* status, m1_u32 code) {
  status->fault = code;
  status->state = 3;
}

__device__ __forceinline__ void m1_copy_typed(
    const m1_u8* input, m1_u8* output, m1_u32 len, m1_u32 dtype) {
  const m1_u32 bytes = dtype == 3 ? len : len * 4u;
  for (m1_u32 i = 0; i < bytes; ++i) output[i] = input[i];
}

__device__ __forceinline__ int m1_i32_div(int left, int right) {
  if (right == 0) return 0;
  if ((m1_u32)left == 0x80000000u && right == -1) return left;
  return left / right;
}

__device__ __forceinline__ int m1_i32_rem(int left, int right) {
  if (right == 0 || ((m1_u32)left == 0x80000000u && right == -1))
    return 0;
  return left % right;
}

__device__ __forceinline__ float m1_intrinsic_load(
    const m1_u8* input, m1_u64 index, m1_u32 mode) {
  if (mode == 0)
    return reinterpret_cast<const float*>(input)[index];
  const m1_u32 bits =
      (m1_u32)reinterpret_cast<const m1_u16*>(input)[index] << 16;
  return __uint_as_float(bits);
}

__device__ __forceinline__ float m1_intrinsic_row_load(
    const m1_u8* input,
    m1_u64 row,
    m1_u32 column,
    m1_u32 stride,
    m1_u32 mode) {
  if (mode != 2u)
    return m1_intrinsic_load(
        input, row * (m1_u64)stride + column, mode);
  const m1_u64 row_address =
      reinterpret_cast<const m1_u64*>(input)[row];
  const m1_u16 value =
      reinterpret_cast<const m1_u16*>(row_address)[column];
  return __uint_as_float((m1_u32)value << 16);
}

__device__ __forceinline__ void m1_reduce_float(
    m1_u32 tag,
    const m1_u8* input,
    m1_u8* output,
    m1_u8* temporary,
    M1ValueDesc in_desc) {
  float* work = reinterpret_cast<float*>(temporary);
  const float* values = reinterpret_cast<const float*>(input);
  float* result = reinterpret_cast<float*>(output);
  for (m1_u32 row = 0; row < in_desc.rows; ++row) {
    const m1_u32 base = row * in_desc.last;
    for (m1_u32 i = 0; i < in_desc.last; ++i)
      work[i] = values[base + i];
    m1_u32 count = in_desc.last;
    if (count == 0) {
      result[row] =
          tag == 0x30 ? 0.0f
                      : (tag == 0x31 ? m1_neg_inf() : m1_pos_inf());
      continue;
    }
    while (count > 1) {
      const m1_u32 chunks = (count + 31u) / 32u;
      for (m1_u32 chunk = 0; chunk < chunks; ++chunk) {
        float lanes[32];
        const float identity =
            tag == 0x30 ? 0.0f
                        : (tag == 0x31 ? m1_neg_inf() : m1_pos_inf());
        for (m1_u32 lane = 0; lane < 32; ++lane) {
          const m1_u32 index = chunk * 32u + lane;
          lanes[lane] = index < count ? work[index] : identity;
        }
        for (m1_u32 offset = 16; offset > 0; offset >>= 1) {
          for (m1_u32 lane = 0; lane < offset; ++lane) {
            if (tag == 0x30)
              lanes[lane] += lanes[lane + offset];
            else if (tag == 0x31)
              lanes[lane] =
                  m1_canonical_max(lanes[lane], lanes[lane + offset]);
            else
              lanes[lane] =
                  m1_canonical_min(lanes[lane], lanes[lane + offset]);
          }
        }
        work[chunk] = lanes[0];
      }
      count = chunks;
    }
    result[row] = work[0];
  }
}

__device__ __forceinline__ void m1_reduce_integer(
    m1_u32 tag,
    const m1_u8* input,
    m1_u8* output,
    m1_u8* temporary,
    M1ValueDesc in_desc) {
  m1_u32* work = reinterpret_cast<m1_u32*>(temporary);
  for (m1_u32 row = 0; row < in_desc.rows; ++row) {
    const m1_u32 base = row * in_desc.last;
    for (m1_u32 i = 0; i < in_desc.last; ++i) {
      work[i] =
          in_desc.dtype == 1
              ? (m1_u32)reinterpret_cast<const int*>(input)[base + i]
              : reinterpret_cast<const m1_u32*>(input)[base + i];
    }
    m1_u32 count = in_desc.last;
    if (count == 0) {
      if (in_desc.dtype == 1) {
        reinterpret_cast<int*>(output)[row] =
            tag == 0x30
                ? 0
                : (tag == 0x31 ? m1_bits_i32(0x80000000u)
                               : 2147483647);
      } else {
        reinterpret_cast<m1_u32*>(output)[row] =
            tag == 0x32 ? 0xffffffffu : 0u;
      }
      continue;
    }
    while (count > 1) {
      const m1_u32 chunks = (count + 31u) / 32u;
      for (m1_u32 chunk = 0; chunk < chunks; ++chunk) {
        m1_u32 lanes[32];
        for (m1_u32 lane = 0; lane < 32; ++lane) {
          const m1_u32 index = chunk * 32u + lane;
          if (index < count)
            lanes[lane] = work[index];
          else if (tag == 0x30)
            lanes[lane] = 0u;
          else if (in_desc.dtype == 1)
            lanes[lane] =
                tag == 0x31 ? 0x80000000u : 0x7fffffffu;
          else
            lanes[lane] = tag == 0x31 ? 0u : 0xffffffffu;
        }
        for (m1_u32 offset = 16; offset > 0; offset >>= 1) {
          for (m1_u32 lane = 0; lane < offset; ++lane) {
            if (tag == 0x30) {
              lanes[lane] += lanes[lane + offset];
            } else if (in_desc.dtype == 1) {
              const int left = m1_bits_i32(lanes[lane]);
              const int right = m1_bits_i32(lanes[lane + offset]);
              lanes[lane] =
                  (m1_u32)(tag == 0x31
                               ? (left > right ? left : right)
                               : (left < right ? left : right));
            } else {
              const m1_u32 left = lanes[lane];
              const m1_u32 right = lanes[lane + offset];
              lanes[lane] =
                  tag == 0x31
                      ? (left > right ? left : right)
                      : (left < right ? left : right);
            }
          }
        }
        work[chunk] = lanes[0];
      }
      count = chunks;
    }
    if (in_desc.dtype == 1)
      reinterpret_cast<int*>(output)[row] = m1_bits_i32(work[0]);
    else
      reinterpret_cast<m1_u32*>(output)[row] = work[0];
  }
}

__device__ __forceinline__ void m1_reduce_argmax(
    const m1_u8* input,
    m1_u8* output,
    m1_u8* temporary,
    M1ValueDesc in_desc) {
  int* result = reinterpret_cast<int*>(output);
  if (in_desc.dtype != 0) {
    M1IntArgmaxCandidate* work =
        reinterpret_cast<M1IntArgmaxCandidate*>(temporary);
    for (m1_u32 row = 0; row < in_desc.rows; ++row) {
      const m1_u32 base = row * in_desc.last;
      for (m1_u32 i = 0; i < in_desc.last; ++i)
        work[i] = {m1_load_index(input, base + i, in_desc.dtype), i, 1u};
      m1_u32 count = in_desc.last;
      if (count == 0) {
        result[row] = 0;
        continue;
      }
      while (count > 1) {
        const m1_u32 chunks = (count + 31u) / 32u;
        for (m1_u32 chunk = 0; chunk < chunks; ++chunk) {
          M1IntArgmaxCandidate lanes[32];
          for (m1_u32 lane = 0; lane < 32; ++lane) {
            const m1_u32 index = chunk * 32u + lane;
            lanes[lane] =
                index < count
                    ? work[index]
                    : M1IntArgmaxCandidate{0ll, 0u, 0u};
          }
          for (m1_u32 offset = 16; offset > 0; offset >>= 1)
            for (m1_u32 lane = 0; lane < offset; ++lane)
              lanes[lane] =
                  m1_int_argmax_combine(lanes[lane], lanes[lane + offset]);
          work[chunk] = lanes[0];
        }
        count = chunks;
      }
      result[row] = (int)work[0].index;
    }
    return;
  }
  const float* values = reinterpret_cast<const float*>(input);
  M1ArgmaxCandidate* work =
      reinterpret_cast<M1ArgmaxCandidate*>(temporary);
  for (m1_u32 row = 0; row < in_desc.rows; ++row) {
    const m1_u32 base = row * in_desc.last;
    for (m1_u32 i = 0; i < in_desc.last; ++i) {
      const float value = values[base + i];
      work[i] = {value, i, m1_isnan(value) ? 0u : 1u, 0u};
    }
    m1_u32 count = in_desc.last;
    if (count == 0) {
      result[row] = 0;
      continue;
    }
    while (count > 1) {
      const m1_u32 chunks = (count + 31u) / 32u;
      for (m1_u32 chunk = 0; chunk < chunks; ++chunk) {
        M1ArgmaxCandidate lanes[32];
        for (m1_u32 lane = 0; lane < 32; ++lane) {
          const m1_u32 index = chunk * 32u + lane;
          lanes[lane] =
              index < count
                  ? work[index]
                  : M1ArgmaxCandidate{m1_neg_inf(), 0u, 0u, 0u};
        }
        for (m1_u32 offset = 16; offset > 0; offset >>= 1)
          for (m1_u32 lane = 0; lane < offset; ++lane)
            lanes[lane] =
                m1_argmax_combine(lanes[lane], lanes[lane + offset]);
        work[chunk] = lanes[0];
      }
      count = chunks;
    }
    result[row] = (int)work[0].index;
  }
}
)PTIR_CUDA";
    source += PTIR_RNG_CUDA_PREAMBLE;
    source += R"PTIR_CUDA(
__device__ __forceinline__ void ptir_m1_execute(
    m1_u32 generated_tag,
    M1Status* status,
    const M1ValueDesc* descriptors,
    const M1OpParams* params,
    const m1_u8* a0,
    const m1_u8* a1,
    const m1_u8* a2,
    m1_u8* o0,
    m1_u8* o1,
    m1_u8* temporary) {
  if (status->state != 1) return;
  M1OpParams p = params[0];
  p.tag = generated_tag;
  const M1ValueDesc d0 = descriptors[p.a0];
  const M1ValueDesc d1 = descriptors[p.a1];
  const M1ValueDesc d2 = descriptors[p.a2];
  const M1ValueDesc out0 = descriptors[p.o0];

  if (p.tag == 0x81) {
    for (m1_u32 i = 0; i < out0.len; ++i) {
      if (p.lit_dtype == 0)
        m1_store_f(o0, i, m1_bits_f32(p.lit_bits));
      else if (p.lit_dtype == 1)
        m1_store_i(o0, i, m1_bits_i32(p.lit_bits));
      else if (p.lit_dtype == 2)
        m1_store_u(o0, i, p.lit_bits);
      else
        m1_store_b(o0, i, p.lit_bits != 0);
    }
    return;
  }
  if (p.tag == 0x90 || p.tag == 0x91) {
    if (out0.dtype == 3) {
      if (p.bool_storage > 1u) {
        m1_fault(status, p.tag);
        return;
      }
      if (p.bool_storage == 1u) {
        for (m1_u32 i = 0; i < out0.len; ++i)
          o0[i] = (a0[i >> 3] >> (i & 7)) & 1u;
      } else {
        for (m1_u32 i = 0; i < out0.len; ++i)
          o0[i] = a0[i] != 0 ? 1 : 0;
      }
    } else {
      m1_copy_typed(a0, o0, out0.len, out0.dtype);
    }
    return;
  }
  if (p.tag == 0x92) {
    const m1_u32 logical_bytes =
        d0.dtype == 3
            ? (p.bool_storage == 1u ? (d0.len + 7u) / 8u : d0.len)
            : d0.len * 4u;
    if (d0.dtype == 3 && p.bool_storage > 1u) {
      m1_fault(status, p.tag);
      return;
    }
    if (logical_bytes > p.sink_bytes) {
      m1_fault(status, p.tag);
      return;
    }
    if (d0.dtype == 3) {
      if (p.bool_storage == 1u) {
        for (m1_u32 i = 0; i < logical_bytes; ++i) o0[i] = 0;
        for (m1_u32 i = 0; i < d0.len; ++i)
          if (a0[i] != 0)
            o0[i >> 3] |= (m1_u8)(1u << (i & 7));
      } else {
        for (m1_u32 i = 0; i < d0.len; ++i)
          o0[i] = a0[i] != 0 ? 1 : 0;
      }
    } else {
      m1_copy_typed(a0, o0, d0.len, d0.dtype);
    }
    for (m1_u32 i = logical_bytes; i < p.sink_bytes; ++i) o0[i] = 0;
    return;
  }
  if (p.tag == 0xA0) {
    if (p.intr == 5u) {
      if (out0.dtype != 2u || out0.len != 1u || a0 == nullptr) {
        m1_fault(status, p.tag);
        return;
      }
      m1_store_u(o0, 0u, reinterpret_cast<const m1_u32*>(a0)[0]);
      return;
    }
    if (p.intr == 6u) {
      if (out0.dtype != 1u || a0 == nullptr) {
        m1_fault(status, p.tag);
        return;
      }
      for (m1_u32 index = 0; index < out0.len; ++index)
        m1_store_i(
            o0, index, reinterpret_cast<const int*>(a0)[index]);
      return;
    }
    if (p.imm == 0u || p.intrinsic_dtype > 2u || a0 == nullptr) {
      m1_fault(status, p.tag);
      return;
    }
    const m1_u32 stride =
        p.intrinsic_row_stride == 0u ? p.imm : p.intrinsic_row_stride;
    if (stride < p.imm) {
      m1_fault(status, p.tag);
      return;
    }
    const m1_u64 first_row =
        (m1_u64)p.intrinsic_row_offset + (m1_u64)p.imm2;
    if (out0.dtype != 0u) {
      m1_fault(status, p.tag);
      return;
    }
    const m1_u32 logical_width =
        out0.last == 0u ? p.imm : out0.last;
    if (logical_width == 0u || stride < logical_width) {
      m1_fault(status, p.tag);
      return;
    }
    for (m1_u32 i = 0; i < out0.len; ++i) {
      const m1_u32 row = i / logical_width;
      const m1_u32 column = i % logical_width;
      m1_store_f(
          o0,
          i,
          m1_intrinsic_row_load(
              a0,
              first_row + (m1_u64)row,
              column,
              stride,
              p.intrinsic_dtype));
    }
    return;
  }
  if (p.tag == 0xA1) {
    m1_copy_typed(a0, o0, out0.len, out0.dtype);
    return;
  }
  if (p.tag == 0xA2) return;

  if (p.tag == 0x01 || p.tag == 0x02 || p.tag == 0x04) {
    for (m1_u32 i = 0; i < out0.len; ++i) {
      const float value = m1_load_f(a0, m1_pick(d0.len, i), d0.dtype);
      if (p.tag == 0x01)
        m1_store_f(o0, i, expf(value));
      else if (p.tag == 0x02)
        m1_store_f(o0, i, logf(value));
      else
        m1_store_f(o0, i, 1.0f / value);
    }
    return;
  }
  if (p.tag == 0x03 || p.tag == 0x05 || p.tag == 0x06) {
    for (m1_u32 i = 0; i < out0.len; ++i) {
      const m1_u32 source_index = m1_pick(d0.len, i);
      if (d0.dtype == 0) {
        const float value = m1_load_f(a0, source_index, d0.dtype);
        const float result =
            p.tag == 0x03
                ? -value
                : (p.tag == 0x05
                       ? fabsf(value)
                       : (value > 0
                              ? 1.0f
                              : (value < 0 ? -1.0f : 0.0f)));
        m1_store_f(o0, i, result);
      } else if (d0.dtype == 1) {
        const int value = m1_load_i(a0, source_index, d0.dtype);
        int result = value;
        if (p.tag == 0x03)
          result = m1_bits_i32(0u - (m1_u32)value);
        else if (p.tag == 0x05)
          result =
              (m1_u32)value == 0x80000000u
                  ? value
                  : (value < 0 ? -value : value);
        else
          result = value > 0 ? 1 : (value < 0 ? -1 : 0);
        m1_store_i(o0, i, result);
      } else if (d0.dtype == 2) {
        const m1_u32 value = m1_load_u(a0, source_index, d0.dtype);
        m1_store_u(
            o0,
            i,
            p.tag == 0x03
                ? 0u - value
                : (p.tag == 0x06 ? (value != 0 ? 1u : 0u) : value));
      } else {
        m1_fault(status, p.tag);
        return;
      }
    }
    return;
  }
  if (p.tag == 0x07) {
    for (m1_u32 i = 0; i < out0.len; ++i) {
      const m1_u32 source_index = m1_pick(d0.len, i);
      if (out0.dtype == 0)
        m1_store_f(o0, i, m1_load_f(a0, source_index, d0.dtype));
      else if (out0.dtype == 1)
        m1_store_i(o0, i, m1_load_i(a0, source_index, d0.dtype));
      else if (out0.dtype == 2)
        m1_store_u(o0, i, m1_load_u(a0, source_index, d0.dtype));
      else
        m1_store_b(o0, i, m1_load_b(a0, source_index, d0.dtype));
    }
    return;
  }

  if ((p.tag >= 0x10 && p.tag <= 0x1D) || p.tag == 0x1F) {
    for (m1_u32 i = 0; i < out0.len; ++i) {
      const m1_u32 xindex = m1_pick(d0.len, i);
      const m1_u32 yindex = m1_pick(d1.len, i);
      if (p.tag >= 0x16 && p.tag <= 0x1D) {
        bool result = false;
        if (p.tag == 0x1C || p.tag == 0x1D) {
          const bool x = m1_load_b(a0, xindex, d0.dtype);
          const bool y = m1_load_b(a1, yindex, d1.dtype);
          result = p.tag == 0x1C ? x && y : x || y;
        } else if (d0.dtype == 0) {
          const float x = m1_load_f(a0, xindex, d0.dtype);
          const float y = m1_load_f(a1, yindex, d1.dtype);
          if (p.tag == 0x16) result = x > y;
          else if (p.tag == 0x17) result = x >= y;
          else if (p.tag == 0x18) result = x == y;
          else if (p.tag == 0x19) result = x != y;
          else if (p.tag == 0x1A) result = x < y;
          else result = x <= y;
        } else if (d0.dtype == 1) {
          const int x = m1_load_i(a0, xindex, d0.dtype);
          const int y = m1_load_i(a1, yindex, d1.dtype);
          if (p.tag == 0x16) result = x > y;
          else if (p.tag == 0x17) result = x >= y;
          else if (p.tag == 0x18) result = x == y;
          else if (p.tag == 0x19) result = x != y;
          else if (p.tag == 0x1A) result = x < y;
          else result = x <= y;
        } else {
          const m1_u32 x = m1_load_u(a0, xindex, d0.dtype);
          const m1_u32 y = m1_load_u(a1, yindex, d1.dtype);
          if (p.tag == 0x16) result = x > y;
          else if (p.tag == 0x17) result = x >= y;
          else if (p.tag == 0x18) result = x == y;
          else if (p.tag == 0x19) result = x != y;
          else if (p.tag == 0x1A) result = x < y;
          else result = x <= y;
        }
        m1_store_b(o0, i, result);
      } else if (d0.dtype == 0) {
        const float x = m1_load_f(a0, xindex, d0.dtype);
        const float y = m1_load_f(a1, yindex, d1.dtype);
        float result = 0.0f;
        if (p.tag == 0x10) result = x + y;
        else if (p.tag == 0x11) result = x - y;
        else if (p.tag == 0x12) result = x * y;
        else if (p.tag == 0x13) result = x / y;
        else if (p.tag == 0x14) result = m1_element_max(x, y);
        else if (p.tag == 0x15) result = m1_element_min(x, y);
        else result = fmodf(x, y);
        m1_store_f(o0, i, result);
      } else if (d0.dtype == 1) {
        const int x = m1_load_i(a0, xindex, d0.dtype);
        const int y = m1_load_i(a1, yindex, d1.dtype);
        int result = 0;
        if (p.tag == 0x10)
          result = m1_bits_i32((m1_u32)x + (m1_u32)y);
        else if (p.tag == 0x11)
          result = m1_bits_i32((m1_u32)x - (m1_u32)y);
        else if (p.tag == 0x12)
          result = m1_bits_i32((m1_u32)x * (m1_u32)y);
        else if (p.tag == 0x13)
          result = m1_i32_div(x, y);
        else if (p.tag == 0x14)
          result = x > y ? x : y;
        else if (p.tag == 0x15)
          result = x < y ? x : y;
        else
          result = m1_i32_rem(x, y);
        m1_store_i(o0, i, result);
      } else {
        const m1_u32 x = m1_load_u(a0, xindex, d0.dtype);
        const m1_u32 y = m1_load_u(a1, yindex, d1.dtype);
        m1_u32 result = 0;
        if (p.tag == 0x10) result = x + y;
        else if (p.tag == 0x11) result = x - y;
        else if (p.tag == 0x12) result = x * y;
        else if (p.tag == 0x13) result = y == 0 ? 0 : x / y;
        else if (p.tag == 0x14) result = x > y ? x : y;
        else if (p.tag == 0x15) result = x < y ? x : y;
        else result = y == 0 ? 0 : x % y;
        m1_store_u(o0, i, result);
      }
    }
    return;
  }
  if (p.tag == 0x1E) {
    for (m1_u32 i = 0; i < out0.len; ++i)
      m1_store_b(
          o0, i, !m1_load_b(a0, m1_pick(d0.len, i), d0.dtype));
    return;
  }
  if (p.tag == 0x20) {
    for (m1_u32 i = 0; i < out0.len; ++i) {
      const bool select =
          m1_load_b(a0, m1_pick(d0.len, i), d0.dtype);
      const m1_u32 xi = m1_pick(d1.len, i);
      const m1_u32 yi = m1_pick(d2.len, i);
      if (out0.dtype == 0)
        m1_store_f(
            o0,
            i,
            select ? m1_load_f(a1, xi, d1.dtype)
                   : m1_load_f(a2, yi, d2.dtype));
      else if (out0.dtype == 1)
        m1_store_i(
            o0,
            i,
            select ? m1_load_i(a1, xi, d1.dtype)
                   : m1_load_i(a2, yi, d2.dtype));
      else if (out0.dtype == 2)
        m1_store_u(
            o0,
            i,
            select ? m1_load_u(a1, xi, d1.dtype)
                   : m1_load_u(a2, yi, d2.dtype));
      else
        m1_store_b(
            o0,
            i,
            select ? m1_load_b(a1, xi, d1.dtype)
                   : m1_load_b(a2, yi, d2.dtype));
    }
    return;
  }

  if (p.tag >= 0x30 && p.tag <= 0x32) {
    if (d0.dtype == 0)
      m1_reduce_float(p.tag, a0, o0, temporary, d0);
    else
      m1_reduce_integer(p.tag, a0, o0, temporary, d0);
    return;
  }
  if (p.tag == 0x33) {
    m1_reduce_argmax(a0, o0, temporary, d0);
    return;
  }
  if (p.tag == 0x38) {
    for (m1_u32 linear = 0; linear < out0.len; ++linear) {
      m1_u32 rem = linear;
      m1_u32 source_index = 0;
      m1_u32 source_stride[4] = {1, 1, 1, 1};
      for (int dim = (int)out0.rank - 2; dim >= 0; --dim) {
        source_stride[dim] =
            source_stride[dim + 1] *
            ((m1_u32)(dim + 1) < d0.rank ? d0.dims[dim + 1] : 1u);
      }
      for (m1_u32 dim = 0; dim < out0.rank; ++dim) {
        m1_u32 stride = 1;
        for (m1_u32 next = dim + 1; next < out0.rank; ++next)
          stride *= out0.dims[next];
        if (stride == 0) stride = 1;
        const m1_u32 coordinate = rem / stride;
        rem %= stride;
        const m1_u32 source_dim =
            dim < d0.rank ? d0.dims[dim] : 1u;
        if (source_dim != 1)
          source_index += coordinate * source_stride[dim];
      }
      if (out0.dtype == 0)
        m1_store_f(o0, linear, m1_load_f(a0, source_index, d0.dtype));
      else if (out0.dtype == 1)
        m1_store_i(o0, linear, m1_load_i(a0, source_index, d0.dtype));
      else if (out0.dtype == 2)
        m1_store_u(o0, linear, m1_load_u(a0, source_index, d0.dtype));
      else
        m1_store_b(o0, linear, m1_load_b(a0, source_index, d0.dtype));
    }
    return;
  }
  if (p.tag == 0x39) {
    m1_copy_typed(a0, o0, out0.len, out0.dtype);
    return;
  }
  if (p.tag == 0x3A) {
    if (d0.rank != 2) {
      m1_fault(status, p.tag);
      return;
    }
    const m1_u32 m = d0.dims[0];
    const m1_u32 n = d0.dims[1];
    for (m1_u32 index = 0; index < m * n; ++index) {
      const m1_u32 source_index = (index % m) * n + index / m;
      if (out0.dtype == 0)
        m1_store_f(
            o0, index, m1_load_f(a0, source_index, d0.dtype));
      else if (out0.dtype == 1)
        m1_store_i(
            o0, index, m1_load_i(a0, source_index, d0.dtype));
      else if (out0.dtype == 2)
        m1_store_u(
            o0, index, m1_load_u(a0, source_index, d0.dtype));
      else
        m1_store_b(
            o0, index, m1_load_b(a0, source_index, d0.dtype));
    }
    return;
  }
  if (p.tag == 0x40 || p.tag == 0x41) {
    for (m1_u32 row = 0; row < d0.rows; ++row) {
      float accumulated = p.tag == 0x40 ? 0.0f : 1.0f;
      for (m1_u32 column = 0; column < d0.last; ++column) {
        const m1_u32 index = row * d0.last + column;
        const float value = m1_load_f(a0, index, d0.dtype);
        accumulated =
            p.tag == 0x40 ? accumulated + value : accumulated * value;
        m1_store_f(o0, index, accumulated);
      }
    }
    return;
  }
  if (p.tag == 0x50) {
    for (m1_u32 position = 0; position < d0.len; ++position) {
      m1_u32 best_index = 0;
      float best_value = m1_nan();
      bool found = false;
      for (m1_u32 candidate = 0; candidate < d0.len; ++candidate) {
        bool used = false;
        for (m1_u32 prior = 0; prior < position; ++prior)
          if (reinterpret_cast<m1_u32*>(o1)[prior] == candidate)
            used = true;
        if (used) continue;
        const float value = m1_load_f(a0, candidate, d0.dtype);
        if (!found ||
            m1_sort_better(value, candidate, best_value, best_index)) {
          found = true;
          best_value = value;
          best_index = candidate;
        }
      }
      m1_store_f(o0, position, best_value);
      m1_store_u(o1, position, best_index);
    }
    return;
  }
  if (p.tag == 0x51) {
    const m1_u32 count = p.imm < d0.last ? p.imm : d0.last;
    for (m1_u32 row = 0; row < d0.rows; ++row) {
      for (m1_u32 position = 0; position < count; ++position) {
        m1_u32 best_index = 0;
        float best_value = m1_nan();
        bool found = false;
        for (m1_u32 candidate = 0; candidate < d0.last; ++candidate) {
          bool used = false;
          for (m1_u32 prior = 0; prior < position; ++prior)
            if (reinterpret_cast<m1_u32*>(o1)[row * count + prior] ==
                candidate)
              used = true;
          if (used) continue;
          const float value =
              m1_load_f(a0, row * d0.last + candidate, d0.dtype);
          if (!found ||
              m1_sort_better(value, candidate, best_value, best_index)) {
            found = true;
            best_value = value;
            best_index = candidate;
          }
        }
        m1_store_f(o0, row * count + position, best_value);
        m1_store_u(o1, row * count + position, best_index);
      }
    }
    return;
  }
  if (p.tag == 0x55) {
    if (d0.rank != 2 || d1.rank != 2) {
      m1_fault(status, p.tag);
      return;
    }
    const m1_u32 m = d0.dims[0];
    const m1_u32 inner = d0.dims[1];
    const m1_u32 n = d1.dims[1];
    for (m1_u32 row = 0; row < m; ++row)
      for (m1_u32 column = 0; column < n; ++column)
        m1_store_f(o0, row * n + column, 0.0f);
    for (m1_u32 row = 0; row < m; ++row)
      for (m1_u32 k = 0; k < inner; ++k) {
        const float left =
            m1_load_f(a0, row * inner + k, d0.dtype);
        if (left == 0.0f) continue;
        for (m1_u32 column = 0; column < n; ++column) {
          const m1_u32 index = row * n + column;
          const float old = m1_load_f(o0, index, 0);
          m1_store_f(
              o0,
              index,
              old + left *
                  m1_load_f(a1, k * n + column, d1.dtype));
        }
      }
    return;
  }
  if (p.tag == 0x58) {
    for (m1_u32 row = 0; row < d0.rows; ++row) {
      const m1_u32 base = row * d0.last;
      if (p.pred_tag == 0) {
        m1_u32 k;
        if (d1.dtype == 1u) {
          const int signed_k =
              m1_load_i(a1, m1_pick(d1.len, row), d1.dtype);
          k = signed_k <= 0 ? 0u : (m1_u32)signed_k;
        } else {
          k = m1_load_u(a1, m1_pick(d1.len, row), d1.dtype);
        }
        if (k > d0.last) k = d0.last;
        for (m1_u32 i = 0; i < d0.last; ++i) {
          const float value = m1_load_f(a0, base + i, d0.dtype);
          int greater = 0;
          if (!m1_isnan(value)) {
            for (m1_u32 j = 0; j < d0.last; ++j) {
              const float other = m1_load_f(a0, base + j, d0.dtype);
              if (!m1_isnan(other) && other > value) ++greater;
            }
          }
          m1_store_b(
              o0,
              base + i,
              !m1_isnan(value) && (m1_u32)greater < k);
        }
      } else if (p.pred_tag == 1) {
        const float threshold =
            m1_load_f(a1, m1_pick(d1.len, row), d1.dtype);
        float exclusive = 0.0f;
        for (m1_u32 position = 0; position < d0.last; ++position) {
          m1_u32 best_index = 0;
          float best_value = m1_nan();
          bool found = false;
          for (m1_u32 candidate = 0;
               candidate < d0.last;
               ++candidate) {
            bool used = false;
            for (m1_u32 prior = 0; prior < position; ++prior)
              if (reinterpret_cast<m1_u32*>(temporary)[prior] ==
                  candidate)
                used = true;
            if (used) continue;
            const float value =
                m1_load_f(a0, base + candidate, d0.dtype);
            if (!found ||
                m1_sort_better(
                    value, candidate, best_value, best_index)) {
              found = true;
              best_value = value;
              best_index = candidate;
            }
          }
          reinterpret_cast<m1_u32*>(temporary)[position] = best_index;
          m1_store_b(o0, base + best_index, exclusive < threshold);
          exclusive += best_value;
        }
      } else {
        const float threshold =
            m1_load_f(a1, m1_pick(d1.len, row), d1.dtype);
        for (m1_u32 i = 0; i < d0.last; ++i)
          m1_store_b(
              o0,
              base + i,
              m1_load_f(a0, base + i, d0.dtype) >= threshold);
      }
    }
    return;
  }
  if (p.tag == 0x60) {
    m1_u32 rest = 1u;
    m1_u32 n0 = 1u;
    if (d0.rank != 0) {
      n0 = d0.dims[0];
      rest = n0 == 0 ? 1u : d0.len / n0;
    }
    for (m1_u32 k = 0; k < d1.len; ++k) {
      const long long index = m1_load_index(a1, k, d1.dtype);
      for (m1_u32 r = 0; r < rest; ++r) {
        const m1_u32 output_index = k * rest + r;
        const bool valid = index >= 0 && (m1_u64)index < n0;
        const m1_u32 source_index =
            valid ? (m1_u32)index * rest + r : 0;
        if (out0.dtype == 0)
          m1_store_f(
              o0,
              output_index,
              valid ? m1_load_f(a0, source_index, d0.dtype) : 0.0f);
        else if (out0.dtype == 1)
          m1_store_i(
              o0,
              output_index,
              valid ? m1_load_i(a0, source_index, d0.dtype) : 0);
        else if (out0.dtype == 2)
          m1_store_u(
              o0,
              output_index,
              valid ? m1_load_u(a0, source_index, d0.dtype) : 0u);
        else
          m1_store_b(
              o0,
              output_index,
              valid && m1_load_b(a0, source_index, d0.dtype));
      }
    }
    return;
  }
  if (p.tag == 0x61) {
    const m1_u32 rows = d0.dims[0];
    const m1_u32 columns = d0.dims[1];
    for (m1_u32 row = 0; row < rows; ++row) {
      const long long column = m1_load_index(a1, row, d1.dtype);
      const bool valid = column >= 0 && (m1_u64)column < columns;
      const m1_u32 source_index =
          valid ? row * columns + (m1_u32)column : 0;
      if (out0.dtype == 0)
        m1_store_f(
            o0,
            row,
            valid ? m1_load_f(a0, source_index, d0.dtype) : 0.0f);
      else if (out0.dtype == 1)
        m1_store_i(
            o0,
            row,
            valid ? m1_load_i(a0, source_index, d0.dtype) : 0);
      else if (out0.dtype == 2)
        m1_store_u(
            o0,
            row,
            valid ? m1_load_u(a0, source_index, d0.dtype) : 0u);
      else
        m1_store_b(
            o0,
            row,
            valid && m1_load_b(a0, source_index, d0.dtype));
    }
    return;
  }
  if (p.tag == 0x62 || p.tag == 0x63) {
    m1_copy_typed(a0, o0, d0.len, d0.dtype);
    m1_u32 rest = 1u;
    m1_u32 n0 = 1u;
    if (d0.rank != 0) {
      n0 = d0.dims[0];
      rest = n0 == 0 ? 1u : d0.len / n0;
    }
    const bool scalar = d2.len == 1 && d1.len * rest != 1;
    for (m1_u32 k = 0; k < d1.len; ++k) {
      const long long index = m1_load_index(a1, k, d1.dtype);
      if (index < 0 || (m1_u64)index >= n0) continue;
      for (m1_u32 r = 0; r < rest; ++r) {
        const m1_u32 dst = (m1_u32)index * rest + r;
        const m1_u32 src = scalar ? 0u : k * rest + r;
        if (d0.dtype == 0) {
          const float value = m1_load_f(a2, src, d2.dtype);
          m1_store_f(
              o0,
              dst,
              p.tag == 0x62 ? m1_load_f(o0, dst, 0) + value : value);
        } else if (d0.dtype == 1) {
          const int value = m1_load_i(a2, src, d2.dtype);
          m1_store_i(
              o0,
              dst,
              p.tag == 0x62
                  ? m1_bits_i32(
                        (m1_u32)m1_load_i(o0, dst, 1) +
                        (m1_u32)value)
                  : value);
        } else if (d0.dtype == 2) {
          const m1_u32 value = m1_load_u(a2, src, d2.dtype);
          m1_store_u(
              o0,
              dst,
              p.tag == 0x62 ? m1_load_u(o0, dst, 2) + value : value);
        } else {
          m1_store_b(o0, dst, m1_load_b(a2, src, d2.dtype));
        }
      }
    }
    return;
  }
  if (p.tag == 0x64) {
    for (m1_u32 i = 0; i < out0.len; ++i) m1_store_u(o0, i, i);
    return;
  }
  if (p.tag == 0x65) {
    const m1_u32 mask_width =
        d0.rank == 0 ? 1u : d0.dims[d0.rank - 1];
    for (m1_u32 i = 0; i < d0.len; ++i) {
      const m1_u32 column = i % mask_width;
      const m1_u32 word = column >> 5;
      const m1_u32 mask =
          word < d1.len ? m1_load_u(a1, word, d1.dtype) : 0u;
      m1_store_f(
          o0,
          i,
          ((mask >> (column & 31)) & 1u) != 0
              ? m1_load_f(a0, i, d0.dtype)
              : m1_neg_inf());
    }
    return;
  }
  if (p.tag == 0x66 || p.tag == 0x67 || p.tag == 0x68) {
    const m1_u32 key_count = p.imm;
    const m1_u32 window = p.tag == 0x67 ? p.imm2 : p.imm3;
    for (m1_u32 index = 0; index < out0.len; ++index) {
      const m1_u32 position_index =
          key_count == 0u ? 0u : index / key_count;
      const m1_u32 key =
          key_count == 0u ? 0u : index % key_count;
      const m1_u32 position =
          m1_load_u(a0, position_index, d0.dtype);
      bool allowed = key_count != 0u && key <= position;
      if (allowed && p.tag != 0x66) {
        const m1_u32 reach =
            key > 0xffffffffu - window ? 0xffffffffu : key + window;
        const bool recent = reach > position;
        allowed =
            p.tag == 0x67 ? recent : (key < p.imm2 || recent);
      }
      m1_store_b(o0, index, allowed);
    }
    return;
  }
  if (p.tag == 0x70 || p.tag == 0x71) {
    if (p.tag == 0x70) {
      const m1_u64 seed =
          ptir_rng_seed_eff_stream((m1_u32)p.rng_seed, p.imm);
      for (m1_u32 i = 0; i < out0.len; ++i) {
        const float uniform = ptir_rng_hash_uniform(seed, i);
        m1_store_f(
            o0,
            i,
            p.kind == 0 ? uniform : -logf(-logf(uniform)));
      }
    } else {
      const m1_u64 key = (m1_u64)m1_load_u(a0, 0, d0.dtype);
      const m1_u64 counter =
          (m1_u64)(d0.len > 1 ? m1_load_u(a0, 1, d0.dtype) : 0u);
      const m1_u64 seed =
          ptir_rng_keyed_seed((m1_u32)key, (m1_u32)counter);
      for (m1_u32 i = 0; i < out0.len; ++i) {
        const float uniform = ptir_rng_hash_uniform(seed, i);
        m1_store_f(
            o0,
            i,
            p.kind == 0 ? uniform : -logf(-logf(uniform)));
      }
    }
    return;
  }
  m1_fault(status, p.tag);
}
)PTIR_CUDA";
    return source;
}

inline GeneratedKernelSource emit_singleton_region_cuda(
    const std::string& entry_name,
    std::uint8_t op_tag) {
    GeneratedKernelSource result;
    result.entry_name = entry_name;
    result.op_tag = op_tag;
    if (!detail::valid_identifier(entry_name)) {
        result.error = "CUDA singleton entry name is not a C identifier";
        return result;
    }
    if (!detail::supported_tag(op_tag)) {
        result.error =
            "unsupported CUDA singleton opcode tag " +
            std::to_string(op_tag);
        return result;
    }
    result.source = singleton_runtime_cuda_source();
    result.source += "\nextern \"C\" __global__ void ";
    result.source += entry_name;
    result.source += R"PTIR_CUDA((
    M1Status* status,
    const M1ValueDesc* descriptors,
    const m1_u8* a0,
    const m1_u8* a1,
    const m1_u8* a2,
    m1_u8* o0,
    m1_u8* o1,
    m1_u8* temporary,
    const M1OpParams* params) {
  if (blockIdx.x != 0 || blockIdx.y != 0 || blockIdx.z != 0 ||
      threadIdx.x != 0 || threadIdx.y != 0 || threadIdx.z != 0)
    return;
  ptir_m1_execute()PTIR_CUDA";
    result.source += std::to_string(static_cast<unsigned int>(op_tag));
    result.source +=
        "u, status, descriptors, params, a0, a1, a2, o0, o1, temporary);\n"
        "}\n";
    result.ok = true;
    return result;
}

}  // namespace pie_cuda_driver::pipeline::generated
