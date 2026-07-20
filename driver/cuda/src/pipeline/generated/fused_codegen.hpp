#pragma once

#include <cstdint>
#include <limits>
#include <optional>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "pipeline/generated/singleton_codegen.hpp"

namespace pie_cuda_driver::pipeline::generated {

inline constexpr std::uint16_t kCudaGeneratedEmitterVersion = 18;

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

inline GeneratedKernelSource emit_fused_region_cuda(
    const std::string& entry_name,
    const pie_native::ptir::plan::StagePlan& stage,
    const pie_native::ptir::plan::Region& region) {
    GeneratedKernelSource result;
    result.entry_name = entry_name;
    if (!detail::valid_identifier(entry_name)) {
        result.error = "CUDA fused entry name is not a C identifier";
        return result;
    }
    if (!validate_generated_region(stage, region, result.error)) {
        return result;
    }

    std::vector<std::uint32_t> bases(stage.ops.size(), 0);
    std::uint32_t next_value = 0;
    for (std::size_t node = 0; node < stage.ops.size(); ++node) {
        bases[node] = next_value;
        next_value += stage.ops[node].op.results;
    }
    if (next_value != stage.value_types.size()) {
        result.error = "fused stage value layout does not match normalized ops";
        return result;
    }
    // Diagnostic (`PIE_STAGE_OP_DUMP`): print the lowered op list for each
    // emitted region — the ground truth for why a pattern analysis (e.g.
    // `analyze_direct_argmax`) accepts or skips a program's shape.
    if (std::getenv("PIE_STAGE_OP_DUMP") != nullptr) {
        std::fprintf(stderr, "[stage-op-dump] region entry=%s nodes=%zu\n",
                     entry_name.c_str(), region.nodes.size());
        for (const std::uint32_t node : region.nodes) {
            const auto& op = stage.ops[node].op;
            std::string args;
            for (const std::uint32_t argument : op.args) {
                args += std::to_string(argument);
                args += ',';
            }
            std::fprintf(
                stderr,
                "[stage-op-dump]  node=%u base=%u tag=%u intr=%u results=%u args=[%s]\n",
                node, bases[node], static_cast<unsigned>(op.tag),
                static_cast<unsigned>(op.intr),
                static_cast<unsigned>(op.results), args.c_str());
        }
    }
    std::vector<std::uint32_t> aliases(next_value);
    for (std::uint32_t value = 0; value < next_value; ++value) {
        aliases[value] = value;
    }
    auto resolve_alias = [&](std::uint32_t value) {
        while (aliases[value] != value) value = aliases[value];
        return value;
    };
    const DirectArgmaxAnalysis direct_argmax =
        analyze_direct_argmax(stage, region, bases);
    const auto& direct_argmax_intrinsic = direct_argmax.intrinsic;
    auto direct_argmax_skipped = direct_argmax.skipped;
    for (const auto& candidate : stage.fused.regions) {
        if (!candidate.library ||
            candidate.library_op != PTIR_LIBRARY_NUCLEUS_SAMPLE ||
            candidate.inputs.size() != 5) {
            continue;
        }
        for (std::uint32_t value :
             {candidate.inputs[0], candidate.inputs[2]}) {
            for (std::size_t depth = 0; depth < 2; ++depth) {
                bool found = false;
                for (const std::uint32_t node : region.nodes) {
                    const auto& op = stage.ops[node].op;
                    if (value < bases[node] ||
                        value >= bases[node] + op.results) {
                        continue;
                    }
                    found = true;
                    direct_argmax_skipped[node] = 1;
                    if (op.tag == PTIR_OP_RESHAPE &&
                        !op.args.empty()) {
                        value = op.args[0];
                    } else {
                        depth = 2;
                    }
                    break;
                }
                if (!found) break;
            }
        }
    }
    std::ostringstream source;
    source << singleton_runtime_cuda_source() << R"PTIR_CUDA(

struct PtirLaneTableHeader {
  m1_u32 abi_version;
  m1_u32 lane_count;
  m1_u32 channel_slots_per_lane;
  m1_u32 flags;
};

struct PtirLaneRecord {
  m1_u64 logits_base;
  m1_u32 logits_row_offset;
  m1_u32 logits_row_count;
  m1_u32 kv_len;
  m1_u32 page_count;
  m1_u32 row_count;
  m1_u32 token_count;
  m1_u32 sampled_rows;
  m1_u32 query_len;
  m1_u32 key_len;
  m1_u32 channel_slot_offset;
  m1_u64 rng_state;
  m1_u64 commit_slot;
  m1_u64 active_row_mask;
  m1_u64 sample_output_channel_mask;
  m1_u64 row_valid;
  m1_u32 row_valid_offset;
  m1_u32 reserved0;
};

struct PtirLaneChannelSlot {
  m1_u64 committed_cell;
  m1_u64 pending_cell;
  m1_u64 expected_head;
  m1_u64 expected_tail;
};

static_assert(sizeof(PtirLaneTableHeader) == 16, "lane header ABI");
static_assert(sizeof(PtirLaneRecord) == 96, "lane record ABI");
static_assert(sizeof(PtirLaneChannelSlot) == 32, "lane channel ABI");

__device__ __forceinline__ void ptir_parallel_copy(
    const m1_u8* input,
    m1_u8* output,
    m1_u32 len,
    m1_u32 dtype) {
  const m1_u32 bytes = dtype == 3u ? len : len * 4u;
  for (m1_u32 index = threadIdx.x; index < bytes; index += blockDim.x)
    output[index] = input[index];
}

__device__ __forceinline__ void ptir_parallel_intrinsic(
    const m1_u8* input,
    m1_u8* output,
    const M1ValueDesc output_desc,
    const M1OpParams p) {
  const m1_u32 width = output_desc.last == 0u ? p.imm : output_desc.last;
  const m1_u32 stride =
      p.intrinsic_row_stride == 0u ? width : p.intrinsic_row_stride;
  const m1_u64 first_row =
      (m1_u64)p.intrinsic_row_offset + (m1_u64)p.imm2;
  for (m1_u32 index = threadIdx.x;
       index < output_desc.len;
       index += blockDim.x) {
    const m1_u32 row = index / width;
    const m1_u32 column = index % width;
    m1_store_f(
        output,
        index,
        m1_intrinsic_row_load(
            input,
            first_row + row,
            column,
            stride,
            p.intrinsic_dtype));
  }
}

__device__ __forceinline__ void ptir_parallel_elementwise(
    m1_u32 tag,
    M1Status* status,
    const M1ValueDesc* descriptors,
    const M1OpParams p,
    const m1_u8* a0,
    const m1_u8* a1,
    const m1_u8* a2,
    m1_u8* o0) {
  const M1ValueDesc d0 = descriptors[p.a0];
  const M1ValueDesc d1 = descriptors[p.a1];
  const M1ValueDesc d2 = descriptors[p.a2];
  const M1ValueDesc out = descriptors[p.o0];
  for (m1_u32 i = threadIdx.x; i < out.len; i += blockDim.x) {
    const m1_u32 xindex = m1_pick(d0.len, i);
    const m1_u32 yindex = m1_pick(d1.len, i);
    if (tag == 0x01u || tag == 0x02u || tag == 0x04u) {
      const float value = m1_load_f(a0, xindex, d0.dtype);
      m1_store_f(
          o0,
          i,
          tag == 0x01u
              ? expf(value)
              : (tag == 0x02u ? logf(value) : 1.0f / value));
      continue;
    }
    if (tag == 0x03u || tag == 0x05u || tag == 0x06u) {
      if (d0.dtype == 0u) {
        const float value = m1_load_f(a0, xindex, d0.dtype);
        const float result =
            tag == 0x03u
                ? -value
                : (tag == 0x05u
                       ? fabsf(value)
                       : (value > 0.0f
                              ? 1.0f
                              : (value < 0.0f ? -1.0f : 0.0f)));
        m1_store_f(o0, i, result);
      } else if (d0.dtype == 1u) {
        const int value = m1_load_i(a0, xindex, d0.dtype);
        int result = value;
        if (tag == 0x03u) result = (int)(0u - (m1_u32)value);
        else if (tag == 0x05u)
          result = (m1_u32)value == 0x80000000u
              ? value
              : (value < 0 ? -value : value);
        else result = value > 0 ? 1 : (value < 0 ? -1 : 0);
        m1_store_i(o0, i, result);
      } else if (d0.dtype == 2u) {
        const m1_u32 value = m1_load_u(a0, xindex, d0.dtype);
        m1_store_u(
            o0,
            i,
            tag == 0x03u
                ? 0u - value
                : (tag == 0x06u ? (value != 0u ? 1u : 0u) : value));
      } else {
        if (threadIdx.x == 0) m1_fault(status, tag);
        return;
      }
      continue;
    }
    if (tag == 0x07u) {
      if (out.dtype == 0u)
        m1_store_f(o0, i, m1_load_f(a0, xindex, d0.dtype));
      else if (out.dtype == 1u)
        m1_store_i(o0, i, m1_load_i(a0, xindex, d0.dtype));
      else if (out.dtype == 2u)
        m1_store_u(o0, i, m1_load_u(a0, xindex, d0.dtype));
      else
        m1_store_b(o0, i, m1_load_b(a0, xindex, d0.dtype));
      continue;
    }
    if ((tag >= 0x10u && tag <= 0x1du) || tag == 0x1fu) {
      if (tag >= 0x16u && tag <= 0x1du) {
        bool result = false;
        if (tag == 0x1cu || tag == 0x1du) {
          const bool left = m1_load_b(a0, xindex, d0.dtype);
          const bool right = m1_load_b(a1, yindex, d1.dtype);
          result = tag == 0x1cu ? left && right : left || right;
        } else if (d0.dtype == 0u) {
          const float left = m1_load_f(a0, xindex, d0.dtype);
          const float right = m1_load_f(a1, yindex, d1.dtype);
          if (tag == 0x16u) result = left > right;
          else if (tag == 0x17u) result = left >= right;
          else if (tag == 0x18u) result = left == right;
          else if (tag == 0x19u) result = left != right;
          else if (tag == 0x1au) result = left < right;
          else result = left <= right;
        } else if (d0.dtype == 1u) {
          const int left = m1_load_i(a0, xindex, d0.dtype);
          const int right = m1_load_i(a1, yindex, d1.dtype);
          if (tag == 0x16u) result = left > right;
          else if (tag == 0x17u) result = left >= right;
          else if (tag == 0x18u) result = left == right;
          else if (tag == 0x19u) result = left != right;
          else if (tag == 0x1au) result = left < right;
          else result = left <= right;
        } else {
          const m1_u32 left = m1_load_u(a0, xindex, d0.dtype);
          const m1_u32 right = m1_load_u(a1, yindex, d1.dtype);
          if (tag == 0x16u) result = left > right;
          else if (tag == 0x17u) result = left >= right;
          else if (tag == 0x18u) result = left == right;
          else if (tag == 0x19u) result = left != right;
          else if (tag == 0x1au) result = left < right;
          else result = left <= right;
        }
        m1_store_b(o0, i, result);
      } else if (d0.dtype == 0u) {
        const float left = m1_load_f(a0, xindex, d0.dtype);
        const float right = m1_load_f(a1, yindex, d1.dtype);
        float result = 0.0f;
        if (tag == 0x10u) result = left + right;
        else if (tag == 0x11u) result = left - right;
        else if (tag == 0x12u) result = left * right;
        else if (tag == 0x13u) result = left / right;
        else if (tag == 0x14u) result = m1_element_max(left, right);
        else if (tag == 0x15u) result = m1_element_min(left, right);
        else result = fmodf(left, right);
        m1_store_f(o0, i, result);
      } else if (d0.dtype == 1u) {
        const int left = m1_load_i(a0, xindex, d0.dtype);
        const int right = m1_load_i(a1, yindex, d1.dtype);
        int result = 0;
        if (tag == 0x10u) result = (int)((m1_u32)left + (m1_u32)right);
        else if (tag == 0x11u)
          result = (int)((m1_u32)left - (m1_u32)right);
        else if (tag == 0x12u)
          result = (int)((m1_u32)left * (m1_u32)right);
        else if (tag == 0x13u) result = m1_i32_div(left, right);
        else if (tag == 0x14u) result = left > right ? left : right;
        else if (tag == 0x15u) result = left < right ? left : right;
        else result = m1_i32_rem(left, right);
        m1_store_i(o0, i, result);
      } else {
        const m1_u32 left = m1_load_u(a0, xindex, d0.dtype);
        const m1_u32 right = m1_load_u(a1, yindex, d1.dtype);
        m1_u32 result = 0u;
        if (tag == 0x10u) result = left + right;
        else if (tag == 0x11u) result = left - right;
        else if (tag == 0x12u) result = left * right;
        else if (tag == 0x13u) result = right == 0u ? 0u : left / right;
        else if (tag == 0x14u) result = left > right ? left : right;
        else if (tag == 0x15u) result = left < right ? left : right;
        else result = right == 0u ? 0u : left % right;
        m1_store_u(o0, i, result);
      }
      continue;
    }
    if (tag == 0x1eu) {
      m1_store_b(o0, i, !m1_load_b(a0, xindex, d0.dtype));
      continue;
    }
    if (tag == 0x20u) {
      const bool condition = m1_load_b(a0, xindex, d0.dtype);
      const m1_u32 left_index = m1_pick(d1.len, i);
      const m1_u32 right_index = m1_pick(d2.len, i);
      if (out.dtype == 0u)
        m1_store_f(
            o0,
            i,
            condition
                ? m1_load_f(a1, left_index, d1.dtype)
                : m1_load_f(a2, right_index, d2.dtype));
      else if (out.dtype == 1u)
        m1_store_i(
            o0,
            i,
            condition
                ? m1_load_i(a1, left_index, d1.dtype)
                : m1_load_i(a2, right_index, d2.dtype));
      else if (out.dtype == 2u)
        m1_store_u(
            o0,
            i,
            condition
                ? m1_load_u(a1, left_index, d1.dtype)
                : m1_load_u(a2, right_index, d2.dtype));
      else
        m1_store_b(
            o0,
            i,
            condition
                ? m1_load_b(a1, left_index, d1.dtype)
                : m1_load_b(a2, right_index, d2.dtype));
      continue;
    }
    if (tag == 0x64u) {
      m1_store_u(o0, i, i);
      continue;
    }
    if (tag == 0x65u) {
      const m1_u32 width = d0.rank == 0u ? 1u : d0.dims[d0.rank - 1u];
      const m1_u32 column = i % width;
      const m1_u32 word = column >> 5;
      const m1_u32 mask =
          word < d1.len ? m1_load_u(a1, word, d1.dtype) : 0u;
      m1_store_f(
          o0,
          i,
          ((mask >> (column & 31u)) & 1u) != 0u
              ? m1_load_f(a0, i, d0.dtype)
              : m1_neg_inf());
      continue;
    }
    if (tag == 0x66u || tag == 0x67u || tag == 0x68u) {
      const m1_u32 key_count = p.imm;
      const m1_u32 position_index =
          key_count == 0u ? 0u : i / key_count;
      const m1_u32 key = key_count == 0u ? 0u : i % key_count;
      const m1_u32 position =
          m1_load_u(a0, position_index, d0.dtype);
      bool allowed = key_count != 0u && key <= position;
      if (allowed && tag != 0x66u) {
        const m1_u32 window = tag == 0x67u ? p.imm2 : p.imm3;
        const m1_u32 reach =
            key > 0xffffffffu - window ? 0xffffffffu : key + window;
        const bool recent = reach > position;
        allowed = tag == 0x67u ? recent : (key < p.imm2 || recent);
      }
      m1_store_b(o0, i, allowed);
      continue;
    }
    if (tag == 0x70u || tag == 0x71u) {
      m1_u64 seed;
      if (tag == 0x70u) {
        seed = ptir_rng_seed_eff_stream((m1_u32)p.rng_seed, p.imm);
      } else {
        const m1_u32 key = m1_load_u(a0, 0u, d0.dtype);
        const m1_u32 counter =
            d0.len > 1u ? m1_load_u(a0, 1u, d0.dtype) : 0u;
        seed = ptir_rng_keyed_seed(key, counter);
      }
      const float uniform = ptir_rng_hash_uniform(seed, i);
      m1_store_f(
          o0,
          i,
          p.kind == 0u ? uniform : -logf(-logf(uniform)));
      continue;
    }
  }
}

__device__ __forceinline__ void ptir_copy_element(
    const m1_u8* input,
    m1_u8* output,
    m1_u32 source,
    m1_u32 destination,
    m1_u32 dtype) {
  if (dtype == 0u)
    m1_store_f(output, destination, m1_load_f(input, source, dtype));
  else if (dtype == 1u)
    m1_store_i(output, destination, m1_load_i(input, source, dtype));
  else if (dtype == 2u)
    m1_store_u(output, destination, m1_load_u(input, source, dtype));
  else
    m1_store_b(output, destination, m1_load_b(input, source, dtype));
}

__device__ __forceinline__ void ptir_zero_element(
    m1_u8* output,
    m1_u32 destination,
    m1_u32 dtype) {
  if (dtype == 0u)
    m1_store_f(output, destination, 0.0f);
  else if (dtype == 1u)
    m1_store_i(output, destination, 0);
  else if (dtype == 2u)
    m1_store_u(output, destination, 0u);
  else
    m1_store_b(output, destination, false);
}

__device__ __forceinline__ void ptir_parallel_broadcast(
    const m1_u8* input,
    m1_u8* output,
    const M1ValueDesc input_desc,
    const M1ValueDesc output_desc) {
  for (m1_u32 linear = threadIdx.x;
       linear < output_desc.len;
       linear += blockDim.x) {
    m1_u32 rem = linear;
    m1_u32 source_index = 0;
    m1_u32 source_stride[4] = {1, 1, 1, 1};
    for (int dim = (int)output_desc.rank - 2; dim >= 0; --dim) {
      source_stride[dim] =
          source_stride[dim + 1] *
          ((m1_u32)(dim + 1) < input_desc.rank
               ? input_desc.dims[dim + 1]
               : 1u);
    }
    for (m1_u32 dim = 0; dim < output_desc.rank; ++dim) {
      m1_u32 stride = 1;
      for (m1_u32 next = dim + 1; next < output_desc.rank; ++next)
        stride *= output_desc.dims[next];
      if (stride == 0u) stride = 1u;
      const m1_u32 coordinate = rem / stride;
      rem %= stride;
      const m1_u32 source_dim =
          dim < input_desc.rank ? input_desc.dims[dim] : 1u;
      if (source_dim != 1u)
        source_index += coordinate * source_stride[dim];
    }
    ptir_copy_element(
        input, output, source_index, linear, output_desc.dtype);
  }
}

__device__ __forceinline__ void ptir_parallel_transpose(
    M1Status* status,
    const m1_u8* input,
    m1_u8* output,
    const M1ValueDesc input_desc,
    const M1ValueDesc output_desc) {
  if (input_desc.rank != 2u) {
    if (threadIdx.x == 0u) m1_fault(status, 0x3au);
    return;
  }
  const m1_u32 rows = input_desc.dims[0];
  const m1_u32 columns = input_desc.dims[1];
  for (m1_u32 index = threadIdx.x;
       index < rows * columns;
       index += blockDim.x) {
    const m1_u32 source =
        (index % rows) * columns + index / rows;
    ptir_copy_element(
        input, output, source, index, output_desc.dtype);
  }
}

__device__ __forceinline__ void ptir_parallel_gather(
    m1_u32 tag,
    const m1_u8* input,
    const m1_u8* indices,
    m1_u8* output,
    const M1ValueDesc input_desc,
    const M1ValueDesc index_desc,
    const M1ValueDesc output_desc) {
  if (tag == 0x61u) {
    const m1_u32 rows = input_desc.dims[0];
    const m1_u32 columns = input_desc.dims[1];
    for (m1_u32 row = threadIdx.x; row < rows; row += blockDim.x) {
      const long long column =
          m1_load_index(indices, row, index_desc.dtype);
      const bool valid =
          column >= 0 && (m1_u64)column < columns;
      if (valid)
        ptir_copy_element(
            input,
            output,
            row * columns + (m1_u32)column,
            row,
            output_desc.dtype);
      else
        ptir_zero_element(output, row, output_desc.dtype);
    }
    return;
  }
  m1_u32 rest = 1u;
  m1_u32 rows = 1u;
  if (input_desc.rank != 0u) {
    rows = input_desc.dims[0];
    rest = rows == 0u ? 1u : input_desc.len / rows;
  }
  const m1_u32 total = index_desc.len * rest;
  for (m1_u32 output_index = threadIdx.x;
       output_index < total;
       output_index += blockDim.x) {
    const m1_u32 k = output_index / rest;
    const m1_u32 r = output_index % rest;
    const long long row =
        m1_load_index(indices, k, index_desc.dtype);
    const bool valid = row >= 0 && (m1_u64)row < rows;
    if (valid)
      ptir_copy_element(
          input,
          output,
          (m1_u32)row * rest + r,
          output_index,
          output_desc.dtype);
    else
      ptir_zero_element(output, output_index, output_desc.dtype);
  }
}

__device__ __forceinline__ void ptir_parallel_pivot(
    const m1_u8* input,
    const m1_u8* threshold,
    m1_u8* output,
    const M1ValueDesc input_desc,
    const M1ValueDesc threshold_desc,
    const M1OpParams p) {
  for (m1_u32 flat = threadIdx.x;
       flat < input_desc.len;
       flat += blockDim.x) {
    const m1_u32 row =
        input_desc.last == 0u ? 0u : flat / input_desc.last;
    const m1_u32 column =
        input_desc.last == 0u ? 0u : flat % input_desc.last;
    const m1_u32 base = row * input_desc.last;
    const float value = m1_load_f(input, flat, input_desc.dtype);
    bool keep = false;
    if (p.pred_tag == 0u) {
      m1_u32 k;
      if (threshold_desc.dtype == 1u) {
        const int signed_k = m1_load_i(
            threshold,
            m1_pick(threshold_desc.len, row),
            threshold_desc.dtype);
        k = signed_k <= 0 ? 0u : (m1_u32)signed_k;
      } else {
        k = m1_load_u(
            threshold,
            m1_pick(threshold_desc.len, row),
            threshold_desc.dtype);
      }
      if (k > input_desc.last) k = input_desc.last;
      m1_u32 greater = 0;
      if (!m1_isnan(value)) {
        for (m1_u32 other_column = 0;
             other_column < input_desc.last;
             ++other_column) {
          const float other = m1_load_f(
              input, base + other_column, input_desc.dtype);
          if (!m1_isnan(other) && other > value) ++greater;
        }
      }
      keep = !m1_isnan(value) && greater < k;
    } else if (p.pred_tag == 2u) {
      const float cutoff = m1_load_f(
          threshold,
          m1_pick(threshold_desc.len, row),
          threshold_desc.dtype);
      keep = value >= cutoff;
    } else {
      // CummassLe consumes sorted inputs; preserve increasing-index prefix order.
      const float cutoff = m1_load_f(
          threshold,
          m1_pick(threshold_desc.len, row),
          threshold_desc.dtype);
      float exclusive = 0.0f;
      for (m1_u32 prior = 0; prior < column; ++prior)
        exclusive += m1_load_f(
            input, base + prior, input_desc.dtype);
      keep = exclusive < cutoff;
    }
    m1_store_b(output, flat, keep);
  }
}

__device__ __forceinline__ void ptir_scatter_updates(
    m1_u32 tag,
    const m1_u8* indices,
    const m1_u8* updates,
    m1_u8* output,
    const M1ValueDesc base_desc,
    const M1ValueDesc index_desc,
    const M1ValueDesc update_desc) {
  m1_u32 rest = 1u;
  m1_u32 rows = 1u;
  if (base_desc.rank != 0u) {
    rows = base_desc.dims[0];
    rest = rows == 0u ? 1u : base_desc.len / rows;
  }
  const bool scalar =
      update_desc.len == 1u && index_desc.len * rest != 1u;
  for (m1_u32 k = 0; k < index_desc.len; ++k) {
    const long long row =
        m1_load_index(indices, k, index_desc.dtype);
    if (row < 0 || (m1_u64)row >= rows) continue;
    for (m1_u32 r = 0; r < rest; ++r) {
      const m1_u32 destination = (m1_u32)row * rest + r;
      const m1_u32 source = scalar ? 0u : k * rest + r;
      if (base_desc.dtype == 0u) {
        const float value =
            m1_load_f(updates, source, update_desc.dtype);
        m1_store_f(
            output,
            destination,
            tag == 0x62u
                ? m1_load_f(output, destination, 0u) + value
                : value);
      } else if (base_desc.dtype == 1u) {
        const int value =
            m1_load_i(updates, source, update_desc.dtype);
        m1_store_i(
            output,
            destination,
            tag == 0x62u
                ? m1_bits_i32(
                      (m1_u32)m1_load_i(output, destination, 1u) +
                      (m1_u32)value)
                : value);
      } else if (base_desc.dtype == 2u) {
        const m1_u32 value =
            m1_load_u(updates, source, update_desc.dtype);
        m1_store_u(
            output,
            destination,
            tag == 0x62u
                ? m1_load_u(output, destination, 2u) + value
                : value);
      } else {
        m1_store_b(
            output,
            destination,
            m1_load_b(updates, source, update_desc.dtype));
      }
    }
  }
}

__device__ __forceinline__ void ptir_parallel_reduce_f32(
    m1_u32 tag,
    const m1_u8* input,
    m1_u8* output,
    m1_u8* temporary,
    const M1ValueDesc input_desc) {
  float* work_a = reinterpret_cast<float*>(temporary);
  const m1_u32 first_chunks = (input_desc.last + 31u) / 32u;
  float* work_b = work_a + first_chunks;
  const float* values = reinterpret_cast<const float*>(input);
  float* result = reinterpret_cast<float*>(output);
  const m1_u32 lane = threadIdx.x & 31u;
  const m1_u32 warp = threadIdx.x >> 5u;
  const m1_u32 warps = blockDim.x >> 5u;
  const unsigned mask = 0xffffffffu;
  for (m1_u32 row = 0; row < input_desc.rows; ++row) {
    const m1_u32 base = row * input_desc.last;
    if (input_desc.last == 0u) {
      if (threadIdx.x == 0u)
        result[row] =
            tag == 0x30u
                ? 0.0f
                : (tag == 0x31u ? m1_neg_inf() : m1_pos_inf());
      __syncthreads();
      continue;
    }
    for (m1_u32 chunk = warp;
         chunk < first_chunks;
         chunk += warps) {
      const m1_u32 index = chunk * 32u + lane;
      const float identity =
          tag == 0x30u
              ? 0.0f
              : (tag == 0x31u ? m1_neg_inf() : m1_pos_inf());
      float value =
          index < input_desc.last ? values[base + index] : identity;
      for (m1_u32 offset = 16u; offset > 0u; offset >>= 1u) {
        const float other = __shfl_down_sync(mask, value, offset);
        if (lane < offset) {
          value =
              tag == 0x30u
                  ? value + other
                  : (tag == 0x31u
                         ? m1_canonical_max(value, other)
                         : m1_canonical_min(value, other));
        }
      }
      if (lane == 0u) work_a[chunk] = value;
    }
    __syncthreads();
    m1_u32 count = first_chunks;
    float* reduction_input = work_a;
    float* reduction_output = work_b;
    while (count > 1u) {
      const m1_u32 chunks = (count + 31u) / 32u;
      for (m1_u32 chunk = warp;
           chunk < chunks;
           chunk += warps) {
        const m1_u32 index = chunk * 32u + lane;
        const float identity =
            tag == 0x30u
                ? 0.0f
                : (tag == 0x31u ? m1_neg_inf() : m1_pos_inf());
        float value =
            index < count ? reduction_input[index] : identity;
        for (m1_u32 offset = 16u; offset > 0u; offset >>= 1u) {
          const float other = __shfl_down_sync(mask, value, offset);
          if (lane < offset) {
            value =
                tag == 0x30u
                    ? value + other
                    : (tag == 0x31u
                           ? m1_canonical_max(
                                 value, other)
                           : m1_canonical_min(
                                 value, other));
          }
        }
        if (lane == 0u) reduction_output[chunk] = value;
      }
      __syncthreads();
      float* swap = reduction_input;
      reduction_input = reduction_output;
      reduction_output = swap;
      count = chunks;
    }
    if (threadIdx.x == 0u) result[row] = reduction_input[0];
    __syncthreads();
  }
}

__device__ __forceinline__ void ptir_parallel_argmax(
    const m1_u8* input,
    m1_u8* output,
    m1_u8* temporary,
    const M1ValueDesc input_desc) {
  int* result = reinterpret_cast<int*>(output);
  const m1_u32 lane = threadIdx.x & 31u;
  const m1_u32 warp = threadIdx.x >> 5u;
  const m1_u32 warps = blockDim.x >> 5u;
  const unsigned mask = 0xffffffffu;
  const m1_u32 first_chunks = (input_desc.last + 31u) / 32u;
  if (input_desc.dtype == 0u) {
    M1ArgmaxCandidate* work_a =
        reinterpret_cast<M1ArgmaxCandidate*>(temporary);
    M1ArgmaxCandidate* work_b = work_a + first_chunks;
    const float* values = reinterpret_cast<const float*>(input);
    for (m1_u32 row = 0; row < input_desc.rows; ++row) {
      const m1_u32 base = row * input_desc.last;
      if (input_desc.last == 0u) {
        if (threadIdx.x == 0u) result[row] = 0;
        __syncthreads();
        continue;
      }
      for (m1_u32 chunk = warp;
           chunk < first_chunks;
           chunk += warps) {
        const m1_u32 index = chunk * 32u + lane;
        M1ArgmaxCandidate candidate =
            index < input_desc.last
                ? M1ArgmaxCandidate{
                      values[base + index],
                      index,
                      m1_isnan(values[base + index]) ? 0u : 1u,
                      0u}
                : M1ArgmaxCandidate{m1_neg_inf(), 0u, 0u, 0u};
        for (m1_u32 offset = 16u; offset > 0u; offset >>= 1u) {
          M1ArgmaxCandidate other{
              __shfl_down_sync(mask, candidate.value, offset),
              __shfl_down_sync(mask, candidate.index, offset),
              __shfl_down_sync(mask, candidate.have, offset),
              0u};
          if (lane < offset)
            candidate = m1_argmax_combine(candidate, other);
        }
        if (lane == 0u) work_a[chunk] = candidate;
      }
      __syncthreads();
      m1_u32 count = first_chunks;
      M1ArgmaxCandidate* reduction_input = work_a;
      M1ArgmaxCandidate* reduction_output = work_b;
      while (count > 1u) {
        const m1_u32 chunks = (count + 31u) / 32u;
        for (m1_u32 chunk = warp;
             chunk < chunks;
             chunk += warps) {
          const m1_u32 index = chunk * 32u + lane;
          M1ArgmaxCandidate candidate =
              index < count
                  ? reduction_input[index]
                  : M1ArgmaxCandidate{m1_neg_inf(), 0u, 0u, 0u};
          for (m1_u32 offset = 16u; offset > 0u; offset >>= 1u) {
            M1ArgmaxCandidate other{
                __shfl_down_sync(mask, candidate.value, offset),
                __shfl_down_sync(mask, candidate.index, offset),
                __shfl_down_sync(mask, candidate.have, offset),
                0u};
            if (lane < offset)
              candidate = m1_argmax_combine(candidate, other);
          }
          if (lane == 0u) reduction_output[chunk] = candidate;
        }
        __syncthreads();
        M1ArgmaxCandidate* swap = reduction_input;
        reduction_input = reduction_output;
        reduction_output = swap;
        count = chunks;
      }
      if (threadIdx.x == 0u)
        result[row] = (int)reduction_input[0].index;
      __syncthreads();
    }
    return;
  }
  M1IntArgmaxCandidate* work_a =
      reinterpret_cast<M1IntArgmaxCandidate*>(temporary);
  M1IntArgmaxCandidate* work_b = work_a + first_chunks;
  for (m1_u32 row = 0; row < input_desc.rows; ++row) {
    const m1_u32 base = row * input_desc.last;
    if (input_desc.last == 0u) {
      if (threadIdx.x == 0u) result[row] = 0;
      __syncthreads();
      continue;
    }
    for (m1_u32 chunk = warp;
         chunk < first_chunks;
         chunk += warps) {
      const m1_u32 index = chunk * 32u + lane;
      M1IntArgmaxCandidate candidate =
          index < input_desc.last
              ? M1IntArgmaxCandidate{
                    m1_load_index(
                        input, base + index, input_desc.dtype),
                    index,
                    1u}
              : M1IntArgmaxCandidate{0ll, 0u, 0u};
      for (m1_u32 offset = 16u; offset > 0u; offset >>= 1u) {
        M1IntArgmaxCandidate other{
            __shfl_down_sync(mask, candidate.value, offset),
            __shfl_down_sync(mask, candidate.index, offset),
            __shfl_down_sync(mask, candidate.have, offset)};
        if (lane < offset)
          candidate = m1_int_argmax_combine(candidate, other);
      }
      if (lane == 0u) work_a[chunk] = candidate;
    }
    __syncthreads();
    m1_u32 count = first_chunks;
    M1IntArgmaxCandidate* reduction_input = work_a;
    M1IntArgmaxCandidate* reduction_output = work_b;
    while (count > 1u) {
      const m1_u32 chunks = (count + 31u) / 32u;
      for (m1_u32 chunk = warp;
           chunk < chunks;
           chunk += warps) {
        const m1_u32 index = chunk * 32u + lane;
        M1IntArgmaxCandidate candidate =
            index < count
                ? reduction_input[index]
                : M1IntArgmaxCandidate{0ll, 0u, 0u};
        for (m1_u32 offset = 16u; offset > 0u; offset >>= 1u) {
          M1IntArgmaxCandidate other{
              __shfl_down_sync(mask, candidate.value, offset),
              __shfl_down_sync(mask, candidate.index, offset),
              __shfl_down_sync(mask, candidate.have, offset)};
          if (lane < offset)
            candidate = m1_int_argmax_combine(candidate, other);
        }
        if (lane == 0u) reduction_output[chunk] = candidate;
      }
      __syncthreads();
      M1IntArgmaxCandidate* swap = reduction_input;
      reduction_input = reduction_output;
      reduction_output = swap;
      count = chunks;
    }
    if (threadIdx.x == 0u)
      result[row] = (int)reduction_input[0].index;
    __syncthreads();
  }
}

__device__ __forceinline__ M1ArgmaxCandidate m1_argmax_warp_reduce(
    M1ArgmaxCandidate candidate, m1_u32 lane) {
  const unsigned mask = 0xffffffffu;
  for (m1_u32 offset = 16u; offset > 0u; offset >>= 1u) {
    M1ArgmaxCandidate other{
        __shfl_down_sync(mask, candidate.value, offset),
        __shfl_down_sync(mask, candidate.index, offset),
        __shfl_down_sync(mask, candidate.have, offset),
        0u};
    if (lane < offset)
      candidate = m1_argmax_combine(candidate, other);
  }
  return candidate;
}

__device__ __forceinline__ M1IntArgmaxCandidate m1_int_argmax_warp_reduce(
    M1IntArgmaxCandidate candidate, m1_u32 lane) {
  const unsigned mask = 0xffffffffu;
  for (m1_u32 offset = 16u; offset > 0u; offset >>= 1u) {
    M1IntArgmaxCandidate other{
        __shfl_down_sync(mask, candidate.value, offset),
        __shfl_down_sync(mask, candidate.index, offset),
        __shfl_down_sync(mask, candidate.have, offset)};
    if (lane < offset)
      candidate = m1_int_argmax_combine(candidate, other);
  }
  return candidate;
}

__device__ __forceinline__ void ptir_fast_argmax(
    const m1_u8* input,
    m1_u8* output,
    const M1ValueDesc input_desc) {
  __shared__ M1ArgmaxCandidate float_candidates[32];
  __shared__ M1IntArgmaxCandidate int_candidates[32];
  int* result = reinterpret_cast<int*>(output);
  const m1_u32 lane = threadIdx.x & 31u;
  const m1_u32 warp = threadIdx.x >> 5u;
  const m1_u32 warps = blockDim.x >> 5u;
  for (m1_u32 row = 0; row < input_desc.rows; ++row) {
    const m1_u32 base = row * input_desc.last;
    if (input_desc.dtype == 0u) {
      M1ArgmaxCandidate candidate{
          m1_neg_inf(), 0u, 0u, 0u};
      for (m1_u32 index = threadIdx.x;
           index < input_desc.last;
           index += blockDim.x) {
        const float value =
            reinterpret_cast<const float*>(input)[base + index];
        const M1ArgmaxCandidate next{
            value, index, m1_isnan(value) ? 0u : 1u, 0u};
        candidate = m1_argmax_combine(candidate, next);
      }
      candidate = m1_argmax_warp_reduce(candidate, lane);
      if (lane == 0u) float_candidates[warp] = candidate;
      __syncthreads();
      if (warp == 0u) {
        candidate =
            lane < warps
                ? float_candidates[lane]
                : M1ArgmaxCandidate{m1_neg_inf(), 0u, 0u, 0u};
        candidate = m1_argmax_warp_reduce(candidate, lane);
        if (lane == 0u) result[row] = (int)candidate.index;
      }
      __syncthreads();
      continue;
    }
    M1IntArgmaxCandidate candidate{0ll, 0u, 0u};
    for (m1_u32 index = threadIdx.x;
         index < input_desc.last;
         index += blockDim.x) {
      const M1IntArgmaxCandidate next{
          m1_load_index(input, base + index, input_desc.dtype),
          index,
          1u};
      candidate = m1_int_argmax_combine(candidate, next);
    }
    candidate = m1_int_argmax_warp_reduce(candidate, lane);
    if (lane == 0u) int_candidates[warp] = candidate;
    __syncthreads();
    if (warp == 0u) {
      candidate =
          lane < warps
              ? int_candidates[lane]
              : M1IntArgmaxCandidate{0ll, 0u, 0u};
      candidate = m1_int_argmax_warp_reduce(candidate, lane);
      if (lane == 0u) result[row] = (int)candidate.index;
    }
    __syncthreads();
  }
}

__device__ __forceinline__ void ptir_fast_argmax_intrinsic(
    const m1_u8* input,
    m1_u8* output,
    const M1ValueDesc input_desc,
    m1_u32 mode,
    m1_u32 stride,
    m1_u32 row_offset) {
  __shared__ M1ArgmaxCandidate candidates[32];
  int* result = reinterpret_cast<int*>(output);
  const m1_u32 lane = threadIdx.x & 31u;
  const m1_u32 warp = threadIdx.x >> 5u;
  const m1_u32 warps = blockDim.x >> 5u;
  for (m1_u32 row = 0; row < input_desc.rows; ++row) {
    M1ArgmaxCandidate candidate{
        m1_neg_inf(), 0u, 0u, 0u};
    for (m1_u32 index = threadIdx.x;
         index < input_desc.last;
         index += blockDim.x) {
      const float value = m1_intrinsic_row_load(
          input,
          (m1_u64)row_offset + row,
          index,
          stride,
          mode);
      const M1ArgmaxCandidate next{
          value, index, m1_isnan(value) ? 0u : 1u, 0u};
      candidate = m1_argmax_combine(candidate, next);
    }
    candidate = m1_argmax_warp_reduce(candidate, lane);
    if (lane == 0u) candidates[warp] = candidate;
    __syncthreads();
    if (warp == 0u) {
      candidate =
          lane < warps
              ? candidates[lane]
              : M1ArgmaxCandidate{m1_neg_inf(), 0u, 0u, 0u};
      candidate = m1_argmax_warp_reduce(candidate, lane);
      if (lane == 0u) result[row] = (int)candidate.index;
    }
    __syncthreads();
  }
}

extern "C" __global__ void )PTIR_CUDA"
           << entry_name << R"PTIR_CUDA((
    const PtirLaneTableHeader* header,
    const PtirLaneRecord* lanes,
    const PtirLaneChannelSlot* channels,
    const M1ValueDesc* all_descriptors,
    const M1OpParams* params,
    const m1_u32* offsets,
    m1_u8* all_scratch,
    m1_u32 value_count,
    m1_u32 scratch_stride,
    m1_u32 temporary_offset,
    m1_u8* pending_flags,
    const m1_u64* intrinsic_bases,
    const m1_u32* intrinsic_modes,
    const m1_u32* intrinsic_widths,
    const m1_u32* intrinsic_strides,
    const m1_u32* intrinsic_offsets) {
  const m1_u32 dispatch_lane = blockIdx.x;
  if (header == nullptr || dispatch_lane >= header->lane_count) return;
  const PtirLaneRecord lane = lanes[dispatch_lane];
  m1_u32* commit = reinterpret_cast<m1_u32*>(lane.commit_slot);
  if (commit == nullptr || *commit == 0u) return;
  if (header->abi_version != )PTIR_CUDA"
           << PTIR_LANE_TABLE_ABI_VERSION << R"PTIR_CUDA(u) {
    if (threadIdx.x == 0u) *commit = 0u;
    return;
  }
  const m1_u8* lane_row_valid =
      reinterpret_cast<const m1_u8*>(lane.row_valid);
  const bool lane_active =
      lane_row_valid == nullptr ||
      lane_row_valid[lane.row_valid_offset] != 0u;
  __shared__ M1Status status;
  if (threadIdx.x == 0u)
    status = M1Status{1u, 0u, 0u, 0u};
  __syncthreads();
  const M1ValueDesc* descriptors =
      all_descriptors + (m1_u64)dispatch_lane * value_count;
  m1_u8* scratch =
      all_scratch + (m1_u64)dispatch_lane * scratch_stride;
  m1_u8* temporary = scratch + temporary_offset;
)PTIR_CUDA";

    auto value_pointer = [&](std::uint32_t value) {
        value = resolve_alias(value);
        return std::string("scratch + offsets[") +
            std::to_string(value) + "]";
    };
    auto parallel_elementwise = [](std::uint8_t tag) {
        return (tag >= PTIR_OP_EXP && tag <= PTIR_OP_CAST) ||
            (tag >= PTIR_OP_ADD && tag <= PTIR_OP_SELECT) ||
            tag == PTIR_OP_IOTA ||
            tag == PTIR_OP_MASK_APPLY_PACKED ||
            tag == PTIR_OP_CAUSAL_MASK ||
            tag == PTIR_OP_SLIDING_WINDOW_MASK ||
            tag == PTIR_OP_SINK_WINDOW_MASK ||
            tag == PTIR_OP_RNG ||
            tag == PTIR_OP_RNG_KEYED;
    };
    for (const std::uint32_t node : region.nodes) {
        const auto& op = stage.ops[node].op;
        const std::uint32_t base = bases[node];
        if (direct_argmax_skipped[node] != 0 &&
            op.tag != PTIR_OP_RESHAPE) {
            continue;
        }
        if (op.tag == PTIR_OP_RESHAPE &&
            std::find(
                region.outputs.begin(),
                region.outputs.end(),
                base) == region.outputs.end()) {
            aliases[base] = resolve_alias(op.args[0]);
            continue;
        }
        std::string a0 = "scratch";
        std::string a1 = "scratch";
        std::string a2 = "scratch";
        std::string o0 = "scratch";
        std::string o1 = "scratch";
        if (!op.args.empty()) a0 = value_pointer(op.args[0]);
        if (op.args.size() > 1) a1 = value_pointer(op.args[1]);
        if (op.args.size() > 2) a2 = value_pointer(op.args[2]);
        if (op.tag == PTIR_OP_PIVOT_THRESHOLD) {
            a1 = value_pointer(op.pred_payload);
        }
        if (op.results > 0) o0 = value_pointer(base);
        if (op.results > 1) o1 = value_pointer(base + 1);

        source << "  {\n"
               << "    M1OpParams p = params[" << node << "u];\n"
               << "    p.rng_seed = 0u;\n";
        if (op.tag == PTIR_OP_CHAN_TAKE ||
            op.tag == PTIR_OP_CHAN_READ ||
            op.tag == PTIR_OP_CHAN_PUT) {
            const auto local = static_cast<std::uint32_t>(op.chan);
            source
                << "    const m1_u32 channel_index = "
                   "lane.channel_slot_offset + "
                << local << "u;\n"
                << "    const PtirLaneChannelSlot channel = "
                   "channels[channel_index];\n";
            if (op.tag == PTIR_OP_CHAN_TAKE ||
                op.tag == PTIR_OP_CHAN_READ) {
                a0 =
                    "reinterpret_cast<const m1_u8*>("
                    "pending_flags[channel_index] != 0u ? "
                    "channel.pending_cell : channel.committed_cell)";
            } else {
                o0 =
                    "reinterpret_cast<m1_u8*>(channel.pending_cell)";
            }
        } else if (op.tag == PTIR_OP_INTRINSIC_VAL) {
            source
                << "    const m1_u32 intrinsic_index = "
                   "dispatch_lane * 7u + p.intr;\n"
                << "    p.intrinsic_dtype = "
                   "intrinsic_modes[intrinsic_index];\n"
                << "    p.imm = intrinsic_widths[intrinsic_index];\n"
                << "    p.intrinsic_row_stride = "
                   "intrinsic_strides[intrinsic_index];\n"
                << "    p.intrinsic_row_offset = "
                   "intrinsic_offsets[intrinsic_index];\n";
            a0 =
                "reinterpret_cast<const m1_u8*>("
                "intrinsic_bases[intrinsic_index])";
        }
        if (op.tag == PTIR_OP_CONST) {
            source
                << "    const M1ValueDesc out = descriptors[p.o0];\n"
                << "    for (m1_u32 i = threadIdx.x; i < out.len; "
                   "i += blockDim.x) {\n"
                << "      if (p.lit_dtype == 0u) "
                   "m1_store_f(" << o0 << ", i, m1_bits_f32(p.lit_bits));\n"
                << "      else if (p.lit_dtype == 1u) "
                   "m1_store_i(" << o0 << ", i, m1_bits_i32(p.lit_bits));\n"
                << "      else if (p.lit_dtype == 2u) "
                   "m1_store_u(" << o0 << ", i, p.lit_bits);\n"
                << "      else m1_store_b(" << o0
                << ", i, p.lit_bits != 0u);\n"
                << "    }\n";
        } else if (
            op.tag == PTIR_OP_CHAN_TAKE ||
            op.tag == PTIR_OP_CHAN_READ) {
            source
                << "    const M1ValueDesc out = descriptors[p.o0];\n"
                << "    ptir_parallel_copy(" << a0 << ", " << o0
                << ", out.len, out.dtype);\n";
        } else if (op.tag == PTIR_OP_CHAN_PUT) {
            source
                << "    const M1ValueDesc input = descriptors[p.a0];\n"
                << "    const m1_u32 logical_bytes = "
                   "input.dtype == 3u ? input.len : input.len * 4u;\n"
                << "    if (logical_bytes > p.sink_bytes) {\n"
                << "      if (threadIdx.x == 0u) m1_fault(&status, "
                << unsigned(op.tag) << "u);\n"
                << "    } else {\n"
                << "      const bool sample_output = "
                   "(lane.sample_output_channel_mask & (1ull << "
                << static_cast<std::uint32_t>(op.chan) << "u)) != 0ull;\n"
                << "      const m1_u8* committed = "
                   "reinterpret_cast<const m1_u8*>("
                   "channel.committed_cell);\n"
                << "      const m1_u32 element_bytes = "
                   "input.dtype == 3u ? 1u : 4u;\n"
                << "      m1_u32 elements_per_validity_row = 0u;\n"
                << "      if (lane.token_count != 0u) {\n"
                << "        if (input.rows == lane.token_count) "
                   "elements_per_validity_row = input.last;\n"
                << "        else if (input.len == lane.token_count) "
                   "elements_per_validity_row = 1u;\n"
                << "      }\n"
                << "      for (m1_u32 byte = threadIdx.x; "
                   "byte < p.sink_bytes; byte += blockDim.x) {\n"
                << "        if (byte >= logical_bytes) {\n"
                << "          " << o0 << "[byte] = 0u;\n"
                << "          continue;\n"
                << "        }\n"
                << "        bool row_active = lane_active;\n"
                << "        if (lane_row_valid != nullptr && "
                   "elements_per_validity_row != 0u) {\n"
                << "          const m1_u32 element = byte / element_bytes;\n"
                << "          const m1_u32 row = "
                   "element / elements_per_validity_row;\n"
                << "          if (row < lane.token_count) "
                   "row_active = lane_row_valid["
                   "lane.row_valid_offset + row] != 0u;\n"
                << "        }\n"
                << "        if (row_active) " << o0 << "[byte] = (" << a0
                << ")[byte];\n"
                << "        else if (sample_output) " << o0
                << "[byte] = 0xffu;\n"
                << "        else if (committed != nullptr) " << o0
                << "[byte] = committed[byte];\n"
                << "        else if (threadIdx.x == 0u) "
                   "m1_fault(&status, "
                << unsigned(op.tag) << "u);\n"
                << "      }\n"
                << "    }\n";
        } else if (op.tag == PTIR_OP_INTRINSIC_VAL) {
            if (op.intr == PTIR_INTR_LAYER ||
                op.intr == PTIR_INTR_MTP_DRAFTS) {
                source << "    if (threadIdx.x == 0u) "
                       << "ptir_m1_execute(" << unsigned(op.tag)
                       << "u, &status, descriptors, &p, "
                       << a0 << ", " << a1 << ", " << a2 << ", "
                       << o0 << ", " << o1 << ", temporary);\n";
            } else {
                source
                    << "    ptir_parallel_intrinsic(" << a0 << ", "
                    << o0 << ", descriptors[p.o0], p);\n";
            }
        } else if (op.tag == PTIR_OP_BROADCAST) {
            source
                << "    ptir_parallel_broadcast(" << a0 << ", " << o0
                << ", descriptors[p.a0], descriptors[p.o0]);\n";
        } else if (op.tag == PTIR_OP_RESHAPE) {
            source
                << "    const M1ValueDesc out = descriptors[p.o0];\n"
                << "    ptir_parallel_copy(" << a0 << ", " << o0
                << ", out.len, out.dtype);\n";
        } else if (op.tag == PTIR_OP_TRANSPOSE) {
            source
                << "    ptir_parallel_transpose(&status, " << a0
                << ", " << o0
                << ", descriptors[p.a0], descriptors[p.o0]);\n";
        } else if (
            op.tag == PTIR_OP_REDUCE_SUM ||
            op.tag == PTIR_OP_REDUCE_MAX ||
            op.tag == PTIR_OP_REDUCE_MIN) {
            if (stage.value_types[op.args[0]].dtype == PTIR_DT_F32) {
                source
                    << "    ptir_parallel_reduce_f32("
                    << unsigned(op.tag) << "u, " << a0 << ", "
                    << o0
                    << ", temporary, descriptors[p.a0]);\n";
            } else {
                source << "    if (threadIdx.x == 0u) "
                       << "ptir_m1_execute(" << unsigned(op.tag)
                       << "u, &status, descriptors, &p, "
                       << a0 << ", " << a1 << ", " << a2 << ", "
                       << o0 << ", " << o1 << ", temporary);\n";
            }
        } else if (op.tag == PTIR_OP_REDUCE_ARGMAX) {
            if (direct_argmax_intrinsic[node] !=
                std::numeric_limits<std::uint16_t>::max()) {
                source
                    << "    const m1_u32 direct_intrinsic_index = "
                       "dispatch_lane * 7u + "
                    << direct_argmax_intrinsic[node] << "u;\n"
                    << "    ptir_fast_argmax_intrinsic(\n"
                    << "        reinterpret_cast<const m1_u8*>("
                       "intrinsic_bases[direct_intrinsic_index]),\n"
                    << "        " << o0 << ",\n"
                    << "        descriptors[p.a0],\n"
                    << "        intrinsic_modes[direct_intrinsic_index],\n"
                    << "        intrinsic_strides[direct_intrinsic_index],\n"
                    << "        intrinsic_offsets[direct_intrinsic_index]);\n";
            } else {
                source
                    << "    ptir_fast_argmax(" << a0 << ", " << o0
                    << ", descriptors[p.a0]);\n";
            }
        } else if (
            op.tag == PTIR_OP_GATHER ||
            op.tag == PTIR_OP_GATHER_ROW) {
            source
                << "    ptir_parallel_gather(" << unsigned(op.tag)
                << "u, " << a0 << ", " << a1 << ", " << o0
                << ", descriptors[p.a0], descriptors[p.a1], "
                   "descriptors[p.o0]);\n";
        } else if (
            op.tag == PTIR_OP_SCATTER_ADD ||
            op.tag == PTIR_OP_SCATTER_SET) {
            source
                << "    ptir_parallel_copy(" << a0 << ", " << o0
                << ", descriptors[p.a0].len, descriptors[p.a0].dtype);\n"
                << "    __syncthreads();\n"
                << "    if (threadIdx.x == 0u) ptir_scatter_updates("
                << unsigned(op.tag) << "u, " << a1 << ", " << a2
                << ", " << o0
                << ", descriptors[p.a0], descriptors[p.a1], "
                   "descriptors[p.a2]);\n";
        } else if (
            op.tag == PTIR_OP_PIVOT_THRESHOLD &&
            op.pred_tag != 1) {
            source
                << "    ptir_parallel_pivot(" << a0 << ", " << a1
                << ", " << o0
                << ", descriptors[p.a0], descriptors[p.a1], p);\n";
        } else if (parallel_elementwise(op.tag)) {
            source
                << "    ptir_parallel_elementwise(" << unsigned(op.tag)
                << "u, &status, descriptors, p, "
                << a0 << ", " << a1 << ", " << a2 << ", "
                << o0 << ");\n";
        } else {
            source << "    if (threadIdx.x == 0u) "
                   << "ptir_m1_execute(" << unsigned(op.tag)
                   << "u, &status, descriptors, &p, "
                   << a0 << ", " << a1 << ", " << a2 << ", "
                   << o0 << ", " << o1 << ", temporary);\n";
        }
        source
            << "    __syncthreads();\n"
            << "    if (status.state != 1u) {\n"
            << "      if (threadIdx.x == 0u) *commit = 0u;\n"
            << "      return;\n"
            << "    }\n";
        if (op.tag == PTIR_OP_CHAN_PUT) {
            source
                << "    if (threadIdx.x == 0u) "
                   "pending_flags[channel_index] = 1u;\n"
                << "    __syncthreads();\n";
        }
        source << "  }\n";
    }
    source << "}\n";
    result.source = source.str();
    result.ok = true;
    return result;
}

}  // namespace pie_cuda_driver::pipeline::generated
