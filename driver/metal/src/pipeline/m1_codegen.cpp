#include "pipeline/m1_codegen.hpp"

#include <algorithm>
#include <functional>
#include <limits>
#include <sstream>

#include "pie_native/ptir/op_table.hpp"

namespace pie::metal::pipeline {
namespace {

using pie_native::ptir::OpCode;
using pie_native::ptir::op_info;

bool supported_tag(std::uint8_t tag) {
    const OpCode code = static_cast<OpCode>(tag);
    return pie_native::ptir::op_is_known(code);
}

std::string common_effect_preamble() {
    return R"MSL(
#include <metal_stdlib>
using namespace metal;
struct M1Status { uint state; uint fault; uint reserved0; uint reserved1; };
struct M1LaneHeader { uint abi_version; uint lane_count; uint channel_count; uint flags; };
struct M1LaneRecord {
  ulong logits_base;
  uint logits_row_offset;
  uint logits_row_count;
  uint kv_len;
  uint page_count;
  uint row_count;
  uint token_count;
  uint sampled_rows;
  uint query_len;
  uint key_len;
  uint channel_slot_offset;
  ulong rng_state;
  ulong commit_slot;
  ulong active_row_mask;
};
struct M1LaneChannelSlot {
  ulong committed_cell;
  ulong pending_cell;
  ulong expected_head;
  ulong expected_tail;
};
)MSL";
}

void emit_word_arguments(std::ostringstream& source, std::size_t count) {
    for (std::size_t channel = 0; channel < count; ++channel) {
        source << ", device ulong* words_" << channel
               << " [[buffer(" << channel + 2 << ")]]";
    }
}

std::string grouped_preamble() {
    return R"MSL(
struct M3LaneHeader { uint abi_version; uint lane_count; uint channel_count; uint flags; };
struct M3LaneRecord {
  ulong logits_base;
  uint logits_row_offset;
  uint logits_row_count;
  uint kv_len;
  uint page_count;
  uint row_count;
  uint token_count;
  uint sampled_rows;
  uint query_len;
  uint key_len;
  uint channel_slot_offset;
  ulong rng_state;
  ulong commit_slot;
  ulong active_row_mask;
};
struct M3LaneChannelSlot {
  ulong committed_cell;
  ulong pending_cell;
  ulong expected_head;
  ulong expected_tail;
};
struct M3ChannelMeta {
  ulong words;
  uint capacity;
  uint flags;
};
struct M3GroupLayout {
  uint lane_count;
  uint value_count;
  uint scratch_stride;
  uint temporary_offset;
  uint vocab;
  uint reserved0;
  uint reserved1;
  uint reserved2;
};
struct M3RowMeta {
  uint offset;
  uint count;
  uint mtp_offset;
  uint reserved;
};
)MSL";
}

bool nucleus_library_region_valid(
    const pie_native::ptir::plan::StagePlan& stage,
    const pie_native::ptir::plan::Region& region) {
    if (!region.library ||
        region.library_op != PTIR_LIBRARY_NUCLEUS_SAMPLE ||
        region.schedule != PTIR_SCHEDULE_LIBRARY ||
        region.nodes.size() != 13 || region.inputs.size() != 3 ||
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
    const auto& logits_type =
        stage.value_types[region.inputs[0]];
    const auto& top_p_type =
        stage.value_types[region.inputs[1]];
    const auto& state_type =
        stage.value_types[region.inputs[2]];
    const auto& output_type =
        stage.value_types[region.outputs[0]];
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
    const std::vector<pie_native::ptir::plan::Dimension> row_dims(
        logits_type.dims.begin(), logits_type.dims.end() - 1);
    return top_p_type.dtype == PTIR_DT_F32 &&
        (top_p_type.dims.empty() ||
         top_p_type.dims.size() == row_dims.size()) &&
        state_type.dtype == PTIR_DT_U32 &&
        state_type.dims.size() == 1 &&
        !state_type.dims[0].symbolic &&
        state_type.dims[0].value == 2 &&
        output_type.dtype == PTIR_DT_I32 &&
        same_dims(output_type.dims, row_dims);
}

bool library_region_valid(
    const pie_native::ptir::plan::StagePlan& stage,
    const pie_native::ptir::plan::Region& region) {
    if (!region.library) return true;
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

std::size_t used_channel_slots(
    const pie_native::ptir::plan::StagePlan& stage) {
    std::size_t count = 0;
    for (const auto& normalized : stage.ops) {
        if (normalized.op.chan >= 0) {
            count = std::max(
                count,
                static_cast<std::size_t>(
                    normalized.op.chan) +
                    1);
        }
    }
    return count;
}

}  // namespace

bool validate_singleton_plan(
    const pie_native::ptir::plan::StagePlan& plan,
    std::vector<M1OpMeta>& operations,
    std::string& error) {
    operations.clear();
    if (plan.signature_hash == 0 ||
        pie_native::ptir::container::fnv1a64(
            plan.signature.data(), plan.signature.size()) !=
            plan.signature_hash ||
        plan.singleton.kind != 0) {
        error = "invalid singleton plan identity";
        return false;
    }
    for (const auto& type : plan.value_types) {
        if (type.dims.size() > 4 ||
            type.dtype > PTIR_DT_BOOL) {
            error = "invalid normalized value type";
            return false;
        }
        std::uint64_t product = 1;
        for (const auto& dimension : type.dims) {
            if (dimension.symbolic) {
                if (dimension.value < PTIR_EXTENT_KV_LEN ||
                    dimension.value > PTIR_EXTENT_KEY_LEN) {
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
        for (const auto& region : partition->regions) {
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
            if (!library_region_valid(plan, region)) {
                error = "library region ABI is invalid";
                return false;
            }
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
        if (!supported_tag(op.tag)) {
            const auto info = op_info(static_cast<OpCode>(op.tag));
            error = "unsupported singleton op " + std::string(info.name);
            return false;
        }
        const auto info =
            op_info(static_cast<OpCode>(op.tag));
        if (op.tag == PTIR_OP_KERNEL_CALL) {
            if (op.name_idx >= plan.names.size() ||
                plan.names[op.name_idx] != "metal.identity" ||
                op.args.size() != 1 ||
                result_base >= plan.value_types.size() ||
                plan.value_types[op.args[0]].dtype !=
                    plan.value_types[result_base].dtype ||
                plan.value_types[op.args[0]].dims.size() !=
                    plan.value_types[result_base].dims.size() ||
                !std::equal(
                    plan.value_types[op.args[0]].dims.begin(),
                    plan.value_types[op.args[0]].dims.end(),
                    plan.value_types[result_base].dims.begin(),
                    [](const auto& left, const auto& right) {
                        return left.symbolic == right.symbolic &&
                               left.value == right.value;
                    })) {
                error =
                    "unsupported Metal semantic kernel boundary";
                return false;
            }
        } else if (op.tag == PTIR_OP_SINK_CALL) {
            if (op.name_idx >= plan.names.size() ||
                plan.names[op.name_idx] != "metal.discard") {
                error =
                    "unsupported Metal semantic sink boundary";
                return false;
            }
        }
        const std::size_t expected_arity =
            op.tag == PTIR_OP_PIVOT_THRESHOLD
                ? 1
                : info.arity;
        if (expected_arity != 0xfe &&
            op.args.size() != expected_arity) {
            error = "normalized op arity mismatch";
            return false;
        }
        if (op.results != info.results ||
            result_base >
                std::numeric_limits<std::uint32_t>::max() -
                    op.results ||
            result_base + op.results >
                plan.value_types.size()) {
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
            (op.pred_tag > 2 ||
             op.pred_payload >= result_base)) {
            error = "pivot predicate payload is out of range";
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
            (!channel_op && op.chan >= 0)) {
            error = "normalized channel slot is invalid";
            return false;
        }
        operations.push_back(
            {static_cast<std::uint32_t>(node), result_base, op});
        result_base += op.results;
    }
    if (plan.singleton.whole_stage_fallback) {
        error =
            "singleton plan requests whole-stage fallback without an "
            "identifiable unsupported op";
        return false;
    }
    if (result_base != plan.value_types.size()) {
        error = "normalized value layout does not match op results";
        return false;
    }
    return true;
}

std::string emit_singleton_region_msl(
    const std::string& runtime_template,
    const std::string& function_name,
    std::uint8_t op_tag) {
    std::ostringstream source;
    source << runtime_template << "\n"
           << "kernel void " << function_name << "(\n"
           << "    device M1Status* status [[buffer(0)]],\n"
           << "    const device M1ValueDesc* descriptors [[buffer(1)]],\n"
           << "    const device uchar* a0 [[buffer(2)]],\n"
           << "    const device uchar* a1 [[buffer(3)]],\n"
           << "    const device uchar* a2 [[buffer(4)]],\n"
           << "    device uchar* o0 [[buffer(5)]],\n"
           << "    device uchar* o1 [[buffer(6)]],\n"
           << "    device uchar* temporary [[buffer(7)]],\n"
           << "    const device M1OpParams* params [[buffer(8)]],\n"
           << "    uint gid [[thread_position_in_grid]]) {\n"
           << "  if (gid == 0) ptir_m1_execute(" << unsigned(op_tag)
           << "u, status, descriptors, params, "
              "a0, a1, a2, o0, o1, temporary);\n"
           << "}\n";
    return source.str();
}

std::string emit_readiness_msl(
    const std::string& function_name,
    const std::vector<M1ChannelEffect>& channels) {
    std::ostringstream source;
    source << common_effect_preamble()
           << "kernel void " << function_name
           << "(device M1Status* status [[buffer(0)]], "
              "const device uchar* lane_bytes [[buffer(1)]]";
    emit_word_arguments(source, channels.size());
    source << ", uint gid [[thread_position_in_grid]]) {\n"
           << "  if (gid != 0) return;\n"
           << "  const device M1LaneHeader* header = "
              "reinterpret_cast<const device M1LaneHeader*>(lane_bytes);\n"
           << "  const device M1LaneChannelSlot* slots = "
              "reinterpret_cast<const device M1LaneChannelSlot*>(lane_bytes + "
              "sizeof(M1LaneHeader) + sizeof(M1LaneRecord));\n"
           << "  if (header->abi_version != "
           << PTIR_LANE_TABLE_ABI_VERSION
           << " || header->lane_count != 1 || "
              "header->channel_count != " << channels.size() << ") { "
              "status->state = 3; status->fault = 0x100; "
              "status->reserved0 = header->abi_version; "
              "status->reserved1 = (header->lane_count << 16) | "
              "header->channel_count; return; }\n"
           << "  status->state = 0; status->fault = 0;\n";
    for (std::size_t channel = 0; channel < channels.size(); ++channel) {
        const M1ChannelEffect effect = channels[channel];
        source << "  {\n"
               << "    const ulong head = words_" << channel << "[0];\n"
               << "    const ulong tail = words_" << channel << "[1];\n"
               << "    if (words_" << channel << "[2] != 0ul || words_"
               << channel << "[3] != 0ul || tail < head) { status->state = 3; "
               << "status->fault = " << (0x200 + channel) << "; return; }\n"
               << "    const ulong expected_head = slots[" << channel
               << "].expected_head;\n"
               << "    const ulong expected_tail = slots[" << channel
               << "].expected_tail;\n"
               << "    if (expected_head != ~0ul && head != expected_head) "
                  "{ status->state = 2; status->fault = "
               << (0x300 + channel) << "; return; }\n";
        if (effect.requires_full) {
            source << "    if (tail <= head) { status->state = 2; status->fault = "
                   << (0x400 + channel) << "; return; }\n";
        }
        if (effect.requires_empty) {
            source << "    if (tail - head >= " << effect.capacity
                   << "ul) { status->state = 2; status->fault = "
                   << (0x480 + channel) << "; return; }\n";
        }
        if (effect.put) {
            const std::uint32_t credit =
                effect.take && effect.requires_full ? 1u : 0u;
            source << "    if (expected_tail == ~0ul || tail != expected_tail || "
                   << "tail - head >= " << effect.capacity
                   << "ul + " << credit << "ul) "
                      "{ status->state = 2; status->fault = "
                   << (0x500 + channel) << "; return; }\n";
        }
        source << "  }\n";
    }
    source << "  status->state = 1;\n}\n";
    return source.str();
}

std::string emit_commit_msl(
    const std::string& function_name,
    const std::vector<M1ChannelEffect>& channels) {
    std::ostringstream source;
    source << common_effect_preamble()
           << "kernel void " << function_name
           << "(device M1Status* status [[buffer(0)]], "
              "const device uchar* lane_bytes [[buffer(1)]]";
    emit_word_arguments(source, channels.size());
    source << ", uint gid [[thread_position_in_grid]]) {\n"
           << "  if (gid != 0 || status->state != 1) return;\n"
           << "  const device M1LaneChannelSlot* slots = "
              "reinterpret_cast<const device M1LaneChannelSlot*>(lane_bytes + "
              "sizeof(M1LaneHeader) + sizeof(M1LaneRecord));\n";
    for (std::size_t channel = 0; channel < channels.size(); ++channel) {
        source << "  {\n"
               << "    const ulong old_head = words_" << channel << "[0];\n"
               << "    const ulong old_tail = words_" << channel << "[1];\n";
        if (channels[channel].take) {
            source << "    if (old_tail > old_head) words_" << channel
                   << "[0] = old_head + 1ul;\n";
        }
        if (channels[channel].put) {
            source << "    words_" << channel << "[1] = old_tail + 1ul;\n";
        }
        source << "  }\n";
    }
    source << "  status->state = 4;\n}\n";
    return source.str();
}

bool emit_fused_region_msl(
    const std::string& runtime_template,
    const std::string& function_name,
    const pie_native::ptir::plan::StagePlan& stage,
    const pie_native::ptir::plan::Region& region,
    std::string& output,
    std::string& error) {
    if (!library_region_valid(stage, region)) {
        error = "library region ABI is invalid";
        return false;
    }
    if (stage.channel_bindings.size() > kMetalM2MaxFusedChannels) {
        error = "fused region exceeds the 12-channel direct-binding limit";
        return false;
    }
    std::vector<std::uint32_t> bases(stage.ops.size(), 0);
    std::uint32_t next_value = 0;
    for (std::size_t node = 0; node < stage.ops.size(); ++node) {
        bases[node] = next_value;
        next_value += stage.ops[node].op.results;
    }

    std::ostringstream source;
    source << runtime_template << "\n"
           << "kernel void " << function_name << "(\n"
           << "    device M1Status* status [[buffer(0)]],\n"
           << "    const device M1ValueDesc* descriptors [[buffer(1)]],\n"
           << "    const device M1OpParams* params [[buffer(2)]],\n"
           << "    const device uint* offsets [[buffer(3)]],\n"
           << "    device uchar* scratch [[buffer(4)]],\n"
           << "    device uchar* temporary [[buffer(5)]],\n"
           << "    const device uchar* logits [[buffer(6)]]";
    for (std::size_t channel = 0;
         channel < stage.channel_bindings.size();
         ++channel) {
        source << ",\n    const device uchar* committed_" << channel
               << " [[buffer(" << 7 + channel * 2 << ")]],\n"
               << "    device uchar* pending_" << channel
               << " [[buffer(" << 8 + channel * 2 << ")]]";
    }
    source << ",\n    uint gid [[thread_position_in_grid]]) {\n"
           << "  if (gid != 0 || status->state != 1) return;\n";
    for (std::size_t channel = 0;
         channel < stage.channel_bindings.size();
         ++channel) {
        source << "  const device uchar* current_" << channel
               << " = committed_" << channel << ";\n";
    }
    for (const std::uint32_t node : region.nodes) {
        if (node >= stage.ops.size()) {
            error = "fused region node out of range";
            return false;
        }
        const auto& op = stage.ops[node].op;
        auto value_ptr = [&](std::uint32_t value) {
            return std::string("scratch + offsets[") +
                   std::to_string(value) + "]";
        };
        std::string a0 = "scratch";
        std::string a1 = "scratch";
        std::string a2 = "scratch";
        std::string o0 = "scratch";
        std::string o1 = "scratch";
        if (!op.args.empty()) a0 = value_ptr(op.args[0]);
        if (op.args.size() > 1) a1 = value_ptr(op.args[1]);
        if (op.args.size() > 2) a2 = value_ptr(op.args[2]);
        if (op.tag == PTIR_OP_PIVOT_THRESHOLD) {
            a1 = value_ptr(op.pred_payload);
        }
        if (op.results > 0) o0 = value_ptr(bases[node]);
        if (op.results > 1) o1 = value_ptr(bases[node] + 1);
        if (op.tag == PTIR_OP_CHAN_TAKE ||
            op.tag == PTIR_OP_CHAN_READ) {
            if (op.chan < 0 ||
                static_cast<std::size_t>(op.chan) >=
                    stage.channel_bindings.size()) {
                error = "fused channel root binding out of range";
                return false;
            }
            a0 = "current_" + std::to_string(op.chan);
        } else if (op.tag == PTIR_OP_CHAN_PUT) {
            if (op.chan < 0 ||
                static_cast<std::size_t>(op.chan) >=
                    stage.channel_bindings.size()) {
                error = "fused channel sink binding out of range";
                return false;
            }
            o0 = "pending_" + std::to_string(op.chan);
        } else if (op.tag == PTIR_OP_INTRINSIC_VAL) {
            a0 = "logits";
        }
        source << "  ptir_m1_execute(" << unsigned(op.tag)
               << "u, status, descriptors, params + " << node << ", "
               << a0 << ", " << a1 << ", " << a2 << ", "
               << o0 << ", " << o1 << ", temporary);\n"
               << "  if (status->state != 1) return;\n";
        if (op.tag == PTIR_OP_CHAN_PUT) {
            source << "  current_" << op.chan << " = pending_"
                   << op.chan << ";\n";
        }
    }
    source << "}\n";
    output = source.str();
    return true;
}

bool emit_grouped_fused_region_msl(
    const std::string& runtime_template,
    const std::string& function_name,
    const pie_native::ptir::plan::StagePlan& stage,
    const pie_native::ptir::plan::Region& region,
    std::string& output,
    std::string& error) {
    if (!library_region_valid(stage, region)) {
        error = "grouped library region ABI is invalid";
        return false;
    }
    std::vector<std::uint32_t> bases(stage.ops.size(), 0);
    std::uint32_t next_value = 0;
    for (std::size_t node = 0; node < stage.ops.size(); ++node) {
        bases[node] = next_value;
        next_value += stage.ops[node].op.results;
    }
    const std::size_t channel_count =
        used_channel_slots(stage);

    std::ostringstream source;
    source << runtime_template << "\n" << grouped_preamble()
           << "kernel void " << function_name << "(\n"
           << "    const device uchar* lane_bytes [[buffer(0)]],\n"
           << "    const device M1ValueDesc* all_descriptors [[buffer(1)]],\n"
           << "    const device M1OpParams* params [[buffer(2)]],\n"
           << "    const device uint* offsets [[buffer(3)]],\n"
           << "    device uchar* all_scratch [[buffer(4)]],\n"
           << "    const device M3GroupLayout* layout [[buffer(5)]],\n"
           << "    const device uint* channel_bindings [[buffer(6)]],\n"
           << "    device uchar* pending_flags [[buffer(7)]],\n"
           << "    const device uint* lane_indices [[buffer(8)]],\n"
           << "    const device M3RowMeta* all_row_meta [[buffer(9)]],\n"
           << "    const device uint* row_indices [[buffer(10)]],\n"
           << "    uint dispatch_lane [[thread_position_in_grid]]) {\n"
           << "  if (dispatch_lane >= layout->lane_count) return;\n"
           << "  const uint lane_index = lane_indices[dispatch_lane];\n"
           << "  const device M3LaneHeader* header = "
              "reinterpret_cast<const device M3LaneHeader*>(lane_bytes);\n"
           << "  const device M3LaneRecord* lanes = "
              "reinterpret_cast<const device M3LaneRecord*>(lane_bytes + sizeof(M3LaneHeader));\n"
           << "  const device M3LaneChannelSlot* slots = "
              "reinterpret_cast<const device M3LaneChannelSlot*>(lane_bytes + "
              "sizeof(M3LaneHeader) + header->lane_count * sizeof(M3LaneRecord));\n"
           << "  const M3LaneRecord lane = lanes[lane_index];\n"
           << "  const M3RowMeta row_meta = all_row_meta[lane_index];\n"
           << "  device M1Status* status = "
              "reinterpret_cast<device M1Status*>(lane.commit_slot);\n"
           << "  if (status->state != 1) return;\n"
           << "  const device M1ValueDesc* descriptors = all_descriptors + "
              "dispatch_lane * layout->value_count;\n"
           << "  const device M1OpParams* lane_params = params + "
              "dispatch_lane * layout->reserved2;\n"
           << "  device uchar* scratch = all_scratch + dispatch_lane * layout->scratch_stride;\n"
           << "  device uchar* temporary = scratch + layout->temporary_offset;\n"
           << "  const device bfloat* logits = "
              "reinterpret_cast<const device bfloat*>(lane.logits_base);\n";
    for (std::size_t channel = 0;
         channel < channel_count;
         ++channel) {
        source << "  const uint dense_" << channel
               << " = channel_bindings[dispatch_lane * layout->reserved0 + "
               << channel << "];\n";
        source << "  const M3LaneChannelSlot channel_" << channel
               << " = slots[lane.channel_slot_offset + "
               << "dense_" << channel << "];\n"
               << "  const uint pending_index_" << channel
               << " = lane.channel_slot_offset + dense_" << channel << ";\n"
               << "  const device uchar* current_" << channel
               << " = reinterpret_cast<const device uchar*>("
               << "pending_flags[pending_index_" << channel
               << "] != 0 ? channel_" << channel << ".pending_cell : channel_"
               << channel << ".committed_cell);\n"
               << "  device uchar* pending_" << channel
               << " = reinterpret_cast<device uchar*>(channel_" << channel
               << ".pending_cell);\n";
    }
    for (const std::uint32_t node : region.nodes) {
        if (node >= stage.ops.size()) {
            error = "grouped fused region node out of range";
            return false;
        }
        const auto& op = stage.ops[node].op;
        auto value_ptr = [&](std::uint32_t value) {
            return std::string("scratch + offsets[") +
                   std::to_string(value) + "]";
        };
        std::string a0 = "scratch", a1 = "scratch", a2 = "scratch";
        std::string o0 = "scratch", o1 = "scratch";
        if (!op.args.empty()) a0 = value_ptr(op.args[0]);
        if (op.args.size() > 1) a1 = value_ptr(op.args[1]);
        if (op.args.size() > 2) a2 = value_ptr(op.args[2]);
        if (op.tag == PTIR_OP_PIVOT_THRESHOLD)
            a1 = value_ptr(op.pred_payload);
        if (op.results > 0) o0 = value_ptr(bases[node]);
        if (op.results > 1) o1 = value_ptr(bases[node] + 1);
        if (op.tag == PTIR_OP_INTRINSIC_VAL &&
            op.intr == PTIR_INTR_MTP_DRAFTS) {
            source
                << "  {\n"
                << "    const M1ValueDesc draft_desc = descriptors["
                << bases[node] << "];\n"
                << "    if (layout->vocab == 0u || "
                   "row_meta.mtp_offset > row_meta.count || "
                   "draft_desc.len > row_meta.count - row_meta.mtp_offset) "
                   "{ m1_fault(status, 0xA0u); return; }\n"
                << "    device int* draft_out = reinterpret_cast<device int*>("
                << o0 << ");\n"
                << "    for (uint draft = 0; draft < draft_desc.len; ++draft) {\n"
                << "      const uint source_row = "
                   "row_indices[row_meta.offset + row_meta.mtp_offset + draft];\n"
                << "      float best_value = -INFINITY;\n"
                << "      uint best_index = 0u;\n"
                << "      bool have = false;\n"
                << "      for (uint column = 0; column < layout->vocab; ++column) {\n"
                << "        const float value = float(logits["
                   "ulong(source_row) * layout->vocab + column]);\n"
                << "        if (!isnan(value) && (!have || value > best_value || "
                   "(value == best_value && column < best_index))) { "
                   "best_value = value; best_index = column; have = true; }\n"
                << "      }\n"
                << "      draft_out[draft] = int(have ? best_index : 0u);\n"
                << "    }\n"
                << "  }\n";
            continue;
        }
        if (op.tag == PTIR_OP_INTRINSIC_VAL &&
            (op.intr == PTIR_INTR_LOGITS ||
             op.intr == PTIR_INTR_MTP_LOGITS)) {
            source
                << "  {\n"
                << "    const M1ValueDesc intrinsic_desc = descriptors["
                << bases[node] << "];\n"
                << "    const uint intrinsic_row_base = "
                << (op.intr == PTIR_INTR_MTP_LOGITS
                        ? "row_meta.mtp_offset"
                        : "0u")
                << ";\n"
                << "    if (layout->vocab == 0u || "
                   "intrinsic_desc.len % layout->vocab != 0u || "
                   "intrinsic_row_base > row_meta.count || "
                   "intrinsic_desc.len / layout->vocab > "
                   "row_meta.count - intrinsic_row_base) "
                   "{ m1_fault(status, 0xA0u); return; }\n"
                << "    device float* intrinsic_out = "
                   "reinterpret_cast<device float*>("
                << o0 << ");\n"
                << "    for (uint element = 0; element < intrinsic_desc.len; "
                   "++element) {\n"
                << "      const uint logical_row = element / layout->vocab;\n"
                << "      const uint column = element % layout->vocab;\n"
                << "      const uint source_row = "
                   "row_indices[row_meta.offset + intrinsic_row_base + "
                   "logical_row];\n"
                << "      intrinsic_out[element] = float(logits["
                   "ulong(source_row) * layout->vocab + column]);\n"
                << "    }\n"
                << "  }\n";
            continue;
        }
        if (op.tag == PTIR_OP_CHAN_TAKE ||
            op.tag == PTIR_OP_CHAN_READ) {
            a0 = "current_" + std::to_string(op.chan);
        } else if (op.tag == PTIR_OP_CHAN_PUT) {
            o0 = "pending_" + std::to_string(op.chan);
        } else if (op.tag == PTIR_OP_INTRINSIC_VAL) {
            a0 = "logits";
        }
        source << "  ptir_m1_execute(" << unsigned(op.tag)
               << "u, status, descriptors, lane_params + " << node << ", "
               << a0 << ", " << a1 << ", " << a2 << ", "
               << o0 << ", " << o1 << ", temporary);\n"
               << "  if (status->state != 1) return;\n";
        if (op.tag == PTIR_OP_CHAN_PUT) {
            source << "  current_" << op.chan << " = pending_"
                   << op.chan << ";\n"
                   << "  pending_flags[pending_index_" << op.chan
                   << "] = 1;\n";
        }
    }
    source << "}\n";
    output = source.str();
    return true;
}

bool emit_grouped_nucleus_msl(
    const std::string& runtime_template,
    const std::string& function_name,
    const pie_native::ptir::plan::StagePlan& stage,
    const pie_native::ptir::plan::Region& region,
    std::string& output,
    std::string& error) {
    if (region.library_op != PTIR_LIBRARY_NUCLEUS_SAMPLE ||
        !library_region_valid(stage, region)) {
        error = "invalid grouped nucleus library region";
        return false;
    }
    const std::uint32_t logits_value = region.inputs[0];
    const std::uint32_t top_p_value = region.inputs[1];
    const std::uint32_t state_value = region.inputs[2];
    const std::uint32_t output_value = region.outputs[0];

    std::ostringstream source;
    source << runtime_template << "\n" << grouped_preamble() << R"MSL(
inline uint m3_nucleus_order_digit(float value, uint pass) {
  if (pass < 8u) {
    if (isnan(value)) return 0u;
    if (value == 0.0f) value = 0.0f;
    const uint bits = as_type<uint>(value);
    const uint ascending =
        (bits & 0x80000000u) != 0u ? ~bits : (bits ^ 0x80000000u);
    const uint descending = ~ascending;
    return (descending >> (pass * 4u)) & 15u;
  }
  return isnan(value) ? 1u : 0u;
}

kernel void )MSL"
           << function_name << R"MSL((
    const device uchar* lane_bytes [[buffer(0)]],
    const device M1ValueDesc* all_descriptors [[buffer(1)]],
    const device M1OpParams* params [[buffer(2)]],
    const device uint* offsets [[buffer(3)]],
    device uchar* all_scratch [[buffer(4)]],
    const device M3GroupLayout* layout [[buffer(5)]],
    const device uint* channel_bindings [[buffer(6)]],
    device uchar* pending_flags [[buffer(7)]],
    const device uint* lane_indices [[buffer(8)]],
    const device M3RowMeta* all_row_meta [[buffer(9)]],
    const device uint* row_indices [[buffer(10)]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint threads [[threads_per_threadgroup]],
    uint group_position [[threadgroup_position_in_grid]]) {
  (void)params;
  (void)channel_bindings;
  (void)pending_flags;
  (void)all_row_meta;
  (void)row_indices;
  if (threads != 256u || layout->reserved1 == 0u) return;
  const uint dispatch_lane = group_position / layout->reserved1;
  const uint row = group_position % layout->reserved1;
  if (dispatch_lane >= layout->lane_count) return;
  const uint lane_index = lane_indices[dispatch_lane];
  const device M3LaneHeader* header =
      reinterpret_cast<const device M3LaneHeader*>(lane_bytes);
  const device M3LaneRecord* lanes =
      reinterpret_cast<const device M3LaneRecord*>(
          lane_bytes + sizeof(M3LaneHeader));
  const M3LaneRecord lane = lanes[lane_index];
  device M1Status* status =
      reinterpret_cast<device M1Status*>(lane.commit_slot);
  if (status->state != 1u) return;
  const device M1ValueDesc* descriptors =
      all_descriptors + dispatch_lane * layout->value_count;
  device uchar* scratch =
      all_scratch + dispatch_lane * layout->scratch_stride;
)MSL";
    source << "  constexpr uint kLogits = " << logits_value << "u;\n"
           << "  constexpr uint kTopP = " << top_p_value << "u;\n"
           << "  constexpr uint kState = " << state_value << "u;\n"
           << "  constexpr uint kOutput = " << output_value << "u;\n";
    source << R"MSL(
  const M1ValueDesc logits_descriptor = descriptors[kLogits];
  if (row >= logits_descriptor.rows) return;
  const uint len = logits_descriptor.last;
  if (len == 0u) {
    if (thread_index == 0u) {
      reinterpret_cast<device int*>(
          scratch + offsets[kOutput])[row] = 0;
    }
    return;
  }
  const device float* logits =
      reinterpret_cast<const device float*>(scratch + offsets[kLogits]) +
      ulong(row) * len;
  const device float* top_p_values =
      reinterpret_cast<const device float*>(scratch + offsets[kTopP]);
  const device uint* state =
      reinterpret_cast<const device uint*>(scratch + offsets[kState]);
  device int* sampled =
      reinterpret_cast<device int*>(scratch + offsets[kOutput]);
  device uchar* workspace_bytes =
      scratch + layout->temporary_offset + ulong(row) * len * 16ul;
  device uint* order_a = reinterpret_cast<device uint*>(workspace_bytes);
  device uint* order_b = order_a + len;
  device float* probabilities =
      reinterpret_cast<device float*>(order_b + len);
  device float* reduction_a = probabilities + len;
  device float* reduction_b =
      reinterpret_cast<device float*>(order_a);

  threadgroup uint digit_offsets[256 * 16];
  threadgroup uint selected_count;
  threadgroup float candidate_values[256];
  threadgroup uint candidate_indices[256];
  threadgroup uchar candidate_have[256];

  for (uint index = thread_index; index < len; index += threads) {
    reduction_a[index] = logits[index];
  }
  threadgroup_barrier(mem_flags::mem_device);
  device float* reduction_input = reduction_a;
  device float* reduction_output = reduction_b;
  uint count = len;
  while (count > 1u) {
    const uint chunks = (count + 31u) / 32u;
    for (uint chunk = thread_index; chunk < chunks; chunk += threads) {
      float values[32];
      for (uint lane_in_chunk = 0; lane_in_chunk < 32u; ++lane_in_chunk) {
        const uint index = chunk * 32u + lane_in_chunk;
        values[lane_in_chunk] =
            index < count ? reduction_input[index] : -INFINITY;
      }
      for (uint offset = 16u; offset > 0u; offset >>= 1u)
        for (uint lane_in_chunk = 0; lane_in_chunk < offset;
             ++lane_in_chunk)
          values[lane_in_chunk] = m1_canonical_max(
              values[lane_in_chunk],
              values[lane_in_chunk + offset]);
      reduction_output[chunk] = values[0];
    }
    threadgroup_barrier(mem_flags::mem_device);
    device float* swap = reduction_input;
    reduction_input = reduction_output;
    reduction_output = swap;
    count = chunks;
  }
  const float maximum_value = reduction_input[0];
  threadgroup_barrier(mem_flags::mem_device);
  for (uint index = thread_index; index < len; index += threads) {
    const float value = precise::exp(logits[index] - maximum_value);
    probabilities[index] = value;
    reduction_a[index] = value;
  }
  threadgroup_barrier(mem_flags::mem_device);
  reduction_input = reduction_a;
  reduction_output = reduction_b;
  count = len;
  while (count > 1u) {
    const uint chunks = (count + 31u) / 32u;
    for (uint chunk = thread_index; chunk < chunks; chunk += threads) {
      float values[32];
      for (uint lane_in_chunk = 0; lane_in_chunk < 32u; ++lane_in_chunk) {
        const uint index = chunk * 32u + lane_in_chunk;
        values[lane_in_chunk] =
            index < count ? reduction_input[index] : 0.0f;
      }
      for (uint offset = 16u; offset > 0u; offset >>= 1u)
        for (uint lane_in_chunk = 0; lane_in_chunk < offset;
             ++lane_in_chunk)
          values[lane_in_chunk] += values[lane_in_chunk + offset];
      reduction_output[chunk] = values[0];
    }
    threadgroup_barrier(mem_flags::mem_device);
    device float* swap = reduction_input;
    reduction_input = reduction_output;
    reduction_output = swap;
    count = chunks;
  }
  const float probability_sum = reduction_input[0];
  threadgroup_barrier(mem_flags::mem_device);
  for (uint index = thread_index; index < len; index += threads) {
    probabilities[index] /= probability_sum;
    order_a[index] = index;
  }
  threadgroup_barrier(mem_flags::mem_device);

  device uint* input_order = order_a;
  device uint* output_order = order_b;
  const uint chunk_begin =
      uint((ulong(len) * thread_index) / threads);
  const uint chunk_end =
      uint((ulong(len) * (thread_index + 1u)) / threads);
  for (uint pass = 0u; pass < 9u; ++pass) {
    uint digit_counts[16];
    uint digit_written[16];
    for (uint digit = 0u; digit < 16u; ++digit) {
      digit_counts[digit] = 0u;
      digit_written[digit] = 0u;
    }
    for (uint position = chunk_begin; position < chunk_end; ++position) {
      const uint index = input_order[position];
      ++digit_counts[
          m3_nucleus_order_digit(probabilities[index], pass)];
    }
    for (uint digit = 0u; digit < 16u; ++digit)
      digit_offsets[thread_index * 16u + digit] =
          digit_counts[digit];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (thread_index == 0u) {
      uint base = 0u;
      for (uint digit = 0u; digit < 16u; ++digit) {
        uint running = base;
        for (uint worker = 0u; worker < threads; ++worker) {
          const uint offset = worker * 16u + digit;
          const uint count_for_worker = digit_offsets[offset];
          digit_offsets[offset] = running;
          running += count_for_worker;
        }
        base = running;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint position = chunk_begin; position < chunk_end; ++position) {
      const uint index = input_order[position];
      const uint digit =
          m3_nucleus_order_digit(probabilities[index], pass);
      output_order[
          digit_offsets[thread_index * 16u + digit] +
          digit_written[digit]++] = index;
    }
    threadgroup_barrier(mem_flags::mem_device);
    device uint* swap = input_order;
    input_order = output_order;
    output_order = swap;
  }

  if (thread_index == 0u) {
    const M1ValueDesc top_p_descriptor = descriptors[kTopP];
    const float threshold =
        top_p_values[top_p_descriptor.len <= 1u ? 0u : row];
    float exclusive = 0.0f;
    uint selected = 0u;
    for (uint position = 0u; position < len; ++position) {
      if (!(exclusive < threshold)) break;
      ++selected;
      exclusive += probabilities[input_order[position]];
    }
    selected_count = selected;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const ulong seed = ptir_rng_keyed_seed(state[0], state[1]);
  float best_value = -INFINITY;
  uint best_index = 0u;
  bool have = false;
  for (uint position = thread_index; position < len; position += threads) {
    const uint index = input_order[position];
    const float uniform =
        ptir_rng_hash_uniform(seed, uint(ulong(row) * len + index));
    const float noise =
        -precise::log(-precise::log(uniform));
    const float score =
        (position < selected_count ? logits[index] : -INFINITY) + noise;
    if (!isnan(score) &&
        (!have || score > best_value ||
         (score == best_value && index < best_index))) {
      best_value = score;
      best_index = index;
      have = true;
    }
  }
  candidate_values[thread_index] = best_value;
  candidate_indices[thread_index] = best_index;
  candidate_have[thread_index] = have ? 1u : 0u;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint offset = 128u; offset > 0u; offset >>= 1u) {
    if (thread_index < offset) {
      const uint other = thread_index + offset;
      if (candidate_have[other] != 0u &&
          (candidate_have[thread_index] == 0u ||
           candidate_values[other] > candidate_values[thread_index] ||
           (candidate_values[other] == candidate_values[thread_index] &&
            candidate_indices[other] < candidate_indices[thread_index]))) {
        candidate_values[thread_index] = candidate_values[other];
        candidate_indices[thread_index] = candidate_indices[other];
        candidate_have[thread_index] = 1u;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (thread_index == 0u)
    sampled[row] =
        int(candidate_have[0] != 0u ? candidate_indices[0] : 0u);
}
)MSL";
    output = source.str();
    return true;
}

bool emit_grouped_topk_msl(
    const std::string& runtime_template,
    const std::string& function_name,
    const pie_native::ptir::plan::StagePlan& stage,
    const pie_native::ptir::plan::Region& region,
    std::string& output,
    std::string& error) {
    if (!region.library ||
        region.library_op != PTIR_LIBRARY_TOP_K ||
        !library_region_valid(stage, region)) {
        error = "invalid grouped TopK library region";
        return false;
    }
    std::vector<std::uint32_t> bases(stage.ops.size(), 0);
    std::uint32_t next_value = 0;
    for (std::size_t node = 0; node < stage.ops.size(); ++node) {
        bases[node] = next_value;
        next_value += stage.ops[node].op.results;
    }
    const std::uint32_t topk_node = region.nodes.front();
    if (topk_node >= stage.ops.size()) {
        error = "TopK library node is out of range";
        return false;
    }
    const auto& topk = stage.ops[topk_node].op;
    if (topk.tag != PTIR_OP_TOP_K || topk.args.size() != 1 ||
        topk.results != 2 ||
        bases[topk_node] + 1 >= stage.value_types.size()) {
        error = "TopK library node is invalid";
        return false;
    }

    std::ostringstream source;
    source << runtime_template << "\n" << grouped_preamble() << R"MSL(
inline uint m3_topk_order_digit(float value, uint pass) {
  if (pass < 8u) {
    if (isnan(value)) return 0u;
    if (value == 0.0f) value = 0.0f;
    const uint bits = as_type<uint>(value);
    const uint ascending =
        (bits & 0x80000000u) != 0u ? ~bits : (bits ^ 0x80000000u);
    return ((~ascending) >> (pass * 4u)) & 15u;
  }
  return isnan(value) ? 1u : 0u;
}

kernel void )MSL"
           << function_name << R"MSL((
    const device uchar* lane_bytes [[buffer(0)]],
    const device M1ValueDesc* all_descriptors [[buffer(1)]],
    const device M1OpParams* params [[buffer(2)]],
    const device uint* offsets [[buffer(3)]],
    device uchar* all_scratch [[buffer(4)]],
    const device M3GroupLayout* layout [[buffer(5)]],
    const device uint* channel_bindings [[buffer(6)]],
    device uchar* pending_flags [[buffer(7)]],
    const device uint* lane_indices [[buffer(8)]],
    const device M3RowMeta* all_row_meta [[buffer(9)]],
    const device uint* row_indices [[buffer(10)]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint threads [[threads_per_threadgroup]],
    uint group_position [[threadgroup_position_in_grid]]) {
  (void)channel_bindings;
  (void)pending_flags;
  (void)all_row_meta;
  (void)row_indices;
  if (threads != 256u || layout->reserved1 == 0u) return;
  const uint dispatch_lane = group_position / layout->reserved1;
  const uint row = group_position % layout->reserved1;
  if (dispatch_lane >= layout->lane_count) return;
  const uint lane_index = lane_indices[dispatch_lane];
  const device M3LaneHeader* header =
      reinterpret_cast<const device M3LaneHeader*>(lane_bytes);
  const device M3LaneRecord* lanes =
      reinterpret_cast<const device M3LaneRecord*>(
          lane_bytes + sizeof(M3LaneHeader));
  device M1Status* status =
      reinterpret_cast<device M1Status*>(lanes[lane_index].commit_slot);
  if (status->state != 1u) return;
  const device M1ValueDesc* descriptors =
      all_descriptors + dispatch_lane * layout->value_count;
  const device M1OpParams* lane_params =
      params + dispatch_lane * layout->reserved2;
  device uchar* scratch =
      all_scratch + dispatch_lane * layout->scratch_stride;
  device uchar* temporary = scratch + layout->temporary_offset;
)MSL";
    source << "  constexpr uint kInput = " << topk.args[0] << "u;\n"
           << "  constexpr uint kValues = " << bases[topk_node] << "u;\n"
           << "  constexpr uint kIndices = " << bases[topk_node] + 1 << "u;\n"
           << "  constexpr uint k = " << topk.imm << "u;\n";
    source << R"MSL(
  const M1ValueDesc input_desc = descriptors[kInput];
  if (row >= input_desc.rows) return;
  const uint len = input_desc.last;
  const device float* input =
      reinterpret_cast<const device float*>(scratch + offsets[kInput]) +
      ulong(row) * len;
  device float* top_values =
      reinterpret_cast<device float*>(scratch + offsets[kValues]);
  device uint* top_indices =
      reinterpret_cast<device uint*>(scratch + offsets[kIndices]);
  device uint* order_a =
      reinterpret_cast<device uint*>(
          temporary + ulong(row) * len * 8ul);
  device uint* order_b = order_a + len;
  threadgroup uint digit_offsets[256 * 16];

  for (uint index = thread_index; index < len; index += threads)
    order_a[index] = index;
  threadgroup_barrier(mem_flags::mem_device);
  device uint* input_order = order_a;
  device uint* output_order = order_b;
  const uint chunk_begin =
      uint((ulong(len) * thread_index) / threads);
  const uint chunk_end =
      uint((ulong(len) * (thread_index + 1u)) / threads);
  for (uint pass = 0u; pass < 9u; ++pass) {
    uint digit_counts[16];
    uint digit_written[16];
    for (uint digit = 0u; digit < 16u; ++digit) {
      digit_counts[digit] = 0u;
      digit_written[digit] = 0u;
    }
    for (uint position = chunk_begin; position < chunk_end; ++position) {
      const uint index = input_order[position];
      ++digit_counts[m3_topk_order_digit(input[index], pass)];
    }
    for (uint digit = 0u; digit < 16u; ++digit)
      digit_offsets[thread_index * 16u + digit] = digit_counts[digit];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (thread_index == 0u) {
      uint base = 0u;
      for (uint digit = 0u; digit < 16u; ++digit) {
        uint running = base;
        for (uint worker = 0u; worker < threads; ++worker) {
          const uint offset = worker * 16u + digit;
          const uint count_for_worker = digit_offsets[offset];
          digit_offsets[offset] = running;
          running += count_for_worker;
        }
        base = running;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint position = chunk_begin; position < chunk_end; ++position) {
      const uint index = input_order[position];
      const uint digit = m3_topk_order_digit(input[index], pass);
      output_order[
          digit_offsets[thread_index * 16u + digit] +
          digit_written[digit]++] = index;
    }
    threadgroup_barrier(mem_flags::mem_device);
    device uint* swap = input_order;
    input_order = output_order;
    output_order = swap;
  }
  if (thread_index == 0u) {
    const uint count = min(k, len);
    for (uint position = 0u; position < count; ++position) {
      const uint index = input_order[position];
      top_values[ulong(row) * k + position] = input[index];
      top_indices[ulong(row) * k + position] = index;
    }
  }
  threadgroup_barrier(mem_flags::mem_device);
)MSL";
    source << "}\n";
    output = source.str();
    return true;
}

std::string emit_grouped_readiness_msl(
    const std::string& function_name) {
    std::ostringstream source;
    source << "#include <metal_stdlib>\nusing namespace metal;\n"
           << "struct M1Status { uint state; uint fault; uint reserved0; uint reserved1; };\n"
           << grouped_preamble()
           << "kernel void " << function_name
           << "(const device uchar* lane_bytes [[buffer(0)]], "
              "const device M3ChannelMeta* meta [[buffer(1)]], "
              "uint lane_index [[thread_position_in_grid]]) {\n"
           << "  const device M3LaneHeader* header = reinterpret_cast<const device M3LaneHeader*>(lane_bytes);\n"
           << "  if (lane_index >= header->lane_count) return;\n"
           << "  const device M3LaneRecord* lanes = reinterpret_cast<const device M3LaneRecord*>(lane_bytes + sizeof(M3LaneHeader));\n"
           << "  const device M3LaneChannelSlot* slots = reinterpret_cast<const device M3LaneChannelSlot*>(lane_bytes + sizeof(M3LaneHeader) + header->lane_count * sizeof(M3LaneRecord));\n"
           << "  const M3LaneRecord lane = lanes[lane_index];\n"
           << "  device M1Status* status = reinterpret_cast<device M1Status*>(lane.commit_slot);\n"
           << "  status->state = 0; status->fault = 0;\n"
           << "  for (uint channel = 0; channel < header->channel_count; ++channel) {\n"
           << "    const M3ChannelMeta m = meta[lane.channel_slot_offset + channel];\n"
           << "    if ((m.flags & 1u) == 0u) continue;\n"
           << "    const device ulong* words = reinterpret_cast<const device ulong*>(m.words);\n"
           << "    const ulong head = words[0], tail = words[1];\n"
           << "    const M3LaneChannelSlot slot = slots[lane.channel_slot_offset + channel];\n"
           << "    if (words[2] != 0ul || words[3] != 0ul || tail < head) "
              "{ status->state = 3; status->fault = 0x700u + channel; return; }\n"
           << "    bool retry = slot.expected_head != ~0ul && head != slot.expected_head;\n"
           << "    retry = retry || (((m.flags & 2u) != 0u) && tail <= head);\n"
           << "    retry = retry || (((m.flags & 4u) != 0u) && tail - head >= ulong(m.capacity));\n"
           << "    if ((m.flags & 16u) != 0u) {\n"
           << "      const ulong credit = ((m.flags & 10u) == 10u) ? 1ul : 0ul;\n"
           << "      retry = retry || slot.expected_tail == ~0ul || tail != slot.expected_tail || "
              "tail - head >= ulong(m.capacity) + credit;\n"
           << "    }\n"
           << "    if (retry) { status->state = (m.flags & 32u) != 0u ? 3u : 2u; "
              "status->fault = 0x780u + channel; return; }\n"
           << "  }\n"
           << "  status->state = 1;\n}\n";
    return source.str();
}

std::string emit_grouped_commit_msl(
    const std::string& function_name) {
    std::ostringstream source;
    source << "#include <metal_stdlib>\nusing namespace metal;\n"
           << "struct M1Status { uint state; uint fault; uint reserved0; uint reserved1; };\n"
           << grouped_preamble()
           << "kernel void " << function_name
           << "(const device uchar* lane_bytes [[buffer(0)]], "
              "const device M3ChannelMeta* meta [[buffer(1)]], "
              "uint lane_index [[thread_position_in_grid]]) {\n"
           << "  const device M3LaneHeader* header = reinterpret_cast<const device M3LaneHeader*>(lane_bytes);\n"
           << "  if (lane_index >= header->lane_count) return;\n"
           << "  const device M3LaneRecord* lanes = reinterpret_cast<const device M3LaneRecord*>(lane_bytes + sizeof(M3LaneHeader));\n"
           << "  const M3LaneRecord lane = lanes[lane_index];\n"
           << "  device M1Status* status = reinterpret_cast<device M1Status*>(lane.commit_slot);\n"
           << "  if (status->state != 1) return;\n"
           << "  for (uint channel = 0; channel < header->channel_count; ++channel) {\n"
           << "    const M3ChannelMeta m = meta[lane.channel_slot_offset + channel];\n"
           << "    if ((m.flags & 1u) == 0u) continue;\n"
           << "    device ulong* words = reinterpret_cast<device ulong*>(m.words);\n"
           << "    const ulong old_head = words[0], old_tail = words[1];\n"
           << "    if ((m.flags & 8u) != 0u && old_tail > old_head) words[0] = old_head + 1ul;\n"
           << "    if ((m.flags & 16u) != 0u) words[1] = old_tail + 1ul;\n"
           << "  }\n"
           << "  status->state = 4;\n}\n";
    return source.str();
}

}  // namespace pie::metal::pipeline
