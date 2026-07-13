#pragma once

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

#include "pie_native/ptir/bound.hpp"
#include "pie_native/ptir/container.hpp"

namespace pie::metal::tests {

inline void put_u16(std::vector<std::uint8_t>& out, std::uint16_t value) {
    out.push_back(static_cast<std::uint8_t>(value));
    out.push_back(static_cast<std::uint8_t>(value >> 8));
}

inline void put_u32(std::vector<std::uint8_t>& out, std::uint32_t value) {
    for (int byte = 0; byte < 4; ++byte) {
        out.push_back(static_cast<std::uint8_t>(value >> (byte * 8)));
    }
}

inline void put_u64(std::vector<std::uint8_t>& out, std::uint64_t value) {
    for (int byte = 0; byte < 8; ++byte) {
        out.push_back(static_cast<std::uint8_t>(value >> (byte * 8)));
    }
}

inline void put_shape(
    std::vector<std::uint8_t>& out,
    const pie_native::ptir::container::CShape& shape) {
    out.push_back(shape.rank);
    for (std::uint8_t dimension = 0; dimension < shape.rank; ++dimension) {
        put_u32(out, shape.dims[dimension]);
    }
}

inline std::vector<std::uint8_t> encode_op(
    const pie_native::ptir::container::COp& op) {
    std::vector<std::uint8_t> out{op.tag};
    auto arg = [&](std::size_t index) {
        put_u32(out, index < op.args.size() ? op.args[index] : 0);
    };
    switch (op.tag) {
        case PTIR_OP_EXP: case PTIR_OP_LOG: case PTIR_OP_NEG:
        case PTIR_OP_RECIP: case PTIR_OP_ABS: case PTIR_OP_SIGN:
        case PTIR_OP_NOT: case PTIR_OP_TRANSPOSE:
        case PTIR_OP_REDUCE_SUM: case PTIR_OP_REDUCE_MAX:
        case PTIR_OP_REDUCE_MIN: case PTIR_OP_REDUCE_ARGMAX:
        case PTIR_OP_CUMSUM: case PTIR_OP_CUMPROD:
        case PTIR_OP_SORT_DESC:
            arg(0);
            break;
        case PTIR_OP_CAST:
            arg(0);
            out.push_back(op.dtype);
            break;
        case PTIR_OP_ADD: case PTIR_OP_SUB: case PTIR_OP_MUL:
        case PTIR_OP_DIV: case PTIR_OP_MAX_ELEM: case PTIR_OP_MIN_ELEM:
        case PTIR_OP_REM: case PTIR_OP_GT: case PTIR_OP_GE:
        case PTIR_OP_EQ: case PTIR_OP_NE: case PTIR_OP_LT:
        case PTIR_OP_LE: case PTIR_OP_AND: case PTIR_OP_OR:
        case PTIR_OP_MATMUL: case PTIR_OP_GATHER:
        case PTIR_OP_GATHER_ROW: case PTIR_OP_MASK_APPLY_PACKED:
            arg(0);
            arg(1);
            break;
        case PTIR_OP_SELECT: case PTIR_OP_SCATTER_ADD:
        case PTIR_OP_SCATTER_SET:
            arg(0);
            arg(1);
            arg(2);
            break;
        case PTIR_OP_BROADCAST: case PTIR_OP_RESHAPE:
            arg(0);
            put_shape(out, op.shape);
            break;
        case PTIR_OP_TOP_K:
            arg(0);
            put_u32(out, op.imm);
            break;
        case PTIR_OP_PIVOT_THRESHOLD:
            arg(0);
            out.push_back(op.pred_tag);
            put_u32(out, op.pred_payload);
            break;
        case PTIR_OP_CAUSAL_MASK:
            arg(0);
            put_u32(out, op.imm);
            break;
        case PTIR_OP_SLIDING_WINDOW_MASK:
            arg(0);
            put_u32(out, op.imm);
            put_u32(out, op.imm2);
            break;
        case PTIR_OP_SINK_WINDOW_MASK:
            arg(0);
            put_u32(out, op.imm);
            put_u32(out, op.imm2);
            put_u32(out, op.imm3);
            break;
        case PTIR_OP_IOTA:
            put_u32(out, op.imm);
            break;
        case PTIR_OP_RNG:
            put_u32(out, op.imm);
            put_shape(out, op.shape);
            out.push_back(op.kind);
            break;
        case PTIR_OP_RNG_KEYED:
            arg(0);
            put_shape(out, op.shape);
            out.push_back(op.kind);
            break;
        case PTIR_OP_CONST:
            out.push_back(op.lit_dtype);
            put_u32(out, op.lit_bits);
            break;
        case PTIR_OP_CHAN_TAKE: case PTIR_OP_CHAN_READ:
            put_u32(out, static_cast<std::uint32_t>(op.chan));
            break;
        case PTIR_OP_CHAN_PUT:
            put_u32(out, static_cast<std::uint32_t>(op.chan));
            arg(0);
            break;
        case PTIR_OP_INTRINSIC_VAL:
            put_u16(out, op.intr);
            out.push_back(op.dtype);
            put_shape(out, op.shape);
            break;
        case PTIR_OP_KERNEL_CALL:
            put_u16(out, op.name_idx);
            out.push_back(op.dtype);
            put_shape(out, op.shape);
            out.push_back(static_cast<std::uint8_t>(op.args.size()));
            for (std::uint32_t value : op.args) put_u32(out, value);
            break;
        case PTIR_OP_SINK_CALL:
            put_u16(out, op.name_idx);
            out.push_back(static_cast<std::uint8_t>(op.args.size()));
            for (std::uint32_t value : op.args) put_u32(out, value);
            break;
        default:
            break;
    }
    return out;
}

inline std::vector<std::uint8_t> plan_signature_bytes(
    std::uint8_t stage,
    const std::vector<pie_native::ptir::container::COp>& ops,
    const pie_native::ptir::bound::StageTypes& types) {
    std::vector<std::uint8_t> bytes{stage};
    for (const auto& op : ops) {
        const auto encoded = encode_op(op);
        put_u32(bytes, static_cast<std::uint32_t>(encoded.size()));
        bytes.insert(bytes.end(), encoded.begin(), encoded.end());
    }
    for (const auto& type : types.value_types) {
        bytes.push_back(static_cast<std::uint8_t>(type.dtype));
        bytes.push_back(static_cast<std::uint8_t>(type.shape.dims.size()));
        for (std::uint32_t dimension : type.shape.dims) {
            put_u32(bytes, dimension);
        }
    }
    return bytes;
}

inline void put_vector(
    std::vector<std::uint8_t>& out,
    const std::vector<std::uint32_t>& values) {
    put_u32(out, static_cast<std::uint32_t>(values.size()));
    for (std::uint32_t value : values) put_u32(out, value);
}

struct ExplicitLibraryRegion {
    std::uint8_t library_op = 0;
    std::vector<std::uint32_t> nodes;
    std::vector<std::uint32_t> inputs;
    std::vector<std::uint32_t> outputs;
};

inline bool library_for_op(
    const pie_native::ptir::container::COp& op,
    std::uint8_t& library) {
    switch (op.tag) {
        case PTIR_OP_TOP_K:
            library = PTIR_LIBRARY_TOP_K;
            return true;
        case PTIR_OP_SORT_DESC:
            library = PTIR_LIBRARY_SORT;
            return true;
        case PTIR_OP_CUMSUM:
        case PTIR_OP_CUMPROD:
            library = PTIR_LIBRARY_SCAN;
            return true;
        case PTIR_OP_MATMUL:
            library = PTIR_LIBRARY_MATMUL;
            return true;
        case PTIR_OP_KERNEL_CALL:
            library = PTIR_LIBRARY_SECOND_PARTY;
            return true;
        default:
            library = 0;
            return false;
    }
}

inline void put_region(
    std::vector<std::uint8_t>& out,
    const std::vector<std::uint32_t>& nodes,
    bool library,
    std::uint8_t library_op,
    const std::vector<std::uint32_t>& inputs = {},
    const std::vector<std::uint32_t>& outputs = {}) {
    out.push_back(library ? 1 : 0);
    out.push_back(library_op);
    out.push_back(
        library ? PTIR_SCHEDULE_LIBRARY
                : PTIR_SCHEDULE_ONE_CTA_PER_ROW);
    put_vector(out, nodes);
    put_vector(out, inputs);
    put_vector(out, outputs);
    put_u32(out, 0);
}

inline void put_partition(
    std::vector<std::uint8_t>& out,
    std::uint8_t kind,
    const std::vector<pie_native::ptir::container::COp>& ops,
    const std::vector<ExplicitLibraryRegion>& explicit_libraries = {}) {
    out.push_back(kind);
    out.push_back(0);
    std::vector<std::vector<std::uint32_t>> regions;
    if (kind == 0) {
        for (std::size_t node = 0; node < ops.size(); ++node) {
            regions.push_back({static_cast<std::uint32_t>(node)});
        }
    } else {
        for (std::size_t node = 0; node < ops.size();) {
            const auto explicit_region = std::find_if(
                explicit_libraries.begin(),
                explicit_libraries.end(),
                [node](const ExplicitLibraryRegion& candidate) {
                    return !candidate.nodes.empty() &&
                        candidate.nodes.front() == node;
                });
            if (explicit_region != explicit_libraries.end()) {
                regions.push_back(explicit_region->nodes);
                node = explicit_region->nodes.back() + 1;
                continue;
            }
            std::uint8_t library = 0;
            if (library_for_op(ops[node], library)) {
                regions.push_back({static_cast<std::uint32_t>(node++)});
                continue;
            }
            std::vector<std::uint32_t> generated;
            do {
                generated.push_back(static_cast<std::uint32_t>(node++));
            } while (
                node < ops.size() &&
                std::none_of(
                    explicit_libraries.begin(),
                    explicit_libraries.end(),
                    [node](const ExplicitLibraryRegion& candidate) {
                        return !candidate.nodes.empty() &&
                            candidate.nodes.front() == node;
                    }) &&
                !library_for_op(ops[node], library));
            regions.push_back(std::move(generated));
        }
    }
    put_u32(out, static_cast<std::uint32_t>(regions.size()));
    for (const auto& nodes : regions) {
        const auto explicit_region = std::find_if(
            explicit_libraries.begin(),
            explicit_libraries.end(),
            [&](const ExplicitLibraryRegion& candidate) {
                return candidate.nodes == nodes;
            });
        if (explicit_region != explicit_libraries.end()) {
            put_region(
                out,
                nodes,
                true,
                explicit_region->library_op,
                explicit_region->inputs,
                explicit_region->outputs);
            continue;
        }
        std::uint8_t library_op = 0;
        const bool library =
            nodes.size() == 1 &&
            library_for_op(ops[nodes[0]], library_op);
        put_region(out, nodes, library, library_op);
    }
}

inline std::vector<std::uint8_t> make_plan(
    const pie_native::ptir::container::Container& container,
    const pie_native::ptir::bound::StageTypes& types,
    const pie_native::ptir::container::CStage& stage,
    const std::vector<ExplicitLibraryRegion>& explicit_libraries = {}) {
    std::vector<std::uint8_t> out{'P', 'T', 'R', 'P'};
    put_u16(out, PTIR_REGION_PLAN_VERSION);
    put_u16(out, PTIR_COMPILER_VERSION);
    out.push_back(stage.stage);
    const std::vector<std::uint8_t> signature_bytes =
        plan_signature_bytes(stage.stage, stage.ops, types);
    const std::uint64_t signature =
        pie_native::ptir::container::fnv1a64(
            signature_bytes.data(), signature_bytes.size());
    put_u64(out, signature);
    put_u32(out, static_cast<std::uint32_t>(signature_bytes.size()));
    out.insert(
        out.end(), signature_bytes.begin(), signature_bytes.end());
    put_u32(out, static_cast<std::uint32_t>(container.channels.size()));
    for (std::uint32_t channel = 0; channel < container.channels.size();
         ++channel) {
        put_u32(out, channel);
    }
    put_u32(out, static_cast<std::uint32_t>(container.names.size()));
    for (const std::string& name : container.names) {
        put_u16(out, static_cast<std::uint16_t>(name.size()));
        out.insert(out.end(), name.begin(), name.end());
    }
    put_u32(out, static_cast<std::uint32_t>(stage.ops.size()));
    for (std::size_t index = 0; index < stage.ops.size(); ++index) {
        const auto encoded = encode_op(stage.ops[index]);
        put_u32(out, static_cast<std::uint32_t>(encoded.size()));
        out.insert(out.end(), encoded.begin(), encoded.end());
        put_u32(out, 1);
        put_u32(out, static_cast<std::uint32_t>(index));
    }
    put_u32(out, static_cast<std::uint32_t>(types.value_types.size()));
    for (const auto& type : types.value_types) {
        out.push_back(static_cast<std::uint8_t>(type.dtype));
        out.push_back(static_cast<std::uint8_t>(type.shape.dims.size()));
        for (std::uint32_t dimension : type.shape.dims) {
            out.push_back(0);
            put_u32(out, dimension);
        }
        out.push_back(0);
    }
    put_partition(out, 0, stage.ops);
    put_partition(out, 1, stage.ops, explicit_libraries);
    return out;
}

inline std::vector<std::uint8_t> upgrade_ptib_v1(
    const std::vector<std::uint8_t>& container_bytes,
    std::vector<std::uint8_t> sidecar,
    const std::vector<std::vector<ExplicitLibraryRegion>>&
        explicit_libraries = {}) {
    pie_native::ptir::container::Container container;
    pie_native::ptir::container::DecodeError container_error;
    pie_native::ptir::bound::Bound bound;
    std::string sidecar_error;
    if (!pie_native::ptir::container::decode(
            container_bytes.data(),
            container_bytes.size(),
            container,
            &container_error) ||
        !pie_native::ptir::bound::parse_sidecar(
            sidecar.data(), sidecar.size(), bound, &sidecar_error) ||
        bound.version != 1 ||
        bound.stages.size() != container.stages.size()) {
        return {};
    }
    sidecar[4] = PTIB_VERSION;
    sidecar[5] = 0;
    put_u32(
        sidecar, static_cast<std::uint32_t>(container.stages.size()));
    for (std::size_t stage = 0; stage < container.stages.size(); ++stage) {
        const auto plan =
            make_plan(
                container,
                bound.stages[stage],
                container.stages[stage],
                stage < explicit_libraries.size()
                    ? explicit_libraries[stage]
                    : std::vector<ExplicitLibraryRegion>{});
        sidecar.push_back(container.stages[stage].stage);
        put_u32(sidecar, static_cast<std::uint32_t>(plan.size()));
        sidecar.insert(sidecar.end(), plan.begin(), plan.end());
    }
    return sidecar;
}

inline std::vector<std::uint8_t> empty_program_ptib_v2(
    const std::vector<std::uint8_t>& container_bytes) {
    pie_native::ptir::container::Container container;
    pie_native::ptir::container::DecodeError error;
    if (!pie_native::ptir::container::decode(
            container_bytes.data(),
            container_bytes.size(),
            container,
            &error) ||
        !container.stages.empty()) {
        return {};
    }
    std::vector<std::uint8_t> sidecar{'P', 'T', 'I', 'B'};
    put_u16(sidecar, PTIB_VERSION);
    put_u16(sidecar, 0);
    put_u64(sidecar, container.hash);
    put_u32(
        sidecar, static_cast<std::uint32_t>(container.channels.size()));
    for (std::size_t channel = 0; channel < container.channels.size();
         ++channel) {
        sidecar.push_back(PTIR_CHAN_FULL_RING);
    }
    put_u32(sidecar, 0);  // readiness
    put_u32(sidecar, 0);  // stages
    put_u32(sidecar, 0);  // plans
    return sidecar;
}

}  // namespace pie::metal::tests
