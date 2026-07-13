#pragma once

#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include "pie_native/ptir/container.hpp"
#include "pie_native/ptir/ptir_abi.h"

namespace pie_native::ptir::plan {

struct Dimension {
    bool symbolic = false;
    std::uint32_t value = 0;
};

struct ValueType {
    std::uint8_t dtype = 0;
    std::vector<Dimension> dims;
    std::uint8_t domain = 0;
};

struct NormalizedOp {
    container::COp op;
    std::vector<std::uint32_t> source_ops;
};

struct ChannelSink {
    std::uint32_t channel_slot = 0;
    std::uint32_t value = 0;
};

struct Region {
    bool library = false;
    std::uint8_t library_op = 0;
    std::uint8_t schedule = 0;
    std::vector<std::uint32_t> nodes;
    // Library inputs are role ordered. Nucleus is [logits, top_p, rng_state],
    // independent of numeric value-id order.
    std::vector<std::uint32_t> inputs;
    std::vector<std::uint32_t> outputs;
    std::vector<ChannelSink> sinks;
};

struct Partition {
    std::uint8_t kind = 0;
    bool whole_stage_fallback = false;
    std::vector<Region> regions;
};

struct StagePlan {
    std::uint8_t stage = 0;
    std::uint64_t signature_hash = 0;
    std::vector<std::uint8_t> signature;
    std::vector<std::uint32_t> channel_bindings;
    std::vector<std::string> names;
    std::vector<NormalizedOp> ops;
    std::vector<ValueType> value_types;
    Partition singleton;
    Partition fused;
};

namespace detail {

struct Reader {
    const std::uint8_t* data = nullptr;
    std::size_t size = 0;
    std::size_t offset = 0;

    std::size_t remaining() const {
        return offset <= size ? size - offset : 0;
    }

    bool bytes(std::size_t count, const std::uint8_t*& value) {
        if (count > remaining()) return false;
        value = data + offset;
        offset += count;
        return true;
    }

    bool u8(std::uint8_t& value) {
        const std::uint8_t* bytes = nullptr;
        if (!this->bytes(1, bytes)) return false;
        value = bytes[0];
        return true;
    }

    bool u16(std::uint16_t& value) {
        const std::uint8_t* bytes = nullptr;
        if (!this->bytes(2, bytes)) return false;
        value = static_cast<std::uint16_t>(bytes[0]) |
            (static_cast<std::uint16_t>(bytes[1]) << 8);
        return true;
    }

    bool u32(std::uint32_t& value) {
        const std::uint8_t* bytes = nullptr;
        if (!this->bytes(4, bytes)) return false;
        value = static_cast<std::uint32_t>(bytes[0]) |
            (static_cast<std::uint32_t>(bytes[1]) << 8) |
            (static_cast<std::uint32_t>(bytes[2]) << 16) |
            (static_cast<std::uint32_t>(bytes[3]) << 24);
        return true;
    }

    bool u64(std::uint64_t& value) {
        const std::uint8_t* bytes = nullptr;
        if (!this->bytes(8, bytes)) return false;
        value = 0;
        for (int byte = 0; byte < 8; ++byte) {
            value |= static_cast<std::uint64_t>(bytes[byte]) << (byte * 8);
        }
        return true;
    }

    bool bounded_count(
        std::uint32_t raw_count,
        std::size_t minimum_record_bytes,
        std::size_t structural_maximum,
        std::size_t& count) const {
        if (static_cast<std::uintmax_t>(raw_count) >
            static_cast<std::uintmax_t>(
                std::numeric_limits<std::size_t>::max())) {
            return false;
        }
        count = static_cast<std::size_t>(raw_count);
        if (minimum_record_bytes == 0 || count > structural_maximum ||
            count > std::numeric_limits<std::size_t>::max() /
                minimum_record_bytes) {
            return false;
        }
        return count * minimum_record_bytes <= remaining();
    }

    bool length_with_tail(
        std::uint32_t raw_length,
        std::size_t required_tail,
        std::size_t& length) const {
        if (static_cast<std::uintmax_t>(raw_length) >
            static_cast<std::uintmax_t>(
                std::numeric_limits<std::size_t>::max())) {
            return false;
        }
        length = static_cast<std::size_t>(raw_length);
        return required_tail <= remaining() &&
            length <= remaining() - required_tail;
    }
};

inline bool scan_shape(Reader& reader) {
    std::uint8_t raw_rank = 0;
    std::size_t rank = 0;
    if (!reader.u8(raw_rank) ||
        !reader.bounded_count(raw_rank, 4, 4, rank)) {
        return false;
    }
    const std::uint8_t* dimensions = nullptr;
    return reader.bytes(rank * 4, dimensions);
}

inline bool scan_planned_op(
    const std::uint8_t* data,
    std::size_t length,
    container::COp* output,
    std::uint32_t& result_count) {
    Reader reader{data, length};
    std::uint8_t tag = 0;
    if (!reader.u8(tag)) return false;
    result_count = 1;
    const std::uint8_t* ignored = nullptr;
    auto fixed = [&](std::size_t count) {
        return reader.bytes(count, ignored);
    };
    switch (tag) {
        case PTIR_OP_EXP: case PTIR_OP_LOG: case PTIR_OP_NEG:
        case PTIR_OP_RECIP: case PTIR_OP_ABS: case PTIR_OP_SIGN:
        case PTIR_OP_NOT: case PTIR_OP_REDUCE_SUM:
        case PTIR_OP_REDUCE_MAX: case PTIR_OP_REDUCE_MIN:
        case PTIR_OP_REDUCE_ARGMAX: case PTIR_OP_TRANSPOSE:
        case PTIR_OP_CUMSUM: case PTIR_OP_CUMPROD:
        case PTIR_OP_IOTA: case PTIR_OP_CHAN_TAKE:
        case PTIR_OP_CHAN_READ:
            if (!fixed(4)) return false;
            break;
        case PTIR_OP_SORT_DESC:
            if (!fixed(4)) return false;
            result_count = 2;
            break;
        case PTIR_OP_CAST: {
            std::uint8_t dtype = 0;
            if (!fixed(4) || !reader.u8(dtype) || dtype > PTIR_DT_BOOL) {
                return false;
            }
            break;
        }
        case PTIR_OP_ADD: case PTIR_OP_SUB: case PTIR_OP_MUL:
        case PTIR_OP_DIV: case PTIR_OP_MAX_ELEM: case PTIR_OP_MIN_ELEM:
        case PTIR_OP_GT: case PTIR_OP_GE: case PTIR_OP_EQ:
        case PTIR_OP_NE: case PTIR_OP_LT: case PTIR_OP_LE:
        case PTIR_OP_AND: case PTIR_OP_OR: case PTIR_OP_REM:
        case PTIR_OP_MATMUL: case PTIR_OP_GATHER:
        case PTIR_OP_GATHER_ROW: case PTIR_OP_MASK_APPLY_PACKED:
        case PTIR_OP_CAUSAL_MASK:
            if (!fixed(8)) return false;
            break;
        case PTIR_OP_TOP_K:
            if (!fixed(8)) return false;
            result_count = 2;
            break;
        case PTIR_OP_SELECT: case PTIR_OP_SCATTER_ADD:
        case PTIR_OP_SCATTER_SET: case PTIR_OP_SLIDING_WINDOW_MASK:
            if (!fixed(12)) return false;
            break;
        case PTIR_OP_SINK_WINDOW_MASK:
            if (!fixed(16)) return false;
            break;
        case PTIR_OP_BROADCAST: case PTIR_OP_RESHAPE:
            if (!fixed(4) || !scan_shape(reader)) return false;
            break;
        case PTIR_OP_PIVOT_THRESHOLD: {
            std::uint8_t predicate = 0;
            if (!fixed(4) || !reader.u8(predicate) || predicate > 2 ||
                !fixed(4)) {
                return false;
            }
            break;
        }
        case PTIR_OP_RNG: case PTIR_OP_RNG_KEYED: {
            std::uint8_t kind = 0;
            if (!fixed(4) || !scan_shape(reader) || !reader.u8(kind) ||
                kind > 1) {
                return false;
            }
            break;
        }
        case PTIR_OP_CONST: {
            std::uint8_t dtype = 0;
            if (!reader.u8(dtype) || dtype > PTIR_DT_BOOL || !fixed(4)) {
                return false;
            }
            break;
        }
        case PTIR_OP_CHAN_PUT:
            if (!fixed(8)) return false;
            result_count = 0;
            break;
        case PTIR_OP_INTRINSIC_VAL: {
            std::uint16_t intrinsic = 0;
            std::uint8_t dtype = 0;
            if (!reader.u16(intrinsic) || intrinsic > PTIR_INTR_MTP_DRAFTS ||
                !reader.u8(dtype) || dtype > PTIR_DT_BOOL ||
                !scan_shape(reader)) {
                return false;
            }
            break;
        }
        case PTIR_OP_KERNEL_CALL: {
            std::uint16_t name = 0;
            std::uint8_t dtype = 0;
            std::uint8_t raw_arguments = 0;
            std::size_t argument_count = 0;
            if (!reader.u16(name) || !reader.u8(dtype) ||
                dtype > PTIR_DT_BOOL || !scan_shape(reader) ||
                !reader.u8(raw_arguments) ||
                !reader.bounded_count(
                    raw_arguments,
                    4,
                    std::numeric_limits<std::uint8_t>::max(),
                    argument_count) ||
                !fixed(argument_count * 4)) {
                return false;
            }
            break;
        }
        case PTIR_OP_SINK_CALL: {
            std::uint16_t name = 0;
            std::uint8_t raw_arguments = 0;
            std::size_t argument_count = 0;
            if (!reader.u16(name) || !reader.u8(raw_arguments) ||
                !reader.bounded_count(
                    raw_arguments,
                    4,
                    std::numeric_limits<std::uint8_t>::max(),
                    argument_count) ||
                !fixed(argument_count * 4)) {
                return false;
            }
            result_count = 0;
            break;
        }
        default:
            return false;
    }
    if (reader.offset != length) return false;
    if (output != nullptr) {
        container::detail::Cur cursor{data, length};
        container::COp decoded;
        container::detail::decode_op(cursor, decoded);
        if (cursor.err || cursor.i != length ||
            decoded.results != result_count) {
            return false;
        }
        *output = std::move(decoded);
    }
    return true;
}

inline bool read_index_vector(
    Reader& reader,
    std::size_t structural_maximum,
    std::size_t upper_bound,
    bool ordered,
    std::vector<std::uint32_t>* output,
    std::size_t& count) {
    std::uint32_t raw_count = 0;
    if (!reader.u32(raw_count) ||
        !reader.bounded_count(
            raw_count, 4, structural_maximum, count)) {
        return false;
    }
    if (output != nullptr) output->reserve(count);
    bool have_previous = false;
    std::uint32_t previous = 0;
    for (std::size_t index = 0; index < count; ++index) {
        std::uint32_t value = 0;
        if (!reader.u32(value) ||
            static_cast<std::uintmax_t>(value) >=
                static_cast<std::uintmax_t>(upper_bound) ||
            (ordered && have_previous && previous >= value)) {
            return false;
        }
        have_previous = true;
        previous = value;
        if (output != nullptr) output->push_back(value);
    }
    return true;
}

inline bool read_partition(
    Reader& reader,
    std::uint8_t expected_kind,
    std::size_t operation_count,
    std::size_t value_count,
    std::size_t channel_count,
    Partition* output,
    std::uint32_t& raw_region_count) {
    std::uint8_t kind = 0;
    std::uint8_t fallback = 0;
    if (!reader.u8(kind) || kind != expected_kind ||
        !reader.u8(fallback) || fallback > 1 ||
        !reader.u32(raw_region_count)) {
        return false;
    }
    std::size_t region_count = 0;
    if (!reader.bounded_count(
            raw_region_count, 19, operation_count, region_count)) {
        return false;
    }
    if (output != nullptr) {
        output->kind = kind;
        output->whole_stage_fallback = fallback != 0;
        output->regions.reserve(region_count);
    }
    for (std::size_t region_index = 0; region_index < region_count;
         ++region_index) {
        Region region;
        std::uint8_t region_kind = 0;
        if (!reader.u8(region_kind) || region_kind > 1 ||
            !reader.u8(region.library_op) ||
            (region_kind == 1 &&
             region.library_op > PTIR_LIBRARY_SECOND_PARTY) ||
            !reader.u8(region.schedule) ||
            region.schedule > PTIR_SCHEDULE_LIBRARY) {
            return false;
        }
        region.library = region_kind == 1;
        std::size_t node_count = 0;
        std::size_t input_count = 0;
        std::size_t output_count = 0;
        if (!read_index_vector(
                reader,
                operation_count,
                operation_count,
                true,
                output != nullptr ? &region.nodes : nullptr,
                node_count) ||
            !read_index_vector(
                reader,
                value_count,
                value_count,
                false,
                output != nullptr ? &region.inputs : nullptr,
                input_count) ||
            !read_index_vector(
                reader,
                value_count,
                value_count,
                false,
                output != nullptr ? &region.outputs : nullptr,
                output_count)) {
            return false;
        }
        std::uint32_t raw_sinks = 0;
        std::size_t sink_count = 0;
        if (!reader.u32(raw_sinks) ||
            !reader.bounded_count(raw_sinks, 8, node_count, sink_count)) {
            return false;
        }
        if (output != nullptr) region.sinks.reserve(sink_count);
        for (std::size_t sink_index = 0; sink_index < sink_count;
             ++sink_index) {
            ChannelSink sink;
            if (!reader.u32(sink.channel_slot) ||
                !reader.u32(sink.value) ||
                static_cast<std::uintmax_t>(sink.channel_slot) >=
                    static_cast<std::uintmax_t>(channel_count) ||
                static_cast<std::uintmax_t>(sink.value) >=
                    static_cast<std::uintmax_t>(value_count)) {
                return false;
            }
            if (output != nullptr) region.sinks.push_back(sink);
        }
        if (region.library &&
            region.library_op == PTIR_LIBRARY_NUCLEUS_SAMPLE &&
            (node_count != 13 || input_count != 3 ||
             output_count != 1 || sink_count != 0)) {
            return false;
        }
        if (output != nullptr) {
            output->regions.push_back(std::move(region));
        }
    }
    return true;
}

inline bool parse_records(
    const std::uint8_t* data,
    std::size_t len,
    StagePlan* output,
    const char*& error) {
    auto fail = [&](const char* message) {
        error = message;
        return false;
    };
    if (data == nullptr || len < 4 || std::memcmp(data, "PTRP", 4) != 0) {
        return fail("invalid region-plan header");
    }
    Reader reader{data, len};
    const std::uint8_t* ignored = nullptr;
    std::uint16_t plan_version = 0;
    std::uint16_t compiler_version = 0;
    std::uint8_t stage = 0;
    std::uint64_t signature_hash = 0;
    if (!reader.bytes(4, ignored) || !reader.u16(plan_version) ||
        !reader.u16(compiler_version)) {
        return fail("invalid region-plan header");
    }
    if (plan_version != PTIR_REGION_PLAN_VERSION ||
        compiler_version != PTIR_COMPILER_VERSION) {
        return fail("unsupported region-plan version");
    }
    if (!reader.u8(stage) || stage > PTIR_STAGE_EPILOGUE ||
        !reader.u64(signature_hash)) {
        return fail("invalid plan stage");
    }

    std::uint32_t raw_signature_length = 0;
    std::size_t signature_length = 0;
    const std::uint8_t* signature = nullptr;
    if (!reader.u32(raw_signature_length) ||
        !reader.length_with_tail(
            raw_signature_length, 0, signature_length) ||
        !reader.bytes(signature_length, signature)) {
        return fail("signature overrun");
    }
    if (container::fnv1a64(signature, signature_length) != signature_hash) {
        return fail("stage signature hash mismatch");
    }
    if (output != nullptr) {
        output->stage = stage;
        output->signature_hash = signature_hash;
        output->signature.assign(signature, signature + signature_length);
    }

    std::uint32_t raw_channels = 0;
    std::size_t channel_count = 0;
    if (!reader.u32(raw_channels) ||
        !reader.bounded_count(
            raw_channels,
            4,
            std::numeric_limits<std::size_t>::max(),
            channel_count)) {
        return fail("plan channel count exceeds remaining bytes");
    }
    if (output != nullptr) output->channel_bindings.reserve(channel_count);
    for (std::size_t index = 0; index < channel_count; ++index) {
        std::uint32_t channel = 0;
        if (!reader.u32(channel)) return fail("plan channel overrun");
        if (output != nullptr) output->channel_bindings.push_back(channel);
    }

    std::uint32_t raw_names = 0;
    std::size_t name_count = 0;
    constexpr std::uintmax_t kNameIndexSpace =
        static_cast<std::uintmax_t>(
            std::numeric_limits<std::uint16_t>::max()) + 1;
    const std::size_t maximum_names =
        kNameIndexSpace > static_cast<std::uintmax_t>(
            std::numeric_limits<std::size_t>::max())
        ? std::numeric_limits<std::size_t>::max()
        : static_cast<std::size_t>(kNameIndexSpace);
    if (!reader.u32(raw_names) ||
        !reader.bounded_count(raw_names, 2, maximum_names, name_count)) {
        return fail("plan name count exceeds structural limit");
    }
    if (output != nullptr) output->names.reserve(name_count);
    for (std::size_t index = 0; index < name_count; ++index) {
        std::uint16_t raw_name_length = 0;
        const std::uint8_t* name = nullptr;
        if (!reader.u16(raw_name_length) ||
            !reader.bytes(raw_name_length, name)) {
            return fail("plan name overrun");
        }
        if (output != nullptr) {
            output->names.emplace_back(
                reinterpret_cast<const char*>(name), raw_name_length);
        }
    }

    std::uint32_t raw_operations = 0;
    std::size_t operation_count = 0;
    if (!reader.u32(raw_operations) ||
        !reader.bounded_count(
            raw_operations,
            12,
            std::numeric_limits<std::size_t>::max(),
            operation_count)) {
        return fail("normalized operation count exceeds remaining bytes");
    }
    if (output != nullptr) output->ops.reserve(operation_count);
    std::uint32_t result_count = 0;
    for (std::size_t index = 0; index < operation_count; ++index) {
        std::uint32_t raw_op_length = 0;
        std::size_t op_length = 0;
        const std::uint8_t* op_bytes = nullptr;
        NormalizedOp normalized;
        std::uint32_t op_results = 0;
        if (!reader.u32(raw_op_length) ||
            !reader.length_with_tail(raw_op_length, 4, op_length) ||
            !reader.bytes(op_length, op_bytes) ||
            !scan_planned_op(
                op_bytes,
                op_length,
                output != nullptr ? &normalized.op : nullptr,
                op_results) ||
            result_count >
                std::numeric_limits<std::uint32_t>::max() - op_results) {
            return fail("invalid normalized operation");
        }
        result_count += op_results;
        std::uint32_t raw_sources = 0;
        std::size_t source_count = 0;
        if (!reader.u32(raw_sources) ||
            !reader.bounded_count(
                raw_sources,
                4,
                std::numeric_limits<std::size_t>::max(),
                source_count)) {
            return fail("operation source count exceeds remaining bytes");
        }
        if (output != nullptr) normalized.source_ops.reserve(source_count);
        for (std::size_t source = 0; source < source_count; ++source) {
            std::uint32_t source_op = 0;
            if (!reader.u32(source_op)) {
                return fail("operation source map overrun");
            }
            if (output != nullptr) {
                normalized.source_ops.push_back(source_op);
            }
        }
        if (output != nullptr) output->ops.push_back(std::move(normalized));
    }

    std::uint32_t raw_values = 0;
    std::size_t value_count = 0;
    if (!reader.u32(raw_values) ||
        !reader.bounded_count(
            raw_values,
            3,
            static_cast<std::size_t>(result_count),
            value_count) ||
        value_count != static_cast<std::size_t>(result_count)) {
        return fail("plan value count exceeds structural limit");
    }
    if (output != nullptr) output->value_types.reserve(value_count);
    for (std::size_t index = 0; index < value_count; ++index) {
        ValueType value_type;
        std::uint8_t raw_rank = 0;
        if (!reader.u8(value_type.dtype) ||
            value_type.dtype > PTIR_DT_BOOL ||
            !reader.u8(raw_rank)) {
            return fail("invalid symbolic value type");
        }
        std::size_t rank = 0;
        if (!reader.bounded_count(raw_rank, 2, 4, rank)) {
            return fail("symbolic type dimensions overrun");
        }
        if (output != nullptr) value_type.dims.reserve(rank);
        for (std::size_t dimension_index = 0;
             dimension_index < rank;
             ++dimension_index) {
            Dimension dimension;
            std::uint8_t tag = 0;
            if (!reader.u8(tag)) return fail("symbolic type overrun");
            if (tag == 0) {
                if (!reader.u32(dimension.value)) {
                    return fail("symbolic type overrun");
                }
            } else if (tag == 1) {
                dimension.symbolic = true;
                std::uint8_t symbolic = 0;
                if (!reader.u8(symbolic) ||
                    symbolic > PTIR_EXTENT_KEY_LEN) {
                    return fail("invalid symbolic extent");
                }
                dimension.value = symbolic;
            } else {
                return fail("invalid dimension tag");
            }
            if (output != nullptr) {
                value_type.dims.push_back(dimension);
            }
        }
        if (!reader.u8(value_type.domain) || value_type.domain > 7) {
            return fail("invalid value domain");
        }
        if (output != nullptr) {
            output->value_types.push_back(std::move(value_type));
        }
    }

    std::uint32_t singleton_regions = 0;
    std::uint32_t fused_regions = 0;
    if (!read_partition(
            reader,
            0,
            operation_count,
            value_count,
            channel_count,
            output != nullptr ? &output->singleton : nullptr,
            singleton_regions) ||
        !read_partition(
            reader,
            1,
            operation_count,
            value_count,
            channel_count,
            output != nullptr ? &output->fused : nullptr,
            fused_regions) ||
        reader.offset != len) {
        return fail("invalid region partition");
    }
    return true;
}

}  // namespace detail

inline bool decode(
    const std::uint8_t* data,
    std::size_t len,
    StagePlan& out,
    std::string* error = nullptr) {
    out = {};
    const char* message = "invalid region plan";
    if (!detail::parse_records(data, len, nullptr, message)) {
        if (error != nullptr) *error = message;
        return false;
    }
    StagePlan decoded;
    if (!detail::parse_records(data, len, &decoded, message)) {
        if (error != nullptr) *error = message;
        return false;
    }
    out = std::move(decoded);
    return true;
}

}  // namespace pie_native::ptir::plan
