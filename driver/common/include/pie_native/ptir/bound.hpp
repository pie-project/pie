#pragma once

// PTIR bound-trace sidecar (`PTIB` v2) parser + container→Trace translation.
//
// Per the Option-B ruling, the driver does NOT re-infer shapes/dtypes: echo's
// `bind()` ships a `PTIB` typed sidecar (PTIR-CONTAINER.md §7) carrying, per
// stage (container order), the (dtype, shape) of each SSA value id, plus the
// channel classes + readiness table. This header parses that blob and folds it
// onto the structural Container (container.hpp) to build an executable Trace
// (trace.hpp) the tier-0 runner / tier-1 codegen consume.
//
// The SSA space is flat PER STAGE (echo: op at position p defines
// next_id..next_id+results). We namespace it into the Trace's global value table
// with a per-stage base offset.
//
// Header-only host C++.

#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

#include "pie_native/ptir/container.hpp"
#include "pie_native/ptir/op_table.hpp"
#include "pie_native/ptir/trace.hpp"

namespace pie_native::ptir::bound {

struct StageTypes {
    std::uint8_t stage = 0;
    std::vector<TensorType> value_types;   // per SSA id, in order
};

struct StagePlan {
    std::uint8_t stage = 0;
    std::vector<std::uint8_t> bytes;
};

struct Bound {
    std::uint16_t version = 0;
    std::uint64_t container_hash = 0;
    std::vector<std::uint8_t> classes;               // per channel (0/1/2)
    std::vector<container::ReadinessEntry> readiness;
    std::vector<StageTypes> stages;                  // container order
    std::vector<StagePlan> plans;                    // compiler-owned region plans
};

inline DType from_wire_dtype(std::uint8_t d) {
    switch (d) {
        case PTIR_DT_F32: return DType::F32;
        case PTIR_DT_I32: return DType::I32;
        case PTIR_DT_U32: return DType::U32;
        case PTIR_DT_BOOL: return DType::Bool;
        case PTIR_DT_ACT: return DType::F32;   // ACT materialized to F32 program-side
    }
    return DType::F32;
}

inline Shape from_cshape(const container::CShape& s) {
    Shape out;
    for (int i = 0; i < s.rank; ++i) out.dims.push_back(s.dims[i]);
    return out;
}

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
        const std::size_t minimum_bytes = count * minimum_record_bytes;
        return minimum_bytes <= remaining();
    }

    bool length(std::uint32_t raw_length, std::size_t& length) const {
        if (static_cast<std::uintmax_t>(raw_length) >
            static_cast<std::uintmax_t>(
                std::numeric_limits<std::size_t>::max())) {
            return false;
        }
        length = static_cast<std::size_t>(raw_length);
        return length <= remaining();
    }
};

inline bool parse_sidecar_records(
    const std::uint8_t* data,
    std::size_t len,
    Bound* output,
    const char*& error) {
    constexpr std::size_t kMaximumStages = 4;
    auto fail = [&](const char* message) {
        error = message;
        return false;
    };
    if (data == nullptr || len < 4 || std::memcmp(data, PTIB_MAGIC, 4) != 0) {
        return fail("bad PTIB magic");
    }

    Reader reader{data, len};
    const std::uint8_t* magic = nullptr;
    std::uint16_t version = 0;
    std::uint16_t flags = 0;
    std::uint64_t container_hash = 0;
    if (!reader.bytes(4, magic) || !reader.u16(version) ||
        !reader.u16(flags) || !reader.u64(container_hash)) {
        return fail("short sidecar");
    }
    (void)flags;
    if (version != 1 && version != PTIB_VERSION) {
        return fail("bad PTIB version");
    }
    if (output != nullptr) {
        output->version = version;
        output->container_hash = container_hash;
    }

    std::uint32_t raw_channels = 0;
    std::size_t channel_count = 0;
    if (!reader.u32(raw_channels) ||
        !reader.bounded_count(
            raw_channels,
            1,
            std::numeric_limits<std::size_t>::max(),
            channel_count)) {
        return fail("sidecar channel count exceeds remaining bytes");
    }
    if (output != nullptr) output->classes.reserve(channel_count);
    for (std::size_t index = 0; index < channel_count; ++index) {
        std::uint8_t channel_class = 0;
        if (!reader.u8(channel_class) || channel_class > PTIR_CHAN_IN_PLACE_UNDO) {
            return fail("invalid sidecar channel class");
        }
        if (output != nullptr) output->classes.push_back(channel_class);
    }

    std::uint32_t raw_readiness = 0;
    std::size_t readiness_count = 0;
    if (!reader.u32(raw_readiness) ||
        !reader.bounded_count(
            raw_readiness, 6, channel_count, readiness_count)) {
        return fail("sidecar readiness count exceeds structural limit");
    }
    if (output != nullptr) output->readiness.reserve(readiness_count);
    for (std::size_t index = 0; index < readiness_count; ++index) {
        std::uint32_t channel = 0;
        std::uint8_t phase = 0;
        std::uint8_t direction = 0;
        if (!reader.u32(channel) || !reader.u8(phase) ||
            !reader.u8(direction)) {
            return fail("sidecar readiness overrun");
        }
        if ((phase > PTIR_STAGE_EPILOGUE &&
             phase != PTIR_PHASE_DESCRIPTOR) ||
            direction > PTIR_NEEDS_EMPTY) {
            return fail("invalid sidecar readiness entry");
        }
        if (output != nullptr) {
            output->readiness.push_back({
                channel,
                phase,
                direction == PTIR_NEEDS_EMPTY
                    ? container::Direction::NeedsEmpty
                    : container::Direction::NeedsFull,
            });
        }
    }

    std::uint32_t raw_stages = 0;
    std::size_t stage_count = 0;
    if (!reader.u32(raw_stages) ||
        !reader.bounded_count(
            raw_stages, 5, kMaximumStages, stage_count)) {
        return fail("sidecar stage count exceeds structural limit");
    }
    if (output != nullptr) output->stages.reserve(stage_count);
    for (std::size_t stage_index = 0; stage_index < stage_count;
         ++stage_index) {
        std::uint8_t stage_tag = 0;
        std::uint32_t raw_values = 0;
        if (!reader.u8(stage_tag) || stage_tag > PTIR_STAGE_EPILOGUE ||
            !reader.u32(raw_values)) {
            return fail("invalid sidecar stage");
        }
        std::size_t value_count = 0;
        if (!reader.bounded_count(
                raw_values,
                2,
                std::numeric_limits<std::size_t>::max(),
                value_count)) {
            return fail("sidecar value count exceeds remaining bytes");
        }
        StageTypes stage;
        stage.stage = stage_tag;
        if (output != nullptr) stage.value_types.reserve(value_count);
        for (std::size_t value_index = 0; value_index < value_count;
             ++value_index) {
            std::uint8_t dtype = 0;
            std::uint8_t raw_rank = 0;
            if (!reader.u8(dtype) || dtype > PTIR_DT_BOOL ||
                !reader.u8(raw_rank) || raw_rank > 4) {
                return fail("invalid sidecar value type");
            }
            std::size_t rank = 0;
            if (!reader.bounded_count(raw_rank, 4, 4, rank)) {
                return fail("sidecar shape dimensions overrun");
            }
            TensorType type;
            type.dtype = from_wire_dtype(dtype);
            if (output != nullptr) type.shape.dims.reserve(rank);
            std::uint64_t elements = 1;
            for (std::size_t dimension_index = 0;
                 dimension_index < rank;
                 ++dimension_index) {
                std::uint32_t dimension = 0;
                if (!reader.u32(dimension) || dimension == 0 ||
                    elements > std::numeric_limits<std::uint64_t>::max() /
                        dimension) {
                    return fail("invalid sidecar shape");
                }
                elements *= dimension;
                if (output != nullptr) {
                    type.shape.dims.push_back(dimension);
                }
            }
            if (output != nullptr) {
                stage.value_types.push_back(std::move(type));
            }
        }
        if (output != nullptr) output->stages.push_back(std::move(stage));
    }

    if (version == PTIB_VERSION) {
        std::uint32_t raw_plans = 0;
        std::size_t plan_count = 0;
        if (!reader.u32(raw_plans) ||
            !reader.bounded_count(
                raw_plans, 5, kMaximumStages, plan_count)) {
            return fail("sidecar plan count exceeds structural limit");
        }
        if (output != nullptr) output->plans.reserve(plan_count);
        for (std::size_t plan_index = 0; plan_index < plan_count;
             ++plan_index) {
            std::uint8_t stage = 0;
            std::uint32_t raw_length = 0;
            std::size_t length = 0;
            if (!reader.u8(stage) || stage > PTIR_STAGE_EPILOGUE ||
                !reader.u32(raw_length) ||
                !reader.length(raw_length, length)) {
                return fail("sidecar plan overrun");
            }
            const std::uint8_t* plan_bytes = nullptr;
            if (!reader.bytes(length, plan_bytes)) {
                return fail("sidecar plan overrun");
            }
            if (output != nullptr) {
                StagePlan plan;
                plan.stage = stage;
                plan.bytes.assign(plan_bytes, plan_bytes + length);
                output->plans.push_back(std::move(plan));
            }
        }
    }
    if (reader.offset != len) return fail("trailing sidecar bytes");
    return true;
}

}  // namespace detail

// Parse a PTIB v1/v2 sidecar. v1 has no compiler plans. Malformed input is
// fully preflighted without allocation before any output record is materialized.
inline bool parse_sidecar(
    const std::uint8_t* data,
    std::size_t len,
    Bound& out,
    std::string* err = nullptr) {
    out = {};
    const char* message = "invalid sidecar";
    if (!detail::parse_sidecar_records(data, len, nullptr, message)) {
        if (err != nullptr) *err = message;
        return false;
    }
    Bound decoded;
    if (!detail::parse_sidecar_records(data, len, &decoded, message)) {
        if (err != nullptr) *err = message;
        return false;
    }
    out = std::move(decoded);
    return true;
}

// ── container + sidecar → executable Trace ──

inline bool map_intrinsic(std::uint16_t intr, Intrinsic& out) {
    switch (intr) {
        case PTIR_INTR_LOGITS: out = Intrinsic::Logits; return true;
        case PTIR_INTR_MTP_LOGITS: out = Intrinsic::MtpLogits; return true;
        case PTIR_INTR_HIDDEN: out = Intrinsic::Hidden; return true;
        case PTIR_INTR_QUERY: out = Intrinsic::Query; return true;
        case PTIR_INTR_VALUE_HEAD: out = Intrinsic::ValueHead; return true;
        case PTIR_INTR_LAYER: out = Intrinsic::Layer; return true;
        case PTIR_INTR_MTP_DRAFTS: out = Intrinsic::MtpDrafts; return true;
        default: return false;
    }
}

struct TranslateResult {
    bool ok = false;
    std::string error;
    Trace trace;
};

// Fold the structural container + its PTIB types into an executable Trace.
inline TranslateResult container_to_trace(const container::Container& c, const Bound& b) {
    TranslateResult r;
    auto fail = [&](const std::string& m) { r.ok = false; r.error = m; return r; };
    if (c.hash != b.container_hash) return fail("sidecar container_hash mismatch");

    Trace& t = r.trace;
    for (const container::CChannel& ch : c.channels) {
        Channel out;
        out.id = (ChannelId)t.channels.size();
        out.type.dtype = from_wire_dtype(ch.dtype);
        out.type.shape = from_cshape(ch.shape);
        out.capacity = ch.capacity;
        out.has_seed = ch.seeded != 0;
        out.host_visible = ch.host_role != PTIR_HOST_NONE;
        out.host_reader = ch.host_role == PTIR_HOST_READER;
        out.extern_dir = ch.extern_dir;
        out.extern_name = ch.extern_name;
        t.channels.push_back(out);
    }
    for (const container::CPort& p : c.ports) {
        PortBinding binding;
        binding.port = p.port;
        binding.channel = p.chan;
        binding.is_const = p.is_const;
        if (p.is_const) {
            binding.const_type.dtype =
                from_wire_dtype(p.const_dtype);
            binding.const_type.shape =
                from_cshape(p.const_shape);
            binding.const_data = p.const_data;
        }
        t.ports.push_back(std::move(binding));
    }

    std::uint32_t global_base = 0;
    for (std::size_t si = 0; si < c.stages.size(); ++si) {
        const container::CStage& cs = c.stages[si];
        if (si >= b.stages.size()) return fail("sidecar missing a stage");
        const StageTypes& types = b.stages[si];
        auto ty = [&](std::uint32_t local) -> TensorType {
            return local < types.value_types.size() ? types.value_types[local] : TensorType{};
        };
        auto gid = [&](std::uint32_t local) { return global_base + local; };

        Stage stage;
        stage.kind = (StageKind)cs.stage;
        std::uint32_t local = 0;
        for (const container::COp& op : cs.ops) {
            OpCode code = (OpCode)op.tag;
            switch (op.tag) {
                case PTIR_OP_CHAN_TAKE: case PTIR_OP_CHAN_READ: {
                    Value v; v.id = gid(local); v.type = ty(local);
                    v.source = (op.tag == PTIR_OP_CHAN_TAKE) ? ValueSource::ChannelTake : ValueSource::ChannelRead;
                    v.channel = (ChannelId)op.chan;
                    t.values.push_back(v); local += 1;
                    if (op.tag == PTIR_OP_CHAN_TAKE) stage.takes.push_back((ChannelId)op.chan);
                    else stage.reads.push_back((ChannelId)op.chan);
                    break;
                }
                case PTIR_OP_CONST: {
                    Value v; v.id = gid(local); v.type = ty(local); v.source = ValueSource::Const;
                    v.lit.dtype = from_wire_dtype(op.lit_dtype); v.lit.bits = op.lit_bits;
                    t.values.push_back(v); local += 1; break;
                }
                case PTIR_OP_INTRINSIC_VAL: {
                    Value v; v.id = gid(local); v.type = ty(local); v.source = ValueSource::Intrinsic;
                    if (!map_intrinsic(op.intr, v.intrinsic)) {
                        return fail("unknown intrinsic tag in stage body");
                    }
                    t.values.push_back(v); local += 1; break;
                }
                case PTIR_OP_CHAN_PUT: {
                    stage.puts.push_back({(ChannelId)op.chan, gid(op.args[0])});
                    break;   // defines 0 ids
                }
                case PTIR_OP_SINK_CALL:
                    break;   // no result; tier-0 without kernels ignores the sink effect
                case PTIR_OP_KERNEL_CALL:
                    return fail("kernel_call needs a second-party kernel (not tier-0 executable)");
                default: {
                    // A compute op — its OpCode byte IS the tag (op_table mirrors ptir_abi).
                    if (!op_is_known(code)) return fail("unknown op tag in stage body");
                    Op o; o.code = code; o.result_type = ty(local); o.result_id = gid(local);
                    o.result_count = op.results;
                    for (std::uint32_t a : op.args) o.args.push_back(gid(a));
                    // op-specific immediates
                    o.imm = op.imm;
                    o.imm2 = op.imm2;
                    o.imm3 = op.imm3;
                    o.rng_kind = op.kind ? RngKind::Gumbel : RngKind::Uniform;
                    o.predicate.tag = (PredTag)op.pred_tag;
                    // Only PivotThreshold populates pred_tag/pred_payload
                    // (container.hpp decode); its payload is a stage-local
                    // ValueId on the wire (interface/ptir container.rs — all
                    // three PredTag variants carry a ValueId, RankLe included),
                    // so it must be remapped through gid() exactly like any
                    // other op operand — NOT treated as an immediate.
                    o.predicate.payload = gid(op.pred_payload);
                    stage.ops.push_back(o);
                    // define result value(s)
                    for (std::uint32_t rr = 0; rr < op.results; ++rr) {
                        Value v; v.id = gid(local + rr); v.type = ty(local + rr); v.source = ValueSource::OpResult;
                        t.values.push_back(v);
                    }
                    local += op.results;
                    break;
                }
            }
        }
        t.stages.push_back(std::move(stage));
        global_base += (std::uint32_t)types.value_types.size();
    }
    r.ok = true;
    return r;
}

}  // namespace pie_native::ptir::bound
