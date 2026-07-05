#pragma once

// PTIR bound-trace sidecar (`PTIB` v1) parser + container→Trace translation.
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
#include <string>
#include <vector>

#include "ptir/container.hpp"
#include "ptir/op_table.hpp"
#include "ptir/trace.hpp"

namespace pie_cuda_driver::ptir::bound {

struct StageTypes {
    std::uint8_t stage = 0;
    std::vector<TensorType> value_types;   // per SSA id, in order
};

struct Bound {
    std::uint64_t container_hash = 0;
    std::vector<std::uint8_t> classes;               // per channel (0/1/2)
    std::vector<container::ReadinessEntry> readiness;
    std::vector<StageTypes> stages;                  // container order
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

// Parse a PTIB v1 sidecar. Returns false on a structural/version/magic error.
inline bool parse_sidecar(const std::uint8_t* data, std::size_t len, Bound& out, std::string* err = nullptr) {
    auto fail = [&](const char* m) { if (err) *err = m; return false; };
    container::detail::Cur c{data, len};
    if (len < 8) return fail("short sidecar");
    if (std::memcmp(data, PTIB_MAGIC, 4) != 0) return fail("bad PTIB magic");
    c.skip(4);
    std::uint16_t ver = c.u16(); c.u16();
    if (ver != PTIB_VERSION) return fail("bad PTIB version");
    out.container_hash = 0;
    for (int b = 0; b < 8; ++b) out.container_hash |= (std::uint64_t)c.u8() << (b * 8);
    std::uint32_t n_ch = c.u32();
    for (std::uint32_t i = 0; i < n_ch; ++i) out.classes.push_back(c.u8());
    std::uint32_t n_rd = c.u32();
    for (std::uint32_t i = 0; i < n_rd; ++i) {
        container::ReadinessEntry e;
        e.chan = c.u32(); e.phase = c.u8();
        e.dir = c.u8() ? container::Direction::NeedsEmpty : container::Direction::NeedsFull;
        out.readiness.push_back(e);
    }
    std::uint32_t n_st = c.u32();
    for (std::uint32_t i = 0; i < n_st; ++i) {
        StageTypes st;
        st.stage = c.u8();
        std::uint32_t nv = c.u32();
        for (std::uint32_t v = 0; v < nv; ++v) {
            TensorType t;
            t.dtype = from_wire_dtype(c.u8());
            t.shape = from_cshape(c.shape());
            st.value_types.push_back(t);
        }
        out.stages.push_back(std::move(st));
    }
    if (c.err) return fail("sidecar overrun");
    return true;
}

// ── container + sidecar → executable Trace ──

inline Intrinsic map_intrinsic(std::uint16_t intr) {
    switch (intr) {
        case PTIR_INTR_LOGITS: return Intrinsic::Logits;
        case PTIR_INTR_MTP_LOGITS: return Intrinsic::MtpLogits;
        case PTIR_INTR_HIDDEN: return Intrinsic::Hidden;
        case PTIR_INTR_QUERY: return Intrinsic::Query;
        case PTIR_INTR_VALUE_HEAD: return Intrinsic::ValueHead;
        default: return Intrinsic::Logits;
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
    for (const container::CPort& p : c.ports)
        t.ports.push_back({p.port, p.chan, p.is_const});

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
                    v.intrinsic = map_intrinsic(op.intr);
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
                    o.rng_kind = op.kind ? RngKind::Gumbel : RngKind::Uniform;
                    o.predicate.tag = (PredTag)op.pred_tag;
                    o.predicate.payload = op.pred_payload;
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

}  // namespace pie_cuda_driver::ptir::bound
