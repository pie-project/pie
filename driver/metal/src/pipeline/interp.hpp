#pragma once

// Metal's temporary production PTIR engine executes channel-plane programs on
// the CPU until Decision 7's generated singleton path passes its M1 gates.
// M0 keeps this interpreter as the behavior oracle while moving its channel
// authority into device-bindable Shared storage. This is a C++ mirror of the
// canonical interpreter
// (interface/ptir/src/interp.rs, the dummy driver's engine). Behavior is
// pinned to that oracle: same readiness rule, same register-semantics commit
// (take pops once, last put wins), same op semantics (argmax tie-break,
// sort_desc NaN order, left-aligned broadcast, splitmix64/hash_uniform RNG).
//
// Decoded program model comes from the shared pure-host PTIR headers under
// driver/common/include/pie_native/ptir (container.hpp / bound.hpp /
// trace.hpp / op_table.hpp — all CUDA-free, namespace pie_native::ptir). One
// divergence from interp.rs is accepted: the
// translated Trace separates stage puts from op order, so a take AFTER a put
// of the same channel WITHIN one stage resolves to the committed cell, not
// the pending put. Cross-stage put→take visibility is preserved (the pending
// overlay applies at stage granularity).
//
// Scope (direct_ffi_fix.md Phase 3 baseline; metal_ptir_plan.md Phase 1
// extends it): programs whose values are Const / ChannelTake / ChannelRead /
// OpResult / Intrinsic(Logits). MtpLogits remains decodable only for oracle
// tests and is rejected from production until bounded device storage exists.
// Host inputs, per-layer taps, and other model
// intrinsics (hidden/query/value-head) are not executable here — they need
// backend feature work this increment does not add.

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "pie_native/ptir/bound.hpp"
#include "pie_native/ptir/plan.hpp"
#include "pie_native/ptir/container.hpp"
#include "pie_native/ptir/op_table.hpp"
#include "pie_native/ptir/rng_contract.generated.h"
#include "pie_native/ptir/trace.hpp"
#include "pipeline/shared_storage.hpp"

namespace pie::metal::pipeline {

namespace cptir = pie_native::ptir;
using cptir::DType;
using cptir::OpCode;
using cptir::Trace;

// ─────────────────────────────── values ────────────────────────────────────

// One SSA value / channel cell: dtype tag + the matching lane vector (bool
// lanes are one byte each, 0/1).
struct Value {
    DType dtype = DType::F32;
    std::vector<float> f;
    std::vector<std::int32_t> i;
    std::vector<std::uint32_t> u;
    std::vector<std::uint8_t> b;

    std::size_t len() const {
        switch (dtype) {
            case DType::F32: return f.size();
            case DType::I32: return i.size();
            case DType::U32: return u.size();
            case DType::Bool: return b.size();
            case DType::Act: return f.size();
        }
        return 0;
    }

    static Value f32(std::vector<float> v) { Value x; x.dtype = DType::F32; x.f = std::move(v); return x; }
    static Value i32(std::vector<std::int32_t> v) { Value x; x.dtype = DType::I32; x.i = std::move(v); return x; }
    static Value u32(std::vector<std::uint32_t> v) { Value x; x.dtype = DType::U32; x.u = std::move(v); return x; }
    static Value boolean(std::vector<std::uint8_t> v) { Value x; x.dtype = DType::Bool; x.b = std::move(v); return x; }
};

inline Value zeros(DType dtype, std::size_t numel) {
    const std::size_t n = std::max<std::size_t>(numel, 1);
    switch (dtype) {
        case DType::I32: return Value::i32(std::vector<std::int32_t>(n, 0));
        case DType::U32: return Value::u32(std::vector<std::uint32_t>(n, 0));
        case DType::Bool: return Value::boolean(std::vector<std::uint8_t>(n, 0));
        default: return Value::f32(std::vector<float>(n, 0.0f));
    }
}

inline bool value_matches(const Value& v, const cptir::TensorType& ty) {
    const DType want = ty.dtype == DType::Act ? DType::F32 : ty.dtype;
    return v.dtype == want && v.len() == std::max<std::uint64_t>(ty.shape.numel(), 1);
}

inline std::vector<float> lanes_f32(const Value& v) {
    switch (v.dtype) {
        case DType::I32: {
            std::vector<float> o(v.i.size());
            for (std::size_t k = 0; k < o.size(); ++k) o[k] = static_cast<float>(v.i[k]);
            return o;
        }
        case DType::U32: {
            std::vector<float> o(v.u.size());
            for (std::size_t k = 0; k < o.size(); ++k) o[k] = static_cast<float>(v.u[k]);
            return o;
        }
        case DType::Bool: {
            std::vector<float> o(v.b.size());
            for (std::size_t k = 0; k < o.size(); ++k) o[k] = v.b[k] ? 1.0f : 0.0f;
            return o;
        }
        default: return v.f;
    }
}

inline std::vector<std::int64_t> lanes_i64(const Value& v) {
    std::vector<std::int64_t> o(v.len());
    switch (v.dtype) {
        case DType::I32: for (std::size_t k = 0; k < o.size(); ++k) o[k] = v.i[k]; break;
        case DType::U32: for (std::size_t k = 0; k < o.size(); ++k) o[k] = v.u[k]; break;
        case DType::Bool: for (std::size_t k = 0; k < o.size(); ++k) o[k] = v.b[k] ? 1 : 0; break;
        default: for (std::size_t k = 0; k < o.size(); ++k) o[k] = static_cast<std::int64_t>(v.f[k]); break;
    }
    return o;
}

inline Value from_i64(DType dtype, const std::vector<std::int64_t>& x) {
    switch (dtype) {
        case DType::U32: {
            std::vector<std::uint32_t> o(x.size());
            for (std::size_t k = 0; k < x.size(); ++k) o[k] = static_cast<std::uint32_t>(x[k]);
            return Value::u32(std::move(o));
        }
        case DType::Bool: {
            std::vector<std::uint8_t> o(x.size());
            for (std::size_t k = 0; k < x.size(); ++k) o[k] = x[k] != 0;
            return Value::boolean(std::move(o));
        }
        case DType::F32: case DType::Act: {
            std::vector<float> o(x.size());
            for (std::size_t k = 0; k < x.size(); ++k) o[k] = static_cast<float>(x[k]);
            return Value::f32(std::move(o));
        }
        default: {
            std::vector<std::int32_t> o(x.size());
            for (std::size_t k = 0; k < x.size(); ++k) o[k] = static_cast<std::int32_t>(x[k]);
            return Value::i32(std::move(o));
        }
    }
}

// Scalar-broadcast index rule shared by every elementwise op (interp.rs pick).
inline std::size_t pick(std::size_t len, std::size_t i) { return len == 1 ? 0 : i; }

// ─────────────────────────── wire cell codec ───────────────────────────────

inline std::size_t wire_cell_bytes(DType dtype, std::size_t numel) {
    return dtype == DType::Bool ? (numel + 7) / 8 : numel * 4;
}

// Decode one ring cell (bool cells are LSB-first bit-packed, D1).
inline bool decode_wire(const std::uint8_t* bytes, std::size_t len, DType dtype,
                        std::size_t numel, Value& out) {
    if (len != wire_cell_bytes(dtype, numel)) return false;
    switch (dtype) {
        case DType::Bool: {
            std::vector<std::uint8_t> o(numel);
            for (std::size_t j = 0; j < numel; ++j) o[j] = (bytes[j / 8] >> (j % 8)) & 1;
            out = Value::boolean(std::move(o));
            return true;
        }
        case DType::I32: {
            std::vector<std::int32_t> o(numel);
            std::memcpy(o.data(), bytes, len);
            out = Value::i32(std::move(o));
            return true;
        }
        case DType::U32: {
            std::vector<std::uint32_t> o(numel);
            std::memcpy(o.data(), bytes, len);
            out = Value::u32(std::move(o));
            return true;
        }
        default: {
            std::vector<float> o(numel);
            std::memcpy(o.data(), bytes, len);
            out = Value::f32(std::move(o));
            return true;
        }
    }
}

// Encode one ring cell in place (dst holds wire_cell_bytes).
inline void encode_wire(const Value& v, std::uint8_t* dst) {
    switch (v.dtype) {
        case DType::Bool: {
            const std::size_t packed = (v.b.size() + 7) / 8;
            std::memset(dst, 0, packed);
            for (std::size_t j = 0; j < v.b.size(); ++j)
                if (v.b[j] != 0) dst[j / 8] |= static_cast<std::uint8_t>(1u << (j % 8));
            break;
        }
        case DType::I32: std::memcpy(dst, v.i.data(), v.i.size() * 4); break;
        case DType::U32: std::memcpy(dst, v.u.data(), v.u.size() * 4); break;
        default: std::memcpy(dst, v.f.data(), v.f.size() * 4); break;
    }
}

// ───────────────────────────── exec plan ───────────────────────────────────

// Per-stage execution index over the translated Trace: the stage's global SSA
// id range and each op keyed by its first result id (roots resolve inline).
struct StagePlan {
    std::size_t stage_index = 0;
    std::uint32_t base = 0;
    std::uint32_t end = 0;
    std::unordered_map<std::uint32_t, const cptir::Op*> op_by_result;
};

struct ConstPortValue {
    std::uint8_t port = 0;
    Value value;
};

// A registered program's decoded, executability-classified form.
//
// Classification, not a capability flag (metal_ptir_plan.md §3.2): a program
// that reads Intrinsic(Logits)/Intrinsic(MtpLogits) is `executable` — it just
// additionally needs a forward pass before `step()` (`needs_logits` /
// `needs_mtp_logits`). Only genuinely unsupported constructs (HostInput,
// per-layer taps, and non-logits intrinsics — hidden/query/value-head) still
// hard-reject with `executable = false` + a precise `reject_reason`.
struct ExecPlan {
    Trace trace;
    cptir::bound::Bound bound;
    std::vector<StagePlan> stages;  // container order (mirrors trace.stages)
    std::vector<cptir::plan::StagePlan> region_plans;
    std::vector<ConstPortValue> const_ports;
    bool executable = false;
    std::string reject_reason;
    // Set together with `executable = true` when the program reads the base
    // Intrinsic(Logits) / Intrinsic(MtpLogits) root(s) — the launch path must
    // run the Metal forward and bind `PassInputs` before `step()`.
    bool needs_logits = false;
    bool needs_mtp_logits = false;

    // One fire consumes (takes) dense channel `c` — a stage ChanTake or a
    // consuming descriptor port. Register-like: at most once per fire.
    bool takes_channel(std::uint32_t c) const {
        for (const auto& st : trace.stages)
            for (auto t : st.takes)
                if (t == c) return true;
        for (const auto& p : trace.ports)
            if (!p.is_const && p.channel == c && cptir::port_consumes(p.port)) return true;
        return false;
    }

    bool puts_channel(std::uint32_t c) const {
        for (const auto& st : trace.stages)
            for (const auto& put : st.puts)
                if (put.channel == c) return true;
        return false;
    }

    bool requires_channel_input(std::uint32_t c) const {
        return std::any_of(
            bound.readiness.begin(), bound.readiness.end(),
            [c](const auto& entry) {
                return entry.chan == c &&
                       entry.dir == cptir::container::Direction::NeedsFull;
            });
    }

    // True if the launch path must run the forward before `step()`.
    bool needs_forward() const {
        return needs_logits || needs_mtp_logits || !trace.ports.empty();
    }
};

inline int bounded_mtp_row_base(
    const ExecPlan& plan,
    std::uint32_t vocab) {
    if (!plan.needs_mtp_logits || vocab == 0) return -1;
    std::uint64_t rows = 0;
    for (const cptir::Value& value : plan.trace.values) {
        if (value.source != cptir::ValueSource::Intrinsic ||
            value.intrinsic != cptir::Intrinsic::Logits) {
            continue;
        }
        const std::uint64_t numel =
            value.type.shape.numel();
        if (numel % vocab != 0) return -1;
        rows = std::max(rows, numel / vocab);
    }
    return rows <=
                   static_cast<std::uint64_t>(
                       std::numeric_limits<int>::max())
               ? static_cast<int>(rows)
               : -1;
}

// Classify an already-translated Trace: split "rejected" (executable=false,
// a precise reason) from "needs forward inputs" (needs_logits). Pure function
// over the decoded Trace
// — no container/sidecar bytes involved — so callers (registration AND
// tests) can classify a hand-built Trace directly.
inline void classify_exec_plan(ExecPlan& out) {
    out.executable = true;
    out.needs_logits = false;
    out.needs_mtp_logits = false;
    out.reject_reason.clear();
    for (const cptir::Value& v : out.trace.values) {
        if (v.source == cptir::ValueSource::Intrinsic) {
            switch (v.intrinsic) {
                case cptir::Intrinsic::Logits:
                    out.needs_logits = true;
                    break;
                case cptir::Intrinsic::MtpLogits:
                case cptir::Intrinsic::MtpDrafts:
                    out.needs_logits = true;
                    out.needs_mtp_logits = true;
                    break;
                default:
                    // The Metal forward exposes no per-layer taps, layer
                    // ordinal, or auxiliary heads in this increment.
                    out.executable = false;
                    out.reject_reason =
                        "program reads an unsupported model intrinsic "
                        "(hidden/query/value-head/layer; Metal forward not wired)";
                    break;
            }
        } else if (v.source == cptir::ValueSource::HostInput) {
            out.executable = false;
            out.reject_reason = "program reads a submit-bound host input (not executable on Metal)";
        }
    }
    for (const cptir::Stage& st : out.trace.stages) {
        if (st.kind == cptir::StageKind::OnAttnProj || st.kind == cptir::StageKind::OnAttn) {
            out.executable = false;
            out.reject_reason = "program attaches per-layer taps (Metal forward not wired)";
        }
    }
    if (!out.executable) {
        out.needs_logits = false;
        out.needs_mtp_logits = false;
    }
}

inline bool validate_region_plan_safety(
    const cptir::plan::StagePlan& plan,
    std::string& error) {
    for (const auto& type : plan.value_types) {
        std::uint64_t product = 1;
        if (type.dims.size() > 4 || type.dtype > PTIR_DT_BOOL) {
            error = "invalid normalized value type";
            return false;
        }
        for (const auto& dimension : type.dims) {
            if (dimension.symbolic) {
                if (dimension.value < PTIR_EXTENT_KV_LEN ||
                    dimension.value > PTIR_EXTENT_KEY_LEN) {
                    error = "invalid symbolic extent role";
                    return false;
                }
            } else if (
                dimension.value == 0 ||
                product >
                    std::numeric_limits<std::uint32_t>::max() /
                        dimension.value) {
                error = "normalized value shape product exceeds u32";
                return false;
            } else {
                product *= dimension.value;
            }
        }
    }
    std::uint32_t result_base = 0;
    for (const auto& normalized : plan.ops) {
        const auto& op = normalized.op;
        if (!cptir::op_is_known(static_cast<OpCode>(op.tag))) {
            error = "unknown normalized operation";
            return false;
        }
        if (result_base >
                std::numeric_limits<std::uint32_t>::max() -
                    op.results ||
            result_base + op.results > plan.value_types.size()) {
            error = "normalized result range is invalid";
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
        if ((op.tag == PTIR_OP_CHAN_TAKE ||
             op.tag == PTIR_OP_CHAN_READ ||
             op.tag == PTIR_OP_CHAN_PUT) &&
            (op.chan < 0 ||
             static_cast<std::size_t>(op.chan) >=
                 plan.channel_bindings.size())) {
            error = "normalized channel slot is invalid";
            return false;
        }
        result_base += op.results;
    }
    if (result_base != plan.value_types.size()) {
        error = "normalized value layout does not match results";
        return false;
    }
    return true;
}

inline bool decode_const_ports(
    const std::uint8_t* data,
    std::size_t len,
    std::vector<ConstPortValue>& ports,
    std::string& error) {
    ports.clear();
    cptir::container::detail::Cur cursor{data, len};
    if (data == nullptr || len < 24 ||
        std::memcmp(data, PTIR_MAGIC, 4) != 0) {
        error = "invalid PTIR header while decoding const ports";
        return false;
    }
    cursor.skip(4);
    const std::uint16_t version = cursor.u16();
    if (version != PTIR_VERSION && version != PTIR_VERSION_EXTERN) {
        error = "unsupported PTIR version while decoding const ports";
        return false;
    }
    cursor.u16();
    const std::uint32_t name_count = cursor.u32();
    const std::uint32_t channel_count = cursor.u32();
    const std::uint32_t port_count = cursor.u32();
    cursor.u32();
    if (version == PTIR_VERSION_EXTERN) cursor.u32();
    for (std::uint32_t name = 0; name < name_count; ++name) {
        const std::uint16_t size = cursor.u16();
        cursor.skip(size);
    }
    for (std::uint32_t channel = 0; channel < channel_count; ++channel) {
        cursor.u8();
        cursor.shape();
        cursor.u32();
        cursor.u8();
        cursor.u8();
    }
    for (std::uint32_t index = 0; index < port_count; ++index) {
        const std::uint8_t port = cursor.u8();
        const std::uint8_t source = cursor.u8();
        if (source == 0) {
            cursor.u32();
            continue;
        }
        if (source != 1) {
            error = "invalid PTIR const-port source";
            return false;
        }
        const std::uint8_t dtype = cursor.u8();
        const cptir::container::CShape shape = cursor.shape();
        std::size_t count = 1;
        for (std::uint8_t dimension = 0; dimension < shape.rank;
             ++dimension) {
            if (shape.dims[dimension] == 0 ||
                count > std::numeric_limits<std::size_t>::max() /
                            shape.dims[dimension]) {
                error = "invalid PTIR const-port shape";
                return false;
            }
            count *= shape.dims[dimension];
        }
        const std::size_t element_bytes =
            dtype == PTIR_DT_BOOL ? 1 : 4;
        if (dtype > PTIR_DT_BOOL ||
            count > std::numeric_limits<std::size_t>::max() /
                        element_bytes ||
            !cursor.need(count * element_bytes)) {
            error = "invalid PTIR const-port payload";
            return false;
        }
        const std::uint8_t* payload = data + cursor.i;
        ConstPortValue decoded;
        decoded.port = port;
        if (dtype == PTIR_DT_F32) {
            std::vector<float> values(count);
            std::memcpy(values.data(), payload, count * sizeof(float));
            decoded.value = Value::f32(std::move(values));
        } else if (dtype == PTIR_DT_I32) {
            std::vector<std::int32_t> values(count);
            std::memcpy(
                values.data(), payload, count * sizeof(std::int32_t));
            decoded.value = Value::i32(std::move(values));
        } else if (dtype == PTIR_DT_U32) {
            std::vector<std::uint32_t> values(count);
            std::memcpy(
                values.data(), payload, count * sizeof(std::uint32_t));
            decoded.value = Value::u32(std::move(values));
        } else {
            std::vector<std::uint8_t> values(payload, payload + count);
            for (std::uint8_t& value : values) value = value != 0;
            decoded.value = Value::boolean(std::move(values));
        }
        cursor.skip(count * element_bytes);
        ports.push_back(std::move(decoded));
    }
    if (cursor.err) {
        error = "PTIR const-port payload overruns container";
        return false;
    }
    return true;
}

// Decode + classify. Registration never fails on executability — the launch
// path rejects with `reject_reason` instead.
inline bool build_exec_plan(const std::uint8_t* container_bytes, std::size_t container_len,
                            const std::uint8_t* sidecar_bytes, std::size_t sidecar_len,
                            ExecPlan& out, std::string* error) {
    out = {};
    cptir::container::Container c;
    cptir::container::DecodeError derr;
    if (!cptir::container::decode(container_bytes, container_len, c, &derr)) {
        if (error != nullptr) *error = "container decode: " + derr.detail;
        return false;
    }
    std::string const_error;
    if (!decode_const_ports(
            container_bytes,
            container_len,
            out.const_ports,
            const_error)) {
        if (error != nullptr) *error = const_error;
        return false;
    }
    if (sidecar_bytes == nullptr || sidecar_len == 0) {
        out.executable = false;
        out.reject_reason = "program has no PTIB sidecar";
        return true;
    }
    std::string serr;
    if (!cptir::bound::parse_sidecar(sidecar_bytes, sidecar_len, out.bound, &serr)) {
        if (error != nullptr) *error = "sidecar decode: " + serr;
        return false;
    }
    cptir::container::Container translated_container = c;
    for (auto& stage : translated_container.stages) {
        std::vector<cptir::container::COp> lowered;
        lowered.reserve(stage.ops.size());
        for (auto op : stage.ops) {
            if (op.tag == PTIR_OP_KERNEL_CALL) {
                if (op.name_idx >= c.names.size() ||
                    c.names[op.name_idx] != "metal.identity" ||
                    op.args.size() != 1) {
                    out.executable = false;
                    out.reject_reason =
                        "program requests an unsupported Metal semantic "
                        "kernel boundary";
                    return true;
                }
                op.tag = PTIR_OP_RESHAPE;
                lowered.push_back(std::move(op));
            } else if (op.tag == PTIR_OP_SINK_CALL) {
                if (op.name_idx >= c.names.size() ||
                    c.names[op.name_idx] != "metal.discard") {
                    out.executable = false;
                    out.reject_reason =
                        "program requests an unsupported Metal semantic "
                        "sink boundary";
                    return true;
                }
            } else {
                lowered.push_back(std::move(op));
            }
        }
        stage.ops = std::move(lowered);
    }
    auto translated = cptir::bound::container_to_trace(
        translated_container, out.bound);
    if (!translated.ok) {
        out.executable = false;
        out.reject_reason = translated.error;
        return true;
    }
    out.trace = std::move(translated.trace);
    if (out.bound.version >= 2) {
        if (out.bound.plans.size() != c.stages.size()) {
            if (error != nullptr) *error = "compiler plan count does not match stages";
            return false;
        }
        for (const cptir::bound::StagePlan& encoded : out.bound.plans) {
            cptir::plan::StagePlan plan;
            std::string plan_error;
            if (!cptir::plan::decode(
                    encoded.bytes.data(), encoded.bytes.size(), plan, &plan_error)) {
                if (error != nullptr) *error = "compiler plan decode: " + plan_error;
                return false;
            }
            if (plan.stage != encoded.stage || plan.signature_hash == 0) {
                if (error != nullptr) *error = "compiler plan identity mismatch";
                return false;
            }
            if (!validate_region_plan_safety(plan, plan_error)) {
                if (error != nullptr) {
                    *error =
                        "compiler plan validation: " + plan_error;
                }
                return false;
            }
            out.region_plans.push_back(std::move(plan));
        }
    }

    std::uint32_t base = 0;
    for (std::size_t si = 0; si < out.trace.stages.size(); ++si) {
        StagePlan plan;
        plan.stage_index = si;
        plan.base = base;
        base += static_cast<std::uint32_t>(out.bound.stages[si].value_types.size());
        plan.end = base;
        for (const cptir::Op& op : out.trace.stages[si].ops) plan.op_by_result.emplace(op.result_id, &op);
        out.stages.push_back(std::move(plan));
    }

    classify_exec_plan(out);
    for (const auto& stage : c.stages) {
        for (const auto& op : stage.ops) {
            if ((op.tag == PTIR_OP_KERNEL_CALL ||
                 op.tag == PTIR_OP_SINK_CALL) &&
                (op.name_idx >= c.names.size() ||
                 c.names[op.name_idx] !=
                     (op.tag == PTIR_OP_KERNEL_CALL
                          ? "metal.identity"
                          : "metal.discard"))) {
                out.executable = false;
                out.needs_logits = false;
                out.needs_mtp_logits = false;
                out.reject_reason =
                    "program requests an unsupported Metal semantic "
                    "library boundary";
            }
        }
    }
    return true;
}

// ─────────────────────────── instance state ────────────────────────────────

// One authoritative channel ring. The exact bytes and head/tail words exposed
// through PieChannelEndpointBinding are the bytes the interpreter reads and
// writes; Apple production allocations are MTLStorageModeShared and therefore
// directly bindable by future generated kernels.
struct ChannelState {
    DType dtype = DType::F32;
    std::size_t numel = 1;
    std::size_t capacity = 1;
    std::size_t cell_bytes = sizeof(float);
    std::size_t cap1 = 2;
    SharedStorage cells;
    SharedStorage words;

    ChannelState(
        DType dtype_in,
        std::size_t numel_in,
        std::size_t capacity_in,
        SharedStorage cells_in,
        SharedStorage words_in)
        : dtype(dtype_in == DType::Act ? DType::F32 : dtype_in),
          numel(std::max<std::size_t>(numel_in, 1)),
          capacity(std::max<std::size_t>(capacity_in, 1)),
          cell_bytes(wire_cell_bytes(dtype, numel)),
          cap1(capacity + 1),
          cells(std::move(cells_in)),
          words(std::move(words_in)) {}

    bool valid() const {
        return cells.valid() && words.valid() &&
               cells.size >= cell_bytes * cap1 &&
               words.size >= 4 * sizeof(std::uint64_t);
    }

    std::uint64_t load_word(std::size_t index) const {
        auto* base = reinterpret_cast<std::uint64_t*>(words.contents);
        return std::atomic_ref<std::uint64_t>(base[index])
            .load(std::memory_order_acquire);
    }

    void store_word(std::size_t index, std::uint64_t value) {
        auto* base = reinterpret_cast<std::uint64_t*>(words.contents);
        std::atomic_ref<std::uint64_t>(base[index])
            .store(value, std::memory_order_release);
    }

    std::uint64_t head() const { return load_word(0); }
    std::uint64_t tail() const { return load_word(1); }
    std::uint64_t poison() const { return load_word(2); }
    std::uint64_t closed() const { return load_word(3); }

    std::size_t size() const {
        const std::uint64_t h = head();
        const std::uint64_t t = tail();
        return t >= h ? static_cast<std::size_t>(t - h) : 0;
    }
    bool empty() const { return size() == 0; }
    bool full() const { return size() >= capacity; }

    std::uint8_t* slot(std::uint64_t sequence) {
        return cells.contents + (sequence % cap1) * cell_bytes;
    }
    const std::uint8_t* slot(std::uint64_t sequence) const {
        return cells.contents + (sequence % cap1) * cell_bytes;
    }

    Value decode_sequence(std::uint64_t sequence) const {
        Value value;
        if (!decode_wire(slot(sequence), cell_bytes, dtype, numel, value)) {
            return zeros(dtype, numel);
        }
        return value;
    }

    void encode_sequence(std::uint64_t sequence, const Value& value) {
        encode_wire(value, slot(sequence));
    }

    Value front() const {
        return decode_sequence(head());
    }

    Value current() const {
        const std::uint64_t h = head();
        if (tail() > h) return decode_sequence(h);
        const std::uint64_t last_slot = (h + cap1 - 1) % cap1;
        return decode_sequence(last_slot);
    }

    bool push(Value value) {
        if (value.dtype != dtype || value.len() != numel || full()) return false;
        const std::uint64_t t = tail();
        encode_sequence(t, value);
        store_word(1, t + 1);
        return true;
    }

    bool pop(Value& value) {
        if (empty()) return false;
        const std::uint64_t h = head();
        value = decode_sequence(h);
        store_word(0, h + 1);
        return true;
    }
};

inline std::shared_ptr<ChannelState> make_host_channel_state(
    DType dtype,
    std::size_t numel,
    std::size_t capacity) {
    const DType concrete = dtype == DType::Act ? DType::F32 : dtype;
    const std::size_t n = std::max<std::size_t>(numel, 1);
    const std::size_t cap = std::max<std::size_t>(capacity, 1);
    const std::size_t bytes = wire_cell_bytes(concrete, n);
    auto state = std::make_shared<ChannelState>(
        concrete,
        n,
        cap,
        make_host_shared_storage(bytes * (cap + 1)),
        make_host_shared_storage(4 * sizeof(std::uint64_t)));
    return state->valid() ? state : nullptr;
}

inline std::shared_ptr<ChannelState> make_platform_channel_state(
    DType dtype, std::size_t numel, std::size_t capacity) {
    const DType concrete = dtype == DType::Act ? DType::F32 : dtype;
    const std::size_t n = std::max<std::size_t>(numel, 1);
    const std::size_t cap = std::max<std::size_t>(capacity, 1);
    const std::size_t bytes = wire_cell_bytes(concrete, n);
    auto state = std::make_shared<ChannelState>(
        concrete,
        n,
        cap,
        make_platform_shared_storage(bytes * (cap + 1)),
        make_platform_shared_storage(4 * sizeof(std::uint64_t)));
    return state->valid() ? state : nullptr;
}

struct InterpInstance {
    std::vector<std::shared_ptr<ChannelState>> channels;
    bool poisoned = false;
};

inline InterpInstance make_instance(
    const ExecPlan& plan,
    const std::vector<std::shared_ptr<ChannelState>>& channels) {
    InterpInstance inst;
    if (channels.size() == plan.trace.channels.size()) {
        inst.channels = channels;
    }
    return inst;
}

// Pure-host test helper. Production Registry bindings supply the endpoint's
// platform-shared ChannelState through the overload above.
inline InterpInstance make_instance(const ExecPlan& plan,
                                    const std::map<std::uint32_t, std::shared_ptr<ChannelState>>& externs,
                                    const std::map<std::uint32_t, Value>& seeds) {
    InterpInstance inst;
    for (std::size_t ci = 0; ci < plan.trace.channels.size(); ++ci) {
        const cptir::Channel& decl = plan.trace.channels[ci];
        auto shared = externs.find(static_cast<std::uint32_t>(ci));
        if (shared != externs.end()) {
            inst.channels.push_back(shared->second);
        } else {
            inst.channels.push_back(make_host_channel_state(
                decl.type.dtype, decl.type.shape.numel(), decl.capacity));
        }
        auto seed = seeds.find(static_cast<std::uint32_t>(ci));
        if (seed != seeds.end() && inst.channels.back()->empty()) {
            (void)inst.channels.back()->push(seed->second);
        }
    }
    return inst;
}

enum class HostOp { Ok, WouldBlock, Poisoned, WrongRole, TypeMismatch };

inline HostOp host_put(InterpInstance& inst, const ExecPlan& plan, std::uint32_t chan, Value v) {
    if (inst.poisoned) return HostOp::Poisoned;
    const cptir::Channel& decl = plan.trace.channels[chan];
    if (!decl.host_visible || decl.host_reader) return HostOp::WrongRole;
    if (!value_matches(v, decl.type)) return HostOp::TypeMismatch;
    ChannelState& st = *inst.channels[chan];
    return st.push(std::move(v)) ? HostOp::Ok : HostOp::WouldBlock;
}

inline HostOp host_take(InterpInstance& inst, const ExecPlan& plan, std::uint32_t chan, Value& out) {
    if (inst.poisoned) return HostOp::Poisoned;
    const cptir::Channel& decl = plan.trace.channels[chan];
    if (!decl.host_reader) return HostOp::WrongRole;
    ChannelState& st = *inst.channels[chan];
    return st.pop(out) ? HostOp::Ok : HostOp::WouldBlock;
}

// ───────────────────────────── op evaluation ────────────────────────────────

namespace detail {

inline float neg_inf() { return -std::numeric_limits<float>::infinity(); }

inline std::size_t canonical_rows(const cptir::Shape& shape) {
    if (shape.dims.size() < 2) return 1;
    std::size_t rows = 1;
    for (std::size_t dimension = 0;
         dimension + 1 < shape.dims.size();
         ++dimension) {
        rows *= shape.dims[dimension];
    }
    return rows;
}

// Canonical logical width-32 tree shared with interface/ptir's reference.
// Physical launch dimensions must never affect this order.
template <class T, class Combine>
inline T canonical_reduce(
    const T* row,
    std::size_t len,
    T identity,
    Combine combine) {
    if (len == 0) return identity;
    std::vector<T> level(row, row + len);
    while (level.size() > 1) {
        std::vector<T> next;
        next.reserve((level.size() + 31) / 32);
        for (std::size_t base = 0; base < level.size(); base += 32) {
            T lanes[32];
            std::fill(std::begin(lanes), std::end(lanes), identity);
            const std::size_t count = std::min<std::size_t>(32, level.size() - base);
            std::copy_n(level.data() + base, count, lanes);
            for (const std::size_t offset : {16u, 8u, 4u, 2u, 1u}) {
                for (std::size_t lane = 0; lane < offset; ++lane) {
                    lanes[lane] = combine(lanes[lane], lanes[lane + offset]);
                }
            }
            next.push_back(lanes[0]);
        }
        level = std::move(next);
    }
    return level[0];
}

struct ArgmaxCandidate {
    float value = neg_inf();
    std::uint32_t index = 0;
    bool have = false;
};

struct IntArgmaxCandidate {
    std::int64_t value = 0;
    std::uint32_t index = 0;
    bool have = false;
};

inline ArgmaxCandidate combine_argmax(
    ArgmaxCandidate left,
    ArgmaxCandidate right) {
    if (!right.have) return left;
    if (!left.have || right.value > left.value ||
        (right.value == left.value && right.index < left.index)) {
        return right;
    }
    return left;
}

inline IntArgmaxCandidate combine_int_argmax(
    IntArgmaxCandidate left,
    IntArgmaxCandidate right) {
    if (!right.have) return left;
    if (!left.have || right.value > left.value ||
        (right.value == left.value && right.index < left.index)) {
        return right;
    }
    return left;
}

inline float canonical_max(float left, float right) {
    const bool left_nan = std::isnan(left);
    const bool right_nan = std::isnan(right);
    if (left_nan && right_nan) return neg_inf();
    if (left_nan) return right;
    if (right_nan) return left;
    if (left == 0.0f && right == 0.0f) {
        return std::signbit(left) && std::signbit(right) ? -0.0f : 0.0f;
    }
    return std::fmax(left, right);
}

inline float canonical_min(float left, float right) {
    const bool left_nan = std::isnan(left);
    const bool right_nan = std::isnan(right);
    if (left_nan && right_nan) return std::numeric_limits<float>::infinity();
    if (left_nan) return right;
    if (right_nan) return left;
    if (left == 0.0f && right == 0.0f) {
        return std::signbit(left) || std::signbit(right) ? -0.0f : 0.0f;
    }
    return std::fmin(left, right);
}

// Argmax pinned contract: lower index wins ties; NaN never selected
// (all-NaN row → 0), evaluated through the canonical tree.
inline std::int32_t argmax_row(const float* row, std::size_t len) {
    std::vector<ArgmaxCandidate> candidates;
    candidates.reserve(len);
    for (std::size_t j = 0; j < len; ++j) {
        candidates.push_back(
            {row[j], static_cast<std::uint32_t>(j), !std::isnan(row[j])});
    }
    const ArgmaxCandidate result = canonical_reduce(
        candidates.data(),
        candidates.size(),
        ArgmaxCandidate{},
        combine_argmax);
    return static_cast<std::int32_t>(result.index);
}

inline std::int32_t argmax_row_i64(
    const std::int64_t* row, std::size_t len) {
    std::vector<IntArgmaxCandidate> candidates;
    candidates.reserve(len);
    for (std::size_t index = 0; index < len; ++index) {
        candidates.push_back({
            row[index], static_cast<std::uint32_t>(index), true});
    }
    return static_cast<std::int32_t>(
        canonical_reduce(
            candidates.data(),
            candidates.size(),
            IntArgmaxCandidate{},
            combine_int_argmax)
            .index);
}

// sort_desc pinned contract: descending; ties → lower original index; NaN
// below −inf (last).
inline std::vector<std::uint32_t> sort_desc_order(const float* row, std::size_t len) {
    std::vector<std::uint32_t> idx(len);
    for (std::size_t j = 0; j < len; ++j) idx[j] = static_cast<std::uint32_t>(j);
    std::stable_sort(idx.begin(), idx.end(), [&](std::uint32_t a, std::uint32_t b) {
        const float x = row[a];
        const float y = row[b];
        const bool nx = std::isnan(x);
        const bool ny = std::isnan(y);
        if (nx != ny) return ny;  // NaN last
        if (nx && ny) return a < b;
        if (x != y) return x > y;
        return a < b;
    });
    return idx;
}

inline std::vector<float> rng_lanes(std::uint64_t seed_eff, std::size_t n, bool gumbel) {
    std::vector<float> o(n);
    for (std::size_t j = 0; j < n; ++j) {
        const float u =
            ptir_rng_hash_uniform(seed_eff, static_cast<std::uint32_t>(j));
        o[j] = gumbel ? -std::log(-std::log(u)) : u;
    }
    return o;
}

template <class FF, class FI>
Value bin_arith(const Value& a, const Value& b, DType dtype, FF f_f, FI f_i) {
    if (dtype == DType::F32 || dtype == DType::Act) {
        const auto av = lanes_f32(a);
        const auto bv = lanes_f32(b);
        const std::size_t n = std::max(av.size(), bv.size());
        std::vector<float> o(n);
        for (std::size_t i = 0; i < n; ++i) o[i] = f_f(av[pick(av.size(), i)], bv[pick(bv.size(), i)]);
        return Value::f32(std::move(o));
    }
    const auto av = lanes_i64(a);
    const auto bv = lanes_i64(b);
    const std::size_t n = std::max(av.size(), bv.size());
    std::vector<std::int64_t> o(n);
    for (std::size_t i = 0; i < n; ++i) o[i] = f_i(av[pick(av.size(), i)], bv[pick(bv.size(), i)]);
    return from_i64(dtype, o);
}

template <class FF, class FI>
Value cmp_op(const Value& a, const Value& b, DType in_dtype, FF f_f, FI f_i) {
    std::vector<std::uint8_t> o;
    if (in_dtype == DType::F32 || in_dtype == DType::Act) {
        const auto av = lanes_f32(a);
        const auto bv = lanes_f32(b);
        const std::size_t n = std::max(av.size(), bv.size());
        o.resize(n);
        for (std::size_t i = 0; i < n; ++i) o[i] = f_f(av[pick(av.size(), i)], bv[pick(bv.size(), i)]) ? 1 : 0;
    } else {
        const auto av = lanes_i64(a);
        const auto bv = lanes_i64(b);
        const std::size_t n = std::max(av.size(), bv.size());
        o.resize(n);
        for (std::size_t i = 0; i < n; ++i) o[i] = f_i(av[pick(av.size(), i)], bv[pick(bv.size(), i)]) ? 1 : 0;
    }
    return Value::boolean(std::move(o));
}

template <class F>
Value map_f32(const Value& v, F f) {
    auto x = lanes_f32(v);
    for (float& e : x) e = f(e);
    return Value::f32(std::move(x));
}

inline Value gather_flat(const Value& v, const std::vector<std::size_t>& idx) {
    constexpr std::size_t kFill = std::numeric_limits<std::size_t>::max();
    switch (v.dtype) {
        case DType::I32: {
            std::vector<std::int32_t> o(idx.size());
            for (std::size_t k = 0; k < idx.size(); ++k) o[k] = idx[k] == kFill ? 0 : v.i[idx[k]];
            return Value::i32(std::move(o));
        }
        case DType::U32: {
            std::vector<std::uint32_t> o(idx.size());
            for (std::size_t k = 0; k < idx.size(); ++k) o[k] = idx[k] == kFill ? 0 : v.u[idx[k]];
            return Value::u32(std::move(o));
        }
        case DType::Bool: {
            std::vector<std::uint8_t> o(idx.size());
            for (std::size_t k = 0; k < idx.size(); ++k) o[k] = idx[k] == kFill ? 0 : v.b[idx[k]];
            return Value::boolean(std::move(o));
        }
        default: {
            std::vector<float> o(idx.size());
            for (std::size_t k = 0; k < idx.size(); ++k) o[k] = idx[k] == kFill ? 0.0f : v.f[idx[k]];
            return Value::f32(std::move(o));
        }
    }
}

// Left-aligned broadcast replicate (interp.rs broadcast_value), dtype-
// preserving: source dims align to the LEADING target dims; missing/1 dims
// replicate.
inline Value broadcast_value(const Value& v, const cptir::Shape& src, const cptir::Shape& target) {
    const std::size_t r = target.dims.size();
    auto sdim = [&](std::size_t i) -> std::uint64_t {
        return i < src.dims.size() ? src.dims[i] : 1;
    };
    std::vector<std::uint64_t> sstride(std::max<std::size_t>(r, 1), 1);
    for (std::size_t i = r >= 1 ? r - 1 : 0; i-- > 0;) sstride[i] = sstride[i + 1] * sdim(i + 1);
    const std::size_t n = static_cast<std::size_t>(target.numel());
    std::vector<std::size_t> idx(n);
    for (std::size_t lin = 0; lin < n; ++lin) {
        std::uint64_t rem = lin;
        std::uint64_t sidx = 0;
        for (std::size_t i = 0; i < r; ++i) {
            std::uint64_t stride = 1;
            for (std::size_t j = i + 1; j < r; ++j) stride *= target.dims[j];
            const std::uint64_t coord = rem / std::max<std::uint64_t>(stride, 1);
            rem %= std::max<std::uint64_t>(stride, 1);
            if (sdim(i) != 1) sidx += coord * sstride[i];
        }
        idx[lin] = static_cast<std::size_t>(sidx);
    }
    return gather_flat(v, idx);
}

}  // namespace detail

// ─────────────────────────────── the pass ───────────────────────────────────

// Per-fire forward inputs the interpreter binds into Intrinsic value roots —
// the Metal mirror of CUDA's `FireInputs` (tier0_runner.hpp) / interp.rs's
// `PassInputs`. `logits` is a `[rows, vocab]` row-major f32 matrix (the
// executor's readout rows for this fire, D3: bf16→f32 conversion happens in
// the executor, not here). `Intrinsic(Logits)` reads row 0; `Intrinsic
// (MtpLogits)` reads the K draft rows starting at `mtp_draft_row` within the
// SAME buffer, falling back to row 0 when unset (`-1`) — exactly the CUDA
// fallback (`tier0_runner.hpp` `resolve_root`'s `ValueSource::Intrinsic`
// case). C1 (channel-plane-only) fires pass a default-constructed
// `PassInputs{}` (`logits == nullptr`); `step()` never dereferences it unless
// the plan's trace actually roots an Intrinsic value.
struct PassInputs {
    const float* logits = nullptr;
    std::uint32_t rows = 0;
    std::uint32_t vocab = 0;
    int mtp_draft_row = -1;
};

struct StepResult {
    bool ok = false;         // false → hard fault (`error`), instance poisoned
    bool committed = false;  // readiness held and channel effects landed
    std::uint32_t missed_channel = 0;
    std::string error;
};

namespace detail {

struct Overlay {
    std::map<std::uint32_t, Value> pending;  // chan → pending put (last wins)
    std::vector<std::uint8_t> taken;
    std::vector<std::uint8_t> put;

    Value resolve(const InterpInstance& inst, std::uint32_t chan) const {
        auto p = pending.find(chan);
        if (p != pending.end()) return p->second;
        const ChannelState& st = *inst.channels[chan];
        return st.current();
    }
    Value take(const InterpInstance& inst, std::uint32_t chan) {
        taken[chan] = 1;
        return resolve(inst, chan);
    }
};

// Evaluate one compute op (SSA args already in `vals`). Mirrors interp.rs
// eval_op case for case; returns false + `error` on a semantic fault.
inline bool eval_op(const cptir::Op& op, const Trace& trace, std::vector<Value>& vals,
                    std::string& error) {
    auto v = [&](std::uint32_t id) -> const Value& { return vals[id]; };
    auto ty = [&](std::uint32_t id) -> const cptir::TensorType& { return trace.values[id].type; };
    auto fault = [&](const std::string& m) {
        error = m;
        return false;
    };
    auto out = [&](Value x) {
        vals[op.result_id] = std::move(x);
        return true;
    };
    const std::uint32_t a0 = op.args.empty() ? 0 : op.args[0];
    const std::uint32_t a1 = op.args.size() > 1 ? op.args[1] : 0;
    const std::uint32_t a2 = op.args.size() > 2 ? op.args[2] : 0;

    switch (op.code) {
        case OpCode::Exp: return out(map_f32(v(a0), [](float x) { return std::exp(x); }));
        case OpCode::Log: return out(map_f32(v(a0), [](float x) { return std::log(x); }));
        case OpCode::Recip: return out(map_f32(v(a0), [](float x) { return 1.0f / x; }));
        case OpCode::Neg: {
            const Value& x = v(a0);
            switch (x.dtype) {
                case DType::F32: case DType::Act: {
                    auto o = x.f;
                    for (float& e : o) e = -e;
                    return out(Value::f32(std::move(o)));
                }
                case DType::I32: {
                    auto o = x.i;
                    for (auto& e : o) e = static_cast<std::int32_t>(0 - static_cast<std::uint32_t>(e));
                    return out(Value::i32(std::move(o)));
                }
                case DType::U32: {
                    auto o = x.u;
                    for (auto& e : o) e = 0u - e;
                    return out(Value::u32(std::move(o)));
                }
                default: return fault("neg on bool");
            }
        }
        case OpCode::Abs: {
            const Value& x = v(a0);
            switch (x.dtype) {
                case DType::F32: case DType::Act: {
                    auto o = x.f;
                    for (float& e : o) e = std::fabs(e);
                    return out(Value::f32(std::move(o)));
                }
                case DType::I32: {
                    auto o = x.i;
                    for (auto& e : o) e = e == std::numeric_limits<std::int32_t>::min() ? e : std::abs(e);
                    return out(Value::i32(std::move(o)));
                }
                default: return out(x);
            }
        }
        case OpCode::Sign: {
            const Value& x = v(a0);
            switch (x.dtype) {
                case DType::F32: case DType::Act: {
                    auto o = x.f;
                    for (float& e : o) e = e > 0.0f ? 1.0f : (e < 0.0f ? -1.0f : 0.0f);
                    return out(Value::f32(std::move(o)));
                }
                case DType::I32: {
                    auto o = x.i;
                    for (auto& e : o) e = e > 0 ? 1 : (e < 0 ? -1 : 0);
                    return out(Value::i32(std::move(o)));
                }
                case DType::U32: {
                    auto o = x.u;
                    for (auto& e : o) e = e != 0 ? 1 : 0;
                    return out(Value::u32(std::move(o)));
                }
                default: return fault("sign on bool");
            }
        }
        case OpCode::Cast: {
            const Value& x = v(a0);
            const DType want = op.result_type.dtype == DType::Act ? DType::F32 : op.result_type.dtype;
            switch (want) {
                case DType::F32: return out(Value::f32(lanes_f32(x)));
                case DType::I32: {
                    if (x.dtype == DType::F32) {
                        const auto f = lanes_f32(x);
                        std::vector<std::int32_t> o(f.size());
                        for (std::size_t k = 0; k < f.size(); ++k) o[k] = static_cast<std::int32_t>(f[k]);
                        return out(Value::i32(std::move(o)));
                    }
                    return out(from_i64(DType::I32, lanes_i64(x)));
                }
                case DType::U32: {
                    if (x.dtype == DType::F32) {
                        const auto f = lanes_f32(x);
                        std::vector<std::uint32_t> o(f.size());
                        for (std::size_t k = 0; k < f.size(); ++k) o[k] = static_cast<std::uint32_t>(f[k]);
                        return out(Value::u32(std::move(o)));
                    }
                    return out(from_i64(DType::U32, lanes_i64(x)));
                }
                default: {
                    const auto f = lanes_f32(x);
                    std::vector<std::uint8_t> o(f.size());
                    for (std::size_t k = 0; k < f.size(); ++k) o[k] = f[k] != 0.0f ? 1 : 0;
                    return out(Value::boolean(std::move(o)));
                }
            }
        }

        case OpCode::Add:
            return out(bin_arith(v(a0), v(a1), ty(a0).dtype, [](float x, float y) { return x + y; },
                                 [](std::int64_t x, std::int64_t y) { return x + y; }));
        case OpCode::Sub:
            return out(bin_arith(v(a0), v(a1), ty(a0).dtype, [](float x, float y) { return x - y; },
                                 [](std::int64_t x, std::int64_t y) { return x - y; }));
        case OpCode::Mul:
            return out(bin_arith(v(a0), v(a1), ty(a0).dtype, [](float x, float y) { return x * y; },
                                 [](std::int64_t x, std::int64_t y) { return x * y; }));
        case OpCode::Div:
            return out(bin_arith(v(a0), v(a1), ty(a0).dtype, [](float x, float y) { return x / y; },
                                 [](std::int64_t x, std::int64_t y) { return y == 0 ? 0 : x / y; }));
        case OpCode::Rem:
            return out(bin_arith(v(a0), v(a1), ty(a0).dtype,
                                 [](float x, float y) { return std::fmod(x, y); },
                                 [](std::int64_t x, std::int64_t y) { return y == 0 ? 0 : x % y; }));
        case OpCode::MaxElem:
            return out(bin_arith(v(a0), v(a1), ty(a0).dtype,
                                 [](float x, float y) { return std::fmax(x, y); },
                                 [](std::int64_t x, std::int64_t y) { return std::max(x, y); }));
        case OpCode::MinElem:
            return out(bin_arith(v(a0), v(a1), ty(a0).dtype,
                                 [](float x, float y) { return std::fmin(x, y); },
                                 [](std::int64_t x, std::int64_t y) { return std::min(x, y); }));

        case OpCode::Gt:
            return out(cmp_op(v(a0), v(a1), ty(a0).dtype, [](float x, float y) { return x > y; },
                              [](std::int64_t x, std::int64_t y) { return x > y; }));
        case OpCode::Ge:
            return out(cmp_op(v(a0), v(a1), ty(a0).dtype, [](float x, float y) { return x >= y; },
                              [](std::int64_t x, std::int64_t y) { return x >= y; }));
        case OpCode::Eq:
            return out(cmp_op(v(a0), v(a1), ty(a0).dtype, [](float x, float y) { return x == y; },
                              [](std::int64_t x, std::int64_t y) { return x == y; }));
        case OpCode::Ne:
            return out(cmp_op(v(a0), v(a1), ty(a0).dtype, [](float x, float y) { return x != y; },
                              [](std::int64_t x, std::int64_t y) { return x != y; }));
        case OpCode::Lt:
            return out(cmp_op(v(a0), v(a1), ty(a0).dtype, [](float x, float y) { return x < y; },
                              [](std::int64_t x, std::int64_t y) { return x < y; }));
        case OpCode::Le:
            return out(cmp_op(v(a0), v(a1), ty(a0).dtype, [](float x, float y) { return x <= y; },
                              [](std::int64_t x, std::int64_t y) { return x <= y; }));
        case OpCode::And: case OpCode::Or: {
            const Value& x = v(a0);
            const Value& y = v(a1);
            if (x.dtype != DType::Bool || y.dtype != DType::Bool) return fault("and/or on non-bool");
            const bool is_and = op.code == OpCode::And;
            const std::size_t n = std::max(x.b.size(), y.b.size());
            std::vector<std::uint8_t> o(n);
            for (std::size_t i = 0; i < n; ++i) {
                const bool p = x.b[pick(x.b.size(), i)] != 0;
                const bool q = y.b[pick(y.b.size(), i)] != 0;
                o[i] = (is_and ? (p && q) : (p || q)) ? 1 : 0;
            }
            return out(Value::boolean(std::move(o)));
        }
        case OpCode::Not: {
            const Value& x = v(a0);
            if (x.dtype != DType::Bool) return fault("not on non-bool");
            auto o = x.b;
            for (auto& e : o) e = e ? 0 : 1;
            return out(Value::boolean(std::move(o)));
        }

        case OpCode::Select: {
            const Value& c = v(a0);
            if (c.dtype != DType::Bool) return fault("select cond");
            const Value& x = v(a1);
            const Value& y = v(a2);
            const std::size_t n = std::max({c.b.size(), x.len(), y.len()});
            auto sel = [&](std::size_t i) { return c.b[pick(c.b.size(), i)] != 0; };
            const DType d = ty(a1).dtype;
            if (d == DType::F32 || d == DType::Act) {
                const auto xf = lanes_f32(x);
                const auto yf = lanes_f32(y);
                std::vector<float> o(n);
                for (std::size_t i = 0; i < n; ++i)
                    o[i] = sel(i) ? xf[pick(xf.size(), i)] : yf[pick(yf.size(), i)];
                return out(Value::f32(std::move(o)));
            }
            if (d == DType::Bool) {
                if (x.dtype != DType::Bool || y.dtype != DType::Bool) return fault("select bool arms");
                std::vector<std::uint8_t> o(n);
                for (std::size_t i = 0; i < n; ++i)
                    o[i] = sel(i) ? x.b[pick(x.b.size(), i)] : y.b[pick(y.b.size(), i)];
                return out(Value::boolean(std::move(o)));
            }
            const auto xi = lanes_i64(x);
            const auto yi = lanes_i64(y);
            std::vector<std::int64_t> o(n);
            for (std::size_t i = 0; i < n; ++i)
                o[i] = sel(i) ? xi[pick(xi.size(), i)] : yi[pick(yi.size(), i)];
            return out(from_i64(d, o));
        }

        case OpCode::ReduceSum: case OpCode::ReduceMax: case OpCode::ReduceMin: {
            const cptir::TensorType& t = ty(a0);
            const std::size_t rows = canonical_rows(t.shape);
            const Value& data = v(a0);
            const std::size_t len = rows == 0 ? 0 : data.len() / rows;
            if (t.dtype == DType::F32 || t.dtype == DType::Act) {
                const auto x = lanes_f32(data);
                std::vector<float> o(rows);
                for (std::size_t r = 0; r < rows; ++r) {
                    const float* row = x.data() + r * len;
                    if (op.code == OpCode::ReduceSum) {
                        o[r] = canonical_reduce(
                            row, len, 0.0f,
                            [](float left, float right) { return left + right; });
                    } else if (op.code == OpCode::ReduceMax) {
                        o[r] = canonical_reduce(
                            row, len, neg_inf(), canonical_max);
                    } else {
                        o[r] = canonical_reduce(
                            row,
                            len,
                            std::numeric_limits<float>::infinity(),
                            canonical_min);
                    }
                }
                return out(Value::f32(std::move(o)));
            }
            const auto x = lanes_i64(data);
            std::vector<std::int64_t> o(rows);
            for (std::size_t r = 0; r < rows; ++r) {
                const std::int64_t* row = x.data() + r * len;
                if (op.code == OpCode::ReduceSum) {
                    o[r] = canonical_reduce(
                        row,
                        len,
                        std::int64_t{0},
                        [](std::int64_t left, std::int64_t right) {
                            const std::uint64_t bits =
                                static_cast<std::uint64_t>(left) +
                                static_cast<std::uint64_t>(right);
                            std::int64_t value;
                            std::memcpy(&value, &bits, sizeof(value));
                            return value;
                        });
                } else if (op.code == OpCode::ReduceMax) {
                    o[r] = canonical_reduce(
                        row,
                        len,
                        std::numeric_limits<std::int64_t>::min(),
                        [](std::int64_t left, std::int64_t right) {
                            return std::max(left, right);
                        });
                } else {
                    o[r] = canonical_reduce(
                        row,
                        len,
                        std::numeric_limits<std::int64_t>::max(),
                        [](std::int64_t left, std::int64_t right) {
                            return std::min(left, right);
                        });
                }
            }
            return out(from_i64(t.dtype, o));
        }
        case OpCode::ReduceArgmax: {
            const cptir::TensorType& t = ty(a0);
            const std::size_t rows = canonical_rows(t.shape);
            const Value& data = v(a0);
            const std::size_t len = rows == 0 ? 0 : data.len() / rows;
            std::vector<std::int32_t> o(rows);
            if (t.dtype == DType::F32 || t.dtype == DType::Act) {
                const auto x = lanes_f32(data);
                for (std::size_t r = 0; r < rows; ++r) {
                    o[r] = argmax_row(x.data() + r * len, len);
                }
            } else {
                const auto x = lanes_i64(data);
                for (std::size_t r = 0; r < rows; ++r) {
                    o[r] = argmax_row_i64(x.data() + r * len, len);
                }
            }
            return out(Value::i32(std::move(o)));
        }
        case OpCode::CumSum: case OpCode::CumProd: {
            const cptir::TensorType& t = ty(a0);
            const std::size_t rows = canonical_rows(t.shape);
            const auto x = lanes_f32(v(a0));
            const std::size_t len = rows == 0 ? 0 : x.size() / rows;
            const bool is_sum = op.code == OpCode::CumSum;
            std::vector<float> o;
            o.reserve(x.size());
            for (std::size_t r = 0; r < rows; ++r) {
                float acc = is_sum ? 0.0f : 1.0f;
                for (std::size_t j = 0; j < len; ++j) {
                    acc = is_sum ? acc + x[r * len + j] : acc * x[r * len + j];
                    o.push_back(acc);
                }
            }
            return out(Value::f32(std::move(o)));
        }

        case OpCode::Broadcast:
            return out(broadcast_value(v(a0), ty(a0).shape, op.result_type.shape));
        case OpCode::Reshape:
            return out(v(a0));  // metadata only (row-major)
        case OpCode::Transpose: {
            const cptir::TensorType& t = ty(a0);
            if (t.shape.dims.size() != 2) return fault("transpose rank");
            const std::size_t m = t.shape.dims[0];
            const std::size_t n = t.shape.dims[1];
            std::vector<std::size_t> idx(m * n);
            for (std::size_t o2 = 0; o2 < m * n; ++o2) idx[o2] = (o2 % m) * n + o2 / m;
            return out(gather_flat(v(a0), idx));
        }

        case OpCode::SortDesc: {
            const auto x = lanes_f32(v(a0));
            const auto order = sort_desc_order(x.data(), x.size());
            std::vector<float> sorted(order.size());
            for (std::size_t k = 0; k < order.size(); ++k) sorted[k] = x[order[k]];
            vals[op.result_id] = Value::f32(std::move(sorted));
            vals[op.result_id + 1] = Value::u32(std::vector<std::uint32_t>(order.begin(), order.end()));
            return true;
        }
        case OpCode::TopK: {
            const cptir::TensorType& t = ty(a0);
            const std::size_t rows = canonical_rows(t.shape);
            const auto x = lanes_f32(v(a0));
            const std::size_t len = rows == 0 ? 0 : x.size() / rows;
            const std::size_t k = op.imm;
            std::vector<float> vs;
            std::vector<std::uint32_t> is;
            vs.reserve(rows * k);
            is.reserve(rows * k);
            for (std::size_t r = 0; r < rows; ++r) {
                const auto order = sort_desc_order(x.data() + r * len, len);
                for (std::size_t p = 0; p < k && p < order.size(); ++p) {
                    vs.push_back(x[r * len + order[p]]);
                    is.push_back(order[p]);
                }
            }
            vals[op.result_id] = Value::f32(std::move(vs));
            vals[op.result_id + 1] = Value::u32(std::move(is));
            return true;
        }
        case OpCode::Matmul: {
            const cptir::TensorType& ta = ty(a0);
            const cptir::TensorType& tb = ty(a1);
            if (ta.shape.dims.size() != 2 || tb.shape.dims.size() != 2) return fault("matmul rank");
            const std::size_t m = ta.shape.dims[0];
            const std::size_t kk = ta.shape.dims[1];
            const std::size_t n = tb.shape.dims[1];
            const auto x = lanes_f32(v(a0));
            const auto y = lanes_f32(v(a1));
            std::vector<float> o(m * n, 0.0f);
            for (std::size_t i = 0; i < m; ++i)
                for (std::size_t l = 0; l < kk; ++l) {
                    const float xv = x[i * kk + l];
                    if (xv == 0.0f) continue;
                    for (std::size_t j = 0; j < n; ++j) o[i * n + j] += xv * y[l * n + j];
                }
            return out(Value::f32(std::move(o)));
        }
        case OpCode::PivotThreshold: {
            const cptir::TensorType& t = ty(a0);
            const std::size_t rows = canonical_rows(t.shape);
            const auto x = lanes_f32(v(a0));
            const std::size_t len = rows == 0 ? 0 : x.size() / rows;
            std::vector<std::uint8_t> keep(x.size(), 0);
            // Predicate payloads are VALUE IDS on the wire for all three tags
            // (interface/ptir container.rs decode) — RankLe included. Already
            // mapped to a global trace id by container_to_trace (bound.hpp
            // gid()), same as any other op operand — no stage-local rebasing
            // needed here.
            for (std::size_t r = 0; r < rows; ++r) {
                const float* row = x.data() + r * len;
                std::uint8_t* k = keep.data() + r * len;
                switch (op.predicate.tag) {
                    case cptir::PredTag::RankLe: {
                        const auto kv = lanes_i64(v(op.predicate.payload));
                        const std::int64_t kk = std::clamp<std::int64_t>(
                            kv[pick(kv.size(), r)], 0, static_cast<std::int64_t>(len));
                        for (std::size_t i = 0; i < len; ++i) {
                            if (std::isnan(row[i])) continue;
                            std::int64_t greater = 0;
                            for (std::size_t j = 0; j < len; ++j)
                                if (!std::isnan(row[j]) && row[j] > row[i]) ++greater;
                            k[i] = greater < kk ? 1 : 0;
                        }
                        break;
                    }
                    case cptir::PredTag::CummassLe: {
                        const auto pv = lanes_f32(v(op.predicate.payload));
                        const float p = pv[pick(pv.size(), r)];
                        const auto order = sort_desc_order(row, len);
                        float excl = 0.0f;
                        for (const auto i : order) {
                            k[i] = excl < p ? 1 : 0;
                            excl += row[i];
                        }
                        break;
                    }
                    case cptir::PredTag::ProbGe: {
                        const auto tv = lanes_f32(v(op.predicate.payload));
                        const float thr = tv[pick(tv.size(), r)];
                        for (std::size_t i = 0; i < len; ++i) k[i] = row[i] >= thr ? 1 : 0;
                        break;
                    }
                }
            }
            return out(Value::boolean(std::move(keep)));
        }

        case OpCode::Gather: {
            const cptir::TensorType& ts = ty(a0);
            std::size_t rest = 1;
            for (std::size_t d = 1; d < ts.shape.dims.size(); ++d) rest *= ts.shape.dims[d];
            const std::size_t n0 = ts.shape.dims.empty() ? 1 : ts.shape.dims[0];
            const auto ix = lanes_i64(v(a1));
            constexpr std::size_t kFill = std::numeric_limits<std::size_t>::max();
            std::vector<std::size_t> flat;
            flat.reserve(ix.size() * rest);
            for (const auto i : ix) {
                if (i >= 0 && static_cast<std::size_t>(i) < n0) {
                    for (std::size_t r = 0; r < rest; ++r)
                        flat.push_back(static_cast<std::size_t>(i) * rest + r);
                } else {
                    flat.insert(flat.end(), rest, kFill);
                }
            }
            return out(gather_flat(v(a0), flat));
        }
        case OpCode::GatherRow: {
            const cptir::TensorType& ts = ty(a0);
            if (ts.shape.dims.size() != 2) return fault("gather_row");
            const std::size_t m = ts.shape.dims[0];
            const std::size_t n = ts.shape.dims[1];
            const auto ix = lanes_i64(v(a1));
            constexpr std::size_t kFill = std::numeric_limits<std::size_t>::max();
            std::vector<std::size_t> flat(m);
            for (std::size_t i = 0; i < m; ++i) {
                const std::int64_t c = ix[i];
                flat[i] = (c >= 0 && static_cast<std::size_t>(c) < n) ? i * n + static_cast<std::size_t>(c)
                                                                      : kFill;
            }
            return out(gather_flat(v(a0), flat));
        }
        case OpCode::ScatterAdd: case OpCode::ScatterSet: {
            const cptir::TensorType& tb = ty(a0);
            std::size_t rest = 1;
            for (std::size_t d = 1; d < tb.shape.dims.size(); ++d) rest *= tb.shape.dims[d];
            const std::size_t n0 = tb.shape.dims.empty() ? 1 : tb.shape.dims[0];
            const auto ix = lanes_i64(v(a1));
            const Value& val = v(a2);
            const bool scalar_val = val.len() == 1 && ix.size() * rest != 1;
            const bool is_add = op.code == OpCode::ScatterAdd;
            if (tb.dtype == DType::F32 || tb.dtype == DType::Act ||
                (is_add && tb.dtype != DType::I32 && tb.dtype != DType::U32)) {
                auto outv = lanes_f32(v(a0));
                const auto vals_f = lanes_f32(val);
                for (std::size_t k = 0; k < ix.size(); ++k) {
                    const std::int64_t i = ix[k];
                    if (i < 0 || static_cast<std::size_t>(i) >= n0) continue;
                    for (std::size_t r = 0; r < rest; ++r) {
                        const float src = scalar_val ? vals_f[0] : vals_f[k * rest + r];
                        float& dst = outv[static_cast<std::size_t>(i) * rest + r];
                        if (is_add) dst += src; else dst = src;
                    }
                }
                return out(Value::f32(std::move(outv)));
            }
            auto outv = lanes_i64(v(a0));
            const auto vals_i = lanes_i64(val);
            for (std::size_t k = 0; k < ix.size(); ++k) {
                const std::int64_t i = ix[k];
                if (i < 0 || static_cast<std::size_t>(i) >= n0) continue;
                for (std::size_t r = 0; r < rest; ++r) {
                    const std::int64_t src = scalar_val ? vals_i[0] : vals_i[k * rest + r];
                    std::int64_t& dst = outv[static_cast<std::size_t>(i) * rest + r];
                    if (is_add) dst += src; else dst = src;
                }
            }
            return out(from_i64(tb.dtype, outv));
        }
        case OpCode::Iota: {
            std::vector<std::uint32_t> o(op.imm);
            for (std::uint32_t j = 0; j < op.imm; ++j) o[j] = j;
            return out(Value::u32(std::move(o)));
        }
        case OpCode::MaskApplyPacked: {
            // Per-row over the LAST axis: bit index is the column, the single
            // packed word row broadcasts across rows.
            const cptir::Shape& ls = ty(a0).shape;
            const std::size_t n = ls.dims.empty() ? 1 : ls.dims.back();
            const auto x = lanes_f32(v(a0));
            const Value& mask = v(a1);
            if (mask.dtype != DType::U32) return fault("mask_apply mask");
            std::vector<float> o(x.size());
            for (std::size_t j = 0; j < x.size(); ++j) {
                const std::size_t c = j % n;
                const std::size_t w = c >> 5;
                const std::uint32_t word = w < mask.u.size() ? mask.u[w] : 0;
                o[j] = ((word >> (c & 31)) & 1u) != 0 ? x[j] : neg_inf();
            }
            return out(Value::f32(std::move(o)));
        }
        case OpCode::CausalMask:
        case OpCode::SlidingWindowMask:
        case OpCode::SinkWindowMask: {
            const Value& positions = v(a0);
            if (positions.dtype != DType::U32) {
                return fault("structured mask positions");
            }
            const std::uint32_t key_count = op.imm;
            const std::uint32_t window =
                op.code == OpCode::SlidingWindowMask
                    ? op.imm2
                    : op.imm3;
            auto saturating_add = [](
                                      std::uint32_t left,
                                      std::uint32_t right) {
                return left >
                               std::numeric_limits<std::uint32_t>::max() -
                                   right
                           ? std::numeric_limits<std::uint32_t>::max()
                           : left + right;
            };
            std::vector<std::uint8_t> mask;
            mask.reserve(
                positions.u.size() *
                static_cast<std::size_t>(key_count));
            for (const std::uint32_t position : positions.u) {
                for (std::uint32_t key = 0; key < key_count; ++key) {
                    bool allowed = key <= position;
                    if (allowed &&
                        op.code != OpCode::CausalMask) {
                        const bool recent =
                            saturating_add(key, window) > position;
                        allowed =
                            op.code == OpCode::SlidingWindowMask
                                ? recent
                                : (key < op.imm2 || recent);
                    }
                    mask.push_back(allowed ? 1 : 0);
                }
            }
            return out(Value::boolean(std::move(mask)));
        }
        case OpCode::Rng: {
            // Ambient-seed form: per-fire seed 0 in the reference interpreter.
            const std::uint64_t seed_eff =
                ptir_rng_seed_eff_stream(0, op.imm);
            const std::size_t n = static_cast<std::size_t>(op.result_type.shape.numel());
            return out(Value::f32(rng_lanes(seed_eff, n, op.rng_kind == cptir::RngKind::Gumbel)));
        }
        case OpCode::RngKeyed: {
            const auto st = lanes_i64(v(a0));
            const std::uint64_t key = static_cast<std::uint64_t>(st[0]) & 0xFFFFFFFFULL;
            const std::uint64_t ctr = st.size() > 1 ? static_cast<std::uint64_t>(st[1]) & 0xFFFFFFFFULL : 0;
            const std::uint64_t seed64 = ptir_rng_keyed_seed(
                static_cast<std::uint32_t>(key),
                static_cast<std::uint32_t>(ctr));
            const std::size_t n = static_cast<std::size_t>(op.result_type.shape.numel());
            return out(Value::f32(rng_lanes(seed64, n, op.rng_kind == cptir::RngKind::Gumbel)));
        }
        case OpCode::KernelCall:
            if (op.args.size() != 1) {
                return fault("Metal identity boundary arity");
            }
            return out(v(a0));
        case OpCode::SinkCall:
            return true;

        default:
            return fault(std::string("op not executable on the Metal host interpreter: ") +
                         std::string(cptir::op_name(op.code)));
    }
}

inline bool exec_stage(InterpInstance& inst, const ExecPlan& plan, const StagePlan& sp,
                       const PassInputs& in, Overlay& ov, std::vector<Value>& vals,
                       std::string& error) {
    const cptir::Stage& stage = plan.trace.stages[sp.stage_index];
    for (std::uint32_t id = sp.base; id < sp.end;) {
        auto found = sp.op_by_result.find(id);
        if (found != sp.op_by_result.end()) {
            if (!eval_op(*found->second, plan.trace, vals, error)) return false;
            id += found->second->result_count;
            continue;
        }
        const cptir::Value& root = plan.trace.values[id];
        switch (root.source) {
            case cptir::ValueSource::Const: {
                switch (root.lit.dtype) {
                    case DType::I32: vals[id] = Value::i32({root.lit.as_i32()}); break;
                    case DType::U32: vals[id] = Value::u32({root.lit.as_u32()}); break;
                    case DType::Bool: vals[id] = Value::boolean({root.lit.as_bool() ? std::uint8_t{1} : std::uint8_t{0}}); break;
                    default: vals[id] = Value::f32({root.lit.as_f32()}); break;
                }
                break;
            }
            case cptir::ValueSource::ChannelTake:
                vals[id] = ov.take(inst, root.channel);
                break;
            case cptir::ValueSource::ChannelRead:
                vals[id] = ov.resolve(inst, root.channel);
                break;
            case cptir::ValueSource::Intrinsic: {
                // Only bounded logits/MTP roots reach here — classify_exec_plan()
                // rejects every other intrinsic before a program is
                // launchable, so any other tag means a plan/interp drift.
                if (root.intrinsic != cptir::Intrinsic::Logits &&
                    root.intrinsic != cptir::Intrinsic::MtpLogits &&
                    root.intrinsic != cptir::Intrinsic::MtpDrafts) {
                    error = "unresolved value root (unsupported intrinsic) reached execution";
                    return false;
                }
                if (in.logits == nullptr || in.vocab == 0) {
                    error = "logits intrinsic unbound (forward did not run before step)";
                    return false;
                }
                const std::uint64_t want = std::max<std::uint64_t>(root.type.shape.numel(), 1);
                const bool drafts =
                    root.intrinsic == cptir::Intrinsic::MtpDrafts;
                const std::uint64_t rows_needed =
                    drafts ? want : (want / in.vocab);
                if (!drafts && want % in.vocab != 0) {
                    error = "logits intrinsic shape mismatch (program vocab != model vocab)";
                    return false;
                }
                std::uint32_t base_row = 0;
                if ((root.intrinsic == cptir::Intrinsic::MtpLogits ||
                     drafts) &&
                    in.mtp_draft_row >= 0) {
                    base_row = static_cast<std::uint32_t>(in.mtp_draft_row);
                }
                if (static_cast<std::uint64_t>(base_row) + rows_needed > in.rows) {
                    error = "logits intrinsic row range exceeds the forward's readout rows";
                    return false;
                }
                if (drafts) {
                    std::vector<std::int32_t> tokens(want, 0);
                    for (std::size_t row = 0; row < want; ++row) {
                        const float* logits =
                            in.logits +
                            (static_cast<std::size_t>(base_row) + row) *
                                in.vocab;
                        bool have = false;
                        float best = neg_inf();
                        std::uint32_t best_index = 0;
                        for (std::uint32_t column = 0;
                             column < in.vocab;
                             ++column) {
                            const float value = logits[column];
                            if (!std::isnan(value) &&
                                (!have || value > best ||
                                 (value == best &&
                                  column < best_index))) {
                                have = true;
                                best = value;
                                best_index = column;
                            }
                        }
                        tokens[row] =
                            static_cast<std::int32_t>(best_index);
                    }
                    vals[id] = Value::i32(std::move(tokens));
                } else {
                    std::vector<float> lane(want);
                    std::memcpy(
                        lane.data(),
                        in.logits +
                            static_cast<std::size_t>(base_row) * in.vocab,
                        want * sizeof(float));
                    vals[id] = Value::f32(std::move(lane));
                }
                break;
            }
            default:
                error = "unresolved value root (intrinsic/host input) reached execution";
                return false;
        }
        id += 1;
    }
    // Puts land on the pass overlay at stage end (register semantics: the
    // Trace model carries no put position within the stage; last put wins).
    for (const cptir::ChannelPut& p : stage.puts) {
        ov.pending.insert_or_assign(p.channel, vals[p.value]);
        ov.put[p.channel] = 1;
    }
    return true;
}

}  // namespace detail

// Execute one pass: readiness → prologue → descriptor ports → epilogue →
// predicated commit (interp.rs step, minus the per-layer taps this
// increment rejects at classification). `in` binds the Intrinsic(Logits)/
// Intrinsic(MtpLogits) roots for C2 (forward-needing) plans; C1
// (channel-plane-only) callers pass `PassInputs{}` (never dereferenced since
// `plan.trace` roots no Intrinsic value in that case).
inline StepResult step(InterpInstance& inst, const ExecPlan& plan, const PassInputs& in = {}) {
    StepResult result;
    if (inst.poisoned) {
        result.error = "instance is poisoned";
        return result;
    }

    for (const auto& e : plan.bound.readiness) {
        const ChannelState& st = *inst.channels[e.chan];
        const bool ok = e.dir == cptir::container::Direction::NeedsFull
                            ? !st.empty()
                            : !st.full();
        if (!ok) {
            result.ok = true;
            result.committed = false;
            result.missed_channel = e.chan;
            return result;
        }
    }

    detail::Overlay ov;
    ov.taken.assign(inst.channels.size(), 0);
    ov.put.assign(inst.channels.size(), 0);
    std::vector<Value> vals(plan.trace.values.size());

    auto run_kind = [&](cptir::StageKind kind) -> bool {
        for (const StagePlan& sp : plan.stages) {
            if (plan.trace.stages[sp.stage_index].kind != kind) continue;
            if (!detail::exec_stage(inst, plan, sp, in, ov, vals, result.error)) return false;
        }
        return true;
    };

    if (!run_kind(cptir::StageKind::Prologue)) {
        inst.poisoned = true;
        return result;
    }
    for (const cptir::PortBinding& p : plan.trace.ports) {
        if (p.is_const) continue;
        if (cptir::port_consumes(p.port)) {
            (void)ov.take(inst, p.channel);
        }
        // Port values feed the forward, which this increment does not run.
    }
    if (!run_kind(cptir::StageKind::Epilogue)) {
        inst.poisoned = true;
        return result;
    }

    // Validate every resulting ring state before writing a pending cell or
    // publishing any head/tail word. This keeps a semantic fault pass-atomic.
    std::vector<std::uint64_t> old_tails(inst.channels.size(), 0);
    std::vector<std::uint64_t> new_heads(inst.channels.size(), 0);
    std::vector<std::uint64_t> new_tails(inst.channels.size(), 0);
    for (std::size_t ci = 0; ci < inst.channels.size(); ++ci) {
        ChannelState& st = *inst.channels[ci];
        const std::uint64_t head = st.head();
        const std::uint64_t tail = st.tail();
        if (tail < head) {
            inst.poisoned = true;
            result.error = "channel " + std::to_string(ci) +
                           ": tail precedes head at commit";
            return result;
        }
        std::uint64_t next_head = head;
        std::uint64_t next_tail = tail;
        std::uint64_t used = tail - head;
        if (ov.taken[ci] && used != 0) {
            ++next_head;
            --used;
        }
        if (ov.put[ci]) {
            if (used >= st.capacity) {
                inst.poisoned = true;
                result.error = "channel " + std::to_string(ci) +
                               ": put overflows capacity " +
                               std::to_string(st.capacity) + " at commit";
                return result;
            }
            ++next_tail;
        }
        old_tails[ci] = tail;
        new_heads[ci] = next_head;
        new_tails[ci] = next_tail;
    }

    // Pending cells become visible only when the corresponding tail word is
    // release-published below.
    for (std::size_t ci = 0; ci < inst.channels.size(); ++ci) {
        if (ov.put[ci]) {
            inst.channels[ci]->encode_sequence(
                old_tails[ci],
                ov.pending.at(static_cast<std::uint32_t>(ci)));
        }
    }
    for (std::size_t ci = 0; ci < inst.channels.size(); ++ci) {
        ChannelState& st = *inst.channels[ci];
        if (new_heads[ci] != st.head()) st.store_word(0, new_heads[ci]);
        if (new_tails[ci] != st.tail()) st.store_word(1, new_tails[ci]);
    }

    result.ok = true;
    result.committed = true;
    return result;
}

}  // namespace pie::metal::pipeline
