#pragma once

// PTIR host interpreter — the Metal driver's CPU execution of channel-plane
// PTIR programs, a C++ mirror of the canonical reference interpreter
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
// OpResult / Intrinsic(Logits) / Intrinsic(MtpLogits) — the latter two need
// a forward pass first (`ExecPlan::needs_logits` / `needs_mtp_logits`,
// `PassInputs`, §5.3). Host inputs, per-layer taps, and other model
// intrinsics (hidden/query/value-head) are not executable here — they need
// backend feature work this increment does not add.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <deque>
#include <limits>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "pie_native/ptir/bound.hpp"
#include "pie_native/ptir/container.hpp"
#include "pie_native/ptir/op_table.hpp"
#include "pie_native/ptir/trace.hpp"

namespace pie_metal_driver::ptir_host {

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

    // True if the launch path must run the forward before `step()`.
    bool needs_forward() const { return needs_logits || needs_mtp_logits; }
};

// Classify an already-translated Trace: split "rejected" (executable=false,
// a precise reason) from "needs forward inputs" (needs_logits/
// needs_mtp_logits, still executable). Pure function over the decoded Trace
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
                    out.needs_mtp_logits = true;
                    break;
                default:
                    // Hidden / Query / ValueHead (and any intrinsic tag the
                    // shared bound.hpp does not map — it falls back to
                    // Logits there, not here) — the Metal forward exposes no
                    // per-layer taps or auxiliary heads in this increment.
                    out.executable = false;
                    out.reject_reason =
                        "program reads an unsupported model intrinsic (hidden/query/"
                        "value-head; Metal forward not wired)";
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

// Decode + classify. Registration never fails on executability — the launch
// path rejects with `reject_reason` instead.
inline bool build_exec_plan(const std::uint8_t* container_bytes, std::size_t container_len,
                            const std::uint8_t* sidecar_bytes, std::size_t sidecar_len,
                            ExecPlan& out, std::string* error) {
    cptir::container::Container c;
    cptir::container::DecodeError derr;
    if (!cptir::container::decode(container_bytes, container_len, c, &derr)) {
        if (error != nullptr) *error = "container decode: " + derr.detail;
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
    auto translated = cptir::bound::container_to_trace(c, out.bound);
    if (!translated.ok) {
        out.executable = false;
        out.reject_reason = translated.error;
        return true;
    }
    out.trace = std::move(translated.trace);

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
    return true;
}

// ─────────────────────────── instance state ────────────────────────────────

// One channel's committed ring (interp.rs ChannelState). Extern channels
// share one state across the exporting and importing instance.
struct ChannelState {
    std::deque<Value> queue;
    std::size_t capacity = 1;
    Value last;
};

struct InterpInstance {
    std::vector<std::shared_ptr<ChannelState>> channels;
    bool poisoned = false;
};

// Bind fresh channel state. `externs[dense]` (optional) supplies the shared
// ring; `seeds[dense]` (optional) pre-fills one committed cell.
inline InterpInstance make_instance(const ExecPlan& plan,
                                    const std::map<std::uint32_t, std::shared_ptr<ChannelState>>& externs,
                                    const std::map<std::uint32_t, Value>& seeds) {
    InterpInstance inst;
    for (std::size_t ci = 0; ci < plan.trace.channels.size(); ++ci) {
        const cptir::Channel& decl = plan.trace.channels[ci];
        auto shared = externs.find(static_cast<std::uint32_t>(ci));
        if (shared != externs.end()) {
            inst.channels.push_back(shared->second);
            continue;
        }
        auto st = std::make_shared<ChannelState>();
        st->capacity = decl.capacity;
        const DType dt = decl.type.dtype == DType::Act ? DType::F32 : decl.type.dtype;
        st->last = zeros(dt, decl.type.shape.numel());
        auto seed = seeds.find(static_cast<std::uint32_t>(ci));
        if (seed != seeds.end()) st->queue.push_back(seed->second);
        inst.channels.push_back(std::move(st));
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
    if (st.queue.size() >= st.capacity) return HostOp::WouldBlock;
    st.queue.push_back(std::move(v));
    return HostOp::Ok;
}

inline HostOp host_take(InterpInstance& inst, const ExecPlan& plan, std::uint32_t chan, Value& out) {
    if (inst.poisoned) return HostOp::Poisoned;
    const cptir::Channel& decl = plan.trace.channels[chan];
    if (!decl.host_reader) return HostOp::WrongRole;
    ChannelState& st = *inst.channels[chan];
    if (st.queue.empty()) return HostOp::WouldBlock;
    out = std::move(st.queue.front());
    st.queue.pop_front();
    st.last = out;
    return HostOp::Ok;
}

// ───────────────────────────── op evaluation ────────────────────────────────

namespace detail {

inline float neg_inf() { return -std::numeric_limits<float>::infinity(); }

// argmax pinned contract: lower index wins ties; NaN never selected
// (all-NaN row → 0).
inline std::int32_t argmax_row(const float* row, std::size_t len) {
    float best = 0.0f;
    bool have = false;
    std::size_t bi = 0;
    for (std::size_t j = 0; j < len; ++j) {
        const float x = row[j];
        if (!std::isnan(x) && (!have || x > best)) {
            best = x;
            have = true;
            bi = j;
        }
    }
    return static_cast<std::int32_t>(bi);
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

inline std::uint64_t splitmix64(std::uint64_t x) {
    x ^= x >> 27;
    x *= 0x3C79AC492BA7B653ULL;
    x ^= x >> 33;
    x *= 0x1C69B3F74AC4AE35ULL;
    x ^= x >> 27;
    return x;
}

inline float hash_uniform(std::uint64_t seed_eff, std::uint32_t j) {
    const std::uint64_t x = seed_eff + 0x9E3779B97F4A7C15ULL * (static_cast<std::uint64_t>(j) + 1);
    const std::uint32_t bits = static_cast<std::uint32_t>(splitmix64(x) >> 40);
    return (static_cast<float>(bits) + 0.5f) * (1.0f / 16777216.0f);
}

inline std::vector<float> rng_lanes(std::uint64_t seed_eff, std::size_t n, bool gumbel) {
    std::vector<float> o(n);
    for (std::size_t j = 0; j < n; ++j) {
        const float u = hash_uniform(seed_eff, static_cast<std::uint32_t>(j));
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
        return st.queue.empty() ? st.last : st.queue.front();
    }
    Value take(const InterpInstance& inst, std::uint32_t chan) {
        taken[chan] = 1;
        return resolve(inst, chan);
    }
};

// Evaluate one compute op (SSA args already in `vals`). Mirrors interp.rs
// eval_op case for case; returns false + `error` on a semantic fault.
inline bool eval_op(const cptir::Op& op, const Trace& trace, std::vector<Value>& vals,
                    std::string& error, std::uint32_t stage_base = 0) {
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
            const std::size_t rows = t.shape.rows();
            const Value& data = v(a0);
            const std::size_t len = rows == 0 ? 0 : data.len() / rows;
            if (t.dtype == DType::F32 || t.dtype == DType::Act) {
                const auto x = lanes_f32(data);
                std::vector<float> o(rows);
                for (std::size_t r = 0; r < rows; ++r) {
                    const float* row = x.data() + r * len;
                    float acc;
                    if (op.code == OpCode::ReduceSum) {
                        acc = 0.0f;
                        for (std::size_t j = 0; j < len; ++j) acc += row[j];
                    } else if (op.code == OpCode::ReduceMax) {
                        acc = neg_inf();
                        for (std::size_t j = 0; j < len; ++j) acc = std::fmax(acc, row[j]);
                    } else {
                        acc = std::numeric_limits<float>::infinity();
                        for (std::size_t j = 0; j < len; ++j) acc = std::fmin(acc, row[j]);
                    }
                    o[r] = acc;
                }
                return out(Value::f32(std::move(o)));
            }
            const auto x = lanes_i64(data);
            std::vector<std::int64_t> o(rows);
            for (std::size_t r = 0; r < rows; ++r) {
                const std::int64_t* row = x.data() + r * len;
                std::int64_t acc = 0;
                if (op.code == OpCode::ReduceSum) {
                    for (std::size_t j = 0; j < len; ++j) acc += row[j];
                } else if (len == 0) {
                    acc = 0;
                } else if (op.code == OpCode::ReduceMax) {
                    acc = row[0];
                    for (std::size_t j = 1; j < len; ++j) acc = std::max(acc, row[j]);
                } else {
                    acc = row[0];
                    for (std::size_t j = 1; j < len; ++j) acc = std::min(acc, row[j]);
                }
                o[r] = acc;
            }
            return out(from_i64(t.dtype, o));
        }
        case OpCode::ReduceArgmax: {
            const cptir::TensorType& t = ty(a0);
            const std::size_t rows = t.shape.rows();
            const auto x = lanes_f32(v(a0));
            const std::size_t len = rows == 0 ? 0 : x.size() / rows;
            std::vector<std::int32_t> o(rows);
            for (std::size_t r = 0; r < rows; ++r) o[r] = argmax_row(x.data() + r * len, len);
            return out(Value::i32(std::move(o)));
        }
        case OpCode::CumSum: case OpCode::CumProd: {
            const cptir::TensorType& t = ty(a0);
            const std::size_t rows = t.shape.rows();
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
            const std::size_t rows = t.shape.rows();
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
            const std::size_t rows = t.shape.rows();
            const auto x = lanes_f32(v(a0));
            const std::size_t len = rows == 0 ? 0 : x.size() / rows;
            std::vector<std::uint8_t> keep(x.size(), 0);
            // Predicate payloads are VALUE IDS on the wire for all three tags
            // (interface/ptir container.rs decode) — RankLe included.
            for (std::size_t r = 0; r < rows; ++r) {
                const float* row = x.data() + r * len;
                std::uint8_t* k = keep.data() + r * len;
                switch (op.predicate.tag) {
                    case cptir::PredTag::RankLe: {
                        const auto kv = lanes_i64(v(stage_base + op.predicate.payload));
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
                        const auto pv = lanes_f32(v(stage_base + op.predicate.payload));
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
                        const auto tv = lanes_f32(v(stage_base + op.predicate.payload));
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
        case OpCode::Rng: {
            // Ambient-seed form: per-fire seed 0 in the reference interpreter.
            const std::uint64_t salt = splitmix64(static_cast<std::uint64_t>(op.imm) *
                                                  0x9E3779B97F4A7C15ULL);
            const std::uint64_t seed_eff = 0xA5A5A5A5ULL ^ salt;
            const std::size_t n = static_cast<std::size_t>(op.result_type.shape.numel());
            return out(Value::f32(rng_lanes(seed_eff, n, op.rng_kind == cptir::RngKind::Gumbel)));
        }
        case OpCode::RngKeyed: {
            const auto st = lanes_i64(v(a0));
            const std::uint64_t key = static_cast<std::uint64_t>(st[0]) & 0xFFFFFFFFULL;
            const std::uint64_t ctr = st.size() > 1 ? static_cast<std::uint64_t>(st[1]) & 0xFFFFFFFFULL : 0;
            const std::uint64_t seed64 = splitmix64((key << 32) | ctr);
            const std::size_t n = static_cast<std::size_t>(op.result_type.shape.numel());
            return out(Value::f32(rng_lanes(seed64, n, op.rng_kind == cptir::RngKind::Gumbel)));
        }

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
            if (!eval_op(*found->second, plan.trace, vals, error, sp.base)) return false;
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
                // Only Logits/MtpLogits reach here — classify_exec_plan()
                // rejects every other intrinsic before a program is
                // launchable, so any other tag means a plan/interp drift.
                if (root.intrinsic != cptir::Intrinsic::Logits &&
                    root.intrinsic != cptir::Intrinsic::MtpLogits) {
                    error = "unresolved value root (unsupported intrinsic) reached execution";
                    return false;
                }
                if (in.logits == nullptr || in.vocab == 0) {
                    error = "logits intrinsic unbound (forward did not run before step)";
                    return false;
                }
                const std::uint64_t want = std::max<std::uint64_t>(root.type.shape.numel(), 1);
                if (want % in.vocab != 0) {
                    error = "logits intrinsic shape mismatch (program vocab != model vocab)";
                    return false;
                }
                const std::uint64_t rows_needed = want / in.vocab;
                std::uint32_t base_row = 0;
                if (root.intrinsic == cptir::Intrinsic::MtpLogits && in.mtp_draft_row >= 0) {
                    base_row = static_cast<std::uint32_t>(in.mtp_draft_row);
                }
                if (static_cast<std::uint64_t>(base_row) + rows_needed > in.rows) {
                    error = "logits intrinsic row range exceeds the forward's readout rows";
                    return false;
                }
                std::vector<float> lane(want);
                std::memcpy(lane.data(),
                           in.logits + static_cast<std::size_t>(base_row) * in.vocab,
                           want * sizeof(float));
                vals[id] = Value::f32(std::move(lane));
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
                            ? !st.queue.empty()
                            : st.queue.size() < st.capacity;
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

    for (std::size_t ci = 0; ci < inst.channels.size(); ++ci) {
        ChannelState& st = *inst.channels[ci];
        if (ov.taken[ci] && !st.queue.empty()) {
            st.last = std::move(st.queue.front());
            st.queue.pop_front();
        }
        if (ov.put[ci]) {
            if (st.queue.size() >= st.capacity) {
                // A non-leading put into a still-full ring — device fault.
                inst.poisoned = true;
                result.error = "channel " + std::to_string(ci) + ": put overflows capacity " +
                               std::to_string(st.capacity) + " at commit";
                return result;
            }
            st.queue.push_back(std::move(ov.pending.at(static_cast<std::uint32_t>(ci))));
        }
    }

    result.ok = true;
    result.committed = true;
    return result;
}

}  // namespace pie_metal_driver::ptir_host
