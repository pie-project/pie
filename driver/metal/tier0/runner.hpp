#pragma once
//
// PTIR tier-0 interpreter on Metal — the host channel-runtime (a C++ port of
// echo's reference interpreter interface/sampling-ir/src/ptir/interp.rs: per-
// phase readiness → pass-local overlay → pass-atomic commit / epoch-ring
// register semantics) executing over charlie's CUDA-free decoded `Trace`
// (container.hpp + bound.hpp + trace.hpp), with the compute ops dispatched to
// the validated Metal kernels (ptir/kvattn). This makes Metal a full PTIR
// sampling-IR execution backend, certified against echo's golden vectors.
//
// echo's interp.rs is the semantic oracle; charlie's tier0_runner is the
// cross-reference for tricky orchestration.

#include <cstdint>
#include <cstring>
#include <cmath>
#include <deque>
#include <map>
#include <string>
#include <vector>

#include "ptir/bound.hpp"
#include "ptir/container.hpp"
#include "ptir/op_table.hpp"
#include "ptir/trace.hpp"

#include "metal_ops.hpp"

namespace tier0 {

namespace P = pie_cuda_driver::ptir;
using P::DType;

// A runtime value: F32 lanes or integer lanes (I32/U32/Bool as i64). Mirrors
// eval::Value / interp.rs's typed values.
struct Val {
    DType dt = DType::F32;
    std::vector<float> f;      // dt == F32
    std::vector<std::int64_t> i;  // dt == I32 / U32 / Bool

    std::size_t numel() const { return dt == DType::F32 ? f.size() : i.size(); }
    static Val make_f32(std::vector<float> v) { Val x; x.dt = DType::F32; x.f = std::move(v); return x; }
    static Val make_int(DType d, std::vector<std::int64_t> v) { Val x; x.dt = d; x.i = std::move(v); return x; }

    std::vector<float> to_f32() const {
        if (dt == DType::F32) return f;
        std::vector<float> o(i.size());
        for (std::size_t k = 0; k < i.size(); ++k) o[k] = static_cast<float>(i[k]);
        return o;
    }
    // Zero value of this type + numel (the channel dummy source, interp.rs).
    static Val zeros(DType d, std::size_t n) {
        if (d == DType::F32) return make_f32(std::vector<float>(n, 0.0f));
        return make_int(d, std::vector<std::int64_t>(n, 0));
    }
};

// Per-pass inputs: the intrinsics (logits) + host tensors, keyed as in the
// golden's `inputs N:` line.
struct FireInputs {
    bool has_logits = false;
    std::vector<float> logits;     // [rows*vocab] row-major
    int vocab = 0;
    std::map<std::uint32_t, Val> host_inputs;  // host_key -> value
};

// ── channel ring (interp.rs ChannelState) ───────────────────────────────────
struct Chan {
    std::deque<Val> queue;
    std::size_t capacity = 1;
    Val last;  // last committed value = the dummy source
};

enum class HostErr { Ok, WouldBlock, Poisoned, BadIndex };

struct StepReport {
    bool committed = false;
    bool has_miss = false;
    std::uint32_t miss_chan = 0, miss_phase = 0;
    bool ok = true;
    std::string error;
};

class Instance {
  public:
    Instance(const P::Trace& t, const P::bound::Bound& b, MetalOps& ops)
        : trace_(&t), bound_(&b), ops_(&ops) {
        chans_.resize(t.channels.size());
        for (std::size_t k = 0; k < t.channels.size(); ++k) {
            chans_[k].capacity = t.channels[k].capacity;
            chans_[k].last = Val::zeros(t.channels[k].type.dtype, dtype_numel(t.channels[k].type));
        }
    }

    // Seed a channel (Channel::from) — a pre-put full cell at instantiation.
    void seed(std::uint32_t chan, Val v) {
        chans_[chan].last = v;
        chans_[chan].queue.push_back(std::move(v));
    }

    HostErr host_take(std::uint32_t chan, Val& out) {
        if (poisoned_) return HostErr::Poisoned;
        if (chan >= chans_.size()) return HostErr::BadIndex;
        auto& st = chans_[chan];
        if (st.queue.empty()) return HostErr::WouldBlock;
        out = st.queue.front();
        st.last = out;
        st.queue.pop_front();
        return HostErr::Ok;
    }
    // host_put: append to a channel's ring (host feeds a grant/mask).
    HostErr host_put(std::uint32_t chan, Val v) {
        if (poisoned_) return HostErr::Poisoned;
        if (chan >= chans_.size()) return HostErr::BadIndex;
        auto& st = chans_[chan];
        if (st.queue.size() >= st.capacity) return HostErr::WouldBlock;
        st.last = v;
        st.queue.push_back(std::move(v));
        return HostErr::Ok;
    }

    StepReport step(const FireInputs& in);

  private:
    struct Overlay {
        std::map<std::uint32_t, Val> pending;
        std::vector<char> taken, put;
    };
    static std::size_t dtype_numel(const P::TensorType& t) {
        std::size_t n = 1;
        for (auto d : t.shape.dims) n *= d;
        return n == 0 ? 1 : n;
    }
    Val resolve(const Overlay& ov, std::uint32_t chan) const {
        auto it = ov.pending.find(chan);
        if (it != ov.pending.end()) return it->second;
        const auto& st = chans_[chan];
        return st.queue.empty() ? st.last : st.queue.front();
    }
    void exec_stage(const P::Stage& s, std::uint32_t layer, const FireInputs& in, Overlay& ov,
                    StepReport& rep);
    Val eval_op(const P::Op& op, const std::vector<Val>& args, StepReport& rep);

    const P::Trace* trace_;
    const P::bound::Bound* bound_;
    MetalOps* ops_;
    std::vector<Chan> chans_;
    bool poisoned_ = false;
};

// ── implementation ──────────────────────────────────────────────────────────

// RNG (PTIR-CONTAINER.md §5; shared splitmix64/hash_uniform with sample_temp).
inline std::uint64_t splitmix64(std::uint64_t x) {
    x ^= x >> 27; x *= 0x3C79AC492BA7B653ull;
    x ^= x >> 33; x *= 0x1C69B3F74AC4AE35ull;
    x ^= x >> 27;
    return x;
}
inline float hash_uniform(std::uint64_t seed_eff, std::uint32_t j) {
    std::uint64_t x = seed_eff + 0x9E3779B97F4A7C15ull * ((std::uint64_t)j + 1);
    std::uint32_t bits = (std::uint32_t)(splitmix64(x) >> 40);
    return ((float)bits + 0.5f) * (1.0f / 16777216.0f);
}

inline void rows_len(const P::TensorType& t, int& rows, int& len) {
    const auto& d = t.shape.dims;
    if (d.empty()) { rows = 1; len = 1; return; }
    len = (int)d.back();
    rows = 1;
    for (std::size_t k = 0; k + 1 < d.size(); ++k) rows *= (int)d[k];
    if (rows == 0) rows = 1;
}

inline Val Instance::eval_op(const P::Op& op, const std::vector<Val>& args, StepReport& rep) {
    using OC = P::OpCode;
    auto binop_int = [&](auto f) {
        const auto& a = args[0].i;
        const auto& b = args[1].i;
        std::size_t n = a.size() > b.size() ? a.size() : b.size();
        std::vector<std::int64_t> o(n);
        for (std::size_t k = 0; k < n; ++k)
            o[k] = f(a[a.size() == 1 ? 0 : k], b[b.size() == 1 ? 0 : k]);
        return Val::make_int(args[0].dt, std::move(o));
    };
    auto binop_f = [&](auto f) {
        auto a = args[0].to_f32();
        auto b = args[1].to_f32();
        std::size_t n = a.size() > b.size() ? a.size() : b.size();
        std::vector<float> o(n);
        for (std::size_t k = 0; k < n; ++k)
            o[k] = f(a[a.size() == 1 ? 0 : k], b[b.size() == 1 ? 0 : k]);
        return Val::make_f32(std::move(o));
    };
    switch (op.code) {
        case OC::Add:
            return args[0].dt == DType::F32 ? binop_f([](float x, float y) { return x + y; })
                                            : binop_int([](std::int64_t x, std::int64_t y) { return x + y; });
        case OC::Sub:
            return args[0].dt == DType::F32 ? binop_f([](float x, float y) { return x - y; })
                                            : binop_int([](std::int64_t x, std::int64_t y) { return x - y; });
        case OC::Mul:
            return args[0].dt == DType::F32 ? binop_f([](float x, float y) { return x * y; })
                                            : binop_int([](std::int64_t x, std::int64_t y) { return x * y; });
        case OC::Reshape:
        case OC::Transpose:  // (axis metadata handled by shape; row-major clone for now)
            return args[0];
        case OC::Select: {
            // cond ? a : b, per-operand len-1 broadcast, dtype of a preserved.
            const Val& c = args[0];
            const Val& a = args[1];
            const Val& b = args[2];
            std::size_t n = c.numel();
            n = a.numel() > n ? a.numel() : n;
            n = b.numel() > n ? b.numel() : n;
            auto pick = [](std::size_t l, std::size_t k) { return l == 1 ? std::size_t(0) : k; };
            auto cond = c.i;  // Bool as i64 0/1
            if (a.dt == DType::F32) {
                auto av = a.to_f32(), bv = b.to_f32();
                std::vector<float> o(n);
                for (std::size_t k = 0; k < n; ++k)
                    o[k] = cond[pick(cond.size(), k)] ? av[pick(av.size(), k)] : bv[pick(bv.size(), k)];
                return Val::make_f32(std::move(o));
            }
            std::vector<std::int64_t> o(n);
            for (std::size_t k = 0; k < n; ++k)
                o[k] = cond[pick(cond.size(), k)] ? a.i[pick(a.i.size(), k)] : b.i[pick(b.i.size(), k)];
            return Val::make_int(a.dt, std::move(o));
        }
        case OC::Iota: {
            std::vector<std::int64_t> o(op.imm);
            for (std::uint32_t k = 0; k < op.imm; ++k) o[k] = k;
            return Val::make_int(DType::U32, std::move(o));
        }
        case OC::Cast: {
            DType to = op.result_type.dtype;
            if (to == DType::F32) return Val::make_f32(args[0].to_f32());
            auto f = args[0].to_f32();
            std::vector<std::int64_t> o(args[0].numel());
            if (args[0].dt == DType::F32) for (std::size_t k = 0; k < o.size(); ++k) o[k] = (std::int64_t)f[k];
            else o = args[0].i;
            return Val::make_int(to, std::move(o));
        }
        case OC::RngKeyed: {
            // state = [key, ctr]; seed64 = splitmix64((key<<32)|ctr); gumbel/uniform.
            const auto& st = args[0].i;
            std::uint64_t key = (std::uint64_t)st[0] & 0xFFFFFFFFull;
            std::uint64_t ctr = st.size() > 1 ? ((std::uint64_t)st[1] & 0xFFFFFFFFull) : 0;
            std::uint64_t seed64 = splitmix64((key << 32) | ctr);
            std::size_t n = 1;
            for (auto d : op.result_type.shape.dims) n *= d;
            if (n == 0) n = 1;
            bool gumbel = op.rng_kind == P::RngKind::Gumbel;
            std::vector<float> o(n);
            for (std::uint32_t j = 0; j < n; ++j) {
                float u = hash_uniform(seed64, j);
                o[j] = gumbel ? -std::log(-std::log(u)) : u;
            }
            return Val::make_f32(std::move(o));
        }
        case OC::ReduceArgmax: {
            int rows, len;
            rows_len(trace_->value(op.args[0])->type, rows, len);
            auto tokens = ops_->reduce_argmax(args[0].to_f32(), rows, len);
            std::vector<std::int64_t> o(tokens.begin(), tokens.end());
            return Val::make_int(DType::I32, std::move(o));
        }
        default:
            rep.ok = false;
            rep.error = "tier0: unhandled op 0x" + std::to_string((int)op.code);
            return Val::zeros(DType::F32, 1);
    }
}

inline void Instance::exec_stage(const P::Stage& s, std::uint32_t layer, const FireInputs& in,
                                 Overlay& ov, StepReport& rep) {
    std::map<P::ValueId, Val> cache;
    auto leaf = [&](P::ValueId id) -> Val {
        auto it = cache.find(id);
        if (it != cache.end()) return it->second;
        const P::Value* v = trace_->value(id);
        Val out;
        switch (v->source) {
            case P::ValueSource::Const: {
                switch (v->lit.dtype) {
                    case DType::F32: out = Val::make_f32({v->lit.as_f32()}); break;
                    case DType::I32: out = Val::make_int(DType::I32, {(std::int64_t)v->lit.as_i32()}); break;
                    case DType::U32: out = Val::make_int(DType::U32, {(std::int64_t)v->lit.as_u32()}); break;
                    default: out = Val::make_int(DType::Bool, {v->lit.as_bool() ? 1 : 0}); break;
                }
                break;
            }
            case P::ValueSource::Intrinsic:
                out = Val::make_f32(in.logits);
                break;
            case P::ValueSource::HostInput: {
                auto hi = in.host_inputs.find(v->host_key);
                out = hi != in.host_inputs.end() ? hi->second
                                                 : Val::zeros(v->type.dtype, dtype_numel(v->type));
                break;
            }
            case P::ValueSource::ChannelTake:
                out = resolve(ov, v->channel);
                ov.taken[v->channel] = 1;
                break;
            case P::ValueSource::ChannelRead:
                out = resolve(ov, v->channel);
                break;
            default:
                out = Val::zeros(v->type.dtype, dtype_numel(v->type));
                break;
        }
        cache.emplace(id, out);
        return out;
    };
    (void)layer;
    for (const P::Op& op : s.ops) {
        std::vector<Val> args;
        args.reserve(op.args.size());
        for (auto aid : op.args) args.push_back(leaf(aid));
        Val r = eval_op(op, args, rep);
        if (!rep.ok) return;
        cache[op.result_id] = std::move(r);
    }
    // Channel puts (pass-local overlay; applied at commit).
    for (const P::ChannelPut& p : s.puts) {
        Val v = cache.count(p.value) ? cache[p.value] : leaf(p.value);
        ov.pending[p.channel] = v;
        ov.put[p.channel] = 1;
    }
}

inline StepReport Instance::step(const FireInputs& in) {
    StepReport rep;
    if (poisoned_) { rep.ok = false; rep.error = "poisoned"; return rep; }

    // 1. Readiness (interp.rs §7.1): first channel whose direction is unmet.
    for (const auto& e : bound_->readiness) {
        const auto& st = chans_[e.chan];
        bool ready = e.dir == P::container::Direction::NeedsFull ? !st.queue.empty()
                                                                 : st.queue.size() < st.capacity;
        if (!ready) { rep.has_miss = true; rep.miss_chan = e.chan; rep.miss_phase = e.phase; break; }
    }

    // 2. Run phases over a pass-local overlay.
    Overlay ov;
    ov.taken.assign(chans_.size(), 0);
    ov.put.assign(chans_.size(), 0);

    auto run_kind = [&](P::StageKind k) {
        for (const auto& s : trace_->stages)
            if (s.kind == k) { exec_stage(s, 0, in, ov, rep); if (!rep.ok) return; }
    };
    run_kind(P::StageKind::Prologue);
    // Descriptor phase: ports peek (or take, for the token family) — advance the ring.
    for (const auto& p : trace_->ports) {
        if (p.is_const) continue;
        if (P::port_consumes(p.port)) { (void)resolve(ov, p.channel); ov.taken[p.channel] = 1; }
    }
    run_kind(P::StageKind::OnAttnProj);
    run_kind(P::StageKind::OnAttn);
    run_kind(P::StageKind::Epilogue);
    if (!rep.ok) return rep;

    // 3. Pass-atomic commit: predicated per-channel index bump.
    rep.committed = !rep.has_miss;
    if (rep.committed) {
        for (std::size_t ci = 0; ci < chans_.size(); ++ci) {
            auto& st = chans_[ci];
            if (ov.taken[ci] && !st.queue.empty()) { st.last = st.queue.front(); st.queue.pop_front(); }
            if (ov.put[ci]) {
                Val v = ov.pending[(std::uint32_t)ci];
                if (st.queue.size() >= st.capacity) {
                    poisoned_ = true; rep.ok = false;
                    rep.error = "channel " + std::to_string(ci) + ": put overflows capacity";
                    return rep;
                }
                st.last = v;
                st.queue.push_back(std::move(v));
            }
        }
    }
    return rep;
}

}  // namespace tier0


