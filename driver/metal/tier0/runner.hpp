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
#include <algorithm>
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
    std::vector<float> logits;     // [rows*vocab] row-major (Intrinsic::Logits)
    bool has_mtp_logits = false;
    std::vector<float> mtp_logits; // [rows*vocab] row-major (Intrinsic::MtpLogits)
    std::vector<float> value_head;              // Intrinsic::ValueHead
    std::vector<std::vector<float>> query;      // Intrinsic::Query, one per layer
    int vocab = 0;
    std::map<std::uint32_t, Val> host_inputs;  // host_key -> value
};

// A second-party kernel host (kernel_call, e.g. Quest's envelope_dot). The
// executor supplies it; a null host means the program uses no kernels.
struct KernelHost {
    virtual ~KernelHost() = default;
    // name = trace name-table entry; result_type gives the expected shape/dtype.
    virtual Val call(const std::string& name, const std::vector<Val>& args,
                     const P::TensorType& result_type) = 0;
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
    // Sink effects fired this pass (sink_call), in execution order.
    struct Sink {
        std::uint16_t name_idx = 0;
        P::StageKind stage = P::StageKind::Epilogue;
        std::uint32_t layer = 0;
        std::vector<Val> args;
    };
    std::vector<Sink> sinks;
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

    StepReport step(const FireInputs& in, KernelHost* host = nullptr);

    // Number of decoder layers the per-layer stages (OnAttnProj/OnAttn) iterate
    // over (model profile property; 1 for the single-layer channel goldens).
    void set_num_layers(std::uint32_t n) { num_layers_ = n; }

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
    // pivot_threshold(RankLe): per-row top-k bool mask. `kv` is the resolved k
    // operand (int lanes; [1] or [rows]). Mirrors eval.rs sort_desc + take(k).
    Val pivot_threshold(const P::Op& op, const Val& input, const Val& kv) const {
        const auto& d = op.result_type.shape.dims;
        int len = d.empty() ? 1 : (int)d.back();
        int rows = 1;
        for (std::size_t k = 0; k + 1 < d.size(); ++k) rows *= (int)d[k];
        if (rows == 0) rows = 1;
        auto x = input.to_f32();
        auto kf = kv.to_f32();
        std::vector<std::int64_t> keep((std::size_t)rows * len, 0);
        for (int r = 0; r < rows; ++r) {
            std::int64_t k = (std::int64_t)(kf.size() == (std::size_t)rows ? kf[r]
                                            : (kf.empty() ? 0.0f : kf[0]));
            k = std::max<std::int64_t>(0, std::min<std::int64_t>(k, len));
            std::vector<int> order(len);
            for (int j = 0; j < len; ++j) order[j] = j;
            std::stable_sort(order.begin(), order.end(),
                             [&](int a, int b) { return x[(std::size_t)r * len + a]
                                                        > x[(std::size_t)r * len + b]; });
            for (int t = 0; t < (int)k; ++t) keep[(std::size_t)r * len + order[t]] = 1;
        }
        return Val::make_int(DType::Bool, std::move(keep));
    }
    void exec_stage(const P::Stage& s, std::uint32_t layer, const FireInputs& in, Overlay& ov,
                    StepReport& rep);
    std::vector<Val> eval_op(const P::Op& op, const std::vector<Val>& args, StepReport& rep);
    const P::Trace* trace_;
    const P::bound::Bound* bound_;
    MetalOps* ops_;
    std::vector<Chan> chans_;
    bool poisoned_ = false;
    std::uint32_t num_layers_ = 1;
    KernelHost* host_ = nullptr;
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

inline std::vector<Val> Instance::eval_op(const P::Op& op, const std::vector<Val>& args, StepReport& rep) {
    using OC = P::OpCode;
    auto one = [](Val v) { return std::vector<Val>{std::move(v)}; };
    auto argtype = [&](std::size_t k) { return trace_->value(op.args[k])->type; };
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
    auto cmp = [&](auto f) {
        auto a = args[0].to_f32();
        auto b = args[1].to_f32();
        std::size_t n = a.size() > b.size() ? a.size() : b.size();
        std::vector<std::int64_t> o(n);
        for (std::size_t k = 0; k < n; ++k)
            o[k] = f(a[a.size() == 1 ? 0 : k], b[b.size() == 1 ? 0 : k]) ? 1 : 0;
        return Val::make_int(DType::Bool, std::move(o));
    };
    auto unary_f = [&](auto f) {
        auto a = args[0].to_f32();
        std::vector<float> o(a.size());
        for (std::size_t k = 0; k < a.size(); ++k) o[k] = f(a[k]);
        return Val::make_f32(std::move(o));
    };
    switch (op.code) {
        case OC::Add:
            return one(args[0].dt == DType::F32 ? binop_f([](float x, float y) { return x + y; })
                                                : binop_int([](std::int64_t x, std::int64_t y) { return x + y; }));
        case OC::Sub:
            return one(args[0].dt == DType::F32 ? binop_f([](float x, float y) { return x - y; })
                                                : binop_int([](std::int64_t x, std::int64_t y) { return x - y; }));
        case OC::Mul:
            return one(args[0].dt == DType::F32 ? binop_f([](float x, float y) { return x * y; })
                                                : binop_int([](std::int64_t x, std::int64_t y) { return x * y; }));
        case OC::Div:
            return one(args[0].dt == DType::F32 ? binop_f([](float x, float y) { return x / y; })
                                                : binop_int([](std::int64_t x, std::int64_t y) { return y ? x / y : 0; }));
        case OC::Rem:
            return one(binop_int([](std::int64_t x, std::int64_t y) { return y ? x % y : 0; }));
        case OC::MaxElem:
            return one(binop_f([](float x, float y) { return x > y ? x : y; }));
        case OC::MinElem:
            return one(binop_f([](float x, float y) { return x < y ? x : y; }));
        case OC::Gt: return one(cmp([](float x, float y) { return x > y; }));
        case OC::Ge: return one(cmp([](float x, float y) { return x >= y; }));
        case OC::Eq: return one(cmp([](float x, float y) { return x == y; }));
        case OC::Ne: return one(cmp([](float x, float y) { return x != y; }));
        case OC::Lt: return one(cmp([](float x, float y) { return x < y; }));
        case OC::Le: return one(cmp([](float x, float y) { return x <= y; }));
        case OC::And: {
            std::size_t n = args[0].i.size() > args[1].i.size() ? args[0].i.size() : args[1].i.size();
            std::vector<std::int64_t> o(n);
            for (std::size_t k = 0; k < n; ++k)
                o[k] = (args[0].i[args[0].i.size() == 1 ? 0 : k] && args[1].i[args[1].i.size() == 1 ? 0 : k]) ? 1 : 0;
            return one(Val::make_int(DType::Bool, std::move(o)));
        }
        case OC::Or: {
            std::size_t n = args[0].i.size() > args[1].i.size() ? args[0].i.size() : args[1].i.size();
            std::vector<std::int64_t> o(n);
            for (std::size_t k = 0; k < n; ++k)
                o[k] = (args[0].i[args[0].i.size() == 1 ? 0 : k] || args[1].i[args[1].i.size() == 1 ? 0 : k]) ? 1 : 0;
            return one(Val::make_int(DType::Bool, std::move(o)));
        }
        case OC::Not: {
            std::vector<std::int64_t> o(args[0].i.size());
            for (std::size_t k = 0; k < o.size(); ++k) o[k] = args[0].i[k] ? 0 : 1;
            return one(Val::make_int(DType::Bool, std::move(o)));
        }
        case OC::Exp: return one(unary_f([](float x) { return std::exp(x); }));
        case OC::Log: return one(unary_f([](float x) { return std::log(x); }));
        case OC::Neg: return one(unary_f([](float x) { return -x; }));
        case OC::Recip: return one(unary_f([](float x) { return 1.0f / x; }));
        case OC::Abs: return one(unary_f([](float x) { return std::fabs(x); }));
        case OC::Reshape:
        case OC::Transpose:
            return one(args[0]);
        case OC::Broadcast: {
            // left-aligned row-major broadcast (eval.rs broadcast_to).
            const auto& td = op.result_type.shape.dims;
            const auto& sd = argtype(0).shape.dims;
            int r = (int)td.size();
            auto sdim = [&](int idx) { return idx < (int)sd.size() ? (std::int64_t)sd[idx] : 1; };
            std::size_t n = 1;
            for (auto d : td) n *= d;
            if (n == 0) n = 1;
            std::vector<std::uint64_t> sstride(r, 1);
            for (int idx = r - 2; idx >= 0; --idx) sstride[idx] = sstride[idx + 1] * (std::uint64_t)sdim(idx + 1);
            auto src_idx = [&](std::uint64_t lin) {
                std::uint64_t rem = lin, si = 0;
                for (int idx = 0; idx < r; ++idx) {
                    std::uint64_t stride = 1;
                    for (int q = idx + 1; q < r; ++q) stride *= (std::uint64_t)td[q];
                    std::uint64_t coord = stride ? rem / stride : 0;
                    if (stride) rem %= stride;
                    if (sdim(idx) != 1) si += coord * sstride[idx];
                }
                return si;
            };
            if (args[0].dt == DType::F32) {
                auto s = args[0].to_f32();
                std::vector<float> o(n);
                for (std::uint64_t k = 0; k < n; ++k) o[k] = s[src_idx(k)];
                return one(Val::make_f32(std::move(o)));
            }
            std::vector<std::int64_t> o(n);
            for (std::uint64_t k = 0; k < n; ++k) o[k] = args[0].i[src_idx(k)];
            return one(Val::make_int(args[0].dt, std::move(o)));
        }
        case OC::Select: {
            const Val& c = args[0]; const Val& a = args[1]; const Val& b = args[2];
            std::size_t n = c.numel();
            n = a.numel() > n ? a.numel() : n;
            n = b.numel() > n ? b.numel() : n;
            auto pick = [](std::size_t l, std::size_t k) { return l == 1 ? std::size_t(0) : k; };
            const auto& cond = c.i;
            if (a.dt == DType::F32) {
                auto av = a.to_f32(), bv = b.to_f32();
                std::vector<float> o(n);
                for (std::size_t k = 0; k < n; ++k)
                    o[k] = cond[pick(cond.size(), k)] ? av[pick(av.size(), k)] : bv[pick(bv.size(), k)];
                return one(Val::make_f32(std::move(o)));
            }
            std::vector<std::int64_t> o(n);
            for (std::size_t k = 0; k < n; ++k)
                o[k] = cond[pick(cond.size(), k)] ? a.i[pick(a.i.size(), k)] : b.i[pick(b.i.size(), k)];
            return one(Val::make_int(a.dt, std::move(o)));
        }
        case OC::Iota: {
            std::vector<std::int64_t> o(op.imm);
            for (std::uint32_t k = 0; k < op.imm; ++k) o[k] = k;
            return one(Val::make_int(DType::U32, std::move(o)));
        }
        case OC::Cast: {
            DType to = op.result_type.dtype;
            if (to == DType::F32) return one(Val::make_f32(args[0].to_f32()));
            std::vector<std::int64_t> o(args[0].numel());
            if (args[0].dt == DType::F32) { auto f = args[0].to_f32(); for (std::size_t k = 0; k < o.size(); ++k) o[k] = (std::int64_t)f[k]; }
            else o = args[0].i;
            return one(Val::make_int(to, std::move(o)));
        }
        case OC::RngKeyed: {
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
            return one(Val::make_f32(std::move(o)));
        }
        case OC::ReduceArgmax: {
            int rows, len; rows_len(argtype(0), rows, len);
            auto tokens = ops_->reduce_argmax(args[0].to_f32(), rows, len);
            return one(Val::make_int(DType::I32, std::vector<std::int64_t>(tokens.begin(), tokens.end())));
        }
        case OC::ReduceSum:
        case OC::ReduceMax:
        case OC::ReduceMin: {
            // Compute on Metal (reduce_*_rows kernels) for the sampling compute.
            int rows, len; rows_len(argtype(0), rows, len);
            const char* fn = op.code == OC::ReduceSum ? "reduce_sum_rows"
                           : op.code == OC::ReduceMax ? "reduce_max_rows" : "reduce_min_rows";
            return one(Val::make_f32(ops_->reduce_rows(fn, args[0].to_f32(), rows, len)));
        }
        case OC::CumProd:
        case OC::CumSum: {
            // Row-local inclusive prefix scan (interp.rs Reduce/RowLocal). Used
            // by spec_verify_greedy: cumprod over {0,1} = prefix-AND accepting
            // the longest matching draft prefix. Host control-flow (like the
            // eq/gt lanes); the heavy per-row work stays on Metal (argmax).
            int rows, len; rows_len(argtype(0), rows, len);
            auto in = args[0].to_f32();
            const bool prod = op.code == OC::CumProd;
            std::vector<float> o(in.size());
            for (int r = 0; r < rows; ++r) {
                float acc = prod ? 1.0f : 0.0f;
                for (int j = 0; j < len; ++j) {
                    float x = in[(std::size_t)r * len + j];
                    acc = prod ? acc * x : acc + x;
                    o[(std::size_t)r * len + j] = acc;
                }
            }
            return one(Val::make_f32(std::move(o)));
        }
        case OC::TopK: {
            int rows, len; rows_len(argtype(0), rows, len);
            int k = (int)op.imm;
            auto x = args[0].to_f32();
            std::vector<float> vs; std::vector<std::int64_t> is;
            for (int r = 0; r < rows; ++r) {
                std::vector<int> ord(len);
                for (int j = 0; j < len; ++j) ord[j] = j;
                std::stable_sort(ord.begin(), ord.end(), [&](int a, int b) { return x[r * len + a] > x[r * len + b]; });
                for (int t = 0; t < k; ++t) { vs.push_back(x[r * len + ord[t]]); is.push_back(ord[t]); }
            }
            return {Val::make_f32(std::move(vs)), Val::make_int(DType::U32, std::move(is))};
        }
        case OC::Gather: {
            const auto& ts = argtype(0).shape.dims;
            std::size_t rest = 1;
            for (std::size_t d = 1; d < ts.size(); ++d) rest *= ts[d];
            if (rest == 0) rest = 1;
            std::size_t n0 = ts.empty() ? args[0].numel() : ts[0];
            const auto& ix = args[1].i;
            if (args[0].dt == DType::F32) {
                auto s = args[0].to_f32();
                std::vector<float> o(ix.size() * rest, 0.0f);
                for (std::size_t j = 0; j < ix.size(); ++j)
                    if (ix[j] >= 0 && (std::size_t)ix[j] < n0)
                        for (std::size_t r = 0; r < rest; ++r) o[j * rest + r] = s[ix[j] * rest + r];
                return one(Val::make_f32(std::move(o)));
            }
            std::vector<std::int64_t> o(ix.size() * rest, 0);
            for (std::size_t j = 0; j < ix.size(); ++j)
                if (ix[j] >= 0 && (std::size_t)ix[j] < n0)
                    for (std::size_t r = 0; r < rest; ++r) o[j * rest + r] = args[0].i[ix[j] * rest + r];
            return one(Val::make_int(args[0].dt, std::move(o)));
        }
        case OC::ScatterSet:
        case OC::ScatterAdd: {
            const auto& tb = argtype(0).shape.dims;
            std::size_t rest = 1;
            for (std::size_t d = 1; d < tb.size(); ++d) rest *= tb[d];
            if (rest == 0) rest = 1;
            std::size_t n0 = tb.empty() ? args[0].numel() : tb[0];
            const auto& ix = args[1].i;
            bool is_add = op.code == OC::ScatterAdd;
            bool scalar_val = args[2].numel() == 1 && ix.size() * rest != 1;
            if (args[0].dt == DType::F32) {
                auto out = args[0].to_f32();
                auto vv = args[2].to_f32();
                for (std::size_t kk = 0; kk < ix.size(); ++kk)
                    if (ix[kk] >= 0 && (std::size_t)ix[kk] < n0)
                        for (std::size_t r = 0; r < rest; ++r) {
                            float src = scalar_val ? vv[0] : vv[kk * rest + r];
                            float& dst = out[ix[kk] * rest + r];
                            dst = is_add ? dst + src : src;
                        }
                return one(Val::make_f32(std::move(out)));
            }
            auto out = args[0].i;
            const auto& vv = args[2].i;
            for (std::size_t kk = 0; kk < ix.size(); ++kk)
                if (ix[kk] >= 0 && (std::size_t)ix[kk] < n0)
                    for (std::size_t r = 0; r < rest; ++r) {
                        std::int64_t src = scalar_val ? vv[0] : vv[kk * rest + r];
                        std::int64_t& dst = out[ix[kk] * rest + r];
                        dst = is_add ? dst + src : src;
                    }
            return one(Val::make_int(args[0].dt, std::move(out)));
        }
        case OC::GatherRow: {
            int m = 1, n = 1;
            const auto& ts = argtype(0).shape.dims;
            if (ts.size() >= 2) { m = (int)ts[0]; n = (int)ts[1]; }
            const auto& ix = args[1].i;
            if (args[0].dt == DType::F32) {
                auto s = args[0].to_f32();
                std::vector<float> o(m, 0.0f);
                for (int r = 0; r < m; ++r) if (ix[r] >= 0 && ix[r] < n) o[r] = s[r * n + ix[r]];
                return one(Val::make_f32(std::move(o)));
            }
            std::vector<std::int64_t> o(m, 0);
            for (int r = 0; r < m; ++r) if (ix[r] >= 0 && ix[r] < n) o[r] = args[0].i[r * n + ix[r]];
            return one(Val::make_int(args[0].dt, std::move(o)));
        }
        default:
            rep.ok = false;
            rep.error = "tier0: unhandled op 0x" + std::to_string((int)op.code) + " (" +
                        std::string(P::op_info(op.code).name) + ")";
            return one(Val::zeros(DType::F32, 1));
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
                // De-hardwiring: the draft speculator binds Intrinsic::MtpLogits
                // (the on-device MTP draft logits); a plain sampler binds
                // Intrinsic::Logits. Same argmax bytecode — only the bound
                // source differs (mtp_argmax vs argmax). Mirrors CUDA Stage 2.
                switch (v->intrinsic) {
                    case P::Intrinsic::MtpLogits:
                        out = Val::make_f32(in.has_mtp_logits ? in.mtp_logits : in.logits);
                        break;
                    case P::Intrinsic::ValueHead:
                        out = Val::make_f32(in.value_head);
                        break;
                    case P::Intrinsic::Query:
                        out = Val::make_f32(layer < in.query.size() ? in.query[layer]
                                                                    : std::vector<float>{});
                        break;
                    default:  // Logits / Hidden
                        out = Val::make_f32(in.logits);
                        break;
                }
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
    for (const P::Op& op : s.ops) {
        // ── structural / host ops that need stage-context (host, layer, the
        //    leaf resolver for the predicate operand) ──
        if (op.code == P::OpCode::KernelCall) {
            std::vector<Val> args;
            for (auto aid : op.args) args.push_back(leaf(aid));
            if (!host_) { rep.ok = false; rep.error = "kernel_call with no kernel host"; return; }
            const std::string& nm = op.name_idx < trace_->names.size()
                                        ? trace_->names[op.name_idx] : std::string();
            cache[op.result_id] = host_->call(nm, args, op.result_type);
            continue;
        }
        if (op.code == P::OpCode::SinkCall) {
            StepReport::Sink sk;
            sk.name_idx = op.name_idx;
            sk.stage = s.kind;
            sk.layer = layer;
            for (auto aid : op.args) sk.args.push_back(leaf(aid));
            rep.sinks.push_back(std::move(sk));
            continue;
        }
        if (op.code == P::OpCode::PivotThreshold) {
            // Per-row top-k bool mask (interp.rs pivot_threshold). The k/thr is a
            // VALUE id in predicate.payload (resolved from the value table).
            Val input = leaf(op.args[0]);
            Val kv = leaf(op.predicate.payload);
            cache[op.result_id] = pivot_threshold(op, input, kv);
            continue;
        }
        std::vector<Val> args;
        args.reserve(op.args.size());
        for (auto aid : op.args) args.push_back(leaf(aid));
        std::vector<Val> results = eval_op(op, args, rep);
        if (!rep.ok) return;
        for (std::uint32_t rr = 0; rr < results.size(); ++rr)
            cache[op.result_id + rr] = std::move(results[rr]);
    }
    // Channel puts (pass-local overlay; applied at commit).
    for (const P::ChannelPut& p : s.puts) {
        Val v = cache.count(p.value) ? cache[p.value] : leaf(p.value);
        ov.pending[p.channel] = v;
        ov.put[p.channel] = 1;
    }
    // Takes/reads whose result is unused (e.g. §6.2 klen/kvm drain) aren't in any
    // op's args — the translator records them explicitly; a take still advances
    // the ring at commit (interp.rs / charlie's collect_stage_channels).
    for (P::ChannelId c : s.takes) ov.taken[c] = 1;
}

inline StepReport Instance::step(const FireInputs& in, KernelHost* host) {
    StepReport rep;
    host_ = host;
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

    auto run_kind = [&](P::StageKind k, std::uint32_t layer) {
        for (const auto& s : trace_->stages)
            if (s.kind == k) { exec_stage(s, layer, in, ov, rep); if (!rep.ok) return; }
    };
    run_kind(P::StageKind::Prologue, 0);
    // Descriptor phase: ports peek (or take, for the token family) — advance the ring.
    for (const auto& p : trace_->ports) {
        if (p.is_const) continue;
        if (P::port_consumes(p.port)) { (void)resolve(ov, p.channel); ov.taken[p.channel] = 1; }
    }
    // Per-layer attention phases (OnAttnProj/OnAttn run once per decoder layer —
    // the quest tap fires envelope_dot + attn_page_mask at layer 0 then 1).
    for (std::uint32_t L = 0; L < num_layers_ && rep.ok; ++L) {
        run_kind(P::StageKind::OnAttnProj, L);
        run_kind(P::StageKind::OnAttn, L);
    }
    run_kind(P::StageKind::Epilogue, 0);
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


