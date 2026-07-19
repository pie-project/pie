#pragma once

// PTIR tier-1 JIT fusion codegen (overview §7.3 "Tier 1 — fuse").
// Generalizes the shipped Sampling-IR NVRTC codegen from
// "the epilogue" to "a stage program": lowers a fusable row-parallel stage into
// ONE fused CUDA-C kernel — a lane per CTA (blockIdx.x = row), per-vocab ops held
// in registers across the row, reductions row-local via a block barrier. The
// readiness check fuses into the kernel prologue and the epoch-ring commit-bump
// into the epilogue — modeled here at the launch boundary for the tier-1
// first cut; the emitted math is the fused core.
//
// SCOPE (tier-1 cut #1): a stage whose value flow is a linear chain of per-vocab
// element-wise ops (map / select / compare / mask_apply / rng) over the intrinsic
// logits row, terminated by ONE row-local reduction (reduce_argmax / reduce_sum
// / reduce_max) whose scalar result is the stage's put/output. This is the
// greedy / temperature-sample epilogue — the Sampling-IR stress case — fused into
// a single kernel (§6.2's two-flat-scatter form is the next cut). Mid-chain
// reductions (softmax nucleus) lower to a ≤2-kernel split, deferred.
//
// The emitted source is self-contained NVRTC-compilable CUDA-C (embeds the RNG /
// bf16 prelude, bit-parity with tier-0's tier0_kernels.cuh + sample_temp.cu). The
// JIT layer (or the test harness) NVRTC-compiles + caches it by the
// container/program hash, exactly the Sampling-IR discipline.

#include <cstdint>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "pie_native/ptir/rng_contract.generated.h"
#include "pie_native/ptir/trace.hpp"

namespace pie_cuda_driver::pipeline {

// Shared pure-host PTIR decode model (trace/op-table/container/bound/
// fire-geometry) now lives in pie_native::ptir (driver/common); bring it into
// scope so the CUDA-side tier-0/1 code below can use it unqualified.
using namespace pie_native::ptir;

// One kernel argument the JIT/test binds at launch, in emission order.
enum class T1ArgKind : std::uint8_t {
    Logits,     // const float* [rows, V]  (intrinsic logits; f32 for tier-1 cut #1)
    RowSeed,    // const uint32_t* [rows]  (ambient RNG seed S per row)
    ChanVecF32, // const float* [V]  (a per-vocab channel-take value)
    ChanMaskU8, // const uint8_t* [V] (a bool channel-take mask, e.g. §3 grammar mask)
    RngState,   // const uint32_t* [2] (a channel-take rng state [key,ctr] for rng_keyed)
    OutToken,   // int* [rows]   (reduce_argmax result)
    OutScalar,  // float* [rows] (reduce_sum / reduce_max result)
    Rows,       // int
    Vocab,      // int
};

struct T1Arg {
    T1ArgKind kind;
    std::uint32_t key = 0;     // HostVec/HostMask: source host_key
    std::uint32_t channel = 0; // ChanVec/ChanMask/RngState: source channel id
};

struct T1Kernel {
    bool        ok = false;
    std::string error;
    std::string entry_name;   // extern "C" __global__ symbol
    std::string source;       // self-contained NVRTC CUDA-C
    std::vector<T1Arg> args;  // launch argument plan, positional
    std::uint32_t vocab = 0;  // per-row length (for the test; also a launch Param)
};

namespace detail_t1 {

// NVRTC-safe device prelude — RNG bit-parity with tier0_kernels.cuh.
inline std::string t1_prelude() {
    std::string source = R"PIECUDA(
#define T1_BLOCK 256
__device__ __forceinline__ float t1_neg_inf(){ return __int_as_float(0xff800000); }
)PIECUDA";
    source += PTIR_RNG_CUDA_PREAMBLE;
    source += R"PIECUDA(
__device__ __forceinline__ float t1_gumbel(unsigned long long se, int j){ float u=ptir_rng_hash_uniform(se,(unsigned int)j); return -logf(-logf(u)); }
__device__ __forceinline__ float t1_gumbel_keyed(unsigned int key, unsigned int ctr, int j){
  unsigned long long s64 = ptir_rng_keyed_seed(key, ctr);
  float u = ptir_rng_hash_uniform(s64, (unsigned int)j); return -logf(-logf(u)); }
)PIECUDA";
    return source;
}

}  // namespace detail_t1

// Lower a fusable epilogue stage to a single fused kernel. `trace` provides the
// value table (types + sources); `stage` is the epilogue. On success `ok` and a
// self-contained kernel; otherwise `error` explains why the stage is not a
// tier-1-cut-#1 fusable shape (the JIT then falls back to tier-0).
inline T1Kernel emit_fused_epilogue(const Trace& trace, const Stage& stage) {
    T1Kernel k;
    auto fail = [&](const std::string& m) { k.ok = false; k.error = m; return k; };

    // Identify the single terminal reduction op + the intrinsic logits root.
    const Op* term = nullptr;
    for (const Op& op : stage.ops) {
        bool is_red = op.code == OpCode::ReduceArgmax || op.code == OpCode::ReduceSum ||
                      op.code == OpCode::ReduceMax;
        if (is_red) { if (term) return fail("more than one terminal reduction (needs >1 kernel)"); term = &op; }
    }
    if (!term) return fail("no terminal reduction — not a tier-1 cut#1 epilogue");

    // Classify every value: per-vocab (depends on logits/rng/host-vec) vs scalar.
    std::unordered_map<ValueId, bool> per_vocab;   // true = per-vocab
    std::unordered_map<ValueId, std::string> expr; // per-vocab: expression in j; scalar: C literal/var
    std::uint32_t vocab = 0;

    // Resolve roots.
    for (const Value& v : trace.values) {
        if (v.source == ValueSource::Intrinsic &&
            (v.intrinsic == Intrinsic::Logits || v.intrinsic == Intrinsic::MtpLogits)) {
            per_vocab[v.id] = true;
            expr[v.id] = "logits[row*V + j]";
            if (!v.type.shape.dims.empty()) vocab = v.type.shape.dims.back();
            k.args.push_back({T1ArgKind::Logits});
        } else if (v.source == ValueSource::Const) {
            per_vocab[v.id] = false;
            std::ostringstream s;
            if (v.type.dtype == DType::F32) {
                s << "__int_as_float(0x" << std::hex << v.lit.as_u32() << std::dec << "u)";
            } else {
                s << v.lit.as_u32() << "u";
            }
            expr[v.id] = s.str();
        } else if (v.source == ValueSource::ChannelTake || v.source == ValueSource::ChannelRead) {
            // A per-vocab channel value (mask/bias [V]) → a per-vocab kernel arg;
            // a small state channel ([key,ctr]) → an rng-state arg.
            std::uint64_t n = v.type.shape.numel();
            if (n >= 2 && v.type.dtype == DType::U32 && n <= 4) {
                int ai = (int)k.args.size();
                k.args.push_back({T1ArgKind::RngState, 0, v.channel});
                per_vocab[v.id] = false;
                expr[v.id] = "rngstate" + std::to_string(ai);   // referenced by rng_keyed
            } else if (v.type.dtype == DType::Bool) {
                int ai = (int)k.args.size();
                k.args.push_back({T1ArgKind::ChanMaskU8, 0, v.channel});
                per_vocab[v.id] = true;
                expr[v.id] = "chanmask" + std::to_string(ai) + "[j]";
            } else if (v.type.dtype == DType::F32) {
                int ai = (int)k.args.size();
                k.args.push_back({T1ArgKind::ChanVecF32, 0, v.channel});
                per_vocab[v.id] = true;
                expr[v.id] = "chanvec" + std::to_string(ai) + "[j]";
            }
        }
    }
    if (vocab == 0) return fail("could not determine vocab length from logits intrinsic");
    k.vocab = vocab;
    bool needs_seed = false;

    // Walk ops in SSA order emitting per-vocab / scalar expressions.
    std::ostringstream body;   // per-vocab op locals, emitted inside the j-loop
    for (const Op& op : stage.ops) {
        if (&op == term) continue;   // terminal handled last
        const OpInfo info = op_info(op.code);
        // gather operand exprs + per-vocab flags
        std::vector<std::string> a;
        bool any_pv = false;
        for (ValueId id : op.args) {
            auto pe = expr.find(id);
            if (pe == expr.end()) return fail("unresolved operand in fused chain");
            a.push_back(pe->second);
            any_pv = any_pv || per_vocab[id];
        }
        std::string var = "v" + std::to_string(op.result_id);
        std::string e;
        switch (op.code) {
            case OpCode::Reshape:   // alias: same per-vocab value, no compute
                per_vocab[op.result_id] = a.empty() ? false : per_vocab[op.args[0]];
                expr[op.result_id] = a.empty() ? "0.0f" : a[0];
                continue;
            case OpCode::Add: e = "(" + a[0] + " + " + a[1] + ")"; break;
            case OpCode::Sub: e = "(" + a[0] + " - " + a[1] + ")"; break;
            case OpCode::Mul: e = "(" + a[0] + " * " + a[1] + ")"; break;
            case OpCode::Div: e = "(" + a[0] + " / " + a[1] + ")"; break;
            case OpCode::MaxElem: e = "fmaxf(" + a[0] + ", " + a[1] + ")"; break;
            case OpCode::MinElem: e = "fminf(" + a[0] + ", " + a[1] + ")"; break;
            case OpCode::Neg: e = "(-(" + a[0] + "))"; break;
            case OpCode::Exp: e = "expf(" + a[0] + ")"; break;
            case OpCode::Log: e = "logf(" + a[0] + ")"; break;
            case OpCode::Recip: e = "(1.0f/(" + a[0] + "))"; break;
            case OpCode::Select:   // select(cond, a, b): cond ? a : b
                e = "((" + a[0] + ") ? (" + a[1] + ") : (" + a[2] + "))";
                any_pv = true;
                break;
            case OpCode::RngKeyed: {   // gumbel/uniform over a channel state [key,ctr]
                std::string st = a.empty() ? "0" : a[0];
                if (op.rng_kind == RngKind::Gumbel)
                    e = "t1_gumbel_keyed(" + st + "[0], " + st + "[1], j)";
                else
                    e = "ptir_rng_hash_uniform(ptir_rng_keyed_seed(" + st +
                        "[0], " + st + "[1]), (unsigned int)j)";
                any_pv = true;
                break;
            }
            case OpCode::Rng: {
                needs_seed = true;
                e = "t1_gumbel(ptir_rng_seed_eff_stream(row_seed[row], " +
                    std::to_string(op.imm) + "u), j)";
                any_pv = true;
                break;
            }
            default:
                return fail("op '" + std::string(info.name) + "' not in tier-1 cut#1 fusable set");
        }
        per_vocab[op.result_id] = any_pv;
        if (any_pv) { body << "        float " << var << " = " << e << ";\n"; expr[op.result_id] = var; }
        else { expr[op.result_id] = e; }
    }

    // Terminal reduction over the (per-vocab) input expression.
    auto tin = expr.find(term->args[0]);
    if (tin == expr.end() || !per_vocab[term->args[0]]) return fail("terminal reduction input is not per-vocab");
    std::string tin_expr = tin->second;

    if (needs_seed) k.args.push_back({T1ArgKind::RowSeed});

    bool is_argmax = term->code == OpCode::ReduceArgmax;
    if (is_argmax) k.args.push_back({T1ArgKind::OutToken});
    else k.args.push_back({T1ArgKind::OutScalar});
    k.args.push_back({T1ArgKind::Rows});
    k.args.push_back({T1ArgKind::Vocab});

    // Assemble the kernel source.
    k.entry_name = "ptir_t1_epilogue";
    std::ostringstream src;
    src << detail_t1::t1_prelude() << "\nextern \"C\" __global__ void " << k.entry_name << "(\n";
    // parameter list in args order
    for (std::size_t i = 0; i < k.args.size(); ++i) {
        const T1Arg& ar = k.args[i];
        switch (ar.kind) {
            case T1ArgKind::Logits:    src << "    const float* __restrict__ logits"; break;
            case T1ArgKind::RowSeed:   src << "    const unsigned int* __restrict__ row_seed"; break;
            case T1ArgKind::ChanVecF32:src << "    const float* __restrict__ chanvec" << i; break;
            case T1ArgKind::ChanMaskU8:src << "    const unsigned char* __restrict__ chanmask" << i; break;
            case T1ArgKind::RngState:  src << "    const unsigned int* __restrict__ rngstate" << i; break;
            case T1ArgKind::OutToken:  src << "    int* __restrict__ out_token"; break;
            case T1ArgKind::OutScalar: src << "    float* __restrict__ out_scalar"; break;
            case T1ArgKind::Rows:      src << "    int rows"; break;
            case T1ArgKind::Vocab:     src << "    int V"; break;
        }
        src << (i + 1 < k.args.size() ? ",\n" : ")\n");
    }
    src << "{\n"
        << "  int row = blockIdx.x; if (row >= rows) return;\n"
        << "  int tid = threadIdx.x;\n";
    if (is_argmax) {
        src << "  __shared__ float sh_v[T1_BLOCK]; __shared__ int sh_i[T1_BLOCK];\n"
            << "  float best = t1_neg_inf(); int bi = 0;\n"
            << "  for (int j = tid; j < V; j += T1_BLOCK) {\n"
            << body.str()
            << "        float f = " << tin_expr << ";\n"
            << "        if (f > best) { best = f; bi = j; }\n"
            << "  }\n"
            << "  sh_v[tid] = best; sh_i[tid] = bi; __syncthreads();\n"
            << "  for (int s = T1_BLOCK/2; s > 0; s >>= 1) {\n"
            << "    if (tid < s) { float ov = sh_v[tid+s]; int oi = sh_i[tid+s];\n"
            << "      if (ov > sh_v[tid] || (ov == sh_v[tid] && oi < sh_i[tid])) { sh_v[tid]=ov; sh_i[tid]=oi; } }\n"
            << "    __syncthreads(); }\n"
            << "  if (tid == 0) out_token[row] = sh_i[0];\n";
    } else {
        bool is_sum = term->code == OpCode::ReduceSum;
        src << "  __shared__ float sh[T1_BLOCK];\n"
            << "  float acc = " << (is_sum ? "0.0f" : "t1_neg_inf()") << ";\n"
            << "  for (int j = tid; j < V; j += T1_BLOCK) {\n"
            << body.str()
            << "        float f = " << tin_expr << ";\n"
            << (is_sum ? "        acc += f;\n" : "        acc = fmaxf(acc, f);\n")
            << "  }\n"
            << "  sh[tid] = acc; __syncthreads();\n"
            << "  for (int s = T1_BLOCK/2; s > 0; s >>= 1) {\n"
            << "    if (tid < s) sh[tid] = " << (is_sum ? "sh[tid] + sh[tid+s]" : "fmaxf(sh[tid], sh[tid+s])") << ";\n"
            << "    __syncthreads(); }\n"
            << "  if (tid == 0) out_scalar[row] = sh[0];\n";
    }
    src << "}\n";
    k.source = src.str();
    k.ok = true;
    return k;
}

}  // namespace pie_cuda_driver::pipeline
