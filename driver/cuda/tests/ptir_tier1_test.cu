// PTIR tier-1 fusion gate test. Emits a fused epilogue
// kernel (tier1_codegen.hpp), NVRTC-compiles + launches it, and asserts its
// output is IDENTICAL to the tier-0 stage-runner on the same trace + inputs —
// the tier-1 == tier-0 exit criterion. Covers greedy argmax and a
// temperature+Gumbel-max sample epilogue (the Sampling-IR stress case), fused
// into ONE kernel.
//
//   nvcc -std=c++17 -arch=sm_89 --extended-lambda --expt-relaxed-constexpr \
//        -Isrc tests/ptir_tier1_test.cu -lnvrtc -lcuda -o ptir_tier1_test

#include <cstdint>
#include <cstdio>
#include <limits>
#include <string>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>
#include <nvrtc.h>

#include "pipeline/tier0/tier0_runner.hpp"
#include "support/tier1_codegen.hpp"

using namespace pie_cuda_driver::pipeline;

namespace {
int g_pass = 0, g_fail = 0;
void expect(bool ok, const std::string& what) {
    if (ok) { ++g_pass; std::printf("  PASS  %s\n", what.c_str()); }
    else    { ++g_fail; std::printf("  FAIL  %s\n", what.c_str()); }
}
#define CU_OK(x) do { CUresult r=(x); if(r!=CUDA_SUCCESS){ const char* s; cuGetErrorString(r,&s); std::printf("CU err %s @ %d: %s\n", #x, __LINE__, s); std::exit(2);} } while(0)
#define RT_OK(x) do { cudaError_t e=(x); if(e!=cudaSuccess){ std::printf("RT err %s @ %d: %s\n", #x, __LINE__, cudaGetErrorString(e)); std::exit(2);} } while(0)

std::string arch_flag() {
    int dev = 0; cudaDeviceProp p{}; cudaGetDevice(&dev); cudaGetDeviceProperties(&p, dev);
    return "--gpu-architecture=compute_" + std::to_string(p.major) + std::to_string(p.minor);
}

CUfunction nvrtc_build(const std::string& src, const std::string& entry, CUmodule* mod_out) {
    nvrtcProgram prog;
    if (nvrtcCreateProgram(&prog, src.c_str(), "ptir_t1.cu", 0, nullptr, nullptr) != NVRTC_SUCCESS) { std::printf("nvrtcCreate fail\n"); std::exit(2); }
    std::string arch = arch_flag();
    const char* opts[] = {arch.c_str()};
    nvrtcResult rc = nvrtcCompileProgram(prog, 1, opts);
    if (rc != NVRTC_SUCCESS) {
        std::size_t ls; nvrtcGetProgramLogSize(prog, &ls);
        std::string log(ls, '\0'); nvrtcGetProgramLog(prog, log.data());
        std::printf("NVRTC compile failed:\n%s\n---SOURCE---\n%s\n", log.c_str(), src.c_str());
        std::exit(2);
    }
    std::size_t ptxs; nvrtcGetPTXSize(prog, &ptxs);
    std::string ptx(ptxs, '\0'); nvrtcGetPTX(prog, ptx.data());
    nvrtcDestroyProgram(&prog);
    CU_OK(cuModuleLoadData(mod_out, ptx.c_str()));
    CUfunction fn; CU_OK(cuModuleGetFunction(&fn, *mod_out, entry.c_str()));
    return fn;
}

Value mk(ValueId id, TensorType ty, ValueSource src) { Value v; v.id = id; v.type = ty; v.source = src; return v; }

// Build a temperature + Gumbel-max epilogue: argmax( logits*(1/T) + gumbel ).
// gumbel=false ⇒ plain greedy argmax(logits).
Trace build_epilogue(std::uint32_t V, bool gumbel, float inv_t, std::uint32_t stream) {
    Trace t;
    Channel tok; tok.id = 0; tok.type = {Shape::vec(1), DType::U32}; tok.capacity = 1; tok.host_visible = true;
    t.channels = {tok};
    TensorType logv{Shape::mat(1, V), DType::F32};
    TensorType tokt{Shape::vec(1), DType::U32};
    Value lg = mk(0, logv, ValueSource::Intrinsic); lg.intrinsic = Intrinsic::Logits; t.values.push_back(lg);
    Stage ep; ep.kind = StageKind::Epilogue;
    ValueId next = 1;
    ValueId reduce_in;
    if (!gumbel) {
        reduce_in = 0;   // argmax(logits) directly
    } else {
        Value c = mk(next, {Shape::scalar(), DType::F32}, ValueSource::Const); c.lit = Literal::f32(inv_t);
        t.values.push_back(c); ValueId cid = next++;
        Value mv = mk(next, logv, ValueSource::OpResult); t.values.push_back(mv); ValueId mulid = next++;
        Op mul; mul.code = OpCode::Mul; mul.args = {0, cid}; mul.result_type = logv; mul.result_id = mulid; ep.ops.push_back(mul);
        Value gv = mk(next, logv, ValueSource::OpResult); t.values.push_back(gv); ValueId gid = next++;
        Op gm; gm.code = OpCode::Rng; gm.rng_kind = RngKind::Gumbel;
        gm.result_type = logv; gm.result_id = gid; gm.imm = stream;
        ep.ops.push_back(gm);
        Value av = mk(next, logv, ValueSource::OpResult); t.values.push_back(av); ValueId aid = next++;
        Op add; add.code = OpCode::Add; add.args = {mulid, gid}; add.result_type = logv; add.result_id = aid; ep.ops.push_back(add);
        reduce_in = aid;
    }
    Value rv = mk(next, tokt, ValueSource::OpResult); t.values.push_back(rv); ValueId rid = next++;
    Op am; am.code = OpCode::ReduceArgmax; am.args = {reduce_in}; am.result_type = tokt; am.result_id = rid; ep.ops.push_back(am);
    ep.puts = {{0, rid}};
    t.stages = {ep};
    return t;
}

void run_case(const std::string& name, std::uint32_t V, bool gumbel, float inv_t, std::uint32_t stream, std::uint32_t seed) {
    Trace t = build_epilogue(V, gumbel, inv_t, stream);

    // random-ish logits
    std::vector<float> logits(V);
    for (std::uint32_t j = 0; j < V; ++j) logits[j] = 2.0f * sinf(0.017f * (j + 3) * (stream + 1)) + 0.001f * j;
    float* d_logits = nullptr; RT_OK(cudaMalloc(&d_logits, V * sizeof(float)));
    RT_OK(cudaMemcpy(d_logits, logits.data(), V * sizeof(float), cudaMemcpyHostToDevice));
    std::uint32_t* d_seed = nullptr; RT_OK(cudaMalloc(&d_seed, sizeof(std::uint32_t)));
    RT_OK(cudaMemcpy(d_seed, &seed, sizeof(seed), cudaMemcpyHostToDevice));

    // ── tier-0 (stage-runner) ──
    Tier0Runner runner(t);
    FireInputs in; in.logits = d_logits; in.vocab = V; in.row_seeds = d_seed;
    PassResult r = runner.run_pass(in);
    std::uint32_t t0_tok = 0; runner.arena().host_take(0, &t0_tok, sizeof(t0_tok));
    expect(r.ok && r.committed, name + ": tier-0 runs");

    // ── tier-1 (fused NVRTC kernel) ──
    T1Kernel k = emit_fused_epilogue(t, t.stages[0]);
    expect(k.ok, name + ": tier-1 emit (" + (k.ok ? "ok" : k.error) + ")");
    if (!k.ok) { cudaFree(d_logits); cudaFree(d_seed); return; }

    CUmodule mod; CUfunction fn = nvrtc_build(k.source, k.entry_name, &mod);
    int* d_out = nullptr; RT_OK(cudaMalloc(&d_out, sizeof(int)));
    int rows = 1, vocab = (int)V;
    // bind args in emitted order
    std::vector<void*> params;
    CUdeviceptr p_logits = (CUdeviceptr)d_logits, p_seed = (CUdeviceptr)d_seed, p_out = (CUdeviceptr)d_out;
    for (const T1Arg& a : k.args) {
        switch (a.kind) {
            case T1ArgKind::Logits:  params.push_back(&p_logits); break;
            case T1ArgKind::RowSeed: params.push_back(&p_seed); break;
            case T1ArgKind::OutToken:params.push_back(&p_out); break;
            case T1ArgKind::Rows:    params.push_back(&rows); break;
            case T1ArgKind::Vocab:   params.push_back(&vocab); break;
            default: std::printf("unexpected arg kind\n"); std::exit(2);
        }
    }
    CU_OK(cuLaunchKernel(fn, 1, 1, 1, 256, 1, 1, 0, nullptr, params.data(), nullptr));
    CU_OK(cuCtxSynchronize());
    int t1_tok = -1; RT_OK(cudaMemcpy(&t1_tok, d_out, sizeof(int), cudaMemcpyDeviceToHost));

    expect((std::uint32_t)t1_tok == t0_tok,
           name + ": tier-1 == tier-0 token (t0=" + std::to_string(t0_tok) + " t1=" + std::to_string(t1_tok) + ")");

    cuModuleUnload(mod);
    cudaFree(d_logits); cudaFree(d_seed); cudaFree(d_out);
}

// §3 epilogue sampling core: argmax( select(mask, logits, -inf) + gumbel_keyed(rng) ).
// Fuses the vocab-sized masked-Gumbel-max into ONE kernel; mask + rng arrive via
// channels (device buffers). Gate: tier-1 == tier-0 on the same trace + channels.
void run_masked_gumbel(const std::string& name, std::uint32_t V, std::uint32_t key, std::uint32_t ctr) {
    Trace t;
    Channel mask; mask.id = 0; mask.type = {Shape::vec(V), DType::Bool}; mask.capacity = 1; mask.has_seed = true;
    Channel rng; rng.id = 1; rng.type = {Shape::vec(2), DType::U32}; rng.capacity = 1; rng.has_seed = true;
    Channel tok; tok.id = 2; tok.type = {Shape::vec(1), DType::U32}; tok.capacity = 1; tok.host_visible = true;
    t.channels = {mask, rng, tok};

    TensorType logv{Shape::mat(1, V), DType::F32}, boolv{Shape::vec(V), DType::Bool};
    TensorType rngt{Shape::vec(2), DType::U32}, tokt{Shape::vec(1), DType::U32}, scal{Shape::scalar(), DType::F32};
    Value v0 = mk(0, logv, ValueSource::Intrinsic); v0.intrinsic = Intrinsic::Logits; t.values.push_back(v0);
    Value v1 = mk(1, boolv, ValueSource::ChannelTake); v1.channel = 0; t.values.push_back(v1);
    Value v2 = mk(2, scal, ValueSource::Const); v2.lit = Literal::f32(-std::numeric_limits<float>::infinity()); t.values.push_back(v2);
    t.values.push_back(mk(3, logv, ValueSource::OpResult));   // masked = select(mask, logits, -inf)
    Value v4 = mk(4, rngt, ValueSource::ChannelTake); v4.channel = 1; t.values.push_back(v4);
    t.values.push_back(mk(5, logv, ValueSource::OpResult));   // g = rng_keyed(rng)
    t.values.push_back(mk(6, logv, ValueSource::OpResult));   // sum
    t.values.push_back(mk(7, tokt, ValueSource::OpResult));   // token = argmax

    Stage ep; ep.kind = StageKind::Epilogue;
    Op sel; sel.code = OpCode::Select; sel.args = {1, 0, 2}; sel.result_type = logv; sel.result_id = 3; ep.ops.push_back(sel);
    Op rk; rk.code = OpCode::RngKeyed; rk.args = {4}; rk.result_type = logv; rk.result_id = 5; rk.rng_kind = RngKind::Gumbel; ep.ops.push_back(rk);
    Op add; add.code = OpCode::Add; add.args = {3, 5}; add.result_type = logv; add.result_id = 6; ep.ops.push_back(add);
    Op am; am.code = OpCode::ReduceArgmax; am.args = {6}; am.result_type = tokt; am.result_id = 7; ep.ops.push_back(am);
    ep.puts = {{2, 7}};
    t.stages = {ep};

    // mask: allow ~2/3 of the vocab (pseudo-random) so it actually constrains.
    std::vector<std::uint8_t> maskv(V);
    for (std::uint32_t j = 0; j < V; ++j) maskv[j] = ((j * 2654435761u) % 3u != 0) ? 1u : 0u;
    std::vector<float> logits(V);
    for (std::uint32_t j = 0; j < V; ++j) logits[j] = 1.5f * cosf(0.021f * (j + 1) * (key % 7 + 1));
    std::vector<std::uint32_t> state{key, ctr};

    // ── tier-0 ──
    Tier0Runner runner(t);
    runner.arena().seed_cell(0, maskv.data(), maskv.size());
    runner.arena().seed_cell(1, state.data(), state.size() * 4);
    float* d_logits = nullptr; RT_OK(cudaMalloc(&d_logits, V * sizeof(float)));
    RT_OK(cudaMemcpy(d_logits, logits.data(), V * sizeof(float), cudaMemcpyHostToDevice));
    FireInputs in; in.logits = d_logits; in.vocab = V;
    PassResult r = runner.run_pass(in);
    std::uint32_t t0 = 0; runner.arena().host_take(2, &t0, sizeof(t0));
    expect(r.ok && r.committed, name + ": tier-0 runs");

    // ── tier-1 ──
    T1Kernel k = emit_fused_epilogue(t, t.stages[0]);
    expect(k.ok, name + ": tier-1 emit (" + (k.ok ? "ok" : k.error) + ")");
    if (!k.ok) { cudaFree(d_logits); return; }
    CUmodule mod; CUfunction fn = nvrtc_build(k.source, k.entry_name, &mod);
    std::uint8_t* d_mask = nullptr; RT_OK(cudaMalloc(&d_mask, V));
    RT_OK(cudaMemcpy(d_mask, maskv.data(), V, cudaMemcpyHostToDevice));
    std::uint32_t* d_state = nullptr; RT_OK(cudaMalloc(&d_state, 2 * sizeof(std::uint32_t)));
    RT_OK(cudaMemcpy(d_state, state.data(), 2 * sizeof(std::uint32_t), cudaMemcpyHostToDevice));
    int* d_out = nullptr; RT_OK(cudaMalloc(&d_out, sizeof(int)));
    int rows = 1, vocab = (int)V;
    CUdeviceptr p_logits = (CUdeviceptr)d_logits, p_mask = (CUdeviceptr)d_mask, p_state = (CUdeviceptr)d_state, p_out = (CUdeviceptr)d_out;
    std::vector<void*> params;
    for (const T1Arg& a : k.args) {
        switch (a.kind) {
            case T1ArgKind::Logits:     params.push_back(&p_logits); break;
            case T1ArgKind::ChanMaskU8: params.push_back(&p_mask); break;
            case T1ArgKind::RngState:   params.push_back(&p_state); break;
            case T1ArgKind::OutToken:   params.push_back(&p_out); break;
            case T1ArgKind::Rows:       params.push_back(&rows); break;
            case T1ArgKind::Vocab:      params.push_back(&vocab); break;
            default: std::printf("unexpected arg\n"); std::exit(2);
        }
    }
    CU_OK(cuLaunchKernel(fn, 1, 1, 1, 256, 1, 1, 0, nullptr, params.data(), nullptr));
    CU_OK(cuCtxSynchronize());
    int t1 = -1; RT_OK(cudaMemcpy(&t1, d_out, sizeof(int), cudaMemcpyDeviceToHost));
    expect((std::uint32_t)t1 == t0, name + ": tier-1 == tier-0 token (t0=" + std::to_string(t0) + " t1=" + std::to_string(t1) + ")");
    cuModuleUnload(mod);
    cudaFree(d_logits); cudaFree(d_mask); cudaFree(d_state); cudaFree(d_out);
}

}  // namespace

int main() {
    CU_OK(cuInit(0));
    CUdevice dev; CU_OK(cuDeviceGet(&dev, 0));
    CUcontext ctx; CU_OK(cuDevicePrimaryCtxRetain(&ctx, dev)); CU_OK(cuCtxSetCurrent(ctx));
    cudaDeviceProp p{}; cudaGetDeviceProperties(&p, 0);
    std::printf("PTIR tier-1 fusion gate — device: %s (sm_%d%d)\n\n", p.name, p.major, p.minor);

    run_case("greedy V=128", 128, false, 0.f, 0, 42);
    run_case("greedy V=32000", 32000, false, 0.f, 0, 7);
    run_case("temp+gumbel V=256 T=0.7 s0", 256, true, 1.0f / 0.7f, 0, 123456);
    run_case("temp+gumbel V=1024 T=1.0 s1", 1024, true, 1.0f, 1, 987654);
    run_case("temp+gumbel V=32000 T=0.8 s2", 32000, true, 1.0f / 0.8f, 2, 555);

    // §3 epilogue sampling-core fusion (masked Gumbel-max), the tier-1 step.
    run_masked_gumbel("section3-core V=64", 64, 123456u, 0u);
    run_masked_gumbel("section3-core V=4096", 4096, 777u, 3u);
    run_masked_gumbel("section3-core V=32000", 32000, 2024u, 9u);

    std::printf("\n==== tier-1 gate: %d passed, %d failed ====\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
