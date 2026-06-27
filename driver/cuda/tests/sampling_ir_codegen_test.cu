// End-to-end test for the sampling-IR codegen (codegen.cpp): lowers golden PSIR
// programs to a KernelDAG, NVRTC-compiles each kernel for the device arch, binds
// buffers (acting as delta's JIT + echo's executor), launches the DAG via the
// driver API, and validates the result against a CPU reference — including
// Gumbel-max parity with the sample_temp.cu noise scheme.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <algorithm>
#include <random>
#include <string>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <nvrtc.h>

#include "sampling_ir/codegen.hpp"
#include "sampling_ir/reader.hpp"
#include "sampling_ir/pie_standard_samplers.h"
#include "sampling_ir_golden_bytecode.h"

using namespace pie_cuda_driver::sampling_ir;

namespace {
int g_fail = 0, g_checks = 0;
void expect(bool c, const char* what) {
    ++g_checks;
    if (!c) { ++g_fail; std::fprintf(stderr, "  FAIL: %s\n", what); }
}

#define CU_CHECK(expr) do { CUresult _e=(expr); if(_e!=CUDA_SUCCESS){ const char* s=nullptr; \
    cuGetErrorString(_e,&s); std::fprintf(stderr,"CU err %s @ %d: %s\n", s?s:"?", __LINE__, #expr); \
    std::exit(2);} } while(0)
#define RT_CHECK(expr) do { cudaError_t _e=(expr); if(_e!=cudaSuccess){ \
    std::fprintf(stderr,"RT err %s @ %d\n", cudaGetErrorString(_e), __LINE__); std::exit(2);} } while(0)
#define NVRTC_CHECK(expr) do { nvrtcResult _e=(expr); if(_e!=NVRTC_SUCCESS){ \
    std::fprintf(stderr,"NVRTC err %s @ %d\n", nvrtcGetErrorString(_e), __LINE__); std::exit(2);} } while(0)

int g_cc_major = 0, g_cc_minor = 0;

CUmodule compile(const std::string& src) {
    nvrtcProgram prog;
    NVRTC_CHECK(nvrtcCreateProgram(&prog, src.c_str(), "psir.cu", 0, nullptr, nullptr));
    char arch[64];
    std::snprintf(arch, sizeof(arch), "--gpu-architecture=compute_%d%d", g_cc_major, g_cc_minor);
    const char* opts[] = {arch, "--std=c++17"};
    nvrtcResult cr = nvrtcCompileProgram(prog, 2, opts);
    if (cr != NVRTC_SUCCESS) {
        size_t n = 0; nvrtcGetProgramLogSize(prog, &n);
        std::string log(n, '\0'); nvrtcGetProgramLog(prog, log.data());
        std::fprintf(stderr, "NVRTC compile failed:\n%s\nSOURCE:\n%s\n", log.c_str(), src.c_str());
        std::exit(2);
    }
    size_t sz = 0; NVRTC_CHECK(nvrtcGetPTXSize(prog, &sz));
    std::string ptx(sz, '\0'); NVRTC_CHECK(nvrtcGetPTX(prog, ptx.data()));
    NVRTC_CHECK(nvrtcDestroyProgram(&prog));
    CUmodule m; CU_CHECK(cuModuleLoadData(&m, ptx.c_str()));
    return m;
}

std::uint32_t phys_bytes(const BufferDecl& b) {
    if (b.cls == BufferClass::IntrinsicLogits) return 2;  // bf16
    return dtype_size(b.dtype);
}

// A tiny SplitMix64 Gumbel reference (mirrors prelude / sample_temp.cu).
float gumbel_ref(std::uint32_t seed, int j) {
    std::uint64_t s = (std::uint64_t)seed ^ 0xA5A5A5A5ULL;
    std::uint64_t x = s + 0x9E3779B97F4A7C15ULL * (std::uint64_t)(j + 1);
    x ^= x >> 27; x *= 0x3C79AC492BA7B653ULL;
    x ^= x >> 33; x *= 0x1C69B3F74AC4AE35ULL;
    x ^= x >> 27;
    std::uint32_t bits = (std::uint32_t)(x >> 40);
    float u = ((float)bits + 0.5f) * (1.0f / 16777216.0f);
    return -std::log(-std::log(u));
}

// Run a lowered DAG. `logits` (length vocab, host f32) is uploaded as bf16 to
// the IntrinsicLogits buffer; `seed` to any host U32 scalar buffer. Returns the
// token written to the (scalar I32) Output buffer.
struct RunResult { std::int32_t token; };

RunResult run_dag(const KernelDAG& dag, const std::vector<float>& logits, std::uint32_t seed) {
    std::vector<CUdeviceptr> dptr(dag.buffers.size(), 0);
    int vocab = (int)logits.size();

    // Allocate + fill every buffer.
    for (const BufferDecl& b : dag.buffers) {
        size_t bytes = (size_t)b.elem_count * phys_bytes(b);
        if (bytes == 0) bytes = 4;
        CU_CHECK(cuMemAlloc(&dptr[b.id], bytes));
        CU_CHECK(cuMemsetD8(dptr[b.id], 0, bytes));
        if (b.cls == BufferClass::IntrinsicLogits) {
            std::vector<__nv_bfloat16> hb(vocab);
            for (int i = 0; i < vocab; ++i) hb[i] = __float2bfloat16(logits[i]);
            CU_CHECK(cuMemcpyHtoD(dptr[b.id], hb.data(), (size_t)vocab * 2));
        } else if (b.cls == BufferClass::HostSubmit || b.cls == BufferClass::HostLate) {
            // MVP host scalar = the per-row u32 seed.
            std::uint32_t v = seed;
            CU_CHECK(cuMemcpyHtoD(dptr[b.id], &v, 4));
        }
    }

    // Compile + launch each kernel in order, honoring its launch geometry.
    for (const KernelDesc& kd : dag.kernels) {
        CUmodule m = compile(kd.source);
        CUfunction fn; CU_CHECK(cuModuleGetFunction(&fn, m, kd.entry_name.c_str()));
        std::vector<void*> args;
        for (const KernelArg& a : kd.args) args.push_back(&dptr[a.buffer]);  // MVP: all Buffer
        std::uint32_t len = 0;
        if (kd.shape == LaunchShape::GridStrideOverLen && kd.len_buffer < dag.buffers.size())
            len = dag.buffers[kd.len_buffer].elem_count;
        LaunchDims d = compute_launch_dims(kd.shape, /*num_rows=*/1, (std::uint32_t)vocab,
                                           len, kd.custom_grid, kd.custom_block);
        CU_CHECK(cuLaunchKernel(fn, d.grid_x, 1, 1, d.block_x, 1, 1, 0, nullptr, args.data(), nullptr));
        CU_CHECK(cuCtxSynchronize());
        CU_CHECK(cuModuleUnload(m));
    }

    // Read the scalar Output buffer (token).
    RunResult r{-1};
    for (const BufferDecl& b : dag.buffers) {
        if (b.cls == BufferClass::Output) {
            CU_CHECK(cuMemcpyDtoH(&r.token, dptr[b.id], 4));
            break;
        }
    }
    for (CUdeviceptr p : dptr) if (p) cuMemFree(p);
    return r;
}

void test_argmax() {
    std::printf("[codegen argmax]\n");
    LowerResult lr = lower_bytecode((const std::uint8_t*)GV_ARGMAX, sizeof(GV_ARGMAX));
    expect(lr.ok, "argmax lowers");
    expect(lr.dag.kernels.size() == 1, "argmax = 1 kernel");
    if (!lr.ok) return;

    // GV_ARGMAX logits are Vector{32000}; use that vocab.
    const int vocab = 32000;
    std::vector<float> logits(vocab);
    std::mt19937 rng(5);
    std::uniform_real_distribution<float> d(-8.f, 8.f);
    for (auto& v : logits) v = d(rng);
    logits[12345] = 50.f;  // unique max (survives bf16 rounding)

    RunResult r = run_dag(lr.dag, logits, 0);
    // CPU argmax over bf16-rounded logits (matches device read).
    int best = 0; float bv = -INFINITY;
    for (int i = 0; i < vocab; ++i) {
        float v = __bfloat162float(__float2bfloat16(logits[i]));
        if (v > bv) { bv = v; best = i; }
    }
    expect(r.token == best, "argmax token matches CPU reference");
}

void test_temp_parity() {
    std::printf("[codegen temp Gumbel-max parity]\n");
    LowerResult lr = lower_bytecode((const std::uint8_t*)GV_TEMP, sizeof(GV_TEMP));
    expect(lr.ok, "temp lowers");
    if (!lr.ok) { std::fprintf(stderr, "  err: %s\n", lr.error.c_str()); return; }
    expect(lr.dag.kernels.size() == 1, "temp = 1 kernel");

    const int vocab = 128;  // GV_TEMP logits Vector{128}
    const float T = 0.8f;   // baked const
    const std::uint32_t seed = 0x1234abcdu;
    std::vector<float> logits(vocab);
    std::mt19937 rng(9);
    std::uniform_real_distribution<float> d(-4.f, 4.f);
    for (auto& v : logits) v = d(rng);

    RunResult r = run_dag(lr.dag, logits, seed);

    // CPU Gumbel-max reference: argmax_j( logit_j/T + gumbel(seed,j) ), with the
    // logit read as bf16 (matching the device path), lowest-index tie-break.
    int best = 0; float bv = -INFINITY;
    for (int j = 0; j < vocab; ++j) {
        float lf = __bfloat162float(__float2bfloat16(logits[j]));
        float score = lf / T + gumbel_ref(seed, j);
        if (score > bv) { bv = score; best = j; }
    }
    expect(r.token == best, "temp Gumbel-max token matches sample_temp-parity reference");
}

void test_minp_parity() {
    std::printf("[codegen temp+min-p logit-space parity]\n");
    LowerResult lr = lower_bytecode((const std::uint8_t*)GV_MINP, sizeof(GV_MINP));
    expect(lr.ok, "min-p lowers");
    if (!lr.ok) { std::fprintf(stderr, "  err: %s\n", lr.error.c_str()); return; }

    const int vocab = 256;          // GV_MINP logits Vector{256}
    const float T = 0.7f;           // baked const
    const float min_p = 0.1f;       // baked const
    const std::uint32_t seed = 0xBADC0DEu;
    std::vector<float> logits(vocab);
    std::mt19937 rng(17);
    std::uniform_real_distribution<float> d(-6.f, 6.f);
    for (auto& v : logits) v = d(rng);

    RunResult r = run_dag(lr.dag, logits, seed);

    // CPU reference faithful to sample_temp.cu's min-p path (logit-space):
    //   inv_T = 1/T ; max_logit over bf16 logits ; threshold = max_logit+log(min_p)
    //   keep = logit >= threshold ; score = logit*inv_T + gumbel ; argmax over kept.
    std::vector<float> lf(vocab);
    float mx = -INFINITY;
    for (int j = 0; j < vocab; ++j) { lf[j] = __bfloat162float(__float2bfloat16(logits[j])); mx = std::max(mx, lf[j]); }
    float inv_T = 1.0f / T;
    float thr = mx + std::log(min_p);
    int best = 0; float bv = -INFINITY;
    for (int j = 0; j < vocab; ++j) {
        float score = (lf[j] >= thr) ? (lf[j] * inv_T + gumbel_ref(seed, j)) : -INFINITY;
        if (score > bv) { bv = score; best = j; }
    }
    expect(r.token == best, "min-p token matches sample_temp-parity reference");
}

void test_topk_parity() {
    std::printf("[codegen top-k + Gumbel-max]\n");
    LowerResult lr = lower_bytecode((const std::uint8_t*)GV_TOPK, sizeof(GV_TOPK));
    expect(lr.ok, "top-k lowers");
    if (!lr.ok) { std::fprintf(stderr, "  err: %s\n", lr.error.c_str()); return; }

    const int vocab = 256, k = 10;
    const float T = 0.9f;
    const std::uint32_t seed = 0x5AFE5EEDu;
    std::vector<float> logits(vocab);
    std::mt19937 rng(21);
    std::uniform_real_distribution<float> d(-6.f, 6.f);
    for (auto& v : logits) v = d(rng);
    for (int i = 0; i < vocab; ++i) logits[i] += 1e-3f * i;  // distinct

    RunResult r = run_dag(lr.dag, logits, seed);

    std::vector<float> lf(vocab);
    for (int j = 0; j < vocab; ++j) lf[j] = __bfloat162float(__float2bfloat16(logits[j]));
    std::vector<float> sorted = lf;
    std::sort(sorted.begin(), sorted.end(), std::greater<float>());
    float kth = sorted[k - 1];
    float inv_T = 1.0f / T;
    int best = 0; float bv = -INFINITY;
    for (int j = 0; j < vocab; ++j) {
        float score = (lf[j] >= kth) ? (lf[j] * inv_T + gumbel_ref(seed, j)) : -INFINITY;
        if (score > bv) { bv = score; best = j; }
    }
    expect(r.token == best, "top-k token matches CPU reference");
}

void test_topp_parity() {
    std::printf("[codegen top-p nucleus + Gumbel-max]\n");
    LowerResult lr = lower_bytecode((const std::uint8_t*)GV_TOPP, sizeof(GV_TOPP));
    expect(lr.ok, "top-p lowers");
    if (!lr.ok) { std::fprintf(stderr, "  err: %s\n", lr.error.c_str()); return; }

    const int vocab = 256;
    const float T = 0.9f, p = 0.9f;
    const std::uint32_t seed = 0xC0FFEE11u;
    std::vector<float> logits(vocab);
    std::mt19937 rng(23);
    std::uniform_real_distribution<float> d(-5.f, 5.f);
    for (auto& v : logits) v = d(rng);

    RunResult r = run_dag(lr.dag, logits, seed);

    // CPU reference: softmax(bf16 logits) -> nucleus threshold (largest tau with
    // mass(prob>=tau) >= p) -> keep prob>=tau -> gumbel-max over kept logits.
    std::vector<float> lf(vocab), prob(vocab);
    float mx = -INFINITY;
    for (int j = 0; j < vocab; ++j) { lf[j] = __bfloat162float(__float2bfloat16(logits[j])); mx = std::max(mx, lf[j]); }
    double Z = 0;
    for (int j = 0; j < vocab; ++j) { prob[j] = std::exp(lf[j] - mx); Z += prob[j]; }
    for (auto& v : prob) v = (float)(v / Z);
    std::vector<float> ps = prob;
    std::sort(ps.begin(), ps.end(), std::greater<float>());
    double cum = 0; float tau = ps.back();
    for (int j = 0; j < vocab; ++j) { cum += ps[j]; if (cum >= p) { tau = ps[j]; break; } }
    float inv_T = 1.0f / T;
    int best = 0; float bv = -INFINITY;
    for (int j = 0; j < vocab; ++j) {
        float score = (prob[j] >= tau) ? (lf[j] * inv_T + gumbel_ref(seed, j)) : -INFINITY;
        if (score > bv) { bv = score; best = j; }
    }
    expect(r.token == best, "top-p token matches CPU nucleus reference");
}

void test_barrier() {
    std::printf("[codegen data-dependent barrier (reduce->gather-row)]\n");
    LowerResult lr = lower_bytecode((const std::uint8_t*)GV_BARRIER, sizeof(GV_BARRIER));
    expect(lr.ok, "barrier lowers");
    if (!lr.ok) { std::fprintf(stderr, "  err: %s\n", lr.error.c_str()); return; }
    expect(lr.dag.kernels.size() == 2, "barrier cut -> 2 kernels");

    // The reduced scalar index must be an intermediate buffer threaded across
    // both kernels (the cross-kernel device-pointer hand-off delta relies on).
    BufferId idx_buf = UINT32_MAX;
    for (const BufferDecl& b : lr.dag.buffers)
        if (b.cls == BufferClass::Intermediate && b.dtype == DType::I32 && b.elem_count == 1)
            idx_buf = b.id;
    bool in_k0 = false, in_k1 = false;
    if (lr.dag.kernels.size() == 2) {
        for (const KernelArg& a : lr.dag.kernels[0].args) if (a.buffer == idx_buf) in_k0 = true;
        for (const KernelArg& a : lr.dag.kernels[1].args) if (a.buffer == idx_buf) in_k1 = true;
    }
    expect(idx_buf != UINT32_MAX && in_k0 && in_k1, "reduced index buffer shared across both kernels");

    // Run it: logits[64], resid[8][64]. token row = argmax(logits); output row
    // must equal resid[row, :].
    const int vocab = 64, rows = 8;
    std::vector<float> logits(vocab);
    std::mt19937 rng(29);
    std::uniform_real_distribution<float> d(-5.f, 5.f);
    for (auto& v : logits) v = d(rng);
    logits[5] = 20.f;  // argmax row = 5 (valid row in the 8-row resid matrix)
    std::vector<float> resid(rows * vocab);
    for (int i = 0; i < rows * vocab; ++i) resid[i] = (float)i * 0.5f;

    // Manual bind/launch (resid matrix host input + f32 vector output).
    std::vector<CUdeviceptr> dptr(lr.dag.buffers.size(), 0);
    for (const BufferDecl& b : lr.dag.buffers) {
        size_t bytes = (size_t)b.elem_count * phys_bytes(b);
        if (bytes == 0) bytes = 4;
        CU_CHECK(cuMemAlloc(&dptr[b.id], bytes));
        CU_CHECK(cuMemsetD8(dptr[b.id], 0, bytes));
        if (b.cls == BufferClass::IntrinsicLogits) {
            std::vector<__nv_bfloat16> hb(vocab);
            for (int i = 0; i < vocab; ++i) hb[i] = __float2bfloat16(logits[i]);
            CU_CHECK(cuMemcpyHtoD(dptr[b.id], hb.data(), (size_t)vocab * 2));
        } else if (b.cls == BufferClass::HostSubmit) {
            CU_CHECK(cuMemcpyHtoD(dptr[b.id], resid.data(), resid.size() * 4));
        }
    }
    for (const KernelDesc& kd : lr.dag.kernels) {
        CUmodule m = compile(kd.source);
        CUfunction fn; CU_CHECK(cuModuleGetFunction(&fn, m, kd.entry_name.c_str()));
        std::vector<void*> args;
        for (const KernelArg& a : kd.args) args.push_back(&dptr[a.buffer]);
        CU_CHECK(cuLaunchKernel(fn, 1, 1, 1, 256, 1, 1, 0, nullptr, args.data(), nullptr));
        CU_CHECK(cuCtxSynchronize());
        CU_CHECK(cuModuleUnload(m));
    }
    std::vector<float> out(vocab, -1.f);
    for (const BufferDecl& b : lr.dag.buffers)
        if (b.cls == BufferClass::Output) { CU_CHECK(cuMemcpyDtoH(out.data(), dptr[b.id], (size_t)vocab * 4)); break; }
    for (CUdeviceptr p : dptr) if (p) cuMemFree(p);

    int jrow = 0; float bv = -INFINITY;
    for (int i = 0; i < vocab; ++i) { float v = __bfloat162float(__float2bfloat16(logits[i])); if (v > bv) { bv = v; jrow = i; } }
    bool row_ok = true;
    for (int c = 0; c < vocab; ++c) if (out[c] != resid[jrow * vocab + c]) row_ok = false;
    expect(row_ok, "gather-row output == resid[argmax_row, :]");
}

void test_matrix_argmax() {
    std::printf("[codegen matrix per-row argmax]\n");
    LowerResult lr = lower_bytecode((const std::uint8_t*)GV_MATARGMAX, sizeof(GV_MATARGMAX));
    expect(lr.ok, "matrix argmax lowers");
    if (!lr.ok) { std::fprintf(stderr, "  err: %s\n", lr.error.c_str()); return; }
    // The matrix-reduce kernel launches grid=rows (Custom shape).
    bool has_matrix_kernel = false;
    for (const KernelDesc& k : lr.dag.kernels)
        if (k.shape == LaunchShape::Custom && k.custom_grid == 4) has_matrix_kernel = true;
    expect(has_matrix_kernel, "matrix kernel launches grid=rows (4)");

    const int rows = 4, V = 128;
    std::vector<float> logits(rows*V);
    std::mt19937 rng(41);
    std::uniform_real_distribution<float> d(-6.f,6.f);
    for (auto& v: logits) v = d(rng);
    for (int r=0;r<rows;++r) logits[r*V + (r*29+5)%V] = 30.f;  // distinct per-row max

    // Bind matrix logits as bf16 + read the Vector{rows} I32 output.
    std::vector<CUdeviceptr> dptr(lr.dag.buffers.size(), 0);
    for (const BufferDecl& b : lr.dag.buffers) {
        size_t bytes = (size_t)b.elem_count * phys_bytes(b);
        if (bytes == 0) bytes = 4;
        CU_CHECK(cuMemAlloc(&dptr[b.id], bytes));
        CU_CHECK(cuMemsetD8(dptr[b.id], 0, bytes));
        if (b.cls == BufferClass::IntrinsicLogits) {
            std::vector<__nv_bfloat16> hb(rows*V);
            for (int i=0;i<rows*V;++i) hb[i] = __float2bfloat16(logits[i]);
            CU_CHECK(cuMemcpyHtoD(dptr[b.id], hb.data(), (size_t)rows*V*2));
        }
    }
    for (const KernelDesc& kd : lr.dag.kernels) {
        CUmodule m = compile(kd.source);
        CUfunction fn; CU_CHECK(cuModuleGetFunction(&fn, m, kd.entry_name.c_str()));
        std::vector<void*> args;
        for (const KernelArg& a : kd.args) args.push_back(&dptr[a.buffer]);
        LaunchDims dd = compute_launch_dims(kd.shape, 1, V, 0, kd.custom_grid, kd.custom_block);
        CU_CHECK(cuLaunchKernel(fn, dd.grid_x, 1, 1, dd.block_x, 1, 1, 0, nullptr, args.data(), nullptr));
        CU_CHECK(cuCtxSynchronize());
        CU_CHECK(cuModuleUnload(m));
    }
    std::vector<int> tok(rows, -1);
    for (const BufferDecl& b : lr.dag.buffers)
        if (b.cls == BufferClass::Output) { CU_CHECK(cuMemcpyDtoH(tok.data(), dptr[b.id], (size_t)rows*4)); break; }
    for (CUdeviceptr p : dptr) if (p) cuMemFree(p);

    bool ok = true;
    for (int r=0;r<rows;++r) {
        int best=0; float bv=-INFINITY;
        for (int j=0;j<V;++j){ float v=__bfloat162float(__float2bfloat16(logits[r*V+j])); if(v>bv){bv=v;best=j;} }
        if (tok[r] != best) ok = false;
    }
    expect(ok, "per-row argmax tokens match CPU reference");
}

// Run a spec-verify DAG: matrix target logits (rows=k+1) + host draft_padded
// Vector{k+1} I32 → Vector{k+1} I32 output (verified prefix + -1 sentinel).
void run_spec_dag(const KernelDAG& dag, int kp1, int V,
                  const std::vector<float>& tlog, const std::vector<int>& draft_padded,
                  std::vector<int>& out) {
    std::vector<CUdeviceptr> dptr(dag.buffers.size(), 0);
    for (const BufferDecl& b : dag.buffers) {
        size_t bytes = (size_t)b.elem_count * phys_bytes(b);
        if (bytes == 0) bytes = 4;
        CU_CHECK(cuMemAlloc(&dptr[b.id], bytes));
        CU_CHECK(cuMemsetD8(dptr[b.id], 0, bytes));
        if (b.cls == BufferClass::IntrinsicLogits) {
            std::vector<__nv_bfloat16> hb(kp1*V);
            for (int i=0;i<kp1*V;++i) hb[i] = __float2bfloat16(tlog[i]);
            CU_CHECK(cuMemcpyHtoD(dptr[b.id], hb.data(), (size_t)kp1*V*2));
        } else if (b.cls == BufferClass::HostSubmit) {
            CU_CHECK(cuMemcpyHtoD(dptr[b.id], draft_padded.data(), draft_padded.size()*4));
        }
    }
    for (const KernelDesc& kd : dag.kernels) {
        CUmodule m = compile(kd.source);
        CUfunction fn; CU_CHECK(cuModuleGetFunction(&fn, m, kd.entry_name.c_str()));
        std::vector<void*> args;
        for (const KernelArg& a : kd.args) args.push_back(&dptr[a.buffer]);
        LaunchDims dd = compute_launch_dims(kd.shape, 1, V, 0, kd.custom_grid, kd.custom_block);
        CU_CHECK(cuLaunchKernel(fn, dd.grid_x, 1, 1, dd.block_x, 1, 1, 0, nullptr, args.data(), nullptr));
        CU_CHECK(cuCtxSynchronize());
        CU_CHECK(cuModuleUnload(m));
    }
    out.assign(kp1, -99);
    for (const BufferDecl& b : dag.buffers)
        if (b.cls == BufferClass::Output) { CU_CHECK(cuMemcpyDtoH(out.data(), dptr[b.id], (size_t)kp1*4)); break; }
    for (CUdeviceptr p : dptr) if (p) cuMemFree(p);
}

void test_spec_greedy_codegen() {
    std::printf("[codegen greedy spec-verify DAG]\n");
    LowerResult lr = lower_bytecode((const std::uint8_t*)GV_SPECGREEDY, sizeof(GV_SPECGREEDY));
    expect(lr.ok, "spec-verify lowers");
    if (!lr.ok) { std::fprintf(stderr, "  err: %s\n", lr.error.c_str()); return; }
    // Matrix-argmax kernel (grid=k+1) + a vector verify kernel (grid=1).
    bool has_mat = false, has_vec = false;
    for (const KernelDesc& k : lr.dag.kernels) {
        if (k.shape == LaunchShape::Custom && k.custom_grid == 5) has_mat = true;
        if (k.shape == LaunchShape::OneBlockPerRow) has_vec = true;
    }
    expect(has_mat && has_vec, "DAG = matrix-argmax kernel (grid=k+1) + vector kernel");

    const int k = 4, kp1 = 5, V = 64;
    std::mt19937 rng(53);
    std::uniform_real_distribution<float> d(-5.f,5.f);

    auto cpu_ref = [&](const std::vector<float>& tlog, const std::vector<int>& draft, std::vector<int>& ref){
        std::vector<int> ttok(kp1);
        for (int r=0;r<kp1;++r){ int ai=0; float mv=-INFINITY; for(int j=0;j<V;++j){ float v=__bfloat162float(__float2bfloat16(tlog[r*V+j])); if(v>mv){mv=v;ai=j;} } ttok[r]=ai; }
        int alen=0; while(alen<k && ttok[alen]==draft[alen]) ++alen;
        ref.assign(kp1,-1);
        for(int i=0;i<=alen;++i) ref[i]=ttok[i];
    };

    auto run_case = [&](const char* name, int reject_at){
        std::vector<float> tlog(kp1*V);
        for (auto& v: tlog) v = d(rng);
        // target argmax per row
        std::vector<int> ttok(kp1);
        for (int r=0;r<kp1;++r){ int ai=0; float mv=-INFINITY; for(int j=0;j<V;++j){ float v=__bfloat162float(__float2bfloat16(tlog[r*V+j])); if(v>mv){mv=v;ai=j;} } ttok[r]=ai; }
        // draft matches argmax except at reject_at (reject_at=k → all accepted)
        std::vector<int> draft_padded(kp1);
        for (int i=0;i<k;++i) draft_padded[i] = (i==reject_at) ? (ttok[i]+1)%V : ttok[i];
        draft_padded[k] = -1;  // sentinel
        std::vector<int> got, ref;
        run_spec_dag(lr.dag, kp1, V, tlog, draft_padded, got);
        cpu_ref(tlog, draft_padded, ref);
        expect(got == ref, name);
    };

    run_case("all-accepted (k drafts + bonus)", k);
    run_case("reject at position 0", 0);
    run_case("reject at position 2", 2);
    run_case("reject at last draft", k-1);
}

void test_rowbcast_codegen() {
    std::printf("[codegen RowBroadcast per-row normalization]\n");
    auto approx = [](float a, float b){ return std::fabs(a-b) <= 1e-3f + 1e-3f*std::fabs(b); };
    LowerResult lr = lower_bytecode((const std::uint8_t*)GV_ROWBCAST, sizeof(GV_ROWBCAST));
    expect(lr.ok, "rowbcast lowers");
    if (!lr.ok) { std::fprintf(stderr, "  err: %s\n", lr.error.c_str()); return; }

    const int rows = 3, cols = 8;
    std::vector<float> logits(rows*cols);
    std::mt19937 rng(61);
    std::uniform_real_distribution<float> d(-2.f,2.f);
    for (auto& v: logits) v = d(rng);

    std::vector<CUdeviceptr> dptr(lr.dag.buffers.size(), 0);
    for (const BufferDecl& b : lr.dag.buffers) {
        size_t bytes = (size_t)b.elem_count * phys_bytes(b);
        if (bytes == 0) bytes = 4;
        CU_CHECK(cuMemAlloc(&dptr[b.id], bytes));
        CU_CHECK(cuMemsetD8(dptr[b.id], 0, bytes));
        if (b.cls == BufferClass::IntrinsicLogits) {
            std::vector<__nv_bfloat16> hb(rows*cols);
            for (int i=0;i<rows*cols;++i) hb[i] = __float2bfloat16(logits[i]);
            CU_CHECK(cuMemcpyHtoD(dptr[b.id], hb.data(), (size_t)rows*cols*2));
        }
    }
    for (const KernelDesc& kd : lr.dag.kernels) {
        CUmodule m = compile(kd.source);
        CUfunction fn; CU_CHECK(cuModuleGetFunction(&fn, m, kd.entry_name.c_str()));
        std::vector<void*> args;
        for (const KernelArg& a : kd.args) args.push_back(&dptr[a.buffer]);
        LaunchDims dd = compute_launch_dims(kd.shape, 1, cols, 0, kd.custom_grid, kd.custom_block);
        CU_CHECK(cuLaunchKernel(fn, dd.grid_x, 1, 1, dd.block_x, 1, 1, 0, nullptr, args.data(), nullptr));
        CU_CHECK(cuCtxSynchronize());
        CU_CHECK(cuModuleUnload(m));
    }
    std::vector<float> probs(rows*cols, -1.f);
    for (const BufferDecl& b : lr.dag.buffers)
        if (b.cls == BufferClass::Output) { CU_CHECK(cuMemcpyDtoH(probs.data(), dptr[b.id], (size_t)rows*cols*4)); break; }
    for (CUdeviceptr p : dptr) if (p) cuMemFree(p);

    // Each row must be a valid softmax: sums to 1 and matches exp/sum reference.
    bool ok = true;
    for (int r=0;r<rows;++r) {
        double Z = 0; std::vector<float> e(cols);
        for (int j=0;j<cols;++j){ e[j] = std::exp(__bfloat162float(__float2bfloat16(logits[r*cols+j]))); Z += e[j]; }
        double smass = 0;
        for (int j=0;j<cols;++j){ smass += probs[r*cols+j]; if (!approx(probs[r*cols+j], (float)(e[j]/Z))) ok = false; }
        if (!approx((float)smass, 1.0f)) ok = false;
    }
    expect(ok, "RowBroadcast per-row softmax normalizes each row to 1");
}

void test_late_barrier_stamp() {
    std::printf("[codegen late-input first-consuming-kernel stamp]\n");
    LowerResult lr = lower_bytecode((const std::uint8_t*)GV_LATEBARRIER, sizeof(GV_LATEBARRIER));
    expect(lr.ok, "late-barrier lowers");
    if (!lr.ok) { std::fprintf(stderr, "  err: %s\n", lr.error.c_str()); return; }
    expect(lr.dag.kernels.size() == 2, "argmax + gather-row = 2 kernels");

    // The HostLate buffer must be stamped with first_consuming_kernel == 1
    // (read only by gather-row in kernel 1, past the argmax barrier in kernel 0).
    bool found_late = false;
    for (const BufferDecl& b : lr.dag.buffers) {
        if (b.cls == BufferClass::HostLate) {
            found_late = true;
            expect(b.first_consuming_kernel == 1, "late buffer first-consuming-kernel == 1");
        } else {
            // non-late buffers carry the unset sentinel.
            expect(b.first_consuming_kernel == 0xFFFFFFFFu, "non-late buffer has no stamp");
        }
    }
    expect(found_late, "HostLate buffer present");
}

// Lower `bytecode` batched, bind B rows + per-row seeds, launch ONE grid=B
// kernel-DAG, return per-row tokens. Mirrors how delta's JIT/echo's executor
// drive the batched path (Param{0}=B dynamic grid, per-row buffers).
bool run_batched(const std::uint8_t* bc, size_t bc_len, int B, int V,
                 const std::vector<float>& logits, const std::vector<std::uint32_t>& seeds,
                 std::vector<int>& tokens, std::string& err) {
    LowerResult lr = lower_bytecode(bc, bc_len, LowerOptions{/*batched=*/true});
    if (!lr.ok) { err = lr.error; return false; }
    std::vector<CUdeviceptr> dptr(lr.dag.buffers.size(), 0);
    BufferId out_buf = 0;
    for (const BufferDecl& b : lr.dag.buffers) {
        std::uint32_t total = (b.batched ? (std::uint32_t)B : 1u) * b.elem_count;
        size_t bytes = (size_t)total * (b.cls == BufferClass::IntrinsicLogits ? 2 : dtype_size(b.dtype));
        if (bytes == 0) bytes = 4;
        CU_CHECK(cuMemAlloc(&dptr[b.id], bytes));
        CU_CHECK(cuMemsetD8(dptr[b.id], 0, bytes));
        if (b.cls == BufferClass::IntrinsicLogits) {
            std::vector<__nv_bfloat16> hb(B*V);
            for (int i=0;i<B*V;++i) hb[i] = __float2bfloat16(logits[i]);
            CU_CHECK(cuMemcpyHtoD(dptr[b.id], hb.data(), (size_t)B*V*2));
        } else if (b.cls == BufferClass::HostSubmit) {
            CU_CHECK(cuMemcpyHtoD(dptr[b.id], seeds.data(), (size_t)B*4));
        } else if (b.cls == BufferClass::Output) {
            out_buf = b.id;
        }
    }
    for (const KernelDesc& kd : lr.dag.kernels) {
        CUmodule m = compile(kd.source);
        CUfunction fn; CU_CHECK(cuModuleGetFunction(&fn, m, kd.entry_name.c_str()));
        std::vector<void*> args;
        for (const KernelArg& a : kd.args) args.push_back(&dptr[a.buffer]);
        LaunchDims dd = compute_launch_dims(kd.shape, B, V, 0, kd.custom_grid, kd.custom_block);
        CU_CHECK(cuLaunchKernel(fn, dd.grid_x, 1, 1, dd.block_x, 1, 1, 0, nullptr, args.data(), nullptr));
        CU_CHECK(cuCtxSynchronize());
        CU_CHECK(cuModuleUnload(m));
    }
    tokens.assign(B, -99);
    CU_CHECK(cuMemcpyDtoH(tokens.data(), dptr[out_buf], (size_t)B*4));
    for (CUdeviceptr p : dptr) if (p) cuMemFree(p);
    return true;
}

void test_batched_temp() {
    std::printf("[codegen batched (M>1) temp — one launch, grid=num_rows]\n");
    LowerResult lr = lower_bytecode((const std::uint8_t*)GV_TEMP, sizeof(GV_TEMP),
                                    LowerOptions{/*batched=*/true});
    expect(lr.ok, "batched temp lowers");
    if (!lr.ok) { std::fprintf(stderr, "  err: %s\n", lr.error.c_str()); return; }
    expect(lr.dag.kernels.size() == 1, "batched = single kernel");
    expect(lr.dag.kernels[0].shape == LaunchShape::OneBlockPerRow,
           "batched kernel shape = OneBlockPerRow (dynamic grid=num_rows)");
    bool all_batched = true;
    for (const BufferDecl& b : lr.dag.buffers) if (!b.batched) all_batched = false;
    expect(all_batched, "all buffers marked batched (per-row elem_count)");

    const int B = 64, V = 128;
    const float T = 0.8f;
    std::vector<float> logits(B*V);
    std::vector<std::uint32_t> seeds(B);
    std::mt19937 rng(71);
    std::uniform_real_distribution<float> d(-4.f,4.f);
    for (auto& v: logits) v = d(rng);
    for (int r=0;r<B;++r) seeds[r] = 0x1000u + r*2654435761u;

    std::vector<int> tok; std::string err;
    expect(run_batched((const std::uint8_t*)GV_TEMP, sizeof(GV_TEMP), B, V, logits, seeds, tok, err),
           "batched temp runs");
    bool ok = true;
    for (int r=0;r<B;++r) {
        int best=0; float bv=-INFINITY;
        for (int j=0;j<V;++j){
            float lf = __bfloat162float(__float2bfloat16(logits[r*V+j]));
            float score = lf/T + gumbel_ref(seeds[r], j);
            if (score > bv) { bv = score; best = j; }
        }
        if (tok[r] != best) ok = false;
    }
    expect(ok, "every batch row's token matches per-row Gumbel-max reference");
}

void test_batched_minp() {
    std::printf("[codegen batched (M>1) temp+min-p]\n");
    const int B = 48, V = 256;
    const float T = 0.7f, min_p = 0.1f;
    std::vector<float> logits(B*V);
    std::vector<std::uint32_t> seeds(B);
    std::mt19937 rng(73);
    std::uniform_real_distribution<float> d(-6.f,6.f);
    for (auto& v: logits) v = d(rng);
    for (int r=0;r<B;++r) seeds[r] = 0xBEEF0000u + r*40503u;
    std::vector<int> tok; std::string err;
    expect(run_batched((const std::uint8_t*)GV_MINP, sizeof(GV_MINP), B, V, logits, seeds, tok, err),
           "batched min-p runs");
    bool ok = true;
    for (int r=0;r<B;++r) {
        std::vector<float> lf(V); float mx=-INFINITY;
        for (int j=0;j<V;++j){ lf[j]=__bfloat162float(__float2bfloat16(logits[r*V+j])); mx=std::max(mx,lf[j]); }
        float inv_T=1.f/T, thr=mx+std::log(min_p);
        int best=0; float bv=-INFINITY;
        for (int j=0;j<V;++j){ float sc=(lf[j]>=thr)?(lf[j]*inv_T+gumbel_ref(seeds[r],j)):-INFINITY; if(sc>bv){bv=sc;best=j;} }
        if (tok[r]!=best) ok=false;
    }
    expect(ok, "every batch row matches per-row min-p Gumbel-max reference");
}

void test_batched_topk() {
    std::printf("[codegen batched (M>1) top-k]\n");
    const int B = 32, V = 256, k = 10;
    const float T = 0.9f;
    std::vector<float> logits(B*V);
    std::vector<std::uint32_t> seeds(B);
    std::mt19937 rng(79);
    std::uniform_real_distribution<float> d(-6.f,6.f);
    for (auto& v: logits) v = d(rng);
    for (int r=0;r<B;++r) { for (int j=0;j<V;++j) logits[r*V+j] += 1e-3f*j; seeds[r] = 0x5A5A0000u + r*2246822519u; }
    std::vector<int> tok; std::string err;
    expect(run_batched((const std::uint8_t*)GV_TOPK, sizeof(GV_TOPK), B, V, logits, seeds, tok, err),
           "batched top-k runs");
    bool ok = true;
    for (int r=0;r<B;++r) {
        std::vector<float> lf(V);
        for (int j=0;j<V;++j) lf[j]=__bfloat162float(__float2bfloat16(logits[r*V+j]));
        std::vector<float> sorted=lf; std::sort(sorted.begin(),sorted.end(),std::greater<float>());
        float kth=sorted[k-1], inv_T=1.f/T;
        int best=0; float bv=-INFINITY;
        for (int j=0;j<V;++j){ float sc=(lf[j]>=kth)?(lf[j]*inv_T+gumbel_ref(seeds[r],j)):-INFINITY; if(sc>bv){bv=sc;best=j;} }
        if (tok[r]!=best) ok=false;
    }
    expect(ok, "every batch row matches per-row top-k reference");
}

void test_gathercols_codegen() {
    std::printf("[codegen GatherCols per-row element gather]\n");
    LowerResult lr = lower_bytecode((const std::uint8_t*)GV_GATHERCOLS, sizeof(GV_GATHERCOLS));
    expect(lr.ok, "gathercols lowers");
    if (!lr.ok) { std::fprintf(stderr, "  err: %s\n", lr.error.c_str()); return; }

    const int rows = 5, cols = 64;
    std::vector<float> logits(rows*cols);
    for (int i=0;i<rows*cols;++i) logits[i] = 0.25f*i;
    std::vector<int> idx = {7, 63, 0, 100, -1};  // OOB(100) + sentinel(-1) → 0

    std::vector<CUdeviceptr> dptr(lr.dag.buffers.size(), 0);
    int outb = -1;
    for (const BufferDecl& b : lr.dag.buffers) {
        size_t bytes = (size_t)b.elem_count * (b.cls == BufferClass::IntrinsicLogits ? 2 : dtype_size(b.dtype));
        if (bytes == 0) bytes = 4;
        CU_CHECK(cuMemAlloc(&dptr[b.id], bytes));
        CU_CHECK(cuMemsetD8(dptr[b.id], 0, bytes));
        if (b.cls == BufferClass::IntrinsicLogits) {
            std::vector<__nv_bfloat16> hb(rows*cols);
            for (int i=0;i<rows*cols;++i) hb[i]=__float2bfloat16(logits[i]);
            CU_CHECK(cuMemcpyHtoD(dptr[b.id], hb.data(), (size_t)rows*cols*2));
        } else if (b.cls == BufferClass::HostSubmit) {
            CU_CHECK(cuMemcpyHtoD(dptr[b.id], idx.data(), (size_t)rows*4));
        } else if (b.cls == BufferClass::Output) outb = b.id;
    }
    for (const KernelDesc& kd : lr.dag.kernels) {
        CUmodule m = compile(kd.source);
        CUfunction fn; CU_CHECK(cuModuleGetFunction(&fn, m, kd.entry_name.c_str()));
        std::vector<void*> args;
        for (const KernelArg& a : kd.args) args.push_back(&dptr[a.buffer]);
        LaunchDims dd = compute_launch_dims(kd.shape, 1, cols, 0, kd.custom_grid, kd.custom_block);
        CU_CHECK(cuLaunchKernel(fn, dd.grid_x, 1, 1, dd.block_x, 1, 1, 0, nullptr, args.data(), nullptr));
        CU_CHECK(cuCtxSynchronize());
        CU_CHECK(cuModuleUnload(m));
    }
    std::vector<float> out(rows, -1.f);
    CU_CHECK(cuMemcpyDtoH(out.data(), dptr[outb], (size_t)rows*4));
    for (CUdeviceptr p : dptr) if (p) cuMemFree(p);

    bool ok = true;
    for (int r=0;r<rows;++r) {
        float ref = (idx[r]>=0 && idx[r]<cols) ? __bfloat162float(__float2bfloat16(logits[r*cols+idx[r]])) : 0.0f;
        if (std::fabs(out[r]-ref) >= 1e-3f) ok = false;
    }
    expect(ok, "GatherCols out[i]=src[i,idx[i]] exact, OOB/sentinel -> 0");
}

void test_batched_argmax() {
    std::printf("[codegen batched (M>1) argmax — inline bf16-cast]\n");
    LowerResult lr = lower_bytecode((const std::uint8_t*)GV_ARGMAX, sizeof(GV_ARGMAX),
                                    LowerOptions{/*batched=*/true});
    expect(lr.ok, "batched argmax lowers");
    if (!lr.ok) { std::fprintf(stderr, "  err: %s\n", lr.error.c_str()); return; }
    expect(lr.dag.kernels.size() == 1, "batched argmax = single kernel");
    expect(lr.dag.kernels[0].shape == LaunchShape::OneBlockPerRow,
           "batched argmax shape = OneBlockPerRow (dynamic grid=num_rows)");

    // Inline bf16-cast (1fe03da2): logits' only consumer (ReduceArgmax) accepts
    // inline reads, so no f32 logits buffer is materialized — the reduce reads
    // bf16 directly (one pass vs the 3-pass cast+materialize+reduce). Structural
    // proof the lever fired: no non-logits buffer is V-sized (per-row elem_count).
    const int V = 32000;
    int vsized_nonlogits = 0;
    for (const BufferDecl& b : lr.dag.buffers)
        if (b.cls != BufferClass::IntrinsicLogits && b.elem_count == (std::uint32_t)V) ++vsized_nonlogits;
    expect(vsized_nonlogits == 0, "inline bf16-cast fired: no materialized f32 logits buffer");

    const int B = 128;  // delta's §2f argmax knee — the exact production-geometry re-bench shape
    std::vector<float> logits(B*V);
    std::mt19937 rng(11);
    std::uniform_real_distribution<float> d(-8.f, 8.f);
    for (auto& v : logits) v = d(rng);
    for (int r=0;r<B;++r) logits[r*V + ((r*1337u) % V)] = 60.f;  // unique max per row (survives bf16)

    std::vector<std::uint32_t> seeds(B, 0);  // greedy argmax: no RNG / no HostSubmit
    std::vector<int> tok; std::string err;
    expect(run_batched((const std::uint8_t*)GV_ARGMAX, sizeof(GV_ARGMAX), B, V, logits, seeds, tok, err),
           "batched argmax runs");
    bool ok = true;
    for (int r=0;r<B;++r) {
        int best=0; float bv=-INFINITY;
        for (int j=0;j<V;++j){ float v=__bfloat162float(__float2bfloat16(logits[r*V+j])); if (v>bv){bv=v;best=j;} }
        if (tok[r] != best) ok = false;
    }
    expect(ok, "every batch row's argmax token matches CPU bf16 reference (inline-cast parity)");
}

void test_spec_lossless() {
    std::printf("[codegen lossless spec-verify (k=1 single-row matrix) — MC distribution lock]\n");
    LowerResult lr = lower_bytecode((const std::uint8_t*)GV_SPECLOSSLESS, sizeof(GV_SPECLOSSLESS));
    expect(lr.ok, "lossless k=1 lowers (single-row Matrix{1,V} ↔ Vector partition by shape class)");
    if (!lr.ok) { std::fprintf(stderr, "  err: %s\n", lr.error.c_str()); return; }

    const int V = 8;
    float target[V] = {2.0f,1.0f,0.5f,0.0f,-0.5f,-1.0f,-1.5f,-2.0f};
    float p[V]; { float mx=-1e30f; for(float v:target) mx=std::max(mx,v);
        float Z=0; for(int i=0;i<V;++i){p[i]=std::expf(target[i]-mx);Z+=p[i];} for(int i=0;i<V;++i)p[i]/=Z; }
    float q[V]; for(int i=0;i<V;++i) q[i]=1.0f/V;  // uniform draft — residual must correct it to p

    std::vector<CUdeviceptr> dptr(lr.dag.buffers.size(), 0);
    int b_draft=-1,b_acc=-1,b_res=-1,b_out=-1,b_q=-1,b_logits=-1;
    for (const BufferDecl& b : lr.dag.buffers) {
        std::uint32_t ec = b.elem_count ? b.elem_count : 1;
        size_t bytes = (size_t)ec * (b.cls==BufferClass::IntrinsicLogits ? 2 : dtype_size(b.dtype));
        if (bytes==0) bytes=4;
        CU_CHECK(cuMemAlloc(&dptr[b.id], bytes)); CU_CHECK(cuMemsetD8(dptr[b.id], 0, bytes));
        if (b.cls==BufferClass::IntrinsicLogits) b_logits=b.id;
        else if (b.cls==BufferClass::Output) b_out=b.id;
        else if (b.cls==BufferClass::HostSubmit) {
            switch (b.input_id) { case 1:b_q=b.id;break; case 2:b_draft=b.id;break;
                case 3:b_acc=b.id;break; case 4:b_res=b.id;break; }
        }
    }
    { std::vector<__nv_bfloat16> hb(V); for(int i=0;i<V;++i)hb[i]=__float2bfloat16(target[i]);
        CU_CHECK(cuMemcpyHtoD(dptr[b_logits], hb.data(), (size_t)V*2)); }
    CU_CHECK(cuMemcpyHtoD(dptr[b_q], q, (size_t)V*4));

    std::vector<CUfunction> fns(lr.dag.kernels.size());
    std::vector<std::vector<void*>> kargs(lr.dag.kernels.size());
    std::vector<LaunchDims> kdims(lr.dag.kernels.size());
    for (size_t k=0;k<lr.dag.kernels.size();++k) {
        const KernelDesc& kd=lr.dag.kernels[k];
        CUmodule m=compile(kd.source); CU_CHECK(cuModuleGetFunction(&fns[k],m,kd.entry_name.c_str()));
        for(const KernelArg& a:kd.args) kargs[k].push_back(&dptr[a.buffer]);
        kdims[k]=compute_launch_dims(kd.shape,1,V,0,kd.custom_grid,kd.custom_block);
    }
    auto run_once=[&](){ for(size_t k=0;k<fns.size();++k)
        CU_CHECK(cuLaunchKernel(fns[k],kdims[k].grid_x,1,1,kdims[k].block_x,1,1,0,nullptr,kargs[k].data(),nullptr));
        CU_CHECK(cuCtxSynchronize()); };
    auto set32=[&](int buf,unsigned v){ CU_CHECK(cuMemcpyHtoD(dptr[buf],&v,4)); };
    auto out_tok=[&](){ int t=-99; CU_CHECK(cuMemcpyDtoH(&t,dptr[b_out],4)); return t; };

    // forced-accept: some accept_seed yields u small enough that draft is emitted.
    bool seen_accept=false;
    set32(b_draft,4); set32(b_res,12345u);
    for(unsigned s=1;s<4000 && !seen_accept;++s){ set32(b_acc,s); run_once(); if(out_tok()==4) seen_accept=true; }
    expect(seen_accept, "forced-accept: accepted position emits the draft token");

    // MC distribution lock: draft x~q (uniform), histogram emitted token ≈ target p.
    const int Nmc=8000; std::vector<long> H(V,0); std::mt19937 rng(424242);
    std::discrete_distribution<int> draw_q(q,q+V); std::uniform_int_distribution<unsigned> ru;
    bool inrange=true;
    for(int i=0;i<Nmc;++i){ set32(b_draft,draw_q(rng)); set32(b_acc,ru(rng)|1u); set32(b_res,ru(rng)|1u);
        run_once(); int o=out_tok(); if(o>=0&&o<V) H[o]++; else { inrange=false; break; } }
    expect(inrange, "every emitted token in [0,V)");
    double maxdev=0; for(int t=0;t<V;++t) maxdev=std::max(maxdev,std::fabs((double)H[t]/Nmc - p[t]));
    std::printf("  MC N=%d max|H/N - p| = %.4f (draft q=uniform, target p=softmax)\n", Nmc, maxdev);
    expect(maxdev < 0.03, "emitted-token histogram matches TARGET p (lossless guarantee P(out=t)=p(t))");

    for(CUdeviceptr pp:dptr) if(pp) cuMemFree(pp);
}

void test_std_argmax_v4() {
    std::printf("[codegen v4 binding-free argmax — baked pie_standard_samplers.h + lower_bytecode_v4]\n");
    StandardSamplerProgram sp = standard_sampler_program(StandardSamplerKind::Argmax, 151936);
    expect(sp.valid, "baked Argmax(151936) program present");
    if (!sp.valid) return;
    // v4 bytecode is binding-free; the manifest supplies slot[0]=Logits.
    LowerResult lr = lower_bytecode_v4(sp.bytecode, sp.len, sp.manifest, LowerOptions{});
    expect(lr.ok, "v4 argmax lowers via manifest (binding-free shim)");
    if (!lr.ok) { std::fprintf(stderr, "  err: %s\n", lr.error.c_str()); return; }
    expect(lr.dag.kernels.size() == 1, "v4 argmax = 1 kernel");

    const int V = 151936;
    std::vector<float> logits(V);
    std::mt19937 rng(909);
    std::uniform_real_distribution<float> d(-8.f, 8.f);
    for (auto& v : logits) v = d(rng);
    logits[98765] = 80.f;  // unique max
    RunResult r = run_dag(lr.dag, logits, 0);
    int best = 0; float bv = -INFINITY;
    for (int i = 0; i < V; ++i) { float v = __bfloat162float(__float2bfloat16(logits[i])); if (v > bv) { bv = v; best = i; } }
    expect(r.token == best, "v4 baked argmax token matches CPU bf16 reference");
}

void test_std_temp_minp_v4() {
    std::printf("[codegen v4 RowSeed emit-swap — temp/min-p token-exact vs sample_temp (col=j, stream:0)]\n");
    const int B = 64, V = 32;
    std::vector<float> logits(B * V), T(B), MP(B);
    std::vector<std::uint32_t> seeds(B);
    std::mt19937 rng(321);
    std::uniform_real_distribution<float> d(-5.f, 5.f);
    for (auto& v : logits) v = d(rng);
    for (int r = 0; r < B; ++r) { T[r] = 0.5f + 0.05f * r; MP[r] = 0.05f + 0.002f * r; seeds[r] = 0x1234u + r * 2654435761u; }

    // Runs a v4 batched program binding logits[N,V] + per-row params + ambient RowSeed[N].
    auto run = [&](const unsigned char* bc, size_t n, const ProgramManifest& m,
                   std::vector<int>& tok) -> bool {
        LowerResult lr = lower_bytecode_v4(bc, n, m, LowerOptions{/*batched=*/true});
        if (!lr.ok) { std::fprintf(stderr, "  lower err: %s\n", lr.error.c_str()); return false; }
        bool has_rowseed = false;
        for (const BufferDecl& b : lr.dag.buffers) if (b.cls == BufferClass::RowSeed) has_rowseed = true;
        expect(has_rowseed, "v4 RNG program emits a RowSeed buffer");
        std::vector<CUdeviceptr> dptr(lr.dag.buffers.size(), 0);
        BufferId outb = 0;
        for (const BufferDecl& b : lr.dag.buffers) {
            std::uint32_t total = (b.batched ? (std::uint32_t)B : 1u) * (b.elem_count ? b.elem_count : 1u);
            size_t bytes = (size_t)total * (b.cls == BufferClass::IntrinsicLogits ? 2 : dtype_size(b.dtype));
            if (bytes == 0) bytes = 4;
            CU_CHECK(cuMemAlloc(&dptr[b.id], bytes)); CU_CHECK(cuMemsetD8(dptr[b.id], 0, bytes));
            if (b.cls == BufferClass::IntrinsicLogits) {
                std::vector<__nv_bfloat16> hb(B * V); for (int i = 0; i < B * V; ++i) hb[i] = __float2bfloat16(logits[i]);
                CU_CHECK(cuMemcpyHtoD(dptr[b.id], hb.data(), (size_t)B * V * 2));
            } else if (b.cls == BufferClass::HostSubmit) {
                CU_CHECK(cuMemcpyHtoD(dptr[b.id], b.input_id == 2 ? MP.data() : T.data(), (size_t)B * 4));
            } else if (b.cls == BufferClass::RowSeed) {
                CU_CHECK(cuMemcpyHtoD(dptr[b.id], seeds.data(), (size_t)B * 4));
            } else if (b.cls == BufferClass::Output) outb = b.id;
        }
        for (const KernelDesc& kd : lr.dag.kernels) {
            CUmodule mo = compile(kd.source); CUfunction fn; CU_CHECK(cuModuleGetFunction(&fn, mo, kd.entry_name.c_str()));
            std::vector<void*> args; for (const KernelArg& a : kd.args) args.push_back(&dptr[a.buffer]);
            LaunchDims dd = compute_launch_dims(kd.shape, B, V, 0, kd.custom_grid, kd.custom_block);
            CU_CHECK(cuLaunchKernel(fn, dd.grid_x, 1, 1, dd.block_x, 1, 1, 0, nullptr, args.data(), nullptr));
            CU_CHECK(cuCtxSynchronize()); CU_CHECK(cuModuleUnload(mo));
        }
        tok.assign(B, -99); CU_CHECK(cuMemcpyDtoH(tok.data(), dptr[outb], (size_t)B * 4));
        for (CUdeviceptr p : dptr) if (p) cuMemFree(p);
        return true;
    };

    const ProgramManifest temp_m = {{BindKind::Logits,0,HostAvailability::SubmitBound},
                                    {BindKind::HostTensor,0,HostAvailability::SubmitBound}};
    const ProgramManifest minp_m = {{BindKind::Logits,0,HostAvailability::SubmitBound},
                                    {BindKind::HostTensor,0,HostAvailability::SubmitBound},
                                    {BindKind::HostTensor,1,HostAvailability::SubmitBound}};
    std::vector<int> tok;
    if (run(GV_STD_TEMP32, sizeof(GV_STD_TEMP32), temp_m, tok)) {
        bool ok = true;
        for (int r = 0; r < B; ++r) {
            int best = 0; float bv = -INFINITY;
            for (int j = 0; j < V; ++j) { float lf = __bfloat162float(__float2bfloat16(logits[r*V+j]));
                float sc = lf / T[r] + gumbel_ref(seeds[r], j); if (sc > bv) { bv = sc; best = j; } }
            if (tok[r] != best) ok = false;
        }
        expect(ok, "v4 temp every row token-exact vs sample_temp (ambient seed[r], stream:0)");
    }
    if (run(GV_STD_MINP32, sizeof(GV_STD_MINP32), minp_m, tok)) {
        bool ok = true;
        for (int r = 0; r < B; ++r) {
            std::vector<float> lf(V); float mx = -INFINITY;
            for (int j = 0; j < V; ++j) { lf[j] = __bfloat162float(__float2bfloat16(logits[r*V+j])); mx = std::max(mx, lf[j]); }
            float thr = mx + std::log(MP[r]); int best = 0; float bv = -INFINITY;
            for (int j = 0; j < V; ++j) { float sc = (lf[j] >= thr) ? (lf[j]/T[r] + gumbel_ref(seeds[r], j)) : -INFINITY; if (sc > bv) { bv = sc; best = j; } }
            if (tok[r] != best) ok = false;
        }
        expect(ok, "v4 min-p every row token-exact vs sample_temp logit-space (ambient seed[r], stream:0)");
    }
}

}  // namespace

int main() {
    CU_CHECK(cuInit(0));
    CUdevice dev; CU_CHECK(cuDeviceGet(&dev, 0));
    CU_CHECK(cuDeviceGetAttribute(&g_cc_major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev));
    CU_CHECK(cuDeviceGetAttribute(&g_cc_minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev));
    CUcontext ctx; CU_CHECK(cuDevicePrimaryCtxRetain(&ctx, dev));
    CU_CHECK(cuCtxSetCurrent(ctx));
    char nm[256]; CU_CHECK(cuDeviceGetName(nm, sizeof(nm), dev));
    std::printf("Device: %s (sm_%d%d)\n", nm, g_cc_major, g_cc_minor);

    test_argmax();
    test_temp_parity();
    test_minp_parity();
    test_topk_parity();
    test_topp_parity();
    test_barrier();
    test_matrix_argmax();
    test_spec_greedy_codegen();
    test_rowbcast_codegen();
    test_late_barrier_stamp();
    test_batched_temp();
    test_batched_minp();
    test_batched_topk();
    test_batched_argmax();
    test_gathercols_codegen();
    test_spec_lossless();
    test_std_argmax_v4();
    test_std_temp_minp_v4();

    CU_CHECK(cuDevicePrimaryCtxRelease(dev));
    std::printf("\n%d checks, %d failures\n", g_checks, g_fail);
    if (g_fail == 0) std::printf("ALL PASS\n");
    return g_fail == 0 ? 0 : 1;
}
