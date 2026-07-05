// End-to-end de-risk for the sampling-IR NVRTC JIT + driver-API launch path.
//
// Proves the whole toolchain on the live GPU with a two-kernel DAG that mirrors
// the spec's motivating cross-kernel value: `j = reduce-sum(accept)` (kernel A)
// is written to a device buffer, then `gather-row(resid, j)` (kernel B) reads
// `j` back from that fixed device pointer to select a row. Exercises:
//   * NVRTC source -> PTX for the live arch (sm_89 query)
//   * cuModuleLoadData -> cuModuleGetFunction (driver API)
//   * cuLaunchKernel of a 2-node DAG (driver API)
//   * cross-kernel value `j` flowing kernel->kernel through device memory
//   * program cache: compile once, fetch on hash hit
//
// Self-contained: pure driver API + NVRTC (no nvcc, no cudart). Failures abort
// with a message + non-zero exit so CTest catches them.

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include <cuda.h>

#include "sampling_ir/jit.hpp"

using namespace pie_cuda_driver::sampling_ir::jit;

namespace {

int g_failures = 0;

#define CHECK(cond)                                                       \
    do {                                                                  \
        if (!(cond)) {                                                    \
            std::fprintf(stderr, "FAIL: %s:%d: %s\n", __FILE__, __LINE__, \
                         #cond);                                          \
            ++g_failures;                                                 \
        }                                                                 \
    } while (0)

void cu_check(CUresult res, const char* expr, int line) {
    if (res != CUDA_SUCCESS) {
        const char* name = nullptr;
        cuGetErrorName(res, &name);
        std::fprintf(stderr, "FATAL: %s:%d: %s -> %s\n", __FILE__, line, expr,
                     name ? name : "?");
        std::exit(2);
    }
}
#define CU(expr) cu_check((expr), #expr, __LINE__)

// Kernel A: j = sum(accept[0..n]). Single power-of-two block reduction. `n` is
// a per-fire Param (changes each fire with no recompile).
const char* kReduceSumSrc = R"CUDA(
extern "C" __global__ void reduce_sum_accept(const int* accept, int* j, unsigned n) {
    __shared__ int s[256];
    unsigned t = threadIdx.x;
    int local = 0;
    for (unsigned i = t; i < n; i += blockDim.x) local += accept[i];
    s[t] = local;
    __syncthreads();
    for (unsigned stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (t < stride) s[t] += s[t + stride];
        __syncthreads();
    }
    if (t == 0) j[0] = s[0];
}
)CUDA";

// Kernel B: out[k] = resid[j * C + k]. Reads `j` written by kernel A.
const char* kGatherRowSrc = R"CUDA(
extern "C" __global__ void gather_row(const float* resid, const int* j, float* out, unsigned c) {
    unsigned k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k < c) {
        unsigned row = (unsigned)j[0];
        out[k] = resid[row * c + k];
    }
}
)CUDA";

constexpr unsigned kN = 10;   // accept length
constexpr unsigned kR = 8;    // resid rows
constexpr unsigned kC = 4;    // resid cols

enum Buf : BufferId { kAccept = 0, kResid = 1, kJ = 2, kOut = 3 };
enum Param : std::uint32_t { kParamN = 0 };  // per-fire: accept length to reduce

KernelDAG build_dag() {
    KernelDAG dag;
    dag.buffers = {
        {kAccept, kN * sizeof(int), ScalarType::I32, /*external=*/false},
        {kResid, kR * kC * sizeof(float), ScalarType::F32, false},
        {kJ, sizeof(int), ScalarType::I32, false},
        {kOut, kC * sizeof(float), ScalarType::F32, false},
    };

    KernelDef a;
    a.name = "reduce_sum_accept";
    a.source = kReduceSumSrc;
    a.grid = {1, 1, 1};
    a.block = {256, 1, 1};
    a.args = {KernelArg::buffer_arg(kAccept), KernelArg::buffer_arg(kJ),
              KernelArg::param_arg(kParamN, ScalarType::U32)};

    KernelDef b;
    b.name = "gather_row";
    b.source = kGatherRowSrc;
    b.grid = {(kC + 255) / 256, 1, 1};
    b.block = {256, 1, 1};
    b.args = {KernelArg::buffer_arg(kResid), KernelArg::buffer_arg(kJ),
              KernelArg::buffer_arg(kOut), KernelArg::u32(kC)};

    dag.kernels = {a, b};

    // Program-cache key: hash the kernel sources (stand-in for the bytecode
    // hash the bridge/codegen will supply in production).
    std::string blob = std::string(a.name) + a.source + b.name + b.source;
    dag.hash = fnv1a64(blob.data(), blob.size());
    return dag;
}

}  // namespace

// ── Batched (M>1) JIT path: dynamic grid=num_rows + capacity-grown intermediate.
// Mirrors charlie's batched codegen contract (grid=num_rows, blockIdx.x=row,
// `if (r>=num_rows) return;`, every value gains a per-row batch dim). Proves the
// JIT infra independently of codegen: GridShape::OneBlockPerRow recomputed per
// fire, a batched Intermediate grown across fires, batched external IO.
namespace batched {

// Kernel A: per row r, scratch[r] = sum(in[r*L .. r*L+L]).
const char* kRowSumSrc = R"CUDA(
extern "C" __global__ void row_sum(const int* in, int* scratch, unsigned num_rows, unsigned L) {
    unsigned r = blockIdx.x;
    if (r >= num_rows) return;
    int s = 0;
    for (unsigned k = 0; k < L; ++k) s += in[r * L + k];
    scratch[r] = s;
}
)CUDA";

// Kernel B: per row r, out[r] = scratch[r] + r (proves cross-kernel batched
// scratch flows row-wise, plus uses the row index).
const char* kRowOutSrc = R"CUDA(
extern "C" __global__ void row_out(const int* scratch, int* out, unsigned num_rows) {
    unsigned r = blockIdx.x;
    if (r >= num_rows) return;
    out[r] = scratch[r] + (int)r;
}
)CUDA";

constexpr unsigned kL = 4;  // per-row input length
enum BBuf : BufferId { kIn = 0, kScratch = 1, kOutB = 2 };
enum BParam : std::uint32_t { kRows = 0 };  // Param 0 = num_rows (jit convention)

KernelDAG build() {
    KernelDAG dag;
    dag.buffers = {
        {kIn, kL * sizeof(int), ScalarType::I32, /*external=*/true, /*batched=*/true},
        {kScratch, sizeof(int), ScalarType::I32, /*external=*/false, /*batched=*/true},
        {kOutB, sizeof(int), ScalarType::I32, /*external=*/true, /*batched=*/true},
    };
    KernelDef a;
    a.name = "row_sum";
    a.source = kRowSumSrc;
    a.block = {1, 1, 1};
    a.grid_shape = GridShape::OneBlockPerRow;  // grid.x = num_rows per fire
    a.args = {KernelArg::buffer_arg(kIn), KernelArg::buffer_arg(kScratch),
              KernelArg::param_arg(kRows, ScalarType::U32), KernelArg::u32(kL)};
    KernelDef b;
    b.name = "row_out";
    b.source = kRowOutSrc;
    b.block = {1, 1, 1};
    b.grid_shape = GridShape::OneBlockPerRow;
    b.args = {KernelArg::buffer_arg(kScratch), KernelArg::buffer_arg(kOutB),
              KernelArg::param_arg(kRows, ScalarType::U32)};
    dag.kernels = {a, b};
    std::string blob = std::string(a.source) + b.source + "batched";
    dag.hash = fnv1a64(blob.data(), blob.size());
    return dag;
}

// Run num_rows rows through the cached program; verify out[r] = sum(in row r)+r.
void run(JitEngine& engine, CompiledProgram& prog, unsigned num_rows) {
    std::vector<int> in(num_rows * kL);
    for (unsigned r = 0; r < num_rows; ++r)
        for (unsigned k = 0; k < kL; ++k) in[r * kL + k] = static_cast<int>(r * 10 + k);

    CUdeviceptr d_in = 0, d_out = 0;
    CU(cuMemAlloc(&d_in, in.size() * sizeof(int)));
    CU(cuMemAlloc(&d_out, num_rows * sizeof(int)));
    CU(cuMemcpyHtoD(d_in, in.data(), in.size() * sizeof(int)));
    engine.bind_buffer(prog, kIn, d_in);
    engine.bind_buffer(prog, kOutB, d_out);

    engine.launch(prog, /*stream=*/0, /*param_values=*/{num_rows, 0, 0});
    CU(cuCtxSynchronize());

    std::vector<int> out(num_rows, -1);
    CU(cuMemcpyDtoH(out.data(), d_out, num_rows * sizeof(int)));
    for (unsigned r = 0; r < num_rows; ++r) {
        int want = 0;
        for (unsigned k = 0; k < kL; ++k) want += in[r * kL + k];
        want += static_cast<int>(r);
        if (out[r] != want)
            std::fprintf(stderr, "FAIL batched: rows=%u out[%u]=%d want=%d\n",
                         num_rows, r, out[r], want);
        CHECK(out[r] == want);
    }
    std::fprintf(stderr, "batched rows=%u: OK\n", num_rows);
    CU(cuMemFree(d_in));
    CU(cuMemFree(d_out));
}

void test(JitEngine& engine) {
    KernelDAG dag = build();
    CompiledProgram& prog = engine.get_or_compile(dag);
    run(engine, prog, 3);    // initial capacity grows 1 -> 3
    run(engine, prog, 10);   // capacity grows 3 -> 10 (realloc batched scratch)
    run(engine, prog, 2);    // smaller fire reuses capacity, no realloc
}

}  // namespace batched

// ── Multi-block-per-row (GridStrideOverVocab): the perf lever for B>=128.
// Proves the JIT's dynamic GridStrideOverVocab grid = ceil(num_rows*vocab/block)
// recomputed per fire drives a reduction where MANY blocks cooperate per row
// (atomicAdd over the flattened [rows, vocab] space) — the launch shape
// charlie's multi-block-per-row reductions will use to close the large-batch
// gap. Verifies the grid count + that batched external IO works under it.
namespace gridstride {

// out[row] = sum over col of in[row*vocab + col]. Grid-stride over rows*vocab;
// every thread atomicAdds into its row's accumulator → blocks_per_row > 1.
const char* kRowReduceSrc = R"CUDA(
extern "C" __global__ void row_reduce_atomic(const int* in, int* out,
                                             unsigned num_rows, unsigned vocab) {
    unsigned total = num_rows * vocab;
    for (unsigned idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total;
         idx += gridDim.x * blockDim.x) {
        unsigned row = idx / vocab;
        atomicAdd(&out[row], in[idx]);
    }
}
)CUDA";

enum GBuf : BufferId { kGIn = 0, kGOut = 1 };
enum GParam : std::uint32_t { kGRows = 0, kGVocab = 1 };
constexpr unsigned kVocab = 300;  // > block (256) so blocks_per_row > 1

KernelDAG build() {
    KernelDAG dag;
    dag.buffers = {
        {kGIn, kVocab * sizeof(int), ScalarType::I32, /*external=*/true, /*batched=*/true},
        {kGOut, sizeof(int), ScalarType::I32, /*external=*/true, /*batched=*/true},
    };
    KernelDef k;
    k.name = "row_reduce_atomic";
    k.source = kRowReduceSrc;
    k.block = {256, 1, 1};
    k.grid_shape = GridShape::GridStrideOverVocab;  // grid.x=ceil(rows*vocab/256)/fire
    k.per_row_len = kVocab;
    k.args = {KernelArg::buffer_arg(kGIn), KernelArg::buffer_arg(kGOut),
              KernelArg::param_arg(kGRows, ScalarType::U32),
              KernelArg::param_arg(kGVocab, ScalarType::U32)};
    dag.kernels = {k};
    std::string blob = std::string(k.source) + "gridstride";
    dag.hash = fnv1a64(blob.data(), blob.size());
    return dag;
}

void run(JitEngine& engine, CompiledProgram& prog, unsigned num_rows) {
    std::vector<int> in(num_rows * kVocab);
    for (unsigned r = 0; r < num_rows; ++r)
        for (unsigned c = 0; c < kVocab; ++c) in[r * kVocab + c] = static_cast<int>(r + 1);

    CUdeviceptr d_in = 0, d_out = 0;
    CU(cuMemAlloc(&d_in, in.size() * sizeof(int)));
    CU(cuMemAlloc(&d_out, num_rows * sizeof(int)));
    CU(cuMemcpyHtoD(d_in, in.data(), in.size() * sizeof(int)));
    CU(cuMemsetD8(d_out, 0, num_rows * sizeof(int)));  // accumulator init
    engine.bind_buffer(prog, kGIn, d_in);
    engine.bind_buffer(prog, kGOut, d_out);

    engine.launch(prog, /*stream=*/0, /*param_values=*/{num_rows, kVocab, 0});
    CU(cuCtxSynchronize());

    std::vector<int> out(num_rows, -1);
    CU(cuMemcpyDtoH(out.data(), d_out, num_rows * sizeof(int)));
    for (unsigned r = 0; r < num_rows; ++r) {
        const int want = static_cast<int>((r + 1) * kVocab);  // vocab copies of (r+1)
        if (out[r] != want)
            std::fprintf(stderr, "FAIL gridstride: rows=%u out[%u]=%d want=%d\n",
                         num_rows, r, out[r], want);
        CHECK(out[r] == want);
    }
    std::fprintf(stderr, "gridstride rows=%u (blocks_per_row=%u): OK\n", num_rows,
                 (kVocab + 255) / 256);
    CU(cuMemFree(d_in));
    CU(cuMemFree(d_out));
}

void test(JitEngine& engine) {
    KernelDAG dag = build();
    CompiledProgram& prog = engine.get_or_compile(dag);
    run(engine, prog, 4);    // grid = ceil(4*300/256) = 5 blocks over 4 rows
    run(engine, prog, 64);   // grid = ceil(64*300/256) = 75 blocks, many per row
}

}  // namespace gridstride


int main() {
    CU(cuInit(0));
    CUdevice dev = 0;
    CU(cuDeviceGet(&dev, 0));
    CUcontext ctx = nullptr;
    CU(cuCtxCreate(&ctx, nullptr, 0, dev));

    // Scope the engine + compiled program so their CUDA resources (modules,
    // device buffers, graph) are released while the context is still alive.
    {
    JitEngine engine;
    std::fprintf(stderr, "JIT target arch: %s\n", engine.arch().c_str());
    CHECK(engine.arch() == "compute_89");  // RTX 4090

    KernelDAG dag = build_dag();
    CompiledProgram& prog = engine.get_or_compile(dag);

    // Host inputs. accept has six 1s -> j == 6. resid[row][k] = row*100 + k.
    std::vector<int> accept = {1, 0, 1, 1, 0, 1, 0, 0, 1, 1};
    std::vector<float> resid(kR * kC);
    for (unsigned r = 0; r < kR; ++r)
        for (unsigned c = 0; c < kC; ++c) resid[r * kC + c] = r * 100.0f + c;

    CU(cuMemcpyHtoD(prog.device_ptr(kAccept), accept.data(),
                    accept.size() * sizeof(int)));
    CU(cuMemcpyHtoD(prog.device_ptr(kResid), resid.data(),
                    resid.size() * sizeof(float)));

    auto fire_and_check = [&](unsigned n, int expect_j) {
        engine.launch(prog, /*stream=*/0, /*param_values=*/{n});
        CU(cuCtxSynchronize());
        int j = -1;
        std::vector<float> out(kC, -1.0f);
        CU(cuMemcpyDtoH(&j, prog.device_ptr(kJ), sizeof(int)));
        CU(cuMemcpyDtoH(out.data(), prog.device_ptr(kOut), kC * sizeof(float)));
        std::fprintf(stderr, "n=%u: j = %d (expect %d)\n", n, j, expect_j);
        CHECK(j == expect_j);
        for (unsigned c = 0; c < kC; ++c) {
            float want = expect_j * 100.0f + c;
            std::fprintf(stderr, "  out[%u] = %.1f (expect %.1f)\n", c, out[c], want);
            CHECK(out[c] == want);
        }
    };

    // Fire 1: reduce all 10 -> six 1s -> j == 6.
    fire_and_check(kN, /*expect_j=*/6);
    // Fire 2: same compiled program, new per-fire Param n=5 (no recompile) ->
    // sum(accept[0..5]) = 1+0+1+1+0 = 3 -> j == 3. Proves submit-bound scalars.
    fire_and_check(5, /*expect_j=*/3);

    // Phase-2: capture the DAG into a CUDA graph and replay with one launch.
    {
        CUstream cap_stream = nullptr;
        CU(cuStreamCreate(&cap_stream, CU_STREAM_NON_BLOCKING));
        engine.instantiate_graph(prog, cap_stream, /*param_values=*/{kN});
        // Scribble the output so the replay has to recompute it.
        std::vector<float> poison(kC, -42.0f);
        CU(cuMemcpyHtoD(prog.device_ptr(kOut), poison.data(), kC * sizeof(float)));
        engine.launch_graph(prog, cap_stream);
        CU(cuStreamSynchronize(cap_stream));
        int gj = -1;
        std::vector<float> gout(kC, -1.0f);
        CU(cuMemcpyDtoH(&gj, prog.device_ptr(kJ), sizeof(int)));
        CU(cuMemcpyDtoH(gout.data(), prog.device_ptr(kOut), kC * sizeof(float)));
        std::fprintf(stderr, "graph replay: j = %d (expect 6)\n", gj);
        CHECK(gj == 6);
        for (unsigned c = 0; c < kC; ++c) {
            CHECK(gout[c] == 6 * 100.0f + c);
        }
        CU(cuStreamDestroy(cap_stream));
    }

    // Program cache: same hash returns the same compiled instance (no recompile).
    CompiledProgram& prog2 = engine.get_or_compile(dag);
    CHECK(&prog2 == &prog);

    // Batched (M>1) path: dynamic grid=num_rows + capacity-grown intermediate.
    batched::test(engine);
    // Multi-block-per-row (GridStrideOverVocab): the B>=128 perf lever's grid.
    gridstride::test(engine);
    }  // engine + program destroyed here, before the context is torn down

    CU(cuCtxDestroy(ctx));

    if (g_failures == 0) {
        std::fprintf(stderr, "sampling_ir_jit: OK\n");
        return 0;
    }
    std::fprintf(stderr, "sampling_ir_jit: %d failure(s)\n", g_failures);
    return 1;
}
