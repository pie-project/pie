#include "ops/gemm.hpp"

#include <cublasLt.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "cuda_check.hpp"
#include "kernels/dequant_fp4.hpp"
#include "kernels/dequant_fp8.hpp"
#include "kernels/add_bias.hpp"
#include "kernels/argmax.hpp"
#include "kernels/gemv.hpp"
#include "kernels/quant_bf16_to_fp8.hpp"
#include "kernels/residual_add.hpp"
#include "ops/tuning_cache.hpp"

#ifdef PIE_CUDA_HAS_MARLIN
#include "marlin_wrapper.hpp"
#endif

namespace pie_cuda_driver::ops {

namespace {

constexpr std::size_t kDefaultLtWorkspaceBytes = 32ull * 1024ull * 1024ull;

struct LtCtx;
thread_local LtCtx* g_runtime_quant_context = nullptr;

// CUBLAS_COMPUTE_32F_FAST_16BF exists to let a matmul over *fp32* operands
// round them to bf16 for the tensor cores. Operands that are already bf16 gain
// nothing from it, and cuBLASLt has no algorithm at all for many bf16 shapes
// under it -- the MLA absorb batches and the MoE expert batch among them. Its
// heuristic query then fails on every call, and cuBLAS silently retries the
// matmul in CUBLAS_COMPUTE_32F. That internal retry is not reliable when eight
// rank threads take it at the same instant: when it loses the race the call
// returns NOT_SUPPORTED or INTERNAL_ERROR, and if it happened inside a graph
// capture the failure also invalidates the capture, so the next GEMM dies far
// from the cause. That is what killed roughly one boot in ten at tp > 1.
// CUBLAS_COMPUTE_32F is what bf16 operands should have been asking for all
// along: same tensor cores, same fp32 accumulate, no fallback to race.
cublasComputeType_t bf16_compute_type() { return CUBLAS_COMPUTE_32F; }

std::size_t checked_mul(std::size_t a, std::size_t b, const char* what) {
    if (a != 0 && b > std::numeric_limits<std::size_t>::max() / a) {
        throw std::runtime_error(
            std::string("runtime quant scratch byte overflow: ") + what);
    }
    return a * b;
}

std::size_t checked_add(std::size_t a, std::size_t b, const char* what) {
    if (b > std::numeric_limits<std::size_t>::max() - a) {
        throw std::runtime_error(
            std::string("runtime quant scratch byte overflow: ") + what);
    }
    return a + b;
}

std::size_t runtime_quant_dequant_bytes(
    const RuntimeQuantScratchSpec& spec)
{
    if (spec.empty()) return 0;
    const std::size_t weight_elems =
        spec.max_dequant_weight_elems > 0
            ? spec.max_dequant_weight_elems
            : checked_mul(spec.max_weight_rows, spec.max_weight_cols,
                          "dequant weight elems");
    const std::size_t weight_bf16_bytes =
        checked_mul(
            weight_elems,
            std::size_t{2},
            "dequant weight bytes");
    const std::size_t residual_bf16_bytes =
        spec.has_int8
            ? checked_mul(
                  checked_mul(spec.max_tokens, spec.max_weight_rows,
                              "dequant output elems"),
                  std::size_t{2},
                  "dequant output bytes")
            : 0;
    return std::max(weight_bf16_bytes, residual_bf16_bytes);
}

void check(cublasStatus_t s, const char* expr) {
    if (s != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error(std::string("cuBLAS error (") +
                                 std::to_string(static_cast<int>(s)) + "): " + expr);
    }
}

std::size_t cublaslt_bf16_workspace_bytes() { return 64ull * 1024ull * 1024ull; }

// Lazily-created singleton keyed by the CALLING THREAD'S CURRENT DEVICE.
//
// Tensor parallelism runs every rank inside this one process, each bound to
// its own device. A plain process-global would therefore hand rank 1 state
// that belongs to rank 0's device: cuBLASLt would run both ranks' matmuls
// against a single scratch buffer (a live data race), and any algorithm that
// zeroes its workspace bakes a memset of that foreign pointer into the
// captured decode graph, which makes `cudaGraphInstantiate` reject the graph
// on every rank but rank 0.
template <typename T>
T& per_device_singleton() {
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    thread_local int cached_device = -1;
    thread_local T* cached = nullptr;
    if (cached != nullptr && cached_device == device) return *cached;

    static std::mutex mu;
    static std::unordered_map<int, std::unique_ptr<T>> by_device;
    std::lock_guard<std::mutex> lock(mu);
    auto& slot = by_device[device];
    if (!slot) slot = std::make_unique<T>();
    cached = slot.get();
    cached_device = device;
    return *cached;
}

struct Bf16LtCtx {
    cublasLtHandle_t handle = nullptr;
    void* workspace = nullptr;
    std::size_t workspace_bytes = cublaslt_bf16_workspace_bytes();

    static Bf16LtCtx& instance() {
        return per_device_singleton<Bf16LtCtx>();
    }

    void ensure() {
        if (!handle) check(cublasLtCreate(&handle), "cublasLtCreate");
        if (!workspace) {
            CUDA_CHECK(cudaMalloc(&workspace, workspace_bytes));
        }
    }
};

struct Bf16LtKey {
    int M = 0;
    int N = 0;
    int K = 0;

    bool operator==(const Bf16LtKey& other) const noexcept {
        return M == other.M && N == other.N && K == other.K;
    }
};

struct Bf16LtKeyHash {
    std::size_t operator()(const Bf16LtKey& key) const noexcept {
        std::size_t h = static_cast<std::size_t>(key.M);
        h = h * 1315423911u + static_cast<std::size_t>(key.N);
        h = h * 1315423911u + static_cast<std::size_t>(key.K);
        return h;
    }
};

struct Bf16LtPlan {
    cublasLtMatmulDesc_t op_desc = nullptr;
    cublasLtMatrixLayout_t a_desc = nullptr;
    cublasLtMatrixLayout_t b_desc = nullptr;
    cublasLtMatrixLayout_t c_desc = nullptr;
    cublasLtMatmulAlgo_t algo{};
    // Every algorithm the heuristic offered for this shape, kept so the
    // autotuner can time them against each other instead of trusting the
    // order they came back in.
    std::vector<cublasLtMatmulHeuristicResult_t> heuristics;

    ~Bf16LtPlan() {
        if (c_desc) cublasLtMatrixLayoutDestroy(c_desc);
        if (b_desc) cublasLtMatrixLayoutDestroy(b_desc);
        if (a_desc) cublasLtMatrixLayoutDestroy(a_desc);
        if (op_desc) cublasLtMatmulDescDestroy(op_desc);
    }

    Bf16LtPlan() = default;
    Bf16LtPlan(const Bf16LtPlan&) = delete;
    Bf16LtPlan& operator=(const Bf16LtPlan&) = delete;
};

struct Bf16LtPlanCache {
    std::mutex mu;
    std::unordered_map<Bf16LtKey, std::shared_ptr<Bf16LtPlan>, Bf16LtKeyHash>
        plans;

    // Per device: a cached `cublasLtMatmulAlgo_t` is selected by heuristics
    // for one handle on one device and must not be replayed on another.
    static Bf16LtPlanCache& instance() {
        return per_device_singleton<Bf16LtPlanCache>();
    }
};

// The handle's stream, or `false` if cuBLAS will not say. Never guess:
// falling back to the null stream would run the GEMV outside the
// caller's ordering and race whatever produced its input.
bool cublas_stream(cublasHandle_t handle, cudaStream_t& stream) {
    return cublasGetStream(handle, &stream) == CUBLAS_STATUS_SUCCESS;
}

int cublaslt_bf16_algo_index_for_shape(int N, int K) {
    // Qwen3-0.6B's lm_head shape (K=1024, very wide N) consistently
    // prefers the third returned Lt heuristic. Larger hidden sizes regress
    // on that choice, so keep the old default for them.
    if (K < 2048 && N >= 12288) return 2;
    // Qwen3.6-35B-A3B's MTP/lm_head shape (K=2048, very wide vocab)
    // is a small but repeatable win on the second returned heuristic.
    if (K == 2048 && N >= 200000) return 1;
    // Qwen3.6-27B's H=5120 projections and lm_head consistently prefer the
    // first returned heuristic. `cublaslt_bf16_min_n` already keeps smaller
    // GEMMs on the regular cuBLAS path.
    if (K == 5120) return 0;
    // Qwen3.6-35B-A3B's hidden-size projections (for example GDN qkv and
    // full-attention q/gate, N≈8k) are faster on the first heuristic. The
    // old generic index 5 regresses the MTP verifier by several percent.
    if (K == 2048 && N >= 6144) return 0;
    // Gemma4 E4B's target lm_head (K=2560, very wide vocab) is slightly
    // faster with the first returned Lt heuristic; keep this narrow so the
    // MTP assistant scorer (K=256) and other projection GEMMs stay unchanged.
    if (K == 2560 && N >= 100000) return 0;
    return 5;
}

int cublaslt_bf16_min_n(int K) {
    // Small hidden-size models (H=1024) only benefited from cuBLASLt on the
    // very wide lm_head; routing their 2k/6k projection GEMMs through Lt was
    // consistently slower. H=2048 keeps the previous threshold because the
    // 1.7B-class models still prefer Lt for their 6k-wide MLP projection.
    //
    // For large hidden sizes, the current Lt heuristic can select kernels
    // that fault on compact multi-row lm_head shapes such as Kimi TP greedy
    // prefill (M small, N ~= 20k, K ~= 7k). The classic cuBLAS path is stable
    // for those shapes and is already used for M=1 decode, so keep Lt out of
    // the large-H wide-output path by default.
    if (K >= 4096) return 32768;
    return K < 2048 ? 12288 : (K == 2048 ? 6144 : 12288);
}

constexpr int kCublasLtBf16MinK = 1024;
constexpr int kCublasLtBf16MinM = 2;
constexpr int kCublasLtBf16MaxN = 0;

// `workspace` defaults to the context's shared scratch. It is overridable
// because the autotuner runs matmuls on a stream of its own, concurrently with
// whatever the caller's stream has in flight, and two matmuls scribbling on one
// scratch buffer silently corrupt each other's results.
bool run_bf16_lt_algo(
    Bf16LtCtx& ctx,
    const Bf16LtPlan& plan,
    const cublasLtMatmulAlgo_t* algo,
    cudaStream_t stream,
    const void* act,
    const void* W,
    void* y,
    float beta,
    void* workspace = nullptr,
    std::size_t workspace_bytes = 0)
{
    const float alpha = 1.f;
    if (workspace == nullptr) {
        workspace = ctx.workspace;
        workspace_bytes = ctx.workspace_bytes;
    }
    const cublasStatus_t st = cublasLtMatmul(
        ctx.handle, plan.op_desc,
        &alpha,
        W, plan.a_desc,
        act, plan.b_desc,
        &beta,
        y, plan.c_desc,
        y, plan.c_desc,
        algo,
        workspace, workspace_bytes, stream);
    return st == CUBLAS_STATUS_SUCCESS;
}

bool run_bf16_lt_plan(
    Bf16LtCtx& ctx,
    const Bf16LtPlan& plan,
    const cublasLtMatmulAlgo_t& algo,
    cublasHandle_t cublas_handle,
    const void* act,
    const void* W,
    void* y,
    float beta,
    void* workspace = nullptr,
    std::size_t workspace_bytes = 0)
{
    cudaStream_t stream = nullptr;
    cublasGetStream(cublas_handle, &stream);
    return run_bf16_lt_algo(ctx, plan, &algo, stream, act, W, y, beta,
                            workspace, workspace_bytes);
}

// Creates the descriptors for a shape and asks cuBLASLt which algorithms it
// would consider. Nothing is run: the caller decides which of `heuristics` to
// use, either by the shape ladder below or by measuring them.
std::shared_ptr<Bf16LtPlan> build_lt_plan(Bf16LtCtx& ctx, int M, int N, int K) {
    auto plan = std::make_shared<Bf16LtPlan>();
    cublasLtMatmulPreference_t pref = nullptr;

    cublasStatus_t st =
        cublasLtMatmulDescCreate(
            &plan->op_desc, bf16_compute_type(), CUDA_R_32F);
    cublasOperation_t transa = CUBLAS_OP_T;
    cublasOperation_t transb = CUBLAS_OP_N;
    if (st == CUBLAS_STATUS_SUCCESS) {
        st = cublasLtMatmulDescSetAttribute(
            plan->op_desc, CUBLASLT_MATMUL_DESC_TRANSA,
            &transa, sizeof(transa));
    }
    if (st == CUBLAS_STATUS_SUCCESS) {
        st = cublasLtMatmulDescSetAttribute(
            plan->op_desc, CUBLASLT_MATMUL_DESC_TRANSB,
            &transb, sizeof(transb));
    }
    if (st == CUBLAS_STATUS_SUCCESS) {
        st = cublasLtMatrixLayoutCreate(&plan->a_desc, CUDA_R_16BF, K, N, K);
    }
    if (st == CUBLAS_STATUS_SUCCESS) {
        st = cublasLtMatrixLayoutCreate(&plan->b_desc, CUDA_R_16BF, K, M, K);
    }
    if (st == CUBLAS_STATUS_SUCCESS) {
        st = cublasLtMatrixLayoutCreate(&plan->c_desc, CUDA_R_16BF, N, M, N);
    }
    if (st == CUBLAS_STATUS_SUCCESS) {
        st = cublasLtMatmulPreferenceCreate(&pref);
    }
    if (st == CUBLAS_STATUS_SUCCESS) {
        st = cublasLtMatmulPreferenceSetAttribute(
            pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
            &ctx.workspace_bytes, sizeof(ctx.workspace_bytes));
    }
    // Bar split-K algorithms that reduce IN PLACE. Those accumulate the
    // partial products straight into the output buffer, serialised by counters
    // in the workspace -- so every partial lands exactly once, but in the order
    // the CTAs happen to arrive. Floating-point addition is not associative, so
    // the last bit of the result depends on GPU scheduling, and a greedy decode
    // will silently pick a different token from one run to the next whenever
    // two logits are close. It is not hypothetical: enabling the fused MoE
    // changed the occupancy enough to flip GLM-5.2's step-13 argmax about half
    // the time, purely through the LM head's split-K order. The other two
    // schemes stage their partials in the workspace and reduce them in a fixed
    // order, which is reproducible; measured cost of the restriction is nil.
    if (st == CUBLAS_STATUS_SUCCESS) {
        const std::uint32_t deterministic_reductions =
            static_cast<std::uint32_t>(CUBLASLT_REDUCTION_SCHEME_MASK) &
            ~static_cast<std::uint32_t>(CUBLASLT_REDUCTION_SCHEME_INPLACE);
        st = cublasLtMatmulPreferenceSetAttribute(
            pref, CUBLASLT_MATMUL_PREF_REDUCTION_SCHEME_MASK,
            &deterministic_reductions, sizeof(deterministic_reductions));
    }

    cublasLtMatmulHeuristicResult_t heuristics[8]{};
    int returned = 0;
    if (st == CUBLAS_STATUS_SUCCESS) {
        st = cublasLtMatmulAlgoGetHeuristic(
            ctx.handle, plan->op_desc, plan->a_desc, plan->b_desc,
            plan->c_desc, plan->c_desc,
            pref, 8, heuristics, &returned);
    }
    if (pref) cublasLtMatmulPreferenceDestroy(pref);
    if (st != CUBLAS_STATUS_SUCCESS || returned <= 0) return nullptr;
    plan->heuristics.assign(heuristics, heuristics + returned);
    plan->algo = heuristics[0].algo;
    return plan;
}

// The plan for a shape, built once and shared. Descriptor creation and the
// heuristic query are host-side work that would otherwise repeat on every
// call.
std::shared_ptr<Bf16LtPlan> lt_plan_for(Bf16LtCtx& ctx, int M, int N, int K) {
    const Bf16LtKey key{M, N, K};
    auto& cache = Bf16LtPlanCache::instance();
    {
        std::lock_guard<std::mutex> lock(cache.mu);
        const auto it = cache.plans.find(key);
        if (it != cache.plans.end()) return it->second;
    }
    auto plan = build_lt_plan(ctx, M, N, K);
    if (!plan) return nullptr;
    std::lock_guard<std::mutex> lock(cache.mu);
    return cache.plans.emplace(key, plan).first->second;
}

bool gemm_bf16_lt_impl(
    cublasHandle_t cublas_handle,
    const void* act, const void* W, void* y,
    int M, int N, int K,
    float beta)
{
    auto& ctx = Bf16LtCtx::instance();
    ctx.ensure();
    auto plan = lt_plan_for(ctx, M, N, K);
    if (!plan) return false;

    const int returned = static_cast<int>(plan->heuristics.size());
    const int preferred = cublaslt_bf16_algo_index_for_shape(N, K);
    const int begin = std::min(preferred, std::max(0, returned - 1));
    for (int pass = 0; pass < 2; ++pass) {
        const int first = (pass == 0) ? begin : 0;
        const int last = (pass == 0) ? begin + 1 : returned;
        for (int i = first; i < last; ++i) {
            if (pass == 1 && i == begin) continue;
            if (run_bf16_lt_plan(ctx, *plan, plan->heuristics[i].algo,
                                 cublas_handle, act, W, y, beta)) {
                return true;
            }
        }
    }
    return false;
}

}  // namespace

void maybe_bench_lm_head_algos(cublasHandle_t, const void*, const void*, void*,
                              int, int, int) {}

CublasHandle::CublasHandle(cudaStream_t stream) {
    check(cublasCreate(&h_), "cublasCreate");
    if (stream) check(cublasSetStream(h_, stream), "cublasSetStream");
    // Allow tensor cores; bf16 multiplies with fp32 accumulation.
    check(cublasSetMathMode(h_, CUBLAS_TENSOR_OP_MATH), "cublasSetMathMode");
}

CublasHandle::~CublasHandle() {
    if (h_) cublasDestroy(h_);
}

void CublasHandle::set_stream(cudaStream_t s) {
    check(cublasSetStream(h_, s), "cublasSetStream");
}

cudaStream_t CublasHandle::stream() const noexcept {
    cudaStream_t s = nullptr;
    cublasGetStream(h_, &s);
    return s;
}

std::size_t runtime_quant_scratch_bytes(
    const RuntimeQuantScratchSpec& spec)
{
    if (spec.empty()) return 0;

    std::size_t bytes = kDefaultLtWorkspaceBytes;
    bytes = checked_add(
        bytes,
        runtime_quant_dequant_bytes(spec),
        "dequant scratch");

    if (spec.has_int8) {
        bytes = checked_add(
            bytes,
            checked_mul(spec.max_tokens, spec.max_weight_cols,
                        "int8 activation bytes"),
            "int8 activation scratch");
        bytes = checked_add(
            bytes,
            checked_mul(spec.max_tokens, sizeof(float),
                        "int8 activation scale bytes"),
            "int8 activation scale scratch");
        bytes = checked_add(
            bytes,
            checked_mul(
                checked_mul(spec.max_tokens, spec.max_weight_rows,
                            "int32 accumulator elems"),
                sizeof(std::int32_t),
                "int32 accumulator scratch"),
            "int32 accumulator scratch");
    }

    return bytes;
}

namespace {

// ---- Dense bf16 GEMM autotuning -------------------------------------------
//
// Every linear layer in the model ends up here, and which kernel is fastest
// for a given (M, N, K) is not something anyone can predict: it depends on the
// shape, the architecture, and the cuBLAS build. This used to be encoded as a
// ladder of hand-written special cases -- "Qwen3.6-27B's H=5120 projections
// prefer the first heuristic", "keep cuBLASLt out of the large-H wide-output
// path" -- which is a list of measurements someone took once, on models that
// are not the ones being served today. Take the measurement here instead.
//
// The candidates are the same three things the ladder was choosing between:
// the warp-per-row GEMV (M=1 only), classic `cublasGemmEx`, and each algorithm
// cuBLASLt's heuristic offers. They are ordered so that the incumbent choice
// comes first, and ties are broken towards the front of the list, so a shape
// where nothing measurably wins keeps doing what it did before.

// A candidate must beat the incumbent by this much to displace it; anything
// closer is treated as a tie. Below this the difference is timing noise, and
// switching on noise would make the kernel choice -- and so the last bit of
// every result -- vary between runs.
constexpr float kGemmTacticMargin = 0.98f;

enum class GemmKind : int { GemmEx = 0, Lt = 1, Gemv = 2 };

struct DenseTactic {
    int kind = static_cast<int>(GemmKind::GemmEx);
    int algo = 0;
};

bool run_gemm_ex(cublasHandle_t handle, const void* act, const void* W, void* y,
                 int M, int N, int K, float beta) {
    const float alpha = 1.f;
    return cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, &alpha, W,
                        CUDA_R_16BF, K, act, CUDA_R_16BF, K, &beta, y,
                        CUDA_R_16BF, N, bf16_compute_type(),
                        CUBLAS_GEMM_DEFAULT_TENSOR_OP) == CUBLAS_STATUS_SUCCESS;
}

// Runs `t` on `handle`'s stream. Returns false if the kernel is not usable for
// this shape, which lets the caller fall back rather than fail.
bool run_dense_tactic(cublasHandle_t handle, const DenseTactic& t,
                      const Bf16LtPlan* plan, const void* act, const void* W,
                      void* y, int M, int N, int K, float beta,
                      void* lt_workspace = nullptr,
                      std::size_t lt_workspace_bytes = 0,
                      const void* bias = nullptr) {
    // Only the GEMV epilogue can absorb a bias. Anything else must decline
    // rather than silently drop it.
    if (bias != nullptr && static_cast<GemmKind>(t.kind) != GemmKind::Gemv) {
        return false;
    }
    switch (static_cast<GemmKind>(t.kind)) {
        case GemmKind::Gemv: {
            cudaStream_t stream = nullptr;
            return (beta == 0.f || beta == 1.f) && M == 1 &&
                   cublas_stream(handle, stream) &&
                   kernels::launch_gemv_bf16(W, act, bias, y, N, K, stream,
                                             beta);
        }
        case GemmKind::Lt: {
            if (plan == nullptr ||
                t.algo >= static_cast<int>(plan->heuristics.size())) {
                return false;
            }
            auto& ctx = Bf16LtCtx::instance();
            return run_bf16_lt_plan(ctx, *plan, plan->heuristics[t.algo].algo,
                                    handle, act, W, y, beta, lt_workspace,
                                    lt_workspace_bytes);
        }
        case GemmKind::GemmEx:
        default:
            return run_gemm_ex(handle, act, W, y, M, N, K, beta);
    }
}

// Tuning has to be able to run while the caller's stream is mid graph capture:
// decode shapes are only ever seen inside `cudaStreamBeginCapture`, so a tuner
// that refused to run there would never see them. Capture is opened in
// `cudaStreamCaptureModeRelaxed`, which permits allocation and cross-stream
// synchronisation from the capturing thread, so the way to stay out of the
// graph is to own everything that carries work: a private stream, private
// events, and private activation and output buffers. The weights are shared,
// but they are read-only and were written long before.
//
// The one thing this deliberately does NOT own is the cuBLAS handle. Creating
// one mid-capture invalidates the capture -- `cublasCreate` initialises on the
// legacy default stream, which implicitly synchronises every blocking stream
// including the one being captured. So borrow the caller's handle and point it
// at the private stream for the duration, restoring it on the way out. Nothing
// else can be using it: we are inside one of its own calls.
struct DenseTuneArena {
    cudaStream_t stream = nullptr;
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    cublasHandle_t handle = nullptr;
    cudaStream_t caller_stream = nullptr;
    void* act = nullptr;
    void* y = nullptr;
    void* workspace = nullptr;
    std::size_t workspace_bytes = 0;

    ~DenseTuneArena() {
        // Restoring the stream also returns the borrowed handle to cuBLAS's
        // own workspace pool, undoing anything the probes did to it.
        if (handle) cublasSetStream(handle, caller_stream);
        if (start) cudaEventDestroy(start);
        if (stop) cudaEventDestroy(stop);
        if (act) cudaFree(act);
        if (y) cudaFree(y);
        if (workspace) cudaFree(workspace);
        if (stream) cudaStreamDestroy(stream);
        cudaGetLastError();
    }

    bool init(cublasHandle_t caller, int M, int N, int K) {
        const std::size_t act_bytes =
            static_cast<std::size_t>(M) * K * sizeof(std::uint16_t);
        const std::size_t y_bytes =
            static_cast<std::size_t>(M) * N * sizeof(std::uint16_t);
        // Must match what the heuristics were queried with, or an algorithm
        // that needs the full amount will be handed less than it asked for.
        workspace_bytes = Bf16LtCtx::instance().workspace_bytes;
        if (cudaMalloc(&act, act_bytes) != cudaSuccess ||
            cudaMalloc(&y, y_bytes) != cudaSuccess ||
            cudaMalloc(&workspace, workspace_bytes) != cudaSuccess ||
            cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) !=
                cudaSuccess ||
            cudaEventCreate(&start) != cudaSuccess ||
            cudaEventCreate(&stop) != cudaSuccess) {
            cudaGetLastError();
            return false;
        }
        if (!cublas_stream(caller, caller_stream)) return false;
        // The probes run beside the caller's stream, not behind it, so
        // anything still in flight there would overlap them. During capture
        // that cannot happen -- capture records, it does not execute, and
        // synchronising a capturing stream is an error -- but everywhere else
        // drain it first. This is once per shape, at the cost of a stall the
        // tuning sync would have imposed anyway.
        cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
        if (cudaStreamIsCapturing(caller_stream, &capture) != cudaSuccess) {
            cudaGetLastError();
            return false;
        }
        if (capture == cudaStreamCaptureStatusNone &&
            cudaStreamSynchronize(caller_stream) != cudaSuccess) {
            cudaGetLastError();
            return false;
        }
        if (cublasSetStream(caller, stream) != CUBLAS_STATUS_SUCCESS) {
            return false;
        }
        handle = caller;
        // A GEMM's cost does not depend on its values, only that they are
        // finite. 0x3C3C is a small positive bf16.
        if (cudaMemsetAsync(act, 0x3C, act_bytes, stream) != cudaSuccess ||
            cudaMemsetAsync(y, 0x3C, y_bytes, stream) != cudaSuccess ||
            cudaStreamSynchronize(stream) != cudaSuccess) {
            cudaGetLastError();
            return false;
        }
        return true;
    }
};

// Elapsed time of the fastest of `kIters` runs, or -1 if the candidate cannot
// run this shape. Failures are expected -- cuBLAS rejects some kernels for
// skinny shapes -- so they drop the candidate rather than propagate.
float time_dense_tactic(DenseTuneArena& arena, const DenseTactic& t,
                        const Bf16LtPlan* plan, const void* W, int M, int N,
                        int K, float beta) {
    constexpr int kWarmup = 3;
    constexpr int kIters = 7;
    for (int i = 0; i < kWarmup; ++i) {
        if (!run_dense_tactic(arena.handle, t, plan, arena.act, W, arena.y, M, N,
                              K, beta, arena.workspace, arena.workspace_bytes)) {
            cudaStreamSynchronize(arena.stream);
            cudaGetLastError();
            return -1.0f;
        }
    }
    if (cudaStreamSynchronize(arena.stream) != cudaSuccess) {
        cudaGetLastError();
        return -1.0f;
    }
    float best = -1.0f;
    for (int i = 0; i < kIters; ++i) {
        cudaEventRecord(arena.start, arena.stream);
        if (!run_dense_tactic(arena.handle, t, plan, arena.act, W, arena.y, M, N,
                              K, beta, arena.workspace, arena.workspace_bytes)) {
            cudaStreamSynchronize(arena.stream);
            cudaGetLastError();
            return -1.0f;
        }
        cudaEventRecord(arena.stop, arena.stream);
        if (cudaEventSynchronize(arena.stop) != cudaSuccess) {
            cudaGetLastError();
            return -1.0f;
        }
        float ms = 0.0f;
        if (cudaEventElapsedTime(&ms, arena.start, arena.stop) != cudaSuccess) {
            cudaGetLastError();
            return -1.0f;
        }
        if (best < 0.0f || ms < best) best = ms;
    }
    return best;
}

std::vector<DenseTactic> dense_candidates(const Bf16LtPlan* plan, int M, int N,
                                          int K, float beta) {
    std::vector<DenseTactic> out;
    // Ordered by what the shape would have used without tuning, because ties
    // resolve to the first entry.
    // beta = 1 too: the GEMV folds the accumulate into its epilogue, and
    // excluding it meant every projection that adds into a residual -- o_proj
    // on every model here -- was decided without its fastest candidate on the
    // ballot.
    if (M == 1 && (beta == 0.f || beta == 1.f)) {
        out.push_back({static_cast<int>(GemmKind::Gemv), 0});
    }
    out.push_back({static_cast<int>(GemmKind::GemmEx), 0});
    if (plan != nullptr) {
        const int preferred = cublaslt_bf16_algo_index_for_shape(N, K);
        const int count = static_cast<int>(plan->heuristics.size());
        if (preferred < count) {
            out.push_back({static_cast<int>(GemmKind::Lt), preferred});
        }
        for (int i = 0; i < count; ++i) {
            if (i == preferred) continue;
            out.push_back({static_cast<int>(GemmKind::Lt), i});
        }
    }
    return out;
}

std::string dense_cache_signature() {
    int device = 0;
    cudaDeviceProp prop{};
    if (cudaGetDevice(&device) != cudaSuccess ||
        cudaGetDeviceProperties(&prop, device) != cudaSuccess) {
        cudaGetLastError();
        return {};
    }
    int version = 0;
    cublasGetVersion(nullptr, &version);
    char buf[384];
    std::snprintf(buf, sizeof(buf),
                  "# pie-dense-gemm v1 sm%d%d cublas=%d dev=%s", prop.major,
                  prop.minor, version, prop.name);
    return buf;
}

struct DenseGemmTuner {
    std::mutex mu;
    std::unordered_map<std::uint64_t, DenseTactic> chosen;
    std::unordered_map<std::uint64_t, int> seen;
    TuningCache disk{"dense_gemm.txt", dense_cache_signature()};

    static DenseGemmTuner& instance() {
        return per_device_singleton<DenseGemmTuner>();
    }
};

// Ceiling on how many shapes will ever be measured, so a workload with an
// unbounded spread of shapes cannot spend unbounded time tuning or grow the
// on-disk cache without limit. The decode lattice is a few dozen shapes per
// model; this is far above it.
constexpr std::size_t kMaxTunedShapes = 1024;

std::uint64_t dense_key(int M, int N, int K, float beta) {
    std::uint64_t h = 0;
    h = tuning_hash(h, static_cast<std::uint64_t>(M));
    h = tuning_hash(h, static_cast<std::uint64_t>(N));
    h = tuning_hash(h, static_cast<std::uint64_t>(K));
    h = tuning_hash(h, beta == 0.f ? 0u : 1u);
    return h;
}

DenseTactic tune_dense(cublasHandle_t caller, const Bf16LtPlan* plan,
                       const void* W, int M, int N, int K, float beta) {
    const std::vector<DenseTactic> candidates =
        dense_candidates(plan, M, N, K, beta);
    DenseTactic best = candidates.front();

    DenseTuneArena arena;
    if (!arena.init(caller, M, N, K)) {
        return best;
    }

    std::vector<std::pair<int, float>> timings;
    timings.reserve(candidates.size());
    float fastest = -1.0f;
    for (int i = 0; i < static_cast<int>(candidates.size()); ++i) {
        const float ms =
            time_dense_tactic(arena, candidates[i], plan, W, M, N, K, beta);
        if (ms <= 0.0f) continue;
        timings.emplace_back(i, ms);
        if (fastest < 0.0f || ms < fastest) fastest = ms;
    }
    // PIE_GEMM_TUNE_LOG also dumps every candidate's measured time, not just
    // the winner: knowing that the GEMV lost is not the same as knowing by how
    // much, and the gap is what says whether a better kernel is worth writing.
    static const bool tune_log = std::getenv("PIE_GEMM_TUNE_LOG") != nullptr;
    if (tune_log) {
        for (const auto& [i, ms] : timings) {
            const int kind = candidates[i].kind;
            std::fprintf(stderr,
                "[gemm-cand] M=%d N=%d K=%d %s(algo=%d) %.1f us\n",
                M, N, K,
                kind == static_cast<int>(GemmKind::Gemv)   ? "gemv"
              : kind == static_cast<int>(GemmKind::Lt)     ? "lt"
                                                           : "gemmex",
                candidates[i].algo, ms * 1000.0f);
        }
    }
    if (fastest <= 0.0f) return best;

    const float cutoff = fastest / kGemmTacticMargin;
    float chosen_ms = fastest;
    for (const auto& [i, ms] : timings) {
        if (ms > cutoff) continue;
        best = candidates[i];
        chosen_ms = ms;
        break;
    }
    return best;
}

// Chooses (and on first sight of a shape, measures) the kernel for this shape.
// Returns false if no measured choice is available, leaving the caller on its
// original path.
bool dense_tactic_for(cublasHandle_t caller, const void* W, int M, int N,
                      int K, float beta, cudaStreamCaptureStatus capturing,
                      const Bf16LtPlan** out_plan, DenseTactic* out) {
    // The arena allocates an M x N output. Tuning a shape whose output alone
    // would rival the KV cache is not worth the memory; those shapes are large
    // enough that cuBLAS's own choice is close to optimal anyway.
    constexpr std::size_t kMaxTuneOutputBytes = 256ull * 1024 * 1024;
    if (static_cast<std::size_t>(M) * N * sizeof(std::uint16_t) >
        kMaxTuneOutputBytes) {
        return false;
    }

    auto& ctx = Bf16LtCtx::instance();
    ctx.ensure();
    // Built for every shape, tuned or not: it is the source of the cuBLASLt
    // candidates, and the tactic we cache names one of them by index.
    std::shared_ptr<Bf16LtPlan> plan =
        lt_plan_for(ctx, M, N, K);
    *out_plan = plan.get();

    auto& tuner = DenseGemmTuner::instance();
    const std::uint64_t key = dense_key(M, N, K, beta);
    std::lock_guard<std::mutex> lock(tuner.mu);
    const auto it = tuner.chosen.find(key);
    if (it != tuner.chosen.end()) {
        *out = it->second;
        return true;
    }
    if (tuner.chosen.size() >= kMaxTunedShapes) return false;

    // Measuring a shape costs ~10 kernel launches per candidate plus a stall,
    // which is only worth paying for a shape that will come back. Decode
    // shapes are seen exactly once here -- during graph capture -- and then
    // replayed forever from the graph, so those must be tuned on sight or
    // never. Everything else is prefill, whose M is the token count and so is
    // effectively arbitrary; make it prove it recurs before spending anything
    // on it.
    if (capturing == cudaStreamCaptureStatusNone && ++tuner.seen[key] < 2) {
        return false;
    }

    DenseTactic tactic{};
    if (!tuner.disk.lookup(key, &tactic.kind, &tactic.algo) ||
        tactic.kind < 0 || tactic.kind > static_cast<int>(GemmKind::Gemv)) {
        tactic = tune_dense(caller, plan.get(), W, M, N, K, beta);
        tuner.disk.store(key, tactic.kind, tactic.algo);
    }
    tuner.chosen.emplace(key, tactic);
    // PIE_GEMM_TUNE_LOG: which kernel a shape ended up on. Logged HERE rather
    // than inside the tuner because the choice is cached on disk, so on any
    // machine that has run the model once the tuner never executes again and a
    // log inside it prints nothing.
    static const bool tune_log = std::getenv("PIE_GEMM_TUNE_LOG") != nullptr;
    if (tune_log) {
        const char* kind =
            tactic.kind == static_cast<int>(GemmKind::Gemv) ? "gemv"
          : tactic.kind == static_cast<int>(GemmKind::Lt)   ? "lt"
                                                            : "gemmex";
        std::fprintf(stderr, "[gemm-tune] M=%d N=%d K=%d -> %s(algo=%d)\n",
                     M, N, K, kind, tactic.algo);
    }
    *out = tactic;
    return true;
}

// Side-effect-free peek at the tuner's verdict for this shape. Deliberately
// does *not* call `dense_tactic_for`: that bumps the `seen` counter and can
// trigger a tune, and asking "would you have picked the GEMV?" must not
// change the answer to "what will you pick?".
bool dense_tactic_already_gemv(int M, int N, int K, float beta) {
    auto& tuner = DenseGemmTuner::instance();
    std::lock_guard<std::mutex> lock(tuner.mu);
    const auto it = tuner.chosen.find(dense_key(M, N, K, beta));
    return it != tuner.chosen.end() &&
           static_cast<GemmKind>(it->second.kind) == GemmKind::Gemv;
}

void gemm_bf16_impl(
    cublasHandle_t handle,
    const void* act, const void* W, void* y,
    int M, int N, int K,
    float beta)
{
    const float alpha = 1.f;
    // Which of the three kernel families below is fastest for this shape is a
    // measurement, not a rule -- so take it, once per shape, and remember it.
    // Everything after this point is the fallback for shapes the tuner
    // declined (too large to allocate a probe output for) or could not run.
    {
        cudaStream_t caller_stream = nullptr;
        cudaStreamCaptureStatus capturing = cudaStreamCaptureStatusNone;
        const Bf16LtPlan* plan = nullptr;
        DenseTactic tactic{};
        if (cublas_stream(handle, caller_stream) &&
            cudaStreamIsCapturing(caller_stream, &capturing) == cudaSuccess &&
            dense_tactic_for(handle, W, M, N, K, beta, capturing, &plan,
                             &tactic) &&
            run_dense_tactic(handle, tactic, plan, act, W, y, M, N, K, beta)) {
            return;
        }
        cudaGetLastError();
    }
    // M=1 is the decode shape: a single activation row against the whole
    // weight, so there is no reuse for a tiled GEMM to exploit and the
    // call is a pure streaming read. cuBLAS picks kernels sized for an M
    // worth filling and reaches roughly half of HBM bandwidth on these;
    // a warp-per-row GEMV nearly doubles it (see `launch_gemv_bf16`).
    cudaStream_t gemv_stream = nullptr;
    if (M == 1 && beta == 0.f &&
        cublas_stream(handle, gemv_stream) &&
        kernels::launch_gemv_bf16(W, act, nullptr, y, N, K, gemv_stream)) {
        return;
    }
    const int lt_max_n = kCublasLtBf16MaxN;
    if (M >= kCublasLtBf16MinM &&
        N >= cublaslt_bf16_min_n(K) &&
        K >= kCublasLtBf16MinK &&
        (lt_max_n == 0 || N <= lt_max_n) &&
        gemm_bf16_lt_impl(handle, act, W, y, M, N, K, beta)) {
        return;
    }
    const auto status = cublasGemmEx(
        handle,
        /*transa=*/CUBLAS_OP_T, /*transb=*/CUBLAS_OP_N,
        /*m=*/N, /*n=*/M, /*k=*/K,
        &alpha,
        /*A=*/W,   CUDA_R_16BF, /*lda=*/K,
        /*B=*/act, CUDA_R_16BF, /*ldb=*/K,
        &beta,
        /*C=*/y,   CUDA_R_16BF, /*ldc=*/N,
        bf16_compute_type(),
        CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    if (status == CUBLAS_STATUS_NOT_SUPPORTED) {
        // `CUBLAS_GEMM_DEFAULT_TENSOR_OP` pins the tensor-core kernel family,
        // and cuBLAS has no member of it for some skinny shapes: the packed
        // q/k/v projection at TP=2 (N = Hq/T + 2*Hk/T = 2048, K = 1024) is
        // rejected at M=1, while the same M=1 succeeds at the TP=1 width
        // (N=4096) and at the packed gate/up width (N=3072). M=1 is not a
        // serving shape — it is the R=1 rung of the graph lattice, so this
        // surfaces only during upfront capture, and under TP it surfaced as a
        // HANG rather than an error: rank 0 threw out of capture while the
        // follower sat in `tp_graph_capture_barrier` waiting for a peer that
        // was never going to arrive.
        //
        // Retry without the tensor-op pin. cuBLAS then picks whatever kernel
        // fits; at M=1 there is no tensor-core throughput to lose anyway.
        const auto retry = cublasGemmEx(
            handle,
            /*transa=*/CUBLAS_OP_T, /*transb=*/CUBLAS_OP_N,
            /*m=*/N, /*n=*/M, /*k=*/K,
            &alpha,
            /*A=*/W,   CUDA_R_16BF, /*lda=*/K,
            /*B=*/act, CUDA_R_16BF, /*ldb=*/K,
            &beta,
            /*C=*/y,   CUDA_R_16BF, /*ldc=*/N,
            bf16_compute_type(),
            CUBLAS_GEMM_DEFAULT);
        if (retry == CUBLAS_STATUS_SUCCESS) return;
        // Neither GemmEx algorithm family covers the shape. cuBLASLt does —
        // it is normally skipped here by the `min_m`/`min_n` heuristics, which
        // exist to pick the FASTER path, not the only working one.
        if (gemm_bf16_lt_impl(handle, act, W, y, M, N, K, beta)) return;
        throw std::runtime_error(
            "cuBLAS error (" + std::to_string(static_cast<int>(retry)) +
            ") after non-tensor-op and cuBLASLt retries: cublasGemmEx[bf16] M=" +
            std::to_string(M) + " N=" + std::to_string(N) +
            " K=" + std::to_string(K));
    }
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error(
            "cuBLAS error (" + std::to_string(static_cast<int>(status)) +
            "): cublasGemmEx[bf16] M=" + std::to_string(M) +
            " N=" + std::to_string(N) + " K=" + std::to_string(K));
    }
}

void gemm_bf16_out_fp32_impl(
    cublasHandle_t handle,
    const void* act,
    const void* W,
    float* y,
    int M,
    int N,
    int K)
{
    const float alpha = 1.f;
    const float beta = 0.f;
    const auto status = cublasGemmEx(
        handle,
        /*transa=*/CUBLAS_OP_T, /*transb=*/CUBLAS_OP_N,
        /*m=*/N, /*n=*/M, /*k=*/K,
        &alpha,
        /*A=*/W,   CUDA_R_16BF, /*lda=*/K,
        /*B=*/act, CUDA_R_16BF, /*ldb=*/K,
        &beta,
        /*C=*/y,   CUDA_R_32F, /*ldc=*/N,
        bf16_compute_type(),
        CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error(
            "cuBLAS error (" + std::to_string(static_cast<int>(status)) +
            "): cublasGemmEx[bf16->fp32] M=" + std::to_string(M) +
            " N=" + std::to_string(N) + " K=" + std::to_string(K));
    }
}

void gemm_bf16_to_fp32_impl(
    cublasHandle_t handle,
    const void* act, const void* W, void* y,
    int M, int N, int K,
    float beta)
{
    const float alpha = 1.f;
    const auto status = cublasGemmEx(
        handle,
        /*transa=*/CUBLAS_OP_T, /*transb=*/CUBLAS_OP_N,
        /*m=*/N, /*n=*/M, /*k=*/K,
        &alpha,
        /*A=*/W,   CUDA_R_16BF, /*lda=*/K,
        /*B=*/act, CUDA_R_16BF, /*ldb=*/K,
        &beta,
        /*C=*/y,   CUDA_R_32F,  /*ldc=*/N,
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error(
            "cuBLAS error (" + std::to_string(static_cast<int>(status)) +
            "): cublasGemmEx[bf16→fp32] M=" + std::to_string(M) +
            " N=" + std::to_string(N) + " K=" + std::to_string(K));
    }
}

void gemm_bf16_cublas_impl(
    cublasHandle_t handle,
    const void* act, const void* W, void* y,
    int M, int N, int K,
    float beta)
{
    const float alpha = 1.f;
    const auto status = cublasGemmEx(
        handle,
        /*transa=*/CUBLAS_OP_T, /*transb=*/CUBLAS_OP_N,
        /*m=*/N, /*n=*/M, /*k=*/K,
        &alpha,
        /*A=*/W,   CUDA_R_16BF, /*lda=*/K,
        /*B=*/act, CUDA_R_16BF, /*ldb=*/K,
        &beta,
        /*C=*/y,   CUDA_R_16BF, /*ldc=*/N,
        bf16_compute_type(),
        CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error(
            "cuBLAS error (" + std::to_string(static_cast<int>(status)) +
            "): cublasGemmEx[bf16:cublas] M=" + std::to_string(M) +
            " N=" + std::to_string(N) + " K=" + std::to_string(K));
    }
}

// Whether `cublasGemmGroupedBatchedEx` can serve a given shape is only
// discoverable by calling it and looking at the status. That is fine on a plain
// stream, but a failed cuBLAS call inside a stream capture INVALIDATES the
// capture, and the next GEMM then dies with an unrelated INTERNAL_ERROR far from
// the cause -- intermittently, because which shapes reach a capture first
// depends on rank timing. So speculate only outside capture, remember the answer
// per shape, and while capturing an untried shape go straight to the batched
// path.
struct GroupedBatchedSupport {
    // -1 unknown, 0 unsupported, 1 supported.
    int lookup(std::uint64_t key) {
        std::lock_guard<std::mutex> lock(mu);
        const auto it = known.find(key);
        return it == known.end() ? -1 : (it->second ? 1 : 0);
    }

    void store(std::uint64_t key, bool supported) {
        std::lock_guard<std::mutex> lock(mu);
        known.emplace(key, supported);
    }

    std::mutex mu;
    std::unordered_map<std::uint64_t, bool> known;
};

bool stream_is_capturing(cublasHandle_t handle) {
    cudaStream_t stream = nullptr;
    if (cublasGetStream(handle, &stream) != CUBLAS_STATUS_SUCCESS) return true;
    cudaStreamCaptureStatus status = cudaStreamCaptureStatusNone;
    if (cudaStreamIsCapturing(stream, &status) != cudaSuccess) return true;
    return status != cudaStreamCaptureStatusNone;
}

void gemm_batched_bf16_impl(
    cublasHandle_t handle,
    const void* const* act_ptrs_dev,
    const void* const* W_ptrs_dev,
    void* const*       y_ptrs_dev,
    int M, int N, int K,
    int batch_count,
    float beta)
{
    if (batch_count <= 0) return;
    const float alpha = 1.f;
    const std::uint64_t grouped_key =
        dense_key(M, N, K, beta) ^
        (static_cast<std::uint64_t>(batch_count) * 0x9E3779B97F4A7C15ull);
    auto& grouped_support = per_device_singleton<GroupedBatchedSupport>();
    const int grouped_known = grouped_support.lookup(grouped_key);
    const bool try_grouped =
        (grouped_known == 1 ||
         (grouped_known < 0 && !stream_is_capturing(handle)));
    if (try_grouped) {
        const cublasOperation_t transa_array[1] = {CUBLAS_OP_T};
        const cublasOperation_t transb_array[1] = {CUBLAS_OP_N};
        const int m_array[1] = {N};
        const int n_array[1] = {M};
        const int k_array[1] = {K};
        const int lda_array[1] = {K};
        const int ldb_array[1] = {K};
        const int ldc_array[1] = {N};
        const int group_size[1] = {batch_count};
        const auto status = cublasGemmGroupedBatchedEx(
            handle,
            transa_array, transb_array,
            m_array, n_array, k_array,
            &alpha,
            W_ptrs_dev, CUDA_R_16BF, lda_array,
            act_ptrs_dev, CUDA_R_16BF, ldb_array,
            &beta,
            y_ptrs_dev, CUDA_R_16BF, ldc_array,
            /*group_count=*/1, group_size,
            bf16_compute_type());
        if (grouped_known < 0) {
            grouped_support.store(grouped_key,
                                  status == CUBLAS_STATUS_SUCCESS);
        }
        if (status == CUBLAS_STATUS_SUCCESS) {
            return;
        }
    }
    const auto status = cublasGemmBatchedEx(
              handle,
              /*transa=*/CUBLAS_OP_T, /*transb=*/CUBLAS_OP_N,
              /*m=*/N, /*n=*/M, /*k=*/K,
              &alpha,
              /*A=*/W_ptrs_dev,   CUDA_R_16BF, /*lda=*/K,
              /*B=*/act_ptrs_dev, CUDA_R_16BF, /*ldb=*/K,
              &beta,
              /*C=*/y_ptrs_dev,   CUDA_R_16BF, /*ldc=*/N,
              batch_count,
              bf16_compute_type(),
              CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    if (status != CUBLAS_STATUS_SUCCESS) {
        // cuBLAS reports INTERNAL_ERROR for anything it cannot explain,
        // including a CUDA error that was already sticky before the call.
        // Report the surrounding CUDA state so the message names the real
        // fault instead of the call that noticed it.
        int device = -1;
        static_cast<void>(cudaGetDevice(&device));
        const cudaError_t pending = cudaPeekAtLastError();
        cudaStream_t stream = nullptr;
        std::string capture = "unknown";
        if (cublasGetStream(handle, &stream) == CUBLAS_STATUS_SUCCESS) {
            cudaStreamCaptureStatus capture_status =
                cudaStreamCaptureStatusNone;
            if (cudaStreamIsCapturing(stream, &capture_status) ==
                cudaSuccess) {
                capture = capture_status == cudaStreamCaptureStatusActive
                    ? "active"
                    : (capture_status ==
                       cudaStreamCaptureStatusInvalidated)
                        ? "INVALIDATED"
                        : "none";
            }
        }
        std::size_t free_bytes = 0, total_bytes = 0;
        static_cast<void>(cudaMemGetInfo(&free_bytes, &total_bytes));
        throw std::runtime_error(
            "cuBLAS error (" + std::to_string(static_cast<int>(status)) +
            "): cublasGemmBatchedEx[bf16] M=" + std::to_string(M) +
            " N=" + std::to_string(N) + " K=" + std::to_string(K) +
            " batch=" + std::to_string(batch_count) +
            " device=" + std::to_string(device) +
            " capture=" + capture +
            " pending_cuda=" + cudaGetErrorName(pending) +
            " free_mib=" + std::to_string(free_bytes >> 20));
    }
}

void gemm_grouped_bf16_impl(
    cublasHandle_t handle,
    const void* const* act_ptrs_host,
    const void* const* W_ptrs_host,
    void* const*       y_ptrs_host,
    const int*         M_array_host,
    int group_count,
    int N,
    int K,
    float beta)
{
    if (group_count <= 0) return;

    std::vector<cublasOperation_t> transa(group_count, CUBLAS_OP_T);
    std::vector<cublasOperation_t> transb(group_count, CUBLAS_OP_N);
    std::vector<int> m(group_count, N);
    std::vector<int> n(group_count);
    std::vector<int> k(group_count, K);
    std::vector<int> lda(group_count, K);
    std::vector<int> ldb(group_count, K);
    std::vector<int> ldc(group_count, N);
    std::vector<int> group_size(group_count, 1);
    std::vector<float> alpha(group_count, 1.f);
    std::vector<float> beta_values(group_count, beta);

    for (int i = 0; i < group_count; ++i) {
        n[i] = M_array_host[i];
    }

    auto run = [&](cublasComputeType_t compute) {
        return cublasGemmGroupedBatchedEx(
            handle,
            transa.data(), transb.data(),
            m.data(), n.data(), k.data(),
            alpha.data(),
            W_ptrs_host,   CUDA_R_16BF, lda.data(),
            act_ptrs_host, CUDA_R_16BF, ldb.data(),
            beta_values.data(),
            y_ptrs_host,   CUDA_R_16BF, ldc.data(),
            group_count,
            group_size.data(),
            compute);
    };
    // No FAST_16BF attempt first: it has no algorithm for these shapes, and a
    // failed call inside a graph capture invalidates the capture.
    const cublasStatus_t status = run(bf16_compute_type());
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error(
            "cuBLAS error (" + std::to_string(static_cast<int>(status)) +
            "): cublasGemmGroupedBatchedEx[bf16] groups=" +
            std::to_string(group_count) + " N=" + std::to_string(N) +
            " K=" + std::to_string(K));
    }
}

[[noreturn]] void unsupported(const char* api,
                              DType act_dtype, DType w_dtype, DType y_dtype) {
    throw std::runtime_error(
        std::string("ops::") + api + ": unsupported dtype combo (act=" +
            dtype_name(act_dtype) + ", w=" + dtype_name(w_dtype) +
        ", y=" + dtype_name(y_dtype) + ")");
}

void validate_quant_weight_view(const char* api, const WeightView& w, int N, int K) {
    if (w.data == nullptr) {
        throw std::runtime_error(std::string(api) + ": quant weight data is null");
    }
    if (w.scale_data == nullptr) {
        throw std::runtime_error(std::string(api) + ": quant scale data is null");
    }
    const bool is_nibble_packed =
        w.dtype == DType::INT4_PACKED || w.dtype == DType::MXFP4_PACKED;
    const std::size_t expected_weight_bytes = is_nibble_packed
        ? (static_cast<std::size_t>(N) * static_cast<std::size_t>(K) + 1) / 2
        : static_cast<std::size_t>(N) * static_cast<std::size_t>(K) *
              dtype_bytes(w.dtype);
    if (w.nbytes < expected_weight_bytes) {
        throw std::runtime_error(
            std::string(api) + ": quant weight buffer is smaller than GEMM "
            "shape requires; have " + std::to_string(w.nbytes) +
            " bytes, need " + std::to_string(expected_weight_bytes) +
            " bytes for N=" + std::to_string(N) +
            " K=" + std::to_string(K));
    }
    // PerTensor → 1 scale; PerChannel → N; PerGroup → N×ceil(K/gs),
    // except 2D block-scaled FP8 (DeepSeek) which is ceil(N/gs)×ceil(K/gs).
    std::size_t expected_scales = 1;
    if (w.quant_kind == QuantMeta::Kind::PerChannel) {
        expected_scales = static_cast<std::size_t>(N);
    } else if (w.quant_kind == QuantMeta::Kind::PerGroup && w.group_size > 0) {
        if (w.dtype == DType::FP8_E4M3) {
            // 2D block-scaled FP8: scales are [ceil(N/gs), ceil(K/gs)]
            expected_scales =
                static_cast<std::size_t>((N + w.group_size - 1) / w.group_size) *
                static_cast<std::size_t>((K + w.group_size - 1) / w.group_size);
        } else {
            expected_scales = static_cast<std::size_t>(N) *
                static_cast<std::size_t>((K + w.group_size - 1) / w.group_size);
        }
    }
    if (w.scale_numel < expected_scales) {
        throw std::runtime_error(
            std::string(api) + ": quant scale tensor is smaller than GEMM "
            "shape requires; have " + std::to_string(w.scale_numel) +
            " values, need " + std::to_string(expected_scales));
    }
}

// ── cuBLASLt FP8 path ─────────────────────────────────────────────────
// cuBLASLt supports mixed FP8(weight) × BF16(act) → BF16(out) with FP32
// accumulation and a per-tensor (or per-channel — tested separately) scale
// pointer for the FP8 operand. Reference impl that this is adapted from:
// flashinfer-src/include/flashinfer/gemm/bmm_fp8.cuh.

void check_lt(cublasStatus_t s, const char* expr) {
    if (s != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error(
            std::string("cuBLASLt error (") + std::to_string(int(s)) + "): " +
            cublasGetStatusString(s) + " at " + expr);
    }
}

#define LT_CHECK(EXPR) ::pie_cuda_driver::ops::check_lt((EXPR), #EXPR)

// Tiny RAII wrappers — we only need the three descriptor types for one
// matmul, so the boilerplate stays inline.
struct LtMatmulDesc {
    cublasLtMatmulDesc_t d = nullptr;
    LtMatmulDesc(cublasComputeType_t compute, cudaDataType_t scale_type) {
        LT_CHECK(cublasLtMatmulDescCreate(&d, compute, scale_type));
    }
    ~LtMatmulDesc() { if (d) cublasLtMatmulDescDestroy(d); }
    LtMatmulDesc(const LtMatmulDesc&) = delete;
    template <typename T>
    void set(cublasLtMatmulDescAttributes_t attr, const T value) {
        LT_CHECK(cublasLtMatmulDescSetAttribute(d, attr, &value, sizeof(T)));
    }
};

struct LtMatrixLayout {
    cublasLtMatrixLayout_t d = nullptr;
    LtMatrixLayout(cudaDataType_t type, std::uint64_t rows, std::uint64_t cols,
                   std::int64_t ld) {
        LT_CHECK(cublasLtMatrixLayoutCreate(&d, type, rows, cols, ld));
    }
    ~LtMatrixLayout() { if (d) cublasLtMatrixLayoutDestroy(d); }
    LtMatrixLayout(const LtMatrixLayout&) = delete;
};

struct LtMatmulPref {
    cublasLtMatmulPreference_t d = nullptr;
    LtMatmulPref() { LT_CHECK(cublasLtMatmulPreferenceCreate(&d)); }
    ~LtMatmulPref() { if (d) cublasLtMatmulPreferenceDestroy(d); }
    LtMatmulPref(const LtMatmulPref&) = delete;
    template <typename T>
    void set(cublasLtMatmulPreferenceAttributes_t attr, const T value) {
        LT_CHECK(cublasLtMatmulPreferenceSetAttribute(d, attr, &value, sizeof(T)));
    }
};

#ifdef PIE_CUDA_HAS_MARLIN
// Per-DEVICE marlin workspace. Marlin's split-K reduce uses one int32
// per SM as a barrier counter; we allocate generously (16 KiB) to cover
// every realistic SM count without per-call allocation. Lazy-init on
// first INT4_PACKED dispatch. Keyed by device so an in-process TP rank
// never barriers through another rank's memory.
struct MarlinWorkspace {
    void* ptr = nullptr;
    std::size_t bytes = 0;
};
struct MarlinBarrierWs : MarlinWorkspace {};
struct MarlinReduceWs : MarlinWorkspace {};
struct MarlinResidualWs : MarlinWorkspace {};

void* marlin_workspace_() {
    auto& ws = per_device_singleton<MarlinBarrierWs>();
    if (!ws.ptr) {
        ws.bytes = 16 * 1024;
        if (cudaMalloc(&ws.ptr, ws.bytes) != cudaSuccess) {
            throw std::runtime_error("marlin: cudaMalloc workspace failed");
        }
        cudaMemset(ws.ptr, 0, ws.bytes);
    }
    return ws.ptr;
}

void* marlin_fp32_reduce_scratch_() {
    auto& ws = per_device_singleton<MarlinReduceWs>();
    if (!ws.ptr) {
        ws.bytes = 32 * 1024 * 1024;
        if (cudaMalloc(&ws.ptr, ws.bytes) != cudaSuccess) {
            throw std::runtime_error(
                "marlin: cudaMalloc fp32 reduce scratch failed");
        }
    }
    return ws.ptr;
}

// Per-device bf16 residual scratch — used when the INT4 dispatcher is
// called with beta=1 (the residual-add fusion the bf16/fp8 paths handle
// natively via cuBLAS's beta param). Marlin overwrites C, so we run it
// into a scratch and add into y in a second pass. Grows monotonically.
void* marlin_residual_scratch_(std::size_t bytes) {
    auto& ws = per_device_singleton<MarlinResidualWs>();
    if (bytes <= ws.bytes) return ws.ptr;
    if (ws.ptr) cudaFree(ws.ptr);
    if (cudaMalloc(&ws.ptr, bytes) != cudaSuccess) {
        throw std::runtime_error(
            "marlin: cudaMalloc residual scratch failed (" +
            std::to_string(bytes) + " bytes)");
    }
    ws.bytes = bytes;
    return ws.ptr;
}
#endif

// Per-process cuBLASLt handle + workspace. One forward thread per rank
// makes a thread-local unnecessary; lazy-init at first FP8 GEMM.
//
// `dequant_scratch` is a sm<89 fallback: when cuBLASLt has no algorithm
// for the FP8×BF16 matmul (true on Ampere/A100), we dequantize the FP8
// weight to bf16 here and run the classic cuBLAS bf16 path. The scratch
// grows monotonically to fit the largest weight we've seen — quant
// projections are loaded once at boot, so the steady-state cost is one
// allocation per unique projection size.
struct LtCtx {
    cublasLtHandle_t handle = nullptr;
    void*            workspace = nullptr;
    DeviceMemoryBlock workspace_block{};
    std::size_t      workspace_bytes = 0;
    int              compute_capability_major = 0;  // 0 = unqueried
    bool             fp8_native_supported = false;

    static LtCtx& instance() {
        if (g_runtime_quant_context != nullptr) {
            return *g_runtime_quant_context;
        }
        static thread_local LtCtx fallback;
        return fallback;
    }

    void ensure_init(std::size_t ws_bytes = kDefaultLtWorkspaceBytes) {
        if (!handle) LT_CHECK(cublasLtCreate(&handle));
        if (!workspace) {
            workspace_block = allocate_device_memory(ws_bytes, 256);
            workspace = workspace_block.ptr;
            workspace_bytes = ws_bytes;
        }
        if (compute_capability_major == 0) {
            int dev = 0;
            CUDA_CHECK(cudaGetDevice(&dev));
            int major = 0, minor = 0;
            CUDA_CHECK(cudaDeviceGetAttribute(
                &major, cudaDevAttrComputeCapabilityMajor, dev));
            CUDA_CHECK(cudaDeviceGetAttribute(
                &minor, cudaDevAttrComputeCapabilityMinor, dev));
            compute_capability_major = major;
            // cuBLASLt FP8 (E4M3) GEMM requires sm89 (Ada) or sm90+
            // (Hopper). On older arch we route through the dequant
            // fallback. We probe on first use rather than trusting the
            // capability check alone — different cuBLAS versions also
            // matter.
            fp8_native_supported = (major > 8) || (major == 8 && minor >= 9);
        }
    }

    // Grow-on-demand device scratch. Caller passes byte size; returns
    // a pointer valid until the next `ensure(bigger_size)` on the same buffer.
    struct GrowScratch {
        DeviceMemoryBlock block{};
        std::size_t bytes = 0;
        bool        sealed = false;
        const char* name = "runtime quant scratch";

        void reserve(std::size_t want) {
            if (want <= bytes) return;
            if (sealed) {
                throw std::runtime_error(
                    std::string(name) +
                    " attempted to grow after CUDA graph reservation: want " +
                    std::to_string(want) + " bytes, have " +
                    std::to_string(bytes) +
                    " bytes. Increase the planner reserve or disable CUDA graphs.");
            }
            free_device_memory(block);
            block = allocate_device_memory(want, 256);
            bytes = want;
        }

        void* ensure(std::size_t want) {
            reserve(want);
            return block.ptr;
        }

        void seal(const char* label) noexcept {
            if (label != nullptr) name = label;
            sealed = true;
        }

        void reset() noexcept {
            free_device_memory(block);
            block = {};
            bytes = 0;
            sealed = false;
        }
    };
    GrowScratch dequant;        // sm<89 FP8 → bf16 weight scratch
    GrowScratch int8_act;       // [M, K] int8 quantised activation
    GrowScratch int8_act_scale; // [M] fp32 act_scale_inv
    GrowScratch int32_acc;      // [M, N] int32 W8A8 accumulator
    GrowScratch fp8_act;        // [M, K] fp8 blockwise-quantised activation
    GrowScratch fp8_act_scale;  // [M, ceil(K/128)] fp32 activation scales
    bool fp8_block_supported = true;  // latched off if Lt has no algo
};

}  // namespace

struct RuntimeQuantContext::Impl {
    LtCtx ctx;
};

RuntimeQuantContext::RuntimeQuantContext()
    : impl_(std::make_unique<Impl>()) {}

RuntimeQuantContext::~RuntimeQuantContext() {
    reset();
    if (impl_->ctx.handle != nullptr) {
        cublasLtDestroy(impl_->ctx.handle);
        impl_->ctx.handle = nullptr;
    }
}

void RuntimeQuantContext::reset() noexcept {
    auto& ctx = impl_->ctx;
    free_device_memory(ctx.workspace_block);
    ctx.workspace_block = {};
    ctx.workspace = nullptr;
    ctx.workspace_bytes = 0;
    ctx.dequant.reset();
    ctx.int8_act.reset();
    ctx.int8_act_scale.reset();
    ctx.int32_acc.reset();
    ctx.fp8_act.reset();
    ctx.fp8_act_scale.reset();
    ctx.fp8_block_supported = true;
}

ScopedRuntimeQuantContext::ScopedRuntimeQuantContext(
    RuntimeQuantContext& context) noexcept
    : previous_(g_runtime_quant_context) {
    g_runtime_quant_context = &context.impl_->ctx;
}

ScopedRuntimeQuantContext::~ScopedRuntimeQuantContext() {
    g_runtime_quant_context = static_cast<LtCtx*>(previous_);
}

void reserve_runtime_quant_scratch(
    const RuntimeQuantScratchSpec& spec,
    bool seal_after_reserve)
{
    if (spec.empty()) return;

    auto& ctx = LtCtx::instance();
    ctx.ensure_init();

    ctx.dequant.reserve(runtime_quant_dequant_bytes(spec));
    if (spec.has_int8) {
        ctx.int8_act.reserve(
            checked_mul(spec.max_tokens, spec.max_weight_cols,
                        "int8 activation bytes"));
        ctx.int8_act_scale.reserve(
            checked_mul(spec.max_tokens, sizeof(float),
                        "int8 activation scale bytes"));
        ctx.int32_acc.reserve(
            checked_mul(
                checked_mul(spec.max_tokens, spec.max_weight_rows,
                            "int32 accumulator elems"),
                sizeof(std::int32_t),
                "int32 accumulator bytes"));
    }

    if (seal_after_reserve) {
        ctx.dequant.seal("runtime quant dequant scratch");
        ctx.int8_act.seal("runtime quant INT8 activation scratch");
        ctx.int8_act_scale.seal("runtime quant INT8 activation-scale scratch");
        ctx.int32_acc.seal("runtime quant INT8 accumulator scratch");
    }
}

void gemm_grouped_act_x_wt_bf16(
    cublasHandle_t handle,
    const void* const* act_ptrs_host,
    const void* const* W_ptrs_host,
    void* const*       y_ptrs_host,
    const int*         M_array_host,
    int group_count,
    int N,
    int K,
    float beta)
{
    gemm_grouped_bf16_impl(
        handle, act_ptrs_host, W_ptrs_host, y_ptrs_host, M_array_host,
        group_count, N, K, beta);
}

namespace {

// Dequant fallback for sm<89 — materialises a bf16 copy of the FP8
// weight, then runs the classic cuBLAS bf16 GEMM. Costs one extra
// memory pass per layer per fire, so it's strictly slower than plain
// bf16 in steady state — but it's correct, and on H100+ the native
// FP8 path takes over automatically.
// ── Dequantized-weight cache ────────────────────────────────────────────
//
// Block-quantized FP8 weights (DeepSeek `weight_block_size = [128, 128]`)
// have no native cuBLASLt path on this platform, so every GEMM re-expands
// the weight to BF16. That costs 5x the weight bandwidth of the matmul and
// dominates decode. Weights are immutable, so the expansion is cached and
// keyed on the source pointer; `PIE_FP8_DEQUANT_CACHE_GB` caps how much
// device memory the cache may hold (0 disables it).
class DequantWeightCache {
 public:
    // Per device, not per process. Every rank of a TP group runs in this one
    // process with its own current device, and the entries here own device
    // memory: a shared cache lets one rank's insert evict and `cudaFree` a
    // pointer belonging to another rank's device, which poisons the context
    // and surfaces as an illegal access in whatever runs next. The same
    // reasoning already governs `Bf16LtCtx`, `Bf16LtPlanCache` and
    // `DenseGemmTuner` a few hundred lines up.
    static DequantWeightCache& instance() {
        return per_device_singleton<DequantWeightCache>();
    }

    // The expansion depends on the weight AND on how it is read, so the key
    // carries the full recipe: sub-slices of one tensor share a base pointer
    // (DeepSeek-V4's per-group `wo_a` starts group 0 at the base), and a
    // pointer-only key would hand a caller a buffer expanded for a different
    // shape or scale block.
    struct Key {
        const void* weight;
        const void* scale;
        int n;
        int k;
        int group;
        int kind;
        bool operator==(const Key& o) const noexcept {
            return weight == o.weight && scale == o.scale && n == o.n &&
                   k == o.k && group == o.group && kind == o.kind;
        }
    };
    struct KeyHash {
        std::size_t operator()(const Key& key) const noexcept {
            std::size_t h = std::hash<const void*>{}(key.weight);
            auto mix = [&h](std::size_t v) {
                h ^= v + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
            };
            mix(std::hash<const void*>{}(key.scale));
            mix(static_cast<std::size_t>(key.n));
            mix(static_cast<std::size_t>(key.k));
            mix(static_cast<std::size_t>(key.group));
            mix(static_cast<std::size_t>(key.kind));
            return h;
        }
    };

    // Returns a device pointer holding the BF16 expansion of `key`, or
    // nullptr when the cache is disabled / full. `*fresh` is set to true
    // when the caller must still run the dequant kernel to fill it.
    void* get(const Key& key, std::size_t bytes, bool* fresh) {
        *fresh = false;
        if (budget_ == 0) return nullptr;
        std::lock_guard<std::mutex> lock(mu_);
        const auto it = entries_.find(key);
        if (it != entries_.end()) return it->second.block.ptr;
        if (used_ + bytes > budget_) return nullptr;
        Entry e;
        e.block = allocate_device_memory(bytes, 256);
        e.bytes = bytes;
        used_ += bytes;
        void* p = e.block.ptr;
        entries_.emplace(key, std::move(e));
        *fresh = true;
        return p;
    }

    /// Private stream used to materialise entries while the caller's stream
    /// is recording a CUDA graph.
    cudaStream_t fill_stream() {
        std::lock_guard<std::mutex> lock(mu_);
        if (fill_stream_ == nullptr) {
            CUDA_CHECK(cudaStreamCreateWithFlags(&fill_stream_,
                                                 cudaStreamNonBlocking));
        }
        return fill_stream_;
    }

    void clear() {
        std::lock_guard<std::mutex> lock(mu_);
        for (auto& kv : entries_) free_device_memory(kv.second.block);
        entries_.clear();
        used_ = 0;
    }

    // Public only so `per_device_singleton` can build one per device; the
    // instance is still reached exclusively through `instance()`.
    DequantWeightCache() {
        // Self-tuning. The singleton is built on the first FP8 GEMM,
        // i.e. after the KV arena is sized, so "free" here is real headroom.
        // A quarter of it is enough for every block-FP8 weight in the models
        // we serve without competing with activations or graph memory.
        std::size_t free_bytes = 0, total_bytes = 0;
        if (cudaMemGetInfo(&free_bytes, &total_bytes) != cudaSuccess) {
            cudaGetLastError();
            return;
        }
        constexpr std::size_t kCap = std::size_t{16} << 30;
        budget_ = std::min(free_bytes / 4, kCap);
    }
    struct Entry { DeviceMemoryBlock block{}; std::size_t bytes = 0; };
    std::mutex mu_;
    std::unordered_map<Key, Entry, KeyHash> entries_;
    std::size_t used_ = 0;
    std::size_t budget_ = 0;
    cudaStream_t fill_stream_ = nullptr;
};

void gemm_fp8_dequant_then_bf16_fallback(
    cublasHandle_t cublas_handle,
    const void* act, const void* w_fp8, const void* w_scale_fp32_dev,
    QuantMeta::Kind scale_kind,
    void* y,
    int M, int N, int K,
    float beta,
    cudaStream_t stream,
    int group_size = 0)
{
    auto& ctx = LtCtx::instance();
    const std::size_t weight_elems =
        static_cast<std::size_t>(N) * static_cast<std::size_t>(K);

    bool needs_fill = true;
    void* bf16_w = DequantWeightCache::instance().get(
        DequantWeightCache::Key{w_fp8, w_scale_fp32_dev, N, K, group_size,
                                static_cast<int>(scale_kind)},
        weight_elems * 2, &needs_fill);
    if (bf16_w != nullptr && !needs_fill) {
        gemm_bf16_impl(cublas_handle, act, bf16_w, y, M, N, K, beta);
        return;
    }
    // A fresh cache entry must be filled for real, right now. Under CUDA-graph
    // capture the caller's stream only *records* work, so a dequant enqueued
    // there leaves the buffer unwritten until the first replay while the entry
    // already reads as filled — every non-graph caller would then multiply by
    // garbage. Fill on a private stream instead (capture is Relaxed, so
    // off-stream work and a sync on it are legal) and let the graph record
    // just the matmul, which is the whole point of caching.
    cudaStream_t fill_stream = stream;
    if (bf16_w != nullptr) {
        cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
        if (cudaStreamIsCapturing(stream, &capture) != cudaSuccess) {
            cudaGetLastError();
            capture = cudaStreamCaptureStatusNone;
        }
        if (capture != cudaStreamCaptureStatusNone) {
            fill_stream = DequantWeightCache::instance().fill_stream();
        }
    } else {
        bf16_w = ctx.dequant.ensure(weight_elems * 2);
    }

    if (scale_kind == QuantMeta::Kind::PerGroup && group_size > 0) {
        kernels::launch_dequant_fp8_e4m3_to_bf16_per_group(
            static_cast<const std::uint8_t*>(w_fp8),
            bf16_w,
            static_cast<const float*>(w_scale_fp32_dev),
            N, K, group_size, fill_stream);
        CUDA_CHECK(cudaGetLastError());
    } else if (scale_kind == QuantMeta::Kind::PerChannel) {
        kernels::launch_dequant_fp8_e4m3_to_bf16_per_channel(
            static_cast<const std::uint8_t*>(w_fp8),
            bf16_w,
            static_cast<const float*>(w_scale_fp32_dev),
            N, K, fill_stream);
        CUDA_CHECK(cudaGetLastError());
    } else {
        float scale = 0.f;
        CUDA_CHECK(cudaMemcpyAsync(&scale, w_scale_fp32_dev, sizeof(float),
                                   cudaMemcpyDeviceToHost, fill_stream));
        CUDA_CHECK(cudaStreamSynchronize(fill_stream));
        kernels::launch_dequant_fp8_e4m3_to_bf16(
            static_cast<const std::uint8_t*>(w_fp8),
            bf16_w, scale, weight_elems, fill_stream);
        CUDA_CHECK(cudaGetLastError());
    }
    if (fill_stream != stream) CUDA_CHECK(cudaStreamSynchronize(fill_stream));
    gemm_bf16_impl(cublas_handle, act, bf16_w, y, M, N, K, beta);
}

void gemm_int8_dequant_then_bf16_fallback(
    cublasHandle_t cublas_handle,
    const void* act,
    const void* w_int8,
    const float* w_scale_inv,
    void* y,
    int M, int N, int K,
    float beta,
    cudaStream_t stream)
{
    auto& ctx = LtCtx::instance();
    const std::size_t weight_elems =
        static_cast<std::size_t>(N) * static_cast<std::size_t>(K);
    void* bf16_w = ctx.dequant.ensure(weight_elems * 2);
    kernels::launch_dequant_int8_to_bf16_per_channel(
        static_cast<const std::int8_t*>(w_int8),
        bf16_w,
        w_scale_inv,
        N,
        K,
        stream);
    CUDA_CHECK(cudaGetLastError());
    gemm_bf16_impl(cublas_handle, act, bf16_w, y, M, N, K, beta);
}

// ── DeepSeek-style W8A8 block FP8 GEMM ──────────────────────────────────
//
// The checkpoint stores `weight [N, K]` as FP8 E4M3 with one FP32 scale per
// 128x128 weight tile (`quantization_config.weight_block_size = [128, 128]`).
// The historical path dequantized the *entire* weight to BF16 on every call,
// which costs 5x the weight bandwidth of the matmul itself and dominates
// decode. Blackwell cuBLASLt can consume the block scales natively, so we
// quantize the activation to FP8 with 1x128 scales along K and issue a true
// FP8 matmul.
//
// Layout (all cuBLASLt operands are column-major):
//   A = weight: row-major [N, K] == col-major [K, N] ld=K, OP_T.
//       BLK128x128 scales are col-major [ceil(K/128), ceil(N/128)], i.e.
//       index k_blk + n_blk*ceil(K/128) -- bit-identical to the
//       row-major [ceil(N/128), ceil(K/128)] tensor the checkpoint ships.
//   B = act:    row-major [M, K] == col-major [K, M] ld=K, OP_N.
//       VEC128 scales are col-major [ceil(K/128), M], index
//       k_blk + m*ceil(K/128) -- identical to our row-major [M, K/128].
//   D = out:    col-major [N, M] ld=N == row-major [M, N].
//
// Both scale conventions are multiplicative (`value = fp8 * scale`), which
// is what the checkpoint's `weight_scale_inv` already stores.
bool gemm_fp8_blockwise_w8a8_impl(
    const void* act, const void* w_fp8, const void* w_scale_fp32_dev,
    void* y,
    int M, int N, int K,
    float beta,
    cudaStream_t stream,
    int group_size)
{
    auto& ctx = LtCtx::instance();
    auto reject = [](const char*) { return false; };
    if (!ctx.fp8_block_supported) return reject("latched-off");
    if (group_size != 128) return reject("group_size");
    // Block scales assume a whole number of 128-wide groups along K; the
    // FP8 tensor-core path additionally needs 16-byte-aligned leading dims.
    if (K % 128 != 0 || N % 16 != 0) return reject("shape");

    ctx.ensure_init();
    const int k_blocks = K / 128;
    void* act_fp8 = ctx.fp8_act.ensure(
        static_cast<std::size_t>(M) * static_cast<std::size_t>(K));
    void* act_scale = ctx.fp8_act_scale.ensure(
        static_cast<std::size_t>(M) * static_cast<std::size_t>(k_blocks) *
        sizeof(float));

    kernels::quantize_bf16_to_fp8_e4m3_per_token_group(
        act, static_cast<std::uint8_t*>(act_fp8),
        static_cast<float*>(act_scale), M, K, 128, stream);

    LtMatmulDesc desc(CUBLAS_COMPUTE_32F, CUDA_R_32F);
    cublasOperation_t op_t = CUBLAS_OP_T;
    cublasOperation_t op_n = CUBLAS_OP_N;
    desc.set(CUBLASLT_MATMUL_DESC_TRANSA, op_t);
    desc.set(CUBLASLT_MATMUL_DESC_TRANSB, op_n);
    std::int32_t a_mode = CUBLASLT_MATMUL_MATRIX_SCALE_BLK128x128_32F;
    std::int32_t b_mode = CUBLASLT_MATMUL_MATRIX_SCALE_VEC128_32F;
    desc.set(CUBLASLT_MATMUL_DESC_A_SCALE_MODE, a_mode);
    desc.set(CUBLASLT_MATMUL_DESC_B_SCALE_MODE, b_mode);
    desc.set(CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, w_scale_fp32_dev);
    desc.set(CUBLASLT_MATMUL_DESC_B_SCALE_POINTER,
             static_cast<const void*>(act_scale));

    LtMatrixLayout a_layout(CUDA_R_8F_E4M3, /*rows=*/K, /*cols=*/N, /*ld=*/K);
    LtMatrixLayout b_layout(CUDA_R_8F_E4M3, /*rows=*/K, /*cols=*/M, /*ld=*/K);
    LtMatrixLayout d_layout(CUDA_R_16BF,    /*rows=*/N, /*cols=*/M, /*ld=*/N);

    LtMatmulPref pref;
    pref.set(CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, ctx.workspace_bytes);

    cublasLtMatmulHeuristicResult_t heur = {};
    int returned = 0;
    const cublasStatus_t hs = cublasLtMatmulAlgoGetHeuristic(
        ctx.handle, desc.d, a_layout.d, b_layout.d,
        d_layout.d, d_layout.d, pref.d, /*requested=*/1, &heur, &returned);
    if (hs != CUBLAS_STATUS_SUCCESS || returned == 0) {
        ctx.fp8_block_supported = false;
        return reject("no-algo");
    }
    const float alpha = 1.f;
    LT_CHECK(cublasLtMatmul(
        ctx.handle, desc.d, &alpha,
        /*A=*/w_fp8, a_layout.d,
        /*B=*/act_fp8, b_layout.d,
        &beta,
        /*C=*/y, d_layout.d,
        /*D=*/y, d_layout.d,
        &heur.algo, ctx.workspace, ctx.workspace_bytes, stream));
    return true;
}

void gemm_fp8_e4m3_w_bf16_act_impl(
    cublasHandle_t cublas_handle,
    const void* act, const void* w_fp8, const void* w_scale_fp32_dev,
    QuantMeta::Kind scale_kind,
    void* y,
    int M, int N, int K,
    float beta,
    cudaStream_t stream,
    int group_size = 0)
{
    if (!w_scale_fp32_dev) {
        throw std::runtime_error(
            "gemm_act_x_w[FP8_E4M3]: scale pointer is null — "
            "weight_scale_inv must be attached to the materialized WeightStore "
            "as an FP32 device tensor before calling FP8 GEMM");
    }
    auto& ctx = LtCtx::instance();
    ctx.ensure_init();

    if (scale_kind == QuantMeta::Kind::PerGroup &&
        gemm_fp8_blockwise_w8a8_impl(act, w_fp8, w_scale_fp32_dev, y,
                                     M, N, K, beta, stream, group_size)) {
        return;
    }

    if (!ctx.fp8_native_supported ||
        scale_kind == QuantMeta::Kind::PerChannel ||
        scale_kind == QuantMeta::Kind::PerGroup) {
        gemm_fp8_dequant_then_bf16_fallback(
            cublas_handle, act, w_fp8, w_scale_fp32_dev, scale_kind, y,
            M, N, K, beta, stream, group_size);
        return;
    }

    // Same row-major-as-col-major reinterpretation as the bf16 path.
    // We compute D'[N,M] = op(A=W) * op(B=act) where
    //   A col-major view of row-major W[N,K]   → [K,N] with ld=K, OP_T → [N,K]
    //   B col-major view of row-major act[M,K] → [K,M] with ld=K, OP_N → [K,M]
    //   D col-major view of row-major y[M,N]   → [N,M] with ld=N
    // → cuBLASLt sees m=N, n=M, k=K.

    LtMatmulDesc desc(CUBLAS_COMPUTE_32F, CUDA_R_32F);
    cublasOperation_t op_t = CUBLAS_OP_T;
    cublasOperation_t op_n = CUBLAS_OP_N;
    desc.set(CUBLASLT_MATMUL_DESC_TRANSA, op_t);
    desc.set(CUBLASLT_MATMUL_DESC_TRANSB, op_n);
    std::int8_t fast_accum = 1;
    desc.set(CUBLASLT_MATMUL_DESC_FAST_ACCUM, fast_accum);
    // FP8-weight scale pointer: cuBLASLt multiplies A by *scale before the
    // matmul. mistral3 stores `weight_scale_inv` such that bf16 = fp8 * scale,
    // which matches this contract exactly.
    desc.set(CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, w_scale_fp32_dev);

    LtMatrixLayout a_layout(CUDA_R_8F_E4M3, /*rows=*/K, /*cols=*/N, /*ld=*/K);
    LtMatrixLayout b_layout(CUDA_R_16BF,    /*rows=*/K, /*cols=*/M, /*ld=*/K);
    LtMatrixLayout d_layout(CUDA_R_16BF,    /*rows=*/N, /*cols=*/M, /*ld=*/N);

    LtMatmulPref pref;
    pref.set(CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, ctx.workspace_bytes);

    cublasLtMatmulHeuristicResult_t heur = {};
    int returned = 0;
    LT_CHECK(cublasLtMatmulAlgoGetHeuristic(
        ctx.handle, desc.d, a_layout.d, b_layout.d,
        d_layout.d, d_layout.d, pref.d, /*requested=*/1,
        &heur, &returned));
    if (returned == 0) {
        // Latched fallback: cache the negative result so subsequent FP8
        // calls skip the heuristic round-trip.
        ctx.fp8_native_supported = false;
        gemm_fp8_dequant_then_bf16_fallback(
            cublas_handle, act, w_fp8, w_scale_fp32_dev, scale_kind, y,
            M, N, K, beta, stream);
        return;
    }

    const float alpha = 1.f;
    LT_CHECK(cublasLtMatmul(
        ctx.handle, desc.d, &alpha,
        /*A=*/w_fp8, a_layout.d,
        /*B=*/act,   b_layout.d,
        &beta,
        /*C=*/y,     d_layout.d,
        /*D=*/y,     d_layout.d,
        &heur.algo, ctx.workspace, ctx.workspace_bytes, stream));
}

// W8A8 INT8 GEMM: bf16 activation → int8 (per-token), int8 weight (per-
// channel scale already attached), cublasGemmEx INT8 → int32 accumulator,
// dequant to bf16 via per-row × per-col scale product.
//
// Sm80 has native INT8 tensor-core GEMM (CUDA_R_8I + CUBLAS_COMPUTE_32I)
// at ~2× bf16 throughput, so this is the real Ampere quant perf win
// (FP8 on sm80 is bf16-equivalent via dequant fallback).
void gemm_int8_w_bf16_act_impl(
    cublasHandle_t cublas_handle,
    const void* act_bf16,        // [M, K] bf16
    const void* w_int8,          // [N, K] int8 (HF Linear layout)
    const float* w_scale_inv,    // [N] fp32 (per-channel)
    void* y_bf16,                // [M, N] bf16
    int M, int N, int K,
    float beta,
    cudaStream_t stream)
{
    auto& ctx = LtCtx::instance();
    ctx.ensure_init();

    if ((M % 4) != 0 || (N % 4) != 0 || (K % 4) != 0) {
        gemm_int8_dequant_then_bf16_fallback(
            cublas_handle, act_bf16, w_int8, w_scale_inv,
            y_bf16, M, N, K, beta, stream);
        return;
    }

    // Stage 1: per-token activation quant.
    const std::size_t act_int8_bytes =
        static_cast<std::size_t>(M) * static_cast<std::size_t>(K);
    const std::size_t act_scale_bytes =
        static_cast<std::size_t>(M) * sizeof(float);
    const std::size_t acc_bytes =
        static_cast<std::size_t>(M) * static_cast<std::size_t>(N) * sizeof(std::int32_t);
    auto* act_int8 = static_cast<std::int8_t*>(
        ctx.int8_act.ensure(act_int8_bytes));
    auto* act_scale = static_cast<float*>(
        ctx.int8_act_scale.ensure(act_scale_bytes));
    auto* acc_int32 = static_cast<std::int32_t*>(
        ctx.int32_acc.ensure(acc_bytes));

    kernels::quantize_bf16_to_int8_per_token(
        act_bf16, act_int8, act_scale, M, K, stream);

    // Stage 2: cublasGemmEx INT8.
    // Same row-major-as-col-major reinterpretation as the bf16 path.
    //   y_int32[m, n] = sum_k act_int8[m, k] * w_int8[n, k]
    // Col-major view:
    //   A = w_int8 [K, N] with ld=K, OP_T → [N, K]
    //   B = act_int8 [K, M] with ld=K, OP_N → [K, M]
    //   D = acc [N, M] with ld=N (col-major) = [M, N] row-major.
    const std::int32_t alpha = 1, c_beta = 0;
    const auto status = cublasGemmEx(
        cublas_handle,
        /*transa=*/CUBLAS_OP_T, /*transb=*/CUBLAS_OP_N,
        /*m=*/N, /*n=*/M, /*k=*/K,
        &alpha,
        /*A=*/w_int8,   CUDA_R_8I,  /*lda=*/K,
        /*B=*/act_int8, CUDA_R_8I,  /*ldb=*/K,
        &c_beta,
        /*C=*/acc_int32, CUDA_R_32I, /*ldc=*/N,
        CUBLAS_COMPUTE_32I,
        CUBLAS_GEMM_DEFAULT);
    if (status != CUBLAS_STATUS_SUCCESS) {
        gemm_int8_dequant_then_bf16_fallback(
            cublas_handle, act_bf16, w_int8, w_scale_inv,
            y_bf16, M, N, K, beta, stream);
        return;
    }

    // Stage 3: dequant int32 → bf16 with per-row × per-col scales.
    //   y[m, n] = acc[m, n] * act_scale_inv[m] * w_scale_inv[n]   (beta=0)
    //   y[m, n] += acc[m, n] * act_scale_inv[m] * w_scale_inv[n]  (beta=1)
    //
    // For beta=1 (residual-add fusion), dequant into a scratch then
    // residual-add — same trick as marlin. For beta=0 dequant straight
    // into y_bf16.
    if (beta == 0.f) {
        kernels::dequant_int32_w8a8_to_bf16(
            acc_int32, act_scale, w_scale_inv, y_bf16, M, N, stream);
    } else {
        const std::size_t mn_bytes =
            static_cast<std::size_t>(M) * static_cast<std::size_t>(N) * 2;
        void* dq_dst = ctx.dequant.ensure(mn_bytes);
        kernels::dequant_int32_w8a8_to_bf16(
            acc_int32, act_scale, w_scale_inv, dq_dst, M, N, stream);
        kernels::launch_residual_add_bf16(
            y_bf16, dq_dst,
            static_cast<std::size_t>(M) * static_cast<std::size_t>(N),
            stream);
    }
}

}  // namespace

void gemm_act_x_w(
    cublasHandle_t handle,
    const void* act,
    WeightView w,
    void* y,
    int M, int N, int K,
    float beta,
    DType act_dtype,
    DType y_dtype)
{
    if (act_dtype == DType::BF16 && w.dtype == DType::BF16 &&
        y_dtype == DType::BF16) {
        gemm_bf16_impl(handle, act, w.data, y, M, N, K, beta);
        return;
    }
    if (act_dtype == DType::BF16 && w.dtype == DType::BF16 &&
        y_dtype == DType::FP32) {
        gemm_bf16_to_fp32_impl(handle, act, w.data, y, M, N, K, beta);
        return;
    }
    if (act_dtype == DType::BF16 && w.dtype == DType::FP8_E4M3 &&
        y_dtype == DType::BF16) {
        // Pull the cuda stream out of the cublas classic handle so the
        // FP8 path runs on the same stream as everything else this layer
        // does. cuBLAS exposes the bound stream via cublasGetStream.
        cudaStream_t stream = nullptr;
        cublasGetStream(handle, &stream);
        if (w.scale_dtype != DType::FP32) {
            throw std::runtime_error(
                "gemm_act_x_w[FP8_E4M3]: scale must be FP32 (got " +
                std::string(dtype_name(w.scale_dtype)) + ")");
        }
        validate_quant_weight_view("gemm_act_x_w[FP8_E4M3]", w, N, K);
        gemm_fp8_e4m3_w_bf16_act_impl(handle, act, w.data, w.scale_data,
                                      w.quant_kind,
                                      y, M, N, K, beta, stream,
                                      w.group_size);
        return;
    }
    if (act_dtype == DType::BF16 && w.dtype == DType::INT8 &&
        y_dtype == DType::BF16) {
        cudaStream_t stream = nullptr;
        cublasGetStream(handle, &stream);
        if (w.scale_dtype != DType::FP32) {
            throw std::runtime_error(
                "gemm_act_x_w[INT8 W8A8]: scale must be FP32 (got " +
                std::string(dtype_name(w.scale_dtype)) + ")");
        }
        if (w.quant_kind != QuantMeta::Kind::PerChannel) {
            throw std::runtime_error(
                "gemm_act_x_w[INT8 W8A8]: only PerChannel weight scale "
                "supported (per-tensor / per-group not yet wired)");
        }
        validate_quant_weight_view("gemm_act_x_w[INT8 W8A8]", w, N, K);
        gemm_int8_w_bf16_act_impl(
            handle, act, w.data,
            static_cast<const float*>(w.scale_data),
            y, M, N, K, beta, stream);
        return;
    }
    if (act_dtype == DType::BF16 && w.dtype == DType::INT4_PACKED &&
        y_dtype == DType::BF16) {
#ifdef PIE_CUDA_HAS_MARLIN
        // Marlin W4A16 GEMM. Per-group bf16 scales, no zero-points (GPTQ
        // symmetric), no act-order. The dispatcher relies on the loader
        // having pre-repacked the weight into marlin's tile layout (via
        // `gptq_marlin_repack`) and stored the per-group scales as the
        // QuantMeta side-tensor.
        cudaStream_t stream = nullptr;
        cublasGetStream(handle, &stream);
        // marlin always overwrites C. For the beta=1 residual-add
        // pattern (o_proj / down_proj fusion), we redirect marlin into
        // a scratch [M, N] bf16 buffer then run the residual-add
        // kernel. Two passes cost ~one extra read/write of MN bf16,
        // which is negligible vs. the matmul.
        const std::size_t mn_bytes =
            static_cast<std::size_t>(M) * static_cast<std::size_t>(N) * 2;
        void* dst = (beta == 0.f) ? y : marlin_residual_scratch_(mn_bytes);
        marlin::launch_gptq_gemm_w4a16_bf16(
            act, w.data, w.scale_data, w.zero_point_data, dst,
            marlin_workspace_(),
            M, N, K, w.group_size,
            /*use_fp32_reduce=*/false,
            stream);
        if (beta != 0.f) {
            kernels::launch_residual_add_bf16(
                y, dst,
                static_cast<std::size_t>(M) * static_cast<std::size_t>(N),
                stream);
        }
        return;
#else
        throw std::runtime_error(
            "gemm_act_x_w[INT4_PACKED]: GPTQ/AWQ W4A16 needs the vendored "
            "marlin kernels, which are not built by default because they "
            "dominate CUDA build time. Reconfigure with "
            "-DPIE_CUDA_BUILD_MARLIN=ON (or PIE_CUDA_BUILD_MARLIN=1).");
#endif
    }
    if (act_dtype == DType::BF16 && w.dtype == DType::MXFP4_PACKED &&
        y_dtype == DType::BF16) {
        cudaStream_t stream = nullptr;
        cublasGetStream(handle, &stream);
        if (w.scale_dtype != DType::UINT8) {
            throw std::runtime_error(
                "gemm_act_x_w[MXFP4]: scale must be raw E8M0 bytes (got " +
                std::string(dtype_name(w.scale_dtype)) + ")");
        }
        if (w.quant_kind != QuantMeta::Kind::PerGroup || w.group_size != 32) {
            throw std::runtime_error(
                "gemm_act_x_w[MXFP4]: expected per-group scales with "
                "group_size=32");
        }
        validate_quant_weight_view("gemm_act_x_w[MXFP4]", w, N, K);

        // Dequant MXFP4 → bf16 in a scratch buffer, then bf16 GEMM.
        // Reuse the LtCtx dequant scratch (auto-grows monotonically). Cost is
        // one extra weight read + write per call, acceptable for prefill /
        // small-batch decode.
        auto& ctx = LtCtx::instance();
        ctx.ensure_init();
        const std::size_t weight_bf16_bytes =
            static_cast<std::size_t>(N) * static_cast<std::size_t>(K) * 2;
        void* bf16_w = ctx.dequant.ensure(weight_bf16_bytes);
        kernels::launch_dequant_mxfp4_to_bf16(
            static_cast<const std::uint8_t*>(w.data),
            static_cast<const std::uint8_t*>(w.scale_data),
            bf16_w, N, K, stream);
        CUDA_CHECK(cudaGetLastError());
        gemm_bf16_impl(handle, act, bf16_w, y, M, N, K, beta);
        return;
    }
    unsupported("gemm_act_x_w", act_dtype, w.dtype, y_dtype);
}

bool lm_head_argmax_supported(WeightView w) {
    return w.data != nullptr && w.dtype == DType::BF16 &&
        w.scale_data == nullptr;
}

std::size_t lm_head_argmax_slab_bytes(int M, int N, int chunk) {
    if (M <= 0 || N <= 0 || chunk <= 0) return 0;
    const std::size_t width = static_cast<std::size_t>(std::min(chunk, N));
    return static_cast<std::size_t>(M) * width * sizeof(__nv_bfloat16);
}

bool lm_head_argmax_chunked(
    cublasHandle_t handle,
    const void* act,
    WeightView w,
    std::int32_t* token_ids,
    void* slab,
    float* acc_val,
    std::int32_t* acc_idx,
    int M, int N, int K,
    int chunk)
{
    if (!lm_head_argmax_supported(w)) return false;
    if (M <= 0 || N <= 0 || K <= 0 || chunk <= 0) return false;
    if (act == nullptr || slab == nullptr ||
        token_ids == nullptr || acc_val == nullptr || acc_idx == nullptr) {
        return false;
    }

    cudaStream_t stream = nullptr;
    cublasGetStream(handle, &stream);
    const auto* weight_rows = static_cast<const std::uint8_t*>(w.data);

    for (int base = 0; base < N; base += chunk) {
        const int width = std::min(chunk, N - base);
        // A slab is rows [base, base+width) of W, which are contiguous, and
        // lands in `slab` as a tightly packed [M, width] -- so the slab's row
        // stride is `width`, not `chunk`, on the ragged final iteration.
        gemm_act_x_w(
            handle, act,
            WeightView::raw(
                weight_rows + static_cast<std::size_t>(base) *
                                  static_cast<std::size_t>(K) * sizeof(__nv_bfloat16),
                DType::BF16),
            slab, M, width, K);
        kernels::launch_argmax_accumulate_bf16(
            slab, M, width, width, base, acc_val, acc_idx,
            /*init=*/base == 0, stream);
    }
    kernels::launch_argmax_finalize_bf16(acc_val, acc_idx, token_ids, M, stream);
    return true;
}

void gemm_act_x_wt_bf16_out_fp32(
    cublasHandle_t handle,
    const void* act,
    const void* W,
    float* y,
    int M,
    int N,
    int K)
{
    gemm_bf16_out_fp32_impl(handle, act, W, y, M, N, K);
}

void gemm_act_x_wt_bf16_cublas(
    cublasHandle_t handle,
    const void* act, const void* W, void* y,
    int M, int N, int K,
    float beta)
{
    // The reason callers pin this entry is that cuBLASLt's heuristic loses on
    // their *batched* shapes -- the note beside each of them names an N in the
    // hundreds. None of that reasoning reaches M=1: a single activation row
    // has no reuse for any tiled GEMM to exploit, Lt or classic, and the
    // warp-per-row GEMV roughly doubles the bandwidth either of them reach
    // (see `launch_gemv_bf16`). Without this, enabling gemma-4's fused
    // gate/up bank moved its decode MLP onto a kernel tiled for an M it does
    // not have -- 1.28 ms/token against 0.32 ms of weights.
    cudaStream_t gemv_stream = nullptr;
    if (M == 1 && beta == 0.f &&
        cublas_stream(handle, gemv_stream) &&
        kernels::launch_gemv_bf16(W, act, nullptr, y, N, K, gemv_stream)) {
        return;
    }
    gemm_bf16_cublas_impl(handle, act, W, y, M, N, K, beta);
}

void gemm_act_x_wt_bias_bf16(
    cublasHandle_t handle,
    const void* act, const void* W, const void* bias, void* y,
    int M, int N, int K,
    cudaStream_t stream,
    float beta)
{
    // Ask the tuner the same question `gemm_bf16_impl` would, rather than
    // peeking at what it has already decided: a shape is seen for the first
    // time *during graph capture*, and a peek would miss then and bake the
    // unfused pair into the graph forever.
    if (bias != nullptr && M == 1) {
        cudaStream_t s = nullptr;
        cudaStreamCaptureStatus capturing = cudaStreamCaptureStatusNone;
        const Bf16LtPlan* plan = nullptr;
        DenseTactic tactic{};
        // `run_dense_tactic` declines any tactic that cannot absorb a bias,
        // so a shape where cuBLAS beats the GEMV is never forced onto the
        // GEMV just to save a launch -- it falls through below.
        if (cublas_stream(handle, s) &&
            cudaStreamIsCapturing(s, &capturing) == cudaSuccess &&
            dense_tactic_for(handle, W, M, N, K, beta, capturing,
                             &plan, &tactic) &&
            run_dense_tactic(handle, tactic, plan, act, W, y, M, N, K,
                             beta, nullptr, 0, bias)) {
            return;
        }
        cudaGetLastError();
    }
    gemm_bf16_impl(handle, act, W, y, M, N, K, beta);
    if (bias != nullptr) {
        kernels::launch_add_bias_bf16(y, bias, M, N, stream);
    }
}

void gemm_batched_act_x_w(
    cublasHandle_t handle,
    const void* const* act_ptrs_dev,
    const void* const* w_ptrs_dev,
    void* const*       y_ptrs_dev,
    int M, int N, int K,
    int batch_count,
    float beta,
    DType act_dtype,
    DType w_dtype,
    DType y_dtype)
{
    if (act_dtype == DType::BF16 && w_dtype == DType::BF16 &&
        y_dtype == DType::BF16) {
        gemm_batched_bf16_impl(handle, act_ptrs_dev, w_ptrs_dev, y_ptrs_dev,
                               M, N, K, batch_count, beta);
        return;
    }
    unsupported("gemm_batched_act_x_w", act_dtype, w_dtype, y_dtype);
}


void mla_absorb_q_to_latent_bf16(
    cublasHandle_t handle,
    const void* q_nope, const void* kv_b_proj, void* q_latent,
    int tokens, int heads, int qk_nope_dim, int v_head_dim, int kv_lora_rank)
{
    if (tokens <= 0 || heads <= 0) return;
    const float alpha = 1.f, beta = 0.f;
    // Row-major C[T, kv_lora] = A[T, nope] @ B[nope, kv_lora] per head, written
    // column-major as C^T = B^T @ A^T.
    const auto status = cublasGemmStridedBatchedEx(
        handle, CUBLAS_OP_N, CUBLAS_OP_N,
        /*m=*/kv_lora_rank, /*n=*/tokens, /*k=*/qk_nope_dim,
        &alpha,
        /*A=*/kv_b_proj, CUDA_R_16BF, /*lda=*/kv_lora_rank,
        /*strideA=*/static_cast<long long>(qk_nope_dim + v_head_dim) * kv_lora_rank,
        /*B=*/q_nope, CUDA_R_16BF, /*ldb=*/heads * qk_nope_dim,
        /*strideB=*/qk_nope_dim,
        &beta,
        /*C=*/q_latent, CUDA_R_16BF, /*ldc=*/heads * kv_lora_rank,
        /*strideC=*/kv_lora_rank,
        /*batchCount=*/heads,
        bf16_compute_type(), CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    check(status, "mla_absorb_q_to_latent_bf16");
}

void mla_absorb_latent_to_v_bf16(
    cublasHandle_t handle,
    const void* attn_latent, const void* kv_b_proj, void* attn_v,
    int tokens, int heads, int qk_nope_dim, int v_head_dim, int kv_lora_rank)
{
    if (tokens <= 0 || heads <= 0) return;
    const float alpha = 1.f, beta = 0.f;
    const auto* wv = static_cast<const __nv_bfloat16*>(kv_b_proj) +
                 static_cast<long long>(qk_nope_dim) * kv_lora_rank;
    // Row-major C[T, v_dim] = A[T, kv_lora] @ W[v_dim, kv_lora]^T per head.
    const auto status = cublasGemmStridedBatchedEx(
        handle, CUBLAS_OP_T, CUBLAS_OP_N,
        /*m=*/v_head_dim, /*n=*/tokens, /*k=*/kv_lora_rank,
        &alpha,
        /*A=*/wv, CUDA_R_16BF, /*lda=*/kv_lora_rank,
        /*strideA=*/static_cast<long long>(qk_nope_dim + v_head_dim) * kv_lora_rank,
        /*B=*/attn_latent, CUDA_R_16BF, /*ldb=*/heads * kv_lora_rank,
        /*strideB=*/kv_lora_rank,
        &beta,
        /*C=*/attn_v, CUDA_R_16BF, /*ldc=*/heads * v_head_dim,
        /*strideC=*/v_head_dim,
        /*batchCount=*/heads,
        bf16_compute_type(), CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    check(status, "mla_absorb_latent_to_v_bf16");
}

}  // namespace pie_cuda_driver::ops
