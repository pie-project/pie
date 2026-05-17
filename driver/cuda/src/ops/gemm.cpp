#include "ops/gemm.hpp"

#include <cublasLt.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>

#include "cuda_check.hpp"
#include "kernels/dequant_fp8.hpp"
#include "kernels/quant_bf16_to_fp8.hpp"
#include "kernels/residual_add.hpp"

#ifdef PIE_CUDA_HAS_MARLIN
#include "marlin_wrapper.hpp"
#endif

namespace pie_cuda_driver::ops {

namespace {

void check(cublasStatus_t s, const char* expr) {
    if (s != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error(std::string("cuBLAS error (") +
                                 std::to_string(static_cast<int>(s)) + "): " + expr);
    }
}

}  // namespace

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

namespace {

void gemm_bf16_impl(
    cublasHandle_t handle,
    const void* act, const void* W, void* y,
    int M, int N, int K,
    float beta)
{
    // P1.2 attempt routed this through cublasLtMatmul; the Lt heuristic
    // returned the same `gemvx` algo cuBLAS picks on its own, so the
    // change was a wash. We keep the Lt scaffold (defined further down)
    // for future epilogue-fusion work but route bf16 gemms back through
    // the simpler GemmEx path here.
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
              CUBLAS_COMPUTE_32F,
              CUBLAS_GEMM_DEFAULT);
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error(
            "cuBLAS error (" + std::to_string(static_cast<int>(status)) +
            "): cublasGemmEx[bf16] M=" + std::to_string(M) +
            " N=" + std::to_string(N) + " K=" + std::to_string(K));
    }
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
    // Same row-major-as-col-major reinterpretation as the unbatched
    // wrapper above: A=W (op_T), B=act (op_N), C=y (col-major NxM).
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
              CUBLAS_COMPUTE_32F,
              CUBLAS_GEMM_DEFAULT);
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error(
            "cuBLAS error (" + std::to_string(static_cast<int>(status)) +
            "): cublasGemmBatchedEx[bf16] M=" + std::to_string(M) +
            " N=" + std::to_string(N) + " K=" + std::to_string(K) +
            " batch=" + std::to_string(batch_count));
    }
}

[[noreturn]] void unsupported(const char* api,
                              DType act_dtype, DType w_dtype, DType y_dtype) {
    throw std::runtime_error(
        std::string("ops::") + api + ": unsupported dtype combo (act=" +
        dtype_name(act_dtype) + ", w=" + dtype_name(w_dtype) +
        ", y=" + dtype_name(y_dtype) + ")");
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
// Per-process marlin workspace. Marlin's split-K reduce uses one int32
// per SM as a barrier counter; we allocate generously (16 KiB) to cover
// every realistic SM count without per-call allocation. Lazy-init on
// first INT4_PACKED dispatch.
void* marlin_workspace_() {
    static void* ws = nullptr;
    static const std::size_t ws_bytes = 16 * 1024;
    if (!ws) {
        if (cudaMalloc(&ws, ws_bytes) != cudaSuccess) {
            throw std::runtime_error("marlin: cudaMalloc workspace failed");
        }
        cudaMemset(ws, 0, ws_bytes);
    }
    return ws;
}

// Per-process bf16 residual scratch — used when the INT4 dispatcher is
// called with beta=1 (the residual-add fusion the bf16/fp8 paths handle
// natively via cuBLAS's beta param). Marlin overwrites C, so we run it
// into a scratch and add into y in a second pass. Grows monotonically.
void* marlin_residual_scratch_(std::size_t bytes) {
    static void* buf = nullptr;
    static std::size_t buf_bytes = 0;
    if (bytes <= buf_bytes) return buf;
    if (buf) cudaFree(buf);
    if (cudaMalloc(&buf, bytes) != cudaSuccess) {
        throw std::runtime_error(
            "marlin: cudaMalloc residual scratch failed (" +
            std::to_string(bytes) + " bytes)");
    }
    buf_bytes = bytes;
    return buf;
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
    std::size_t      workspace_bytes = 0;
    int              compute_capability_major = 0;  // 0 = unqueried
    bool             fp8_native_supported = false;

    static LtCtx& instance() {
        static LtCtx ctx;
        return ctx;
    }

    void ensure_init(std::size_t ws_bytes = 32 * 1024 * 1024) {
        if (!handle) LT_CHECK(cublasLtCreate(&handle));
        if (!workspace) {
            CUDA_CHECK(cudaMalloc(&workspace, ws_bytes));
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
    // a pointer valid until the next `ensure(bigger_size)` on the same
    // buffer. Cleared at process exit (LtCtx is a static singleton).
    struct GrowScratch {
        void*       p = nullptr;
        std::size_t bytes = 0;
        void* ensure(std::size_t want) {
            if (want <= bytes) return p;
            if (p) CUDA_CHECK(cudaFree(p));
            CUDA_CHECK(cudaMalloc(&p, want));
            bytes = want;
            return p;
        }
    };
    GrowScratch dequant;        // sm<89 FP8 → bf16 weight scratch
    GrowScratch int8_act;       // [M, K] int8 quantised activation
    GrowScratch int8_act_scale; // [M] fp32 act_scale_inv
    GrowScratch int32_acc;      // [M, N] int32 W8A8 accumulator
};

// Dequant fallback for sm<89 — materialises a bf16 copy of the FP8
// weight, then runs the classic cuBLAS bf16 GEMM. Costs one extra
// memory pass per layer per fire, so it's strictly slower than plain
// bf16 in steady state — but it's correct, and on H100+ the native
// FP8 path takes over automatically.
void gemm_fp8_dequant_then_bf16_fallback(
    cublasHandle_t cublas_handle,
    const void* act, const void* w_fp8, const void* w_scale_fp32_dev,
    QuantMeta::Kind scale_kind,
    void* y,
    int M, int N, int K,
    float beta,
    cudaStream_t stream)
{
    auto& ctx = LtCtx::instance();
    const std::size_t weight_elems =
        static_cast<std::size_t>(N) * static_cast<std::size_t>(K);
    void* bf16_w = ctx.dequant.ensure(weight_elems * 2);

    if (scale_kind == QuantMeta::Kind::PerChannel) {
        // [N] device scale → broadcast across K columns. Stays on
        // device throughout — no host sync needed.
        kernels::launch_dequant_fp8_e4m3_to_bf16_per_channel(
            static_cast<const std::uint8_t*>(w_fp8),
            bf16_w,
            static_cast<const float*>(w_scale_fp32_dev),
            N, K, stream);
    } else {
        // Per-tensor: pull the scalar to host. One sync per layer per
        // fire — acceptable on this fallback path.
        float scale = 0.f;
        CUDA_CHECK(cudaMemcpyAsync(&scale, w_scale_fp32_dev, sizeof(float),
                                   cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        kernels::launch_dequant_fp8_e4m3_to_bf16(
            static_cast<const std::uint8_t*>(w_fp8),
            bf16_w, scale, weight_elems, stream);
    }
    gemm_bf16_impl(cublas_handle, act, bf16_w, y, M, N, K, beta);
}

// ── cuBLASLt BF16 path (P1.2) ─────────────────────────────────────────
//
// Same row-major y[M,N] = act[M,K] @ W[N,K]^T problem as `gemm_bf16_impl`,
// but routed through cuBLASLt with `cublasLtMatmulAlgoGetHeuristic` so
// the engine picks a small-tile tensor-core kernel for decode-shaped
// matmuls (small batch, wide MN). Heuristic queries are cached by shape
// + beta — the shape set per layer is fixed at boot, so steady-state has
// zero heuristic calls.

struct LtBf16AlgoKey {
    int M, N, K;
    bool beta_one;
    bool operator==(const LtBf16AlgoKey& o) const noexcept {
        return M == o.M && N == o.N && K == o.K && beta_one == o.beta_one;
    }
};

struct LtBf16AlgoKeyHash {
    std::size_t operator()(const LtBf16AlgoKey& k) const noexcept {
        // FNV-1a-ish — only a few keys per process (one per matmul
        // shape per arch), so collision resistance doesn't matter.
        std::size_t h = 1469598103934665603ULL;
        for (auto v : {k.M, k.N, k.K, k.beta_one ? 1 : 0}) {
            h ^= static_cast<std::size_t>(v);
            h *= 1099511628211ULL;
        }
        return h;
    }
};

struct LtBf16AlgoCache {
    // The cached algo descriptor is portable across calls of the same
    // shape; cuBLASLt internally owns no per-call state.
    std::unordered_map<LtBf16AlgoKey, cublasLtMatmulAlgo_t, LtBf16AlgoKeyHash> map;
    std::mutex mu;
};

LtBf16AlgoCache& bf16_algo_cache_() {
    static LtBf16AlgoCache c;
    return c;
}

void gemm_bf16_lt_impl(
    cublasHandle_t handle,
    const void* act, const void* W, void* y,
    int M, int N, int K,
    float beta)
{
    auto& ctx = LtCtx::instance();
    ctx.ensure_init();

    // Inherit the stream from the legacy cuBLAS handle so call-site
    // ordering is preserved (everything in the forward pass shares one
    // stream today; defensive in case that changes).
    cudaStream_t stream = nullptr;
    if (handle) {
        // cublasGetStream is the documented way to extract the stream
        // from the legacy handle. Errors here are non-fatal — null stream
        // means the default stream, which is what we want anyway.
        cublasGetStream(handle, &stream);
    }

    const float alpha = 1.f;

    // Same row-major-as-col-major reinterpretation: A=W (op_T),
    // B=act (op_N), C=y (col-major NxM, ld=N).
    LtMatmulDesc desc(CUBLAS_COMPUTE_32F, CUDA_R_32F);
    const cublasOperation_t op_t = CUBLAS_OP_T;
    const cublasOperation_t op_n = CUBLAS_OP_N;
    desc.set(CUBLASLT_MATMUL_DESC_TRANSA, op_t);
    desc.set(CUBLASLT_MATMUL_DESC_TRANSB, op_n);

    // A: W stored row-major [N, K] = col-major [K, N], lda=K.
    LtMatrixLayout A(CUDA_R_16BF, /*rows=*/K, /*cols=*/N, /*ld=*/K);
    // B: act stored row-major [M, K] = col-major [K, M], ldb=K.
    LtMatrixLayout B(CUDA_R_16BF, /*rows=*/K, /*cols=*/M, /*ld=*/K);
    // C/D: y stored row-major [M, N] = col-major [N, M], ldc=N.
    LtMatrixLayout C(CUDA_R_16BF, /*rows=*/N, /*cols=*/M, /*ld=*/N);

    const bool beta_one = (beta != 0.f);
    LtBf16AlgoKey key{M, N, K, beta_one};

    cublasLtMatmulAlgo_t algo;
    bool have_algo = false;
    auto& cache = bf16_algo_cache_();
    {
        std::lock_guard<std::mutex> g(cache.mu);
        auto it = cache.map.find(key);
        if (it != cache.map.end()) {
            algo = it->second;
            have_algo = true;
        }
    }

    if (!have_algo) {
        LtMatmulPref pref;
        pref.set(CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, ctx.workspace_bytes);

        cublasLtMatmulHeuristicResult_t result{};
        int returned = 0;
        const auto hs = cublasLtMatmulAlgoGetHeuristic(
            ctx.handle, desc.d, A.d, B.d, C.d, C.d, pref.d,
            /*requestedAlgoCount=*/1, &result, &returned);
        if (hs != CUBLAS_STATUS_SUCCESS || returned == 0) {
            // No suitable algo (rare — e.g. unusual K). Fall back to the
            // legacy gemm path so we never silently emit no-op.
            const auto status = cublasGemmEx(
                handle, op_t, op_n, /*m=*/N, /*n=*/M, /*k=*/K,
                &alpha, W, CUDA_R_16BF, K, act, CUDA_R_16BF, K,
                &beta,  y, CUDA_R_16BF, N,
                CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
            if (status != CUBLAS_STATUS_SUCCESS) {
                throw std::runtime_error(
                    "cuBLASLt heuristic + GemmEx fallback both failed for "
                    "M=" + std::to_string(M) + " N=" + std::to_string(N) +
                    " K=" + std::to_string(K));
            }
            return;
        }
        algo = result.algo;
        std::lock_guard<std::mutex> g(cache.mu);
        cache.map[key] = algo;
    }

    LT_CHECK(cublasLtMatmul(
        ctx.handle, desc.d,
        &alpha,
        W,   A.d,
        act, B.d,
        &beta,
        y,   C.d,
        y,   C.d,
        &algo,
        ctx.workspace, ctx.workspace_bytes,
        stream));
}

void gemm_fp8_e4m3_w_bf16_act_impl(
    cublasHandle_t cublas_handle,
    const void* act, const void* w_fp8, const void* w_scale_fp32_dev,
    QuantMeta::Kind scale_kind,
    void* y,
    int M, int N, int K,
    float beta,
    cudaStream_t stream)
{
    if (!w_scale_fp32_dev) {
        throw std::runtime_error(
            "gemm_act_x_w[FP8_E4M3]: scale pointer is null — "
            "weight_scale_inv must be attached to the materialized WeightStore "
            "as an FP32 device tensor before calling FP8 GEMM");
    }
    auto& ctx = LtCtx::instance();
    ctx.ensure_init();

    // For now, route per-channel through the dequant fallback regardless
    // of GPU. The cuBLASLt vector-scale attribute (CUBLASLT_..._SCALE_
    // VECTOR_POINTER + CUBLASLT_MATMUL_DESC_A_SCALE_MODE) is Hopper-only
    // and lands in a follow-up. Per-tensor takes the native LT path on
    // sm89+ as before.
    if (!ctx.fp8_native_supported || scale_kind == QuantMeta::Kind::PerChannel) {
        gemm_fp8_dequant_then_bf16_fallback(
            cublas_handle, act, w_fp8, w_scale_fp32_dev, scale_kind, y,
            M, N, K, beta, stream);
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
        throw std::runtime_error(
            "cuBLAS error (" + std::to_string(static_cast<int>(status)) +
            "): cublasGemmEx[int8 W8A8] M=" + std::to_string(M) +
            " N=" + std::to_string(N) + " K=" + std::to_string(K));
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
        gemm_fp8_e4m3_w_bf16_act_impl(handle, act, w.data, w.scale_data,
                                      w.quant_kind,
                                      y, M, N, K, beta, stream);
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
            "gemm_act_x_w[INT4_PACKED]: marlin is not compiled into this "
            "build (PIE_CUDA_BUILD_MARLIN was OFF). Reconfigure cmake with "
            "-DPIE_CUDA_BUILD_MARLIN=ON to enable W4A16 GEMM.");
#endif
    }
    unsupported("gemm_act_x_w", act_dtype, w.dtype, y_dtype);
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

}  // namespace pie_cuda_driver::ops
