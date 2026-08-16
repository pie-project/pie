// The C++ half of the §45 parity check — gate-gemm-service.
//
// Four host launchers left `gemm/gemm.cpp` in §45 and became Rust in
// `driver-cuda/src/bind/service.rs`. They contain no `__global__` and never
// did: each is an argument tuple handed to cuBLAS. So the question the port
// has to answer is not "is the arithmetic the same" — it is the same library
// doing the arithmetic — but "is the TUPLE the same", and a transposed `lda`
// or a stride off by a head answers that with a matrix full of plausible
// numbers.
//
// This file is the four bodies AS THEY WERE, copied out of the ARCHIVE crate
// — `git show 06074cbf3:crates/kernels-cuda/csrc/src/gemm/gemm.cpp`, a path
// that has to be reached through `git show` because `85c6c674b` deleted that
// crate whole — and edited only where a symbol they closed over is not here
// (`check`, `bf16_compute_type`, the `throw` text). It drives them over the
// shape table in `SHAPES` below with the pseudo-random bf16 that
// `tests/gemm_service_parity.rs` generates from the same recurrence, and
// prints one line per case:
//
//     <case name> <dims...> <fnv1a64 of the output bytes> <byte count>
//
// The Rust test builds the same transcript from `bind::service::*` and
// requires it to be equal LINE BY LINE. Not a norm, not a tolerance, not a
// cell count: the bytes of the output buffer, hashed, per shape.
//
// Regenerate with `run.sh`. The transcript is cuBLAS's, so it is specific to
// the GPU that produced it — the same way every other device golden in this
// directory is, and this tree's `build.rs` compiles for `sm_89` and nothing
// else.

#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

// ── what the bodies closed over ────────────────────────────────────────────

cublasComputeType_t bf16_compute_type() { return CUBLAS_COMPUTE_32F; }

void check(cublasStatus_t status, const char* what) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        std::fprintf(stderr, "cuBLAS error (%d): %s\n",
                     static_cast<int>(status), what);
        std::exit(2);
    }
}

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "CUDA error (%s): %s\n", cudaGetErrorString(e),
                     what);
        std::exit(2);
    }
}

// ── the four bodies, verbatim ──────────────────────────────────────────────

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
    check(status, "cublasGemmEx[bf16->fp32]");
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
    check(status, "cublasGemmGroupedBatchedEx[bf16]");
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

// ── hostile input, and the hash of an answer ───────────────────────────────
//
// Spelled the same way in `tests/gemm_service_parity.rs`. `env-audit`
// measured benign inputs showing ZERO difference everywhere while wide
// exponents exposed real ones and flipped two verdicts, so the generator is
// built to spread the exponent, not the mantissa: sign uniform, exponent over
// 2^-17..2^18, and every 17th element exactly zero so the sparsity a real
// activation has is represented too. Products stay finite for every K here.

uint64_t xs(uint64_t& s) {
    s ^= s >> 12;
    s ^= s << 25;
    s ^= s >> 27;
    return s * 0x2545F4914F6CDD1DULL;
}

void fill_bf16(std::vector<uint16_t>& v, uint64_t seed) {
    uint64_t s = seed | 1ULL;
    for (size_t i = 0; i < v.size(); ++i) {
        const uint64_t u = xs(s);
        if (i % 17 == 16) { v[i] = 0; continue; }
        const uint16_t sign = static_cast<uint16_t>((u >> 31) & 1);
        const uint16_t exp = static_cast<uint16_t>(110 + ((u >> 17) % 36));
        const uint16_t mant = static_cast<uint16_t>((u >> 3) & 0x7f);
        v[i] = static_cast<uint16_t>((sign << 15) | (exp << 7) | mant);
    }
}

uint64_t fnv1a(const void* data, size_t n) {
    const auto* p = static_cast<const uint8_t*>(data);
    uint64_t h = 0xcbf29ce484222325ULL;
    for (size_t i = 0; i < n; ++i) {
        h = (h ^ p[i]) * 0x100000001b3ULL;
    }
    return h;
}

void emit(const std::string& name, const void* host, size_t bytes) {
    std::printf("%s %016llx %zu\n", name.c_str(),
                static_cast<unsigned long long>(fnv1a(host, bytes)), bytes);
}

void* dev(const void* host, size_t bytes) {
    void* p = nullptr;
    ck(cudaMalloc(&p, bytes ? bytes : 16), "cudaMalloc");
    if (bytes) ck(cudaMemcpy(p, host, bytes, cudaMemcpyHostToDevice), "H2D");
    return p;
}

// ── the shape table ────────────────────────────────────────────────────────

struct Dense { int m, n, k; };
// M, N and K each vary, K=1 and N=1 are degenerate axes, and M=0 is the
// empty rectangle a decode step with nothing scheduled hands in.
const Dense DENSE[] = {
    {1, 1, 1}, {1, 64, 128}, {7, 33, 17},  {64, 64, 64},
    {128, 1, 256}, {3, 5, 1}, {0, 64, 64}, {33, 129, 65},
};

struct Grouped { int groups; int ms[6]; int n, k; float beta; };
// `group_count == 0` is the refusal the archive made before any launch and
// the port keeps; beta != 0 is the accumulate form the row's last operand
// selects, and it reads the output buffer, so it pins the ldc too.
const Grouped GROUPED[] = {
    {1, {1, 0, 0, 0, 0, 0}, 16, 32, 0.f},
    {3, {1, 4, 9, 0, 0, 0}, 32, 48, 0.f},
    {3, {1, 4, 9, 0, 0, 0}, 32, 48, 1.f},
    {4, {2, 7, 1, 16, 0, 0}, 17, 33, 0.f},
    {6, {1, 1, 1, 1, 1, 1}, 5, 7, 0.5f},
    {0, {0, 0, 0, 0, 0, 0}, 8, 8, 0.f},
};

struct Mla { int tokens, heads, qk_nope, v_head, kv_lora; };
// tokens=0 and heads=0 are the two halves of the archive's early return; the
// odd row exists because every stride here is a product of two of these and a
// transposition survives any shape where two of them agree.
const Mla MLA[] = {
    {1, 1, 1, 1, 1},   {2, 4, 8, 16, 8},  {5, 3, 7, 9, 11},
    {1, 16, 128, 128, 512}, {0, 4, 8, 16, 8}, {3, 0, 8, 16, 8},
    {9, 2, 3, 5, 7},
};

}  // namespace

int main() {
    cublasHandle_t handle = nullptr;
    check(cublasCreate(&handle), "cublasCreate");

    for (const auto& d : DENSE) {
        std::vector<uint16_t> act(static_cast<size_t>(d.m) * d.k);
        std::vector<uint16_t> w(static_cast<size_t>(d.n) * d.k);
        fill_bf16(act, 0x9E3779B97F4A7C15ULL ^ (static_cast<uint64_t>(d.m) << 32 |
                  static_cast<uint64_t>(d.n) << 16 | static_cast<uint64_t>(d.k)));
        fill_bf16(w, 0xD1B54A32D192ED03ULL ^ (static_cast<uint64_t>(d.k) << 32 |
                  static_cast<uint64_t>(d.n) << 16 | static_cast<uint64_t>(d.m)));
        const size_t ybytes = static_cast<size_t>(d.m) * d.n * 4;
        std::vector<float> y(static_cast<size_t>(d.m) * d.n, -1.f);
        void* dact = dev(act.data(), act.size() * 2);
        void* dw = dev(w.data(), w.size() * 2);
        void* dy = dev(y.data(), ybytes);
        gemm_bf16_out_fp32_impl(handle, dact, dw, static_cast<float*>(dy),
                                d.m, d.n, d.k);
        ck(cudaDeviceSynchronize(), "sync");
        if (ybytes) ck(cudaMemcpy(y.data(), dy, ybytes, cudaMemcpyDeviceToHost), "D2H");
        char nm[64];
        std::snprintf(nm, sizeof nm, "out_fp32 m=%d n=%d k=%d", d.m, d.n, d.k);
        emit(nm, y.data(), ybytes);
        cudaFree(dact); cudaFree(dw); cudaFree(dy);
    }

    for (const auto& g : GROUPED) {
        std::vector<const void*> actp(g.groups), wp(g.groups);
        std::vector<void*> yp(g.groups);
        std::vector<void*> owned;
        std::vector<uint16_t> yhost;
        size_t total = 0;
        for (int i = 0; i < g.groups; ++i) total += static_cast<size_t>(g.ms[i]) * g.n;
        yhost.assign(total, 0);
        // The output starts as recognisable garbage so a beta != 0 case that
        // failed to READ it would not accidentally agree.
        fill_bf16(yhost, 0x27D4EB2F165667C5ULL ^ static_cast<uint64_t>(g.groups));
        size_t off = 0;
        for (int i = 0; i < g.groups; ++i) {
            std::vector<uint16_t> a(static_cast<size_t>(g.ms[i]) * g.k);
            std::vector<uint16_t> w(static_cast<size_t>(g.n) * g.k);
            fill_bf16(a, 0x94D049BB133111EBULL ^ (static_cast<uint64_t>(i) << 40) ^
                      (static_cast<uint64_t>(g.n) << 8) ^ static_cast<uint64_t>(g.k));
            fill_bf16(w, 0xBF58476D1CE4E5B9ULL ^ (static_cast<uint64_t>(i) << 40) ^
                      (static_cast<uint64_t>(g.k) << 8) ^ static_cast<uint64_t>(g.n));
            void* da = dev(a.data(), a.size() * 2);
            void* dw = dev(w.data(), w.size() * 2);
            void* dy = dev(yhost.data() + off, static_cast<size_t>(g.ms[i]) * g.n * 2);
            owned.push_back(da); owned.push_back(dw); owned.push_back(dy);
            actp[i] = da; wp[i] = dw; yp[i] = dy;
            off += static_cast<size_t>(g.ms[i]) * g.n;
        }
        std::vector<int> ms(g.ms, g.ms + (g.groups ? g.groups : 1));
        gemm_grouped_bf16_impl(handle, actp.data(), wp.data(), yp.data(),
                               ms.data(), g.groups, g.n, g.k, g.beta);
        ck(cudaDeviceSynchronize(), "sync");
        off = 0;
        for (int i = 0; i < g.groups; ++i) {
            const size_t nb = static_cast<size_t>(g.ms[i]) * g.n * 2;
            if (nb) ck(cudaMemcpy(yhost.data() + off, yp[i], nb, cudaMemcpyDeviceToHost), "D2H");
            off += static_cast<size_t>(g.ms[i]) * g.n;
        }
        char nm[96];
        std::snprintf(nm, sizeof nm, "grouped g=%d n=%d k=%d beta=%g",
                      g.groups, g.n, g.k, static_cast<double>(g.beta));
        emit(nm, yhost.data(), total * 2);
        for (void* p : owned) cudaFree(p);
    }

    for (const auto& m : MLA) {
        const size_t bank = static_cast<size_t>(m.heads) *
                            (static_cast<size_t>(m.qk_nope) + m.v_head) * m.kv_lora;
        std::vector<uint16_t> kvb(bank ? bank : 1);
        fill_bf16(kvb, 0x2545F4914F6CDD1DULL ^ (static_cast<uint64_t>(m.heads) << 32) ^
                  (static_cast<uint64_t>(m.qk_nope) << 16) ^ static_cast<uint64_t>(m.kv_lora));

        // q -> latent
        {
            std::vector<uint16_t> q(static_cast<size_t>(m.tokens) * m.heads * m.qk_nope);
            fill_bf16(q, 0x9E3779B185EBCA87ULL ^ (static_cast<uint64_t>(m.tokens) << 32) ^
                      static_cast<uint64_t>(m.heads));
            const size_t ob = static_cast<size_t>(m.tokens) * m.heads * m.kv_lora;
            std::vector<uint16_t> out(ob ? ob : 1, 0xBEEF);
            void* dq = dev(q.data(), q.size() * 2);
            void* dk = dev(kvb.data(), kvb.size() * 2);
            void* dout = dev(out.data(), out.size() * 2);
            mla_absorb_q_to_latent_bf16(handle, dq, dk, dout, m.tokens, m.heads,
                                        m.qk_nope, m.v_head, m.kv_lora);
            ck(cudaDeviceSynchronize(), "sync");
            ck(cudaMemcpy(out.data(), dout, out.size() * 2, cudaMemcpyDeviceToHost), "D2H");
            char nm[128];
            std::snprintf(nm, sizeof nm, "absorb_q t=%d h=%d nope=%d v=%d lora=%d",
                          m.tokens, m.heads, m.qk_nope, m.v_head, m.kv_lora);
            emit(nm, out.data(), ob * 2);
            cudaFree(dq); cudaFree(dk); cudaFree(dout);
        }
        // latent -> v
        {
            std::vector<uint16_t> lat(static_cast<size_t>(m.tokens) * m.heads * m.kv_lora);
            fill_bf16(lat, 0xFF51AFD7ED558CCDULL ^ (static_cast<uint64_t>(m.tokens) << 32) ^
                      static_cast<uint64_t>(m.kv_lora));
            const size_t ob = static_cast<size_t>(m.tokens) * m.heads * m.v_head;
            std::vector<uint16_t> out(ob ? ob : 1, 0xBEEF);
            void* dl = dev(lat.data(), lat.size() * 2);
            void* dk = dev(kvb.data(), kvb.size() * 2);
            void* dout = dev(out.data(), out.size() * 2);
            mla_absorb_latent_to_v_bf16(handle, dl, dk, dout, m.tokens, m.heads,
                                        m.qk_nope, m.v_head, m.kv_lora);
            ck(cudaDeviceSynchronize(), "sync");
            ck(cudaMemcpy(out.data(), dout, out.size() * 2, cudaMemcpyDeviceToHost), "D2H");
            char nm[128];
            std::snprintf(nm, sizeof nm, "absorb_v t=%d h=%d nope=%d v=%d lora=%d",
                          m.tokens, m.heads, m.qk_nope, m.v_head, m.kv_lora);
            emit(nm, out.data(), ob * 2);
            cudaFree(dl); cudaFree(dk); cudaFree(dout);
        }
    }

    cublasDestroy(handle);
    return 0;
}
