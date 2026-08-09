// The one BEHAVIOURAL change §45 makes, measured — gate-gemm-service.
//
// `gemm::act_x_wt_bias_bf16` had a fused arm: at `M == 1` with a bias it ran
// the warp-per-row GEMV with the bias in the epilogue, one launch instead of
// two. The Rust composition in `bind::service` cannot take that arm — its two
// steps are `gemm::act_x_wt_bf16` and then `norm::add_bias_bf16` — so the
// fused launch is gone and one extra launch per biased M=1 projection is
// paid.
//
// `gemv.hpp:25-28` says that costs no ACCURACY, because the epilogue computes
// `out[n] = bf16(bf16(dot) + bias[n])` and *"the double rounding is
// intentional"*, making it bit-identical to running `add_bias_bf16`
// afterwards. That sentence is the whole justification for the composition,
// and it was prose. This measures it.
//
// Both halves are the ARCHIVE'S OWN CODE, linked out of
// `libpie_kernels_cuda.a`: `gemm::gemv_bf16` with a bias, against
// `gemm::gemv_bf16` with `nullptr` followed by `norm::add_bias_bf16`. If the
// hashes agree on every shape, the composition writes the archive's bytes; if
// they do not, the header is wrong and §45 changed an answer.
//
// Hostile input, same generator as `oracle.cu`: wide exponents, uniform sign,
// every seventeenth element exactly zero. A bias fold is exactly where a
// benign input hides a rounding difference, because `dot + bias` only rounds
// differently when the two are far apart in magnitude.

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace pie_cuda_driver::kernels::gemm {
bool gemv_bf16(const void* weight, const void* act, const void* bias, void* out,
               int N, int K, cudaStream_t stream, float beta);
}
namespace pie_cuda_driver::kernels::norm {
void add_bias_bf16(void* y, const void* bias, int num_rows, int dim,
                   cudaStream_t stream);
}

namespace {

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "CUDA error (%s): %s\n", cudaGetErrorString(e), what);
        std::exit(2);
    }
}

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
    for (size_t i = 0; i < n; ++i) h = (h ^ p[i]) * 0x100000001b3ULL;
    return h;
}

void* dev(const void* host, size_t bytes) {
    void* p = nullptr;
    ck(cudaMalloc(&p, bytes ? bytes : 16), "cudaMalloc");
    if (bytes) ck(cudaMemcpy(p, host, bytes, cudaMemcpyHostToDevice), "H2D");
    return p;
}

// N and K both move; the alignment predicate `gemv_bf16` applies is a
// function of K, so a K that is not a multiple of 8 is included on purpose --
// the launcher REFUSES those, which is a host decision made before any launch
// and is reported here rather than hidden.
struct Shape { int n, k; };
const Shape SHAPES[] = {
    {32, 2048}, {2048, 4096}, {4096, 2048}, {8192, 2048},
    {1, 8},     {5, 7},       {33, 129},    {128, 64},
};

}  // namespace

int main() {
    using namespace pie_cuda_driver::kernels;
    for (const auto& s : SHAPES) {
        std::vector<uint16_t> w(static_cast<size_t>(s.n) * s.k), act(s.k), bias(s.n);
        fill_bf16(w, 0xD1B54A32D192ED03ULL ^ (static_cast<uint64_t>(s.n) << 20) ^ s.k);
        fill_bf16(act, 0x9E3779B97F4A7C15ULL ^ static_cast<uint64_t>(s.k));
        fill_bf16(bias, 0xBF58476D1CE4E5B9ULL ^ static_cast<uint64_t>(s.n));

        std::vector<uint16_t> fused(s.n, 0), split(s.n, 0);
        void* dw = dev(w.data(), w.size() * 2);
        void* da = dev(act.data(), act.size() * 2);
        void* db = dev(bias.data(), bias.size() * 2);
        void* df = dev(fused.data(), fused.size() * 2);
        void* ds = dev(split.data(), split.size() * 2);

        const bool ok_fused =
            gemm::gemv_bf16(dw, da, db, df, s.n, s.k, nullptr, 0.f);
        const bool ok_split =
            gemm::gemv_bf16(dw, da, nullptr, ds, s.n, s.k, nullptr, 0.f);
        if (ok_split) norm::add_bias_bf16(ds, db, 1, s.n, nullptr);
        ck(cudaDeviceSynchronize(), "sync");

        if (ok_fused != ok_split) {
            std::printf("fold n=%d k=%d REFUSAL-DISAGREES fused=%d split=%d\n",
                        s.n, s.k, static_cast<int>(ok_fused),
                        static_cast<int>(ok_split));
        } else if (!ok_fused) {
            // A refusal is not a failure: `gemv_bf16` returns false when the
            // shape or alignment is unsupported and enqueues nothing, and the
            // caller goes to cuBLAS. Recorded so the transcript cannot be
            // green because everything declined.
            std::printf("fold n=%d k=%d REFUSED-BOTH\n", s.n, s.k);
        } else {
            ck(cudaMemcpy(fused.data(), df, fused.size() * 2, cudaMemcpyDeviceToHost), "D2H");
            ck(cudaMemcpy(split.data(), ds, ds ? split.size() * 2 : 0, cudaMemcpyDeviceToHost), "D2H");
            size_t diff = 0;
            for (int i = 0; i < s.n; ++i) diff += (fused[i] != split[i]);
            std::printf("fold n=%d k=%d fused=%016llx split=%016llx differing=%zu of %d\n",
                        s.n, s.k,
                        static_cast<unsigned long long>(fnv1a(fused.data(), fused.size() * 2)),
                        static_cast<unsigned long long>(fnv1a(split.data(), split.size() * 2)),
                        diff, s.n);
        }
        cudaFree(dw); cudaFree(da); cudaFree(db); cudaFree(df); cudaFree(ds);
    }
    return 0;
}
