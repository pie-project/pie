// Standalone repro for the flashinfer TopKTopPSamplingFromProb illegal-memory-access.
// Builds a hand-crafted fp32 probs buffer and invokes the kernel exactly the
// way our wrapper does. Run under `compute-sanitizer --tool memcheck` to
// localize the OOB.

#include <cstdio>
#include <vector>

#include <cuda_runtime.h>
#include <flashinfer/sampling.cuh>

#define CHK(expr) do { auto _e = (expr); if (_e != cudaSuccess) { \
    std::fprintf(stderr, "CUDA error: %s at %s:%d — %s\n", \
                 cudaGetErrorString(_e), __FILE__, __LINE__, #expr); \
    return 1; } } while (0)

int main() {
    constexpr int N = 26;        // matches our prefill case
    constexpr int V = 151936;    // Qwen3-0.6B vocab
    constexpr int NUM_SAMPLING = 1;

    // Allocate probs buffer: [N, V] fp32, uniform 1/V then renormalized.
    float* d_probs = nullptr;
    CHK(cudaMalloc(&d_probs, sizeof(float) * N * V));
    {
        std::vector<float> h(N * V, 1.0f / V);
        CHK(cudaMemcpy(d_probs, h.data(), h.size() * sizeof(float), cudaMemcpyHostToDevice));
    }

    // Per-row top_k / top_p / seed of length N (flashinfer indexes by row_idx).
    std::vector<int32_t>  h_top_k(N, 0);     // 0 = no filter (we hope)
    std::vector<float>    h_top_p(N, 1.0f);  // 1.0 = no filter
    std::vector<uint64_t> h_seed (N, 12345ull);

    int32_t*  d_top_k = nullptr;
    float*    d_top_p = nullptr;
    uint64_t* d_seed  = nullptr;
    CHK(cudaMalloc(&d_top_k, sizeof(int32_t)  * N));
    CHK(cudaMalloc(&d_top_p, sizeof(float)    * N));
    CHK(cudaMalloc(&d_seed,  sizeof(uint64_t) * N));
    CHK(cudaMemcpy(d_top_k, h_top_k.data(), sizeof(int32_t)  * N, cudaMemcpyHostToDevice));
    CHK(cudaMemcpy(d_top_p, h_top_p.data(), sizeof(float)    * N, cudaMemcpyHostToDevice));
    CHK(cudaMemcpy(d_seed,  h_seed.data(),  sizeof(uint64_t) * N, cudaMemcpyHostToDevice));

    // Sample 1 row at index 25 (the last token in the batch).
    std::vector<int32_t> h_idx{25};
    int32_t* d_idx = nullptr;
    int32_t* d_out = nullptr;
    bool*    d_valid = nullptr;
    CHK(cudaMalloc(&d_idx,   sizeof(int32_t) * NUM_SAMPLING));
    CHK(cudaMalloc(&d_out,   sizeof(int32_t) * NUM_SAMPLING));
    CHK(cudaMalloc(&d_valid, sizeof(bool)    * NUM_SAMPLING));
    CHK(cudaMemcpy(d_idx, h_idx.data(), sizeof(int32_t) * NUM_SAMPLING, cudaMemcpyHostToDevice));

    auto status = ::flashinfer::sampling::TopKTopPSamplingFromProb<float, int32_t>(
        d_probs,
        d_top_k, d_top_p,
        d_out,
        /*valid=*/d_valid,
        d_idx,
        /*batch_size=*/static_cast<uint32_t>(NUM_SAMPLING),
        /*top_k_val=*/0, /*top_p_val=*/1.0f,
        /*d=*/static_cast<uint32_t>(V),
        /*deterministic=*/false,
        d_seed, /*seed_val=*/0,
        /*offset_arr=*/nullptr, /*offset_val=*/0,
        /*stream=*/nullptr);
    if (status != cudaSuccess) {
        std::fprintf(stderr, "TopKTopP launch returned: %s\n", cudaGetErrorString(status));
        return 2;
    }

    CHK(cudaDeviceSynchronize());

    int32_t out = 0;
    CHK(cudaMemcpy(&out, d_out, sizeof(int32_t), cudaMemcpyDeviceToHost));
    std::printf("Sampled token id: %d (vocab=%d)\n", out, V);

    cudaFree(d_probs); cudaFree(d_top_k); cudaFree(d_top_p);
    cudaFree(d_seed);  cudaFree(d_idx);   cudaFree(d_out);
    return 0;
}
