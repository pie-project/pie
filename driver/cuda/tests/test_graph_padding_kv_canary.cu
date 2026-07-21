#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include <cuda_runtime.h>

#include "kernels/kv_paged.hpp"
#include "kernels/mla_paged.hpp"
#include "kernels/split_packed.hpp"

namespace k = pie_cuda_driver::kernels;

namespace {

void check(cudaError_t error, const char* expression, int line) {
    if (error != cudaSuccess) {
        std::fprintf(
            stderr, "%s:%d: %s: %s\n", __FILE__, line, expression,
            cudaGetErrorString(error));
        std::exit(2);
    }
}

#define CUDA_CHECK(expr) check((expr), #expr, __LINE__)

constexpr int kRows = 2;
constexpr int kQHeads = 2;
constexpr int kKvHeads = 2;
constexpr int kHeadDim = 64;
constexpr int kPageSize = 4;
constexpr std::uint16_t kCanary = 0xa55a;
constexpr std::size_t kPageElements =
    static_cast<std::size_t>(kPageSize) * kKvHeads * kHeadDim;

struct Inputs {
    cudaStream_t stream = nullptr;
    std::uint16_t* packed = nullptr;
    std::uint16_t* q = nullptr;
    std::uint16_t* k_pages = nullptr;
    std::uint16_t* v_pages = nullptr;
    std::uint16_t* q_weight = nullptr;
    std::uint16_t* k_weight = nullptr;
    std::int32_t* positions = nullptr;
    std::uint32_t* qo_indptr = nullptr;
    std::uint32_t* page_indices = nullptr;
    std::uint32_t* page_indptr = nullptr;
    std::uint32_t* last_page_lens = nullptr;
    std::uint32_t* write_page = nullptr;
    std::uint32_t* write_offset = nullptr;
    std::uint8_t* row_valid = nullptr;

    Inputs() {
        const std::size_t packed_elements =
            static_cast<std::size_t>(kRows) *
            (kQHeads + 2 * kKvHeads) * kHeadDim;
        const std::vector<std::uint16_t> packed_h(packed_elements, 0x3f80);
        const std::vector<std::uint16_t> weight_h(kHeadDim, 0x3f80);
        const std::int32_t positions_h[kRows] = {0, 0};
        const std::uint32_t indptr_h[kRows + 1] = {0, 1, 2};
        const std::uint32_t pages_h[kRows] = {0, 1};
        const std::uint32_t last_h[kRows] = {1, 1};
        const std::uint32_t offsets_h[kRows] = {0, 0};
        const std::uint8_t valid_h[kRows] = {1, 0};

        CUDA_CHECK(cudaStreamCreate(&stream));
        CUDA_CHECK(cudaMalloc(&packed, packed_h.size() * sizeof(*packed)));
        CUDA_CHECK(cudaMalloc(
            &q, static_cast<std::size_t>(kRows) * kQHeads * kHeadDim *
                    sizeof(*q)));
        CUDA_CHECK(cudaMalloc(&k_pages, 2 * kPageElements * sizeof(*k_pages)));
        CUDA_CHECK(cudaMalloc(&v_pages, 2 * kPageElements * sizeof(*v_pages)));
        CUDA_CHECK(cudaMalloc(&q_weight, kHeadDim * sizeof(*q_weight)));
        CUDA_CHECK(cudaMalloc(&k_weight, kHeadDim * sizeof(*k_weight)));
        CUDA_CHECK(cudaMalloc(&positions, sizeof(positions_h)));
        CUDA_CHECK(cudaMalloc(&qo_indptr, sizeof(indptr_h)));
        CUDA_CHECK(cudaMalloc(&page_indices, sizeof(pages_h)));
        CUDA_CHECK(cudaMalloc(&page_indptr, sizeof(indptr_h)));
        CUDA_CHECK(cudaMalloc(&last_page_lens, sizeof(last_h)));
        CUDA_CHECK(cudaMalloc(&write_page, sizeof(pages_h)));
        CUDA_CHECK(cudaMalloc(&write_offset, sizeof(offsets_h)));
        CUDA_CHECK(cudaMalloc(&row_valid, sizeof(valid_h)));

        CUDA_CHECK(cudaMemcpy(
            packed, packed_h.data(), packed_h.size() * sizeof(*packed),
            cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(
            q_weight, weight_h.data(), kHeadDim * sizeof(*q_weight),
            cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(
            k_weight, weight_h.data(), kHeadDim * sizeof(*k_weight),
            cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(
            positions, positions_h, sizeof(positions_h),
            cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(
            qo_indptr, indptr_h, sizeof(indptr_h), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(
            page_indices, pages_h, sizeof(pages_h), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(
            page_indptr, indptr_h, sizeof(indptr_h), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(
            last_page_lens, last_h, sizeof(last_h), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(
            write_page, pages_h, sizeof(pages_h), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(
            write_offset, offsets_h, sizeof(offsets_h),
            cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(
            row_valid, valid_h, sizeof(valid_h), cudaMemcpyHostToDevice));
    }

    ~Inputs() {
        cudaFree(row_valid);
        cudaFree(write_offset);
        cudaFree(write_page);
        cudaFree(last_page_lens);
        cudaFree(page_indptr);
        cudaFree(page_indices);
        cudaFree(qo_indptr);
        cudaFree(positions);
        cudaFree(k_weight);
        cudaFree(q_weight);
        cudaFree(v_pages);
        cudaFree(k_pages);
        cudaFree(q);
        cudaFree(packed);
        cudaStreamDestroy(stream);
    }

    pie_cuda_driver::KvCacheLayerView layer() const {
        pie_cuda_driver::KvCacheLayerView view;
        view.num_pages = 2;
        view.page_size = kPageSize;
        view.num_kv_heads = kKvHeads;
        view.head_dim = kHeadDim;
        view.k_pages = k_pages;
        view.v_pages = v_pages;
        view.hnd_layout = false;
        view.native_bf16 = true;
        return view;
    }

    void reset_pages() const {
        const std::vector<std::uint16_t> canary(2 * kPageElements, kCanary);
        CUDA_CHECK(cudaMemcpy(
            k_pages, canary.data(), canary.size() * sizeof(*k_pages),
            cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(
            v_pages, canary.data(), canary.size() * sizeof(*v_pages),
            cudaMemcpyHostToDevice));
    }

    bool pad_page_untouched() const {
        std::vector<std::uint16_t> k_h(kPageElements);
        std::vector<std::uint16_t> v_h(kPageElements);
        CUDA_CHECK(cudaMemcpy(
            k_h.data(), k_pages + kPageElements,
            kPageElements * sizeof(*k_pages), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(
            v_h.data(), v_pages + kPageElements,
            kPageElements * sizeof(*v_pages), cudaMemcpyDeviceToHost));
        for (std::size_t i = 0; i < kPageElements; ++i) {
            if (k_h[i] != kCanary || v_h[i] != kCanary) return false;
        }
        return true;
    }
};

template <class Launch>
bool run_graph_canary(const char* model, Inputs& in, Launch&& launch) {
    in.reset_pages();
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t exec = nullptr;
    CUDA_CHECK(cudaStreamBeginCapture(in.stream, cudaStreamCaptureModeGlobal));
    launch();
    CUDA_CHECK(cudaStreamEndCapture(in.stream, &graph));
    CUDA_CHECK(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
    CUDA_CHECK(cudaGraphLaunch(exec, in.stream));
    CUDA_CHECK(cudaStreamSynchronize(in.stream));
    const bool ok = in.pad_page_untouched();
    std::printf("[%s] %s graph-pad KV canary\n", ok ? " ok " : "FAIL", model);
    CUDA_CHECK(cudaGraphExecDestroy(exec));
    CUDA_CHECK(cudaGraphDestroy(graph));
    return ok;
}

}  // namespace

int main() {
    Inputs in;
    bool ok = true;

    const auto llama_fused = [&] {
        k::launch_qkv_decode_qk_norm_rope_write_kv_bf16(
            in.packed, in.q, in.k_pages, in.v_pages, in.q_weight, in.k_weight,
            in.positions, nullptr, in.page_indices, in.page_indptr,
            in.last_page_lens, nullptr, nullptr, in.row_valid, kRows, kQHeads,
            kKvHeads, kHeadDim, kPageSize, false, 10000.0f, 1.0e-6f,
            in.stream);
    };
    ok &= run_graph_canary("LlamaLikeModel", in, llama_fused);
    ok &= run_graph_canary("Qwen3VLModel", in, llama_fused);

    const auto explicit_write = [&] {
        k::launch_write_kv_explicit_bf16(
            in.layer(), in.packed, in.packed, in.write_page, in.write_offset,
            kRows, in.stream, in.row_valid);
    };
    ok &= run_graph_canary("Qwen35Model", in, explicit_write);
    ok &= run_graph_canary("Qwen35MoeModel", in, explicit_write);

    const auto gemma_fused = [&] {
        k::launch_qkv_packed_qk_norm_rope_vnorm_write_kv_bf16(
            in.packed, in.q, in.k_pages, in.v_pages, in.q_weight, in.k_weight,
            in.positions, in.page_indices, in.page_indptr, in.last_page_lens,
            in.row_valid, kRows, kQHeads, kKvHeads, kHeadDim, kPageSize, false,
            10000.0f, 1.0e-6f, in.stream);
    };
    ok &= run_graph_canary("Gemma4Model/fused", in, gemma_fused);

    const auto generic_write = [&] {
        k::launch_write_kv_to_pages(
            in.layer(), in.packed, in.packed, in.qo_indptr, in.page_indices,
            in.page_indptr, in.last_page_lens, kRows, kRows, in.stream,
            in.row_valid);
    };
    ok &= run_graph_canary("Gemma4Model/fallback", in, generic_write);
    ok &= run_graph_canary("NemotronHModel", in, generic_write);

    const auto mla_write = [&] {
        k::launch_write_mla_to_pages_bf16(
            in.k_pages, in.v_pages, in.packed, in.packed,
            in.qo_indptr, in.page_indices, in.page_indptr,
            in.last_page_lens, kRows, kRows, kPageSize,
            /*kv_lora_rank=*/8, /*qk_rope_head_dim=*/4,
            in.stream, in.row_valid);
    };
    ok &= run_graph_canary("KimiModel", in, mla_write);
    ok &= run_graph_canary("Glm5Model", in, mla_write);

    return ok ? 0 : 1;
}
