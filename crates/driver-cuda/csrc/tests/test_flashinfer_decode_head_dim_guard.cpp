#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>

#include "attention_workspace.hpp"
#include "attn/attention_flashinfer.hpp"

int main() {
    using namespace pie_cuda_driver;

    std::uint32_t kv_page_indptr[2] = {0, 1};
    auto plan = kernels::attn::make_decode_plan();
    AttentionWorkspace workspace;

    try {
        kernels::attn::plan_attention_flashinfer_decode_bf16(
            *plan,
            kv_page_indptr,
            /*num_requests=*/1,
            /*num_q_heads=*/1,
            /*num_kv_heads=*/1,
            /*head_dim=*/80,
            /*page_size=*/16,
            workspace.view(),
            /*stream=*/nullptr,
            /*enable_cuda_graph=*/false);
    } catch (const std::runtime_error& e) {
        const std::string msg = e.what();
        if (msg.find("unsupported head_dim 80") != std::string::npos) {
            std::puts("flashinfer decode head_dim guard ok");
            return 0;
        }
        std::fprintf(stderr, "unexpected error: %s\n", e.what());
        return 1;
    }

    std::fprintf(stderr, "head_dim=80 decode planning unexpectedly succeeded\n");
    return 1;
}
