// Cross-request batching (#10) device helpers — see group.hpp for the design.

#include "sampling_ir/group.hpp"

#include <cstdint>

namespace pie_cuda_driver::sampling_ir {

void gather_logits_bf16(const void* src_base, std::span<const std::uint32_t> rows,
                        std::uint32_t vocab, void* dst, cudaStream_t stream) {
    // Compact a scattered group into [Ng, V] bf16: one D2D row copy per row. The
    // batched launch reads dst contiguously (base + g*V). #11's codegen row-index
    // indirection replaces this with an in-kernel gather (no copy).
    const std::size_t row_bytes = static_cast<std::size_t>(vocab) * sizeof(std::uint16_t);
    const char* src = static_cast<const char*>(src_base);
    char* d = static_cast<char*>(dst);
    for (std::size_t g = 0; g < rows.size(); ++g) {
        cudaMemcpyAsync(d + g * row_bytes,
                        src + static_cast<std::size_t>(rows[g]) * row_bytes, row_bytes,
                        cudaMemcpyDeviceToDevice, stream);
    }
}

void scatter_tokens_i32(const void* src_compact, std::span<const std::uint32_t> rows,
                        void* dst_base, cudaStream_t stream) {
    // Scatter the group's [Ng] compact tokens back to their original rows.
    constexpr std::size_t tok = sizeof(std::int32_t);
    const char* src = static_cast<const char*>(src_compact);
    char* d = static_cast<char*>(dst_base);
    for (std::size_t g = 0; g < rows.size(); ++g) {
        cudaMemcpyAsync(d + static_cast<std::size_t>(rows[g]) * tok, src + g * tok, tok,
                        cudaMemcpyDeviceToDevice, stream);
    }
}

}  // namespace pie_cuda_driver::sampling_ir
