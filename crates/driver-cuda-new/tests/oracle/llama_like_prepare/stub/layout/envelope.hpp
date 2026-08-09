#pragma once
// Stub for csrc/src/layout/envelope.hpp.
//
// `launch_envelope_seed_empty_bf16` writes +inf/-inf into every envelope entry
// on the device. Its EFFECT is a kernel and out of scope here; that it is
// called once per owning layer, with the resolved head count and head dim, is
// not -- so the stub records the call.
#include <cstdint>
#include <string>
#include <vector>
namespace pie_cuda_driver::kernels::layout {
inline std::vector<std::string> g_seed_log;
inline void launch_envelope_seed_empty_bf16(std::uint16_t* mn, std::uint16_t* mx,
                                            int pages, int kv_heads, int head_dim,
                                            void*) {
    g_seed_log.push_back("seed(min=" + std::string(mn ? "p" : "null") +
                         ",max=" + std::string(mx ? "p" : "null") +
                         ",pages=" + std::to_string(pages) +
                         ",kvh=" + std::to_string(kv_heads) +
                         ",hd=" + std::to_string(head_dim) + ")");
}
}  // namespace pie_cuda_driver::kernels::layout
