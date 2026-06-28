// #6 WS8 P2 — inter-pass next-input link device machinery. See next_input.hpp.

#include "sampling_ir/next_input.hpp"

#include <cstdint>

namespace pie_cuda_driver::sampling_ir {

void inject_next_input(const void* sampled, std::span<const NextInputLink> links,
                       void* next_input, cudaStream_t stream) {
    // Per link, copy one i32 token from the producer's sampled[src_row] to the
    // consumer's next_input[dest_pos]. One D2D copy per link (the #11-style gather
    // kernel that fuses these into one launch is a later perf item, like #10's).
    constexpr std::size_t tok = sizeof(std::int32_t);
    const char* src = static_cast<const char*>(sampled);
    char* dst = static_cast<char*>(next_input);
    for (const NextInputLink& l : links) {
        if (l.dest_pos == kIgnorePosition) continue;  // -1 ignore lane
        cudaMemcpyAsync(dst + static_cast<std::size_t>(l.dest_pos) * tok,
                        src + static_cast<std::size_t>(l.src_row) * tok, tok,
                        cudaMemcpyDeviceToDevice, stream);
    }
}

void inject_next_input_after(const void* sampled, std::span<const NextInputLink> links,
                             void* next_input, cudaEvent_t producer_done,
                             cudaStream_t stream) {
    // Gate the inject (and the forward-only graph the caller replays after it on
    // `stream`) behind the producer's sample completion — no host await.
    cudaStreamWaitEvent(stream, producer_done, 0);
    inject_next_input(sampled, links, next_input, stream);
}

}  // namespace pie_cuda_driver::sampling_ir
