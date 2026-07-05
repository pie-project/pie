// Phase-3 P2 — device word-publish kernel. See frame_carrier_kernels.hpp.

#include "sampling_ir/frame_carrier_kernels.hpp"

#include <cstdio>
#include <cstdlib>

namespace pie_cuda_driver::sampling_ir {

namespace {
// Host-computed head/tail packed for by-value launch (no extra H2D copy). `n`
// bounded by kMaxPublishChannels (asserted at the launcher).
struct PublishWords {
    std::uint32_t n = 0;
    std::uint64_t pacing = 0;
    std::uint64_t ht[kMaxPublishChannels * 2] = {};  // [head0,tail0, head1,tail1, ...]
};

__global__ void publish_words_kernel(std::uint64_t* __restrict__ words, PublishWords p) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    for (std::uint32_t c = 0; c < p.n; ++c) {
        words[1 + 2 * c] = p.ht[2 * c];      // WordLayout::head(c)
        words[2 + 2 * c] = p.ht[2 * c + 1];  // WordLayout::tail(c)
    }
    __threadfence_system();                  // head/tail visible before pacing (publish-before-wake)
    words[0] = p.pacing;
}
}  // namespace

void launch_publish_words(std::uint64_t* words_dev, std::uint32_t n_channels,
                          const std::uint32_t* head, const std::uint32_t* tail,
                          std::uint64_t pacing, cudaStream_t stream) {
    if (words_dev == nullptr) return;
    if (n_channels > kMaxPublishChannels) {
        std::fprintf(stderr, "[frame_carrier] publish_words: %u channels > max %u\n",
                     n_channels, kMaxPublishChannels);
        std::abort();
    }
    PublishWords p;
    p.n = n_channels;
    p.pacing = pacing;
    for (std::uint32_t c = 0; c < n_channels; ++c) {
        p.ht[2 * c] = head[c];
        p.ht[2 * c + 1] = tail[c];
    }
    publish_words_kernel<<<1, 1, 0, stream>>>(words_dev, p);
}

}  // namespace pie_cuda_driver::sampling_ir
