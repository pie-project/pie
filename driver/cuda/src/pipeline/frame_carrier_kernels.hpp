#pragma once

// Phase-3 P2 — device word-publish launcher. The frame carrier's pinned words
// are `cudaHostAllocMapped` (P1), so a device kernel can store into them on the
// copy stream: the head/tail/pacing publish becomes DEVICE-issued + stream-
// ordered (boundary.md §8 "k_commit_bump stores head/tail into mapped pinned
// words" — realized as a copy-stream epilogue, WITHOUT touching the verified
// pass-atomic commit kernel). The monotonic-produced-count semantics are the
// caller's (host-computed `head`/`tail`, passed by value — no extra H2D copy);
// the device-AUTONOMOUS counter (host out of the per-fire loop) rides with the
// async scheduler regime (P5).

#include <cstdint>
#include <cuda_runtime.h>

namespace pie_cuda_driver::sampling_ir {

// Max host-visible channels published per fire (by-value kernel-param bound).
inline constexpr std::uint32_t kMaxPublishChannels = 64;

// Store `head`/`tail` (WordLayout: words[1+2c]/[2+2c]) then `pacing` (words[0])
// into `words_dev` (the instance's MAPPED word base, device pointer) on `stream`.
// A release threadfence orders the head/tail stores ahead of the pacing store
// (publish-before-wake), mirroring the host atomic-release form.
void launch_publish_words(std::uint64_t* words_dev, std::uint32_t n_channels,
                          const std::uint32_t* head, const std::uint32_t* tail,
                          std::uint64_t pacing, cudaStream_t stream);

// Same as `launch_publish_words`, but only publishes when `*commit_dev != 0`.
// A non-committed fire leaves the mapped words unchanged, so any speculative D2H
// mirror bytes remain invisible to host readers because tail does not advance.
void launch_publish_words_if_committed(std::uint64_t* words_dev,
                                       const std::uint32_t* commit_dev,
                                       std::uint32_t n_channels,
                                       const std::uint32_t* head,
                                       const std::uint32_t* tail,
                                       std::uint64_t pacing,
                                       cudaStream_t stream);

}  // namespace pie_cuda_driver::sampling_ir
