// X2 — standalone device de-risk for the CUDA frames/mirrors carrier (bravo).
// Verifies, in isolation on a live GPU (no executor / no forward), the real-device
// bind-time frame primitives over the `pie_frame_*` isolation surface:
//   * register → bind allocates a DEVICE frame + PINNED mirror + PINNED ring words,
//     handing back three distinct, non-null bases (B5), fixed for the lifetime (B6).
//   * WRITE leg (pie_frame_write): H2D a known pattern into the device frame.
//   * close reclaims the regions (live_instances → 0).
// Exercises the real extern "C" FFI surface the runtime consumes. Needs a GPU;
// device validation is deferred to the later batch per the velocity-shift directive.

#include "sampling_ir/frame_carrier.hpp"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>

using pie_cuda_driver::sampling_ir::FrameCarrierEngine;
using pie_cuda_driver::sampling_ir::WordLayout;

#define CK(x)                                                                  \
    do {                                                                       \
        cudaError_t e_ = (x);                                                  \
        if (e_ != cudaSuccess) {                                               \
            std::fprintf(stderr, "CUDA err %s @ %s:%d\n",                      \
                         cudaGetErrorString(e_), __FILE__, __LINE__);          \
            return 1;                                                          \
        }                                                                      \
    } while (0)

#define EXPECT(cond, msg)                                                      \
    do {                                                                       \
        if (!(cond)) { std::fprintf(stderr, "FAIL: %s\n", (msg)); return 1; }  \
    } while (0)

int main() {
    FrameCarrierEngine& eng = FrameCarrierEngine::instance();

    // ── register → bind ───────────────────────────────────────────────────────
    std::vector<std::uint8_t> trace(64, 0xAB);
    const std::uint64_t prog = eng.register_program(trace.data(), trace.size());
    EXPECT(prog != 0, "register_program returned 0");

    std::uint64_t frame_base = 0, mirror_base = 0, word_base = 0;
    const std::uint64_t inst =
        eng.bind_instance(prog, &frame_base, &mirror_base, &word_base);
    EXPECT(inst != 0, "bind_instance returned 0");
    EXPECT(frame_base != 0 && mirror_base != 0 && word_base != 0, "null base");
    EXPECT(frame_base != mirror_base && mirror_base != word_base &&
               frame_base != word_base,
           "bases not distinct");
    EXPECT(eng.live_instances() == 1, "live_instances != 1 after bind");

    // ── WRITE leg: H2D a known pattern into the device frame ───────────────────
    const std::size_t N = 64;  // == frame_bytes == mirror_bytes (trace len 64)
    std::vector<std::uint8_t> pattern(N);
    for (std::size_t i = 0; i < N; ++i) pattern[i] = static_cast<std::uint8_t>(i + 1);
    eng.carry_in(inst, pattern.data(), N, /*frame_offset=*/0);

    // Drain the copy stream (the H2D is enqueued on it), then read the device frame
    // back directly to confirm the WRITE leg landed the pattern.
    CK(cudaStreamSynchronize(eng.copy_stream()));
    std::vector<std::uint8_t> readback(N, 0);
    CK(cudaMemcpy(readback.data(), reinterpret_cast<const void*>(frame_base), N,
                  cudaMemcpyDeviceToHost));
    EXPECT(std::memcmp(readback.data(), pattern.data(), N) == 0,
           "device frame != written pattern");

    // ── close reclaims the regions ────────────────────────────────────────────
    eng.close_instance(inst);
    EXPECT(eng.live_instances() == 0, "live_instances != 0 after close");

    // ── REAL channel-list bind + publish (the unification's live substrate) ────
    // Two host-visible channels: ch0 cell=8B cap1=4, ch1 cell=16B cap1=2.
    const std::uint32_t cell_bytes[2] = {8, 16};
    const std::uint32_t cap1[2] = {4, 2};
    std::uint64_t fb2 = 0, mirror2 = 0, words2 = 0;
    const std::uint64_t inst2 =
        eng.bind_channels(2, cell_bytes, cap1, &fb2, &mirror2, &words2);
    EXPECT(inst2 != 0, "bind_channels returned 0");
    EXPECT(mirror2 != 0 && words2 != 0, "null mirror/word base");
    // frame_base is vestigial (registry owns cells) → 0 by design.
    EXPECT(fb2 == 0, "frame_base should be vestigial (0)");

    // Publish a fire: ch0 → cell bytes {0x10..}, ring slot 1, head 0 tail 2;
    //                 ch1 → cell bytes {0x20..}, ring slot 0, head 0 tail 1; pacing=1.
    std::vector<std::uint8_t> c0(8), c1(16);
    for (std::size_t i = 0; i < 8; ++i) c0[i] = 0x10 + static_cast<std::uint8_t>(i);
    for (std::size_t i = 0; i < 16; ++i) c1[i] = 0x20 + static_cast<std::uint8_t>(i);
    const void* src[2] = {c0.data(), c1.data()};
    const std::uint32_t ring_index[2] = {1, 0};
    const std::uint32_t head[2] = {0, 0};
    const std::uint32_t tail[2] = {2, 1};
    eng.publish(inst2, 2, src, ring_index, head, tail, /*pacing=*/1);

    // Read back: pacing word[0] == 1; head/tail words; mirror ring slots hold cells.
    const std::uint64_t* w = reinterpret_cast<const std::uint64_t*>(words2);
    EXPECT(w[0] == 1, "pacing word[0] != 1");
    EXPECT(w[WordLayout::head(0)] == 0 && w[WordLayout::tail(0)] == 2,
           "ch0 head/tail words wrong");
    EXPECT(w[WordLayout::poison(0)] == 0, "ch0 poison word wrong");
    EXPECT(w[WordLayout::head(1)] == 0 && w[WordLayout::tail(1)] == 1,
           "ch1 head/tail words wrong");
    EXPECT(w[WordLayout::poison(1)] == 0, "ch1 poison word wrong");
    const std::uint8_t* m = reinterpret_cast<const std::uint8_t*>(mirror2);
    // ch0 ring base = 0, slot 1 → offset 1*8 = 8.
    EXPECT(std::memcmp(m + 8, c0.data(), 8) == 0, "ch0 mirror slot != published");
    // ch1 ring base = 8*4 = 32, slot 0 → offset 32.
    EXPECT(std::memcmp(m + 32, c1.data(), 16) == 0, "ch1 mirror slot != published");

    eng.close_instance(inst2);
    EXPECT(eng.live_instances() == 0, "live_instances != 0 after close(inst2)");

    std::printf("test_frame_carrier_device: OK\n");
    return 0;
}
