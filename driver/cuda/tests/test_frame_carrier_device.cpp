// X2 — standalone device de-risk for the CUDA frames/mirrors carrier (bravo).
// Verifies, in isolation on a live GPU (no executor / no forward), the real-device
// dual of X1's mock control plane — the primitives the runtime `CudaControlPlane`
// consumes over the `pie_frame_*` FFI:
//   * register → bind allocates a DEVICE frame + PINNED mirror + PINNED ring words,
//     handing back three distinct, non-null bases (B5), fixed for the lifetime (B6).
//   * WRITE leg (pie_frame_write): H2D a known pattern into the device frame.
//   * the CARRIER (pie_frame_carry): D2H-mirrors the committed frame into the
//     pinned mirror, publishes the pinned ring word (word[0]=target), and fires the
//     copy-stream host callback (the X0 wake bridge) once it lands.
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

// The carrier's completion callback (the X0 wake bridge stand-in): flips a flag
// once the copy-stream D2H mirror + ring-word publish have landed.
static void CUDART_CB carry_done_cb(void* ud) {
    *static_cast<int*>(ud) = 1;
}

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

    // ── the CARRIER: mirror the whole committed frame + publish word[0] ───────
    int done_flag = 0;
    eng.carry(inst, /*frame_off=*/0, /*mirror_off=*/0, /*n_bytes=*/0,
              /*word_index=*/0, /*target=*/1, /*forward_evt=*/nullptr,
              carry_done_cb, &done_flag);

    // Drain the copy stream: the H2D, D2H mirror, ring-word publish, and host
    // callback are all enqueued on it, in order.
    CK(cudaStreamSynchronize(eng.copy_stream()));

    EXPECT(done_flag == 1, "carrier completion callback did not fire");

    // The pinned mirror holds the frame's committed cells (host reads it directly).
    const auto* mirror = reinterpret_cast<const std::uint8_t*>(mirror_base);
    EXPECT(std::memcmp(mirror, pattern.data(), N) == 0, "mirror != written frame");

    // The pinned ring word passed the target (B9 — the host waits on this word).
    const auto* words = reinterpret_cast<const std::uint64_t*>(word_base);
    EXPECT(words[0] == 1, "ring word[0] not published to target");

    // ── close reclaims the regions ────────────────────────────────────────────
    eng.close_instance(inst);
    EXPECT(eng.live_instances() == 0, "live_instances != 0 after close");

    std::printf("test_frame_carrier_device: OK\n");
    return 0;
}
