#pragma once

// X2 — CUDA frames/mirrors carrier (Runtime–Driver Boundary B5/B6/B8/B9/B11/B13).
//
// Owns the stable frame/mirror/word regions returned by direct instance bind:
//   * frame_base   — a DEVICE frame (`cudaMalloc`). Cells live at
//                    `frame_base + channel offset + ring index`. FIXED for the
//                    instance's lifetime (B6) — a dedicated allocation, never
//                    slab-recycled, so wakers + direct reads have a stable base.
//   * mirror_base  — a PINNED host mirror the host reads committed cells from
//                    (B8/B13 — reads are pure loads from here, never through the
//                    driver).
//   * word_base    — PINNED host ring-index words the host waits on (B9 — the
//                    driver advances the word; a waiter resolves when it passes).
//
// The CARRIER is the direct driver<->inferlet frame transport: on a batch commit
// it D2H-mirrors the instance's committed frame cells into the pinned mirror on a
// dedicated non-blocking copy stream, publishes the pinned ring-index word
// (stream-ordered, B11 publish-before-wake), then runs a completion host callback
// — the X0 wake bridge that resolves the parked host future. The value path never
// travels through the driver; the boundary is addresses plus wakes (C5).
//
// This mirrors the `tensor_io.{hpp,cpp}` substrate (dedicated copy stream +
// async copies + `cudaLaunchHostFunc` completion + an `extern "C" pie_*` surface
// the runtime declares directly), and is built + device-verified in isolation
// (the `test_frame_carrier_device` target) before it composes into the executor.
//
// Channel layout is established once at bind and stays stable through close.

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <unordered_map>
#include <vector>

#include <cuda_runtime.h>

namespace pie_cuda_driver::sampling_ir {

struct WordLayout {
    __host__ __device__ static constexpr std::uint32_t pacing() noexcept { return 0; }
    __host__ __device__ static constexpr std::uint32_t head(std::uint32_t channel) noexcept {
        return 1 + 3 * channel;
    }
    __host__ __device__ static constexpr std::uint32_t tail(std::uint32_t channel) noexcept {
        return 2 + 3 * channel;
    }
    __host__ __device__ static constexpr std::uint32_t poison(std::uint32_t channel) noexcept {
        return 3 + 3 * channel;
    }
    __host__ __device__ static constexpr std::uint32_t words(std::uint32_t channels) noexcept {
        return 1 + 3 * channels;
    }
};

// One host-visible channel's slot in the per-instance pinned mirror (the real
// channel-list layout that supersedes the provisional trace-length sizing). The
// mirror holds a full ring (`cap1` cells) per host-visible channel so run-ahead
// depth-N fires can publish before the host takes; `mirror_off` is the byte
// offset of the channel's ring base within the pinned mirror.
struct FrameChannel {
    std::uint32_t cell_bytes = 0;  // one cell's byte size (shape.numel * dtype)
    std::uint32_t cap1 = 1;        // ring capacity+1 (matches the device ring)
    std::size_t   mirror_off = 0;  // byte offset of this channel's ring in the mirror
};

// Per-instance frame layout. The REAL channel-list layout (boundary.md §2): the
// pinned mirror is sized to the host-visible channels' rings and the pinned words
// are `WordLayout::words(n)` = pacing[0] + head/tail/poison per host-visible
// channel.
// `frame_bytes` is vestigial (the device cells live in the shared
// DeviceChannelRegistry, not a per-instance contiguous frame — the Q1 reconcile);
// kept non-zero so the legacy isolation-test bind still allocates a device probe.
struct FrameLayout {
    std::size_t frame_bytes = 0;   // device frame (vestigial — registry owns cells)
    std::size_t mirror_bytes = 0;  // pinned mirror  (host reads committed cells here)
    std::size_t word_bytes = 0;    // pinned ring-index words (host waits on these)
    std::vector<FrameChannel> host_vis;  // host-visible channels, dense-visible order
};

// One bound instance's frame regions. Addresses are FIXED for the instance's
// lifetime (B6): each is a dedicated backing allocation (never recycled through a
// slab free-list), released only at `close_instance`.
struct FrameInstance {
    std::uint64_t program = 0;
    void*          device_frame = nullptr;  // frame_base  (device, vestigial)
    void*          host_mirror = nullptr;   // mirror_base (pinned host)
    std::uint64_t* host_words = nullptr;    // word_base   (pinned host ring words)
    std::uint64_t* words_dev = nullptr;     // P1: MAPPED device ptr for host_words
    std::size_t    frame_bytes = 0;
    std::size_t    mirror_bytes = 0;
    std::size_t    word_bytes = 0;
    std::vector<FrameChannel> host_vis;     // real channel-list layout (publish target)
};

// Owns the dedicated non-blocking copy stream + the instance table. One per
// process for the MVP (the `TensorIoEngine` pattern); made executor-scoped at the
// reconcile if multiple executors ever coexist.
class FrameCarrierEngine {
public:
    static FrameCarrierEngine& instance();

    // B4 — register a trace: compute its (provisional) frame layout and return a
    // stable 1-based program handle. 0 is never a valid handle.
    std::uint64_t register_program(const std::uint8_t* trace, std::size_t trace_len);

    // B4/B5 — bind an instance: allocate its device frame + pinned mirror + pinned
    // words and write the three bases out. Returns the 1-based instance id, or 0 if
    // `program` is unknown (out params left untouched).
    std::uint64_t bind_instance(std::uint64_t program,
                                std::uint64_t* out_frame_base,
                                std::uint64_t* out_mirror_base,
                                std::uint64_t* out_word_base);

    // B4/B5 (REAL layout) — bind an instance directly from its host-visible
    // channel list (`cell_bytes[i]`, `cap1[i]` for i in [0,n)). Allocates the
    // pinned mirror (sum of per-channel rings) + the pinned words
    // (`WordLayout::words(n)` = pacing[0] + head/tail/poison per channel) and
    // writes the
    // three bases. Returns the 1-based instance id. This is the unification's live
    // bind (the PTIR side owns the channel list); `bind_instance` above stays for
    // the isolation test's provisional path.
    std::uint64_t bind_channels(std::uint32_t n_channels,
                                const std::uint32_t* cell_bytes,
                                const std::uint32_t* cap1,
                                std::uint64_t* out_frame_base,
                                std::uint64_t* out_mirror_base,
                                std::uint64_t* out_word_base);

    // Same as `bind_channels` but keyed by a CALLER-supplied instance id (the wire
    // instance id the runtime assigned). Lets the runtime query this frame's
    // mirror/word bases + layout by that same id (`layout`), so the value path needs
    // no minted-id round-trip (the unification's id-reconcile). Returns `key`.
    std::uint64_t bind_channels_keyed(std::uint64_t key, std::uint32_t n_channels,
                                      const std::uint32_t* cell_bytes,
                                      const std::uint32_t* cap1,
                                      std::uint64_t* out_frame_base,
                                      std::uint64_t* out_mirror_base,
                                      std::uint64_t* out_word_base);

    // COMMIT publish (migration form, B11) — after a fire commits, copy each
    // host-visible channel's committed cell bytes (`src[i]`, already harvested to
    // host) into the pinned mirror ring at ring index `tail_pre[i]` (the slot the
    // fire published into), then publish the post-commit `head[i]`/`tail[i]` into
    // the pinned words and advance pacing[0] to `pacing`. Publish-before-wake: the
    // caller wakes the parked host AFTER this returns (the value is visible first).
    void publish(std::uint64_t instance, std::uint32_t n_channels,
                 const void* const* src, const std::uint32_t* ring_index,
                 const std::uint32_t* head, const std::uint32_t* tail,
                 std::uint64_t pacing);

    // COMMIT publish, DEVICE value path (Phase 3, B11/C5) — same as `publish` but
    // `src[i]` are DEVICE committed-cell pointers: each cell moves by DMA
    // (`cudaMemcpyAsync` D2H on the copy stream) straight into the pinned mirror
    // ring slot `ring_index[i]` — no host bounce buffer (the harvest+memcpy is
    // gone). The copy stream is synced before the head/tail/pacing words are
    // published (publish-before-wake: the value lands before pacing[0]).
    void publish_device(std::uint64_t instance, std::uint32_t n_channels,
                        const void* const* src, const std::uint32_t* ring_index,
                        const std::uint32_t* head, const std::uint32_t* tail,
                        const std::uint32_t* commit_dev,
                        std::uint64_t pacing);

    // LOOKUP (U4 flip) — expose an instance's pinned mirror/word bases + the exact
    // per-channel layout (cell_bytes, cap1, mirror byte offset) so the runtime
    // reads committed cells straight from the mirror with the device's strides.
    // Fills up to `n_channels` entries; returns the true host-visible channel count
    // (0 if `instance` is unknown / has no frame).
    std::uint32_t layout(std::uint64_t instance, std::uint32_t n_channels,
                         std::uint32_t* out_cell_bytes, std::uint32_t* out_cap1,
                         std::uint64_t* out_mirror_off, std::uint64_t* out_mirror_base,
                         std::uint64_t* out_word_base);

    // B6 — release an instance's frame/mirror/word regions. Fail-loud on an unknown
    // or already-closed instance (the tensor_io house style: a loud abort, never a
    // silent trap). The §5.2 grace-period discipline is the caller's.
    void close_instance(std::uint64_t instance);

    // WRITE leg (host inferlet -> device frame): async H2D `n_bytes` of `host_src`
    // into the instance's device frame at `frame_offset`, on the copy stream. The
    // input direction of the transport (bounds-checked <= frame_bytes).
    void carry_in(std::uint64_t instance, const void* host_src, std::size_t n_bytes,
                  std::size_t frame_offset);

    cudaStream_t copy_stream() const { return stream_; }

    // Introspection for the isolation test (asserts bind allocated distinct,
    // non-null bases and close reclaimed them).
    std::size_t live_instances() const;

    FrameCarrierEngine(const FrameCarrierEngine&) = delete;
    FrameCarrierEngine& operator=(const FrameCarrierEngine&) = delete;

private:
    FrameCarrierEngine();
    ~FrameCarrierEngine();

    FrameInstance* lookup(std::uint64_t instance);  // caller holds mu_; nullptr if unknown

    // Shared bind body (caller holds mu_): alloc the pinned mirror+words for `id`.
    std::uint64_t bind_channels_locked(std::uint64_t id, std::uint32_t n_channels,
                                       const std::uint32_t* cell_bytes,
                                       const std::uint32_t* cap1,
                                       std::uint64_t* out_frame_base,
                                       std::uint64_t* out_mirror_base,
                                       std::uint64_t* out_word_base);

    cudaStream_t stream_ = nullptr;  // the dedicated non-blocking copy stream
    mutable std::mutex mu_;
    std::vector<FrameLayout> programs_;             // program id -> layout (1-based)
    std::unordered_map<std::uint64_t, FrameInstance*> instances_;  // instance id -> regions
    std::uint64_t next_program_ = 1;
    std::uint64_t next_instance_ = 1;
};

}  // namespace pie_cuda_driver::sampling_ir

// Legacy frame-carrier isolation surface used only by its device test.
extern "C" {

// B4 — register a trace; returns a 1-based program handle (0 = rejected).
std::uint64_t pie_frame_register(const std::uint8_t* trace, std::size_t trace_len);

// B4/B5 — bind an instance; writes the frame/mirror/word bases. Returns the 1-based
// instance id (0 = unknown program).
std::uint64_t pie_frame_bind(std::uint64_t program, std::uint64_t* out_frame_base,
                             std::uint64_t* out_mirror_base, std::uint64_t* out_word_base);

// B4/B5 (REAL layout) — bind an instance from its host-visible channel list.
// `cell_bytes`/`cap1` are `n_channels`-long arrays (dense host-visible order).
// Writes the frame/mirror/word bases; returns the 1-based instance id.
std::uint64_t pie_frame_bind_channels(std::uint32_t n_channels,
                                      const std::uint32_t* cell_bytes,
                                      const std::uint32_t* cap1,
                                      std::uint64_t* out_frame_base,
                                      std::uint64_t* out_mirror_base,
                                      std::uint64_t* out_word_base);

// Same, keyed by the caller-supplied wire instance id (the unification bind).
std::uint64_t pie_frame_bind_channels_keyed(std::uint64_t key, std::uint32_t n_channels,
                                            const std::uint32_t* cell_bytes,
                                            const std::uint32_t* cap1,
                                            std::uint64_t* out_frame_base,
                                            std::uint64_t* out_mirror_base,
                                            std::uint64_t* out_word_base);

// COMMIT publish (migration form) — copy each channel's committed cell bytes
// (`src[i]`) into the pinned mirror at ring slot `ring_index[i]`, publish
// `head[i]`/`tail[i]` into the pinned words, advance pacing[0] to `pacing`.
void pie_frame_publish(std::uint64_t instance, std::uint32_t n_channels,
                       const void* const* src, const std::uint32_t* ring_index,
                       const std::uint32_t* head, const std::uint32_t* tail,
                       std::uint64_t pacing);

// LOOKUP (U4 flip) — write the instance's pinned mirror/word bases + up to
// `n_channels` per-channel {cell_bytes, cap1, mirror_off}; returns the true
// host-visible channel count (0 = unknown instance / no frame).
std::uint32_t pie_frame_layout(std::uint64_t instance, std::uint32_t n_channels,
                               std::uint32_t* out_cell_bytes, std::uint32_t* out_cap1,
                               std::uint64_t* out_mirror_off,
                               std::uint64_t* out_mirror_base,
                               std::uint64_t* out_word_base);

// B6 — release an instance (fail-loud on unknown/closed).
void pie_frame_close(std::uint64_t instance);

// WRITE leg — async H2D host cells into the device frame (input direction).
void pie_frame_write(std::uint64_t instance, const void* host_src, std::size_t n_bytes,
                     std::size_t frame_offset);

}  // extern "C"
