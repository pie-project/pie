#pragma once

// #6 WS-L4 (echo) — host<->driver tensor-I/O fast-path (the @ingim direct-FFI model).
//
// SEPARABLE module (the next_input.hpp / group.hpp pattern): the dedicated copy
// stream + the pinned/device buffer primitives + the async H2D/D2H copies, built
// and device-verified in isolation BEFORE composing into the executor at golf's
// WIT freeze. The thin executor wiring (the eager-D2H hook at the sample tail, the
// HostLate binding source, the R12 acquire-wait) lands at the cut; this module is
// the surface-agnostic substrate underneath it.
//
// Design (manager-locked, R11/R12/R13):
//   * Copy stream is DRIVER-INTERNAL — one non-blocking stream per executor
//     (the executor.cpp:547 / weight_copy_engine pattern). The host never threads
//     a raw cudaStream_t; it holds only an opaque event handle.
//   * WRITE (late mu/mask, H2D — cut #2): async H2D on the copy stream, then the
//     R12 device dirty-flag SELF-ARMS (a 4-byte H2D of READY, stream-ordered AFTER
//     the payload) — "Model A", no host callback, ready => data-landed by stream
//     order. Returns a completion event for the host's OPTIONAL 1a-sync /
//     miss-deadline poll. The 1c device-gate spins on the flag (acquire), no event.
//   * READ-CACHE (output->tensor, D2H — cut #1, verify-critical): eager_d2h_after
//     D2H's the program's output (in ws) into the Tensor's host PINNED buffer on
//     the copy stream, ordered after the forward sample-done event (the one data
//     dep). output().await drains the returned event; tensor.read = host memcpy.
//     This generalizes the existing executor.cpp:3347 pi.sampled->sampled_pinned_buf
//     D2H, retargeted from cublas.stream() to the copy stream.
//   * The late-input DEVICE buffer + its R12 flag are CO-ALLOCATED here
//     (device_alloc), handle-lifetime (R11 create-once). The flag is the late
//     handle's metadata word — echo-owned end-to-end (alloc / self-arm / poll);
//     delta's carrier is the separate device->device leg and uses
//     cudaStreamWaitEvent, not the flag.

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <unordered_map>
#include <vector>

#include <cuda_runtime.h>

namespace pie_cuda_driver::sampling_ir {

// R12 per-binding dirty-flag states (a device u32 word).
constexpr std::uint32_t kLateFlagClear = 0u;  // clear-on-bind
constexpr std::uint32_t kLateFlagReady = 1u;  // set after the H2D payload lands

// Device-buffer alignment — 256 B for coalesced D2H/H2D.
constexpr std::size_t kTensorAlign = 256u;

// One slab + size-class free-list sub-allocator, shared substrate for BOTH
// dedicated tensor-I/O arenas (@ingim's directive: never per-tensor cudaMalloc /
// cudaHostAlloc — those device-malloc / page-lock syscalls stall the stream and
// fragment memory if done per inferlet-tensor construction every step). The
// backing syscall runs ONCE per slab (init + rare growth); each per-tensor
// alloc/free is a cheap bump + exact-fit size-class free-list reuse. Inferlet
// mask/bias + sampler-output tensors are near-uniform in size, so the exact-fit
// free-list gives near-perfect reuse with zero allocator traffic after warmup.
//
// Two SEPARATE instances (isolated from each other + from the KV/RS working-set
// arena, so inferlet-tensor churn never fragments the main working set):
//   * Device       — cudaMalloc-backed; INPUT late mask/bias/input buffers (cut #2).
//   * PinnedHost    — cudaHostAlloc-backed; OUTPUT read-cache buffers (cut #1).
class SlabArena {
public:
    enum class Backing { Device, PinnedHost };

    SlabArena(Backing backing, std::size_t default_slab_bytes);
    ~SlabArena();

    SlabArena(const SlabArena&) = delete;
    SlabArena& operator=(const SlabArena&) = delete;

    // 256-aligned sub-alloc; reuses a freed same-size block if available.
    void* alloc(std::size_t n_bytes);
    // Return a block to its size-class free-list (no backing free until teardown).
    void  free(void* ptr);

    // Introspection (tests assert no per-tensor backing syscall): the backing
    // alloc count == the number of slabs, NOT the number of allocs.
    std::size_t backing_alloc_calls() const { return backing_alloc_calls_; }
    std::size_t slab_count() const { return slabs_.size(); }

private:
    struct Slab { void* base = nullptr; std::size_t size = 0; std::size_t used = 0; };
    void add_slab(std::size_t min_bytes);

    Backing     backing_;
    std::size_t default_slab_bytes_;
    std::size_t backing_alloc_calls_ = 0;
    std::vector<Slab> slabs_;
    std::unordered_map<void*, std::size_t> block_size_;            // ptr -> rounded size
    std::unordered_map<std::size_t, std::vector<void*>> free_list_;  // size -> free blocks
    std::mutex mu_;
};

// Owns the dedicated copy stream + the H2D/D2H primitives. One per executor
// (process-global for the M=1 MVP, like executor.cpp's sampled_pinned_buf; made
// executor-scoped at the cut if multiple executors ever coexist).
class TensorIoEngine {
public:
    static TensorIoEngine& instance();

    // ── OUTPUT read-cache (cut #1, verify-critical) ───────────────────────────
    // Host pinned buffer = the eager-D2H destination + the Tensor resource backing.
    void* pinned_alloc(std::size_t n_bytes);
    void  pinned_free(void* host_ptr);
    // D2H device_src -> host_pinned_dst on the copy stream, ordered AFTER `produced`
    // (the forward sample-done event) — the single real data dependency. Returns a
    // completion event the executor/host syncs before tensor.read.
    cudaEvent_t eager_d2h_after(void* host_pinned_dst, const void* device_src,
                                std::size_t n_bytes, cudaEvent_t produced);

    // Batch eager-D2H of program output VALUES into their host pinned dsts (the
    // cut #1 output→tensor fast-path). For each output k: D2H srcs[k] → dst_ptrs[k]
    // (nbytes[k] bytes, bounds-checked ≤ dst_lens[k]) on the copy stream, ordered
    // after `sample_done` (the forward sample-done event). Returns t_d2h_done (a
    // copy-stream event) for the host completion wait + delta's case-(b) WAR guard.
    // dst_ptrs[k]==0 or srcs[k]==null ⇒ that output is skipped (unbound slot).
    cudaEvent_t eager_d2h_outputs(const std::uint64_t* dst_ptrs,
                                  const std::uint32_t* dst_lens,
                                  const void* const* srcs,
                                  const std::size_t* nbytes,
                                  std::size_t count,
                                  cudaEvent_t sample_done);

    // Enqueue a host callback on the copy stream — runs AFTER all prior copy-stream
    // work (the eager-D2H) completes. Used by the executor's (a2) seam to fire the
    // forward-done send only once the pinned buffer is filled (full overlap; the
    // fire thread returns immediately). The callback must not call CUDA APIs.
    void enqueue_completion(cudaHostFn_t fn, void* user_data);

    // ── INPUT late-channel (cut #2, post-MATCH=true) ──────────────────────────
    // Co-allocate the late-input device buffer + its R12 flag (flag cleared).
    void device_alloc(std::size_t n_bytes, void** out_device_dst,
                      std::uint32_t** out_device_flag);
    void device_free(void* device_dst, std::uint32_t* device_flag);
    // Async H2D host_src -> device_dst on the copy stream, then self-arm the flag
    // (Model A). Returns a completion event for optional host sync.
    cudaEvent_t write_async(void* device_dst, const void* host_src,
                            std::size_t n_bytes, std::uint32_t* device_flag);
    // R12 clear-on-bind: reset the flag to CLEAR (copy-stream ordered before the
    // next write so a stale READY can never leak across fires).
    void clear_flag(std::uint32_t* device_flag);

    // ── event helpers ─────────────────────────────────────────────────────────
    void event_sync(cudaEvent_t ev);   // synchronize + reclaim (destroy)
    int  event_query(cudaEvent_t ev);  // 1 = ready, 0 = pending (miss-deadline poll)

    cudaStream_t copy_stream() const { return stream_; }

    // Arena introspection passthrough (tests assert no per-tensor backing syscall).
    std::size_t device_backing_alloc_calls() const { return device_arena_.backing_alloc_calls(); }
    std::size_t pinned_backing_alloc_calls() const { return pinned_arena_.backing_alloc_calls(); }

    TensorIoEngine(const TensorIoEngine&) = delete;
    TensorIoEngine& operator=(const TensorIoEngine&) = delete;

private:
    TensorIoEngine();

    cudaStream_t   stream_  = nullptr;  // the dedicated non-blocking copy stream
    std::uint32_t* h_ready_ = nullptr;  // pinned source for the READY flag set
    std::uint32_t* h_clear_ = nullptr;  // pinned source for the CLEAR flag set
    SlabArena device_arena_;            // INPUT late mask/bias buffers (cut #2)
    SlabArena pinned_arena_;            // OUTPUT read-cache buffers (cut #1)
};

}  // namespace pie_cuda_driver::sampling_ir

// ── extern "C" FFI surface (host-declared in Rust; cbindgen-shaped like the
//    pie_loader_* / pie_pinned_* surface). bravo's host tensor resource calls
//    these directly into pie_driver_cuda_lib, bypassing the IPC/driver channel. ──
extern "C" {

// OUTPUT read-cache (cut #1)
void* pie_pinned_alloc(std::size_t n_bytes);
void  pie_pinned_free(void* host_ptr);

// INPUT late-channel (cut #2). device_alloc co-allocates the R12 flag (echo-owned).
void  pie_device_alloc(std::size_t n_bytes, void** out_device_dst,
                       std::uint32_t** out_device_flag);
void  pie_device_free(void* device_dst, std::uint32_t* device_flag);

// Returns an opaque completion event (a cudaEvent_t) for optional host sync.
void* pie_tensor_write_async(void* device_dst, const void* host_src,
                             std::size_t n_bytes, std::uint32_t* device_flag);

// Event helpers (the opaque handle returned by pie_tensor_write_async).
void  pie_event_sync(void* ev);   // 1a-sync wait + reclaim
int   pie_event_query(void* ev);  // miss-deadline poll: 1 = ready, 0 = pending

}  // extern "C"
