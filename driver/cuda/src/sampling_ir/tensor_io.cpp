// #6 WS-L4 (echo) — tensor-I/O fast-path implementation. See tensor_io.hpp.

#include "sampling_ir/tensor_io.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>

namespace pie_cuda_driver::sampling_ir {

namespace {
// Fail-loud local check (the module is built into the light isolation target with
// no driver-lib include; the executor-grade CUDA_CHECK is picked up at the cut).
inline void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "[tensor_io] CUDA error: %s (%s)\n",
                     cudaGetErrorString(e), what);
        std::abort();
    }
}
}  // namespace
#define TIO_CK(x) ck((x), #x)

namespace {
constexpr std::size_t kDefaultDeviceSlabBytes = 8u  * 1024u * 1024u;  // ~13 [vocab]f32
constexpr std::size_t kDefaultPinnedSlabBytes = 4u  * 1024u * 1024u;  // read-cache outputs
inline std::size_t align_up(std::size_t n, std::size_t a) { return (n + a - 1) / a * a; }
}  // namespace

// ── SlabArena ────────────────────────────────────────────────────────────────

SlabArena::SlabArena(Backing backing, std::size_t default_slab_bytes)
    : backing_(backing), default_slab_bytes_(default_slab_bytes) {}

SlabArena::~SlabArena() {
    for (const Slab& s : slabs_) {
        if (!s.base) continue;
        if (backing_ == Backing::Device) cudaFree(s.base);
        else                             cudaFreeHost(s.base);
    }
}

void SlabArena::add_slab(std::size_t min_bytes) {
    const std::size_t sz = std::max(default_slab_bytes_, min_bytes);
    void* base = nullptr;
    if (backing_ == Backing::Device) {
        TIO_CK(cudaMalloc(&base, sz));
    } else {
        TIO_CK(cudaHostAlloc(&base, sz, cudaHostAllocDefault));
    }
    ++backing_alloc_calls_;  // ONE backing syscall per slab, never per tensor
    slabs_.push_back({base, sz, 0});
}

void* SlabArena::alloc(std::size_t n_bytes) {
    std::lock_guard<std::mutex> guard(mu_);
    const std::size_t sz = align_up(n_bytes == 0 ? kTensorAlign : n_bytes, kTensorAlign);
    // Exact-fit free-list reuse first — zero allocator traffic after warmup.
    auto it = free_list_.find(sz);
    if (it != free_list_.end() && !it->second.empty()) {
        void* p = it->second.back();
        it->second.pop_back();
        return p;
    }
    // Bump from the current slab; grow by a fresh slab if it can't fit.
    if (slabs_.empty() || slabs_.back().used + sz > slabs_.back().size) {
        add_slab(sz);
    }
    Slab& s = slabs_.back();
    void* p = static_cast<char*>(s.base) + s.used;
    s.used += sz;
    block_size_[p] = sz;  // lives for the arena lifetime (a block's size never changes)
    return p;
}

void SlabArena::free(void* ptr) {
    if (!ptr) return;
    std::lock_guard<std::mutex> guard(mu_);
    auto it = block_size_.find(ptr);
    if (it == block_size_.end()) {
        std::fprintf(stderr, "[tensor_io] SlabArena::free of a non-arena ptr\n");
        std::abort();
    }
    free_list_[it->second].push_back(ptr);
}

TensorIoEngine& TensorIoEngine::instance() {
    static TensorIoEngine engine;
    return engine;
}

TensorIoEngine::TensorIoEngine()
    : device_arena_(SlabArena::Backing::Device, kDefaultDeviceSlabBytes),
      pinned_arena_(SlabArena::Backing::PinnedHost, kDefaultPinnedSlabBytes) {
    // Dedicated non-blocking copy stream (the executor.cpp:547 / weight_copy_engine
    // pattern) — copies here never serialize against the forward stream.
    TIO_CK(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));
    // Pinned constant sources so the flag set/clear are stream-ordered H2D copies
    // (a byte-memset can't write the exact u32 value READY=1). These two are tiny
    // one-time engine-lifetime allocs, not per-tensor — fine as direct cudaHostAlloc.
    TIO_CK(cudaHostAlloc(reinterpret_cast<void**>(&h_ready_),
                         sizeof(std::uint32_t), cudaHostAllocDefault));
    TIO_CK(cudaHostAlloc(reinterpret_cast<void**>(&h_clear_),
                         sizeof(std::uint32_t), cudaHostAllocDefault));
    *h_ready_ = kLateFlagReady;
    *h_clear_ = kLateFlagClear;
}

// ── OUTPUT read-cache ────────────────────────────────────────────────────────

void* TensorIoEngine::pinned_alloc(std::size_t n_bytes) {
    // Arena-backed: sub-alloc from the pinned-host slab, NOT a per-output
    // cudaHostAlloc (page-lock syscall) — @ingim's symmetric anti-per-alloc win.
    return pinned_arena_.alloc(n_bytes);
}

void TensorIoEngine::pinned_free(void* host_ptr) {
    pinned_arena_.free(host_ptr);
}

cudaEvent_t TensorIoEngine::eager_d2h_after(void* host_pinned_dst,
                                            const void* device_src,
                                            std::size_t n_bytes,
                                            cudaEvent_t produced) {
    // Order the D2H after the forward's sample-done — the one real data dep. The
    // carrier (delta) stays on the forward stream, so this never false-serializes.
    TIO_CK(cudaStreamWaitEvent(stream_, produced, 0));
    TIO_CK(cudaMemcpyAsync(host_pinned_dst, device_src, n_bytes,
                           cudaMemcpyDeviceToHost, stream_));
    cudaEvent_t ev = nullptr;
    TIO_CK(cudaEventCreateWithFlags(&ev, cudaEventDisableTiming));
    TIO_CK(cudaEventRecord(ev, stream_));
    return ev;
}

cudaEvent_t TensorIoEngine::eager_d2h_outputs(const std::uint64_t* dst_ptrs,
                                              const std::uint32_t* dst_lens,
                                              const void* const* srcs,
                                              const std::size_t* nbytes,
                                              std::size_t count,
                                              cudaEvent_t sample_done) {
    // One wait for the whole batch — all output values are produced by the time the
    // sample/program completes (sample_done). The carrier RETAIN reads pi.sampled on
    // the forward stream (FIFO-safe); this copy-stream batch is the read delta's
    // case-(b) guard protects against t+1's reuse.
    TIO_CK(cudaStreamWaitEvent(stream_, sample_done, 0));
    for (std::size_t k = 0; k < count; ++k) {
        if (dst_ptrs[k] == 0 || srcs[k] == nullptr) continue;  // unbound slot
        if (nbytes[k] > static_cast<std::size_t>(dst_lens[k])) {
            std::fprintf(stderr,
                         "[tensor_io] eager_d2h_outputs: output %zu n=%zu > cap=%u\n",
                         k, nbytes[k], dst_lens[k]);
            std::abort();  // fail-loud bounds-check (host sizes the dst at submit)
        }
        TIO_CK(cudaMemcpyAsync(reinterpret_cast<void*>(dst_ptrs[k]), srcs[k],
                               nbytes[k], cudaMemcpyDeviceToHost, stream_));
    }
    cudaEvent_t t_d2h_done = nullptr;
    TIO_CK(cudaEventCreateWithFlags(&t_d2h_done, cudaEventDisableTiming));
    TIO_CK(cudaEventRecord(t_d2h_done, stream_));
    return t_d2h_done;
}

void TensorIoEngine::enqueue_completion(cudaHostFn_t fn, void* user_data) {
    // Runs on a driver-internal thread once all prior copy-stream work (the
    // eager-D2H) has completed — the (a2) seam fires forward-done here so the host's
    // output().await sees a filled pinned buffer. No CUDA calls inside `fn`.
    TIO_CK(cudaLaunchHostFunc(stream_, fn, user_data));
}

// ── INPUT late-channel ───────────────────────────────────────────────────────

void TensorIoEngine::device_alloc(std::size_t n_bytes, void** out_device_dst,
                                  std::uint32_t** out_device_flag) {
    // ONE arena block holds the [n]-byte buffer (256-aligned) followed by the R12
    // flag word — so the flag is the handle's adjacent metadata (delta's "u32 in
    // the late-tensor handle metadata") at zero extra allocator traffic. No
    // per-tensor cudaMalloc: the block is a sub-alloc from the device slab.
    const std::size_t buf = align_up(n_bytes == 0 ? kTensorAlign : n_bytes, kTensorAlign);
    void* block = device_arena_.alloc(buf + kTensorAlign);  // + one aligned flag slot
    auto* flag = reinterpret_cast<std::uint32_t*>(static_cast<char*>(block) + buf);
    // Clear-on-construct, async on the copy stream (ordered before any later write
    // on the same stream → a reused block can't leak a stale READY).
    TIO_CK(cudaMemsetAsync(flag, 0, sizeof(std::uint32_t), stream_));
    *out_device_dst = block;
    *out_device_flag = flag;
}

void TensorIoEngine::device_free(void* device_dst, std::uint32_t* /*device_flag*/) {
    // The flag lives inside the block (tail), so returning the base frees both.
    device_arena_.free(device_dst);
}

cudaEvent_t TensorIoEngine::write_async(void* device_dst, const void* host_src,
                                        std::size_t n_bytes,
                                        std::uint32_t* device_flag) {
    TIO_CK(cudaMemcpyAsync(device_dst, host_src, n_bytes,
                           cudaMemcpyHostToDevice, stream_));
    // Model A: arm the R12 flag stream-ordered AFTER the payload H2D, so a reader
    // that observes READY is guaranteed to see the landed data (acquire/release via
    // stream order). No host callback.
    TIO_CK(cudaMemcpyAsync(device_flag, h_ready_, sizeof(std::uint32_t),
                           cudaMemcpyHostToDevice, stream_));
    cudaEvent_t ev = nullptr;
    TIO_CK(cudaEventCreateWithFlags(&ev, cudaEventDisableTiming));
    TIO_CK(cudaEventRecord(ev, stream_));
    return ev;
}

void TensorIoEngine::clear_flag(std::uint32_t* device_flag) {
    TIO_CK(cudaMemcpyAsync(device_flag, h_clear_, sizeof(std::uint32_t),
                           cudaMemcpyHostToDevice, stream_));
}

// ── event helpers ────────────────────────────────────────────────────────────

void TensorIoEngine::event_sync(cudaEvent_t ev) {
    TIO_CK(cudaEventSynchronize(ev));
    TIO_CK(cudaEventDestroy(ev));
}

int TensorIoEngine::event_query(cudaEvent_t ev) {
    cudaError_t e = cudaEventQuery(ev);
    if (e == cudaSuccess) return 1;
    if (e == cudaErrorNotReady) return 0;
    ck(e, "cudaEventQuery");
    return 0;
}

}  // namespace pie_cuda_driver::sampling_ir

// ── extern "C" FFI surface ───────────────────────────────────────────────────

using pie_cuda_driver::sampling_ir::TensorIoEngine;

extern "C" {

void* pie_pinned_alloc(std::size_t n_bytes) {
    return TensorIoEngine::instance().pinned_alloc(n_bytes);
}

void pie_pinned_free(void* host_ptr) {
    TensorIoEngine::instance().pinned_free(host_ptr);
}

void pie_device_alloc(std::size_t n_bytes, void** out_device_dst,
                      std::uint32_t** out_device_flag) {
    TensorIoEngine::instance().device_alloc(n_bytes, out_device_dst,
                                            out_device_flag);
}

void pie_device_free(void* device_dst, std::uint32_t* device_flag) {
    TensorIoEngine::instance().device_free(device_dst, device_flag);
}

void* pie_tensor_write_async(void* device_dst, const void* host_src,
                             std::size_t n_bytes, std::uint32_t* device_flag) {
    cudaEvent_t ev = TensorIoEngine::instance().write_async(
        device_dst, host_src, n_bytes, device_flag);
    return static_cast<void*>(ev);
}

void pie_event_sync(void* ev) {
    TensorIoEngine::instance().event_sync(static_cast<cudaEvent_t>(ev));
}

int pie_event_query(void* ev) {
    return TensorIoEngine::instance().event_query(static_cast<cudaEvent_t>(ev));
}

}  // extern "C"
