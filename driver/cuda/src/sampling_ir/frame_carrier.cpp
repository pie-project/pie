// X2 — CUDA frames/mirrors carrier implementation. See frame_carrier.hpp.
//
// The real-device backing of X1's `FrameAddresses` triple + the direct
// driver<->inferlet frame transport. Built + device-verified in isolation
// (test_frame_carrier_device) before it composes into the executor, exactly like
// the tensor_io.cpp substrate it mirrors.

#include "sampling_ir/frame_carrier.hpp"

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>

namespace pie_cuda_driver::sampling_ir {

namespace {
// Fail-loud local check (the module is built into the light isolation target with
// no driver-lib include; the executor-grade CUDA_CHECK is picked up at the cut) —
// the tensor_io.cpp `ck` idiom.
inline void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "[frame_carrier] CUDA error: %s (%s)\n",
                     cudaGetErrorString(e), what);
        std::abort();
    }
}

// Provisional trace-derived sizing — identical in shape to X1's `MockProgram`
// (frame sized from the trace, tiny mirror/word rings). guru swaps this for the
// real channel-list layout at the reconcile.
constexpr std::size_t kMinFrameBytes = 64;
constexpr std::size_t kMirrorBytes = 64;
constexpr std::size_t kWordBytes = 64;  // 8 u64 ring-index words

// The carrier's copy-stream completion callback. Runs on a driver-internal thread
// once all prior copy-stream work (the D2H mirror) has landed. Publishes the
// pinned ring-index word (B11 — a plain CPU store into pinned host memory, visible
// before the wake), then fires the X0 wake bridge. MUST NOT call CUDA APIs.
struct CarryCtx {
    std::uint64_t* word_slot;  // pinned host ring-index word to publish (B9)
    std::uint64_t  target;     // value to publish
    FrameCarryDone done;       // X0 wake bridge (runtime trampoline), may be null
    void*          user;       // opaque runtime completion context
};

// The completion callback (`cuda_carry_done`) is a stable static — registered
// ONCE by the runtime via `pie_frame_set_carry_done`, NOT threaded per fire. A
// `carry()` whose `done` arg is null uses this registered default, so the (a)
// BRIDGE executor call site passes only `{user_data, word_index}` and never
// handles the fn-ptr. Atomic: set once at init, read on every carry.
std::atomic<FrameCarryDone> g_carry_done{nullptr};

void CUDART_CB frame_carry_host_cb(void* p) {
    auto* ctx = static_cast<CarryCtx*>(p);
    // Publish the committed head as a MONOTONIC-MAX, not a plain store (B11 —
    // publish-before-signal via RELEASE). The head value is baked at ENQUEUE (submit
    // order N, N+1) but written here at COMMIT (async). The carrier's SINGLE copy
    // stream (see the ctor: one `stream_` for every instance + fire) commits carries
    // FIFO, so per-instance commits already land in enqueue order today — the
    // load-bearing invariant that keeps BOTH this head monotonic AND the D2H mirror
    // fresh (the mirror is likewise last-writer-wins). The max is cheap
    // defense-in-depth for the HEAD alone: if two carries for one instance ever
    // committed out of order (a future per-instance / multi-stream topology, e.g.
    // depth-2 run-ahead on separate streams), a plain store could regress the head →
    // the reader's `wake_past` epoch-Filters → stall + lost advance. NOTE the mirror
    // still requires the single-stream FIFO; the max guards the head word only.
    if (ctx->word_slot != nullptr) {
        std::uint64_t cur = __atomic_load_n(ctx->word_slot, __ATOMIC_RELAXED);
        while (ctx->target > cur &&
               !__atomic_compare_exchange_n(ctx->word_slot, &cur, ctx->target,
                                            /*weak=*/true, __ATOMIC_RELEASE,
                                            __ATOMIC_RELAXED)) {
            // CAS failure reloads `cur`; retry until target <= cur or the swap lands.
        }
    }
    if (ctx->done != nullptr) {
        ctx->done(ctx->user);  // wake the parked host future through the X0 table
    }
    delete ctx;
}
}  // namespace

#define FC_CK(x) ck((x), #x)

FrameCarrierEngine& FrameCarrierEngine::instance() {
    static FrameCarrierEngine engine;
    return engine;
}

FrameCarrierEngine::FrameCarrierEngine() {
    // Dedicated non-blocking copy stream (the tensor_io.cpp / weight_copy_engine
    // pattern) — the carrier's copies never serialize against the forward stream.
    //
    // LOAD-BEARING INVARIANT: this is the ONE copy stream every carry (all instances
    // + fires) runs on. Its FIFO ordering is what makes an instance's carries commit
    // in enqueue order → the monotonic head lands monotonically (no regression) AND
    // the D2H mirror stays fresh, even under X4 depth-2 run-ahead; and it's what
    // `close_instance`'s single `cudaStreamSynchronize(stream_)` drains. If this ever
    // becomes multi-stream-per-instance (for D2H parallelism), the head publish must
    // switch to a monotonic-max store and close must drain per-instance events.
    FC_CK(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));
}

FrameCarrierEngine::~FrameCarrierEngine() {
    for (FrameInstance* fi : instances_) {
        if (fi == nullptr) continue;
        if (fi->device_frame) cudaFree(fi->device_frame);
        if (fi->host_mirror) cudaFreeHost(fi->host_mirror);
        if (fi->host_words) cudaFreeHost(fi->host_words);
        delete fi;
    }
    if (stream_) cudaStreamDestroy(stream_);
}

std::uint64_t FrameCarrierEngine::register_program(const std::uint8_t* /*trace*/,
                                                   std::size_t trace_len) {
    std::lock_guard<std::mutex> guard(mu_);
    programs_.push_back(FrameLayout{
        /*frame_bytes=*/std::max(trace_len, kMinFrameBytes),
        /*mirror_bytes=*/kMirrorBytes,
        /*word_bytes=*/kWordBytes,
    });
    return next_program_++;  // 1-based; the just-pushed layout is at id-1
}

std::uint64_t FrameCarrierEngine::bind_instance(std::uint64_t program,
                                                std::uint64_t* out_frame_base,
                                                std::uint64_t* out_mirror_base,
                                                std::uint64_t* out_word_base) {
    std::lock_guard<std::mutex> guard(mu_);
    const std::size_t pidx = static_cast<std::size_t>(program - 1);
    if (program == 0 || pidx >= programs_.size()) {
        return 0;  // unknown program — the runtime turns 0 into a loud Err
    }
    const FrameLayout layout = programs_[pidx];

    auto* fi = new FrameInstance();
    fi->program = program;
    fi->frame_bytes = layout.frame_bytes;
    fi->mirror_bytes = layout.mirror_bytes;
    fi->word_bytes = layout.word_bytes;
    // Dedicated backing allocations — the bases are FIXED for the instance's
    // lifetime (B6), so they are never slab-recycled; freed only at close.
    FC_CK(cudaMalloc(&fi->device_frame, layout.frame_bytes));
    FC_CK(cudaHostAlloc(&fi->host_mirror, layout.mirror_bytes, cudaHostAllocDefault));
    FC_CK(cudaHostAlloc(reinterpret_cast<void**>(&fi->host_words), layout.word_bytes,
                        cudaHostAllocDefault));
    // Zero the pinned ring words so a waiter never observes a stale index (B9).
    std::fill(reinterpret_cast<char*>(fi->host_words),
              reinterpret_cast<char*>(fi->host_words) + layout.word_bytes, 0);

    const std::uint64_t id = next_instance_++;
    if (instances_.size() < id) instances_.resize(id, nullptr);
    instances_[id - 1] = fi;

    *out_frame_base = reinterpret_cast<std::uint64_t>(fi->device_frame);
    *out_mirror_base = reinterpret_cast<std::uint64_t>(fi->host_mirror);
    *out_word_base = reinterpret_cast<std::uint64_t>(fi->host_words);
    return id;
}

FrameInstance* FrameCarrierEngine::lookup(std::uint64_t instance) {
    const std::size_t idx = static_cast<std::size_t>(instance - 1);
    if (instance == 0 || idx >= instances_.size()) return nullptr;
    return instances_[idx];
}

void FrameCarrierEngine::close_instance(std::uint64_t instance) {
    // B6/§5.2 grace — DRAIN before free (the free-before-drain UAF class): every
    // carry's D2H mirror + word publish + host_cb runs on `stream_`, so freeing this
    // instance's device frame / pinned mirror / pinned words while a carry is still
    // in flight lets that pending carry write into FREED memory (and a reader's
    // PinnedRingWord load reads freed pinned words). Sync the copy stream first. It
    // is engine-global + stable, so we sync OUTSIDE the registry lock (B1: never hold
    // a lock across a GPU wait). The runtime's InFlightTracker close-gate is the
    // higher-level B6 grace; this makes the carrier UAF-safe standalone too.
    FC_CK(cudaStreamSynchronize(stream_));
    std::lock_guard<std::mutex> guard(mu_);
    FrameInstance* fi = lookup(instance);
    if (fi == nullptr) {
        std::fprintf(stderr, "[frame_carrier] close of unknown/closed instance %llu\n",
                     static_cast<unsigned long long>(instance));
        std::abort();  // fail-loud (never a silent trap) — the tensor_io house style
    }
    if (fi->device_frame) FC_CK(cudaFree(fi->device_frame));
    if (fi->host_mirror) FC_CK(cudaFreeHost(fi->host_mirror));
    if (fi->host_words) FC_CK(cudaFreeHost(fi->host_words));
    delete fi;
    instances_[instance - 1] = nullptr;
}

void FrameCarrierEngine::carry_in(std::uint64_t instance, const void* host_src,
                                  std::size_t n_bytes, std::size_t frame_offset) {
    std::lock_guard<std::mutex> guard(mu_);
    FrameInstance* fi = lookup(instance);
    if (fi == nullptr) {
        std::fprintf(stderr, "[frame_carrier] carry_in on unknown instance %llu\n",
                     static_cast<unsigned long long>(instance));
        std::abort();
    }
    if (frame_offset + n_bytes > fi->frame_bytes) {
        std::fprintf(stderr,
                     "[frame_carrier] carry_in OOB: off=%zu n=%zu > frame=%zu\n",
                     frame_offset, n_bytes, fi->frame_bytes);
        std::abort();
    }
    FC_CK(cudaMemcpyAsync(static_cast<char*>(fi->device_frame) + frame_offset, host_src,
                          n_bytes, cudaMemcpyHostToDevice, stream_));
}

void FrameCarrierEngine::carry(std::uint64_t instance, std::size_t frame_offset,
                               std::size_t mirror_offset, std::size_t n_bytes,
                               std::size_t word_index, std::uint64_t target,
                               void* forward_evt,
                               FrameCarryDone done, void* user_data) {
    std::lock_guard<std::mutex> guard(mu_);
    FrameInstance* fi = lookup(instance);
    if (fi == nullptr) {
        std::fprintf(stderr, "[frame_carrier] carry on unknown instance %llu\n",
                     static_cast<unsigned long long>(instance));
        std::abort();
    }
    // n_bytes == 0 ⇒ mirror the whole committed frame (layout-agnostic default).
    if (n_bytes == 0) {
        const std::size_t frame_room =
            frame_offset < fi->frame_bytes ? fi->frame_bytes - frame_offset : 0;
        const std::size_t mirror_room =
            mirror_offset < fi->mirror_bytes ? fi->mirror_bytes - mirror_offset : 0;
        n_bytes = std::min(frame_room, mirror_room);
    }
    if (frame_offset + n_bytes > fi->frame_bytes ||
        mirror_offset + n_bytes > fi->mirror_bytes) {
        std::fprintf(stderr,
                     "[frame_carrier] carry OOB: foff=%zu moff=%zu n=%zu "
                     "(frame=%zu mirror=%zu)\n",
                     frame_offset, mirror_offset, n_bytes, fi->frame_bytes,
                     fi->mirror_bytes);
        std::abort();
    }
    if ((word_index + 1) * sizeof(std::uint64_t) > fi->word_bytes) {
        std::fprintf(stderr, "[frame_carrier] carry word idx %zu > words=%zu\n",
                     word_index, fi->word_bytes / sizeof(std::uint64_t));
        std::abort();
    }

    // Cross-stream ordering (B11 wake-correctness): the committed frame cells are
    // produced by the executor's FORWARD stream, NOT this copy stream. Wait on the
    // forward-done event before the D2H so the mirror can never capture pre-commit
    // cells — otherwise the ring-word publish + X0 wake would fire on garbage, and
    // X3/X4 consumers would resolve on an uncommitted frame. null ⇒ no forward
    // stream (the isolation target) ⇒ no wait. The tensor_io opaque-event pattern.
    if (forward_evt != nullptr) {
        FC_CK(cudaStreamWaitEvent(stream_, static_cast<cudaEvent_t>(forward_evt), 0));
    }

    // The carrier: D2H the committed frame cells into the pinned mirror on the copy
    // stream (B8/B13 — the host reads the mirror directly, never through us).
    if (n_bytes > 0) {
        FC_CK(cudaMemcpyAsync(static_cast<char*>(fi->host_mirror) + mirror_offset,
                              static_cast<char*>(fi->device_frame) + frame_offset,
                              n_bytes, cudaMemcpyDeviceToHost, stream_));
    }
    // Stream-ordered AFTER the mirror: publish the pinned ring word + wake (B11).
    // `done == null` ⇒ the once-registered `g_carry_done` (the (a) BRIDGE executor
    // passes null; the pre-bridge runtime may pass the fn directly — both resolve
    // to the same stable `cuda_carry_done`).
    FrameCarryDone effective_done = done ? done : g_carry_done.load(std::memory_order_acquire);
    auto* ctx = new CarryCtx{fi->host_words + word_index, target, effective_done, user_data};
    FC_CK(cudaLaunchHostFunc(stream_, frame_carry_host_cb, ctx));
}

std::size_t FrameCarrierEngine::live_instances() const {
    std::lock_guard<std::mutex> guard(mu_);
    std::size_t n = 0;
    for (const FrameInstance* fi : instances_) {
        if (fi != nullptr) ++n;
    }
    return n;
}

}  // namespace pie_cuda_driver::sampling_ir

// ── extern "C" FFI surface ───────────────────────────────────────────────────

using pie_cuda_driver::sampling_ir::FrameCarrierEngine;

extern "C" {

std::uint64_t pie_frame_register(const std::uint8_t* trace, std::size_t trace_len) {
    return FrameCarrierEngine::instance().register_program(trace, trace_len);
}

std::uint64_t pie_frame_bind(std::uint64_t program, std::uint64_t* out_frame_base,
                             std::uint64_t* out_mirror_base, std::uint64_t* out_word_base) {
    return FrameCarrierEngine::instance().bind_instance(program, out_frame_base,
                                                        out_mirror_base, out_word_base);
}

void pie_frame_close(std::uint64_t instance) {
    FrameCarrierEngine::instance().close_instance(instance);
}

void pie_frame_write(std::uint64_t instance, const void* host_src, std::size_t n_bytes,
                     std::size_t frame_offset) {
    FrameCarrierEngine::instance().carry_in(instance, host_src, n_bytes, frame_offset);
}

void pie_frame_carry(std::uint64_t instance, std::size_t frame_offset,
                     std::size_t mirror_offset, std::size_t n_bytes,
                     std::size_t word_index, std::uint64_t target,
                     void* forward_evt,
                     void (*done)(void*), void* user_data) {
    FrameCarrierEngine::instance().carry(instance, frame_offset, mirror_offset, n_bytes,
                                         word_index, target, forward_evt, done, user_data);
}

void pie_frame_set_carry_done(void (*done)(void*)) {
    // Register the stable completion callback ONCE (the runtime's cuda_carry_done).
    // A `pie_frame_carry` with `done == null` then uses it — so the executor never
    // threads the fn-ptr per fire.
    pie_cuda_driver::sampling_ir::g_carry_done.store(done, std::memory_order_release);
}

}  // extern "C"
