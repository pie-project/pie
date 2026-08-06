// mtl4_context.mm — Obj-C++ implementation of the Metal-4 wrapper scaffold.
//
// Implements RawMetalContext on top of the Metal-4 objects verified working in
// beta's mtl4probe.mm: MTL4CommandQueue / double-buffered MTL4CommandAllocator /
// MTL4CommandBuffer / MTL4ComputeCommandEncoder / MTL4ArgumentTable / MTLResidencySet /
// MTL4Compiler (runtime newLibraryWithSource — no offline metallib needed on this box).
//
// Static CPU-visible resources use one Shared placement heap. KV, state, and
// scratch use Private placement-sparse VAs backed by Shared 256 MiB heap chunks.
// Argument tables are built ONCE per (Kernel, layer) dispatch instance (I2); only IO
// slot CONTENTS change per token (I1) so the encoded command buffer stays byte-identical.

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#import <dispatch/dispatch.h>

#include "mtl4_context.hpp"
#include "observability.hpp"
#include "elastic.hpp"

#include <chrono>
#include <sys/stat.h>
#include <mach/mach.h>
#include <mach/mach_host.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <strings.h>
#include <unistd.h>
#include <algorithm>
#include <atomic>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace pie::metal {

MetalStorageFacts query_metal_storage_facts() {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (device == nil) {
        throw std::runtime_error("no Metal device");
    }
    const MTLResourceOptions options =
        MTLResourceStorageModeShared | MTLResourceHazardTrackingModeUntracked;
    const MTLSizeAndAlign sa = [device
        heapBufferSizeAndAlignWithLength:1
        options:options];
    const long page = ::sysconf(_SC_PAGESIZE);
    return MetalStorageFacts{
        .alignment = static_cast<std::uint32_t>(std::max<NSUInteger>(1, sa.align)),
        .page_size = static_cast<std::uint32_t>(page > 0 ? page : 1),
    };
}

namespace {
using clk = std::chrono::high_resolution_clock;
inline double nowms() {
    return std::chrono::duration<double, std::milli>(clk::now().time_since_epoch()).count();
}
inline size_t align_up(size_t v, size_t a) { return (v + (a - 1)) & ~(a - 1); }
// The arg-table key is the FLAT DISPATCH ORDINAL alone (beta's DAG walker, 0..321,
// or -1 for a singleton). Kind is decorative: within one layer Rms/Residual recur,
// so (kind, layer) collides — the ordinal is the only unique key.
inline int argkey(int ordinal) { return ordinal; }

// MSL caps `[[buffer(n)]]` at n <= 30, so a kernel can never reference more than
// 31 buffer bindings and the argument tables are sized to exactly that. Binds
// past it are a driver bug, not a capacity shortfall -- an argument table
// silently ignores an out-of-range index, so they are rejected loudly instead.
inline constexpr std::uint32_t kMaxArgBinds = 31;
}  // namespace

// ── Per-step encoder state (transient, lives across one run_step) ─────────────
struct StepState {
    id<MTL4ComputeCommandEncoder> en = nil;
    RawMetalContext::Impl*        ctx = nullptr;
    // Per-dispatch attribution, off unless PIE_METAL_DISPATCH_TRACE is set.
    // A fire is a hundred-odd dispatches behind one fence, so "the GPU spent
    // 81 ms" is not an answer to "which kernel"; this brackets every dispatch
    // with a timestamp and reports the shares. It lives in the encoder rather
    // than in a family's encode function so that one implementation covers
    // every model and every kernel.
    void* trace_heap = nullptr;
    std::uint32_t trace_slots = 0;
    std::uint32_t trace_n = 0;
    std::vector<std::string> trace_labels;
    std::string trace_pso;
};

// `PIE_METAL_DISPATCH_TRACE=<n>` prints the accumulated table every n fires.
int dispatch_trace_every() {
    static const int n = [] {
        const char* e = getenv("PIE_METAL_DISPATCH_TRACE");
        if (e == nullptr || *e == '\0') return 0;
        const int v = atoi(e);
        return v > 0 ? v : 1;
    }();
    return n;
}

// ── RawMetalContext::Impl — owns the Metal-4 device objects ───────────────────
struct RawMetalContext::Impl {
    // Heap slots memoized by the argument-table slot they are bound to, so a
    // constant rebound every fire is allocated once. See `const_slot`.
    std::unordered_map<std::uint64_t, SlotHandle> const_slots;
    id<MTLDevice>            dev   = nil;
    id<MTL4CommandQueue>     queue = nil;
    id<MTL4CommandQueue>     mapping_queue = nil;
    id<MTL4CommandAllocator> alloc[2] = {nil, nil};   // double-buffered
    id<MTLHeap>              heap  = nil;
    id<MTLResidencySet>      rs    = nil;
    id<MTL4Compiler>         compiler = nil;
    id<MTL4PipelineDataSetSerializer> pipeline_serializer = nil;
    id<MTLSharedEvent>       event = nil;
    uint64_t                 ev_val = 0;

    // Most recent Metal 4 commit feedback. Metal delivers it on its own queue,
    // so every read/write goes through the mutex.
    // Commit feedback handlers can fire after the context is torn down, so the
    // state they touch is kept alive by the block itself rather than by Impl.
    struct FeedbackSlot {
        std::mutex        mutex;
        GpuCommitFeedback value;
    };
    std::shared_ptr<FeedbackSlot> feedback = std::make_shared<FeedbackSlot>();

    // The one submit primitive. Every path that puts work on `queue` goes
    // through here: it attaches the commit-feedback handler, submits all `n`
    // command buffers in a single `commit:count:options:`, and signals the
    // queue timeline once for the whole batch.
    //
    // `wait_value > 0` makes the batch wait on the timeline first (GPU-side
    // serialization for the autoregressive dependency). Returns the value the
    // batch signals on completion.
    uint64_t commit_and_signal(const id<MTL4CommandBuffer> __strong* cbs,
                               NSUInteger n,
                               uint64_t wait_value);

    size_t heap_size = 0;
    size_t bump      = 0;             // running heap offset
    bool   resident  = false;         // make_resident() idempotency guard

    NSMutableArray*           retained = nil;  // keeps PSOs / sub-buffers alive
    std::unordered_set<void*> retained_psos;
    NSMutableDictionary*      argtables = nil;  // NSNumber(argkey) -> id<MTL4ArgumentTable>
    // Key is (ordinal << 8) | bind_index -> the address bound there. One
    // container, not two: the set that used to sit beside this was exactly its
    // key set, so every bind paid two hashes and every released ordinal paid two
    // erases per probe, over a table the prefill grows to ~123k entries.
    std::unordered_map<uint64_t, uint64_t> bound_arg_addresses;
    std::unordered_map<void*, size_t> external_allocations;
    bool saw_ptir_compile = false;
    bool last_ptir_fast_math_disabled = false;
    uint64_t surfaced_feedback_error = 0;
    std::atomic<bool> force_wait_timeout_once{false};
    // Set when a wait was abandoned. Sticky: see run_steps.
    bool wedged = false;

    // Standalone-buffer allocation accounting (all non-heap buffers,
    // including buffers retained by the transient pool).
    size_t standalone_count = 0;
    size_t standalone_bytes = 0;

    static constexpr size_t kSparseTileBytes = 16ull * 1024;
    static constexpr size_t kElasticChunkBytes = 256ull * 1024 * 1024;

    struct ElasticChunk {
        void* heap = nullptr;
        void* alias = nullptr;
        size_t size = 0;
        size_t mapped = 0;
    };
    struct ElasticAllocation {
        void* buffer = nullptr;
        size_t virtual_bytes = 0;
        size_t committed_bytes = 0;
        std::vector<ElasticChunk> chunks;
    };
    struct PendingElasticRelease {
        uint64_t event_value = 0;
        std::vector<void*> objects;
    };
    std::unordered_map<void*, ElasticAllocation> elastic_allocations;
    std::vector<PendingElasticRelease> pending_elastic_releases;
    size_t elastic_budget_bytes = 0;
    size_t elastic_pressure_floor_bytes = 0;
    size_t elastic_reserved_bytes = 0;
    size_t elastic_committed_bytes = 0;
    std::shared_ptr<std::atomic<std::uint32_t>> memory_pressure_level =
        std::make_shared<std::atomic<std::uint32_t>>(0);
    dispatch_source_t memory_pressure_source = nullptr;

    struct TransientAllocation {
        size_t size_class = 0;
        bool in_use = false;
    };
    std::map<size_t, std::vector<SlotHandle>> transient_free;
    std::unordered_map<void*, TransientAllocation> transient_allocations;
    TransientBufferPoolStats transient_stats{
        .capacity_bytes = size_t{1} << 30,
    };

    StepState step;  // active step (during run_step)

    // ── Continuous-async keepalive (separate queue + background thread) ──
    id<MTL4CommandQueue>     ka_queue = nil;
    id<MTL4CommandAllocator> ka_alloc = nil;
    id<MTLComputePipelineState> ka_pso = nil;
    id<MTLBuffer>            ka_sink  = nil;   // dummy device sink (atomic, never read)
    id<MTLBuffer>            ka_iters = nil;   // constant spin count
    id<MTL4ArgumentTable>    ka_at4   = nil;
    id<MTLSharedEvent>       ka_event = nil;
    std::atomic<bool>        ka_run{false};
    std::thread              ka_thread;

    id<MTL4ArgumentTable> argtable_for(int ordinal, bool create);
    void collect_elastic_releases();
    // Wait for the queue to reach `value`, bounded. False means the driver
    // gave up and the context is finished; see the definition.
    bool await_event(uint64_t value);
    void drain_mapping_through(uint64_t value);
    uint64_t schedule_mapping(
        id<MTLBuffer> buffer,
        id<MTLHeap> heap,
        const MTL4UpdateSparseBufferMappingOperation& operation);
    size_t effective_elastic_budget_bytes() const;
};

size_t RawMetalContext::Impl::effective_elastic_budget_bytes() const {
    const std::uint32_t level =
        memory_pressure_level->load(std::memory_order_acquire);
    if (level == 0) return elastic_budget_bytes;
    const size_t pressure_limit =
        level >= 2
            ? elastic_pressure_floor_bytes
            : std::max(elastic_pressure_floor_bytes, elastic_budget_bytes / 2);
    return std::min(elastic_budget_bytes, pressure_limit);
}

void RawMetalContext::Impl::collect_elastic_releases() {
    const uint64_t signaled = event.signaledValue;
    auto out = pending_elastic_releases.begin();
    for (auto it = pending_elastic_releases.begin();
         it != pending_elastic_releases.end();
         ++it) {
        if (it->event_value <= signaled) {
            bool residency_changed = false;
            for (void* object : it->objects) {
                id value = (__bridge id)object;
                if ([value conformsToProtocol:@protocol(MTLHeap)]) {
                    [rs removeAllocation:value];
                    residency_changed = true;
                }
                [retained removeObject:value];
            }
            if (residency_changed) [rs commit];
            continue;
        }
        if (out != it) *out = std::move(*it);
        ++out;
    }
    pending_elastic_releases.erase(out, pending_elastic_releases.end());
}

// How long the driver waits for a completion fence before it stops waiting.
// Split into probes so a step that is merely slow is still counted as slow the
// moment it passes the first one, while the budget is what decides to stop.
//
// A step here is a command buffer for one token or one prefill chunk, and the
// slowest measured on this machine -- a 192-token prefill through a 30B
// mixture -- is about 200 ms. Two orders of magnitude past that is not a slow
// GPU; it is one that is not coming back, and waiting on it produced no
// diagnostic whatsoever. The symptom that brought this here was twenty-two
// minutes of silence inside `waitUntilSignaledValue`, with the driver's own
// "timed out before its completion fence" message sitting unreachable behind
// a bare retry loop.
static constexpr int kWaitProbeMs = 5000;
static constexpr int kWaitProbes  = 12;

bool RawMetalContext::Impl::await_event(uint64_t value) {
    if (event == nil) return true;
    for (int probe = 0; probe < kWaitProbes; ++probe) {
        if (force_wait_timeout_once.exchange(false)) break;
        if ([event waitUntilSignaledValue:value timeoutMS:kWaitProbeMs]) return true;
        if (probe == 0) m0_timing_counters().record_forward_wait_timeout();
    }
    // Sticky. A command buffer that has not signalled may still be executing,
    // and both things this driver does after a wait -- resetting the allocator
    // a buffer was drawn from, and releasing the heaps it reads -- are unsafe
    // while that is true. There is no way back from here, so say so once and
    // refuse everything after, rather than waiting forever in silence.
    if (!wedged) {
        wedged = true;
        fprintf(stderr,
                "[pie-metal] the GPU did not reach event %llu within %d ms; this context is "
                "abandoned because its command buffers may still be running\n",
                (unsigned long long)value, kWaitProbes * kWaitProbeMs);
    }
    return false;
}

void RawMetalContext::Impl::drain_mapping_through(uint64_t value) {
    if (value != 0 && !await_event(value)) {
        // Deliberately NOT collecting: `collect_elastic_releases` hands heaps
        // back, and the wait that would have proved nothing is still reading
        // them is the one that just failed. Leaking them is the safe half of a
        // bad situation; the destructor that used to sit here forever was not.
        return;
    }
    collect_elastic_releases();
}

uint64_t RawMetalContext::Impl::schedule_mapping(
    id<MTLBuffer> buffer,
    id<MTLHeap> mapping_heap,
    const MTL4UpdateSparseBufferMappingOperation& operation) {
    collect_elastic_releases();
    const uint64_t wait_value = ev_val;
    if (wait_value != 0) {
        [mapping_queue waitForEvent:event value:wait_value];
    }
    [mapping_queue updateBufferMappings:buffer
                                   heap:mapping_heap
                             operations:&operation
                                  count:1];
    const uint64_t done_value = ++ev_val;
    [mapping_queue signalEvent:event value:done_value];
    [queue waitForEvent:event value:done_value];
    return done_value;
}

id<MTL4ArgumentTable> RawMetalContext::Impl::argtable_for(int ordinal, bool create) {
    NSNumber* key = @(argkey(ordinal));
    id<MTL4ArgumentTable> t = argtables[key];
    if (t == nil && create) {
        MTL4ArgumentTableDescriptor* ad = [MTL4ArgumentTableDescriptor new];
        ad.maxBufferBindCount = kMaxArgBinds;
        NSError* e = nil;
        t = [dev newArgumentTableWithDescriptor:ad error:&e];
        if (t == nil) {
            fprintf(stderr, "[pie-metal] argtable create failed: %s\n",
                    e.localizedDescription.UTF8String);
            return nil;
        }
        argtables[key] = t;
    }
    return t;
}

// ── StepEncoder bridges ───────────────────────────────────────────────────────
void StepEncoder::set_pso(Pso pso) {
    auto* s = static_cast<StepState*>(impl_);
    id<MTLComputePipelineState> p = (__bridge id<MTLComputePipelineState>)pso.obj;
    [s->en setComputePipelineState:p];
    if (dispatch_trace_every() > 0) {
        s->trace_pso = p.label != nil ? p.label.UTF8String : "<unlabelled>";
    }
}
void StepEncoder::set_argtable(Kernel k, int ordinal) {
    (void)k;  // decorative tag; ordinal is the key
    set_argtable_ordinal(ordinal);
}
void StepEncoder::set_argtable_ordinal(int ordinal) {
    auto* s = static_cast<StepState*>(impl_);
    id<MTL4ArgumentTable> t = s->ctx->argtable_for(ordinal, /*create=*/false);
    if (t == nil) {
        fprintf(stderr, "[pie-metal] no argument table bound for ordinal=%d\n", ordinal);
        return;
    }
    [s->en setArgumentTable:t];
}
void StepEncoder::dispatch(Grid grid, Threadgroup tg) {
    auto* s = static_cast<StepState*>(impl_);
    // Relaxed granularity, so the pair costs no encoder split; the cost of the
    // marks themselves is charged to whichever dispatch they bracket, which is
    // the same for all of them and so does not move the shares.
    const bool trace = dispatch_trace_every() > 0 && s->trace_heap != nullptr &&
                       2 * (s->trace_n + 1) <= s->trace_slots;
    if (trace) mark_timestamp(s->trace_heap, 2 * s->trace_n, /*precise=*/true);
    [s->en dispatchThreads:MTLSizeMake(grid.x, grid.y, grid.z)
        threadsPerThreadgroup:MTLSizeMake(tg.x, tg.y, tg.z)];
    if (trace) {
        mark_timestamp(s->trace_heap, 2 * s->trace_n + 1, /*precise=*/true);
        s->trace_labels.push_back(s->trace_pso);
        ++s->trace_n;
    }
}
// Map the pure-C++ BarrierVisibility to MTL4VisibilityOptions, honoring a one-shot
// `PIE_BARRIER_VIS=none|device` global override (delta's visibility sweep): when set it
// forces ALL barriers regardless of the per-call argument; when absent the per-call arg
// wins (beta's per-edge hazard model: Device for true heap-RAW, None for ordering-only).
static MTL4VisibilityOptions resolve_barrier_vis(BarrierVisibility req) {
    static const int override_mode = [] {
        return -1;
    }();
    const int mode = override_mode >= 0
                         ? override_mode
                         : (req == BarrierVisibility::Device ? 1 : 0);
    return mode == 1 ? MTL4VisibilityOptionDevice : MTL4VisibilityOptionNone;
}
void StepEncoder::barrier(BarrierVisibility vis) {
    auto* s = static_cast<StepState*>(impl_);
    // Intra-encoder (intra-pass) dispatch→dispatch RAW/WAR hazard ordering. MUST be the
    // *EncoderStages* variant — barrierAfterQueueStages is a cross-command-buffer/queue
    // barrier and does NOT order dispatches within the same compute encoder (verified:
    // queue-stage barrier let layer-0 RMSNorm read stale embed → non-deterministic garbage).
    // visibilityOptions selects the cache behavior: Device flushes to the GPU coherence
    // point (correct for a real heap RAW); ExecutionOnly (None) orders without a flush
    // (cheaper; valid for ordering-only edges / UMA L2-coherent reads). See resolve_*.
    [s->en barrierAfterEncoderStages:MTLStageDispatch
                   beforeEncoderStages:MTLStageDispatch
                     visibilityOptions:resolve_barrier_vis(vis)];
}
void StepEncoder::mark_timestamp(void* heap, uint32_t idx, bool precise) {
    if (heap == nullptr) return;
    auto* s = static_cast<StepState*>(impl_);
    [s->en writeTimestampWithGranularity:(precise ? MTL4TimestampGranularityPrecise
                                                  : MTL4TimestampGranularityRelaxed)
                                intoHeap:(__bridge id<MTL4CounterHeap>)heap
                                 atIndex:idx];
}

// ── RawMetalContext ───────────────────────────────────────────────────────────
RawMetalContext::RawMetalContext() : impl_(std::make_unique<Impl>()) {}
RawMetalContext::~RawMetalContext() {
    if (impl_ != nullptr) {
        if (impl_->memory_pressure_source != nullptr) {
            dispatch_source_cancel(impl_->memory_pressure_source);
            dispatch_source_set_event_handler(
                impl_->memory_pressure_source, nullptr);
            impl_->memory_pressure_source = nullptr;
        }
        impl_->drain_mapping_through(impl_->ev_val);
    }
}

static std::atomic<size_t> g_working_set_override{0};

void RawMetalContext::set_device_working_set_bytes_for_test(size_t bytes) {
    g_working_set_override.store(bytes, std::memory_order_relaxed);
}

size_t RawMetalContext::device_working_set_bytes() {
    if (const size_t forced = g_working_set_override.load(std::memory_order_relaxed);
        forced != 0) {
        return forced;
    }
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    return dev == nil ? 0 : size_t(dev.recommendedMaxWorkingSetSize);
}

static std::atomic<size_t> g_reclaimable_override{0};

bool RawMetalContext::device_working_set_is_forced() {
    return g_working_set_override.load(std::memory_order_relaxed) != 0;
}

void RawMetalContext::set_host_reclaimable_bytes_for_test(size_t bytes) {
    g_reclaimable_override.store(bytes, std::memory_order_relaxed);
}

size_t RawMetalContext::host_reclaimable_bytes() {
    if (const size_t forced = g_reclaimable_override.load(std::memory_order_relaxed);
        forced != 0) {
        return forced;
    }
    vm_size_t page = 0;
    if (host_page_size(mach_host_self(), &page) != KERN_SUCCESS) return 0;
    vm_statistics64_data_t vm{};
    mach_msg_type_number_t count = HOST_VM_INFO64_COUNT;
    if (host_statistics64(mach_host_self(), HOST_VM_INFO64,
                          reinterpret_cast<host_info64_t>(&vm),
                          &count) != KERN_SUCCESS) {
        return 0;
    }
    // `external_page_count` is the file-backed set and is already a subset of
    // active+inactive, so counting inactive and external both would double-
    // count the clean file pages that make up most of a freshly-loaded model's
    // footprint. Take inactive plus purgeable plus free, and add only the
    // file-backed pages the kernel has parked as speculative.
    const uint64_t pages = uint64_t(vm.free_count) + vm.inactive_count +
                           vm.purgeable_count + vm.speculative_count;
    return size_t(pages * uint64_t(page));
}

std::unique_ptr<RawMetalContext> RawMetalContext::create(
    size_t heap_bytes,
    size_t elastic_budget_bytes) {
    auto ctx = std::unique_ptr<RawMetalContext>(new RawMetalContext());
    auto& I = *ctx->impl_;

    I.dev = MTLCreateSystemDefaultDevice();
    if (I.dev == nil) { fprintf(stderr, "[pie-metal] no Metal device\n"); return nullptr; }
    if (![I.dev supportsFamily:MTLGPUFamilyMetal4]) {
        fprintf(stderr,
                "[pie-metal] device '%s' does not support MTLGPUFamilyMetal4; "
                "this driver is Metal 4 only (macOS 26+ / MSL 4.0)\n",
                I.dev.name.UTF8String);
        return nullptr;
    }

    I.queue    = [I.dev newMTL4CommandQueue];
    I.mapping_queue = [I.dev newMTL4CommandQueue];
    I.alloc[0] = [I.dev newCommandAllocator];
    I.alloc[1] = [I.dev newCommandAllocator];
    I.event    = [I.dev newSharedEvent];
    if (I.queue == nil || I.mapping_queue == nil ||
        I.alloc[0] == nil || I.alloc[1] == nil || I.event == nil) {
        fprintf(stderr, "[pie-metal] MTL4 queue/allocator/event creation failed\n");
        return nullptr;
    }
    I.elastic_budget_bytes = align_up(
        elastic_budget_bytes,
        pie::elastic::kLogicalPageBytes);
    if (const char* floor = std::getenv("PIE_METAL_PRESSURE_FLOOR_BYTES")) {
        char* end = nullptr;
        const unsigned long long parsed = std::strtoull(floor, &end, 10);
        if (end != floor && end != nullptr && *end == '\0') {
            const size_t requested = static_cast<size_t>(
                std::min<unsigned long long>(
                    parsed,
                    I.elastic_budget_bytes));
            I.elastic_pressure_floor_bytes = std::min(
                I.elastic_budget_bytes,
                align_up(
                    requested,
                    pie::elastic::kLogicalPageBytes));
        }
    }
    I.memory_pressure_source = dispatch_source_create(
        DISPATCH_SOURCE_TYPE_MEMORYPRESSURE,
        0,
        DISPATCH_MEMORYPRESSURE_NORMAL |
            DISPATCH_MEMORYPRESSURE_WARN |
            DISPATCH_MEMORYPRESSURE_CRITICAL,
        dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0));
    if (I.memory_pressure_source != nullptr) {
        dispatch_source_t source = I.memory_pressure_source;
        auto pressure_level = I.memory_pressure_level;
        dispatch_source_set_event_handler(I.memory_pressure_source, ^{
            const unsigned long pressure =
                dispatch_source_get_data(source);
            const std::uint32_t level =
                (pressure & DISPATCH_MEMORYPRESSURE_CRITICAL) != 0
                    ? 2u
                    : ((pressure & DISPATCH_MEMORYPRESSURE_WARN) != 0 ? 1u : 0u);
            pressure_level->store(level, std::memory_order_release);
        });
        dispatch_resume(I.memory_pressure_source);
    }

    NSError* e = nil;
    MTL4PipelineDataSetSerializerDescriptor* serializer_descriptor =
        [MTL4PipelineDataSetSerializerDescriptor new];
    serializer_descriptor.configuration =
        MTL4PipelineDataSetSerializerConfigurationCaptureBinaries;
    I.pipeline_serializer =
        [I.dev newPipelineDataSetSerializerWithDescriptor:serializer_descriptor];
    MTL4CompilerDescriptor* cd = [MTL4CompilerDescriptor new];
    cd.pipelineDataSetSerializer = I.pipeline_serializer;
    I.compiler = [I.dev newCompilerWithDescriptor:cd error:&e];
    if (I.compiler == nil) {
        fprintf(stderr, "[pie-metal] compiler create failed: %s\n",
                e.localizedDescription.UTF8String);
        return nullptr;
    }

    MTLHeapDescriptor* hd = [MTLHeapDescriptor new];
    hd.type        = MTLHeapTypePlacement;
    hd.storageMode = MTLStorageModeShared;   // UMA: contents() valid for all slots
    hd.hazardTrackingMode = MTLHazardTrackingModeUntracked;
    hd.size        = heap_bytes;
    I.heap = [I.dev newHeapWithDescriptor:hd];
    if (I.heap == nil) {
        fprintf(stderr, "[pie-metal] heap alloc failed (%zu bytes)\n", heap_bytes);
        return nullptr;
    }
    I.heap_size = heap_bytes;

    MTLResidencySetDescriptor* rsd = [MTLResidencySetDescriptor new];
    I.rs = [I.dev newResidencySetWithDescriptor:rsd error:&e];
    if (I.rs == nil) {
        fprintf(stderr, "[pie-metal] residency set failed: %s\n",
                e.localizedDescription.UTF8String);
        return nullptr;
    }

    I.retained  = [NSMutableArray new];
    I.argtables = [NSMutableDictionary new];
    return ctx;
}

SlotHandle RawMetalContext::const_slot(int ordinal, std::uint8_t index, size_t bytes) {
    const std::uint64_t key =
        (static_cast<std::uint64_t>(static_cast<std::uint32_t>(ordinal)) << 8) | index;
    auto& cache = impl_->const_slots;
    const auto it = cache.find(key);
    // A different SIZE at the same slot would be a different constant, which
    // this cache cannot serve; allocate afresh rather than hand back a slot too
    // small to hold it.
    if (it != cache.end() && it->second.size >= bytes) return it->second;
    SlotHandle s = heap_alloc(bytes);
    if (s.valid()) cache[key] = s;
    return s;
}

SlotHandle RawMetalContext::heap_alloc(size_t size, size_t align) {
    auto& I = *impl_;
    SlotHandle h;
    if (size == 0) return h;

    MTLResourceOptions opts =
        MTLResourceStorageModeShared | MTLResourceHazardTrackingModeUntracked;
    MTLSizeAndAlign sa = [I.dev heapBufferSizeAndAlignWithLength:size options:opts];
    size_t a = align > sa.align ? align : sa.align;
    size_t off = align_up(I.bump, a);
    if (off + sa.size > I.heap_size) {
        fprintf(stderr, "[pie-metal] heap OOM: need %zu at off %zu, cap %zu\n",
                sa.size, off, I.heap_size);
        return h;
    }
    id<MTLBuffer> buf = [I.heap newBufferWithLength:size options:opts offset:off];
    if (buf == nil) { fprintf(stderr, "[pie-metal] placement buffer failed\n"); return h; }
    [I.retained addObject:buf];
    I.bump = off + sa.size;

    h.buffer       = (__bridge void*)buf;
    h.contents_ptr = buf.contents;
    h.gpu_address  = buf.gpuAddress;
    h.offset       = off;
    h.size         = size;
    return h;
}

SlotHandle RawMetalContext::create_standalone_buffer(size_t size) {
    auto& I = *impl_;
    SlotHandle h;
    if (size == 0) return h;
    id<MTLBuffer> buf = [I.dev newBufferWithLength:size options:MTLResourceStorageModeShared];
    if (buf == nil) {
        fprintf(stderr, "[pie-metal] standalone buffer alloc failed (%zu bytes)\n", size);
        return h;
    }
    [I.retained addObject:buf];  // keep it alive until release_standalone_buffer
    // Incremental residency: safe to add + commit MORE allocations after the
    // initial make_resident() (Metal 4's MTLResidencySet supports growing the
    // set across its lifetime, not just a one-time build).
    [I.rs addAllocation:buf];
    [I.rs commit];
    if (I.resident) [I.rs requestResidency];
    I.standalone_count += 1;
    I.standalone_bytes += size;

    h.buffer       = (__bridge void*)buf;
    h.contents_ptr = buf.contents;
    h.gpu_address  = buf.gpuAddress;
    h.offset       = 0;
    h.size         = size;
    return h;
}

SlotHandle RawMetalContext::wrap_host_memory(void* ptr, size_t size) {
    auto& I = *impl_;
    SlotHandle h;
    if (ptr == nullptr || size == 0) return h;
    id<MTLBuffer> buf = [I.dev newBufferWithBytesNoCopy:ptr
                                                 length:size
                                                options:MTLResourceStorageModeShared
                                            deallocator:nil];
    if (buf == nil) {
        fprintf(stderr, "[pie-metal] no-copy buffer over host memory failed (%zu bytes)\n",
                size);
        return h;
    }
    [I.retained addObject:buf];
    [I.rs addAllocation:buf];
    [I.rs commit];
    if (I.resident) [I.rs requestResidency];

    h.buffer       = (__bridge void*)buf;
    h.contents_ptr = buf.contents;
    h.gpu_address  = buf.gpuAddress;
    h.offset       = 0;
    h.size         = size;
    return h;
}

SlotHandle RawMetalContext::create_elastic_buffer(
    size_t size,
    size_t initial_commit_bytes) {
    auto& I = *impl_;
    SlotHandle h;
    if (size == 0) return h;
    const size_t virtual_bytes = align_up(
        size,
        Impl::kSparseTileBytes);
    id<MTLBuffer> buffer = [I.dev
        newBufferWithLength:virtual_bytes
                   options:MTLResourceStorageModePrivate
   placementSparsePageSize:MTLSparsePageSize16];
    if (buffer == nil) {
        fprintf(
            stderr,
            "[pie-metal] placement sparse buffer creation failed (%zu bytes)\n",
            virtual_bytes);
        return h;
    }
    [I.retained addObject:buffer];
    [I.rs addAllocation:buffer];
    [I.rs commit];
    if (I.resident) [I.rs requestResidency];
    I.elastic_allocations.emplace(
        (__bridge void*)buffer,
        Impl::ElasticAllocation{
            .buffer = (__bridge void*)buffer,
            .virtual_bytes = virtual_bytes,
        });

    h.buffer = (__bridge void*)buffer;
    h.gpu_address = buffer.gpuAddress;
    h.size = size;
    h.elastic = true;
    if (initial_commit_bytes != 0 &&
        !ensure_elastic_buffer(h, initial_commit_bytes)) {
        release_elastic_buffer(h);
        return {};
    }
    return h;
}

bool RawMetalContext::ensure_elastic_buffer(
    const SlotHandle& h,
    size_t bytes) {
    auto& I = *impl_;
    if (bytes > h.size) return false;
    if (!h.elastic || h.buffer == nullptr) return bytes <= h.size;
    auto found = I.elastic_allocations.find(h.buffer);
    if (found == I.elastic_allocations.end()) return false;
    auto& allocation = found->second;
    const size_t target = align_up(
        std::min(bytes, allocation.virtual_bytes),
        Impl::kSparseTileBytes);
    if (target <= allocation.committed_bytes) return true;
    const size_t delta = target - allocation.committed_bytes;
    const size_t budget = I.effective_elastic_budget_bytes();
    if (delta > budget - std::min(budget, I.elastic_reserved_bytes)) {
        return false;
    }
    I.elastic_reserved_bytes += delta;

    while (allocation.committed_bytes < target) {
        if (allocation.chunks.empty() ||
            allocation.chunks.back().mapped ==
                allocation.chunks.back().size) {
            const size_t chunk_offset =
                allocation.chunks.size() * Impl::kElasticChunkBytes;
            const size_t chunk_bytes = std::min(
                Impl::kElasticChunkBytes,
                allocation.virtual_bytes - chunk_offset);
            MTLHeapDescriptor* descriptor = [MTLHeapDescriptor new];
            descriptor.type = MTLHeapTypePlacement;
            descriptor.storageMode = MTLStorageModeShared;
            descriptor.hazardTrackingMode = MTLHazardTrackingModeUntracked;
            descriptor.size = chunk_bytes;
            descriptor.maxCompatiblePlacementSparsePageSize =
                MTLSparsePageSize16;
            id<MTLHeap> heap = [I.dev newHeapWithDescriptor:descriptor];
            if (heap == nil) {
                I.elastic_reserved_bytes -=
                    target - allocation.committed_bytes;
                return false;
            }
            id<MTLBuffer> alias = [heap
                newBufferWithLength:chunk_bytes
                           options:MTLResourceStorageModeShared
                            offset:0];
            if (alias == nil) {
                I.elastic_reserved_bytes -=
                    target - allocation.committed_bytes;
                return false;
            }
            [I.retained addObject:heap];
            [I.retained addObject:alias];
            [I.rs addAllocation:heap];
            [I.rs commit];
            if (I.resident) [I.rs requestResidency];
            allocation.chunks.push_back({
                .heap = (__bridge void*)heap,
                .alias = (__bridge void*)alias,
                .size = chunk_bytes,
            });
        }

        auto& chunk = allocation.chunks.back();
        const size_t grow = std::min(
            target - allocation.committed_bytes,
            chunk.size - chunk.mapped);
        MTL4UpdateSparseBufferMappingOperation operation{};
        operation.mode = MTLSparseTextureMappingModeMap;
        operation.bufferRange = NSMakeRange(
            allocation.committed_bytes / Impl::kSparseTileBytes,
            grow / Impl::kSparseTileBytes);
        operation.heapOffset = chunk.mapped / Impl::kSparseTileBytes;
        I.schedule_mapping(
            (__bridge id<MTLBuffer>)allocation.buffer,
            (__bridge id<MTLHeap>)chunk.heap,
            operation);
        chunk.mapped += grow;
        allocation.committed_bytes += grow;
        I.elastic_committed_bytes += grow;
    }
    return true;
}

bool RawMetalContext::ensure_elastic_buffers_atomically(
    const std::vector<std::pair<SlotHandle, size_t>>& targets) {
    auto& I = *impl_;
    std::vector<std::pair<SlotHandle, size_t>> normalized;
    std::unordered_map<void*, std::size_t> by_buffer;
    for (const auto& [slot, bytes] : targets) {
        const auto [found, inserted] =
            by_buffer.emplace(slot.buffer, normalized.size());
        if (inserted) {
            normalized.emplace_back(slot, bytes);
        } else {
            normalized[found->second].second =
                std::max(normalized[found->second].second, bytes);
        }
    }
    std::vector<std::pair<SlotHandle, size_t>> prior;
    prior.reserve(normalized.size());
    size_t total_delta = 0;
    for (const auto& [slot, bytes] : normalized) {
        if (!slot.elastic || slot.buffer == nullptr || bytes > slot.size) {
            return false;
        }
        const auto found = I.elastic_allocations.find(slot.buffer);
        if (found == I.elastic_allocations.end()) return false;
        const size_t target = align_up(
            std::min(bytes, found->second.virtual_bytes),
            Impl::kSparseTileBytes);
        prior.emplace_back(slot, found->second.committed_bytes);
        if (target > found->second.committed_bytes) {
            const size_t delta = target - found->second.committed_bytes;
            if (delta > std::numeric_limits<size_t>::max() - total_delta) {
                return false;
            }
            total_delta += delta;
        }
    }
    const size_t budget = I.effective_elastic_budget_bytes();
    if (total_delta > budget -
                          std::min(budget, I.elastic_reserved_bytes)) {
        return false;
    }
    for (const auto& [slot, bytes] : normalized) {
        if (ensure_elastic_buffer(slot, bytes)) continue;
        for (auto it = prior.rbegin(); it != prior.rend(); ++it) {
            trim_elastic_buffer(it->first, it->second);
        }
        return false;
    }
    return true;
}

bool RawMetalContext::trim_elastic_buffer(
    const SlotHandle& h,
    size_t bytes) {
    auto& I = *impl_;
    if (!h.elastic || h.buffer == nullptr) return bytes <= h.size;
    auto found = I.elastic_allocations.find(h.buffer);
    if (found == I.elastic_allocations.end()) return false;
    auto& allocation = found->second;
    const size_t target = align_up(
        std::min(bytes, allocation.virtual_bytes),
        Impl::kSparseTileBytes);
    uint64_t last_done = 0;
    size_t released_bytes = 0;
    while (allocation.committed_bytes > target &&
           !allocation.chunks.empty()) {
        auto& chunk = allocation.chunks.back();
        const size_t shrink = std::min(
            allocation.committed_bytes - target,
            chunk.mapped);
        MTL4UpdateSparseBufferMappingOperation operation{};
        operation.mode = MTLSparseTextureMappingModeUnmap;
        operation.bufferRange = NSMakeRange(
            (allocation.committed_bytes - shrink) /
                Impl::kSparseTileBytes,
            shrink / Impl::kSparseTileBytes);
        operation.heapOffset = 0;
        last_done = I.schedule_mapping(
            (__bridge id<MTLBuffer>)allocation.buffer,
            nil,
            operation);
        chunk.mapped -= shrink;
        allocation.committed_bytes -= shrink;
        released_bytes += shrink;
        if (chunk.mapped == 0) {
            I.pending_elastic_releases.push_back({
                .event_value = last_done,
                .objects = {chunk.alias, chunk.heap},
            });
            allocation.chunks.pop_back();
        }
    }
    I.drain_mapping_through(last_done);
    I.elastic_committed_bytes -= std::min(
        I.elastic_committed_bytes, released_bytes);
    I.elastic_reserved_bytes -= std::min(
        I.elastic_reserved_bytes, released_bytes);
    return true;
}

void RawMetalContext::release_elastic_buffer(const SlotHandle& h) {
    auto& I = *impl_;
    if (!h.elastic || h.buffer == nullptr) return;
    trim_elastic_buffer(h, 0);
    auto found = I.elastic_allocations.find(h.buffer);
    if (found == I.elastic_allocations.end()) return;
    id<MTLBuffer> buffer = (__bridge id<MTLBuffer>)h.buffer;
    [I.rs removeAllocation:buffer];
    [I.rs commit];
    [I.retained removeObject:buffer];
    I.elastic_allocations.erase(found);
}

bool RawMetalContext::zero_buffer_range(
    const SlotHandle& h,
    size_t offset,
    size_t bytes) {
    if (offset > h.size || bytes > h.size - offset) return false;
    if (!h.elastic) {
        if (h.contents() == nullptr) return false;
        std::memset(static_cast<char*>(h.contents()) + offset, 0, bytes);
        return true;
    }
    auto found = impl_->elastic_allocations.find(h.buffer);
    if (found == impl_->elastic_allocations.end() ||
        offset + bytes > found->second.committed_bytes) {
        return false;
    }
    size_t cursor = offset;
    size_t remaining = bytes;
    while (remaining != 0) {
        const size_t chunk_index =
            cursor / Impl::kElasticChunkBytes;
        const size_t chunk_offset =
            cursor % Impl::kElasticChunkBytes;
        const auto& chunk = found->second.chunks.at(chunk_index);
        const size_t count = std::min(
            remaining,
            chunk.mapped - chunk_offset);
        auto alias = (__bridge id<MTLBuffer>)chunk.alias;
        std::memset(
            static_cast<char*>(alias.contents) + chunk_offset,
            0,
            count);
        cursor += count;
        remaining -= count;
    }
    return true;
}

bool RawMetalContext::copy_buffer_range(
    const SlotHandle& dst,
    size_t dst_offset,
    const SlotHandle& src,
    size_t src_offset,
    size_t bytes) {
    if (dst_offset > dst.size || bytes > dst.size - dst_offset ||
        src_offset > src.size || bytes > src.size - src_offset) {
        return false;
    }
    auto span = [&](const SlotHandle& h, size_t offset)
        -> std::pair<void*, size_t> {
        if (!h.elastic) {
            if (h.contents() == nullptr) return {nullptr, 0};
            return {
                static_cast<char*>(h.contents()) + offset,
                h.size - offset,
            };
        }
        const auto found = impl_->elastic_allocations.find(h.buffer);
        if (found == impl_->elastic_allocations.end() ||
            offset >= found->second.committed_bytes) {
            return {nullptr, 0};
        }
        const size_t chunk_index =
            offset / Impl::kElasticChunkBytes;
        const size_t chunk_offset =
            offset % Impl::kElasticChunkBytes;
        const auto& chunk = found->second.chunks.at(chunk_index);
        auto alias = (__bridge id<MTLBuffer>)chunk.alias;
        return {
            static_cast<char*>(alias.contents) + chunk_offset,
            chunk.mapped - chunk_offset,
        };
    };

    size_t remaining = bytes;
    while (remaining != 0) {
        auto [dst_ptr, dst_available] = span(dst, dst_offset);
        auto [src_ptr, src_available] = span(src, src_offset);
        if (dst_ptr == nullptr || src_ptr == nullptr) return false;
        const size_t count = std::min(
            remaining,
            std::min(dst_available, src_available));
        std::memmove(dst_ptr, src_ptr, count);
        dst_offset += count;
        src_offset += count;
        remaining -= count;
    }
    return true;
}

size_t RawMetalContext::elastic_page_bytes() const {
    return pie::elastic::kLogicalPageBytes;
}

size_t RawMetalContext::elastic_budget_pages() const {
    return pie::elastic::pages_for_bytes(
        impl_->effective_elastic_budget_bytes());
}

size_t RawMetalContext::elastic_committed_pages() const {
    return pie::elastic::pages_for_bytes(impl_->elastic_committed_bytes);
}

void RawMetalContext::set_memory_pressure_level_for_test(
    std::uint32_t level) {
    impl_->memory_pressure_level->store(
        std::min<std::uint32_t>(level, 2u),
        std::memory_order_release);
}

void RawMetalContext::drain_elastic_mappings() {
    impl_->drain_mapping_through(impl_->ev_val);
}

size_t RawMetalContext::pending_elastic_release_count() const {
    return impl_->pending_elastic_releases.size();
}

SlotHandle RawMetalContext::acquire_transient_buffer(size_t size) {
    auto& I = *impl_;
    SlotHandle result;
    if (size == 0) return result;

    size_t size_class = 256;
    while (size_class < size) {
        if (size_class > std::numeric_limits<size_t>::max() / 2) {
            ++I.transient_stats.allocation_failures;
            return result;
        }
        size_class *= 2;
    }
    if (size_class > I.transient_stats.capacity_bytes) {
        ++I.transient_stats.allocation_failures;
        return result;
    }

    auto matching = I.transient_free.find(size_class);
    if (matching != I.transient_free.end() &&
        !matching->second.empty()) {
        result = matching->second.back();
        matching->second.pop_back();
        I.transient_allocations[result.buffer].in_use = true;
        ++I.transient_stats.reuse_hits;
        --I.transient_stats.cached_buffers;
        I.transient_stats.cached_bytes -= result.size;
        ++I.transient_stats.in_use_buffers;
        I.transient_stats.in_use_bytes += result.size;
        return result;
    }

    auto release_cached = [&](SlotHandle handle) {
        I.transient_allocations.erase(handle.buffer);
        --I.transient_stats.resident_buffers;
        I.transient_stats.resident_bytes -= handle.size;
        --I.transient_stats.cached_buffers;
        I.transient_stats.cached_bytes -= handle.size;
        ++I.transient_stats.evictions;
        release_standalone_buffer(handle);
    };
    while (I.transient_stats.resident_bytes + size_class >
           I.transient_stats.capacity_bytes) {
        auto bucket = I.transient_free.end();
        while (bucket != I.transient_free.begin()) {
            --bucket;
            if (!bucket->second.empty()) break;
        }
        if (bucket == I.transient_free.end() || bucket->second.empty()) break;
        SlotHandle evicted = bucket->second.back();
        bucket->second.pop_back();
        release_cached(evicted);
    }

    if (I.transient_stats.resident_bytes + size_class >
        I.transient_stats.capacity_bytes) {
        ++I.transient_stats.allocation_failures;
        return result;
    }
    result = create_standalone_buffer(size_class);
    if (!result.valid()) {
        ++I.transient_stats.allocation_failures;
        return result;
    }
    I.transient_allocations.emplace(
        result.buffer,
        Impl::TransientAllocation{
            .size_class = size_class,
            .in_use = true,
        });
    ++I.transient_stats.allocations;
    ++I.transient_stats.resident_buffers;
    I.transient_stats.resident_bytes += size_class;
    I.transient_stats.peak_resident_bytes = std::max(
        I.transient_stats.peak_resident_bytes,
        I.transient_stats.resident_bytes);
    ++I.transient_stats.in_use_buffers;
    I.transient_stats.in_use_bytes += result.size;
    return result;
}

void RawMetalContext::recycle_transient_buffer(const SlotHandle& h) {
    auto& I = *impl_;
    const auto allocation = I.transient_allocations.find(h.buffer);
    if (allocation == I.transient_allocations.end() ||
        !allocation->second.in_use) {
        return;
    }
    allocation->second.in_use = false;
    --I.transient_stats.in_use_buffers;
    I.transient_stats.in_use_bytes -= allocation->second.size_class;
    ++I.transient_stats.recycles;

    auto& bucket = I.transient_free[allocation->second.size_class];
    // Cache depth by size class rather than a flat 8. One M3 fire acquires ~13
    // buffers and most of them land in the smallest classes, so a flat 8
    // overflowed every fire -- and the overflow path is not cheap: releasing a
    // buffer commits the residency set and linear-scans the retained array, and
    // the next fire then re-allocates. It cost 0.375ms of a ~1.2ms gap between
    // two forwards, for buffers a few hundred bytes wide. The byte budget below
    // is what actually bounds this; the depth only stops one class hoarding it.
    const size_t size_class = allocation->second.size_class;
    const size_t cache_depth =
        std::clamp<size_t>((size_t{1} << 20) / std::max<size_t>(size_class, 1),
                           8, 64);
    if (bucket.size() < cache_depth &&
        I.transient_stats.resident_bytes <=
            I.transient_stats.capacity_bytes) {
        bucket.push_back(h);
        ++I.transient_stats.cached_buffers;
        I.transient_stats.cached_bytes += allocation->second.size_class;
        return;
    }

    const size_t bytes = allocation->second.size_class;
    I.transient_allocations.erase(allocation);
    --I.transient_stats.resident_buffers;
    I.transient_stats.resident_bytes -= bytes;
    ++I.transient_stats.evictions;
    release_standalone_buffer(h);
}

TransientBufferPoolStats
RawMetalContext::transient_buffer_pool_stats() const {
    return impl_->transient_stats;
}

void RawMetalContext::set_transient_buffer_pool_limit_for_test(size_t bytes) {
    auto& I = *impl_;
    I.transient_stats.capacity_bytes = std::max<size_t>(bytes, 256);
    while (I.transient_stats.resident_bytes >
           I.transient_stats.capacity_bytes) {
        auto bucket = I.transient_free.end();
        while (bucket != I.transient_free.begin()) {
            --bucket;
            if (!bucket->second.empty()) break;
        }
        if (bucket == I.transient_free.end() || bucket->second.empty()) break;
        SlotHandle evicted = bucket->second.back();
        bucket->second.pop_back();
        I.transient_allocations.erase(evicted.buffer);
        --I.transient_stats.resident_buffers;
        I.transient_stats.resident_bytes -= evicted.size;
        --I.transient_stats.cached_buffers;
        I.transient_stats.cached_bytes -= evicted.size;
        ++I.transient_stats.evictions;
        release_standalone_buffer(evicted);
    }
}

void RawMetalContext::use_external_buffer(const SlotHandle& h) {
    auto& I = *impl_;
    if (h.buffer == nullptr) return;
    auto [entry, inserted] =
        I.external_allocations.emplace(h.buffer, 0);
    ++entry->second;
    if (!inserted) return;
    id<MTLBuffer> buffer = (__bridge id<MTLBuffer>)h.buffer;
    [I.rs addAllocation:buffer];
    [I.rs commit];
    if (I.resident) [I.rs requestResidency];
}

void RawMetalContext::release_external_buffer(const SlotHandle& h) {
    auto& I = *impl_;
    if (h.buffer == nullptr) return;
    const auto entry = I.external_allocations.find(h.buffer);
    if (entry == I.external_allocations.end()) return;
    if (--entry->second != 0) return;
    I.external_allocations.erase(entry);
    id<MTLBuffer> buffer = (__bridge id<MTLBuffer>)h.buffer;
    [I.rs removeAllocation:buffer];
    [I.rs commit];
    if (I.resident) [I.rs requestResidency];
}

size_t RawMetalContext::external_buffer_count() const {
    return impl_->external_allocations.size();
}

void RawMetalContext::release_standalone_buffer(const SlotHandle& h) {
    auto& I = *impl_;
    if (h.buffer == nullptr) return;
    id<MTLBuffer> buf = (__bridge id<MTLBuffer>)h.buffer;
    // Drop from residency first, then from the retained-alive array so ARC can
    // free the allocation. removeObject uses pointer identity here (same buf).
    [I.rs removeAllocation:buf];
    [I.rs commit];
    if (I.resident) [I.rs requestResidency];
    const NSUInteger before = I.retained.count;
    [I.retained removeObject:buf];
    if (I.retained.count < before) {
        // Only adjust accounting when we actually owned it (idempotent /
        // defensive against a double-release or a foreign handle).
        I.standalone_count -= (I.standalone_count > 0 ? 1 : 0);
        I.standalone_bytes -= (I.standalone_bytes >= h.size ? h.size : I.standalone_bytes);
    }
}

size_t RawMetalContext::standalone_buffer_count() const { return impl_->standalone_count; }
size_t RawMetalContext::standalone_bytes() const { return impl_->standalone_bytes; }


void RawMetalContext::make_resident() {
    auto& I = *impl_;
    if (!I.resident) {
        [I.rs addAllocation:I.heap];   // whole heap resident ONCE (I2); covers all
        [I.rs commit];                 // placement sub-buffers, incl. ones allocated later
        I.resident = true;
    }
    [I.rs requestResidency];
}

void RawMetalContext::arg_bind(Kernel k, int ordinal, uint8_t bind_index, SlotHandle slot,
                               size_t offset) {
    (void)k;  // decorative tag; ordinal is the key
    arg_bind_ordinal(ordinal, bind_index, slot, offset);
}

void RawMetalContext::arg_bind_ordinal(int ordinal, uint8_t bind_index, SlotHandle slot,
                                       size_t offset) {
    auto& I = *impl_;
    if (bind_index >= kMaxArgBinds) {
        fprintf(stderr,
                "[pie-metal] arg_bind ordinal %d index %u: MSL allows at most %u "
                "buffer bindings ([[buffer(0..%u)]]); the bind was ignored\n",
                ordinal, unsigned(bind_index), kMaxArgBinds, kMaxArgBinds - 1);
        return;
    }
    id<MTL4ArgumentTable> t = I.argtable_for(ordinal, /*create=*/true);
    if (t == nil) return;
    [t setAddress:(slot.gpu_address + offset) atIndex:bind_index];
    const uint64_t key = (uint64_t(uint32_t(ordinal)) << 8) | bind_index;
    I.bound_arg_addresses[key] = slot.gpu_address + offset;
}

bool RawMetalContext::arg_slot_is_bound(int ordinal, uint8_t bind_index) const {
    const auto key = (uint64_t(uint32_t(ordinal)) << 8) | bind_index;
    return impl_->bound_arg_addresses.find(key) != impl_->bound_arg_addresses.end();
}

uint64_t RawMetalContext::arg_slot_address(int ordinal, uint8_t bind_index) const {
    const auto key = (uint64_t(uint32_t(ordinal)) << 8) | bind_index;
    const auto it = impl_->bound_arg_addresses.find(key);
    return it == impl_->bound_arg_addresses.end() ? 0 : it->second;
}

void RawMetalContext::release_argtable_ordinal(int ordinal) {
    auto& I = *impl_;
    [I.argtables removeObjectForKey:@(argkey(ordinal))];
    // A key is (ordinal << 8) | bind_index and bind_index is a uint8_t, so an
    // ordinal owns at most 256 keys and they can be probed directly. Scanning
    // the whole set instead was O(all bound slots) per released ordinal, and
    // the prefill alone binds ~123k of them (34 token DAGs x 363 dispatches x
    // their slots) -- 2.7ms per fire, which was the entire remaining host gap
    // between two forwards.
    for (std::uint32_t bind = 0; bind < 256u; ++bind) {
        const std::uint64_t key =
            (static_cast<std::uint64_t>(static_cast<std::uint32_t>(ordinal)) << 8) | bind;
        I.bound_arg_addresses.erase(key);
    }
}

namespace {

/// Whether `lib` really exports `fn`.
///
/// Asked BEFORE building a pipeline descriptor, because a
/// `MTL4LibraryFunctionDescriptor` naming a function the library does not have
/// is not an error the compiler returns -- Metal asserts inside
/// `setComputeFunction:` and kills the process. That is the wrong outcome for a
/// name this driver derives: `sdpa_paged_decode_bfloat16_d_96` is what a
/// checkpoint with an uninstantiated head width asks for, and the honest answer
/// is a refusal that names the width, not a crash with no context.
bool library_has_function(id<MTLLibrary> lib, const std::string& fn) {
    NSString* want = [NSString stringWithUTF8String:fn.c_str()];
    for (NSString* have in lib.functionNames) {
        if ([have isEqualToString:want]) return true;
    }
    return false;
}

Pso compile_pso_impl(
    RawMetalContext::Impl& I,
    const std::string& src,
    const std::string& fn,
    MTLCompileOptions* options,
    MTL4CompilerTaskOptions* task_options,
    std::string* error) {
    Pso out;
    NSError* e = nil;
    // Single choke point for the MSL dialect. This driver is Metal 4 only, and
    // the runtime shader compiler otherwise defaults to an older standard --
    // which silently hides <metal_tensor> and the MetalPerformancePrimitives
    // tensor ops from every kernel. Pinning it here (rather than at each of the
    // four call sites) keeps the dialect a property of the driver, not of the
    // caller that happened to build the options object.
    if (options == nil) options = [MTLCompileOptions new];
    options.languageVersion = MTLLanguageVersion4_0;
    id<MTLLibrary> lib =
        [I.dev newLibraryWithSource:[NSString stringWithUTF8String:src.c_str()]
                            options:options
                              error:&e];
    if (lib == nil) {
        if (error) *error = e.localizedDescription.UTF8String;
        return out;
    }
    if (!library_has_function(lib, fn)) {
        if (error) *error = "the library compiled but exports no '" + fn + "'";
        return out;
    }
    MTL4LibraryFunctionDescriptor* fd = [MTL4LibraryFunctionDescriptor new];
    fd.name = [NSString stringWithUTF8String:fn.c_str()];
    fd.library = lib;
    MTL4ComputePipelineDescriptor* pd = [MTL4ComputePipelineDescriptor new];
    pd.computeFunctionDescriptor = fd;
    // The entrypoint's name, carried on the pipeline, is what per-dispatch
    // tracing has to report -- a `Pso` is an opaque pointer and an ordinal
    // names a position in a DAG rather than a kernel.
    pd.label = fd.name;
    id<MTLComputePipelineState> pso =
        [I.compiler newComputePipelineStateWithDescriptor:pd
                                      compilerTaskOptions:task_options
                                                    error:&e];
    if (pso == nil) {
        if (error) *error = e.localizedDescription.UTF8String;
        return out;
    }
    [I.retained addObject:pso];
    out.obj = (__bridge void*)pso;
    I.retained_psos.insert(out.obj);
    return out;
}

bool read_metal_source_at(
    const std::string& path,
    std::string& source,
    std::string* error,
    int depth) {
    NSError* e = nil;
    NSString* src = [NSString
        stringWithContentsOfFile:[NSString stringWithUTF8String:path.c_str()]
                        encoding:NSUTF8StringEncoding
                           error:&e];
    if (src == nil) {
        if (error) {
            *error =
                std::string("read failed: ") + e.localizedDescription.UTF8String;
        }
        return false;
    }
    source = src.UTF8String;

    // Metal's runtime shader compiler does no filesystem include lookup, so a
    // `#include "..."` in a kernel source is spliced in here. That is the only
    // way two `.metal` files can share a definition: each one is handed to
    // `newLibraryWithSource:` on its own, so anything they both need would
    // otherwise have to be written twice. The 4-bit codecs are, which is what
    // made this general -- it used to resolve one hardcoded filename.
    //
    // Angle-bracket includes are the system headers and are left alone.
    if (depth > 8) {
        if (error) *error = "include nesting too deep at '" + path + "'";
        return false;
    }
    const std::size_t separator = path.find_last_of("/\\");
    const std::string dir =
        separator == std::string::npos ? std::string{} : path.substr(0, separator + 1);
    constexpr std::string_view kDirective = "#include \"";
    for (std::size_t at = source.find(kDirective); at != std::string::npos;
         at = source.find(kDirective, at)) {
        // Only at the start of a line: the same characters inside a string or a
        // comment are not a directive.
        if (at != 0 && source[at - 1] != '\n') {
            at += kDirective.size();
            continue;
        }
        const std::size_t name_at = at + kDirective.size();
        const std::size_t close = source.find('"', name_at);
        if (close == std::string::npos) {
            if (error) *error = "unterminated #include in '" + path + "'";
            return false;
        }
        std::string included;
        if (!read_metal_source_at(dir + source.substr(name_at, close - name_at), included, error,
                                  depth + 1)) {
            return false;
        }
        source.replace(at, close + 1 - at, included);
        at += included.size();
    }
    return true;
}

void configure_ptir_math_options(
    MTLCompileOptions* options,
    bool& strict_math) {
    options.mathMode = MTLMathModeSafe;
    options.mathFloatingPointFunctions =
        MTLMathFloatingPointFunctionsPrecise;
    strict_math = options.mathMode == MTLMathModeSafe;
}

}  // namespace

bool read_metal_source(
    const std::string& path,
    std::string& source,
    std::string* error) {
    return read_metal_source_at(path, source, error, 0);
}

Pso RawMetalContext::compile_pso(const std::string& src, const std::string& fn,
                                  std::string* error) {
    return compile_pso_impl(*impl_, src, fn, nil, nil, error);
}

Pso RawMetalContext::compile_pso_from_file(const std::string& path, const std::string& fn,
                                           std::string* error) {
    std::string source;
    return read_metal_source(path, source, error)
               ? compile_pso(source, fn, error)
               : Pso{};
}

std::string RawMetalContext::pso_archive_dir() {
    if (const char* override_dir = getenv("PIE_METAL_PSO_CACHE")) return override_dir;
    const char* home = getenv("HOME");
    if (home == nullptr || *home == '\0') return std::string();
    return std::string(home) + "/Library/Caches/pie-metal";
}

namespace {
inline void hash_bytes(std::uint64_t& h, const void* p, size_t n) {
    const auto* b = static_cast<const unsigned char*>(p);
    for (size_t i = 0; i < n; ++i) { h ^= b[i]; h *= 1099511628211ull; }
}

// Names the archive for one batch. Every input that can change the compiled
// binaries goes into the key: which entrypoints were asked for, out of which
// files, and the resolved source of each of those files -- so editing a .metal
// source, or anything it includes, invalidates the archive instead of silently
// serving a stale pipeline.
std::string batch_archive_path(
    const std::vector<RawMetalContext::PsoFileRequest>& requests) {
    const std::string dir = RawMetalContext::pso_archive_dir();
    if (dir.empty()) return std::string();

    std::uint64_t h = 1469598103934665603ull;
    std::string last_path;
    for (const auto& r : requests) {
        hash_bytes(h, r.path.data(), r.path.size());
        hash_bytes(h, r.function.data(), r.function.size());
        if (r.path == last_path) continue;
        last_path = r.path;
        // The RESOLVED source, not the file's size and mtime. A source that
        // includes another would otherwise keep its key when the included file
        // changed, and serve a pipeline compiled from the old definition --
        // which is worse than a slow start, because it looks like it worked.
        std::string resolved;
        if (read_metal_source(r.path, resolved, nullptr)) {
            hash_bytes(h, resolved.data(), resolved.size());
        }
    }

    if (![[NSFileManager defaultManager] createDirectoryAtPath:@(dir.c_str())
                                   withIntermediateDirectories:YES
                                                    attributes:nil
                                                         error:nil]) {
        return std::string();
    }
    char name[64];
    std::snprintf(name, sizeof(name), "/psos-%016llx.mtl4archive",
                  (unsigned long long)h);
    return dir + name;
}

// Every edit to a kernel source strands the archive keyed to the old one, so
// without this the cache grows by ~3 MB per edit and never shrinks.
void prune_stale_archives(const std::string& dir) {
    static constexpr double kMaxAgeSeconds = 14 * 24 * 60 * 60;
    NSFileManager* fm = [NSFileManager defaultManager];
    NSArray<NSString*>* entries =
        [fm contentsOfDirectoryAtPath:@(dir.c_str()) error:nil];
    NSDate* now = [NSDate date];
    for (NSString* entry in entries) {
        if (![entry hasSuffix:@".mtl4archive"]) continue;
        NSString* full = [@(dir.c_str()) stringByAppendingPathComponent:entry];
        NSDate* modified = [fm attributesOfItemAtPath:full error:nil][NSFileModificationDate];
        if (modified != nil && [now timeIntervalSinceDate:modified] > kMaxAgeSeconds) {
            [fm removeItemAtPath:full error:nil];
        }
    }
}
}  // namespace

std::vector<Pso> RawMetalContext::compile_psos_from_files(
    const std::vector<PsoFileRequest>& requests,
    std::vector<std::string>* errors,
    bool use_archive_cache) {
    auto& I = *impl_;
    std::vector<Pso> out(requests.size());
    if (errors != nullptr) errors->assign(requests.size(), std::string{});
    if (requests.empty()) return out;

    // A hit means every pipeline below is fetched from the archive rather than
    // compiled; a miss means we compile now and write the archive at the end.
    const std::string archive_path =
        use_archive_cache ? batch_archive_path(requests) : std::string();
    MTL4CompilerTaskOptions* task = nil;
    if (!archive_path.empty()) {
        NSURL* url = [NSURL fileURLWithPath:@(archive_path.c_str())];
        if ([[NSFileManager defaultManager] fileExistsAtPath:url.path]) {
            NSError* archive_error = nil;
            id<MTL4Archive> archive = [I.dev newArchiveWithURL:url error:&archive_error];
            if (archive != nil) {
                task = [MTL4CompilerTaskOptions new];
                task.lookupArchives = @[archive];
            }
        }
    }

    // ── Stage 1: one library per distinct source file ──
    // The batch asks for many entry points out of a handful of files, so each
    // file is read and compiled exactly once instead of once per entry point.
    std::vector<std::string> paths;
    std::unordered_map<std::string, size_t> path_index;
    std::vector<size_t> request_library(requests.size(), 0);
    for (size_t i = 0; i < requests.size(); ++i) {
        auto [it, inserted] =
            path_index.emplace(requests[i].path, paths.size());
        if (inserted) paths.push_back(requests[i].path);
        request_library[i] = it->second;
    }

    std::vector<id<MTLLibrary>> libraries(paths.size(), nil);
    std::vector<std::string> library_errors(paths.size());
    // Compilation stays synchronous on purpose. Metal serializes the work in
    // its own compiler service, so driving it from extra threads measured no
    // faster, and the completion-handler API overruns the stack of Metal's
    // scheduler threads on batches this size (EXC_BAD_ACCESS in the stack
    // guard region under MTLCompilerFunctionRequest::serializedRequest).
    for (size_t li = 0; li < paths.size(); ++li) {
        std::string source;
        std::string read_error;
        if (!read_metal_source(paths[li], source, &read_error)) {
            library_errors[li] = read_error;
            continue;
        }
        MTLCompileOptions* options = [MTLCompileOptions new];
        options.languageVersion = MTLLanguageVersion4_0;
        NSError* le = nil;
        id<MTLLibrary> lib =
            [I.dev newLibraryWithSource:[NSString stringWithUTF8String:source.c_str()]
                                options:options
                                  error:&le];
        if (lib != nil) libraries[li] = lib;
        else library_errors[li] = le != nil ? le.localizedDescription.UTF8String
                                            : "library compile failed";
    }

    // ── Stage 2: every pipeline state off those libraries ──
    std::vector<id<MTLComputePipelineState>> states(requests.size(), nil);
    std::vector<std::string> pso_errors(requests.size());
    for (size_t i = 0; i < requests.size(); ++i) {
        const size_t li = request_library[i];
        if (libraries[li] == nil) {
            pso_errors[i] = library_errors[li];
            continue;
        }
        if (!library_has_function(libraries[li], requests[i].function)) {
            pso_errors[i] =
                "the library compiled but exports no '" + requests[i].function + "'";
            continue;
        }
        MTL4LibraryFunctionDescriptor* fd = [MTL4LibraryFunctionDescriptor new];
        fd.name = [NSString stringWithUTF8String:requests[i].function.c_str()];
        fd.library = libraries[li];
        MTL4ComputePipelineDescriptor* pd = [MTL4ComputePipelineDescriptor new];
        pd.computeFunctionDescriptor = fd;
        pd.label = fd.name;
        NSError* pe = nil;
        id<MTLComputePipelineState> pso =
            [I.compiler newComputePipelineStateWithDescriptor:pd
                                          compilerTaskOptions:task
                                                        error:&pe];
        if (pso != nil) states[i] = pso;
        else pso_errors[i] = pe != nil ? pe.localizedDescription.UTF8String
                                       : "pipeline build failed";
    }

    for (size_t i = 0; i < requests.size(); ++i) {
        if (states[i] == nil) {
            if (errors != nullptr) (*errors)[i] = pso_errors[i];
            continue;
        }
        [I.retained addObject:states[i]];
        out[i].obj = (__bridge void*)states[i];
        I.retained_psos.insert(out[i].obj);
    }

    // Only write on a miss, and only when the whole batch built: a partial
    // archive would be served back as if it were complete.
    const bool all_built =
        std::none_of(states.begin(), states.end(),
                     [](id<MTLComputePipelineState> p) { return p == nil; });
    if (task == nil && all_built && !archive_path.empty() &&
        I.pipeline_serializer != nil) {
        NSError* serialize_error = nil;
        NSURL* url = [NSURL fileURLWithPath:@(archive_path.c_str())];
        if (![I.pipeline_serializer serializeAsArchiveAndFlushToURL:url
                                                              error:&serialize_error]) {
            fprintf(stderr, "[pie-metal] pipeline archive write failed: %s\n",
                    serialize_error.localizedDescription.UTF8String);
        } else {
            prune_stale_archives(pso_archive_dir());
        }
    }
    return out;
}

Pso RawMetalContext::compile_precise_pso_from_file(const std::string& path,
                                                   const std::string& fn,
                                                   std::string* error) {
    std::string source;
    if (!read_metal_source(path, source, error)) return Pso{};
    // Load-time transcode has to reproduce MLX's arithmetic bit for bit, and
    // fast math rewrites `x / s` into `x * (1/s)` -- one ulp there moves a
    // rounded 4-bit code by a whole step.
    MTLCompileOptions* options = [MTLCompileOptions new];
    bool strict = false;
    configure_ptir_math_options(options, strict);
    return compile_pso_impl(*impl_, source, fn, options, nil, error);
}

Pso RawMetalContext::compile_ptir_pso(
    const std::string& src,
    const std::string& fn,
    std::string* error) {
    MTLCompileOptions* options = [MTLCompileOptions new];
    impl_->saw_ptir_compile = true;
    configure_ptir_math_options(
        options, impl_->last_ptir_fast_math_disabled);
    return compile_pso_impl(*impl_, src, fn, options, nil, error);
}

Pso RawMetalContext::compile_ptir_pso_from_file(
    const std::string& path,
    const std::string& fn,
    std::string* error) {
    std::string source;
    return read_metal_source(path, source, error)
               ? compile_ptir_pso(source, fn, error)
               : Pso{};
}

bool RawMetalContext::last_ptir_compile_disabled_fast_math() const {
    return impl_->saw_ptir_compile && impl_->last_ptir_fast_math_disabled;
}

Pso RawMetalContext::compile_ptir_pso_cached(
    const std::string& source,
    const std::string& function,
    const std::string& archive_path,
    bool* cache_hit,
    std::string* error) {
    if (cache_hit != nullptr) *cache_hit = false;
    MTL4CompilerTaskOptions* task = nil;
    id<MTL4Archive> archive = nil;
    if (!archive_path.empty()) {
        NSURL* url = [NSURL fileURLWithPath:
            [NSString stringWithUTF8String:archive_path.c_str()]];
        if ([[NSFileManager defaultManager] fileExistsAtPath:url.path]) {
            NSError* archive_error = nil;
            archive = [impl_->dev newArchiveWithURL:url error:&archive_error];
            if (archive != nil) {
                task = [MTL4CompilerTaskOptions new];
                task.lookupArchives = @[archive];
                if (cache_hit != nullptr) *cache_hit = true;
            }
        }
    }

    MTLCompileOptions* options = [MTLCompileOptions new];
    impl_->saw_ptir_compile = true;
    configure_ptir_math_options(
        options, impl_->last_ptir_fast_math_disabled);
    Pso result =
        compile_pso_impl(*impl_, source, function, options, task, error);
    if (!result.valid() || archive_path.empty() || archive != nil ||
        impl_->pipeline_serializer == nil) {
        return result;
    }

    NSURL* url = [NSURL fileURLWithPath:
        [NSString stringWithUTF8String:archive_path.c_str()]];
    NSError* serialize_error = nil;
    if (![impl_->pipeline_serializer
            serializeAsArchiveAndFlushToURL:url
                                      error:&serialize_error]) {
        if (error != nullptr) {
            *error = std::string("pipeline archive: ") +
                     serialize_error.localizedDescription.UTF8String;
        }
        release_pso(result);
        return Pso{};
    }
    return result;
}

void RawMetalContext::release_pso(Pso pso) {
    if (!pso.valid() || impl_->retained_psos.erase(pso.obj) == 0) return;
    id<MTLComputePipelineState> object =
        (__bridge id<MTLComputePipelineState>)pso.obj;
    [impl_->retained removeObject:object];
}

size_t RawMetalContext::retained_pso_count() const {
    return impl_->retained_psos.size();
}

std::uint64_t RawMetalContext::device_cache_id() const {
    const char* name = impl_->dev.name.UTF8String;
    std::uint64_t hash = 0xcbf29ce484222325ULL;
    if (name != nullptr) {
        for (const unsigned char* cursor =
                 reinterpret_cast<const unsigned char*>(name);
             *cursor != 0;
             ++cursor) {
            hash ^= *cursor;
            hash *= 0x100000001b3ULL;
        }
    }
    const std::uint64_t registry = impl_->dev.registryID;
    for (int byte = 0; byte < 8; ++byte) {
        hash ^= static_cast<std::uint8_t>(registry >> (byte * 8));
        hash *= 0x100000001b3ULL;
    }
    return hash;
}

uint32_t RawMetalContext::pso_max_threads(Pso pso) const {
    if (pso.obj == nullptr) return 0;
    auto p = (__bridge id<MTLComputePipelineState>)pso.obj;
    return static_cast<uint32_t>(p.maxTotalThreadsPerThreadgroup);
}

void* RawMetalContext::create_timestamp_heap(uint32_t count) {
    auto& I = *impl_;
    if (count == 0) return nullptr;
    MTL4CounterHeapDescriptor* d = [MTL4CounterHeapDescriptor new];
    d.type  = MTL4CounterHeapTypeTimestamp;
    d.count = count;
    NSError* e = nil;
    id<MTL4CounterHeap> h = [I.dev newCounterHeapWithDescriptor:d error:&e];
    if (h == nil) {
        fprintf(stderr, "[pie-metal] timestamp heap create failed (%u): %s\n",
                count, e.localizedDescription.UTF8String);
        return nullptr;
    }
    [I.retained addObject:h];  // context-owned until destruction
    return (__bridge void*)h;
}

void RawMetalContext::resolve_timestamps(void* heap, uint32_t count, uint64_t* out) {
    if (heap == nullptr || out == nullptr || count == 0) return;
    // CPU-timeline resolve: valid because run_step already waited the shared event, so all
    // timestamp writes have completed. Entries are tightly-packed MTL4TimestampHeapEntry
    // (a single uint64_t each) -> copy the nanosecond ticks straight out.
    id<MTL4CounterHeap> h = (__bridge id<MTL4CounterHeap>)heap;
    NSData* data = [h resolveCounterRange:NSMakeRange(0, count)];
    if (data == nil) {
        fprintf(stderr, "[pie-metal] timestamp resolve returned nil (count=%u)\n", count);
        return;
    }

    const auto* entries = static_cast<const MTL4TimestampHeapEntry*>(data.bytes);
    const uint32_t n = std::min<uint32_t>(count, uint32_t(data.length / sizeof(MTL4TimestampHeapEntry)));
    for (uint32_t i = 0; i < n; ++i) out[i] = entries[i].timestamp;
}

void RawMetalContext::release_timestamp_heap(void* heap) {
    if (heap == nullptr) return;
    id<MTL4CounterHeap> counter =
        (__bridge id<MTL4CounterHeap>)heap;
    [impl_->retained removeObject:counter];
}

uint64_t RawMetalContext::Impl::commit_and_signal(
    const id<MTL4CommandBuffer> __strong* cbs,
    NSUInteger n,
    uint64_t wait_value) {
    if (wait_value > 0) [queue waitForEvent:event value:wait_value];

    // The value this batch will signal, known before the commit so the feedback
    // handler can tag itself with the timeline point it describes.
    const uint64_t signal_value = ev_val + 1;

    // A fresh options object per commit: `addFeedbackHandler:` appends, so a
    // shared instance would accumulate one handler per step forever, and the
    // class is documented as not thread-safe.
    MTL4CommitOptions* options = [MTL4CommitOptions new];
    std::shared_ptr<FeedbackSlot> slot = feedback;
    [options addFeedbackHandler:^(id<MTL4CommitFeedback> fb) {
        GpuCommitFeedback got;
        got.event_value = signal_value;
        got.gpu_start_s = fb.GPUStartTime;
        got.gpu_end_s   = fb.GPUEndTime;
        got.gpu_ms      = (fb.GPUEndTime - fb.GPUStartTime) * 1000.0;
        if (fb.error != nil) {
            got.had_error = true;
            got.error = fb.error.localizedDescription.UTF8String;
            // The localized description of an MTL4 queue error is "the
            // operation couldn't be completed", which names nothing. The
            // domain, the code and the underlying error are what say what
            // actually went wrong -- and a 19 GB model that fails to commit
            // gives no other clue.
            fprintf(stderr,
                    "[pie-metal] GPU commit error (event=%llu): %s [%s code %ld]%s%s\n",
                    (unsigned long long)signal_value, got.error.c_str(),
                    fb.error.domain.UTF8String, (long)fb.error.code,
                    fb.error.userInfo[NSUnderlyingErrorKey] != nil ? " underlying: " : "",
                    fb.error.userInfo[NSUnderlyingErrorKey] != nil
                        ? [[fb.error.userInfo[NSUnderlyingErrorKey] description] UTF8String]
                        : "");
        }
        std::lock_guard<std::mutex> lock(slot->mutex);
        // Feedback blocks can land out of order; keep the newest only.
        if (got.event_value >= slot->value.event_value) slot->value = got;
    }];

    [queue commit:cbs count:n options:options];
    [queue signalEvent:event value:signal_value];
    ev_val = signal_value;
    return signal_value;
}

GpuCommitFeedback RawMetalContext::last_commit_feedback() const {
    std::lock_guard<std::mutex> lock(impl_->feedback->mutex);
    return impl_->feedback->value;
}

// Fold whatever feedback has landed for `event_value` into `tm`. The handler is
// asynchronous, so a miss here is normal and simply leaves `gpu_ms` at zero.
static void apply_commit_feedback(const RawMetalContext& ctx,
                                  uint64_t event_value,
                                  StepTiming& tm) {
    // The fence has already been reached, but the feedback block is dispatched
    // asynchronously. Normal inference does not put that CPU scheduling delay
    // on every token: the handler logs an error immediately, and run_steps
    // promotes a late error to a sticky failure before the next submission.
    // Tracing and the sync-feedback probe do wait because their output
    // specifically asks for this event's calibrated GPU timestamps.
    //
    // `PIE_METAL_GPU_METER` used to be a third reason and is not one any more:
    // all three of the meters it fed are `if constexpr (false)` in
    // `forward.cpp`, so arming it here bought a spin of up to 200 x 50us per
    // fire for output that cannot be printed.
    GpuCommitFeedback fb = ctx.last_commit_feedback();
    const bool synchronous =
        dispatch_trace_every() > 0 || getenv("PIE_METAL_SYNC_FEEDBACK") != nullptr;
    if (synchronous) {
        for (int spin = 0; fb.event_value != event_value && spin < 200; ++spin) {
            std::this_thread::sleep_for(std::chrono::microseconds(50));
            fb = ctx.last_commit_feedback();
        }
    }
    if (fb.event_value != event_value) return;
    tm.gpu_ms = fb.gpu_ms;
    tm.gpu_error = fb.had_error;
    tm.gpu_error_text = fb.error;
}

/// One command buffer, encoded and closed.
///
/// Shared so the ordered and the unordered runner cannot drift on what encoding
/// a segment means. Not in an anonymous namespace: `StepEncoder`'s constructor
/// is private, and a name the header cannot see is a name it cannot befriend.
void* encode_one_command_buffer(void* ctx_impl, int ab,
                                const std::function<void(StepEncoder&)>& encode_fn) {
    auto& I = *static_cast<RawMetalContext::Impl*>(ctx_impl);
    id<MTL4CommandBuffer> cb = [I.dev newCommandBuffer];
    [cb beginCommandBufferWithAllocator:I.alloc[ab]];
    [cb useResidencySet:I.rs];
    id<MTL4ComputeCommandEncoder> en = [cb computeCommandEncoder];

    I.step.en  = en;
    I.step.ctx = &I;
    if (dispatch_trace_every() > 0 && I.step.trace_heap == nullptr) {
        // Sized once and reused: a fire is bounded by its DAG, and a heap
        // reallocated per fire would be timing the allocator.
        // The driver caps a timestamp heap and says only "invalid heap size",
        // so the size is found rather than assumed: the largest power of two
        // it will hand over, down to a floor below which tracing is pointless.
        for (std::uint32_t want = 8192; want >= 256; want /= 2) {
            MTL4CounterHeapDescriptor* hd = [MTL4CounterHeapDescriptor new];
            hd.type = MTL4CounterHeapTypeTimestamp;
            hd.count = want;
            NSError* he = nil;
            id<MTL4CounterHeap> h = [I.dev newCounterHeapWithDescriptor:hd error:&he];
            if (h == nil) continue;
            [I.retained addObject:h];
            I.step.trace_heap = (__bridge void*)h;
            I.step.trace_slots = want;
            break;
        }
        if (I.step.trace_heap == nullptr) {
            fprintf(stderr, "[trace] no timestamp heap: dispatch tracing is off\n");
        }
    }
    I.step.trace_n = 0;
    I.step.trace_labels.clear();
    StepEncoder se(&I.step);
    encode_fn(se);

    [en endEncoding];
    [cb endCommandBuffer];
    I.step.en = nil;
    // Hands the caller a reference rather than a borrowed pointer. A plain
    // `__bridge` here compiles and then crashes in `objc_retain`: ARC releases
    // `cb` when this function returns, so the caller bridges back a pointer to
    // a command buffer that is already gone. The call sites balance this with
    // `__bridge_transfer`.
    return (__bridge_retained void*)cb;
}

// Resolve the bracketed timestamps of the fire just waited for and fold them
// into a running table. Reported as SHARES of the fire's own GPU time, which
// needs no tick-to-nanosecond calibration: the question a trace answers is
// which kernel to go and look at, and the absolute figure beside it is the
// commit feedback's, which is already calibrated.
void report_dispatch_trace(RawMetalContext::Impl& I, double gpu_ms) {
    const int every = dispatch_trace_every();
    if (every <= 0 || I.step.trace_heap == nullptr || I.step.trace_n == 0) return;
    const std::uint32_t n = I.step.trace_n;
    std::vector<std::uint64_t> ticks(std::size_t(2) * n, 0);
    {
        id<MTL4CounterHeap> h = (__bridge id<MTL4CounterHeap>)I.step.trace_heap;
        NSData* d = [h resolveCounterRange:NSMakeRange(0, 2 * n)];
        if (d == nil) return;
        const auto* e = static_cast<const MTL4TimestampHeapEntry*>(d.bytes);
        const std::size_t got = d.length / sizeof(MTL4TimestampHeapEntry);
        for (std::size_t i = 0; i < ticks.size() && i < got; ++i) ticks[i] = e[i].timestamp;
    }
    static std::map<std::string, std::pair<double, long>> table;
    static double total_ticks = 0.0;
    static double total_gpu_ms = 0.0;
    static long fires = 0;
    for (std::uint32_t i = 0; i < n; ++i) {
        const std::uint64_t a = ticks[2 * i], b = ticks[2 * i + 1];
        // A relaxed timestamp is not ordered against the dispatch it brackets,
        // so a pair can come back inverted. Dropping those is honest; counting
        // them as zero would quietly shrink whichever kernel is noisiest.
        if (b <= a) continue;
        auto& slot = table[I.step.trace_labels[i]];
        slot.first += double(b - a);
        slot.second += 1;
        total_ticks += double(b - a);
    }
    total_gpu_ms += gpu_ms;
    if (++fires % every != 0) return;
    std::vector<std::pair<std::string, std::pair<double, long>>> rows(table.begin(), table.end());
    std::sort(rows.begin(), rows.end(),
              [](const auto& x, const auto& y) { return x.second.first > y.second.first; });
    fprintf(stderr, "[trace] %ld fires, %.2f ms of GPU, %zu kernels\n", fires, total_gpu_ms,
            rows.size());
    for (const auto& [name, v] : rows) {
        const double share = total_ticks > 0 ? v.first / total_ticks : 0.0;
        fprintf(stderr, "[trace]  %6.2f%%  %8.3f ms  n=%-6ld %s\n", 100.0 * share,
                share * total_gpu_ms, v.second, name.c_str());
    }
    table.clear();
    total_ticks = 0.0;
    total_gpu_ms = 0.0;
}

StepTiming RawMetalContext::run_step(const std::function<void(StepEncoder&)>& encode_fn,
                                     int ab) {
    return run_steps({encode_fn}, ab);
}

StepTiming RawMetalContext::run_steps(
    const std::vector<std::function<void(StepEncoder&)>>& encode_fns,
    int ab) {
    auto& I = *impl_;
    StepTiming tm;
    ab &= 1;
    const GpuCommitFeedback pending = last_commit_feedback();
    if (pending.had_error && pending.event_value > I.surfaced_feedback_error) {
        I.surfaced_feedback_error = pending.event_value;
        I.wedged = true;
        tm.gpu_error = true;
        tm.gpu_error_text = pending.error;
        return tm;
    }
    if (I.wedged) {
        tm.timed_out = true;
        return tm;
    }
    if (encode_fns.empty()) {
        tm.completed = true;
        return tm;
    }

    double t0 = nowms();
    [I.alloc[ab] reset];
    // One allocator can back several command buffers; `reset` only requires
    // that every buffer drawn from it has completed, which the wait below
    // guarantees before the next reset.
    std::vector<id<MTL4CommandBuffer>> cbs;
    cbs.reserve(encode_fns.size());
    for (const auto& encode_fn : encode_fns) {
        cbs.push_back((__bridge_transfer id<MTL4CommandBuffer>)
            encode_one_command_buffer(&I, ab, encode_fn));
    }
    double t1 = nowms();

    const uint64_t signalled = I.commit_and_signal(cbs.data(), cbs.size(), 0);

    const auto wait_begin = M0TimingCounters::Clock::now();
    if (I.await_event(signalled)) {
        tm.completed = true;
    } else {
        tm.timed_out = true;
        tm.completed = false;
    }
    m0_timing_counters().record_forward_wait(
        M0TimingCounters::Clock::now() - wait_begin);
    double t2 = nowms();

    tm.encode_ms   = t1 - t0;
    tm.gpu_exec_ms = t2 - t1;
    apply_commit_feedback(*this, signalled, tm);
    report_dispatch_trace(I, tm.gpu_ms > 0.0 ? tm.gpu_ms : tm.gpu_exec_ms);
    return tm;
}

StepTiming RawMetalContext::run_segments(
    const std::vector<std::function<void(StepEncoder&)>>& encode_fns,
    const std::function<void(std::size_t)>& between,
    int ab) {
    auto& I = *impl_;
    StepTiming tm;
    ab &= 1;
    const GpuCommitFeedback pending = last_commit_feedback();
    if (pending.had_error && pending.event_value > I.surfaced_feedback_error) {
        I.surfaced_feedback_error = pending.event_value;
        I.wedged = true;
        tm.gpu_error = true;
        tm.gpu_error_text = pending.error;
        return tm;
    }
    if (I.wedged) {
        tm.timed_out = true;
        return tm;
    }
    if (encode_fns.empty()) {
        tm.completed = true;
        return tm;
    }

    std::uint64_t last_signal = 0;
    for (std::size_t i = 0; i < encode_fns.size(); ++i) {
        const double t0 = nowms();
        // Legal every time round: `reset` needs every buffer drawn from this
        // allocator to have completed, and the previous iteration waited for
        // exactly that. Resetting per segment rather than once is what keeps a
        // long model from growing the allocator by its layer count.
        [I.alloc[ab] reset];
        id<MTL4CommandBuffer> cb = (__bridge_transfer id<MTL4CommandBuffer>)
            encode_one_command_buffer(&I, ab, encode_fns[i]);
        const double t1 = nowms();

        last_signal = I.commit_and_signal(&cb, 1, 0);
        const auto wait_begin = M0TimingCounters::Clock::now();
        const bool ok = I.await_event(last_signal);
        m0_timing_counters().record_forward_wait(
            M0TimingCounters::Clock::now() - wait_begin);
        const double t2 = nowms();

        tm.encode_ms += t1 - t0;
        tm.gpu_exec_ms += t2 - t1;
        if (!ok) {
            // A segment that never finished leaves the host holding results
            // that were never computed, so `between` is not called and the
            // remaining segments are not encoded.
            tm.timed_out = true;
            tm.completed = false;
            apply_commit_feedback(*this, last_signal, tm);
            return tm;
        }
        if (between) between(i);
    }
    tm.completed = true;
    apply_commit_feedback(*this, last_signal, tm);
    return tm;
}

void RawMetalContext::force_next_wait_timeout_for_test() {
    impl_->force_wait_timeout_once.store(true);
}

uint64_t RawMetalContext::last_event() const { return impl_->ev_val; }

// ── Continuous-async keepalive ───────────────────────────────────────────────
void RawMetalContext::start_keepalive(uint32_t spin_iters, uint32_t threadgroups,
                                      uint32_t depth) {
    auto& I = *impl_;
    if (I.ka_run.load()) return;
    if (depth < 2) depth = 2;
    if (threadgroups < 1) threadgroups = 1;

    // Lazily build the keepalive queue + spin PSO on first use.
    if (I.ka_queue == nil) {
        I.ka_queue = [I.dev newMTL4CommandQueue];
        I.ka_alloc = [I.dev newCommandAllocator];
        I.ka_event = [I.dev newSharedEvent];

        const char* src = R"(
#include <metal_stdlib>
using namespace metal;
kernel void ka_spin(device atomic_uint* sink   [[buffer(0)]],
                    constant uint&      iters  [[buffer(1)]],
                    uint                tid    [[thread_position_in_grid]]) {
    uint acc = tid * 2654435761u + 1u;
    for (uint i = 0; i < iters; ++i) acc = acc * 1664525u + 1013904223u;
    if (acc == 0xFFFFFFFFu)  // never true in practice; defeats dead-code elimination
        atomic_fetch_add_explicit(sink, acc, memory_order_relaxed);
}
)";
        Pso p = compile_pso(src, "ka_spin", nullptr);
        if (!p.valid()) { fprintf(stderr, "[pie-metal] keepalive PSO compile failed\n"); return; }
        I.ka_pso = (__bridge id<MTLComputePipelineState>)p.obj;

        I.ka_sink  = [I.dev newBufferWithLength:sizeof(uint32_t)
                                        options:MTLResourceStorageModeShared];
        I.ka_iters = [I.dev newBufferWithLength:sizeof(uint32_t)
                                        options:MTLResourceStorageModeShared];
        *static_cast<uint32_t*>(I.ka_iters.contents) = spin_iters;

        MTL4ArgumentTableDescriptor* ad = [MTL4ArgumentTableDescriptor new];
        ad.maxBufferBindCount = 2;
        NSError* e = nil;
        I.ka_at4 = [I.dev newArgumentTableWithDescriptor:ad error:&e];
        if (I.ka_at4 == nil) { fprintf(stderr, "[pie-metal] keepalive argtable failed\n"); return; }
        [I.ka_at4 setAddress:I.ka_sink.gpuAddress  atIndex:0];
        [I.ka_at4 setAddress:I.ka_iters.gpuAddress atIndex:1];
    } else {
        *static_cast<uint32_t*>(I.ka_iters.contents) = spin_iters;
    }

    // A keepalive-local residency set covering the sink/iters buffers.
    NSError* e = nil;
    MTLResidencySetDescriptor* rsd = [MTLResidencySetDescriptor new];
    id<MTLResidencySet> ka_rs = [I.dev newResidencySetWithDescriptor:rsd error:&e];
    [ka_rs addAllocation:I.ka_sink];
    [ka_rs addAllocation:I.ka_iters];
    [ka_rs commit];

    I.ka_run.store(true);
    const uint32_t tg = threadgroups;
    const uint32_t inflight = depth;
    I.ka_thread = std::thread([&I, ka_rs, tg, inflight]() {
        uint64_t committed = 0;
        const MTLSize grid = MTLSizeMake(tg * 64, 1, 1);     // 64 threads/threadgroup
        const MTLSize tgsz = MTLSizeMake(64, 1, 1);
        while (I.ka_run.load(std::memory_order_relaxed)) {
            // Bound in-flight to `inflight` without ever fully draining (keeps overlap).
            if (committed >= inflight)
                [I.ka_event waitUntilSignaledValue:(committed - inflight + 1) timeoutMS:5000];
            [I.ka_alloc reset];
            id<MTL4CommandBuffer> cb = [I.dev newCommandBuffer];
            [cb beginCommandBufferWithAllocator:I.ka_alloc];
            [cb useResidencySet:ka_rs];
            id<MTL4ComputeCommandEncoder> en = [cb computeCommandEncoder];
            [en setComputePipelineState:I.ka_pso];
            [en setArgumentTable:I.ka_at4];
            [en dispatchThreads:grid threadsPerThreadgroup:tgsz];
            [en endEncoding];
            [cb endCommandBuffer];
            [I.ka_queue commit:&cb count:1];
            [I.ka_queue signalEvent:I.ka_event value:++committed];
        }
        // Drain remaining in-flight before returning.
        [I.ka_event waitUntilSignaledValue:committed timeoutMS:5000];
    });
}

void RawMetalContext::stop_keepalive() {
    auto& I = *impl_;
    if (!I.ka_run.load()) return;
    I.ka_run.store(false);
    if (I.ka_thread.joinable()) I.ka_thread.join();
}

}  // namespace pie::metal
