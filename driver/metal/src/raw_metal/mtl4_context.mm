// mtl4_context.mm — Obj-C++ implementation of the Metal-4 wrapper scaffold.
//
// Implements RawMetalContext on top of the Metal-4 objects verified working in
// beta's mtl4probe.mm: MTL4CommandQueue / double-buffered MTL4CommandAllocator /
// MTL4CommandBuffer / MTL4ComputeCommandEncoder / MTL4ArgumentTable / MTLResidencySet /
// MTL4Compiler (runtime newLibraryWithSource — no offline metallib needed on this box).
//
// Heap model: ONE placement MTLHeap (Shared storage, UMA) bump-sub-allocated by
// heap_alloc; the whole heap is made resident ONCE via an MTLResidencySet (I2).
// Argument tables are built ONCE per (Kernel, layer) dispatch instance (I2); only IO
// slot CONTENTS change per token (I1) so the encoded command buffer stays byte-identical.

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

#include "mtl4_context.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <strings.h>
#include <algorithm>
#include <atomic>
#include <thread>
#include <unordered_map>
#include <unordered_set>

namespace pie_metal_driver::raw_metal {

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
}  // namespace

// ── Per-step encoder state (transient, lives across one run_step) ─────────────
struct StepState {
    id<MTL4ComputeCommandEncoder> en = nil;
    RawMetalContext::Impl*        ctx = nullptr;
};

// ── RawMetalContext::Impl — owns the Metal-4 device objects ───────────────────
struct RawMetalContext::Impl {
    id<MTLDevice>            dev   = nil;
    id<MTL4CommandQueue>     queue = nil;
    id<MTL4CommandAllocator> alloc[2] = {nil, nil};   // double-buffered
    id<MTLHeap>              heap  = nil;
    id<MTLResidencySet>      rs    = nil;
    id<MTL4Compiler>         compiler = nil;
    id<MTLSharedEvent>       event = nil;
    uint64_t                 ev_val = 0;

    size_t heap_size = 0;
    size_t bump      = 0;             // running heap offset
    bool   resident  = false;         // make_resident() idempotency guard

    NSMutableArray*           retained = nil;  // keeps PSOs / sub-buffers alive
    NSMutableDictionary*      argtables = nil;  // NSNumber(argkey) -> id<MTL4ArgumentTable>
    std::unordered_set<uint64_t> bound_arg_slots;
    std::unordered_map<uint64_t, uint64_t> bound_arg_addresses;

    // Phase 3 (review item 4): standalone-buffer allocation accounting (the
    // paged-KV pool buffers only, NOT heap sub-allocations). Kept exact by
    // create_standalone_buffer / release_standalone_buffer so a lifecycle
    // probe can prove grow/shrink does not leak.
    size_t standalone_count = 0;
    size_t standalone_bytes = 0;

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
};

id<MTL4ArgumentTable> RawMetalContext::Impl::argtable_for(int ordinal, bool create) {
    NSNumber* key = @(argkey(ordinal));
    id<MTL4ArgumentTable> t = argtables[key];
    if (t == nil && create) {
        MTL4ArgumentTableDescriptor* ad = [MTL4ArgumentTableDescriptor new];
        ad.maxBufferBindCount = 16;  // widest kernel (Sdpa=8) with margin
        NSError* e = nil;
        t = [dev newArgumentTableWithDescriptor:ad error:&e];
        if (t == nil) {
            fprintf(stderr, "[raw_metal] argtable create failed: %s\n",
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
    [s->en setComputePipelineState:(__bridge id<MTLComputePipelineState>)pso.obj];
}
void StepEncoder::set_argtable(Kernel k, int ordinal) {
    (void)k;  // decorative tag; ordinal is the key
    set_argtable_ordinal(ordinal);
}
void StepEncoder::set_argtable_ordinal(int ordinal) {
    auto* s = static_cast<StepState*>(impl_);
    id<MTL4ArgumentTable> t = s->ctx->argtable_for(ordinal, /*create=*/false);
    if (t == nil) {
        fprintf(stderr, "[raw_metal] no argument table bound for ordinal=%d\n", ordinal);
        return;
    }
    [s->en setArgumentTable:t];
}
void StepEncoder::dispatch(Grid grid, Threadgroup tg) {
    auto* s = static_cast<StepState*>(impl_);
    [s->en dispatchThreads:MTLSizeMake(grid.x, grid.y, grid.z)
        threadsPerThreadgroup:MTLSizeMake(tg.x, tg.y, tg.z)];
}
// Map the pure-C++ BarrierVisibility to MTL4VisibilityOptions, honoring a one-shot
// `PIE_BARRIER_VIS=none|device` global override (delta's visibility sweep): when set it
// forces ALL barriers regardless of the per-call argument; when absent the per-call arg
// wins (beta's per-edge hazard model: Device for true heap-RAW, None for ordering-only).
static MTL4VisibilityOptions resolve_barrier_vis(BarrierVisibility req) {
    static const int override_mode = [] {
        const char* e = getenv("PIE_BARRIER_VIS");
        if (!e) return -1;
        if (strcasecmp(e, "none") == 0 || strcmp(e, "0") == 0) return 0;
        if (strcasecmp(e, "device") == 0 || strcmp(e, "1") == 0) return 1;
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
RawMetalContext::~RawMetalContext() = default;

std::unique_ptr<RawMetalContext> RawMetalContext::create(size_t heap_bytes) {
    auto ctx = std::unique_ptr<RawMetalContext>(new RawMetalContext());
    auto& I = *ctx->impl_;

    I.dev = MTLCreateSystemDefaultDevice();
    if (I.dev == nil) { fprintf(stderr, "[raw_metal] no Metal device\n"); return nullptr; }

    I.queue    = [I.dev newMTL4CommandQueue];
    I.alloc[0] = [I.dev newCommandAllocator];
    I.alloc[1] = [I.dev newCommandAllocator];
    I.event    = [I.dev newSharedEvent];

    NSError* e = nil;
    MTL4CompilerDescriptor* cd = [MTL4CompilerDescriptor new];
    I.compiler = [I.dev newCompilerWithDescriptor:cd error:&e];
    if (I.compiler == nil) {
        fprintf(stderr, "[raw_metal] compiler create failed: %s\n",
                e.localizedDescription.UTF8String);
        return nullptr;
    }

    MTLHeapDescriptor* hd = [MTLHeapDescriptor new];
    hd.type        = MTLHeapTypePlacement;
    hd.storageMode = MTLStorageModeShared;   // UMA: contents() valid for all slots
    hd.size        = heap_bytes;
    I.heap = [I.dev newHeapWithDescriptor:hd];
    if (I.heap == nil) {
        fprintf(stderr, "[raw_metal] heap alloc failed (%zu bytes)\n", heap_bytes);
        return nullptr;
    }
    I.heap_size = heap_bytes;

    MTLResidencySetDescriptor* rsd = [MTLResidencySetDescriptor new];
    I.rs = [I.dev newResidencySetWithDescriptor:rsd error:&e];
    if (I.rs == nil) {
        fprintf(stderr, "[raw_metal] residency set failed: %s\n",
                e.localizedDescription.UTF8String);
        return nullptr;
    }

    I.retained  = [NSMutableArray new];
    I.argtables = [NSMutableDictionary new];
    return ctx;
}

SlotHandle RawMetalContext::heap_alloc(size_t size, size_t align) {
    auto& I = *impl_;
    SlotHandle h;
    if (size == 0) return h;

    MTLResourceOptions opts = MTLResourceStorageModeShared;
    MTLSizeAndAlign sa = [I.dev heapBufferSizeAndAlignWithLength:size options:opts];
    size_t a = align > sa.align ? align : sa.align;
    size_t off = align_up(I.bump, a);
    if (off + sa.size > I.heap_size) {
        fprintf(stderr, "[raw_metal] heap OOM: need %zu at off %zu, cap %zu\n",
                sa.size, off, I.heap_size);
        return h;
    }
    id<MTLBuffer> buf = [I.heap newBufferWithLength:size options:opts offset:off];
    if (buf == nil) { fprintf(stderr, "[raw_metal] placement buffer failed\n"); return h; }
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
        fprintf(stderr, "[raw_metal] standalone buffer alloc failed (%zu bytes)\n", size);
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
    id<MTL4ArgumentTable> t = I.argtable_for(ordinal, /*create=*/true);
    if (t == nil) return;
    [t setAddress:(slot.gpu_address + offset) atIndex:bind_index];
    const uint64_t key = (uint64_t(uint32_t(ordinal)) << 8) | bind_index;
    I.bound_arg_slots.insert(key);
    I.bound_arg_addresses[key] = slot.gpu_address + offset;
}

bool RawMetalContext::arg_slot_is_bound(int ordinal, uint8_t bind_index) const {
    const auto key = (uint64_t(uint32_t(ordinal)) << 8) | bind_index;
    return impl_->bound_arg_slots.find(key) != impl_->bound_arg_slots.end();
}

uint64_t RawMetalContext::arg_slot_address(int ordinal, uint8_t bind_index) const {
    const auto key = (uint64_t(uint32_t(ordinal)) << 8) | bind_index;
    const auto it = impl_->bound_arg_addresses.find(key);
    return it == impl_->bound_arg_addresses.end() ? 0 : it->second;
}

Pso RawMetalContext::compile_pso(const std::string& src, const std::string& fn,
                                 std::string* error) {
    auto& I = *impl_;
    Pso out;
    NSError* e = nil;
    id<MTLLibrary> lib =
        [I.dev newLibraryWithSource:[NSString stringWithUTF8String:src.c_str()]
                            options:nil
                              error:&e];
    if (lib == nil) {
        if (error) *error = e.localizedDescription.UTF8String;
        return out;
    }
    MTL4LibraryFunctionDescriptor* fd = [MTL4LibraryFunctionDescriptor new];
    fd.name = [NSString stringWithUTF8String:fn.c_str()];
    fd.library = lib;
    MTL4ComputePipelineDescriptor* pd = [MTL4ComputePipelineDescriptor new];
    pd.computeFunctionDescriptor = fd;
    id<MTLComputePipelineState> pso =
        [I.compiler newComputePipelineStateWithDescriptor:pd
                                      compilerTaskOptions:nil
                                                    error:&e];
    if (pso == nil) {
        if (error) *error = e.localizedDescription.UTF8String;
        return out;
    }
    [I.retained addObject:pso];
    out.obj = (__bridge void*)pso;
    return out;
}

Pso RawMetalContext::compile_pso_from_file(const std::string& path, const std::string& fn,
                                           std::string* error) {
    NSError* e = nil;
    NSString* src = [NSString stringWithContentsOfFile:[NSString stringWithUTF8String:path.c_str()]
                                              encoding:NSUTF8StringEncoding
                                                 error:&e];
    if (src == nil) {
        if (error) *error = std::string("read failed: ") + e.localizedDescription.UTF8String;
        return Pso{};
    }
    return compile_pso(src.UTF8String, fn, error);
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
        fprintf(stderr, "[raw_metal] timestamp heap create failed (%u): %s\n",
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
        fprintf(stderr, "[raw_metal] timestamp resolve returned nil (count=%u)\n", count);
        return;
    }
    const auto* entries = static_cast<const MTL4TimestampHeapEntry*>(data.bytes);
    const uint32_t n = std::min<uint32_t>(count, uint32_t(data.length / sizeof(MTL4TimestampHeapEntry)));
    for (uint32_t i = 0; i < n; ++i) out[i] = entries[i].timestamp;
}

StepTiming RawMetalContext::run_step(const std::function<void(StepEncoder&)>& encode_fn,
                                     int ab) {
    auto& I = *impl_;
    StepTiming tm;
    ab &= 1;

    double t0 = nowms();
    [I.alloc[ab] reset];
    id<MTL4CommandBuffer> cb = [I.dev newCommandBuffer];
    [cb beginCommandBufferWithAllocator:I.alloc[ab]];
    [cb useResidencySet:I.rs];
    id<MTL4ComputeCommandEncoder> en = [cb computeCommandEncoder];

    I.step.en  = en;
    I.step.ctx = &I;
    StepEncoder se(&I.step);
    encode_fn(se);

    [en endEncoding];
    [cb endCommandBuffer];
    double t1 = nowms();

    [I.queue commit:&cb count:1];
    [I.queue signalEvent:I.event value:++I.ev_val];
    [I.event waitUntilSignaledValue:I.ev_val timeoutMS:5000];
    double t2 = nowms();

    I.step.en = nil;
    tm.encode_ms   = t1 - t0;
    tm.gpu_exec_ms = t2 - t1;
    return tm;
}

uint64_t RawMetalContext::commit_step_async(const std::function<void(StepEncoder&)>& encode_fn,
                                            int ab) {
    return commit_step_async_dep(encode_fn, ab, 0);
}

uint64_t RawMetalContext::commit_step_async_dep(const std::function<void(StepEncoder&)>& encode_fn,
                                                int ab, uint64_t wait_value) {
    auto& I = *impl_;
    ab &= 1;
    // Caller guarantees the allocator for `ab` is free (its prior step completed) via
    // sync_event() before reuse — depth-2 pipelining over the two double-buffered allocators.
    [I.alloc[ab] reset];
    id<MTL4CommandBuffer> cb = [I.dev newCommandBuffer];
    [cb beginCommandBufferWithAllocator:I.alloc[ab]];
    [cb useResidencySet:I.rs];
    id<MTL4ComputeCommandEncoder> en = [cb computeCommandEncoder];
    I.step.en  = en;
    I.step.ctx = &I;
    StepEncoder se(&I.step);
    encode_fn(se);
    [en endEncoding];
    [cb endCommandBuffer];
    // GPU-side serialization: make this CB wait for `wait_value` (the prior step's completion)
    // on the queue timeline before executing. This enforces the autoregressive token dependency
    // (single-stream steps CANNOT overlap on the GPU) while keeping the HOST non-blocking, so
    // the host CB-build for step i+1 overlaps GPU(i) and step i+1 starts the instant i finishes
    // (zero host gap -> holds the clock). Without it the GPU overlaps independent CBs = the
    // throughput regime, not single-stream.
    if (wait_value > 0) [I.queue waitForEvent:I.event value:wait_value];
    [I.queue commit:&cb count:1];
    [I.queue signalEvent:I.event value:++I.ev_val];
    I.step.en = nil;
    return I.ev_val;
}

void RawMetalContext::sync_event(uint64_t value) {
    [impl_->event waitUntilSignaledValue:value timeoutMS:5000];
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
        if (!p.valid()) { fprintf(stderr, "[raw_metal] keepalive PSO compile failed\n"); return; }
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
        if (I.ka_at4 == nil) { fprintf(stderr, "[raw_metal] keepalive argtable failed\n"); return; }
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

}  // namespace pie_metal_driver::raw_metal
