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
#include "observability.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <strings.h>
#include <unistd.h>
#include <algorithm>
#include <atomic>
#include <limits>
#include <map>
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
    const MTLSizeAndAlign sa = [device
        heapBufferSizeAndAlignWithLength:1
        options:MTLResourceStorageModeShared];
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
    id<MTL4PipelineDataSetSerializer> pipeline_serializer = nil;
    id<MTLSharedEvent>       event = nil;
    uint64_t                 ev_val = 0;

    size_t heap_size = 0;
    size_t bump      = 0;             // running heap offset
    bool   resident  = false;         // make_resident() idempotency guard

    NSMutableArray*           retained = nil;  // keeps PSOs / sub-buffers alive
    std::unordered_set<void*> retained_psos;
    NSMutableDictionary*      argtables = nil;  // NSNumber(argkey) -> id<MTL4ArgumentTable>
    std::unordered_set<uint64_t> bound_arg_slots;
    std::unordered_map<uint64_t, uint64_t> bound_arg_addresses;
    std::unordered_map<void*, size_t> external_allocations;
    bool saw_ptir_compile = false;
    bool last_ptir_fast_math_disabled = false;
    std::atomic<bool> force_wait_timeout_once{false};

    // Standalone-buffer allocation accounting (all non-heap buffers,
    // including buffers retained by the transient pool).
    size_t standalone_count = 0;
    size_t standalone_bytes = 0;

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
};

id<MTL4ArgumentTable> RawMetalContext::Impl::argtable_for(int ordinal, bool create) {
    NSNumber* key = @(argkey(ordinal));
    id<MTL4ArgumentTable> t = argtables[key];
    if (t == nil && create) {
        MTL4ArgumentTableDescriptor* ad = [MTL4ArgumentTableDescriptor new];
        ad.maxBufferBindCount = 31;  // M1 readiness: status + lane + up to 29 channels
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
        fprintf(stderr, "[pie-metal] no argument table bound for ordinal=%d\n", ordinal);
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
    if (I.dev == nil) { fprintf(stderr, "[pie-metal] no Metal device\n"); return nullptr; }

    I.queue    = [I.dev newMTL4CommandQueue];
    I.alloc[0] = [I.dev newCommandAllocator];
    I.alloc[1] = [I.dev newCommandAllocator];
    I.event    = [I.dev newSharedEvent];

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

SlotHandle RawMetalContext::heap_alloc(size_t size, size_t align) {
    auto& I = *impl_;
    SlotHandle h;
    if (size == 0) return h;

    MTLResourceOptions opts = MTLResourceStorageModeShared;
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
    constexpr size_t kMaxCachedPerSizeClass = 8;
    if (bucket.size() < kMaxCachedPerSizeClass &&
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

void RawMetalContext::release_argtable_ordinal(int ordinal) {
    auto& I = *impl_;
    [I.argtables removeObjectForKey:@(argkey(ordinal))];
    for (auto iterator = I.bound_arg_slots.begin();
         iterator != I.bound_arg_slots.end();) {
        if (static_cast<std::uint32_t>(*iterator >> 8) ==
            static_cast<std::uint32_t>(ordinal)) {
            I.bound_arg_addresses.erase(*iterator);
            iterator = I.bound_arg_slots.erase(iterator);
        } else {
            ++iterator;
        }
    }
}

namespace {

Pso compile_pso_impl(
    RawMetalContext::Impl& I,
    const std::string& src,
    const std::string& fn,
    MTLCompileOptions* options,
    MTL4CompilerTaskOptions* task_options,
    std::string* error) {
    Pso out;
    NSError* e = nil;
    id<MTLLibrary> lib =
        [I.dev newLibraryWithSource:[NSString stringWithUTF8String:src.c_str()]
                            options:options
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

bool read_metal_source(
    const std::string& path,
    std::string& source,
    std::string* error) {
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
    return true;
}

void configure_ptir_math_options(
    MTLCompileOptions* options,
    bool& strict_math) {
    if (@available(macOS 15.0, *)) {
        options.mathMode = MTLMathModeSafe;
        options.mathFloatingPointFunctions =
            MTLMathFloatingPointFunctionsPrecise;
        strict_math = options.mathMode == MTLMathModeSafe;
        return;
    }
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
    options.fastMathEnabled = NO;
    strict_math = options.fastMathEnabled == NO;
#pragma clang diagnostic pop
}

}  // namespace

bool read_ptir_msl_source(
    const std::string& path,
    std::string& source,
    std::string* error) {
    if (!read_metal_source(path, source, error)) return false;
    constexpr std::string_view include =
        "#include \"ptir_rng.generated.metal\"";
    const std::size_t first = source.find(include);
    if (first == std::string::npos) return true;

    const std::size_t separator = path.find_last_of("/\\");
    const std::string preamble_path =
        (separator == std::string::npos
             ? std::string{}
             : path.substr(0, separator + 1)) +
        "ptir_rng.generated.metal";
    std::string preamble;
    if (!read_metal_source(preamble_path, preamble, error)) return false;
    for (std::size_t position = first;
         position != std::string::npos;
         position = source.find(include, position + preamble.size())) {
        source.replace(position, include.size(), preamble);
    }
    return true;
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
    return read_ptir_msl_source(path, source, error)
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
    const auto wait_begin = M0TimingCounters::Clock::now();
    BOOL signaled = I.force_wait_timeout_once.exchange(false)
                        ? NO
                        : [I.event waitUntilSignaledValue:I.ev_val
                                               timeoutMS:5000];
    tm.timed_out = signaled == NO;
    if (tm.timed_out) {
        m0_timing_counters().record_forward_wait_timeout();
    }
    while (signaled == NO) {
        signaled = [I.event waitUntilSignaledValue:I.ev_val
                                        timeoutMS:5000];
    }
    tm.completed = true;
    m0_timing_counters().record_forward_wait(
        M0TimingCounters::Clock::now() - wait_begin);
    double t2 = nowms();

    I.step.en = nil;
    tm.encode_ms   = t1 - t0;
    tm.gpu_exec_ms = t2 - t1;
    return tm;
}

void RawMetalContext::force_next_wait_timeout_for_test() {
    impl_->force_wait_timeout_once.store(true);
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
    while (![impl_->event waitUntilSignaledValue:value timeoutMS:5000]) {
    }
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
