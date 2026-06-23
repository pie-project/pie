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

namespace pie_metal_driver::raw_metal {

namespace {
using clk = std::chrono::high_resolution_clock;
inline double nowms() {
    return std::chrono::duration<double, std::milli>(clk::now().time_since_epoch()).count();
}
inline size_t align_up(size_t v, size_t a) { return (v + (a - 1)) & ~(a - 1); }
// Stable integer key for an argument-table dispatch instance.
inline int argkey(Kernel k, int layer) {
    return (static_cast<int>(k) << 8) | (static_cast<uint8_t>(layer + 1));
}
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

    NSMutableArray*           retained = nil;  // keeps PSOs / sub-buffers alive
    NSMutableDictionary*      argtables = nil;  // NSNumber(argkey) -> id<MTL4ArgumentTable>

    StepState step;  // active step (during run_step)

    id<MTL4ArgumentTable> argtable_for(Kernel k, int layer, bool create);
};

id<MTL4ArgumentTable> RawMetalContext::Impl::argtable_for(Kernel k, int layer, bool create) {
    NSNumber* key = @(argkey(k, layer));
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
void StepEncoder::set_argtable(Kernel k, int layer) {
    auto* s = static_cast<StepState*>(impl_);
    id<MTL4ArgumentTable> t = s->ctx->argtable_for(k, layer, /*create=*/false);
    if (t == nil) {
        fprintf(stderr, "[raw_metal] no argument table bound for kernel=%d layer=%d\n",
                static_cast<int>(k), layer);
        return;
    }
    [s->en setArgumentTable:t];
}
void StepEncoder::dispatch(Grid grid, Threadgroup tg) {
    auto* s = static_cast<StepState*>(impl_);
    [s->en dispatchThreads:MTLSizeMake(grid.x, grid.y, grid.z)
        threadsPerThreadgroup:MTLSizeMake(tg.x, tg.y, tg.z)];
}
void StepEncoder::barrier() {
    auto* s = static_cast<StepState*>(impl_);
    [s->en barrierAfterQueueStages:MTLStageDispatch
                      beforeStages:MTLStageDispatch
                 visibilityOptions:MTL4VisibilityOptionDevice];
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

void RawMetalContext::make_resident() {
    auto& I = *impl_;
    [I.rs addAllocation:I.heap];   // whole heap resident ONCE (I2)
    [I.rs commit];
    [I.rs requestResidency];
}

void RawMetalContext::arg_bind(Kernel k, int layer, uint8_t bind_index, SlotHandle slot,
                               size_t offset) {
    auto& I = *impl_;
    id<MTL4ArgumentTable> t = I.argtable_for(k, layer, /*create=*/true);
    if (t == nil) return;
    [t setAddress:(slot.gpu_address + offset) atIndex:bind_index];
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

}  // namespace pie_metal_driver::raw_metal
