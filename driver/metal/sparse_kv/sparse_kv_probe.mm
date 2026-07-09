// sparse_kv_probe.mm — Metal 4 placement sparse buffer: KV-cache elasticity verification
//
// Tests whether Metal 4 placement sparse buffers give a CUDA-VMM-style elastic KV model:
// reserve a large VIRTUAL buffer with no backing, map/unmap placement-heap tiles onto
// byte ranges, recycle tiles in a chunked heap pool, and reclaim OS footprint by
// releasing empty heap chunks. See README.md for the claim and measured results.
//
// Build (CommandLineTools OK; runtime shader compile, no offline metal compiler needed):
//   clang++ -fobjc-arc -fmodules -O2 -framework Metal -framework Foundation \
//         sparse_kv_probe.mm -o sparse_kv_probe && ./sparse_kv_probe
//
// Environment verified on: Apple M1 Max, macOS 26.3 (25D125), Metal 4.
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#import <mach/mach.h>

static double phys_mb(void){ task_vm_info_data_t i; mach_msg_type_number_t c=TASK_VM_INFO_COUNT;
    task_info(mach_task_self(),TASK_VM_INFO,(task_info_t)&i,&c); return i.phys_footprint/1e6; }
static double nowms(void){ return [NSDate date].timeIntervalSince1970*1000.0; }

static id<MTLDevice> gDev;
static void rep(NSString*l){ NSLog(@"  %-38@ metal.alloc=%8.2f MB  proc.phys=%8.2f MB", l, gDev.currentAllocatedSize/1e6, phys_mb()); }

// map/unmap helper (blocks until GPU mapping op completes)
static void doMap(id<MTL4CommandQueue> mq, id<MTLSharedEvent> ev, uint64_t*v,
                  id<MTLBuffer> buf, id<MTLHeap> heap, BOOL map,
                  NSUInteger bufTileOff, NSUInteger nTiles, NSUInteger heapTileOff){
    MTL4UpdateSparseBufferMappingOperation op = {
        .mode = map ? MTLSparseTextureMappingModeMap : MTLSparseTextureMappingModeUnmap,
        .bufferRange = NSMakeRange(bufTileOff, nTiles),
        .heapOffset = heapTileOff,
    };
    [mq updateBufferMappings:buf heap:(map?heap:nil) operations:&op count:1];
    [mq signalEvent:ev value:++(*v)];
    [ev waitUntilSignaledValue:*v timeoutMS:10000];
}

int main(void){ @autoreleasepool {
    gDev = MTLCreateSystemDefaultDevice();
    id<MTL4CommandQueue> mq = [gDev newMTL4CommandQueue];
    id<MTLCommandQueue>  lq = [gDev newCommandQueue];
    id<MTLSharedEvent>   ev = [gDev newSharedEvent];
    uint64_t v = 0;

    const MTLSparsePageSize PS = MTLSparsePageSize64;
    const NSUInteger TILE = [gDev sparseTileSizeInBytesForSparsePageSize:PS];  // 64 KB
    const NSUInteger VIRT = 8ull<<30;                                          // 8 GB virtual
    NSLog(@"tile=%luKB virt=%.0fGB", (unsigned long)TILE/1024, VIRT/1e9);

    // ---- P1: virtual buffer is free; heap is the physical cost ----
    NSLog(@"\n=== P1: virtual buffer free, heap = physical, map adds nothing ===");
    rep(@"start");
    id<MTLBuffer> buf = [gDev newBufferWithLength:VIRT options:MTLResourceStorageModePrivate placementSparsePageSize:PS];
    rep(@"after 8GB VIRTUAL sparse buf");

    const NSUInteger HEAP = 128ull<<20;              // 128 MB pool
    const NSUInteger HTILES = HEAP/TILE;             // 2048 tiles
    MTLHeapDescriptor *hd = [MTLHeapDescriptor new];
    hd.type=MTLHeapTypePlacement; hd.storageMode=MTLStorageModePrivate; hd.size=HEAP;
    hd.maxCompatiblePlacementSparsePageSize=PS;
    id<MTLHeap> heap = [gDev newHeapWithDescriptor:hd];
    rep(@"after 128MB placement heap");
    doMap(mq,ev,&v,buf,heap,YES,0,HTILES,0);
    rep(@"after MAP all 2048 tiles");

    // ---- P2: data real (GPU write, blit read) ----
    NSLog(@"\n=== P2: mapped tiles hold real data ===");
    @autoreleasepool {
        id<MTLLibrary> lib=[gDev newLibraryWithSource:
            @"#include <metal_stdlib>\nusing namespace metal;\n"
             "kernel void f(device uint*p[[buffer(0)]],uint g[[thread_position_in_grid]]){p[g]=g*2654435761u;}"
            options:nil error:nil];
        id<MTLComputePipelineState> pso=[gDev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"f"] error:nil];
        id<MTLBuffer> stg=[gDev newBufferWithLength:HEAP options:MTLResourceStorageModeShared];
        id<MTLCommandBuffer> cb=[lq commandBuffer];
        [cb encodeWaitForEvent:ev value:v];
        id<MTLComputeCommandEncoder> ce=[cb computeCommandEncoder];
        [ce setComputePipelineState:pso]; [ce useResource:buf usage:MTLResourceUsageWrite];
        [ce setBuffer:buf offset:0 atIndex:0];
        NSUInteger nw=HEAP/4;
        [ce dispatchThreads:MTLSizeMake(nw,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)]; [ce endEncoding];
        id<MTLBlitCommandEncoder> be=[cb blitCommandEncoder];
        [be copyFromBuffer:buf sourceOffset:0 toBuffer:stg destinationOffset:0 size:HEAP]; [be endEncoding];
        [cb commit]; [cb waitUntilCompleted];
        uint32_t*p=(uint32_t*)stg.contents; uint64_t bad=0;
        for(NSUInteger i=0;i<nw;i++) if(p[i]!=(uint32_t)(i*2654435761u)) bad++;
        NSLog(@"  GPU wrote %.0fMB -> mismatches=%llu %@", HEAP/1e6, bad, bad?@"[FAIL]":@"[PASS]");
    }

    // ---- P3: unmap alone does NOT reduce footprint; releasing heap does ----
    NSLog(@"\n=== P3: unmap != free; heap release == free ===");
    rep(@"before unmap (heap still alive)");
    doMap(mq,ev,&v,buf,heap,NO,0,HTILES,0);
    rep(@"after UNMAP all tiles (heap kept)");     // expect ~unchanged: tiles recycled into MY heap
    heap=nil;
    rep(@"after RELEASE heap object");             // expect metal.alloc drops by ~128MB
    buf=nil;
    rep(@"after RELEASE virtual buffer");

    // ---- P4: elastic chunked pool (grow peak, then shrink real footprint) ----
    NSLog(@"\n=== P4: elastic chunked-heap KV pool (peak grows, then footprint shrinks) ===");
    const NSUInteger CHUNK=64ull<<20; const NSUInteger CT=CHUNK/TILE; const int NCHUNK=8;   // 8 x 64MB = 512MB peak
    id<MTLBuffer> kv=[gDev newBufferWithLength:VIRT options:MTLResourceStorageModePrivate placementSparsePageSize:PS];
    NSMutableArray *chunks=[NSMutableArray array];
    rep(@"pool start (only virtual kv buf)");
    for(int c=0;c<NCHUNK;c++){
        MTLHeapDescriptor*d=[MTLHeapDescriptor new];
        d.type=MTLHeapTypePlacement; d.storageMode=MTLStorageModePrivate; d.size=CHUNK; d.maxCompatiblePlacementSparsePageSize=PS;
        id<MTLHeap> h=[gDev newHeapWithDescriptor:d]; [chunks addObject:h];
        doMap(mq,ev,&v,kv,h, YES, (NSUInteger)c*CT, CT, 0);   // map chunk c onto kv tiles [c*CT, +CT)
    }
    rep([NSString stringWithFormat:@"PEAK: %d chunks mapped (%luMB)",NCHUNK,(unsigned long)(NCHUNK*CHUNK)/(1<<20)]);
    // shrink: KV shrinks -> unmap upper 6 chunks and release those empty heap objects
    for(int c=NCHUNK-1;c>=2;c--){
        doMap(mq,ev,&v,kv,nil,NO,(NSUInteger)c*CT,CT,0);
        [chunks removeObjectAtIndex:c];              // release the now-empty 64MB heap chunk
    }
    rep(@"after SHRINK to 2 chunks (release 6 empty)");   // expect metal.alloc ~= 2*64MB
    [chunks removeAllObjects]; kv=nil;
    rep(@"after full release");

    // ---- P5: map/unmap latency sweep ----
    NSLog(@"\n=== P5: map/unmap latency vs #tiles (128MB heap) ===");
    id<MTLBuffer> pb=[gDev newBufferWithLength:VIRT options:MTLResourceStorageModePrivate placementSparsePageSize:PS];
    MTLHeapDescriptor*pd=[MTLHeapDescriptor new];
    pd.type=MTLHeapTypePlacement; pd.storageMode=MTLStorageModePrivate; pd.size=HEAP; pd.maxCompatiblePlacementSparsePageSize=PS;
    id<MTLHeap> ph=[gDev newHeapWithDescriptor:pd];
    NSUInteger sizes[]={1,8,64,512,2048};
    for(int k=0;k<5;k++){
        NSUInteger n=sizes[k]; int reps=50;
        double tmap=0,tun=0;
        for(int r=0;r<reps;r++){
            double a=nowms(); doMap(mq,ev,&v,pb,ph,YES,0,n,0); tmap+=nowms()-a;
            double b=nowms(); doMap(mq,ev,&v,pb,ph,NO, 0,n,0); tun +=nowms()-b;
        }
        NSLog(@"  n=%4lu tiles (%5.1fMB): map=%.3f ms  unmap=%.3f ms  (avg over %d, incl GPU rtrip)",
              (unsigned long)n, n*TILE/1e6, tmap/reps, tun/reps, reps);
    }
    ph=nil; pb=nil;
    NSLog(@"\nDONE.");
    return 0;
}}
