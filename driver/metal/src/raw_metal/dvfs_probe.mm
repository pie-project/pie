// dvfs_probe.mm — isolate WHICH GPU clock domain downclocks on idle (core vs memory/fabric).
//
// Two microkernels under a controlled idle-sweep:
//   alu_burn : pure-ALU (long dependent-FMA loop in registers, ~1 word global write) -> tracks GPU CORE clock.
//   mem_stream: pure global-memory streaming read+reduce over a 512MB buffer -> tracks DRAM/FABRIC bandwidth.
//
// For each idle_us we usleep (GPU queue empty -> GPU idles), then dispatch the kernel K times back-to-back
// and record per-dispatch GPU active time (GPUEndTime-GPUStartTime). Report steady-state median.
//
// FINDING (M1 Max, 24-core, 2026-06-24): the idle downclock is the GPU CORE clock, NOT memory.
//   alu_burn  inflates to EXACTLY 2.000x at >=300us idle (3.78 -> 7.56ms) = the 1296->648MHz core-clock ratio.
//   mem_stream is essentially FLAT (<=1.06x) across the whole sweep -> memory bandwidth (~333 GB/s) is IMMUNE.
//   powermetrics corroborates: GPU active freq drops 1296 -> ~644 MHz (2.01x) in lock-step with the idle phases.
//   => The 1.99x clock-ratio vs ~1.8x decode-time-ratio gap is the MEMORY-BOUND (weight-streaming) fraction of
//      decode being immune to the downclock, pulling the blended time ratio below the pure core-clock ratio.
//      No hidden second cause. The clean constant 2x multiplier (not front-loaded) also rules out a fixed
//      per-step wake-latency residual: it is a sustained clock-RATE effect.
//   Threshold (this cadence): between 100us (1.09x) and 300us (2.0x) idle -> discrete P-state step, not warmup.
//
// clang++ -std=c++17 -ObjC++ -fobjc-arc -O2 dvfs_probe.mm -framework Metal -framework Foundation -o dvfs_probe
// (or: cmake --build . --target dvfs_probe). Run: ./dvfs_probe

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <unistd.h>
#include <vector>

static const char* kSrc = R"METAL(
#include <metal_stdlib>
using namespace metal;

// Pure-ALU: long dependent-FMA chain in registers. iters scales the work.
// One word of global traffic (the guarded store) so the compiler can't elide it.
kernel void alu_burn(device float* out [[buffer(0)]],
                     constant uint& iters [[buffer(1)]],
                     uint gid [[thread_position_in_grid]]) {
    float a = float(gid) * 1.0000001f + 1.0f;
    float b = 0.9999999f;
    for (uint i = 0; i < iters; ++i) {
        a = fma(a, b, 1.0000001f);
        b = fma(b, a, 0.9999999f);
    }
    if (a == -123456.0f) out[gid] = a + b;  // never true; defeats DCE
}

// Pure-memory: each thread strides through a big global buffer summing words.
// Bandwidth-bound; ALU is trivial vs the loads. 'words' = elements to touch per thread.
kernel void mem_stream(device const float* in [[buffer(0)]],
                       device float* out [[buffer(1)]],
                       constant uint& n [[buffer(2)]],
                       constant uint& stride [[buffer(3)]],
                       constant uint& passes [[buffer(4)]],
                       uint gid [[thread_position_in_grid]],
                       uint gsz [[threads_per_grid]]) {
    float s = 0.0f;
    for (uint p = 0; p < passes; ++p)
        for (uint i = gid; i < n; i += gsz * stride) s += in[i];
    if (s == -123456.0f) out[gid] = s;  // defeats DCE
}
)METAL";

static double medianOf(std::vector<double>& v) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    return v[v.size()/2];
}

int main(int argc, char** argv) {
    @autoreleasepool {
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        id<MTLCommandQueue> q = [dev newCommandQueue];

        NSError* err = nil;
        id<MTLLibrary> lib = [dev newLibraryWithSource:[NSString stringWithUTF8String:kSrc]
                                               options:nil error:&err];
        if (!lib) { fprintf(stderr, "lib: %s\n", err.localizedDescription.UTF8String); return 1; }
        id<MTLComputePipelineState> aluPso =
            [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"alu_burn"] error:&err];
        id<MTLComputePipelineState> memPso =
            [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"mem_stream"] error:&err];

        // Workloads sized so each dispatch is ~3-4ms hot (decode-step scale).
        const uint aluThreads = 1u << 20;   // 1M threads
        const uint aluIters   = 1700;       // ~3.8ms hot
        const uint memBytes   = 512u << 20; // 512 MB buffer -> bandwidth-bound (> cache)
        const uint memN       = memBytes / 4;
        const uint memThreads = 1u << 20;
        const uint memStride  = 1;          // each thread strides by gsz; full-buffer sweep
        const uint memPasses  = 3;          // repeat the buffer sweep -> ~3.8ms hot

        id<MTLBuffer> outBuf = [dev newBufferWithLength:(aluThreads>memThreads?aluThreads:memThreads)*4
                                               options:MTLResourceStorageModePrivate];
        id<MTLBuffer> inBuf  = [dev newBufferWithLength:memBytes options:MTLResourceStorageModePrivate];

        auto runOnce = [&](bool alu) -> double {
            id<MTLCommandBuffer> cb = [q commandBuffer];
            id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
            if (alu) {
                [e setComputePipelineState:aluPso];
                [e setBuffer:outBuf offset:0 atIndex:0];
                [e setBytes:&aluIters length:4 atIndex:1];
                [e dispatchThreads:MTLSizeMake(aluThreads,1,1)
                    threadsPerThreadgroup:MTLSizeMake(256,1,1)];
            } else {
                [e setComputePipelineState:memPso];
                [e setBuffer:inBuf offset:0 atIndex:0];
                [e setBuffer:outBuf offset:0 atIndex:1];
                [e setBytes:&memN length:4 atIndex:2];
                [e setBytes:&memStride length:4 atIndex:3];
                [e setBytes:&memPasses length:4 atIndex:4];
                [e dispatchThreads:MTLSizeMake(memThreads,1,1)
                    threadsPerThreadgroup:MTLSizeMake(256,1,1)];
            }
            [e endEncoding];
            [cb commit];
            [cb waitUntilCompleted];
            return (cb.GPUEndTime - cb.GPUStartTime) * 1e3; // ms
        };

        const int idle_us[] = {0, 25, 100, 300, 1000, 3000};
        const int K = 70;       // dispatches per idle setting
        const int warm = 25;    // skip warmup

        printf("# M1 Max DVFS domain probe (GPU active ms per dispatch, steady-state median of %d)\n", K-warm);
        printf("# kernel       idle_us   median_ms   ratio_vs_idle0\n");

        for (int alu = 1; alu >= 0; --alu) {
            const char* name = alu ? "alu_burn " : "mem_stream";
            // hot baseline first (idle0) to normalize
            double base = 0;
            for (int s = 0; s < (int)(sizeof(idle_us)/sizeof(int)); ++s) {
                int iu = idle_us[s];
                std::vector<double> samples;
                // warm: keep GPU hot before the timed run for idle0; for idle>0 the per-dispatch
                // usleep injects the idle each step (the regime we care about).
                for (int k = 0; k < K; ++k) {
                    if (iu > 0) usleep(iu);
                    double t = runOnce(alu == 1);
                    if (k >= warm) samples.push_back(t);
                }
                double med = medianOf(samples);
                if (s == 0) base = med;
                printf("%s   %7d   %9.3f   %6.3fx\n", name, iu, med, med/base);
            }
            printf("\n");
        }
    }
    return 0;
}
