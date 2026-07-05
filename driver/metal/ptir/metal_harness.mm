// MetalHarness implementation (Objective-C++). See metal_harness.hpp.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <cstring>

#include "metal_harness.hpp"

namespace ptir_metal {

struct MetalHarness::Impl {
    id<MTLDevice> device = nil;
    id<MTLCommandQueue> queue = nil;
    id<MTLLibrary> library = nil;
};

MetalHarness::MetalHarness() : impl_(new Impl()) {
    @autoreleasepool {
        impl_->device = MTLCreateSystemDefaultDevice();
        if (impl_->device) {
            impl_->queue = [impl_->device newCommandQueue];
        } else {
            error_ = "no default Metal device";
        }
    }
}

MetalHarness::~MetalHarness() {
    @autoreleasepool {
        impl_->library = nil;
        impl_->queue = nil;
        impl_->device = nil;
    }
    delete impl_;
}

bool MetalHarness::ok() const { return impl_->device != nil && impl_->queue != nil; }

std::string MetalHarness::device_name() const {
    if (!impl_->device) return "";
    return std::string([[impl_->device name] UTF8String]);
}

bool MetalHarness::load_library(const std::string& path) {
    @autoreleasepool {
        NSString* nspath = [NSString stringWithUTF8String:path.c_str()];
        NSError* err = nil;
        NSString* src = [NSString stringWithContentsOfFile:nspath
                                                  encoding:NSUTF8StringEncoding
                                                     error:&err];
        if (!src) {
            error_ = std::string("cannot read kernel source: ") +
                     (err ? [[err description] UTF8String] : path.c_str());
            return false;
        }
        MTLCompileOptions* opts = [[MTLCompileOptions alloc] init];
        // IEEE-exact math is mandatory for bit-exact cross-backend parity: fast
        // math would use reciprocal-approx division / reassociated adds and
        // diverge from the Rust reference. Disable it (mathMode on macOS 15+,
        // the deprecated fastMathEnabled otherwise).
        if ([opts respondsToSelector:@selector(setMathMode:)]) {
            opts.mathMode = MTLMathModeSafe;
        } else {
            opts.fastMathEnabled = NO;
        }
        impl_->library = [impl_->device newLibraryWithSource:src options:opts error:&err];
        if (!impl_->library) {
            error_ = std::string("kernel compile failed: ") +
                     (err ? [[err description] UTF8String] : "unknown");
            return false;
        }
        return true;
    }
}

bool MetalHarness::run(const std::string& fn_name, std::vector<Arg>& args,
                       std::uint32_t grid_threads) {
    @autoreleasepool {
        if (!impl_->library) {
            error_ = "no library loaded";
            return false;
        }
        NSError* err = nil;
        NSString* nsfn = [NSString stringWithUTF8String:fn_name.c_str()];
        id<MTLFunction> fn = [impl_->library newFunctionWithName:nsfn];
        if (!fn) {
            error_ = std::string("no such kernel function: ") + fn_name;
            return false;
        }
        id<MTLComputePipelineState> pso =
            [impl_->device newComputePipelineStateWithFunction:fn error:&err];
        if (!pso) {
            error_ = std::string("pipeline creation failed: ") +
                     (err ? [[err description] UTF8String] : "unknown");
            return false;
        }

        std::vector<id<MTLBuffer>> bufs(args.size());
        for (std::size_t i = 0; i < args.size(); ++i) {
            std::size_t n = args[i].bytes ? args[i].bytes : 1;
            bufs[i] = [impl_->device newBufferWithLength:n
                                                 options:MTLResourceStorageModeShared];
            if (!args[i].is_output && args[i].data) {
                std::memcpy([bufs[i] contents], args[i].data, args[i].bytes);
            } else {
                std::memset([bufs[i] contents], 0, n);
            }
        }

        id<MTLCommandBuffer> cb = [impl_->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:pso];
        for (std::size_t i = 0; i < bufs.size(); ++i) {
            [enc setBuffer:bufs[i] offset:0 atIndex:i];
        }
        NSUInteger tg = pso.maxTotalThreadsPerThreadgroup;
        if (tg > grid_threads) tg = grid_threads;
        if (tg == 0) tg = 1;
        [enc dispatchThreads:MTLSizeMake(grid_threads, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        if (cb.status != MTLCommandBufferStatusCompleted) {
            error_ = std::string("command buffer failed: ") +
                     (cb.error ? [[cb.error description] UTF8String] : "unknown");
            return false;
        }

        for (std::size_t i = 0; i < args.size(); ++i) {
            if (args[i].is_output && args[i].data) {
                std::memcpy(args[i].data, [bufs[i] contents], args[i].bytes);
            }
        }
        return true;
    }
}

long first_bit_diff(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size()) return 0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        std::uint32_t ba, bb;
        std::memcpy(&ba, &a[i], 4);
        std::memcpy(&bb, &b[i], 4);
        if (ba != bb) return static_cast<long>(i);
    }
    return -1;
}

}  // namespace ptir_metal
