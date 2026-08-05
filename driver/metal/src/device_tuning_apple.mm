// The Apple half of the device query: Metal for the family, IOKit for the core
// count. Split from `device_tuning.cpp` because this is Objective-C and the
// table it feeds is not.

#include "device_tuning.hpp"

#include <IOKit/IOKitLib.h>
#import <Metal/Metal.h>

namespace pie::metal {

namespace {

/// IOKit publishes `gpu-core-count` on the AGXAccelerator service; `MTLDevice`
/// does not expose it at all. Absent on a machine whose driver does not publish
/// it, which is why the caller must tolerate 0 rather than branch on a count it
/// assumes it has.
int query_gpu_core_count() {
    io_iterator_t it = IO_OBJECT_NULL;
    if (IOServiceGetMatchingServices(kIOMainPortDefault,
                                     IOServiceMatching("AGXAccelerator"),
                                     &it) != KERN_SUCCESS) {
        return 0;
    }
    int cores = 0;
    while (io_object_t svc = IOIteratorNext(it)) {
        CFTypeRef v = IORegistryEntryCreateCFProperty(
            svc, CFSTR("gpu-core-count"), kCFAllocatorDefault, 0);
        if (v != nullptr) {
            if (CFGetTypeID(v) == CFNumberGetTypeID()) {
                int n = 0;
                if (CFNumberGetValue(static_cast<CFNumberRef>(v), kCFNumberIntType, &n)) {
                    cores = n;
                }
            }
            CFRelease(v);
        }
        IOObjectRelease(svc);
        if (cores != 0) break;
    }
    IOObjectRelease(it);
    return cores;
}

/// NEWEST-FIRST, and that is the whole of the correctness argument: the Apple
/// families are cumulative, so an M4 answers yes to Apple7 too. Probing in
/// ascending order would report every Apple Silicon GPU ever made as an
/// Apple7 and silently give all of them the M1 constants -- a bug that looks
/// exactly like the tuning layer not existing.
int query_apple_family() {
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    if (dev == nil) return 0;
    const struct {
        MTLGPUFamily family;
        int n;
    } families[] = {
        {MTLGPUFamilyApple9, 9}, {MTLGPUFamilyApple8, 8}, {MTLGPUFamilyApple7, 7},
        {MTLGPUFamilyApple6, 6}, {MTLGPUFamilyApple5, 5},
    };
    for (const auto& f : families) {
        if ([dev supportsFamily:f.family]) return f.n;
    }
    return 0;
}

}  // namespace

DeviceInfo query_device_info() {
    DeviceInfo i;
    i.apple_family = query_apple_family();
    i.gpu_core_count = query_gpu_core_count();
    return i;
}

}  // namespace pie::metal
