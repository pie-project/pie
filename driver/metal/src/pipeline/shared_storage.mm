#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "pipeline/shared_storage.hpp"

namespace pie::metal::pipeline {

SharedStorage make_platform_shared_storage(std::size_t size) {
    SharedStorage storage;
    if (size == 0) return storage;

    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (device == nil) return storage;
    id<MTLBuffer> buffer =
        [device newBufferWithLength:size options:MTLResourceStorageModeShared];
    if (buffer == nil) return storage;
    std::memset(buffer.contents, 0, size);

    void* retained = (__bridge_retained void*)buffer;
    storage.owner = std::shared_ptr<void>(
        retained,
        [](void* ptr) {
            if (ptr == nullptr) return;
            id released = (__bridge_transfer id)ptr;
            (void)released;
        });
    storage.contents = static_cast<std::uint8_t*>(buffer.contents);
    storage.native_buffer = (__bridge void*)buffer;
    storage.gpu_address = buffer.gpuAddress;
    storage.size = size;
    return storage;
}

}  // namespace pie::metal::pipeline
