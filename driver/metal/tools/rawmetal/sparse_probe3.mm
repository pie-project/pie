#include <cstdio>
#include <cstring>

#include "mtl4_context.hpp"

int main() {
    constexpr std::size_t kCopyBytes = 1u << 20;
    auto ctx = pie::metal::RawMetalContext::create(4u << 20, 16u << 20);
    if (ctx == nullptr) return 1;
    auto source = ctx->create_elastic_buffer(8u << 20);
    auto destination = ctx->create_elastic_buffer(8u << 20);
    auto input = ctx->create_standalone_buffer(kCopyBytes);
    auto output = ctx->create_standalone_buffer(kCopyBytes);
    if (!source.valid() || !destination.valid() ||
        !input.valid() || !output.valid()) {
        return 2;
    }
    std::memset(input.contents(), 0xa5, kCopyBytes);
    std::memset(output.contents(), 0, kCopyBytes);
    if (!ctx->ensure_elastic_buffers_atomically({
            {source, 4u << 20},
            {destination, 4u << 20},
        }) ||
        !ctx->copy_buffer_range(source, 0, input, 0, kCopyBytes) ||
        !ctx->trim_elastic_buffer(source, 2u << 20) ||
        !ctx->ensure_elastic_buffer(source, 4u << 20) ||
        !ctx->copy_buffer_range(
            destination,
            0,
            source,
            0,
            kCopyBytes) ||
        !ctx->copy_buffer_range(
            output,
            0,
            destination,
            0,
            kCopyBytes) ||
        std::memcmp(input.contents(), output.contents(), kCopyBytes) != 0) {
        return 3;
    }
    ctx->release_elastic_buffer(source);
    ctx->release_elastic_buffer(destination);
    ctx->release_standalone_buffer(input);
    ctx->release_standalone_buffer(output);
    std::puts("SPARSE_PROBE3_OK");
    return 0;
}
