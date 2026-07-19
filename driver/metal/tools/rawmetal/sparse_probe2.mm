#include <cstdio>

#include "mtl4_context.hpp"

int main() {
    auto ctx = pie::metal::RawMetalContext::create(4u << 20, 4u << 20);
    if (ctx == nullptr) return 1;
    auto first = ctx->create_elastic_buffer(4u << 20);
    auto second = ctx->create_elastic_buffer(4u << 20);
    if (!first.valid() || !second.valid()) return 2;
    if (ctx->ensure_elastic_buffers_atomically({
            {first, 4u << 20},
            {second, 4u << 20},
        })) {
        return 3;
    }
    if (ctx->elastic_committed_pages() != 0) return 4;
    if (!ctx->ensure_elastic_buffer(first, 4u << 20) ||
        ctx->ensure_elastic_buffer(second, 4u << 20)) {
        return 5;
    }
    ctx->release_elastic_buffer(first);
    ctx->release_elastic_buffer(second);
    std::puts("SPARSE_PROBE2_OK");
    return 0;
}
