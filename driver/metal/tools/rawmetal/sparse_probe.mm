#include <cstdio>

#include "mtl4_context.hpp"

int main() {
    auto ctx = pie::metal::RawMetalContext::create(4u << 20, 64u << 20);
    if (ctx == nullptr) return 1;
    auto buffer = ctx->create_elastic_buffer(64u << 20);
    if (!buffer.valid() || !buffer.elastic) return 2;
    const auto address = buffer.gpu_address;
    if (!ctx->ensure_elastic_buffer(buffer, 32u << 20) ||
        buffer.gpu_address != address) {
        return 3;
    }
    if (!ctx->trim_elastic_buffer(buffer, 2u << 20) ||
        ctx->elastic_committed_pages() != 1) {
        return 4;
    }
    ctx->release_elastic_buffer(buffer);
    if (ctx->elastic_committed_pages() != 0 ||
        ctx->pending_elastic_release_count() != 0) {
        return 5;
    }
    std::puts("SPARSE_PROBE_OK");
    return 0;
}
