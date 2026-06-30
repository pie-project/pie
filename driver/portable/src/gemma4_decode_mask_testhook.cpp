// Test-only entry called from pie-server #[cfg(test)] Rust tests; no release
// code references it, so the linker can garbage-collect it from shipped binaries.

#include <cstdint>

#include <ggml.h>

#include "graph_gemma4.hpp"
#include "plan.hpp"  // MASK_PAD

extern "C" int pie_portable_test_gemma4_manual_decode_mask_view_status() {
    ggml_init_params ip{
        /*.mem_size   =*/ ggml_tensor_overhead() * 8,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context* ctx = ggml_init(ip);
    if (ctx == nullptr) return 10;

    constexpr std::int32_t max_n_kv = 64;
    constexpr std::int32_t n_req = 3;
    ggml_tensor* packed = ggml_new_tensor_4d(
        ctx, GGML_TYPE_F16, max_n_kv, pie_portable_driver::MASK_PAD, 1, n_req);
    ggml_tensor* viewed = pie_portable_driver::gemma4_manual_decode_mask_view(
        ctx, packed, max_n_kv, n_req);

    int status = 0;
    if (viewed == nullptr) {
        status = 11;
    } else if (viewed->ne[0] != max_n_kv ||
               viewed->ne[1] != 1 ||
               viewed->ne[2] != 1 ||
               viewed->ne[3] != n_req) {
        status = 12;
    } else if (viewed->nb[1] != packed->nb[1] ||
               viewed->nb[2] != packed->nb[2] ||
               viewed->nb[3] != packed->nb[3]) {
        status = 13;
    }
    ggml_free(ctx);
    return status;
}
