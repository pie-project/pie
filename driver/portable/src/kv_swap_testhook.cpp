// Test-only entries exercising KvCachePaged::page_bytes() for a NON-UNIFORM
// (Gemma 4 alternating-attention) KV cache, from a pie-server #[cfg(test)]
// Rust test. The KV-swap / ToT-fork copy loop in service/inproc_service.cpp
// sizes and offsets every page-copy with `kv.page_bytes(layer)`; a single
// uniform page-bytes value (layer 0) over-reads the smaller full-attention
// layers and trips ggml's `offset + size <= nbytes` abort under multi-request
// host-swap or deep-tree fork. These hooks build a CPU-backend non-uniform KV
// and report each layer's page bytes + tensor nbytes so the Rust test can
// prove the per-layer value (a) differs across layers and (b) keeps every page
// in bounds where the uniform value would not.
//
// Lives in the driver lib and is exercised from the `pie` bin because CI runs
// `cargo test -p pie-server --bin pie`; driver/portable ctest never runs in CI.
// The symbols are GC'd out of release builds — no Rust code references them.

#include <cstdint>
#include <memory>
#include <vector>

#include <ggml.h>
#include <ggml-cpu.h>

#include "kv_cache.hpp"

using pie_portable_driver::KvCachePaged;

namespace {

// Gemma-4-shaped two-layer cache: layer 0 = sliding (8 kv-heads × 256
// head_dim), layer 1 = full attention (1 kv-head × 512 head_dim). F16,
// page_size 16, 4 pages. page_bytes: L0 = 8*256*16*2 = 65536, L1 =
// 1*512*16*2 = 16384 — deliberately unequal.
std::unique_ptr<KvCachePaged> make_non_uniform_kv() {
    ggml_backend_t backend = ggml_backend_cpu_init();
    if (!backend) return nullptr;
    // The cache owns no backend handle to free, but the CPU backend is a
    // process-wide singleton-ish handle; leaking it in a test hook is benign.
    std::vector<std::int32_t> kv_heads = {8, 1};
    std::vector<std::int32_t> head_dim = {256, 512};
    return std::make_unique<KvCachePaged>(
        backend, kv_heads, head_dim,
        /*total_pages=*/4, /*page_size=*/16, GGML_TYPE_F16);
}

}  // namespace

extern "C" {

// Page bytes of layer `layer` (== the value the copy loop uses to size/offset
// each per-layer page copy). Returns -1 on construction failure.
std::int64_t pie_portable_test_kv_page_bytes(std::int32_t layer) {
    auto kv = make_non_uniform_kv();
    if (!kv) return -1;
    return static_cast<std::int64_t>(kv->page_bytes(layer));
}

// Total bytes of layer `layer`'s K tensor (the bound a page copy must respect).
std::int64_t pie_portable_test_kv_layer_nbytes(std::int32_t layer) {
    auto kv = make_non_uniform_kv();
    if (!kv) return -1;
    return static_cast<std::int64_t>(ggml_nbytes(kv->k(layer)));
}

std::int32_t pie_portable_test_kv_total_pages() {
    auto kv = make_non_uniform_kv();
    if (!kv) return -1;
    return kv->total_pages();
}

}  // extern "C"
