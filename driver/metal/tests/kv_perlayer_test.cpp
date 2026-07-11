// Per-layer-geometry PagedKvCache test (gemma4 gate): mixed head_dim across
// layers + cross-layer KV-sharing (zero-size shared layers). Returns non-zero
// on any failure.

#include <cstdio>
#include <exception>
#include <vector>

#include <mlx/mlx.h>

#include "mlx/kv_cache.hpp"

namespace mx = mlx::core;
using namespace pie_metal_driver;

static int failures = 0;
#define CHECK(cond, msg) do { if (!(cond)) { std::printf("FAIL: %s\n", msg); ++failures; } } while (0)

int main() {
    // 4 layers: sliding(256), full(512), sliding(256), shared(n_pages=0).
    std::vector<PagedKvLayerSpec> specs = {
        {2, 1, 256}, {2, 1, 512}, {2, 1, 256}, {0, 1, 256}};
    PagedKvCache kv(/*page_size=*/32, specs, DType::BF16);

    CHECK(kv.n_layers() == 4, "n_layers");
    CHECK(kv.head_dim(0) == 256 && kv.head_dim(1) == 512, "per-layer head_dim");
    CHECK(kv.k_pages(0).shape(3) == 256, "L0 buffer head_dim");
    CHECK(kv.k_pages(1).shape(3) == 512, "L1 buffer head_dim");
    CHECK(kv.is_shared(3) && kv.k_pages(3).shape(0) == 0, "shared layer empty");

    // Scatter 3 tokens into the full-attn (512) layer; verify the write width.
    mx::array k = mx::ones({3, 1, 512}, mx::bfloat16);
    mx::array v = mx::ones({3, 1, 512}, mx::bfloat16);
    mx::array idx = mx::array({0, 1, 2}, {3}, mx::int32);
    kv.append(1, k, v, idx);
    kv.eval();
    float s = mx::sum(mx::astype(kv.k_pages(1), mx::float32)).item<float>();
    CHECK(s == static_cast<float>(3 * 512), "L1 scatter sum");

    // append on a shared layer must throw.
    bool threw = false;
    try {
        kv.append(3, k, v, idx);
    } catch (const std::exception&) {
        threw = true;
    }
    CHECK(threw, "shared-layer append throws");

    if (failures == 0) std::printf("PASS: per-layer KV (gemma4)\n");
    return failures == 0 ? 0 : 1;
}
