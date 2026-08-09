// The live-KvCache oracle — gate-kvcache-live.
//
// The layout oracle next door (`../kv_cache/`) proves WHAT the cache
// allocates. This one proves the POINTER WIRING of the live object: which
// tensor lands in which field of `layer_view` (2,966 calls in the generated
// bodies), how the accessors resolve through `kv_source_layer`, what
// `page_buffers` hands the swap path, the envelope seed launches, and the
// clamp-and-ratio the elastic forwarding applies.
//
// Same replaced surface as the layout oracle: `DeviceTensor::allocate`
// (shared recorder, now also naming each allocation `t#K`), the elastic
// arena, and the envelope seed launcher. Everything that DECIDES is the
// shipping `kv_cache.cpp`.
//
// Pointers are reported symbolically (`t#K`, `null`) — a golden full of
// fabricated addresses would be a golden about the recorder's arithmetic.
// `scheme` and `storage_dtype` are printed as integers; both enums are
// declared without explicit values on the C++ side and with pinned
// discriminants on the Rust side, so the transcript is also a check that
// the two stayed aligned.

#include <cstdio>
#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "store/kv_cache.hpp"
#include "store/elastic.hpp"

using pie_cuda_driver::KvCache;
using pie_cuda_driver::KvCacheFormat;

namespace pie_cuda_driver {
// tensor_recorder.cpp's extras.
void reset_alloc_log();
const std::vector<std::string>& alloc_log();
std::string tensor_name(const void* ptr);
namespace kernels::layout {
extern std::vector<std::string> g_seed_log;
}
extern std::vector<std::string> g_arena_log;
}  // namespace pie_cuda_driver

namespace {

constexpr char SEP = '\x1f';

std::string g_case;
std::size_t g_alloc_drained = 0;
std::size_t g_seed_drained = 0;
std::size_t g_arena_drained = 0;

void flush() {
    const auto& allocs = pie_cuda_driver::alloc_log();
    for (std::size_t i = g_alloc_drained; i < allocs.size(); ++i) {
        std::printf("%s%calloc%c%s\n", g_case.c_str(), SEP, SEP,
                    allocs[i].c_str());
    }
    g_alloc_drained = allocs.size();
    const auto& seeds = pie_cuda_driver::kernels::layout::g_seed_log;
    for (std::size_t i = g_seed_drained; i < seeds.size(); ++i) {
        std::printf("%s%cseed%c%s\n", g_case.c_str(), SEP, SEP,
                    seeds[i].c_str());
    }
    g_seed_drained = seeds.size();
    const auto& arena = pie_cuda_driver::g_arena_log;
    for (std::size_t i = g_arena_drained; i < arena.size(); ++i) {
        std::printf("%s%carena%c%s\n", g_case.c_str(), SEP, SEP,
                    arena[i].c_str());
    }
    g_arena_drained = arena.size();
}

void begin_case(const std::string& name) {
    pie_cuda_driver::reset_alloc_log();
    pie_cuda_driver::kernels::layout::g_seed_log.clear();
    pie_cuda_driver::g_arena_log.clear();
    g_alloc_drained = 0;
    g_seed_drained = 0;
    g_arena_drained = 0;
    g_case = name;
    std::printf("%s%ccase-begin\n", g_case.c_str(), SEP);
}

std::string sym(const void* p) { return pie_cuda_driver::tensor_name(p); }

void scalars_row(KvCache& c) {
    std::printf("%s%cscalars%c%d%c%d%c%d%c%d%c%d%c%s%c%d%c%d\n", g_case.c_str(),
                SEP, SEP, c.num_layers(), SEP, c.num_pages(), SEP,
                c.page_size(), SEP, c.num_kv_heads(), SEP, c.head_dim(), SEP,
                c.format().name.c_str(), SEP, c.hnd_layout() ? 1 : 0, SEP,
                c.envelopes_enabled() ? 1 : 0);
}

void view_row(KvCache& c, int layer) {
    const auto v = c.layer_view(layer);
    std::printf(
        "%s%cview%cL%d%c%d%c%d%c%d%c%d%c%d%c%d%c%d%c%d%c%s%c%s%c%s%c%s%c%s%c%s"
        "%c%s%c%s%c%d%c%d\n",
        g_case.c_str(), SEP, SEP, layer, SEP, v.source_layer, SEP, v.num_pages,
        SEP, v.page_size, SEP, v.num_kv_heads, SEP, v.head_dim, SEP,
        static_cast<int>(v.scheme), SEP, static_cast<int>(v.storage_dtype),
        SEP, v.block_size, SEP, sym(v.k_pages).c_str(), SEP,
        sym(v.v_pages).c_str(), SEP, sym(v.k_scales).c_str(), SEP,
        sym(v.v_scales).c_str(), SEP, sym(v.k_bf16_pages).c_str(), SEP,
        sym(v.v_bf16_pages).c_str(), SEP, sym(v.k_env_min).c_str(), SEP,
        sym(v.k_env_max).c_str(), SEP, v.hnd_layout ? 1 : 0, SEP,
        v.native_bf16 ? 1 : 0);
}

void acc_row(KvCache& c, int layer) {
    std::printf("%s%cacc%cL%d%c%s%c%s%c%s%c%s%c%s%c%s%c%d%c%d\n",
                g_case.c_str(), SEP, SEP, layer, SEP, sym(c.k(layer)).c_str(),
                SEP, sym(c.v(layer)).c_str(), SEP, sym(c.k_scale(layer)).c_str(),
                SEP, sym(c.v_scale(layer)).c_str(), SEP,
                sym(c.k_for_attention(layer)).c_str(), SEP,
                sym(c.v_for_attention(layer)).c_str(), SEP,
                c.head_dim_at(layer), SEP, c.num_kv_heads_at(layer));
}

void pb_row(KvCache& c, int layer) {
    std::string row;
    for (const auto& b : c.page_buffers(layer)) {
        row += SEP + sym(b.data) + ":" + std::to_string(b.page_bytes);
    }
    std::printf("%s%cpb%cL%d%s\n", g_case.c_str(), SEP, SEP, layer,
                row.c_str());
}

void walk(KvCache& c) {
    scalars_row(c);
    for (int l = 0; l < c.num_layers(); ++l) {
        view_row(c, l);
        acc_row(c, l);
        pb_row(c, l);
    }
    flush();
}

}  // namespace

int main() {
    // a. Homogeneous native bf16, envelopes off: the baseline wiring.
    unsetenv("PIE_CUDA_KV_ENVELOPES");
    begin_case("a-hom-bf16");
    {
        auto c = KvCache::allocate(3, 4, 8, 2, 16,
                                   pie_cuda_driver::kv_cache_format_from_string("bf16"));
        flush();
        walk(c);
    }

    // b. Same stack with envelopes: the seed calls, the arena escape, and
    //    env pointers joining every view.
    setenv("PIE_CUDA_KV_ENVELOPES", "1", 1);
    begin_case("b-hom-bf16-env");
    {
        auto c = KvCache::allocate(3, 4, 8, 2, 16,
                                   pie_cuda_driver::kv_cache_format_from_string("bf16"));
        flush();
        walk(c);
    }

    // c. Per-layer stack with KV sharing and per-layer head counts
    //    (the gemma-4 shape): aliased slots resolve to their SOURCE's
    //    dims, envelopes seed only owning slots.
    begin_case("c-per-layer-env");
    {
        auto c = KvCache::allocate_per_layer(
            4, 3, 4, 4, {32, 32, 64, 64}, {0, 0, 2, 2}, {4, 4, 2, 2},
            pie_cuda_driver::kv_cache_format_from_string("bf16"));
        flush();
        walk(c);
    }

    // c2. The same shape with alias entries that DISAGREE with their
    //     source's — physically meaningless (an aliased slot allocates
    //     nothing) but exactly what separates `layer_view` reading the
    //     SOURCE's dims from a port reading the layer's own. The `acc`
    //     row prints `head_dim_at(layer)`, which does read the layer's
    //     own entry, so both spellings are pinned side by side.
    begin_case("c2-alias-dims-differ");
    {
        auto c = KvCache::allocate_per_layer(
            4, 2, 4, 4, {32, 80, 64, 96}, {0, 0, 2, 2}, {4, 8, 2, 6},
            pie_cuda_driver::kv_cache_format_from_string("bf16"));
        flush();
        walk(c);
    }

    // d. A scaled quantized format: side scales in the views and buffers,
    //    the bf16 mirror carrying attention, and the envelope REQUEST
    //    ignored because the storage tier is not what attention reads.
    begin_case("d-int8-scales-env-skipped");
    {
        auto c = KvCache::allocate(
            2, 3, 4, 2, 16,
            pie_cuda_driver::kv_cache_format_from_string("int8_per_token_head"));
        flush();
        walk(c);
    }

    // e. The FP4 block format: packed storage head_dim, block scales.
    unsetenv("PIE_CUDA_KV_ENVELOPES");
    begin_case("e-nvfp4");
    {
        auto c = KvCache::allocate(1, 2, 4, 2, 32,
                                   pie_cuda_driver::kv_cache_format_from_string("nvfp4"));
        flush();
        walk(c);
    }

    // f. Zero pages with envelopes requested: every tensor is zero-byte,
    //    the envelope pass runs and seeds NOTHING, and still flips the
    //    enabled bit — a view then reports null envelopes.
    setenv("PIE_CUDA_KV_ENVELOPES", "1", 1);
    begin_case("f-zero-pages-env");
    {
        auto c = KvCache::allocate(2, 0, 8, 2, 16,
                                   pie_cuda_driver::kv_cache_format_from_string("bf16"));
        flush();
        walk(c);
    }

    // g. The elastic forwarding: the clamp and the used/capacity ratio,
    //    plus the null-allocator and zero-page no-ops.
    unsetenv("PIE_CUDA_KV_ENVELOPES");
    begin_case("g-elastic");
    {
        auto c = KvCache::allocate(1, 10, 8, 2, 16,
                                   pie_cuda_driver::kv_cache_format_from_string("bf16"));
        flush();
        std::printf("%s%ccall%censure-before-set(3)\n", g_case.c_str(), SEP, SEP);
        c.ensure_pages(3);  // no allocator: silence
        std::printf("%s%ccommitted%c%zu\n", g_case.c_str(), SEP, SEP,
                    c.committed_bytes());
        c.set_elastic_allocator(
            std::make_shared<pie_cuda_driver::CudaArenaAllocator>());
        for (int pages : {-5, 0, 2, 99}) {
            std::printf("%s%ccall%censure(%d)\n", g_case.c_str(), SEP, SEP,
                        pages);
            c.ensure_pages(pages);
            flush();
            std::printf("%s%ccommitted%c%zu\n", g_case.c_str(), SEP, SEP,
                        c.committed_bytes());
        }
        std::printf("%s%ccall%ctrim(4)\n", g_case.c_str(), SEP, SEP);
        c.trim_pages(4);
        flush();
        std::printf("%s%ccommitted%c%zu\n", g_case.c_str(), SEP, SEP,
                    c.committed_bytes());
    }

    // h. `enable_envelopes` after the fact: an early return when they are
    //    already on, a refusal when they are not.
    setenv("PIE_CUDA_KV_ENVELOPES", "1", 1);
    begin_case("h-enable-envelopes");
    {
        auto on = KvCache::allocate(1, 2, 4, 2, 16,
                                    pie_cuda_driver::kv_cache_format_from_string("bf16"));
        flush();
        on.enable_envelopes();
        std::printf("%s%cenable-when-on%cok\n", g_case.c_str(), SEP, SEP);
        unsetenv("PIE_CUDA_KV_ENVELOPES");
        auto off = KvCache::allocate(1, 2, 4, 2, 16,
                                     pie_cuda_driver::kv_cache_format_from_string("bf16"));
        flush();
        try {
            off.enable_envelopes();
            std::printf("%s%cenable-when-off%cok\n", g_case.c_str(), SEP, SEP);
        } catch (const std::exception&) {
            std::printf("%s%cenable-when-off%cthrew\n", g_case.c_str(), SEP,
                        SEP);
        }
    }

    return 0;
}
