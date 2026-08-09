// Drives the real `KvCache` over a grid of layer stacks and reports, for each,
// the exact sequence of tensor allocations it makes.
//
// The transcript is the allocation LOG, not the resulting object: what matters
// about `allocate_per_layer` is which tensors it creates, with what extents,
// in what order, and the object exposes none of that. A cache that allocated
// the right total bytes in the wrong per-layer split would look identical from
// the outside and be wrong in exactly the way that costs a boot.

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include "store/kv_cache.hpp"
#include "store/kv_cache_format.hpp"
#include "model/config.hpp"

namespace pie_cuda_driver {
extern std::vector<std::string> g_alloc_log;
void reset_alloc_log();
namespace kernels::layout {
extern std::vector<std::string> g_seed_log;
}
}  // namespace pie_cuda_driver

using namespace pie_cuda_driver;

namespace {

constexpr char kSep = '\x1f';

struct FormatCase {
    const char* label;
    KvCacheScheme scheme;
    KvCacheScaleLayout scale;
    DType storage;
    int block;
};

// Every format the shipping `KvCacheFormat` table can produce, by the three
// fields `kv_cache.cpp` actually branches on. Built here rather than parsed
// from a name so the grid covers combinations the name table does not spell --
// a scale layout on a native dtype, a block size of zero on a blocked layout
// -- because `allocate` branches on the fields, not on the name.
const std::vector<FormatCase> kFormats = {
    {"bf16", KvCacheScheme::Native, KvCacheScaleLayout::None, DType::BF16, 0},
    {"fp16", KvCacheScheme::Native, KvCacheScaleLayout::None, DType::FP16, 0},
    {"fp8pt", KvCacheScheme::Fp8PerTensor, KvCacheScaleLayout::None,
     DType::FP8_E4M3, 0},
    {"fp8pth", KvCacheScheme::Fp8PerTokenHead, KvCacheScaleLayout::PerTokenHead,
     DType::FP8_E4M3, 0},
    {"int8pth", KvCacheScheme::Int8PerTokenHead,
     KvCacheScaleLayout::PerTokenHead, DType::INT8, 0},
    {"fp4b16", KvCacheScheme::Fp4Block, KvCacheScaleLayout::PerTokenHeadBlock,
     DType::FP8_E4M3, 16},
    {"fp4b32", KvCacheScheme::Fp4Block, KvCacheScaleLayout::PerTokenHeadBlock,
     DType::FP8_E4M3, 32},
    // block_size 0 on a blocked layout: `allocate` substitutes 16 at the call
    // site rather than at parse time, so this is a live path.
    {"fp4b0", KvCacheScheme::Fp4Block, KvCacheScaleLayout::PerTokenHeadBlock,
     DType::FP8_E4M3, 0},
    // A scale tier on a native BF16 storage dtype. `is_native_bf16()` is true
    // here, so this allocates scales AND skips the mirror -- a combination the
    // name table never produces and nothing else in the sweep reaches.
    {"bf16scaled", KvCacheScheme::Native, KvCacheScaleLayout::PerTokenHead,
     DType::BF16, 0},
};

KvCacheFormat make_format(const FormatCase& f) {
    KvCacheFormat out;
    out.name = f.label;
    out.scheme = f.scheme;
    out.scale_layout = f.scale;
    out.storage_dtype = f.storage;
    out.block_size = f.block;
    return out;
}

std::string join(const std::vector<std::string>& v) {
    std::string out;
    for (std::size_t i = 0; i < v.size(); ++i) {
        if (i != 0) out += kSep;
        out += v[i];
    }
    return out;
}

// Reads back what the constructed cache reports, so the transcript covers the
// ACCESSORS as well as the allocation: `resolve_`, `head_dim_at`,
// `num_kv_heads_at`, and which of the six tensors a layer view points at.
std::string readback(KvCache& c, int num_layers) {
    std::vector<std::string> rows;
    for (int i = 0; i < num_layers; ++i) {
        const KvCacheLayerView v = c.layer_view(i);
        std::string r = "L" + std::to_string(i);
        r += " src=" + std::to_string(v.source_layer);
        r += " hd=" + std::to_string(v.head_dim);
        r += " kvh=" + std::to_string(v.num_kv_heads);
        r += " pages=" + std::to_string(v.num_pages);
        r += " psz=" + std::to_string(v.page_size);
        r += " bs=" + std::to_string(v.block_size);
        r += " hnd=" + std::to_string(static_cast<int>(v.hnd_layout));
        r += " nat=" + std::to_string(static_cast<int>(v.native_bf16));
        // Pointer IDENTITY, not value: whether the attention input aliases the
        // storage tier is the question, and the addresses themselves are an
        // artefact of the recorder.
        r += " kattn=";
        r += (v.k_bf16_pages == nullptr ? "null"
              : v.k_bf16_pages == v.k_pages ? "same" : "mirror");
        r += " vattn=";
        r += (v.v_bf16_pages == nullptr ? "null"
              : v.v_bf16_pages == v.v_pages ? "same" : "mirror");
        r += " kscale=" + std::string(v.k_scales ? "p" : "null");
        r += " vscale=" + std::string(v.v_scales ? "p" : "null");
        r += " env=" + std::string(v.k_env_min ? "p" : "null");
        r += " shared=";
        r += (v.k_pages == c.layer_view(v.source_layer).k_pages ? "1" : "0");
        const auto bufs = c.page_buffers(i);
        r += " bufs=" + std::to_string(bufs.size());
        for (const auto& b : bufs) r += "/" + std::to_string(b.page_bytes);
        rows.push_back(r);
    }
    return join(rows);
}

void emit(const std::string& id, const std::string& body) {
    std::cout << id << '|' << body << '\n';
}

void run_homogeneous(const std::string& id, int layers, int pages, int page_size,
                     int kv_heads, int head_dim, const KvCacheFormat& fmt) {
    reset_alloc_log();
    kernels::layout::g_seed_log.clear();
    try {
        KvCache c = KvCache::allocate(layers, pages, page_size, kv_heads,
                                      head_dim, fmt);
        emit(id, "OK|" + join(g_alloc_log) + "|" +
                     join(kernels::layout::g_seed_log) + "|" +
                     readback(c, layers));
    } catch (const std::exception& e) {
        emit(id, std::string("FAILED|") + e.what() + "|" + join(g_alloc_log));
    }
}

void run_per_layer(const std::string& id, int layers, int pages, int page_size,
                   int kv_heads, const std::vector<int>& hd,
                   const std::vector<int>& src, const std::vector<int>& kvh,
                   const KvCacheFormat& fmt) {
    reset_alloc_log();
    kernels::layout::g_seed_log.clear();
    try {
        KvCache c = KvCache::allocate_per_layer(layers, pages, page_size,
                                                kv_heads, hd, src, kvh, fmt);
        emit(id, "OK|" + join(g_alloc_log) + "|" +
                     join(kernels::layout::g_seed_log) + "|" +
                     readback(c, layers));
    } catch (const std::exception& e) {
        emit(id, std::string("FAILED|") + e.what() + "|" + join(g_alloc_log));
    }
}

}  // namespace

int main() {
    std::cout << "# kv_cache oracle v1\n";

    // 1. Homogeneous stacks across every format and a range of shapes.
    for (const auto& f : kFormats) {
        const KvCacheFormat fmt = make_format(f);
        for (int layers : {1, 2, 8}) {
            for (int pages : {0, 1, 64, 4096}) {
                for (int page_size : {1, 16, 32}) {
                    for (int kv_heads : {1, 4, 8}) {
                        for (int head_dim : {64, 128, 576}) {
                            run_homogeneous(
                                std::string("homo/") + f.label + "/" +
                                    std::to_string(layers) + "/" +
                                    std::to_string(pages) + "/" +
                                    std::to_string(page_size) + "/" +
                                    std::to_string(kv_heads) + "/" +
                                    std::to_string(head_dim),
                                layers, pages, page_size, kv_heads, head_dim,
                                fmt);
                        }
                    }
                }
            }
        }
    }

    // 2. Odd head dimensions, which only matter for a blocked scale tier:
    // `blocks = ceil(head_dim / block_size)` is the one place the shape is not
    // a straight product of the arguments.
    for (const auto& f : kFormats) {
        const KvCacheFormat fmt = make_format(f);
        for (int head_dim : {1, 2, 15, 17, 31, 33, 63, 65, 127, 129, 191}) {
            run_homogeneous(std::string("odd/") + f.label + "/" +
                                std::to_string(head_dim),
                            2, 32, 16, 4, head_dim, fmt);
        }
    }

    // 3. The `allocate(dtype)` overload, which synthesises the format itself
    // and names BF16 "bf16" rather than going through `dtype_name`.
    for (DType d : {DType::BF16, DType::FP16, DType::FP8_E4M3, DType::INT8,
                    DType::FP32}) {
        reset_alloc_log();
        kernels::layout::g_seed_log.clear();
        try {
            KvCache c = KvCache::allocate(2, 64, 16, 4, 128, d);
            emit(std::string("dtype/") + std::to_string(static_cast<int>(d)),
                 "OK|" + join(g_alloc_log) + "|" + c.format().name + "|" +
                     readback(c, 2));
        } catch (const std::exception& e) {
            emit(std::string("dtype/") + std::to_string(static_cast<int>(d)),
                 std::string("FAILED|") + e.what());
        }
    }

    // 4. Per-layer stacks. The sharing patterns are the ones the shipping
    // models produce -- gemma's sliding/full alternation, a single shared
    // source, every layer its own -- plus the degenerate ones.
    const std::vector<std::pair<const char*, std::vector<int>>> kShares = {
        {"none", {}},
        {"self", {0, 1, 2, 3, 4, 5}},
        {"all-to-0", {0, 0, 0, 0, 0, 0}},
        {"pairs", {0, 0, 2, 2, 4, 4}},
        {"gemma-5in6", {0, 0, 0, 0, 0, 5}},
        {"last-source", {5, 5, 5, 5, 5, 5}},
        {"forward-ref", {1, 1, 3, 3, 5, 5}},
    };
    for (const auto& f : kFormats) {
        const KvCacheFormat fmt = make_format(f);
        for (const auto& [share_label, src] : kShares) {
            for (int variant = 0; variant < 4; ++variant) {
                std::vector<int> hd;
                std::vector<int> kvh;
                switch (variant) {
                    case 0:
                        break;  // both empty: the scalar path
                    case 1:
                        hd = {64, 128, 64, 128, 64, 128};
                        break;
                    case 2:
                        kvh = {1, 2, 4, 8, 4, 2};
                        break;
                    default:
                        hd = {576, 64, 128, 256, 128, 64};
                        kvh = {1, 8, 4, 2, 4, 8};
                        break;
                }
                run_per_layer(std::string("perlayer/") + f.label + "/" +
                                  share_label + "/v" + std::to_string(variant),
                              6, 128, 16, 4, hd, src, kvh, fmt);
            }
        }
    }

    // 5. The three length validations, each on its own, plus a length that is
    // right for one vector and wrong for another.
    {
        const KvCacheFormat fmt = make_format(kFormats[0]);
        run_per_layer("bad/hd-short", 4, 32, 16, 4, {64, 64}, {}, {}, fmt);
        run_per_layer("bad/hd-long", 4, 32, 16, 4, {64, 64, 64, 64, 64}, {}, {},
                      fmt);
        run_per_layer("bad/src-short", 4, 32, 16, 4, {}, {0, 0}, {}, fmt);
        run_per_layer("bad/kvh-short", 4, 32, 16, 4, {}, {}, {1, 2}, fmt);
        run_per_layer("bad/hd-ok-kvh-short", 4, 32, 16, 4, {64, 64, 64, 64}, {},
                      {1, 2}, fmt);
        // Two and three bad vectors at once. Any one of these alone only
        // pins WHICH check exists; only a collision pins the ORDER they run
        // in, and the order is what the operator reads first.
        run_per_layer("bad/hd-and-src-short", 4, 32, 16, 4, {64, 64}, {0, 0}, {},
                      fmt);
        run_per_layer("bad/src-and-kvh-short", 4, 32, 16, 4, {}, {0, 0}, {1, 2},
                      fmt);
        run_per_layer("bad/all-three-short", 4, 32, 16, 4, {64}, {0}, {1}, fmt);
        run_per_layer("bad/zero-layers-with-vectors", 0, 32, 16, 4, {64}, {}, {},
                      fmt);
        run_per_layer("bad/all-empty-zero-layers", 0, 32, 16, 4, {}, {}, {},
                      fmt);
    }

    // 6. `head_dim_` is `per_layer_head_dim[0]` -- or ZERO when the vector is
    // empty, which then feeds every `head_dim_at` call. A per-layer stack with
    // no head-dim vector therefore allocates zero-width tensors, and that is
    // the shipping behaviour rather than a fallback to the scalar.
    {
        const KvCacheFormat fmt = make_format(kFormats[0]);
        run_per_layer("scalar-hd/empty", 3, 32, 16, 4, {}, {}, {}, fmt);
        run_per_layer("scalar-hd/first-is-576", 3, 32, 16, 4, {576, 64, 64}, {},
                      {}, fmt);
    }

    // 7. The free functions at the bottom of the file, which the memory
    // planner calls and which must agree with what `allocate` reserves.
    for (const auto& f : kFormats) {
        const KvCacheFormat fmt = make_format(f);
        for (int page_size : {1, 16, 32}) {
            for (int kv_heads : {1, 4, 8}) {
                for (int head_dim : {64, 128, 576}) {
                    std::cout << "bytes/" << f.label << "/" << page_size << "/"
                              << kv_heads << "/" << head_dim << "|"
                              << kv_cache_device_bytes_per_page(
                                     fmt, page_size, kv_heads, head_dim)
                              << '\n';
                }
            }
        }
        HfConfig cfg;
        cfg.num_hidden_layers = 6;
        cfg.num_key_value_heads = 8;
        cfg.head_dim_kernel = 128;
        for (int tp : {0, 1, 2, 3, 8, 16}) {
            std::cout << "homobytes/" << f.label << "/" << tp << "|"
                      << kv_page_bytes_homogeneous(cfg, tp, fmt) << '\n';
            for (const auto& [share_label, src] : kShares) {
                std::cout << "layerbytes/" << f.label << "/" << tp << "/"
                          << share_label << "|"
                          << kv_page_bytes_per_layer(cfg, {}, {}, src, tp, fmt)
                          << '/'
                          << kv_page_bytes_per_layer(
                                 cfg, {64, 128, 64, 128, 64, 128}, {}, src, tp,
                                 fmt)
                          << '/'
                          << kv_page_bytes_per_layer(cfg, {},
                                                     {1, 2, 4, 8, 4, 2}, src,
                                                     tp, fmt)
                          << '\n';
            }
        }
    }

    // 8. Envelopes. `envelopes_requested()` reads PIE_CUDA_KV_ENVELOPES on
    // every call, so switching it on here exercises the whole tier: the
    // allocator-binding swap, the per-slot guard, the seed launch, and the
    // native-BF16/NHD restriction that silently skips the tier for every other
    // format.
    setenv("PIE_CUDA_KV_ENVELOPES", "1", 1);
    for (const auto& f : kFormats) {
        const KvCacheFormat fmt = make_format(f);
        for (int layers : {1, 3}) {
            for (int pages : {0, 64}) {
                for (int kv_heads : {1, 4}) {
                    for (int head_dim : {64, 576}) {
                        run_homogeneous(std::string("env/") + f.label + "/" +
                                            std::to_string(layers) + "/" +
                                            std::to_string(pages) + "/" +
                                            std::to_string(kv_heads) + "/" +
                                            std::to_string(head_dim),
                                        layers, pages, 16, kv_heads, head_dim,
                                        fmt);
                    }
                }
            }
        }
        for (const auto& [share_label, src] : kShares) {
            run_per_layer(std::string("envshare/") + f.label + "/" + share_label,
                          6, 64, 16, 4, {576, 64, 128, 256, 128, 64}, src,
                          {1, 8, 4, 2, 4, 8}, fmt);
            run_per_layer(std::string("envshare/") + f.label + "/" + share_label +
                              "/scalar",
                          6, 64, 16, 4, {}, src, {}, fmt);
        }
    }
    // Every accepted spelling of the switch, and the ones that must NOT enable
    // it -- "on"/"true"/"1" only, so "yes", "TRUE" and "0" leave it off.
    for (const char* v : {"1", "true", "on", "0", "yes", "TRUE", "On", "",
                          "false", "2"}) {
        setenv("PIE_CUDA_KV_ENVELOPES", v, 1);
        std::cout << "envswitch/" << (v[0] == '\0' ? "<empty>" : v) << '|'
                  << (KvCache::envelopes_requested() ? 1 : 0) << '\n';
    }
    unsetenv("PIE_CUDA_KV_ENVELOPES");
    std::cout << "envswitch/<unset>|"
              << (KvCache::envelopes_requested() ? 1 : 0) << '\n';

    // 9. `enable_envelopes()` always throws: the tier is allocated with the
    // pool or not at all, because the page count was chosen to leave room for
    // it. Reaching it means a program asked for envelopes on a cache sized
    // without them.
    {
        KvCache c = KvCache::allocate(1, 8, 16, 1, 64, make_format(kFormats[0]));
        try {
            c.enable_envelopes();
            std::cout << "enable-late|RETURNED\n";
        } catch (const std::exception& e) {
            std::cout << "enable-late|THREW|" << e.what() << '\n';
        }
    }

    return 0;
}
