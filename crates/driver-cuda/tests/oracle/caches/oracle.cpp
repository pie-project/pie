// Drives the real `MlaCache`, `DsV4CompressCache` and `SwapPool` allocation
// paths over a grid and reports, for each, exactly what memory each one asks
// for and in what order.
//
// The transcript is the ALLOCATION LOG, not the resulting object. All three
// classes expose almost nothing: `MlaCache` hands back a view of pointers,
// `DsV4CompressCache` has no accessors for its per-layer widths at all, and
// `SwapPool` reports one aggregate `bytes_per_page()`. A cache that allocated
// the right total in the wrong per-layer split, or in the wrong order, would
// be indistinguishable from outside and wrong in exactly the way that costs a
// boot.
//
// The copy half of `swap_pool.cpp` is proved separately by
// tests/oracle/store; what this covers is the two constructors, which that
// oracle does not reach.

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "cuda_runtime.h"
#include "store/dsv4_compress_cache.hpp"
#include "store/kv_cache.hpp"
#include "store/mla_cache.hpp"
#include "store/swap_pool.hpp"
#include "model/config.hpp"

namespace pie_cuda_driver {
extern std::vector<std::string> g_alloc_log;
void reset_alloc_log();
}  // namespace pie_cuda_driver

using namespace pie_cuda_driver;

namespace {

constexpr char kSep = '\x1f';

std::string join(const std::vector<std::string>& v) {
    std::string out;
    for (std::size_t i = 0; i < v.size(); ++i) {
        if (i != 0) out += ',';
        out += v[i];
    }
    return out;
}

void begin_case() {
    oracle_cuda::reset_case();
    reset_alloc_log();
}

void emit(const std::string& id, const std::vector<std::string>& fields) {
    std::string row = id;
    for (const auto& f : fields) {
        row += kSep;
        row += f;
    }
    std::cout << row << '\n';
}

// ---------------------------------------------------------------------------
// 1. MlaCache::allocate
// ---------------------------------------------------------------------------
//
// The dimension grid deliberately includes zero and negative extents on every
// axis, because the validation is the only branch in the function and it is
// an all-or-nothing `||` chain: a single dropped clause lets a zero through to
// a zero-byte tensor whose pointer is null, and the MLA kernel dereferences it.

struct MlaCase {
    const char* label;
    int layers, pages, page_size, lora, rope;
    DType dtype;
};

const std::vector<MlaCase> kMlaCases = {
    {"tiny", 1, 1, 1, 1, 1, DType::BF16},
    {"ds3", 61, 512, 16, 512, 64, DType::BF16},
    {"ds3-fp16", 61, 512, 16, 512, 64, DType::FP16},
    {"kimi", 27, 128, 64, 576, 64, DType::BF16},
    {"lopsided", 3, 7, 5, 1, 4096, DType::BF16},
    {"one-layer", 1, 4096, 1, 512, 64, DType::FP16},
    // Every axis at zero, then at -1. Twelve rejections; each names the axis
    // so a reordered check is visible rather than absorbed.
    {"bad/layers0", 0, 8, 8, 8, 8, DType::BF16},
    {"bad/pages0", 8, 0, 8, 8, 8, DType::BF16},
    {"bad/psize0", 8, 8, 0, 8, 8, DType::BF16},
    {"bad/lora0", 8, 8, 8, 0, 8, DType::BF16},
    {"bad/rope0", 8, 8, 8, 8, 0, DType::BF16},
    {"bad/layers-1", -1, 8, 8, 8, 8, DType::BF16},
    {"bad/pages-1", 8, -1, 8, 8, 8, DType::BF16},
    {"bad/psize-1", 8, 8, -1, 8, 8, DType::BF16},
    {"bad/lora-1", 8, 8, 8, -1, 8, DType::BF16},
    {"bad/rope-1", 8, 8, 8, 8, -1, DType::BF16},
    // Two bad axes at once: the dimension check is a single `||`, so this
    // must produce the same message as either one alone.
    {"bad/two", 0, 0, 8, 8, 8, DType::BF16},
    {"bad/all", 0, 0, 0, 0, 0, DType::BF16},
    // Dtype rejections. FP32 is the interesting one: it is a perfectly good
    // float, so the restriction is about the kernel and not about precision.
    {"bad/fp32", 8, 8, 8, 8, 8, DType::FP32},
    {"bad/int8", 8, 8, 8, 8, 8, DType::INT8},
    {"bad/fp8", 8, 8, 8, 8, 8, DType::FP8_E4M3},
    {"bad/fp8e5", 8, 8, 8, 8, 8, DType::FP8_E5M2},
    {"bad/u8", 8, 8, 8, 8, 8, DType::UINT8},
    {"bad/i32", 8, 8, 8, 8, 8, DType::INT32},
    {"bad/i64", 8, 8, 8, 8, 8, DType::INT64},
    {"bad/int4", 8, 8, 8, 8, 8, DType::INT4_PACKED},
    {"bad/mxfp4", 8, 8, 8, 8, 8, DType::MXFP4_PACKED},
    // A bad dimension AND a bad dtype: dimensions are checked first.
    {"bad/order", 0, 8, 8, 8, 8, DType::FP32},
};

void run_mla() {
    for (const auto& c : kMlaCases) {
        begin_case();
        const std::string id = std::string("mla/") + c.label;
        try {
            MlaCache cache = MlaCache::allocate(c.layers, c.pages, c.page_size,
                                                c.lora, c.rope, c.dtype);
            std::vector<std::string> f;
            f.push_back("ok");
            f.push_back("allocs=" + join(g_alloc_log));
            // The per-layer view, sampled at the first, middle and last
            // layer: it carries the dimensions the kernel reads, and it
            // reads them from the cache's own fields rather than from the
            // arguments, so a mis-stored field shows up here and nowhere else.
            std::vector<std::string> views;
            for (int l : {0, c.layers / 2, c.layers - 1}) {
                MlaCacheLayerView v = cache.layer_view(l);
                views.push_back(
                    "L" + std::to_string(v.layer) + ":p" +
                    std::to_string(v.num_pages) + ":s" +
                    std::to_string(v.page_size) + ":r" +
                    std::to_string(v.kv_lora_rank) + ":q" +
                    std::to_string(v.qk_rope_head_dim) + ":ckv" +
                    oracle_cuda::where(v.ckv_pages) + ":kpe" +
                    oracle_cuda::where(v.kpe_pages));
            }
            f.push_back("views=" + join(views));
            // Page buffers: the widths differ between ckv and kpe, which is
            // what makes an MLA pool ragged for the swap planner.
            std::vector<std::string> pbs;
            for (int l : {0, c.layers - 1}) {
                for (const auto& pb : cache.page_buffers(l)) {
                    pbs.push_back(oracle_cuda::where(pb.data) + "/" +
                                  std::to_string(pb.page_bytes));
                }
            }
            f.push_back("pages=" + join(pbs));
            emit(id, f);
        } catch (const std::exception& e) {
            emit(id, {"throw", e.what(), "allocs=" + join(g_alloc_log)});
        }
    }
}

// ---------------------------------------------------------------------------
// 2. DsV4CompressCache::allocate
// ---------------------------------------------------------------------------

struct DsCase {
    const char* label;
    std::vector<int> ratios;
    int layers, head_dim, pages, page_size;
    int fail_memset_at;
};

const std::vector<DsCase> kDsCases = {
    {"none", {}, 8, 128, 16, 16, -1},
    {"all2", {2, 2, 2, 2}, 4, 128, 16, 16, -1},
    // ratio 4 is the one that doubles the window coefficient, so state_kv and
    // state_score become twice as wide while comp_kv does not.
    {"all4", {4, 4, 4, 4}, 4, 128, 16, 16, -1},
    {"mixed", {0, 2, 4, 8, 0, 16}, 6, 64, 8, 32, -1},
    {"negatives", {-1, 2, -4, 4}, 4, 64, 8, 32, -1},
    // A ratios list that is non-empty but compresses nothing. The early
    // return only tests `ratios.empty()`, so the layer table is built at full
    // size and stays entirely blank -- which is the one shape that separates
    // `empty()` (is the TABLE empty) from "did anything allocate".
    {"all-zero-ratios", {0, 0}, 2, 64, 8, 16, -1},
    {"all-neg-ratios", {-1, -2}, 2, 64, 8, 16, -1},
    // A ratios list shorter than the layer count leaves the tail alone; a
    // longer one is truncated. Both are supported inputs, not caller errors.
    {"short-ratios", {4}, 6, 64, 8, 16, -1},
    {"long-ratios", {4, 4, 4, 4, 4, 4}, 2, 64, 8, 16, -1},
    {"zero-layers", {4, 4}, 0, 64, 8, 16, -1},
    {"neg-layers", {4, 4}, -3, 64, 8, 16, -1},
    // head_dim 0 makes every tensor zero-byte, so `allocate` still walks the
    // layers but the null-pointer guard skips every memset.
    {"hd0", {4, 2}, 2, 0, 8, 16, -1},
    {"hd-neg", {4, 2}, 2, -8, 8, 16, -1},
    {"pages0", {4, 4}, 2, 64, 0, 16, -1},
    {"psize0", {4, 4}, 2, 64, 8, 0, -1},
    {"pages-neg", {4, 4}, 2, 64, -1, 16, -1},
    {"psize-neg", {4, 4}, 2, 64, 8, -1, -1},
    {"big", {2, 4, 8}, 3, 192, 64, 64, -1},
    // The best-effort zeroing. Failing memset 0 abandons the REST OF THAT
    // LAYER (a `break`, not a `continue`) while the next layer starts over,
    // so failing 0 and failing 1 produce different transcripts and failing 3
    // shows the second layer failing while the first completed.
    {"fail0", {2, 2}, 2, 64, 8, 16, 0},
    {"fail1", {2, 2}, 2, 64, 8, 16, 1},
    {"fail2", {2, 2}, 2, 64, 8, 16, 2},
    {"fail3", {2, 2}, 2, 64, 8, 16, 3},
    {"fail4", {2, 2}, 2, 64, 8, 16, 4},
    {"fail-never", {2, 2}, 2, 64, 8, 16, 99},
};

void run_dsv4() {
    for (const auto& c : kDsCases) {
        begin_case();
        oracle_cuda::fail_memset_at(c.fail_memset_at);
        HfConfig cfg;
        cfg.num_hidden_layers = c.layers;
        cfg.head_dim = c.head_dim;
        cfg.dsv4_compress_ratios = c.ratios;
        const std::string id = std::string("dsv4/") + c.label;
        try {
            DsV4CompressCache cache =
                DsV4CompressCache::allocate(cfg, c.pages, c.page_size);
            std::vector<std::string> f;
            f.push_back("ok");
            f.push_back("allocs=" + join(g_alloc_log));
            f.push_back("ops=" + join(oracle_cuda::log()));
            f.push_back("psize=" + std::to_string(cache.page_size()));
            f.push_back("empty=" + std::string(cache.empty() ? "y" : "n"));
            std::vector<std::string> layers;
            for (int li = 0; li < c.layers; ++li) {
                // `state_width` is NOT bounds-checked -- it indexes `layers_`
                // directly, unlike `has_layer` immediately above it in the
                // same header -- so it is only safe to call once `has_layer`
                // has said the index exists.
                layers.push_back(
                    std::to_string(li) + ":" +
                    (cache.has_layer(li) ? "y" : "n") + ":" +
                    (cache.has_layer(li) ? std::to_string(cache.state_width(li))
                                         : "-"));
            }
            f.push_back("layers=" + join(layers));
            f.push_back("bpt=" + std::to_string(dsv4_compress_bytes_per_token(cfg)));
            emit(id, f);
        } catch (const std::length_error&) {
            // `layers_.resize(static_cast<std::size_t>(L))` with a negative L
            // sign-extends to a length near 2^64 and throws. The TEXT is a
            // libstdc++ artifact ("vector::_M_default_append"; libc++ says
            // something else entirely), so only the fact of the rejection is
            // pinned -- the port is not obliged to reproduce another
            // standard library's diagnostics.
            emit(id, {"throw", "length_error", "allocs=" + join(g_alloc_log)});
        } catch (const std::exception& e) {
            emit(id, {"throw", e.what(), "allocs=" + join(g_alloc_log)});
        }
    }
}

// ---------------------------------------------------------------------------
// 3. SwapPool::allocate and SwapPool::allocate_for_cache
// ---------------------------------------------------------------------------

struct SwapCase {
    const char* label;
    int layers, pages, page_size, kv_heads, head_dim;
    DType dtype;
};

const std::vector<SwapCase> kSwapCases = {
    {"tiny", 1, 1, 1, 1, 1, DType::BF16},
    {"llama8b", 32, 64, 16, 8, 128, DType::BF16},
    {"fp16", 4, 8, 16, 4, 128, DType::FP16},
    // A dtype whose element size is not 2. `allocate` multiplies the request
    // by `dtype_bytes(dtype)` while `allocate_for_cache` takes the width from
    // the cache, so the two disagree for anything the KV format quantises.
    {"fp8", 4, 8, 16, 4, 128, DType::FP8_E4M3},
    {"fp32", 4, 8, 16, 4, 128, DType::FP32},
    {"int8", 2, 4, 8, 2, 64, DType::INT8},
    // The degenerate paths. `page_bytes_` is assigned BEFORE the early return
    // in `allocate`, so a pool that allocated nothing still reports a
    // non-zero bytes-per-page whenever the layer count was positive.
    {"pages0", 8, 0, 16, 4, 128, DType::BF16},
    {"pages-neg", 8, -4, 16, 4, 128, DType::BF16},
    {"layers0", 0, 8, 16, 4, 128, DType::BF16},
    {"layers-neg", -2, 8, 16, 4, 128, DType::BF16},
    {"both0", 0, 0, 16, 4, 128, DType::BF16},
    // Zero on an axis that does NOT gate the early return: the pool is built,
    // every buffer is zero bytes, and bytes-per-page is zero.
    {"psize0", 4, 8, 0, 4, 128, DType::BF16},
    {"kvh0", 4, 8, 16, 0, 128, DType::BF16},
    {"hd0", 4, 8, 16, 4, 0, DType::BF16},
    // Negative on an axis that does not gate the early return. Every factor
    // is `static_cast<std::size_t>`ed, so a negative extent does not clamp --
    // it sign-extends to a number near 2^64 and the product wraps. The
    // request that reaches `cudaMallocHost` is then whatever survived the
    // wrap, which is exactly the case where an unchecked `int` from a config
    // turns into an allocation nobody can explain.
    {"psize-neg", 4, 8, -16, 4, 128, DType::BF16},
    {"kvh-neg", 4, 8, 16, -4, 128, DType::BF16},
    {"hd-neg", 4, 8, 16, 4, -128, DType::BF16},
    {"hd-neg1", 1, 1, 1, 1, -1, DType::BF16},
};

void run_swap_uniform() {
    for (const auto& c : kSwapCases) {
        begin_case();
        const std::string id = std::string("swap/uniform/") + c.label;
        try {
            SwapPool pool = SwapPool::allocate(c.layers, c.pages, c.page_size,
                                               c.kv_heads, c.head_dim, c.dtype);
            emit(id, {
                "ok",
                "ops=" + join(oracle_cuda::log()),
                "layers=" + std::to_string(pool.num_layers()),
                "pages=" + std::to_string(pool.num_pages()),
                "bpp=" + std::to_string(pool.bytes_per_page()),
                "streams=" +
                    std::string(pool.stream() ? "y" : "n") +
                    (pool.restore_stream() ? "y" : "n"),
                "distinct=" +
                    std::string(pool.stream() != pool.restore_stream() ? "y" : "n"),
            });
        } catch (const std::exception& e) {
            emit(id, {"throw", e.what(), "ops=" + join(oracle_cuda::log())});
        }
    }
}

// The cache-driven constructor. Sweeps the same formats the KV cache oracle
// covers, because the number of host buffers per layer is `page_buffers()`
// and that is 2 for a native cache and 4 for a scaled one.

struct CacheCase {
    const char* label;
    KvCacheScheme scheme;
    KvCacheScaleLayout scale;
    DType storage;
    int block;
    int layers, pages, page_size, kv_heads, head_dim;
    int host_pages;
};

const std::vector<CacheCase> kCacheCases = {
    {"bf16", KvCacheScheme::Native, KvCacheScaleLayout::None, DType::BF16, 0,
     4, 8, 16, 4, 128, 6},
    {"fp16", KvCacheScheme::Native, KvCacheScaleLayout::None, DType::FP16, 0,
     2, 8, 16, 4, 128, 3},
    {"fp8pt", KvCacheScheme::Fp8PerTensor, KvCacheScaleLayout::None,
     DType::FP8_E4M3, 0, 3, 8, 16, 4, 128, 5},
    // Four device buffers per layer. `allocate` would have made two.
    {"fp8pth", KvCacheScheme::Fp8PerTokenHead, KvCacheScaleLayout::PerTokenHead,
     DType::FP8_E4M3, 0, 3, 8, 16, 4, 128, 5},
    {"int8pth", KvCacheScheme::Int8PerTokenHead,
     KvCacheScaleLayout::PerTokenHead, DType::INT8, 0, 2, 8, 16, 4, 128, 4},
    {"fp4b16", KvCacheScheme::Fp4Block, KvCacheScaleLayout::PerTokenHeadBlock,
     DType::FP8_E4M3, 16, 2, 8, 16, 4, 128, 4},
    {"fp4b32", KvCacheScheme::Fp4Block, KvCacheScaleLayout::PerTokenHeadBlock,
     DType::FP8_E4M3, 32, 2, 8, 16, 4, 128, 4},
    {"host0", KvCacheScheme::Native, KvCacheScaleLayout::None, DType::BF16, 0,
     4, 8, 16, 4, 128, 0},
    {"host-neg", KvCacheScheme::Native, KvCacheScaleLayout::None, DType::BF16,
     0, 4, 8, 16, 4, 128, -3},
    {"nolayers", KvCacheScheme::Native, KvCacheScaleLayout::None, DType::BF16,
     0, 0, 8, 16, 4, 128, 4},
    // A device cache with zero pages: its page buffers still have a width, so
    // the host pool is allocated at full size against an empty device side.
    {"devpages0", KvCacheScheme::Native, KvCacheScaleLayout::None, DType::BF16,
     0, 2, 0, 16, 4, 128, 4},
    {"big", KvCacheScheme::Native, KvCacheScaleLayout::None, DType::BF16, 0,
     8, 32, 32, 8, 64, 12},
};

void run_swap_for_cache() {
    for (const auto& c : kCacheCases) {
        begin_case();
        const std::string id = std::string("swap/cache/") + c.label;
        try {
            KvCacheFormat fmt;
            fmt.scheme = c.scheme;
            fmt.scale_layout = c.scale;
            fmt.storage_dtype = c.storage;
            fmt.block_size = c.block;
            KvCache cache = KvCache::allocate(c.layers, c.pages, c.page_size,
                                              c.kv_heads, c.head_dim, fmt);
            const std::size_t dev_allocs = g_alloc_log.size();
            oracle_cuda::note("--");
            SwapPool pool = SwapPool::allocate_for_cache(cache, c.host_pages);
            std::vector<std::string> f;
            f.push_back("ok");
            f.push_back("devallocs=" + std::to_string(dev_allocs));
            f.push_back("ops=" + join(oracle_cuda::log()));
            f.push_back("layers=" + std::to_string(pool.num_layers()));
            f.push_back("pages=" + std::to_string(pool.num_pages()));
            f.push_back("bpp=" + std::to_string(pool.bytes_per_page()));
            f.push_back("streams=" + std::string(pool.stream() ? "y" : "n") +
                        (pool.restore_stream() ? "y" : "n"));
            // What the copy loops will index against, per layer. If this is
            // wider or longer than the host side the pool built, the copy
            // loops read `host_pools_[layer][b]` out of bounds.
            std::vector<std::string> devbufs;
            for (int l = 0; l < c.layers; ++l) {
                std::string row = std::to_string(l) + ":";
                for (const auto& pb : cache.page_buffers(l)) {
                    row += std::to_string(pb.page_bytes) + "/";
                }
                devbufs.push_back(row);
            }
            f.push_back("dev=" + join(devbufs));
            emit(id, f);
        } catch (const std::exception& e) {
            emit(id, {"throw", e.what(), "ops=" + join(oracle_cuda::log())});
        }
    }
}

}  // namespace

int main() {
    run_mla();
    run_dsv4();
    run_swap_uniform();
    run_swap_for_cache();
    return 0;
}
