// Combined differential oracle for the store/ layers ported into
// driver-cuda-new. Every .inc it includes is EXTRACTED VERBATIM from the real
// sources by extract.sh, so this cannot silently drift from what it tests.
//
// The CUDA surface is stubbed (stub/cuda_runtime.h) rather than mocked away:
// cudaMemcpyAsync really copies, so the swap routines move real bytes between
// real buffers and the transcript can hash the result. That verifies the
// copies' semantics, not just the offsets they computed.
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

#include <cuda_runtime.h>

#include "cuda_check.hpp"
#include "store/kv_cache_format.hpp"

// ---------------------------------------------------------------------------
// Pool identity. Buffers are registered so a raw pointer can be printed as
// (pool id, byte offset) instead of an address that changes every run.
// ---------------------------------------------------------------------------
namespace {
struct Reg { const std::uint8_t* base; std::size_t len; int id; };
std::vector<Reg> g_pools;

void register_pool(const void* p, std::size_t len, int id) {
    g_pools.push_back({static_cast<const std::uint8_t*>(p), len, id});
}
std::pair<int, long long> identify(const void* p) {
    const auto* q = static_cast<const std::uint8_t*>(p);
    for (const auto& r : g_pools) {
        if (q >= r.base && q < r.base + r.len) return {r.id, q - r.base};
    }
    return {-1, -1};
}
std::uint64_t fnv1a(const void* data, std::size_t n) {
    std::uint64_t h = 0xcbf29ce484222325ull;
    const auto* p = static_cast<const std::uint8_t*>(data);
    for (std::size_t i = 0; i < n; ++i) { h ^= p[i]; h *= 0x100000001b3ull; }
    return h;
}
bool g_trace = false;
}  // namespace

// The stub's cudaMemcpyAsync routes here so the plan is observable.
void oracle_note_copy(void* dst, const void* src, std::size_t n) {
    if (!g_trace) return;
    auto d = identify(dst);
    auto s = identify(src);
    printf("SWAPOP\t%d\t%lld\t%d\t%lld\t%zu\n", d.first, d.second, s.first, s.second, n);
}

namespace pie_cuda_driver {

// --- stand-ins carrying only what the extracted code reads -----------------
struct HfConfig {
    int head_dim = 128;
    std::vector<int> dsv4_compress_ratios;
};

struct PlannerProfileKey {
    std::string gpu_name;
    int compute_major = 0;
    int compute_minor = 0;
    int sm_count = 0;
    std::string kv_cache_dtype;
    int tp_size = 1;
    std::string model_type;
    int hidden_size = 0;
    int num_hidden_layers = 0;
    int num_attention_heads = 0;
    int num_key_value_heads = 0;
    int head_dim = 0;
};

// MlaCache::page_buffers reads dtype_, page_size_, kv_lora_rank_,
// qk_rope_head_dim_ and calls .data() on its tensors.
struct FakeTensor { void* p = nullptr; void* data() const { return p; } };
class MlaCache {
  public:
    struct PageBuffer { void* data = nullptr; std::size_t page_bytes = 0; };
    int num_layers_ = 0, num_pages_ = 0, page_size_ = 0;
    int kv_lora_rank_ = 0, qk_rope_head_dim_ = 0;
    DType dtype_ = DType::BF16;
    std::vector<FakeTensor> ckv_layers_, kpe_layers_;
    std::vector<PageBuffer> page_buffers(int layer);
};
#include "mla.inc"

namespace { 
#include "dsv4.inc"
}  // namespace

// RecurrentStateCache's addressing reads the shape members plus
// linear_layer_index_ and the two base pointers.
class RecurrentStateCache {
  public:
    std::vector<int> linear_layer_index_;
    int max_slots_ = 1, conv_dim_ = 0, conv_kernel_ = 0;
    int v_heads_ = 0, head_k_dim_ = 0, head_v_dim_ = 0, hidden_size_ = 0;
    bool recurrent_state_bf16_ = false;
    FakeTensor conv_states_, recurrent_states_, recurrent_states_bf16_;
    struct MtpBuf { std::uint16_t* p = nullptr; std::uint16_t* data() const { return p; } };
    MtpBuf mtp_pending_hidden_;
#include "rec_strides.inc"
    void* conv_state(int layer, int slot);
    void* recurrent_state_raw(int layer, int slot);
    void* mtp_pending_hidden(int slot);
};
#include "rec_addr.inc"

// SwapPool's copy routines read num_layers_, host_pools_, the two streams,
// and cache.page_buffers(layer).
class KvCache {
  public:
    struct PageBuffer { void* data = nullptr; std::size_t page_bytes = 0; };
    std::vector<std::vector<PageBuffer>> bufs;
    std::vector<PageBuffer> page_buffers(int layer) { return bufs[layer]; }
};
class SwapPool {
  public:
    struct HostBuffer { void* data = nullptr; std::size_t page_bytes = 0; };
    int num_layers_ = 0;
    std::vector<std::vector<HostBuffer>> host_pools_;
    cudaStream_t stream_ = nullptr;
    cudaStream_t restore_stream_ = nullptr;
    void synchronize() const {}
    void copy_d2h_async(KvCache&, std::span<const std::uint32_t>, std::span<const std::uint32_t>);
    void copy_h2d_async(KvCache&, std::span<const std::uint32_t>, std::span<const std::uint32_t>);
    void copy_d2d_async(KvCache&, std::span<const std::uint32_t>, std::span<const std::uint32_t>);
    void copy_h2h_async(std::span<const std::uint32_t>, std::span<const std::uint32_t>);
};
namespace {
#include "swap_helpers.inc"
}  // namespace
#include "swap_copy.inc"

namespace {
#include "profile.inc"
}  // namespace

}  // namespace pie_cuda_driver

using namespace pie_cuda_driver;

// ---------------------------------------------------------------------------
int main() {
    // === MLA page geometry ===================================================
    const std::vector<int> pages = {1, 2, 8, 64, 4096};
    const std::vector<int> psizes = {1, 16, 32, 64, 128};
    const std::vector<int> ranks = {1, 64, 128, 512, 576};
    const std::vector<int> ropes = {1, 16, 64, 128, 192};
    for (DType dt : {DType::BF16, DType::FP16})
    for (int np : pages) for (int ps : psizes) for (int r : ranks) for (int q : ropes) {
        MlaCache m;
        m.num_layers_ = 4; m.num_pages_ = np; m.page_size_ = ps;
        m.kv_lora_rank_ = r; m.qk_rope_head_dim_ = q; m.dtype_ = dt;
        m.ckv_layers_.resize(4); m.kpe_layers_.resize(4);
        auto b = m.page_buffers(0);
        printf("MLA\t%d\t%d\t%d\t%d\t%d\t%zu\t%zu\n", (int)dt, np, ps, r, q,
               b[0].page_bytes, b[1].page_bytes);
    }

    // === DSV4 compressor geometry ===========================================
    const std::vector<std::vector<int>> ratio_sets = {
        {}, {0}, {1}, {2}, {4}, {8}, {-1}, {2, 4}, {4, 4, 4},
        {0, 2, 0, 4}, {1, 2, 3, 4, 5, 6, 7, 8}, {4, 0, -3, 16}};
    for (const auto& rs : ratio_sets) for (int hd : {1, 16, 64, 128, 192, 576}) {
        HfConfig cfg; cfg.head_dim = hd; cfg.dsv4_compress_ratios = rs;
        printf("DSV4\t%zu\t%d\t%zu", rs.size(), hd, dsv4_compress_bytes_per_token(cfg));
        for (int r : rs) printf("\t%d:%d", r, compressor_coff(r));
        printf("\n");
    }

    // === Recurrent state addressing =========================================
    const std::vector<std::vector<bool>> stacks = {
        {true}, {false}, {true, true}, {true, false, true},
        {true, false, false, true, false, false, false, true},
        {false, false, true, true}, {false, false, false, false}};
    for (const auto& st : stacks)
    for (bool bf16 : {false, true})
    for (int slots : {1, 2, 16})
    for (int ck : {2, 4}) for (int cd : {8, 4096})
    for (int vh : {1, 32}) for (int hk : {8, 128}) for (int hv : {8, 128})
    for (int hs : {0, 2048}) {
        RecurrentStateCache c;
        c.max_slots_ = slots; c.conv_dim_ = cd; c.conv_kernel_ = ck;
        c.v_heads_ = vh; c.head_k_dim_ = hk; c.head_v_dim_ = hv;
        c.hidden_size_ = hs; c.recurrent_state_bf16_ = bf16;
        int next = 0;
        c.linear_layer_index_.assign(st.size(), -1);
        for (std::size_t i = 0; i < st.size(); ++i) if (st[i]) c.linear_layer_index_[i] = next++;
        // Distinct non-null fake bases: `mtp_pending_hidden` early-returns
        // on a null base, so a zero base would make that axis test nothing.
        auto* CONV = reinterpret_cast<std::uint8_t*>(0x10000000ull);
        auto* REC  = reinterpret_cast<std::uint8_t*>(0x20000000ull);
        auto* RECB = reinterpret_cast<std::uint8_t*>(0x30000000ull);
        auto* MTP  = reinterpret_cast<std::uint8_t*>(0x40000000ull);
        c.conv_states_.p = CONV;
        c.recurrent_states_.p = REC;
        c.recurrent_states_bf16_.p = RECB;
        c.mtp_pending_hidden_.p = hs > 0 ? reinterpret_cast<std::uint16_t*>(MTP) : nullptr;
        auto* rec_base = bf16 ? RECB : REC;
        printf("RECSTRIDE\t%zu\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%zu\t%zu\t%zu\n",
               st.size(), (int)bf16, slots, ck, cd, vh, hk, hv, hs,
               c.conv_slot_stride_bytes(), c.recurrent_slot_stride_floats(),
               c.recurrent_slot_stride_bytes());
        for (std::size_t L = 0; L < st.size(); ++L) for (int s = 0; s < slots; ++s) {
            long long cs = -1, rs = -1;
            try { void* p = c.conv_state((int)L, s);
                  cs = p ? (long long)((std::uint8_t*)p - CONV) : -1; } catch (...) { cs = -2; }
            try { void* p = c.recurrent_state_raw((int)L, s);
                  rs = p ? (long long)((std::uint8_t*)p - rec_base) : -1; } catch (...) { rs = -2; }
            printf("RECADDR\t%zu\t%d\t%d\t%zu\t%d\t%lld\t%lld\n",
                   st.size(), (int)bf16, slots, L, s, cs, rs);
        }
        for (int s = 0; s < slots + 1; ++s) {
            long long m = -1;
            try { void* p = c.mtp_pending_hidden(s);
                  m = p ? (long long)((std::uint8_t*)p - MTP) : -1; } catch (...) { m = -2; }
            printf("RECMTP\t%d\t%d\t%d\t%lld\n", slots, hs, s, m);
        }
    }

        // Out-of-bounds sweep, kept separate from the main grid so it stays cheap.
    // This is the only place the throw-vs-null split is exercised for
    // conv_state / recurrent_state_raw.
    for (const auto& st : stacks) {
        RecurrentStateCache c;
        c.max_slots_ = 4; c.conv_dim_ = 4096; c.conv_kernel_ = 4;
        c.v_heads_ = 32; c.head_k_dim_ = 128; c.head_v_dim_ = 128;
        c.hidden_size_ = 2048; c.recurrent_state_bf16_ = false;
        {   int next = 0;
            c.linear_layer_index_.assign(st.size(), -1);
            for (std::size_t i = 0; i < st.size(); ++i) if (st[i]) c.linear_layer_index_[i] = next++;
        }
        auto* CONV = reinterpret_cast<std::uint8_t*>(0x10000000ull);
        auto* REC  = reinterpret_cast<std::uint8_t*>(0x20000000ull);
        c.conv_states_.p = CONV;
        c.recurrent_states_.p = REC;
        c.recurrent_states_bf16_.p = nullptr;
        c.mtp_pending_hidden_.p = reinterpret_cast<std::uint16_t*>(0x40000000ull);
        for (int layer = 0; layer <= (int)st.size() + 1; ++layer) {
            for (int s = 0; s <= 5; ++s) {
                long long cs, rs;
                try { void* p = c.conv_state(layer, s);
                      cs = p ? (long long)((std::uint8_t*)p - CONV) : -1; } catch (...) { cs = -2; }
                try { void* p = c.recurrent_state_raw(layer, s);
                      rs = p ? (long long)((std::uint8_t*)p - REC) : -1; } catch (...) { rs = -2; }
                printf("RECOOB\t%d\t%d\t%d\t%lld\t%lld\n", (int)st.size(), layer, s, cs, rs);
            }
        }
    }

// === Swap copy plans, executed on real memory ===========================
    g_trace = true;
    struct Case { int layers; std::vector<std::size_t> widths; std::vector<std::uint32_t> src, dst; };
    const std::vector<Case> cases = {
        {1, {64}, {0}, {0}},
        {1, {64}, {0, 1}, {1, 0}},
        {2, {64, 64}, {0, 2}, {3, 1}},
        {3, {32, 128}, {0, 1, 2}, {4, 5, 6}},
        {2, {16, 256}, {7}, {0}},
        {1, {64}, {}, {}},
    };
    int ci = 0;
    for (const auto& c : cases) {
        for (int dir = 0; dir < 4; ++dir) {
            g_pools.clear();
            const std::size_t NPAGES = 8;
            // Two independent pool families: device and host.
            std::vector<std::vector<std::vector<std::uint8_t>>> dev(c.layers), host(c.layers);
            KvCache cache; cache.bufs.resize(c.layers);
            SwapPool sp; sp.num_layers_ = c.layers; sp.host_pools_.resize(c.layers);
            for (int L = 0; L < c.layers; ++L) {
                for (std::size_t b = 0; b < c.widths.size(); ++b) {
                    dev[L].emplace_back(c.widths[b] * NPAGES);
                    host[L].emplace_back(c.widths[b] * NPAGES);
                }
            }
            for (int L = 0; L < c.layers; ++L) {
                for (std::size_t b = 0; b < c.widths.size(); ++b) {
                    auto& dv = dev[L][b]; auto& hv = host[L][b];
                    // Distinguishable fill: byte value encodes pool and offset.
                    for (std::size_t i = 0; i < dv.size(); ++i)
                        dv[i] = (std::uint8_t)(0x10 + L * 3 + b * 7 + (i % 251));
                    for (std::size_t i = 0; i < hv.size(); ++i)
                        hv[i] = (std::uint8_t)(0x80 + L * 5 + b * 11 + (i % 241));
                    cache.bufs[L].push_back({dv.data(), c.widths[b]});
                    sp.host_pools_[L].push_back({hv.data(), c.widths[b]});
                    register_pool(dv.data(), dv.size(), 1000 + L * 10 + (int)b);
                    register_pool(hv.data(), hv.size(), 2000 + L * 10 + (int)b);
                }
            }
            printf("SWAPCASE\t%d\t%d\n", ci, dir);
            std::span<const std::uint32_t> ss(c.src), ds(c.dst);
            try {
                switch (dir) {
                    case 0: sp.copy_d2h_async(cache, ss, ds); break;
                    case 1: sp.copy_h2d_async(cache, ss, ds); break;
                    case 2: sp.copy_d2d_async(cache, ss, ds); break;
                    case 3: sp.copy_h2h_async(ss, ds); break;
                }
            } catch (const std::exception& e) {
                printf("SWAPERR\t%s\n", e.what());
            }
            for (int L = 0; L < c.layers; ++L)
                for (std::size_t b = 0; b < c.widths.size(); ++b)
                    printf("SWAPHASH\t%d\t%zu\t%016llx\t%016llx\n", L, b,
                           (unsigned long long)fnv1a(dev[L][b].data(), dev[L][b].size()),
                           (unsigned long long)fnv1a(host[L][b].data(), host[L][b].size()));
        }
        ++ci;
    }
    // A mismatched pair must throw before touching anything.
    {
        g_pools.clear();
        KvCache cache; cache.bufs.resize(1);
        std::vector<std::uint8_t> d(64), h(64);
        cache.bufs[0].push_back({d.data(), 64});
        SwapPool sp; sp.num_layers_ = 1; sp.host_pools_.resize(1);
        sp.host_pools_[0].push_back({h.data(), 64});
        std::vector<std::uint32_t> a = {0, 1}, b = {0};
        try { sp.copy_d2h_async(cache, a, b); }
        catch (const std::exception& e) { printf("SWAPERR\t%s\n", e.what()); }
    }
    g_trace = false;

    // === Planner profile key ================================================
    const std::vector<std::string> gpu_names = {
        "NVIDIA H100 80GB HBM3", "NVIDIA GeForce RTX 4090", "",
        "a\"b\\c", "tab\there", "line\nbreak", "ctrl\x01" "char", "ctrl\x1c" "har", "slash/es", "\xe6\x97\xa5\xe6\x9c\xac"};
    for (const auto& gn : gpu_names) for (int major : {7, 9, 12}) for (int sm : {0, 132, -1}) {
        PlannerProfileKey k;
        k.gpu_name = gn; k.compute_major = major; k.compute_minor = 0;
        k.sm_count = sm; k.kv_cache_dtype = "bf16"; k.tp_size = 1;
        k.model_type = "llama"; k.hidden_size = 8192; k.num_hidden_layers = 80;
        k.num_attention_heads = 64; k.num_key_value_heads = 8; k.head_dim = 128;
        printf("KEYJSON\t%s\n", key_to_json(k).dump().c_str());
    }
    // Matching semantics: mutate one field at a time, in each of six ways.
    {
        PlannerProfileKey k;
        k.gpu_name = "NVIDIA H100 80GB HBM3"; k.compute_major = 9; k.compute_minor = 0;
        k.sm_count = 132; k.kv_cache_dtype = "bf16"; k.tp_size = 1;
        k.model_type = "llama"; k.hidden_size = 8192; k.num_hidden_layers = 80;
        k.num_attention_heads = 64; k.num_key_value_heads = 8; k.head_dim = 128;
        const char* fields[] = {"gpu_name", "compute_major", "compute_minor", "sm_count",
                                "kv_cache_dtype", "tp_size", "model_type", "hidden_size",
                                "num_hidden_layers", "num_attention_heads",
                                "num_key_value_heads", "head_dim"};
        printf("KEYMATCH\tbaseline\t-\t%d\n", (int)key_matches(key_to_json(k), k));
        for (const char* f : fields) {
            for (int mode = 0; mode < 6; ++mode) {
                nlohmann::json j = key_to_json(k);
                switch (mode) {
                    case 0: j.erase(f); break;                    // missing
                    case 1: j[f] = nullptr; break;                // null
                    case 2: j[f] = true; break;                   // bool
                    case 3: j[f] = 132.0; break;                  // float
                    case 4: j[f] = "132"; break;                  // string
                    case 5: j[f] = 132; break;                    // integer
                }
                printf("KEYMATCH\t%s\t%d\t%d\n", f, mode, (int)key_matches(j, k));
            }
        }
    }
    return 0;
}
