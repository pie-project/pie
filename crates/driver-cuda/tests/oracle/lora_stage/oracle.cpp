// The lora staging oracle — gate-lora, slice A (the stage phase).
//
// Compiles the REAL `llama_like.cpp` (the prepare oracle's construction and
// stub tree) and drives `llama_like_lora_stage`: the fire-scoped staging
// the engine runs OUTSIDE any capture region. What the golden pins:
//
//   * the validation chain — nine ways a lane table is refused, each with
//     the C++'s message;
//   * the arena discipline (256-aligned bump allocs, the doubling growth
//     with a 1 MiB floor, reset per fire);
//   * the bf16 casts (which adapter, how many elements, to which arena
//     offset);
//   * same-shape grouping: the (rank, d_in, d_out) key, groups of one
//     pruned, the disjoint-span precondition falling back to per-lane,
//     scale lanes never grouped;
//   * the grouped xA^T scratch layout (per-lane element offsets) and the
//     POINTER SLAB — every slot of every layer, symbolically, because the
//     slab is what a captured body replays against;
//   * the splitmix fingerprint over lane structure, adapter addresses and
//     the post-staging arena base (addresses are fabricated
//     deterministically on both sides, so the VALUE is comparable);
//   * `grouping_desc`, the operator's one-line instrument.
//
// `apply()` — the body-time launches — is the gate's slice B, landing with
// the emitter work that generates its call sites.

#include <cstdio>
#include <cstring>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include "model/llama_like/llama_like.hpp"
#include "model/lora.hpp"
#include "model/workspace.hpp"

using pie_cuda_driver::DeviceTensor;
using pie_cuda_driver::DType;
using pie_cuda_driver::HfConfig;
using pie_cuda_driver::model::LlamaLikeForwardCfg;
using pie_cuda_driver::model::LlamaLikePlanState;
using pie_cuda_driver::model::LoraLaneView;
using pie_cuda_driver::model::LoraTable;
using pie_cuda_driver::model::Workspace;

namespace {

constexpr char SEP = '\x1f';
std::string g_case;
std::map<const void*, std::pair<std::string, std::size_t>> g_regions;
int g_next_dev = 0;

void note(const std::string& body) {
    std::printf("%s%c%s\n", g_case.c_str(), SEP, body.c_str());
}

std::string where(const void* p) {
    if (p == nullptr) return "null";
    auto it = g_regions.upper_bound(p);
    if (it != g_regions.begin()) {
        --it;
        const auto* base = static_cast<const unsigned char*>(it->first);
        const auto* q = static_cast<const unsigned char*>(p);
        const std::size_t off = static_cast<std::size_t>(q - base);
        if (off < it->second.second) {
            return it->second.first + "+" + std::to_string(off);
        }
    }
    return "unknown";
}

void name_region(const void* p, std::size_t bytes, const std::string& name) {
    g_regions[p] = {name, bytes};
}

}  // namespace

// ── recorders ───────────────────────────────────────────────────────────────

namespace pie_cuda_driver {

// The arena's DeviceBuffer backing. Deterministic bases: dev#K at
// 0x10000000 + K * 0x1000000, so the fingerprint's arena-address mix is a
// number both sides can compute.
DeviceMemoryBlock allocate_device_memory(std::size_t bytes, std::size_t) {
    if (bytes == 0) return DeviceMemoryBlock{nullptr, false};
    void* p = reinterpret_cast<void*>(
        static_cast<std::uintptr_t>(0x10000000) +
        static_cast<std::uintptr_t>(g_next_dev) * 0x1000000);
    name_region(p, bytes, "dev#" + std::to_string(g_next_dev));
    ++g_next_dev;
    note("dev-alloc dev#" + std::to_string(g_next_dev - 1) +
         " bytes=" + std::to_string(bytes));
    return DeviceMemoryBlock{p, false};
}

void free_device_memory(DeviceMemoryBlock) noexcept {}

namespace kernels::quant {
void cast_fp32_to_bf16(const void* src, void* dst, std::size_t elems,
                       cudaStream_t) {
    note("cast src=" + where(src) + " dst=" + where(dst) +
         " elems=" + std::to_string(elems));
}
}  // namespace kernels::quant

// The plan-state destructor needs the deleters; nothing plans here.
namespace kernels::attn {
struct DecodePlanCache {};
struct PrefillPlanCache {};
void DecodePlanCacheDeleter::operator()(DecodePlanCache* p) const noexcept {
    delete p;
}
void PrefillPlanCacheDeleter::operator()(PrefillPlanCache* p) const noexcept {
    delete p;
}
}  // namespace kernels::attn

}  // namespace pie_cuda_driver

// The slab upload: print every slot symbolically — the slab is the staged
// truth a captured body replays against.
cudaError_t cudaMemcpyAsync(void* dst, const void* src, std::size_t bytes,
                            cudaMemcpyKind kind, cudaStream_t) {
    std::string body = "slab dst=" + where(dst) + " kind=" +
                       std::to_string(static_cast<int>(kind)) + " slots=[";
    const auto* slots = static_cast<const void* const*>(src);
    for (std::size_t i = 0; i < bytes / sizeof(void*); ++i) {
        if (i) body += ',';
        body += where(slots[i]);
    }
    note(body + "]");
    return cudaSuccess;
}

namespace {

// The fire fixture: a Workspace whose five lora-read buffers are views
// over fabricated, registered addresses; the arena starts empty and grows
// through the recorder.
struct Fixture {
    Workspace ws;
    HfConfig cfg{};
    LlamaLikeForwardCfg fwd;
    LlamaLikePlanState state;

    static constexpr int kH = 8;    // hidden
    static constexpr int kHq = 12;  // q width
    static constexpr int kHk = 4;   // kv width
    static constexpr int kI = 16;   // intermediate (rank ceiling)
    static constexpr int kLayers = 2;

    int hk = kHk;

    explicit Fixture(int max_tokens, int kv_width = kHk) : hk(kv_width) {
        cfg.num_hidden_layers = kLayers;
        cfg.hidden_size = kH;
        cfg.num_attention_heads = kHq;  // head_dim 1: widths in elements
        cfg.num_key_value_heads = kv_width;
        cfg.head_dim = 1;
        cfg.head_dim_kernel = 1;
        cfg.intermediate_size = kI;
        auto view = [&](const char* name, int width) {
            void* base = reinterpret_cast<void*>(
                static_cast<std::uintptr_t>(0x1000) *
                (1 + static_cast<std::uintptr_t>(g_regions.size())));
            const std::size_t bytes =
                static_cast<std::size_t>(max_tokens) * width * 2;
            name_region(base, bytes, name);
            return DeviceTensor::view(base, DType::BF16,
                                      {max_tokens, width});
        };
        ws.y = view("ws.y", kH);
        ws.norm_x = view("ws.norm_x", kH);
        ws.q = view("ws.q", kHq);
        ws.v = view("ws.v", kv_width);
        ws.gate = view("ws.gate", kI);
    }
};

// Adapter addresses: registered constants, distinct per lane.
const void* adapter(const char* name, int ordinal) {
    void* p = reinterpret_cast<void*>(
        static_cast<std::uintptr_t>(0x40000000) +
        static_cast<std::uintptr_t>(ordinal) * 0x100000);
    name_region(p, 0x100000, name);
    return p;
}

LoraLaneView lane(
    const void* a, const void* b, std::uint64_t sites, std::uint32_t start,
    std::uint32_t count, std::uint32_t layers, std::uint32_t rank,
    std::uint32_t d_in, std::uint32_t d_out,
    LoraLaneView::Form form = LoraLaneView::Form::LowRank) {
    LoraLaneView v;
    v.a = a;
    v.b = b;
    v.sites_bits = sites;
    v.token_start = start;
    v.token_count = count;
    v.num_layers = layers;
    v.rank = rank;
    v.d_in = d_in;
    v.d_out = d_out;
    v.form = form;
    return v;
}

void run_stage(const char* name, Fixture& f, const std::vector<LoraLaneView>& lanes,
               int total_tokens) {
    g_case = name;
    const LoraTable table{lanes.data(),
                          static_cast<std::uint32_t>(lanes.size())};
    note("call lanes=" + std::to_string(lanes.size()) + " N=" +
         std::to_string(total_tokens));
    try {
        const std::uint64_t fp = pie_cuda_driver::model::llama_like_lora_stage(
            f.state, f.ws, &table, f.cfg, f.fwd, total_tokens, nullptr);
        char buf[32];
        std::snprintf(buf, sizeof buf, "0x%016llx",
                      static_cast<unsigned long long>(fp));
        note(std::string("staged fp=") + buf + " handle=" +
             (f.state.lora_staged != nullptr ? "set" : "null") + " table=" +
             (f.state.lora_staged_table == &table ? "this" : "other") +
             " desc=" +
             (f.state.lora_staged != nullptr
                  ? f.state.lora_staged->grouping_desc()
                  : "-"));
    } catch (const std::exception& e) {
        note(std::string("threw ") + e.what());
    }
}

}  // namespace

int main(int argc, char**) {
    (void)argc;
    const std::uint64_t Q = pie_cuda_driver::model::kLoraSiteQ;
    const std::uint64_t V = pie_cuda_driver::model::kLoraSiteV;
    const std::uint64_t K = pie_cuda_driver::model::kLoraSiteK;

    // a. The no-program paths: a null table, then an unusable one.
    {
        Fixture f(16);
        g_case = "a-null";
        const std::uint64_t fp = pie_cuda_driver::model::llama_like_lora_stage(
            f.state, f.ws, nullptr, f.cfg, f.fwd, 16, nullptr);
        note("fp=" + std::to_string(fp) + " handle=" +
             (f.state.lora_staged != nullptr ? "set" : "null"));
        const LoraTable empty{nullptr, 0};
        const std::uint64_t fp2 = pie_cuda_driver::model::llama_like_lora_stage(
            f.state, f.ws, &empty, f.cfg, f.fwd, 16, nullptr);
        note("empty fp=" + std::to_string(fp2));
    }

    // b. One low-rank lane on q|v: two casts, no grouping, the arena's
    //    first growth, and the fingerprint.
    {
        Fixture f(16);
        run_stage("b-solo", f,
                  {lane(adapter("A0", 0), adapter("B0", 1), Q, 0, 8,
                        Fixture::kLayers, 2, Fixture::kH, Fixture::kHq)},
                  16);
    }

    // b2. A second fire on the SAME state re-stages: arena reset (offsets
    //     restart), handle replaced.
    {
        Fixture f(16);
        std::vector<LoraLaneView> lanes{
            lane(adapter("A0", 0), adapter("B0", 1), Q, 0, 8,
                 Fixture::kLayers, 2, Fixture::kH, Fixture::kHq)};
        run_stage("b2-first", f, lanes, 16);
        run_stage("b2-restage", f, lanes, 16);
    }

    // c. Two same-shape disjoint lanes GROUP: "2xr2", packed xA^T
    //    offsets, and the full pointer slab.
    {
        Fixture f(16);
        run_stage(
            "c-grouped", f,
            {lane(adapter("A0", 0), adapter("B0", 1), Q, 0, 6,
                  Fixture::kLayers, 2, Fixture::kH, Fixture::kHq),
             lane(adapter("A1", 2), adapter("B1", 3), Q, 6, 4,
                  Fixture::kLayers, 2, Fixture::kH, Fixture::kHq)},
            16);
    }

    // c2. Mixed q and v members in ONE group: needs Hq == Hk so the two
    //     lanes share the (rank, d_in, d_out) key — the per-site slot runs
    //     (nq=1, nv=1) are what this case pins.
    {
        Fixture f(16, Fixture::kHq);
        run_stage(
            "c2-grouped-qv", f,
            {lane(adapter("A0", 0), adapter("B0", 1), Q, 0, 6,
                  Fixture::kLayers, 2, Fixture::kH, Fixture::kHq),
             lane(adapter("A1", 2), adapter("B1", 3), V, 6, 4,
                  Fixture::kLayers, 2, Fixture::kH, Fixture::kHq)},
            16);
    }

    // d. Different shapes do not group ("none(2 solo)"); a scale lane
    //    never groups.
    {
        Fixture f(16);
        run_stage(
            "d-shapes-differ", f,
            {lane(adapter("A0", 0), adapter("B0", 1), Q, 0, 6,
                  Fixture::kLayers, 2, Fixture::kH, Fixture::kHq),
             lane(adapter("A1", 2), adapter("B1", 3), Q, 6, 4,
                  Fixture::kLayers, 4, Fixture::kH, Fixture::kHq)},
            16);
    }
    {
        Fixture f(16);
        run_stage(
            "d2-scale-lane", f,
            {lane(adapter("L0", 0), nullptr, V, 0, 6, Fixture::kLayers, 0, 0,
                  Fixture::kHk, LoraLaneView::Form::Scale),
             lane(adapter("A1", 2), adapter("B1", 3), Q, 6, 4,
                  Fixture::kLayers, 2, Fixture::kH, Fixture::kHq)},
            16);
    }

    // e. Overlapping spans fall back to per-lane pairs.
    {
        Fixture f(16);
        run_stage(
            "e-overlap", f,
            {lane(adapter("A0", 0), adapter("B0", 1), Q, 0, 8,
                  Fixture::kLayers, 2, Fixture::kH, Fixture::kHq),
             lane(adapter("A1", 2), adapter("B1", 3), Q, 4, 8,
                  Fixture::kLayers, 2, Fixture::kH, Fixture::kHq)},
            16);
    }

    // f. A zero-count lane is silently dropped.
    {
        Fixture f(16);
        run_stage(
            "f-empty-span", f,
            {lane(adapter("A0", 0), adapter("B0", 1), Q, 0, 0,
                  Fixture::kLayers, 2, Fixture::kH, Fixture::kHq),
             lane(adapter("A1", 2), adapter("B1", 3), Q, 0, 8,
                  Fixture::kLayers, 2, Fixture::kH, Fixture::kHq)},
            16);
    }

    // g. The refusal chain, one case per guard.
    {
        Fixture f(16);
        run_stage("g1-null-adapter", f,
                  {lane(nullptr, adapter("B0", 1), Q, 0, 8, Fixture::kLayers,
                        2, Fixture::kH, Fixture::kHq)},
                  16);
        run_stage("g2-no-sites", f,
                  {lane(adapter("A0", 0), adapter("B0", 1), 0, 0, 8,
                        Fixture::kLayers, 2, Fixture::kH, Fixture::kHq)},
                  16);
        run_stage("g3-unknown-bits", f,
                  {lane(adapter("A0", 0), adapter("B0", 1), 1ull << 9, 0, 8,
                        Fixture::kLayers, 2, Fixture::kH, Fixture::kHq)},
                  16);
        run_stage("g4-reserved-site", f,
                  {lane(adapter("A0", 0), adapter("B0", 1), K, 0, 8,
                        Fixture::kLayers, 2, Fixture::kH, Fixture::kHq)},
                  16);
        run_stage("g5-layers", f,
                  {lane(adapter("A0", 0), adapter("B0", 1), Q, 0, 8, 7, 2,
                        Fixture::kH, Fixture::kHq)},
                  16);
        run_stage("g6-d-in", f,
                  {lane(adapter("A0", 0), adapter("B0", 1), Q, 0, 8,
                        Fixture::kLayers, 2, 5, Fixture::kHq)},
                  16);
        run_stage("g7-d-out", f,
                  {lane(adapter("A0", 0), adapter("B0", 1), Q, 0, 8,
                        Fixture::kLayers, 2, Fixture::kH, 5)},
                  16);
        run_stage("g8-rank", f,
                  {lane(adapter("A0", 0), adapter("B0", 1), Q, 0, 8,
                        Fixture::kLayers, Fixture::kI + 1, Fixture::kH,
                        Fixture::kHq)},
                  16);
        run_stage("g9-span", f,
                  {lane(adapter("A0", 0), adapter("B0", 1), Q, 10, 8,
                        Fixture::kLayers, 2, Fixture::kH, Fixture::kHq)},
                  16);
    }

    return 0;
}
