// The score-capture oracle — gate-score-capture.
//
// Compiles the REAL `model/attn_score.cu` against the REAL
// `hook_sideband_arena.cpp` and drives both captures across their guard
// chain, their CSR arithmetic, and their arena traffic:
//
//   * the decode capture: ragged offsets from the page CSR (raw scaled by
//     heads, folded not), the slot carve (raw / folded / indptr, each
//     256-aligned), the every-acquire folded memset and CSR upload, the
//     fold-heads launch at publish, the payload, and slot stability across
//     layers of one fire;
//   * the prefill capture: the same, with the window factor in the raw rows,
//     publish WITHOUT a launch, and the two refusal ceilings (int32 element
//     offsets, the 1 GiB byte cap);
//   * the guard chain of both: null hooks, an uninterested program, an
//     uncapturable layer, zero heads, an unusable observation, an all-empty
//     fire, and the nested-capture refusal;
//   * `prepare_decode_score_capture`: the replay-refresh helper's plan — the
//     same layout arithmetic as the capture's own acquire, checked by
//     printing both symbolically;
//   * `default_attn_score_window`, swept by env across processes.
//
// Replaced surfaces: the CUDA entry points (recorders — offsets inside the
// arena's score slot are printed as `blk#K+N`, and the CSR upload prints its
// host CONTENT, because the rebasing arithmetic is the subject), the fold
// launch (a recorder), and a one-integer `KvCache`. The arena is real.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include "model/attn_score.hpp"
#include "model/attn_observation.hpp"
#include "model/hook_sideband_arena.hpp"
#include "model/stage_hooks.hpp"
#include "store/kv_cache.hpp"

using pie_cuda_driver::KvCache;
using pie_cuda_driver::model::AttentionObservation;
using pie_cuda_driver::model::AttentionScores;
using pie_cuda_driver::model::HookSidebandArena;
using pie_cuda_driver::model::LayerPrefillScoreCapture;
using pie_cuda_driver::model::LayerScoreCapture;
using pie_cuda_driver::model::StageHooks;

namespace {

constexpr char SEP = '\x1f';

std::string g_case;
std::map<const void*, std::pair<std::string, std::size_t>> g_regions;
int g_next_block = 0;

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

void name_fixed(const void* p, const char* name) {
    g_regions[p] = {name, 1};
}

std::string join_u32(const std::uint32_t* p, std::size_t n) {
    std::string s = "[";
    for (std::size_t i = 0; i < n; ++i) {
        if (i) s += ',';
        s += std::to_string(p[i]);
    }
    return s + "]";
}

}  // namespace

// ── the CUDA recorders ──────────────────────────────────────────────────────

cudaError_t cudaMalloc(void** ptr, std::size_t bytes) {
    *ptr = std::malloc(bytes == 0 ? 1 : bytes);
    const std::string name = "blk#" + std::to_string(g_next_block++);
    g_regions[*ptr] = {name, bytes == 0 ? 1 : bytes};
    note("malloc " + name + " bytes=" + std::to_string(bytes));
    return cudaSuccess;
}

cudaError_t cudaFree(void* ptr) {
    if (ptr != nullptr) {
        note("free " + where(ptr));
        g_regions.erase(ptr);
        std::free(ptr);
    }
    return cudaSuccess;
}

cudaError_t cudaStreamSynchronize(cudaStream_t) {
    note("sync");
    return cudaSuccess;
}

cudaError_t cudaMemsetAsync(void* ptr, int value, std::size_t bytes,
                            cudaStream_t) {
    note("memset " + where(ptr) + " val=" + std::to_string(value) +
         " len=" + std::to_string(bytes));
    return cudaSuccess;
}

cudaError_t cudaMemcpyAsync(void* dst, const void* src, std::size_t bytes,
                            cudaMemcpyKind kind, cudaStream_t) {
    // The CSR upload: print the host CONTENT, not the pointer — the ragged
    // offsets are the arithmetic under test.
    std::string body = "upload dst=" + where(dst) + " kind=" +
                       std::to_string(static_cast<int>(kind)) + " csr=[";
    const auto* v = static_cast<const std::int32_t*>(src);
    for (std::size_t i = 0; i < bytes / sizeof(std::int32_t); ++i) {
        if (i) body += ',';
        body += std::to_string(v[i]);
    }
    note(body + "]");
    return cudaSuccess;
}

namespace pie_cuda_driver::kernels::attn {

void attn_score_fold_heads(
    const float* scores,
    const std::int32_t* score_indptr_d,
    const std::uint32_t* kv_page_indptr_d,
    const std::uint32_t* kv_last_page_lens_d,
    int page_size,
    int num_requests,
    int num_q_heads,
    float* folded,
    cudaStream_t) {
    note("fold raw=" + where(scores) + " indptr=" + where(score_indptr_d) +
         " kvpp=" + where(kv_page_indptr_d) + " lens=" +
         where(kv_last_page_lens_d) + " psz=" + std::to_string(page_size) +
         " R=" + std::to_string(num_requests) + " qh=" +
         std::to_string(num_q_heads) + " out=" + where(folded));
}

}  // namespace pie_cuda_driver::kernels::attn

// ── the fire fixture ────────────────────────────────────────────────────────

namespace {

struct Fire {
    KvCache kv;
    std::vector<std::uint32_t> kvpp_h;
    std::vector<std::uint32_t> lens_h;
    std::vector<std::uint32_t> qo_h;
    AttentionObservation obs;
    StageHooks hooks;

    Fire(int page_size,
         std::vector<std::uint32_t> kvpp,
         std::vector<std::uint32_t> lens,
         HookSidebandArena* arena)
        : kv(page_size), kvpp_h(std::move(kvpp)), lens_h(std::move(lens)) {
        const int requests = static_cast<int>(kvpp_h.size()) - 1;
        qo_h.assign(static_cast<std::size_t>(requests) + 1, 0);
        for (int i = 0; i <= requests; ++i) {
            qo_h[static_cast<std::size_t>(i)] = static_cast<std::uint32_t>(i);
        }
        obs.kv = &kv;
        obs.kv_page_indices_d =
            reinterpret_cast<const std::uint32_t*>(0x10);
        obs.kv_page_indptr_d =
            reinterpret_cast<const std::uint32_t*>(0x20);
        obs.kv_last_page_lens_d =
            reinterpret_cast<const std::uint32_t*>(0x30);
        obs.qo_indptr_h = qo_h.data();
        obs.kv_page_indptr_h = kvpp_h.data();
        obs.kv_last_page_lens_h = lens_h.data();
        obs.num_requests = requests;
        obs.total_tokens = requests;
        hooks.wants_attn_score = true;
        hooks.observation = &obs;
        hooks.sideband_arena = arena;
    }
};

void payload_row(const char* label, const AttentionScores* s,
                 std::uint32_t expect_requests) {
    if (s == nullptr) {
        note(std::string(label) + " payload=null");
        return;
    }
    note(std::string(label) + " payload values=" + where(s->values) +
         " offsets=" + join_u32(s->offsets_h, s->num_requests + 1) +
         " R=" + std::to_string(s->num_requests) + " layer=" +
         std::to_string(s->layer) + " usable=" +
         std::to_string(s->usable() ? 1 : 0) + " expectR=" +
         std::to_string(expect_requests));
}

void begin_case(const char* name) {
    g_case = name;
    note("case-begin");
}

}  // namespace

int main(int argc, char** argv) {
    if (argc >= 2 && std::strcmp(argv[1], "window") == 0) {
        g_case = argc >= 3 ? argv[2] : "window";
        note("window=" +
             std::to_string(pie_cuda_driver::model::default_attn_score_window()));
        return 0;
    }

    name_fixed(reinterpret_cast<const void*>(0x20), "kvpp_d");
    name_fixed(reinterpret_cast<const void*>(0x30), "lens_d");

    HookSidebandArena arena;

    // a. The decode capture's happy path, twice on one fire: the second
    //    layer must land at the same slot base (arena stability), and the
    //    folded memset must run BOTH times.
    begin_case("a-decode");
    {
        Fire f(16, {0, 2, 5, 6}, {3, 16, 1}, &arena);
        for (std::uint32_t layer : {0u, 7u}) {
            LayerScoreCapture cap(&f.hooks, layer, 4, true, nullptr);
            note("L" + std::to_string(layer) + " active=" +
                 std::to_string(cap.active() ? 1 : 0) + " raw=" +
                 where(cap.raw()) + " indptr=" + where(cap.indptr_d()));
            payload_row("pre-publish", cap.scores(), 3);
            cap.publish(f.obs.kv_page_indptr_d, f.obs.kv_last_page_lens_d,
                        16);
            payload_row("published", cap.scores(), 3);
            // A second publish is a no-op — no second fold row.
            cap.publish(f.obs.kv_page_indptr_d, f.obs.kv_last_page_lens_d,
                        16);
        }
    }

    // b. The guard chain: each refusal leaves the capture inactive and the
    //    arena untouched.
    begin_case("b-guards");
    {
        Fire f(16, {0, 2}, {3}, &arena);
        LayerScoreCapture null_hooks(nullptr, 0, 4, true, nullptr);
        note("null-hooks active=" +
             std::to_string(null_hooks.active() ? 1 : 0));
        f.hooks.wants_attn_score = false;
        LayerScoreCapture unwanted(&f.hooks, 0, 4, true, nullptr);
        note("unwanted active=" + std::to_string(unwanted.active() ? 1 : 0));
        f.hooks.wants_attn_score = true;
        LayerScoreCapture windowed(&f.hooks, 0, 4, false, nullptr);
        note("uncapturable active=" +
             std::to_string(windowed.active() ? 1 : 0));
        LayerScoreCapture no_heads(&f.hooks, 0, 0, true, nullptr);
        note("zero-heads active=" + std::to_string(no_heads.active() ? 1 : 0));
        const AttentionObservation* saved = f.hooks.observation;
        f.hooks.observation = nullptr;
        LayerScoreCapture no_obs(&f.hooks, 0, 4, true, nullptr);
        note("no-obs active=" + std::to_string(no_obs.active() ? 1 : 0));
        f.hooks.observation = saved;
        Fire empty(16, {0, 0, 0}, {0, 0}, &arena);
        LayerScoreCapture zero(&empty.hooks, 0, 4, true, nullptr);
        note("all-empty active=" + std::to_string(zero.active() ? 1 : 0));
    }

    // c. Nested captures: the inner one stands down.
    begin_case("c-nested");
    {
        Fire f(16, {0, 2, 5}, {3, 16}, &arena);
        LayerScoreCapture outer(&f.hooks, 0, 2, true, nullptr);
        LayerScoreCapture inner(&f.hooks, 0, 2, true, nullptr);
        note("outer=" + std::to_string(outer.active() ? 1 : 0) +
             " inner=" + std::to_string(inner.active() ? 1 : 0));
    }

    // c2. What the depth guard PROTECTS: a refused inner capture with
    //     DIFFERENT geometry must not clobber the scratch the outer
    //     capture's published offsets point into. Without the guard the
    //     inner constructor would recompute the CSRs before discovering
    //     the arena busy, and the outer payload would silently describe
    //     the wrong fire.
    begin_case("c2-nested-clobber");
    {
        Fire fa(16, {0, 2, 5, 6}, {3, 16, 1}, &arena);
        Fire fb(16, {0, 9}, {4}, &arena);
        LayerScoreCapture outer(&fa.hooks, 0, 2, true, nullptr);
        {
            LayerScoreCapture inner(&fb.hooks, 0, 2, true, nullptr);
            note("inner=" + std::to_string(inner.active() ? 1 : 0));
        }
        outer.publish(fa.obs.kv_page_indptr_d, fa.obs.kv_last_page_lens_d,
                      16);
        payload_row("outer", outer.scores(), 3);
    }

    // d. Publish after the fire geometry is torn down: the C++ throws
    //    rather than folding against a stale view.
    begin_case("d-publish-lost-geometry");
    {
        Fire f(16, {0, 2, 5}, {3, 16}, &arena);
        LayerScoreCapture cap(&f.hooks, 3, 2, true, nullptr);
        f.hooks.observation = nullptr;
        try {
            cap.publish(f.obs.kv_page_indptr_d, f.obs.kv_last_page_lens_d,
                        16);
            note("no-throw");
        } catch (const std::exception&) {
            note("threw");
        }
    }

    // e. The u32 ceiling on raw elements: 9 heads over ~0.5G KV positions
    //    crosses 2^32 and the capture refuses.
    begin_case("e-u32-ceiling");
    {
        Fire f(256, {0u, 2000000u}, {256u}, &arena);
        LayerScoreCapture cap(&f.hooks, 0, 9, true, nullptr);
        note("active=" + std::to_string(cap.active() ? 1 : 0));
    }

    // f. `prepare_decode_score_capture`: the replay-refresh plan against
    //    the same fire as `a` — its folded/indptr addresses must be the
    //    ones the capture's own acquire carves.
    begin_case("f-prepare");
    {
        Fire f(16, {0, 2, 5, 6}, {3, 16, 1}, &arena);
        const auto plan = pie_cuda_driver::model::prepare_decode_score_capture(
            &arena, f.obs, 4, nullptr);
        note(std::string("plan ok=") + (plan.ok ? "1" : "0") + " folded=" +
             where(plan.folded) + " indptr=" + where(plan.indptr_d) +
             " csr_h=" +
             join_u32(reinterpret_cast<const std::uint32_t*>(
                          plan.indptr_h_data),
                      plan.num_requests + 1) +
             " folded_h=" + join_u32(plan.folded_offsets_h,
                                     plan.num_requests + 1) +
             " R=" + std::to_string(plan.num_requests));
        const auto refused = pie_cuda_driver::model::prepare_decode_score_capture(
            nullptr, f.obs, 4, nullptr);
        note(std::string("null-arena ok=") + (refused.ok ? "1" : "0"));
        const auto no_heads = pie_cuda_driver::model::prepare_decode_score_capture(
            &arena, f.obs, 0, nullptr);
        note(std::string("zero-heads ok=") + (no_heads.ok ? "1" : "0"));
    }

    // g. The prefill capture: the window factor in the raw CSR, publish
    //    without a launch, and the accessors.
    begin_case("g-prefill");
    {
        Fire f(16, {0, 2, 5, 6}, {3, 16, 1}, &arena);
        LayerPrefillScoreCapture cap(&f.hooks, 11, 4, 8, true, nullptr);
        note("active=" + std::to_string(cap.active() ? 1 : 0) + " raw=" +
             where(cap.raw()) + " folded=" + where(cap.folded()) +
             " indptr=" + where(cap.indptr_d()) + " window=" +
             std::to_string(cap.window()));
        payload_row("pre-publish", cap.scores(), 3);
        cap.publish();
        payload_row("published", cap.scores(), 3);
    }

    // h. The prefill ceilings: the int32 element bound and the 1 GiB byte
    //    cap, and the guard chain shared with decode.
    begin_case("h-prefill-ceilings");
    {
        // 4 heads * 64 window * ~0.13G positions ~= 2^35 raw elements.
        Fire big(256, {0u, 2000000u}, {256u}, &arena);
        LayerPrefillScoreCapture cap(&big.hooks, 0, 4, 64, true, nullptr);
        note("huge active=" + std::to_string(cap.active() ? 1 : 0));
        Fire f(16, {0, 2}, {3}, &arena);
        LayerPrefillScoreCapture no_window(&f.hooks, 0, 4, 0, true, nullptr);
        note("zero-window active=" +
             std::to_string(no_window.active() ? 1 : 0));
        LayerPrefillScoreCapture windowed(&f.hooks, 0, 4, 8, false, nullptr);
        note("uncapturable active=" +
             std::to_string(windowed.active() ? 1 : 0));
    }

    // i. Decode and prefill scratch are SEPARATE: a decode capture between
    //    two prefill captures must not clobber the prefill CSR contents.
    begin_case("i-scratch-separation");
    {
        Fire f(16, {0, 2, 5, 6}, {3, 16, 1}, &arena);
        LayerPrefillScoreCapture pf(&f.hooks, 0, 2, 4, true, nullptr);
        pf.publish();
        payload_row("prefill", pf.scores(), 3);
        {
            // Destroyed before the read-back below — depth guards, not
            // scratch, are what make the two coexist.
            LayerScoreCapture dec(&f.hooks, 1, 2, true, nullptr);
            note("decode active=" + std::to_string(dec.active() ? 1 : 0));
        }
        payload_row("prefill-after-decode", pf.scores(), 3);
    }

    return 0;
}
