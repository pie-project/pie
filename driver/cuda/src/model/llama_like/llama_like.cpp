#include "model/llama_like/llama_like.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <optional>
#include <stdexcept>

#include <cuda_runtime.h>

#include <string>
#include <vector>

#include "kernels/custom_all_reduce.hpp"
#include "cuda_check.hpp"
#include "kernels/add_bias.hpp"
#include "kernels/argmax.hpp"
#include "kernels/dtype_cast.hpp"
#include "kernels/embed.hpp"
#include "kernels/gather_rows.hpp"
#include "kernels/head_dim_pad.hpp"
#include "kernels/kv_paged.hpp"
#include "kernels/residual_add.hpp"
#include "kernels/rmsnorm.hpp"
#include "kernels/rope.hpp"
#include "kernels/split_packed.hpp"
#include "kernels/swiglu.hpp"
#include "model/qwen3_vl/qwen3_vl_vision_forward.hpp"
#include "model/attn_page_mask.hpp"
#include "model/attn_score.hpp"
#include "model/lora.hpp"
#include "model/stage_hooks.hpp"
#include "ops/attention_flashinfer.hpp"
#include "ops/gemm.hpp"

namespace pie_cuda_driver::model {

namespace {

// THE MIXED FIRE (M-3): the custom-mask suffix dispatch runs
// CONCURRENTLY with the prefix causal dispatch — disjoint attn_out row
// windows, read-only-shared q and KV (both dispatches run after the
// fire's KV write). One side stream + fork/join events per process:
// fires are launched sequentially from one thread, so the events'
// record/wait pairs are totally ordered and reuse is safe.
// PIE_SPATIAL_STREAM=0 serializes onto the main stream (bisection).
struct SpatialSideStream {
    cudaStream_t stream = nullptr;
    cudaEvent_t fork = nullptr;
    cudaEvent_t join = nullptr;
    // The THIRD lane (no-demotion 3-way): the plain-decode middle's
    // stream, forked/joined alongside the custom's — causal(main) ∥
    // decode(stream2) ∥ custom(stream).
    cudaStream_t stream2 = nullptr;
    cudaEvent_t fork2 = nullptr;
    cudaEvent_t join2 = nullptr;
    SpatialSideStream() {
        CUDA_CHECK(cudaStreamCreateWithFlags(
            &stream, cudaStreamNonBlocking));
        CUDA_CHECK(cudaEventCreateWithFlags(
            &fork, cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreateWithFlags(
            &join, cudaEventDisableTiming));
        CUDA_CHECK(cudaStreamCreateWithFlags(
            &stream2, cudaStreamNonBlocking));
        CUDA_CHECK(cudaEventCreateWithFlags(
            &fork2, cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreateWithFlags(
            &join2, cudaEventDisableTiming));
    }
};

// The mixed fire's SECOND prefill-family plan needs its own workspace:
// the prefix causal plan and the suffix custom plan are both
// PrefillPlan-family and write per-request scheduling metadata at the
// same int-workspace offsets — planning both into one AttentionWorkspace
// clobbers the first (measured: illegal access at the causal launch).
// The pure-decode split never collides (decode planner + prefill
// planner) and qwen2_5's split has a plan-free prefix. Lazy singleton:
// one worker thread launches fires sequentially, and plan+dispatch of
// one fire pair against the same instance.
inline AttentionWorkspace& spatial_suffix_ws() {
    static AttentionWorkspace ws = AttentionWorkspace::allocate();
    return ws;
}

// ④ Act 1 (banded depth): one dedicated workspace per band slot — the
// same same-family-planner isolation the suffix workspace exists for,
// per band. Lazy per slot; band count is capped at 3 (frame gate).
inline AttentionWorkspace& depth_band_ws(int i) {
    static std::array<std::unique_ptr<AttentionWorkspace>, 3> pool;
    auto& slot = pool[static_cast<std::size_t>(i)];
    if (!slot) {
        slot = std::make_unique<AttentionWorkspace>(
            AttentionWorkspace::allocate());
    }
    return *slot;
}

inline bool spatial_stream_enabled() {
    static const bool on = [] {
        const char* v = std::getenv("PIE_SPATIAL_STREAM");
        return v == nullptr || v[0] != '0';
    }();
    return on;
}

}  // namespace

// The interpreter's TU pairs the mixed tail dispatch against the same
// dedicated workspace the mixed prepare planned into.
AttentionWorkspace& spatial_suffix_attn_ws() { return spatial_suffix_ws(); }

// ④ Act 1: the interpreter's banded tail dispatch pairs against the
// same per-band workspaces the prepare planned into.
AttentionWorkspace& depth_band_attn_ws_public(int i) {
    return depth_band_ws(i);
}

namespace {

inline SpatialSideStream& spatial_side_stream() {
    static SpatialSideStream s;
    return s;
}

inline void maybe_add_bias(
    void* out, const DeviceTensor* bias_tensor,
    int N, int dim, cudaStream_t stream)
{
    if (bias_tensor == nullptr) return;
    kernels::launch_add_bias_bf16(out, bias_tensor->data(), N, dim, stream);
}

// Row `row` of a row-major bf16 buffer of width `width`. Used to hand a
// kernel the tail of a buffer whose prefix another kernel owns.
inline void* bf16_row(void* base, int row, int width)
{
    return static_cast<std::uint16_t*>(base) +
           static_cast<std::ptrdiff_t>(row) * width;
}

inline const void* bf16_row(const void* base, int row, int width)
{
    return static_cast<const std::uint16_t*>(base) +
           static_cast<std::ptrdiff_t>(row) * width;
}

bool decode_full_attention_variant_enabled() {
    static const bool enabled = [] {
        const char* v = std::getenv("PIE_CUDA_DECODE_FULL_ATTENTION");
        if (v == nullptr || v[0] == '\0') return true;
        return v[0] != '0';
    }();
    return enabled;
}

// PIE_LORA_GROUPED: same-shape lora lanes share one grouped-GEMM launch
// per correction GEMM instead of per-lane pairs. Default ON.
//
// This gate is a PARITY INSTRUMENT, not a permanent tuning knob: grouped
// GEMM changes the floating-point reduction order versus the per-lane
// pairs, so byte-identity between the two paths is not a meaningful bar
// at lane-count > 1 — the A/B this gate affords (same concurrent batch,
// grouped vs per-lane) is how a divergence gets classified (late/sporadic
// = reduction noise, immediate+total = bug; the tp_equivalence reasoning).
// Once the grouped path has soaked, the gate should be deleted, not kept.
bool lora_grouped_enabled() {
    static const bool enabled = [] {
        const char* v = std::getenv("PIE_LORA_GROUPED");
        if (v == nullptr || v[0] == '\0') return true;
        return v[0] != '0';
    }();
    return enabled;
}

// Per-fire lora state (§5.1 CORRECTION): validated lanes plus their adapter
// weights cast to bf16. The channel cells the resolver harvested are f32 —
// the PTIR channel vocabulary has no bf16 wire dtype — while the projection
// buffers the delta accumulates into are bf16, so each lane's A/B is cast
// once per fire here and the per-layer work is then two plain bf16 GEMMs.
// Re-cast every fire rather than cached: an adapter swap is a channel
// re-seed (never a re-trace), so the f32 cell contents may change under a
// cached cast without any signal the body could key invalidation on.
//
// The cast buffers are stream-ordered pool allocations (cudaMallocAsync /
// cudaFreeAsync — the `resolve_lane_attn_score` pattern) rather than
// Workspace members: rank, d_out, and lane count are program facts unknown
// when the Workspace is sized at load, and a lora fire never enters CUDA
// graph capture (`forward_graph_replay_eligible` excludes it), so body-time
// allocation is legal.
struct LoraFireState {
    struct Lane {
        const LoraLaneView* view = nullptr;
        void* a_bf16 = nullptr;  // [num_layers, R, d_in]
        void* b_bf16 = nullptr;  // [num_layers, d_out, R]
        // Grouped lowering (below): lane's element offset into the shared
        // xA^T scratch, and whether a group claimed it. Solo lanes keep
        // offset 0 — they reuse the scratch base sequentially as before.
        std::size_t xa_offset = 0;
        bool grouped = false;
    };
    // One same-shape group: lanes whose (rank, d_in, d_out, num_layers)
    // agree, lowered through ops::gemm_grouped_act_x_wt_bf16 (shared N/K
    // per call, per-lane M = token span). Only groups of size >= 2 are
    // kept; a group of 1 has nothing to share and stays a per-lane pair.
    struct Group {
        int rank = 0;
        int d_in = 0;
        int d_out = 0;
        std::vector<std::size_t> members;  // indices into `lanes`
        // Members carrying the q / v site (subset sizes for the second
        // grouped GEMM's per-site calls).
        int nq = 0;
        int nv = 0;
        // Per-member token spans, precomputed at STAGE time (campaign
        // step 2): the grouped calls read them per layer, and computing
        // them in apply() was the last per-layer host work.
        std::vector<int> m, mq, mv;
        // This group's slot offset (in pointer slots) within one layer's
        // slice of `ptr_slab`; layout below.
        std::size_t slab_off = 0;
    };
    std::vector<Group> groups;
    std::vector<Lane> lanes;
    // Device-resident pointer-array storage for the grouped calls,
    // [num_layers, slab_stride] of void*. cublasGemmGroupedBatchedEx does
    // NOT consume its A/B/C pointer arrays synchronously at call time:
    // transient host arrays handed to back-to-back grouped calls produced
    // illegal-address/misaligned faults on this platform (measured,
    // scratchpad grouped_repro*.cu — per-call stream syncs or
    // device-resident arrays both cure it). The nemotron_h MoE consumer
    // passes device-resident arrays for the same reason; each (layer,
    // group) here gets its own slot so nothing is ever overwritten while
    // a prior call may still read it.
    void* ptr_slab = nullptr;
    std::size_t slab_stride = 0;  // pointer slots per layer
    cudaStream_t stream = nullptr;

    LoraFireState(const LoraTable& table,
                  const HfConfig& cfg,
                  int N, int H, int Hq, int Hk, int I, int T,
                  cudaStream_t s,
                  LoraStageArena& arena,
                  const void* qkv_in,
                  void* q_out,
                  void* v_out,
                  void* xa_scratch)
        : stream(s)
    {
        // Campaign step 1: every buffer below comes from the per-fire
        // bump arena — no body-time cudaMallocAsync, nothing to free.
        arena.reset();
        if (T != 1) {
            // The site widths below (Hq/Hk) are the UNSHARDED projection
            // widths B was traced against; a TP rank holds only its slice.
            // The capability gate keeps lora off TP configs — this is the
            // per-fire restatement.
            throw std::runtime_error(
                "lora is not supported under tensor parallelism");
        }
        lanes.reserve(table.count);
        for (std::uint32_t i = 0; i < table.count; ++i) {
            const LoraLaneView& lane = table.lanes[i];
            const bool scale_lane =
                lane.form == LoraLaneView::Form::Scale;
            if (lane.a == nullptr || (!scale_lane && lane.b == nullptr)) {
                throw std::runtime_error(
                    "lora lane carries a null adapter address");
            }
            if (lane.sites_bits == 0) {
                throw std::runtime_error("lora lane names no site");
            }
            if ((lane.sites_bits & ~kLoraSitesKnown) != 0) {
                throw std::runtime_error(
                    "lora SITES names bits outside the site vocabulary "
                    "(bits " + std::to_string(lane.sites_bits) + ")");
            }
            if ((lane.sites_bits & ~kLoraSitesConsumed) != 0) {
                // Refused loudly at first use, per the vocabulary contract
                // in model/lora.hpp: v0 consumes q and v only.
                throw std::runtime_error(
                    "lora site not implemented by this forward (v0 applies "
                    "q and v only; SITES bits " +
                    std::to_string(lane.sites_bits) + ")");
            }
            if (lane.num_layers != static_cast<std::uint32_t>(
                                       cfg.num_hidden_layers)) {
                throw std::runtime_error(
                    "lora adapter declares " +
                    std::to_string(lane.num_layers) + " layers, model has " +
                    std::to_string(cfg.num_hidden_layers));
            }
            if (!scale_lane &&
                lane.d_in != static_cast<std::uint32_t>(H)) {
                throw std::runtime_error(
                    "lora adapter d_in " + std::to_string(lane.d_in) +
                    " != hidden size " + std::to_string(H));
            }
            auto require_width = [&](std::uint64_t bit, int width,
                                     const char* site) {
                if ((lane.sites_bits & bit) != 0 &&
                    lane.d_out != static_cast<std::uint32_t>(width)) {
                    throw std::runtime_error(
                        std::string("lora adapter d_out ") +
                        std::to_string(lane.d_out) + " != " + site +
                        " projection width " + std::to_string(width));
                }
            };
            require_width(kLoraSiteQ, Hq, "q");
            require_width(kLoraSiteV, Hk, "v");
            if (scale_lane) {
                // The SCALE form (IA3): stage the l vector's bf16 cast;
                // no rank, no scratch, no grouping — apply() multiplies
                // the span rows elementwise per consumed site.
                if (lane.token_start > static_cast<std::uint32_t>(N) ||
                    lane.token_count >
                        static_cast<std::uint32_t>(N) - lane.token_start) {
                    throw std::runtime_error(
                        "lora scale lane token span exceeds the fire");
                }
                if (lane.token_count == 0) continue;
                const std::size_t l_elems = static_cast<std::size_t>(
                    lane.num_layers) * lane.d_out;
                Lane out{&lane, nullptr, nullptr};
                out.a_bf16 = arena.alloc(l_elems * 2);
                lanes.push_back(out);
                kernels::launch_cast_fp32_to_bf16(
                    lane.a, out.a_bf16, l_elems, stream);
                continue;
            }
            if (lane.rank == 0 ||
                lane.rank > static_cast<std::uint32_t>(I)) {
                // The xA^T scratch below aliases ws.gate ([max_tokens, I]),
                // so a rank above the MLP intermediate width would overrun
                // it. Any real adapter rank is orders of magnitude smaller.
                throw std::runtime_error(
                    "lora rank " + std::to_string(lane.rank) +
                    " is zero or exceeds the scratch width " +
                    std::to_string(I));
            }
            if (lane.token_start > static_cast<std::uint32_t>(N) ||
                lane.token_count >
                    static_cast<std::uint32_t>(N) - lane.token_start) {
                throw std::runtime_error(
                    "lora lane token span [" +
                    std::to_string(lane.token_start) + ", +" +
                    std::to_string(lane.token_count) +
                    ") exceeds the fire's " + std::to_string(N) + " rows");
            }
            if (lane.token_count == 0) continue;

            const std::size_t a_elems = static_cast<std::size_t>(
                lane.num_layers) * lane.rank * lane.d_in;
            const std::size_t b_elems = static_cast<std::size_t>(
                lane.num_layers) * lane.d_out * lane.rank;
            Lane out{&lane, nullptr, nullptr};
            out.a_bf16 = arena.alloc(a_elems * 2);
            out.b_bf16 = arena.alloc(b_elems * 2);
            lanes.push_back(out);
            kernels::launch_cast_fp32_to_bf16(
                lane.a, out.a_bf16, a_elems, stream);
            kernels::launch_cast_fp32_to_bf16(
                lane.b, out.b_bf16, b_elems, stream);
        }

        // ── Same-shape lane grouping (Stage 5's consumer increment) ──
        //
        // The planner story, honestly: `fire_plan.rs` marks the
        // projection_weights site PerLane and stops there — the
        // classification is device-independent (pie-application-plan.md
        // §4.4). Whether same-shape lanes then SHARE one grouped-GEMM
        // launch is device knowledge: stage0-l40s.md §3.1 measured
        // batched matching the hand-written padded kernel at 1.00–1.03×
        // and beating separate launches by up to 24.75× exactly when
        // shapes share, and only the driver knows this device offers a
        // kernel for it (cublasGemmGroupedBatchedEx). Same argument as
        // fast_rows' dual derivation (stage1-notes "Stage 5"): the
        // planner owns the order and the class, the driver applies
        // device-side knowledge to a device-shaped fact. Handing a
        // "Grouped" lowering across the ABI would force the scheduler to
        // model a kernel table it cannot see.
        //
        // The grouping key is the GEMM-shape tuple (rank, d_in, d_out):
        // both correction GEMMs share N/K within a group (xA^T: N=rank,
        // K=d_in; (xA^T)B^T: N=d_out, K=rank). Layer-slice compatibility
        // (num_layers, the per-layer weight stride) is NOT in the key
        // because the lane loop above already validated every lane to
        // cfg.num_hidden_layers — uniform by construction, unlike d_in,
        // which is also validated uniform today but stays in the key as
        // the statement of what the grouped calls actually require.
        //
        // Precondition: all lanes' token spans pairwise disjoint. One
        // grouped call runs its lanes' beta=1 accumulations concurrently,
        // so overlapping projection rows would race — the per-lane pairs
        // are stream-ordered and tolerate overlap. Lanes are distinct
        // programs over distinct requests, so overlap is malformed
        // geometry; if it ever appears, fall back to per-lane (which
        // stays correct) rather than refuse the fire.
        if (lora_grouped_enabled() && lanes.size() >= 2 &&
            lane_spans_disjoint()) {
            for (std::size_t i = 0; i < lanes.size(); ++i) {
                const LoraLaneView& v = *lanes[i].view;
                if (v.form == LoraLaneView::Form::Scale) {
                    // Scale lanes are elementwise — nothing to group.
                    continue;
                }
                Group* g = nullptr;
                for (Group& cand : groups) {
                    if (cand.rank == static_cast<int>(v.rank) &&
                        cand.d_in == static_cast<int>(v.d_in) &&
                        cand.d_out == static_cast<int>(v.d_out)) {
                        g = &cand;
                        break;
                    }
                }
                if (g == nullptr) {
                    groups.push_back(Group{
                        static_cast<int>(v.rank), static_cast<int>(v.d_in),
                        static_cast<int>(v.d_out), {}});
                    g = &groups.back();
                }
                g->members.push_back(i);
            }
            // Groups of 1 keep the existing per-lane pair.
            groups.erase(
                std::remove_if(groups.begin(), groups.end(),
                               [](const Group& g) {
                                   return g.members.size() < 2;
                               }),
                groups.end());
            // Scratch layout for the grouped xA^T intermediate: grouped
            // lanes get exclusive [t, R] regions packed contiguously
            // (per-lane element offsets), because one grouped call writes
            // them all concurrently. Bound: disjoint spans give
            // sum(t_i) <= N, and rank <= I was validated per lane, so
            // sum(t_i * R_i) <= N * I <= the ws.gate alias's
            // [max_tokens, I] extent. The check restates the chunk-3
            // bound verification for the new layout; it is unreachable
            // given the disjointness precondition, but the scratch is an
            // alias, so an overrun would corrupt live state silently.
            std::size_t xa_total = 0;
            for (const Group& g : groups) {
                for (std::size_t idx : g.members) {
                    lanes[idx].grouped = true;
                    lanes[idx].xa_offset = xa_total;
                    xa_total += static_cast<std::size_t>(
                                    lanes[idx].view->token_count) *
                                lanes[idx].view->rank;
                }
            }
            if (xa_total > static_cast<std::size_t>(N) *
                               static_cast<std::size_t>(I)) {
                throw std::runtime_error(
                    "lora grouped xA^T scratch layout (" +
                    std::to_string(xa_total) + " elems) exceeds the " +
                    std::to_string(N) + "x" + std::to_string(I) +
                    " ws.gate alias bound");
            }
            // Pointer-slab layout (see the `ptr_slab` member note for why
            // the arrays must be device-resident): per layer, per group,
            // consecutive slot runs [x(n) a(n) xa(n)][q_act q_w q_y](nq
            // each)[v_act v_w v_y](nv each), n = group size.
            for (Group& g : groups) {
                for (std::size_t idx : g.members) {
                    const std::uint64_t bits = lanes[idx].view->sites_bits;
                    if ((bits & kLoraSiteQ) != 0) ++g.nq;
                    if ((bits & kLoraSiteV) != 0) ++g.nv;
                }
                g.slab_off = slab_stride;
                slab_stride += 3 * g.members.size() +
                               3 * static_cast<std::size_t>(g.nq) +
                               3 * static_cast<std::size_t>(g.nv);
            }
            if (slab_stride > 0) {
                ptr_slab = arena.alloc(
                    static_cast<std::size_t>(cfg.num_hidden_layers) *
                    slab_stride * sizeof(void*));
                // ── The STAGE phase proper (campaign step 2): the whole
                // slab — every layer, every group — is computed here and
                // uploaded ONCE. Everything a slot holds is a fire
                // constant: arena addresses, layer-strided adapter
                // slices, and the ws buffer rows the caller passed in.
                // apply() is left with launches only, which is what a
                // captured body requires.
                std::vector<const void*> slab_host;
                slab_host.resize(
                    static_cast<std::size_t>(cfg.num_hidden_layers) *
                    slab_stride, nullptr);
                for (int layer = 0;
                     layer < cfg.num_hidden_layers; ++layer) {
                    for (Group& g : groups) {
                        std::vector<const void*> staged;
                        std::vector<const void*> a_run, xa_run;
                        std::vector<const void*> q_act, q_w, q_y;
                        std::vector<const void*> v_act, v_w, v_y;
                        if (layer == 0) {
                            g.m.clear(); g.mq.clear(); g.mv.clear();
                        }
                        for (std::size_t idx : g.members) {
                            const Lane& lane = lanes[idx];
                            const LoraLaneView& v = *lane.view;
                            const auto* a_l = static_cast<
                                const std::uint16_t*>(lane.a_bf16) +
                                static_cast<std::size_t>(layer) *
                                    g.rank * g.d_in;
                            const auto* b_l = static_cast<
                                const std::uint16_t*>(lane.b_bf16) +
                                static_cast<std::size_t>(layer) *
                                    g.d_out * g.rank;
                            void* xa = static_cast<std::uint16_t*>(
                                           xa_scratch) + lane.xa_offset;
                            staged.push_back(bf16_row(
                                qkv_in,
                                static_cast<int>(v.token_start), H));
                            a_run.push_back(a_l);
                            xa_run.push_back(xa);
                            if (layer == 0) {
                                g.m.push_back(
                                    static_cast<int>(v.token_count));
                            }
                            if ((v.sites_bits & kLoraSiteQ) != 0) {
                                q_act.push_back(xa);
                                q_w.push_back(b_l);
                                q_y.push_back(bf16_row(
                                    q_out,
                                    static_cast<int>(v.token_start),
                                    Hq));
                                if (layer == 0) {
                                    g.mq.push_back(static_cast<int>(
                                        v.token_count));
                                }
                            }
                            if ((v.sites_bits & kLoraSiteV) != 0) {
                                v_act.push_back(xa);
                                v_w.push_back(b_l);
                                v_y.push_back(bf16_row(
                                    v_out,
                                    static_cast<int>(v.token_start),
                                    Hk));
                                if (layer == 0) {
                                    g.mv.push_back(static_cast<int>(
                                        v.token_count));
                                }
                            }
                        }
                        const void** slot = slab_host.data() +
                            static_cast<std::size_t>(layer) *
                                slab_stride + g.slab_off;
                        auto put = [&slot](
                            const std::vector<const void*>& run) {
                            for (const void* ptr : run) *slot++ = ptr;
                        };
                        put(staged); put(a_run); put(xa_run);
                        put(q_act); put(q_w); put(q_y);
                        put(v_act); put(v_w); put(v_y);
                    }
                }
                CUDA_CHECK(cudaMemcpyAsync(
                    ptr_slab, slab_host.data(),
                    slab_host.size() * sizeof(void*),
                    cudaMemcpyHostToDevice, stream));
            }
        }
    }

    // True iff no two lanes' token spans overlap (lanes with empty spans
    // were dropped at construction). Precondition for the grouped lowering.
    bool lane_spans_disjoint() const {
        std::vector<const LoraLaneView*> by_start;
        by_start.reserve(lanes.size());
        for (const Lane& lane : lanes) by_start.push_back(lane.view);
        std::sort(by_start.begin(), by_start.end(),
                  [](const LoraLaneView* a, const LoraLaneView* b) {
                      return a->token_start < b->token_start;
                  });
        for (std::size_t i = 1; i < by_start.size(); ++i) {
            if (by_start[i - 1]->token_start + by_start[i - 1]->token_count >
                by_start[i]->token_start) {
                return false;
            }
        }
        return true;
    }

    // One-line grouping summary for PIE_LORA_FIRE_TRACE, e.g. "3xr8",
    // "2xr8+1solo", "none(2 solo)", "off". Verification instrument: a
    // single-lane fire must read "none(1 solo)" — the grouped path did
    // not engage.
    std::string grouping_desc() const {
        if (!lora_grouped_enabled()) return "off";
        std::size_t solo = 0;
        for (const Lane& lane : lanes) {
            if (!lane.grouped) ++solo;
        }
        if (groups.empty()) {
            return "none(" + std::to_string(solo) + " solo)";
        }
        std::string s;
        for (const Group& g : groups) {
            s += (s.empty() ? "" : ",") +
                 std::to_string(g.members.size()) + "xr" +
                 std::to_string(g.rank);
        }
        if (solo > 0) s += "+" + std::to_string(solo) + "solo";
        return s;
    }

    LoraFireState(const LoraFireState&) = delete;
    LoraFireState& operator=(const LoraFireState&) = delete;

    // Arena-backed (campaign step 1): nothing to free — the next fire's
    // reset reclaims the space, stream-ordered behind this fire's reads.
    ~LoraFireState() = default;

    // The CORRECTION at layer L (§5.1): `x(W+BA)^T = xW^T + (xA^T)B^T`.
    // Called immediately after the base q/v projections materialize in the
    // ws buffers, before bias/qk-norm/rope and before the KV append — the
    // delta lands on the projection output, exactly where `W + BA` would
    // have put it.
    //
    // Two lowerings of the same math, chosen at fire setup (see the
    // grouping block in the constructor for why the choice lives HERE and
    // not in fire_plan):
    //   * solo lanes — one [t, R] GEMM into scratch, then one beta=1 GEMM
    //     per named site into that lane's token rows, stream-ordered;
    //   * same-shape groups — both GEMMs go through
    //     ops::gemm_grouped_act_x_wt_bf16 (shared N/K, per-lane M;
    //     pointer arrays staged into the device-resident `ptr_slab` —
    //     see its member note), one launch per correction GEMM per group.
    // Ranks may still differ ACROSS groups (different rank = different
    // traced program, co-batched); §3.1 measured bucketing losing to
    // padding, so nothing here pads a lane up to a foreign rank — lanes
    // group only when the shapes already agree.
    //
    // NOTE the grouped path changes the floating-point reduction order vs
    // the per-lane pairs (different cuBLAS kernel), so byte-identity
    // between the two is not expected at group size >= 2; PIE_LORA_GROUPED
    // is the A/B instrument for that comparison.
    void apply(cublasHandle_t handle,
               int layer,
               const void* qkv_in, int H, int Hq, int Hk,
               void* q_out, void* v_out,
               void* xa_scratch) const
    {
        for (const Lane& lane : lanes) {
            if (lane.grouped) continue;
            const LoraLaneView& v = *lane.view;
            const int t = static_cast<int>(v.token_count);
            if (v.form == LoraLaneView::Form::Scale) {
                // Applied in the scale pass BELOW — after every delta
                // (solo and grouped) has landed, so a same-site
                // low-rank + scale composes as s ⊙ (y + B(Ax)) — DoRA.
                continue;
            }
            const int R = static_cast<int>(v.rank);
            const auto* a_l = static_cast<const std::uint16_t*>(lane.a_bf16) +
                static_cast<std::size_t>(layer) * R * v.d_in;
            const auto* b_l = static_cast<const std::uint16_t*>(lane.b_bf16) +
                static_cast<std::size_t>(layer) * v.d_out * R;
            const void* x = bf16_row(
                qkv_in, static_cast<int>(v.token_start), H);
            // xA^T -> scratch [t, R]. Solo lanes reuse the same scratch
            // base: the stream orders each lane's pair before the next
            // lane's overwrite (and before any group's grouped write).
            ops::gemm_act_x_wt_bf16(
                handle, x, a_l, xa_scratch, t, R, H);
            if ((v.sites_bits & kLoraSiteQ) != 0) {
                ops::gemm_act_x_wt_bf16(
                    handle, xa_scratch, b_l,
                    bf16_row(q_out, static_cast<int>(v.token_start), Hq),
                    t, static_cast<int>(v.d_out), R, /*beta=*/1.f);
            }
            if ((v.sites_bits & kLoraSiteV) != 0) {
                ops::gemm_act_x_wt_bf16(
                    handle, xa_scratch, b_l,
                    bf16_row(v_out, static_cast<int>(v.token_start), Hk),
                    t, static_cast<int>(v.d_out), R, /*beta=*/1.f);
            }
        }
        for (const Group& g : groups) {
            const std::size_t n = g.members.size();
            // Campaign step 2: the slab was fully staged at fire setup —
            // this is slot arithmetic and launches, nothing else (what a
            // captured body requires). The per-member M arrays were
            // precomputed there too.
            void** slot = static_cast<void**>(ptr_slab) +
                          static_cast<std::size_t>(layer) * slab_stride +
                          g.slab_off;
            const void* const* x_ptrs = slot;
            const void* const* a_ptrs = x_ptrs + n;
            const auto* xa_ptrs = x_ptrs + 2 * n;
            ops::gemm_grouped_act_x_wt_bf16(
                handle, x_ptrs, a_ptrs,
                const_cast<void* const*>(xa_ptrs),
                g.m.data(), static_cast<int>(n), g.rank, g.d_in);
            // (xA^T)B^T per site subset: shared N=d_out, K=rank, beta=1.
            // A lane contributes to the q call, the v call, or both, per
            // its SITES bits; d_out is shared across the group by the
            // grouping key, so the subsets keep the shared-N contract.
            // Per-site M arrays: the site subsets preserve member order,
            // so the span list is the same prefix-free filter of `m`.
            if (g.nq > 0) {
                const auto* base = x_ptrs + 3 * n;
                ops::gemm_grouped_act_x_wt_bf16(
                    handle, base, base + g.nq,
                    const_cast<void* const*>(base + 2 * g.nq),
                    g.mq.data(), g.nq, g.d_out, g.rank, /*beta=*/1.f);
            }
            if (g.nv > 0) {
                const auto* base = x_ptrs + 3 * n + 3 * g.nq;
                ops::gemm_grouped_act_x_wt_bf16(
                    handle, base, base + g.nv,
                    const_cast<void* const*>(base + 2 * g.nv),
                    g.mv.data(), g.nv, g.d_out, g.rank, /*beta=*/1.f);
            }
        }
        // ── The scale pass: AFTER every delta (solo + grouped), so a
        // same-site low-rank + scale composes as s ⊙ (y + B(Ax)) —
        // DoRA's order; a lone scale lane is IA3 unchanged. ──
        for (const Lane& lane : lanes) {
            const LoraLaneView& v = *lane.view;
            if (v.form != LoraLaneView::Form::Scale) continue;
            const int t = static_cast<int>(v.token_count);
            const auto* l_l =
                static_cast<const std::uint16_t*>(lane.a_bf16) +
                static_cast<std::size_t>(layer) * v.d_out;
            if ((v.sites_bits & kLoraSiteQ) != 0) {
                kernels::launch_scale_rows_bf16(
                    bf16_row(q_out,
                             static_cast<int>(v.token_start), Hq),
                    l_l, t, static_cast<int>(v.d_out), stream);
            }
            if ((v.sites_bits & kLoraSiteV) != 0) {
                kernels::launch_scale_rows_bf16(
                    bf16_row(v_out,
                             static_cast<int>(v.token_start), Hk),
                    l_l, t, static_cast<int>(v.d_out), stream);
            }
        }
    }
};

inline void apply_rope(
    const LlamaLikeForwardCfg& fwd_cfg,
    const HfConfig& cfg,
    void* q, void* k,
    const std::int32_t* positions,
    int N, int num_q_heads, int num_kv_heads, int head_dim,
    cudaStream_t stream)
{
    if (fwd_cfg.rope_kind == RopeKind::YaRN) {
        kernels::launch_rope_yarn_bf16(
            q, k, positions,
            N, num_q_heads, num_kv_heads, head_dim,
            cfg.rope_theta,
            fwd_cfg.yarn_factor,
            fwd_cfg.yarn_low_freq_factor,
            fwd_cfg.yarn_high_freq_factor,
            fwd_cfg.yarn_original_max_position,
            stream);
    } else if (fwd_cfg.rope_kind == RopeKind::YaRNOriginal) {
        kernels::launch_rope_yarn_original_bf16(
            q, k, positions,
            N, num_q_heads, num_kv_heads, head_dim,
            cfg.rope_theta,
            fwd_cfg.yarn_factor,
            fwd_cfg.yarn_beta_fast,
            fwd_cfg.yarn_beta_slow,
            fwd_cfg.yarn_attention_factor,
            fwd_cfg.yarn_original_max_position,
            stream);
    } else {
        kernels::launch_rope_bf16(
            q, k, positions,
            N, num_q_heads, num_kv_heads, head_dim,
            cfg.rope_theta, stream);
    }
}

}  // namespace

// Bug#2 A/B: the fused decode QKV+qk-norm+rope+KV-write kernel
// (`launch_qkv_decode_qk_norm_rope_write_kv_bf16`) is the R>1 concurrent-decode
// suspect (the standalone BatchDecode attention is proven per-request-correct,
// so the corruption is upstream in KV/Q production). PIE_CUDA_DECODE_FUSED_POST=0
// falls back to the non-fused split-qkv + separate rope + `write_kv_to_pages`
// (the verified `resolve_dst` path). If the fleet goes 8/8 with it off, the
// fused kernel is the bug. Default on.
//
// External linkage (declared in llama_like.hpp) because the declared
// executor's fused decode-QKV peephole (declared_forward.cpp) must read the
// SAME gate: the peephole fires exactly when this branch would.
bool decode_fused_post_enabled() {
    static const bool enabled = [] {
        const char* v = std::getenv("PIE_CUDA_DECODE_FUSED_POST");
        if (v == nullptr || v[0] == '\0') return true;
        return v[0] != '0';
    }();
    return enabled;
}

// The kvpp SENTRY (the probabilistic composed-R=32 fault, 2026-08-04:
// flashinfer's PrefillSplitQOKVIndptr asserting on kv_indptr garbage —
// e.g. -1160232626 at entry 13 — in the first-boot window, twice
// sighted, never under trace). Host plan inputs are validated HERE,
// before any planner consumes them, and a violation dumps the arrays
// UNCONDITIONALLY and refuses the fire cleanly — the heisenbug becomes
// a self-documenting event at its next occurrence instead of a deep
// planner assert. Cost: one R-length scan per prepare.
static void kvpp_sentry(
    const char* what,
    const std::uint32_t* qo_indptr_h,
    const std::uint32_t* kv_page_indptr_h,
    int num_requests)
{
    for (int r = 0; r < num_requests; ++r) {
        const bool qo_bad =
            qo_indptr_h != nullptr && qo_indptr_h[r + 1] < qo_indptr_h[r];
        const bool kv_bad = kv_page_indptr_h != nullptr &&
                            kv_page_indptr_h[r + 1] < kv_page_indptr_h[r];
        if (!qo_bad && !kv_bad) continue;
        std::fprintf(stderr,
                     "[kvpp-sentry] %s: NON-MONOTONE host plan input at "
                     "lane %d of %d\n",
                     what, r, num_requests);
        for (int i = 0; i <= num_requests; ++i) {
            std::fprintf(
                stderr, "[kvpp-sentry]   [%d] qo=%d kvpp=%d\n", i,
                qo_indptr_h != nullptr
                    ? static_cast<std::int32_t>(qo_indptr_h[i])
                    : -1,
                kv_page_indptr_h != nullptr
                    ? static_cast<std::int32_t>(kv_page_indptr_h[i])
                    : -1);
        }
        throw std::runtime_error(
            std::string("kvpp sentry: non-monotone host plan input (") +
            what + ") — the composed-placeholder fault; arrays dumped");
    }
}

void prepare_llama_like_decode_plan(
    LlamaLikePlanState& state,
    AttentionWorkspace& attn_ws,
    KvCache& cache,
    const HfConfig& cfg,
    const LlamaLikeForwardCfg& fwd_cfg,
    const std::uint32_t* qo_indptr_h,
    const std::uint32_t* kv_page_indices_d,
    const std::uint32_t* kv_page_indptr_h,
    const std::uint32_t* kv_page_indptr_d,
    const std::uint32_t* kv_last_page_lens_h,
    const std::uint32_t* kv_last_page_lens_d,
    int total_tokens,
    int num_requests,
    bool is_pure_decode,
    bool have_custom_mask,
    std::uint32_t attn_score_window,
    std::uint32_t unmasked_prefix_rows,
    const std::uint32_t* mask_suffix_page_counts_h,
    const std::uint32_t* mask_suffix_last_lens_h,
    std::uint32_t full_depth_rows,
    const std::uint32_t* depth_band_k,
    const std::uint32_t* depth_band_rows,
    std::uint32_t depth_band_count)
{
    // The prepare hook runs OUTSIDE any cuStreamCapture region. It updates
    // pinned/device buffers in `attn_ws` that the captured body reads via
    // cudaMemcpyAsync at replay time, so the same captured graph stays
    // correct across fires with different KV lengths.
    state.use_xqa_decode = false;
    state.xqa_max_pages_per_seq = 0;
    state.use_prefill_plan = false;
    state.use_prefill_decode_plan = false;
    state.use_mask_decode_plan = false;
    state.prefill_score_window = 0;
    state.spatial_mask_split = -1;
    state.spatial_mask_row_split = -1;
    // NS-2 (the spatial mask fire, PIE_SPATIAL_MASK): a masked pure-decode
    // fire with a planned 0 < unmasked prefix < R builds BOTH plans — the
    // decode side over the prefix (a recursive prepare with the same host
    // CSR arrays truncated to `split`), then the mask plan over the
    // REBASED suffix. The gate conditions mirror run_forward_dispatch's
    // exactly (the engine plans UNPLANNED for hooked/lora fires, so those
    // never reach here with a split).
    static const bool spatial_mask_on = [] {
        const char* v = std::getenv("PIE_SPATIAL_MASK");
        return v == nullptr || v[0] != '0';
    }();
    if (spatial_mask_on && have_custom_mask && is_pure_decode &&
        unmasked_prefix_rows != 0xffffffffu &&
        unmasked_prefix_rows < static_cast<std::uint32_t>(num_requests) &&
        // Padded head dims (phi3) keep the fire-level arm: the split's
        // row offsets are logical-width, the padded staging is not.
        cfg.head_dim == cfg.head_dim_kernel &&
        // The XQA prefix is not wired (its fire-wide prepare is R-shaped);
        // deployments that would pick XQA keep the fire-level mask arm.
        !(fwd_cfg.use_xqa_decode && cache.format().is_native_bf16() &&
          !cache.hnd_layout())) {
        const int split = static_cast<int>(unmasked_prefix_rows);
        const int rs = num_requests - split;
        // Prefix decode plans (whichever this deployment picks), over the
        // same arrays at R' = split. Pure decode: total_tokens' == split.
        // split == 0 (all-masked composed fire) has no prefix and builds
        // no decode plan.
        if (split > 0) {
            prepare_llama_like_decode_plan(
                state, attn_ws, cache, cfg, fwd_cfg,
                qo_indptr_h, kv_page_indices_d, kv_page_indptr_h,
                kv_page_indptr_d, kv_last_page_lens_h, kv_last_page_lens_d,
                split, split, /*is_pure_decode=*/true,
                /*have_custom_mask=*/false, attn_score_window);
        }
        // The recursion reset the state flags; restore the split AFTER it.
        std::vector<std::uint32_t> qo_suffix(
            static_cast<std::size_t>(rs) + 1);
        std::vector<std::uint32_t> kvpp_suffix(
            static_cast<std::size_t>(rs) + 1);
        // Composed-envelope fires: the host wire CSRs are placeholders;
        // the RESOLVER's per-suffix-lane geometry (threaded from the
        // frame's spatial pack) is the planning truth. Wire-composed
        // fires fall back to the host CSR slices.
        for (int i = 0; i <= rs; ++i) {
            qo_suffix[static_cast<std::size_t>(i)] =
                static_cast<std::uint32_t>(i);
        }
        if (mask_suffix_page_counts_h != nullptr) {
            kvpp_suffix[0] = 0;
            for (int i = 0; i < rs; ++i) {
                kvpp_suffix[static_cast<std::size_t>(i) + 1] =
                    kvpp_suffix[static_cast<std::size_t>(i)] +
                    mask_suffix_page_counts_h[i];
            }
        } else {
            const std::uint32_t page_base = kv_page_indptr_h[split];
            for (int i = 0; i <= rs; ++i) {
                kvpp_suffix[static_cast<std::size_t>(i)] =
                    kv_page_indptr_h[split + i] - page_base;
            }
        }
        if (!state.mask_decode_plan) {
            state.mask_decode_plan = ops::make_prefill_plan();
        }
        if (std::getenv("PIE_KVPP_TRACE") != nullptr) {
            std::fprintf(stderr,
                         "[kvpp-sfx] R=%d split=%d rs=%d counts=%d sfx=[",
                         num_requests, split, rs,
                         mask_suffix_page_counts_h != nullptr ? 1 : 0);
            for (int i = 0; i <= rs; ++i)
                std::fprintf(stderr, "%u,", kvpp_suffix[i]);
            std::fprintf(stderr, "] host_tail=[");
            for (int i = split; i <= num_requests; ++i)
                std::fprintf(stderr, "%u,", kv_page_indptr_h[i]);
            std::fprintf(stderr, "]\n");
        }
        const int T = (fwd_cfg.tp_size > 0) ? fwd_cfg.tp_size : 1;
        ops::plan_attention_flashinfer_prefill_bf16(
            *state.mask_decode_plan,
            qo_suffix.data(),
            kvpp_suffix.data(),
            mask_suffix_last_lens_h != nullptr
                ? mask_suffix_last_lens_h
                : kv_last_page_lens_h + split,
            rs,
            rs,
            cfg.num_attention_heads / T,
            cfg.num_key_value_heads / T,
            cfg.head_dim_kernel,
            cache.page_size(),
            // The dedicated suffix workspace (the two-plans lesson) —
            // ALSO the concurrency precondition: the suffix custom
            // dispatch now overlaps the prefix decode on the side
            // stream, so their scratch must be disjoint.
            spatial_suffix_ws(),
            /*stream=*/nullptr,
            fwd_cfg.decode_plan_cuda_graph,
            /*window_left=*/-1,
            /*full_attention_variant=*/false,
            cache.hnd_layout(),
            /*causal_mask=*/false,
            /*custom_mask=*/true);
        state.use_mask_decode_plan = true;
        state.spatial_mask_split = split;
        state.spatial_mask_row_split = split;
        return;
    }
    // THE MIXED FIRE (M-2): a prefill-shaped masked fire with a planned
    // unmasked prefix — prefill and plain-decode rows keep the causal
    // prefill dispatch over the prefix, the masked 1-TOKEN suffix takes
    // the custom dispatch (the decode-class suffix machinery verbatim:
    // identity qo, absolute device CSRs at +split_req). The planned
    // word counts TOKEN ROWS; the request split derives from the host
    // qo indptr. Shape guards keep the fire-level arm loudly-silently:
    // any mismatch just declines the split (the body's UNPLANNED
    // endpoint), never throws — the engine relax may send shapes v0
    // does not serve (multi-token masked members).
    if (spatial_mask_on && have_custom_mask && !is_pure_decode &&
        unmasked_prefix_rows != 0xffffffffu &&
        unmasked_prefix_rows > 0 &&
        unmasked_prefix_rows < static_cast<std::uint32_t>(total_tokens) &&
        cfg.head_dim == cfg.head_dim_kernel &&
        fwd_cfg.per_layer_window_left.empty()) {
        // The planned word is the REQUEST/lane index (the engine's
        // program-row domain — measured live: R=4 qo=[0,221,222,223,224]
        // plans 3, the masked member's lane start, not token row 223).
        // Pure-decode fires never showed the distinction (row == lane).
        const int split_req =
            unmasked_prefix_rows < static_cast<std::uint32_t>(num_requests)
                ? static_cast<int>(unmasked_prefix_rows)
                : -1;
        bool suffix_decode = split_req > 0 && split_req < num_requests;
        if (suffix_decode) {
            for (int r = split_req; r < num_requests; ++r) {
                if (qo_indptr_h[r + 1] - qo_indptr_h[r] != 1) {
                    suffix_decode = false;
                    break;
                }
            }
        }
        if (suffix_decode) {
            if (std::getenv("PIE_KVPP_TRACE") != nullptr) {
                std::fprintf(stderr, "[kvpp] R=%d split_req=%d qo=[", 
                             num_requests, split_req);
                for (int r = 0; r <= num_requests; ++r)
                    std::fprintf(stderr, "%u,", qo_indptr_h[r]);
                std::fprintf(stderr, "] kvpp=[");
                for (int r = 0; r <= num_requests; ++r)
                    std::fprintf(stderr, "%u,", kv_page_indptr_h[r]);
                std::fprintf(stderr, "]\n");
            }
            const int rs = num_requests - split_req;
            const int T = (fwd_cfg.tp_size > 0) ? fwd_cfg.tp_size : 1;
            const int num_q_heads_local = cfg.num_attention_heads / T;
            const int num_kv_heads_local = cfg.num_key_value_heads / T;
            // The prefix causal plan: the fire's host arrays' heads ARE
            // the prefix's truth (requests [0, split_req), tokens
            // [0, planned rows)). No score capture: hooks are UNPLANNED.
            if (!state.prefill_plan) {
                state.prefill_plan = ops::make_prefill_plan();
            }
            ops::plan_attention_flashinfer_prefill_bf16(
                *state.prefill_plan,
                qo_indptr_h,
                kv_page_indptr_h,
                kv_last_page_lens_h,
                static_cast<int>(qo_indptr_h[split_req]),
                split_req,
                num_q_heads_local,
                num_kv_heads_local,
                cfg.head_dim_kernel,
                cache.page_size(),
                attn_ws,
                /*stream=*/nullptr,
                fwd_cfg.decode_plan_cuda_graph,
                fwd_cfg.sliding_window,
                /*full_attention_variant=*/false,
                cache.hnd_layout(),
                /*causal_mask=*/true,
                /*custom_mask=*/false);
            state.use_prefill_plan = true;
            // NO-DEMOTION: split the prefix again — prefill lanes
            // first (the seriation's multi_token term), then the
            // plain-decode middle [P, split_req), which gets the
            // DECODE kernel instead of demoting to the causal prefill.
            // P derives from the host qo (first width-1 request).
            state.mixed_mid_decode_plan.reset();
            state.mixed_mid_start = -1;
            {
                int P = split_req;
                for (int r = 0; r < split_req; ++r) {
                    if (qo_indptr_h[r + 1] - qo_indptr_h[r] == 1) {
                        P = r;
                        break;
                    }
                }
                const int mid = split_req - P;
                // PIE_MIXED_MID=0 disarms (the demotion-vs-decode A/B
                // instrument; default ON).
                static const bool mid_armed = [] {
                    const char* v = std::getenv("PIE_MIXED_MID");
                    return v == nullptr || v[0] != '0';
                }();
                if (mid_armed && mid > 0 && P > 0 &&
                    !fwd_cfg.force_prefill_path &&
                    !fwd_cfg.use_prefill_decode_plan) {
                    // Middle decode plan over requests [P, split_req):
                    // kvpp rebased to the middle's page base (the
                    // suffix-plan pattern, third application). Decode
                    // and prefill plan REGIONS are disjoint within one
                    // workspace (the NS-2 precedent), so attn_ws holds
                    // the prefix-causal + this decode plan together.
                    std::vector<std::uint32_t> kvpp_mid(
                        static_cast<std::size_t>(mid) + 1);
                    const std::uint32_t mid_base = kv_page_indptr_h[P];
                    for (int i = 0; i <= mid; ++i) {
                        kvpp_mid[static_cast<std::size_t>(i)] =
                            kv_page_indptr_h[P + i] - mid_base;
                    }
                    if (!state.mixed_mid_decode_plan) {
                        state.mixed_mid_decode_plan =
                            ops::make_decode_plan();
                    }
                    ops::plan_attention_flashinfer_decode(
                        *state.mixed_mid_decode_plan, kvpp_mid.data(),
                        mid, num_q_heads_local, num_kv_heads_local,
                        cfg.head_dim_kernel, cache.page_size(), attn_ws,
                        /*stream=*/nullptr,
                        fwd_cfg.decode_plan_cuda_graph,
                        decode_full_attention_variant_enabled() &&
                            fwd_cfg.sliding_window < 0 &&
                            fwd_cfg.per_layer_window_left.empty(),
                        cache.hnd_layout());
                    state.mixed_mid_start = P;
                    // Re-plan the prefix CAUSAL to the prefill lanes
                    // only (requests [0, P), tokens qo[P]) — the middle
                    // now belongs to the decode kernel.
                    ops::plan_attention_flashinfer_prefill_bf16(
                        *state.prefill_plan,
                        qo_indptr_h,
                        kv_page_indptr_h,
                        kv_last_page_lens_h,
                        static_cast<int>(qo_indptr_h[P]),
                        P,
                        num_q_heads_local,
                        num_kv_heads_local,
                        cfg.head_dim_kernel,
                        cache.page_size(),
                        attn_ws,
                        /*stream=*/nullptr,
                        fwd_cfg.decode_plan_cuda_graph,
                        fwd_cfg.sliding_window,
                        /*full_attention_variant=*/false,
                        cache.hnd_layout(),
                        /*causal_mask=*/true,
                        /*custom_mask=*/false);
                }
            }
            // The suffix mask plan: identity qo over the 1-token rows,
            // page geometry from the resolver counts when threaded
            // (composed envelopes) or the host CSR slice (wire lanes) —
            // the decode-class suffix block verbatim.
            std::vector<std::uint32_t> qo_suffix(
                static_cast<std::size_t>(rs) + 1);
            std::vector<std::uint32_t> kvpp_suffix(
                static_cast<std::size_t>(rs) + 1);
            for (int i = 0; i <= rs; ++i) {
                qo_suffix[static_cast<std::size_t>(i)] =
                    static_cast<std::uint32_t>(i);
            }
            if (mask_suffix_page_counts_h != nullptr) {
                kvpp_suffix[0] = 0;
                for (int i = 0; i < rs; ++i) {
                    kvpp_suffix[static_cast<std::size_t>(i) + 1] =
                        kvpp_suffix[static_cast<std::size_t>(i)] +
                        mask_suffix_page_counts_h[i];
                }
            } else {
                const std::uint32_t page_base = kv_page_indptr_h[split_req];
                for (int i = 0; i <= rs; ++i) {
                    kvpp_suffix[static_cast<std::size_t>(i)] =
                        kv_page_indptr_h[split_req + i] - page_base;
                }
            }
            if (!state.mask_decode_plan) {
                state.mask_decode_plan = ops::make_prefill_plan();
            }
            ops::plan_attention_flashinfer_prefill_bf16(
                *state.mask_decode_plan,
                qo_suffix.data(),
                kvpp_suffix.data(),
                mask_suffix_last_lens_h != nullptr
                    ? mask_suffix_last_lens_h
                    : kv_last_page_lens_h + split_req,
                rs,
                rs,
                num_q_heads_local,
                num_kv_heads_local,
                cfg.head_dim_kernel,
                cache.page_size(),
                spatial_suffix_ws(),
                /*stream=*/nullptr,
                fwd_cfg.decode_plan_cuda_graph,
                /*window_left=*/-1,
                /*full_attention_variant=*/false,
                cache.hnd_layout(),
                /*causal_mask=*/false,
                /*custom_mask=*/true);
            state.use_mask_decode_plan = true;
            state.spatial_mask_split = split_req;
            state.spatial_mask_row_split =
                static_cast<int>(qo_indptr_h[split_req]);
            return;
        }
    }
    if (have_custom_mask) {
        // Pure-decode custom-mask fires plan into their DEDICATED slot
        // (see LlamaLikePlanState::mask_decode_plan); prefill-shaped
        // custom-mask fires keep the prefill slot.
        auto& mask_plan = is_pure_decode ? state.mask_decode_plan
                                         : state.prefill_plan;
        if (!mask_plan) {
            mask_plan = ops::make_prefill_plan();
        }
        const int T = (fwd_cfg.tp_size > 0) ? fwd_cfg.tp_size : 1;
        const int num_q_heads_local  = cfg.num_attention_heads / T;
        const int num_kv_heads_local = cfg.num_key_value_heads / T;
        ops::plan_attention_flashinfer_prefill_bf16(
            *mask_plan,
            qo_indptr_h,
            kv_page_indptr_h,
            kv_last_page_lens_h,
            total_tokens,
            num_requests,
            num_q_heads_local,
            num_kv_heads_local,
            cfg.head_dim_kernel,
            cache.page_size(),
            attn_ws,
            /*stream=*/nullptr,
            fwd_cfg.decode_plan_cuda_graph,
            /*window_left=*/-1,
            /*full_attention_variant=*/false,
            cache.hnd_layout(),
            /*causal_mask=*/false,
            /*custom_mask=*/true);
        if (is_pure_decode) {
            state.use_mask_decode_plan = true;
        } else {
            state.use_prefill_plan = true;
        }
        return;
    }
    if (is_pure_decode && fwd_cfg.use_xqa_decode &&
        cache.format().is_native_bf16() && !cache.hnd_layout()) {
        int max_pages = 1;
        for (int r = 0; r < num_requests; ++r) {
            const int pages = static_cast<int>(
                kv_page_indptr_h[r + 1] - kv_page_indptr_h[r]);
            max_pages = std::max(max_pages, pages);
        }
        state.use_xqa_decode = true;
        state.xqa_max_pages_per_seq =
            ops::xqa_decode_page_bucket(max_pages);
        // ④ envelope banding: this deployment's band walk is PLAN-FREE
        // (the XQA arm reads the fire's staged device CSRs and takes the
        // row count as a parameter), so stamping k/rows is ALL the
        // prepare owes — no flashinfer band plans. Without this stamp
        // the early return demoted every XQA deployment (14B) to the
        // full-depth walk with neither [depth-bands] nor DECLINE.
        state.depth_band_count = 0;
        if (depth_band_count >= 1 && depth_band_count <= 3 &&
            !have_custom_mask) {
            for (std::uint32_t j = 0; j < depth_band_count; ++j) {
                state.depth_band_k[j] = depth_band_k[j];
                state.depth_band_rows[j] = depth_band_rows[j];
            }
            state.depth_band_count = depth_band_count;
        }
        if (std::getenv("PIE_REGION_TRACE") != nullptr) {
            std::fprintf(stderr,
                         "[band-prep] xqa-branch in=%u stamped=%u\n",
                         depth_band_count, state.depth_band_count);
        }
        return;
    }
    if (!is_pure_decode) {
        // Real prefill/mixed batches share one attention schedule across all
        // layers when the model has a single global attention window. Plan it
        // once here so the forward body can dispatch per layer without doing
        // host planner work inside the model loop. Models with alternating
        // sliding-window layouts keep the older per-layer planner path.
        if (fwd_cfg.per_layer_window_left.empty()) {
            if (!state.prefill_plan) {
                state.prefill_plan = ops::make_prefill_plan();
            }
            const int T = (fwd_cfg.tp_size > 0) ? fwd_cfg.tp_size : 1;
            const int num_q_heads_local  = cfg.num_attention_heads / T;
            const int num_kv_heads_local = cfg.num_key_value_heads / T;
            // SnapKV observes the tail of the prompt, so the capture has to
            // be decided HERE: the planner picks SM90-vs-FA2, and only FA2 is
            // instrumented. A sliding-window model is excluded for the same
            // reason the decode capture excludes one -- `LogitsMask` runs
            // after the hook, so the captured row would describe positions the
            // softmax discards.
            const std::uint32_t score_window =
                (fwd_cfg.sliding_window < 0) ? attn_score_window : 0u;
            ops::plan_attention_flashinfer_prefill_bf16(
                *state.prefill_plan,
                qo_indptr_h,
                kv_page_indptr_h,
                kv_last_page_lens_h,
                total_tokens,
                num_requests,
                num_q_heads_local,
                num_kv_heads_local,
                cfg.head_dim_kernel,
                cache.page_size(),
                attn_ws,
                /*stream=*/nullptr,
                /*graph_mode_plan=*/false,
                fwd_cfg.sliding_window,
                /*full_attention_variant=*/false,
                cache.hnd_layout(),
                /*causal_mask=*/true,
                /*custom_mask=*/false,
                /*wants_prefill_score=*/score_window > 0);
            state.use_prefill_plan = true;
            state.prefill_score_window = score_window;
        }
        return;
    }
    if (fwd_cfg.force_prefill_path) {
        state.use_prefill_plan = false;
        state.use_prefill_decode_plan = false;
        // ④ envelope banding, force_prefill deployment (14B-class: the
        // GQA ratio keeps the decode kernel out, decode runs the
        // PLAN-FREE prefill dispatch): a band is the prefix call
        // N = R = rows on that same dispatch — the spatial split's
        // prefix already runs exactly this shape. k/rows is all the
        // prepare owes; there are no plans on this deployment at all.
        state.depth_band_count = 0;
        if (depth_band_count >= 1 && depth_band_count <= 3 &&
            is_pure_decode && !have_custom_mask) {
            for (std::uint32_t j = 0; j < depth_band_count; ++j) {
                state.depth_band_k[j] = depth_band_k[j];
                state.depth_band_rows[j] = depth_band_rows[j];
            }
            state.depth_band_count = depth_band_count;
        }
        if (std::getenv("PIE_REGION_TRACE") != nullptr) {
            std::fprintf(stderr,
                         "[band-prep] force-prefill-branch in=%u stamped=%u\n",
                         depth_band_count, state.depth_band_count);
        }
        return;
    }
    const int min_prefill_decode_pages =
        std::max(0, fwd_cfg.prefill_decode_min_kv_pages);
    kvpp_sentry("prepare", qo_indptr_h, kv_page_indptr_h, num_requests);
    // ④ Act 1: bands are re-stamped per fire; a deployment branch that
    // returns early must not leave a previous fire's bands armed.
    state.depth_band_count = 0;
    std::uint64_t total_kv_pages = 0;
    for (int r = 0; r < num_requests; ++r) {
        total_kv_pages += static_cast<std::uint64_t>(
            kv_page_indptr_h[r + 1] - kv_page_indptr_h[r]);
    }
    const int avg_kv_pages = num_requests > 0
        ? static_cast<int>((total_kv_pages + num_requests - 1) / num_requests)
        : 0;
    state.use_prefill_decode_plan =
        fwd_cfg.use_prefill_decode_plan &&
        (min_prefill_decode_pages == 0 ||
         avg_kv_pages >= min_prefill_decode_pages);
    if (state.use_prefill_decode_plan) {
        if (!state.prefill_decode_plan) {
            state.prefill_decode_plan = ops::make_prefill_plan();
        }
        const int T = (fwd_cfg.tp_size > 0) ? fwd_cfg.tp_size : 1;
        const int num_q_heads_local  = cfg.num_attention_heads / T;
        const int num_kv_heads_local = cfg.num_key_value_heads / T;
        auto& qo_indptr_h = state.prefill_decode_qo_indptr_h;
        qo_indptr_h.resize(num_requests + 1);
        for (int r = 0; r <= num_requests; ++r) {
            qo_indptr_h[r] = static_cast<std::uint32_t>(r);
        }
        const int min_full_attention_pages =
            std::max(0, fwd_cfg.prefill_decode_full_attention_min_kv_pages);
        const bool full_attention_variant =
            fwd_cfg.prefill_decode_full_attention_min_requests > 0 &&
            num_requests >= fwd_cfg.prefill_decode_full_attention_min_requests &&
            (min_full_attention_pages == 0 ||
             avg_kv_pages >= min_full_attention_pages) &&
            fwd_cfg.sliding_window < 0 &&
            fwd_cfg.per_layer_window_left.empty();
        ops::plan_attention_flashinfer_prefill_bf16(
            *state.prefill_decode_plan, qo_indptr_h.data(), kv_page_indptr_h,
            kv_last_page_lens_h,
            /*total_tokens=*/num_requests, num_requests,
            num_q_heads_local, num_kv_heads_local, cfg.head_dim_kernel,
            cache.page_size(), attn_ws, /*stream=*/nullptr,
            fwd_cfg.decode_plan_cuda_graph, fwd_cfg.sliding_window,
            full_attention_variant, cache.hnd_layout(),
            /*causal_mask=*/false);
        // ④ Act 1 (banded depth, prefill family): a band's prefix
        // dispatch on this deployment is the planned causal prefill —
        // one plan per boundary, identity-qo prefix restriction, each
        // in its OWN workspace (the per-band isolation rule).
        if (depth_band_count >= 1 && depth_band_count <= 3 &&
            is_pure_decode && !have_custom_mask) {
            for (std::uint32_t j = 0; j < depth_band_count; ++j) {
                const std::uint32_t rows = depth_band_rows[j];
                state.depth_band_k[j] = depth_band_k[j];
                state.depth_band_rows[j] = rows;
                if (rows == 0) continue;
                if (!state.depth_band_prefill_plans[j]) {
                    state.depth_band_prefill_plans[j] =
                        ops::make_prefill_plan();
                }
                ops::plan_attention_flashinfer_prefill_bf16(
                    *state.depth_band_prefill_plans[j],
                    qo_indptr_h.data(), kv_page_indptr_h,
                    kv_last_page_lens_h,
                    /*total_tokens=*/static_cast<int>(rows),
                    static_cast<int>(rows),
                    num_q_heads_local, num_kv_heads_local,
                    cfg.head_dim_kernel, cache.page_size(),
                    depth_band_ws(static_cast<int>(j)),
                    /*stream=*/nullptr,
                    fwd_cfg.decode_plan_cuda_graph, fwd_cfg.sliding_window,
                    full_attention_variant, cache.hnd_layout(),
                    /*causal_mask=*/false);
            }
            state.depth_band_count = depth_band_count;
        }
        if (std::getenv("PIE_REGION_TRACE") != nullptr) {
            std::fprintf(stderr,
                         "[band-prep] prefill-branch in=%u stamped=%u\n",
                         depth_band_count, state.depth_band_count);
        }
        return;
    }
    if (!state.decode_plan) {
        state.decode_plan = ops::make_decode_plan();
    }
    const int T = (fwd_cfg.tp_size > 0) ? fwd_cfg.tp_size : 1;
    const int num_q_heads_local  = cfg.num_attention_heads / T;
    const int num_kv_heads_local = cfg.num_key_value_heads / T;
    ops::plan_attention_flashinfer_decode(
        *state.decode_plan, kv_page_indptr_h, num_requests,
        num_q_heads_local, num_kv_heads_local, cfg.head_dim_kernel,
        cache.page_size(), attn_ws, /*stream=*/nullptr,
        fwd_cfg.decode_plan_cuda_graph,
        decode_full_attention_variant_enabled() &&
            fwd_cfg.sliding_window < 0 && fwd_cfg.per_layer_window_left.empty(),
        cache.hnd_layout());
    // STRUCTURAL S-2: the depth union's PREFIX plan — requests
    // [0, split) at layers [k, L). Same planner family as the full plan
    // above, so it plans against the SECONDARY workspace (the
    // two-plans-one-workspace lesson); the body's range-2 dispatch pairs
    // with it. Shape guards DECLINE (leave the slot null and the body's
    // loud check speaks) rather than throw.
    if (full_depth_rows != 0xffffffffu &&
        full_depth_rows > 0 &&
        full_depth_rows < static_cast<std::uint32_t>(num_requests) &&
        is_pure_decode && !have_custom_mask) {
        if (!state.depth_prefix_decode_plan) {
            state.depth_prefix_decode_plan = ops::make_decode_plan();
        }
        ops::plan_attention_flashinfer_decode(
            *state.depth_prefix_decode_plan, kv_page_indptr_h,
            static_cast<int>(full_depth_rows),
            num_q_heads_local, num_kv_heads_local, cfg.head_dim_kernel,
            cache.page_size(), spatial_suffix_attn_ws(),
            /*stream=*/nullptr,
            fwd_cfg.decode_plan_cuda_graph,
            decode_full_attention_variant_enabled() &&
                fwd_cfg.sliding_window < 0 &&
                fwd_cfg.per_layer_window_left.empty(),
            cache.hnd_layout());
    }
    // V2 rung ④ Act 1 (banded depth): one prefix decode plan per band
    // boundary, deepest-first, each against its own workspace. A band
    // whose start row is 0 needs no plan — nothing lives past it and
    // the body stops walking layers there.
    state.depth_band_count = 0;
    if (depth_band_count >= 1 && depth_band_count <= 3 &&
        is_pure_decode && !have_custom_mask) {
        for (std::uint32_t j = 0; j < depth_band_count; ++j) {
            const std::uint32_t rows = depth_band_rows[j];
            state.depth_band_k[j] = depth_band_k[j];
            state.depth_band_rows[j] = rows;
            if (rows == 0) continue;
            // ④ envelope banding: the XQA deployment is PLAN-FREE — its
            // kernels read the (compose-written) device CSRs directly and
            // take the row count as a parameter, so a band is just the
            // prefix call R = rows. No flashinfer band plans are built
            // (the host CSRs are 1-page placeholders on the composed
            // path and would plan garbage); the band walk dispatches XQA
            // against the fire's own staged workspace instead.
            if (state.use_xqa_decode) continue;
            if (!state.depth_band_plans[j]) {
                state.depth_band_plans[j] = ops::make_decode_plan();
            }
            ops::plan_attention_flashinfer_decode(
                *state.depth_band_plans[j], kv_page_indptr_h,
                static_cast<int>(rows),
                num_q_heads_local, num_kv_heads_local, cfg.head_dim_kernel,
                cache.page_size(), depth_band_ws(static_cast<int>(j)),
                /*stream=*/nullptr,
                fwd_cfg.decode_plan_cuda_graph,
                decode_full_attention_variant_enabled() &&
                    fwd_cfg.sliding_window < 0 &&
                    fwd_cfg.per_layer_window_left.empty(),
                cache.hnd_layout());
        }
        state.depth_band_count = depth_band_count;
    }
    if (std::getenv("PIE_REGION_TRACE") != nullptr) {
        std::fprintf(stderr,
                     "[band-prep] decode-branch in=%u stamped=%u\n",
                     depth_band_count, state.depth_band_count);
    }
}

std::uint32_t llama_like_decode_graph_layout(
    const LlamaLikePlanState& state)
{
    // NS-3: a spatial-split fire's captured body bakes BOTH plans' grids
    // and the split-derived pointer offsets — all three join the layout
    // (splitmix, the layout hashes' existing posture).
    if (state.spatial_mask_split >= 0 && state.use_mask_decode_plan &&
        state.mask_decode_plan) {
        auto mix = [](std::uint64_t x) {
            x += 0x9e3779b97f4a7c15ull;
            x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
            x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
            return x ^ (x >> 31);
        };
        std::uint64_t h = mix(
            0x5350414Cull ^
            static_cast<std::uint64_t>(state.spatial_mask_split));
        h = mix(h ^ ops::prefill_plan_graph_layout(
                        *state.mask_decode_plan));
        if (state.spatial_mask_split > 0 && state.decode_plan) {
            h = mix(h ^ ops::decode_plan_graph_layout(*state.decode_plan));
        }
        return static_cast<std::uint32_t>(h & 0x00ffffffu);
    }
    if (state.use_xqa_decode) {
        return ops::xqa_decode_graph_layout(state.xqa_max_pages_per_seq);
    }
    if (state.use_prefill_decode_plan && state.prefill_decode_plan) {
        return ops::prefill_plan_graph_layout(*state.prefill_decode_plan);
    }
    if (state.use_prefill_plan && state.prefill_plan) {
        return ops::prefill_plan_graph_layout(*state.prefill_plan);
    }
    if (state.use_mask_decode_plan && state.mask_decode_plan) {
        return ops::prefill_plan_graph_layout(*state.mask_decode_plan);
    }
    if (!state.decode_plan) return 0;
    return ops::decode_plan_graph_layout(*state.decode_plan);
}

std::uint32_t llama_like_supergraph_graph_layout(
    const LlamaLikePlanState& state)
{
    // The UNION key's layout (S3): the supergraph contains BOTH the
    // decode dispatch (else arm) and the custom-mask prefill dispatch
    // (mask arm), so its capture validity spans both plans' kernel
    // configurations. Each fire's prepare refreshes exactly the live
    // arm's plan and leaves the other stable, so masked and unmasked
    // fires at one (R, N) compose the SAME pair and share the exec —
    // which is the whole point. A plan whose re-plan shifts its layout
    // shifts the key and recaptures, exactly the existing per-layout
    // discipline.
    const std::uint32_t decode_side = state.use_xqa_decode
        ? ops::xqa_decode_graph_layout(state.xqa_max_pages_per_seq)
        : (state.decode_plan
               ? ops::decode_plan_graph_layout(*state.decode_plan)
               : 0u);
    const std::uint32_t mask_side =
        state.mask_decode_plan
            ? ops::prefill_plan_graph_layout(*state.mask_decode_plan)
            : 0u;
    // splitmix-style mix so the pair cannot alias a plain layout.
    std::uint32_t h = decode_side + 0x9e3779b9u;
    h ^= mask_side + 0x85ebca6bu + (h << 6) + (h >> 2);
    return h;
}

void llama_like_forward_paged(
    const Qwen3Weights& w,
    const HfConfig& cfg,
    const LlamaLikeForwardCfg& fwd_cfg,
    const LlamaLikePlanState& plan_state,
    Workspace& ws,
    KvCache& cache,
    AttentionWorkspace& attn_ws,
    ops::CublasHandle& cublas,
    const std::int32_t* token_ids,
    const std::int32_t* positions,
    const std::uint32_t* qo_indptr,
    const std::uint32_t* kv_page_indices,
    const std::uint32_t* kv_page_indptr,
    const std::uint32_t* kv_last_page_lens,
    const std::uint32_t* qo_indptr_h,
    const std::uint32_t* kv_page_indptr_h,
    int N,
    int R,
    bool is_pure_decode,
    const std::int32_t* logit_row_indices_d,
    int num_logit_rows,
    const std::uint8_t* custom_mask_d,
    const std::int32_t* custom_mask_indptr_d,
    const std::uint32_t* w_page_d,
    const std::uint32_t* w_off_d,
    const std::uint8_t* row_valid_d,
    bool has_write_desc,
    int runtime_window_left,
    const LlamaLikeVisionInputs* vision,
    const StageHooks* hooks,
    const LoraTable* lora,
    const std::uint32_t* peel_window_d,
    std::uint32_t unmasked_prefix_rows,
    const std::uint32_t* mask_suffix_qo_indptr_d,
    const std::uint32_t* mask_suffix_kv_page_indptr_d,
    std::uint32_t max_layers,
    std::uint32_t full_depth_rows)
{
    // Tensor-parallel local dims. tp_size == 1 reverts to single-GPU
    // shapes; the local *_local fields just shadow the unsharded value.
    // Every tp-aware dim (q/k heads, intermediate, lm_head bound to embed)
    // must divide cleanly by tp_size — checked at engine load.
    const int T  = (fwd_cfg.tp_size > 0) ? fwd_cfg.tp_size : 1;
    const int H  = cfg.hidden_size;
    const int Hq_full = cfg.num_attention_heads * cfg.head_dim;
    const int Hk_full = cfg.num_key_value_heads * cfg.head_dim;
    const int I_full  = cfg.intermediate_size;
    const int Hq = Hq_full / T;
    const int Hk = Hk_full / T;
    const int I  = I_full / T;
    const int num_q_heads_local  = cfg.num_attention_heads / T;
    const int num_kv_heads_local = cfg.num_key_value_heads / T;
    const int V  = cfg.vocab_size;
    const int d  = cfg.head_dim;
    const int dk = cfg.head_dim_kernel;            // padded HEAD_DIM the
                                                   // attention kernel runs at
    const bool head_dim_padded = (d != dk);
    const float eps = cfg.rms_norm_eps;
    // Inherit the stream bound to `cublas` so manual kernel launches stay
    // on the same stream as the cublas matmuls. The graph-capture path
    // in batch/forward.cpp binds `cublas` to its `cstream` for the
    // duration of capture so every launch in this body — cublas-issued or
    // not — lands on the captured graph.
    cudaStream_t stream = cublas.stream();
    NcclComm* tp = (T > 1) ? fwd_cfg.tp_comm : nullptr;
    const bool native_bf16_kv_cache = cache.format().is_native_bf16();

    // §5.1 CORRECTION: validate the fire's lora lanes and stage their
    // adapter weights (f32 channel cells → bf16) once, before the layer
    // loop. `has_lora` gates every lora-conditional below; when false the
    // body is byte-for-byte what it was ("with no adapters the code is
    // what it was").
    const bool has_lora = lora != nullptr && lora->usable();
    // Campaign step 3a: prefer the ENGINE-staged state (staged outside
    // any capture region; identity-checked against this fire's table).
    // Local staging remains the fallback for callers that never invoke
    // the stage hook.
    const LoraFireStateHandle* lora_staged =
        (has_lora && plan_state.lora_staged_table == lora)
            ? plan_state.lora_staged.get()
            : nullptr;
    std::optional<LoraFireState> lora_state;
    if (has_lora && lora_staged == nullptr) {
        lora_state.emplace(
            *lora, cfg, N, H, Hq, Hk, I, T, stream, ws.lora_arena,
            fwd_cfg.norm_placement == NormPlacement::Post
                ? static_cast<const void*>(ws.y.data())
                : static_cast<const void*>(ws.norm_x.data()),
            ws.q.data(), ws.v.data(), ws.gate.data());
    }
    // Co-batch evidence, PIE_HOOK_PREFIX_TRACE's pattern: one line per
    // fire proving how many request rows this fire carries (R) and how
    // many of them are adapter lanes with which token spans. R > lanes'
    // covered rows means adapter and no-adapter lanes shared the fire.
    if (has_lora && std::getenv("PIE_LORA_FIRE_TRACE") != nullptr) {
        std::string spans;
        for (std::uint32_t i = 0; i < lora->count; ++i) {
            const LoraLaneView& lane = lora->lanes[i];
            spans += (i == 0 ? "" : ",");
            spans += std::to_string(lane.token_start) + "+" +
                     std::to_string(lane.token_count);
        }
        std::fprintf(stderr,
                     "[lora-fire] R=%d lanes=%u spans=%s grouping=%s%s\n",
                     R, lora->count, spans.c_str(),
                     lora_staged != nullptr
                         ? lora_staged->grouping_desc().c_str()
                         : lora_state->grouping_desc().c_str(),
                     lora_staged != nullptr ? " (engine-staged)" : "");
    }

    // When head_dim is padded, the attention kernel runs at `dk`
    // (e.g. 128) and consumes Q/K/V from the *padded* workspace
    // buffers; the packed buffers stay at `d` (e.g. 96) for the
    // GEMM in/out paths. flashinfer's softmax expects
    // `1/sqrt(head_dim)`; with padding we need `1/sqrt(d)` (the
    // real head_dim) regardless of `dk`.
    const float sm_scale_override = head_dim_padded
        ? (1.0f / std::sqrt(static_cast<float>(d)))
        : -1.f;  // -1 = let the dispatch pick `1/sqrt(dk)`

    // 1. Embed.
    kernels::launch_embed_bf16(
        token_ids, w.embed->data(), ws.y.data(),
        N, H, cfg.vocab_size, stream);

    // 1b. Qwen3-VL multimodal: encode each image and overwrite its soft-token
    // rows in the embed output; also stash the deepstack merger outputs (added
    // into the hidden state on image rows after early decoder layers below).
    if (vision != nullptr && vision->vision_in != nullptr &&
        vision->vision_in->num_images > 0) {
        scatter_qwen3vl_vision(
            *vision->vision_in,
            static_cast<__nv_bfloat16*>(ws.y.data()),
            N, H,
            static_cast<__nv_bfloat16*>(vision->deepstack_scratch),
            vision->num_deepstack, cublas.handle(), stream);
    }

    // Some GQA group sizes (Qwen2 small models — 6, 7) aren't in
    // flashinfer's decode dispatch table; for those we run the prefill
    // kernel even for qo_len==1 batches. The runtime decision lives in
    // a single bool: anything past the plan_attention call uses it.
    // Beam / custom-mask decode (SEAM-1, overview §6.2): a decode-shaped batch
    // (qo_len==1/req) that carries a per-cell custom mask cannot use the decode
    // or xqa kernels — they have no per-cell mask (fork-freeze needs a mid-page
    // hole in a non-last page). Route it through the custom-mask prefill path
    // (the `else if (custom_mask_d)` branch below). Disabling `use_decode_path`
    // ALSO disables the fused decode qkv-postproc (below), so the per-beam KV
    // write goes through `launch_write_kv_to_pages`, whose page-run derivation
    // (abs_kv_pos = klen-1) lands each beam's new token on (last live page,
    // last_page_len-1) = the correct freeze/heir write target.
    const bool has_custom_mask = (custom_mask_d != nullptr);
    const bool use_xqa_decode_path =
        is_pure_decode && !has_custom_mask && plan_state.use_xqa_decode;
    const bool use_decode_path =
        is_pure_decode && !has_custom_mask &&
        (!fwd_cfg.force_prefill_path || use_xqa_decode_path);
    const bool use_prefill_decode_path =
        use_decode_path && !use_xqa_decode_path &&
        plan_state.use_prefill_decode_plan;

    // Decode plan was set up by the prepare hook (runs outside any
    // cudaStream capture region) — the body just reads from
    // `plan_state.decode_plan`. Keeps the body purely device-side so
    // the executor can capture it into a CUDA graph.
    const ops::DecodePlanCache* decode_plan =
        plan_state.decode_plan ? plan_state.decode_plan.get() : nullptr;
    const ops::PrefillPlanCache* prefill_decode_plan =
        plan_state.prefill_decode_plan ? plan_state.prefill_decode_plan.get() : nullptr;
    const ops::PrefillPlanCache* prefill_plan =
        plan_state.prefill_plan ? plan_state.prefill_plan.get() : nullptr;
    // SnapKV's observation window only exists on the true prefill branch: the
    // prepare hook plans the score-capturing dispatch there and nowhere else,
    // so the flag is a restatement of what that plan already committed to
    // rather than an independent decision the body could get wrong.
    const bool use_prefill_score_path =
        plan_state.prefill_score_window > 0 &&
        plan_state.use_prefill_plan && prefill_plan != nullptr &&
        !use_decode_path && !has_custom_mask;

    if (use_xqa_decode_path) {
        ops::prepare_attention_xqa_decode_bf16(
            kv_page_indices,
            kv_page_indptr,
            kv_last_page_lens,
            R,
            cache.page_size(),
            plan_state.xqa_max_pages_per_seq,
            attn_ws,
            stream);
    }

    // Track A: the fire's page mask. Allocated once for the whole layer loop
    // because page geometry belongs to the fire, not to the layer. The
    // constructor is where a fire that cannot safely honour a mask is refused
    // -- loudly, before any layer runs, rather than by quietly attending over
    // the wrong pages.
    model::FirePageMask page_mask(hooks, stream);

    const bool post_norm = fwd_cfg.norm_placement == NormPlacement::Post;
    bool have_next_attn_norm = false;
    bool have_final_norm = false;
    const void* final_norm_buf = nullptr;
    bool rope_table_ready = false;
    const float* rope_table = nullptr;

    // STRUCTURAL v0 (S-1): a truncated fire runs layers [0, k) and the
    // unchanged tail takes the head at layer k (logit lens). k == L is
    // the identity by construction.
    const int layer_bound =
        max_layers != 0xffffffffu &&
        max_layers < static_cast<std::uint32_t>(cfg.num_hidden_layers)
            ? static_cast<int>(max_layers)
            : cfg.num_hidden_layers;
    // STRUCTURAL S-2: the layer body, parameterized by (rows, plan, ws)
    // so the depth union can run it over two ranges. The lambda's
    // parameters SHADOW the outer locals of the same names — the body
    // text is unchanged. Depth fires are pure decode, so N == R and
    // the full-depth prefix is rows [0, split) of every base array.
    const auto run_layer = [&](const int L, const int N, const int R,
                               const ops::DecodePlanCache* decode_plan,
                               AttentionWorkspace& attn_ws,
                               const ops::PrefillPlanCache*
                                   prefill_plan_override = nullptr) {
        // Tier 2: hook stages fire only while the hook rows are live —
        // a truncated hook region's rows freeze past its k in the banded
        // walk, and an invocation there would observe garbage rows. The
        // same bound caps both ledgers (frame prep planned_layers,
        // dispatch finish expected_layers) via hook_region_k, so the
        // three agree by construction.
        const model::StageHooks* layer_hooks =
            (hooks != nullptr &&
             static_cast<std::uint32_t>(L) < hooks->hook_rows_k)
                ? hooks
                : nullptr;
        const auto& layer = w.layers[L];

        // Pre-norm: norm(y) → norm_x; QKV reads from norm_x.
        // Post-norm (OLMo-3): QKV reads y directly; norm_attn applied
        //                     to attn_out *before* the residual add.
        const void* qkv_in = ws.y.data();
        if (!post_norm) {
            if (!have_next_attn_norm) {
                kernels::launch_rmsnorm_bf16(
                    ws.y.data(), layer.attn_norm->data(), ws.norm_x.data(),
                    N, H, eps, stream);
            }
            qkv_in = ws.norm_x.data();
            have_next_attn_norm = false;
        }

        // QKV: fused path when the bind helper materialised
        // `qkv_proj_fused` (single wide gemm + split kernel), unfused
        // fallback for quantized projections / TP-sharded loads /
        // architectures that haven't opted in yet.
        const bool q_norm_is_per_head =
            layer.q_norm && layer.q_norm->shape().size() == 1 &&
            layer.q_norm->shape()[0] == d;
        const bool k_norm_is_per_head =
            layer.k_norm && layer.k_norm->shape().size() == 1 &&
            layer.k_norm->shape()[0] == d;
        const bool use_fused_qkv = (layer.qkv_proj_fused != nullptr) &&
                                   !ws.qkv_fused.empty();
        // The hook-free fast prefix. With no hooks every row is it; with
        // hooks, the dispatch proved rows [0, fast_rows) belong to no
        // attention-stage program, so they may take the fused postprocess
        // while the tail runs the hook-visible unfused path — in the same
        // fire. Pure decode (a predicate condition below) maps request rows
        // onto token rows 1:1, which is what lets one row count partition
        // both the QKV postprocess and the KV write.
        const int fast_rows = layer_hooks == nullptr
            ? R
            : std::min(static_cast<int>(layer_hooks->hook_free_prefix_rows), R);
        const bool fused_decode_qkv_post =
            use_fused_qkv &&
            // Device-window capture (peel_window_d): the branch must not
            // depend on THIS fire's split — the captured body emits both
            // Peel regions windowed, and a replay may put any row count
            // in either. Host mode keeps the fast_rows>0 economy.
            (peel_window_d != nullptr || fast_rows > 0) &&
            decode_fused_post_enabled() &&
            // A fused edge cannot be a merge point (§5.1): the fused decode
            // kernel writes V straight to the paged cache, so there is no
            // materialized v to accumulate the lora delta into before the
            // append. When the fire carries lora lanes, q/k/v must
            // materialize in the ws buffers so the CORRECTION lands before
            // kv_append; the unfused postprocess below is exactly that path.
            !has_lora &&
            is_pure_decode &&
            !has_custom_mask &&
            (!has_write_desc || (w_page_d != nullptr && w_off_d != nullptr)) &&
            native_bf16_kv_cache &&
            !head_dim_padded &&
            !fwd_cfg.use_qkv_bias &&
            fwd_cfg.use_qk_norm &&
            q_norm_is_per_head && k_norm_is_per_head &&
            fwd_cfg.rope_kind == RopeKind::Standard;
        // The rows the fused postprocess does NOT own. Zero on the classic
        // all-fused fire; N on the classic all-unfused one. Pure decode has
        // N == R, so the same count serves token- and request-indexed calls.
        const int unfused_tail_rows = fused_decode_qkv_post ? N - fast_rows : N;
        if (L == 0 && layer_hooks != nullptr && std::getenv("PIE_HOOK_PREFIX_TRACE")) {
            std::fprintf(stderr,
                         "[hook-prefix] R=%d fast_rows=%d fused=%d\n",
                         R, fast_rows, fused_decode_qkv_post ? 1 : 0);
        }
        if (use_fused_qkv) {
            ops::gemm_act_x_w(cublas.handle(),
                qkv_in, ops::WeightView(*layer.qkv_proj_fused),
                ws.qkv_fused.data(), N, Hq + 2 * Hk, H);
            if (fused_decode_qkv_post) {
                if (!rope_table_ready && !ws.rope_table.empty()) {
                    kernels::launch_rope_standard_table(
                        positions,
                        static_cast<float*>(ws.rope_table.data()),
                        N, d, cfg.rope_theta, stream);
                    rope_table =
                        static_cast<const float*>(ws.rope_table.data());
                    rope_table_ready = true;
                }
                if (peel_window_d != nullptr) {
                    // Device-window capture: both regions launch at the
                    // full-N grid and read the split from the device word,
                    // so the exec replays across row splits.
                    kernels::launch_qkv_decode_qk_norm_rope_write_kv_bf16_devwin(
                        ws.qkv_fused.data(),
                        ws.q.data(),
                        cache.k(L), cache.v(L),
                        layer.q_norm->data(), layer.k_norm->data(),
                        positions,
                        rope_table,
                        kv_page_indices, kv_page_indptr, kv_last_page_lens,
                        has_write_desc ? w_page_d : nullptr,
                        has_write_desc ? w_off_d : nullptr,
                        row_valid_d,
                        peel_window_d,
                        N, num_q_heads_local, num_kv_heads_local, d,
                        cache.page_size(), cache.hnd_layout(),
                        cfg.rope_theta, eps, stream);
                    kernels::launch_split_qkv_bf16_devwin(
                        ws.qkv_fused.data(),
                        ws.q.data(), ws.k.data(), ws.v.data(),
                        peel_window_d, N, Hq, Hk, stream);
                } else {
                    kernels::launch_qkv_decode_qk_norm_rope_write_kv_bf16(
                        ws.qkv_fused.data(),
                        ws.q.data(),
                        cache.k(L), cache.v(L),
                        layer.q_norm->data(), layer.k_norm->data(),
                        positions,
                        rope_table,
                        kv_page_indices, kv_page_indptr, kv_last_page_lens,
                        has_write_desc ? w_page_d : nullptr,
                        has_write_desc ? w_off_d : nullptr,
                        row_valid_d,
                        fast_rows, num_q_heads_local, num_kv_heads_local, d,
                        cache.page_size(), cache.hnd_layout(),
                        cfg.rope_theta, eps, stream);
                    if (unfused_tail_rows > 0) {
                        // The hook-visible tail: split into ws.q/k/v at their
                        // ABSOLUTE row offsets, so the full-N hook below still
                        // observes one contiguous query buffer.
                        kernels::launch_split_qkv_bf16(
                            bf16_row(ws.qkv_fused.data(), fast_rows,
                                     Hq + 2 * Hk),
                            bf16_row(ws.q.data(), fast_rows, Hq),
                            bf16_row(ws.k.data(), fast_rows, Hk),
                            bf16_row(ws.v.data(), fast_rows, Hk),
                            unfused_tail_rows, Hq, Hk, stream);
                    }
                }
            } else {
                kernels::launch_split_qkv_bf16(
                    ws.qkv_fused.data(),
                    ws.q.data(), ws.k.data(), ws.v.data(),
                    N, Hq, Hk, stream);
            }
        } else {
            ops::gemm_act_x_w(cublas.handle(),
                qkv_in, make_weight_view(layer.q_proj, layer.q_proj_quant),
                ws.q.data(), N, Hq, H);
            ops::gemm_act_x_w(cublas.handle(),
                qkv_in, make_weight_view(layer.k_proj, layer.k_proj_quant),
                ws.k.data(), N, Hk, H);
            ops::gemm_act_x_w(cublas.handle(),
                qkv_in, make_weight_view(layer.v_proj, layer.v_proj_quant),
                ws.v.data(), N, Hk, H);
        }

        // §5.1 CORRECTION, applied the moment the base q/v projections
        // exist in ws.q / ws.v and before anything consumes them (bias,
        // qk-norm, rope, KV append). The fused decode postprocess is
        // disabled above when `has_lora`, so both the fused-QKV and unfused
        // branches land here with fully materialized [N, ·] buffers.
        //
        // Scratch: xA^T borrows ws.gate. At this point in the layer, gate's
        // last write was the PREVIOUS layer's swiglu output, consumed by
        // that layer's down_proj GEMM; nothing reads it again before this
        // layer's MLP overwrites it. Its [max_tokens, I] extent bounds
        // both layouts the apply uses: every solo lane's [t, R] use
        // (rank <= I is validated at fire setup), and the grouped
        // lanes' packed per-lane regions (sum of spans x rank <= N * I,
        // verified against this alias's bound at fire setup).
        if (has_lora && lora_staged != nullptr) {
            lora_staged->apply(
                cublas.handle(), L, qkv_in, H, Hq, Hk,
                ws.q.data(), ws.v.data(), ws.gate.data());
        } else if (has_lora) {
            lora_state->apply(
                cublas.handle(), L, qkv_in, H, Hq, Hk,
                ws.q.data(), ws.v.data(), ws.gate.data());
        }

        if (!fused_decode_qkv_post && fwd_cfg.use_qkv_bias) {
            maybe_add_bias(ws.q.data(), layer.q_bias, N, Hq, stream);
            maybe_add_bias(ws.k.data(), layer.k_bias, N, Hk, stream);
            maybe_add_bias(ws.v.data(), layer.v_bias, N, Hk, stream);
        }

        // q_norm / k_norm: two conventions ship in the wild.
        //   * Per-head (Qwen3, OLMo-2 small, Gemma-3): weight shape
        //     `[head_dim]`. RMSNorm rolls each head's `d` channels
        //     independently — `num_rows = N*num_heads`, `hidden = d`.
        //   * Global (OLMo-2 7B+, OLMo-3): weight shape `[H_*]`
        //     (per-rank in TP). HF flattens [N, num_heads, d] → [N, H_*]
        //     and applies one RMSNorm with the full vector — `num_rows
        //     = N`, `hidden = num_heads * d`. Pre-rope behaviour
        //     differs from per-head by the shared scale across heads.
        // Dispatch by inspecting the bound q/k_norm shape; bind code
        // doesn't reshape, so the tensor's leading dim tells us.
        auto rmsnorm_qk = [&](void* x, const DeviceTensor* w,
                              int num_heads_local, int per_rank_H) {
            const bool global_norm = (w->shape().size() == 1 &&
                                      w->shape()[0] == per_rank_H);
            if (global_norm) {
                kernels::launch_rmsnorm_bf16(
                    x, w->data(), x, N, per_rank_H, eps, stream);
            } else {
                kernels::launch_rmsnorm_bf16(
                    x, w->data(), x,
                    N * num_heads_local, d, eps, stream);
            }
        };
        const bool fuse_qk_norm_rope =
            fwd_cfg.use_qk_norm &&
            q_norm_is_per_head && k_norm_is_per_head &&
            fwd_cfg.rope_kind == RopeKind::Standard;
        const bool use_mrope =
            fwd_cfg.rope_kind == RopeKind::MRopeInterleaved &&
            vision != nullptr && vision->mrope_positions != nullptr &&
            q_norm_is_per_head && k_norm_is_per_head;
        if (fused_decode_qkv_post) {
            // Rows [0, fast_rows): Q was normalized/rotated and K/V were
            // written directly to the paged cache by the fused decode
            // postprocess above. The hook-visible tail takes the same
            // per-head-norm + standard-rope transform the predicate
            // guaranteed, over its own rows only.
            if (peel_window_d != nullptr) {
                kernels::launch_qk_rmsnorm_rope_bf16_devwin(
                    ws.q.data(), ws.k.data(),
                    layer.q_norm->data(), layer.k_norm->data(),
                    positions,
                    peel_window_d, N,
                    num_q_heads_local, num_kv_heads_local, d,
                    cfg.rope_theta, eps, stream);
            } else if (unfused_tail_rows > 0) {
                kernels::launch_qk_rmsnorm_rope_bf16(
                    bf16_row(ws.q.data(), fast_rows, Hq),
                    bf16_row(ws.k.data(), fast_rows, Hk),
                    layer.q_norm->data(), layer.k_norm->data(),
                    positions + fast_rows,
                    unfused_tail_rows,
                    num_q_heads_local, num_kv_heads_local, d,
                    cfg.rope_theta, eps, stream);
            }
        } else if (use_mrope) {
            // Qwen3-VL: fused per-head q/k RMSNorm + interleaved 3-axis M-RoPE.
            kernels::launch_qk_rmsnorm_mrope_bf16(
                ws.q.data(), ws.k.data(),
                layer.q_norm->data(), layer.k_norm->data(),
                vision->mrope_positions,
                N, num_q_heads_local, num_kv_heads_local, d,
                cfg.rope_theta, eps,
                fwd_cfg.mrope_section_t, fwd_cfg.mrope_section_h,
                fwd_cfg.mrope_section_w, stream);
        } else if (fuse_qk_norm_rope) {
            kernels::launch_qk_rmsnorm_rope_bf16(
                ws.q.data(), ws.k.data(),
                layer.q_norm->data(), layer.k_norm->data(),
                positions, N, num_q_heads_local, num_kv_heads_local, d,
                cfg.rope_theta, eps, stream);
        } else {
            if (fwd_cfg.use_qk_norm && layer.q_norm) {
                rmsnorm_qk(ws.q.data(), layer.q_norm, num_q_heads_local, Hq);
            }
            if (fwd_cfg.use_qk_norm && layer.k_norm) {
                rmsnorm_qk(ws.k.data(), layer.k_norm, num_kv_heads_local, Hk);
            }

            apply_rope(fwd_cfg, cfg,
                       ws.q.data(), ws.k.data(), positions,
                       N, num_q_heads_local, num_kv_heads_local, d,
                       stream);
        }

        // Fires POST-rope (and post q/k-norm): the query a PTIR program
        // observes here is the one that actually enters attention, so an
        // observer scoring it against the cached keys -- which are stored
        // post-rope -- compares in the same space. Placing it on the raw
        // projection instead would silently mis-rank pages for Quest.
        // Re-seed to "keep everything" before the hook that may narrow it. A
        // layer whose program writes no mask must attend over its full page
        // list, and must never inherit the previous layer's selection.
        page_mask.begin_layer(stream);

        invoke_stage_hook(
            layer_hooks,
            StageHookPoint::OnAttnProj,
            ws.q.data(),
            static_cast<std::uint32_t>(N),
            static_cast<std::uint32_t>(Hq),
            static_cast<std::uint32_t>(L),
            stream,
            /*query_is_f32=*/false,
            {.mask_sink = page_mask.sink()});

        // Pad Q/K/V to `dk` when the model's head_dim isn't a flashinfer
        // dispatch value. The padded buffers are zero on the trailing
        // `dk - d` cols per head; QK·V dot products therefore equal the
        // unpadded ones, and `sm_scale = 1/sqrt(d)` keeps the softmax
        // scaled to the real head dim.
        const void* attn_q   = ws.q.data();
        const void* attn_k   = ws.k.data();
        const void* attn_v   = ws.v.data();
        void* attn_out_buf   = ws.attn_out.data();
        if (!fused_decode_qkv_post && head_dim_padded) {
            kernels::launch_pad_head_dim_bf16(
                ws.q.data(), ws.q_padded.data(),
                N, num_q_heads_local, d, dk, stream);
            kernels::launch_pad_head_dim_bf16(
                ws.k.data(), ws.k_padded.data(),
                N, num_kv_heads_local, d, dk, stream);
            kernels::launch_pad_head_dim_bf16(
                ws.v.data(), ws.v_padded.data(),
                N, num_kv_heads_local, d, dk, stream);
            attn_q = ws.q_padded.data();
            attn_k = ws.k_padded.data();
            attn_v = ws.v_padded.data();
            attn_out_buf = ws.attn_out_padded.data();
        }
        auto kv_view = cache.layer_view(L);
        if (fused_decode_qkv_post) {
            // Rows [0, fast_rows) were already written by
            // launch_qkv_decode_qk_norm_rope_write_kv_bf16; only the
            // hook-visible tail still needs its K/V appended.
            if (peel_window_d != nullptr) {
                if (has_write_desc) {
                    kernels::launch_write_kv_explicit_bf16_devwin(
                        kv_view,
                        ws.k.data(), ws.v.data(),
                        w_page_d, w_off_d,
                        peel_window_d, N, stream, row_valid_d);
                } else {
                    kernels::launch_write_kv_to_pages_bf16_devwin(
                        kv_view,
                        ws.k.data(), ws.v.data(),
                        qo_indptr, kv_page_indices, kv_page_indptr,
                        kv_last_page_lens,
                        peel_window_d, N, R, stream, row_valid_d);
                }
            } else if (unfused_tail_rows > 0) {
                if (has_write_desc) {
                    kernels::launch_write_kv_explicit_bf16(
                        kv_view,
                        bf16_row(ws.k.data(), fast_rows, Hk),
                        bf16_row(ws.v.data(), fast_rows, Hk),
                        w_page_d + fast_rows, w_off_d + fast_rows,
                        unfused_tail_rows, stream,
                        row_valid_d != nullptr ? row_valid_d + fast_rows
                                               : nullptr);
                } else {
                    kernels::launch_write_kv_to_pages(
                        kv_view,
                        ws.k.data(), ws.v.data(),
                        qo_indptr, kv_page_indices, kv_page_indptr,
                        kv_last_page_lens,
                        N, R, stream, row_valid_d,
                        /*first_token=*/fast_rows);
                }
            }
        } else if (has_write_desc) {
            // B2: explicit-descriptor KV write. Each query TOKEN writes its new
            // K/V into the program-supplied (physical page id `w_page_d[c]`,
            // offset `w_off_d[c]`) target — the WSlot/WOff lowering — rather than
            // re-deriving the position from the page table + last_page_len. Beam
            // fork/freeze correctness: a frozen fork's cell is not overwritten (a
            // sibling's mask hides it). The write count is the number of query
            // TOKENS N: a decode-shaped fire has N==R (one new token per lane); a
            // variable-length prompt PREFILL has one lane (R==1) but N>1 tokens,
            // so N (not R) cells must be written from `attn_k`/`attn_v` [N,·] and
            // the N-entry WSlot/WOff descriptor.
            kernels::launch_write_kv_explicit_bf16(
                kv_view,
                const_cast<void*>(attn_k), const_cast<void*>(attn_v),
                w_page_d, w_off_d, N, stream, row_valid_d);
        } else {
            kernels::launch_write_kv_to_pages(
                kv_view,
                const_cast<void*>(attn_k), const_cast<void*>(attn_v),
                qo_indptr, kv_page_indices, kv_page_indptr, kv_last_page_lens,
                N, R, stream, row_valid_d);
        }

        // Per-layer sliding-window dispatch (OLMo-3, Mistral). When
        // `per_layer_window_left` is empty we fall back to the global
        // `sliding_window`; that single value is broadcast to every
        // layer (used by Mistral / Gemma-2 single-mode and Phi-3).
        const int layer_window_left = runtime_window_left >= -1
            ? runtime_window_left
            : (!fwd_cfg.per_layer_window_left.empty() &&
             L < static_cast<int>(fwd_cfg.per_layer_window_left.size()))
               ? fwd_cfg.per_layer_window_left[L]
               : fwd_cfg.sliding_window;

        // Track B: observe the attention this layer is about to compute, when
        // the fire's PTIR programs read `AttnScore`. `LayerScoreCapture` is a
        // no-op otherwise, and the substitution below is the whole cost --
        // the scores come out of the kernel's own logits, so nothing is
        // recomputed and nothing can drift from what attention actually used.
        //
        // Only the plain decode path is capture-capable. The other branches
        // capture nothing, and the PTIR side then throws at the hook rather
        // than handing a program a buffer nobody wrote.
        // Track A: honour the page mask this layer's program wrote. The
        // selection is applied by *gathering the page table* -- FlashInfer
        // already takes the page list as a launch argument, so there is no
        // kernel change and no replan. The fire's own CSR is left untouched:
        // it remains the source of truth for the KV append and for `kv_len`,
        // and compacting it in place would corrupt the cache.
        const std::uint32_t* attn_page_indices = kv_page_indices;
        const std::uint32_t* attn_page_indptr = kv_page_indptr;
        const std::uint32_t* attn_last_page_lens = kv_last_page_lens;
        if (page_mask.written_for(static_cast<std::uint32_t>(L))) {
            // AC-4: a SPATIAL fire's hooked lanes ride the prefix decode
            // dispatch, which consumes the substituted views — the split
            // branch is a paged-decode consumer too.
            const bool spatial_decode_consumer =
                plan_state.spatial_mask_split >= 0 &&
                is_pure_decode && decode_plan != nullptr;
            if ((!use_decode_path && !spatial_decode_consumer) ||
                decode_plan == nullptr) {
                throw std::runtime_error(
                    "attn_page_mask was written but this layer does not take "
                    "the paged decode path, which is the only one whose page "
                    "list can be substituted");
            }
            // A split-KV plan derives its request/tile indices FROM the page
            // counts, so handing it a shorter list silently attends over the
            // wrong tiles. The static non-split plan does not, which is why
            // substitution is legal there and nowhere else.
            if (!ops::decode_plan_is_page_count_independent(*decode_plan)) {
                throw std::runtime_error(
                    "attn_page_mask requires a page-count-independent decode "
                    "plan; this fire planned split-KV");
            }
            page_mask.compact(
                kv_page_indices, kv_page_indptr, kv_last_page_lens,
                static_cast<std::uint32_t>(R), stream);
            attn_page_indices = page_mask.page_indices();
            attn_page_indptr = page_mask.page_indptr();
            attn_last_page_lens = page_mask.last_page_lens();
        }

        // Exactly one of these can be active. The prefill capture is offered
        // only on the branch its plan was built for -- the prepare hook is
        // what decided FA2-vs-SM90, so the body cannot opt in on its own --
        // and the decode capture stands down whenever the prefill one is
        // eligible, because both publish through the same `OnAttn` binding and
        // a second binding would shadow the first.
        const bool prefill_capture_eligible =
            use_prefill_score_path && layer_window_left < 0;
        model::LayerScoreCapture score_capture(
            layer_hooks,
            static_cast<std::uint32_t>(L),
            static_cast<std::uint32_t>(num_q_heads_local),
            /*capturable=*/layer_window_left < 0 && !prefill_capture_eligible,
            stream);
        model::LayerPrefillScoreCapture prefill_score_capture(
            layer_hooks,
            static_cast<std::uint32_t>(L),
            static_cast<std::uint32_t>(num_q_heads_local),
            plan_state.prefill_score_window,
            /*capturable=*/prefill_capture_eligible,
            stream);

        if (use_xqa_decode_path) {
            ops::launch_attention_xqa_decode_bf16_prepared(
                attn_q, kv_view.k_bf16_pages, kv_view.v_bf16_pages, attn_out_buf,
                R, num_q_heads_local, num_kv_heads_local, dk,
                cache.page_size(), plan_state.xqa_max_pages_per_seq,
                attn_ws, stream, sm_scale_override);
        } else if (use_prefill_decode_path) {
            const int num_pages_in_batch = kv_page_indptr_h[R];
            kernels::launch_dequant_kv_cache_layer_to_bf16_active(
                kv_view, kv_page_indices, num_pages_in_batch, stream);
            ops::dispatch_attention_flashinfer_prefill_bf16(
                *(prefill_plan_override != nullptr ? prefill_plan_override
                                                   : prefill_decode_plan),
                attn_q, kv_view.k_bf16_pages, kv_view.v_bf16_pages, attn_out_buf,
                qo_indptr, kv_page_indices, kv_page_indptr, kv_last_page_lens,
                attn_ws, stream, /*logits_soft_cap=*/0.f, sm_scale_override);
        } else if (use_decode_path) {
            if (score_capture.active()) {
                ops::dispatch_attention_flashinfer_decode_capture(
                    *decode_plan,
                    attn_q, kv_view, attn_out_buf,
                    attn_page_indices, attn_page_indptr, attn_last_page_lens,
                    attn_ws, stream,
                    score_capture.raw(), score_capture.indptr_d(),
                    layer_window_left,
                    /*logits_soft_cap=*/0.f, sm_scale_override);
                // Scores are indexed by the list attention actually used, so
                // the capture is published against the same (possibly
                // compacted) CSR. A program then reads exactly as many scores
                // as there were keys.
                score_capture.publish(
                    attn_page_indptr, attn_last_page_lens, cache.page_size());
            } else {
                ops::dispatch_attention_flashinfer_decode(
                    *decode_plan,
                    attn_q, kv_view, attn_out_buf,
                    attn_page_indices, attn_page_indptr, attn_last_page_lens,
                    attn_ws, stream, layer_window_left,
                    /*logits_soft_cap=*/0.f, sm_scale_override);
            }
        } else if (custom_mask_d) {
            const ops::PrefillPlanCache* mask_plan = is_pure_decode
                ? (plan_state.use_mask_decode_plan
                       ? plan_state.mask_decode_plan.get()
                       : nullptr)
                : (plan_state.use_prefill_plan ? prefill_plan : nullptr);
            if (mask_plan == nullptr) {
                throw std::runtime_error(
                    "custom attention mask has no prepared prefill plan");
            }
            // NS-2 (the spatial mask fire): decode dispatch over the
            // unmasked prefix, the custom kernel over the REBASED masked
            // suffix — two kernels, one fire, everything else full-N
            // shared. Mirrors the interpreter's split branch exactly.
            if (is_pure_decode && plan_state.spatial_mask_split >= 0 &&
                unmasked_prefix_rows != 0xffffffffu) {
                const int split = plan_state.spatial_mask_split;
                if (split != static_cast<int>(unmasked_prefix_rows)) {
                    throw std::runtime_error(
                        "spatial mask: the planned split and the prepared "
                        "split drifted");
                }
                // NO-DEMOTION (the user's directive): the two kernels of
                // this split have disjoint outputs and read-only-shared
                // inputs, so the SUFFIX custom dispatch overlaps the
                // prefix on the side stream (fork here, join before the
                // shared tail). Its plan lives in the dedicated
                // workspace, so the concurrent scratch is disjoint.
                // Stream-capture safe: the fork/join events become
                // graph dependencies, exactly the cross-stream capture
                // pattern, so the split-keyed execs replay the overlap.
                const bool side_on2 =
                    spatial_stream_enabled() && split > 0;
                cudaStream_t suffix_stream2 = stream;
                SpatialSideStream* ss2 = nullptr;
                if (side_on2) {
                    ss2 = &spatial_side_stream();
                    suffix_stream2 = ss2->stream;
                    CUDA_CHECK(cudaEventRecord(ss2->fork, stream));
                    CUDA_CHECK(cudaStreamWaitEvent(
                        ss2->stream, ss2->fork, 0));
                }
                if (split > 0) {
                    if (fwd_cfg.force_prefill_path) {
                        // The deployment's decode form is the plan-free
                        // prefill dispatch (GQA ratio outside the decode
                        // kernel's set): stage the PLAIN lanes' pages
                        // (beyond the split the host CSR may be a
                        // composed-envelope placeholder) and run it over
                        // the prefix — pure decode, so tokens == rows ==
                        // split and every CSR's `[0, split]` head is the
                        // prefix's truth.
                        kernels::launch_dequant_kv_cache_layer_to_bf16_active(
                            kv_view, kv_page_indices,
                            kv_page_indptr_h[split], stream);
                        ops::launch_attention_flashinfer_prefill(
                            attn_q, kv_view, attn_out_buf,
                            qo_indptr, kv_page_indices, kv_page_indptr,
                            kv_last_page_lens,
                            qo_indptr_h, kv_page_indptr_h,
                            split, split, num_q_heads_local, attn_ws,
                            stream, layer_window_left,
                            /*logits_soft_cap=*/0.f, sm_scale_override);
                    } else if (decode_plan == nullptr) {
                        throw std::runtime_error(
                            "spatial mask: split active but prepare built "
                            "no prefix decode plan");
                    } else {
                        // AC-4: the ATTN page views (hook-narrowed when
                        // sites ran, aliases of the raw CSRs otherwise)
                        // — hooked prefix lanes keep their page masks.
                        // hook×mask: the prefix decode IS the paged decode
                        // path, and the hook rows live in it (seriation
                        // puts masked rows in the suffix), so the score
                        // capture rides here exactly as in the unsplit
                        // decode arm — request ordinals start at row 0,
                        // identical indexing.
                        if (score_capture.active()) {
                            ops::dispatch_attention_flashinfer_decode_capture(
                                *decode_plan,
                                attn_q, kv_view, attn_out_buf,
                                attn_page_indices, attn_page_indptr,
                                attn_last_page_lens,
                                attn_ws, stream,
                                score_capture.raw(),
                                score_capture.indptr_d(),
                                layer_window_left,
                                /*logits_soft_cap=*/0.f, sm_scale_override);
                            score_capture.publish(
                                attn_page_indptr, attn_last_page_lens,
                                cache.page_size());
                        } else {
                            ops::dispatch_attention_flashinfer_decode(
                                *decode_plan,
                                attn_q, kv_view, attn_out_buf,
                                attn_page_indices, attn_page_indptr,
                                attn_last_page_lens,
                                attn_ws, stream, layer_window_left,
                                /*logits_soft_cap=*/0.f, sm_scale_override);
                        }
                    }
                }
                // BASE buffers + ABSOLUTE device CSR values at +split —
                // see the interpreter's split branch for why no rebasing.
                if (mask_suffix_qo_indptr_d == nullptr) {
                    throw std::runtime_error(
                        "spatial mask: suffix qo identity missing");
                }
                // Hybrid addressing, measured live: the kernel's q/o rows are
                // plan/qo[0]-relative (offset pointers + the identity qo), the
                // KV side reads the device CSR ABSOLUTELY (base indices +
                // +split indptr — the composed device truth, no host rebase).
                ops::dispatch_attention_flashinfer_prefill_custom(
                    *mask_plan,
                    bf16_row(attn_q, split, Hq), kv_view,
                    bf16_row(attn_out_buf, split, Hq),
                    mask_suffix_qo_indptr_d,
                    kv_page_indices,
                    kv_page_indptr + split,
                    kv_last_page_lens + split,
                    custom_mask_d, custom_mask_indptr_d + split,
                    spatial_suffix_ws(), suffix_stream2);
                if (side_on2) {
                    CUDA_CHECK(cudaEventRecord(ss2->join, ss2->stream));
                    CUDA_CHECK(cudaStreamWaitEvent(stream, ss2->join, 0));
                }
            } else if (!is_pure_decode &&
                       plan_state.spatial_mask_split >= 0 &&
                       plan_state.spatial_mask_row_split >= 0 &&
                       unmasked_prefix_rows != 0xffffffffu) {
                // THE MIXED FIRE (M-2+M-3): causal prefill dispatch over
                // the prefix (prefill + plain-decode rows, requests
                // [0, split_req), token rows [0, split_rows)) CONCURRENT
                // with the custom dispatch over the masked 1-token
                // suffix on the side stream. `mask_plan` here is the
                // PREFIX CAUSAL plan (the prefill slot); the suffix's
                // mask plan lives in its dedicated slot. Suffix
                // addressing is the decode-class hybrid verbatim —
                // identity qo, absolute device CSRs at +split_req, q/out
                // offsets at split_rows.
                const int split_req = plan_state.spatial_mask_split;
                const int split_rows = plan_state.spatial_mask_row_split;
                if (split_req !=
                    static_cast<int>(unmasked_prefix_rows)) {
                    throw std::runtime_error(
                        "spatial mask (mixed): the planned split and the "
                        "prepared split drifted");
                }
                if (mask_suffix_qo_indptr_d == nullptr) {
                    throw std::runtime_error(
                        "spatial mask (mixed): suffix qo identity missing");
                }
                const ops::PrefillPlanCache* suffix_plan =
                    plan_state.use_mask_decode_plan
                        ? plan_state.mask_decode_plan.get()
                        : nullptr;
                if (suffix_plan == nullptr) {
                    throw std::runtime_error(
                        "spatial mask (mixed): prepare built no suffix "
                        "mask plan");
                }
                const bool side_on = spatial_stream_enabled();
                // Diagnostic span timing (PIE_SPATIAL_STREAM_TIMING=1):
                // layer 0's fork->join wall, printed per fire — the
                // overlap evidence when no profiler is installed
                // (compare side=1 against PIE_SPATIAL_STREAM=0 runs).
                static const bool span_timing = [] {
                    const char* v =
                        std::getenv("PIE_SPATIAL_STREAM_TIMING");
                    return v != nullptr && v[0] == '1';
                }();
                static cudaEvent_t span_t0 = nullptr;
                static cudaEvent_t span_t1 = nullptr;
                const bool time_this = span_timing && L == 0;
                if (time_this && span_t0 == nullptr) {
                    CUDA_CHECK(cudaEventCreate(&span_t0));
                    CUDA_CHECK(cudaEventCreate(&span_t1));
                }
                if (time_this) {
                    CUDA_CHECK(cudaEventRecord(span_t0, stream));
                }
                cudaStream_t custom_stream = stream;
                SpatialSideStream* ss = nullptr;
                if (side_on) {
                    ss = &spatial_side_stream();
                    custom_stream = ss->stream;
                    CUDA_CHECK(cudaEventRecord(ss->fork, stream));
                    CUDA_CHECK(cudaStreamWaitEvent(
                        ss->stream, ss->fork, 0));
                }
                ops::dispatch_attention_flashinfer_prefill_custom(
                    *suffix_plan,
                    bf16_row(attn_q, split_rows, Hq), kv_view,
                    bf16_row(attn_out_buf, split_rows, Hq),
                    mask_suffix_qo_indptr_d,
                    kv_page_indices,
                    kv_page_indptr + split_req,
                    kv_last_page_lens + split_req,
                    custom_mask_d, custom_mask_indptr_d + split_req,
                    spatial_suffix_ws(), custom_stream);
                // The prefix causal dispatch on the main stream — the
                // bf16 view needs the prefix pages staged (no-op on
                // native-bf16 caches; the custom dispatch takes the
                // layer view whole and needs none).
                kernels::launch_dequant_kv_cache_layer_to_bf16_active(
                    kv_view, kv_page_indices,
                    kv_page_indptr_h[split_req], stream);
                ops::dispatch_attention_flashinfer_prefill_bf16(
                    *mask_plan,
                    attn_q, kv_view.k_bf16_pages, kv_view.v_bf16_pages,
                    attn_out_buf,
                    qo_indptr, kv_page_indices, kv_page_indptr,
                    kv_last_page_lens,
                    attn_ws, stream, /*logits_soft_cap=*/0.f,
                    sm_scale_override);
                // NO-DEMOTION: the plain-decode middle takes the DECODE
                // kernel (its own plan over requests [P, split_req),
                // kvpp-rebased; q/out at row qo[P] — pure-decode middle
                // so row == request there). Same-stream after the
                // causal launch (both read-only on q/KV; outputs
                // disjoint) — the async launches already overlap on
                // the device; the side stream stays the custom's.
                if (plan_state.mixed_mid_decode_plan &&
                    plan_state.mixed_mid_start >= 0) {
                    const int P = plan_state.mixed_mid_start;
                    const int mid_row =
                        static_cast<int>(qo_indptr_h[P]);
                    // The third lane: the middle's decode overlaps the
                    // causal on its OWN stream (its plan region in
                    // attn_ws is the decode family's — disjoint from
                    // the causal's prefill region; outputs disjoint by
                    // rows).
                    cudaStream_t mid_stream = stream;
                    if (side_on && ss != nullptr) {
                        mid_stream = ss->stream2;
                        CUDA_CHECK(cudaEventRecord(ss->fork2, stream));
                        CUDA_CHECK(cudaStreamWaitEvent(
                            ss->stream2, ss->fork2, 0));
                    }
                    ops::dispatch_attention_flashinfer_decode(
                        *plan_state.mixed_mid_decode_plan,
                        bf16_row(attn_q, mid_row, Hq), kv_view,
                        bf16_row(attn_out_buf, mid_row, Hq),
                        kv_page_indices,
                        kv_page_indptr + P,
                        kv_last_page_lens + P,
                        attn_ws, mid_stream, layer_window_left,
                        /*logits_soft_cap=*/0.f, sm_scale_override);
                    if (side_on && ss != nullptr) {
                        CUDA_CHECK(cudaEventRecord(
                            ss->join2, ss->stream2));
                        CUDA_CHECK(cudaStreamWaitEvent(
                            stream, ss->join2, 0));
                    }
                }
                if (side_on) {
                    CUDA_CHECK(cudaEventRecord(ss->join, ss->stream));
                    CUDA_CHECK(cudaStreamWaitEvent(stream, ss->join, 0));
                }
                if (time_this) {
                    CUDA_CHECK(cudaEventRecord(span_t1, stream));
                    CUDA_CHECK(cudaEventSynchronize(span_t1));
                    float ms = 0.f;
                    CUDA_CHECK(cudaEventElapsedTime(
                        &ms, span_t0, span_t1));
                    std::fprintf(
                        stderr,
                        "[spatial-stream] L0 attn span %.3f ms "
                        "(side=%d)\n",
                        ms, side_on ? 1 : 0);
                }
            } else {
                ops::dispatch_attention_flashinfer_prefill_custom(
                    *mask_plan,
                    attn_q, kv_view, attn_out_buf,
                    qo_indptr, kv_page_indices, kv_page_indptr,
                    kv_last_page_lens, custom_mask_d, custom_mask_indptr_d,
                    attn_ws, stream);
            }
        } else if (plan_state.use_prefill_plan && prefill_plan != nullptr) {
            const int num_pages_in_batch = kv_page_indptr_h[R];
            kernels::launch_dequant_kv_cache_layer_to_bf16_active(
                kv_view, kv_page_indices, num_pages_in_batch, stream);
            if (prefill_score_capture.active()) {
                ops::dispatch_attention_flashinfer_prefill_capture_bf16(
                    *prefill_plan,
                    attn_q, kv_view.k_bf16_pages, kv_view.v_bf16_pages,
                    attn_out_buf,
                    qo_indptr, kv_page_indices, kv_page_indptr,
                    kv_last_page_lens, attn_ws, stream,
                    prefill_score_capture.raw(),
                    prefill_score_capture.folded(),
                    prefill_score_capture.indptr_d(),
                    prefill_score_capture.window(),
                    /*logits_soft_cap=*/0.f, sm_scale_override);
                prefill_score_capture.publish();
            } else {
                ops::dispatch_attention_flashinfer_prefill_bf16(
                    *prefill_plan,
                    attn_q, kv_view.k_bf16_pages, kv_view.v_bf16_pages,
                    attn_out_buf,
                    qo_indptr, kv_page_indices, kv_page_indptr,
                    kv_last_page_lens,
                    attn_ws, stream, /*logits_soft_cap=*/0.f, sm_scale_override);
            }
        } else {
            ops::launch_attention_flashinfer_prefill(
                attn_q, kv_view, attn_out_buf,
                qo_indptr, kv_page_indices, kv_page_indptr, kv_last_page_lens,
                qo_indptr_h, kv_page_indptr_h,
                N, R, num_q_heads_local, attn_ws, stream, layer_window_left,
                /*logits_soft_cap=*/0.f, sm_scale_override);
        }
        invoke_stage_hook(
            layer_hooks,
            StageHookPoint::OnAttn,
            ws.q.data(),
            static_cast<std::uint32_t>(N),
            static_cast<std::uint32_t>(Hq),
            static_cast<std::uint32_t>(L),
            stream,
            /*query_is_f32=*/false,
            {.scores = score_capture.scores() != nullptr
                           ? score_capture.scores()
                           : prefill_score_capture.scores()});

        // Strip the trailing pad cols off the attention output before
        // it feeds the o_proj GEMM (which expects `[N, num_q*head_dim]`,
        // not `[N, num_q*head_dim_kernel]`).
        if (head_dim_padded) {
            kernels::launch_strip_head_dim_bf16(
                attn_out_buf, ws.attn_out.data(),
                N, num_q_heads_local, d, dk, stream);
        }

        bool have_mlp_norm = false;
        if (!post_norm) {
            // o_proj is row-parallel: each rank's GEMM produces a partial
            // [N, H] contribution. Single-GPU fuses it into y as a
            // residual-add (beta=1); under TP we go via a scratch
            // (ws.norm_x is free here — it held the QKV input before),
            // all-reduce the partials, then add to the residual.
            if (T == 1) {
                ops::gemm_act_x_w(cublas.handle(),
                    ws.attn_out.data(), make_weight_view(layer.o_proj, layer.o_proj_quant),
                    ws.y.data(), N, H, Hq, /*beta=*/1.f);
            } else {
                ops::gemm_act_x_w(cublas.handle(),
                    ws.attn_out.data(), make_weight_view(layer.o_proj, layer.o_proj_quant),
                    ws.norm_x.data(), N, H, Hq, /*beta=*/0.f);
                auto* fused_ar = tp->custom_all_reduce();
                if (fused_ar != nullptr &&
                    fused_ar->can_fuse_residual_rmsnorm(N, H, stream)) {
                    fused_ar->all_reduce_residual_rmsnorm_bf16(
                        ws.norm_x.data(), ws.y.data(), layer.mlp_norm->data(),
                        ws.norm_y.data(), N, H, eps, stream);
                } else {
                    tp->all_reduce_bf16_out(ws.norm_x.data(), ws.norm_y.data(),
                        static_cast<std::size_t>(N) * H, ncclSum, stream);
                    kernels::launch_residual_add_rmsnorm_bf16(
                        ws.y.data(), ws.norm_y.data(), layer.mlp_norm->data(),
                        ws.norm_y.data(), N, H, eps, stream);
                }
                have_mlp_norm = true;
            }
        } else {
            // Post-norm: o_proj writes to norm_x (a scratch we own here),
            // norm_attn(norm_x) → norm_y, then y += norm_y.
            ops::gemm_act_x_w(cublas.handle(),
                ws.attn_out.data(), make_weight_view(layer.o_proj, layer.o_proj_quant),
                ws.norm_x.data(), N, H, Hq, /*beta=*/0.f);
            if (T > 1) {
                tp->all_reduce_bf16(ws.norm_x.data(),
                    static_cast<std::size_t>(N) * H, ncclSum, stream);
            }
            kernels::launch_rmsnorm_bf16(
                ws.norm_x.data(), layer.attn_norm->data(), ws.norm_y.data(),
                N, H, eps, stream);
            kernels::launch_residual_add_bf16(
                ws.y.data(), ws.norm_y.data(), N * H, stream);
        }

        // MLP block.
        const void* mlp_in = ws.y.data();
        if (!post_norm) {
            if (!have_mlp_norm) {
                kernels::launch_rmsnorm_bf16(
                    ws.y.data(), layer.mlp_norm->data(), ws.norm_y.data(),
                    N, H, eps, stream);
            }
            mlp_in = ws.norm_y.data();
        }

        // gate + up: same fused-vs-unfused dispatch as QKV.
        const bool use_fused_gu = (layer.gate_up_proj_fused != nullptr) &&
                                  !ws.gate_up_fused.empty();
        if (use_fused_gu) {
            ops::gemm_act_x_w(cublas.handle(),
                mlp_in, ops::WeightView(*layer.gate_up_proj_fused),
                ws.gate_up_fused.data(), N, 2 * I, H);
            kernels::launch_chunked_swiglu_bf16(
                ws.gate_up_fused.data(), ws.gate.data(), N, I, stream);
        } else {
            ops::gemm_act_x_w(cublas.handle(),
                mlp_in, make_weight_view(layer.gate_proj, layer.gate_proj_quant),
                ws.gate.data(), N, I, H);
            ops::gemm_act_x_w(cublas.handle(),
                mlp_in, make_weight_view(layer.up_proj, layer.up_proj_quant),
                ws.up.data(),   N, I, H);
            kernels::launch_swiglu_bf16(
                ws.gate.data(), ws.up.data(), ws.gate.data(),
                N * I, stream);
        }

        if (!post_norm) {
            // down_proj is row-parallel: same all-reduce + residual-add
            // dance as o_proj. ws.norm_x is free here (it last held the
            // mlp pre-norm input on the post-norm path; on pre-norm it
            // hasn't been touched since QKV).
            if (T == 1) {
                ops::gemm_act_x_w(cublas.handle(),
                    ws.gate.data(), make_weight_view(layer.down_proj, layer.down_proj_quant),
                    ws.y.data(), N, H, I, /*beta=*/1.f);
            } else {
                ops::gemm_act_x_w(cublas.handle(),
                    ws.gate.data(), make_weight_view(layer.down_proj, layer.down_proj_quant),
                    ws.norm_x.data(), N, H, I, /*beta=*/0.f);
                auto* fused_ar = tp->custom_all_reduce();
                if (fused_ar != nullptr &&
                    fused_ar->can_fuse_residual_rmsnorm(N, H, stream)) {
                    if (L + 1 < cfg.num_hidden_layers) {
                        fused_ar->all_reduce_residual_rmsnorm_bf16(
                            ws.norm_x.data(), ws.y.data(),
                            w.layers[L + 1].attn_norm->data(),
                            ws.norm_x.data(), N, H, eps, stream);
                        have_next_attn_norm = true;
                    } else {
                        fused_ar->all_reduce_residual_rmsnorm_bf16(
                            ws.norm_x.data(), ws.y.data(),
                            w.final_norm->data(),
                            ws.norm_y.data(), N, H, eps, stream);
                        have_final_norm = true;
                        final_norm_buf = ws.norm_y.data();
                    }
                } else {
                    tp->all_reduce_bf16_out(ws.norm_x.data(), ws.norm_y.data(),
                        static_cast<std::size_t>(N) * H, ncclSum, stream);
                    kernels::launch_residual_add_bf16(
                        ws.y.data(), ws.norm_y.data(), N * H, stream);
                }
            }
        } else {
            // Post-norm MLP: down_proj → norm_x scratch, norm_mlp, += y.
            ops::gemm_act_x_w(cublas.handle(),
                ws.gate.data(), make_weight_view(layer.down_proj, layer.down_proj_quant),
                ws.norm_x.data(), N, H, I, /*beta=*/0.f);
            if (T > 1) {
                tp->all_reduce_bf16(ws.norm_x.data(),
                    static_cast<std::size_t>(N) * H, ncclSum, stream);
            }
            kernels::launch_rmsnorm_bf16(
                ws.norm_x.data(), layer.mlp_norm->data(), ws.norm_y.data(),
                N, H, eps, stream);
            kernels::launch_residual_add_bf16(
                ws.y.data(), ws.norm_y.data(), N * H, stream);
        }

        // Qwen3-VL DeepStack: after decoder layers 0..num_deepstack-1, add the
        // matching deepstack merger output into the hidden state. The scatter
        // zeroed the scratch and wrote only image rows, so a whole-tensor
        // residual-add leaves text rows unchanged (HF `_deepstack_process`).
        if (vision != nullptr && vision->deepstack_scratch != nullptr &&
            L < vision->num_deepstack) {
            const auto* ds = static_cast<const std::uint16_t*>(
                                 vision->deepstack_scratch) +
                             static_cast<std::size_t>(L) * N * H;
            kernels::launch_residual_add_bf16(
                ws.y.data(), ds, static_cast<std::size_t>(N) * H, stream);
        }
    };
    // ④ tier 1 (Act-2 order): hook fires band. The frame gate arms bands
    // only when every hooked region is FULL-DEPTH in the plain prefix —
    // inside [0, band_rows[j]) at every banded layer — so the per-layer
    // hook invocations inside run_layer cover exactly the hook rows'
    // planned depth, and the score capture rides each band's own plan.
    const bool bands_runnable =
        plan_state.depth_band_count >= 1 && is_pure_decode &&
        !has_custom_mask &&
        // A page-mask-writing hook substitutes the fire's page table at
        // full R on the paged decode path; the banded layers run attention
        // at live<R rows, and the compaction and the fire then disagree on
        // request count (caught by the soak's h2o lane). Observation-only
        // hooks band; Track-B hooks keep the pre-band servers.
        (hooks == nullptr || !hooks->wants_page_mask) &&
        (use_decode_path || use_prefill_decode_path ||
         fwd_cfg.force_prefill_path) &&
        layer_bound == cfg.num_hidden_layers;
    if (plan_state.depth_band_count >= 1 && !bands_runnable &&
        std::getenv("PIE_SPATIAL_MASK_TRACE") != nullptr) {
        // Degrade loudly-quietly: the fire runs full depth (today's
        // demotion) rather than dying — deployments the banded walk
        // does not serve yet (XQA) or shapes the frame gate should
        // have declined.
        std::fprintf(stderr, "[depth-bands] DECLINE R=%d\n", R);
    }
    if (bands_runnable) {
        // ④ Act 1 (banded depth): distinct-k bands, deepest-first. At
        // any layer the live rows are the prefix [0, band_rows[j]) of
        // the interval containing it (the seriation's deepest-first
        // invariant); frozen rows ride to the one tail exactly as the
        // S-2 union's suffix does. band_rows[j] == 0 ends the walk —
        // nothing lives past that band (the all-truncated fire's
        // bonus: layers past the deepest k never launch).
        const int m = static_cast<int>(plan_state.depth_band_count);
        if (std::getenv("PIE_SPATIAL_MASK_TRACE") != nullptr) {
            std::fprintf(stderr, "[depth-bands] R=%d m=%d", R, m);
            for (int j = 0; j < m; ++j) {
                std::fprintf(
                    stderr, " (k=%u rows=%u)",
                    plan_state.depth_band_k[static_cast<std::size_t>(j)],
                    plan_state
                        .depth_band_rows[static_cast<std::size_t>(j)]);
            }
            std::fprintf(stderr, "\n");
        }
        const int k_min = static_cast<int>(
            plan_state.depth_band_k[static_cast<std::size_t>(m - 1)]);
        for (int L = 0; L < k_min; ++L) {
            run_layer(L, N, R, decode_plan, attn_ws);
        }
        for (int j = m - 1; j >= 0; --j) {
            const int live = static_cast<int>(
                plan_state.depth_band_rows[static_cast<std::size_t>(j)]);
            if (live == 0) break;
            const int from = static_cast<int>(
                plan_state.depth_band_k[static_cast<std::size_t>(j)]);
            const int to =
                j == 0 ? cfg.num_hidden_layers
                       : static_cast<int>(plan_state.depth_band_k
                             [static_cast<std::size_t>(j - 1)]);
            const ops::DecodePlanCache* band_plan =
                plan_state.depth_band_plans[static_cast<std::size_t>(j)]
                    .get();
            const ops::PrefillPlanCache* band_prefill =
                plan_state
                    .depth_band_prefill_plans[static_cast<std::size_t>(j)]
                    .get();
            // ④ envelope banding: the XQA arm is PLAN-FREE — it reads
            // the fire's staged device CSRs and takes the row count as
            // a parameter, so the band's prefix call needs neither a
            // band plan nor a separate workspace (the per-band
            // workspace isolation exists for flashinfer plan state; the
            // XQA staging in the fire's own workspace is read-only to
            // the launches and the band call must see it).
            // Plan-free deployments (XQA, and force_prefill's plan-free
            // prefill dispatch) band by the prefix row count alone.
            const bool plan_free_bands =
                use_xqa_decode_path ||
                (fwd_cfg.force_prefill_path && !use_prefill_decode_path);
            if (!plan_free_bands &&
                (use_prefill_decode_path ? band_prefill == nullptr
                                         : band_plan == nullptr)) {
                throw std::runtime_error(
                    "depth bands: band active but prepare built no "
                    "plan for it");
            }
            for (int L = from; L < to; ++L) {
                run_layer(L, live, live, band_plan,
                          plan_free_bands ? attn_ws : depth_band_ws(j),
                          band_prefill);
            }
        }
    } else if (full_depth_rows != 0xffffffffu &&
        (has_custom_mask || hooks != nullptr ||
         // Deployments without the decode kernel (force_prefill /
         // prefill_decode_plan) cannot run the WINDOWED range-2 (its
         // prefix dispatch is the decode plan) — the stash form serves
         // them instead, with m_start = R (no full-depth suffix: the
         // stash covers [t_start, R), layers [k, L) run full-N, and
         // the discarded middle is the whole truncated tail).
         !use_decode_path || use_prefill_decode_path)) {
        // AC-1 (mask x depth), the stash/restore form: order is
        // [plain | truncated | masked], so the full-depth rows are
        // non-contiguous {[0, t_start) ∪ [m_start, N)}. Rather than
        // window every kernel, layers [k, L) run FULL-N — the
        // truncated middle computes discarded garbage (its [k, L) KV
        // slabs are dead weight its own re-runs never read) — with the
        // residual stream's truncated rows STASHED at layer k and
        // RESTORED before the tail, so the one tail reads layer-k
        // hidden for the truncated rows and layer-L for everyone else.
        const int t_start = static_cast<int>(full_depth_rows);
        // The truncated middle ends where the first full-depth suffix
        // block begins — the hooked block when present (its start rides
        // the hook-free-prefix word via fast_rows... the mask split is
        // the v0 anchor; hooked+depth composition keeps m_start = the
        // mask word since hooked lanes sort between).
        // AC-5 anchor refinement: the middle ends at the first
        // mask/hook block AFTER the truncated rows — the TRAILING
        // tail. The seriation may place hook (or masked) FULL-DEPTH
        // rows in the prefix, before the truncated block; those rows
        // are part of [0, t_start) and must not drag m_start below it
        // (the derivation already counted them into the split — see
        // region_plans.hpp block_ok). A marker at or before t_start
        // is a prefix block, not the tail.
        int m_start = R;
        if (has_custom_mask && plan_state.spatial_mask_split >= 0 &&
            plan_state.spatial_mask_split > t_start) {
            m_start =
                std::min<int>(m_start, plan_state.spatial_mask_split);
        }
        if (hooks != nullptr) {
            const int hook_start =
                static_cast<int>(hooks->hook_free_prefix_rows);
            if (hook_start >= 0 && hook_start <= R &&
                hook_start > t_start) {
                m_start = std::min<int>(m_start, hook_start);
            }
        }
        // t_start == 0 is legal: no plain block, the truncated middle
        // starts at row 0 ([truncated | masked]).
        if (layer_bound >= cfg.num_hidden_layers || t_start < 0 ||
            m_start <= t_start || m_start > R || !is_pure_decode ||
            (has_custom_mask && plan_state.spatial_mask_split < 0)) {
            throw std::runtime_error(
                "depth union (masked): planned shape and prepared "
                "state drifted");
        }
        static DeviceBuffer<std::uint8_t> stash;
        const std::size_t bytes =
            static_cast<std::size_t>(m_start - t_start) * H * 2;
        if (stash.size() < bytes) {
            stash = DeviceBuffer<std::uint8_t>(bytes);
        }
        for (int L = 0; L < layer_bound; ++L) {
            run_layer(L, N, R, decode_plan, attn_ws);
        }
        CUDA_CHECK(cudaMemcpyAsync(
            stash.data(), bf16_row(ws.y.data(), t_start, H), bytes,
            cudaMemcpyDeviceToDevice, stream));
        for (int L = layer_bound; L < cfg.num_hidden_layers; ++L) {
            run_layer(L, N, R, decode_plan, attn_ws);
        }
        CUDA_CHECK(cudaMemcpyAsync(
            bf16_row(ws.y.data(), t_start, H), stash.data(), bytes,
            cudaMemcpyDeviceToDevice, stream));
    } else if (full_depth_rows != 0xffffffffu) {
        // The depth union (S-2): layers [0, k) run every row, layers
        // [k, L) run the full-depth prefix only, and the unchanged
        // full-N tail below is BOTH heads (the suffix rows' hidden
        // state froze at layer k — the logit-lens head; the one-tail
        // design, north-star-dsl.md). Guards are loud: the engine
        // plans this shape only for plain pure-decode fires.
        const int split = static_cast<int>(full_depth_rows);
        const ops::DecodePlanCache* prefix_plan =
            plan_state.depth_prefix_decode_plan
                ? plan_state.depth_prefix_decode_plan.get()
                : nullptr;
        if (layer_bound >= cfg.num_hidden_layers || split <= 0 ||
            split >= R || !is_pure_decode || has_custom_mask ||
            !use_decode_path || use_prefill_decode_path ||
            use_xqa_decode_path) {
            throw std::runtime_error(
                "depth union: planned split reached an unsupported "
                "fire shape (engine/driver gate drift)");
        }
        if (prefix_plan == nullptr) {
            throw std::runtime_error(
                "depth union: split active but prepare built no "
                "prefix decode plan");
        }
        if (std::getenv("PIE_SPATIAL_MASK_TRACE") != nullptr) {
            std::fprintf(stderr,
                         "[depth-union] R=%d split=%d k=%d\n",
                         R, split, layer_bound);
        }
        for (int L = 0; L < layer_bound; ++L) {
            run_layer(L, N, R, decode_plan, attn_ws);
        }
        for (int L = layer_bound; L < cfg.num_hidden_layers; ++L) {
            run_layer(L, split, split, prefix_plan,
                      spatial_suffix_attn_ws());
        }
    } else {
        for (int L = 0; L < layer_bound; ++L) {
            run_layer(L, N, R, decode_plan, attn_ws);
        }
    }

    if (fwd_cfg.emit_logits) {
        const bool compact_logits =
            logit_row_indices_d != nullptr && num_logit_rows > 0 &&
            num_logit_rows < N;
        const void* lm_head_input =
            have_final_norm ? final_norm_buf : ws.norm_y.data();
        int lm_head_rows = N;
        if (compact_logits) {
            if (have_final_norm) {
                kernels::launch_gather_bf16_rows(
                    static_cast<const std::uint16_t*>(final_norm_buf),
                    logit_row_indices_d,
                    static_cast<std::uint16_t*>(ws.norm_x.data()),
                    num_logit_rows, H, stream);
                lm_head_input = ws.norm_x.data();
            } else {
                kernels::launch_gather_bf16_rows(
                    static_cast<const std::uint16_t*>(ws.y.data()),
                    logit_row_indices_d,
                    static_cast<std::uint16_t*>(ws.norm_x.data()),
                    num_logit_rows, H, stream);
                kernels::launch_rmsnorm_bf16(
                    ws.norm_x.data(), w.final_norm->data(), ws.norm_y.data(),
                    num_logit_rows, H, eps, stream);
                lm_head_input = ws.norm_y.data();
            }
            lm_head_rows = num_logit_rows;
        } else {
            // Full [N,V] emit (the PTIR `intrinsics::logits()` path). ALWAYS
            // recompute the final norm from `ws.y` here — do NOT fall through to
            // the `have_final_norm ? final_norm_buf` default. `final_norm_buf`
            // (the TP fused-AR's `ws.norm_y`) can be stale/overwritten by the
            // time the full-logits emit runs, so a PTIR stage-runner reading
            // these logits saw garbage (§6.2: 19148 vs 14582). `ws.y` is
            // the full pre-norm hidden (the fused-AR updates it in place via
            // `residual_inout`), so `rmsnorm(ws.y)` reproduces the correct
            // final-normed activation.
            kernels::launch_rmsnorm_bf16(
                ws.y.data(), w.final_norm->data(), ws.norm_y.data(),
                N, H, eps, stream);
            lm_head_input = ws.norm_y.data();
        }
        // Fused LM head + greedy argmax: the vocabulary is reduced slab by
        // slab as it is produced, so the [rows, vocab] logits never exist
        // (§20.37). The slab scratch is carved out of `ws.logits` -- by
        // construction the buffer this path is not filling -- so the fused
        // route allocates nothing.
        //
        // There is deliberately no quiet fallback. By the time the forward
        // runs, `prepare_step` has already put `kGvFusedArgmax` in the graph
        // key and `settle_step` will hand the epilogue `ws.sampled_tokens`
        // whatever happens here; materializing logits instead would leave the
        // epilogue publishing uninitialised memory as token ids. Every
        // condition below is established before the fire is admitted
        // (`ModelCapabilities::supports_fused_lm_head_argmax` for the weight,
        // and `lm_head_rows <= workspace_logits_rows` for the shapes), so this
        // is an assertion, not a branch.
        const int chunk = fwd_cfg.logits_argmax_chunk_tokens;
        if (chunk > 0) {
            const auto rows = static_cast<std::size_t>(lm_head_rows);
            const std::size_t accum = rows * kernels::kArgmaxAccumSlots;
            if (ws.sampled_tokens.numel() < rows ||
                ws.argmax_acc_val.numel() < accum ||
                ws.argmax_acc_idx.numel() < accum) {
                throw std::runtime_error(
                    "fused lm_head argmax: workspace holds fewer rows than "
                    "this fire samples");
            }
            if (ops::lm_head_argmax_slab_bytes(lm_head_rows, V, chunk) >
                ws.logits.nbytes()) {
                throw std::runtime_error(
                    "fused lm_head argmax: vocabulary slab does not fit the "
                    "logits arena");
            }
            if (!ops::lm_head_argmax_chunked(
                    cublas.handle(), lm_head_input, *w.lm_head,
                    static_cast<std::int32_t*>(ws.sampled_tokens.data()),
                    ws.logits.data(),
                    static_cast<float*>(ws.argmax_acc_val.data()),
                    static_cast<std::int32_t*>(ws.argmax_acc_idx.data()),
                    lm_head_rows, V, H, chunk)) {
                throw std::runtime_error(
                    "fused lm_head argmax: lm_head weight is not dense BF16, "
                    "yet the model advertised the capability");
            }
        } else {
            ops::gemm_act_x_w(cublas.handle(),
                lm_head_input, *w.lm_head, ws.logits.data(),
                lm_head_rows, V, H);
        }
    }
}

RopeKind rope_kind_from_hf_config(const HfConfig& hf) {
    using RopeScaling = HfConfig::RopeScaling;
    switch (hf.rope_scaling_kind) {
    case RopeScaling::Llama3:
        return RopeKind::YaRN;
    case RopeScaling::OriginalYaRN:
        return RopeKind::YaRNOriginal;
    case RopeScaling::None:
        return RopeKind::Standard;
    }
    return RopeKind::Standard;
}

void apply_rope_config(LlamaLikeForwardCfg& fwd_cfg, const HfConfig& hf) {
    fwd_cfg.rope_kind                  = rope_kind_from_hf_config(hf);
    fwd_cfg.yarn_factor                = hf.rope_factor;
    fwd_cfg.yarn_low_freq_factor       = hf.rope_low_freq_factor;
    fwd_cfg.yarn_high_freq_factor      = hf.rope_high_freq_factor;
    fwd_cfg.yarn_original_max_position = hf.rope_original_max_position;
    fwd_cfg.yarn_beta_fast             = hf.rope_beta_fast;
    fwd_cfg.yarn_beta_slow             = hf.rope_beta_slow;
    fwd_cfg.yarn_attention_factor      = hf.rope_attention_factor;
}


std::uint64_t llama_like_lora_stage(
    LlamaLikePlanState& state,
    Workspace& ws,
    const LoraTable* lora,
    const HfConfig& cfg,
    const LlamaLikeForwardCfg& fwd_cfg,
    int total_tokens,
    cudaStream_t stream)
{
    if (lora == nullptr || !lora->usable()) {
        state.lora_staged.reset();
        state.lora_staged_table = nullptr;
        return 0;
    }
    const int H = cfg.hidden_size;
    const int Hq = cfg.num_attention_heads * cfg.head_dim;
    const int Hk = cfg.num_key_value_heads * cfg.head_dim;
    const int I = cfg.intermediate_size;
    const int T = fwd_cfg.tp_size > 0 ? fwd_cfg.tp_size : 1;
    const bool post_norm =
        fwd_cfg.norm_placement == NormPlacement::Post;
    state.lora_staged = std::make_unique<LoraFireStateHandle>(
        *lora, cfg, total_tokens, H, Hq, Hk, I, T, stream, ws,
        post_norm ? static_cast<const void*>(ws.y.data())
                  : static_cast<const void*>(ws.norm_x.data()),
        ws.q.data(), ws.v.data(), ws.gate.data());
    state.lora_staged_table = lora;

    // Fingerprint: everything a captured lora body bakes. splitmix mix.
    auto mix = [](std::uint64_t x) {
        x += 0x9e3779b97f4a7c15ull;
        x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
        x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
        return x ^ (x >> 31);
    };
    std::uint64_t h = mix(static_cast<std::uint64_t>(lora->count));
    h ^= mix(static_cast<std::uint64_t>(total_tokens));
    h ^= mix(lora_grouped_enabled() ? 1u : 2u);
    h ^= mix(reinterpret_cast<std::uintptr_t>(ws.lora_arena.buf.data()));
    for (std::uint32_t i = 0; i < lora->count; ++i) {
        const LoraLaneView& v = lora->lanes[i];
        h ^= mix(v.rank) + mix(v.d_in) * 3 + mix(v.d_out) * 5;
        h ^= mix(v.sites_bits) + mix(v.token_start) * 7 +
             mix(v.token_count) * 11;
        h ^= mix(reinterpret_cast<std::uintptr_t>(v.a));
        h ^= mix(reinterpret_cast<std::uintptr_t>(v.b));
    }
    return h == 0 ? 1 : h;
}

// ── LoraFireStateHandle: the opaque fire-scoped staging the declared
// executor shares with this body (llama_like.hpp) ─────────────────────
LoraFireStateHandle::LoraFireStateHandle(
    const LoraTable& table,
    const HfConfig& cfg,
    int total_tokens,
    int hidden,
    int q_width,
    int kv_width,
    int intermediate,
    int tp_size,
    cudaStream_t stream,
    Workspace& ws,
    const void* qkv_in,
    void* q_out,
    void* v_out,
    void* xa_scratch)
    : impl_(new LoraFireState(
          table, cfg, total_tokens, hidden, q_width, kv_width,
          intermediate, tp_size, stream, ws.lora_arena,
          qkv_in, q_out, v_out, xa_scratch)) {}

LoraFireStateHandle::~LoraFireStateHandle() {
    delete static_cast<LoraFireState*>(impl_);
}

void LoraFireStateHandle::apply(
    cublasHandle_t handle,
    int layer,
    const void* qkv_in,
    int hidden,
    int q_width,
    int kv_width,
    void* q_out,
    void* v_out,
    void* xa_scratch) const {
    static_cast<const LoraFireState*>(impl_)->apply(
        handle, layer, qkv_in, hidden, q_width, kv_width,
        q_out, v_out, xa_scratch);
}

std::string LoraFireStateHandle::grouping_desc() const {
    return static_cast<const LoraFireState*>(impl_)->grouping_desc();
}

}  // namespace pie_cuda_driver::model
