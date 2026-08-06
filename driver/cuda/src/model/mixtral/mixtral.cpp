#include <cstring>
#include "model/mixtral/mixtral.hpp"
#include "model/stage_hooks.hpp"

#include <cstdlib>
#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <array>
#include <vector>

#include <cuda_runtime.h>

#include "cuda_check.hpp"
#include "device_buffer.hpp"
#include "kernels/add_bias.hpp"
#include "kernels/attn_sink.hpp"
#include "kernels/dequant_fp4.hpp"
#include "kernels/dequant_wna16.hpp"
#include "kernels/deinterleave.hpp"
#include "kernels/embed.hpp"
#include "kernels/gather_rows.hpp"
#include "kernels/residual_add.hpp"
#include "kernels/kv_paged.hpp"
#include "kernels/moe_dispatch.hpp"
#ifdef PIE_CUDA_HAS_MARLIN_MOE
  #include "marlin_moe_wrapper.hpp"
#endif
#include "kernels/rmsnorm.hpp"
#include "kernels/rope.hpp"
#include <cstdio>
#include "kernels/swiglu.hpp"
#include "kernels/topk_softmax.hpp"
#include "ops/gemm.hpp"
#include "ops/attention_flashinfer.hpp"
#include "ops/attention_flashinfer_hopper.hpp"

namespace pie_cuda_driver::model {

namespace {

// ── Phase profiler (PIE_MIXTRAL_PROFILE=1) ────────────────────────────────
// Mixtral/GPT-OSS had no per-stage timing, so "the MoE is slow" could not be
// resolved into which stage is slow. Same shape as the Kimi/Nemotron/Qwen3.5
// profilers: CUDA events around each stage, accumulated per fire.
// Stage timings come from a POOL of events resolved with exactly one sync, at
// teardown. Recording a pair and synchronising on it per stage per layer put a
// full host/device round trip inside every interval it was trying to measure:
// a 24-layer fire took ~96 of those samples, each carrying its own sync
// latency, and the totals came out well above the wall clock they were meant
// to explain. Nothing is read until the fire is over.
struct MixtralPhaseProfile {
    bool enabled = false;
    cudaEvent_t a{}, b{};
    double attn = 0, router = 0, moe_gate_up = 0, moe_down = 0;
    double o_proj = 0, epilogue = 0;
    int last_N = 0;

    // Intervals may nest, so opens are held on a stack and each close pairs
    // with the most recent open.
    struct Open { double* dst; std::size_t ev; };
    std::vector<cudaEvent_t> pool;
    std::vector<Open> stack;
    std::vector<std::array<std::size_t, 2>> spans;
    std::vector<double*> span_dst;
    std::size_t used = 0;

    // Recording an event on a CAPTURING stream does not measure the replay, it
    // corrupts the capture. Refuse instead of producing a graph that fails to
    // instantiate -- profiling wants PIE_CUDA_DISABLE_GRAPH_CAPTURE=1.
    static bool capturing(cudaStream_t stream) {
        cudaStreamCaptureStatus st = cudaStreamCaptureStatusNone;
        if (cudaStreamIsCapturing(stream, &st) != cudaSuccess) return true;
        return st != cudaStreamCaptureStatusNone;
    }
    std::size_t claim(cudaStream_t stream) {
        if (used == pool.size()) {
            cudaEvent_t e{};
            cudaEventCreate(&e);
            pool.push_back(e);
        }
        const std::size_t i = used++;
        last_stream = stream;
        CUDA_CHECK(cudaEventRecord(pool[i], stream));
        return i;
    }
    void open(double* dst, cudaStream_t stream) {
        if (capturing(stream)) return;
        stack.push_back({dst, claim(stream)});
    }
    void close(cudaStream_t stream) {
        if (capturing(stream) || stack.empty()) return;
        const Open o = stack.back();
        stack.pop_back();
        spans.push_back({o.ev, claim(stream)});
        span_dst.push_back(o.dst);
    }
    cudaStream_t last_stream{};
    void resolve() {
        // A fire has several exits, so anything still open at teardown is
        // closed here rather than at each of them.
        while (!stack.empty() && last_stream) close(last_stream);
        if (used == 0) return;
        CUDA_CHECK(cudaEventSynchronize(pool[used - 1]));
        for (std::size_t i = 0; i < spans.size(); ++i) {
            float ms = 0.f;
            CUDA_CHECK(cudaEventElapsedTime(
                &ms, pool[spans[i][0]], pool[spans[i][1]]));
            *span_dst[i] += static_cast<double>(ms);
        }
        used = 0;
        stack.clear();
        spans.clear();
        span_dst.clear();
    }

    // Printed from the destructor because `mixtral_forward_paged` has several
    // returns (compact-logits epilogue, plain epilogue) and a stage total that
    // only some of them reach would be silently wrong.
    ~MixtralPhaseProfile() {
        if (!enabled) return;
        resolve();
        std::fprintf(stderr,
            // `sum` is what the stages below account for; the rest of the
            // fire -- the MoE dispatch, the glu, the scatter, the norms and
            // residuals around them -- is the wall clock minus this, and at
            // gpt-oss's decode shape that remainder is the largest single
            // item left.
            "[MX-PROF] N=%d attn=%.3f o_proj=%.3f router=%.3f "
            "moe_gate_up=%.3f moe_down=%.3f | sum=%.3f ms\n",
            last_N, attn, o_proj, router, moe_gate_up, moe_down,
            attn + o_proj + router + moe_gate_up + moe_down + epilogue);
        cudaEventDestroy(a);
        cudaEventDestroy(b);
        for (cudaEvent_t e : pool) cudaEventDestroy(e);
    }
};

// Whether the expert-grouped gate/up kernel is worth its overhead.
//
// Grouping trades a bigger grid (one block per expert x row slab, most of them
// empty at small batches) for weight reuse across the tokens that picked the
// same expert. The reuse factor is routes/num_experts, so below roughly two
// routes per expert the extra empty blocks and the prefix scan cost more than
// the traffic they save -- measured at N=2 the grouped kernel is ~40% slower.
// `PIE_MXFP4_MOE_GROUPED` forces the choice for A/B comparison: the two
// kernels must agree token-for-token.
bool mxfp4_moe_grouped_choice(int routes, int num_experts) {
    static const int forced = [] {
        const char* v = std::getenv("PIE_MXFP4_MOE_GROUPED");
        if (v == nullptr || v[0] == '\0') return -1;
        return v[0] == '1' ? 1 : 0;
    }();
    if (forced >= 0) return forced == 1;
    return num_experts > 0 && routes >= 2 * num_experts;
}

// The expert-indexed Marlin MoE: one launch covers every expert, which is what
// vLLM runs for this model on A100. Microbenchmarked against the kernels it
// replaces at gpt-oss's shape (`driver/cuda/bench/moe_bench.cu`, gate_up ms):
//   N=8   per-route 0.183  grouped 0.255  marlin 0.081
//   N=32  per-route 0.697  grouped 0.592  marlin 0.113
//   N=128 per-route 2.745  grouped 1.735  marlin 0.146
// At N=32 that is 1274 GB/s of weight traffic, i.e. the kernel is finally
// bandwidth-bound instead of unpack-bound.
bool mxfp4_marlin_moe_enabled() {
    static const bool on = [] {
        const char* v = std::getenv("PIE_MXFP4_MARLIN_MOE");
        return v == nullptr || v[0] != '0';   // on unless disabled
    }();
    return on;
}

bool mixtral_profile_enabled() {
    static const bool on = [] {
        const char* v = std::getenv("PIE_MIXTRAL_PROFILE");
        return v != nullptr && v[0] == '1';
    }();
    return on;
}

template <class F>
void mx_stage(MixtralPhaseProfile& p, double* dst, cudaStream_t stream, F&& fn) {
    if (!p.enabled || dst == nullptr) { fn(); return; }
    p.open(dst, stream);
    fn();
    p.close(stream);
}

const DeviceTensor& must(const LoadedModel& e, const std::string& name) {
    if (!e.has(name)) {
        throw std::runtime_error("mixtral: missing weight '" + name + "'");
    }
    return e.get(name);
}

}  // namespace

MixtralWeights bind_mixtral(const LoadedModel& engine) {
    const auto& cfg = engine.hf_config();
    const int E = cfg.num_experts;
    if (E <= 0) {
        throw std::runtime_error(
            "mixtral: hf_config.num_experts must be > 0; check the loader");
    }

    MixtralWeights w;
    w.embed      = &must(engine, "model.embed_tokens.weight");
    w.final_norm = &must(engine, "model.norm.weight");
    if (engine.has("lm_head.weight")) {
        w.lm_head = &engine.get("lm_head.weight");
    } else if (cfg.tie_word_embeddings) {
        w.lm_head = w.embed;
    } else {
        throw std::runtime_error(
            "mixtral: lm_head missing and tie_word_embeddings=false");
    }

    w.layers.resize(static_cast<std::size_t>(cfg.num_hidden_layers));
    for (int i = 0; i < cfg.num_hidden_layers; ++i) {
        const std::string p = "model.layers." + std::to_string(i) + ".";
        auto& L = w.layers[i];
        L.attn_norm = &must(engine, p + "input_layernorm.weight");
        L.mlp_norm  = &must(engine, p + "post_attention_layernorm.weight");
        L.q_proj    = &must(engine, p + "self_attn.q_proj.weight");
        L.k_proj    = &must(engine, p + "self_attn.k_proj.weight");
        L.v_proj    = &must(engine, p + "self_attn.v_proj.weight");
        L.o_proj    = &must(engine, p + "self_attn.o_proj.weight");

        L.router    = &must(engine, p + "block_sparse_moe.gate.weight");
        L.experts.resize(static_cast<std::size_t>(E));
        for (int e = 0; e < E; ++e) {
            const std::string ep = p + "block_sparse_moe.experts." +
                                   std::to_string(e) + ".";
            // HF Mixtral expert weight layout: w1=gate, w2=down, w3=up.
            L.experts[e].w_gate = &must(engine, ep + "w1.weight");
            L.experts[e].w_down = &must(engine, ep + "w2.weight");
            L.experts[e].w_up   = &must(engine, ep + "w3.weight");
        }
    }
    return w;
}

namespace {

// Build per-expert (token, weight) lists from a [N, K] topk decision.
// Returns one vector<int32> of token indices and one matching vector<float>
// of routing weights per expert. CPU-side because routing is O(N·K) and
// the per-expert dense GEMMs dominate runtime.
struct ExpertRouting {
    std::vector<std::vector<std::int32_t>> token_idx;
    std::vector<std::vector<float>>        weights;
};

ExpertRouting build_routing(
    const std::vector<std::int32_t>& topk_idx_h,
    const std::vector<float>& topk_w_h,
    int N, int K, int E)
{
    ExpertRouting r;
    r.token_idx.assign(E, {});
    r.weights.assign(E, {});
    for (int n = 0; n < N; ++n) {
        for (int k = 0; k < K; ++k) {
            const int e = topk_idx_h[n * K + k];
            if (e < 0 || e >= E) continue;
            r.token_idx[e].push_back(n);
            r.weights[e].push_back(topk_w_h[n * K + k]);
        }
    }
    return r;
}

// The four device-side arrays the fused MXFP4 decode kernels index by expert.
//
// One struct rather than four loose vectors so that the two paths that fill it
// -- a slab that holds every expert and one that pages the routed set -- state
// the slot's tensor names in one place. An entry left null is an expert this
// step did not route to, which the kernels never read.
struct ExpertPtrTable {
    explicit ExpertPtrTable(int experts)
        : gate_up(static_cast<std::size_t>(experts), nullptr),
          gate_up_scale(static_cast<std::size_t>(experts), nullptr),
          down(static_cast<std::size_t>(experts), nullptr),
          down_scale(static_cast<std::size_t>(experts), nullptr) {}

    std::vector<const std::uint8_t*> gate_up;
    std::vector<const std::uint8_t*> gate_up_scale;
    std::vector<const std::uint8_t*> down;
    std::vector<const std::uint8_t*> down_scale;
};

}  // namespace

void mixtral_forward_paged(
    const MixtralWeights& w,
    const HfConfig& cfg,
    const LlamaLikeForwardCfg& fwd_cfg,
    int num_experts,
    int top_k,
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
    const std::uint8_t* row_valid_d)
{
    // TP-local dims. tp_size == 1 keeps the original single-GPU shapes.
    // For Mixtral we shard *within* each expert (per-expert TP), not
    // across experts: every rank still runs the full expert routing but
    // each expert's gate/up/down weights are split along axis 0 / axis 1.
    const int T  = (fwd_cfg.tp_size > 0) ? fwd_cfg.tp_size : 1;
    const int H = cfg.hidden_size;
    const int Hq = (cfg.num_attention_heads * cfg.head_dim) / T;
    const int Hk = (cfg.num_key_value_heads * cfg.head_dim) / T;
    const int I  = cfg.intermediate_size / T;
    const int Ip = (w.mxfp4_intermediate_padded > I)
        ? w.mxfp4_intermediate_padded
        : I;
    const int num_q_heads_local  = cfg.num_attention_heads / T;
    const int num_kv_heads_local = cfg.num_key_value_heads / T;
    const int V  = cfg.vocab_size;
    const int d  = cfg.head_dim;
    const float eps = cfg.rms_norm_eps;
    cudaStream_t stream = nullptr;
    NcclComm* tp = (T > 1) ? fwd_cfg.tp_comm : nullptr;
    const bool tp_is_leader = (T == 1) || (tp != nullptr && tp->rank() == 0);

    const bool use_decode_path = is_pure_decode && !fwd_cfg.force_prefill_path;
    MixtralPhaseProfile prof;
    prof.enabled = mixtral_profile_enabled();
    prof.last_N = N;
    if (prof.enabled) {
        CUDA_CHECK(cudaEventCreate(&prof.a));
        CUDA_CHECK(cudaEventCreate(&prof.b));
    }
    const bool any_sinks = [&]{
        for (const auto& L : w.layers) {
            if (L.attn_sinks != nullptr) return true;
        }
        return false;
    }();
    // [N, num_attention_heads] fp32 — written by flashinfer per layer when
    // sinks are active, then consumed by the rescale post-pass. Per-layer
    // overwrite is fine; we only need the layer's own lse during its own
    // rescale step. Allocate once per fire instead of once per layer.
    DeviceBuffer<float> d_lse;
    float* lse_ptr = nullptr;
    if (any_sinks) {
        d_lse = DeviceBuffer<float>::alloc(
            static_cast<std::size_t>(N) * num_q_heads_local);
        lse_ptr = d_lse.data();
    }

    kernels::launch_embed_bf16(
        token_ids, w.embed->data(), ws.y.data(), N, H, V, stream);

    ops::DecodePlanCachePtr decode_plan;
    if (use_decode_path) {
        decode_plan = ops::make_decode_plan();
        ops::plan_attention_flashinfer_decode(
            *decode_plan, kv_page_indptr_h, R,
            num_q_heads_local, num_kv_heads_local, d,
            cache.page_size(), attn_ws, stream,
            /*enable_cuda_graph=*/true,
            /*full_attention_variant=*/false,
            cache.hnd_layout());
    }

    // ── Sliding-window page trim ──────────────────────────────────────
    // gpt-oss alternates a 128-token window with full attention, but the
    // paged decode kernel reads every page it is given and only masks the
    // out-of-window ones, so the windowed layers were paying full-context
    // traffic for a window that never grows. Hand them a shorter page list
    // instead: the decode query sits at the END of its range, so dropping
    // whole pages off the FRONT leaves the last `window+1` tokens in place
    // and `window_left` still masks against the same positions.
    //
    // One plan serves both page lists: it is `page_count_independent`, so its
    // descriptor is a function of the request count and not of how many pages
    // each request holds, and a second planner run would carve from offset 0
    // of the same int workspace as the first.
    //
    // WHICH pages survive is decided on the DEVICE, and unconditionally. This
    // model does not override `graph_layout`, so one captured graph serves
    // every context length; a host-computed count is frozen into that capture,
    // and a fire that skips the trim emits a different kernel sequence
    // entirely. Replaying either against a shorter request walks off the front
    // of its page list. Keeping one page more than the window needs also makes
    // the count safe for any `last_page_len` without reading it back.
    const int page_size = static_cast<int>(cache.page_size());
    int trim_window = -1;
    if (use_decode_path && page_size > 0) {
        for (int w : fwd_cfg.per_layer_window_left) {
            if (w < 0) continue;
            if (trim_window < 0) { trim_window = w; }
            // A second distinct window would need a second page view and a
            // second plan; no current model has one, so decline instead of
            // trimming some windowed layers and not others.
            else if (trim_window != w) { trim_window = -1; break; }
        }
    }
    DeviceBuffer<std::uint32_t> win_indices;
    DeviceBuffer<std::uint32_t> win_indptr;
    if (trim_window >= 0 && R > 0) {
        const int keep_max = 1 + (trim_window + 1 + page_size - 1) / page_size;
        win_indices = DeviceBuffer<std::uint32_t>::alloc(
            static_cast<std::size_t>(R) * keep_max);
        win_indptr = DeviceBuffer<std::uint32_t>::alloc(
            static_cast<std::size_t>(R) + 1);
        kernels::launch_build_window_page_view(
            kv_page_indices, kv_page_indptr, keep_max,
            win_indptr.data(), win_indices.data(), R, stream);
    }

    // ── Full-attention KV split ───────────────────────────────────────
    // The other half of gpt-oss's layers see the whole context, and there the
    // trim above has nothing to drop. What is wrong with them is parallelism:
    // eight kv heads and one request is eight CTAs on 132 SMs, and the
    // microbench puts that ~46x off the kernel's own bandwidth roofline (2.25
    // MB of KV read in 37 us). Split the range into `kMixtralFullSplits`
    // one-token requests over consecutive slices, share the query with
    // `broadcast_q`, and fold the partials with `MergeStates`.
    //
    // The slice count is FIXED. It is the request count the plan's descriptor
    // and the launch geometry are derived from, so it must not track a context
    // length that grows under a captured graph -- which is also why the view
    // itself is built on the device and the launch is unconditional.
    // 16 slices x 8 kv heads is 128 CTAs, about one wave on an H100. Swept at
    // gpt-oss's decode shape (ctx ~1100): 8 -> 239.2, 16 -> 235.6, 24 -> 241.3,
    // 32 -> 238.3 tok/s, which is flat inside a run-to-run spread of about 5 --
    // once the machine is full, more slices only add partials to merge. The
    // knob stays for the next shape that needs sweeping. It must be FIXED
    // within a process: it is the request count the plan descriptor and the
    // launch geometry are derived from.
    static const int kMixtralFullSplits = [] {
        const char* v = std::getenv("PIE_MIXTRAL_FULL_SPLITS");
        const int n = (v != nullptr) ? std::atoi(v) : 16;
        return (n >= 1 && n <= 128) ? n : 16;
    }();
    ops::DecodePlanCachePtr split_plan;
    DeviceBuffer<std::uint32_t> split_indptr, split_indices, split_last;
    DeviceBuffer<std::uint16_t> split_partial;
    DeviceBuffer<float> split_lse, split_lse_merged;
    bool has_full_layer = false;
    for (int L = 0; L < cfg.num_hidden_layers; ++L) {
        const int w_l = (L < (int)fwd_cfg.per_layer_window_left.size())
                            ? fwd_cfg.per_layer_window_left[L]
                            : fwd_cfg.sliding_window;
        if (w_l < 0) { has_full_layer = true; break; }
    }
    if (use_decode_path && has_full_layer && R == 1 && page_size > 0 &&
        custom_mask_d == nullptr) {
        const int splits = kMixtralFullSplits;
        split_indptr = DeviceBuffer<std::uint32_t>::alloc(splits + 1);
        split_last = DeviceBuffer<std::uint32_t>::alloc(splits);
        // Sized from the model's context limit, not from this fire's page
        // count: the latter is a host value, and a capture made while it was
        // small would replay into an allocation too short for a longer
        // request. The pool size would also be safe but is far larger than any
        // one request can hold, and this buffer is cut per fire.
        const std::size_t max_req_pages =
            (static_cast<std::size_t>(cfg.max_position_embeddings) +
             page_size - 1) / page_size;
        split_indices = DeviceBuffer<std::uint32_t>::alloc(
            static_cast<std::size_t>(splits) +
            std::min<std::size_t>(max_req_pages,
                                  static_cast<std::size_t>(cache.num_pages())));
        kernels::launch_build_full_split_view(
            kv_page_indptr, kv_last_page_lens, splits, page_size,
            split_indptr.data(), split_indices.data(), split_last.data(),
            kv_page_indices, stream);
        split_plan = ops::make_decode_plan();
        // Past the primary plan's descriptor, which is sized for R.
        ops::set_decode_plan_int_base(*split_plan, 1u << 20);
        // The descriptor is page-count independent, so the counts handed to
        // the planner only have to be a well-formed indptr over `splits`
        // requests; the real ranges reach the LAUNCH, from the device.
        std::vector<std::uint32_t> plan_indptr_h(splits + 1);
        for (int i = 0; i <= splits; ++i) {
            plan_indptr_h[i] = static_cast<std::uint32_t>(i);
        }
        ops::plan_attention_flashinfer_decode(
            *split_plan, plan_indptr_h.data(), splits,
            num_q_heads_local, num_kv_heads_local, d, page_size,
            attn_ws, stream, /*enable_cuda_graph=*/true,
            /*full_attention_variant=*/true, cache.hnd_layout());
        const std::size_t rows =
            static_cast<std::size_t>(splits) * num_q_heads_local;
        split_partial = DeviceBuffer<std::uint16_t>::alloc(
            rows * static_cast<std::size_t>(d));
        split_lse = DeviceBuffer<float>::alloc(rows);
        split_lse_merged =
            DeviceBuffer<float>::alloc(static_cast<std::size_t>(num_q_heads_local));
    }

    // Per-fire scratch for MoE routing. Sized for the worst case (N
    // tokens × K experts each); reallocated per call which is fine
    // since N changes, but the host-side vectors avoid touching cuda
    // alloc. Device-side topk buffers go through DeviceBuffer for ABI
    // simplicity; if profiling shows alloc latency we can hoist these
    // into Workspace.
    auto d_topk_idx = DeviceBuffer<std::int32_t>::alloc(
        static_cast<std::size_t>(N) * top_k);
    auto d_topk_w   = DeviceBuffer<float>::alloc(
        static_cast<std::size_t>(N) * top_k);
    // Per-expert scratch for gathered inputs and projection outputs.
    // Worst case: a single expert receives all N*K routes. Pre-size to
    // that bound to avoid re-allocating inside the layer loop.
    const std::size_t max_routed = static_cast<std::size_t>(N) * top_k;
    auto d_expert_in    = DeviceBuffer<std::uint16_t>::alloc(max_routed * H);
    auto d_expert_gate  = DeviceBuffer<std::uint16_t>::alloc(max_routed * Ip);
    auto d_expert_up    = DeviceBuffer<std::uint16_t>::alloc(max_routed * Ip);
    auto d_expert_out   = DeviceBuffer<std::uint16_t>::alloc(max_routed * H);
    auto d_expert_idx   = DeviceBuffer<std::int32_t>::alloc(max_routed);
    auto d_expert_w     = DeviceBuffer<float>::alloc(max_routed);

    // Fused MXFP4 decode GEMV admission. The kernel reads one copy of an
    // expert's packed weights per *route*; the materializing path reads one
    // per *distinct expert*, but then writes and re-reads a 4x-larger bf16
    // expansion and pays a cuBLAS launch per expert, so it needs routes to
    // run many times ahead of num_experts before it catches up. Measured on
    // gpt-oss (H=I=2880, top_k=4, E=8), fused vs materializing:
    //
    //     routes    4     32    128    256    512
    //     speedup  4.8x  4.4x  1.9x   1.1x   0.7x
    //
    // so the crossover sits near 32x num_experts, and the threshold scales
    // with E because the materializing path's cost scales with the number
    // of *distinct* experts touched. It also keeps prefill out: a frame of
    // more than a few dozen tokens is far past the cap, and there the
    // tensor-core GEMM the materializing path feeds is the right kernel.
    // PIE_MXFP4_DECODE_ROUTES overrides it for tuning.
    const bool mxfp4_decode_gemv_available =
        !w.layers.empty() &&
        !w.layers[0].expert_gate_up_packed_ptrs.empty() &&
        H % 32 == 0 && I % 32 == 0;
    if (mxfp4_decode_gemv_available && w.mxfp4_decode_max_routes == 0) {
        w.mxfp4_decode_max_routes = 32 * num_experts;
    }
    const bool use_mxfp4_decode_gemv =
        mxfp4_decode_gemv_available &&
        N * top_k <= w.mxfp4_decode_max_routes;
    // Sized from the live frame, never from the cap: the cap is an
    // admission threshold that can legitimately exceed any frame's routes,
    // and sizing scratch by it would either waste an arena or -- if a later
    // frame grew -- read out of bounds.
    const std::size_t gemv_routes =
        use_mxfp4_decode_gemv ? max_routed : 0;
    auto d_mxfp4_act_fp16 = DeviceBuffer<std::uint16_t>::alloc(
        use_mxfp4_decode_gemv ? static_cast<std::size_t>(N) * H : 0);
    auto d_mxfp4_route_gate =
        DeviceBuffer<std::uint16_t>::alloc(gemv_routes * I);
    auto d_mxfp4_route_up =
        DeviceBuffer<std::uint16_t>::alloc(gemv_routes * I);
    auto d_mxfp4_route_act_fp16 =
        DeviceBuffer<std::uint16_t>::alloc(gemv_routes * I);
    // Route bucketing for the expert-grouped gate/up kernel: routes sorted by
    // expert plus the per-expert histogram. Sized from the live frame like the
    // rest of this scratch.
    auto d_moe_sorted_routes =
        DeviceBuffer<std::int32_t>::alloc(gemv_routes);
    auto d_moe_route_to_row =
        DeviceBuffer<std::int32_t>::alloc(gemv_routes);
    auto d_moe_expert_counts = DeviceBuffer<std::int32_t>::alloc(
        use_mxfp4_decode_gemv ? static_cast<std::size_t>(num_experts) : 0);
    auto d_mxfp4_route_out =
        DeviceBuffer<std::uint16_t>::alloc(gemv_routes * H);
    auto d_mxfp4_moe_out = DeviceBuffer<std::uint16_t>::alloc(
        use_mxfp4_decode_gemv ? static_cast<std::size_t>(N) * H : 0);
#ifdef PIE_CUDA_HAS_MARLIN_MOE
    // Marlin consumes the padded block-sorted form, and writes gate/up at the
    // PADDED intermediate width (the packed weights are aligned to 128), so
    // these cannot share the gemv scratch above.
    const int Ip_marlin = w.mxfp4_intermediate_padded > 0
                              ? w.mxfp4_intermediate_padded : I;
    const bool use_marlin_moe =
        use_mxfp4_decode_gemv && mxfp4_marlin_moe_enabled() &&
        w.mxfp4_intermediate_padded > 0;
    const int marlin_block = 16;
    const int marlin_max_blocks =
        use_marlin_moe
            ? (static_cast<int>(max_routed) + num_experts * (marlin_block - 1) +
               marlin_block - 1) / marlin_block
            : 0;
    auto d_marlin_sorted = DeviceBuffer<std::int32_t>::alloc(
        static_cast<std::size_t>(marlin_max_blocks) * marlin_block);
    auto d_marlin_expert_ids =
        DeviceBuffer<std::int32_t>::alloc(marlin_max_blocks);
    auto d_marlin_npast =
        DeviceBuffer<std::int32_t>::alloc(use_marlin_moe ? 1 : 0);
    auto d_marlin_gate = DeviceBuffer<std::uint16_t>::alloc(
        use_marlin_moe ? max_routed * Ip_marlin : 0);
    auto d_marlin_up = DeviceBuffer<std::uint16_t>::alloc(
        use_marlin_moe ? max_routed * Ip_marlin : 0);
    // fc2 reads K = padded intermediate, so the activation fc1 produced has
    // to be laid out at that stride with the tail left zero.
    auto d_marlin_act = DeviceBuffer<std::uint16_t>::alloc(
        use_marlin_moe ? max_routed * Ip_marlin : 0);
    auto d_marlin_ws = DeviceBuffer<std::uint8_t>::alloc(
        use_marlin_moe
            ? marlin_moe::marlin_moe_workspace_bytes(Ip_marlin, marlin_block)
            : 0);
#endif
    for (int L = 0; L < cfg.num_hidden_layers; ++L) {
        const auto& layer = w.layers[L];

        // Per-layer attention window: full causal (-1) for plain Mixtral;
        // GPT-OSS alternates sliding/full per `fwd_cfg.per_layer_window_left`.
        const int layer_window =
            (L < (int)fwd_cfg.per_layer_window_left.size())
                ? fwd_cfg.per_layer_window_left[L]
                : fwd_cfg.sliding_window;

        // ── Attention block (identical to llama_like pre-norm path) ──
        if (prof.enabled) prof.open(&prof.attn, stream);
        kernels::launch_rmsnorm_bf16(
            ws.y.data(), layer.attn_norm->data(), ws.norm_x.data(),
            N, H, eps, stream);
        // Bias folded into the projection: at decode these route to the
        // warp-per-row GEMV, whose epilogue absorbs it for free. gpt-oss
        // biases all three, so this is 3 launches per layer removed.
        ops::gemm_act_x_wt_bias_bf16(cublas.handle(),
            ws.norm_x.data(), layer.q_proj->data(),
            layer.q_bias ? layer.q_bias->data() : nullptr,
            ws.q.data(), N, Hq, H, stream);
        ops::gemm_act_x_wt_bias_bf16(cublas.handle(),
            ws.norm_x.data(), layer.k_proj->data(),
            layer.k_bias ? layer.k_bias->data() : nullptr,
            ws.k.data(), N, Hk, H, stream);
        ops::gemm_act_x_wt_bias_bf16(cublas.handle(),
            ws.norm_x.data(), layer.v_proj->data(),
            layer.v_bias ? layer.v_bias->data() : nullptr,
            ws.v.data(), N, Hk, H, stream);

        kernels::launch_rope_bf16(
            ws.q.data(), ws.k.data(), positions,
            N, num_q_heads_local, num_kv_heads_local, d,
            cfg.rope_theta, stream);
        // Fires POST-rope (and post q/k-norm): the query a PTIR program
        // observes here is the one that actually enters attention, so an
        // observer scoring it against the cached keys -- which are stored
        // post-rope -- compares in the same space. Placing it on the raw
        // projection instead would silently mis-rank pages for Quest.
        invoke_stage_hook(
            StageHookPoint::OnAttnProj, ws.q.data(),
            static_cast<std::uint32_t>(N),
            static_cast<std::uint32_t>(Hq),
            static_cast<std::uint32_t>(L), stream);

        auto kv_view = cache.layer_view(L);
        kernels::launch_write_kv_to_pages(
            kv_view, ws.k.data(), ws.v.data(),
            qo_indptr, kv_page_indices, kv_page_indptr, kv_last_page_lens,
            N, R, stream, row_valid_d);

        // Only ask flashinfer for lse on layers that actually use sinks.
        // Saves a per-layer kernel write on plain Mixtral, and on
        // gpt-oss layers that turn out to have nullptr sinks.
        float* layer_lse = (layer.attn_sinks != nullptr) ? lse_ptr : nullptr;

        const bool use_full_split =
            use_decode_path && !split_indices.empty() && layer_window < 0;
        if (use_full_split) {
            ops::dispatch_attention_flashinfer_decode_bf16(
                *split_plan, ws.q.data(),
                kv_view.k_pages, kv_view.v_pages,
                split_partial.data(), split_indices.data(),
                split_indptr.data(), split_last.data(),
                attn_ws, stream, /*window_left=*/-1,
                /*logits_soft_cap=*/0.f, /*sm_scale=*/-1.f,
                split_lse.data(), /*broadcast_q=*/true);
            ops::merge_attention_states_bf16(
                split_partial.data(), split_lse.data(),
                ws.attn_out.data(), split_lse_merged.data(),
                kMixtralFullSplits, 1, num_q_heads_local, d, stream);
        } else if (use_decode_path) {
            // One plan serves both page lists. With `enable_cuda_graph` the
            // scheduler declines to split KV, and what is left of the plan --
            // request/tile indices, o_indptr, padded batch -- is a function of
            // the REQUEST COUNT, not of how many pages each request holds. So
            // the trimmed list rides the plan already built, and no second
            // planner run competes for offset 0 of the shared int workspace.
            const bool trimmed =
                !win_indices.empty() && layer_window == trim_window;
            ops::dispatch_attention_flashinfer_decode(
                *decode_plan,
                ws.q.data(), kv_view, ws.attn_out.data(),
                trimmed ? win_indices.data() : kv_page_indices,
                trimmed ? win_indptr.data() : kv_page_indptr,
                kv_last_page_lens,
                attn_ws, stream,
                /*window_left=*/layer_window,
                /*logits_soft_cap=*/0.f,
                /*sm_scale=*/-1.f,
                layer_lse);
        } else if (custom_mask_d) {
            ops::launch_attention_flashinfer_prefill_custom(
                ws.q.data(), kv_view, ws.attn_out.data(),
                qo_indptr, kv_page_indices, kv_page_indptr, kv_last_page_lens,
                custom_mask_d, custom_mask_indptr_d,
                qo_indptr_h, kv_page_indptr_h,
                N, R, num_q_heads_local, attn_ws, stream,
                /*window_left=*/-1,
                /*logits_soft_cap=*/0.f, /*sm_scale=*/-1.f,
                layer_lse);
        } else {
            ops::launch_attention_flashinfer_prefill(
                ws.q.data(), kv_view, ws.attn_out.data(),
                qo_indptr, kv_page_indices, kv_page_indptr, kv_last_page_lens,
                qo_indptr_h, kv_page_indptr_h,
                N, R, num_q_heads_local, attn_ws, stream,
                /*window_left=*/layer_window,
                /*logits_soft_cap=*/0.f,
                /*sm_scale=*/-1.f,
                layer_lse);
        }
        // GPT-OSS: rescale o by `sigmoid(lse - sink_h)` to apply the
        // softmax-denominator extension that flashinfer's DefaultAttention
        // doesn't emit natively. Per-rank shard count under TP.
        if (layer.attn_sinks != nullptr) {
            // On a split layer each slice's lse is a partial; the total the
            // sink extension needs is the one MergeStates just folded.
            kernels::launch_attention_sink_rescale_bf16(
                ws.attn_out.data(),
                use_full_split ? split_lse_merged.data() : layer_lse,
                layer.attn_sinks->data(),
                N, num_q_heads_local, d, stream);
        }
        invoke_stage_hook(
            StageHookPoint::OnAttn, ws.q.data(),
            static_cast<std::uint32_t>(N),
            static_cast<std::uint32_t>(Hq),
            static_cast<std::uint32_t>(L), stream);

        if (prof.enabled) prof.close(stream);
        if (prof.enabled) prof.open(&prof.o_proj, stream);

        // o_proj is row-parallel under TP: write to scratch, all-reduce,
        // residual-add into y. o_bias (replicated; e.g. GPT-OSS) only goes
        // in once on the leader so the all-reduce sums it exactly once.
        if (T == 1) {
            ops::gemm_act_x_wt_bf16(cublas.handle(),
                ws.attn_out.data(), layer.o_proj->data(), ws.y.data(),
                N, H, Hq, /*beta=*/1.f);
            if (layer.o_bias) kernels::launch_add_bias_bf16(
                ws.y.data(), layer.o_bias->data(), N, H, stream);
        } else {
            ops::gemm_act_x_wt_bias_bf16(cublas.handle(),
                ws.attn_out.data(), layer.o_proj->data(),
                (layer.o_bias && tp_is_leader) ? layer.o_bias->data() : nullptr,
                ws.norm_x.data(), N, H, Hq, stream);
            tp->all_reduce_bf16(ws.norm_x.data(),
                static_cast<std::size_t>(N) * H, ncclSum, stream);
            kernels::launch_residual_add_bf16(
                ws.y.data(), ws.norm_x.data(), N * H, stream);
        }

        // ── Sparse-MoE block ──
        if (prof.enabled) prof.close(stream);   // o_proj
        if (prof.enabled) prof.open(&prof.router, stream);
        kernels::launch_rmsnorm_bf16(
            ws.y.data(), layer.mlp_norm->data(), ws.norm_y.data(),
            N, H, eps, stream);

        // 1. Router logits, top-K + softmax + renormalize. We piggy-back
        // on `ws.gate` as scratch for the [N, num_experts] router logits
        // — its allocation is `[max_tokens, intermediate]` which is
        // always ≥ [N, num_experts] for any production config (E ≤ 64,
        // I ≥ 4096).
        ops::gemm_act_x_wt_bias_bf16(cublas.handle(),
            ws.norm_y.data(), layer.router->data(),
            layer.router_bias ? layer.router_bias->data() : nullptr,
            ws.gate.data(), N, num_experts, H, stream);
        kernels::launch_topk_softmax_bf16(
            ws.gate.data(), d_topk_idx.data(), d_topk_w.data(),
            N, num_experts, top_k, stream);

        // 1b. Fused MXFP4 decode GEMV.
        //
        // The generic path below materializes every routed expert's bf16
        // weights before calling cuBLAS. At decode that dominates the step
        // (63% of GPU time on gpt-oss): each token rewrites hundreds of MiB
        // of bf16 that is read exactly once. The fused GEMV reads the packed
        // nibbles straight out of HBM instead, so the traffic is the 4-bit
        // weight and nothing else -- and because it takes `topk_idx` on the
        // device, the D2H sync in step 2 disappears with it.
        //
        // It is a *per-route* kernel, so its weight traffic grows with
        // routes while the materializing path's grows with distinct experts.
        // Below `mxfp4_decode_max_routes` the fused path reads strictly
        // less; above it the materializing path amortises better and wins.
        // Streaming does not have to give the fused path up.
        //
        // The kernel wants a device array of per-expert pointers, and a
        // streamed expert's pointer is wherever its slot is -- but a slot's
        // pointers are fixed at slot creation and a page-in changes only the
        // bytes behind them, so the array is a few hundred bytes to rewrite
        // per layer. Page in the routed set, point the four weight arrays at
        // their slots, and the same kernel runs. Measured, this is the single
        // largest number in the whole feature: giving it up cost 5.2x, far
        // more than any miss rate does.
        //
        // Two things it costs. The routed set has to be known on the host, so
        // the D2H the fused path was written to avoid comes back -- one per
        // layer, against a kernel worth five times the step. And every routed
        // expert is pinned at once, so a slab that cannot hold the layer's
        // routed set falls back to the per-expert loop rather than deadlock
        // against its own pins.
        if (prof.enabled) prof.close(stream);
        bool streamed_fused = false;
        // The routing table is read back at most once per layer. The fused
        // paged path below needs it to decide what to page in, and the generic
        // dispatch further down needs it to build its per-expert lists; when a
        // layer takes both -- which is exactly the case where the slab is too
        // small for the routed set -- reading it twice would drain the stream
        // twice in the configuration already under the most pressure.
        std::vector<std::int32_t> topk_idx_h;
        bool topk_idx_read = false;
        // Whether the fused decode kernels can run at all: they index the
        // experts through a device-side pointer array, so a layer that has no
        // such array, or is not on the decode path, has nothing to fuse.
        const bool fusable = layer.expert_cache != nullptr &&
            use_mxfp4_decode_gemv &&
            !layer.expert_gate_up_packed_ptrs.empty();
        if (fusable) {
            // The four arrays the fused kernels index by expert, and the one
            // place their runtime names are written down. Both the static and the
            // paged path fill the same table from the same slot layout; they
            // differ only in how they come by the slot.
            ExpertPtrTable ptrs(num_experts);
            const auto read_slot = [&](int e, const WeightStore& slot) {
                ptrs.gate_up[e] = static_cast<const std::uint8_t*>(
                    slot.get("gate_up_proj.weight").data());
                ptrs.gate_up_scale[e] = static_cast<const std::uint8_t*>(
                    slot.get("gate_up_proj.weight_scale").data());
                ptrs.down[e] = static_cast<const std::uint8_t*>(
                    slot.get("down_proj.weight").data());
                ptrs.down_scale[e] = static_cast<const std::uint8_t*>(
                    slot.get("down_proj.weight_scale").data());
            };
            const auto upload_ptrs = [&] {
                const std::size_t nb =
                    static_cast<std::size_t>(num_experts) * sizeof(const void*);
                const auto up = [&](const DeviceBuffer<const std::uint8_t*>& d,
                                    const std::vector<const std::uint8_t*>& h) {
                    CUDA_CHECK(cudaMemcpyAsync(
                        const_cast<void*>(static_cast<const void*>(d.data())),
                        h.data(), nb, cudaMemcpyHostToDevice, stream));
                };
                up(layer.expert_gate_up_packed_ptrs, ptrs.gate_up);
                up(layer.expert_gate_up_scale_ptrs, ptrs.gate_up_scale);
                up(layer.expert_down_packed_ptrs, ptrs.down);
                up(layer.expert_down_scale_ptrs, ptrs.down_scale);
            };

            // Once the slab holds the whole group, nothing about residency
            // can change again: every expert owns a slot for the rest of the
            // process. The routed set was only ever read back so the host
            // could decide what to page in and where to point the kernel, and
            // both answers are now fixed -- so write the pointer array once,
            // for every expert, and stop reading anything back.
            //
            // This is the difference between streaming costing 18% and costing
            // nothing. The D2H is a full stream drain, so it converts a
            // pipeline that ran the host a whole token ahead of the device
            // into one that serialises at every layer; deleting it puts the
            // pipeline back.
            if (!layer.expert_ptrs_static && layer.expert_cache->all_placed()) {
                bool complete = true;
                for (int e = 0; e < num_experts; ++e) {
                    const WeightStore* slot = layer.expert_cache->store_of(
                        layer.expert_group, static_cast<std::uint32_t>(e));
                    if (slot == nullptr) { complete = false; break; }
                    read_slot(e, *slot);
                }
                if (complete) {
                    upload_ptrs();
                    // The kernels below read the array on the same stream, so the
                    // upload is ordered ahead of them; but a later forward will
                    // not re-issue it, so make sure it has landed before the flag
                    // says it did.
                    CUDA_CHECK(cudaStreamSynchronize(stream));
                    layer.expert_ptrs_static = true;
                }
            }
            if (layer.expert_ptrs_static) {
                streamed_fused = true;
            } else {
                topk_idx_h.resize(static_cast<std::size_t>(N) * top_k);
                CUDA_CHECK(cudaMemcpyAsync(topk_idx_h.data(), d_topk_idx.data(),
                                           topk_idx_h.size() * sizeof(std::int32_t),
                                           cudaMemcpyDeviceToHost, stream));
                CUDA_CHECK(cudaStreamSynchronize(stream));
                topk_idx_read = true;
                std::vector<int> routed;
                std::vector<char> seen(static_cast<std::size_t>(num_experts), 0);
                for (const std::int32_t e : topk_idx_h) {
                    if (e >= 0 && e < num_experts && !seen[e]) {
                        seen[e] = 1;
                        routed.push_back(e);
                    }
                }
                if (routed.size() <= layer.expert_cache->num_slots()) {
                    for (const int e : routed) {
                        layer.expert_cache->prefetch(layer.expert_group,
                                                     static_cast<std::uint32_t>(e));
                    }
                    for (const int e : routed) {
                        read_slot(e, layer.expert_cache->ensure_resident(
                            layer.expert_group, static_cast<std::uint32_t>(e),
                            stream));
                    }
                    // Pageable source, so the driver stages it before
                    // returning and `ptrs` may die at the end of this block;
                    // the DMA that follows is ordered on `stream` ahead of the
                    // kernels.
                    upload_ptrs();
                    streamed_fused = true;
                }
            }
        }
#ifdef PIE_CUDA_HAS_MARLIN_MOE
        // Expert-indexed Marlin: three launches for the whole layer instead of
        // one pass per route. Needs the native (unfused gate/up) MXFP4 slabs,
        // which only the resident backend publishes.
        if (use_marlin_moe && layer.expert_cache == nullptr &&
            !layer.experts.empty() &&
            layer.experts[0].format == MixtralExpertWeightFormat::Mxfp4NativeGemm &&
            layer.experts[0].w_gate_mxfp4 != nullptr) {
            const int routes = N * top_k;
            const auto& e0 = layer.experts[0];
            if (!layer.marlin_scales_ready) {
                // One-time per layer: the checkpoint's `[E, n, k/32]` scales
                // are the transpose of what Marlin walks.
                const std::size_t gu_elems =
                    static_cast<std::size_t>(num_experts) * Ip_marlin * (H / 32);
                const std::size_t dn_elems =
                    static_cast<std::size_t>(num_experts) * H * (Ip_marlin / 32);
                layer.marlin_gate_scales =
                    DeviceBuffer<std::uint8_t>::alloc(gu_elems);
                layer.marlin_up_scales =
                    DeviceBuffer<std::uint8_t>::alloc(gu_elems);
                layer.marlin_down_scales =
                    DeviceBuffer<std::uint8_t>::alloc(dn_elems);
                kernels::launch_transpose_expert_scales_u8(
                    e0.w_gate_mxfp4_scale->data(),
                    layer.marlin_gate_scales.data(), num_experts, Ip_marlin,
                    H / 32, stream);
                kernels::launch_transpose_expert_scales_u8(
                    e0.w_up_mxfp4_scale->data(),
                    layer.marlin_up_scales.data(), num_experts, Ip_marlin,
                    H / 32, stream);
                kernels::launch_transpose_expert_scales_u8(
                    e0.w_down_mxfp4_scale->data(),
                    layer.marlin_down_scales.data(), num_experts, H,
                    Ip_marlin / 32, stream);
                CUDA_CHECK(cudaStreamSynchronize(stream));
                layer.marlin_scales_ready = true;
            }
            kernels::launch_moe_align_decode(
                d_topk_idx.data(), d_marlin_sorted.data(),
                d_marlin_expert_ids.data(), /*route_to_aligned_row=*/nullptr,
                routes, num_experts, marlin_block, marlin_max_blocks,
                d_marlin_npast.data(), stream);
            // `prob_m`/`top_k` describe how the kernel turns a sorted entry
            // into an A row: it reads A row `sorted[i] / top_k` and treats
            // entries at or past `prob_m * top_k` as padding. The two
            // projections therefore differ. fc1 consumes the per-TOKEN hidden
            // state, so it is (prob_m = tokens, top_k = K); fc2 consumes the
            // per-ROUTE activation fc1 produced, so it is (prob_m = routes,
            // top_k = 1) over the same sorted array.
            const auto moe_gemm = [&](const void* act, const void* w_packed,
                                      const void* w_scale, void* out,
                                      int prob_m_arg, int top_k_arg,
                                      int prob_n, int prob_k, bool mul_w) {
                marlin_moe::launch_mxfp4_moe_gemm_w4a16_bf16(
                    act, w_packed, w_scale, /*bias=*/nullptr, out,
                    /*reduce_scratch=*/nullptr, d_marlin_ws.data(),
                    d_marlin_sorted.data(), d_marlin_expert_ids.data(),
                    d_marlin_npast.data(),
                    mul_w ? static_cast<const float*>(d_topk_w.data()) : nullptr,
                    marlin_block, num_experts, top_k_arg, mul_w,
                    prob_m_arg, prob_n, prob_k, stream);
            };
            moe_gemm(ws.norm_y.data(), e0.w_gate_mxfp4->data(),
                     layer.marlin_gate_scales.data(), d_marlin_gate.data(),
                     N, top_k, Ip_marlin, H, false);
            moe_gemm(ws.norm_y.data(), e0.w_up_mxfp4->data(),
                     layer.marlin_up_scales.data(), d_marlin_up.data(),
                     N, top_k, Ip_marlin, H, false);
            // GPT-OSS publishes its expert biases at the UNPADDED width, which
            // is not the stride Marlin's own bias epilogue assumes.
            if (e0.b_gate != nullptr) {
                kernels::launch_add_moe_route_bias_bf16(
                    d_marlin_gate.data(), e0.b_gate->data(),
                    d_topk_idx.data(), routes, I, Ip_marlin, stream);
                kernels::launch_add_moe_route_bias_bf16(
                    d_marlin_up.data(), e0.b_up->data(),
                    d_topk_idx.data(), routes, I, Ip_marlin, stream);
            }
            if (cfg.swiglu_limit > 0.f) {
                kernels::launch_gpt_oss_glu_strided_bf16(
                    d_marlin_gate.data(), d_marlin_up.data(),
                    d_marlin_act.data(), routes, I, Ip_marlin, Ip_marlin,
                    stream, cfg.swiglu_limit);
            } else {
                kernels::launch_gpt_oss_glu_strided_bf16(
                    d_marlin_gate.data(), d_marlin_up.data(),
                    d_marlin_act.data(), routes, I, Ip_marlin, Ip_marlin,
                    stream, /*limit=*/0.f);
            }
            moe_gemm(d_marlin_act.data(), e0.w_down_mxfp4->data(),
                     layer.marlin_down_scales.data(), d_mxfp4_route_out.data(),
                     routes, /*top_k=*/1, H, Ip_marlin, false);
            if (e0.b_down != nullptr && tp_is_leader) {
                kernels::launch_add_moe_route_bias_bf16(
                    d_mxfp4_route_out.data(), e0.b_down->data(),
                    d_topk_idx.data(), routes, H, H, stream);
            }
            kernels::launch_token_batched_weighted_sum_bf16(
                d_mxfp4_moe_out.data(), d_mxfp4_route_out.data(),
                static_cast<const float*>(d_topk_w.data()), N, top_k, H,
                stream);
            if (T > 1) {
                tp->all_reduce_bf16(d_mxfp4_moe_out.data(),
                    static_cast<std::size_t>(N) * H, ncclSum, stream);
            }
            kernels::launch_residual_add_bf16(
                ws.y.data(), d_mxfp4_moe_out.data(), N * H, stream);
            continue;
        }
#endif
        if ((layer.expert_cache == nullptr || streamed_fused) &&
            use_mxfp4_decode_gemv &&
            !layer.expert_gate_up_packed_ptrs.empty()) {
            const int routes = N * top_k;
            kernels::launch_bf16_to_fp16(
                ws.norm_y.data(), d_mxfp4_act_fp16.data(),
                static_cast<std::size_t>(N) * H, stream);
            mx_stage(prof, &prof.moe_gate_up, stream, [&]{
            // Group the routes by expert first: without it every route
            // re-streams its expert's slab, so weight traffic scales with
            // tokens and the decode throughput is flat in batch size.
            if (mxfp4_moe_grouped_choice(routes, num_experts)) {
                kernels::launch_moe_bucket_exact(
                    d_topk_idx.data(), d_moe_sorted_routes.data(),
                    d_moe_route_to_row.data(), d_moe_expert_counts.data(),
                    routes, num_experts, stream);
                kernels::launch_mxfp4_moe_gate_up_decode_grouped_bf16(
                    d_mxfp4_act_fp16.data(),
                    d_moe_sorted_routes.data(), d_moe_expert_counts.data(),
                    layer.expert_gate_up_packed_ptrs.data(),
                    layer.expert_gate_up_scale_ptrs.data(),
                    layer.expert_gate_bias_ptrs.data(),
                    layer.expert_up_bias_ptrs.data(),
                    d_mxfp4_route_gate.data(), d_mxfp4_route_up.data(),
                    num_experts, top_k, H, I, stream);
            } else {
                kernels::launch_mxfp4_moe_gate_up_decode_bf16(
                    d_mxfp4_act_fp16.data(), d_topk_idx.data(),
                    layer.expert_gate_up_packed_ptrs.data(),
                    layer.expert_gate_up_scale_ptrs.data(),
                    layer.expert_gate_bias_ptrs.data(),
                    layer.expert_up_bias_ptrs.data(),
                    d_mxfp4_route_gate.data(), d_mxfp4_route_up.data(),
                    N, top_k, H, I, stream);
            }
            });
            if (cfg.swiglu_limit > 0.f) {
                kernels::launch_gpt_oss_glu_bf16(
                    d_mxfp4_route_gate.data(), d_mxfp4_route_up.data(),
                    d_mxfp4_route_gate.data(),
                    static_cast<int>(static_cast<std::size_t>(routes) * I),
                    stream, /*limit=*/cfg.swiglu_limit);
            } else {
                kernels::launch_swiglu_bf16(
                    d_mxfp4_route_gate.data(), d_mxfp4_route_up.data(),
                    d_mxfp4_route_gate.data(),
                    static_cast<std::size_t>(routes) * I, stream);
            }
            kernels::launch_bf16_to_fp16(
                d_mxfp4_route_gate.data(), d_mxfp4_route_act_fp16.data(),
                static_cast<std::size_t>(routes) * I, stream);
            // b_down is replicated across ranks, so only the leader adds it;
            // the all-reduce below would otherwise sum it T times.
            mx_stage(prof, &prof.moe_down, stream, [&]{
            kernels::launch_mxfp4_moe_down_decode_bf16(
                d_mxfp4_route_act_fp16.data(), d_topk_idx.data(),
                layer.expert_down_packed_ptrs.data(),
                layer.expert_down_scale_ptrs.data(),
                tp_is_leader ? layer.expert_down_bias_ptrs.data() : nullptr,
                d_mxfp4_route_out.data(), N, top_k, H, I, stream);
            });
            kernels::launch_token_batched_weighted_sum_bf16(
                d_mxfp4_moe_out.data(), d_mxfp4_route_out.data(),
                static_cast<const float*>(d_topk_w.data()),
                N, top_k, H, stream);
            if (T > 1) {
                tp->all_reduce_bf16(d_mxfp4_moe_out.data(),
                    static_cast<std::size_t>(N) * H, ncclSum, stream);
            }
            kernels::launch_residual_add_bf16(
                ws.y.data(), d_mxfp4_moe_out.data(), N * H, stream);
            if (streamed_fused) {
                // Safe with the kernels above still queued: a page-in that
                // wants one of these slots synchronizes `stream` before it
                // overwrites anything.
                layer.expert_cache->end_batch();
            }
            continue;
        }

        // 2. D2H copy of routing decisions; build per-expert lists.
        std::vector<float>        topk_w_h  (static_cast<std::size_t>(N) * top_k);
        if (!topk_idx_read) {
            topk_idx_h.resize(static_cast<std::size_t>(N) * top_k);
            CUDA_CHECK(cudaMemcpyAsync(topk_idx_h.data(), d_topk_idx.data(),
                                       topk_idx_h.size() * sizeof(std::int32_t),
                                       cudaMemcpyDeviceToHost, stream));
        }
        CUDA_CHECK(cudaMemcpyAsync(topk_w_h.data(), d_topk_w.data(),
                                   topk_w_h.size() * sizeof(float),
                                   cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        const auto routing = build_routing(topk_idx_h, topk_w_h,
                                           N, top_k, num_experts);

        // The router has spoken, and the dispatch below is a serial walk over
        // what it chose. Tell the page cache the whole set now, so the reads
        // for expert 7 are in flight while expert 0's GEMMs run.
        if (layer.expert_cache != nullptr) {
            for (int e = 0; e < num_experts; ++e) {
                if (!routing.token_idx[e].empty()) {
                    layer.expert_cache->prefetch(layer.expert_group,
                                                 static_cast<std::uint32_t>(e));
                }
            }
        }

        // Under TP every rank computes 1/T of every expert's down_proj.
        // Each scatter_add accumulates a *partial* contribution; we
        // collect those in ws.norm_x (zero-initialised), all-reduce after
        // all experts, then residual-add the full MoE delta into ws.y.
        // tp_size == 1 keeps the original "scatter directly into ws.y"
        // path so single-GPU performance is unchanged.
        void* moe_target = ws.y.data();
        if (T > 1) {
            CUDA_CHECK(cudaMemsetAsync(
                ws.norm_x.data(), 0,
                static_cast<std::size_t>(N) * H * sizeof(std::uint16_t),
                stream));
            moe_target = ws.norm_x.data();
        }

        // 3. Per-expert dispatch.
        for (int e = 0; e < num_experts; ++e) {
            const auto& tok_idx = routing.token_idx[e];
            const auto& weights = routing.weights[e];
            const int Ne = static_cast<int>(tok_idx.size());
            if (Ne == 0) continue;

            CUDA_CHECK(cudaMemcpyAsync(
                d_expert_idx.data(), tok_idx.data(),
                Ne * sizeof(std::int32_t), cudaMemcpyHostToDevice, stream));
            CUDA_CHECK(cudaMemcpyAsync(
                d_expert_w.data(), weights.data(),
                Ne * sizeof(float), cudaMemcpyHostToDevice, stream));

            // Gather norm_y rows routed to this expert.
            kernels::launch_gather_bf16_rows(
                static_cast<const std::uint16_t*>(ws.norm_y.data()),
                d_expert_idx.data(),
                d_expert_in.data(),
                Ne, H, stream);

            // SwiGLU MLP.
            //
            // When the layer's experts are streamed, the resident entry holds
            // only the biases; page this one in and read its weights and
            // factors out of the slot. Everything else below is unchanged,
            // because the slot holds exactly what one stride of the bank held
            // -- the group's plan is the bank's expression with the expert
            // axis fixed at one.
            MixtralExpertWeights paged_in;
            if (layer.expert_cache != nullptr) {
                const WeightStore& slot = layer.expert_cache->ensure_resident(
                    layer.expert_group, static_cast<std::uint32_t>(e), stream);
                paged_in = layer.experts[e];
                paged_in.w_gate_up = &slot.get("gate_up_proj.weight");
                paged_in.w_gate_up_scale = &slot.get("gate_up_proj.weight_scale");
                paged_in.w_down_packed = &slot.get("down_proj.weight");
                paged_in.w_down_scale = &slot.get("down_proj.weight_scale");
            }
            const auto& expert = layer.expert_cache != nullptr
                                     ? paged_in
                                     : layer.experts[e];
            // Unpin where the loop leaves, so the pin covers exactly the
            // launches that read the slot and no more. A later page-in that
            // wants this slot syncs `stream` before overwriting it, so nothing
            // races with the kernels queued above.
            struct Unpin {
                GroupStreamCache* cache;
                ~Unpin() { if (cache != nullptr) cache->end_batch(); }
            } unpin{layer.expert_cache};
            const void* gate_w = nullptr;
            const void* up_w = nullptr;
            const void* down_w = nullptr;
            if (expert.format == MixtralExpertWeightFormat::Mxfp4NativeGemm) {
                if (!expert.w_gate_mxfp4 || !expert.w_gate_mxfp4_scale ||
                    !expert.w_up_mxfp4 || !expert.w_up_mxfp4_scale ||
                    !expert.w_down_mxfp4 || !expert.w_down_mxfp4_scale) {
                    throw std::runtime_error(
                        "mixtral/gpt_oss: incomplete native MXFP4 expert backend");
                }
                ops::gemm_act_x_w(cublas.handle(),
                    d_expert_in.data(),
                    ops::WeightView::mxfp4_marlin(
                        *expert.w_gate_mxfp4, *expert.w_gate_mxfp4_scale),
                    d_expert_gate.data(), Ne, Ip, H);
                ops::gemm_act_x_w(cublas.handle(),
                    d_expert_in.data(),
                    ops::WeightView::mxfp4_marlin(
                        *expert.w_up_mxfp4, *expert.w_up_mxfp4_scale),
                    d_expert_up.data(), Ne, Ip, H);
                if (expert.b_gate) kernels::launch_add_bias_bf16_strided(
                    d_expert_gate.data(), expert.b_gate->data(), Ne, I, Ip,
                    stream);
                if (expert.b_up) kernels::launch_add_bias_bf16_strided(
                    d_expert_up.data(), expert.b_up->data(), Ne, I, Ip,
                    stream);
                if (cfg.swiglu_limit > 0.f) {
                    kernels::launch_gpt_oss_glu_bf16(
                        d_expert_gate.data(), d_expert_up.data(),
                        d_expert_gate.data(),
                        static_cast<int>(static_cast<std::size_t>(Ne) * Ip), stream,
                        /*limit=*/cfg.swiglu_limit);
                } else {
                    kernels::launch_swiglu_bf16(
                        d_expert_gate.data(), d_expert_up.data(),
                        d_expert_gate.data(),
                        static_cast<std::size_t>(Ne) * Ip, stream);
                }
                ops::gemm_act_x_w(cublas.handle(),
                    d_expert_gate.data(),
                    ops::WeightView::mxfp4_marlin(
                        *expert.w_down_mxfp4, *expert.w_down_mxfp4_scale),
                    d_expert_out.data(), Ne, H, Ip);
                if (expert.b_down && tp_is_leader) kernels::launch_add_bias_bf16(
                    d_expert_out.data(), expert.b_down->data(), Ne, H, stream);
                kernels::launch_scatter_add_weighted_bf16(
                    moe_target, d_expert_out.data(),
                    d_expert_idx.data(), d_expert_w.data(),
                    Ne, H, stream);
                continue;
            }
            if (expert.format == MixtralExpertWeightFormat::Mxfp4RoutedDequant) {
                if (!expert.w_gate_up || !expert.w_gate_up_scale ||
                    !expert.w_down_packed || !expert.w_down_scale ||
                    w.mxfp4_gate_up_bf16_scratch.empty() ||
                    w.mxfp4_gate_bf16_scratch.empty() ||
                    w.mxfp4_up_bf16_scratch.empty() ||
                    w.mxfp4_down_bf16_scratch.empty()) {
                    throw std::runtime_error(
                        "mixtral/gpt_oss: incomplete MXFP4 expert backend");
                }
                kernels::launch_dequant_mxfp4_to_bf16(
                    static_cast<const std::uint8_t*>(expert.w_gate_up->data()),
                    static_cast<const std::uint8_t*>(
                        expert.w_gate_up_scale->data()),
                    w.mxfp4_gate_up_bf16_scratch.data(),
                    2 * I, H, stream);
                kernels::launch_dequant_mxfp4_to_bf16(
                    static_cast<const std::uint8_t*>(
                        expert.w_down_packed->data()),
                    static_cast<const std::uint8_t*>(
                        expert.w_down_scale->data()),
                    w.mxfp4_down_bf16_scratch.data(),
                    H, I, stream);
                kernels::launch_deinterleave_rows_bf16(
                    w.mxfp4_gate_up_bf16_scratch.data(),
                    w.mxfp4_gate_bf16_scratch.data(),
                    w.mxfp4_up_bf16_scratch.data(),
                    I, H, stream);
                gate_w = w.mxfp4_gate_bf16_scratch.data();
                up_w = w.mxfp4_up_bf16_scratch.data();
                down_w = w.mxfp4_down_bf16_scratch.data();
            } else {
                gate_w = expert.w_gate->data();
                up_w = expert.w_up->data();
                down_w = expert.w_down->data();
            }
            ops::gemm_act_x_wt_bf16(cublas.handle(),
                d_expert_in.data(), gate_w,
                d_expert_gate.data(), Ne, I, H);
            ops::gemm_act_x_wt_bf16(cublas.handle(),
                d_expert_in.data(), up_w,
                d_expert_up.data(), Ne, I, H);
            if (expert.b_gate) kernels::launch_add_bias_bf16(
                d_expert_gate.data(), expert.b_gate->data(), Ne, I, stream);
            if (expert.b_up) kernels::launch_add_bias_bf16(
                d_expert_up.data(), expert.b_up->data(), Ne, I, stream);
            if (cfg.swiglu_limit > 0.f) {
                kernels::launch_gpt_oss_glu_bf16(
                    d_expert_gate.data(), d_expert_up.data(),
                    d_expert_gate.data(),
                    static_cast<int>(static_cast<std::size_t>(Ne) * I), stream,
                    /*limit=*/cfg.swiglu_limit);
            } else {
                kernels::launch_swiglu_bf16(
                    d_expert_gate.data(), d_expert_up.data(),
                    d_expert_gate.data(),
                    static_cast<std::size_t>(Ne) * I, stream);
            }
            ops::gemm_act_x_wt_bf16(cublas.handle(),
                d_expert_gate.data(), down_w,
                d_expert_out.data(), Ne, H, I);
            // b_down is replicated across ranks; only the leader applies
            // it so the all-reduce sums it once. Plain Mixtral has no
            // b_down so this branch is dead until GPT-OSS.
            if (expert.b_down && tp_is_leader) kernels::launch_add_bias_bf16(
                d_expert_out.data(), expert.b_down->data(), Ne, H, stream);

            // Scatter into ws.y (TP=1) or moe_target scratch (TP>1) with
            // routing weight, residual-add style.
            kernels::launch_scatter_add_weighted_bf16(
                moe_target, d_expert_out.data(),
                d_expert_idx.data(), d_expert_w.data(),
                Ne, H, stream);
        }

        if (T > 1) {
            tp->all_reduce_bf16(ws.norm_x.data(),
                static_cast<std::size_t>(N) * H, ncclSum, stream);
            kernels::launch_residual_add_bf16(
                ws.y.data(), ws.norm_x.data(), N * H, stream);
        }
    }

    if (!fwd_cfg.emit_logits) {
        return;
    }
    if (prof.enabled) prof.open(&prof.epilogue, stream);
    // Compact logits: gather only the rows that will be sampled before the
    // lm_head, instead of materializing [N, vocab]. Every other family already
    // declares this; without it the batch engine hands the device-side sampler
    // an empty row list and its descriptor channel is never produced.
    const bool compact_logits =
        logit_row_indices_d != nullptr && num_logit_rows > 0 &&
        num_logit_rows < N;
    const int lm_head_rows = compact_logits ? num_logit_rows : N;
    if (compact_logits) {
        kernels::launch_gather_bf16_rows(
            static_cast<const std::uint16_t*>(ws.y.data()),
            logit_row_indices_d,
            static_cast<std::uint16_t*>(ws.norm_x.data()),
            num_logit_rows, H, stream);
        kernels::launch_rmsnorm_bf16(
            ws.norm_x.data(), w.final_norm->data(), ws.norm_y.data(),
            num_logit_rows, H, eps, stream);
        ops::gemm_act_x_wt_bf16(cublas.handle(),
            ws.norm_y.data(), w.lm_head->data(), ws.logits.data(),
            lm_head_rows, V, H);
        if (prof.enabled) prof.close(stream);
        return;
    }
    kernels::launch_rmsnorm_bf16(
        ws.y.data(), w.final_norm->data(), ws.norm_x.data(),
        N, H, eps, stream);
    ops::gemm_act_x_wt_bf16(cublas.handle(),
        ws.norm_x.data(), w.lm_head->data(), ws.logits.data(),
        N, V, H);
    if (prof.enabled) prof.close(stream);
}

}  // namespace pie_cuda_driver::model
