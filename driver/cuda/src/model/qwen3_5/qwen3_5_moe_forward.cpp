#include "model/qwen3_5/qwen3_5_moe_forward.hpp"
#include "model/stage_hooks.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cctype>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <sstream>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "cuda_check.hpp"
#include "kernels/causal_conv1d.hpp"
#include "kernels/deinterleave.hpp"
#include "kernels/embed.hpp"
#include "kernels/gated_delta_net.hpp"
#include "kernels/gather_rows.hpp"
#include "kernels/kv_paged.hpp"
#include "kernels/moe_dispatch.hpp"
#include "kernels/moe_grouped_gemm.hpp"
#include "ops/flashinfer_moe.hpp"
#include "model/qwen3_5/qwen3_5_moe.hpp"
#include "kernels/residual_add.hpp"
#include <mutex>
#include <set>
#include <tuple>
#include <utility>

#include "kernels/rmsnorm.hpp"
#include "kernels/rope.hpp"
#include "kernels/slot_ops.hpp"
#include "kernels/swiglu.hpp"
#include "kernels/topk_softmax.hpp"
#include "ops/attention_flashinfer.hpp"
#include "ops/attention_naive.hpp"
#include "ops/attention_naive_paged.hpp"
#include "ops/gemm.hpp"

namespace pie_cuda_driver::model {

namespace {

// RMSNorm dispatch: Qwen3.5 / 3.6-MoE store gamma centered at zero and
// apply `(1 + w) * x_hat` (Gemma-style); Qwen3-MoE (Qwen3-30B-A3B) uses
// the standard `w * x_hat`. The bind layer wires the same struct for
// both so the forward picks the right kernel based on `cfg.model_type`.
inline bool uses_gemma_rmsnorm(const HfConfig& cfg) {
    return cfg.model_type != "qwen3_moe";
}

bool qwen35_moe_profile_enabled() {
    static const bool enabled = [] {
        const char* v = std::getenv("PIE_QWEN35_MOE_PROFILE");
        return v != nullptr && v[0] != '\0' && v[0] != '0';
    }();
    return enabled;
}

std::uint64_t qwen35_moe_profile_print_limit() {
    static const std::uint64_t limit = [] {
        const char* v = std::getenv("PIE_QWEN35_MOE_PROFILE_LIMIT");
        if (v == nullptr || v[0] == '\0') return std::uint64_t{8};
        const long parsed = std::strtol(v, nullptr, 10);
        return parsed > 0 ? static_cast<std::uint64_t>(parsed) : std::uint64_t{0};
    }();
    return limit;
}

bool qwen35_moe_profile_all_ranks() {
    static const bool enabled = [] {
        const char* v = std::getenv("PIE_QWEN35_MOE_PROFILE_ALL_RANKS");
        return v != nullptr && v[0] != '\0' && v[0] != '0';
    }();
    return enabled;
}

bool mtp_profile_enabled() {
    static const bool enabled = [] {
        const char* v = std::getenv("PIE_MTP_PROFILE");
        return v != nullptr && v[0] != '\0' && v[0] != '0';
    }();
    return enabled;
}

std::uint64_t mtp_profile_print_limit() {
    static const std::uint64_t limit = [] {
        const char* v = std::getenv("PIE_MTP_PROFILE_LIMIT");
        if (v == nullptr || v[0] == '\0') return std::uint64_t{8};
        const long parsed = std::strtol(v, nullptr, 10);
        return parsed > 0 ? static_cast<std::uint64_t>(parsed) : std::uint64_t{0};
    }();
    return limit;
}

int qwen35_gdn_cached_prefill_max_tokens() {
    static const int max_tokens = [] {
        const char* v = std::getenv("PIE_QWEN35_GDN_CACHED_PREFILL_MAX_TOKENS");
        if (v == nullptr || v[0] == '\0') return 0;
        return std::max(0, std::atoi(v));
    }();
    return max_tokens;
}

int qwen35_gdn_warp_tiled_max_tokens() {
    static const int max_tokens = [] {
        const char* v = std::getenv("PIE_QWEN35_GDN_WARP_TILED_MAX_TOKENS");
        if (v == nullptr || v[0] == '\0') return 64;
        return std::max(0, std::atoi(v));
    }();
    return max_tokens;
}

bool moe_path_log_enabled() {
    static const bool enabled = [] {
        const char* v = std::getenv("PIE_MOE_PATH_LOG");
        return v != nullptr && v[0] != '\0' && v[0] != '0';
    }();
    return enabled;
}

// Every block is one entry of a batched GEMM and reads a whole expert
// weight, and a block belongs to exactly one expert — so the row count
// pads up to roughly `active_experts * block`. With 256 experts holding
// only a few routes each, 16 padded 512 real routes to 4352 rows for the
// same number of weight reads; 8 halves the padding. Measured on
// Qwen3.6-35B-A3B, 128 requests x 256 tokens: 1197 tok/s at 8 against
// 1107 at 16, and it falls off either side (921 at 32, 586 at 64).
//
// The optimum tracks the per-expert weight size, so a single constant is
// wrong; see `qwen35_moe_aligned_decode_block_size` below for the rule and
// the measurements behind it.
// The shared expert's two projections are dense GEMMs over N rows with
// N_gemm = 2*Is and H. cuBLAS covers those with only a handful of thread
// blocks (4 at Is=256), so they run at a few percent of the machine. When
// Is == Im their weights have exactly the routed expert shapes, so they can
// instead ride along as one more expert in the routed batched GEMM.
//
// MEASURED, AND IT DOES NOT PAY. Qwen3.6-35B-A3B tp2: the profiled decode
// step at N=128 does improve (moe_shared 2.26 -> 0.85 ms, total 40.02 ->
// 39.34), but end to end it is noise at 128 requests (3131 vs 3121 tok/s)
// and a REGRESSION at 256 (4102 vs 4183, -1.9%).
//
// The reason is the useful part: the routed batched GEMM costs roughly in
// proportion to its BLOCK COUNT, not to the distinct weight bytes it reads.
// The folded shared blocks all point at one weight and hit L2, yet still
// cost ~their share of the GEMM (+0.56 ms for 16 of 352 blocks at N=128).
// The fold adds ceil(N/block) blocks, so its cost grows with N while the
// standalone shared GEMM's is flat in N (weight-bound) -- hence the
// crossover between 128 and 256 rows.
//
// Kept off by default, and kept at all because the same measurement says
// the ~100 padded blocks of every routed GEMM (352 blocks for ~252 active
// experts) are NOT free either. A MoE kernel that skips fully-padded blocks
// the way vLLM's `fused_moe_kernel` does would recover far more than this,
// and would want the shared expert folded in as well.
// Skips the padding blocks that the static batch-count bound forces on the
// cuBLAS path. On by default; set to 0 to fall back to cuBLAS.
bool moe_grouped_gemm_enabled() {
    static const bool enabled = [] {
        const char* v = std::getenv("PIE_QWEN35_MOE_GROUPED_GEMM");
        if (v == nullptr || v[0] == '\0') return true;
        return v[0] != '0';
    }();
    return enabled;
}

// flashinfer's CUTLASS grouped MoE. It takes unpermuted rows plus the topk
// indices/scales and does permute -> GEMM1 -> swiglu -> GEMM2 -> scaled
// unpermute internally, so it needs neither the aligned-block padding nor
// the static worst-case batch count that the cuBLAS path is forced into
// under graph capture, and its grouped GEMM tiles M properly. Those are
// exactly the two costs that keep PIE's MoE at ~700 GB/s against vLLM's
// 1220 on the same bytes.
// Decode batches are bounded by the scheduler's concurrency, so the fused
// MoE workspace is sized for this rather than for a prefill's token count.
constexpr int kFusedMoeMaxRows = 512;

bool moe_flashinfer_enabled() {
    static const bool enabled = [] {
        const char* v = std::getenv("PIE_QWEN35_MOE_FLASHINFER");
        if (v == nullptr || v[0] == '\0') return true;
        return v[0] != '0';
    }();
    return enabled;
}

// True when the routed gate/up weights were stored in flashinfer's
// [linear|gate] order at bind time. Both MoE paths must agree.
bool moe_gate_up_swapped() {
    return model::qwen35_moe_gate_up_swapped();
}

bool shared_fold_enabled() {
    static const bool enabled = [] {
        const char* v = std::getenv("PIE_QWEN35_MOE_FOLD_SHARED");
        if (v == nullptr || v[0] == '\0') return false;
        return v[0] != '0';
    }();
    return enabled;
}

// Two forces pull opposite ways. A bigger block means fewer batch entries,
// and the routed GEMM's cost scales with entry count (established by the
// shared-expert fold measurement above: 16 extra entries out of 352 cost
// +0.56 ms even though they all read one L2-resident weight). But a bigger
// block also means more padded rows to gather and multiply. Fewer entries
// wins while each expert's weight is small enough that the per-entry cost
// is not amortised; once the weight is large, the padded rows dominate.
//
// Measured on Qwen3.6-35B-A3B (128 requests x 256 tokens), whose two
// topologies straddle the crossover:
//   tp=1, 4.2 MB per expert : 8 -> 1499 tok/s, 16 -> 1432
//   tp=2, 2.1 MB per expert : 16 -> 3226 tok/s, 8 -> 3184
//     and on the profiled step, 8 -> 40.60 ms, 12 -> 40.03, 16 -> 40.02,
//     24 -> 41.21, the gain concentrated in the GEMMs (gate_up
//     11.15 -> 10.64 ms, down 8.70 -> 7.88).
// 32 and above is worse at both.
int qwen35_moe_aligned_decode_block_size(int inter_local, int hidden) {
    const char* v = std::getenv("PIE_QWEN35_MOE_ALIGNED_DECODE_BLOCK");
    if (v != nullptr && v[0] != '\0') {
        char* end = nullptr;
        long parsed_long = std::strtol(v, &end, 10);
        if (end != v) {
            int parsed = static_cast<int>(parsed_long);
            if (parsed <= 1) return 0;
            if (parsed > 64) parsed = 64;
            return parsed;
        }
    }
    if (inter_local <= 0 || hidden <= 0) return 8;
    const std::size_t gate_up_bytes = static_cast<std::size_t>(2) *
        inter_local * hidden * sizeof(std::uint16_t);
    return gate_up_bytes <= (std::size_t{3} << 20) ? 16 : 8;
}

int qwen35_moe_aligned_decode_min_routes() {
    static const int min_routes = [] {
        const char* v = std::getenv("PIE_QWEN35_MOE_ALIGNED_DECODE_MIN_ROUTES");
        if (v == nullptr || v[0] == '\0') return 64;
        return std::clamp(std::atoi(v), 0, 4096);
    }();
    return min_routes;
}

// Row count above which a NON-pure-decode step leaves the device-side
// MoE dispatch for the host-driven one. The host path resolves routing on
// the CPU, which costs a device sync per layer and then walks all
// `num_experts` slots issuing per-expert copies and GEMMs — 256 of them
// on Qwen3.6, 40 times per step. It exists for prefill, where the per
// expert row counts are large enough to amortise that; on a continuous
// batching mixed step it is simply the wrong path, and the steps that
// take it are the ones with the most rows to lose.
//
// No clamp: the cap is a tuning knob, not a safety bound, and clamping it
// to 128 silently pinned mixed steps of 144-252 rows to the host path.
//
// The default follows rows per expert rather than rows: an N-token step
// gives each expert about N*K/E rows, and the host path only earns its
// per-layer sync and its `num_experts` dispatches once that is a
// GEMM-sized number. At Qwen3.6's E=256, K=8, 1024 tokens is ~32 rows
// per expert. A real bulk prefill (8192 tokens) still takes the host
// path. Measured, 128 requests x 256 tokens: 64 gave 918-986 tok/s,
// everything from 128 up gave 994-1116 (within run-to-run spread).
int qwen35_moe_decode_fast_max_tokens() {
    static const int max_tokens = [] {
        const char* v = std::getenv("PIE_QWEN35_MOE_DECODE_FAST_N");
        if (v == nullptr || v[0] == '\0') return 1024;
        return std::max(0, std::atoi(v));
    }();
    return max_tokens;
}

// The routed decode GEMMs are M=1 streaming reads. A dedicated GEMV beats
// `cublasGemmBatchedEx` on them; see `moe_decode_gemv_bf16_kernel`.
bool qwen35_moe_gemv_decode_enabled() {
    static const bool enabled = [] {
        const char* v = std::getenv("PIE_QWEN35_MOE_GEMV_DECODE");
        if (v == nullptr || v[0] == '\0') return true;
        return v[0] != '0';
    }();
    return enabled;
}

bool qwen35_moe_wmma_decode_enabled() {
    static const bool enabled = [] {
        const char* v = std::getenv("PIE_QWEN35_MOE_WMMA_DECODE");
        return v != nullptr && v[0] != '\0' && v[0] != '0';
    }();
    return enabled;
}

enum class MtpMoeMode {
    Full,
    SharedOnly,
    Skip,
};

MtpMoeMode mtp_moe_mode() {
    static const MtpMoeMode mode = [] {
        const char* v = std::getenv("PIE_MTP_MOE_MODE");
        if (v == nullptr || v[0] == '\0') return MtpMoeMode::Full;
        std::string s(v);
        std::transform(s.begin(), s.end(), s.begin(),
            [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        if (s == "shared" || s == "shared_only" || s == "shared-only") {
            return MtpMoeMode::SharedOnly;
        }
        if (s == "skip" || s == "none" || s == "off") {
            return MtpMoeMode::Skip;
        }
        return MtpMoeMode::Full;
    }();
    return mode;
}

// Timing means synchronizing on an event, which CUDA forbids while the stream
// is capturing a graph, and which under TP stalls one rank inside a collective
// the other rank has already left. Either way the engine dies mid-run instead
// of reporting numbers, so a capturing stream runs the work untimed.
inline bool profile_stream_is_capturing(cudaStream_t stream) {
    cudaStreamCaptureStatus status = cudaStreamCaptureStatusNone;
    if (cudaStreamIsCapturing(stream, &status) != cudaSuccess) return true;
    return status != cudaStreamCaptureStatusNone;
}

struct Qwen35MoeForwardProfile {
    bool enabled = false;
    bool timing_suspended = false;
    int tp_rank = 0;
    int N = 0;
    int R = 0;
    bool pure_decode = false;
    int linear_layers = 0;
    int full_layers = 0;
    int moe_layers = 0;

    double embed_ms = 0.0;
    double norm_ms = 0.0;
    double linear_attn_ms = 0.0;
    double linear_proj_ms = 0.0;
    double linear_conv_ms = 0.0;
    double linear_prep_ms = 0.0;
    double linear_recur_ms = 0.0;
    double linear_post_ms = 0.0;
    double full_attn_ms = 0.0;
    double moe_router_ms = 0.0;
    double moe_routed_ms = 0.0;
    double moe_route_setup_ms = 0.0;
    double moe_align_ms = 0.0;
    double moe_gather_ms = 0.0;
    double moe_ptrs_ms = 0.0;
    double moe_gate_up_ms = 0.0;
    double moe_act_ms = 0.0;
    double moe_down_ms = 0.0;
    double moe_reduce_ms = 0.0;
    double moe_shared_ms = 0.0;
    double moe_shared_gate_up_ms = 0.0;
    double moe_shared_down_ms = 0.0;
    double moe_shared_gate_ms = 0.0;
    double moe_allreduce_ms = 0.0;

    double residual_ms = 0.0;
    double lm_head_ms = 0.0;
    double forward_ms = 0.0;

    cudaEvent_t forward_start = nullptr;
    cudaEvent_t forward_stop = nullptr;

    // Each timed stage takes its own event pair out of `pool` and defers the
    // readback to `end()`. Synchronizing per stage instead -- which this
    // profiler used to do -- drains the pipeline several hundred times per
    // step, so every stage paid a launch latency it does not pay in
    // production and small stages read 5-10x their true cost.
    std::vector<cudaEvent_t> pool;
    std::size_t pool_used = 0;
    std::vector<std::pair<double*, std::size_t>> pending;

    ~Qwen35MoeForwardProfile() {
        if (forward_start != nullptr) cudaEventDestroy(forward_start);
        if (forward_stop != nullptr) cudaEventDestroy(forward_stop);
        for (cudaEvent_t e : pool) cudaEventDestroy(e);
    }

    void ensure_events() {
        if (forward_start != nullptr) return;
        CUDA_CHECK(cudaEventCreate(&forward_start));
        CUDA_CHECK(cudaEventCreate(&forward_stop));
    }

    // Index of a fresh start event; its stop event is the next slot.
    std::size_t acquire_pair() {
        const std::size_t idx = pool_used;
        while (pool.size() < idx + 2) {
            cudaEvent_t e = nullptr;
            CUDA_CHECK(cudaEventCreate(&e));
            pool.push_back(e);
        }
        pool_used = idx + 2;
        return idx;
    }

    void begin(int n, int r, bool decode, int rank, cudaStream_t stream) {
        enabled = qwen35_moe_profile_enabled();
        if (!enabled) return;
        ensure_events();
        tp_rank = rank;
        N = n;
        R = r;
        pure_decode = decode;
        linear_layers = 0;
        full_layers = 0;
        moe_layers = 0;
        embed_ms = norm_ms = linear_attn_ms = full_attn_ms = 0.0;
        linear_proj_ms = linear_conv_ms = linear_prep_ms = 0.0;
        linear_recur_ms = linear_post_ms = 0.0;
        moe_router_ms = moe_routed_ms = moe_shared_ms = moe_allreduce_ms = 0.0;
        moe_route_setup_ms = moe_gate_up_ms = moe_act_ms = moe_down_ms = 0.0;
        moe_align_ms = moe_gather_ms = moe_ptrs_ms = 0.0;
        moe_reduce_ms = moe_shared_gate_up_ms = moe_shared_down_ms = 0.0;
        moe_shared_gate_ms = 0.0;
        residual_ms = lm_head_ms = forward_ms = 0.0;
        pool_used = 0;
        pending.clear();
        timing_suspended = profile_stream_is_capturing(stream);
        if (timing_suspended) return;
        CUDA_CHECK(cudaEventRecord(forward_start, stream));
    }

    void end(cudaStream_t stream) {
        if (!enabled || timing_suspended) return;
        CUDA_CHECK(cudaEventRecord(forward_stop, stream));
        CUDA_CHECK(cudaEventSynchronize(forward_stop));
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, forward_start, forward_stop));
        forward_ms = ms;
        // Every stage event is already complete now that `forward_stop` has,
        // so the whole step's timings read back without another sync.
        for (const auto& entry : pending) {
            float stage_ms = 0.0f;
            CUDA_CHECK(cudaEventElapsedTime(
                &stage_ms, pool[entry.second], pool[entry.second + 1]));
            add(*entry.first, stage_ms);
        }
        pending.clear();
        pool_used = 0;
    }

    void add(double& dst, float ms) {
        dst += static_cast<double>(ms);
    }
};

// Stages may nest: each one takes its own event pair, and nothing is read
// back until the step ends.
template <class F>
void profile_cuda_stage(
    Qwen35MoeForwardProfile* profile,
    double* dst,
    cudaStream_t stream,
    F&& fn)
{
    if (profile == nullptr || !profile->enabled || dst == nullptr ||
        profile->timing_suspended || profile_stream_is_capturing(stream)) {
        fn();
        return;
    }
    const std::size_t idx = profile->acquire_pair();
    CUDA_CHECK(cudaEventRecord(profile->pool[idx], stream));
    fn();
    CUDA_CHECK(cudaEventRecord(profile->pool[idx + 1], stream));
    profile->pending.emplace_back(dst, idx);
}

template <class F>
void profile_cuda_detail_stage(
    Qwen35MoeForwardProfile* profile,
    double* dst,
    cudaStream_t stream,
    F&& fn)
{
    profile_cuda_stage(profile, dst, stream, std::forward<F>(fn));
}

void maybe_print_profile(const Qwen35MoeForwardProfile& p) {
    if (!p.enabled) return;
    if (p.tp_rank != 0 && !qwen35_moe_profile_all_ranks()) return;
    static std::uint64_t seq = 0;
    ++seq;
    const std::uint64_t limit = qwen35_moe_profile_print_limit();
    if (limit == 0 || seq > limit) return;

    const double named =
        p.embed_ms + p.norm_ms + p.linear_attn_ms + p.full_attn_ms +
        p.moe_router_ms + p.moe_routed_ms + p.moe_shared_ms +
        p.moe_allreduce_ms + p.residual_ms + p.lm_head_ms;
    const double other = p.forward_ms > named ? p.forward_ms - named : 0.0;
    // One buffered line, one write. TP ranks profile concurrently, and a
    // chain of `std::cerr <<` interleaves their fields mid-number, which
    // silently corrupts anything parsing the output.
    std::ostringstream os;
    os
        << "[pie-qwen35-moe-profile] seq=" << seq
        << " rank=" << p.tp_rank
        << " N=" << p.N
        << " R=" << p.R
        << " decode=" << (p.pure_decode ? 1 : 0)
        << " layers_linear=" << p.linear_layers
        << " layers_full=" << p.full_layers
        << " layers_moe=" << p.moe_layers
        << " total_ms=" << p.forward_ms
        << " embed_ms=" << p.embed_ms
        << " norm_ms=" << p.norm_ms
        << " linear_attn_ms=" << p.linear_attn_ms
        << " linear_proj_ms=" << p.linear_proj_ms
        << " linear_conv_ms=" << p.linear_conv_ms
        << " linear_prep_ms=" << p.linear_prep_ms
        << " linear_recur_ms=" << p.linear_recur_ms
        << " linear_post_ms=" << p.linear_post_ms
        << " full_attn_ms=" << p.full_attn_ms
        << " moe_router_ms=" << p.moe_router_ms
        << " moe_routed_ms=" << p.moe_routed_ms
        << " moe_route_setup_ms=" << p.moe_route_setup_ms
        << " moe_align_ms=" << p.moe_align_ms
        << " moe_gather_ms=" << p.moe_gather_ms
        << " moe_ptrs_ms=" << p.moe_ptrs_ms
        << " moe_gate_up_ms=" << p.moe_gate_up_ms
        << " moe_act_ms=" << p.moe_act_ms
        << " moe_down_ms=" << p.moe_down_ms
        << " moe_reduce_ms=" << p.moe_reduce_ms
        << " moe_shared_ms=" << p.moe_shared_ms
        << " moe_shared_gate_up_ms=" << p.moe_shared_gate_up_ms
        << " moe_shared_down_ms=" << p.moe_shared_down_ms
        << " moe_shared_gate_ms=" << p.moe_shared_gate_ms
        << " moe_allreduce_ms=" << p.moe_allreduce_ms
        << " residual_ms=" << p.residual_ms
        << " lm_head_ms=" << p.lm_head_ms
        << " other_ms=" << other
        << "\n";
    std::cerr << os.str() << std::flush;
}

struct MtpProfile {
    bool enabled = false;
    int N = 0;
    double input_fc_ms = 0.0;
    double attn_ms = 0.0;
    double moe_ms = 0.0;
    double lm_head_ms = 0.0;
    double total_ms = 0.0;
    cudaEvent_t total_start = nullptr;
    cudaEvent_t total_stop = nullptr;
    cudaEvent_t stage_start = nullptr;
    cudaEvent_t stage_stop = nullptr;

    ~MtpProfile() {
        if (total_start != nullptr) cudaEventDestroy(total_start);
        if (total_stop != nullptr) cudaEventDestroy(total_stop);
        if (stage_start != nullptr) cudaEventDestroy(stage_start);
        if (stage_stop != nullptr) cudaEventDestroy(stage_stop);
    }

    void ensure_events() {
        if (total_start != nullptr) return;
        CUDA_CHECK(cudaEventCreate(&total_start));
        CUDA_CHECK(cudaEventCreate(&total_stop));
        CUDA_CHECK(cudaEventCreate(&stage_start));
        CUDA_CHECK(cudaEventCreate(&stage_stop));
    }

    void begin(int n, cudaStream_t stream) {
        enabled = mtp_profile_enabled();
        if (!enabled) return;
        ensure_events();
        N = n;
        input_fc_ms = attn_ms = moe_ms = lm_head_ms = total_ms = 0.0;
        CUDA_CHECK(cudaEventRecord(total_start, stream));
    }

    void end(cudaStream_t stream) {
        if (!enabled) return;
        CUDA_CHECK(cudaEventRecord(total_stop, stream));
        CUDA_CHECK(cudaEventSynchronize(total_stop));
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, total_start, total_stop));
        total_ms = static_cast<double>(ms);
    }
};

template <class F>
void profile_mtp_stage(
    MtpProfile& profile,
    double& dst,
    cudaStream_t stream,
    F&& fn)
{
    if (!profile.enabled) {
        fn();
        return;
    }
    CUDA_CHECK(cudaEventRecord(profile.stage_start, stream));
    fn();
    CUDA_CHECK(cudaEventRecord(profile.stage_stop, stream));
    CUDA_CHECK(cudaEventSynchronize(profile.stage_stop));
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, profile.stage_start, profile.stage_stop));
    dst += static_cast<double>(ms);
}

void maybe_print_mtp_profile(const MtpProfile& p) {
    if (!p.enabled) return;
    static std::uint64_t seq = 0;
    ++seq;
    const std::uint64_t limit = mtp_profile_print_limit();
    if (limit == 0 || seq > limit) return;
    const double named = p.input_fc_ms + p.attn_ms + p.moe_ms + p.lm_head_ms;
    const double other = p.total_ms > named ? p.total_ms - named : 0.0;
    std::cerr
        << "[pie-mtp-profile] seq=" << seq
        << " N=" << p.N
        << " total_ms=" << p.total_ms
        << " input_fc_ms=" << p.input_fc_ms
        << " attn_ms=" << p.attn_ms
        << " moe_ms=" << p.moe_ms
        << " lm_head_ms=" << p.lm_head_ms
        << " other_ms=" << other
        << "\n";
}

inline void rmsnorm_bf16_dispatch(
    const HfConfig& cfg,
    const void* x, const void* weight, void* y,
    int num_rows, int hidden, float eps, cudaStream_t stream)
{
    if (uses_gemma_rmsnorm(cfg)) {
        kernels::launch_rmsnorm_gemma_bf16(x, weight, y,
            num_rows, hidden, eps, stream);
    } else {
        kernels::launch_rmsnorm_bf16(x, weight, y,
            num_rows, hidden, eps, stream);
    }
}

}  // namespace

// Force the general (per-expert) dispatch even for shapes the decode fast path
// would take. Streaming already lands here because it has no fused slab to
// stride, so this is what makes "same weights, same path" a runnable
// comparison: without it a resident/streamed diff confounds the residency
// change with a change of kernel.
bool qwen35_moe_force_general_path() {
    static const bool on = [] {
        const char* v = std::getenv("PIE_QWEN35_MOE_FORCE_GENERAL");
        return v != nullptr && v[0] == '1';
    }();
    return on;
}

Qwen3_5MoeMlpWorkspace Qwen3_5MoeMlpWorkspace::allocate(
    int max_tokens, int hidden, int num_experts, int top_k,
    int moe_intermediate, int shared_intermediate)
{
    Qwen3_5MoeMlpWorkspace ws;
    const std::size_t N    = static_cast<std::size_t>(max_tokens);
    const std::size_t maxR = N * top_k;            // worst-case routes
    const std::size_t H    = static_cast<std::size_t>(hidden);
    const std::size_t I    = static_cast<std::size_t>(moe_intermediate);
    const std::size_t Ish  = static_cast<std::size_t>(shared_intermediate);

    ws.router_logits = DeviceBuffer<std::uint16_t>::alloc(N * num_experts);
    ws.topk_idx      = DeviceBuffer<std::int32_t>::alloc(N * top_k);
    ws.topk_weights  = DeviceBuffer<float>::alloc(N * top_k);

    ws.expert_in      = DeviceBuffer<std::uint16_t>::alloc(maxR * H);
    ws.expert_gate_up = DeviceBuffer<std::uint16_t>::alloc(maxR * 2 * I);
    ws.expert_act     = DeviceBuffer<std::uint16_t>::alloc(maxR * I);
    ws.expert_out     = DeviceBuffer<std::uint16_t>::alloc(maxR * H);
    ws.expert_idx     = DeviceBuffer<std::int32_t>::alloc(maxR);
    ws.expert_w       = DeviceBuffer<float>::alloc(maxR);

    ws.shared_gate       = DeviceBuffer<std::uint16_t>::alloc(N * Ish);
    ws.shared_up         = DeviceBuffer<std::uint16_t>::alloc(N * Ish);
    ws.shared_gate_up    = DeviceBuffer<std::uint16_t>::alloc(N * (2 * Ish + 1));
    ws.shared_act        = DeviceBuffer<std::uint16_t>::alloc(N * Ish);
    ws.shared_out        = DeviceBuffer<std::uint16_t>::alloc(N * H);
    ws.shared_gate_logit = DeviceBuffer<std::uint16_t>::alloc(N * 1);

    ws.moe_out = DeviceBuffer<std::uint16_t>::alloc(N * H);
    ws.a_gu_ptrs     = DeviceBuffer<const std::uint16_t*>::alloc(3 * maxR + 8);
    // Sized for the worst-case aligned block count, which exceeds maxR only
    // if block_size is 1; `+ 2 * maxR` covers the routed padding plus the
    // folded shared-expert blocks for every block size.
    const std::size_t ptr_slots = 3 * maxR + 8;
    ws.b_gu_ptrs     = DeviceBuffer<const std::uint16_t*>::alloc(ptr_slots);
    ws.c_gu_ptrs     = DeviceBuffer<std::uint16_t*>::alloc(ptr_slots);
    ws.a_dn_ptrs     = DeviceBuffer<const std::uint16_t*>::alloc(ptr_slots);
    ws.b_dn_ptrs     = DeviceBuffer<const std::uint16_t*>::alloc(ptr_slots);
    ws.c_dn_ptrs     = DeviceBuffer<std::uint16_t*>::alloc(ptr_slots);
    ws.batch_weights = DeviceBuffer<float>::alloc(maxR);

    ws.aligned_block_size =
        qwen35_moe_aligned_decode_block_size(moe_intermediate, hidden);
    if (ws.aligned_block_size > 1 && maxR > 0 && num_experts > 0) {
        const std::size_t active_expert_cap =
            std::min<std::size_t>(static_cast<std::size_t>(num_experts), maxR);
        const std::size_t block =
            static_cast<std::size_t>(ws.aligned_block_size);
        const std::size_t routed_blocks =
            (maxR + active_expert_cap * (block - 1) + block - 1) / block;
        // The shared expert is folded in as one more expert, occupying
        // ceil(N / block) blocks at the end. `maxR = maxN * top_k`, and top_k
        // is at least 1, so maxR bounds maxN.
        const std::size_t shared_blocks = (maxR + block - 1) / block;
        const std::size_t max_blocks = routed_blocks + shared_blocks;
        ws.aligned_rows_capacity = max_blocks * block;
        ws.aligned_route_ids =
            DeviceBuffer<std::int32_t>::alloc(ws.aligned_rows_capacity);
        ws.aligned_expert_ids =
            DeviceBuffer<std::int32_t>::alloc(max_blocks);
        ws.aligned_expert_in =
            DeviceBuffer<std::uint16_t>::alloc(ws.aligned_rows_capacity * H);
        ws.aligned_gate_up =
            DeviceBuffer<std::uint16_t>::alloc(ws.aligned_rows_capacity * 2 * I);
        ws.aligned_act =
            DeviceBuffer<std::uint16_t>::alloc(ws.aligned_rows_capacity * I);
        ws.aligned_out =
            DeviceBuffer<std::uint16_t>::alloc(ws.aligned_rows_capacity * H);
    }
    if (moe_flashinfer_enabled() && ops::flashinfer_cutlass_moe_enabled()) {
        // Sized for decode, not for the prefill high-water mark: the fused
        // path only runs on the decode fast path, and its workspace scales
        // with rows * top_k. At tp=1 the whole 68 GB model sits on one
        // device and a prefill-sized workspace does not fit the budget.
        ws.cutlass_max_rows = std::min(max_tokens, kFusedMoeMaxRows);
        const std::size_t bytes = ops::flashinfer_cutlass_moe_workspace_bytes(
            ops::MoeActivation::Swiglu, ws.cutlass_max_rows, hidden,
            moe_intermediate, num_experts, top_k,
            /*tp_size=*/1, /*tp_rank=*/0);
        if (bytes > 0) {
            ws.cutlass_ws = DeviceBuffer<std::uint8_t>::alloc(bytes);
            ws.cutlass_row_map = DeviceBuffer<std::int32_t>::alloc(
                static_cast<std::size_t>(ws.cutlass_max_rows) * top_k);
        }
    }
    return ws;
}

namespace {

// `linear_attn_body` and `full_attn_body` below are near-clones of the
// helpers in `qwen3_5_forward.cpp`. The only difference is the
// per-layer-weights type they consume (`Qwen3_5MoeLayerWeights` vs
// `Qwen3_5LayerWeights`). De-duplicating via a template would require
// hoisting the helpers out of the anonymous namespace and parameter-
// izing on the layer struct; that's a defensible refactor for later
// but we keep the small amount of copied code local to each arch
// while the schemas may still drift.

// Build per-expert routing lists from device-side topk decisions.
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

// Linear-attn body (replica of qwen3_5_forward.cpp's logic, against
// MoeLayerWeights). Reads `ws.norm_x`, writes contribution into
// `ws.norm_y`. Multi-request semantics match qwen3_5_forward's
// linear_attn_layer_body — see the comment block there.
void linear_attn_body(
    const Qwen3_5MoeLayerWeights& Lw,
    const HfConfig& cfg,
    const Qwen3_5ForwardCfg& fwd_cfg,
    Workspace& ws,
    Qwen3_5LinearAttnWorkspace& la,
    RecurrentStateCache& state_cache,
    int layer_idx, int linear_idx, int N_new, int R, bool is_pure_decode,
    const std::int32_t*  slot_ids_h,
    const std::int32_t*  slot_ids_d,
    const std::uint32_t* qo_new_h,
    const std::uint32_t* qo_new_d,
    ops::CublasHandle& cublas, cudaStream_t stream,
    Qwen35MoeForwardProfile* profile,
    const std::int32_t* commit_len = nullptr,
    const std::uint32_t* rs_buffer_slot_ids_h = nullptr,
    const std::uint32_t* rs_buffer_slot_indptr_h = nullptr,
    bool rs_buffer_write = false,
    // The buffered write folds the WHOLE extended row into the state.
    bool rs_fold_after_write = false,
    bool rs_buffer_fold = false,
    // Buffer READ path -- see qwen3_5_forward.cpp's linear_attn_layer_body for
    // the full rationale; this body mirrors it against MoE weights.
    const std::uint32_t* rs_buffer_read_slot_ids_h = nullptr,
    const std::uint32_t* rs_buffer_read_indptr_h = nullptr,
    const std::uint32_t* rs_buffer_read_lens_h = nullptr,
    const std::uint32_t* rs_buffer_heads_h = nullptr,
    const std::uint32_t* qo_ext_h = nullptr,
    const std::uint32_t* qo_ext_d = nullptr,
    int N_ext = 0,
    // MIXED pass only: one byte per request, non-zero where this row's
    // recurrent/conv state must persist. Null means every row agrees and
    // `write_state` alone decides.
    const std::uint8_t* rs_write_state_mask = nullptr)
{
    // The linear layers run over the EXTENDED token layout `[B_r | T_r]` per
    // request: a recurrence always resumes from the folded boundary, so any
    // token buffered past it must be replayed first. `N` / `qo_indptr_*` are
    // rebound to that space; `N_new` / `qo_new_h` stay in the fire's own.
    const bool has_buffer_read =
        rs_buffer_read_lens_h != nullptr && qo_ext_h != nullptr &&
        qo_ext_d != nullptr && N_ext > 0;
    const int N = has_buffer_read ? N_ext : N_new;
    const std::uint32_t* qo_indptr_h = has_buffer_read ? qo_ext_h : qo_new_h;
    const std::uint32_t* qo_indptr_d = has_buffer_read ? qo_ext_d : qo_new_d;
    auto read_len_for = [&](int r) -> int {
        return has_buffer_read
            ? static_cast<int>(rs_buffer_read_lens_h[r])
            : 0;
    };
    // Logical buffer token L of request r lives at physical `head + L`. A fold
    // absorbs tokens off the FRONT of the buffer but can only release WHOLE
    // covered pages, so a fold that lands mid-page leaves the survivors where
    // they were. Reading or writing at the logical index would then re-scan
    // tokens that are already inside the folded state, or overwrite live ones.
    auto head_for = [&](int r) -> int {
        return rs_buffer_heads_h != nullptr
            ? static_cast<int>(rs_buffer_heads_h[r])
            : 0;
    };
    const int T        = std::max(1, fwd_cfg.tp_size);
    const int H        = cfg.hidden_size;
    const int K_h      = cfg.linear_num_key_heads / T;
    const int V_h      = cfg.linear_num_value_heads / T;
    const int K_d      = cfg.linear_key_head_dim;
    const int V_d      = cfg.linear_value_head_dim;
    const int K_dim    = K_h * K_d;
    const int V_dim    = V_h * V_d;
    const int conv_dim = 2 * K_dim + V_dim;
    const int conv_K   = cfg.linear_conv_kernel_dim;
    // An extended layout has N != R, so the decode kernels cannot describe it.
    const bool linear_decode =
        is_pure_decode && !rs_buffer_write && !has_buffer_read;
    NcclComm* tp = (T > 1) ? fwd_cfg.tp_comm : nullptr;
    auto slot_for = [&](int r) -> int {
        return slot_ids_h ? slot_ids_h[r] : 0;
    };
    // Frozen verify: produce outputs but persist no recurrent/conv state (the
    // repair forward advances it through [input|accepted]). See qwen3_5_forward.
    // A buffered write leaves the folded state alone -- UNLESS it also folds.
    // The extended `[buffered | new]` layout IS the row's buffer token space,
    // so when the fold takes the WHOLE of it the boundary is the last extended
    // token and the ordinary end-of-sequence writeback already lands there.
    // (A boundary strictly inside is a different problem: `commit_len`
    // truncates the sequence rather than just moving the snapshot, so the
    // tokens past it would get no outputs. The planner refuses that.)
    const bool write_state =
        !state_cache.verify_frozen() && (!rs_buffer_write || rs_fold_after_write);

    const void* z_data = la.z.data();
    const void* a_data = la.a.data();
    const void* b_data = la.b.data();
    void* qkv_in_data = la.mixed_qkv.data();
    profile_cuda_detail_stage(
        profile, profile ? &profile->linear_proj_ms : nullptr,
        stream, [&] {
            if (rs_buffer_fold) {
                const int page = state_cache.rs_buffer_page_tokens();
                const std::size_t slab_a =
                    static_cast<std::size_t>(page) * conv_dim;
                const std::size_t slab_b =
                    slab_a + static_cast<std::size_t>(page) * V_h;
                for (int r = 0; r < R; ++r) {
                    const int qo0 = static_cast<int>(qo_indptr_h[r]);
                    const int nr =
                        static_cast<int>(qo_indptr_h[r + 1]) - qo0;
                    // The fold replays LOGICAL tokens [0, nr), which live at
                    // physical [head, head+nr).
                    const int head = head_for(r);
                    const int page_first = (head / page) * page;
                    const std::uint32_t s0 =
                        rs_buffer_slot_indptr_h[r];
                    const std::uint32_t s1 =
                        rs_buffer_slot_indptr_h[r + 1];
                    for (std::uint32_t j = 0; s0 + j < s1; ++j) {
                        const int page_tok0 =
                            page_first + static_cast<int>(j) * page;
                        const int phys0 = std::max(page_tok0, head);
                        const int count =
                            std::min(page_tok0 + page, head + nr) - phys0;
                        if (count <= 0) break;
                        const std::size_t in_page =
                            static_cast<std::size_t>(phys0 - page_tok0);
                        const int tok0 = phys0 - head;
                        auto* slab = static_cast<std::uint16_t*>(
                            state_cache.rs_buffer_slab(
                                linear_idx,
                                static_cast<int>(
                                    rs_buffer_slot_ids_h[s0 + j])));
                        CUDA_CHECK(cudaMemcpyAsync(
                            la.mixed_qkv.data() +
                                static_cast<std::size_t>(qo0 + tok0) * conv_dim,
                            slab + in_page * conv_dim,
                            static_cast<std::size_t>(count) * conv_dim *
                                sizeof(std::uint16_t),
                            cudaMemcpyDeviceToDevice, stream));
                        CUDA_CHECK(cudaMemcpyAsync(
                            la.a.data() +
                                static_cast<std::size_t>(qo0 + tok0) * V_h,
                            slab + slab_a + in_page * V_h,
                            static_cast<std::size_t>(count) * V_h *
                                sizeof(std::uint16_t),
                            cudaMemcpyDeviceToDevice, stream));
                        CUDA_CHECK(cudaMemcpyAsync(
                            la.b.data() +
                                static_cast<std::size_t>(qo0 + tok0) * V_h,
                            slab + slab_b + in_page * V_h,
                            static_cast<std::size_t>(count) * V_h *
                                sizeof(std::uint16_t),
                            cudaMemcpyDeviceToDevice, stream));
                    }
                }
                return;
            }
            // The qkv/z and b/a fusions are independent: b/a is always
            // fused (tiny weights, same per-GEMV floor), qkv/z only when
            // the weight duplication fits.
            // `src0` indexes the fire's own tokens (norm_x, z); `dst0` the
            // extended recurrence layout (mixed_qkv, a, b).
            auto in_proj_rows = [&](int src0, int dst0, int rows) {
                if (rows <= 0) return;
                const auto* x =
                    static_cast<const std::uint16_t*>(ws.norm_x.data()) +
                    static_cast<std::size_t>(src0) * H;
                if (Lw.la_in_proj_qkvz != nullptr) {
                    ops::gemm_act_x_wt_bf16(cublas.handle(),
                        x, Lw.la_in_proj_qkvz->data(),
                        la.mixed_qkvz.data(), rows, conv_dim + V_dim, H);
                    kernels::launch_split_bf16_rows(
                        la.mixed_qkvz.data(),
                        la.mixed_qkv.data() +
                            static_cast<std::size_t>(dst0) * conv_dim,
                        la.z.data() + static_cast<std::size_t>(src0) * V_dim,
                        rows, conv_dim, V_dim, stream);
                } else {
                    ops::gemm_act_x_wt_bf16(cublas.handle(),
                        x, Lw.la_in_proj_qkv->data(),
                        la.mixed_qkv.data() +
                            static_cast<std::size_t>(dst0) * conv_dim,
                        rows, conv_dim, H);
                    ops::gemm_act_x_wt_bf16(cublas.handle(),
                        x, Lw.la_in_proj_z->data(),
                        la.z.data() + static_cast<std::size_t>(src0) * V_dim,
                        rows, V_dim, H);
                }
                if (Lw.la_in_proj_ba != nullptr) {
                    ops::gemm_act_x_wt_bf16(cublas.handle(),
                        x, Lw.la_in_proj_ba->data(),
                        la.ba.data(), rows, 2 * V_h, H);
                    kernels::launch_split_qwen_gdn_ba_bf16(
                        la.ba.data(),
                        la.b.data() + static_cast<std::size_t>(dst0) * V_h,
                        la.a.data() + static_cast<std::size_t>(dst0) * V_h,
                        rows, V_h, stream);
                } else {
                    ops::gemm_act_x_wt_bf16(cublas.handle(),
                        x, Lw.la_in_proj_a->data(),
                        la.a.data() + static_cast<std::size_t>(dst0) * V_h,
                        rows, V_h, H);
                    ops::gemm_act_x_wt_bf16(cublas.handle(),
                        x, Lw.la_in_proj_b->data(),
                        la.b.data() + static_cast<std::size_t>(dst0) * V_h,
                        rows, V_h, H);
                }
            };
            if (!has_buffer_read) {
                in_proj_rows(0, 0, N);
            } else {
                for (int r = 0; r < R; ++r) {
                    const int src0 = static_cast<int>(qo_new_h[r]);
                    const int rows =
                        static_cast<int>(qo_new_h[r + 1]) - src0;
                    in_proj_rows(
                        src0,
                        static_cast<int>(qo_indptr_h[r]) + read_len_for(r),
                        rows);
                }
            }
            if (has_buffer_read) {
                // Gather each request's buffered in-proj activations into rows
                // [qo_ext[r], qo_ext[r] + B_r): the prefix the recurrence must
                // replay before it reaches this fire's own tokens.
                const int page = state_cache.rs_buffer_page_tokens();
                const std::size_t slab_a =
                    static_cast<std::size_t>(page) * conv_dim;
                const std::size_t slab_b =
                    slab_a + static_cast<std::size_t>(page) * V_h;
                for (int r = 0; r < R; ++r) {
                    const int qo0 = static_cast<int>(qo_indptr_h[r]);
                    const int br  = read_len_for(r);
                    if (br <= 0) continue;
                    const int head = head_for(r);
                    const int page_first = (head / page) * page;
                    const std::uint32_t s0 = rs_buffer_read_indptr_h[r];
                    const std::uint32_t s1 = rs_buffer_read_indptr_h[r + 1];
                    for (std::uint32_t j = 0; s0 + j < s1; ++j) {
                        const int page_tok0 =
                            page_first + static_cast<int>(j) * page;
                        const int phys0 = std::max(page_tok0, head);
                        const int cnt =
                            std::min(page_tok0 + page, head + br) - phys0;
                        if (cnt <= 0) break;
                        const std::size_t in_page =
                            static_cast<std::size_t>(phys0 - page_tok0);
                        const int tok0 = phys0 - head;
                        auto* slab = static_cast<std::uint16_t*>(
                            state_cache.rs_buffer_slab(
                                linear_idx,
                                static_cast<int>(
                                    rs_buffer_read_slot_ids_h[s0 + j])));
                        if (slab == nullptr) {
                            throw std::runtime_error(
                                "RS buffer read names a slab this layer has "
                                "not allocated");
                        }
                        CUDA_CHECK(cudaMemcpyAsync(
                            la.mixed_qkv.data() +
                                static_cast<std::size_t>(qo0 + tok0) * conv_dim,
                            slab + in_page * conv_dim,
                            static_cast<std::size_t>(cnt) * conv_dim *
                                sizeof(std::uint16_t),
                            cudaMemcpyDeviceToDevice, stream));
                        CUDA_CHECK(cudaMemcpyAsync(
                            la.a.data() +
                                static_cast<std::size_t>(qo0 + tok0) * V_h,
                            slab + slab_a + in_page * V_h,
                            static_cast<std::size_t>(cnt) * V_h *
                                sizeof(std::uint16_t),
                            cudaMemcpyDeviceToDevice, stream));
                        CUDA_CHECK(cudaMemcpyAsync(
                            la.b.data() +
                                static_cast<std::size_t>(qo0 + tok0) * V_h,
                            slab + slab_b + in_page * V_h,
                            static_cast<std::size_t>(cnt) * V_h *
                                sizeof(std::uint16_t),
                            cudaMemcpyDeviceToDevice, stream));
                    }
                }
            }
            if (rs_buffer_write) {
                const int page = state_cache.rs_buffer_page_tokens();
                const std::size_t slab_a =
                    static_cast<std::size_t>(page) * conv_dim;
                const std::size_t slab_b =
                    slab_a + static_cast<std::size_t>(page) * V_h;
                for (int r = 0; r < R; ++r) {
                    const int qo0 = static_cast<int>(qo_indptr_h[r]);
                    const int nr =
                        static_cast<int>(qo_indptr_h[r + 1]) - qo0;
                    // Only rows at or past B_r are this fire's to write; the
                    // ones below were gathered from the buffer. page_span()
                    // starts at the page CONTAINING B_r, so the first listed
                    // page can begin before the appended span.
                    const int head = head_for(r);
                    const int w0 = head + read_len_for(r);
                    const int wn = nr - read_len_for(r);
                    if (wn <= 0) continue;
                    const int page_first = (w0 / page) * page;
                    const std::uint32_t s0 =
                        rs_buffer_slot_indptr_h[r];
                    const std::uint32_t s1 =
                        rs_buffer_slot_indptr_h[r + 1];
                    for (std::uint32_t j = 0; s0 + j < s1; ++j) {
                        const int page_tok0 =
                            page_first + static_cast<int>(j) * page;
                        const int phys0 = std::max(page_tok0, w0);
                        const int count =
                            std::min(page_tok0 + page, w0 + wn) - phys0;
                        if (count <= 0) break;
                        const std::size_t in_page =
                            static_cast<std::size_t>(phys0 - page_tok0);
                        const int tok0 = phys0 - head;
                        auto* slab = static_cast<std::uint16_t*>(
                            state_cache.rs_buffer_slab(
                                linear_idx,
                                static_cast<int>(
                                    rs_buffer_slot_ids_h[s0 + j])));
                        launch_copy_if_valid_slot(
                            reinterpret_cast<const std::uint8_t*>(
                                la.mixed_qkv.data() +
                                static_cast<std::size_t>(qo0 + tok0) * conv_dim),
                            reinterpret_cast<std::uint8_t*>(
                                slab + in_page * conv_dim),
                            static_cast<std::size_t>(count) * conv_dim *
                                sizeof(std::uint16_t),
                            slot_ids_d, r, stream);
                        launch_copy_if_valid_slot(
                            reinterpret_cast<const std::uint8_t*>(
                                la.a.data() +
                                static_cast<std::size_t>(qo0 + tok0) * V_h),
                            reinterpret_cast<std::uint8_t*>(
                                slab + slab_a + in_page * V_h),
                            static_cast<std::size_t>(count) * V_h *
                                sizeof(std::uint16_t),
                            slot_ids_d, r, stream);
                        launch_copy_if_valid_slot(
                            reinterpret_cast<const std::uint8_t*>(
                                la.b.data() +
                                static_cast<std::size_t>(qo0 + tok0) * V_h),
                            reinterpret_cast<std::uint8_t*>(
                                slab + slab_b + in_page * V_h),
                            static_cast<std::size_t>(count) * V_h *
                                sizeof(std::uint16_t),
                            slot_ids_d, r, stream);
                    }
                }
            }
        });

    profile_cuda_detail_stage(
        profile, profile ? &profile->linear_conv_ms : nullptr,
        stream, [&] {
        auto* qkv_in_base   = static_cast<std::uint16_t*>(qkv_in_data);
        auto* qkv_post_base = la.mixed_qkv_post.data();
        if (linear_decode) {
            if (slot_ids_d != nullptr) {
                kernels::launch_causal_conv1d_update_batched_bf16(
                    qkv_in_base, Lw.la_conv1d_w->data(),
                    Lw.la_conv1d_b ? Lw.la_conv1d_b->data() : nullptr,
                    state_cache.conv_state(layer_idx, /*slot=*/0),
                    slot_ids_d,
                    static_cast<long long>(state_cache.conv_kernel()) *
                        state_cache.conv_dim(),
                    qkv_post_base,
                    R, conv_dim, conv_K, stream);
            } else {
                kernels::launch_causal_conv1d_update_bf16(
                    qkv_in_base, Lw.la_conv1d_w->data(),
                    Lw.la_conv1d_b ? Lw.la_conv1d_b->data() : nullptr,
                    state_cache.conv_state(layer_idx, 0),
                    qkv_post_base,
                    conv_dim, conv_K, stream);
            }
        } else {
            if (slot_ids_d != nullptr && qo_indptr_d != nullptr) {
                kernels::launch_causal_conv1d_prefill_batched_bf16(
                    qkv_in_base, Lw.la_conv1d_w->data(),
                    Lw.la_conv1d_b ? Lw.la_conv1d_b->data() : nullptr,
                    qkv_post_base,
                    state_cache.conv_state(layer_idx, /*slot=*/0),
                    slot_ids_d, qo_indptr_d,
                    static_cast<long long>(state_cache.conv_kernel()) *
                        state_cache.conv_dim(),
                    R, conv_dim, conv_K, stream, write_state, commit_len,
                    rs_write_state_mask);
            } else {
                for (int r = 0; r < R; ++r) {
                    const int t0 = static_cast<int>(qo_indptr_h[r]);
                    const int Nr = static_cast<int>(qo_indptr_h[r + 1]) - t0;
                    if (Nr <= 0) continue;
                    const std::size_t off = static_cast<std::size_t>(t0) * conv_dim;
                    kernels::launch_causal_conv1d_prefill_bf16(
                        qkv_in_base + off, Lw.la_conv1d_w->data(),
                        Lw.la_conv1d_b ? Lw.la_conv1d_b->data() : nullptr,
                        qkv_post_base + off,
                        state_cache.conv_state(layer_idx, slot_for(r)),
                        Nr, conv_dim, conv_K, stream);
                }
            }
        }
    });

    auto* qkv_base = la.mixed_qkv_post.data();
    const bool use_warp_tiled_recurrent =
        !linear_decode &&
        slot_ids_d != nullptr &&
        qo_indptr_d != nullptr &&
        N <= qwen35_gdn_warp_tiled_max_tokens() &&
        K_d <= 256 &&
        commit_len == nullptr;
    // Same reasoning as the dense forward: V_h == K_h is the repeat=1 case,
    // not a shape the GQA kernel cannot express, and excluding it only hid
    // the tuned SMEM step kernel from models with equal linear key/value
    // head counts. See qwen3_5_forward.cpp for the full argument and
    // driver/cuda/tests/gdn_recurrent_step_parity.cu for the guard. No
    // Qwen3.5 MoE checkpoint that fits this box has equal head counts, so
    // this arm rides on the equivalence proof rather than a local measurement.
    const bool use_decode_gqa_recurrent =
        linear_decode &&
        slot_ids_d != nullptr &&
        V_h % K_h == 0;
    profile_cuda_detail_stage(
        profile, profile ? &profile->linear_prep_ms : nullptr,
        stream, [&] {
        kernels::launch_qwen_gdn_post_conv_prep_bf16(
            qkv_base, a_data, b_data,
            Lw.la_A_log_fp32, Lw.la_dt_bias->data(),
            la.q_pre.data(), la.k_pre.data(), la.v_fp32.data(),
            la.g_log.data(), la.beta.data(),
            N, K_h, V_h, K_d, V_d, conv_dim, stream);

        if (V_h != K_h && !use_warp_tiled_recurrent &&
            !use_decode_gqa_recurrent) {
            kernels::launch_repeat_interleave_heads_fp32(
                la.q_pre.data(), la.q_norm.data(), N, K_h, V_h, K_d, stream);
            kernels::launch_repeat_interleave_heads_fp32(
                la.k_pre.data(), la.k_norm.data(), N, K_h, V_h, K_d, stream);
        }
    });
    invoke_stage_hook(
        StageHookPoint::OnAttnProj, la.q_pre.data(),
        static_cast<std::uint32_t>(N),
        static_cast<std::uint32_t>(K_h * K_d),
        static_cast<std::uint32_t>(layer_idx), stream,
        /*query_is_f32=*/true);
    const float* q_recur_full =
        (V_h == K_h) ? la.q_pre.data() : la.q_norm.data();
    const float* k_recur_full =
        (V_h == K_h) ? la.k_pre.data() : la.k_norm.data();

    profile_cuda_detail_stage(
        profile, profile ? &profile->linear_recur_ms : nullptr,
        stream, [&] {
            const std::size_t qk_step = static_cast<std::size_t>(V_h) * K_d;
            const std::size_t v_step  = static_cast<std::size_t>(V_dim);
            const std::size_t gh_step = static_cast<std::size_t>(V_h);
            const bool state_bf16 = state_cache.recurrent_state_bf16();
            void* state_slot0 = state_cache.recurrent_state_raw(
                layer_idx, /*slot=*/0);
            const auto slot_stride = static_cast<long long>(
                state_cache.recurrent_slot_stride_floats());
            if (linear_decode) {
                if (slot_ids_d != nullptr) {
                    if (use_decode_gqa_recurrent) {
                        if (state_bf16) {
                            kernels::launch_recurrent_gated_delta_step_batched_gqa_state_bf16(
                                la.q_pre.data(),
                                la.k_pre.data(),
                                la.v_fp32.data(),
                                la.g_log.data(),
                                la.beta.data(),
                                state_slot0,
                                slot_ids_d,
                                slot_stride,
                                la.core_out.data(),
                                R, K_h, V_h, K_d, V_d, stream);
                        } else {
                            kernels::launch_recurrent_gated_delta_step_batched_gqa(
                                la.q_pre.data(),
                                la.k_pre.data(),
                                la.v_fp32.data(),
                                la.g_log.data(),
                                la.beta.data(),
                                static_cast<float*>(state_slot0),
                                slot_ids_d,
                                slot_stride,
                                la.core_out.data(),
                                R, K_h, V_h, K_d, V_d, stream);
                        }
                    } else {
                        if (state_bf16) {
                            kernels::launch_recurrent_gated_delta_step_batched_state_bf16(
                                q_recur_full,
                                k_recur_full,
                                la.v_fp32.data(),
                                la.g_log.data(),
                                la.beta.data(),
                                state_slot0,
                                slot_ids_d,
                                slot_stride,
                                la.core_out.data(),
                                R, V_h, K_d, V_d, stream);
                        } else {
                            kernels::launch_recurrent_gated_delta_step_batched(
                                q_recur_full,
                                k_recur_full,
                                la.v_fp32.data(),
                                la.g_log.data(),
                                la.beta.data(),
                                static_cast<float*>(state_slot0),
                                slot_ids_d,
                                slot_stride,
                                la.core_out.data(),
                                R, V_h, K_d, V_d, stream);
                        }
                    }
                } else {
                    if (state_bf16) {
                        kernels::launch_recurrent_gated_delta_step_state_bf16(
                            q_recur_full,
                            k_recur_full,
                            la.v_fp32.data(),
                            la.g_log.data(),
                            la.beta.data(),
                            state_slot0,
                            la.core_out.data(),
                            /*B=*/1, V_h, K_d, V_d, stream);
                    } else {
                        kernels::launch_recurrent_gated_delta_step(
                            q_recur_full,
                            k_recur_full,
                            la.v_fp32.data(),
                            la.g_log.data(),
                            la.beta.data(),
                            static_cast<float*>(state_slot0),
                            la.core_out.data(),
                            /*B=*/1, V_h, K_d, V_d, stream);
                    }
                }
            } else {
                if (slot_ids_d != nullptr && qo_indptr_d != nullptr) {
                    if (use_warp_tiled_recurrent && V_h != K_h) {
                        if (state_bf16) {
                            kernels::launch_chunk_gated_delta_prefill_batched_warp_tiled_gqa_state_bf16(
                                la.q_pre.data(),
                                la.k_pre.data(),
                                la.v_fp32.data(),
                                la.g_log.data(),
                                la.beta.data(),
                                state_slot0,
                                slot_ids_d, qo_indptr_d,
                                slot_stride,
                                la.core_out.data(),
                                R, K_h, V_h, K_d, V_d,
                                stream, write_state, rs_write_state_mask);
                        } else {
                            kernels::launch_chunk_gated_delta_prefill_batched_warp_tiled_gqa(
                                la.q_pre.data(),
                                la.k_pre.data(),
                                la.v_fp32.data(),
                                la.g_log.data(),
                                la.beta.data(),
                                static_cast<float*>(state_slot0),
                                slot_ids_d, qo_indptr_d,
                                slot_stride,
                                la.core_out.data(),
                                R, K_h, V_h, K_d, V_d,
                                stream, write_state, rs_write_state_mask);
                        }
                    } else if (use_warp_tiled_recurrent) {
                        if (state_bf16) {
                            kernels::launch_chunk_gated_delta_prefill_batched_warp_tiled_state_bf16(
                                q_recur_full,
                                k_recur_full,
                                la.v_fp32.data(),
                                la.g_log.data(),
                                la.beta.data(),
                                state_slot0,
                                slot_ids_d, qo_indptr_d,
                                slot_stride,
                                la.core_out.data(),
                                R, V_h, K_d, V_d,
                                stream, write_state, rs_write_state_mask);
                        } else {
                            kernels::launch_chunk_gated_delta_prefill_batched_warp_tiled(
                                q_recur_full,
                                k_recur_full,
                                la.v_fp32.data(),
                                la.g_log.data(),
                                la.beta.data(),
                                static_cast<float*>(state_slot0),
                                slot_ids_d, qo_indptr_d,
                                slot_stride,
                                la.core_out.data(),
                                R, V_h, K_d, V_d,
                                stream, write_state, rs_write_state_mask);
                        }
                    } else if (
                        commit_len == nullptr &&
                        N <= qwen35_gdn_cached_prefill_max_tokens()) {
                        if (state_bf16) {
                            kernels::launch_chunk_gated_delta_prefill_batched_cached_state_bf16(
                                q_recur_full,
                                k_recur_full,
                                la.v_fp32.data(),
                                la.g_log.data(),
                                la.beta.data(),
                                state_slot0,
                                slot_ids_d, qo_indptr_d,
                                slot_stride,
                                la.core_out.data(),
                                R, V_h, K_d, V_d,
                                stream, write_state, rs_write_state_mask);
                        } else {
                            kernels::launch_chunk_gated_delta_prefill_batched_cached(
                                q_recur_full,
                                k_recur_full,
                                la.v_fp32.data(),
                                la.g_log.data(),
                                la.beta.data(),
                                static_cast<float*>(state_slot0),
                                slot_ids_d, qo_indptr_d,
                                slot_stride,
                                la.core_out.data(),
                                R, V_h, K_d, V_d,
                                stream, write_state, rs_write_state_mask);
                        }
                    } else {
                        if (state_bf16) {
                            // K_h==V_h identity: the MoE path still materialises
                            // the expanded q/k (no use_batched_fla_gqa here), so
                            // the GQA-aware kernel sees h_k==h and is unchanged.
                            kernels::launch_chunk_gated_delta_prefill_batched_state_bf16(
                                q_recur_full,
                                k_recur_full,
                                la.v_fp32.data(),
                                la.g_log.data(),
                                la.beta.data(),
                                state_slot0,
                                slot_ids_d, qo_indptr_d,
                                slot_stride,
                                la.core_out.data(),
                                R, V_h, V_h, K_d, V_d, stream, write_state,
                                commit_len);
                        } else {
                            kernels::launch_chunk_gated_delta_prefill_batched(
                                q_recur_full,
                                k_recur_full,
                                la.v_fp32.data(),
                                la.g_log.data(),
                                la.beta.data(),
                                static_cast<float*>(state_slot0),
                                slot_ids_d, qo_indptr_d,
                                slot_stride,
                                la.core_out.data(),
                                R, V_h, V_h, K_d, V_d, stream, write_state,
                                commit_len);
                        }
                    }
                } else {
                    for (int r = 0; r < R; ++r) {
                        const int t0 = static_cast<int>(qo_indptr_h[r]);
                        const int Nr = static_cast<int>(qo_indptr_h[r + 1]) - t0;
                        if (Nr <= 0) continue;
                        const std::size_t qk_off = static_cast<std::size_t>(t0) * qk_step;
                        const std::size_t v_off  = static_cast<std::size_t>(t0) * v_step;
                        const std::size_t gh_off = static_cast<std::size_t>(t0) * gh_step;
                        void* state_slot = state_cache.recurrent_state_raw(
                            layer_idx, slot_for(r));
                        if (state_bf16) {
                            kernels::launch_chunk_gated_delta_prefill_state_bf16(
                                q_recur_full + qk_off,
                                k_recur_full + qk_off,
                                la.v_fp32.data() + v_off,
                                la.g_log.data()  + gh_off,
                                la.beta.data()   + gh_off,
                                state_slot,
                                la.core_out.data() + v_off,
                                Nr, V_h, K_d, V_d, /*chunk_size=*/64, stream);
                        } else {
                            kernels::launch_chunk_gated_delta_prefill(
                                q_recur_full + qk_off,
                                k_recur_full + qk_off,
                                la.v_fp32.data() + v_off,
                                la.g_log.data()  + gh_off,
                                la.beta.data()   + gh_off,
                                static_cast<float*>(state_slot),
                                la.core_out.data() + v_off,
                                Nr, V_h, K_d, V_d, /*chunk_size=*/64, stream);
                        }
                    }
                }
            }
        });
    invoke_stage_hook(
        StageHookPoint::OnAttn, la.q_pre.data(),
        static_cast<std::uint32_t>(N),
        static_cast<std::uint32_t>(K_h * K_d),
        static_cast<std::uint32_t>(layer_idx), stream,
        /*query_is_f32=*/true);
    if (commit_len != nullptr) return;

    // Back to the fire's own token space: drop each request's replayed prefix.
    // The shift is left-overlapping, so it cannot be done in place; `v_fp32`
    // has the identical shape and is dead once the FLA returns, which makes it
    // a free landing pad for a path most fires never take.
    const float* core_rows = la.core_out.data();
    if (has_buffer_read) {
        float* packed = la.v_fp32.data();
        for (int r = 0; r < R; ++r) {
            const int src0 =
                static_cast<int>(qo_indptr_h[r]) + read_len_for(r);
            const int dst0 = static_cast<int>(qo_new_h[r]);
            const int rows = static_cast<int>(qo_new_h[r + 1]) - dst0;
            if (rows <= 0) continue;
            CUDA_CHECK(cudaMemcpyAsync(
                packed + static_cast<std::size_t>(dst0) * V_dim,
                la.core_out.data() + static_cast<std::size_t>(src0) * V_dim,
                static_cast<std::size_t>(rows) * V_dim * sizeof(float),
                cudaMemcpyDeviceToDevice, stream));
        }
        core_rows = packed;
    }

    profile_cuda_detail_stage(
        profile, profile ? &profile->linear_post_ms : nullptr,
        stream, [&] {
    kernels::launch_rmsnorm_gated_fp32_in_bf16(
        core_rows, z_data, Lw.la_norm_w_fp32,
        la.core_out_bf16.data(),
        N_new * V_h, V_d, /*eps=*/cfg.rms_norm_eps, stream);
    // out_proj: TP=1 fuses residual via beta=1; TP>1 row-parallel +
    // all-reduce + residual-add.
    if (T == 1) {
        ops::gemm_act_x_wt_bf16(cublas.handle(),
            la.core_out_bf16.data(), Lw.la_out_proj->data(),
            ws.y.data(), N_new, H, V_dim, /*beta=*/1.f);
    } else {
        ops::gemm_act_x_wt_bf16(cublas.handle(),
            la.core_out_bf16.data(), Lw.la_out_proj->data(),
            ws.norm_y.data(), N_new, H, V_dim, /*beta=*/0.f);
        tp->all_reduce_bf16(ws.norm_y.data(),
            static_cast<std::size_t>(N_new) * H, ncclSum, stream);
        kernels::launch_residual_add_bf16(
            ws.y.data(), ws.norm_y.data(),
            static_cast<std::size_t>(N_new) * H, stream);
    }
    });
}

// Full-attention body (replica of qwen3_5_forward.cpp's logic).
void full_attn_body(
    const Qwen3_5MoeLayerWeights& Lw,
    const HfConfig& cfg,
    const Qwen3_5ForwardCfg& fwd_cfg,
    Workspace& ws,
    Qwen3_5LinearAttnWorkspace& la,
    KvCache& cache, AttentionWorkspace& attn_ws,
    const ops::DecodePlanCache* decode_plan,
    const ops::PrefillPlanCache* prefill_plan,
    int model_layer, int kv_layer, int N, int R,
    const std::int32_t* positions,
    const std::uint32_t* qo_indptr,
    const std::uint32_t* kv_page_indices,
    const std::uint32_t* kv_page_indptr,
    const std::uint32_t* kv_last_page_lens,
    const std::uint32_t* qo_indptr_h,
    const std::uint32_t* kv_page_indptr_h,
    const std::uint32_t* w_page_d,
    const std::uint32_t* w_off_d,
    const std::uint8_t* row_valid_d,
    bool has_write_desc,
    ops::CublasHandle& cublas, cudaStream_t stream)
{
    const int T  = std::max(1, fwd_cfg.tp_size);
    const int H  = cfg.hidden_size;
    const int num_q_heads_local  = cfg.num_attention_heads / T;
    const int num_kv_heads_local = cfg.num_key_value_heads / T;
    const int Hq = num_q_heads_local * cfg.head_dim;
    const int Hk = num_kv_heads_local * cfg.head_dim;
    const int d  = cfg.head_dim;
    const int rotary_dim = std::max<int>(2,
        2 * static_cast<int>(0.5f * cfg.partial_rotary_factor * d));
    const float eps = cfg.rms_norm_eps;
    NcclComm* tp = (T > 1) ? fwd_cfg.tp_comm : nullptr;

    // Qwen3.5 / 3.6-MoE fuse the per-head sigmoid output gate into
    // q_proj as a [2*Hq, H] tensor — rows [0,Hq) are q, rows [Hq,2*Hq)
    // are the gate logits. Qwen3-MoE (Qwen3-30B-A3B) ships plain q_proj
    // [Hq, H] with no output gate, so the GEMM goes straight into ws.q.
    if (cfg.attn_output_gate) {
        ops::gemm_act_x_w(cublas.handle(),
            ws.norm_x.data(), make_weight_view(Lw.fa_q_proj, Lw.fa_q_proj_quant),
            la.fa_qg_packed.data(), N, 2 * Hq, H);
        kernels::launch_split_q_gate_bf16(
            la.fa_qg_packed.data(), ws.q.data(), la.fa_gate.data(),
            N, num_q_heads_local, d, stream);
    } else {
        ops::gemm_act_x_w(cublas.handle(),
            ws.norm_x.data(), make_weight_view(Lw.fa_q_proj, Lw.fa_q_proj_quant),
            ws.q.data(), N, Hq, H);
    }

    ops::gemm_act_x_w(cublas.handle(),
        ws.norm_x.data(), make_weight_view(Lw.fa_k_proj, Lw.fa_k_proj_quant),
        ws.k.data(), N, Hk, H);
    ops::gemm_act_x_w(cublas.handle(),
        ws.norm_x.data(), make_weight_view(Lw.fa_v_proj, Lw.fa_v_proj_quant),
        ws.v.data(), N, Hk, H);

    rmsnorm_bf16_dispatch(cfg,
        ws.q.data(), Lw.fa_q_norm->data(), ws.q.data(),
        N * num_q_heads_local, d, eps, stream);
    rmsnorm_bf16_dispatch(cfg,
        ws.k.data(), Lw.fa_k_norm->data(), ws.k.data(),
        N * num_kv_heads_local, d, eps, stream);

    kernels::launch_rope_partial_bf16(
        ws.q.data(), ws.k.data(), positions,
        N, num_q_heads_local, num_kv_heads_local,
        d, rotary_dim, cfg.rope_theta, stream);
    // Fires POST-rope (and post q/k-norm): the query a PTIR program
    // observes here is the one that actually enters attention, so an
    // observer scoring it against the cached keys -- which are stored
    // post-rope -- compares in the same space.
    invoke_stage_hook(
        StageHookPoint::OnAttnProj, ws.q.data(),
        static_cast<std::uint32_t>(N),
        static_cast<std::uint32_t>(Hq),
        static_cast<std::uint32_t>(model_layer), stream);

    auto kv_view = cache.layer_view(kv_layer);
    if (has_write_desc) {
        kernels::launch_write_kv_explicit_bf16(
            kv_view, ws.k.data(), ws.v.data(),
            w_page_d, w_off_d, N, stream, row_valid_d);
    } else {
        kernels::launch_write_kv_to_pages(
            kv_view, ws.k.data(), ws.v.data(),
            qo_indptr, kv_page_indices, kv_page_indptr, kv_last_page_lens,
            N, R, stream);
    }

    // Decode and planned-prefill paths are graph-friendly: the host-side
    // FlashInfer planning was hoisted to the executor prepare hook.
    const bool use_small_prefill_naive =
        decode_plan == nullptr &&
        prefill_plan == nullptr &&
        fwd_cfg.small_prefill_naive_attention_max_tokens > 0 &&
        N <= fwd_cfg.small_prefill_naive_attention_max_tokens &&
        kv_view.is_native_bf16() && !kv_view.hnd_layout;
    if (decode_plan) {
        ops::dispatch_attention_flashinfer_decode(
            *decode_plan,
            ws.q.data(), kv_view, ws.attn_out.data(),
            kv_page_indices, kv_page_indptr, kv_last_page_lens,
            attn_ws, stream);
    } else if (prefill_plan) {
        ops::dispatch_attention_flashinfer_prefill_bf16(
            *prefill_plan,
            ws.q.data(), kv_view.k_bf16_pages, kv_view.v_bf16_pages,
            ws.attn_out.data(),
            qo_indptr, kv_page_indices, kv_page_indptr, kv_last_page_lens,
            attn_ws, stream);
    } else if (use_small_prefill_naive) {
        ops::launch_attention_naive_paged_bf16(
            ws.q.data(), kv_view.k_bf16_pages, kv_view.v_bf16_pages,
            ws.attn_out.data(),
            qo_indptr, kv_page_indices, kv_page_indptr, kv_last_page_lens,
            N, R, num_q_heads_local, num_kv_heads_local, d,
            cache.page_size(), stream);
    } else {
        ops::launch_attention_flashinfer_prefill(
            ws.q.data(), kv_view, ws.attn_out.data(),
            qo_indptr, kv_page_indices, kv_page_indptr, kv_last_page_lens,
            qo_indptr_h, kv_page_indptr_h,
            N, R, num_q_heads_local, attn_ws, stream);
    }
    if (cfg.attn_output_gate) {
        kernels::launch_sigmoid_gate_inplace_bf16(
            ws.attn_out.data(), la.fa_gate.data(), N * Hq, stream);
    }
    invoke_stage_hook(
        StageHookPoint::OnAttn, ws.q.data(),
        static_cast<std::uint32_t>(N),
        static_cast<std::uint32_t>(Hq),
        static_cast<std::uint32_t>(model_layer), stream);

    // o_proj: TP=1 fuses residual via beta=1; TP>1 row-parallel +
    // all-reduce + residual-add.
    if (T == 1) {
        ops::gemm_act_x_w(cublas.handle(),
            ws.attn_out.data(), make_weight_view(Lw.fa_o_proj, Lw.fa_o_proj_quant),
            ws.y.data(), N, H, Hq, /*beta=*/1.f);
    } else {
        ops::gemm_act_x_w(cublas.handle(),
            ws.attn_out.data(), make_weight_view(Lw.fa_o_proj, Lw.fa_o_proj_quant),
            ws.norm_y.data(), N, H, Hq, /*beta=*/0.f);
        tp->all_reduce_bf16(ws.norm_y.data(),
            static_cast<std::size_t>(N) * H, ncclSum, stream);
        kernels::launch_residual_add_bf16(
            ws.y.data(), ws.norm_y.data(),
            static_cast<std::size_t>(N) * H, stream);
    }
}

// MoE block: routed experts + shared expert with sigmoid gate.
// Reads `ws.norm_x`, writes the combined routed-expert + shared-expert
// contribution directly into `ws.norm_y` (the residual buffer the caller
// will add into `ws.y`).
bool moe_block(
    const Qwen3_5MoeLayerWeights& Lw,
    const HfConfig& cfg,
    const Qwen3_5ForwardCfg& fwd_cfg,
    Workspace& ws,
    Qwen3_5MoeMlpWorkspace& moe_ws,
    int N,
    bool is_pure_decode,
    ops::CublasHandle& cublas, cudaStream_t stream,
    Qwen35MoeForwardProfile* profile)
{
    const int T = std::max(1, fwd_cfg.tp_size);
    const int H = cfg.hidden_size;
    const int E = cfg.num_experts;
    const int K = cfg.num_experts_per_tok;
    // Both routed and shared experts shard along the intermediate axis
    // (column-parallel gate/up + row-parallel down). The engine load loop
    // streams per-rank slices of `experts.gate_up_proj` / `experts.down_proj`
    // straight from the safetensors mmap, so each rank only allocates its
    // own Im_local-sized portion and the per-expert GEMMs run at the
    // sharded width. We do one all-reduce at the end of the block,
    // covering both routed and shared partial sums.
    const int Im = cfg.moe_intermediate_size / T;            // routed: sharded
    const int Is = cfg.shared_expert_intermediate_size / T;  // shared: sharded
    // Set by the decode fast path when the shared expert's projections were
    // folded into the routed batched GEMM; the shared block below then only
    // has to apply the sigmoid scalar gate.
    bool moe_shared_folded = false;
    NcclComm* tp = (T > 1) ? fwd_cfg.tp_comm : nullptr;
    // Streamed experts have no fused slab, so every device-side path that
    // builds pointer arrays by striding one is off the table. The general
    // path already dispatches one expert at a time, which is exactly the
    // granularity a slot has.
    const bool streamed = Lw.expert_cache != nullptr;
    const bool use_decode_fast_path =
        !streamed && !qwen35_moe_force_general_path() &&
        (is_pure_decode ||
         (N > 0 && N <= qwen35_moe_decode_fast_max_tokens()));
    if (moe_path_log_enabled()) {
        // One line per distinct (N, pure_decode, path) triple, so a whole
        // run reports which shapes took which path without 40 lines per
        // step drowning the log.
        static std::mutex seen_mutex;
        static std::set<std::tuple<int, bool, bool>> seen;
        const auto key = std::make_tuple(N, is_pure_decode, use_decode_fast_path);
        bool fresh = false;
        {
            std::lock_guard<std::mutex> lock(seen_mutex);
            fresh = seen.insert(key).second;
        }
        if (fresh) {
            std::fprintf(
                stderr, "[pie-moe-path] N=%d pure_decode=%d fast_path=%d\n",
                N, is_pure_decode ? 1 : 0, use_decode_fast_path ? 1 : 0);
        }
    }
    const bool add_to_residual = (T == 1) && use_decode_fast_path;
    void* moe_out = add_to_residual ? ws.y.data() : ws.norm_y.data();

    // ── Routed experts ────────────────────────────────────────────
    // 1. Router logits.
    profile_cuda_stage(profile, profile ? &profile->moe_router_ms : nullptr,
        stream, [&] {
            ops::gemm_act_x_wt_bf16(cublas.handle(),
                ws.norm_x.data(), Lw.moe_router->data(),
                moe_ws.router_logits.data(), N, E, H);
            // 2. Top-K + softmax + renormalize.
            kernels::launch_topk_softmax_bf16(
                moe_ws.router_logits.data(),
                moe_ws.topk_idx.data(), moe_ws.topk_weights.data(),
                N, E, K, stream);
        });

    // 3. Routing decisions. The default pure-decode path stays entirely
    //    on-device (so the layer is graph-capturable). The prefill/mixed
    //    path needs host routing to bucket tokens per expert.
    std::vector<std::int32_t> topk_idx_h;
    std::vector<float>        topk_w_h;
    if (!use_decode_fast_path) {
        topk_idx_h.resize((std::size_t)N * K);
        topk_w_h.resize((std::size_t)N * K);
        CUDA_CHECK(cudaMemcpyAsync(topk_idx_h.data(), moe_ws.topk_idx.data(),
                                   topk_idx_h.size() * sizeof(std::int32_t),
                                   cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaMemcpyAsync(topk_w_h.data(), moe_ws.topk_weights.data(),
                                   topk_w_h.size() * sizeof(float),
                                   cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    // 4. (For the prefill/mixed path only: zero moe_out before scatter_add.)
    //    Decode fast-path weighted-sum overwrites norm_y, so the memset
    //    there would be wasted work.

    // 5. Per-expert dispatch.
    const std::size_t expert_stride_gu =
        static_cast<std::size_t>(2) * Im * H;  // bf16 elements per expert in gate_up_proj
    const std::size_t expert_stride_dn =
        static_cast<std::size_t>(H) * Im;       // bf16 elements per expert in down_proj

    if (use_decode_fast_path) {
        // Decode fast-path. Fully on-device pipeline (graph-capturable):
        //   1. Build gate_up/down cuBLAS pointer arrays for every
        //      token/expert route (N*K rows) with no D2H sync.
        //   2. `cublasGemmBatchedEx` for gate_up (N*K batches, M=1).
        //   3. `chunked_swiglu` over [N*K, 2*Im].
        //   4. `cublasGemmBatchedEx` for down_proj (N*K batches, M=1).
        //   5. Weighted sum collapses [N, K, H] -> [N, H].
        //
        // Every step has fixed kernel topology and stable device-pointer
        // arguments, so the executor's graph-capture path can fire
        // for the whole forward.
        profile_cuda_stage(profile, profile ? &profile->moe_routed_ms : nullptr,
            stream, [&] {
                const int routes = N * K;
                // Weights are already sharded per rank, so the runner is told
                // tp_size=1 and given this rank's own slice.
                // The runner overwrites its output, so when the caller
                // wanted the residual folded in it writes to scratch and a
                // separate add follows -- still far cheaper than the path
                // this replaces, and it is what makes tp=1 (where
                // `add_to_residual` is always set) reach this kernel.
                void* fused_out = add_to_residual ? ws.norm_y.data() : moe_out;
                if (!moe_ws.cutlass_ws.empty() &&
                    N <= moe_ws.cutlass_max_rows &&
                    ops::flashinfer_cutlass_moe_bf16(
                        ops::MoeActivation::Swiglu,
                        static_cast<const std::uint16_t*>(ws.norm_x.data()),
                        moe_ws.topk_idx.data(),
                        moe_ws.topk_weights.data(),
                        static_cast<const std::uint16_t*>(
                            Lw.moe_gate_up_proj->data()),
                        static_cast<const std::uint16_t*>(
                            Lw.moe_down_proj->data()),
                        static_cast<std::uint16_t*>(fused_out),
                        moe_ws.cutlass_ws.data(),
                        moe_ws.cutlass_ws.size(),
                        moe_ws.cutlass_row_map.data(),
                        N, H, Im, E, K,
                        /*tp_size=*/1, /*tp_rank=*/0, stream)) {
                    if (add_to_residual) {
                        kernels::launch_residual_add_bf16(
                            moe_out, ws.norm_y.data(),
                            static_cast<std::size_t>(N) * H, stream);
                    }
                    return;
                }
                const int block = moe_ws.aligned_block_size;
                const bool use_aligned_decode =
                    block > 1 &&
                    routes >= qwen35_moe_aligned_decode_min_routes() &&
                    !moe_ws.aligned_expert_in.empty();
                if (use_aligned_decode) {
                    const int active_expert_cap = std::min(E, routes);
                    const int routed_blocks =
                        (routes + active_expert_cap * (block - 1) +
                         block - 1) / block;
                    // Fold the shared expert in as one more expert. Its
                    // weights are [2*Is, H] and [H, Is], which are EXACTLY the
                    // routed expert shapes when Is == Im, so the same batched
                    // GEMM covers both. That replaces two GEMMs that cuBLAS
                    // covers with only ~4 thread blocks (N=512 on 108 SMs)
                    // with ceil(N/block) more blocks of an already-saturated
                    // one.
                    const bool fold_shared = shared_fold_enabled() &&
                        Is > 0 && Is == Im &&
                        Lw.shared_gate_up_proj != nullptr &&
                        Lw.shared_down_proj != nullptr &&
                        !Lw.shared_down_proj_quant.has_value();
                    const int shared_blocks =
                        fold_shared ? (N + block - 1) / block : 0;
                    const int max_blocks = routed_blocks + shared_blocks;
                    const int aligned_rows = max_blocks * block;
                    const int shared_row_begin =
                        fold_shared ? routed_blocks * block : -1;
                    moe_shared_folded = fold_shared;
                    if (static_cast<std::size_t>(aligned_rows) >
                        moe_ws.aligned_rows_capacity) {
                        throw std::runtime_error(
                            "qwen3.5-moe aligned decode scratch too small");
                    }

                    profile_cuda_detail_stage(
                        profile, profile ? &profile->moe_route_setup_ms : nullptr,
                        stream, [&] {
                            profile_cuda_detail_stage(
                                profile, profile ? &profile->moe_align_ms : nullptr,
                                stream, [&] {
                            kernels::launch_moe_align_decode(
                                moe_ws.topk_idx.data(),
                                moe_ws.aligned_route_ids.data(),
                                moe_ws.aligned_expert_ids.data(),
                                /*route_to_aligned_row=*/nullptr,
                                routes, E, block, routed_blocks, stream);
                                });
                            profile_cuda_detail_stage(
                                profile, profile ? &profile->moe_gather_ms : nullptr,
                                stream, [&] {
                            kernels::launch_gather_moe_aligned_inputs_bf16(
                                ws.norm_x.data(), moe_ws.aligned_route_ids.data(),
                                moe_ws.aligned_expert_in.data(),
                                routes, aligned_rows, K, H,
                                shared_row_begin, N, stream);
                                });
                            profile_cuda_detail_stage(
                                profile, profile ? &profile->moe_ptrs_ms : nullptr,
                                stream, [&] {
                            kernels::launch_build_moe_ptrs_aligned_bf16(
                                moe_ws.aligned_expert_ids.data(),
                                Lw.moe_gate_up_proj->data(),
                                Lw.moe_down_proj->data(),
                                moe_ws.aligned_expert_in.data(),
                                moe_ws.aligned_gate_up.data(),
                                moe_ws.aligned_act.data(),
                                moe_ws.aligned_out.data(),
                                reinterpret_cast<const void**>(moe_ws.a_gu_ptrs.data()),
                                reinterpret_cast<const void**>(moe_ws.b_gu_ptrs.data()),
                                reinterpret_cast<void**>(moe_ws.c_gu_ptrs.data()),
                                reinterpret_cast<const void**>(moe_ws.a_dn_ptrs.data()),
                                reinterpret_cast<const void**>(moe_ws.b_dn_ptrs.data()),
                                reinterpret_cast<void**>(moe_ws.c_dn_ptrs.data()),
                                max_blocks, block, H, Im,
                                fold_shared ? routed_blocks : max_blocks,
                                fold_shared ? Lw.shared_gate_up_proj->data()
                                            : nullptr,
                                fold_shared ? Lw.shared_down_proj->data()
                                            : nullptr,
                                stream);
                                });
                        });

                    // Aligned gate_up: M=block_size, N=2*Im, K=H.
                    // cuBLAS must be launched with the worst-case batch
                    // count under graph capture, and that bound cannot drop
                    // below the expert count while the routing needs about a
                    // third of it. The grouped kernel takes the same bound as
                    // a grid but returns immediately on padding blocks.
                    // Decided per projection: the kernel wins on the short-K
                    // one and loses on the long-K one, so they do not share
                    // an answer.
                    const bool grouped_ok =
                        moe_grouped_gemm_enabled() && !fold_shared;
                    const bool grouped_gu = grouped_ok &&
                        kernels::moe_grouped_gemm_bf16_supported(
                            block, 2 * Im, H);
                    const bool grouped_dn = grouped_ok &&
                        kernels::moe_grouped_gemm_bf16_supported(
                            block, H, Im);
                    profile_cuda_detail_stage(
                        profile, profile ? &profile->moe_gate_up_ms : nullptr,
                        stream, [&] {
                            if (grouped_gu) {
                                kernels::launch_moe_grouped_gemm_bf16(
                                    moe_ws.aligned_expert_in.data(),
                                    Lw.moe_gate_up_proj->data(),
                                    moe_ws.aligned_gate_up.data(),
                                    moe_ws.aligned_expert_ids.data(),
                                    max_blocks, block, 2 * Im, H, stream);
                            } else {
                                ops::gemm_batched_act_x_wt_bf16(cublas.handle(),
                                    reinterpret_cast<const void* const*>(
                                        moe_ws.b_gu_ptrs.data()),
                                    reinterpret_cast<const void* const*>(
                                        moe_ws.a_gu_ptrs.data()),
                                    reinterpret_cast<void* const*>(moe_ws.c_gu_ptrs.data()),
                                    block, 2 * Im, H, max_blocks);
                            }
                        });

                    profile_cuda_detail_stage(
                        profile, profile ? &profile->moe_act_ms : nullptr,
                        stream, [&] {
                            kernels::launch_chunked_swiglu_bf16(
                                moe_ws.aligned_gate_up.data(),
                                moe_ws.aligned_act.data(),
                                aligned_rows, Im, stream,
                                /*gate_second=*/moe_gate_up_swapped());
                        });

                    // Aligned down_proj: M=block_size, N=H, K=Im.
                    profile_cuda_detail_stage(
                        profile, profile ? &profile->moe_down_ms : nullptr,
                        stream, [&] {
                            if (grouped_dn) {
                                kernels::launch_moe_grouped_gemm_bf16(
                                    moe_ws.aligned_act.data(),
                                    Lw.moe_down_proj->data(),
                                    moe_ws.aligned_out.data(),
                                    moe_ws.aligned_expert_ids.data(),
                                    max_blocks, block, H, Im, stream);
                            } else {
                                ops::gemm_batched_act_x_wt_bf16(cublas.handle(),
                                    reinterpret_cast<const void* const*>(
                                        moe_ws.b_dn_ptrs.data()),
                                    reinterpret_cast<const void* const*>(
                                        moe_ws.a_dn_ptrs.data()),
                                    reinterpret_cast<void* const*>(moe_ws.c_dn_ptrs.data()),
                                    block, H, Im, max_blocks);
                            }
                        });

                    profile_cuda_detail_stage(
                        profile, profile ? &profile->moe_reduce_ms : nullptr,
                        stream, [&] {
                            kernels::launch_reorder_moe_aligned_output_bf16(
                                moe_ws.aligned_out.data(),
                                moe_ws.aligned_route_ids.data(),
                                moe_ws.expert_out.data(),
                                routes, aligned_rows, H,
                                shared_row_begin, N,
                                fold_shared ? moe_ws.shared_out.data()
                                            : nullptr,
                                stream);
                            if (add_to_residual) {
                                kernels::launch_token_batched_weighted_sum_add_bf16(
                                    moe_out, moe_ws.expert_out.data(),
                                    moe_ws.topk_weights.data(),
                                    N, K, H, stream);
                            } else {
                                kernels::launch_token_batched_weighted_sum_bf16(
                                    moe_out, moe_ws.expert_out.data(),
                                    moe_ws.topk_weights.data(),
                                    N, K, H, stream);
                            }
                        });
                } else if (qwen35_moe_gemv_decode_enabled() &&
                           (H % 8) == 0 && (Im % 8) == 0) {
                    profile_cuda_detail_stage(
                        profile, profile ? &profile->moe_gate_up_ms : nullptr,
                        stream, [&] {
                            kernels::launch_moe_gate_up_decode_gemv_bf16(
                                moe_ws.topk_idx.data(),
                                ws.norm_x.data(),
                                Lw.moe_gate_up_proj->data(),
                                moe_ws.expert_gate_up.data(),
                                N, K, H, Im, stream);
                        });

                    profile_cuda_detail_stage(
                        profile, profile ? &profile->moe_act_ms : nullptr,
                        stream, [&] {
                            kernels::launch_chunked_swiglu_bf16(
                                moe_ws.expert_gate_up.data(),
                                moe_ws.expert_act.data(),
                                routes, Im, stream,
                                /*gate_second=*/moe_gate_up_swapped());
                        });

                    profile_cuda_detail_stage(
                        profile, profile ? &profile->moe_down_ms : nullptr,
                        stream, [&] {
                            kernels::launch_moe_down_decode_gemv_bf16(
                                moe_ws.topk_idx.data(),
                                moe_ws.expert_act.data(),
                                Lw.moe_down_proj->data(),
                                moe_ws.expert_out.data(),
                                N, K, H, Im, stream);
                        });

                    profile_cuda_detail_stage(
                        profile, profile ? &profile->moe_reduce_ms : nullptr,
                        stream, [&] {
                            if (add_to_residual) {
                                kernels::launch_token_batched_weighted_sum_add_bf16(
                                    moe_out, moe_ws.expert_out.data(),
                                    moe_ws.topk_weights.data(),
                                    N, K, H, stream);
                            } else {
                                kernels::launch_token_batched_weighted_sum_bf16(
                                    moe_out, moe_ws.expert_out.data(),
                                    moe_ws.topk_weights.data(),
                                    N, K, H, stream);
                            }
                        });
                } else if (qwen35_moe_wmma_decode_enabled() &&
                           (H % 16) == 0 && (Im % 16) == 0) {
                    profile_cuda_detail_stage(
                        profile, profile ? &profile->moe_gate_up_ms : nullptr,
                        stream, [&] {
                            kernels::launch_moe_gate_up_decode_wmma_bf16(
                                moe_ws.topk_idx.data(),
                                ws.norm_x.data(),
                                Lw.moe_gate_up_proj->data(),
                                moe_ws.expert_gate_up.data(),
                                N, K, H, Im, stream);
                        });

                    profile_cuda_detail_stage(
                        profile, profile ? &profile->moe_act_ms : nullptr,
                        stream, [&] {
                            kernels::launch_chunked_swiglu_bf16(
                                moe_ws.expert_gate_up.data(),
                                moe_ws.expert_act.data(),
                                routes, Im, stream,
                                /*gate_second=*/moe_gate_up_swapped());
                        });

                    profile_cuda_detail_stage(
                        profile, profile ? &profile->moe_down_ms : nullptr,
                        stream, [&] {
                            kernels::launch_moe_down_decode_wmma_bf16(
                                moe_ws.topk_idx.data(),
                                moe_ws.expert_act.data(),
                                Lw.moe_down_proj->data(),
                                moe_ws.expert_out.data(),
                                N, K, H, Im, stream);
                        });

                    profile_cuda_detail_stage(
                        profile, profile ? &profile->moe_reduce_ms : nullptr,
                        stream, [&] {
                            if (add_to_residual) {
                                kernels::launch_token_batched_weighted_sum_add_bf16(
                                    moe_out, moe_ws.expert_out.data(),
                                    moe_ws.topk_weights.data(),
                                    N, K, H, stream);
                            } else {
                                kernels::launch_token_batched_weighted_sum_bf16(
                                    moe_out, moe_ws.expert_out.data(),
                                    moe_ws.topk_weights.data(),
                                    N, K, H, stream);
                            }
                        });
                } else {
                    profile_cuda_detail_stage(
                        profile, profile ? &profile->moe_route_setup_ms : nullptr,
                        stream, [&] {
                            kernels::launch_build_moe_ptrs_decode_batched_bf16(
                                moe_ws.topk_idx.data(),
                                moe_ws.topk_weights.data(),
                                Lw.moe_gate_up_proj->data(),
                                Lw.moe_down_proj->data(),
                                ws.norm_x.data(),
                                moe_ws.expert_gate_up.data(),
                                moe_ws.expert_act.data(),
                                moe_ws.expert_out.data(),
                                reinterpret_cast<const void**>(moe_ws.a_gu_ptrs.data()),
                                reinterpret_cast<const void**>(moe_ws.b_gu_ptrs.data()),
                                reinterpret_cast<void**>(moe_ws.c_gu_ptrs.data()),
                                reinterpret_cast<const void**>(moe_ws.a_dn_ptrs.data()),
                                reinterpret_cast<const void**>(moe_ws.b_dn_ptrs.data()),
                                reinterpret_cast<void**>(moe_ws.c_dn_ptrs.data()),
                                moe_ws.batch_weights.data(),
                                N, K, H, Im, stream);
                        });

                    // gate_up batched GEMM: M=1, N=2*Im, K=H, batch=N*top_k.
                    profile_cuda_detail_stage(
                        profile, profile ? &profile->moe_gate_up_ms : nullptr,
                        stream, [&] {
                            ops::gemm_batched_act_x_wt_bf16(cublas.handle(),
                                reinterpret_cast<const void* const*>(
                                    moe_ws.b_gu_ptrs.data()),
                                reinterpret_cast<const void* const*>(
                                    moe_ws.a_gu_ptrs.data()),
                                reinterpret_cast<void* const*>(moe_ws.c_gu_ptrs.data()),
                                /*M=*/1, /*N=*/2 * Im, /*K=*/H,
                                /*batch_count=*/routes);
                        });

                    // SwiGLU on [N*top_k, 2*Im] -> [N*top_k, Im].
                    profile_cuda_detail_stage(
                        profile, profile ? &profile->moe_act_ms : nullptr,
                        stream, [&] {
                            kernels::launch_chunked_swiglu_bf16(
                                moe_ws.expert_gate_up.data(),
                                moe_ws.expert_act.data(),
                                routes, Im, stream,
                                /*gate_second=*/moe_gate_up_swapped());
                        });

                    // down_proj batched GEMM: M=1, N=H, K=Im, batch=N*top_k.
                    profile_cuda_detail_stage(
                        profile, profile ? &profile->moe_down_ms : nullptr,
                        stream, [&] {
                            ops::gemm_batched_act_x_wt_bf16(cublas.handle(),
                                reinterpret_cast<const void* const*>(
                                    moe_ws.b_dn_ptrs.data()),
                                reinterpret_cast<const void* const*>(
                                    moe_ws.a_dn_ptrs.data()),
                                reinterpret_cast<void* const*>(moe_ws.c_dn_ptrs.data()),
                                /*M=*/1, /*N=*/H, /*K=*/Im,
                                /*batch_count=*/routes);
                        });

                    // Sum each token's K routed outputs into norm_y.
                    profile_cuda_detail_stage(
                        profile, profile ? &profile->moe_reduce_ms : nullptr,
                        stream, [&] {
                            if (add_to_residual) {
                                kernels::launch_token_batched_weighted_sum_add_bf16(
                                    moe_out, moe_ws.expert_out.data(),
                                    moe_ws.batch_weights.data(),
                                    N, K, H, stream);
                            } else {
                                kernels::launch_token_batched_weighted_sum_bf16(
                                    moe_out, moe_ws.expert_out.data(),
                                    moe_ws.batch_weights.data(),
                                    N, K, H, stream);
                            }
                        });
                }
            });
    } else {
        // General path (prefill / multi-token). Build per-expert routing
        // lists on host and gather/scatter via the existing kernels.
        // Zero moe_out before the scatter_add accumulation.
        profile_cuda_stage(profile, profile ? &profile->moe_routed_ms : nullptr,
            stream, [&] {
                CUDA_CHECK(cudaMemsetAsync(ws.norm_y.data(), 0,
                    (std::size_t)N * H * sizeof(std::uint16_t), stream));
                const auto routing = build_routing(topk_idx_h, topk_w_h, N, K, E);
                if (streamed) {
                    // Routing is known for the whole layer before any of it
                    // runs, and the experts are paged one at a time, so say up
                    // front which ones are wanted. The read for expert e+1 then
                    // overlaps the GEMMs for expert e instead of following
                    // them. This is the only point in the pass that knows the
                    // order, and it is why prefetching is worth stating rather
                    // than leaving to the page cache's readahead.
                    for (int e = 0; e < E; ++e) {
                        if (routing.token_idx[e].empty()) continue;
                        Lw.expert_cache->prefetch(
                            Lw.expert_group, static_cast<std::uint32_t>(e));
                    }
                }
                for (int e = 0; e < E; ++e) {
                    const auto& tok_idx = routing.token_idx[e];
                    const auto& wts     = routing.weights[e];
                    const int Ne = static_cast<int>(tok_idx.size());
                    if (Ne == 0) continue;

                    CUDA_CHECK(cudaMemcpyAsync(
                        moe_ws.expert_idx.data(), tok_idx.data(),
                        Ne * sizeof(std::int32_t), cudaMemcpyHostToDevice, stream));
                    CUDA_CHECK(cudaMemcpyAsync(
                        moe_ws.expert_w.data(), wts.data(),
                        Ne * sizeof(float), cudaMemcpyHostToDevice, stream));

                    kernels::launch_gather_bf16_rows(
                        static_cast<const std::uint16_t*>(ws.norm_x.data()),
                        moe_ws.expert_idx.data(),
                        moe_ws.expert_in.data(),
                        Ne, H, stream);

                    const std::uint16_t* gate_up_w = nullptr;
                    const std::uint16_t* down_w = nullptr;
                    if (streamed) {
                        // Page it in. The slot holds exactly what one stride of
                        // the stack would have held: the group's plan is the
                        // stack's expression with the expert axis removed.
                        const WeightStore& slot = Lw.expert_cache->ensure_resident(
                            Lw.expert_group, static_cast<std::uint32_t>(e), stream);
                        gate_up_w = static_cast<const std::uint16_t*>(
                            slot.get("gate_up_proj").data());
                        down_w = static_cast<const std::uint16_t*>(
                            slot.get("down_proj").data());
                    } else {
                        gate_up_w = static_cast<const std::uint16_t*>(
                                        Lw.moe_gate_up_proj->data())
                                    + e * expert_stride_gu;
                        down_w = static_cast<const std::uint16_t*>(
                                     Lw.moe_down_proj->data())
                                 + e * expert_stride_dn;
                    }
                    ops::gemm_act_x_wt_bf16(cublas.handle(),
                        moe_ws.expert_in.data(), gate_up_w,
                        moe_ws.expert_gate_up.data(), Ne, 2 * Im, H);

                    kernels::launch_chunked_swiglu_bf16(
                        moe_ws.expert_gate_up.data(),
                        moe_ws.expert_act.data(),
                        Ne, Im, stream,
                        /*gate_second=*/moe_gate_up_swapped());

                    ops::gemm_act_x_wt_bf16(cublas.handle(),
                        moe_ws.expert_act.data(), down_w,
                        moe_ws.expert_out.data(), Ne, H, Im);

                    kernels::launch_scatter_add_weighted_bf16(
                        ws.norm_y.data(), moe_ws.expert_out.data(),
                        moe_ws.expert_idx.data(), moe_ws.expert_w.data(),
                        Ne, H, stream);

                    if (streamed) {
                        // Unpin per expert, not per layer: the pin then covers
                        // exactly the launches that read the slot. Holding a
                        // whole layer's routed set would make the slab's
                        // minimum size the number of experts a step happens to
                        // route to. Nothing races -- a later page-in that wants
                        // this slot syncs `stream` before overwriting it.
                        Lw.expert_cache->end_batch();
                    }
                }
            });
    }

    // ── Shared expert (Qwen3.5 / 3.6-MoE: always-on dense MLP + sigmoid
    //    gate). Qwen3-MoE has no shared expert — skip the whole block
    //    when the bind didn't wire `shared_*` pointers (Is == 0).
    if (Is > 0 && Lw.shared_gate_proj != nullptr) {
        profile_cuda_stage(profile, profile ? &profile->moe_shared_ms : nullptr,
            stream, [&] {
                const bool fused_shared_scalar_gate =
                    Lw.shared_gate_up_gate_proj != nullptr;
                // Folded: the routed batched GEMM already produced
                // `moe_ws.shared_out`, so only the scalar gate is left.
                if (moe_shared_folded) {
                } else if (fused_shared_scalar_gate) {
                    profile_cuda_detail_stage(
                        profile, profile ? &profile->moe_shared_gate_up_ms : nullptr,
                        stream, [&] {
                            ops::gemm_act_x_w(cublas.handle(),
                                ws.norm_x.data(),
                                ops::WeightView(*Lw.shared_gate_up_gate_proj),
                                moe_ws.shared_gate_up.data(), N, 2 * Is + 1, H);
                            kernels::launch_chunked_swiglu_strided_bf16(
                                moe_ws.shared_gate_up.data(),
                                moe_ws.shared_act.data(), N, Is, 2 * Is + 1, stream);
                        });
                } else if (Lw.shared_gate_up_proj != nullptr) {
                    profile_cuda_detail_stage(
                        profile, profile ? &profile->moe_shared_gate_up_ms : nullptr,
                        stream, [&] {
                            ops::gemm_act_x_w(cublas.handle(),
                                ws.norm_x.data(), ops::WeightView(*Lw.shared_gate_up_proj),
                                moe_ws.shared_gate_up.data(), N, 2 * Is, H);
                            kernels::launch_chunked_swiglu_bf16(
                                moe_ws.shared_gate_up.data(),
                                moe_ws.shared_act.data(), N, Is, stream);
                        });
                } else {
                    profile_cuda_detail_stage(
                        profile, profile ? &profile->moe_shared_gate_up_ms : nullptr,
                        stream, [&] {
                            ops::gemm_act_x_w(cublas.handle(),
                                ws.norm_x.data(),
                                make_weight_view(
                                    Lw.shared_gate_proj, Lw.shared_gate_proj_quant),
                                moe_ws.shared_gate.data(), N, Is, H);
                            ops::gemm_act_x_w(cublas.handle(),
                                ws.norm_x.data(),
                                make_weight_view(
                                    Lw.shared_up_proj, Lw.shared_up_proj_quant),
                                moe_ws.shared_up.data(), N, Is, H);
                            kernels::launch_swiglu_bf16(
                                moe_ws.shared_gate.data(), moe_ws.shared_up.data(),
                                moe_ws.shared_act.data(),
                                N * Is, stream);
                        });
                }
                if (!moe_shared_folded) {
                    profile_cuda_detail_stage(
                        profile, profile ? &profile->moe_shared_down_ms : nullptr,
                        stream, [&] {
                            ops::gemm_act_x_w(cublas.handle(),
                                moe_ws.shared_act.data(),
                                make_weight_view(
                                    Lw.shared_down_proj, Lw.shared_down_proj_quant),
                                moe_ws.shared_out.data(), N, H, Is);
                        });
                }

                // shared_gate logit [N, 1] = norm_x @ shared_gate.weight.T
                profile_cuda_detail_stage(
                    profile, profile ? &profile->moe_shared_gate_ms : nullptr,
                    stream, [&] {
                        if (fused_shared_scalar_gate) {
                            const auto* scalar_gate =
                                moe_ws.shared_gate_up.data() +
                                static_cast<std::size_t>(2 * Is);
                            kernels::launch_sigmoid_scalar_gate_strided_add_bf16(
                                moe_out, moe_ws.shared_out.data(),
                                scalar_gate,
                                N, H, 2 * Is + 1, stream);
                        } else if (Lw.shared_gate != nullptr &&
                                   N <= qwen35_moe_decode_fast_max_tokens() &&
                                   !Lw.shared_gate_quant.has_value()) {
                            kernels::launch_sigmoid_dot_scalar_gate_add_bf16(
                                ws.norm_x.data(),
                                Lw.shared_gate->data(),
                                moe_out,
                                moe_ws.shared_out.data(),
                                N, H, stream);
                        } else {
                            ops::gemm_act_x_w(cublas.handle(),
                                ws.norm_x.data(),
                                make_weight_view(Lw.shared_gate, Lw.shared_gate_quant),
                                moe_ws.shared_gate_logit.data(), N, 1, H);

                            // shared_out *= sigmoid(scalar_gate[n]) per token,
                            // broadcast across all H channels.
                            kernels::launch_sigmoid_scalar_gate_add_bf16(
                                moe_out, moe_ws.shared_out.data(),
                                moe_ws.shared_gate_logit.data(),
                                N, H, stream);
                        }
                    });
            });
    }

    if (T > 1) {
        profile_cuda_stage(profile, profile ? &profile->moe_allreduce_ms : nullptr,
            stream, [&] {
                tp->all_reduce_bf16(ws.norm_y.data(),
                    (std::size_t)N * H, ncclSum, stream);
            });
    }
    return add_to_residual;
}

}  // namespace

void qwen3_5_moe_forward_paged(
    const Qwen3_5MoeWeights& w,
    const HfConfig& cfg,
    const Qwen3_5ForwardCfg& fwd_cfg,
    Qwen3_5PlanState& plan_state,
    Workspace& ws,
    Qwen3_5LinearAttnWorkspace& la_ws,
    Qwen3_5MoeMlpWorkspace& moe_ws,
    KvCache& cache,
    RecurrentStateCache& state_cache,
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
    int total_tokens, int num_requests,
    bool is_pure_decode,
    const std::uint8_t* /*mask_d*/,
    const std::int32_t* /*mask_indptr_d*/,
    const std::uint32_t* w_page_d,
    const std::uint32_t* w_off_d,
    const std::uint8_t* row_valid_d,
    bool has_write_desc,
    const std::int32_t* slot_ids_h,
    const std::uint8_t* is_fresh_h,
    const std::int32_t* slot_ids_d,
    const std::uint8_t* is_fresh_d,
    const std::int32_t* logit_row_indices_d,
    int num_logit_rows,
    const std::int32_t* commit_advance_gather,
    const std::uint32_t* rs_buffer_slot_ids_h,
    const std::uint32_t* rs_buffer_slot_indptr_h,
    const std::int32_t* rs_fold_lens,
    const std::uint32_t* rs_fold_lens_h,
    bool rs_buffer_write,
    bool rs_buffer_fold,
    const std::uint32_t* rs_buffer_read_slot_ids_h,
    const std::uint32_t* rs_buffer_read_indptr_h,
    const std::uint32_t* rs_buffer_read_lens_h,
    const std::uint32_t* rs_buffer_heads_h)
{
    // Recurrent-only commit-advance (see qwen3_5_forward.cpp): re-run only the
    // linear-attn block over the accepted tokens (gathered from the verify
    // stash), advancing rs_cache state — no embed/reset/attention/MLP/lm_head.
    const bool commit_advance =
        commit_advance_gather != nullptr || rs_buffer_fold;
    if ((rs_buffer_write || rs_buffer_fold) &&
        (rs_buffer_slot_ids_h == nullptr ||
         rs_buffer_slot_indptr_h == nullptr ||
         slot_ids_h == nullptr ||
         slot_ids_d == nullptr)) {
        throw std::runtime_error(
            "buffered RS execution is missing its slot bindings");
    }
    if (rs_buffer_fold && rs_fold_lens == nullptr) {
        throw std::runtime_error(
            "buffered RS fold is missing per-request commit lengths");
    }

    // Pure-Qwen3-MoE (Qwen3-30B-A3B, model_type == "qwen3_moe") has no
    // linear-attn layers; the per-slot rs_cache is unused. Qwen3.5 /
    // 3.6-MoE additionally fires the linear-attn body — those layers
    // consume slot_ids_h / is_fresh_h to drive per-request state.
    const bool has_linear_attn_layers = std::any_of(
        w.layers.begin(), w.layers.end(),
        [](const Qwen3_5MoeLayerWeights& Lw) {
            return Lw.kind == Qwen3_5MoeLayerWeights::Kind::LinearAttn;
        });
    const int H  = cfg.hidden_size;
    const int V  = cfg.vocab_size;
    const int N  = total_tokens;
    const int R  = num_requests;
    const float eps = cfg.rms_norm_eps;
    cudaStream_t stream = cublas.stream();
    // A buffered write that ALSO folds: the boundary is a `commit_len` over
    // the extended layout rather than a separate replay pass, so it rides the
    // ordinary write path. `rs_fold_lens` is emitted for every row of every
    // buffered pass, so an all-zero one means "pure append" and must not turn
    // the pass into a fold.
    //
    // A MIXED fire is the same thing per row: one request folds while another
    // only appends. The rows share an initial state, an extended layout and
    // their outputs -- they differ ONLY in whether the recurrence persists.
    // So the pass-level `write_state` becomes "does ANY row persist", refined
    // by a per-row device mask. A row persists when it folds, or when it has
    // no buffer of its own at all (an empty write CSR span), which is the
    // plain in-forward advance riding along in a buffered fire.
    std::vector<std::uint8_t> rs_write_mask_h;
    const std::uint8_t* rs_write_state_mask_d = nullptr;
    bool write_folds = false;
    if (rs_buffer_write && rs_fold_lens_h != nullptr &&
        rs_buffer_slot_indptr_h != nullptr) {
        rs_write_mask_h.resize(static_cast<std::size_t>(R));
        bool all_persist = true;
        for (int r = 0; r < R; ++r) {
            const bool row_buffered =
                rs_buffer_slot_indptr_h[r + 1] > rs_buffer_slot_indptr_h[r];
            const bool persists = rs_fold_lens_h[r] != 0 || !row_buffered;
            rs_write_mask_h[r] = persists ? 1 : 0;
            write_folds = write_folds || persists;
            all_persist = all_persist && persists;
        }
        if (write_folds && !all_persist) {
            CUDA_CHECK(cudaMemcpyAsync(
                la_ws.rs_write_state_mask.data(), rs_write_mask_h.data(),
                rs_write_mask_h.size() * sizeof(std::uint8_t),
                cudaMemcpyHostToDevice, stream));
            rs_write_state_mask_d = la_ws.rs_write_state_mask.data();
        }
    }

    // The extended token layout for the buffer-read path, built once per fire
    // because every linear layer runs over the same rows.
    std::vector<std::uint32_t> qo_ext_h;
    const std::uint32_t* qo_ext_d = nullptr;
    int n_ext = 0;
    const bool has_buffer_read =
        rs_buffer_read_lens_h != nullptr &&
        rs_buffer_read_indptr_h != nullptr &&
        rs_buffer_read_slot_ids_h != nullptr &&
        std::any_of(rs_buffer_read_lens_h, rs_buffer_read_lens_h + R,
                    [](std::uint32_t len) { return len != 0; });
    if (has_buffer_read) {
        if (rs_buffer_fold) {
            throw std::runtime_error(
                "an RS fold replays the buffer itself; it cannot also carry a "
                "buffer read");
        }
        if (qo_indptr_h == nullptr) {
            throw std::runtime_error(
                "RS buffer read needs the host-side qo_indptr to place each "
                "request's replayed prefix");
        }
        qo_ext_h.resize(static_cast<std::size_t>(R) + 1);
        qo_ext_h[0] = 0;
        for (int r = 0; r < R; ++r) {
            const std::uint32_t rows =
                (qo_indptr_h[r + 1] - qo_indptr_h[r]) + rs_buffer_read_lens_h[r];
            qo_ext_h[r + 1] = qo_ext_h[r] + rows;
        }
        n_ext = static_cast<int>(qo_ext_h[R]);
        if (la_ws.max_tokens > 0 && n_ext > la_ws.max_tokens) {
            throw std::runtime_error(
                "RS buffer replay plus new tokens (" + std::to_string(n_ext) +
                ") exceeds the linear-attention workspace capacity (" +
                std::to_string(la_ws.max_tokens) + ")");
        }
        CUDA_CHECK(cudaMemcpyAsync(
            la_ws.qo_ext.data(), qo_ext_h.data(),
            qo_ext_h.size() * sizeof(std::uint32_t),
            cudaMemcpyHostToDevice, stream));
        qo_ext_d = la_ws.qo_ext.data();
    }
    if (has_write_desc) {
        const bool has_full_attention = std::any_of(
            w.layers.begin(), w.layers.end(), [](const auto& layer) {
                return layer.kind ==
                    Qwen3_5MoeLayerWeights::Kind::FullAttn;
            });
        if (w_page_d == nullptr || w_off_d == nullptr ||
            !cache.format().is_native_bf16() ||
            !has_full_attention) {
            throw std::runtime_error(
                "Qwen3.5-MoE explicit KV writes are unsupported by this layout");
        }
    }
    Qwen35MoeForwardProfile profile;
    const int tp_rank = (fwd_cfg.tp_comm != nullptr) ? fwd_cfg.tp_comm->rank() : 0;
    profile.begin(N, R, is_pure_decode, tp_rank, stream);

    if (has_linear_attn_layers &&
        !(commit_advance && !rs_buffer_fold) &&
        !rs_buffer_write) {
        if (slot_ids_h != nullptr && is_fresh_h != nullptr) {
            if (std::any_of(is_fresh_h, is_fresh_h + R, [](auto fresh) {
                    return fresh != 0;
                })) {
                if (slot_ids_d != nullptr && is_fresh_d != nullptr) {
                    state_cache.reset_slots_if_fresh(
                        slot_ids_d, is_fresh_d, R, stream);
                } else {
                    for (int r = 0; r < R; ++r) {
                        if (is_fresh_h[r]) {
                            state_cache.reset_slot(slot_ids_h[r], stream);
                        }
                    }
                }
            }
        } else if (!is_pure_decode) {
            state_cache.reset(stream);
        }
    }

    // Decode plan was refreshed by `prepare_qwen3_5_decode_plan` before
    // this body call. Avoids host work inside any cudaStream capture.
    const ops::DecodePlanCache* decode_plan =
        plan_state.decode_plan ? plan_state.decode_plan.get() : nullptr;
    const ops::PrefillPlanCache* prefill_plan =
        (plan_state.use_prefill_plan && plan_state.prefill_plan)
            ? plan_state.prefill_plan.get()
            : nullptr;

    if (!commit_advance) {
        profile_cuda_stage(&profile, &profile.embed_ms, stream, [&] {
            kernels::launch_embed_bf16(
                token_ids, w.embed->data(), ws.y.data(),
                N, H, cfg.vocab_size, stream);
        });
    }

    int linear_idx = 0;  // compact linear-layer index (stash/gather key)
    for (std::size_t L = 0; L < w.layers.size(); ++L) {
        const auto& Lw = w.layers[L];
        const bool is_linear =
            Lw.kind == Qwen3_5MoeLayerWeights::Kind::LinearAttn;

        if (commit_advance) {
            if (!is_linear) continue;
            if (!rs_buffer_fold) {
                const std::uint16_t* stash =
                    static_cast<const std::uint16_t*>(
                        state_cache.verify_hidden_stash_layer(linear_idx));
                kernels::launch_gather_bf16_rows(
                    stash, commit_advance_gather,
                    static_cast<std::uint16_t*>(ws.norm_x.data()),
                    N, H, stream);
            }
            linear_attn_body(
                Lw, cfg, fwd_cfg, ws, la_ws, state_cache,
                static_cast<int>(L), linear_idx, N, R, is_pure_decode,
                slot_ids_h, slot_ids_d, qo_indptr_h, qo_indptr,
                cublas, stream, &profile,
                rs_buffer_fold ? rs_fold_lens : nullptr,
                rs_buffer_slot_ids_h, rs_buffer_slot_indptr_h,
                /*rs_buffer_write=*/false, /*rs_fold_after_write=*/false,
                rs_buffer_fold,
                // A fold IS the replay; it never carries a read as well. It
                // still needs the head: its own gather is physical.
                nullptr, nullptr, nullptr, rs_buffer_heads_h,
                nullptr, nullptr, 0);
            ++linear_idx;
            continue;
        }

        profile_cuda_stage(&profile, &profile.norm_ms, stream, [&] {
            rmsnorm_bf16_dispatch(cfg,
                ws.y.data(), Lw.attn_norm_pre->data(), ws.norm_x.data(),
                N, H, eps, stream);
        });

        if (is_linear) {
            ++profile.linear_layers;
            const int li = linear_idx;
            if (state_cache.verify_frozen() &&
                state_cache.verify_hidden_stash_enabled()) {
                void* stash =
                    state_cache.verify_hidden_stash_layer(linear_idx);
                if (stash != nullptr) {
                    CUDA_CHECK(cudaMemcpyAsync(
                        stash, ws.norm_x.data(),
                        static_cast<std::size_t>(N) * H *
                            sizeof(std::uint16_t),
                        cudaMemcpyDeviceToDevice, stream));
                }
            }
            ++linear_idx;
            profile_cuda_stage(&profile, &profile.linear_attn_ms, stream, [&] {
                linear_attn_body(
                    Lw, cfg, fwd_cfg, ws, la_ws, state_cache,
                    static_cast<int>(L), li, N, R, is_pure_decode,
                    slot_ids_h, slot_ids_d, qo_indptr_h, qo_indptr,
                    cublas, stream, &profile, /*commit_len=*/nullptr,
                    rs_buffer_slot_ids_h, rs_buffer_slot_indptr_h,
                    rs_buffer_write, /*rs_fold_after_write=*/write_folds,
                    /*rs_buffer_fold=*/false,
                    has_buffer_read ? rs_buffer_read_slot_ids_h : nullptr,
                    has_buffer_read ? rs_buffer_read_indptr_h : nullptr,
                    has_buffer_read ? rs_buffer_read_lens_h : nullptr,
                    rs_buffer_heads_h,
                    has_buffer_read ? qo_ext_h.data() : nullptr,
                    qo_ext_d, n_ext, rs_write_state_mask_d);
            });
        } else {
            ++profile.full_layers;
            profile_cuda_stage(&profile, &profile.full_attn_ms, stream, [&] {
                full_attn_body(
                    Lw, cfg, fwd_cfg, ws, la_ws, cache, attn_ws,
                    decode_plan, prefill_plan, static_cast<int>(L),
                    Lw.kv_layer,
                    N, num_requests,
                    positions, qo_indptr, kv_page_indices, kv_page_indptr,
                    kv_last_page_lens, qo_indptr_h, kv_page_indptr_h,
                    w_page_d, w_off_d, row_valid_d, has_write_desc,
                    cublas, stream);
            });
        }
        // (Post-attention residual fused into the body's final GEMM
        //  via beta=1 on TP=1; on TP>1 the body did the all-reduce and
        //  residual_add itself. ws.y holds the post-attention state.)

        // Post-attention norm + MoE block + residual.
        profile_cuda_stage(&profile, &profile.norm_ms, stream, [&] {
            rmsnorm_bf16_dispatch(cfg,
                ws.y.data(), Lw.mlp_norm_pre->data(), ws.norm_x.data(),
                N, H, eps, stream);
        });
        ++profile.moe_layers;
        const bool moe_added_to_residual = moe_block(
            Lw, cfg, fwd_cfg, ws, moe_ws, N, is_pure_decode,
            cublas, stream, &profile);
        if (!moe_added_to_residual) {
            profile_cuda_stage(&profile, &profile.residual_ms, stream, [&] {
                kernels::launch_residual_add_bf16(
                    ws.y.data(), ws.norm_y.data(),
                    (std::size_t)N * H, stream);
            });
        }
    }

    // State-only forward (num_logit_rows < 0) or recurrent-only commit-advance:
    // logits/hidden are discarded, so skip final_norm + lm_head + final copy.
    // See qwen3_5_forward.
    if (num_logit_rows < 0 || commit_advance) {
        profile.end(stream);
        maybe_print_profile(profile);
        return;
    }

    profile_cuda_stage(&profile, &profile.lm_head_ms, stream, [&] {
        const bool compact_logits =
            logit_row_indices_d != nullptr && num_logit_rows > 0 &&
            num_logit_rows < N;
        int lm_head_rows = N;
        const void* lm_head_input = ws.norm_x.data();
        if (compact_logits) {
            kernels::launch_gather_bf16_rows(
                static_cast<const std::uint16_t*>(ws.y.data()),
                logit_row_indices_d,
                static_cast<std::uint16_t*>(ws.norm_y.data()),
                num_logit_rows, H, stream);
            rmsnorm_bf16_dispatch(cfg,
                ws.norm_y.data(), w.final_norm->data(), ws.norm_x.data(),
                num_logit_rows, H, eps, stream);
            lm_head_rows = num_logit_rows;
        } else {
            rmsnorm_bf16_dispatch(cfg,
                ws.y.data(), w.final_norm->data(), ws.norm_x.data(),
                N, H, eps, stream);
        }
        ops::gemm_act_x_wt_bf16(cublas.handle(),
            lm_head_input, w.lm_head->data(),
            ws.logits.data(), lm_head_rows, V, H);
        if (!compact_logits) {
            CUDA_CHECK(cudaMemcpyAsync(
                ws.y.data(), ws.norm_x.data(),
                static_cast<std::size_t>(N) * H * sizeof(std::uint16_t),
                cudaMemcpyDeviceToDevice, stream));
        }
    });
    profile.end(stream);
    maybe_print_profile(profile);
}

namespace {

void mtp_full_attn_no_cache_moe(
    const Qwen3_5MoeLayerWeights& Lw,
    const HfConfig& cfg,
    const Qwen3_5ForwardCfg& fwd_cfg,
    Workspace& ws,
    Qwen3_5LinearAttnWorkspace& la,
    KvCache& cache,
    int N,
    int draft_step,
    const std::int32_t* position_ids,
    const std::int32_t* request_ids,
    const std::uint32_t* kv_page_indices,
    const std::uint32_t* kv_page_indptr,
    const std::uint32_t* kv_last_page_lens,
    int max_global_tokens,
    ops::CublasHandle& cublas,
    cudaStream_t stream)
{
    const int T = std::max(1, fwd_cfg.tp_size);
    const int H = cfg.hidden_size;
    const int q_heads = cfg.num_attention_heads / T;
    const int kv_heads = cfg.num_key_value_heads / T;
    const int Hq = q_heads * cfg.head_dim;
    const int Hk = kv_heads * cfg.head_dim;
    const int d = cfg.head_dim;
    const int rotary_dim = std::max<int>(2,
        2 * static_cast<int>(0.5f * cfg.partial_rotary_factor * d));
    const float eps = cfg.rms_norm_eps;
    NcclComm* tp = (T > 1) ? fwd_cfg.tp_comm : nullptr;
    const std::size_t kv_step_offset =
        static_cast<std::size_t>(draft_step) * N * Hk;
    auto* k_step = static_cast<std::uint16_t*>(ws.k.data()) + kv_step_offset;
    auto* v_step = static_cast<std::uint16_t*>(ws.v.data()) + kv_step_offset;

    if (cfg.attn_output_gate) {
        ops::gemm_act_x_w(cublas.handle(),
            ws.norm_x.data(), make_weight_view(Lw.fa_q_proj, Lw.fa_q_proj_quant),
            la.fa_qg_packed.data(), N, 2 * Hq, H);
        kernels::launch_split_q_gate_bf16(
            la.fa_qg_packed.data(), ws.q.data(), la.fa_gate.data(),
            N, q_heads, cfg.head_dim, stream);
    } else {
        ops::gemm_act_x_w(cublas.handle(),
            ws.norm_x.data(), make_weight_view(Lw.fa_q_proj, Lw.fa_q_proj_quant),
            ws.q.data(), N, Hq, H);
    }

    ops::gemm_act_x_w(cublas.handle(),
        ws.norm_x.data(), make_weight_view(Lw.fa_k_proj, Lw.fa_k_proj_quant),
        k_step, N, Hk, H);
    ops::gemm_act_x_w(cublas.handle(),
        ws.norm_x.data(), make_weight_view(Lw.fa_v_proj, Lw.fa_v_proj_quant),
        v_step, N, Hk, H);
    rmsnorm_bf16_dispatch(cfg,
        ws.q.data(), Lw.fa_q_norm->data(), ws.q.data(),
        N * q_heads, d, eps, stream);
    rmsnorm_bf16_dispatch(cfg,
        k_step, Lw.fa_k_norm->data(), k_step,
        N * kv_heads, d, eps, stream);
    kernels::launch_rope_partial_bf16(
        ws.q.data(), k_step, position_ids,
        N, q_heads, kv_heads, d, rotary_dim, cfg.rope_theta, stream);
    const auto mtp_kv = cache.layer_view(Lw.kv_layer);
    ops::launch_attention_mtp_paged_history_bf16(
        ws.q.data(), mtp_kv.k_bf16_pages, mtp_kv.v_bf16_pages,
        ws.k.data(), ws.v.data(), ws.attn_out.data(),
        position_ids, request_ids,
        kv_page_indices, kv_page_indptr, kv_last_page_lens,
        N, draft_step + 1, N, max_global_tokens, cache.page_size(),
        q_heads, kv_heads, d, mtp_kv.hnd_layout,
        fwd_cfg.mtp_global_cache_uses_prefix_position, stream);
    if (cfg.attn_output_gate) {
        kernels::launch_sigmoid_gate_inplace_bf16(
            ws.attn_out.data(), la.fa_gate.data(), N * Hq, stream);
    }

    if (T == 1) {
        ops::gemm_act_x_w(cublas.handle(),
            ws.attn_out.data(), make_weight_view(Lw.fa_o_proj, Lw.fa_o_proj_quant),
            ws.y.data(), N, H, Hq, /*beta=*/1.f);
    } else {
        ops::gemm_act_x_w(cublas.handle(),
            ws.attn_out.data(), make_weight_view(Lw.fa_o_proj, Lw.fa_o_proj_quant),
            ws.norm_y.data(), N, H, Hq, /*beta=*/0.f);
        tp->all_reduce_bf16(ws.norm_y.data(),
            static_cast<std::size_t>(N) * H, ncclSum, stream);
        kernels::launch_residual_add_bf16(
            ws.y.data(), ws.norm_y.data(),
            static_cast<std::size_t>(N) * H, stream);
    }
}

bool mtp_shared_expert_only_moe(
    const Qwen3_5MoeLayerWeights& Lw,
    const HfConfig& cfg,
    const Qwen3_5ForwardCfg& fwd_cfg,
    Workspace& ws,
    Qwen3_5MoeMlpWorkspace& moe_ws,
    int N,
    ops::CublasHandle& cublas,
    cudaStream_t stream)
{
    const int T = std::max(1, fwd_cfg.tp_size);
    const int H = cfg.hidden_size;
    const int Is = cfg.shared_expert_intermediate_size / T;
    NcclComm* tp = (T > 1) ? fwd_cfg.tp_comm : nullptr;
    if (Is <= 0 || Lw.shared_gate_proj == nullptr) return false;

    const bool fused_shared_scalar_gate =
        Lw.shared_gate_up_gate_proj != nullptr;
    if (fused_shared_scalar_gate) {
        ops::gemm_act_x_w(cublas.handle(),
            ws.norm_x.data(), ops::WeightView(*Lw.shared_gate_up_gate_proj),
            moe_ws.shared_gate_up.data(), N, 2 * Is + 1, H);
        kernels::launch_chunked_swiglu_strided_bf16(
            moe_ws.shared_gate_up.data(), moe_ws.shared_act.data(),
            N, Is, 2 * Is + 1, stream);
    } else if (Lw.shared_gate_up_proj != nullptr) {
        ops::gemm_act_x_w(cublas.handle(),
            ws.norm_x.data(), ops::WeightView(*Lw.shared_gate_up_proj),
            moe_ws.shared_gate_up.data(), N, 2 * Is, H);
        kernels::launch_chunked_swiglu_bf16(
            moe_ws.shared_gate_up.data(), moe_ws.shared_act.data(),
            N, Is, stream);
    } else {
        ops::gemm_act_x_w(cublas.handle(),
            ws.norm_x.data(),
            make_weight_view(Lw.shared_gate_proj, Lw.shared_gate_proj_quant),
            moe_ws.shared_gate.data(), N, Is, H);
        ops::gemm_act_x_w(cublas.handle(),
            ws.norm_x.data(),
            make_weight_view(Lw.shared_up_proj, Lw.shared_up_proj_quant),
            moe_ws.shared_up.data(), N, Is, H);
        kernels::launch_swiglu_bf16(
            moe_ws.shared_gate.data(), moe_ws.shared_up.data(),
            moe_ws.shared_act.data(), N * Is, stream);
    }
    ops::gemm_act_x_w(cublas.handle(),
        moe_ws.shared_act.data(),
        make_weight_view(Lw.shared_down_proj, Lw.shared_down_proj_quant),
        moe_ws.shared_out.data(), N, H, Is);
    if (fused_shared_scalar_gate) {
        const auto* scalar_gate =
            moe_ws.shared_gate_up.data() + static_cast<std::size_t>(2 * Is);
        kernels::launch_sigmoid_scalar_gate_strided_inplace_bf16(
            moe_ws.shared_out.data(), scalar_gate, N, H, 2 * Is + 1, stream);
    } else {
        ops::gemm_act_x_w(cublas.handle(),
            ws.norm_x.data(),
            make_weight_view(Lw.shared_gate, Lw.shared_gate_quant),
            moe_ws.shared_gate_logit.data(), N, 1, H);
        kernels::launch_sigmoid_scalar_gate_inplace_bf16(
            moe_ws.shared_out.data(), moe_ws.shared_gate_logit.data(),
            N, H, stream);
    }

    CUDA_CHECK(cudaMemcpyAsync(
        ws.norm_y.data(), moe_ws.shared_out.data(),
        static_cast<std::size_t>(N) * H * sizeof(std::uint16_t),
        cudaMemcpyDeviceToDevice, stream));
    if (T > 1) {
        tp->all_reduce_bf16(
            ws.norm_y.data(), static_cast<std::size_t>(N) * H,
            ncclSum, stream);
    }
    return true;
}

}  // namespace

void qwen3_5_moe_mtp_process_cache(
    const Qwen3_5MoeWeights& w,
    const HfConfig& cfg,
    const Qwen3_5ForwardCfg& fwd_cfg,
    Workspace& ws,
    Qwen3_5LinearAttnWorkspace& la_ws,
    KvCache& cache,
    RecurrentStateCache& state_cache,
    ops::CublasHandle& cublas,
    const std::int32_t* token_ids,
    const std::int32_t* positions,
    const std::uint32_t* qo_indptr,
    const std::uint32_t* kv_page_indices,
    const std::uint32_t* kv_page_indptr,
    const std::uint32_t* kv_last_page_lens,
    const std::int32_t* slot_ids_d,
    const std::int32_t* source_row_indices,
    int total_tokens,
    int num_requests)
{
    if (!w.mtp || total_tokens <= 0 || num_requests <= 0) return;
    const auto& mtp = *w.mtp;
    const auto& Lw = mtp.layer;
    if (Lw.kv_layer < 0) return;

    const int H = cfg.hidden_size;
    const int T = std::max(1, fwd_cfg.tp_size);
    const int kv_heads = cfg.num_key_value_heads / T;
    const int Hk = kv_heads * cfg.head_dim;
    const int d = cfg.head_dim;
    const int rotary_dim = std::max<int>(2,
        2 * static_cast<int>(0.5f * cfg.partial_rotary_factor * d));
    const float eps = cfg.rms_norm_eps;
    cudaStream_t stream = cublas.stream();

    void* pending = state_cache.mtp_pending_hidden(0);
    const void* target_hidden = ws.y.data();
    if (source_row_indices != nullptr) {
        kernels::launch_gather_bf16_rows(
            static_cast<const std::uint16_t*>(ws.y.data()),
            source_row_indices,
            static_cast<std::uint16_t*>(ws.norm_x.data()),
            total_tokens, H, stream);
        target_hidden = ws.norm_x.data();
    }
    ops::launch_mtp_shift_hidden_bf16(
        target_hidden, pending, qo_indptr, slot_ids_d, ws.norm_y.data(),
        total_tokens, num_requests, H, stream);
    ops::launch_mtp_update_pending_hidden_bf16(
        target_hidden, pending, qo_indptr, slot_ids_d, num_requests, H, stream);
    kernels::launch_embed_bf16(
        token_ids, mtp.embed->data(), ws.norm_x.data(),
        total_tokens, H, cfg.vocab_size, stream);
    rmsnorm_bf16_dispatch(cfg,
        ws.norm_x.data(), mtp.pre_fc_norm_embedding->data(), ws.q.data(),
        total_tokens, H, eps, stream);
    rmsnorm_bf16_dispatch(cfg,
        ws.norm_y.data(), mtp.pre_fc_norm_hidden->data(), ws.attn_out.data(),
        total_tokens, H, eps, stream);
    kernels::launch_concat_bf16_rows(
        ws.q.data(), ws.attn_out.data(), ws.mtp_concat.data(),
        total_tokens, H, H, stream);
    ops::gemm_act_x_w(cublas.handle(),
        ws.mtp_concat.data(), *mtp.fc, ws.norm_y.data(),
        total_tokens, H, 2 * H);
    rmsnorm_bf16_dispatch(cfg,
        ws.norm_y.data(), Lw.attn_norm_pre->data(), ws.norm_x.data(),
        total_tokens, H, eps, stream);
    ops::gemm_act_x_w(cublas.handle(),
        ws.norm_x.data(), make_weight_view(Lw.fa_k_proj, Lw.fa_k_proj_quant),
        ws.k.data(), total_tokens, Hk, H);
    ops::gemm_act_x_w(cublas.handle(),
        ws.norm_x.data(), make_weight_view(Lw.fa_v_proj, Lw.fa_v_proj_quant),
        ws.v.data(), total_tokens, Hk, H);
    rmsnorm_bf16_dispatch(cfg,
        ws.k.data(), Lw.fa_k_norm->data(), ws.k.data(),
        total_tokens * kv_heads, d, eps, stream);
    kernels::launch_rope_partial_bf16(
        /*q=*/nullptr, ws.k.data(), positions,
        total_tokens, 0, kv_heads, d, rotary_dim, cfg.rope_theta, stream);
    kernels::launch_write_kv_to_pages(
        cache.layer_view(Lw.kv_layer),
        ws.k.data(), ws.v.data(), qo_indptr, kv_page_indices,
        kv_page_indptr, kv_last_page_lens, total_tokens, num_requests,
        stream);
}

void qwen3_5_moe_mtp_forward(
    const Qwen3_5MoeWeights& w,
    const HfConfig& cfg,
    const Qwen3_5ForwardCfg& fwd_cfg,
    Workspace& ws,
    Qwen3_5LinearAttnWorkspace& la_ws,
    Qwen3_5MoeMlpWorkspace& moe_ws,
    KvCache& cache,
    ops::CublasHandle& cublas,
    const std::int32_t* token_ids,
    const std::int32_t* position_ids,
    const std::int32_t* base_hidden_row_indices,
    const std::int32_t* request_ids,
    const std::uint32_t* kv_page_indices,
    const std::uint32_t* kv_page_indptr,
    const std::uint32_t* kv_last_page_lens,
    std::int32_t* sampled_token_ids,
    int num_tokens,
    int draft_step,
    int max_global_tokens)
{
    (void)sampled_token_ids;
    if (!w.mtp || num_tokens <= 0) return;
    const auto& mtp = *w.mtp;
    const auto& Lw = mtp.layer;
    const int H = cfg.hidden_size;
    const int V = cfg.vocab_size;
    const float eps = cfg.rms_norm_eps;
    cudaStream_t stream = cublas.stream();
    MtpProfile profile;
    profile.begin(num_tokens, stream);

    profile_mtp_stage(profile, profile.input_fc_ms, stream, [&] {
    kernels::launch_gather_bf16_rows(
        static_cast<const std::uint16_t*>(ws.y.data()),
        base_hidden_row_indices,
        static_cast<std::uint16_t*>(ws.norm_y.data()),
        num_tokens, H, stream);
    kernels::launch_embed_bf16(
        token_ids, mtp.embed->data(), ws.norm_x.data(),
        num_tokens, H, cfg.vocab_size, stream);
    rmsnorm_bf16_dispatch(cfg,
        ws.norm_x.data(), mtp.pre_fc_norm_embedding->data(), ws.q.data(),
        num_tokens, H, eps, stream);
    rmsnorm_bf16_dispatch(cfg,
        ws.norm_y.data(), mtp.pre_fc_norm_hidden->data(), ws.y.data(),
        num_tokens, H, eps, stream);
    kernels::launch_concat_bf16_rows(
        ws.q.data(), ws.y.data(), ws.mtp_concat.data(),
        num_tokens, H, H, stream);
    ops::gemm_act_x_w(cublas.handle(),
        ws.mtp_concat.data(), *mtp.fc, ws.y.data(),
        num_tokens, H, 2 * H);
    });

    profile_mtp_stage(profile, profile.attn_ms, stream, [&] {
    rmsnorm_bf16_dispatch(cfg,
        ws.y.data(), Lw.attn_norm_pre->data(), ws.norm_x.data(),
        num_tokens, H, eps, stream);
    mtp_full_attn_no_cache_moe(
        Lw, cfg, fwd_cfg, ws, la_ws, cache, num_tokens, draft_step,
        position_ids, request_ids, kv_page_indices, kv_page_indptr,
        kv_last_page_lens, max_global_tokens, cublas, stream);
    });

    profile_mtp_stage(profile, profile.moe_ms, stream, [&] {
    rmsnorm_bf16_dispatch(cfg,
        ws.y.data(), Lw.mlp_norm_pre->data(), ws.norm_x.data(),
        num_tokens, H, eps, stream);
    const MtpMoeMode mode = mtp_moe_mode();
    bool add_moe_residual = false;
    if (mode == MtpMoeMode::Full) {
        const bool moe_added_to_residual = moe_block(
            Lw, cfg, fwd_cfg, ws, moe_ws, num_tokens,
            /*is_pure_decode=*/true, cublas, stream, /*profile=*/nullptr);
        add_moe_residual = !moe_added_to_residual;
    } else if (mode == MtpMoeMode::SharedOnly) {
        add_moe_residual = mtp_shared_expert_only_moe(
            Lw, cfg, fwd_cfg, ws, moe_ws, num_tokens, cublas, stream);
    }
    if (add_moe_residual) {
        kernels::launch_residual_add_bf16(
            ws.y.data(), ws.norm_y.data(),
            static_cast<std::size_t>(num_tokens) * H, stream);
    }
    });

    profile_mtp_stage(profile, profile.lm_head_ms, stream, [&] {
    rmsnorm_bf16_dispatch(cfg,
        ws.y.data(), mtp.norm->data(), ws.norm_x.data(),
        num_tokens, H, eps, stream);
    ops::gemm_act_x_wt_bf16(cublas.handle(),
        ws.norm_x.data(), w.lm_head->data(),
        ws.logits.data(), num_tokens, V, H);
    CUDA_CHECK(cudaMemcpyAsync(
        ws.y.data(), ws.norm_x.data(),
        static_cast<std::size_t>(num_tokens) * H * sizeof(std::uint16_t),
        cudaMemcpyDeviceToDevice, stream));
    });
    profile.end(stream);
    maybe_print_mtp_profile(profile);
}

std::size_t qwen3_5_moe_workspace_bytes(const HfConfig& cfg,
                                        int N, int tp_size) {
    if (cfg.num_experts <= 0 || cfg.num_experts_per_tok <= 0 ||
        cfg.moe_intermediate_size <= 0) {
        return 0;
    }
    const int T = std::max(1, tp_size);
    const std::size_t n = static_cast<std::size_t>(N);
    const std::size_t maxR = n * cfg.num_experts_per_tok;
    const std::size_t H = static_cast<std::size_t>(cfg.hidden_size);
    const std::size_t I =
        static_cast<std::size_t>(cfg.moe_intermediate_size / T);
    const std::size_t Ish =
        static_cast<std::size_t>(
            std::max(0, cfg.shared_expert_intermediate_size / T));
    auto u16 = [](std::size_t elems) { return elems * 2; };
    auto i32 = [](std::size_t elems) { return elems * 4; };
    auto fp32 = [](std::size_t elems) { return elems * 4; };
    auto aligned_decode_block = [] {
        const char* v = std::getenv("PIE_QWEN35_MOE_ALIGNED_DECODE_BLOCK");
        if (v == nullptr || v[0] == '\0') return 16;
        char* end = nullptr;
        long parsed_long = std::strtol(v, &end, 10);
        if (end == v) return 16;
        int parsed = static_cast<int>(parsed_long);
        if (parsed <= 1) return 0;
        if (parsed < 4) parsed = 4;
        if (parsed > 64) parsed = 64;
        return parsed;
    };
    std::size_t bytes = 0;
    bytes += u16(n * cfg.num_experts);
    bytes += i32(n * cfg.num_experts_per_tok);
    bytes += fp32(n * cfg.num_experts_per_tok);
    bytes += u16(maxR * H);
    bytes += u16(maxR * 2 * I);
    bytes += u16(maxR * I);
    bytes += u16(maxR * H);
    bytes += i32(maxR);
    bytes += fp32(maxR);
    bytes += u16(n * Ish);
    bytes += u16(n * Ish);
    bytes += u16(n * Ish);
    bytes += u16(n * H);
    bytes += u16(n);
    bytes += u16(n * H);
    bytes += maxR * (6 * sizeof(void*) + sizeof(float));
    const int aligned_block = aligned_decode_block();
    if (aligned_block > 1 && maxR > 0) {
        const std::size_t block = static_cast<std::size_t>(aligned_block);
        const std::size_t active_expert_cap =
            std::min<std::size_t>(static_cast<std::size_t>(cfg.num_experts), maxR);
        const std::size_t max_blocks =
            (maxR + active_expert_cap * (block - 1) + block - 1) / block;
        const std::size_t aligned_rows = max_blocks * block;
        bytes += i32(aligned_rows);
        bytes += i32(max_blocks);
        bytes += u16(aligned_rows * H);
        bytes += u16(aligned_rows * 2 * I);
        bytes += u16(aligned_rows * I);
        bytes += u16(aligned_rows * H);
    }
    return bytes;
}

}  // namespace pie_cuda_driver::model
