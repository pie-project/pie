#include "memory_planner.hpp"
#include "../batch/planner_calibration.hpp"
#include "planner_profile_cache.hpp"
#include "recurrent_state_cache.hpp"
#include "kv_cache.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include <cuda_runtime.h>

#include "../batch/workspace.hpp"
#include "../config.hpp"
#include "../batch/persistent_inputs.hpp"
#include "kv_cache.hpp"
#include "../model/config.hpp"
#include "../model/gemma4/gemma4.hpp"
#include "../model/deepseek_v4/deepseek_v4_forward.hpp"
#include "dsv4_compress_cache.hpp"
#include "../model/glm5/glm5_forward.hpp"
#include "../model/kimi_k3/kimi_k3_forward.hpp"
#include "../model/kimi/kimi_forward.hpp"
#include "../model/loaded_model.hpp"
#include "../model/nemotron_h/nemotron_h_forward.hpp"
#include "../model/qwen3_5/qwen3_5_forward.hpp"
#include "../model/qwen3_5/qwen3_5_moe_forward.hpp"
#include "../model/workspace.hpp"
#include "../ops/gemm.hpp"

namespace pie_cuda_driver {
namespace {

// Every candidate layout must hold at least this much KV, independent of the
// request cap: below it a boot would admit so few sequences that admission and
// eviction cannot recover. Named because the "no viable layout" diagnostic
// reports it back to the operator.
constexpr std::size_t kMinKvTokensFloor = 32768;

std::size_t align_up(std::size_t n, std::size_t a) {
    return (n + a - 1) / a * a;
}

int clamp_pow2_nearest(int value, int lo, int hi) {
    value = std::max(lo, std::min(value, hi));
    int p = 1;
    while (p < value && p <= hi / 2) p <<= 1;
    const int lower = std::max(lo, p >> 1);
    const int upper = std::min(hi, p);
    if (upper <= lower) return lower;
    return (value - lower <= upper - value) ? lower : upper;
}


void min_into(CudaMemoryPlan& dst, const CudaMemoryPlan& src) {
    if (src.kv_page_size < dst.kv_page_size) {
        dst.kv_page_size = src.kv_page_size;
        dst.kv_page_bytes = src.kv_page_bytes;
    } else if (src.kv_page_size == dst.kv_page_size) {
        dst.kv_page_bytes = std::max(
            dst.kv_page_bytes, src.kv_page_bytes);
    }
    dst.max_workspace_tokens = std::min(dst.max_workspace_tokens,
                                        src.max_workspace_tokens);
    dst.max_requests = std::min(dst.max_requests, src.max_requests);
    dst.max_page_refs = std::min(dst.max_page_refs, src.max_page_refs);
    dst.attn_float_workspace_bytes = std::max(
        dst.attn_float_workspace_bytes, src.attn_float_workspace_bytes);
    dst.runtime_quant_scratch_bytes = std::max(
        dst.runtime_quant_scratch_bytes, src.runtime_quant_scratch_bytes);
    dst.persistent_input_bytes = std::max(
        dst.persistent_input_bytes, src.persistent_input_bytes);
    dst.capacity.max_forward_tokens = std::min(
        dst.capacity.max_forward_tokens, src.capacity.max_forward_tokens);
    dst.capacity.max_forward_requests = std::min(
        dst.capacity.max_forward_requests, src.capacity.max_forward_requests);
    dst.capacity.max_page_refs = std::min(
        dst.capacity.max_page_refs, src.capacity.max_page_refs);
    dst.capacity.max_logit_rows = std::min(
        dst.capacity.max_logit_rows, src.capacity.max_logit_rows);
    dst.capacity.max_prob_rows = std::min(
        dst.capacity.max_prob_rows, src.capacity.max_prob_rows);
    dst.capacity.max_custom_mask_bytes = std::min(
        dst.capacity.max_custom_mask_bytes, src.capacity.max_custom_mask_bytes);
    dst.capacity.max_sampler_rows = std::min(
        dst.capacity.max_sampler_rows, src.capacity.max_sampler_rows);
    dst.capacity.max_logprob_labels = std::min(
        dst.capacity.max_logprob_labels, src.capacity.max_logprob_labels);
}

CudaMemoryPlan tp_min_plan(const pie_cuda_driver::Config& cfg,
                           const CudaMemoryPlan& local) {
    if (cfg.distributed.tp_size <= 1) return local;
    const std::string& key = cfg.distributed.nccl_unique_id_hex;
    if (key.empty()) return local;

    struct State {
        std::mutex mu;
        std::condition_variable cv;
        int arrived = 0;
        bool ready = false;
        CudaMemoryPlan min_plan;
    };
    static std::mutex registry_mu;
    static std::unordered_map<std::string, std::shared_ptr<State>> registry;

    std::shared_ptr<State> st;
    {
        std::lock_guard<std::mutex> lk(registry_mu);
        auto& entry = registry[key];
        if (!entry) entry = std::make_shared<State>();
        st = entry;
    }

    std::unique_lock<std::mutex> lk(st->mu);
    if (st->arrived == 0) {
        st->min_plan = local;
    } else {
        min_into(st->min_plan, local);
    }
    ++st->arrived;
    if (st->arrived >= cfg.distributed.tp_size) {
        st->ready = true;
        st->cv.notify_all();
    } else {
        st->cv.wait(lk, [&] { return st->ready; });
    }
    return st->min_plan;
}

bool is_auto_memory_profile(const std::string& profile) {
    return profile == "auto";
}

std::vector<std::string> planner_policy_profiles(const std::string& profile) {
    if (!is_auto_memory_profile(profile)) {
        return {profile};
    }
    // `auto` is not a third concrete layout. It evaluates the concrete policy
    // families and chooses by the unified objective below.
    //
    // Four families, three nameable profiles: "balanced" and "capacity" stay
    // in this search but can no longer be pinned from config. Keeping them
    // here is what makes the value reduction free -- `auto` is the default, so
    // narrowing its search would have changed what every deployment does.
    return {"latency", "balanced", "throughput", "capacity"};
}

int derive_kv_page_size_for_profile(const std::string& profile, int tp_size) {
    // FlashInfer paged attention supports 16/32-token pages for the CUDA
    // backends we use. vLLM keeps 16 as its general default; SGLang picks by
    // backend. Pie's balanced/throughput/capacity profiles benefit from fewer
    // page refs and lower scheduler/metadata pressure, while latency keeps the
    // finer 16-token granularity for short and mixed workloads.
    if (tp_size == 1 && (profile == "latency" || profile == "balanced" ||
                         profile == "throughput")) {
        return 16;
    }
    return 32;
}


std::vector<int> derive_kv_page_size_candidates(
    const pie_cuda_driver::Config& cfg,
    const pie_cuda_driver::HfConfig& /*hf*/,
    const cudaDeviceProp& /*prop*/) {
    // A pinned page size is a single-candidate lattice: the operator has
    // already made the choice this search exists to make.
    if (cfg.batching.kv_page_size > 0) {
        return {static_cast<int>(cfg.batching.kv_page_size)};
    }
    std::vector<int> xs;
    const int tp_size = std::max(1, cfg.distributed.tp_size);
    for (const auto& profile : planner_policy_profiles(cfg.batching.memory_profile)) {
        xs.push_back(derive_kv_page_size_for_profile(profile, tp_size));
    }
    xs.push_back(16);
    xs.push_back(32);
    std::sort(xs.begin(), xs.end());
    xs.erase(std::unique(xs.begin(), xs.end()), xs.end());
    return xs;
}

double log2_ratio(int value, int target) {
    const double v = static_cast<double>(std::max(1, value));
    const double t = static_cast<double>(std::max(1, target));
    return std::log2(v / t);
}

double target_saturation_score(int value, int target) {
    const double capped = static_cast<double>(
        std::min(std::max(1, value), std::max(1, target)));
    const double t = static_cast<double>(std::max(1, target));
    return std::log2(capped + 1.0) / std::log2(t + 1.0);
}

int profile_decode_target(const std::string& profile,
                          const pie_cuda_driver::Config& cfg,
                          const cudaDeviceProp& prop) {
    // Decode throughput has a real knee: below it we leave SMs underfed,
    // above it we often inflate attention/KV pressure without increasing
    // useful device occupancy. The first-order knee tracks SM count; larger
    // GPUs need enough independent rows to keep the decode matmuls full.
    const int sm_factor =
        (profile == "latency" || profile == "capacity") ? 4 : 6;
    int target = clamp_pow2_nearest(
        prop.multiProcessorCount * sm_factor, 64, 2048);
    return target;
}

int profile_decode_target(const pie_cuda_driver::Config& cfg,
                          const cudaDeviceProp& prop) {
    return profile_decode_target(
        is_auto_memory_profile(cfg.batching.memory_profile)
            ? "throughput"
            : cfg.batching.memory_profile,
        cfg,
        prop);
}

int profile_prefill_target(const std::string& profile,
                           const pie_cuda_driver::Config& cfg,
                           const cudaDeviceProp& prop) {
    const int tp = std::max(1, cfg.distributed.tp_size);
    const int tp_factor = std::min(tp, 2);
    const bool wide_prefill_device = prop.major >= 12;
    const int sm_factor =
        profile == "throughput" ? (wide_prefill_device ? 64 : 32) : 16;
    const int max_target =
        profile == "throughput" ? (wide_prefill_device ? 8192 : 4096) : 8192;
    int target = clamp_pow2_nearest(
        prop.multiProcessorCount * sm_factor * tp_factor, 512, max_target);
    if (profile == "latency" || profile == "capacity") {
        target = std::max(512, target / 2);
    }
    return target;
}

int profile_prefill_target(const pie_cuda_driver::Config& cfg,
                           const cudaDeviceProp& prop) {
    return profile_prefill_target(
        is_auto_memory_profile(cfg.batching.memory_profile)
            ? "throughput"
            : cfg.batching.memory_profile,
        cfg,
        prop);
}

void uniq_clip_desc(std::vector<int>& xs, int cap) {
    for (int& x : xs) x = std::max(1, std::min(x, cap));
    std::sort(xs.begin(), xs.end());
    xs.erase(std::unique(xs.begin(), xs.end()), xs.end());
    std::reverse(xs.begin(), xs.end());
}

int prefill_candidate_cap(const cudaDeviceProp& prop) {
    return prop.major >= 12 ? 16384 : 8192;
}


}  // namespace

int derive_kv_page_size(const Config& cfg,
                        const HfConfig& /*hf*/,
                        const cudaDeviceProp& /*prop*/) {
    return derive_kv_page_size_for_profile(
        is_auto_memory_profile(cfg.batching.memory_profile)
            ? "throughput"
            : cfg.batching.memory_profile,
        std::max(1, cfg.distributed.tp_size));
}

CudaMemoryPlan plan_cuda_memory(
    const pie_cuda_driver::Config& cfg,
    const pie_cuda_driver::HfConfig& hf,
    int max_intermediate,
    int max_Hq,
    int max_Hk,
    bool gemma4_selected,
    const std::vector<int>& gemma4_per_layer_head_dim,
    const std::vector<int>& gemma4_per_layer_num_kv_heads,
    const std::vector<int>& gemma4_kv_source_layer,
    bool qwen3_5_selected,
    bool qwen3_5_moe_selected,
    int qwen3_5_linear_layers,
    bool nemotron_h_selected,
    int nemotron_h_mamba_layers,
    bool deepseek_v4_selected,
    bool kimi_selected,
    bool glm5_selected,
    bool kimi_k3_selected,
    const pie_cuda_driver::KvCacheFormat& kv_format,
    const pie_cuda_driver::ops::RuntimeQuantScratchSpec& runtime_quant_scratch_base,
    bool verbose)
{
    int dev_id = 0;
    CUDA_CHECK(cudaGetDevice(&dev_id));
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev_id));
    std::size_t free_bytes = 0;
    std::size_t total_bytes = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));

    const std::size_t current_used = total_bytes - free_bytes;
    const std::size_t graph_runtime_reserve = std::max<std::size_t>(
        512ull * 1024 * 1024,
        static_cast<std::size_t>(static_cast<double>(total_bytes) * 0.01));
    const std::size_t safety = std::min<std::size_t>(
        1024ull * 1024 * 1024,
        graph_runtime_reserve);
    const std::size_t usable = static_cast<std::size_t>(
        static_cast<double>(total_bytes) * cfg.batching.gpu_mem_utilization);
    if (usable <= current_used + safety) {
        throw std::runtime_error(
            "cuda memory planner: no budget left after weights. usable=" +
            std::to_string(usable / (1024 * 1024)) + " MiB, used=" +
            std::to_string(current_used / (1024 * 1024)) + " MiB, safety=" +
            std::to_string(safety / (1024 * 1024)) + " MiB");
    }
    const std::size_t budget = usable - current_used - safety;
    // Published before anything reads it, so a calibration sweep can record the
    // budget it ran inside alongside the shape it chose.
    set_planner_budget_bytes(budget);

    const int tp_size = std::max(1, cfg.distributed.tp_size);
    const bool auto_profile = is_auto_memory_profile(cfg.batching.memory_profile);
    std::vector<std::string> policy_profiles =
        planner_policy_profiles(cfg.batching.memory_profile);
    const bool narrow_latency_auto =
        auto_profile && prop.multiProcessorCount < 100 && hf.hidden_size <= 2048;
    if (narrow_latency_auto) {
        policy_profiles = {"latency"};
    }
    const bool score_as_auto = auto_profile && !narrow_latency_auto;
    // Read once: this boot either measures or serves, and the answer decides
    // both which candidates exist and which one gets built.
    const bool calibrating = planner_calibration_requested();
    const std::vector<int> kv_page_sizes =
        derive_kv_page_size_candidates(cfg, hf, prop);

    const std::size_t per_kv_token_bytes =
        deepseek_v4_selected
            ? static_cast<std::size_t>(hf.num_hidden_layers) *
                  pie_cuda_driver::kv_cache_device_bytes_per_page(
                      kv_format, 1, 1, hf.head_dim) +
                  pie_cuda_driver::dsv4_compress_bytes_per_token(hf)
            : (kimi_selected || glm5_selected || kimi_k3_selected)
            ? static_cast<std::size_t>(hf.num_hidden_layers) *
                  (static_cast<std::size_t>(hf.kv_lora_rank) +
                   static_cast<std::size_t>(hf.qk_rope_head_dim)) *
                  sizeof(std::uint16_t) +
                  static_cast<std::size_t>(hf.num_hidden_layers) *
                      pie_cuda_driver::kv_cache_device_bytes_per_page(
                          kv_format, 1, 1, 1)
            : gemma4_selected
            ? pie_cuda_driver::kv_page_bytes_per_layer(hf, gemma4_per_layer_head_dim,
                                      gemma4_per_layer_num_kv_heads,
                                      gemma4_kv_source_layer, tp_size,
                                      kv_format)
            : nemotron_h_selected
                ? pie_cuda_driver::model::kv_page_bytes_nemotron_h(hf, tp_size, kv_format)
            : pie_cuda_driver::kv_page_bytes_homogeneous(hf, tp_size, kv_format);
    if (per_kv_token_bytes == 0) {
        throw std::runtime_error("cuda memory planner: computed zero KV page bytes");
    }
    const std::size_t global_per_kv_token_bytes =
        per_kv_token_bytes * static_cast<std::size_t>(tp_size);
    const int throughput_decode_target =
        profile_decode_target("throughput", cfg, prop);
    const bool kv_heavy_auto_model =
        global_per_kv_token_bytes >= 192ull * 1024ull;
    const int auto_decode_target =
        std::min(kv_heavy_auto_model ? 256 : 512, throughput_decode_target);
    constexpr int forced_prefill = 0;
    // Qwen3-8B on L40-class TP1 has a measured prefill-shape knee above the
    // generic 8k cap: 12k keeps the initial 512-request prompt wave in a
    // faster two-chunk cadence without shrinking decode residency below R=512.
    const bool prefer_qwen3_8b_prefill_shape =
        auto_profile && forced_prefill == 0 && tp_size == 1 &&
        prop.major >= 8 && prop.major < 12 &&
        prop.multiProcessorCount >= 100 &&
        hf.model_type == "qwen3" && hf.hidden_size == 4096;
    // The same dense Qwen3-8B shape has a different TP2 knee on Ada/L40:
    // smaller page-16 KV and a ~5.5k token workspace reduce graph/arena
    // pressure enough to beat vLLM, while the generic 8k/6k candidates are
    // consistently slower. Keep the predicate architectural rather than
    // checkpoint-name based.
    const bool prefer_qwen3_8b_tp2_ada_shape =
        auto_profile && forced_prefill == 0 && tp_size == 2 &&
        prop.major == 8 && prop.minor == 9 &&
        prop.multiProcessorCount >= 100 &&
        hf.model_type == "qwen3" && hf.hidden_size == 4096;
    const bool prefer_nemotron_h_tp2_ada_prefill_shape =
        auto_profile && forced_prefill == 0 && tp_size == 2 &&
        prop.major == 8 && prop.minor == 9 &&
        prop.multiProcessorCount >= 100 && nemotron_h_selected;
    const int base_prefill_cap = prefill_candidate_cap(prop);
    const int prefill_cap =
        forced_prefill > 0
            ? std::max(forced_prefill, base_prefill_cap)
            : (prefer_qwen3_8b_prefill_shape
                   ? std::max(base_prefill_cap, 12288)
                   : base_prefill_cap);
    const int auto_prefill_target =
        prefer_qwen3_8b_tp2_ada_shape
            ? 5632
            : (prefer_qwen3_8b_prefill_shape
                   ? prefill_cap
                   : std::min(
                         prefill_cap,
                         2 * profile_prefill_target("throughput", cfg, prop)));

    // Kimi-K3 keeps the same kind of per-request linear-attention state, and
    // its widths coincide: KDA is not grouped, so `linear_num_key_heads ==
    // linear_num_value_heads` and the two head dims are equal, which makes the
    // `2*K + V` conv width below exactly K3's three short-convolution bands.
    // Leaving K3 out reserves **zero** bytes for its state slots, and the
    // planner then hands out a `max_requests` the cache cannot back.
    const bool has_qwen_linear_state =
        qwen3_5_selected || qwen3_5_moe_selected || kimi_k3_selected;
    const std::size_t K_dim =
        static_cast<std::size_t>(
            std::max(0, hf.linear_num_key_heads / tp_size)) *
        std::max(0, hf.linear_key_head_dim);
    const std::size_t V_dim =
        static_cast<std::size_t>(
            std::max(0, hf.linear_num_value_heads / tp_size)) *
        std::max(0, hf.linear_value_head_dim);
    const std::size_t conv_dim = 2 * K_dim + V_dim;
    // Recurrent slabs are bf16 by default (PIE_QWEN35_RS_STATE_DTYPE);
    // size slots with the same dtype `RecurrentStateCache::allocate`
    // will use, or the planner over-reserves ~2x the recurrent
    // footprint and caps state_slots / KV pages too low.
    const std::size_t recurrent_elem_bytes =
        pie_cuda_driver::RecurrentStateCache::recurrent_state_bf16_default()
            ? sizeof(std::uint16_t)
            : sizeof(float);
    const std::size_t per_slot_recurrent =
        static_cast<std::size_t>(
            std::max(0, hf.linear_num_value_heads / tp_size)) *
        std::max(0, hf.linear_key_head_dim) *
        std::max(0, hf.linear_value_head_dim) * recurrent_elem_bytes;
    const std::size_t per_slot_conv =
        static_cast<std::size_t>(std::max(0, hf.linear_conv_kernel_dim)) *
        conv_dim * sizeof(std::uint16_t);
    std::size_t state_slot_bytes = has_qwen_linear_state
        ? static_cast<std::size_t>(std::max(0, qwen3_5_linear_layers)) *
              (per_slot_recurrent + per_slot_conv)
        : 0;
    if (nemotron_h_selected) {
        state_slot_bytes =
            pie_cuda_driver::model::nemotron_h_state_slot_bytes(hf, nemotron_h_mamba_layers, tp_size);
    }

    struct Candidate {
        CudaMemoryPlan plan;
        std::string policy_profile;
        int decode_target = 0;
        int prefill_target = 0;
        double score = -std::numeric_limits<double>::infinity();
        std::size_t arena_bytes = 0;
        std::size_t kv_tokens = 0;
    };
    std::vector<Candidate> candidates;

    for (const auto& policy_profile : policy_profiles) {
    const int decode_target = profile_decode_target(policy_profile, cfg, prop);
    const int prefill_target = profile_prefill_target(policy_profile, cfg, prop);

    std::vector<int> Ns = {
        2 * prefill_target,
        prefill_target,
        std::max(1, prefill_target / 2),
        1024,
        512,
    };
    if (policy_profile == "throughput") {
        Ns.push_back(4 * prefill_target);
    }
    if (policy_profile == "capacity") {
        Ns.push_back(std::max(1, prefill_target / 4));
    }
    if (score_as_auto) {
        Ns.push_back(4 * prefill_target);
        Ns.push_back(std::max(1, prefill_target / 4));
        if (prefer_qwen3_8b_tp2_ada_shape) {
            Ns.push_back(5632);
        }
    }
    if (forced_prefill > 0) {
        Ns.push_back(forced_prefill);
    }
    std::vector<int> Rs = {
        2 * decode_target,
        decode_target,
        std::max(1, decode_target / 2),
        256,
        128,
        64,
        32,
    };
    if (policy_profile == "throughput" || score_as_auto) {
        Rs.push_back(4 * decode_target);
    }
    if (policy_profile == "latency") {
        Rs.push_back(std::max(1, decode_target / 4));
    }
    // A calibration boot searches the space the DEVICE allows, not the space
    // the score prefers. `Ns` and `Rs` above are generated from this profile's
    // targets and then clipped by `prefill_cap` -- which is the analytic model
    // both proposing the candidates and bounding them, so a measurement taken
    // inside it can trim the model's answer but never overrule it. That is the
    // circularity: `prefer_qwen3_8b_prefill_shape` claims the budget should be
    // HIGHER than the generic cap, and a sweep confined below the cap cannot
    // evaluate that claim in either direction.
    //
    // So widen both ladders to a geometric sweep and let the only real
    // boundary do the cutting: `arena + persistent_bytes >= budget` below,
    // which is memory, not preference.
    if (calibrating) {
        for (int n = 256; n <= 131072; n *= 2) Ns.push_back(n);
        for (int r = 16; r <= 4096; r *= 2) Rs.push_back(r);
    }
    uniq_clip_desc(Ns, calibrating ? 131072 : prefill_cap);
    uniq_clip_desc(Rs, 4096);

    // A pinned axis is a single-candidate lattice. This is the point of
    // `pie config tune` writing what it measured: with the shape pinned,
    // the analytic score and every per-(model, GPU) special case below stop
    // running at all, because there is nothing left for them to choose between.
    //
    // Not honoured during a calibration boot -- that boot exists to search, and
    // seeding a search from its own previous answer makes the value a one-way
    // ratchet. Same argument as the profile cache at the bottom of this file.
    if (!calibrating && cfg.batching.max_forward_tokens > 0) {
        Ns.assign(1, static_cast<int>(cfg.batching.max_forward_tokens));
    }
    if (!calibrating && cfg.batching.max_forward_requests > 0) {
        Rs.assign(1, static_cast<int>(cfg.batching.max_forward_requests));
    }
    // Quest key envelopes are `[num_pages, kv_heads, head_dim]` bf16 x2 per
    // layer, i.e. per PAGE rather than per token, so they do not scale with
    // `kv_page_size`. They live in the same arena as the pages and must be
    // charged here: the pool is sized to consume the device, so a page count
    // picked without them leaves nothing to allocate them from.
    // `KvCache::envelopes_requested()` reads the same switch, and the two MUST
    // agree or the cache will overrun its budget.
    const std::size_t envelope_bytes_per_page =
        pie_cuda_driver::KvCache::envelopes_requested()
            ? 2ull * sizeof(std::uint16_t) *
                  static_cast<std::size_t>(hf.num_hidden_layers) *
                  static_cast<std::size_t>(
                      std::max(1, hf.num_key_value_heads / std::max(1, tp_size))) *
                  static_cast<std::size_t>(hf.head_dim_kernel)
            : 0ull;
    for (int kv_page_size : kv_page_sizes) {
        const std::size_t per_page_bytes =
            per_kv_token_bytes * static_cast<std::size_t>(kv_page_size) +
            envelope_bytes_per_page;
        if (per_page_bytes == 0) continue;
        for (int N : Ns) {
            for (int R0 : Rs) {
                if (R0 > N) continue;
            const int max_page_refs = std::max(262144, R0 * 512);
            const int max_custom_mask_bytes =
                std::max(8 * 1024 * 1024,
                         std::min(128 * 1024 * 1024,
                                  static_cast<int>(
                                      (static_cast<std::int64_t>(N) *
                                       std::max(1024, R0 * 64) + 7) / 8)));
            const int output_rows = R0;
            int mtp_drafts_per_program = 0;
            if (qwen3_5_selected || qwen3_5_moe_selected) {
                // Same clamp as `configured_mtp_num_drafts` in context.cpp;
                // the arena must be sized for exactly the K that will run.
                mtp_drafts_per_program =
                    std::clamp(cfg.model.mtp_num_drafts, 0, 32);
            }
            std::size_t arena = 0;
            arena += pie_cuda_driver::model::workspace_bytes(
                hf, N, output_rows, max_intermediate, max_Hq, max_Hk,
                R0 * mtp_drafts_per_program);
            if (qwen3_5_selected || qwen3_5_moe_selected) {
                arena += pie_cuda_driver::model::qwen3_5_la_workspace_bytes(
                    hf, N, tp_size);
            }
            if (qwen3_5_moe_selected) {
                arena += pie_cuda_driver::model::qwen3_5_moe_workspace_bytes(
                    hf, N, tp_size);
            }
            if (nemotron_h_selected) {
                arena += pie_cuda_driver::model::nemotron_h_workspace_bytes(
                    hf, N, tp_size);
            }
            if (gemma4_selected && hf.gemma4_enable_moe) {
                arena += pie_cuda_driver::model::gemma4_moe_workspace_bytes(
                    hf, N);
            }
            if (deepseek_v4_selected) {
                arena += pie_cuda_driver::model::dsv4_workspace_bytes(
                    hf, N, output_rows, tp_size);
            }
            if (kimi_selected) {
                arena += pie_cuda_driver::model::kimi_workspace_bytes(
                    hf, N, output_rows, tp_size);
            }
            if (glm5_selected) {
                arena += pie_cuda_driver::model::glm5_workspace_bytes(
                    hf, N, output_rows, hf.max_position_embeddings, tp_size);
            }
            if (kimi_k3_selected) {
                arena += pie_cuda_driver::model::kimi_k3_workspace_bytes(
                    hf, N, output_rows, tp_size);
            }
            const std::size_t attn_float_bytes =
                pie_cuda_driver::attention_float_workspace_bytes(
                    hf, cfg, prop);
            arena += attn_float_bytes;     // AttentionWorkspace float section
            arena += 8ull * 1024 * 1024;  // AttentionWorkspace int section
            const std::size_t persistent_bytes =
                pie_cuda_driver::persistent_input_bytes(
                    N, R0, max_page_refs, max_custom_mask_bytes);
            auto quant_scratch_spec = runtime_quant_scratch_base;
            quant_scratch_spec.max_tokens = static_cast<std::size_t>(N);
            const std::size_t runtime_quant_scratch_bytes =
                pie_cuda_driver::ops::runtime_quant_scratch_bytes(
                    quant_scratch_spec);
            arena += runtime_quant_scratch_bytes;
            arena = align_up(arena, 2ull * 1024 * 1024);
            if (arena + persistent_bytes >= budget) continue;

            int R = R0;
            const int state_slots = state_slot_bytes > 0 ? R : 0;
            const std::size_t state_bytes =
                static_cast<std::size_t>(state_slots) * state_slot_bytes;
            const std::size_t minimum_wave_kv_bytes =
                static_cast<std::size_t>(R) * per_page_bytes;
            if (arena + persistent_bytes >= budget ||
                state_bytes > budget - arena - persistent_bytes ||
                minimum_wave_kv_bytes >
                    budget - arena - persistent_bytes - state_bytes) {
                continue;
            }
            const int kv_pages = static_cast<int>(std::min<std::size_t>(
                static_cast<std::size_t>(std::numeric_limits<int>::max()),
                budget / per_page_bytes));
            if (kv_pages <= 0) continue;
            const std::size_t kv_tokens =
                static_cast<std::size_t>(kv_pages) * kv_page_size;
            // A candidate only needs enough KV to be viable for early decode;
            // admission and eviction can handle longer tails. Scoring, however,
            // should value layouts that keep a realistic long-output cohort
            // resident. Using the same small horizon for both made auto prefer
            // very large request caps that fragmented 512-token generations.
            const bool kv_heavy_model =
                global_per_kv_token_bytes >= 192ull * 1024ull;
            const bool low_horizon_kv_heavy =
                kv_heavy_model &&
                (prop.major >= 12 ||
                 total_bytes >= 120ull * 1024ull * 1024ull * 1024ull);
            const double min_kv_horizon =
                score_as_auto
                    ? (low_horizon_kv_heavy ? 128.0 : 256.0)
                    : policy_profile == "latency"
                          ? 256.0
                          : policy_profile == "throughput"
                              ? 512.0
                              : 608.0;
            const double score_kv_horizon =
                score_as_auto ? (low_horizon_kv_heavy ? 384.0 : 544.0) : 608.0;
            const std::size_t min_kv_tokens = std::max<std::size_t>(
                kMinKvTokensFloor,
                static_cast<std::size_t>(
                    std::ceil(static_cast<double>(R) * min_kv_horizon)));
            if (kv_tokens < min_kv_tokens) continue;

            CudaMemoryPlan p;
            p.kv_page_size = kv_page_size;
            p.max_workspace_tokens = N;
            p.max_requests = R;
            p.max_page_refs = std::max(262144, R * 512);
            p.kv_page_bytes = per_page_bytes;
            p.attn_float_workspace_bytes = attn_float_bytes;
            p.runtime_quant_scratch_bytes = runtime_quant_scratch_bytes;
            p.persistent_input_bytes = persistent_bytes;
            p.capacity = PlannedForwardLimits{
                N,
                R,
                p.max_page_refs,
                output_rows,
                output_rows,
                max_custom_mask_bytes,
                output_rows,
                output_rows,
            };

            const int score_decode_target =
                score_as_auto ? auto_decode_target : decode_target;
            const int score_prefill_target =
                score_as_auto ? auto_prefill_target : prefill_target;
            const double prefill_score =
                target_saturation_score(N, score_prefill_target);
            const double decode_score =
                target_saturation_score(R, score_decode_target);
            const double decode_shape_penalty =
                std::abs(log2_ratio(R, score_decode_target));
            const double prefill_shape_penalty =
                std::abs(log2_ratio(N, score_prefill_target));
            const double prefill_overshoot_penalty =
                std::max(0.0, log2_ratio(N, score_prefill_target));
            const double kv_score =
                std::log1p(static_cast<double>(kv_tokens) / 65536.0);
            const double kv_headroom =
                static_cast<double>(kv_tokens) /
                std::max(1.0, static_cast<double>(R) * score_kv_horizon);
            const double kv_headroom_score = std::log1p(kv_headroom);
            const double min_headroom =
                score_as_auto ? 1.0 :
                policy_profile == "capacity" ? 1.0 :
                policy_profile == "throughput" ? 1.0 :
                1.25;
            const double kv_headroom_penalty =
                kv_headroom < min_headroom ? (min_headroom - kv_headroom) : 0.0;
            const double pressure =
                static_cast<double>(
                    arena + persistent_bytes + state_bytes +
                    minimum_wave_kv_bytes) /
                static_cast<double>(budget);
            double page_score = 0.0;
            if (score_as_auto) {
                if (prefer_qwen3_8b_tp2_ada_shape) {
                    page_score = (kv_page_size == 16) ? 0.35 : -0.10;
                } else if (tp_size == 1) {
                    page_score = (kv_page_size == 16) ? 0.20 : -0.05;
                } else {
                    const bool latency_shaped =
                        policy_profile == "latency" && R <= 256;
                    const bool metadata_heavy =
                        R >= 512 || N >= 4096 || p.max_page_refs >= 262144;
                    if (latency_shaped && !metadata_heavy) {
                        page_score = (kv_page_size == 16) ? 0.20 : -0.05;
                    } else {
                        page_score = (kv_page_size == 32) ? 0.20 : 0.0;
                    }
                }
            } else if (policy_profile == "latency") {
                page_score = (kv_page_size == 16) ? 0.20 : -0.20;
            } else if (policy_profile == "throughput") {
                page_score = (tp_size == 1)
                    ? ((kv_page_size == 16) ? 0.25 : -0.10)
                    : ((kv_page_size == 32) ? 0.25 : 0.0);
            } else {
                page_score = (tp_size == 1)
                    ? ((kv_page_size == 16) ? 0.15 : -0.05)
                    : ((kv_page_size == 32) ? 0.15 : 0.0);
            }
            double score = 0.0;
            const auto& profile = policy_profile;
            if (score_as_auto) {
                const double cohort_score =
                    target_saturation_score(R, score_decode_target);
                const double kv_residency_score =
                    std::log1p(kv_headroom) +
                    std::log1p(static_cast<double>(kv_tokens) / 131072.0);
                const double arena_mib =
                    static_cast<double>(arena) /
                    static_cast<double>(1024ull * 1024ull);
                const bool enough_kv_headroom = kv_headroom >= min_headroom;
                const double arena_penalty =
                    enough_kv_headroom
                        ? pressure * 0.25
                        : arena_mib / 1024.0 + pressure * 0.75;
                const double prefill_weight =
                    enough_kv_headroom ? (tp_size > 1 ? 4.0 : 3.0) : 2.0;
                const double kv_weight =
                    enough_kv_headroom ? 2.0 : 4.0;
                const double prefill_underfill_penalty =
                    prefer_qwen3_8b_prefill_shape
                        ? std::max(0.0,
                                   -log2_ratio(N, score_prefill_target))
                        : 0.0;
                const double prefill_target_bonus =
                    (enough_kv_headroom &&
                     N >= score_prefill_target &&
                     R >= score_decode_target)
                        ? 1.25
                        : 0.0;
                score = cohort_score * 6.0 +
                        decode_score * 4.0 +
                        prefill_score * prefill_weight +
                        kv_residency_score * kv_weight +
                        prefill_target_bonus +
                        page_score -
                        decode_shape_penalty * 6.0 -
                        prefill_underfill_penalty *
                            (enough_kv_headroom ? 2.0 : 0.5) -
                        prefill_overshoot_penalty * 0.75 -
                        prefill_shape_penalty * 0.5 -
                        kv_headroom_penalty * 4.0 -
                        arena_penalty;
            } else if (profile == "capacity") {
                score = kv_score * 9.0 + kv_headroom_score * 4.0 +
                        decode_score * 2.5 + page_score -
                        decode_shape_penalty * 8.0 -
                        prefill_shape_penalty * 2.0 -
                        kv_headroom_penalty * 4.0 -
                        static_cast<double>(arena) /
                            static_cast<double>(512ull * 1024ull * 1024ull);
            } else if (profile == "throughput") {
                score = prefill_score * 3.0 + decode_score * 5.0 +
                        kv_score * 1.25 + kv_headroom_score * 2.0 +
                        page_score -
                        decode_shape_penalty * 4.0 -
                        prefill_shape_penalty * 0.75 -
                        kv_headroom_penalty * 3.0 - pressure;
            } else if (profile == "latency") {
                score = prefill_score + decode_score * 1.5 +
                        kv_score * 1.25 + kv_headroom_score + page_score -
                        decode_shape_penalty * 2.0 -
                        static_cast<double>(R) / std::max(1, N) -
                        pressure * 2.0;
            } else {
                score = prefill_score * 1.5 + decode_score * 3.0 +
                        kv_score * 3.0 + kv_headroom_score * 2.0 +
                        page_score -
                        decode_shape_penalty * 4.0 -
                        prefill_shape_penalty -
                        kv_headroom_penalty * 3.0 - pressure * 2.0;
            }
            if (qwen3_5_moe_selected && tp_size > 1 &&
                (auto_profile || profile == "latency") &&
                forced_prefill == 0) {
                // Qwen3.6-MoE TP2 is decode-heavy but still suffers when
                // the prompt wave is split into 1k-token chunks. The measured
                // knee on L40 is N=2048: R128/N1024 loses throughput, while
                // R128/N2048 matches the older R64/N2048 path.
                score += (N >= 2048) ? 1.5 : -1.5;
                score -= std::abs(log2_ratio(N, 2048)) * 4.0;
            }
            if (prefer_nemotron_h_tp2_ada_prefill_shape) {
                // Nemotron-H TP2 prompt bursts on L40 need the 8k workspace
                // to keep 128 short prompts in one prefill batch. The 8k
                // plan still leaves 256 recurrent slots, so decode residency
                // does not regress versus the 4k auto-selected shape.
                score += (N >= 8192) ? 1.5 : -1.5;
                score -= std::abs(log2_ratio(N, 8192)) * 4.0;
            }
            if (forced_prefill > 0) {
                score += (N == forced_prefill) ? 1000.0 : -1000.0;
            }
            candidates.push_back(Candidate{
                p,
                policy_profile,
                score_decode_target,
                score_prefill_target,
                score,
                arena,
                kv_tokens});
            }
        }
    }
    }

    if (candidates.empty()) {
        std::string why =
            "cuda memory planner: no viable forward/KV layout fits budget " +
            std::to_string(budget / (1024 * 1024)) + " MiB";
        // A pin is the likeliest reason a lattice that normally has hundreds of
        // candidates has none -- and the operator can act on that, where they
        // cannot act on "no layout fits".
        if (cfg.batching.max_forward_tokens > 0 ||
            cfg.batching.max_forward_requests > 0) {
            why +=
                ". [driver] max_forward_tokens/max_forward_requests pin the "
                "shape to a single candidate; unset them to let the planner "
                "choose, or re-run `pie config tune` on this machine";
        } else if (per_kv_token_bytes > 0) {
            // Unpinned, the usual reason is simply that the weights left too
            // little behind to clear the KV floor every candidate must meet.
            // "No layout fits" is not something an operator can act on; the
            // shortfall and the utilization that would cover it are.
            const std::size_t have_tokens = budget / per_kv_token_bytes;
            const std::size_t need_bytes =
                kMinKvTokensFloor * per_kv_token_bytes;
            // Round the advice UP to the next hundredth: truncating it hands
            // back a utilization that still lands under the floor, which is
            // worse than no advice at all.
            const double need_util =
                total_bytes > 0
                    ? std::ceil(
                          static_cast<double>(
                              need_bytes + current_used + safety) /
                          static_cast<double>(total_bytes) * 100.0) / 100.0
                    : 0.0;
            char util_text[8] = {};
            std::snprintf(util_text, sizeof(util_text), "%.2f", need_util);
            why += ". KV needs " +
                   std::to_string(per_kv_token_bytes / 1024) +
                   " KiB/token, so this budget holds ~" +
                   std::to_string(have_tokens) + " tokens but every layout "
                   "needs at least " + std::to_string(kMinKvTokensFloor) +
                   ". Raise [driver] gpu_mem_utilization (>= " +
                   std::string(util_text) +
                   " here), shrink the weights (`kv_cache_dtype`/quantization), "
                   "or add a GPU";
        }
        throw std::runtime_error(why);
    }

    // A measured shape beats a scored one. `calibrate_memory_planner` times the
    // real forward step across the ladder and records the winner here; when it
    // has run for this exact (device, model, tp, kv format) we take its answer
    // and skip the analytic score entirely. Fields the calibrator did not
    // measure stay zero and match anything.
    bool selected_from_profile = false;
    auto best_it = candidates.end();
    // Two guards beyond `auto_profile`, matching the fingerprint overrides
    // below:
    //   * `forced_prefill == 0` — an explicit `PIE_CUDA_PREFILL_TOKENS` is an
    //     operator instruction. Every other override defers to it; the cache
    //     must too, or a stale file silently discards what was asked for.
    //   * not calibrating — the sweep can only explore up to the arena this
    //     plan builds. Seeding a calibration run from its OWN previous answer
    //     makes the budget a one-way ratchet: once it lowers, no later sweep
    //     can look above the lowered value. A calibration run therefore plans
    //     from the rule path, so every sweep starts from the same ceiling.
    const bool use_profile_cache =
        auto_profile && forced_prefill == 0 && !planner_calibration_requested();
    if (use_profile_cache) {
        std::string cache_error;
        auto measured =
            planner_profile_cache_lookup(
                make_planner_profile_key(prop, hf, tp_size, kv_format),
                &cache_error);
        if (!cache_error.empty()) {
            std::cerr << "[pie-driver-cuda] memory planner: ignored profile "
                      << "cache " << planner_profile_cache_path().string()
                      << ": " << cache_error << "\n";
        }
        // A shape is only an answer to the budget it was measured under. The
        // key pins the device and the model, and neither notices that this boot
        // has materially more or less memory to give than the sweep did --
        // another process holding VRAM, or a checkpoint requantized offline by
        // `pie model build`. Left unchecked this fails in the quiet
        // direction: with a LARGER budget the measured shape is still feasible,
        // so it is selected, and the extra memory is simply never used.
        if (measured.has_value() && measured->budget_bytes > 0) {
            const double drift =
                std::abs(static_cast<double>(budget) -
                         static_cast<double>(measured->budget_bytes)) /
                static_cast<double>(measured->budget_bytes);
            if (drift > kPlannerBudgetTolerance) {
                std::cerr << "[pie-driver-cuda] memory planner: profile cache "
                          << "was measured against a "
                          << (measured->budget_bytes / (1024 * 1024))
                          << " MiB budget and this boot has "
                          << (budget / (1024 * 1024)) << " MiB ("
                          << static_cast<int>(drift * 100.0)
                          << "% apart); the measurement does not describe this "
                          << "machine, so the scored rule decides. Re-run "
                          << "`pie config tune` if the change is "
                          << "permanent, or free the device if it is not.\n";
                measured.reset();
            }
        }
        if (measured.has_value()) {
            for (auto it = candidates.begin(); it != candidates.end(); ++it) {                if (!measured->policy_profile.empty() &&
                    it->policy_profile != measured->policy_profile) {
                    continue;
                }
                if (measured->kv_page_size > 0 &&
                    it->plan.kv_page_size != measured->kv_page_size) {
                    continue;
                }
                if (measured->max_forward_tokens > 0 &&
                    it->plan.max_workspace_tokens !=
                        measured->max_forward_tokens) {
                    continue;
                }
                if (measured->max_forward_requests > 0 &&
                    it->plan.max_requests != measured->max_forward_requests) {
                    continue;
                }
                // The cache pins only what was measured; among the candidates
                // that satisfy it the score still decides, so a sweep over one
                // axis does not silently freeze the others.
                if (!selected_from_profile || best_it->score < it->score) {
                    best_it = it;
                    selected_from_profile = true;
                }
            }
            if (!selected_from_profile) {
                // A measured entry that lands on no feasible candidate is the
                // worst failure mode this cache has: it looks exactly like no
                // cache at all. Say so rather than falling through silently.
                std::cerr << "[pie-driver-cuda] memory planner: profile cache "
                          << "pins max_forward_tokens="
                          << measured->max_forward_tokens
                          << " but no candidate layout matches it; falling "
                          << "back to the scored rule. Delete "
                          << planner_profile_cache_path().string()
                          << " to re-calibrate.\n";
            }
        }
    }
    // A calibration boot builds the CEILING of the feasible region rather than
    // the score's pick, because a bigger arena can run a smaller shape and not
    // the other way round. With the ceiling built, the sweep's downward-only
    // ladder stops being a restriction and becomes the correct direction.
    //
    // The ceiling is the largest explorable rectangle: the sweep runs shapes
    // with `requests <= max_requests` and `tokens <= max_workspace_tokens`, so
    // the box it can cover is the product. This is one point on a frontier, not
    // the frontier -- a taller box is a narrower one -- and covering the whole
    // frontier would take more than one calibration boot. Worth saying rather
    // than implying.
    //
    // The small KV pool this leaves is free here: a calibration boot serves
    // nothing, so pages it does not have are pages nothing wanted.
    if (calibrating) {
        best_it = std::max_element(
            candidates.begin(), candidates.end(),
            [](const Candidate& a, const Candidate& b) {
                const auto area = [](const Candidate& c) {
                    return static_cast<std::int64_t>(c.plan.max_workspace_tokens) *
                           static_cast<std::int64_t>(c.plan.max_requests);
                };
                return area(a) < area(b);
            });
        std::cerr << "[pie-driver-cuda] memory planner: calibration boot -- "
                  << candidates.size() << " candidates fit the "
                  << (budget / (1024 * 1024)) << " MiB budget; building the "
                  << "largest at N=" << best_it->plan.max_workspace_tokens
                  << " R=" << best_it->plan.max_requests
                  << " page_size=" << best_it->plan.kv_page_size
                  << " (score would have picked N="
                  << std::max_element(
                         candidates.begin(), candidates.end(),
                         [](const Candidate& a, const Candidate& b) {
                             return a.score < b.score;
                         })->plan.max_workspace_tokens
                  << ")\n";
        // The paragraph above justifies the starved KV pool with "a
        // calibration boot serves nothing" -- and nothing in this process
        // enforces that. Calibration does not exit; when it is done this
        // arena is what serves. Say so here, because the symptom on the far
        // side (a small page pool, and the sweep's seconds on every start)
        // reads as a hardware limit rather than as a flag left on.
        std::cerr << "[pie-driver-cuda] memory planner: this arena is built to "
                     "be MEASURED, not to serve -- its KV pool is the smallest "
                     "the largest forward shape leaves. Unset [driver] "
                     "calibrate_planner before serving from this config.\n";
    }
    if (best_it == candidates.end()) {
        if (prefer_qwen3_8b_prefill_shape) {
            const int preferred_tokens = prefill_cap;
            auto preferred_it = candidates.end();
            for (auto it = candidates.begin(); it != candidates.end(); ++it) {
                if (it->plan.max_workspace_tokens != preferred_tokens ||
                    it->plan.max_requests < auto_decode_target ||
                    it->plan.kv_page_size != 16) {
                    continue;
                }
                if (preferred_it == candidates.end() ||
                    preferred_it->score < it->score) {
                    preferred_it = it;
                }
            }
            if (preferred_it != candidates.end()) {
                best_it = preferred_it;
            }
        }
    }
    if (best_it == candidates.end()) {
        best_it = std::max_element(candidates.begin(), candidates.end(),
        [](const Candidate& a, const Candidate& b) {
            return a.score < b.score;
        });
    }
    // Planner introspection: the selected plan alone cannot tell you WHY it
    // won, nor what the score-ranked runner-up was. Both are needed to judge
    // whether an override is load-bearing or dead weight.
    if constexpr (false) {
        const int want = 1;
        std::vector<const Candidate*> ranked;
        ranked.reserve(candidates.size());
        for (const auto& c : candidates) ranked.push_back(&c);
        std::stable_sort(ranked.begin(), ranked.end(),
                         [](const Candidate* a, const Candidate* b) {
                             return a->score > b->score;
                         });
        std::cerr << "[pie-driver-cuda] planner candidates: "
                  << candidates.size() << " feasible, top "
                  << std::min<std::size_t>(ranked.size(),
                                           static_cast<std::size_t>(want))
                  << " by score (selector="
                  << (selected_from_profile ? "profiled" : "rule") << ")\n";
        for (std::size_t i = 0;
             i < ranked.size() && i < static_cast<std::size_t>(want); ++i) {
            const Candidate* c = ranked[i];
            std::cerr << "[pie-driver-cuda]   #" << (i + 1)
                      << (c == &*best_it ? " *" : "  ")
                      << " score=" << c->score
                      << " profile=" << c->policy_profile
                      << " page=" << c->plan.kv_page_size
                      << " N=" << c->plan.max_workspace_tokens
                      << " R=" << c->plan.max_requests
                      << " arena=" << (c->arena_bytes / (1024 * 1024))
                      << " MiB persist="
                      << (c->plan.persistent_input_bytes / (1024 * 1024))
                      << " MiB kv_tok=" << c->kv_tokens << "\n";
        }
        if (&*best_it != ranked.front()) {
            std::cerr << "[pie-driver-cuda]   selected is NOT the top-scoring "
                      << "candidate: an override moved it (N="
                      << best_it->plan.max_workspace_tokens
                      << " R=" << best_it->plan.max_requests
                      << " score=" << best_it->score << ")\n";
        }
    }

    CudaMemoryPlan best_plan = tp_min_plan(cfg, best_it->plan);
    const std::string best_policy_profile = best_it->policy_profile;
    const int selected_decode_target = best_it->decode_target;
    const int selected_prefill_target = best_it->prefill_target;
    if (verbose) {
        const auto& p = best_plan;
        std::cerr << "[pie-driver-cuda] memory planner: profile="
                  << cfg.batching.memory_profile
                  << " resolved_profile=" << best_policy_profile
                  << " selector="
                  << (selected_from_profile ? "profiled" : "rule")
                  << " util=" << cfg.batching.gpu_mem_utilization
                  << " total=" << (total_bytes / (1024 * 1024)) << " MiB"
                  << " sm=" << prop.multiProcessorCount
                  << " tp=" << tp_size
                  << " decode_target=" << selected_decode_target
                  << " prefill_target=" << selected_prefill_target
                  << " page_size=" << p.kv_page_size
                  << " (auto)"
                  << " used_after_weights=" << (current_used / (1024 * 1024))
                  << " MiB safety=" << (safety / (1024 * 1024)) << " MiB"
                  << " budget=" << (budget / (1024 * 1024)) << " MiB"
                  << " N=" << p.max_workspace_tokens
                  << " R=" << p.max_requests
                  << " page_refs=" << p.max_page_refs
                  << " persistent_inputs="
                  << (p.persistent_input_bytes / (1024 * 1024)) << " MiB"
                  << " rq_scratch="
                  << (p.runtime_quant_scratch_bytes / (1024 * 1024))
                  << " MiB"
                  << " logical_kv_pages="
                  << (budget / p.kv_page_bytes)
                  << " kv_tokens="
                  << ((budget / p.kv_page_bytes) *
                      static_cast<std::size_t>(p.kv_page_size))
                  << " logical_state_slots="
                  << (state_slot_bytes == 0 ? 0 : p.max_requests)
                  << "\n";
    }
    return best_plan;
}

}  // namespace pie_cuda_driver
