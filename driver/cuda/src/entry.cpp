#include "model/mistral3.hpp"
// pie_driver_cuda — native CUDA backend library entry point.
//
// All meaningful logic lives in `run_impl`; the `extern "C"` wrapper
// at the bottom catches any escaping C++ exception so we never
// propagate across the FFI boundary (which would be UB). Mirrors
// driver/portable/src/entry.cpp's shape — see that file for the
// invariants.

#include "entry.hpp"

#include <algorithm>
#include <atomic>
#include <barrier>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdlib>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <unordered_map>
#include <utility>

#include <CLI/CLI.hpp>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <fstream>
#include <nlohmann/json.hpp>
#include <vector>

#include "attention_workspace.hpp"
#include "brle.hpp"
#include "config.hpp"
#include "custom_all_reduce.hpp"
#include "cuda_check.hpp"
#include "model/loaded_model.hpp"
#include "kernels/argmax.hpp"
#include "kernels/sample_flashinfer.hpp"
#include "kernels/sample_temp.hpp"
#include "kv_cache.hpp"
#include "model/bound_model.hpp"
#include "model/gemma2.hpp"
#include "model/gemma3n.hpp"
#include "model/gemma4.hpp"
#include "model/gemma4_mtp.hpp"
#include "model/gpt_oss.hpp"
#include "model/llama_like.hpp"
#include "model/mixtral.hpp"
#include "model/qwen3.hpp"
#include "model/qwen3_5.hpp"
#include "model/qwen3_5_forward.hpp"
#include "model/qwen3_5_moe.hpp"
#include "model/qwen3_5_moe_forward.hpp"
#include "model/qwen3_forward.hpp"
#include "qwen3_5_state_cache.hpp"
#include "swap_pool.hpp"
#include <thread>
#include <unistd.h>
#include "ops/gemm.hpp"
#include "executor/executor.hpp"
#include "service/inproc_service.hpp"
#include <pie_bridge/inproc_server.hpp>

namespace {

std::mutex g_servers_mu;
std::vector<pie_driver::InProcServer*> g_servers;
std::atomic<pie_driver::InProcServer*> g_signal_server{nullptr};

void register_server(pie_driver::InProcServer* server) {
    std::lock_guard<std::mutex> lk(g_servers_mu);
    g_servers.push_back(server);
    g_signal_server.store(server);
}

void unregister_server(pie_driver::InProcServer* server) {
    std::lock_guard<std::mutex> lk(g_servers_mu);
    g_servers.erase(
        std::remove(g_servers.begin(), g_servers.end(), server),
        g_servers.end());
    if (g_signal_server.load() == server) {
        g_signal_server.store(g_servers.empty() ? nullptr : g_servers.back());
    }
}

void stop_servers() {
    std::vector<pie_driver::InProcServer*> servers;
    {
        std::lock_guard<std::mutex> lk(g_servers_mu);
        servers = g_servers;
    }
    for (auto* server : servers) {
        if (server != nullptr) server->stop();
    }
}

std::string trim_ascii(std::string s) {
    while (!s.empty() &&
           (s.back() == '\n' || s.back() == '\r' || s.back() == ' ' ||
            s.back() == '\t')) {
        s.pop_back();
    }
    std::size_t start = 0;
    while (start < s.size() &&
           (s[start] == ' ' || s[start] == '\t' || s[start] == '\n' ||
            s[start] == '\r')) {
        ++start;
    }
    if (start > 0) s.erase(0, start);
    return s;
}

bool looks_like_hf_snapshot(const std::filesystem::path& path) {
    return std::filesystem::exists(path / "config.json");
}

std::optional<std::filesystem::path> resolve_hf_cache_snapshot(
    const std::filesystem::path& repo_dir) {
    const auto snapshots_dir = repo_dir / "snapshots";
    if (!std::filesystem::is_directory(snapshots_dir)) return std::nullopt;

    const auto main_ref = repo_dir / "refs" / "main";
    if (std::filesystem::is_regular_file(main_ref)) {
        std::ifstream in(main_ref);
        std::string sha;
        std::getline(in, sha);
        sha = trim_ascii(sha);
        if (!sha.empty()) {
            const auto candidate = snapshots_dir / sha;
            if (looks_like_hf_snapshot(candidate)) return candidate;
        }
    }

    std::optional<std::filesystem::path> only_snapshot;
    int count = 0;
    for (const auto& entry : std::filesystem::directory_iterator(snapshots_dir)) {
        if (!entry.is_directory()) continue;
        if (!looks_like_hf_snapshot(entry.path())) continue;
        only_snapshot = entry.path();
        ++count;
        if (count > 1) return std::nullopt;
    }
    return only_snapshot;
}

std::optional<std::filesystem::path> discover_gemma4_mtp_snapshot_dir(
    const std::filesystem::path& target_snapshot_dir) {
    const auto direct = std::filesystem::path(
        target_snapshot_dir.string() + "-assistant");
    if (looks_like_hf_snapshot(direct)) return direct;

    for (auto cur = target_snapshot_dir;
         !cur.empty() && cur != cur.parent_path();
         cur = cur.parent_path()) {
        const std::string name = cur.filename().string();
        if (name.rfind("models--", 0) != 0) continue;
        const auto assistant_repo =
            cur.parent_path() / (name + "-assistant");
        if (auto snapshot = resolve_hf_cache_snapshot(assistant_repo)) {
            return snapshot;
        }
        break;
    }
    return std::nullopt;
}

void on_signal(int) {
    if (auto* server = g_signal_server.load()) server->stop();
}

pie_cuda_driver::model::RopeKind rope_kind_from_config(
    const pie_cuda_driver::HfConfig& hf) {
    using RopeScaling = pie_cuda_driver::HfConfig::RopeScaling;
    using RopeKind = pie_cuda_driver::model::RopeKind;
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

void apply_rope_config(
    pie_cuda_driver::model::LlamaLikeForwardCfg& fwd_cfg,
    const pie_cuda_driver::HfConfig& hf) {
    fwd_cfg.rope_kind                  = rope_kind_from_config(hf);
    fwd_cfg.yarn_factor                = hf.rope_factor;
    fwd_cfg.yarn_low_freq_factor       = hf.rope_low_freq_factor;
    fwd_cfg.yarn_high_freq_factor      = hf.rope_high_freq_factor;
    fwd_cfg.yarn_original_max_position = hf.rope_original_max_position;
    fwd_cfg.yarn_beta_fast             = hf.rope_beta_fast;
    fwd_cfg.yarn_beta_slow             = hf.rope_beta_slow;
    fwd_cfg.yarn_attention_factor      = hf.rope_attention_factor;
}

// All TP ranks in one DP group are threads in the same pie-server
// process. Rendezvous via an in-process `std::barrier` keyed by the
// shared `nccl_unique_id_hex` (which is per-DP-group by construction).
// `nccl_unique_id_hex` doubles as `tp_cpu_gate_key` for the per-fire
// CPU gate downstream (executor/executor.cpp).
void tp_startup_cpu_barrier(const pie_cuda_driver::Config& cfg) {
    if (cfg.distributed.tp_size <= 1) return;

    const std::string& key = cfg.distributed.nccl_unique_id_hex;
    if (key.empty()) return;

    static std::mutex registry_mu;
    static std::unordered_map<std::string, std::shared_ptr<std::barrier<>>>
        registry;

    std::shared_ptr<std::barrier<>> b;
    {
        std::lock_guard<std::mutex> lk(registry_mu);
        auto& entry = registry[key];
        if (!entry) {
            entry = std::make_shared<std::barrier<>>(cfg.distributed.tp_size);
        }
        b = entry;
    }
    b->arrive_and_wait();
}

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

struct PlannedForwardLimits {
    int max_forward_tokens = 0;
    int max_forward_requests = 0;
    int max_page_refs = 0;
    int max_logit_rows = 0;
    int max_prob_rows = 0;
    int max_custom_mask_bytes = 0;
    int max_sampler_rows = 0;
    int max_logprob_labels = 0;
};

struct CudaMemoryPlan {
    int kv_page_size = 0;
    int max_workspace_tokens = 0;
    int max_requests = 0;
    int max_page_refs = 0;
    int kv_pages = 0;
    int state_slots = 0;
    std::size_t attn_float_workspace_bytes = 0;
    std::size_t runtime_quant_scratch_bytes = 0;
    std::size_t arena_bytes = 0;
    std::size_t kv_bytes = 0;
    std::size_t state_bytes = 0;
    PlannedForwardLimits capacity;
};

void min_into(CudaMemoryPlan& dst, const CudaMemoryPlan& src) {
    dst.kv_page_size = std::min(dst.kv_page_size, src.kv_page_size);
    dst.max_workspace_tokens = std::min(dst.max_workspace_tokens,
                                        src.max_workspace_tokens);
    dst.max_requests = std::min(dst.max_requests, src.max_requests);
    dst.max_page_refs = std::min(dst.max_page_refs, src.max_page_refs);
    dst.kv_pages = std::min(dst.kv_pages, src.kv_pages);
    dst.state_slots = std::min(dst.state_slots, src.state_slots);
    dst.attn_float_workspace_bytes = std::max(
        dst.attn_float_workspace_bytes, src.attn_float_workspace_bytes);
    dst.runtime_quant_scratch_bytes = std::max(
        dst.runtime_quant_scratch_bytes, src.runtime_quant_scratch_bytes);
    dst.arena_bytes = std::max(dst.arena_bytes, src.arena_bytes);
    dst.kv_bytes = std::min(dst.kv_bytes, src.kv_bytes);
    dst.state_bytes = std::min(dst.state_bytes, src.state_bytes);
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

pie_cuda_driver::ops::RuntimeQuantScratchSpec runtime_quant_scratch_spec(
    const pie_cuda_driver::LoadedModel& engine,
    std::size_t max_tokens)
{
    pie_cuda_driver::ops::RuntimeQuantScratchSpec spec;
    spec.max_tokens = max_tokens;

    const auto& store = engine.weight_store();
    for (const auto& item : store.quant_meta_map()) {
        const auto& name = item.first;
        auto it = store.find(name);
        if (it == store.end()) continue;
        const auto& tensor = it->second.tensor;
        if (tensor.shape().size() != 2) continue;

        if (tensor.dtype() == pie_cuda_driver::DType::FP8_E4M3) {
            spec.has_fp8 = true;
        } else if (tensor.dtype() == pie_cuda_driver::DType::INT8) {
            spec.has_int8 = true;
        } else {
            continue;
        }

        spec.max_weight_rows = std::max<std::size_t>(
            spec.max_weight_rows,
            static_cast<std::size_t>(std::max<std::int64_t>(
                0, tensor.shape()[0])));
        spec.max_weight_cols = std::max<std::size_t>(
            spec.max_weight_cols,
            static_cast<std::size_t>(std::max<std::int64_t>(
                0, tensor.shape()[1])));
    }

    return spec;
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

std::size_t workspace_bytes(const pie_cuda_driver::HfConfig& cfg,
                            int N, int output_rows,
                            int max_intermediate,
                            int max_Hq, int max_Hk) {
    const auto bf16 = [](std::size_t elems) { return elems * 2; };
    const auto fp32 = [](std::size_t elems) { return elems * 4; };
    const std::size_t n = static_cast<std::size_t>(N);
    const std::size_t o = static_cast<std::size_t>(std::max(1, output_rows));
    std::size_t bytes = 0;
    bytes += bf16(n * cfg.hidden_size);              // y
    bytes += bf16(n * cfg.hidden_size);              // norm_x
    bytes += bf16(n * cfg.hidden_size);              // spec_hidden
    bytes += bf16(n * (max_Hq + 2 * max_Hk));        // qkv_fused
    bytes += bf16(n * (2 * max_intermediate));       // gate_up_fused
    bytes += fp32(n * cfg.head_dim);                 // rope_table
    bytes += bf16(n * max_Hq);                       // q
    bytes += bf16(n * max_Hk);                       // k
    bytes += bf16(n * max_Hk);                       // v
    bytes += bf16(n * max_Hq);                       // attn_out
    bytes += bf16(n * cfg.hidden_size);              // norm_y
    bytes += bf16(n * max_intermediate);             // gate
    bytes += bf16(n * max_intermediate);             // up
    bytes += bf16(o * cfg.vocab_size);               // logits
    bytes += fp32(o * cfg.vocab_size);               // probs
    if (cfg.head_dim != cfg.head_dim_kernel) {
        const int q_heads = max_Hq / std::max(1, cfg.head_dim);
        const int kv_heads = max_Hk / std::max(1, cfg.head_dim);
        const int Hq_pad = q_heads * cfg.head_dim_kernel;
        const int Hk_pad = kv_heads * cfg.head_dim_kernel;
        bytes += bf16(n * Hq_pad);
        bytes += bf16(n * Hk_pad);
        bytes += bf16(n * Hk_pad);
        bytes += bf16(n * Hq_pad);
    }
    return bytes;
}

std::size_t qwen3_5_la_workspace_bytes(const pie_cuda_driver::HfConfig& cfg,
                                       int N, int tp_size = 1) {
    if (cfg.linear_num_key_heads <= 0 || cfg.linear_num_value_heads <= 0) {
        return 0;
    }
    const int T = std::max(1, tp_size);
    const std::size_t n = static_cast<std::size_t>(N);
    const std::size_t k_dim =
        static_cast<std::size_t>(cfg.linear_num_key_heads / T) *
        cfg.linear_key_head_dim;
    const std::size_t v_dim =
        static_cast<std::size_t>(cfg.linear_num_value_heads / T) *
        cfg.linear_value_head_dim;
    const std::size_t conv_dim = 2 * k_dim + v_dim;
    const std::size_t v_h =
        static_cast<std::size_t>(cfg.linear_num_value_heads / T);
    const std::size_t k_h =
        static_cast<std::size_t>(cfg.linear_num_key_heads / T);
    const std::size_t hq =
        static_cast<std::size_t>(cfg.num_attention_heads / T) * cfg.head_dim;
    std::size_t bytes = 0;
    auto u16 = [](std::size_t elems) { return elems * 2; };
    auto fp32 = [](std::size_t elems) { return elems * 4; };
    bytes += u16(n * conv_dim);          // mixed_qkv
    bytes += u16(n * (conv_dim + v_dim)); // mixed_qkvz
    bytes += u16(n * 2 * v_h);           // ba
    bytes += u16(n * conv_dim);          // mixed_qkv_post
    bytes += u16(n * v_dim);             // z
    bytes += u16(n * v_h);               // a
    bytes += u16(n * v_h);               // b
    bytes += fp32(n * v_h * cfg.linear_key_head_dim); // q_norm
    bytes += fp32(n * v_h * cfg.linear_key_head_dim); // k_norm
    bytes += fp32(n * v_dim);            // v_fp32
    bytes += fp32(n * v_h);              // g_log
    bytes += fp32(n * v_h);              // beta
    bytes += fp32(n * v_dim);            // core_out
    bytes += u16(n * v_dim);             // core_out_bf16
    bytes += u16(n * k_dim);             // q_raw
    bytes += u16(n * k_dim);             // k_raw
    bytes += u16(n * v_dim);             // v_raw
    bytes += fp32(n * k_h * cfg.linear_key_head_dim); // q_pre
    bytes += fp32(n * k_h * cfg.linear_key_head_dim); // k_pre
    bytes += u16(n * 2 * hq);            // fa_qg_packed
    bytes += u16(n * hq);                // fa_gate
    return bytes;
}

std::size_t qwen3_5_moe_workspace_bytes(const pie_cuda_driver::HfConfig& cfg,
                                        int N, int tp_size = 1) {
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
    bytes += maxR *
             (6 * sizeof(void*) + sizeof(float));
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

std::size_t gemma4_moe_workspace_bytes(const pie_cuda_driver::HfConfig& cfg,
                                       int N) {
    if (!cfg.gemma4_enable_moe || cfg.num_experts <= 0 ||
        cfg.num_experts_per_tok <= 0 || cfg.moe_intermediate_size <= 0) {
        return 0;
    }
    const std::size_t n = static_cast<std::size_t>(N);
    const std::size_t maxR = n * cfg.num_experts_per_tok;
    const std::size_t H = static_cast<std::size_t>(cfg.hidden_size);
    const std::size_t I = static_cast<std::size_t>(cfg.moe_intermediate_size);
    auto u16 = [](std::size_t elems) { return elems * 2; };
    auto i32 = [](std::size_t elems) { return elems * 4; };
    auto fp32 = [](std::size_t elems) { return elems * 4; };
    std::size_t bytes = 0;
    bytes += u16(n * H);
    bytes += u16(n * cfg.num_experts);
    bytes += i32(n * cfg.num_experts_per_tok);
    bytes += fp32(n * cfg.num_experts_per_tok);
    bytes += u16(n * H);
    bytes += u16(maxR * H);
    bytes += u16(maxR * 2 * I);
    bytes += u16(maxR * I);
    bytes += u16(maxR * H);
    bytes += i32(maxR);
    bytes += fp32(maxR);
    bytes += u16(n * H);
    bytes += static_cast<std::size_t>(cfg.num_experts_per_tok) *
             (6 * sizeof(void*) + sizeof(float));
    return bytes;
}

std::size_t persistent_input_bytes(int N, int R, int max_page_refs,
                                   int max_custom_mask_bytes) {
    std::size_t bytes = 0;
    bytes += static_cast<std::size_t>(N) * (4 + 4 + 4); // tokens/positions/sampled
    bytes += static_cast<std::size_t>(R + 1) * (4 + 4); // qo + kv indptr
    bytes += static_cast<std::size_t>(R) * (4 + 4 + 1); // last lens + slot ids + fresh
    bytes += static_cast<std::size_t>(max_page_refs) * 4;
    bytes += static_cast<std::size_t>(max_custom_mask_bytes);
    bytes += static_cast<std::size_t>(R + 1) * 4;       // mask indptr
    bytes += static_cast<std::size_t>(N) *
             (sizeof(float) * 3 + sizeof(std::int32_t) * 4 +
              sizeof(std::uint32_t) + sizeof(std::uint64_t) + sizeof(bool));
    return bytes;
}

bool flashinfer_decode_supports_gqa(int gqa) {
    return gqa == 1 || gqa == 2 || gqa == 3 || gqa == 4 || gqa == 8;
}

bool xqa_decode_enabled_by_env() {
    const char* v = std::getenv("PIE_CUDA_XQA_DECODE");
    if (v == nullptr || v[0] == '\0') return true;
    return v[0] != '0';
}

int qwen35_small_spec_graph_tokens() {
    static const int tokens = [] {
        const char* v = std::getenv("PIE_QWEN35_SPEC_VERIFY_GRAPH_N");
        if (v == nullptr || v[0] == '\0') return 17;
        return std::clamp(std::atoi(v), 0, 64);
    }();
    return tokens;
}

bool qwen35_forward_profile_enabled() {
    static const bool enabled = [] {
        const char* v = std::getenv("PIE_QWEN35_FORWARD_PROFILE");
        return v != nullptr && v[0] != '\0' && v[0] != '0';
    }();
    return enabled;
}

int configured_mtp_num_drafts(const pie_cuda_driver::Config& cfg) {
    static const int forced = [] {
        const char* v = std::getenv("PIE_MTP_DRAFT_TOKENS");
        if (v == nullptr || v[0] == '\0') return -1;
        return std::clamp(std::atoi(v), 0, 32);
    }();
    if (forced >= 0) return forced;
    return cfg.model.mtp_num_drafts;
}

int qwen35_mtp_draft_position_offset() {
    static const int offset = [] {
        const char* v = std::getenv("PIE_QWEN35_MTP_POSITION_OFFSET");
        if (v == nullptr || v[0] == '\0') return 0;
        return std::clamp(std::atoi(v), 0, 2);
    }();
    return offset;
}

bool qwen35_mtp_fused_gemv_enabled() {
    static const bool enabled = [] {
        const char* v = std::getenv("PIE_QWEN35_MTP_FUSED_GEMV");
        if (v == nullptr || v[0] == '\0') return false;
        return v[0] != '0';
    }();
    return enabled;
}

bool qwen35_mtp_prefix_global_cache() {
    static const bool enabled = [] {
        const char* v = std::getenv("PIE_QWEN35_MTP_PREFIX_GLOBAL");
        if (v == nullptr || v[0] == '\0') return true;
        return v[0] != '0';
    }();
    return enabled;
}

bool has_non_full_attention_layers(const pie_cuda_driver::HfConfig& hf) {
    return std::any_of(
        hf.layer_types.begin(),
        hf.layer_types.end(),
        [](const std::string& t) { return t != "full_attention"; });
}

std::size_t attention_float_workspace_bytes(
    const pie_cuda_driver::HfConfig& hf,
    const pie_cuda_driver::Config& cfg,
    const cudaDeviceProp&,
    int max_requests)
{
    const bool qwen_hybrid =
        hf.model_type == "qwen3_5" ||
        hf.model_type == "qwen3_5_text" ||
        hf.model_type == "qwen3_5_moe" ||
        hf.model_type == "qwen3_5_moe_text";
    const std::size_t base =
        qwen_hybrid ? 128ull * 1024 * 1024 : 80ull * 1024 * 1024;
    const int tp_size = std::max(1, cfg.distributed.tp_size);
    if (tp_size != 1 || max_requests <= 0) {
        return base;
    }
    if (hf.num_key_value_heads <= 0 ||
        hf.num_attention_heads % hf.num_key_value_heads != 0) {
        return base;
    }
    const int gqa = hf.num_attention_heads / hf.num_key_value_heads;
    const bool gqa_in_decode_set = flashinfer_decode_supports_gqa(gqa);
    const bool supported_head_dim =
        hf.head_dim_kernel == 64 || hf.head_dim_kernel == 128 ||
        hf.head_dim_kernel == 256 || hf.head_dim_kernel == 512;
    if (!gqa_in_decode_set || !supported_head_dim || hf.sliding_window >= 0 ||
        has_non_full_attention_layers(hf)) {
        return base;
    }

    const std::size_t q_heads =
        static_cast<std::size_t>(hf.num_attention_heads / tp_size);
    const std::size_t head_dim = static_cast<std::size_t>(hf.head_dim_kernel);
    const std::size_t cta_tile_q = 16;
    const std::size_t padded_batch =
        align_up(static_cast<std::size_t>(max_requests) * 2, 128);
    const std::size_t tmp_v =
        q_heads * padded_batch * cta_tile_q * head_dim * sizeof(float);
    const std::size_t tmp_s =
        q_heads * padded_batch * cta_tile_q * sizeof(float);
    const std::size_t planned = tmp_v + tmp_s + 16ull * 1024 * 1024;
    return std::max(base, align_up(planned, 16ull * 1024 * 1024));
}

std::size_t kv_cache_device_bytes_per_page(
    const pie_cuda_driver::KvCacheFormat& format,
    int page_size,
    int num_kv_heads,
    int head_dim)
{
    std::size_t bytes =
        format.total_bytes_per_page(page_size, num_kv_heads, head_dim);
    if (!format.is_native_bf16()) {
        bytes += 2 * static_cast<std::size_t>(page_size) *
                 static_cast<std::size_t>(num_kv_heads) *
                 static_cast<std::size_t>(head_dim) *
                 pie_cuda_driver::dtype_bytes(pie_cuda_driver::DType::BF16);
    }
    return bytes;
}

std::size_t kv_page_bytes_homogeneous(const pie_cuda_driver::HfConfig& cfg,
                                      int tp_size,
                                      const pie_cuda_driver::KvCacheFormat& format) {
    const int kv_heads = cfg.num_key_value_heads / std::max(1, tp_size);
    return static_cast<std::size_t>(cfg.num_hidden_layers) *
           kv_cache_device_bytes_per_page(
               format, 1, kv_heads, cfg.head_dim_kernel);
}

std::size_t kv_page_bytes_per_layer(
    const pie_cuda_driver::HfConfig& cfg,
    const std::vector<int>& per_layer_head_dim,
    const std::vector<int>& kv_source_layer,
    int tp_size,
    const pie_cuda_driver::KvCacheFormat& format) {
    std::size_t per_token = 0;
    const int kv_heads = cfg.num_key_value_heads / std::max(1, tp_size);
    for (int i = 0; i < cfg.num_hidden_layers; ++i) {
        const bool is_source = kv_source_layer.empty() || kv_source_layer[i] == i;
        if (!is_source) continue;
        const int hd = per_layer_head_dim.empty()
            ? cfg.head_dim_kernel
            : per_layer_head_dim[i];
        per_token += kv_cache_device_bytes_per_page(format, 1, kv_heads, hd);
    }
    return per_token;
}

bool is_auto_memory_profile(const std::string& profile) {
    return profile == "auto";
}

std::vector<std::string> planner_policy_profiles(const std::string& profile) {
    if (!is_auto_memory_profile(profile)) {
        return {profile};
    }
    // `auto` is not a fifth concrete layout. It evaluates the concrete
    // policy families and chooses by the unified objective below.
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

int derive_kv_page_size(const pie_cuda_driver::Config& cfg,
                        const pie_cuda_driver::HfConfig& /*hf*/,
                        const cudaDeviceProp& /*prop*/) {
    return derive_kv_page_size_for_profile(
        is_auto_memory_profile(cfg.batching.memory_profile)
            ? "throughput"
            : cfg.batching.memory_profile,
        std::max(1, cfg.distributed.tp_size));
}

std::vector<int> derive_kv_page_size_candidates(
    const pie_cuda_driver::Config& cfg,
    const pie_cuda_driver::HfConfig& /*hf*/,
    const cudaDeviceProp& /*prop*/) {
    if (const char* forced = std::getenv("PIE_CUDA_KV_PAGE_SIZE")) {
        const int v = std::atoi(forced);
        if (v > 0) {
            return {v};
        }
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

int forced_prefill_tokens() {
    static const int tokens = [] {
        const char* v = std::getenv("PIE_CUDA_PREFILL_TOKENS");
        if (v == nullptr || v[0] == '\0') return 0;
        return std::max(0, std::atoi(v));
    }();
    return tokens;
}

std::filesystem::path cuda_planner_profile_path() {
    if (const char* home = std::getenv("HOME")) {
        if (home[0] != '\0') {
            return std::filesystem::path(home) / ".cache" / "pie" /
                   "cuda_memory_profiles.json";
        }
    }
    return {};
}

bool json_required_eq(const nlohmann::json& key,
                      const char* name,
                      const std::string& expected) {
    auto it = key.find(name);
    return it != key.end() && it->is_string() &&
           it->get<std::string>() == expected;
}

bool json_required_eq(const nlohmann::json& key,
                      const char* name,
                      int expected) {
    auto it = key.find(name);
    return it != key.end() && it->is_number_integer() &&
           it->get<int>() == expected;
}

bool planner_profile_key_matches(const nlohmann::json& key,
                                 const cudaDeviceProp& prop,
                                 const pie_cuda_driver::HfConfig& hf,
                                 int tp_size,
                                 const pie_cuda_driver::KvCacheFormat& kv_format) {
    const auto kv_it = key.find("kv_cache_dtype");
    const bool kv_matches =
        (kv_it != key.end() && kv_it->is_string())
            ? kv_it->get<std::string>() == kv_format.name
            : kv_format.is_native_bf16();
    return json_required_eq(key, "gpu_name", std::string(prop.name)) &&
           json_required_eq(key, "compute_major", prop.major) &&
           json_required_eq(key, "compute_minor", prop.minor) &&
           json_required_eq(key, "sm_count", prop.multiProcessorCount) &&
           kv_matches &&
           json_required_eq(key, "tp_size", tp_size) &&
           json_required_eq(key, "model_type", hf.model_type) &&
           json_required_eq(key, "hidden_size", hf.hidden_size) &&
           json_required_eq(key, "num_hidden_layers", hf.num_hidden_layers) &&
           json_required_eq(key, "num_attention_heads", hf.num_attention_heads) &&
           json_required_eq(key, "num_key_value_heads", hf.num_key_value_heads) &&
           json_required_eq(key, "head_dim", hf.head_dim_kernel);
}

CudaMemoryPlan plan_cuda_memory(
    const pie_cuda_driver::Config& cfg,
    const pie_cuda_driver::HfConfig& hf,
    int max_intermediate,
    int max_Hq,
    int max_Hk,
    bool is_gemma4_arch,
    const std::vector<int>& gemma4_per_layer_head_dim,
    const std::vector<int>& gemma4_kv_source_layer,
    bool is_qwen3_5_arch,
    bool is_qwen3_5_moe_arch,
    int qwen3_5_linear_layers,
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
    const std::vector<int> kv_page_sizes =
        derive_kv_page_size_candidates(cfg, hf, prop);

    const std::size_t per_kv_token_bytes =
        is_gemma4_arch
            ? kv_page_bytes_per_layer(hf, gemma4_per_layer_head_dim,
                                      gemma4_kv_source_layer, tp_size,
                                      kv_format)
            : kv_page_bytes_homogeneous(hf, tp_size, kv_format);
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
    const int forced_prefill = forced_prefill_tokens();
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

    const bool has_linear_state = is_qwen3_5_arch || is_qwen3_5_moe_arch;
    const std::size_t K_dim =
        static_cast<std::size_t>(
            std::max(0, hf.linear_num_key_heads / tp_size)) *
        std::max(0, hf.linear_key_head_dim);
    const std::size_t V_dim =
        static_cast<std::size_t>(
            std::max(0, hf.linear_num_value_heads / tp_size)) *
        std::max(0, hf.linear_value_head_dim);
    const std::size_t conv_dim = 2 * K_dim + V_dim;
    const std::size_t per_slot_recurrent =
        static_cast<std::size_t>(
            std::max(0, hf.linear_num_value_heads / tp_size)) *
        std::max(0, hf.linear_key_head_dim) *
        std::max(0, hf.linear_value_head_dim) * sizeof(float);
    const std::size_t per_slot_conv =
        static_cast<std::size_t>(std::max(0, hf.linear_conv_kernel_dim)) *
        conv_dim * sizeof(std::uint16_t);
    const std::size_t state_slot_bytes = has_linear_state
        ? static_cast<std::size_t>(std::max(0, qwen3_5_linear_layers)) *
          (per_slot_recurrent + per_slot_conv)
        : 0;

    struct Candidate {
        CudaMemoryPlan plan;
        std::string policy_profile;
        int decode_target = 0;
        int prefill_target = 0;
        double score = -std::numeric_limits<double>::infinity();
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
    uniq_clip_desc(Ns, prefill_cap);
    uniq_clip_desc(Rs, 4096);
    for (int kv_page_size : kv_page_sizes) {
        const std::size_t per_page_bytes =
            per_kv_token_bytes * static_cast<std::size_t>(kv_page_size);
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
            std::size_t arena = 0;
            arena += workspace_bytes(
                hf, N, output_rows, max_intermediate, max_Hq, max_Hk);
            if (is_qwen3_5_arch || is_qwen3_5_moe_arch) {
                arena += qwen3_5_la_workspace_bytes(hf, N, tp_size);
            }
            if (is_qwen3_5_moe_arch) {
                arena += qwen3_5_moe_workspace_bytes(hf, N, tp_size);
            }
            if (is_gemma4_arch && hf.gemma4_enable_moe) {
                arena += gemma4_moe_workspace_bytes(hf, N);
            }
            const std::size_t attn_float_bytes =
                attention_float_workspace_bytes(hf, cfg, prop, R0);
            arena += attn_float_bytes;     // AttentionWorkspace float section
            arena += 8ull * 1024 * 1024;  // AttentionWorkspace int section
            arena += persistent_input_bytes(N, R0, max_page_refs,
                                            max_custom_mask_bytes);
            auto quant_scratch_spec = runtime_quant_scratch_base;
            quant_scratch_spec.max_tokens = static_cast<std::size_t>(N);
            const std::size_t runtime_quant_scratch_bytes =
                pie_cuda_driver::ops::runtime_quant_scratch_bytes(
                    quant_scratch_spec);
            arena += runtime_quant_scratch_bytes;
            arena = align_up(arena, 2ull * 1024 * 1024);
            if (arena >= budget) continue;

            int R = R0;
            int state_slots = 0;
            std::size_t state_bytes = 0;
            if (state_slot_bytes > 0) {
                const std::size_t affordable = (budget - arena) / state_slot_bytes;
                state_slots = static_cast<int>(
                    std::min<std::size_t>(static_cast<std::size_t>(R), affordable));
                if (state_slots <= 0) continue;
                R = std::min(R, state_slots);
                state_bytes = static_cast<std::size_t>(state_slots) * state_slot_bytes;
            }
            if (arena + state_bytes >= budget) continue;
            const std::size_t remaining = budget - arena - state_bytes;
            const int kv_pages = static_cast<int>(remaining / per_page_bytes);
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
                32768,
                static_cast<std::size_t>(
                    std::ceil(static_cast<double>(R) * min_kv_horizon)));
            if (kv_tokens < min_kv_tokens) continue;

            CudaMemoryPlan p;
            p.kv_page_size = kv_page_size;
            p.max_workspace_tokens = N;
            p.max_requests = R;
            p.max_page_refs = std::max(262144, R * 512);
            p.kv_pages = kv_pages;
            p.state_slots = state_slots;
            p.attn_float_workspace_bytes = attn_float_bytes;
            p.runtime_quant_scratch_bytes = runtime_quant_scratch_bytes;
            p.arena_bytes = arena;
            p.kv_bytes = static_cast<std::size_t>(kv_pages) * per_page_bytes;
            p.state_bytes = state_bytes;
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
                static_cast<double>(arena + state_bytes) /
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
            if (is_qwen3_5_moe_arch && tp_size > 1 &&
                (auto_profile || profile == "latency") &&
                forced_prefill == 0) {
                // Qwen3.6-MoE TP2 is decode-heavy but still suffers when
                // the prompt wave is split into 1k-token chunks. The measured
                // knee on L40 is N=2048: R128/N1024 loses throughput, while
                // R128/N2048 matches the older R64/N2048 path.
                score += (N >= 2048) ? 1.5 : -1.5;
                score -= std::abs(log2_ratio(N, 2048)) * 4.0;
            }
            if (forced_prefill > 0) {
                score += (N == forced_prefill) ? 1000.0 : -1000.0;
            }
            candidates.push_back(Candidate{
                p,
                policy_profile,
                score_decode_target,
                score_prefill_target,
                score});
            }
        }
    }
    }

    if (candidates.empty()) {
        throw std::runtime_error(
            "cuda memory planner: no viable forward/KV layout fits budget " +
            std::to_string(budget / (1024 * 1024)) + " MiB");
    }

    bool selected_from_profile = false;
    auto best_it = candidates.end();
    if (auto_profile) {
        const auto path = cuda_planner_profile_path();
        if (!path.empty() && std::filesystem::exists(path)) {
            try {
                std::ifstream in(path);
                nlohmann::json root = nlohmann::json::parse(in);
                const auto* entries =
                    root.contains("entries") && root["entries"].is_array()
                        ? &root["entries"]
                        : nullptr;
                if (entries != nullptr) {
                    for (const auto& entry : *entries) {
                        if (!entry.contains("key") || !entry.contains("plan")) {
                            continue;
                        }
                        if (!planner_profile_key_matches(
                                entry["key"], prop, hf, tp_size,
                                kv_format)) {
                            continue;
                        }
                        const auto& plan = entry["plan"];
                        const std::string prof =
                            plan.value("policy_profile", std::string{});
                        const int page_size =
                            plan.value("kv_page_size", 0);
                        const int tokens =
                            plan.value("max_forward_tokens", 0);
                        const int requests =
                            plan.value("max_forward_requests", 0);
                        for (auto it = candidates.begin();
                             it != candidates.end(); ++it) {
                            if (!prof.empty() && it->policy_profile != prof) {
                                continue;
                            }
                            if (page_size > 0 &&
                                it->plan.kv_page_size != page_size) {
                                continue;
                            }
                            if (tokens > 0 &&
                                it->plan.max_workspace_tokens != tokens) {
                                continue;
                            }
                            if (requests > 0 &&
                                it->plan.max_requests != requests) {
                                continue;
                            }
                            best_it = it;
                            selected_from_profile = true;
                            break;
                        }
                        if (selected_from_profile) break;
                    }
                }
            } catch (const std::exception& e) {
                if (verbose) {
                    std::cerr << "[pie-driver-cuda] memory planner: ignored "
                              << "profile cache "
                              << path.string() << ": " << e.what() << "\n";
                }
            }
        }
    }
    if (best_it == candidates.end()) {
        if (prefer_qwen3_8b_prefill_shape) {
            auto preferred_it = candidates.end();
            for (auto it = candidates.begin(); it != candidates.end(); ++it) {
                if (it->plan.max_workspace_tokens != prefill_cap ||
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
                  << " arena=" << (p.arena_bytes / (1024 * 1024)) << " MiB"
                  << " rq_scratch="
                  << (p.runtime_quant_scratch_bytes / (1024 * 1024))
                  << " MiB"
                  << " kv_pages=" << p.kv_pages
                  << " kv_tokens="
                  << (static_cast<std::size_t>(p.kv_pages) *
                      p.kv_page_size)
                  << " state_slots=" << p.state_slots
                  << "\n";
    }
    return best_plan;
}

}  // namespace

namespace {

// Run a one-shot forward pass on a binary file of i32 token ids and dump
// the last token's logits (bf16, [vocab]) to `logits_out`. Used by the
// numeric-parity harness; never invoked through the shmem path.
int run_parity(const pie_cuda_driver::Config& cfg,
               const std::string& tokens_in,
               const std::string& logits_out,
               bool paged,
               bool decode_after_prefill = false,
               pie_cuda_driver::NcclComm* tp_comm = nullptr)
{
    auto engine = pie_cuda_driver::LoadedModel::load(cfg, tp_comm);
    const auto& mt_for_parity = engine.hf_config().model_type;
    const bool is_gpt_oss  = (mt_for_parity == "gpt_oss");
    const bool is_gemma3n  = (mt_for_parity == "gemma3n" || mt_for_parity == "gemma3n_text");
    const bool is_qwen3_5  = (mt_for_parity == "qwen3_5" || mt_for_parity == "qwen3_5_text");
    const bool is_qwen3_5_moe = (mt_for_parity == "qwen3_5_moe" || mt_for_parity == "qwen3_5_moe_text"
                                 || mt_for_parity == "qwen3_moe");
    {
        const bool supported =
            mt_for_parity == "qwen3" || is_qwen3_5 || is_qwen3_5_moe
         || mt_for_parity == "qwen2"
         || mt_for_parity == "llama" || mt_for_parity == "llama3"
         || mt_for_parity == "mistral" || mt_for_parity == "mistral3"
         || is_gpt_oss
         || is_gemma3n;
        if (!supported) {
            std::cerr << "[parity] unsupported model_type: " << mt_for_parity << "\n";
            return 2;
        }
        if ((is_gpt_oss || is_gemma3n || is_qwen3_5 || is_qwen3_5_moe) && !paged) {
            std::cerr << "[parity] " << mt_for_parity << " requires --parity-paged\n";
            return 2;
        }
    }
    pie_cuda_driver::model::Qwen3Weights weights;
    pie_cuda_driver::model::MixtralWeights weights_mixtral;
    pie_cuda_driver::model::Gemma3nWeights weights_gemma3n;
    pie_cuda_driver::model::Qwen3_5Weights weights_qwen3_5;
    pie_cuda_driver::model::Qwen3_5MoeWeights weights_qwen3_5_moe;
    if (is_gpt_oss) {
        weights_mixtral = pie_cuda_driver::model::bind_gpt_oss(engine);
    } else if (is_gemma3n) {
        weights_gemma3n = pie_cuda_driver::model::bind_gemma3n(engine);
    } else if (is_qwen3_5) {
        weights_qwen3_5 = pie_cuda_driver::model::bind_qwen3_5(engine);
    } else if (is_qwen3_5_moe) {
        weights_qwen3_5_moe = pie_cuda_driver::model::bind_qwen3_5_moe(engine);
    } else {
        weights = pie_cuda_driver::model::bind_llama_like(
            engine, /*drop_fused_originals=*/false);
    }

    // Read tokens from disk.
    std::vector<std::int32_t> host_tokens;
    {
        std::ifstream in(tokens_in, std::ios::binary);
        if (!in) { std::cerr << "cannot open " << tokens_in << "\n"; return 3; }
        in.seekg(0, std::ios::end);
        const auto bytes = in.tellg();
        in.seekg(0, std::ios::beg);
        if (bytes <= 0 || bytes % 4 != 0) {
            std::cerr << "[parity] " << tokens_in << " is not a multiple of 4 bytes\n";
            return 3;
        }
        host_tokens.resize(static_cast<std::size_t>(bytes) / 4);
        in.read(reinterpret_cast<char*>(host_tokens.data()), bytes);
    }
    const int N = static_cast<int>(host_tokens.size());
    std::cerr << "[parity] running forward on " << N << " tokens\n";

    std::vector<std::int32_t> host_positions(N);
    for (int i = 0; i < N; ++i) host_positions[i] = i;

    // Upload to device.
    std::int32_t* d_tokens = nullptr;
    std::int32_t* d_positions = nullptr;
    CUDA_CHECK(cudaMalloc(&d_tokens, sizeof(std::int32_t) * N));
    CUDA_CHECK(cudaMalloc(&d_positions, sizeof(std::int32_t) * N));
    CUDA_CHECK(cudaMemcpy(d_tokens, host_tokens.data(), sizeof(std::int32_t) * N,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_positions, host_positions.data(), sizeof(std::int32_t) * N,
                          cudaMemcpyHostToDevice));

    auto ws = pie_cuda_driver::model::Qwen3Workspace::allocate(engine.hf_config(), N);
    pie_cuda_driver::ops::CublasHandle cublas;

    // For `decode_after_prefill`, the row index in `ws.logits` we want to
    // dump at the end is the LAST row written by the *last* call. Default
    // (single-prefill) is N-1; decode mode overwrites only row 0 in the
    // second call, so the dump position becomes 0.
    int dump_row = N - 1;

    if (paged) {
        // Build a single-request paged layout that mirrors what the runtime
        // would send for a fresh request: pages [0..ceil(N/page_size)],
        // last_page_len computed accordingly.
        //
        // `decode_after_prefill` mode: instead of one prefill of N tokens,
        // do (a) prefill of the first N-1 tokens followed by (b) a single
        // decode-shaped step (qo_len=1) at position N-1. The dumped
        // logits come from step (b) — they should match HF's logits at
        // position N-1, just produced via the decode kernel + cached KV
        // read instead of a fresh prefill. Catches decode-only bugs that
        // multi-step prefill parity can't see.
        if (decode_after_prefill && N < 2) {
            std::cerr << "[parity] --parity-decode-after-prefill requires "
                         "at least 2 tokens; got " << N << "\n";
            return 5;
        }
        const int prefill_N  = decode_after_prefill ? (N - 1) : N;
        int parity_dev_id = 0;
        CUDA_CHECK(cudaGetDevice(&parity_dev_id));
        cudaDeviceProp parity_prop{};
        CUDA_CHECK(cudaGetDeviceProperties(&parity_prop, parity_dev_id));
        const int page_size = derive_kv_page_size(
            cfg, engine.hf_config(), parity_prop);
        const int total_pages = (N + page_size - 1) / page_size;

        auto cache = pie_cuda_driver::KvCache::allocate(
            engine.hf_config().num_hidden_layers,
            std::max(total_pages, 1),
            page_size,
            engine.hf_config().num_key_value_heads,
            engine.hf_config().head_dim_kernel,
            pie_cuda_driver::kv_cache_format_from_string(
                cfg.batching.kv_cache_dtype, cfg.model.dtype));

        auto parity_attn_ws = pie_cuda_driver::AttentionWorkspace::allocate();

        // qwen3_5 / qwen3_5_moe need their own scratch + rs_cache storage.
        pie_cuda_driver::model::Qwen3_5LinearAttnWorkspace q35_la_ws;
        pie_cuda_driver::Qwen3_5StateCache q35_state_cache;
        pie_cuda_driver::model::Qwen3_5MoeMlpWorkspace q35_moe_ws;
        if (is_qwen3_5 || is_qwen3_5_moe) {
            const auto& cfg_q = engine.hf_config();
            const int K_dim = cfg_q.linear_num_key_heads * cfg_q.linear_key_head_dim;
            const int V_dim = cfg_q.linear_num_value_heads * cfg_q.linear_value_head_dim;
            const int conv_dim = 2 * K_dim + V_dim;
            q35_la_ws = pie_cuda_driver::model::Qwen3_5LinearAttnWorkspace::allocate(
                N, conv_dim, cfg_q.linear_num_value_heads,
                cfg_q.linear_num_key_heads,
                cfg_q.linear_key_head_dim, cfg_q.linear_value_head_dim,
                /*hq=*/cfg_q.num_attention_heads * cfg_q.head_dim);
            const std::size_t num_layers = is_qwen3_5
                ? weights_qwen3_5.layers.size()
                : weights_qwen3_5_moe.layers.size();
            std::vector<bool> layer_is_linear(num_layers);
            for (std::size_t L = 0; L < num_layers; ++L) {
                const bool is_linear = is_qwen3_5
                    ? (weights_qwen3_5.layers[L].kind ==
                       pie_cuda_driver::model::Qwen3_5LayerWeights::Kind::LinearAttn)
                    : (weights_qwen3_5_moe.layers[L].kind ==
                       pie_cuda_driver::model::Qwen3_5MoeLayerWeights::Kind::LinearAttn);
                layer_is_linear[L] = is_linear;
            }
            q35_state_cache = pie_cuda_driver::Qwen3_5StateCache::allocate(
                layer_is_linear, conv_dim, cfg_q.linear_conv_kernel_dim,
                cfg_q.linear_num_value_heads,
                cfg_q.linear_key_head_dim, cfg_q.linear_value_head_dim,
                cfg_q.hidden_size);
            if (is_qwen3_5_moe) {
                q35_moe_ws = pie_cuda_driver::model::Qwen3_5MoeMlpWorkspace::allocate(
                    N, cfg_q.hidden_size,
                    cfg_q.num_experts, cfg_q.num_experts_per_tok,
                    cfg_q.moe_intermediate_size,
                    cfg_q.shared_expert_intermediate_size);
            }
        }

        // Build the per-arch fwd_cfg once; reused across the prefill and
        // (optional) decode calls.
        pie_cuda_driver::model::LlamaLikeForwardCfg fwd_cfg{};
        if (is_gpt_oss) {
            const auto& hf = engine.hf_config();
            fwd_cfg.use_qkv_bias = hf.attention_bias;
            apply_rope_config(fwd_cfg, hf);
            fwd_cfg.sliding_window             = hf.sliding_window;
            for (const auto& t : hf.layer_types) {
                fwd_cfg.per_layer_window_left.push_back(
                    (t == "sliding_attention") ? hf.sliding_window : -1);
            }
            // Decode-after-prefill mode exercises both the prefill and
            // decode kernels in sequence. force_prefill_path would defeat
            // the purpose of the test, so leave it false.
            fwd_cfg.force_prefill_path = !decode_after_prefill;
        }

        // Helper to run one paged forward call. `total_n` is qo_len, `kv_n`
        // is the post-write KV length (for the indptr/last_page_len math).
        // `tok_d` / `pos_d` are device pointers to the inputs for this call.
        auto run_call = [&](const std::int32_t* tok_d,
                            const std::int32_t* pos_d,
                            int total_n, int kv_n, bool is_decode) {
            const int n_pages_kv = (kv_n + page_size - 1) / page_size;
            std::vector<std::uint32_t> h_qo  = {0u, (std::uint32_t)total_n};
            std::vector<std::uint32_t> h_pp  = {0u, (std::uint32_t)n_pages_kv};
            std::vector<std::uint32_t> h_pi(n_pages_kv);
            for (int i = 0; i < n_pages_kv; ++i) h_pi[i] = (std::uint32_t)i;
            std::vector<std::uint32_t> h_lpl = {
                (std::uint32_t)(((kv_n - 1) % page_size) + 1)
            };

            std::uint32_t *d_qo, *d_pi, *d_pp, *d_lpl;
            CUDA_CHECK(cudaMalloc(&d_qo,  4 * h_qo.size()));
            CUDA_CHECK(cudaMalloc(&d_pi,  4 * h_pi.size()));
            CUDA_CHECK(cudaMalloc(&d_pp,  4 * h_pp.size()));
            CUDA_CHECK(cudaMalloc(&d_lpl, 4 * h_lpl.size()));
            CUDA_CHECK(cudaMemcpy(d_qo,  h_qo.data(),  4 * h_qo.size(),  cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_pi,  h_pi.data(),  4 * h_pi.size(),  cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_pp,  h_pp.data(),  4 * h_pp.size(),  cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_lpl, h_lpl.data(), 4 * h_lpl.size(), cudaMemcpyHostToDevice));

            if (is_gpt_oss) {
                const auto& hf = engine.hf_config();
                pie_cuda_driver::model::mixtral_forward_paged(
                    weights_mixtral, engine.hf_config(), fwd_cfg,
                    hf.num_experts, hf.num_experts_per_tok,
                    ws, cache, parity_attn_ws, cublas,
                    tok_d, pos_d,
                    d_qo, d_pi, d_pp, d_lpl,
                    /*qo_indptr_h=*/h_qo.data(),
                    /*kv_page_indptr_h=*/h_pp.data(),
                    /*total_tokens=*/total_n, /*num_requests=*/1,
                    /*is_pure_decode=*/is_decode);
            } else if (is_gemma3n) {
                pie_cuda_driver::model::Gemma3nForwardCfg gemma3n_fwd{};
                gemma3n_fwd.final_logit_softcap =
                    engine.hf_config().gemma_final_logit_softcap;
                gemma3n_fwd.force_prefill_path = !decode_after_prefill;
                pie_cuda_driver::model::gemma3n_forward_paged(
                    weights_gemma3n, engine.hf_config(), gemma3n_fwd,
                    ws, cache, parity_attn_ws, cublas,
                    tok_d, pos_d,
                    d_qo, d_pi, d_pp, d_lpl,
                    /*qo_indptr_h=*/h_qo.data(),
                    /*kv_page_indptr_h=*/h_pp.data(),
                    /*total_tokens=*/total_n, /*num_requests=*/1,
                    /*is_pure_decode=*/is_decode);
            } else if (is_qwen3_5) {
                pie_cuda_driver::model::Qwen3_5ForwardCfg q35_fwd{};
                q35_fwd.force_prefill_path = !decode_after_prefill;
                q35_fwd.tp_size = cfg.distributed.tp_size;
                q35_fwd.tp_comm = tp_comm;
                pie_cuda_driver::model::Qwen3_5PlanState q35_plan;
                pie_cuda_driver::model::prepare_qwen3_5_decode_plan(
                    q35_plan, parity_attn_ws, cache, engine.hf_config(),
                    q35_fwd, h_qo.data(), h_pp.data(), h_lpl.data(),
                    /*total_tokens=*/total_n, /*num_requests=*/1,
                    is_decode);
                pie_cuda_driver::model::qwen3_5_forward_paged(
                    weights_qwen3_5, engine.hf_config(), q35_fwd, q35_plan,
                    ws, q35_la_ws, cache, q35_state_cache,
                    parity_attn_ws, cublas,
                    tok_d, pos_d,
                    d_qo, d_pi, d_pp, d_lpl,
                    /*qo_indptr_h=*/h_qo.data(),
                    /*kv_page_indptr_h=*/h_pp.data(),
                    /*total_tokens=*/total_n, /*num_requests=*/1,
                    /*is_pure_decode=*/is_decode,
                    /*mask_d=*/nullptr, /*mask_indptr_d=*/nullptr);
            } else if (is_qwen3_5_moe) {
                pie_cuda_driver::model::Qwen3_5ForwardCfg q35_fwd{};
                q35_fwd.force_prefill_path = !decode_after_prefill;
                q35_fwd.tp_size = cfg.distributed.tp_size;
                q35_fwd.tp_comm = tp_comm;
                pie_cuda_driver::model::Qwen3_5PlanState q35_plan;
                pie_cuda_driver::model::prepare_qwen3_5_decode_plan(
                    q35_plan, parity_attn_ws, cache, engine.hf_config(),
                    q35_fwd, h_qo.data(), h_pp.data(), h_lpl.data(),
                    /*total_tokens=*/total_n, /*num_requests=*/1,
                    is_decode);
                pie_cuda_driver::model::qwen3_5_moe_forward_paged(
                    weights_qwen3_5_moe, engine.hf_config(), q35_fwd, q35_plan,
                    ws, q35_la_ws, q35_moe_ws,
                    cache, q35_state_cache,
                    parity_attn_ws, cublas,
                    tok_d, pos_d,
                    d_qo, d_pi, d_pp, d_lpl,
                    /*qo_indptr_h=*/h_qo.data(),
                    /*kv_page_indptr_h=*/h_pp.data(),
                    /*total_tokens=*/total_n, /*num_requests=*/1,
                    /*is_pure_decode=*/is_decode,
                    /*mask_d=*/nullptr, /*mask_indptr_d=*/nullptr);
            } else {
                pie_cuda_driver::model::qwen3_forward_paged(
                    weights, engine.hf_config(), ws, cache, parity_attn_ws, cublas,
                    tok_d, pos_d,
                    d_qo, d_pi, d_pp, d_lpl,
                    /*qo_indptr_h=*/h_qo.data(),
                    /*kv_page_indptr_h=*/h_pp.data(),
                    /*total_tokens=*/total_n, /*num_requests=*/1,
                    /*is_pure_decode=*/is_decode);
            }

            cudaFree(d_qo); cudaFree(d_pi); cudaFree(d_pp); cudaFree(d_lpl);
        };

        // Prefill on the first prefill_N tokens.
        run_call(d_tokens, d_positions, prefill_N, prefill_N, /*is_decode=*/false);

        if (decode_after_prefill) {
            // Single decode step at position N-1, reading KV [0, N-1) and
            // appending the K/V for position N-1.
            run_call(d_tokens + (N - 1), d_positions + (N - 1),
                     /*total_n=*/1, /*kv_n=*/N, /*is_decode=*/true);
            // The decode call wrote logits for one token at row 0.
            dump_row = 0;
        }

    } else {
        pie_cuda_driver::model::qwen3_forward_prefill(
            weights, engine.hf_config(), ws, cublas, d_tokens, d_positions, N);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // Greedy sample over all rows on the GPU, then echo the last-token
    // id to stderr — the parity harness picks it up and cross-checks
    // against numpy.argmax of the dumped logits.
    // The last call wrote `last_n_rows` rows of logits into ws.logits.
    // For single-prefill: N rows, dump row N-1. For decode-after-prefill:
    // the decode call wrote 1 row at index 0, dump row 0.
    const int last_n_rows = decode_after_prefill ? 1 : N;
    {
        const int V = engine.hf_config().vocab_size;
        std::int32_t* d_sampled = nullptr;
        CUDA_CHECK(cudaMalloc(&d_sampled, sizeof(std::int32_t) * last_n_rows));
        pie_cuda_driver::kernels::launch_argmax_bf16(
            ws.logits.data(), d_sampled, last_n_rows, V, /*stream=*/nullptr);
        std::vector<std::int32_t> host_sampled(last_n_rows);
        CUDA_CHECK(cudaMemcpy(host_sampled.data(), d_sampled,
                              sizeof(std::int32_t) * last_n_rows,
                              cudaMemcpyDeviceToHost));
        cudaFree(d_sampled);
        std::cerr << "[parity] gpu argmax last-token id = "
                  << host_sampled.back() << "\n";
    }

    // Copy last-token logits row out as bf16 (we'll convert in Python).
    const int V = engine.hf_config().vocab_size;
    std::vector<std::uint16_t> host_logits(V);  // bf16 viewed as u16
    const auto* base = static_cast<const std::uint16_t*>(ws.logits.data());
    CUDA_CHECK(cudaMemcpy(host_logits.data(),
                          base + static_cast<std::size_t>(dump_row) * V,
                          V * sizeof(std::uint16_t),
                          cudaMemcpyDeviceToHost));

    {
        std::ofstream out(logits_out, std::ios::binary);
        if (!out) { std::cerr << "cannot open " << logits_out << "\n"; return 4; }
        out.write(reinterpret_cast<const char*>(host_logits.data()),
                  host_logits.size() * 2);
    }
    std::cerr << "[parity] wrote " << V << " bf16 logits to " << logits_out << "\n";

    cudaFree(d_tokens);
    cudaFree(d_positions);
    return 0;
}

}  // namespace

namespace {

// `vtable_opt` is non-null for the in-process serve loop; null for the
// parity-only standalone entry (`pie_driver_cuda_run`), which exits
// after running the parity test and never enters serve_forever.
int run_impl(int argc,
             char** argv,
             int install_signal_handlers,
             pie_driver_cuda_ready_cb ready_cb,
             void* ready_ctx,
             const pie_driver::PieInProcVTable* vtable_opt) {
    if (ready_cb == nullptr) {
        std::cerr << "[pie-driver-cuda] fatal: ready_cb is null\n";
        return -1;
    }
    CLI::App app{"pie_driver_cuda — native CUDA backend for Pie"};
    std::string config_path = "dev.toml";
    app.add_option("-c,--config", config_path, "Path to TOML config")
        ->check(CLI::ExistingFile);

    std::string parity_tokens, parity_out;
    bool parity_paged = false;
    bool parity_decode_after_prefill = false;
    auto* parity = app.add_option_group("parity", "Numeric-parity test entry");
    parity->add_option("--parity-tokens", parity_tokens,
                       "Path to a binary file of i32 token ids");
    parity->add_option("--parity-out", parity_out,
                       "Where to write the last-token logits as bf16 [vocab]");
    parity->add_flag("--parity-paged", parity_paged,
                     "Run the paged forward path (wire-shaped KV layout)");
    parity->add_flag("--parity-decode-after-prefill", parity_decode_after_prefill,
                     "After prefill on the first N-1 tokens, run a single "
                     "qo_len=1 decode step at position N-1 and dump that "
                     "step's logits. Exercises the decode kernel + KV-cache "
                     "read path in addition to prefill. Requires --parity-paged.");

    // Default-on under llama-like. `enable_cuda_graph=true` on the
    // flashinfer DecodePlan side pins plan_info layout (padded_batch_size,
    // request_indices_offset, …) across fires; per-fire DecodePlan calls
    // only update int_buf content (request_indices, block_valid_mask), and
    // device pointers stay stable. See `forward_fn.graph_safe = true` below.
    bool use_cuda_graphs = true;
    app.add_flag("--cuda-graphs,!--no-cuda-graphs", use_cuda_graphs,
                 "Capture decode forward into CUDA graphs and replay per "
                 "shape bucket. Default on for cuda_native.");

    // Tensor-parallel knobs. Override [distributed] in the TOML when
    // present so the wrapper can launch ad-hoc TP groups without
    // rewriting the config file. Empty unique-id means "fall back to
    // TOML".
    int cli_tp_size = -1, cli_tp_rank = -1;
    std::string cli_nccl_unique_id_hex;
    app.add_option("--tp-size", cli_tp_size,
                   "Tensor-parallel world size (overrides [distributed].tp_size).");
    app.add_option("--tp-rank", cli_tp_rank,
                   "This process's rank in the TP group (0..tp_size).");
    app.add_option("--nccl-unique-id-hex", cli_nccl_unique_id_hex,
                   "Hex-encoded ncclUniqueId shared across all ranks of "
                   "the TP group. Only required when tp_size > 1.");

    CLI11_PARSE(app, argc, argv);

    auto cfg = pie_cuda_driver::load_config(config_path);
    if (cli_tp_size >= 1) cfg.distributed.tp_size = cli_tp_size;
    if (cli_tp_rank >= 0) cfg.distributed.tp_rank = cli_tp_rank;
    if (!cli_nccl_unique_id_hex.empty())
        cfg.distributed.nccl_unique_id_hex = cli_nccl_unique_id_hex;
    const bool verbose = cfg.runtime.verbose;
    if (cfg.distributed.tp_size > 1 &&
        cfg.distributed.tp_rank > 0 &&
        cfg.distributed.nccl_unique_id_hex.empty()) {
        std::cerr << "[pie-driver-cuda] rank " << cfg.distributed.tp_rank
                  << " requires --nccl-unique-id-hex "
                  << "(or [distributed].nccl_unique_id_hex)\n";
        return 1;
    }

    if (!parity_tokens.empty()) {
        // Parity argument validation up-front so we don't go through
        // NCCL bootstrap only to fail on bad CLI. The actual parity
        // dispatch is deferred until after NCCL init so a TP-mode
        // parity test can drive collectives.
        if (parity_out.empty()) {
            std::cerr << "--parity-tokens requires --parity-out\n";
            return 1;
        }
        if (parity_decode_after_prefill && !parity_paged) {
            std::cerr << "--parity-decode-after-prefill requires --parity-paged\n";
            return 1;
        }
    }

    // Informational logs go to stderr — stdout is reserved for the READY
    // handshake line consumed by the host process.
    if (verbose) {
        std::cerr << "[pie-driver-cuda] config loaded\n"
                  << "  model.snap_dir  = " << cfg.model.snapshot_dir << "\n"
	                  << "  model.device    = " << cfg.model.device << "\n"
	                  << "  model.dtype     = " << cfg.model.dtype << "\n"
	                  << "  model.mxfp4_moe = " << cfg.model.mxfp4_moe << "\n"
	                  << "  tp_size         = " << cfg.distributed.tp_size << "\n"
                  << "  tp_rank         = " << cfg.distributed.tp_rank << "\n";
    }

    // Bind the requested CUDA device before NCCL init — ncclCommInitRank
    // captures whatever is current on the calling thread.
    {
        CUDA_CHECK(cudaSetDevice(
            pie_cuda_driver::parse_cuda_device_id(cfg.model.device)));
    }

    pie_cuda_driver::NcclComm tp_comm;
    if (cfg.distributed.tp_size > 1) {
        // ncclGetUniqueId opens a TCP bootstrap listener inside the
        // calling process — it must outlive the rendezvous. Rank 0
        // generates the id (when no id was passed in), prints it on
        // stdout for the wrapper to relay, then proceeds straight into
        // ncclCommInitRank. Followers receive the id from the wrapper
        // via --nccl-unique-id-hex / [distributed].nccl_unique_id_hex.
        ncclUniqueId uid;
        if (cfg.distributed.tp_rank == 0 &&
            cfg.distributed.nccl_unique_id_hex.empty()) {
            NCCL_CHECK(ncclGetUniqueId(&uid));
            const auto hex = pie_cuda_driver::nccl_unique_id_to_hex(uid);
            std::cout << "NCCL_UID " << hex << std::endl;
        } else {
            uid = pie_cuda_driver::nccl_unique_id_from_hex(
                cfg.distributed.nccl_unique_id_hex);
        }
        tp_comm = pie_cuda_driver::NcclComm(
            cfg.distributed.tp_size, cfg.distributed.tp_rank, uid);
        if (verbose) {
            std::cerr << "[pie-driver-cuda] NCCL comm initialised "
                      << "(world=" << tp_comm.world_size()
                      << ", rank=" << tp_comm.rank() << ")\n";
        }
        tp_startup_cpu_barrier(cfg);

        // Smoke test: every rank contributes (rank+1); sum should be
        // world*(world+1)/2. Catches mis-numbered ranks at startup.
        cudaStream_t s = nullptr;
        CUDA_CHECK(cudaStreamCreate(&s));
        int* d_v = nullptr;
        CUDA_CHECK(cudaMalloc(&d_v, sizeof(int)));
        const int rank1 = cfg.distributed.tp_rank + 1;
        CUDA_CHECK(cudaMemcpyAsync(d_v, &rank1, sizeof(int),
                                   cudaMemcpyHostToDevice, s));
        NCCL_CHECK_ASYNC(ncclAllReduce(d_v, d_v, 1, ncclInt32, ncclSum,
                                       tp_comm.comm(), s),
                         tp_comm.comm());
        int h_v = 0;
        CUDA_CHECK(cudaMemcpyAsync(&h_v, d_v, sizeof(int),
                                   cudaMemcpyDeviceToHost, s));
        CUDA_CHECK(cudaStreamSynchronize(s));
        CUDA_CHECK(cudaFree(d_v));
        CUDA_CHECK(cudaStreamDestroy(s));
        const int W = cfg.distributed.tp_size;
        const int expected = W * (W + 1) / 2;
        if (h_v != expected) {
            std::cerr << "[pie-driver-cuda] NCCL smoke test FAILED: got "
                      << h_v << ", expected " << expected << "\n";
            return 3;
        }
        if (verbose) {
            std::cerr << "[pie-driver-cuda] NCCL smoke test ok ("
                      << h_v << "==" << expected << ")\n";
        }
    }
    pie_cuda_driver::NcclComm* tp_comm_ptr =
        (cfg.distributed.tp_size > 1) ? &tp_comm : nullptr;

    // Parity mode: every rank participates so collectives complete;
    // only rank 0 dumps logits to disk. The harness compares rank 0's
    // output against a single-GPU reference run.
    if (!parity_tokens.empty()) {
        const std::string out_path = (cfg.distributed.tp_rank == 0)
            ? parity_out
            : (parity_out + ".rank" +
               std::to_string(cfg.distributed.tp_rank));
        return run_parity(cfg, parity_tokens, out_path, parity_paged,
                          parity_decode_after_prefill, tp_comm_ptr);
    }

    auto engine = pie_cuda_driver::LoadedModel::load(cfg, tp_comm_ptr);

    {
        const auto& mt = engine.hf_config().model_type;
        // Llama-like family. Same RMSNorm + RoPE + GQA + SwiGLU graph; the
        // only branch is whether per-head q/k_norm exists (Qwen3 quirk),
        // which is captured in HfConfig.use_qk_norm.
        const bool supported =
            mt == "qwen3"
         || mt == "qwen3_5" || mt == "qwen3_5_text"
         || mt == "qwen3_5_moe" || mt == "qwen3_5_moe_text"
         || mt == "qwen3_moe"
         || mt == "qwen2"
         || mt == "llama" || mt == "llama3"
         || mt == "mistral" || mt == "mistral3" || mt == "ministral3"
         || mt == "mixtral"
         || mt == "gpt_oss"
         || mt == "phi3"
         // OLMo-V1 (`mt == "olmo"`) used LayerNorm, not RMSNorm — its
         // schema is genuinely different and was never wired up. OLMo-2
         // and OLMo-3 share the post-norm + q/k-norm + RMSNorm setup
         // that `bind_olmo3` materialises, so we accept both here.
         || mt == "olmo2" || mt == "olmo3"
         || mt == "gemma2"
         || mt == "gemma3" || mt == "gemma3_text"
         || mt == "gemma4" || mt == "gemma4_text"
         || mt == "gemma3n" || mt == "gemma3n_text";
        if (!supported) {
            std::cerr << "[pie-driver-cuda] arch '" << mt
                      << "' not yet supported (Qwen 2/3, Llama-3, "
                      << "Mistral, Mixtral, GPT-OSS, Phi-3, OLMo-3, Gemma-2/3/4)\n";
            return 2;
        }
    }
    // Centralized bound-model selection. The forward setup below keeps local
    // references for now so the rest of the serving path stays unchanged.
    auto bound_model = pie_cuda_driver::model::bind_cuda_model(engine, verbose);
    auto& weights_llama = bound_model.llama;
    auto& weights_gemma = bound_model.gemma;
    auto& weights_gemma4 = bound_model.gemma4;
    auto& weights_gemma3n = bound_model.gemma3n;
    auto& weights_mixtral = bound_model.mixtral;
    auto& weights_qwen3_5 = bound_model.qwen3_5;
    auto& weights_qwen3_5_moe = bound_model.qwen3_5_moe;

    const bool is_gemma_arch = bound_model.is_gemma();
    const bool is_gemma4_arch = bound_model.is_gemma4();
    const bool is_gemma3n_arch = bound_model.is_gemma3n();
    const bool is_mixtral_arch = bound_model.is_mixtral();
    const bool is_qwen3_5_arch = bound_model.is_qwen3_5();
    const bool is_qwen3_5_moe_arch = bound_model.is_qwen3_5_moe();
    const int native_mtp_num_drafts = configured_mtp_num_drafts(cfg);

    const std::size_t num_layers_bound = bound_model.num_layers();
    if (verbose) {
        std::cerr << "[pie-driver-cuda] schema bound: "
                  << num_layers_bound << " layers ("
                  << engine.hf_config().model_type
                  << (engine.hf_config().use_qk_norm ? ", q/k norm" : "")
                  << ")\n";
    }

    std::optional<pie_cuda_driver::model::Gemma4MtpWeights> gemma4_mtp_weights;
    pie_cuda_driver::model::Gemma4MtpRuntimeConfig gemma4_mtp_runtime;
    std::string mtp_snapshot_dir = cfg.model.mtp_assistant_snapshot_dir;
    std::string mtp_snapshot_source = mtp_snapshot_dir.empty() ? "" : "config";
    if (mtp_snapshot_dir.empty()) {
        if (const char* env = std::getenv("PIE_GEMMA4_MTP_SNAPSHOT_DIR")) {
            mtp_snapshot_dir = env;
            mtp_snapshot_source = "env";
        }
    }
    if (is_gemma4_arch && native_mtp_num_drafts > 0 &&
        mtp_snapshot_dir.empty()) {
        if (auto discovered = discover_gemma4_mtp_snapshot_dir(
                std::filesystem::path(cfg.model.snapshot_dir))) {
            mtp_snapshot_dir = discovered->string();
            mtp_snapshot_source = "auto";
            if (verbose && cfg.distributed.tp_rank == 0) {
                std::cerr << "[pie-driver-cuda] Gemma4 MTP assistant "
                          << "auto-discovered: " << mtp_snapshot_dir
                          << "\n";
            }
        }
    }
    if (is_gemma4_arch && native_mtp_num_drafts > 0 &&
        !mtp_snapshot_dir.empty()) {
        if (cfg.distributed.tp_size > 1) {
            if (verbose && cfg.distributed.tp_rank == 0) {
                std::cerr << "[pie-driver-cuda] Gemma4 MTP disabled under "
                          << "tensor parallelism for this build\n";
            }
        } else {
            gemma4_mtp_weights.emplace(
                pie_cuda_driver::model::load_gemma4_mtp_weights(
                    std::filesystem::path(mtp_snapshot_dir),
                    cfg.model.device,
                    engine.hf_config(),
                    weights_gemma4,
                    gemma4_mtp_runtime,
                    verbose));
            if (verbose && cfg.distributed.tp_rank == 0 &&
                !mtp_snapshot_source.empty()) {
                std::cerr << "[pie-driver-cuda] Gemma4 MTP assistant source="
                          << mtp_snapshot_source << "\n";
            }
        }
    } else if (is_gemma4_arch && native_mtp_num_drafts > 0 &&
               verbose && cfg.distributed.tp_rank == 0) {
        std::cerr << "[pie-driver-cuda] Gemma4 MTP system drafter not "
                  << "enabled: assistant checkpoint not found; set "
                  << "mtp_assistant_snapshot_dir or "
                  << "PIE_GEMMA4_MTP_SNAPSHOT_DIR\n";
    }

    // Pre-allocate persistent rs_cache state for serving. CUDA-native no longer
    // accepts manual batch/KV sizing from public config; after weights are
    // resident we plan the forward arena, optional linear-attn rs_cache,
    // and remaining KV pages from gpu_mem_utilization + memory_profile.
    // Per-arch worst-case workspace dims. Gemma-4 has both
    // `use_double_wide_mlp` (intermediate doubles on shared layers)
    // and dual head_dim (sliding=256 vs full=512), so ws.q/k/v need
    // the full-attention sizing. Other archs use the single config
    // values.
    const int local_tp_size = std::max(1, cfg.distributed.tp_size);
    const int local_q_heads =
        engine.hf_config().num_attention_heads / local_tp_size;
    const int local_kv_heads =
        engine.hf_config().num_key_value_heads / local_tp_size;
    int max_mlp_intermediate =
        engine.hf_config().intermediate_size / local_tp_size;
    int max_Hq = local_q_heads * engine.hf_config().head_dim;
    int max_Hk = local_kv_heads * engine.hf_config().head_dim;
    if (is_gemma4_arch) {
        for (int v : weights_gemma4.per_layer_intermediate) {
            // Gemma-4 binds this from the already-loaded projection shape,
            // so it is already per-rank under TP.
            const int local_v = v;
            if (local_v > max_mlp_intermediate) max_mlp_intermediate = local_v;
        }
        for (int d : weights_gemma4.per_layer_head_dim) {
            const int Hq = local_q_heads * d;
            const int Hk = local_kv_heads * d;
            if (Hq > max_Hq) max_Hq = Hq;
            if (Hk > max_Hk) max_Hk = Hk;
        }
    } else if (is_gemma3n_arch) {
        // Per-layer intermediate (HF stores it as a list); head_dim is
        // uniform across layers on gemma3n, so KV cache can use the
        // standard allocator.
        for (int v : weights_gemma3n.per_layer_intermediate) {
            const int local_v = v / local_tp_size;
            if (local_v > max_mlp_intermediate) max_mlp_intermediate = local_v;
        }
    }

    std::vector<bool> qwen3_5_layer_is_linear;
    if (is_qwen3_5_arch || is_qwen3_5_moe_arch) {
        const std::size_t num_layers = is_qwen3_5_arch
            ? weights_qwen3_5.layers.size()
            : weights_qwen3_5_moe.layers.size();
        qwen3_5_layer_is_linear.resize(num_layers);
        for (std::size_t L = 0; L < num_layers; ++L) {
            const bool is_linear = is_qwen3_5_arch
                ? (weights_qwen3_5.layers[L].kind ==
                   pie_cuda_driver::model::Qwen3_5LayerWeights::Kind::LinearAttn)
                : (weights_qwen3_5_moe.layers[L].kind ==
                   pie_cuda_driver::model::Qwen3_5MoeLayerWeights::Kind::LinearAttn);
            qwen3_5_layer_is_linear[L] = is_linear;
        }
    }

    const int qwen3_5_linear_layers = static_cast<int>(std::count(
        qwen3_5_layer_is_linear.begin(), qwen3_5_layer_is_linear.end(), true));
    const auto kv_format = pie_cuda_driver::kv_cache_format_from_string(
        cfg.batching.kv_cache_dtype, cfg.model.dtype);
    const bool graph_capable_forward =
        use_cuda_graphs && bound_model.is_llama_like() &&
        kv_format.is_native_bf16();
    const auto runtime_quant_scratch_base =
        graph_capable_forward
            ? runtime_quant_scratch_spec(engine, /*max_tokens=*/0)
            : pie_cuda_driver::ops::RuntimeQuantScratchSpec{};

    const CudaMemoryPlan mem_plan = plan_cuda_memory(
        cfg, engine.hf_config(), max_mlp_intermediate, max_Hq, max_Hk,
        is_gemma4_arch, weights_gemma4.per_layer_head_dim,
        weights_gemma4.kv_source_layer, is_qwen3_5_arch,
        is_qwen3_5_moe_arch, qwen3_5_linear_layers,
        kv_format, runtime_quant_scratch_base, verbose);
    const int max_workspace_tokens = mem_plan.max_workspace_tokens;
    // `mem_plan.kv_pages` is the runtime-visible KV capacity. CUDA graph
    // padding needs one isolated page for synthetic rows when replaying a
    // bucket larger than the real request count; charge that implementation
    // detail to the planner's safety headroom instead of reducing the
    // advertised runtime pool.
    const int runtime_kv_pages = mem_plan.kv_pages;
    const int physical_kv_pages =
        mem_plan.kv_pages > 0 ? mem_plan.kv_pages + 1 : mem_plan.kv_pages;
    const int graph_pad_page =
        mem_plan.kv_pages > 0 ? runtime_kv_pages : -1;

    auto ws = pie_cuda_driver::model::Qwen3Workspace::allocate_full(
        engine.hf_config(), max_workspace_tokens,
        max_mlp_intermediate, max_Hq, max_Hk,
        mem_plan.capacity.max_logit_rows);

    auto kv_cache =
        is_gemma4_arch
            ? pie_cuda_driver::KvCache::allocate_per_layer(
                  engine.hf_config().num_hidden_layers,
                  physical_kv_pages,
                  mem_plan.kv_page_size,
                  local_kv_heads,
                  weights_gemma4.per_layer_head_dim,
                  weights_gemma4.kv_source_layer,
                  weights_gemma4.per_layer_num_kv_heads,
                  kv_format)
            : pie_cuda_driver::KvCache::allocate(
                  engine.hf_config().num_hidden_layers,
                  physical_kv_pages,
                  mem_plan.kv_page_size,
                  local_kv_heads,
                  engine.hf_config().head_dim_kernel,
                  kv_format);

    auto attn_ws = pie_cuda_driver::AttentionWorkspace::allocate(
        mem_plan.attn_float_workspace_bytes, 8ull * 1024 * 1024);

    // Plan-state holders used by the prepare/body split for graph-friendly
    // dispatch. Allocated unconditionally — empty on archs that don't use
    // them. `qwen3_5_plan_state` is shared between qwen3_5 and qwen3_5_moe
    // (they share `prepare_qwen3_5_decode_plan`).
    pie_cuda_driver::model::Qwen3_5PlanState qwen3_5_plan_state;

    // Qwen3.5 / Qwen3.6-MoE linear-attention extras: per-layer rs_cache
    // + a per-call workspace. Inert (default-constructed) on every other
    // arch. The MoE arch additionally needs a routed-experts workspace.
    pie_cuda_driver::model::Qwen3_5LinearAttnWorkspace qwen3_5_la_ws;
    pie_cuda_driver::Qwen3_5StateCache qwen3_5_state_cache;
    pie_cuda_driver::model::Qwen3_5MoeMlpWorkspace qwen3_5_moe_ws;
    int qwen3_5_runtime_rs_slots = 0;
    int qwen3_5_scratch_rs_slot = -1;
    if (is_qwen3_5_arch || is_qwen3_5_moe_arch) {
        const auto& cfg_q = engine.hf_config();
        const int q35_tp_size = std::max(1, cfg.distributed.tp_size);
        const int local_linear_key_heads =
            cfg_q.linear_num_key_heads / q35_tp_size;
        const int local_linear_value_heads =
            cfg_q.linear_num_value_heads / q35_tp_size;
        const int K_dim = local_linear_key_heads * cfg_q.linear_key_head_dim;
        const int V_dim = local_linear_value_heads * cfg_q.linear_value_head_dim;
        const int conv_dim = 2 * K_dim + V_dim;
        qwen3_5_la_ws = pie_cuda_driver::model::Qwen3_5LinearAttnWorkspace::allocate(
            max_workspace_tokens, conv_dim,
            local_linear_value_heads,
            local_linear_key_heads,
            cfg_q.linear_key_head_dim,
            cfg_q.linear_value_head_dim,
            /*hq=*/(cfg_q.num_attention_heads / q35_tp_size) *
                cfg_q.head_dim);
        // Allocate per-slot state for the linear-attn layers. The memory
        // planner sizes runtime slots before KV pages and clamps max forward
        // requests to the resulting slot count. Keep one unadvertised slot as
        // a rollback scratch for system-spec draft verification, plus a small
        // prefix-snapshot bank so partial MTP rejection can restore accepted
        // recurrent state without replaying the target model.
        const int q35_planned_slots = std::max<int>(1, mem_plan.state_slots);
        qwen3_5_runtime_rs_slots = std::max<int>(1, q35_planned_slots - 1);
        qwen3_5_scratch_rs_slot = qwen3_5_runtime_rs_slots;
        const int q35_spec_snapshot_slots = [] {
            const char* v = std::getenv("PIE_QWEN35_RS_SNAPSHOT_SLOTS");
            if (v == nullptr || v[0] == '\0') return 8;
            return std::clamp(std::atoi(v), 0, 16);
        }();
        const int q35_alloc_slots =
            qwen3_5_runtime_rs_slots + 1 + q35_spec_snapshot_slots;
        qwen3_5_state_cache = pie_cuda_driver::Qwen3_5StateCache::allocate(
            qwen3_5_layer_is_linear, conv_dim, cfg_q.linear_conv_kernel_dim,
            local_linear_value_heads,
            cfg_q.linear_key_head_dim,
            cfg_q.linear_value_head_dim,
            cfg_q.hidden_size,
            q35_alloc_slots);
        const std::size_t per_slot_recurrent_bytes =
            static_cast<std::size_t>(local_linear_value_heads) *
            cfg_q.linear_key_head_dim *
            cfg_q.linear_value_head_dim * sizeof(float);
        const std::size_t per_slot_conv_bytes =
            static_cast<std::size_t>(cfg_q.linear_conv_kernel_dim) *
            conv_dim * sizeof(std::uint16_t);
        const std::size_t num_linear_layers = qwen3_5_linear_layers;
        const std::size_t total_bytes = num_linear_layers *
            static_cast<std::size_t>(q35_alloc_slots) *
            (per_slot_recurrent_bytes + per_slot_conv_bytes);
        const std::size_t mtp_pending_bytes =
            static_cast<std::size_t>(q35_alloc_slots) *
            static_cast<std::size_t>(cfg_q.hidden_size) *
            sizeof(std::uint16_t);
        if (verbose) {
            std::cerr << "[pie-driver-cuda] qwen3.5 rs_cache: "
                      << num_linear_layers << " linear layers, "
                      << qwen3_5_runtime_rs_slots
                      << " runtime slots + 1 scratch + "
                      << q35_spec_snapshot_slots << " prefix snapshots, "
                      << (per_slot_recurrent_bytes + per_slot_conv_bytes)
                      << " B/slot (recurrent="
                      << per_slot_recurrent_bytes << " conv="
                      << per_slot_conv_bytes << "), mtp_pending="
                      << (mtp_pending_bytes / (1024 * 1024)) << " MiB, total ~"
                      << ((total_bytes + mtp_pending_bytes) / (1024 * 1024))
                      << " MiB\n";
        }

        if (is_qwen3_5_moe_arch) {
            qwen3_5_moe_ws = pie_cuda_driver::model::Qwen3_5MoeMlpWorkspace::allocate(
                max_workspace_tokens,
                cfg_q.hidden_size,
                cfg_q.num_experts,
                cfg_q.num_experts_per_tok,
                cfg_q.moe_intermediate_size / q35_tp_size,
                cfg_q.shared_expert_intermediate_size / q35_tp_size);
        }
    }

    auto swap_pool = pie_cuda_driver::SwapPool::allocate_for_cache(
        kv_cache, static_cast<int>(cfg.batching.swap_pool_size));

    pie_cuda_driver::ops::CublasHandle cublas;
    auto runtime_quant_scratch = runtime_quant_scratch_base;
    runtime_quant_scratch.max_tokens =
        static_cast<std::size_t>(max_workspace_tokens);
    if (!runtime_quant_scratch.empty()) {
        pie_cuda_driver::ops::reserve_runtime_quant_scratch(
            runtime_quant_scratch,
            /*seal_after_reserve=*/true);
        CUDA_CHECK(cudaDeviceSynchronize());
        if (verbose) {
            std::cerr << "[pie-driver-cuda] runtime quant graph scratch: "
                      << (runtime_quant_scratch.has_fp8 ? "fp8" : "")
                      << (runtime_quant_scratch.has_fp8 &&
                          runtime_quant_scratch.has_int8 ? "+" : "")
                      << (runtime_quant_scratch.has_int8 ? "int8" : "")
                      << " max_tokens=" << runtime_quant_scratch.max_tokens
                      << " max_N=" << runtime_quant_scratch.max_weight_rows
                      << " max_K=" << runtime_quant_scratch.max_weight_cols
                      << " reserved="
                      << (mem_plan.runtime_quant_scratch_bytes /
                          (1024 * 1024))
                      << " MiB (sealed for CUDA graphs)\n";
        }
    }

    // Persistent input buffers, sized for the planned worst case so
    // device pointers stay stable across fires (prereq for graphs).
    auto persistent_inputs = pie_cuda_driver::PersistentInputs::allocate(
        max_workspace_tokens,
        /*max_requests=*/mem_plan.max_requests,
        /*max_kv_pages=*/mem_plan.max_page_refs,
        mem_plan.capacity.max_custom_mask_bytes);

    std::optional<pie_cuda_driver::model::Gemma4MtpWorkspace> gemma4_mtp_ws;
    if (gemma4_mtp_weights) {
        gemma4_mtp_ws.emplace(
            pie_cuda_driver::model::Gemma4MtpWorkspace::allocate(
                *gemma4_mtp_weights,
                mem_plan.max_requests,
                mem_plan.max_page_refs,
                native_mtp_num_drafts));
        if (verbose) {
            std::cerr << "[pie-driver-cuda] Gemma4 MTP system drafter enabled: "
                      << "drafts=" << native_mtp_num_drafts
                      << " max_requests=" << mem_plan.max_requests
                      << " page_refs=" << mem_plan.max_page_refs << "\n";
        }
    }

    pie_cuda_driver::CustomAllReduce custom_ar;
    if (tp_comm_ptr != nullptr && vtable_opt != nullptr &&
        cfg.distributed.tp_size == 2) {
        custom_ar = pie_cuda_driver::CustomAllReduce(
            *tp_comm_ptr, /*same_process=*/true,
            /*max_bytes=*/8 * 1024 * 1024,
            /*rank_data_bytes=*/8 * 1024 * 1024,
            /*fusion_max_tokens=*/mem_plan.max_requests,
            /*fusion_hidden=*/engine.hf_config().hidden_size);
        custom_ar.register_buffer(*tp_comm_ptr, ws.norm_x.data(),
                                  ws.norm_x.nbytes());
        tp_comm_ptr->set_custom_all_reduce(&custom_ar);
    }

    if (verbose) {
        std::cerr << "[pie-driver-cuda] kv_cache: "
                  << runtime_kv_pages << " runtime pages";
        if (graph_pad_page >= 0) {
            std::cerr << " (+1 graph pad page)";
        }
        std::cerr << " × "
                  << kv_cache.page_size() << " tokens; "
                  << "format=" << kv_cache.format().name << "; "
                  << "workspace tokens=" << max_workspace_tokens
                  << "; max requests=" << mem_plan.max_requests
                  << "; page_refs=" << mem_plan.max_page_refs
                  << "; arena ~" << (mem_plan.arena_bytes / (1024 * 1024))
                  << " MiB"
                  << "; rq_scratch="
                  << (mem_plan.runtime_quant_scratch_bytes / (1024 * 1024))
                  << " MiB"
                  << "; attn_ws="
                  << (mem_plan.attn_float_workspace_bytes / (1024 * 1024))
                  << " MiB"
                  << "; swap_pool=" << swap_pool.num_pages() << " pages\n";
    }

    // Followers skip the server: rank 0 owns the fast path and broadcasts
    // each fire to followers via NCCL. tp_follower_serve (entered at the
    // end of run_impl) consumes those broadcasts and exits via
    // `tp_send_shutdown` from rank 0 once the next broadcast completes.
    const bool is_tp_follower =
        cfg.distributed.tp_size > 1 && cfg.distributed.tp_rank > 0;
    std::unique_ptr<pie_driver::InProcServer> server_p;
    if (!is_tp_follower && vtable_opt != nullptr) {
        // Response scratch lives in the per-backend `ResponseBuilder`
        // inside Executor — no central byte buffer on this path.
        server_p = std::make_unique<pie_driver::InProcServer>(*vtable_opt);
        register_server(server_p.get());
    } else if (!is_tp_follower && vtable_opt == nullptr) {
        // Parity-only invocation should have returned by now (the parity
        // branch above exits before reaching here). Falling through means
        // the caller didn't set parity flags — error out instead of
        // hanging without a server.
        std::cerr << "[pie-driver-cuda] standalone binary supports parity "
                     "tests only; embed via pie_driver_cuda_run_inproc\n";
        return 2;
    }

    if (install_signal_handlers) {
        std::signal(SIGINT, on_signal);
        std::signal(SIGTERM, on_signal);
    }

    std::uint64_t handled = 0;

    pie_cuda_driver::ForwardGraphCache graph_cache;

    // Per-arch forward knobs from the loaded HF config.
    pie_cuda_driver::model::LlamaLikeForwardCfg fwd_cfg{};
    pie_cuda_driver::model::Gemma2ForwardCfg gemma_fwd_cfg{};
    pie_cuda_driver::model::Gemma4ForwardCfg gemma4_fwd_cfg{};
    {
        const auto& hf = engine.hf_config();
        const std::string& mt = hf.model_type;
        fwd_cfg.use_qk_norm        = hf.use_qk_norm;
        fwd_cfg.use_qkv_bias       = hf.attention_bias;
        // OLMo-2 and OLMo-3 are the post-norm + q/k-norm architectures
        // bind_olmo3 materialises; everything else uses the standard
        // Llama pre-norm placement. q/k norms are forced on regardless
        // of the (sometimes missing) `use_qk_norm` config field.
        const bool is_olmo_post_norm = (mt == "olmo2" || mt == "olmo3");
        fwd_cfg.norm_placement = is_olmo_post_norm
            ? pie_cuda_driver::model::NormPlacement::Post
            : pie_cuda_driver::model::NormPlacement::Pre;
        if (is_olmo_post_norm) {
            fwd_cfg.use_qk_norm = true;
        }
        apply_rope_config(fwd_cfg, hf);
        fwd_cfg.sliding_window            = hf.sliding_window;
        // FlashInfer's decode dispatch set covers {1, 2, 3, 4, 8}. Other
        // GQA ratios use the prefill path for decode-only batches as well.
        const int gqa = hf.num_attention_heads / hf.num_key_value_heads;
        const bool gqa_in_decode_set = flashinfer_decode_supports_gqa(gqa);
        fwd_cfg.force_prefill_path = !gqa_in_decode_set;
        fwd_cfg.decode_plan_cuda_graph = use_cuda_graphs;
        // Tensor-parallel state. tp_comm == nullptr at tp_size == 1
        // keeps the original single-GPU branches in the forward kernels.
        fwd_cfg.tp_size = cfg.distributed.tp_size;
        fwd_cfg.tp_comm = tp_comm_ptr;
        fwd_cfg.emit_logits = (cfg.distributed.tp_rank == 0);
        {
            const int T = std::max(1, cfg.distributed.tp_size);
            const int local_q_heads = hf.num_attention_heads / T;
            const int local_kv_heads = hf.num_key_value_heads / T;
            fwd_cfg.use_xqa_decode =
                xqa_decode_enabled_by_env() &&
                pie_cuda_driver::ops::xqa_decode_bf16_supported(
                    local_q_heads, local_kv_heads, hf.head_dim_kernel,
                    mem_plan.kv_page_size, hf.sliding_window,
                    /*logits_soft_cap=*/0.f, /*sm_scale=*/-1.f) &&
                !has_non_full_attention_layers(hf);
            if (fwd_cfg.use_xqa_decode) {
                fwd_cfg.force_prefill_path = false;
                // Per-rank, per-device init of the selected XQA kernel's
                // smem attribute. FlashInfer sets this in a process-global
                // static initializer, which is not enough once TP ranks bind
                // different current devices.
                if (local_q_heads > 0 && local_kv_heads > 0 &&
                    local_q_heads % local_kv_heads == 0) {
                    pie_cuda_driver::ops::xqa_decode_bf16_warmup_current_device(
                        local_q_heads / local_kv_heads, mem_plan.kv_page_size);
                }
            }
        }

        // Gemma-2 / Gemma-3 forward knobs. `query_pre_attn_scalar` and
        // `final_logit_softcapping` come straight from the HF config —
        // see `loader/hf_config.cpp` for the parsing.
        gemma_fwd_cfg.query_pre_attn_scalar = hf.gemma_query_pre_attn_scalar;
        gemma_fwd_cfg.final_logit_softcap   = hf.gemma_final_logit_softcap;
        gemma_fwd_cfg.attn_logit_softcap    = hf.gemma_attn_logit_softcap;
        gemma_fwd_cfg.use_qk_norm           = (mt == "gemma3" || mt == "gemma3_text");
        gemma_fwd_cfg.force_prefill_path    = !gqa_in_decode_set;
        gemma_fwd_cfg.tp_size = cfg.distributed.tp_size;
        gemma_fwd_cfg.tp_comm = tp_comm_ptr;

        // Build the per-layer attention type → window_left + rope_theta
        // tables. Sliding layers get the configured window; full layers
        // pass -1 (kept for symmetry — flashinfer treats `-1` as "no
        // sliding"). For Gemma-3, sliding layers use the local-base
        // RoPE freq while full layers stick with `rope_theta`.
        const bool homogeneous = !has_non_full_attention_layers(hf);
        if (!homogeneous) {
            gemma_fwd_cfg.per_layer_window_left.reserve(hf.layer_types.size());
            gemma_fwd_cfg.per_layer_rope_theta.reserve(hf.layer_types.size());
            fwd_cfg.per_layer_window_left.reserve(hf.layer_types.size());
            for (const auto& t : hf.layer_types) {
                const bool is_sliding = (t == "sliding_attention");
                const int window = is_sliding ? hf.sliding_window : -1;
                gemma_fwd_cfg.per_layer_window_left.push_back(window);
                fwd_cfg.per_layer_window_left.push_back(window);
                const float theta =
                    (is_sliding && hf.rope_local_base_freq > 0.f)
                        ? hf.rope_local_base_freq
                        : hf.rope_theta;
                gemma_fwd_cfg.per_layer_rope_theta.push_back(theta);
            }
        }

        cudaDeviceProp serving_prop{};
        int serving_dev = 0;
        CUDA_CHECK(cudaGetDevice(&serving_dev));
        CUDA_CHECK(cudaGetDeviceProperties(&serving_prop, serving_dev));
        const bool prefill_decode_supported_head_dim =
            hf.head_dim_kernel == 64 || hf.head_dim_kernel == 128 ||
            hf.head_dim_kernel == 256 || hf.head_dim_kernel == 512;
        const bool force_prefill_decode_plan = [] {
            const char* v = std::getenv("PIE_CUDA_PREFILL_DECODE_PLAN");
            return v != nullptr && v[0] != '\0' && v[0] != '0';
        }();
        fwd_cfg.use_prefill_decode_plan =
            (serving_prop.major >= 9 || force_prefill_decode_plan) &&
            cfg.distributed.tp_size == 1 &&
            gqa_in_decode_set &&
            !fwd_cfg.force_prefill_path &&
            prefill_decode_supported_head_dim &&
            fwd_cfg.sliding_window < 0 &&
            fwd_cfg.per_layer_window_left.empty();
        if (fwd_cfg.use_prefill_decode_plan) {
            const std::size_t rank_kv_token_bytes =
                kv_page_bytes_homogeneous(
                    hf, std::max(1, cfg.distributed.tp_size), kv_format);
            const std::size_t global_kv_token_bytes =
                rank_kv_token_bytes *
                static_cast<std::size_t>(std::max(1, cfg.distributed.tp_size));
            const bool kv_heavy_attention =
                global_kv_token_bytes >= 192ull * 1024ull;
            // The dedicated decode kernel is faster for short KV histories.
            // Switch to the prefill-plan path only after the batch has enough
            // average KV pages for split-KV/full-attention work to pay for
            // itself.
            fwd_cfg.prefill_decode_min_kv_pages =
                kv_heavy_attention ? 1 : 7;
            if (const char* v = std::getenv("PIE_CUDA_PREFILL_DECODE_MIN_KV_PAGES")) {
                fwd_cfg.prefill_decode_min_kv_pages =
                    std::max(0, std::atoi(v));
            }
            fwd_cfg.prefill_decode_full_attention_min_requests = 256;
            fwd_cfg.prefill_decode_full_attention_min_kv_pages =
                kv_heavy_attention ? 1 : 7;
            if (const char* v = std::getenv("PIE_CUDA_PREFILL_DECODE_FULL_MIN_KV_PAGES")) {
                fwd_cfg.prefill_decode_full_attention_min_kv_pages =
                    std::max(0, std::atoi(v));
            }
            if (const char* v = std::getenv("PIE_CUDA_PREFILL_DECODE_NOGRAPHS")) {
                if (v[0] != '\0' && v[0] != '0') {
                    fwd_cfg.decode_plan_cuda_graph = false;
                }
            }
        }

        if (verbose) {
            const char* rope_name =
                (fwd_cfg.rope_kind == pie_cuda_driver::model::RopeKind::YaRN)
                    ? "yarn"
                    : (fwd_cfg.rope_kind ==
                       pie_cuda_driver::model::RopeKind::YaRNOriginal)
                          ? "yarn-original"
                          : "standard";
            std::cerr << "[pie-driver-cuda] model_type=" << mt
                      << " use_qk_norm=" << fwd_cfg.use_qk_norm
                      << " use_qkv_bias=" << fwd_cfg.use_qkv_bias
                      << " rope=" << rope_name
                      << " prefill_decode_plan="
                      << (fwd_cfg.use_prefill_decode_plan ? "on" : "off")
                      << " xqa_decode="
                      << (fwd_cfg.use_xqa_decode ? "on" : "off")
                      << " decode_plan_graph="
                      << (fwd_cfg.decode_plan_cuda_graph ? "on" : "off")
                      << " full_attn_min_R="
                      << fwd_cfg.prefill_decode_full_attention_min_requests
                      << "\n";
        }
    }

    if (is_gemma4_arch) {
        const auto& hf = engine.hf_config();
        gemma4_fwd_cfg.final_logit_softcap = hf.gemma_final_logit_softcap;
        const int gqa = hf.num_attention_heads / hf.num_key_value_heads;
        const bool gqa_in_decode_set = flashinfer_decode_supports_gqa(gqa);
        gemma4_fwd_cfg.force_prefill_path = !gqa_in_decode_set;
        gemma4_fwd_cfg.tp_size = cfg.distributed.tp_size;
        gemma4_fwd_cfg.tp_comm = tp_comm_ptr;
    }

    // Build the type-erased forward closure once. The captures live in
    // `main`'s scope (weights_*, fwd_cfg, gemma_fwd_cfg) and persist for
    // the lifetime of the server.
    pie_cuda_driver::ForwardFn forward_fn;
    pie_cuda_driver::NativeSystemDrafter system_drafter;
    pie_cuda_driver::model::LlamaLikePlanState llama_plan;
    // Gemma-4 26B-A4B's MoE block needs a routed-experts workspace
    // alongside the dense forward state. Inert (zero-byte) on dense
    // E2B / E4B / 31B variants.
    pie_cuda_driver::model::Gemma4MoeMlpWorkspace gemma4_moe_ws;
    if (is_gemma4_arch && engine.hf_config().gemma4_enable_moe) {
        const auto& hf_cfg = engine.hf_config();
        gemma4_moe_ws = pie_cuda_driver::model::Gemma4MoeMlpWorkspace::allocate(
            max_workspace_tokens,
            hf_cfg.hidden_size,
            hf_cfg.num_experts,
            hf_cfg.num_experts_per_tok,
            hf_cfg.moe_intermediate_size /
                std::max(1, cfg.distributed.tp_size));
    }
    if (is_gemma4_arch) {
        gemma4_moe_ws.allocate_row_decode(max_workspace_tokens);
    }
    if (is_gemma4_arch &&
        engine.hf_config().gemma_hidden_size_per_layer_input > 0) {
        const auto& hf_cfg = engine.hf_config();
        gemma4_moe_ws.allocate_ple(
            max_workspace_tokens,
            hf_cfg.num_hidden_layers *
                hf_cfg.gemma_hidden_size_per_layer_input);
    }
    if (is_gemma4_arch) {
        forward_fn = [&engine, &weights_gemma4, &gemma4_moe_ws, gemma4_fwd_cfg](
            pie_cuda_driver::model::Qwen3Workspace& ws,
            pie_cuda_driver::KvCache& cache,
            pie_cuda_driver::AttentionWorkspace& attn_ws,
            pie_cuda_driver::ops::CublasHandle& cublas,
            const std::int32_t* tok, const std::int32_t* pos,
            const std::uint32_t* qo_indptr,
            const std::uint32_t* kv_page_indices,
            const std::uint32_t* kv_page_indptr,
            const std::uint32_t* kv_last_page_lens,
            const std::uint32_t* qo_indptr_h,
            const std::uint32_t* kv_page_indices_h,
            const std::uint32_t* kv_page_indptr_h,
            const std::uint32_t* kv_last_page_lens_h,
            int N, int R, bool is_pure_decode,
            const std::uint8_t* mask_d, const std::int32_t* mask_indptr_d,
            const std::int32_t* slot_ids_h, const std::uint8_t* is_fresh_h,
            const std::int32_t* slot_ids_d,
            const std::int32_t* logit_row_indices_d,
            int num_logit_rows,
            bool /*tp_greedy_argmax*/) {
            pie_cuda_driver::model::gemma4_forward_paged(
                weights_gemma4, engine.hf_config(), gemma4_fwd_cfg,
                ws, gemma4_moe_ws, cache, attn_ws, cublas,
                tok, pos,
                qo_indptr, kv_page_indices, kv_page_indptr, kv_last_page_lens,
                qo_indptr_h, kv_page_indices_h, kv_page_indptr_h,
                kv_last_page_lens_h,
                N, R, is_pure_decode, mask_d, mask_indptr_d,
                logit_row_indices_d, num_logit_rows);
        };
        forward_fn.supports_compact_logits = true;
        forward_fn.set_logits_argmax_only =
            [](bool enabled) {
                pie_cuda_driver::model::set_gemma4_logits_argmax_only(enabled);
            };
        forward_fn.prepare = [&engine, &weights_gemma4, &gemma4_moe_ws,
                              &kv_cache, gemma4_fwd_cfg](
            pie_cuda_driver::AttentionWorkspace& attn_ws,
            const pie_cuda_driver::ForwardFn::PrepareInputs& prep) {
            pie_cuda_driver::model::prepare_gemma4_decode_plans(
                weights_gemma4, engine.hf_config(), gemma4_fwd_cfg,
                gemma4_moe_ws, kv_cache, attn_ws,
                prep.qo_indptr_h,
                prep.kv_page_indices_h,
                prep.kv_page_indptr_h,
                prep.kv_last_page_lens_h,
                prep.total_tokens,
                prep.num_requests,
                prep.is_pure_decode);
        };
        const char* gemma4_profile_env = std::getenv("PIE_GEMMA4_FORWARD_PROFILE");
        forward_fn.graph_safe =
            kv_cache.format().is_native_bf16() &&
            !(gemma4_profile_env != nullptr &&
              gemma4_profile_env[0] != '\0' &&
              gemma4_profile_env[0] != '0');
        forward_fn.graph_layout = [&gemma4_moe_ws]() {
            return pie_cuda_driver::model::gemma4_decode_graph_layout(
                gemma4_moe_ws);
        };
    } else if (is_gemma3n_arch) {
        // Loader-only milestone: bind_gemma3n loads every tensor; the
        // forward function (AltUp predict/correct + Laurel + activation
        // sparsity + PLE input gate) is a follow-up. The stub throws
        // with a clear message at the first fire_batch.
        pie_cuda_driver::model::Gemma3nForwardCfg gemma3n_fwd_cfg{};
        gemma3n_fwd_cfg.final_logit_softcap = engine.hf_config().gemma_final_logit_softcap;
        gemma3n_fwd_cfg.tp_size = cfg.distributed.tp_size;
        gemma3n_fwd_cfg.tp_comm = tp_comm_ptr;
        forward_fn = [&engine, &weights_gemma3n, gemma3n_fwd_cfg](
            pie_cuda_driver::model::Qwen3Workspace& ws,
            pie_cuda_driver::KvCache& cache,
            pie_cuda_driver::AttentionWorkspace& attn_ws,
            pie_cuda_driver::ops::CublasHandle& cublas,
            const std::int32_t* tok, const std::int32_t* pos,
            const std::uint32_t* qo_indptr,
            const std::uint32_t* kv_page_indices,
            const std::uint32_t* kv_page_indptr,
            const std::uint32_t* kv_last_page_lens,
            const std::uint32_t* qo_indptr_h,
            const std::uint32_t* kv_page_indices_h,
            const std::uint32_t* kv_page_indptr_h,
            const std::uint32_t* kv_last_page_lens_h,
            int N, int R, bool is_pure_decode,
            const std::uint8_t* mask_d, const std::int32_t* mask_indptr_d,
            const std::int32_t* slot_ids_h, const std::uint8_t* is_fresh_h,
            const std::int32_t* slot_ids_d,
            const std::int32_t* logit_row_indices_d,
            int num_logit_rows,
            bool tp_greedy_argmax) {
            pie_cuda_driver::model::gemma3n_forward_paged(
                weights_gemma3n, engine.hf_config(), gemma3n_fwd_cfg,
                ws, cache, attn_ws, cublas,
                tok, pos,
                qo_indptr, kv_page_indices, kv_page_indptr, kv_last_page_lens,
                qo_indptr_h, kv_page_indptr_h,
                N, R, is_pure_decode, mask_d, mask_indptr_d);
        };
    } else if (is_gemma_arch) {
        forward_fn = [&engine, &weights_gemma, gemma_fwd_cfg](
            pie_cuda_driver::model::Qwen3Workspace& ws,
            pie_cuda_driver::KvCache& cache,
            pie_cuda_driver::AttentionWorkspace& attn_ws,
            pie_cuda_driver::ops::CublasHandle& cublas,
            const std::int32_t* tok, const std::int32_t* pos,
            const std::uint32_t* qo_indptr,
            const std::uint32_t* kv_page_indices,
            const std::uint32_t* kv_page_indptr,
            const std::uint32_t* kv_last_page_lens,
            const std::uint32_t* qo_indptr_h,
            const std::uint32_t* kv_page_indices_h,
            const std::uint32_t* kv_page_indptr_h,
            const std::uint32_t* kv_last_page_lens_h,
            int N, int R, bool is_pure_decode,
            const std::uint8_t* mask_d, const std::int32_t* mask_indptr_d,
            const std::int32_t* slot_ids_h, const std::uint8_t* is_fresh_h,
            const std::int32_t* slot_ids_d,
            const std::int32_t* logit_row_indices_d,
            int num_logit_rows,
            bool tp_greedy_argmax) {
            pie_cuda_driver::model::gemma2_forward_paged(
                weights_gemma, engine.hf_config(), gemma_fwd_cfg,
                ws, cache, attn_ws, cublas,
                tok, pos,
                qo_indptr, kv_page_indices, kv_page_indptr, kv_last_page_lens,
                qo_indptr_h, kv_page_indptr_h,
                N, R, is_pure_decode, mask_d, mask_indptr_d);
        };
    } else if (is_mixtral_arch) {
        // Mixtral reuses LlamaLikeForwardCfg for its (identical) attention
        // half. The MoE block reads num_experts / num_experts_per_tok
        // straight from HfConfig.
        const int num_experts = engine.hf_config().num_experts;
        const int top_k       = engine.hf_config().num_experts_per_tok;
        forward_fn = [&engine, &weights_mixtral, fwd_cfg, num_experts, top_k](
            pie_cuda_driver::model::Qwen3Workspace& ws,
            pie_cuda_driver::KvCache& cache,
            pie_cuda_driver::AttentionWorkspace& attn_ws,
            pie_cuda_driver::ops::CublasHandle& cublas,
            const std::int32_t* tok, const std::int32_t* pos,
            const std::uint32_t* qo_indptr,
            const std::uint32_t* kv_page_indices,
            const std::uint32_t* kv_page_indptr,
            const std::uint32_t* kv_last_page_lens,
            const std::uint32_t* qo_indptr_h,
            const std::uint32_t* kv_page_indices_h,
            const std::uint32_t* kv_page_indptr_h,
            const std::uint32_t* kv_last_page_lens_h,
            int N, int R, bool is_pure_decode,
            const std::uint8_t* mask_d, const std::int32_t* mask_indptr_d,
            const std::int32_t* slot_ids_h, const std::uint8_t* is_fresh_h,
            const std::int32_t* slot_ids_d,
            const std::int32_t* logit_row_indices_d,
            int num_logit_rows,
            bool /*tp_greedy_argmax*/) {
            pie_cuda_driver::model::mixtral_forward_paged(
                weights_mixtral, engine.hf_config(), fwd_cfg,
                num_experts, top_k,
                ws, cache, attn_ws, cublas,
                tok, pos,
                qo_indptr, kv_page_indices, kv_page_indptr, kv_last_page_lens,
                qo_indptr_h, kv_page_indptr_h,
                N, R, is_pure_decode, mask_d, mask_indptr_d);
        };
    } else if (is_qwen3_5_arch) {
        const int q35_tp_size = cfg.distributed.tp_size;
        pie_cuda_driver::NcclComm* q35_tp_comm = tp_comm_ptr;
        forward_fn.prepare = [&engine, &kv_cache, &qwen3_5_plan_state,
                              q35_tp_size, q35_tp_comm](
            pie_cuda_driver::AttentionWorkspace& attn_ws,
            const pie_cuda_driver::ForwardFn::PrepareInputs& prep) {
            pie_cuda_driver::model::Qwen3_5ForwardCfg q35_fwd{};
            const auto& hf_q = engine.hf_config();
            const int gqa_q = hf_q.num_attention_heads /
                              std::max(1, hf_q.num_key_value_heads);
            q35_fwd.force_prefill_path =
                !flashinfer_decode_supports_gqa(gqa_q);
            q35_fwd.small_prefill_naive_attention_max_tokens =
                qwen35_small_spec_graph_tokens();
            q35_fwd.tp_size = q35_tp_size;
            q35_fwd.tp_comm = q35_tp_comm;
            pie_cuda_driver::model::prepare_qwen3_5_decode_plan(
                qwen3_5_plan_state, attn_ws, kv_cache, engine.hf_config(),
                q35_fwd, prep.qo_indptr_h, prep.kv_page_indptr_h,
                prep.kv_last_page_lens_h, prep.total_tokens,
                prep.num_requests, prep.is_pure_decode);
        };
        forward_fn.body = [&engine, &weights_qwen3_5, &qwen3_5_la_ws,
                           &qwen3_5_state_cache, &qwen3_5_plan_state,
                           q35_tp_size, q35_tp_comm](
            pie_cuda_driver::model::Qwen3Workspace& ws,
            pie_cuda_driver::KvCache& cache,
            pie_cuda_driver::AttentionWorkspace& attn_ws,
            pie_cuda_driver::ops::CublasHandle& cublas,
            const std::int32_t* tok, const std::int32_t* pos,
            const std::uint32_t* qo_indptr,
            const std::uint32_t* kv_page_indices,
            const std::uint32_t* kv_page_indptr,
            const std::uint32_t* kv_last_page_lens,
            const std::uint32_t* qo_indptr_h,
            const std::uint32_t* kv_page_indices_h,
            const std::uint32_t* kv_page_indptr_h,
            const std::uint32_t* kv_last_page_lens_h,
            int N, int R, bool is_pure_decode,
            const std::uint8_t* mask_d, const std::int32_t* mask_indptr_d,
            const std::int32_t* slot_ids_h, const std::uint8_t* is_fresh_h,
            const std::int32_t* slot_ids_d,
            const std::int32_t* logit_row_indices_d,
            int num_logit_rows,
            bool /*tp_greedy_argmax*/) {
            pie_cuda_driver::model::Qwen3_5ForwardCfg q35_fwd{};
            const auto& hf_q = engine.hf_config();
            const int gqa_q = hf_q.num_attention_heads /
                              std::max(1, hf_q.num_key_value_heads);
            q35_fwd.force_prefill_path =
                !flashinfer_decode_supports_gqa(gqa_q);
            q35_fwd.small_prefill_naive_attention_max_tokens =
                qwen35_small_spec_graph_tokens();
            q35_fwd.tp_size = q35_tp_size;
            q35_fwd.tp_comm = q35_tp_comm;
            pie_cuda_driver::model::qwen3_5_forward_paged(
                weights_qwen3_5, engine.hf_config(), q35_fwd, qwen3_5_plan_state,
                ws, qwen3_5_la_ws, cache, qwen3_5_state_cache,
                attn_ws, cublas,
                tok, pos,
                qo_indptr, kv_page_indices, kv_page_indptr, kv_last_page_lens,
                qo_indptr_h, kv_page_indptr_h,
                N, R, is_pure_decode, mask_d, mask_indptr_d,
                slot_ids_h, is_fresh_h, slot_ids_d,
                logit_row_indices_d, num_logit_rows);
        };
        if (weights_qwen3_5.mtp.has_value() && native_mtp_num_drafts > 0) {
            const int mtp_position_offset = qwen35_mtp_draft_position_offset();
            const bool mtp_prefix_global = qwen35_mtp_prefix_global_cache();
            system_drafter.max_drafts = native_mtp_num_drafts;
            system_drafter.draft_position_offset = mtp_position_offset;
            system_drafter.draft_global_cache_uses_prefix_position =
                mtp_prefix_global;
            system_drafter.draft_step_writes_sampled_tokens =
                weights_qwen3_5.mtp->lm_head_scale_inv != nullptr ||
                qwen35_mtp_fused_gemv_enabled();
            system_drafter.commit_verified_prefix =
                [&engine, &weights_qwen3_5, &qwen3_5_la_ws,
                 &qwen3_5_state_cache, q35_tp_size, q35_tp_comm](
                const pie_cuda_driver::NativeSystemCommitInputs& in) {
                pie_cuda_driver::model::Qwen3_5ForwardCfg q35_fwd{};
                q35_fwd.tp_size = q35_tp_size;
                q35_fwd.tp_comm = q35_tp_comm;
                pie_cuda_driver::model::qwen3_5_mtp_process_cache(
                    weights_qwen3_5, engine.hf_config(), q35_fwd,
                    in.target_ws, qwen3_5_la_ws, in.kv_cache,
                    qwen3_5_state_cache, in.cublas,
                    in.token_ids, in.positions, in.qo_indptr,
                    in.kv_page_indices, in.kv_page_indptr,
                    in.kv_last_page_lens, in.slot_ids,
                    in.source_row_indices, in.total_tokens,
                    in.num_requests);
            };
            system_drafter.draft_step =
                [&engine, &weights_qwen3_5, &qwen3_5_la_ws,
                 q35_tp_size, q35_tp_comm, mtp_prefix_global](
                pie_cuda_driver::model::Qwen3Workspace& ws,
                pie_cuda_driver::KvCache& cache,
                pie_cuda_driver::ops::CublasHandle& cublas,
                const std::int32_t* tok,
                const std::int32_t* pos,
                const std::int32_t* base_hidden_row_indices,
                const std::int32_t* request_ids,
                const std::uint32_t* kv_page_indices,
                const std::uint32_t* kv_page_indptr,
                const std::uint32_t* kv_last_page_lens,
                std::int32_t* sampled_token_ids,
                int N,
                int draft_step,
                int max_global_tokens) {
                pie_cuda_driver::model::Qwen3_5ForwardCfg q35_fwd{};
                q35_fwd.tp_size = q35_tp_size;
                q35_fwd.tp_comm = q35_tp_comm;
                q35_fwd.mtp_global_cache_uses_prefix_position =
                    mtp_prefix_global;
                pie_cuda_driver::model::qwen3_5_mtp_forward(
                    weights_qwen3_5, engine.hf_config(), q35_fwd,
                    ws, qwen3_5_la_ws, cache, cublas,
                    tok, pos, base_hidden_row_indices, request_ids,
                    kv_page_indices, kv_page_indptr, kv_last_page_lens,
                    sampled_token_ids, N, draft_step, max_global_tokens);
            };
        }
        forward_fn.graph_safe =
            kv_cache.format().is_native_bf16() &&
            !qwen35_forward_profile_enabled();
        forward_fn.supports_compact_logits = true;
        forward_fn.graph_layout = [&qwen3_5_plan_state]() {
            return pie_cuda_driver::model::qwen3_5_decode_graph_layout(
                qwen3_5_plan_state);
        };
        forward_fn.supports_small_prefill_graph =
            kv_cache.format().is_native_bf16() && !kv_cache.hnd_layout() &&
            qwen35_small_spec_graph_tokens() > 0;
    } else if (is_qwen3_5_moe_arch) {
        const int q35moe_tp_size = cfg.distributed.tp_size;
        pie_cuda_driver::NcclComm* q35moe_tp_comm = tp_comm_ptr;
        forward_fn.prepare = [&engine, &kv_cache, &qwen3_5_plan_state,
                              q35moe_tp_size, q35moe_tp_comm](
            pie_cuda_driver::AttentionWorkspace& attn_ws,
            const pie_cuda_driver::ForwardFn::PrepareInputs& prep) {
            pie_cuda_driver::model::Qwen3_5ForwardCfg q35_fwd{};
            const auto& hf_q = engine.hf_config();
            const int gqa_q = hf_q.num_attention_heads /
                              std::max(1, hf_q.num_key_value_heads);
            q35_fwd.force_prefill_path =
                !flashinfer_decode_supports_gqa(gqa_q);
            q35_fwd.small_prefill_naive_attention_max_tokens =
                qwen35_small_spec_graph_tokens();
            q35_fwd.tp_size = q35moe_tp_size;
            q35_fwd.tp_comm = q35moe_tp_comm;
            pie_cuda_driver::model::prepare_qwen3_5_decode_plan(
                qwen3_5_plan_state, attn_ws, kv_cache, engine.hf_config(),
                q35_fwd, prep.qo_indptr_h, prep.kv_page_indptr_h,
                prep.kv_last_page_lens_h, prep.total_tokens,
                prep.num_requests, prep.is_pure_decode);
        };
        forward_fn.body = [&engine, &weights_qwen3_5_moe, &qwen3_5_la_ws,
                           &qwen3_5_moe_ws, &qwen3_5_state_cache,
                           &qwen3_5_plan_state,
                           q35moe_tp_size, q35moe_tp_comm](
            pie_cuda_driver::model::Qwen3Workspace& ws,
            pie_cuda_driver::KvCache& cache,
            pie_cuda_driver::AttentionWorkspace& attn_ws,
            pie_cuda_driver::ops::CublasHandle& cublas,
            const std::int32_t* tok, const std::int32_t* pos,
            const std::uint32_t* qo_indptr,
            const std::uint32_t* kv_page_indices,
            const std::uint32_t* kv_page_indptr,
            const std::uint32_t* kv_last_page_lens,
            const std::uint32_t* qo_indptr_h,
            const std::uint32_t* kv_page_indices_h,
            const std::uint32_t* kv_page_indptr_h,
            const std::uint32_t* kv_last_page_lens_h,
            int N, int R, bool is_pure_decode,
            const std::uint8_t* mask_d, const std::int32_t* mask_indptr_d,
            const std::int32_t* slot_ids_h, const std::uint8_t* is_fresh_h,
            const std::int32_t* slot_ids_d,
            const std::int32_t* logit_row_indices_d,
            int num_logit_rows,
            bool /*tp_greedy_argmax*/) {
            pie_cuda_driver::model::Qwen3_5ForwardCfg q35_fwd{};
            const auto& hf_q = engine.hf_config();
            const int gqa_q = hf_q.num_attention_heads /
                              std::max(1, hf_q.num_key_value_heads);
            q35_fwd.force_prefill_path =
                !flashinfer_decode_supports_gqa(gqa_q);
            q35_fwd.small_prefill_naive_attention_max_tokens =
                qwen35_small_spec_graph_tokens();
            q35_fwd.tp_size = q35moe_tp_size;
            q35_fwd.tp_comm = q35moe_tp_comm;
            pie_cuda_driver::model::qwen3_5_moe_forward_paged(
                weights_qwen3_5_moe, engine.hf_config(), q35_fwd,
                qwen3_5_plan_state,
                ws, qwen3_5_la_ws, qwen3_5_moe_ws,
                cache, qwen3_5_state_cache,
                attn_ws, cublas,
                tok, pos,
                qo_indptr, kv_page_indices, kv_page_indptr, kv_last_page_lens,
                qo_indptr_h, kv_page_indptr_h,
                N, R, is_pure_decode, mask_d, mask_indptr_d,
                slot_ids_h, is_fresh_h, slot_ids_d,
                logit_row_indices_d, num_logit_rows);
        };
        if (weights_qwen3_5_moe.mtp.has_value() && native_mtp_num_drafts > 0) {
            const int mtp_position_offset = qwen35_mtp_draft_position_offset();
            const bool mtp_prefix_global = qwen35_mtp_prefix_global_cache();
            system_drafter.max_drafts = native_mtp_num_drafts;
            system_drafter.draft_position_offset = mtp_position_offset;
            system_drafter.draft_global_cache_uses_prefix_position =
                mtp_prefix_global;
            system_drafter.commit_verified_prefix =
                [&engine, &weights_qwen3_5_moe, &qwen3_5_la_ws,
                 &qwen3_5_state_cache, q35moe_tp_size, q35moe_tp_comm](
                const pie_cuda_driver::NativeSystemCommitInputs& in) {
                pie_cuda_driver::model::Qwen3_5ForwardCfg q35_fwd{};
                q35_fwd.tp_size = q35moe_tp_size;
                q35_fwd.tp_comm = q35moe_tp_comm;
                pie_cuda_driver::model::qwen3_5_moe_mtp_process_cache(
                    weights_qwen3_5_moe, engine.hf_config(), q35_fwd,
                    in.target_ws, qwen3_5_la_ws, in.kv_cache,
                    qwen3_5_state_cache, in.cublas,
                    in.token_ids, in.positions, in.qo_indptr,
                    in.kv_page_indices, in.kv_page_indptr,
                    in.kv_last_page_lens, in.slot_ids,
                    in.source_row_indices, in.total_tokens,
                    in.num_requests);
            };
            system_drafter.draft_step =
                [&engine, &weights_qwen3_5_moe, &qwen3_5_la_ws,
                 &qwen3_5_moe_ws, q35moe_tp_size, q35moe_tp_comm,
                 mtp_prefix_global](
                pie_cuda_driver::model::Qwen3Workspace& ws,
                pie_cuda_driver::KvCache& cache,
                pie_cuda_driver::ops::CublasHandle& cublas,
                const std::int32_t* tok,
                const std::int32_t* pos,
                const std::int32_t* base_hidden_row_indices,
                const std::int32_t* request_ids,
                const std::uint32_t* kv_page_indices,
                const std::uint32_t* kv_page_indptr,
                const std::uint32_t* kv_last_page_lens,
                std::int32_t* sampled_token_ids,
                int N,
                int draft_step,
                int max_global_tokens) {
                (void)sampled_token_ids;
                pie_cuda_driver::model::Qwen3_5ForwardCfg q35_fwd{};
                q35_fwd.tp_size = q35moe_tp_size;
                q35_fwd.tp_comm = q35moe_tp_comm;
                q35_fwd.mtp_global_cache_uses_prefix_position =
                    mtp_prefix_global;
                pie_cuda_driver::model::qwen3_5_moe_mtp_forward(
                    weights_qwen3_5_moe, engine.hf_config(), q35_fwd,
                    ws, qwen3_5_la_ws, qwen3_5_moe_ws, cache, cublas,
                    tok, pos, base_hidden_row_indices, request_ids,
                    kv_page_indices, kv_page_indptr, kv_last_page_lens,
                    nullptr, N, draft_step, max_global_tokens);
            };
        }
        const char* q35moe_profile_env = std::getenv("PIE_QWEN35_MOE_PROFILE");
        forward_fn.graph_safe =
            !(q35moe_profile_env != nullptr &&
              q35moe_profile_env[0] != '\0' &&
              q35moe_profile_env[0] != '0');
        forward_fn.graph_layout = [&qwen3_5_plan_state]() {
            return pie_cuda_driver::model::qwen3_5_decode_graph_layout(
                qwen3_5_plan_state);
        };
        forward_fn.supports_compact_logits = true;
        forward_fn.supports_small_prefill_graph =
            kv_cache.format().is_native_bf16() && !kv_cache.hnd_layout() &&
            qwen35_small_spec_graph_tokens() > 0;
    } else {
        // Llama-like decode is graph-replay-safe because (a) the body
        // is host-work-free (the prepare hook hoisted DecodePlan out of
        // the capture region); (b) flashinfer's plan_info layout is
        // pinned across fires when `enable_cuda_graph=true` —
        // `padded_batch_size = max_grid_size / gdy` (stable), and the
        // int_buf offsets are deterministic from that; (c) per-fire,
        // DecodePlan only refreshes int_buf content (request_indices,
        // kv_tile_indices, o_indptr, block_valid_mask) at the same
        // device offsets, so the captured kernel reads fresh data through
        // its stable pointer args.
        // Quantized KV currently dequantizes active physical pages into a BF16
        // scratch cache before FlashInfer. That dequant launch shape depends
        // on the live page count, while decode graph keys only bucket request
        // count/layout, so replay can leave newly-active pages stale.
        forward_fn.graph_safe = kv_cache.format().is_native_bf16();
        forward_fn.supports_compact_logits = true;
        forward_fn.supports_tp_greedy_argmax =
            cfg.distributed.tp_size > 1 &&
            weights_llama.lm_head_tp_shard != nullptr;
        forward_fn.prepare = [&engine, &kv_cache, &fwd_cfg, &llama_plan](
            pie_cuda_driver::AttentionWorkspace& attn_ws,
            const pie_cuda_driver::ForwardFn::PrepareInputs& prep) {
            pie_cuda_driver::model::prepare_llama_like_decode_plan(
                llama_plan, attn_ws, kv_cache, engine.hf_config(),
                fwd_cfg,
                prep.qo_indptr_h,
                prep.kv_page_indices_d,
                prep.kv_page_indptr_h,
                prep.kv_page_indptr_d,
                prep.kv_last_page_lens_h,
                prep.kv_last_page_lens_d,
                prep.total_tokens,
                prep.num_requests,
                prep.is_pure_decode);
        };
        forward_fn.graph_layout = [&llama_plan]() {
            return pie_cuda_driver::model::llama_like_decode_graph_layout(
                llama_plan);
        };
        forward_fn.body = [&engine, &weights_llama, fwd_cfg, &llama_plan](
            pie_cuda_driver::model::Qwen3Workspace& ws,
            pie_cuda_driver::KvCache& cache,
            pie_cuda_driver::AttentionWorkspace& attn_ws,
            pie_cuda_driver::ops::CublasHandle& cublas,
            const std::int32_t* tok, const std::int32_t* pos,
            const std::uint32_t* qo_indptr,
            const std::uint32_t* kv_page_indices,
            const std::uint32_t* kv_page_indptr,
            const std::uint32_t* kv_last_page_lens,
            const std::uint32_t* qo_indptr_h,
            const std::uint32_t* kv_page_indices_h,
            const std::uint32_t* kv_page_indptr_h,
            const std::uint32_t* kv_last_page_lens_h,
            int N, int R, bool is_pure_decode,
            const std::uint8_t* mask_d, const std::int32_t* mask_indptr_d,
            const std::int32_t* slot_ids_h, const std::uint8_t* is_fresh_h,
            const std::int32_t* slot_ids_d,
            const std::int32_t* logit_row_indices_d,
            int num_logit_rows,
            bool tp_greedy_argmax) {
            pie_cuda_driver::model::llama_like_forward_paged(
                weights_llama, engine.hf_config(), fwd_cfg, llama_plan,
                ws, cache, attn_ws, cublas,
                tok, pos,
                qo_indptr, kv_page_indices, kv_page_indptr, kv_last_page_lens,
                qo_indptr_h, kv_page_indptr_h,
                N, R, is_pure_decode,
                logit_row_indices_d, num_logit_rows,
                tp_greedy_argmax,
                mask_d, mask_indptr_d);
        };
    }

    if (gemma4_mtp_weights && gemma4_mtp_ws) {
        system_drafter.max_drafts = native_mtp_num_drafts;
        system_drafter.draft_next =
            [&weights_gemma4, &mtp_w = *gemma4_mtp_weights,
             &mtp_ws = *gemma4_mtp_ws, gemma4_mtp_runtime](
                const pie_cuda_driver::SystemSpecDraftInputs& in,
                std::span<pie_driver::PerRequestOutput> per_req) {
                pie_cuda_driver::model::gemma4_mtp_draft(
                    mtp_w, weights_gemma4, mtp_ws, gemma4_mtp_runtime,
                    in, per_req);
            };
    }

    pie_cuda_driver::Executor executor{
        engine, ws, kv_cache, attn_ws, cublas,
        max_workspace_tokens,
        mem_plan.max_requests,
        graph_pad_page,
        persistent_inputs, verbose, std::move(forward_fn),
        std::move(system_drafter),
        use_cuda_graphs ? &graph_cache : nullptr,
        /*tp_comm=*/tp_comm_ptr,
        /*tp_cpu_gate_key=*/{},
        /*rs_cache=*/((is_qwen3_5_arch || is_qwen3_5_moe_arch) ? &qwen3_5_state_cache : nullptr),
        /*rs_cache_scratch_slot=*/qwen3_5_scratch_rs_slot,
        /*response_builder=*/{},
    };
    executor.tp_cpu_gate_key = cfg.distributed.nccl_unique_id_hex;
    // Pass-level speculation is runtime-owned. `.system_speculation()` is
    // driver-owned when a native drafter is configured.
    if (verbose && use_cuda_graphs) {
        std::cerr << "[pie-driver-cuda] CUDA graphs enabled (experimental)\n";
    }

    // TP ranks run as independent driver instances. Followers can reach the
    // first NCCL receive before rank 0 has finished building its CUDA serving
    // state; posting that idle receive while the leader is still allocating can
    // show as a persistent 100% GPU-util spin and has reproduced startup
    // wedges. Rendezvous on CPU after all persistent allocations are complete,
    // then pre-capture any graph-safe decode lattice before rank 0 publishes
    // readiness and followers enter the NCCL loop.
    tp_startup_cpu_barrier(cfg);
    if (use_cuda_graphs) {
        pie_cuda_driver::capture_forward_graph_lattice(executor);
    }
    tp_startup_cpu_barrier(cfg);

    if (is_tp_follower) {
        if (verbose) {
            std::cerr << "[pie-driver-cuda] tp follower rank "
                      << cfg.distributed.tp_rank
                      << " ready (waiting on rank-0 broadcasts"
                      << (executor.tp_cpu_gate_key.empty()
                              ? ", cpu_gate=off"
                              : ", cpu_gate=on")
                      << ")\n";
        }
        // Followers: block on rank-0 broadcasts until shutdown.
        std::atomic<bool> stop{false};
        pie_cuda_driver::tp_follower_serve(executor, stop);
    } else {
        // Capabilities reflect both the loaded HF config and the live
        // KV cache. Only rank 0 reports — the wrapper expects exactly
        // one READY per TP group.
        auto c = engine.capabilities();
        c.total_pages = runtime_kv_pages;
        c.swap_pool_size = swap_pool.num_pages();
        const bool rs_cache_required =
            (is_qwen3_5_arch || is_qwen3_5_moe_arch) &&
            qwen3_5_state_cache.max_slots() > 0;
        const std::uint64_t rs_cache_slots = rs_cache_required
            ? static_cast<std::uint64_t>(qwen3_5_runtime_rs_slots)
            : 0;
        const std::uint64_t rs_cache_slot_bytes = rs_cache_required
            ? static_cast<std::uint64_t>(qwen3_5_linear_layers) *
                  (qwen3_5_state_cache.conv_slot_stride_bytes() +
                   qwen3_5_state_cache.recurrent_slot_stride_bytes()) +
                  static_cast<std::uint64_t>(
                      std::max(0, qwen3_5_state_cache.hidden_size())) *
                      sizeof(std::uint16_t)
            : 0;
        const bool rs_cache_spec_rollback =
            rs_cache_required && cfg.distributed.tp_size <= 1 &&
            qwen3_5_scratch_rs_slot >= 0;
        const bool system_speculation_supported =
            static_cast<bool>(executor.system_drafter);
        const auto max_forward_requests_caps = rs_cache_required
            ? std::min<std::uint64_t>(
                  static_cast<std::uint64_t>(mem_plan.capacity.max_forward_requests),
                  rs_cache_slots)
            : static_cast<std::uint64_t>(mem_plan.capacity.max_forward_requests);
        nlohmann::json caps = {
            {"total_pages",            c.total_pages},
            {"kv_page_size",           mem_plan.kv_page_size},
            {"swap_pool_size",         c.swap_pool_size},
            {"rs_cache_required",      rs_cache_required},
            {"rs_cache_slots",         rs_cache_slots},
            {"rs_cache_slot_bytes",    rs_cache_slot_bytes},
            {"rs_cache_spec_rollback", rs_cache_spec_rollback},
            {"system_speculation_supported", system_speculation_supported},
            {"default_system_speculation", system_speculation_supported},
            {"max_forward_tokens",     mem_plan.capacity.max_forward_tokens},
            {"max_forward_requests",   max_forward_requests_caps},
            {"max_page_refs",          mem_plan.capacity.max_page_refs},
            {"max_logit_rows",         mem_plan.capacity.max_logit_rows},
            {"max_prob_rows",          mem_plan.capacity.max_prob_rows},
            {"max_custom_mask_bytes",  mem_plan.capacity.max_custom_mask_bytes},
            {"max_sampler_rows",       mem_plan.capacity.max_sampler_rows},
            {"max_logprob_labels",     mem_plan.capacity.max_logprob_labels},
            {"arch_name",              c.arch_name},
            {"vocab_size",             c.vocab_size},
            {"max_model_len",          c.max_model_len},
            {"activation_dtype",       c.activation_dtype},
            {"snapshot_dir",           c.snapshot_dir},
        };
        if (verbose) {
            std::cerr << "[pie-driver-cuda] forward_limits: "
                      << "tokens=" << mem_plan.capacity.max_forward_tokens
                      << " requests=" << mem_plan.capacity.max_forward_requests
                      << " page_refs=" << mem_plan.capacity.max_page_refs
                      << " logit_rows=" << mem_plan.capacity.max_logit_rows
                      << " prob_rows=" << mem_plan.capacity.max_prob_rows
                      << " custom_mask_bytes="
                      << mem_plan.capacity.max_custom_mask_bytes
                      << " sampler_rows=" << mem_plan.capacity.max_sampler_rows
                      << " logprob_labels="
                      << mem_plan.capacity.max_logprob_labels
                      << "\n";
        }
        const std::string caps_json = caps.dump();
        ready_cb(caps_json.c_str(), ready_ctx);

        if (verbose) {
            std::cerr << "[pie-driver-cuda] serving on in-process channel\n";
        }
        pie_cuda_driver::service::InProcService service{
            executor, kv_cache, swap_pool};
        service.serve_forever(*server_p);
        handled = service.handled();
        // Leader exited serve loop — wake followers so they can tear
        // down cleanly.
        if (cfg.distributed.tp_size > 1) {
            pie_cuda_driver::tp_send_shutdown(
                *tp_comm_ptr, executor.tp_cpu_gate_key);
        }
    }

    if (server_p) {
        unregister_server(server_p.get());
    }
    if (verbose) {
        std::cerr << "[pie-driver-cuda] shutting down (handled " << handled
                  << " requests)\n";
    }
    return 0;
}

}  // namespace

// Standalone-binary entry. Now parity-test-only — if `--parity-tokens` is
// supplied the engine runs one forward pass and exits; otherwise we
// error out (use `pie_driver_cuda_run_inproc` for serve). The standalone
// `pie_driver_cuda` executable exists solely to host the parity tests
// under `driver/cuda/tests/`.
extern "C" int pie_driver_cuda_run(int argc,
                                   char** argv,
                                   int install_signal_handlers,
                                   pie_driver_cuda_ready_cb ready_cb,
                                   void* ready_ctx) {
    try {
        return run_impl(argc, argv, install_signal_handlers, ready_cb, ready_ctx,
                        /*vtable_opt=*/nullptr);
    } catch (const std::exception& e) {
        std::cerr << "[pie-driver-cuda] fatal: " << e.what() << "\n";
        return -1;
    } catch (...) {
        std::cerr << "[pie-driver-cuda] fatal: unknown exception\n";
        return -1;
    }
}

extern "C" int pie_driver_cuda_run_inproc(int argc,
                                          char** argv,
                                          int install_signal_handlers,
                                          pie_driver_cuda_ready_cb ready_cb,
                                          void* ready_ctx,
                                          pie_driver::PieInProcVTable vtable) {
    try {
        return run_impl(argc, argv, install_signal_handlers, ready_cb, ready_ctx,
                        &vtable);
    } catch (const std::exception& e) {
        std::cerr << "[pie-driver-cuda] fatal: " << e.what() << "\n";
        return -1;
    } catch (...) {
        std::cerr << "[pie-driver-cuda] fatal: unknown exception\n";
        return -1;
    }
}

// Reaches into the same server registry the SIGINT/SIGTERM handler uses.
// One host process can embed multiple same-flavor DP replicas, so stop
// every live driver server (shmem or inproc) rather than only the most
// recently registered one.
extern "C" void pie_driver_cuda_request_stop(void) {
    stop_servers();
}
