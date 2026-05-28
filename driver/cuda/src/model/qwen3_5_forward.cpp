#include "model/qwen3_5_forward.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <utility>

#include <cuda_runtime.h>

#include "cuda_check.hpp"
#include "kernels/argmax.hpp"
#include "kernels/causal_conv1d.hpp"
#include "kernels/deinterleave.hpp"
#include "kernels/embed.hpp"
#include "kernels/gated_delta_net.hpp"
#include "kernels/gather_rows.hpp"
#include "kernels/kv_paged.hpp"
#include "kernels/residual_add.hpp"
#include "kernels/rmsnorm.hpp"
#include "kernels/rope.hpp"
#include "kernels/split_packed.hpp"
#include "kernels/swiglu.hpp"
#include "ops/attention_naive.hpp"
#include "ops/attention_naive_paged.hpp"
#include "ops/attention_flashinfer.hpp"
#include "ops/gemm.hpp"

namespace pie_cuda_driver::model {

Qwen3_5LinearAttnWorkspace Qwen3_5LinearAttnWorkspace::allocate(
    int max_tokens, int conv_dim, int v_h, int k_h, int k_d, int v_d,
    int hq)
{
    Qwen3_5LinearAttnWorkspace ws;
    const std::size_t N     = static_cast<std::size_t>(max_tokens);
    const std::size_t v_dim = static_cast<std::size_t>(v_h) * v_d;
    const std::size_t k_dim = static_cast<std::size_t>(k_h) * k_d;
    ws.mixed_qkv      = DeviceBuffer<std::uint16_t>::alloc(N * conv_dim);
    ws.mixed_qkvz     = DeviceBuffer<std::uint16_t>::alloc(N * (conv_dim + v_dim));
    ws.ba             = DeviceBuffer<std::uint16_t>::alloc(N * (std::size_t)2 * v_h);
    ws.mixed_qkv_post = DeviceBuffer<std::uint16_t>::alloc(N * conv_dim);
    ws.z              = DeviceBuffer<std::uint16_t>::alloc(N * v_dim);
    ws.a              = DeviceBuffer<std::uint16_t>::alloc(N * v_h);
    ws.b              = DeviceBuffer<std::uint16_t>::alloc(N * v_h);
    ws.q_norm   = DeviceBuffer<float>::alloc(N * (std::size_t)v_h * k_d);
    ws.k_norm   = DeviceBuffer<float>::alloc(N * (std::size_t)v_h * k_d);
    ws.v_fp32   = DeviceBuffer<float>::alloc(N * v_dim);
    ws.g_log    = DeviceBuffer<float>::alloc(N * v_h);
    ws.beta     = DeviceBuffer<float>::alloc(N * v_h);
    ws.core_out = DeviceBuffer<float>::alloc(N * v_dim);
    ws.core_out_bf16 = DeviceBuffer<std::uint16_t>::alloc(N * v_dim);
    ws.q_raw = DeviceBuffer<std::uint16_t>::alloc(N * k_dim);
    ws.k_raw = DeviceBuffer<std::uint16_t>::alloc(N * k_dim);
    ws.v_raw = DeviceBuffer<std::uint16_t>::alloc(N * v_dim);
    ws.q_pre = DeviceBuffer<float>::alloc(N * (std::size_t)k_h * k_d);
    ws.k_pre = DeviceBuffer<float>::alloc(N * (std::size_t)k_h * k_d);
    ws.fa_qg_packed = DeviceBuffer<std::uint16_t>::alloc(N * (std::size_t)2 * hq);
    ws.fa_gate      = DeviceBuffer<std::uint16_t>::alloc(N * (std::size_t)hq);
    return ws;
}

namespace {

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

bool qwen35_forward_profile_enabled() {
    static const bool enabled = [] {
        const char* v = std::getenv("PIE_QWEN35_FORWARD_PROFILE");
        return v != nullptr && v[0] != '\0' && v[0] != '0';
    }();
    return enabled;
}

std::uint64_t qwen35_forward_profile_print_limit() {
    static const std::uint64_t limit = [] {
        const char* v = std::getenv("PIE_QWEN35_FORWARD_PROFILE_LIMIT");
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

bool qwen35_mtp_fused_gemv_enabled() {
    static const bool enabled = [] {
        const char* v = std::getenv("PIE_QWEN35_MTP_FUSED_GEMV");
        if (v == nullptr || v[0] == '\0') return false;
        return v[0] != '0';
    }();
    return enabled;
}

bool qwen35_mtp_shift_prefix_cache() {
    static const bool enabled = [] {
        const char* v = std::getenv("PIE_QWEN35_MTP_SHIFT_PREFIX_CACHE");
        if (v == nullptr || v[0] == '\0') return false;
        return v[0] != '0';
    }();
    return enabled;
}

void qwen35_dense_mlp_block(
    const Qwen3_5LayerWeights& Lw,
    const HfConfig& cfg,
    const Qwen3_5ForwardCfg& fwd_cfg,
    Qwen3Workspace& ws,
    ops::CublasHandle& cublas,
    int N,
    cudaStream_t stream)
{
    const int H = cfg.hidden_size;
    const int T_mlp = std::max(1, fwd_cfg.tp_size);
    const int I = cfg.intermediate_size / T_mlp;
    NcclComm* tp_mlp = (T_mlp > 1) ? fwd_cfg.tp_comm : nullptr;
    if (Lw.gate_up_proj_fused != nullptr && !ws.gate_up_fused.empty()) {
        ops::gemm_act_x_w(cublas.handle(),
            ws.norm_x.data(), ops::WeightView(*Lw.gate_up_proj_fused),
            ws.gate_up_fused.data(), N, 2 * I, H);
        kernels::launch_chunked_swiglu_bf16(
            ws.gate_up_fused.data(), ws.gate.data(), N, I, stream);
    } else {
        ops::gemm_act_x_w(cublas.handle(),
            ws.norm_x.data(), make_weight_view(Lw.gate_proj, Lw.gate_proj_quant),
            ws.gate.data(), N, I, H);
        ops::gemm_act_x_w(cublas.handle(),
            ws.norm_x.data(), make_weight_view(Lw.up_proj, Lw.up_proj_quant),
            ws.up.data(), N, I, H);
        kernels::launch_swiglu_bf16(
            ws.gate.data(), ws.up.data(), ws.gate.data(),
            N * I, stream);
    }

    // down_proj: TP=1 fuses residual via beta=1; TP>1 is row-parallel
    // and needs all-reduce before adding back to the residual stream.
    if (T_mlp == 1) {
        ops::gemm_act_x_w(cublas.handle(),
            ws.gate.data(), make_weight_view(Lw.down_proj, Lw.down_proj_quant),
            ws.y.data(), N, H, I, /*beta=*/1.f);
    } else {
        ops::gemm_act_x_w(cublas.handle(),
            ws.gate.data(), make_weight_view(Lw.down_proj, Lw.down_proj_quant),
            ws.norm_y.data(), N, H, I, /*beta=*/0.f);
        tp_mlp->all_reduce_bf16(ws.norm_y.data(),
            static_cast<std::size_t>(N) * H, ncclSum, stream);
        kernels::launch_residual_add_bf16(
            ws.y.data(), ws.norm_y.data(),
            static_cast<std::size_t>(N) * H, stream);
    }
}

struct MtpProfile {
    bool enabled = false;
    int N = 0;
    double input_fc_ms = 0.0;
    double attn_ms = 0.0;
    double mlp_ms = 0.0;
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
        input_fc_ms = attn_ms = mlp_ms = lm_head_ms = total_ms = 0.0;
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
    const double named = p.input_fc_ms + p.attn_ms + p.mlp_ms + p.lm_head_ms;
    const double other = p.total_ms > named ? p.total_ms - named : 0.0;
    std::cerr
        << "[pie-mtp-profile] seq=" << seq
        << " N=" << p.N
        << " total_ms=" << p.total_ms
        << " input_fc_ms=" << p.input_fc_ms
        << " attn_ms=" << p.attn_ms
        << " mlp_ms=" << p.mlp_ms
        << " lm_head_ms=" << p.lm_head_ms
        << " other_ms=" << other
        << "\n";
}

struct ForwardProfile {
    bool enabled = false;
    int N = 0;
    int R = 0;
    int num_logit_rows = 0;
    int linear_layers = 0;
    int full_layers = 0;
    bool pure_decode = false;
    double reset_ms = 0.0;
    double embed_ms = 0.0;
    double attn_norm_ms = 0.0;
    double linear_attn_ms = 0.0;
    double linear_proj_ms = 0.0;
    double linear_conv_ms = 0.0;
    double linear_prep_ms = 0.0;
    double linear_recur_ms = 0.0;
    double linear_post_ms = 0.0;
    double linear_out_ms = 0.0;
    double full_attn_ms = 0.0;
    double mlp_norm_ms = 0.0;
    double mlp_ms = 0.0;
    double final_norm_ms = 0.0;
    double lm_head_ms = 0.0;
    double final_copy_ms = 0.0;
    double total_ms = 0.0;
    cudaEvent_t total_start = nullptr;
    cudaEvent_t total_stop = nullptr;
    cudaEvent_t stage_start = nullptr;
    cudaEvent_t stage_stop = nullptr;

    ~ForwardProfile() {
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

    void begin(
        int n,
        int r,
        bool decode,
        int logit_rows,
        cudaStream_t stream)
    {
        enabled = qwen35_forward_profile_enabled();
        if (!enabled) return;
        cudaStreamCaptureStatus capture_status = cudaStreamCaptureStatusNone;
        CUDA_CHECK(cudaStreamIsCapturing(stream, &capture_status));
        if (capture_status != cudaStreamCaptureStatusNone) {
            enabled = false;
            return;
        }
        ensure_events();
        N = n;
        R = r;
        pure_decode = decode;
        num_logit_rows = logit_rows;
        linear_layers = full_layers = 0;
        reset_ms = embed_ms = attn_norm_ms = linear_attn_ms = full_attn_ms = 0.0;
        linear_proj_ms = linear_conv_ms = linear_prep_ms = linear_recur_ms = 0.0;
        linear_post_ms = linear_out_ms = 0.0;
        mlp_norm_ms = mlp_ms = final_norm_ms = lm_head_ms = final_copy_ms = 0.0;
        total_ms = 0.0;
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
void profile_forward_stage(
    ForwardProfile& profile,
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

template <class F>
void profile_forward_stage_ptr(
    ForwardProfile* profile,
    double ForwardProfile::*field,
    cudaStream_t stream,
    F&& fn)
{
    if (profile == nullptr) {
        fn();
        return;
    }
    profile_forward_stage(*profile, (*profile).*field, stream, std::forward<F>(fn));
}

void maybe_print_forward_profile(const ForwardProfile& p) {
    if (!p.enabled) return;
    static std::uint64_t seq = 0;
    ++seq;
    const std::uint64_t limit = qwen35_forward_profile_print_limit();
    if (limit == 0 || seq > limit) return;
    const double named =
        p.reset_ms + p.embed_ms + p.attn_norm_ms + p.linear_attn_ms +
        p.full_attn_ms + p.mlp_norm_ms + p.mlp_ms + p.final_norm_ms +
        p.lm_head_ms + p.final_copy_ms;
    const double other = p.total_ms > named ? p.total_ms - named : 0.0;
    std::cerr
        << "[pie-qwen35-forward-profile] seq=" << seq
        << " N=" << p.N
        << " R=" << p.R
        << " pure_decode=" << (p.pure_decode ? 1 : 0)
        << " logit_rows=" << p.num_logit_rows
        << " linear_layers=" << p.linear_layers
        << " full_layers=" << p.full_layers
        << " total_ms=" << p.total_ms
        << " reset_ms=" << p.reset_ms
        << " embed_ms=" << p.embed_ms
        << " attn_norm_ms=" << p.attn_norm_ms
        << " linear_attn_ms=" << p.linear_attn_ms
        << " linear_proj_ms=" << p.linear_proj_ms
        << " linear_conv_ms=" << p.linear_conv_ms
        << " linear_prep_ms=" << p.linear_prep_ms
        << " linear_recur_ms=" << p.linear_recur_ms
        << " linear_post_ms=" << p.linear_post_ms
        << " linear_out_ms=" << p.linear_out_ms
        << " full_attn_ms=" << p.full_attn_ms
        << " mlp_norm_ms=" << p.mlp_norm_ms
        << " mlp_ms=" << p.mlp_ms
        << " final_norm_ms=" << p.final_norm_ms
        << " lm_head_ms=" << p.lm_head_ms
        << " final_copy_ms=" << p.final_copy_ms
        << " other_ms=" << other
        << "\n";
}

// Linear-attn layer body. Reads `ws.norm_x` (post-input-layernorm
// activations) and writes the layer's contribution into `ws.norm_y`
// (which the caller adds to `ws.y` as the residual).
//
// Multi-request: every state-touching kernel is the batched variant —
// causal_conv1d_{update,prefill} and {recurrent,chunk}_gated_delta_*
// each fire as a single grid indirected through `slot_ids_d` (and on
// prefill `qo_indptr_d` too) to address each request's slab. One
// launch per kernel per layer regardless of R, eliminating
// `R × num_layers × {2,4}` per-token / per-fire launch overhead.
//
// `slot_ids_h` and `slot_ids_d` describe the same mapping in host and
// device memory respectively. When both are nullptr (parity /
// single-request callers), every kernel takes its legacy single-request
// path against slot 0 — matches the R=1 layout exactly.
void linear_attn_layer_body(
    const Qwen3_5LayerWeights& Lw,
    const HfConfig& cfg,
    const Qwen3_5ForwardCfg& fwd_cfg,
    Qwen3Workspace& ws,
    Qwen3_5LinearAttnWorkspace& la,
    RecurrentStateCache& state_cache,
    int layer_idx,
    int N, int R,
    bool is_pure_decode,
    const std::int32_t*  slot_ids_h,
    const std::int32_t*  slot_ids_d,
    const std::uint32_t* qo_indptr_h,
    const std::uint32_t* qo_indptr_d,
    ops::CublasHandle& cublas,
    cudaStream_t stream,
    ForwardProfile* profile = nullptr)
{
    // TP-local dims for linear-attention. tp_size == 1 keeps everything
    // unsharded. The K/V head counts must divide tp_size (checked at
    // engine load); each rank operates on its 1/T head share.
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

    auto slot_for = [&](int r) -> int {
        return slot_ids_h ? slot_ids_h[r] : 0;
    };
    const int snapshot_base_slot =
        (R == 1) ? state_cache.spec_snapshot_base_slot() : -1;
    const int snapshot_count =
        snapshot_base_slot >= 0 ? state_cache.spec_snapshot_count() : 0;

    // ── In-projections ────────────────────────────────────────────
    // Linear-attn projections stay bf16 (no QuantMeta companion in
    // Qwen3_5LayerWeights) — the implicit WeightView ctor pulls the
    // bf16 tensor through unchanged.
    profile_forward_stage_ptr(profile, &ForwardProfile::linear_proj_ms, stream, [&] {
        if (Lw.la_in_proj_qkvz != nullptr && Lw.la_in_proj_ba != nullptr) {
            ops::gemm_act_x_w(cublas.handle(),
                ws.norm_x.data(), *Lw.la_in_proj_qkvz,
                la.mixed_qkvz.data(), N, conv_dim + V_dim, H);
            ops::gemm_act_x_w(cublas.handle(),
                ws.norm_x.data(), *Lw.la_in_proj_ba,
                la.ba.data(), N, 2 * V_h, H);
            kernels::launch_split_qwen_gdn_projections_bf16(
                la.mixed_qkvz.data(), la.ba.data(),
                la.mixed_qkv.data(), la.z.data(), la.b.data(), la.a.data(),
                N, conv_dim, V_dim, V_h, stream);
        } else {
            // mixed_qkv [N, conv_dim] = norm_x @ in_proj_qkv.T
            ops::gemm_act_x_w(cublas.handle(),
                ws.norm_x.data(), *Lw.la_in_proj_qkv,
                la.mixed_qkv.data(), N, conv_dim, H);
            // z [N, V_dim] = norm_x @ in_proj_z.T
            ops::gemm_act_x_w(cublas.handle(),
                ws.norm_x.data(), *Lw.la_in_proj_z,
                la.z.data(), N, V_dim, H);
            // a [N, V_h] = norm_x @ in_proj_a.T   (b symmetric)
            ops::gemm_act_x_w(cublas.handle(),
                ws.norm_x.data(), *Lw.la_in_proj_a,
                la.a.data(), N, V_h, H);
            ops::gemm_act_x_w(cublas.handle(),
                ws.norm_x.data(), *Lw.la_in_proj_b,
                la.b.data(), N, V_h, H);
        }
    });

    // ── Causal depthwise conv1d (kernel=K, fused silu) ────────────
    // Per-request conv_state lives in `state_cache.conv_state(layer, slot)`.
    // Layout: conv_state is [conv_K, conv_dim] bf16 per slot.
    auto* qkv_in_base   = la.mixed_qkv.data();
    auto* qkv_post_base = la.mixed_qkv_post.data();
    profile_forward_stage_ptr(profile, &ForwardProfile::linear_conv_ms, stream, [&] {
        if (is_pure_decode) {
            // N == R; one token per request. Decode hot path → batched
            // kernel: one launch picks per-request slot via slot_ids_d.
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
                // Legacy single-request path (parity entrypoint, slot 0).
                kernels::launch_causal_conv1d_update_bf16(
                    qkv_in_base, Lw.la_conv1d_w->data(),
                    Lw.la_conv1d_b ? Lw.la_conv1d_b->data() : nullptr,
                    state_cache.conv_state(layer_idx, 0),
                    qkv_post_base,
                    conv_dim, conv_K, stream);
            }
        } else {
            // Prefill: batched kernel — one launch over (C, R) blocks; each
            // block reads its own (t0_r, Nr_r) window from qo_indptr_d and
            // walks tokens internally, persisting the trailing K-window
            // into the request's slot. Falls back to host-loop for the
            // legacy single-request parity entrypoint.
            if (slot_ids_d != nullptr && qo_indptr_d != nullptr) {
                kernels::launch_causal_conv1d_prefill_batched_snapshot_bf16(
                    qkv_in_base, Lw.la_conv1d_w->data(),
                    Lw.la_conv1d_b ? Lw.la_conv1d_b->data() : nullptr,
                    qkv_post_base,
                    state_cache.conv_state(layer_idx, /*slot=*/0),
                    slot_ids_d, qo_indptr_d,
                    static_cast<long long>(state_cache.conv_kernel()) *
                        state_cache.conv_dim(),
                    R, conv_dim, conv_K,
                    snapshot_base_slot, snapshot_count, stream);
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

    // ── Split mixed_qkv_post + prep recurrent inputs ─────────────
    // mixed_qkv_post[N, conv_dim] packs [q_raw | k_raw | v_raw]. The fused
    // prep kernel emits compact q/k heads, fp32 v, and per-head g/beta.
    auto* qkv_base = la.mixed_qkv_post.data();
    const bool use_warp_tiled_recurrent =
        !is_pure_decode &&
        slot_ids_d != nullptr &&
        qo_indptr_d != nullptr &&
        N <= qwen35_gdn_warp_tiled_max_tokens() &&
        K_d <= 256;
    const bool use_decode_gqa_recurrent =
        is_pure_decode &&
        slot_ids_d != nullptr &&
        V_h != K_h &&
        V_h % K_h == 0;
    profile_forward_stage_ptr(profile, &ForwardProfile::linear_prep_ms, stream, [&] {
        kernels::launch_qwen_gdn_post_conv_prep_bf16(
            qkv_base, la.a.data(), la.b.data(),
            Lw.la_A_log_fp32, Lw.la_dt_bias->data(),
            la.q_pre.data(), la.k_pre.data(), la.v_fp32.data(),
            la.g_log.data(), la.beta.data(),
            N, K_h, V_h, K_d, V_d, conv_dim, stream);

        // repeat_interleave from K_h to V_h heads. The warp-tiled verifier
        // and GQA decode recurrent kernels can index the compact K_h layout
        // directly, so skip materialisation there.
        if (V_h != K_h && !use_warp_tiled_recurrent &&
            !use_decode_gqa_recurrent) {
            kernels::launch_repeat_interleave_heads_fp32(
                la.q_pre.data(), la.q_norm.data(), N, K_h, V_h, K_d, stream);
            kernels::launch_repeat_interleave_heads_fp32(
                la.k_pre.data(), la.k_norm.data(), N, K_h, V_h, K_d, stream);
        }
    });
    const float* q_recur_full =
        (V_h == K_h) ? la.q_pre.data() : la.q_norm.data();
    const float* k_recur_full =
        (V_h == K_h) ? la.k_pre.data() : la.k_norm.data();

        // ── Recurrent update ───────────────────────────────────────────
        // Both decode and prefill: one batched launch over (R, V_h) blocks
        // with per-block slot (and on prefill, qo_indptr) indirection.
        // Decode = one token per request; prefill = each block walks its
        // own T_r-token window from `qo_indptr_d` along the recurrence.
    profile_forward_stage_ptr(profile, &ForwardProfile::linear_recur_ms, stream, [&] {
        {
            const std::size_t qk_step = static_cast<std::size_t>(V_h) * K_d;
            const std::size_t v_step  = static_cast<std::size_t>(V_dim);
            const std::size_t gh_step = static_cast<std::size_t>(V_h);
            const bool state_bf16 = state_cache.recurrent_state_bf16();
            void* state_slot0 = state_cache.recurrent_state_raw(
                layer_idx, /*slot=*/0);
            const auto slot_stride = static_cast<long long>(
                state_cache.recurrent_slot_stride_floats());
            if (is_pure_decode) {
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
                            kernels::launch_chunk_gated_delta_prefill_batched_warp_tiled_gqa_snapshot_state_bf16(
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
                                snapshot_base_slot, snapshot_count, stream);
                        } else {
                            kernels::launch_chunk_gated_delta_prefill_batched_warp_tiled_gqa_snapshot(
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
                                snapshot_base_slot, snapshot_count, stream);
                        }
                    } else if (use_warp_tiled_recurrent) {
                        if (state_bf16) {
                            kernels::launch_chunk_gated_delta_prefill_batched_warp_tiled_snapshot_state_bf16(
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
                                snapshot_base_slot, snapshot_count, stream);
                        } else {
                            kernels::launch_chunk_gated_delta_prefill_batched_warp_tiled_snapshot(
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
                                snapshot_base_slot, snapshot_count, stream);
                        }
                    } else if (N <= qwen35_gdn_cached_prefill_max_tokens()) {
                        if (state_bf16) {
                            kernels::launch_chunk_gated_delta_prefill_batched_cached_snapshot_state_bf16(
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
                                snapshot_base_slot, snapshot_count, stream);
                        } else {
                            kernels::launch_chunk_gated_delta_prefill_batched_cached_snapshot(
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
                                snapshot_base_slot, snapshot_count, stream);
                        }
                    } else {
                        if (state_bf16) {
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
                                R, V_h, K_d, V_d, stream);
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
                                R, V_h, K_d, V_d, stream);
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
        }
    });

    // ── core_out (fp32) → fused RMSNormGated → bf16 ────────────────
    // core_out has [N, V_h, V_d] layout = [N, V_dim] flat. We want
    // RMSNormGated over V_d per (n, h). Treat as [N*V_h, V_d].
    //
    // Fuses the previous (launch_fp32_to_bf16 → launch_rmsnorm_gated_bf16)
    // pair into a single kernel that reads fp32 x directly. Per-row
    // HBM IO drops from 12*V_d bytes to 8*V_d bytes (eliminates the
    // intermediate bf16 round-trip), and we save one kernel launch
    // per linear layer per fire (6216 launches eliminated at saturated
    // Qwen3.5-4B 512×128).
    profile_forward_stage_ptr(profile, &ForwardProfile::linear_post_ms, stream, [&] {
        kernels::launch_rmsnorm_gated_fp32_in_bf16(
            la.core_out.data(), la.z.data(), Lw.la_norm_w_fp32,
            la.core_out_bf16.data(),
            N * V_h, V_d, /*eps=*/cfg.rms_norm_eps, stream);
    });

    // ── out_proj: [N, V_dim] → [N, H]. On TP=1 we fuse the residual via
    //    beta=1; on TP>1 the proj is row-parallel so we write to scratch,
    //    all-reduce, then residual-add into y.
    NcclComm* tp = (T > 1) ? fwd_cfg.tp_comm : nullptr;
    profile_forward_stage_ptr(profile, &ForwardProfile::linear_out_ms, stream, [&] {
        if (T == 1) {
            ops::gemm_act_x_w(cublas.handle(),
                la.core_out_bf16.data(), *Lw.la_out_proj,
                ws.y.data(), N, H, V_dim, /*beta=*/1.f);
        } else {
            ops::gemm_act_x_w(cublas.handle(),
                la.core_out_bf16.data(), *Lw.la_out_proj,
                ws.norm_y.data(), N, H, V_dim, /*beta=*/0.f);
            tp->all_reduce_bf16(ws.norm_y.data(),
                static_cast<std::size_t>(N) * H, ncclSum, stream);
            kernels::launch_residual_add_bf16(
                ws.y.data(), ws.norm_y.data(),
                static_cast<std::size_t>(N) * H, stream);
        }
    });
}

// Full-attention layer body. Reads `ws.norm_x`, writes contribution
// to `ws.norm_y`. KV cache and flashinfer mirror the qwen3 path.
void full_attn_layer_body(
    const Qwen3_5LayerWeights& Lw,
    const HfConfig& cfg,
    const Qwen3_5ForwardCfg& fwd_cfg,
    Qwen3Workspace& ws,
    Qwen3_5LinearAttnWorkspace& la,
    KvCache& cache,
    AttentionWorkspace& attn_ws,
    const ops::DecodePlanCache* decode_plan,  // non-null on decode path
    const ops::PrefillPlanCache* prefill_plan,
    int kv_layer,
    int N, int R,
    const std::int32_t* positions,
    const std::uint32_t* qo_indptr,
    const std::uint32_t* kv_page_indices,
    const std::uint32_t* kv_page_indptr,
    const std::uint32_t* kv_last_page_lens,
    const std::uint32_t* qo_indptr_h,
    const std::uint32_t* kv_page_indptr_h,
    ops::CublasHandle& cublas,
    cudaStream_t stream)
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

    // ── q/k/v projections (q is 2× wide for the output gate) ──────
    const int qgkv_dim = 2 * Hq + 2 * Hk;
    const bool use_fused_qgkv =
        Lw.fa_qgkv_proj_fused != nullptr &&
        !ws.gate_up_fused.empty() &&
        ws.gate_up_fused.numel() >= static_cast<std::size_t>(N) * qgkv_dim;
    if (use_fused_qgkv) {
        ops::gemm_act_x_w(cublas.handle(),
            ws.norm_x.data(), ops::WeightView(*Lw.fa_qgkv_proj_fused),
            ws.gate_up_fused.data(), N, qgkv_dim, H);
        kernels::launch_split_qkv_bf16(
            ws.gate_up_fused.data(),
            la.fa_qg_packed.data(), ws.k.data(), ws.v.data(),
            N, 2 * Hq, Hk, stream);
    } else {
        // qg_packed [N, 2*Hq] = norm_x @ q_proj.T
        ops::gemm_act_x_w(cublas.handle(),
            ws.norm_x.data(), make_weight_view(Lw.fa_q_proj, Lw.fa_q_proj_quant),
            la.fa_qg_packed.data(), N, 2 * Hq, H);
        ops::gemm_act_x_w(cublas.handle(),
            ws.norm_x.data(), make_weight_view(Lw.fa_k_proj, Lw.fa_k_proj_quant),
            ws.k.data(), N, Hk, H);
        ops::gemm_act_x_w(cublas.handle(),
            ws.norm_x.data(), make_weight_view(Lw.fa_v_proj, Lw.fa_v_proj_quant),
            ws.v.data(), N, Hk, H);
    }
    kernels::launch_split_q_gate_bf16(
        la.fa_qg_packed.data(), ws.q.data(), la.fa_gate.data(),
        N, num_q_heads_local, d, stream);

    // ── q_norm / k_norm (gemma-style (1+w)·x_hat) ─────────────────
    kernels::launch_rmsnorm_gemma_bf16(
        ws.q.data(), Lw.fa_q_norm->data(), ws.q.data(),
        N * num_q_heads_local, d, eps, stream);
    kernels::launch_rmsnorm_gemma_bf16(
        ws.k.data(), Lw.fa_k_norm->data(), ws.k.data(),
        N * num_kv_heads_local, d, eps, stream);

    // ── Partial RoPE ──────────────────────────────────────────────
    kernels::launch_rope_partial_bf16(
        ws.q.data(), ws.k.data(), positions,
        N, num_q_heads_local, num_kv_heads_local,
        d, rotary_dim, cfg.rope_theta, stream);

    // ── Write K/V to paged cache ──────────────────────────────────
    auto kv_view = cache.layer_view(kv_layer);
    kernels::launch_write_kv_to_pages(
        kv_view, ws.k.data(), ws.v.data(),
        qo_indptr, kv_page_indices, kv_page_indptr, kv_last_page_lens,
        N, R, stream);

    // ── Flashinfer attention ──────────────────────────────────────
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

    // ── Output gate: attn_out *= sigmoid(gate) ────────────────────
    kernels::launch_sigmoid_gate_inplace_bf16(
        ws.attn_out.data(), la.fa_gate.data(), N * Hq, stream);

    // ── o_proj fused with post-attn residual on TP=1; on TP>1 row-
    //    parallel: write to scratch, all-reduce, residual-add to y.
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

}  // namespace

void prepare_qwen3_5_decode_plan(
    Qwen3_5PlanState& state,
    AttentionWorkspace& attn_ws,
    KvCache& cache,
    const HfConfig& cfg,
    const Qwen3_5ForwardCfg& fwd_cfg,
    const std::uint32_t* qo_indptr_h,
    const std::uint32_t* kv_page_indptr_h,
    const std::uint32_t* kv_last_page_lens_h,
    int total_tokens,
    int num_requests,
    bool is_pure_decode,
    cudaStream_t stream)
{
    state.use_prefill_plan = false;
    if (!is_pure_decode || fwd_cfg.force_prefill_path) {
        state.decode_plan.reset();
        if (cache.format().is_native_bf16() && !cache.hnd_layout()) {
            if (!state.prefill_plan) {
                state.prefill_plan = ops::make_prefill_plan();
            }
            const int T = std::max(1, fwd_cfg.tp_size);
            const bool enable_graph =
                fwd_cfg.small_prefill_naive_attention_max_tokens > 0 &&
                total_tokens <=
                    fwd_cfg.small_prefill_naive_attention_max_tokens;
            ops::plan_attention_flashinfer_prefill_bf16(
                *state.prefill_plan,
                qo_indptr_h,
                kv_page_indptr_h,
                kv_last_page_lens_h,
                total_tokens,
                num_requests,
                cfg.num_attention_heads / T,
                cfg.num_key_value_heads / T,
                cfg.head_dim,
                cache.page_size(),
                attn_ws,
                stream,
                enable_graph,
                /*window_left=*/-1,
                /*full_attention_variant=*/false,
                cache.hnd_layout(),
                /*causal_mask=*/true);
            state.use_prefill_plan = true;
        }
        return;
    }
    if (!state.decode_plan) {
        state.decode_plan = ops::make_decode_plan();
    }
    // The decode kernel runs on per-rank slices of Q / KV — its tile
    // geometry must be planned for the per-rank head count, not the
    // full unsharded count. (Mistral-7B TP=2 happens to work with
    // either value because gqa is invariant under sharding *and* the
    // sharded Q tile size still rounds favorably; Qwen3.6-MoE
    // (head_dim=256, num_heads=16, num_kv=2 → 16/2=8 per-rank-q,
    // 1 per-rank-kv) does not — flashinfer's chunk-metadata read
    // overruns its 256-byte allocation when the full 16/2 plan meets
    // a small per-rank kv_chunks count.)
    const int T = std::max(1, fwd_cfg.tp_size);
    ops::plan_attention_flashinfer_decode(
        *state.decode_plan, kv_page_indptr_h, num_requests,
        cfg.num_attention_heads / T,
        cfg.num_key_value_heads / T,
        cfg.head_dim,
        cache.page_size(), attn_ws, stream,
        /*enable_cuda_graph=*/true,
        /*full_attention_variant=*/false,
        cache.hnd_layout());
}

std::uint32_t qwen3_5_decode_graph_layout(
    const Qwen3_5PlanState& state)
{
    if (state.use_prefill_plan && state.prefill_plan) {
        return ops::prefill_plan_graph_layout(*state.prefill_plan);
    }
    if (!state.decode_plan) return 0;
    return ops::decode_plan_graph_layout(*state.decode_plan);
}

void qwen3_5_forward_paged(
    const Qwen3_5Weights& w,
    const HfConfig& cfg,
    const Qwen3_5ForwardCfg& fwd_cfg,
    Qwen3_5PlanState& plan_state,
    Qwen3Workspace& ws,
    Qwen3_5LinearAttnWorkspace& la_ws,
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
    const std::int32_t* slot_ids_h,
    const std::uint8_t* is_fresh_h,
    const std::int32_t* slot_ids_d,
    const std::int32_t* logit_row_indices_d,
    int num_logit_rows)
{
    const int H  = cfg.hidden_size;
    const int V  = cfg.vocab_size;
    const int N  = total_tokens;
    const int R  = num_requests;
    const float eps = cfg.rms_norm_eps;
    cudaStream_t stream = cublas.stream();
    ForwardProfile profile;
    profile.begin(N, R, is_pure_decode, num_logit_rows, stream);

    // Per-slot reset for any request whose slot was just (re)assigned.
    // The runtime guarantees a slot is_fresh only on the first fire of a
    // context (which is always a prefill), so zeroing here matches the
    // "fresh state before consumption" semantic without disturbing
    // continuing decodes that share the fire. Legacy path (slot_ids_h
    // null) keeps the old "reset all on prefill" behaviour for the
    // parity entry point; max_slots == 1 there makes the two equivalent.
    profile_forward_stage(profile, profile.reset_ms, stream, [&] {
        if (slot_ids_h != nullptr && is_fresh_h != nullptr) {
            for (int r = 0; r < R; ++r) {
                if (is_fresh_h[r]) {
                    state_cache.reset_slot(slot_ids_h[r], stream);
                }
            }
        } else if (!is_pure_decode) {
            state_cache.reset(stream);
        }
    });

    // Decode plan was refreshed by `prepare_qwen3_5_decode_plan` before
    // this body call (in serving) or as part of the host-side parity
    // setup. Reading it from `plan_state` keeps host work — and its
    // attendant cudaMemcpyAsync H2D from a stack-allocated indptr_h_buf
    // — out of any cudaStream capture region.
    const ops::DecodePlanCache* decode_plan =
        plan_state.decode_plan ? plan_state.decode_plan.get() : nullptr;
    const ops::PrefillPlanCache* prefill_plan =
        (plan_state.use_prefill_plan && plan_state.prefill_plan)
            ? plan_state.prefill_plan.get()
            : nullptr;

    // 1. Embed.
    profile_forward_stage(profile, profile.embed_ms, stream, [&] {
        kernels::launch_embed_bf16(
            token_ids, w.embed->data(), ws.y.data(),
            N, H, cfg.vocab_size, stream);
    });

    // 2. Per-layer.
    for (std::size_t L = 0; L < w.layers.size(); ++L) {
        const auto& Lw = w.layers[L];

        // Pre-attention norm: y → norm_x.
        profile_forward_stage(profile, profile.attn_norm_ms, stream, [&] {
            kernels::launch_rmsnorm_gemma_bf16(
                ws.y.data(), Lw.attn_norm_pre->data(), ws.norm_x.data(),
                N, H, eps, stream);
        });

        if (Lw.kind == Qwen3_5LayerWeights::Kind::LinearAttn) {
            ++profile.linear_layers;
            profile_forward_stage(profile, profile.linear_attn_ms, stream, [&] {
                linear_attn_layer_body(
                    Lw, cfg, fwd_cfg, ws, la_ws, state_cache,
                    static_cast<int>(L), N, R, is_pure_decode,
                    slot_ids_h, slot_ids_d, qo_indptr_h, qo_indptr,
                    cublas, stream, &profile);
            });
        } else {
            ++profile.full_layers;
            profile_forward_stage(profile, profile.full_attn_ms, stream, [&] {
                full_attn_layer_body(
                    Lw, cfg, fwd_cfg, ws, la_ws, cache, attn_ws,
                    decode_plan, prefill_plan, Lw.kv_layer,
                    N, num_requests,
                    positions, qo_indptr, kv_page_indices, kv_page_indptr,
                    kv_last_page_lens, qo_indptr_h, kv_page_indptr_h,
                    cublas, stream);
            });
        }

        // (Post-attention residual is fused into the body's final GEMM
        //  via beta=1 on tp_size==1, or all-reduce + residual_add for
        //  tp_size>1. Either way ws.y has the post-attn state at this
        //  point.)

        // Post-attention norm + SwiGLU MLP + residual.
        profile_forward_stage(profile, profile.mlp_norm_ms, stream, [&] {
            kernels::launch_rmsnorm_gemma_bf16(
                ws.y.data(), Lw.mlp_norm_pre->data(), ws.norm_x.data(),
                N, H, eps, stream);
        });
        profile_forward_stage(profile, profile.mlp_ms, stream, [&] {
            qwen35_dense_mlp_block(Lw, cfg, fwd_cfg, ws, cublas, N, stream);
        });
    }

    // 3. Final norm.
    profile_forward_stage(profile, profile.final_norm_ms, stream, [&] {
        kernels::launch_rmsnorm_gemma_bf16(
            ws.y.data(), w.final_norm->data(), ws.norm_x.data(),
            N, H, eps, stream);
    });

    // 4. lm_head. For prompt/prefill-style fires the runtime may need
    // logits for only a small sampler subset. Keep the full hidden stream in
    // ws.y for MTP/state plumbing, but feed lm_head from gathered rows.
    profile_forward_stage(profile, profile.lm_head_ms, stream, [&] {
        if (logit_row_indices_d != nullptr &&
            num_logit_rows > 0 &&
            num_logit_rows < N) {
            kernels::launch_gather_bf16_rows(
                static_cast<const std::uint16_t*>(ws.norm_x.data()),
                logit_row_indices_d,
                static_cast<std::uint16_t*>(ws.norm_y.data()),
                num_logit_rows, H, stream);
            ops::gemm_act_x_w(cublas.handle(),
                ws.norm_y.data(), *w.lm_head,
                ws.logits.data(), num_logit_rows, V, H);
        } else {
            ops::gemm_act_x_w(cublas.handle(),
                ws.norm_x.data(), *w.lm_head,
                ws.logits.data(), N, V, H);
        }
    });
    profile_forward_stage(profile, profile.final_copy_ms, stream, [&] {
        CUDA_CHECK(cudaMemcpyAsync(
            ws.y.data(), ws.norm_x.data(),
            static_cast<std::size_t>(N) * H * sizeof(std::uint16_t),
            cudaMemcpyDeviceToDevice, stream));
    });
    profile.end(stream);
    maybe_print_forward_profile(profile);
}

namespace {

void mtp_full_attn_no_cache(
    const Qwen3_5LayerWeights& Lw,
    const HfConfig& cfg,
    const Qwen3_5ForwardCfg& fwd_cfg,
    Qwen3Workspace& ws,
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
    const std::size_t kv_step_offset =
        static_cast<std::size_t>(draft_step) * N * Hk;
    auto* k_step = static_cast<std::uint16_t*>(ws.k.data()) + kv_step_offset;
    auto* v_step = static_cast<std::uint16_t*>(ws.v.data()) + kv_step_offset;

    const int qgkv_dim = 2 * Hq + 2 * Hk;
    const bool use_fused_qgkv =
        Lw.fa_qgkv_proj_fused != nullptr &&
        !ws.gate_up_fused.empty() &&
        ws.gate_up_fused.numel() >= static_cast<std::size_t>(N) * qgkv_dim;
    if (use_fused_qgkv) {
        ops::gemm_act_x_w(cublas.handle(),
            ws.norm_x.data(), ops::WeightView(*Lw.fa_qgkv_proj_fused),
            ws.gate_up_fused.data(), N, qgkv_dim, H);
        kernels::launch_split_qkv_bf16(
            ws.gate_up_fused.data(),
            la.fa_qg_packed.data(), k_step, v_step,
            N, 2 * Hq, Hk, stream);
    } else {
        ops::gemm_act_x_w(cublas.handle(),
            ws.norm_x.data(), make_weight_view(Lw.fa_q_proj, Lw.fa_q_proj_quant),
            la.fa_qg_packed.data(), N, 2 * Hq, H);
        ops::gemm_act_x_w(cublas.handle(),
            ws.norm_x.data(), make_weight_view(Lw.fa_v_proj, Lw.fa_v_proj_quant),
            v_step, N, Hk, H);
        ops::gemm_act_x_w(cublas.handle(),
            ws.norm_x.data(), make_weight_view(Lw.fa_k_proj, Lw.fa_k_proj_quant),
            k_step, N, Hk, H);
    }
    kernels::launch_split_q_gate_bf16(
        la.fa_qg_packed.data(), ws.q.data(), la.fa_gate.data(),
        N, q_heads, cfg.head_dim, stream);

    kernels::launch_rmsnorm_gemma_bf16(
        ws.q.data(), Lw.fa_q_norm->data(), ws.q.data(),
        N * q_heads, d, eps, stream);
    kernels::launch_rmsnorm_gemma_bf16(
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
    kernels::launch_sigmoid_gate_inplace_bf16(
        ws.attn_out.data(), la.fa_gate.data(), N * Hq, stream);

    NcclComm* tp = (T > 1) ? fwd_cfg.tp_comm : nullptr;
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

}  // namespace

void qwen3_5_mtp_process_cache(
    const Qwen3_5Weights& w,
    const HfConfig& cfg,
    const Qwen3_5ForwardCfg& fwd_cfg,
    Qwen3Workspace& ws,
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
    kernels::launch_rmsnorm_gemma_bf16(
        ws.norm_x.data(), mtp.pre_fc_norm_embedding->data(), ws.q.data(),
        total_tokens, H, eps, stream);
    kernels::launch_rmsnorm_gemma_bf16(
        ws.norm_y.data(), mtp.pre_fc_norm_hidden->data(), ws.attn_out.data(),
        total_tokens, H, eps, stream);
    kernels::launch_concat_bf16_rows(
        ws.q.data(), ws.attn_out.data(), ws.mtp_concat.data(),
        total_tokens, H, H, stream);
    ops::gemm_act_x_w(cublas.handle(),
        ws.mtp_concat.data(), *mtp.fc, ws.norm_y.data(),
        total_tokens, H, 2 * H);
    kernels::launch_rmsnorm_gemma_bf16(
        ws.norm_y.data(), Lw.attn_norm_pre->data(), ws.norm_x.data(),
        total_tokens, H, eps, stream);
    ops::gemm_act_x_w(cublas.handle(),
        ws.norm_x.data(), make_weight_view(Lw.fa_k_proj, Lw.fa_k_proj_quant),
        ws.k.data(), total_tokens, Hk, H);
    ops::gemm_act_x_w(cublas.handle(),
        ws.norm_x.data(), make_weight_view(Lw.fa_v_proj, Lw.fa_v_proj_quant),
        ws.v.data(), total_tokens, Hk, H);
    kernels::launch_rmsnorm_gemma_bf16(
        ws.k.data(), Lw.fa_k_norm->data(), ws.k.data(),
        total_tokens * kv_heads, d, eps, stream);
    const bool shifted_prefix_cache = qwen35_mtp_shift_prefix_cache();
    if (shifted_prefix_cache) {
        kernels::launch_rope_partial_bf16_position_delta(
            /*q=*/nullptr, ws.k.data(), positions, -1,
            total_tokens, 0, kv_heads, d, rotary_dim, cfg.rope_theta, stream);
        kernels::launch_write_kv_to_pages_at_positions_bf16(
            cache.layer_view(Lw.kv_layer),
            ws.k.data(), ws.v.data(), positions, -1,
            qo_indptr, kv_page_indices, kv_page_indptr,
            total_tokens, num_requests, stream);
    } else {
        kernels::launch_rope_partial_bf16(
            /*q=*/nullptr, ws.k.data(), positions,
            total_tokens, 0, kv_heads, d, rotary_dim, cfg.rope_theta, stream);
        kernels::launch_write_kv_to_pages(
            cache.layer_view(Lw.kv_layer),
            ws.k.data(), ws.v.data(), qo_indptr, kv_page_indices,
            kv_page_indptr, kv_last_page_lens, total_tokens, num_requests,
            stream);
    }
}

void qwen3_5_mtp_forward(
    const Qwen3_5Weights& w,
    const HfConfig& cfg,
    const Qwen3_5ForwardCfg& fwd_cfg,
    Qwen3Workspace& ws,
    Qwen3_5LinearAttnWorkspace& la_ws,
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
    kernels::launch_rmsnorm_gemma_bf16(
        ws.norm_x.data(), mtp.pre_fc_norm_embedding->data(), ws.q.data(),
        num_tokens, H, eps, stream);
    kernels::launch_rmsnorm_gemma_bf16(
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
    kernels::launch_rmsnorm_gemma_bf16(
        ws.y.data(), Lw.attn_norm_pre->data(), ws.norm_x.data(),
        num_tokens, H, eps, stream);
    mtp_full_attn_no_cache(
        Lw, cfg, fwd_cfg, ws, la_ws, cache, num_tokens, draft_step,
        position_ids, request_ids, kv_page_indices, kv_page_indptr,
        kv_last_page_lens, max_global_tokens, cublas, stream);
    });

    profile_mtp_stage(profile, profile.mlp_ms, stream, [&] {
    kernels::launch_rmsnorm_gemma_bf16(
        ws.y.data(), Lw.mlp_norm_pre->data(), ws.norm_x.data(),
        num_tokens, H, eps, stream);
    qwen35_dense_mlp_block(
        Lw, cfg, fwd_cfg, ws, cublas, num_tokens, stream);
    });

    profile_mtp_stage(profile, profile.lm_head_ms, stream, [&] {
    kernels::launch_rmsnorm_gemma_bf16(
        ws.y.data(), mtp.norm->data(), ws.norm_x.data(),
        num_tokens, H, eps, stream);
    ops::WeightView lm_head(*w.lm_head);
    if (mtp.lm_head != nullptr) {
        lm_head = ops::WeightView(*mtp.lm_head);
    }
    if (sampled_token_ids != nullptr &&
        mtp.lm_head_scale_inv != nullptr &&
        lm_head.dtype == DType::INT8) {
        kernels::launch_lm_head_gemv_argmax_int8(
            ws.norm_x.data(),
            static_cast<const std::int8_t*>(lm_head.data),
            mtp.lm_head_scale_inv->data(),
            sampled_token_ids, num_tokens, H, V, stream);
    } else if (sampled_token_ids != nullptr &&
               qwen35_mtp_fused_gemv_enabled() &&
               lm_head.dtype == DType::BF16) {
        kernels::launch_lm_head_gemv_argmax_bf16(
            ws.norm_x.data(), lm_head.data,
            sampled_token_ids, num_tokens, H, V, stream);
    } else {
        ops::maybe_bench_lm_head_algos(cublas.handle(),
            ws.norm_x.data(), lm_head.data, ws.logits.data(),
            num_tokens, V, H);
        ops::gemm_act_x_w(cublas.handle(),
            ws.norm_x.data(), lm_head,
            ws.logits.data(), num_tokens, V, H);
    }
    CUDA_CHECK(cudaMemcpyAsync(
        ws.y.data(), ws.norm_x.data(),
        static_cast<std::size_t>(num_tokens) * H * sizeof(std::uint16_t),
        cudaMemcpyDeviceToDevice, stream));
    });
    profile.end(stream);
    maybe_print_mtp_profile(profile);
}

std::size_t qwen3_5_la_workspace_bytes(const HfConfig& cfg,
                                       int N, int tp_size) {
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
    bytes += u16(n * conv_dim);
    bytes += u16(n * (conv_dim + v_dim));
    bytes += u16(n * 2 * v_h);
    bytes += u16(n * conv_dim);
    bytes += u16(n * v_dim);
    bytes += u16(n * v_h);
    bytes += u16(n * v_h);
    bytes += fp32(n * v_h * cfg.linear_key_head_dim);
    bytes += fp32(n * v_h * cfg.linear_key_head_dim);
    bytes += fp32(n * v_dim);
    bytes += fp32(n * v_h);
    bytes += fp32(n * v_h);
    bytes += fp32(n * v_dim);
    bytes += u16(n * v_dim);
    bytes += u16(n * k_dim);
    bytes += u16(n * k_dim);
    bytes += u16(n * v_dim);
    bytes += fp32(n * k_h * cfg.linear_key_head_dim);
    bytes += fp32(n * k_h * cfg.linear_key_head_dim);
    bytes += u16(n * 2 * hq);
    bytes += u16(n * hq);
    return bytes;
}

}  // namespace pie_cuda_driver::model
