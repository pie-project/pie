#include "model/kimi_k3/kimi_k3_forward.hpp"

#include "model/act_dump.hpp"

#include <algorithm>
#include <cmath>
#include <map>
#include <string>
#include <utility>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include <cuda_runtime.h>

#include "cuda_check.hpp"
#include "kernels/attn_res.hpp"
#include "kernels/causal_conv1d.hpp"
#include "kernels/dequant_fp4.hpp"
#include "kernels/dequant_wna16.hpp"
#include "kernels/embed.hpp"
#include "kernels/gated_delta_net.hpp"
#include "kernels/gather_rows.hpp"
#include "kernels/kda.hpp"
#include "kernels/kimi_mla.hpp"
#include "kernels/mla_paged.hpp"
#include "kernels/moe_dispatch.hpp"
#include "kernels/residual_add.hpp"
#include "kernels/rmsnorm.hpp"
#include "kernels/swiglu.hpp"
#include "kernels/topk_softmax.hpp"
#include "ops/flashinfer_moe.hpp"

namespace pie_cuda_driver::model {

int kimi_k3_num_attn_res_blocks(const HfConfig& cfg) {
    if (cfg.attn_res_block_size <= 0) return 0;
    // One block per layer index divisible by the stride, i.e. ceil(L / bs).
    return (cfg.num_hidden_layers + cfg.attn_res_block_size - 1) /
           cfg.attn_res_block_size;
}

namespace {

// The routed decode GEMV wins only at a single token: one warp per output row
// doing scalar math, against a batched GEMM on tensor cores. Override with
// `PIE_MOE_GEMV_MAX_TOKENS`.
constexpr int kKimiK3MoeGemvMaxTokens = 1;
// The four-bit GEMV reads a quarter of the bytes the bf16 one does for the
// same arithmetic, so the point where the batched GEMM's tiling starts to win
// sits further out. Verified against vLLM at this token count.
constexpr int kKimiK3MoeMxfp4MaxTokens = 8;

int local_or_zero(int total, int tp) { return total > 0 ? total / tp : 0; }

/// Turn a sticky launch error into an exception that names where it happened.
/// A bad launch is otherwise reported by whatever synchronises next, which on
/// this path is the sampler epilogue -- a stage that has nothing to do with it.
void check_launch(const char* where, int layer) {
    const cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(
            std::string("kimi_k3: ") + where + " (layer " +
            std::to_string(layer) + "): " + cudaGetErrorString(err));
    }
}

/// Per-phase GPU timing, off unless `PIE_K3_PHASE_PROFILE` is set.
///
/// `PIE_STEP_PROFILE` gives the whole forward and `act_dump` gives values;
/// neither says which phase owns the milliseconds. Events are recorded on the
/// forward's own stream and read only after the last one has completed, so an
/// enabled run reorders nothing and a disabled one is a load and a branch.
class PhaseProfiler {
  public:
    static constexpr bool enabled() { return false; }

    // Decode fires are captured into a graph, and a capturing stream may not
    // be synchronised -- `report` would turn a profiling run into a failed
    // load. Prefill, which is what this is for, is never captured.
    static bool capturing(cudaStream_t stream) {
        cudaStreamCaptureStatus status = cudaStreamCaptureStatusNone;
        return cudaStreamIsCapturing(stream, &status) == cudaSuccess &&
               status != cudaStreamCaptureStatusNone;
    }

    void begin(const char* name, cudaStream_t stream) {
        if (!enabled() || capturing(stream)) return;
        Slot& slot = slot_for(name);
        cudaEventRecord(slot.start, stream);
        open_ = &slot;
    }

    void end(cudaStream_t stream) {
        if (!enabled() || open_ == nullptr || capturing(stream)) return;
        cudaEventRecord(open_->stop, stream);
        pending_.push_back(open_);
        open_ = nullptr;
    }

    void report(int tokens) {
        if (!enabled() || pending_.empty()) return;
        for (Slot* s : pending_) {
            cudaEventSynchronize(s->stop);
            float ms = 0.f;
            cudaEventElapsedTime(&ms, s->start, s->stop);
            s->total += ms;
        }
        pending_.clear();
        std::vector<std::pair<std::string, float>> rows;
        float total = 0.f;
        for (auto& entry : slots_) {
            if (entry.second.total > 0.f) {
                rows.emplace_back(entry.first, entry.second.total);
                total += entry.second.total;
            }
            entry.second.total = 0.f;
        }
        std::sort(rows.begin(), rows.end(),
                  [](const std::pair<std::string, float>& a,
                     const std::pair<std::string, float>& b) {
                      return a.second > b.second;
                  });
        std::fprintf(stderr, "[k3-phase] tokens=%d total=%.2f ms\n", tokens, total);
        for (const auto& row : rows) {
            std::fprintf(stderr, "[k3-phase]   %-18s %7.2f ms  %5.1f%%\n",
                         row.first.c_str(), row.second,
                         100.0 * row.second / (total > 0.f ? total : 1.f));
        }
        std::fflush(stderr);
    }

  private:
    struct Slot {
        cudaEvent_t start = nullptr;
        cudaEvent_t stop = nullptr;
        float total = 0.f;
    };
    // One event pair per phase *name*, reused across layers: a phase that
    // appears in six layers accumulates six intervals into one row, which is
    // the number worth reading.
    Slot& slot_for(const char* name) {
        auto it = slots_.find(name);
        if (it == slots_.end()) {
            Slot s;
            cudaEventCreate(&s.start);
            cudaEventCreate(&s.stop);
            it = slots_.emplace(name, s).first;
        }
        return it->second;
    }
    std::map<std::string, Slot> slots_;
    std::vector<Slot*> pending_;
    Slot* open_ = nullptr;
};

/// Scoped begin/end, so a phase cannot be left open by an early return.
struct PhaseScope {
    PhaseScope(PhaseProfiler& p, const char* name, cudaStream_t s)
        : prof(p), stream(s) { prof.begin(name, s); }
    ~PhaseScope() { prof.end(stream); }
    PhaseProfiler& prof;
    cudaStream_t stream;
};

}  // namespace

KimiK3Workspace KimiK3Workspace::allocate(
    const HfConfig& cfg, int max_tokens, int max_logit_rows, int tp_size)
{
    const int T = std::max(1, tp_size);
    const int N = std::max(1, max_tokens);
    const int O = std::max(1, max_logit_rows > 0 ? max_logit_rows : max_tokens);
    const int H = cfg.hidden_size;
    const int mla_heads = cfg.num_attention_heads / T;
    const int q_nope = cfg.qk_nope_head_dim;
    const int q_rope = cfg.qk_rope_head_dim;
    const int v_dim = cfg.v_head_dim;
    const int q_lora = cfg.q_lora_rank;
    const int kv_lora = cfg.kv_lora_rank;
    const int kda_heads = cfg.linear_num_value_heads / T;
    const int kda_dim = cfg.linear_value_head_dim;
    const int W = kda_heads * kda_dim;
    const int latent = cfg.routed_expert_hidden_size > 0
                           ? cfg.routed_expert_hidden_size : H;
    const int dense_I = local_or_zero(cfg.intermediate_size, T);
    const int routed_I = local_or_zero(cfg.moe_intermediate_size, T);
    const int shared_I = local_or_zero(cfg.shared_expert_intermediate_size, T);
    const int max_I = std::max(1, std::max(dense_I, shared_I));
    const int Ktop = std::max(1, cfg.num_experts_per_tok);
    const int routes = N * Ktop;

    if (H <= 0 || mla_heads <= 0 || kda_heads <= 0 || kda_dim <= 0 ||
        q_lora <= 0 || kv_lora <= 0 || latent <= 0) {
        throw std::runtime_error(
            "kimi_k3: cannot allocate workspace with unset dimensions");
    }

    KimiK3Workspace ws;
    ws.num_blocks = kimi_k3_num_attn_res_blocks(cfg);
    ws.y        = DeviceTensor::allocate(DType::BF16, {N, H});
    ws.hidden   = DeviceTensor::allocate(DType::BF16, {N, H});
    ws.norm_x   = DeviceTensor::allocate(DType::BF16, {N, H});
    ws.norm_y   = DeviceTensor::allocate(DType::BF16, {N, H});
    ws.attn_out = DeviceTensor::allocate(DType::BF16, {N, H});
    ws.mlp_out  = DeviceTensor::allocate(DType::BF16, {N, H});
    if (ws.num_blocks > 0) {
        // Block-major, so pushing a block is one contiguous [N, H] write.
        ws.blocks = DeviceTensor::allocate(DType::BF16, {ws.num_blocks, N, H});
    }

    ws.q_a           = DeviceTensor::allocate(DType::BF16, {N, q_lora});
    ws.q_b           = DeviceTensor::allocate(DType::BF16, {N, mla_heads * (q_nope + q_rope)});
    ws.q_nope        = DeviceTensor::allocate(DType::BF16, {N, mla_heads * q_nope});
    ws.q_pe          = DeviceTensor::allocate(DType::BF16, {N, mla_heads * q_rope});
    ws.kv_a_mqa      = DeviceTensor::allocate(DType::BF16, {N, kv_lora + q_rope});
    ws.kv_c          = DeviceTensor::allocate(DType::BF16, {N, kv_lora});
    ws.k_pe          = DeviceTensor::allocate(DType::BF16, {N, q_rope});
    ws.q_nope_latent = DeviceTensor::allocate(DType::BF16, {N, mla_heads * kv_lora});
    ws.attn_latent   = DeviceTensor::allocate(DType::BF16, {N, mla_heads * kv_lora});
    ws.attn_v        = DeviceTensor::allocate(DType::BF16, {N, mla_heads * v_dim});
    if (cfg.mla_output_gate) {
        ws.mla_gate = DeviceTensor::allocate(DType::BF16, {N, mla_heads * v_dim});
    }

    ws.kda_q        = DeviceTensor::allocate(DType::BF16, {N, W});
    ws.kda_k        = DeviceTensor::allocate(DType::BF16, {N, W});
    ws.kda_v        = DeviceTensor::allocate(DType::BF16, {N, W});
    ws.kda_qc       = DeviceTensor::allocate(DType::BF16, {N, W});
    ws.kda_kc       = DeviceTensor::allocate(DType::BF16, {N, W});
    ws.kda_vc       = DeviceTensor::allocate(DType::BF16, {N, W});
    ws.kda_g_raw    = DeviceTensor::allocate(DType::BF16, {N, W});
    ws.kda_f_a      = DeviceTensor::allocate(DType::BF16, {N, kda_dim});
    ws.kda_beta_raw = DeviceTensor::allocate(DType::BF16, {N, kda_heads});
    ws.kda_gproj    = DeviceTensor::allocate(DType::BF16, {N, W});
    ws.kda_gate     = DeviceTensor::allocate(DType::FP32, {N, W});
    ws.kda_beta     = DeviceTensor::allocate(DType::FP32, {N, kda_heads});
    ws.kda_qf       = DeviceTensor::allocate(DType::FP32, {N, W});
    ws.kda_kf       = DeviceTensor::allocate(DType::FP32, {N, W});
    ws.kda_vf       = DeviceTensor::allocate(DType::FP32, {N, W});
    ws.kda_o        = DeviceTensor::allocate(DType::FP32, {N, W});

    ws.gate       = DeviceTensor::allocate(DType::BF16, {N, max_I});
    ws.up         = DeviceTensor::allocate(DType::BF16, {N, max_I});
    ws.latent_in  = DeviceTensor::allocate(DType::BF16, {N, latent});
    ws.latent_out = DeviceTensor::allocate(DType::BF16, {N, latent});
    ws.router_logits = DeviceTensor::allocate(DType::BF16, {N, std::max(1, cfg.num_experts)});
    ws.topk_idx      = DeviceTensor::allocate(DType::INT32, {N, Ktop});
    ws.topk_weights  = DeviceTensor::allocate(DType::FP32, {N, Ktop});
    ws.expert_out    = DeviceTensor::allocate(DType::BF16, {routes, latent});
    ws.shared_out    = DeviceTensor::allocate(DType::BF16, {N, H});

    if (routed_I > 0 && cfg.num_experts > 0) {
        const int block = kernels::moe_aligned_block(routes, cfg.num_experts);
        const int active_expert_cap = std::min(cfg.num_experts, routes);
        const int max_blocks =
            (routes + active_expert_cap * (kernels::kMoeAlignedBlockMin - 1) +
             kernels::kMoeAlignedBlockMin - 1) /
            kernels::kMoeAlignedBlockMin;
        const int aligned_rows =
            std::max(max_blocks * kernels::kMoeAlignedBlockMin,
                     ((routes + active_expert_cap * (block - 1) + block - 1) /
                      block) * block);
        ws.aligned_block_size = block;
        ws.aligned_max_blocks = max_blocks;
        ws.aligned_route_ids  = DeviceTensor::allocate(DType::INT32, {aligned_rows});
        ws.aligned_expert_ids = DeviceTensor::allocate(DType::INT32, {max_blocks});
        ws.aligned_expert_in  = DeviceTensor::allocate(DType::BF16, {aligned_rows, latent});
        ws.aligned_gate_up    = DeviceTensor::allocate(DType::BF16, {aligned_rows, 2 * routed_I});
        ws.aligned_act        = DeviceTensor::allocate(DType::BF16, {aligned_rows, routed_I});
        ws.aligned_out        = DeviceTensor::allocate(DType::BF16, {aligned_rows, latent});
        // Only the decode GEMV reads these, so they are sized off its token
        // cap rather than `max_tokens`; at the cap this is a few MB.
        const std::int64_t gemv_n =
            std::min<std::int64_t>(max_tokens,
                                   ops::moe_gemv_max_tokens(kKimiK3MoeMxfp4MaxTokens));
        if (gemv_n > 0) {
            ws.mxfp4_act_fp16 =
                DeviceTensor::allocate(DType::FP16, {gemv_n, latent});
            ws.mxfp4_route_act_fp16 =
                DeviceTensor::allocate(DType::FP16, {gemv_n * Ktop, routed_I});
        }
        const std::int64_t ptrs = max_blocks;
        for (DeviceTensor* p : {&ws.a_gu_ptrs, &ws.b_gu_ptrs, &ws.c_gu_ptrs,
                                &ws.a_dn_ptrs, &ws.b_dn_ptrs, &ws.c_dn_ptrs}) {
            *p = DeviceTensor::allocate(DType::INT64, {ptrs});
        }
    }

    ws.logits = DeviceTensor::allocate(DType::BF16, {O, cfg.vocab_size});
    return ws;
}

std::size_t kimi_k3_workspace_bytes(
    const HfConfig& cfg, int max_tokens, int max_logit_rows, int tp_size)
{
    const int T = std::max(1, tp_size);
    const std::size_t N = static_cast<std::size_t>(std::max(1, max_tokens));
    const std::size_t H = static_cast<std::size_t>(cfg.hidden_size);
    const std::size_t blocks =
        static_cast<std::size_t>(kimi_k3_num_attn_res_blocks(cfg));
    const std::size_t kda_w =
        static_cast<std::size_t>(cfg.linear_num_value_heads / T) *
        static_cast<std::size_t>(cfg.linear_value_head_dim);
    const std::size_t latent = static_cast<std::size_t>(
        cfg.routed_expert_hidden_size > 0 ? cfg.routed_expert_hidden_size
                                          : cfg.hidden_size);
    // Rows of [N, H] bf16: y, hidden, norm_x, norm_y, attn_out, mlp_out,
    // shared_out, plus the block stack.
    std::size_t bytes = (7 + blocks) * N * H * 2;
    // KDA: seven [N, W] bf16 and six [N, W] fp32.
    bytes += N * kda_w * (7 * 2 + 6 * 4);
    // MoE latent scratch, generously: the aligned path's worst case is
    // `routes` rows of `latent` plus the gate/up pair at the routed width.
    const std::size_t routes = N * static_cast<std::size_t>(
        std::max(1, cfg.num_experts_per_tok));
    bytes += routes * latent * 2 * 3;
    bytes += routes * static_cast<std::size_t>(
        std::max(1, cfg.moe_intermediate_size / T)) * 2 * 3;
    bytes += N * static_cast<std::size_t>(cfg.vocab_size) * 2;
    // MLA scratch and the odds and ends; 25% covers them without a
    // per-tensor tally that would drift from `allocate` above.
    return bytes + bytes / 4;
}

void kimi_k3_forward_paged(
    const KimiK3Weights& w,
    const HfConfig& cfg,
    const KimiK3ForwardCfg& fwd_cfg,
    KimiPlanState& mla_plan,
    KimiK3Workspace& ws,
    MlaCache& mla_cache,
    RecurrentStateCache& kda_cache,
    AttentionWorkspace& attn_ws,
    ops::CublasHandle& cublas,
    void* logits_out,
    const std::int32_t* token_ids,
    const std::int32_t* /*positions*/,
    const std::uint32_t* qo_indptr,
    const std::uint32_t* kv_page_indices,
    const std::uint32_t* kv_page_indptr,
    const std::uint32_t* kv_last_page_lens,
    const std::uint32_t* /*qo_indptr_h*/,
    const std::uint32_t* /*kv_page_indptr_h*/,
    const std::int32_t*  kda_slot_ids_h,
    const std::int32_t*  kda_slot_ids_d,
    const std::uint8_t*  kda_is_fresh_h,
    const std::uint8_t*  kda_is_fresh_d,
    int total_tokens,
    int num_requests,
    bool is_pure_decode,
    const std::uint8_t* row_valid_d,
    const std::int32_t* logit_row_indices_d,
    int num_logit_rows)
{
    // The executor's stream, never the legacy default: ordering against the
    // epilogue depends on it, and a launch on stream 0 is illegal inside a
    // graph capture.
    cudaStream_t stream = cublas.stream();
    const int T = std::max(1, fwd_cfg.tp_size);
    NcclComm* tp = fwd_cfg.tp_comm;
    const int H = cfg.hidden_size;
    const int V = cfg.vocab_size;
    const float eps = cfg.rms_norm_eps;

    const int mla_heads = cfg.num_attention_heads / T;
    const int q_nope = cfg.qk_nope_head_dim;
    const int q_rope = cfg.qk_rope_head_dim;
    const int v_dim = cfg.v_head_dim;
    const int q_lora = cfg.q_lora_rank;
    const int kv_lora = cfg.kv_lora_rank;

    const int kda_heads = cfg.linear_num_value_heads / T;
    const int kda_dim = cfg.linear_value_head_dim;
    const int W = kda_heads * kda_dim;
    const int conv_k = cfg.linear_conv_kernel_dim;
    const float kda_scale = 1.f / std::sqrt(static_cast<float>(kda_dim));

    const int latent = cfg.routed_expert_hidden_size > 0
                           ? cfg.routed_expert_hidden_size : H;
    const int dense_I = local_or_zero(cfg.intermediate_size, T);
    const int routed_I = local_or_zero(cfg.moe_intermediate_size, T);
    const int shared_I = local_or_zero(cfg.shared_expert_intermediate_size, T);
    const int E = cfg.num_experts;
    const int K = cfg.num_experts_per_tok;
    const int bs = cfg.attn_res_block_size;

    const std::size_t rows_H = static_cast<std::size_t>(total_tokens) * H;
    // The AttnRes stack is allocated `[num_blocks, max_tokens, H]`, so a block
    // is `max_tokens * H` apart -- not `total_tokens * H`, which is only the
    // live part of each block.
    const int block_rows =
        ws.blocks.empty() ? 0 : static_cast<int>(ws.blocks.shape()[1]);
    const std::size_t block_stride =
        static_cast<std::size_t>(block_rows) * static_cast<std::size_t>(H);

    // ── Embed ────────────────────────────────────────────────────────
    if (w.embed_tp_sharded) {
        kernels::launch_embed_bf16_vocab_shard(
            token_ids, w.embed->data(), ws.y.data(), total_tokens, H,
            static_cast<int>(w.embed->shape()[0]), w.embed_tp_vocab_offset,
            stream);
        if (T > 1 && tp != nullptr) {
            tp->all_reduce_bf16(ws.y.data(), rows_H, ncclSum, stream);
        }
    } else {
        kernels::launch_embed_bf16(token_ids, w.embed->data(), ws.y.data(),
                                   total_tokens, H, cfg.vocab_size, stream);
    }

    act_dump_step_begin(stream);
    act_dump_bf16("embed", ws.y.data(), total_tokens, H, stream);

    if (kda_slot_ids_d == nullptr) {
        throw std::runtime_error(
            "kimi_k3: KDA layers need per-request recurrent-state slots");
    }

    // A fresh request's KDA state has to start at zero. Nothing else zeroes
    // it -- the slab is arena memory the cache hands out as-is -- and the
    // delta rule reads the state before it writes any of it, so skipping this
    // does not degrade the output, it produces NaN from the first token.
    if (kda_slot_ids_h != nullptr && kda_is_fresh_h != nullptr) {
        if (std::any_of(kda_is_fresh_h, kda_is_fresh_h + num_requests,
                        [](auto fresh) { return fresh != 0; })) {
            if (kda_is_fresh_d != nullptr) {
                kda_cache.reset_slots_if_fresh(kda_slot_ids_d, kda_is_fresh_d,
                                               num_requests, stream);
            } else {
                for (int r = 0; r < num_requests; ++r) {
                    if (kda_is_fresh_h[r]) {
                        kda_cache.reset_slot(kda_slot_ids_h[r], stream);
                    }
                }
            }
        }
    } else if (!is_pure_decode) {
        kda_cache.reset(stream);
    }

    int open_blocks = 0;

    PhaseProfiler prof;
    for (int li = 0; li < cfg.num_hidden_layers; ++li) {
        const auto& Lw = w.layers[static_cast<std::size_t>(li)];

        // ── AttnRes: blend, then maybe open a block ─────────────────
        const void* layer_in = ws.y.data();
        if (bs > 0 && open_blocks > 0) {
            kernels::launch_attn_res_blend_bf16(
                ws.y.data(), ws.blocks.data(), Lw.attn_res_norm->data(),
                Lw.attn_res_proj->data(), ws.hidden.data(),
                total_tokens, open_blocks, H, block_rows, eps, stream);
            layer_in = ws.hidden.data();
        }
        bool prefix_zeroed = false;
        if (bs > 0 && li % bs == 0) {
            // The value pushed is the *unblended* running sum, which is why
            // this reads `y` and not `layer_in`.
            CUDA_CHECK(cudaMemcpyAsync(
                static_cast<std::uint16_t*>(ws.blocks.data()) +
                    static_cast<std::size_t>(open_blocks) * block_stride,
                ws.y.data(), rows_H * sizeof(std::uint16_t),
                cudaMemcpyDeviceToDevice, stream));
            ++open_blocks;
            prefix_zeroed = true;
        }

        kernels::launch_rmsnorm_bf16(layer_in, Lw.attn_norm->data(),
                                     ws.norm_x.data(), total_tokens, H, eps,
                                     stream);
        act_dump_bf16(act_dump_layer_tag("norm_x", li).c_str(),
                      ws.norm_x.data(), total_tokens, H, stream);

        // ── Attention ──────────────────────────────────────────────
        if (Lw.kind == KimiK3LayerWeights::Kind::Kda) {
            prof.begin("kda_qkv_proj", stream);
            ops::gemm_act_x_w(cublas.handle(), ws.norm_x.data(), *Lw.kda_q_proj,
                              ws.kda_q.data(), total_tokens, W, H);
            ops::gemm_act_x_w(cublas.handle(), ws.norm_x.data(), *Lw.kda_k_proj,
                              ws.kda_k.data(), total_tokens, W, H);
            ops::gemm_act_x_w(cublas.handle(), ws.norm_x.data(), *Lw.kda_v_proj,
                              ws.kda_v.data(), total_tokens, W, H);
            prof.end(stream);

            prof.begin("kda_conv_gate", stream);
            // Depthwise causal conv with fused silu, one per projection. The
            // three conv states live side by side in the slot's conv slab.
            // `RecurrentStateCache` indexes by the *model* layer, mapping to
            // its own compacted slot internally -- `Lw.kda_layer` is already
            // compacted, and passing it here would silently read another
            // layer's state.
            auto* conv_base = static_cast<std::uint16_t*>(kda_cache.conv_state(li));
            const long long conv_slot_stride =
                static_cast<long long>(kda_cache.conv_slot_stride_bytes() /
                                       sizeof(std::uint16_t));
            const long long conv_part = static_cast<long long>(conv_k) * W;
            // `x` and `y` must not alias: every output position reads a
            // K-wide window of the input, and a neighbouring thread would read
            // an element this one has already overwritten.
            struct ConvPart {
                DeviceTensor* in;
                DeviceTensor* out;
                const DeviceTensor* weight;
                long long offset;
            };
            const ConvPart parts[3] = {
                {&ws.kda_q, &ws.kda_qc, Lw.kda_q_conv1d, 0},
                {&ws.kda_k, &ws.kda_kc, Lw.kda_k_conv1d, conv_part},
                {&ws.kda_v, &ws.kda_vc, Lw.kda_v_conv1d, 2 * conv_part},
            };
            for (const ConvPart& p : parts) {
                void* state = conv_base + p.offset;
                if (is_pure_decode) {
                    kernels::launch_causal_conv1d_update_batched_bf16(
                        p.in->data(), p.weight->data(), /*bias=*/nullptr,
                        state, kda_slot_ids_d,
                        conv_slot_stride, p.out->data(), num_requests, W, conv_k,
                        stream);
                } else {
                    kernels::launch_causal_conv1d_prefill_batched_bf16(
                        p.in->data(), p.weight->data(), /*bias=*/nullptr,
                        p.out->data(), state,
                        kda_slot_ids_d,
                        qo_indptr,
                        conv_slot_stride, num_requests, W, conv_k, stream);
                }
            }

            act_dump_bf16(act_dump_layer_tag("kda_conv_q", li).c_str(),
                          ws.kda_qc.data(), total_tokens, W, stream);
            act_dump_bf16(act_dump_layer_tag("kda_conv_v", li).c_str(),
                          ws.kda_vc.data(), total_tokens, W, stream);
            // Decay gate: the rank-`head_dim` bottleneck first, then out to
            // every head channel.
            ops::gemm_act_x_w(cublas.handle(), ws.norm_x.data(), *Lw.kda_f_a_proj,
                              ws.kda_f_a.data(), total_tokens, kda_dim, H);
            ops::gemm_act_x_w(cublas.handle(), ws.kda_f_a.data(), *Lw.kda_f_b_proj,
                              ws.kda_g_raw.data(), total_tokens, W, kda_dim);
            ops::gemm_act_x_w(cublas.handle(), ws.norm_x.data(), *Lw.kda_b_proj,
                              ws.kda_beta_raw.data(), total_tokens, kda_heads, H);
            kernels::launch_kda_gate_beta_bf16(
                ws.kda_g_raw.data(), ws.kda_beta_raw.data(),
                Lw.kda_A_log_fp32, Lw.kda_dt_bias_fp32,
                static_cast<float*>(ws.kda_gate.data()),
                static_cast<float*>(ws.kda_beta.data()),
                total_tokens, kda_heads, kda_dim, cfg.kda_gate_lower_bound,
                stream);

            act_dump_bf16(act_dump_layer_tag("kda_g_raw", li).c_str(),
                          ws.kda_g_raw.data(), total_tokens, W, stream);
            act_dump_f32(act_dump_layer_tag("kda_gate", li).c_str(),
                         static_cast<const float*>(ws.kda_gate.data()),
                         total_tokens, W, stream);
            act_dump_f32(act_dump_layer_tag("kda_beta", li).c_str(),
                         static_cast<const float*>(ws.kda_beta.data()),
                         total_tokens, kda_heads, stream);
            // q and k are L2-normalised per head, and q carries the 1/sqrt(D)
            // scale, exactly as the reference kernel does in-place.
            kernels::launch_l2norm_scale_bf16_to_fp32(
                ws.kda_qc.data(), static_cast<float*>(ws.kda_qf.data()),
                total_tokens * kda_heads, kda_dim, kda_scale, 1e-6f, stream);
            kernels::launch_l2norm_scale_bf16_to_fp32(
                ws.kda_kc.data(), static_cast<float*>(ws.kda_kf.data()),
                total_tokens * kda_heads, kda_dim, 1.f, 1e-6f, stream);
            kernels::launch_bf16_to_fp32(
                ws.kda_vc.data(), static_cast<float*>(ws.kda_vf.data()),
                static_cast<std::size_t>(total_tokens) * W, stream);

            const long long rec_stride =
                static_cast<long long>(kda_cache.recurrent_slot_stride_floats());
            // KDA keeps its state in fp32. The reference recurrence is fp32
            // throughout -- the delta rule subtracts the read-back memory from
            // v, so the state carries the small differences the next token
            // reads -- and `RecurrentStateCache` defaults to Qwen3.5's bf16
            // state, which is a different model. Ask for fp32 and say so.
            prof.end(stream);
            if (kda_cache.recurrent_state_is_bf16()) {
                throw std::runtime_error(
                    "kimi_k3: the recurrent state cache is bf16; KDA needs fp32. "
                    "Set PIE_QWEN35_RS_STATE_DTYPE=fp32.");
            }
            float* rec_base =
                static_cast<float*>(kda_cache.recurrent_state_raw(li, /*slot=*/0));
            if (is_pure_decode) {
                kernels::launch_kda_recurrent_step_batched(
                    static_cast<const float*>(ws.kda_qf.data()),
                    static_cast<const float*>(ws.kda_kf.data()),
                    static_cast<const float*>(ws.kda_vf.data()),
                    static_cast<const float*>(ws.kda_gate.data()),
                    static_cast<const float*>(ws.kda_beta.data()),
                    rec_base, kda_slot_ids_d,
                    rec_stride, static_cast<float*>(ws.kda_o.data()),
                    num_requests, kda_heads, kda_dim, stream);
            } else {
                PhaseScope ps(prof, "kda_recurrence", stream);
                kernels::launch_kda_prefill_batched(
                    static_cast<const float*>(ws.kda_qf.data()),
                    static_cast<const float*>(ws.kda_kf.data()),
                    static_cast<const float*>(ws.kda_vf.data()),
                    static_cast<const float*>(ws.kda_gate.data()),
                    static_cast<const float*>(ws.kda_beta.data()),
                    rec_base, kda_slot_ids_d,
                    qo_indptr,
                    rec_stride, static_cast<float*>(ws.kda_o.data()),
                    num_requests, kda_heads, kda_dim, stream);
            }

            act_dump_f32(act_dump_layer_tag("kda_qf", li).c_str(),
                         static_cast<const float*>(ws.kda_qf.data()),
                         total_tokens, W, stream);
            act_dump_f32(act_dump_layer_tag("kda_o", li).c_str(),
                         static_cast<const float*>(ws.kda_o.data()),
                         total_tokens, W, stream);
            prof.begin("kda_gate_out_proj", stream);
            ops::gemm_act_x_w(cublas.handle(), ws.norm_x.data(), *Lw.kda_g_proj,
                              ws.kda_gproj.data(), total_tokens, W, H);
            kernels::launch_kda_o_norm_gated_bf16(
                static_cast<const float*>(ws.kda_o.data()), ws.kda_gproj.data(),
                Lw.kda_o_norm_fp32, ws.kda_q.data(), total_tokens, kda_heads,
                kda_dim, eps, stream);
            act_dump_bf16(act_dump_layer_tag("attn_v", li).c_str(),
                          ws.kda_q.data(), total_tokens, W, stream);
            ops::gemm_act_x_w(cublas.handle(), ws.kda_q.data(), *Lw.kda_o_proj,
                              ws.attn_out.data(), total_tokens, H, W);
            prof.end(stream);
        } else {
            prof.begin("mla", stream);
            ops::gemm_act_x_w(cublas.handle(), ws.norm_x.data(), *Lw.q_a_proj,
                              ws.q_a.data(), total_tokens, q_lora, H);
            ops::gemm_act_x_w(cublas.handle(), ws.norm_x.data(),
                              *Lw.kv_a_proj_with_mqa, ws.kv_a_mqa.data(),
                              total_tokens, kv_lora + q_rope, H);
            kernels::launch_rmsnorm_bf16(ws.q_a.data(), Lw.q_a_norm->data(),
                                         ws.q_a.data(), total_tokens, q_lora,
                                         eps, stream);
            act_dump_bf16(act_dump_layer_tag("q_a", li).c_str(), ws.q_a.data(),
                          total_tokens, q_lora, stream);
            ops::gemm_act_x_w(cublas.handle(), ws.q_a.data(), *Lw.q_b_proj,
                              ws.q_b.data(), total_tokens,
                              mla_heads * (q_nope + q_rope), q_lora);
            act_dump_bf16(act_dump_layer_tag("q_b", li).c_str(), ws.q_b.data(),
                          total_tokens, mla_heads * (q_nope + q_rope), stream);

            kernels::launch_kimi_split_kv_a_norm_bf16(
                ws.kv_a_mqa.data(), Lw.kv_a_norm->data(), ws.kv_c.data(),
                ws.k_pe.data(), total_tokens, kv_lora, q_rope, eps, stream);
            kernels::launch_kimi_split_q_b_bf16(
                ws.q_b.data(), ws.q_nope.data(), ws.q_pe.data(), total_tokens,
                mla_heads, q_nope, q_rope, stream);
            act_dump_bf16(act_dump_layer_tag("kv_c", li).c_str(), ws.kv_c.data(),
                          total_tokens, kv_lora, stream);

            // NoPE: `mla_use_nope` means the rope halves are carried through
            // untouched. There is deliberately no `launch_rope_bf16` here --
            // KDA supplies this model's only position signal.
            auto layer_view = mla_cache.layer_view(Lw.kv_layer);
            kernels::launch_write_mla_to_pages(
                layer_view, ws.kv_c.data(), ws.k_pe.data(), qo_indptr,
                kv_page_indices, kv_page_indptr, kv_last_page_lens,
                total_tokens, num_requests, stream, row_valid_d);

            ops::mla_absorb_q_to_latent_bf16(
                cublas.handle(), ws.q_nope.data(), Lw.kv_b_proj->data(),
                ws.q_nope_latent.data(), total_tokens, mla_heads, q_nope, v_dim,
                kv_lora);
            if (!mla_plan.mla_plan) {
                throw std::runtime_error(
                    "kimi_k3: MLA plan missing; the prepare hook did not run");
            }
            ops::dispatch_attention_mla_bf16(
                *mla_plan.mla_plan, ws.q_nope_latent.data(), ws.q_pe.data(),
                layer_view, ws.attn_latent.data(), kv_page_indices, attn_ws,
                stream, /*lse_out=*/nullptr, qo_indptr, kv_page_indptr,
                kv_last_page_lens);
            ops::mla_absorb_latent_to_v_bf16(
                cublas.handle(), ws.attn_latent.data(), Lw.kv_b_proj->data(),
                ws.attn_v.data(), total_tokens, mla_heads, q_nope, v_dim,
                kv_lora);

            if (cfg.mla_output_gate) {
                ops::gemm_act_x_w(cublas.handle(), ws.norm_x.data(),
                                  *Lw.mla_g_proj, ws.mla_gate.data(),
                                  total_tokens, mla_heads * v_dim, H);
                kernels::launch_sigmoid_gate_inplace_bf16(
                    ws.attn_v.data(), ws.mla_gate.data(),
                    total_tokens * mla_heads * v_dim, stream);
            }
            act_dump_bf16(act_dump_layer_tag("attn_v", li).c_str(),
                          ws.attn_v.data(), total_tokens, mla_heads * v_dim,
                          stream);
            ops::gemm_act_x_w(cublas.handle(), ws.attn_v.data(), *Lw.o_proj,
                              ws.attn_out.data(), total_tokens, H,
                              mla_heads * v_dim);
        }

        check_launch("attention", li);
        if (T > 1 && tp != nullptr) {
            tp->all_reduce_bf16(ws.attn_out.data(), rows_H, ncclSum, stream);
        }
        prof.end(stream);
        act_dump_bf16(act_dump_layer_tag("attn_out", li).c_str(),
                      ws.attn_out.data(), total_tokens, H, stream);

        // `prefix_sum = None` after a push, so the attention output *is* the
        // new running sum rather than an addend.
        if (prefix_zeroed) {
            CUDA_CHECK(cudaMemcpyAsync(ws.y.data(), ws.attn_out.data(),
                                       rows_H * sizeof(std::uint16_t),
                                       cudaMemcpyDeviceToDevice, stream));
        } else {
            kernels::launch_residual_add_bf16(ws.y.data(), ws.attn_out.data(),
                                              rows_H, stream);
        }
        act_dump_bf16(act_dump_layer_tag("post_attn", li).c_str(), ws.y.data(),
                      total_tokens, H, stream);

        // ── AttnRes: blend again before the MLP ────────────────────
        const void* mlp_in = ws.y.data();
        if (bs > 0 && open_blocks > 0) {
            kernels::launch_attn_res_blend_bf16(
                ws.y.data(), ws.blocks.data(), Lw.mlp_res_norm->data(),
                Lw.mlp_res_proj->data(), ws.hidden.data(), total_tokens,
                open_blocks, H, block_rows, eps, stream);
            mlp_in = ws.hidden.data();
        }
        kernels::launch_rmsnorm_bf16(mlp_in, Lw.mlp_norm->data(),
                                     ws.norm_y.data(), total_tokens, H, eps,
                                     stream);

        // ── MLP ────────────────────────────────────────────────────
        prof.begin("mlp", stream);
        if (!Lw.is_moe) {
            ops::gemm_act_x_w(cublas.handle(), ws.norm_y.data(),
                              *Lw.dense_gate_proj, ws.gate.data(), total_tokens,
                              dense_I, H);
            ops::gemm_act_x_w(cublas.handle(), ws.norm_y.data(),
                              *Lw.dense_up_proj, ws.up.data(), total_tokens,
                              dense_I, H);
            kernels::launch_situ_bf16(ws.gate.data(), ws.up.data(),
                                      ws.gate.data(), total_tokens * dense_I,
                                      cfg.situ_beta, cfg.situ_linear_beta,
                                      stream);
            ops::gemm_act_x_w(cublas.handle(), ws.gate.data(),
                              *Lw.dense_down_proj, ws.mlp_out.data(),
                              total_tokens, H, dense_I);
        } else {
            // Router reads the *full-width* hidden, before the latent
            // projection, and scores with sigmoid + noaux_tc.
            ops::gemm_act_x_w(cublas.handle(), ws.norm_y.data(), *Lw.router,
                              ws.router_logits.data(), total_tokens, E, H);
            // noaux_tc: score = sigmoid(logit), selected by
            // `score + e_score_correction_bias`, weighted by the *unbiased*
            // score, renormalised to sum 1. The reference computes the logits
            // in fp32; this GEMM is bf16, which is the one place K3's router
            // is coarser here than upstream.
            kernels::launch_topk_sigmoid_bf16(
                ws.router_logits.data(),
                static_cast<std::int32_t*>(ws.topk_idx.data()),
                static_cast<float*>(ws.topk_weights.data()),
                Lw.e_score_correction_bias != nullptr
                    ? static_cast<const float*>(Lw.e_score_correction_bias->data())
                    : nullptr,
                total_tokens, E, K, cfg.norm_topk_prob,
                cfg.routed_scaling_factor, stream);

            // Down into the latent, where the experts live.
            const void* expert_in = ws.norm_y.data();
            if (Lw.routed_down_proj != nullptr) {
                ops::gemm_act_x_w(cublas.handle(), ws.norm_y.data(),
                                  *Lw.routed_down_proj, ws.latent_in.data(),
                                  total_tokens, latent, H);
                expert_in = ws.latent_in.data();
            }
            act_dump_bf16(act_dump_layer_tag("moe_latent", li).c_str(),
                          expert_in, total_tokens, latent, stream);

            const int routes = total_tokens * K;
            // At one token per request the routed GEMMs are pure streaming
            // reads with no weight reuse, and the aligned path has to pad 8
            // real rows out to a whole block per active expert -- hundreds of
            // rows of arithmetic for eight. One warp per output row skips all
            // of it. Above that token count the batched GEMM's tiling starts
            // paying for itself, which is the same crossover GLM-5 measured.
            const bool gemv_ok =
                total_tokens <= ops::moe_gemv_max_tokens(kKimiK3MoeGemvMaxTokens) &&
                (latent % 8) == 0 && (routed_I % 8) == 0;
            // Four-bit weights if the binder found them. A decode step reads
            // the whole routed bank to produce one token and reuses none of
            // it, so the MoE is pure streaming and the packed form is worth
            // exactly its compression: a quarter of the bytes for the same
            // arithmetic. This is the single largest term in K3's decode.
            const bool mxfp4_ok =
                total_tokens <= ops::moe_gemv_max_tokens(kKimiK3MoeMxfp4MaxTokens) &&
                (latent % 8) == 0 && (routed_I % 8) == 0 &&
                !Lw.expert_gate_up_packed_ptrs.empty() &&
                ws.mxfp4_act_fp16.numel() > 0 &&
                total_tokens <= ws.mxfp4_act_fp16.shape()[0];
            if (mxfp4_ok) {
                kernels::launch_bf16_to_fp16(
                    expert_in, ws.mxfp4_act_fp16.data(),
                    static_cast<std::size_t>(total_tokens) * latent, stream);
                // The two halves land in separate buffers, so this is the
                // plain SiTU rather than the chunked `[gate|up]` one.
                void* gate_out = ws.aligned_gate_up.data();
                void* up_out = static_cast<std::uint16_t*>(gate_out) +
                               static_cast<std::size_t>(routes) * routed_I;
                kernels::launch_mxfp4_moe_gate_up_decode_bf16(
                    ws.mxfp4_act_fp16.data(),
                    static_cast<const std::int32_t*>(ws.topk_idx.data()),
                    Lw.expert_gate_up_packed_ptrs.data(),
                    Lw.expert_gate_up_scale_ptrs.data(),
                    /*gate_bias=*/nullptr, /*up_bias=*/nullptr,
                    gate_out, up_out, total_tokens, K, latent, routed_I, stream);
                kernels::launch_situ_bf16(
                    gate_out, up_out, ws.aligned_act.data(),
                    routes * routed_I, cfg.situ_beta, cfg.situ_linear_beta,
                    stream);
                kernels::launch_bf16_to_fp16(
                    ws.aligned_act.data(), ws.mxfp4_route_act_fp16.data(),
                    static_cast<std::size_t>(routes) * routed_I, stream);
                kernels::launch_mxfp4_moe_down_decode_bf16(
                    ws.mxfp4_route_act_fp16.data(),
                    static_cast<const std::int32_t*>(ws.topk_idx.data()),
                    Lw.expert_down_packed_ptrs.data(),
                    Lw.expert_down_scale_ptrs.data(),
                    /*down_bias=*/nullptr, ws.expert_out.data(),
                    total_tokens, K, latent, routed_I, stream);
                kernels::launch_token_batched_weighted_sum_bf16(
                    ws.latent_out.data(), ws.expert_out.data(),
                    static_cast<const float*>(ws.topk_weights.data()),
                    total_tokens, K, latent, stream);
            } else if (gemv_ok) {
                kernels::launch_moe_gate_up_decode_gemv_bf16(
                    static_cast<const std::int32_t*>(ws.topk_idx.data()),
                    expert_in, Lw.moe_gate_up_bf16->data(),
                    ws.aligned_gate_up.data(),
                    total_tokens, K, latent, routed_I, stream);
                kernels::launch_chunked_situ_bf16(
                    ws.aligned_gate_up.data(), ws.aligned_act.data(), routes,
                    routed_I, cfg.situ_beta, cfg.situ_linear_beta,
                    kimi_k3_moe_gate_up_swapped(), stream);
                kernels::launch_moe_down_decode_gemv_bf16(
                    static_cast<const std::int32_t*>(ws.topk_idx.data()),
                    ws.aligned_act.data(), Lw.moe_down_bf16->data(),
                    ws.expert_out.data(),
                    total_tokens, K, latent, routed_I, stream);
                kernels::launch_token_batched_weighted_sum_bf16(
                    ws.latent_out.data(), ws.expert_out.data(),
                    static_cast<const float*>(ws.topk_weights.data()),
                    total_tokens, K, latent, stream);
            } else {
            if (ws.aligned_block_size <= 0) {
                throw std::runtime_error("kimi_k3: aligned MoE scratch missing");
            }
            // The block is the smaller of what the workspace was sized for and
            // what this token count wants; `max_blocks` is then the worst-case
            // routing skew, every route landing in its own padded block.
            const int block = std::min(ws.aligned_block_size,
                                       kernels::moe_aligned_block(routes, E));
            const int active = std::min(E, routes);
            const int nblocks =
                (routes + active * (block - 1) + block - 1) / block;
            const int aligned_rows = nblocks * block;
            if (nblocks > ws.aligned_max_blocks ||
                aligned_rows >
                    static_cast<int>(ws.aligned_expert_in.shape()[0])) {
                throw std::runtime_error("kimi_k3: aligned MoE scratch too small");
            }
            kernels::launch_moe_align_decode(
                static_cast<const std::int32_t*>(ws.topk_idx.data()),
                static_cast<std::int32_t*>(ws.aligned_route_ids.data()),
                static_cast<std::int32_t*>(ws.aligned_expert_ids.data()),
                /*route_to_aligned_row=*/nullptr, routes, E, block, nblocks,
                stream);
            kernels::launch_gather_moe_aligned_inputs_bf16(
                expert_in,
                static_cast<const std::int32_t*>(ws.aligned_route_ids.data()),
                ws.aligned_expert_in.data(), routes, aligned_rows, K, latent,
                /*shared_row_begin=*/-1, total_tokens, stream);
            kernels::launch_build_moe_ptrs_aligned_bf16(
                static_cast<const std::int32_t*>(ws.aligned_expert_ids.data()),
                Lw.moe_gate_up_bf16->data(), Lw.moe_down_bf16->data(),
                ws.aligned_expert_in.data(), ws.aligned_gate_up.data(),
                ws.aligned_act.data(), ws.aligned_out.data(),
                reinterpret_cast<const void**>(ws.a_gu_ptrs.data()),
                reinterpret_cast<const void**>(ws.b_gu_ptrs.data()),
                reinterpret_cast<void**>(ws.c_gu_ptrs.data()),
                reinterpret_cast<const void**>(ws.a_dn_ptrs.data()),
                reinterpret_cast<const void**>(ws.b_dn_ptrs.data()),
                reinterpret_cast<void**>(ws.c_dn_ptrs.data()),
                nblocks, block, latent, routed_I,
                /*routed_blocks=*/nblocks, nullptr, nullptr, stream);
            ops::gemm_batched_act_x_wt_bf16(
                cublas.handle(),
                reinterpret_cast<const void* const*>(ws.b_gu_ptrs.data()),
                reinterpret_cast<const void* const*>(ws.a_gu_ptrs.data()),
                reinterpret_cast<void* const*>(ws.c_gu_ptrs.data()),
                block, 2 * routed_I, latent, nblocks);
            kernels::launch_chunked_situ_bf16(
                ws.aligned_gate_up.data(), ws.aligned_act.data(), aligned_rows,
                routed_I, cfg.situ_beta, cfg.situ_linear_beta,
                kimi_k3_moe_gate_up_swapped(), stream);
            ops::gemm_batched_act_x_wt_bf16(
                cublas.handle(),
                reinterpret_cast<const void* const*>(ws.b_dn_ptrs.data()),
                reinterpret_cast<const void* const*>(ws.a_dn_ptrs.data()),
                reinterpret_cast<void* const*>(ws.c_dn_ptrs.data()),
                block, latent, routed_I, nblocks);
            kernels::launch_reorder_moe_aligned_output_bf16(
                ws.aligned_out.data(),
                static_cast<const std::int32_t*>(ws.aligned_route_ids.data()),
                ws.expert_out.data(), routes, aligned_rows, latent,
                /*shared_row_begin=*/-1, total_tokens, nullptr, stream);
            kernels::launch_token_batched_weighted_sum_bf16(
                ws.latent_out.data(), ws.expert_out.data(),
                static_cast<const float*>(ws.topk_weights.data()), total_tokens,
                K, latent, stream);
            }  // end aligned batched path

            // Each rank holds a slice of the experts' intermediate, so the
            // latent has to be summed before it leaves the expert bank -- not
            // after `routed_expert_up_proj`, which is replicated.
            if (T > 1 && tp != nullptr) {
                tp->all_reduce_bf16(
                    ws.latent_out.data(),
                    static_cast<std::size_t>(total_tokens) * latent, ncclSum,
                    stream);
            }
            if (Lw.routed_norm != nullptr) {
                kernels::launch_rmsnorm_bf16(
                    ws.latent_out.data(), Lw.routed_norm->data(),
                    ws.latent_out.data(), total_tokens, latent, eps, stream);
            }
            if (Lw.routed_up_proj != nullptr) {
                ops::gemm_act_x_w(cublas.handle(), ws.latent_out.data(),
                                  *Lw.routed_up_proj, ws.mlp_out.data(),
                                  total_tokens, H, latent);
            } else {
                CUDA_CHECK(cudaMemcpyAsync(
                    ws.mlp_out.data(), ws.latent_out.data(),
                    rows_H * sizeof(std::uint16_t), cudaMemcpyDeviceToDevice,
                    stream));
            }

            // The shared expert reads the pre-latent hidden.
            if (shared_I > 0 && Lw.shared_down_proj != nullptr) {
                ops::gemm_act_x_w(cublas.handle(), ws.norm_y.data(),
                                  *Lw.shared_gate_proj, ws.gate.data(),
                                  total_tokens, shared_I, H);
                ops::gemm_act_x_w(cublas.handle(), ws.norm_y.data(),
                                  *Lw.shared_up_proj, ws.up.data(),
                                  total_tokens, shared_I, H);
                kernels::launch_situ_bf16(ws.gate.data(), ws.up.data(),
                                          ws.gate.data(),
                                          total_tokens * shared_I,
                                          cfg.situ_beta, cfg.situ_linear_beta,
                                          stream);
                ops::gemm_act_x_w(cublas.handle(), ws.gate.data(),
                                  *Lw.shared_down_proj, ws.shared_out.data(),
                                  total_tokens, H, shared_I);
                if (T > 1 && tp != nullptr) {
                    tp->all_reduce_bf16(ws.shared_out.data(), rows_H, ncclSum,
                                        stream);
                }
                kernels::launch_residual_add_bf16(ws.mlp_out.data(),
                                                  ws.shared_out.data(), rows_H,
                                                  stream);
            }
        }

        if (!Lw.is_moe && T > 1 && tp != nullptr) {
            tp->all_reduce_bf16(ws.mlp_out.data(), rows_H, ncclSum, stream);
        }
        check_launch("mlp", li);
        kernels::launch_residual_add_bf16(ws.y.data(), ws.mlp_out.data(),
                                          rows_H, stream);
        prof.end(stream);
        act_dump_bf16(act_dump_layer_tag("out", li).c_str(), ws.y.data(),
                      total_tokens, H, stream);
    }

    if (!fwd_cfg.emit_logits) {
        return;
    }

    // ── Tail: one more AttnRes blend, then norm + lm_head ────────────
    prof.begin("tail_lm_head", stream);
    const void* final_in = ws.y.data();
    if (bs > 0 && open_blocks > 0) {
        kernels::launch_attn_res_blend_bf16(
            ws.y.data(), ws.blocks.data(), w.output_res_norm->data(),
            w.output_res_proj->data(), ws.hidden.data(), total_tokens,
            open_blocks, H, block_rows, eps, stream);
        final_in = ws.hidden.data();
    }
    act_dump_bf16("attn_res_out", final_in, total_tokens, H, stream);

    const bool compact_logits = logit_row_indices_d != nullptr &&
                                num_logit_rows > 0 &&
                                num_logit_rows < total_tokens;
    const int rows = compact_logits ? num_logit_rows : total_tokens;
    if (compact_logits) {
        kernels::launch_gather_bf16_rows(
            static_cast<const std::uint16_t*>(final_in), logit_row_indices_d,
            static_cast<std::uint16_t*>(ws.norm_x.data()), num_logit_rows, H,
            stream);
        final_in = ws.norm_x.data();
    }
    kernels::launch_rmsnorm_bf16(final_in, w.final_norm->data(),
                                 ws.norm_y.data(), rows, H, eps, stream);
    act_dump_bf16("final_norm", ws.norm_y.data(), rows, H, stream);
    if (w.lm_head_tp_sharded) {
        throw std::runtime_error(
            "kimi_k3: sharded lm_head is not supported in this forward");
    }
    ops::gemm_act_x_w(cublas.handle(), ws.norm_y.data(), *w.lm_head, logits_out,
                      rows, V, H);
    act_dump_bf16("logits", logits_out, rows, V, stream);
    check_launch("lm_head", -1);
    prof.end(stream);
    prof.report(total_tokens);
}

}  // namespace pie_cuda_driver::model
