#include "model/llama_like.hpp"

#include <cuda_runtime.h>

#include "cuda_check.hpp"
#include "kernels/add_bias.hpp"
#include "kernels/embed.hpp"
#include "kernels/fused_qk_norm_rope.hpp"
#include "kernels/head_dim_pad.hpp"
#include "kernels/kv_paged.hpp"
#include "kernels/residual_add.hpp"
#include "kernels/rmsnorm.hpp"
#include "kernels/rope.hpp"
#include "kernels/swiglu.hpp"
#include "kernels/unpack_qkv.hpp"
#include "ops/attention_flashinfer.hpp"
#include "ops/gemm.hpp"

namespace pie_cuda_driver::model {

namespace {

inline void maybe_add_bias(
    void* out, const DeviceTensor* bias_tensor,
    int N, int dim, cudaStream_t stream)
{
    if (bias_tensor == nullptr) return;
    kernels::launch_add_bias_bf16(out, bias_tensor->data(), N, dim, stream);
}

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

void prepare_llama_like_decode_plan(
    LlamaLikePlanState& state,
    AttentionWorkspace& attn_ws,
    KvCache& cache,
    const HfConfig& cfg,
    const LlamaLikeForwardCfg& fwd_cfg,
    const std::uint32_t* kv_page_indptr_h,
    int num_requests,
    bool is_pure_decode)
{
    // The prepare hook runs OUTSIDE any cuStreamCapture region. It updates
    // pinned/device buffers in `attn_ws` that the captured body reads via
    // cudaMemcpyAsync at replay time, so the same captured graph stays
    // correct across fires with different KV lengths.
    if (!is_pure_decode || fwd_cfg.force_prefill_path) {
        return;
    }
    if (!state.decode_plan) {
        state.decode_plan = ops::make_decode_plan();
    }
    // Skip re-planning when the shape key is unchanged from the last
    // fire. The per-fire KV index updates live in `pi.kv_page_indices`
    // / `pi.kv_last_page_lens` (device pointers + H2D refresh) which
    // the dispatch reads directly — those are NOT in the plan_info
    // struct and don't require re-planning. Verified safe with
    // `enable_cuda_graph=true` (flashinfer's plan_info fields are
    // determined entirely by shape constants in that mode).
    if (state.cached_num_requests == num_requests) {
        return;
    }
    const int T = (fwd_cfg.tp_size > 0) ? fwd_cfg.tp_size : 1;
    const int num_q_heads_local  = cfg.num_attention_heads / T;
    const int num_kv_heads_local = cfg.num_key_value_heads / T;
    ops::plan_attention_flashinfer_decode_bf16(
        *state.decode_plan, kv_page_indptr_h, num_requests,
        num_q_heads_local, num_kv_heads_local, cfg.head_dim_kernel,
        cache.page_size(), attn_ws, /*stream=*/nullptr);
    state.cached_num_requests = num_requests;
}

void llama_like_forward_paged(
    const Qwen3Weights& w,
    const HfConfig& cfg,
    const LlamaLikeForwardCfg& fwd_cfg,
    const LlamaLikePlanState& plan_state,
    Qwen3Workspace& ws,
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
    const std::uint8_t* custom_mask_d,
    const std::int32_t* custom_mask_indptr_d)
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
    const int Hq_kern = num_q_heads_local * dk;
    const int Hk_kern = num_kv_heads_local * dk;
    const bool head_dim_padded = (d != dk);
    const float eps = cfg.rms_norm_eps;
    // Stream comes from the cublas handle. The request handler swaps
    // the cublas stream to the capture stream before
    // `cudaStreamBeginCapture` and restores it after, so when this
    // body runs inside a captured region all its kernel launches land
    // on the captured stream (and thus into the graph). Outside the
    // captured region the stream stays nullptr (legacy default).
    cudaStream_t stream = nullptr;
    cublasGetStream(cublas.handle(), &stream);
    NcclComm* tp = (T > 1) ? fwd_cfg.tp_comm : nullptr;

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

    // Some GQA group sizes (Qwen2 small models — 6, 7) aren't in
    // flashinfer's decode dispatch table; for those we run the prefill
    // kernel even for qo_len==1 batches. The runtime decision lives in
    // a single bool: anything past the plan_attention call uses it.
    const bool use_decode_path = is_pure_decode && !fwd_cfg.force_prefill_path;

    // Decode plan was set up by the prepare hook (runs outside any
    // cudaStream capture region) — the body just reads from
    // `plan_state.decode_plan`. Keeps the body purely device-side so
    // the request handler can capture it into a CUDA graph.
    const ops::DecodePlanCache* decode_plan =
        plan_state.decode_plan ? plan_state.decode_plan.get() : nullptr;

    const bool post_norm = fwd_cfg.norm_placement == NormPlacement::Post;

    // GEMM-M padding: round per-fire row count up to the next power
    // of two (min 8). cublas's heuristic at small / odd M routes to
    // the `gemvx` batched-GEMV path; at the bucketed M values it
    // picks a tensor-core GEMM. Padding only the GEMM calls keeps
    // the change scoped — input rows in [N..pN) are uninitialized
    // junk and the corresponding output rows are never read
    // downstream (per-row independence of matmul guarantees rows
    // [0..N) are unchanged).
    auto next_pow2 = [](int x) {
        if (x <= 8) return 8;
        int r = 8;
        while (r < x) r <<= 1;
        return r;
    };
    const int pN = next_pow2(N);

    for (int L = 0; L < cfg.num_hidden_layers; ++L) {
        const auto& layer = w.layers[L];

        // Pre-norm: norm(y) → norm_x; QKV reads from norm_x.
        // Post-norm (OLMo-3): QKV reads y directly; norm_attn applied
        //                     to attn_out *before* the residual add.
        const void* qkv_in = ws.y.data();
        if (!post_norm) {
            kernels::launch_rmsnorm_bf16(
                ws.y.data(), layer.attn_norm->data(), ws.norm_x.data(),
                N, H, eps, stream);
            qkv_in = ws.norm_x.data();
        }

        // QKV — single fused GEMM + unpack when a packed weight was
        // materialised at bind time (bf16-only path, no biases). Falls
        // back to three separate GEMMs for quantized or biased layers.
        // TP-safe: T=1 here in the bench, and the fused weight only
        // exists when q_proj/k_proj/v_proj are all bf16 (handled at
        // bind time).
        const bool fuse_qkv = (layer.qkv_proj != nullptr) &&
                              !fwd_cfg.use_qkv_bias;
        if (fuse_qkv) {
            ops::gemm_act_x_w(cublas.handle(),
                qkv_in, ops::WeightView(*layer.qkv_proj),
                ws.qkv_packed.data(), pN, Hq + 2 * Hk, H);
            kernels::launch_unpack_qkv_bf16(
                ws.qkv_packed.data(),
                ws.q.data(), ws.k.data(), ws.v.data(),
                pN, Hq, Hk, stream);
        } else {
            ops::gemm_act_x_w(cublas.handle(),
                qkv_in, make_weight_view(layer.q_proj, layer.q_proj_quant),
                ws.q.data(), pN, Hq, H);
            ops::gemm_act_x_w(cublas.handle(),
                qkv_in, make_weight_view(layer.k_proj, layer.k_proj_quant),
                ws.k.data(), pN, Hk, H);
            ops::gemm_act_x_w(cublas.handle(),
                qkv_in, make_weight_view(layer.v_proj, layer.v_proj_quant),
                ws.v.data(), pN, Hk, H);
        }

        if (fwd_cfg.use_qkv_bias) {
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
        // qk-norm + RoPE — fused into one kernel when:
        //   * both norms are per-head (weight shape `[head_dim]`),
        //   * RoPE is the standard (non-YaRN) variant,
        //   * head_dim is supported by the fused kernel (64/128/256).
        // Saves 2 launches per layer (rmsnorm × 2 → just rope) plus
        // the read/write churn on Q/K through gmem.
        const bool both_per_head_norm =
            fwd_cfg.use_qk_norm && layer.q_norm && layer.k_norm &&
            layer.q_norm->shape().size() == 1 &&
            layer.q_norm->shape()[0] == d &&
            layer.k_norm->shape().size() == 1 &&
            layer.k_norm->shape()[0] == d;
        const bool fuse_qknorm_rope =
            both_per_head_norm &&
            fwd_cfg.rope_kind == RopeKind::Standard &&
            (d == 64 || d == 128 || d == 256);
        if (fuse_qknorm_rope) {
            kernels::launch_fused_qk_norm_rope_bf16(
                ws.q.data(), ws.k.data(),
                positions,
                layer.q_norm->data(), layer.k_norm->data(),
                N, num_q_heads_local, num_kv_heads_local, d,
                eps, cfg.rope_theta, stream);
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

        // Pad Q/K/V to `dk` when the model's head_dim isn't a flashinfer
        // dispatch value. The padded buffers are zero on the trailing
        // `dk - d` cols per head; QK·V dot products therefore equal the
        // unpadded ones, and `sm_scale = 1/sqrt(d)` keeps the softmax
        // scaled to the real head dim.
        const void* attn_q   = ws.q.data();
        const void* attn_k   = ws.k.data();
        const void* attn_v   = ws.v.data();
        void* attn_out_buf   = ws.attn_out.data();
        if (head_dim_padded) {
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

        kernels::launch_write_kv_to_pages_bf16(
            cache.k(L), cache.v(L),
            const_cast<void*>(attn_k), const_cast<void*>(attn_v),
            qo_indptr, kv_page_indices, kv_page_indptr, kv_last_page_lens,
            N, R, cache.page_size(), num_kv_heads_local, dk, stream);

        // Per-layer sliding-window dispatch (OLMo-3, Mistral). When
        // `per_layer_window_left` is empty we fall back to the global
        // `sliding_window`; that single value is broadcast to every
        // layer (used by Mistral / Gemma-2 single-mode and Phi-3).
        const int layer_window_left =
            (!fwd_cfg.per_layer_window_left.empty() &&
             L < static_cast<int>(fwd_cfg.per_layer_window_left.size()))
                ? fwd_cfg.per_layer_window_left[L]
                : fwd_cfg.sliding_window;

        if (use_decode_path) {
            ops::dispatch_attention_flashinfer_decode_bf16(
                *decode_plan,
                attn_q, cache.k(L), cache.v(L), attn_out_buf,
                kv_page_indices, kv_page_indptr, kv_last_page_lens,
                attn_ws, stream, layer_window_left,
                /*logits_soft_cap=*/0.f, sm_scale_override);
        } else if (custom_mask_d) {
            ops::launch_attention_flashinfer_prefill_custom_bf16(
                attn_q, cache.k(L), cache.v(L), attn_out_buf,
                qo_indptr, kv_page_indices, kv_page_indptr, kv_last_page_lens,
                custom_mask_d, custom_mask_indptr_d,
                qo_indptr_h, kv_page_indptr_h,
                N, R, num_q_heads_local, num_kv_heads_local, dk,
                cache.page_size(), attn_ws, stream);
        } else {
            ops::launch_attention_flashinfer_prefill_bf16(
                attn_q, cache.k(L), cache.v(L), attn_out_buf,
                qo_indptr, kv_page_indices, kv_page_indptr, kv_last_page_lens,
                qo_indptr_h, kv_page_indptr_h,
                N, R, num_q_heads_local, num_kv_heads_local, dk,
                cache.page_size(), attn_ws, stream, layer_window_left,
                /*logits_soft_cap=*/0.f, sm_scale_override);
        }

        // Strip the trailing pad cols off the attention output before
        // it feeds the o_proj GEMM (which expects `[N, num_q*head_dim]`,
        // not `[N, num_q*head_dim_kernel]`).
        if (head_dim_padded) {
            kernels::launch_strip_head_dim_bf16(
                attn_out_buf, ws.attn_out.data(),
                N, num_q_heads_local, d, dk, stream);
        }

        if (!post_norm) {
            // o_proj is row-parallel: each rank's GEMM produces a partial
            // [N, H] contribution. Single-GPU fuses it into y as a
            // residual-add (beta=1); under TP we go via a scratch
            // (ws.norm_x is free here — it held the QKV input before),
            // all-reduce the partials, then add to the residual.
            if (T == 1) {
                ops::gemm_act_x_w(cublas.handle(),
                    ws.attn_out.data(), make_weight_view(layer.o_proj, layer.o_proj_quant),
                    ws.y.data(), pN, H, Hq, /*beta=*/1.f);
            } else {
                ops::gemm_act_x_w(cublas.handle(),
                    ws.attn_out.data(), make_weight_view(layer.o_proj, layer.o_proj_quant),
                    ws.norm_x.data(), pN, H, Hq, /*beta=*/0.f);
                tp->all_reduce_bf16(ws.norm_x.data(),
                    static_cast<std::size_t>(N) * H, ncclSum, stream);
                kernels::launch_residual_add_bf16(
                    ws.y.data(), ws.norm_x.data(), N * H, stream);
            }
        } else {
            // Post-norm: o_proj writes to norm_x (a scratch we own here),
            // norm_attn(norm_x) → norm_y, then y += norm_y.
            ops::gemm_act_x_w(cublas.handle(),
                ws.attn_out.data(), make_weight_view(layer.o_proj, layer.o_proj_quant),
                ws.norm_x.data(), pN, H, Hq, /*beta=*/0.f);
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
            kernels::launch_rmsnorm_bf16(
                ws.y.data(), layer.mlp_norm->data(), ws.norm_y.data(),
                N, H, eps, stream);
            mlp_in = ws.norm_y.data();
        }

        // Gate / up — fused via chunked-swiglu when a packed weight is
        // available (bf16, no quant). Single GEMM produces `[N, 2*I]`;
        // swiglu reads gate from offset 0, up from offset I.
        if (layer.gate_up_proj) {
            ops::gemm_act_x_w(cublas.handle(),
                mlp_in, ops::WeightView(*layer.gate_up_proj),
                ws.gate_up.data(), pN, 2 * I, H);
            kernels::launch_chunked_swiglu_bf16(
                ws.gate_up.data(), ws.gate.data(), N, I, stream);
        } else {
            ops::gemm_act_x_w(cublas.handle(),
                mlp_in, make_weight_view(layer.gate_proj, layer.gate_proj_quant),
                ws.gate.data(), pN, I, H);
            ops::gemm_act_x_w(cublas.handle(),
                mlp_in, make_weight_view(layer.up_proj, layer.up_proj_quant),
                ws.up.data(),   pN, I, H);
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
                    ws.y.data(), pN, H, I, /*beta=*/1.f);
            } else {
                ops::gemm_act_x_w(cublas.handle(),
                    ws.gate.data(), make_weight_view(layer.down_proj, layer.down_proj_quant),
                    ws.norm_x.data(), pN, H, I, /*beta=*/0.f);
                tp->all_reduce_bf16(ws.norm_x.data(),
                    static_cast<std::size_t>(N) * H, ncclSum, stream);
                kernels::launch_residual_add_bf16(
                    ws.y.data(), ws.norm_x.data(), N * H, stream);
            }
        } else {
            // Post-norm MLP: down_proj → norm_x scratch, norm_mlp, += y.
            ops::gemm_act_x_w(cublas.handle(),
                ws.gate.data(), make_weight_view(layer.down_proj, layer.down_proj_quant),
                ws.norm_x.data(), pN, H, I, /*beta=*/0.f);
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
    }

    kernels::launch_rmsnorm_bf16(
        ws.y.data(), w.final_norm->data(), ws.norm_x.data(),
        N, H, eps, stream);
    ops::gemm_act_x_w(cublas.handle(),
        ws.norm_x.data(), *w.lm_head, ws.logits.data(),
        pN, V, H);
}

}  // namespace pie_cuda_driver::model
