#include "model/qwen3_forward.hpp"

#include <cuda_runtime.h>

#include "cuda_check.hpp"
#include "kernels/embed.hpp"
#include "kernels/kv_paged.hpp"
#include "kernels/rmsnorm.hpp"
#include "kernels/rope.hpp"
#include "kernels/fused_qk_norm_rope.hpp"
#include "kernels/swiglu.hpp"
#include "kernels/unpack_qkv.hpp"
#include "ops/attention_flashinfer.hpp"
#include "ops/attention_naive.hpp"
#include "ops/gemm.hpp"

namespace pie_cuda_driver::model {

Qwen3Workspace Qwen3Workspace::allocate_full(
    const HfConfig& cfg, int max_tokens,
    int max_intermediate, int max_Hq, int max_Hk)
{
    const int H  = cfg.hidden_size;
    const int Hq = max_Hq;
    const int Hk = max_Hk;
    const int I  = max_intermediate;
    const int V  = cfg.vocab_size;
    const int N  = max_tokens;

    Qwen3Workspace ws;
    ws.y         = DeviceTensor::allocate(DType::BF16, {N, H});
    ws.norm_x    = DeviceTensor::allocate(DType::BF16, {N, H});
    ws.q         = DeviceTensor::allocate(DType::BF16, {N, Hq});
    ws.k         = DeviceTensor::allocate(DType::BF16, {N, Hk});
    ws.v         = DeviceTensor::allocate(DType::BF16, {N, Hk});
    ws.qkv_packed = DeviceTensor::allocate(DType::BF16, {N, Hq + 2 * Hk});
    ws.attn_out  = DeviceTensor::allocate(DType::BF16, {N, Hq});
    ws.norm_y    = DeviceTensor::allocate(DType::BF16, {N, H});
    ws.gate      = DeviceTensor::allocate(DType::BF16, {N, I});
    ws.up        = DeviceTensor::allocate(DType::BF16, {N, I});
    ws.gate_up   = DeviceTensor::allocate(DType::BF16, {N, 2 * I});
    ws.logits    = DeviceTensor::allocate(DType::BF16, {N, V});
    ws.probs     = DeviceTensor::allocate(DType::FP32, {N, V});

    // Padded q/k/v/attn_out only when head_dim != head_dim_kernel
    // (currently only Phi-3 at 96 → 128). Empty allocations otherwise
    // — the forward path detects the empty-state and aliases the
    // packed buffers.
    if (cfg.head_dim != cfg.head_dim_kernel) {
        const int Hq_pad = cfg.num_attention_heads * cfg.head_dim_kernel;
        const int Hk_pad = cfg.num_key_value_heads * cfg.head_dim_kernel;
        ws.q_padded        = DeviceTensor::allocate(DType::BF16, {N, Hq_pad});
        ws.k_padded        = DeviceTensor::allocate(DType::BF16, {N, Hk_pad});
        ws.v_padded        = DeviceTensor::allocate(DType::BF16, {N, Hk_pad});
        ws.attn_out_padded = DeviceTensor::allocate(DType::BF16, {N, Hq_pad});
    }
    return ws;
}

Qwen3Workspace Qwen3Workspace::allocate_with_max_intermediate(
    const HfConfig& cfg, int max_tokens, int max_intermediate)
{
    const int Hq = cfg.num_attention_heads * cfg.head_dim;
    const int Hk = cfg.num_key_value_heads * cfg.head_dim;
    return allocate_full(cfg, max_tokens, max_intermediate, Hq, Hk);
}

Qwen3Workspace Qwen3Workspace::allocate(const HfConfig& cfg, int max_tokens) {
    return allocate_with_max_intermediate(cfg, max_tokens, cfg.intermediate_size);
}

void qwen3_forward_prefill(
    const Qwen3Weights& w,
    const HfConfig& cfg,
    Qwen3Workspace& ws,
    ops::CublasHandle& cublas,
    const std::int32_t* token_ids,
    const std::int32_t* positions,
    int N)
{
    const int H  = cfg.hidden_size;
    const int Hq = cfg.num_attention_heads * cfg.head_dim;
    const int Hk = cfg.num_key_value_heads * cfg.head_dim;
    const int I  = cfg.intermediate_size;
    const int V  = cfg.vocab_size;
    const int d  = cfg.head_dim;
    const float eps = cfg.rms_norm_eps;
    cudaStream_t stream = nullptr;  // default stream; cublas already bound here

    // 1. Embed.
    kernels::launch_embed_bf16(
        token_ids, w.embed->data(), ws.y.data(),
        N, H, cfg.vocab_size, stream);

    for (int L = 0; L < cfg.num_hidden_layers; ++L) {
        const auto& layer = w.layers[L];

        // 2. Attention RMSNorm.
        kernels::launch_rmsnorm_bf16(
            ws.y.data(), layer.attn_norm->data(), ws.norm_x.data(),
            N, H, eps, stream);

        // 3. QKV — single fused GEMM + unpack when a packed weight was
        // materialised at bind time (bf16-only path). See qwen3_forward_paged
        // for the rationale.
        if (layer.qkv_proj) {
            ops::gemm_act_x_w(cublas.handle(),
                ws.norm_x.data(), ops::WeightView(*layer.qkv_proj),
                ws.qkv_packed.data(), N, Hq + 2 * Hk, H);
            kernels::launch_unpack_qkv_bf16(
                ws.qkv_packed.data(),
                ws.q.data(), ws.k.data(), ws.v.data(),
                N, Hq, Hk, stream);
        } else {
            ops::gemm_act_x_w(cublas.handle(),
                ws.norm_x.data(), make_weight_view(layer.q_proj, layer.q_proj_quant),
                ws.q.data(), N, Hq, H);
            ops::gemm_act_x_w(cublas.handle(),
                ws.norm_x.data(), make_weight_view(layer.k_proj, layer.k_proj_quant),
                ws.k.data(), N, Hk, H);
            ops::gemm_act_x_w(cublas.handle(),
                ws.norm_x.data(), make_weight_view(layer.v_proj, layer.v_proj_quant),
                ws.v.data(), N, Hk, H);
        }

        // 4-5. q/k norm + RoPE — fused into one launch when both norms
        // are present and head_dim is supported. See qwen3_forward_paged
        // for the rationale.
        const bool fuse_qknorm_rope_prefill =
            layer.q_norm && layer.k_norm &&
            (d == 64 || d == 128 || d == 256);
        if (fuse_qknorm_rope_prefill) {
            kernels::launch_fused_qk_norm_rope_bf16(
                ws.q.data(), ws.k.data(),
                positions,
                layer.q_norm->data(), layer.k_norm->data(),
                N, cfg.num_attention_heads, cfg.num_key_value_heads, d,
                eps, cfg.rope_theta, stream);
        } else {
            if (layer.q_norm) {
                kernels::launch_rmsnorm_bf16(
                    ws.q.data(), layer.q_norm->data(), ws.q.data(),
                    N * cfg.num_attention_heads, d, eps, stream);
            }
            if (layer.k_norm) {
                kernels::launch_rmsnorm_bf16(
                    ws.k.data(), layer.k_norm->data(), ws.k.data(),
                    N * cfg.num_key_value_heads, d, eps, stream);
            }
            kernels::launch_rope_bf16(
                ws.q.data(), ws.k.data(), positions,
                N, cfg.num_attention_heads, cfg.num_key_value_heads, d,
                cfg.rope_theta, stream);
        }

        // 6. Attention (naive O(N^2) over full sequence).
        ops::launch_attention_naive_bf16(
            ws.q.data(), ws.k.data(), ws.v.data(), ws.attn_out.data(),
            N, cfg.num_attention_heads, cfg.num_key_value_heads, d, stream);

        // 7. Output projection + residual: y = y + attn_out @ o_proj^T.
        ops::gemm_act_x_w(cublas.handle(),
            ws.attn_out.data(), make_weight_view(layer.o_proj, layer.o_proj_quant),
            ws.y.data(), N, H, Hq, /*beta=*/1.f);

        // 8. MLP RMSNorm.
        kernels::launch_rmsnorm_bf16(
            ws.y.data(), layer.mlp_norm->data(), ws.norm_y.data(),
            N, H, eps, stream);

        // 9-10. Gate / up + SwiGLU — fused via chunked-swiglu when a
        // packed weight is available.
        if (layer.gate_up_proj) {
            ops::gemm_act_x_w(cublas.handle(),
                ws.norm_y.data(), ops::WeightView(*layer.gate_up_proj),
                ws.gate_up.data(), N, 2 * I, H);
            kernels::launch_chunked_swiglu_bf16(
                ws.gate_up.data(), ws.gate.data(), N, I, stream);
        } else {
            ops::gemm_act_x_w(cublas.handle(),
                ws.norm_y.data(), make_weight_view(layer.gate_proj, layer.gate_proj_quant),
                ws.gate.data(), N, I, H);
            ops::gemm_act_x_w(cublas.handle(),
                ws.norm_y.data(), make_weight_view(layer.up_proj, layer.up_proj_quant),
                ws.up.data(),   N, I, H);
            kernels::launch_swiglu_bf16(
                ws.gate.data(), ws.up.data(), ws.gate.data(),
                N * I, stream);
        }

        // 11. Down projection + residual: y = y + gate @ down_proj^T.
        ops::gemm_act_x_w(cublas.handle(),
            ws.gate.data(), make_weight_view(layer.down_proj, layer.down_proj_quant),
            ws.y.data(), N, H, I, /*beta=*/1.f);
    }

    // 12. Final RMSNorm.
    kernels::launch_rmsnorm_bf16(
        ws.y.data(), w.final_norm->data(), ws.norm_x.data(),
        N, H, eps, stream);

    // 13. lm_head: logits[N, V] = norm_x[N, H] @ lm_head[V, H]^T.
    // (lm_head is currently always bf16; if it ever gets quantized, plumb
    //  a top-level QuantMeta companion on Qwen3Weights for it.)
    ops::gemm_act_x_w(cublas.handle(),
        ws.norm_x.data(), *w.lm_head, ws.logits.data(),
        N, V, H);
}

void qwen3_forward_paged(
    const Qwen3Weights& w,
    const HfConfig& cfg,
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
    const std::uint8_t*  custom_mask_d,
    const std::int32_t*  custom_mask_indptr_d)
{
    const int H  = cfg.hidden_size;
    const int Hq = cfg.num_attention_heads * cfg.head_dim;
    const int Hk = cfg.num_key_value_heads * cfg.head_dim;
    const int I  = cfg.intermediate_size;
    const int V  = cfg.vocab_size;
    const int d  = cfg.head_dim;
    const float eps = cfg.rms_norm_eps;
    cudaStream_t stream = nullptr;

    // GEMM padding: round per-fire row count up to the next power of 2
    // (min bucket = 8). cublas's heuristic at small / odd M routes to
    // the `gemvx` batched-GEMV path; at the bucketed M values it picks
    // a tensor-core GEMM. Padding only the GEMM calls (not rmsnorm,
    // rope, attention, kv-write) keeps the change scoped: the input
    // rows in [N..pN) are uninitialized junk and the corresponding
    // output rows are never read downstream (per-row independence in
    // matmul guarantees rows [0..N) of the output are unchanged).
    auto next_pow2 = [](int x) {
        if (x <= 8) return 8;
        int r = 8;
        while (r < x) r <<= 1;
        return r;
    };
    const int pN = next_pow2(N);

    // 1. Embed.
    kernels::launch_embed_bf16(
        token_ids, w.embed->data(), ws.y.data(),
        N, H, cfg.vocab_size, stream);

    // Hoist the flashinfer decode plan out of the per-layer loop. The
    // plan depends only on (kv_page_indptr_h, num_requests, num_q_heads,
    // num_kv_heads, head_dim, page_size) — all stable across layers.
    // Doing it once here saves 27 redundant DecodePlan calls per fire
    // (each ~25 µs of host work + a small device kernel).
    ops::DecodePlanCachePtr decode_plan;
    if (is_pure_decode) {
        decode_plan = ops::make_decode_plan();
        ops::plan_attention_flashinfer_decode_bf16(
            *decode_plan,
            kv_page_indptr_h,
            R,
            cfg.num_attention_heads,
            cfg.num_key_value_heads,
            d,
            cache.page_size(),
            attn_ws,
            stream);
    }

    for (int L = 0; L < cfg.num_hidden_layers; ++L) {
        const auto& layer = w.layers[L];

        // 2. attn norm
        kernels::launch_rmsnorm_bf16(
            ws.y.data(), layer.attn_norm->data(), ws.norm_x.data(),
            N, H, eps, stream);

        // 3. QKV — single fused GEMM + unpack when a packed weight was
        // materialised at bind time (bf16-only path). Falls back to
        // three separate GEMMs for quantized layers; semantics are
        // identical either way (ws.q/k/v end up filled the same).
        if (layer.qkv_proj) {
            ops::gemm_act_x_w(cublas.handle(),
                ws.norm_x.data(), ops::WeightView(*layer.qkv_proj),
                ws.qkv_packed.data(), pN, Hq + 2 * Hk, H);
            kernels::launch_unpack_qkv_bf16(
                ws.qkv_packed.data(),
                ws.q.data(), ws.k.data(), ws.v.data(),
                pN, Hq, Hk, stream);
        } else {
            ops::gemm_act_x_w(cublas.handle(),
                ws.norm_x.data(), make_weight_view(layer.q_proj, layer.q_proj_quant),
                ws.q.data(), pN, Hq, H);
            ops::gemm_act_x_w(cublas.handle(),
                ws.norm_x.data(), make_weight_view(layer.k_proj, layer.k_proj_quant),
                ws.k.data(), pN, Hk, H);
            ops::gemm_act_x_w(cublas.handle(),
                ws.norm_x.data(), make_weight_view(layer.v_proj, layer.v_proj_quant),
                ws.v.data(), pN, Hk, H);
        }

        // 4-5. q/k norm + RoPE — fused into one launch when both q_norm
        // and k_norm are present (Qwen3 / Gemma-3 / OLMo-3 pattern) and
        // the head_dim is supported (64/128/256). Saves 2 kernel
        // launches per layer over the unfused path.
        const bool fuse_qknorm_rope =
            layer.q_norm && layer.k_norm &&
            (d == 64 || d == 128 || d == 256);
        if (fuse_qknorm_rope) {
            kernels::launch_fused_qk_norm_rope_bf16(
                ws.q.data(), ws.k.data(),
                positions,
                layer.q_norm->data(), layer.k_norm->data(),
                N, cfg.num_attention_heads, cfg.num_key_value_heads, d,
                eps, cfg.rope_theta, stream);
        } else {
            if (layer.q_norm) {
                kernels::launch_rmsnorm_bf16(
                    ws.q.data(), layer.q_norm->data(), ws.q.data(),
                    N * cfg.num_attention_heads, d, eps, stream);
            }
            if (layer.k_norm) {
                kernels::launch_rmsnorm_bf16(
                    ws.k.data(), layer.k_norm->data(), ws.k.data(),
                    N * cfg.num_key_value_heads, d, eps, stream);
            }
            kernels::launch_rope_bf16(
                ws.q.data(), ws.k.data(), positions,
                N, cfg.num_attention_heads, cfg.num_key_value_heads, d,
                cfg.rope_theta, stream);
        }

        // 6. Write current K/V into the page table for this layer.
        kernels::launch_write_kv_to_pages_bf16(
            cache.k(L), cache.v(L),
            ws.k.data(), ws.v.data(),
            qo_indptr, kv_page_indices, kv_page_indptr, kv_last_page_lens,
            N, R, cache.page_size(), cfg.num_key_value_heads, d, stream);

        // 7. Paged attention. flashinfer decode for pure-decode batches,
        // flashinfer prefill (causal) otherwise. The naive paged kernel is
        // retained as a debug fallback but is no longer on the hot path.
        if (is_pure_decode) {
            // flashinfer decode kernel doesn't support custom masks (its
            // single-query-per-request shape can't express arbitrary
            // patterns). Custom-mask decode workloads would need to route
            // through prefill instead — TODO if any inferlet hits that.
            // Plan is hoisted (computed once before the loop); each layer
            // only does the dispatch.
            ops::dispatch_attention_flashinfer_decode_bf16(
                *decode_plan,
                ws.q.data(), cache.k(L), cache.v(L), ws.attn_out.data(),
                kv_page_indices, kv_page_indptr, kv_last_page_lens,
                attn_ws, stream);
        } else if (custom_mask_d) {
            ops::launch_attention_flashinfer_prefill_custom_bf16(
                ws.q.data(), cache.k(L), cache.v(L), ws.attn_out.data(),
                qo_indptr, kv_page_indices, kv_page_indptr, kv_last_page_lens,
                custom_mask_d, custom_mask_indptr_d,
                qo_indptr_h, kv_page_indptr_h,
                N, R, cfg.num_attention_heads, cfg.num_key_value_heads, d,
                cache.page_size(), attn_ws, stream);
        } else {
            ops::launch_attention_flashinfer_prefill_bf16(
                ws.q.data(), cache.k(L), cache.v(L), ws.attn_out.data(),
                qo_indptr, kv_page_indices, kv_page_indptr, kv_last_page_lens,
                qo_indptr_h,
                kv_page_indptr_h,
                N, R, cfg.num_attention_heads, cfg.num_key_value_heads, d,
                cache.page_size(), attn_ws, stream);
        }

        // 8. O proj + residual — padded M; the [N..pN) rows of attn_out
        // are stale junk and the matching rows of y get garbage that's
        // never read (final rmsnorm + lm_head operate on real N).
        ops::gemm_act_x_w(cublas.handle(),
            ws.attn_out.data(), make_weight_view(layer.o_proj, layer.o_proj_quant),
            ws.y.data(), pN, H, Hq, /*beta=*/1.f);

        // 9. mlp norm
        kernels::launch_rmsnorm_bf16(
            ws.y.data(), layer.mlp_norm->data(), ws.norm_y.data(),
            N, H, eps, stream);

        // 10-11. gate / up + SwiGLU — fused via chunked-swiglu when the
        // packed weight was materialised. Reads gate / up halves from
        // a single `[N, 2*I]` GEMM output, writes silu(gate)*up directly
        // to ws.gate. Falls back to two separate GEMMs + classic
        // swiglu when the layer is quantized.
        if (layer.gate_up_proj) {
            ops::gemm_act_x_w(cublas.handle(),
                ws.norm_y.data(), ops::WeightView(*layer.gate_up_proj),
                ws.gate_up.data(), pN, 2 * I, H);
            kernels::launch_chunked_swiglu_bf16(
                ws.gate_up.data(), ws.gate.data(), N, I, stream);
        } else {
            ops::gemm_act_x_w(cublas.handle(),
                ws.norm_y.data(), make_weight_view(layer.gate_proj, layer.gate_proj_quant),
                ws.gate.data(), pN, I, H);
            ops::gemm_act_x_w(cublas.handle(),
                ws.norm_y.data(), make_weight_view(layer.up_proj, layer.up_proj_quant),
                ws.up.data(),   pN, I, H);
            kernels::launch_swiglu_bf16(
                ws.gate.data(), ws.up.data(), ws.gate.data(),
                N * I, stream);
        }

        // 12. down + residual — padded M. ws.gate[N..pN] contains the
        // raw gate*up GEMM output (not SwiGLU'd, but never read for
        // those rows since the residual add only matters at [0..N)).
        ops::gemm_act_x_w(cublas.handle(),
            ws.gate.data(), make_weight_view(layer.down_proj, layer.down_proj_quant),
            ws.y.data(), pN, H, I, /*beta=*/1.f);
    }

    // 13. final norm
    kernels::launch_rmsnorm_bf16(
        ws.y.data(), w.final_norm->data(), ws.norm_x.data(),
        N, H, eps, stream);

    // 14. lm_head — padded M. Final rmsnorm only normalizes [0..N) so
    // norm_x[N..pN] is stale; lm_head writes pN logit rows but only
    // [0..N) are consumed by the sampler.
    ops::gemm_act_x_w(cublas.handle(),
        ws.norm_x.data(), *w.lm_head, ws.logits.data(),
        pN, V, H);
}

}  // namespace pie_cuda_driver::model
