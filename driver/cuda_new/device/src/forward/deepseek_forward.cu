#include "deepseek_forward.cuh"

#include "moe_mlp.cuh"  // moe_mlp_block_bf16

#include "../kernels/argmax.cuh"
#include "../kernels/embed.cuh"
#include "../kernels/residual_add.cuh"
#include "../kernels/rmsnorm.cuh"
#include "../kernels/swiglu.cuh"
#include "../ops/gemm.cuh"

#include <cuda_runtime.h>

namespace pie_cuda_device::forward {

cudaError_t deepseek_forward_bf16(
    cublasHandle_t cublas, cudaStream_t stream,
    const std::int32_t* token_ids, const DeepseekWeights& w,
    const std::int32_t* positions,
    void* ckv_pages, void* kpe_pages,
    const std::uint32_t* qo_indptr, const std::uint32_t* kv_page_indices,
    const std::uint32_t* kv_page_indptr, const std::uint32_t* kv_last_page_lens,
    void* out_logits, std::int32_t* out_token_ids,
    int num_tokens, int num_requests, const DeepseekDims& dims) {
    const int T = num_tokens;
    const int H = dims.hidden;
    const int V = dims.vocab;
    constexpr std::size_t es = sizeof(std::uint16_t);  // bf16

    // Per-layer cache stride (in elements). Same convention as mla_forward:
    // ckv: L * num_pages * page_size * kv_lora_rank;
    // kpe: L * num_pages * page_size * qk_rope_head_dim.
    const std::size_t ckv_layer_stride_elems =
        (std::size_t)dims.num_pages * dims.page_size * dims.kv_lora_rank;
    const std::size_t kpe_layer_stride_elems =
        (std::size_t)dims.num_pages * dims.page_size * dims.qk_rope_head_dim;

    // Residual-stream scratch. The attention sublayer (mla_block_bf16) and the
    // MoE FFN (moe_mlp_block_bf16) allocate their own per-block activation
    // scratch and run on / write to buffers we pass; here we keep the residual
    // stream `hidden`, the final-norm output `normed`, and the FFN sublayer's
    // pre-FFN norm `ffn_normed` + FFN output `ffn_out` (residual-added back).
    void* hidden = nullptr;
    void* normed = nullptr;
    void* ffn_normed = nullptr;
    void* ffn_out = nullptr;
    void* gate = nullptr;
    void* up = nullptr;
    void* mlp = nullptr;
    void* bufs[7] = {};
    const std::size_t bsz[7] = {
        (std::size_t)T * H * es,                  // hidden
        (std::size_t)T * H * es,                  // normed (final)
        (std::size_t)T * H * es,                  // ffn_normed
        (std::size_t)T * H * es,                  // ffn_out
        (std::size_t)T * dims.dense_inter * es,   // gate (dense path)
        (std::size_t)T * dims.dense_inter * es,   // up   (dense path)
        (std::size_t)T * dims.dense_inter * es,   // mlp  (dense path)
    };
    for (int i = 0; i < 7; ++i) {
        if (cudaError_t e = cudaMalloc(&bufs[i], bsz[i]); e != cudaSuccess) {
            for (int j = 0; j < i; ++j) cudaFree(bufs[j]);
            return e;
        }
    }
    hidden = bufs[0];
    normed = bufs[1];
    ffn_normed = bufs[2];
    ffn_out = bufs[3];
    gate = bufs[4];
    up = bufs[5];
    mlp = bufs[6];

    auto cleanup = [&]() {
        for (int i = 0; i < 7; ++i) cudaFree(bufs[i]);
    };

    // embed: token_ids -> residual stream (no scale, as in mla_forward).
    kernels::embed_bf16(token_ids, w.embed, hidden, T, H, V, stream);

    // N DeepSeek decoder layers: MLA attention sublayer (does its own attn_norm
    // + attention + o_proj + residual) then FFN sublayer (ffn_norm + dense
    // SwiGLU MLP or sparse-MoE MLP + residual).
    cudaError_t rc = cudaSuccess;
    for (int L = 0; L < dims.n_layers && rc == cudaSuccess; ++L) {
        const DeepseekLayerWeights& lw = w.layers[L];

        // --- attention sublayer (in place on hidden) ---
        void* ckv_L = static_cast<char*>(ckv_pages) +
                      (std::size_t)L * ckv_layer_stride_elems * es;
        void* kpe_L = static_cast<char*>(kpe_pages) +
                      (std::size_t)L * kpe_layer_stride_elems * es;
        rc = mla_block_bf16(
            cublas, stream, hidden, lw.attn, positions, ckv_L, kpe_L,
            qo_indptr, kv_page_indices, kv_page_indptr, kv_last_page_lens,
            T, num_requests, H, dims.num_heads, dims.q_lora_rank,
            dims.kv_lora_rank, dims.qk_nope_head_dim, dims.qk_rope_head_dim,
            dims.v_head_dim, dims.page_size, dims.rms_eps, dims.sm_scale,
            dims.rope_theta);
        if (rc != cudaSuccess) break;

        // --- FFN sublayer ---
        // ffn_normed = rmsnorm(hidden, ffn_norm)
        kernels::rmsnorm_bf16(hidden, lw.ffn_norm, ffn_normed, T, H, dims.rms_eps,
                              stream);

        if (L < dims.first_k_dense) {
            // Dense SwiGLU MLP -> ffn_out (mirrors llama_layer.cu MLP block).
            const int I = dims.dense_inter;
            ops::gemm_act_x_wt_bf16(cublas, ffn_normed, lw.w_gate, gate, T, I, H, 0.f);
            ops::gemm_act_x_wt_bf16(cublas, ffn_normed, lw.w_up, up, T, I, H, 0.f);
            kernels::swiglu_bf16(gate, up, mlp, T * I, stream);
            ops::gemm_act_x_wt_bf16(cublas, mlp, lw.w_down, ffn_out, T, H, I, 0.f);
            // residual add the dense FFN output back into hidden.
            kernels::residual_add_bf16(hidden, ffn_out, (std::size_t)T * H, stream);
        } else {
            // Sparse top-K MoE MLP -> ffn_out (validated dense reference path).
            // moe_mlp_block_bf16 synchronizes internally and frees its scratch.
            rc = moe_mlp_block_bf16(cublas, stream, ffn_normed, lw.router_w,
                                    lw.wgu, lw.wdown, ffn_out, T, H,
                                    dims.moe_inter, dims.num_experts,
                                    dims.top_k);
            if (rc != cudaSuccess) break;
            kernels::residual_add_bf16(hidden, ffn_out, (std::size_t)T * H, stream);
        }
    }
    if (rc != cudaSuccess) {
        cleanup();
        return rc;
    }

    // final norm -> lm_head -> greedy argmax.
    kernels::rmsnorm_bf16(hidden, w.final_norm, normed, T, H, dims.rms_eps, stream);
    ops::gemm_act_x_wt_bf16(cublas, normed, w.lm_head, out_logits, T, V, H, 0.f);
    kernels::argmax_bf16(out_logits, out_token_ids, T, V, stream);

    cudaError_t e = cudaStreamSynchronize(stream);
    cleanup();
    return e;
}

}  // namespace pie_cuda_device::forward
