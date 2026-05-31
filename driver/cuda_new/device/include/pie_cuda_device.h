// pie_cuda_device — flat C ABI for the thin CUDA device library.
//
// This header is THE seam of the driver/cuda_new rewrite. The Rust
// control plane (`control/`) owns every decision; this library owns
// every kernel sequence. The boundary is deliberately coarse: one call
// per forward body / prepare / sample / alloc, so FFI overhead is
// irrelevant (~5 calls per token-step, each 100s of µs–ms).
//
// Design rules:
//   * Handles are opaque. Lifetime is explicit (create / destroy).
//   * Argument structs are POD (pointers + scalars), C-layout, stable.
//     `PieForwardInputs` mirrors driver/cuda's existing `ForwardInputs`
//     (executor.hpp:81) almost verbatim — that struct was already the
//     "all metadata for one body call" bundle.
//   * No C++ types cross the boundary; no exceptions cross the boundary
//     (entry points are noexcept and return PieStatus).
//   * The host-side `prepare` / device-side `body` two-phase split
//     (graph-safety, forward_graph.hpp:14-31) is preserved as two calls.
#ifndef PIE_CUDA_DEVICE_H
#define PIE_CUDA_DEVICE_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// ---------------------------------------------------------------------------
// Status / versioning
// ---------------------------------------------------------------------------

typedef enum PieStatus {
    PIE_OK = 0,
    PIE_ERR_INVALID_ARG = 1,
    PIE_ERR_CUDA = 2,
    PIE_ERR_OOM = 3,
    PIE_ERR_UNSUPPORTED_ARCH = 4,
    PIE_ERR_INTERNAL = 5,
} PieStatus;

// ABI version. Bumped on any breaking change to a struct layout or
// signature below. The Rust side asserts a match at load.
//   v1: lifecycle + construction/hot-path/graph stubs.
//   v2: raw device memory + first lifted kernel (rmsnorm).
//   v3: + residual_add, swiglu block primitives.
//   v4: + rope (kernel), gemm (bf16 cuBLAS op).
//   v5: + embed, argmax, naive paged attention (single-layer forward set).
//   v6: + composed llama_like decoder layer (first pie_body fragment).
//   v7: + real pie_ws_alloc + full llama_like forward (embed→layers→lm_head).
//   v8: + pie_cuda_device_props (SM count / compute capability for planning).
//   v9: + kv-append scatter, temp sampling, dtype casts, gather_rows.
//   v10: forward uses the KV scatter + takes num_kv_pages (general paged KV).
//   v11: + Gemma kernels (rmsnorm 1+w, geglu_tanh, logit_softcap), FP8 dequant.
//   v12: + YaRN RoPE (Llama-3), MoE router (topk_softmax) + chunked_swiglu.
//   v13: + rope_partial (Gemma-4), causal_conv1d (Mamba), int4 WNA16 dequant.
//   v14: + MoE MLP block (dense reference form).
//   v15: + MLA attention block (DeepSeek-style, absorbed, latent paged attn).
//   v16: + AltUp predict/correct (Gemma-3n/4 alternating residual streams).
//   v17: + grouped per-expert GEMM (sparse-MoE GEMM after dispatch scatter).
//   v18: + full MLA forward (embed → N×mla_block → final norm → lm_head → argmax).
//   v19: + full dense-MoE forward (llama attention + top-K MoE FFN per layer).
//   v20: + Mamba-2/SSD selective scan (Nemotron-H recurrence after causal conv).
//   v21: + int4 (u4b8) fused quant GEMM (internalized de-branded Marlin) + repack.
//   v22: + sparse token-dispatched MoE block (dispatch → grouped GEMM → combine).
//   v23: + Nemotron-H Mamba-2 mixer block (in_proj → conv → scan → gated norm).
//   v24: + full DeepSeek-V3/V4 forward (MLA attention + per-layer dense/MoE FFN).
//   v25: + Gemma-3/4 forward (sandwich norms, softcaps, sliding/full, embed scale).
//   v26: + Nemotron-H whole-model forward (hybrid Mamba/attention/FFN schedule).
//   v27: + fp8 (fe4m3fn) fused quant GEMM (qgemm fan-out over the int4 template).
//   v28: + per-head q/k-norm in the llama layer (Qwen3); q_norm/k_norm weights.
//   v29: + additive q/k/v projection biases in the llama layer (Qwen2).
#define PIE_CUDA_DEVICE_ABI_VERSION 31u
uint32_t pie_cuda_abi_version(void);

// Last error string for the calling thread (valid until the next ABI
// call on this thread). NULL when the last call returned PIE_OK.
const char* pie_cuda_last_error(void);

// ---------------------------------------------------------------------------
// Opaque handles — owned by the device lib, lifetime driven by Rust.
// ---------------------------------------------------------------------------

typedef struct PieDevCtx    PieDevCtx;     // device, stream, cuBLAS handle
typedef struct PieWeights   PieWeights;    // resident model weights
typedef struct PieKvCache   PieKvCache;    // paged KV / MLA / recurrent storage
typedef struct PieWorkspace PieWorkspace;  // per-step activation buffers + pinned inputs
typedef struct PieGraphExec PieGraphExec;  // a captured cudaGraphExec_t

// Selects which per-arch body/prepare to dispatch. Kept in lockstep
// with control/src/arch (ArchId).
typedef enum PieArchId {
    PIE_ARCH_LLAMA_LIKE = 0,
    PIE_ARCH_QWEN3 = 1,
    PIE_ARCH_QWEN3_5 = 2,
    PIE_ARCH_QWEN3_5_MOE = 3,
    PIE_ARCH_MIXTRAL = 4,
    PIE_ARCH_GEMMA2 = 5,
    PIE_ARCH_GEMMA3N = 6,
    PIE_ARCH_GEMMA4 = 7,
    PIE_ARCH_NEMOTRON_H = 8,
    PIE_ARCH_DEEPSEEK_V4 = 9,
    PIE_ARCH_KIMI = 10,
    PIE_ARCH_GLM5 = 11,
    PIE_ARCH_GPT_OSS = 12,
} PieArchId;

// ---------------------------------------------------------------------------
// POD argument structs (C layout, stable).
// ---------------------------------------------------------------------------

// All metadata to execute one forward body call. Mirrors driver/cuda's
// ForwardFn::ForwardInputs (executor.hpp:81). Pointers are device
// pointers unless suffixed _h (host/pinned).
typedef struct PieForwardInputs {
    const int32_t*  token_ids;
    const int32_t*  positions;

    const uint32_t* qo_indptr_d;
    const uint32_t* kv_page_indices_d;
    const uint32_t* kv_page_indptr_d;
    const uint32_t* kv_last_page_lens_d;
    const uint32_t* qo_indptr_h;
    const uint32_t* kv_page_indices_h;
    const uint32_t* kv_page_indptr_h;
    const uint32_t* kv_last_page_lens_h;

    int32_t total_tokens;     // N
    int32_t num_requests;     // R
    int32_t is_pure_decode;   // bool

    const uint8_t* custom_mask_d;          // optional BRLE-packed mask
    const int32_t* custom_mask_indptr_d;

    const int32_t* slot_ids_h;             // optional rs-cache slots
    const uint8_t* is_fresh_h;
    const int32_t* slot_ids_d;

    const int32_t* logit_row_indices_d;    // optional compact-logit gather
    int32_t        num_logit_rows;

    int32_t tp_greedy_argmax;              // bool

    const int32_t* commit_advance_gather_d;  // optional recurrent commit-advance
} PieForwardInputs;

// Host-side planning inputs (the prepare half). Mirrors
// ForwardFn::PrepareInputs (executor.hpp:128).
typedef struct PiePrepareInputs {
    const uint32_t* qo_indptr_h;
    const uint32_t* kv_page_indices_h;
    const uint32_t* kv_page_indices_d;
    const uint32_t* kv_page_indptr_h;
    const uint32_t* kv_page_indptr_d;
    const uint32_t* kv_last_page_lens_h;
    const uint32_t* kv_last_page_lens_d;
    int32_t total_tokens;
    int32_t num_requests;
    int32_t is_pure_decode;  // bool
} PiePrepareInputs;

// Sampling policy computed by Rust (control/src/sampler.rs). The argmax
// / categorical kernels stay C++; the policy + seeding is Rust.
typedef struct PieSampleParams {
    const float*   temperature;   // [num_rows] device
    const float*   top_p;         // [num_rows] device, NaN/1.0 = disabled
    const int32_t* top_k;         // [num_rows] device, <=0 = disabled
    const uint64_t* seed;         // [num_rows] device per-row PRNG seed
    int32_t num_rows;
    int32_t greedy;               // bool: argmax fast path, ignore temp/top_*
} PieSampleParams;

// KV cache geometry. Per-layer vectors are optional (NULL => homogeneous,
// use the scalar). Covers the cases driver/cuda's KvCache::allocate_per_layer
// handles (Gemma-4 dual head_dim + KV sharing).
typedef struct PieKvLayout {
    int32_t num_layers;
    int32_t num_pages;
    int32_t page_size;
    int32_t num_kv_heads;            // scalar fallback
    int32_t head_dim;                // scalar fallback
    uint32_t format;                 // PieKvFormat
    int32_t hnd_layout;              // bool (flashinfer HND)
    const int32_t* per_layer_head_dim;     // optional [num_layers]
    const int32_t* per_layer_num_kv_heads; // optional [num_layers]
    const int32_t* kv_source_layer;        // optional [num_layers] (KV sharing)
} PieKvLayout;

typedef enum PieKvFormat {
    PIE_KV_BF16 = 0,
    PIE_KV_FP8 = 1,
    PIE_KV_INT8 = 2,
    PIE_KV_FP4 = 3,
} PieKvFormat;

// Activation workspace sizing. Derived by the Rust arch + mem planner.
typedef struct PieWorkspaceDims {
    int32_t max_tokens;        // forward capacity (N)
    int32_t max_requests;      // R
    int32_t hidden_size;
    int32_t intermediate_size;
    int32_t num_heads;
    int32_t num_kv_heads;
    int32_t head_dim;
    int32_t vocab_size;
    int32_t num_layers;
    int32_t recurrent_state_slots;  // 0 if no Mamba/linear-attn state
    int32_t moe_experts;            // 0 if dense
} PieWorkspaceDims;

// ---------------------------------------------------------------------------
// Lifecycle / context
// ---------------------------------------------------------------------------

// Initialize the device library (one-time, process-global). `device_ordinal`
// is the CUDA device. Out param receives the context handle.
PieStatus pie_cuda_ctx_create(int32_t device_ordinal, PieDevCtx** out_ctx);
PieStatus pie_cuda_ctx_destroy(PieDevCtx* ctx);

// Total / free device memory in bytes (for the memory planner probe).
PieStatus pie_cuda_mem_info(PieDevCtx* ctx, size_t* out_free, size_t* out_total);

// SM count + compute capability (major.minor) — inputs to mem::plan's
// per-profile sizing.
PieStatus pie_cuda_device_props(PieDevCtx* ctx, int32_t* out_sm_count,
                                int32_t* out_major, int32_t* out_minor);

// ---------------------------------------------------------------------------
// Raw device memory — generic plumbing used by ops, the input staging
// buffers, and tests. Buffers are raw device pointers; lifetime is
// explicit. Copies run on the context stream; call pie_cuda_stream_sync
// before reading a D2H result on the host.
// ---------------------------------------------------------------------------

PieStatus pie_cuda_malloc(PieDevCtx* ctx, size_t nbytes, void** out_ptr);
PieStatus pie_cuda_free(PieDevCtx* ctx, void* ptr);
PieStatus pie_cuda_memcpy_h2d(PieDevCtx* ctx, void* dst, const void* src, size_t nbytes);
PieStatus pie_cuda_memcpy_d2h(PieDevCtx* ctx, void* dst, const void* src, size_t nbytes);
PieStatus pie_cuda_memcpy_d2d(PieDevCtx* ctx, void* dst, const void* src, size_t nbytes);
PieStatus pie_cuda_stream_sync(PieDevCtx* ctx);

// ---------------------------------------------------------------------------
// Lifted kernels — phase 1 carves these behind the ABI; bodies are lifted
// verbatim from driver/cuda/src/kernels. The first one proves the seam
// end-to-end (Rust → FFI → CUDA kernel → result).
// ---------------------------------------------------------------------------

// Row-wise RMSNorm over bf16 (no weight+1 convention):
//   y[r,:] = x[r,:] * rsqrt(mean(x[r,:]^2) + eps) * weight
// x / y: [num_rows, hidden] bf16 row-major; weight: [hidden] bf16.
PieStatus pie_cuda_rmsnorm_bf16(PieDevCtx* ctx, const void* x, const void* weight,
                                void* y, int32_t num_rows, int32_t hidden, float eps);

// In-place elementwise bf16 add: y[i] = round_bf16(y[i] + x[i]) over n elems.
PieStatus pie_cuda_residual_add_bf16(PieDevCtx* ctx, void* y, const void* x, size_t n);

// SwiGLU MLP activation: y = silu(gate) * up (elementwise, bf16, n elems).
PieStatus pie_cuda_swiglu_bf16(PieDevCtx* ctx, const void* gate, const void* up,
                               void* y, int32_t num_elements);

// In-place RoPE on Q [num_tokens, num_q_heads, head_dim] and K
// [num_tokens, num_kv_heads, head_dim] (bf16). positions: [num_tokens] int32
// device. interleaved=0 → NeoX (Llama/Qwen); !=0 → GPT-J (GLM).
PieStatus pie_cuda_rope_bf16(PieDevCtx* ctx, void* q, void* k, const int32_t* positions,
                             int32_t num_tokens, int32_t num_q_heads, int32_t num_kv_heads,
                             int32_t head_dim, float theta, int32_t interleaved);

// bf16 GEMM: y = act @ W^T (+ beta*y). act [M,K], W [N,K] row-major; y [M,N].
PieStatus pie_cuda_gemm_bf16(PieDevCtx* ctx, const void* act, const void* w, void* y,
                             int32_t M, int32_t N, int32_t K, float beta);

// Token-id → hidden embedding lookup: y[n,:] = weight[token_ids[n], :].
// token_ids [num_tokens] int32; weight [vocab, hidden] bf16; y bf16.
PieStatus pie_cuda_embed_bf16(PieDevCtx* ctx, const int32_t* token_ids, const void* weight,
                              void* y, int32_t num_tokens, int32_t hidden, int32_t vocab);

// Per-row greedy argmax over [num_rows, vocab] bf16 logits → [num_rows] int32
// token ids (lowest-index tie-break). NOTE: pass an even `vocab` — the kernel's
// default vectorized path reads bf16 pairs.
PieStatus pie_cuda_argmax_bf16(PieDevCtx* ctx, const void* logits, int32_t* token_ids,
                               int32_t num_rows, int32_t vocab);

// Append freshly-computed K/V into the paged cache (NHD when hnd_layout=0).
// k_curr/v_curr [total_tokens, num_kv_heads, head_dim] bf16; pages
// [num_pages, page_size, num_kv_heads, head_dim] bf16. CSR page lists on dev.
PieStatus pie_cuda_write_kv_to_pages_bf16(
    PieDevCtx* ctx, void* k_pages, void* v_pages, const void* k_curr, const void* v_curr,
    const uint32_t* qo_indptr_d, const uint32_t* kv_page_indices_d,
    const uint32_t* kv_page_indptr_d, const uint32_t* kv_last_page_lens_d,
    int32_t total_tokens, int32_t num_requests, int32_t page_size, int32_t num_kv_heads,
    int32_t head_dim, int32_t hnd_layout);

// Per-row temperature-scaled multinomial sampling (Gumbel-max). logits
// [num_rows, vocab] bf16; temperatures [num_rows] f32 (temp<=0 → argmax,
// truncation ignored); seeds [num_rows] u32; out [num_rows] i32. The
// truncation arrays top_ps/top_ks/min_ps are each [num_rows] or NULL
// (NULL = that filter off for all rows): top_p>=1/<=0 off, top_k<=0 off,
// min_p=0 off. The kept set is the intersection of the enabled filters.
PieStatus pie_cuda_sample_temp_bf16(
    PieDevCtx* ctx, const void* logits, const float* temperatures, const float* top_ps,
    const int32_t* top_ks, const float* min_ps, const uint32_t* seeds, int32_t* out,
    int32_t num_rows, int32_t vocab);

// Element-wise dtype casts (n elements). Used by the loader for non-bf16
// checkpoints. fp16/fp32 → bf16 round-nearest-even; bf16 → fp32 exact.
PieStatus pie_cuda_cast_fp16_to_bf16(PieDevCtx* ctx, const void* src, void* dst, size_t n);
PieStatus pie_cuda_cast_fp32_to_bf16(PieDevCtx* ctx, const void* src, void* dst, size_t n);
PieStatus pie_cuda_cast_bf16_to_fp32(PieDevCtx* ctx, const void* src, void* dst, size_t n);

// Gather rows of a [num_src_rows, vocab] bf16 buffer into [num_dst_rows,
// vocab] by `row_indices` (compact logits — lm_head only sampled rows).
PieStatus pie_cuda_gather_bf16_rows(
    PieDevCtx* ctx, const uint16_t* src, const int32_t* row_indices, uint16_t* dst,
    int32_t num_dst_rows, int32_t vocab);

// Gemma family. rmsnorm with the `(1 + w)` convention; GeGLU-tanh MLP
// activation; in-place logit softcap `x = cap * tanh(x / cap)`.
PieStatus pie_cuda_rmsnorm_gemma_bf16(PieDevCtx* ctx, const void* x, const void* weight,
                                      void* y, int32_t num_rows, int32_t hidden, float eps);
PieStatus pie_cuda_geglu_tanh_bf16(PieDevCtx* ctx, const void* gate, const void* up,
                                   void* y, int32_t num_elements);
PieStatus pie_cuda_logit_softcap_bf16(PieDevCtx* ctx, void* x, float cap, size_t n);

// FP8 (E4M3) → bf16 dequant with a scalar scale (for FP8 checkpoints).
// fp8_in: [n] raw e4m3 bytes; bf16_out: [n] bf16; out = e4m3(byte) * scale.
PieStatus pie_cuda_dequant_fp8_e4m3_to_bf16(PieDevCtx* ctx, const uint8_t* fp8_in,
                                            void* bf16_out, float scale, size_t n);

// YaRN RoPE (Llama-3 / Mistral long-context frequency scaling), in place on
// q/k. `factor=1` ⇒ un-scaled base RoPE. See pie_cuda_rope_bf16 for layout.
PieStatus pie_cuda_rope_yarn_bf16(
    PieDevCtx* ctx, void* q, void* k, const int32_t* positions, int32_t num_tokens,
    int32_t num_q_heads, int32_t num_kv_heads, int32_t head_dim, float theta, float factor,
    float low_freq_factor, float high_freq_factor, int32_t original_max_position);

// MoE router: top-K experts + softmax (over all experts) renormalized to the
// picked K. logits [N, num_experts] bf16 → topk_idx [N, K] i32, topk_w [N, K] f32.
PieStatus pie_cuda_topk_softmax_bf16(
    PieDevCtx* ctx, const void* logits, int32_t* topk_idx, float* topk_w, int32_t N,
    int32_t num_experts, int32_t K);

// Fused MoE expert activation: y[n,i] = silu(packed[n,i]) * packed[n, I+i].
// packed [N, 2*I] bf16 (gate then up); y [N, I] bf16.
PieStatus pie_cuda_chunked_swiglu_bf16(PieDevCtx* ctx, const void* packed, void* y,
                                       int32_t N, int32_t I);

// Partial RoPE (Gemma-4 full-attention layers): rotates only the first
// rotary_dim of each head (NeoX pairing); trailing dims pass through. q/k
// in place [num_tokens, n_*_heads, head_dim] bf16.
PieStatus pie_cuda_rope_partial_bf16(PieDevCtx* ctx, void* q, void* k, const int32_t* positions,
    int32_t num_tokens, int32_t num_q_heads, int32_t num_kv_heads, int32_t head_dim,
    int32_t rotary_dim, float theta);

// Causal depthwise conv1d (Mamba/SSM conv step), prefill, SiLU-fused.
// x/y [N, C] bf16; weight [C, K] bf16; bias [C] bf16 (NULL = none).
PieStatus pie_cuda_causal_conv1d_prefill_bf16(PieDevCtx* ctx, const void* x, const void* weight,
    const void* bias, void* y, int32_t N, int32_t C, int32_t K);

// WNA16 int4 (uint4b8) → bf16 group-wise dequant (GPTQ/AWQ/compressed-tensors).
// packed [out_dim, in_dim/8] int32; scale [out_dim, in_dim/group_size] bf16.
PieStatus pie_cuda_dequant_wna16_int4b8_to_bf16(PieDevCtx* ctx, const int32_t* packed,
    const void* scale_bf16, void* out_bf16, int32_t out_dim, int32_t in_dim, int32_t group_size);

// Naive paged-KV attention (causal; no flashinfer). q/o [total_tokens,
// num_q_heads, head_dim] bf16; k_pages/v_pages [num_pages, page_size,
// num_kv_heads, head_dim] bf16. CSR page lists on device. sm_scale < 0 →
// 1/sqrt(head_dim); window_left < 0 → full causal.
PieStatus pie_cuda_attention_naive_paged_bf16(
    PieDevCtx* ctx, const void* q, const void* k_pages, const void* v_pages, void* o,
    const uint32_t* qo_indptr_d, const uint32_t* kv_page_indices_d,
    const uint32_t* kv_page_indptr_d, const uint32_t* kv_last_page_lens_d,
    int32_t total_tokens, int32_t num_requests, int32_t num_q_heads, int32_t num_kv_heads,
    int32_t head_dim, int32_t page_size, int32_t window_left, float sm_scale);

// ---------------------------------------------------------------------------
// Composed forward (phase-1 vertical slice). One llama-like decoder layer,
// chaining the lifted primitives in C++. The first `pie_body` fragment;
// generalized into the full IModel-style body path later. See
// forward/llama_layer.cuh for the slice simplifications (per-call scratch,
// contiguous KV append).
// ---------------------------------------------------------------------------

// Per-layer weights (device bf16). HF row-major: wq [n_q_heads*head_dim,
// hidden], wk/wv [n_kv_heads*head_dim, hidden], wo [hidden, n_q_heads*
// head_dim], w_gate/w_up [intermediate, hidden], w_down [hidden,
// intermediate], attn_norm/ffn_norm [hidden].
typedef struct PieLlamaLayerWeights {
    const void* attn_norm;
    const void* wq;
    const void* wk;
    const void* wv;
    const void* wo;
    const void* ffn_norm;
    const void* w_gate;
    const void* w_up;
    const void* w_down;
    // Per-head q/k RMSNorm gains [head_dim] (Qwen3); both null = no qk-norm.
    const void* q_norm;
    const void* k_norm;
    // Additive q/k/v projection biases (Qwen2); null = no bias.
    const void* q_bias;
    const void* k_bias;
    const void* v_bias;
} PieLlamaLayerWeights;

// Runs one llama-like decoder layer in place on `hidden` [num_tokens,
// hidden_size] bf16. positions [num_tokens] int32 (device). k_pages/v_pages
// [num_pages, page_size, n_kv_heads, head_dim] bf16. CSR page lists on device.
PieStatus pie_cuda_llama_layer_bf16(
    PieDevCtx* ctx, void* hidden, const PieLlamaLayerWeights* w, const int32_t* positions,
    void* k_pages, void* v_pages, const uint32_t* qo_indptr_d,
    const uint32_t* kv_page_indices_d, const uint32_t* kv_page_indptr_d,
    const uint32_t* kv_last_page_lens_d, int32_t num_tokens, int32_t num_requests,
    int32_t hidden_size, int32_t n_q_heads, int32_t n_kv_heads, int32_t head_dim,
    int32_t intermediate, int32_t page_size, float rms_eps, float rope_theta);

// Whole-model weights. `layers` is a host array of `n_layers` per-layer
// weight structs (each field a device pointer). embed/lm_head [vocab,
// hidden], final_norm [hidden] — device bf16.
typedef struct PieLlamaWeights {
    const void* embed;
    const PieLlamaLayerWeights* layers;
    int32_t n_layers;
    const void* final_norm;
    const void* lm_head;
} PieLlamaWeights;

// Full llama-like forward (prefill): embed → N layers → final norm →
// lm_head → argmax. token_ids/out_token_ids [num_tokens] int32 device;
// out_logits [num_tokens, vocab] bf16 device. KV for layer L is at
// kv_k/kv_v + L*(page_size*n_kv_heads*head_dim) bf16 elements. Uses `ws`
// scratch (allocated via pie_ws_alloc).
PieStatus pie_cuda_llama_forward_bf16(
    PieDevCtx* ctx, PieWorkspace* ws, const int32_t* token_ids, const PieLlamaWeights* w,
    const int32_t* positions, void* kv_k, void* kv_v, const uint32_t* qo_indptr_d,
    const uint32_t* kv_page_indices_d, const uint32_t* kv_page_indptr_d,
    const uint32_t* kv_last_page_lens_d, void* out_logits, int32_t* out_token_ids,
    int32_t num_tokens, int32_t num_requests, int32_t hidden_size, int32_t n_q_heads,
    int32_t n_kv_heads, int32_t head_dim, int32_t intermediate, int32_t page_size,
    int32_t num_kv_pages, int32_t vocab, float rms_eps, float rope_theta);

// MoE MLP sublayer (dense reference form): router → top-K softmax → per-expert
// (gate_up → swiglu → down) → weighted top-K combine. All bf16. hidden/out
// [T, hidden]; router_w [E, hidden]; wgu [E, 2*intermediate, hidden] (gate||up);
// wdown [E, hidden, intermediate]. (Sparse token-dispatch is the perf follow-up.)
PieStatus pie_cuda_moe_mlp_block_bf16(
    PieDevCtx* ctx, const void* hidden, const void* router_w, const void* wgu,
    const void* wdown, void* out, int32_t num_tokens, int32_t hidden_size,
    int32_t intermediate, int32_t num_experts, int32_t top_k);

// Sparse (token-dispatched) MoE block — perf-real alternative to
// pie_cuda_moe_mlp_block_bf16 with IDENTICAL semantics + weight layouts (router →
// top-K softmax → dispatch-scatter → grouped GEMM gate||up → swiglu → grouped
// GEMM down → weighted top-K combine). Same args as the dense block.
PieStatus pie_cuda_moe_sparse_block_bf16(
    PieDevCtx* ctx, const void* hidden, const void* router_w, const void* wgu,
    const void* wdown, void* out, int32_t num_tokens, int32_t hidden_size,
    int32_t intermediate, int32_t num_experts, int32_t top_k);

// Nemotron-H Mamba-2 mixer per-layer weights (device bf16, row-major). H=hidden,
// intermediate = num_heads*head_dim, conv_dim = intermediate + 2*n_groups*
// state_size, d_in_proj = intermediate + conv_dim + num_heads. in_proj_w
// [d_in_proj,H], conv_w [conv_dim,conv_kernel], conv_bias [conv_dim] (nullable),
// a_log/d/dt_bias [num_heads], norm_weight [intermediate], out_proj_w [H,intermediate].
typedef struct PieNemotronMambaWeights {
    const void* in_proj_w;
    const void* conv_w;
    const void* conv_bias;
    const void* a_log;
    const void* d;
    const void* dt_bias;
    const void* norm_weight;
    const void* out_proj_w;
} PieNemotronMambaWeights;

// One Nemotron-H Mamba-2 mixer block (prefill, single request, fresh state):
// in_proj → split(z|x|B|C|dt) → causal conv → SSD selective scan → gated RMSNorm
// → out_proj, residual-added in place on `hidden` [num_tokens, hidden_size] bf16.
PieStatus pie_cuda_nemotron_mamba_block_bf16(
    PieDevCtx* ctx, void* hidden, const PieNemotronMambaWeights* w, int32_t num_tokens,
    int32_t hidden_size, int32_t num_heads, int32_t head_dim, int32_t state_size,
    int32_t n_groups, int32_t conv_kernel, float rms_eps, float time_step_min);

// Nemotron-H whole-model forward (hybrid Mamba/attention/FFN). Per-layer weights
// carry a `kind` ('M' Mamba | 'A' attention | 'F' FFN); only the matching union
// member is read. Attention = standard GQA; FFN = dense SwiGLU (stand-in for the
// upstream relu² MoE). The forward manages its own Mamba state + per-attn-layer
// KV pool internally (single fresh prefill). All weights device bf16, row-major.
typedef struct PieNemotronAttnWeights {
    const void* attn_norm;
    const void* wq;
    const void* wk;
    const void* wv;
    const void* wo;
} PieNemotronAttnWeights;

typedef struct PieNemotronFfnWeights {
    const void* ffn_norm;
    const void* w_gate;
    const void* w_up;
    const void* w_down;
} PieNemotronFfnWeights;

typedef struct PieNemotronLayerWeights {
    char kind;                          // 'M' | 'A' | 'F'
    const void* mamba_pre_norm;         // [H], used iff kind=='M'
    PieNemotronMambaWeights mamba;      // used iff kind=='M'
    PieNemotronAttnWeights attn;        // used iff kind=='A'
    PieNemotronFfnWeights ffn;          // used iff kind=='F'
} PieNemotronLayerWeights;

typedef struct PieNemotronWeights {
    const void* embed;
    const PieNemotronLayerWeights* layers;   // host array, length n_layers
    int32_t n_layers;
    const void* final_norm;
    const void* lm_head;
} PieNemotronWeights;

// `kinds_host` is a host [n_layers] char array (the 'M'/'A'/'F' schedule).
PieStatus pie_cuda_nemotron_forward_bf16(
    PieDevCtx* ctx, const int32_t* token_ids, const PieNemotronWeights* w,
    const int32_t* positions, void* out_logits, int32_t* out_token_ids, int32_t num_tokens,
    const char* kinds_host, int32_t hidden_size, int32_t vocab, int32_t mamba_num_heads,
    int32_t mamba_head_dim, int32_t mamba_state_size, int32_t mamba_n_groups,
    int32_t mamba_conv_kernel, float time_step_min, int32_t attn_n_q_heads,
    int32_t attn_n_kv_heads, int32_t attn_head_dim, int32_t page_size, float rope_theta,
    int32_t ffn_intermediate, float rms_eps);

// Per-layer MLA (multi-head latent attention) weights — device bf16, row-major.
// H = hidden_size, nh = num_heads. W_uk/W_uv are PRE-TRANSPOSED per head for the
// absorbed inference form (see device/src/forward/mla_block.cuh for the full
// layout contract): W_uk [nh, kv_lora_rank, qk_nope_head_dim], W_uv [nh,
// v_head_dim, kv_lora_rank]. W_q_a [q_lora_rank, H], W_q_b [nh*(qk_nope+qk_rope),
// q_lora_rank], W_kv_a [kv_lora_rank+qk_rope, H], W_o [H, nh*v_head_dim].
typedef struct PieMlaLayerWeights {
    const void* attn_norm;
    const void* w_q_a;
    const void* q_a_ln;
    const void* w_q_b;
    const void* w_kv_a;
    const void* kv_a_ln;
    const void* w_uk;
    const void* w_uv;
    const void* w_o;
} PieMlaLayerWeights;

// One DeepSeek-style MLA block in place on `hidden` [num_tokens, hidden_size]
// bf16 (absorbed form; latent paged attention). positions [num_tokens] int32.
// ckv_pages [num_pages, page_size, kv_lora_rank] / kpe_pages [num_pages,
// page_size, qk_rope_head_dim] bf16; CSR page lists on device (uint32).
PieStatus pie_cuda_mla_block_bf16(
    PieDevCtx* ctx, void* hidden, const PieMlaLayerWeights* w, const int32_t* positions,
    void* ckv_pages, void* kpe_pages, const uint32_t* qo_indptr_d,
    const uint32_t* kv_page_indices_d, const uint32_t* kv_page_indptr_d,
    const uint32_t* kv_last_page_lens_d, int32_t num_tokens, int32_t num_requests,
    int32_t hidden_size, int32_t num_heads, int32_t q_lora_rank, int32_t kv_lora_rank,
    int32_t qk_nope_head_dim, int32_t qk_rope_head_dim, int32_t v_head_dim,
    int32_t page_size, float rms_eps, float sm_scale, float rope_theta);

// Whole-model MLA weights. `layers` is a host array of `n_layers`
// PieMlaLayerWeights (each field a device pointer). embed/lm_head [vocab,
// hidden], final_norm [hidden] — device bf16.
typedef struct PieMlaWeights {
    const void* embed;
    const PieMlaLayerWeights* layers;
    int32_t n_layers;
    const void* final_norm;
    const void* lm_head;
} PieMlaWeights;

// Full MLA forward (prefill): embed → N MLA blocks → final norm → lm_head →
// argmax. token_ids/out_token_ids [num_tokens] int32 device; out_logits
// [num_tokens, vocab] bf16 device. ckv_pages [n_layers, num_pages, page_size,
// kv_lora_rank] / kpe_pages [n_layers, num_pages, page_size, qk_rope_head_dim]
// bf16 (layer L = slice L); CSR page lists shared across layers (uint32 device).
PieStatus pie_cuda_mla_forward_bf16(
    PieDevCtx* ctx, const int32_t* token_ids, const PieMlaWeights* w, const int32_t* positions,
    void* ckv_pages, void* kpe_pages, const uint32_t* qo_indptr_d,
    const uint32_t* kv_page_indices_d, const uint32_t* kv_page_indptr_d,
    const uint32_t* kv_last_page_lens_d, void* out_logits, int32_t* out_token_ids,
    int32_t num_tokens, int32_t num_requests, int32_t hidden_size, int32_t num_heads,
    int32_t q_lora_rank, int32_t kv_lora_rank, int32_t qk_nope_head_dim,
    int32_t qk_rope_head_dim, int32_t v_head_dim, int32_t vocab, int32_t page_size,
    int32_t num_pages, float rms_eps, float sm_scale, float rope_theta);

// Dense-MoE per-layer weights (device bf16, HF row-major). Hq=n_q_heads*head_dim,
// Hkv=n_kv_heads*head_dim, H=hidden, E=num_experts, I=intermediate. wq [Hq,H],
// wk/wv [Hkv,H], wo [H,Hq], router_w [E,H], wgu [E,2I,H] (gate||up), wdown [E,H,I].
typedef struct PieMoeLayerWeights {
    const void* attn_norm;
    const void* wq;
    const void* wk;
    const void* wv;
    const void* wo;
    const void* ffn_norm;
    const void* router_w;
    const void* wgu;
    const void* wdown;
} PieMoeLayerWeights;

// Whole-model dense-MoE weights. `layers` host array of `n_layers` per-layer
// structs. embed/lm_head [vocab,hidden], final_norm [hidden] — device bf16.
typedef struct PieMoeWeights {
    const void* embed;
    const PieMoeLayerWeights* layers;
    int32_t n_layers;
    const void* final_norm;
    const void* lm_head;
} PieMoeWeights;

// Full dense-MoE forward (prefill): embed → N×(llama attention + top-K MoE FFN)
// → final norm → lm_head → argmax. Per-layer paged KV pools kv_k/kv_v [n_layers,
// num_kv_pages, page_size, n_kv_heads, head_dim] bf16 (layer L = slice L); CSR
// page lists on device (uint32). out_logits [num_tokens,vocab] bf16, out_token_ids
// [num_tokens] i32.
PieStatus pie_cuda_moe_forward_bf16(
    PieDevCtx* ctx, const int32_t* token_ids, const PieMoeWeights* w, const int32_t* positions,
    void* kv_k, void* kv_v, const uint32_t* qo_indptr_d, const uint32_t* kv_page_indices_d,
    const uint32_t* kv_page_indptr_d, const uint32_t* kv_last_page_lens_d, void* out_logits,
    int32_t* out_token_ids, int32_t num_tokens, int32_t num_requests, int32_t num_kv_pages,
    int32_t hidden_size, int32_t n_q_heads, int32_t n_kv_heads, int32_t head_dim,
    int32_t intermediate, int32_t num_experts, int32_t top_k, int32_t vocab, int32_t page_size,
    float rms_eps, float rope_theta);

// Per-layer DeepSeek weights. `attn` = the 9 MLA attention pointers; `ffn_norm`
// + the dense (w_gate/w_up/w_down, used when the layer index < first_k_dense) OR
// MoE (router_w/wgu/wdown, used otherwise) FFN pointers. Unused set may be null.
typedef struct PieDeepseekLayerWeights {
    PieMlaLayerWeights attn;
    const void* ffn_norm;
    const void* w_gate;
    const void* w_up;
    const void* w_down;
    const void* router_w;
    const void* wgu;
    const void* wdown;
} PieDeepseekLayerWeights;

// Whole-model DeepSeek weights. `layers` host array of `n_layers` per-layer
// structs. embed/lm_head [vocab,hidden], final_norm [hidden] — device bf16.
typedef struct PieDeepseekWeights {
    const void* embed;
    const PieDeepseekLayerWeights* layers;
    int32_t n_layers;
    const void* final_norm;
    const void* lm_head;
} PieDeepseekWeights;

// Full DeepSeek-V3/V4 forward (prefill): embed → N×(MLA attention + dense|MoE
// FFN) → final norm → lm_head → argmax. Layers [0,first_k_dense) use the dense
// SwiGLU FFN, the rest use the top-K MoE FFN. ckv/kpe pages laid out [n_layers,
// num_pages, page_size, *] (layer L = slice L); CSR page lists shared (uint32).
PieStatus pie_cuda_deepseek_forward_bf16(
    PieDevCtx* ctx, const int32_t* token_ids, const PieDeepseekWeights* w,
    const int32_t* positions, void* ckv_pages, void* kpe_pages, const uint32_t* qo_indptr_d,
    const uint32_t* kv_page_indices_d, const uint32_t* kv_page_indptr_d,
    const uint32_t* kv_last_page_lens_d, void* out_logits, int32_t* out_token_ids,
    int32_t num_tokens, int32_t num_requests, int32_t first_k_dense, int32_t hidden_size,
    int32_t num_heads, int32_t q_lora_rank, int32_t kv_lora_rank, int32_t qk_nope_head_dim,
    int32_t qk_rope_head_dim, int32_t v_head_dim, int32_t dense_inter, int32_t moe_inter,
    int32_t num_experts, int32_t top_k, int32_t vocab, int32_t page_size, int32_t num_pages,
    float rms_eps, float sm_scale, float rope_theta);

// Per-layer Gemma "sandwich" weights (device bf16, row-major): 4 (1+w) RMSNorm
// gains + standard attention + geglu MLP projections. Hq=n_q_heads*head_dim,
// Hkv=n_kv_heads*head_dim. wq [Hq,H], wk/wv [Hkv,H], wo [H,Hq], w_gate/w_up
// [I,H], w_down [H,I].
typedef struct PieGemmaLayerWeights {
    const void* input_ln;
    const void* post_attn_ln;
    const void* pre_ffn_ln;
    const void* post_ffn_ln;
    const void* wq;
    const void* wk;
    const void* wv;
    const void* wo;
    const void* w_gate;
    const void* w_up;
    const void* w_down;
} PieGemmaLayerWeights;

// Whole-model Gemma weights. `layers` host array of `n_layers` per-layer structs.
// embed/lm_head [vocab,hidden], final_norm [hidden] (rmsnorm_gemma 1+w gain).
typedef struct PieGemmaWeights {
    const void* embed;
    const PieGemmaLayerWeights* layers;
    int32_t n_layers;
    const void* final_norm;
    const void* lm_head;
} PieGemmaWeights;

// Full Gemma-3/4 forward (prefill): embed ×embed_scale → N sandwich layers
// (sliding/full per `window_left`, attn logit soft-cap) → final norm → lm_head →
// final logit soft-cap → argmax. `window_left_host` is a host [n_layers] int
// array (per-layer left window; <0 = full); null → use `window_left_all`. Soft-
// caps <=0 disable. qk_norm must be 0 and altup_num_inputs 1 (both deferred).
PieStatus pie_cuda_gemma_forward_bf16(
    PieDevCtx* ctx, const int32_t* token_ids, const PieGemmaWeights* w, const int32_t* positions,
    void* k_pages, void* v_pages, const uint32_t* qo_indptr_d, const uint32_t* kv_page_indices_d,
    const uint32_t* kv_page_indptr_d, const uint32_t* kv_last_page_lens_d, void* out_logits,
    int32_t* out_token_ids, int32_t num_tokens, int32_t num_requests, int32_t hidden_size,
    int32_t n_q_heads, int32_t n_kv_heads, int32_t head_dim, int32_t intermediate, int32_t vocab,
    int32_t page_size, int32_t num_pages, const int32_t* window_left_host, int32_t window_left_all,
    float attn_logit_softcap, float final_logit_softcap, float embed_scale, float rms_eps,
    float rope_theta, int32_t qk_norm, int32_t altup_num_inputs);

// AltUp ("Alternating Updates", Gemma-3n/4): K parallel residual streams.
// predict: predictions[k,t,h] = streams[k,t,h] + Σ_j coefs[t,j,k]·streams[j,t,h].
// correct: corrected[k,t,h] = predictions[k,t,h] +
//          (activated[t,h] - predictions[active,t,h])·correction_coefs_p1[t,k].
// streams/predictions/corrected [K,T,H] bf16; coefs [T,K,K] / correction_coefs_p1
// [T,K] fp32 (reduce in fp32 to limit K-sum round-off). `active` is the active
// stream index; `correction_coefs_p1` already has the +1 folded in.
PieStatus pie_cuda_altup_predict_bf16(
    PieDevCtx* ctx, const void* streams, const float* coefs, void* predictions,
    int32_t k_streams, int32_t num_tokens, int32_t hidden_size);
PieStatus pie_cuda_altup_correct_bf16(
    PieDevCtx* ctx, const void* predictions, const void* activated,
    const float* correction_coefs_p1, void* corrected, int32_t k_streams,
    int32_t num_tokens, int32_t hidden_size, int32_t active_idx);

// Grouped per-expert GEMM (sparse MoE, after dispatch scatter): for every row r,
// y[r,:] = x[r,:] @ W_{e(r)}^T. x [total_rows,K] / w [E,N,K] / y [total_rows,N]
// device bf16, row-major; rows GROUPED BY EXPERT. `expert_offsets` is a [E+1]
// int32 **HOST** prefix sum (offsets[0]=0, offsets[E]=total_rows) — host because
// per-group counts must be known on the host to launch each cuBLAS call.
PieStatus pie_cuda_grouped_gemm_bf16(
    PieDevCtx* ctx, const void* x, const void* w, const int32_t* expert_offsets_host,
    void* y, int32_t total_rows, int32_t num_experts, int32_t n_out, int32_t k_in);

// Mamba-2 / SSD selective scan (Nemotron-H recurrence after the causal conv).
// conv_out [N, conv_dim] / dt [N, num_heads] bf16; A/D/dt_bias [num_heads] fp32;
// dt_precomputed/dA_precomputed [N, num_heads] fp32 (nullable → computed inline);
// ssm_state_base [slots, num_heads, head_dim, state_size] bf16 (read+written in
// place); slot_ids [R] int32 (nullable → slot 0); qo_indptr [R+1] uint32;
// y [N, intermediate] bf16. See device/src/kernels/ssm_scan.cuh for the recurrence
// + the conv_out packing (B/C grouped, conv_dim = intermediate + 2*n_groups*state).
PieStatus pie_cuda_ssm_selective_scan_bf16(
    PieDevCtx* ctx, const void* conv_out, const void* dt, const float* a, const float* d,
    const float* dt_bias, const float* dt_precomputed, const float* da_precomputed,
    void* ssm_state_base, const int32_t* slot_ids, const uint32_t* qo_indptr, void* y,
    int32_t num_requests, int32_t num_heads, int32_t head_dim, int32_t state_size,
    int32_t n_groups, int32_t conv_dim, int32_t intermediate, float time_step_min);

// Fused int4 (u4b8, GPTQ-style symmetric, implicit zero-point 8) weight-only
// GEMM — internalized de-branded Marlin. out[M,N] = act[M,K] @ dequant(qweight)^T,
// fp32 accumulate → bf16 out. `qweight_packed` is PREPACKED via
// pie_cuda_qgemm_w4a16_repack (NOT the raw checkpoint layout). scales_bf16
// [num_groups, N] bf16 (num_groups = group_size>0 ? K/group_size : 1; logical
// layout — the launcher permutes internally). `workspace` int32 scratch of at
// least pie_cuda_qgemm_w4a16_workspace_ints(N,M), ZEROED before each call (holds
// reduction locks). group_size ∈ {128,-1}; K%64==0, N%64==0. sms=0 auto-detects.
PieStatus pie_cuda_qgemm_w4a16_bf16(
    PieDevCtx* ctx, const void* act_bf16, const int32_t* qweight_packed,
    const void* scales_bf16, void* out_bf16, int32_t m, int32_t n, int32_t k,
    int32_t group_size, int32_t* workspace, int32_t sms);

// Prepack GPTQ-packed int4 weights [K/8, N] int32 (8 nibbles/int32 along K;
// nibble j of (kp,n) = int4 for k=kp*8+j, col n, stored unsigned [0,15]) into the
// kernel's tile/interleave layout [K/16, N*16/8] int32. Call once per weight.
PieStatus pie_cuda_qgemm_w4a16_repack(
    PieDevCtx* ctx, const int32_t* qweight_rowmajor_packed, int32_t* qweight_out,
    int32_t n, int32_t k);

// Required w4a16 workspace size, in int32 elements (allocate once at init; the
// caller must zero it before each pie_cuda_qgemm_w4a16_bf16 call).
int32_t pie_cuda_qgemm_w4a16_workspace_ints(int32_t n, int32_t max_m);

// fp8 (fe4m3fn) weight-only fused GEMM — the 8-bit fan-out over the same qgemm
// template. out[M,N] = act[M,K] @ dequant(qweight)^T. `qweight_fp8` is PREPACKED
// via pie_cuda_qgemm_w8a16_fp8_repack (input [K/4,N] int32, 4 fe4m3fn bytes per
// int32 along K). scales_bf16 [num_groups,N] logical (launcher permutes + folds
// the fp8 exponent bias). workspace zeroed each call. group_size ∈ {128,-1}.
PieStatus pie_cuda_qgemm_w8a16_fp8_bf16(
    PieDevCtx* ctx, const void* act_bf16, const void* qweight_fp8, const void* scales_bf16,
    void* out_bf16, int32_t m, int32_t n, int32_t k, int32_t group_size, int32_t* workspace,
    int32_t sms);
PieStatus pie_cuda_qgemm_w8a16_fp8_repack(
    PieDevCtx* ctx, const int32_t* qweight_rowmajor_packed, int32_t* qweight_out, int32_t n,
    int32_t k);
int32_t pie_cuda_qgemm_w8a16_fp8_workspace_ints(int32_t n, int32_t max_m);

// ---------------------------------------------------------------------------
// Construction primitives (Rust builder drives these; replaces run_impl's
// alloc cascade). Weight binding consumes a handle from pie-weight-loader.
// ---------------------------------------------------------------------------

// `loader` is an opaque pointer into the Rust storage-program loader
// (pie-weight-loader); the device lib calls back to stream tensors H2D.
PieStatus pie_weights_bind(PieDevCtx* ctx, PieArchId arch, void* loader,
                           PieWeights** out_weights);
PieStatus pie_weights_destroy(PieWeights* weights);

PieStatus pie_kv_alloc(PieDevCtx* ctx, const PieKvLayout* layout,
                       PieKvCache** out_kv);
PieStatus pie_kv_destroy(PieKvCache* kv);

PieStatus pie_ws_alloc(PieDevCtx* ctx, const PieWorkspaceDims* dims,
                       PieWorkspace** out_ws);
PieStatus pie_ws_destroy(PieWorkspace* ws);

// Bytes for one KV page in the given layout (memory planner arithmetic).
size_t pie_kv_page_bytes(const PieKvLayout* layout);

// ---------------------------------------------------------------------------
// Hot path — one call each. Cheap across FFI.
// ---------------------------------------------------------------------------

// Refresh the workspace's pinned/device input buffers from `in`. Pointer
// stability of these buffers is what makes graph replay safe; Rust calls
// this every fire (direct, capture, or replay).
PieStatus pie_upload_inputs(PieWorkspace* ws, const PieForwardInputs* in);

// Host-side plan (flashinfer DecodePlan etc.) → device buffers. Must run
// before every fire, including before graph replay.
PieStatus pie_prepare(PieArchId arch, PieWorkspace* ws,
                      const PiePrepareInputs* in);

// Device-side forward body — the kernel sequence. No host loops/allocs;
// re-reads buffers that `pie_upload_inputs` / `pie_prepare` refreshed.
PieStatus pie_body(PieArchId arch, PieWeights* weights, PieWorkspace* ws,
                   PieKvCache* kv, const PieForwardInputs* in);

// Sample tokens from the workspace's logits into `out_tokens` (device,
// [num_rows]). Greedy fast path when params->greedy.
PieStatus pie_sample(PieWorkspace* ws, const PieSampleParams* params,
                     int32_t* out_tokens);

// ---------------------------------------------------------------------------
// CUDA graph — Rust owns policy (bucket choice, when to capture); the
// device lib owns the cudaGraph mechanics. Capture wraps a single body
// fire on the context stream.
// ---------------------------------------------------------------------------

PieStatus pie_graph_capture(PieDevCtx* ctx, PieArchId arch, PieWeights* weights,
                            PieWorkspace* ws, PieKvCache* kv,
                            const PieForwardInputs* in, PieGraphExec** out_exec);
PieStatus pie_graph_launch(PieGraphExec* exec, PieDevCtx* ctx);
PieStatus pie_graph_destroy(PieGraphExec* exec);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // PIE_CUDA_DEVICE_H
