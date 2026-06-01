//! Raw `extern "C"` declarations over `device/include/pie_cuda_device.h`.
//!
//! Hand-written rather than bindgen'd: the ABI is small, curated, and
//! stable, and a hand-written surface is the single readable place to
//! audit the boundary. Keep it byte-for-byte in lockstep with the header
//! (PIE_CUDA_DEVICE_ABI_VERSION gates a runtime check in `device.rs`).
#![allow(non_camel_case_types, dead_code)]

use std::os::raw::{c_char, c_void};

pub const ABI_VERSION: u32 = 32;

#[repr(C)]
#[derive(Copy, Clone, Eq, PartialEq, Debug)]
pub enum PieStatus {
    Ok = 0,
    InvalidArg = 1,
    Cuda = 2,
    Oom = 3,
    UnsupportedArch = 4,
    Internal = 5,
}

// Mirrors PieArchId in the header. Keep discriminants in lockstep.
#[repr(i32)]
#[derive(Copy, Clone, Eq, PartialEq, Debug)]
pub enum PieArchId {
    LlamaLike = 0,
    Qwen3 = 1,
    Qwen3_5 = 2,
    Qwen3_5Moe = 3,
    Mixtral = 4,
    Gemma2 = 5,
    Gemma3n = 6,
    Gemma4 = 7,
    NemotronH = 8,
    DeepseekV4 = 9,
    Kimi = 10,
    Glm5 = 11,
    GptOss = 12,
}

// Opaque handles — never constructed in Rust, only held as pointers.
#[repr(C)] pub struct PieDevCtx { _p: [u8; 0] }
#[repr(C)] pub struct PieWeights { _p: [u8; 0] }
#[repr(C)] pub struct PieKvCache { _p: [u8; 0] }
#[repr(C)] pub struct PieWorkspace { _p: [u8; 0] }
#[repr(C)] pub struct PieGraphExec { _p: [u8; 0] }

#[repr(C)]
pub struct PieForwardInputs {
    pub token_ids: *const i32,
    pub positions: *const i32,
    pub qo_indptr_d: *const u32,
    pub kv_page_indices_d: *const u32,
    pub kv_page_indptr_d: *const u32,
    pub kv_last_page_lens_d: *const u32,
    pub qo_indptr_h: *const u32,
    pub kv_page_indices_h: *const u32,
    pub kv_page_indptr_h: *const u32,
    pub kv_last_page_lens_h: *const u32,
    pub total_tokens: i32,
    pub num_requests: i32,
    pub is_pure_decode: i32,
    pub custom_mask_d: *const u8,
    pub custom_mask_indptr_d: *const i32,
    pub slot_ids_h: *const i32,
    pub is_fresh_h: *const u8,
    pub slot_ids_d: *const i32,
    pub logit_row_indices_d: *const i32,
    pub num_logit_rows: i32,
    pub tp_greedy_argmax: i32,
    pub commit_advance_gather_d: *const i32,
}

#[repr(C)]
pub struct PiePrepareInputs {
    pub qo_indptr_h: *const u32,
    pub kv_page_indices_h: *const u32,
    pub kv_page_indices_d: *const u32,
    pub kv_page_indptr_h: *const u32,
    pub kv_page_indptr_d: *const u32,
    pub kv_last_page_lens_h: *const u32,
    pub kv_last_page_lens_d: *const u32,
    pub total_tokens: i32,
    pub num_requests: i32,
    pub is_pure_decode: i32,
}

#[repr(C)]
pub struct PieSampleParams {
    pub temperature: *const f32,
    pub top_p: *const f32,
    pub top_k: *const i32,
    pub seed: *const u64,
    pub num_rows: i32,
    pub greedy: i32,
}

#[repr(C)]
pub struct PieKvLayout {
    pub num_layers: i32,
    pub num_pages: i32,
    pub page_size: i32,
    pub num_kv_heads: i32,
    pub head_dim: i32,
    pub format: u32,
    pub hnd_layout: i32,
    pub per_layer_head_dim: *const i32,
    pub per_layer_num_kv_heads: *const i32,
    pub kv_source_layer: *const i32,
}

#[repr(C)]
pub struct PieLlamaLayerWeights {
    pub attn_norm: *const c_void,
    pub wq: *const c_void,
    pub wk: *const c_void,
    pub wv: *const c_void,
    pub wo: *const c_void,
    pub ffn_norm: *const c_void,
    pub w_gate: *const c_void,
    pub w_up: *const c_void,
    pub w_down: *const c_void,
    pub q_norm: *const c_void,
    pub k_norm: *const c_void,
    pub q_bias: *const c_void,
    pub k_bias: *const c_void,
    pub v_bias: *const c_void,
}

#[repr(C)]
pub struct PieLlamaWeights {
    pub embed: *const c_void,
    pub layers: *const PieLlamaLayerWeights,
    pub n_layers: i32,
    pub final_norm: *const c_void,
    pub lm_head: *const c_void,
}

#[repr(C)]
pub struct PieMlaLayerWeights {
    pub attn_norm: *const c_void,
    pub w_q_a: *const c_void,
    pub q_a_ln: *const c_void,
    pub w_q_b: *const c_void,
    pub w_kv_a: *const c_void,
    pub kv_a_ln: *const c_void,
    pub w_uk: *const c_void,
    pub w_uv: *const c_void,
    pub w_o: *const c_void,
}

#[repr(C)]
pub struct PieMlaWeights {
    pub embed: *const c_void,
    pub layers: *const PieMlaLayerWeights,
    pub n_layers: i32,
    pub final_norm: *const c_void,
    pub lm_head: *const c_void,
}

#[repr(C)]
pub struct PieGemmaLayerWeights {
    pub input_ln: *const c_void,
    pub post_attn_ln: *const c_void,
    pub pre_ffn_ln: *const c_void,
    pub post_ffn_ln: *const c_void,
    pub wq: *const c_void,
    pub wk: *const c_void,
    pub wv: *const c_void,
    pub wo: *const c_void,
    pub w_gate: *const c_void,
    pub w_up: *const c_void,
    pub w_down: *const c_void,
}

#[repr(C)]
pub struct PieGemmaWeights {
    pub embed: *const c_void,
    pub layers: *const PieGemmaLayerWeights,
    pub n_layers: i32,
    pub final_norm: *const c_void,
    pub lm_head: *const c_void,
}

#[repr(C)]
pub struct PieDeepseekLayerWeights {
    pub attn: PieMlaLayerWeights,
    pub ffn_norm: *const c_void,
    pub w_gate: *const c_void,
    pub w_up: *const c_void,
    pub w_down: *const c_void,
    pub router_w: *const c_void,
    pub wgu: *const c_void,
    pub wdown: *const c_void,
}

#[repr(C)]
pub struct PieDeepseekWeights {
    pub embed: *const c_void,
    pub layers: *const PieDeepseekLayerWeights,
    pub n_layers: i32,
    pub final_norm: *const c_void,
    pub lm_head: *const c_void,
}

#[repr(C)]
pub struct PieNemotronAttnWeights {
    pub attn_norm: *const c_void,
    pub wq: *const c_void,
    pub wk: *const c_void,
    pub wv: *const c_void,
    pub wo: *const c_void,
}

#[repr(C)]
pub struct PieNemotronFfnWeights {
    pub ffn_norm: *const c_void,
    pub w_gate: *const c_void,
    pub w_up: *const c_void,
    pub w_down: *const c_void,
}

#[repr(C)]
pub struct PieNemotronLayerWeights {
    pub kind: i8, // 'M' | 'A' | 'F'
    pub mamba_pre_norm: *const c_void,
    pub mamba: PieNemotronMambaWeights,
    pub attn: PieNemotronAttnWeights,
    pub ffn: PieNemotronFfnWeights,
}

#[repr(C)]
pub struct PieNemotronWeights {
    pub embed: *const c_void,
    pub layers: *const PieNemotronLayerWeights,
    pub n_layers: i32,
    pub final_norm: *const c_void,
    pub lm_head: *const c_void,
}

#[repr(C)]
pub struct PieNemotronMambaWeights {
    pub in_proj_w: *const c_void,
    pub conv_w: *const c_void,
    pub conv_bias: *const c_void,
    pub a_log: *const c_void,
    pub d: *const c_void,
    pub dt_bias: *const c_void,
    pub norm_weight: *const c_void,
    pub out_proj_w: *const c_void,
}

#[repr(C)]
pub struct PieMoeLayerWeights {
    pub attn_norm: *const c_void,
    pub wq: *const c_void,
    pub wk: *const c_void,
    pub wv: *const c_void,
    pub wo: *const c_void,
    pub ffn_norm: *const c_void,
    pub router_w: *const c_void,
    pub wgu: *const c_void,
    pub wdown: *const c_void,
}

#[repr(C)]
pub struct PieMoeWeights {
    pub embed: *const c_void,
    pub layers: *const PieMoeLayerWeights,
    pub n_layers: i32,
    pub final_norm: *const c_void,
    pub lm_head: *const c_void,
}

#[repr(C)]
pub struct PieWorkspaceDims {
    pub max_tokens: i32,
    pub max_requests: i32,
    pub hidden_size: i32,
    pub intermediate_size: i32,
    pub num_heads: i32,
    pub num_kv_heads: i32,
    pub head_dim: i32,
    pub vocab_size: i32,
    pub num_layers: i32,
    pub recurrent_state_slots: i32,
    pub moe_experts: i32,
}

unsafe extern "C" {
    pub fn pie_cuda_abi_version() -> u32;
    pub fn pie_cuda_last_error() -> *const c_char;

    pub fn pie_cuda_ctx_create(device_ordinal: i32, out: *mut *mut PieDevCtx) -> PieStatus;
    pub fn pie_cuda_ctx_destroy(ctx: *mut PieDevCtx) -> PieStatus;
    pub fn pie_cuda_mem_info(ctx: *mut PieDevCtx, free: *mut usize, total: *mut usize) -> PieStatus;
    pub fn pie_cuda_device_props(ctx: *mut PieDevCtx, sm_count: *mut i32, major: *mut i32,
                                 minor: *mut i32) -> PieStatus;

    pub fn pie_cuda_malloc(ctx: *mut PieDevCtx, nbytes: usize, out: *mut *mut c_void) -> PieStatus;
    pub fn pie_cuda_free(ctx: *mut PieDevCtx, ptr: *mut c_void) -> PieStatus;
    pub fn pie_cuda_memcpy_h2d(ctx: *mut PieDevCtx, dst: *mut c_void, src: *const c_void, nbytes: usize) -> PieStatus;
    pub fn pie_cuda_memcpy_d2h(ctx: *mut PieDevCtx, dst: *mut c_void, src: *const c_void, nbytes: usize) -> PieStatus;
    pub fn pie_cuda_memcpy_d2d(ctx: *mut PieDevCtx, dst: *mut c_void, src: *const c_void, nbytes: usize) -> PieStatus;
    pub fn pie_cuda_stream_sync(ctx: *mut PieDevCtx) -> PieStatus;

    pub fn pie_cuda_rmsnorm_bf16(ctx: *mut PieDevCtx, x: *const c_void, weight: *const c_void,
                                 y: *mut c_void, num_rows: i32, hidden: i32, eps: f32) -> PieStatus;
    pub fn pie_cuda_residual_add_bf16(ctx: *mut PieDevCtx, y: *mut c_void, x: *const c_void,
                                      n: usize) -> PieStatus;
    pub fn pie_cuda_swiglu_bf16(ctx: *mut PieDevCtx, gate: *const c_void, up: *const c_void,
                                y: *mut c_void, num_elements: i32) -> PieStatus;
    pub fn pie_cuda_rope_bf16(ctx: *mut PieDevCtx, q: *mut c_void, k: *mut c_void,
                              positions: *const i32, num_tokens: i32, num_q_heads: i32,
                              num_kv_heads: i32, head_dim: i32, theta: f32,
                              interleaved: i32) -> PieStatus;
    pub fn pie_cuda_gemm_bf16(ctx: *mut PieDevCtx, act: *const c_void, w: *const c_void,
                              y: *mut c_void, m: i32, n: i32, k: i32, beta: f32) -> PieStatus;
    pub fn pie_cuda_embed_bf16(ctx: *mut PieDevCtx, token_ids: *const i32, weight: *const c_void,
                               y: *mut c_void, num_tokens: i32, hidden: i32, vocab: i32) -> PieStatus;
    pub fn pie_cuda_argmax_bf16(ctx: *mut PieDevCtx, logits: *const c_void, token_ids: *mut i32,
                                num_rows: i32, vocab: i32) -> PieStatus;
    pub fn pie_cuda_write_kv_to_pages_bf16(
        ctx: *mut PieDevCtx, k_pages: *mut c_void, v_pages: *mut c_void, k_curr: *const c_void,
        v_curr: *const c_void, qo_indptr_d: *const u32, kv_page_indices_d: *const u32,
        kv_page_indptr_d: *const u32, kv_last_page_lens_d: *const u32, total_tokens: i32,
        num_requests: i32, page_size: i32, num_kv_heads: i32, head_dim: i32,
        hnd_layout: i32) -> PieStatus;
    pub fn pie_cuda_sample_temp_bf16(
        ctx: *mut PieDevCtx, logits: *const c_void, temperatures: *const f32, top_ps: *const f32,
        top_ks: *const i32, min_ps: *const f32, seeds: *const u32, out: *mut i32, num_rows: i32,
        vocab: i32) -> PieStatus;
    pub fn pie_cuda_cast_fp16_to_bf16(ctx: *mut PieDevCtx, src: *const c_void, dst: *mut c_void, n: usize) -> PieStatus;
    pub fn pie_cuda_cast_fp32_to_bf16(ctx: *mut PieDevCtx, src: *const c_void, dst: *mut c_void, n: usize) -> PieStatus;
    pub fn pie_cuda_cast_bf16_to_fp32(ctx: *mut PieDevCtx, src: *const c_void, dst: *mut c_void, n: usize) -> PieStatus;
    pub fn pie_cuda_gather_bf16_rows(
        ctx: *mut PieDevCtx, src: *const u16, row_indices: *const i32, dst: *mut u16,
        num_dst_rows: i32, vocab: i32) -> PieStatus;
    pub fn pie_cuda_rmsnorm_gemma_bf16(ctx: *mut PieDevCtx, x: *const c_void, weight: *const c_void,
                                       y: *mut c_void, num_rows: i32, hidden: i32, eps: f32) -> PieStatus;
    pub fn pie_cuda_geglu_tanh_bf16(ctx: *mut PieDevCtx, gate: *const c_void, up: *const c_void,
                                    y: *mut c_void, num_elements: i32) -> PieStatus;
    pub fn pie_cuda_logit_softcap_bf16(ctx: *mut PieDevCtx, x: *mut c_void, cap: f32, n: usize) -> PieStatus;
    pub fn pie_cuda_dequant_fp8_e4m3_to_bf16(ctx: *mut PieDevCtx, fp8_in: *const u8,
                                             bf16_out: *mut c_void, scale: f32, n: usize) -> PieStatus;
    pub fn pie_cuda_rope_yarn_bf16(
        ctx: *mut PieDevCtx, q: *mut c_void, k: *mut c_void, positions: *const i32,
        num_tokens: i32, num_q_heads: i32, num_kv_heads: i32, head_dim: i32, theta: f32,
        factor: f32, low_freq_factor: f32, high_freq_factor: f32,
        original_max_position: i32) -> PieStatus;
    pub fn pie_cuda_topk_softmax_bf16(
        ctx: *mut PieDevCtx, logits: *const c_void, topk_idx: *mut i32, topk_w: *mut f32,
        n: i32, num_experts: i32, k: i32) -> PieStatus;
    pub fn pie_cuda_chunked_swiglu_bf16(ctx: *mut PieDevCtx, packed: *const c_void, y: *mut c_void,
                                        n: i32, i: i32) -> PieStatus;
    pub fn pie_cuda_rope_partial_bf16(
        ctx: *mut PieDevCtx, q: *mut c_void, k: *mut c_void, positions: *const i32,
        num_tokens: i32, num_q_heads: i32, num_kv_heads: i32, head_dim: i32, rotary_dim: i32,
        theta: f32) -> PieStatus;
    pub fn pie_cuda_causal_conv1d_prefill_bf16(
        ctx: *mut PieDevCtx, x: *const c_void, weight: *const c_void, bias: *const c_void,
        y: *mut c_void, n: i32, c: i32, k: i32) -> PieStatus;
    pub fn pie_cuda_dequant_wna16_int4b8_to_bf16(
        ctx: *mut PieDevCtx, packed: *const i32, scale_bf16: *const c_void, out_bf16: *mut c_void,
        out_dim: i32, in_dim: i32, group_size: i32) -> PieStatus;
    pub fn pie_cuda_moe_mlp_block_bf16(
        ctx: *mut PieDevCtx, hidden: *const c_void, router_w: *const c_void, wgu: *const c_void,
        wdown: *const c_void, out: *mut c_void, num_tokens: i32, hidden_size: i32,
        intermediate: i32, num_experts: i32, top_k: i32) -> PieStatus;
    pub fn pie_cuda_mla_block_bf16(
        ctx: *mut PieDevCtx, hidden: *mut c_void, w: *const PieMlaLayerWeights,
        positions: *const i32, ckv_pages: *mut c_void, kpe_pages: *mut c_void,
        qo_indptr_d: *const u32, kv_page_indices_d: *const u32, kv_page_indptr_d: *const u32,
        kv_last_page_lens_d: *const u32, num_tokens: i32, num_requests: i32, hidden_size: i32,
        num_heads: i32, q_lora_rank: i32, kv_lora_rank: i32, qk_nope_head_dim: i32,
        qk_rope_head_dim: i32, v_head_dim: i32, page_size: i32, rms_eps: f32, sm_scale: f32,
        rope_theta: f32) -> PieStatus;
    pub fn pie_cuda_altup_predict_bf16(
        ctx: *mut PieDevCtx, streams: *const c_void, coefs: *const f32, predictions: *mut c_void,
        k_streams: i32, num_tokens: i32, hidden_size: i32) -> PieStatus;
    pub fn pie_cuda_altup_correct_bf16(
        ctx: *mut PieDevCtx, predictions: *const c_void, activated: *const c_void,
        correction_coefs_p1: *const f32, corrected: *mut c_void, k_streams: i32,
        num_tokens: i32, hidden_size: i32, active_idx: i32) -> PieStatus;
    pub fn pie_cuda_grouped_gemm_bf16(
        ctx: *mut PieDevCtx, x: *const c_void, w: *const c_void, expert_offsets_host: *const i32,
        y: *mut c_void, total_rows: i32, num_experts: i32, n_out: i32, k_in: i32) -> PieStatus;
    pub fn pie_cuda_mla_forward_bf16(
        ctx: *mut PieDevCtx, token_ids: *const i32, w: *const PieMlaWeights, positions: *const i32,
        ckv_pages: *mut c_void, kpe_pages: *mut c_void, qo_indptr_d: *const u32,
        kv_page_indices_d: *const u32, kv_page_indptr_d: *const u32,
        kv_last_page_lens_d: *const u32, out_logits: *mut c_void, out_token_ids: *mut i32,
        num_tokens: i32, num_requests: i32, hidden_size: i32, num_heads: i32, q_lora_rank: i32,
        kv_lora_rank: i32, qk_nope_head_dim: i32, qk_rope_head_dim: i32, v_head_dim: i32,
        vocab: i32, page_size: i32, num_pages: i32, rms_eps: f32, sm_scale: f32,
        rope_theta: f32) -> PieStatus;
    pub fn pie_cuda_deepseek_forward_bf16(
        ctx: *mut PieDevCtx, token_ids: *const i32, w: *const PieDeepseekWeights,
        positions: *const i32, ckv_pages: *mut c_void, kpe_pages: *mut c_void,
        qo_indptr_d: *const u32, kv_page_indices_d: *const u32, kv_page_indptr_d: *const u32,
        kv_last_page_lens_d: *const u32, out_logits: *mut c_void, out_token_ids: *mut i32,
        num_tokens: i32, num_requests: i32, first_k_dense: i32, hidden_size: i32, num_heads: i32,
        q_lora_rank: i32, kv_lora_rank: i32, qk_nope_head_dim: i32, qk_rope_head_dim: i32,
        v_head_dim: i32, dense_inter: i32, moe_inter: i32, num_experts: i32, top_k: i32,
        vocab: i32, page_size: i32, num_pages: i32, rms_eps: f32, sm_scale: f32,
        rope_theta: f32) -> PieStatus;
    pub fn pie_cuda_gemma_forward_bf16(
        ctx: *mut PieDevCtx, ws: *mut PieWorkspace, token_ids: *const i32,
        w: *const PieGemmaWeights, positions: *const i32,
        k_pages: *mut c_void, v_pages: *mut c_void, qo_indptr_d: *const u32,
        kv_page_indices_d: *const u32, kv_page_indptr_d: *const u32, kv_last_page_lens_d: *const u32,
        out_logits: *mut c_void, out_token_ids: *mut i32, num_tokens: i32, num_requests: i32,
        hidden_size: i32, n_q_heads: i32, n_kv_heads: i32, head_dim: i32, intermediate: i32,
        vocab: i32, page_size: i32, num_pages: i32, window_left_host: *const i32,
        window_left_all: i32, attn_logit_softcap: f32, final_logit_softcap: f32, embed_scale: f32,
        rms_eps: f32, rope_theta: f32, qk_norm: i32, altup_num_inputs: i32) -> PieStatus;
    pub fn pie_cuda_nemotron_forward_bf16(
        ctx: *mut PieDevCtx, token_ids: *const i32, w: *const PieNemotronWeights,
        positions: *const i32, out_logits: *mut c_void, out_token_ids: *mut i32, num_tokens: i32,
        kinds_host: *const i8, hidden_size: i32, vocab: i32, mamba_num_heads: i32,
        mamba_head_dim: i32, mamba_state_size: i32, mamba_n_groups: i32, mamba_conv_kernel: i32,
        time_step_min: f32, attn_n_q_heads: i32, attn_n_kv_heads: i32, attn_head_dim: i32,
        page_size: i32, rope_theta: f32, ffn_intermediate: i32, rms_eps: f32) -> PieStatus;
    pub fn pie_cuda_moe_forward_bf16(
        ctx: *mut PieDevCtx, token_ids: *const i32, w: *const PieMoeWeights, positions: *const i32,
        kv_k: *mut c_void, kv_v: *mut c_void, qo_indptr_d: *const u32,
        kv_page_indices_d: *const u32, kv_page_indptr_d: *const u32,
        kv_last_page_lens_d: *const u32, out_logits: *mut c_void, out_token_ids: *mut i32,
        num_tokens: i32, num_requests: i32, num_kv_pages: i32, hidden_size: i32, n_q_heads: i32,
        n_kv_heads: i32, head_dim: i32, intermediate: i32, num_experts: i32, top_k: i32,
        vocab: i32, page_size: i32, rms_eps: f32, rope_theta: f32) -> PieStatus;
    pub fn pie_cuda_ssm_selective_scan_bf16(
        ctx: *mut PieDevCtx, conv_out: *const c_void, dt: *const c_void, a: *const f32,
        d: *const f32, dt_bias: *const f32, dt_precomputed: *const f32, da_precomputed: *const f32,
        ssm_state_base: *mut c_void, slot_ids: *const i32, qo_indptr: *const u32, y: *mut c_void,
        num_requests: i32, num_heads: i32, head_dim: i32, state_size: i32, n_groups: i32,
        conv_dim: i32, intermediate: i32, time_step_min: f32) -> PieStatus;
    pub fn pie_cuda_qgemm_w4a16_bf16(
        ctx: *mut PieDevCtx, act_bf16: *const c_void, qweight_packed: *const i32,
        scales_bf16: *const c_void, out_bf16: *mut c_void, m: i32, n: i32, k: i32,
        group_size: i32, workspace: *mut i32, sms: i32) -> PieStatus;
    pub fn pie_cuda_qgemm_w4a16_repack(
        ctx: *mut PieDevCtx, qweight_rowmajor_packed: *const i32, qweight_out: *mut i32,
        n: i32, k: i32) -> PieStatus;
    pub fn pie_cuda_qgemm_w4a16_workspace_ints(n: i32, max_m: i32) -> i32;
    pub fn pie_cuda_qgemm_w8a16_fp8_bf16(
        ctx: *mut PieDevCtx, act_bf16: *const c_void, qweight_fp8: *const c_void,
        scales_bf16: *const c_void, out_bf16: *mut c_void, m: i32, n: i32, k: i32,
        group_size: i32, workspace: *mut i32, sms: i32) -> PieStatus;
    pub fn pie_cuda_qgemm_w8a16_fp8_repack(
        ctx: *mut PieDevCtx, qweight_rowmajor_packed: *const i32, qweight_out: *mut i32,
        n: i32, k: i32) -> PieStatus;
    pub fn pie_cuda_qgemm_w8a16_fp8_workspace_ints(n: i32, max_m: i32) -> i32;
    pub fn pie_cuda_moe_sparse_block_bf16(
        ctx: *mut PieDevCtx, hidden: *const c_void, router_w: *const c_void, wgu: *const c_void,
        wdown: *const c_void, out: *mut c_void, num_tokens: i32, hidden_size: i32,
        intermediate: i32, num_experts: i32, top_k: i32) -> PieStatus;
    pub fn pie_cuda_nemotron_mamba_block_bf16(
        ctx: *mut PieDevCtx, hidden: *mut c_void, w: *const PieNemotronMambaWeights,
        num_tokens: i32, hidden_size: i32, num_heads: i32, head_dim: i32, state_size: i32,
        n_groups: i32, conv_kernel: i32, rms_eps: f32, time_step_min: f32) -> PieStatus;
    pub fn pie_cuda_attention_naive_paged_bf16(
        ctx: *mut PieDevCtx, q: *const c_void, k_pages: *const c_void, v_pages: *const c_void,
        o: *mut c_void, qo_indptr_d: *const u32, kv_page_indices_d: *const u32,
        kv_page_indptr_d: *const u32, kv_last_page_lens_d: *const u32, total_tokens: i32,
        num_requests: i32, num_q_heads: i32, num_kv_heads: i32, head_dim: i32, page_size: i32,
        window_left: i32, sm_scale: f32) -> PieStatus;
    pub fn pie_cuda_llama_layer_bf16(
        ctx: *mut PieDevCtx, hidden: *mut c_void, w: *const PieLlamaLayerWeights,
        positions: *const i32, k_pages: *mut c_void, v_pages: *mut c_void,
        qo_indptr_d: *const u32, kv_page_indices_d: *const u32, kv_page_indptr_d: *const u32,
        kv_last_page_lens_d: *const u32, num_tokens: i32, num_requests: i32, hidden_size: i32,
        n_q_heads: i32, n_kv_heads: i32, head_dim: i32, intermediate: i32, page_size: i32,
        rms_eps: f32, rope_theta: f32) -> PieStatus;
    pub fn pie_cuda_llama_forward_bf16(
        ctx: *mut PieDevCtx, ws: *mut PieWorkspace, token_ids: *const i32,
        w: *const PieLlamaWeights, positions: *const i32, kv_k: *mut c_void, kv_v: *mut c_void,
        qo_indptr_d: *const u32, kv_page_indices_d: *const u32, kv_page_indptr_d: *const u32,
        kv_last_page_lens_d: *const u32, out_logits: *mut c_void, out_token_ids: *mut i32,
        num_tokens: i32, num_requests: i32, hidden_size: i32, n_q_heads: i32, n_kv_heads: i32,
        head_dim: i32, intermediate: i32, page_size: i32, num_kv_pages: i32, vocab: i32,
        rms_eps: f32, rope_theta: f32) -> PieStatus;

    pub fn pie_weights_bind(ctx: *mut PieDevCtx, arch: PieArchId, loader: *mut c_void,
                            out: *mut *mut PieWeights) -> PieStatus;
    pub fn pie_weights_destroy(w: *mut PieWeights) -> PieStatus;
    pub fn pie_kv_alloc(ctx: *mut PieDevCtx, layout: *const PieKvLayout,
                        out: *mut *mut PieKvCache) -> PieStatus;
    pub fn pie_kv_destroy(kv: *mut PieKvCache) -> PieStatus;
    pub fn pie_ws_alloc(ctx: *mut PieDevCtx, dims: *const PieWorkspaceDims,
                        out: *mut *mut PieWorkspace) -> PieStatus;
    pub fn pie_ws_destroy(ws: *mut PieWorkspace) -> PieStatus;
    pub fn pie_kv_page_bytes(layout: *const PieKvLayout) -> usize;

    pub fn pie_upload_inputs(ws: *mut PieWorkspace, inp: *const PieForwardInputs) -> PieStatus;
    pub fn pie_prepare(arch: PieArchId, ws: *mut PieWorkspace,
                       inp: *const PiePrepareInputs) -> PieStatus;
    pub fn pie_body(arch: PieArchId, w: *mut PieWeights, ws: *mut PieWorkspace,
                    kv: *mut PieKvCache, inp: *const PieForwardInputs) -> PieStatus;
    pub fn pie_sample(ws: *mut PieWorkspace, params: *const PieSampleParams,
                      out_tokens: *mut i32) -> PieStatus;

    pub fn pie_graph_capture(ctx: *mut PieDevCtx, arch: PieArchId, w: *mut PieWeights,
                             ws: *mut PieWorkspace, kv: *mut PieKvCache,
                             inp: *const PieForwardInputs,
                             out: *mut *mut PieGraphExec) -> PieStatus;
    pub fn pie_graph_launch(exec: *mut PieGraphExec, ctx: *mut PieDevCtx) -> PieStatus;
    pub fn pie_graph_destroy(exec: *mut PieGraphExec) -> PieStatus;
}
