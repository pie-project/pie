//! Safe RAII wrappers over the raw `ffi` handles. Each owns a device-lib
//! pointer and calls the matching destroy entry on `Drop`. This is the
//! only module that touches raw pointers; everything above (executor,
//! builder, sampler) works in safe Rust.

use std::ffi::CStr;
use std::os::raw::c_void;

use anyhow::{Result, bail};

use crate::ffi;

/// Translate a `PieStatus` into a `Result`, pulling the device lib's
/// thread-local error string on failure.
fn check(status: ffi::PieStatus, what: &str) -> Result<()> {
    if status == ffi::PieStatus::Ok {
        return Ok(());
    }
    // SAFETY: pie_cuda_last_error returns a thread-local C string valid
    // until the next ABI call on this thread; we copy it immediately.
    let detail = unsafe {
        let p = ffi::pie_cuda_last_error();
        if p.is_null() {
            String::new()
        } else {
            CStr::from_ptr(p).to_string_lossy().into_owned()
        }
    };
    bail!("{what} failed: {status:?}{}", if detail.is_empty() {
        String::new()
    } else {
        format!(" — {detail}")
    });
}

/// Owns a `PieDevCtx`. Created once per device at startup.
pub struct Device(*mut ffi::PieDevCtx);

impl Device {
    pub fn new(ordinal: i32) -> Result<Self> {
        // Fail fast if the Rust ABI view drifts from the linked library.
        let linked = unsafe { ffi::pie_cuda_abi_version() };
        if linked != ffi::ABI_VERSION {
            bail!("pie_cuda_device ABI mismatch: header {}, linked {linked}", ffi::ABI_VERSION);
        }
        let mut ctx = std::ptr::null_mut();
        check(unsafe { ffi::pie_cuda_ctx_create(ordinal, &mut ctx) }, "pie_cuda_ctx_create")?;
        Ok(Device(ctx))
    }

    pub(crate) fn raw(&self) -> *mut ffi::PieDevCtx {
        self.0
    }

    /// (free, total) device bytes — feeds the memory planner.
    pub fn mem_info(&self) -> Result<(usize, usize)> {
        let (mut free, mut total) = (0usize, 0usize);
        check(
            unsafe { ffi::pie_cuda_mem_info(self.0, &mut free, &mut total) },
            "pie_cuda_mem_info",
        )?;
        Ok((free, total))
    }

    /// (sm_count, compute_major, compute_minor) — feeds `mem::plan`.
    pub fn props(&self) -> Result<(i32, i32, i32)> {
        let (mut sm, mut major, mut minor) = (0i32, 0i32, 0i32);
        check(
            unsafe { ffi::pie_cuda_device_props(self.0, &mut sm, &mut major, &mut minor) },
            "pie_cuda_device_props",
        )?;
        Ok((sm, major, minor))
    }

    /// Allocate a raw device buffer of `nbytes`. Freed on `DeviceBuffer` drop.
    pub fn alloc(&self, nbytes: usize) -> Result<DeviceBuffer<'_>> {
        let mut ptr: *mut c_void = std::ptr::null_mut();
        check(unsafe { ffi::pie_cuda_malloc(self.0, nbytes, &mut ptr) }, "pie_cuda_malloc")?;
        Ok(DeviceBuffer { dev: self, ptr, len: nbytes })
    }

    /// Block until all stream work (copies, kernels) has completed. Call
    /// before reading a D2H result on the host.
    pub fn sync(&self) -> Result<()> {
        check(unsafe { ffi::pie_cuda_stream_sync(self.0) }, "pie_cuda_stream_sync")
    }

    /// Row-wise bf16 RMSNorm: `y = x * rsqrt(mean(x^2)+eps) * weight`.
    /// Buffers hold bf16 row-major data ([rows, hidden] / [hidden]).
    pub fn rmsnorm_bf16(&self, x: &DeviceBuffer, weight: &DeviceBuffer, y: &DeviceBuffer,
                        rows: i32, hidden: i32, eps: f32) -> Result<()> {
        check(
            unsafe {
                ffi::pie_cuda_rmsnorm_bf16(self.0, x.ptr, weight.ptr, y.ptr, rows, hidden, eps)
            },
            "pie_cuda_rmsnorm_bf16",
        )
    }

    /// In-place bf16 residual add: `y[i] = round_bf16(y[i] + x[i])`.
    pub fn residual_add_bf16(&self, y: &DeviceBuffer, x: &DeviceBuffer, n: usize) -> Result<()> {
        check(
            unsafe { ffi::pie_cuda_residual_add_bf16(self.0, y.ptr, x.ptr, n) },
            "pie_cuda_residual_add_bf16",
        )
    }

    /// SwiGLU MLP activation: `y = silu(gate) * up` (elementwise bf16).
    pub fn swiglu_bf16(&self, gate: &DeviceBuffer, up: &DeviceBuffer, y: &DeviceBuffer,
                       num_elements: i32) -> Result<()> {
        check(
            unsafe { ffi::pie_cuda_swiglu_bf16(self.0, gate.ptr, up.ptr, y.ptr, num_elements) },
            "pie_cuda_swiglu_bf16",
        )
    }

    /// In-place RoPE on Q/K (bf16). `positions` holds `num_tokens` i32 values.
    /// `interleaved=false` → NeoX (Llama/Qwen); `true` → GPT-J (GLM).
    #[allow(clippy::too_many_arguments)]
    pub fn rope_bf16(&self, q: &DeviceBuffer, k: &DeviceBuffer, positions: &DeviceBuffer,
                     num_tokens: i32, num_q_heads: i32, num_kv_heads: i32, head_dim: i32,
                     theta: f32, interleaved: bool) -> Result<()> {
        check(
            unsafe {
                ffi::pie_cuda_rope_bf16(self.0, q.ptr, k.ptr, positions.ptr as *const i32,
                    num_tokens, num_q_heads, num_kv_heads, head_dim, theta, interleaved as i32)
            },
            "pie_cuda_rope_bf16",
        )
    }

    /// bf16 GEMM `y = act @ Wᵀ + beta*y`. act [M,K], W [N,K] row-major; y [M,N].
    pub fn gemm_bf16(&self, act: &DeviceBuffer, w: &DeviceBuffer, y: &DeviceBuffer,
                     m: i32, n: i32, k: i32, beta: f32) -> Result<()> {
        check(
            unsafe { ffi::pie_cuda_gemm_bf16(self.0, act.ptr, w.ptr, y.ptr, m, n, k, beta) },
            "pie_cuda_gemm_bf16",
        )
    }

    /// Embedding lookup: `y[n,:] = weight[token_ids[n], :]`. `token_ids` holds
    /// `num_tokens` i32; `weight` is [vocab, hidden] bf16; `y` is bf16.
    pub fn embed_bf16(&self, token_ids: &DeviceBuffer, weight: &DeviceBuffer, y: &DeviceBuffer,
                      num_tokens: i32, hidden: i32, vocab: i32) -> Result<()> {
        check(
            unsafe {
                ffi::pie_cuda_embed_bf16(self.0, token_ids.ptr as *const i32, weight.ptr, y.ptr,
                    num_tokens, hidden, vocab)
            },
            "pie_cuda_embed_bf16",
        )
    }

    /// Per-row greedy argmax over [num_rows, vocab] bf16 logits → i32 token ids
    /// (lowest-index tie-break). `vocab` should be even (vectorized kernel path).
    pub fn argmax_bf16(&self, logits: &DeviceBuffer, token_ids: &DeviceBuffer,
                       num_rows: i32, vocab: i32) -> Result<()> {
        check(
            unsafe {
                ffi::pie_cuda_argmax_bf16(self.0, logits.ptr, token_ids.ptr as *mut i32, num_rows, vocab)
            },
            "pie_cuda_argmax_bf16",
        )
    }

    /// Naive paged-KV causal attention. q/o [total_tokens, num_q_heads, head_dim]
    /// bf16; k/v pages [num_pages, page_size, num_kv_heads, head_dim] bf16. CSR
    /// page lists are u32 device buffers. `sm_scale < 0` → 1/√head_dim;
    /// `window_left < 0` → full causal.
    #[allow(clippy::too_many_arguments)]
    pub fn attention_naive_paged_bf16(
        &self, q: &DeviceBuffer, k_pages: &DeviceBuffer, v_pages: &DeviceBuffer, o: &DeviceBuffer,
        qo_indptr: &DeviceBuffer, kv_page_indices: &DeviceBuffer, kv_page_indptr: &DeviceBuffer,
        kv_last_page_lens: &DeviceBuffer, total_tokens: i32, num_requests: i32, num_q_heads: i32,
        num_kv_heads: i32, head_dim: i32, page_size: i32, window_left: i32, sm_scale: f32,
    ) -> Result<()> {
        check(
            unsafe {
                ffi::pie_cuda_attention_naive_paged_bf16(
                    self.0, q.ptr, k_pages.ptr, v_pages.ptr, o.ptr,
                    qo_indptr.ptr as *const u32, kv_page_indices.ptr as *const u32,
                    kv_page_indptr.ptr as *const u32, kv_last_page_lens.ptr as *const u32,
                    total_tokens, num_requests, num_q_heads, num_kv_heads, head_dim, page_size,
                    window_left, sm_scale)
            },
            "pie_cuda_attention_naive_paged_bf16",
        )
    }

    /// Append freshly-computed K/V into the paged cache (`hnd_layout=false` → NHD).
    #[allow(clippy::too_many_arguments)]
    pub fn write_kv_to_pages_bf16(
        &self, k_pages: &DeviceBuffer, v_pages: &DeviceBuffer, k_curr: &DeviceBuffer,
        v_curr: &DeviceBuffer, qo_indptr: &DeviceBuffer, kv_page_indices: &DeviceBuffer,
        kv_page_indptr: &DeviceBuffer, kv_last_page_lens: &DeviceBuffer, total_tokens: i32,
        num_requests: i32, page_size: i32, num_kv_heads: i32, head_dim: i32, hnd_layout: bool,
    ) -> Result<()> {
        check(
            unsafe {
                ffi::pie_cuda_write_kv_to_pages_bf16(
                    self.0, k_pages.ptr, v_pages.ptr, k_curr.ptr, v_curr.ptr,
                    qo_indptr.ptr as *const u32, kv_page_indices.ptr as *const u32,
                    kv_page_indptr.ptr as *const u32, kv_last_page_lens.ptr as *const u32,
                    total_tokens, num_requests, page_size, num_kv_heads, head_dim,
                    hnd_layout as i32)
            },
            "pie_cuda_write_kv_to_pages_bf16",
        )
    }

    /// Per-row temperature sampling (Gumbel-max) with optional top-p / top-k /
    /// min-p truncation. temps/seeds/out are [num_rows]; temp<=0 → argmax.
    /// `top_ps`/`top_ks`/`min_ps` are each `Some([num_rows] device buffer)` or
    /// `None` (that filter off for all rows); the kept set is their
    /// intersection. top_p in (0,1), top_k>0, min_p>0 enable each filter.
    #[allow(clippy::too_many_arguments)]
    pub fn sample_temp_bf16(
        &self, logits: &DeviceBuffer, temperatures: &DeviceBuffer,
        top_ps: Option<&DeviceBuffer>, top_ks: Option<&DeviceBuffer>, min_ps: Option<&DeviceBuffer>,
        seeds: &DeviceBuffer, out: &DeviceBuffer, num_rows: i32, vocab: i32,
    ) -> Result<()> {
        let top_ps_ptr = top_ps.map_or(std::ptr::null(), |b| b.ptr as *const f32);
        let top_ks_ptr = top_ks.map_or(std::ptr::null(), |b| b.ptr as *const i32);
        let min_ps_ptr = min_ps.map_or(std::ptr::null(), |b| b.ptr as *const f32);
        check(
            unsafe {
                ffi::pie_cuda_sample_temp_bf16(self.0, logits.ptr, temperatures.ptr as *const f32,
                    top_ps_ptr, top_ks_ptr, min_ps_ptr, seeds.ptr as *const u32,
                    out.ptr as *mut i32, num_rows, vocab)
            },
            "pie_cuda_sample_temp_bf16",
        )
    }

    /// Element-wise cast (`n` elements). fp16/fp32 → bf16 (round-nearest-even).
    pub fn cast_fp16_to_bf16(&self, src: &DeviceBuffer, dst: &DeviceBuffer, n: usize) -> Result<()> {
        check(unsafe { ffi::pie_cuda_cast_fp16_to_bf16(self.0, src.ptr, dst.ptr, n) }, "pie_cuda_cast_fp16_to_bf16")
    }
    pub fn cast_fp32_to_bf16(&self, src: &DeviceBuffer, dst: &DeviceBuffer, n: usize) -> Result<()> {
        check(unsafe { ffi::pie_cuda_cast_fp32_to_bf16(self.0, src.ptr, dst.ptr, n) }, "pie_cuda_cast_fp32_to_bf16")
    }
    pub fn cast_bf16_to_fp32(&self, src: &DeviceBuffer, dst: &DeviceBuffer, n: usize) -> Result<()> {
        check(unsafe { ffi::pie_cuda_cast_bf16_to_fp32(self.0, src.ptr, dst.ptr, n) }, "pie_cuda_cast_bf16_to_fp32")
    }

    /// Gather rows of a [_, vocab] bf16 buffer into [num_dst_rows, vocab] by index.
    pub fn gather_bf16_rows(&self, src: &DeviceBuffer, row_indices: &DeviceBuffer,
                            dst: &DeviceBuffer, num_dst_rows: i32, vocab: i32) -> Result<()> {
        check(
            unsafe {
                ffi::pie_cuda_gather_bf16_rows(self.0, src.ptr as *const u16,
                    row_indices.ptr as *const i32, dst.ptr as *mut u16, num_dst_rows, vocab)
            },
            "pie_cuda_gather_bf16_rows",
        )
    }

    /// YaRN RoPE on Q/K (in place). `factor=1.0` ⇒ un-scaled base RoPE.
    #[allow(clippy::too_many_arguments)]
    pub fn rope_yarn_bf16(&self, q: &DeviceBuffer, k: &DeviceBuffer, positions: &DeviceBuffer,
                          num_tokens: i32, num_q_heads: i32, num_kv_heads: i32, head_dim: i32,
                          theta: f32, factor: f32, low_freq_factor: f32, high_freq_factor: f32,
                          original_max_position: i32) -> Result<()> {
        check(unsafe {
            ffi::pie_cuda_rope_yarn_bf16(self.0, q.ptr, k.ptr, positions.ptr as *const i32,
                num_tokens, num_q_heads, num_kv_heads, head_dim, theta, factor, low_freq_factor,
                high_freq_factor, original_max_position)
        }, "pie_cuda_rope_yarn_bf16")
    }

    /// MoE router: top-K experts + softmax (over all) renormalized weights.
    pub fn topk_softmax_bf16(&self, logits: &DeviceBuffer, topk_idx: &DeviceBuffer,
                             topk_w: &DeviceBuffer, n: i32, num_experts: i32, k: i32) -> Result<()> {
        check(unsafe {
            ffi::pie_cuda_topk_softmax_bf16(self.0, logits.ptr, topk_idx.ptr as *mut i32,
                topk_w.ptr as *mut f32, n, num_experts, k)
        }, "pie_cuda_topk_softmax_bf16")
    }

    /// Fused MoE expert activation: `y[n,i] = silu(packed[n,i]) * packed[n,I+i]`.
    pub fn chunked_swiglu_bf16(&self, packed: &DeviceBuffer, y: &DeviceBuffer, n: i32, i: i32) -> Result<()> {
        check(unsafe { ffi::pie_cuda_chunked_swiglu_bf16(self.0, packed.ptr, y.ptr, n, i) },
              "pie_cuda_chunked_swiglu_bf16")
    }

    /// Partial RoPE (Gemma-4): rotate the first `rotary_dim` of each head; rest pass through.
    #[allow(clippy::too_many_arguments)]
    pub fn rope_partial_bf16(&self, q: &DeviceBuffer, k: &DeviceBuffer, positions: &DeviceBuffer,
                             num_tokens: i32, num_q_heads: i32, num_kv_heads: i32, head_dim: i32,
                             rotary_dim: i32, theta: f32) -> Result<()> {
        check(unsafe {
            ffi::pie_cuda_rope_partial_bf16(self.0, q.ptr, k.ptr, positions.ptr as *const i32,
                num_tokens, num_q_heads, num_kv_heads, head_dim, rotary_dim, theta)
        }, "pie_cuda_rope_partial_bf16")
    }

    /// Causal depthwise conv1d (Mamba), prefill, SiLU-fused. x/y [N,C], weight [C,K], bias [C] (optional).
    pub fn causal_conv1d_prefill_bf16(&self, x: &DeviceBuffer, weight: &DeviceBuffer,
                                      bias: Option<&DeviceBuffer>, y: &DeviceBuffer,
                                      n: i32, c: i32, k: i32) -> Result<()> {
        let bias_ptr = bias.map(|b| b.ptr as *const c_void).unwrap_or(std::ptr::null());
        check(unsafe {
            ffi::pie_cuda_causal_conv1d_prefill_bf16(self.0, x.ptr, weight.ptr, bias_ptr, y.ptr, n, c, k)
        }, "pie_cuda_causal_conv1d_prefill_bf16")
    }

    /// WNA16 int4 (uint4b8) → bf16 group-wise dequant. packed is int32; scale/out bf16.
    pub fn dequant_wna16_int4b8_to_bf16(&self, packed: &DeviceBuffer, scale_bf16: &DeviceBuffer,
                                        out_bf16: &DeviceBuffer, out_dim: i32, in_dim: i32,
                                        group_size: i32) -> Result<()> {
        check(unsafe {
            ffi::pie_cuda_dequant_wna16_int4b8_to_bf16(self.0, packed.ptr as *const i32,
                scale_bf16.ptr, out_bf16.ptr, out_dim, in_dim, group_size)
        }, "pie_cuda_dequant_wna16_int4b8_to_bf16")
    }

    /// Dense MoE MLP block: router → top-K softmax → per-expert
    /// (gate_up → swiglu → down) → weighted combine. hidden/out [T,H];
    /// router_w [E,H]; wgu [E,2I,H] (gate||up); wdown [E,H,I]. All bf16.
    #[allow(clippy::too_many_arguments)]
    pub fn moe_mlp_block_bf16(&self, hidden: &DeviceBuffer, router_w: &DeviceBuffer,
                              wgu: &DeviceBuffer, wdown: &DeviceBuffer, out: &DeviceBuffer,
                              num_tokens: i32, hidden_size: i32, intermediate: i32,
                              num_experts: i32, top_k: i32) -> Result<()> {
        check(unsafe {
            ffi::pie_cuda_moe_mlp_block_bf16(self.0, hidden.ptr, router_w.ptr, wgu.ptr, wdown.ptr,
                out.ptr, num_tokens, hidden_size, intermediate, num_experts, top_k)
        }, "pie_cuda_moe_mlp_block_bf16")
    }

    /// Gemma RMSNorm: `y = (1 + weight) * x_hat`.
    pub fn rmsnorm_gemma_bf16(&self, x: &DeviceBuffer, weight: &DeviceBuffer, y: &DeviceBuffer,
                              rows: i32, hidden: i32, eps: f32) -> Result<()> {
        check(unsafe { ffi::pie_cuda_rmsnorm_gemma_bf16(self.0, x.ptr, weight.ptr, y.ptr, rows, hidden, eps) },
              "pie_cuda_rmsnorm_gemma_bf16")
    }

    /// Gemma MLP activation: `y = gelu_tanh(gate) * up` (elementwise bf16).
    pub fn geglu_tanh_bf16(&self, gate: &DeviceBuffer, up: &DeviceBuffer, y: &DeviceBuffer,
                           num_elements: i32) -> Result<()> {
        check(unsafe { ffi::pie_cuda_geglu_tanh_bf16(self.0, gate.ptr, up.ptr, y.ptr, num_elements) },
              "pie_cuda_geglu_tanh_bf16")
    }

    /// In-place logit softcap: `x = cap * tanh(x / cap)` over `n` bf16 elems.
    pub fn logit_softcap_bf16(&self, x: &DeviceBuffer, cap: f32, n: usize) -> Result<()> {
        check(unsafe { ffi::pie_cuda_logit_softcap_bf16(self.0, x.ptr, cap, n) },
              "pie_cuda_logit_softcap_bf16")
    }

    /// FP8 (E4M3) → bf16 dequant with a scalar scale: `out = e4m3(in) * scale`.
    pub fn dequant_fp8_e4m3_to_bf16(&self, fp8_in: &DeviceBuffer, bf16_out: &DeviceBuffer,
                                    scale: f32, n: usize) -> Result<()> {
        check(unsafe { ffi::pie_cuda_dequant_fp8_e4m3_to_bf16(self.0, fp8_in.ptr as *const u8, bf16_out.ptr, scale, n) },
              "pie_cuda_dequant_fp8_e4m3_to_bf16")
    }

    /// One llama-like decoder layer in place on `hidden` [num_tokens, hidden]
    /// bf16. Composes the lifted primitives C++-side (v1 vertical slice — see
    /// the device lib's forward/llama_layer.cuh for slice simplifications).
    #[allow(clippy::too_many_arguments)]
    pub fn llama_layer_bf16(
        &self, hidden: &DeviceBuffer, w: &LlamaLayerWeights, positions: &DeviceBuffer,
        k_pages: &DeviceBuffer, v_pages: &DeviceBuffer, qo_indptr: &DeviceBuffer,
        kv_page_indices: &DeviceBuffer, kv_page_indptr: &DeviceBuffer,
        kv_last_page_lens: &DeviceBuffer, dims: &LlamaLayerDims,
    ) -> Result<()> {
        let nul = std::ptr::null::<c_void>();
        let fw = ffi::PieLlamaLayerWeights {
            attn_norm: w.attn_norm.ptr, wq: w.wq.ptr, wk: w.wk.ptr, wv: w.wv.ptr,
            wo: w.wo.ptr, ffn_norm: w.ffn_norm.ptr, w_gate: w.w_gate.ptr,
            w_up: w.w_up.ptr, w_down: w.w_down.ptr,
            q_norm: w.q_norm.map_or(nul, |b| b.ptr as *const c_void),
            k_norm: w.k_norm.map_or(nul, |b| b.ptr as *const c_void),
            q_bias: w.q_bias.map_or(nul, |b| b.ptr as *const c_void),
            k_bias: w.k_bias.map_or(nul, |b| b.ptr as *const c_void),
            v_bias: w.v_bias.map_or(nul, |b| b.ptr as *const c_void),
        };
        check(
            unsafe {
                ffi::pie_cuda_llama_layer_bf16(
                    self.0, hidden.ptr, &fw, positions.ptr as *const i32, k_pages.ptr,
                    v_pages.ptr, qo_indptr.ptr as *const u32, kv_page_indices.ptr as *const u32,
                    kv_page_indptr.ptr as *const u32, kv_last_page_lens.ptr as *const u32,
                    dims.num_tokens, dims.num_requests, dims.hidden_size, dims.n_q_heads,
                    dims.n_kv_heads, dims.head_dim, dims.intermediate, dims.page_size,
                    dims.rms_eps, dims.rope_theta)
            },
            "pie_cuda_llama_layer_bf16",
        )
    }

    /// One DeepSeek-style MLA block in place on `hidden` [num_tokens, hidden]
    /// bf16 (absorbed form, latent paged attention). W_uk/W_uv pre-transposed.
    #[allow(clippy::too_many_arguments)]
    pub fn mla_block_bf16(
        &self, hidden: &DeviceBuffer, w: &MlaLayerWeights, positions: &DeviceBuffer,
        ckv_pages: &DeviceBuffer, kpe_pages: &DeviceBuffer, qo_indptr: &DeviceBuffer,
        kv_page_indices: &DeviceBuffer, kv_page_indptr: &DeviceBuffer,
        kv_last_page_lens: &DeviceBuffer, dims: &MlaBlockDims,
    ) -> Result<()> {
        let fw = ffi::PieMlaLayerWeights {
            attn_norm: w.attn_norm.ptr, w_q_a: w.w_q_a.ptr, q_a_ln: w.q_a_ln.ptr,
            w_q_b: w.w_q_b.ptr, w_kv_a: w.w_kv_a.ptr, kv_a_ln: w.kv_a_ln.ptr,
            w_uk: w.w_uk.ptr, w_uv: w.w_uv.ptr, w_o: w.w_o.ptr,
        };
        check(
            unsafe {
                ffi::pie_cuda_mla_block_bf16(
                    self.0, hidden.ptr, &fw, positions.ptr as *const i32, ckv_pages.ptr,
                    kpe_pages.ptr, qo_indptr.ptr as *const u32, kv_page_indices.ptr as *const u32,
                    kv_page_indptr.ptr as *const u32, kv_last_page_lens.ptr as *const u32,
                    dims.num_tokens, dims.num_requests, dims.hidden_size, dims.num_heads,
                    dims.q_lora_rank, dims.kv_lora_rank, dims.qk_nope_head_dim,
                    dims.qk_rope_head_dim, dims.v_head_dim, dims.page_size, dims.rms_eps,
                    dims.sm_scale, dims.rope_theta)
            },
            "pie_cuda_mla_block_bf16",
        )
    }

    /// AltUp predict: predictions[k,t,h] = streams[k,t,h] + Σ_j coefs[t,j,k]·
    /// streams[j,t,h]. streams/predictions [K,T,H] bf16; coefs [T,K,K] fp32.
    pub fn altup_predict_bf16(&self, streams: &DeviceBuffer, coefs: &DeviceBuffer,
                              predictions: &DeviceBuffer, k: i32, t: i32, h: i32) -> Result<()> {
        check(unsafe {
            ffi::pie_cuda_altup_predict_bf16(self.0, streams.ptr, coefs.ptr as *const f32,
                predictions.ptr, k, t, h)
        }, "pie_cuda_altup_predict_bf16")
    }

    /// AltUp correct: corrected[k,t,h] = predictions[k,t,h] + (activated[t,h] -
    /// predictions[active,t,h])·correction_coefs_p1[t,k]. predictions/corrected
    /// [K,T,H] bf16; activated [T,H] bf16; correction_coefs_p1 [T,K] fp32 (+1 folded).
    #[allow(clippy::too_many_arguments)]
    pub fn altup_correct_bf16(&self, predictions: &DeviceBuffer, activated: &DeviceBuffer,
                              correction_coefs_p1: &DeviceBuffer, corrected: &DeviceBuffer,
                              k: i32, t: i32, h: i32, active_idx: i32) -> Result<()> {
        check(unsafe {
            ffi::pie_cuda_altup_correct_bf16(self.0, predictions.ptr, activated.ptr,
                correction_coefs_p1.ptr as *const f32, corrected.ptr, k, t, h, active_idx)
        }, "pie_cuda_altup_correct_bf16")
    }

    /// Grouped per-expert GEMM (sparse MoE): y[r,:] = x[r,:] @ W_{e(r)}^T, rows
    /// grouped by expert. x [total_rows,K], w [E,N,K], y [total_rows,N] device
    /// bf16. `expert_offsets` is a HOST [E+1] prefix sum (offsets[0]=0,
    /// offsets[E]=total_rows).
    #[allow(clippy::too_many_arguments)]
    pub fn grouped_gemm_bf16(&self, x: &DeviceBuffer, w: &DeviceBuffer, expert_offsets: &[i32],
                             y: &DeviceBuffer, total_rows: i32, num_experts: i32, n_out: i32,
                             k_in: i32) -> Result<()> {
        debug_assert_eq!(expert_offsets.len(), num_experts as usize + 1);
        check(unsafe {
            ffi::pie_cuda_grouped_gemm_bf16(self.0, x.ptr, w.ptr, expert_offsets.as_ptr(), y.ptr,
                total_rows, num_experts, n_out, k_in)
        }, "pie_cuda_grouped_gemm_bf16")
    }

    /// Full MLA forward: embed → N MLA blocks (each with its own per-layer cache
    /// slice) → final RMSNorm → lm_head → argmax. Writes out_logits [num_tokens,
    /// vocab] bf16 + out_token_ids [num_tokens] i32. ckv/kpe pages are laid out
    /// [n_layers, num_pages, page_size, *]; CSR page lists shared across layers.
    #[allow(clippy::too_many_arguments)]
    pub fn mla_forward_bf16(
        &self, token_ids: &DeviceBuffer, embed: &DeviceBuffer, layers: &[MlaLayerWeights],
        final_norm: &DeviceBuffer, lm_head: &DeviceBuffer, positions: &DeviceBuffer,
        ckv_pages: &DeviceBuffer, kpe_pages: &DeviceBuffer, qo_indptr: &DeviceBuffer,
        kv_page_indices: &DeviceBuffer, kv_page_indptr: &DeviceBuffer,
        kv_last_page_lens: &DeviceBuffer, out_logits: &DeviceBuffer, out_token_ids: &DeviceBuffer,
        num_tokens: i32, num_requests: i32, dims: &MlaForwardDims,
    ) -> Result<()> {
        let ffi_layers: Vec<ffi::PieMlaLayerWeights> = layers.iter().map(|w| {
            ffi::PieMlaLayerWeights {
                attn_norm: w.attn_norm.ptr, w_q_a: w.w_q_a.ptr, q_a_ln: w.q_a_ln.ptr,
                w_q_b: w.w_q_b.ptr, w_kv_a: w.w_kv_a.ptr, kv_a_ln: w.kv_a_ln.ptr,
                w_uk: w.w_uk.ptr, w_uv: w.w_uv.ptr, w_o: w.w_o.ptr,
            }
        }).collect();
        let bundle = ffi::PieMlaWeights {
            embed: embed.ptr, layers: ffi_layers.as_ptr(), n_layers: layers.len() as i32,
            final_norm: final_norm.ptr, lm_head: lm_head.ptr,
        };
        check(
            unsafe {
                ffi::pie_cuda_mla_forward_bf16(
                    self.0, token_ids.ptr as *const i32, &bundle, positions.ptr as *const i32,
                    ckv_pages.ptr, kpe_pages.ptr, qo_indptr.ptr as *const u32,
                    kv_page_indices.ptr as *const u32, kv_page_indptr.ptr as *const u32,
                    kv_last_page_lens.ptr as *const u32, out_logits.ptr,
                    out_token_ids.ptr as *mut i32, num_tokens, num_requests, dims.hidden,
                    dims.num_heads, dims.q_lora_rank, dims.kv_lora_rank, dims.qk_nope_head_dim,
                    dims.qk_rope_head_dim, dims.v_head_dim, dims.vocab, dims.page_size,
                    dims.num_pages, dims.rms_eps, dims.sm_scale, dims.rope_theta)
            },
            "pie_cuda_mla_forward_bf16",
        )
    }

    /// Full DeepSeek-V3/V4 forward: embed → N×(MLA attention + dense|MoE FFN) →
    /// final RMSNorm → lm_head → argmax. Layers `[0, first_k_dense)` use the dense
    /// SwiGLU FFN (`w_gate/w_up/w_down`), the rest the top-K MoE FFN
    /// (`router_w/wgu/wdown`); the unused set per layer is `None`.
    #[allow(clippy::too_many_arguments)]
    pub fn deepseek_forward_bf16(
        &self, token_ids: &DeviceBuffer, embed: &DeviceBuffer, layers: &[DeepseekLayerWeights],
        final_norm: &DeviceBuffer, lm_head: &DeviceBuffer, positions: &DeviceBuffer,
        ckv_pages: &DeviceBuffer, kpe_pages: &DeviceBuffer, qo_indptr: &DeviceBuffer,
        kv_page_indices: &DeviceBuffer, kv_page_indptr: &DeviceBuffer,
        kv_last_page_lens: &DeviceBuffer, out_logits: &DeviceBuffer, out_token_ids: &DeviceBuffer,
        num_tokens: i32, num_requests: i32, dims: &DeepseekForwardDims,
    ) -> Result<()> {
        let opt = |o: Option<&DeviceBuffer>| o.map_or(std::ptr::null::<c_void>(), |b| b.ptr as *const c_void);
        let ffi_layers: Vec<ffi::PieDeepseekLayerWeights> = layers.iter().map(|l| {
            ffi::PieDeepseekLayerWeights {
                attn: ffi::PieMlaLayerWeights {
                    attn_norm: l.attn.attn_norm.ptr, w_q_a: l.attn.w_q_a.ptr,
                    q_a_ln: l.attn.q_a_ln.ptr, w_q_b: l.attn.w_q_b.ptr, w_kv_a: l.attn.w_kv_a.ptr,
                    kv_a_ln: l.attn.kv_a_ln.ptr, w_uk: l.attn.w_uk.ptr, w_uv: l.attn.w_uv.ptr,
                    w_o: l.attn.w_o.ptr,
                },
                ffn_norm: l.ffn_norm.ptr,
                w_gate: opt(l.w_gate), w_up: opt(l.w_up), w_down: opt(l.w_down),
                router_w: opt(l.router_w), wgu: opt(l.wgu), wdown: opt(l.wdown),
            }
        }).collect();
        let bundle = ffi::PieDeepseekWeights {
            embed: embed.ptr, layers: ffi_layers.as_ptr(), n_layers: layers.len() as i32,
            final_norm: final_norm.ptr, lm_head: lm_head.ptr,
        };
        check(
            unsafe {
                ffi::pie_cuda_deepseek_forward_bf16(
                    self.0, token_ids.ptr as *const i32, &bundle, positions.ptr as *const i32,
                    ckv_pages.ptr, kpe_pages.ptr, qo_indptr.ptr as *const u32,
                    kv_page_indices.ptr as *const u32, kv_page_indptr.ptr as *const u32,
                    kv_last_page_lens.ptr as *const u32, out_logits.ptr,
                    out_token_ids.ptr as *mut i32, num_tokens, num_requests, dims.first_k_dense,
                    dims.hidden, dims.num_heads, dims.q_lora_rank, dims.kv_lora_rank,
                    dims.qk_nope_head_dim, dims.qk_rope_head_dim, dims.v_head_dim, dims.dense_inter,
                    dims.moe_inter, dims.num_experts, dims.top_k, dims.vocab, dims.page_size,
                    dims.num_pages, dims.rms_eps, dims.sm_scale, dims.rope_theta)
            },
            "pie_cuda_deepseek_forward_bf16",
        )
    }

    /// Full Gemma-3/4 forward: embed ×embed_scale → N sandwich layers (per-layer
    /// sliding/full via `dims.window_left`, attn soft-cap) → final norm → lm_head
    /// → final soft-cap → argmax. (qk-norm/AltUp deferred: qk_norm=0, altup=1.)
    #[allow(clippy::too_many_arguments)]
    pub fn gemma_forward_bf16(
        &self, ws: &Workspace, token_ids: &DeviceBuffer, embed: &DeviceBuffer,
        layers: &[GemmaLayerWeights],
        final_norm: &DeviceBuffer, lm_head: &DeviceBuffer, positions: &DeviceBuffer,
        k_pages: &DeviceBuffer, v_pages: &DeviceBuffer, qo_indptr: &DeviceBuffer,
        kv_page_indices: &DeviceBuffer, kv_page_indptr: &DeviceBuffer,
        kv_last_page_lens: &DeviceBuffer, out_logits: &DeviceBuffer, out_token_ids: &DeviceBuffer,
        num_tokens: i32, num_requests: i32, dims: &GemmaForwardDims,
    ) -> Result<()> {
        let ffi_layers: Vec<ffi::PieGemmaLayerWeights> = layers.iter().map(|l| {
            ffi::PieGemmaLayerWeights {
                input_ln: l.input_ln.ptr, post_attn_ln: l.post_attn_ln.ptr,
                pre_ffn_ln: l.pre_ffn_ln.ptr, post_ffn_ln: l.post_ffn_ln.ptr, wq: l.wq.ptr,
                wk: l.wk.ptr, wv: l.wv.ptr, wo: l.wo.ptr, w_gate: l.w_gate.ptr, w_up: l.w_up.ptr,
                w_down: l.w_down.ptr,
            }
        }).collect();
        let bundle = ffi::PieGemmaWeights {
            embed: embed.ptr, layers: ffi_layers.as_ptr(), n_layers: layers.len() as i32,
            final_norm: final_norm.ptr, lm_head: lm_head.ptr,
        };
        let wl = if dims.window_left.is_empty() {
            std::ptr::null()
        } else {
            dims.window_left.as_ptr()
        };
        check(
            unsafe {
                ffi::pie_cuda_gemma_forward_bf16(
                    self.0, ws.ptr, token_ids.ptr as *const i32, &bundle,
                    positions.ptr as *const i32,
                    k_pages.ptr, v_pages.ptr, qo_indptr.ptr as *const u32,
                    kv_page_indices.ptr as *const u32, kv_page_indptr.ptr as *const u32,
                    kv_last_page_lens.ptr as *const u32, out_logits.ptr,
                    out_token_ids.ptr as *mut i32, num_tokens, num_requests, dims.hidden,
                    dims.n_q_heads, dims.n_kv_heads, dims.head_dim, dims.intermediate, dims.vocab,
                    dims.page_size, dims.num_pages, wl, dims.window_left_all,
                    dims.attn_logit_softcap, dims.final_logit_softcap, dims.embed_scale,
                    dims.rms_eps, dims.rope_theta, dims.qk_norm, dims.altup_num_inputs)
            },
            "pie_cuda_gemma_forward_bf16",
        )
    }

    /// Full Nemotron-H forward: embed → N hybrid layers (per `layers[L]` kind:
    /// Mamba mixer | GQA attention | dense SwiGLU FFN) → final norm → lm_head →
    /// argmax. Manages its own Mamba state + per-attn-layer KV pool internally
    /// (single fresh prefill). `conv_bias` in the Mamba variant is optional.
    #[allow(clippy::too_many_arguments)]
    pub fn nemotron_forward_bf16(
        &self, token_ids: &DeviceBuffer, embed: &DeviceBuffer, layers: &[NemotronLayer],
        final_norm: &DeviceBuffer, lm_head: &DeviceBuffer, positions: &DeviceBuffer,
        out_logits: &DeviceBuffer, out_token_ids: &DeviceBuffer, num_tokens: i32,
        dims: &NemotronForwardDims,
    ) -> Result<()> {
        let nul = std::ptr::null::<c_void>();
        let null_mamba = || ffi::PieNemotronMambaWeights {
            in_proj_w: nul, conv_w: nul, conv_bias: nul, a_log: nul, d: nul, dt_bias: nul,
            norm_weight: nul, out_proj_w: nul,
        };
        let null_attn = || ffi::PieNemotronAttnWeights {
            attn_norm: nul, wq: nul, wk: nul, wv: nul, wo: nul,
        };
        let null_ffn = || ffi::PieNemotronFfnWeights { ffn_norm: nul, w_gate: nul, w_up: nul, w_down: nul };
        let mut kinds: Vec<i8> = Vec::with_capacity(layers.len());
        let ffi_layers: Vec<ffi::PieNemotronLayerWeights> = layers.iter().map(|l| match l {
            NemotronLayer::Mamba { pre_norm, w } => {
                kinds.push(b'M' as i8);
                ffi::PieNemotronLayerWeights {
                    kind: b'M' as i8, mamba_pre_norm: pre_norm.ptr,
                    mamba: ffi::PieNemotronMambaWeights {
                        in_proj_w: w.in_proj_w.ptr, conv_w: w.conv_w.ptr,
                        conv_bias: w.conv_bias.map_or(nul, |b| b.ptr as *const c_void),
                        a_log: w.a_log.ptr, d: w.d.ptr, dt_bias: w.dt_bias.ptr,
                        norm_weight: w.norm_weight.ptr, out_proj_w: w.out_proj_w.ptr,
                    },
                    attn: null_attn(), ffn: null_ffn(),
                }
            }
            NemotronLayer::Attn { attn_norm, wq, wk, wv, wo } => {
                kinds.push(b'A' as i8);
                ffi::PieNemotronLayerWeights {
                    kind: b'A' as i8, mamba_pre_norm: nul, mamba: null_mamba(),
                    attn: ffi::PieNemotronAttnWeights {
                        attn_norm: attn_norm.ptr, wq: wq.ptr, wk: wk.ptr, wv: wv.ptr, wo: wo.ptr,
                    },
                    ffn: null_ffn(),
                }
            }
            NemotronLayer::Ffn { ffn_norm, w_gate, w_up, w_down } => {
                kinds.push(b'F' as i8);
                ffi::PieNemotronLayerWeights {
                    kind: b'F' as i8, mamba_pre_norm: nul, mamba: null_mamba(), attn: null_attn(),
                    ffn: ffi::PieNemotronFfnWeights {
                        ffn_norm: ffn_norm.ptr, w_gate: w_gate.ptr, w_up: w_up.ptr, w_down: w_down.ptr,
                    },
                }
            }
        }).collect();
        let bundle = ffi::PieNemotronWeights {
            embed: embed.ptr, layers: ffi_layers.as_ptr(), n_layers: layers.len() as i32,
            final_norm: final_norm.ptr, lm_head: lm_head.ptr,
        };
        check(
            unsafe {
                ffi::pie_cuda_nemotron_forward_bf16(
                    self.0, token_ids.ptr as *const i32, &bundle, positions.ptr as *const i32,
                    out_logits.ptr, out_token_ids.ptr as *mut i32, num_tokens, kinds.as_ptr(),
                    dims.hidden, dims.vocab, dims.mamba_num_heads, dims.mamba_head_dim,
                    dims.mamba_state_size, dims.mamba_n_groups, dims.mamba_conv_kernel,
                    dims.time_step_min, dims.attn_n_q_heads, dims.attn_n_kv_heads, dims.attn_head_dim,
                    dims.page_size, dims.rope_theta, dims.ffn_intermediate, dims.rms_eps)
            },
            "pie_cuda_nemotron_forward_bf16",
        )
    }

    /// Full dense-MoE forward: embed → N×(llama attention + top-K MoE FFN) →
    /// final RMSNorm → lm_head → argmax. Per-layer paged KV pools kv_k/kv_v
    /// [n_layers, num_kv_pages, page_size, n_kv_heads, head_dim] bf16. Writes
    /// out_logits [num_tokens,vocab] bf16 + out_token_ids [num_tokens] i32.
    #[allow(clippy::too_many_arguments)]
    pub fn moe_forward_bf16(
        &self, token_ids: &DeviceBuffer, embed: &DeviceBuffer, layers: &[MoeLayerWeights],
        final_norm: &DeviceBuffer, lm_head: &DeviceBuffer, positions: &DeviceBuffer,
        kv_k: &DeviceBuffer, kv_v: &DeviceBuffer, qo_indptr: &DeviceBuffer,
        kv_page_indices: &DeviceBuffer, kv_page_indptr: &DeviceBuffer,
        kv_last_page_lens: &DeviceBuffer, out_logits: &DeviceBuffer, out_token_ids: &DeviceBuffer,
        num_tokens: i32, num_requests: i32, num_kv_pages: i32, dims: &MoeForwardDims,
    ) -> Result<()> {
        let ffi_layers: Vec<ffi::PieMoeLayerWeights> = layers.iter().map(|w| {
            ffi::PieMoeLayerWeights {
                attn_norm: w.attn_norm.ptr, wq: w.wq.ptr, wk: w.wk.ptr, wv: w.wv.ptr, wo: w.wo.ptr,
                ffn_norm: w.ffn_norm.ptr, router_w: w.router_w.ptr, wgu: w.wgu.ptr,
                wdown: w.wdown.ptr,
            }
        }).collect();
        let bundle = ffi::PieMoeWeights {
            embed: embed.ptr, layers: ffi_layers.as_ptr(), n_layers: layers.len() as i32,
            final_norm: final_norm.ptr, lm_head: lm_head.ptr,
        };
        check(
            unsafe {
                ffi::pie_cuda_moe_forward_bf16(
                    self.0, token_ids.ptr as *const i32, &bundle, positions.ptr as *const i32,
                    kv_k.ptr, kv_v.ptr, qo_indptr.ptr as *const u32,
                    kv_page_indices.ptr as *const u32, kv_page_indptr.ptr as *const u32,
                    kv_last_page_lens.ptr as *const u32, out_logits.ptr,
                    out_token_ids.ptr as *mut i32, num_tokens, num_requests, num_kv_pages,
                    dims.hidden_size, dims.n_q_heads, dims.n_kv_heads, dims.head_dim,
                    dims.intermediate, dims.num_experts, dims.top_k, dims.vocab, dims.page_size,
                    dims.rms_eps, dims.rope_theta)
            },
            "pie_cuda_moe_forward_bf16",
        )
    }

    /// Mamba-2/SSD selective scan (Nemotron-H). `dt_precomputed`/`da_precomputed`/
    /// `slot_ids` are optional (None → inline dt/dA compute / slot 0). ssm_state
    /// is read+written in place. See ffi for the tensor layouts.
    #[allow(clippy::too_many_arguments)]
    pub fn ssm_selective_scan_bf16(
        &self, conv_out: &DeviceBuffer, dt: &DeviceBuffer, a: &DeviceBuffer, d: &DeviceBuffer,
        dt_bias: &DeviceBuffer, dt_precomputed: Option<&DeviceBuffer>,
        da_precomputed: Option<&DeviceBuffer>, ssm_state: &DeviceBuffer,
        slot_ids: Option<&DeviceBuffer>, qo_indptr: &DeviceBuffer, y: &DeviceBuffer,
        dims: &SsmDims,
    ) -> Result<()> {
        let nf = std::ptr::null::<f32>();
        check(
            unsafe {
                ffi::pie_cuda_ssm_selective_scan_bf16(
                    self.0, conv_out.ptr, dt.ptr, a.ptr as *const f32, d.ptr as *const f32,
                    dt_bias.ptr as *const f32, dt_precomputed.map_or(nf, |b| b.ptr as *const f32),
                    da_precomputed.map_or(nf, |b| b.ptr as *const f32), ssm_state.ptr,
                    slot_ids.map_or(std::ptr::null::<i32>(), |b| b.ptr as *const i32),
                    qo_indptr.ptr as *const u32, y.ptr, dims.num_requests, dims.num_heads,
                    dims.head_dim, dims.state_size, dims.n_groups, dims.conv_dim,
                    dims.intermediate, dims.time_step_min)
            },
            "pie_cuda_ssm_selective_scan_bf16",
        )
    }

    /// Internalized int4 (u4b8) fused quant GEMM: out[M,N] = act[M,K] @
    /// dequant(qweight)^T. `qweight` PREPACKED via [`Device::qgemm_w4a16_repack`];
    /// `scales` [num_groups,N] bf16; `workspace` ≥ `qgemm_w4a16_workspace_ints(N,M)`
    /// int32, ZEROED before each call. group_size ∈ {128,-1}. sms=0 auto.
    #[allow(clippy::too_many_arguments)]
    pub fn qgemm_w4a16_bf16(&self, act: &DeviceBuffer, qweight: &DeviceBuffer,
                            scales: &DeviceBuffer, out: &DeviceBuffer, m: i32, n: i32, k: i32,
                            group_size: i32, workspace: &DeviceBuffer, sms: i32) -> Result<()> {
        check(unsafe {
            ffi::pie_cuda_qgemm_w4a16_bf16(self.0, act.ptr, qweight.ptr as *const i32, scales.ptr,
                out.ptr, m, n, k, group_size, workspace.ptr as *mut i32, sms)
        }, "pie_cuda_qgemm_w4a16_bf16")
    }

    /// Prepack GPTQ-packed int4 [K/8,N] int32 → kernel tile layout [K/16,N*16/8] int32.
    pub fn qgemm_w4a16_repack(&self, qweight_rowmajor: &DeviceBuffer, qweight_out: &DeviceBuffer,
                              n: i32, k: i32) -> Result<()> {
        check(unsafe {
            ffi::pie_cuda_qgemm_w4a16_repack(self.0, qweight_rowmajor.ptr as *const i32,
                qweight_out.ptr as *mut i32, n, k)
        }, "pie_cuda_qgemm_w4a16_repack")
    }

    /// Required w4a16 workspace size in int32 elements (zero before each gemm).
    pub fn qgemm_w4a16_workspace_ints(&self, n: i32, max_m: i32) -> i32 {
        unsafe { ffi::pie_cuda_qgemm_w4a16_workspace_ints(n, max_m) }
    }

    /// fp8 (fe4m3fn) fused quant GEMM: out[M,N] = act[M,K] @ dequant(qweight)^T.
    /// `qweight` PREPACKED via [`Device::qgemm_w8a16_fp8_repack`]; `scales`
    /// [num_groups,N] bf16 (logical); `workspace` zeroed before each call.
    #[allow(clippy::too_many_arguments)]
    pub fn qgemm_w8a16_fp8_bf16(&self, act: &DeviceBuffer, qweight: &DeviceBuffer,
                                scales: &DeviceBuffer, out: &DeviceBuffer, m: i32, n: i32, k: i32,
                                group_size: i32, workspace: &DeviceBuffer, sms: i32) -> Result<()> {
        check(unsafe {
            ffi::pie_cuda_qgemm_w8a16_fp8_bf16(self.0, act.ptr, qweight.ptr, scales.ptr, out.ptr,
                m, n, k, group_size, workspace.ptr as *mut i32, sms)
        }, "pie_cuda_qgemm_w8a16_fp8_bf16")
    }

    /// Prepack fe4m3fn fp8 weights [K/4,N] int32 → kernel tile layout
    /// [K/16, N*16/4] int32.
    pub fn qgemm_w8a16_fp8_repack(&self, qweight_rowmajor: &DeviceBuffer, qweight_out: &DeviceBuffer,
                                  n: i32, k: i32) -> Result<()> {
        check(unsafe {
            ffi::pie_cuda_qgemm_w8a16_fp8_repack(self.0, qweight_rowmajor.ptr as *const i32,
                qweight_out.ptr as *mut i32, n, k)
        }, "pie_cuda_qgemm_w8a16_fp8_repack")
    }

    /// Required w8a16-fp8 workspace size in int32 elements (zero before each gemm).
    pub fn qgemm_w8a16_fp8_workspace_ints(&self, n: i32, max_m: i32) -> i32 {
        unsafe { ffi::pie_cuda_qgemm_w8a16_fp8_workspace_ints(n, max_m) }
    }

    /// Sparse token-dispatched MoE block — drop-in for [`Device::moe_mlp_block_bf16`]
    /// (identical semantics + weight layouts) via dispatch → grouped GEMM → combine.
    #[allow(clippy::too_many_arguments)]
    pub fn moe_sparse_block_bf16(&self, hidden: &DeviceBuffer, router_w: &DeviceBuffer,
                                 wgu: &DeviceBuffer, wdown: &DeviceBuffer, out: &DeviceBuffer,
                                 num_tokens: i32, hidden_size: i32, intermediate: i32,
                                 num_experts: i32, top_k: i32) -> Result<()> {
        check(unsafe {
            ffi::pie_cuda_moe_sparse_block_bf16(self.0, hidden.ptr, router_w.ptr, wgu.ptr,
                wdown.ptr, out.ptr, num_tokens, hidden_size, intermediate, num_experts, top_k)
        }, "pie_cuda_moe_sparse_block_bf16")
    }

    /// Nemotron-H Mamba-2 mixer block (prefill, single request): in_proj → split →
    /// causal conv → SSD scan → gated RMSNorm → out_proj, residual in place on
    /// `hidden` [num_tokens, hidden_size] bf16. `conv_bias` optional.
    #[allow(clippy::too_many_arguments)]
    pub fn nemotron_mamba_block_bf16(&self, hidden: &DeviceBuffer, w: &NemotronMambaWeights,
                                     num_tokens: i32, hidden_size: i32, num_heads: i32,
                                     head_dim: i32, state_size: i32, n_groups: i32,
                                     conv_kernel: i32, rms_eps: f32, time_step_min: f32)
                                     -> Result<()> {
        let fw = ffi::PieNemotronMambaWeights {
            in_proj_w: w.in_proj_w.ptr, conv_w: w.conv_w.ptr,
            conv_bias: w.conv_bias.map_or(std::ptr::null(), |b| b.ptr as *const c_void),
            a_log: w.a_log.ptr, d: w.d.ptr, dt_bias: w.dt_bias.ptr,
            norm_weight: w.norm_weight.ptr, out_proj_w: w.out_proj_w.ptr,
        };
        check(unsafe {
            ffi::pie_cuda_nemotron_mamba_block_bf16(self.0, hidden.ptr, &fw, num_tokens,
                hidden_size, num_heads, head_dim, state_size, n_groups, conv_kernel, rms_eps,
                time_step_min)
        }, "pie_cuda_nemotron_mamba_block_bf16")
    }

    /// Allocate a pre-sized activation workspace (un-stubs `pie_ws_alloc`).
    #[allow(clippy::too_many_arguments)]
    pub fn workspace(&self, max_tokens: i32, hidden: i32, n_q_heads: i32, n_kv_heads: i32,
                     head_dim: i32, intermediate: i32, vocab: i32) -> Result<Workspace<'_>> {
        let dims = ffi::PieWorkspaceDims {
            max_tokens, max_requests: max_tokens, hidden_size: hidden,
            intermediate_size: intermediate, num_heads: n_q_heads, num_kv_heads: n_kv_heads,
            head_dim, vocab_size: vocab, num_layers: 0, recurrent_state_slots: 0, moe_experts: 0,
        };
        let mut ptr = std::ptr::null_mut();
        check(unsafe { ffi::pie_ws_alloc(self.0, &dims, &mut ptr) }, "pie_ws_alloc")?;
        Ok(Workspace { ptr, _dev: self })
    }

    /// Full llama-like forward: embed → N layers → final norm → lm_head →
    /// argmax. Writes `out_logits` [num_tokens, vocab] bf16 and `out_token_ids`
    /// [num_tokens] i32 (greedy). `layers` are the per-layer weights, in order.
    #[allow(clippy::too_many_arguments)]
    pub fn llama_forward_bf16(
        &self, ws: &Workspace, token_ids: &DeviceBuffer, embed: &DeviceBuffer,
        layers: &[LlamaLayerWeights], final_norm: &DeviceBuffer, lm_head: &DeviceBuffer,
        positions: &DeviceBuffer, kv_k: &DeviceBuffer, kv_v: &DeviceBuffer,
        qo_indptr: &DeviceBuffer, kv_page_indices: &DeviceBuffer, kv_page_indptr: &DeviceBuffer,
        kv_last_page_lens: &DeviceBuffer, out_logits: &DeviceBuffer, out_token_ids: &DeviceBuffer,
        dims: &LlamaForwardDims,
    ) -> Result<()> {
        let ffi_layers: Vec<ffi::PieLlamaLayerWeights> = layers
            .iter()
            .map(|w| ffi::PieLlamaLayerWeights {
                attn_norm: w.attn_norm.ptr, wq: w.wq.ptr, wk: w.wk.ptr, wv: w.wv.ptr,
                wo: w.wo.ptr, ffn_norm: w.ffn_norm.ptr, w_gate: w.w_gate.ptr,
                w_up: w.w_up.ptr, w_down: w.w_down.ptr,
                q_norm: w.q_norm.map_or(std::ptr::null(), |b| b.ptr as *const c_void),
                k_norm: w.k_norm.map_or(std::ptr::null(), |b| b.ptr as *const c_void),
                q_bias: w.q_bias.map_or(std::ptr::null(), |b| b.ptr as *const c_void),
                k_bias: w.k_bias.map_or(std::ptr::null(), |b| b.ptr as *const c_void),
                v_bias: w.v_bias.map_or(std::ptr::null(), |b| b.ptr as *const c_void),
            })
            .collect();
        let bundle = ffi::PieLlamaWeights {
            embed: embed.ptr, layers: ffi_layers.as_ptr(), n_layers: layers.len() as i32,
            final_norm: final_norm.ptr, lm_head: lm_head.ptr,
        };
        check(
            unsafe {
                ffi::pie_cuda_llama_forward_bf16(
                    self.0, ws.ptr, token_ids.ptr as *const i32, &bundle,
                    positions.ptr as *const i32, kv_k.ptr, kv_v.ptr, qo_indptr.ptr as *const u32,
                    kv_page_indices.ptr as *const u32, kv_page_indptr.ptr as *const u32,
                    kv_last_page_lens.ptr as *const u32, out_logits.ptr,
                    out_token_ids.ptr as *mut i32, dims.num_tokens, dims.num_requests,
                    dims.hidden_size, dims.n_q_heads, dims.n_kv_heads, dims.head_dim,
                    dims.intermediate, dims.page_size, dims.num_kv_pages, dims.vocab,
                    dims.rms_eps, dims.rope_theta)
            },
            "pie_cuda_llama_forward_bf16",
        )
    }
}

/// Borrowed per-layer weight buffers for [`Device::llama_layer_bf16`].
pub struct LlamaLayerWeights<'a> {
    pub attn_norm: &'a DeviceBuffer<'a>,
    pub wq: &'a DeviceBuffer<'a>,
    pub wk: &'a DeviceBuffer<'a>,
    pub wv: &'a DeviceBuffer<'a>,
    pub wo: &'a DeviceBuffer<'a>,
    pub ffn_norm: &'a DeviceBuffer<'a>,
    pub w_gate: &'a DeviceBuffer<'a>,
    pub w_up: &'a DeviceBuffer<'a>,
    pub w_down: &'a DeviceBuffer<'a>,
    /// Per-head q/k RMSNorm gains [head_dim] (Qwen3); `None` = no qk-norm.
    pub q_norm: Option<&'a DeviceBuffer<'a>>,
    pub k_norm: Option<&'a DeviceBuffer<'a>>,
    /// Additive q/k/v projection biases (Qwen2); `None` = no bias.
    pub q_bias: Option<&'a DeviceBuffer<'a>>,
    pub k_bias: Option<&'a DeviceBuffer<'a>>,
    pub v_bias: Option<&'a DeviceBuffer<'a>>,
}

/// Shapes / hyperparameters for [`Device::llama_layer_bf16`].
pub struct LlamaLayerDims {
    pub num_tokens: i32,
    pub num_requests: i32,
    pub hidden_size: i32,
    pub n_q_heads: i32,
    pub n_kv_heads: i32,
    pub head_dim: i32,
    pub intermediate: i32,
    pub page_size: i32,
    pub rms_eps: f32,
    pub rope_theta: f32,
}

/// Borrowed per-layer MLA weight buffers for [`Device::mla_block_bf16`].
/// W_uk/W_uv are pre-transposed per head (absorbed form); see the device header.
pub struct MlaLayerWeights<'a> {
    pub attn_norm: &'a DeviceBuffer<'a>,
    pub w_q_a: &'a DeviceBuffer<'a>,
    pub q_a_ln: &'a DeviceBuffer<'a>,
    pub w_q_b: &'a DeviceBuffer<'a>,
    pub w_kv_a: &'a DeviceBuffer<'a>,
    pub kv_a_ln: &'a DeviceBuffer<'a>,
    pub w_uk: &'a DeviceBuffer<'a>,
    pub w_uv: &'a DeviceBuffer<'a>,
    pub w_o: &'a DeviceBuffer<'a>,
}

/// Shapes / hyperparameters for [`Device::mla_block_bf16`].
pub struct MlaBlockDims {
    pub num_tokens: i32,
    pub num_requests: i32,
    pub hidden_size: i32,
    pub num_heads: i32,
    pub q_lora_rank: i32,
    pub kv_lora_rank: i32,
    pub qk_nope_head_dim: i32,
    pub qk_rope_head_dim: i32,
    pub v_head_dim: i32,
    pub page_size: i32,
    pub rms_eps: f32,
    pub sm_scale: f32,
    pub rope_theta: f32,
}

/// Model dims for [`Device::mla_forward_bf16`] (n_layers derived from the
/// `layers` slice). `num_pages` is pages PER LAYER (sets the cache stride).
pub struct MlaForwardDims {
    pub hidden: i32,
    pub num_heads: i32,
    pub q_lora_rank: i32,
    pub kv_lora_rank: i32,
    pub qk_nope_head_dim: i32,
    pub qk_rope_head_dim: i32,
    pub v_head_dim: i32,
    pub vocab: i32,
    pub page_size: i32,
    pub num_pages: i32,
    pub rms_eps: f32,
    pub sm_scale: f32,
    pub rope_theta: f32,
}

/// Borrowed per-layer DeepSeek weights for [`Device::deepseek_forward_bf16`].
/// `attn` = the MLA attention pointers; the dense FFN set (`w_gate/w_up/w_down`)
/// is `Some` for layers `< first_k_dense`, the MoE set (`router_w/wgu/wdown`)
/// `Some` otherwise — the unused set is `None` (passed as null).
pub struct DeepseekLayerWeights<'a> {
    pub attn: MlaLayerWeights<'a>,
    pub ffn_norm: &'a DeviceBuffer<'a>,
    pub w_gate: Option<&'a DeviceBuffer<'a>>,
    pub w_up: Option<&'a DeviceBuffer<'a>>,
    pub w_down: Option<&'a DeviceBuffer<'a>>,
    pub router_w: Option<&'a DeviceBuffer<'a>>,
    pub wgu: Option<&'a DeviceBuffer<'a>>,
    pub wdown: Option<&'a DeviceBuffer<'a>>,
}

/// Model dims for [`Device::deepseek_forward_bf16`] (n_layers from the slice).
pub struct DeepseekForwardDims {
    pub first_k_dense: i32,
    pub hidden: i32,
    pub num_heads: i32,
    pub q_lora_rank: i32,
    pub kv_lora_rank: i32,
    pub qk_nope_head_dim: i32,
    pub qk_rope_head_dim: i32,
    pub v_head_dim: i32,
    pub dense_inter: i32,
    pub moe_inter: i32,
    pub num_experts: i32,
    pub top_k: i32,
    pub vocab: i32,
    pub page_size: i32,
    pub num_pages: i32,
    pub rms_eps: f32,
    pub sm_scale: f32,
    pub rope_theta: f32,
}

/// Borrowed per-layer Gemma "sandwich" weight buffers for [`Device::gemma_forward_bf16`].
pub struct GemmaLayerWeights<'a> {
    pub input_ln: &'a DeviceBuffer<'a>,
    pub post_attn_ln: &'a DeviceBuffer<'a>,
    pub pre_ffn_ln: &'a DeviceBuffer<'a>,
    pub post_ffn_ln: &'a DeviceBuffer<'a>,
    pub wq: &'a DeviceBuffer<'a>,
    pub wk: &'a DeviceBuffer<'a>,
    pub wv: &'a DeviceBuffer<'a>,
    pub wo: &'a DeviceBuffer<'a>,
    pub w_gate: &'a DeviceBuffer<'a>,
    pub w_up: &'a DeviceBuffer<'a>,
    pub w_down: &'a DeviceBuffer<'a>,
}

/// Model dims for [`Device::gemma_forward_bf16`] (n_layers from the slice).
/// `window_left` is a per-layer left-window array (empty → use `window_left_all`
/// for all layers; <0 = full attention). qk_norm must be 0, altup_num_inputs 1.
pub struct GemmaForwardDims {
    pub hidden: i32,
    pub n_q_heads: i32,
    pub n_kv_heads: i32,
    pub head_dim: i32,
    pub intermediate: i32,
    pub vocab: i32,
    pub page_size: i32,
    pub num_pages: i32,
    pub window_left: Vec<i32>,
    pub window_left_all: i32,
    pub attn_logit_softcap: f32,
    pub final_logit_softcap: f32,
    pub embed_scale: f32,
    pub rms_eps: f32,
    pub rope_theta: f32,
    pub qk_norm: i32,
    pub altup_num_inputs: i32,
}

/// One Nemotron-H hybrid layer for [`Device::nemotron_forward_bf16`] — a tagged
/// union over the three layer kinds (Mamba mixer | GQA attention | dense FFN).
pub enum NemotronLayer<'a> {
    Mamba { pre_norm: &'a DeviceBuffer<'a>, w: NemotronMambaWeights<'a> },
    Attn {
        attn_norm: &'a DeviceBuffer<'a>,
        wq: &'a DeviceBuffer<'a>,
        wk: &'a DeviceBuffer<'a>,
        wv: &'a DeviceBuffer<'a>,
        wo: &'a DeviceBuffer<'a>,
    },
    Ffn {
        ffn_norm: &'a DeviceBuffer<'a>,
        w_gate: &'a DeviceBuffer<'a>,
        w_up: &'a DeviceBuffer<'a>,
        w_down: &'a DeviceBuffer<'a>,
    },
}

/// Model dims for [`Device::nemotron_forward_bf16`] (n_layers + the kind
/// schedule come from the `layers` slice). Mamba/attention/FFN dims are shared
/// across layers of that kind.
pub struct NemotronForwardDims {
    pub hidden: i32,
    pub vocab: i32,
    pub mamba_num_heads: i32,
    pub mamba_head_dim: i32,
    pub mamba_state_size: i32,
    pub mamba_n_groups: i32,
    pub mamba_conv_kernel: i32,
    pub time_step_min: f32,
    pub attn_n_q_heads: i32,
    pub attn_n_kv_heads: i32,
    pub attn_head_dim: i32,
    pub page_size: i32,
    pub rope_theta: f32,
    pub ffn_intermediate: i32,
    pub rms_eps: f32,
}

/// Borrowed per-layer dense-MoE weight buffers for [`Device::moe_forward_bf16`].
pub struct MoeLayerWeights<'a> {
    pub attn_norm: &'a DeviceBuffer<'a>,
    pub wq: &'a DeviceBuffer<'a>,
    pub wk: &'a DeviceBuffer<'a>,
    pub wv: &'a DeviceBuffer<'a>,
    pub wo: &'a DeviceBuffer<'a>,
    pub ffn_norm: &'a DeviceBuffer<'a>,
    pub router_w: &'a DeviceBuffer<'a>,
    pub wgu: &'a DeviceBuffer<'a>,
    pub wdown: &'a DeviceBuffer<'a>,
}

/// Model dims for [`Device::moe_forward_bf16`] (n_layers from the slice).
pub struct MoeForwardDims {
    pub hidden_size: i32,
    pub n_q_heads: i32,
    pub n_kv_heads: i32,
    pub head_dim: i32,
    pub intermediate: i32,
    pub num_experts: i32,
    pub top_k: i32,
    pub vocab: i32,
    pub page_size: i32,
    pub rms_eps: f32,
    pub rope_theta: f32,
}

/// Borrowed Nemotron-H Mamba-2 mixer weights for [`Device::nemotron_mamba_block_bf16`].
/// `conv_bias` is optional. See ffi for layouts.
pub struct NemotronMambaWeights<'a> {
    pub in_proj_w: &'a DeviceBuffer<'a>,
    pub conv_w: &'a DeviceBuffer<'a>,
    pub conv_bias: Option<&'a DeviceBuffer<'a>>,
    pub a_log: &'a DeviceBuffer<'a>,
    pub d: &'a DeviceBuffer<'a>,
    pub dt_bias: &'a DeviceBuffer<'a>,
    pub norm_weight: &'a DeviceBuffer<'a>,
    pub out_proj_w: &'a DeviceBuffer<'a>,
}

/// Shapes for [`Device::ssm_selective_scan_bf16`]. conv_dim = intermediate +
/// 2*n_groups*state_size; intermediate = num_heads*head_dim.
pub struct SsmDims {
    pub num_requests: i32,
    pub num_heads: i32,
    pub head_dim: i32,
    pub state_size: i32,
    pub n_groups: i32,
    pub conv_dim: i32,
    pub intermediate: i32,
    pub time_step_min: f32,
}

/// Shapes / hyperparameters for [`Device::llama_forward_bf16`].
pub struct LlamaForwardDims {
    pub num_tokens: i32,
    pub num_requests: i32,
    pub hidden_size: i32,
    pub n_q_heads: i32,
    pub n_kv_heads: i32,
    pub head_dim: i32,
    pub intermediate: i32,
    pub page_size: i32,
    pub num_kv_pages: i32,
    pub vocab: i32,
    pub rms_eps: f32,
    pub rope_theta: f32,
}

impl Drop for Device {
    fn drop(&mut self) {
        unsafe { ffi::pie_cuda_ctx_destroy(self.0) };
    }
}

// The handles below follow the same pattern. Construction (alloc/bind) is
// driven by `builder.rs`; the hot-path methods (upload/prepare/body/sample)
// are added in phase 3 when `executor.rs` is wired. Stubbed here so the
// module tree and Drop semantics are in place.

pub struct Weights(*mut ffi::PieWeights);
impl Drop for Weights {
    fn drop(&mut self) { unsafe { ffi::pie_weights_destroy(self.0) }; }
}

pub struct KvCache(*mut ffi::PieKvCache);
impl Drop for KvCache {
    fn drop(&mut self) { unsafe { ffi::pie_kv_destroy(self.0) }; }
}

/// Pre-allocated activation scratch (un-stubs `pie_ws_alloc`). Tied to its
/// `Device` so it cannot outlive it; freed on drop.
pub struct Workspace<'a> {
    ptr: *mut ffi::PieWorkspace,
    _dev: &'a Device,
}
impl Drop for Workspace<'_> {
    fn drop(&mut self) { unsafe { ffi::pie_ws_destroy(self.ptr) }; }
}

pub struct GraphExec(*mut ffi::PieGraphExec);
impl Drop for GraphExec {
    fn drop(&mut self) { unsafe { ffi::pie_graph_destroy(self.0) }; }
}

/// A raw device buffer tied to its `Device` (cannot outlive it). Freed on
/// drop. Copies stage through the device's stream — call `Device::sync`
/// before reading a `download`ed host slice.
pub struct DeviceBuffer<'a> {
    dev: &'a Device,
    ptr: *mut c_void,
    len: usize,
}

impl DeviceBuffer<'_> {
    /// Copy a host slice up to the device (H2D). Bytes must fit the buffer.
    pub fn upload<T: Copy>(&self, data: &[T]) -> Result<()> {
        let nbytes = std::mem::size_of_val(data);
        assert!(nbytes <= self.len, "upload {nbytes}B into {}B buffer", self.len);
        check(
            unsafe {
                ffi::pie_cuda_memcpy_h2d(self.dev.raw(), self.ptr, data.as_ptr() as *const c_void, nbytes)
            },
            "pie_cuda_memcpy_h2d",
        )
    }

    /// Device-to-device copy of `nbytes` from byte offset `src_off` to byte
    /// offset `dst_off` within this buffer — used for paged-KV page moves
    /// (Copy D2D: context fork / prefix share). Ranges must stay in bounds;
    /// stream-ordered, so a later forward on the same stream sees the result
    /// without an explicit sync.
    pub fn copy_within_d2d(&self, dst_off: usize, src_off: usize, nbytes: usize) -> Result<()> {
        assert!(
            src_off + nbytes <= self.len && dst_off + nbytes <= self.len,
            "copy_within_d2d out of bounds: src {src_off}+{nbytes}, dst {dst_off}+{nbytes}, len {}",
            self.len
        );
        let dst = unsafe { (self.ptr as *mut u8).add(dst_off) as *mut c_void };
        let src = unsafe { (self.ptr as *const u8).add(src_off) as *const c_void };
        check(
            unsafe { ffi::pie_cuda_memcpy_d2d(self.dev.raw(), dst, src, nbytes) },
            "pie_cuda_memcpy_d2d",
        )
    }

    /// D2H copy of `nbytes` from byte offset `dev_off` of this buffer into the
    /// host slice `dst` (whose len must be ≥ nbytes). For paged-KV swap-out
    /// (Copy D2H). Caller syncs before trusting `dst`.
    pub fn copy_to_host_at(&self, dst: &mut [u8], dev_off: usize, nbytes: usize) -> Result<()> {
        assert!(dev_off + nbytes <= self.len && nbytes <= dst.len(), "copy_to_host_at out of bounds");
        let src = unsafe { (self.ptr as *const u8).add(dev_off) as *const c_void };
        check(
            unsafe { ffi::pie_cuda_memcpy_d2h(self.dev.raw(), dst.as_mut_ptr() as *mut c_void, src, nbytes) },
            "pie_cuda_memcpy_d2h",
        )
    }

    /// H2D copy of `nbytes` from host slice `src` into byte offset `dev_off` of
    /// this buffer. For paged-KV swap-in (Copy H2D).
    pub fn copy_from_host_at(&self, src: &[u8], dev_off: usize, nbytes: usize) -> Result<()> {
        assert!(dev_off + nbytes <= self.len && nbytes <= src.len(), "copy_from_host_at out of bounds");
        let dst = unsafe { (self.ptr as *mut u8).add(dev_off) as *mut c_void };
        check(
            unsafe { ffi::pie_cuda_memcpy_h2d(self.dev.raw(), dst, src.as_ptr() as *const c_void, nbytes) },
            "pie_cuda_memcpy_h2d",
        )
    }

    /// Copy the device buffer down into a host slice (D2H). The caller must
    /// `Device::sync` before trusting the contents.
    pub fn download<T: Copy>(&self, out: &mut [T]) -> Result<()> {
        let nbytes = std::mem::size_of_val(out);
        assert!(nbytes <= self.len, "download {nbytes}B from {}B buffer", self.len);
        check(
            unsafe {
                ffi::pie_cuda_memcpy_d2h(self.dev.raw(), out.as_mut_ptr() as *mut c_void, self.ptr, nbytes)
            },
            "pie_cuda_memcpy_d2h",
        )
    }
}

impl Drop for DeviceBuffer<'_> {
    fn drop(&mut self) {
        unsafe { ffi::pie_cuda_free(self.dev.raw(), self.ptr) };
    }
}

/// Truncating f32 → bf16 (high 16 bits). `bf16_to_f32` round-trips it
/// exactly; output rounding in the kernel is compared with tolerance.
pub fn f32_to_bf16(x: f32) -> u16 {
    (x.to_bits() >> 16) as u16
}

/// bf16 (as u16 bits) → f32, exact.
pub fn bf16_to_f32(b: u16) -> f32 {
    f32::from_bits((b as u32) << 16)
}

/// Round-to-nearest-even f32 → bf16, matching the device's `__float2bfloat16`.
/// The layer reference rounds intermediates through this to mirror the
/// kernels' bf16 materialization, keeping the parity tolerance tight.
pub fn f32_to_bf16_rne(x: f32) -> u16 {
    let bits = x.to_bits();
    if bits & 0x7fff_ffff > 0x7f80_0000 {
        return (bits >> 16) as u16; // NaN: pass the high bits through
    }
    let bias = 0x7fff + ((bits >> 16) & 1);
    ((bits + bias) >> 16) as u16
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Acquire device 0, or return None so the test skips (rather than
    /// fails) on a machine with no usable GPU.
    fn device_or_skip(name: &str) -> Option<Device> {
        match Device::new(0) {
            Ok(d) => Some(d),
            Err(e) => {
                eprintln!("skipping {name} (no usable device): {e:#}");
                None
            }
        }
    }

    // End-to-end through the ABI seam: upload bf16 → run the lifted CUDA
    // kernel → download → compare against a CPU reference over the same
    // bf16 inputs (tolerance covers the kernel's bf16 output rounding).
    #[test]
    fn rmsnorm_bf16_parity() {
        let Some(dev) = device_or_skip("rmsnorm_bf16_parity") else { return };

        let (rows, hidden) = (4usize, 128usize);
        let eps = 1e-6f32;
        let x_f32: Vec<f32> = (0..rows * hidden).map(|i| ((i as f32) * 0.137).sin() * 2.0).collect();
        let w_f32: Vec<f32> = (0..hidden).map(|i| 0.5 + (i % 4) as f32 * 0.1).collect();
        let x_bf: Vec<u16> = x_f32.iter().map(|&v| f32_to_bf16(v)).collect();
        let w_bf: Vec<u16> = w_f32.iter().map(|&v| f32_to_bf16(v)).collect();

        let xb = dev.alloc(x_bf.len() * 2).unwrap();
        let wb = dev.alloc(w_bf.len() * 2).unwrap();
        let yb = dev.alloc(rows * hidden * 2).unwrap();
        xb.upload(&x_bf).unwrap();
        wb.upload(&w_bf).unwrap();

        dev.rmsnorm_bf16(&xb, &wb, &yb, rows as i32, hidden as i32, eps).unwrap();
        dev.sync().unwrap();

        let mut y_bf = vec![0u16; rows * hidden];
        yb.download(&mut y_bf).unwrap();
        dev.sync().unwrap();

        for r in 0..rows {
            let ss: f32 = (0..hidden).map(|h| bf16_to_f32(x_bf[r * hidden + h]).powi(2)).sum();
            let inv_rms = 1.0f32 / (ss / hidden as f32 + eps).sqrt();
            for h in 0..hidden {
                let v = bf16_to_f32(x_bf[r * hidden + h]);
                let wv = bf16_to_f32(w_bf[h]);
                let want = v * inv_rms * wv;
                let got = bf16_to_f32(y_bf[r * hidden + h]);
                let tol = 0.02 * want.abs() + 0.02; // bf16 output rounding
                assert!(
                    (got - want).abs() <= tol,
                    "row {r} col {h}: got {got}, want {want} (tol {tol})"
                );
            }
        }
    }

    #[test]
    fn residual_add_bf16_parity() {
        let Some(dev) = device_or_skip("residual_add_bf16_parity") else { return };
        let n = 257usize; // deliberately not a multiple of the 256 block size
        let y_f32: Vec<f32> = (0..n).map(|i| (i as f32 * 0.05).cos() * 3.0).collect();
        let x_f32: Vec<f32> = (0..n).map(|i| (i as f32 * 0.11).sin() * 1.5).collect();
        let y_bf: Vec<u16> = y_f32.iter().map(|&v| f32_to_bf16(v)).collect();
        let x_bf: Vec<u16> = x_f32.iter().map(|&v| f32_to_bf16(v)).collect();

        let yb = dev.alloc(n * 2).unwrap();
        let xb = dev.alloc(n * 2).unwrap();
        yb.upload(&y_bf).unwrap();
        xb.upload(&x_bf).unwrap();
        dev.residual_add_bf16(&yb, &xb, n).unwrap();
        dev.sync().unwrap();
        let mut out = vec![0u16; n];
        yb.download(&mut out).unwrap();
        dev.sync().unwrap();

        for i in 0..n {
            let want = bf16_to_f32(y_bf[i]) + bf16_to_f32(x_bf[i]);
            let got = bf16_to_f32(out[i]);
            let tol = 0.02 * want.abs() + 0.02;
            assert!((got - want).abs() <= tol, "i {i}: got {got}, want {want} (tol {tol})");
        }
    }

    #[test]
    fn swiglu_bf16_parity() {
        let Some(dev) = device_or_skip("swiglu_bf16_parity") else { return };
        let n = 300usize;
        let gate_f32: Vec<f32> = (0..n).map(|i| (i as f32 * 0.07).sin() * 2.0).collect();
        let up_f32: Vec<f32> = (0..n).map(|i| 0.5 + (i as f32 * 0.03).cos()).collect();
        let gate_bf: Vec<u16> = gate_f32.iter().map(|&v| f32_to_bf16(v)).collect();
        let up_bf: Vec<u16> = up_f32.iter().map(|&v| f32_to_bf16(v)).collect();

        let gb = dev.alloc(n * 2).unwrap();
        let ub = dev.alloc(n * 2).unwrap();
        let yb = dev.alloc(n * 2).unwrap();
        gb.upload(&gate_bf).unwrap();
        ub.upload(&up_bf).unwrap();
        dev.swiglu_bf16(&gb, &ub, &yb, n as i32).unwrap();
        dev.sync().unwrap();
        let mut out = vec![0u16; n];
        yb.download(&mut out).unwrap();
        dev.sync().unwrap();

        for i in 0..n {
            let g = bf16_to_f32(gate_bf[i]);
            let silu = g / (1.0 + (-g).exp());
            let want = silu * bf16_to_f32(up_bf[i]);
            let got = bf16_to_f32(out[i]);
            let tol = 0.02 * want.abs() + 0.02;
            assert!((got - want).abs() <= tol, "i {i}: got {got}, want {want} (tol {tol})");
        }
    }

    #[test]
    fn rope_bf16_parity() {
        let Some(dev) = device_or_skip("rope_bf16_parity") else { return };
        let (num_tokens, nq, nkv, hd) = (3usize, 2usize, 1usize, 8usize);
        let theta = 10000.0f32;
        let q_f32: Vec<f32> = (0..num_tokens * nq * hd).map(|i| (i as f32 * 0.1).sin()).collect();
        let k_f32: Vec<f32> = (0..num_tokens * nkv * hd).map(|i| (i as f32 * 0.13).cos()).collect();
        let positions: Vec<i32> = (0..num_tokens as i32).collect();
        let q_bf: Vec<u16> = q_f32.iter().map(|&v| f32_to_bf16(v)).collect();
        let k_bf: Vec<u16> = k_f32.iter().map(|&v| f32_to_bf16(v)).collect();

        let qb = dev.alloc(q_bf.len() * 2).unwrap();
        let kb = dev.alloc(k_bf.len() * 2).unwrap();
        let pb = dev.alloc(positions.len() * 4).unwrap();
        qb.upload(&q_bf).unwrap();
        kb.upload(&k_bf).unwrap();
        pb.upload(&positions).unwrap();
        dev.rope_bf16(&qb, &kb, &pb, num_tokens as i32, nq as i32, nkv as i32, hd as i32, theta, false).unwrap();
        dev.sync().unwrap();
        let mut q_out = vec![0u16; q_bf.len()];
        let mut k_out = vec![0u16; k_bf.len()];
        qb.download(&mut q_out).unwrap();
        kb.download(&mut k_out).unwrap();
        dev.sync().unwrap();

        // NeoX reference: pair (i, i+half), freq = theta^(-2i/hd), ang = pos*freq.
        let half = hd / 2;
        let check_rope = |inp: &[u16], out: &[u16], n_heads: usize| {
            for n in 0..num_tokens {
                let pos = positions[n] as f32;
                for h in 0..n_heads {
                    let base = (n * n_heads + h) * hd;
                    for i in 0..half {
                        let freq = theta.powf(-2.0 * i as f32 / hd as f32);
                        let (sin_v, cos_v) = (pos * freq).sin_cos();
                        let a = bf16_to_f32(inp[base + i]);
                        let b = bf16_to_f32(inp[base + i + half]);
                        let want0 = a * cos_v - b * sin_v;
                        let want1 = b * cos_v + a * sin_v;
                        let got0 = bf16_to_f32(out[base + i]);
                        let got1 = bf16_to_f32(out[base + i + half]);
                        let tol = 0.03 * want0.abs().max(want1.abs()) + 0.02;
                        assert!(
                            (got0 - want0).abs() <= tol && (got1 - want1).abs() <= tol,
                            "n{n} h{h} i{i}: got ({got0},{got1}) want ({want0},{want1}) tol {tol}"
                        );
                    }
                }
            }
        };
        check_rope(&q_bf, &q_out, nq);
        check_rope(&k_bf, &k_out, nkv);
    }

    #[test]
    fn gemm_bf16_parity() {
        let Some(dev) = device_or_skip("gemm_bf16_parity") else { return };
        let (m, n, k) = (3usize, 4usize, 32usize);
        let act_f32: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.05).sin()).collect();
        let w_f32: Vec<f32> = (0..n * k).map(|i| (i as f32 * 0.07).cos() * 0.5).collect();
        let act_bf: Vec<u16> = act_f32.iter().map(|&v| f32_to_bf16(v)).collect();
        let w_bf: Vec<u16> = w_f32.iter().map(|&v| f32_to_bf16(v)).collect();

        let ab = dev.alloc(act_bf.len() * 2).unwrap();
        let wb = dev.alloc(w_bf.len() * 2).unwrap();
        let yb = dev.alloc(m * n * 2).unwrap();
        ab.upload(&act_bf).unwrap();
        wb.upload(&w_bf).unwrap();
        dev.gemm_bf16(&ab, &wb, &yb, m as i32, n as i32, k as i32, 0.0).unwrap();
        dev.sync().unwrap();
        let mut y_out = vec![0u16; m * n];
        yb.download(&mut y_out).unwrap();
        dev.sync().unwrap();

        // y[mm,nn] = sum_k act[mm,k] * w[nn,k]   (act @ Wᵀ)
        for mm in 0..m {
            for nn in 0..n {
                let want: f32 = (0..k)
                    .map(|kk| bf16_to_f32(act_bf[mm * k + kk]) * bf16_to_f32(w_bf[nn * k + kk]))
                    .sum();
                let got = bf16_to_f32(y_out[mm * n + nn]);
                let tol = 0.05 * want.abs() + 0.15;
                assert!((got - want).abs() <= tol, "y[{mm},{nn}]: got {got}, want {want} (tol {tol})");
            }
        }
    }

    #[test]
    fn embed_bf16_parity() {
        let Some(dev) = device_or_skip("embed_bf16_parity") else { return };
        let (vocab, hidden, num_tokens) = (10usize, 8usize, 4usize);
        let w_f32: Vec<f32> = (0..vocab * hidden).map(|i| i as f32 * 0.25 - 3.0).collect();
        let w_bf: Vec<u16> = w_f32.iter().map(|&v| f32_to_bf16(v)).collect();
        let tokens: Vec<i32> = vec![3, 0, 9, 5];

        let wb = dev.alloc(w_bf.len() * 2).unwrap();
        let tb = dev.alloc(tokens.len() * 4).unwrap();
        let yb = dev.alloc(num_tokens * hidden * 2).unwrap();
        wb.upload(&w_bf).unwrap();
        tb.upload(&tokens).unwrap();
        dev.embed_bf16(&tb, &wb, &yb, num_tokens as i32, hidden as i32, vocab as i32).unwrap();
        dev.sync().unwrap();
        let mut y = vec![0u16; num_tokens * hidden];
        yb.download(&mut y).unwrap();
        dev.sync().unwrap();

        // Pure gather → bit-exact.
        for n in 0..num_tokens {
            let tok = tokens[n] as usize;
            for h in 0..hidden {
                assert_eq!(y[n * hidden + h], w_bf[tok * hidden + h], "n{n} h{h}");
            }
        }
    }

    #[test]
    fn argmax_bf16_parity() {
        let Some(dev) = device_or_skip("argmax_bf16_parity") else { return };
        let (num_rows, vocab) = (5usize, 38usize); // even vocab (vectorized path)
        let mut logits_f32: Vec<f32> = (0..num_rows * vocab).map(|i| (i as f32 * 0.31).sin()).collect();
        // Plant a tie at the row-0 max (idx 4 and 31): lowest index must win.
        logits_f32[4] = 2.0;
        logits_f32[31] = 2.0;
        let logits_bf: Vec<u16> = logits_f32.iter().map(|&v| f32_to_bf16(v)).collect();

        let lb = dev.alloc(logits_bf.len() * 2).unwrap();
        let tb = dev.alloc(num_rows * 4).unwrap();
        lb.upload(&logits_bf).unwrap();
        dev.argmax_bf16(&lb, &tb, num_rows as i32, vocab as i32).unwrap();
        dev.sync().unwrap();
        let mut got = vec![0i32; num_rows];
        tb.download(&mut got).unwrap();
        dev.sync().unwrap();

        for r in 0..num_rows {
            // CPU argmax over the same bf16 values, lowest-index tie-break
            // (update only on strictly greater).
            let mut best = 0usize;
            let mut best_v = bf16_to_f32(logits_bf[r * vocab]);
            for i in 1..vocab {
                let v = bf16_to_f32(logits_bf[r * vocab + i]);
                if v > best_v {
                    best_v = v;
                    best = i;
                }
            }
            assert_eq!(got[r] as usize, best, "row {r}");
        }
        assert_eq!(got[0], 4, "tie must resolve to the lowest index");
    }

    #[test]
    fn attention_naive_paged_bf16_parity() {
        let Some(dev) = device_or_skip("attention_naive_paged_bf16_parity") else { return };
        // 1 request, GQA (2 q-heads → 1 kv-head), causal prefill of 4 tokens.
        let (nq, nkv, hd, page_size, total) = (2usize, 1usize, 64usize, 16usize, 4usize);
        let q_f32: Vec<f32> = (0..total * nq * hd).map(|i| (i as f32 * 0.017).sin() * 0.5).collect();
        let kv_elems = page_size * nkv * hd; // [1 page, page_size, nkv, hd]
        let mut k_f32 = vec![0f32; kv_elems];
        let mut v_f32 = vec![0f32; kv_elems];
        for t in 0..total {
            for d in 0..hd {
                k_f32[t * hd + d] = ((t * hd + d) as f32 * 0.013).cos() * 0.5;
                v_f32[t * hd + d] = ((t * hd + d) as f32 * 0.011).sin() * 0.5;
            }
        }
        let q_bf: Vec<u16> = q_f32.iter().map(|&x| f32_to_bf16(x)).collect();
        let k_bf: Vec<u16> = k_f32.iter().map(|&x| f32_to_bf16(x)).collect();
        let v_bf: Vec<u16> = v_f32.iter().map(|&x| f32_to_bf16(x)).collect();

        let qb = dev.alloc(q_bf.len() * 2).unwrap();
        let kb = dev.alloc(k_bf.len() * 2).unwrap();
        let vb = dev.alloc(v_bf.len() * 2).unwrap();
        let ob = dev.alloc(total * nq * hd * 2).unwrap();
        let qo_indptr: Vec<u32> = vec![0, total as u32];
        let kv_page_indices: Vec<u32> = vec![0];
        let kv_page_indptr: Vec<u32> = vec![0, 1];
        let kv_last: Vec<u32> = vec![total as u32];
        let qib = dev.alloc(qo_indptr.len() * 4).unwrap();
        let kpi = dev.alloc(kv_page_indices.len() * 4).unwrap();
        let kpp = dev.alloc(kv_page_indptr.len() * 4).unwrap();
        let klp = dev.alloc(kv_last.len() * 4).unwrap();
        qb.upload(&q_bf).unwrap();
        kb.upload(&k_bf).unwrap();
        vb.upload(&v_bf).unwrap();
        qib.upload(&qo_indptr).unwrap();
        kpi.upload(&kv_page_indices).unwrap();
        kpp.upload(&kv_page_indptr).unwrap();
        klp.upload(&kv_last).unwrap();

        dev.attention_naive_paged_bf16(
            &qb, &kb, &vb, &ob, &qib, &kpi, &kpp, &klp,
            total as i32, 1, nq as i32, nkv as i32, hd as i32, page_size as i32, -1, -1.0,
        ).unwrap();
        dev.sync().unwrap();
        let mut o = vec![0u16; total * nq * hd];
        ob.download(&mut o).unwrap();
        dev.sync().unwrap();

        // CPU causal reference over the same bf16 inputs: query i attends kv 0..=i.
        let scale = 1.0f32 / (hd as f32).sqrt();
        for i in 0..total {
            for h in 0..nq {
                let mut scores: Vec<f32> = (0..=i)
                    .map(|j| {
                        let dot: f32 = (0..hd)
                            .map(|d| bf16_to_f32(q_bf[(i * nq + h) * hd + d]) * bf16_to_f32(k_bf[j * hd + d]))
                            .sum();
                        dot * scale
                    })
                    .collect();
                let m = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let mut denom = 0f32;
                for s in scores.iter_mut() {
                    *s = (*s - m).exp();
                    denom += *s;
                }
                for d in 0..hd {
                    let acc: f32 = (0..=i).map(|j| scores[j] / denom * bf16_to_f32(v_bf[j * hd + d])).sum();
                    let got = bf16_to_f32(o[(i * nq + h) * hd + d]);
                    let tol = 0.03 * acc.abs() + 0.03;
                    assert!((got - acc).abs() <= tol, "i{i} h{h} d{d}: got {got}, want {acc} (tol {tol})");
                }
            }
        }
    }

    // Upload a host slice into a fresh device buffer (test helper).
    fn up<'a, T: Copy>(dev: &'a Device, data: &[T]) -> DeviceBuffer<'a> {
        let b = dev.alloc(std::mem::size_of_val(data)).unwrap();
        b.upload(data).unwrap();
        b
    }

    // Composed llama_like layer vs. an independent CPU f32 reference. The
    // reference rounds each materialized buffer through bf16-RNE to mirror
    // the kernels, so the tolerance reflects only fp32 reduction order +
    // transcendental-approx drift, not accumulated bf16 loss.
    #[test]
    fn llama_layer_bf16_parity() {
        let Some(dev) = device_or_skip("llama_layer_bf16_parity") else { return };
        let (t, h, nq, nkv, hd, inter, page_size) = (4usize, 64, 2, 1, 32, 128, 16);
        let (hq, hkv) = (nq * hd, nkv * hd);
        let (eps, theta) = (1e-5f32, 10000.0f32);

        let seq = |seed: f32, n: usize, scale: f32| -> Vec<f32> {
            (0..n).map(|i| ((i as f32 + seed) * 0.1).sin() * scale).collect()
        };
        let hidden0 = seq(1.0, t * h, 1.0);
        let attn_norm: Vec<f32> = seq(2.0, h, 0.2).iter().map(|v| 1.0 + v).collect();
        let ffn_norm: Vec<f32> = seq(3.0, h, 0.2).iter().map(|v| 1.0 + v).collect();
        let wq = seq(4.0, hq * h, 0.1);
        let wk = seq(5.0, hkv * h, 0.1);
        let wv = seq(6.0, hkv * h, 0.1);
        let wo = seq(7.0, h * hq, 0.1);
        let w_gate = seq(8.0, inter * h, 0.1);
        let w_up = seq(9.0, inter * h, 0.1);
        let w_down = seq(10.0, h * inter, 0.1);
        let positions: Vec<i32> = (0..t as i32).collect();

        // bf16 round-trip (truncation, as inputs are uploaded) so device and
        // reference operate on identical values.
        let rt = |v: &[f32]| -> Vec<f32> { v.iter().map(|&x| bf16_to_f32(f32_to_bf16(x))).collect() };
        let bf = |v: &[f32]| -> Vec<u16> { v.iter().map(|&x| f32_to_bf16(x)).collect() };

        // --- device ---
        let hb = up(&dev, &bf(&hidden0));
        let (anb, fnb) = (up(&dev, &bf(&attn_norm)), up(&dev, &bf(&ffn_norm)));
        let (wqb, wkb, wvb) = (up(&dev, &bf(&wq)), up(&dev, &bf(&wk)), up(&dev, &bf(&wv)));
        let wob = up(&dev, &bf(&wo));
        let (wgb, wub, wdb) = (up(&dev, &bf(&w_gate)), up(&dev, &bf(&w_up)), up(&dev, &bf(&w_down)));
        let pb = up(&dev, &positions);
        let kpages = dev.alloc(page_size * nkv * hd * 2).unwrap();
        let vpages = dev.alloc(page_size * nkv * hd * 2).unwrap();
        let qib = up(&dev, &[0u32, t as u32]);
        let kpi = up(&dev, &[0u32]);
        let kpp = up(&dev, &[0u32, 1u32]);
        let klp = up(&dev, &[t as u32]);

        let weights = LlamaLayerWeights {
            attn_norm: &anb, wq: &wqb, wk: &wkb, wv: &wvb, wo: &wob,
            ffn_norm: &fnb, w_gate: &wgb, w_up: &wub, w_down: &wdb, q_norm: None, k_norm: None, q_bias: None, k_bias: None, v_bias: None,
        };
        let dims = LlamaLayerDims {
            num_tokens: t as i32, num_requests: 1, hidden_size: h as i32, n_q_heads: nq as i32,
            n_kv_heads: nkv as i32, head_dim: hd as i32, intermediate: inter as i32,
            page_size: page_size as i32, rms_eps: eps, rope_theta: theta,
        };
        dev.llama_layer_bf16(&hb, &weights, &pb, &kpages, &vpages, &qib, &kpi, &kpp, &klp, &dims)
            .unwrap();
        dev.sync().unwrap();
        let mut out = vec![0u16; t * h];
        hb.download(&mut out).unwrap();
        dev.sync().unwrap();

        // --- CPU reference (f32, bf16-RNE materialization) ---
        let r = |x: f32| bf16_to_f32(f32_to_bf16_rne(x));
        let gemm = |act: &[f32], w: &[f32], m: usize, n: usize, k: usize| -> Vec<f32> {
            (0..m * n)
                .map(|idx| {
                    let (mm, nn) = (idx / n, idx % n);
                    r((0..k).map(|kk| act[mm * k + kk] * w[nn * k + kk]).sum())
                })
                .collect()
        };
        let rmsnorm = |x: &[f32], wt: &[f32], rows: usize, cols: usize| -> Vec<f32> {
            let mut y = vec![0f32; rows * cols];
            for row in 0..rows {
                let ss: f32 = (0..cols).map(|c| x[row * cols + c].powi(2)).sum();
                let inv = 1.0 / (ss / cols as f32 + eps).sqrt();
                for c in 0..cols {
                    y[row * cols + c] = r(x[row * cols + c] * inv * wt[c]);
                }
            }
            y
        };
        let rope_buf = |buf: &mut [f32], n_heads: usize| {
            let half = hd / 2;
            for n in 0..t {
                let pos = positions[n] as f32;
                for hh in 0..n_heads {
                    let base = (n * n_heads + hh) * hd;
                    for i in 0..half {
                        let (s, c) = (pos * theta.powf(-2.0 * i as f32 / hd as f32)).sin_cos();
                        let (a, b) = (buf[base + i], buf[base + i + half]);
                        buf[base + i] = r(a * c - b * s);
                        buf[base + i + half] = r(b * c + a * s);
                    }
                }
            }
        };
        let resid = |a: &[f32], b: &[f32]| -> Vec<f32> {
            a.iter().zip(b).map(|(&x, &y)| r(x + y)).collect()
        };

        let an = rt(&attn_norm);
        let fnn = rt(&ffn_norm);
        let (wqd, wkd, wvd, wod) = (rt(&wq), rt(&wk), rt(&wv), rt(&wo));
        let (wgd, wud, wdd) = (rt(&w_gate), rt(&w_up), rt(&w_down));

        let mut hid = rt(&hidden0);
        let normed = rmsnorm(&hid, &an, t, h);
        let mut q = gemm(&normed, &wqd, t, hq, h);
        let mut k = gemm(&normed, &wkd, t, hkv, h);
        let v = gemm(&normed, &wvd, t, hkv, h);
        rope_buf(&mut q, nq);
        rope_buf(&mut k, nkv);
        // attention (causal, GQA) over the rope'd q and k/v.
        let scale = 1.0 / (hd as f32).sqrt();
        let group = nq / nkv;
        let mut attn_o = vec![0f32; t * hq];
        for i in 0..t {
            for hh in 0..nq {
                let kvh = hh / group;
                let mut sc: Vec<f32> = (0..=i)
                    .map(|j| {
                        let dot: f32 = (0..hd)
                            .map(|d| q[(i * nq + hh) * hd + d] * k[(j * nkv + kvh) * hd + d])
                            .sum();
                        dot * scale
                    })
                    .collect();
                let m = sc.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let mut den = 0f32;
                for s in sc.iter_mut() {
                    *s = (*s - m).exp();
                    den += *s;
                }
                for d in 0..hd {
                    let acc: f32 = (0..=i).map(|j| sc[j] / den * v[(j * nkv + kvh) * hd + d]).sum();
                    attn_o[(i * nq + hh) * hd + d] = r(acc);
                }
            }
        }
        let o = gemm(&attn_o, &wod, t, h, hq);
        hid = resid(&hid, &o);
        let normed2 = rmsnorm(&hid, &fnn, t, h);
        let gate = gemm(&normed2, &wgd, t, inter, h);
        let upp = gemm(&normed2, &wud, t, inter, h);
        let mlp: Vec<f32> = gate.iter().zip(&upp).map(|(&g, &u)| r(g / (1.0 + (-g).exp()) * u)).collect();
        let mlp_out = gemm(&mlp, &wdd, t, h, inter);
        let want = resid(&hid, &mlp_out);

        for i in 0..t * h {
            let got = bf16_to_f32(out[i]);
            let tol = 0.05 * want[i].abs() + 0.1;
            assert!((got - want[i]).abs() <= tol, "elem {i}: got {got}, want {} (tol {tol})", want[i]);
        }
    }

    // Full forward: embed → N layers → final norm → lm_head → argmax, vs an
    // independent CPU f32 reference (bf16-RNE materialization). Checks logits
    // parity and that the greedy token is a (near-)argmax of the ref logits.
    #[test]
    fn llama_forward_bf16_parity() {
        let Some(dev) = device_or_skip("llama_forward_bf16_parity") else { return };
        let (n_layers, t, h, nq, nkv, hd, inter, vocab, page_size) =
            (2usize, 3, 32, 2, 1, 16, 64, 64, 8);
        let (hq, hkv) = (nq * hd, nkv * hd);
        let (eps, theta) = (1e-5f32, 10000.0f32);
        let seq = |seed: f32, n: usize, scale: f32| -> Vec<f32> {
            (0..n).map(|i| ((i as f32 + seed) * 0.1).sin() * scale).collect()
        };
        let one_plus = |v: Vec<f32>| -> Vec<f32> { v.iter().map(|x| 1.0 + x).collect() };
        let bf = |v: &[f32]| -> Vec<u16> { v.iter().map(|&x| f32_to_bf16(x)).collect() };
        let rt = |v: &[f32]| -> Vec<f32> { v.iter().map(|&x| bf16_to_f32(f32_to_bf16(x))).collect() };

        struct Lw {
            an: Vec<f32>, fnn: Vec<f32>, wq: Vec<f32>, wk: Vec<f32>, wv: Vec<f32>,
            wo: Vec<f32>, wg: Vec<f32>, wu: Vec<f32>, wd: Vec<f32>,
        }
        let embed_w = seq(100.0, vocab * h, 0.5);
        let final_norm = one_plus(seq(101.0, h, 0.2));
        let lm_head = seq(102.0, vocab * h, 0.1);
        let lws: Vec<Lw> = (0..n_layers)
            .map(|l| {
                let s = l as f32 * 17.0;
                Lw {
                    an: one_plus(seq(s + 1.0, h, 0.2)), fnn: one_plus(seq(s + 2.0, h, 0.2)),
                    wq: seq(s + 3.0, hq * h, 0.1), wk: seq(s + 4.0, hkv * h, 0.1),
                    wv: seq(s + 5.0, hkv * h, 0.1), wo: seq(s + 6.0, h * hq, 0.1),
                    wg: seq(s + 7.0, inter * h, 0.1), wu: seq(s + 8.0, inter * h, 0.1),
                    wd: seq(s + 9.0, h * inter, 0.1),
                }
            })
            .collect();
        let token_ids: Vec<i32> = vec![5, 17, 2];
        let positions: Vec<i32> = (0..t as i32).collect();

        // --- device ---
        let ws = dev
            .workspace(t as i32, h as i32, nq as i32, nkv as i32, hd as i32, inter as i32, vocab as i32)
            .unwrap();
        let tib = up(&dev, &token_ids);
        let pb = up(&dev, &positions);
        let embb = up(&dev, &bf(&embed_w));
        let fnb = up(&dev, &bf(&final_norm));
        let lmb = up(&dev, &bf(&lm_head));
        let anb: Vec<_> = lws.iter().map(|w| up(&dev, &bf(&w.an))).collect();
        let fnnb: Vec<_> = lws.iter().map(|w| up(&dev, &bf(&w.fnn))).collect();
        let wqb: Vec<_> = lws.iter().map(|w| up(&dev, &bf(&w.wq))).collect();
        let wkb: Vec<_> = lws.iter().map(|w| up(&dev, &bf(&w.wk))).collect();
        let wvb: Vec<_> = lws.iter().map(|w| up(&dev, &bf(&w.wv))).collect();
        let wob: Vec<_> = lws.iter().map(|w| up(&dev, &bf(&w.wo))).collect();
        let wgb: Vec<_> = lws.iter().map(|w| up(&dev, &bf(&w.wg))).collect();
        let wub: Vec<_> = lws.iter().map(|w| up(&dev, &bf(&w.wu))).collect();
        let wdb: Vec<_> = lws.iter().map(|w| up(&dev, &bf(&w.wd))).collect();
        let layers: Vec<LlamaLayerWeights> = (0..n_layers)
            .map(|l| LlamaLayerWeights {
                attn_norm: &anb[l], wq: &wqb[l], wk: &wkb[l], wv: &wvb[l], wo: &wob[l],
                ffn_norm: &fnnb[l], w_gate: &wgb[l], w_up: &wub[l], w_down: &wdb[l],
                q_norm: None, k_norm: None, q_bias: None, k_bias: None, v_bias: None,
            })
            .collect();
        let kv_k = dev.alloc(n_layers * page_size * hkv * 2).unwrap();
        let kv_v = dev.alloc(n_layers * page_size * hkv * 2).unwrap();
        let qib = up(&dev, &[0u32, t as u32]);
        let kpi = up(&dev, &[0u32]);
        let kpp = up(&dev, &[0u32, 1u32]);
        let klp = up(&dev, &[t as u32]);
        let out_logits = dev.alloc(t * vocab * 2).unwrap();
        let out_tokens = dev.alloc(t * 4).unwrap();
        let dims = LlamaForwardDims {
            num_tokens: t as i32, num_requests: 1, hidden_size: h as i32, n_q_heads: nq as i32,
            n_kv_heads: nkv as i32, head_dim: hd as i32, intermediate: inter as i32,
            page_size: page_size as i32, num_kv_pages: 1, vocab: vocab as i32, rms_eps: eps,
            rope_theta: theta,
        };
        dev.llama_forward_bf16(&ws, &tib, &embb, &layers, &fnb, &lmb, &pb, &kv_k, &kv_v, &qib,
                               &kpi, &kpp, &klp, &out_logits, &out_tokens, &dims)
            .unwrap();
        dev.sync().unwrap();
        let mut dlogits = vec![0u16; t * vocab];
        let mut dtokens = vec![0i32; t];
        out_logits.download(&mut dlogits).unwrap();
        out_tokens.download(&mut dtokens).unwrap();
        dev.sync().unwrap();

        // --- CPU reference ---
        let r = |x: f32| bf16_to_f32(f32_to_bf16_rne(x));
        let gemm = |act: &[f32], w: &[f32], m: usize, n: usize, k: usize| -> Vec<f32> {
            (0..m * n)
                .map(|idx| r((0..k).map(|kk| act[(idx / n) * k + kk] * w[(idx % n) * k + kk]).sum()))
                .collect()
        };
        let rmsnorm = |x: &[f32], wt: &[f32], rows: usize, cols: usize| -> Vec<f32> {
            let mut y = vec![0f32; rows * cols];
            for row in 0..rows {
                let ss: f32 = (0..cols).map(|c| x[row * cols + c].powi(2)).sum();
                let inv = 1.0 / (ss / cols as f32 + eps).sqrt();
                for c in 0..cols {
                    y[row * cols + c] = r(x[row * cols + c] * inv * wt[c]);
                }
            }
            y
        };
        let rope_buf = |buf: &mut [f32], n_heads: usize| {
            let half = hd / 2;
            for n in 0..t {
                let pos = positions[n] as f32;
                for hh in 0..n_heads {
                    let base = (n * n_heads + hh) * hd;
                    for i in 0..half {
                        let (s, c) = (pos * theta.powf(-2.0 * i as f32 / hd as f32)).sin_cos();
                        let (a, b) = (buf[base + i], buf[base + i + half]);
                        buf[base + i] = r(a * c - b * s);
                        buf[base + i + half] = r(b * c + a * s);
                    }
                }
            }
        };
        let resid = |a: &[f32], b: &[f32]| -> Vec<f32> { a.iter().zip(b).map(|(&x, &y)| r(x + y)).collect() };
        let attn = |q: &[f32], k: &[f32], v: &[f32]| -> Vec<f32> {
            let scale = 1.0 / (hd as f32).sqrt();
            let group = nq / nkv;
            let mut o = vec![0f32; t * hq];
            for i in 0..t {
                for hh in 0..nq {
                    let kvh = hh / group;
                    let mut sc: Vec<f32> = (0..=i)
                        .map(|j| {
                            let dot: f32 = (0..hd)
                                .map(|d| q[(i * nq + hh) * hd + d] * k[(j * nkv + kvh) * hd + d])
                                .sum();
                            dot * scale
                        })
                        .collect();
                    let m = sc.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                    let mut den = 0f32;
                    for x in sc.iter_mut() {
                        *x = (*x - m).exp();
                        den += *x;
                    }
                    for d in 0..hd {
                        o[(i * nq + hh) * hd + d] =
                            r((0..=i).map(|j| sc[j] / den * v[(j * nkv + kvh) * hd + d]).sum());
                    }
                }
            }
            o
        };
        let layer_ref = |hid: &[f32], w: &Lw| -> Vec<f32> {
            let normed = rmsnorm(hid, &rt(&w.an), t, h);
            let mut q = gemm(&normed, &rt(&w.wq), t, hq, h);
            let mut k = gemm(&normed, &rt(&w.wk), t, hkv, h);
            let v = gemm(&normed, &rt(&w.wv), t, hkv, h);
            rope_buf(&mut q, nq);
            rope_buf(&mut k, nkv);
            let ao = attn(&q, &k, &v);
            let o = gemm(&ao, &rt(&w.wo), t, h, hq);
            let hid2 = resid(hid, &o);
            let normed2 = rmsnorm(&hid2, &rt(&w.fnn), t, h);
            let gate = gemm(&normed2, &rt(&w.wg), t, inter, h);
            let upp = gemm(&normed2, &rt(&w.wu), t, inter, h);
            let mlp: Vec<f32> = gate.iter().zip(&upp).map(|(&g, &u)| r(g / (1.0 + (-g).exp()) * u)).collect();
            let mo = gemm(&mlp, &rt(&w.wd), t, h, inter);
            resid(&hid2, &mo)
        };

        let embed_d = rt(&embed_w);
        let mut hid = vec![0f32; t * h];
        for n in 0..t {
            let tok = token_ids[n] as usize;
            hid[n * h..n * h + h].copy_from_slice(&embed_d[tok * h..tok * h + h]);
        }
        for w in &lws {
            hid = layer_ref(&hid, w);
        }
        let normed = rmsnorm(&hid, &rt(&final_norm), t, h);
        let logits_ref = gemm(&normed, &rt(&lm_head), t, vocab, h);

        // logits parity
        for i in 0..t * vocab {
            let got = bf16_to_f32(dlogits[i]);
            let tol = 0.06 * logits_ref[i].abs() + 0.15;
            assert!((got - logits_ref[i]).abs() <= tol, "logit {i}: got {got}, want {} (tol {tol})", logits_ref[i]);
        }
        // the device's greedy token must be a near-argmax of the ref logits
        for n in 0..t {
            let dtok = dtokens[n] as usize;
            let refmax = (0..vocab).map(|j| logits_ref[n * vocab + j]).fold(f32::NEG_INFINITY, f32::max);
            let chosen = logits_ref[n * vocab + dtok];
            assert!(chosen >= refmax - (0.06 * refmax.abs() + 0.2),
                    "row {n}: device token {dtok} (ref logit {chosen}) far below ref max {refmax}");
        }
    }

    // General paged KV: a forward whose KV spans multiple pages must produce
    // the same logits as the same forward in a single big page — paging is a
    // storage layout, not a math change. This exercises the multi-page scatter
    // + per-layer stride + attention-across-pages that single-page tests miss.
    #[test]
    fn llama_forward_multipage_equiv() {
        let Some(dev) = device_or_skip("llama_forward_multipage_equiv") else { return };
        let (n_layers, t, h, nq, nkv, hd, inter, vocab) = (2usize, 10, 32, 2, 1, 16, 64, 64);
        let (hq, hkv) = (nq * hd, nkv * hd);
        let (eps, theta) = (1e-5f32, 10000.0f32);
        let seq = |seed: f32, n: usize, scale: f32| -> Vec<u16> {
            (0..n).map(|i| f32_to_bf16(((i as f32 + seed) * 0.1).sin() * scale)).collect()
        };
        let one = |seed: f32, n: usize| -> Vec<u16> {
            (0..n).map(|i| f32_to_bf16(1.0 + ((i as f32 + seed) * 0.1).sin() * 0.2)).collect()
        };

        let embb = up(&dev, &seq(1.0, vocab * h, 0.5));
        let fnb = up(&dev, &one(2.0, h));
        let lmb = up(&dev, &seq(3.0, vocab * h, 0.1));
        let mut keep: Vec<DeviceBuffer> = Vec::new(); // own the per-layer weight buffers
        let mut layers = Vec::new();
        for l in 0..n_layers {
            let s = (l * 100) as f32;
            let bufs = [
                up(&dev, &one(s + 4.0, h)), up(&dev, &seq(s + 6.0, hq * h, 0.1)),
                up(&dev, &seq(s + 7.0, hkv * h, 0.1)), up(&dev, &seq(s + 8.0, hkv * h, 0.1)),
                up(&dev, &seq(s + 9.0, h * hq, 0.1)), up(&dev, &one(s + 5.0, h)),
                up(&dev, &seq(s + 10.0, inter * h, 0.1)), up(&dev, &seq(s + 11.0, inter * h, 0.1)),
                up(&dev, &seq(s + 12.0, h * inter, 0.1)),
            ];
            let base = keep.len();
            keep.extend(bufs);
            layers.push(base);
        }
        let lw: Vec<LlamaLayerWeights> = layers.iter().map(|&b| LlamaLayerWeights {
            attn_norm: &keep[b], wq: &keep[b + 1], wk: &keep[b + 2], wv: &keep[b + 3],
            wo: &keep[b + 4], ffn_norm: &keep[b + 5], w_gate: &keep[b + 6], w_up: &keep[b + 7],
            w_down: &keep[b + 8], q_norm: None, k_norm: None, q_bias: None, k_bias: None, v_bias: None,
        }).collect();

        let tib = up(&dev, &(0..t as i32).map(|i| (i * 7 % vocab as i32)).collect::<Vec<_>>());
        let pb = up(&dev, &(0..t as i32).collect::<Vec<_>>());
        let ws = dev.workspace(t as i32, h as i32, nq as i32, nkv as i32, hd as i32, inter as i32, vocab as i32).unwrap();

        // Run the forward with a given (page_size, num_pages) layout → logits.
        let run = |page_size: usize, num_pages: usize, page_indices: &[u32], last_len: u32| -> Vec<u16> {
            let kv_k = dev.alloc(n_layers * num_pages * page_size * hkv * 2).unwrap();
            let kv_v = dev.alloc(n_layers * num_pages * page_size * hkv * 2).unwrap();
            let qib = up(&dev, &[0u32, t as u32]);
            let kpi = up(&dev, page_indices);
            let kpp = up(&dev, &[0u32, page_indices.len() as u32]);
            let klp = up(&dev, &[last_len]);
            let out_logits = dev.alloc(t * vocab * 2).unwrap();
            let out_tokens = dev.alloc(t * 4).unwrap();
            let dims = LlamaForwardDims {
                num_tokens: t as i32, num_requests: 1, hidden_size: h as i32, n_q_heads: nq as i32,
                n_kv_heads: nkv as i32, head_dim: hd as i32, intermediate: inter as i32,
                page_size: page_size as i32, num_kv_pages: num_pages as i32, vocab: vocab as i32,
                rms_eps: eps, rope_theta: theta,
            };
            dev.llama_forward_bf16(&ws, &tib, &embb, &lw, &fnb, &lmb, &pb, &kv_k, &kv_v, &qib,
                                   &kpi, &kpp, &klp, &out_logits, &out_tokens, &dims).unwrap();
            dev.sync().unwrap();
            let mut log = vec![0u16; t * vocab];
            out_logits.download(&mut log).unwrap();
            dev.sync().unwrap();
            log
        };

        // single page big enough for all t tokens
        let single = run(16, 1, &[0], t as u32);
        // 3 pages of 4 (t=10 → pages 0,1 full, page 2 has 2): exercises spanning
        let multi = run(4, 3, &[0, 1, 2], (t - 2 * 4) as u32);

        // identical kernels, identical math, only the storage layout differs.
        for i in 0..t * vocab {
            let (a, b) = (bf16_to_f32(single[i]), bf16_to_f32(multi[i]));
            assert!((a - b).abs() <= 1e-2 * a.abs().max(1.0) + 1e-2,
                    "logit {i}: single {a} vs multi-page {b}");
        }
    }

    #[test]
    fn write_kv_to_pages_bf16_parity() {
        let Some(dev) = device_or_skip("write_kv_to_pages_bf16_parity") else { return };
        let (nkv, hd, page_size, t) = (2usize, 8usize, 16usize, 4usize);
        let k_curr: Vec<u16> = (0..t * nkv * hd).map(|i| f32_to_bf16((i as f32 * 0.05).sin())).collect();
        let v_curr: Vec<u16> = (0..t * nkv * hd).map(|i| f32_to_bf16((i as f32 * 0.07).cos())).collect();
        let kb = up(&dev, &k_curr);
        let vb = up(&dev, &v_curr);
        let kpages = dev.alloc(page_size * nkv * hd * 2).unwrap();
        let vpages = dev.alloc(page_size * nkv * hd * 2).unwrap();
        let qib = up(&dev, &[0u32, t as u32]);
        let kpi = up(&dev, &[0u32]);
        let kpp = up(&dev, &[0u32, 1]);
        let klp = up(&dev, &[t as u32]);
        dev.write_kv_to_pages_bf16(&kpages, &vpages, &kb, &vb, &qib, &kpi, &kpp, &klp,
            t as i32, 1, page_size as i32, nkv as i32, hd as i32, false).unwrap();
        dev.sync().unwrap();
        let mut kp = vec![0u16; page_size * nkv * hd];
        kpages.download(&mut kp).unwrap();
        dev.sync().unwrap();
        // token t → page 0, slot t (NHD): kp[(t*nkv + h)*hd + d] == k_curr[(t*nkv + h)*hd + d]
        for t_ in 0..t {
            for i in 0..nkv * hd {
                assert_eq!(kp[t_ * nkv * hd + i], k_curr[t_ * nkv * hd + i], "k slot {t_} elem {i}");
            }
        }
    }

    #[test]
    fn sample_temp_bf16_parity() {
        let Some(dev) = device_or_skip("sample_temp_bf16_parity") else { return };
        let (num_rows, vocab) = (3usize, 32usize);
        let logits_f32: Vec<f32> = (0..num_rows * vocab).map(|i| (i as f32 * 0.21).sin()).collect();
        let logits_bf: Vec<u16> = logits_f32.iter().map(|&v| f32_to_bf16(v)).collect();
        let lb = up(&dev, &logits_bf);
        // row 0 greedy (temp 0); rows 1,2 sampled — used for the determinism check.
        let tb = up(&dev, &[0.0f32, 0.8, 1.5]);
        let mb = up(&dev, &[0.0f32, 0.0, 0.0]);
        let sb = up(&dev, &[11u32, 22, 33]);
        let ob = dev.alloc(num_rows * 4).unwrap();
        dev.sample_temp_bf16(&lb, &tb, None, None, Some(&mb), &sb, &ob, num_rows as i32, vocab as i32).unwrap();
        dev.sync().unwrap();
        let mut out1 = vec![0i32; num_rows];
        ob.download(&mut out1).unwrap();
        dev.sync().unwrap();
        // row 0 (temp 0) == plain argmax over bf16 logits (lowest-index tie-break)
        let mut best = 0usize;
        let mut bv = bf16_to_f32(logits_bf[0]);
        for i in 1..vocab {
            let v = bf16_to_f32(logits_bf[i]);
            if v > bv { bv = v; best = i; }
        }
        assert_eq!(out1[0] as usize, best, "temp=0 row must be argmax");
        // determinism: same seeds → identical output
        let ob2 = dev.alloc(num_rows * 4).unwrap();
        dev.sample_temp_bf16(&lb, &tb, None, None, Some(&mb), &sb, &ob2, num_rows as i32, vocab as i32).unwrap();
        dev.sync().unwrap();
        let mut out2 = vec![0i32; num_rows];
        ob2.download(&mut out2).unwrap();
        dev.sync().unwrap();
        assert_eq!(out1, out2, "sampler must be deterministic for fixed seeds");
        for &tk in &out1 {
            assert!((0..vocab as i32).contains(&tk));
        }

        // ── top-k / top-p truncation correctness (no PRNG replication needed):
        // a truncation that keeps only the single highest-prob token MUST
        // collapse every row to the argmax regardless of temperature/seed.
        let argmax_per_row: Vec<i32> = (0..num_rows)
            .map(|r| {
                let base = r * vocab;
                let mut bi = 0usize;
                let mut bvv = bf16_to_f32(logits_bf[base]);
                for i in 1..vocab {
                    let v = bf16_to_f32(logits_bf[base + i]);
                    if v > bvv { bvv = v; bi = i; }
                }
                bi as i32
            })
            .collect();
        let hot_temp = up(&dev, &vec![1.5f32; num_rows]); // all rows stochastic
        // top_k = 1 → only the max-logit token survives → argmax.
        let topk1 = up(&dev, &vec![1i32; num_rows]);
        let obk = dev.alloc(num_rows * 4).unwrap();
        dev.sample_temp_bf16(&lb, &hot_temp, None, Some(&topk1), None, &sb, &obk, num_rows as i32, vocab as i32).unwrap();
        dev.sync().unwrap();
        let mut outk = vec![0i32; num_rows];
        obk.download(&mut outk).unwrap();
        dev.sync().unwrap();
        assert_eq!(outk, argmax_per_row, "top_k=1 must collapse to argmax");
        // top_p → 0 (tiny) → nucleus is just the max-prob token → argmax.
        let topp_tiny = up(&dev, &vec![1.0e-6f32; num_rows]);
        let obp = dev.alloc(num_rows * 4).unwrap();
        dev.sample_temp_bf16(&lb, &hot_temp, Some(&topp_tiny), None, None, &sb, &obp, num_rows as i32, vocab as i32).unwrap();
        dev.sync().unwrap();
        let mut outp = vec![0i32; num_rows];
        obp.download(&mut outp).unwrap();
        dev.sync().unwrap();
        assert_eq!(outp, argmax_per_row, "tiny top_p must collapse to argmax");
        // top_k = K (a few): the drawn token must be within the K highest logits
        // for every seed — exercise a spread of seeds.
        let k = 4usize;
        let topk = up(&dev, &vec![k as i32; num_rows]);
        let mut topk_sets: Vec<Vec<i32>> = Vec::with_capacity(num_rows);
        for r in 0..num_rows {
            let base = r * vocab;
            let mut idx: Vec<usize> = (0..vocab).collect();
            idx.sort_by(|&a, &b| {
                bf16_to_f32(logits_bf[base + b])
                    .partial_cmp(&bf16_to_f32(logits_bf[base + a]))
                    .unwrap()
            });
            topk_sets.push(idx[..k].iter().map(|&i| i as i32).collect());
        }
        for seed_base in [1u32, 7, 100, 2024, 55555] {
            let seeds = up(&dev, &(0..num_rows as u32).map(|r| seed_base + r).collect::<Vec<_>>());
            let obkk = dev.alloc(num_rows * 4).unwrap();
            dev.sample_temp_bf16(&lb, &hot_temp, None, Some(&topk), None, &seeds, &obkk, num_rows as i32, vocab as i32).unwrap();
            dev.sync().unwrap();
            let mut o = vec![0i32; num_rows];
            obkk.download(&mut o).unwrap();
            dev.sync().unwrap();
            for r in 0..num_rows {
                assert!(topk_sets[r].contains(&o[r]), "top_k={k} drew token {} outside the top-{k} set for row {r}", o[r]);
            }
        }
    }

    #[test]
    fn dtype_cast_parity() {
        let Some(dev) = device_or_skip("dtype_cast_parity") else { return };
        let n = 64usize;
        let src_f32: Vec<f32> = (0..n).map(|i| (i as f32 * 0.13).sin() * 3.0).collect();
        // fp32 → bf16 (round-nearest-even)
        let sb = up(&dev, &src_f32);
        let db = dev.alloc(n * 2).unwrap();
        dev.cast_fp32_to_bf16(&sb, &db, n).unwrap();
        dev.sync().unwrap();
        let mut got = vec![0u16; n];
        db.download(&mut got).unwrap();
        dev.sync().unwrap();
        for i in 0..n {
            assert_eq!(got[i], f32_to_bf16_rne(src_f32[i]), "fp32→bf16 elem {i}");
        }
        // bf16 → fp32 (exact: bits << 16)
        let bf: Vec<u16> = src_f32.iter().map(|&v| f32_to_bf16(v)).collect();
        let bb = up(&dev, &bf);
        let fb = dev.alloc(n * 4).unwrap();
        dev.cast_bf16_to_fp32(&bb, &fb, n).unwrap();
        dev.sync().unwrap();
        let mut gf = vec![0f32; n];
        fb.download(&mut gf).unwrap();
        dev.sync().unwrap();
        for i in 0..n {
            assert_eq!(gf[i].to_bits(), (bf[i] as u32) << 16, "bf16→fp32 elem {i}");
        }
    }

    #[test]
    fn gather_bf16_rows_parity() {
        let Some(dev) = device_or_skip("gather_bf16_rows_parity") else { return };
        let (num_src, vocab) = (5usize, 10usize);
        let src: Vec<u16> = (0..num_src * vocab).map(|i| i as u16).collect();
        let row_indices: Vec<i32> = vec![2, 0, 4];
        let sb = up(&dev, &src);
        let rb = up(&dev, &row_indices);
        let db = dev.alloc(row_indices.len() * vocab * 2).unwrap();
        dev.gather_bf16_rows(&sb, &rb, &db, row_indices.len() as i32, vocab as i32).unwrap();
        dev.sync().unwrap();
        let mut got = vec![0u16; row_indices.len() * vocab];
        db.download(&mut got).unwrap();
        dev.sync().unwrap();
        for (d, &r) in row_indices.iter().enumerate() {
            for c in 0..vocab {
                assert_eq!(got[d * vocab + c], src[r as usize * vocab + c], "dst[{d},{c}]");
            }
        }
    }

    #[test]
    fn rmsnorm_gemma_bf16_parity() {
        let Some(dev) = device_or_skip("rmsnorm_gemma_bf16_parity") else { return };
        let (rows, hidden, eps) = (4usize, 64usize, 1e-5f32);
        let x_f: Vec<f32> = (0..rows * hidden).map(|i| (i as f32 * 0.05).sin()).collect();
        let w_f: Vec<f32> = (0..hidden).map(|i| (i as f32 * 0.03).cos() * 0.2).collect();
        let xb = up(&dev, &x_f.iter().map(|&v| f32_to_bf16(v)).collect::<Vec<_>>());
        let wb = up(&dev, &w_f.iter().map(|&v| f32_to_bf16(v)).collect::<Vec<_>>());
        let yb = dev.alloc(rows * hidden * 2).unwrap();
        dev.rmsnorm_gemma_bf16(&xb, &wb, &yb, rows as i32, hidden as i32, eps).unwrap();
        dev.sync().unwrap();
        let mut y = vec![0u16; rows * hidden];
        yb.download(&mut y).unwrap();
        dev.sync().unwrap();
        for r in 0..rows {
            let ss: f32 = (0..hidden).map(|h| bf16_to_f32(f32_to_bf16(x_f[r * hidden + h])).powi(2)).sum();
            let inv = 1.0 / (ss / hidden as f32 + eps).sqrt();
            for h in 0..hidden {
                let xv = bf16_to_f32(f32_to_bf16(x_f[r * hidden + h]));
                let wv = bf16_to_f32(f32_to_bf16(w_f[h]));
                let want = xv * inv * (1.0 + wv); // Gemma (1+w)
                let got = bf16_to_f32(y[r * hidden + h]);
                assert!((got - want).abs() <= 0.03 * want.abs() + 0.02, "r{r}h{h}: {got} vs {want}");
            }
        }
    }

    #[test]
    fn geglu_tanh_bf16_parity() {
        let Some(dev) = device_or_skip("geglu_tanh_bf16_parity") else { return };
        let n = 128usize;
        let g_f: Vec<f32> = (0..n).map(|i| (i as f32 * 0.07).sin() * 2.0).collect();
        let u_f: Vec<f32> = (0..n).map(|i| (i as f32 * 0.05).cos()).collect();
        let gb = up(&dev, &g_f.iter().map(|&v| f32_to_bf16(v)).collect::<Vec<_>>());
        let ub = up(&dev, &u_f.iter().map(|&v| f32_to_bf16(v)).collect::<Vec<_>>());
        let yb = dev.alloc(n * 2).unwrap();
        dev.geglu_tanh_bf16(&gb, &ub, &yb, n as i32).unwrap();
        dev.sync().unwrap();
        let mut y = vec![0u16; n];
        yb.download(&mut y).unwrap();
        dev.sync().unwrap();
        let c = 0.7978845608028654f32;
        for i in 0..n {
            let g = bf16_to_f32(f32_to_bf16(g_f[i]));
            let u = bf16_to_f32(f32_to_bf16(u_f[i]));
            let want = 0.5 * g * (1.0 + (c * (g + 0.044715 * g * g * g)).tanh()) * u;
            let got = bf16_to_f32(y[i]);
            assert!((got - want).abs() <= 0.03 * want.abs() + 0.02, "i{i}: {got} vs {want}");
        }
    }

    #[test]
    fn logit_softcap_bf16_parity() {
        let Some(dev) = device_or_skip("logit_softcap_bf16_parity") else { return };
        let (n, cap) = (64usize, 30.0f32);
        let x_f: Vec<f32> = (0..n).map(|i| (i as f32 * 0.5) - 16.0).collect();
        let bf: Vec<u16> = x_f.iter().map(|&v| f32_to_bf16(v)).collect();
        let xb = up(&dev, &bf);
        dev.logit_softcap_bf16(&xb, cap, n).unwrap();
        dev.sync().unwrap();
        let mut y = vec![0u16; n];
        xb.download(&mut y).unwrap();
        dev.sync().unwrap();
        for i in 0..n {
            let want = cap * (bf16_to_f32(bf[i]) / cap).tanh();
            let got = bf16_to_f32(y[i]);
            assert!((got - want).abs() <= 0.03 * want.abs() + 0.02, "i{i}: {got} vs {want}");
        }
    }

    #[test]
    fn dequant_fp8_e4m3_parity() {
        let Some(dev) = device_or_skip("dequant_fp8_e4m3_parity") else { return };
        // e4m3 bytes: 0x38=1.0, 0x40=2.0, 0x30=0.5, 0xB8=-1.0 → ×scale.
        let fp8 = [0x38u8, 0x40, 0x30, 0xB8];
        let scale = 2.0f32;
        let fb = up(&dev, &fp8);
        let ob = dev.alloc(fp8.len() * 2).unwrap();
        dev.dequant_fp8_e4m3_to_bf16(&fb, &ob, scale, fp8.len()).unwrap();
        dev.sync().unwrap();
        let mut y = vec![0u16; fp8.len()];
        ob.download(&mut y).unwrap();
        dev.sync().unwrap();
        let want = [2.0f32, 4.0, 1.0, -2.0];
        for i in 0..fp8.len() {
            assert!((bf16_to_f32(y[i]) - want[i]).abs() <= 0.02, "i{i}: {} vs {}", bf16_to_f32(y[i]), want[i]);
        }
    }

    // YaRN with factor=1 must equal the un-scaled base RoPE (validates wiring
    // without re-deriving the piecewise frequency formula — the agent's
    // standalone test already covered the scaled case).
    #[test]
    fn rope_yarn_factor1_equals_base() {
        let Some(dev) = device_or_skip("rope_yarn_factor1_equals_base") else { return };
        let (nt, nq, nkv, hd, theta) = (3usize, 2usize, 1usize, 64usize, 10000.0f32);
        let q: Vec<u16> = (0..nt * nq * hd).map(|i| f32_to_bf16((i as f32 * 0.1).sin())).collect();
        let k: Vec<u16> = (0..nt * nkv * hd).map(|i| f32_to_bf16((i as f32 * 0.13).cos())).collect();
        let pb = up(&dev, &(0..nt as i32).collect::<Vec<_>>());
        let (qa, ka) = (up(&dev, &q), up(&dev, &k));
        dev.rope_bf16(&qa, &ka, &pb, nt as i32, nq as i32, nkv as i32, hd as i32, theta, false).unwrap();
        let (qb, kb) = (up(&dev, &q), up(&dev, &k));
        dev.rope_yarn_bf16(&qb, &kb, &pb, nt as i32, nq as i32, nkv as i32, hd as i32, theta, 1.0, 1.0, 4.0, 8192).unwrap();
        dev.sync().unwrap();
        let dl = |b: &DeviceBuffer, n: usize| { let mut v = vec![0u16; n]; b.download(&mut v).unwrap(); v };
        let (a, b, ak, bk) = (dl(&qa, q.len()), dl(&qb, q.len()), dl(&ka, k.len()), dl(&kb, k.len()));
        dev.sync().unwrap();
        for i in 0..q.len() {
            let (x, y) = (bf16_to_f32(a[i]), bf16_to_f32(b[i]));
            assert!((x - y).abs() <= 1e-2 * x.abs().max(1.0) + 1e-2, "q[{i}]: base {x} yarn1 {y}");
        }
        for i in 0..k.len() {
            let (x, y) = (bf16_to_f32(ak[i]), bf16_to_f32(bk[i]));
            assert!((x - y).abs() <= 1e-2 * x.abs().max(1.0) + 1e-2, "k[{i}]: base {x} yarn1 {y}");
        }
    }

    #[test]
    fn topk_softmax_bf16_parity() {
        let Some(dev) = device_or_skip("topk_softmax_bf16_parity") else { return };
        let (n, ne, k) = (4usize, 8usize, 2usize);
        let logits: Vec<u16> = (0..n * ne).map(|i| f32_to_bf16((i as f32 * 0.37).sin() * 2.0)).collect();
        let lb = up(&dev, &logits);
        let ib = dev.alloc(n * k * 4).unwrap();
        let wb = dev.alloc(n * k * 4).unwrap();
        dev.topk_softmax_bf16(&lb, &ib, &wb, n as i32, ne as i32, k as i32).unwrap();
        dev.sync().unwrap();
        let mut idx = vec![0i32; n * k];
        let mut w = vec![0f32; n * k];
        ib.download(&mut idx).unwrap();
        wb.download(&mut w).unwrap();
        dev.sync().unwrap();
        for r in 0..n {
            // softmax over ALL experts (stable), top-K by descending prob
            // (strict >, lowest index on tie), renormalized by the picked sum.
            let row: Vec<f32> = (0..ne).map(|e| bf16_to_f32(logits[r * ne + e])).collect();
            let m = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let exps: Vec<f32> = row.iter().map(|&x| (x - m).exp()).collect();
            let z: f32 = exps.iter().sum();
            let probs: Vec<f32> = exps.iter().map(|&e| e / z).collect();
            let mut p = probs.clone();
            let mut picks = Vec::new();
            for _ in 0..k {
                let (mut bi, mut bv) = (0usize, f32::NEG_INFINITY);
                for e in 0..ne {
                    if p[e] > bv { bv = p[e]; bi = e; }
                }
                picks.push(bi);
                p[bi] = -1.0;
            }
            let wsum: f32 = picks.iter().map(|&e| probs[e]).sum();
            for j in 0..k {
                assert_eq!(idx[r * k + j] as usize, picks[j], "row {r} pick {j}");
                let want = probs[picks[j]] / wsum;
                assert!((w[r * k + j] - want).abs() <= 0.02, "row {r} w{j}: {} vs {want}", w[r * k + j]);
            }
        }
    }

    #[test]
    fn chunked_swiglu_bf16_parity() {
        let Some(dev) = device_or_skip("chunked_swiglu_bf16_parity") else { return };
        let (n, idim) = (4usize, 64usize);
        let packed: Vec<u16> = (0..n * 2 * idim).map(|x| f32_to_bf16((x as f32 * 0.05).sin() * 1.5)).collect();
        let pb = up(&dev, &packed);
        let yb = dev.alloc(n * idim * 2).unwrap();
        dev.chunked_swiglu_bf16(&pb, &yb, n as i32, idim as i32).unwrap();
        dev.sync().unwrap();
        let mut y = vec![0u16; n * idim];
        yb.download(&mut y).unwrap();
        dev.sync().unwrap();
        for nn in 0..n {
            for ii in 0..idim {
                let g = bf16_to_f32(packed[nn * 2 * idim + ii]);
                let u = bf16_to_f32(packed[nn * 2 * idim + idim + ii]);
                let want = (g / (1.0 + (-g).exp())) * u;
                let got = bf16_to_f32(y[nn * idim + ii]);
                assert!((got - want).abs() <= 0.02, "n{nn}i{ii}: {got} vs {want}");
            }
        }
    }

    #[test]
    fn rope_partial_bf16_parity() {
        let Some(dev) = device_or_skip("rope_partial_bf16_parity") else { return };
        let (nt, nq, nkv, hd, rot, theta) = (3usize, 2usize, 1usize, 64usize, 32usize, 10000.0f32);
        let q: Vec<u16> = (0..nt * nq * hd).map(|i| f32_to_bf16((i as f32 * 0.1).sin())).collect();
        let k: Vec<u16> = (0..nt * nkv * hd).map(|i| f32_to_bf16((i as f32 * 0.13).cos())).collect();
        let pos: Vec<i32> = (0..nt as i32).collect();
        let qb = up(&dev, &q);
        let kb = up(&dev, &k);
        let pb = up(&dev, &pos);
        dev.rope_partial_bf16(&qb, &kb, &pb, nt as i32, nq as i32, nkv as i32, hd as i32, rot as i32, theta).unwrap();
        dev.sync().unwrap();
        let mut qo = vec![0u16; q.len()];
        let mut ko = vec![0u16; k.len()];
        qb.download(&mut qo).unwrap();
        kb.download(&mut ko).unwrap();
        dev.sync().unwrap();
        let half = hd / 2;
        let check = |inp: &[u16], out: &[u16], n_heads: usize| {
            for n in 0..nt {
                let p = pos[n] as f32;
                for h in 0..n_heads {
                    let base = (n * n_heads + h) * hd;
                    for i in 0..hd {
                        let got = bf16_to_f32(out[base + i]);
                        if i < rot / 2 {
                            let (s, c) = (p * theta.powf(-2.0 * i as f32 / hd as f32)).sin_cos();
                            let (a, b) = (bf16_to_f32(inp[base + i]), bf16_to_f32(inp[base + i + half]));
                            let want = a * c - b * s;
                            assert!((got - want).abs() <= 0.03 * want.abs() + 0.02, "rot-lo n{n}h{h}i{i}");
                        } else if (half..half + rot / 2).contains(&i) {
                            let pi = i - half;
                            let (s, c) = (p * theta.powf(-2.0 * pi as f32 / hd as f32)).sin_cos();
                            let (a, b) = (bf16_to_f32(inp[base + pi]), bf16_to_f32(inp[base + i]));
                            let want = b * c + a * s;
                            assert!((got - want).abs() <= 0.03 * want.abs() + 0.02, "rot-hi n{n}h{h}i{i}");
                        } else {
                            assert_eq!(out[base + i], inp[base + i], "passthrough n{n}h{h}i{i}");
                        }
                    }
                }
            }
        };
        check(&q, &qo, nq);
        check(&k, &ko, nkv);
    }

    #[test]
    fn causal_conv1d_prefill_bf16_parity() {
        let Some(dev) = device_or_skip("causal_conv1d_prefill_bf16_parity") else { return };
        let (nn, c, k) = (6usize, 8usize, 4usize);
        let x: Vec<f32> = (0..nn * c).map(|i| (i as f32 * 0.1).sin()).collect();
        let w: Vec<f32> = (0..c * k).map(|i| (i as f32 * 0.07).cos() * 0.5).collect();
        let bias: Vec<f32> = (0..c).map(|i| (i as f32 * 0.3).sin() * 0.1).collect();
        let xb = up(&dev, &x.iter().map(|&v| f32_to_bf16(v)).collect::<Vec<_>>());
        let wb = up(&dev, &w.iter().map(|&v| f32_to_bf16(v)).collect::<Vec<_>>());
        let bb = up(&dev, &bias.iter().map(|&v| f32_to_bf16(v)).collect::<Vec<_>>());
        let yb = dev.alloc(nn * c * 2).unwrap();
        dev.causal_conv1d_prefill_bf16(&xb, &wb, Some(&bb), &yb, nn as i32, c as i32, k as i32).unwrap();
        dev.sync().unwrap();
        let mut y = vec![0u16; nn * c];
        yb.download(&mut y).unwrap();
        dev.sync().unwrap();
        // y[t,ch] = silu( bias[ch] + Σ_k w[ch,k]·x[t-(K-1)+k, ch] ), left-pad zeros.
        for t in 0..nn {
            for ch in 0..c {
                let mut acc = bf16_to_f32(f32_to_bf16(bias[ch]));
                for kk in 0..k {
                    let ti = t as i32 - (k as i32 - 1) + kk as i32;
                    if ti >= 0 {
                        acc += bf16_to_f32(f32_to_bf16(w[ch * k + kk]))
                            * bf16_to_f32(f32_to_bf16(x[(ti as usize) * c + ch]));
                    }
                }
                let want = acc / (1.0 + (-acc).exp());
                let got = bf16_to_f32(y[t * c + ch]);
                assert!((got - want).abs() <= 0.03 * want.abs() + 0.02, "t{t}ch{ch}: {got} vs {want}");
            }
        }
    }

    #[test]
    fn dequant_wna16_int4b8_parity() {
        let Some(dev) = device_or_skip("dequant_wna16_int4b8_parity") else { return };
        let (out_dim, in_dim, group) = (2usize, 32usize, 16usize);
        let vals: Vec<Vec<i32>> = (0..out_dim)
            .map(|o| (0..in_dim).map(|c| ((o * in_dim + c) as i32 % 15) - 7).collect())
            .collect();
        let wpr = in_dim / 8; // int32 words per row
        let mut packed = vec![0i32; out_dim * wpr];
        for o in 0..out_dim {
            for c in 0..in_dim {
                let nib = ((vals[o][c] + 8) & 0xF) as u32; // uint4b8
                packed[o * wpr + c / 8] |= (nib << ((c % 8) * 4)) as i32;
            }
        }
        let gpr = in_dim / group; // groups per row
        let scales: Vec<f32> = (0..out_dim * gpr).map(|i| 0.25 * (1.0 + i as f32)).collect();
        let pb = up(&dev, &packed);
        let sb = up(&dev, &scales.iter().map(|&v| f32_to_bf16(v)).collect::<Vec<_>>());
        let ob = dev.alloc(out_dim * in_dim * 2).unwrap();
        dev.dequant_wna16_int4b8_to_bf16(&pb, &sb, &ob, out_dim as i32, in_dim as i32, group as i32).unwrap();
        dev.sync().unwrap();
        let mut out = vec![0u16; out_dim * in_dim];
        ob.download(&mut out).unwrap();
        dev.sync().unwrap();
        for o in 0..out_dim {
            for c in 0..in_dim {
                let scale = bf16_to_f32(f32_to_bf16(scales[o * gpr + c / group]));
                let want = vals[o][c] as f32 * scale;
                let got = bf16_to_f32(out[o * in_dim + c]);
                assert!((got - want).abs() <= 0.02 + 0.02 * want.abs(), "o{o}c{c}: {got} vs {want}");
            }
        }
    }

    #[test]
    fn moe_mlp_block_bf16_parity() {
        let Some(dev) = device_or_skip("moe_mlp_block_bf16_parity") else { return };
        let (t, h, i, e, k) = (4usize, 16usize, 8usize, 4usize, 2usize);

        // hidden mostly-positive so router logits are dominated by the per-expert
        // scale below ⇒ top-K picks are well separated and bf16 rounding can't flip.
        let hidden_f: Vec<f32> = (0..t * h).map(|x| 0.5 + 0.3 * (x as f32 * 0.17).sin()).collect();
        // router_w[e,h] = 0.1·(e+1) + 0.01·sin(h): logit ≈ 0.1·(e+1)·Σ_h hidden ⇒ ↑ in e.
        let router_f: Vec<f32> = (0..e * h)
            .map(|x| { let (ee, hh) = (x / h, x % h); 0.1 * (ee as f32 + 1.0) + 0.01 * (hh as f32).sin() })
            .collect();
        let wgu_f: Vec<f32> = (0..e * 2 * i * h).map(|x| (x as f32 * 0.013).cos() * 0.4).collect();
        let wdown_f: Vec<f32> = (0..e * h * i).map(|x| (x as f32 * 0.011).sin() * 0.3).collect();

        let q = |v: &[f32]| -> Vec<u16> { v.iter().map(|&x| f32_to_bf16(x)).collect() };
        let (hidden_bf, router_bf, wgu_bf, wdown_bf) =
            (q(&hidden_f), q(&router_f), q(&wgu_f), q(&wdown_f));

        let hb = up(&dev, &hidden_bf);
        let rb = up(&dev, &router_bf);
        let gub = up(&dev, &wgu_bf);
        let db = up(&dev, &wdown_bf);
        let ob = dev.alloc(t * h * 2).unwrap();
        dev.moe_mlp_block_bf16(&hb, &rb, &gub, &db, &ob,
            t as i32, h as i32, i as i32, e as i32, k as i32).unwrap();
        dev.sync().unwrap();
        let mut out = vec![0u16; t * h];
        ob.download(&mut out).unwrap();
        dev.sync().unwrap();

        // ---- CPU reference: decode bf16 inputs, bf16-materialize each stage ----
        let bf = |x: f32| bf16_to_f32(f32_to_bf16(x));
        let d = |v: &[u16]| -> Vec<f32> { v.iter().map(|&x| bf16_to_f32(x)).collect() };
        let (hd, rd, gud, dd) = (d(&hidden_bf), d(&router_bf), d(&wgu_bf), d(&wdown_bf));

        let mut out_ref = vec![0f32; t * h];
        for tt in 0..t {
            // router logits [E] (bf16-materialized), softmax over all experts.
            let logit: Vec<f32> = (0..e)
                .map(|ee| bf((0..h).map(|hh| hd[tt * h + hh] * rd[ee * h + hh]).sum()))
                .collect();
            let m = logit.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let exps: Vec<f32> = logit.iter().map(|&x| (x - m).exp()).collect();
            let z: f32 = exps.iter().sum();
            let probs: Vec<f32> = exps.iter().map(|&x| x / z).collect();
            // top-K (descending prob, strict >, lowest idx), renorm by picked sum.
            let mut p = probs.clone();
            let mut picks = Vec::new();
            for _ in 0..k {
                let (mut bi, mut bv) = (0usize, f32::NEG_INFINITY);
                for ee in 0..e { if p[ee] > bv { bv = p[ee]; bi = ee; } }
                picks.push(bi); p[bi] = -1.0;
            }
            let wsum: f32 = picks.iter().map(|&ee| probs[ee]).sum();
            for &ee in &picks {
                let wk = probs[ee] / wsum;
                // gate||up [2I] = hidden @ wgu_eᵀ ; swiglu → mlp [I] ; down → ffn [H].
                let gu: Vec<f32> = (0..2 * i)
                    .map(|j| bf((0..h).map(|hh| hd[tt * h + hh] * gud[ee * 2 * i * h + j * h + hh]).sum()))
                    .collect();
                let mlp: Vec<f32> = (0..i)
                    .map(|ii| { let (g, u) = (gu[ii], gu[i + ii]); bf((g / (1.0 + (-g).exp())) * u) })
                    .collect();
                for hh in 0..h {
                    let ffn = bf((0..i).map(|ii| mlp[ii] * dd[ee * h * i + hh * i + ii]).sum());
                    out_ref[tt * h + hh] += wk * ffn;
                }
            }
        }
        for tt in 0..t {
            for hh in 0..h {
                let got = bf16_to_f32(out[tt * h + hh]);
                let want = out_ref[tt * h + hh];
                let tol = 0.06 * want.abs() + 0.2;
                assert!((got - want).abs() <= tol, "out[{tt},{hh}]: got {got} want {want} (tol {tol})");
            }
        }
    }

    #[test]
    fn mla_block_bf16_smoke() {
        // Exact numerical parity for the absorbed MLA formulation lives in the
        // standalone selftest (device/src/forward/mla_block_selftest.cu — exact
        // bf16 match). This test locks the *ABI marshalling* into the suite by
        // driving the full block through the C ABI and checking: (a) with W_o=0
        // the block is an EXACT residual passthrough (hidden += o_proj, o_proj=0)
        // ⇒ every pointer/dim is marshalled correctly and the whole chain is
        // finite (a NaN/Inf anywhere would corrupt the zero-weighted o_proj via
        // Inf·0); (b) with a real W_o the output stays finite and actually moves.
        let Some(dev) = device_or_skip("mla_block_bf16_smoke") else { return };
        let (t, h, nh) = (4usize, 256usize, 2usize);
        let (q_lora, kv_lora) = (96usize, 128usize);
        let (qk_nope, qk_rope, v_hd) = (128usize, 64usize, 128usize);
        let page_size = 16usize;

        let wb = |n: usize, f: f32, s: f32| -> Vec<u16> {
            (0..n).map(|i| f32_to_bf16((i as f32 * f).sin() * s)).collect()
        };
        let attn_norm = up(&dev, &wb(h, 0.10, 1.0));
        let q_a_ln = up(&dev, &wb(q_lora, 0.11, 1.0));
        let kv_a_ln = up(&dev, &wb(kv_lora, 0.12, 1.0));
        let w_q_a = up(&dev, &wb(q_lora * h, 0.013, 0.05));
        let w_q_b = up(&dev, &wb(nh * (qk_nope + qk_rope) * q_lora, 0.011, 0.05));
        let w_kv_a = up(&dev, &wb((kv_lora + qk_rope) * h, 0.009, 0.05));
        let w_uk = up(&dev, &wb(nh * kv_lora * qk_nope, 0.007, 0.05));
        let w_uv = up(&dev, &wb(nh * v_hd * kv_lora, 0.006, 0.05));

        let hidden0 = wb(t * h, 0.05, 1.0);
        let positions = up(&dev, &[0i32, 1, 2, 3]);
        let qo_indptr = up(&dev, &[0u32, t as u32]);
        let kv_page_indices = up(&dev, &[0u32]);
        let kv_page_indptr = up(&dev, &[0u32, 1]);
        let kv_last_page_lens = up(&dev, &[t as u32]);

        let dims = MlaBlockDims {
            num_tokens: t as i32, num_requests: 1, hidden_size: h as i32, num_heads: nh as i32,
            q_lora_rank: q_lora as i32, kv_lora_rank: kv_lora as i32,
            qk_nope_head_dim: qk_nope as i32, qk_rope_head_dim: qk_rope as i32,
            v_head_dim: v_hd as i32, page_size: page_size as i32, rms_eps: 1e-6,
            sm_scale: 1.0 / ((kv_lora + qk_rope) as f32).sqrt(), rope_theta: 10000.0,
        };

        // Run the block in place on a fresh hidden + freshly-zeroed cache.
        let run = |w_o: &DeviceBuffer| -> Vec<u16> {
            let hidden = up(&dev, &hidden0);
            let ckv = up(&dev, &vec![0u16; page_size * kv_lora]);
            let kpe = up(&dev, &vec![0u16; page_size * qk_rope]);
            let w = MlaLayerWeights {
                attn_norm: &attn_norm, w_q_a: &w_q_a, q_a_ln: &q_a_ln, w_q_b: &w_q_b,
                w_kv_a: &w_kv_a, kv_a_ln: &kv_a_ln, w_uk: &w_uk, w_uv: &w_uv, w_o,
            };
            dev.mla_block_bf16(&hidden, &w, &positions, &ckv, &kpe, &qo_indptr,
                &kv_page_indices, &kv_page_indptr, &kv_last_page_lens, &dims).unwrap();
            dev.sync().unwrap();
            let mut out = vec![0u16; t * h];
            hidden.download(&mut out).unwrap();
            dev.sync().unwrap();
            out
        };

        // (a) W_o = 0 ⇒ exact residual passthrough (also proves the chain is finite).
        let w_o_zero = up(&dev, &vec![0u16; h * nh * v_hd]);
        assert_eq!(run(&w_o_zero), hidden0, "W_o=0 must be an exact residual passthrough");

        // (b) real W_o ⇒ output stays finite and actually changes.
        let w_o = up(&dev, &wb(h * nh * v_hd, 0.005, 0.05));
        let out = run(&w_o);
        let mut changed = false;
        for (&g, &h0) in out.iter().zip(hidden0.iter()) {
            assert!(bf16_to_f32(g).is_finite(), "MLA output must be finite");
            if g != h0 { changed = true; }
        }
        assert!(changed, "a non-zero W_o must change the block output");
    }

    #[test]
    fn altup_predict_bf16_parity() {
        let Some(dev) = device_or_skip("altup_predict_bf16_parity") else { return };
        let (k, t, h) = (3usize, 4usize, 8usize);
        let streams: Vec<u16> = (0..k * t * h).map(|i| f32_to_bf16((i as f32 * 0.07).sin())).collect();
        let coefs: Vec<f32> = (0..t * k * k).map(|i| (i as f32 * 0.11).cos() * 0.5).collect();
        let sb = up(&dev, &streams);
        let cb = up(&dev, &coefs);
        let pb = dev.alloc(k * t * h * 2).unwrap();
        dev.altup_predict_bf16(&sb, &cb, &pb, k as i32, t as i32, h as i32).unwrap();
        dev.sync().unwrap();
        let mut pred = vec![0u16; k * t * h];
        pb.download(&mut pred).unwrap();
        dev.sync().unwrap();
        for kk in 0..k {
            for tt in 0..t {
                for hh in 0..h {
                    // predictions[k,t,h] = streams[k,t,h] + Σ_j coefs[t,j,k]·streams[j,t,h]
                    let mut want = bf16_to_f32(streams[(kk * t + tt) * h + hh]);
                    for jj in 0..k {
                        want += coefs[(tt * k + jj) * k + kk]
                            * bf16_to_f32(streams[(jj * t + tt) * h + hh]);
                    }
                    let got = bf16_to_f32(pred[(kk * t + tt) * h + hh]);
                    let tol = 0.03 * want.abs() + 0.05;
                    assert!((got - want).abs() <= tol, "k{kk}t{tt}h{hh}: {got} vs {want} (tol {tol})");
                }
            }
        }
    }

    #[test]
    fn altup_correct_bf16_parity() {
        let Some(dev) = device_or_skip("altup_correct_bf16_parity") else { return };
        let (k, t, h, active) = (3usize, 4usize, 8usize, 1usize);
        let pred: Vec<u16> = (0..k * t * h).map(|i| f32_to_bf16((i as f32 * 0.05).sin())).collect();
        let act: Vec<u16> = (0..t * h).map(|i| f32_to_bf16((i as f32 * 0.09).cos())).collect();
        let ccp1: Vec<f32> = (0..t * k).map(|i| (i as f32 * 0.13).sin() * 0.5 + 1.0).collect();
        let pb = up(&dev, &pred);
        let ab = up(&dev, &act);
        let ccb = up(&dev, &ccp1);
        let ob = dev.alloc(k * t * h * 2).unwrap();
        dev.altup_correct_bf16(&pb, &ab, &ccb, &ob, k as i32, t as i32, h as i32, active as i32).unwrap();
        dev.sync().unwrap();
        let mut corr = vec![0u16; k * t * h];
        ob.download(&mut corr).unwrap();
        dev.sync().unwrap();
        for kk in 0..k {
            for tt in 0..t {
                for hh in 0..h {
                    // corrected[k,t,h] = pred[k,t,h] + (act[t,h] - pred[active,t,h])·ccp1[t,k]
                    let p = bf16_to_f32(pred[(kk * t + tt) * h + hh]);
                    let a = bf16_to_f32(act[tt * h + hh]);
                    let pa = bf16_to_f32(pred[(active * t + tt) * h + hh]);
                    let want = p + (a - pa) * ccp1[tt * k + kk];
                    let got = bf16_to_f32(corr[(kk * t + tt) * h + hh]);
                    let tol = 0.03 * want.abs() + 0.05;
                    assert!((got - want).abs() <= tol, "k{kk}t{tt}h{hh}: {got} vs {want} (tol {tol})");
                }
            }
        }
    }

    #[test]
    fn grouped_gemm_bf16_parity() {
        let Some(dev) = device_or_skip("grouped_gemm_bf16_parity") else { return };
        let (e, k, n) = (3usize, 32usize, 16usize);
        let offsets = [0i32, 5, 5, 12]; // expert 1 empty → exercises the M_e==0 skip
        let total = *offsets.last().unwrap() as usize;
        let x: Vec<u16> = (0..total * k).map(|i| f32_to_bf16((i as f32 * 0.03).sin())).collect();
        let w: Vec<u16> = (0..e * n * k).map(|i| f32_to_bf16((i as f32 * 0.017).cos() * 0.5)).collect();
        let xb = up(&dev, &x);
        let wb = up(&dev, &w);
        let yb = dev.alloc(total * n * 2).unwrap();
        dev.grouped_gemm_bf16(&xb, &wb, &offsets, &yb, total as i32, e as i32, n as i32, k as i32).unwrap();
        dev.sync().unwrap();
        let mut y = vec![0u16; total * n];
        yb.download(&mut y).unwrap();
        dev.sync().unwrap();
        for r in 0..total {
            // row r belongs to the first expert ex with r < offsets[ex+1].
            let ex = (0..e).find(|&ee| (r as i32) < offsets[ee + 1]).unwrap();
            for nn in 0..n {
                let want: f32 = (0..k)
                    .map(|kk| bf16_to_f32(x[r * k + kk]) * bf16_to_f32(w[(ex * n + nn) * k + kk]))
                    .sum();
                let got = bf16_to_f32(y[r * n + nn]);
                let tol = 0.05 * want.abs() + 0.1;
                assert!((got - want).abs() <= tol, "r{r}(e{ex})n{nn}: {got} vs {want} (tol {tol})");
            }
        }
    }

    #[test]
    fn mla_forward_bf16_smoke() {
        // Exact numerical parity for the full MLA forward lives in the standalone
        // selftest (device/src/forward/mla_forward_selftest.cu — exact logits +
        // 0/4 argmax mismatch). This locks the *ABI marshalling* of the layered
        // weight bundle into the suite: run embed→2×mla_block→norm→lm_head→argmax
        // and assert logits finite + non-degenerate, argmax ids valid, and the
        // run is deterministic (two identical runs).
        let Some(dev) = device_or_skip("mla_forward_bf16_smoke") else { return };
        let (h, nh) = (256usize, 2usize);
        let (q_lora, kv_lora) = (96usize, 128usize);
        let (qk_nope, qk_rope, v_hd) = (128usize, 64usize, 128usize);
        let (n_layers, vocab, t, page_size, num_pages) = (2usize, 32usize, 4usize, 16usize, 1usize);

        let wb = |n: usize, f: f32, s: f32| -> Vec<u16> {
            (0..n).map(|i| f32_to_bf16((i as f32 * f).sin() * s)).collect()
        };
        // One weight set shared across both layers (enough to exercise marshalling).
        let attn_norm = up(&dev, &wb(h, 0.10, 1.0));
        let q_a_ln = up(&dev, &wb(q_lora, 0.11, 1.0));
        let kv_a_ln = up(&dev, &wb(kv_lora, 0.12, 1.0));
        let w_q_a = up(&dev, &wb(q_lora * h, 0.013, 0.05));
        let w_q_b = up(&dev, &wb(nh * (qk_nope + qk_rope) * q_lora, 0.011, 0.05));
        let w_kv_a = up(&dev, &wb((kv_lora + qk_rope) * h, 0.009, 0.05));
        let w_uk = up(&dev, &wb(nh * kv_lora * qk_nope, 0.007, 0.05));
        let w_uv = up(&dev, &wb(nh * v_hd * kv_lora, 0.006, 0.05));
        let w_o = up(&dev, &wb(h * nh * v_hd, 0.005, 0.05));
        let embed = up(&dev, &wb(vocab * h, 0.02, 1.0));
        let final_norm = up(&dev, &wb(h, 0.08, 1.0));
        let lm_head = up(&dev, &wb(vocab * h, 0.017, 0.05));

        let layer = || MlaLayerWeights {
            attn_norm: &attn_norm, w_q_a: &w_q_a, q_a_ln: &q_a_ln, w_q_b: &w_q_b,
            w_kv_a: &w_kv_a, kv_a_ln: &kv_a_ln, w_uk: &w_uk, w_uv: &w_uv, w_o: &w_o,
        };
        let layers = [layer(), layer()];

        let token_ids = up(&dev, &[1i32, 7, 3, 0]);
        let positions = up(&dev, &[0i32, 1, 2, 3]);
        let qo_indptr = up(&dev, &[0u32, t as u32]);
        let kv_page_indices = up(&dev, &[0u32]);
        let kv_page_indptr = up(&dev, &[0u32, 1]);
        let kv_last_page_lens = up(&dev, &[t as u32]);

        let dims = MlaForwardDims {
            hidden: h as i32, num_heads: nh as i32, q_lora_rank: q_lora as i32,
            kv_lora_rank: kv_lora as i32, qk_nope_head_dim: qk_nope as i32,
            qk_rope_head_dim: qk_rope as i32, v_head_dim: v_hd as i32, vocab: vocab as i32,
            page_size: page_size as i32, num_pages: num_pages as i32, rms_eps: 1e-6,
            sm_scale: 1.0 / ((kv_lora + qk_rope) as f32).sqrt(), rope_theta: 10000.0,
        };

        let run = || -> (Vec<u16>, Vec<i32>) {
            let ckv = up(&dev, &vec![0u16; n_layers * num_pages * page_size * kv_lora]);
            let kpe = up(&dev, &vec![0u16; n_layers * num_pages * page_size * qk_rope]);
            let logits = dev.alloc(t * vocab * 2).unwrap();
            let toks = dev.alloc(t * 4).unwrap();
            dev.mla_forward_bf16(&token_ids, &embed, &layers, &final_norm, &lm_head, &positions,
                &ckv, &kpe, &qo_indptr, &kv_page_indices, &kv_page_indptr, &kv_last_page_lens,
                &logits, &toks, t as i32, 1, &dims).unwrap();
            dev.sync().unwrap();
            let mut lg = vec![0u16; t * vocab];
            let mut tk = vec![0i32; t];
            logits.download(&mut lg).unwrap();
            toks.download(&mut tk).unwrap();
            dev.sync().unwrap();
            (lg, tk)
        };

        let (lg, tk) = run();
        for &v in &lg { assert!(bf16_to_f32(v).is_finite(), "logits must be finite"); }
        for &id in &tk { assert!((0..vocab as i32).contains(&id), "argmax id {id} out of range"); }
        // non-degenerate: logits within a token row are not all identical.
        let row0 = &lg[0..vocab];
        assert!(row0.iter().any(|&v| v != row0[0]), "logits row 0 is degenerate (all equal)");
        // determinism: a second run yields identical logits + ids.
        let (lg2, tk2) = run();
        assert_eq!(lg, lg2, "MLA forward must be deterministic (logits)");
        assert_eq!(tk, tk2, "MLA forward must be deterministic (argmax)");
    }

    #[test]
    fn moe_forward_bf16_smoke() {
        // Exact parity is the standalone selftest (moe_forward_selftest.cu — exact
        // logits + argmax match). This locks the ABI marshalling of the layered
        // MoE weight bundle into the suite: finite logits, valid + non-degenerate
        // argmax, deterministic across two runs.
        let Some(dev) = device_or_skip("moe_forward_bf16_smoke") else { return };
        let (h, nqh, nkvh, hd) = (64usize, 4usize, 2usize, 16usize);
        let (e, top_k, inter) = (4usize, 2usize, 128usize);
        let (n_layers, vocab, t, page_size, num_pages) = (2usize, 32usize, 4usize, 16usize, 1usize);
        let (hq, hkv) = (nqh * hd, nkvh * hd);

        let wb = |n: usize, f: f32, s: f32| -> Vec<u16> {
            (0..n).map(|i| f32_to_bf16((i as f32 * f).sin() * s)).collect()
        };
        let attn_norm = up(&dev, &wb(h, 0.10, 1.0));
        let ffn_norm = up(&dev, &wb(h, 0.09, 1.0));
        let wq = up(&dev, &wb(hq * h, 0.013, 0.05));
        let wk = up(&dev, &wb(hkv * h, 0.014, 0.05));
        let wv = up(&dev, &wb(hkv * h, 0.015, 0.05));
        let wo = up(&dev, &wb(h * hq, 0.012, 0.05));
        let router_w = up(&dev, &wb(e * h, 0.02, 0.05));
        let wgu = up(&dev, &wb(e * 2 * inter * h, 0.007, 0.05));
        let wdown = up(&dev, &wb(e * h * inter, 0.006, 0.05));
        let embed = up(&dev, &wb(vocab * h, 0.02, 1.0));
        let final_norm = up(&dev, &wb(h, 0.08, 1.0));
        let lm_head = up(&dev, &wb(vocab * h, 0.017, 0.05));

        let layer = || MoeLayerWeights {
            attn_norm: &attn_norm, wq: &wq, wk: &wk, wv: &wv, wo: &wo, ffn_norm: &ffn_norm,
            router_w: &router_w, wgu: &wgu, wdown: &wdown,
        };
        let layers = [layer(), layer()];

        let token_ids = up(&dev, &[1i32, 7, 3, 0]);
        let positions = up(&dev, &[0i32, 1, 2, 3]);
        let qo_indptr = up(&dev, &[0u32, t as u32]);
        let kv_page_indices = up(&dev, &[0u32]);
        let kv_page_indptr = up(&dev, &[0u32, 1]);
        let kv_last_page_lens = up(&dev, &[t as u32]);

        let dims = MoeForwardDims {
            hidden_size: h as i32, n_q_heads: nqh as i32, n_kv_heads: nkvh as i32,
            head_dim: hd as i32, intermediate: inter as i32, num_experts: e as i32,
            top_k: top_k as i32, vocab: vocab as i32, page_size: page_size as i32, rms_eps: 1e-6,
            rope_theta: 10000.0,
        };

        let run = || -> (Vec<u16>, Vec<i32>) {
            let kv_k = up(&dev, &vec![0u16; n_layers * num_pages * page_size * hkv]);
            let kv_v = up(&dev, &vec![0u16; n_layers * num_pages * page_size * hkv]);
            let logits = dev.alloc(t * vocab * 2).unwrap();
            let toks = dev.alloc(t * 4).unwrap();
            dev.moe_forward_bf16(&token_ids, &embed, &layers, &final_norm, &lm_head, &positions,
                &kv_k, &kv_v, &qo_indptr, &kv_page_indices, &kv_page_indptr, &kv_last_page_lens,
                &logits, &toks, t as i32, 1, num_pages as i32, &dims).unwrap();
            dev.sync().unwrap();
            let mut lg = vec![0u16; t * vocab];
            let mut tk = vec![0i32; t];
            logits.download(&mut lg).unwrap();
            toks.download(&mut tk).unwrap();
            dev.sync().unwrap();
            (lg, tk)
        };

        let (lg, tk) = run();
        for &v in &lg { assert!(bf16_to_f32(v).is_finite(), "logits must be finite"); }
        for &id in &tk { assert!((0..vocab as i32).contains(&id), "argmax id {id} out of range"); }
        let row0 = &lg[0..vocab];
        assert!(row0.iter().any(|&v| v != row0[0]), "logits row 0 is degenerate (all equal)");
        let (lg2, tk2) = run();
        assert_eq!(lg, lg2, "MoE forward must be deterministic (logits)");
        assert_eq!(tk, tk2, "MoE forward must be deterministic (argmax)");
    }

    #[test]
    fn ssm_selective_scan_bf16_parity() {
        // Single token (N=1, 1 request), state starts at 0 → no recurrence carry:
        //   dt   = max(softplus(dt_raw + dt_bias_h), tmin)
        //   y[dim] = Σ_s (dt·B_s·x[dim])·C_s + D_h·x[dim]
        // (the multi-token carry is covered exactly by ssm_scan_selftest.cu). The
        // CPU ref uses clean fp32; tolerance absorbs the kernel's bf16 state store.
        let Some(dev) = device_or_skip("ssm_selective_scan_bf16_parity") else { return };
        let (heads, hd, state, groups) = (4usize, 4usize, 8usize, 2usize);
        let inter = heads * hd;                       // 16
        let conv_dim = inter + 2 * groups * state;    // 48
        let tmin = 0.001f32;

        let conv_out: Vec<u16> = (0..conv_dim).map(|i| f32_to_bf16((i as f32 * 0.05).sin() * 0.5)).collect();
        let dt_raw: Vec<u16> = (0..heads).map(|i| f32_to_bf16((i as f32 * 0.3).cos() * 0.5)).collect();
        let a_f: Vec<f32> = (0..heads).map(|i| -0.5 - 0.3 * i as f32).collect(); // negative
        let d_f: Vec<f32> = (0..heads).map(|i| 0.1 * (i as f32 + 1.0)).collect();
        let dtb_f: Vec<f32> = (0..heads).map(|i| 0.05 * i as f32).collect();

        let cb = up(&dev, &conv_out);
        let dtb = up(&dev, &dt_raw);
        let ab = up(&dev, &a_f);
        let db = up(&dev, &d_f);
        let dtbias = up(&dev, &dtb_f);
        let state_buf = up(&dev, &vec![0u16; heads * hd * state]); // 1 slot, zeroed
        let qo_indptr = up(&dev, &[0u32, 1u32]);
        let yb = dev.alloc(inter * 2).unwrap();
        let dims = SsmDims {
            num_requests: 1, num_heads: heads as i32, head_dim: hd as i32, state_size: state as i32,
            n_groups: groups as i32, conv_dim: conv_dim as i32, intermediate: inter as i32,
            time_step_min: tmin,
        };
        dev.ssm_selective_scan_bf16(&cb, &dtb, &ab, &db, &dtbias, None, None, &state_buf, None,
            &qo_indptr, &yb, &dims).unwrap();
        dev.sync().unwrap();
        let mut y = vec![0u16; inter];
        yb.download(&mut y).unwrap();
        dev.sync().unwrap();

        let softplus = |x: f32| (1.0 + x.exp()).ln();
        let hpg = heads / groups; // heads per group
        for hh in 0..heads {
            let dtv = softplus(bf16_to_f32(dt_raw[hh]) + dtb_f[hh]).max(tmin);
            let g = hh / hpg;
            for dim in 0..hd {
                let x = bf16_to_f32(conv_out[hh * hd + dim]);
                let mut want = 0f32;
                for s in 0..state {
                    let b_s = bf16_to_f32(conv_out[inter + g * state + s]);
                    let c_s = bf16_to_f32(conv_out[inter + groups * state + g * state + s]);
                    want += (dtv * b_s * x) * c_s;
                }
                want += d_f[hh] * x;
                let got = bf16_to_f32(y[hh * hd + dim]);
                let tol = 0.04 * want.abs() + 0.05;
                assert!((got - want).abs() <= tol, "h{hh}d{dim}: {got} vs {want} (tol {tol})");
            }
        }
    }

    #[test]
    fn qgemm_w4a16_bf16_parity() {
        // Internalized de-branded Marlin int4 (u4b8) fused GEMM through the ABI:
        // out[M,N] = act[M,K] @ dequant(qweight)^T, vs the dequant-then-matmul
        // oracle (the very equivalence NEW_DRIVER §10 requires: fused ≡ dequant).
        let Some(dev) = device_or_skip("qgemm_w4a16_bf16_parity") else { return };
        let (m, n, k, group) = (16usize, 256usize, 512usize, 128usize);
        let num_groups = k / group; // 4
        // q[k][n] ∈ [0,15] (u4b8: stored v ↦ signed v-8). Deterministic spread.
        let qval = |kk: usize, nn: usize| -> i32 { ((kk * 7 + nn * 3 + 1) % 16) as i32 };
        // GPTQ row-major pack: [K/8, N] int32, nibble j of (kp,n) = q[kp*8+j][n].
        let mut qrow = vec![0i32; (k / 8) * n];
        for kk in 0..k {
            for nn in 0..n {
                qrow[(kk / 8) * n + nn] |= (qval(kk, nn) & 0xF) << (4 * (kk % 8));
            }
        }
        let scale_f: Vec<f32> = (0..num_groups * n).map(|i| 0.03 + 0.02 * (i as f32 * 0.05).sin()).collect();
        let scale_bf: Vec<u16> = scale_f.iter().map(|&v| f32_to_bf16(v)).collect();
        let act: Vec<u16> = (0..m * k).map(|i| f32_to_bf16((i as f32 * 0.02).sin() * 0.5)).collect();

        let qrow_b = up(&dev, &qrow);
        let qpacked = dev.alloc((k / 16) * (n * 16 / 8) * 4).unwrap();
        dev.qgemm_w4a16_repack(&qrow_b, &qpacked, n as i32, k as i32).unwrap();
        let scales_b = up(&dev, &scale_bf);
        let act_b = up(&dev, &act);
        let out_b = dev.alloc(m * n * 2).unwrap();
        let ws_ints = dev.qgemm_w4a16_workspace_ints(n as i32, m as i32);
        let ws = up(&dev, &vec![0i32; ws_ints as usize]); // zeroed
        dev.qgemm_w4a16_bf16(&act_b, &qpacked, &scales_b, &out_b, m as i32, n as i32, k as i32,
            group as i32, &ws, 0).unwrap();
        dev.sync().unwrap();
        let mut out = vec![0u16; m * n];
        out_b.download(&mut out).unwrap();
        dev.sync().unwrap();

        // oracle: w_f[n,k] = (q-8)·scale[k/group, n]; y[m,n] = Σ_k act[m,k]·w_f[n,k]
        for mm in 0..m {
            for nn in 0..n {
                let want: f32 = (0..k).map(|kk| {
                    let w = (qval(kk, nn) - 8) as f32 * bf16_to_f32(scale_bf[(kk / group) * n + nn]);
                    bf16_to_f32(act[mm * k + kk]) * w
                }).sum();
                let got = bf16_to_f32(out[mm * n + nn]);
                let tol = 0.05 * want.abs() + 0.08;
                assert!((got - want).abs() <= tol, "out[{mm},{nn}]: {got} vs {want} (tol {tol})");
            }
        }
    }

    #[test]
    fn qgemm_w8a16_fp8_bf16_parity() {
        // fp8 (fe4m3fn) fan-out over the qgemm template, through the ABI, vs the
        // dequant-then-matmul oracle. Weights start as fe4m3fn BYTES (decoded by a
        // matching e4m3fn decoder for the oracle), so no host encode is needed.
        let Some(dev) = device_or_skip("qgemm_w8a16_fp8_bf16_parity") else { return };
        let (m, n, k, group) = (16usize, 256usize, 512usize, 128usize);
        let num_groups = k / group;
        // e4m3fn decode (bias 7; exp==0 → subnormal). Matches the kernel's dequant.
        let dec = |b: u8| -> f32 {
            let s = if b & 0x80 != 0 { -1.0 } else { 1.0 };
            let exp = ((b >> 3) & 0xF) as i32;
            let mant = (b & 0x7) as f32;
            if exp == 0 { s * (mant / 8.0) * 2f32.powi(-6) }
            else { s * (1.0 + mant / 8.0) * 2f32.powi(exp - 7) }
        };
        // byte(k,n): exp ∈ [2,8] (no subnormal edge, no exp==15 NaN), mant ∈ [0,7].
        let byte = |kk: usize, nn: usize| -> u8 {
            let idx = kk * n + nn;
            let sign = ((idx / 64) & 1) as u8;
            let exp = (2 + (idx % 7)) as u8;
            let mant = (idx % 8) as u8;
            (sign << 7) | (exp << 3) | mant
        };
        // GPTQ-style pack: [K/4, N] int32, byte j of (kp,n) = fp8 for k=kp*4+j.
        let mut qrow = vec![0i32; (k / 4) * n];
        for kk in 0..k {
            for nn in 0..n {
                qrow[(kk / 4) * n + nn] |= ((byte(kk, nn) as u32) << (8 * (kk % 4))) as i32;
            }
        }
        let scale_f: Vec<f32> = (0..num_groups * n).map(|i| 0.03 + 0.02 * (i as f32 * 0.05).sin()).collect();
        let scale_bf: Vec<u16> = scale_f.iter().map(|&v| f32_to_bf16(v)).collect();
        let act: Vec<u16> = (0..m * k).map(|i| f32_to_bf16((i as f32 * 0.02).sin() * 0.5)).collect();

        let qrow_b = up(&dev, &qrow);
        let qpacked = dev.alloc((k / 16) * (n * 16 / 4) * 4).unwrap();
        dev.qgemm_w8a16_fp8_repack(&qrow_b, &qpacked, n as i32, k as i32).unwrap();
        let scales_b = up(&dev, &scale_bf);
        let act_b = up(&dev, &act);
        let out_b = dev.alloc(m * n * 2).unwrap();
        let ws_ints = dev.qgemm_w8a16_fp8_workspace_ints(n as i32, m as i32);
        let ws = up(&dev, &vec![0i32; ws_ints as usize]);
        dev.qgemm_w8a16_fp8_bf16(&act_b, &qpacked, &scales_b, &out_b, m as i32, n as i32, k as i32,
            group as i32, &ws, 0).unwrap();
        dev.sync().unwrap();
        let mut out = vec![0u16; m * n];
        out_b.download(&mut out).unwrap();
        dev.sync().unwrap();

        for mm in 0..m {
            for nn in 0..n {
                let want: f32 = (0..k).map(|kk| {
                    let w = dec(byte(kk, nn)) * bf16_to_f32(scale_bf[(kk / group) * n + nn]);
                    bf16_to_f32(act[mm * k + kk]) * w
                }).sum();
                let got = bf16_to_f32(out[mm * n + nn]);
                let tol = 0.05 * want.abs() + 0.1;
                assert!((got - want).abs() <= tol, "fp8 out[{mm},{nn}]: {got} vs {want} (tol {tol})");
            }
        }
    }

    #[test]
    fn moe_sparse_block_bf16_parity() {
        // The sparse routed path must equal the validated dense moe_mlp_block on
        // identical inputs (same per-(token,expert) arithmetic; only the
        // permutation/combine bookkeeping differs).
        let Some(dev) = device_or_skip("moe_sparse_block_bf16_parity") else { return };
        let (t, h, i, e, k) = (8usize, 64usize, 128usize, 4usize, 2usize);
        let wb = |n: usize, f: f32, s: f32| -> Vec<u16> {
            (0..n).map(|x| f32_to_bf16((x as f32 * f).sin() * s)).collect()
        };
        let hidden = up(&dev, &wb(t * h, 0.05, 1.0));
        let router_w = up(&dev, &wb(e * h, 0.02, 0.5));
        let wgu = up(&dev, &wb(e * 2 * i * h, 0.013, 0.05));
        let wdown = up(&dev, &wb(e * h * i, 0.011, 0.05));
        let run = |sparse: bool| -> Vec<u16> {
            let out = dev.alloc(t * h * 2).unwrap();
            if sparse {
                dev.moe_sparse_block_bf16(&hidden, &router_w, &wgu, &wdown, &out,
                    t as i32, h as i32, i as i32, e as i32, k as i32).unwrap();
            } else {
                dev.moe_mlp_block_bf16(&hidden, &router_w, &wgu, &wdown, &out,
                    t as i32, h as i32, i as i32, e as i32, k as i32).unwrap();
            }
            dev.sync().unwrap();
            let mut o = vec![0u16; t * h];
            out.download(&mut o).unwrap();
            dev.sync().unwrap();
            o
        };
        let dense = run(false);
        let sparse = run(true);
        for (idx, (&s, &dv)) in sparse.iter().zip(dense.iter()).enumerate() {
            let (gs, gd) = (bf16_to_f32(s), bf16_to_f32(dv));
            let tol = 0.03 * gd.abs() + 0.05;
            assert!((gs - gd).abs() <= tol, "elem {idx}: sparse {gs} vs dense {gd} (tol {tol})");
        }
    }

    #[test]
    fn nemotron_mamba_block_bf16_smoke() {
        // Exact parity is the standalone selftest (nemotron_block_selftest.cu). ABI
        // marshalling smoke: out_proj fuses the residual (beta=1), so out_proj_w=0
        // ⇒ exact residual passthrough (proves the whole in_proj→split→conv→scan→
        // gated-norm chain marshalled + finite); a real out_proj_w changes the
        // output, staying finite + deterministic.
        let Some(dev) = device_or_skip("nemotron_mamba_block_bf16_smoke") else { return };
        let (t, h) = (6usize, 64usize);
        let (heads, hd, state, groups, kconv) = (4usize, 16usize, 16usize, 2usize, 4usize);
        let inter = heads * hd;                    // 64
        let conv_dim = inter + 2 * groups * state; // 128
        let d_in_proj = inter + conv_dim + heads;  // 196
        let wb = |n: usize, f: f32, s: f32| -> Vec<u16> {
            (0..n).map(|x| f32_to_bf16((x as f32 * f).sin() * s)).collect()
        };
        let in_proj_w = up(&dev, &wb(d_in_proj * h, 0.007, 0.05));
        let conv_w = up(&dev, &wb(conv_dim * kconv, 0.02, 0.3));
        let conv_bias = up(&dev, &wb(conv_dim, 0.03, 0.1));
        let a_log = up(&dev, &wb(heads, 0.1, 0.5));
        let d_w = up(&dev, &wb(heads, 0.2, 0.5));
        let dt_bias = up(&dev, &wb(heads, 0.15, 0.3));
        let norm_weight = up(&dev, &wb(inter, 0.09, 1.0));
        let hidden0 = wb(t * h, 0.05, 1.0);

        let run = |out_proj_w: &DeviceBuffer| -> Vec<u16> {
            let hidden = up(&dev, &hidden0);
            let w = NemotronMambaWeights {
                in_proj_w: &in_proj_w, conv_w: &conv_w, conv_bias: Some(&conv_bias),
                a_log: &a_log, d: &d_w, dt_bias: &dt_bias, norm_weight: &norm_weight, out_proj_w,
            };
            dev.nemotron_mamba_block_bf16(&hidden, &w, t as i32, h as i32, heads as i32, hd as i32,
                state as i32, groups as i32, kconv as i32, 1e-6, 0.001).unwrap();
            dev.sync().unwrap();
            let mut o = vec![0u16; t * h];
            hidden.download(&mut o).unwrap();
            dev.sync().unwrap();
            o
        };

        let zero_w = up(&dev, &vec![0u16; h * inter]);
        assert_eq!(run(&zero_w), hidden0, "out_proj_w=0 must be an exact residual passthrough");
        let real_w = up(&dev, &wb(h * inter, 0.006, 0.05));
        let out = run(&real_w);
        let mut changed = false;
        for (&g, &h0) in out.iter().zip(hidden0.iter()) {
            assert!(bf16_to_f32(g).is_finite(), "output must be finite");
            if g != h0 { changed = true; }
        }
        assert!(changed, "a non-zero out_proj_w must change the output");
        assert_eq!(run(&real_w), out, "block must be deterministic");
    }

    #[test]
    fn deepseek_forward_bf16_smoke() {
        // Exact parity is the standalone selftest (deepseek_forward_selftest.cu —
        // exact logits + 0/4 argmax, both FFN paths). This locks the ABI
        // marshalling of the DeepSeek bundle (nested MLA attn + per-layer dense|MoE
        // FFN) into the suite. n_layers=2, first_k_dense=1 ⇒ layer 0 dense, layer 1 MoE.
        let Some(dev) = device_or_skip("deepseek_forward_bf16_smoke") else { return };
        let (h, nh) = (256usize, 2usize);
        let (q_lora, kv_lora) = (96usize, 128usize);
        let (qk_nope, qk_rope, v_hd) = (128usize, 64usize, 128usize);
        let (vocab, t, page_size, num_pages) = (32usize, 4usize, 16usize, 1usize);
        let (e, top_k, moe_inter, dense_inter) = (4usize, 2usize, 128usize, 128usize);

        let wb = |n: usize, f: f32, s: f32| -> Vec<u16> {
            (0..n).map(|i| f32_to_bf16((i as f32 * f).sin() * s)).collect()
        };
        let attn_norm = up(&dev, &wb(h, 0.10, 1.0));
        let q_a_ln = up(&dev, &wb(q_lora, 0.11, 1.0));
        let kv_a_ln = up(&dev, &wb(kv_lora, 0.12, 1.0));
        let w_q_a = up(&dev, &wb(q_lora * h, 0.013, 0.05));
        let w_q_b = up(&dev, &wb(nh * (qk_nope + qk_rope) * q_lora, 0.011, 0.05));
        let w_kv_a = up(&dev, &wb((kv_lora + qk_rope) * h, 0.009, 0.05));
        let w_uk = up(&dev, &wb(nh * kv_lora * qk_nope, 0.007, 0.05));
        let w_uv = up(&dev, &wb(nh * v_hd * kv_lora, 0.006, 0.05));
        let w_o = up(&dev, &wb(h * nh * v_hd, 0.005, 0.05));
        let ffn_norm = up(&dev, &wb(h, 0.08, 1.0));
        let dgate = up(&dev, &wb(dense_inter * h, 0.014, 0.05));
        let dup = up(&dev, &wb(dense_inter * h, 0.015, 0.05));
        let ddown = up(&dev, &wb(h * dense_inter, 0.016, 0.05));
        let router_w = up(&dev, &wb(e * h, 0.02, 0.05));
        let wgu = up(&dev, &wb(e * 2 * moe_inter * h, 0.007, 0.05));
        let wdown = up(&dev, &wb(e * h * moe_inter, 0.006, 0.05));
        let embed = up(&dev, &wb(vocab * h, 0.02, 1.0));
        let final_norm = up(&dev, &wb(h, 0.085, 1.0));
        let lm_head = up(&dev, &wb(vocab * h, 0.017, 0.05));

        let attn = || MlaLayerWeights {
            attn_norm: &attn_norm, w_q_a: &w_q_a, q_a_ln: &q_a_ln, w_q_b: &w_q_b,
            w_kv_a: &w_kv_a, kv_a_ln: &kv_a_ln, w_uk: &w_uk, w_uv: &w_uv, w_o: &w_o,
        };
        let layers = [
            DeepseekLayerWeights { // layer 0: dense SwiGLU FFN
                attn: attn(), ffn_norm: &ffn_norm,
                w_gate: Some(&dgate), w_up: Some(&dup), w_down: Some(&ddown),
                router_w: None, wgu: None, wdown: None,
            },
            DeepseekLayerWeights { // layer 1: top-K MoE FFN
                attn: attn(), ffn_norm: &ffn_norm,
                w_gate: None, w_up: None, w_down: None,
                router_w: Some(&router_w), wgu: Some(&wgu), wdown: Some(&wdown),
            },
        ];
        let n_layers = layers.len();

        let token_ids = up(&dev, &[1i32, 7, 3, 0]);
        let positions = up(&dev, &[0i32, 1, 2, 3]);
        let qo_indptr = up(&dev, &[0u32, t as u32]);
        let kv_page_indices = up(&dev, &[0u32]);
        let kv_page_indptr = up(&dev, &[0u32, 1]);
        let kv_last_page_lens = up(&dev, &[t as u32]);

        let dims = DeepseekForwardDims {
            first_k_dense: 1, hidden: h as i32, num_heads: nh as i32, q_lora_rank: q_lora as i32,
            kv_lora_rank: kv_lora as i32, qk_nope_head_dim: qk_nope as i32,
            qk_rope_head_dim: qk_rope as i32, v_head_dim: v_hd as i32,
            dense_inter: dense_inter as i32, moe_inter: moe_inter as i32, num_experts: e as i32,
            top_k: top_k as i32, vocab: vocab as i32, page_size: page_size as i32,
            num_pages: num_pages as i32, rms_eps: 1e-6,
            sm_scale: 1.0 / ((kv_lora + qk_rope) as f32).sqrt(), rope_theta: 10000.0,
        };

        let run = || -> (Vec<u16>, Vec<i32>) {
            let ckv = up(&dev, &vec![0u16; n_layers * num_pages * page_size * kv_lora]);
            let kpe = up(&dev, &vec![0u16; n_layers * num_pages * page_size * qk_rope]);
            let logits = dev.alloc(t * vocab * 2).unwrap();
            let toks = dev.alloc(t * 4).unwrap();
            dev.deepseek_forward_bf16(&token_ids, &embed, &layers, &final_norm, &lm_head,
                &positions, &ckv, &kpe, &qo_indptr, &kv_page_indices, &kv_page_indptr,
                &kv_last_page_lens, &logits, &toks, t as i32, 1, &dims).unwrap();
            dev.sync().unwrap();
            let mut lg = vec![0u16; t * vocab];
            let mut tk = vec![0i32; t];
            logits.download(&mut lg).unwrap();
            toks.download(&mut tk).unwrap();
            dev.sync().unwrap();
            (lg, tk)
        };
        let (lg, tk) = run();
        for &v in &lg { assert!(bf16_to_f32(v).is_finite(), "logits must be finite"); }
        for &id in &tk { assert!((0..vocab as i32).contains(&id), "argmax id {id} out of range"); }
        let row0 = &lg[0..vocab];
        assert!(row0.iter().any(|&v| v != row0[0]), "logits row 0 is degenerate (all equal)");
        let (lg2, _) = run();
        assert_eq!(lg, lg2, "DeepSeek forward must be deterministic");
    }

    #[test]
    fn gemma_forward_bf16_smoke() {
        // Exact parity is gemma_forward_selftest.cu. ABI-marshalling smoke: finite
        // logits, valid+non-degenerate argmax, deterministic. Exercises the
        // sliding(layer0=2)/full(layer1) alternation + both soft-caps + √H embed scale.
        let Some(dev) = device_or_skip("gemma_forward_bf16_smoke") else { return };
        let (h, nq, nkv, hd, inter) = (64usize, 4, 2, 16, 128);
        let (n_layers, vocab, t, page_size, num_pages) = (2usize, 32, 4, 16, 1);
        let (hq, hkv) = (nq * hd, nkv * hd);

        let wb = |n: usize, f: f32, s: f32| -> Vec<u16> {
            (0..n).map(|i| f32_to_bf16((i as f32 * f).sin() * s)).collect()
        };
        let n1 = up(&dev, &wb(h, 0.10, 1.0));
        let n2 = up(&dev, &wb(h, 0.11, 1.0));
        let n3 = up(&dev, &wb(h, 0.12, 1.0));
        let n4 = up(&dev, &wb(h, 0.13, 1.0));
        let wq = up(&dev, &wb(hq * h, 0.013, 0.1));
        let wk = up(&dev, &wb(hkv * h, 0.014, 0.1));
        let wv = up(&dev, &wb(hkv * h, 0.015, 0.1));
        let wo = up(&dev, &wb(h * hq, 0.012, 0.1));
        let wg = up(&dev, &wb(inter * h, 0.007, 0.1));
        let wu = up(&dev, &wb(inter * h, 0.008, 0.1));
        let wd = up(&dev, &wb(h * inter, 0.006, 0.1));
        let embed = up(&dev, &wb(vocab * h, 0.02, 1.0));
        let final_norm = up(&dev, &wb(h, 0.085, 1.0));
        let lm_head = up(&dev, &wb(vocab * h, 0.017, 0.1));

        let layer = || GemmaLayerWeights {
            input_ln: &n1, post_attn_ln: &n2, pre_ffn_ln: &n3, post_ffn_ln: &n4,
            wq: &wq, wk: &wk, wv: &wv, wo: &wo, w_gate: &wg, w_up: &wu, w_down: &wd,
        };
        let layers = [layer(), layer()];

        let token_ids = up(&dev, &[1i32, 7, 3, 0]);
        let positions = up(&dev, &[0i32, 1, 2, 3]);
        let qo_indptr = up(&dev, &[0u32, t as u32]);
        let kv_page_indices = up(&dev, &[0u32]);
        let kv_page_indptr = up(&dev, &[0u32, 1]);
        let kv_last_page_lens = up(&dev, &[t as u32]);

        let dims = GemmaForwardDims {
            hidden: h as i32, n_q_heads: nq as i32, n_kv_heads: nkv as i32, head_dim: hd as i32,
            intermediate: inter as i32, vocab: vocab as i32, page_size: page_size as i32,
            num_pages: num_pages as i32, window_left: vec![2, -1], window_left_all: -1,
            attn_logit_softcap: 50.0, final_logit_softcap: 30.0, embed_scale: (h as f32).sqrt(),
            rms_eps: 1e-6, rope_theta: 10000.0, qk_norm: 0, altup_num_inputs: 1,
        };

        let ws = dev.workspace(t as i32, h as i32, nq as i32, nkv as i32, hd as i32,
                               inter as i32, vocab as i32).unwrap();
        let run = || -> (Vec<u16>, Vec<i32>) {
            let kv_k = up(&dev, &vec![0u16; n_layers * num_pages * page_size * hkv]);
            let kv_v = up(&dev, &vec![0u16; n_layers * num_pages * page_size * hkv]);
            let logits = dev.alloc(t * vocab * 2).unwrap();
            let toks = dev.alloc(t * 4).unwrap();
            dev.gemma_forward_bf16(&ws, &token_ids, &embed, &layers, &final_norm, &lm_head, &positions,
                &kv_k, &kv_v, &qo_indptr, &kv_page_indices, &kv_page_indptr, &kv_last_page_lens,
                &logits, &toks, t as i32, 1, &dims).unwrap();
            dev.sync().unwrap();
            let mut lg = vec![0u16; t * vocab];
            let mut tk = vec![0i32; t];
            logits.download(&mut lg).unwrap();
            toks.download(&mut tk).unwrap();
            dev.sync().unwrap();
            (lg, tk)
        };
        let (lg, tk) = run();
        for &v in &lg { assert!(bf16_to_f32(v).is_finite(), "logits must be finite"); }
        for &id in &tk { assert!((0..vocab as i32).contains(&id), "argmax id {id} out of range"); }
        let row0 = &lg[0..vocab];
        assert!(row0.iter().any(|&v| v != row0[0]), "logits row 0 is degenerate");
        let (lg2, _) = run();
        assert_eq!(lg, lg2, "Gemma forward must be deterministic");
    }

    #[test]
    fn nemotron_forward_bf16_smoke() {
        // Exact parity is nemotron_forward_selftest.cu. ABI-marshalling smoke for
        // the hybrid stack: schedule [M,A,F,M] exercises Mamba mixer + GQA
        // attention + dense FFN dispatch. Finite logits, valid argmax, deterministic.
        let Some(dev) = device_or_skip("nemotron_forward_bf16_smoke") else { return };
        let (h, vocab, t) = (64usize, 32, 4);
        let (mh, mhd, state, groups, kconv) = (4usize, 16, 16, 2, 4);
        let (nq, nkv, ahd, page_size) = (4usize, 2, 16, 16);
        let ffn_inter = 128usize;
        let inter_m = mh * mhd; // 64
        let conv_dim = inter_m + 2 * groups * state; // 128
        let d_in_proj = inter_m + conv_dim + mh; // 196
        let (hq, hkv) = (nq * ahd, nkv * ahd);

        let wb = |n: usize, f: f32, s: f32| -> Vec<u16> {
            (0..n).map(|i| f32_to_bf16((i as f32 * f).sin() * s)).collect()
        };
        let m_pre = up(&dev, &wb(h, 0.10, 1.0));
        let in_proj = up(&dev, &wb(d_in_proj * h, 0.007, 0.05));
        let conv_w = up(&dev, &wb(conv_dim * kconv, 0.02, 0.3));
        let conv_b = up(&dev, &wb(conv_dim, 0.03, 0.1));
        let a_log = up(&dev, &wb(mh, 0.1, 0.5));
        let d_w = up(&dev, &wb(mh, 0.2, 0.5));
        let dt_b = up(&dev, &wb(mh, 0.15, 0.3));
        let m_norm = up(&dev, &wb(inter_m, 0.09, 1.0));
        let out_proj = up(&dev, &wb(h * inter_m, 0.006, 0.05));
        let a_norm = up(&dev, &wb(h, 0.11, 1.0));
        let wq = up(&dev, &wb(hq * h, 0.013, 0.1));
        let wk = up(&dev, &wb(hkv * h, 0.014, 0.1));
        let wv = up(&dev, &wb(hkv * h, 0.015, 0.1));
        let wo = up(&dev, &wb(h * hq, 0.012, 0.1));
        let f_norm = up(&dev, &wb(h, 0.12, 1.0));
        let fg = up(&dev, &wb(ffn_inter * h, 0.007, 0.1));
        let fu = up(&dev, &wb(ffn_inter * h, 0.008, 0.1));
        let fd = up(&dev, &wb(h * ffn_inter, 0.006, 0.1));
        let embed = up(&dev, &wb(vocab * h, 0.02, 1.0));
        let final_norm = up(&dev, &wb(h, 0.085, 1.0));
        let lm_head = up(&dev, &wb(vocab * h, 0.017, 0.1));

        let mamba_w = || NemotronMambaWeights {
            in_proj_w: &in_proj, conv_w: &conv_w, conv_bias: Some(&conv_b), a_log: &a_log,
            d: &d_w, dt_bias: &dt_b, norm_weight: &m_norm, out_proj_w: &out_proj,
        };
        let layers = [
            NemotronLayer::Mamba { pre_norm: &m_pre, w: mamba_w() },
            NemotronLayer::Attn { attn_norm: &a_norm, wq: &wq, wk: &wk, wv: &wv, wo: &wo },
            NemotronLayer::Ffn { ffn_norm: &f_norm, w_gate: &fg, w_up: &fu, w_down: &fd },
            NemotronLayer::Mamba { pre_norm: &m_pre, w: mamba_w() },
        ];

        let token_ids = up(&dev, &[1i32, 7, 3, 0]);
        let positions = up(&dev, &[0i32, 1, 2, 3]);
        let dims = NemotronForwardDims {
            hidden: h as i32, vocab: vocab as i32, mamba_num_heads: mh as i32,
            mamba_head_dim: mhd as i32, mamba_state_size: state as i32, mamba_n_groups: groups as i32,
            mamba_conv_kernel: kconv as i32, time_step_min: 0.001, attn_n_q_heads: nq as i32,
            attn_n_kv_heads: nkv as i32, attn_head_dim: ahd as i32, page_size: page_size as i32,
            rope_theta: 10000.0, ffn_intermediate: ffn_inter as i32, rms_eps: 1e-6,
        };

        let run = || -> (Vec<u16>, Vec<i32>) {
            let logits = dev.alloc(t * vocab * 2).unwrap();
            let toks = dev.alloc(t * 4).unwrap();
            dev.nemotron_forward_bf16(&token_ids, &embed, &layers, &final_norm, &lm_head,
                &positions, &logits, &toks, t as i32, &dims).unwrap();
            dev.sync().unwrap();
            let mut lg = vec![0u16; t * vocab];
            let mut tk = vec![0i32; t];
            logits.download(&mut lg).unwrap();
            toks.download(&mut tk).unwrap();
            dev.sync().unwrap();
            (lg, tk)
        };
        let (lg, tk) = run();
        for &v in &lg { assert!(bf16_to_f32(v).is_finite(), "logits must be finite"); }
        for &id in &tk { assert!((0..vocab as i32).contains(&id), "argmax id {id} out of range"); }
        let row0 = &lg[0..vocab];
        assert!(row0.iter().any(|&v| v != row0[0]), "logits row 0 is degenerate");
        let (lg2, _) = run();
        assert_eq!(lg, lg2, "Nemotron forward must be deterministic");
    }
}
