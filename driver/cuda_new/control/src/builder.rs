//! Construction — replaces `driver/cuda/src/entry.cpp::run_impl` (the
//! 1,286-line, 55-branch god-function). Ties `arch::detect` + `mem::plan` +
//! `loader` into one `model dir → ready Model` path. There is no
//! `is_*_arch` cascade: `detect()` picks the arch from the config, and the
//! rest is a straight flow (config → spec → plan → load → alloc).
//!
//! Slice scope (documented): dense bf16 llama-like; the KV cache is one
//! page per layer for the prefill forward (the contiguous-KV-append slice).
//! `mem::plan` still computes the real capacity numbers (`num_pages`,
//! `max_tokens`, `max_requests`) — the eventual READY handshake / paged KV
//! cache uses those; wiring the full paged cache + scatter is the next step.

use std::path::Path;

use anyhow::{Context, Result, anyhow};

use std::sync::atomic::{AtomicU64, Ordering};

use pie_bridge::{ForwardRequest, ForwardResponse, Sampler};

use crate::arch::{self, ArchSpec};
use crate::device::{Device, DeviceBuffer, Workspace, bf16_to_f32};
use crate::ffi::PieArchId;
use crate::loader::{
    DeepseekConfig, GemmaConfig, LoadedDeepseek, LoadedGemma, LoadedLlama, LoadedMoe, MoeConfig,
};
use crate::mem::{self, MemPlan, Profile};

/// Boot configuration — the cuda_new analog of the `[model.driver]` TOML.
pub struct BootConfig {
    pub device_ordinal: i32,
    pub memory_profile: Profile,
    pub gpu_mem_utilization: f64,
    /// KV page size (tokens per page).
    pub page_size: usize,
    /// Pages allocated per layer in the KV / MLA cache — the batch capacity a
    /// `fire` can spread requests across. (`mem::plan` computes the real number;
    /// the builder uses this directly so the slice stays self-contained.)
    pub num_kv_pages: usize,
    /// Host (CPU) KV page slots for swap-out/in under memory pressure (Copy
    /// D2H/H2D/H2H). `0` disables host swap (advertised as `swap_pool_size: 0`).
    pub swap_pool_size: usize,
}

/// Per-arch resident state: the loaded weights + the cache shape that arch
/// uses. The `is_*_arch` cascade is replaced by `detect()` choosing the variant
/// here; each `prefill` dispatches on it. Llama-like uses a paged K/V cache +
/// a forward workspace; DeepSeek-MLA uses the compressed-latent ckv/kpe cache
/// (the per-layer block allocates its own scratch, so no workspace).
enum Backend<'a> {
    Llama {
        w: LoadedLlama<'a>,
        kv_k: DeviceBuffer<'a>,
        kv_v: DeviceBuffer<'a>,
        ws: Workspace<'a>,
    },
    Deepseek {
        w: LoadedDeepseek<'a>,
        ckv: DeviceBuffer<'a>,
        kpe: DeviceBuffer<'a>,
    },
    Moe {
        w: LoadedMoe<'a>,
        kv_k: DeviceBuffer<'a>,
        kv_v: DeviceBuffer<'a>,
    },
    Gemma {
        w: LoadedGemma<'a>,
        kv_k: DeviceBuffer<'a>,
        kv_v: DeviceBuffer<'a>,
        ws: Workspace<'a>,
    },
}

/// A fully-constructed, ready-to-run model: weights resident, cache + workspace
/// allocated, and the derived capacity plan retained for capability reporting.
pub struct Model<'a> {
    pub spec: ArchSpec,
    pub plan: MemPlan,
    backend: Backend<'a>,
    page_size: usize,
    /// Pages allocated per layer (the KV/MLA cache stride a `fire` lays requests
    /// across). Passed to the forward as `num_kv_pages`.
    num_pages: usize,
    /// Host KV swap pool (Copy D2H/H2D/H2H). Interior-mutable so `copy_pages`
    /// can stage through it behind `&self`.
    swap: std::cell::RefCell<SwapPool>,
}

/// Recursively drop JSON object keys whose value is `null` so serde field
/// defaults apply (a present `null` fails a non-Option field; `#[serde(default)]`
/// only covers ABSENT keys). Recurses into nested objects/arrays (e.g. the
/// multimodal `text_config`).
fn strip_json_nulls(v: &mut serde_json::Value) {
    match v {
        serde_json::Value::Object(map) => {
            map.retain(|_, val| !val.is_null());
            for val in map.values_mut() {
                strip_json_nulls(val);
            }
        }
        serde_json::Value::Array(arr) => {
            for val in arr.iter_mut() {
                strip_json_nulls(val);
            }
        }
        _ => {}
    }
}

/// Build a ready model from a directory holding `config.json` and
/// `model.safetensors`.
pub fn build<'a>(dev: &'a Device, model_dir: &Path, cfg: &BootConfig) -> Result<Model<'a>> {
    // 1. config.json → HfConfig → arch → spec  (no is_*_arch cascade)
    let cfg_path = model_dir.join("config.json");
    // Read to a Value first and strip null-valued keys: HF configs (esp.
    // multimodal ones like gemma-4-E4B) ship explicit `null`s (e.g.
    // `num_experts: null`), and serde's `#[serde(default)]` only covers ABSENT
    // keys — a present `null` errors against a non-Option field. Stripping
    // nulls turns them into "absent" → the field default.
    let mut raw: serde_json::Value = serde_json::from_reader(
        std::fs::File::open(&cfg_path).with_context(|| format!("opening {cfg_path:?}"))?,
    )
    .context("reading config.json")?;
    strip_json_nulls(&mut raw);
    let hf: arch::HfConfig =
        serde_json::from_value(raw).context("parsing config.json")?;
    // Multimodal checkpoints (gemma-4-E4B) nest the LM dims under `text_config`.
    let hf = hf.resolve_multimodal();
    let archi = arch::detect(&hf)?;
    let spec = archi.spec(&hf)?;
    let (nkv, hd) = (spec.num_kv_heads, spec.head_dim);

    // 2. memory plan (the real construction decision)
    let (free, total) = dev.mem_info()?;
    let (sm_count, major, minor) = dev.props()?;
    let plan_in = mem::PlanInputs {
        free_bytes: free,
        total_bytes: total,
        weight_bytes: 0, // informational; planner uses the live probe
        gpu_mem_utilization: cfg.gpu_mem_utilization,
        sm_count,
        compute_major: major,
        compute_minor: minor,
        tp_size: 1,
        num_layers: spec.num_layers,
        hidden_size: spec.hidden_size,
        intermediate_size: spec.intermediate_size,
        vocab_size: spec.vocab_size,
        head_dim: hd,
        head_dim_kernel: hd,
        num_attention_heads: spec.num_heads,
        num_key_value_heads: nkv,
        max_model_len: spec.max_model_len,
        model_type: hf.dispatch_key(),
        kv_bytes_per_token: match &spec.mla {
            // MLA: one compressed latent + rope key per layer (bf16, 2B each).
            Some(m) => spec.num_layers * (m.kv_lora_rank + m.qk_rope_head_dim) * 2,
            None => spec.num_layers * nkv * hd * 4, // k+v, bf16 (2B each)
        },
        recurrent_slot_bytes: 0,
        max_intermediate: spec.intermediate_size,
        max_hq: spec.num_heads * hd,
        max_hk: nkv * hd,
        extra_arena_bytes_per_token: 0,
        runtime_quant_scratch_base_bytes: 0,
        runtime_quant_scratch_bytes_per_token: 0,
    };
    let plan = mem::plan(&plan_in, cfg.memory_profile).map_err(|e| anyhow!("mem::plan: {e:?}"))?;

    // 3-4. Per-arch weights + cache. `detect()` chose the kind; build the
    // matching backend (the `is_*_arch` cascade is gone). Slice: 1 page/layer
    // for the prefill forward.
    let page_size = cfg.page_size;
    let num_pages = cfg.num_kv_pages.max(1);
    let backend = build_backend(dev, model_dir, &spec, &hf, page_size, num_pages)?;

    // Host (CPU) swap pool for paged-K/V backends (MLA host-swap deferred). A
    // slot holds one physical KV page across all layers (k and v separately).
    let kv_backend = matches!(
        backend,
        Backend::Llama { .. } | Backend::Moe { .. } | Backend::Gemma { .. }
    );
    let (slots, slot_bytes) = if cfg.swap_pool_size > 0 && kv_backend {
        let hkv = spec.num_kv_heads * spec.head_dim;
        (cfg.swap_pool_size, spec.num_layers * page_size * hkv * 2)
    } else {
        (0, 0)
    };
    let swap = std::cell::RefCell::new(SwapPool {
        slots,
        slot_bytes,
        k: vec![0u8; slots * slot_bytes],
        v: vec![0u8; slots * slot_bytes],
    });
    Ok(Model { spec, plan, backend, page_size, num_pages, swap })
}

/// Host (pinned-able) KV page pool for swap-out/in under memory pressure. Each
/// of `slots` slots holds one physical KV page across all layers, `slot_bytes`
/// each, for K and V separately. `slots == 0` ⇒ swap disabled.
struct SwapPool {
    slots: usize,
    slot_bytes: usize,
    k: Vec<u8>,
    v: Vec<u8>,
}

/// Build the per-arch resident backend (weights + cache + any workspace). One
/// match arm per forward family — the replacement for run_impl's branch maze.
fn build_backend<'a>(
    dev: &'a Device,
    model_dir: &Path,
    spec: &ArchSpec,
    hf: &arch::HfConfig,
    page_size: usize,
    num_pages: usize,
) -> Result<Backend<'a>> {
    // The loaders take the model DIR and resolve single-file vs sharded
    // (`model.safetensors` or `model.safetensors.index.json`) internally.
    let st_path = model_dir.to_path_buf();
    match spec.id {
        PieArchId::LlamaLike | PieArchId::Qwen3 => {
            let (nkv, hd) = (spec.num_kv_heads, spec.head_dim);
            let w = LoadedLlama::load(dev, &st_path, hd, spec.rms_norm_eps, spec.rope_theta)?;
            let kv_elems = spec.num_layers * num_pages * page_size * nkv * hd;
            let kv_k = dev.alloc(kv_elems * 2)?;
            let kv_v = dev.alloc(kv_elems * 2)?;
            // Workspace token capacity = the full KV capacity (num_pages *
            // page_size), which is exactly the `max_forward_tokens` we report
            // in the READY caps. A single fire can never reference more tokens
            // than the cache holds, so the per-token scratch can't overflow —
            // including a multi-page prefill (T > page_size), the runtime's
            // real prefill shape. (Sizing this to `page_size` silently
            // corrupted/faulted any prefill longer than one page.)
            let ws_max_tokens = (num_pages * page_size) as i32;
            let ws = dev.workspace(
                ws_max_tokens, spec.hidden_size as i32, spec.num_heads as i32, nkv as i32,
                hd as i32, spec.intermediate_size as i32, spec.vocab_size as i32,
            )?;
            Ok(Backend::Llama { w, kv_k, kv_v, ws })
        }
        PieArchId::DeepseekV4 | PieArchId::Kimi | PieArchId::Glm5 => {
            let mla = spec.mla.ok_or_else(|| anyhow!("MLA arch '{:?}' has no mla dims in spec", spec.id))?;
            let dcfg = DeepseekConfig {
                vocab: spec.vocab_size,
                hidden: spec.hidden_size,
                n_layers: spec.num_layers,
                num_heads: spec.num_heads,
                q_lora_rank: mla.q_lora_rank,
                kv_lora_rank: mla.kv_lora_rank,
                qk_nope_head_dim: mla.qk_nope_head_dim,
                qk_rope_head_dim: mla.qk_rope_head_dim,
                v_head_dim: mla.v_head_dim,
                first_k_dense: hf.first_k_dense_replace,
                dense_inter: spec.intermediate_size,
                moe_inter: if spec.moe_intermediate_size > 0 {
                    spec.moe_intermediate_size
                } else {
                    spec.intermediate_size
                },
                num_experts: spec.moe_experts,
                top_k: spec.num_experts_per_tok,
                rms_eps: spec.rms_norm_eps,
                rope_theta: spec.rope_theta,
            };
            let w = LoadedDeepseek::load(dev, &st_path, dcfg)?;
            // MLA latent cache: [n_layers, num_pages, page_size, *], zeroed.
            let ckv = upload_zeros(dev, spec.num_layers * num_pages * page_size * mla.kv_lora_rank)?;
            let kpe = upload_zeros(dev, spec.num_layers * num_pages * page_size * mla.qk_rope_head_dim)?;
            Ok(Backend::Deepseek { w, ckv, kpe })
        }
        PieArchId::Qwen3_5Moe => {
            let (nkv, hd) = (spec.num_kv_heads, spec.head_dim);
            let mcfg = MoeConfig {
                vocab: spec.vocab_size,
                hidden: spec.hidden_size,
                n_layers: spec.num_layers,
                n_q_heads: spec.num_heads,
                n_kv_heads: nkv,
                head_dim: hd,
                moe_inter: if spec.moe_intermediate_size > 0 {
                    spec.moe_intermediate_size
                } else {
                    spec.intermediate_size
                },
                num_experts: spec.moe_experts,
                top_k: spec.num_experts_per_tok,
                rms_eps: spec.rms_norm_eps,
                rope_theta: spec.rope_theta,
            };
            let w = LoadedMoe::load(dev, &st_path, mcfg)?;
            let kv_elems = spec.num_layers * num_pages * page_size * nkv * hd;
            let kv_k = dev.alloc(kv_elems * 2)?;
            let kv_v = dev.alloc(kv_elems * 2)?;
            Ok(Backend::Moe { w, kv_k, kv_v })
        }
        PieArchId::Gemma4 => {
            let (nkv, hd) = (spec.num_kv_heads, spec.head_dim);
            // Per-layer sliding/full window: SlidingAttention → `sliding_window-1`
            // visible-left tokens; FullAttention → -1 (no window). Empty when the
            // arch has no window configured at all (full causal).
            let window_left: Vec<i32> = if spec.sliding_window.is_some() {
                spec.layer_kinds
                    .iter()
                    .map(|k| match k {
                        crate::arch::LayerKind::SlidingAttention => {
                            spec.sliding_window.map(|w| w as i32 - 1).unwrap_or(-1)
                        }
                        _ => -1,
                    })
                    .collect()
            } else {
                Vec::new()
            };
            let gcfg = GemmaConfig {
                head_dim: hd,
                rms_eps: spec.rms_norm_eps,
                rope_theta: spec.rope_theta,
                attn_logit_softcap: spec.attn_logit_softcap.unwrap_or(0.0),
                final_logit_softcap: spec.final_logit_softcap.unwrap_or(0.0),
                window_left,
            };
            let w = LoadedGemma::load(dev, &st_path, gcfg)?;
            let kv_elems = spec.num_layers * num_pages * page_size * nkv * hd;
            let kv_k = dev.alloc(kv_elems * 2)?;
            let kv_v = dev.alloc(kv_elems * 2)?;
            // Persistent activation + attn-plan scratch (no per-fire cudaMalloc;
            // also the CUDA-graph prerequisite). Sized to the full KV token
            // capacity like the llama arm.
            let ws_max_tokens = (num_pages * page_size) as i32;
            let ws = dev.workspace(
                ws_max_tokens, spec.hidden_size as i32, spec.num_heads as i32, nkv as i32,
                hd as i32, spec.intermediate_size as i32, spec.vocab_size as i32,
            )?;
            Ok(Backend::Gemma { w, kv_k, kv_v, ws })
        }
        other => Err(anyhow!(
            "builder: arch {other:?} detected but its forward/cache is not yet wired in the \
             builder (its device forward exists; builder routing is incremental — see PLAN B2)"
        )),
    }
}

/// Per-row sampling parameters flattened from a wire [`Sampler`] into the
/// scalar form the device kernel consumes. Disabled filters use the kernel's
/// sentinels (top_p ≥ 1, top_k ≤ 0, min_p ≤ 0); `temperature <= 0` ⇒ argmax.
#[derive(Clone, Copy)]
struct SamplerRow {
    temperature: f32,
    top_p: f32,
    top_k: i32,
    min_p: f32,
    seed: u32,
}

/// Monotonic seed source for samplers that don't carry a caller seed (every
/// variant except `Multinomial{seed != 0}`). A fresh value per draw makes
/// stochastic rows actually vary. The wire contract for an unseeded sampler is
/// "fresh per-fire random seed", so the counter is XORed with a per-process
/// entropy base (clock + pid) — otherwise two fresh processes would replay the
/// same "random" stream from the fixed initial counter.
static SAMPLE_SEED_CTR: AtomicU64 = AtomicU64::new(0x243F6A8885A308D3);

fn fresh_seed() -> u32 {
    use std::sync::OnceLock;
    static BASE: OnceLock<u64> = OnceLock::new();
    let base = *BASE.get_or_init(|| {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0);
        nanos ^ (std::process::id() as u64).wrapping_mul(0x9E3779B97F4A7C15)
    });
    let v = SAMPLE_SEED_CTR.fetch_add(0x9E3779B97F4A7C15, Ordering::Relaxed) ^ base;
    // Mix and force non-zero (0 is the kernel's "no seed" sentinel for greedy
    // rows, but a stochastic row must never collapse to a constant).
    ((v ^ (v >> 33)) as u32) | 1
}

impl SamplerRow {
    fn greedy() -> Self {
        Self { temperature: 0.0, top_p: 1.0, top_k: 0, min_p: 0.0, seed: 0 }
    }

    fn from_sampler(s: Option<&Sampler>) -> Self {
        match s {
            Some(Sampler::Multinomial { temperature, seed }) => Self {
                temperature: *temperature,
                top_p: 1.0,
                top_k: 0,
                min_p: 0.0,
                seed: if *seed != 0 { *seed } else { fresh_seed() },
            },
            Some(Sampler::TopK { temperature, k }) => Self {
                temperature: *temperature, top_p: 1.0, top_k: *k as i32, min_p: 0.0, seed: fresh_seed(),
            },
            Some(Sampler::TopP { temperature, p }) => Self {
                temperature: *temperature, top_p: *p, top_k: 0, min_p: 0.0, seed: fresh_seed(),
            },
            Some(Sampler::MinP { temperature, p }) => Self {
                temperature: *temperature, top_p: 1.0, top_k: 0, min_p: *p, seed: fresh_seed(),
            },
            Some(Sampler::TopKTopP { temperature, k, p }) => Self {
                temperature: *temperature, top_p: *p, top_k: *k as i32, min_p: 0.0, seed: fresh_seed(),
            },
            // RawLogits / Dist / Logprob(s) / Entropy / Embedding / None: no
            // token-sampling semantics here yet → greedy argmax fallback.
            _ => Self::greedy(),
        }
    }
}

/// True for samplers that emit a sampled token id (vs a side-channel payload
/// — RawLogits/Dist/Logprob(s)/Entropy/Embedding).
fn is_token_sampler(s: &Sampler) -> bool {
    matches!(
        s,
        Sampler::Multinomial { .. }
            | Sampler::TopK { .. }
            | Sampler::TopP { .. }
            | Sampler::MinP { .. }
            | Sampler::TopKTopP { .. }
    )
}

/// Numerically-stable log-sum-exp of an f32 logit row.
fn logsumexp(logits: &[f32]) -> f32 {
    let m = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if !m.is_finite() {
        return m;
    }
    m + logits.iter().map(|&l| (l - m).exp()).sum::<f32>().ln()
}

impl Model<'_> {
    /// Run one single-request prefill forward, dispatching on the backend
    /// (paged-KV llama vs MLA-latent DeepSeek). Returns `(out_logits,
    /// out_tokens)` device buffers (the latter is the forward's greedy argmax).
    /// Syncs before returning so the uploaded input buffers are safe to drop.
    fn run_forward<'b>(
        &self, dev: &'b Device, token_ids: &[i32],
    ) -> Result<(DeviceBuffer<'b>, DeviceBuffer<'b>)> {
        let t = token_ids.len();
        if t == 0 || t > self.page_size {
            return Err(anyhow!("prefill length {t} must be in 1..={}", self.page_size));
        }
        let vocab = self.spec.vocab_size;
        let positions: Vec<i32> = (0..t as i32).collect();
        let tib = upload(dev, token_ids)?;
        let pb = upload(dev, &positions)?;
        let qib = upload(dev, &[0u32, t as u32])?;
        let kpi = upload(dev, &[0u32])?;
        let kpp = upload(dev, &[0u32, 1])?;
        let klp = upload(dev, &[t as u32])?;
        let out_logits = dev.alloc(t * vocab * 2)?;
        let out_tokens = dev.alloc(t * 4)?;
        let (ti, np) = (t as i32, self.num_pages as i32);
        self.dispatch_forward(
            dev, &tib, &pb, &qib, &kpi, &kpp, &klp, &out_logits, &out_tokens, ti, 1, np)?;
        dev.sync()?;
        Ok((out_logits, out_tokens))
    }

    /// Dispatch one forward on the backend with caller-assembled inputs + CSR
    /// page lists (shared by single-request `run_forward` and batched `fire`).
    #[allow(clippy::too_many_arguments)]
    fn dispatch_forward(
        &self, dev: &Device, tib: &DeviceBuffer, pb: &DeviceBuffer, qib: &DeviceBuffer,
        kpi: &DeviceBuffer, kpp: &DeviceBuffer, klp: &DeviceBuffer, out_logits: &DeviceBuffer,
        out_tokens: &DeviceBuffer, num_tokens: i32, num_requests: i32, num_pages: i32,
    ) -> Result<()> {
        let ps = self.page_size as i32;
        match &self.backend {
            Backend::Llama { w, kv_k, kv_v, ws } => w.forward(
                dev, ws, tib, pb, kv_k, kv_v, qib, kpi, kpp, klp, out_logits, out_tokens,
                num_tokens, num_requests, ps, num_pages,
            ),
            Backend::Deepseek { w, ckv, kpe } => w.forward(
                dev, tib, pb, ckv, kpe, qib, kpi, kpp, klp, out_logits, out_tokens,
                num_tokens, num_requests, ps, num_pages,
            ),
            Backend::Moe { w, kv_k, kv_v } => w.forward(
                dev, tib, pb, kv_k, kv_v, qib, kpi, kpp, klp, out_logits, out_tokens,
                num_tokens, num_requests, ps, num_pages,
            ),
            Backend::Gemma { w, kv_k, kv_v, ws } => w.forward(
                dev, ws, tib, pb, kv_k, kv_v, qib, kpi, kpp, klp, out_logits, out_tokens,
                num_tokens, num_requests, ps, num_pages,
            ),
        }
    }

    /// Single-request prefill, greedy. Returns the argmax next-token id per
    /// position. `token_ids.len()` must fit one KV page (`page_size`).
    pub fn prefill_greedy(&self, dev: &Device, token_ids: &[i32]) -> Result<Vec<i32>> {
        let (_logits, out_tokens) = self.run_forward(dev, token_ids)?;
        let mut toks = vec![0i32; token_ids.len()];
        out_tokens.download(&mut toks)?;
        dev.sync()?;
        Ok(toks)
    }

    /// Single-request prefill with temperature sampling. Runs the forward for
    /// logits, then `sample_temp_bf16` (Gumbel-max) per row. `temperature <= 0`
    /// collapses to argmax; `base_seed` makes the draw reproducible.
    pub fn prefill_sample(&self, dev: &Device, token_ids: &[i32], temperature: f32, base_seed: u32)
        -> Result<Vec<i32>> {
        let t = token_ids.len();
        let vocab = self.spec.vocab_size;
        let (out_logits, _tokens) = self.run_forward(dev, token_ids)?;

        // Sample from the logits. Per-row params (uniform temperature here);
        // distinct per-row seeds keep the Gumbel noise independent.
        let temps = vec![temperature; t];
        let minps = vec![0.0f32; t];
        let seeds: Vec<u32> = (0..t as u32).map(|r| base_seed.wrapping_add(r)).collect();
        let tb = upload(dev, &temps)?;
        let mb = upload(dev, &minps)?;
        let sb = upload(dev, &seeds)?;
        let sampled = dev.alloc(t * 4)?;
        dev.sample_temp_bf16(&out_logits, &tb, None, None, Some(&mb), &sb, &sampled, t as i32, vocab as i32)?;
        dev.sync()?;
        let mut toks = vec![0i32; t];
        sampled.download(&mut toks)?;
        dev.sync()?;
        Ok(toks)
    }

    /// Batched multi-request prefill — the `handle_fire_batch` core. Lays R
    /// independent requests across the paged KV cache (request i gets
    /// `ceil(len_i / page_size)` contiguous pages), runs ONE forward, and
    /// returns the greedy next token at each request's last position. Requests
    /// are independent (per-request causal mask + private KV pages), so each
    /// result equals running that request alone — the batched-equivalence
    /// invariant the executor relies on.
    pub fn fire_batch(&self, dev: &Device, requests: &[Vec<i32>]) -> Result<Vec<i32>> {
        let r = requests.len();
        if r == 0 {
            return Ok(Vec::new());
        }
        let ps = self.page_size;
        let vocab = self.spec.vocab_size;

        // Assemble the batched inputs + flashinfer-style CSR page lists.
        let mut tokens: Vec<i32> = Vec::new();
        let mut positions: Vec<i32> = Vec::new();
        let mut qo_indptr: Vec<u32> = vec![0];
        let mut kv_page_indptr: Vec<u32> = vec![0];
        let mut kv_page_indices: Vec<u32> = Vec::new();
        let mut kv_last_page_lens: Vec<u32> = Vec::new();
        let mut next_page: u32 = 0;
        for q in requests {
            let len = q.len();
            if len == 0 || len > ps * self.num_pages {
                return Err(anyhow!("fire_batch: request length {len} out of 1..={}", ps * self.num_pages));
            }
            tokens.extend_from_slice(q);
            positions.extend(0..len as i32);
            qo_indptr.push(qo_indptr.last().unwrap() + len as u32);
            let npages = len.div_ceil(ps);
            for _ in 0..npages {
                kv_page_indices.push(next_page);
                next_page += 1;
            }
            kv_page_indptr.push(kv_page_indptr.last().unwrap() + npages as u32);
            kv_last_page_lens.push((len - (npages - 1) * ps) as u32);
        }
        if next_page as usize > self.num_pages {
            return Err(anyhow!(
                "fire_batch: batch needs {next_page} pages > {} allocated (raise num_kv_pages)",
                self.num_pages
            ));
        }
        let total = tokens.len();

        let tib = upload(dev, &tokens)?;
        let pb = upload(dev, &positions)?;
        let qib = upload(dev, &qo_indptr)?;
        let kpi = upload(dev, &kv_page_indices)?;
        let kpp = upload(dev, &kv_page_indptr)?;
        let klp = upload(dev, &kv_last_page_lens)?;
        let out_logits = dev.alloc(total * vocab * 2)?;
        let out_tokens = dev.alloc(total * 4)?;
        self.dispatch_forward(
            dev, &tib, &pb, &qib, &kpi, &kpp, &klp, &out_logits, &out_tokens,
            total as i32, r as i32, self.num_pages as i32,
        )?;
        dev.sync()?;
        let mut toks = vec![0i32; total];
        out_tokens.download(&mut toks)?;
        dev.sync()?;

        // Next token per request = the forward's greedy argmax at its last position.
        Ok((0..r).map(|i| toks[qo_indptr[i + 1] as usize - 1]).collect())
    }

    /// Process `new_tokens` for a SINGLE request whose cache already holds
    /// `pre_len` tokens (request 0, pages `0..ceil((pre_len+len)/page_size)`).
    /// Prefill is `pre_len == 0`; a decode step is `new_tokens.len() == 1` with
    /// `pre_len` = current sequence length. The KV-append kernel writes the new
    /// tokens at offset `pre_len` (it derives `pre = total − new`), and attention
    /// reads `[0, pre_len+len)`. Returns the per-position greedy argmax.
    fn step_single(&self, dev: &Device, new_tokens: &[i32], pre_len: usize) -> Result<Vec<i32>> {
        let ps = self.page_size;
        let new_len = new_tokens.len();
        let total = pre_len + new_len;
        let npages = total.div_ceil(ps);
        if npages > self.num_pages {
            return Err(anyhow!(
                "step: sequence length {total} exceeds cache capacity {} tokens",
                self.num_pages * ps
            ));
        }
        let vocab = self.spec.vocab_size;
        let positions: Vec<i32> = (pre_len as i32..total as i32).collect();
        let kv_page_indices: Vec<u32> = (0..npages as u32).collect();
        let tib = upload(dev, new_tokens)?;
        let pb = upload(dev, &positions)?;
        let qib = upload(dev, &[0u32, new_len as u32])?;
        let kpi = upload(dev, &kv_page_indices)?;
        let kpp = upload(dev, &[0u32, npages as u32])?;
        let klp = upload(dev, &[(total - (npages - 1) * ps) as u32])?;
        let out_logits = dev.alloc(new_len * vocab * 2)?;
        let out_tokens = dev.alloc(new_len * 4)?;
        self.dispatch_forward(
            dev, &tib, &pb, &qib, &kpi, &kpp, &klp, &out_logits, &out_tokens,
            new_len as i32, 1, self.num_pages as i32,
        )?;
        dev.sync()?;
        let mut toks = vec![0i32; new_len];
        out_tokens.download(&mut toks)?;
        dev.sync()?;
        Ok(toks)
    }

    /// Run one forward over caller-assembled inputs + CSR page lists (the
    /// runtime's `ForwardRequest` shape). Returns the resident `(out_logits
    /// [T, vocab] bf16, out_tokens [T] greedy-argmax i32, T)` device buffers,
    /// synced. The serve loop samples from `out_logits`; the argmax buffer is
    /// the greedy fast path.
    #[allow(clippy::too_many_arguments)]
    fn forward_logits_csr<'b>(
        &self, dev: &'b Device, tokens: &[i32], positions: &[i32], qo_indptr: &[u32],
        kv_page_indices: &[u32], kv_page_indptr: &[u32], kv_last_page_lens: &[u32],
    ) -> Result<(DeviceBuffer<'b>, DeviceBuffer<'b>, usize)> {
        let t = tokens.len();
        let vocab = self.spec.vocab_size;
        let num_requests = qo_indptr.len().saturating_sub(1).max(1) as i32;
        let tib = upload(dev, tokens)?;
        let pb = upload(dev, positions)?;
        let qib = upload(dev, qo_indptr)?;
        let kpi = upload(dev, kv_page_indices)?;
        let kpp = upload(dev, kv_page_indptr)?;
        let klp = upload(dev, kv_last_page_lens)?;
        let out_logits = dev.alloc(t * vocab * 2)?;
        let out_tokens = dev.alloc(t * 4)?;
        self.dispatch_forward(dev, &tib, &pb, &qib, &kpi, &kpp, &klp, &out_logits, &out_tokens,
            t as i32, num_requests, self.num_pages as i32)?;
        dev.sync()?;
        Ok((out_logits, out_tokens, t))
    }

    /// Run one forward and return the per-token greedy argmax (`out_token_ids`).
    #[allow(clippy::too_many_arguments)]
    pub fn forward_tokens(
        &self, dev: &Device, tokens: &[i32], positions: &[i32], qo_indptr: &[u32],
        kv_page_indices: &[u32], kv_page_indptr: &[u32], kv_last_page_lens: &[u32],
    ) -> Result<Vec<i32>> {
        let t = tokens.len();
        if t == 0 {
            return Ok(Vec::new());
        }
        let (_logits, out_tokens, _t) = self.forward_logits_csr(
            dev, tokens, positions, qo_indptr, kv_page_indices, kv_page_indptr, kv_last_page_lens)?;
        let mut toks = vec![0i32; t];
        out_tokens.download(&mut toks)?;
        dev.sync()?;
        Ok(toks)
    }

    /// Per-row Gumbel-max sampling over a pre-gathered compact `[S, vocab]` bf16
    /// buffer (temperature + top-p/top-k/min-p; `temperature <= 0` ⇒ argmax).
    /// Returns the `S` token ids in row order.
    fn sample_compact(
        &self, dev: &Device, compact: &DeviceBuffer, params: &[SamplerRow], s: usize, vocab: usize,
    ) -> Result<Vec<u32>> {
        let temps: Vec<f32> = params.iter().map(|p| p.temperature).collect();
        let topps: Vec<f32> = params.iter().map(|p| p.top_p).collect();
        let topks: Vec<i32> = params.iter().map(|p| p.top_k).collect();
        let minps: Vec<f32> = params.iter().map(|p| p.min_p).collect();
        let seeds: Vec<u32> = params.iter().map(|p| p.seed).collect();
        let tb = upload(dev, &temps)?;
        let pp = upload(dev, &topps)?;
        let kk = upload(dev, &topks)?;
        let mp = upload(dev, &minps)?;
        let sb = upload(dev, &seeds)?;
        let sampled = dev.alloc(s * 4)?;
        dev.sample_temp_bf16(compact, &tb, Some(&pp), Some(&kk), Some(&mp), &sb, &sampled,
            s as i32, vocab as i32)?;
        dev.sync()?;
        let mut toks = vec![0i32; s];
        sampled.download(&mut toks)?;
        dev.sync()?;
        Ok(toks.into_iter().map(|x| x as u32).collect())
    }

    /// Capacity numbers for the READY handshake.
    pub fn caps(&self) -> (usize, usize, usize) {
        (self.num_pages, self.spec.vocab_size, self.spec.max_model_len)
    }

    /// Serve one wire `ForwardRequest`: run the forward, then for each sampling
    /// row emit the output its `Sampler` asks for, dispatched to the matching
    /// `ForwardResponse` channel. Token samplers (Multinomial/TopK/TopP/MinP/
    /// TopKTopP, and the default greedy) → `tokens`. The side-channel samplers
    /// are computed host-side from the row's f32 logits: `RawLogits` →
    /// `logits_bytes` (native-endian f32), `Logprob`/`Logprobs` → `logprobs_*`
    /// (log-softmax at the labels), `Entropy` → `entropies` (−Σp·log p),
    /// `Dist` → `dists_*` (top-`num_tokens` of softmax(logits/T)). Each channel
    /// carries its own per-request CSR; channels with no rows stay empty so the
    /// runtime keeps its token-only fast path. (Embedding has no producer in the
    /// reference driver either → greedy-token fallback.)
    pub fn serve_forward(&self, dev: &Device, fr: &ForwardRequest) -> Result<ForwardResponse> {
        let r = fr.qo_indptr.len().saturating_sub(1);
        if fr.token_ids.is_empty() || r == 0 {
            return Ok(ForwardResponse { num_requests: r as u32, tokens_indptr: vec![0], ..Default::default() });
        }
        let tok_i32: Vec<i32> = fr.token_ids.iter().map(|&x| x as i32).collect();
        let positions: Vec<i32> = fr.position_ids.iter().map(|&x| x as i32).collect();
        let vocab = self.spec.vocab_size;
        let (out_logits, _argmax, t) = self.forward_logits_csr(dev, &tok_i32, &positions,
            &fr.qo_indptr, &fr.kv_page_indices, &fr.kv_page_indptr, &fr.kv_last_page_lens)?;

        // Plan the sampling rows + their wire Sampler, keeping per-request ranges.
        let per_request_sampling =
            fr.sampling_indptr.len() == r + 1 && !fr.sampling_indices.is_empty();
        let has_samplers = fr.sampler_indptr.len() == r + 1 && !fr.samplers.is_empty();
        let mut rows: Vec<i32> = Vec::new();
        let mut samplers: Vec<Option<&Sampler>> = Vec::new();
        let mut req_ranges: Vec<(usize, usize)> = Vec::with_capacity(r);
        for i in 0..r {
            let start = rows.len();
            if per_request_sampling {
                let (lo, hi) = (fr.sampling_indptr[i] as usize, fr.sampling_indptr[i + 1] as usize);
                let slo = if has_samplers { fr.sampler_indptr[i] as usize } else { 0 };
                for (j, s) in (lo..hi).enumerate() {
                    rows.push((fr.sampling_indices[s] as usize).min(t - 1) as i32);
                    samplers.push(if has_samplers { fr.samplers.get(slo + j) } else { None });
                }
            } else {
                let last = (fr.qo_indptr[i + 1] as usize).saturating_sub(1).min(t - 1);
                rows.push(last as i32);
                samplers.push(None);
            }
            req_ranges.push((start, rows.len()));
        }
        let s = rows.len();
        if s == 0 {
            return Ok(ForwardResponse { num_requests: r as u32, tokens_indptr: vec![0; r + 1], ..Default::default() });
        }

        // Gather all sampling-row logits once → compact [S, vocab] bf16.
        let ridx = upload(dev, &rows)?;
        let compact = dev.alloc(s * vocab * 2)?;
        dev.gather_bf16_rows(&out_logits, &ridx, &compact, s as i32, vocab as i32)?;

        // GPU-sample every row (non-token rows get a greedy placeholder, ignored).
        let params: Vec<SamplerRow> = samplers
            .iter()
            .map(|&sm| match sm {
                Some(s) if !is_token_sampler(s) => SamplerRow::greedy(),
                _ => SamplerRow::from_sampler(sm),
            })
            .collect();
        let token_ids_all = self.sample_compact(dev, &compact, &params, s, vocab)?;

        // Side-channel rows need the f32 logits on the host; only download then.
        let any_non_token = samplers.iter().any(|sm| matches!(sm, Some(s) if !is_token_sampler(s)));
        let host: Option<Vec<u16>> = if any_non_token {
            let mut b = vec![0u16; s * vocab];
            compact.download(&mut b)?;
            dev.sync()?;
            Some(b)
        } else {
            None
        };
        let row_f32 = |k: usize| -> Vec<f32> {
            host.as_ref().unwrap()[k * vocab..(k + 1) * vocab].iter().map(|&b| bf16_to_f32(b)).collect()
        };

        // Per-channel accumulators (each its own per-request CSR).
        let (mut tokens, mut tokens_indptr) = (Vec::<u32>::new(), vec![0u32]);
        let (mut d_req, mut d_kv) = (vec![0u32], vec![0u32]);
        let (mut d_ids, mut d_probs) = (Vec::<u32>::new(), Vec::<f32>::new());
        let (mut lg_req, mut lg_byte, mut lg_bytes) = (vec![0u32], vec![0u32], Vec::<u8>::new());
        let (mut lp_req, mut lp_val, mut lp_vals) = (vec![0u32], vec![0u32], Vec::<f32>::new());
        let (mut ent, mut ent_indptr) = (Vec::<f32>::new(), vec![0u32]);
        let (mut any_d, mut any_lg, mut any_lp, mut any_ent) = (false, false, false, false);

        for i in 0..r {
            let (start, end) = req_ranges[i];
            for k in start..end {
                match samplers[k] {
                    Some(Sampler::RawLogits) => {
                        for v in row_f32(k) {
                            lg_bytes.extend_from_slice(&v.to_ne_bytes());
                        }
                        lg_byte.push(lg_bytes.len() as u32);
                        any_lg = true;
                    }
                    Some(Sampler::Logprob { token_id }) => {
                        let lf = row_f32(k);
                        let lse = logsumexp(&lf);
                        lp_vals.push(lf.get(*token_id as usize).copied().unwrap_or(f32::NEG_INFINITY) - lse);
                        lp_val.push(lp_vals.len() as u32);
                        any_lp = true;
                    }
                    Some(Sampler::Logprobs { token_ids }) => {
                        let lf = row_f32(k);
                        let lse = logsumexp(&lf);
                        for &tid in token_ids {
                            lp_vals.push(lf.get(tid as usize).copied().unwrap_or(f32::NEG_INFINITY) - lse);
                        }
                        lp_val.push(lp_vals.len() as u32);
                        any_lp = true;
                    }
                    Some(Sampler::Entropy) => {
                        let lf = row_f32(k);
                        let lse = logsumexp(&lf);
                        // H = lse − Σ pᵢ·lᵢ, pᵢ = exp(lᵢ − lse).
                        let dot: f32 = lf.iter().map(|&l| (l - lse).exp() * l).sum();
                        ent.push(lse - dot);
                        any_ent = true;
                    }
                    Some(Sampler::Dist { temperature, num_tokens }) => {
                        let lf = row_f32(k);
                        let temp = if *temperature > 0.0 { *temperature } else { 1.0 };
                        let scaled: Vec<f32> = lf.iter().map(|&l| l / temp).collect();
                        let lse = logsumexp(&scaled);
                        let n = (*num_tokens as usize).min(vocab);
                        let mut idx: Vec<u32> = (0..vocab as u32).collect();
                        let cmp = |a: &u32, b: &u32| {
                            scaled[*b as usize]
                                .partial_cmp(&scaled[*a as usize])
                                .unwrap_or(std::cmp::Ordering::Equal)
                        };
                        if n > 0 && n < vocab {
                            idx.select_nth_unstable_by(n - 1, cmp);
                        }
                        idx.truncate(n);
                        idx.sort_unstable_by(cmp);
                        for id in idx {
                            d_ids.push(id);
                            d_probs.push((scaled[id as usize] - lse).exp());
                        }
                        d_kv.push(d_ids.len() as u32);
                        any_d = true;
                    }
                    // Token samplers, default greedy (None), and Embedding fallback.
                    _ => tokens.push(token_ids_all[k]),
                }
            }
            tokens_indptr.push(tokens.len() as u32);
            d_req.push((d_kv.len() - 1) as u32);
            lg_req.push((lg_byte.len() - 1) as u32);
            lp_req.push((lp_val.len() - 1) as u32);
            ent_indptr.push(ent.len() as u32);
        }

        let mut resp = ForwardResponse {
            num_requests: r as u32, tokens_indptr, tokens, ..Default::default()
        };
        if any_d {
            resp.dists_req_indptr = d_req;
            resp.dists_kv_indptr = d_kv;
            resp.dists_ids = d_ids;
            resp.dists_probs = d_probs;
        }
        if any_lg {
            resp.logits_req_indptr = lg_req;
            resp.logits_byte_indptr = lg_byte;
            resp.logits_bytes = lg_bytes;
        }
        if any_lp {
            resp.logprobs_req_indptr = lp_req;
            resp.logprobs_val_indptr = lp_val;
            resp.logprobs_values = lp_vals;
        }
        if any_ent {
            resp.entropies_indptr = ent_indptr;
            resp.entropies = ent;
        }
        Ok(resp)
    }

    /// Host KV swap slots advertised in the READY caps (`0` = no host swap).
    pub fn swap_pool_size(&self) -> usize {
        self.swap.borrow().slots
    }

    /// Serve a `CopyRequest`: move paged KV between physical page slots.
    /// `Kv` only (recurrent-state copy isn't needed by the transformer
    /// backends). Directions:
    ///   * `D2D` — GPU→GPU page copy (context fork / prefix share); all backends.
    ///   * `D2H`/`H2D`/`H2H` — swap a page to/from the host pool (memory-pressure
    ///     eviction/restore); paged-K/V backends only (MLA host-swap deferred).
    /// Copies each `(src, dst)` page across every layer in both K and V.
    /// Stream-ordered against the forwards, then synced (cold path).
    pub fn copy_pages(&self, dev: &Device, req: &pie_bridge::CopyRequest) -> Result<()> {
        use pie_bridge::{CopyDir, CopyResource};
        if req.resource != CopyResource::Kv {
            return Err(anyhow!(
                "copy_pages: resource {:?} unsupported (no recurrent-state cache)",
                req.resource
            ));
        }
        if req.srcs.len() != req.dsts.len() {
            return Err(anyhow!(
                "copy_pages: srcs/dsts length mismatch ({} vs {})",
                req.srcs.len(),
                req.dsts.len()
            ));
        }
        let (n_layers, ps, np, es) = (self.spec.num_layers, self.page_size, self.num_pages, 2usize);

        // GPU→GPU page copy (works for every backend, incl. MLA ckv/kpe).
        if req.dir == CopyDir::D2D {
            let copy_buf = |buf: &DeviceBuffer, width: usize| -> Result<()> {
                let page_bytes = ps * width * es;
                let layer_stride = np * page_bytes;
                for (&src, &dst) in req.srcs.iter().zip(req.dsts.iter()) {
                    let (src, dst) = (src as usize, dst as usize);
                    if src >= np || dst >= np {
                        return Err(anyhow!(
                            "copy_pages: page index out of range (src {src}, dst {dst}, num_pages {np})"
                        ));
                    }
                    if src == dst {
                        continue;
                    }
                    for l in 0..n_layers {
                        let base = l * layer_stride;
                        buf.copy_within_d2d(base + dst * page_bytes, base + src * page_bytes, page_bytes)?;
                    }
                }
                Ok(())
            };
            match &self.backend {
                Backend::Llama { kv_k, kv_v, .. }
                | Backend::Moe { kv_k, kv_v, .. }
                | Backend::Gemma { kv_k, kv_v, .. } => {
                    let width = self.spec.num_kv_heads * self.spec.head_dim;
                    copy_buf(kv_k, width)?;
                    copy_buf(kv_v, width)?;
                }
                Backend::Deepseek { ckv, kpe, .. } => {
                    let mla = self.spec.mla
                        .ok_or_else(|| anyhow!("copy_pages: MLA backend without mla dims"))?;
                    copy_buf(ckv, mla.kv_lora_rank)?;
                    copy_buf(kpe, mla.qk_rope_head_dim)?;
                }
            }
            dev.sync()?;
            return Ok(());
        }

        // Host swap (D2H/H2D/H2H) — paged-K/V backends + an allocated pool.
        let (kv_k, kv_v) = match &self.backend {
            Backend::Llama { kv_k, kv_v, .. }
            | Backend::Moe { kv_k, kv_v, .. }
            | Backend::Gemma { kv_k, kv_v, .. } => (kv_k, kv_v),
            Backend::Deepseek { .. } => {
                return Err(anyhow!("copy_pages: host KV swap not supported for MLA backends"));
            }
        };
        let mut pool = self.swap.borrow_mut();
        if pool.slots == 0 {
            return Err(anyhow!("copy_pages: host swap requested but swap_pool_size=0"));
        }
        let width = self.spec.num_kv_heads * self.spec.head_dim;
        let page_bytes = ps * width * es;
        let layer_stride = np * page_bytes; // device per-layer stride
        let slot_bytes = pool.slot_bytes; // host per-slot (== n_layers * page_bytes)
        for (&a, &b) in req.srcs.iter().zip(req.dsts.iter()) {
            let (a, b) = (a as usize, b as usize);
            match req.dir {
                // GPU page `a` → host slot `b`.
                CopyDir::D2H => {
                    if a >= np || b >= pool.slots {
                        return Err(anyhow!("copy D2H out of range (gpu {a}/{np}, slot {b}/{})", pool.slots));
                    }
                    for l in 0..n_layers {
                        let (dbase, hbase) = (l * layer_stride + a * page_bytes, b * slot_bytes + l * page_bytes);
                        kv_k.copy_to_host_at(&mut pool.k[hbase..hbase + page_bytes], dbase, page_bytes)?;
                        kv_v.copy_to_host_at(&mut pool.v[hbase..hbase + page_bytes], dbase, page_bytes)?;
                    }
                }
                // Host slot `a` → GPU page `b`.
                CopyDir::H2D => {
                    if b >= np || a >= pool.slots {
                        return Err(anyhow!("copy H2D out of range (slot {a}/{}, gpu {b}/{np})", pool.slots));
                    }
                    for l in 0..n_layers {
                        let (dbase, hbase) = (l * layer_stride + b * page_bytes, a * slot_bytes + l * page_bytes);
                        // split_at_mut not needed: distinct buffers k/v.
                        let (kh, vh) = (pool.k[hbase..hbase + page_bytes].to_vec(), pool.v[hbase..hbase + page_bytes].to_vec());
                        kv_k.copy_from_host_at(&kh, dbase, page_bytes)?;
                        kv_v.copy_from_host_at(&vh, dbase, page_bytes)?;
                    }
                }
                // Host slot `a` → host slot `b`.
                CopyDir::H2H => {
                    if a >= pool.slots || b >= pool.slots {
                        return Err(anyhow!("copy H2H out of range (slots {a},{b}/{})", pool.slots));
                    }
                    if a != b {
                        let (sa, sb) = (a * slot_bytes, b * slot_bytes);
                        pool.k.copy_within(sa..sa + slot_bytes, sb);
                        pool.v.copy_within(sa..sa + slot_bytes, sb);
                    }
                }
                CopyDir::D2D => unreachable!(),
            }
        }
        dev.sync()?;
        Ok(())
    }

    /// Single-request greedy generation: prefill `prompt`, then autoregressive
    /// decode `n_new − 1` more tokens (each a 1-token forward reading the cached
    /// KV). Returns the `n_new` generated tokens. This is the minimal decode
    /// loop; multi-request batched decode is the executor's further work.
    pub fn generate_greedy(&self, dev: &Device, prompt: &[i32], n_new: usize) -> Result<Vec<i32>> {
        if prompt.is_empty() || n_new == 0 {
            return Err(anyhow!("generate_greedy: empty prompt or n_new == 0"));
        }
        let mut out = Vec::with_capacity(n_new);
        let mut next = *self.step_single(dev, prompt, 0)?.last().unwrap(); // prefill
        out.push(next);
        let mut pre_len = prompt.len();
        for _ in 1..n_new {
            next = self.step_single(dev, &[next], pre_len)?[0]; // decode 1 token
            pre_len += 1;
            out.push(next);
        }
        Ok(out)
    }
}

fn upload<'a, T: Copy>(dev: &'a Device, data: &[T]) -> Result<DeviceBuffer<'a>> {
    let b = dev.alloc(std::mem::size_of_val(data))?;
    b.upload(data)?;
    Ok(b)
}

/// Allocate a zeroed bf16 device buffer of `n_elems` u16 elements (caches).
fn upload_zeros(dev: &Device, n_elems: usize) -> Result<DeviceBuffer<'_>> {
    let b = dev.alloc(n_elems * 2)?;
    b.upload(&vec![0u16; n_elems])?;
    Ok(b)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::{Device, f32_to_bf16};
    use serde_json::json;

    fn syn(seed: f32, n: usize, scale: f32) -> Vec<u16> {
        (0..n).map(|i| f32_to_bf16(((i as f32 + seed) * 0.1).sin() * scale)).collect()
    }

    fn write_safetensors(path: &Path, tensors: &[(String, Vec<usize>, Vec<u16>)]) {
        let mut data = Vec::new();
        let mut header = serde_json::Map::new();
        for (name, shape, vals) in tensors {
            let begin = data.len();
            for &v in vals {
                data.extend_from_slice(&v.to_le_bytes());
            }
            header.insert(
                name.clone(),
                json!({"dtype": "BF16", "shape": shape, "data_offsets": [begin, data.len()]}),
            );
        }
        let hjson = serde_json::to_vec(&header).unwrap();
        let mut out = (hjson.len() as u64).to_le_bytes().to_vec();
        out.extend_from_slice(&hjson);
        out.extend_from_slice(&data);
        std::fs::write(path, out).unwrap();
    }

    #[test]
    fn build_and_prefill_from_model_dir() {
        let dev = match Device::new(0) {
            Ok(d) => d,
            Err(e) => {
                eprintln!("skipping build_and_prefill_from_model_dir (no device): {e:#}");
                return;
            }
        };
        let (vocab, hidden, n_layers, nq, nkv, hd, inter) = (64usize, 32, 2, 2, 1, 16, 64);
        let (hq, hkv) = (nq * hd, nkv * hd);
        let dir = std::env::temp_dir().join(format!("cuda_new_build_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();

        // config.json (HF snake_case fields HfConfig deserializes).
        let config = json!({
            "model_type": "llama", "architectures": ["LlamaForCausalLM"],
            "hidden_size": hidden, "intermediate_size": inter, "num_hidden_layers": n_layers,
            "num_attention_heads": nq, "num_key_value_heads": nkv, "head_dim": hd,
            "vocab_size": vocab, "max_position_embeddings": 128,
            "rms_norm_eps": 1e-5, "rope_theta": 10000.0,
        });
        std::fs::write(dir.join("config.json"), serde_json::to_vec_pretty(&config).unwrap()).unwrap();

        // model.safetensors (synthetic llama tensors).
        let mut tensors: Vec<(String, Vec<usize>, Vec<u16>)> = vec![
            ("model.embed_tokens.weight".into(), vec![vocab, hidden], syn(1.0, vocab * hidden, 0.5)),
            ("model.norm.weight".into(), vec![hidden], syn(2.0, hidden, 1.0)),
            ("lm_head.weight".into(), vec![vocab, hidden], syn(3.0, vocab * hidden, 0.1)),
        ];
        for i in 0..n_layers {
            let s = (i * 100) as f32;
            let p = format!("model.layers.{i}");
            tensors.push((format!("{p}.input_layernorm.weight"), vec![hidden], syn(s + 4.0, hidden, 1.0)));
            tensors.push((format!("{p}.post_attention_layernorm.weight"), vec![hidden], syn(s + 5.0, hidden, 1.0)));
            tensors.push((format!("{p}.self_attn.q_proj.weight"), vec![hq, hidden], syn(s + 6.0, hq * hidden, 0.1)));
            tensors.push((format!("{p}.self_attn.k_proj.weight"), vec![hkv, hidden], syn(s + 7.0, hkv * hidden, 0.1)));
            tensors.push((format!("{p}.self_attn.v_proj.weight"), vec![hkv, hidden], syn(s + 8.0, hkv * hidden, 0.1)));
            tensors.push((format!("{p}.self_attn.o_proj.weight"), vec![hidden, hq], syn(s + 9.0, hidden * hq, 0.1)));
            tensors.push((format!("{p}.mlp.gate_proj.weight"), vec![inter, hidden], syn(s + 10.0, inter * hidden, 0.1)));
            tensors.push((format!("{p}.mlp.up_proj.weight"), vec![inter, hidden], syn(s + 11.0, inter * hidden, 0.1)));
            tensors.push((format!("{p}.mlp.down_proj.weight"), vec![hidden, inter], syn(s + 12.0, hidden * inter, 0.1)));
        }
        write_safetensors(&dir.join("model.safetensors"), &tensors);

        // Drive util off live memory so the planner sees a small positive
        // budget regardless of how occupied this (shared) GPU is.
        let (free, total) = dev.mem_info().unwrap();
        if free < 3 * 1024 * 1024 * 1024 {
            eprintln!("skipping build_and_prefill_from_model_dir (<3 GiB free)");
            let _ = std::fs::remove_dir_all(&dir);
            return;
        }
        let used = total - free;
        let util = (((used as f64) + 2.0e9) / total as f64).min(0.99);
        let cfg = BootConfig {
            device_ordinal: 0, memory_profile: Profile::Auto, gpu_mem_utilization: util,
            page_size: 16, num_kv_pages: 4, swap_pool_size: 2,
        };
        let model = build(&dev, &dir, &cfg).unwrap();
        assert_eq!(model.swap_pool_size(), 2, "swap pool advertised");

        // construction wired through: arch detect → spec, mem::plan, load.
        assert_eq!(model.spec.num_layers, n_layers);
        assert_eq!(model.spec.num_heads, nq);
        assert_eq!(model.spec.vocab_size, vocab);
        assert!(model.plan.num_pages > 0, "planner produced no KV pages");
        assert!(model.plan.max_tokens > 0, "planner produced no token capacity");

        // and the model runs a forward producing in-range tokens.
        let toks = model.prefill_greedy(&dev, &[5, 17, 2]).unwrap();
        assert_eq!(toks.len(), 3);
        for &tk in &toks {
            assert!((0..vocab as i32).contains(&tk), "token {tk} out of range");
        }

        // temperature sampling: temp=0 ≡ greedy argmax; fixed seed ≡ deterministic.
        let sampled0 = model.prefill_sample(&dev, &[5, 17, 2], 0.0, 42).unwrap();
        assert_eq!(sampled0, toks, "temp=0 sampling must equal greedy argmax");
        let s1 = model.prefill_sample(&dev, &[5, 17, 2], 0.8, 7).unwrap();
        let s2 = model.prefill_sample(&dev, &[5, 17, 2], 0.8, 7).unwrap();
        assert_eq!(s1, s2, "sampling must be deterministic for a fixed seed");
        for &tk in &s1 {
            assert!((0..vocab as i32).contains(&tk));
        }

        // Batched fire (B4 core): 2 requests of different lengths in ONE forward,
        // laid across separate KV pages, must each match running it alone (greedy).
        let (r0, r1) = (vec![5i32, 17, 2], vec![9i32, 1]);
        let batched = model.fire_batch(&dev, &[r0.clone(), r1.clone()]).unwrap();
        assert_eq!(batched.len(), 2);
        let single0 = *model.prefill_greedy(&dev, &r0).unwrap().last().unwrap();
        let single1 = *model.prefill_greedy(&dev, &r1).unwrap().last().unwrap();
        assert_eq!(batched[0], single0, "batched req0 next-token != single-run");
        assert_eq!(batched[1], single1, "batched req1 next-token != single-run");

        // Decode (A7): autoregressive generation must equal re-prefilling the
        // grown sequence at each step (incremental decode ≡ full prefill — the
        // KV-append writes the new token at the cached offset; attention reads
        // the full span).
        let prompt = vec![5i32, 17, 2];
        let generated = model.generate_greedy(&dev, &prompt, 6).unwrap();
        println!(
            "[e2e] dense-llama: arch={:?} n_layers={} | prompt={:?} -> generated={:?}",
            model.spec.id, model.spec.num_layers, prompt, generated
        );
        assert_eq!(generated.len(), 6);
        for &t in &generated { assert!((0..vocab as i32).contains(&t)); }
        let mut seq = prompt.clone();
        for k in 0..3 {
            let refnext = *model.prefill_greedy(&dev, &seq).unwrap().last().unwrap();
            assert_eq!(generated[k], refnext, "decode step {k}: incremental != re-prefill");
            seq.push(generated[k]);
        }

        // Multi-page prefill (A7 extension): a SINGLE request longer than
        // page_size (16) must span multiple KV pages in ONE forward — the shape
        // the runtime sends for a real prompt. Incremental decode crosses a page
        // boundary one token at a time; this crosses it within a single prefill,
        // a distinct kernel path. The last-position next-token must match the
        // token-by-token decode of the same sequence (decode ≡ prefill, now
        // across pages).
        let long: Vec<i32> = (0..28).map(|i| (i * 5) % vocab as i32).collect();
        let mp = model.fire_batch(&dev, &[long.clone()]).unwrap();
        assert_eq!(mp.len(), 1);
        assert!((0..vocab as i32).contains(&mp[0]), "multi-page token {} out of range", mp[0]);
        // Reference: prefill the first page (16 tokens) then decode the rest one
        // at a time; the final next-token must equal the single-shot prefill's.
        let ref_mp = {
            let head = &long[..16];
            let mut next = *model.step_single(&dev, head, 0).unwrap().last().unwrap();
            for (k, &tok) in long.iter().enumerate().skip(16) {
                let _ = next;
                next = model.step_single(&dev, &[tok], k).unwrap()[0];
            }
            next
        };
        assert_eq!(mp[0], ref_mp, "multi-page prefill last-token != incremental decode");

        // serve_forward (the wire ForwardRequest path the run_inproc serve loop
        // drives) must equal greedy prefill at the sampled position.
        let fr = pie_bridge::ForwardRequest {
            token_ids: vec![5, 17, 2],
            position_ids: vec![0, 1, 2],
            qo_indptr: vec![0, 3],
            kv_page_indices: vec![0],
            kv_page_indptr: vec![0, 1],
            kv_last_page_lens: vec![3],
            sampling_indices: vec![2],
            sampling_indptr: vec![0, 1],
            ..Default::default()
        };
        let resp = model.serve_forward(&dev, &fr).unwrap();
        assert_eq!(resp.num_requests, 1);
        assert_eq!(resp.tokens.len(), 1);
        let greedy_last = *model.prefill_greedy(&dev, &[5, 17, 2]).unwrap().last().unwrap();
        assert_eq!(resp.tokens[0] as i32, greedy_last, "serve_forward != greedy prefill");

        // Multi-request batched serve_forward: two requests of different lengths
        // in ONE fire, laid across separate pages, each greedy-sampled at its
        // last position — the response CSR must partition per request and each
        // token must match running that request alone (the serving-robustness
        // path: concurrent requests batched into one driver fire).
        let mr = pie_bridge::ForwardRequest {
            token_ids: vec![5, 17, 2, 9, 1],
            position_ids: vec![0, 1, 2, 0, 1],
            qo_indptr: vec![0, 3, 5],
            kv_page_indices: vec![0, 1],
            kv_page_indptr: vec![0, 1, 2],
            kv_last_page_lens: vec![3, 2],
            sampling_indices: vec![2, 4],
            sampling_indptr: vec![0, 1, 2],
            ..Default::default()
        };
        let mresp = model.serve_forward(&dev, &mr).unwrap();
        assert_eq!(mresp.num_requests, 2);
        assert_eq!(mresp.tokens_indptr, vec![0, 1, 2], "per-request token CSR");
        assert_eq!(mresp.tokens.len(), 2);
        let g0 = *model.prefill_greedy(&dev, &[5, 17, 2]).unwrap().last().unwrap();
        let g1 = *model.prefill_greedy(&dev, &[9, 1]).unwrap().last().unwrap();
        assert_eq!(mresp.tokens[0] as i32, g0, "batched req0 != single-run");
        assert_eq!(mresp.tokens[1] as i32, g1, "batched req1 != single-run");

        // Copy (D2D) page replication: prefill a context into page 0, copy
        // page 0 → page 3, then a 1-token decode reading the COPY must produce
        // the same next token as one reading the ORIGINAL — proving the KV was
        // replicated faithfully across every layer (context-fork / prefix-share).
        let prefill = pie_bridge::ForwardRequest {
            token_ids: vec![5, 17, 2, 9, 1],
            position_ids: vec![0, 1, 2, 3, 4],
            qo_indptr: vec![0, 5],
            kv_page_indices: vec![0],
            kv_page_indptr: vec![0, 1],
            kv_last_page_lens: vec![5],
            sampling_indices: vec![4],
            sampling_indptr: vec![0, 1],
            ..Default::default()
        };
        let _ = model.serve_forward(&dev, &prefill).unwrap(); // writes KV into page 0
        model
            .copy_pages(
                &dev,
                &pie_bridge::CopyRequest {
                    dir: pie_bridge::CopyDir::D2D,
                    srcs: vec![0],
                    dsts: vec![3],
                    resource: pie_bridge::CopyResource::Kv,
                },
            )
            .unwrap();
        let decode_at = |page: u32| {
            let req = pie_bridge::ForwardRequest {
                token_ids: vec![7],
                position_ids: vec![5],
                qo_indptr: vec![0, 1],
                kv_page_indices: vec![page],
                kv_page_indptr: vec![0, 1],
                kv_last_page_lens: vec![6],
                sampling_indices: vec![0],
                sampling_indptr: vec![0, 1],
                ..Default::default()
            };
            model.serve_forward(&dev, &req).unwrap().tokens[0]
        };
        let t_orig = decode_at(0);
        let t_copy = decode_at(3);
        assert_eq!(t_orig, t_copy, "decode over D2D-copied page != decode over original");

        // Host swap (D2H → H2D) round-trip: evict page 0 to host slot 0, restore
        // it into page 2, and a decode reading page 2 must match the original —
        // proving the eviction/restore path preserves KV across all layers.
        model.copy_pages(&dev, &pie_bridge::CopyRequest {
            dir: pie_bridge::CopyDir::D2H, srcs: vec![0], dsts: vec![0],
            resource: pie_bridge::CopyResource::Kv,
        }).unwrap();
        model.copy_pages(&dev, &pie_bridge::CopyRequest {
            dir: pie_bridge::CopyDir::H2D, srcs: vec![0], dsts: vec![2],
            resource: pie_bridge::CopyResource::Kv,
        }).unwrap();
        assert_eq!(decode_at(2), t_orig, "decode over swapped-out-and-back page != original");

        // Non-token sampler outputs (response side channels). Reference = the
        // greedy argmax token; these validate the channel math without a
        // separate logits oracle.
        let greedy = *model.prefill_greedy(&dev, &[5, 17, 2]).unwrap().last().unwrap();
        let base_fr = |smp: Sampler| pie_bridge::ForwardRequest {
            token_ids: vec![5, 17, 2],
            position_ids: vec![0, 1, 2],
            qo_indptr: vec![0, 3],
            kv_page_indices: vec![0],
            kv_page_indptr: vec![0, 1],
            kv_last_page_lens: vec![3],
            sampling_indices: vec![2],
            sampling_indptr: vec![0, 1],
            samplers: vec![smp],
            sampler_indptr: vec![0, 1],
            ..Default::default()
        };
        // Logprobs over the whole vocab: exp() must sum to 1 (it's a log-softmax)
        // and the max-logprob index is the greedy token.
        let lp = model
            .serve_forward(&dev, &base_fr(Sampler::Logprobs { token_ids: (0..vocab as u32).collect() }))
            .unwrap();
        // (Compare by VALUE, not argmax index: the near-flat synthetic logits
        // have ties, and the device argmax_bf16 / host max_by break them
        // differently — but the greedy token's value must equal the maximum.)
        assert!(lp.tokens.is_empty(), "logprobs row emits no token");
        assert_eq!(lp.logprobs_values.len(), vocab);
        let psum: f32 = lp.logprobs_values.iter().map(|&v| v.exp()).sum();
        assert!((psum - 1.0).abs() < 1e-2, "Σexp(logprob) = {psum}, expected ≈1 (log-softmax)");
        let lp_max = lp.logprobs_values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        assert!((lp.logprobs_values[greedy as usize] - lp_max).abs() < 1e-2,
            "greedy token is not (near-)max logprob");
        // RawLogits: vocab f32s; greedy token's logit is the max.
        let rl = model.serve_forward(&dev, &base_fr(Sampler::RawLogits)).unwrap();
        assert_eq!(rl.logits_bytes.len(), vocab * 4);
        let lf: Vec<f32> = rl.logits_bytes.chunks_exact(4)
            .map(|c| f32::from_ne_bytes(c.try_into().unwrap())).collect();
        let lf_max = lf.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        assert!((lf[greedy as usize] - lf_max).abs() < 1e-2, "greedy token not max raw-logit");
        // Entropy ∈ [0, ln V].
        let en = model.serve_forward(&dev, &base_fr(Sampler::Entropy)).unwrap();
        assert_eq!(en.entropies.len(), 1);
        assert!(en.entropies[0] >= -1e-3 && en.entropies[0] <= (vocab as f32).ln() + 1e-3,
            "entropy {} out of [0, ln V]", en.entropies[0]);
        // Dist top-k: 4 (id,prob) pairs, probs descending in (0,1], summing ≤ 1.
        let di = model.serve_forward(&dev, &base_fr(Sampler::Dist { temperature: 1.0, num_tokens: 4 })).unwrap();
        assert_eq!(di.dists_ids.len(), 4);
        assert_eq!(di.dists_probs.len(), 4);
        for w in di.dists_probs.windows(2) {
            assert!(w[0] >= w[1] - 1e-6, "dist probs not descending");
        }
        let dsum: f32 = di.dists_probs.iter().sum();
        assert!(dsum <= 1.0 + 1e-3, "top-4 dist probs sum {dsum} > 1");
        for &p in &di.dists_probs {
            assert!(p > 0.0 && p <= 1.0 + 1e-6, "dist prob {p} out of (0,1]");
        }

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// End-to-end text completion on a *real* model (Qwen2-0.5B): config.json →
    /// detect (qwen2 → LlamaLike + qkv-bias) → build → tokenize a prompt →
    /// generate_greedy → detokenize. This is the driver-level `pie run
    /// text-completion` proof: if the qkv-bias path is wrong the completion is
    /// garbage, so a coherent continuation is also the bias-correctness check.
    #[test]
    fn text_completion_qwen2_0_5b() {
        let snap = std::path::Path::new(
            "/root/.cache/huggingface/hub/models--Qwen--Qwen2-0.5B/snapshots/\
             91d2aff3f957f99e4c74c962f2f408dcc88a18d8",
        );
        if !snap.join("config.json").exists() || !snap.join("tokenizer.json").exists() {
            eprintln!("skipping text_completion_qwen2_0_5b (snapshot absent)");
            return;
        }
        let dev = match Device::new(0) {
            Ok(d) => d,
            Err(e) => {
                eprintln!("skipping text_completion_qwen2_0_5b (no device): {e:#}");
                return;
            }
        };
        let (free, total) = dev.mem_info().unwrap();
        if free < 3 * 1024 * 1024 * 1024 {
            eprintln!("skipping text_completion_qwen2_0_5b (<3 GiB free)");
            return;
        }
        let used = total - free;
        let util = (((used as f64) + 2.5e9) / total as f64).min(0.99);
        let cfg = BootConfig {
            device_ordinal: 0,
            memory_profile: Profile::Auto,
            gpu_mem_utilization: util,
            page_size: 16,
            num_kv_pages: 8,
            swap_pool_size: 0,
        };
        let model = build(&dev, snap, &cfg).unwrap();
        // qwen2 → LlamaLike + qkv-bias (the path under test).
        assert!(matches!(model.spec.id, PieArchId::LlamaLike));

        let tok = tokenizers::Tokenizer::from_file(snap.join("tokenizer.json"))
            .expect("load tokenizer.json");
        let prompt = "The capital of France is";
        let enc = tok.encode(prompt, false).expect("encode");
        let ids: Vec<i32> = enc.get_ids().iter().map(|&u| u as i32).collect();
        assert!(!ids.is_empty(), "tokenizer produced no ids");

        let n_new = 16;
        let generated = model.generate_greedy(&dev, &ids, n_new).unwrap();
        assert_eq!(generated.len(), n_new);
        let gen_u32: Vec<u32> = generated.iter().map(|&i| i as u32).collect();
        let completion = tok.decode(&gen_u32, false).expect("decode");

        println!("[text-completion] prompt    = {prompt:?}");
        println!("[text-completion] prompt ids = {ids:?}");
        println!("[text-completion] gen ids    = {generated:?}");
        println!("[text-completion] completion = {completion:?}");

        // Coherence sanity: the continuation must be non-empty printable text,
        // not a wall of replacement chars / control bytes (the failure mode of a
        // broken bias/qkv path on a real model).
        assert!(!completion.trim().is_empty(), "empty completion");
        let printable = completion
            .chars()
            .filter(|c| !c.is_control() && *c != '\u{FFFD}')
            .count();
        assert!(
            printable * 2 >= completion.chars().count(),
            "completion looks like garbage (mostly non-printable): {completion:?}"
        );
    }

    #[test]
    fn build_and_prefill_deepseek() {
        // Full B2 path for the headline frontier: config.json → detect
        // (DeepseekV4) → MLA/MoE spec → mem::plan → build_backend (LoadedDeepseek
        // + MLA latent cache) → prefill_greedy (deepseek_forward) → tokens.
        let dev = match Device::new(0) {
            Ok(d) => d,
            Err(e) => {
                eprintln!("skipping build_and_prefill_deepseek (no device): {e:#}");
                return;
            }
        };
        let (vocab, hidden, nh) = (32usize, 256, 2);
        let (q_lora, kv_lora, qk_nope, qk_rope, v_head) = (96usize, 128, 128, 64, 128);
        let (n_layers, first_k_dense, e, top_k, moe_inter, dense_inter) = (2usize, 1, 4, 2, 128, 128);
        let (qb_out, kva_out, kvb_out, ov) =
            (nh * (qk_nope + qk_rope), kv_lora + qk_rope, nh * (qk_nope + v_head), nh * v_head);
        let dir = std::env::temp_dir().join(format!("cuda_new_dsbuild_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();

        let config = json!({
            "model_type": "deepseek_v3", "architectures": ["DeepseekV3ForCausalLM"],
            "hidden_size": hidden, "intermediate_size": dense_inter, "num_hidden_layers": n_layers,
            "num_attention_heads": nh, "num_key_value_heads": nh,
            "vocab_size": vocab, "max_position_embeddings": 128,
            "rms_norm_eps": 1e-6, "rope_theta": 10000.0,
            "q_lora_rank": q_lora, "kv_lora_rank": kv_lora,
            "qk_nope_head_dim": qk_nope, "qk_rope_head_dim": qk_rope, "v_head_dim": v_head,
            "n_routed_experts": e, "num_experts_per_tok": top_k, "moe_intermediate_size": moe_inter,
            "first_k_dense_replace": first_k_dense,
        });
        std::fs::write(dir.join("config.json"), serde_json::to_vec_pretty(&config).unwrap()).unwrap();

        let mut tensors: Vec<(String, Vec<usize>, Vec<u16>)> = vec![
            ("model.embed_tokens.weight".into(), vec![vocab, hidden], syn(1.0, vocab * hidden, 0.5)),
            ("model.norm.weight".into(), vec![hidden], syn(2.0, hidden, 1.0)),
            ("lm_head.weight".into(), vec![vocab, hidden], syn(3.0, vocab * hidden, 0.1)),
        ];
        for i in 0..n_layers {
            let s = (i * 100) as f32;
            let p = format!("model.layers.{i}");
            let mut push = |suf: &str, shape: Vec<usize>, seed: f32, scale: f32| {
                let n: usize = shape.iter().product();
                tensors.push((format!("{p}.{suf}"), shape, syn(s + seed, n, scale)));
            };
            push("input_layernorm.weight", vec![hidden], 4.0, 1.0);
            push("self_attn.q_a_proj.weight", vec![q_lora, hidden], 5.0, 0.05);
            push("self_attn.q_a_layernorm.weight", vec![q_lora], 6.0, 1.0);
            push("self_attn.q_b_proj.weight", vec![qb_out, q_lora], 7.0, 0.05);
            push("self_attn.kv_a_proj_with_mqa.weight", vec![kva_out, hidden], 8.0, 0.05);
            push("self_attn.kv_a_layernorm.weight", vec![kv_lora], 9.0, 1.0);
            push("self_attn.kv_b_proj.weight", vec![kvb_out, kv_lora], 10.0, 0.05);
            push("self_attn.o_proj.weight", vec![hidden, ov], 11.0, 0.05);
            push("post_attention_layernorm.weight", vec![hidden], 12.0, 1.0);
            if i < first_k_dense {
                push("mlp.gate_proj.weight", vec![dense_inter, hidden], 13.0, 0.05);
                push("mlp.up_proj.weight", vec![dense_inter, hidden], 14.0, 0.05);
                push("mlp.down_proj.weight", vec![hidden, dense_inter], 15.0, 0.05);
            } else {
                push("mlp.gate.weight", vec![e, hidden], 16.0, 0.05);
                for x in 0..e {
                    let sx = 20.0 + x as f32 * 3.0;
                    let ep = format!("mlp.experts.{x}");
                    push(&format!("{ep}.gate_proj.weight"), vec![moe_inter, hidden], sx, 0.05);
                    push(&format!("{ep}.up_proj.weight"), vec![moe_inter, hidden], sx + 1.0, 0.05);
                    push(&format!("{ep}.down_proj.weight"), vec![hidden, moe_inter], sx + 2.0, 0.05);
                }
            }
        }
        write_safetensors(&dir.join("model.safetensors"), &tensors);

        let (free, total) = dev.mem_info().unwrap();
        if free < 3 * 1024 * 1024 * 1024 {
            eprintln!("skipping build_and_prefill_deepseek (<3 GiB free)");
            let _ = std::fs::remove_dir_all(&dir);
            return;
        }
        let util = ((((total - free) as f64) + 2.0e9) / total as f64).min(0.99);
        let cfg = BootConfig {
            device_ordinal: 0, memory_profile: Profile::Auto, gpu_mem_utilization: util,
            page_size: 16, num_kv_pages: 4, swap_pool_size: 0,
        };
        let model = build(&dev, &dir, &cfg).unwrap();

        // detect routed to DeepSeek-MLA with the MLA + MoE spec populated.
        assert_eq!(model.spec.id, crate::ffi::PieArchId::DeepseekV4);
        let mla = model.spec.mla.expect("deepseek spec must carry MLA dims");
        assert_eq!(mla.kv_lora_rank, kv_lora);
        assert_eq!(model.spec.moe_experts, e);

        // and the model runs a DeepSeek prefill producing in-range tokens.
        let toks = model.prefill_greedy(&dev, &[5, 17, 2]).unwrap();
        assert_eq!(toks.len(), 3);
        for &tk in &toks {
            assert!((0..vocab as i32).contains(&tk), "token {tk} out of range");
        }

        // e2e generation on the GPU + MLA decode≡re-prefill (validates the MLA
        // latent-cache write at the cached offset across decode steps).
        let prompt = vec![5i32, 17, 2];
        let generated = model.generate_greedy(&dev, &prompt, 6).unwrap();
        println!(
            "[e2e] DeepSeek-MLA: arch={:?} n_layers={} kv_lora={} experts={} | prompt={:?} -> generated={:?}",
            model.spec.id, model.spec.num_layers, mla.kv_lora_rank, model.spec.moe_experts,
            prompt, generated
        );
        assert_eq!(generated.len(), 6);
        for &t in &generated { assert!((0..vocab as i32).contains(&t)); }
        let mut seq = prompt.clone();
        for k in 0..3 {
            let refnext = *model.prefill_greedy(&dev, &seq).unwrap().last().unwrap();
            assert_eq!(generated[k], refnext, "DeepSeek decode step {k}: incremental != re-prefill");
            seq.push(generated[k]);
        }

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn build_and_prefill_moe() {
        // 2nd frontier through the same builder path: config.json → detect
        // (Qwen3_5Moe) → MoE spec → build_backend (LoadedMoe + paged KV) →
        // prefill (moe_forward). (qk-norm is a deferred refinement; this checks
        // the route/load/run pipeline, not Qwen3-MoE numerical fidelity.)
        let dev = match Device::new(0) {
            Ok(d) => d,
            Err(e) => {
                eprintln!("skipping build_and_prefill_moe (no device): {e:#}");
                return;
            }
        };
        let (vocab, hidden, nq, nkv, hd) = (32usize, 64, 4, 2, 16);
        let (n_layers, e, top_k, moe_inter) = (2usize, 4, 2, 128);
        let (hq, hkv) = (nq * hd, nkv * hd);
        let dir = std::env::temp_dir().join(format!("cuda_new_moebuild_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();

        let config = json!({
            "model_type": "qwen3_moe", "architectures": ["Qwen3MoeForCausalLM"],
            "hidden_size": hidden, "intermediate_size": moe_inter, "num_hidden_layers": n_layers,
            "num_attention_heads": nq, "num_key_value_heads": nkv, "head_dim": hd,
            "vocab_size": vocab, "max_position_embeddings": 128,
            "rms_norm_eps": 1e-6, "rope_theta": 10000.0,
            "n_routed_experts": e, "num_experts_per_tok": top_k, "moe_intermediate_size": moe_inter,
        });
        std::fs::write(dir.join("config.json"), serde_json::to_vec_pretty(&config).unwrap()).unwrap();

        let mut tensors: Vec<(String, Vec<usize>, Vec<u16>)> = vec![
            ("model.embed_tokens.weight".into(), vec![vocab, hidden], syn(1.0, vocab * hidden, 0.5)),
            ("model.norm.weight".into(), vec![hidden], syn(2.0, hidden, 1.0)),
            ("lm_head.weight".into(), vec![vocab, hidden], syn(3.0, vocab * hidden, 0.1)),
        ];
        for i in 0..n_layers {
            let s = (i * 100) as f32;
            let p = format!("model.layers.{i}");
            let mut push = |suf: &str, shape: Vec<usize>, seed: f32, scale: f32| {
                let n: usize = shape.iter().product();
                tensors.push((format!("{p}.{suf}"), shape, syn(s + seed, n, scale)));
            };
            push("input_layernorm.weight", vec![hidden], 4.0, 1.0);
            push("self_attn.q_proj.weight", vec![hq, hidden], 5.0, 0.1);
            push("self_attn.k_proj.weight", vec![hkv, hidden], 6.0, 0.1);
            push("self_attn.v_proj.weight", vec![hkv, hidden], 7.0, 0.1);
            push("self_attn.o_proj.weight", vec![hidden, hq], 8.0, 0.1);
            push("post_attention_layernorm.weight", vec![hidden], 9.0, 1.0);
            push("mlp.gate.weight", vec![e, hidden], 10.0, 0.1);
            for x in 0..e {
                let sx = 20.0 + x as f32 * 3.0;
                let ep = format!("mlp.experts.{x}");
                push(&format!("{ep}.gate_proj.weight"), vec![moe_inter, hidden], sx, 0.1);
                push(&format!("{ep}.up_proj.weight"), vec![moe_inter, hidden], sx + 1.0, 0.1);
                push(&format!("{ep}.down_proj.weight"), vec![hidden, moe_inter], sx + 2.0, 0.1);
            }
        }
        write_safetensors(&dir.join("model.safetensors"), &tensors);

        let (free, total) = dev.mem_info().unwrap();
        if free < 3 * 1024 * 1024 * 1024 {
            eprintln!("skipping build_and_prefill_moe (<3 GiB free)");
            let _ = std::fs::remove_dir_all(&dir);
            return;
        }
        let util = ((((total - free) as f64) + 2.0e9) / total as f64).min(0.99);
        let cfg = BootConfig {
            device_ordinal: 0, memory_profile: Profile::Auto, gpu_mem_utilization: util,
            page_size: 16, num_kv_pages: 4, swap_pool_size: 0,
        };
        let model = build(&dev, &dir, &cfg).unwrap();

        assert_eq!(model.spec.id, crate::ffi::PieArchId::Qwen3_5Moe);
        assert_eq!(model.spec.moe_experts, e);
        let toks = model.prefill_greedy(&dev, &[5, 17, 2]).unwrap();
        assert_eq!(toks.len(), 3);
        for &tk in &toks {
            assert!((0..vocab as i32).contains(&tk), "token {tk} out of range");
        }

        // e2e generation on the GPU + MoE decode≡re-prefill.
        let prompt = vec![5i32, 17, 2];
        let generated = model.generate_greedy(&dev, &prompt, 6).unwrap();
        println!(
            "[e2e] Qwen3.5-MoE: arch={:?} n_layers={} experts={} top_k={} | prompt={:?} -> generated={:?}",
            model.spec.id, model.spec.num_layers, model.spec.moe_experts,
            model.spec.num_experts_per_tok, prompt, generated
        );
        assert_eq!(generated.len(), 6);
        for &t in &generated { assert!((0..vocab as i32).contains(&t)); }
        let mut seq = prompt.clone();
        for k in 0..3 {
            let refnext = *model.prefill_greedy(&dev, &seq).unwrap().last().unwrap();
            assert_eq!(generated[k], refnext, "MoE decode step {k}: incremental != re-prefill");
            seq.push(generated[k]);
        }

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn build_and_prefill_gemma() {
        // 4th backend through the builder: config.json → detect (Gemma4) →
        // sandwich-norm spec (sliding/full per-layer window + soft-caps + √H
        // embed scale) → build_backend (LoadedGemma + paged KV) → prefill
        // (gemma_forward). qk-norm + AltUp are deferred (A1), so this checks the
        // route/load/run pipeline + the per-layer window plumbing.
        let dev = match Device::new(0) {
            Ok(d) => d,
            Err(e) => {
                eprintln!("skipping build_and_prefill_gemma (no device): {e:#}");
                return;
            }
        };
        let (vocab, hidden, nq, nkv, hd, inter, n_layers) = (32usize, 64, 4, 2, 16, 128, 2);
        let (hq, hkv) = (nq * hd, nkv * hd);
        let dir = std::env::temp_dir().join(format!("cuda_new_gemmabuild_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();

        let config = json!({
            "model_type": "gemma3", "architectures": ["Gemma3ForCausalLM"],
            "hidden_size": hidden, "intermediate_size": inter, "num_hidden_layers": n_layers,
            "num_attention_heads": nq, "num_key_value_heads": nkv, "head_dim": hd,
            "vocab_size": vocab, "max_position_embeddings": 128,
            "rms_norm_eps": 1e-6, "rope_theta": 10000.0,
            // window larger than the test seq ⇒ no truncation, but exercises the
            // per-layer sliding/full window_left plumbing (layer 0 sliding, 1 full).
            "sliding_window": 4096, "sliding_window_pattern": 2,
            "attn_logit_softcapping": 50.0, "final_logit_softcapping": 30.0,
            "tie_word_embeddings": true,
        });
        std::fs::write(dir.join("config.json"), serde_json::to_vec_pretty(&config).unwrap()).unwrap();

        // Gemma weights: 4 sandwich norms/layer, qkvo, gate/up/down, tied lm_head.
        let mut tensors: Vec<(String, Vec<usize>, Vec<u16>)> = vec![
            ("model.embed_tokens.weight".into(), vec![vocab, hidden], syn(1.0, vocab * hidden, 0.2)),
            ("model.norm.weight".into(), vec![hidden], syn(2.0, hidden, 1.0)),
        ];
        for i in 0..n_layers {
            let s = (i * 100) as f32;
            let p = format!("model.layers.{i}");
            let mut push = |suf: &str, shape: Vec<usize>, seed: f32, scale: f32| {
                let n: usize = shape.iter().product();
                tensors.push((format!("{p}.{suf}"), shape, syn(s + seed, n, scale)));
            };
            push("input_layernorm.weight", vec![hidden], 3.0, 1.0);
            push("post_attention_layernorm.weight", vec![hidden], 4.0, 1.0);
            push("pre_feedforward_layernorm.weight", vec![hidden], 5.0, 1.0);
            push("post_feedforward_layernorm.weight", vec![hidden], 6.0, 1.0);
            push("self_attn.q_proj.weight", vec![hq, hidden], 7.0, 0.1);
            push("self_attn.k_proj.weight", vec![hkv, hidden], 8.0, 0.1);
            push("self_attn.v_proj.weight", vec![hkv, hidden], 9.0, 0.1);
            push("self_attn.o_proj.weight", vec![hidden, hq], 10.0, 0.1);
            push("mlp.gate_proj.weight", vec![inter, hidden], 11.0, 0.1);
            push("mlp.up_proj.weight", vec![inter, hidden], 12.0, 0.1);
            push("mlp.down_proj.weight", vec![hidden, inter], 13.0, 0.1);
        }
        write_safetensors(&dir.join("model.safetensors"), &tensors);

        let (free, total) = dev.mem_info().unwrap();
        if free < 3 * 1024 * 1024 * 1024 {
            eprintln!("skipping build_and_prefill_gemma (<3 GiB free)");
            let _ = std::fs::remove_dir_all(&dir);
            return;
        }
        let util = ((((total - free) as f64) + 2.0e9) / total as f64).min(0.99);
        let cfg = BootConfig {
            device_ordinal: 0, memory_profile: Profile::Auto, gpu_mem_utilization: util,
            page_size: 16, num_kv_pages: 4, swap_pool_size: 0,
        };
        let model = build(&dev, &dir, &cfg).unwrap();
        assert_eq!(model.spec.id, crate::ffi::PieArchId::Gemma4);

        let toks = model.prefill_greedy(&dev, &[5, 17, 2]).unwrap();
        assert_eq!(toks.len(), 3);
        for &tk in &toks {
            assert!((0..vocab as i32).contains(&tk), "gemma token {tk} out of range");
        }
        // e2e generation + decode≡re-prefill (the cached-KV invariant on Gemma).
        let prompt = vec![5i32, 17, 2];
        let generated = model.generate_greedy(&dev, &prompt, 6).unwrap();
        println!(
            "[e2e] Gemma: arch={:?} n_layers={} | prompt={:?} -> generated={:?}",
            model.spec.id, model.spec.num_layers, prompt, generated
        );
        assert_eq!(generated.len(), 6);
        for &t in &generated { assert!((0..vocab as i32).contains(&t)); }
        let mut seq = prompt.clone();
        for k in 0..3 {
            let refnext = *model.prefill_greedy(&dev, &seq).unwrap().last().unwrap();
            assert_eq!(generated[k], refnext, "Gemma decode step {k}: incremental != re-prefill");
            seq.push(generated[k]);
        }

        let _ = std::fs::remove_dir_all(&dir);
    }
}
