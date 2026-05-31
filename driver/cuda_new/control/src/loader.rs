//! Minimal safetensors weight loader — phase 2. Replaces the synthetic
//! weights with real on-disk tensors: reads a safetensors checkpoint,
//! uploads each llama tensor to a device buffer (via the existing memory
//! ABI), and assembles the weight set `Device::llama_forward_bf16` consumes.
//!
//! This is the fat-Rust-control-plane realization of weight binding: the
//! control plane owns loading and uploads directly; there is no separate
//! C++ `pie_weights_bind` entry on this path.
//!
//! Slice scope (documented): BF16/F16/F32 tensors (F16/F32 cast to bf16 on
//! upload); `std::fs::read` (not mmap); HF llama naming. The phase-2
//! completion reuses/extends `driver/weight_loader` (`pie-weight-loader`)
//! for quantization, tensor-parallel sharding, and GGUF.

use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result, bail};

use crate::device::{
    DeepseekForwardDims, DeepseekLayerWeights, Device, DeviceBuffer, GemmaForwardDims,
    GemmaLayerWeights, LlamaForwardDims, LlamaLayerWeights, MlaLayerWeights, MoeForwardDims,
    MoeLayerWeights, Workspace,
};

/// llama dimensions. `head_dim` / `rms_eps` / `rope_theta` come from the
/// config; the rest are inferred from tensor shapes.
#[derive(Clone, Debug)]
pub struct LlamaConfig {
    pub vocab: usize,
    pub hidden: usize,
    pub n_layers: usize,
    pub n_q_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub intermediate: usize,
    pub rms_eps: f32,
    pub rope_theta: f32,
}

struct TensorMeta {
    dtype: String,
    shape: Vec<usize>,
    shard: usize, // index into `Safetensors.shards`
    begin: usize, // byte offset within that shard's data region
    end: usize,
}

/// One or more safetensors files presented as a single tensor namespace. A
/// single `model.safetensors` is one shard; a sharded checkpoint
/// (`model.safetensors.index.json` + `model-0000N-of-...` files) is N shards
/// with a `weight_map` routing each tensor to its shard.
struct Safetensors {
    shards: Vec<Vec<u8>>,
    data_start: Vec<usize>, // per-shard `8 + header_len`
    header: HashMap<String, TensorMeta>,
}

/// Parse one safetensors file's header into (header, data_start, bytes), with
/// every tensor tagged `shard = shard_idx`.
fn parse_one(path: &Path, shard_idx: usize) -> Result<(HashMap<String, TensorMeta>, usize, Vec<u8>)> {
    let bytes = std::fs::read(path).with_context(|| format!("reading {path:?}"))?;
    if bytes.len() < 8 {
        bail!("safetensors {path:?} too small");
    }
    let hlen = u64::from_le_bytes(bytes[0..8].try_into().unwrap()) as usize;
    if 8 + hlen > bytes.len() {
        bail!("safetensors {path:?} header ({hlen} B) overruns file");
    }
    let hjson: serde_json::Value =
        serde_json::from_slice(&bytes[8..8 + hlen]).context("parsing safetensors header")?;
    let obj = hjson.as_object().context("safetensors header is not an object")?;
    let mut header = HashMap::new();
    for (name, v) in obj {
        if name == "__metadata__" {
            continue;
        }
        let dtype = v["dtype"].as_str().context("tensor dtype")?.to_string();
        let shape = v["shape"]
            .as_array()
            .context("tensor shape")?
            .iter()
            .map(|x| x.as_u64().unwrap_or(0) as usize)
            .collect();
        let offs = v["data_offsets"].as_array().context("data_offsets")?;
        let begin = offs[0].as_u64().context("offset[0]")? as usize;
        let end = offs[1].as_u64().context("offset[1]")? as usize;
        header.insert(name.clone(), TensorMeta { dtype, shape, shard: shard_idx, begin, end });
    }
    Ok((header, 8 + hlen, bytes))
}

/// Open a model's weights from `path`, which may be: a single `.safetensors`
/// file; a directory with `model.safetensors`; or a directory with a sharded
/// `model.safetensors.index.json`. (HF checkpoints are commonly sharded.)
fn read_safetensors(path: &Path) -> Result<Safetensors> {
    if path.is_dir() {
        let index = path.join("model.safetensors.index.json");
        if index.exists() {
            return read_sharded(path, &index);
        }
        let single = path.join("model.safetensors");
        if single.exists() {
            let (header, ds, bytes) = parse_one(&single, 0)?;
            return Ok(Safetensors { shards: vec![bytes], data_start: vec![ds], header });
        }
        bail!("no model.safetensors[.index.json] under {path:?}");
    }
    let (header, ds, bytes) = parse_one(path, 0)?;
    Ok(Safetensors { shards: vec![bytes], data_start: vec![ds], header })
}

/// Load a sharded checkpoint: read the `weight_map`, load each distinct shard
/// file once, and route each tensor to its shard.
fn read_sharded(dir: &Path, index_path: &Path) -> Result<Safetensors> {
    let idx: serde_json::Value =
        serde_json::from_slice(&std::fs::read(index_path)?).context("parsing index.json")?;
    let wm = idx["weight_map"].as_object().context("index.json weight_map")?;
    // Distinct shard filenames, stable order.
    let mut files: Vec<String> =
        wm.values().filter_map(|v| v.as_str().map(String::from)).collect();
    files.sort();
    files.dedup();
    let file_idx: HashMap<&str, usize> =
        files.iter().enumerate().map(|(i, f)| (f.as_str(), i)).collect();

    let mut shards = Vec::with_capacity(files.len());
    let mut data_start = Vec::with_capacity(files.len());
    let mut per_shard: Vec<HashMap<String, TensorMeta>> = Vec::with_capacity(files.len());
    for (i, f) in files.iter().enumerate() {
        let (h, ds, bytes) = parse_one(&dir.join(f), i)?;
        shards.push(bytes);
        data_start.push(ds);
        per_shard.push(h);
    }
    // Combined namespace: each tensor's meta comes from its shard's header.
    let mut header = HashMap::new();
    for (name, vfile) in wm {
        let f = vfile.as_str().context("shard filename")?;
        let si = *file_idx.get(f).with_context(|| format!("unknown shard {f}"))?;
        let m = per_shard[si]
            .get(name)
            .with_context(|| format!("tensor `{name}` not found in shard {f}"))?;
        header.insert(
            name.clone(),
            TensorMeta { dtype: m.dtype.clone(), shape: m.shape.clone(), shard: si, begin: m.begin, end: m.end },
        );
    }
    Ok(Safetensors { shards, data_start, header })
}

impl Safetensors {
    fn meta(&self, name: &str) -> Result<&TensorMeta> {
        self.header.get(name).with_context(|| format!("missing tensor `{name}`"))
    }

    fn has(&self, name: &str) -> bool {
        self.header.contains_key(name)
    }

    /// Raw bytes, dtype string, and element count for a tensor.
    fn tensor_raw(&self, name: &str) -> Result<(&[u8], &str, usize)> {
        let m = self.meta(name)?;
        let numel: usize = m.shape.iter().product();
        let base = self.data_start[m.shard];
        let slice = &self.shards[m.shard][base + m.begin..base + m.end];
        Ok((slice, m.dtype.as_str(), numel))
    }
}

/// Upload a tensor to a bf16 device buffer, casting from F16/F32 on the fly
/// (most HF checkpoints are BF16 or F16; some ship F32 scales/norms).
fn upload_tensor<'a>(dev: &'a Device, st: &Safetensors, name: &str) -> Result<DeviceBuffer<'a>> {
    let (raw, dtype, numel) = st.tensor_raw(name)?;
    match dtype {
        "BF16" => {
            let buf = dev.alloc(raw.len())?;
            buf.upload(raw)?;
            Ok(buf)
        }
        // F16/F32: stage the raw bytes, cast to bf16. `dev.sync()` before the
        // staging buffer is freed so the cast has finished reading it.
        "F16" => {
            let tmp = dev.alloc(raw.len())?;
            tmp.upload(raw)?;
            let out = dev.alloc(numel * 2)?;
            dev.cast_fp16_to_bf16(&tmp, &out, numel)?;
            dev.sync()?;
            Ok(out)
        }
        "F32" => {
            let tmp = dev.alloc(raw.len())?;
            tmp.upload(raw)?;
            let out = dev.alloc(numel * 2)?;
            dev.cast_fp32_to_bf16(&tmp, &out, numel)?;
            dev.sync()?;
            Ok(out)
        }
        other => bail!("tensor `{name}`: dtype {other} unsupported (BF16/F16/F32 only)"),
    }
}

/// Read a tensor as host bf16 (`u16`) elements. BF16-only — the host-side
/// weight surgery (DeepSeek `kv_b_proj` split/transpose, MoE expert stacking)
/// needs to permute elements before upload, and most frontier checkpoints are
/// BF16. F16/F32 surgery is deferred (those can be host-decoded later).
fn tensor_bf16_host(st: &Safetensors, name: &str) -> Result<Vec<u16>> {
    let (raw, dtype, _numel) = st.tensor_raw(name)?;
    match dtype {
        "BF16" => Ok(raw
            .chunks_exact(2)
            .map(|c| u16::from_le_bytes([c[0], c[1]]))
            .collect()),
        other => bail!("tensor `{name}`: dtype {other} unsupported for host-side surgery (BF16 only)"),
    }
}

/// Split DeepSeek's fused `kv_b_proj.weight` into the per-head absorbed up-proj
/// weights `mla_block` expects. Input `kv_b` is row-major bf16
/// `[nh*(qk_nope+v_head), kv_lora]`: per head, the first `qk_nope` rows are the
/// K-NoPE up-proj `Kup[d][l]` ([qk_nope, kv_lora]) and the next `v_head` rows are
/// the V up-proj `Vup[vd][l]` ([v_head, kv_lora]). The block uses each as a
/// `gemm act@wᵀ` weight, so (see mla_block.cu):
///   * `W_uk[h]` is `[kv_lora, qk_nope]` with `W_uk[h][l][d] = Kup[d][l]` → **transpose**.
///   * `W_uv[h]` is `[v_head, kv_lora]` with `W_uv[h][vd][l] = Vup[vd][l]` → **direct**.
/// Returns `(W_uk [nh,kv_lora,qk_nope], W_uv [nh,v_head,kv_lora])` host bf16.
fn split_kv_b_proj(
    kv_b: &[u16],
    nh: usize,
    qk_nope: usize,
    v_head: usize,
    kv_lora: usize,
) -> (Vec<u16>, Vec<u16>) {
    let per_head_out = qk_nope + v_head;
    let mut w_uk = vec![0u16; nh * kv_lora * qk_nope];
    let mut w_uv = vec![0u16; nh * v_head * kv_lora];
    for h in 0..nh {
        let base = h * per_head_out * kv_lora;
        // K-up [qk_nope, kv_lora] (row d, col l) → W_uk[h][l][d] (transpose).
        for d in 0..qk_nope {
            for l in 0..kv_lora {
                w_uk[(h * kv_lora + l) * qk_nope + d] = kv_b[base + d * kv_lora + l];
            }
        }
        // V-up [v_head, kv_lora] (row vd, col l) → W_uv[h][vd][l] (direct copy).
        let vbase = base + qk_nope * kv_lora;
        for vd in 0..v_head {
            for l in 0..kv_lora {
                w_uv[(h * v_head + vd) * kv_lora + l] = kv_b[vbase + vd * kv_lora + l];
            }
        }
    }
    (w_uk, w_uv)
}

/// Stack per-expert MoE FFN weights into the contiguous tensors the dense MoE
/// block (`moe_mlp_block_bf16`) consumes. Per expert: `gate`/`up` are
/// `[inter, hidden]`, `down` is `[hidden, inter]`. Produces
/// `wgu [E, 2*inter, hidden]` (gate rows then up rows per expert) and
/// `wdown [E, hidden, inter]`. All host bf16.
fn stack_moe_experts(
    gates: &[Vec<u16>],
    ups: &[Vec<u16>],
    downs: &[Vec<u16>],
    inter: usize,
    hidden: usize,
) -> (Vec<u16>, Vec<u16>) {
    let e = gates.len();
    let mut wgu = vec![0u16; e * 2 * inter * hidden];
    let mut wdown = vec![0u16; e * hidden * inter];
    for x in 0..e {
        let gu = x * 2 * inter * hidden;
        wgu[gu..gu + inter * hidden].copy_from_slice(&gates[x]);
        wgu[gu + inter * hidden..gu + 2 * inter * hidden].copy_from_slice(&ups[x]);
        let d = x * hidden * inter;
        wdown[d..d + hidden * inter].copy_from_slice(&downs[x]);
    }
    (wgu, wdown)
}

/// Upload a host bf16 (`u16`) slice into a fresh device buffer. Used for the
/// surgery outputs (split/transposed `W_uk`, stacked MoE `wgu`/`wdown`).
fn upload_bf16_host<'a>(dev: &'a Device, vals: &[u16]) -> Result<DeviceBuffer<'a>> {
    let buf = dev.alloc(vals.len() * 2)?;
    buf.upload(vals)?;
    Ok(buf)
}

/// A llama model resident on the device — owns one bf16 buffer per weight.
pub struct LoadedLlama<'a> {
    pub cfg: LlamaConfig,
    embed: DeviceBuffer<'a>,
    final_norm: DeviceBuffer<'a>,
    lm_head: DeviceBuffer<'a>,
    attn_norm: Vec<DeviceBuffer<'a>>,
    ffn_norm: Vec<DeviceBuffer<'a>>,
    wq: Vec<DeviceBuffer<'a>>,
    wk: Vec<DeviceBuffer<'a>>,
    wv: Vec<DeviceBuffer<'a>>,
    wo: Vec<DeviceBuffer<'a>>,
    wg: Vec<DeviceBuffer<'a>>,
    wu: Vec<DeviceBuffer<'a>>,
    wd: Vec<DeviceBuffer<'a>>,
    qn: Vec<Option<DeviceBuffer<'a>>>, // per-head q-norm (Qwen3); empty/None for Llama
    kn: Vec<Option<DeviceBuffer<'a>>>,
    qb: Vec<Option<DeviceBuffer<'a>>>, // q/k/v proj biases (Qwen2); None for Llama/Qwen3
    kb: Vec<Option<DeviceBuffer<'a>>>,
    vb: Vec<Option<DeviceBuffer<'a>>>,
}

impl<'a> LoadedLlama<'a> {
    /// Load a safetensors checkpoint onto `dev`. Dimensions are inferred from
    /// the tensor shapes given `head_dim`; `rms_eps` / `rope_theta` come from
    /// the model config.
    pub fn load(
        dev: &'a Device,
        path: &Path,
        head_dim: usize,
        rms_eps: f32,
        rope_theta: f32,
    ) -> Result<Self> {
        let st = read_safetensors(path)?;

        // Infer dims from shapes (HF llama layout).
        let embed_m = st.meta("model.embed_tokens.weight")?;
        if embed_m.shape.len() != 2 {
            bail!("embed_tokens.weight is not 2-D");
        }
        let (vocab, hidden) = (embed_m.shape[0], embed_m.shape[1]);
        let n_layers = (0usize..)
            .take_while(|i| {
                st.header.contains_key(&format!("model.layers.{i}.input_layernorm.weight"))
            })
            .count();
        if n_layers == 0 {
            bail!("no decoder layers found (model.layers.0.* missing)");
        }
        if head_dim == 0 {
            bail!("head_dim must be > 0");
        }
        let n_q_heads = st.meta("model.layers.0.self_attn.q_proj.weight")?.shape[0] / head_dim;
        let n_kv_heads = st.meta("model.layers.0.self_attn.k_proj.weight")?.shape[0] / head_dim;
        let intermediate = st.meta("model.layers.0.mlp.gate_proj.weight")?.shape[0];
        let cfg = LlamaConfig {
            vocab, hidden, n_layers, n_q_heads, n_kv_heads, head_dim, intermediate, rms_eps, rope_theta,
        };

        let (mut attn_norm, mut ffn_norm) = (Vec::new(), Vec::new());
        let (mut wq, mut wk, mut wv, mut wo) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        let (mut wg, mut wu, mut wd) = (Vec::new(), Vec::new(), Vec::new());
        let (mut qn, mut kn) = (Vec::new(), Vec::new());
        let (mut qb, mut kb, mut vb) = (Vec::new(), Vec::new(), Vec::new());
        for i in 0..n_layers {
            let p = format!("model.layers.{i}");
            attn_norm.push(upload_tensor(dev, &st, &format!("{p}.input_layernorm.weight"))?);
            wq.push(upload_tensor(dev, &st, &format!("{p}.self_attn.q_proj.weight"))?);
            wk.push(upload_tensor(dev, &st, &format!("{p}.self_attn.k_proj.weight"))?);
            wv.push(upload_tensor(dev, &st, &format!("{p}.self_attn.v_proj.weight"))?);
            wo.push(upload_tensor(dev, &st, &format!("{p}.self_attn.o_proj.weight"))?);
            ffn_norm.push(upload_tensor(dev, &st, &format!("{p}.post_attention_layernorm.weight"))?);
            wg.push(upload_tensor(dev, &st, &format!("{p}.mlp.gate_proj.weight"))?);
            wu.push(upload_tensor(dev, &st, &format!("{p}.mlp.up_proj.weight"))?);
            wd.push(upload_tensor(dev, &st, &format!("{p}.mlp.down_proj.weight"))?);
            // Qwen3 / OLMo-3 per-head q/k-norm (optional — absent for Llama).
            let qn_name = format!("{p}.self_attn.q_norm.weight");
            let kn_name = format!("{p}.self_attn.k_norm.weight");
            qn.push(if st.has(&qn_name) { Some(upload_tensor(dev, &st, &qn_name)?) } else { None });
            kn.push(if st.has(&kn_name) { Some(upload_tensor(dev, &st, &kn_name)?) } else { None });
            // Qwen2 q/k/v projection biases (optional — absent for Llama/Qwen3).
            let load_opt = |dev: &'a Device, st: &Safetensors, name: String| -> Result<Option<DeviceBuffer<'a>>> {
                if st.has(&name) { Ok(Some(upload_tensor(dev, st, &name)?)) } else { Ok(None) }
            };
            qb.push(load_opt(dev, &st, format!("{p}.self_attn.q_proj.bias"))?);
            kb.push(load_opt(dev, &st, format!("{p}.self_attn.k_proj.bias"))?);
            vb.push(load_opt(dev, &st, format!("{p}.self_attn.v_proj.bias"))?);
        }
        let embed = upload_tensor(dev, &st, "model.embed_tokens.weight")?;
        let final_norm = upload_tensor(dev, &st, "model.norm.weight")?;
        // Tied embeddings (Qwen2/3 small, Gemma): no lm_head.weight → reuse embed.
        let lm_head_name =
            if st.has("lm_head.weight") { "lm_head.weight" } else { "model.embed_tokens.weight" };
        let lm_head = upload_tensor(dev, &st, lm_head_name)?;
        dev.sync()?; // F16/F32 casts complete before staging buffers drop

        Ok(LoadedLlama {
            cfg, embed, final_norm, lm_head, attn_norm, ffn_norm, wq, wk, wv, wo, wg, wu, wd, qn, kn,
            qb, kb, vb,
        })
    }

    fn layer_weights(&self) -> Vec<LlamaLayerWeights<'_>> {
        (0..self.cfg.n_layers)
            .map(|l| LlamaLayerWeights {
                attn_norm: &self.attn_norm[l], wq: &self.wq[l], wk: &self.wk[l], wv: &self.wv[l],
                wo: &self.wo[l], ffn_norm: &self.ffn_norm[l], w_gate: &self.wg[l],
                w_up: &self.wu[l], w_down: &self.wd[l], q_norm: self.qn[l].as_ref(),
                k_norm: self.kn[l].as_ref(), q_bias: self.qb[l].as_ref(),
                k_bias: self.kb[l].as_ref(), v_bias: self.vb[l].as_ref(),
            })
            .collect()
    }

    /// Run the full forward on the loaded weights. KV / page lists and I/O
    /// buffers are caller-provided (sized for `num_tokens`).
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self, dev: &Device, ws: &Workspace, token_ids: &DeviceBuffer, positions: &DeviceBuffer,
        kv_k: &DeviceBuffer, kv_v: &DeviceBuffer, qo_indptr: &DeviceBuffer,
        kv_page_indices: &DeviceBuffer, kv_page_indptr: &DeviceBuffer,
        kv_last_page_lens: &DeviceBuffer, out_logits: &DeviceBuffer, out_tokens: &DeviceBuffer,
        num_tokens: i32, num_requests: i32, page_size: i32, num_kv_pages: i32,
    ) -> Result<()> {
        let c = &self.cfg;
        let layers = self.layer_weights();
        let dims = LlamaForwardDims {
            num_tokens, num_requests, hidden_size: c.hidden as i32, n_q_heads: c.n_q_heads as i32,
            n_kv_heads: c.n_kv_heads as i32, head_dim: c.head_dim as i32,
            intermediate: c.intermediate as i32, page_size, num_kv_pages, vocab: c.vocab as i32,
            rms_eps: c.rms_eps, rope_theta: c.rope_theta,
        };
        dev.llama_forward_bf16(
            ws, token_ids, &self.embed, &layers, &self.final_norm, &self.lm_head, positions,
            kv_k, kv_v, qo_indptr, kv_page_indices, kv_page_indptr, kv_last_page_lens, out_logits,
            out_tokens, &dims,
        )
    }
}

/// Gemma-3/4 runtime dims the builder fills from the `ArchSpec` (structural
/// dims — vocab/hidden/heads/intermediate/n_layers — are inferred from the
/// checkpoint shapes, like Llama). `window_left` is per-layer (`-1` = full
/// causal, else the sliding left-context); empty = full across the stack.
/// `*_softcap` of `0.0` disables that soft-cap. Embed scale = √hidden.
#[derive(Clone)]
pub struct GemmaConfig {
    pub head_dim: usize,
    pub rms_eps: f32,
    pub rope_theta: f32,
    pub attn_logit_softcap: f32,
    pub final_logit_softcap: f32,
    pub window_left: Vec<i32>,
}

/// A Gemma-3/4 model resident on the device — the "sandwich norm" transformer
/// (input / post-attn / pre-ffn / post-ffn norms per layer) with √hidden embed
/// scaling, attn+final logit soft-caps, and sliding/full per-layer windows.
/// Matches `gemma_forward.cu`'s supported subset (per-head qk-norm and AltUp
/// are deferred there — A1 — so they're intentionally not loaded/applied).
pub struct LoadedGemma<'a> {
    cfg: GemmaConfig,
    vocab: usize,
    hidden: usize,
    n_layers: usize,
    n_q_heads: usize,
    n_kv_heads: usize,
    intermediate: usize,
    embed_scale: f32,
    embed: DeviceBuffer<'a>,
    final_norm: DeviceBuffer<'a>,
    lm_head: DeviceBuffer<'a>,
    input_ln: Vec<DeviceBuffer<'a>>,
    post_attn_ln: Vec<DeviceBuffer<'a>>,
    pre_ffn_ln: Vec<DeviceBuffer<'a>>,
    post_ffn_ln: Vec<DeviceBuffer<'a>>,
    wq: Vec<DeviceBuffer<'a>>,
    wk: Vec<DeviceBuffer<'a>>,
    wv: Vec<DeviceBuffer<'a>>,
    wo: Vec<DeviceBuffer<'a>>,
    wg: Vec<DeviceBuffer<'a>>,
    wu: Vec<DeviceBuffer<'a>>,
    wd: Vec<DeviceBuffer<'a>>,
}

impl<'a> LoadedGemma<'a> {
    pub fn load(dev: &'a Device, path: &Path, cfg: GemmaConfig) -> Result<Self> {
        let st = read_safetensors(path)?;
        let embed_m = st.meta("model.embed_tokens.weight")?;
        if embed_m.shape.len() != 2 {
            bail!("embed_tokens.weight is not 2-D");
        }
        let (vocab, hidden) = (embed_m.shape[0], embed_m.shape[1]);
        let n_layers = (0usize..)
            .take_while(|i| {
                st.header.contains_key(&format!("model.layers.{i}.input_layernorm.weight"))
            })
            .count();
        if n_layers == 0 {
            bail!("no decoder layers found (model.layers.0.* missing)");
        }
        let hd = cfg.head_dim;
        if hd == 0 {
            bail!("gemma head_dim must be > 0");
        }
        let n_q_heads = st.meta("model.layers.0.self_attn.q_proj.weight")?.shape[0] / hd;
        let n_kv_heads = st.meta("model.layers.0.self_attn.k_proj.weight")?.shape[0] / hd;
        let intermediate = st.meta("model.layers.0.mlp.gate_proj.weight")?.shape[0];

        let (mut input_ln, mut post_attn_ln) = (Vec::new(), Vec::new());
        let (mut pre_ffn_ln, mut post_ffn_ln) = (Vec::new(), Vec::new());
        let (mut wq, mut wk, mut wv, mut wo) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        let (mut wg, mut wu, mut wd) = (Vec::new(), Vec::new(), Vec::new());
        for i in 0..n_layers {
            let p = format!("model.layers.{i}");
            input_ln.push(upload_tensor(dev, &st, &format!("{p}.input_layernorm.weight"))?);
            post_attn_ln.push(upload_tensor(dev, &st, &format!("{p}.post_attention_layernorm.weight"))?);
            pre_ffn_ln.push(upload_tensor(dev, &st, &format!("{p}.pre_feedforward_layernorm.weight"))?);
            post_ffn_ln.push(upload_tensor(dev, &st, &format!("{p}.post_feedforward_layernorm.weight"))?);
            wq.push(upload_tensor(dev, &st, &format!("{p}.self_attn.q_proj.weight"))?);
            wk.push(upload_tensor(dev, &st, &format!("{p}.self_attn.k_proj.weight"))?);
            wv.push(upload_tensor(dev, &st, &format!("{p}.self_attn.v_proj.weight"))?);
            wo.push(upload_tensor(dev, &st, &format!("{p}.self_attn.o_proj.weight"))?);
            wg.push(upload_tensor(dev, &st, &format!("{p}.mlp.gate_proj.weight"))?);
            wu.push(upload_tensor(dev, &st, &format!("{p}.mlp.up_proj.weight"))?);
            wd.push(upload_tensor(dev, &st, &format!("{p}.mlp.down_proj.weight"))?);
        }
        let embed = upload_tensor(dev, &st, "model.embed_tokens.weight")?;
        let final_norm = upload_tensor(dev, &st, "model.norm.weight")?;
        // Gemma ties the lm_head to the embedding table.
        let lm_head_name =
            if st.has("lm_head.weight") { "lm_head.weight" } else { "model.embed_tokens.weight" };
        let lm_head = upload_tensor(dev, &st, lm_head_name)?;
        dev.sync()?;

        let embed_scale = (hidden as f32).sqrt();
        Ok(LoadedGemma {
            cfg, vocab, hidden, n_layers, n_q_heads, n_kv_heads, intermediate, embed_scale,
            embed, final_norm, lm_head, input_ln, post_attn_ln, pre_ffn_ln, post_ffn_ln,
            wq, wk, wv, wo, wg, wu, wd,
        })
    }

    fn layer_weights(&self) -> Vec<GemmaLayerWeights<'_>> {
        (0..self.n_layers)
            .map(|l| GemmaLayerWeights {
                input_ln: &self.input_ln[l], post_attn_ln: &self.post_attn_ln[l],
                pre_ffn_ln: &self.pre_ffn_ln[l], post_ffn_ln: &self.post_ffn_ln[l],
                wq: &self.wq[l], wk: &self.wk[l], wv: &self.wv[l], wo: &self.wo[l],
                w_gate: &self.wg[l], w_up: &self.wu[l], w_down: &self.wd[l],
            })
            .collect()
    }

    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self, dev: &Device, token_ids: &DeviceBuffer, positions: &DeviceBuffer,
        kv_k: &DeviceBuffer, kv_v: &DeviceBuffer, qo_indptr: &DeviceBuffer,
        kv_page_indices: &DeviceBuffer, kv_page_indptr: &DeviceBuffer,
        kv_last_page_lens: &DeviceBuffer, out_logits: &DeviceBuffer, out_tokens: &DeviceBuffer,
        num_tokens: i32, num_requests: i32, page_size: i32, num_kv_pages: i32,
    ) -> Result<()> {
        let layers = self.layer_weights();
        let dims = GemmaForwardDims {
            hidden: self.hidden as i32, n_q_heads: self.n_q_heads as i32,
            n_kv_heads: self.n_kv_heads as i32, head_dim: self.cfg.head_dim as i32,
            intermediate: self.intermediate as i32, vocab: self.vocab as i32, page_size,
            num_pages: num_kv_pages, window_left: self.cfg.window_left.clone(), window_left_all: -1,
            attn_logit_softcap: self.cfg.attn_logit_softcap,
            final_logit_softcap: self.cfg.final_logit_softcap, embed_scale: self.embed_scale,
            rms_eps: self.cfg.rms_eps, rope_theta: self.cfg.rope_theta,
            // qk-norm + AltUp are deferred in gemma_forward.cu (A1).
            qk_norm: 0, altup_num_inputs: 1,
        };
        dev.gemma_forward_bf16(
            token_ids, &self.embed, &layers, &self.final_norm, &self.lm_head, positions,
            kv_k, kv_v, qo_indptr, kv_page_indices, kv_page_indptr, kv_last_page_lens,
            out_logits, out_tokens, num_tokens, num_requests, &dims,
        )
    }
}

/// DeepSeek-V3/V4 dimensions. Supplied by the builder from `arch::detect`'s
/// `ArchSpec` (the loader trusts these rather than re-inferring the MLA/MoE
/// dims from shapes). `head_dim` for MLA is `qk_nope + qk_rope`.
#[derive(Clone, Debug)]
pub struct DeepseekConfig {
    pub vocab: usize,
    pub hidden: usize,
    pub n_layers: usize,
    pub num_heads: usize,
    pub q_lora_rank: usize,
    pub kv_lora_rank: usize,
    pub qk_nope_head_dim: usize,
    pub qk_rope_head_dim: usize,
    pub v_head_dim: usize,
    pub first_k_dense: usize,
    pub dense_inter: usize,
    pub moe_inter: usize,
    pub num_experts: usize,
    pub top_k: usize,
    pub rms_eps: f32,
    pub rope_theta: f32,
}

/// A DeepSeek model resident on the device. Owns one bf16 buffer per weight,
/// including the surgery outputs (`w_uk` transposed, `wgu`/`wdown` stacked).
/// Layers `[0, first_k_dense)` carry the dense FFN buffers; the rest the MoE
/// buffers (the unused set per layer is `None`).
pub struct LoadedDeepseek<'a> {
    pub cfg: DeepseekConfig,
    embed: DeviceBuffer<'a>,
    final_norm: DeviceBuffer<'a>,
    lm_head: DeviceBuffer<'a>,
    attn_norm: Vec<DeviceBuffer<'a>>,
    w_q_a: Vec<DeviceBuffer<'a>>,
    q_a_ln: Vec<DeviceBuffer<'a>>,
    w_q_b: Vec<DeviceBuffer<'a>>,
    w_kv_a: Vec<DeviceBuffer<'a>>,
    kv_a_ln: Vec<DeviceBuffer<'a>>,
    w_uk: Vec<DeviceBuffer<'a>>,
    w_uv: Vec<DeviceBuffer<'a>>,
    w_o: Vec<DeviceBuffer<'a>>,
    ffn_norm: Vec<DeviceBuffer<'a>>,
    dgate: Vec<Option<DeviceBuffer<'a>>>,
    dup: Vec<Option<DeviceBuffer<'a>>>,
    ddown: Vec<Option<DeviceBuffer<'a>>>,
    router_w: Vec<Option<DeviceBuffer<'a>>>,
    wgu: Vec<Option<DeviceBuffer<'a>>>,
    wdown: Vec<Option<DeviceBuffer<'a>>>,
}

impl<'a> LoadedDeepseek<'a> {
    /// Load a DeepSeek safetensors checkpoint onto `dev` per `cfg`. Performs the
    /// host-side weight surgery: splits `kv_b_proj` into per-head `W_uk`
    /// (transposed) / `W_uv`, and stacks the per-expert MoE FFN weights into
    /// `wgu`/`wdown`. (HF DeepSeek tensor naming; BF16 surgery tensors.)
    pub fn load(dev: &'a Device, path: &Path, cfg: DeepseekConfig) -> Result<Self> {
        let st = read_safetensors(path)?;
        let nh = cfg.num_heads;
        let mut attn_norm = Vec::new();
        let (mut w_q_a, mut q_a_ln, mut w_q_b) = (Vec::new(), Vec::new(), Vec::new());
        let (mut w_kv_a, mut kv_a_ln, mut w_uk, mut w_uv, mut w_o) =
            (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
        let mut ffn_norm = Vec::new();
        let (mut dgate, mut dup, mut ddown) = (Vec::new(), Vec::new(), Vec::new());
        let (mut router_w, mut wgu, mut wdown) = (Vec::new(), Vec::new(), Vec::new());

        for i in 0..cfg.n_layers {
            let p = format!("model.layers.{i}");
            attn_norm.push(upload_tensor(dev, &st, &format!("{p}.input_layernorm.weight"))?);
            w_q_a.push(upload_tensor(dev, &st, &format!("{p}.self_attn.q_a_proj.weight"))?);
            q_a_ln.push(upload_tensor(dev, &st, &format!("{p}.self_attn.q_a_layernorm.weight"))?);
            w_q_b.push(upload_tensor(dev, &st, &format!("{p}.self_attn.q_b_proj.weight"))?);
            w_kv_a.push(upload_tensor(dev, &st, &format!("{p}.self_attn.kv_a_proj_with_mqa.weight"))?);
            kv_a_ln.push(upload_tensor(dev, &st, &format!("{p}.self_attn.kv_a_layernorm.weight"))?);
            // kv_b_proj surgery: split per head into W_uk (transposed) + W_uv.
            let kv_b = tensor_bf16_host(&st, &format!("{p}.self_attn.kv_b_proj.weight"))?;
            let (uk, uv) = split_kv_b_proj(
                &kv_b, nh, cfg.qk_nope_head_dim, cfg.v_head_dim, cfg.kv_lora_rank);
            w_uk.push(upload_bf16_host(dev, &uk)?);
            w_uv.push(upload_bf16_host(dev, &uv)?);
            w_o.push(upload_tensor(dev, &st, &format!("{p}.self_attn.o_proj.weight"))?);
            ffn_norm.push(upload_tensor(dev, &st, &format!("{p}.post_attention_layernorm.weight"))?);

            if i < cfg.first_k_dense {
                dgate.push(Some(upload_tensor(dev, &st, &format!("{p}.mlp.gate_proj.weight"))?));
                dup.push(Some(upload_tensor(dev, &st, &format!("{p}.mlp.up_proj.weight"))?));
                ddown.push(Some(upload_tensor(dev, &st, &format!("{p}.mlp.down_proj.weight"))?));
                router_w.push(None);
                wgu.push(None);
                wdown.push(None);
            } else {
                dgate.push(None);
                dup.push(None);
                ddown.push(None);
                router_w.push(Some(upload_tensor(dev, &st, &format!("{p}.mlp.gate.weight"))?));
                let (mut gs, mut us, mut ds) = (Vec::new(), Vec::new(), Vec::new());
                for e in 0..cfg.num_experts {
                    let ep = format!("{p}.mlp.experts.{e}");
                    gs.push(tensor_bf16_host(&st, &format!("{ep}.gate_proj.weight"))?);
                    us.push(tensor_bf16_host(&st, &format!("{ep}.up_proj.weight"))?);
                    ds.push(tensor_bf16_host(&st, &format!("{ep}.down_proj.weight"))?);
                }
                let (gu, dn) = stack_moe_experts(&gs, &us, &ds, cfg.moe_inter, cfg.hidden);
                wgu.push(Some(upload_bf16_host(dev, &gu)?));
                wdown.push(Some(upload_bf16_host(dev, &dn)?));
            }
        }
        let embed = upload_tensor(dev, &st, "model.embed_tokens.weight")?;
        let final_norm = upload_tensor(dev, &st, "model.norm.weight")?;
        let lm_head = upload_tensor(dev, &st, "lm_head.weight")?;
        dev.sync()?; // surgery uploads complete before staging Vecs drop

        Ok(LoadedDeepseek {
            cfg, embed, final_norm, lm_head, attn_norm, w_q_a, q_a_ln, w_q_b, w_kv_a, kv_a_ln,
            w_uk, w_uv, w_o, ffn_norm, dgate, dup, ddown, router_w, wgu, wdown,
        })
    }

    fn layer_weights(&self) -> Vec<DeepseekLayerWeights<'_>> {
        (0..self.cfg.n_layers)
            .map(|l| DeepseekLayerWeights {
                attn: MlaLayerWeights {
                    attn_norm: &self.attn_norm[l], w_q_a: &self.w_q_a[l], q_a_ln: &self.q_a_ln[l],
                    w_q_b: &self.w_q_b[l], w_kv_a: &self.w_kv_a[l], kv_a_ln: &self.kv_a_ln[l],
                    w_uk: &self.w_uk[l], w_uv: &self.w_uv[l], w_o: &self.w_o[l],
                },
                ffn_norm: &self.ffn_norm[l],
                w_gate: self.dgate[l].as_ref(), w_up: self.dup[l].as_ref(),
                w_down: self.ddown[l].as_ref(), router_w: self.router_w[l].as_ref(),
                wgu: self.wgu[l].as_ref(), wdown: self.wdown[l].as_ref(),
            })
            .collect()
    }

    /// Run the full DeepSeek forward on the loaded weights. Per-layer MLA cache
    /// (ckv/kpe) + page lists + I/O buffers are caller-provided.
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self, dev: &Device, token_ids: &DeviceBuffer, positions: &DeviceBuffer,
        ckv_pages: &DeviceBuffer, kpe_pages: &DeviceBuffer, qo_indptr: &DeviceBuffer,
        kv_page_indices: &DeviceBuffer, kv_page_indptr: &DeviceBuffer,
        kv_last_page_lens: &DeviceBuffer, out_logits: &DeviceBuffer, out_tokens: &DeviceBuffer,
        num_tokens: i32, num_requests: i32, page_size: i32, num_pages: i32,
    ) -> Result<()> {
        let c = &self.cfg;
        let layers = self.layer_weights();
        let dims = DeepseekForwardDims {
            first_k_dense: c.first_k_dense as i32, hidden: c.hidden as i32,
            num_heads: c.num_heads as i32, q_lora_rank: c.q_lora_rank as i32,
            kv_lora_rank: c.kv_lora_rank as i32, qk_nope_head_dim: c.qk_nope_head_dim as i32,
            qk_rope_head_dim: c.qk_rope_head_dim as i32, v_head_dim: c.v_head_dim as i32,
            dense_inter: c.dense_inter as i32, moe_inter: c.moe_inter as i32,
            num_experts: c.num_experts as i32, top_k: c.top_k as i32, vocab: c.vocab as i32,
            page_size, num_pages,
            sm_scale: 1.0 / ((c.kv_lora_rank + c.qk_rope_head_dim) as f32).sqrt(),
            rms_eps: c.rms_eps, rope_theta: c.rope_theta,
        };
        dev.deepseek_forward_bf16(
            token_ids, &self.embed, &layers, &self.final_norm, &self.lm_head, positions,
            ckv_pages, kpe_pages, qo_indptr, kv_page_indices, kv_page_indptr, kv_last_page_lens,
            out_logits, out_tokens, num_tokens, num_requests, &dims,
        )
    }
}

/// Dense-MoE (Qwen3.5-MoE / GPT-OSS) dimensions, supplied by the builder from
/// `arch::detect`'s `ArchSpec`. Standard GQA attention + a top-K routed MoE FFN.
#[derive(Clone, Debug)]
pub struct MoeConfig {
    pub vocab: usize,
    pub hidden: usize,
    pub n_layers: usize,
    pub n_q_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub moe_inter: usize,
    pub num_experts: usize,
    pub top_k: usize,
    pub rms_eps: f32,
    pub rope_theta: f32,
}

/// A dense-MoE model resident on the device — standard attention weights plus
/// the stacked per-expert MoE FFN (`wgu`/`wdown` via `stack_moe_experts`).
/// (First pass: no q/k-norm or shared expert — refinements `moe_forward` defers.)
pub struct LoadedMoe<'a> {
    pub cfg: MoeConfig,
    embed: DeviceBuffer<'a>,
    final_norm: DeviceBuffer<'a>,
    lm_head: DeviceBuffer<'a>,
    attn_norm: Vec<DeviceBuffer<'a>>,
    wq: Vec<DeviceBuffer<'a>>,
    wk: Vec<DeviceBuffer<'a>>,
    wv: Vec<DeviceBuffer<'a>>,
    wo: Vec<DeviceBuffer<'a>>,
    ffn_norm: Vec<DeviceBuffer<'a>>,
    router_w: Vec<DeviceBuffer<'a>>,
    wgu: Vec<DeviceBuffer<'a>>,
    wdown: Vec<DeviceBuffer<'a>>,
}

impl<'a> LoadedMoe<'a> {
    /// Load a Qwen-MoE-style safetensors checkpoint per `cfg`. Standard
    /// q/k/v/o attention + `mlp.gate` router + `mlp.experts.{e}.{gate,up,down}`
    /// stacked into `wgu`/`wdown`.
    pub fn load(dev: &'a Device, path: &Path, cfg: MoeConfig) -> Result<Self> {
        let st = read_safetensors(path)?;
        let (mut attn_norm, mut ffn_norm) = (Vec::new(), Vec::new());
        let (mut wq, mut wk, mut wv, mut wo) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        let (mut router_w, mut wgu, mut wdown) = (Vec::new(), Vec::new(), Vec::new());
        for i in 0..cfg.n_layers {
            let p = format!("model.layers.{i}");
            attn_norm.push(upload_tensor(dev, &st, &format!("{p}.input_layernorm.weight"))?);
            wq.push(upload_tensor(dev, &st, &format!("{p}.self_attn.q_proj.weight"))?);
            wk.push(upload_tensor(dev, &st, &format!("{p}.self_attn.k_proj.weight"))?);
            wv.push(upload_tensor(dev, &st, &format!("{p}.self_attn.v_proj.weight"))?);
            wo.push(upload_tensor(dev, &st, &format!("{p}.self_attn.o_proj.weight"))?);
            ffn_norm.push(upload_tensor(dev, &st, &format!("{p}.post_attention_layernorm.weight"))?);
            router_w.push(upload_tensor(dev, &st, &format!("{p}.mlp.gate.weight"))?);
            let (mut gs, mut us, mut ds) = (Vec::new(), Vec::new(), Vec::new());
            for e in 0..cfg.num_experts {
                let ep = format!("{p}.mlp.experts.{e}");
                gs.push(tensor_bf16_host(&st, &format!("{ep}.gate_proj.weight"))?);
                us.push(tensor_bf16_host(&st, &format!("{ep}.up_proj.weight"))?);
                ds.push(tensor_bf16_host(&st, &format!("{ep}.down_proj.weight"))?);
            }
            let (gu, dn) = stack_moe_experts(&gs, &us, &ds, cfg.moe_inter, cfg.hidden);
            wgu.push(upload_bf16_host(dev, &gu)?);
            wdown.push(upload_bf16_host(dev, &dn)?);
        }
        let embed = upload_tensor(dev, &st, "model.embed_tokens.weight")?;
        let final_norm = upload_tensor(dev, &st, "model.norm.weight")?;
        let lm_head = upload_tensor(dev, &st, "lm_head.weight")?;
        dev.sync()?;
        Ok(LoadedMoe {
            cfg, embed, final_norm, lm_head, attn_norm, wq, wk, wv, wo, ffn_norm, router_w, wgu,
            wdown,
        })
    }

    fn layer_weights(&self) -> Vec<MoeLayerWeights<'_>> {
        (0..self.cfg.n_layers)
            .map(|l| MoeLayerWeights {
                attn_norm: &self.attn_norm[l], wq: &self.wq[l], wk: &self.wk[l], wv: &self.wv[l],
                wo: &self.wo[l], ffn_norm: &self.ffn_norm[l], router_w: &self.router_w[l],
                wgu: &self.wgu[l], wdown: &self.wdown[l],
            })
            .collect()
    }

    /// Run the full dense-MoE forward on the loaded weights.
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self, dev: &Device, token_ids: &DeviceBuffer, positions: &DeviceBuffer,
        kv_k: &DeviceBuffer, kv_v: &DeviceBuffer, qo_indptr: &DeviceBuffer,
        kv_page_indices: &DeviceBuffer, kv_page_indptr: &DeviceBuffer,
        kv_last_page_lens: &DeviceBuffer, out_logits: &DeviceBuffer, out_tokens: &DeviceBuffer,
        num_tokens: i32, num_requests: i32, page_size: i32, num_kv_pages: i32,
    ) -> Result<()> {
        let c = &self.cfg;
        let layers = self.layer_weights();
        let dims = MoeForwardDims {
            hidden_size: c.hidden as i32, n_q_heads: c.n_q_heads as i32,
            n_kv_heads: c.n_kv_heads as i32, head_dim: c.head_dim as i32,
            intermediate: c.moe_inter as i32, num_experts: c.num_experts as i32,
            top_k: c.top_k as i32, vocab: c.vocab as i32, page_size, rms_eps: c.rms_eps,
            rope_theta: c.rope_theta,
        };
        dev.moe_forward_bf16(
            token_ids, &self.embed, &layers, &self.final_norm, &self.lm_head, positions, kv_k,
            kv_v, qo_indptr, kv_page_indices, kv_page_indptr, kv_last_page_lens, out_logits,
            out_tokens, num_tokens, num_requests, num_kv_pages, &dims,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::{f32_to_bf16, f32_to_bf16_rne};
    use serde_json::json;

    // Upload a typed host slice into a fresh device buffer.
    fn up_dev<'a, T: Copy>(dev: &'a Device, data: &[T]) -> DeviceBuffer<'a> {
        let b = dev.alloc(std::mem::size_of_val(data)).unwrap();
        b.upload(data).unwrap();
        b
    }

    fn syn(seed: f32, n: usize, scale: f32) -> Vec<u16> {
        (0..n).map(|i| f32_to_bf16(((i as f32 + seed) * 0.1).sin() * scale)).collect()
    }

    // Write a minimal BF16 safetensors file from (name, shape, bf16 values).
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

    fn write_safetensors_f32(path: &Path, tensors: &[(String, Vec<usize>, Vec<f32>)]) {
        let mut data = Vec::new();
        let mut header = serde_json::Map::new();
        for (name, shape, vals) in tensors {
            let begin = data.len();
            for &v in vals {
                data.extend_from_slice(&v.to_le_bytes());
            }
            header.insert(
                name.clone(),
                json!({"dtype": "F32", "shape": shape, "data_offsets": [begin, data.len()]}),
            );
        }
        let hjson = serde_json::to_vec(&header).unwrap();
        let mut out = (hjson.len() as u64).to_le_bytes().to_vec();
        out.extend_from_slice(&hjson);
        out.extend_from_slice(&data);
        std::fs::write(path, out).unwrap();
    }

    // Loading an F32 checkpoint: tensors are cast to bf16 on upload, bit-exact
    // round-to-nearest-even (matches `cast_fp32_to_bf16`). (BF16/F16 share the
    // path; F16 host-encoding is awkward in pure Rust so F32 stands in here.)
    #[test]
    fn load_f32_checkpoint() {
        let dev = match Device::new(0) {
            Ok(d) => d,
            Err(e) => {
                eprintln!("skipping load_f32_checkpoint (no device): {e:#}");
                return;
            }
        };
        let (vocab, hidden, nq, nkv, hd, inter) = (32usize, 16, 2, 1, 8, 32);
        let (hq, hkv) = (nq * hd, nkv * hd);
        let g = |seed: f32, n: usize, scale: f32| -> Vec<f32> {
            (0..n).map(|i| ((i as f32 + seed) * 0.1).sin() * scale).collect()
        };
        let p = "model.layers.0";
        let tensors: Vec<(String, Vec<usize>, Vec<f32>)> = vec![
            ("model.embed_tokens.weight".into(), vec![vocab, hidden], g(1.0, vocab * hidden, 0.5)),
            ("model.norm.weight".into(), vec![hidden], g(2.0, hidden, 1.0)),
            ("lm_head.weight".into(), vec![vocab, hidden], g(3.0, vocab * hidden, 0.1)),
            (format!("{p}.input_layernorm.weight"), vec![hidden], g(4.0, hidden, 1.0)),
            (format!("{p}.post_attention_layernorm.weight"), vec![hidden], g(5.0, hidden, 1.0)),
            (format!("{p}.self_attn.q_proj.weight"), vec![hq, hidden], g(6.0, hq * hidden, 0.1)),
            (format!("{p}.self_attn.k_proj.weight"), vec![hkv, hidden], g(7.0, hkv * hidden, 0.1)),
            (format!("{p}.self_attn.v_proj.weight"), vec![hkv, hidden], g(8.0, hkv * hidden, 0.1)),
            (format!("{p}.self_attn.o_proj.weight"), vec![hidden, hq], g(9.0, hidden * hq, 0.1)),
            (format!("{p}.mlp.gate_proj.weight"), vec![inter, hidden], g(10.0, inter * hidden, 0.1)),
            (format!("{p}.mlp.up_proj.weight"), vec![inter, hidden], g(11.0, inter * hidden, 0.1)),
            (format!("{p}.mlp.down_proj.weight"), vec![hidden, inter], g(12.0, hidden * inter, 0.1)),
        ];
        let path = std::env::temp_dir().join(format!("cuda_new_f32_{}.safetensors", std::process::id()));
        write_safetensors_f32(&path, &tensors);

        let loaded = LoadedLlama::load(&dev, &path, hd, 1e-5, 10000.0).unwrap();
        assert_eq!(loaded.cfg.vocab, vocab);
        assert_eq!(loaded.cfg.n_layers, 1);
        assert_eq!(loaded.cfg.n_q_heads, nq);

        // F32 → bf16 cast on load is RNE, bit-exact.
        let check = |buf: &DeviceBuffer, want_f32: &[f32]| {
            let mut got = vec![0u16; want_f32.len()];
            buf.download(&mut got).unwrap();
            dev.sync().unwrap();
            for (i, &f) in want_f32.iter().enumerate() {
                assert_eq!(got[i], f32_to_bf16_rne(f), "cast mismatch at {i}");
            }
        };
        check(&loaded.embed, &tensors[0].2);
        check(&loaded.lm_head, &tensors[2].2);
        check(&loaded.wq[0], &tensors[5].2);

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn load_and_forward_safetensors() {
        let dev = match Device::new(0) {
            Ok(d) => d,
            Err(e) => {
                eprintln!("skipping load_and_forward_safetensors (no device): {e:#}");
                return;
            }
        };
        let (vocab, hidden, n_layers, nq, nkv, hd, inter) = (64usize, 32, 2, 2, 1, 16, 64);
        let (hq, hkv) = (nq * hd, nkv * hd);

        // Build synthetic llama tensors (HF names) and write a safetensors file.
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
        let path = std::env::temp_dir().join(format!("cuda_new_loader_{}.safetensors", std::process::id()));
        write_safetensors(&path, &tensors);

        let loaded = LoadedLlama::load(&dev, &path, hd, 1e-5, 10000.0).unwrap();

        // (a) inferred dims correct.
        assert_eq!(loaded.cfg.vocab, vocab);
        assert_eq!(loaded.cfg.hidden, hidden);
        assert_eq!(loaded.cfg.n_layers, n_layers);
        assert_eq!(loaded.cfg.n_q_heads, nq);
        assert_eq!(loaded.cfg.n_kv_heads, nkv);
        assert_eq!(loaded.cfg.intermediate, inter);

        // (b) name→buffer mapping + upload are bit-exact (download a sample).
        let check = |buf: &DeviceBuffer, want: &[u16]| {
            let mut got = vec![0u16; want.len()];
            buf.download(&mut got).unwrap();
            dev.sync().unwrap();
            assert_eq!(&got, want, "downloaded tensor bytes differ from file");
        };
        check(&loaded.embed, &tensors[0].2);
        check(&loaded.lm_head, &tensors[2].2);
        check(&loaded.wq[0], &tensors[5].2); // tensors[5] = layer-0 q_proj

        // (c) forward smoke-run on the loaded weights → tokens in range.
        let (t, page_size) = (3usize, 8usize);
        let ws = dev
            .workspace(t as i32, hidden as i32, nq as i32, nkv as i32, hd as i32, inter as i32, vocab as i32)
            .unwrap();
        let tib = up_dev(&dev, &[5i32, 17, 2]);
        let pb = up_dev(&dev, &[0i32, 1, 2]);
        let kv_k = dev.alloc(n_layers * page_size * hkv * 2).unwrap();
        let kv_v = dev.alloc(n_layers * page_size * hkv * 2).unwrap();
        let qib = up_dev(&dev, &[0u32, t as u32]);
        let kpi = up_dev(&dev, &[0u32]);
        let kpp = up_dev(&dev, &[0u32, 1]);
        let klp = up_dev(&dev, &[t as u32]);
        let out_logits = dev.alloc(t * vocab * 2).unwrap();
        let out_tokens = dev.alloc(t * 4).unwrap();
        loaded
            .forward(&dev, &ws, &tib, &pb, &kv_k, &kv_v, &qib, &kpi, &kpp, &klp, &out_logits,
                     &out_tokens, t as i32, 1, page_size as i32, 1)
            .unwrap();
        dev.sync().unwrap();
        let mut toks = vec![0i32; t];
        out_tokens.download(&mut toks).unwrap();
        dev.sync().unwrap();
        for &tk in &toks {
            assert!((0..vocab as i32).contains(&tk), "token {tk} out of range");
        }
        let _ = std::fs::remove_file(&path);
    }

    // ── DeepSeek weight-surgery helpers (B3) — pure, no device needed ──

    #[test]
    fn split_kv_b_proj_transposes_k_keeps_v() {
        // nh=2, qk_nope=2, v_head=3, kv_lora=4. kv_b[i] = i, so the math is
        // checkable by hand. Per head the layout is [K-up (2×4) | V-up (3×4)].
        let (nh, qk_nope, v_head, kv_lora) = (2usize, 2usize, 3usize, 4usize);
        let n = nh * (qk_nope + v_head) * kv_lora; // 40
        let kv_b: Vec<u16> = (0..n as u16).collect();
        let (w_uk, w_uv) = split_kv_b_proj(&kv_b, nh, qk_nope, v_head, kv_lora);

        // Head 0 K-up rows: Kup[0]=[0,1,2,3], Kup[1]=[4,5,6,7].
        // W_uk[0][l][d] = Kup[d][l]  → row-major [kv_lora, qk_nope]:
        //   l=0:(d0,d1)=(0,4) l=1:(1,5) l=2:(2,6) l=3:(3,7)
        assert_eq!(&w_uk[0..kv_lora * qk_nope], &[0, 4, 1, 5, 2, 6, 3, 7]);
        // Head 0 V-up rows start at qk_nope*kv_lora = 8: Vup = 8..20, copied direct.
        assert_eq!(&w_uv[0..v_head * kv_lora], &(8u16..20).collect::<Vec<_>>()[..]);

        // Spot-check head 1 transpose too (base = (qk_nope+v_head)*kv_lora = 20).
        // Kup[0]=[20,21,22,23], Kup[1]=[24,25,26,27] → W_uk[1] = [20,24,21,25,22,26,23,27].
        assert_eq!(
            &w_uk[kv_lora * qk_nope..2 * kv_lora * qk_nope],
            &[20, 24, 21, 25, 22, 26, 23, 27]
        );
    }

    #[test]
    fn stack_moe_experts_layout() {
        // E=2, inter=2, hidden=3. gate/up [inter,hidden]=6 elems; down [hidden,inter]=6.
        let (inter, hidden) = (2usize, 3usize);
        let gates = vec![vec![10u16, 11, 12, 13, 14, 15], vec![20, 21, 22, 23, 24, 25]];
        let ups = vec![vec![30u16, 31, 32, 33, 34, 35], vec![40, 41, 42, 43, 44, 45]];
        let downs = vec![vec![50u16, 51, 52, 53, 54, 55], vec![60, 61, 62, 63, 64, 65]];
        let (wgu, wdown) = stack_moe_experts(&gates, &ups, &downs, inter, hidden);
        // wgu [E, 2*inter, hidden]: expert0 = gate0 || up0, expert1 = gate1 || up1.
        assert_eq!(&wgu[0..6], &gates[0][..]);
        assert_eq!(&wgu[6..12], &ups[0][..]);
        assert_eq!(&wgu[12..18], &gates[1][..]);
        assert_eq!(&wgu[18..24], &ups[1][..]);
        // wdown [E, hidden, inter]: expert slabs copied direct.
        assert_eq!(&wdown[0..6], &downs[0][..]);
        assert_eq!(&wdown[6..12], &downs[1][..]);
    }

    /// Load + forward the REAL Qwen3-4B checkpoint on the GPU (sharded bf16 +
    /// qk-norm + tied embeddings). Proves the rewrite runs a real frontier model
    /// end-to-end. Dumps last-token logits to /tmp for parity vs driver/cuda.
    /// Skips if the model or enough free GPU memory isn't present.
    #[test]
    fn forward_real_qwen3_4b() {
        let dev = match Device::new(0) {
            Ok(d) => d,
            Err(e) => { eprintln!("skip forward_real_qwen3_4b (no device): {e:#}"); return; }
        };
        let base = "/root/.pie/programs/models--Qwen--Qwen3-4B/snapshots";
        let snap = match std::fs::read_dir(base).ok().and_then(|mut it| it.find_map(|e| e.ok())) {
            Some(e) => e.path(),
            None => { eprintln!("skip forward_real_qwen3_4b (model absent)"); return; }
        };
        let (free, _) = dev.mem_info().unwrap();
        if free < 14 * 1024 * 1024 * 1024 {
            eprintln!("skip forward_real_qwen3_4b (<14 GiB free)");
            return;
        }

        // Qwen3-4B: head_dim=128 (explicit, ≠ hidden/heads), rms_eps 1e-6, rope 1e6.
        let t0 = std::time::Instant::now();
        let m = LoadedLlama::load(&dev, &snap, 128, 1e-6, 1_000_000.0).unwrap();
        dev.sync().unwrap();
        let load_ms = t0.elapsed().as_millis();
        let c = &m.cfg;
        println!(
            "[qwen3-4b] loaded {}ms: vocab={} hidden={} layers={} n_q={} n_kv={} head_dim={} inter={}",
            load_ms, c.vocab, c.hidden, c.n_layers, c.n_q_heads, c.n_kv_heads, c.head_dim, c.intermediate
        );

        let toks: Vec<i32> = vec![785, 9707, 11, 1879, 374]; // arbitrary prompt ids
        let t = toks.len();
        let (page_size, num_pages) = (16usize, 1usize);
        let ws = dev.workspace(t as i32, c.hidden as i32, c.n_q_heads as i32, c.n_kv_heads as i32,
            c.head_dim as i32, c.intermediate as i32, c.vocab as i32).unwrap();
        let tib = up_dev(&dev, &toks);
        let pb = up_dev(&dev, &(0..t as i32).collect::<Vec<_>>());
        let hkv = c.n_kv_heads * c.head_dim;
        let kv_k = up_dev(&dev, &vec![0u16; c.n_layers * num_pages * page_size * hkv]);
        let kv_v = up_dev(&dev, &vec![0u16; c.n_layers * num_pages * page_size * hkv]);
        let qib = up_dev(&dev, &[0u32, t as u32]);
        let kpi = up_dev(&dev, &[0u32]);
        let kpp = up_dev(&dev, &[0u32, 1]);
        let klp = up_dev(&dev, &[t as u32]);
        let out_logits = dev.alloc(t * c.vocab * 2).unwrap();
        let out_tokens = dev.alloc(t * 4).unwrap();

        // Time the forward (warm: run once, then time a second pass).
        m.forward(&dev, &ws, &tib, &pb, &kv_k, &kv_v, &qib, &kpi, &kpp, &klp, &out_logits,
            &out_tokens, t as i32, 1, page_size as i32, num_pages as i32).unwrap();
        dev.sync().unwrap();
        let t1 = std::time::Instant::now();
        m.forward(&dev, &ws, &tib, &pb, &kv_k, &kv_v, &qib, &kpi, &kpp, &klp, &out_logits,
            &out_tokens, t as i32, 1, page_size as i32, num_pages as i32).unwrap();
        dev.sync().unwrap();
        let fwd_us = t1.elapsed().as_micros();

        let mut tk = vec![0i32; t];
        out_tokens.download(&mut tk).unwrap();
        let mut lg = vec![0u16; t * c.vocab];
        out_logits.download(&mut lg).unwrap();
        dev.sync().unwrap();
        // finite + valid argmax.
        for &v in &lg { assert!(crate::device::bf16_to_f32(v).is_finite(), "qwen3-4b logits finite"); }
        for &id in &tk { assert!((0..c.vocab as i32).contains(&id), "qwen3-4b argmax {id} in range"); }
        println!("[qwen3-4b] prefill {} tok in {}us → argmax(next)={:?}", t, fwd_us, tk);

        // Dump last-token logits (f32) for parity comparison vs driver/cuda.
        let last = &lg[(t - 1) * c.vocab..t * c.vocab];
        let last_f32: Vec<f32> = last.iter().map(|&v| crate::device::bf16_to_f32(v)).collect();
        let mut bytes = Vec::with_capacity(last_f32.len() * 4);
        for &v in &last_f32 { bytes.extend_from_slice(&v.to_le_bytes()); }
        let out = format!("/tmp/qwen3_4b_cuda_new_logits_{}.bin", std::process::id());
        let _ = std::fs::write(&out, &bytes);
        println!("[qwen3-4b] last-token logits ({} f32) → {out}", last_f32.len());
    }

    #[test]
    fn load_and_forward_deepseek() {
        // End-to-end DeepSeek-MLA bring-up: synthetic checkpoint → load (kv_b
        // split/transpose + MoE stacking) → full deepseek_forward → tokens. Proves
        // the detect→load→route→forward pipeline for the headline frontier.
        let dev = match Device::new(0) {
            Ok(d) => d,
            Err(e) => {
                eprintln!("skipping load_and_forward_deepseek (no device): {e:#}");
                return;
            }
        };
        let (vocab, hidden, nh) = (32usize, 256, 2);
        let (q_lora, kv_lora, qk_nope, qk_rope, v_head) = (96usize, 128, 128, 64, 128);
        let (n_layers, first_k_dense, e, top_k, moe_inter, dense_inter) = (2usize, 1, 4, 2, 128, 128);
        let (qb_out, kva_out, kvb_out, ov) =
            (nh * (qk_nope + qk_rope), kv_lora + qk_rope, nh * (qk_nope + v_head), nh * v_head);

        let mut tensors: Vec<(String, Vec<usize>, Vec<u16>)> = vec![
            ("model.embed_tokens.weight".into(), vec![vocab, hidden], syn(1.0, vocab * hidden, 0.5)),
            ("model.norm.weight".into(), vec![hidden], syn(2.0, hidden, 1.0)),
            ("lm_head.weight".into(), vec![vocab, hidden], syn(3.0, vocab * hidden, 0.1)),
        ];
        for i in 0..n_layers {
            let s = (i * 100) as f32;
            let p = format!("model.layers.{i}");
            let mut push = |suffix: &str, shape: Vec<usize>, seed: f32, scale: f32| {
                let n: usize = shape.iter().product();
                tensors.push((format!("{p}.{suffix}"), shape, syn(s + seed, n, scale)));
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
                    push(&format!("mlp.experts.{x}.gate_proj.weight"), vec![moe_inter, hidden], sx, 0.05);
                    push(&format!("mlp.experts.{x}.up_proj.weight"), vec![moe_inter, hidden], sx + 1.0, 0.05);
                    push(&format!("mlp.experts.{x}.down_proj.weight"), vec![hidden, moe_inter], sx + 2.0, 0.05);
                }
            }
        }
        let path = std::env::temp_dir().join(format!("cuda_new_deepseek_{}.safetensors", std::process::id()));
        write_safetensors(&path, &tensors);

        let cfg = DeepseekConfig {
            vocab, hidden, n_layers, num_heads: nh, q_lora_rank: q_lora, kv_lora_rank: kv_lora,
            qk_nope_head_dim: qk_nope, qk_rope_head_dim: qk_rope, v_head_dim: v_head,
            first_k_dense, dense_inter, moe_inter, num_experts: e, top_k, rms_eps: 1e-6,
            rope_theta: 10000.0,
        };
        let loaded = LoadedDeepseek::load(&dev, &path, cfg).unwrap();
        assert_eq!(loaded.cfg.n_layers, n_layers);

        // On-device W_uk[0] must equal the host kv_b split (validates the upload of
        // the transposed surgery output, end to end).
        let kvb = tensors
            .iter()
            .find(|(n, _, _)| n == "model.layers.0.self_attn.kv_b_proj.weight")
            .map(|(_, _, v)| v.clone())
            .unwrap();
        let (uk_expect, _) = split_kv_b_proj(&kvb, nh, qk_nope, v_head, kv_lora);
        let mut uk_got = vec![0u16; uk_expect.len()];
        loaded.w_uk[0].download(&mut uk_got).unwrap();
        dev.sync().unwrap();
        assert_eq!(uk_got, uk_expect, "on-device W_uk[0] != host split");

        // e2e forward → next tokens in range.
        let (t, page_size, num_pages) = (4usize, 16usize, 1usize);
        let tib = up_dev(&dev, &[1i32, 7, 3, 0]);
        let pb = up_dev(&dev, &[0i32, 1, 2, 3]);
        let ckv = up_dev(&dev, &vec![0u16; n_layers * num_pages * page_size * kv_lora]);
        let kpe = up_dev(&dev, &vec![0u16; n_layers * num_pages * page_size * qk_rope]);
        let qib = up_dev(&dev, &[0u32, t as u32]);
        let kpi = up_dev(&dev, &[0u32]);
        let kpp = up_dev(&dev, &[0u32, 1]);
        let klp = up_dev(&dev, &[t as u32]);
        let out_logits = dev.alloc(t * vocab * 2).unwrap();
        let out_tokens = dev.alloc(t * 4).unwrap();
        loaded
            .forward(&dev, &tib, &pb, &ckv, &kpe, &qib, &kpi, &kpp, &klp, &out_logits, &out_tokens,
                     t as i32, 1, page_size as i32, num_pages as i32)
            .unwrap();
        dev.sync().unwrap();
        let mut toks = vec![0i32; t];
        out_tokens.download(&mut toks).unwrap();
        dev.sync().unwrap();
        for &tk in &toks {
            assert!((0..vocab as i32).contains(&tk), "token {tk} out of range");
        }
        let _ = std::fs::remove_file(&path);
    }
}
