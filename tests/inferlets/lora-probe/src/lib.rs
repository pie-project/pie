//! LoRA configuration-sink probe (tensor-ir-log.md §6.5, plan §5.1).
//!
//! Structurally naive-baseline (one N-wide prefill fire, then a
//! device-carried decode loop) plus exactly one addition: every forward pass
//! carries a PROLOGUE `intrinsics::kernel::lora(A, B, SITES)` whose adapter
//! weights are host-seeded channels. The A/B contents are a deterministic
//! pattern; `adapter_scale` is folded into B's seed (the same slot the LoRA
//! `alpha/R` folds into, §6.5), so:
//!
//!   * `adapter_scale = 0.0` — B is all zeros, the CORRECTION term
//!     `(xAᵀ)Bᵀ` is exactly zero, and the output must be byte-identical to
//!     naive-baseline at the same seed/prompt (§5.1: "with no adapters the
//!     code is what it was" — here, zero-B).
//!   * `adapter_scale > 0.0` — the delta lands on the q projection at every
//!     layer; the text differs and must be deterministic across runs.
//!
//! v0 sites: `q` only (bit 0 of the llama-like site vocabulary,
//! driver/cuda/src/model/lora.hpp). One lane carries ONE (A, B) pair with one
//! trace-known d_out, and qwen3-0.6b's q width (2048) differs from its v
//! width (1024), so a q+v adapter would need either per-site pairs or a
//! packed layout — the algebra proof only needs one site. `v` is the
//! documented next step.

use inferlet::ptir::prelude::*;
use inferlet::{Result, model as wit_model};
use serde::{Deserialize, Serialize};

// Qwen3-0.6B adapter geometry: trace-known shape (a different rank is a
// different traced program); the CONTENTS are per-instance data.
const NUM_LAYERS: u32 = 28;
const RANK: u32 = 8;
const D_IN: u32 = 1024; // hidden_size
const D_OUT: u32 = 2048; // q width = 16 heads * head_dim 128
const SITE_Q: u32 = 1 << 0;

#[derive(Deserialize)]
struct Input {
    #[serde(default = "default_prompt")]
    prompt: String,
    #[serde(default = "default_temperature")]
    temperature: f32,
    #[serde(default = "default_max_tokens")]
    max_tokens: usize,
    #[serde(default = "default_seed")]
    seed: u32,
    /// Scale folded into B's seed contents. 0.0 = zero-B adapter.
    #[serde(default)]
    adapter_scale: f32,
    /// Which surface attaches the adapter: "sink" (kernel::lora, the
    /// original) or "adapter" (fwd.adapter, the PEFT v0a surface —
    /// must be byte-identical: same channels, same lowering).
    #[serde(default = "default_surface")]
    surface: String,
    /// Which sites carry adapters: "q" (the original single-site probe)
    /// or "qv" (the per-site-pairs rung: distinct shapes per site,
    /// adapter surface only).
    #[serde(default = "default_sites")]
    sites: String,
}

fn default_sites() -> String {
    "q".into()
}

fn default_surface() -> String {
    "sink".into()
}

fn default_prompt() -> String {
    "Write a short paragraph about naive sampling.".into()
}

fn default_temperature() -> f32 {
    1.0
}

fn default_max_tokens() -> usize {
    32
}

fn default_seed() -> u32 {
    0x7ce1
}

#[derive(Serialize)]
struct Output {
    sampler: &'static str,
    text: String,
    count: usize,
    adapter_scale: f32,
}

/// Splitmix-style integer hash: deterministic, platform-independent.
fn hash_u32(mut x: u32) -> u32 {
    x = x.wrapping_add(0x9e37_79b9);
    x ^= x >> 16;
    x = x.wrapping_mul(0x85eb_ca6b);
    x ^= x >> 13;
    x = x.wrapping_mul(0xc2b2_ae35);
    x ^= x >> 16;
    x
}

/// Deterministic pattern in [-amp, amp).
fn pattern(i: u32, salt: u32, amp: f32) -> f32 {
    let h = hash_u32(i ^ salt);
    ((h % 10_000) as f32 / 10_000.0 - 0.5) * 2.0 * amp
}

/// One sampling step: temperature, then a Gumbel-max draw over the full
/// vocab. Byte-for-byte the naive-baseline step — the comparison depends on
/// it.
fn step(logits: Tensor, temperature: f32, rng_state: impl AsTensor + Copy) -> Tensor {
    let scaled = if temperature == 1.0 {
        logits
    } else {
        div(&logits, temperature)
    };
    gumbel_max(scaled, rng_state)
}

#[inferlet::main]
async fn main(input: Input) -> Result<Output> {
    if !input.temperature.is_finite() || input.temperature <= 0.0 {
        return Err("temperature must be finite and greater than 0".into());
    }
    if !input.adapter_scale.is_finite() {
        return Err("adapter_scale must be finite".into());
    }
    let max_tokens = input.max_tokens;
    let temperature = input.temperature;
    let adapter_scale = input.adapter_scale;
    let ws = WorkingSet::new();
    let page_size = ws.page_size();

    if max_tokens == 0 {
        return Ok(Output {
            sampler: "lora-probe",
            text: String::new(),
            count: 0,
            adapter_scale,
        });
    }

    // ── Adapter weights: host-built, deterministic. ──
    // A: [num_layers, R, d_in]; B: [num_layers, d_out, R] with adapter_scale
    // folded into the contents (there is no scalar argument — §6.5). Each
    // PASS gets its own seeded channel pair below: a channel's seed is
    // consumed by the first pass that binds it, so a pair cannot be shared
    // across the prefill and decode passes.
    let a_len = (NUM_LAYERS * RANK * D_IN) as usize;
    let b_len = (NUM_LAYERS * D_OUT * RANK) as usize;
    let a_host: Vec<f32> = (0..a_len as u32)
        .map(|i| pattern(i, 0x0a0a_a0a0, 0.05))
        .collect();
    let b_host: Vec<f32> = (0..b_len as u32)
        .map(|i| pattern(i, 0x0b0b_b0b0, 0.5) * adapter_scale)
        .collect();
    // The v site's pair (per-site rung): SAME d_in and rank, its OWN
    // d_out (v width) and its own deterministic contents.
    const D_OUT_V: u32 = 1024; // kv width = 8 heads * head_dim 128
    let bv_len = (NUM_LAYERS * D_OUT_V * RANK) as usize;
    let bv_host: Vec<f32> = (0..bv_len as u32)
        .map(|i| pattern(i, 0x0c0c_c0c0, 0.5) * adapter_scale)
        .collect();
    let make_lora_channels = |a_host: &Vec<f32>, b_host: &Vec<f32>| {
        (
            Channel::from_shaped([NUM_LAYERS, RANK, D_IN], a_host.clone())
                .named("lora_a"),
            Channel::from_shaped([NUM_LAYERS, D_OUT, RANK], b_host.clone())
                .named("lora_b"),
        )
    };

    let mut prompt = wit_model::encode(&input.prompt);
    if prompt.is_empty() {
        prompt.push(0);
    }
    let n = prompt.len() as u32;
    let max_pages = (n + max_tokens as u32 + 1).div_ceil(page_size).max(1);
    ws.reserve(max_pages)
        .map_err(|e| format!("reserve KV: {e}"))?;

    let mut generated: Vec<u32> = Vec::with_capacity(max_tokens);

    // ── PREFILL (chunked, C-wide) — naive-baseline's shape, plus the lora
    // prologue on every pass so the whole forward applies the delta. ──
    let prompt_i32: Vec<i32> = prompt.iter().map(|&t| t as i32).collect();
    let spans = prefill_chunks(n, None);
    let pipe = Pipeline::new();

    let mut g0 = 0i32;
    for &(base, end) in &spans {
        let len = end - base;

        let toks_p =
            Channel::from(prompt_i32[base as usize..end as usize].to_vec()).named("toks_p");
        let embed_indptr_p = Channel::from(vec![0u32, len]).named("embed_indptr_p");
        let positions_p = Channel::from((base..end).collect::<Vec<_>>()).named("positions_p");
        let pages_p = Channel::from((0..max_pages).collect::<Vec<_>>()).named("pages_p");
        let page_indptr_p =
            Channel::from(vec![0u32, end.div_ceil(page_size)]).named("page_indptr_p");
        let w_slot_p =
            Channel::from((base..end).map(|p| p / page_size).collect::<Vec<_>>()).named("w_slot_p");
        let w_off_p =
            Channel::from((base..end).map(|p| p % page_size).collect::<Vec<_>>()).named("w_off_p");
        let kv_len_p = Channel::from(vec![end]).named("kv_len_p");
        let rng_p = Channel::from(vec![input.seed, 0]).named("rng_p");
        let tok_out_p = Channel::new([1], dtype::i32).named("tok_out_p");

        let (lora_a, lora_b) = make_lora_channels(&a_host, &b_host);
        let fwd_p = ForwardPass::new();
        // The configuration sink: reads are peeks (no edge onto the decode
        // chain), SITES is trace-known placement. The "adapter" surface
        // states the SAME thing through the PEFT v0a classifier.
        if input.surface == "adapter" {
            use inferlet::ptir::adapter::{mm, Site};
            fwd_p
                .adapter(Site::Q, |x, y| y + mm(&lora_b, mm(&lora_a, x)))
                .map_err(|e| e.to_string())?;
            if input.sites == "qv" {
                let av = Channel::from_shaped(
                    [NUM_LAYERS, RANK, D_IN], a_host.clone())
                    .named("lora_a_v");
                let bv = Channel::from_shaped(
                    [NUM_LAYERS, D_OUT_V, RANK], bv_host.clone())
                    .named("lora_b_v");
                fwd_p
                    .adapter(Site::V, |x, y| y + mm(&bv, mm(&av, x)))
                    .map_err(|e| e.to_string())?;
            }
        } else if input.surface == "clone" {
            let (a2, b2) = (lora_a.clone(), lora_b.clone());
            fwd_p.prologue(move || {
                intrinsics::kernel::lora(
                    a2.read(),
                    b2.read(),
                    Tensor::constant(SITE_Q),
                );
            });
        } else {
            fwd_p.prologue(move || {
                intrinsics::kernel::lora(
                    lora_a.read(),
                    lora_b.read(),
                    Tensor::constant(SITE_Q),
                );
            });
        }
        fwd_p.embed(&toks_p, &embed_indptr_p)?;
        fwd_p.attention(
            &ws,
            ..,
            ..,
            &kv_len_p,
            &pages_p,
            &page_indptr_p,
            &w_slot_p,
            &w_off_p,
            &positions_p,
            None,
        )?;
        fwd_p.epilogue(move || {
            let r = rng_p.take();
            let logits = intrinsics::logits();
            let token = step(logits, temperature, &r);
            let r_next = add(&r, iota(2));
            tok_out_p.put(&token);
            rng_p.put(&r_next);
        });

        fwd_p
            .submit(&pipe)
            .map_err(|e| format!("prefill submit @{base}: {e}"))?;

        g0 = tok_out_p
            .take()
            .get::<i32>()
            .await
            .map_err(|e| format!("g0 take @{base}: {e}"))?[0];
    }
    generated.push(g0 as u32);

    // ── DECODE LOOP (1-wide, run-ahead) — naive-baseline's shape, plus the
    // lora prologue. ──
    if generated.len() < max_tokens {
        let tok_in = Channel::from(vec![g0; 1]).named("tok_in");
        let rng = Channel::from(vec![input.seed ^ 0x5bd1, 0]).named("rng");
        let tok_out = Channel::new([1], dtype::i32)
            .capacity(channel_capacity() as u32)
            .named("tok_out");
        let lane1 = Channel::from(vec![0u32, 1u32]).named("embed_indptr");
        let positions = Channel::from(vec![n]).named("positions");
        let pages = Channel::from((0..max_pages).collect::<Vec<_>>()).named("pages");
        let page_indptr =
            Channel::from(vec![0u32, (n + 1).div_ceil(page_size)]).named("page_indptr");
        let w_slot = Channel::from(vec![n / page_size]).named("w_slot");
        let w_off = Channel::from(vec![n % page_size]).named("w_off");
        let kv_len = Channel::from(vec![n + 1]).named("kv_len");

        let (lora_a, lora_b) = make_lora_channels(&a_host, &b_host);
        let fwd = ForwardPass::new();
        if input.surface == "adapter" {
            use inferlet::ptir::adapter::{mm, Site};
            fwd.adapter(Site::Q, |x, y| y + mm(&lora_b, mm(&lora_a, x)))
                .map_err(|e| e.to_string())?;
            if input.sites == "qv" {
                let av = Channel::from_shaped(
                    [NUM_LAYERS, RANK, D_IN], a_host.clone())
                    .named("lora_a_v");
                let bv = Channel::from_shaped(
                    [NUM_LAYERS, D_OUT_V, RANK], bv_host.clone())
                    .named("lora_b_v");
                fwd.adapter(Site::V, |x, y| y + mm(&bv, mm(&av, x)))
                    .map_err(|e| e.to_string())?;
            }
        } else {
            fwd.prologue(move || {
                intrinsics::kernel::lora(
                    lora_a.read(),
                    lora_b.read(),
                    Tensor::constant(SITE_Q),
                );
            });
        }
        fwd.embed(&tok_in, &lane1)?;
        fwd.attention(
            &ws,
            ..,
            (n / page_size)..,
            &kv_len,
            &pages,
            &page_indptr,
            &w_slot,
            &w_off,
            &positions,
            None,
        )?;
        fwd.epilogue(move || {
            let length = kv_len.take().tensor();
            let r = rng.take();
            let logits = intrinsics::logits();
            let token = step(logits, temperature, &r);

            let r_next = add(&r, iota(2));
            let next_length = add(&length, 1u32);
            let page_count = div(add(&next_length, page_size - 1), page_size);

            tok_in.put(&token);
            kv_len.put(&next_length);
            positions.put(&length);
            w_slot.put(div(&length, page_size));
            w_off.put(rem(&length, page_size));
            page_indptr.take();
            page_indptr.put(mul(iota(2), broadcast(&page_count, [2])));
            tok_out.put(&token);
            rng.put(&r_next);
        });

        let budget = max_tokens - 1;
        run_ahead(&pipe, &fwd, budget, async || {
            let t = tok_out
                .take()
                .get::<i32>()
                .await
                .map_err(|e| format!("tok_out.take @{}: {e}", generated.len()))?[0];
            generated.push(t as u32);
            Ok(ControlFlow::Continue(()))
        })
        .await?;
    }
    pipe.close();

    Ok(Output {
        sampler: "lora-probe",
        text: wit_model::decode(&generated)?,
        count: generated.len(),
        adapter_scale,
    })
}
