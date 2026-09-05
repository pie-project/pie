//! **THE ADAPTER PROBE** — naive-baseline plus one PEFT adapter, and nothing
//! else different (alto adapter §6, campaign gate A-1).
//!
//! # What this program is for
//!
//! `cuda_lora_parity` asks the one LoRA question that can be answered without
//! a reference implementation: *with no adapter, is the answer the base
//! model's?* The base model is sitting right there. So this inferlet is
//! `naive-baseline` — same prompt, same seed, same temperature, same Gumbel-max
//! draw, same epilogue — with exactly one addition: every forward pass carries
//! a `Pass::adapter` at the mixer-output site, whose `A` and `B` weights are
//! host-built, deterministic, and seeded into channels.
//!
//! * `adapter_scale = 0.0` — `B` is all zeros, the correction `B(Ax)` is
//!   EXACTLY zero (adding a bf16 zero is exact), and the output must be
//!   byte-identical to `naive-baseline` at the same seed and prompt.
//! * `adapter_scale > 0.0` — the delta lands on the mixer output at every
//!   layer, the token moves, and it moves the SAME WAY every run.
//!
//! # Where the weights actually go (alto adapter §6.1)
//!
//! The channels are a NAMING device, not a weight transport. The engine reads
//! the sink's channels once, at instance bind, converts the f32 cells into the
//! banks' own bf16 and lands them in a bank slot; no fire ever reads the cell.
//! That is why the seeds below are stated once per pass and never re-published:
//! swapping an adapter is re-BINDING, never re-publishing.
//!
//! # The site, and why there is only one
//!
//! `qwen_3`'s text corrects the MIXER OUTPUT and nothing else
//! (`crates/models/src/qwen_3/forward.rs`, `ops::linear::lora_correct` over the
//! reduced mixer result), so [`Site::O`] is the only site this family can
//! honestly serve today — adapter.md §2's "one site, not six", and §4's wave 4
//! is where the other five arrive. Both ends of that site are the replicated
//! hidden stream, so `A` is `[layers, rank, hidden]` and `B` is
//! `[layers, hidden, rank]` at ONE width: qwen35-d0.8b's 1024.
//!
//! # The geometry, and why it is a parameter
//!
//! The defaults are qwen35-d0.8b's — 24 layers, hidden 1024, the bank's own
//! rank 16 (`Adapters { slots: 8, rank: 16 }`). A different SKU passes its own
//! numbers rather than editing this file, because a rank is TRACE-KNOWN: a
//! different rank is a different traced program, which is exactly what a
//! parameter that reaches the channel shape expresses.

use inferlet::eta::adapter::{Site, mm};
use inferlet::eta::hybrid::prelude::*;
use inferlet::{Result, model as wit_model};
use serde::{Deserialize, Serialize};

/// qwen35-d0.8b's bank, as the model text declares it: `Adapters { slots: 8,
/// rank: 16 }` over 24 layers of hidden 1024.
const DEF_RANK: u32 = 16;
const DEF_LAYERS: u32 = 24;
const DEF_HIDDEN: u32 = 1024;

/// The mixer-output site's bit, for the raw-sink surface below — the same
/// number [`Site::O`] answers, spelled here because the sink intrinsic takes a
/// constant rather than the enum.
const SITE_O: u32 = 1 << 3;

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
    /// The scale folded into `B`'s contents. `0.0` is the zero-`B` adapter —
    /// a correction that is exactly zero — and there is no scalar argument to
    /// the sink because the LoRA `alpha/R` folds into `B` exactly here
    /// (adapter.md §2: weights are data).
    #[serde(default)]
    adapter_scale: f32,
    /// Which surface states the adapter: `"adapter"` (the current SDK
    /// spelling, `Pass::adapter` with its closed expression language) or
    /// `"sink"` (the raw `intrinsics::kernel::lora` prologue the surface
    /// lowers to). The two must be BYTE-IDENTICAL: same channels, same
    /// lowering, same sink.
    #[serde(default = "default_surface")]
    surface: String,
    /// `"lowrank"` (the served form) or `"scale"` (IA3's two-argument
    /// spelling). The scale form is REFUSED by the engine by name —
    /// `model-ir` declares no `AdapterScale` op for a bank to be read by — so
    /// this exists to make that refusal reachable from a guest.
    #[serde(default = "default_form")]
    form: String,
    /// Model geometry overrides; the defaults are qwen35-d0.8b's.
    #[serde(default)]
    layers: Option<u32>,
    #[serde(default)]
    hidden: Option<u32>,
    #[serde(default)]
    rank: Option<u32>,
}

fn default_surface() -> String {
    "adapter".into()
}

fn default_form() -> String {
    "lowrank".into()
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
    /// What the adapter's shape actually was, so a gate reading this JSON can
    /// say which geometry produced the text rather than assuming the defaults.
    rank: u32,
    layers: u32,
    hidden: u32,
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
/// vocab. Byte-for-byte `naive-baseline`'s step — the parity depends on it.
fn step(logits: Tensor, temperature: f32, rng_state: &Tensor) -> Tensor {
    let scaled = if temperature == 1.0 {
        logits
    } else {
        &logits / temperature
    };
    gumbel_max(scaled, rng_state)
}

/// The adapter's two planes, in the orientations §6.3 fixes: `A` rank-major
/// `[layers, rank, hidden]`, `B` out-major `[layers, hidden, rank]`.
///
/// A fresh pair PER PASS, because a channel's seed is consumed by the first
/// pass that binds it — the prefill and the decode are two instances and each
/// one lands its own copy.
struct Weights {
    a: Vec<f32>,
    b: Vec<f32>,
    layers: u32,
    hidden: u32,
    rank: u32,
}

impl Weights {
    fn build(layers: u32, hidden: u32, rank: u32, scale: f32) -> Weights {
        let a = (0..layers * rank * hidden)
            .map(|i| pattern(i, 0x0a0a_a0a0, 0.05))
            .collect();
        // **THE SCALE IS FOLDED INTO `B` AND NOWHERE ELSE.** At `scale = 0.0`
        // every element is exactly `0.0`, so `B(Ax)` is exactly zero and the
        // parity claim is an identity rather than a tolerance.
        let b = (0..layers * hidden * rank)
            .map(|i| pattern(i, 0x0b0b_b0b0, 0.5) * scale)
            .collect();
        Weights {
            a,
            b,
            layers,
            hidden,
            rank,
        }
    }

    fn channels(&self) -> (Channel, Channel) {
        (
            Channel::from_shaped([self.layers, self.rank, self.hidden], self.a.clone())
                .named("lora_a"),
            Channel::from_shaped([self.layers, self.hidden, self.rank], self.b.clone())
                .named("lora_b"),
        )
    }

    /// IA3's per-channel vector, for the refusal probe: `[layers, hidden]`,
    /// ones plus a deterministic deviation.
    fn scale_channel(&self) -> Channel {
        Channel::from_shaped(
            [self.layers, self.hidden],
            (0..self.layers * self.hidden)
                .map(|i| 1.0f32 + pattern(i, 0x0d0d_d0d0, 0.2))
                .collect::<Vec<f32>>(),
        )
        .named("lora_l")
    }
}

/// State the adapter on one pass, through whichever surface was asked for.
///
/// An adapter is a per-layer prologue sink, and nothing about it knows whether
/// the layer it corrects sits in an attention trunk or a hybrid one — which is
/// exactly why the one pass type serves both.
fn attach(fwd: &ForwardPass, w: &Weights, surface: &str, form: &str) -> Result<()> {
    if form == "scale" {
        let l = w.scale_channel();
        return fwd
            .adapter(Site::O, |_x, y| inferlet::eta::adapter::scale(y, &l))
            .map_err(|e| e.into());
    }
    let (a, b) = w.channels();
    if surface == "sink" {
        // The RAW sink, which is what the surface lowers to. Stated here so a
        // gate can assert the two spellings answer the same bytes: same
        // channels, same trace-known placement constant, same `SinkCall`.
        fwd.prologue(move || {
            intrinsics::kernel::lora(a.read(), b.read(), Tensor::constant(SITE_O));
        });
        return Ok(());
    }
    // The CURRENT SDK SPELLING: a closed expression the surface CLASSIFIES —
    // never interprets — into the low-rank lowering `y + mm(b, mm(a, x))`.
    fwd.adapter(Site::O, |x, y| y + mm(&b, mm(&a, x)))
        .map_err(|e| e.into())
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
    let layers = input.layers.unwrap_or(DEF_LAYERS);
    let hidden = input.hidden.unwrap_or(DEF_HIDDEN);
    let rank = input.rank.unwrap_or(DEF_RANK);
    let surface = input.surface.clone();
    let form = input.form.clone();
    let ws = WorkingSet::new();
    // One recurrent working set for this one sequence on a hybrid model (the
    // engine requires one per request row); none on a pure-attention one. It
    // is bound by EVERY pass this program builds — each prefill chunk and the
    // decode — so the recurrence sees the whole sequence in order.
    let rs_ws: Vec<RsWorkingSet> = match model::pass_kind() {
        model::ForwardKind::Attention => Vec::new(),
        model::ForwardKind::Hybrid => vec![RsWorkingSet::new()],
        model::ForwardKind::Recurrent => {
            return Err(
                "this program has no recurrent-only path (the adapter site it probes is \
                        an attention trunk's mixer output)"
                    .into(),
            );
        }
        model::ForwardKind::Diffusion => {
            return Err("this program decodes a token at a time; a diffusion model wants a canvas loop".into());
        }
    };
    let page_size = kv_page_size();

    let report = |text: String, count: usize| Output {
        sampler: "lora-probe",
        text,
        count,
        adapter_scale,
        rank,
        layers,
        hidden,
    };

    if max_tokens == 0 {
        return Ok(report(String::new(), 0));
    }

    // The model's opening (`<bos>` where it has one) before the raw text —
    // the SAME opening `naive-baseline` puts there. This probe's whole claim
    // is "naive-baseline plus one adapter and nothing else different", and
    // on gemma the two openings answer different text at a zero adapter
    // (" much like the…" against " of of of of"), which the A-1 gate then
    // reads as the adapter path writing where it must not.
    let mut prompt = inferlet::chat::prefix();
    prompt.extend(wit_model::encode(&input.prompt));
    if prompt.is_empty() {
        prompt.push(0);
    }
    let n = prompt.len() as u32;
    let max_pages = (n + max_tokens as u32 + 1).div_ceil(page_size).max(1);
    ws.reserve(max_pages)
        .map_err(|e| format!("reserve KV: {e}"))?;

    let mut generated: Vec<u32> = Vec::with_capacity(max_tokens);

    // ── PREFILL (chunked, C-wide) — naive-baseline's shape, plus the adapter
    //    on every pass so the whole forward applies the delta. ──
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

        let weights = Weights::build(layers, hidden, rank, adapter_scale);
        let fwd_p = ForwardPass::new();
        attach(&fwd_p, &weights, &surface, &form)?;
        fwd_p.embed(&toks_p, &embed_indptr_p)?;
        fwd_p.attention(
            Some(KvBinding {
                working_set: &ws,
                geometry: KvGeometry {
                    readable_pages: ..,
                    writable_pages: ..,
                    kv_len: &kv_len_p,
                    pages: &pages_p,
                    page_indptr: &page_indptr_p,
                    w_slot: &w_slot_p,
                    w_off: &w_off_p,
                    positions: &positions_p,
                    mask: None,
                },
            }),
            &rs_ws,
            RsGeometry {
                fold_len: None,
                buffer: 0..0,
            },
        )?;
        fwd_p.epilogue(move || {
            let r = rng_p.take();
            let logits = intrinsics::logits();
            let token = step(logits, temperature, &r);
            let r_next = &r + iota(2);
            tok_out_p.put(&token);
            rng_p.put(&r_next);
        });

        fwd_p
            .submit(&pipe)
            .map_err(|e| format!("prefill submit @{base}: {e}"))?;

        g0 = tok_out_p
            .take_host::<i32>()
            .await
            .map_err(|e| format!("g0 take @{base}: {e}"))?;
    }
    generated.push(g0 as u32);

    // ── DECODE LOOP (1-wide, run-ahead) — naive-baseline's shape, plus the
    //    adapter. ──
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

        let weights = Weights::build(layers, hidden, rank, adapter_scale);
        let fwd = ForwardPass::new();
        attach(&fwd, &weights, &surface, &form)?;
        fwd.embed(&tok_in, &lane1)?;
        fwd.attention(
            Some(KvBinding {
                working_set: &ws,
                geometry: KvGeometry {
                    readable_pages: ..,
                    writable_pages: (n / page_size)..,
                    kv_len: &kv_len,
                    pages: &pages,
                    page_indptr: &page_indptr,
                    w_slot: &w_slot,
                    w_off: &w_off,
                    positions: &positions,
                    mask: None,
                },
            }),
            &rs_ws,
            RsGeometry {
                fold_len: None,
                buffer: 0..0,
            },
        )?;
        fwd.epilogue(move || {
            let length = kv_len.take();
            let r = rng.take();
            let logits = intrinsics::logits();
            let token = step(logits, temperature, &r);

            let r_next = &r + iota(2);
            let next_length = &length + 1u32;
            let page_count = next_length.div_ceil(page_size);

            tok_in.put(&token);
            kv_len.put(&next_length);
            positions.put(&length);
            w_slot.put(&length / page_size);
            w_off.put(&length % page_size);
            page_indptr.put(indptr(1, &page_count));
            tok_out.put(&token);
            rng.put(&r_next);
        });

        let budget = max_tokens - 1;
        run_ahead(&pipe, &fwd, budget, async || {
            let t = tok_out
                .take_host::<i32>()
                .await
                .map_err(|e| format!("tok_out.take @{}: {e}", generated.len()))?;
            generated.push(t as u32);
            Ok(ControlFlow::Continue(()))
        })
        .await?;
    }
    pipe.close();

    let count = generated.len();
    Ok(report(wit_model::decode(&generated)?, count))
}
