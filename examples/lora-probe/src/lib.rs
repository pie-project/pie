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
//! (`ops::linear::lora_correct` over the reduced mixer result), so [`Site::O`] is the only site this family can
//! honestly serve today — adapter.md §2's "one site, not six", and §4's wave 4
//! is where the other five arrive. Both ends of that site are the replicated
//! hidden stream, so `A` is `[layers, rank, hidden]` and `B` is
//! `[layers, hidden, rank]` at ONE width: qwen35-d0.8b's 1024.
//!
//! # The geometry, and why it is a parameter
//!
//! The defaults are qwen35-d0.8b's — 24 layers, hidden 1024, the bank's own
//! rank 16 (`Adapters { slots: 8, rank: 16 }`). A different deployment passes its own
//! numbers rather than editing this file, because a rank is TRACE-KNOWN: a
//! different rank is a different traced program, which is exactly what a
//! parameter that reaches the channel shape expresses. gemma-4-26b-a4b is
//! `{"layers": 30, "hidden": 2816, "rank": 16}`.

use inferlet::eta::adapter::{Site, mm};
use inferlet::eta::prelude::*;
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
    /// Which surface states the adapter: `"adapter"` (the current library
    /// spelling, `Pass::adapter` with its closed expression language) or
    /// `"sink"` (the raw `intrinsics::kernel::lora` prologue the surface
    /// lowers to). The two must be BYTE-IDENTICAL: same channels, same
    /// lowering, same sink.
    #[serde(default = "default_surface")]
    surface: String,
    /// `"lowrank"` (the served form) or `"scale"` (IA3's two-argument
    /// spelling). The scale form is REFUSED by the engine by name —
    /// `poem-ir` declares no `AdapterScale` op for a bank to be read by — so
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
/// pass that binds it — the fill and the decode are two instances and each
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
            Channel::from_shaped([self.layers, self.rank, self.hidden], self.a.clone()),
            Channel::from_shaped([self.layers, self.hidden, self.rank], self.b.clone()),
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
        return fwd.adapter(Site::O, |_x, y| inferlet::eta::adapter::scale(y, &l));
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
    // The CURRENT library SPELLING: a closed expression the surface CLASSIFIES —
    // never interprets — into the low-rank lowering `y + mm(b, mm(a, x))`.
    fwd.adapter(Site::O, |x, y| y + mm(&b, mm(&a, x)))
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

    let report = |text: String, count: usize| Output {
        sampler: "lora-probe",
        text,
        count,
        adapter_scale,
        rank,
        layers,
        hidden,
    };

    // The model's opening (`<bos>` where it has one) before the raw text —
    // the SAME opening `naive-baseline` puts there. This probe's whole claim
    // is "naive-baseline plus one adapter and nothing else different", and
    // on gemma the two openings answer different text at a zero adapter
    // (" much like the…" against " of of of of"), which the A-1 gate then
    // reads as the adapter path writing where it must not.
    let prompt: Vec<i32> = inferlet::chat::prefix()
        .into_iter()
        .chain(wit_model::encode(&input.prompt))
        .map(|t| t as i32)
        .collect();
    let (&last, head) = prompt.split_last().ok_or("empty prompt")?;
    let pipe = Pipeline::new();
    if !head.is_empty() {
        let weights = Weights::build(layers, hidden, rank, adapter_scale);
        let fill = ForwardPass::new();
        attach(&fill, &weights, &surface, &form)?;
        fill.embed(
            &Channel::from(head),
            &Channel::from([0u32, head.len() as u32]),
        )?;
        fill.state(&[&ws], StateGeometry::folding())?;
        fill.submit(&pipe)
            .map_err(|e| format!("prefill submit: {e}"))?;
    }

    // ── DECODE LOOP (1-wide, run-ahead) — naive-baseline's shape, plus the
    //    adapter. ──
    let tok_in = Channel::from([last]);
    let rng = Channel::from([input.seed, 0]);
    let tok_out = Channel::new([1], dtype::i32).capacity(channel_capacity() as u32);

    let weights = Weights::build(layers, hidden, rank, adapter_scale);
    let fwd = ForwardPass::new();
    attach(&fwd, &weights, &surface, &form)?;
    fwd.embed(&tok_in, &Channel::from([0u32, 1]))?;
    fwd.state(&[&ws], StateGeometry::folding())?;
    fwd.epilogue(move || {
        let r = rng.take();
        let logits = intrinsics::output(0, "logits");
        let token = step(logits, temperature, &r);

        let r_next = &r + iota(2);

        tok_in.put(&token);
        tok_out.put(&token);
        rng.put(&r_next);
    });

    let mut generated: Vec<u32> = Vec::with_capacity(max_tokens);
    run_ahead(&pipe, &fwd, max_tokens, async || {
        let t = tok_out
            .take_host::<i32>()
            .await
            .map_err(|e| format!("tok_out.take @{}: {e}", generated.len()))?;
        generated.push(t as u32);
        Ok(ControlFlow::Continue(()))
    })
    .await?;
    pipe.close();

    let count = generated.len();
    Ok(report(wit_model::decode(&generated)?, count))
}
