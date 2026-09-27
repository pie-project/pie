//! Classifier-free guidance for language models (Sanchez et al., 2306.17806).
//!
//! Each decode fire runs the bound model over two rows with independent KV states: the
//! conditional stream sees the full prompt, the unconditional stream sees only
//! the negative prompt (empty by default). The paper's Eqn 4 is applied to
//! **log-probabilities**, not raw logits — the two streams have different
//! partition functions, so subtracting raw logits would inject a meaningless
//! per-stream constant:
//!
//! ```text
//! log P_cfg(w) ∝ log P(w | uncond) + γ · [log P(w | cond) − log P(w | uncond)]
//! ```
//!
//! and renormalised. γ = 1 collapses to plain conditional decoding, which the
//! reported `guidance_shift` statistic must confirm exactly.
//!
//! ## Source
//!
//! Sanchez et al., *Stay on topic with Classifier-Free Guidance* —
//! <https://arxiv.org/abs/2306.17806> (Eq. 7).
//!
//! Faithfulness: **Exact (equivalent form)** — the paper writes the rule over
//! log-probabilities and this works in logits, which differ by the per-stream
//! `logsumexp` constant. That constant is uniform over the vocabulary, so it
//! shifts every entry of the blended vector by the same amount and cancels in
//! the softmax. See
//! `inference-time-algorithms/10-implementation-faithfulness-audit.md`.

use inferlet::chat;
use inferlet::eta::hybrid::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Deserialize)]
struct Input {
    #[serde(default = "default_prompt")]
    prompt: String,
    #[serde(default)]
    negative_prompt: String,
    #[serde(default = "default_max_tokens")]
    max_tokens: usize,
    #[serde(default = "default_guidance")]
    guidance: f32,
}

fn default_prompt() -> String {
    "Explain why the sky appears blue.".into()
}

fn default_max_tokens() -> usize {
    32
}

fn default_guidance() -> f32 {
    1.5
}

/// The report the other sampler programs answer with (`tova-attention`,
/// `entropy-adaptive-temperature`, ...): the text, the counts, and what the
/// rule did. `#[inferlet::main]` serialises a struct with serde_json and
/// returns a `String` verbatim, so the prose this program used to return
/// ("<text>\n\n[cfg] guidance=...") reached `scripts/bench/pie_bench.py` as a
/// reply it could not parse -- "Expecting value" at the first character of the
/// continuation, or an unescaped control character once the continuation
/// happened to open with a quote.
#[derive(Serialize)]
struct Output {
    sampler: &'static str,
    /// The continuation, greedily decoded from the guided distribution.
    text: String,
    /// How many tokens it is.
    count: usize,
    /// The same tokens, for the harness's token-parity gate.
    token_ids: Vec<u32>,
    /// Conditional prompt tokens, chat-templated.
    prompt_len: u32,
    guidance: f32,
    /// Guided picks, the stop token included.
    steps: u64,
    /// Fraction of steps at which guidance moved the argmax.
    guidance_shift: f64,
    /// Mean KL(P_cfg || P_cond) over the steps, in nats.
    mean_kl: f64,
    /// At `guidance = 1` the rule is the identity, so a shift or a KL beyond
    /// float noise is a defect in this arithmetic or in the backend under it.
    identity_violation: bool,
}

/// Eqn 4 of 2306.17806 in the log-domain, plus the two diagnostics that prove
/// the γ = 1 identity: `shift` is 1 when guidance moved the argmax, and `kl` is
/// KL(P_cfg ‖ P_cond), which is exactly 0 at γ = 1.
fn guided_pick(
    cond_logits: &Tensor,
    uncond_logits: &Tensor,
    gamma: f32,
) -> (Tensor, Tensor, Tensor) {
    let cond = log_softmax(cond_logits);
    let uncond = log_softmax(uncond_logits);
    let guided = log_softmax(&uncond + (&cond - &uncond) * gamma);

    let token = reduce_argmax(&guided);
    let cond_token = reduce_argmax(&cond);
    let shift = cast(ne(&token, &cond_token), dtype::i32);
    let kl = reduce_sum(exp(&guided) * (&guided - &cond));
    (
        reshape(cast(&token, dtype::i32), [1]),
        reshape(shift, [1]),
        reshape(kl, [1]),
    )
}

/// One recurrent working set PER SEQUENCE on a hybrid model (the engine
/// requires one per request row, and guidance runs two rows: the conditional
/// stream and the unconditional one); none on a pure-attention model, where an
/// empty binding IS the attention pass. Guidance never buffers: every fire
/// folds straight into the recurrence, so the fold boundary is the fire's own
/// extent.
fn rs_for_sequence() -> Result<Vec<RsWorkingSet>> {
    match model::pass_kind() {
        model::ForwardKind::Attention => Ok(Vec::new()),
        model::ForwardKind::Hybrid => Ok(vec![RsWorkingSet::new()]),
        model::ForwardKind::Recurrent => Err(
            "classifier-free guidance has no recurrent-only path: it reads two KV streams".into(),
        ),
        model::ForwardKind::Diffusion => Err(
            "this program decodes a token at a time; a diffusion model wants a canvas loop".into(),
        ),
    }
}

#[inferlet::main]
async fn main(input: Input) -> Result<Output> {
    if !input.guidance.is_finite() || input.guidance < 0.0 {
        return Err("guidance must be finite and non-negative".into());
    }

    let max_tokens =
        u32::try_from(input.max_tokens).map_err(|_| "max_tokens exceeds the u32 range")?;
    let vocab = model::output_vocab_size();
    let gamma = input.guidance;
    let stop_tokens = chat::stop_tokens();
    let page_t = kv_page_size();

    let mut cond_prompt = chat::system_user("You are a helpful assistant.", &input.prompt);
    cond_prompt.extend(chat::cue());
    if cond_prompt.is_empty() {
        cond_prompt.push(0);
    }
    let nc = u32::try_from(cond_prompt.len()).map_err(|_| "prompt is too long")?;
    if input.max_tokens == 0 {
        return report(&[], nc, gamma, 0, 0, 0.0);
    }
    // The unconditional stream drops the conditioning text. With no negative
    // prompt it keeps only the chat scaffolding, which is the paper's
    // "unconditional" ∅ context for an instruction-tuned model.
    let mut uncond_prompt =
        chat::system_user("You are a helpful assistant.", &input.negative_prompt);
    uncond_prompt.extend(chat::cue());
    if uncond_prompt.is_empty() {
        uncond_prompt.push(0);
    }

    let nu = u32::try_from(uncond_prompt.len()).map_err(|_| "negative prompt is too long")?;
    // Both streams are rows of one working set: row 0 (unconditional) owns
    // pages [0, up), row 1 (conditional) owns [up, up + cp).
    let up = (nu + max_tokens).div_ceil(page_t);
    let cp = (nc + max_tokens).div_ceil(page_t);
    let ws = WorkingSet::new();
    ws.reserve(up + cp).context("reserve KV")?;
    let rs: Vec<RsWorkingSet> = rs_for_sequence()?
        .into_iter()
        .chain(rs_for_sequence()?)
        .collect();
    let pipeline = Pipeline::new();

    // Both prompts but their last tokens prefill as the two rows of one fire;
    // the last tokens are the first decode fire's input, so both logit rows
    // are born together.
    if nu < 2 || nc < 2 {
        return Err("each templated prompt must hold at least two tokens".into());
    }
    let (pu, pc) = (nu - 1, nc - 1);
    let toks: Vec<i32> = uncond_prompt[..pu as usize]
        .iter()
        .chain(&cond_prompt[..pc as usize])
        .map(|&t| t as i32)
        .collect();
    let rows = || (0..pu).map(|p| (0, p)).chain((0..pc).map(|p| (up, p)));
    let pre_toks = Channel::from(toks).named("prefill_tokens");
    let pre_indptr = Channel::from([0u32, pu, pu + pc]).named("prefill_embed_indptr");
    let pre_pages =
        Channel::from_iter((0..pu.div_ceil(page_t)).chain(up..up + pc.div_ceil(page_t)));
    let pre_page_indptr = Channel::from([
        0u32,
        pu.div_ceil(page_t),
        pu.div_ceil(page_t) + pc.div_ceil(page_t),
    ]);
    let pre_slot = Channel::from_iter(rows().map(|(base, p)| base + p / page_t));
    let pre_off = Channel::from_iter(rows().map(|(_, p)| p % page_t));
    let pre_pos = Channel::from_iter(rows().map(|(_, p)| p));
    let pre_klen = Channel::from([pu, pc]);
    let drop = Channel::new([1], dtype::i32).named("prefill_drop");
    let prefill = ForwardPass::new();
    prefill.embed(&pre_toks, &pre_indptr)?;
    prefill.attention(
        Some(KvBinding {
            working_set: &ws,
            geometry: KvGeometry {
                readable_pages: ..,
                writable_pages: ..,
                kv_len: &pre_klen,
                pages: &pre_pages,
                page_indptr: &pre_page_indptr,
                w_slot: &pre_slot,
                w_off: &pre_off,
                positions: &pre_pos,
                mask: None,
            },
        }),
        &rs,
        RsGeometry {
            fold_len: None,
            buffer: 0..0,
        },
    )?;
    // A pass needs an epilogue; this one's output is drained and dropped.
    prefill.epilogue(move || {
        let flat = reshape(intrinsics::logits(), [2 * vocab]);
        drop.put(reshape(cast(reduce_argmax(flat), dtype::i32), [1]));
    });
    prefill.submit(&pipeline).context("prefill")?;
    drop.take_host::<i32>().await?;

    // ---- decode: one two-row fire per token, device-resident -------------
    // Row 0 is the unconditional stream, row 1 the conditional one. The epilogue
    // mixes both logit rows on the device and feeds the guided token back to
    // both rows, so only the token and its two diagnostics cross to the host.
    let n_pages = up + cp;
    let live_pages = |pc0: u32, pc1: u32| -> (Vec<u32>, [u32; 3]) {
        let pages = (0..n_pages)
            .map(|i| if i < pc0 { i } else { i + up - pc0 })
            .collect();
        (pages, [0, pc0, pc0 + pc1])
    };
    let (pages0, indptr0) = live_pages(nu.div_ceil(page_t), nc.div_ceil(page_t));
    let toks = Channel::from([
        *uncond_prompt.last().unwrap() as i32,
        *cond_prompt.last().unwrap() as i32,
    ])
    .named("toks");
    let embed_indptr = Channel::from([0u32, 1, 2]).named("embed_indptr");
    let pos = Channel::from([nu - 1, nc - 1]).named("positions");
    let klen = Channel::from([nu, nc]).named("kv_len");
    let pages = Channel::from(pages0).named("pages");
    let page_indptr = Channel::from(indptr0).named("page_indptr");
    let w_slot = Channel::from([(nu - 1) / page_t, up + (nc - 1) / page_t]).named("w_slot");
    let w_off = Channel::from([(nu - 1) % page_t, (nc - 1) % page_t]).named("w_off");
    let out = Channel::new([3], dtype::f32)
        .capacity(channel_capacity() as u32)
        .named("out");

    let decode = ForwardPass::new();
    decode.embed(&toks, &embed_indptr)?;
    decode.attention(
        Some(KvBinding {
            working_set: &ws,
            geometry: KvGeometry {
                readable_pages: ..,
                writable_pages: ..,
                kv_len: &klen,
                pages: &pages,
                page_indptr: &page_indptr,
                w_slot: &w_slot,
                w_off: &w_off,
                positions: &pos,
                mask: None,
            },
        }),
        &rs,
        RsGeometry {
            fold_len: None,
            buffer: 0..0,
        },
    )?;
    decode.epilogue(move || {
        let logits = reshape(intrinsics::logits(), [2, vocab]);
        let uncond = reshape(gather(&logits, iota(1)), [vocab]);
        let cond = reshape(gather(&logits, iota(1) + 1u32), [vocab]);
        let (token, shift, kl) = guided_pick(&cond, &uncond, gamma);

        let length = klen.take();
        let next_length = &length + 1u32;
        let base = iota(2) * up;
        toks.put(broadcast(&token, [2]));
        pos.put(&length);
        klen.put(&next_length);
        w_slot.put(&base + &length / page_t);
        w_off.put(&length % page_t);

        let page_count = next_length.div_ceil(page_t);
        let pc0 = broadcast(gather(&page_count, iota(1)), [n_pages]);
        let i = iota(n_pages);
        pages.put(select(lt(&i, &pc0), &i, &i + up - &pc0));
        let k = iota(3);
        let prefix = gather(cumsum(&page_count), max_elem(&k, 1u32) - 1u32);
        page_indptr.put(&prefix * cast(gt(&k, 0u32), dtype::u32));

        let token_f = broadcast(cast(&token, dtype::f32), [3]);
        let shift_f = broadcast(cast(&shift, dtype::f32), [3]);
        let kl_b = broadcast(&kl, [3]);
        out.put(select(
            eq(&k, 0u32),
            &token_f,
            select(eq(&k, 1u32), &shift_f, &kl_b),
        ));
    });

    let mut generated = Vec::with_capacity(input.max_tokens);
    let mut shifts = 0u64;
    let mut kl_total = 0f64;
    let mut scored = 0u64;
    run_ahead(&pipeline, &decode, input.max_tokens, async || {
        let row = out.take_host::<Vec<f32>>().await?;
        let token = row[0] as u32;
        shifts += row[1] as u64;
        kl_total += row[2] as f64;
        scored += 1;
        if stop_tokens.contains(&token) {
            return Ok(ControlFlow::Break(()));
        }
        generated.push(token);
        Ok(ControlFlow::Continue(()))
    })
    .await?;

    report(&generated, nc, gamma, shifts, scored, kl_total)
}

fn report(
    generated: &[u32],
    prompt_len: u32,
    gamma: f32,
    shifts: u64,
    scored: u64,
    kl_total: f64,
) -> Result<Output> {
    let text = model::decode(generated)?;
    let mean_kl = if scored == 0 {
        0.0
    } else {
        kl_total / scored as f64
    };
    // KL(P||Q) is non-negative by Jensen, so a negative one is float error in
    // the sum and not a measurement. At gamma = 1 the guided log-probs equal
    // the conditional ones up to rounding, every term is that rounding, and
    // the total lands either side of zero at around 1e-9 -- which printed as
    // `mean_kl=-0.0000` and made the sign bit of a value indistinguishable
    // from zero decide whether the identity looked like it held.
    //
    // Clamped only inside the noise. A mean KL genuinely below -1e-6 is a
    // defect in this arithmetic or in the backend under it, and stays visible
    // and negative so that it can be seen.
    let mean_kl = if (-1e-6..0.0).contains(&mean_kl) {
        0.0
    } else {
        mean_kl
    };
    let identity_violation = (gamma - 1.0).abs() < 1e-6 && (shifts > 0 || mean_kl > 1e-3);
    Ok(Output {
        sampler: "classifier-free-guidance",
        text,
        count: generated.len(),
        token_ids: generated.to_vec(),
        prompt_len,
        guidance: gamma,
        steps: scored,
        guidance_shift: shifts as f64 / scored.max(1) as f64,
        mean_kl,
        identity_violation,
    })
}
