//! The generation loop: a one-pass prefill, then a device-carried decode pass
//! whose epilogue samples the next token and feeds it straight back into the
//! channel `embed` reads. The host drains a mirror of the sampled tokens
//! behind the runtime's run-ahead window.
//!
//! The sampler is one ETA epilogue: the OpenAI penalties (presence,
//! frequency, and the multiplicative repetition penalty) over a vocabulary
//! histogram carried on the device, an optional grammar mask, temperature,
//! then top-k / top-p truncation and a Gumbel-max draw. Temperature zero is
//! exact greedy.
//!
//! With a grammar the loop is depth-1 — the mask for fire k+1 exists only
//! once fire k's token has advanced the matcher — so constrained generation
//! submits one fire at a time; unconstrained generation runs ahead.

use std::ops::ControlFlow;

use inferlet::eta::prelude::*;
use inferlet::grammar::Matcher;
use inferlet::mask::unpack_mask;
use inferlet::media;

use crate::prompt::Prompt;

/// Sampling parameters. The defaults are the OpenAI defaults: temperature 1,
/// no truncation, no penalties.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Sampling {
    pub speculative: bool,
    pub temperature: f32,
    pub top_p: f32,
    /// 0 means off.
    pub top_k: u32,
    pub seed: u32,
    pub presence_penalty: f32,
    pub frequency_penalty: f32,
    /// 1.0 means off.
    pub repetition_penalty: f32,
}

impl Default for Sampling {
    fn default() -> Self {
        Self {
            speculative: false,
            temperature: 1.0,
            top_p: 1.0,
            top_k: 0,
            seed: 0x51ed,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
            repetition_penalty: 1.0,
        }
    }
}

impl Sampling {
    /// The API's 64-bit seed folded into the sampler's 32-bit state.
    pub fn set_seed(&mut self, seed: i64) {
        self.seed = (seed as u64 as u32) ^ ((seed as u64 >> 32) as u32);
    }

    fn penalized(&self) -> bool {
        self.presence_penalty != 0.0
            || self.frequency_penalty != 0.0
            || self.repetition_penalty != 1.0
    }

    pub fn validate(&self) -> Result<()> {
        if !self.temperature.is_finite() || self.temperature < 0.0 {
            return Err("temperature must be a finite number >= 0".into());
        }
        if !(self.top_p > 0.0 && self.top_p <= 1.0) {
            return Err("top_p must be in (0, 1]".into());
        }
        if !self.presence_penalty.is_finite() || !self.frequency_penalty.is_finite() {
            return Err("penalties must be finite".into());
        }
        if !(self.repetition_penalty.is_finite() && self.repetition_penalty > 0.0) {
            return Err("repetition_penalty must be > 0".into());
        }
        Ok(())
    }
}

/// The sampled token as a rank-0 i32, and the next output-token histogram.
/// `histogram` is `(counts, present)`: how often each token was generated,
/// and whether it was in the prompt.
fn sample(
    vocab: u32,
    s: Sampling,
    r: &Tensor,
    histogram: Option<(&Tensor, &Tensor)>,
    mask: Option<&Tensor>,
) -> (Tensor, Option<Tensor>) {
    let mut logits = reshape(intrinsics::logits(), [vocab]);
    if let Some((counts, present)) = histogram {
        let zero = broadcast(0.0f32, [vocab]);
        let seen_out = gt(counts, &zero);
        let seen_prompt = gt(present, broadcast(0.5f32, [vocab]));
        let seen = or(&seen_out, &seen_prompt);
        if s.repetition_penalty != 1.0 {
            let rp = broadcast(s.repetition_penalty, [vocab]);
            let positive = gt(&logits, &zero);
            let repenalized = select(&positive, &logits / &rp, &logits * &rp);
            logits = select(&seen, &repenalized, &logits);
        }
        if s.frequency_penalty != 0.0 {
            logits = &logits - broadcast(s.frequency_penalty, [vocab]) * counts;
        }
        if s.presence_penalty != 0.0 {
            logits = &logits - broadcast(s.presence_penalty, [vocab]) * cast(&seen_out, dtype::f32);
        }
    }
    if let Some(mask) = mask {
        logits = mask_apply(&logits, mask);
    }
    let token = if s.temperature <= 0.0 {
        reduce_argmax(&logits)
    } else {
        let scaled = &logits / s.temperature.max(1e-4);
        let mut keep: Option<Tensor> = None;
        if s.top_k > 0 || s.top_p < 1.0 {
            let probs = softmax(&scaled);
            if s.top_k > 0 {
                keep = Some(pivot_threshold(&probs, rank_le(s.top_k)));
            }
            if s.top_p < 1.0 {
                let nucleus = pivot_threshold(&probs, cummass_le(s.top_p));
                keep = Some(match keep {
                    Some(k) => and(&k, &nucleus),
                    None => nucleus,
                });
            }
        }
        let truncated = match keep {
            Some(keep) => mask_apply(&scaled, keep),
            None => scaled,
        };
        gumbel_max(&truncated, r)
    };
    let next = histogram.map(|(counts, _)| scatter_add(counts, &token, 1.0f32));
    (token, next)
}

/// Generate up to `max_tokens` after `prompt`; each picture's span is
/// prefilled whole in a fire of its own. Every sampled token — the
/// stop token included, if one comes — goes to `on_token`, which ends
/// generation by returning `Break`; the return value is how many tokens
/// were sampled, and how many of the prompt's were mapped in from the
/// prefix cache.
///
/// With a `matcher`, every fire's logits are masked to what the grammar
/// allows next, and generation ends when the grammar terminates.
pub async fn generate(
    prompt: &Prompt,
    sampling: Sampling,
    max_tokens: usize,
    matcher: Option<&Matcher>,
    on_token: &mut dyn FnMut(u32) -> ControlFlow<()>,
) -> Result<(usize, usize)> {
    sampling.validate()?;
    let images = &prompt.images;
    let ends = &prompt.ends;
    let prompt = &prompt.tokens[..];
    if max_tokens == 0 {
        return Ok((0, 0));
    }
    if sampling.speculative
        && sampling.temperature == 0.0
        && !sampling.penalized()
        && matcher.is_none()
        && !prompt.is_empty()
        && images.is_empty()
        && let Some(draft) = model::draft_block()
        && !draft.bidirectional
        && draft.proposals_from == 1
        && (2..=16).contains(&draft.rows)
    {
        return crate::speculative::generate(prompt, ends, max_tokens, draft, on_token).await;
    }
    if model::pass_kind() == model::ForwardKind::Diffusion {
        return Err(
            "this program decodes a token at a time; a diffusion model wants a canvas loop".into(),
        );
    }
    let vocab = model::output_vocab_size();
    let n = prompt.len() as u32;

    let ws = WorkingSet::new();
    let pipe = Pipeline::new();

    let penalized = sampling.penalized();
    let present: Vec<f32> = if penalized {
        let mut v = vec![0.0f32; vocab as usize];
        for &t in prompt {
            if (t as usize) < v.len() {
                v[t as usize] = 1.0;
            }
        }
        v
    } else {
        Vec::new()
    };

    // Rotations past an M-RoPE span lag its rows by `token_count - position_span`.
    let runs: Vec<(u32, u32, u32)> = images
        .iter()
        .map(|(start, image)| {
            let end = start + image.tokens().len() as u32;
            (*start, end, image.token_count() - image.position_span())
        })
        .collect();
    let rope_delta: u32 = runs.iter().map(|r| r.2).sum();
    let mut spans = Vec::new();
    let mut at = 0;
    for &(start, end, _) in runs.iter().chain([(n, n, 0)].iter()) {
        if start > at {
            spans.push((at, start));
        }
        if end > start {
            spans.push((start, end));
        }
        at = end;
    }

    let prompt_i32: Vec<i32> = prompt.iter().map(|&t| t as i32).collect();
    let mut g0 = 0i32;
    for (k, &(base, end)) in spans.iter().enumerate() {
        let len = end - base;
        let lag: u32 = runs.iter().filter(|r| r.1 <= base).map(|r| r.2).sum();
        let carried = images.iter().position(|(start, _)| *start == base);
        let toks = Channel::from(&prompt_i32[base as usize..end as usize]).named("toks_p");
        let embed_indptr = Channel::from([0u32, len]).named("embed_indptr_p");
        // Past an image's M-RoPE span, rows are roped behind their slots.
        let positions = (lag > 0)
            .then(|| Channel::from_iter((base..end).map(|row| row - lag)).named("positions_p"));
        let rng = Channel::from([sampling.seed, k as u32]).named("rng_p");
        let hist = penalized.then(|| {
            (
                Channel::from(vec![0.0f32; vocab as usize]).named("counts_p"),
                Channel::from(present.clone()).named("present_p"),
            )
        });
        let mask = matcher.map(|_| Channel::new([vocab], dtype::bool).named("mask_p"));
        let g0_ch = Channel::new([1], dtype::i32).named("g0");

        let fwd = ForwardPass::new();
        fwd.embed(&toks, &embed_indptr)?;
        if let Some(at) = carried {
            fwd.media(&[media::Span::Image(&images[at].1)])?;
        }
        fwd.state(
            &[&ws],
            StateGeometry {
                positions: positions.as_ref(),
                ..StateGeometry::folding()
            },
        )?;
        fwd.epilogue(move || {
            let r = rng.take();
            let taken = hist.map(|(c, p)| (c.take(), p.take()));
            let m = mask.map(|m| m.take());
            let (token, _) = sample(
                vocab,
                sampling,
                &r,
                taken.as_ref().map(|(c, p)| (c, p)),
                m.as_ref(),
            );
            g0_ch.put(reshape(token, [1]));
        });
        if let (Some(m), Some(matcher)) = (mask, matcher) {
            m.put(unpack_mask(&matcher.mask(), vocab));
        }
        // Checkpoints only where messages end: each costs a state slot.
        fwd.checkpoints(ends)?;
        fwd.submit(&pipe)
            .with_context(|| format!("prefill submit @{base}"))?;
        g0 = g0_ch
            .take_host::<i32>()
            .await
            .with_context(|| format!("prefill drain @{base}"))?;
    }
    let cached = ws.cached_tokens();

    let mut sampled = 1usize;
    if let Some(matcher) = matcher {
        matcher
            .accept_tokens(&[g0 as u32])
            .context("advance grammar")?;
    }
    if on_token(g0 as u32).is_break()
        || max_tokens == 1
        || matcher.is_some_and(|m| m.is_terminated())
    {
        pipe.close();
        return Ok((sampled, cached as usize));
    }

    let tok_in = Channel::from([g0]).named("tok_in");
    let pos = (rope_delta > 0).then(|| {
        (
            Channel::from([n - rope_delta]).named("pos"),
            Channel::from([n - rope_delta + 1]).named("next_pos"),
        )
    });
    let out = Channel::new([1], dtype::i32)
        .capacity(channel_capacity() as u32)
        .named("out");
    let rng = Channel::from([sampling.seed ^ 0x9e37, 1u32]).named("rng");
    let lane1 = Channel::from([0u32, 1u32]).named("embed_indptr");
    let hist = penalized.then(|| {
        let mut counts0 = vec![0.0f32; vocab as usize];
        if let Ok(g0) = usize::try_from(g0)
            && g0 < counts0.len()
        {
            counts0[g0] = 1.0;
        }
        (
            Channel::from(counts0).named("counts"),
            Channel::from(present.clone()).named("present"),
        )
    });
    let mask = matcher.map(|_| Channel::new([vocab], dtype::bool).named("mask"));

    let fwd = ForwardPass::new();
    fwd.embed(&tok_in, &lane1)?;
    fwd.state(
        &[&ws],
        StateGeometry {
            positions: pos.as_ref().map(|(pos, _)| pos),
            ..StateGeometry::folding()
        },
    )?;
    fwd.epilogue(move || {
        let r = rng.take();
        let taken = hist.map(|(c, p)| (c.take(), p.take()));
        let m = mask.map(|m| m.take());

        let (token, counts_next) = sample(
            vocab,
            sampling,
            &r,
            taken.as_ref().map(|(c, p)| (c, p)),
            m.as_ref(),
        );
        let token = reshape(token, [1]);
        let r_next = &r + iota(2);

        tok_in.put(&token);
        out.put(&token);
        if let Some((pos, next)) = pos {
            let at = next.take();
            pos.put(&at);
            next.put(&(&at + 1u32));
        }
        rng.put(&r_next);
        if let (Some((c, p)), Some(next), Some((_, present))) = (hist, counts_next, taken) {
            c.put(&next);
            p.put(&present);
        }
    });

    let budget = max_tokens - 1;
    match (matcher, mask) {
        (Some(matcher), Some(mask)) => {
            let mut fired = 0usize;
            while fired < budget {
                mask.put(unpack_mask(&matcher.mask(), vocab));
                fwd.submit(&pipe).context("constrained decode submit")?;
                fired += 1;
                let t = out.take_host::<Vec<i32>>().await?;
                let token = *t.first().unwrap_or(&0) as u32;
                sampled += 1;
                matcher.accept_tokens(&[token]).context("advance grammar")?;
                if on_token(token).is_break() || matcher.is_terminated() {
                    break;
                }
            }
        }
        _ => {
            run_ahead(&pipe, &fwd, budget, async || {
                let t = out.take_host::<Vec<i32>>().await?;
                let token = *t.first().unwrap_or(&0) as u32;
                sampled += 1;
                Ok(on_token(token))
            })
            .await?;
        }
    }
    pipe.close();
    Ok((sampled, cached as usize))
}
