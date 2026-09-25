//! The generation loop: chunked prefill, then a device-carried decode pass
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

use inferlet::eta::hybrid::prelude::*;
use inferlet::grammar::Matcher;
use inferlet::mask::unpack_mask;

/// Sampling parameters. The defaults are the OpenAI defaults: temperature 1,
/// no truncation, no penalties.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Sampling {
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

/// Generate up to `max_tokens` after `prompt`. Every sampled token — the
/// stop token included, if one comes — goes to `on_token`, which ends
/// generation by returning `Break`; the return value is how many tokens
/// were sampled.
///
/// With a `matcher`, every fire's logits are masked to what the grammar
/// allows next, and generation ends when the grammar terminates.
pub async fn generate(
    prompt: &[u32],
    sampling: Sampling,
    max_tokens: usize,
    matcher: Option<&Matcher>,
    on_token: &mut dyn FnMut(u32) -> ControlFlow<()>,
) -> Result<usize> {
    sampling.validate()?;
    if max_tokens == 0 {
        return Ok(0);
    }
    let rs_ws: Vec<RsWorkingSet> = match model::pass_kind() {
        model::ForwardKind::Attention => Vec::new(),
        model::ForwardKind::Hybrid => vec![RsWorkingSet::new()],
        model::ForwardKind::Recurrent => {
            return Err("this program has no recurrent-only path (it needs a KV cache)".into());
        }
        model::ForwardKind::Diffusion => {
            return Err(
                "this program decodes a token at a time; a diffusion model wants a canvas loop"
                    .into(),
            );
        }
    };
    let vocab = model::output_vocab_size();
    let page_t = kv_page_size();
    let n = prompt.len() as u32;
    let pool_pages = (n + max_tokens as u32 + 2).div_ceil(page_t);
    let ws = WorkingSet::new();
    let slots = ws.reserve(pool_pages).context("reserve KV pages")?;
    let pool_ids = slots.ids().to_vec();
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

    let prompt_i32: Vec<i32> = prompt.iter().map(|&t| t as i32).collect();
    let mut g0 = 0i32;
    for (k, &(base, end)) in prefill_chunks(n, None).iter().enumerate() {
        let len = end - base;
        let toks = Channel::from(&prompt_i32[base as usize..end as usize]).named("toks_p");
        let embed_indptr = Channel::from([0u32, len]).named("embed_indptr_p");
        let positions = Channel::from_iter(base..end).named("positions_p");
        let w_slot = Channel::from_iter((base..end).map(|c| pool_ids[(c / page_t) as usize]))
            .named("w_slot_p");
        let w_off = Channel::from_iter((base..end).map(|c| c % page_t)).named("w_off_p");
        let klen = Channel::from([end]).named("klen_p");
        let pages = Channel::from(pool_ids.clone()).named("pages_p");
        let page_indptr = Channel::from([0u32, end.div_ceil(page_t)]).named("pidx_p");
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
        fwd.attention(
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
        fwd.submit(&pipe)
            .with_context(|| format!("prefill submit @{base}"))?;
        g0 = g0_ch
            .take_host::<i32>()
            .await
            .with_context(|| format!("prefill drain @{base}"))?;
    }

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
        return Ok(sampled);
    }

    let slot_n = pool_ids[(n / page_t) as usize];
    let tok_in = Channel::from([g0]).named("tok_in");
    let pos = Channel::from([n]).named("pos");
    let fill = Channel::from([n + 1]).named("fill");
    let klen = Channel::from([n + 1]).named("klen");
    let w_slot = Channel::from([slot_n]).named("w_slot");
    let w_off = Channel::from([n % page_t]).named("w_off");
    let pages = Channel::from(pool_ids.clone()).named("pages");
    let page_indptr = Channel::from([0u32, (n + 1).div_ceil(page_t)]).named("page_indptr");
    let pool_ids_ch = Channel::from(pool_ids.clone()).named("pool_ids");
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
    fwd.attention(
        Some(KvBinding {
            working_set: &ws,
            geometry: KvGeometry {
                readable_pages: ..,
                writable_pages: (n / page_t)..,
                kv_len: &klen,
                pages: &pages,
                page_indptr: &page_indptr,
                w_slot: &w_slot,
                w_off: &w_off,
                positions: &pos,
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
        let base = fill.take(); // [1] u32 — the position this fire writes
        let pids = pool_ids_ch.take();
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

        let logical_slot = &base / page_t;
        let w_slot_v = gather(&pids, &logical_slot);
        let w_off_v = &base % page_t;
        let klen_v = &base + 1u32;
        let next_free = &base + 1u32;
        let pages_v = reshape(&pids, [pool_pages]);
        let page_count = klen_v.div_ceil(page_t);
        let pidx_v = indptr(1, &page_count);

        tok_in.put(&token);
        out.put(&token);
        w_slot.put(&w_slot_v);
        w_off.put(&w_off_v);
        klen.put(&klen_v);
        pos.put(&base);
        fill.put(&next_free);
        pages.put(&pages_v);
        page_indptr.put(&pidx_v);
        rng.put(&r_next);
        pool_ids_ch.put(&pids);
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
    Ok(sampled)
}
