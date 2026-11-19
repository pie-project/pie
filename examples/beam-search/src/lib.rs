//! On-device beam search over per-beam page lists.
//!
//! The prompt — chat-templated, or `prompt_tokens` as given — is every beam's
//! common ancestry. It is prefilled ONCE into a working set and folded, and
//! that fire's epilogue picks the first `beams` tokens (the top of the
//! prompt's next-token distribution), as greedy decoding picks its first.
//!
//! The beams then share that working set's frozen prompt, and each keeps its
//! own list of the pages its generated slots lie in (`inferlet::beam`): a
//! beam's rows read the prompt and its own pages only, a read extent of its
//! own length, with no mask. Every decode fire's epilogue picks the next
//! `beams` (parent, token) pairs and reorders the lists by the parents
//! (`BeamGroup::reorder`): a child shares its parent's full pages; the
//! parent's first child keeps its tail page, every other child has the tail
//! copied into a free page before the next fire appends. Pages no beam reads
//! any more are freed a fire late.
//!
//! The search runs `max_tokens` steps: a beam is not retired at end of
//! sequence, so width 1 is greedy decoding of the same prompt to the same
//! length (the beam identity, checked every step against the raw argmax).

use inferlet::beam::BeamGroup;
use inferlet::chat;
use inferlet::eta::prelude::*;
use serde::Deserialize;

#[derive(Deserialize)]
struct Input {
    #[serde(default = "default_prompt")]
    prompt: String,
    /// The prompt as token ids, in place of the chat-templated `prompt`.
    #[serde(default)]
    prompt_tokens: Option<Vec<u32>>,
    #[serde(default = "default_system")]
    system: String,
    /// Turn a thinking model's thinking off, as the model's own template does
    /// for `enable_thinking: false`.
    #[serde(default)]
    no_think: bool,
    #[serde(default = "default_max_tokens")]
    max_tokens: usize,
    /// Beam width. `1` degenerates to greedy decoding (the beam identity).
    #[serde(default = "default_beams")]
    beams: u32,
    #[serde(default)]
    ckpt_sync: bool,
    #[serde(default)]
    rollback_mode: Option<String>,
    /// Rescore each final hypothesis as a lone sequence, one token a fire on
    /// a working set of its own, and require its summed log-probability to
    /// equal its beam's score: each beam read its own ancestry and nothing of
    /// its siblings'.
    #[serde(default)]
    verify: bool,
}

fn default_prompt() -> String {
    "Write a short paragraph about beam search.".into()
}

fn default_system() -> String {
    "You are a helpful, respectful and honest assistant.".into()
}

fn default_max_tokens() -> usize {
    16
}

fn default_beams() -> u32 {
    2
}

fn advance_hypotheses(
    hypotheses: &[Vec<u32>],
    picked: &[i32],
    parents: &[u32],
) -> Result<Vec<Vec<u32>>> {
    parents
        .iter()
        .zip(picked)
        .map(|(&parent, &token)| {
            let mut hypothesis = hypotheses
                .get(parent as usize)
                .ok_or_else(|| format!("invalid parent beam {parent}"))?
                .clone();
            hypothesis.push(token as u32);
            Ok(hypothesis)
        })
        .collect()
}

#[inferlet::main]
async fn main(input: Input) -> Result<String> {
    let max_steps = input.max_tokens;
    let b = input.beams;
    if b == 0 {
        return Err("beams must be at least 1".into());
    }
    if max_steps == 0 {
        return Ok(String::new());
    }
    let v = model::output_vocab_size();
    if b > v {
        return Err(format!("beams ({b}) exceeds the vocabulary ({v})"));
    }

    let prompt: Vec<u32> = match &input.prompt_tokens {
        Some(tokens) => tokens.clone(),
        None => {
            let mut p = chat::system_user(&input.system, &input.prompt);
            p.extend(chat::cue_thinking(!input.no_think));
            p
        }
    };
    if prompt.is_empty() {
        return Err("empty prompt".into());
    }
    let n = u32::try_from(prompt.len()).map_err(|_| "the prompt exceeds the u32 range")?;
    let decodes = max_steps - 1;
    let ws = WorkingSet::new();

    let pipeline = Pipeline::new();

    // Step 0: the prompt's own next-token distribution. Every beam is the
    // prompt, so the first `b` tokens are its top `b`.
    let prompt_i32: Vec<i32> = prompt.iter().map(|&t| t as i32).collect();
    let toks_p = Channel::from(&prompt_i32[..]);
    let embed_indptr_p = Channel::from([0u32, n]);
    let out_p = Channel::new([b], dtype::i32);
    let scr_p = Channel::new([b], dtype::f32);
    let greedy_p = Channel::new([1], dtype::i32);
    let fwd_p = ForwardPass::new();
    fwd_p.embed(&toks_p, &embed_indptr_p)?;
    fwd_p
        .state(&[&ws], StateGeometry::folding())
        .context("bind the prompt prefill")?;
    fwd_p.epilogue(move || {
        let logits = reshape(intrinsics::output(0, "logits"), [v]);
        let (s, i) = top_k(log_softmax(&logits), b);
        out_p.put(cast(&i, dtype::i32));
        scr_p.put(&s);
        greedy_p.put(reshape(reduce_argmax(&logits), [1]));
    });
    fwd_p.submit(&pipeline).context("prefill submit")?;
    let first = out_p
        .take_host::<Vec<i32>>()
        .await
        .context("prefill drain")?;
    let mut final_scores = scr_p
        .take_host::<Vec<f32>>()
        .await
        .context("prefill drain")?;
    let greedy = greedy_p
        .take_host::<Vec<i32>>()
        .await
        .context("prefill drain")?;
    let mut greedy_mismatches = usize::from(first.first() != greedy.first());
    let mut hypotheses: Vec<Vec<u32>> = first.iter().map(|&token| vec![token as u32]).collect();

    if decodes > 0 {
        let scores = Channel::from(&final_scores[..]);
        let toks = Channel::from(&first[..]);
        let out = Channel::new([b], dtype::i32).capacity(channel_capacity() as u32);
        let out_par = Channel::new([b], dtype::u32).capacity(channel_capacity() as u32);
        let out_scr = Channel::new([b], dtype::f32).capacity(channel_capacity() as u32);
        let out_greedy = Channel::new([b], dtype::i32).capacity(channel_capacity() as u32);
        let lanes_b = Channel::from_iter(0u32..=b);

        let fwd = ForwardPass::new();
        fwd.embed(&toks, &lanes_b)?;
        let pool: Vec<&WorkingSet> = vec![&ws; b as usize];
        // Every beam reads the folded prompt and its own pages: beam `i`'s
        // first decode writes slot `n` on its own first page.
        let decodes_u32 = u32::try_from(decodes).map_err(|_| "max_tokens exceeds u32")?;
        let group = BeamGroup::new(&ws, b, n, decodes_u32).context("reserve the beams' pages")?;

        fwd.state(&pool, group.geometry())
            .context("bind the beams' page lists")?;
        fwd.epilogue(move || {
            // `logits()` squeezes a one-row fire to `[v]`: reshape so `beams == 1` meets `[B, 1]` scores.
            let logits = reshape(intrinsics::output(0, "logits"), [b, v]);
            let cand = broadcast(reshape(scores.take(), [b, 1]), [b, v]) + log_softmax(&logits);
            let (s, i) = top_k(reshape(cand, [b * v]), b);
            let parent = &i / v;
            let tok_i = cast(&i % v, dtype::i32);

            scores.put(&s);
            toks.put(&tok_i);
            out.put(&tok_i);
            out_par.put(&parent);
            out_scr.put(&s);
            // Cross-check: at `beams == 1` the raw-logit argmax must equal the beam pick every step.
            out_greedy.put(reshape(reduce_argmax(&logits), [b]));

            group.reorder(&cast(&parent, dtype::u32));
        });

        let mut step = 0usize;
        let mode = input
            .rollback_mode
            .as_deref()
            .unwrap_or(if input.ckpt_sync {
                "ckpt_sync"
            } else {
                "flush_free"
            });
        let ckpt_sync = mode == "ckpt_sync";
        let stop_wait = mode == "stop_wait";
        let mut held_ckpt = if ckpt_sync || stop_wait {
            ws.fork(&pipeline).ok()
        } else {
            None
        };
        if ckpt_sync || stop_wait {
            for _ in 0..decodes {
                if let Ok(c) = ws.fork(&pipeline) {
                    held_ckpt = Some(c);
                }
                fwd.submit(&pipeline).with_context(|| format!("@{step}"))?;
                let picked = out
                    .take_host::<Vec<i32>>()
                    .await
                    .with_context(|| format!("@{step}"))?;
                let parents = out_par
                    .take_host::<Vec<u32>>()
                    .await
                    .with_context(|| format!("@{step}"))?;
                final_scores = out_scr
                    .take_host::<Vec<f32>>()
                    .await
                    .with_context(|| format!("@{step}"))?;
                let greedy = out_greedy
                    .take_host::<Vec<i32>>()
                    .await
                    .with_context(|| format!("@{step}"))?;
                if stop_wait || parents.iter().enumerate().any(|(i, &p)| p as usize != i) {
                    if let Some(ref ckpt_ws) = held_ckpt {
                        if let Ok(restore_ws) = ckpt_ws.fork(&pipeline) {
                            let toks_r = Channel::from(&picked[..1]);
                            let ind_r = Channel::from([0u32, 1]);
                            let out_r = Channel::new([1], dtype::i32);
                            let fwd_r = ForwardPass::new();
                            fwd_r.embed(&toks_r, &ind_r)?;
                            fwd_r.state(&[&restore_ws], StateGeometry::folding())?;
                            fwd_r.epilogue(move || {
                                out_r.put(reshape(
                                    reduce_argmax(intrinsics::output(0, "logits")),
                                    [1],
                                ));
                            });
                            fwd_r.submit(&pipeline)?;
                            let _ = out_r.take_host::<Vec<i32>>().await?;
                        }
                    }
                }
                greedy_mismatches += picked.iter().zip(&greedy).filter(|(p, g)| p != g).count();
                hypotheses = advance_hypotheses(&hypotheses, &picked, &parents)?;
                step += 1;
            }
        } else {
            run_ahead(&pipeline, &fwd, decodes, async || {
                let picked = out
                    .take_host::<Vec<i32>>()
                    .await
                    .with_context(|| format!("@{step}"))?;
                let parents = out_par
                    .take_host::<Vec<u32>>()
                    .await
                    .with_context(|| format!("@{step}"))?;

                final_scores = out_scr
                    .take_host::<Vec<f32>>()
                    .await
                    .with_context(|| format!("@{step}"))?;
                let greedy = out_greedy
                    .take_host::<Vec<i32>>()
                    .await
                    .with_context(|| format!("@{step}"))?;
                greedy_mismatches += picked.iter().zip(&greedy).filter(|(p, g)| p != g).count();
                hypotheses = advance_hypotheses(&hypotheses, &picked, &parents)?;
                step += 1;
                Ok(ControlFlow::Continue(()))
            })
            .await?;
        }
    }
    pipeline.close();

    let best_lane = final_scores
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.total_cmp(b))
        .map(|(index, _)| index)
        .ok_or("beam search produced no hypotheses")?;
    // The beam identity is only an identity at width 1; at B > 1 the top-B
    // block legitimately picks non-argmax continuations for the lower lanes.
    if b == 1 && greedy_mismatches != 0 {
        return Err(format!(
            "beam identity violated: width-1 beam search disagreed with greedy \
                 argmax on {greedy_mismatches} of {max_steps} steps"
        ));
    }
    let text = model::decode(&hypotheses[best_lane])?;
    let verified = if input.verify {
        let mut worst = 0f32;
        for (lane, hypothesis) in hypotheses.iter().enumerate() {
            let alone = rescore(&prompt, hypothesis).await?;
            let apart = (alone - final_scores[lane]).abs();
            // bf16 logits: a step's log-probability rounds to about 1e-2.
            let room = 0.02 * hypothesis.len() as f32 + 0.05;
            if apart > room {
                return Err(format!(
                    "beam {lane} scored {:.4}, and its hypothesis alone scores {alone:.4}: \
                     {apart:.4} apart, past {room:.3}; its recurrence or attention read \
                     another beam's cells",
                    final_scores[lane]
                ));
            }
            worst = worst.max(apart);
        }
        format!(" verified={} worst_apart={worst:.4}", hypotheses.len())
    } else {
        String::new()
    };
    Ok(format!(
        "{text}\n[beam] width={b} steps={max_steps} best_score={:.4} greedy_mismatches={greedy_mismatches}{verified}",
        final_scores[best_lane]
    ))
}

/// The summed log-probability of `hypothesis` after `prompt`, teacher-forced
/// one token a fire on a working set of its own: what a beam that read only
/// its own ancestry scores.
async fn rescore(prompt: &[u32], hypothesis: &[u32]) -> Result<f32> {
    let v = model::output_vocab_size();
    let ws = WorkingSet::new();
    let pipeline = Pipeline::new();
    let mut total = 0f32;
    for (at, &target) in hypothesis.iter().enumerate() {
        let fed: Vec<i32> = if at == 0 {
            prompt.iter().map(|&t| t as i32).collect()
        } else {
            vec![hypothesis[at - 1] as i32]
        };
        let rows = u32::try_from(fed.len()).map_err(|_| "a prompt past u32 rows")?;
        let toks = Channel::from(&fed[..]);
        let indptr = Channel::from([0u32, rows]);
        let wanted = Channel::from([target]);
        let out = Channel::new([1], dtype::f32);
        let fwd = ForwardPass::new();
        fwd.embed(&toks, &indptr)?;
        fwd.state(&[&ws], StateGeometry::folding())
            .context("bind the rescoring set")?;
        fwd.epilogue(move || {
            let logits = reshape(intrinsics::output(0, "logits"), [v]);
            out.put(gather(log_softmax(&logits), wanted.take()));
        });
        fwd.submit(&pipeline).context("rescore submit")?;
        let got = out.take_host::<Vec<f32>>().await.context("rescore drain")?;
        total += got.first().copied().ok_or("the rescoring fire returned nothing")?;
    }
    pipeline.close();
    Ok(total)
}
