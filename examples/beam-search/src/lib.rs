//! On-device beam search with logical ancestry masks.
//!
//! The prompt — chat-templated, or `prompt_tokens` as given — is every beam's
//! common ancestry. It is prefilled ONCE into the head of a working set, and
//! that fire's epilogue picks the first `beams` tokens (the top of the
//! prompt's next-token distribution), as greedy decoding picks its first.
//!
//! The beams then share that ONE working set, a pool repeated for every lane.
//! Each surviving beam appends its new token at the next free pool slot
//! (`slot = fill + lane`, computed on the device), and its ancestry is encoded
//! in the per-beam mask — inherit the parent's mask (`gather(mask, parent)`)
//! then set the one new cell (`eq(col, slot)`). That is the entire fork/prune
//! mechanism. Every mask admits the prompt's slots. A beam's lane reads the
//! pool up to its own slot, as every lane the host places does (it states no
//! `kv_len`): its ancestors all stand at earlier steps' slots, below this
//! step's, and its mask hides the other beams' cells among them.
//!
//! The search runs `max_tokens` steps: a beam is not retired at end of
//! sequence, so width 1 is greedy decoding of the same prompt to the same
//! length (the beam identity, checked every step against the raw argmax).

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
    // The prompt's slots, then one slot per beam per decode step.
    let decodes = max_steps - 1;
    let pool_len = u32::try_from(decodes)
        .ok()
        .and_then(|steps| steps.checked_mul(b))
        .and_then(|cells| cells.checked_add(n))
        .ok_or("beams * max_tokens exceeds the u32 range")?;
    let ws = WorkingSet::new();
    // The device chooses the beams' slots, so the pool is reserved up front.
    ws.reserve(pool_len).context("ws.reserve pool")?;

    let pipeline = Pipeline::new();

    // Step 0: the prompt's own next-token distribution. Every beam is the
    // prompt, so the first `b` tokens are its top `b`.
    let prompt_i32: Vec<i32> = prompt.iter().map(|&t| t as i32).collect();
    let toks_p = Channel::from(&prompt_i32[..]).named("toks_p");
    let embed_indptr_p = Channel::from([0u32, n]).named("embed_indptr_p");
    let out_p = Channel::new([b], dtype::i32).named("out_p");
    let scr_p = Channel::new([b], dtype::f32).named("scr_p");
    let greedy_p = Channel::new([1], dtype::i32).named("greedy_p");
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
        let scores = Channel::from(&final_scores[..]).named("scores");
        let toks = Channel::from(&first[..]).named("toks");
        let out = Channel::new([b], dtype::i32)
            .capacity(channel_capacity() as u32)
            .named("out");
        let out_par = Channel::new([b], dtype::u32)
            .capacity(channel_capacity() as u32)
            .named("out_par");
        let out_scr = Channel::new([b], dtype::f32)
            .capacity(channel_capacity() as u32)
            .named("out_scr");
        let out_greedy = Channel::new([b], dtype::i32)
            .capacity(channel_capacity() as u32)
            .named("out_greedy");
        let lanes_b = Channel::from_iter(0u32..=b).named("embed_indptr");

        let fwd = ForwardPass::new();
        fwd.embed(&toks, &lanes_b)?;
        let pool: Vec<&WorkingSet> = vec![&ws; b as usize];

        // The first decode writes beam `i`'s token at slot `n + i`; it sees
        // the prompt and its own cell.
        let init_mask: Vec<bool> = (0..b)
            .flat_map(|lane| (0..pool_len).map(move |p| p < n || p == n + lane))
            .collect();
        let mask = Channel::from_shaped([b, pool_len], init_mask).named("mask");
        let slots = Channel::from_iter(n..n + b).named("slots");
        // A beam's RoPE position is its depth in the sequence, not its pool slot.
        let pos = Channel::from(vec![n; b as usize]).named("pos");
        // Where the fire after this one writes its first beam.
        let fill = Channel::from([n + b]).named("fill");
        fwd.mask(&mask)?;
        fwd.state(
            &pool,
            StateGeometry {
                positions: Some(&pos),
                slots: Some(&slots),
                kv_len: None,
                fold: Fold::None,
            },
        )
        .context("bind the beam pool")?;
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

            let base = fill.take(); // [1]
            let base_b = broadcast(reshape(&base, [1]), [b]); // [1] -> [B]
            let wpos = &base_b + iota(b); // [B]

            let inherited = gather(mask.take(), &parent); // bool [B,pool_len]
            let col = broadcast(reshape(iota(pool_len), [1, pool_len]), [b, pool_len]);
            let wpos_b = broadcast(reshape(&wpos, [b, 1]), [b, pool_len]);
            let newpos = eq(col, wpos_b); // bool [B,pool_len]
            mask.put(or(inherited, &newpos));

            // Device-resolved geometry is loop-carried: the host never drains
            // these rings, so every fire's values are re-put here.
            let filled = &base + b; // [1]
            slots.put(&wpos);
            pos.put(pos.take() + 1u32);
            fill.put(&filled);
        });

        let mut step = 0usize;
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
    Ok(format!(
        "{text}\n[beam] width={b} steps={max_steps} best_score={:.4} greedy_mismatches={greedy_mismatches}",
        final_scores[best_lane]
    ))
}
