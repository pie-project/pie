//! On-device beam search with logical ancestry masks.
//!
//! The beams share ONE working set, a pool repeated for every lane. Each
//! surviving beam appends its new token at the next free pool slot
//! (`slot = fill + lane`, computed on the device), and its ancestry is encoded
//! in the per-beam mask — inherit the parent's mask
//! (`gather(mask, parent)`) then set the one new cell (`eq(col, slot)`). That is
//! the entire fork/prune mechanism.
//!
//! **IT DOES NOT READ `prompt`.** Every beam is seeded with `BOS` at
//! position 0 and decodes from there, so its output is the same whatever the
//! caller asks — that is the mechanism under test, not a defect, and it is
//! why the curated cases gate on beam-against-greedy rather than on the
//! answer attending a prompt.

use inferlet::eta::prelude::*;
use serde::Deserialize;

// Qwen3-0.6B
const BOS: i32 = 1;

#[derive(Deserialize)]
struct Input {
    #[serde(default = "default_max_tokens")]
    max_tokens: usize,
    /// Beam width. `1` degenerates to greedy decoding (the beam identity).
    #[serde(default = "default_beams")]
    beams: u32,
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
    let v = model::output_vocab_size();

    let scores = {
        let mut initial_scores = vec![f32::NEG_INFINITY; b as usize];
        initial_scores[0] = 0.0;
        Channel::from(initial_scores)
    };
    let toks = Channel::from(vec![BOS; b as usize]);
    let out = Channel::new([b], dtype::i32).capacity(channel_capacity() as u32);
    let out_par = Channel::new([b], dtype::u32).capacity(channel_capacity() as u32);
    let out_scr = Channel::new([b], dtype::f32).capacity(channel_capacity() as u32);
    let out_greedy = Channel::new([b], dtype::i32).capacity(channel_capacity() as u32);
    let lanes_b = Channel::from_iter(0u32..=b);

    let pipeline = Pipeline::new();
    let fwd = ForwardPass::new();
    fwd.embed(&toks, &lanes_b)?;

    let mut hypotheses = vec![Vec::<u32>::new(); b as usize];
    let mut final_scores = vec![f32::NEG_INFINITY; b as usize];
    let mut greedy_mismatches = 0usize;
    let pool_len = u32::try_from(max_steps)
        .ok()
        .and_then(|steps| steps.checked_mul(b))
        .and_then(|cells| cells.checked_add(1))
        .ok_or("beams * max_tokens exceeds the u32 range")?;
    let ws = WorkingSet::new();
    // The device chooses the slots, so the pool is reserved up front.
    ws.reserve(pool_len).context("ws.reserve pool")?;
    let pool: Vec<&WorkingSet> = vec![&ws; b as usize];

    let init_mask: Vec<bool> = (0..b).flat_map(|_| (0..pool_len).map(|p| p == 0)).collect();
    let mask = Channel::from_shaped([b, pool_len], init_mask);
    let slots = Channel::from(vec![0u32; b as usize]);
    // A beam's RoPE position is its depth, not its pool slot.
    let pos = Channel::from(vec![0u32; b as usize]);
    let fill = Channel::from([1u32]);
    let kv_len = Channel::from(vec![1u32; b as usize]);
    fwd.mask(&mask)?;
    fwd.state(
        &pool,
        StateGeometry {
            positions: Some(&pos),
            slots: Some(&slots),
            kv_len: Some(&kv_len),
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
        kv_len.put(broadcast(reshape(&filled, [1]), [b]));
        pos.put(pos.take() + 1u32);
        fill.put(&filled);
    });

    let mut step = 0usize;
    run_ahead(&pipeline, &fwd, max_steps, async || {
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
