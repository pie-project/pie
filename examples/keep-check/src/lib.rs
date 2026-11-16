//! A gate for keeps (a fold that sorts a set's buffer): a lane that keeps
//! part of its buffer answers as the same lane unfolded whose mask hides what
//! the keep dropped.
//!
//! Both runs prefill the prompt unfolded, then decode greedily. The keeping
//! run's first decode masks the dropped rows and states the keep: the rows
//! `[drop_from, drop_to)` are dropped, the rest committed, or, from
//! `buffer_from` on, left buffered (a suffix, as a recurrent model needs).
//! Its later decodes state nothing. The reference run masks the same rows on
//! every decode and folds nothing. The keeping run's rows past the dropped
//! ones stand at lower slots, with the positions they had.

use inferlet::eta::prelude::*;
use inferlet::{Result, model as wit_model};
use serde::{Deserialize, Serialize};

#[derive(Deserialize)]
struct Input {
    #[serde(default = "default_prompt")]
    prompt: String,
    #[serde(default = "default_steps")]
    steps: usize,
    #[serde(default = "default_drop_from")]
    drop_from: u32,
    #[serde(default = "default_drop_to")]
    drop_to: u32,
    /// From this column on, the keep leaves rows buffered; none: it commits
    /// every row it keeps.
    #[serde(default)]
    buffer_from: Option<u32>,
    /// The keeping run masks its later decodes too, admitting every column,
    /// so both runs read through the same masked kernel.
    #[serde(default = "default_mask_after")]
    mask_after: bool,
    /// The control: the reference hides nothing, so it differs from the
    /// keeping run wherever the dropped rows mattered.
    #[serde(default)]
    reference_sees_all: bool,
}

fn default_mask_after() -> bool {
    true
}

/// The logits a step is compared by: the top `TOP`, with their tokens.
const TOP: u32 = 16;

fn default_prompt() -> String {
    "The quick brown fox jumps over the lazy dog, and then it runs into the forest \
     where it finds a small river. Tell me what happens next in the story."
        .into()
}

fn default_steps() -> usize {
    16
}

fn default_drop_from() -> u32 {
    5
}

fn default_drop_to() -> u32 {
    12
}

#[derive(Serialize)]
struct Output {
    prompt_tokens: u32,
    keep_tokens: Vec<u32>,
    mask_tokens: Vec<u32>,
    /// Each step's largest difference between the two runs' top logits,
    /// where both rank the same tokens; `null` where they do not.
    top_diff: Vec<Option<f32>>,
    /// Each step's top logit and its margin over the second, in the
    /// reference: a step whose tokens differ at a margin near the noise is a
    /// tie, not a disagreement.
    top_logit: Vec<f32>,
    margin: Vec<f32>,
    same_tokens: bool,
}

/// The column's outcome: 2 dropped, 1 buffered, 0 committed.
fn outcome(column: u32, drop: (u32, u32), buffer_from: Option<u32>) -> u32 {
    if drop.0 <= column && column < drop.1 {
        2
    } else if buffer_from.is_some_and(|from| column >= from) {
        1
    } else {
        0
    }
}

async fn run(
    prompt: &[i32],
    steps: usize,
    drop: (u32, u32),
    buffer_from: Option<u32>,
    keep: bool,
    mask_after: bool,
    hides: bool,
) -> Result<(Vec<u32>, Vec<(Vec<f32>, Vec<i32>)>)> {
    let n = prompt.len() as u32;
    let v = model::output_vocab_size();
    let buffered = (0..n + 1)
        .filter(|&c| outcome(c, drop, buffer_from) == 1)
        .count() as u32;
    let ws = WorkingSet::new();
    let pipe = Pipeline::new();

    let toks = Channel::from(prompt.to_vec());
    let indptr = Channel::from(vec![0u32, n]);
    let first = Channel::new([1], dtype::i32);
    let fwd = ForwardPass::new();
    fwd.embed(&toks, &indptr)?;
    fwd.state(&[&ws], StateGeometry::default())?;
    fwd.epilogue(move || {
        first.put(reshape(reduce_argmax(intrinsics::output(0, "logits")), [1]));
    });
    fwd.submit(&pipe).map_err(|e| format!("prefill: {e}"))?;
    let mut token = first.take_host::<i32>().await?;

    let mut tokens = Vec::with_capacity(steps);
    let mut logits = Vec::with_capacity(steps);
    for step in 0..steps {
        // The decode row's column: nothing is folded before the keep. After
        // it the keeping run's buffer is what it left buffered and the rows
        // decoded since.
        let extent = n + step as u32 + 1;
        let tok = Channel::from(vec![token]);
        let lane = Channel::from(vec![0u32, 1]);
        let out_tok = Channel::new([1], dtype::i32);
        let out_values = Channel::new([TOP], dtype::f32);
        let out_ids = Channel::new([TOP], dtype::i32);
        let fwd = ForwardPass::new();
        fwd.embed(&tok, &lane)?;
        let after = keep && step > 0;
        let masks = !after || mask_after;
        let mask = if after {
            let columns = buffered + step as u32;
            Channel::from_shaped([1, columns], vec![true; columns as usize])
        } else {
            Channel::from_shaped(
                [1, extent],
                (0..extent)
                    .map(|c| !(hides && drop.0 <= c && c < drop.1))
                    .collect::<Vec<bool>>(),
            )
        };
        if masks {
            fwd.mask(&mask)?;
        }
        let kept = Channel::from_shaped(
            [1, extent],
            (0..extent)
                .map(|c| outcome(c, drop, buffer_from))
                .collect::<Vec<u32>>(),
        );
        let geometry = if keep && step == 0 {
            StateGeometry {
                fold: Fold::Keep(&kept),
                ..StateGeometry::default()
            }
        } else {
            StateGeometry::default()
        };
        fwd.state(&[&ws], geometry)?;
        fwd.epilogue(move || {
            let l = reshape(intrinsics::output(0, "logits"), [v]);
            out_tok.put(reshape(reduce_argmax(&l), [1]));
            let (values, ids) = top_k(&l, TOP);
            out_values.put(&values);
            out_ids.put(cast(&ids, dtype::i32));
        });
        fwd.submit(&pipe).map_err(|e| format!("decode {step}: {e}"))?;
        let t = out_tok.take_host::<i32>().await?;
        let values = out_values.take_host::<Vec<f32>>().await?;
        let ids = out_ids.take_host::<Vec<i32>>().await?;
        logits.push((values, ids));
        tokens.push(t as u32);
        token = t;
    }
    pipe.close();
    Ok((tokens, logits))
}

#[inferlet::main]
async fn main(input: Input) -> Result<Output> {
    let mut prompt = inferlet::chat::prefix();
    prompt.extend(wit_model::encode(&input.prompt));
    let prompt: Vec<i32> = prompt.iter().map(|&t| t as i32).collect();
    let n = prompt.len() as u32;
    let drop = (input.drop_from, input.drop_to);
    if !(drop.0 < drop.1 && drop.1 < n) {
        return Err(format!(
            "drop {drop:?} must lie inside the {n}-token prompt, before its last token"
        ));
    }
    if input.buffer_from.is_some_and(|from| from < drop.1) {
        return Err("buffer_from must come after the dropped rows (a suffix)".into());
    }
    let (keep_tokens, keep_logits) =
        run(&prompt, input.steps, drop, input.buffer_from, true, input.mask_after, true).await?;
    let (mask_tokens, mask_logits) = run(
        &prompt,
        input.steps,
        drop,
        input.buffer_from,
        false,
        false,
        !input.reference_sees_all,
    )
    .await?;
    let top_diff = keep_logits
        .iter()
        .zip(&mask_logits)
        .map(|((a, a_ids), (b, b_ids))| {
            (a_ids[..4] == b_ids[..4]).then(|| {
                a.iter()
                    .zip(b)
                    .map(|(x, y)| (x - y).abs())
                    .fold(0f32, f32::max)
            })
        })
        .collect();
    let top_logit = mask_logits.iter().map(|(values, _)| values[0]).collect();
    let margin = mask_logits
        .iter()
        .map(|(values, _)| values[0] - values[1])
        .collect();
    Ok(Output {
        prompt_tokens: n,
        same_tokens: keep_tokens == mask_tokens,
        keep_tokens,
        mask_tokens,
        top_diff,
        top_logit,
        margin,
    })
}
