//! A lane folding `lag` rows behind its decodes answers, step for step,
//! what the lane folding each row as it lands answers, to the tolerance of
//! the two forms of the recurrence (engine-cuda's `golden_lagged`): its
//! decode step reads the buffered rows as a linear attention over their
//! stored keys and corrected values from the committed state, where the
//! folding lane steps and stores the state a row. A prompt folded as it
//! lands, then 30 decodes of a fixed script, each buffering its row and
//! folding so `lag` rows stay unfolded (every seventh step folding them
//! all, every fifth none), at lags 1, 2, 3 and 9; each step's logits within
//! 0.05 of their scale of the folded lane's, and no greedy token flipped
//! off a near tie. qwen35-tiny (gated-delta layers). Skipped without a
//! PJRT plugin.

use engine::chunk::{SlotKind, SlotRole};
use engine::fire::{FoldLen, RsReset, RsVerb};

use crate::common_llm::{Model, Refs, Seq};

const STEPS: u32 = 30;
const PROMPT: u32 = 37;

/// Each step's carried rows before it (the lagged lane's), for a probe's
/// print: `PIE_LAG_STEPS` prints each step, `PIE_LAGS` the lags to run.
fn served(lag: u32) -> Vec<Vec<f32>> {
    served_with(lag).0
}

fn served_with(lag: u32) -> (Vec<Vec<f32>>, Vec<u32>) {
    let mut model = Model::load_with("qwen35-tiny", None, 256, 7, None);
    let mut refs = Refs::of(&model);
    let shape = refs
        .shapes(|s| s.role == SlotRole::RsBuffer)
        .into_iter()
        .next()
        .expect("a recurrent plan states an rs list");
    let SlotKind::List { tokens: page, .. } = shape.kind else {
        panic!("the rs list is a list shape");
    };
    let pages: Vec<u32> = (0..(PROMPT + STEPS + 1).div_ceil(page))
        .map(|_| refs.take(shape.id))
        .collect();
    let vocab = model.vocab;
    let mut seq = Seq::new(0);
    let prompt: Vec<u32> = (0..PROMPT).map(|t| (t * 7 + 11) % vocab).collect();
    let lane = seq.lane(&model, &mut refs, prompt);
    let mut steps = Vec::new();
    steps.push(model.fire(vec![lane]).pop().expect("a row"));
    seq.held = PROMPT;
    let (mut fed, mut folded) = (PROMPT, PROMPT);
    let mut carried = vec![0u32];
    for at in 0..STEPS {
        let token = (fed * 7 + 11) % vocab;
        let mut lane = seq.lane(&model, &mut refs, vec![token]);
        let end = fed + 1;
        carried.push(fed - folded);
        if lag > 0 {
            let fold = match at {
                _ if at % 7 == 6 => end - folded,
                _ if at % 5 == 4 => 0,
                _ => (end - folded).saturating_sub(lag),
            };
            lane.rs = RsVerb::Buffer {
                pages: pages.clone(),
                base: 0,
                folded,
                fold: FoldLen::Host(fold),
                lineage: engine::fire::Lineage::Set,
            };
            lane.rs_reset = RsReset::Held;
            folded += fold;
        } else {
            folded += 1;
        }
        steps.push(model.fire(vec![lane]).pop().expect("a row"));
        seq.held += 1;
        fed = end;
    }
    (steps, carried)
}

fn argmax(x: &[f32]) -> usize {
    x.iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map_or(0, |(i, _)| i)
}

#[test]
fn a_lagged_decode_answers_the_folded_lane() {
    let Some(_device) = crate::common_dit::device() else {
        return;
    };
    let folded = served(0);
    let lags: Vec<u32> = std::env::var("PIE_LAGS").map_or_else(
        |_| vec![1, 2, 3, 9],
        |v| v.split(',').map(|l| l.parse().expect("a lag")).collect(),
    );
    for lag in lags {
        let (lagged, carried) = served_with(lag);
        assert_eq!(lagged.len(), folded.len());
        let (mut worst, mut flips) = (0f32, 0usize);
        for (step, (l, f)) in lagged.iter().zip(&folded).enumerate() {
            let scale = f.iter().fold(0f32, |m, v| m.max(v.abs())).max(f32::MIN_POSITIVE);
            let apart = l.iter().zip(f).fold(0f32, |m, (a, b)| m.max((a - b).abs()));
            worst = worst.max(apart / scale);
            if std::env::var_os("PIE_LAG_STEPS").is_some() {
                println!(
                    "lag {lag} step {step}: carried {} apart {:e}",
                    carried[step],
                    apart / scale
                );
            }
            if argmax(l) != argmax(f) {
                let mut sorted = f.clone();
                sorted.sort_by(|x, y| y.total_cmp(x));
                if (sorted[0] - sorted[1]) / scale > 0.02 {
                    flips += 1;
                }
            }
        }
        println!(
            "qwen35-tiny lagged lag {lag}: {} steps, logits apart by {worst:e} of their scale, \
             {flips} greedy token(s) flipped off a near tie",
            lagged.len()
        );
        assert!(
            worst < 0.05 && flips == 0,
            "at a lag of {lag} the lagged lane's logits lie {worst:e} of their scale from the \
             folded lane's, {flips} greedy token(s) flipped"
        );
    }
}
