//! A decode of several lanes answers, lane by lane, what each answers
//! decoded alone (contract A: a token parts only on a near-tie), over the
//! u4g64 deployments, whose projections carry an affine offset. libtpu
//! miscompiled that offset as a dot in gpt-oss's multi-lane decode, and the
//! batched head answered other tokens than the lone one at every step
//! (FOLLOWUPS F65). Skipped without a PJRT plugin.

use crate::common_llm::{Model, Refs, Seq, tokens};

const PROMPT: usize = 24;
const LANES: u32 = 4;
const STEPS: u32 = 4;
/// The u4g64 deployments, cut to two layers: gpt-oss's head is 45 groups
/// of a 201088 vocabulary (where the miscompile bit), qwen3's 64 groups.
const DEPLOYMENTS: [&str; 2] = ["gptoss-20b-mini-u4g64-mxfp4-kv-bf16", "qwen3-8b-u4g64-kv-bf16"];

fn argmax(row: &[f32]) -> usize {
    row.iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map_or(0, |(at, _)| at)
}

/// Whether `got` answers `want`'s token, or parts from it only on a
/// near-tie of `want`'s two best.
fn agrees(want: &[f32], got: &[f32]) -> bool {
    let (a, b) = (argmax(want), argmax(got));
    if a == b {
        return true;
    }
    let scale = want.iter().fold(0f32, |m, &x| m.max(x.abs())).max(1.0);
    (want[a] - want[b]).abs() <= 2e-2 * scale
}

fn answers_alone(name: &str) {
    let mut model = Model::load(name, Some(2), 256, 7);
    let vocab = model.vocab;
    let mut refs = Refs::of(&model);
    // Lane `l` batched in slot `l`, alone in slot `LANES + l`.
    let mut batched: Vec<Seq> = (0..LANES).map(Seq::new).collect();
    let mut alone: Vec<Seq> = (0..LANES).map(|l| Seq::new(LANES + l)).collect();
    let prompt = |l: u32| tokens(u64::from(l) + 3, PROMPT, vocab);
    for l in 0..LANES {
        for seq in [&mut batched[l as usize], &mut alone[l as usize]] {
            let lane = seq.lane(&model, &mut refs, prompt(l));
            model.fire(vec![lane]);
            seq.held += PROMPT as u32;
        }
    }
    let mut worst = 0f32;
    for step in 0..STEPS {
        let token = |l: usize| (step * 7 + 5 + l as u32 * 11) % vocab;
        let lanes = (0..LANES as usize)
            .map(|l| batched[l].lane(&model, &mut refs, vec![token(l)]))
            .collect();
        let together = model.fire(lanes);
        for l in 0..LANES as usize {
            let lane = alone[l].lane(&model, &mut refs, vec![token(l)]);
            let by_itself = model.fire(vec![lane]);
            let apart = by_itself[0]
                .iter()
                .zip(&together[l])
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            worst = worst.max(apart);
            assert!(
                agrees(&by_itself[0], &together[l]),
                "{name} step {step}: lane {l} answers {} beside the others and {} alone \
                 (logits apart by {apart})",
                argmax(&together[l]),
                argmax(&by_itself[0])
            );
        }
        for seq in batched.iter_mut().chain(alone.iter_mut()) {
            seq.held += 1;
        }
    }
    eprintln!("{name}: a batched lane's logits lie within {worst:.3} of its lone run's");
}

#[test]
fn a_batched_decode_answers_what_each_lane_answers_alone() {
    let Some(_device) = crate::common_dit::device() else {
        return;
    };
    for name in DEPLOYMENTS {
        answers_alone(name);
    }
}
