#![cfg(target_vendor = "apple")]

//! The lagged fold's gate on Metal (engine-cuda's `golden_lagged`,
//! new-memory/designs/lagged-fold.md): a hybrid decode that folds only the
//! rows a lag behind its depth, its later rows buffered and read each fire
//! as a linear attention over the committed state (the carried scan),
//! answers each step what the same tokens folded every step answer. On
//! qwen35-tiny, a 37-token prompt then 45 decodes, at lags 11, 20 and 45,
//! every 5th step folding none and step 38 all (so a lag of 45 carries
//! past 32 rows, where the carried scan runs a head's value rows a
//! threadgroup): each step's
//! logits within 5% of their scale of the unlagged walk's, and no greedy
//! token moved where the unlagged walk's top two stand 0.02 apart.

use engine::fire::{FoldLen, Lineage, RsReset, RsVerb};
use engine_metal::serve::Seated;
use engine_metal::Lane;
use poem::Request;

use super::a_scattered_pool_serves_the_same_logits::{fixture_of, load_with};

const PROMPT: u32 = 37;
const STEPS: u32 = 45;
const LAGS: [u32; 3] = [11, 20, 45];

fn spread(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max)
}

fn argmax(v: &[f32]) -> (usize, f32) {
    v.iter()
        .enumerate()
        .fold((0, f32::MIN), |best, (at, &x)| if x > best.1 { (at, x) } else { best })
}

#[test]
fn a_lagged_decode_answers_the_unlagged() {
    if !engine_metal::device::present() {
        return;
    }
    let fixture = fixture_of("qwen35-tiny", None);
    let mut shell = load_with(&fixture, 2048);
    assert!(shell.serves_rs_verbs(), "a hybrid model");
    let word = |n: u32| fixture.trace.facts.word(&Request::new(n, false));
    let prompt: Vec<u32> = (0..PROMPT).map(|at| (11 + at * 37) % 256).collect();
    let tokens: Vec<u32> = (0..STEPS).map(|at| (5 + at * 53) % 256).collect();
    let per_page = shell.pool_layout().rs_tokens.max(1);
    let fire = |shell: &mut engine_metal::Shell,
                slot: u32,
                held: u32,
                toks: &[u32],
                rs: &RsVerb,
                reset: RsReset| {
        shell
            .fire_seated(&[Seated {
                held: Some(held),
                rs,
                rs_reset: reset,
                ..Seated::of(Lane { slot, word: word(toks.len() as u32), tokens: toks })
            }])
            .expect("the fire lands")[0]
            .clone()
    };
    // Slot 0 the unlagged walk; a slot a lag beside it, each holding the
    // prompt folded.
    let slots: Vec<u32> = (0..=LAGS.len() as u32).collect();
    for &slot in &slots {
        shell.open(slot).expect("opens");
        fire(&mut shell, slot, 0, &prompt, &RsVerb::Fold, RsReset::Fresh);
    }
    let mut folded_at = vec![PROMPT; LAGS.len()];
    let mut failures = Vec::new();
    for (k, &token) in tokens.iter().enumerate() {
        let held = PROMPT + k as u32;
        let want = fire(&mut shell, 0, held, &[token], &RsVerb::Fold, RsReset::Held);
        let scale = want.iter().fold(0f32, |m, &x| m.max(x.abs())).max(1e-6);
        let (top, top_logit) = argmax(&want);
        let runner_up = want
            .iter()
            .enumerate()
            .filter(|&(at, _)| at != top)
            .fold(f32::MIN, |m, (_, &x)| m.max(x));
        for (at, &lag) in LAGS.iter().enumerate() {
            let slot = slots[at + 1];
            // The rows through this fire's own, less the lag, fold: the
            // runtime's accepted length; every 5th step none, step 27 all.
            let through = held + 1;
            let carried = held - folded_at[at];
            let target = if k == 38 {
                through
            } else if k % 5 == 4 {
                folded_at[at]
            } else {
                through.saturating_sub(lag).max(folded_at[at])
            };
            let fold = target - folded_at[at];
            let pages: Vec<u32> = (0..(through - PROMPT).div_ceil(per_page)).collect();
            let verb = RsVerb::Buffer {
                pages,
                base: PROMPT,
                folded: folded_at[at],
                fold: FoldLen::Host(fold),
                lineage: Lineage::Set,
            };
            let got = fire(&mut shell, slot, held, &[token], &verb, RsReset::Held);
            folded_at[at] = target;
            let rel = spread(&want, &got) / scale;
            let flipped = argmax(&got).0 != top && top_logit - runner_up > 0.02;
            eprintln!(
                "lag {lag} step {k}: carries {carried} rows, {rel:.4} of the logits' scale{}",
                if flipped { ", greedy flipped" } else { "" }
            );
            if rel > 0.05 || flipped {
                failures.push(format!("lag {lag} step {k}: {rel:.4}{}", if flipped { " (flip)" } else { "" }));
            }
        }
    }
    assert!(failures.is_empty(), "a lagged decode parts from the unlagged: {failures:?}");
}
