#![cfg(target_vendor = "apple")]

//! A lane that decodes on a hybrid model without folding (beam_search's
//! decode: `Fold::None`, its recurrent rows buffered and every fire
//! replaying them from the folded state) answers each step what the same
//! tokens folded step by step answer. Qwen3.5's served beam went wrong by
//! its twelfth step at every width while its stage answered exactly: its
//! logits were the replay's.

use engine::fire::{FoldLen, RsReset, RsVerb};
use engine_metal::serve::Seated;
use engine_metal::Lane;
use poem::Request;

use super::a_scattered_pool_serves_the_same_logits::{fixture_of, load_with};

const PROMPT: u32 = 20;
const STEPS: u32 = 24;

fn spread(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max)
}

#[test]
fn an_unfolded_decode_is_the_folded_one() {
    if !engine_metal::device::present() {
        return;
    }
    let fixture = fixture_of("qwen35-tiny", None);
    let mut shell = load_with(&fixture, 1024);
    assert!(shell.serves_rs_verbs(), "a hybrid model");
    let word = |n: u32| fixture.trace.facts.word(&Request::new(n, false));
    let prompt: Vec<u32> = (0..PROMPT).map(|at| (11 + at * 37) % 256).collect();
    let tokens: Vec<u32> = (0..STEPS).map(|at| (5 + at * 53) % 256).collect();
    let per_page = shell.pool_layout().rs_tokens.max(1);
    eprintln!("buffer page: {per_page} rows");
    let fold = RsVerb::Fold;
    let fire = |shell: &mut engine_metal::Shell, slot: u32, held: u32, toks: &[u32], rs: &RsVerb, reset: RsReset| {
        shell
            .fire_seated(&[Seated {
                replay_stated: None,
                held: Some(held),
                rs,
                rs_reset: reset,
                ..Seated::of(Lane { slot, word: word(toks.len() as u32), tokens: toks })
            }])
            .expect("the fire lands")[0]
            .clone()
    };
    // Both slots hold the prompt, folded.
    for slot in [0, 1] {
        shell.open(slot).expect("opens");
        fire(&mut shell, slot, 0, &prompt, &fold, RsReset::Fresh);
    }
    let mut first = None;
    let mut worst = 0.0f32;
    for (k, &token) in tokens.iter().enumerate() {
        let held = PROMPT + k as u32;
        let folded = fire(&mut shell, 0, held, &[token], &fold, RsReset::Held);
        // Never folding: the state holds the prompt, and the buffer every
        // decode row since.
        // The pages the buffered rows take, as a runtime grants them.
        let pages: Vec<u32> = (0..(k as u32 + 1).div_ceil(per_page)).collect();
        let unfolded = RsVerb::Buffer {
            pages,
            base: PROMPT,
            folded: PROMPT,
            fold: FoldLen::Host(0),
            // Alone on its set: every buffered slot.
            lineage: engine::fire::Lineage::Set,
        };
        let replayed = fire(&mut shell, 1, held, &[token], &unfolded, RsReset::Held);
        // A replaying decode reads its buffered rows as a linear attention
        // over the committed state (the carried scan): the folded walk's
        // arithmetic in another f32 order, so the gap is relative to the
        // logits' scale (engine-cuda's golden_lagged bound), and no argmax
        // moves where the top two stand apart.
        let d = spread(&folded, &replayed);
        let scale = folded.iter().fold(0f32, |m, &x| m.max(x.abs())).max(1e-6);
        let argmax = |v: &[f32]| {
            v.iter()
                .enumerate()
                .fold((0usize, f32::MIN), |best, (at, &x)| if x > best.1 { (at, x) } else { best })
        };
        let (top, top_logit) = argmax(&folded);
        let runner_up = folded
            .iter()
            .enumerate()
            .filter(|&(at, _)| at != top)
            .fold(f32::MIN, |m, (_, &x)| m.max(x));
        let flipped = argmax(&replayed).0 != top && top_logit - runner_up > 0.02;
        let rel = d / scale;
        worst = worst.max(rel);
        eprintln!("step {k} (replays {} rows): {d:.5} ({rel:.4} of the logits' scale)", k + 1);
        if (rel > 0.05 || flipped) && first.is_none() {
            first = Some(k);
        }
    }
    assert!(first.is_none(), "an unfolded decode parts from the folded one at step {first:?}: worst {worst} of the logits' scale");
}
