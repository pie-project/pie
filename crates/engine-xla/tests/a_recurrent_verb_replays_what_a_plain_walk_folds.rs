//! A buffered recurrent verb answers what a plain walk does: two tokens
//! buffered without folding, then replayed ahead of a third and folded with
//! it, leave the third's logits (and the state) where decoding the three one
//! by one leaves them. Needs a hybrid model (PIE_XLA_SNAPSHOT + SKU or
//! PIE_XLA_ARTIFACT).

mod common;

use engine::fire::{FoldLen, RsReset, RsVerb};
use engine_xla::{Boot, DeviceBoot, Lane, Seated, Shell};
use poem::{Platform, Request};
use poem_compiler::Budget;

fn argmax(v: &[f32]) -> usize {
    let mut best = 0;
    for (i, x) in v.iter().enumerate() {
        if *x > v[best] {
            best = i;
        }
    }
    best
}

#[test]
fn a_replayed_buffer_folds_to_the_plain_walk() {
    let Some(m) = common::model() else {
        eprintln!("not asked: set PIE_XLA_SNAPSHOT + PIE_XLA_SKU or PIE_XLA_ARTIFACT");
        return;
    };
    let facts = m.sku.trace(models::Platform::Xla).facts;
    let word = |q: u32| facts.word(&Request::new(q, false));
    let context = 256;
    let _device = engine_xla::bench::lock_device();
    let mut shell = Shell::load(Boot {
        trace: m.sku.trace(Platform::Xla),
        contract: &m.contract,
        checkpoint: &m.checkpoint,
        budget: Budget::new(4, context),
        page_size: 16,
        context,
        slots: 4,
        pages: 4 * context / 16,
        device: &DeviceBoot::default(),
        patches: None,
    })
    .expect("the shell loads");
    assert!(shell.serves_rs_verbs(), "the model is hybrid");

    let prompt: Vec<u32> = (0..9).map(|i| 1000 + 37 * i).collect();
    let tail = [4242u32, 777, 31337];
    let p = prompt.len() as u32;

    // A: decode the three one by one on slot 0 (pages 0..).
    let pages_a: Vec<u32> = (0..16).collect();
    let pages_b: Vec<u32> = (16..32).collect();
    let seat = |slot, tokens: &'static [u32], pages: &'static [u32], held| {
        let mut s = Seated::of(Lane {
            slot,
            word: word(tokens.len() as u32),
            tokens,
        });
        s.pages = pages;
        s.held = Some(held);
        s
    };
    let prompt: &'static [u32] = Box::leak(prompt.into_boxed_slice());
    let pages_a: &'static [u32] = Box::leak(pages_a.into_boxed_slice());
    let pages_b: &'static [u32] = Box::leak(pages_b.into_boxed_slice());
    let mut s = seat(0, prompt, pages_a, 0);
    s.rs_reset = RsReset::Fresh;
    shell.fire_seated(&[s]).expect("A's prefill");
    let mut last = Vec::new();
    for (i, t) in tail.iter().enumerate() {
        let one: &'static [u32] = Box::leak(vec![*t].into_boxed_slice());
        let mut s = seat(0, one, pages_a, p + i as u32);
        s.rs_reset = RsReset::Held;
        last = shell.fire_seated(&[s]).expect("A's decode").remove(0);
    }

    // B: the same prompt on slot 1, then [a, b] buffered and not folded,
    // then [c] with a and b replayed ahead of it and all three folded.
    let mut s = seat(1, prompt, pages_b, 0);
    s.rs_reset = RsReset::Fresh;
    shell.fire_seated(&[s]).expect("B's prefill");
    let two: &'static [u32] = Box::leak(tail[..2].to_vec().into_boxed_slice());
    let buffer = RsVerb::Buffer {
        pages: vec![3],
        at: 0,
        fold: FoldLen::Host(0),
        replay: 0,
    };
    let mut s = seat(1, two, pages_b, p);
    s.rs_reset = RsReset::Held;
    s.rs = Box::leak(Box::new(buffer));
    shell.fire_seated(&[s]).expect("B's buffered pair");
    let one: &'static [u32] = Box::leak(vec![tail[2]].into_boxed_slice());
    let replay = RsVerb::Buffer {
        pages: vec![3],
        at: 2,
        fold: FoldLen::Host(3),
        replay: 2,
    };
    let mut s = seat(1, one, pages_b, p + 2);
    s.rs_reset = RsReset::Held;
    s.rs = Box::leak(Box::new(replay));
    let got = shell.fire_seated(&[s]).expect("B's replay").remove(0);

    let worst = last
        .iter()
        .zip(&got)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    eprintln!(
        "plain argmax {} / replayed argmax {}, max |Δlogit| {worst}",
        argmax(&last),
        argmax(&got)
    );
    assert_eq!(argmax(&last), argmax(&got));
    assert!(
        worst < 0.5,
        "the replayed fold drifts {worst} from the plain walk"
    );
}
