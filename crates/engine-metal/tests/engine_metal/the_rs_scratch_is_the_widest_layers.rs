#![cfg(target_vendor = "apple")]

//! A fire's recurrent scratch holds the widest layer's extended planes, not
//! every layer's: planes of one role whose ops never overlap share a slot,
//! one recurrent layer after another (`rs::Layout::share`). A replay at the
//! top rung through the shared slots answers as a plain fold: window A
//! buffered, then B replaying A and folding it, against A folded then B.

use engine::fire::{FoldLen, RsReset, RsVerb};
use engine_metal::serve::Seated;
use engine_metal::Lane;

use super::a_scattered_pool_serves_the_same_logits::{fixture_of, load_with, word};

#[test]
fn the_rs_scratch_is_the_widest_layers() {
    if !engine_metal::device::present() {
        eprintln!("skipping: no Metal device");
        return;
    }
    let fixture = fixture_of("qwen35-tiny", None);
    let mut shell = load_with(&fixture, 256);
    assert!(shell.serves_rs_verbs(), "a recurrent model");
    let rooms = shell.rs_rooms();
    eprintln!("each rung's rs scratch: {rooms:?}");
    let a: Vec<u32> = (0..6u32).map(|at| (11 + at * 37) % 256).collect();
    let b: Vec<u32> = (0..6u32).map(|at| (101 + at * 37) % 256).collect();
    let per_page = shell.pool_layout().rs_tokens.max(1);
    let buffer = |fold: u32| RsVerb::Buffer {
        pages: (0..12u32.div_ceil(per_page)).collect(),
        base: 0,
        folded: 0,
        fold: FoldLen::Host(fold),
        lineage: engine::fire::Lineage::Set,
    };
    let (hold, replay) = (buffer(0), buffer(6));
    let fold = RsVerb::Fold;
    for slot in 0..2 {
        shell.open(slot).expect("a slot opens");
    }
    let mut fire = |slot: u32, tokens: &[u32], rs: &RsVerb, held: u32| -> Vec<f32> {
        shell
            .fire_seated(&[Seated {
                replay_stated: None,
                held: Some(held),
                rs,
                rs_reset: if held == 0 { RsReset::Fresh } else { RsReset::Held },
                ..Seated::of(Lane {
                    slot,
                    word: word(&fixture.trace, tokens.len() as u32),
                    tokens,
                })
            }])
            .expect("the fire lands")
            .remove(0)
    };
    fire(0, &a, &fold, 0);
    let plain = fire(0, &b, &fold, 6);
    fire(1, &a, &hold, 0);
    let replayed = fire(1, &b, &replay, 6);
    let apart = plain
        .iter()
        .zip(&replayed)
        .map(|(x, y)| (x - y).abs())
        .fold(0f32, f32::max);
    eprintln!("B after replay(A) vs B after fold(A): {apart}");
    assert!(apart <= 0.05, "a replay through the shared scratch answers as the fold: {apart}");
}
