#![cfg(target_vendor = "apple")]

//! A fire whose verb states a frozen length behind where the engine's last
//! fold left the record (the runtime learns a device-counted fold only when
//! its fire settles, and states the length before it to a fire staged
//! sooner) replays from where the record stands, not from the stated
//! length: the walk with the stale statement lands the logits and the state
//! of the walk stated exactly, bit for bit. Without it the rows the record
//! already froze would be replayed onto it a second time, silently.

use engine::fire::{FoldLen, RsReset, RsVerb};
use engine_metal::serve::Seated;
use engine_metal::{Lane, Shell};

use super::a_scattered_pool_serves_the_same_logits::{Fixture, fixture_of, load_with, same, word};

fn buffer(pages: u32, folded: u32, fold: u32) -> RsVerb {
    RsVerb::Buffer {
        pages: (0..pages).collect(),
        base: 0,
        folded,
        fold: FoldLen::Host(fold),
        lineage: engine::fire::Lineage::Set,
    }
}

fn fire(
    shell: &mut Shell,
    fixture: &Fixture,
    slot: u32,
    tokens: &[u32],
    rs: &RsVerb,
    reset: RsReset,
    held: u32,
) -> Vec<Vec<f32>> {
    shell
        .fire_seated(&[Seated {
            replay_stated: None,
            held: Some(held),
            rs,
            rs_reset: reset,
            ..Seated::of(Lane {
                slot,
                word: word(&fixture.trace, tokens.len() as u32),
                tokens,
            })
        }])
        .expect("the fire lands")
}

/// Buffers five rows, then two more folding three, then two more: the last
/// fire stated `third_folded` as its frozen length. Every logit row and the
/// slot's state after.
fn walk(shell: &mut Shell, fixture: &Fixture, slot: u32, third_folded: u32) -> (Vec<Vec<f32>>, Vec<u8>) {
    let layout = shell.pool_layout();
    let pages = 12u32.div_ceil(layout.rs_tokens.max(1)) + 1;
    let prompt: Vec<u32> = (0..5).map(|at| (7 + at * 37) % 256).collect();
    let (second, third) = ([42u32, 43], [44u32, 45]);
    shell.open(slot).expect("the slot opens");
    let mut out = fire(shell, fixture, slot, &prompt, &buffer(pages, 0, 0), RsReset::Fresh, 0);
    out.extend(fire(shell, fixture, slot, &second, &buffer(pages, 0, 3), RsReset::Held, 5));
    out.extend(fire(
        shell,
        fixture,
        slot,
        &third,
        &buffer(pages, third_folded, 0),
        RsReset::Held,
        7,
    ));
    (out, shell.state_bytes(slot).expect("the slot reads back"))
}

#[test]
fn a_stale_frozen_length_replays_from_where_the_record_stands() {
    if !engine_metal::device::present() {
        eprintln!("skipping: no Metal device");
        return;
    }
    let fixture = fixture_of("qwen35-tiny", None);
    let mut shell = load_with(&fixture, 1024);
    assert!(shell.serves_rs_verbs(), "a hybrid model buffers recurrent rows");
    // The second fire froze the record through slot 3: stated exactly, and
    // stale (the length before that fold, as a fire staged before it
    // settled is told).
    let (exact, exact_state) = walk(&mut shell, &fixture, 0, 3);
    let (stale, stale_state) = walk(&mut shell, &fixture, 1, 0);
    same("a stale frozen length against the exact one", &stale, &exact);
    assert_eq!(
        stale_state, exact_state,
        "the record after the stale statement is the record after the exact one"
    );
}
