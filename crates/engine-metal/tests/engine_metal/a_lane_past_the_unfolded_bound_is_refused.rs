#![cfg(target_vendor = "apple")]

//! A lane carrying more than `engine::fire::UNFOLDED_ROWS` recurrent rows
//! into a fire (as a device-geometry fire may: its slots are the guest's,
//! which the runtime cannot bound) is refused by name before anything is
//! encoded, so nothing past the rung's rs region is written: its state reads
//! back as it was. A lane carrying exactly the bound fires.

use engine::fire::{FoldLen, RsReset, RsVerb};
use engine_metal::serve::Seated;
use engine_metal::{Lane, Shell};

use super::a_scattered_pool_serves_the_same_logits::{Fixture, fixture_of, load_with, word};

fn buffer(pages: u32, folded: u32, fold: u32) -> RsVerb {
    RsVerb::Buffer {
        pages: (0..pages).collect(),
        base: 0,
        folded,
        fold: FoldLen::Host(fold),
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
) -> Result<Vec<Vec<f32>>, String> {
    shell
        .fire_seated(&[Seated {
            held: Some(held),
            rs,
            rs_reset: reset,
            ..Seated::of(Lane {
                slot,
                word: word(&fixture.trace, tokens.len() as u32),
                tokens,
            })
        }])
        .map_err(|why| why.to_string())
}

#[test]
fn a_lane_past_the_unfolded_bound_is_refused() {
    if !engine_metal::device::present() {
        eprintln!("skipping: no Metal device");
        return;
    }
    let fixture = fixture_of("qwen35-tiny", None);
    let mut shell = load_with(&fixture, 1024);
    assert!(
        shell.serves_rs_verbs(),
        "a hybrid model buffers recurrent rows"
    );
    let bound = engine::fire::UNFOLDED_ROWS;
    let rows: Vec<u32> = (0..=bound).map(|at| (7 + at * 37) % 256).collect();
    let next = [42u32];
    let pages = 6;

    // Past the bound: every row buffered, none folded, then one more fire.
    shell.open(0).expect("slot 0 opens");
    fire(
        &mut shell,
        &fixture,
        0,
        &rows,
        &buffer(pages, 0, 0),
        RsReset::Fresh,
        0,
    )
    .expect("buffering the bound and one more row runs");
    let before = shell.state_bytes(0).expect("slot 0 reads back");
    let why = fire(
        &mut shell,
        &fixture,
        0,
        &next,
        &buffer(pages, 0, 0),
        RsReset::Held,
        bound + 1,
    )
    .expect_err("a lane carrying past the bound is refused");
    assert!(
        why.contains("UNFOLDED_ROWS") && why.contains(&format!("carries {}", bound + 1)),
        "the refusal names the lane's rows and the bound: {why}"
    );
    assert_eq!(
        shell.state_bytes(0).expect("slot 0 reads back"),
        before,
        "the refused fire wrote nothing"
    );

    // At the bound: one row folded, so the next fire carries exactly it.
    shell.open(1).expect("slot 1 opens");
    fire(
        &mut shell,
        &fixture,
        1,
        &rows,
        &buffer(pages, 0, 1),
        RsReset::Fresh,
        0,
    )
    .expect("buffering, folding one row, runs");
    fire(
        &mut shell,
        &fixture,
        1,
        &next,
        &buffer(pages, 1, 0),
        RsReset::Held,
        bound + 1,
    )
    .expect("a lane carrying exactly the bound fires");
    eprintln!(
        "a lane past {bound} carried rows is refused and writes nothing; one at {bound} fires"
    );
}
