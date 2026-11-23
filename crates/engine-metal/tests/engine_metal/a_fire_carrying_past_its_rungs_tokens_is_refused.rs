#![cfg(target_vendor = "apple")]

//! A fire whose lanes carry more recurrent rows into it than the tokens of
//! the rung it stages in, counted as the runtime counts them toward the
//! rung (`engine::fire::rung_holds`, `engine::fire::lane_carried`: a lane of
//! more than one own row all its carried rows, a decode lane none), is
//! refused by name before anything is encoded, so nothing past the rung's
//! rs scratch is written: its state reads back as it was. A lane carrying
//! exactly the rung's tokens fires at it; a decode lane carrying past them
//! fires too, its step reading them in place through its page table (a
//! beam in a parents pass carries every row since its fold). Each rung's
//! rs scratch holds its own rows and as many again.

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
fn a_fire_carrying_past_its_rungs_tokens_is_refused() {
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
    let rooms = shell.rs_rooms();
    // The rung the second fires stage in: the smallest of 8 tokens or more.
    let (rung, room) = *rooms
        .iter()
        .find(|(tokens, _)| *tokens >= 8)
        .expect("a rung of 8 tokens or more");
    assert!(room > 0, "the rung holds rs scratch");
    let rows: Vec<u32> = (0..=rung).map(|at| (7 + at * 37) % 256).collect();
    let next = [42u32];
    // A lane of two own rows: its carried rows count toward the rung.
    let wide = [42u32, 43];
    let layout = shell.pool_layout();
    let pages = (rung + 2).div_ceil(layout.rs_tokens.max(1)) + 1;

    // At the widest rung: slot 0 buffers the rung's tokens and one more row,
    // folding none; slot 1 the same rows, folding one.
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
    .expect("buffering the rung's tokens and one more row runs");
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

    // At the rung: slot 0, two rows of its own, carries one row past its
    // tokens.
    shell
        .set_rung(u32::MAX, rung)
        .expect("the rung lowers");
    let before = shell.state_bytes(0).expect("slot 0 reads back");
    let why = fire(
        &mut shell,
        &fixture,
        0,
        &wide,
        &buffer(pages, 0, 0),
        RsReset::Held,
        rung + 1,
    )
    .expect_err("a fire carrying past its rung's tokens is refused");
    assert!(
        why.contains("rung_holds")
            && why.contains(&format!("carry {}", rung + 1))
            && why.contains(&format!("past the {rung} tokens")),
        "the refusal names the fire's carried rows and the rung's tokens: {why}"
    );
    assert_eq!(
        shell.state_bytes(0).expect("slot 0 reads back"),
        before,
        "the refused fire wrote nothing"
    );
    // Slot 1 carries exactly the rung's tokens: it fires at the rung.
    fire(
        &mut shell,
        &fixture,
        1,
        &next,
        &buffer(pages, 1, 0),
        RsReset::Held,
        rung + 1,
    )
    .expect("a fire carrying exactly its rung's tokens fires");
    // Slot 0 again, one row of its own: a decode lane's carried rows count
    // toward no rung, so it fires carrying past the rung's tokens.
    fire(
        &mut shell,
        &fixture,
        0,
        &next,
        &buffer(pages, 0, 0),
        RsReset::Held,
        rung + 1,
    )
    .expect("a decode lane carrying past its rung's tokens fires in place");
    eprintln!(
        "rung {rung}: rs scratch {room} B; a two-row fire carrying {} rows is refused and \
         writes nothing, one carrying {rung} fires, a decode carrying {} fires; every rung \
         {rooms:?}",
        rung + 1,
        rung + 1
    );
}
