//! A served load fired before the engine places its programs is refused by
//! name, not submitted: fired so, it hung the device on what placing them
//! sets up. Once they are placed, it fires.

use crate::backend::{Lane, Seated};
use crate::model_logits::{CONTEXT, fixture, load_served, word};

/// A served load fired before the engine places its programs is refused by
/// name, not submitted (fired so, it hung the device); once they are
/// placed, it fires.
#[test]
fn a_served_load_fires_nothing_before_its_programs_are_placed() {
    let fixture = fixture("qwen3-micro-text", None);
    let Some(mut shell) = load_served(&fixture, CONTEXT) else {
        return;
    };
    let tokens = [11u32];
    let lane = || {
        Seated::of(Lane {
            slot: 0,
            word: word(&fixture.trace, 1),
            tokens: &tokens,
        })
    };
    shell.open(0).expect("the seat opens");
    let refused = shell
        .fire_seated(&[lane()])
        .expect_err("a fire before the programs are placed is refused");
    assert!(refused.to_string().contains("programs"), "{refused}");
    shell.programs_placed();
    shell.fire_seated(&[lane()]).expect("the placed load fires");
}
