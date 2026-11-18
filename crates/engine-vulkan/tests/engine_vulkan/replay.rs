//! A decode whose keys grow past its pages replays the recording its first
//! fire kept: the page count is no part of what a recording binds.

use crate::backend::{Lane, Seated};
use crate::model_logits::{CONTEXT, PAGE, fixture, load, word};

#[test]
fn a_decode_whose_keys_grow_past_its_pages_replays_its_recording() {
    let fixture = fixture("qwen3-micro-text", None);
    let Some(mut shell) = load(&fixture, CONTEXT) else {
        return;
    };
    let tokens: Vec<u32> = (0..CONTEXT).map(|at| (at * 37 + 11) % 256).collect();
    let lane = |held: u32, rows: usize| Seated {
        held: Some(held),
        ..Seated::of(Lane {
            slot: 0,
            word: word(&fixture.trace, rows as u32),
            tokens: &tokens[..rows],
        })
    };
    let prompt = 20u32;
    shell.fire_seated(&[lane(0, prompt as usize)]).expect("the prompt fires");
    let mut held = prompt;
    // The first decode of each arm records.
    for _ in 0..4 {
        shell.fire_seated(&[lane(held, 1)]).expect("a decode fires");
        held += 1;
    }
    shell.recorded_serving();
    // Six pages of keys more, every decode one key further on.
    for _ in 0..6 * PAGE {
        shell.fire_seated(&[lane(held, 1)]).expect("a decode fires");
        held += 1;
    }
    assert_eq!(
        (shell.recorded_serving(), shell.unkept_serving()),
        (0, 0),
        "a decode {} pages of keys further on recorded or walked unkept",
        6
    );
}
