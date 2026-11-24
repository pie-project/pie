#![cfg(target_vendor = "apple")]

//! A row's causal bound is its row order, the cache row: a stated position
//! rotates the row and bounds nothing. Beam search states a cell's depth as
//! its position while it lands at a pool slot past it, and a bound taken
//! from the stated position would hide keys a row's own lane wrote before
//! it. Here a lane's four rows land at slots 0..4 and all state position 0:
//! the last row still reads the three before it, so two lanes whose middle
//! tokens differ answer differently.

use engine_metal::serve::Seated;
use engine_metal::{Lane, Shell};

use super::a_scattered_pool_serves_the_same_logits::{Fixture, fixture_of, load_with, word};

fn last_row(shell: &mut Shell, fixture: &Fixture, slot: u32, tokens: &[u32]) -> Vec<f32> {
    let positions = vec![0u32; tokens.len()];
    let seated = Seated {
        replay_stated: None,
        held: Some(0),
        positions: &positions,
        ..Seated::of(Lane {
            slot,
            word: word(&fixture.trace, tokens.len() as u32),
            tokens,
        })
    };
    let mut rows = shell.fire_seated(&[seated]).expect("the fire lands");
    rows.remove(0)
}

#[test]
fn a_stated_position_bounds_no_key() {
    if !engine_metal::device::present() {
        eprintln!("skipping: no Metal device");
        return;
    }
    let fixture = fixture_of("qwen3-micro-text", None);
    let mut shell = load_with(&fixture, 256);
    shell.open(0).expect("slot 0 opens");
    shell.open(1).expect("slot 1 opens");
    let one = last_row(&mut shell, &fixture, 0, &[11, 48, 85, 122]);
    let two = last_row(&mut shell, &fixture, 1, &[11, 200, 7, 122]);
    let apart = one
        .iter()
        .zip(&two)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    assert!(
        apart > 1e-3,
        "the last row's logits are the same ({apart}) whatever the rows before it say: \
         its attention was bounded by its stated position, not its cache row"
    );
}
