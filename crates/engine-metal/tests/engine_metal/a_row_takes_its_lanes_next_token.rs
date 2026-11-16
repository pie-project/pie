#![cfg(target_vendor = "apple")]

//! `inputs.next_tokens()`, the table a draft head's ids are read from: row
//! i of a lane is fed token i + 1 of the same lane, as vLLM's speculator
//! shifts its ids; a lane's last row the token past it the runtime holds
//! (`lookahead`), else its own (the forward puts the sampled token there).
//! The forward outputs the embedding of each row's next token. A lane of
//! the one token `t` is fed `t` itself, so every row read out of two lanes
//! in one fire must equal, bit for bit, the one-token lane of the token it
//! is fed, never across the lane boundary.

use engine_metal::serve::Seated;
use engine_metal::{Lane, Shell};
use poem::{Dtype, ForwardHybrid, Input, Platform, StateRows, Weight, ops};

use super::a_scattered_pool_serves_the_same_logits::{Fixture, fixture_from, load_with, word};

const VOCAB: u32 = 64;
const WIDTH: u64 = 128;

struct Shifted;

impl ForwardHybrid for Shifted {
    fn states(&self) -> StateRows {
        StateRows::new()
    }

    fn forward(&self, inputs: Input) {
        let table = Weight::sym("embed", [u64::from(VOCAB), WIDTH], Dtype::Bf16);
        let fed = ops::gather::embed(&inputs.next_tokens(), &table, VOCAB);
        let fed = ops::gather::gather_rows(&fed, &inputs.readout_rows());
        poem::output("logits", &fed, None);
    }
}

/// Fires `lanes` (tokens, lookahead, the row to read) together, one row
/// of each read out (a fire reads a lane's one row back): each lane's row.
fn fire(
    shell: &mut Shell,
    fixture: &Fixture,
    lanes: &[(&[u32], Option<u32>, u32)],
) -> Vec<Vec<f32>> {
    let reads: Vec<[u32; 1]> = lanes.iter().map(|&(_, _, row)| [row]).collect();
    let seated: Vec<Seated<'_>> = lanes
        .iter()
        .zip(&reads)
        .enumerate()
        .map(|(slot, (&(tokens, lookahead, _), read))| Seated {
            kv_less: true,
            readout: Some(read.as_slice()),
            lookahead,
            ..Seated::of(Lane {
                slot: slot as u32,
                word: word(&fixture.trace, tokens.len() as u32),
                tokens,
            })
        })
        .collect();
    shell.fire_seated(&seated).expect("the fire runs")
}

/// Row i's token i + 1 within its lane; its last row's the lookahead, else
/// its own.
fn shifted(tokens: &[u32], lookahead: Option<u32>) -> Vec<u32> {
    tokens
        .iter()
        .skip(1)
        .copied()
        .chain(Some(lookahead.unwrap_or(tokens[tokens.len() - 1])))
        .collect()
}

#[test]
fn a_row_takes_its_lanes_next_token() {
    if !engine_metal::device::present() {
        eprintln!("skipping: no Metal device");
        return;
    }
    let trace = poem::trace_hybrid("next-token-micro", &Shifted, Platform::Metal);
    assert!(trace.reads_next_tokens());
    let fixture = fixture_from(trace, "next-token-micro");
    let mut shell = load_with(&fixture, 256);
    for slot in 0..2 {
        shell.open(slot).expect("the slot opens");
    }
    // Each token's own row: a one-token lane is fed its own token.
    let own: Vec<Vec<f32>> = (0..VOCAB)
        .map(|token| fire(&mut shell, &fixture, &[(&[token], None, 0)]).remove(0))
        .collect();
    let first: &[u32] = &[3, 41, 7, 0, 63, 2, 19];
    let second: &[u32] = &[1, 54, 8, 5];
    for lookahead in [None, Some(29)] {
        let want = [shifted(first, lookahead), shifted(second, None)];
        for row in 0..first.len().max(second.len()) {
            let at = [row.min(first.len() - 1), row.min(second.len() - 1)];
            let got = fire(
                &mut shell,
                &fixture,
                &[
                    (first, lookahead, at[0] as u32),
                    (second, None, at[1] as u32),
                ],
            );
            for lane in 0..2 {
                let next = want[lane][at[lane]];
                assert!(
                    got[lane].len() == WIDTH as usize
                        && got[lane]
                            .iter()
                            .zip(&own[next as usize])
                            .all(|(a, b)| a.to_bits() == b.to_bits()),
                    "lane {lane} row {} (lookahead {lookahead:?}): not fed token {next}, its \
                     lane's next",
                    at[lane]
                );
            }
        }
    }
    eprintln!("every row is fed its lane's next token, a lane's last its lookahead or its own");
}
