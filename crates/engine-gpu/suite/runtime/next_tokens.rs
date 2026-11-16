//! `inputs.next_tokens()`, the table a draft head's ids are read from: row
//! i of a lane is fed token i + 1 of the same lane, as vLLM's speculator
//! shifts its ids, and a lane's last row its own (the forward puts the
//! sampled token there). Two kv-less lanes in one fire, every row read out:
//! each row's embedding is the next token's row of the table, never across
//! the lane boundary.

use engine::Engine;
use engine::fire::{FrameSubmission, Lane, Readout, Step};
use poem::{Dtype, ForwardHybrid, Input, Request, StateRows, Weight, ops};

use super::outputs::{Loaded, VOCAB, WIDTH, loaded};

struct Shifted;

impl ForwardHybrid for Shifted {
    fn states(&self) -> StateRows {
        StateRows::new()
    }
    fn forward(&self, inputs: Input) {
        let table = Weight::sym("embed", [u64::from(VOCAB), u64::from(WIDTH)], Dtype::Bf16);
        let fed = ops::gather::embed(&inputs.next_tokens(), &table, VOCAB);
        poem::output("logits", &fed, None);
    }
}

#[test]
fn a_row_takes_its_lanes_next_token() {
    let Some(Loaded {
        mut engine,
        trace,
        planes,
        dir,
        ..
    }) = loaded("next-tokens", &Shifted)
    else {
        return;
    };
    assert!(trace.reads_next_tokens());
    let table = &planes["embed"].1;
    let lanes: Vec<Vec<u32>> = vec![vec![3, 11, 7, 0, 15, 2, 9], vec![1, 14, 8, 5, 12]];
    let step = Step {
        lanes: lanes
            .iter()
            .enumerate()
            .map(|(slot, tokens)| {
                let rows = tokens.len() as u32;
                Lane {
                    slot: slot as u32,
                    word: trace.facts.word(&Request::new(rows, false)),
                    tokens: tokens.clone(),
                    readout: Readout::Rows((0..rows).collect()),
                    kv_less: true,
                    ..Lane::default()
                }
            })
            .collect(),
        attachments: Vec::new(),
        media: Vec::new(),
        voxels: Vec::new(),
    };
    let mut ticket = engine
        .submit(&FrameSubmission::of(step))
        .expect("the fire of every row fires");
    engine.settle_frame(&mut ticket).expect("the frame settles");
    let readouts = &ticket.steps[0].readouts;
    assert_eq!(readouts.len(), lanes.len(), "a readout a lane");
    let w = WIDTH as usize;
    for (lane, (tokens, readout)) in lanes.iter().zip(readouts).enumerate() {
        assert_eq!(readout.rows as usize, tokens.len(), "lane {lane}: every row");
        let shifted = tokens.iter().skip(1).chain(tokens.last());
        for (row, &next) in shifted.enumerate() {
            let want = &table[next as usize * w..][..w];
            let have = &readout.values[row * w..][..w];
            for (c, (&have, &want)) in have.iter().zip(want).enumerate() {
                assert!(
                    (have - want).abs() <= 1e-2 * want.abs().max(1.0),
                    "lane {lane} row {row} column {c}: fed {have}, token {next}'s row is {want}"
                );
            }
        }
    }
    drop(engine);
    let _ = std::fs::remove_dir_all(&dir);
}
