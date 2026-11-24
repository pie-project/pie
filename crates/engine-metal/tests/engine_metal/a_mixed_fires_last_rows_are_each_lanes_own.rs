#![cfg(target_vendor = "apple")]

//! A lane's last row of an output written over every token row of a fire
//! (`InputAxis::Row`, not gathered at the readout rows) is its own token
//! row, whatever lanes share the fire, in a load that gathers readout rows
//! (its logits) too: never the lane's index among the fire's readout rows,
//! which is another lane's row wherever a prompt sits between decodes.
//!
//! The micro text model gathers its logits; its final hidden rows, read out
//! under `hidden` (the readout gather's input, every token row), are the
//! token-row output. Fired as [decode, prompt, decode], each lane's last
//! `hidden` row is the row it reads alone.

use poem_ir::{Def, Operation, Output, RuntimeInput, Trace};

use engine_metal::serve::Seated;
use engine_metal::{Lane, Shell};

use super::a_scattered_pool_serves_the_same_logits::{
    Fixture, fixture_from, fixture_of, load_with, word,
};

const CONTEXT: u32 = 256;
const HELD: u32 = 6;

/// The micro text model with its final hidden rows read out as `hidden`,
/// last of its outputs (the row a lane reads back).
fn with_hidden() -> Fixture {
    let mut trace: Trace = fixture_of("qwen3-micro-text", None).trace.clone();
    let readout = trace
        .values
        .iter()
        .position(|decl| matches!(decl.def, Def::Input(RuntimeInput::ReadoutRows)))
        .expect("the model gathers its readout rows");
    let (x, layer) = trace
        .nodes
        .iter()
        .find_map(|node| match &node.op {
            Operation::Gather(poem_ir::Gather::Rows { x, rows, .. }) if rows.0 as usize == readout => {
                Some((*x, node.layer))
            }
            _ => None,
        })
        .expect("a gather of the readout rows");
    trace.outputs.push(Output {
        name: "hidden".to_string(),
        value: x,
        layer,
        grid: None,
        planes: Vec::new(),
    });
    fixture_from(trace, "qwen3-micro-text-hidden")
}

/// The value read out as `hidden`.
fn hidden_value(trace: &Trace) -> poem_ir::ValueId {
    trace
        .outputs
        .iter()
        .find(|output| output.name == "hidden")
        .expect("the fixture reads out its hidden rows")
        .value
}

/// The lanes' new tokens: a decode, a prompt, a decode.
fn tokens() -> Vec<Vec<u32>> {
    vec![vec![17], (0..5).map(|t| 23 + t * 11).collect(), vec![41]]
}

/// Each lane of `which` (a decode prefilled its history alone first), fired
/// together; each lane's read-back row.
fn fire(fixture: &Fixture, which: &[usize]) -> Vec<Vec<f32>> {
    let mut shell: Shell = load_with(fixture, CONTEXT);
    let all = tokens();
    for &lane in which {
        shell.open(lane as u32).expect("the slot opens");
        if all[lane].len() == 1 {
            let history: Vec<u32> = (0..HELD).map(|t| (t * 7 + 3 + lane as u32 * 5) % 200).collect();
            shell
                .fire_seated(&[Seated {
                    replay_stated: None,
                    held: Some(0),
                    ..Seated::of(Lane {
                        slot: lane as u32,
                        word: word(&fixture.trace, HELD),
                        tokens: &history,
                    })
                }])
                .expect("the decode's history prefills");
        }
    }
    let lanes: Vec<Seated<'_>> = which
        .iter()
        .map(|&lane| Seated {
            replay_stated: None,
            held: Some(if all[lane].len() == 1 { HELD } else { 0 }),
            ..Seated::of(Lane {
                slot: lane as u32,
                word: word(&fixture.trace, all[lane].len() as u32),
                tokens: &all[lane],
            })
        })
        .collect();
    shell.fire_seated(&lanes).expect("the fire runs")
}

#[test]
fn a_mixed_fires_last_rows_are_each_lanes_own() {
    if !engine_metal::device::present() {
        eprintln!("skipping: no Metal device");
        return;
    }
    let fixture = with_hidden();
    let mixed = fire(&fixture, &[0, 1, 2]);
    assert_eq!(mixed.len(), 3, "each lane's row");
    let hidden = poem_exec::store::kv::width_of(&fixture.trace, hidden_value(&fixture.trace))
        .expect("the hidden rows' width");
    for (lane, got) in mixed.iter().enumerate() {
        assert_eq!(got.len() as u64, hidden, "lane {lane} reads back the token-row output");
        let alone = fire(&fixture, &[lane]);
        let want = &alone[0];
        let scale = want.iter().map(|v| v.abs()).fold(1f32, f32::max);
        let worst = got.iter().zip(want).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
        assert!(
            worst <= 1e-2 * scale,
            "lane {lane}'s last `hidden` row in [decode, prompt, decode] is not its own: worst \
             {worst} of {scale}"
        );
    }
}
