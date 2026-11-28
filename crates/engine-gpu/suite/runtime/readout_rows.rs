//! A kv-less lane that reads out every row it holds, as a float pass does
//! (an embedding's, a cacheless encoder's: `Readout::Rows(0..rows)`), beside
//! another such lane in one fire: every row of each lands, in order, against
//! the host's table. The runtime fires these whole whatever a fixed readout
//! store holds, so the engine's readout landing must carry every row.

use engine::Engine;
use engine::fire::{FrameSubmission, Lane, Readout, Step};
use poem::Request;

use super::outputs::{Loaded, Rows, VOCAB, WIDTH, loaded};

#[test]
fn a_kv_less_lane_reads_out_every_row_it_holds() {
    let Some(Loaded {
        mut engine,
        trace,
        planes,
        dir,
        ..
    }) = loaded("every-row", &Rows)
    else {
        return;
    };
    let table = &planes["embed"].1;
    // Two lanes filling the fire's 16 tokens, each token its own row.
    let lanes: Vec<Vec<u32>> = vec![
        vec![3, 11, 7, 0, 15, 2, 9, 4, 13, 6],
        vec![1, 14, 8, 5, 12, 10],
    ];
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
        shape: None,
    };
    let mut ticket = engine
        .submit(&FrameSubmission::of(step))
        .expect("the fire of every row fires");
    engine.settle_frame(&mut ticket).expect("the frame settles");
    let readouts = &ticket.steps[0].readouts;
    assert_eq!(readouts.len(), lanes.len(), "a readout a lane");
    for (lane, (tokens, readout)) in lanes.iter().zip(readouts).enumerate() {
        assert_eq!(
            readout.rows as usize,
            tokens.len(),
            "lane {lane}: every row it holds read out"
        );
        assert_eq!(readout.width, WIDTH);
        assert_eq!(readout.values.len(), tokens.len() * WIDTH as usize);
        for (row, &token) in tokens.iter().enumerate() {
            assert!(token < VOCAB);
            let want = &table[(token * WIDTH) as usize..][..WIDTH as usize];
            let have = &readout.values[row * WIDTH as usize..][..WIDTH as usize];
            for (c, (&have, &want)) in have.iter().zip(want).enumerate() {
                assert!(
                    (have - want).abs() <= 1e-2 * want.abs().max(1.0),
                    "lane {lane} row {row} (token {token}) column {c}: device {have} \
                     against host {want}"
                );
            }
        }
    }
    eprintln!(
        "{} rows read out over {} kv-less lanes, each its token's row",
        lanes.iter().map(Vec::len).sum::<usize>(),
        lanes.len()
    );
    drop(engine);
    let _ = std::fs::remove_dir_all(&dir);
}
