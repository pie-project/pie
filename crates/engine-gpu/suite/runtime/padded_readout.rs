//! A draft head scatters the trunk's samples over its lane's next tokens at
//! the readout rows (`gather.scatter_rows`, Qwen3.5's MTP head). Only a live
//! readout row is a sample of the fire's: a padded one placed over row 0
//! would hand row 0 a stale sample (engine-cuda's bodies padded readout rows
//! with row 0 to their ceiling, and Qwen3.5-9B's head then drafted nothing).
//! engine-gpu lays a fire's readout rows as its lanes name them and a
//! recording replays only for the same ones; this guards that a scatter over
//! them still places only the live ones, fired twice so a recording replays.
//!
//! Each row's sample here is the table row its own index names, so a pad
//! would land table row 0 at row 0; the readout row reads row 0 back through
//! the lane's tokens (token 5 is 0), whose next token is the lane's second, 1.

use engine::Engine;
use engine::fire::{FrameSubmission, Lane, Readout, Step};
use poem::{Dtype, ForwardHybrid, Input, Request, StateRows, Weight, ops};

use super::outputs::{Loaded, VOCAB, WIDTH, loaded};

struct Placed;

impl ForwardHybrid for Placed {
    fn states(&self) -> StateRows {
        StateRows::new()
    }
    fn forward(&self, inputs: Input) {
        let table = Weight::sym("embed", [u64::from(VOCAB), u64::from(WIDTH)], Dtype::Bf16);
        let nexts = ops::gather::embed(&inputs.next_tokens(), &table, VOCAB);
        let sampled = ops::gather::embed(&inputs.readout_rows(), &table, VOCAB);
        let placed = ops::gather::scatter_rows(&sampled, &inputs.readout_rows(), &nexts);
        let by_token = ops::gather::gather_rows(&placed, &inputs.tokens());
        let read = ops::gather::gather_rows(&by_token, &inputs.readout_rows());
        poem::output("logits", &read, None);
    }
}

#[test]
fn a_padded_readout_scatters_nothing() {
    let Some(Loaded {
        mut engine,
        trace,
        planes,
        dir,
        ..
    }) = loaded("padded-readout", &Placed)
    else {
        return;
    };
    let table = &planes["embed"].1;
    let w = WIDTH as usize;
    let tokens: Vec<u32> = vec![3, 1, 4, 2, 6, 0, 5];
    let want = &table[w..][..w];
    // Twice: a recording replays on the second.
    for fire in 0..2 {
        let step = Step {
            lanes: vec![Lane {
                slot: 0,
                word: trace.facts.word(&Request::new(tokens.len() as u32, false)),
                tokens: tokens.clone(),
                readout: Readout::Rows(vec![5]),
                kv_less: true,
                ..Lane::default()
            }],
            attachments: Vec::new(),
            media: Vec::new(),
            voxels: Vec::new(),
        };
        let mut ticket = engine
            .submit(&FrameSubmission::of(step))
            .expect("the fire runs");
        engine.settle_frame(&mut ticket).expect("the frame settles");
        let readout = &ticket.steps[0].readouts[0];
        assert_eq!(readout.rows, 1, "fire {fire}: the one row read out");
        for (c, (&have, &want)) in readout.values[..w].iter().zip(want).enumerate() {
            assert!(
                (have - want).abs() <= 1e-2 * want.abs().max(1.0),
                "fire {fire} column {c}: row 0 holds {have}, its next token's row (token 1) is \
                 {want}; a padded readout's sample would be table row 0"
            );
        }
    }
    drop(engine);
    let _ = std::fs::remove_dir_all(&dir);
}
